#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""


import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
import wandb

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from alignment.data import DEFAULT_CHAT_TEMPLATE
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from dataclasses import asdict
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
import utils
### get dataaset token from .env
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

logger = logging.getLogger(__name__)



############################ Code to name checkpoints based on the number of examples seen ############################
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import os
import shutil
import torch

class ExampleCheckpointRenamerCallback(TrainerCallback):
    def __init__(self, effective_batch_size):
        self.effective_batch_size = effective_batch_size

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Only run on main process (rank 0)
        if not args.process_index == 0:
            return control

        # Compute total examples seen across all GPUs
        total_examples_seen = state.global_step * self.effective_batch_size

        if control.should_save:
            original_ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            renamed_ckpt = os.path.join(args.output_dir, f"checkpoint-{total_examples_seen}-examples")
            if os.path.exists(original_ckpt):
                shutil.move(original_ckpt, renamed_ckpt)
                print(f"Renamed checkpoint: {original_ckpt} → {renamed_ckpt}")

        return control

########################################################################################

def main():

    from huggingface_hub import login
    login(HF_TOKEN)

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    aspect = model_args.aspect

    ####### maybe change later? #####
    dataset_configs =[aspect]

    generation_type = data_args.generation_type
    prompt_type = data_args.prompt_type
    model = model_args.model_name_or_path.split("/")[-1]
    training_args.output_dir = f"{training_args.output_dir}/{model}/{generation_type}/{prompt_type}/{aspect}"
    print("################ OUTPUT DIR ################")
    print(training_args.output_dir)

    WANDB_RUN_NAME = f"{model}_{generation_type}_{prompt_type}_{aspect}"


    if training_args.local_rank == 0:
        wandb.init(project=model_args.wandb_project, name=WANDB_RUN_NAME)
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=dataset_configs,
        # columns_to_keep=["prompt"],
        columns_to_keep=[],

    )


    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    # Set chat template if not already set
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    train_dataset = raw_datasets["train"].map(
        utils.get_prompt,
        fn_kwargs={
            "aspect": aspect,
            "task": "train",
            "generation_type": data_args.generation_type,
            "prompt_type": prompt_type,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc=f"Applying {prompt_type} prompt template",
        # load_from_cache_file = False,
    )
    eval_dataset = raw_datasets["test"].map(
        utils.get_prompt,
        fn_kwargs={
            "aspect": aspect,
            "task": 'evaluation',
            "generation_type": data_args.generation_type,
            "prompt_type": prompt_type,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc=f"Applying {prompt_type} prompt template",
        # load_from_cache_file = False,
    )

    #####################
    # Apply chat template Or Build the Instruction dataset
    ##################### 
    if data_args.prompt_type == 'chat':
        train_dataset = train_dataset.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            desc="Applying chat template",
            load_from_cache_file = False,

        )
        eval_dataset = eval_dataset.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            desc="Applying chat template",
            load_from_cache_file = False,

        )


    #### get the max number of tokens in the dataset. get the tallest sample and then encode it
    longest_sample = max(train_dataset, key=lambda x: len(x['text']))
    max_tokens = len(tokenizer(longest_sample['text'])['input_ids'])
    print("################ MAX TOKENS ################")
    print(max_tokens)




    with open("prompt.txt", "w") as f:
        with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
            for index in random.sample(range(len(train_dataset)), 1):
                logger.info(f"Sample {index} of the processed training set:\n\n{train_dataset[index]['text']}")
                f.write(f"{train_dataset[index]['text']}\n\n")
            for index in random.sample(range(len(eval_dataset)), 1):
                logger.info(f"Sample {index} of the processed evaluation set:\n\n{eval_dataset[index]['text']}")
                f.write(f"{eval_dataset[index]['text']}\n\n")


    ########### Train on Completion only ################
    if model_args.train_on_completion_only:
        if data_args.prompt_type == 'chat':
            if "R1" in model_args.model_name_or_path:
                response_template = "<｜Assistant｜>"
            else:
                response_template = "<|assistant|>\n"

        elif data_args.prompt_type == 'instruction':
            response_template = "\n\n###Output:\n"
        
        ## convert the response template to token ids
        response_template = tokenizer(response_template)['input_ids'][2:]

        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)



    ### In the previous version on SFTTRAINER this used to be passed directly to the trainer, now it's part of the config
    training_args.model_init_kwargs = model_kwargs
    training_args.dataset_text_field = "text"
    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        data_collator=collator if model_args.train_on_completion_only else None,
        # callbacks=callbacks
    )



    train_dataloader = trainer.get_train_dataloader()
    first_batch = next(iter(train_dataloader))
    print(first_batch['input_ids'][0])

    input_ids_batch = first_batch["input_ids"] 
    decoded_texts = [tokenizer.decode(input_ids, skip_special_tokens=False) for input_ids in input_ids_batch]
    print('############################# DECODED TEXTS #############################')
    print(decoded_texts[0])

    ############# To use FSDP #############
    if trainer.is_fsdp_enabled: 
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT") 

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")

    # checkpoint = None
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        print("----------------- last_checkpoint",last_checkpoint)
    else:
        checkpoint = None


    ## No resume from checkpoint if training_args.resume_from_checkpoint is None
    checkpoint = None
    print("---------------------- checkpoint",checkpoint)


    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "model_name" : model_args.model_name_or_path,
        "dataset_name" : aspect,
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        ## set padding side to right for for flashattention error
        trainer.tokenizer.padding_side='right'
        trainer.model.config.use_cache=False
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


    # Finish wandb run
    if trainer.accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
