#!/usr/bin/env python
# coding: utf-8

import torch
import random
from transformers import HfArgumentParser, Seq2SeqTrainingArguments,EarlyStoppingCallback

import logging

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from datasets import load_dataset, concatenate_datasets,Value
import numpy as np
from typing import Union, Optional
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset, AutoModel
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    #glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,GenerationConfig
)
from arguments import ModelArguments, DataArguments
import wandb
from nltk.tokenize import sent_tokenize
import nltk
from evaluate import load

nltk.download("punkt")
logger = logging.getLogger(__name__)

import pathlib
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from evaluate import load


from utils import *
import numpy as np
from peft import PeftModel    
import logging
import os
from torch.utils.data import DataLoader 
from tqdm import tqdm
from typing import List


from llama import Dialog, Llama
import fire



def inference(prompts, tokenizer, generation_config, model):
    
   
    encoding = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0001,
            repetition_penalty = 5.0,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            num_beams=10,
            # generation_config=generation_config,
        )
    answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
    output_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)

    return output_text
        

if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()





    


    val_dataset = get_dataset(
    dataset_path = data_args.dataset,
    split='test',
    field=data_args.prompt_key,
    num_shots=data_args.num_shots,
    explicit_errors = data_args.explicit_errors)


    print(f'Loaded {len(val_dataset)} samples')
    print(f'Randon sample:')
    sample = random.choice(val_dataset)
    for k,v in sample.items():
        print(f'{k}: {v}')

    val_dataloader = DataLoader(val_dataset, batch_size=training_args.per_device_train_batch_size,collate_fn=lambda x: x )
    bertscore = load("bertscore")


    if model_args.model_name_or_path != 'Meta-Llama-3-8B' and model_args.model_name_or_path != 'Meta-Llama-3-8B-Instruct':

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        # use_flash_attention_2=model_args.use_flash_attention_2,
        use_flash_attention_2=0,

        )
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     return_dict=True,
        #     load_in_8bit=True,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        # )

        if model_args.checkpoint_path:
            print(f'Loading model from {model_args.checkpoint_path}')
            adapter_checkpoint  = model_args.checkpoint_path
            model = PeftModel.from_pretrained(model, adapter_checkpoint)

        else:
            print(f'Loading Base Model {model_args.model_name_or_path}')

        
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

        model = model.eval()
        generation_config = GenerationConfig.from_pretrained("mistralai/Mistral-7B-v0.1")

        # Define PAD Token = BOS Token
        tokenizer.pad_token = tokenizer.bos_token
        model.config.pad_token_id = model.config.bos_token_id


        original_predictions = []
        masked_words = []

        torch.cuda.empty_cache()

        predicitons = []
        outputs = []
        cleaned_predictions = []

        for batch in tqdm(val_dataloader):

            prompts = [x['prompt'] for x in batch]
            ans = []

            output_text = inference(prompts=prompts, tokenizer=tokenizer, generation_config=generation_config, model=model)
            labels = []
            for i,ex in enumerate(batch):
                # labels.append(f"{i+1}. {ex['explanation']}\n")
                labels.append(ex['label'])

            predicitons.extend(output_text)
            outputs.extend(labels)


    #############################3 LLAMA3 Evaluation #######################
    else:

        print(' ================ Loading LLAMA3 ===============')

        if model_args.model_name_or_path == 'Meta-Llama-3-8B-Instruct':
            ckpt = '/l/users/abdelrahman.sadallah/llama3/Meta-Llama-3-8B-Instruct'
            tokenizer_path = '/l/users/abdelrahman.sadallah/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model'
        else:
            ckpt = '/l/users/abdelrahman.sadallah/llama3/Meta-Llama-3-8B'
            tokenizer_path = '/l/users/abdelrahman.sadallah/llama3/Meta-Llama-3-8B/tokenizer.model'

        generator = Llama.build(
        ckpt_dir=ckpt,
        tokenizer_path=tokenizer_path,
        max_seq_len=2048,
        max_batch_size=training_args.per_device_train_batch_size,
        )
        predicitons = []
        outputs = []
        cleaned_predictions = []
        for batch in tqdm(val_dataloader):
            labels = []
            prompts = [x['prompt'] for x in batch]
            for i,ex in enumerate(batch):
                # labels.append(f"{i+1}. {ex['explanation']}\n")
                labels.append(ex['label'])

            if model_args.model_name_or_path == 'Meta-Llama-3-8B-Instruct':
                dialogs: List[Dialog] = [ ]
                for p in prompts:
                    d = [{"role": "user", "content":p}]
                    dialogs.append(d)

                prompts = dialogs

                model_outputs = generator.chat_completion(
                        prompts,
                        max_gen_len=256,
                        temperature=0.0001,
                        # top_p=0.9,
                        )
            else:
                model_outputs = generator.text_completion(
                        prompts,
                        max_gen_len=256,
                        temperature=0.0001,
                        # top_p=0.9,
                        )

            predicitons.extend([z['generation'] for z in model_outputs] if model_args.model_name_or_path == 'Meta-Llama-3-8B' else [z['generation']['content'] for z in model_outputs])
            outputs.extend(labels)

    for z in predicitons:

        inside = 0
        clean_str = []
        for l in z.split('\n'):
            l = l.strip()
            if re.match(r'^\d+\.', l):
                inside = 1
                clean_str.append(l)
            elif inside:
                break
        cleaned_predictions.append('\n'.join(clean_str) )
    
    assert len(predicitons) == len(outputs) == len(cleaned_predictions)

    if '/' in model_args.model_name_or_path:
        modelname = model_args.model_name_or_path.split('/')[-1]
    else:
        modelname = model_args.model_name_or_path
    with open(f'outputs/{modelname}_outputs_{data_args.num_shots}-shots_{data_args.explicit_errors}-explicit.txt', 'w') as f:
        for i in range(len(predicitons)):
            f.write(f"Error_sentence: {val_dataset[i]['incorrect_sentence']}\n")
            f.write(f"Correct_sentence: {val_dataset[i]['correct_sentence']}\n")
            f.write(f"Prediction: {predicitons[i]}\n")
            f.write(f"cleaned: {cleaned_predictions[i]}\n")
            f.write(f"Label: {outputs[i]}\n")
            f.write('\n\n\n')

    ##################################33 EVALUATE ############################
    # print('Random selected samples:')
    # for  i,x in enumerate (random.choices(list(zip(predicitons,outputs)),k=5)):
    #     print(f'Prediction: {x[0]}')
    #     print(f'Label: {x[1]}')
    #     print('\n\n\n')

    metrics = bertscore.compute(predictions=predicitons, references=outputs, lang="en")
    cleaned_metrics = bertscore.compute(predictions=cleaned_predictions, references=outputs, lang="en")

    f1 = np.array(metrics['f1']).mean()
    preceision = np.array(metrics['precision']).mean()
    recall = np.array(metrics['recall']).mean()

    cleaned_f1 = np.array(cleaned_metrics['f1']).mean()
    cleaned_preceision = np.array(cleaned_metrics['precision']).mean()
    cleaned_recall = np.array(cleaned_metrics['recall']).mean()



    print(f'Model: {model_args.model_name_or_path}')
    print('F1:', f1)
    print('Precision:', preceision)
    print('Recall:', recall)
    print('Cleaned F1:', cleaned_f1)
    print('Cleaned Precision:', cleaned_preceision)
    print('Cleaned Recall:', cleaned_recall)

    save_file = f'outputs/{modelname}_metrics_{data_args.num_shots}-shots_{data_args.explicit_errors}-explicit.txt'
    with open(save_file, 'w') as f:
        f.write(f"BERTScore: \n F1: {f1} \n Precision: {preceision} \n Recall: {recall} \n")

    
