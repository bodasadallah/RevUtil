from vllm import LLM, SamplingParams
import os
import sys
from args_parser import get_args
import datasets
import os
import sys
from tqdm import tqdm
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
from utils import get_prompt, get_stats
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from inference_utils import extract_predictions
import json

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

if __name__ == "__main__":

    args = get_args()

    LORA_PATH = None
    enable_lora = False

    model_name = args.full_model_name.split('/')[-1]
    checkpoint_parent_path = args.checkpoint_parent_path

    BASE_MODEL = args.full_model_name
    checkpoint_path = os.path.join(
            checkpoint_parent_path,
            args.finetuning_type,
            model_name,args.generation_type,
            args.prompt_type, 
            args.dataset_config, 
            'checkpoint-'+args.step)

    if args.finetuning_type =='full':
        BASE_MODEL = checkpoint_path
        print('*' * 20,'Loading the full model', '*' * 20)
    elif args.finetuning_type == 'adapters':
        enable_lora = True
        LORA_PATH = checkpoint_path
        print('*' * 20,'Loading a model with adapters', '*' * 20)
    elif args.finetuning_type == 'base':
        print('*' * 20,'Loading the base model', '*' * 20)


    save_dir = os.path.join(args.output_path,model_name,args.generation_type,args.prompt_type, args.dataset_config)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)





    llm = LLM(BASE_MODEL,
            enable_lora=enable_lora,
            tensor_parallel_size = args.tensor_parallel_size,
            gpu_memory_utilization=0.9)

    sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_new_tokens,)



    configs = args.dataset_config.split(',')
    splits = args.dataset_split.split(',')
    label_dict = {}
    for config in configs:
        for split in splits:
            
            print('Evaluating the model on', config, 'aspect and', split, 'split')
            print('*' * 20, 'loading the dataset', '*' * 20)



            ### Load the data
            raw_data = datasets.load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)

            #### Process the dataset to get the prompts
            processed_data = []
            for row in tqdm(raw_data):
                prompt = get_prompt(row, aspect=args.dataset_config,task='evaluation',generation_type=args.generation_type, prompt_type=args.prompt_type)
                processed_data.append(prompt['text'])


            print('Total number of prompts:', len(processed_data))
            print('Example prompt:', processed_data[0])

            if args.prompt_type == 'chat':
                outputs = llm.chat(
                messages=processed_data,
                chat_template=DEFAULT_CHAT_TEMPLATE if model_name == 'scitulu-7b' else None,
                sampling_params=sampling_params,
                use_tqdm=True,
                lora_request= LoRARequest("my_adapter", 1, LORA_PATH) if enable_lora else None,)

            else:
                outputs = llm.generate(
                    prompts=processed_data,
                    sampling_params=sampling_params,
                    use_tqdm=True,
                    lora_request= LoRARequest("my_adapter", 1, LORA_PATH) if enable_lora else None,)



            
            raw_outputs_name = os.path.join(save_dir, f'raw_outputs_{config}_{split}.jsonl') 
            with open(os.path.join(save_dir, 'raw_outputs.jsonl'), 'w') as f:
                for output in outputs:
                    generated_text = output.outputs[0].text
                    # prompt = output.prompt
                    # f.write(prompt + '\n')
                    raw_pred = {'generated_text': generated_text}
                    f.write(json.dumps(raw_pred) + '\n')

            try:

                ########## Extract model predeicitons #############33
                predictions = extract_predictions(outputs)
                ### Save the model predictions to a jsonl file

                label_dict[f'{config}_{split}'] = []    
                aspects = [ 'actionability', 'grounding_specificity','verifiability', 'helpfulness'] if split == 'all' else [split]
                for aspect in aspects:
                    gold_data_name = args.gold_label_format.replace('ASPECT', aspect)
                    gold_data = raw_data[gold_data_name]
                    preds = [p[f'{aspect}_label'] for p in predictions]
                    assert len(preds) == len(gold_data), 'The number of predictions and gold labels do not match'

                    label_dict[f'{config}_{split}'].append({'gold': gold_data, 'preds': preds, aspect: aspect})

                predictions_name = os.path.join(save_dir, f'predictions_{config}_{split}.jsonl')
                with open(os.path.join(save_dir, 'predictions.jsonl'), 'w') as f:
                    for prediction in predictions:
                        f.write(json.dumps(prediction) + "\n")

                ### Get the stats
            except:
                print('Could not extract predictions, you need to do the evaluation manually')


    ######################### Save the results to a file #######################
    results_file_name = os.path.join(save_dir, f'results_{config}_{split}.txt')
    with open(results_file_name, 'w') as f:
        ########## Get stats for all items in the dict
        for key in label_dict:
            for d in label_dict[key]:

                gold = d['gold']
                preds = d['preds']
                aspect = d['aspect']
                stats = get_stats(gold, preds, aspect)
                stat_dict = get_stats(preds, gold, aspect)
                f.write('*' * 20 + f'{aspect}'+ '*' * 20 + '\n')
                ## remove the acc from the dict, and for spearman, only include the correlation
                for k,v in stat_dict.items():
                    if 'accuracy' in k:
                        continue
                    if 'spearman' in k:
                        v = v[0]
                    ## round the values
                    if isinstance(v, float):
                        v = round(v, 3)
                    f.write(f'{k}: {v}\n')
                f.write('-'*50+'\n')
