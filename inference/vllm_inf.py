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

    model_name = args.base_model_name.split('/')[-1]
    save_dir = os.path.join(args.output_path,model_name, args.dataset_config)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ### Load the data
    raw_data = datasets.load_dataset(args.dataset_name, args.dataset_config)
    if 'test' in raw_data.keys():
        raw_data = raw_data['test']
    else:
        raw_data = raw_data['train']
    
    #### Process the dataset to get the prompts
    processed_data = []
    for row in tqdm(raw_data):
        prompt = get_prompt(row, aspect=args.dataset_config,task='evaluation',  evaluation_type='score_only')

        processed_data.append(prompt)


    ### Load the model
    enable_lora = True if args.finetune_model_name is not None else False

    llm = LLM(model=args.base_model_name,
            enable_lora=enable_lora,
            tensor_parallel_size = args.tensor_parallel_size,
            gpu_memory_utilization=0.9)

    sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_new_tokens,)

    if args.finetune_model_name is not None:
       
        print('*' * 20,'Loading a lora finetuned model', '*' * 20)
        outputs = llm.chat(
        messages=processed_data,
        chat_template=DEFAULT_CHAT_TEMPLATE,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("my_adapter", 1, args.finetune_model_name))


    else:
        print('*' * 20, 'Evaluating the base model', '*' * 20)
        outputs = llm.chat(
        messages=processed_data,
        chat_template=DEFAULT_CHAT_TEMPLATE,
        sampling_params=sampling_params,
        use_tqdm=True,)


    ### save the model outputs in file named raw_outputs.txt
    with open(os.path.join(save_dir, 'raw_outputs.jsonl'), 'w') as f:
        for output in outputs:
            generated_text = output.outputs[0].text
            # prompt = output.prompt
            # f.write(prompt + '\n')
            raw_pred = {'generated_text': generated_text}
            f.write(json.dumps(raw_pred) + '\n')

    ########## Extract model predeicitons #############33
    predictions = extract_predictions(outputs)
    ### Save the model predictions to a jsonl file
    with open(os.path.join(save_dir, 'predictions.jsonl'), 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")
            

   