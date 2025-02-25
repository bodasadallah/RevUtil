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
if __name__ == "__main__":

    args = get_args()
    save_dir = os.path.join(args.output_path, args.dataset_config)
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
    llm = LLM(model=args.base_model_name, enable_lora=True)
    
    chat_template = '''{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}'''

    sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_new_tokens,)

    if args.finetune_model_name is not None:
       
        print('Loading a lora finetuned model')
        outputs = llm.chat(
        messages=processed_data,
        chat_template=chat_template,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("my_adapter", 1, args.finetune_model_name))


    else:
        print('Evaluating the base model')
        outputs = llm.generate(
        processed_data,
        chat_template=chat_template,
        sampling_params=sampling_params,)

    ########### Extract model predeicitons #############33
    predictions = extract_predictions(outputs)

    gold_labels = raw_data[f'{args.dataset_config}_label']

    assert len(predictions) == len(gold_labels), 'The number of predictions and gold labels should be the same'

    agreements_dict = get_stats(predictions, gold_labels, args.dataset_config)

    ### save the model outputs in file named raw_outputs.txt
    with open(os.path.join(save_dir, 'raw_outputs.txt'), 'w') as f:
        for output in outputs:
            generated_text = output.outputs[0].text
            prompt = output.prompt
            f.write(prompt + '\n')
            f.write(generated_text + '\n')

   