from vllm import LLM, SamplingParams
import os
import sys
from args_parser import get_args
import datasets
import os
import sys
from tqdm import tqdm
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from utils import get_prompt, get_stats, get_alpha_scores, evaluate_rationale
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from inference_utils import extract_predictions
import json
import ast

### get dataaset token from .env
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
accepted_annotators =  {
    'boda' : "boda",
    '6740484e188a64793529ee77' : "Annotator1",
    '6686ebe474531e4a1975636f': "Annotator2"
 }



def chatgpt_inference(args, raw_data,raw_outputs_name,save_dir, temperature=0.0):

    mode = args.generation_type
    aspect=args.training_aspects

    chatgpt_batch_input_path = os.path.join(save_dir, f'chatgpt_batch_input_{aspect}_{mode}.jsonl')
    chatgpt_batch_meta_data_path = os.path.join(save_dir, f'chatgpt_batch_meta_data_{aspect}_{mode}.json')
    chatgpt_batch_output_path = os.path.join(save_dir, f'chatgpt_batch_output_{aspect}_{mode}.jsonl')

    ### if the raw outputs file already exists, skip the evaluation
    if os.path.exists(raw_outputs_name):
        print('The raw outputs file already exists, skipping the predictions generation')
        return


    from openai import OpenAI, AsyncOpenAI
    import dotenv
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    sys.path.append(parent_dir)
    dotenv.load_dotenv()

    client = OpenAI(api_key=os.environ.get("KEY"))

    model_name = 'gpt-4o'

    print(chatgpt_batch_meta_data_path)
    try:
        batch_data = json.load(open(chatgpt_batch_meta_data_path, 'r'))
    except:
        print('Batch file does not exist, creating a new one')
        batch_data = None


    if batch_data and client.batches.retrieve(batch_data['batch_id']).status == 'completed':
        print(f"Batch file for {batch_data['aspect']} and {mode} has been completed")

        ### Get chatgpt outupt and process them
        aspect = batch_data['aspect']
        batch_id = batch_data['batch_id']
        output_file_id = client.batches.retrieve(batch_id).output_file_id
        chatgpt_response =  client.files.content(output_file_id)
        file_path = chatgpt_batch_output_path
        with open(file_path, 'w') as file:
            file.write(chatgpt_response.text + '\n')

        chatgpt_response = pd.read_json(file_path, lines=True)

        with open(chatgpt_batch_input_path, 'r') as f:
            inputs = [json.loads(line) for line in f]

        outputs = []
        
        # Convert the 'custom_id' column to string type
        chatgpt_response['custom_id'] = chatgpt_response['custom_id'].astype(str)
        for i  in range(len(inputs)):
            id = inputs[i]['custom_id']
            chatgpt_row = chatgpt_response[chatgpt_response['custom_id']==str(id)]

            ## if the row is not found, then it was failed, then skip it
            if chatgpt_row.shape[0] == 0:
                print('The row is not found, skipping it')
                answer = None
                continue
            else:
                chatgpt_row = chatgpt_row.iloc[0].copy()
                answer = chatgpt_row['response']['body']['choices'][0]['message']['content']

            outputs.append(answer)
        
        with open(raw_outputs_name, 'w') as f:
            for output in outputs:
                generated_text = output
                raw_pred = {'generated_text': generated_text}
                f.write(json.dumps(raw_pred) + '\n')


    elif not os.path.exists(chatgpt_batch_input_path):

        print('The raw outputs file does not exist, generating the predictions')
        lines = []

        processed_data = []
        for row in tqdm(raw_data):
            prompt = get_prompt(row, 
                                aspect=args.training_aspects,
                                task='evaluation',
                                generation_type=args.generation_type, 
                                prompt_type=args.prompt_type,
                                finetuning_type=args.finetuning_type,
                                model=args.full_model_name)
            
            processed_data.append(prompt['text'])


            line = {
                "custom_id": f"{row['id']}", 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {"model": model_name,
                "temperature": temperature,
                "messages": 
                [{"role": "user", "content": prompt['text']}],}}
            
            lines.append(line)

            
        print(f'sample of the prompts is {lines[0]}')

        ### Write batch input file
        batch_file_path = chatgpt_batch_input_path
        with open(batch_file_path, 'w') as f:
            for l in lines:
                json.dump(l, f)
                f.write('\n')


        # upload the batch file
        batch_input_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch")

        ### create the batch request
        batch_input_file_id = batch_input_file.id

        batch_data = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": f"batch file for  {aspect} {mode} model gpt-4o, temperature {temperature} for Evaluation",
            })
        
        batch_metadata = {
            "batch_id": batch_data.id,
            "aspect": aspect,
            "batch_input_file_id": batch_input_file_id,
            "batch_file_path": batch_file_path,
        }

        with open(f"{chatgpt_batch_meta_data_path}", 'w') as f:
            json.dump(batch_metadata, f, indent=4)
            
        print(f"Batch file for {aspect}, {mode} is created and uploaded to the server")




def vllm_inferece(args, raw_data,raw_outputs_name, sampling_params,llm, enable_lora, LORA_PATH, model_name):

    ### if the raw outputs file already exists, skip the evaluation
    if  not os.path.exists(raw_outputs_name):
        print('The raw outputs file does not exist, generating the predictions')
        #### Process the dataset to get the prompts
        processed_data = []
        for row in tqdm(raw_data):
            prompt = get_prompt(row, 
                                aspect=args.training_aspects,
                                task='evaluation',
                                generation_type=args.generation_type, 
                                prompt_type=args.prompt_type,
                                finetuning_type=args.finetuning_type,
                                model=args.full_model_name)
            
            ###################### prometheus prompt ######################
            if 'prometheus' in  args.full_model_name:
                
                ABS_SYSTEM_PROMPT =  "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
                prompt['text']  =    [{'role': 'user', 'content':  ABS_SYSTEM_PROMPT + "\n\n"  + prompt['text']}]

            processed_data.append(prompt['text'])

        print('Total number of prompts:', len(processed_data))
        print('Example prompt:', processed_data[0])


        if args.prompt_type == 'chat' or  'prometheus' in  args.full_model_name:

            print('***************************Using chat prompt***************************')
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
        with open(raw_outputs_name, 'w') as f:
            for output in outputs:
                generated_text = output.outputs[0].text
                raw_pred = {'generated_text': generated_text}
                f.write(json.dumps(raw_pred) + '\n')

    else:
        print('The raw outputs file already exists, skipping the predictions generation')

def write_stats_to_file(label_dict, results_file_name):

    results_dict = {}
    
    ########## Get stats for all items in the dict
    for key in label_dict:
        results_dict[key] = {}
        for d in label_dict[key]:

            gold = d['gold']
            preds = d['preds']
            aspect = d['aspect']
            print('Calculating the stats for', aspect)
            results_dict[key][aspect] = {}

            ### check if the gold label is dict, then we do pair-wise comparison
            is_dict = False
            try:
                if type(ast.literal_eval(gold[0])) == dict:
                    is_dict = True
            except Exception as e:
                print('Could not evaluate the gold label, it is not a dict')
                is_dict = False

            if is_dict:
                print('The gold label is a dict, doing pair-wise comparison')


                annotations = {}
                for i in range(len(gold)):
                    gold[i] = ast.literal_eval(gold[i])
                    for j in range(len(gold[i]['annotators'])):
                        annotator = gold[i]['annotators'][j]
                        annotator = accepted_annotators[annotator]
                        if annotator not in annotations:
                            annotations[annotator] = []
                        label = str(gold[i]['labels'][j])
                        annotations[annotator].append(label)
                
                for annotator in annotations:
                    assert len(annotations[annotator]) == len(preds), 'The number of predictions and gold labels do not match'

                    stat_dict = get_stats(preds, annotations[annotator], aspect)
                    stat_dict = process_stat_dict(stat_dict)
                    results_dict[key][aspect][annotator] = stat_dict


                # Calculate the average stat_dict across annotators
                average_stat_dict = {}
                num_annotators = len(annotations)
                for annotator in annotations:
                    for metric, value in results_dict[key][aspect][annotator].items():
                        if metric not in average_stat_dict:
                            average_stat_dict[metric] = 0.0
                        average_stat_dict[metric] += value

                for metric in average_stat_dict:
                    average_stat_dict[metric] /= num_annotators

                # Calculate Krippendorff's alpha
                annotations_plus_predictions = list(annotations.values()) + [preds]
                alpha = get_alpha_scores(annotations_plus_predictions, aspect)
                average_stat_dict['krippendorff_alpha'] = alpha

                average_stat_dict = process_stat_dict(average_stat_dict)

                results_dict[key][aspect]['total_stats'] = average_stat_dict
          
            ############## if the gold label is not a dict, we only have one label ##############
            ################### This happens for evlauation against the Test set ##################
            else:
                stat_dict = process_stat_dict(get_stats(preds, gold, aspect))

                gold_rationales = d.get('gold_data_rationale', None)
                preds_rationales = d.get('preds_rationale', None)


                if gold_rationales:

                    rationale_results = evaluate_rationale(gold_rationales, preds_rationales, pred_scores = preds, 
                                                                gold_scores = gold, aspect = aspect)

                # Add all keys from rationale_results to stat_dict
                if rationale_results:
                    for k, value in rationale_results.items():
                        stat_dict[k] = value


                stat_dict = process_stat_dict(stat_dict)

                results_dict[key][aspect]['total_stats'] = stat_dict

    with open(results_file_name, 'w') as f:
        f.write(json.dumps(results_dict, indent=4))
        print('Results saved to', results_file_name)






def process_stat_dict(stat_dict):
    processed_stat_dict = {}
    for k, v in stat_dict.items():
        # if 'accuracy' in k:
        #     continue
        if 'spearman' in k or 'pearson' in k or 'tau' in k:
            v = v[0] if isinstance(v, tuple) else v
        if isinstance(v, float):
            v = round(v, 3)
        processed_stat_dict[k] = v

    return processed_stat_dict



if __name__ == "__main__":

    args = get_args()

    LORA_PATH = None
    enable_lora = False

    model_name = args.full_model_name.split('/')[-1]
    checkpoint_parent_path = args.checkpoint_parent_path

    BASE_MODEL = args.full_model_name

    if args.step == '0':
        checkpoint_path = os.path.join(
                checkpoint_parent_path,
                args.finetuning_type,
                model_name,args.generation_type,
                args.prompt_type, 
                args.training_aspects)
    else:
        checkpoint_path = os.path.join(
        checkpoint_parent_path,
        args.finetuning_type,
        model_name,args.generation_type,
        args.prompt_type, 
        args.training_aspects, 
        'checkpoint-'+args.step)
        

    if args.finetuning_type =='full':
        BASE_MODEL = checkpoint_path
        print('*' * 20,'Loading the full model', '*' * 20)
    elif args.finetuning_type == 'adapters':
        enable_lora = True
        LORA_PATH = checkpoint_path
        print('*' * 20,'Loading a model with adapters', '*' * 20)
    elif args.finetuning_type == 'baseline':
        print('*' * 20,'Loading the base model', '*' * 20)
    else:
        print('Please provide a valid finetuning type')
        sys.exit(1)

    dataset_name = args.dataset_name.split('/')[-1]


    save_dir = os.path.join(
        args.output_path,
        model_name,
        args.generation_type,
        args.prompt_type,
        args.training_aspects,
        "step-"+args.step,
        dataset_name,)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ### create metadata json file, and put all args in it
    metadata = {}
    for arg in vars(args):
        metadata[arg] = getattr(args, arg)
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    llm = None

    if 'gpt' not in BASE_MODEL.lower():
        llm = LLM(BASE_MODEL,
                enable_lora=enable_lora,
                max_lora_rank=64,
                # tensor_parallel_size = 1,
                tensor_parallel_size = args.tensor_parallel_size,
                gpu_memory_utilization=0.95,
                max_num_seqs=args.max_num_seqs,
                max_model_len= 8196
                )

    sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_new_tokens,)




    configs = args.dataset_config.split(',')
    splits = args.dataset_split.split(',')
    label_dict = {}
    for config in configs:
        for split in splits:

            raw_outputs_name = os.path.join(save_dir, f'raw_outputs_{config}_{split}.jsonl') 
            print('Evaluating the model on', config, 'aspect and', split, 'split')
            print('*' * 20, 'loading the dataset', '*' * 20)

            ### Load the data
            raw_data = datasets.load_dataset(args.dataset_name, config, split=split, token=HF_TOKEN)

            # Remove the "paper_text" column from the dataset if it exists
            if "paper_text" in raw_data.column_names:
                print('Removing the paper_text column from the dataset')
                raw_data = raw_data.remove_columns("paper_text")

            ##################### INFERENCE THE MODEL ########################

            if 'gpt' not in BASE_MODEL.lower():
                print('Running the model with vllm')
                vllm_inferece(args, raw_data,raw_outputs_name, sampling_params,llm, enable_lora, LORA_PATH, model_name)
            else:
                chatgpt_inference(args, raw_data,raw_outputs_name, save_dir, temperature=args.temperature)


            outputs = []
            with open(raw_outputs_name, 'r') as f:
                for line in f:
                    outputs.append(json.loads(line))

            try:
                
                print('The number of outputs:', len(outputs))
                print('The first output:', outputs[0])
                ########## Extract model predeicitons #############33
                predictions = extract_predictions(outputs)

                print(predictions[0])
                ### Get the stats
            except Exception as e:
                print('Could not extract predictions, you need to do the evaluation manually')
                print(e)

            label_dict[f'{config}_{split}'] = []    
            aspects = [ 'actionability', 'grounding_specificity','verifiability', 'helpfulness'] if (config in  ['all','combined_main_aspects','context_experiment_with_paper_text']) else [config]
            for aspect in aspects:
                print('Extracting predictions for', aspect)
                gold_data_name = args.gold_label_format.replace('ASPECT', aspect)
                gold_data = raw_data[gold_data_name]


                gold_data_rationale = []
                preds_rationale = []
                if 'automatic' in args.dataset_name:
                    print('The dataset is automatic, so we need to get the rationale')
                    
                    rationale_key = gold_data_name.replace('score', 'rationale')
                    # print('The rationale key is:', rationale_key)
                    gold_data_rationale = raw_data[rationale_key]
                    # print('The gold data rationale:', gold_data_rationale[0])
                    preds_rationale = [p[f'{aspect}_rationale'] for p in predictions]
                    # print('The preds rationale:', preds_rationale[0])
                    assert len(preds_rationale) == len(gold_data_rationale), 'The number of predictions and gold rationales do not match'

                ## convert the gold data to string
                gold_data = [str(g) for g in gold_data]
                print('Total number of gold labels:', len(gold_data))
                preds = [p[f'{aspect}_label'] for p in predictions]

                print('Total number of predictions:', len(preds))
                assert len(preds) == len(gold_data), 'The number of predictions and gold labels do not match'

                label_dict[f'{config}_{split}'].append({'gold': gold_data, 'preds': preds, 'aspect': aspect, 'gold_data_rationale': gold_data_rationale, 'preds_rationale': preds_rationale})

            predictions_name = os.path.join(save_dir, f'predictions_{config}_{split}.jsonl')
            with open(predictions_name, 'w') as f:
                print('Saving the predictions to', predictions_name)
                for prediction in predictions:
                    f.write(json.dumps(prediction) + "\n")


    configs_name= '_'.join(configs)
    ######################### Save the results to a file #######################
    results_file_name = os.path.join(save_dir, f'results_{configs_name}_{split}.txt')

    write_stats_to_file(label_dict, results_file_name)


