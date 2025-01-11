from vllm import LLM, SamplingParams
import os
import sys
from args_parser import get_args
from inference_utils import prepare_vllm_inference, vllm_inference, chatgpt_inference, prepare_openai_inference
import datasets
from torch.utils.data import DataLoader 
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from utils import convert_ternay_prompt_to_prometheus_prompt
import pandas as pd
from prompts import *



# %%
import os
import sys
from pathlib import Path
from utils import labels_stats, extract_label
import pandas as pd
from prompts import PROMPTS
from tqdm import tqdm

module_path = Path(os.path.abspath("")).parent
print(module_path)
sys.path.append(str(module_path))


if __name__ == "__main__":

    args = get_args()
    input_df = pd.read_csv(args.input_path)

    ## create directory to save outputs if it doesn't exist
    ## Check for the last version for this model and then create a new directory with the version number + 1
    # first create the base directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    ## get the version number
    version = 1
    while os.path.exists(os.path.join(args.output_dir,f'{version}')):
        version += 1
    
    ## create the version directory
    output_dir = os.path.join(args.output_dir,f'{version}')
    os.makedirs(output_dir)


    result_path = f'{output_dir}/results.csv'
    stats_path = f'{output_dir}/stats.txt'

    ### aterate over args, and write them to file
    args_path = f'{output_dir}/args.txt'
    with open(args_path, 'w') as f:
        for key in vars(args):
            f.write(f'{key}: {vars(args)[key]}\n')
        
        f.write(f'Prompt: {PROMPTS[args.prompt]}\n')


    for column in input_df.columns:
        input_df[column] = input_df[column].astype(str)

    input_df = datasets.Dataset.from_pandas(input_df)

    input_dataset = input_df
    print(f'Input Dataset \n {input_df}') 
    if args.total_points  != 0:
        input_dataset =  input_dataset.select(range(args.total_points))
    
    input_dataloader = DataLoader(input_dataset,batch_size=args.batch_size)

    model_name = args.model_name
    aspects = ['actionability','politeness','verifiability','specificity']

    if args.aspect != 'all':
        aspects = [args.aspect]


    print('Model name:', model_name)

    if 'gemma' in model_name:
        model, tokenizer, sampling_params = prepare_vllm_inference(model_name, args.temperature, args.top_p, args.max_new_tokens)


    elif 'prometheus' in model_name:
        model = VLLM(model=model_name)
        sampling_params = {'temperature': args.temperature, 'top_p': args.top_p, 'max_tokens': args.max_new_tokens}


        # model = PrometheusEval(model=model, absolute_grade_template=PROM_ABSOLUTE_PROMPT_WO_REF)
        model = PrometheusEval(model=model, absolute_grade_template=PROMPTS[args.prompt])

    elif 'gpt' in model_name:
        chatgpt_client = prepare_openai_inference(args.chatgpt_key)

    # input_df = input_df.sample(1000)

    label_output = { 
    }
    
    for aspect in aspects:
        label_output[f'llm_{aspect}'] = []
        label_output[f'llm_feedback_{aspect}'] = []
        print('Evaluating aspect:', aspect)

        for batch in tqdm(input_dataloader):
        
            
            log_print = False


            batch_inputs = []
            review_points = batch['review_point']
            feedbacks = []
            # print(f'Batch size: {len(batch)}')
            # print(f'batch: {batch}')
            # print(review_points)

            # if  'prometheus' not in model_name:
            for j in range(len(review_points)):
                review_point = review_points[j]
                aspect_desc = PROMPTS[aspect]
                prompt = PROMPTS[args.prompt].format(aspect=aspect, aspect_description =aspect_desc, review=review_point)
                batch_inputs.append(prompt)


            if 'gemma' in model_name:

                outputs = vllm_inference(model,model_name, tokenizer, batch_inputs, sampling_params)

                if not log_print:
                    log_print = True
                    print(f'Prompt: {batch_inputs[0]}')
                    print(f'Output: {outputs[0]}')

  
            elif 'gpt' in model_name:


                outputs = chatgpt_inference(chatgpt_client,
                                            model_name,
                                            batch_inputs, 
                                            args.temperature, 
                                            args.top_p, 
                                            args.max_new_tokens, 
                                            )

                if not log_print:
                    log_print = True
                    print(f'Prompt: {batch_inputs[0]}')
                    print(f'Output: {outputs[0]}')

                    
            elif 'prometheus' in model_name:
                # instruction, rubric_data = convert_ternay_prompt_to_prometheus_prompt(aspect)
                # score_rubric = PROM_SCORE_RUBRIC_TEMPLATE_TERNARY.format(**rubric_data)


                ########## NEW PROMPTS ##############
                # instruction = PROMPTS['PROM_INSTRUCTION']
                # instructions = [instruction for _ in range(len(review_points))]
                score_rubric = ASPECTS_CRITERIA[aspect]


                ########### CHANGED HERE ##########
                instructions = batch_inputs

                feedbacks, outputs = model.absolute_grade(
                instructions=instructions,
                responses=review_points,
                rubric=score_rubric,
                params=sampling_params)

                if not log_print:
                    log_print = True
                    print(f'Instruction:\n {instructions[0]}')
                    # print(f'Rubric:\n {score_rubric}')
                    print(f'Feedback:\n {feedbacks[0]}')
                    print(f'Output:\n {outputs[0]}')






            batch_outputs = []

            for j,output in enumerate(outputs):
                

                if 'prometheus' in model_name:
                    extracted_score = str(output)
                    feedback = feedbacks[j]
                else:
                    feedback, extracted_score = extract_label(output)

                # print(f'raw output: {output}')
                # print(f'Extracted score: {extracted_score}')
                # print(f'Feedback: {feedback}')

                extracted_score = str(extracted_score)
                if extracted_score not in ['1','-1','0']:
                    extracted_score = 'NO_LABEL'
                if not feedback:
                    feedback = 'NO_FEEDBACK'


                label_output[f'llm_{aspect}'].append(extracted_score)
                
                label_output[f'llm_feedback_{aspect}'].append(feedback) 
                
                # batch[j][f'llm_{aspect}'] = extracted_score
                
    # print('\n\n',label_output, '\n\n\n')
    ## Add the labels to the input dataframe
    for key in label_output.keys():

        # print(f'Key: {key}, Length: {len(label_output[key])}')
        # print('labels:',label_output[key])
        label_output[key] = label_output[key] + ['NO_LABEL' for _ in range(len(input_df) - len(label_output[key]))]

        if key in input_df.column_names:
            input_df = input_df.remove_columns(key).add_column(key,label_output[key])
        else:
            input_df = input_df.add_column(key,label_output[key])


    ## convert  dataset to pandas dataframe
    input_df = input_df.to_pandas()

    print(input_df.head(20))
    for column in input_df.columns:
        if 'llm' in column:
            n = len(input_df[input_df[column] != 'NO_LABEL'])
            print(f'sucess for aspect {column} is: {n} out of {len(input_df)}')
            

   


    # print(input_df)
    input_df.to_csv(result_path, index=False)

    if args.run_statistics:
        # Generate stats
        labels_stats(input_df,args.compare_to_human,stats_path=stats_path)


