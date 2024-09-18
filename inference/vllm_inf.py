from vllm import LLM, SamplingParams
import os
import sys
from args_parser import get_args
from inference_utils import prepare_vllm_inference, vllm_inference, chatgpt_inference
import datasets
from torch.utils.data import DataLoader 



# %%
import os
import sys
from pathlib import Path
from utils import labels_stats, extract_label
import pandas as pd
from prompts import PROMPTS
from tqdm import tqdm
import ast

module_path = Path(os.path.abspath("")).parent
print(module_path)
sys.path.append(str(module_path))


if __name__ == "__main__":

    args = get_args()
    input_df = pd.read_csv(args.input_path)

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


    


    # input_df = input_df.sample(1000)

    label_output = { 
    }
    
    for aspect in aspects:
        label_output[f'llm_{aspect}'] = []
        print('Evaluating aspect:', aspect)

        for batch in tqdm(input_dataloader):
        

            batch_inputs = []
            review_points = batch['review_point']

            # print(f'Batch size: {len(batch)}')
            # print(f'batch: {batch}')
            # print(review_points)

            for j in range(len(review_points)):
                review_point = review_points[j]
                aspect_desc = PROMPTS[aspect]
                prompt = PROMPTS[args.prompt].format(aspect=aspect, aspect_description =aspect_desc, review=review_point)
                batch_inputs.append(prompt)


            if 'gemma' in model_name:
                outputs = vllm_inference(model, tokenizer, batch_inputs, sampling_params)
  
            elif 'gpt' in model_name:
                outputs = chatgpt_inference(model_name, batch_inputs, args.temperature, args.top_p, args.max_new_tokens)


            batch_outputs = []
            for j,output in enumerate(outputs):
                
                extracted_score = extract_label(output)


                # print(output)
                # print(f'Extracted score: {extracted_score}')


                label_output[f'llm_{aspect}'].append(extracted_score)

                
                # batch[j][f'llm_{aspect}'] = extracted_score
                

    # print('\n\n',label_output, '\n\n\n')
    ## Add the labels to the input dataframe
    for key in label_output:
        label_output[key] = label_output[key] + ['NO_LABEL' for _ in range(len(input_df) - len(label_output[key]))]

        input_df = input_df.remove_columns(key).add_column(key,label_output[key])


    ## convert  dataset to pandas dataframe
    input_df = input_df.to_pandas()

    print(input_df.head(20))

    # print(input_df)
    input_df.to_csv(args.output_path, index=False)

    if args.run_statistics:
        # Generate stats
        labels_stats(input_df,args.compare_to_human,args.statistics_path)


