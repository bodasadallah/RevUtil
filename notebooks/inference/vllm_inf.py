from vllm import LLM, SamplingParams
import os
import sys
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

llm = LLM(
    model="google/gemma-2-9b-it",
    gpu_memory_utilization=0.9,
    max_model_len=2048
)

# %%
import os
import sys
from pathlib import Path
from utils import labels_stats, extract_label
module_path = Path(os.path.abspath("")).parent
print(module_path)
sys.path.append(str(module_path))

# from notebooks.data_processing.filter_reviews import filter_reviews
# from notebooks.data_processing.semantic_segmentation import split_paragraph

# from review_rewrite.notebooks.data_processing.filter_reviews import filter_reviews
# from review_rewrite.notebooks.data_processing.semantic_segmentation import split_paragraph


import pandas as pd
from prompts import PROMPTS
from tqdm import tqdm
import ast

# split_reviews = pd.read_csv("/fsx/homes/Abdelrahman.Sadallah@mbzuai.ac.ae/mbzuai/review_rewrite/data/all_reviews.csv")
split_reviews = pd.read_csv("/fsx/homes/Abdelrahman.Sadallah@mbzuai.ac.ae/mbzuai/review_rewrite/data/all_review_points.csv")

tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(
    temperature=0.0, top_p=1, max_tokens=512,
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
)

for column in split_reviews.columns:
    split_reviews[column] = split_reviews[column].astype(str)

# %%


print(split_reviews.columns)

aspects = ['actionability','politeness','verifiability','specificity']
cnt = 0
batch_size = 2

# split_reviews = split_reviews.sample(1000)

for aspect in aspects:

    print('Evaluating aspect:', aspect)

    for i, row in enumerate(tqdm(range(0, len(split_reviews), batch_size))):
    

        batch_inputs = []
        for j in range(batch_size):
            if i+j >= len(split_reviews):
                break
            row = split_reviews.iloc[i+j]
            review_point = row['review_point']

            aspect_desc = PROMPTS[aspect]
            prompt = PROMPTS['ternary_score_prompt'].format(aspect=aspect, aspect_description =aspect_desc, review=review_point)
            conversation = [
                            {
                                'role': 'user', 'content': prompt
                            }
                        ]
            c = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            batch_inputs.append(c)


        outputs = llm.generate(batch_inputs, sampling_params, use_tqdm= False)

        for j,output in enumerate(outputs):
            output = output.outputs[0].text.strip()
            
            extracted_score = extract_label(output)

            if extracted_score not in ['0','1','-1']:
                extracted_score = 'NO_LABEL'

            if i+j >= len(split_reviews):
                break
            split_reviews.at[i+j, f'llm_{aspect}'] = extracted_score


# print(split_reviews)

labels_stats(split_reviews)

split_reviews.to_csv("../../data/gemma_annotated.csv", index=False)

