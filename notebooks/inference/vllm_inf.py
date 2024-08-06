from vllm import LLM, SamplingParams
import os
import sys
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

llm = LLM(
    model="google/gemma-2-9b-it",
    gpu_memory_utilization=0.9,
    max_model_len=4000
)

# %%
import os
import sys
from pathlib import Path
module_path = Path(os.path.abspath("")).parent
print(module_path)
sys.path.append(str(module_path))
from data_processing.filter_reviews import filter_reviews
from data_processing.semantic_segmentation import split_paragraph


import pandas as pd
from prompts import PROMPTS
from tqdm import tqdm
reviews = pd.read_csv("../../data/reviewer2_ARR_2022_reviews.csv")
split_reviews = pd.read_csv("../../data/reviewer2_ARR_2022_split_reviews.csv")
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(
    temperature=0.0, top_p=1, max_tokens=32,
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
)


# %%




aspects = ['Actionability','Politeness','Verifiability','Specificity']
cnt = 0
for i, row in split_reviews.iterrows():#tqdm(split_reviews.iterrows(), total=split_reviews.shape[0]):

    cur_split_review = row['split_review']
    cur_split_review = cur_split_review.split('$$$')    
    num_of_points = len(cur_split_review)
    for aspect in aspects:
        aspect_desc = PROMPTS[aspect]
        print('evaluting aspect:', aspect)
        aspect_score = 0
        for j,review in enumerate(cur_split_review):
            print(f'evaluting review:{j} out of {num_of_points}')


            prompt = PROMPTS['binary_score_prompt'].format(aspect=aspect, aspect_description =aspect_desc, review=review)

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
            inputs = [c]
            outputs = llm.generate(inputs, sampling_params, use_tqdm= False)[0].outputs[0].text.strip()

            if outputs != '0' or outputs != '1':
                outputs = 0
            outputs = int(outputs)
            aspect_score += outputs

        reviews.at[i, aspect] = aspect_score / num_of_points

    cnt += 1
    print(cnt)
reviews.to_csv("../../data/reviewer2_ARR_2022_reviews_gemma2.csv", index=False)

