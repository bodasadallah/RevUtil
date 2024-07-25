from vllm import LLM, SamplingParams
import os
import sys
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

llm = LLM(
    model="google/gemma-2-9b-it", 
)

# %%
import os
import sys
from pathlib import Path
module_path = Path(os.path.abspath("")).parent
print(module_path)
sys.path.append(str(module_path))

import pandas as pd
from prompts import PROMPTS
from tqdm import tqdm
reviews = pd.read_csv("../../data/reviewer2_ARR_2022_reviews.csv")
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(
    temperature=0.0, top_p=1, max_tokens=32,
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
)


# %%

aspects = ['Actionability','Constructiveness or Politeness','Credibility or Verifiability','Specificity']
for i, row in tqdm(reviews.iterrows(), total=reviews.shape[0]):
    review = row['focused_review']


    for aspect in aspects:
        aspect_desc = PROMPTS[aspect]
        prompt = PROMPTS['base_prompt'].format(aspect=aspect, aspect_description =aspect_desc, review=review)

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

        reviews.at[i, aspect] = outputs


reviews.to_csv("../../data/reviewer2_ARR_2022_reviews_gemma2.csv", index=False)

