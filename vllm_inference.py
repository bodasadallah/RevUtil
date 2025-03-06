import argparse
import random
import os
from pathlib import Path

import pandas as pd
from vllm import LLM as vLLM, SamplingParams

from prompt import (
    SYSTEM_PROMPT,
    BASE_PROMPT,
    BASE_PROMPT_EXAMPLES,
    ASPECTS_NO_EXAMPLES,
    ASPECTS_WITH_EXAMPLES,
)

def get_prompt(
    review_point,
    aspect,
    prompt_type,
    in_context_examples,
    num_examples_per_label=1,
) -> str:

    prompt = ""
    if prompt_type == "definitions":
        prompt = BASE_PROMPT.format(
            review_point=review_point,
            aspect=aspect,
            aspect_description=ASPECTS_NO_EXAMPLES[aspect],
        )

    elif prompt_type == "definitions_examples":
        prompt = BASE_PROMPT.format(
            review_point=review_point, aspect=aspect, aspect_description=ASPECTS_WITH_EXAMPLES[aspect]
        )
    elif (
        prompt_type == "definitions_incontext_learning"
        or prompt_type == "chain_of_thoughts"
    ):
        examples = ""
        examples_str = []
        ##3 group examples by the label and choose a random example from each group
        for label in in_context_examples[f"{aspect}_label"].unique():
            for _ in range(num_examples_per_label):
                ## keep sampling a line till it is not the same as the currrent review point
                while True:
                    row = in_context_examples[
                        in_context_examples[f"{aspect}_label"] == label
                    ].sample(1)
                    row = row.iloc[0]
                    if row["review_point"] != review_point:
                        break

                score = row[f"{aspect}_label"]
                rationale = (
                    row["rationale"]
                    if prompt_type == "definitions_incontext_learning"
                    else row["chain_of_thoughts"]
                )

                examples_str.append(
                    f"""
    Review Point: {row['review_point']}
    rationale: {rationale}
    score: {score}
    """
                )
        ## shuffle the list
        random.shuffle(examples_str)
        examples = "\n".join(examples_str)

        ## for verifiability, we have two tasks
        prompt = BASE_PROMPT_EXAMPLES.format(
            review_point=review_point,
            aspect=aspect,
            aspect_description=ASPECTS_WITH_EXAMPLES[aspect],
            examples=examples,
        )
    return prompt

def vllm_inputs(
    test_data,
    aspect,
    prompt_type,
    num_examples_per_label,
    in_context_examples
):
    lines = []

    for i, row in test_data.iterrows():
        review_point = row["review_point"]
        prompt = get_prompt(
            review_point=review_point,
            aspect=aspect,
            prompt_type=prompt_type,
            in_context_examples=in_context_examples,
            num_examples_per_label=num_examples_per_label,
        )
        lines.append(prompt)
    return lines

def preprocess_inputs(inputs: list[str], tokenizer):
    processed_inputs = []
    for p in inputs:
        p = [
            {
                "role": "user",
                "content": SYSTEM_PROMPT + " " + p,
            }
        ]
        p = tokenizer.apply_chat_template(
            p, tokenize=False, add_generation_prompt=True
        )
        processed_inputs.append(p)
    return processed_inputs

def main(args):

    if "deepseek" in args.model:
        file_path = f"./chatgpt/outputs/{args.model}_{Path(args.data_path).stem}_results.xlsx"
    else:
        file_path = f"./chatgpt/outputs/{args.model}_temp-{args.temperature:.2f}-topp-{args.top_p:0.2f}-topk-{args.top_k:02d}_{Path(args.data_path).stem}_results.xlsx"
    print(f"Saving results to {file_path}")

    all_incontext_examples = pd.read_excel('./chatgpt/test_data/in_context_examples.xlsx', sheet_name=None)
    in_context_examples = all_incontext_examples[args.aspect]

    llm_kwargs = {}
    # max_model_len = 8192
    max_model_len = 4096 + 2048
    LLM = vLLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        **llm_kwargs,
    )
    TOKENIZER = LLM.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=2048,
    )

    prompt_type = "definitions_incontext_learning"
    num_examples_per_label = 5

    aspect2inputs = {}


    if args.aspect in ["verifiability_extraction", "verifiability_verification"]:
        sheet_name = "verifiability"
    else:
        sheet_name = args.aspect
    test_data = pd.read_excel(args.data_path, sheet_name=sheet_name)
    ### check if test_data has id column and add one if not
    if "id" not in test_data.columns:
        test_data["id"] = range(1, len(test_data) + 1)
        with pd.ExcelWriter(
            args.data_path, mode="a", engine="openpyxl", if_sheet_exists="replace"
        ) as writer:
            test_data.to_excel(writer, sheet_name=args.aspect, index=False)

    inputs = vllm_inputs(
        test_data=test_data,
        aspect=args.aspect,
        prompt_type=prompt_type,
        num_examples_per_label=num_examples_per_label,
        in_context_examples=in_context_examples,
    )
    inputs = preprocess_inputs(inputs, tokenizer=TOKENIZER)
    aspect2inputs[args.aspect] = inputs

    batch_size = 128
    outputs = []
    num_batches = len(inputs) // batch_size
    if len(inputs) % batch_size != 0:
        num_batches += 1
    for i in range(num_batches):
        batch_inputs = inputs[i * batch_size : (i + 1) * batch_size]
        outputs += LLM.generate(batch_inputs, sampling_params)
    # outputs = LLM.generate(inputs, sampling_params)

    outputs = list(map(lambda o: o.outputs[0].text, outputs))
    test_data[f"prompt"] = inputs
    test_data[f"generation"] = outputs

    writer_kwargs = {
        "mode": "a" if os.path.exists(file_path) else "w",
        "if_sheet_exists": "replace" if os.path.exists(file_path) else None,
    }
    with pd.ExcelWriter(file_path, engine="openpyxl", **writer_kwargs) as writer:
        test_data.to_excel(writer, sheet_name=args.aspect, index=False)

if __name__ == "__main__":

    aspects = [
        "actionability",
        "grounding_specificity",
        "verifiability_verification",
        "verifiability_extraction",
        "helpfulness"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--aspect", type=str, choices=aspects, required=True, help="Aspect must be one of: " + ", ".join(aspects))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)

    args = parser.parse_args()
    print(args)
    main(args)

