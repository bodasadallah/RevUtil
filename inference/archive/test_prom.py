from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT
from utils import convert_ternay_prompt_to_prometheus_prompt
import pandas as pd



SCORE_RUBRIC_TEMPLATE_TERNARY = """
[{criteria}]
Score -1: {score_negone_description}
Score 0: {score_zero_description}
Score 1: {score_one_description}
""".strip()


ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer number that is either -1, 0, or 1. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number that is either -1, 0, or 1)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """

model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT_WO_REF)


df = pd.read_csv('/home/abdelrahman.sadallah/mbzuai/review_rewrite/data/annotated_review_points_batch_2.csv')


aspects = ['actionability','politeness','verifiability','specificity']
sample = df.iloc[50]

for aspect in aspects:
    instruction, rubric_data = convert_ternay_prompt_to_prometheus_prompt(aspect)
    response = sample['review_point']

    score_rubric = SCORE_RUBRIC_TEMPLATE_TERNARY.format(**rubric_data)


    feedback, score = judge.single_absolute_grade(
        instruction=instruction,
        response=response,
        rubric=score_rubric,

    )

    print("Aspect:\n", aspect)
    print("Review:\n", response)
    print("Instruction:\n", instruction)
    print("raw Rubric Data:\n", rubric_data)
    print("Rubric:\n", score_rubric)
    print("Feedback:\n", feedback)
    print("Score:\n", score)