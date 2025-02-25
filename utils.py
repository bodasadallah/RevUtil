
from datasets import load_dataset, load_from_disk
import re

from finetuning.prompts import *
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score,cohen_kappa_score, accuracy_score
from scipy import stats

annotators_unique_id_batch_id_map = {
    "boda" : "boda",
    "6158bb338b6122275bc191e3": ["TxZsPCly"],
    "6740484e188a64793529ee77": ["LbMNie2g", "tpuEIMI7", "x8G3vam1", "JMHUapmO", "s46d5sbV", "BXzyXMPh"],
    "6686ebe474531e4a1975636f": ["ZGOQesrg", "nZ3mMZhw", "sBdHWtl1", "qDjzAPQZ", "hgdl9t28", "DFou9r7M"],
    "66cc9c5dc9e1102dc1e40bab": ["2B7nNvBS", "aAISOUiD"],
    "60916bcb7a32838a66b8cb82": ["3O1LNSAk"],
    "67294835e2705b67a725c994": ["9BW3mEvI"],
    "66f8522292767238ed42ebbd": ["pGkwDdmV"],
    "6686f834fd4c9f0bdb7bc8b8": ["fZav2G06", "FrNoCAp2"],
    "6715f821d59317137a6a123b": ["DVRTnFRi", "mw4PwQRk"]
}
annotators_unique_id_batch_id_map_inv ={
    "boda": "boda",
    "TxZsPCly": "6158bb338b6122275bc191e3",
    "LbMNie2g": "6740484e188a64793529ee77",
    "tpuEIMI7": "6740484e188a64793529ee77",
    "x8G3vam1": "6740484e188a64793529ee77",
    "JMHUapmO": "6740484e188a64793529ee77",
    "s46d5sbV": "6740484e188a64793529ee77",
    "BXzyXMPh": "6740484e188a64793529ee77",
    "ZGOQesrg": "6686ebe474531e4a1975636f",
    "nZ3mMZhw": "6686ebe474531e4a1975636f",
    "sBdHWtl1": "6686ebe474531e4a1975636f",
    "qDjzAPQZ": "6686ebe474531e4a1975636f",
    "hgdl9t28": "6686ebe474531e4a1975636f",
    "DFou9r7M": "6686ebe474531e4a1975636f",
    "2B7nNvBS": "66cc9c5dc9e1102dc1e40bab",
    "aAISOUiD": "66cc9c5dc9e1102dc1e40bab",
    "3O1LNSAk": "60916bcb7a32838a66b8cb82",
    "9BW3mEvI": "67294835e2705b67a725c994",
    "pGkwDdmV": "66f8522292767238ed42ebbd",
    "fZav2G06": "6686f834fd4c9f0bdb7bc8b8",
    "FrNoCAp2": "6686f834fd4c9f0bdb7bc8b8",
    "DVRTnFRi": "6715f821d59317137a6a123b",
    "mw4PwQRk": "6715f821d59317137a6a123b"
}









## Create a prompt for each row in the dataset
def get_prompt(row,aspect= 'all',task='train', evaluation_type='score_only'):
    aspects = [ 'actionability', 'grounding_specificity', 'verifiability', 'helpfulness']
    review_point = row['review_point']

    ## if aspect type is string, convert to list
    if isinstance(aspect, str):
        if aspect == 'all':
            considered_aspects = aspects
        else:
            considered_aspects = [aspect]
    else:
        considered_aspects = aspect
    prompt = []


    ################### SYSTEM CONTENT ###################

    SYSTEM_CONTENT = PROMPT_HEADER

    DEFINITIONS = ''
    for aspect in considered_aspects:
        aspect_definition = ASPECT_DEFINITIONS[aspect]
        DEFINITIONS += f'''Aspect: {aspect}\n{aspect_definition}'''

    SYSTEM_CONTENT += DEFINITIONS
    prompt.append({'role': 'system', 'content': SYSTEM_CONTENT})



    ################### USER CONTENT ###################
    if evaluation_type == 'score_only':
        USER_CONTENT = SCORE_ONLY_PROMPT_TAIL.format(review_point=review_point)
    else:
        USER_CONTENT = SCORE_AND_RATIONALE_PROMPT_TAIL.format(review_point=review_point)
    prompt.append({'role': 'user', 'content': USER_CONTENT})


    ################### LABEL CONTENT ###################
    if task == 'train':
        labels_dict = {}
        for aspect in considered_aspects:
            aspect_score_key = f'chatgpt_{aspect}_score'
            aspect_rationale_key = f'chatgpt_{aspect}_rationale'
            aspect_label = row[aspect_score_key]
            aspect_rationale = row[aspect_rationale_key]
            if aspect_label != 'X':
                aspect_label = str(int(aspect_label))
            labels_dict[f'{aspect}_label'] = aspect_label

            if evaluation_type != 'score_only':
                aspect_rationale = row[aspect_rationale_key]
                aspect_rationale_key = f'chatgpt_{aspect}_rationale'
                labels_dict[f'{aspect}_rationale'] = aspect_rationale
        prompt.append({'role': 'assistant', 'content': str(labels_dict)})


    return prompt

row = {'review_point': 'I think the author should provide more examples to support their argument.', 'chatgpt_actionability_score': 1.0, 'chatgpt_actionability_rationale': 'The author should provide more examples to support their argument.', 'chatgpt_grounding_specificity_score': 1.0, 'chatgpt_grounding_specificity_rationale': 'The author should provide more examples to support their argument.', 'chatgpt_verifiability_score': 1.0, 'chatgpt_verifiability_rationale': 'The author should provide more examples to support their argument.', 'chatgpt_helpfulness_score': 1.0, 'chatgpt_helpfulness_rationale': 'The author should provide more examples to support their argument.'}
pr = get_prompt(row,aspect= 'all',task='train', evaluation_type='score_only')

with open('prompt.txt', 'w') as f:
    for item in pr:
        f.write("%s\n" % item)

def extract_output(batch):    
    outputs = []

    return outputs
            


def get_stats(pred, gold, aspect):
    stats_dict = {}

    if aspect in ['actionability', 'grounding_specificity', 'helpfulness']:
        stats_dict['accuracy'] = accuracy_score(pred, gold)
        stats_dict['f1'] = f1_score(pred, gold, average="micro")
        stats_dict['kappa'] = cohen_kappa_score(pred, gold)
        stats_dict['kappa_linear'] = cohen_kappa_score(pred, gold, weights='linear')
        stats_dict['kappa_quadratic'] = cohen_kappa_score(pred, gold, weights='quadratic')
        stats_dict['spearman'] = stats.spearmanr(pred, gold)

    elif aspect == 'verifiability':
        new_pred = []
        new_gold = []
        new_pred_X = []
        new_gold_X = []
        for x, y in zip(pred, gold):
            if x in ['X', 'x', 'NO CLAIM']: x = 'X'
            if y in ['X', 'x', 'NO CLAIM']: y = 'X'

            if x == 'X' or y == 'X':
                x = 0 if x == 'X' else 1
                y = 0 if y == 'X' else 1
                new_pred_X.append(x)
                new_gold_X.append(y)
            else:
                new_pred.append(x)
                new_gold.append(y)
        gold = new_gold
        pred = new_pred
        stats_dict['accuracy'] = accuracy_score(pred, gold)
        stats_dict['f1'] = f1_score(pred, gold, average="micro")
        stats_dict['kappa'] = cohen_kappa_score(pred, gold)
        stats_dict['kappa_linear'] = cohen_kappa_score(pred, gold, weights='linear')
        stats_dict['kappa_quadratic'] = cohen_kappa_score(pred, gold, weights='quadratic')
        stats_dict['spearman'] = stats.spearmanr(pred, gold)
        stats_dict['accuracy_X'] = accuracy_score(new_pred_X, new_gold_X)
        stats_dict['f1_X'] = f1_score(new_pred_X, new_gold_X, average="micro")

    elif aspect in ["professional_tone", 'valid_point', 'addressed_to_author']:
        stats_dict['f1'] = f1_score(pred, gold, average="micro")

    return stats_dict
