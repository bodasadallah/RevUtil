import os
from vllm import LLM, SamplingParams
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import json

import re


import re
import ast


def extract_dict(text):
    match = re.search(r'\{[^{}]*\}', text)  # Extract only the first {...} block
    if match:
        dict_str = match.group()  # Get extracted dictionary string
        
        try:
            return ast.literal_eval(dict_str)  # Convert to Python dictionary safely
        except (SyntaxError, ValueError) as e:
            print(f"Parsing error: {e}\nProblematic string: {dict_str}")
            return None
    
    return None  # Return None if no dictionary found


print(extract_dict("{'actionability_label': '3', 'grounding_specificity_label': '3', 'verifiability_label': '3', 'helpfulness_label': '3'}'} \n <|system|>\nYou are an expert in evaluating peer review comments with respect to" ))
def extract_predictions(model_outputs):
    """
    Parses a list of model-generated texts to extract labels and returns a dictionary.
    
    :param model_outputs: List of strings containing model-generated text with labels.
    :return: List of dictionaries with extracted labels.
    """
    extracted_data = []
    
    for text in model_outputs:

        if 'outputs' in text.keys():
            text = text.outputs[0].text
        elif 'generated_text' in text.keys():
            text = text['generated_text']

        extracted_dict = extract_dict(text)
        if  not extracted_dict:
            extracted_data.append({
                'actionability_label': None,
                'grounding_specificity_label': None,
                'verifiability_label': None,
                'helpfulness_label': None
            })
            continue

        parsed_result = {
            'actionability_label': str(extracted_dict.get('actionability_label', None)),
            'grounding_specificity_label':  str(extracted_dict.get('grounding_specificity_label', None)),
            'verifiability_label':  str(extracted_dict.get('verifiability_label', None)),
            'helpfulness_label':  str(extracted_dict.get('helpfulness_label', None))
        }

        extracted_data.append(parsed_result)
    
    return extracted_data

aspects = [ 'actionability', 'grounding_specificity','verifiability', 'helpfulness']
def get_gold_labels(raw_data, dataset_config,aspect_row_name='chatgpt_ASPECT_score'):
    
    gold_labels = []
    dataset_config = aspects if dataset_config == 'all' else dataset_config
    if type(dataset_config) == str:
        dataset_config = [dataset_config]
    
    for row in raw_data:
        row_data = {}
        for aspect in dataset_config:
            row_name = aspect_row_name.replace('ASPECT', aspect)
            if row_name in row:
                row_data[aspect] = row[row_name]
            else:
                row_data[aspect] = None
        gold_labels.append(row_data)

    return gold_labels


