import os
from vllm import LLM, SamplingParams
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import json

import re


import re
import ast


import re
import json


known_keys = ['actionability_rationale', 'actionability_label', 'grounding_specificity_rationale' 'grounding_specificity_label', 'verifiability_rationale', 'verifiability_label', 'helpfulness_rationale', 'helpfulness_label']

def escape_inner_quotes(text):
    """Finds specified rationale fields and escapes only internal double quotes."""
    fields = [
        "actionability_rationale",
        "grounding_specificity_rationale",
        "verifiability_rationale",
        "helpfulness_rationale"
    ]
    
    for field in fields:
        pattern = fr'("{field}"\s*:\s*")(.*?)("[\}},])'  # Escape closing brace
        matches = list(re.finditer(pattern, text, re.DOTALL))  # Find all matches first
        
        for match in reversed(matches):  # Process from last to first to avoid index shifting
            before, rationale, after = match.groups()
            escaped_rationale = rationale.replace('"', '\\"')  # Escape only inner quotes
            text = text[:match.start(2)] + escaped_rationale + text[match.end(2):]
    
    return text
def extract_dict(text):
    text = text.replace("\n", " ")  # Remove newlines
    text = text.replace("'", '"')  # Replace single quotes with double quotes
    text = escape_inner_quotes(text)  # Fix quotes inside rationale fields
    print(text)

    match = re.search(r'\{.*?\}', text, re.DOTALL)  # Extract first {...} block
    if match:
        dict_str = match.group()  # Get extracted dictionary string
        
        try:
            return json.loads(dict_str)  # Convert to Python dictionary safely
        except json.JSONDecodeError as e:
            print(f"Parsing error: {e}\nProblematic string: {dict_str}")
            return None
    
    return None  # Return None if no dictionary found

# # Example usage
# input_text = """{"actionability_rationale": "The review comment suggests that the authors should show the gradient conflicts ratio for AlphaNets trained with alpha-divergence in "Table 8" to provide insights. While the action is explicit, it lacks concrete guidance on how to implement this suggestion, such as specifying which parts of the paper should include this information or how to present the gradient conflicts ratio. The authors are given a clear direction but without detailed instructions on execution, making the comment somewhat actionable.", "actionability_label": "3"}"""
# print(extract_dict(input_text))



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


