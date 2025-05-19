
import json

import re


import re
import ast


import re
import json


known_keys = ['actionability_rationale', 'actionability_label', 'grounding_specificity_rationale' 'grounding_specificity_label', 'verifiability_rationale', 'verifiability_label', 'helpfulness_rationale', 'helpfulness_label']


def replace_category_names(text):
    category_mapping = {
        "1: Unverifiable": 1,
        "2: Borderline Verifiable": 2,
        "3: Somewhat Verifiable": 3,
        "4: Mostly Verifiable": 4,
        "5: Fully Verifiable": 5,
        "X: No Claim": "X",

        "1: Unactionable": 1,
        "2: Borderline Actionable": 2,
        "3: Somewhat Actionable": 3,
        "4: Mostly Actionable": 4,
        "5: Highly Actionable": 5,

        "1: Not Grounded": 1,
        "2: Weakly Grounded and Not Specific": 2,
        "3: Weakly Grounded and Specific": 3,
        "4: Fully Grounded and Under-Specific": 4,
        "5: Fully Grounded and Specific": 5,

        "1: Not Helpful at All": 1,
        "2: Barely Helpful": 2,
        "3: Somewhat Helpful": 3,
        "4: Mostly Helpful": 4,
        "5: Highly Helpful": 5
    }

    # Normalize dictionary for case-insensitive matching
    category_mapping_lower = {k.lower(): v for k, v in category_mapping.items()}
    partial_mapping_lower = {k.split(": ", 1)[-1].lower(): v for k, v in category_mapping.items()}

    def replace_match(match):
        matched_text = match.group(0).lower()  # Normalize case
        return str(category_mapping_lower.get(matched_text, partial_mapping_lower.get(matched_text, match.group(0))))

    # Replace full matches first (case-insensitive)
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, category_mapping_lower.keys())) + r')\b', re.IGNORECASE)
    text = pattern.sub(replace_match, text)

    # Replace partial matches (case-insensitive)
    pattern_partial = re.compile(r'\b(?:' + '|'.join(map(re.escape, partial_mapping_lower.keys())) + r')\b', re.IGNORECASE)
    text = pattern_partial.sub(replace_match, text)

    return text


expected_keys = [
    "actionability_rationale",
    "actionability_label",
    "grounding_specificity_rationale",
    "grounding_specificity_label",
    "verifiability_rationale",
    "verifiability_label",
    "helpfulness_rationale",
    "helpfulness_label"
]

import json
import re

import json

def extract_valid_json(text):
    label_keys = [
        "actionability_label",
        "grounding_specificity_label",
        "verifiability_label",
        "helpfulness_label"
    ]

    # Initialize result with None values
    result = {key: None for key in label_keys}

    # Match patterns like: "key": "1", "key": 1, or "key": "X"
    pattern = r'"(' + '|'.join(label_keys) + r')"\s*:\s*"?(X|\d+)"?'

    matches = re.findall(pattern, text)

    for key, val in matches:
        result[key] = str(val)  # Always store as string for valid JSON output

    return json.dumps(result, indent=2)



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
    original  = text

    text = replace_category_names(text)  # Replace category names with numbers
    ## remove double spaces
    text = re.sub(' +', ' ', text)
    ## remove ``` from the text
    # text = text.replace('```', '')
    text = text.replace("-", "")  # Remove leading hyphens
    text = text.replace("\n", " ")  # Remove newlines
    text = text.replace("\\'", "'")  # Fix incorrectly escaped single quotes
    text = text.replace('\\"s', "'s")  # Fix incorrect escaped possessive 's
    text = text.replace("\\\\'", "\\\"")
    text = text.replace("\\\\", "\\")

    text = text.replace("[", "")  # Remove square brackets
    text = text.replace("]", "")  # Remove square brackets

    ############## For Prometheus2 #################
    # text = text.replace("[", '"')  # Replace single quotes with double quotes
    # text = text.replace("]", '"')  # Replace single quotes with double quotes



    ## if text begin with comma or space, remove it
    if text[0] == ',' or text[0] == ' ':
        text = text[1:]

    text = text.replace("\\\\", "\\") # Fix double backslashes
    dict_str  = "" 
    if "```" in text:
        text = text + '#'
        match = re.search(r"```(?:json)?\s*(.*?)(```)?#", text, re.DOTALL)
        if match:
            text = match.group(0)
            ## remove the ```json  and ``` from the text
            text = text.replace('```json', '')
            text = text.replace('```', '')
            text = text.replace('#', '') 

    text = text.strip()  # Remove leading and trailing whitespace

    if not text:
        return None
    
    ############# Comment for Prometheus2 ##############
    if text[0] != '{':
        text = '{' + text + '}'


    ################## cut the text if there is two newlines. This is for Prmetheus2 #########
    # if '\n\n' in text:
    #     halfs = text = text.split('\n\n', 1)
    #     text = halfs[0] if "actionability_label" in halfs[0] else halfs[1]

    if '{' not in text:
        text = '{' + text + '}'


    text = text.replace(" }  { ", ',')  # Remove newlines between dictionaries


    text = text.replace("\n", ' ')  # Remove newlines

    # print(f"Text after processing: {text} \n\n\n\n\n\n\n")

    

    ############################# Prometheus2 ##########################
    # text = extract_valid_json(text)
    # text = text.replace('\\', '')  # Remove single quotes
    # print(f"Text after processing: {text} \n\n\n")

    ############ Some cases doesn't work with replacing the quotes, so trying both ways
    text2 = text
    try:
        text = text.replace("'", '"')  # Replace single quotes with double quotes
        text = escape_inner_quotes(text)  # Fix quotes inside rationale fields
        match = re.search(r'\{.*?\}', text, re.DOTALL)  # Extract first {...} block
        if match:
            dict_str = match.group()  # Get extracted dictionary string
        return json.loads(dict_str)  # Convert to Python dictionary safely
    except json.JSONDecodeError:
        print("Replacing quotes didn't work, trying without replacing quotes.")
        text = text2  # Revert to original text
        match = re.search(r'\{.*?\}', text, re.DOTALL)  # Extract first {...} block
        if match:
            dict_str = match.group()  # Get extracted dictionary string
        try:
            return json.loads(dict_str)  # Convert to Python dictionary safely
        except json.JSONDecodeError as e:
            print(f"Parsing error: {e}\nProblematic string: {dict_str}")
            return None

# Example usage
# input_text = """{\n  \"actionability_rationale\": \"The review clearly identifies specific areas for improvement and provides detailed explanations for the suggested changes. It explicitly asks why certain options were chosen over others, which is a concrete action for the authors to consider. The review also suggests alternative approaches, making it highly actionable.\",\n  \"actionability_label\": \"5\"\n}\n\n{\n  \"grounding_specificity_rationale\": \"The review precisely identifies the specific parts of the paper that need attention. It mentions the use of the 'and' operator and the 'element-wise max' function, providing a clear context for these suggestions. The review also references the 'or' operator and 'element-wise min' function, showing a good understanding of the paper's content and structure.\",\n  \"grounding_specificity_label\": \"5\"\n}\n\n{\n  \"verifiability_rationale\": \"The review provides a logical explanation for its suggestions, comparing the current approach to alternative methods and explaining why the chosen options might be better. It also asks for clarification on why certain options were preferred, which invites further discussion and justification. The review's claims are based on clear reasoning and comparisons, making it verifiable.\",\n  \"verifiability_label\": \"5\"\n}\n\n{\n  \"helpfulness_rationale\": \"The review is highly helpful as it identifies specific issues and provides clear explanations for the suggested changes. It also asks for clarification on certain points, which encourages further discussion and improvement. The review's suggestions are well-reasoned and actionable, making it very beneficial for the authors.\",\n  \"helpfulness_label\": \"5\"\n}\n\n{\n  \"actionability_rationale\": \"The review clearly identifies specific areas for improvement and provides detailed explanations for the suggested changes. It explicitly asks why certain options were chosen over others, which is a concrete action for the authors to consider. The review also suggests alternative approaches, making it highly actionable.\",\n  \"actionability_label\": \"5\"\n}\n\n{\n  \"grounding_specificity_rationale\": \"The review precisely identifies the specific parts of the paper that need attention. It mentions the use of the 'and' operator and the 'element-wise max' function, providing a clear context for these suggestions. The review also references the 'or' operator and 'element-wise min' function, showing a good understanding of the paper's content and structure.\",\n  \"grounding_specificity_label\": \"5\"\n}\n\n{\n  \"verifiability_rationale\": \"The review provides a logical explanation for its suggestions, comparing the current approach to alternative methods and explaining why the chosen options might be better. It also asks for clarification on why certain options were preferred, which invites further discussion and justification. The review's claims are based on clear reasoning and comparisons, making it verifiable.\",\n  \"verifiability_label\": \"5\"\n}\n\n{\n  \"helpfulness_rationale\": \"The review is highly helpful as it identifies specific issues and provides clear explanations for the suggested changes. It also asks for clarification on certain points, which encourages further discussion and improvement. The review's suggestions are well-reasoned and actionable, making it very beneficial for the authors.\",\n  \"helpfulness_label\": \"5\"\n}\n{\n  \"actionability_rationale\": \"The review clearly identifies specific areas for improvement and provides detailed explanations for the suggested changes. It explicitly asks why certain options were chosen over others, which is a concrete action for the authors to consider. The review also suggests alternative approaches, making it highly actionable.\",\n  \"actionability_label\": \"5\"\n}\n\n{\n  \"grounding_specificity_rationale\": \"The review precisely identifies the specific parts of the paper that need attention. It mentions the use of the 'and' operator and the 'element-wise max' function, providing a clear context for these suggestions. The review also references the 'or' operator and 'element-wise min' function, showing a good understanding of the paper's content and structure.\",\n  \"grounding_specificity_label\": \"5\"\n}\n\n{\n  \"verifiability_rationale\": \"The review provides a logical explanation for its suggestions, comparing the current approach to alternative methods and explaining why the chosen options might be better. It also asks for clarification on why certain options were preferred, which invites further discussion and justification. The review's claims are based on clear reasoning and comparisons, making it verifiable.\",\n  \"verifiability_label\": \"5\"\n}\n"""
# print(extract_dict(input_text))

# import json
# lines = []
# with open('/home/abdelrahman.sadallah/mbzuai/review_rewrite/inference/evalute_outputs/base_model/Flow-Judge-v0.1/score_rationale/chat/all/step-0/review_evaluation_human_annotation/raw_outputs_combined_main_aspects_full.jsonl', 'r') as f:
#     for line in f:
#         lines.append(json.loads(line))

# cnt = 0
# keys_cnt = 0
# for line in lines:
#     line = line['generated_text']
#     if extract_dict(line):
#         cnt += 1
#         keys_cnt += sum(1 for value in extract_dict(line).values() if value is not None)
#     else:
#         print(f"^^^^^^^^^^^^^^6Invalid JSON:^^^^^^^^^^^ \n {line} \n\n\n")
#         print(extract_dict(line))
#         # break

# print(f"Number of lines with valid JSON: {cnt}")
# print(f"Number of keys in valid JSON: {keys_cnt}")



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
                'helpfulness_label': None,
                'actionability_rationale': None,
                'grounding_specificity_rationale': None,
                'verifiability_rationale': None,
                'helpfulness_rationale': None
            })
            continue

        parsed_result = {
            'actionability_label': str(extracted_dict.get('actionability_label', None)),
            'grounding_specificity_label':  str(extracted_dict.get('grounding_specificity_label', None)),
            'verifiability_label':  str(extracted_dict.get('verifiability_label', None)),
            'helpfulness_label':  str(extracted_dict.get('helpfulness_label', None)),
            ### rationale keys
            'actionability_rationale':  str(extracted_dict.get('actionability_rationale', None)),
            'grounding_specificity_rationale':  str(extracted_dict.get('grounding_specificity_rationale', None)),
            'verifiability_rationale':  str(extracted_dict.get('verifiability_rationale', None)),
            'helpfulness_rationale':  str(extracted_dict.get('helpfulness_rationale', None))
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


