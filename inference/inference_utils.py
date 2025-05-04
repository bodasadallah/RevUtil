
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
    
    # if text[0] != '{':
    #     text = '{' + text + '}'

    # if
    if '{' not in text:
        text = '{' + text + '}'


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
# input_text = """\"actionability_rationale\": \"The comment implies an action for the authors to consider. It suggests adding more downstream search methods, which would make the paper more comprehensive.\",\n\"actionability_label\": \"Somewhat Actionable\",\n\"grounding_specificity_rationale\": \"The comment refers to a specific part of the paper, which is the downstream search method section. It is grounded in the fact that the author is questioning the limited options provided for downstream searches.\",\n\"grounding_specificity_label\": \"Fully Grounded and Specific\",\n\"verifiability_rationale\": \"The comment is based on the author's experience and knowledge of other papers in the field. It is not directly verifiable, but it is a common practice to include multiple downstream search methods for a comprehensive study.\",\n\"verifiability_label\": \"Not Verifiable\",\n\"helpfulness_rationale\": \"The comment provides a suggestion that could improve the paper, making it more valuable for the authors. However, it is not very specific in terms of how to implement the suggested change.\",\n\"helpfulness_label\": \"Somewhat Helpful\"\n\nOutput:\n{\"actionability_rationale\": \"The comment implies an action for the authors to consider. It suggests adding more downstream search methods, which would make the paper more comprehensive.\", \"actionability_label\": \"Somewhat Actionable\", \"grounding_specificity_rationale\": \"The comment refers to a specific part of the paper, which is the downstream search method section. It is grounded in the fact that the author is questioning the limited options provided for downstream searches.\", \"grounding_specificity_label\": \"Fully Grounded and Specific\", \"verifiability_rationale\": \"The comment is based on the author\\u0027s experience and knowledge of other papers in the field. It is not directly verifiable, but it is a common practice to include multiple downstream search methods for a comprehensive study.\", \"verifiability_label\": \"Not Verifiable\", \"helpfulness_rationale\": \"The comment provides a suggestion that could improve the paper, making it more valuable for the authors. However, it is not very specific in terms of how to implement the suggested change.\", \"helpfulness_label\": \"Somewhat Helpful\"}s"""
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


