
from datasets import load_dataset, load_from_disk
import re

import wandb

from prompt import *
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score,cohen_kappa_score, accuracy_score
from scipy import stats
import numpy as np
import ast
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score,cohen_kappa_score
import numpy as np
# from prompts import PROMPTS
import re
from transformers.integrations import WandbCallback
import random
import string
from transformers import GenerationConfig
import torch
import json
annotators_unique_id_batch_id_map = {
    "boda" : "boda",
    "6158bb338b6122275bc191e3": ["TxZsPCly"],
    "6740484e188a64793529ee77": ["LbMNie2g", "tpuEIMI7", "x8G3vam1", "JMHUapmO", "s46d5sbV", "BXzyXMPh","T5yf801Z"],
    "6686ebe474531e4a1975636f": ["ZGOQesrg", "nZ3mMZhw", "sBdHWtl1", "qDjzAPQZ", "hgdl9t28", "DFou9r7M", "HXjcIUXf"],
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
    "mw4PwQRk": "6715f821d59317137a6a123b",
    'HXjcIUXf': '6686ebe474531e4a1975636f',
    'T5yf801Z': '6740484e188a64793529ee77',
}









## Create a prompt for each row in the dataset
def get_prompt(row,aspect= 'all',task='train', generation_type='score_only', prompt_type='chat', finetuning_type='adapters'):
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

    if task == 'train':
        labels_dict = {}
        for aspect in considered_aspects:
            aspect_score_key = f'chatgpt_{aspect}_score'
            aspect_rationale_key = f'chatgpt_{aspect}_rationale'

            ## Add Rationale 
            if generation_type != 'score_only':
                aspect_rationale = row[aspect_rationale_key]
                aspect_rationale_key = f'chatgpt_{aspect}_rationale'

                # if the rationale has double quotes, escpae them
                # aspect_rationale = aspect_rationale.replace('"', '\\"')
                labels_dict[f'{aspect}_rationale'] = aspect_rationale
              
            ## Add Score
            aspect_label = row[aspect_score_key]
            aspect_rationale = row[aspect_rationale_key]
            if aspect_label != 'X':
                aspect_label = str(int(aspect_label))
            labels_dict[f'{aspect}_label'] = aspect_label

        
        labels_dict = json.dumps(labels_dict, indent=4)


    # if finetuning_type == 'baseline':
    #     assert prompt_type == 'chat', 'Baseline model only supports chat prompt type'

    if prompt_type == 'chat':
        prompt = []
        ################## SYSTEM CONTENT ###################
        SYSTEM_CONTENT = PROMPT_HEADER
        DEFINITIONS = ''
        for aspect in considered_aspects:
            aspect_definition = ASPECTS_NO_EXAMPLES[aspect]
            DEFINITIONS += f'''Aspect: {aspect}\n{aspect_definition}'''

        SYSTEM_CONTENT += DEFINITIONS
        prompt.append({'role': 'system', 'content': SYSTEM_CONTENT})

        ################### USER CONTENT ###################
        if finetuning_type != 'baseline':
            if generation_type == 'score_only':
                USER_CONTENT = SCORE_ONLY_PROMPT_TAIL.format(review_point=review_point)
            else:
                USER_CONTENT = SCORE_AND_RATIONALE_PROMPT_TAIL.format(review_point=review_point)

        else:
            if generation_type == 'score_only':
                USER_CONTENT = BASE_MODEL_SCORE_ONLY_PROMPT_TAIL.format(review_point=review_point)
            else:
                USER_CONTENT = BASE_MODEL_SCORE_AND_RATIONALE_PROMPT_TAIL.format(review_point=review_point)





        prompt.append({'role': 'user', 'content': USER_CONTENT})





        ################### LABEL CONTENT ###################
        if task == 'train':
            prompt.append({'role': 'assistant', 'content': str(labels_dict)})




    ####### If we want to generate instruction prompt, we need to return in in the text column
    elif prompt_type == 'instruction':
        prompt_header = PROMPT_HEADER
        aspect_definitions = ''
        for aspect in considered_aspects:
            aspect_definition = ASPECTS_NO_EXAMPLES[aspect]
            aspect_definitions += f'''Aspect: {aspect}\n{aspect_definition}\n'''

        prompt = f'''###Task Description:
{prompt_header}

{aspect_definitions}
'''
        
        ################### USER CONTENT ###################
        if finetuning_type != 'baseline':
            if generation_type == 'score_only':
                prompt += INSTRUCTION_SCORE_ONLY_PROMPT_TAIL.format(review_point=review_point)
            else:
                prompt += INSTRUCTION_SCORE_AND_RATIONALE_PROMPT_TAIL.format(review_point=review_point)

        else:
            if generation_type == 'score_only':
                prompt += INSTRUCTION_BASE_MODEL_SCORE_ONLY_PROMPT_TAIL.format(review_point=review_point)
            else:
                prompt += INSTRUCTION_BASE_MODEL_SCORE_AND_RATIONALE_PROMPT_TAIL.format(review_point=review_point)


        prompt += '''\n\n###Output:\n'''
        
        if task == 'train':
            prompt += str(labels_dict)


    row['text'] = prompt
    return row

    

row = {'review_point': 'I think the author should provide more examples to support their argument.', 'chatgpt_actionability_score': 1.0, 'chatgpt_actionability_rationale': 'The author should provide more examples to support their argument.', 'chatgpt_grounding_specificity_score': 1.0, 'chatgpt_grounding_specificity_rationale': 'The author should provide more examples to support their argument.', 'chatgpt_verifiability_score': 1.0, 'chatgpt_verifiability_rationale': 'The author should provide more examples to support their argument.', 'chatgpt_helpfulness_score': 1.0, 'chatgpt_helpfulness_rationale': 'The author should provide more examples to support their argument.'}
pr = get_prompt(row,aspect= 'all',task='train', generation_type='score_only', prompt_type = 'instruction')

with open('prompt.txt', 'w') as f:
    if isinstance(pr, list):
        for item in pr:
            f.write("%s\n" % item)
    else:
        f.write(pr['text'])



#### Callback to log samples during training
class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        "A CallBack to log samples a wandb.Table during training"
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                            max_new_tokens=max_new_tokens)
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    def samples_table(self, examples):
        "Create a wandb.Table to store the generations"
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        return records_table
        
    def on_evaluate(self, args, state, control,  **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})



def get_stats(pred, gold, aspect):
    stats_dict = {}

    original_len = len(pred)
    ### Filter out the labels that are not in the possible labels
    possible_labels = [ '1', '2', '3', '4', '5', 'X']
    filtered_pred = []
    filtered_gold = []
    for i in range(len(pred)):
        if pred[i] in possible_labels and gold[i] in possible_labels:
            filtered_pred.append(pred[i])
            filtered_gold.append(gold[i])
    pred = filtered_pred
    gold = filtered_gold

    filtered_len = len(pred)

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
            x = str(x)
            y = str(y)
            
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

    stats_dict['original_len'] = original_len
    stats_dict['filtered_len'] = filtered_len
    
    return stats_dict











def labels_stats(df,compare_to_human = False, stats_path = None):
    aspects = ['actionability','politeness','verifiability','specificity']


    def convert_human_labels_to_int(x):
        x = str(x)
        if '.0' in x:
            return x.replace('.0','')
    ## convert df columns type to string

    for column in df.columns:

        if 'human' in column:
            df[column] = df[column].apply(convert_human_labels_to_int)
        
        df[column] = df[column].astype(str)

    with open(stats_path, 'w') as f:

        print('Number of points:', len(df))
        f.write(f'Number of points: {len(df)}\n')
        print('Number of reviews:', len(df['review_id'].unique()))
        f.write(f'Number of reviews: {len(df["review_id"].unique())}\n')


        num_llm_labels = len(df.loc[df['llm_actionability'].isin(['0','1','-1'])])
        num_human_labels = len(df.loc[df['human_actionability'].isin(['0','1','-1'])])
                             
        print('Number of reviews with LLM labels:',num_llm_labels )
        f.write(f'Number of reviews with LLM labels: {num_llm_labels}\n')
        print('Number of reviews with human labels:', num_human_labels)
        f.write(f'Number of reviews with human labels: {num_human_labels}\n')

        print('Number of reviews with both human and LLM labels:', len(df.loc[df['llm_actionability'].isin(['0','1','-1']) & df['human_actionability'].isin(['0','1','-1'])]))
        f.write(f'Number of reviews with both human and LLM labels: {len(df.loc[df["llm_actionability"].isin(["0","1","-1"]) & df["human_actionability"].isin(["0","1","-1"])])}\n')

        print('-'*100)
        f.write('-'*100 + '\n')
        for aspect in aspects:
            
            human_labels = []
            llm_labels = []
            llm_key = f'llm_{aspect}'
            human_key = f'human_{aspect}'

            aspect_llm_labels = len(df.loc[df[llm_key].isin(['0','1','-1'])])
            aspect_human_labels = len(df.loc[df[human_key].isin(['0','1','-1'])])
            labels = ['-1', '0', '1']

            print('\n')
            f.write('\n')
            print(f'Stats for {aspect} aspect\n')
            f.write(f'Stats for {aspect} aspect\n')

            print(f'LLM labels: {aspect_llm_labels}')
            f.write(f'LLM labels: {aspect_llm_labels}\n')

            print(f'Human labels: {aspect_human_labels}')
            f.write(f'Human labels: {aspect_human_labels}\n')

            # only consider rows that have valid labels  of 0,1,-1 for both human and llm
            curr_df = df.loc[df[llm_key].isin(['0','1','-1']) & df[human_key].isin(['0','1','-1'])]


            print(f'Number of points that have labels for both human and LLM: {len(curr_df)}')
            f.write(f'Number of points that have labels for both human and LLM: {len(curr_df)}\n')

            for label in labels:
                print(f'LLM {label} labels: {len(curr_df[curr_df[llm_key] == label])}')
                f.write(f'LLM {label} labels: {len(curr_df[curr_df[llm_key] == label])}\n')
            #     llm_labels.append(len(curr_df[curr_df[llm_key] == label]))

            for label in labels:
                print(f'Human {label} labels: {len(curr_df[curr_df[human_key] == label])}')
                f.write(f'Human {label} labels: {len(curr_df[curr_df[human_key] == label])}\n')
            #     human_labels.append(len(curr_df[curr_df[human_key] == label]))

            # f.write(str(curr_df[human_key].value_counts()))


            if compare_to_human:

                ## Calcualte the F1 score

                f1 = f1_score(curr_df[human_key], curr_df[llm_key], labels=labels, average='micro')
                kappa_score = cohen_kappa_score(curr_df[human_key], curr_df[llm_key])
                linear_kappa_score = cohen_kappa_score(curr_df[human_key], curr_df[llm_key],weights='linear')
                quadratic_kappa_score = cohen_kappa_score(curr_df[human_key], curr_df[llm_key],weights='quadratic')

                print(f'Kappa score for {aspect} aspect:', kappa_score)
                f.write(f'Kappa score for {aspect} aspect: {kappa_score}\n')

                print(f'Linear Kappa score for {aspect} aspect:', linear_kappa_score)
                f.write(f'Linear Kappa score for {aspect} aspect: {linear_kappa_score}\n')
                
                print(f'Quadratic Kappa score for {aspect} aspect:', quadratic_kappa_score)
                f.write(f'Quadratic Kappa score for {aspect} aspect: {quadratic_kappa_score}\n')

                print(f'F1 score for {aspect} aspect:', f1)
                f.write(f'F1 score for {aspect} aspect: {f1}\n')

                print('Confusion matrix for LLM labels:')
                f.write('Confusion matrix for LLM labels:\n')

                cf = confusion_matrix(curr_df[human_key], curr_df[llm_key], labels=labels)
                print(np.array2string(cf, separator=', '))
                f.write(np.array2string(cf, separator=', '))

                print(f'Confusion matrix for {aspect} aspect')
                ConfusionMatrixDisplay(cf, display_labels=labels ).plot()
                
            print('-'*100)

        



def merge_short_sentences(paragpraphs):
    ret = []
    res = ''
    for p in paragpraphs:
        # if this paragraph is short, then add it to the next one
        if len(p.split()) < 5:
            res = res + p
        else:
            ret.append(res + p)
            res = ''
    if res:
        ret.append(res)  

    return ret

# Filters reviews points  based on the length of the points.
# We only take revies that has a length of one STD away from the mean
def filter_reviews(df, review_field, exclude_short = True, exclude_long = False):

    print('filtering reviwes, and only considering the ones with length of one STD away form mean.')
    print('Number of reviews before filtering:', len(df))


    lengths = []
    num_points = 0
    for x in df[review_field].tolist():
        points  = x
        num_points += len(points)
        for point in points:
            lengths.append(len(point.split()))

    print('Number of the review points before filtering:', num_points)
    lengths = np.array(lengths)
    mean, std = np.mean(lengths.astype(int)), np.std(lengths.astype(int))

    min_length, max_length = mean- 1*std, mean+ 1*std
    print('mean:', mean, 'std:', std, 'min:', min_length, 'max:', max_length)


    num_points_after = 0
    for i,r in tqdm(df.iterrows(),total=len(df)):
        splitted_review = []
        for review in r[review_field]:
            if exclude_short and len(review.split()) < min_length:
                continue
            if exclude_long and len(review.split()) > max_length:
                continue
            
            splitted_review.append(review)

        num_points_after += len(splitted_review)
        df.at[i, 'split_review'] = splitted_review



    print('Number of the reviews after filtering:', num_points_after)

    return df



def clean_text(text):

    # Remove these words if they occur in the beginning of the text "Weaknesses, Strengths, Comments, Suggestions, Feedback" these can be followed by : , and can be lowe case
    text = re.sub(r'^(weaknesses|strengths|comments|suggestions|feedback)[:]*', '', text, flags=re.IGNORECASE)


    # Replace multiple spaces with a single space
    text = re.sub(r'[ ]{2,}', ' ', text)
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{2,}', '\n', text)
    # Strip leading and trailing spaces/newlines from each line
    text = '\n'.join(line.strip() for line in text.splitlines())
    text = text.replace('  ',' ')


    lines = text.split('\n')
    new_lines = []
    for l in lines:
        if l.strip():
            if len(l.split()) < 2 and new_lines:
                new_lines[-1] = new_lines[-1] + ' ' + l
            else:
                new_lines.append(l)
            
    text = '\n'.join(new_lines)
    
    return text





def generate_username_password_list(n, username_length=8, password_length=12):
    """
    Generates a list of N usernames and passwords.

    :param n: Number of usernames and passwords to generate
    :param username_length: Length of each username (default 8)
    :param password_length: Length of each password (default 12)
    :return: A list of dictionaries with 'username' and 'password' keys
    """
    user_list = []

    for _ in range(n):
        # Generate a random username without commas
        username = ''.join(random.choices(string.ascii_letters + string.digits, k=username_length))
        
        # Generate a random password without commas
        password_characters = string.ascii_letters + string.digits + ''.join(c for c in string.punctuation if c != ',')
        password = ''.join(random.choices(password_characters, k=password_length))

        user_list.append({"username": username, "password": password})

    return user_list


def save_to_file(user_list, filename="user_credentials.txt"):
    """
    Saves the list of usernames and passwords to a file.

    :param user_list: List of dictionaries with 'username' and 'password' keys
    :param filename: Name of the file to save the data (default 'user_credentials.txt')
    """
    with open(filename, "w") as file:
        for user in user_list:
            file.write(f"{user['username']}, {user['password']}\n")



def read_from_file(filename="user_credentials.txt"):
    """
    Reads the list of usernames and passwords from a file.

    :param filename: Name of the file to read the data from (default 'user_credentials.txt')
    :return: A list of dictionaries with 'username' and 'password' keys
    """
    user_list = []
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split(", ")
            username = parts[0].strip()
            password = parts[1].strip()
            user_list.append({"username": username, "password": password})
    return user_list