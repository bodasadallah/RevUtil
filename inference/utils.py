
import numpy as np
import ast
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score,cohen_kappa_score
import numpy as np
# from prompts import PROMPTS
import re




# def extract_label(text):
#     text = text.replace('*','').lower()
    

#     for l in text.split('\n'):
#         if 'the aspect score is:' in l:
#             for w in l.split():
#                 if w in ['0','1','-1']:
#                     return str(w)
#     return 'NO_LABEL'

def extract_label(output):

    output = output.lower()
    # Extract feedback using a regular expression to capture everything before 'Score:'
    feedback_match = re.search(r'^(.*)score:', output, re.DOTALL)
    feedback = feedback_match.group(1).strip() if feedback_match else None
    
    # Extract score by finding the integer after 'Score:'
    score_match = re.search(r'score:\s*(-?\d)', output)
    score = int(score_match.group(1)) if score_match else None

    if score and not(-1 <= score <= 1):  # Ensure the result is within the valid range
        score = None

    return   feedback, str(score),

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


def prepare_df_to_annotation(df, output_path, total_points = 150):

    print('preparing the data frame for annotation')
    
    ## Adding ID for each review
    df['review_id'] = df.index

    final_df = []
    for i, r in df.iterrows():


        splitted_reviews = ast.literal_eval(r['split_review'])

        for sr in splitted_reviews:
            row = {}
            row['review_id'] = r['review_id']
            row['source'] = r['source']
            row['focused_review'] = r['focused_review']
            row['review_point'] = sr
            row['human_actionability'] = ''
            row['human_specificity'] = ''
            row['human_verifiability'] = ''
            row['human_politeness'] = ''
            row['llm_actionability'] = ''
            row['llm_specificity'] = ''
            row['llm_verifiability'] = ''
            row['llm_politeness'] = ''
            final_df.append(row)

    final_df = pd.DataFrame(final_df)
    
    ## shuffle the data frame
    if total_points == 0:
        total_points = len(final_df)


    final_df = final_df.sample(total_points)
    final_df.to_csv(output_path, index=False)


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



def convert_ternay_prompt_to_prometheus_prompt(aspect):

    prompt = PROMPTS['ternary_score_prompt']
    instruction = prompt.split('ASPECT:')[0]
    criteria = aspect + ':' + PROMPTS[aspect].split('A score of 1,')[0]
    neg_one_desc = ''
    zero_desc = ''
    one_desc = ''

    for l in PROMPTS[aspect].split('\n'):
        if 'A score of -1' in l:
            neg_one_desc = l
        if 'A score of 0' in l:
            zero_desc = l
        if 'A score of 1' in l:
            one_desc = l
    
    rubric_data = {
        "criteria":criteria,
        "score_one_description":one_desc,
        "score_zero_description":zero_desc,
        "score_negone_description":neg_one_desc
    }

    return instruction, rubric_data


# df = pd.read_csv('/home/abdelrahman.sadallah/mbzuai/review_rewrite/outputs/test_output.csv')

# labels_stats(df,compare_to_human = True, stats_path = '/home/abdelrahman.sadallah/mbzuai/review_rewrite/outputs/stats.txt')



import random
import string

import random
import string

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