import re
import pandas as pd
from utils import filter_reviews
from tqdm import tqdm
from semantic_segmentation import merge_short_sentences


############ This has a limitation for the cases when the point enumeration becomes more than 9
############ This also have an issue with examples like "3.2.1" as it will be considered as a point
def split_into_points(text):
    splits_positions = []
    open_parantheses = 0
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)  
    for i, char in enumerate(text):

        # keep track for the open parantheses, for the cases when we have (123)
        if char == '(':
            open_parantheses += 1
        if char == ')':
            open_parantheses -= 1

        ## Ideally, we don't want to split the text if we are inside a parantheses
        if open_parantheses > 0:
            continue

        # match points that starts with:  - , * , = followed by a space
        if char in ['-','*', '='] and text[i+1] == ' ':
            if i > 0:
                # make sure this point is preceded by a sentence ending or new line
                if  text[i-1] in ['.','!','?','\n'] or text[i-2] in ['.','!','?','\n']:
                    splits_positions.append(i)
            else:
                splits_positions.append(i)

        # If the char is a number 
        if char.isnumeric():

            ## check for x.x and x.x.x: 
            if text[i+1] == '.' and text[i+2].isnumeric():
                i = i+2
                continue
            
                
            ## match if the number is followed by: ')' or '.' or ':'
            if text[i+1] in [')','.',':'] and text[i+2] != '\n':
                if i > 1:
                    # make sure this point is preceded by a sentence ending or new line
                    if  text[i-1] in [' ','\n'] or text[i-2] in ['.','!','?','\n']:
                        splits_positions.append(i)
                else:
                    splits_positions.append(i)
            ## Match when we have a number at the begining of a new line 
            elif (i == 0 or text[i-1] == '\n') and text[i+1] == ' ':
                splits_positions.append(i)

    splits_positions = [0] + splits_positions 
    points = [ text[splits_positions[i]:splits_positions[i+1]].strip() for i in range(len(splits_positions)-1)]
    points.append(text[splits_positions[-1]:].strip())

    points = [point for point in points if point]
    return points


# Get the abimportsolute postion of each begining of a word in the text
def word_begging_postition(text):
    positions = []
    length = 0
    for word in text.split():
        positions.append(length)
        length += len(word)+1
    return positions


def split_into_points_new(text):
    splits_positions = []
    open_parantheses = 0
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)  

    word_beging_possitions = word_begging_postition(text)
    new_line_positions = [i for i, char in enumerate(text) if char == '\n']
    words = text.split()
    words = [words.lower() for words in words]
    for i, word in enumerate(words):

        if open_parantheses == 0:
            if word in ['-','*', '=', '•', '+']:
                if i > 0:
                    # make sure this point is preceded by a sentence ending or new line
                    word_begin_pos = word_beging_possitions[i]
                    # check if the word is at the begining of a new line or the last words ended with a sentence ending.
                    if  ( (word_begin_pos -1) in new_line_positions ) or  ( words[i-1][-1] in ['.','!','?', ':'] ):
                        splits_positions.append(word_beging_possitions[i])
                else:
                    splits_positions.append(word_beging_possitions[i])

            ## Match if this is a number, and not preceeded by a word that is a section or figure or table
            ## this matches the cases when we have NUM,  that is not a section or figure or table
            # This also make sure that we are not in the middle of a sentence
            if word.isnumeric():
                if i and words[i-1].replace('.','').replace(':','') \
                    not in ['tab','fig', 'table', 'figure', 'sec', 'section','al']\
                    and words[i-1][-1] in ['.','!','?']:

                    splits_positions.append(word_beging_possitions[i])
                elif not i:
                    splits_positions.append(word_beging_possitions[i])
            
            # This matches the cases when we have NUM that is followed by a '.' or ':' or ')'
            # This also make sure that we are not in the middle of a sentence
            if re.match(r"[a-zA-z]*\d+[.:)]", word):

                if i and (words[i-1][-1] in ['.','!','?', ':']):
                    splits_positions.append(word_beging_possitions[i])
                elif not i:
                    splits_positions.append(word_beging_possitions[i])               

                # if this has a closed parantheses, then we add a dummy ones, so we don't mess the count
                if ')' in word:
                    open_parantheses += word.count(')')
         
        # keep track for the parantheses
        open_parantheses += word.count('(')
        open_parantheses -= word.count(')')


    splits_positions = [0] + splits_positions 
    points = [ text[splits_positions[i]:splits_positions[i+1]].strip() for i in range(len(splits_positions)-1)]
    points.append(text[splits_positions[-1]:].strip())

    points = [point for point in points if point]
    return points






# Example usage:
if __name__ == "__main__":
    paragraph = '''
- This is the (n+1)st paper on discussing biases in models and datasets and it's not clear to me whether this specific bias hasn't been discovered before - not a single 2021 paper cited - models investigated (ESIM, etc.) are a bit old, no novel model is included - the explanation via probing is a bit trivial (this probing has been criticized btw., as unreliable [1]); it's also not clear to me why BERT performs best in the probing, but is least prone to the length diversion bias, questioning the apparent explanation - in l.237, authors write "except for one combination", but there are quite a few negative signs in Table 3, indicating that adversarial training is less often helpful - the losses from length bias are often small, e.g., 1 percentage point for BERT [1] https://aclanthology.org/2020.conll-1.8.pdf 
- Textual matching is also relevant for evaluation metrics [2], where similar biases (e.g., lexical overlap) are discovered. It would be interesting to extend this analysis and also consider models from such communities [2] https://aclanthology.org/2021.emnlp-main.701/ 
'''
    points = split_into_points_new(paragraph)
    for i, point in enumerate(points, 1):
        print(f"Point {i}: {point}")

    
    df = pd.read_csv('../../data/reviewer2_ARR_2022_reviews_gemma2.csv')

    df = filter_reviews(df,'focused_review')

    new_col = []
    with open('/fsx/homes/Abdelrahman.Sadallah@mbzuai.ac.ae/mbzuai/peerq-generation/outputs/test_new_spliting.txt','w')as f:

        for i,r in tqdm(df.iterrows(),total=len(df)):
            focused_review = re.sub(r'\s+', ' ', r['focused_review'])
            splitted_reviews = split_into_points_new(focused_review)
            splitted_reviews = merge_short_sentences(splitted_reviews)

            f.write(f'Focused review:\n\n{r["focused_review"]}\n\n')
            for sr in splitted_reviews:
                f.write(f'Review Point: {sr}\n')
            f.write('='*50 + '\n\n')