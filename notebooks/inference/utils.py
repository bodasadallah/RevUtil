
import numpy as np
import ast
from tqdm import tqdm
import pandas as pd







def extract_label(text):
    text = text.replace('*','').lower()
    

    for l in text.split('\n'):
        if 'the aspect score is:' in l:
            for w in l.split():
                if w in ['0','1','-1']:
                    return str(w)
    return 'NO_LABEL'

def labels_stats(df,compare_to_human = False):
    print('Number of points:', len(df))
    print('Number of reviews:', len(df['review_id'].unique()))
    print('Number of reviews with LLM labels:', len(df[df['llm_actionability'] != 'NO_LABEL']))
    print('Number of reviews with human labels:', len(df[df['human_actionability'] != 'NO_LABEL']))
    print('Number of reviews with both human and LLM labels:', len(df[(df['llm_actionability'] != 'NO_LABEL') & (df['human_actionability'] != 'NO_LABEL')]))

    print(df['llm_actionability'].value_counts())

    for aspect in ['actionability','politeness','verifiability','specificity']:
        
        llm_key = f'llm_{aspect}'
        human_key = f'human_{aspect}'
        print(f'Stats for {aspect} aspect')
        print(f'LLM labels: {len(df[df[llm_key] != "NO_LABEL"])}')
        for label in ['0', '1', '-1']:
            print(f'LLM {label} labels: {len(df[df[llm_key] == label])}')
        print(f'Human labels: {len(df[df[human_key] != "NO_LABEL"])}')
        for label in ['0', '1', '-1']:
            print(f'Human {label} labels: {len(df[df[human_key] == label])}')
        
        print('-'*100)
    print('='*100)



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
    min_length, max_length = mean-std, mean+std
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




t = '''
sdfsdfsdfsdfds
fsdfsdf
**Therefore, the aspect score is: **0**
sdjgljsdgljslkdjgsd sldkjglsdkj
sdgfdsgsd
s
b
2wq'''
print(clean_text(t))
# print(extract_label(t))