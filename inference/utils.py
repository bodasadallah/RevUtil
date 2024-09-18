
import numpy as np
import ast
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np






def extract_label(text):
    text = text.replace('*','').lower()
    

    for l in text.split('\n'):
        if 'the aspect score is:' in l:
            for w in l.split():
                if w in ['0','1','-1']:
                    return str(w)
    return 'NO_LABEL'

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

        for aspect in aspects:
            
            human_labels = []
            llm_labels = []
            llm_key = f'llm_{aspect}'
            human_key = f'human_{aspect}'

            aspect_llm_labels = len(df.loc[df[llm_key].isin(['0','1','-1'])])
            aspect_human_labels = len(df.loc[df[human_key].isin(['0','1','-1'])])
            labels = ['-1', '0', '1']

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

                f1 = f1_score(curr_df[human_key], curr_df[llm_key], labels=labels, average='macro')
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






# df = pd.read_csv('/home/abdelrahman.sadallah/mbzuai/review_rewrite/outputs/test_output.csv')

# labels_stats(df,compare_to_human = True, stats_path = '/home/abdelrahman.sadallah/mbzuai/review_rewrite/outputs/stats.txt')