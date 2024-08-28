
import numpy as np
import ast
from tqdm import tqdm


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
    ## Filtering reviews to remove ones with no valid lists
    df = df[df[review_field].apply(lambda x: len(x) > 0)]    
    df = df.dropna(subset=[review_field])
    print('Number of reviews before removing one-point reviews:', len(df))
    ####################### Removing reviews with one point #######################
    df = df[df[review_field].apply(lambda x: len(x) > 1)]
    print('Number of reviews with more than one point:', len(df))

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

