
import numpy as np



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

# Filters reviews based on the length of the reviews.
# We only take revies that has a length of one STD away from the mean
def filter_reviews(df, review_field):

    print('filtering reviwes, and only considering the ones with length of one STD away form mean.')

    print('Number of the reviews before filtering:', len(df))
    lengths = [len(x.split()) for x in df[review_field].tolist()]
    lengths = [x for x in lengths if x < 600 and x > 10]
    lengths = np.array(lengths)
    mean, std = np.mean(lengths.astype(int)), np.std(lengths.astype(int))
    min_length, max_length = mean-std, mean+std
    print('mean:', mean, 'std:', std, 'min:', min_length, 'max:', max_length)

    # filter the df based on the length of the reviews
    df = df[df[review_field].apply(lambda x: len(x.split()) > min_length and len(x.split()) < max_length)]
    print('Number of the reviews after filtering:', len(df))
    return df
