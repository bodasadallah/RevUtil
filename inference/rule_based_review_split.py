import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
import importlib
from similarity.normalized_levenshtein import NormalizedLevenshtein
module_path = Path(os.path.abspath("")).parent.parent
print(module_path)
sys.path.append(str(module_path))


import re
import pandas as pd
from notebooks.inference.utils import filter_reviews, merge_short_sentences
from tqdm import tqdm


normalized_levenshtein = NormalizedLevenshtein()


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


CONFERENCES = ['AAAI', 'AAMAS', 'ACL', 'ACMMM', 'ASPLOS', 'CAV', 'CCS', 'CHI', 'COLT', 'CRYPTO', 
 'CSCL', 'DCC', 'DSN', 'FOCS', 'FOGA', 'HPCA', 'ICAPS', 'ICCV', 'ICDE', 'ICDM', 
 'ICFP', 'ICIS', 'ICML', 'ICSE', 'IJCAI', 'IJCAR', 'INFOCOM', 'IPSN', 'ISCA', 
 'ISMAR', 'ISSAC', 'ISWC', 'JCDL', 'KR', 'LICS', 'MOBICOM', 'NIPS', 'OOPSLA', 
 'OSDI', 'PERCOM', 'PERVASIVE', 'PLDI', 'PODC', 'PODS', 'POPL', 'RSS', 'RTSS', 
 'SENSYS', 'SIGCOMM', 'SIGGRAPH', 'SIGIR', 'SIGKDD', 'SIGMETRICS', 'SIGMOD', 
 'SODA', 'SOSP', 'STOC', 'UAI', 'VLDB', 'WWW', 'ACM-HT', 'AH', 'AID', 'AIED', 
 'AIIM', 'AIME', 'ALENEX', 'ALIFE', 'AMAI', 'AMIA', 'AOSD', 'APPROX', 'ASAP', 
 'ASE', 'ASIACRYPT', 'ATVA', 'AVSS', 'BMVC', 'BPM', 'CADE', 'CAIP', 'CANIM', 
 'CASES', 'CBSE', 'CC', 'CCC', 'CCGRID', 'CDC', 'CGI', 'CGO', 'CIDR', 'CIKM', 
 'CLUSTER', 'COCOON', 'COLING', 'CONCUR', 'CP', 'CPAIOR', 'CSB', 'CSCW', 'CSFW', 
 'CSSAC', 'CVPR', 'DAC', 'DAS', 'DASFAA', 'DATE', 'DEXA', 'DIGRA', 'DIS', 'DISC', 
 'DOOD', 'DUX', 'EAAI', 'EACL', 'EASE', 'EC', 'ECAI', 'ECCV', 'ECDL', 'ECIS', 
 'ECML', 'ECOOP', 'ECRTS', 'ECSCW', 'EDBT', 'EKAW', 'EMMSAD', 'EMNLP', 'EMSOFT', 
 'ESA', 'ESEM', 'ESOP', 'ESORICS', 'ESQARU', 'ESWC', 'EUROGRAPH', 'EWSN', 'FCCM', 
 'FLOPS', 'FME', 'FODO', 'FORTE', 'FPSAC', 'FSE', 'FSR', 'FUZZ-IEEE', 'GD', 
 'HOTNETS', 'HPDC', 'ICADL', 'ICALP', 'ICALT', 'ICARCV', 'ICC', 'ICCAD', 'ICCL', 
 'ICCS', 'ICCS', 'ICDAR', 'ICDCS', 'ICDT', 'ICECCS', 'ICER', 'ICGG', 'ICIAP', 
 'ICIP', 'ICLP', 'ICMAS', 'ICNN', 'ICNP', 'ICONIP', 'ICPP', 'ICPR', 'ICS', 
 'ICSM', 'ICSOC', 'ICSP', 'ICSPC', 'ICSR', 'ICTL', 'IDA', 'IEEE-CEC', 'IEEE-MM', 
 'IEEETKDE', 'IJCNLP', 'IJCNN', 'ILPS', 'IM', 'IMC', 'INTERACT', 'IPCO', 'IPDPS', 
 'ISAAC', 'ISD', 'ISESE', 'ISMB', 'ISR', 'ISSCC', 'ISSR', 'ISSRE', 'ISSTA', 
 'ISTA', 'ISTCS', 'ISWC', 'ITS', 'IUI', 'IVCNZ', 'JELIA', 'K-CAP', 'LCN', 
 'LCTES', 'LPAR', 'LPNMR', 'MASCOTS', 'MASS', 'MICRO', 'Middleware', 'MIR', 
 'MMCN', 'MMSP', 'MOBIHOC', 'MobileHCI', 'Mobiquitous', 'Mobisys', 'MODELS', 
 'MSWIM', 'NAACL', 'NDSS', 'NetStore', 'Networking 200X', 'NOSSDAV', 'NSDI', 
 'OPENARCH', 'P2P', 'PACT', 'PADL', 'PADS', 'PAKDD', 'PDC', 'PEPM', 'PERFORMANCE', 
 'PG', 'PKDD', 'PPoPP', 'PPSN', 'PRO-VE', 'PT', 'QoSA', 'QSIC', 'RAID', 'RANDOM', 
 'RE', 'RECOMB', 'RoboCup', 'RST', 'RTA', 'RTAS', 'SARA', 'SAS', 'SAT', 'SCA', 
 'SCC', 'SCG', 'SCOPES', 'SDM', 'SIGCSE', 'SMS', 'SPAA', 'SPICE', 'SRDS', 
 'SSDBM', 'SSPR', 'SSR', 'SSTD', 'STACS', 'SUPER', 'SWAT', 'TABLEAUX', 'TACAS', 
 'TARK', 'TIME', 'TREC', 'UIST', 'UM', 'USENIX', 'VIS', 'VL/HCC', 'VLSI', 
 'VMCAI', 'WACV', 'WADS', 'WISE', 'WoWMoM', 'WPHOL', 'AAAAECC', 'AAIM', 'ACAL', 
 'ACCV', 'ACE', 'ACIS', 'ACISP', 'ACIVS', 'ACOSM', 'ACRA', 'ACS', 'ACSAC', 
 'ACSC', 'ACSD', 'ADBIS', 'ADC', 'ADCS', 'ADHOC-NOW', 'ADTI', 'AI*IA', 'AINA', 
 'AISP', 'ALEX', 'ALG', 'ALP', 'ALTAW', 'AMCIS', 'AMOC', 'ANALCO', 'ANNIE', 
 'ANTS', 'ANZIIS', 'AofA', 'AOIR', 'AOIS', 'AOSE', 'APAMI', 'APBC', 'APCC', 
 'APCHI', 'APLAS', 'APNOMS', 'APSEC', 'APWEB', 'ARA', 'ARES', 'ASADM', 'ASIAN', 
 'ASS', 'ASWEC', 'AUIC', 'AusAI', 'AusDM', 'AusWIT', 'AWOCA', 'AWRE', 'AWTI', 
 'BASYS', 'BNCOD', 'Broadnets', 'CAAI', 'CAAN', 'CACSD', 'CAIA', 'CATS', 
 'CCA', 'CCCG', 'CCW', 'CD', 'CEAS', 'CEC/EEE', 'CGA', 'CHES', 'CIAA', 'CIAC', 
 'CICLING', 'CISTM', 'CITB', 'COCOA', 'COMAD', 'COMMONSENSE', 'CompLife', 
 'COMPSAC', 'CONPAR', 'CPM', 'CSL', 'DAC', 'DAFX', 'DAIS', 'DaWaK', 'DB&IS', 
 'DCOSS', 'DICTA', 'DISRA', 'DITW', 'DLT', 'DMTCS', 'DNA', 'DSOM', 'DS-RT', 
 'DSS', 'DX', 'DYSPAN', 'ECAIM', 'ECAL', 'ECBS', 'ECCB', 'ECEG', 'ECIME', 
 'ECIR', 'ED-MEDIA', 'EDOC', 'EEE', 'EGC', 'Emnets', 'EPIA', 'ER', 'ERCIM/CSCLPERCIM', 
 'ESEA', 'ESEC', 'ESM', 'ESS', 'EuAda', 'EUROGP', 'EuroPDP', 'EUSIPCO', 
 'EWLR', 'FASE', 'FCKAML', 'FCT', 'FEM', 'FEWFDB', 'FIE', 'FINCRY', 'FOSSACS', 
 'FSENCRY', 'FTP', 'FTRTFT', 'FUN', 'GECCO', 'GLOBECOM', 'GMP', 'GPCE', 
 'HASE', 'HICSS', 'HLT', 'HPCN', 'HPSR', 'IAAI', 'ICA3PP', 'ICAIL']

SKIP_WORDS = ['tab','fig', 'table', 'figure', 'sec', 'section','al', 'eqn', 'equation' , 'figs', 'eq', 'vol', 'volume', 'chap', 'chapter']
POINT_DELIMITERS = ['.', '-', '*', '•', '+']
SENTENCE_ENDINGS = ['.','!','?', ':',';']

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
            if word in POINT_DELIMITERS:
                if i > 0:
                    # make sure this point is preceded by a sentence ending or new line
                    word_begin_pos = word_beging_possitions[i]
                    # check if the word is at the begining of a new line or the last words ended with a sentence ending.
                    if  ( (word_begin_pos -1) in new_line_positions ) or  ( words[i-1][-1] in ['.','!','?', ':'] ):
                        splits_positions.append(word_beging_possitions[i])
                else:
                    splits_positions.append(word_beging_possitions[i])

            ## match (W123) or (w123)
            if re.match(r'\([w|W]\d+\)',word):
                splits_positions.append(word_beging_possitions[i])


            ## Match if this is a number, and not preceeded by a word that is a section or figure or table
            ## this matches the cases when we have NUM,  that is not a section or figure or table
            # This also make sure that we are not in the middle of a sentence
            if word.isnumeric():
                if i and words[i-1].replace('.','').replace(':','') \
                    not in (SKIP_WORDS + CONFERENCES)\
                    and words[i-1][-1] in SENTENCE_ENDINGS:

                    splits_positions.append(word_beging_possitions[i])
                elif not i:
                    splits_positions.append(word_beging_possitions[i])
            

    

            # This matches the cases when we have NUM that is followed by a '.' or ':' or ')'
            # This also make sure that we are not in the middle of a sentence
            # make sure this is not a NUM.NUM
            if re.match(r"[a-zA-z]*\d+[.:)]", word) and not re.match(r"\d+.\d+", word):

                if i and (words[i-1][-1] in SENTENCE_ENDINGS):
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



ARROWS= ['->', '=>','-->','==>', '→', '⟶']
## Filters review points that are typos fixes, and review points that doesn't start with dot, dash, star, or number
def filter_typos(points):
    quoted = r'["\'“”](.*?)["\'“”]'
    for point in list(points):
        
        # remove the point if it has the word typo
        if 'typo' in point.lower():
            points.remove(point)
            continue

        erased = False
        # Filter out the points that are typos fixes. We detect typos with the following patterns:
        for x in ARROWS:
            if x in point.split():            
                qouted_text = re.findall(quoted,point)
                # print(qouted_text)
                for i, q in enumerate(qouted_text):
                    for j, q2 in enumerate(qouted_text):
                        if i != j  and normalized_levenshtein.distance(q, q2) < 0.6:
                            points.remove(point)
                            erased = True
                            break
                    if erased:
                        break 
            if erased:
                break
        

    return points

def only_bullet_points(points):
    filtered_points = []
    for point in list(points):
        # Filter out the points that doesn't start with dot, dash, star, or number
        if point[0] in POINT_DELIMITERS or point[0].isnumeric():
            filtered_points.append(point)     
    return filtered_points



def split_and_filter(df,
                    review_key,
                    filter_short_reviews=True, 
                    do_filter_typos=True, 
                    consider_only_bullet_points=True,
                    exclude_short = True,
                    exclude_long = False):

    all_cnt = 0
    cnt_after_typos_bullet_points = 0
    cnt_one_point_reviews = 0

    split_review_column = []

    print(f'Initial number of reviews: {len(df)}')
    ## Filtering reviews to remove ones with no valid lists
    df = df[df[review_key].apply(lambda x: len(x) > 0)]    
    df = df.dropna(subset=[review_key])
    print('Number of reviews after removing  zero-length review', len(df))


    ## Split the reviews into points
    for i,r in tqdm(df.iterrows(),total=len(df)):
        focused_review = re.sub(r'\s+', ' ', r[review_key])
        ## remove the first occurence of weakness or Weakness
        remove_list = ['weakness', 'Weakness','weaknesses', 'Weaknesses']
        for word in remove_list:
            if focused_review.startswith(word) or focused_review.startswith(word + ':'):
                focused_review  = ' '.join(focused_review.split()[1:])
                
        # Split thre review, and merge short sentences
        splitted_reviews = split_into_points_new(focused_review)
        splitted_reviews = merge_short_sentences(splitted_reviews)



        ####################### Removing reviews with one point #######################
        if len(splitted_reviews) == 1:
            cnt_one_point_reviews += 1
            splitted_reviews = []


        all_cnt += len(splitted_reviews)

        if filter_short_reviews:
        ########## Filter short reviews  ###################
            for review in list(splitted_reviews):
                if len(review.split()) < 10: 
                    splitted_reviews.remove(review)

        if do_filter_typos:
            splitted_reviews = filter_typos(splitted_reviews)
        
        if consider_only_bullet_points:
            splitted_reviews = only_bullet_points(splitted_reviews)

        cnt_after_typos_bullet_points += len(splitted_reviews)


        split_review_column.append(splitted_reviews)


    df['split_review'] = split_review_column


    print('Number of reviews with one point: (these reviews will be excluded)', cnt_one_point_reviews)
    print(f'Number of all points (for reviews that has more than one point): {all_cnt}')
    print(f'Number of points after filtering typos and considering only bullet points: {cnt_after_typos_bullet_points}')

    #drop Empty reviews
    df.drop(df[df['split_review'].apply(lambda x: len(x) == 0)].index, inplace=True)

    ### Filter short reviews
    df = filter_reviews(df, 'split_review', exclude_short = exclude_short, exclude_long = exclude_long)

    #drop Empty reviews
    df.drop(df[df['split_review'].apply(lambda x: len(x) == 0)].index, inplace=True)


    return df

# Example usage:
if __name__ == "__main__":
#     paragraph = '''
# Weaknesses:
# (W1) When it comes to the proposed searching methodology and implications for the broader NAS research, the paper does not include any significantly novel parts -- pretty much every single element or insights has been already proposed/made (see details), however not necessarily in the context of (w2) the proposed system is not trivial in the sense that I can imagine putting everything together and achieving strong results was a lot of work, th
# '''
#     points = split_into_points_new(paragraph)
#     for i, point in enumerate(points, 1):
#         print(f"Point {i}: {point}")

    
    df = pd.read_csv('/fsx/homes/Abdelrahman.Sadallah@mbzuai.ac.ae/mbzuai/review_rewrite/data/all_reviews.csv')
    df['focused_review'] = df['focused_review'].astype(str)

    # df = filter_reviews(df,'focused_review')


    cnt1 = 0
    cnt2 = 0
    new_col = []
    with open('/fsx/homes/Abdelrahman.Sadallah@mbzuai.ac.ae/mbzuai/review_rewrite/outputs/test_filtering.txt','w')as f:

        for i,r in tqdm(df.iterrows(),total=len(df)):
            focused_review = re.sub(r'\s+', ' ', r['focused_review'])
            splitted_reviews = split_into_points_new(focused_review)
            splitted_reviews = merge_short_sentences(splitted_reviews)

            cnt1 += len(splitted_reviews)
            cnt2 += len(filter_review_points(splitted_reviews))
            # if splitted_reviews != filter_review_points(splitted_reviews):
            #     f.write(f'Original Review: {r["focused_review"]}\n\n')
            #     f.write(f'Splitted Review: {splitted_reviews}\n\n')
            #     f.write(f'Filtered Review: {filter_review_points(splitted_reviews)}\n\n')
            #     f.write('='*50)

            # f.write(f'Focused review:\n\n{r["focused_review"]}\n\n')
            # for sr in splitted_reviews:
            #     f.write(f'Review Point: {sr}\n')
            # f.write('='*50 + '\n\n')
    print(cnt1,cnt2)