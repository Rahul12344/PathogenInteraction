import os
import sys
import pandas as pd
import re
from tqdm import tqdm
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config

def normalize(text):
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()
    text = text.lower()
    text = ''.join([i for i in text if not i.isdigit()])
    
    filtered_sentence = []
    for w in text.split(' '): 
        if w not in stop_words: 
            w = w.lower()
            w = re.sub(r'[\r\n]+', ' ', w)
            w = re.sub(r'[^\x00-\x7F]+', ' ', w)
            filtered_sentence.append(w) 
            
    return ' '.join(filtered_sentence)

def read_filename(c, dataset, filename):
    with open(os.path.join(c[f'{dataset}_dir'], 'text', filename)) as f:
        filtered_sentence = []
        lower_cased_ = f.read()
        for w in lower_cased_.split(' '): 
            if w not in stop_words: 
                filtered_sentence.append(w) 
    return normalize(' '.join(filtered_sentence))

if __name__ == '__main__':
    c = config()
    
    # load the data
    df = pd.read_csv(os.path.join(c['data-directory'], 'dataset_df.csv'))
    
    df = df[df['dataset'] == c['dataset_to_predict']]
    positives = df[(df['label'] == 1) & (df['dataset'] == c['dataset_to_predict'])]['file_name'].tolist()
    negatives = df[(df['label'] == 0) & (df['dataset'] == c['dataset_to_predict'])].sample(n=int(c['ratio']*len(positives)), random_state=c['random_state'])['file_name'].tolist()
    
    rows = []
    for i, file_name in tqdm(enumerate(positives)):
        rows.append((i, read_filename(c, c['dataset_to_predict'], file_name), 1))
    for i, file_name in tqdm(enumerate(negatives)):
        rows.append((i+len(positives), read_filename(c, c['dataset_to_predict'], file_name), 0))
    
    df = pd.DataFrame(rows, columns=['abstract', 'text', 'label'])
    test_df = df.sample(frac=0.2).reset_index(drop=True)
    df = df.drop(test_df.index).reset_index(drop=True)
    dev_df = df.sample(frac=0.25).reset_index(drop=True)
    df = df.drop(dev_df.index).reset_index(drop=True)
    df.to_csv(os.path.join(c['virus_dir'], 'train.tsv'), index=False, sep='\t')
    test_df.to_csv(os.path.join(c['virus_dir'], 'test.tsv'), index=False, sep='\t')
    dev_df.to_csv(os.path.join(c['virus_dir'], 'dev.tsv'), index=False, sep='\t')
    
