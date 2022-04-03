import json 
import pickle
import pandas as pd

def load_json(filename):
    with open(filename, 'r') as f:
        data  = json.loads(f.read())
    return data

def get_language(corpus, lang):
    corpus_lang = {}
    for doc in corpus.keys():
        corpus_lang[doc] = corpus[doc][lang]
    return corpus_lang

def resize_docs(corpus, size):
    doc_num = []
    texts  = []
    for doc in corpus.keys():
        if len(' '.join(corpus[doc]).split(' ')) < size:
            doc_num.append(doc)
            texts.append(' '.join(corpus[doc]))
            continue
        i = 0
        doc_len = len(corpus[doc])
        while i < doc_len:
            new_text = corpus[doc][i]
            i += 1 
            while (len(new_text.split(' ')) < size and i < doc_len):
                new_text = new_text + ' ' + corpus[doc][i]
                i += 1
            doc_num.append(doc)
            texts.append(new_text)
            
    corpus_df = pd.DataFrame()
    corpus_df['doc'] = doc_num
    corpus_df['doc'] = pd.to_numeric(corpus_df['doc'])
    corpus_df['text'] = texts
    
    return corpus_df

def load_pickle(filename):
    with open(filename, 'rb') as file:
        df = pickle.load(file)
    return df

def save_pickle(df, filename):
    with open(filename, 'wb') as file:
        pickle.dump(df, file)

def make_numeric(df):
    cols = df.columns
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    return df