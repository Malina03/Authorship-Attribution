import pandas as pd
import spacy
import contextualSpellCheck
import re
import math
from readability import Readability, exceptions
from spacy_syllables import SpacySyllables


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('syllables', after='tagger', config={"lang": "en_US"})

contextualSpellCheck.add_to_pipe(nlp)

def get_trigram_features(pos_string):
    # Trigrams list
    # NVN NNV VNN NADPPROPN NNCCONJ NOUNPREPDET DETNOUNPREP VNV NNN  NOUNPREPNOUN VPREPDET
    features = {}
    features['nvn'] = len(re.findall('(NOUN VERB NOUN)', pos_string))
    features['nnv'] = len(re.findall('(NOUN NOUN VERB)', pos_string))
    features['vnn'] = len(re.findall('(VERB NOUN NOUN)', pos_string))
    features['vnv'] = len(re.findall('(VERB NOUN VERB)', pos_string))
    features['nap'] = len(re.findall('(NOUN ADP PROPN)', pos_string))
    features['nnc'] = len(re.findall('(NOUN NOUN CCONJ)', pos_string))
    features['nad'] = len(re.findall('(NOUN ADP DET)', pos_string))
    features['dna'] = len(re.findall('(DET NOUN ADP)', pos_string))
    features['nnn'] = len(re.findall('(NOUN NOUN NOUN)', pos_string))
    features['nan'] = len(re.findall('(NOUN ADP NOUN)', pos_string))
    features['vad'] = len(re.findall('(VERB ADP DET)', pos_string))
    
    return features 

def rank_words(word_freq):
    freq_sorted = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
    prev = max(freq_sorted.values())
    rank = 1
    ranks = {}
    for word, freq in freq_sorted.items():
        if freq < prev:
            rank += 1
            prev = freq
        ranks[word] = rank
    return ranks

def get_average_word_rank(ranks):
    return sum(list(ranks.values()))/len(ranks)

def get_word_freq(word, word_freq):
    if word in word_freq.keys():
        return word_freq[word]
    else:
        return 0
    
def compute_freq_of_ranks(freq_dict, word_ranks):
    '''
    Make a dictionary where keys represent ranks and the values are the frequency of the ranks. 
    '''
    freq_of_ranks = {}
    for word in freq_dict.keys():
        rank = word_ranks[word]
        if rank in freq_of_ranks.keys():
            freq_of_ranks[rank] += freq_dict[word]
        else:
            freq_of_ranks[rank] = freq_dict[word]
            
    return freq_of_ranks    

def get_yule_score(freq_dict, word_ranks):
    
    freq_of_ranks = compute_freq_of_ranks(freq_dict, word_ranks)
    tokens = sum([(freq * rank) for rank, freq in freq_of_ranks.items()])
    words_in_freq = sum([freq * (rank ** 2) for rank, freq in freq_of_ranks.items()])
    k = 10000 * ((words_in_freq-tokens)/tokens**2)
    
    return k


def get_honore(freq_dict, text_len):
    hapax_legomena = list(freq_dict.values()).count(1)
    if hapax_legomena == len(freq_dict):
        dif = 0.0001
    else:
        dif = hapax_legomena/len(freq_dict)
    return 100 * math.log(text_len)/(1-dif)

def map_reading_ease(data):
    filtered = data.loc[data['f_reading']!= 0]
    filtered['f_reading'].mask(filtered['f_reading'] == 'very_easy', -3, inplace=True)
    filtered['f_reading'].mask(filtered['f_reading'] == 'easy', -2, inplace=True)
    filtered['f_reading'].mask(filtered['f_reading'] == 'fairly_easy', -1, inplace=True)
    filtered['f_reading'].mask(filtered['f_reading'] == 'standard', 0, inplace=True)
    filtered['f_reading'].mask(filtered['f_reading'] == 'difficult', 1, inplace=True)
    filtered['f_reading'].mask(filtered['f_reading'] == 'fairly_difficult', 2, inplace=True)
    filtered['f_reading'].mask(filtered['f_reading'] == 'very_confusing', 3, inplace=True)

    return filtered


def get_features(df, doc_id):
    
    id_tags = df[doc_id].unique()
    ids = id_tags[:10]
    all_features = pd.DataFrame(columns=['id','yule', 'fk_grade', 'f_reading', 'gunning_fog', 'honore_r', 'avg_word_length', 'syllable_no', 'spelling_errors', 'no_tag', 'sym', 'punct', 'mean_word_rank', 'of_freq', 'is_freq', 'the_freq', 'been_freq','nvn','nnv','vnn','vnv','nap','nnc','nad','dna','nnn','nan','vad'])
    i = 0
    for id in ids:
        features_author = []
        for text in df.loc[df[doc_id]==id, 'text']:
            
            if i % 50 == 0:
                print("Computed features of {} of texts".format(i/len(df['text'])))
            i += 1
            
            if len(text) > 1000000:
                print("Text too long. Skipped.")
                continue
            try:
                tokenized = nlp(text)
            except RuntimeError:
                print("Runtime")
                continue
    
            pos_only = []
            word_freq = {}
            word_length = 0
            syllable_no = 0
            word_count = 0
            
            for token in tokenized:
                if not token.text.isspace():
                    pos_only.append(token.pos_)
                    if token.pos_ not in ['X', 'SYM', 'PUNCT']:
                        word_length += len(token.text)
                        if token._.syllables_count:
                            syllable_no += token._.syllables_count
                        word_count += 1
                    # print(token.lemma_, token.pos_, token.tag_)
                    if token.text not in word_freq.keys():
                        word_freq[token.text] = 1
                    else:
                        word_freq[token.text] += 1
                        
            if len(pos_only) < 1 or word_count < 1:
                continue
                    
            features = get_trigram_features(' '.join(pos_only))
            
            features['id']=id
            features['spelling_errors'] = len(tokenized._.suggestions_spellCheck)
            features['no_tag'] = pos_only.count('X')
            features['sym'] = pos_only.count('SYM')
            features['punct'] = pos_only.count('PUNCT')
            freq_sorted = dict(sorted(word_freq.items(), key=lambda item: item[1]))
            
            word_ranks = rank_words(freq_sorted)
            features['yule'] = get_yule_score(freq_sorted, word_ranks)
            
            
            try:
                r = Readability(text)
                features['fk_grade'] = r.flesch_kincaid().score
                features['f_reading'] = r.flesch().score
                features['gunning_fog'] = r.gunning_fog().score
            except:
                features['fk_grade'] = 0
                features['f_reading'] = 0
                features['gunning_fog'] = 0
                
            
            features['honore_r'] = get_honore(freq_sorted, len(pos_only))
            
            features['mean_word_rank'] = get_average_word_rank(word_ranks)
            features['of_freq'] = get_word_freq('of', freq_sorted)
            features['is_freq'] = get_word_freq('is', freq_sorted)
            features['the_freq'] = get_word_freq('the', freq_sorted)
            features['been_freq'] = get_word_freq('been', freq_sorted)

            features['avg_word_length'] = word_length/word_count
            features['syllable_no'] = syllable_no/word_count
            
            features_author.append(features)
        
        features_df = pd.DataFrame(features_author)
        all_features = all_features.append(features_df, ignore_index=True)
        
    return all_features



# def main():
    
#     nlp = spacy.load("en_core_web_sm")
#     nlp.add_pipe('syllables', after='tagger', config={"lang": "en_US"})
#     contextualSpellCheck.add_to_pipe(nlp)
    
#     data_set = pd.read_csv('../data/blogtext.csv')
#     data_bio = data_set[data_set['topic']=='Biotech']
#     data = data_bio[['id', 'text', 'date']]
#     df = data[data.groupby('id')['id'].transform('size') > 50]

#     features = get_features(df, nlp)
#     with open('../output/features/features_biotech.pkl', 'wb') as file:
#         pickle.dump(features, file)
    
# if __name__ == "__main__":
#     main()