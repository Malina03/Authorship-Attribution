import matplotlib.pyplot as plt


def plot_one_author_pos_trigrams(df, doc):
    one_doc = df.loc[df['id']==doc]
    cols = ['nvn','nnv','vnn','vnv','nap','nnc','nad','dna','nnn','nan','vad']
    plt.figure()
    plt.ylabel('POS trigram frequency')
    plt.xlabel('POS trigram')
    for index, row in one_doc.iterrows():
        plt.scatter(cols, row[cols], label=index)
    leg = plt.legend()

def plot_mean_pos_trigrams(df):
    '''
    input: df grouped by id
    '''
    ids = df['id'].unique()
    plt.figure()
    plt.ylabel('POS trigram frequency')
    plt.xlabel('POS trigram')
    cols = ['nvn','nnv','vnn','vnv','nap','nnc','nad','dna','nnn','nan','vad']
    for id in ids:
        plt.scatter(cols, df.loc[df['id']==id, df.columns.isin(cols)], label=id)            
    leg = plt.legend()
 
def plot_word_freq(df):
    
    ids = df['id'].unique()
    plt.figure()
    plt.ylabel('Word frequency')
    plt.xlabel('Words')
    
    cols =['of_freq', 'is_freq', 'the_freq', 'been_freq']
    
    for id in ids:
        plt.scatter(cols, df.loc[df['id']==id, df.columns.isin(cols)], label=id)       
    leg = plt.legend()
    
def plot_yule(df):
    '''
    The larger Yule's K, the smaller the diversity of the vocabulary (and thus, arguably, the easier the text).
    '''
    ids = df['id'].unique()
    
    fig, ax = plt.subplots()
   
    plt.ylabel('Yule Score')
    plt.xlabel('Author ID')
    gap = 0
    for id in ids:
        plt.bar(0 + gap, df.loc[df['id']==id, 'yule'], width = 0.35)  
        gap += 0.5     
    
    ax.set_xticklabels(df['id'].unique())
    
  
def plot_gunning_fog(df):
    '''
    Inidicates grade number 
    '''
    ids = df['id'].unique()
    
    fig, ax = plt.subplots()
   
    plt.ylabel('Gunning Fog Score')
    plt.xlabel('Author ID')
    gap = 0
    for id in ids:
        plt.bar(0 + gap, df.loc[df['id']==id, 'gunning_fog'], width = 0.35)  
        gap += 0.5     
    
    ax.set_xticklabels(ids)
    
def plot_fk_grade(df):
    '''
    Inidicates grade number 
    '''
    ids = df['id'].unique()
    
    fig, ax = plt.subplots()
   
    plt.ylabel('Fleisch Grade Score')
    plt.xlabel('Author ID')
    gap = 0
    for id in ids:
        plt.bar(0 + gap, df.loc[df['id']==id, 'fk_grade'], width = 0.35)  
        gap += 0.5     
    ax.set_xticklabels(ids)


def plot_f_reading(df):
    '''
    Inidicates reading ease. The lower, the harder to read, negative -> confusing.
    '''
    ids = df['id'].unique()
    
    fig, ax = plt.subplots()
   
    plt.ylabel('Fleisch Reading Ease Score')
    plt.xlabel('Author ID')
    gap = 0
    for id in ids:
        plt.bar(0 + gap, df.loc[df['id']==id, 'f_reading'], width = 0.35)  
        gap += 0.5     
    ax.set_xticklabels(ids)

def plot_honore_R(df):
    '''
    Higher the richer the vocab
    '''
    ids = df['id'].unique()
    
    fig, ax = plt.subplots()
   
    plt.ylabel('Honore R Score')
    plt.xlabel('Author ID')
    gap = 0
    for id in ids:
        plt.bar(0 + gap, df.loc[df['id']==id, 'honore_r'], width = 0.35)  
        gap += 0.5     
    ax.set_xticklabels(ids)


def plot_feature(df, feature, title):
    ids = df['id'].unique()
    
    fig, ax = plt.subplots()
   
    plt.ylabel(title)
    plt.xlabel('Author ID')
    gap = 0
    for id in ids:
        plt.bar(0 + gap, df.loc[df['id']==id, feature], width = 0.35)  
        gap += 0.5     
    ax.set_xticklabels(ids)