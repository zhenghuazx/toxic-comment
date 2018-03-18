'''
 # Created by hua.zheng on 2/19/18.
 # run this script to generate out-of-vocabulary word embedding by using fasttext
 # my self-generated embedding is dumped in ./data/word2vec-raw.txt
'''

import re
import numpy as np
import pandas as pd
from fastText import load_model
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from util.lookup import APPO
from tqdm import tqdm


embedType='char2vec'

def substitute_repeats_fixed_len(text, nchars, ntimes=3):
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes-1), r"\1", text)

def substitute_repeats(text, ntimes=3):
    copy = text
    for nchars in range(1, 20):
        text = substitute_repeats_fixed_len(text, nchars, ntimes)
    if cosine_similarity(ft_model.get_word_vector(copy), ft_model.get_word_vector(text)) < 0.2:
        return copy
    else:
        return text

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # removing usernames
    s = re.sub("\[\[.*\]", "_USERNAME_", s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()

print('\nLoading data')
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
train['comment_text'] = train['comment_text'].fillna('_NA_')
test['comment_text'] = test['comment_text'].fillna('_NA_')

ft_model = load_model('wiki.en.bin')
n_features = ft_model.get_dimension()
classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]
def aphostophe(words):
    return [APPO[re.sub(r"\s\s+", '', w)] if re.sub(r"\s\s+", '', w) in APPO else w for w in words]

def reduce_text(text):
    return " ".join(aphostophe(text.lower().split(' ')))


train['comment_text'] = train['comment_text'].apply(reduce_text)
test['comment_text'] = test['comment_text'].apply(reduce_text)
train['comment'] = train['comment_text'].apply(normalize)
test['comment'] = test['comment_text'].apply(normalize)

if embedType == 'word2vec':
    tokenizer = Tokenizer(filters='', lower=True, split=" ")
else:
    max_features = 200000
    tokenizer = Tokenizer(num_words=max_features, char_level=True)

tokenizer.fit_on_texts(train['comment'].tolist() + test['comment'].tolist())

ft_model = load_model('/Users/hua.zheng/Documents/Project/kaggle/toxic/library/fastText/wiki.en.bin')
with open('char2vec.txt', 'a') as file:
    for word in tokenizer.word_index:
        vec = ft_model.get_word_vector(word).astype('float32')
        vec_str = np.array2string(vec, formatter={'float_kind': lambda x: "%.4f" % x}, max_line_width=3000)[1:-1]
        file.write(word +' ' + vec_str  + '\n')
