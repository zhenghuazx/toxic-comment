import re
import random
import numpy as np
from tqdm import tqdm

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from lookup import APPO

# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46371
def substitute_repeats_fixed_len(text, nchars, ntimes=3):
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes-1), r"\1", text)


def substitute_repeats(text, ntimes=3):
    for nchars in range(1, 20):
        text = substitute_repeats_fixed_len(text, nchars, ntimes)
    return text


def split_text_and_digits(text, regexps):
    for regexp in regexps:
        result = regexp.match(text)
        if result is not None:
            return ' '.join(result.groups())
    return text

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    s = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", s)
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\=\;\:\|\n])', ' ', s)
    # removing usernames
    s = re.sub("\[\[.*\]", "_USERNAME_", s)
    s = re.sub(r",", " ", s)
    s = re.sub(r"\.", " ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\/", " ", s)
    s = re.sub(r"\^", " ^ ", s)
    s = re.sub(r"\+", " + ", s)
    s = re.sub(r"\-", " - ", s)
    s = re.sub(r"\=", " = ", s)
    s = re.sub(r" e g ", " eg ", s)
    s = re.sub(r" b g ", " bg ", s)
    s = re.sub(r" u s ", " american ", s)
    s = re.sub(r" 9 11 ", "911", s)
    s = re.sub(r"e - mail", "email", s)
    s = re.sub(r"j k", "jk", s)
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

def clean_text(df, tokinizer, wrong_words_dict, regexps, autocorrect=True):
    df.fillna("__NA__", inplace=True)
    texts = df.tolist()
    result = []
    for text in tqdm(texts):
        tokens = tokinizer.tokenize(text.lower())
        tokens = [APPO[re.sub(r"\s\s+", '', token)] if re.sub(r"\s\s+", '', token) in APPO else token for token in
                  tokens]
        #tokens = [split_text_and_digits(token, regexps) for token in tokens]
        tokens = [substitute_repeats(token, 3) for token in tokens]
        text = ' '.join(tokens)
        if autocorrect:
            for wrong, right in wrong_words_dict.items():
                text = text.replace(wrong, right)
        result.append(normalize(text))
    return result


def uniq_words_in_text(text):
    return ' '.join(list(set(text.split())))


def delete_unknown_words(text, embeds):
    return ' '.join([word for word in text.split() if word in embeds])


def convert_text2seq(train_texts, test_texts, max_words, max_seq_len, embeds, lower=True, char_level=False, uniq=False, use_only_exists_words=False, position='pre'):
    tokenizer = Tokenizer(num_words=max_words, lower=lower, char_level=char_level)
    texts = train_texts + test_texts
    if uniq:
        texts = [uniq_words_in_text(text) for text in texts]
    if use_only_exists_words:
        texts = [delete_unknown_words(text, embeds) for text in texts]
    tokenizer.fit_on_texts(texts)
    word_seq_train = tokenizer.texts_to_sequences(train_texts)
    word_seq_test = tokenizer.texts_to_sequences(test_texts)
    word_index = tokenizer.word_index
    word_seq_train = list(sequence.pad_sequences(word_seq_train, maxlen=max_seq_len, padding=position, truncating=position))
    word_seq_test = list(sequence.pad_sequences(word_seq_test, maxlen=max_seq_len, padding=position, truncating=position))
    return word_seq_train, word_seq_test, word_index


def get_embedding_matrix(embed_dim, embeds, max_words, word_index):
    words_not_found = []
    nb_words = min(max_words, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeds[word]
        if embedding_vector is not None and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    return embedding_matrix, words_not_found


def split_data_idx(n, test_size=0.2, shuffle=True, random_state=0):
    train_size = 1 - test_size
    idxs = np.arange(n)
    if shuffle:
        random.seed(random_state)
        random.shuffle(idxs)
    return idxs[:int(train_size*n)], idxs[int(train_size*n):]


def split_data(x, x_aux, y, test_size=0.2, shuffle=True, random_state=0):
    n = len(x)
    train_idxs, test_idxs = split_data_idx(n, test_size, shuffle, random_state)
    return np.array(x[train_idxs]), np.array(x[test_idxs]), x_aux[train_idxs], x_aux[test_idxs],  y[train_idxs], y[test_idxs], train_idxs, test_idxs
