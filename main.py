'''
 # Created by hua.zheng on 2/17/18.
'''

import re
import string
import os.path
import argparse
import logging
from six import iteritems
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import string
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.client import device_lib
from sklearn.model_selection import KFold
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

from lib.utils import load_data, Embeds, Logger, Params
from lib.features import calc_text_uniq_words, get_bow, get_tfidf, get_most_informative_features
from lib.preprocessing import clean_text, convert_text2seq, get_embedding_matrix, split_data
from lib.models import rcnn, cnn, charrnn, dense, rnn, capsule, save_predictions
from lib.cnn import cnn2d, mvcnn
from lib.rnn import cnn2rnn
from lib.train import train, continue_train
from lib.metrics import calc_metrics, get_metrics, print_metrics


def main(model,
         auxiliary = True,
         model_label = 'rcnn',
         rnn_type='gru',
         padding = 'pre',
         reg = 's',
         prefix = "crawl",
         embedding_file_type = "word2vec",
         train_fname = "./data/train.csv",
         test_fname = "./data/test.csv",
         embeds_fname = "./data/GoogleNews-vectors-negative300.bin",
         logger_fname = "./logs/log-aws",
         mode = "all",
         wrong_words_fname = "./data/correct_words.csv",
         format_embeds = "binary",
         config = "./config.json",
         output_dir = "./out",
         norm_prob = False,
         norm_prob_koef = 1,
         gpus = 0,
         char_level = False,
         random_seed = 2018,
         num_folds = 5):

    embedding_type = prefix + "_" + embedding_file_type

    logger = Logger(logging.getLogger(), logger_fname)

    # ====Detect GPUs====
    logger.debug(device_lib.list_local_devices())

    # ====Load data====
    logger.info('Loading data...')
    train_df = load_data(train_fname)
    test_df = load_data(test_fname)

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    num_classes = len(target_labels)

    # ====Load additional data====
    logger.info('Loading additional data...')
    # swear_words = load_data(swear_words_fname, func=lambda x: set(x.T[0]), header=None)
    wrong_words_dict = load_data(wrong_words_fname, func=lambda x: {val[0]: val[1] for val in x})

    tokinizer = RegexpTokenizer(r'\S+')
    regexps = [re.compile("([a-zA-Z]+)([0-9]+)"), re.compile("([0-9]+)([a-zA-Z]+)")]

    # ====Load word vectors====
    logger.info('Loading embeddings...')
    if model != 'mvcnn':
        embed_dim = 300
        embeds = Embeds(embeds_fname, embedding_file_type, format=format_embeds)

    if mode in ('preprocess', 'all'):
        logger.info('Generating indirect features...')
        # https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda
        # Word count in each comment:
        train_df['count_word'] = train_df["comment_text"].apply(lambda x: len(str(x).split()))
        test_df['count_word'] = test_df["comment_text"].apply(lambda x: len(str(x).split()))
        # Unique word count
        train_df['count_unique_word'] = train_df["comment_text"].apply(lambda x: len(set(str(x).split())))
        test_df['count_unique_word'] = test_df["comment_text"].apply(lambda x: len(set(str(x).split())))
        # Letter count
        train_df['count_letters'] = train_df["comment_text"].apply(lambda x: len(str(x)))
        test_df['count_letters'] = test_df["comment_text"].apply(lambda x: len(str(x)))
        # punctuation count
        train_df["count_punctuations"] = train_df["comment_text"].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation]))
        test_df["count_punctuations"] = test_df["comment_text"].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation]))
        # upper case words count
        train_df["count_words_upper"] = train_df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()]))
        test_df["count_words_upper"] = test_df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()]))
        # title case words count
        train_df["count_words_title"] = train_df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()]))
        test_df["count_words_title"] = test_df["comment_text"].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()]))
        # Word count percent in each comment:
        train_df['word_unique_pct'] = train_df['count_unique_word'] * 100 / train_df['count_word']
        test_df['word_unique_pct'] = test_df['count_unique_word'] * 100 / test_df['count_word']
        # Punct percent in each comment:
        train_df['punct_pct'] = train_df['count_punctuations'] * 100 / train_df['count_word']
        test_df['punct_pct'] = test_df['count_punctuations'] * 100 / test_df['count_word']
        # Average length of the words
        train_df["mean_word_len"] = train_df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        test_df["mean_word_len"] = test_df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        # upper case words percentage
        train_df["words_upper_pct"] = train_df["count_words_upper"] * 100 / train_df['count_word']
        test_df["words_upper_pct"] = test_df["count_words_upper"] * 100 / test_df['count_word']
        # title case words count
        train_df["words_title_pct"] = train_df["count_words_title"] * 100 / train_df['count_word']
        test_df["words_title_pct"] = test_df["count_words_title"] * 100 / test_df['count_word']
        # remove columns
        train_df = train_df.drop('count_word', 1)
        train_df = train_df.drop('count_unique_word', 1)
        train_df = train_df.drop('count_punctuations', 1)
        train_df = train_df.drop('count_words_upper', 1)
        train_df = train_df.drop('count_words_title', 1)
        test_df = test_df.drop('count_word', 1)
        test_df = test_df.drop('count_unique_word', 1)
        test_df = test_df.drop('count_punctuations', 1)
        test_df = test_df.drop('count_words_upper', 1)
        test_df = test_df.drop('count_words_title', 1)

        logger.info('Cleaning text...')
        train_df['comment_text_clear'] = clean_text(train_df['comment_text'], tokinizer, wrong_words_dict, regexps,
                                                    autocorrect=False)
        test_df['comment_text_clear'] = clean_text(test_df['comment_text'], tokinizer, wrong_words_dict, regexps,
                                                   autocorrect=False)
        if reg == 'w':
            # remove all punctuations
            train_df.to_csv(os.path.join(output_dir, 'train_clear_w.csv'), index=False)
            test_df.to_csv(os.path.join(output_dir, 'test_clear_w.csv'), index=False)
            train_df = pd.read_csv(os.path.join(output_dir, 'train_clear_w.csv'))
            test_df = pd.read_csv(os.path.join(output_dir, 'test_clear_w.csv'))
        elif reg == 's':
            # split by S+ keep all punctuations
            train_df.to_csv(os.path.join(output_dir, 'train_clear.csv'), index=False)
            test_df.to_csv(os.path.join(output_dir, 'test_clear.csv'), index=False)
            train_df = pd.read_csv(os.path.join(output_dir, 'train_clear.csv'))
            test_df = pd.read_csv(os.path.join(output_dir, 'test_clear.csv'))

    if mode == 'preprocess':
        return

    if mode == 'processed':
        if reg == 'w':
            train_df = pd.read_csv(os.path.join(output_dir, 'train_clear_w.csv'))
            test_df = pd.read_csv(os.path.join(output_dir, 'test_clear_w.csv'))
        elif reg == 's':
            train_df = pd.read_csv(os.path.join(output_dir, 'train_clear.csv'))
            test_df = pd.read_csv(os.path.join(output_dir, 'test_clear.csv'))

    logger.info('Calc text length...')
    train_df.fillna('unknown', inplace=True)
    test_df.fillna('unknown', inplace=True)
    train_df['text_len'] = train_df['comment_text_clear'].apply(lambda words: len(words.split()))
    test_df['text_len'] = test_df['comment_text_clear'].apply(lambda words: len(words.split()))
    max_seq_len = np.round(train_df['text_len'].mean() + 3 * train_df['text_len'].std()).astype(int)
    logger.debug('Max seq length = {}'.format(max_seq_len))

    # ====Prepare data to NN====
    logger.info('Converting texts to sequences...')
    max_words = 100000
    if char_level:
        max_seq_len = 1200

    train_df['comment_seq'], test_df['comment_seq'], word_index = convert_text2seq(
        train_df['comment_text_clear'].tolist(), test_df['comment_text_clear'].tolist(), max_words, max_seq_len, embeds,
        lower=True, char_level=char_level, uniq=True, use_only_exists_words=True, position=padding)
    logger.debug('Dictionary size = {}'.format(len(word_index)))

    logger.info('Preparing embedding matrix...')
    if model != 'mvcnn':
        embedding_matrix, words_not_found = get_embedding_matrix(embed_dim, embeds, max_words, word_index)

    logger.debug('Embedding matrix shape = {}'.format(np.shape(embedding_matrix)))
    logger.debug('Number of null word embeddings = {}'.format(np.sum(np.sum(embedding_matrix, axis=1) == 0)))

    # ====Train/test split data====
    # train/val
    x_aux = np.matrix([
        train_df["word_unique_pct"].tolist(),
        train_df["punct_pct"].tolist(),
        train_df["mean_word_len"].tolist(),
        train_df["words_upper_pct"].tolist(),
        train_df["words_title_pct"].tolist()], dtype='float32').transpose((1, 0))
    x = np.array(train_df['comment_seq'].tolist())
    y = np.array(train_df[target_labels].values)
    x_train_nn, x_test_nn, x_aux_train_nn, x_aux_test_nn, y_train_nn, y_test_nn, train_idxs, test_idxs = \
        split_data(x, np.squeeze(np.asarray(x_aux)),y,test_size=0.2,shuffle=True,random_state=2018)
    # test set
    test_df_seq = np.array(test_df['comment_seq'].tolist())
    test_aux = np.matrix([
        train_df["word_unique_pct"].tolist(),
        train_df["punct_pct"].tolist(),
        train_df["mean_word_len"].tolist(),
        train_df["words_upper_pct"].tolist(),
        train_df["words_title_pct"].tolist()], dtype='float32').transpose((1, 0))
    test_df_seq_aux = np.squeeze(np.asarray(test_aux))
    y_nn = []
    logger.debug('X shape = {}'.format(np.shape(x_train_nn)))

    # ====Train models====
    params = Params(config)
    if model_label == None:
        logger.warn('Should choose a model to train')
        return

    if model_label == 'dense':
        model = dense(embedding_matrix,
                            num_classes,
                            max_seq_len,
                            dense_dim=params.get('dense').get('dense_dim'),
                            n_layers=params.get('dense').get('n_layers'),
                            concat=params.get('dense').get('concat'),
                            dropout_val=params.get('dense').get('dropout_val'),
                            l2_weight_decay=params.get('dense').get('l2_weight_decay'),
                            pool=params.get('dense').get('pool'),
                            train_embeds=params.get('dense').get('train_embeds'),
                            add_sigmoid=True,
                            gpus=gpus)
    if model_label == 'cnn':
        model = cnn(embedding_matrix,
                        num_classes,
                        max_seq_len,
                        num_filters=params.get('cnn').get('num_filters'),
                        l2_weight_decay=params.get('cnn').get('l2_weight_decay'),
                        dropout_val=params.get('cnn').get('dropout_val'),
                        dense_dim=params.get('cnn').get('dense_dim'),
                        train_embeds=params.get('cnn').get('train_embeds'),
                        n_cnn_layers=params.get('cnn').get('n_cnn_layers'),
                        pool=params.get('cnn').get('pool'),
                        add_embeds=params.get('cnn').get('add_embeds'),
                        auxiliary=auxiliary,
                        add_sigmoid=True,
                        gpus=gpus)
    if model_label == 'cnn2d':
        model = cnn2d(embedding_matrix,
                            num_classes,
                            max_seq_len,
                            num_filters=params.get('cnn2d').get('num_filters'),
                            l2_weight_decay=params.get('cnn2d').get('l2_weight_decay'),
                            dropout_val=params.get('cnn2d').get('dropout_val'),
                            dense_dim=params.get('cnn2d').get('dense_dim'),
                            train_embeds=params.get('cnn2d').get('train_embeds'),
                            add_embeds=params.get('cnn2d').get('add_embeds'),
                            auxiliary=auxiliary,
                            add_sigmoid=True,
                            gpus=gpus)

    if model_label == 'lstm':
        model = rnn(embedding_matrix,
                         num_classes,
                         max_seq_len,
                         l2_weight_decay=params.get('lstm').get('l2_weight_decay'),
                         rnn_dim=params.get('lstm').get('rnn_dim'),
                         dropout_val=params.get('lstm').get('dropout_val'),
                         dense_dim=params.get('lstm').get('dense_dim'),
                         n_branches=params.get('lstm').get('n_branches'),
                         n_rnn_layers=params.get('lstm').get('n_rnn_layers'),
                         n_dense_layers=params.get('lstm').get('n_dense_layers'),
                         train_embeds=params.get('lstm').get('train_embeds'),
                         mask_zero=params.get('lstm').get('mask_zero'),
                         kernel_regularizer=params.get('lstm').get('kernel_regularizer'),
                         recurrent_regularizer=params.get('lstm').get('recurrent_regularizer'),
                         activity_regularizer=params.get('lstm').get('activity_regularizer'),
                         dropout=params.get('lstm').get('dropout'),
                         recurrent_dropout=params.get('lstm').get('recurrent_dropout'),
                         auxiliary=auxiliary,
                         add_sigmoid=True,
                         gpus=gpus,
                         rnn_type='lstm')
    if model_label == 'gru':
        model = rnn(embedding_matrix,
                        num_classes,
                        max_seq_len,
                        l2_weight_decay=params.get('gru').get('l2_weight_decay'),
                        rnn_dim=params.get('gru').get('rnn_dim'),
                        dropout_val=params.get('gru').get('dropout_val'),
                        dense_dim=params.get('gru').get('dense_dim'),
                        n_branches=params.get('gru').get('n_branches'),
                        n_rnn_layers=params.get('gru').get('n_rnn_layers'),
                        n_dense_layers=params.get('gru').get('n_dense_layers'),
                        train_embeds=params.get('gru').get('train_embeds'),
                        mask_zero=params.get('gru').get('mask_zero'),
                        kernel_regularizer=params.get('gru').get('kernel_regularizer'),
                        recurrent_regularizer=params.get('gru').get('recurrent_regularizer'),
                        activity_regularizer=params.get('gru').get('activity_regularizer'),
                        dropout=params.get('gru').get('dropout'),
                        recurrent_dropout=params.get('gru').get('recurrent_dropout'),
                        auxiliary=auxiliary,
                        add_sigmoid=True,
                        gpus=gpus,
                        rnn_type='gru')

    if model_label == 'charrnn':
        model = charrnn(len(word_index),
                                num_classes,
                                max_seq_len,
                                rnn_dim=params.get('charrnn').get('rnn_dim'),
                                dropout_val=params.get('charrnn').get('dropout_val'),
                                auxiliary=auxiliary,
                                dropout=params.get('charrnn').get('dropout'),
                                recurrent_dropout=params.get('charrnn').get('recurrent_dropout'),
                                add_sigmoid=True,
                                gpus=gpus,
                                rnn_type=rnn_type)
    if model_label == 'cnn2rnn':
        model = cnn2rnn(embedding_matrix,
                                num_classes,
                                max_seq_len,
                                rnn_type=rnn_type)
    if model_label == 'dpcnn':
        model = dpcnn(embedding_matrix,
                      num_classes,
                      max_seq_len,
                      num_filters=params.get('dpcnn').get('num_filters'),
                      dense_dim=params.get('dpcnn').get('dense_dim'),
                      add_sigmoid=True,
                      gpus=gpus)

    if model_label == 'rcnn':
        model = rcnn(embedding_matrix,
                          num_classes,
                          max_seq_len,
                          rnn_dim=params.get('rcnn').get('rnn_dim'),
                          dropout_val=params.get('rcnn').get('dropout_val'),
                          dense_dim=params.get('rcnn').get('dense_dim'),
                          train_embeds=params.get('rcnn').get('train_embeds'),
                          auxiliary=auxiliary,
                          dropout=params.get('rcnn').get('dropout'),
                          recurrent_dropout=params.get('rcnn').get('recurrent_dropout'),
                          add_sigmoid=True,
                          gpus=gpus,
                          rnn_type=rnn_type)
    if model_label == 'capsule':
        model = capsule(embedding_matrix,
                                num_classes,
                                max_seq_len,
                                auxiliary=auxiliary,
                                Num_capsule=params.get('capsule').get('Num_capsule'),
                                Routings=params.get('capsule').get('Routing'),
                                add_sigmoid=params.get('capsule').get('add_sigmoid'),
                                mask_zero=params.get('capsule').get('mask_zero'),
                                gpus=gpus,
                                rnn_type='gru')  # lstm may diverge but gru works better

    if model == 'mvcnn':
        embeds_fname1 = "./data/crawl-300d-2M.vec"  # "./data/crawl-300d-2M.vec  word2vec-raw.txt
        embeds_fname2 = "./data/glove.840B.300d.txt"
        embeds_fname3 = "./data/GoogleNews-vectors-negative300.bin"
        embed_dim = 300
        embeds1 = Embeds(embeds_fname1, "glove", format='file')
        embeds2 = Embeds(embeds_fname2, "fasttext", format='file')
        embeds3 = Embeds(embeds_fname3, "word2vec", format='binary')
        embedding_matrix1, words_not_found1 = get_embedding_matrix(embed_dim, embeds1, max_words, word_index)
        embedding_matrix2, words_not_found2 = get_embedding_matrix(embed_dim, embeds2, max_words, word_index)
        #embedding_matrix3, words_not_found3 = get_embedding_matrix(embed_dim, embeds3, max_words, word_index)
        model = mvcnn(embedding_matrix1,
                            embedding_matrix2,
                            num_classes,
                            max_seq_len,
                            auxiliary=auxiliary,
                            gpus=gpus)

    # ====k-fold cross validations split data====
    logger.info('Run k-fold cross validation...')
    params = Params(config)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    oof_train = np.zeros((x.shape[0], num_classes))
    oof_test_skf = []

    for i, (train_index, test_index) in enumerate(kf.split(x, y)):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_aux_train, x_test, x_aux_test = x[train_index], x_aux[train_index], x[test_index], x_aux[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logger.info('Start training {}-th fold'.format(i))
        if auxiliary:
            inputs = [x_train, x_aux_train]
            inputs_val = [x_test, x_aux_test]
            output = [test_df_seq, test_df_seq_aux]
        else:
            inputs = x_train
            inputs_val = x_test
            output = test_df_seq
        hist = train(x_train=inputs,  # [x_train, x_aux_train] when auxiliary input is allowed.
                     y_train=y_train,
                     x_val=inputs_val,  # [x_test, x_aux_test],
                     y_val=y_test,
                     model=model,
                     batch_size=params.get(model_label).get('batch_size'),
                     num_epochs=params.get(model_label).get('num_epochs'),
                     learning_rate=params.get(model_label).get('learning_rate'),
                     early_stopping_delta=params.get(model_label).get('early_stopping_delta'),
                     early_stopping_epochs=params.get(model_label).get('early_stopping_epochs'),
                     use_lr_strategy=params.get(model_label).get('use_lr_strategy'),
                     lr_drop_koef=params.get(model_label).get('lr_drop_koef'),
                     epochs_to_drop=params.get(model_label).get('epochs_to_drop'),
                     model_checkpoint_dir=os.path.join('.', 'model_checkpoint', reg, model_label, embedding_type, padding, str(i)),
                     logger=logger)

        model.load_weights(
            os.path.join('.', 'model_checkpoint', reg, model_label, embedding_type, padding, str(i), 'weights.h5'))
        oof_train[test_index, :] = model.predict(inputs_val)  # model.predict([x_test, x_aux_test])
        proba = model.predict(output)  # model.predict([test_df_seq, test_df_seq_aux])
        oof_test_skf.append(proba)
        result = pd.read_csv("./data/sample_submission.csv")
        result[target_labels] = proba
        ithfold_path = "./cv/{}/{}/{}/{}/{}".format(reg, model_label, embedding_type, padding, i)
        if not os.path.exists(ithfold_path):
            os.makedirs(ithfold_path)

        result.to_csv(os.path.join(ithfold_path, 'sub.csv'), index=False)
        # model.save(os.path.join(ithfold_path,'weights.h5'))

    # dump oof_test and oof_train for later slacking
    # oof_train:
    oof_train_path = "./cv/{}/{}/{}/{}/oof_train".format(reg, model_label, embedding_type, padding)
    if not os.path.exists(oof_train_path):
        os.makedirs(oof_train_path)

    np.savetxt(os.path.join(oof_train_path, "oof_train.csv"), oof_train, fmt='%.24f', delimiter=' ')
    # oof_test: stacking version
    oof_test = np.array(oof_test_skf).mean(axis=0)
    oof_test_path = "./cv/{}/{}/{}/{}/oof_test".format(reg, model_label, embedding_type, padding)
    if not os.path.exists(oof_test_path):
        os.makedirs(oof_test_path)

    np.savetxt(os.path.join(oof_test_path, "oof_test.csv"), oof_test, fmt='%.24f', delimiter=' ')
    # oof_test: submission version
    result[target_labels] = oof_test
    oof_test_bag_path = "./cv/{}/{}/{}/{}/bagged".format(reg, model_label, embedding_type, padding)
    if not os.path.exists(oof_test_bag_path):
        os.makedirs(oof_test_bag_path)

    result.to_csv(os.path.join(oof_test_bag_path, "sub.csv"), index=False)

if __name__ == '__main__':
    main(auxiliary=True,
         model_label='rcnn',
         rnn_type='gru',
         padding='pre',
         reg='s',
         prefix="crawl",
         embedding_file_type="word2vec",
         train_fname="./data/train.csv",
         test_fname="./data/test.csv",
         embeds_fname="./data/GoogleNews-vectors-negative300.bin",
         logger_fname="./logs/log-aws",
         mode="all",
         wrong_words_fname="./data/correct_words.csv",
         format_embeds="binary",
         config="./config.json",
         output_dir="./out",
         norm_prob=False,
         norm_prob_koef=1,
         gpus=0,
         char_level=False,
         random_seed=2018,
         num_folds=5)