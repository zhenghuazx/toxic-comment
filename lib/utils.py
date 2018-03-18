import re
import sys
import json
import logging
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.layers import K, Activation
from keras.engine import Layer
try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import nirvana_dl
except ImportError:
    pass


def load_data(fname, **kwargs):
    func = kwargs.get('func', None)
    if func is not None:
        del kwargs['func']
    df = pd.read_csv(fname, **kwargs)
    if func is not None:
        return func(df.values)
    return df


class Params(object):
    def __init__(self, config=None):
        self._params = self._common_init()
        config_params = self._load_from_file(config)
        self._update_params(config_params)

    def _load_from_file(self, fname):
        if fname is None:
            return {}
        elif fname == 'nirvana':
            return nirvana_dl.params()
        elif isinstance(fname, dict):
            return fname
        with open(fname) as f:
            return json.loads(f.read())

    def _common_init(self):
        common_params = {
            'model_file': None,
            'batch_size': 64,
            'num_epochs': 30,
            'learning_rate': 0.001,
            'use_lr_strategy': True,
            'lr_drop_koef': 0.9,
            'epochs_to_drop': 1,
            'early_stopping_delta': 0.0004,
            'early_stopping_epochs': 5,
            'l2_weight_decay': 0.0001,
            'dropout_val': 0.25,
            'dense_dim': 64,
            'mask_zero': False,
            'train_embeds': False,
            'auxiliary': True,
            'model_checkpoint_dir': os.path.join('.', 'model_checkpoint')}
        params = {'models': [],
                  'dense': common_params,
                  'rcnn': common_params,
                  'cnn': common_params,
                  'lstm': common_params,
                  'gru': common_params}

        params['models'] = ['dense', 'rcnn', 'cnn', 'lstm', 'gru', 'charrnn', 'cnn2d', 'mvcnn', 'cnn2rnn', 'capsule']

        params['dense']['dense_dim'] = 64
        params['dense']['n_layers'] = 10
        params['dense']['concat'] = 0
        params['dense']['pool'] = 'max'

        params['rcnn']['num_filters'] = 64
        params['rcnn']['pool'] = 'max'
        params['rcnn']['add_embeds'] = False
        params['rcnn']['rnn_dim'] = 128
        params['rcnn']['dropout'] = 0.2
        params['rcnn']['recurrent_dropout'] = 0.2

        params['cnn']['num_filters'] = 64
        params['cnn']['pool'] = 'max'
        params['cnn']['n_cnn_layers'] = 2
        params['cnn']['add_embeds'] = True
        params['cnn']['learning_rate'] = 0.0005

        params['capsule']['Num_capsule'] = 10
        params['capsule']['Routing'] = 5
        params['capsule']['add_sigmoid'] = True
        params['capsule']['learning_rate'] = 0.0005

        params['cnn2d']['num_filters'] = 64
        params['cnn2d']['pool'] = 'max'
        params['cnn2d']['n_cnn_layers'] = 2
        params['cnn2d']['add_embeds'] = True
        params['cnn2d']['learning_rate'] = 0.0005


        params['lstm']['rnn_dim'] = 64
        params['lstm']['n_branches'] = 0
        params['lstm']['n_rnn_layers'] = 1
        params['lstm']['n_dense_layers'] = 1
        params['lstm']['kernel_regularizer'] = None
        params['lstm']['recurrent_regularizer'] = None
        params['lstm']['activity_regularizer'] = None
        params['lstm']['dropout'] = 0.2
        params['lstm']['recurrent_dropout'] = 0.2

        params['gru']['rnn_dim'] = 64
        params['gru']['n_branches'] = 0
        params['gru']['n_rnn_layers'] = 1
        params['gru']['n_dense_layers'] = 1
        params['gru']['kernel_regularizer'] = None
        params['gru']['recurrent_regularizer'] = None
        params['gru']['activity_regularizer'] = None
        params['gru']['dropout'] = 0.2
        params['gru']['recurrent_dropout'] = 0.2

        params['charrnn']['rnn_dim'] = 128
        params['charrnn']['dropout'] = 0.2
        params['charrnn']['learning_rate'] = 0.0003
        params['charrnn']['recurrent_dropout'] = 0.2
        params['charrnn']['rnn_type'] = 'gru'

        params['catboost'] = {
            'add_bow': False,
            'bow_top': 100,
            'iterations': 1000,
            'depth': 6,
            'rsm': 1,
            'learning_rate': 0.01,
            'device_config': None}
        return params

    def _update_params(self, params):
        if params is not None and params:
            for key in params.keys():
                if isinstance(params[key], dict):
                    self._params.setdefault(key, {})
                    self._params[key].update(params[key])
                else:
                    self._params[key] = params[key]

    def get(self, key):
        return self._params.get(key, None)


class Embeds(object):
    def __init__(self, fname, w2v_type='fasttext', format='file'):
        if format in ('json', 'pickle'):
            self.load(fname, format)
        elif w2v_type == 'fasttext':
            self.model = self._read_fasttext(fname)
        elif w2v_type == 'glove':
            self.model = self._read_glove(fname)
        elif w2v_type == 'word2vec':
            self.model = KeyedVectors.load_word2vec_format(fname, binary=format == 'binary')
        else:
            self.model = {}

    def __getitem__(self, key):
        try:
            return self.model[key]
        except KeyError:
            return None

    def __contains__(self, key):
        return self[key] is not None

    def _process_line(self, line):
        line = line.rstrip().split(' ')
        word = line[0]
        vec = line[1:]
        return word, [float(val) for val in vec]
    def _read_glove(self, fname):
        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
        model = dict(get_coefs(*o.split(' ')) for o in open(fname))
        return model

    def _read_fasttext(self, fname):
        with open(fname) as f:
            # uncomment if first line is vocabulary size and embedding size
            tech_line = f.readline()
            dict_size, vec_size = self._process_line(tech_line)
            print('dict_size = {}'.format(dict_size))
            print('vec_size = {}'.format(vec_size))
            model = {}
            for line in tqdm(f, file=sys.stdout):
                word, vec = self._process_line(line)
                model[word] = vec
        return model

    def save(self, fname, format='json'):
        if format == 'json':
            with open(fname, 'w') as f:
                json.dump(self.model, f)
        elif format == 'pickle':
            with open(fname, 'wb') as f:
                pickle.dump(self.model, f)
        return self

    def load(self, fname, format='json'):
        if format == 'json':
            with open(fname) as f:
                self.model = json.load(f)
        elif format == 'pickle':
            with open(fname, 'rb') as f:
                self.model = pickle.load(f)
        return self


class Logger(object):
    def __init__(self, logger, fname=None,
                 format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"):
        self.logFormatter = logging.Formatter(format)
        self.rootLogger = logger
        self.rootLogger.setLevel(logging.DEBUG)

        self.consoleHandler = logging.StreamHandler(sys.stdout)
        self.consoleHandler.setFormatter(self.logFormatter)
        self.rootLogger.addHandler(self.consoleHandler)

        if fname is not None:
            self.fileHandler = logging.FileHandler(fname)
            self.fileHandler.setFormatter(self.logFormatter)
            self.rootLogger.addHandler(self.fileHandler)

    def warn(self, message):
        self.rootLogger.warn(message)

    def info(self, message):
        self.rootLogger.info(message)

    def debug(self, message):
        self.rootLogger.debug(message)


class GlobalZeroMaskedAveragePooling1D(Layer):
    def __init__(self, **kwargs):
        super(GlobalZeroMaskedAveragePooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, x, mask=None):
        mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
        n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)
        x_mean = K.sum(x, axis=1, keepdims=False) / (n + 1)
        return K.cast(x_mean, 'float32')

    def compute_mask(self, x, mask=None):
        return None


class GlobalSumPooling1D(Layer):
    def __init__(self, **kwargs):
        super(GlobalSumPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, x, mask=None):
        return K.sum(x, axis=1, keepdims=False)

    def compute_mask(self, x, mask=None):
        return None


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)