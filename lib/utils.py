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

        params['models'] = ['dense', 'rcnn', 'cnn', 'lstm', 'gru', 'charrnn', 'cnn2d', 'mvcnn', 'cnn2rnn', 'capsule', 'dpcnn']

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

        params['dpcnn']['num_filters'] = 64
        params['dpcnn']['dense_dim'] = 256
        params['dpcnn']['filter_size'] = 3

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

    # refer https://github.com/titu1994/Keras-Multiplicative-LSTM

    __all__ = ['MultiplicativeLSTM']

    from keras import backend as K
    from keras import activations
    from keras import initializers
    from keras import regularizers
    from keras import constraints
    from keras.engine import Layer
    from keras.engine import InputSpec
    from keras.legacy import interfaces
    from keras.layers import Recurrent

    import tensorflow as tf
    tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

    def _time_distributed_dense(x, w, b=None, dropout=None,
                                input_dim=None, output_dim=None,
                                timesteps=None, training=None):
        """Apply `y . w + b` for every temporal slice y of x.
        # Arguments
            x: input tensor.
            w: weight matrix.
            b: optional bias vector.
            dropout: wether to apply dropout (same dropout mask
                for every temporal slice of the input).
            input_dim: integer; optional dimensionality of the input.
            output_dim: integer; optional dimensionality of the output.
            timesteps: integer; optional number of timesteps.
            training: training phase tensor or boolean.
        # Returns
            Output tensor.
        """
        if not input_dim:
            input_dim = K.shape(x)[2]
        if not timesteps:
            timesteps = K.shape(x)[1]
        if not output_dim:
            output_dim = K.shape(w)[1]

    class MultiplicativeLSTM(Recurrent):
        """Multiplicative Long-Short Term Memory unit - https://arxiv.org/pdf/1609.07959.pdf
        # Arguments
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use
                (see [activations](../activations.md)).
                If you pass None, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            recurrent_activation: Activation function to use
                for the recurrent step
                (see [activations](../activations.md)).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
                used for the linear transformation of the inputs.
                (see [initializers](../initializers.md)).
            recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix,
                used for the linear transformation of the recurrent state.
                (see [initializers](../initializers.md)).
            bias_initializer: Initializer for the bias vector
                (see [initializers](../initializers.md)).
            unit_forget_bias: Boolean.
                If True, add 1 to the bias of the forget gate at initialization.
                Setting it to true will also force `bias_initializer="zeros"`.
                This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            recurrent_regularizer: Regularizer function applied to
                the `recurrent_kernel` weights matrix
                (see [regularizer](../regularizers.md)).
            bias_regularizer: Regularizer function applied to the bias vector
                (see [regularizer](../regularizers.md)).
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
                (see [regularizer](../regularizers.md)).
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix
                (see [constraints](../constraints.md)).
            recurrent_constraint: Constraint function applied to
                the `recurrent_kernel` weights matrix
                (see [constraints](../constraints.md)).
            bias_constraint: Constraint function applied to the bias vector
                (see [constraints](../constraints.md)).
            dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the recurrent state.
        # References
            - [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
            - [Learning to forget: Continual prediction with MultiplicativeLSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
            - [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
            - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        """

        @interfaces.legacy_recurrent_support
        def __init__(self, units,
                     activation='tanh',
                     recurrent_activation='hard_sigmoid',
                     use_bias=True,
                     kernel_initializer='glorot_uniform',
                     recurrent_initializer='orthogonal',
                     bias_initializer='zeros',
                     unit_forget_bias=True,
                     kernel_regularizer=None,
                     recurrent_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     recurrent_constraint=None,
                     bias_constraint=None,
                     dropout=0.,
                     recurrent_dropout=0.,
                     **kwargs):
            super(MultiplicativeLSTM, self).__init__(**kwargs)
            self.units = units
            self.activation = activations.get(activation)
            self.recurrent_activation = activations.get(recurrent_activation)
            self.use_bias = use_bias

            self.kernel_initializer = initializers.get(kernel_initializer)
            self.recurrent_initializer = initializers.get(recurrent_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.unit_forget_bias = unit_forget_bias

            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)

            self.kernel_constraint = constraints.get(kernel_constraint)
            self.recurrent_constraint = constraints.get(recurrent_constraint)
            self.bias_constraint = constraints.get(bias_constraint)

            self.dropout = min(1., max(0., dropout))
            self.recurrent_dropout = min(1., max(0., recurrent_dropout))
            self.state_spec = [InputSpec(shape=(None, self.units)),
                               InputSpec(shape=(None, self.units))]

        def build(self, input_shape):
            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            batch_size = input_shape[0] if self.stateful else None
            self.input_dim = input_shape[2]
            self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

            self.states = [None, None]
            if self.stateful:
                self.reset_states()

            self.kernel = self.add_weight(shape=(self.input_dim, self.units * 5),
                                          name='kernel',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units * 5),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint)

            if self.use_bias:
                if self.unit_forget_bias:
                    def bias_initializer(shape, *args, **kwargs):
                        return K.concatenate([
                            self.bias_initializer((self.units,), *args, **kwargs),
                            initializers.Ones()((self.units,), *args, **kwargs),
                            self.bias_initializer((self.units * 3,), *args, **kwargs),
                        ])
                else:
                    bias_initializer = self.bias_initializer
                self.bias = self.add_weight(shape=(self.units * 5,),
                                            name='bias',
                                            initializer=bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None

            self.kernel_i = self.kernel[:, :self.units]
            self.kernel_f = self.kernel[:, self.units: self.units * 2]
            self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
            self.kernel_o = self.kernel[:, self.units * 3: self.units * 4]
            self.kernel_m = self.kernel[:, self.units * 4:]

            self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
            self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
            self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
            self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3: self.units * 4]
            self.recurrent_kernel_m = self.recurrent_kernel[:, self.units * 4:]

            if self.use_bias:
                self.bias_i = self.bias[:self.units]
                self.bias_f = self.bias[self.units: self.units * 2]
                self.bias_c = self.bias[self.units * 2: self.units * 3]
                self.bias_o = self.bias[self.units * 3: self.units * 4]
                self.bias_m = self.bias[self.units * 4:]
            else:
                self.bias_i = None
                self.bias_f = None
                self.bias_c = None
                self.bias_o = None
                self.bias_m = None
            self.built = True

        def preprocess_input(self, inputs, training=None):
            if self.implementation == 0:
                input_shape = K.int_shape(inputs)
                input_dim = input_shape[2]
                timesteps = input_shape[1]

                x_i = _time_distributed_dense(inputs, self.kernel_i, self.bias_i,
                                              self.dropout, input_dim, self.units,
                                              timesteps, training=training)
                x_f = _time_distributed_dense(inputs, self.kernel_f, self.bias_f,
                                              self.dropout, input_dim, self.units,
                                              timesteps, training=training)
                x_c = _time_distributed_dense(inputs, self.kernel_c, self.bias_c,
                                              self.dropout, input_dim, self.units,
                                              timesteps, training=training)
                x_o = _time_distributed_dense(inputs, self.kernel_o, self.bias_o,
                                              self.dropout, input_dim, self.units,
                                              timesteps, training=training)
                x_m = _time_distributed_dense(inputs, self.kernel_m, self.bias_m,
                                              self.dropout, input_dim, self.units,
                                              timesteps, training=training)
                return K.concatenate([x_i, x_f, x_c, x_o, x_m], axis=2)
            else:
                return inputs

        def get_constants(self, inputs, training=None):
            constants = []
            if self.implementation != 0 and 0 < self.dropout < 1:
                input_shape = K.int_shape(inputs)
                input_dim = input_shape[-1]
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, int(input_dim)))

                def dropped_inputs():
                    return K.dropout(ones, self.dropout)

                dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(5)]
                constants.append(dp_mask)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(5)])

            if 0 < self.recurrent_dropout < 1:
                ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
                ones = K.tile(ones, (1, self.units))

                def dropped_inputs():
                    return K.dropout(ones, self.recurrent_dropout)

                rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                                ones,
                                                training=training) for _ in range(5)]
                constants.append(rec_dp_mask)
            else:
                constants.append([K.cast_to_floatx(1.) for _ in range(5)])
            return constants

        def step(self, inputs, states):
            h_tm1 = states[0]
            c_tm1 = states[1]
            dp_mask = states[2]
            rec_dp_mask = states[3]

            if self.implementation == 2:
                z = K.dot(inputs * dp_mask[0], self.kernel)
                z += z * K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)  # applies m instead of h_tm1 to z
                if self.use_bias:
                    z = K.bias_add(z, self.bias)

                z0 = z[:, :self.units]
                z1 = z[:, self.units: 2 * self.units]
                z2 = z[:, 2 * self.units: 3 * self.units]
                z3 = z[:, 3 * self.units: 4 * self.units]
                z4 = z[:, 4 * self.units:]  # just elementwise multiplication, no activation functions

                i = self.recurrent_activation(z0)
                f = self.recurrent_activation(z1)
                c = f * c_tm1 + i * self.activation(z2)
                o = self.recurrent_activation(z3)
            else:
                if self.implementation == 0:
                    x_i = inputs[:, :self.units]
                    x_f = inputs[:, self.units: 2 * self.units]
                    x_c = inputs[:, 2 * self.units: 3 * self.units]
                    x_o = inputs[:, 3 * self.units: 4 * self.units]
                    x_m = inputs[:, 4 * self.units:]
                elif self.implementation == 1:
                    x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
                    x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
                    x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
                    x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o
                    x_m = K.dot(inputs * dp_mask[4], self.kernel_m) + self.bias_m
                else:
                    raise ValueError('Unknown `implementation` mode.')

                m = x_m * K.dot(h_tm1 * rec_dp_mask[4], self.recurrent_kernel_m)  # elementwise multiplication m
                i = self.recurrent_activation(x_i + K.dot(m * rec_dp_mask[0], self.recurrent_kernel_i))
                f = self.recurrent_activation(x_f + K.dot(m * rec_dp_mask[1], self.recurrent_kernel_f))
                c = f * c_tm1 + i * self.activation(x_c + K.dot(m * rec_dp_mask[2], self.recurrent_kernel_c))
                o = self.recurrent_activation(x_o + K.dot(m * rec_dp_mask[3], self.recurrent_kernel_o))
            h = o * self.activation(c)
            if 0 < self.dropout + self.recurrent_dropout:
                h._uses_learning_phase = True
            return h, [h, c]

        def get_config(self):
            config = {'units': self.units,
                      'activation': activations.serialize(self.activation),
                      'recurrent_activation': activations.serialize(self.recurrent_activation),
                      'use_bias': self.use_bias,
                      'kernel_initializer': initializers.serialize(self.kernel_initializer),
                      'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                      'bias_initializer': initializers.serialize(self.bias_initializer),
                      'unit_forget_bias': self.unit_forget_bias,
                      'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                      'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                      'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                      'kernel_constraint': constraints.serialize(self.kernel_constraint),
                      'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                      'bias_constraint': constraints.serialize(self.bias_constraint),
                      'dropout': self.dropout,
                      'recurrent_dropout': self.recurrent_dropout}
            base_config = super(MultiplicativeLSTM, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))