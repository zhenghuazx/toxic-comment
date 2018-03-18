'''
 # Created by hua.zheng on 3/8/18.
'''

import tensorflow.contrib.keras as keras
from keras.engine import Layer, InputSpec, InputLayer
from keras.engine import Layer, InputSpec
import tensorflow as tf
from keras.models import Model, Sequential
from keras.utils import multi_gpu_model
from keras.layers import Dropout, Embedding, concatenate
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, ZeroPadding1D,GlobalMaxPooling1D, GlobalAveragePooling1D, \
    AveragePooling1D, SpatialDropout1D
from keras.layers import Dense, Input, Flatten, BatchNormalization
from keras.layers import Concatenate, Dot, Merge, Multiply, RepeatVector
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import SimpleRNN, LSTM, GRU, Lambda, Permute

from keras.layers.core import Reshape, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.constraints import maxnorm
from keras.regularizers import l2

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, axis=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
        assert axis in [1,2],  'expected dimensions (samples, filters, convolved_values),\
                   cannot fold along samples dimension or axis not in list [1,2]'
        self.axis = axis
        # need to switch the axis with the last elemnet
        # to perform transpose for tok k elements since top_k works in last axis
        self.transpose_perm = [0,1,2] #default
        self.transpose_perm[self.axis] = 2
        self.transpose_perm[2] = self.axis
    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        input_shape_list[self.axis] = self.k
        return tuple(input_shape_list)
    def call(self, x):
        # swap sequence dimension to get top k elements along axis=1
        transposed_for_topk = tf.transpose(x, perm=self.transpose_perm)
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(transposed_for_topk, k=self.k, sorted=True, name=None)[0]
        # return back to normal dimension but now sequence dimension has only k elements
        # performing another transpose will get the tensor back to its original shape
        # but will have k as its axis_1 size
        transposed_back = tf.transpose(top_k, perm=self.transpose_perm)
        return transposed_back

class Folding(Layer):
    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], int(input_shape[2]/2))
    def call(self, x):
        input_shape = x.get_shape().as_list()
        # split the tensor along dimension 2 into dimension_axis_size/2
        # which will give us 2 tensors
        splits = tf.split(x, num_or_size_splits=int(input_shape[2]/2), axis=2)
        # reduce sums of the pair of rows we have split onto
        reduce_sums = [tf.reduce_sum(split, axis=2) for split in splits]
        # stack them up along the same axis we have reduced
        row_reduced = tf.stack(reduce_sums, axis=2)
        return row_reduced


def vdnn(embedding_matrix, num_classes, max_seq_len, num_filters=2, filter_sizes=[64, 128, 256, 512], l2_weight_decay=0.0001, dropout_val=0.5,
        dense_dim=32, add_sigmoid=True, train_embeds=False, auxiliary=True, gpus=0, n_cnn_layers=1, pool='max',
        add_embeds=False):
    #input_ = Input(shape=(max_seq_len,))
    model = Sequential([
        Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  input_length=max_seq_len,
                  trainable=train_embeds),
        Conv1D(embedding_matrix.shape[1], 3, padding="valid")
    ])
    # 4 pairs of convolution blocks followed by pooling
    for filter_size in filter_sizes:
        # each iteration is a convolution block
        for cb_i in range(num_filters):
            model.add(Conv1D(filter_size, 3, padding="same"))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(Conv1D(filter_size, 3, padding="same"))
            model.add(BatchNormalization())
            model.add(Activation("relu"))
        model.add(MaxPooling1D(pool_size=2, strides=3))
    # model.add(KMaxPooling(k=2))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(256, activation="relu"))
    if add_sigmoid:
        model.add(Dense(num_classes, activation='sigmoid'))
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


def mvcnn(embedding_matrix1, embedding_matrix2, num_classes, max_seq_len, num_filters=2, filter_sizes=[3, 5], l2_weight_decay=0.0001, dropout_val=0.5,
        dense_dim=32, add_sigmoid=True, train_embeds=False, auxiliary=True, gpus=0):
    text_seq_input = Input(shape=(max_seq_len,), dtype='int32')
    text_embedding1 = Embedding(embedding_matrix1.shape[0],
                  embedding_matrix1.shape[1],
                  weights=[embedding_matrix1],
                  input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    text_embedding2 = Embedding(embedding_matrix2.shape[0],
                  embedding_matrix2.shape[1],
                  weights=[embedding_matrix2],
                  input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    k_top = 4
    layer_1 = []
    for text_embedding in [text_embedding1, text_embedding2]:
        conv_pools = []
        for filter_size in filter_sizes:
            l_zero = ZeroPadding1D((filter_size - 1, filter_size - 1))(text_embedding)
            l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = KMaxPooling(k=30, axis=1)(l_conv)
            conv_pools.append((filter_size, l_pool))
        layer_1.append(conv_pools)
    last_layer = []
    for layer in layer_1:  # no of embeddings used
        for (filter_size, input_feature_maps) in layer:
            l_zero = ZeroPadding1D((filter_size - 1, filter_size - 1))(input_feature_maps)
            l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = KMaxPooling(k=k_top, axis=1)(l_conv)
            last_layer.append(l_pool)
    l_merge = Concatenate(axis=1)(last_layer)
    l_flat = Flatten()(l_merge)
    l_dense = Dense(128, activation='relu')(l_flat)
    if auxiliary:
        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = Concatenate()([l_dense, auxiliary_input])
    l_out = Dense(num_classes, activation='sigmoid')(x)
    if auxiliary:
        model = Model(inputs=[text_seq_input, auxiliary_input], outputs=l_out)
    else:
        model = Model(inputs=text_seq_input, outputs=l_out)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


def mgcnn(embedding_matrix1, embedding_matrix2,embedding_matrix3, num_classes, max_seq_len, num_filters=2, filter_sizes=[3,5], l2_weight_decay=0.0001, dropout_val=0.5,
        dense_dim=32, add_sigmoid=True, train_embeds=False, auxiliary=True, gpus=0):
    text_seq_input = Input(shape=(max_seq_len,), dtype='int32')
    text_embedding1 = Embedding(embedding_matrix1.shape[0],
                  embedding_matrix1.shape[1],
                  weights=[embedding_matrix1],
                  input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    text_embedding2 = Embedding(embedding_matrix2.shape[0],
                  embedding_matrix2.shape[1],
                  weights=[embedding_matrix2],
                  input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    text_embedding3 = Embedding(embedding_matrix3.shape[0],
                  embedding_matrix3.shape[1],
                  weights=[embedding_matrix3],
                  input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    k_top = 4
    conv_pools = []
    for text_embedding in [text_embedding1, text_embedding2, text_embedding3]:
        for filter_size in filter_sizes:
            l_zero = ZeroPadding1D((filter_size - 1, filter_size - 1))(text_embedding)
            l_conv = Conv1D(filters=64, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = GlobalMaxPooling1D()(l_conv)
            conv_pools.append(l_pool)
    l_merge = Concatenate(axis=1)(conv_pools)
    l_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(l_merge)
    l_out = Dense(num_classes, activation='sigmoid')(l_dense)
    model = Model(inputs=text_seq_input, outputs=l_out)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model

# can be an alternative to mvcnn
def mvcnn3(embedding_matrix1, embedding_matrix2, embedding_matrix3, num_classes, max_seq_len, num_filters=2, filter_sizes=[3, 5], l2_weight_decay=0.0001, dropout_val=0.5,
        dense_dim=32, add_sigmoid=True, train_embeds=False, auxiliary=True, gpus=0):
    text_seq_input = Input(shape=(max_seq_len,), dtype='int32')
    text_embedding1 = Embedding(embedding_matrix1.shape[0],
                  embedding_matrix1.shape[1],
                  weights=[embedding_matrix1],
                  input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    text_embedding2 = Embedding(embedding_matrix2.shape[0],
                  embedding_matrix2.shape[1],
                  weights=[embedding_matrix2],
                  input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    text_embedding3 = Embedding(embedding_matrix3.shape[0],
                  embedding_matrix3.shape[1],
                  weights=[embedding_matrix3],
                  input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    k_top = 4
    layer_1 = []
    for text_embedding in [text_embedding1, text_embedding2, text_embedding3]:
        conv_pools = []
        for filter_size in filter_sizes:
            l_zero = ZeroPadding1D((filter_size - 1, filter_size - 1))(text_embedding)
            l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = KMaxPooling(k=30, axis=1)(l_conv)
            conv_pools.append((filter_size, l_pool))
        layer_1.append(conv_pools)
    last_layer = []
    for layer in layer_1:  # no of embeddings used
        for (filter_size, input_feature_maps) in layer:
            l_zero = ZeroPadding1D((filter_size - 1, filter_size - 1))(input_feature_maps)
            l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
            l_pool = KMaxPooling(k=k_top, axis=1)(l_conv)
            last_layer.append(l_pool)
    l_merge = Concatenate(axis=1)(last_layer)
    l_flat = Flatten()(l_merge)
    l_dense = Dense(128, activation='relu')(l_flat)
    l_out = Dense(num_classes, activation='sigmoid')(l_dense)
    model = Model(inputs=[text_seq_input], outputs=l_out)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


def cnn2d(embedding_matrix, num_classes, max_seq_len, num_filters=64, filter_sizes=[1,2,3,5], l2_weight_decay=0.0001, dropout_val=0.25,
        dense_dim=32, add_sigmoid=True, train_embeds=False, auxiliary=True, gpus=0, n_cnn_layers=1, pool='max',
        add_embeds=False):
    text_seq_input = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                  embedding_matrix.shape[1],
                  weights=[embedding_matrix],
                  #input_length=max_seq_len,
                  trainable=train_embeds)(text_seq_input)
    x = SpatialDropout1D(0.3)(embeds)
    x = Reshape((max_seq_len, embedding_matrix.shape[1], 1))(x)
    pooled = []
    for i in filter_sizes:
        conv = Conv2D(num_filters, kernel_size=(i, embedding_matrix.shape[1]), kernel_initializer='normal', activation='elu')(x)
        maxpool = MaxPooling2D(pool_size=(max_seq_len - i + 1, 1))(conv)
        #avepool = AveragePooling2D(pool_size=(max_seq_len - i + 1, 1))(conv)
        #globalmax = GlobalMaxPooling2D()(conv)
        pooled.append(maxpool)
    z = Concatenate(axis=1)(pooled)
    z = Flatten()(z)
    z = BatchNormalization()(z)
    z = Dropout(dropout_val)(z)
    if auxiliary:
        auxiliary_input = Input(shape=(5,), name='aux_input')
        z = Concatenate()([z, auxiliary_input])
    output = Dense(num_classes, activation="sigmoid")(z)
    if auxiliary:
        model = Model(inputs=[text_seq_input, auxiliary_input], outputs=output)
    else:
        model = Model(inputs=text_seq_input, outputs=output)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model





