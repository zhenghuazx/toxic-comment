from __future__ import absolute_import
from keras import regularizers
from keras.regularizers import l1,l2,l1_l2
from keras.utils import multi_gpu_model
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Concatenate, Embedding, concatenate
from keras.layers import Dense, Bidirectional, LSTM, GRU, CuDNNLSTM, CuDNNGRU
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Cropping1D, Reshape, BatchNormalization, SpatialDropout1D
from lib.utils import GlobalZeroMaskedAveragePooling1D, GlobalSumPooling1D, Attention

def rcnn(embedding_matrix, num_classes, max_seq_len, rnn_dim = 128, num_filters=64, l2_weight_decay=0.0001, dropout_val=0.25, dense_dim=32, auxiliary = False, dropout=0.2, recurrent_dropout=0.2, add_sigmoid=True, train_embeds=False, gpus=0, add_embeds=True, rnn_type='gru'):
    if rnn_type == 'lstm':
        RNN = CuDNNLSTM if gpus > 0 else LSTM
    elif rnn_type == 'gru':
        RNN = CuDNNGRU if gpus > 0 else GRU
    else:
        RNN = CuDNNLSTM if gpus > 0 else LSTM

    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       trainable=train_embeds)(input_)
    x = SpatialDropout1D(dropout_val)(embeds)
    x = Bidirectional(RNN(rnn_dim, return_sequences = True, dropout=dropout, recurrent_dropout=recurrent_dropout))(x)
    x = Conv1D(num_filters, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    if auxiliary:
        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = Concatenate()([x, auxiliary_input])

    x = Dense(num_classes, activation = "sigmoid")(x)
    if auxiliary:
        model = Model(inputs=[input_, auxiliary_input], outputs=x)
    else:
        model = Model(inputs=input_, outputs=x)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model

def charrnn(embedding_matrix, num_classes, max_seq_len, rnn_dim = 128, num_filters=64, auxiliary = True, l2_weight_decay=0.0001, dropout_val=0.25, dropout= 0.2,recurrent_dropout=0.2, dense_dim=32, add_sigmoid=True, train_embeds=False, gpus=0, n_cnn_layers=1, pool='max', add_embeds=True, rnn_type='lstm'):
    if rnn_type == 'lstm':
        RNN = CuDNNLSTM if gpus > 0 else LSTM
    elif rnn_type == 'gru':
        RNN = CuDNNGRU if gpus > 0 else GRU
    if pool == 'max':
        Pooling = MaxPooling1D
    elif pool == 'avg':
        Pooling = AveragePooling1D
    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       trainable=train_embeds)(input_)
    x = SpatialDropout1D(dropout_val)(embeds)
    w = Conv1D(filters=16,kernel_size=7, padding='same', activation='relu')(x)
    w = Pooling(pool_size=2, strides=1)(w)
    w = BatchNormalization()(w)
    w = Conv1D(filters=32,kernel_size=5, padding='same', activation='relu')(w)
    w = Pooling(pool_size=2, strides=1)(w)
    w = BatchNormalization()(w)
    w = Conv1D(filters=32,kernel_size=3, padding='same', activation='relu')(w)
    w = Pooling(pool_size=2, strides=1)(w)
    w = BatchNormalization()(w)
    w1 = GlobalMaxPooling1D()(w)
    w2 = GlobalAveragePooling1D()(w)
    x = Bidirectional(RNN(rnn_dim, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))(x)
    z1 = GlobalMaxPooling1D()(x)
    z2 = GlobalAveragePooling1D()(x)
    x = Concatenate()([z1, z2, w1, w2])
    x = Dropout(dropout_val)(x)
    if auxiliary:
        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = Concatenate()([x, auxiliary_input])

    x = Dense(dense_dim, activation="relu", kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    if add_sigmoid:
        x = Dense(num_classes, activation="sigmoid")(x)
    if auxiliary:
        model = Model(inputs=[input_, auxiliary_input], outputs=x)
    else:
        model = Model(inputs=input_, outputs=x)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


def cnn(embedding_matrix, num_classes, max_seq_len, num_filters=64, l2_weight_decay=0.0001, dropout_val=0.5,
        dense_dim=32, add_sigmoid=True, train_embeds=False, auxiliary=True, gpus=0, n_cnn_layers=1, pool='max',
        add_embeds=False):
    if pool == 'max':
        Pooling = MaxPooling1D
        GlobalPooling = GlobalMaxPooling1D
    elif pool == 'avg':
        Pooling = AveragePooling1D
        GlobalPooling = GlobalAveragePooling1D
    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       trainable=train_embeds)(input_)
    # x = embeds
    x = SpatialDropout1D(0.2)(embeds)

    x0 = Conv1D(num_filters, 1, activation='relu', padding='same')(x)
    x0 = Pooling(3)(x0)

    x1 = Conv1D(num_filters, 1, activation='relu', padding='same')(x)
    x1 = Conv1D(num_filters, 3, activation='relu', padding='same')(x1)
    x1 = Conv1D(num_filters, 5, activation='relu', padding='same')(x1)
    x1 = Conv1D(num_filters, 7, activation='relu', padding='same')(x1)
    x1 = Pooling(3)(x1)
    x2 = Conv1D(num_filters, 3, activation='relu', padding='same')(x)
    x2 = Conv1D(num_filters, 5, activation='relu', padding='same')(x2)
    x2 = Pooling(3)(x2)

    x3 = Conv1D(num_filters, 4, activation='relu', padding='same')(x)
    x3 = Conv1D(num_filters, 7, activation='relu', padding='same')(x3)
    x3 = Pooling(3)(x3)

    x4 = Conv1D(num_filters, 5, activation='relu', padding='same')(x)
    x4 = Conv1D(num_filters, 3, activation='relu', padding='same')(x4)
    x4 = Pooling(3)(x4)
    x = Concatenate()([x0, x1, x2, x3, x4])
    for i in range(n_cnn_layers - 1):
        x = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
        x = Pooling(3)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_val)(x)

    x_0 = Conv1D(num_filters, 1, activation='relu', padding='same')(x)
    x_0 = Pooling(3)(x_0)

    x_1 = Conv1D(num_filters, 1, activation='relu', padding='same')(x)
    x_1 = Conv1D(num_filters, 3, activation='relu', padding='same')(x_1)
    x_1 = Conv1D(num_filters, 5, activation='relu', padding='same')(x_1)
    x_1 = Conv1D(num_filters, 7, activation='relu', padding='same')(x_1)
    x_1 = Pooling(3)(x_1)
    x_2 = Conv1D(num_filters, 3, activation='relu', padding='same')(x)
    x_2 = Conv1D(num_filters, 5, activation='relu', padding='same')(x_2)
    x_2 = Conv1D(num_filters, 7, activation='relu', padding='same')(x_2)
    x_2 = Pooling(3)(x_2)

    x_3 = Conv1D(num_filters, 4, activation='relu', padding='same')(x)
    x_3 = Conv1D(num_filters, 7, activation='relu', padding='same')(x_3)
    x_3 = Pooling(3)(x_3)

    x_4 = Conv1D(num_filters, 5, activation='relu', padding='same')(x)
    x_4 = Conv1D(num_filters, 3, activation='relu', padding='same')(x_4)
    x_4 = Pooling(3)(x_4)
    x = Concatenate()([x_0, x_1, x_2, x_3, x_4])

    for i in range(n_cnn_layers - 1):
        x = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
        x = Pooling()(x)

    z_0 = Conv1D(num_filters, 1, activation='relu', padding='same')(x)
    z_0 = Pooling(3)(z_0)

    z_1 = Conv1D(num_filters, 1, activation='relu', padding='same')(x)
    z_1 = Conv1D(num_filters, 3, activation='relu', padding='same')(z_1)
    z_1 = Conv1D(num_filters, 5, activation='relu', padding='same')(z_1)
    z_1 = Conv1D(num_filters, 7, activation='relu', padding='same')(z_1)
    z_1 = Pooling(3)(z_1)
    z_2 = Conv1D(num_filters, 3, activation='relu', padding='same')(x)
    z_2 = Conv1D(num_filters, 5, activation='relu', padding='same')(z_2)
    z_2 = Conv1D(num_filters, 7, activation='relu', padding='same')(z_2)
    z_2 = Pooling(3)(z_2)

    z_3 = Conv1D(num_filters, 4, activation='relu', padding='same')(x)
    z_3 = Conv1D(num_filters, 7, activation='relu', padding='same')(z_3)
    z_3 = Pooling(3)(z_3)

    z_4 = Conv1D(num_filters, 5, activation='relu', padding='same')(x)
    z_4 = Conv1D(num_filters, 3, activation='relu', padding='same')(z_4)
    z_4 = Pooling(3)(z_4)
    x = Concatenate()([z_0, z_1, z_2, z_3, z_4])

    for i in range(n_cnn_layers - 1):
        x = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
        x = Pooling(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_val)(x)

    w_2 = Conv1D(num_filters, 3, activation='relu', padding='same')(x)
    w_2 = Conv1D(num_filters, 5, activation='relu', padding='same')(w_2)
    w_2 = Conv1D(num_filters, 7, activation='relu', padding='same')(w_2)

    w_3 = Conv1D(num_filters, 3, activation='relu', padding='same')(x)
    w_3 = Conv1D(num_filters, 3, activation='relu', padding='same')(w_3)

    w_4 = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
    w_4 = Conv1D(num_filters, 5, activation='relu', padding='same')(w_4)
    w_4 = Conv1D(num_filters, 3, activation='relu', padding='same')(w_4)
    x = Concatenate()([w_2, w_3, w_4])

    x = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
    x = GlobalPooling()(x)
    if add_embeds:
        x1 = Conv1D(num_filters, 7, activation='relu', padding='same')(embeds)
        x1 = GlobalPooling()(x1)
        x = Concatenate()([x, x1])
    x = Dropout(dropout_val)(x)
    if auxiliary:
        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = Concatenate()([x, auxiliary_input])
    x = BatchNormalization()(x)
    x = Dropout(dropout_val)(x)
    x = Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    if add_sigmoid:
        x = Dense(num_classes, activation='sigmoid')(x)
    if auxiliary:
        model = Model(inputs=[input_, auxiliary_input], outputs=x)
    else:
        model = Model(inputs=input_, outputs=x)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


def _get_regularizer(regularizer_name, weight):
    if regularizer_name is None:
        return None
    if regularizer_name == 'l1':
        return l1(weight)
    if regularizer_name == 'l2':
        return l2(weight)
    if regularizer_name == 'l1_l2':
        return l1_l2(weight)
    return None

def rnn(embedding_matrix, num_classes,  max_seq_len, l2_weight_decay=0.0001, rnn_dim=100, dropout_val=0.3, dense_dim=32, n_branches=0, n_rnn_layers=1, n_dense_layers=1, add_sigmoid=True, train_embeds=False, gpus=0, rnn_type='lstm', mask_zero=True, auxiliary=True, kernel_regularizer=None, recurrent_regularizer=None, activity_regularizer=None, dropout=0.2, recurrent_dropout=0.2):
    rnn_regularizers = {'kernel_regularizer': _get_regularizer(kernel_regularizer, l2_weight_decay),
                        'recurrent_regularizer': _get_regularizer(recurrent_regularizer, l2_weight_decay),
                        'activity_regularizer': _get_regularizer(activity_regularizer, l2_weight_decay)}
    if gpus == 0:
        rnn_regularizers['dropout'] = dropout
        rnn_regularizers['recurrent_dropout'] = recurrent_dropout
    if rnn_type == 'lstm':
        RNN = CuDNNLSTM if gpus > 0 else LSTM
    elif rnn_type == 'gru':
        RNN = CuDNNGRU if gpus > 0 else GRU
    mask_zero = mask_zero and gpus == 0

    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       mask_zero=mask_zero,
                       trainable=train_embeds)(input_)
    embeds = SpatialDropout1D(0.2)(embeds)
    branches = []
    for _ in range(n_branches):
        branch = Bidirectional(RNN(rnn_dim, return_sequences=True, **rnn_regularizers))(embeds)
        branch = Dropout(dropout_val)(branch)
        branches.append(branch)
    if n_branches > 1:
        x = Concatenate()(branches)
    elif n_branches == 1:
        x = branches[0]
    else:
        x = embeds
    '''
    for _ in range(n_rnn_layers):
        x = Bidirectional(RNN(rnn_dim, return_sequences=True, **rnn_regularizers))(x)
        z0 = Cropping1D(cropping=(-1, 0))(x)
        z1 = GlobalMaxPooling1D()(x)
        z2 = GlobalAveragePooling1D()(x)
        x = Concatenate()([z0, z1, z2])
        x = Dropout(dropout_val)(x)
    '''
    x = Bidirectional(RNN(rnn_dim, return_sequences=True, **rnn_regularizers))(x)
    z0 = Cropping1D(cropping=(max_seq_len - 1, 0))(x)
    z0 = Reshape([rnn_dim * 2])(z0)
    z1 = GlobalMaxPooling1D()(x)
    z2 = GlobalAveragePooling1D()(x)
    z3 = Attention(max_seq_len)(x)
    x = Concatenate()([z0, z1, z2, z3])
    x = Dropout(dropout_val)(x)
    if auxiliary:
        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = Concatenate()([x, auxiliary_input])
    for _ in range(n_dense_layers-1):
        x = Dense(dense_dim, activation="relu")(x)
        x = Dropout(dropout_val)(x)
    x = Dropout(dropout_val)(x)
    if add_sigmoid:
        x = Dense(num_classes, activation="sigmoid")(x)
    if auxiliary:
        model = Model(inputs=[input_, auxiliary_input], outputs=x)
    else:
        model = Model(inputs=input_, outputs=x)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model



def dense(embedding_matrix, num_classes, max_seq_len, dense_dim=100, n_layers=10, concat=0, dropout_val=0.5, l2_weight_decay=0.0001, pool='max', add_sigmoid=True, train_embeds=False, gpus=0):
    GlobalPool = {
        'avg': GlobalZeroMaskedAveragePooling1D,
        'max': GlobalMaxPooling1D,
        'sum': GlobalSumPooling1D
    }

    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       trainable=train_embeds)(input_)

    if isinstance(pool, list) and len(pool) > 1:
        to_concat = []
        for p in pool:
            to_concat.append(GlobalPool[p]()(embeds))
        x = Concatenate()(to_concat)
    else:
        x = GlobalPool[pool]()(embeds)
    prev = []
    for i in range(n_layers):
        if concat > 0:
            if i == 0:
                prev.append(x)
                continue
            elif i % concat == 0:
                prev.append(x)
                x = Concatenate(axis=-1)(prev)
        x = Dense(dense_dim, activation="relu")(x)
        x = Dropout(dropout_val)(x)
    output_ = Dense(dense_dim, activation="relu", kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    if add_sigmoid:
        output_ = Dense(num_classes, activation="sigmoid")(output_)
    model = Model(inputs=input_, outputs=output_)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


def capsule(embedding_matrix, num_classes,  max_seq_len,Dim_capsule = 16, Num_capsule = 10, Routings=5, l2_weight_decay=0.0001, rnn_dim=128, dropout_val=0.3, dense_dim=32, n_branches=0, n_rnn_layers=1, n_dense_layers=1, add_sigmoid=True, train_embeds=False, gpus=0, rnn_type='lstm', mask_zero=True, auxiliary=True, kernel_regularizer=None, recurrent_regularizer=None, activity_regularizer=None, dropout=0.2, recurrent_dropout=0.2):
    if rnn_type == 'lstm':
        RNN = CuDNNLSTM if gpus > 0 else LSTM
    elif rnn_type == 'gru':
        RNN = CuDNNGRU if gpus > 0 else GRU
    else:
        RNN = CuDNNLSTM if gpus > 0 else LSTM
    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       mask_zero=mask_zero,
                       trainable=train_embeds)(input_)
    embeds = SpatialDropout1D(0.25)(embeds)
    x = Bidirectional(RNN(rnn_dim, return_sequences=True, activation='relu', dropout=dropout, recurrent_dropout=recurrent_dropout))(embeds)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    capsule = Flatten()(capsule)
    x = Dropout(dropout_val)(capsule)
    if auxiliary:
        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = Concatenate()([x, auxiliary_input])
    x = Dense(num_classes, activation="sigmoid")(x)
    if auxiliary:
        model = Model(inputs=[input_, auxiliary_input], outputs=x)
    else:
        model = Model(inputs=input_, outputs=x)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model

def save_predictions(df, predictions, target_labels, additional_name=None):
    for i, label in enumerate(target_labels):
        if additional_name is not None:
            label = '{}_{}'.format(additional_name, label)
        df[label] = predictions[:, i]

