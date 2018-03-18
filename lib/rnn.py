
from __future__ import absolute_import
from keras import regularizers
from keras.regularizers import l1,l2,l1_l2
from keras.utils import multi_gpu_model
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Concatenate, Embedding, concatenate
from keras.layers import Dense, Bidirectional, LSTM, GRU, CuDNNLSTM, CuDNNGRU
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Cropping1D, Reshape, BatchNormalization, SpatialDropout1D
from lib.utils import GlobalZeroMaskedAveragePooling1D, GlobalSumPooling1D, Attention, MultiplicativeLSTM


def cnn2rnn(embedding_matrix, num_classes, max_seq_len,filter_sizes=[3, 4, 5], l2_weight_decay=0.0001, dropout_val=0.3,
        dense_dim=32, add_sigmoid=True, train_embeds=False, auxiliary=True, gpus=0, n_cnn_layers=1, pool='max',
        add_embeds=False, rnn_type = 'lstm'):
    if rnn_type == 'lstm':
        RNN = CuDNNLSTM if gpus > 0 else LSTM
    elif rnn_type == 'gru':
        RNN = CuDNNGRU if gpus > 0 else GRU
    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       trainable=train_embeds)(input_)
    convs = []
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=64, kernel_size=filter_size, padding='valid', activation='relu')(embeds)
        l_pool = MaxPooling1D(filter_size)(l_conv)
        convs.append(l_pool)
    cnn_feature_maps = Concatenate(axis=1)(convs)
    sentence_encoder = RNN(64, return_sequences=False)(cnn_feature_maps)
    fc_layer = Dense(128, activation="relu")(sentence_encoder)
    output = Dense(num_classes, activation="sigmoid")(fc_layer)
    model = Model(inputs=[input_], outputs=[output])
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model

def mulrnn(embedding_matrix, num_classes,  max_seq_len, l2_weight_decay=0.0001, rnn_dim=100, dropout_val=0.3, dense_dim=32, add_sigmoid=True, train_embeds=False, gpus=0, rnn_type='lstm', mask_zero=True, auxiliary=True, kernel_regularizer=None, recurrent_regularizer=None, activity_regularizer=None, dropout=0.2, recurrent_dropout=0.2):
    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       trainable=train_embeds)(input_)
    x = MultiplicativeLSTM(32)(embeds)
    output = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=input_, outputs=output)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model
