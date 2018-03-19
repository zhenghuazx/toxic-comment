# toxic-comment
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zhenghuazx/toxic-comment/edit/master/LICENSE)

This project is launched for Kaggle Competition: [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) --- build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate.

------------------
## About the project.
This is my first kaggle game after graduation and made a lot of mistakes at the begining. After two-month soloing the game, I found it more important to learn than struggle in LB. 

------------------
## Models and LB Scores
### Recurrent Neural Nets
Refer to [here](https://github.com/zhenghuazx/toxic-comment/blob/master/lib/models.py) for code.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

### Other Models (excluded)
- **DCNN:** 
Refer to [Kalchbrenner et al (2014)](https://arxiv.org/abs/1404.2188). I am still working on testing the models. Please refer [bicepjai](https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/deep_models/paper_03_med_cnn/utils.py) for code.

- **CHAR-RNN:**
This model works on character level and utilize cnn to capture ngrams with rnn on the top of them. It takes 2 hours on AWS p2.xlarge for each epoch while gives ~ 0.045 validation logloss on 20% hold-out after 5 epochs. So I stopped the cross validation and excluded the model.
```python
def charrnn(char_num, num_classes, max_seq_len, filter_sizes=[3, 4, 5, 6, 7], rnn_dim = 128, num_filters=64, l2_weight_decay=0.0001, dropout_val=0.25, dense_dim=32, auxiliary = False, dropout=0.2, recurrent_dropout=0.2, add_sigmoid=True, train_embeds=False, gpus=0, add_embeds=True, rnn_type='gru'):
    if rnn_type == 'lstm':
        RNN = CuDNNLSTM if gpus > 0 else LSTM
    elif rnn_type == 'gru':
        RNN = CuDNNGRU if gpus > 0 else GRU
    else:
        RNN = CuDNNLSTM if gpus > 0 else LSTM
    input_ = Input(shape=(max_seq_len,))
    x = Embedding(char_num + 1, 300)(input_)
    x = SpatialDropout1D(dropout_val)(x)
    convs = []
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, padding='valid', activation='relu')(x)
        l_pool = MaxPooling1D(filter_size)(l_conv)
        convs.append(l_pool)
    x = Concatenate(axis=1)(convs)
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
```

- **Multiplicative LSTM for sequence modelling:**
Refer to [Krause et al (2016)](https://arxiv.org/pdf/1609.07959.pdf). Tried but gave it up.
```python
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
```
- **Very Deep CNN:**
This model works on both word and char level and use deeper architecture. It takes very fast on AWS p2.xlarge but only gives ~ 0.045 validation logloss on 20% hold-out. So I excluded the model.
```python
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
```
