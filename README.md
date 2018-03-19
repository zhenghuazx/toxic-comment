# toxic-comment
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zhenghuazx/toxic-comment/edit/master/LICENSE)

This project is launched for Kaggle Competition: [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) --- build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate.

------------------
## About the project.
This is my first kaggle game after graduation and made a lot of mistakes at the begining. After two-month soloing the game, I found it more important to learn than struggle in LB. 

------------------
## Models and LB Scores

### other models
- DCNN [Kalchbrenner et al (2014)](https://arxiv.org/abs/1404.2188)
I am still working on testing the models. Please refer [bicepjai](https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/deep_models/paper_03_med_cnn/utils.py) for code.

#### charrnn 
- This model works on character level and utilize cnn to capture ngrams and with rnn on the top of them. It takes 2 hours on AWS p2.xlarge for each epoch while gives ~ 0.045 validation logloss on 20% hold-out. So I stopped the cross validation and exclude the model.
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
