from keras import optimizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, ModelCheckpoint
import os
from math import floor
import numpy as np
from keras import backend as K
from sklearn.metrics import roc_auc_score
import logging

def step_decay(initial_lr, lr_drop_koef, epochs_to_drop, epoch):
    return initial_lr * pow(lr_drop_koef, floor((1 + epoch) / epochs_to_drop))


class LossHistory(Callback):
    def __init__(self, initial_lr, lr_drop_koef, epochs_to_drop):
        self.initial_lr = initial_lr
        self.lr_drop_koef = lr_drop_koef
        self.epochs_to_drop = epochs_to_drop

    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(self.initial_lr, self.lr_drop_koef, self.epochs_to_drop, len(self.losses)))


class roc_callback(Callback):
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, roc_val))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def define_callbacks(early_stopping_delta, x_val, y_val, early_stopping_epochs=5, use_lr_strategy=True, initial_lr=0.005, lr_drop_koef=0.66, epochs_to_drop=6, model_checkpoint_dir=None):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_delta, patience=early_stopping_epochs, verbose=1)
    callbacks_list = [early_stopping]
    if model_checkpoint_dir is not None:
        if not os.path.exists(model_checkpoint_dir):
            os.makedirs(model_checkpoint_dir)
        checkpoint_path = os.path.join(model_checkpoint_dir, 'weights.h5')
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
        callbacks_list.append(model_checkpoint)
    if use_lr_strategy:
        epochs_to_drop = float(epochs_to_drop)
        loss_history = LossHistory(initial_lr, lr_drop_koef, epochs_to_drop)
        lrate = LearningRateScheduler(lambda epoch: step_decay(initial_lr, lr_drop_koef, epochs_to_drop, epoch))
        callbacks_list.append(loss_history)
        callbacks_list.append(lrate)
    callbacks_list.append(roc_callback(x_val, y_val))
    return callbacks_list


def train(x_train, y_train, x_val, y_val, model, batch_size, num_epochs, learning_rate=0.001, early_stopping_delta=0.0, early_stopping_epochs=5, use_lr_strategy=True, lr_drop_koef=0.66, epochs_to_drop=5, model_checkpoint_dir=None, logger=None):
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    if logger is not None:
        model.summary(print_fn=lambda line: logger.debug(line))
    else:
        model.summary()

    callbacks_list = define_callbacks(early_stopping_delta,
                                    x_val,
                                    y_val,
                                    early_stopping_epochs,
                                    use_lr_strategy=use_lr_strategy,
                                    initial_lr=learning_rate,
                                    lr_drop_koef=lr_drop_koef,
                                    epochs_to_drop=epochs_to_drop,
                                    model_checkpoint_dir=model_checkpoint_dir)

    hist = model.fit(x_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=num_epochs,
                     callbacks=callbacks_list,
                     #validation_split=0.1,
                     validation_data=(x_val, y_val),
                     shuffle=True,
                     verbose=1)
    return hist

def continue_train(x_train, y_train, x_val, y_val, model, batch_size, num_epochs, learning_rate_decay, learning_rate=0.001, early_stopping_delta=0.0, early_stopping_iters=5, use_lr_strategy=True, lr_drop_koef=0.66, epochs_to_drop=5):
    callbacks_list = define_callbacks(early_stopping_delta,
                                    x_val,
                                    y_val,
                                    early_stopping_iters,
                                    use_lr_strategy=use_lr_strategy,
                                    initial_lr=learning_rate,
                                    lr_drop_koef=lr_drop_koef,
                                    epochs_to_drop=epochs_to_drop)

    hist = model.fit(x_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=num_epochs,
                     callbacks=callbacks_list,
                     #validation_split=0.1,
                     validation_data=(x_val, y_val),
                     shuffle=True,
                     verbose=1)
    return hist
