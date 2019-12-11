#! /usr/bin/python3


import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


class Metrics(tf.keras.callbacks.Callback):
    """Calculate metrics for a dev-/testset and add them to the logs."""
    def __init__(self, clf, data, steps, interval, prefix=''):
        self.clf = clf
        self.data = data
        self.steps = steps
        self.interval = interval
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.interval == 0:
            y_true, y_pred = self.clf.eval(self.data, self.steps, desc=self.prefix)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        else:
            p, r, f, s = np.nan, np.nan, np.nan, np.nan
        logs_new = {'{}_precision'.format(self.prefix): p,
                    '{}_recall'.format(self.prefix): r,
                    '{}_f1'.format(self.prefix): f,
                    '{}_support'.format(self.prefix): s}
        logs.update(logs_new)


class Saver(tf.keras.callbacks.Callback):
    """Save the model."""
    def __init__(self, path, interval):
        self.path = path
        self.interval = interval

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.interval == 0:
            file_name = os.path.join(self.path, 'model.{:03d}.h5'.format(epoch))
            self.model.save(file_name)


# pylint: disable=E1101
class LeafClassifier(object):
    """This classifier assigns labels to sequences based on words and HTML tags."""
    def __init__(self, input_size, num_layers, hidden_size, dropout, dense_size):
        """Construct the network."""
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dense_size = dense_size
        self.model = self._get_model()

    def _get_model(self):
        """Return a keras model."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(None, self.input_size)))
        model.add(tf.keras.layers.Dense(self.dense_size, activation='relu'))
        model.add(tf.keras.layers.Masking(mask_value=0))
        for _ in range(self.num_layers):
            lstm = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)
            model.add(tf.keras.layers.Bidirectional(lstm))
        model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, train_dataset, train_steps, epochs, log_file, ckpt, class_weight=(1, 1),
              dev_dataset=None, dev_steps=None, test_dataset=None, test_steps=None, interval=1):
        """Train a number of input sequences."""
        callbacks = [Saver(ckpt, interval)]
        if dev_dataset is not None:
            callbacks.append(Metrics(self, dev_dataset, dev_steps, interval, 'dev'))
        if test_dataset is not None:
            callbacks.append(Metrics(self, test_dataset, test_steps, interval, 'test'))
        callbacks.append(tf.keras.callbacks.CSVLogger(log_file))

        self.model.fit(train_dataset, steps_per_epoch=train_steps, epochs=epochs,
                       callbacks=callbacks, class_weight=class_weight)

    def eval(self, dataset, steps, desc=None):
        """Evaluate the model on the test data and return the metrics."""
        y_true, y_pred = [], []
        for b_x, b_y in tqdm(dataset, total=steps, desc=desc):
            # somehow this cast is necessary
            b_x = tf.dtypes.cast(b_x, 'float32')

            y_true.extend(b_y.numpy().flatten())
            y_pred.extend(np.around(self.model.predict_on_batch(b_x)).flatten())
        return y_true, y_pred
