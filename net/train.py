#! /usr/bin/python3


import argparse
import os
import pickle
import csv
import math

import tensorflow as tf
from sklearn.utils import class_weight

from leaf_classifier import LeafClassifier


def get_dataset(dataset_file, batch_size, repeat=True):
    def _read_example(example):
        desc = {
            'doc_feature_list': tf.io.VarLenFeature(tf.int64),
            'doc_label_list': tf.io.VarLenFeature(tf.int64)
        }
        _, seq_features = tf.io.parse_single_sequence_example(example, sequence_features=desc)
        return tf.sparse.to_dense(seq_features['doc_feature_list']), \
               tf.sparse.to_dense(seq_features['doc_label_list'])

    buffer_size = 10 * batch_size
    dataset = tf.data.TFRecordDataset([dataset_file]) \
        .map(_read_example, num_parallel_calls=4) \
        .prefetch(buffer_size) \
        .padded_batch(
            batch_size=batch_size,
            padded_shapes=([None, None], [None, 1]),
            padding_values=(tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64))) \
        .shuffle(buffer_size=buffer_size)
    if repeat:
        return dataset.repeat()
    return dataset


def get_class_weights(train_set_file):
    y_train = []
    for _, y in get_dataset(train_set_file, 1, False):
        y_train.extend(y.numpy().flatten())
    return class_weight.compute_class_weight('balanced', [0, 1], y_train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DATA_DIR', help='Directory of files produced by the preprocessing script')
    ap.add_argument('-l', '--num_layers', type=int, default=2, help='The number of RNN layers')
    ap.add_argument('-u', '--hidden_units', type=int, default=256,
                    help='The number of hidden LSTM units')
    ap.add_argument('-d', '--dropout', type=float, default=0.5, help='The dropout percentage')
    ap.add_argument('-s', '--dense_size', type=int, default=256, help='Size of the dense layer')
    ap.add_argument('-e', '--epochs', type=int, default=20, help='The number of epochs')
    ap.add_argument('-b', '--batch_size', type=int, default=16, help='The batch size')
    ap.add_argument('--interval', type=int, default=5,
                    help='Calculate metrics and save the model after this many epochs')
    ap.add_argument('--working_dir', default='train', help='Where to save checkpoints and logs')
    args = ap.parse_args()

    info_file = os.path.join(args.DATA_DIR, 'info.pkl')
    with open(info_file, 'rb') as fp:
        info = pickle.load(fp)
        train_steps = math.ceil(info['num_train_examples'] / args.batch_size)

    train_set_file = os.path.join(args.DATA_DIR, 'train.tfrecords')
    train_dataset = get_dataset(train_set_file, args.batch_size)

    dev_set_file = os.path.join(args.DATA_DIR, 'dev.tfrecords')
    if os.path.isfile(dev_set_file):
        dev_dataset = get_dataset(dev_set_file, 1, repeat=False)
    else:
        dev_dataset = None
    
    test_set_file = os.path.join(args.DATA_DIR, 'test.tfrecords')
    if os.path.isfile(test_set_file):
        test_dataset = get_dataset(test_set_file, 1, repeat=False)
    else:
        test_dataset = None

    class_weights = get_class_weights(train_set_file)
    print('using class weights {}'.format(class_weights))

    kwargs = {'input_size': info['num_words'] + info['num_tags'],
              'hidden_size': args.hidden_units,
              'num_layers': args.num_layers,
              'dropout': args.dropout,
              'dense_size': args.dense_size}
    clf = LeafClassifier(**kwargs)

    ckpt_dir = os.path.join(args.working_dir, 'ckpt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    os.makedirs(ckpt_dir, exist_ok=True)

    params_file = os.path.join(args.working_dir, 'params.csv')
    print('writing {}...'.format(params_file))
    with open(params_file, 'w') as fp:
        writer = csv.writer(fp)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])

    clf.train(train_dataset, train_steps, args.epochs, log_file, ckpt_dir, class_weights,
              dev_dataset, info.get('num_dev_examples'),
              test_dataset, info.get('num_test_examples'),
              args.interval)


if __name__ == '__main__':
    main()
