#! /usr/bin/python3


import argparse
import os
import random

from misc import util


def split(filenames, k):
    """Split the list of filenames into "k" folds."""
    random.shuffle(filenames)
    fold_size = int(len(filenames) / k)
    return list(util.chunks(filenames, fold_size))


def save(dir_path, folds):
    """Save the folds in separate files."""
    if os.path.isfile(dir_path):
        raise ValueError('{} is a file'.format(dir_path))
    os.makedirs(dir_path, exist_ok=True)

    for i in range(len(folds)):
        test = folds[i]
        dev_idx = (i + 1) % len(folds)
        dev = folds[dev_idx]
        train = []
        for j in range(len(folds)):
            if j != i and j != dev_idx:
                train.extend(folds[j])

        fname = os.path.join(dir_path, 'test_{}.txt'.format(i + 1))
        with open(fname, 'w', encoding='utf-8') as hfile:
            hfile.write('\n'.join(sorted(test)))
        fname = os.path.join(dir_path, 'dev_{}.txt'.format(i + 1))
        with open(fname, 'w', encoding='utf-8') as hfile:
            hfile.write('\n'.join(sorted(dev)))
        fname = os.path.join(dir_path, 'train_{}.txt'.format(i + 1))
        with open(fname, 'w', encoding='utf-8') as hfile:
            hfile.write('\n'.join(sorted(train)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('DIR', help='A directory where the HTML files are located')
    ap.add_argument('-k', type=int, default='5', help='Number of folds')
    ap.add_argument('--save', default='.', help='Where to save the result')
    args = ap.parse_args()

    filenames = util.get_filenames(args.DIR, '.html')
    folds = split(filenames, args.k)
    save(args.save, folds)


if __name__ == '__main__':
    main()
