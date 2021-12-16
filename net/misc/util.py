#! /usr/bin/python3


import random
import os


# everything that is a child of one of these tags is considered boilerplate and hence ignored by the
# classifier
TAGS_TO_IGNORE = {'head', 'iframe', 'script', 'meta', 'link', 'style', 'input', 'checkbox',
                  'button', 'noscript'}


def get_int_map(items, offset=0):
    """Return a dict that maps each unique item in "items" to a unique int ID."""
    # we sort the items to guarantee that we get the same mapping every time for a fixed input
    return {item: i + offset for i, item in enumerate(sorted(set(items)))}


def get_filenames(dir_path, filetype='.html'):
    """Return absolute paths to all files of a given type in a directory."""
    # join the dir path and the filename, then filter out directories
    all_files = filter(os.path.isfile, map(lambda x: os.path.join(dir_path, x), os.listdir(dir_path)))
    filtered_files = filter(lambda x: x.endswith(filetype), all_files)
    return list(filtered_files)
