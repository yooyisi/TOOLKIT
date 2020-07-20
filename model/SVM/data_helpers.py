# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import re
from io import open

import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\u4e00-\u9fa5？，！。；]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def one_hot(labels, use_fixed_map=False):
    res = []
    if use_fixed_map:
        with open('./model/label.dict', 'r', encoding='utf-8') as infile:
            label_dict = json.load(infile)
        str_num_dict = {v:k for k,v in label_dict.items()}
        for label in labels:
            indix = int(str_num_dict.get(label))
            empty_vec = [0] * len(label_dict)
            empty_vec[indix] = 1
            res.append(empty_vec)
        res = np.array(res)
        return res

    label_set = list(set(labels))
    label_dict = {}
    for lal in label_set:
        indix = label_set.index(lal)
        label_dict[indix] = lal
    with open('./model/label.dict', 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(label_dict, ensure_ascii=False))

    for label in labels:
        indix = label_set.index(label)
        empty_vec = [0] * len(label_set)
        empty_vec[indix] = 1
        res.append(empty_vec)
    res = np.array(res)
    return res


def load_data_and_labels_multi_classes(data_file):
    """

    """
    # Load data from files
    positive_examples = list(open(data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    x_text = [s.split('\t')[1] for s in positive_examples]
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    labels = [s.split('\t')[0] for s in positive_examples]
    labels = one_hot(labels)

    return [x_text, labels]


def load_data_and_strlabels_multi_classes(data_file):
    """

    """
    # Load data from files
    positive_examples = list(open(data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    x_text = [s.split('\t')[1] for s in positive_examples]
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    labels = [s.split('\t')[0] for s in positive_examples]

    return [x_text, labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring


def load_vector_bin(fname, binary=True):
    encoding = 'utf-8'
    unicode_errors = 'strict'

    with open(fname, 'rb') as fin:
        header = str(fin.readline(), encoding=encoding)
        vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format
        vector_size = vector_size
        vectors = np.zeros((vocab_size, vector_size))

        vocab = {}
        def add_word(word, weights):
            word_id = len(vocab)
            if word in vocab:
                print("duplicate word '%s' in %s, ignoring all but first", word, fname)
                return
            vocab[word] = word_id
            vectors[word_id] = weights

        if binary:
            binary_len = dtype(REAL).itemsize * vector_size
            for _ in range(vocab_size):
                # mixed text and binary: read text first, then binary
                word = []
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors=unicode_errors)

                weights = fromstring(fin.read(binary_len), dtype=REAL)
                add_word(word, weights)
        else:
            for line_no in range(vocab_size):
                line = fin.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                word, weights = parts[0], [REAL(x) for x in parts[1:]]
                add_word(word, weights)
    if vectors.shape[0] != len(vocab):
        print(
            "duplicate words detected, shrinking matrix size from %i to %i",
            vectors.shape[0], len(vocab)
        )
        vectors = ascontiguousarray(vectors[: len(vocab)])
    assert (len(vocab), vector_size) == vectors.shape

    print("loaded %s matrix from %s", vectors.shape, fname)
    return vectors


