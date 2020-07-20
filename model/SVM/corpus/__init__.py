# -*- coding: utf-8 -*-
import os
import sys
from itertools import islice
import codecs
FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(FILE_PATH) + '/..'
sys.path.append(PARENT_PATH)

train_data_file = FILE_PATH + '/kw_log_train.txt'

import jieba


# 给一个文本，标签的List就好
def get_data_tag_pairs():
    train_data, train_corpus = [], []
    with codecs.open(train_data_file, 'r',encoding="utf8") as fp:
        for line in islice(fp, 0, None):
            line = preProcessing(line)
            train_corpus.append(line)
            pairs = line.split("\t")
            # print(pairs)
            if len(pairs) < 2:
                print(line)
            text = pairs[1]
            word_list = jieba.lcut(text)
            words_str = " ".join(word_list)
            train_data.append((words_str, pairs[0], pairs[1]))

    with codecs.open(train_data_file, 'w', encoding="utf8") as fout:
        train_corpus = sorted(train_corpus)
        fout.write('\n'.join(train_corpus))

    return train_data


# 给一个文本，标签的List就好
def get_format_text(sentence):
    text = preProcessing(sentence)

    word_list = jieba.lcut(text)  # cut_all=True
    seg_list = []
    for w in word_list:
        seg_list.append(w)
    words_str = " ".join(seg_list)

    return words_str


def preProcessing(str):
    # remove special characters
    str = str.strip("\r\n")
    str = str.replace(" ", "")
    return str


def keyword_statics():
    class_wl = {}
    with codecs.open(train_data_file, 'r',encoding="utf8") as fp:
        for line in islice(fp, 0, None):
            line = preProcessing(line)
            pairs = line.split("\t")
            if len(pairs) < 2:
                print(line)
            text = pairs[1]
            word_list = jieba.lcut(text)
            class_wl[pairs[0]] = class_wl.get(pairs[0], [])
            class_wl[pairs[0]].extend(word_list)
    class_wl = {k: list(set(v)) for k, v in class_wl.items()}

    for c, wl in class_wl.items():
        unique_w = []
        for w in wl:
            appear = sum([1 if w in class_wl[ci] else 0 for ci in class_wl.keys()])
            appear_classes = [ci for ci in class_wl.keys() if w in class_wl[ci]]
            if appear == 1:
                unique_w.append(w)
            else:
                print(w)
                print(' '.join(appear_classes))
        # print(c)
        # print(' '.join(unique_w))
