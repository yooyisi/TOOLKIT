# -*- coding: utf-8 -*-
import os
import re
import sys
from itertools import islice

FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(FILE_PATH) + '/..'
sys.path.append(PARENT_PATH)

train_data_file = FILE_PATH + '/train.txt'
stop_word_file = FILE_PATH + '/stop_word.txt'

import ambiguitySegmenter as myjieba

KEEP_POS = {'cont': 'CONTINENT', 'nco': 'COUNTRY', 'npr': 'PROVINCE', 'na': 'CITY', 'naf': 'CITY', 'region': 'REGION',
            'nregion': 'REGION',
            'tjd': 'POI', 'hs': 'COMMERCIAL_ZONE',
            'mt': 'TIME',
            'nf': 'AIRPORT', 'np': 'AIRLINE', 'fc': 'SEATCLASS',
            'pr': 'PRICE', 'tl': 'STAY',
            'hb': '酒店', 'ht': '酒店', 'star': '酒店', 'hotelbed': '酒店'}

stopword_list = list(open(stop_word_file, "r").readlines())
stopword_list = [x.strip() for x in stopword_list]


# 给一个文本，标签的List就好
def get_data_tag_pairs():
    train_data, train_corpus = [], []
    with open(train_data_file, 'r') as fp:
        for line in islice(fp, 0, None):
            line = preProcessing(line)
            train_corpus.append(line)
            pairs = line.split("\t")
            text = pairs[1]
            if type(text) is str:
                text = text.decode('utf-8')
            # if u'晚' in text:
            #     print text
            word_list, pos_list = myjieba.lcut_with_tag(text)  # cut_all=True
            seg_list = []
            for w, pos in zip(word_list, pos_list):
                if w.encode('utf-8') in stopword_list:
                    continue
                if pos in KEEP_POS:
                    if len(re.findall(u'tl', pos)) > 0 and len(re.findall(u'[晚宿]', w)) > 0:
                        seg_list.append('酒店')
                    else:
                        seg_list.append(KEEP_POS[pos])
                else:
                    if type(w) is not str:
                        w = w.encode('utf-8')
                    seg_list.append(w)
            words_str = " ".join(seg_list)
            train_data.append((words_str, pairs[0], pairs[1]))

    with open(train_data_file, 'w') as fout:
        train_corpus = sorted(train_corpus)
        fout.write('\n'.join(train_corpus))

    return train_data


# 给一个文本，标签的List就好
def get_format_text(sentence):
    text = preProcessing(sentence)
    if type(text) is str:
        text = text.decode('utf-8')

    word_list, pos_list = myjieba.lcut_with_tag(text)  # cut_all=True
    seg_list = []
    for w, pos in zip(word_list, pos_list):
        if w.encode('utf-8') in stopword_list:
            continue
        if pos in KEEP_POS:
            seg_list.append(KEEP_POS[pos])
        else:
            if type(w) is not str:
                w = w.encode('utf-8')
            seg_list.append(w)
    words_str = " ".join(seg_list)

    return words_str


def preProcessing(str):
    # remove special characters
    str = str.strip("\n")
    str = str.replace(" ", "")
    return str
