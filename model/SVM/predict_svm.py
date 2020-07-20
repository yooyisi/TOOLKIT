# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
from io import open
import config


FILE_PATH = os.path.split(os.path.realpath(__file__))[0]

uni_bi_tri_gram = '1'

with open(config.vec_transformer_file, 'rb') as fid:
    count_vec = pickle.load(fid)
with open(config.tfidf_transformer_file, 'rb') as fid:
    tfidf_transformer = pickle.load(fid)


with open(config.classifier_file, 'rb') as fid:
    classifier = pickle.load(fid)

classes = classifier.classes_


def fenci(x_text):
    x_text_after_seg = []
    for xi in x_text:
        unigram = [i for i in xi]
        bigram = [unigram[i] + unigram[i+1] for i in range(len(unigram)-1)]
        trigram = [unigram[i] + unigram[i+1] + unigram[i+2] for i in range(len(unigram)-2)]
        if uni_bi_tri_gram == '1':
            wl = unigram[:]
        elif uni_bi_tri_gram == '2':
            wl = unigram + bigram
        else:
            wl = unigram + bigram + trigram
        x_text_after_seg.append(' '.join(wl))
    return x_text_after_seg


def get_intent_sent(sentence):

    x = fenci([sentence])

    # 用已经有的词典向量化sentence
    wordArray = count_vec.transform(x)
    word_tfidf_array = tfidf_transformer.transform(wordArray).toarray()

    pred_res = classifier.predict(word_tfidf_array)
    pred_prob = classifier.predict_proba(word_tfidf_array)
    intent = pred_res[0]

    return intent, 1


get_intent_sent('i的那颗大卡扣旅游港澳')
