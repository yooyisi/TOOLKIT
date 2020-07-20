# -*- coding: utf-8 -*-
import os
import re
import sys


FILE_PATH = os.path.split(os.path.realpath(__file__))[0]

classifier_file = FILE_PATH + '/model/classifier.pkl'
vec_transformer_file = FILE_PATH + '/model/vec_transformer.pkl'
tfidf_transformer_file = FILE_PATH + '/model/tfidf_transformer.pkl'


class PKL_FILE:
    def __init__(self, domain_name, vec_tranformer, tfidf_tranformer, clf):
        self.domain_name = domain_name
        self.vec_tranformer = vec_tranformer
        self.tfidf_tranformer = tfidf_tranformer
        self.clf = clf


intent_domain_files = {'top': {}, 'open': {}, 'hotel': {}, 'flight': {}}
for domain_name in intent_domain_files.keys():
    intent_domain_files[domain_name] = PKL_FILE(
        domain_name,
        re.sub('_toreplace_', domain_name, vec_transformer_file),
        re.sub('_toreplace_', domain_name, tfidf_transformer_file),
        re.sub('_toreplace_', domain_name, classifier_file)
    )

