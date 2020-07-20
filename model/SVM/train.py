# -*- coding: utf-8 -*-
import pickle
import random
from itertools import islice

import jieba
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import config
import corpus

import data_helpers

NUM_SHOW_FEATURES = 100
SPLIT_RATIO = 0.9
FOLDS = 10
VOCALBULARY_SIZE = 0
# classifier = SVC(kernel='linear', probability=True)
classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
count_vec = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b', ngram_range=(1, 1))
tfidf_transformer = TfidfTransformer(use_idf=False)

uni_bi_tri_gram = '1'
dataset = 'kw'  # kw


def preprocess_(train_file, test_file):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text_train, y_str_train = data_helpers.load_data_and_strlabels_multi_classes(train_file)
    x_text_test, y_str_test = data_helpers.load_data_and_strlabels_multi_classes(test_file)
    y_train = data_helpers.one_hot(y_str_train)
    y_dev = data_helpers.one_hot(y_str_test, use_fixed_map=True)

    def fenci(x_text):
        x_text_after_seg = []
        for xi in x_text:
            unigram_w = jieba.lcut(xi)
            bigram_w = [unigram_w[i] + unigram_w[i + 1] for i in range(len(unigram_w) - 1)]

            unigram = [i for i in xi]
            bigram = [unigram[i] + unigram[i+1] for i in range(len(unigram)-1)]
            trigram = [unigram[i] + unigram[i+1] + unigram[i+2] for i in range(len(unigram)-2)]
            if uni_bi_tri_gram == '1':
                wl = unigram[:]
            elif uni_bi_tri_gram == '2':
                wl = unigram + bigram
            elif uni_bi_tri_gram == '2w':
                wl = unigram + bigram + bigram_w
            else:
                wl = unigram + bigram + trigram
            x_text_after_seg.append(' '.join(wl))
        return x_text_after_seg
    x_text_train = fenci(x_text_train)
    x_text_test = fenci(x_text_test)

    def two_list2dict(key_list, val_list):
        dic = {}
        for a, b in zip(key_list, val_list):
            dic[a] = dic.get(a, [])
            dic[a].append(b)

        return dic

    def balance_train_by_sampling(train_clas_sample_dict, num):
        train_data, labels = [], []
        for k, v in train_clas_sample_dict.items():
            for i in range(num):
                train_data.append(random.choice(v))
                labels.append(k)
        return train_data, labels

    dic = two_list2dict(y_str_train, x_text_train)
    x_text_train, y_str_train = balance_train_by_sampling(dic, 80)

    total_count_vec = count_vec.fit_transform(x_text_train)
    x_text_train_vec = tfidf_transformer.fit_transform(total_count_vec).toarray()
    total_count_vec = count_vec.transform(x_text_test)
    x_text_test_vec = tfidf_transformer.transform(total_count_vec).toarray()

    with open(config.vec_transformer_file, 'wb') as fid:
        pickle.dump(count_vec, fid)
    with open(config.tfidf_transformer_file, 'wb') as fid:
        pickle.dump(tfidf_transformer, fid)

    return x_text_train_vec, y_str_train, x_text_train, x_text_test_vec, y_str_test, x_text_test


def trainAndClassify():
    train_accuracies = []
    accuracies = []

    if dataset == 'kw':
        vec_train, lb_train, train_text, vec_test, lb_test, test_text = preprocess_('corpus/kw_log_train.txt', 'corpus/kw_log_test.txt')
    else:
        vec_train, lb_train, train_text, vec_test, lb_test, test_text = preprocess_('corpus/mix_train.txt', 'corpus/mix_test.txt')
    classifier.fit(vec_train, lb_train)

    pred_res = classifier.predict(vec_train)
    # pred_prob = svm_classifier.predict_proba(vec_train)
    pred_res_list = pred_res.tolist()
    accuracy = overlapping_percentage(pred_res_list, lb_train, train_text)
    train_accuracies.append(accuracy)
    print('train Accuracy :' + str(accuracy))
    print(metrics.classification_report(lb_train, pred_res_list))

    pred_res = classifier.predict(vec_test)
    pred_prob = classifier.predict_proba(vec_test)
    pred_res_list = pred_res.tolist()
    pred_score_list = np.max(pred_prob, axis=1)

    accuracy = overlapping_percentage(pred_res_list, lb_test, test_text, pred_score_list)
    accuracies.append(accuracy)
    print(metrics.classification_report(lb_test, pred_res_list))

    print("Train Accu:", 100*np.mean(train_accuracies))
    print("Accuracies:", 100*np.mean(accuracies))

    import pickle
    with open(config.classifier_file, 'wb') as fid:
        pickle.dump(classifier, fid)
    return np.mean(accuracies)


def overlapping_percentage(predit, actual, text=None, score=None):
    correct_num = 0
    for i in range(0, len(predit)):
        # probs = pred_prob[i]
        # max_prob = max(probs)
        # if max_prob < 0.8:
        #     # print 'xxxx'
        #     # print text[i]
        #     # print max_prob
        #     pass
        if predit[i] == actual[i]:
            correct_num += 1
            # print(text[i])
            # print(predit[i])
            # print('correct')
            # if score is not None:
            #     print(score[i])
            # print('')
            pass
        else:
            print(text[i])
            print(predit[i])
            print(actual[i])
            if score is not None:
                print(score[i])
            print('')
            pass
    return (1.0 * correct_num) / len(predit)


def realExampleTry(sentence):
    # 先分词
    words_str = corpus.get_format_text(sentence)
    # 用已经有的词典向量化sentence

    wordArray = count_vec.transform([words_str])
    word_tfidf_array = tfidf_transformer.transform(wordArray).toarray()
    # 用训练好的分类器去预测
    pred_res = classifier.predict(word_tfidf_array)
    print(sentence)
    print(pred_res[0])
    return pred_res[0]


def test_on_dataset(test_text_and_labels, topics):
    correct_lables = []
    results = []
    for text, label in test_text_and_labels:
        correct_lables.append(label)
        # 先分词
        seg_list = text.split()

        # 用分类器预测
        words_str = " ".join(seg_list)
        # 用已经有的词典向量化sentence
        wordArray = count_vec.transform([words_str])
        word_tfidf_array = tfidf_transformer.transform(wordArray).toarray()
        # 用训练好的分类器去预测
        pred_res = classifier.predict(word_tfidf_array)
        results.append(str(pred_res[0]))
    print(overlapping_percentage(results, correct_lables))


def clean_train_data(train_data_path):
    dataset = []
    with open(train_data_path, 'r', encoding='utf-8') as fin:
        for line in islice(fin, 0, None):
            dataset.append(line)
    dataset = list(set(dataset))
    dataset = sorted(dataset)
    with open(train_data_path, 'w', encoding='utf-8') as fout:
        fout.write(''.join(dataset))


def main():
    trainAndClassify()
    # realExampleTry("发现目的地")
    # realExampleTry("四星级的")
    # realExampleTry("订机票")
    # realExampleTry("住宿当天天气")
    # realExampleTry("吃饭的地方")


if __name__ == "__main__":
    main()
    clean_train_data('corpus/kw_log_train.txt')
