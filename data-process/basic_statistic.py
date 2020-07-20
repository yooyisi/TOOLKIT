import itertools
import math

import jieba


def idf(sent_list):
    word_dic = {}
    for sent in sent_list:
        wl = jieba.lcut(sent)
        for w in set(wl):
            word_dic[w] = word_dic.get(w, 0) + 1

    for w, count in word_dic.items():
        word_dic[w] = math.log(1.0*len(sent_list)/count, 10)
    return word_dic


def tfidf(sent_list):
    word_dic = {}
    for sent in sent_list:
        wl = jieba.lcut(sent)

    return word_dic


def stat_sentences(sentlist):
    # char level
    # output：最大句子长度，字典
    charlist = [list(sent) for sent in sentlist]
    max_len = max([len(i) for i in charlist])
    charlist = list(itertools.chain.from_iterable(charlist))
    charset = set(charlist)

    map_id_char, map_char_id = {}, {}
    for i, ch in enumerate(charset):
        map_char_id[ch] = i
        map_id_char[i] = ch
    return max_len, map_id_char, map_char_id


def auc(path, step=0.01):
    '''
    显示ROC曲线，并找最佳阈值
    :param path: 数据文件路径，每一行：label \t pos_probability
    :return:
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    lines = open(path, 'r').readlines()
    dataset = [x.split('\t') for x in lines]
    dataset = [[l, float(p)] for [l, p] in dataset]

    x,y = [0],[0]
    best_dif, best_threshold = -1, -1
    for threshold in np.arange(1.0, 0.0, -step):
        count_tp, count_fp = 0, 0
        count_true_samples = 0
        for [label, pos_prob] in dataset:
            if label == '1' and pos_prob>=threshold:
                count_tp += 1
            elif label == '0' and pos_prob>=threshold:
                count_fp += 1
            if label == '1':
                count_true_samples += 1
        xi,yi = 1.0*count_fp/count_true_samples, 1.0*count_tp/count_true_samples
        if yi-xi > best_dif:
            print(xi, yi, best_threshold, threshold)
            best_dif = yi-xi
            best_threshold = threshold
        x.append(1.0*count_fp/count_true_samples)
        y.append(1.0*count_tp/count_true_samples)

    print('best threshold', best_threshold)

    x += [1]
    y += [1]

    auc_area = 0
    for i in range(len(x)-1):
        auc_area += (x[i+1] - x[i])*y[i+1]
    print('auc', auc_area)

    plt.step(x, y, label='pre (default)')
    plt.plot(x, y, 'o--', color='grey', alpha=0.3)

    xdiag,ydiag = np.arange(0.0, 1.0, 0.01),np.arange(0.0, 1.0, 0.01)
    plt.plot(xdiag,ydiag, '-', color='red', alpha=0.3)

    plt.grid(axis='x', color='0.95')
    plt.legend(title='Parameter where:')
    plt.title('plt.step(where=...)')
    plt.show()

auc('auc.txt')