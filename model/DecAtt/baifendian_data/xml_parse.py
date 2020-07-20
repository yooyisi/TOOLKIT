# -*- coding: utf-8 -*-

"""
现在的负样本是只取了一个问题下的，没有去取两个不同问题之间的
todo 考虑扩大负样本
"""

import xml.dom.minidom as xmldom
domobj = xmldom.parse('train_set.xml')

traincorpus = domobj.childNodes[0]
dataset = []
data2text = []
for question in traincorpus.childNodes:
    if question.nodeName == 'Questions':
        pos, neg = [], []
        for eq in question.childNodes:
            if eq.nodeName == 'EquivalenceQuestions':
                for text in eq.childNodes:
                    if text.nodeName == 'question' and text.firstChild:
                        pos.append(text.firstChild.data)
            elif eq.nodeName == 'NotEquivalenceQuestions':
                for text in eq.childNodes:
                    if text.nodeName == 'question' and text.firstChild:
                        neg.append(text.firstChild.data)
        # sample
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                dataset.append([1, pos[i], pos[j]])
                data2text.append('\t'.join([pos[i], pos[j], '1']))
        for i in range(len(pos)):
            for j in range(len(neg)):
                dataset.append([0, pos[i], neg[j]])
                data2text.append('\t'.join([pos[i], neg[j], '0']))

import pickle
with open('data.pickle', 'wb') as f:
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

with open('train.txt', 'w', encoding='utf-8') as fout:
    fout.write('\n'.join(data2text))
