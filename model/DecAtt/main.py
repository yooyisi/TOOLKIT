from __future__ import division

import collections
import argparse
import math
import pickle

from util import *
import time
import torch
import random
import numpy as np
from torch.autograd import Variable
from torchtext.vocab import Vectors, Vocab

from model import DecAtt
from baseModel import BASE


def create_batch(data, from_index, to_index):
    if to_index > len(data):
        to_index = len(data)
    lsize = 0
    rsize = 0
    lsize_list = []
    rsize_list = []
    for i in range(from_index, to_index):
        length = len(data[i][0]) + 2
        lsize_list.append(length)
        if length > lsize:
            lsize = length
        length = len(data[i][1]) + 2
        rsize_list.append(length)
        if length > rsize:
            rsize = length
    lsent = data[from_index][0]
    lsent = ['bos'] + lsent + ['oov' for k in range(lsize - 1 - len(lsent))]
    left_sents = torch.cat((dict[word].view(1, -1) for word in lsent))
    left_sents = torch.unsqueeze(left_sents, 0)

    rsent = data[from_index][1]
    rsent = ['bos'] + rsent + ['oov' for k in range(rsize - 1 - len(rsent))]
    right_sents = torch.cat((dict[word].view(1, -1) for word in rsent))
    right_sents = torch.unsqueeze(right_sents, 0)

    labels = [data[from_index][2]]

    for i in range(from_index + 1, to_index):
        lsent = data[i][0]
        lsent = ['bos'] + lsent + ['oov' for k in range(lsize - 1 - len(lsent))]
        left_sent = torch.cat((dict[word].view(1, -1) for word in lsent))
        left_sent = torch.unsqueeze(left_sent, 0)
        left_sents = torch.cat([left_sents, left_sent])

        rsent = data[i][1]
        rsent = ['bos'] + rsent + ['oov' for k in range(rsize - 1 - len(rsent))]
        right_sent = torch.cat((dict[word].view(1, -1) for word in rsent))
        right_sent = torch.unsqueeze(right_sent, 0)
        right_sents = torch.cat((right_sents, right_sent))

        labels.append(data[i][2])


if __name__ == '__main__':
    EMBEDDING_DIM = 300
    PROJECTED_EMBEDDING_DIM = 300

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e', dest='num_epochs', default=5000, type=int, help='Number of epochs')
    parser.add_argument('--b', dest='batch_size', default=32, help='Batch size', type=int)
    parser.add_argument('--u', dest='num_units', help='Number of hidden units', default=100, type=int)
    parser.add_argument('--r', help='Learning rate', type=float, default=0.05, dest='rate')
    parser.add_argument('--lower', help='Lowercase the corpus', default=True, action='store_true')
    parser.add_argument('--model', help='Model selection', default='DecAtt', type=str)
    parser.add_argument('--optim', help='Optimizer algorithm', default='adagrad',
                        choices=['adagrad', 'adadelta', 'adam'])
    parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm', type=float, default=5)
    num_class = 2

    with open('data/data.pickle', 'rb') as f:
        data = pickle.load(f)
    random.shuffle(data)

    max_len = 0
    words = []

    for [_, sen1, sen2] in data:
        max_len = max(len(sen1), max_len)
        max_len = max(len(sen2), max_len)
        words.extend(list(sen1))
        words.extend(list(sen2))

    vectors = Vectors('wiki.en.vec', cache='data/')
    vocab = Vocab(collections.Counter(words), vectors=vectors, specials=['<pad>', '<unk>'], min_freq=1)

    def text2nparray(text, max_text_len=max_len):
        nparray = vocab.stoi['<pad>'] * np.ones([max_text_len], dtype=np.int64)
        nparray[:len(text)] = [vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>'] for x in text]
        return nparray

    data = [[label, text2nparray(sen1), text2nparray(sen2), len(sen1), len(sen2)] for [label, sen1, sen2] in data]

    train_pairs = data[:int(0.8*len(data))]
    dev_pairs = data[int(0.8*len(data)):]
    test_pairs = data[int(0.8*len(data)):]

    args = parser.parse_args()
    print('Model: %s' % args.model)
    print('Read data ...')
    print('Number of training pairs: %d' % len(train_pairs))
    print('Number of development pairs: %d' % len(dev_pairs))
    print('Number of testing pairs: %d' % len(test_pairs))

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    tokens = []
    dict = {}
    word2id = {}

    tokens.append('oov')
    tokens.append('bos')

    model = BASE(200, num_class, len(tokens), EMBEDDING_DIM, vocab)

    criterion = torch.nn.NLLLoss(size_average=True)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05, weight_decay=5e-5)

    print('Start training...')
    batch_counter = 0
    best_dev_loss = 10e10
    best_dev_loss_m = 10e10
    best_dev_loss_um = 10e10
    accumulated_loss = 0
    report_interval = 1000
    model.train()

    batch_index = 0
    for epoch in range(num_epochs):
        batch_counter = 0
        accumulated_loss = 0
        model.train()
        print('--' * 20)
        start_time = time.time()
        num_batches = math.floor(len(train_pairs) / batch_size)
        train_batch_i = 0
        train_sents_scaned = 0
        train_num_correct = 0
        while train_batch_i < 5:
            batch_data = train_pairs[train_batch_i*batch_size: (train_batch_i+1)*batch_size]
            train_batch_i += 1
            left_sents, right_sents, labels, lsize_list, rsize_list \
                = torch.LongTensor([i[1] for i in batch_data]),\
                  torch.LongTensor([i[2] for i in batch_data]),\
                  torch.LongTensor([i[0] for i in batch_data]),\
                  torch.LongTensor([i[3] for i in batch_data]),\
                  torch.LongTensor([i[4] for i in batch_data])

            if torch.cuda.is_available():
                left_sents = left_sents.cuda()
                right_sents = right_sents.cuda()
                labels = labels.cuda()
                lsize_list = lsize_list.cuda()
                rsize_list = rsize_list.cuda()

            train_sents_scaned += len(labels)
            optimizer.zero_grad()

            output = model(left_sents, right_sents, lsize_list, rsize_list)
            result = output.data.cpu().numpy()
            a = np.argmax(result, axis=1)
            b = labels.data.cpu().numpy()
            train_num_correct += np.sum(a == b)
            loss = criterion(output, labels)
            loss.backward()

            grad_norm = 0.

            optimizer.step()
            batch_counter += 1
            accumulated_loss += loss.data.item()
            if batch_counter % report_interval == 0:
                msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
                msg += '\t train batch loss: %f' % (accumulated_loss / train_sents_scaned)
                print(msg)

        # valid after each epoch
        model.eval()
        dev_batch_index = 0
        dev_num_correct = 0
        msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
        accumulated_loss = 0
        dev_batch_i = 0
        num_batches = math.floor(len(dev_pairs) / batch_size)
        pred = []
        gold = []
        while dev_batch_i < num_batches:
            batch_data = dev_pairs[dev_batch_i*batch_size: (dev_batch_i+1)*batch_size]
            dev_batch_i += 1
            left_sents, right_sents, labels, lsize_list, rsize_list \
                = torch.LongTensor([i[1] for i in batch_data]), \
                  torch.LongTensor([i[2] for i in batch_data]), \
                  torch.LongTensor([i[0] for i in batch_data]), \
                  torch.LongTensor([i[3] for i in batch_data]), \
                  torch.LongTensor([i[4] for i in batch_data])

            if torch.cuda.is_available():
                left_sents = left_sents.cuda()
                right_sents = right_sents.cuda()
                labels = labels.cuda()
                lsize_list = lsize_list.cuda()
                rsize_list = rsize_list.cuda()

            output = model(left_sents, right_sents, lsize_list, rsize_list)
            result = np.exp(output.data.cpu().numpy())
            loss = criterion(output, labels)
            accumulated_loss += loss.data.item()
            a = np.argmax(result, axis=1)
            b = labels.data.cpu().numpy()
            dev_num_correct += np.sum(a == b)
        dev_acc = dev_num_correct / (num_batches * batch_size)
        msg += '\t dev loss: %f\t dev acc: %f' % (accumulated_loss, dev_acc)
        print(msg)