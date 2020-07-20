# -*- coding: utf-8 -*-
# @Time: 2020/3/18 14:58
# @Author:
import math

import torch
from torch import nn
import torch.nn.functional as f
import pytorch_transformers


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super(SelfAttention).__init__()
        self.dk = math.sqrt(hidden_dim)
        self.Wq = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def forward(self, x):
        # x: (batch_size, sentence_length, embedding_dim)
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        z = torch.matmul(f.softmax(torch.div(torch.matmul(q, k.t()), self.dk), dim=1), v)
        return z


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_head):
        super(MultiHeadAttention).__init__()
        self.dk = torch.sqrt(hidden_dim)

        self.head_list = []
        for i in range(num_head):

            self.head_list.append(
                nn.Sequential
            )
        self.Wq = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.Wo = nn.Linear(hidden_dim*num_head, hidden_dim, bias=False)

    def forward(self, *input):