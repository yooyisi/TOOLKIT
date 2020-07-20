# -*- coding: utf-8 -*-
# @Time: 2020/3/20 16:25
# @Author:
import numpy as np
import torch
from torch import nn


class TrigonometricEmbedding(nn.Module):
    def __init__(self, n_position, vec_dim=64):
        super(TrigonometricEmbedding, self).__init__()
        self.vec_dim = vec_dim
        self.n_position = n_position

    def forward(self):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.vec_dim) for i in range(self.vec_dim)]
            if pos != 0 else np.zeros(self.vec_dim) for pos in range(self.n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        return torch.from_numpy(position_enc)


class MatrixEmbedding(nn.Module):
    def __init__(self, max_position_embeddings=512, hidden_size=768):
        super(MatrixEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings
