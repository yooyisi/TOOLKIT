
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def generate_mask_2(values, sent_sizes):
    mask_matrix = np.zeros((len(sent_sizes), max(sent_sizes), values.size(2)))
    for i in range(len(sent_sizes)):
        mask_matrix[i][:sent_sizes[i]][:] = 1
    if torch.cuda.is_available():
        mask_matrix = torch.Tensor(mask_matrix).cuda()
    else:
        mask_matrix = torch.Tensor(mask_matrix)
    return values * Variable(mask_matrix)


def generate_mask(lsent_sizes, rsent_sizes):
    mask_matrix = np.zeros((len(lsent_sizes), max(lsent_sizes), max(rsent_sizes)))
    for i in range(len(lsent_sizes)):
        mask_matrix[i][:lsent_sizes[i]][:rsent_sizes[i]] = 1
    if torch.cuda.is_available():
        mask_matrix = torch.Tensor(mask_matrix).cuda()
    else:
        mask_matrix = torch.Tensor(mask_matrix)
    return Variable(mask_matrix)


class BASE(nn.Module):
    """
        Implementation of the multi feed forward network model described in
        the paper "A Decomposable Attention Model for Natural Language
        Inference" by Parikh et al., 2016.

        It applies feedforward MLPs to combinations of parts of the two sentences,
        without any recurrent structure.
    """

    def __init__(self, num_units, num_classes, vocab_size, embedding_size,
                 vocab, training=True, project_input=True,
                 use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
        Create the model based on MLP networks.

        :param num_units: size of the networks
        :param num_classes: number of classes in the problem
        :param vocab_size: size of the vocabulary
        :param embedding_size: size of each word embedding
        :param use_intra_attention: whether to use intra-attention model
        :param training: whether to create training tensors (optimizer)
        :param project_input: whether to project input embeddings to a
            different dimensionality
        :param distance_biases: number of different distances with biases used
            in the intra-attention model
        """
        super(BASE, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input
        self.embedding_size = embedding_size
        self.distance_biases = distance_biases
        self.intra_attention = False
        self.max_sentence_length = max_sentence_length

        self.vocab_size, self.embedding_dim = vocab.vectors.size()
        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight.data = vocab.vectors
        self.embedding_layer.weight.requires_grad = False

        self.bias_embedding = nn.Embedding(max_sentence_length, 1)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear_layer_project = nn.Linear(embedding_size, num_units, bias=False)

        self.linear_layer = nn.Linear(2 * self.embedding_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def aggregate(self, v1, v2):
        """
        Aggregate the representations induced from both sentences and their
        representations

        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
        """
        v1_sum = torch.sum(v1, 1)
        v2_sum = torch.sum(v2, 1)
        out = self.linear_layer_aggregate(torch.cat([v1_sum, v2_sum], 1))
        return out

    def forward(self, sent1, sent2, lsize_list, rsize_list):
        sent1 = self.embedding_layer(sent1)
        sent2 = self.embedding_layer(sent2)

        # average emb
        sent1_avg = torch.div(torch.squeeze(torch.sum(sent1, dim=1)), torch.unsqueeze(lsize_list.float(), dim=1))
        sent2_avg = torch.div(torch.squeeze(torch.sum(sent2, dim=1)), torch.unsqueeze(rsize_list.float(), dim=1))

        out = self.linear_layer(torch.cat((sent1_avg, sent2_avg), dim=1))
        res = self.softmax(out)

        return res
