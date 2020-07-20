# -*- coding: utf-8 -*-
# @Time: 2020/1/17 16:26
# @Author:
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, units):
        super(Dense).__init__()

    def forward(self, in_tensor, ):
        h = nn.Linear(in_tensor, units, bias=True)
        out = nn.ReLU(h)
        return out


class Attention(nn.Module):
    def __init__(self, attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads = num_attention_heads

        # components
        self.Linear_1 = nn.Linear(from_tensor_2d.size, self.num_attention_heads * self.size_per_head, bias=True)
        self.Linear_2 = nn.Linear(to_tensor_2d.size, self.num_attention_heads * self.size_per_head, bias=True)
        self.Linear_3 = nn.Linear(to_tensor_2d.size, self.num_attention_heads * self.size_per_head, bias=True)
        self.relu = nn.ReLU()


    def forward(self, from_tensor, to_tensor):
        """

        :param from_tensor: [batch_size, from_seq_length, from_width]
        :param to_tensor: [batch_size, to_seq_length, to_width]
        :return:
        """
        # [B*L, W]
        from_tensor_2d = convert2d(from_tensor)
        to_tensor_2d = convert2d(to_tensor)

        # [B*Length, HeadSize*NumHeads]
        query_layer = self.relu(self.Linear_1(from_tensor_2d, self.num_attention_heads * self.size_per_head))
        key_layer = self.relu(self.Linear_2(to_tensor_2d, self.num_attention_heads * self.size_per_head))
        value_layer = self.relu(self.Linear_3(to_tensor_2d, self.num_attention_heads * self.size_per_head))

        # `query_layer` = [B, Head_size, Length, Num_Head]
        query_layer = transpose_for_scores(query_layer, batch_size,
                                           num_attention_heads, from_seq_length, size_per_head)
        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                         to_seq_length, size_per_head)

        # [B, Head_size, From_length, To_length]
        attention_scores = torch.matmul(query_layer, key_layer)


class Encode(nn.Module):
    def __init__(self):
        super(Encode).__init__()
        self.linear_attention = nn.Linear()
        self.linear_intermediate = nn.Linear()
        self.linear_output = nn.Linear()


    def forward(self, layer_input):

        # attention
        attention_out = attention_layer(layer_input)
        attention_output = dense(attention_out)
        attention_output = dropout(attention_output)
        attention_output = layer_norm(attention_output + layer_input)

        # intermediate
        intermediate_output = dense(attention_output, intermediate_size)

        # output
        layer_output = dense(intermediate_output, hidden_size)
        layer_output = dropout(layer_output)
        layer_output = layer_norm(layer_output + attention_output)

        return layer_output


class Transformer(nn.Module):
    def __init__(self, input_tensor,
                 attention_mask=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 do_return_all_layers=False):
        super(Transformer, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.encoders = nn.ModuleList(Encode() for _ in range(num_hidden_layers))

    def forward(self, input):
        previous_out = input
        for layer_id in range(self.num_hidden_layers):


        return previous_out
