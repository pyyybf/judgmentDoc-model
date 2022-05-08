# -*- coding: UTF-8 -*-
"""
@author:黄云云
@file:models.py
@time:2022/03/22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class ThreeLayers(nn.Module):
    def __init__(self, args):
        super(ThreeLayers, self).__init__()
        self.embedding = BertModel.from_pretrained(args.pretrain_model_name)
        self.fact_convs = nn.ModuleList([
            nn.Conv1d(args.embedding_dim, args.filters_num, args.kernel_size_1),
            nn.Conv1d(args.filters_num, args.filters_num, args.kernel_size_2)
        ])
        self.article_convs = nn.ModuleList([
            nn.Conv1d(args.embedding_dim, args.filters_num, args.kernel_size_1),
            nn.Conv1d(args.filters_num, args.filters_num, args.kernel_size_2),
        ])

        self.conv_paddings = nn.ModuleList([
            nn.ConstantPad1d((args.kernel_size_1 // 2 - 1, args.kernel_size_1 // 2), 0.),
            nn.ConstantPad1d((args.kernel_size_2 // 2 - 1, args.kernel_size_2 // 2), 0.)
        ])

        self.ffs = nn.ModuleList(
            [nn.Linear(args.embedding_dim, args.linear_output)] +
            [nn.Linear(args.filters_num, args.linear_output) for _ in range(2)] +
            [nn.Linear(args.article_len * 3, args.linear_output)]
        )

        self.predict = nn.Linear(args.linear_output, 2)

    def forward(self, fact_ii, fact_am, article_ii, article_am):
        input_ids = torch.cat((fact_ii, article_ii), dim=-1)
        attention_mask = torch.cat((fact_am, article_am), dim=-1)
        embeddings = self.embedding(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        length = embeddings.size(1) // 2
        fact = embeddings[:, :length, :]
        article = embeddings[:, length:, :]

        fact = fact * fact_am.unsqueeze(dim=-1).repeat(1, 1, 768)
        article = article * article_am.unsqueeze(dim=-1).repeat(1, 1, 768)

        fact, article = fact[:, 1:-1, :], article[:, 1:-1, :]

        # zero-layer
        inter_0 = self.interaction(fact, article).unsqueeze(-1)  # (batch_size, article_len, 1)
        inter_repeat_0 = inter_0.repeat(1, 1, fact.size(-1))  # (batch_size, article_len, embedding_dim)
        info_article_0 = inter_repeat_0.mul(article)  # (batch_size, article_len, embedding_dim)

        fusion_output_0 = self.ffs[0](info_article_0)
        fusion_output_max_0 = torch.max(fusion_output_0, dim=-1)[0]

        # fisrt-layer
        input_fact_1 = self.conv_paddings[0](fact.permute(0, 2, 1))
        input_article_1 = self.conv_paddings[0](article.permute(0, 2, 1))

        fact_conv_1 = self.fact_convs[0](input_fact_1).permute(0, 2, 1)
        article_conv_1 = self.article_convs[0](input_article_1).permute(0, 2, 1)

        inter_1 = self.interaction(fact_conv_1, article_conv_1).unsqueeze(-1)
        inter_repeat_1 = inter_1.repeat(1, 1, article_conv_1.size(-1))
        info_article_1 = inter_repeat_1.mul(article_conv_1)

        fusion_output_1 = self.ffs[1](info_article_1)
        fusion_output_max_1 = torch.max(fusion_output_1, dim=-1)[0]

        # second_layer
        input_fact_2 = self.conv_paddings[1](fact_conv_1.permute(0, 2, 1))
        input_article_2 = self.conv_paddings[1](article_conv_1.permute(0, 2, 1))

        fact_conv_2 = self.fact_convs[1](input_fact_2).permute(0, 2, 1)
        article_conv_2 = self.article_convs[1](input_article_2).permute(0, 2, 1)

        inter_2 = self.interaction(fact_conv_2, article_conv_2).unsqueeze(-1)
        inter_repeat_2 = inter_2.repeat(1, 1, article_conv_2.size(-1))
        info_article_2 = inter_repeat_2.mul(article_conv_2)

        fusion_output_2 = self.ffs[2](info_article_2)
        fusion_output_max_2 = torch.max(fusion_output_2, dim=-1)[0]

        fusion_output = torch.cat([fusion_output_max_0, fusion_output_max_1, fusion_output_max_2],
                                  dim=-1)
        fusion_output = F.dropout(F.relu(self.ffs[-1](fusion_output)))

        output = self.predict(fusion_output)

        return output

    @staticmethod
    def interaction(x1, x2):
        """

        :param x1: (batch_size, x1_seq_len, feature_size)
        :param x2: (batch_size, x2_seq_len, feature_size)
        :return: (batch_size, x2_seq_len)
        """
        dot_matrix = torch.matmul(x1, x2.permute(0, 2, 1))  # (batch_size, x1_seq_len, x2_seq_len)

        x1_2_x2 = F.softmax(dot_matrix, dim=2)
        x2_2_x1 = F.softmax(dot_matrix, dim=1)

        x1_weight = torch.sum(x2_2_x1, dim=2).unsqueeze(dim=1)  # (batch_size, 1, x1_seq_len)
        x2_weight = torch.matmul(x1_weight, x1_2_x2).squeeze(dim=1)  # (batch_size, x2_seq_len)

        return x2_weight

    @staticmethod
    def get_name():
        return 'Three Layers Based Pretrain'

    @staticmethod
    def get_lr():
        return 1e-5

    @staticmethod
    def get_weight_decay():
        return 0
