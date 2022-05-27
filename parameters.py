# -*- coding: UTF-8 -*-
"""
@author:黄云云
@file:parameters.py
@time:2022/03/22
"""

import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', dest='seed', type=int, default=1024)
    parser.add_argument('--crime', type=str, default='traffic', choices=['traffic', 'hurt'])
    parser.add_argument('--fact_len', dest='fact_len', type=int, default=50)
    parser.add_argument('--article_len', dest='article_len', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='ThreeLayers')
    parser.add_argument('--pretrain_model_name', type=str, default='clue/albert_chinese_tiny')
    parser.add_argument('--tokenizer_name', type=str, default='clue/albert_chinese_tiny')

    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--recommendation_corpus_name', type=str, default='labeled_corpus.txt')
    parser.add_argument('--split_data_dir', type=str, default='/split_data/')
    parser.add_argument('--train_recommendation_corpus_name', type=str, default='train_data.txt')
    parser.add_argument('--valid_recommendation_corpus_name', type=str, default='valid_data.txt')
    parser.add_argument('--test_recommendation_corpus_name', type=str, default='test_data.txt')
    parser.add_argument('--law_article_content_name', type=str, default='article_dict.txt')
    parser.add_argument('--law_article_qhj_dict_name', type=str, default='law_qhj_dict.json')

    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=312)
    parser.add_argument('--filters_num', dest='filters_num', type=int, default=128)
    parser.add_argument('--kernel_size_1', dest='kernel_size_1', type=int, default=2)
    parser.add_argument('--kernel_size_2', dest='kernel_size_2', type=int, default=4)
    parser.add_argument('--linear_output', dest='linear_output', type=int, default=128)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    parser.add_argument('--negative_multiple', dest='negative_multiple', type=int, default=12)
    parser.add_argument('--earlystop_patience', dest='earlystop_patience', type=int, default=10)

    # test check
    parser.add_argument('--txt_name', dest='txt_name', type=str, default='001')

    args = parser.parse_args()

    return args
