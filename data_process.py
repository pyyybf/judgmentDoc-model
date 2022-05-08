# -*- coding: UTF-8 -*-
"""
@author:黄云云
@file:data_process.py
@time:2022/03/22
"""
import os
from random import shuffle
from sklearn.model_selection import train_test_split


def generate_dataset_for_recommendation(args):
    all_data = load_txt(args.data_dir + args.crime + '/' + args.recommendation_corpus_name)
    cases = set()
    for line in all_data:
        cases.add(line.split('|')[0])

    # 按案件划分训练集、测试集
    train_cases, test_cases = train_test_split(list(cases), shuffle=True, test_size=0.2)
    valid_cases, test_cases = train_test_split(test_cases, shuffle=True, test_size=0.5)

    train_data, test_data, valid_data = [], [], []
    for line in all_data:
        case = line.split('|')[0]
        if case in train_cases:
            train_data.append(line)
        elif case in test_cases:
            test_data.append(line)
        else:
            valid_data.append(line)

    if not os.path.exists(args.data_dir + args.crime + args.split_data_dir):
        os.makedirs(args.data_dir + args.crime + args.split_data_dir)

    write_txt(train_data, args.data_dir + args.crime + args.split_data_dir + args.train_recommendation_corpus_name)
    write_txt(valid_data, args.data_dir + args.crime + args.split_data_dir + args.valid_recommendation_corpus_name)
    write_txt(test_data, args.data_dir + args.crime + args.split_data_dir + args.test_recommendation_corpus_name)


def load_txt(path):
    f = open(path, "r", encoding="utf8")
    data = f.readlines()
    f.close()

    return data


def write_txt(all_data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for data in all_data:
            f.write(data.strip() + '\n')


def get_process_data(data):
    pos_res = []
    neg_res = []
    for line in data:
        items = line.strip().split('|')
        assert len(items) == 4, ValueError("The number of items in this line is less than 4, content:" + line)
        fact = items[1]
        positive_samples = set([int(num) for num in items[2].split(',')])
        negative_samples = set([int(num) for num in items[3].split(',')])

        for sample in positive_samples:
            pos_res.append([fact, sample, 1])

        for sample in negative_samples:
            neg_res.append([fact, sample, 0])

    shuffle(pos_res)
    shuffle(neg_res)
    return pos_res, neg_res


def get_article_dict(args):
    article_dict = load_txt(args.data_dir + args.crime + '/' + args.law_article_content_name)
    article_dict = {int(item[0]): item[1] for item in [line.split('|') for line in article_dict]}

    for idx in article_dict:
        article_content = article_dict[idx].split(':')[1].replace('\n', '').replace('\t', '').replace(' ', '')
        article_dict[idx] = article_content

    return article_dict
