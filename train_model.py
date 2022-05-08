# -*- coding: UTF-8 -*-
"""
@author:潘越
@file:train_model.py
@time:2022/04/04
"""

import os
import random
import numpy as np

from parameters import parse
from train_tools import train_model
from data_process import generate_dataset_for_recommendation, load_txt, get_process_data, get_article_dict
from models import *


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


args = parse()
set_seed(args.seed)

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if not os.path.exists(args.checkpoint_dir + args.crime + '/'):
    os.makedirs(args.checkpoint_dir + args.crime + '/')

article_dict = get_article_dict(args)

if not os.path.exists(args.data_dir + args.crime + args.split_data_dir + args.train_recommendation_corpus_name):
    generate_dataset_for_recommendation(args)

train_data = load_txt(args.data_dir + args.crime + args.split_data_dir + args.train_recommendation_corpus_name)
train_data = get_process_data(train_data)

valid_data = load_txt(args.data_dir + args.crime + args.split_data_dir + args.valid_recommendation_corpus_name)
valid_data = get_process_data(valid_data)

test_data = load_txt(args.data_dir + args.crime + args.split_data_dir + args.test_recommendation_corpus_name)
test_data = get_process_data(test_data)

model = globals()[args.model_name](args)

train_model(model, args, train_data, valid_data, test_data, article_dict)
