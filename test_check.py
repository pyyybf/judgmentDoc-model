# -*- coding: UTF-8 -*-
"""
@author:潘越
@file:test_check.py
@time:2022/04/10
"""
from check_tools import print_json, check
from parameters import parse


def single_test(crime, text_name):
    print("Start checking {0}.txt in {1}...".format(text_name, crime))
    test_text = open('./test_txt/{0}/{1}.txt'.format(crime, text_name), 'r', encoding='utf-8')
    print_json(check(test_text.read(), crime))
    test_text.close()


args = parse()
single_test(args.crime, args.txt_name)
