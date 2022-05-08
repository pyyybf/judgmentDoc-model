# -*- coding: UTF-8 -*-
"""
@author:潘越
@file:test_check.py
@time:2022/04/10
"""
from check_tools import print_json, check


def single_test(crime, text_name):
    test_text = open('./test_txt/{0}/{1}'.format(crime, text_name), 'r', encoding='utf-8')
    print_json(check(test_text.read(), crime))
    test_text.close()


if __name__ == '__main__':
    crime = 'traffic'
    text_name = '001.txt'
    single_test(crime, text_name)
