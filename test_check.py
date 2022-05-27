# -*- coding: UTF-8 -*-
"""
@author:潘越
@file:test_check.py
@time:2022/04/10
"""
import os

from check_tools import check, print_json, print_res_count
from parameters import parse


def test_all():
    print("Start checking all texts...")
    for crime in ['traffic', 'hurt']:
        for txt_name in os.listdir('./test_txt/{0}'.format(crime)):
            print('{0}/{1}: '.format(crime, txt_name), end='')
            try:
                with open('./test_txt/{0}/{1}'.format(crime, txt_name), 'r', encoding='utf-8') as test_text:
                    check_res = check(test_text.read(), crime)
                    print_res_count(check_res)
                    test_text.close()
            except:
                print('  -----ERROR')


def single_test(crime, text_name):
    if not os.path.exists('./test_txt/{0}/{1}.txt'.format(crime, text_name)):
        print("File {0}.txt in {1} does not exist.".format(text_name, crime))
        return
    print("Start checking {0}.txt in {1}...".format(text_name, crime))
    test_text = open('./test_txt/{0}/{1}.txt'.format(crime, text_name), 'r', encoding='utf-8')
    print_json(check(test_text.read(), crime))
    test_text.close()


if __name__ == '__main__':
    args = parse()
    if args.test_all:
        test_all()
    else:
        single_test(args.crime, args.txt_name)
