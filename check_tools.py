# -*- coding: UTF-8 -*-
"""
@author:潘越
@file:check_tools.py
@time:2022/04/13
"""

import json
import re
import os

from transformers import BertTokenizer

from parameters import parse
from data_process import get_article_dict
from models import *

# 段落状态
DSR = 'dsr'
SSJL = 'ssjl'
AJJBQK = 'ajjbqk'
CPFXGC = 'cpfxgc'
PJJG = 'pjjg'

# 当事人正则匹配
DSR_PATTERNS = [re.compile(r'^公诉机关'), re.compile(r'^被告'), re.compile(r'^(指定)*辩护人'), re.compile(r'^翻译'),
                re.compile(r'^委托代理人'), re.compile(r'^(刑事)*附带'), re.compile(r'^被害人'), re.compile(r'^(诉讼)*代表人'),
                re.compile(r'^(主要)*负责人'), re.compile(r'^法定'), re.compile(r'^(共同)*诉讼代理人'), re.compile(r'^以*上[^海虞]'),
                re.compile(r'^原告'), re.compile(r'^[一二三四五六七八九十]')]
# 法条条目正则匹配
NUMBER_PATTERN = re.compile(r'第?.*条')

# 中文数字词汇表
CH_NUMBERS_VOCAB = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
# 法条条目词汇表
ARTICLE_NUMBER_VOCAB = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九',
                        '百', '十', '千', '第', '条', '款', '项', '（', '）']
# 法条间连接词汇表
CONJUNCTION_VOCAB = ['、', '，', '和', '与', '及']

# 法条->数据库id字典
ARTICLE_SQLID_DICT = {}
MAX_SQLID = 0
with open('./dict/article_sqlId_dict.json', 'r', encoding='utf-8') as article_sqlId_dict_json:
    ARTICLE_SQLID_DICT = json.load(article_sqlId_dict_json)
    MAX_SQLID = len(ARTICLE_SQLID_DICT.keys())
    article_sqlId_dict_json.close()
ARTICLE_SQLID_DICT_KEYS = ARTICLE_SQLID_DICT.keys()


def split_paragraph(paragraphs):
    dsr_paragraphs, ssjl_paragraphs, ajjbqk_paragraphs, cpfxgc_paragraphs, pjjg_paragraphs = [], [], [], [], []
    cur_state = DSR  # 开头是当事人段

    # 按逻辑段拆分
    for paragraph in paragraphs:
        # 处理自然段
        if cur_state == DSR:  # 当事人段
            tmp_paragraph = paragraph
            while tmp_paragraph[0] in CH_NUMBERS_VOCAB:  # 是数字，去掉开头再匹配
                tmp_paragraph = tmp_paragraph[1:]
            for dsr_pattern in DSR_PATTERNS:
                if dsr_pattern.match(tmp_paragraph):
                    dsr_paragraphs.append(tmp_paragraph)
                    break
            else:  # 开始诉讼记录段
                cur_state = SSJL
        if cur_state == SSJL:
            if re.match(r'^经审理查明', paragraph) or len(ssjl_paragraphs) > 0:  # 开始案件基本情况段
                cur_state = AJJBQK
            else:
                ssjl_paragraphs.append(paragraph)
        if cur_state == AJJBQK:
            if re.match(r'^本院认为', paragraph):  # 开始裁判分析过程段
                cur_state = CPFXGC
            else:
                ajjbqk_paragraphs.append(paragraph)
        if cur_state == CPFXGC:
            cpfxgc_paragraphs.append(paragraph)
            if re.match(r'.*判决如下[：；].*', paragraph) or re.match(r'.*如下判决[：；].*', paragraph):  # 下一段开始是判决结果段
                paragraph = paragraph.replace('判决如下；', '判决如下：').replace('如下判决；', '如下判决：')
                if not (paragraph.endswith('判决如下：') or paragraph.endswith('如下判决：')):
                    pjrx_idx = paragraph.find('判决如下：')
                    if pjrx_idx == -1:
                        pjrx_idx = paragraph.find('如下判决：')
                    cpfxgc_paragraphs[-1] = paragraph[:pjrx_idx + 5]
                    pjjg_paragraphs.append(paragraph[pjrx_idx + 5:])
                cur_state = PJJG
                continue
        if cur_state == PJJG:
            pjjg_paragraphs.append(paragraph)

    return dsr_paragraphs, ssjl_paragraphs, ajjbqk_paragraphs, cpfxgc_paragraphs, pjjg_paragraphs


def predict_fact_map(ajjbqk_paragraphs, crime):
    '''
    案件基本情况段 ajjbqk
    '''
    fact_map = []  # fact_map每项为一个自然段转换成的{sentence:句子,articleId:法条id列表}列表
    for ajjbqk_paragraph in ajjbqk_paragraphs:
        sentences = ajjbqk_paragraph.split('。')
        full_stop = False
        if sentences[-1] == '':  # 不能直接去掉最后一个，有标点错误没打句号的
            sentences.pop()
            full_stop = True
        for sentence in sentences:
            fact_map.append({
                'sentence': sentence + '。',  # 每个都补上句号，虽然有出现逗号加句号的情况
                'articleId': []
            })
        if not full_stop:  # 原文非句号结尾，去掉补充的句号
            fact_map[-1]['sentence'] = fact_map[-1]['sentence'][:-1]

    # 加载模型
    args = parse()
    model = ThreeLayers(args)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(
        torch.load("./checkpoints/{0}/ThreeLayers".format(crime), map_location=torch.device(device)))
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_name)

    # 加载文本处理数据
    article_dict = get_article_dict(args)

    # 开始预测每句事实所需法条
    for fact_item in fact_map:
        fact = fact_item['sentence']
        fact = tokenizer(fact, return_tensors='pt', padding='max_length', truncation=True, max_length=args.fact_len + 2)
        fact_ii = fact['input_ids'].to(device)
        fact_am = fact['attention_mask'].to(device)

        for article_idx in article_dict:
            article = article_dict[article_idx]
            article = tokenizer(article, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=args.article_len + 2)
            article['input_ids'] = torch.cat(
                [torch.full([article['input_ids'].size(0), 1], 2), article['input_ids'][:, 1:]], dim=-1)
            article_ii = article['input_ids'].to(device)
            article_am = article['attention_mask'].to(device)

            pred = model(fact_ii, fact_am, article_ii, article_am)
            pred = torch.nn.functional.softmax(pred, dim=-1)[:, 1]
            if torch.ge(pred, 0.6):
                fact_item['articleId'].append(article_idx)

    return fact_map


def get_article_map(cpfxgc_paragraphs):
    article_map = []
    for cpfxgc_paragraph in cpfxgc_paragraphs:
        cur_number = ''  # 条数
        for i in range(len(cpfxgc_paragraph)):
            if cpfxgc_paragraph[i] == '《':
                cur_law = ''
                i += 1
                while cpfxgc_paragraph[i] != '》':
                    cur_law += cpfxgc_paragraph[i]
                    i += 1
                article_map.append({
                    'name': cur_law,
                    'numbers': []
                })
                # 现在cpfxgc_paragraph[i]是》
                i += 1  # 第
                in_brackets = False
                while True:
                    if cpfxgc_paragraph[i] in ARTICLE_NUMBER_VOCAB or in_brackets:  # 当前字符符合词汇表或在括号中
                        if cpfxgc_paragraph[i] == '（':
                            in_brackets = True
                        elif cpfxgc_paragraph[i] == '）':
                            in_brackets = False
                        cur_number += cpfxgc_paragraph[i]
                    else:  # 不属于词汇表
                        if cur_number != '':
                            # 寻找cur_number中的“条”字
                            tiao_idx = cur_number.find('条')
                            if tiao_idx != -1:
                                article_map[-1]['numbers'].append(cur_number[:tiao_idx + 1])
                            cur_number = ''
                        if cpfxgc_paragraph[i] not in CONJUNCTION_VOCAB:  # 后面无了
                            break
                    i += 1

    return article_map


def add_relations(sqlId_lawx_map, text_list):
    # 开始处理各个relations
    sqlId_lawx_map_keys = sqlId_lawx_map.keys()
    lawx_factx_map = {}  # lawx有哪些factx引用过
    for text_item in text_list:
        if text_item['type'] == 1:  # Fact
            new_relations = []
            text_item['needs'] = []
            for relation in text_item['relations']:
                if relation < MAX_SQLID:
                    if str(relation) in sqlId_lawx_map_keys:
                        lawx = sqlId_lawx_map[str(relation)]
                        new_relations.append(lawx)
                        if lawx not in lawx_factx_map.keys():
                            lawx_factx_map[lawx] = []
                        lawx_factx_map[lawx].append(text_item['id'])
                    else:
                        text_item['needs'].append(relation)
                else:
                    text_item['needs'].append(relation)
            text_item['relations'] = new_relations
        elif text_item['type'] == 2:  # Law
            if text_item['id'] in lawx_factx_map:
                text_item['relations'] = lawx_factx_map[text_item['id']]
                text_item['relations'].append('conclusion-1')
        elif text_item['type'] == 3:  # Conclusion
            text_item['relations'] = list(lawx_factx_map.keys())

    return text_list


def check(text, crime):
    # 全文拆分
    paragraphs = [i.strip() for i in text.split('\n')]

    # 整篇文书分为 当事人dsr、诉讼记录ssjl、案件基本情况ajjbqk、裁判分析过程cpfxgc、判决结果pjjg
    dsr_paragraphs, ssjl_paragraphs, ajjbqk_paragraphs, cpfxgc_paragraphs, pjjg_paragraphs = split_paragraph(paragraphs)

    # 处理逻辑段中的自然段
    # 当事人段&诉讼记录段 dsr&ssjl：不用操作，都是普通文本

    # 案件基本情况段 ajjbqk
    # fact_map：[{sentence:句子,articleId:法条id列表}, ...]
    fact_map = predict_fact_map(ajjbqk_paragraphs, crime)

    # 裁判分析过程段 cpfxgc
    # article_map：[{name:法律名字,numbers:[条目]}, ...]
    article_map = get_article_map(cpfxgc_paragraphs)

    # 判决结果段 pjjg：不用处理，最后直接塞进去

    # 需要的信息提取完了，开始生成结果
    # 整篇文书分为 当事人dsr、诉讼记录ssjl、案件基本情况ajjbqk、裁判分析过程cpfxgc、判决结果pjjg
    text_list = []
    text_count, fact_count, law_count, conclusion_count = 1, 1, 1, 1

    cur_content = '\n'.join(dsr_paragraphs + ssjl_paragraphs)  # 当事人段和诉讼记录段必为普通文本

    # fact_map：[{sentence:事实句,articleId:[法条id列表]}]
    fact_idx = 0
    for ajjbqk_paragraph in ajjbqk_paragraphs:
        cur_content += '\n'
        cur_idx = 0
        while ajjbqk_paragraph.find(fact_map[fact_idx]['sentence'], cur_idx, len(ajjbqk_paragraph)) != -1:
            fact_start_idx = ajjbqk_paragraph.find(fact_map[fact_idx]['sentence'])  # 事实句起始idx
            cur_content += ajjbqk_paragraph[cur_idx:fact_start_idx]  # 把上次末尾idx到这次事实起始idx的都塞进普通文本
            text_list.append({
                'id': 'text-{0}'.format(text_count),
                'content': cur_content,
                'type': 0
            })
            text_count += 1
            cur_content = ''
            # 插入事实Fact
            if len(fact_map[fact_idx]['articleId']) > 0:
                text_list.append({
                    'id': 'fact-{0}'.format(fact_count),
                    'content': fact_map[fact_idx]['sentence'],
                    'type': 1,
                    'count': fact_count,
                    'relations': fact_map[fact_idx]['articleId']  # 先用sqlId，最后确定sqlId对应law-x后再重新处理
                })
                fact_count += 1
            else:
                text_list[-1]['content'] += fact_map[fact_idx]['sentence']
            cur_idx = fact_start_idx + len(fact_map[fact_idx]['sentence'])
            fact_idx += 1
            if cur_idx >= len(ajjbqk_paragraph) or fact_idx >= len(fact_map):
                break
        if cur_idx < len(ajjbqk_paragraph):
            cur_content += ajjbqk_paragraph[cur_idx:]

    # 裁判分析过程段 cpfxgc
    # article_map：[{name:法律名字,numbers:[条数]}]
    article_idx = 0
    sqlId_lawx_map = {}  # sqlId映射到law-x
    for cpfxgc_paragraph in cpfxgc_paragraphs:
        cur_content += '\n'
        cur_idx = 0
        while cpfxgc_paragraph.find(article_map[article_idx]['name'], cur_idx, len(cpfxgc_paragraph)) != -1:
            article_name_start_idx = cpfxgc_paragraph.find(article_map[article_idx]['name'], cur_idx,
                                                           len(cpfxgc_paragraph)) - 1  # 减1算上书名号
            cur_content += cpfxgc_paragraph[cur_idx:article_name_start_idx]  # 把上次末尾idx到这次法律标题起始idx的都塞进普通文本
            text_list.append({
                'id': 'text-{0}'.format(text_count),
                'content': cur_content,
                'type': 0
            })
            text_count += 1
            cur_content = ''
            cur_idx = article_name_start_idx
            # 开始添加条数
            if len(article_map[article_idx]['numbers']) > 0:
                name = article_map[article_idx]['name']
                key_name = ("最高人民法院" if name.endswith("解释") and not name.startswith("最高人民法院") else "") + name
                key = "《{0}》{1}".format(key_name, NUMBER_PATTERN.findall(article_map[article_idx]['numbers'][0])[0])
                if key in ARTICLE_SQLID_DICT_KEYS:
                    sql_id = ARTICLE_SQLID_DICT[key]
                    sqlId_lawx_map[str(sql_id)] = "law-{0}".format(law_count)
                else:
                    sql_id = -1
                text_list.append({
                    'id': 'law-{0}'.format(law_count),
                    'content': '《{0}》{1}'.format(name, article_map[article_idx]['numbers'][0]),
                    'type': 2,
                    'articleId': sql_id,
                    'relations': []  # 空着，后面统一处理
                })
                law_count += 1
                cur_idx += len(name) + 2 + len(article_map[article_idx]['numbers'][0])
                for number in article_map[article_idx]['numbers'][1:]:
                    # 先处理两条法条中间的普通文本
                    number_start_idx = cpfxgc_paragraph.find(number, cur_idx, len(cpfxgc_paragraph))
                    text_list.append({
                        'id': 'text-{0}'.format(text_count),
                        'content': cpfxgc_paragraph[cur_idx:number_start_idx],
                        'type': 0
                    })
                    text_count += 1
                    # 塞法条
                    key_number = NUMBER_PATTERN.findall(number)[0]
                    key_number = ('' if key_number.startswith('第') else '第') + key_number
                    key = "《{0}》{1}".format(key_name, key_number)
                    if key in ARTICLE_SQLID_DICT_KEYS:
                        sql_id = ARTICLE_SQLID_DICT[key]
                        sqlId_lawx_map[str(sql_id)] = "law-{0}".format(law_count)
                    else:
                        sql_id = -1
                    text_list.append({
                        'id': 'law-{0}'.format(law_count),
                        'content': number,
                        'type': 2,
                        'articleId': sql_id,
                        'relations': []  # 空着，后面统一处理
                    })
                    law_count += 1
                    cur_idx = number_start_idx + len(number)
            else:  # 莫得具体条目，普通文本处理，塞进上一条普通文本
                text_list[-1]['content'] += "《{0}》".format(article_map[article_idx]['name'])
                cur_idx += len(article_map[article_idx]['name']) + 2
            article_idx += 1
            if cur_idx >= len(cpfxgc_paragraph) or article_idx >= len(article_map):
                break
        if cur_idx <= len(cpfxgc_paragraph):
            cur_content += cpfxgc_paragraph[cur_idx:]

    # pjjg段
    if cur_content != '':
        text_list.append({
            'id': 'text-{0}'.format(text_count),
            'content': cur_content,
            'type': 0
        })
        text_count += 1
    text_list[-1]['content'] += "\n"
    conclusion_paragraph_count = 0
    for pjjg_paragraph in pjjg_paragraphs:
        if not re.match(r'^如不服', pjjg_paragraph):
            conclusion_paragraph_count += 1
        else:
            break
    text_list.append({
        'id': 'conclusion-1',
        'content': '{0}'.format('\n'.join(pjjg_paragraphs[:conclusion_paragraph_count])),
        'type': 3,
        'count': 1,
        'relations': []  # 后面处理
    })
    if conclusion_paragraph_count < len(pjjg_paragraphs):
        text_list.append({
            'id': 'text-{0}'.format(text_count),
            'content': '\n{0}'.format('\n'.join(pjjg_paragraphs[conclusion_paragraph_count:])),
            'type': 0,
        })
        text_count += 1

    # 补充relations
    text_list = add_relations(sqlId_lawx_map, text_list)

    return text_list  # 结果


def print_res_count(res):
    count = [0, 0, 0, 0]
    for item in res:
        if not item['content'] == '':
            count[item['type']] += 1
    print('事实{0}条，引用法条{1}条，结论{2}条'.format(count[1], count[2], count[3]))


def print_json(data):
    print(json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))


if __name__ == '__main__':
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
