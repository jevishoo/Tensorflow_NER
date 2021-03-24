"""
    @author: Jevis-Hoo
    @Date: 2020/5/21 21:10
    @Description: 
"""

import pandas as pd
import numpy as np
import codecs
import re
import json
import sys

# 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
sys.path.append("/home/hezoujie/Competition/CCKS_Military_NER")
from config import Config
from .data_consistent import DataConsistent
from .transfer_data import TransferData

"""
按照标点符号切割的预处理数据
"""


def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 1:
        return index_list
    else:
        return -1


def load_data(df):
    label_type_list = []
    start_list = []
    end_list = []
    entity_list = []
    max_len = 0
    len_list = []

    for i in df.index:
        s_list = []
        e_list = []
        l_list = []
        sub_ent_list = []
        ent_list = []

        # if len(df['originalText'][i]) == 400:
        #     print(df['originalText'][i])
        len_list.append(len(df['originalText'][i]))

        if len(df['originalText'][i]) > max_len:
            max_len = len(df['originalText'][i])

        # 提取所有实体
        for j in range(len(df['entities'][i])):
            start_pos = df['entities'][i][j]['start_pos']
            end_pos = df['entities'][i][j]['end_pos']
            entity = df['originalText'][i][start_pos - 1:end_pos]
            sub_ent_list.append("".join(entity))

        for j in range(len(df['entities'][i])):
            start_pos = df['entities'][i][j]['start_pos']
            end_pos = df['entities'][i][j]['end_pos']

            # 重复实体位置的加入
            entity = df['originalText'][i][start_pos - 1:end_pos]
            index = find_all(entity, df['originalText'][i])

            # 防止实体是另外实体的子串
            is_sub_entity = False
            for k in range(len(sub_ent_list)):
                if sub_ent_list[k].find(entity) != -1 and sub_ent_list[k] != entity:
                    is_sub_entity = True
                    break

            # 这里的判断有问题(当有重复实体，是子串时，可能忽略掉，造成少添加的后果)
            if index != -1 and (not is_sub_entity):  # 有重复实体，不是子串，全部加入
                length = end_pos - start_pos
                for en_index in range(len(index)):
                    s_list.append(index[en_index] + 1)
                    e_list.append(index[en_index] + 1 + length)
                    l_list.append(df['entities'][i][j]['label_type'])
                    ent_list.append("".join(entity))
            else:
                s_list.append(start_pos)
                e_list.append(end_pos)
                l_list.append(df['entities'][i][j]['label_type'])
                ent_list.append("".join(entity))

        label_type_list.append(l_list)
        start_list.append(s_list)
        end_list.append(e_list)
        entity_list.append(";".join(ent_list))

    print("平均长度是：" + str(np.mean(len_list)))
    return label_type_list, start_list, end_list, entity_list, max_len


def main():
    """
    train_doc_path = config.source_data_dir + 'train/'
    train_output_path = data_dir + 'mid_train.txt'

    DC = DataConsistent(train_doc_path, train_output_path)
    DC.main_consistent()

    train_doc_path = config.source_data_dir + 'validate_data.json'
    train_output_path = data_dir + 'mid_test.txt'

    DC = DataConsistent(train_doc_path, train_output_path)
    DC.test_consistent()

    data_list = []
    for line in open(data_dir + 'mid_train.txt', 'r', encoding='utf-8'):
        data = eval(line)
        data_list.append(data)

    data_df = pd.DataFrame(data_list)

    # 切分训练集，分成训练集和验证集，尝试8折切割
    train_df = data_df.sample(frac=0.9)
    # train_df = data_df.sample(frac=0.9875)
    row_list = []
    for index in train_df.index:
        row_list.append(index)
    dev_df = data_df.drop(row_list, axis=0)

    train_label_type_list, train_start_list, train_end_list, train_entity_list, train_max_len = load_data(train_df)
    train_df['entities'] = train_entity_list
    train_df['label_type'] = train_label_type_list
    train_df['start_pos'] = train_start_list
    train_df['end_pos'] = train_end_list

    dev_label_type_list, dev_start_list, dev_end_list, dev_entity_list, dev_max_len = load_data(dev_df)
    dev_df['entities'] = dev_entity_list
    dev_df['label_type'] = dev_label_type_list
    dev_df['start_pos'] = dev_start_list
    dev_df['end_pos'] = dev_end_list

    entity_list = []
    entity_list.extend(train_entity_list)
    entity_list.extend(dev_entity_list)
    # print(entity_list)
    dict_entity = {"entity_list": entity_list}
    entity_df = pd.DataFrame(dict_entity)
    entity_df.to_csv(data_dir + 'entity.csv', encoding='utf-8')

    # 测试集
    test_list = []
    len_test = []
    max_len = max(train_max_len, dev_max_len)

    for line in open(data_dir + 'mid_test.txt', 'r', encoding='utf-8'):
        data = eval(line)
        test_list.append(data)
        len_test.append(len(data['originalText']))
        if len(data['originalText']) > max_len:
            max_len = len(data['originalText'])

    print(len_test)
    print('max len: ' + str(max_len))
    test_df = pd.DataFrame(test_list)

    # 找出所有的非中文、非英文和非数字符号
    additional_chars = set()
    for t in list(test_df.originalText) + list(train_df.originalText):
        additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(t)))
    # 一些需要保留的符号
    extra_chars = set("!#$%&\()*+,-.\'\"/:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】 ⅣⅰⅱⅲⅳⅢⅡⅠ—℃°")

    additional_chars = additional_chars.difference(extra_chars)
    print("特殊符号：", additional_chars)

    print('Train Set Size:', train_df.shape)
    print('Train Set Size:', dev_df.shape)

    train_text_list = train_df['originalText'].tolist()
    train_label_list = train_df['entities'].tolist()

    dev_text_list = dev_df['originalText'].tolist()
    dev_label_list = dev_df['entities'].tolist()

    test_text_list = test_df['originalText'].tolist()

    train_dict = {'originalText': train_text_list, 'entities': train_label_list,
                  'label_type': train_label_type_list, 'start_pos': train_start_list,
                  'end_pos': train_end_list}

    train_df = pd.DataFrame(train_dict)
    train_df.to_csv(data_dir + 'new_train.csv')

    dev_dict = {'originalText': dev_text_list, 'entities': dev_label_list,
                'label_type': dev_label_type_list, 'start_pos': dev_start_list,
                'end_pos': dev_end_list}
    dev_df = pd.DataFrame(dev_dict)
    dev_df.to_csv(data_dir + 'new_dev.csv')

    test_dict = {'originalText': test_text_list}
    test_df = pd.DataFrame(test_dict)

    print('训练集:', train_df.shape)
    print('验证集:', dev_df.shape)
    print('测试集:', test_df.shape)

    with codecs.open(data_dir + 'test.txt', 'w', encoding='utf-8') as up:
        for row in test_df.iloc[:].itertuples():

            text_lbl = row.originalText
            for c1 in text_lbl:
                up.write('{0} {1}\n'.format(c1, 'O'))

            up.write('\n')
    """
    train_doc_path = data_dir + 'new_train.csv'
    train_bio_output_path = data_dir + 'train_iob.txt'
    handler = TransferData(train_doc_path, train_bio_output_path)
    handler.transfer()

    # train_iobes_output_path = data_dir + 'train.txt'
    # handler.update_tag_scheme(train_bio_output_path, train_iobes_output_path)

    dev_doc_path = config.new_data_process_quarter_final + 'new_dev.csv'
    dev_bio_output_path = data_dir + 'dev_iob.txt'

    handler = TransferData(dev_doc_path, dev_bio_output_path)
    handler.transfer()

    # dev_iobes_output_path = data_dir + 'dev.txt'
    # handler.update_tag_scheme(dev_bio_output_path, dev_iobes_output_path)
    print("数据转换成功")


if __name__ == "__main__":
    config = Config()
    len_treshold = 358  # 每条数据的最大长度, 留下两个位置给[CLS]和[SEP]
    data_dir = config.new_data_process_quarter_final
    print(data_dir)

    main()
