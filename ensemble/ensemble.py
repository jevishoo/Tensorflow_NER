import os
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append("/home/hezoujie/Competition/CCKS_Military_NER")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config
from train_fine_tune import decode, get_text_and_entity

config = Config()


def get_label(num):
    if num in [2, 3, 4, 5]:
        return "试验要素"
    elif num in [6, 7, 8, 9]:
        return "性能指标"
    elif num in [10, 11, 12, 13]:
        return "系统组成"
    else:
        return "任务场景"


def vote_ensemble(path, dataset, output_path, remove_list):
    single_model_list = [x for x in os.listdir(path) if dataset + '_result_detail' in x]
    print('ensemble from file: ')
    for file_name in single_model_list:
        print(file_name)

    pred_list = OrderedDict()
    ldct_list = []
    text_index = -1  # 保证加入的ldct不是ernie模型的
    for index, file in enumerate(single_model_list):
        if file not in remove_list:  # 预测所有模型
            text_index = index
            print(index)
            print('Text File: ', file)
            break
    print('Ensembling.....')
    for index, file in enumerate(single_model_list):
        if file in remove_list:
            # print('remove file: ', file)
            continue
        print('Ensemble file:', path + file)
        with open(path + file) as f:
            for i, line in tqdm(enumerate(f.readlines())):
                item = json.loads(line)

                if i not in pred_list:
                    pred_list[i] = []
                pred_list[i].append(item['pred'])
                if index == text_index:
                    ldct_list.append(item['ldct_list'])

    # print(len(pred_list))
    # print(len(ldct_list))

    y_pred_list = []
    print('Getting Result.....')
    for key in tqdm(pred_list.keys()):
        pred_key = np.concatenate(pred_list[key])  # 3维
        j = 0
        temp_list = []
        for i in range(config.train_batch_size):
            temp = []
            while True:
                try:
                    temp.append(pred_key[j])
                    j += config.train_batch_size
                except:
                    j = 0
                    j += i + 1
                    break

            temp_T = np.array(temp).T  # 转置
            pred = []
            for line in temp_T:
                pred.append(np.argmax(np.bincount(line)))  # 找出列表中出现次数最多的值
            temp_list.append(pred)
        y_pred_list.append(temp_list)

    ldct_list_tokens = np.concatenate(ldct_list)
    # print(ldct_list)
    ldct_list_text = []
    for tokens in tqdm(ldct_list_tokens):
        text = "".join(tokens)
        ldct_list_text.append(text)

    # 测试集
    # print(y_pred_list)
    # print(ldct_list_tokens)

    y_pred_list, y_pred_entity_list, entity_type_list, pred_start_pos_list, pred_end_pos_list = get_text_and_entity(
        ldct_list_tokens, y_pred_list)
    # print("============================")
    # print(entity_type_list)
    # print("============================")
    y_pred_label_list = [i for i in y_pred_list if i != []]

    label_type_list = []
    for i in range(len(pred_start_pos_list)):
        label_list = []
        for j in range(len(pred_start_pos_list[i])):
            label_type_num = y_pred_label_list[i][pred_start_pos_list[i][j]]
            # print(label_type_num)
            label_type = get_label(label_type_num)
            label_list.append(label_type)
        label_type_list.append(label_list)

    assert len(y_pred_entity_list) == 100
    dict_data = {
        'y_pred_label_list': label_type_list,
        'pred_start_pos_list': pred_start_pos_list,
        'pred_end_pos_list': pred_end_pos_list,
        'y_pred_entity_list': y_pred_entity_list,
        'ldct_list_text': ldct_list_text,
    }
    df = pd.DataFrame(dict_data)
    df = df.fillna("0")
    df.to_csv(output_path + dataset + '_result.csv', encoding='utf-8')


def score_average_ensemble(path, dataset, output_path, remove_list):
    single_model_list = [x for x in os.listdir(path) if dataset + '_result_detail' in x]
    print('ensemble from file: ', len(single_model_list))
    for file_name in single_model_list:
        print(file_name)
    logits_list = OrderedDict()
    trans_list = OrderedDict()
    lengths_list = OrderedDict()
    ldct_list = []

    text_index = -1
    for index, file in enumerate(single_model_list):
        if file not in remove_list:  # 预测所有模型
            text_index = index
            print('Text File: ', file)
            print(text_index)
            break

    for index, file in enumerate(single_model_list):
        if file in remove_list:
            print('remove file: ', file)
            continue
        with open(path + file) as f:
            for i, line in tqdm(enumerate(f.readlines())):
                item = json.loads(line)

                if i not in logits_list:
                    logits_list[i] = []
                    trans_list[i] = []
                    lengths_list[i] = []

                logits_list[i].append(item['logit'])
                trans_list[i].append(item['trans'])
                lengths_list[i].append(item['lengths'])
                if index == text_index:
                    ldct_list.append(item['ldct_list'])

    y_pred_list = []
    for key in tqdm(logits_list.keys()):
        logits_key = logits_list[key]
        logits_key = np.mean(logits_key, axis=0)

        trans_key = np.array(trans_list[key])
        trans_key = np.mean(trans_key, axis=0)

        lengths_key = np.array(lengths_list[key])
        lengths_key = np.mean(lengths_key, axis=0).astype(int)

        pred = decode(logits_key, lengths_key, trans_key)
        y_pred_list.append(pred)

    ldct_list_tokens = np.concatenate(ldct_list)
    ldct_list_text = []

    for tokens in tqdm(ldct_list_tokens):
        text = "".join(tokens)
        ldct_list_text.append(text)

    # 测试集
    print(len(ldct_list_tokens))
    y_pred_list, y_pred_entity_list = get_text_and_entity(ldct_list_tokens, y_pred_list)

    print(len(y_pred_entity_list))
    dict_data = {
        'y_pred_entity_list': y_pred_entity_list,
        'ldct_list_text': ldct_list_text,
    }
    df = pd.DataFrame(dict_data)
    df = df.fillna("0")
    df.to_csv(output_path + 'test_result.csv', encoding='utf-8')


if __name__ == '__main__':
    remove_list = [
        'test_result_detail_model_0.69_0.76_0.7216-448.txt',
        'test_result_detail_model_0.71_0.79_0.7493-896.txt',
        'test_result_detail_model_0.72_0.76_0.7384-1344.txt',
        'test_result_detail_model_0.79_0.84_0.8158-1386.txt',

        'test_result_detail_model_0.85_0.87_0.8571-1584.txt',
        'test_result_detail_model_0.88_0.92_0.8974-2376.txt',

        'test_result_detail_model_0.77_0.77_0.7674-297.txt'
    ]
    # 测试集
    # score_average_ensemble(config.ensemble_source_file, 'test', config.ensemble_result_file, remove_list)
    vote_ensemble(config.ensemble_source_file, 'test', config.ensemble_result_file, remove_list)

    # 验证集
    # vote_ensemble(config.ensemble_source_file, 'dev', config.ensemble_result_file, remove_list)
    # score_average_ensemble(config.ensemble_source_file, 'dev', config.ensemble_result_file, remove_list)
