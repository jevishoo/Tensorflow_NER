# -*- coding: utf-8 -*-
"""
    @author: Jevis-Hoo
    @Date: 2020/5/27 9:29
    @Description: 
"""

import os
import pandas as pd
import sys

sys.path.append("/home/hezoujie/Competition/CCKS_Military_NER")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块

from config import Config


class Evaluate:

    def __init__(self, true_file_path, predict_file_path):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.label_list = ['试验要素', '性能指标', '系统组成', '任务场景']
        self.true_file = true_file_path
        self.predict_file = predict_file_path

    def evaluate(self, category):
        true_file = pd.read_csv(self.true_file)
        predict_file = pd.read_csv(self.predict_file)
        category_true_list = []
        category_predict_list = []

        len_true = 0  # TP+FN 所有正确样本数
        for i in range(true_file.shape[0]):
            category_true = []
            entity_list = true_file['entities'][i].split(";")
            true_label_type = eval(true_file['label_type'][i])
            for j in range(len(true_label_type)):
                if true_label_type[j] == category:
                    category_true.append(entity_list[j])
                    len_true += 1
            category_true_list.append(category_true)

            category_predict = []
            entity_list = predict_file['y_pred_entity_list'][i].split(";")
            predict_label_type = eval(predict_file['y_pred_label_list'][i])
            for j in range(len(entity_list)):
                if predict_label_type[j] == category:
                    category_predict.append(entity_list[j])
            category_predict_list.append(category_predict)

        # 求准确率
        len_predict_true = 0  # TP 所有预测样本和正确样本交集数(预测正确数)
        len_predict = 0  # TP+FP 所有预测样本数
        error_predict = []  # 预测错误的集合
        for i in range(len(category_predict_list)):
            predict_true = set(category_predict_list[i]) & set(category_true_list[i])
            len_predict_true += len(predict_true)

            category_predict = set(category_predict_list[i])
            len_predict += len(category_predict)

            for e in category_predict_list[i]:
                if (e not in category_true_list[i]):
                    error_predict.append(e)
        if (len_predict > 0):
            precision = len_predict_true * 1.0 / len_predict
        else:
            precision = -1

        # 求召回率
        not_predict = []  # 遗漏预测的集合
        for i in range(len(category_predict_list)):
            predict_true = set(category_predict_list[i]) & set(category_true_list[i])
            for i_j in category_true_list[i]:
                if (i_j not in predict_true):
                    not_predict.append(i_j)
        if (len_true > 0):
            recall = len_predict_true / len_true
        else:
            recall = -1

        # 求F1
        if (precision + recall > 0):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = -1

        return [precision, recall, f1, error_predict, not_predict]

    def evaluate_main(self):
        result = []
        for label_type in self.label_list:
            c_res = self.evaluate(label_type)
            p = c_res[0]
            r = c_res[1]
            f1 = c_res[2]
            error_predict = c_res[3]
            not_predict = c_res[4]
            result.append([label_type, p, r, f1, error_predict, not_predict])
        r_df = pd.DataFrame(result, columns=['category', 'precision', 'recall', 'F1', 'error_predict', 'not_predict'])
        return r_df


def main():
    config = Config()

    true_path = config.new_data_process_quarter_final + 'new_dev.csv'
    predict_path = config.ensemble_result_file + 'dev_result.csv'

    Eva = Evaluate(true_path, predict_path)
    result = Eva.evaluate_main()
    result.to_csv(config.new_data_process_quarter_final + 'performance.csv', encoding='utf-8')


if __name__ == "__main__":
    main()
