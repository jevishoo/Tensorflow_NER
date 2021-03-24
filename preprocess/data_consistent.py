"""
    @author: Jevis-Hoo
    @Date: 2020/5/20 21:23
    @Description: 
"""

# 数据处理第一步
# 数据转换（将originalText中的中文标点符号全部转换成英文，大写英文全部转换成小写英文）
import os
import json
import re
import sys

# 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
sys.path.append("/home/hezoujie/Competition/CCKS_Military_NER")
from config import Config


class DataConsistent:
    def __init__(self, original_doc_path, output_doc_path):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.doc_path = original_doc_path
        self.output_path = output_doc_path

    # 全角转半角
    def strQ2B(self, sent: str) -> str:
        ss = []
        for s in sent:
            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 全角空格直接转换
                    inside_code = 32
                elif inside_code == 12290:
                    inside_code = 12290
                elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                    inside_code -= 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        res = ''.join(ss)
        return res

    def lower(self, sent: str) -> str:
        sent_2 = sent.lower()
        return sent_2

    def stop_words(self, sent: str) -> str:
        sent_1 = sent.replace('\x1a', '?')
        # sent_2 = sent_1.rstrip('\r\n')
        sent_2 = sent_1
        sent_3 = sent_2.replace('“', '"')
        sent_4 = sent_3.replace('”', '"')
        sent_5 = sent_4.replace('—', '-')
        sent_6 = sent_5.replace('?', ',')

        return sent_6

    """
        1.本次文本中"?"并没有"?"功能，而是连接和结束 --> 故需要修改为"-","。"
        2.","修改为"，"
        3."."在中文之间修改为"。"其余位置不用修改
    """

    def process_special_char(self, sent):
        sent_list = list(sent)
        for i in range(len(sent_list)):
            if sent_list[i] == ",":
                sent_list[i] = '，'

            elif sent_list[i] == ".":
                if i == len(sent) - 1:
                    sent_list[i] = '。'
                else:
                    if (47 < ord(sent_list[i - 1]) < 58) or (
                            96 < ord(sent_list[i - 1]) < 123) or (
                            47 < ord(sent_list[i + 1]) < 58) or (
                            96 < ord(sent_list[i + 1]) < 123) or (
                            "(" in sent_list[:i] and ")" in sent_list[i:]
                    ):
                        sent_list[i] = '.'
                    else:
                        sent_list[i] = '。'

            elif sent_list[i] == "?":
                if i == len(sent) - 1:
                    sent_list[i] = '。'
                else:
                    if (47 < ord(sent_list[i - 1]) < 58) or (
                            96 < ord(sent_list[i - 1]) < 123) or (
                            47 < ord(sent_list[i + 1]) < 58) or (
                            96 < ord(sent_list[i + 1]) < 123):
                        sent_list[i] = '-'

                    elif sent_list[i + 1] == "。":
                        sent_list[i] = '。'

                    else:
                        sent_list[i] = '，'

        process_text = ''.join(sent_list)

        return process_text

    def main_consistent(self):
        filelists = os.listdir(self.doc_path)

        sort_num_first = []
        for file in filelists:
            sort_num_first.append(int(file.split("_")[2].split(".")[0]))  # 根据 _ 分割，然后根据空格分割，转化为数字类型
            sort_num_first.sort()

        sorted_file = []
        for sort_num in sort_num_first:
            for file in filelists:
                if str(sort_num) == file.split("_")[2].split(".")[0]:
                    sorted_file.append(file)

        f_1 = open(self.output_path, 'w', encoding='utf-8')
        for file in sorted_file:
            filepath = os.path.join(self.doc_path, file)
            with open(filepath, 'r', encoding='gbk') as f:
                row_dict = json.load(f)
                row_dict_1 = dict()
                original_text = row_dict['originalText']

                tokens = list(original_text)
                assert len(original_text) == len(tokens)
                entities = []

                for e in row_dict['entities']:
                    # tmp_tokens = tokens  #保证处理过程中tokens不被修改
                    a = e['start_pos']  #
                    b = e['end_pos']
                    label = e['label_type']
                    overlap = e["overlap"]
                    cut = 0
                    for t in range(len(tokens)):
                        if t == e['start_pos'] - 1:  # 因为是从1开始记索引 到了a位
                            a -= cut
                        if t == e['end_pos'] - 1:
                            b -= cut
                            break  # 对应的每个实体位置在这里重新校正完毕
                        tt = tokens[t]
                        if tt == " ":  # 如果是空格则对应的标签索引前面减去1 对应位置取消
                            cut += 1
                            # tokens[t] = ''
                    entities.append({
                        "label_type": label,
                        "overlap": overlap,
                        "start_pos": a,
                        "end_pos": b
                    })
                # print(len(tokens))
                tokens = [i for i in tokens if i != ' ']  # 最后把空格筛除
                text = ''.join(tokens)  # 相当于自动把空格部分干掉了

                original_text_2 = self.strQ2B(text)
                original_text_3 = self.lower(original_text_2)
                original_text_4 = self.stop_words(original_text_3)
                # original_text_5 = self.process_special_char(original_text_4)

                row_dict_1['originalText'] = original_text_4
                row_dict_1['entities'] = entities

                line_1 = json.dumps(row_dict_1, ensure_ascii=False)
                f_1.write(line_1 + '\n')
            f.close()
        f_1.close()

    def test_consistent(self):
        f_1 = open(self.output_path, 'w', encoding='utf-8')
        with open(self.doc_path, 'r', encoding='utf-8') as f:
            row_dict = json.load(f)
            row_dict_1 = dict()
            for i in range(len(row_dict)):
                original_text = row_dict[list(row_dict.keys())[i]]

                tokens = list(original_text)
                assert len(original_text) == len(tokens)

                tokens = [i for i in tokens if i != ' ']  # 最后把空格筛除
                text = ''.join(tokens)  # 相当于自动把空格部分干掉了

                original_text_2 = self.strQ2B(text)
                original_text_3 = self.lower(original_text_2)
                original_text_4 = self.stop_words(original_text_3)
                # original_text_5 = self.process_special_char(original_text_4)

                row_dict_1['originalText'] = original_text_4
                line_1 = json.dumps(row_dict_1, ensure_ascii=False)
                f_1.write(line_1 + '\n')
        f_1.close()
        f.close()


if __name__ == '__main__':
    config = Config()
    data_dir = config.new_data_process_quarter_final

    train_doc_path = config.source_data_dir + 'train/'
    train_output_path = config.new_data_process_quarter_final + 'mid_train.txt'

    DC = DataConsistent(train_doc_path, train_output_path)
    DC.main_consistent()
