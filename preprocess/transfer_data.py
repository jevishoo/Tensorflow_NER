"""
    @author: Jevis-Hoo
    @Date: 2020/5/20 21:41
    @Description:
"""
import pandas as pd
import sys
import re

# 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
sys.path.append("/home/hezoujie/Competition/CCKS_Military_NER")
from config import Config


class TransferData:
    def __init__(self, doc_path, out_path):
        self.label_dict = {
            '试验要素': 'ELE',
            '性能指标': 'IND',
            '系统组成': 'CON',
            '任务场景': 'SIT',
        }
        self.cate_dict = {
            'O': 1,
            'ELE-I': 2,
            'ELE-B': 3,
            'ELE-E': 4,
            'ELE-S': 5,
            'IND-I': 6,
            'IND-B': 7,
            'IND-E': 8,
            'IND-S': 9,
            'CON-I': 10,
            'CON-B': 11,
            'CON-E': 12,
            'CON-S': 13,
            'SIT-I': 14,
            'SIT-B': 15,
            'SIT-E': 16,
            'SIT-S': 17,
        }
        self.doc_path = doc_path
        self.out_path = out_path

    def transfer(self):
        with open(self.out_path, 'w+', encoding='utf-8') as up:
            data = pd.read_csv(self.doc_path, encoding='utf-8')
            for i in range(data.shape[0]):
                content = data["originalText"][i].strip()
                res_dict = {}
                length = len(eval(data["start_pos"][i]))

                # 发现训练数据中有Entities无标签数据，在生成train数据时直接舍弃
                if length != 0:
                    for j in range(length):
                        # print(eval(data['start_pos'][i]))
                        start = eval(data['start_pos'][i])[j] - 1
                        end = eval(data['end_pos'][i])[j] - 1
                        label = eval(data['label_type'][i])[j]
                        # print(label)
                        # print(content[start:end])

                        label_id = self.label_dict.get(label)

                        for m in range(start, end + 1):
                            if m == start:
                                label_cate = 'B-' + label
                                # label_cate = label_id + '-B'
                            else:
                                label_cate = 'I-' + label
                                # label_cate = label_id + '-I'
                            res_dict[m] = label_cate

                for index, char in enumerate(content):
                    char_label = res_dict.get(index, 'O')
                    # print(char, char_label)
                    up.write(char + ' ' + char_label + '\n')
                up.write('\n')
        up.close()

    def zero_digits(self, s):
        """
        把句子中的数字统一用0替换.
        """
        return re.sub('\d', '0', s)

    def load_sentences(self, path):
        """
        加载训练样本，一句话就是一个样本。
        训练样本中，每一行是这样的：长 B-Dur，即字和对应的标签
        句子之间使用空行隔开的
        return : sentences: [[[['无', 'O'], ['长', 'B-Dur'], ['期', 'I-Dur'],...]]
        """

        sentences = []
        sentence = []

        for line in open(path, 'r', encoding='utf8'):

            """ 如果包含有数字，就把每个数字用0替换 """
            # line = line.rstrip()
            # line = self.zero_digits(line)

            """ 如果不是句子结束的换行符，就继续添加单词到句子中 """
            if line != "\n":
                word_pair = ["<unk>", line[2:]] if line[0] == " " else line.split()
                assert len(word_pair) == 2
                sentence.append(word_pair)

            else:
                """ 如果遇到换行符，说明一个句子处理完毕 """
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []

        return sentences

    def iob_iobes(self, tags):
        """
        IOB -> IOBES
        """
        new_tags = []

        for i, tag in enumerate(tags):
            if tag == 'O':
                new_tags.append(tag)
            # elif tag.split('-')[-1] == 'B':
            elif tag.split('-')[0] == 'B':
                # if i + 1 != len(tags) and tags[i + 1].split('-')[-1] == 'I':
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    # new_tags.append(tag.replace('-B', '-S'))
                    new_tags.append(tag.replace('B-', 'S-'))
            # elif tag.split('-')[-1] == 'I':
            elif tag.split('-')[0] == 'I':
                # if i + 1 < len(tags) and tags[i + 1].split('-')[-1] == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    # new_tags.append(tag.replace('-I', '-E'))
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise Exception('Invalid IOB format!')
        return new_tags

    def iobes_iob(self, tags):
        """
        IOBES -> IOB
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag.split('-')[0] == 'B':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'I':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'S':
                new_tags.append(tag.replace('-S', '-B'))
            elif tag.split('-')[0] == 'E':
                new_tags.append(tag.replace('-E', '-I'))
            elif tag.split('-')[0] == 'O':
                new_tags.append(tag)
            else:
                raise Exception('Invalid format!')
        return new_tags

    def update_tag_scheme(self, doc_path, out_path):
        sentences = self.load_sentences(doc_path)
        """ 将IOB格式转化为IOBES格式 """

        with open(out_path, "w") as f:
            for i, s in enumerate(sentences):
                char = [w[0] for w in s]
                tags = [w[-1] for w in s]
                new_tags = self.iob_iobes(tags)
                for j in range(len(new_tags)):
                    f.write(char[j] + ' ' + new_tags[j] + '\n')
                f.write('\n')
        # f1 = open(out_path + "sentences.txt", "w")
        # f2 = open(out_path + "tags.txt", "w")
        # for i, s in enumerate(sentences):
        #     char = [w[0] for w in s]
        #     tags = [w[-1] for w in s]
        #     new_tags = self.iob_iobes(tags)
        #     for j in range(len(new_tags)):
        #         f1.write(char[j] + ' ')
        #         f2.write(new_tags[j] + ' ')
        #     f1.write('\n')
        #     f2.write('\n')


if __name__ == '__main__':
    config = Config()

    # train_IOB_output_path = config.new_data_process_quarter_final + 'train_iob.txt'
    train_IOB_output_path = 'train_iob.txt'
    # train_IOBES_output_path = config.new_data_process_quarter_final
    train_IOBES_output_path = 'train.txt'
    handler = TransferData(train_IOB_output_path, train_IOBES_output_path)
    handler.update_tag_scheme(train_IOB_output_path, train_IOBES_output_path)

    # dev_IOB_output_path = config.new_data_process_quarter_final + 'dev_iob.txt'
    dev_IOB_output_path = 'dev_iob.txt'
    # dev_IOBES_output_path = config.new_data_process_quarter_final
    dev_IOBES_output_path = 'dev.txt'
    handler = TransferData(dev_IOB_output_path, dev_IOBES_output_path)
    handler.update_tag_scheme(dev_IOB_output_path, dev_IOBES_output_path)

    print("数据转换成功")
