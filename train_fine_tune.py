import os
import time
import json
import tqdm
from config import Config
from model import Model
from utils import DataIterator
import numpy as np
from bert import tokenization
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.contrib.crf import viterbi_decode
import tensorflow as tf

"""1: roberta_wwm_large, 256-idcnn 8折数据"""  # 有问题
"""2: roberta_wwm_large, 256-bilstm 8折数据"""
"""3: roberta_wwm_large, 动态融合, 512-bilstm 8折数据"""
"""4: albert, 256-bilstm"""

"""5: roberta_zh, 256-bilstm, 无中文括号5折数据"""
"""6: roberta_wwm_large, 256-bilstm, 无中文括号5折数据"""
"""7: roberta_wwm_large 动态融合, 128-bilstm (轻量级), 无中文括号5折数据"""

gpu_id = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
# 混合精度
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = '1'


print('GPU ID: ', gpu_id)
print('Model Type: ', Config().model_type)
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('bilstm embedding ', Config().lstm_dim)
print('use original bert ', Config().use_origin_bert)


def train(train_iter, test_iter, config):
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=session_conf)

    with session.as_default():
        print("**** Build Model ****")
        model = Model(config)  # 读取模型结构图

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(config.model_dir, "runs_" + gpu_id, timestamp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_dir + '/' + 'config.json', 'w', encoding='utf-8') as file:
            json.dump(config.__dict__, file)
        print("Writing to {}\n".format(out_dir))

        if config.continue_training:
            print('recover from: {}'.format(config.checkpoint_path))
            model.saver.restore(session, config.checkpoint_path)
        else:
            session.run(tf.global_variables_initializer())

        """
        在config.py设置了10个epoch
        这么设置的目的是多保存几个模型，再通过check_F1.py来查看每次训练得到的最高F1模型，取最优模型进行预测。
        """
        cum_step = 0
        loss_list = [2000]
        print("**** Start Training ****")
        for i in range(config.train_epoch):  # 训练
            for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm.tqdm(
                    train_iter):

                feed_dict = {
                    model.input_x_word: np.asarray(input_ids_list),
                    model.input_mask: np.asarray(input_mask_list),
                    model.input_relation: np.asarray(label_ids_list),
                    model.input_x_len: seq_length,
                    model.keep_prob: config.keep_prob,
                    model.is_training: True
                }

                _, step, loss, embed_lr, lr = session.run(
                    fetches=[model.train_op,
                             model.global_step,
                             model.loss,
                             model.embed_learning_rate,
                             model.learning_rate
                             ],
                    feed_dict=feed_dict)

                loss_list.append(loss)

                if cum_step % 10 == 0:
                    format_str = 'step {}, loss {:.2f} embed_lr {:.6f} lr {:.6f}'
                    print(
                        format_str.format(
                            step, loss, embed_lr, lr)
                    )
                    data = pd.DataFrame(data=loss_list, columns=['loss'], index=None)
                    data.to_csv(out_dir + '/loss.csv')
                    plot_loss(loss_list[1:], out_dir)

                cum_step += 1

            P, R, F, dev_result = set_test(model, test_iter, session)
            print('Dev set : Step_{}, Precision_{}, Recall_{},F1_{}'.format(cum_step, P, R, F))
            if F > 0.72:  # 保存F1大于0的模型
                model.saver.save(session, os.path.join(out_dir, 'model_{:.2f}_{:.2f}_{:.4f}'.format(P, R, F)),
                                 global_step=step)
                dev_result.to_csv(out_dir + '/model_{:.2f}_{:.2f}_{:.4f}'.format(P, R, F) + '_dev_result.csv')


def plot_loss(loss, out_path):
    step = [i + 1 for i in range(len(loss))]

    # 生成图表
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500  # 分辨率
    plt.plot(step, loss, color='r')
    # 设置横坐标为year，纵坐标为population，标题为Population year correspondence
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Loss Iterator')
    # 保存图表
    plt.savefig(out_path + '/loss.svg', dpi=500, bbox_inches='tight')
    plt.savefig(out_path + '/loss.eps', dpi=500, bbox_inches='tight')


def get_text_and_entity(input_tokens_list, y_label_list):
    temp = []
    for batch_y_list in y_label_list:
        temp += batch_y_list
    y_label_list = temp

    y_entity_list = []  # 标签
    start_pos_list = []  # 开始位置索引
    end_pos_list = []  # 结束位置索引
    entity_type_list = []

    for i, input_tokens in enumerate(input_tokens_list):
        ys = y_label_list[i]  # 每条数据对应的数字标签列表
        temp = []
        label_list = []

        s_list = []
        e_list = []
        e_type = []
        is_start = False

        for index, num in enumerate(ys):
            # if (num in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]) \
            if (num in [2, 4, 6, 8, 10, 12, 14, 16]) \
                    and len(temp) == 0:  # B S (标签开头及单独)
                s_list.append(index)
                type_num = int((num - 1) / 4)
                if type_num == 0:
                    e_type.append("ELE")
                    # e_type.append("CONT")
                elif type_num == 1:
                    e_type.append("IND")
                    # e_type.append("EDU")
                elif type_num == 2:
                    e_type.append("CON")
                    # e_type.append("LOC")
                elif type_num == 3:
                    e_type.append("SIT")
                    # e_type.append("NAME")
                elif type_num == 4:
                    e_type.append("ORG")
                elif type_num == 5:
                    e_type.append("PRO")
                elif type_num == 6:
                    e_type.append("RACE")
                else:
                    e_type.append("TITLE")

                is_start = True
                temp.append(input_tokens[index])
            # elif (num in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]) \
            elif (num in [1, 3, 5, 7, 9, 11, 13, 15]) \
                    and len(temp) > 0:  # I/M E (标签中间及结尾)
                temp.append(input_tokens[index])
            elif len(temp) > 0:
                if is_start:
                    e_list.append(index - 1)
                is_start = False

                label_list.append("".join(temp))
                temp = []

        if len(s_list) != len(e_list):
            print(ys)
            # print(e_type)
            # print(s_list)
            # print(e_list)
            # print(label_list)

        y_entity_list.append("|".join(label_list))
        start_pos_list.append(s_list)
        end_pos_list.append(e_list)
        entity_type_list.append(e_type)

    return y_label_list, y_entity_list, entity_type_list, start_pos_list, end_pos_list


def decode(logits, lengths, matrix):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * Config().relation_num + [0]])
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)

        paths.append(path[1:])

    return paths


def set_operation(row):
    content_list = row.split(';')
    content_list_after_set = list(set(content_list))
    return ";".join(content_list_after_set)


def flatten(input_list):
    output_list = []
    while True:
        if input_list == []:
            break
        for index, value in enumerate(input_list):
            # index :索引序列  value:索引序列对应的值
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
            # 同时列出数据和数据下标，一般用在 for 循环当中。
            if type(value) == list:
                input_list = value + input_list[index + 1:]
                break  # 这里跳出for循环后，从While循环进入的时候index是更新后的input_list新开始算的。
            else:
                output_list.append(value)
                input_list.pop(index)
                break
    return output_list


def get_P_R_F(dev_pd):
    dev_pd = dev_pd.fillna("0")
    # dev_pd['y_pred_entity'] = dev_pd['y_pred_entity'].apply(set_operation)
    # dev_pd['y_true_entity'] = dev_pd['y_true_entity'].apply(set_operation)
    y_true_entity_list = list(dev_pd['y_true_entity'])
    y_pred_entity_list = list(dev_pd['y_pred_entity'])
    y_true_entity_type_list = list(dev_pd['y_true_entity_type'])
    y_pred_entity_type_list = list(dev_pd['y_pred_entity_type'])

    TP = 0
    FP = 0
    FN = 0

    # class_dict_data = {
    #     "CONT": [0, 0, 0],
    #     "EDU": [0, 0, 0],
    #     "LOC": [0, 0, 0],
    #     "NAME": [0, 0, 0],
    #     "ORG": [0, 0, 0],
    #     "PRO": [0, 0, 0],
    #     "RACE": [0, 0, 0],
    #     "TITLE": [0, 0, 0]
    # }
    class_dict_data = {
        "ELE": [0, 0, 0],
        "SIT": [0, 0, 0],
        "CON": [0, 0, 0],
        "IND": [0, 0, 0],
    }

    y_not_pred = []
    y_pred_true = []
    y_pred_false = []
    for i, y_true_entity in enumerate(y_true_entity_list):
        y_pred_entity = y_pred_entity_list[i].split('|')
        y_true_entity = y_true_entity.split('|')

        if y_pred_entity == ['']:
            continue

        # print(len(y_pred_entity_type_list[i]))
        # print(len(y_pred_entity))
        # assert len(y_pred_entity) == len(y_pred_entity_type_list[i])

        current_TP = 0
        current_class_dict_data = {
            "ELE": 0,
            "SIT": 0,
            "CON": 0,
            "IND": 0,
        }
        # current_class_dict_data = {
        #     "CONT": 0,
        #     "EDU": 0,
        #     "LOC": 0,
        #     "NAME": 0,
        #     "ORG": 0,
        #     "PRO": 0,
        #     "RACE": 0,
        #     "TITLE": 0
        # }
        y_pred_true_list = []
        temp_true, temp_false, temp_not = [], [], []

        for j, y_pred in enumerate(y_pred_entity):
            if y_pred in y_true_entity:
                current_TP += 1  # 粗

                if y_pred_entity_type_list[i][j] == y_true_entity_type_list[i][y_true_entity.index(y_pred)]:
                    current_class_dict_data[y_pred_entity_type_list[i][j]] += 1
                    # current_TP += 1  # 细

                    class_dict_data[y_pred_entity_type_list[i][j]][0] += 1  # class TP
                    y_pred_true_list.append(y_pred)
                    temp_true.append("".join(y_pred))
                else:
                    temp_false.append("".join(y_pred))
                    class_dict_data[y_true_entity_type_list[i][y_true_entity.index(y_pred)]][1] += 1  # class FP
                    # FP += 1  # 细
            else:
                temp_false.append("".join(y_pred))
                class_dict_data[y_pred_entity_type_list[i][j]][1] += 1  # class FP
                FP += 1

        TP += current_TP
        FN += (len(y_true_entity) - current_TP)

        from collections import Counter
        # print(current_class_dict_data)
        # print(Counter(y_true_entity_type_list[i]))

        for pred_type in set(y_pred_entity_type_list[i]):
            class_dict_data[pred_type][2] += Counter(y_true_entity_type_list[i])[pred_type] - current_class_dict_data[
                pred_type]  # class FN

        for y_true in y_true_entity:
            if y_true not in y_pred_true_list:
                temp_not.append("".join(y_true))

        y_pred_true.append(";".join(temp_true))
        y_pred_false.append(";".join(temp_false))
        y_not_pred.append(";".join(temp_not))

    dict_data = {
        "y_not_pred": y_not_pred,
        "y_pred_true": y_pred_true,
        "y_pred_false": y_pred_false
    }

    # print(y_not_pred)
    # print(y_pred_true)
    # print(y_pred_false)

    dev_data = pd.DataFrame.from_dict(dict_data, orient='index')
    class_data = pd.DataFrame.from_dict(class_dict_data, orient='index', columns=['TP', 'FP', 'FN'])
    print(class_data)

    class_dict_prf_data = class_dict_data.copy()
    for key, values in class_dict_prf_data.items():
        try:
            p = class_dict_data[key][0] / (class_dict_data[key][0] + class_dict_data[key][1])
        except:
            p = 0

        try:
            r = class_dict_data[key][0] / (class_dict_data[key][0] + class_dict_data[key][2])
        except:
            r = 0

        try:
            f = 2 * p * r / (p + r)
        except:
            f = 0

        values[0] = p
        values[1] = r
        values[2] = f

    class_prf_data = pd.DataFrame.from_dict(class_dict_prf_data, orient='index', columns=['P', 'R', 'F1'])
    print(class_prf_data)

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    try:
        F = 2 * P * R / (P + R)
    except:
        F = 0

    return P, R, F, dev_data


def set_test(model, test_iter, session):
    if not test_iter.is_test:
        test_iter.is_test = True

    y_pred_list = []
    y_true_list = []
    ldct_list_tokens = []
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length, tokens_list in tqdm.tqdm(
            test_iter):
        feed_dict = {
            model.input_x_word: input_ids_list,
            model.input_x_len: seq_length,
            model.input_relation: label_ids_list,
            model.input_mask: input_mask_list,
            model.keep_prob: 1.0,
            model.is_training: False,
        }

        lengths, logits, trans = session.run(
            fetches=[model.lengths, model.logits, model.trans],
            feed_dict=feed_dict
        )

        predict = decode(logits, lengths, trans)
        y_pred_list.append(predict)
        y_true_list.append(label_ids_list)
        ldct_list_tokens.append(tokens_list)

    ldct_list_tokens = np.concatenate(ldct_list_tokens)
    ldct_list_text = []
    for tokens in ldct_list_tokens:
        text = "".join(tokens)
        ldct_list_text.append(text)

    # 获取验证集文本及其标签
    y_pred_list, y_pred_entity_list, y_pred_entity_type_list, _, _ = get_text_and_entity(ldct_list_tokens, y_pred_list)
    y_true_list, y_true_entity_list, y_true_entity_type_list, _, _ = get_text_and_entity(ldct_list_tokens, y_true_list)

    dict_data = {
        'y_true_entity': y_true_entity_list,
        'y_pred_entity': y_pred_entity_list,
        'y_pred_entity_type': y_pred_entity_type_list,
        'y_true_entity_type': y_true_entity_type_list,
        'y_pred_text': ldct_list_text
    }
    df = pd.DataFrame(dict_data)

    precision, recall, f1, dev_result = get_P_R_F(df)

    return precision, recall, f1, dev_result


if __name__ == '__main__':
    config = Config()
    result_data_dir = config.new_data_process_quarter_final
    print('Data dir: ', result_data_dir)

    vocab_file = config.vocab_file  # 通用词典

    do_lower_case = False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_iter = DataIterator(config.train_batch_size, data_file=result_data_dir + 'train.txt',
                              use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    print('GET!!')
    dev_iter = DataIterator(config.train_batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert,
                            tokenizer=tokenizer,
                            seq_length=config.sequence_length, is_test=True)

    config.num_records = train_iter.num_records
    train(train_iter, dev_iter, config)
