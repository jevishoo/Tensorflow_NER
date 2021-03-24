import pandas as pd
import json
import sys

sys.path.append("/home/hezoujie/Competition/CCKS_Military_NER")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块

from config import Config

config = Config()
data_dir = config.new_data_process_quarter_final
ensemble_result_file = config.ensemble_result_file  # 单模结果路径

test_result_pd = pd.read_csv(ensemble_result_file + 'test_result.csv', encoding='utf-8')

y_pred_label_list = test_result_pd['y_pred_label_list'].tolist()
pred_start_pos_list = test_result_pd['pred_start_pos_list'].tolist()
pred_end_pos_list = test_result_pd['pred_end_pos_list'].tolist()
test_cut_text_list = test_result_pd['ldct_list_text'].tolist()
y_pred_entity_list = test_result_pd['y_pred_entity_list'].tolist()


def set_operation(row):
    content_list = row.split('|')
    content_list_after_set = list(set(content_list))
    return ";".join(content_list_after_set)


def entity_operation(row):
    """
    删除nan和长度小于2的标签
    """
    entity_list = row.split('|')
    result_entity = []
    for entity in entity_list:
        # if entity != 'nan' and len(entity) >= 2:
        if entity != 'nan':
            result_entity.append(entity)

    return ";".join(result_entity)


if __name__ == '__main__':
    # 对被切分的测试集进行拼接
    pre_index = 0
    repair_text_list = []
    repair_entity_list = []
    with open(data_dir + 'cut_index_list.json', 'r') as f:
        load_dict = json.load(f)
        cut_index_list = load_dict['cut_index_list']

    y_pred_entity_list = [str(item) for item in y_pred_entity_list]
    for i, seg_num in enumerate(cut_index_list):
        if i == 0:
            text = "".join(test_cut_text_list[: seg_num])
            entity = ";".join(y_pred_entity_list[: seg_num])
            repair_text_list.append(text)
            repair_entity_list.append(entity)

        else:
            text = "".join(test_cut_text_list[pre_index: pre_index + seg_num])
            entity = ";".join([str(entity) for entity in y_pred_entity_list[pre_index: pre_index + seg_num]])
            repair_text_list.append(text)
            repair_entity_list.append(entity)
        pre_index += seg_num

    dict_data = {
        'label_type': y_pred_label_list,
        'start_pos': pred_start_pos_list,
        'end_pos': pred_end_pos_list,
        'entities': repair_entity_list,
        'originalText': repair_text_list
    }

    final_result = pd.DataFrame(dict_data)

    final_result.to_csv(ensemble_result_file + 'final_result_with_text.csv', index=False, encoding='utf-8')
    final_result[['entities']].to_csv(ensemble_result_file + 'final_result.csv', index=False, encoding='utf-8')

    print('final_result_with_text  Done...')
