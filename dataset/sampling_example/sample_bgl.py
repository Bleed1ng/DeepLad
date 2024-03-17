import os
import pandas as pd
import numpy as np

para = {
    "window_size": 0.5,  # 每个滑动窗口的持续时间，单位是小时。0.5意味着每个窗口的持续时间是半小时，也就是30分钟。
    "step_size": 0.2,  # 滑动窗口移动的步长，单位是小时。0.2意味着每次窗口向前移动0.2小时，也就是12分钟。
    "structured_file": "bgl/BGL_100k_structured.csv",
    "BGL_sequence": 'bgl/BGL_sequence.csv'
}


def load_BGL():
    """
    --------------BGL_structured.csv--------------
     label｜           time            ｜ event_id
       -  ｜2005-06-03-15.42.50.363779 ｜ E189
       -  ｜2005-06-03-15.42.50.527847 ｜ E189
    ----------------------------------------------
    根据time计算seconds_since
    :return: label、seconds_since、event_id
    """
    structured_file = para["structured_file"]
    bgl_structured = pd.read_csv(structured_file)
    bgl_structured["time"] = pd.to_datetime(bgl_structured["time"], format="%Y-%m-%d-%H.%M.%S.%f")
    # 计算每条日志的时间与第一条日志的时间差
    bgl_structured["seconds_since"] = (
            bgl_structured['time'] - bgl_structured['time'][0]
    ).dt.total_seconds().astype(int)
    # 将label列的数据转换为二进制标签，其中"-"表示正常（转换为0），非"-"表示异常（转换为1）。
    bgl_structured['label'] = (bgl_structured['label'] != '-').astype(int)
    return bgl_structured


def bgl_sampling(bgl_structured):
    """
    BGL数据集的滑动窗口采样
    :param bgl_structured:
    :return:
    """
    label_data, time_data, event_mapping_data = (bgl_structured['label'].values,
                                                 bgl_structured['seconds_since'].values,
                                                 bgl_structured['event_id'].values)
    log_size = len(label_data)
    start_time = time_data[0]
    start_index = 0
    end_index = 0
    start_end_index_list = []
    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < start_time + para["window_size"] * 3600:
            end_index += 1
            end_time = cur_time
        else:
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
            break
    while end_index < log_size:
        start_time = start_time + para["step_size"] * 3600
        end_time = end_time + para["step_size"] * 3600
        for i in range(start_index, end_index):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j
        start_end_pair = tuple((start_index, end_index))
        start_end_index_list.append(start_end_pair)
    # start_end_index_list is the window divided by window_size and step_size,
    # the front is the sequence number of the beginning of the window, 
    # and the end is the sequence number of the end of the window
    inst_number = len(start_end_index_list)
    print('there are %d instances (sliding windows) in this dataset' % inst_number)

    # 获取每个时间窗口中的所有日志索引，范围从start_index到end_index
    expanded_indexes_list = [[] for i in range(inst_number)]
    expanded_event_list = [[] for i in range(inst_number)]

    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)
            expanded_event_list[i].append(event_mapping_data[l])
    # =============get labels and event count of each sliding window =========#

    labels = []

    for j in range(inst_number):
        label = 0  # 0表示成功，1表示失败
        for k in expanded_indexes_list[j]:
            # 如果其中一个序列是异常的（1），则将该序列标记为异常
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies" % sum(labels))

    BGL_sequence = pd.DataFrame(columns=['sequence', 'label'])
    BGL_sequence['sequence'] = expanded_event_list
    BGL_sequence['label'] = labels
    BGL_sequence.to_csv(para["BGL_sequence"], index=None)


if __name__ == "__main__":
    bgl_structured = load_BGL()
    bgl_sampling(bgl_structured)
