import ast
import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list
    将输入的列表l截断为长度为n，或在列表长度小于n的情况下，在列表末尾填充0，使其长度达到n
    """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


# 下采样
def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


# 滑动窗口采样
def sliding_window(data_dir, datatype, window_size, sample_ratio=1):
    """
        对日志键序列进行滑动窗口采样。
        result_logs: 5 5 5 22 11 9 11 9 11 9 ——> label: 26

        参数:
        data_dir (str): 数据文件所在的目录。
        datatype (str): 要处理的数据类型。可以是 'train' 或 'val'。
        window_size (int): 滑动窗口的大小。
        sample_ratio (float, 可选): 应采样的总日志数的比例。默认为1。

        返回:
        dict: 包含采样日志的字典。
        list: 包含对应于采样日志的标签的列表。

        """
    event2semantic_vec = read_json(data_dir + 'log_key_seq/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {'Sequentials': [], 'Quantitatives': [], 'Semantics': []}
    labels = []
    if datatype == 'train':
        data_dir += 'log_key_seq/hdfs_train'
    if datatype == 'val':
        data_dir += 'log_key_seq/hdfs_test_normal'

    with open(data_dir, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            for i in range(len(line) - window_size):
                sequential_pattern = list(line[i:i + window_size])
                quantitative_pattern = [0] * 28
                log_counter = Counter(sequential_pattern)

                for key in log_counter:
                    quantitative_pattern[key] = log_counter[key]
                semantic_pattern = []
                for event in sequential_pattern:
                    if event == 0:
                        semantic_pattern.append([-1] * 300)
                    else:
                        semantic_pattern.append(event2semantic_vec[str(event - 1)])
                sequential_pattern = np.array(sequential_pattern)[:, np.newaxis]
                quantitative_pattern = np.array(quantitative_pattern)[:, np.newaxis]
                result_logs['Sequentials'].append(sequential_pattern)
                result_logs['Quantitatives'].append(quantitative_pattern)
                result_logs['Semantics'].append(semantic_pattern)
                # 把窗口之后的下一个日志键作为标签
                labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir, len(result_logs['Sequentials'])))

    return result_logs, labels


# 会话窗口采样
def session_window(data_dir, datatype, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + 'log_key_seq/event2semantic_vec.json')
    result_logs = {'Sequentials': [], 'Quantitatives': [], 'Semantics': []}
    labels = []

    if datatype == 'train':
        data_dir += 'log_key_seq/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += 'log_key_seq/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += 'log_key_seq/robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        origin_seq = [
            int(event_id) for event_id in train_df["Sequence"][i].split(' ')
        ]
        sequential_pattern = trp(origin_seq, 50)
        semantic_pattern = []
        for event in sequential_pattern:
            if event == 0:
                semantic_pattern.append([-1] * 300)
            else:
                semantic_pattern.append(event2semantic_vec[str(event - 1)])
        quantitative_pattern = [0] * 29
        log_counter = Counter(sequential_pattern)

        for key in log_counter:
            quantitative_pattern[key] = log_counter[key]

        sequential_pattern = np.array(sequential_pattern)[:, np.newaxis]
        quantitative_pattern = np.array(quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(sequential_pattern)
        result_logs['Quantitatives'].append(quantitative_pattern)
        result_logs['Semantics'].append(semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print('Number of sessions({}): {}'.format(data_dir, len(result_logs['Semantics'])))
    return result_logs, labels


# 对参数值向量数据进行滑动窗口采样
def sliding_window_param(data_dir, datatype, window_size, sample_ratio=1):
    """
        对参数值向量数据进行滑动窗口采样。
        param_vec_list: [[123, fsd, fw], [124, ffd, gfd], [132, 1ds, fds3]] ——> label: [135, fdsa, gre]

        参数:
        data_dir (str): 数据文件所在的目录。
        datatype (str): 要处理的数据类型。可以是 'train' 或 'val'。
        window_size (int): 滑动窗口的大小。
        sample_ratio (float, 可选): 应采样的总日志数的比例。默认为1。

        返回:
        dict: 包含采样日志的字典。
        list: 包含对应于采样日志的标签的列表。

    """
    num_sessions = 0
    result_logs = {'ParamVecList': []}
    labels = []
    if datatype == 'train':
        data_dir += 'sampling_example/log_key_seq/HDFS_2k_sequence.csv'
    if datatype == 'val':
        data_dir += 'log_key_seq/hdfs_test_normal'

    # 从csv文件中读取数据，取第三列的参数值向量列表，并转化为实际的list类型
    df = pd.read_csv(data_dir)
    param_vec_list = df.iloc[:, 2].apply(ast.literal_eval)

    for line in tqdm(param_vec_list):
        num_sessions += 1
        for i in range(len(line) - window_size):
            current_param_vec = list(line[i:i + window_size])
            current_param_vec = np.array(current_param_vec)[:, np.newaxis]
            result_logs['ParamVecList'].append(current_param_vec)
            # 把窗口之后的下一个日志键作为标签
            labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir, len(result_logs['ParamVecList'])))

    return result_logs, labels

# a = sliding_window_param(data_dir='../../data/', datatype='train', window_size=10, sample_ratio=1)