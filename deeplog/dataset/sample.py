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


# 上采样
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
        在提供的日志上执行滑动窗口采样。

        参数:
        data_dir (str): 数据文件所在的目录。
        datatype (str): 要处理的数据类型。可以是 'train' 或 'val'。
        window_size (int): 滑动窗口的大小。
        sample_ratio (float, 可选): 应采样的总日志数的比例。默认为1。

        返回:
        dict: 包含采样日志的字典。
        list: 包含对应于采样日志的标签的列表。

        该函数首先读取一个包含事件到其语义向量的映射的JSON文件。
        然后初始化会话数、结果日志和标签。
        根据数据类型，它将适当的文件路径附加到数据目录。
        然后打开文件并读取每一行。
        对于每一行，它增加会话数并将行转换为整数元组。
        然后在行上执行滑动窗口采样。
        对于每个窗口，它创建一个顺序模式、一个定量模式和一个语义模式。
        顺序模式是窗口中的事件列表。
        定量模式是窗口中每个事件的计数列表。
        语义模式是每个事件对应的语义向量的列表。
        然后将模式附加到结果日志，将窗口后的下一个事件的标签附加到标签。
        如果采样比率不是1，那么它在结果日志和标签上执行下采样。
        最后，它打印会话数和序列数，并返回结果日志和标签。
        """
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {'Sequentials': [], 'Quantitatives': [], 'Semantics': []}
    labels = []
    if datatype == 'train':
        data_dir += 'hdfs/hdfs_train'
    if datatype == 'val':
        data_dir += 'hdfs/hdfs_test_normal'

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
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    result_logs = {'Sequentials': [], 'Quantitatives': [], 'Semantics': []}
    labels = []

    if datatype == 'train':
        data_dir += 'hdfs/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += 'hdfs/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += 'hdfs/robust_log_test.csv'

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
