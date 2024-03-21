# 根据会话标识符对HDFS日志进行采样，将日志转换为序列数据

import ast
import os
import re
from collections import OrderedDict
from itertools import repeat
from tqdm import tqdm

import pandas as pd


def hdfs_sampling(log_file, window='session', window_size=10):
    """
    对HDFS日志文件进行会话窗口采样
    把解析后的日志文件转换成序列数据，(BlockId, LogKeyList, ParamVecList) 三列的形式。

    e.g.:
    blk_-8775602795571523802, "['E7', 'E7']", "[['81110103321', 'blk_-8775602795571523802', 'mnt/hadoop/dfs/data/xxx'],
                                                ['81110103403', 'blk_-8775602795571523802', 'mnt/hadoop/dfs/xxx']]"

    参数:
        log_file (str): 日志文件的路径
        window (str): 窗口类型，默认为 'session'
        window_size (int): 窗口大小，默认为0

    返回:
        None
    """
    assert window == 'session', 'Only window=session is supported for HDFS_2k dataset.'
    print("Loading", log_file)
    df_logs = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    data_dict = OrderedDict()
    for idx, row in tqdm(df_logs.iterrows(), desc='Sampling: ', total=df_logs.shape[0]):
        # 使用正则表达式找到每行中Content列中的blk_id，并将结果存储在blk_id_list中
        blk_id_list = re.findall(r'(blk_-?\d+)', row['Content'])
        # 将blk_id_list中的重复元素去掉
        blk_id_set = set(blk_id_list)
        for blk_id in blk_id_set:
            if blk_id not in data_dict:
                data_dict[blk_id] = {'EventIdList': [], 'ParameterList': []}
            data_dict[blk_id]['EventIdList'].append(row['EventId'])
            timestamp = str(row['Date']) + str(row['Time'])
            param_list = ast.literal_eval(row['ParameterList'])
            param_list = [timestamp] + param_list
            data_dict[blk_id]['ParameterList'].append(param_list)

    data_df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
    data_df.columns = ['BlockId', 'EventIdList', 'ParameterList']
    data_df.to_csv('hdfs/HDFS.log_param_sequence.csv', index=False)


def session_sampling(log_list, window='session', window_size=10):
    """
    对HDFS日志列表进行会话窗口采样,把解析出日志键的日志列表转换成日志键序列

    参数:
        log_key_list: [log_id, content, log_key]
    返回:
        [{blk_id: 'blk_1', seq: [1, 2, 3, 3, 5]},
         {blk_id: 'blk_2', seq: [1, 2, 3, 4, 0]},
         {blk_id: 'blk_3', seq: [1, 2, 3, 0, 0]}]
    """
    assert window == 'session', '特定数据集仅适用于会话窗口采样。'
    data_dict = OrderedDict()
    for line in log_list:
        # 使用正则表达式找到每行中content列中的blk_id，并将结果存储在blk_id_list中，用set做一下去重
        blk_id_set = set(re.findall(r'(blk_-?\d+)', line['content']))
        for blk_id in blk_id_set:
            if blk_id not in data_dict:
                data_dict[blk_id] = {'log_key_seq': []}
            data_dict[blk_id]['log_key_seq'].append(line['log_key'])

    sequence_list = [{'blk_id': blk_id, 'seq': data_dict[blk_id]['log_key_seq']} for blk_id in data_dict]

    # 如果日志键序列长度小于window_size + 1，则在末尾补0，直到其长度达到window_size + 1。
    blk_seq_list = []
    for blk_seq in sequence_list:
        blk_seq['seq'] = blk_seq['seq'] + list(repeat(0, window_size + 1 - len(blk_seq['seq'])))
        blk_seq_list.append(blk_seq)

    return blk_seq_list


def session_sampling_from_file(log_file, window_size=10):
    """
    对HDFS日志文件进行会话窗口采样,把解析后的日志文件转换成日志键序列，输出到csv文件中。
    参数:
        log_file (str): 日志文件的路径
        window_size (int): 窗口大小
    """
    print("Sampling: ", log_file)
    df_logs = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    data_dict = OrderedDict()
    for idx, row in tqdm(df_logs.iterrows(), desc='Sampling... ', total=df_logs.shape[0]):
        # 使用正则表达式找到每行中content列中的blk_id，并将结果存储在blk_id_list中，用set做一下去重
        blk_id_set = set(re.findall(r'(blk_-?\d+)', row['content']))
        for blk_id in blk_id_set:
            if blk_id not in data_dict:
                data_dict[blk_id] = {'log_key_seq': []}
            data_dict[blk_id]['log_key_seq'].append(row['log_key'])

    sequence_list = [{'blk_id': blk_id, 'seq': data_dict[blk_id]['log_key_seq']} for blk_id in data_dict]

    # 如果日志键序列长度小于window_size + 1，则在末尾补0，直到其长度达到window_size + 1。
    blk_seq_list = []
    for blk_seq in tqdm(sequence_list, desc='Padding...  ', total=len(sequence_list)):
        blk_seq['seq'] = blk_seq['seq'] + list(repeat(0, window_size + 1 - len(blk_seq['seq'])))
        blk_seq_list.append(blk_seq)

    # 保存采样结果到csv
    df_seq = pd.DataFrame(blk_seq_list)
    df_seq.columns = ['blk_id', 'log_key_seq']
    df_seq.to_csv('hdfs/HDFS.log_sequence.csv', index=False)


# session_sampling_from_file('../data/spell_result/HDFS.log_structured.csv')
