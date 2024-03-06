import ast
import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict


def hdfs_sampling(log_file, window='session', window_size=0):
    """
        对HDFS日志文件进行会话窗口采样
        把解析后的日志文件转换成序列数据，(BlockId, LogKeyList, ParamVecList) 三列的形式。

        e.g.:
        blk_-8775602795571523802, "['E7', 'E7']", "[['81110103321', 'blk_-8775602795571523802', 'mnt/hadoop/dfs/data/xxx'], ['81110103403', 'blk_-8775602795571523802', 'mnt/hadoop/dfs/xxx']]"

        参数:
            log_file (str): 日志文件的路径
            window (str): 窗口类型，默认为 'session'
            window_size (int): 窗口大小，默认为0

        返回:
            None
        """
    assert window == 'session', 'Only window=session is supported for HDFS_2k dataset. HDFS数据集仅适用于会话窗口采样。'
    print("Loading", log_file)
    struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    data_dict = OrderedDict()
    for idx, row in struct_log.iterrows():
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
    data_df.to_csv('HDFS/HDFS_100k_sequence.csv', index=None)


# hdfs_sampling('../sampling_example/HDFS/HDFS_2k.log_structured.csv')
hdfs_sampling('hdfs/HDFS_100k.log_structured.csv')
