import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict


def hdfs_sampling(log_file, window='session', window_size=0):
    """
        对HDFS日志文件进行采样
        把解析后的日志文件转换成序列数据

        参数:
            log_file (str): 日志文件的路径
            window (str): 窗口类型，默认为 'session'
            window_size (int): 窗口大小，默认为0

        返回:
            None
        """
    assert window == 'session', 'Only window=session is supported for HDFS dataset.'
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
                data_dict[blk_id] = []
            data_dict[blk_id].append(row['EventId'])
    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    data_df.to_csv('hdfs/HDFS_sequence.csv', index=None)


# hdfs_sampling('../logs_dataset/HDFS/HDFS_2k.log_structured.csv')
hdfs_sampling('hdfs/HDFS_100k.log_structured.csv')
