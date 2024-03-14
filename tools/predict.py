import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.log import log_dataset
from dataset.sample import session_window


# 去重，统计每种序列出现的次数，并且用-1填充长度不够的序列
def generate(name):
    window_size = 10
    hdfs = {}
    length = 0
    with open('../data/HDFS/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            # 如果列表的长度小于window_size + 1，则在列表的末尾添加 - 1，直到其长度达到window_size + 1。
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


def generate_sequence(sequence_list):
    window_size = 10
    hdfs = {}
    length = 0
    for ln in sequence_list:
        ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
        # 如果列表的长度小于window_size + 1，则在列表的末尾添加 - 1，直到其长度达到window_size + 1。
        ln = ln + [-1] * (window_size + 1 - len(ln))
        hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
        length += 1
    print('Number of sessions({}): {}'.format('log_key_list', len(hdfs)))
    return hdfs, length


class Predictor:
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])  # 加载模型参数
        model.eval()  # 设置模型为评估模式
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate('hdfs_test_normal')
        test_abnormal_loader, test_abnormal_length = generate('hdfs_test_abnormal')
        TP = 0  # true positive
        FP = 0  # false positive
        # Test the model
        start_time = time.time()
        with torch.no_grad():
            # 去重后遍历每种序列，避免对相同的序列重复预测
            for line in tqdm(test_normal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]  # 取出窗口大小的序列
                    label = line[i + self.window_size]  # 取出窗口后的标签
                    seq1 = [0] * 28
                    log_counter = Counter(seq0)
                    for key in log_counter:
                        seq1[key] = log_counter[key]

                    seq0 = (torch
                            .tensor(seq0, dtype=torch.float)
                            .view(-1, self.window_size, self.input_size)
                            .to(self.device))
                    seq1 = (torch
                            .tensor(seq1, dtype=torch.float)
                            .view(-1, self.num_classes, self.input_size)
                            .to(self.device))
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        # 预测为异常日志（Positive）,实际为正常日志（N），预测结果错误（F），为假阳性（FP）
                        FP += test_normal_loader[line]
                        break

        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_counter = Counter(seq0)
                    for key in log_counter:
                        seq1[key] = log_counter[key]

                    seq0 = (
                        torch.tensor(seq0, dtype=torch.float)
                        .view(-1, self.window_size, self.input_size)
                        .to(self.device)
                    )
                    seq1 = (
                        torch.tensor(seq1, dtype=torch.float)
                        .view(-1, self.num_classes, self.input_size)
                        .to(self.device)
                    )
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        # 预测为异常日志（Positive）,实际为异常日志（P），预测结果正确（T），为真阳性（TP）
                        TP += test_abnormal_loader[line]
                        break

        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, '
            'Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(TP, FP, FN, P, R, F1)
        )
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.data_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'true positive (TP): {}, true negative (TN): {}, false positive (FP): {}, false negative (FN): {}, '
            'Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(TP, TN, FP, FN, P, R, F1)
        )

    # 检测
    def detect(self, sequence_list):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])  # 加载模型参数
        model.eval()  # 设置模型为评估模式
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate_sequence(sequence_list=sequence_list)
        with torch.no_grad():
            # 去重后遍历每种序列，避免对相同的序列重复预测
            for line in test_normal_loader.keys():
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]  # 取出窗口大小的序列
                    label = line[i + self.window_size]  # 取出窗口后的标签
                    seq1 = [0] * 28
                    log_counter = Counter(seq0)
                    for key in log_counter:
                        seq1[key] = log_counter[key]
                    seq0 = (torch
                            .tensor(seq0, dtype=torch.float)
                            .view(-1, self.window_size, self.input_size)
                            .to(self.device))
                    seq1 = (torch
                            .tensor(seq1, dtype=torch.float)
                            .view(-1, self.num_classes, self.input_size)
                            .to(self.device))
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        print("=====检测到异常日志=====")
                        break
                    else:
                        print("正常")
                        break
