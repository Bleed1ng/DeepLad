import argparse
import sys

sys.path.append('../')

from deeplog.models.lstm import DeepLog, LogAnomaly, RobustLog
from deeplog.tools.predict import Predictor
from deeplog.tools.train import Trainer
from deeplog.tools.utils import *

# Deeplog- Log Key Anomaly Detection model 日志键序列异常检测模型

# Config Parameters
options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cpu"

# Sample
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# 是否要使用的Features: 序列特征、数量特征、语义特征
options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False
options['feature_num'] = sum([options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 28

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1

# 优化器
options['optimizer'] = 'adam'
options['lr'] = 0.001  # 学习率
options['max_epoch'] = 370  # 训练的epoch数量
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = "../result/deeplog/"

# Predict
options['model_path'] = "../result/deeplog/deeplog_last.pth"
options['num_candidates'] = 9

seed_everything(seed=1234)


def train():
    model = DeepLog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(model, options)
    trainer.start_train()


def predict():
    model = DeepLog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predictor = Predictor(model, options)
    predictor.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
