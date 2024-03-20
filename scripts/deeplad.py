import argparse

from nn_models.lstm import DeepLog
from tools.predict import Predictor
from tools.train import Trainer
from tools.utils import *

options = dict()
options['data_dir'] = '../sampling/hdfs/'
options['window_size'] = 10
options['device'] = "cpu"

# Sample
options['sample'] = "sliding_window"

options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False
options['feature_num'] = sum([options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 36

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1

# 优化器
options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 370
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplad"
options['save_dir'] = "../result/deeplad/"

# Predict
options['model_path'] = "../result/deeplog/deeplad_last.pth"
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
