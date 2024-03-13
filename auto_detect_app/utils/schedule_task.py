from nn_models.lstm import DeepLog
from tools.predict import Predictor
from apscheduler.schedulers.background import BackgroundScheduler

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

options['batch_size'] = 2048
# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 28

# Predict
options['model_path'] = "../result/deeplog/deeplog_last.pth"
options['num_candidates'] = 9


def predict():
    model = DeepLog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predictor = Predictor(model, options)
    predictor.detect()


def my_job():
    print("Job executed!")


def schedule_jobs():
    scheduler = BackgroundScheduler()
    scheduler.add_job(my_job, 'interval', seconds=10)
    scheduler.start()
