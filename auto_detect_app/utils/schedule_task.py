import json
from nn_models.lstm import DeepLog
from tools.predict import Predictor
from apscheduler.schedulers.background import BackgroundScheduler
from elasticsearch import Elasticsearch
from auto_detect_app.services.elasticsearch_service import ElasticSearchService

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


def detect_job():
    es = ElasticSearchService()
    # 定时检测过去一段时间内的日志数据
    # 1. 从ES中获取过去一段时间内的日志数据，并且要往前多取10条数据，因为窗口大小为10
    index = 'hdfs_sample_logs_*'
    body = {
        "query": {
            "term": {
                "Pid": "2665"
            }
        },
        "size": 10
    }
    # 使用实例调用search方法
    results = es.search(index, body)
    # 以json格式打印结果，要进行换行缩进
    print(json.dumps(results, indent=4))

    # 2. 对日志数据的content字段进行解析，将该批次日志转换为模型的输入格式，即日志键序列
    # 3. 使用模型进行检测
    # 4. 将预测结果存入ES中
    print("Job executed!")


def schedule_jobs():
    scheduler = BackgroundScheduler()
    scheduler.add_job(detect_job, 'interval', seconds=10)
    scheduler.start()
