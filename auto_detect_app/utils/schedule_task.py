import json
from nn_models.lstm import DeepLog
from tools.predict import Predictor
from apscheduler.schedulers.background import BackgroundScheduler
from auto_detect_app.services.elasticsearch_service import ElasticSearchService
from auto_detect_app.config import configure_logger

logger = configure_logger()

options = dict()
options['data_dir'] = '../data/'
options['model_path'] = "../result/deeplog/deeplog_last.pth"
options['num_candidates'] = 9
options['device'] = "cpu"

# Sample
options['sample'] = "sliding_window"
options['window_size'] = 10

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


def schedule_jobs():
    scheduler = BackgroundScheduler()
    scheduler.add_job(detect_job, 'interval', seconds=5)
    scheduler.start()


def predict(sequence_list):
    model = DeepLog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predictor = Predictor(model, options)
    predictor.detect(sequence_list)


# 定时检测过去一段时间内的日志数据
def detect_job():
    es = ElasticSearchService()
    """
    1. 从ES中获取过去一段时间内的日志数据，并且要往前多取10条数据，因为窗口大小为10
    in:
    out: content_list
    """
    index = 'hdfs_sample_logs_*'
    body = {
        "query": {
            "term": {
                "Pid": "2665"
            }
        },
        "size": 10
    }
    results = es.search(index, body)
    # print(json.dumps(results, indent=2))
    if results['hits']['total']['value'] == 0:
        logger.info("该批次待检测日志查询为空")
        return

    batch_log_list = []
    for hit in results['hits']['hits']:
        log_dict = {
            'id': hit['_id'],
            '@timestamp': hit['_source']['@timestamp'],
            'content': hit['_source']['Content']
        }
        batch_log_list.append(log_dict)

    """
    2. 对日志数据的content字段进行解析，将该批次日志转换为模型的输入格式，即日志键序列
        (1) 用spell解析content，得到对应的日志键列表（还可以额外取到参数值向量列表）
        (2) 进行窗口采样，得到日志键序列
    in: content_list
    out: log_key_sequence
    """
    sequence_list = []

    """
    3. 使用模型进行检测
    """
    # predict(sequence_list)

    """
    4. 将预测结果存入ES中
    """
    logger.info("检测完成")
