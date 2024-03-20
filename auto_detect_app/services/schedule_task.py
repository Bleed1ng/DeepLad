import json

from apscheduler.schedulers.background import BackgroundScheduler

from auto_detect_app.config import configure_logger
from auto_detect_app.utils.elasticsearch_utils import ElasticSearchService
from sampling.sample_hdfs import session_sampling
from logparser.Spell import Spell
from nn_models.lstm import DeepLog
from tools.predict import Predictor

logger = configure_logger()

options = dict()
options['data_dir'] = '../data/'
options['model_path'] = "../result/deeplad/deeplad_last.pth"
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
options['num_classes'] = 36


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
    # 1. 从ES中获取过去一段时间内的日志数据，并且要往前多取10条数据，因为窗口大小为10
    es = ElasticSearchService()
    es_results = es.search_logs()
    if es_results['hits']['total']['value'] == 0:
        logger.info("该批次待检测日志查询为空")
        return

    batch_log_list = []
    for hit in es_results['hits']['hits']:
        log_dict = {
            'log_id': hit['_id'],
            'content': hit['_source']['Content']
        }
        batch_log_list.append(log_dict)

    # 2. 对日志数据的content字段进行解析，将该批次日志转换为模型的输入格式，即日志键序列
    #     (1) 用spell解析content，得到log_key_seq: [log_id, content, log_key]
    #     (2) 进行窗口采样，得到日志键序列
    result_dir = '/Users/Bleeding/Projects/BJTU/DeepLad/data/spell_result/'  # todo: 保存解析结果的目录，后续改为从Redis中获取
    parser = Spell.LogParser(outdir=result_dir)
    log_key_list = parser.parse_log_from_list(batch_log_list)  # 解析
    session_seq_list = session_sampling(log_key_list)  # 采样

    # 3. 使用模型进行检测
    predict(session_seq_list)

    """
    4. 将预测结果存入ES中
    """
    # logger.info("检测完成")
