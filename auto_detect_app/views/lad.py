from flask import Blueprint, jsonify, request
from nn_models.lstm import DeepLog
from tools.predict import Predictor

# 创建蓝图
first_test = Blueprint('first_test', __name__)
detect_task = Blueprint('detect_task', __name__)


@first_test.route('/hello', methods=['GET'])
def query_data():
    # 获取查询参数
    query_param = request.args.get('param', default=None, type=str)
    print("查询参数: ", query_param)
    output = "请求成功"
    print(output)
    return jsonify(message=output)


options = dict()
options['data_dir'] = '../data/'  # 数据目录
options['model_path'] = "../result/deeplog/deeplog_last.pth"  # 模型路径
options['window_size'] = 10
options['num_candidates'] = 9
options['device'] = "cpu"

# Sample
options['sample'] = "sliding_window"
options['window_size'] = 10

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


@detect_task.route('/detect', methods=['POST'])
def predict():
    model = DeepLog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predictor = Predictor(model, options)
    predictor.detect()
    return jsonify(message="检测完成")
