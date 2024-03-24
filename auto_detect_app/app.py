from flask import Flask
from flask_redis import FlaskRedis

from auto_detect_app.utils.redis_client import RedisClient
from views.lad import first_test, detect_task
from auto_detect_app.services.schedule_task import schedule_jobs
import config

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
app.config['REDIS_URL'] = config.RedisConfig.REDIS_URL
redis_client = FlaskRedis(app)

# 注册蓝图
app.register_blueprint(first_test, url_prefix='/test')
app.register_blueprint(detect_task, url_prefix='/task')

if __name__ == '__main__':
    redis_client = RedisClient()
    redis_client.load_parse_to_redis()  # 加载日志键和模版到Redis
    schedule_jobs()  # 启动定时任务
    app.run()
    # schedule_task.detect_job()
