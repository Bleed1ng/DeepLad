from flask import Flask

from views.lad import first_test, detect_task
from auto_detect_app.services.schedule_task import schedule_jobs
from utils.redis_utils import RedisUtils
import config

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['REDIS_URL'] = config.RedisConfig.REDIS_URL
redis_utils = RedisUtils(app)

# 注册蓝图
app.register_blueprint(first_test, url_prefix='/test')
app.register_blueprint(detect_task, url_prefix='/task')


@app.route('/set/<key>/<value>')
def set_key(key, value):
    # 调用RedisUtils的set方法设置键值对
    redis_utils.set(key, value)
    return f'成功设置{key}为{value}'


@app.route('/get/<key>')
def get_key(key):
    # 调用RedisUtils的get方法获取键对应的值
    value = redis_utils.get(key)
    if value:
        return f'{key}的值是{value}'
    else:
        return f'没有找到{key}'


@app.route('/delete/<key>')
def delete_key(key):
    # 调用RedisUtils的delete方法删除键
    result = redis_utils.delete(key)
    if result:
        return f'成功删除{key}'
    else:
        return f'没有找到{key}'


if __name__ == '__main__':
    schedule_jobs()  # 启动定时任务
    app.run()
