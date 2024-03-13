from flask import Flask
from views.lad import first_test, detect_task
from utils.schedule_task import schedule_jobs

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 注册蓝图
app.register_blueprint(first_test, url_prefix='/test')
app.register_blueprint(detect_task, url_prefix='/task')

if __name__ == '__main__':
    schedule_jobs()  # 启动定时任务
    app.run()
