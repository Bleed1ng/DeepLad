from flask import request, jsonify
from flask_redis import FlaskRedis


class RedisUtils:
    # 初始化方法，接收一个Flask应用对象作为参数
    def __init__(self, app):
        self.redis_client = FlaskRedis()  # 创建redis客户端
        self.redis_client.init_app(app)  # 初始化redis客户端

    # 封装set方法，接收键和值作为参数
    def set(self, key, value):
        self.redis_client.set(key, value)  # 使用redis客户端设置键值对

    # 封装get方法，接收键作为参数，返回值或者None
    def get(self, key):
        value = self.redis_client.get(key)  # 使用redis客户端获取键对应的值
        if value:
            return value.decode()  # 如果值存在，返回解码后的字符串
        else:
            return None  # 如果值不存在，返回None

    # 封装delete方法，接收键作为参数，返回删除结果
    def delete(self, key):
        result = self.redis_client.delete(key)  # 使用redis客户端删除键
        return result  # 返回删除结果，0表示失败，1表示成功
