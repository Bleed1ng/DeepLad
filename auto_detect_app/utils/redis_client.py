import pandas as pd


class RedisClient:
    # 初始化方法，接收一个Flask应用对象作为参数
    def __init__(self):
        from auto_detect_app.app import redis_client
        self.redis_client = redis_client

    def set(self, key, value):
        self.redis_client.set(key, value)

    def get(self, key):
        value = self.redis_client.get(key)
        if value:
            return value.decode()
        else:
            return None

    def delete(self, key):
        result = self.redis_client.delete(key)
        return result  # 返回删除结果，0表示失败，1表示成功

    def get_parse_result(self):
        """
        查询所有的日志键和模版，前缀为 log_key_
        :return:
        """
        keys = self.redis_client.keys('log_key_*')
        result = {}
        for key in keys:
            result[key.decode()] = self.redis_client.get(key).decode()
        return result

    def load_parse_to_redis(self):
        """ file -> redis
        项目启动时调用一次，从文件中加载日志键和模版，存储到Redis缓存中
        :return:
        """
        with open('../data/spell_result/HDFS.log_templates.csv', 'r') as file:
            df = pd.read_csv(file)
            for row in df.itertuples():
                log_key = row[1]
                log_template = row[2]
                self.set('log_key_' + str(log_key), log_template)

    def load_parse_to_file(self):
        """ redis -> file
        从Redis中加载日志键和模版，全覆盖存储到文件中
        :return:
        """
        keys = self.redis_client.keys('log_key_*')
        result = []
        for key in keys:
            result.append([key.decode(), self.redis_client.get(key).decode()])
        df = pd.DataFrame(result, columns=['log_key', 'log_template'])
        df.to_csv('../data/spell_result/HDFS.log_templates.csv', index=False)
