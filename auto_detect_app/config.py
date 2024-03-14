import logging


class DatabaseConfig:
    DB_HOST = 'localhost'
    DB_PORT = 3306
    DB_USER = 'user'
    DB_PASSWORD = 'password'
    DB_NAME = 'database'


class ElasticSearchConfig:
    ES_HOST = 'localhost'
    ES_PORT = 9200


def configure_logger():
    """
    配置日志
    :return: logger
    """
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)

    # 文件打印
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # 控制台打印
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
