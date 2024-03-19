import json

from elasticsearch import Elasticsearch

from auto_detect_app import config


class ElasticSearchService:
    def __init__(self):
        self.es = Elasticsearch([{'host': config.ElasticSearchConfig.ES_HOST,
                                  'port': config.ElasticSearchConfig.ES_PORT}])

    def search(self, index, body, size=10000):
        return self.es.search(index=index, body=body, size=size)

    def insert(self, index, body):
        return self.es.index(index=index, body=body)

    def delete(self, index, doc_id):
        res = self.es.delete(index=index, id=doc_id)
        return res['result']

    def update(self, index, doc_id, doc):
        res = self.es.update(index=index, id=doc_id, body={"doc": doc})
        return res['result']

    def search_logs(self):
        index = 'hdfs_sample_logs_*'
        body = {
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": "2008-11-09T20:35:15",
                        "lte": "2008-11-09T20:35:45"
                    }
                }
            }
        }
        return self.search(index, body)


# main方法测试
if __name__ == '__main__':
    es = ElasticSearchService()
    result = es.search_logs()
    # 打印结果数量
    print(result['hits']['total']['value'])
