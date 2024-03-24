from elasticsearch import Elasticsearch, helpers

from auto_detect_app import config


class ElasticSearchService:
    def __init__(self):
        self.es = Elasticsearch([{'host': config.ElasticSearchConfig.ES_HOST,
                                  'port': config.ElasticSearchConfig.ES_PORT}])

    def search(self, index, body, size=1000):
        return self.es.search(index=index, body=body, size=size)

    def search_logs(self):
        index = 'hdfs_sample_logs_*'
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match_phrase": {"content": "blk_-5939456537580499544"}},
                        {"match_phrase": {"content": "blk_8338750703123779266"}},
                        {"match_phrase": {"content": "blk_-1608999687919862906"}},
                        {"match_phrase": {"content": "blk_7503483334202473044"}},
                        {"match_phrase": {"content": "blk_-3544583377289625738"}},
                        {"match_phrase": {"content": "blk_7359269325129318656"}},
                    ]
                }
                # ,
                # "range": {
                #     "datetime": {
                #         "gte": "2008-11-08T20:35:15",
                #         "lte": "2008-11-12T20:35:45"
                #     }
                # }
            },
            "sort": [
                {"datetime": {"order": "asc"}}
            ]
        }
        return self.search(index, body)

    def insert(self, index, body):
        return self.es.index(index=index, body=body)

    def delete(self, index, doc_id):
        res = self.es.delete(index=index, id=doc_id)
        return res['result']

    def update(self, index, doc_id, body):
        res = self.es.update(index=index, id=doc_id, body=body)
        return res['result']

    def update_logs_bulk(self, doc_ids):
        index = 'hdfs_2k_sample_logs_*'

        def generate_bulk_data():
            for doc_id in doc_ids:
                # 根据索引名称前缀"xxx_index_*"和文档的"_id"查询出对应的索引名称"_index"
                search_body = {
                    "query": {
                        "match": {
                            "_id": doc_id
                        }
                    }
                }
                res = self.es.search(index=index, body=search_body)
                index_name = res['hits']['hits'][0]['_index']
                # 装填批量更新数据
                yield {
                    "_op_type": 'update',
                    "_index": index_name,
                    "_id": doc_id,
                    "doc": {
                        "is_abnormal": True
                    }
                }

        helpers.bulk(self.es, generate_bulk_data())


# main方法测试
if __name__ == '__main__':
    es = ElasticSearchService()
    es.update_logs_bulk(['2IzOX44BFKwXE7K_69ue', 'aYzOX44BFKwXE7K_69ue'])
