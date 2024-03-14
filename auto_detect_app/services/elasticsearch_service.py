from elasticsearch import Elasticsearch

from auto_detect_app import config


class ElasticSearchService:
    def __init__(self):
        self.es = Elasticsearch([{'host': config.ElasticSearchConfig.ES_HOST,
                                  'port': config.ElasticSearchConfig.ES_PORT}])

    def search(self, index, body):
        return self.es.search(index=index, body=body)

    def insert(self, index, body):
        return self.es.index(index=index, body=body)

    def delete(self, index, doc_id):
        res = self.es.delete(index=index, id=doc_id)
        return res['result']

    def update(self, index, doc_id, doc):
        res = self.es.update(index=index, id=doc_id, body={"doc": doc})
        return res['result']
