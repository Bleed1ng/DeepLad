import os
import re
import string
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from auto_detect_app.utils import redis_client


class LCSObject:
    """ Class object to store a log group with the same template
    """

    def __init__(self, log_template=[], log_id_list=[]):
        self.log_template = log_template
        self.log_id_list = log_id_list


class LogCluster:
    """ 用于存储具有相同模板的日志组的类对象
    """

    def __init__(self, log_key=-1, log_template=[], log_id_list=[]):
        self.log_key = log_key
        self.log_template = log_template
        self.log_id_list = log_id_list
        self.size = len(log_id_list)


class Node:
    """ A node in prefix tree data structure
    """

    def __init__(self, token='', templateNo=0):
        self.log_cluster = None
        self.token = token
        self.templateNo = templateNo
        self.childD = dict()


class LogParser:
    """ LogParser class

    Attributes
    ----------
        path : the path of the input file
        log_name : the file name of the input file
        savePath : the path of the output file
        tau : how much percentage of tokens matched to merge a log message (合并日志消息匹配标记的百分比是多少)
    """

    def __init__(self, indir='./', outdir='./result/', log_format=None, tau=0.5, rex=[], keep_para=True):
        self.path = indir
        self.log_name = None
        self.savePath = outdir
        self.tau = tau
        self.logformat = log_format
        self.df_log = None
        self.rex = rex
        self.keep_para = keep_para

    def LCS(self, seq1, seq2):
        """
        计算两个序列的最长公共子序列
        :param seq1:
        :param seq2:
        :return:
        """
        lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1 - 1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2 - 1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1 - 1] == seq2[lenOfSeq2 - 1]
                result.insert(0, seq1[lenOfSeq1 - 1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def SimpleLoopMatch(self, log_cluster_list, seq):
        for log_cluster in log_cluster_list:
            if float(len(log_cluster.log_template)) < 0.5 * len(seq):
                continue
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            token_set = set(seq)
            if all(token in token_set or token == '<*>' for token in log_cluster.log_template):
                return log_cluster
        return None

    def PrefixTreeMatch(self, parentn, seq, idx):
        retLogClust = None
        length = len(seq)
        for i in range(idx, length):
            if seq[i] in parentn.childD:
                childn = parentn.childD[seq[i]]
                if childn.log_cluster is not None:
                    const_log_message = [w for w in childn.log_cluster.log_template if w != '<*>']
                    if float(len(const_log_message)) >= self.tau * length:
                        return childn.log_cluster
                else:
                    return self.PrefixTreeMatch(childn, seq, i + 1)
        return retLogClust

    def LCSMatch(self, log_cluster_list, seq):
        retLogClust = None

        maxLen = -1
        maxlcs = []
        maxClust = None
        set_seq = set(seq)
        size_seq = len(seq)
        for log_cluster in log_cluster_list:
            set_template = set(log_cluster.log_template)
            if len(set_seq & set_template) < 0.5 * size_seq:
                continue
            lcs = self.LCS(seq, log_cluster.log_template)
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(log_cluster.log_template) < len(maxClust.log_template)):
                maxLen = len(lcs)
                maxlcs = lcs
                maxClust = log_cluster

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, lcs, seq):
        """
        生成日志模版
        :param lcs:
        :param seq:
        :return:
        """
        retVal = []
        if not lcs:
            return retVal

        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append('<*>')
            if not lcs:
                break
        if i < len(seq):
            retVal.append('<*>')
        return retVal

    def addSeqToPrefixTree(self, rootn, new_cluster):
        parentn = rootn
        seq = new_cluster.log_template
        seq = [w for w in seq if w != '<*>']

        for i in range(len(seq)):
            tokenInSeq = seq[i]
            # Match
            if tokenInSeq in parentn.childD:
                parentn.childD[tokenInSeq].templateNo += 1
                # Do not Match
            else:
                parentn.childD[tokenInSeq] = Node(token=tokenInSeq, templateNo=1)
            parentn = parentn.childD[tokenInSeq]

        if parentn.log_cluster is None:
            parentn.log_cluster = new_cluster

    def removeSeqFromPrefixTree(self, rootn, new_cluster):
        parentn = rootn
        seq = new_cluster.log_template
        seq = [w for w in seq if w != '<*>']

        for tokenInSeq in seq:
            if tokenInSeq in parentn.childD:
                matchedNode = parentn.childD[tokenInSeq]
                if matchedNode.templateNo == 1:
                    del parentn.childD[tokenInSeq]
                    break
                else:
                    matchedNode.templateNo -= 1
                    parentn = matchedNode

    def printTree(self, node, dep):
        pStr = ''
        for i in xrange(dep):
            pStr += '\t'

        if node.token == '':
            pStr += 'Root'
        else:
            pStr += node.token
            if node.log_cluster is not None:
                pStr += '-->' + ' '.join(node.log_cluster.log_template)
        print(pStr + ' (' + str(node.templateNo) + ')')

        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def output_result_file(self, log_cluster_list):
        """
        将日志聚类结果输出到csv文件
        :param log_cluster_list:
        :return:
        """
        templates = [0] * self.df_log.shape[0]
        log_key_seq = [0] * self.df_log.shape[0]
        df_event = []
        event_id = 0
        # 按log模版的分类，分配event_id
        for log_cluster in log_cluster_list:
            template_str = ' '.join(log_cluster.log_template)
            event_id += 1
            for log_id in log_cluster.log_id_list:
                templates[log_id - 1] = template_str
                log_key_seq[log_id - 1] = event_id
            df_event.append([event_id, template_str, len(log_cluster.log_id_list)])
        df_event = pd.DataFrame(df_event, columns=['log_key', 'log_template', 'occurrences'])

        self.df_log['log_key'] = log_key_seq
        self.df_log['log_template'] = templates
        if self.keep_para:
            self.df_log["param_vec_list"] = self.df_log.apply(self.get_parameter_list, axis=1)
        # df_log只保留log_key, content两列
        self.df_log = self.df_log[['log_key', 'content']]
        self.df_log.to_csv(os.path.join(self.savePath, self.log_name + '_structured.csv'), index=False)

        df_event.to_csv(os.path.join(self.savePath, self.log_name + '_templates.csv'), index=False)

    def parse(self, log_name):
        start_time = datetime.now()
        self.log_name = log_name
        self.load_data()
        root_node = Node()
        log_cluster_list = []

        for idx, line in tqdm(self.df_log.iterrows(),
                              desc='Parsing file: ' + os.path.join(self.path, log_name),
                              total=self.df_log.shape[0]):
            log_id = line['LineId']
            log_message_list = list(
                filter(
                    lambda x: x != '',
                    re.split(r'[\s=:,]', self.preprocess(line['content']))
                )
            )
            const_log_message_list = [w for w in log_message_list if w != '<*>']

            # Find an existing matched log cluster
            match_cluster = self.PrefixTreeMatch(root_node, const_log_message_list, 0)
            if match_cluster is None:
                match_cluster = self.SimpleLoopMatch(log_cluster_list, const_log_message_list)
                if match_cluster is None:
                    match_cluster = self.LCSMatch(log_cluster_list, log_message_list)
                    # Match no existing log cluster
                    if match_cluster is None:
                        new_cluster = LCSObject(log_template=log_message_list, log_id_list=[log_id])
                        log_cluster_list.append(new_cluster)
                        self.addSeqToPrefixTree(root_node, new_cluster)
                    # Add the new log message to the existing cluster
                    else:
                        new_template = self.getTemplate(self.LCS(log_message_list, match_cluster.log_template),
                                                        match_cluster.log_template)
                        if ' '.join(new_template) != ' '.join(match_cluster.log_template):
                            self.removeSeqFromPrefixTree(root_node, match_cluster)
                            match_cluster.log_template = new_template
                            self.addSeqToPrefixTree(root_node, match_cluster)
            if match_cluster:
                match_cluster.log_id_list.append(log_id)

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        self.output_result_file(log_cluster_list)
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def parse_log_from_list(self, batch_log_list):
        """
        从日志列表集合中解析日志，以现有的模版进行匹配，如果没有匹配到则新建一个模版
        :param batch_log_list: 待检测的日志列表
        :return: log_key_list: [log_id, content, log_key] 解析后得到日志键的日志列表
        """
        root_node = Node()
        log_key_list = []
        log_cluster_list = []
        # 从文件中加载已有的模版
        # self.savePath = '../data/spell_result/'
        # log_templates_file = os.path.join(self.savePath, 'HDFS.log_templates.csv')
        # if os.path.exists(log_templates_file):
        #     df_event = pd.read_csv(log_templates_file)
        #     for idx, row in df_event.iterrows():
        #         log_cluster = LogCluster(log_key=row['log_key'],
        #                                  log_template=row['log_template'].split(),
        #                                  log_id_list=[])
        #         self.addSeqToPrefixTree(root_node, log_cluster)  # 将现有的日志模版存入前缀树
        #         log_cluster_list.append(log_cluster)

        # 从Redis中加载已有的模版（如有）
        redis = redis_client.RedisClient()
        log_template_list = redis.get_parse_result()
        for log_key, log_template in log_template_list.items():
            log_cluster = LogCluster(log_key=int(log_key.split('_')[-1]),
                                     log_template=log_template.split(),
                                     log_id_list=[])
            self.addSeqToPrefixTree(root_node, log_cluster)
            log_cluster_list.append(log_cluster)

        for line in batch_log_list:
            log_id = line['log_id']
            content = line['content']
            log_message_list = list(
                filter(
                    lambda x: x != '',
                    re.split(r'[\s=:,]', self.preprocess(line['content']))
                )
            )
            const_log_message_list = [w for w in log_message_list if w != '<*>']

            # 匹配现有的日志模版
            match_cluster = self.PrefixTreeMatch(root_node, const_log_message_list, 0)
            if match_cluster is None:
                match_cluster = self.SimpleLoopMatch(log_cluster_list, const_log_message_list)
                if match_cluster is None:
                    match_cluster = self.LCSMatch(log_cluster_list, log_message_list)
                    # 三种方式均没有匹配到现有的日志模版，则新建一个日志模版，log_key根据已有的模版数量+1
                    if match_cluster is None:
                        new_cluster = LogCluster(log_key=len(log_cluster_list) + 1,
                                                 log_template=log_message_list,
                                                 log_id_list=[log_id])
                        log_cluster_list.append(new_cluster)
                        self.addSeqToPrefixTree(root_node, new_cluster)
                    # 用LCS匹配到了现有的日志模版
                    else:
                        new_template = self.getTemplate(
                            self.LCS(log_message_list, match_cluster.log_template),
                            match_cluster.log_template
                        )
                        if ' '.join(new_template) != ' '.join(match_cluster.log_template):
                            self.removeSeqFromPrefixTree(root_node, match_cluster)
                            match_cluster.log_template = new_template
                            match_cluster.size = len(match_cluster.log_id_list)
                            self.addSeqToPrefixTree(root_node, match_cluster)
            if match_cluster:
                match_cluster.log_id_list.append(log_id)
                match_cluster.size = len(match_cluster.log_id_list)
                # 同时存入Redis
                redis.set('log_key_' + str(match_cluster.log_key), ' '.join(match_cluster.log_template))

            # 将解析完的日志键序列存入log_key_list
            log_dict = {
                'log_id': log_id,
                'log_key': match_cluster.log_key if match_cluster else -1,
                'content': content
            }
            log_key_list.append(log_dict)
        return log_key_list

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.log_name),
                                            regex,
                                            headers)

    def preprocess(self, line):
        """预处理日志消息，将日志消息中的正则表达式匹配的内容替换为'<*>'
        """
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers):
        """ 将日志数据转换为DataFrame
        """
        log_messages = []
        line_count = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    line_count += 1
                except Exception as e:
                    pass
        log_df = pd.DataFrame(log_messages, columns=headers)
        log_df.insert(0, 'LineId', None)
        log_df['LineId'] = [i + 1 for i in range(line_count)]
        return log_df

    def generate_logformat_regex(self, logformat):
        """ 生成拆分日志消息的正则表达式
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                # '\s+' --> '\\\\s+' 没懂为啥？？？
                splitter = re.sub(' +', '\\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"\s<.{1,5}>\s", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'[^A-Za-z0-9]+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        parameter_list = [para.strip(string.punctuation).strip(' ') for para in parameter_list]
        return parameter_list
