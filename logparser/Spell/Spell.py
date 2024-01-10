"""
Description : This file implements the Spell algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import re
import os
import pandas as pd
import hashlib
from datetime import datetime
import string


class LCSObject:
    """ Class object to store a log group with the same template
    """

    def __init__(self, logTemplate=[], log_id_list=[]):
        self.logTemplate = logTemplate
        self.log_id_list = log_id_list


class Node:
    """ A node in prefix tree data structure
    """

    def __init__(self, token='', templateNo=0):
        self.logClust = None
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

    def SimpleLoopMatch(self, log_clust_list, seq):
        for logClust in log_clust_list:
            if float(len(logClust.logTemplate)) < 0.5 * len(seq):
                continue
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            token_set = set(seq)
            if all(token in token_set or token == '<*>' for token in logClust.logTemplate):
                return logClust
        return None

    def PrefixTreeMatch(self, parentn, seq, idx):
        retLogClust = None
        length = len(seq)
        for i in range(idx, length):
            if seq[i] in parentn.childD:
                childn = parentn.childD[seq[i]]
                if childn.logClust is not None:
                    const_log_message = [w for w in childn.logClust.logTemplate if w != '<*>']
                    if float(len(const_log_message)) >= self.tau * length:
                        return childn.logClust
                else:
                    return self.PrefixTreeMatch(childn, seq, i + 1)

        return retLogClust

    def LCSMatch(self, log_clust_list, seq):
        retLogClust = None

        maxLen = -1
        maxlcs = []
        maxClust = None
        set_seq = set(seq)
        size_seq = len(seq)
        for logClust in log_clust_list:
            set_template = set(logClust.logTemplate)
            if len(set_seq & set_template) < 0.5 * size_seq:
                continue
            lcs = self.LCS(seq, logClust.logTemplate)
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(logClust.logTemplate) < len(maxClust.logTemplate)):
                maxLen = len(lcs)
                maxlcs = lcs
                maxClust = logClust

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, lcs, seq):
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
        seq = new_cluster.logTemplate
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

        if parentn.logClust is None:
            parentn.logClust = new_cluster

    def removeSeqFromPrefixTree(self, rootn, new_cluster):
        parentn = rootn
        seq = new_cluster.logTemplate
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

    # 将日志聚类结果输出到文件
    def output_result_file(self, log_clust_list):
        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        df_event = []
        event_id = 0
        # 按log模版的分类，分配event_id
        for log_clust in log_clust_list:
            template_str = ' '.join(log_clust.logTemplate)
            event_id += 1
            # 或根据log模版生成md5值，取前8位作为event_id
            # event_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for log_id in log_clust.log_id_list:
                templates[log_id - 1] = template_str
                ids[log_id - 1] = event_id
            df_event.append([event_id, template_str, len(log_clust.log_id_list)])
        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.log_name + '_structured.csv'), index=False)
        df_event.to_csv(os.path.join(self.savePath, self.log_name + '_templates.csv'), index=False)

    def printTree(self, node, dep):
        pStr = ''
        for i in xrange(dep):
            pStr += '\t'

        if node.token == '':
            pStr += 'Root'
        else:
            pStr += node.token
            if node.logClust is not None:
                pStr += '-->' + ' '.join(node.logClust.logTemplate)
        print(pStr + ' (' + str(node.templateNo) + ')')

        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def parse(self, log_name):
        start_time = datetime.now()
        print('Parsing file: ' + os.path.join(self.path, log_name))
        self.log_name = log_name
        self.load_data()
        rootNode = Node()
        log_clust_list = []

        count = 0
        for idx, line in self.df_log.iterrows():
            log_id = line['LineId']
            log_message_list = list(
                filter(
                    lambda x: x != '',
                    re.split(r'[\s=:,]', self.preprocess(line['Content']))
                )
            )
            const_log_message_list = [w for w in log_message_list if w != '<*>']

            # Find an existing matched log cluster
            match_cluster = self.PrefixTreeMatch(rootNode, const_log_message_list, 0)

            if match_cluster is None:
                match_cluster = self.SimpleLoopMatch(log_clust_list, const_log_message_list)

                if match_cluster is None:
                    match_cluster = self.LCSMatch(log_clust_list, log_message_list)

                    # Match no existing log cluster
                    if match_cluster is None:
                        new_cluster = LCSObject(logTemplate=log_message_list, log_id_list=[log_id])
                        log_clust_list.append(new_cluster)
                        self.addSeqToPrefixTree(rootNode, new_cluster)
                    # Add the new log message to the existing cluster
                    else:
                        newTemplate = self.getTemplate(self.LCS(log_message_list, match_cluster.logTemplate),
                                                       match_cluster.logTemplate)
                        if ' '.join(newTemplate) != ' '.join(match_cluster.logTemplate):
                            self.removeSeqFromPrefixTree(rootNode, match_cluster)
                            match_cluster.logTemplate = newTemplate
                            self.addSeqToPrefixTree(rootNode, match_cluster)
            if match_cluster:
                match_cluster.log_id_list.append(log_id)

            # 打印进度
            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.output_result_file(log_clust_list)
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.log_name),
                                            regex,
                                            headers,
                                            self.logformat)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe 
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
        """ Function to generate regular expression to split log messages
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
