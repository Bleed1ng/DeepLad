import Spell

# ===== HDFS =====
input_dir = '../../data/HDFS_2k/'
log_file = 'HDFS_2k.log'
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # 日志格式

# ===== BGL =====
# input_dir = '../../../dataset/BGL/'
# log_file = 'BGL.log'
# log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'

# input_dir = '../../../dataset/syitsm_logs/'
# log_file = 'syitsm.log'
# log_format = '[<DateTime>][<Level>][<RequestId>][<Thread>][<Tid>][<CodeLine>] - <Content>'

output_dir = '../../data/spell_result/'  # 解析结果的输出目录
tau = 0.5  # Message type threshold (default: 0.5)
regex = []  # Regular expression list for optional preprocessing (default: [])

parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.parse(log_file)
