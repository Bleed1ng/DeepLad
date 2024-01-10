#!/usr/bin/env python
import sys

from logparser import Spell

input_dir = '../../data/logs_dataset/HDFS/'  # The input directory of log file
output_dir = '../../data/logs_dataset/Spell_result/'  # The output directory of parsing results
log_file = 'HDFS_2k.log'  # The input log file name
log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
tau = 0.5  # Message type threshold (default: 0.5)
regex = []  # Regular expression list for optional preprocessing (default: [])

parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.parse(log_file)
