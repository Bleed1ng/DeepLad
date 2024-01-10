import re
import os
import pandas as pd
import hashlib
from datetime import datetime
import string

# 把完整日志的header去掉，只保留json内容

input_dir = '/Users/Bleeding/Desktop/xwrz/'
output_dir = '/Users/Bleeding/Desktop/es_input/'
file_name = 'datacollect_app_error_data.2023-11-08.0.log'

with open(input_dir + file_name, 'r') as input_file:
    lines = input_file.readlines()[:100]

stripped_lines = [line[25:] for line in lines]

with open(output_dir + 'es_input_error.json', 'w') as output_file:
    output_file.writelines(stripped_lines)

