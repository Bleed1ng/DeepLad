import csv

input_dir = '../../data/logs_dataset/Spell_result/'  # The input directory of log file
output_dir = '../../data/logs_dataset/param_vec_result/'  # The output directory of parsing results
file_name = 'HDFS_2k.log_structured.csv'


def param_value_sample(file_path):
    """
    从csv文件中读取参数列表
    :param file_path:
    :return:
    """
    parameter_lists = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            parameter_lists.append(row['ParameterList'])
    return parameter_lists


file_path = input_dir + file_name
param_vec_list = param_value_sample(file_path)
# 按行打印列表
for param_vec in param_vec_list:
    print(param_vec)
