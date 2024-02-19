# /Users/Bleeding/Desktop/syjk2.txt
# 读取txt文件的每一行，按照逗号分割列
def readtxt(col):
    file = open("/Users/Bleeding/Desktop/syjk2.txt", "r")
    res_list = []
    for line in file.readlines():
        columns = line.split(",")
        if len(columns) > col:
            res_list.append(columns[col].strip())
    file.close()

    res_set = set(res_list)
    for item in res_set:
        print("\"" + item + "\",")


def readtxt(col, value):
    """
    读取txt文件的每一行，按照逗号分割列，找到第0列中值为value的行，打印输出第col列
    输出的每一行顺序与原文件中的顺序一致
    :param col: 1 or 2 表示要打印输出的列
    :param value: 第0列中的值
    :return:
    """
    file = open("/Users/Bleeding/Desktop/syjk2.txt", "r")
    res_list = []
    for line in file.readlines():
        columns = line.split(",")
        if len(columns) > col:
            if columns[0].strip() == value:
                res_list.append(columns[col].strip())
    file.close()

    res_set = set(res_list)
    for item in res_set:
        print("\"" + item + "\",")


readtxt(2, "sb_itszt_service_provider")
