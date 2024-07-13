import pandas as pd
import numpy as np

def read_File(file_path, N):
    # 指定Excel文件路径
    # excel_file_path = "/home/yinjinjiao/pycharm_project/DNR/data/IEEE33-1.csv"
    data_node = pd.read_csv(file_path).values

    # N = 33  # N节点数   L边数
    L, num_cols = data_node.shape

    Vbase = 12.66 * 1000
    Sbase = 10 * 1000000
    Zbase = Vbase ** 2 / Sbase

    # 初始化数组所有值为0 设置相连的点的边为1
    node_con = np.zeros((N, N), dtype=int)
    r = np.zeros(L)
    x = np.zeros(L)
    for i in range(L):
        n, h = int(data_node[i][1]), int(data_node[i][2])
        node_con[n][h] = 1
        node_con[h][n] = 1
        r[i] = data_node[i][3] / Zbase
        x[i] = data_node[i][4] / Zbase

    # 转换阻抗和导纳 负荷单位转化
    data_node1 = np.zeros((L, num_cols))

    for i in range(L):
        data_node[i][3] = data_node[i][3] / Zbase
        data_node[i][4] = data_node[i][4] / Zbase
        data_node1[i][0] = data_node[i][0]
        data_node1[i][1] = data_node[i][1]
        data_node1[i][2] = data_node[i][2]
        data_node1[i][3] = data_node[i][3] / (data_node[i][3] ** 2 + data_node[i][4] ** 2)
        data_node1[i][4] = data_node[i][4] / (data_node[i][3] ** 2 + data_node[i][4] ** 2)
        data_node1[i][5] = data_node[i][5] * 1000 / Sbase
        data_node1[i][6] = data_node[i][6] * 1000 / Sbase

    return L, data_node1, node_con, r, x


# file_path = "/home/yinjinjiao/pycharm_project/DNR/data/IEEE70/IEEE70-1.csv"
# L, data_node1, node_con, r, x = read_File(file_path, 70)
# print(L)