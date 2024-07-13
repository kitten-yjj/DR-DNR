import csv
import random
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar


def read_File(file_path, N):
    # 指定Excel文件路径
    # excel_file_path = "/home/yinjinjiao/pycharm_project/DNR/data/IEEE33-1.csv"
    data_node = pd.read_csv(file_path).values

    # N = 33  # N节点数   L边数
    L, num_cols = data_node.shape

    Vbase = 12.66 * 1000
    Sbase = 10 * 1000000
    Zbase = Vbase ** 2 / Sbase
    weigh = [0.64,0.6,0.58,0.56,0.56,0.58,0.64,0.76,0.87,0.95,0.99,1,0.99,1,1,0.97,0.96,0.96,0.93,0.92,0.92,0.93,0.87,0.72]

    node_load = np.zeros((24, N, 2))
    for t in range(24):
        for i in range(0, 68):
            # print(data_node[i-1][5])
            j = int(data_node[i][2])
            node_load[t][j][0] = (data_node[i][5] * weigh[t]) / Sbase * 1000
            node_load[t][j][1] = (data_node[i][6] * weigh[t]) / Sbase * 1000

    return node_load

# excel_file_path = "/home/yinjinjiao/pycharm_project/DNR/data/IEEE33-1.csv"

