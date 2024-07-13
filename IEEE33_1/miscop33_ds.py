import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from DNR.IEEE33 import readLoad
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from DNR.IEEE33_1 import readFile

# 记录开始时间
start_time = time.time()

# 定义目标函数的系数
N = 33
L = 37
T1 = 24
s_on = 5
baseMVA = 10
Vbase = 12.66 * 1000
Sbase = 10000000
w_no = 50
cLoss = 1
cP = 0
cW = 0.5
# 设置w-距离的相关参数
# e = 0.0068065967597781205
e = 0.0000068
probability = 0.2
Sg = {8, 25, 30}
# Sg = {6, 25, 30}
# Sg = {8, 24, 30}
Sw = {0, 8, 25, 30}
# Sg = {8, 25, 31}

# fileURL = "/home/yinjinjiao/pycharm_project/DNR/data/IEEE33/"
fileURL = "/Users/yinjinjiao/Documents/project/DNR/data/IEEE33/"

# 读取节点数据 L边数，data_node数据矩阵，node_con节点连接矩阵，r电阻矩阵
file_path = fileURL + "IEEE33-1.csv"
L, data_node1, node_con, r_ij, x_ij = readFile.read_File(file_path, 33)
data_node = pd.read_csv(file_path).values

# 获取节点T1小时的负荷
data_load = readLoad.read_File(file_path, 33)

# 获取分布式发电得历史数据
PG_samples = np.zeros((T1, N, w_no))
DG = [f"{fileURL}DG{i}-{w_no}.csv" for i in range(N)]
for t in range(T1):
    for i in range(N):
        if i in Sg:
            PG_samples[t][i] = pd.read_csv(DG[i]).iloc[t] / Sbase * 1000

r_ij = np.zeros(L)
x_ij = np.zeros(L)
to_node = np.zeros((N, L))   # 流入节点的支路
from_node = np.zeros((N, L))  # 流出节点的支路
U_max = [1.06 ** 2] * N
U_min = [0.94 ** 2] * N
Sij_max = 15 / baseMVA
M = 1.06 ** 2 - 0.94 ** 2
for i in range(L):
    r_ij[i] = data_node[i][3] / (Vbase ** 2 / Sbase)
    x_ij[i] = data_node[i][4] / (Vbase ** 2 / Sbase)
for k in range(L):
    to_node[int(data_node[k][2])][k] = 1
    from_node[int(data_node[k][1])][k] = 1


ak = np.zeros(2)
bk = np.zeros(2)
ck = np.zeros(2)
Di = np.zeros((T1, N, 2))
Ci = np.zeros((T1, N, 2))
ak[0] = 0 * cW
ak[1] = 1 / (1 - probability) * cW
bk[0] = 1 * cW
bk[1] = 1 * cW - 1 / (1 - probability) * cW
ck[0] = 0
ck[1] = -1 / (1 - probability) * cW
# x <= 220  and -x >= -150
for t1 in range(T1):
    for i in range(N):
        if i in Sg:
            Ci[t1][i][0] = -1
            Ci[t1][i][1] = 1
            Di[t1][i][0] = -np.min(PG_samples[t1][i])
            Di[t1][i][1] = np.max(PG_samples[t1][i])

weght = [0.2, 0.4, 0.3, 0.18, 0.17, 0.2, 0.22, 0.2, 0.19, 0.17, 0.2, 0.22, 0.38, 0.58, 0.79, 0.81, 0.72, 0.78, 0.71,
        0.73, 0.6, 0.4, 0.28, 0.29]
pg_av = np.zeros((T1, N))
pg_max = np.zeros((T1, N))
pg_min = np.zeros((T1, N))
qg_max = np.zeros((T1, N))
qg_min = np.zeros((T1, N))
for t1 in range(T1):  # 修改数据的时候记得改这里
    pg_max[t1][0] = 1
    qg_max[t1][0] = 1
    qg_min[t1][0] = -1
    for i in Sg:
        pg_av[t1][i] = np.average(PG_samples[t1][i])
        pg_min[t1][i] = np.min(PG_samples[t1][i])
        pg_max[t1][i] = np.max(PG_samples[t1][i])
        qg_max[t1][i] = np.max(PG_samples[t1][i])
s_num = [4]*24
s_num[0] = 10

def ro_solve(t_start, z0_ij):
    # 创建模型
    m = gp.Model()
    # 定义变量
    z_ij = m.addVars(L, lb=0, ub=1, vtype=GRB.BINARY, name="z_ij")
    kl = m.addVars(N, N, lb=0, ub=1, vtype=GRB.BINARY, name="kl")
    rr_ij = m.addVars(L, lb=0, ub=1, vtype=GRB.BINARY, name="rr_ij")
    b_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_ij")
    U = m.addVars(T1, N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="U")
    L_ij = m.addVars(T1, L, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_ij")
    P_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_ij")
    Q_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_ij")
    P_g = m.addVars(T1, N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_g")
    Q_g = m.addVars(T1, N, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_g")

    # 创建目标函数
    obj = gp.LinExpr()
    obj1 = gp.LinExpr()
    for t1 in range(T1):
        for h in range(L):
            obj += (r_ij[h] * L_ij[t1, h])*cLoss
        obj1 += P_g[t1, 0]
    obj += obj1 * cP
    m.setObjective(obj, GRB.MINIMIZE)

    # 添加约束
    for t1 in range(T1):
        for j in range(N):
            Pjk = gp.LinExpr()
            Qjk = gp.LinExpr()
            Pij = gp.LinExpr()
            Qij = gp.LinExpr()
            for k in range(L):
                if from_node[j][k] == 1:
                    Pjk += P_ij[t1, k]
                    Qjk += Q_ij[t1, k]
            for i in range(L):
                if to_node[j][i] == 1:
                    Pij += P_ij[t1, i] - r_ij[i] * L_ij[t1, i]
                    Qij += Q_ij[t1, i] - x_ij[i] * L_ij[t1, i]
            m.addConstr(P_g[t1, j] - data_load[t1+t_start][j][0] + Pij - Pjk == 0, name="")
            m.addConstr(Q_g[t1, j] - data_load[t1+t_start][j][1] + Qij - Qjk == 0, name="")

    # for t1 in range(T1):
        # for h in range(L):
        #     i, j = int(data_node[h][1]), int(data_node[h][2])
        #     m.addConstr(U[t1, i] - U[t1, j] <=
        #                 (1 - z_ij[h]) * M + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
        #                 - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
        #     m.addConstr(U[t1, i] - U[t1, j] >=
        #                 -(1 - z_ij[h]) * M + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
        #                 - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            m.addConstr(U[t1, i] - U[t1, j] ==
                        b_ij[t1, h] + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
                        - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
            m.addConstr(b_ij[t1, h] <= M * (1 - z_ij[h]), name="")
            m.addConstr(b_ij[t1, h] >= -M * (1 - z_ij[h]), name="")

    for t1 in range(T1):
        for h in range(L):
            m.addConstr(Sij_max ** 2 * z_ij[h] >= P_ij[t1, h] ** 2 + Q_ij[t1, h] ** 2, name="")

    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            m.addConstr(4 * (P_ij[t1, h] ** 2 + Q_ij[t1, h] ** 2) + (L_ij[t1, h] - U[t1, i]) ** 2
                        <= (L_ij[t1, h] + U[t1, i]) ** 2, name="")

    m.addConstr(N - 1 == gp.quicksum(z_ij[h] for h in range(L)), name="")
    for h in range(L):
        i, j = int(data_node[h][1]), int(data_node[h][2])
        m.addConstr(z_ij[h] == kl[i, j] + kl[j, i], name="")
    lin = gp.LinExpr()
    for i in range(1, N):
        lin = 0
        for j in range(N):
            if node_con[i][j] == 1:
                lin += kl[i, j]
        m.addConstr(lin == 1, name="")
    for j in range(N):
        m.addConstr((0 == kl[0, j]), name="")

    for h in range(L):
        m.addConstr(z0_ij[h] - z_ij[h] <= rr_ij[h], name="")
        m.addConstr(z_ij[h] - z0_ij[h] <= rr_ij[h], name="")
    m.addConstr(gp.quicksum(rr_ij[h] for h in range(L)) <= s_num[t_start], name="")

    for t1 in range(T1):
        m.addConstr(U[t1, 0] == 1, name="")

    cos_x = 0.9
    sin_x = np.sqrt(1 - cos_x ** 2)
    tan_x = sin_x / cos_x
    for t1 in range(T1):
        for i in range(N):
            if i in Sg:
                m.addConstr(P_g[t1, i] <= pg_min[t1 + t_start][i], name="")
                m.addConstr(P_g[t1, i] >= 0, name="")
                m.addConstr(Q_g[t1, i] == tan_x * P_g[t1, i], name="")
    for t1 in range(T1):
        for i in range(N):
            if i not in Sg:
                m.addConstr(P_g[t1, i] <= pg_max[t1 + t_start][i], name="")
                m.addConstr(P_g[t1, i] >= pg_min[t1 + t_start][i], name="")
                m.addConstr(Q_g[t1, i] <= qg_max[t1 + t_start][i], name="")
                m.addConstr(Q_g[t1, i] >= qg_min[t1 + t_start][i], name="")
        for i in range(1, N):
            m.addConstr(U[t1, i] <= U_max[i], name="")
            m.addConstr(U[t1, i] >= U_min[i], name="")

    # 模型更新
    m.update()
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 0.001  # 设置目标间隙为 1%
    m.optimize()

    U_value = np.zeros((T1, N))
    zij = np.zeros(L)
    for h in range(L):
        zij[h] = z_ij[h].x
    for t1 in range(T1):
        for i in range(N):
            U_value[t1][i] = np.sqrt(U[t1, i].x)
    u_Max = np.zeros((T1,L))
    U_Max = np.zeros(T1)
    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            if P_ij[t1, h].x > 0 and z_ij[h].x > 0.01:
                u_Max[t1][h] = np.sqrt(U[t1, i].x) - np.sqrt(U[t1, j].x)
            else:
                u_Max[t1][h] = np.sqrt(U[t1, j].x) - np.sqrt(U[t1, i].x)
    for t1 in range(T1):
        U_Max[t1] = np.max(u_Max[t1])
    u_Max = np.zeros((T1, L))
    U_Max = np.zeros(T1)
    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            if P_ij[t1, h].x > 0 and z_ij[h].x > 0.01:
                u_Max[t1][h] = np.sqrt(U[t1, i].x) - np.sqrt(U[t1, j].x)
            else:
                u_Max[t1][h] = np.sqrt(U[t1, j].x) - np.sqrt(U[t1, i].x)
    for t1 in range(T1):
        U_Max[t1] = np.max(u_Max[t1])
    return U_value, zij, obj.getValue(), U_Max, obj1.getValue()
def ds_solve(t_start, z0_ij):
    # 创建模型
    m = gp.Model()
    # 定义变量
    z_ij = m.addVars(L, lb=0, ub=1, vtype=GRB.BINARY, name="z_ij")
    kl = m.addVars(N, N, lb=0, ub=1, vtype=GRB.BINARY, name="kl")
    rr_ij = m.addVars(L, lb=0, ub=1, vtype=GRB.BINARY, name="rr_ij")
    b_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_ij")
    U = m.addVars(T1, N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="U")
    L_ij = m.addVars(T1, L, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_ij")
    P_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_ij")
    Q_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_ij")
    P_g = m.addVars(T1, N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_g")
    Q_g = m.addVars(T1, N, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_g")
    # x = m.addVars(T1, L, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")

    # 创建目标函数
    obj = gp.LinExpr()
    obj1 = gp.LinExpr()
    for t1 in range(T1):
        for h in range(L):
            obj += ((r_ij[h] * L_ij[t1, h])*cLoss)
        obj1 += P_g[t1, 0]
    obj += obj1 * cP
    m.setObjective(obj, GRB.MINIMIZE)

    # 添加约束
    for t1 in range(T1):
        for j in range(N):
            Pjk = gp.LinExpr()
            Qjk = gp.LinExpr()
            Pij = gp.LinExpr()
            Qij = gp.LinExpr()
            for k in range(L):
                if from_node[j][k] == 1:
                    Pjk += P_ij[t1, k]
                    Qjk += Q_ij[t1, k]
            for i in range(L):
                if to_node[j][i] == 1:
                    Pij += P_ij[t1, i] - r_ij[i] * L_ij[t1, i]
                    Qij += Q_ij[t1, i] - x_ij[i] * L_ij[t1, i]
            m.addConstr(P_g[t1, j] - data_load[t1+t_start][j][0] + Pij - Pjk == 0, name="")
            m.addConstr(Q_g[t1, j] - data_load[t1+t_start][j][1] + Qij - Qjk == 0, name="")

    # for t1 in range(T1):
    #     for h in range(L):
    #         i, j = int(data_node[h][1]), int(data_node[h][2])
    #         m.addConstr(U[t1, i] - U[t1, j] <=
    #                     (1 - z_ij[h]) * M + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
    #                     - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
    #         m.addConstr(U[t1, i] - U[t1, j] >=
    #                     -(1 - z_ij[h]) * M + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
    #                     - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            m.addConstr(U[t1, i] - U[t1, j] ==
                        b_ij[t1, h] + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
                        - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
            m.addConstr(b_ij[t1, h] <= M * (1 - z_ij[h]), name="")
            m.addConstr(b_ij[t1, h] >= -M * (1 - z_ij[h]), name="")

    for t1 in range(T1):
        for h in range(L):
            m.addConstr(Sij_max ** 2 * z_ij[h] >= P_ij[t1, h] ** 2 + Q_ij[t1, h] ** 2, name="")

    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            m.addConstr(4 * (P_ij[t1, h] ** 2 + Q_ij[t1, h] ** 2) + (L_ij[t1, h] - U[t1, i]) ** 2
                        <= (L_ij[t1, h] + U[t1, i]) ** 2, name="")


    m.addConstr(N - 1 == gp.quicksum(z_ij[h] for h in range(L)), name="")
    for h in range(L):
        i, j = int(data_node[h][1]), int(data_node[h][2])
        m.addConstr(z_ij[h] == kl[i, j] + kl[j, i], name="")
    lin = gp.LinExpr()
    for i in range(1, N):
        lin = 0
        for j in range(N):
            if node_con[i][j] == 1:
                lin += kl[i, j]
        m.addConstr(lin == 1, name="")
    for j in range(N):
        m.addConstr((0 == kl[0, j]), name="")

    for h in range(L):
        m.addConstr(z0_ij[h] - z_ij[h] <= rr_ij[h], name="")
        m.addConstr(z_ij[h] - z0_ij[h] <= rr_ij[h], name="")
    m.addConstr(gp.quicksum(rr_ij[h] for h in range(L)) <= s_num[t_start], name="")

    for t1 in range(T1):
        m.addConstr(U[t1, 0] == 1, name="")

    cos_x = 0.9
    sin_x = np.sqrt(1 - cos_x ** 2)
    tan_x = sin_x / cos_x
    for t1 in range(T1):
        for i in range(N):
            if i in Sg:
                m.addConstr(P_g[t1, i] == pg_av[t1+t_start][i], name="")
                m.addConstr(Q_g[t1, i] == tan_x * P_g[t1, i], name="")
    for t1 in range(T1):
        for i in range(N):
            if i not in Sg:
                m.addConstr(P_g[t1, i] <= pg_max[t1 + t_start][i], name="")
                m.addConstr(P_g[t1, i] >= pg_min[t1 + t_start][i], name="")
                m.addConstr(Q_g[t1, i] <= qg_max[t1 + t_start][i], name="")
                m.addConstr(Q_g[t1, i] >= qg_min[t1 + t_start][i], name="")
        for i in range(1, N):
            m.addConstr(U[t1, i] <= U_max[i], name="")
            m.addConstr(U[t1, i] >= U_min[i], name="")

    # 模型更新
    m.update()
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 0.001  # 设置目标间隙为 1%
    # 模型更新
    m.update()
    m.optimize()

    U_value = np.zeros((T1, N))
    zij = np.zeros(L)
    for h in range(L):
        zij[h] = z_ij[h].x
    for t1 in range(T1):
        for i in range(N):
            U_value[t1][i] = np.sqrt(U[t1, i].x)

    u_Max = np.zeros((T1, L))
    U_Max = np.zeros(T1)
    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            if P_ij[t1, h].x > 0 and z_ij[h].x > 0.01:
                u_Max[t1][h] = np.sqrt(U[t1, i].x) - np.sqrt(U[t1, j].x)
            else:
                u_Max[t1][h] = np.sqrt(U[t1, j].x) - np.sqrt(U[t1, i].x)
    for t1 in range(T1):
        U_Max[t1] = np.max(u_Max[t1])
    return U_value, zij, obj.getValue(), U_Max, obj1.getValue()
def dro_solve(p, t_start, z0_ij):
    # 创建模型
    m = gp.Model()
    # 定义变量
    z_ij = m.addVars(L, lb=0, ub=1, vtype=GRB.BINARY, name="z_ij")
    kl = m.addVars(N, N, lb=0, ub=1, vtype=GRB.BINARY, name="kl")
    rr_ij = m.addVars(L, lb=0, ub=1, vtype=GRB.BINARY, name="rr_ij")
    b_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_ij")
    U = m.addVars(T1, N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="U")
    L_ij = m.addVars(T1, L, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="L_ij")
    P_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_ij")
    Q_ij = m.addVars(T1, L, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_ij")
    P_g = m.addVars(T1, N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_g")
    Q_g = m.addVars(T1, N, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_g")

    # 目标函数的
    t = m.addVars(T1, N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t")
    s = m.addVars(T1, N, w_no, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="s")
    Ri = m.addVars(T1, N, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Ri")
    r = m.addVars(T1, 2, N, w_no, 2, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r")
    w = m.addVars(T1, 2, N, w_no, 2, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")
    loss = m.addVars(T1, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="loss")

    pai = m.addVars(T1, N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pai")
    yk = m.addVars(T1, N, w_no, lb=float('-inf'), ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="yk")
    r1 = m.addVars(T1, 2, N, w_no, 2, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="r1")
    w1 = m.addVars(T1, 2, N, w_no, 2, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w1")
    x1 = m.addVars(T1, L, lb=float('-inf'), ub=0, vtype=GRB.CONTINUOUS, name="x1")

    p1 = p
    for t1 in range(T1):
        for i in range(N):
            if i in Sg:
                m.addConstr(pai[t1, i] * e + 1.0 / w_no * gp.quicksum(yk[t1, i, j] for j in range(w_no)) <= 0, name="")
                for j in range(w_no):        # 加上t_start
                    m.addConstr(x1[t1, i] +
                                r1[t1, 0, i, j, 0] * (Di[t1+t_start][i][0] - Ci[t1+t_start][i][0] * PG_samples[t1+t_start][i][j]) +
                                r1[t1, 0, i, j, 1] * (Di[t1+t_start][i][1] - Ci[t1+t_start][i][1] * PG_samples[t1+t_start][i][j])
                                <= yk[t1, i, j], name="")
                    m.addConstr(1 / p1 * P_g[t1, i] - 1 / p1 * PG_samples[t1+t_start][i][j] + (1 - 1 / p1) * x1[t1, i] +
                                r1[t1, 1, i, j, 0] * (Di[t1+t_start][i][0] - Ci[t1+t_start][i][0] * PG_samples[t1+t_start][i][j]) +
                                r1[t1, 1, i, j, 1] * (Di[t1+t_start][i][1] - Ci[t1+t_start][i][1] * PG_samples[t1+t_start][i][j])
                                <= yk[t1, i, j], name="")
                    m.addConstr(w1[t1, 0, i, j, 0] + w1[t1, 0, i, j, 1] <= pai[t1, i], name="")
                    m.addConstr(w1[t1, 1, i, j, 0] + w1[t1, 1, i, j, 1] <= pai[t1, i], name="")
                    m.addConstr(w1[t1, 0, i, j, 0] - w1[t1, 0, i, j, 1] ==
                                (Ci[t1+t_start][i][0] * r1[t1, 0, i, j, 0] + Ci[t1+t_start][i][1] * r1[t1, 0, i, j, 1] - 0), name="")
                    m.addConstr(w1[t1, 1, i, j, 0] - w1[t1, 1, i, j, 1] ==
                                (Ci[t1+t_start][i][0] * r1[t1, 1, i, j, 0] + Ci[t1+t_start][i][1] * r1[t1, 1, i, j, 1] + 1 / p1),
                                name="")

    # 创建目标函数
    obj = gp.LinExpr()
    h1 = gp.LinExpr()
    obj1 = gp.LinExpr()
    for t1 in range(T1):
        for i in range(N):
            if i in Sg:
                total = 0
                for j in range(w_no):
                    total += s[t1, i, j]
                obj += (Ri[t1, i] * e + total / w_no)
    for t1 in range(T1):
        m.addConstr(loss[t1] == gp.quicksum(r_ij[h] * L_ij[t1, h] for h in range(L)), name="")
        h1 += loss[t1]
        obj1 += P_g[t1, 0]
    obj = h1*cLoss + obj + obj1*cP
    m.setObjective(obj, GRB.MINIMIZE)

    for t1 in range(T1):
        for k in range(2):
            for i in range(N):
                if i in Sg:
                    for j in range(w_no):
                        m.addConstr( # loss[t1] / 3 +
                            (bk[k] * t[t1, i] + ck[k] * P_g[t1, i] + ak[k] * PG_samples[t1+t_start][i][j] +
                             r[t1, k, i, j, 0] * (Di[t1+t_start][i][0] - Ci[t1+t_start][i][0] * PG_samples[t1+t_start][i][j]) +
                             r[t1, k, i, j, 1] * (Di[t1+t_start][i][1] - Ci[t1+t_start][i][1] * PG_samples[t1+t_start][i][j]))
                            <= s[t1, i, j], name="")
                        m.addConstr(w[t1, k, i, j, 0] + w[t1, k, i, j, 1] <= Ri[t1, i], name="")
                        m.addConstr(w[t1, k, i, j, 0] - w[t1, k, i, j, 1] ==
                                    (Ci[t1+t_start][i][0] * r[t1, k, i, j, 0] + Ci[t1+t_start][i][1] * r[t1, k, i, j, 1] - ak[k]), name="")

    m.addConstr(N - 1 == gp.quicksum(z_ij[h] for h in range(L)), name="")
    for h in range(L):
        i, j = int(data_node[h][1]), int(data_node[h][2])
        m.addConstr(z_ij[h] == kl[i, j] + kl[j, i], name="")

    lin = gp.LinExpr()
    for i in range(1, N):
        lin = 0
        for j in range(N):
            if node_con[i][j] == 1:
                lin += kl[i, j]
        m.addConstr(lin == 1, name="")

    for j in range(N):
        m.addConstr((0 == kl[0, j]), name="")

    for h in range(L):
        m.addConstr(z0_ij[h] - z_ij[h] <= rr_ij[h], name="")
        m.addConstr(z_ij[h] - z0_ij[h] <= rr_ij[h], name="")
    m.addConstr(gp.quicksum(rr_ij[h] for h in range(L)) <= s_num[t_start], name="")

    # 添加约束
    for t1 in range(T1):
        for j in range(N):
            Pjk = gp.LinExpr()
            Qjk = gp.LinExpr()
            Pij = gp.LinExpr()
            Qij = gp.LinExpr()
            for k in range(L):
                if from_node[j][k] == 1:
                    Pjk += P_ij[t1, k]
                    Qjk += Q_ij[t1, k]
            for i in range(L):
                if to_node[j][i] == 1:
                    Pij += P_ij[t1, i] - r_ij[i] * L_ij[t1, i]
                    Qij += Q_ij[t1, i] - x_ij[i] * L_ij[t1, i]
            m.addConstr(P_g[t1, j] - data_load[t1+t_start][j][0] + Pij - Pjk == 0, name="")  #加上t_start
            m.addConstr(Q_g[t1, j] - data_load[t1+t_start][j][1] + Qij - Qjk == 0, name="")

    # for t1 in range(T1):
    #     for h in range(L):
    #         i, j = int(data_node[h][1]), int(data_node[h][2])
    #         m.addConstr(U[t1, i] - U[t1, j] <=
    #                     (1 - z_ij[h]) * M + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
    #                     - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
    #         m.addConstr(U[t1, i] - U[t1, j] >=
    #                     -(1 - z_ij[h]) * M + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
    #                     - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            m.addConstr(U[t1, i] - U[t1, j] ==
                        b_ij[t1, h] + 2 * (r_ij[h] * P_ij[t1, h] + x_ij[h] * Q_ij[t1, h])
                        - (r_ij[h] ** 2 + x_ij[h] ** 2) * L_ij[t1, h], name="")
            m.addConstr(b_ij[t1, h] <= M * (1 - z_ij[h]), name="")
            m.addConstr(b_ij[t1, h] >= -M * (1 - z_ij[h]), name="")

    for t1 in range(T1):
        for h in range(L):
            m.addConstr(Sij_max ** 2 * z_ij[h] >= P_ij[t1, h] ** 2 + Q_ij[t1, h] ** 2, name="")

    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            m.addConstr(4 * (P_ij[t1, h] ** 2 + Q_ij[t1, h] ** 2) + (L_ij[t1, h] - U[t1, i]) ** 2
                        <= (L_ij[t1, h] + U[t1, i]) ** 2, name="")

    for t1 in range(T1):
        m.addConstr(U[t1, 0] == 1, name="")

    cos_x = 0.9
    sin_x = np.sqrt(1 - cos_x ** 2)
    tan_x = sin_x / cos_x
    for t1 in range(T1):
        for i in range(N):
            if i in Sg:
                m.addConstr(Q_g[t1, i] == tan_x * P_g[t1, i], name="")

    for t1 in range(T1):
        for i in range(N): #加上+t_start
            m.addConstr(P_g[t1, i] <= pg_max[t1+t_start][i], name="")
            m.addConstr(P_g[t1, i] >= pg_min[t1+t_start][i], name="")
            m.addConstr(Q_g[t1, i] <= qg_max[t1+t_start][i], name="")
            m.addConstr(Q_g[t1, i] >= qg_min[t1+t_start][i], name="")

        for i in range(1, N):
            m.addConstr(U[t1, i] <= U_max[i], name="")
            m.addConstr(U[t1, i] >= U_min[i], name="")

    # 模型更新
    m.update()
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 0.001  # 设置目标间隙为 1%
    # m.Params.Threads = 32
    m.optimize()

    U_value = np.zeros((T1, N))
    zij = np.zeros(L)
    gq = np.zeros((T1, N))
    loss_net = 0
    for h in range(L):
        zij[h] = z_ij[h].x
    for t1 in range(T1):
        for i in range(N):
            U_value[t1][i] = np.sqrt(U[t1, i].x)
            gq[t1][i] = P_g[t1, i].x
        loss_net += loss[t1].x
    u_Max = np.zeros((T1, L))
    U_Max = np.zeros(T1)
    for t1 in range(T1):
        for h in range(L):
            i, j = int(data_node[h][1]), int(data_node[h][2])
            if P_ij[t1, h].x > 0 and z_ij[h].x > 0.01:
                u_Max[t1][h] = np.sqrt(U[t1, i].x) - np.sqrt(U[t1, j].x)
            else:
                u_Max[t1][h] = np.sqrt(U[t1, j].x) - np.sqrt(U[t1, i].x)
    for t1 in range(T1):
        U_Max[t1] = np.max(u_Max[t1])
    # print(gq)
    return U_value, zij, obj.getValue(), loss_net, gq, U_Max, obj1.getValue()

T1 = 1
z0_ij = [1]*L
for h in range(N, L):
    z0_ij[h] = 0
obj_ds = []
loss_ds = []
uGap_ds = []
station_ds = []
z_ds = np.empty((0, L))
u_ds = np.empty((0, N))
# 记录开始时间
start_time1 = time.time()
for t1 in range(int(24/T1)):
    if t1*T1 >= 24:
        break
    u0_ds, z0_ds, loss0_ds, uGap0_ds, station0_ds = ds_solve(int(t1*T1), z0_ij)
    uGap_ds.append(np.array(uGap0_ds))
    z0_ij = z0_ds
    z0_ds = z0_ds.reshape(1, -1)
    u0_ds = u0_ds.reshape(1, -1)
    u_ds = np.concatenate((u_ds, u0_ds), axis=0)
    z_ds = np.concatenate((z_ds, z0_ds), axis=0)
    loss_ds.append(np.array(loss0_ds))
    station_ds.append(station0_ds)
    # print(f"===== {t1}  ======")
# 记录结束时间
end_time1 = time.time()
# 计算程序运行时间
execution_time1 = end_time1 - start_time1

p = 0.1
z0_ij = [1]*L
for h in range(N, L):
    z0_ij[h] = 0
obj_dro = []
loss_dro = []
uGap_dro = []
station_dro = []
z_dro = np.empty((0, L))
pg_dro = np.empty((0, N))
u_dro = np.empty((0, N))
# 记录开始时间
start_time2 = time.time()
for t1 in range(int(24/T1)):
    if t1*T1 >= 24:
        break
    u0_dro, z0_dro, obj0_dro, loss0_dro, pg0_dro, uGap0_dro, station0_dro = dro_solve(p, int(t1*T1), z0_ij)
    uGap_dro.append(np.array(uGap0_dro))
    z0_ij = z0_dro
    z0_dro = z0_dro.reshape(1, -1)
    u0_dro = u0_dro.reshape(1, -1)
    pg0_dro = pg0_dro.reshape(1, -1)
    u_dro = np.concatenate((u_dro, u0_dro), axis=0)
    z_dro = np.concatenate((z_dro, z0_dro), axis=0)
    pg_dro = np.concatenate((pg_dro, pg0_dro), axis=0)
    loss_dro.append(np.array(loss0_dro))
    obj_dro.append(np.array(obj0_dro))
    station_dro.append(np.array(station0_dro))
    # print(f"===== {t1}  ======")
# 记录结束时间
end_time2 = time.time()
# 计算程序运行时间
execution_time2 = end_time2 - start_time2

z0_ij = [1]*L
for h in range(N, L):
    z0_ij[h] = 0
obj_ro = []
loss_ro = []
uGap_ro = []
station_ro = []
z_ro = np.empty((0, L))
u_ro = np.empty((0, N))
# 记录开始时间
start_time3 = time.time()
for t1 in range(int(24/T1)):
    if t1*T1 >= 24:
        break
    u0_ro, z0_ro, loss0_ro, uGap0_ro, station0_ro = ro_solve(int(t1*T1), z0_ij)
    uGap_ro.append(np.array(uGap0_ro))
    z0_ij = z0_ro
    z0_ro = z0_ro.reshape(1, -1)
    u0_ro = u0_ro.reshape(1, -1)
    u_ro = np.concatenate((u_ro, u0_ro), axis=0)
    z_ro = np.concatenate((z_ro, z0_ro), axis=0)
    loss_ro.append(np.array(loss0_ro))
    station_ro.append(np.array(station0_ro))
    # # print(f"===== {t1}  ======")
# 记录结束时间
end_time3 = time.time()
# 计算程序运行时间
execution_time3 = end_time3 - start_time3
i = 1
for t1 in range(int(24 / T1)):
    print(i, end=": ")
    i += 1
    for h in range(L):
        if z_ds[t1][h] <= 0.01:
            print(h+1, end=",")
    print(" ")
print("   ====   ==== ")
i = 1
for t1 in range(int(24 / T1)):
    print(i, end=": ")
    i += 1
    for h in range(L):
        if z_dro[t1][h] <= 0.01:
            print(h+1, end=",")
    print(" ")
print("   ====   ==== ")
i = 1
for t1 in range(int(24 / T1)):
    print(i, end=": ")
    i += 1
    for h in range(L):
        if z_ro[t1][h] <= 0.01:
            print(h+1, end=",")
    print(" ")

end_time = time.time()

# 计算程序运行时间
execution_time = end_time - start_time
print("程序运行时间：", execution_time, "秒")
print("随机程序运行时间：", execution_time1, "秒")
print("分布鲁邦程序运行时间：", execution_time2, "秒")
print("鲁邦程序运行时间：", execution_time3, "秒")
print(f"ds:{sum(loss_ds)*(10**4)} station:{sum(station_ds)*(10**4)}")
print(f"dro:{sum(loss_dro)*(10**4)} station:{sum(station_dro)*(10**4)}  obj:{sum(obj_dro)*(10**4)}")
print(f"ro:{sum(loss_ro)*(10**4)} station:{sum(station_ro)*(10**4)}")
print(f"Sg:{Sg}")
print(f"K={w_no}")# for t1 in range(24):
# for t1 in range(24):
#     print(f"t={t1+1}", np.min(u_dro[t1])-np.min(u_ds[t1]), np.min(u_dro[t1])-np.min(u_ro[t1]))
#     print(uGap_dro[t1]-uGap_ds[t1], uGap_dro[t1]-uGap_ro[t1])

# np.savetxt(fileURL+'ds33.txt', z_ds, delimiter=' ', fmt='%d')
# np.savetxt(fileURL+'dro33.txt', z_dro, delimiter=' ', fmt='%d')
# print(pg_dro)
# pg_dro = pg_dro*10000
# np.savetxt(fileURL+f'pg_dro{w_no}.txt', pg_dro, delimiter=' ', fmt='%0.4f')
# p = int((1-p)*100)
# np.savetxt(fileURL+f'u_dro{w_no}_{p}.txt', u_dro, delimiter=' ', fmt='%0.16f')
# np.savetxt(fileURL+f'u_ds{w_no}_{p}.txt', u_ds, delimiter=' ', fmt='%0.16f')
# np.savetxt(fileURL+f'u_ro{w_no}_{p}.txt', u_ro, delimiter=' ', fmt='%0.16f')

for t1 in range(24):
    x = np.arange(1, 34)

    # 创建一个绘图区域
    fig, ax = plt.subplots()

    # u_ds[t1] = sorted(u_ds[t1], reverse=True)
    # u_dro[t1] = sorted(u_dro[t1], reverse=True)
    # u_ro[t1] = sorted(u_ro[t1], reverse=True)

    # 绘制折线图
    ax.plot(x, u_ds[t1], color='b', linestyle='-', label='DS-DNR')  # 实线
    ax.plot(x, u_ro[t1], color='g', linestyle=':', label='RO-DNR')  # 点线
    ax.plot(x, u_dro[t1], color='r', linestyle='--', label='DRO-DNR')  # 虚线

    # ax.plot(x, u_ds[t1], color='b', label='DS-DNR')
    # ax.plot(x, u_ro[t1], color='g', label='RO-DNR')
    # ax.plot(x, u_dro[t1], color='r', label='DRO-DNR')

    # 绘制柱状图
    # bar_width = 0.25  # 每个柱的宽度
    # opacity = 0.8
    # # 定义每个数组的柱的位置
    # index1 = x - bar_width
    # index2 = x
    # index3 = x + bar_width
    # # 绘制每个数组的柱状图
    # rects1 = ax.bar(index1, u_ds[t1], bar_width, alpha=opacity, color='b', label='DS-DNR')
    # rects3 = ax.bar(index3, u_ro[t1], bar_width, alpha=opacity, color='g', label='RO-DNR')
    # rects2 = ax.bar(index2, u_dro[t1], bar_width, alpha=opacity, color='r', label='DRO-DNR')


    # 添加图例和标签
    # 设置横坐标每隔5个显示一个标签
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) if i % 5 == 0 else '' for i in x])
    ax.set_xlabel(f'Buses (Period {t1 + 1})')
    ax.set_ylabel('Voltage Magnitude (p.u.)')
    ax.legend()

    # 设置纵坐标范围
    y = int(np.min([np.min(u_ds[t1]), np.min(u_dro[t1]), np.min(u_ro[t1])])*100)/100
    ax.set_ylim(y, 1.0)

    # 显示图形
    plt.tight_layout()

    # 如果需要保存图形，可以取消下面两行的注释
    # plt.savefig(f"/Users/yinjinjiao/Documents/project/DNR/picture/IEEE33/{t1}.svg", format='svg')

    # 显示图形
    # plt.show()

    # 关闭图形以释放内存
    plt.close(fig)
    # 每个请求之间等待1秒钟
    # time.sleep(0.5)
