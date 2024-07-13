import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
fileURL = "/Users/yinjinjiao/Documents/project/DNR/data/IEEE33/"

def monte_carlo_sampling(mean, std_dev, num_samples, min_value, max_value):
    """
    使用蒙特卡洛方法从正态分布中抽取随机样本，并将样本限制在指定的区间内
    :param mean: 正态分布的均值
    :param std_dev: 正态分布的标准差
    :param num_samples: 要抽取的样本数量
    :param min_value: 样本的最小值
    :param max_value: 样本的最大值
    :return: 包含随机样本的NumPy数组
    """
    samples = np.random.normal(mean, std_dev, num_samples)
    while np.any(samples < min_value) or np.any(samples > max_value):
        invalid_samples_mask = (samples < min_value) | (samples > max_value)
        samples[invalid_samples_mask] = np.random.normal(mean, std_dev, invalid_samples_mask.sum())
    return samples

weght = [0.2, 0.4, 0.3, 0.18, 0.17, 0.2, 0.22, 0.2, 0.19, 0.17, 0.2, 0.22, 0.38, 0.58, 0.79, 0.81, 0.72
    , 0.78, 0.71, 0.73, 0.6, 0.4, 0.28, 0.29]
# 设定正态分布的均值和标准差

# 抽取100个样本
num_samples = 50
N = 33
T = 24
Sg = {8, 25, 30}
Tag = {50, 100, 200, 500, 1000}
def relity():
    mean_value = 200  # 均值
    std_deviation = 2  # 标准差
    reliability = np.zeros((5, 3))
    pg_av = np.zeros((T, N))
    T_itr = -1
    PG_samples = np.zeros((T, N, 100))
    DG = [f"{fileURL}DG{i}-100.csv" for i in range(N)]
    for t in range(24):
        for i in range(N):
            if i in Sg:
                PG_samples[t][i] = pd.read_csv(DG[i]).iloc[t]
                pg_av[t][i] = np.average(PG_samples[t][i])
    samples = np.zeros((N, T, num_samples))
    for t in range(24):
        samples[25][t] = monte_carlo_sampling(mean_value * weght[t], std_deviation, num_samples,
                                              np.min(PG_samples[t][25]), np.max(PG_samples[t][25]))
        samples[30][t] = monte_carlo_sampling(mean_value * weght[t], std_deviation, num_samples,
                                              np.min(PG_samples[t][30]), np.max(PG_samples[t][30]))
    mean_value = 100  # 均值
    for t in range(24):
        samples[8][t] = monte_carlo_sampling(mean_value * weght[t], std_deviation, num_samples,
                                             np.min(PG_samples[t][8]), np.max(PG_samples[t][8]), )

    for tag in Tag:
        T_itr += 1
        PG_samples = np.zeros((24, N))
        PG_samples1 = np.zeros((24, N))
        DG = f"{fileURL}pg_dro{int(tag)}.txt"
        # DG1 = f"{fileURL}pg_ep{int(tag)}.txt"
        PG_samples = np.genfromtxt(DG, delimiter=' ', filling_values=0)
        # PG_samples1 = np.genfromtxt(DG1, delimiter=' ', filling_values=0)
        y = 0
        for i in Sg:
            for t1 in range(24):
                for n in range(50):
                    y += 1
                    testX = samples[i][t1][n]
                    if testX > PG_samples[t1][i]:
                        # print(testX, PG_samples[t1][i])
                        reliability[T_itr][1] += 1
                    # if testX > PG_samples1[t1][i]:
                    #     reliability[T_itr][2] += 1
                    if testX > pg_av[t1][i]:
                        reliability[T_itr][0] += 1
    reliability = reliability / 3600
    for i in range(5):
        reliability[i][2] = 1
    return reliability

count = 30
reliability1 = np.zeros((5, 3))
reli = np.zeros((30, 5, 3))
for i in range(count):
    reli[i] = relity()
for i in range(5):
    for j in range(3):
        sum = 0
        for t in range(count):
            sum = sum + reli[t][i][j]
        sum = sum/count
        reliability1[i][j] = sum

np.savetxt(fileURL+f'reliability1.txt', reliability1, delimiter=' ', fmt='%0.4f')

print(reliability1)