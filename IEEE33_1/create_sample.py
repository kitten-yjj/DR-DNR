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
    # while np.any(samples < min_value) or np.any(samples > max_value):
    #     invalid_samples_mask = (samples < min_value) | (samples > max_value)
    #     samples[invalid_samples_mask] = np.random.normal(mean, std_dev, invalid_samples_mask.sum())
    return samples

weght = [0.2, 0.4, 0.3, 0.18, 0.17, 0.2, 0.22, 0.2, 0.19, 0.17, 0.2, 0.22, 0.38, 0.58, 0.79, 0.81, 0.72
    , 0.78, 0.71, 0.73, 0.6, 0.4, 0.28, 0.29]

# 获取分布式发电得历史数据
Sg = {8,25,30}
N = 33
PG_samples = np.zeros((24, N, 50))
pg_max = np.zeros((24, N))
pg_min = np.zeros((24, N))
DG = [f"{fileURL}DG{i}-50.csv" for i in range(N)]
for t in range(24):
    for i in range(N):
        if i in Sg:
            PG_samples[t][i] = pd.read_csv(DG[i]).iloc[t]
        pg_max[t][i] = np.max(PG_samples[t][i])
        pg_min[t][i] = np.min(PG_samples[t][i])
# 设定正态分布的均值和标准差
mean_value = 200  # 均值
std_deviation = 3  # 标准差
num_samples = 50
sg = {24}
samples = np.zeros((24, num_samples))
for i in sg:
    for t in range(24):
        samples[t] = monte_carlo_sampling(mean_value * weght[t], std_deviation, num_samples, pg_min[t][i], pg_max[t][i])
    # 将采样点保存到CSV文件
    np.savetxt(f'{fileURL}DG{i}-{num_samples}.csv', samples, delimiter=',', fmt='%.2f',
               header=",".join([f"n{i + 1}" for i in range(num_samples)]), comments='')



