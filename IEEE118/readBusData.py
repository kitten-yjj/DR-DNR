import numpy as np

fileURL = "/home/yinjinjiao/pycharm_project/DNR/data/IEEE118/"

# try:
#     # 读取txt文件，排除最后一列数据
data1 = np.loadtxt(fileURL + 'bus118.txt', usecols=range(12))
print("数据的形状：", data1.shape)
print("前几行数据：")
print(data1)
# except IOError:
#     print("文件不存在或无法打开。")
# except ValueError:
#     print("文件包含无效数据，无法解析。")


# try:
#     # 读取txt文件，排除最后一列数据
data2 = np.loadtxt(fileURL + 'branch118.txt', usecols=range(12))
print("数据的形状：", data2.shape)
print("前几行数据：")
print(data2)
# except IOError:
#     print("文件不存在或无法打开。")
# except ValueError:
#     print("文件包含无效数据，无法解析。")

data = np.zeros((132, 7))
for i in range(132):
    data[i][0] = int(i + 1)
    data[i][1] = int(data2[i][0])
    data[i][2] = int(data2[i][1])
    data[i][3] = data2[i][2]
    data[i][4] = data2[i][3]
    if i < 118:
        data[i][5] = data1[i][2]
        data[i][6] = data1[i][3]
    else:
        data[i][5] = -1
        data[i][6] = -1

print(data)

np.savetxt('/home/yinjinjiao/pycharm_project/DNR/data/IEEE118/IEEE118.csv', data, delimiter=',', fmt='%.4f', header=",".join([f"n{i+1}" for i in range(7)]), comments='')
