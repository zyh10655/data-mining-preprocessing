import numpy as np
import pandas as pd
import csv

# Acc as global
global Acc

# name of files
source_file = 'assessment.CSV'
handled_file = 'car.csv'

# load data
data_file = open(handled_file, 'w')


# turn string to numbers
def find_index(x, y):
    return [i for i in range(len(y)) if y[i] == x]


# define buying list
def handleBuying(inputs):
    Buying_list = ['vhigh', 'high', 'med', 'low']
    if inputs[0] in Buying_list:
        return find_index(inputs[0], Buying_list)[0]


# define maint list
def handleMaint(inputs):
    maint_list = ['vhigh', 'high', 'med', 'low']
    if inputs[1] in maint_list:
        return find_index(inputs[1], maint_list)[0]


# define Door list
def handleDoors(inputs):
    Door_list = ['2', '3', '4', '5more']
    if inputs[2] in Door_list:
        return find_index(inputs[2], Door_list)[0]


# define person list
def handlePersons(inputs):
    persons_list = ['2', '4', 'more']
    if inputs[3] in persons_list:
        return find_index(inputs[3], persons_list)[0]

#define lug_boot
def handleLB(inputs):
    lug_boot = ['small', 'med', 'big']
    if inputs[4] in lug_boot:
        return find_index(inputs[4], lug_boot)[0]

#define safety
def handleSafety(inputs):
    safety = ['low', 'med', 'high']
    if inputs[5] in safety:
        return find_index(inputs[5], safety)[0]

#define Acc
def handleAcc(inputs):
    Acc = ['unacc', 'acc', 'good', 'vgood']
    if inputs[6] in Acc:
        return find_index(inputs[6], Acc)[0]
    else:
        Acc.append(inputs[6])
        return find_index(inputs[6], Acc)[0]


# main
if __name__ == '__main__':
    #  loop read file data
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        csv_writer = csv.writer(data_file)
        count = 0
        for row in csv_reader:
            temp_line = np.array(row)
            temp_line[0] = handleBuying(row)
            temp_line[1] = handleMaint(row)
            temp_line[2] = handleDoors(row)
            temp_line[3] = handlePersons(row)
            temp_line[4] = handleLB(row)
            temp_line[5] = handleSafety(row)
            temp_line[6] = handleAcc(row)
            csv_writer.writerow(temp_line)
            count += 1

            # print the outline status
            print(count, 'status:', temp_line[0], temp_line[1], temp_line[2],
                  temp_line[3], temp_line[4], temp_line[5], temp_line[6])
        data_file.close()

# import os
# import csv
# import numpy as np
# import pandas as pd
# from sklearn import metrics
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn import neighbors
#
# #-----------------------------------------第一步 加载数据集-----------------------------------------
# fr= open("E:\\723 Assessment\\assessment.CSV")
# lines = fr.readlines()
# line_nums = len(lines)
# print(line_nums)
#
# #创建line_nums行 para_num列的矩阵
# x_mat = np.zeros((line_nums, 6))
# y_label = []
#
# #划分数据集
# for i in range(line_nums):
#     line = lines[i].strip()
#     item_mat = line.split(',')
#     x_mat[i, :] = item_mat[0:6]    #前41个特征
#     y_label.append(item_mat[-1])  #类标
# fr.close()
# print(x_mat.shape)
# print(len(y_label))
#
# #-----------------------------------------第二步 划分数据集-----------------------------------------
# y = []
# for n in y_label:
#     y.append(int(n))
# y =  np.array(y, dtype = int) #list转换数组
#
# #划分数据集 测试集30%
# train_data, test_data, train_target, test_target = train_test_split(x_mat, y, test_size=0.3, random_state=4)
# print (train_data.shape, train_target.shape)
# print (test_data.shape, test_target.shape)
#
#
# #-----------------------------------------第三步 KNN训练-----------------------------------------
# clf = neighbors.KNeighborsClassifier()
# clf.fit(train_data, train_target)
# print (clf)
# result = clf.predict(test_data)
# print (result)
# print (test_target)
#
#
# #-----------------------------------------第四步 评价算法-----------------------------------------
# print (sum(result==test_target)) #预测结果与真实结果比对
# print(metrics.classification_report(test_target, result))  #准确率 召回率 F值
#
#
# #----------------------------------------第五步 降维可视化---------------------------------------
# pca = PCA(n_components=2)
# newData = pca.fit_transform(test_data)
# plt.figure()
# plt.scatter(newData[:,0], newData[:,1], c=test_target, s=50)
# plt.show()

# import numpy as np
# import pandas as pd
# import csv
#
# """
# 功能：数据预处理 将KDD99数据集中字符型转换为数值型
# 原文：https://blog.csdn.net/asialee_bird/article/details/80491256
#
# 强烈推荐博友们阅读asialee_bird大神的文章及Github代码，非常厉害的一位博主。
# 修订：Eastmount 2019-11-22
# """
#
# # label_list为全局变量
# global safety
# def preHandel_data():
#     source_file = 'assessment.CSV'
#     handled_file = 'car.csv'
#     data_file = open(handled_file, 'w', newline='')
#     with open(source_file, 'r') as data_source:
#         csv_reader = csv.reader(data_source)
#         csv_writer = csv.writer(data_file)
#         count = 0  # 行数
#         for row in csv_reader:
#             temp_line = np.array(row)
#             temp_line[1] = handleBuying(row)  # 将源文件行中3种协议类型转换成数字标识
#             temp_line[2] = handleMaint(row)  # 将源文件行中70种网络服务类型转换成数字标识
#             temp_line[3] = handleDoors(row)  # 将源文件行中11种网络连接状态转换成数字标识
#             temp_line[4] = handlePersons(row)  # 将源文件行中23种攻击类型转换成数字标识
#             temp_line[5] = handleLB(row)
#             temp_line[6] = handleSafety(row)
#             temp_line[7] = handleAcc(row)
#             csv_writer.writerow(temp_line)
#             count += 1
#
#             # 输出每行数据中所修改后的状态
#             print(count, 'status:', temp_line[1], temp_line[2], temp_line[3],
#                   temp_line[4], temp_line[5], temp_line[6], temp_line[7])
#         data_file.close()
#
# # 文件名
#
#
# # 文件写入操作
#
#
# # 将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
# def find_index(x, y):
#     return [i for i in range(len(y)) if y[i] == x]
#
#
# # 定义将源文件行中3种协议类型转换成数字标识的函数
# def handleBuying(inputs):
#     Buying_list = ['v-high', 'high', 'med', 'low']
#     if inputs[0] in Buying_list:
#         return find_index(inputs[1], Buying_list)[0]
#
#
# # 定义将源文件行中70种网络服务类型转换成数字标识的函数
# def handleMaint(inputs):
#     maint_list = ['v-high', 'high', 'med', 'low']
#     if inputs[1] in maint_list:
#         return find_index(inputs[2], maint_list)[0]
#
#
# # 定义将源文件行中11种网络连接状态转换成数字标识的函数
# def handleDoors(inputs):
#     Door_list = ['2', '3', '4', '5-more']
#     if inputs[2] in Door_list:
#         return find_index(inputs[3], Door_list)[0]
#
#
# # 定义将源文件行中攻击类型转换成数字标识的函数(训练集中共出现了22个攻击类型，而剩下的17种只在测试集中出现)
# def handlePersons(inputs):
#     persons_list = ['2', '4', 'more']
#     # 在函数内部使用全局变量并修改它
#     if inputs[3] in persons_list:
#         return find_index(inputs[4], persons_list)[0]
#
#
# def handleLB(inputs):
#     lug_boot = ['small', 'med', 'big']
#     # 在函数内部使用全局变量并修改它
#     if inputs[4] in lug_boot:
#         return find_index(inputs[5], lug_boot)[0]
#
#
# def handleSafety(inputs):
#     safety = ['low', 'med', 'high']
#     # 在函数内部使用全局变量并修改它
#     if inputs[5] in safety:
#         return find_index(inputs[6], safety)[0]
#
#
# def handleAcc(inputs):
#     Acc = ['unacc', 'acc', 'good', 'vgood']
#     global Acc
#     if inputs[6] in Acc:
#         return find_index(inputs[6], Acc)[0]
#     else:
#         Acc.append(inputs[6])
#         return find_index(inputs[6], Acc)[0]
#
# if __name__=='__main__':
#     global Acc   #声明一个全局变量的列表并初始化为空
#     Acc=[]
#     preHandel_data()