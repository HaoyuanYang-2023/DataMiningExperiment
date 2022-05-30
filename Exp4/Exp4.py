#!/usr/bin/env python 　
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/30 21:26
# @Author  : Yang Haoyuan
# @Email   : 2723701951@qq.com
# @File    : Exp4.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import argparse

parser = argparse.ArgumentParser(description='Exp4')
parser.add_argument('--mode', type=str, choices=["KFold", "train", "test"])
parser.add_argument('--k', type=int, default=7)
parser.add_argument('--AGE', type=str, choices=["youth", "middle_aged", "senior"])
parser.add_argument('--INCOME', type=str, choices=["high", "medium", "low"])
parser.add_argument('--STUDENT', type=str, choices=["yes", "no"])
parser.add_argument('--CREDIT', type=str, choices=["excellent", "fair"], default="fair")
parser.set_defaults(augment=True)
args = parser.parse_args()
print(args)


# 载入数据集
def loadDataset(filename):
    dataSet = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            p_tmp = [str(i) for i in lines.split(sep="\t")]
            p_tmp[len(p_tmp) - 1] = p_tmp[len(p_tmp) - 1].strip("\n")
            dataSet.append(p_tmp)

    return pd.DataFrame(dataSet, columns=["AGE", "INCOME", "STUDENT", "CREDIT", "BUY"])


# 计算总样本数和各类数量
def count_total(data):
    count = {}
    group_df = data.groupby(["BUY"])

    count["yes"] = group_df.size()["yes"]
    count["no"] = group_df.size()["no"]

    total = count["yes"] + count["no"]
    return count, total


# 计算各类概率
def cal_base_rates(categories, total):
    rates = {}
    for label in categories:
        priori_prob = categories[label] / total
        rates[label] = priori_prob
    return rates


# 计算各特征条件概率
def f_prob(data, count):
    likelihood = {'yes': {}, 'no': {}}

    # 根据AGE(youth, middle_aged, senior)和BUY(yes, no)统计概率
    df_group = data.groupby(['AGE', 'BUY'])
    try:
        c = df_group.size()["youth", "yes"]
    except:
        c = 0
    likelihood['yes']['youth'] = c / count['yes']

    try:
        c = df_group.size()["youth", "no"]
    except:
        c = 0
    likelihood['no']['youth'] = c / count['no']

    try:
        c = df_group.size()["middle_aged", "yes"]
    except:
        c = 0
    likelihood['yes']['middle_aged'] = c / count['yes']

    try:
        c = df_group.size()["middle_aged", "no"]
    except:
        c = 0
    likelihood['no']['middle_aged'] = c / count['no']

    try:
        c = df_group.size()["senior", "yes"]
    except:
        c = 0
    likelihood['yes']['senior'] = c / count['yes']

    try:
        c = df_group.size()["senior", "no"]
    except:
        c = 0
    likelihood['no']['senior'] = c / count['no']

    # 根据INCOME(high, medium, low)和BUY(yes, no)统计概率
    df_group = data.groupby(['INCOME', 'BUY'])
    try:
        c = df_group.size()["high", "yes"]
    except:
        c = 0
    likelihood['yes']['high'] = c / count['yes']

    try:
        c = df_group.size()["high", "no"]
    except:
        c = 0
    likelihood['no']['high'] = c / count['no']

    try:
        c = df_group.size()["medium", "yes"]
    except:
        c = 0
    likelihood['yes']['medium'] = c / count['yes']

    try:
        c = df_group.size()["medium", "no"]
    except:
        c = 0
    likelihood['no']['medium'] = c / count['no']

    try:
        c = df_group.size()["low", "yes"]
    except:
        c = 0
    likelihood['yes']['low'] = c / count['yes']

    try:
        c = df_group.size()["low", "no"]
    except:
        c = 0
    likelihood['no']['low'] = c / count['no']

    # 根据STUDENT(yes, no)和BUY(yes, no)统计概率
    df_group = data.groupby(['STUDENT', 'BUY'])
    try:
        c = df_group.size()["yes", "yes"]
    except:
        c = 0
    likelihood['yes']['yes'] = c / count['yes']

    try:
        c = df_group.size()["yes", "no"]
    except:
        c = 0
    likelihood['no']['yes'] = c / count['no']

    try:
        c = df_group.size()["no", "yes"]
    except:
        c = 0
    likelihood['yes']['no'] = c / count['yes']

    try:
        c = df_group.size()["no", "no"]
    except:
        c = 0
    likelihood['no']['no'] = c / count['no']

    # 根据CREDIT(excellent, fair)和BUY(yes, no)统计概率
    df_group = data.groupby(['CREDIT', 'BUY'])
    try:
        c = df_group.size()["excellent", "yes"]
    except:
        c = 0
    likelihood['yes']['excellent'] = c / count['yes']

    try:
        c = df_group.size()["excellent", "no"]
    except:
        c = 0
    likelihood['no']['excellent'] = c / count['no']

    try:
        c = df_group.size()["fair", "yes"]
    except:
        c = 0
    likelihood['yes']['fair'] = c / count['yes']

    try:
        c = df_group.size()["fair", "no"]
    except:
        c = 0
    likelihood['no']['fair'] = c / count['no']

    return likelihood


# 训练
def train(train_data):
    # 获取各类数量和训练样本总数
    count, total = count_total(train_data)
    # 获取先验概率
    priori_prob = cal_base_rates(count, total)
    # 保存先验概率
    np.save("priori_prob.npy", priori_prob)
    # 获取各特征的条件概率
    feature_prob = f_prob(train_data, count)
    # 保存条件概率
    np.save("feature_prob.npy", feature_prob)
    print("训练完成")


# 分类器
def NaiveBayesClassifier(AGE=None, INCOME=None, STUDENT=None, CREDIT=None):
    res = {}

    # 根据特征计算各类的概率
    for label in ['yes', 'no']:
        prob = priori_prob[label]
        prob *= feature_prob[label][AGE] * feature_prob[label][INCOME] * feature_prob[label][STUDENT] \
                * feature_prob[label][CREDIT]
        res[label] = prob
    print("预测概率：", res)
    # 选择概率最高的类作为分类结果
    res = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return res[0][0]


# 测试
def test(test_data):
    correct = 0
    for idx, row in test_data.iterrows():
        prob = NaiveBayesClassifier(row["AGE"], row["INCOME"], row["STUDENT"], row["CREDIT"])
        if prob == row["BUY"]:
            correct = correct + 1
    return correct / test_data.shape[0]


# 启用k-折交叉验证
def KFoldEnabled():
    kf = KFold(n_splits=args.k)
    data_set = loadDataset("date.txt")
    corr = 0
    for train_idx, test_idx in kf.split(data_set):
        train_data = data_set.loc[train_idx]
        test_data = data_set.loc[test_idx]
        train(train_data)
        corr = corr + test(test_data)

    print("k折交叉验证正确率: ", corr / 7)


if __name__ == '__main__':
    if args.mode == "KFold":
        KFoldEnabled()
    if args.mode == "train":
        train_data = loadDataset("date.txt")
        train(train_data)
    if args.mode == "test":
        priori_prob = np.load('priori_prob.npy', allow_pickle=True).item()
        print("先验概率: ", priori_prob)
        feature_prob = np.load('feature_prob.npy', allow_pickle=True).item()
        print("特性分类概率: ", feature_prob)
        ret = NaiveBayesClassifier(args.AGE, args.INCOME, args.STUDENT, args.CREDIT)
        print("预测结果: ", ret)
