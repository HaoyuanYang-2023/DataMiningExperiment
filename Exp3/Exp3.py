#!/usr/bin/env python 　
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/30 21:29
# @Author  : Yang Haoyuan
# @Email   : 2723701951@qq.com
# @File    : Exp3.pyDa
# @Software: PyCharm
from numpy import *
import pandas as pd
from apyori import apriori
import argparse

parser = argparse.ArgumentParser(description='Exp3')
parser.add_argument('--Dataset', type=str, choices=["1019", "1020", "1021"])
parser.add_argument('--minSup', type=float)

parser.set_defaults(augment=True)
args = parser.parse_args()
print(args)


# 读取数据集
def loadDataSet():
    data = pd.read_csv("data.csv")
    data_19 = data[:7693]
    data_20 = data[7693:17589]
    data_21 = data[17589:]

    # 按照流水号分组,把同一个篮子里的商品分到一组
    data_df_list = [data_19.groupby(['SerNo'])['GoodID'], data_20.groupby(['SerNo'])['GoodID'],
                    data_21.groupby(['SerNo'])['GoodID']]
    list_ = []
    list_re = []

    for df in data_df_list:
        df_ = pd.DataFrame(df)
        for i in range(df_.shape[0]):
            tmp = []
            # 将每个篮子里的商品编号添加到同一事务项
            l = df_.loc[i][1].values.tolist()
            for gid in l:
                tmp.append(str(gid)[:5])
            # 添加至项集
            list_.append(tmp)
        list_re.append(list_)

    return list_re[0], list_re[1], list_re[2]


# 创造1-项集C1
def createC1(dataSet):
    C1 = []
    # 遍历dataset中每一个事务中的每一个item，加入C1
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    # 将C1的每一项映射为冻结集合
    # 为什么使用冻结集合？
    # 因为在集合的关系中，有集合的中的元素是另一个集合的情况，但是普通集合（set）本身是可变的，
    # 那么它的实例就不能放在另一个集合中（set中的元素必须是不可变类型）。
    # 所以，frozenset提供了不可变的集合的功能，当集合不可变时，它就满足了作为集合中的元素的要求，就可以放在另一个集合中了。
    # python要求字典的键是可哈希的，set是可变的，不可哈希，frozenset是不可变的，可哈希
    return list(map(frozenset, C1))


# 计算支持度,筛选满足要求的k-项集成为频繁项集Lk
def scanD(dataSet, Ck, minSupport):
    ssCnt = {}
    # 遍历计算Ck中的每一项在Dataset中的出现次数
    for tid in dataSet:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # 数据集中总共的事务数量
    numItems = float(len(dataSet))
    retList = []
    supportData = {}
    # 计算Ck中每一项的支持度
    for key in ssCnt:
        support = ssCnt[key]
        if support >= minSupport:
            # 大于最小支持度加入结果集合
            retList.append(key)
            # 记录支持度数据
            supportData[key] = support
    return retList, supportData


# 从Lk频繁项集中产生Ck+1
def aprioriGen(Lk, k):
    lenLk = len(Lk)
    # 临时字典，存储
    temp = []
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 两两合并，执行了 lenLk！次
            L1 = Lk[i] | Lk[j]
            if len(L1) == k:
                if L1 not in temp:
                    # 如果合并后的子项元素有k个，满足要求
                    temp.append(L1)

    return temp


def apriori(dataSet, minSupport=0.5):
    # 构建C1
    C1 = createC1(dataSet)
    print("C1: ", C1)
    print('\n')

    D = list(map(set, dataSet))

    # 构建L1,获取置信度
    L1, supportData = scanD(D, C1, minSupport)
    # print("Support Data for C1: ", supportData)
    # print('\n')
    print("L1: ", L1)
    print('\n')

    Lk_List = [L1]
    k = 2
    while len(Lk_List[k - 2]) > 0:
        print()
        # 构建Ck
        Ck = aprioriGen(Lk_List[k - 2], k)
        # Ck为空，不能合并，迭代终止
        if not Ck:
            print("项集为空")
            print('\n')
            break
        print("C" + str(k) + ": ", Ck)
        print('\n')
        # 构建LK，获取Ck各项的支持度
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        # 新的Lk加入Lk_List
        Lk_List.append(Lk)
        # print("Support Data for C" + str(k) + ": ", supK)
        # print('\n')

        # 频繁项集为空，终止迭代
        if not Lk:
            print("C" + str(k) + "频繁项集为空")
            print('\n')
            break
        print("L" + str(k) + ": ", Lk)
        print('\n')
        k += 1

    return Lk_List, supportData


if __name__ == "__main__":
    dataSet19, dataSet20, dataSet21 = loadDataSet()
    # 根据参数对不同数据集进行挖掘
    if args.Dataset == "1019":
        L, suppData = apriori(dataSet19, args.minSup)
        print(L)
        # print(suppData)
    if args.Dataset == "1020":
        L, suppData = apriori(dataSet20, args.minSup)
        # print(L)
        print(suppData)
    if args.Dataset == "1021":
        L, suppData = apriori(dataSet21, args.minSup)
        print(L)
        # print(suppData)



