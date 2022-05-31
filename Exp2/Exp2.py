#!/usr/bin/env python 　
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/5/30 21:27
# @Author  : Yang Haoyuan
# @Email   : 2723701951@qq.com
# @File    : Exp2.py
# @Software: PyCharm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Exp2')
parser.add_argument('--Shop', type=str, default="1019", choices=["1019", "1020", "1021"])
parser.add_argument('--Good', type=str, default="10010油", choices=["10010油", "10020面制品", "10030米和粉", "10088粮油类赠品"])

parser.set_defaults(augment=True)
args = parser.parse_args()
print(args)


# 读取1019,1020,1021三个商店的数据
def getData():
    data = pd.read_csv("data.csv")
    data_19 = data[:7693]
    data_20 = data[7693:17589]
    data_21 = data[17589:]
    return data_19, data_20, data_21


# 构建数据立方体，数据结构采用DataFrame
def make_cuboid(data):
    arr = [[ 0.0, 0.0, 0.0, 0.0],
           [ 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0],
           [ 0.0, 0.0, 0.0, 0.0],
           [ 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0]]
    dataFrame = pd.DataFrame(arr, columns=["10010油", "10020面制品", "10030米和粉", "10088粮油类赠品"],
                             index=["13", "14", "15", "16", "17", "18", "19"])
    # 按日期进行筛选，把各日期的数据放入list中
    t = [data.loc[data["Date"] == 20030413, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030414, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030415, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030416, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030417, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030418, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030419, ["Date", "GoodID", "Num", "Price"]]
         ]
    idx = 13

    for df in t:



        # 按照商品类别，将各类商品各日期销售总额计算出来并保存
        _df = df[df["GoodID"] >= 1001000]
        _df = _df[_df["GoodID"] <= 1001099]
        _sum = 0

        for index, row in _df.iterrows():
            _sum = _sum + row["Num"] * row["Price"]
        dataFrame.loc[(str(idx), "10010油")] = _sum
        _sum = 0

        _df = df[df["GoodID"] >= 1002000]
        _df = _df[_df["GoodID"] <= 1002099]
        for index, row in _df.iterrows():
            _sum = _sum + row["Num"] * row["Price"]
        dataFrame.loc[(str(idx), "10020面制品")] = _sum
        _sum = 0

        _df = df[df["GoodID"] >= 1003000]
        _df = _df[_df["GoodID"] <= 1003099]
        for index, row in _df.iterrows():
            _sum = _sum + row["Num"] * row["Price"]
        dataFrame.loc[(str(idx), "10030米和粉")] = _sum
        _sum = 0

        _df = df[df["GoodID"] >= 1008800]
        _df = _df[_df["GoodID"] <= 1008899]
        for index, row in _df.iterrows():
            _sum = _sum + row["Num"] * row["Price"]
        dataFrame.loc[(str(idx), "10088粮油类赠品")] = _sum
        _sum = 0

        idx = idx + 1

    return dataFrame


if __name__ == "__main__":
    data_1019, data_1020, data_1021 = getData()

    # 各数据立方体按照4为小数保存到txt文件
    df_1019 = make_cuboid(data_1019)
    df_1019.applymap('{:.4f}'.format).to_csv("1019.txt", index=False)

    df_1020 = make_cuboid(data_1020)
    df_1020.applymap('{:.4f}'.format).to_csv("1020.txt", index=False)

    df_1021 = make_cuboid(data_1021)
    df_1021.applymap('{:.4f}'.format).to_csv("1021.txt", index=False)

    # 三维数据立方体保存到txt文件中
    data = pd.concat([df_1019, df_1020, df_1021], keys=["1019", "1020", "1021"], names=["Shop", "Date"])
    data.to_csv("data_cubiod.csv")

    # "1020商店10010油类商品13日总的销售额
    print("1020商店10010油类商品13日总的销售额", format(data.loc[("1020", "13"), "10010油"], '.2f'))

    # 1020商店10030米和粉总的销售额
    df = data.loc["1020"]
    print("1020商店10030米和粉总的销售额", format(df["10030米和粉"].sum(), '.2f'))

    # 指定商店指定货物的销售总额
    df = data.loc[args.Shop]
    print(args.Shop + "商店" + args.Good + "的销售额", format(df[args.Good].sum(), '.2f'))
