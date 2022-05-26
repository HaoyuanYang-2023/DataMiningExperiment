import numpy as np
import pandas as pd


def getData():
    data = pd.read_csv("data.csv")
    data_19 = data[:7693]
    data_20 = data[7693:17589]
    data_21 = data[17589:]
    return data_19, data_20, data_21


def make_cuboid(data):
    arr = [[20030413, 0.0, 0.0, 0.0, 0.0],
           [20030414, 0.0, 0.0, 0.0, 0.0],
           [20030415, 0.0, 0.0, 0.0, 0.0],
           [20030416, 0.0, 0.0, 0.0, 0.0],
           [20030417, 0.0, 0.0, 0.0, 0.0],
           [20030418, 0.0, 0.0, 0.0, 0.0],
           [20030419, 0.0, 0.0, 0.0, 0.0]]
    dataFrame = pd.DataFrame(arr, columns=["Date", "10010油", "10020面制品", "10030米和粉", "10088粮油类赠品"])
    # c print(dataFrame)
    t = [data.loc[data["Date"] == 20030413, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030414, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030415, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030416, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030417, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030418, ["Date", "GoodID", "Num", "Price"]],
         data.loc[data["Date"] == 20030419, ["Date", "GoodID", "Num", "Price"]]
         ]
    idx = 0
    for df in t:

        _df = df[df["GoodID"] >= 1001000]
        _df = _df[_df["GoodID"] <= 1001099]
        _sum = 0

        for index, row in _df.iterrows():
            _sum = _sum + row["Num"] * row["Price"]
        dataFrame.loc[idx, "10010油"] = _sum
        _sum = 0

        _df = df[df["GoodID"] >= 1002000]
        _df = _df[_df["GoodID"] <= 1002099]
        for index, row in _df.iterrows():
            _sum = _sum + row["Num"] * row["Price"]
        dataFrame.loc[idx, "10020面制品"] = _sum
        _sum = 0

        _df = df[df["GoodID"] >= 1003000]
        _df = _df[_df["GoodID"] <= 1003099]
        for index, row in _df.iterrows():
            _sum = _sum + row["Num"] * row["Price"]
        dataFrame.loc[idx, "10030米和粉"] = _sum
        _sum = 0

        _df = df[df["GoodID"] >= 1008800]
        _df = _df[_df["GoodID"] <= 1008899]
        for index, row in _df.iterrows():
            _sum = _sum + row["Num"] * row["Price"]
        dataFrame.loc[idx, "10088粮油类赠品"] = _sum
        _sum = 0

        idx = idx + 1

    print(dataFrame)
    return dataFrame


if __name__ == "__main__":
    data_1019, data_1020, data_1021 = getData()

    df_1019 = make_cuboid(data_1019)
    df_1019.applymap('{:.4f}'.format).to_csv("1019.txt", index=False)

    df_1020 = make_cuboid(data_1019)
    df_1020.applymap('{:.4f}'.format).to_csv("1020.txt", index=False)

    df_1021 = make_cuboid(data_1019)
    df_1021.applymap('{:.4f}'.format).to_csv("1021.txt", index=False)

    data = pd.DataFrame([[data_1019, data_1020, data_1021]], columns=["1019", "1020", "1021"])
    print(data)
