import argparse
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='Exp1')
parser.add_argument('-files', nargs='+')

parser.set_defaults(augment=True)
args = parser.parse_args()
print(args)


# 从txt文件中读取数据，并做初步处理，数据以numpy array的格式返回
def load_data():
    files = args.files

    data = []
    sales = []
    count = 0

    for filename in files:
        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                new_p_tmp = []
                p_tmp = [str(i) for i in lines.split(sep="\t")]  # 将整行数据分割处理
                # p_tmp 为每一条记录的list，其中数据类型均为str

                p_tmp[3] = p_tmp[3].replace('/', '', 2).replace('年', '', 1).replace('月', '', 1).replace('日', '', 1)
                # p_tmp[3]为日期，对于日期空缺，根据流水单号将其填补
                # 变换日期格式，为200304xx格式
                # 空缺日期从流水号中补齐
                if p_tmp[3] == "###":
                    p_tmp[3] = p_tmp[0][:8]
                else:
                    p_tmp[3] = p_tmp[3][0:4] + '0' + p_tmp[3][4:len(p_tmp[3])]

                # 去除最后的换行符
                p_tmp[len(p_tmp) - 1] = p_tmp[len(p_tmp) - 1].strip("\n")
                # p_tmp[0]为流水号，由于所有record均是2003年4月的记录，所以我去掉200304以进行规约，减小后期分析数据所需空间
                # 同时去掉了'-'，是其能够转化为浮点型参与分析运算
                # 为每一条记录加一个序号，与其索引一致，方便后期运算
                p_tmp[0] = p_tmp[0].replace('-', '')
                p_tmp.insert(0, count)
                p_tmp[0] = count
                count = count + 1
                # 将所有数据转换为浮点型，以便后期运算
                for i in p_tmp:
                    i = float(i)
                    new_p_tmp.append(i)
                # 将小于零的购买记录和总价转正
                if new_p_tmp[7] < 0:
                    new_p_tmp[7] = -new_p_tmp[7]
                    new_p_tmp[9] = -new_p_tmp[9]

                # 向sales中添加数据
                sales.append([new_p_tmp[0], new_p_tmp[6], new_p_tmp[7]])
                data.append(new_p_tmp)  # 添加新读取的数据
                # 首条数据商品序号为1
                if len(data) == 1:
                    data[0][5] = 1
                # 流水号与之前的不同的商品序号为1
                if data[len(data) - 1][1] != data[len(data) - 2][1]:
                    data[len(data) - 1][5] = 1
                # 每一个购物篮商品序号从1逐次递增
                if (data[len(data) - 1][5] - data[len(data) - 2][5]) > 1 and (
                        data[len(data) - 1][1] == data[len(data) - 2][1]):
                    d = data[len(data) - 1][5]
                    d = d - 1
                    data[len(data) - 1][5] = d
        print(len(data))

    return np.array(data), np.array(sales)


# 展示聚类后的情况
def show(label_pred, X, count):
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    for i in range(len(label_pred)):
        if label_pred[i] == 0:
            x0.append(X[i])
        if label_pred[i] == 1:
            x1.append(X[i])
        if label_pred[i] == 2:
            x2.append(X[i])
        if label_pred[i] == 3:
            x3.append(X[i])
    # print(count[0])
    plt.scatter(np.ones((count[0], 1)), np.array(x0), color='blue', label='label0')
    plt.scatter(np.ones((count[1], 1)), np.array(x1), color='red', label='label1')
    plt.scatter(np.ones((count[2], 1)), np.array(x2), color='yellow', label='label2')
    plt.scatter(np.ones((count[3], 1)), np.array(x3), color='black', label='label3')

    plt.xlim(0, 2)
    plt.ylim(-100, 1200)
    plt.legend(loc=2)
    plt.savefig("fig_1.png")
    plt.clf()


# 进行K-Means聚类，发现单次购买数量的异常点
def K_Means(sale_data, data_raw, data):
    sale_data.reshape(len(sale_data), -1)

    estimator = KMeans(n_clusters=4)  # 构造聚类器，聚4类
    estimator.fit(sale_data)  # 聚类
    labels = estimator.labels_  # 获取聚类标签
    sale_class = [[], [], [], []]
    count = [0, 0, 0, 0]
    # 统计每一类的个数
    for i in range(len(sales)):
        if labels[i] == 0:
            sale_class[0].append(sales[i])
            count[0] = count[0] + 1
        if labels[i] == 1:
            sale_class[1].append(sales[i])
            count[1] = count[1] + 1
        if labels[i] == 2:
            sale_class[2].append(sales[i])
            count[2] = count[2] + 1
        if labels[i] == 3:
            sale_class[3].append(sales[i])
            count[3] = count[3] + 1
    # 获取数量最少的两个类，视为异常类
    count_np = np.array(count)
    idx = np.argpartition(count_np, 2)
    count_index = idx[0]
    count_index_2 = idx[1]

    # 每个异常类的数量变为该商品的平均销售数量
    for sale in sale_class[count_index]:
        filter = np.asarray([sale[1]])
        _class = data_raw[np.in1d(data_raw[:, 6], filter)]
        data[int(sale[0])][7] = np.array(_class[:, 7]).mean()
        # 总价相应变化
        data[int(sale[0])][9] = data[int(sale[0])][7] * data[int(sale[0])][8]

    for sale in sale_class[count_index_2]:
        filter = np.asarray([sale[1]])
        _class = data_raw[np.in1d(data_raw[:, 6], filter)]
        data[int(sale[0])][7] = np.array(_class[:, 7]).mean()
        # 总价相应变化
        data[int(sale[0])][9] = data[int(sale[0])][7] * data[int(sale[0])][8]

    # 展示聚类结果
    show(labels, sale_data, count)
    # 更新的数据返回
    return data


# 主成分分析
def PCA_(X):
    # 标准化
    X_std = StandardScaler().fit(X).transform(X)
    # 构建协方差矩阵
    cov_mat = np.cov(X_std.T)
    # 特征值和特征向量
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # 求出特征值的和
    tot = sum(eigen_vals)
    # 求出每个特征值占的比例
    var_exp = [(i / tot) for i in eigen_vals]

    cum_var_exp = np.cumsum(var_exp)
    # 绘图，展示各个属性贡献量
    plt.bar(range(len(eigen_vals)), var_exp, width=1.0, bottom=0.0, alpha=0.5, label='individual explained variance')
    plt.step(range(len(eigen_vals)), cum_var_exp, where='post', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.savefig("fig_2.png")
    # 返回贡献量最小的属性的索引
    return np.argmin(var_exp)


if __name__ == "__main__":
    # 装载数据
    data_raw, sales = load_data()
    print(data_raw.shape)
    # 对商品购买情况进行聚类，并消除异常何人噪声
    data = K_Means(data_raw=data_raw, data=data_raw, sale_data=sales[:, 2:])
    # 进行主成分分析
    min_col_index = PCA_(data[:, 2:])
    # 根据主成分分析结果去掉多于属性（门店号或者PSNO）
    # 去掉总价格这一冗余属性
    data = np.hstack((data[:, 1:min_col_index + 1], data[:, min_col_index + 2:9]))
    # 保存在CVS文件中
    pd.DataFrame(data, columns=["SerNo", "PSNo", "Data", "GoodNo", "GoodID", "Num", "Price"]).to_csv("data.csv",index=False)
