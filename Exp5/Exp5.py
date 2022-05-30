import math
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Exp5")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--n", type=int, default=2)
parser.add_argument("--dataset", type=str, default="data.txt")

parser.set_defaults(augment=True)
args = parser.parse_args()
print(args)


def loadDataset(filename):
    dataSet = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            p_tmp = [str(i) for i in lines.split(sep="\t")]
            p_tmp[len(p_tmp) - 1] = p_tmp[len(p_tmp) - 1].strip("\n")
            for i in range(len(p_tmp)):
                p_tmp[i] = float(p_tmp[i])
            dataSet.append(p_tmp)

    return dataSet


def euclid(p1, p2, n):
    distance = 0
    for i in range(n):
        distance = distance + (p1[i] - p2[i]) ** 2
    return math.sqrt(distance)


def init_centroids(dataSet, k, n):
    _min = dataSet.min(axis=0)
    _max = dataSet.max(axis=0)

    centre = np.empty((k, n))
    # centre = np.array(dataSet).mean()
    for i in range(k):
        for j in range(n):
            # centre[i][j] = _min[j] + i*(_max[j]-_min[j])/k
            centre[i][j] = random.uniform(_min[j], _max[j])
    # print(centre)
    return centre


def cal_distance(dataSet, centroids, k, n):
    # 每个点到每个中心点的距离矩阵
    # print("CEN", centroids)
    dis = np.empty((len(dataSet), k))
    for i in range(len(dataSet)):
        for j in range(k):
            dis[i][j] = euclid(dataSet[i], centroids[j], n)
    return dis


def KMeans_Cluster(dataSet, k, n, epochs):
    epoch = 0
    centroids = init_centroids(dataSet, k, n)
    while epoch < epochs:

        distance = cal_distance(dataSet, centroids, k, n)
        # print("CEN", centroids)
        # print("DIS", distance)

        classify = []
        for i in range(k):
            classify.append([])

        # 比较距离并分类
        for i in range(len(dataSet)):
            List = distance[i].tolist()
            # 因为初始中心的选取完全随机，所以存在第一次分类，类的数量不足k的情况
            # 这里作为异常捕获，也就是distance[i]=nan的时候，证明类的数量不足
            # 则再次递归聚类，直到正常为止，返回聚类标签和中心点
            try:
                index = List.index(distance[i].min())
            except:
                labels, centroids = KMeans_Cluster(dataSet=np.array(data_set), k=args.k, n=args.n, epochs=args.epochs)
                return labels, centroids

            classify[index].append(i)
        # print("CLASS", classify)
        # 构造新的中心点
        new_centroids = np.empty((k, n))
        for i in range(len(classify)):
            # print(i)
            # print(dataSet[classify[i]])
            #
            for j in range(n):
                # print(classify[i])
                # print(dataSet[classify[i]][:, j:j + 1])
                new_centroids[i][j] = np.sum(dataSet[classify[i]][:, j:j + 1]) / len(classify[i])
                # print("IJ", new_centroids[i][j])
                # new_centroids[i][1] = np.sum(dataSet[classify[i]][1]) / len(classify[i])

        # 比较新的中心点和旧的中心点是否一样
        if (new_centroids == centroids).all():
            # print("Epochs: ", epoch)
            label_pred = np.empty(len(data_set))
            for i in range(k):
                label_pred[classify[i]] = i

            return label_pred, centroids
        else:
            centroids = new_centroids
            epoch = epoch + 1


def show(label_pred, X, centroids):
    x = []
    for i in range(args.k):
        x.append([])

    for k in range(args.k):

        for i in range(len(label_pred)):
            _l = int(label_pred[i])
            x[_l].append(X[i])
    print(x)
    for i in range(args.k):
        plt.scatter(np.array(x[i])[:, 0], np.array(x[i])[:, 1], color=plt.cm.Set1(i % 8), label='label'+str(i))
    plt.scatter(x=centroids[:, 0], y=centroids[:, 1], marker='*', label='pred_center')
    plt.legend(loc=3)
    plt.show()


if __name__ == "__main__":
    data_set = loadDataset(args.dataset)
    plt.scatter(np.array(data_set)[:, :1], np.array(data_set)[:, 1:])
    plt.show()
    # print("Data Set: ", data_set)
    labels, centroids = KMeans_Cluster(dataSet=np.array(data_set), k=args.k, n=args.n, epochs=args.epochs)
    print("Classes: ", labels)
    print("Centers: ", centroids)
    show(X=np.array(data_set), label_pred=labels, centroids=centroids)
