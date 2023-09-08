import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def minMaxDemo():
    data = pd.read_csv("scaling-dating.txt", delimiter="\t")
    data = data.iloc[:, :3]
    transfer = MinMaxScaler(feature_range=(1, 2))
    data_new = transfer.fit_transform(data)
    print("data:\n", data_new)


def stand_demo():
    data = pd.read_csv("scaling-dating.txt", delimiter="\t")
    data = data.iloc[:, :3]
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print("data:\n", data_new)

stand_demo()