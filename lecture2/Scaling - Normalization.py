from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def minmax_demo():
    data = pd.read_csv("dating.txt",delimiter="\t" )
    data = data.iloc[:,:3]
    transfer = MinMaxScaler(feature_range=(1,2))
    data_new = transfer.fit_transform(data)
    print ("data:\n", data_new)

    
minmax_demo()

