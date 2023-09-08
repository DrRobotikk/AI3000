from sklearn.preprocessing import StandardScaler
import pandas as pd

def stand_demo():
    data = pd.read_csv("dating.txt", delimiter="\t")
    data = data.iloc[:,:3]
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print ("data:\n", data_new)

    
stand_demo()

