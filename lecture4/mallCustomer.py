import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA


data = pd.read_csv('Mall_Customers.csv')

enCoder = OneHotEncoder()
genderEncoded = enCoder.fit_transform(data[["Gender"]])

x = np.concatenate((data[["Age", "Annual Income (k$)",
                   "Spending Score (1-100)"]].values, genderEncoded), axis=1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


bandwith = estimate_bandwidth(x_scaled, quantile=0.1, n_samples=len(x_scaled))
meanshift = MeanShift(bandwidth=bandwith)

nClusters = len(np.unique(meanshift.labels_))
print(f"Number of clusters: {nClusters}")

kmeans = KMeans(init='k-means++', n_clusters=nClusters, random_state=42)
kmeans.fit(x_scaled)

data["cluster"] = kmeans.labels_

pca = PCA(nComponents=2)
x_pca = pca.fit_transform(x_scaled)
data["pca1"] = x_pca[:, 0]
data["pca2"] = x_pca[:, 1]

plt.figure(figsize=(10, 8))
for cluster in data["cluster"].unique():
    plt.scatter(data[data["cluster"] == cluster]["pca1"],
                data[data["cluster"] == cluster]["pca2"],
                label=f"Cluster {cluster}", alpha=0.7)

plt.title("K-means Clustering Result")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.legend()
plt.grid()
plt.show()
