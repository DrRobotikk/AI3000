import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Load input data
X = pd.read_csv("Mall_Customers.csv", delimiter=',')
num_clusters = 5

# Convert 'Annual Income (k$)' and 'Spending Score (1-100)' columns to numeric
X['Annual Income (k$)'] = pd.to_numeric(
    X['Annual Income (k$)'], errors='coerce')
X['Spending Score (1-100)'] = pd.to_numeric(
    X['Spending Score (1-100)'], errors='coerce')

# Remove rows with NaN values
X = X.dropna(subset=['Annual Income (k$)', 'Spending Score (1-100)'])

# Plot input data
plt.figure()
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
            marker='o', facecolors='none', edgecolors='orange', s=80)

# Create KMeans object
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

# Train the KMeans clustering model
kmeans.fit(X[['Annual Income (k$)', 'Spending Score (1-100)']])

# Step size of the mesh
step_size = 0.01

# Define the grid of points to plot the boundaries
x_vals, y_vals = np.meshgrid(np.arange(X['Annual Income (k$)'].min() - 1, X['Annual Income (k$)'].max() + 1, step_size),
                             np.arange(X['Spending Score (1-100)'].min() - 1, X['Spending Score (1-100)'].max() + 1, step_size))

# Predict output labels for all the points on the grid
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

# Plot different regions and color them
output = output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

# Overlay input points
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
            marker='o', facecolors='none', edgecolors='blue', s=80)

# Plot the centers of clusters
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=210, linewidths=4, color='red',
            zorder=12, facecolors='black')

plt.title('Boundaries of clusters')
plt.xlim(x_vals.min(), x_vals.max())
plt.ylim(y_vals.min(), y_vals.max())
plt.xticks(())
plt.yticks(())

plt.show()
