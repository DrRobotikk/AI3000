import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Load your dataset
data = pd.read_csv("train.csv", nrows=10)
X = data.iloc[:, 0:20]

# Replace missing values with median
X_filled = X.fillna(X.median())

# Standardize the data (recommended before PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# Perform PCA
n_components = 5  # Choose the number of principal components you want
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the transformed data
pca_columns = [f"PC{i+1}" for i in range(n_components)]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Display the explained variance ratio of each component
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Display the transformed data
print(X_pca_df)
