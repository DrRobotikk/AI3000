import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data from the "dating.txt" file into a DataFrame
data = pd.read_csv('dating.txt', delimiter='\t')

# Step 2: Select the first 3 columns as the feature matrix X and the last column as the target variable Y
X = data.iloc[:, :3]  # Select the first 3 columns as features
Y = data.iloc[:, -1]  # Select the last column as the target variable

# Step 3: Normalize (scale) the feature matrix X using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Print the original data and the scaled data
print("Original Data:")
print(data.head())

print("\nScaled Data:")
scaled_data = pd.DataFrame(X_scaled, columns=X.columns)
scaled_data['Target'] = Y  # Add the target variable back to the scaled data
print(scaled_data.head())
