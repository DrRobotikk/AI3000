# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



data = pd.read_csv("FB Competition.csv")
#data = data.query("x<2.5 & x>2 & y<2.5 & y>2")

time_value = pd.to_datetime(data["time"], unit="s")
data["day"] = time_value.dt.day
data["hour"] = time_value.dt.hour
data["weekday"] = time_value.dt.weekday

#x = data[["x", "y", "day", "hour", "weekday"]]
x = data[["x", "y"]]
y = data["place_id"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a KNN classifier with k=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fit the model on the training data
knn_classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = knn_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


    

