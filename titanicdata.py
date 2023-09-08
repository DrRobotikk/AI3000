import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score




data = pd.read_csv("Titanic.csv")




data["Sex"] = data["Sex"].map({"female": 0,"male": 1})

data["Age"] = data["Age"].fillna(data["Age"].mean())

x = data[["Pclass","Age","Sex"]]
y = data["Survived"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# build decision tree
classifier = DecisionTreeClassifier()

# Fit the model on the training data
classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#plot tree
plt.figure(figsize=(12, 8))
plot_tree(classifier,feature_names=["Pclass","Age","Sex"],class_names=["Not survived","survived"], filled=True)
plt.show()

new_data = pd.DataFrame([[1,22,0]],columns=["Pclass","Age","Sex"])
predictions = classifier.predict(new_data)
predictions_mapped = ["Survived" if prediction == 1 else "Dead" for prediction in predictions]

print("Predictions: ", predictions_mapped)
