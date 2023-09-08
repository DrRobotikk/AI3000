from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Define the categories
categories = ['Human', 'Penguin', 'Octopus', 'Alien']

# Create the OneHotEncoder instance
encoder = OneHotEncoder(categories=[categories])

# Fit and transform the data
data = [['Human'], ['Penguin'], ['Octopus'], ['Alien']]
encoded_data = encoder.fit_transform(data).toarray()

# Print the encoded data
print(encoded_data)
