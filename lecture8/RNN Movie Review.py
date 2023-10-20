import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import nltk

# Custom function to map numerical ratings to sentiment labels
def map_ratings_to_sentiment(rating):
    try:
        rating = int(rating)  # Try to convert the rating to an integer
        return 1 if rating >= 5 else 0  # Positive: 1, Negative: 0
    except ValueError:
        return None  # Return None for rows with invalid ratings

# Load the CSV file into a DataFrame
df = pd.read_csv("MovieReviews.csv")

# Map numerical ratings to sentiment labels, filter out rows with invalid ratings
df['Rating'] = df['Rating'].apply(map_ratings_to_sentiment)

# Drop rows with missing values in the 'Rating' or 'Review' columns
df.dropna(subset=['Rating', 'Review'], inplace=True)

# Tokenize and preprocess text data
tokenizer = Tokenizer(num_words=10000)  # Adjust num_words as needed
tokenizer.fit_on_texts(df['Review'])
sequences = tokenizer.texts_to_sequences(df['Review'])
X = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed

y = df['Rating'].values

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create an RNN (LSTM) model for text classification
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')

