import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np

# Sample movie reviews dataset (positive and negative reviews)
reviews = [
    "This movie was fantastic and I loved it!",
    "Terrible movie, a complete waste of time.",
    "The acting and direction were superb.",
    "I couldn't stand the plot and the characters were poorly developed."
]

# Corresponding sentiment labels (1 for positive, 0 for negative)
sentiments = [1, 0, 1, 0]

# Tokenize the text data
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

# Pad sequences to a fixed length
max_length = 10
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Create the RNN model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert labels to a NumPy array
labels = np.array(sentiments)

# Train the model
model.fit(padded_sequences, labels, epochs=10)

# Evaluate the model
test_reviews = [
    "I absolutely enjoyed the film!",
]
test_sequences = tokenizer.texts_to_sequences(test_reviews)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')
predictions = model.predict(padded_test_sequences)

for i in range(len(test_reviews)):
    sentiment = "Positive" if predictions[i] > 0.5 else "Negative"
    print(f"Review: {test_reviews[i]}\nSentiment: {sentiment}\n")

