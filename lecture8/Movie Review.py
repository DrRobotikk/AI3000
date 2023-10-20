import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
import nltk

# Extract features from the input list of words
def extract_features(words):
    return dict([(word, True) for word in words])

# Custom function to map numerical ratings to sentiment labels
def map_ratings_to_sentiment(rating):
    try:
        rating = int(rating)  # Try to convert the rating to an integer
        return "Positive" if rating >= 5 else "Negative"
    except ValueError:
        return None  # Return None for rows with invalid ratings

# Load the CSV file into a DataFrame
df = pd.read_csv("MovieReviews.csv")

# Map numerical ratings to sentiment labels, filter out rows with invalid ratings
df['Rating'] = df['Rating'].apply(map_ratings_to_sentiment)

# Drop rows with missing values in the 'Rating' or 'Review' columns
df.dropna(subset=['Rating', 'Review'], inplace=True)

# Extract the features from the reviews
features = [(extract_features(nltk.word_tokenize(review)), rating) for review, rating in zip(df['Review'], df['Rating'])]

# Define the train and test split (80% and 20%)
threshold = 0.8
num_data = len(features)
num_train = int(threshold * num_data)

# Create training and testing datasets
features_train = features[:num_train]
features_test = features[num_train:]

# Print the number of datapoints used
print('\nNumber of training datapoints:', len(features_train))
print('Number of test datapoints:', len(features_test))

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(features_train)
print('\nAccuracy of the classifier:', nltk_accuracy(classifier, features_test))

N = 15
print('\nTop ' + str(N) + ' most informative words:')
for i, item in enumerate(classifier.most_informative_features()[:N]):
    print(str(i + 1) + '. ' + item[0])

# Test input movie reviews
input_reviews = [
    'The costumes in this movie were great',
    'I think the story was terrible and the characters were very weak',
    'People say that the director of the movie is amazing',
    'This is such an idiotic movie. I will not recommend it to anyone.'
]

print("\nMovie review predictions:")
for review in input_reviews:
    print("\nReview:", review)

    # Compute the probabilities
    probabilities = classifier.prob_classify(extract_features(nltk.word_tokenize(review)))

    # Pick the maximum value
    predicted_sentiment = probabilities.max()

    # Print outputs
    print("Predicted sentiment:", predicted_sentiment)
    print("Probability:", round(probabilities.prob(predicted_sentiment), 2))
