import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the dataset
# The dataset should have columns: 'comment', 'date', and 'failure_flag' (1 for failure, 0 otherwise)
data_path = "comments.csv"  # Replace with your file path
data = pd.read_csv(data_path)

# Display first few rows of data
print("Sample Data:")
print(data.head())

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Sentiment Analysis
print("Performing Sentiment Analysis...")
data['sentiment'] = data['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Visualizing sentiment over time
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['sentiment'], label='Sentiment Polarity', alpha=0.75)
plt.title("Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Sentiment Polarity")
plt.legend()
plt.show()

# TF-IDF Vectorization
print("Performing TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(data['comment']).toarray()

# Combine TF-IDF features with sentiment
print("Combining Features...")
X = np.hstack((X_tfidf, data['sentiment'].values.reshape(-1, 1)))
y = data['failure_flag']

# Split data into training and testing sets
print("Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
print("Training Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
print("Testing Model...")
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualizing Feature Importance
print("Visualizing Feature Importance...")
feature_importances = model.feature_importances_
important_features = sorted(
    zip(feature_importances, vectorizer.get_feature_names_out()), reverse=True
)[:10]
labels, values = zip(*important_features)

plt.figure(figsize=(12, 6))
plt.barh(labels, values, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features for Failure Prediction")
plt.gca().invert_yaxis()
plt.show()

# Predict future failures
print("Predicting Future Failures...")
future_comments = ["The system is slow and unresponsive.", "Everything is working fine."]
future_sentiments = [TextBlob(comment).sentiment.polarity for comment in future_comments]
future_tfidf = vectorizer.transform(future_comments).toarray()
future_features = np.hstack((future_tfidf, np.array(future_sentiments).reshape(-1, 1)))
future_predictions = model.predict(future_features)

for comment, pred in zip(future_comments, future_predictions):
    print(f"Comment: '{comment}' - Predicted Failure: {bool(pred)}")
