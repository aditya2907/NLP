import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from datetime import datetime, timedelta

# Step 1: Load Data from comments.csv
def load_data_from_csv(file_path):
    # Load the CSV file into a Pandas DataFrame
    data = pd.read_csv(file_path)

    # Ensure 'date' column is in datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Convert 'system_id' to integer if not already
    data['system_id'] = data['system_id'].astype(int)
    return data

# Step 2: Process the Data
def process_and_train_model(data):
    # Sentiment Analysis
    data['sentiment'] = data['comment'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # TF-IDF Vectorization for 'comment'
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_tfidf = vectorizer.fit_transform(data['comment']).toarray()

    # Combine features: TF-IDF, sentiment, and system ID
    X = np.hstack((X_tfidf, data['sentiment'].values.reshape(-1, 1), data['system_id'].values.reshape(-1, 1)))
    y = data['failure_flag']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, vectorizer

# Step 3: Predict Failures for a New System ID and Date
def predict_failure(model, vectorizer, data, system_id, input_date):
    # Convert date to datetime
    input_date = pd.to_datetime(input_date)

    # Filter relevant data (e.g., recent comments for the system)
    recent_comments = data[(data['system_id'] == system_id) & (data['date'] >= input_date - timedelta(days=7))]
    
    if recent_comments.empty:
        print(f"No recent data found for System {system_id}. Unable to predict.")
        return

    # Aggregate recent data (e.g., mean sentiment)
    mean_sentiment = recent_comments['sentiment'].mean()

    # Example input comment for TF-IDF (replace with a meaningful one if available)
    example_comment = "System showing unusual behavior."
    tfidf_vector = vectorizer.transform([example_comment]).toarray()

    # Combine features
    input_features = np.hstack((tfidf_vector, [[mean_sentiment]], [[system_id]]))
    
    # Predict
    prediction = model.predict(input_features)
    print(f"Prediction for System {system_id} on {input_date.date()}: {'Failure' if prediction[0] == 1 else 'Success'}")

# Main Execution
if __name__ == "__main__":
    # Load data from CSV
    csv_file_path = "comments.csv"  # Replace with the correct path to your file
    df = load_data_from_csv(csv_file_path)
    print("Data loaded from CSV:")
    print(df.head())

    # Train model
    model, vectorizer = process_and_train_model(df)

    # Example prediction
    predict_failure(model, vectorizer, df, 101, "2024-11-06")  # Example system_id = 101
    predict_failure(model, vectorizer, df, 102, "2024-11-06")  # Example system_id = 102
