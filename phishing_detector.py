import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv('phishing_site_urls.csv')  # Replace 'dataset.csv' with your dataset filename

# Step 2: Display the first few rows to check the structure
print("Dataset Preview:")
print(data.head())  # Displays the first 5 rows

# Step 3: Check for basic details of the dataset
print("\nDataset Info:")
print(data.info())  # Provides column types, non-null counts, etc.

# Step 4: Check the distribution of labels (phishing vs legitimate)
print("\nLabel Distribution:")
print(data['Label'].value_counts())  # Count phishing (1) and legitimate (0) rows

# Step 1: Remove missing rows
data = data.dropna()  # Removes rows with any missing values
print("\nNumber of rows after removing missing values:", len(data))

# Step 2: Remove duplicate rows
data = data.drop_duplicates()  # Removes duplicate rows
print("\nNumber of rows after removing duplicates:", len(data))

# Step 1: Separate features (URLs) and labels
X = data['URL']  # URLs or textual data to analyze
y = data['Label']  # Labels: 1 = phishing, 0 = legitimate

# Display sample features and labels
print("\nSample Features (X):")
print(X.head())  # First few URLs
print("\nSample Labels (y):")
print(y.head())  # Corresponding labels

from sklearn.model_selection import train_test_split

# Step 1: Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Verify the split
print("\nTraining Set Size:", len(X_train))
print("Testing Set Size:", len(X_test))

# Convert URLs to numerical data using TF-IDF
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def predict_url(url):
    url_vectorized = vectorizer.transform([url])
    prediction = model.predict(url_vectorized)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.externals import joblib  # For loading the trained model

# Step 1: Initialize Flask App
app = Flask(__name__)

# Step 2: Load Trained Model
model = joblib.load('phishing_model.pkl')  # Replace with your model filename
vectorizer = joblib.load('vectorizer.pkl')  # Replace with your vectorizer filename

# Step 3: Home Route
@app.route('/')
def home():
    return render_template('index.html')  # Render the main HTML page

# Step 4: Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']  # Get URL input from the form
    features = vectorizer.transform([url])  # Vectorize the input URL
    prediction = model.predict(features)[0]  # Predict using the loaded model
    
    # Determine result
    result = "Phishing" if prediction == 1 else "Legitimate"
    
    return render_template('result.html', url=url, result=result)  # Render result page

# Step 5: Run the App
if __name__ == '__main__':
    app.run(debug=True)
