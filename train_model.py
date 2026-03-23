import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load datasets
fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")

# Add label columns
fake['label'] = 0   # 0 = Fake
true['label'] = 1   # 1 = Real

# Combine and shuffle
df = pd.concat([fake, true]).sample(frac=1, random_state=42)

# Split features and labels
X = df['text']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_vect, y)

# Save model and vectorizer
joblib.dump(model, "lr_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")