import pandas as pd
import pickle
from preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- Step 1: Load dataset ---
df = pd.read_csv("/Users/pavithrajayapal/Desktop/Pavi/fake_review_prediction/dataset/custom_review_dataset_with_text_labels (6).csv")
reviews = df['text']
labels = df['rating']

def convert_rating(r):
    if r <= 2:
        return 0   # Negative
    elif r == 3:
        return 1   # Neutral
    else:
        return 2   # Positive

labels = labels.apply(convert_rating)

# --- Step 2: Preprocess ---
cleaned_reviews = reviews.apply(lambda x: " ".join(clean_text(x)))

# --- Step 3: Split ---
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_reviews,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f"Training on {len(X_train)} reviews...")
print(f"Testing on {len(X_test)} reviews...")

# --- Step 4: TF-IDF ---
vectorizer = TfidfVectorizer(max_features=5000,
    ngram_range=(1,2))

X_train_tfidf = vectorizer.fit_transform(X_train)   # learn vocab
X_test_tfidf = vectorizer.transform(X_test)         # apply vocab

# --- Step 5: Train model ---
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)

class_weights = dict(zip(classes, weights))

sample_weights = np.array([class_weights[y] for y in y_train])

model = MultinomialNB()
model.fit(X_train_tfidf, y_train, sample_weight=sample_weights)

# --- Step 6: Predict ---
y_pred = model.predict(X_test_tfidf)

# --- Step 7: Evaluate ---
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"🎯 Accuracy Score: {accuracy:.2%}")
print("-" * 30)
print(classification_report(y_test, y_pred))

print(labels.unique())

print(X_train.head())
print(X_test.head())
print(labels.value_counts())

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# --- Step 8: Save ---
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved!")