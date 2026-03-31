import pickle
from preprocess import clean_text

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
#taking user input
review = input("Enter a review: ")
#apply preprocessing
cleaned = clean_text(review)
cleaned_text = " ".join(cleaned)
#convert to tf-idf
vector = vectorizer.transform([cleaned_text])
#predict rating


rating = model.predict(vector)[0]

print("\n🔍 Prediction Result")
print("-" * 30)

print(f"⭐ Predicted Rating: {rating}/5")

# Add meaning (very important)
if rating >= 4:
    print("😊 Positive Review")
elif rating == 3:
    print("😐 Neutral Review")
else:
    print("😞 Negative Review")

print("-" * 30)

