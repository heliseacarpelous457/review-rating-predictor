import string
from nltk.corpus import stopwords

# Load stopwords once
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # 4. Return tokens
    return words
if __name__ == "__main__":
    sample = "This Product is AMAZING!!!"
    print(clean_text(sample))