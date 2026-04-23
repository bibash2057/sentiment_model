import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = data = {
    'review': [
        "This product is amazing and works perfectly", # Positive
        "Waste of money, very disappointed",           # Negative
        "Best purchase I made this year",               # Positive
        "The quality is poor and it broke quickly",     # Negative
        "I love this, highly recommended",              # Positive
        "Horrible service and bad experience"           # Negative
    ],
    'sentiment': [1, 0, 1, 0, 1, 0] # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)


# 2. THE VECTORIZER: Convert text to "TF-IDF" numbers
# This calculates how important a word is in a sentence
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df["review"])
y = df['sentiment']

# 3. THE CLASSIFIER: Using Naive Bayes (great for text)
model = MultinomialNB()
model.fit(x,y)

# 4. SAVE THE BRAIN: Export so we can use it in our apps
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, 'vectorizer.pkl')
print("Model trained and saved locally!")