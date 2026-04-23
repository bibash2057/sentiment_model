import joblib

# Load your saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load('vectorizer.pkl')

def analyze_review (text):
    # Convert input text to the same number format as the training
    text_vector = tfidf.transform([text])

    # Predict!
    prediction = model.predict(text_vector)

    return "Positive 😊" if prediction[0] == 1 else "Negative 😠"

print(analyze_review("This product is amazing and works perfectly"))
print(analyze_review("It is a very bad product"))