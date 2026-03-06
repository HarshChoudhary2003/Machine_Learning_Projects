import pickle
import re
from src.preprocessing import clean_text, preprocess_text

def predict_sentiment(text, model, vectorizer):
    # Preprocess
    cleaned = clean_text(text)
    processed = preprocess_text(cleaned)
    
    # Vectorize
    vec = vectorizer.transform([processed])
    
    # Predict
    pred = model.predict(vec)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
    
    sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment, prob

def main():
    # Load model and vectorizer
    try:
        model = pickle.load(open("models/best_model.pkl", "rb"))
        vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        print("Model and Vectorizer loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Please run main.py first.")
        return

    # Example test
    test_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
        "I hated this film. It was boring, long, and the characters were annoying.",
        "An okay movie, but not something I would watch again. A bit average."
    ]

    print("\nTesting Model on Sample Reviews:")
    print("-" * 40)
    for review in test_reviews:
        sentiment, _ = predict_sentiment(review, model, vectorizer)
        print(f"Review: {review[:60]}...")
        print(f"Prediction: {sentiment}")
        print("-" * 40)

    # Interactive loop
    print("\nTry it yourself! Enter a review (or 'q' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break
        sentiment, prob = predict_sentiment(user_input, model, vectorizer)
        print(f"Sentiment: {sentiment}")
        if prob is not None:
             print(f"Confidence: {max(prob)*100:.2f}%")

if __name__ == "__main__":
    main()
