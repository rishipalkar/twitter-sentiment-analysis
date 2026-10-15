import sys
import joblib
from pathlib import Path
from preprocess import clean_tweet   # reuse the exact same cleaning as training

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

UNCERTAIN_THRESHOLD = 0.60   # below this confidence -> "Uncertain"

_vectorizer = None
_models = {}


def _load(model_key):
    """Load vectorizer/model once and cache them."""
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    if model_key not in _models:
        _models[model_key] = joblib.load(MODELS_DIR / f"{model_key}.joblib")
    return _vectorizer, _models[model_key]


def predict_sentiment(text, model_key="logistic_regression"):
    """Return a dict with the label, confidence, and cleaned text."""
    vectorizer, model = _load(model_key)
    cleaned = clean_tweet(text)

    if not cleaned:
        return {"label": "Unknown", "confidence": 0.0, "cleaned": cleaned,
                "note": "Nothing left after cleaning (only mentions/links/stopwords?)"}

    features = vectorizer.transform([cleaned])
    proba = model.predict_proba(features)[0]          # [P(negative), P(positive)]
    pred = int(proba.argmax())
    confidence = float(proba[pred])

    if confidence < UNCERTAIN_THRESHOLD:
        label = "Uncertain"
    else:
        label = "Positive" if pred == 1 else "Negative"

    return {"label": label, "confidence": confidence, "cleaned": cleaned,
            "p_negative": float(proba[0]), "p_positive": float(proba[1])}


def print_result(text, result):
    print(f"\nTweet     : {text}")
    print(f"Cleaned   : {result['cleaned']}")
    print(f"Sentiment : {result['label']} (confidence: {result['confidence']:.1%})")


def run_demo():
    samples = [
        "I absolutely love this new phone, the camera is amazing!",
        "Worst customer service ever. I am so disappointed.",
        "This movie was not good at all",
        "This movie was not bad at all",
        "Can't wait for the weekend, so excited!!! #happy",
        "Feeling sick and miss my friends today",
        "Just reached the office",
    ]
    print("=" * 60)
    print("DEMO PREDICTIONS")
    print("=" * 60)
    for s in samples:
        print_result(s, predict_sentiment(s))


def run_interactive():
    print("\nTweet Sentiment Analyzer  (type 'quit' to exit)")
    while True:
        text = input("\nEnter a tweet: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not text:
            continue
        print_result(text, predict_sentiment(text))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Usage: python predict.py "some tweet text"
        tweet = " ".join(sys.argv[1:])
        print_result(tweet, predict_sentiment(tweet))
    else:
        run_demo()
        run_interactive()