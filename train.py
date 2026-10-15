import time
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "sentiment140_clean.csv"
TEST_PATH = BASE_DIR / "data" / "test_split.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    # 1. Load cleaned data
    df = pd.read_csv(CLEAN_PATH).dropna(subset=["clean_text"])
    print("Loaded:", df.shape)

    # 2. Train/test split (stratified keeps classes balanced in both sets)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
    )
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # 3. TF-IDF features (unigrams + bigrams)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100000,
        min_df=3,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])
    y_train, y_test = train_df["label"], test_df["label"]
    print("TF-IDF matrix shape (train):", X_train.shape)

    # 4. Train models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
        "naive_bayes": MultinomialNB(alpha=1.0),
    }

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"\n[{name}]")
        print(f"  Training time : {elapsed:.1f}s")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy : {test_acc:.4f}")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    # 5. Save vectorizer and test set for evaluation/demo
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\nSaved models + vectorizer to {MODELS_DIR}")
    print(f"Saved test set to {TEST_PATH}")


if __name__ == "__main__":
    main()