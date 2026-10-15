import re
import html
import pandas as pd
import nltk
from pathlib import Path
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "data" / "sentiment140_sample.csv"
CLEAN_PATH = BASE_DIR / "data" / "sentiment140_clean.csv"

# Download stopwords once (skipped automatically if already present)
nltk.download("stopwords", quiet=True)

# Keep negations: they flip sentiment ("not good" != "good")
NEGATIONS = {
    "no", "nor", "not", "never", "don", "don't", "doesn", "doesn't", "didn",
    "didn't", "isn", "isn't", "wasn", "wasn't", "aren", "aren't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't", "couldn", "couldn't",
    "shouldn", "shouldn't", "haven", "haven't", "hasn", "hasn't", "hadn",
    "hadn't", "can't", "cannot", "ain", "mustn", "mustn't", "needn",
    "needn't", "shan", "shan't", "mightn", "mightn't",
}
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")          # 3+ repeated chars -> 2
NON_LETTER_RE = re.compile(r"[^a-z\s']")      # keep letters, spaces, apostrophes
SPACES_RE = re.compile(r"\s+")


def clean_tweet(text):
    """Clean a single tweet and return the processed string."""
    text = html.unescape(str(text)).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)         # #happy -> happy
    text = REPEAT_RE.sub(r"\1\1", text)        # soooo -> soo
    text = NON_LETTER_RE.sub(" ", text)
    text = SPACES_RE.sub(" ", text).strip()

    tokens = [
        w.strip("'") for w in text.split()
        if w.strip("'") and w.strip("'") not in STOP_WORDS and len(w.strip("'")) > 1
    ]
    return " ".join(tokens)


def preprocess_dataset(df):
    """Add a 'clean_text' column and drop tweets that became empty."""
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_tweet)
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"Dropped {before - len(df)} tweets that were empty after cleaning.")
    return df


if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_PATH)
    print("Loaded:", df.shape)

    df = preprocess_dataset(df)

    print("\nBefore vs After (first 8 tweets):")
    for _, row in df.head(8).iterrows():
        print(f"  RAW  : {row['text']}")
        print(f"  CLEAN: {row['clean_text']}\n")

    df.to_csv(CLEAN_PATH, index=False)
    print("Final shape:", df.shape)
    print("Label counts:\n", df["label"].value_counts())
    print(f"\nSaved cleaned data to {CLEAN_PATH}")