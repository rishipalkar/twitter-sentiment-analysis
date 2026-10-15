import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "training.1600000.processed.noemoticon.csv"
SAMPLE_PATH = BASE_DIR / "data" / "sentiment140_sample.csv"
SAMPLE_PER_CLASS = 50000   # 50k negative + 50k positive = 100k tweets
RANDOM_STATE = 42


def load_raw_data(path=RAW_PATH):
    """Load the raw Sentiment140 CSV (no header, latin-1 encoding)."""
    cols = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(path, encoding="latin-1", names=cols)
    return df


def prepare_sample(df, n_per_class=SAMPLE_PER_CLASS):
    """Keep only needed columns, map labels to 0/1, take a balanced sample."""
    df = df[["target", "text"]].copy()

    # Original labels: 0 = negative, 4 = positive -> convert to 0 / 1
    df["label"] = df["target"].map({0: 0, 4: 1})
    df = df.drop(columns=["target"])

    # Balanced sample: take n rows from each class separately, then combine
    neg = df[df["label"] == 0].sample(n=n_per_class, random_state=RANDOM_STATE)
    pos = df[df["label"] == 1].sample(n=n_per_class, random_state=RANDOM_STATE)

    sample = (
        pd.concat([neg, pos])
          .sample(frac=1, random_state=RANDOM_STATE)   # shuffle
          .reset_index(drop=True)
    )
    return sample[["text", "label"]]


if __name__ == "__main__":
    df = load_raw_data()
    print("Full dataset shape:", df.shape)
    print("\nOriginal label counts:\n", df["target"].value_counts())

    sample = prepare_sample(df)
    sample.to_csv(SAMPLE_PATH, index=False)

    print("\nSample shape:", sample.shape)
    print("Sample label counts (0=neg, 1=pos):\n", sample["label"].value_counts())
    print("\nFirst 5 rows:\n", sample.head())
    print(f"\nSaved sample to {SAMPLE_PATH}")