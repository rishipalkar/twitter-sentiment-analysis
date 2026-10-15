import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
TEST_PATH = BASE_DIR / "data" / "test_split.csv"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
}
LABELS = ["Negative", "Positive"]


def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison(metrics_df, path):
    ax = metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]] \
        .plot(kind="bar", figsize=(8, 5), ylim=(0.7, 0.85), rot=0)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def show_top_words(model, vectorizer, n=15):
    """Words with the strongest positive/negative weights in Logistic Regression."""
    words = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    order = np.argsort(coefs)
    neg_words = words[order[:n]]
    pos_words = words[order[-n:][::-1]]

    print("\nTop words -> NEGATIVE sentiment:")
    print(", ".join(neg_words))
    print("\nTop words -> POSITIVE sentiment:")
    print(", ".join(pos_words))

    # Save as a chart too
    top_idx = np.concatenate([order[:n], order[-n:]])
    plt.figure(figsize=(8, 6))
    colors = ["#d9534f"] * n + ["#5cb85c"] * n
    plt.barh(words[top_idx], coefs[top_idx], color=colors)
    plt.title("Top Words Driving Sentiment (Logistic Regression)")
    plt.xlabel("Coefficient (negative <- 0 -> positive)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_words.png", dpi=150)
    plt.close()


def main():
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["clean_text"])
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    X_test = vectorizer.transform(test_df["clean_text"])
    y_test = test_df["label"]

    rows = []
    for key, display_name in MODEL_NAMES.items():
        model = joblib.load(MODELS_DIR / f"{key}.joblib")
        y_pred = model.predict(X_test)

        rows.append({
            "Model": display_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        })

        print("=" * 55)
        print(display_name)
        print("=" * 55)
        print(classification_report(y_test, y_pred, target_names=LABELS, digits=4))

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm, "\n")
        plot_confusion_matrix(cm, f"Confusion Matrix - {display_name}",
                              OUT_DIR / f"confusion_{key}.png")

        if key == "logistic_regression":
            lr_model = model

    metrics_df = pd.DataFrame(rows)
    print("\nSUMMARY TABLE")
    print(metrics_df.round(4).to_string(index=False))

    metrics_df.round(4).to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    plot_comparison(metrics_df, OUT_DIR / "model_comparison.png")
    show_top_words(lr_model, vectorizer)

    print(f"\nAll charts and tables saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()