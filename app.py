import pandas as pd
import streamlit as st
from pathlib import Path
from predict import predict_sentiment

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"

st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="🐦", layout="centered")

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Naive Bayes": "naive_bayes",
}

EXAMPLES = [
    "I absolutely love this new phone, the camera is amazing!",
    "Worst customer service ever. I am so disappointed.",
    "This movie was not bad at all",
    "Feeling sick and miss my friends today",
]

if "tweet" not in st.session_state:
    st.session_state.tweet = ""


def set_example(text):
    st.session_state.tweet = text


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "Trained on 100k tweets from the Sentiment140 dataset using "
        "TF-IDF (unigrams + bigrams) features."
    )
    st.markdown(
        "**Limitations**\n\n"
        "- English only\n"
        "- Only positive / negative (no neutral class)\n"
        "- Struggles with sarcasm"
    )

# ---------------- Main page ----------------
st.title("🐦 Tweet Sentiment Analyzer")
st.caption("Enter a tweet and the model will predict whether it is positive or negative.")

tab_analyze, tab_perf = st.tabs(["Analyze", "Model Performance"])

# ---------- Tab 1: Analyze ----------
with tab_analyze:
    st.text_area("Your tweet", key="tweet", height=120,
                 placeholder="Type or paste a tweet here...")

    st.write("Try an example:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        cols[i % 2].button(example, key=f"ex_{i}", on_click=set_example,
                           args=(example,), use_container_width=True)

    if st.button("Analyze Sentiment", type="primary"):
        tweet = st.session_state.tweet.strip()
        if not tweet:
            st.warning("Please enter a tweet first.")
        else:
            result = predict_sentiment(tweet, MODEL_OPTIONS[model_name])
            label = result["label"]

            if label == "Positive":
                st.success(f"😊 Positive  ({result['confidence']:.1%} confidence)")
            elif label == "Negative":
                st.error(f"😞 Negative  ({result['confidence']:.1%} confidence)")
            elif label == "Uncertain":
                st.warning(f"🤔 Uncertain  ({result['confidence']:.1%} confidence). "
                           "The model is not sure about this one.")
            else:
                st.info(result.get("note", "Could not analyze this text."))

            if "p_positive" in result:
                st.write("**Probability breakdown**")
                st.progress(result["p_positive"], text=f"Positive: {result['p_positive']:.1%}")
                st.progress(result["p_negative"], text=f"Negative: {result['p_negative']:.1%}")

            with st.expander("See the cleaned text the model actually used"):
                st.code(result["cleaned"] or "(empty)")

# ---------- Tab 2: Model Performance ----------
with tab_perf:
    metrics_path = OUT_DIR / "metrics_summary.csv"
    if metrics_path.exists():
        st.subheader("Metrics on the test set")
        st.dataframe(pd.read_csv(metrics_path), hide_index=True, use_container_width=True)
    else:
        st.info("Run evaluate.py first to generate the metrics.")

    charts = [
        ("model_comparison.png", "Model comparison"),
        ("confusion_logistic_regression.png", "Confusion matrix: Logistic Regression"),
        ("confusion_naive_bayes.png", "Confusion matrix: Naive Bayes"),
        ("top_words.png", "Top words driving sentiment"),
    ]
    for filename, caption in charts:
        path = OUT_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption, width=520)