import streamlit as st
import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# STYLING HELPERS (CSS)
# =========================

def add_custom_css():
    st.markdown(
        """
        <style>
        /* Main page padding */
        .main > div {
            padding-top: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
            padding-bottom: 3rem;
        }

        /* Hero banner */
        .hero {
            background: linear-gradient(90deg, #2563EB, #1D4ED8);
            padding: 24px 32px;
            border-radius: 18px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.25);
        }

        .hero h1 {
            font-size: 30px;
            margin-bottom: 0.3rem;
        }

        .hero p {
            font-size: 15px;
            opacity: 0.95;
        }

        /* Section titles */
        .section-title {
            font-size: 20px;
            font-weight: 600;
            margin-top: 10px;
            margin-bottom: 5px;
        }

        /* Make metric cards a bit nicer */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border-radius: 14px;
            padding: 12px 16px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.10);
        }

        /* Buttons */
        .stButton > button {
            border-radius: 12px;
            background: #2563EB;
            color: white;
            border: none;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 3px 10px rgba(37, 99, 235, 0.4);
        }
        .stButton > button:hover {
            background: #1D4ED8;
        }

        textarea {
            border-radius: 12px !important;
        }

        /* Dataframe styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# CACHED LOADERS
# =========================

@st.cache_resource
def load_svm_model(path="svm_pipeline.pkl"):
    return joblib.load(path)

@st.cache_resource
def load_transformer_model(model_dir="asabeysj/lagos-flood-bert"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

@st.cache_data
def load_dataset(path="synthetic_lagos_floods.csv"):
    df = pd.read_csv(path)
    # ensure clean_text exists
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"]
    return df


LABELS = ["negative", "neutral", "positive"]


def predict_svm(text, svm_model):
    return svm_model.predict([text])[0]


def predict_transformer(text, tokenizer, model):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
    return LABELS[pred_id]


# =========================
# PAGE SECTIONS
# =========================

def show_overview(df, model_choice):
    st.markdown('<div class="section-title">📊 Sentiment Overview</div>', unsafe_allow_html=True)

    # Sentiment counts
    sentiment_counts = df["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0)
    total = len(df)

    if total > 0:
        neg_pct = sentiment_counts["negative"] / total * 100
        neu_pct = sentiment_counts["neutral"] / total * 100
        pos_pct = sentiment_counts["positive"] / total * 100
    else:
        neg_pct = neu_pct = pos_pct = 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tweets", total)
    with col2:
        st.metric("Negative %", f"{neg_pct:.1f}%")
    with col3:
        st.metric("Neutral %", f"{neu_pct:.1f}%")
    with col4:
        st.metric("Positive %", f"{pos_pct:.1f}%")

    # Alert logic
    st.markdown("")

    if total > 0:
        if neg_pct >= 60:
            st.error("🚨 **Crisis Alert:** Very high negative sentiment detected in Lagos flood-related posts.")
        elif neg_pct >= 40:
            st.warning("⚠ **Moderate Concern:** Negative sentiment is elevated. Situation requires monitoring.")
        else:
            st.success("✅ **Situation Stable:** No critical spike in negative sentiment detected.")

    st.markdown('<div class="section-title">📈 Sentiment Distribution</div>', unsafe_allow_html=True)
    chart_df = pd.DataFrame(
        {
            "Sentiment": sentiment_counts.index,
            "Count": sentiment_counts.values,
        }
    )
    st.bar_chart(chart_df.set_index("Sentiment"))

    st.info(
        f"The chart above shows sentiment distribution for Lagos flood-related posts using **{model_choice}**."
    )


def show_tweet_table(df, model_choice):
    st.markdown('<div class="section-title">💬 Classified Tweets Explorer</div>', unsafe_allow_html=True)
    st.write(
        f"Below is a sample of tweets and their predicted sentiment using **{model_choice}**. "
        "You can use this to inspect real examples of public reactions."
    )

    st.dataframe(
        df[["text", "predicted_sentiment", "label"]].rename(
            columns={
                "text": "Tweet",
                "predicted_sentiment": "Predicted Sentiment",
                "label": "Original Label (Synthetic)",
            }
        ).head(100),
        use_container_width=True,
    )


def show_manual_tool(model_choice, svm_model, tokenizer, transformer_model):
    st.markdown('<div class="section-title">🧪 Manual Tweet Classification</div>', unsafe_allow_html=True)
    st.write(
        "Paste any tweet, message, or simulated report about flooding in Lagos. "
        "The system will classify it using the selected model."
    )

    user_text = st.text_area(
        "Enter tweet text:",
        height=100,
        placeholder="e.g. 'water don full everywhere for Lekki o, we need help abeg'",
    )

    if st.button("Classify Sentiment"):
        if not user_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            if model_choice == "SVM (Baseline)":
                pred = predict_svm(user_text, svm_model)
            else:
                pred = predict_transformer(user_text, tokenizer, transformer_model)

            st.success(f"Predicted Sentiment: **{pred.upper()}**")

            if pred == "negative":
                st.write(
                    "This indicates a **negative sentiment**, suggesting distress, concern, or dissatisfaction. "
                    "For crisis managers, such posts may contain early warnings, calls for help, "
                    "or evidence of service failures."
                )
            elif pred == "neutral":
                st.write(
                    "This indicates a **neutral sentiment**, usually describing events or conditions without strong emotion. "
                    "Neutral posts can be valuable for situational awareness and factual updates."
                )
            else:
                st.write(
                    "This indicates a **positive sentiment**, often reflecting relief, successful interventions, "
                    "or improving conditions. For agencies, this can help validate the effectiveness of response actions."
                )


def show_model_info():
    st.markdown('<div class="section-title">⚙️ Model Information</div>', unsafe_allow_html=True)
    st.write(
        """
        This prototype system compares two different approaches to sentiment analysis:

        1. **Support Vector Machine (SVM) – Baseline**
           - Uses TF–IDF features extracted from preprocessed tweets.
           - Represents a traditional machine learning pipeline.
           - Computationally lightweight and easy to deploy.

        2. **Transformer-Based Model (BERT) – Advanced**
           - Uses a pre-trained BERT encoder fine-tuned on Lagos flood-related data.
           - Captures deeper semantic and contextual information.
           - More robust to informal language, including Nigerian Pidgin.

        In the full dissertation, these models are evaluated using accuracy, precision, recall, and F1-score,
        and their suitability for deployment in Nigerian crisis management agencies such as NEMA and LASEMA is
        critically discussed.
        """
    )


# =========================
# MAIN APP
# =========================

def main():
    st.set_page_config(
        page_title="Lagos Flood Crisis Monitoring Dashboard",
        layout="wide",
    )

    add_custom_css()

    # Hero section
    st.markdown(
        """
        <div class="hero">
            <h1>🛰 Lagos Flood Crisis Monitoring System</h1>
            <p>
                Real-time sentiment analysis of social media posts about flooding in Lagos, 
                designed to support situational awareness and decision-making for crisis managers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load resources
    with st.spinner("Loading models and dataset..."):
        df = load_dataset()
        svm_model = load_svm_model()
        tokenizer, transformer_model = load_transformer_model()

    # Sidebar
    st.sidebar.title("⚙️ Controls")

    model_choice = st.sidebar.radio(
        "Model for analysis:",
        ["Transformer (BERT)", "SVM (Baseline)"],
        index=0,
    )

    sentiment_filter = st.sidebar.multiselect(
        "Filter by predicted sentiment:",
        options=LABELS,
        default=LABELS,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This dashboard is part of a research project on real-time sentiment analysis for "
        "crisis management in Lagos, Nigeria."
    )

    # Apply selected model to dataset for dashboard
    with st.spinner("Applying selected model to dataset..."):
        if model_choice == "SVM (Baseline)":
            df["predicted_sentiment"] = df["clean_text"].apply(lambda x: predict_svm(x, svm_model))
        else:
            preds = []
            for txt in df["text"]:
                pred = predict_transformer(txt, tokenizer, transformer_model)
                preds.append(pred)
            df["predicted_sentiment"] = preds

    # Filter by sentiment
    filtered_df = df[df["predicted_sentiment"].isin(sentiment_filter)].copy()

    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "💬 Tweets", "🧪 Manual Test", "ℹ️ Model Info"]
    )

    with tab1:
        show_overview(filtered_df, model_choice)

    with tab2:
        show_tweet_table(filtered_df, model_choice)

    with tab3:
        show_manual_tool(model_choice, svm_model, tokenizer, transformer_model)

    with tab4:
        show_model_info()


if __name__ == "__main__":
    main()
