import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
import streamlit as st
import pandas as pd
import joblib
import torch
import random
import datetime
from collections import Counter

import plotly.express as px

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============
# STYLING (CSS)
# ==============

def add_custom_css():
    st.markdown(
        """
        <style>
        /* Page padding */
        .main > div {
            padding-top: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
            padding-bottom: 3rem;
        }

        /* Hero banner */
        .hero {
            background: linear-gradient(90deg, #1D4ED8, #1E40AF);
            padding: 24px 32px;
            border-radius: 18px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.35);
        }
        .hero h1 {
            font-size: 30px;
            margin-bottom: 0.2rem;
        }
        .hero p {
            font-size: 14px;
            opacity: 0.95;
            margin-bottom: 0;
        }

        .section-title {
            font-size: 20px;
            font-weight: 600;
            margin-top: 10px;
            margin-bottom: 5px;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border-radius: 14px;
            padding: 12px 16px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15);
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

        /* Dataframe container */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Live tag */
        .live-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            background: #DC2626;
            color: white;
            font-weight: 600;
            font-size: 11px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .tweet-card {
            padding: 10px 14px;
            border-radius: 12px;
            margin-bottom: 8px;
            background: #FFFFFF;
            box-shadow: 0 1px 5px rgba(15, 23, 42, 0.1);
        }

        .tweet-meta {
            font-size: 11px;
            color: #6B7280;
        }

        .sent-pill {
            display:inline-block;
            padding:2px 8px;
            border-radius:999px;
            font-size:11px;
            font-weight:600;
        }
        .sent-negative {
            background:#FEE2E2;
            color:#B91C1C;
        }
        .sent-neutral {
            background:#E5E7EB;
            color:#374151;
        }
        .sent-positive {
            background:#DCFCE7;
            color:#166534;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============
# CACHED LOADERS
# ==============

@st.cache_resource
def load_svm_model(path="svm_pipeline.pkl"):
    return joblib.load(path)

@st.cache_resource
def load_transformer_model(model_dir="./bert_lagos_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

@st.cache_data
def load_dataset(path="synthetic_lagos_floods.csv"):
    df = pd.read_csv(path)

    # Ensure clean_text
    if "clean_text" not in df.columns:
        df["clean_text"] = df["text"]

    # Add a fake timestamp if none exists (for "time trend" and "live" simulation)
    if "timestamp" not in df.columns:
        now = datetime.datetime.now()
        # spread timestamps over last 6 hours
        times = [
            now - datetime.timedelta(minutes=i)
            for i in range(len(df))
        ]
        times.reverse()
        df["timestamp"] = times

    return df


LABELS = ["negative", "neutral", "positive"]
DISTRESS_KEYWORDS = [
    "help", "trapped", "drowning", "die", "emergency", "stuck",
    "no rescue", "need help", "house don flood", "water don full",
]


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


# ==============
# HELPER ANALYTICS
# ==============

def compute_severity_score(df):
    """Compute a simple crisis severity score (0–100)."""
    total = len(df)
    if total == 0:
        return 0, "Stable"

    # Negative sentiment proportion
    neg_count = (df["predicted_sentiment"] == "negative").sum()
    neg_pct = neg_count / total

    # Distress keyword hits
    text_concat = " ".join(df["text"].str.lower().tolist())
    distress_hits = sum(text_concat.count(k) for k in DISTRESS_KEYWORDS)

    # Simple scoring heuristic
    score = (neg_pct * 60) + min(distress_hits * 2, 40)  # cap the keyword contribution
    score = min(score, 100)

    if score >= 75:
        level = "Severe"
    elif score >= 55:
        level = "High"
    elif score >= 35:
        level = "Moderate"
    else:
        level = "Stable"

    return int(score), level


def keyword_frequency(df, top_n=15):
    """Return top keywords based on simple token split on whitespace."""
    all_tokens = " ".join(df["text"].str.lower().tolist()).split()
    # very naive cleaning
    tokens = [t.strip(".,!?;:()\"'") for t in all_tokens if len(t) > 3]
    counter = Counter(tokens)
    return counter.most_common(top_n)


def simple_topics_from_keywords(freqs, n_topics=4, per_topic=3):
    """Very simple 'topic-style' groups just for visual grouping in the UI."""
    topics = []
    for i in range(n_topics):
        chunk = freqs[i * per_topic : (i + 1) * per_topic]
        if chunk:
            topics.append(chunk)
    return topics


# ==============
# PAGE SECTIONS
# ==============

def show_overview(df, model_choice):
    st.markdown('<div class="section-title">📊 Situation Overview</div>', unsafe_allow_html=True)

    sentiment_counts = df["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0)
    total = len(df)

    if total > 0:
        neg_pct = sentiment_counts["negative"] / total * 100
        neu_pct = sentiment_counts["neutral"] / total * 100
        pos_pct = sentiment_counts["positive"] / total * 100
    else:
        neg_pct = neu_pct = pos_pct = 0.0

    severity_score, severity_level = compute_severity_score(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tweets", total)
    with col2:
        st.metric("Negative %", f"{neg_pct:.1f}%")
    with col3:
        st.metric("Neutral %", f"{neu_pct:.1f}%")
    with col4:
        st.metric("Positive %", f"{pos_pct:.1f}%")

    # Alert card
    if severity_level == "Severe":
        st.error(f"🚨 SEVERE CRISIS: Severity score {severity_score}/100. Immediate attention required.")
    elif severity_level == "High":
        st.warning(f"⚠ HIGH RISK: Severity score {severity_score}/100. Situation is deteriorating.")
    elif severity_level == "Moderate":
        st.info(f"ℹ MODERATE RISK: Severity score {severity_score}/100. Monitor closely.")
    else:
        st.success(f"✅ STABLE: Severity score {severity_score}/100. No critical spikes detected.")

    st.markdown('<div class="section-title">📈 Sentiment Distribution</div>', unsafe_allow_html=True)
    chart_df = pd.DataFrame(
        {"Sentiment": sentiment_counts.index, "Count": sentiment_counts.values}
    )
    fig = px.bar(
        chart_df,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={
            "negative": "#EF4444",
            "neutral": "#9CA3AF",
            "positive": "#22C55E",
        },
        text="Count",
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Number of Tweets",
        template="simple_white",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"This overview reflects sentiment based on **{model_choice}** for Lagos flood-related posts in the dataset."
    )


def show_live_feed(df):
    st.markdown(
        '<div class="section-title">🛰 Live Feed Simulation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="live-pill">LIVE</span> <span style="font-size:12px;color:#6B7280;">Simulated from synthetic dataset</span>',
        unsafe_allow_html=True,
    )

    # Random sample of recent tweets to simulate "live"
    df_sorted = df.sort_values("timestamp", ascending=False)
    sample = df_sorted.head(50)  # most recent 50
    # On each refresh, pick 5 random ones out of the recent 50
    live_sample = sample.sample(min(5, len(sample))) if len(sample) > 0 else pd.DataFrame()

    if live_sample.empty:
        st.info("No tweets available in the dataset.")
        return

    for _, row in live_sample.iterrows():
        sent = row["predicted_sentiment"]
        text = row["text"]
        ts = row["timestamp"]

        if isinstance(ts, str):
            ts_str = ts
        else:
            ts_str = ts.strftime("%Y-%m-%d %H:%M")

        if sent == "negative":
            sent_class = "sent-negative"
        elif sent == "neutral":
            sent_class = "sent-neutral"
        else:
            sent_class = "sent-positive"

        st.markdown(
            f"""
            <div class="tweet-card">
                <div class="tweet-meta">{ts_str}</div>
                <div style="margin-top:4px;margin-bottom:6px;">{text}</div>
                <span class="sent-pill {sent_class}">{sent.upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("The feed above randomly samples recent tweets to give a 'live' situational view.")


def show_analytics(df):
    st.markdown('<div class="section-title">📊 Analytics & Key Themes</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Keyword frequency
    with col1:
        st.markdown("**Top Keywords**")
        freqs = keyword_frequency(df, top_n=12)
        if freqs:
            freq_df = pd.DataFrame(freqs, columns=["Keyword", "Count"])
            fig_kw = px.bar(
                freq_df,
                x="Keyword",
                y="Count",
                text="Count",
                template="simple_white",
            )
            fig_kw.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.write("Not enough data to compute keywords.")

    # Simple "topics"
    with col2:
        st.markdown("**Simple Topic Groups (Heuristic)**")
        if freqs:
            topics = simple_topics_from_keywords(freqs, n_topics=4, per_topic=3)
            if not topics:
                st.write("Not enough keywords to derive topics.")
            else:
                for i, topic in enumerate(topics, start=1):
                    chips = " • ".join(f"{w} ({c})" for w, c in topic)
                    st.markdown(f"- **Topic {i}:** {chips}")
        else:
            st.write("No keywords available.")

    st.markdown("---")
    st.markdown("**Sentiment Over Time (Aggregated)**")

    # Time trend (very simple: group by hour or 30-min window)
    if "timestamp" in df.columns and len(df) > 0:
        time_df = df.copy()
        time_df["time_bucket"] = pd.to_datetime(time_df["timestamp"]).dt.floor("30min")
        trend = (
            time_df.groupby(["time_bucket", "predicted_sentiment"])["text"]
            .count()
            .reset_index()
            .rename(columns={"text": "Count"})
        )

        fig_trend = px.line(
            trend,
            x="time_bucket",
            y="Count",
            color="predicted_sentiment",
            markers=True,
            template="simple_white",
            color_discrete_map={
                "negative": "#EF4444",
                "neutral": "#9CA3AF",
                "positive": "#22C55E",
            },
        )
        fig_trend.update_layout(
            xaxis_title="Time",
            yaxis_title="Number of Tweets",
            legend_title="Sentiment",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.write("No timestamp information available to show trend.")


def show_tweet_table(df, model_choice):
    st.markdown(
        '<div class="section-title">💬 Tweets Explorer</div>',
        unsafe_allow_html=True,
    )
    st.write(
        f"Inspect sample tweets and their classified sentiment using **{model_choice}**."
    )

    st.dataframe(
        df[["timestamp", "text", "predicted_sentiment", "label"]]
        .rename(
            columns={
                "timestamp": "Time",
                "text": "Tweet",
                "predicted_sentiment": "Predicted Sentiment",
                "label": "Original Label (Synthetic)",
            }
        )
        .sort_values("Time", ascending=False)
        .head(200),
        use_container_width=True,
    )


def show_manual_tool(model_choice, svm_model, tokenizer, transformer_model):
    st.markdown(
        '<div class="section-title">🧪 Manual Tweet Classification</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Paste any tweet or short message related to flooding in Lagos. "
        "The system will classify it using the selected model."
    )

    user_text = st.text_area(
        "Enter tweet text:",
        height=120,
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
                    "This indicates a **negative sentiment**, suggesting distress, fear, or dissatisfaction. "
                    "In an operational setting, such posts may signal urgent needs or failures in response."
                )
            elif pred == "neutral":
                st.write(
                    "This indicates a **neutral sentiment**, often descriptive or informational. "
                    "These posts are useful for building situational awareness without strong emotional polarity."
                )
            else:
                st.write(
                    "This indicates a **positive sentiment**, often reflecting successful interventions or relief. "
                    "These can help crisis managers assess where conditions are improving."
                )


def show_model_info():
    st.markdown(
        '<div class="section-title">ℹ️ Model & System Information</div>',
        unsafe_allow_html=True,
    )
    st.write(
        """
        This prototype operationalises the proposed architecture from the dissertation:

        1. **Data Ingestion & Preprocessing**
           - Synthetic dataset representing Lagos flood-related tweets.
           - Includes Nigerian English and Pidgin expressions (e.g., 'water don full', 'house don flood').
           - Preprocessing includes tokenisation, basic normalisation, and cleaning.

        2. **Baseline Model – SVM with TF–IDF**
           - Classical machine learning pipeline.
           - Uses TF–IDF features over preprocessed text.
           - Serves as a transparent and lightweight baseline.

        3. **Advanced Model – Transformer (BERT)**
           - BERT-based sequence classification model fine-tuned on the synthetic Lagos flood corpus.
           - Captures richer contextual information and is more robust to informal crisis language.

        4. **Dashboard & Decision Support**
           - Real-time style feed of classified posts.
           - Sentiment distribution and severity scoring.
           - Keyword and simple topic analysis.
           - Manual classification tool for what-if analysis.

        In the dissertation, these components are linked to the Design Science Research (DSR) methodology,
        providing a concrete artefact that can be adapted by agencies such as NEMA and LASEMA.
        """
    )


def show_auto_report(df, model_choice):
    st.markdown(
        '<div class="section-title">📝 Auto-Generated Situation Report (Prototype)</div>',
        unsafe_allow_html=True,
    )

    total = len(df)
    sentiment_counts = df["predicted_sentiment"].value_counts().reindex(LABELS, fill_value=0)
    severity_score, severity_level = compute_severity_score(df)
    top_keywords = keyword_frequency(df, top_n=5)

    neg = sentiment_counts["negative"]
    neu = sentiment_counts["neutral"]
    pos = sentiment_counts["positive"]

    report_lines = []

    report_lines.append(
        f"As of the current analysis window, a total of **{total}** Lagos flood-related posts were "
        f"processed using the **{model_choice}** model."
    )
    report_lines.append(
        f"Of these, approximately **{neg}** ({neg/total*100:.1f}%) were classified as negative, "
        f"**{neu}** ({neu/total*100:.1f}%) as neutral, and **{pos}** ({pos/total*100:.1f}%) as positive."
        if total > 0
        else "No data available for this period."
    )
    report_lines.append(
        f"The computed **crisis severity score** is **{severity_score}/100**, corresponding to a "
        f"**{severity_level.upper()}** risk level."
    )

    if top_keywords:
        kws = ", ".join([w for w, _ in top_keywords[:5]])
        report_lines.append(
            f"Frequently occurring terms in the discourse include: {kws}, suggesting key concerns and themes "
            "in public reporting and perception."
        )

    report_lines.append(
        "From an operational perspective, these insights could support Nigerian emergency management agencies "
        "in prioritising areas for intervention, monitoring shifts in public sentiment, and assessing the "
        "impact of response activities over time."
    )

    report = "\n\n".join(report_lines)
    st.write(report)


# ==============
# MAIN APP
# ==============

def main():
    st.set_page_config(
        page_title="Lagos Flood Crisis Monitoring Dashboard",
        layout="wide",
    )

    add_custom_css()

    # Hero banner
    st.markdown(
        """
        <div class="hero">
            <h1>🛰 Lagos Flood Crisis Monitoring System</h1>
            <p>
                Real-time-style sentiment analysis of social media posts about flooding in Lagos, 
                designed to enhance situational awareness and decision-making for crisis managers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading models and dataset..."):
        df = load_dataset()
        svm_model = load_svm_model()
        tokenizer, transformer_model = load_transformer_model()

    # Sidebar controls
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
        "Prototype dashboard for research on real-time sentiment analysis in Nigerian crisis management."
    )

    # Apply selected model
    with st.spinner("Classifying dataset with selected model..."):
        if model_choice == "SVM (Baseline)":
            df["predicted_sentiment"] = df["clean_text"].apply(lambda x: predict_svm(x, svm_model))
        else:
            preds = []
            for txt in df["text"]:
                preds.append(predict_transformer(txt, tokenizer, transformer_model))
            df["predicted_sentiment"] = preds

    # Filter
    filtered_df = df[df["predicted_sentiment"].isin(sentiment_filter)].copy()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Overview",
            "🛰 Live Feed",
            "📈 Analytics",
            "💬 Tweets",
            "🧪 Manual Test",
            "ℹ️ Model & Report",
        ]
    )

    with tab1:
        show_overview(filtered_df, model_choice)

    with tab2:
        show_live_feed(filtered_df)

    with tab3:
        show_analytics(filtered_df)

    with tab4:
        show_tweet_table(filtered_df, model_choice)

    with tab5:
        show_manual_tool(model_choice, svm_model, tokenizer, transformer_model)

    with tab6:
        show_model_info()
        st.markdown("---")
        show_auto_report(filtered_df, model_choice)


if __name__ == "__main__":
    main()
