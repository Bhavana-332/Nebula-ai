import streamlit as st
from textblob import TextBlob

st.set_page_config(page_title="Nebula AI", layout="wide")

st.title("🌌 Nebula — Sentiment & Disinformation Tracker")
st.caption("AI-powered social monitoring")

topic = st.text_input("🔍 Enter Brand / Person / Topic")

posts = [
    "This brand is amazing!",
    "Service quality is terrible",
    "I love their new update",
    "Worst experience ever",
    "Not bad but could improve"
]

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

if topic:

    positive = 0
    negative = 0
    neutral = 0

    results = []

    for post in posts:
        sentiment = analyze_sentiment(post)
        results.append((post, sentiment))

        if sentiment == "Positive":
            positive += 1
        elif sentiment == "Negative":
            negative += 1
        else:
            neutral += 1

    total = len(posts)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Positive", f"{int((positive/total)*100)}%")
    col2.metric("Neutral", f"{int((neutral/total)*100)}%")
    col3.metric("Negative", f"{int((negative/total)*100)}%")

    risk_score = int((negative/total)*100)
    col4.metric("Risk Score", f"{risk_score}/100")

    st.subheader("🚨 Alerts")

    if risk_score > 40:
        st.error("Potential coordinated negative campaign detected")
    else:
        st.success("Normal discussion activity")

    st.subheader("🧾 Live Feed")

    for post, sentiment in results:
        st.write(f"**{sentiment}** → {post}")