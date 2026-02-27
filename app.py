import streamlit as st
import praw
import feedparser
from textblob import TextBlob
from streamlit_autorefresh import st_autorefresh

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Nebula AI Monitor",
    layout="wide"
)

st.title("🛰️ Nebula — Live Sentiment & Disinformation Tracker")

# Auto refresh every 15 seconds
st_autorefresh(interval=15000, key="refresh")

# ---------------- SENTIMENT ----------------
def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return "🟢 Positive"
    elif score < -0.1:
        return "🔴 Negative"
    return "🟡 Neutral"

# ---------------- REDDIT CONNECTION ----------------
reddit = praw.Reddit(
    client_id=st.secrets["REDDIT_CLIENT_ID"],
    client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
    user_agent=st.secrets["REDDIT_USER_AGENT"]
)

def get_reddit_posts(keyword):
    posts = []

    try:
        for submission in reddit.subreddit("all").search(
                keyword,
                sort="new",
                limit=15):

            text = submission.title + " " + submission.selftext
            posts.append(text)

    except Exception:
        st.warning("Reddit fetch error")

    return posts

# ---------------- BBC NEWS ----------------
def get_bbc_news(keyword):

    url = "http://feeds.bbci.co.uk/news/rss.xml"
    feed = feedparser.parse(url)

    news = []

    for entry in feed.entries:
        if keyword.lower() in entry.title.lower():
            news.append(entry.title)

    return news

# ---------------- INPUT ----------------
topic = st.text_input(
    "🔎 Enter Company / Brand / Topic",
    placeholder="Example: Tesla, Apple, AI, Elections"
)

# ---------------- DASHBOARD ----------------
if topic:

    col1, col2 = st.columns(2)

    # -------- REDDIT --------
    with col1:
        st.subheader("💬 Live Reddit Discussions")

        reddit_posts = get_reddit_posts(topic)

        if reddit_posts:
            for post in reddit_posts:
                sentiment = analyze_sentiment(post)

                st.markdown(
                    f"""
                    **{sentiment}**
                    
                    {post}
                    ---
                    """
                )
        else:
            st.info("No Reddit discussions found.")

    # -------- BBC NEWS --------
    with col2:
        st.subheader("📰 BBC Live News")

        news_posts = get_bbc_news(topic)

        if news_posts:
            for news in news_posts:
                sentiment = analyze_sentiment(news)

                st.markdown(
                    f"""
                    **{sentiment}**
                    
                    {news}
                    ---
                    """
                )
        else:
            st.info("No BBC news found.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "Nebula AI • Real-time Reddit + BBC Monitoring • Sentiment Intelligence Dashboard"
)
