import streamlit as st
from textblob import TextBlob
import random
import pandas as pd

# ---------------- PAGE CONFIG (ONLY ONCE!) ----------------
st.set_page_config(
    page_title="Nebula — Sentiment & Disinformation Tracker",
    page_icon="🛰️",
    layout="wide",
)

# ---------------- CSS (Better UI) ----------------
st.markdown(
    """
<style>
/* Hide Streamlit footer/menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

:root{
  --bg1: #0b0f1a;
  --bg2: #0a1a1a;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --accent: #66d9ff;
  --good: rgba(46, 204, 113, 0.20);
  --warn: rgba(241, 196, 15, 0.20);
  --bad:  rgba(231, 76, 60, 0.20);
}

.stApp{
  background:
    radial-gradient(1200px 500px at 15% 10%, rgba(102,217,255,0.22), transparent 60%),
    radial-gradient(900px 500px at 70% 20%, rgba(155,89,182,0.18), transparent 60%),
    radial-gradient(900px 600px at 30% 90%, rgba(46,204,113,0.12), transparent 60%),
    linear-gradient(160deg, var(--bg1), var(--bg2));
  color: var(--text);
}

.big-title{
  font-size: 3rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0.2rem 0 0.2rem 0;
}

.subtitle{
  color: var(--muted);
  margin-top: -0.2rem;
  margin-bottom: 1.1rem;
}

.hero{
  padding: 1.2rem 1.2rem;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: linear-gradient(120deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  margin-bottom: 1rem;
}

.card{
  padding: 1rem 1rem;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--card);
  box-shadow: 0 8px 20px rgba(0,0,0,0.20);
}

.metric-card{
  padding: 1.0rem 1.0rem;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  box-shadow: 0 10px 26px rgba(0,0,0,0.22);
  height: 100%;
}

.metric-label{
  color: var(--muted);
  font-weight: 600;
  font-size: 0.95rem;
}
.metric-value{
  font-size: 2.1rem;
  font-weight: 800;
  margin-top: 0.35rem;
}

.pill{
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.85);
  font-size: 0.8rem;
  margin-right: 0.35rem;
}

.smallpill{
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.82);
  font-size: 0.78rem;
  margin-right: 0.35rem;
}

.feed-item{
  padding: 0.75rem 0.85rem;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  margin-bottom: 0.55rem;
}

.feed-label{
  font-weight: 800;
  margin-right: 0.4rem;
}
.feed-pos{ color: #67ff9a; }
.feed-neu{ color: #aab2bd; }
.feed-neg{ color: #ff6b6b; }

.alert-good{
  padding: 0.85rem 1.0rem;
  border-radius: 16px;
  border: 1px solid rgba(46,204,113,0.35);
  background: rgba(46,204,113,0.18);
}

.alert-warn{
  padding: 0.85rem 1.0rem;
  border-radius: 16px;
  border: 1px solid rgba(241,196,15,0.35);
  background: rgba(241,196,15,0.18);
}

.alert-bad{
  padding: 0.85rem 1.0rem;
  border-radius: 16px;
  border: 1px solid rgba(231,76,60,0.35);
  background: rgba(231,76,60,0.18);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("## 🛰️ Nebula Controls")

    mode = st.selectbox(
        "Mode",
        ["Demo (Mock Stream)", "Manual Input"],
        index=0,
    )

    auto_refresh = st.toggle("Auto-refresh feed (demo)", value=True)

    st.markdown("---")

    st.markdown("### 📄 CSV Upload (optional)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    csv_col = st.text_input("Column name that has text", value="text")

    st.markdown("---")
    st.caption("Tip: If CSV is uploaded, it will be used first (highest priority).")

# ---------------- HEADER / HERO ----------------
st.markdown(
    """
<div class="hero">
  <div class="pill">AI-powered social monitoring (demo)</div>
  <div class="big-title">Nebula — Sentiment & Disinformation Tracker</div>
  <div class="subtitle">Track public sentiment, detect narrative risk, and preview a live feed (mock stream).</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- INPUT: TOPIC ----------------
topic = st.text_input("🔎 Enter Brand / Person / Topic", value="", placeholder="Example: iPhone, Tesla, KLU, Elections...")

# ---------------- BUILD FEED (CSV > MANUAL > DEMO) ----------------
feed = []

# CSV mode (highest priority)
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if csv_col not in df_in.columns:
            st.error(f"Column '{csv_col}' not found. Available columns: {list(df_in.columns)}")
        else:
            feed = df_in[csv_col].astype(str).dropna().tolist()
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# Manual mode
if (not feed) and mode == "Manual Input":
    raw = st.text_area(
        "✍️ Paste posts (one post per line)",
        height=160,
        placeholder="Example:\nI love this product\nWorst service ever\nNot bad but expensive",
    )
    feed = [line.strip() for line in raw.splitlines() if line.strip()]

# Demo mode (fallback)
if (not feed) and mode == "Demo (Mock Stream)":
    t = topic.strip() if topic.strip() else "this topic"
    feed = [
        f"{t} is amazing!",
        f"Mixed thoughts about {t}...",
        f"{t} service quality is terrible",
        f"I love the new update from {t}",
        f"Worst experience ever with {t}",
        f"{t} support was helpful 👍",
        f"I regret buying {t} 😬",
        f"{t} is a scam?? not sure 👀",
    ]

# ---------------- SENTIMENT ANALYSIS ----------------
def classify_sentiment(text: str):
    blob = TextBlob(text)
    pol = float(blob.sentiment.polarity)
    if pol > 0.1:
        return "Positive", pol
    elif pol < -0.1:
        return "Negative", pol
    else:
        return "Neutral", pol

rows = []
pos = neu = neg = 0

for txt in feed:
    label, pol = classify_sentiment(txt)
    if label == "Positive":
        pos += 1
    elif label == "Negative":
        neg += 1
    else:
        neu += 1
    rows.append({"text": txt, "polarity": pol, "sentiment": label})

total = max(len(feed), 1)
pos_pct = round((pos / total) * 100)
neu_pct = round((neu / total) * 100)
neg_pct = round((neg / total) * 100)

# Simple risk score (demo formula)
# More negative => higher risk, plus small bump if many posts
risk_score = int(min(100, (neg_pct * 1.2) + (len(feed) * 0.6)))

# ---------------- TOP METRICS + RIGHT PANEL LAYOUT ----------------
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    # Metrics row
    m1, m2, m3, m4 = st.columns(4, gap="medium")

    with m1:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Positive</div>
  <div class="metric-value">{pos_pct}%</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Neutral</div>
  <div class="metric-value">{neu_pct}%</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Negative</div>
  <div class="metric-value">{neg_pct}%</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-label">Risk Score</div>
  <div class="metric-value">{risk_score}/100</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")

    # Alerts
    st.markdown("## 🚨 Alerts")
    if risk_score < 35:
        st.markdown('<div class="alert-good">Normal discussion activity ✅</div>', unsafe_allow_html=True)
    elif risk_score < 65:
        st.markdown('<div class="alert-warn">Elevated negativity detected ⚠️</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-bad">High risk: potential narrative spike / coordinated negativity 🔥</div>', unsafe_allow_html=True)

    st.write("")

    # Live feed
    st.markdown("## 🧾 Live Feed (mock)")
    for item in rows[:12]:
        s = item["sentiment"]
        if s == "Positive":
            cls = "feed-pos"
        elif s == "Negative":
            cls = "feed-neg"
        else:
            cls = "feed-neu"

        st.markdown(
            f"""
<div class="feed-item">
  <span class="feed-label {cls}">{s} →</span>
  <span>{item["text"]}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    # Download report
    if rows:
        df_out = pd.DataFrame(rows)
        st.markdown("### ⬇️ Download report")
        st.download_button(
            "Download CSV report",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="nebula_report.csv",
            mime="text/csv",
        )

with right:
    st.markdown("## 📊 Trend (demo)")
    st.markdown('<span class="smallpill">Last 30 mins</span><span class="smallpill">Mock stream</span>', unsafe_allow_html=True)

    # Fake trend points
    seed_key = (topic.strip() if topic.strip() else "nebula") + str(len(feed))
    random.seed(seed_key)
    base = max(10, 50 - neg_pct)
    points = [max(0, min(100, base + random.randint(-18, 18))) for _ in range(18)]
    st.line_chart(points)

    st.write("")
    st.markdown("## 🧠 Insights (demo)")
    tname = topic.strip() if topic.strip() else "Topic"
    st.markdown(
        f"""
<div class="card">
  <div style="color: rgba(255,255,255,0.80); font-weight:800; margin-bottom:0.4rem;">Top talking points</div>
  <ul style="margin: 0.2rem 0 0 1.0rem; color: rgba(255,255,255,0.85);">
    <li>{tname} update / experience</li>
    <li>Service & support</li>
    <li>Price / value</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("## 🧩 Detection knobs (demo)")
    with st.expander("Advanced (optional)", expanded=False):
        spike = st.slider("Spike sensitivity", 1, 10, 6)
        platforms = st.multiselect(
            "Platforms",
            ["X/Twitter", "Reddit", "YouTube", "News"],
            default=["X/Twitter", "Reddit"],
        )
        st.checkbox("Auto-refresh feed", value=auto_refresh)

    if mode == "Demo (Mock Stream)" and auto_refresh:
        st.caption("Auto-refresh is ON (demo). Refresh the page to see small changes.")

# Small footer note
st.caption("Nebula v1 demo — sentiment is calculated using TextBlob polarity.")
