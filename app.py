import streamlit as st
from textblob import TextBlob
import random
import time

# =========================
# PAGE CONFIG (ONLY ONCE!)
# =========================
st.set_page_config(
    page_title="Nebula — Sentiment & Disinformation Tracker",
    page_icon="🛰️",
    layout="wide",
)

# =========================
# STYLES (UI polish)
# =========================
st.markdown(
    """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
  background: radial-gradient(1200px 800px at 20% 10%, rgba(120, 170, 255, 0.10), transparent 55%),
              radial-gradient(900px 600px at 70% 30%, rgba(200, 100, 255, 0.10), transparent 60%),
              radial-gradient(900px 700px at 40% 90%, rgba(50, 220, 180, 0.10), transparent 55%),
              #0b0f1a;
  color: rgba(255,255,255,0.92);
}

/* Container cards */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.hero {
  background: linear-gradient(120deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 22px;
  padding: 22px 24px;
  margin-bottom: 16px;
}

.kpi-wrap {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 14px;
}

.kpi {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 14px 16px;
}

.kpi .label {
  font-size: 13px;
  opacity: 0.8;
  margin-bottom: 6px;
}

.kpi .value {
  font-size: 34px;
  font-weight: 800;
  letter-spacing: -0.5px;
}

.smallpill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.10);
  font-size: 12px;
  margin-right: 8px;
  opacity: 0.9;
}

.alert-ok {
  background: rgba(0, 200, 120, 0.15);
  border: 1px solid rgba(0, 200, 120, 0.25);
  border-radius: 14px;
  padding: 12px 14px;
}

.alert-warn {
  background: rgba(255, 170, 0, 0.15);
  border: 1px solid rgba(255, 170, 0, 0.25);
  border-radius: 14px;
  padding: 12px 14px;
}

.alert-bad {
  background: rgba(255, 70, 70, 0.15);
  border: 1px solid rgba(255, 70, 70, 0.25);
  border-radius: 14px;
  padding: 12px 14px;
}

hr {
  border: none;
  height: 1px;
  background: rgba(255,255,255,0.10);
  margin: 14px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def safe_sentiment(text: str) -> float:
    """Returns polarity in [-1, 1]."""
    text = (text or "").strip()
    if not text:
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def label_from_polarity(p: float) -> str:
    if p > 0.10:
        return "Positive"
    if p < -0.10:
        return "Negative"
    return "Neutral"


def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def risk_score(neg_pct: int, spike_sensitivity: int) -> int:
    # Simple heuristic (demo)
    base = neg_pct * 1.4
    spike_boost = (spike_sensitivity - 5) * 3
    return int(clamp(base + spike_boost, 0, 100))


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🛰️ Nebula Controls")
    mode = st.selectbox("Mode", ["Demo (Mock Stream)", "Manual Input"])
    auto_refresh = st.toggle("Auto-refresh feed", value=True)

    st.markdown("---")
    st.markdown("### 🧩 Detection knobs (demo)")
    spike_sensitivity = st.slider("Spike sensitivity", 1, 10, 6)
    platforms = st.multiselect(
        "Platforms",
        ["X/Twitter", "Reddit", "YouTube", "News", "Instagram"],
        default=["X/Twitter", "Reddit"],
    )

    st.markdown("---")
    st.caption("Tip: If you change UI later, commit again with a new message ✅")


# =========================
# HEADER / HERO
# =========================
st.markdown(
    """
<div class="hero">
  <div class="smallpill">AI-powered social monitoring (demo)</div>
  <h1 style="margin: 10px 0 6px 0;">Nebula — Sentiment & Disinformation Tracker</h1>
  <div style="opacity:0.85; font-size:15px;">
    Track public sentiment, detect narrative risk, and preview a live feed (mock stream).
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# TOP INPUT
# =========================
topic = st.text_input("🔎 Enter Brand / Person / Topic", placeholder="e.g., iPhone, H&M, Elections...")
topic = topic.strip() if topic else ""

# =========================
# DATA (Demo vs Manual)
# =========================
demo_pool = [
    "Great experience with {t} today 💯",
    "Love the new update from {t} ✅",
    "{t} support was helpful 👍",
    "Mixed thoughts about {t}...",
    "Not sure how I feel about {t} 😐",
    "Service quality is terrible",
    "Worst experience ever",
    "{t} is a scam?? not sure 👀",
    "Price is too high for what you get",
    "Could be better, honestly",
]

manual_text = ""
if mode == "Manual Input":
    manual_text = st.text_area(
        "✍️ Paste text to analyze (one post / review / tweet)",
        height=120,
        placeholder="Paste any message here…",
    )

# generate feed
def make_feed(t: str):
    t = t if t else "this topic"
    feed = []
    random.seed(100 + len(t))
    for _ in range(9):
        msg = random.choice(demo_pool).format(t=t)
        feed.append(msg)
    return feed

feed = make_feed(topic)

# If manual input is present, inject it at the top
if mode == "Manual Input" and manual_text.strip():
    feed = [manual_text.strip()] + feed[:8]

# sentiments
polarities = [safe_sentiment(x) for x in feed]
labels = [label_from_polarity(p) for p in polarities]

pos = sum(1 for x in labels if x == "Positive")
neu = sum(1 for x in labels if x == "Neutral")
neg = sum(1 for x in labels if x == "Negative")
total = max(1, len(labels))

pos_pct = int(round(pos * 100 / total))
neu_pct = int(round(neu * 100 / total))
neg_pct = int(round(neg * 100 / total))
risk = risk_score(neg_pct, spike_sensitivity)

# =========================
# LAYOUT (KPIs + Trend)
# =========================
left, right = st.columns([1.25, 1])

with left:
    st.markdown(
        f"""
<div class="kpi-wrap">
  <div class="kpi"><div class="label">Positive</div><div class="value">{pos_pct}%</div></div>
  <div class="kpi"><div class="label">Neutral</div><div class="value">{neu_pct}%</div></div>
  <div class="kpi"><div class="label">Negative</div><div class="value">{neg_pct}%</div></div>
  <div class="kpi"><div class="label">Risk Score</div><div class="value">{risk}/100</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("## 🚨 Alerts")
    if risk < 35:
        st.markdown('<div class="alert-ok">Normal discussion activity ✅</div>', unsafe_allow_html=True)
    elif risk < 70:
        st.markdown(
            '<div class="alert-warn">Elevated negativity detected ⚠️ (monitor closely)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-bad">High risk detected 🔥 Possible coordinated narrative / spike</div>',
            unsafe_allow_html=True,
        )

    st.markdown("## 🧾 Live Feed (mock)")
    for lab, msg in zip(labels, feed):
        st.markdown(f"**{lab} →** {msg}")

with right:
    st.markdown("## 📊 Trend (demo)")
    st.markdown(
        f'<span class="smallpill">Last 30 mins</span><span class="smallpill">{mode}</span>',
        unsafe_allow_html=True,
    )

    # Fake trend data (but stable)
    random.seed(42 + len(topic))
    points = []
    score = risk
    for _ in range(18):
        score = clamp(score + random.randint(-18, 18), 0, 100)
        points.append(score)
    st.line_chart(points)

    st.markdown("## 🧠 Insights (demo)")
    t = topic if topic else "Topic"
    st.markdown(
        f"""
<div class="card">
  <div style="opacity:0.85; font-weight:700;">Top talking points</div>
  <ul style="margin: 0.6rem 0 0 1.0rem; opacity:0.9;">
    <li>{t} update / experience</li>
    <li>Service & support</li>
    <li>Price / value</li>
  </ul>
  <div style="opacity:0.6; font-size:12px; margin-top:10px;">
    Platforms: {", ".join(platforms) if platforms else "None selected"}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# OPTIONAL: AUTO REFRESH
# =========================
if mode == "Demo (Mock Stream)" and auto_refresh:
    # lightweight auto-refresh effect
    time.sleep(0.4)
    st.rerun()
