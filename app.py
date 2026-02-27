# ✅ Nebula v1 Stable Version
import streamlit as st
from textblob import TextBlob
import random

# ---------- PAGE ----------
st.set_page_config(
    page_title="Nebula — Sentiment & Disinformation Tracker",
    page_icon="🛰️",
    layout="wide"
)

# ---------- CSS ----------
st.markdown("""
<style>
/* Hide Streamlit footer/menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

:root{
  --bg: #0b0f1a;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --green: #2dd4bf;
  --red: #fb7185;
  --amber:#fbbf24;
  --blue:#60a5fa;
}

.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, rgba(96,165,250,0.18), transparent 60%),
              radial-gradient(900px 500px at 80% 20%, rgba(45,212,191,0.14), transparent 60%),
              radial-gradient(700px 500px at 50% 90%, rgba(251,113,133,0.12), transparent 60%),
              var(--bg);
  color: var(--text);
}

.block-container {padding-top: 2rem; padding-bottom: 2rem;}

.hero {
  padding: 1.4rem 1.6rem;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
  border: 1px solid var(--border);
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

.badge {
  display: inline-block;
  font-size: 0.85rem;
  padding: .25rem .6rem;
  border-radius: 999px;
  background: rgba(96,165,250,0.14);
  border: 1px solid rgba(96,165,250,0.25);
  color: var(--muted);
}

.card {
  padding: 1.1rem 1.1rem;
  border-radius: 18px;
  background: var(--card);
  border: 1px solid var(--border);
  box-shadow: 0 10px 28px rgba(0,0,0,0.25);
}

.card strong {color: var(--muted); font-weight: 600;}
.big {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-top: 0.2rem;
}

.subtle {color: var(--muted);}

.alert-good{
  padding: 0.9rem 1rem;
  border-radius: 16px;
  background: rgba(45,212,191,0.12);
  border: 1px solid rgba(45,212,191,0.22);
}

.alert-warn{
  padding: 0.9rem 1rem;
  border-radius: 16px;
  background: rgba(251,191,36,0.12);
  border: 1px solid rgba(251,191,36,0.22);
}

.alert-bad{
  padding: 0.9rem 1rem;
  border-radius: 16px;
  background: rgba(251,113,133,0.12);
  border: 1px solid rgba(251,113,133,0.22);
}

.divider {height: 1px; background: rgba(255,255,255,0.08); margin: 1.0rem 0;}

.smallpill{
  display:inline-block;
  padding: .25rem .6rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  color: var(--muted);
  font-size: .82rem;
  margin-right: .4rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
positive_templates = [
    "Love the new update from {t} ✅",
    "{t} is improving a lot lately 🔥",
    "Great experience with {t} today 💯",
    "{t} support was helpful 👍",
]
negative_templates = [
    "Worst experience ever with {t} 😤",
    "{t} service quality is terrible",
    "I regret buying {t} 😑",
    "{t} is a scam?? not sure 👀",
]
neutral_templates = [
    "Mixed thoughts about {t}…",
    "Not sure how I feel about {t}.",
    "Seeing both good and bad about {t}.",
]

def analyze_sentiment(texts):
    pos, neu, neg = 0, 0, 0
    for s in texts:
        p = TextBlob(s).sentiment.polarity
        if p > 0.15:
            pos += 1
        elif p < -0.15:
            neg += 1
        else:
            neu += 1
    total = max(1, len(texts))
    return round(pos*100/total), round(neu*100/total), round(neg*100/total)

def risk_score(neg_pct, topic_len):
    base = int(neg_pct * 0.9)
    noise = random.randint(-6, 8)
    topic_bonus = 4 if topic_len > 10 else 0
    return max(0, min(100, base + noise + topic_bonus))

def alert_text(score):
    if score <= 33:
        return "Normal discussion activity ✅", "good"
    elif score <= 66:
        return "Elevated risk — monitor keywords & spikes ⚠️", "warn"
    return "High risk — potential coordinated narrative 🚨", "bad"

# ---------- UI ----------
st.markdown("""
<div class="hero">
  <div class="badge">AI-powered social monitoring (demo)</div>
  <h1 style="margin:0.6rem 0 0.2rem 0; font-size: 3rem; line-height:1.05;">
    🛰️ Nebula — Sentiment & Disinformation Tracker
  </h1>
  <div class="subtle" style="font-size:1.05rem;">
    Track public sentiment, detect narrative risk, and preview a live feed (mock stream).
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    topic = st.text_input("🔍 Enter Brand / Person / Topic", value="iPhone").strip()
    if not topic:
        topic = "Topic"

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Create dynamic mock feed
    random.seed(len(topic))
    sample = []
    for _ in range(3):
        sample.append(random.choice(positive_templates).format(t=topic))
    for _ in range(2):
        sample.append(random.choice(neutral_templates).format(t=topic))
    for _ in range(3):
        sample.append(random.choice(negative_templates).format(t=topic))
    random.shuffle(sample)

    pos_pct, neu_pct, neg_pct = analyze_sentiment(sample)
    score = risk_score(neg_pct, len(topic))
    alert_msg, level = alert_text(score)

    # METRICS ROW
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(f"<div class='card'><strong>Positive</strong><div class='big'>{pos_pct}%</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><strong>Neutral</strong><div class='big'>{neu_pct}%</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><strong>Negative</strong><div class='big'>{neg_pct}%</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><strong>Risk Score</strong><div class='big'>{score}/100</div></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🚨 Alerts")

    if level == "good":
        st.markdown(f"<div class='alert-good'>{alert_msg}</div>", unsafe_allow_html=True)
    elif level == "warn":
        st.markdown(f"<div class='alert-warn'>{alert_msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert-bad'>{alert_msg}</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🧾 Live Feed (mock)")
    for line in sample[:8]:
        tag = "Positive" if TextBlob(line).sentiment.polarity > 0.15 else ("Negative" if TextBlob(line).sentiment.polarity < -0.15 else "Neutral")
        st.markdown(f"**{tag} →** {line}")

with right:
    st.markdown("### 📊 Trend (demo)")
    st.markdown("<span class='smallpill'>Last 30 mins</span><span class='smallpill'>Mock stream</span>", unsafe_allow_html=True)

    # Fake trend data (but looks good)
    random.seed(100 + len(topic))
    points = [max(0, min(100, score + random.randint(-18, 18))) for _ in range(18)]
    st.line_chart(points)

    st.write("")
    st.markdown("### 🧠 Insights (demo)")
    st.markdown("<div class='card'>"
                "<div class='subtle'>Top talking points</div>"
                f"<ul style='margin: 0.6rem 0 0 1.0rem; color: rgba(255,255,255,0.85);'>"
                f"<li>{topic} update / experience</li>"
                f"<li>Service & support</li>"
                f"<li>Price / value</li>"
                f"</ul>"
                "</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🧩 Detection knobs (demo)")
    with st.expander("Advanced (optional)", expanded=False):
        st.slider("Spike sensitivity", 1, 10, 6)
        st.multiselect("Platforms", ["X/Twitter", "Reddit", "YouTube", "News"], default=["X/Twitter", "Reddit"])
        st.checkbox("Auto-refresh feed", value=True)
