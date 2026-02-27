import streamlit as st
import random

# Page settings
st.set_page_config(page_title="Nebula AI", page_icon="🌌", layout="wide")

# Simple UI styling
st.markdown("""
<style>
.stApp {background: #0b1220;}
.big-title {font-size:50px; font-weight:800; color:white;}
.sub {color:#aab3c5; font-size:16px;}
.card {
background: rgba(255,255,255,0.05);
border: 1px solid rgba(255,255,255,0.10);
border-radius: 18px;
padding: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title section
st.markdown('<div class="big-title">Nebula — Sentiment & Disinformation Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Track sentiment • Detect narrative risk • Live feed (demo)</div><br>', unsafe_allow_html=True)

# Input
topic = st.text_input("🔎 Enter Brand / Person / Topic", placeholder="Example: iPhone, Tesla, Bitcoin...")

# Dummy demo values (changes each time)
pos = random.randint(10, 70)
neg = random.randint(10, 70)
neu = 100 - (pos + neg)
if neu < 0:
    neu = 0
risk = min(100, neg + random.randint(5, 20))

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card"><h4>✅ Positive</h4><h2>{pos}%</h2></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><h4>😐 Neutral</h4><h2>{neu}%</h2></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card"><h4>❌ Negative</h4><h2>{neg}%</h2></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="card"><h4>⚠ Risk Score</h4><h2>{risk}/100</h2></div>', unsafe_allow_html=True)

st.write("")
# Alerts
st.subheader("🚨 Alerts")
if risk > 60:
    st.error("High risk discussion detected (demo)")
else:
    st.success("Normal discussion activity (demo)")

# Live feed
st.subheader("📰 Live Feed (demo)")
st.write(f"Positive → People are loving {topic or 'this'}")
st.write(f"Neutral → Some mixed opinions about {topic or 'this'}")
st.write(f"Negative → Complaints increasing about {topic or 'this'}")
