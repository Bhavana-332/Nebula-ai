import streamlit as st
from textblob import TextBlob
import random
import pandas as pd
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

# ---------------- PAGE CONFIG (ONLY ONCE!) ----------------
st.set_page_config(
    page_title="Nebula — Sentiment & Disinformation Tracker",
    page_icon="🛰️",
    layout="wide",
)

# ---------------- CSS (UI) ----------------
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

:root{
  --bg1: #0b0f1a;
  --bg2: #0a1a1a;
  --card: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --good: rgba(46,204,113,0.18);
  --warn: rgba(241,196,15,0.18);
  --bad:  rgba(231,76,60,0.18);
}

.stApp{
  background:
    radial-gradient(1200px 500px at 15% 10%, rgba(102,217,255,0.22), transparent 60%),
    radial-gradient(900px 500px at 70% 20%, rgba(155,89,182,0.18), transparent 60%),
    radial-gradient(900px 600px at 30% 90%, rgba(46,204,113,0.12), transparent 60%),
    linear-gradient(160deg, var(--bg1), var(--bg2));
  color: var(--text);
}

.hero{
  padding: 1.2rem;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: linear-gradient(120deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  margin-bottom: 1rem;
}

.big-title{font-size: 3rem; font-weight: 800; letter-spacing:-0.02em; margin:0.2rem 0;}
.subtitle{color: var(--muted); margin-top:-0.2rem; margin-bottom:0.2rem;}

.metric-card{
  padding: 1rem;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
  box-shadow: 0 10px 26px rgba(0,0,0,0.22);
  height: 100%;
}
.metric-label{color: var(--muted); font-weight: 600; font-size: 0.95rem;}
.metric-value{font-size: 2.1rem; font-weight: 800; margin-top: 0.35rem;}

.card{
  padding: 1rem;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--card);
  box-shadow: 0 8px 20px rgba(0,0,0,0.20);
}

.feed-item{
  padding: 0.75rem 0.85rem;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  margin-bottom: 0.55rem;
}

.feed-label{font-weight: 800; margin-right: 0.4rem;}
.feed-pos{ color: #67ff9a; }
.feed-neu{ color: #aab2bd; }
.feed-neg{ color: #ff6b6b; }

.alert-good{padding: 0.85rem 1rem; border-radius: 16px; border: 1px solid rgba(46,204,113,0.35); background: var(--good);}
.alert-warn{padding: 0.85rem 1rem; border-radius: 16px; border: 1px solid rgba(241,196,15,0.35); background: var(--warn);}
.alert-bad{ padding: 0.85rem 1rem; border-radius: 16px; border: 1px solid rgba(231,76,60,0.35); background: var(--bad);}

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

mark{
  padding: 0.08rem 0.28rem;
  border-radius: 8px;
  background: rgba(241,196,15,0.25);
  border: 1px solid rgba(241,196,15,0.35);
  color: rgba(255,255,255,0.95);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- NLP UTILITIES ----------------
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with","from",
    "is","am","are","was","were","be","been","being","this","that","these","those","it","its","i","you",
    "we","they","he","she","them","my","your","our","their","me","us","as","so","not","no","yes","very",
    "just","too","also","can","could","should","would","will","did","do","does","done","have","has","had"
}

RISKY_TERMS = [
    "fake","fraud","scam","boycott","rumor","rumour","misinfo","misinformation","propaganda",
    "hoax","exposed","leak","leaked","illegal","corrupt","bribe","threat","hate","toxic",
    "cancel","ban","conspiracy","bot","bots","troll","trolls","spam"
]

def tokenize(text: str):
    words = re.findall(r"[a-zA-Z0-9_#']+", (text or "").lower())
    words = [w.strip("'") for w in words if w.strip("'")]
    return words

def extract_keywords(texts, top_k=10):
    all_words = []
    for t in texts:
        for w in tokenize(t):
            if w in STOPWORDS or len(w) < 3 or w.isdigit():
                continue
            all_words.append(w)
    return Counter(all_words).most_common(top_k)

def risky_hits(text: str):
    t = (text or "").lower()
    hits = []
    for term in RISKY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", t):
            hits.append(term)
    return hits

def highlight_risky(text: str):
    out = text
    for term in sorted(RISKY_TERMS, key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(term)})\b", r"<mark>\1</mark>", out)
    return out

def classify_sentiment(text: str):
    pol = float(TextBlob(text).sentiment.polarity)
    if pol > 0.1:
        return "Positive", pol
    elif pol < -0.1:
        return "Negative", pol
    return "Neutral", pol

def normalize_text(text: str):
    t = (text or "").lower()
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[^a-z0-9#\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def jaccard(a_tokens, b_tokens):
    A = set(a_tokens)
    B = set(b_tokens)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("## 🛰️ Nebula Controls")
    mode = st.selectbox("Mode", ["Demo (Mock Stream)", "Manual Input"], index=0)
    auto_refresh = st.toggle("Auto-refresh feed (demo)", value=True)

    st.markdown("---")
    st.markdown("### 📄 CSV Upload (optional)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    csv_col = st.text_input("Text column name", value="text")
    user_col = st.text_input("User column (optional)", value="user")
    time_col = st.text_input("Time column (optional)", value="timestamp")
    platform_col = st.text_input("Platform column (optional)", value="platform")

    st.markdown("---")
    st.markdown("### 🔎 Feed Filter")
    sentiment_filter = st.selectbox("Show", ["All", "Positive", "Neutral", "Negative"], index=0)

    st.markdown("---")
    st.markdown("### 🧩 Detection knobs (demo)")
    spike_sensitivity = st.slider("Spike sensitivity", 1, 10, 6)
    similarity_threshold = st.slider("Coordination similarity threshold", 0.30, 0.90, 0.55, 0.05)
    min_cluster_size = st.slider("Minimum cluster size", 2, 10, 3)

# ---------------- HEADER ----------------
st.markdown(
    """
<div class="hero">
  <div class="pill">Nebula v4</div>
  <div class="pill">Bot-score + Coordination Clusters</div>
  <div class="big-title">Nebula — Sentiment & Disinformation Tracker</div>
  <div class="subtitle">Sentiment + risky terms + suspicious coordination (demo logic)</div>
</div>
""",
    unsafe_allow_html=True,
)

topic = st.text_input("🔎 Enter Brand / Person / Topic", value="", placeholder="Example: iPhone, Tesla, Elections...")

# ---------------- BUILD FEED (CSV > MANUAL > DEMO) ----------------
records = []

# CSV priority
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        if csv_col not in df_in.columns:
            st.error(f"Text column '{csv_col}' not found. Available: {list(df_in.columns)}")
        else:
            for _, row in df_in.iterrows():
                txt = str(row.get(csv_col, "")).strip()
                if not txt:
                    continue
                rec = {
                    "text": txt,
                    "user": str(row.get(user_col, "user_unknown")) if user_col in df_in.columns else "user_unknown",
                    "platform": str(row.get(platform_col, "unknown")) if platform_col in df_in.columns else "unknown",
                    "timestamp": str(row.get(time_col, "")) if time_col in df_in.columns else "",
                }
                records.append(rec)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# Manual priority
if (not records) and mode == "Manual Input":
    raw = st.text_area(
        "✍️ Paste posts (one post per line)",
        height=160,
        placeholder="Example:\nI love this product\nWorst service ever\nNot bad but expensive",
    )
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    now = datetime.now()
    for i, t in enumerate(lines):
        records.append({
            "text": t,
            "user": f"user_{100+i}",
            "platform": "manual",
            "timestamp": (now - timedelta(minutes=len(lines)-i)).isoformat(timespec="seconds")
        })

# Demo fallback
if (not records) and mode == "Demo (Mock Stream)":
    t = topic.strip() if topic.strip() else "this topic"
    demo_texts = [
        f"Love the new update from {t} ✅",
        f"{t} is improving a lot lately 🔥",
        f"Mixed thoughts about {t}…",
        f"Not sure how I feel about {t}.",
        f"Worst experience ever with {t} 😤",
        f"{t} service quality is terrible",
        f"{t} is a scam?? not sure 👀",
        f"People are spreading a rumor about {t}",
        f"I think bots are spamming posts about {t}",
        f"Please boycott {t}!!!",
        # coordinated-ish duplicates
        f"{t} is a scam boycott now",
        f"{t} is a scam boycott now!!",
        f"{t} scam boycott now",
    ]
    random.seed(99 + len(t))
    random.shuffle(demo_texts)

    now = datetime.now()
    demo_users = [f"user_{random.randint(10, 999)}" for _ in range(len(demo_texts))]
    demo_platforms = [random.choice(["X/Twitter", "Reddit", "News", "Instagram"]) for _ in range(len(demo_texts))]
    for i, txt in enumerate(demo_texts):
        # simulate bursts: some posts within seconds
        burst = random.choice([0, 0, 0, 20, 40, 90])
        ts = (now - timedelta(seconds=(len(demo_texts)-i)*30 - burst)).isoformat(timespec="seconds")
        records.append({"text": txt, "user": demo_users[i], "platform": demo_platforms[i], "timestamp": ts})

# ---------------- ANALYSIS ----------------
rows = []
pos = neu = neg = 0
risky_term_counts = Counter()
total_risky_mentions = 0

for rec in records:
    txt = rec["text"]
    label, pol = classify_sentiment(txt)
    hits = risky_hits(txt)
    total_risky_mentions += len(hits)
    for h in hits:
        risky_term_counts[h] += 1

    if label == "Positive":
        pos += 1
    elif label == "Negative":
        neg += 1
    else:
        neu += 1

    rows.append({
        "user": rec.get("user", "unknown"),
        "platform": rec.get("platform", "unknown"),
        "timestamp": rec.get("timestamp", ""),
        "text": txt,
        "polarity": pol,
        "sentiment": label,
        "risky_terms": ", ".join(hits),
        "norm": normalize_text(txt),
        "tokens": tokenize(txt),
    })

total = max(len(rows), 1)
pos_pct = round((pos / total) * 100)
neu_pct = round((neu / total) * 100)
neg_pct = round((neg / total) * 100)

# Risk heuristic (demo)
risk_score = int(min(100, neg_pct * 1.1 + (total_risky_mentions * 3) + (spike_sensitivity - 5) * 3))

# Keywords
top_keywords = extract_keywords([r["text"] for r in rows], top_k=10)
top_keywords_str = ", ".join([f"{w}({c})" for w, c in top_keywords]) if top_keywords else "No keywords yet"
top_risky = risky_term_counts.most_common(8)
top_risky_str = ", ".join([f"{w}({c})" for w, c in top_risky]) if top_risky else "None detected"

# ---------------- BOT-SCORE (demo, privacy-friendly) ----------------
# We only flag "suspicious behavior patterns" (repetition + burst + risky terms), not identity.
user_posts = defaultdict(list)
for r in rows:
    user_posts[r["user"]].append(r)

# duplicates / near duplicates by normalized text
norm_counts = Counter([r["norm"] for r in rows if r["norm"]])
duplicate_norms = {k for k, v in norm_counts.items() if v >= 2}

def parse_time(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def user_bot_score(u: str):
    posts = user_posts[u]
    if not posts:
        return 0

    # repetition ratio
    norms = [p["norm"] for p in posts if p["norm"]]
    rep = sum(1 for n in norms if n in duplicate_norms)
    rep_ratio = rep / max(1, len(norms))

    # risky term mentions
    risky_mentions = sum(0 if not p["risky_terms"] else len(p["risky_terms"].split(", ")) for p in posts)

    # burstiness: very small time gaps (if timestamps exist)
    times = [parse_time(p["timestamp"]) for p in posts]
    times = [t for t in times if t is not None]
    times.sort()
    fast_gaps = 0
    for i in range(1, len(times)):
        gap = (times[i] - times[i-1]).total_seconds()
        if gap <= 45:  # within 45 seconds
            fast_gaps += 1
    burst_ratio = fast_gaps / max(1, len(times)-1)

    # short/templated text signal (low unique tokens)
    uniq_tokens = set()
    total_tokens = 0
    for p in posts:
        toks = tokenize(p["text"])
        total_tokens += len(toks)
        uniq_tokens.update(toks)
    diversity = (len(uniq_tokens) / max(1, total_tokens))  # lower => more templated
    templated = 1.0 - min(1.0, diversity * 2)  # heuristic

    # weighted score (0..100)
    score = (
        rep_ratio * 45 +
        burst_ratio * 25 +
        min(1.0, risky_mentions / max(1, len(posts)*1.2)) * 20 +
        templated * 10
    )
    return int(max(0, min(100, score)))

bot_scores = []
for u in user_posts:
    bot_scores.append((u, user_bot_score(u), len(user_posts[u])))
bot_scores.sort(key=lambda x: x[1], reverse=True)

# ---------------- COORDINATION CLUSTERS (similar posts) ----------------
# Simple clustering: connect posts if Jaccard similarity >= threshold (demo)
# Works well enough for hackathon.
n = len(rows)
visited = [False] * n
clusters = []

for i in range(n):
    if visited[i]:
        continue
    # build cluster around i
    cluster = [i]
    visited[i] = True
    for j in range(i+1, n):
        if visited[j]:
            continue
        sim = jaccard(rows[i]["tokens"], rows[j]["tokens"])
        if sim >= similarity_threshold:
            cluster.append(j)
            visited[j] = True
    if len(cluster) >= min_cluster_size:
        clusters.append(cluster)

# rank clusters by size + negativity
def cluster_score(idx_list):
    size = len(idx_list)
    negs = sum(1 for k in idx_list if rows[k]["sentiment"] == "Negative")
    return (size, negs)

clusters.sort(key=cluster_score, reverse=True)

# ---------------- UI LAYOUT ----------------
left, right = st.columns([1.15, 1.0], gap="large")

with left:
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Positive</div><div class="metric-value">{pos_pct}%</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Neutral</div><div class="metric-value">{neu_pct}%</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Negative</div><div class="metric-value">{neg_pct}%</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Risk Score</div><div class="metric-value">{risk_score}/100</div></div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("## 🚨 Alerts")
    if risk_score < 35:
        st.markdown('<div class="alert-good">Normal discussion activity ✅</div>', unsafe_allow_html=True)
    elif risk_score < 65:
        st.markdown('<div class="alert-warn">Elevated risk ⚠️ (negativity / risky terms rising)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-bad">High risk 🚨 possible coordinated narrative / spike</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("## 🧾 Live Feed (highlight risky terms)")
    shown = 0
    for r in rows:
        if sentiment_filter != "All" and r["sentiment"] != sentiment_filter:
            continue

        cls = "feed-neu"
        if r["sentiment"] == "Positive":
            cls = "feed-pos"
        elif r["sentiment"] == "Negative":
            cls = "feed-neg"

        st.markdown(
            f"""
<div class="feed-item">
  <div style="opacity:0.85; font-size:0.85rem;">
    <span class="pill">{r["platform"]}</span>
    <span class="pill">@{r["user"]}</span>
    <span class="pill">{r["timestamp"] if r["timestamp"] else "time-unknown"}</span>
  </div>
  <div style="margin-top:0.45rem;">
    <span class="feed-label {cls}">{r["sentiment"]} →</span>
    <span>{highlight_risky(r["text"])}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        shown += 1
        if shown >= 12:
            break

    # Download report
    df_out = pd.DataFrame([{
        "user": r["user"],
        "platform": r["platform"],
        "timestamp": r["timestamp"],
        "text": r["text"],
        "sentiment": r["sentiment"],
        "polarity": r["polarity"],
        "risky_terms": r["risky_terms"],
    } for r in rows])

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
    random.seed((topic.strip() if topic.strip() else "nebula") + str(len(rows)) + str(risk_score))
    base = max(10, min(90, risk_score))
    points = [max(0, min(100, base + random.randint(-18, 18))) for _ in range(18)]
    st.line_chart(points)

    st.write("")
    st.markdown("## 🧠 Insights")
    st.markdown(
        f"""
<div class="card">
  <div style="font-weight:800; margin-bottom:0.35rem;">Top Keywords</div>
  <div style="opacity:0.85;">{top_keywords_str}</div>
  <hr style="border:none;height:1px;background:rgba(255,255,255,0.10);margin:0.8rem 0;">
  <div style="font-weight:800; margin-bottom:0.35rem;">Risky Terms Detected</div>
  <div style="opacity:0.85;">{top_risky_str}</div>
  <hr style="border:none;height:1px;background:rgba(255,255,255,0.10);margin:0.8rem 0;">
  <div style="opacity:0.75; font-size:0.95rem;">
    Total posts analyzed: <b>{len(rows)}</b><br>
    Total risky mentions: <b>{total_risky_mentions}</b><br>
    Similarity threshold: <b>{similarity_threshold:.2f}</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("## 🤖 Bot-score (demo)")
    top_suspicious = [(u,s,c) for (u,s,c) in bot_scores if s >= 55][:8]
    if not top_suspicious:
        st.success("No high bot-scores detected (demo).")
    else:
        st.warning("Suspicious accounts by behavior pattern (demo heuristic):")
        table = pd.DataFrame(top_suspicious, columns=["user", "bot_score", "posts"])
        st.dataframe(table, use_container_width=True)

        st.download_button(
            "Download bot-score list",
            table.to_csv(index=False).encode("utf-8"),
            file_name="nebula_bot_scores.csv",
            mime="text/csv",
        )

    st.write("")
    st.markdown("## 🕸️ Coordination clusters (similar posts)")
    if not clusters:
        st.info("No coordination clusters detected (with current threshold). Try lowering the threshold.")
    else:
        st.error(f"Detected {len(clusters)} suspicious clusters (demo).")
        # show top 3 clusters
        for idx, cl in enumerate(clusters[:3], start=1):
            st.markdown(f"### Cluster {idx} — size {len(cl)}")
            # show a few samples
            sample_ids = cl[:5]
            for k in sample_ids:
                r = rows[k]
                st.write(f"- @{r['user']} ({r['platform']}): {r['text']}")
            st.markdown("---")

st.caption(
    "Nebula v4 — demo heuristics for sentiment + risky terms + bot-score + coordination clusters. "
    "This flags behavior patterns, not real identity verification."
)
