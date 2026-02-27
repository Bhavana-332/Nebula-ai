import streamlit as st

st.set_page_config(page_title="Nebula • Insights", layout="wide")

st.title("🧠 Insights")
st.caption("Talking points & topics (demo)")

st.markdown("### Top talking points")
st.markdown("""
- Product update / experience  
- Service & support  
- Price / value  
""")

st.info("In v2, this will be generated from your live feed (keywords + topics).")
