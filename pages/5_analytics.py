import streamlit as st
from config import SAFETY_DISCLAIMER
from utils.history_manager import get_history_stats, get_all_history, init_db
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analytics | Mushroom Classifier", page_icon="📊", layout="wide")
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@600;700;800;900&display=swap');</style>", unsafe_allow_html=True)
st.markdown(f"<style>{open('assets/styles.css', 'r', encoding='utf-8').read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="nav-header">
    <div class="nav-brand">
        <span class="nav-logo">🍄</span>
        <div class="nav-title-group">
            <div class="nav-title">Mushroom Identifier</div>
            <div class="nav-subtitle">AI Edibility Analysis</div>
        </div>
    </div>
    <div class="nav-badge">Analytics</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card" style="padding:32px;margin-bottom:24px;">
<div class="page-header" style="margin-top:80px;">
    <span class="page-title">📊 Analytics</span>
    <p class="page-subtitle">Overview of your prediction activity.</p>
</div>
""", unsafe_allow_html=True)

init_db()

try:
    stats = get_history_stats()
    if stats["total"] == 0:
        st.markdown("""
        <div class="info-card">
            <span class="info-icon">📊</span>
            <div class="info-title">No Data Yet</div>
            <div class="info-body">Make some predictions to see analytics here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", stats["total"])
        col2.metric("Edible", stats["edible"])
        col3.metric("Poisonous", stats["poisonous"])
        col4.metric("Avg Confidence", f"{stats['avg_confidence']}%")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.1);margin-top:24px;'>", unsafe_allow_html=True)
        st.subheader("Edible vs Poisonous Ratio")

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['Edible', 'Poisonous']
        sizes = [stats["edible"], stats["poisonous"]]
        colors = ['#4CAF50', '#f44336']

        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.write("Not enough data to plot.")
except Exception as e:
    st.error(f"Error loading analytics: {e}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="warn-box" style="margin-top:24px;">⚠️ <strong>Safety Warning:</strong> {SAFETY_DISCLAIMER}</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:16px;">
    <button class="btn-secondary" onclick="window.location.href='/'">🏠 Back to Home</button>
</div>
""", unsafe_allow_html=True)
