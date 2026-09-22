import streamlit as st
import pandas as pd
from config import SAFETY_DISCLAIMER
from utils.history_manager import get_all_history, init_db
from utils.export_utils import export_to_csv, export_to_json

st.set_page_config(page_title="History | Mushroom Classifier", page_icon="📜", layout="wide")
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
    <div class="nav-badge">Prediction History</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header" style="margin-top:80px;">
    <span class="page-title">📜 Prediction History</span>
    <p class="page-subtitle">Review and export your past classifications.</p>
</div>
""", unsafe_allow_html=True)

init_db()

try:
    df = get_all_history()
    if df.empty:
        st.markdown("""
        <div class="info-card">
            <span class="info-icon">📋</span>
            <div class="info-title">No History Yet</div>
            <div class="info-body">Make a classification first to see your predictions here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            csv_data = export_to_csv(df)
            st.download_button("📥 Download CSV", data=csv_data, file_name="mushroom_history.csv", mime="text/csv")
        with col2:
            json_data = export_to_json(df)
            st.download_button("📥 Download JSON", data=json_data, file_name="mushroom_history.json", mime="application/json")
except Exception as e:
    st.error(f"Error loading history: {e}")

st.markdown(f"""
<div class="warn-box" style="margin-top:24px;">⚠️ <strong>Safety Warning:</strong> {SAFETY_DISCLAIMER}</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:16px;">
    <button class="btn-secondary" onclick="window.location.href='/'">🏠 Back to Home</button>
</div>
""", unsafe_allow_html=True)
