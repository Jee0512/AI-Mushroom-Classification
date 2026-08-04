"""
Prediction History Page - View, filter, search, and export prediction history.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from components.cards import metric_card, info_card, status_badge
from utils.export_utils import export_to_csv, export_to_json, export_to_excel, get_export_filename
from config import SUCCESS_COLOR, DANGER_COLOR, INFO_COLOR


def app():
    """Render the Prediction History page."""
    
    st.markdown(
        """
        <div class="fade-in">
            <h1 style="margin-bottom: 0.5rem;">📜 Prediction History</h1>
            <p style="color: var(--text-secondary); font-size: 1.05rem; margin-bottom: 1.5rem;">
                View, search, and export all your mushroom predictions. 
                History is stored locally in your browser session and SQLite database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Get History Data ───────────────────────────────────────────
    history = st.session_state.get("prediction_history", [])
    
    if not history:
        st.markdown(
            """
            <div class="card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📜</div>
                <h3>No Prediction History Yet</h3>
                <p style="color: var(--text-secondary);">
                    Start by making predictions on the 
                    <strong>Manual Prediction</strong> or <strong>Image Prediction</strong> pages.
                    All your predictions will appear here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    
    # ── Convert to DataFrame ───────────────────────────────────────
    df = pd.DataFrame(history)
    
    # ── Summary Stats ──────────────────────────────────────────────
    st.markdown("### 📊 Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        metric_card("Total", str(len(df)), icon="📊", color=INFO_COLOR)
    with col2:
        edible_count = len(df[df["prediction"] == "EDIBLE"]) if "prediction" in df.columns else 0
        metric_card("Edible", str(edible_count), icon="🍄", color=SUCCESS_COLOR)
    with col3:
        poisonous_count = len(df[df["prediction"] == "POISONOUS"]) if "prediction" in df.columns else 0
        metric_card("Poisonous", str(poisonous_count), icon="☠️", color=DANGER_COLOR)
    with col4:
        manual_count = len(df[df["method"] == "Manual"]) if "method" in df.columns else 0
        metric_card("Manual", str(manual_count), icon="🔬", color="#1565C0")
    with col5:
        image_count = len(df[df["method"] == "Image"]) if "method" in df.columns else 0
        metric_card("Image", str(image_count), icon="📸", color="#FF8F00")
    
    st.markdown("---")
    
    # ── Filters ────────────────────────────────────────────────────
    st.markdown("### 🔍 Filter History")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        method_filter = st.selectbox(
            "Method",
            ["All", "Manual", "Image"],
            key="history_method_filter",
        )
    
    with col_f2:
        prediction_filter = st.selectbox(
            "Prediction",
            ["All", "EDIBLE", "POISONOUS"],
            key="history_pred_filter",
        )
    
    with col_f3:
        # Date range
        date_options = ["All Time", "Today", "Last 7 Days", "Last 30 Days", "Custom"]
        date_filter = st.selectbox("Time Range", date_options, key="history_date_filter")
    
    with col_f4:
        search_query = st.text_input("🔎 Search", placeholder="Search features, model...")
    
    # ── Apply Filters ──────────────────────────────────────────────
    filtered_df = df.copy()
    
    if method_filter != "All" and "method" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["method"] == method_filter]
    
    if prediction_filter != "All" and "prediction" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["prediction"] == prediction_filter]
    
    if date_filter == "Today" and "timestamp" in filtered_df.columns:
        today = datetime.now().strftime("%Y-%m-%d")
        filtered_df = filtered_df[filtered_df["timestamp"].str.startswith(today)]
    elif date_filter == "Last 7 Days" and "timestamp" in filtered_df.columns:
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        filtered_df = filtered_df[filtered_df["timestamp"] >= cutoff]
    elif date_filter == "Last 30 Days" and "timestamp" in filtered_df.columns:
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        filtered_df = filtered_df[filtered_df["timestamp"] >= cutoff]
    
    if search_query and not filtered_df.empty:
        mask = filtered_df.astype(str).apply(
            lambda row: row.str.contains(search_query, case=False, na=False).any(), axis=1
        )
        filtered_df = filtered_df[mask]
    
    # ── Display Results ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📋 Results ({len(filtered_df)} records)")
    
    if filtered_df.empty:
        st.info("No records match the selected filters.")
    else:
        # Prepare display columns
        display_df = filtered_df.copy()
        
        # Format for display
        if "features" in display_df.columns:
            display_df = display_df.drop(columns=["features"], errors="ignore")
        
        # Reorder columns
        display_cols = ["timestamp", "method", "prediction", "confidence", "risk_level", "model"]
        display_cols = [c for c in display_cols if c in display_df.columns]
        display_df = display_df[display_cols]
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            "timestamp": "Timestamp",
            "method": "Method",
            "prediction": "Prediction",
            "confidence": "Confidence",
            "risk_level": "Risk Level",
            "model": "Model",
        })
        
        # Display as a styled dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "Method": st.column_config.TextColumn("Method", width="small"),
                "Prediction": st.column_config.TextColumn("Prediction", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
                "Model": st.column_config.TextColumn("Model", width="medium"),
            },
        )
        
        # ── Detail View ────────────────────────────────────────────
        st.markdown("### 🔍 Record Details")
        st.markdown("Click on a record to view its details.")
        
        record_ids = list(range(len(filtered_df)))
        selected_idx = st.selectbox(
            "Select a record to view details",
            options=record_ids,
            format_func=lambda i: f"Record #{i+1} - {filtered_df.iloc[i].get('timestamp', 'N/A')} - {filtered_df.iloc[i].get('prediction', 'N/A')}",
            key="history_record_select",
        )
        
        if selected_idx is not None and selected_idx < len(filtered_df):
            record = filtered_df.iloc[selected_idx].to_dict()
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown(
                    f"""
                    <div class="card">
                        <h4>Prediction Details</h4>
                        <p><strong>Timestamp:</strong> {record.get('timestamp', 'N/A')}</p>
                        <p><strong>Method:</strong> {record.get('method', 'N/A')}</p>
                        <p><strong>Prediction:</strong> {record.get('prediction', 'N/A')}</p>
                        <p><strong>Confidence:</strong> {record.get('confidence', 'N/A')}</p>
                        <p><strong>Risk Level:</strong> {record.get('risk_level', 'N/A')}</p>
                        <p><strong>Model:</strong> {record.get('model', 'N/A')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col_d2:
                if record.get("image_name"):
                    st.markdown(
                        f"""
                        <div class="card">
                            <h4>Image Information</h4>
                            <p><strong>Image Name:</strong> {record.get('image_name', 'N/A')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                if record.get("features"):
                    st.markdown(
                        """
                        <div class="card">
                            <h4>Manual Features</h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    try:
                        import json
                        features = json.loads(record["features"]) if isinstance(record["features"], str) else record["features"]
                        if features:
                            features_df = pd.DataFrame(
                                list(features.items()),
                                columns=["Feature", "Value"]
                            )
                            st.dataframe(features_df, use_container_width=True, hide_index=True)
                    except (json.JSONDecodeError, TypeError):
                        st.write("Features data not available in readable format.")
    
    # ── Export Section ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export History")
    st.markdown("Download your prediction history in various formats.")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    
    with col_e1:
        csv_data = export_to_csv(filtered_df.to_dict("records"))
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=get_export_filename("mushroom_predictions", "csv"),
            mime="text/csv",
            use_container_width=True,
        )
    
    with col_e2:
        json_data = export_to_json(filtered_df.to_dict("records"))
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name=get_export_filename("mushroom_predictions", "json"),
            mime="application/json",
            use_container_width=True,
        )
    
    with col_e3:
        excel_data = export_to_excel(filtered_df.to_dict("records"))
        st.download_button(
            label="📗 Download Excel",
            data=excel_data,
            file_name=get_export_filename("mushroom_predictions", "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    
    # ── Clear History ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗑️ Manage History")
    
    col_cl1, col_cl2 = st.columns(2)
    
    with col_cl1:
        if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
            confirm = st.checkbox("Are you sure you want to delete all history?")
            if confirm:
                st.session_state["prediction_history"] = []
                st.success("✅ All prediction history has been cleared.")
                st.rerun()
    
    with col_cl2:
        st.markdown(
            """
            <div style="padding: 0.5rem; color: var(--text-light); font-size: 0.85rem;">
                💡 <strong>Tip:</strong> History is stored in your browser session. 
                Export important records before clearing.
            </div>
            """,
            unsafe_allow_html=True,
        )
