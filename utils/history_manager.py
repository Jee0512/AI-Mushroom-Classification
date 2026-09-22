"""
Prediction History Manager - SQLite-based storage and retrieval.
Handles saving, loading, filtering, searching, and exporting prediction history.
"""
import streamlit as st
import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from config import HISTORY_DB_PATH, MAX_HISTORY_RECORDS


def get_db_connection():
    """Get a connection to the SQLite database."""
    # Ensure directory exists
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            method TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence TEXT NOT NULL,
            risk_level TEXT,
            features TEXT,
            image_name TEXT,
            model TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def save_prediction(entry: dict):
    """
    Save a prediction entry to the database.
    
    Args:
        entry: Dictionary with keys: timestamp, method, prediction, confidence,
               risk_level, features (optional), image_name (optional), model
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Convert features dict to JSON string if present
    features_json = None
    if "features" in entry and entry["features"]:
        features_json = json.dumps(entry["features"])
    
    cursor.execute("""
        INSERT INTO prediction_history 
        (timestamp, method, prediction, confidence, risk_level, features, image_name, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entry.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        entry.get("method", "Manual"),
        entry.get("prediction", "Unknown"),
        entry.get("confidence", "0%"),
        entry.get("risk_level", "Unknown"),
        features_json,
        entry.get("image_name", None),
        entry.get("model", "Unknown"),
    ))
    
    conn.commit()
    conn.close()


def get_all_history() -> pd.DataFrame:
    """
    Retrieve all prediction history as a DataFrame.
    
    Returns:
        DataFrame with columns: id, timestamp, method, prediction, confidence,
        risk_level, features, image_name, model, created_at
    """
    init_db()
    conn = get_db_connection()
    
    query = "SELECT * FROM prediction_history ORDER BY id DESC"
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df


def get_filtered_history(
    method: str = None,
    prediction: str = None,
    date_from: str = None,
    date_to: str = None,
    search_query: str = None,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Retrieve filtered prediction history.
    
    Args:
        method: Filter by method ("Manual" or "Image")
        prediction: Filter by prediction ("EDIBLE" or "POISONOUS")
        date_from: Start date (YYYY-MM-DD)
        date_to: End date (YYYY-MM-DD)
        search_query: Search in features or image name
        limit: Maximum number of records to return
    
    Returns:
        Filtered DataFrame
    """
    conn = get_db_connection()
    
    conditions = []
    params = []
    
    if method:
        conditions.append("method = ?")
        params.append(method)
    
    if prediction:
        conditions.append("prediction = ?")
        params.append(prediction)
    
    if date_from:
        conditions.append("date(timestamp) >= date(?)")
        params.append(date_from)
    
    if date_to:
        conditions.append("date(timestamp) <= date(?)")
        params.append(date_to)
    
    if search_query:
        conditions.append("(features LIKE ? OR image_name LIKE ? OR model LIKE ?)")
        params.extend([f"%{search_query}%", f"%{search_query}%", f"%{search_query}%"])
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query = f"SELECT * FROM prediction_history WHERE {where_clause} ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    df = pd.read_sql_query(query, conn, params=params)
    
    conn.close()
    return df


def delete_history(record_id: int = None):
    """
    Delete prediction history records.
    
    Args:
        record_id: If provided, delete only this record. Otherwise, delete all.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if record_id:
        cursor.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
    else:
        cursor.execute("DELETE FROM prediction_history")
    
    conn.commit()
    conn.close()


def get_history_stats() -> dict:
    """
    Get statistics about the prediction history.
    
    Returns:
        Dictionary with stats: total, edible, poisonous, manual, image, etc.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    cursor.execute("SELECT COUNT(*) FROM prediction_history")
    stats["total"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE prediction = 'EDIBLE'")
    stats["edible"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE prediction = 'POISONOUS'")
    stats["poisonous"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE method = 'Manual'")
    stats["manual"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE method = 'Image'")
    stats["image"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(CAST(REPLACE(confidence, '%', '') AS FLOAT)) FROM prediction_history")
    avg_conf = cursor.fetchone()[0]
    stats["avg_confidence"] = round(avg_conf, 2) if avg_conf else 0
    
    conn.close()
    return stats


def export_to_csv(history_df: pd.DataFrame) -> bytes:
    """Export history DataFrame to CSV bytes."""
    return history_df.to_csv(index=False).encode("utf-8")


def export_to_json(history_df: pd.DataFrame) -> bytes:
    """Export history DataFrame to JSON bytes."""
    return history_df.to_json(orient="records", indent=2).encode("utf-8")


def sync_session_to_db():
    """Sync prediction history from session state to database."""
    history = st.session_state.get("prediction_history", [])
    if not history:
        return
    
    for entry in history:
        save_prediction(entry)


def load_history_to_session():
    """Load prediction history from database to session state."""
    df = get_all_history()
    if not df.empty:
        # Convert DataFrame rows to list of dicts
        history = df.to_dict("records")
        st.session_state["prediction_history"] = history
