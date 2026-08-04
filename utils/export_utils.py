"""
Export utilities for the Mushroom Classifier.
Provides functions to export data in various formats (CSV, JSON, Excel).
"""
import pandas as pd
import io
import json
from datetime import datetime


def export_to_csv(data: list, filename: str = None) -> bytes:
    """
    Export data to CSV format.
    
    Args:
        data: List of dictionaries
        filename: Optional filename for the export
    
    Returns:
        CSV bytes
    """
    if not data:
        return b""
    
    df = pd.DataFrame(data)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()


def export_to_json(data: list, filename: str = None) -> bytes:
    """
    Export data to JSON format.
    
    Args:
        data: List of dictionaries
        filename: Optional filename for the export
    
    Returns:
        JSON bytes
    """
    if not data:
        return b"[]"
    
    json_str = json.dumps(data, indent=2, default=str)
    return json_str.encode("utf-8")


def export_to_excel(data: list, filename: str = None) -> bytes:
    """
    Export data to Excel format.
    
    Args:
        data: List of dictionaries
        filename: Optional filename for the export
    
    Returns:
        Excel bytes
    """
    if not data:
        return b""
    
    df = pd.DataFrame(data)
    excel_buffer = io.BytesIO()
    
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
        
        # Auto-adjust column widths
        for column in df:
            col_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
            writer.sheets["Predictions"].column_dimensions[column].width = min(col_width, 50)
    
    excel_buffer.seek(0)
    return excel_buffer.getvalue()


def get_export_filename(prefix: str = "mushroom_predictions", ext: str = "csv") -> str:
    """
    Generate a timestamped filename for exports.
    
    Args:
        prefix: File prefix
        ext: File extension
    
    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{ext}"
