import os
import pandas as pd
import libsql
from dotenv import load_dotenv


def fetch_raw_from_db(conn):
    """Fetches already-staged raw data for feature engineering."""
    print("Fetching raw data from Turso...")
    data = {
        # Using exact column names from your notebook's CREATE TABLE statements
        'market': pd.read_sql("SELECT * FROM market_data", conn),
        'financials': pd.read_sql("SELECT * FROM financial_filings_raw", conn),
        'sec': pd.read_sql("SELECT * FROM sec_mda_risk", conn),
        'transcripts': pd.read_sql("SELECT * FROM transcripts", conn)
    }
    return data