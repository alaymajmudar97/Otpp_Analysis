import json
import os
import re

import libsql
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas.tseries.offsets import BDay
from transformers import pipeline
from src.database_builder import get_db_connection, semantic_sentence_splitter


def load_finbert(model_name="ProsusAI/finbert"):
    """Load and return the FinBERT sentiment model."""
    print(f"Loading {model_name}...")
    return pipeline("text-classification", model=model_name, top_k=None)


def flatten_json_text(value):
    """Convert stored transcript JSON into plain text."""
    if value is None or value == "":
        return ""

    try:
        data = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return str(value)

    if isinstance(data, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in data
        ).strip()

    if isinstance(data, dict):
        return str(data.get("text", "")).strip()

    return str(data).strip()




def normalize_finbert_output(raw_output):
    """Normalize FinBERT output into a simple list of results."""
    if not raw_output:
        return []
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
        return raw_output[0]
    if isinstance(raw_output, list):
        return raw_output
    return []


def score_chunk(finbert, chunk):
    """Score one text chunk with FinBERT."""
    results = normalize_finbert_output(finbert(chunk[:1500]))
    pos = 0.0
    neg = 0.0
    neu = 0.0

    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("label") == "positive":
            pos = float(item.get("score", 0.0))
        elif item.get("label") == "negative":
            neg = float(item.get("score", 0.0))
        elif item.get("label") == "neutral":
            neu = float(item.get("score", 0.0))

    return {"sentiment": pos - neg, "neutrality": neu}


def get_metrics_from_chunks(finbert, chunks):
    """Aggregate sentiment metrics across multiple text chunks."""
    chunks = [chunk for chunk in chunks if str(chunk).strip()]
    if not chunks:
        return {"sentiment": 0.0, "neutrality": 0.0, "dispersion": 0.0}

    net_sentiments = []
    neutral_scores = []

    for chunk in chunks:
        scores = score_chunk(finbert, chunk)
        net_sentiments.append(scores["sentiment"])
        neutral_scores.append(scores["neutrality"])

    return {
        "sentiment": float(np.mean(net_sentiments)),
        "neutrality": float(np.mean(neutral_scores)),
        "dispersion": float(np.std(net_sentiments)) if len(net_sentiments) > 1 else 0.0,
    }


def score_sec_chunks_with_finbert( conn, finbert, table_name="sec_mda_risk"):
    """Score SEC text chunks and write the results back to the database."""
    print(f"Scoring SEC chunks from {table_name}...")
    conn.cursor.execute(f"SELECT chunk_id, content FROM {table_name}")
    rows = conn.cursor.fetchall()

    if not rows:
        print("No SEC chunks found.")
        return

    updates = []

    for chunk_id, content in rows:
        scores = score_chunk(finbert, str(content))
        updates.append((scores["sentiment"], scores["neutrality"], chunk_id))

    conn.cursor.executemany(
        f"""
        UPDATE {table_name}
        SET sentiment_score = ?, neutral_score = ?
        WHERE chunk_id = ?
        """,
        updates,
    )

    conn.commit()
    print(f"Updated {len(updates)} SEC chunks.")


def process_transcripts(finbert, df_transcripts):
    """Build transcript sentiment features from prepped and Q&A text."""
    print(f"Processing {len(df_transcripts)} transcripts...")
    results = []

    for _, row in df_transcripts.iterrows():
        prepped_text = flatten_json_text(row.get("content_prepped", ""))
        qa_text = flatten_json_text(row.get("content_qa", ""))

        prepped_chunks = semantic_sentence_splitter(prepped_text)
        qa_chunks = semantic_sentence_splitter(qa_text)

        prepped_metrics = get_metrics_from_chunks(finbert, prepped_chunks)
        qa_metrics = get_metrics_from_chunks(finbert, qa_chunks)

        results.append(
            {
                "effective_date": pd.to_datetime(row["effective_date"]),
                "prepped_sentiment": prepped_metrics["sentiment"],
                "prepped_neutral": prepped_metrics["neutrality"],
                "prepped_dispersion": prepped_metrics["dispersion"],
                "qa_sentiment": qa_metrics["sentiment"],
                "qa_neutral": qa_metrics["neutrality"],
                "qa_dispersion": qa_metrics["dispersion"],
            }
        )

    return pd.DataFrame(results)



def build_sec_features(df_raw):
    """
    Pulls raw SEC chunks, for quarterly mDA and Risk labels, calculates Mean & Std (Dispersion), 
    and engineers QoQ Deltas 
    """
    print("Engineering MDA and Risk Sentiment Features...")
    
    
    df_raw['filing_date'] = pd.to_datetime(df_raw['filing_date'])
    
    
    df_raw['item_type'] = df_raw['item_type'].replace('RiskFactorsUpdate', 'RiskFactors')

    # 1. Named Aggregation (Now including the Neutral mean)
    df_grouped = df_raw.groupby(['doc_id', 'filing_date', 'item_type']).agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        neutral_mean=('neutral_score', 'mean')
    ).reset_index()

    # 2. The Pivot (Fanning out 3 distinct metrics per section)
    df_pivot = df_grouped.pivot_table(
        index=['doc_id', 'filing_date'], 
        columns='item_type', 
        values=['sentiment_mean', 'sentiment_std', 'neutral_mean']
    )

    # 3. Flatten the Pivot MultiIndex
    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    # 4. Clean Renaming (Mapping all new columns)
    df_pivot = df_pivot.rename(columns={
        'MDA_sentiment_mean': 'MDA_Sentiment', 
        'MDA_sentiment_std': 'MDA_Dispersion',
        'MDA_neutral_mean': 'MDA_Neutrality',
        'RiskFactors_sentiment_mean': 'Risk_Combined_Mean',
        'RiskFactors_sentiment_std': 'Risk_Combined_Std',
        'RiskFactors_neutral_mean': 'Risk_Combined_Neutrality'
    })

    # 5. The Time-Series Delta (Quarter-over-Quarter Change)
    df_pivot = df_pivot.sort_values(by=['filing_date']).reset_index(drop=True)
    df_pivot['MDA_Delta'] = df_pivot['MDA_Sentiment'].diff()
    df_pivot['Risk_Delta'] = df_pivot['Risk_Combined_Mean'].diff()

    # 6. The Executive vs Legal Spread
    
    # Fill any NaN standard deviations with 0
    df_pivot = df_pivot.fillna({'MDA_Dispersion': 0, 'Risk_Combined_Std': 0})

    # 7. The Look-Ahead Bias Protection (Effective Date T+1)
    df_pivot['effective_date'] = df_pivot['filing_date'] + pd.Timedelta(days=1)
    
    print(f"Generated NLP features for {len(df_pivot)} documents.")
    return df_pivot    

def calculate_financial_ratios(df_fin):
    """Create simple financial ratio features from filing data."""
    
    df = df_fin.copy()
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    df = df.sort_values("effective_date")

    df["roa"] = df["net_income"] / df["total_assets"]
    df["debt_to_asset"] = df["total_liability"] / df["total_assets"]
    df["cash_coverage"] = df["op_cash_flow"] / df["total_liability"]
    df["fcf_margin"] = (df["op_cash_flow"] + df["capex"]) / df["revenue"]
    df["net_income_qoq"] = df["net_income"].pct_change(1)

    return df[
        [
            "effective_date",
            "roa",
            "cash_coverage",
            "debt_to_asset",
            "fcf_margin",
            "net_income_qoq",
        ]
    ]


def add_temporal_signals(df_golden):
    """Add days-since-event features for earnings and filings."""
    print("Calculating temporal signals...")
    df = df_golden.copy().sort_values("trading_date")
    df["trading_date"] = pd.to_datetime(df["trading_date"])

    df["is_earnings_day"] = df["qa_sentiment"].notna()
    df["is_filing_day"] = df["sec_sentiment"].notna()

    df["last_earnings_date"] = df["trading_date"].where(df["is_earnings_day"]).ffill()
    df["last_filing_date"] = df["trading_date"].where(df["is_filing_day"]).ffill()

    #Calculate temporal (days passed)
    df["days_since_earnings"] = (df["trading_date"] - df["last_earnings_date"]).dt.days
    df["days_since_filing"] = (df["trading_date"] - df["last_filing_date"]).dt.days

    df["days_since_earnings"] = df["days_since_earnings"].fillna(999)
    df["days_since_filing"] = df["days_since_filing"].fillna(999)

    df.drop(
        columns=[
            "is_earnings_day",
            "is_filing_day",
            "last_earnings_date",
            "last_filing_date",
        ],
        inplace=True,
    )

    return df


def build_market_and_vol_features(df, window=21):
    """
    Institutional Grade Engineering: 
    - All features lagged by 1 to prevent look-ahead bias.
    - Features anchored to t-1, Target anchored to t.
    """
    print("Engineering sanitized features...")
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.sort_values("trading_date")

    # 1. Market Returns (The Source of Truth)
    df["msft_return"] = df["msft_close"].pct_change()
    df["qqq_return"] = df["qqq_close"].pct_change()
    df["spy_return"] = df["spy_close"].pct_change()

    # 2. Performance Spread (Lagged by 1 to be safe)
    df["msft_vs_tech"] = (df["msft_return"] - df["qqq_return"]).shift(1)
    df["msft_vs_market"] = (df["msft_return"] - df["spy_return"]).shift(1)

    # 3. Macro features (Lagged by 1 to represent yesterday's information)
    df["vix_level"] = df["vix_close"].shift(1)
    df["vix_5d_trend"] = df["vix_close"].diff(5).shift(1)
    df["yield_10y_level"] = df["tnx_yield"].shift(1)
    df["yield_10y_delta_5d"] = df["tnx_yield"].diff(5).shift(1)

    # 4. Volatility Estimators (Anchored to past, shifted by 1)
    df["vol_rolling_21d"] = (df["msft_return"].rolling(window).std().shift(1)) * np.sqrt(252)
    df["qqq_vol_21d"] = (df["qqq_return"].rolling(window).std().shift(1)) * np.sqrt(252)

    # 5. Garman Klass (Lagged to represent yesterday's risk profile)
    log_ho = np.log(df["msft_high"] / df["msft_open"])
    log_lo = np.log(df["msft_low"] / df["msft_open"])
    log_co = np.log(df["msft_close"] / df["msft_open"])
    gk_daily = 0.5 * (log_ho - log_lo) ** 2 - (2 * np.log(2) - 1) * log_co**2
    df["vol_garman_klass"] = np.sqrt(np.maximum(gk_daily, 0)).rolling(window).mean().shift(1) * np.sqrt(252)

    # 6. Liquidity (Lagged)
    df["vol_surge"] = (df["msft_volume"] / df["msft_volume"].rolling(window).mean()).shift(1)
    dollar_vol = df["msft_close"] * df["msft_volume"]
    df["amihud_ratio"] = (df["msft_return"].abs() / dollar_vol).shift(1) * 1e6

    # 7. Moving Averages (Lagged)
    df["dist_from_ma200"] = ((df["msft_close"] / df["msft_close"].rolling(200).mean()) - 1).shift(1)

    # 8. Target Feature (Future Volatility)
    # This is the only one NOT lagged by 1, as it represents the future.
    df["target_vol_21d"] = df["msft_return"].rolling(window).std().shift(-window) * np.sqrt(252)

    return df.dropna() # Critical: Removes NaNs from shifts

def fetch_table(conn, query):
    """Run a SQL query and return the result as a dataframe."""
    return pd.read_sql(query, conn)


def build_feature_dataset(conn, finbert):
    """Load source tables, merge features, and return the final dataset."""
    print("Fetching source tables...")
    df_market = fetch_table(conn, "SELECT * FROM market_data ORDER BY trading_date")
    df_transcripts = fetch_table(conn, "SELECT * FROM earnings_transcripts ORDER BY effective_date")
    df_sec = fetch_table(conn, "SELECT * FROM sec_mda_risk ORDER BY filing_date")
    df_fin = fetch_table(conn, "SELECT * FROM sec_financials ORDER BY effective_date")

    transcript_features = process_transcripts(finbert, df_transcripts)
    sec_features = build_sec_features(df_sec)
    fin_features = calculate_financial_ratios(df_fin)

    df_market["trading_date"] = pd.to_datetime(df_market["trading_date"])
    transcript_features["effective_date"] = pd.to_datetime(transcript_features["effective_date"])
    sec_features["effective_date"] = pd.to_datetime(sec_features["effective_date"])
    fin_features["effective_date"] = pd.to_datetime(fin_features["effective_date"])

    df = df_market.merge(
        transcript_features,
        left_on="trading_date",
        right_on="effective_date",
        how="left",
    )

    df = df.merge(
        sec_features.drop(columns=["filing_date"]),
        left_on="trading_date",
        right_on="effective_date",
        how="left",
        suffixes=("", "_sec"),
    )

    df = df.merge(
        fin_features,
        left_on="trading_date",
        right_on="effective_date",
        how="left",
        suffixes=("", "_fin"),
    )

    effective_date_cols = [col for col in df.columns if col.startswith("effective_date")]
    if effective_date_cols:
        df.drop(columns=effective_date_cols, inplace=True)

    df = add_temporal_signals(df)
    df = df.sort_values("trading_date").ffill()
    df = build_market_and_vol_features(df)

    print(f"Final feature dataset shape: {df.shape}")
    return df


def save_feature_dataset(df, output_path="./data/df_features.csv"):
    """Save the final feature dataframe to CSV."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")


def main():
    """Run the full feature engineering pipeline."""
    conn = get_db_connection()
    finbert = load_finbert()

    score_sec_chunks_with_finbert(conn, finbert)

    df_features = build_feature_dataset(conn, finbert)
    save_feature_dataset(df_features)


if __name__ == "__main__":
    main()
