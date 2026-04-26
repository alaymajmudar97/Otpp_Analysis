import numpy as np
from transformers import pipeline

import pandas as pd
import numpy as np
import json
import re
from transformers import pipeline
from pandas.tseries.offsets import BDay
class NLPFeatureEngine:
    def __init__(self, model_name="ProsusAI/finbert"):
        print(f"Loading {model_name} into memory...")
        # Load the model once for the entire class to save memory
        self.finbert = pipeline("text-classification", model=model_name, top_k=None)
        
    def semantic_sentence_splitter(self, text, max_chars=1200):
        """Splits 'wall of text' into chunks at sentence boundaries."""
        if not text: return []
        sentences = re.split(r'(?<=[.!?]) +', str(text))
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        if current_chunk: chunks.append(current_chunk.strip())
        return chunks

    def get_metrics_from_chunks(self, chunks):
        """
        THE UNIVERSAL SCORER.
        Takes any list of text chunks, runs FinBERT, and returns the 3 Alpha Metrics.
        """
        if not chunks:
            return {'sentiment': 0.0, 'neutrality': 0.0, 'dispersion': 0.0}
            
        net_sentiments = []
        neutral_scores = [] 
        
        for chunk in chunks:
            raw_output = self.finbert(chunk[:1500])[0]
            pos, neg, neu = 0.0, 0.0, 0.0
            
            # Map the probabilities safely
            for item in raw_output:
                if item['label'] == 'positive': pos = item['score']
                elif item['label'] == 'negative': neg = item['score']
                elif item['label'] == 'neutral': neu = item['score']
                    
            net_sentiments.append(pos - neg)
            neutral_scores.append(neu)
            
        return {
            'sentiment': np.mean(net_sentiments),
            'neutrality': np.mean(neutral_scores),
            # Standard Deviation calculates the Dispersion
            'dispersion': np.std(net_sentiments) if len(net_sentiments) > 1 else 0.0
        }

    # ==========================================
    # PIPELINE 1: TRANSCRIPTS (Raw Text -> Metrics)
    # ==========================================
    def process_transcripts(self, df_transcripts):
        """Extracts high-resolution signals from Prepped vs Q&A blocks."""
        print(f"Processing {len(df_transcripts)} Transcripts...")
        results = []
        
        for _, row in df_transcripts.iterrows():
            def flatten_json(val):
                if not val: return ""
                try:
                    data = json.loads(val) if isinstance(val, str) else val
                    return " ".join([item.get("text", "") for item in data])
                except: return str(val)
                
            # 1. Prepare Text
            prepped_text = flatten_json(row.get('content_prepped', ''))
            qa_text = flatten_json(row.get('content_qa', ''))
            
            # 2. Chunk Text
            p_chunks = self.semantic_sentence_splitter(prepped_text)
            q_chunks = self.semantic_sentence_splitter(qa_text)
            
            # 3. Route through Universal Scorer
            p_metrics = self.get_metrics_from_chunks(p_chunks)
            q_metrics = self.get_metrics_from_chunks(q_chunks)
            
            results.append({
                'effective_date': pd.to_datetime(row['effective_date']),
                'prepped_sentiment': p_metrics['sentiment'],
                'prepped_neutral': p_metrics['neutrality'],
                'prepped_dispersion': p_metrics['dispersion'],
                'qa_sentiment': q_metrics['sentiment'],
                'qa_neutral': q_metrics['neutrality'],
                'qa_dispersion': q_metrics['dispersion']
            })
            
        return pd.DataFrame(results)

    # ==========================================
    # PIPELINE 2: SEC FILINGS (Chunks -> Daily Aggregation)
    # ==========================================
    

def build_sec_features(df_raw):
    """
    Pulls raw SEC chunks, harmonizes Risk labels, calculates Mean & Std (Dispersion), 
    and engineers QoQ Deltas and intra-document spreads.
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
    df = df_fin.copy()
    # Sort chronologically
    
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df = df.sort_values('effective_date')
    
    # 1. Stationary Ratios
    df['roa'] = df['net_income'] / df['total_assets']
    df['debt_to_asset'] = df['total_liability'] / df['total_assets']
    df['cash_coverage'] = df['op_cash_flow'] / df['total_liability']
    
    # FCF Margin (Capex is negative in your DB)
    df['fcf_margin'] = (df['op_cash_flow'] + df['capex']) / df['revenue']
    
    # 2. Momentum
    df['net_income_qoq'] = df['net_income'].pct_change(1)
    
    return  df[['effective_date', 'roa', 'cash_coverage', 'debt_to_asset', 'fcf_margin', 'net_income_qoq']]






def add_temporal_signals(df_golden):
    """
    Calculates how 'stale' the NLP information is.
    MUST BE RUN: After merging the dataframes, but BEFORE forward-filling (ffill).
    """
    print("⏳ Calculating Information Decay (Temporal Signals)...")
    df = df_golden.copy().sort_values('trading_date')
    
    # ==========================================
    # 1. Detect Event Days
    # ==========================================
    # If the NLP column is NOT missing (not NaN), an event happened that day.
    df['is_earnings_day'] = df['qa_sentiment'].notna()
    df['is_filing_day'] = df['MDA_Sentiment'].notna()
    
    # ==========================================
    # 2. Anchor the Dates
    # ==========================================
    # Stamp the actual trading date onto the event, then drag that date forward
    df['last_earnings_date'] = df['trading_date'].where(df['is_earnings_day']).ffill()
    df['last_filing_date'] = df['trading_date'].where(df['is_filing_day']).ffill()
    
    # ==========================================
    # 3. Calculate Days Elapsed (Decay)
    # ==========================================
    # Subtract the dragged date from the current trading date
    df['days_since_earnings'] = (df['trading_date'] - df['last_earnings_date']).dt.days
    df['days_since_filing'] = (df['trading_date'] - df['last_filing_date']).dt.days
    
    # Fill early NaNs (before the very first filing of the dataset) with a high penalty number
    df['days_since_earnings'] = df['days_since_earnings'].fillna(999)
    df['days_since_filing'] = df['days_since_filing'].fillna(999)
    
    # ==========================================
    # 4. Cleanup
    # ==========================================
    df.drop(columns=[
        'is_earnings_day', 'is_filing_day', 
        'last_earnings_date', 'last_filing_date'
    ], inplace=True)
    
    return df


def build_market_and_vol_features(df_golden):
    """
    The Ultimate Market Engine. 
    Calculates Returns, Garman-Klass Volatility, Sector Alpha, Macro Trends, and Liquidity.
    Assumes df_golden has: msft_open, msft_high, msft_low, msft_close, msft_volume, 
                           qqq_close, spy_close, vix_close, tnx_yield
    """
    print("📈 Engineering Market, Macro, Sector, and Volatility Signals...")
    
    df = df_golden.copy()
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    df.sort_values('trading_date', inplace=True)
    window = 21 # 1 Trading Month
    
    # ==========================================
    # 1. BASE RETURNS (The Foundation)
    # ==========================================
    df['msft_return'] = df['msft_close'].pct_change()
    df['qqq_return'] = df['qqq_close'].pct_change()
    df['spy_return'] = df['spy_close'].pct_change()
    
    # ==========================================
    # 2. SECTOR & MARKET ALPHA (Relative Context)
    # ==========================================
    # If negative, MSFT is dropping while the market is fine (Idiosyncratic Risk)
    df['msft_vs_tech'] = df['msft_return'] - df['qqq_return']
    df['msft_vs_market'] = df['msft_return'] - df['spy_return']
    
    # ==========================================
    # 3. MACRO SIGNALS (External Pressures)
    # ==========================================
    df['vix_level'] = df['vix_close']
    df['vix_5d_trend'] = df['vix_close'].diff(5)
    
    df['yield_10y_level'] = df['tnx_yield']
    df['yield_10y_delta_5d'] = df['tnx_yield'].diff(5) # The "Shock" indicator
    
    # ==========================================
    # 4. VOLATILITY ESTIMATORS (Risk Tracking)
    # ==========================================
    # Standard Close-to-Close
    df['vol_rolling_21d'] = df['msft_return'].rolling(window).std() * np.sqrt(252)
    df['qqq_vol_21d'] = df['qqq_return'].rolling(window).std() * np.sqrt(252)
    
    # Garman-Klass (Intraday High-Resolution Risk)
    log_ho = np.log(df['msft_high'] / df['msft_open'])
    log_lo = np.log(df['msft_low'] / df['msft_open'])
    log_co = np.log(df['msft_close'] / df['msft_open'])
    
    gk_daily = 0.5 * (log_ho - log_lo)**2 - (2 * np.log(2) - 1) * log_co**2
    # Ensure no negative values before sqrt (can happen with tiny data errors)
    gk_daily = np.maximum(gk_daily, 0) 
    df['vol_garman_klass'] = np.sqrt(gk_daily).rolling(window).mean() * np.sqrt(252)
    
    # ==========================================
    # 5. LIQUIDITY RISK (Amihud & Volume)
    # ==========================================
    df['vol_surge'] = df['msft_volume'] / df['msft_volume'].rolling(window).mean()
    
    # Amihud Illiquidity (Price impact per dollar traded)
    dollar_vol = df['msft_close'] * df['msft_volume']
    df['amihud_ratio'] = (df['msft_return'].abs() / dollar_vol) * 1e6 
    
    # ==========================================
    # 6. TECHNICALS (Mean Reversion)
    # ==========================================
    df['ma200'] = df['msft_close'].rolling(window=200).mean()
    # Distance from 200 DMA (The "Rubber Band" effect)
    df['dist_from_ma200'] = (df['msft_close'] / df['ma200']) - 1
    
    # Clean up intermediate tech column
    df.drop(columns=['ma200'], inplace=True)
    
    # ==========================================
    # 7. THE TARGET VARIABLE (Y)
    # ==========================================
    # We shift the 21-day realized volatility BACKWARDS by 21 days.
    # This means today's row contains the actual volatility that occurred over the NEXT month.
    df['target_vol_21d'] = df['msft_return'].shift(-window).rolling(window).std() * np.sqrt(252)
    
    return df

