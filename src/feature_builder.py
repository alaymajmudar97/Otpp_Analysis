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
    def aggregate_sec_chunks(self, df_sec_chunks):
        """
        Rolls up already-scored SEC chunks into document-level daily features.
        Assumes df_sec_chunks has 'pos', 'neg', and 'neutral' columns.
        """
        print(f"Aggregating {len(df_sec_chunks)} SEC chunks...")
        df = df_sec_chunks.copy()
        
        # Calculate Net Sentiment if not already present
        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = df['pos'] - df['neg']
            
        # The Mathematical Rollup
        sec_features = df.groupby('filing_date').agg(
            sec_sentiment=('sentiment_score', 'mean'),
            sec_neutral=('neutral_score', 'mean'),
            sec_dispersion=('sentiment_score', 'std')
        ).reset_index()
        
        sec_features['filing_date'] = pd.to_datetime(sec_features['filing_date'])
        
        # 2. Shift forward by 1 Business Day (skips weekends automatically)
        sec_features['effective_date'] = sec_features['filing_date'] + BDay(1)
        # Filings with only 1 chunk will return NaN for Standard Deviation. Fill with 0.
        sec_features['sec_dispersion'] = sec_features['sec_dispersion'].fillna(0)
        
        return sec_features
        
        
def calculate_financial_ratios(df_fin):
    df = df_fin.copy()
    # Sort chronologically
    df = df.sort_values('effective_date')
    
    # 1. Stationary Ratios
    df['roa'] = df['net_income'] / df['total_assets']
    df['debt_to_asset'] = df['total_liability'] / df['total_assets']
    df['cash_coverage'] = df['cash_eq'] / df['total_liability']
    
    # FCF Margin (Capex is negative in your DB)
    df['fcf_margin'] = (df['op_cash_flow'] + df['capex']) / df['revenue']
    
    # 2. Momentum
    df['net_income_qoq'] = df['net_income'].pct_change(1)
    
    return df.fillna(0)

def add_temporal_signals(df_golden):
    """Adds 'Days Since' features and Macro momentum."""
    df = df_golden.copy()
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    
    # 1. Days Since Last SEC Filing
    df['last_filing_date'] = df['filing_date'].ffill()
    df['days_since_filing'] = (df['trading_date'] - pd.to_datetime(df['last_filing_date'])).dt.days
    
    # 2. Days Since Last Earnings Call
    df['last_earnings_date'] = df['effective_date_transcript'].ffill()
    df['days_since_earnings'] = (df['trading_date'] - pd.to_datetime(df['last_earnings_date'])).dt.days
    
    # 3. Macro Trend (VIX Momentum)
    df['vix_5d_trend'] = df['vix_close'].diff().rolling(window=5).mean()
    
    return df.fillna(0)


def build_volatility_signals(df_market):
    """Adds historical risk regime features and the Target Variable."""
    df = df_market.copy().sort_values('trading_date')
    
    # 1. Realized Volatility (Rolling 21-day std of log returns)
    df['vol_rolling_21d'] = df['msft_return'].rolling(window=21).std() * np.sqrt(252)
    
    # 2. VIX Momentum (Macro fear signal)
    df['vix_5d_trend'] = df['vix_close'].diff(5)
    
    # 3. TARGET VARIABLE: 21-day Forward Realized Volatility
    # This is what LightGBM will try to predict.
    df['target_vol_21d'] = (
        df['msft_return']
        .shift(-21) # Look ahead 21 days
        .rolling(window=21)
        .std() * np.sqrt(252)
    )
    
    return df