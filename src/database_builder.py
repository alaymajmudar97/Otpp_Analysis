import os
import re
import json
import numpy as np
import pandas as pd
import yfinance as yf
import libsql
from datetime import datetime, timedelta
from dotenv import load_dotenv

# API Libraries
from edgar import Company, set_identity
import earningscall



#Database Setup
def get_db_connection():
    load_dotenv()
    try:
        conn = libsql.connect(
            database=os.getenv("TURSO_URL"), 
            auth_token=os.getenv("TURSO_TOKEN")
        )
        return conn
    except Exception as e:
        print(f" Database Connection Error: {e}")
        return None

def setup_sec_api():
    """
    Initializes the edgartools SEC identity.
    Requires SEC_API_USER_AGENT in your .env file (e.g. 'Name name@email.com')
    """
    load_dotenv()
    user_agent = os.getenv("SEC_API_USER_AGENT")
    if not user_agent:
        raise ValueError("SEC_API_USER_AGENT not found in .env file.")
    set_identity(user_agent)
    print(f" SEC API Identity Set: {user_agent}")



# Table Creation Scripts
def create_tables(conn):
    """Creates all required tables if they don't exist."""

    print("Building database schemas...")
    
    conn.cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        trading_date TEXT PRIMARY KEY, msft_close REAL, msft_volume INTEGER, 
        msft_return REAL, vix_close REAL, tnx_yield REAL, msft_open REAL, msft_high REAL,
        msft_low REAL, qqq_close Real, spy_close Real
    )""")
    
    conn.cursor.execute("""
    CREATE TABLE IF NOT EXISTS financial_filings_raw (
        filing_id TEXT PRIMARY KEY, ticker TEXT, filing_date TEXT, effective_date TEXT,
        fiscal_year INTEGER, fiscal_period TEXT, revenue REAL, net_income REAL,
        op_cash_flow REAL, capex REAL, cash_eq REAL, total_liability REAL,
        total_assets REAL, total_equity REAL
    )""")
    
    conn.cursor.execute("""
    CREATE TABLE IF NOT EXISTS sec_chunks (
        chunk_id TEXT PRIMARY KEY, doc_id TEXT, ticker TEXT, filing_date TEXT,
        item_type TEXT, chunk_index INTEGER, content TEXT, sentiment_score REAL
    )""")
    
    conn.cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        transcript_id TEXT PRIMARY KEY, ticker TEXT, event_timestamp TEXT, 
        effective_date TEXT, fiscal_year INTEGER, fiscal_period TEXT, 
        content_prepped TEXT, content_qa TEXT, sentiment_score REAL DEFAULT 0.0
    )""")
    
    conn.commit()
    print("All tables verified/created.")



#Market Data Pipeline
def ingest_market_data(conn):
    """
    Fetches historical market and macro data for MSFT and key indices, calculates returns, and stores in Turso."""
    
    print("\n--- Starting Market Data Ingestion ---")
    tickers = ["MSFT", "^VIX", "^TNX", 'QQQ', 'SPY']
    start_date = "2018-01-01"
    end_date = "2026-04-30" # Adjust to current date
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    closes = data['Close'].copy()
    volumes = data['Volume']['MSFT'].copy() # We only need MSFT volume
    open = data['Open']['MSFT'].copy()
    high = data['High']['MSFT'].copy()
    low = data['Low']['MSFT'].copy()
    
    df = pd.DataFrame(index=closes.index)
    df['msft_close'] = closes['MSFT']
    df['msft_volume'] = volumes
    df['msft_open'] = open
    df['msft_high'] = high
    df['msft_low'] = low
    df['vix_close'] = closes['^VIX']
    df['tnx_yield'] = closes['^TNX']
    df['qqq_close'] = closes['QQQ']
    df['spy_close'] = closes['SPY']
    df['msft_return'] = np.log(df['msft_close'] / df['msft_close'].shift(1))
    df = df.dropna()

    # Prepare for Turso ingestion
    records = []
    for date, row in df.iterrows():
        # Format date to standard string YYYY-MM-DD
        date_str = date.strftime('%Y-%m-%d')
        records.append((
            date_str, 
            float(row['msft_close']), 
            int(row['msft_volume']), 
            float(row['msft_open']),
            float(row['msft_high']),
            float(row['msft_low']),
            float(row['msft_return']), 
            float(row['vix_close']), 
            float(row['tnx_yield']),
            float(row['qqq_close']),
            float(row['spy_close'])
        ))
        
    # Batch Insert
    print(f"Pushing {len(records)} trading days to Turso...")
    conn.cursor.executemany("""
        INSERT OR REPLACE INTO market_data 
        (trading_date, msft_close, msft_volume, msft_open, msft_high, msft_low, msft_return, vix_close, tnx_yield, qqq_close, spy_close) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    print("Market and Macro data successfully stored!")



# Financial Filings Pipeline
def map_msft_fiscal_metadata(filing_date, form_type):
    """
    Helper function to determine fiscal year and period based on MSFT's unique fiscal calendar and filing type.

    Args:
        filing_date : datetime
        form_type : str

    Returns:
        fiscal_year, fiscal_period
    """
    month, year = filing_date.month, filing_date.year
    if "10-K" in form_type: return year, "FY"
    if month in [10, 11, 12]: return year + 1, "Q1"
    if month in [1, 2, 3]: return year, "Q2"
    if month in [4, 5, 6]: return year, "Q3"
    return None, None

def calculate_Q4(raw_df):
    """
    In order to provide quarterly update for the model, a manual process of calculating financial values
    using 10-k and 10-Q form has been applied. 

    Args:
        raw_df: dataframe containing all the financial values for all 3 quarters. 
    returns:
        dataframe containing q4 financial values
    Logic: 
    For each year, we get quarter and FY specific records 
    and we subtract, for example Q4 (income) = FY (income) - (All 3 quarter income)
    
    """
    final_records = []
    for year in sorted(raw_df['fiscal_year'].unique()):
        year_data = raw_df[raw_df['fiscal_year'] == year]
        qs = year_data[year_data['fiscal_period'].isin(['Q1', 'Q2', 'Q3'])]
        fy = year_data[year_data['fiscal_period'] == 'FY']
        
        for _, row in qs.iterrows():
            final_records.append(row.to_dict())
            
        if not fy.empty:
            if len(qs) == 3:
                fy_row = fy.iloc[0].copy()
                fy_row['revenue'] -= qs['revenue'].sum()
                fy_row['net_income'] -= qs['net_income'].sum()
                fy_row['op_cash_flow'] -= qs['op_cash_flow'].sum()
                fy_row['capex'] -= qs['capex'].sum()
                fy_row['fiscal_period'] = 'Q4'
                fy_row['filing_id'] = fy_row['filing_id'].replace('FY', 'Q4')
                final_records.append(fy_row.to_dict())
            else:
                print(f"  Alert: FY {year} missing Q1/Q2/Q3 filing. Cannot safely calculate Q4.")
        else:
            print(f"  Info: FY {year} is incomplete (No 10-K yet). Skipping Q4 calculation.")

    clean_df = pd.DataFrame(final_records)
    clean_df['filing_date'] = clean_df['filing_date'].astype(str)
    clean_df['effective_date'] = clean_df['effective_date'].astype(str)
    return clean_df.sort_values(['fiscal_year', 'fiscal_period'], ascending=[False, False]).reset_index(drop=True)

def ingest_financials(conn, ticker="MSFT"):
    
    """
    Ingest all Financial raw values using Edgar API. We use API obj() object to get 
    financials, balance sheet and cash flow statements. 
    All the data is stored at a quarter level  (inferred Q4 data) and stored inside DB
    
    """
    print("\n--- Starting Financials Ingestion ---")
    company = Company(ticker)
    filings = company.get_filings(form=["10-K", "10-Q"]).filter(filing_date="2019-01-01:")
    
    raw_data = []
    for filing in filings:
        try:
            f_year, f_period = map_msft_fiscal_metadata(filing.filing_date, filing.form)            
            f_id = f"{ticker}_{f_year}_{f_period}_FILING"
            eff_date = (filing.filing_date + timedelta(days=1)).strftime('%Y-%m-%d')
            report = filing.obj()

            inc_df = report.financials.income_statement().to_dataframe()
            rev = inc_df[inc_df['standard_concept']=='Revenue'].iloc[0, 3]
            ni = inc_df[inc_df['standard_concept']=='NetIncome'].iloc[0, 3]

            cf_df = report.financials.cashflow_statement().to_dataframe()
            ocf = cf_df[cf_df['label']=='Net cash from operations'].iloc[0, 3]
            capex = cf_df[cf_df['standard_concept']=='CapitalExpenses'].iloc[0, 3]

            bal_df = report.financials.balance_sheet().to_dataframe()
            cash = bal_df[bal_df['standard_concept']=='CashAndMarketableSecurities'].iloc[0, 3]
            assets = bal_df[bal_df['standard_concept']=='Assets'].iloc[0, 3]
            liab = bal_df[bal_df['standard_concept']=='Liabilities'].iloc[0, 3]
            eq = bal_df[bal_df['standard_concept']=='AllEquityBalance'].iloc[0, 3]
            
            raw_data.append({
                'filing_id': f_id, 'ticker': ticker, 'filing_date': filing.filing_date,
                'effective_date': eff_date, 'fiscal_year': f_year, 'fiscal_period': f_period,
                "revenue": float(rev), "net_income": float(ni), "op_cash_flow": float(ocf),
                "capex": float(capex), "cash_eq": float(cash), "total_liability": float(liab),
                "total_assets": float(assets), "total_equity": float(eq)
            })
        except Exception as e:
            print(f"Skipping {filing.form} for {filing.filing_date}: {e}")
            continue
            
    final_df = calculate_Q4(pd.DataFrame(raw_data))
    records = final_df.values.tolist()
    conn.cursor().executemany(
        "INSERT OR REPLACE INTO financial_filings_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
        records
    )
    conn.commit()
    print(f"Stored {len(records)} financial quarters to Turso.")


#SEC (MD&A and Risk Factors) Pipeline
def semantic_sentence_splitter(text, max_chars=1500):
    """
    Helper function to ensure sentence based chunking of sec filings and transcripts
    and 512 token limit for finbert.  

    Args:
        text: "Msft Q3 revenue is.. " 
        max_chars: "maximum character length for each chunk, default set to ~1500 to stay within 512 token limit of finbert"
    Returns:
        list of chunks 
    """
    if not text: return []
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def isolate_mda_blocks(raw_mda_text):
    """
   From MD&A raw text, isolate the most relevant sections that discuss economic conditions,\
       fiscal year comparisons, and planned uses of capital.
       This is done by identifying key section headers and extracting text until we hit a new section or irrelevant content.
       This helps focus our analysis on the most forward-looking and decision-useful narrative content for our volatility prediction model.
       We focus on sections that discuss economic conditions, fiscal year comparisons, and planned uses of capital,
       as these are most likely to contain insights about future risks and opportunities that could impact stock volatility.
       By filtering out less relevant sections (like detailed metrics or critical accounting policies),
       we can reduce noise and improve the signal quality for our NLP analysis.
    Args:
        raw_mda_text: "Full MD&A text"

    Returns:
        A string containing only the most relevant MD&A sections for analysis.
    """
    if not raw_mda_text: return ""
    target_regex = re.compile(r'Economic Conditions|Fiscal Year \d{4} Compared|Three Months Ended.*?Compared with|Other Planned Uses', re.IGNORECASE)
    stop_regex = re.compile(r'Metrics|Six Months Ended|Nine Months Ended|OTHER INCOME|INCOME TAXES|NON-GAAP|CRITICAL ACCOUNTING|STATEMENT OF MANAGEMENT', re.IGNORECASE)
    
    lines = raw_mda_text.replace('\r\n', '\n').split('\n')
    extracted = []
    is_reading = False 
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if target_regex.search(line_str):
            is_reading = True
        if stop_regex.search(line_str):
            is_reading = False
        if is_reading:
            extracted.append(line_str)
    return "\n".join(extracted)

def clean_mda_narrative(raw_text):
    """
    Extracts pure narrative from raw MD&A  blocks 
    by aggressively filtering out tabular data, financial headers, and numerical values related to tables.

    Because SEC tables lose their HTML structure during text extraction, this function 
    relies on density based estimate to differentiate between narrative NLP signal and 
    numerical noise.
    Args:
        raw_text (str): The raw, unparsed text extracted from the SEC filing.

    Returns:
        str: The cleaned narrative string,       
        
    Filtering Criteria Applied:
        1. Word Matching: Drops known table headers/footers (e.g., 'unaudited', 'in millions').
        2. Regex Matching: Drops standard SEC comparative headers (e.g., 'Fiscal Year... Compared with').
        3. Alphabetic Density: Drops lines where less than 50% of the characters are letters 
           (effectively destroying rows of financial tables full of numbers and $ signs).
        4. Word Count: Drops orphaned lines with fewer than 5 words to remove lingering sub-headers.    
        """
        
        
    if not raw_text:
        return ""
    
    lines = raw_text.replace('\r\n', '\n').split('\n')
    clean_lines = []
    words_to_remove = {'three months ended', 'percentage change', 'in millions', 'unaudited', 'total revenue', 'gross margin', 'net income', 'operating expenses'}
    comp_regex = r'(?:Fiscal Year|Three Months Ended|Six Months Ended).*?Compared with'

    for line in lines:
        line_str = line.strip()
        if not line_str:
            clean_lines.append("")
            continue
        if line_str.lower() in words_to_remove or re.search(comp_regex, line_str, re.IGNORECASE): continue
        
        t_chars = len(line_str)
        if t_chars > 0 and (sum(c.isalpha() for c in line_str) / t_chars) < 0.5: continue  # Drop lines where less than 50% of characters are letters (likely tables)
        if len(line_str.split()) < 5: continue
        clean_lines.append(line_str)

    return re.sub(r'\n{3,}', '\n\n', "\n".join(clean_lines)).strip()



def process_filing_to_chunks(filing, ticker, f_date):
    """
    For each filing (10-K, 10-Q) this function extracts md&a and risk section, cleans them for input to finbert model. 
    
    
    """
    try:
        doc = filing.obj()
    except Exception:
        return []

    form = filing.form
    doc_id = f"{ticker}_{form.replace('-', '')}_{str(f_date).replace('-', '')}"
    mda_raw, risk_raw = "", ""
    
    if form == "10-K":
        mda_raw = doc.management_discussion
        risk_raw = doc.risk_factors
    elif form == "10-Q":
        mda_raw = doc['Part I, Item 2']
        risk_raw = doc['Part II, Item 1A']
    
    chunks_to_db = []

    # Process MD&A
    if mda_raw:
        routed_mda = isolate_mda_blocks(mda_raw)
        clean_mda = clean_mda_narrative(routed_mda) # BUG FIX: Assigned output to clean_mda
        mda_chunks = semantic_sentence_splitter(clean_mda, max_chars=1500)
        for i, content in enumerate(mda_chunks):
            chunks_to_db.append((f"{doc_id}_MDA_{i}", doc_id, ticker, str(f_date), 'MDA', i, content, 0.0))

    # Process Risk Factors
    if risk_raw:
        if form == "10-Q" and len(risk_raw) < 300 and "no material changes" in risk_raw.lower():
            chunks_to_db.append((f"{doc_id}_RISK_0", doc_id, ticker, str(f_date), 'RiskFactorsUpdate', 0, "No material changes to risk factors.", 0.0))
        else:
            clean_risk = re.sub(r'\s+', ' ', risk_raw).strip()
            risk_chunks = semantic_sentence_splitter(clean_risk, max_chars=1500)
            item_label = 'RiskFactors' if form == '10-K' else 'RiskFactorsUpdate'
            for i, content in enumerate(risk_chunks):
                chunks_to_db.append((f"{doc_id}_{item_label}_{i}", doc_id, ticker, str(f_date), item_label, i, content, 0.0))

    return chunks_to_db

def ingest_sec_chunks( conn, ticker="MSFT"):
    """
    Manages the extraction, cleaning, and chunking of critical narrative sections 
    (MD&A and Risk Factors) from raw SEC 10-K and 10-Q filings.

    Uses raw SEC Edgar data for downstream  NLP pipeline. 
    It dynamically routes the document parsing based on the filing type, passes the raw 
    text through aggressive cleaning criteria, and splits the  narrative into 
    tokenization-safe chunks formatted for Database insertion and downstream sentiment analysis.

    Args:
            conn: Db connection object
            ticker: str
   
    """
    print("\n--- Starting SEC MD&A and Risk Extraction ---")

    company = Company(ticker)
    filings = company.get_filings(form=["10-K", "10-Q"]).filter(filing_date="2018-01-01:")
    
    all_chunks = []
    for filing in filings:
        chunks = process_filing_to_chunks(filing, ticker, filing.filing_date)
        all_chunks.extend(chunks)
        
    conn.cursor().executemany(
        "INSERT OR REPLACE INTO sec_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
        all_chunks
    )
    conn.commit()
    print(f"Stored {len(all_chunks)} SEC text chunks to Turso.")


#Transcripts Pipeline
def split_transcript_by_turns(turns):
    """
    Helper Function to split transcript into prepared and Question and answer section.
    Uses regex and word matching logic.
    
    args: turn (json object having speaker_name, text and title)
    returns prepped_text and QAA_text

    

    """
    prepped_list, qa_list = [], []
    is_qa_started = False
    qa_markers = ["q and a", "q&a", "q and a"]

    for turn in turns:
        speaker_name = turn.speaker_info.name 
        speaker_title = turn.speaker_info.title  
        text = turn.text if turn.text else ""
        
        if not is_qa_started and any(marker in text.lower() for marker in qa_markers):
            is_qa_started = True
            continue 

        entry = {"speaker_name": speaker_name, "speaker_title": speaker_title, "text": text}
        qa_list.append(entry) if is_qa_started else prepped_list.append(entry)

    return prepped_list, qa_list

def ingest_transcripts(conn, ticker="MSFT"):
    """
    Uses Earnings Call API to fetch transcripts for a ticker. 
    We grab transcripts at level 2 which gives us speaker level distinction. The transcripts are 
    separated into prepped and qa for sentiment analysis.   

    """
    print("\n--- Starting Earnings Call Transcripts Ingestion ---")
    company = earningscall.get_company(ticker)
    historical_data = []
    start_date = datetime(2019, 1, 1)

    for event in company.events():
        event_date = event.conference_date.replace(tzinfo=None)
        if not (start_date <= event_date <= datetime.today()):
            continue

        effective_date = (event.conference_date + timedelta(days=1)).strftime('%Y-%m-%d')
        f_year = event.year
        f_period = f"Q{event.quarter}" if event.quarter < 4 else "FY"
        t_id = f"{ticker}_{f_year}_{f_period}_TRANSCRIPT"
        
        try:
            transcript = company.get_transcript(event=event, level=2)
            if not transcript or not transcript.speakers: continue
            
            prepped, qa = split_transcript_by_turns(transcript.speakers)
            
            historical_data.append((
                t_id, ticker, event.conference_date.strftime('%Y-%m-%d %H:%M:%S'), 
                effective_date, f_year, f_period, json.dumps(prepped), json.dumps(qa)
            ))
        except Exception as e:
            print(f"Skipping Q{event.quarter} {event.year}: {e}")

    conn.cursor().executemany(
        "INSERT OR REPLACE INTO transcripts (transcript_id, ticker, event_timestamp, effective_date, fiscal_year, fiscal_period, content_prepped, content_qa) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
        historical_data
    )
    conn.commit()
    print(f" Stored {len(historical_data)} transcripts to Turso.")




# Final Script to run all ingest functions. 
if __name__ == "__main__":
    print("Initializing ETL Pipeline...")
    
    # 1. Setup SEC Identity (Required before instantiating Company())
    setup_sec_api()
    
    # 2. Connect to Database
    db_conn = get_db_connection()
    
    if db_conn:
        
        create_tables(db_conn)
        
        # 3. Run Pipelines
        ingest_market_data(db_conn)
        ingest_financials(db_conn, ticker="MSFT")
        ingest_sec_chunks(db_conn, ticker="MSFT")
        ingest_transcripts(db_conn, ticker="MSFT")
        
    print(" Full Database Build Complete!")