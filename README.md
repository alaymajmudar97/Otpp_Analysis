# MSFT 21-Day Volatility Forecasting Project

##  Summary

This project asks a simple question: **Can we forecast Microsoft's (MSFT) future volatility more accurately if we look at what management is actually saying, rather than just looking at stock price movements?** 
I built a machine learning pipeline that combines traditional market signals, macroeconomic indicators, and accounting data with sentiment extracted directly from SEC filings and earnings call transcripts. The result is a time-aware LightGBM model that predicts the next 21 trading days of realized volatility. 

---

## Project Objective

The goal is to predict **21-day forward realized volatility** for MSFT, acting as a forward-looking risk estimate. I am not trying to predict stock price direction or next-day returns. Core hypothesis is that unstructured sentiment features—when combined with traditional quantitative market signals—can incrementally improve near-term risk forecasting

## How the Pipeline Works

The pipeline operates in six main stages:

1. **Ingest:** Download raw structured (market/financial) and unstructured (textual) data into a local database.
2. **Process Text:** Convert raw SEC filings and earnings transcripts into point-in-time sentiment scores.
3. **Feature Engineering:** Build lagged market, macro, accounting, and event-recency features.
4. **Target Definition:** Calculate the 21-day forward rolling volatility target.
5. **Modeling:** Train and validate a LightGBM regression model using strict time-series cross-validation.
6. **Evaluation:** Assess performance and interpret model decisions using SHAP and business-oriented metrics.

---

## Repository Structure

| File | Role | Description |
| :--- | :--- | :--- |
| `database_builder.py` | Data Ingestion | Pulls market data, SEC filings, and earnings call transcripts into Turso database tables. |
| `feature_builder.py` | Feature Engineering | Transforms raw data into NLP, market, macro, accounting, and temporal recency features. |
| `df_features.csv` | Modeling Data | The final daily feature matrix containing all predictors and the target variable. |
| `Volatility-Forecasting.pptx` | Presentation | Details the objective, architecture, leakage controls, validation approach, and final results. |

---

## Data Sources & Inputs

### 1. Market and Macro Data
 Fetches historical data starting from 2018 for MSFT, the VIX, a 10-year Treasury yield proxy (^TNX), QQQ, and SPY using the `yfinance` library. 

These raw numbers are transformed into risk-sensitive features, such as rolling volatility, cross-asset relative returns, yield changes, and volume-based stress indicators. You will see features like `vixlevel`, `volrolling21d`, `qqqvol21d`, and `volsurge` in the final dataset.

### 2. Financial Filings (Accounting)
The pipeline extracts 10-K and 10-Q filings from the SEC, pulling raw line items like revenue, net income, and operating cash flow. Because raw financial numbers naturally grow over time, I normalize them into stationary accounting ratios like Return on Assets (`roa`), `debttoasset`, and Free Cash Flow margin (`fcfmargin`) for modeling.

### 3. SEC Filing Text
Extract the "Management Discussion and Analysis" and "Risk Factor" sections from the SEC filings. The text goes through a rigorous cleaning process: I filter out tables, remove boilerplate, and chop the text into sentence-safe chunks that fit into our NLP model. These chunks are scored for sentiment and aggregated into daily document-level features. 

### 4. Earnings Call Transcripts
using earnings call API,  pulled Microsoft earnings call transcripts from 2019 onwards (API limit No data before 2019).  splitted the text into two distinct parts: **Management Prepared Remarks** and **Analyst Q&A**. These sections are scored independently because scripted management messaging and unscripted Q&A dynamics often signal different types of risk.

---



## NLP Methodology

### FinBERT Scoring
Uses the `ProsusAI/finbert` model to extract sentiment. FinBERT is specifically pre-trained on financial text, meaning it understands the nuances of corporate filings much better than a generic sentiment analyzer. 

For every text chunk, I calculate net sentiment (positive minus negative), average neutrality, and sentiment dispersion. This consistent scoring logic is applied across all text sources, providing a uniform set of features for the model.

---

## The Target Variable

The target is **21-day forward realized volatility**. We calculate this by taking the standard deviation of MSFT's daily returns over a rolling 21-day window, annualized, and shifted forward to ensure zero look-ahead bias.

### Feature Groupings Overview
* **Accounting:** `roa`, `cashcoverage`, `debttoasset`, `fcfmargin`
* **SEC Text:** `MDA_Sentiment`, `Risk_Factor_Sentiment`
* **Earnings Call Text:** `preppedsentiment`, `qasentiment`, `qadispersion`
* **Event Recency:** `dayssinceearnings`, `dayssincefiling`
* **Market Context:** `msftreturn`, `qqqreturn`, `msftvsmarket`
* **Macro Regime:** `vixlevel`, `vix5dtrend`, `yield10ylevel`
* **Volatility Signals:** `volrolling21d`, `volsurge`, `distfromma200`
* **Target Label:** `targetvol21d`

---

## Model Performance

The LightGBM model gives the following key metrics on the unseen test set:

| Metric | Result |
| :--- | :--- |
| **Sign Accuracy** | 82% |
| **Test MAE** | 7.26% |
| **Test RMSE** | 9.28% |
| **Train RMSE** | 7.29% |
| **Mean Bias Error** | +0.0084 |


---

## Why This Pipeline Works

* **Mixed Data Sources:** Volatility is driven by both market mechanics and human narrative. Combining price action with management tone provides a much richer signal.
* **Realistic Time Handling:** Daily market data and quarterly earnings update at very different speeds. The pipeline preserves original event timing and uses "days since" recency features to manage this asynchronous flow smoothly.
* **Integrity First:** The model's foundation is built on preventing data leakage. Forward targets and validation gaps ensure the model actually learns rather than just memorizes.
* **Interpretability:** By using LightGBM and SHAP values, we avoid the "black box" problem. We can clearly explain *why* the model made a specific risk forecast based on economic and sentiment drivers.

---

## Known Limitations

* **Single Asset Focus:** The model is currently tuned specifically for MSFT. Cross-sectional generalization across other equities has not yet been tested.
* **Signal Frequency:** Earnings and filings only happen quarterly. While market data fills the daily gaps, the text signals update infrequently.
* **Extreme Tail Risk:** As shown by the gap between MAE and RMSE, the model—like most quantitative systems—struggles to perfectly predict the exact magnitude of sudden, massive "Black Swan" volatility spikes.

---

## Future Roadmap

To improve the signal and make the model more responsive, the next phases of development will focus on:
* **Higher Frequency Text:** Integrating 8-K filings and weekly financial news to catch material events between quarterly earnings.
* **Targeted Topic Analysis:** Moving beyond general sentiment to zero-shot classification, allowing us to track specific themes like "AI Infrastructure," "Cloud Growth," or "Regulatory Risk."
* **Sector Awareness:** Tracking sentiment for peer competitors and the broader tech sector to distinguish between Microsoft-specific issues and general market contagion.