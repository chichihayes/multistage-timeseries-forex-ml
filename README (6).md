# forex-intelligence-ml-pipeline

A multi-stage machine learning pipeline for EUR/USD directional forecasting, anomaly detection, and signal aggregation. Built end-to-end in Python, the system combines deep learning forecasting models, statistical anomaly detection, PCA-based correlation analysis, NLP sentiment classification, and LLM-assisted signal synthesis into a single structured analytical workflow.

This is not a research prototype. Every stage produces a saved output that feeds the next, and the final stage delivers a confidence-scored directional call with a full audit trail.

---

## What the pipeline does

The pipeline runs in eleven sequential stages:

1. Data collection from two sources — Yahoo Finance (2025-2026) and Twelve Data API (2024) — covering approximately two years of hourly EUR/USD OHLC data
2. Data merging and cleaning — deduplication, timezone normalisation, schema alignment, and sanity checks across both sources
3. Feature engineering — construction of 20 technical indicators including RSI, Bollinger Bands, ATR, EMA, ROC, and momentum features
4. Feature selection — Lasso regression with TimeSeriesSplit cross-validation to identify which features carry predictive signal without look-ahead bias
5. Architecture benchmarking — training and evaluation of five deep learning architectures (LSTM, Deep LSTM, GRU, Bidirectional LSTM, CNN-LSTM) on identical splits using RMSE, MAE, directional accuracy, and Sign AUC
6. LSTM Model 1 — predicts the exact EUR/USD close price one hour ahead
7. LSTM Model 2 — predicts cumulative pip movement over the next five hours
8. Anomaly detection — dual-method pipeline using Z-score flagging (3.0 sigma threshold on returns) and Isolation Forest (200 estimators, 1% contamination) across the full price history
9. Correlation and eigenvalue analysis — rolling 168-hour PCA across six major currency pairs (EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD) to track market concentration and structural regime shifts
10. News sentiment analysis — real-time financial news classification across bullish and bearish categories using the Alpha Vantage NEWS_SENTIMENT API
11. LLM signal aggregation — Gemini 2.5 Flash via OpenRouter synthesises all pipeline outputs into a structured directional call with confidence scoring

---

## Architecture overview

```
Yahoo Finance API  ─┐
                    ├─ Merge & Clean ─ Feature Engineering ─ Lasso Selection
Twelve Data API    ─┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          │                    │                    │
                    LSTM Model 1h       LSTM Model 5h       Architecture
                    (close price)     (cumulative pips)    Benchmark (5 models)
                          │                    │
                          └─────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
       Anomaly Detection    Correlation / PCA      News Sentiment
       (Z-score + IForest)  (6 pairs, rolling)    (Alpha Vantage)
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                         Signal Aggregation
                         (confidence score)
                                    │
                         LLM Final Analysis
                         (Gemini 2.5 Flash)
                                    │
                    BUY / SELL call with reasoning
```

---

## Stack

| Component | Tool |
|---|---|
| Data collection | yfinance, Twelve Data API, Alpha Vantage API |
| Data processing | Pandas, NumPy |
| Feature selection | Scikit-learn (LassoCV, TimeSeriesSplit) |
| Deep learning | TensorFlow / Keras |
| Anomaly detection | Scikit-learn (IsolationForest), SciPy |
| Correlation analysis | NumPy (PCA via covariance decomposition) |
| Sentiment analysis | Alpha Vantage NEWS_SENTIMENT API |
| LLM integration | Gemini 2.5 Flash via OpenRouter |
| Visualisation | Matplotlib |
| Environment | Google Colab |

---

## Setup

### 1. Open in Google Colab

The notebook is designed to run in Google Colab. Open the `.ipynb` file directly from GitHub using the Colab badge or by uploading it manually.

### 2. Set API keys

In Colab, add the following to your Secrets (the key icon in the left sidebar):

```
OPENROUTER_API_KEY   your OpenRouter key (get one free at openrouter.ai)
```

The Twelve Data and Alpha Vantage API keys are set inline in the notebook. Replace the placeholder values with your own free-tier keys from:
- twelvedata.com
- alphavantage.co

### 3. Run all cells in order

Each stage saves its outputs as CSV files which are consumed by the next stage. Run the notebook top to bottom. Do not skip sections.

---

## Outputs

Each stage saves a named CSV file to the working directory:

| File | Contents |
|---|---|
| `eurusd_clean.csv` | Merged and cleaned OHLC dataset |
| `eurusd_features.csv` | Full feature-engineered dataset (20 features) |
| `lasso_feature_importance.csv` | Feature coefficients from Lasso selection |
| `lstm_test_predictions.csv` | LSTM Model 1h test set predictions vs actuals |
| `t5_predictions.csv` | LSTM Model 5h cumulative pip predictions |
| `eurusd_anomaly_flags.csv` | Per-row anomaly flags (Z-score and Isolation Forest) |
| `anomaly_signal.csv` | Aggregated anomaly signal for last 24 hours |
| `correlation_signal.csv` | PCA eigenvalue and dominant pair outputs |
| `news_sentiment_articles.csv` | Raw classified news articles |
| `sentiment_signal.csv` | Aggregated sentiment score and label |
| `aggregated_prediction.csv` | Combined signal record with confidence score |
| `llm_conversation_log.csv` | LLM query and full structured response |
| `signal_aggregation_prediction.png` | Visualisation of prediction vs recent price history |

---

## Model evaluation

The five architectures benchmarked in Stage 5 are compared on the same train/test split across four metrics:

- RMSE (root mean squared error on close price)
- MAE (mean absolute error on close price)
- Directional accuracy (percentage of correct up/down calls)
- Sign AUC (AUC of the directional classification)

The best-performing architecture on directional accuracy is carried forward into Stage 6 as LSTM Model 1h.

---

## LLM final analysis

The final stage passes all pipeline outputs to Gemini 2.5 Flash via OpenRouter with a structured system prompt that requires the model to reason through each signal in a fixed format:

```
SIGNAL ANALYSIS      — individual assessment of each pipeline output
WEIGHT OF EVIDENCE   — whether signals agree or conflict, and which carry more weight
FINAL CALL           — BUY or SELL with one sentence of committed reasoning
```

The model is explicitly instructed not to hedge. A definitive directional call is required.

---

## Limitations

- The pipeline is built for analysis and research purposes. It is not connected to a live trading execution system.
- Twelve Data free tier limits request frequency. The notebook includes 15-second sleep intervals between quarterly data pulls to stay within the rate limit.
- Forex markets are affected by macroeconomic events that no model can anticipate. Anomaly detection flags abnormal price behaviour but cannot predict the cause.
- All analysis is academic. This is not financial advice.

---

## Repository structure

```
forex-intelligence-ml-pipeline/
├── forex_intelligence_ml_pipeline.ipynb    main notebook
├── README.md
└── requirements.txt                        optional — for local Python environments
```

---

## Requirements (for local use)

If running locally rather than in Colab:

```
yfinance
requests
pandas
numpy
scikit-learn
tensorflow
matplotlib
scipy
```

Install with:

```bash
pip install yfinance requests pandas numpy scikit-learn tensorflow matplotlib scipy
```

---

## Author

Hayes Abayomi Chichieze Israel
github.com/chichihayes
