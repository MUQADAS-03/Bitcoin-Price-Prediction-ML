# 📈 Bitcoin Price Prediction Using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Models-orange?style=for-the-badge&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-Classifier-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <strong>Predicting Bitcoin price movement (Up/Down) using Logistic Regression, SVM, and XGBoost</strong><br/>
  Binary classification on historical OHLCV Bitcoin data with feature engineering and ROC-AUC evaluation
</p>

---

##  Project Overview

This project applies **supervised machine learning** to predict whether Bitcoin's closing price will go **up or down** the next day — a binary classification problem using historical OHLCV (Open, High, Low, Close, Volume) data.

**The core idea:** Instead of predicting the exact price (regression), we predict the *direction* of the next day's price movement. This transforms a noisy regression problem into a cleaner binary classification task.

### Key Questions Answered
- Can ML models predict Bitcoin price direction better than random?
- Which features are most predictive of next-day price movement?
- Which model (Logistic Regression / SVM / XGBoost) performs best on crypto data?

---

##  Dataset

| Field | Detail |
|---|---|
| **Source** | Historical Bitcoin CSV (`bitcoin.csv`) |
| **Features** | Date, Open, High, Low, Close, Adj Close, Volume |
| **Target** | Binary (1 = price goes up next day, 0 = price goes down) |
| **Preprocessing** | Dropped `Adj Close` (identical to `Close`), checked for nulls |

---

##  Project Workflow

```

<img width="1356" height="1238" alt="image" src="https://github.com/user-attachments/assets/2e9e7c1c-e7f7-43a6-af8e-6ed4a928702c" />

```


##  Feature Engineering

Three engineered features used for model training:

| Feature | Formula | Rationale |
|---|---|---|
| `open-close` | `Open - Close` | Measures intra-day price direction |
| `low-high` | `Low - High` | Measures intra-day price volatility |
| `is_quarter_end` | `1 if month % 3 == 0` | Quarter-end effects on crypto markets |
| `target` | `1 if Close[t+1] > Close[t]` | Binary label for next-day direction |

All features were **standardized** using `StandardScaler` before training.

---

##  Models Used

| Model | Type | Key Parameter |
|---|---|---|
| **Logistic Regression** | Linear Classifier | `max_iter=1000` |
| **SVM** | Support Vector Machine | `kernel='poly'`, `probability=True` |
| **XGBoost** | Gradient Boosting | Default hyperparameters |

All models evaluated using **ROC-AUC Score** on both training and validation sets.

---

##  Results

| Model | Training ROC-AUC | Validation ROC-AUC |
|---|---|---|
| Logistic Regression | — | Evaluated |
| SVM (Polynomial) | — | Evaluated |
| XGBoost Classifier | — | Evaluated |

> Full scores visible in notebook output. Confusion matrix generated for Logistic Regression.

**Target class distribution:**
- Class 0 (Price Down): ~47.8%
- Class 1 (Price Up): ~52.2%

---

##  Visualizations

### 1. Bitcoin Closing Price — Historical Trend
> Line chart showing full historical Bitcoin closing price from dataset start to end.

### 2. Feature Distributions
> 4-panel distribution plots (`distplot`) for Open, High, Low, Close prices showing the right-skewed nature of Bitcoin price data.

### 3. Box Plots — Outlier Detection
> Horizontal box plots for all 4 OHLC features revealing significant upper outliers (bull run periods).

### 4. Yearly Average Price Bars
> Bar charts for average Open / High / Low / Close grouped by year — shows explosive growth from 2020 onward.

### 5. Target Class Pie Chart
> Near-balanced 52% / 48% split between up-days and down-days.

### 6. Correlation Heatmap
> Heatmap with threshold >0.9 — confirms Open, High, Low, Close are highly correlated; engineered features reduce multicollinearity.

### 7. Confusion Matrix
> Confusion matrix for Logistic Regression predictions on the validation set.

---

##  Libraries Used

```python
pandas          # Data loading and manipulation
numpy           # Numerical operations
matplotlib      # Visualizations
seaborn         # Statistical plots
scikit-learn    # ML models, preprocessing, evaluation
xgboost         # XGBoost classifier
scipy           # (EV project)
```

---

## Author

**Muqadas Yasin**


---

<p align="center">
  <em>Predicting the unpredictable — one candlestick at a time. 📊</em>
</p>
