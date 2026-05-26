"""
Stock Trend Prediction Pipeline — Corrected Version
=====================================================
Changes applied:
  1. 5-day target horizon (shift -5 instead of -1)
  2. Company-wise chronological 80/20 train-test split
  3. Sentiment shifted by 1 day per company (temporal leakage fix)
  4. All 4 models preserved (RF, XGBoost, SVM, Stacking)
  5. 3 new features: Price_Distance, Range_Pct, Volume_Ratio
  6. Raw price columns removed from features
  7. Validation diagnostics printed
  8. TimeSeriesSplit cross-validation per company
"""

# ============================================================
# CELL 1 — Data Download & Concatenation (unchanged)
# ============================================================
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

stocks = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]

all_data = []

for stock in stocks:
    df = yf.download(
        stock,
        start="2015-01-01",
        end="2025-01-01"
    )

    # Flatten multi-level columns returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # reset date from index into column
    df.reset_index(inplace=True)

    # add company column
    df["Company"] = stock

    # keep only needed columns
    df = df[[
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Company"
    ]]

    all_data.append(df)

# vertical concatenation
fdf = pd.concat(all_data, ignore_index=True)

print("=" * 60)
print("CELL 1 — Raw data loaded")
print(f"Shape: {fdf.shape}")
print(fdf.head())


# ============================================================
# CELL 2 — Feature Engineering (existing + 3 NEW features)
# ============================================================

# --- Existing features ---
fdf["Return"] = fdf.groupby("Company")[
    "Close"
].pct_change()

fdf["MA10"] = fdf.groupby("Company")[
    "Close"
].transform(lambda x: x.rolling(10).mean())

fdf["MA50"] = fdf.groupby("Company")[
    "Close"
].transform(lambda x: x.rolling(50).mean())

fdf["Volatility"] = fdf.groupby(
    "Company"
)["Return"].transform(
    lambda x: x.rolling(10).std()
)

fdf["Momentum"] = fdf.groupby(
    "Company"
)["Close"].diff(5)

delta = fdf.groupby("Company")[
    "Close"
].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

fdf["RSI"] = 100 - (100 / (1 + rs))

fdf["Range"] = (
    fdf["High"] - fdf["Low"]
)

fdf["MA_Ratio"] = (
    fdf["MA10"] /
    fdf["MA50"]
)

# --- NEW Feature 5a: Price_Distance ---
# How far current price is from 50-day MA (relative)
fdf["Price_Distance"] = (fdf["Close"] - fdf["MA50"]) / fdf["MA50"]

# --- NEW Feature 5b: Range_Pct ---
# Daily range as a fraction of closing price
fdf["Range_Pct"] = (fdf["High"] - fdf["Low"]) / fdf["Close"]

# --- NEW Feature 5c: Volume_Ratio ---
# Volume relative to its own 20-day rolling average (per company)
fdf["Volume_Ratio"] = fdf.groupby("Company")["Volume"].transform(
    lambda x: x / x.rolling(20).mean()
)

print("\n" + "=" * 60)
print("CELL 2 — Features engineered (including 3 new)")
print(f"Columns: {list(fdf.columns)}")


# ============================================================
# CELL 3 — Drop NaNs from rolling windows, extract date parts
# ============================================================

fdf = fdf.dropna()

fdf["Month"] = fdf["Date"].dt.month
fdf["DayOfWeek"] = fdf["Date"].dt.dayofweek

print(f"\nAfter dropna: {fdf.shape}")


# ============================================================
# CELL 4 — 5-DAY TARGET HORIZON (Change #1)
# ============================================================
# OLD: fdf.groupby("Company")["Close"].shift(-1) > fdf["Close"]
# NEW: fdf.groupby("Company")["Close"].shift(-5) > fdf["Close"]

fdf["Target"] = (
    fdf.groupby("Company")[
        "Close"
    ].shift(-5) >
    fdf["Close"]
).astype(int)

# Drop rows where Target is NaN (last 5 rows per company)
# shift(-5) produces NaN for the last 5 rows of each company,
# and the comparison with NaN gives False → 0, which is incorrect.
# We must explicitly mark and drop these.
fdf["_target_valid"] = fdf.groupby("Company")["Close"].shift(-5).notna()
fdf = fdf[fdf["_target_valid"]].drop(columns=["_target_valid"])

print(f"\nAfter 5-day target creation: {fdf.shape}")
print(f"Target distribution:\n{fdf['Target'].value_counts(normalize=True)}")


# ============================================================
# CELL 5 — Company Encoding
# ============================================================
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

fdf["Company_Code"] = (
    encoder.fit_transform(
        fdf["Company"]
    )
)

print(f"\nCompany encoding: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")


# ============================================================
# CELL 6 — Sentiment Merge + Temporal Leakage Fix (Change #3)
# ============================================================

# Rename Date for merge
fdf.rename(columns={"Date": "date"}, inplace=True)

# Load sentiment
sentiment_df = pd.read_csv(
    r"C:\Users\arije\OneDrive\Desktop\resume projects\stock trend prediction\news_with_sentiment.csv"
)
sentiment_df = sentiment_df[
    ["date", "Company", "Sentiment"]
]

# Aggregate daily sentiment per company
daily_sentiment = (
    sentiment_df.groupby(
        ["date", "Company"]
    )["Sentiment"]
    .mean()
    .reset_index()
)

daily_sentiment["date"] = pd.to_datetime(
    daily_sentiment["date"]
)

# Merge
fdf = pd.merge(
    fdf,
    daily_sentiment,
    on=["date", "Company"],
    how="left"
)

# Fill missing sentiment with 0
fdf["Sentiment"] = fdf["Sentiment"].fillna(0)

# --- TEMPORAL LEAKAGE FIX (Change #3) ---
# Shift sentiment by 1 day within each company
# so we only use PAST sentiment, never same-day
fdf["Sentiment"] = fdf.groupby("Company")["Sentiment"].shift(1)
fdf.dropna(inplace=True)

print(f"\nAfter sentiment merge + shift: {fdf.shape}")


# ============================================================
# CELL 7 — Feature Selection (Change #6)
# ============================================================
# REMOVED: Open, High, Low, Close (raw prices — scale-dependent)
# REMOVED: Volume, MA10, MA50, Range (raw/absolute values)
# KEPT: all relative/normalized features

features = [
    "Return",
    "MA_Ratio",
    "Volatility",
    "Momentum",
    "RSI",
    "Range_Pct",
    "Price_Distance",
    "Volume_Ratio",
    "Sentiment",
    "Company_Code",
]

print(f"\nFeature list ({len(features)} features):")
for i, f in enumerate(features, 1):
    print(f"  {i}. {f}")


# ============================================================
# CELL 8 — Company-Wise Chronological Train-Test Split (Change #2)
# ============================================================

train_parts = []
test_parts = []

for company, group in fdf.groupby("Company"):
    group = group.sort_values("date")
    n = len(group)
    split = int(n * 0.8)
    train_parts.append(group.iloc[:split])
    test_parts.append(group.iloc[split:])

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

X_train = train_df[features]
y_train = train_df["Target"]
X_test = test_df[features]
y_test = test_df["Target"]

# --- VALIDATION DIAGNOSTICS (Change #7) ---
print("\n" + "=" * 60)
print("VALIDATION DIAGNOSTICS")
print("=" * 60)
print(f"Train size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")
print(f"\nTrain target distribution:")
print(y_train.value_counts(normalize=True).to_string())
print(f"\nTest target distribution:")
print(y_test.value_counts(normalize=True).to_string())
print(f"\nCompanies in TRAIN: {sorted(train_df['Company'].unique())}")
print(f"Companies in TEST:  {sorted(test_df['Company'].unique())}")

# Verify no company is missing
train_companies = set(train_df["Company"].unique())
test_companies = set(test_df["Company"].unique())
all_companies = set(fdf["Company"].unique())
assert train_companies == all_companies, f"Missing from train: {all_companies - train_companies}"
assert test_companies == all_companies, f"Missing from test: {all_companies - test_companies}"
print("\n✓ All companies present in both train and test sets")

print(f"\nDate ranges per company:")
for company in sorted(all_companies):
    tr = train_df[train_df["Company"] == company]
    te = test_df[test_df["Company"] == company]
    print(f"  {company}:")
    print(f"    Train: {tr['date'].min().date()} → {tr['date'].max().date()} ({len(tr)} rows)")
    print(f"    Test:  {te['date'].min().date()} → {te['date'].max().date()} ({len(te)} rows)")


# ============================================================
# CELL 9 — Scaling (SVM needs it)
# ============================================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# CELL 10 — Time-Series Cross-Validation Per Company (Change #8)
# ============================================================
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score

def per_company_tscv(model, fdf_subset, features, target_col="Target", n_splits=5, use_scaler=False):
    """
    Perform TimeSeriesSplit CV separately within each company,
    then aggregate scores across all folds from all companies.

    Returns dict with mean/std of accuracy and f1.
    """
    all_acc = []
    all_f1 = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for company, group in fdf_subset.groupby("Company"):
        group = group.sort_values("date")
        X_comp = group[features].values
        y_comp = group[target_col].values

        for train_idx, val_idx in tscv.split(X_comp):
            X_tr, X_val = X_comp[train_idx], X_comp[val_idx]
            y_tr, y_val = y_comp[train_idx], y_comp[val_idx]

            if use_scaler:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_val = sc.transform(X_val)

            from sklearn.base import clone
            m = clone(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)

            all_acc.append(accuracy_score(y_val, preds))
            all_f1.append(f1_score(y_val, preds, average="weighted"))

    return {
        "accuracy_mean": np.mean(all_acc),
        "accuracy_std": np.std(all_acc),
        "f1_mean": np.mean(all_f1),
        "f1_std": np.std(all_f1),
        "n_folds_total": len(all_acc),
    }


# ============================================================
# CELL 11 — Random Forest (preserved, same param grid)
# ============================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'random_state': [42]
}

# Use TimeSeriesSplit for CV inside RandomizedSearchCV
rf_search = RandomizedSearchCV(
    RandomForestClassifier(),
    rf_params,
    n_iter=20,
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=-1,
    random_state=42,
    verbose=1,
    scoring="accuracy"
)

rf = rf_search.fit(X_train, y_train)
print("Best RF Parameters:", rf.best_params_)
print("Best RF CV Score:", rf.best_score_)

rf_pred = rf.predict(X_test)
print(f"RF Holdout Accuracy: {accuracy_score(y_test, rf_pred):.4f}")

# Per-company TSCV for RF
print("\nPer-company TimeSeriesSplit CV for RF:")
rf_tscv = per_company_tscv(rf.best_estimator_, train_df, features, n_splits=5)
print(f"  Accuracy: {rf_tscv['accuracy_mean']:.4f} ± {rf_tscv['accuracy_std']:.4f}")
print(f"  F1:       {rf_tscv['f1_mean']:.4f} ± {rf_tscv['f1_std']:.4f}")
print(f"  Total folds: {rf_tscv['n_folds_total']}")


# ============================================================
# CELL 12 — XGBoost (preserved, same param grid)
# ============================================================
from xgboost import XGBClassifier

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

xgb_params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 2, 3, 4, 5],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0.8, 1.0, 1.2],
    'random_state': [42]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    xgb_params,
    n_iter=30,
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=-1,
    random_state=42,
    verbose=1,
    scoring="accuracy"
)

xgb = xgb_search.fit(X_train, y_train)
print("Best XGB Parameters:", xgb.best_params_)
print("Best XGB CV Score:", xgb.best_score_)

xgb_pred = xgb.predict(X_test)
print(f"XGB Holdout Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")

# Per-company TSCV for XGB
print("\nPer-company TimeSeriesSplit CV for XGBoost:")
xgb_tscv = per_company_tscv(xgb.best_estimator_, train_df, features, n_splits=5)
print(f"  Accuracy: {xgb_tscv['accuracy_mean']:.4f} ± {xgb_tscv['accuracy_std']:.4f}")
print(f"  F1:       {xgb_tscv['f1_mean']:.4f} ± {xgb_tscv['f1_std']:.4f}")
print(f"  Total folds: {xgb_tscv['n_folds_total']}")


# ============================================================
# CELL 13 — SVM (preserved, same param grid, uses scaled data)
# ============================================================
from sklearn.svm import SVC

print("\n" + "=" * 60)
print("TRAINING SVM")
print("=" * 60)

svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 1.0],
    'probability': [True],
    'random_state': [42]
}

svm_search = RandomizedSearchCV(
    SVC(),
    svm_params,
    n_iter=20,
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=-1,
    random_state=42,
    verbose=1,
    scoring="accuracy"
)

svm = svm_search.fit(
    X_train_scaled,
    y_train
)
print("Best SVM Parameters:", svm.best_params_)
print("Best SVM CV Score:", svm.best_score_)

svm_pred = svm.predict(X_test_scaled)
print(f"SVM Holdout Accuracy: {accuracy_score(y_test, svm_pred):.4f}")

# Per-company TSCV for SVM (with scaling)
print("\nPer-company TimeSeriesSplit CV for SVM:")
svm_tscv = per_company_tscv(svm.best_estimator_, train_df, features, n_splits=5, use_scaler=True)
print(f"  Accuracy: {svm_tscv['accuracy_mean']:.4f} ± {svm_tscv['accuracy_std']:.4f}")
print(f"  F1:       {svm_tscv['f1_mean']:.4f} ± {svm_tscv['f1_std']:.4f}")
print(f"  Total folds: {svm_tscv['n_folds_total']}")


# ============================================================
# CELL 14 — Stacking Ensemble (preserved)
# ============================================================
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

print("\n" + "=" * 60)
print("TRAINING STACKING ENSEMBLE")
print("=" * 60)

# Tune meta model
meta_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200, 500, 1000],
    'class_weight': ['balanced', None],
    'random_state': [42]
}

meta_search = RandomizedSearchCV(
    LogisticRegression(),
    meta_params,
    n_iter=15,
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Tuning meta model...")
meta_search.fit(
    rf_search.predict_proba(X_train),
    y_train
)
meta_model = meta_search.best_estimator_
print("Best Meta Model Parameters:", meta_search.best_params_)
print("Best Meta Model CV Score:", meta_search.best_score_)

# Create stacking classifier with tuned base models and meta model
stack_model = StackingClassifier(
    estimators=[
        ("rf", rf.best_estimator_),
        ("xgb", xgb.best_estimator_),
        ("svm", svm.best_estimator_)
    ],
    final_estimator=meta_model,
    cv=TimeSeriesSplit(n_splits=5)
)

print("Training stacking model with tuned parameters...")
stack_model.fit(
    X_train_scaled,
    y_train
)
print("Stacking model training complete!")

stack_pred = stack_model.predict(X_test_scaled)
stack_acc = accuracy_score(y_test, stack_pred)
print(f"Stacking Holdout Accuracy: {stack_acc:.4f}")


# ============================================================
# CELL 15 — Final Summary Report
# ============================================================
from sklearn.metrics import classification_report, confusion_matrix

print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print("\n1. RANDOM FOREST")
print(f"   Holdout Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"   TSCV Accuracy:    {rf_tscv['accuracy_mean']:.4f} ± {rf_tscv['accuracy_std']:.4f}")
print(f"   TSCV F1:          {rf_tscv['f1_mean']:.4f} ± {rf_tscv['f1_std']:.4f}")
for param, value in rf.best_params_.items():
    print(f"   {param}: {value}")

print("\n2. XGBOOST")
print(f"   Holdout Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"   TSCV Accuracy:    {xgb_tscv['accuracy_mean']:.4f} ± {xgb_tscv['accuracy_std']:.4f}")
print(f"   TSCV F1:          {xgb_tscv['f1_mean']:.4f} ± {xgb_tscv['f1_std']:.4f}")
for param, value in xgb.best_params_.items():
    print(f"   {param}: {value}")

print("\n3. SVM")
print(f"   Holdout Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
print(f"   TSCV Accuracy:    {svm_tscv['accuracy_mean']:.4f} ± {svm_tscv['accuracy_std']:.4f}")
print(f"   TSCV F1:          {svm_tscv['f1_mean']:.4f} ± {svm_tscv['f1_std']:.4f}")
for param, value in svm.best_params_.items():
    print(f"   {param}: {value}")

print("\n4. META MODEL (Logistic Regression)")
for param, value in meta_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"   Best CV Score: {meta_search.best_score_:.4f}")

print("\n5. STACKING ENSEMBLE")
print(f"   Holdout Accuracy: {stack_acc:.4f}")

print("\n" + "-" * 60)
print("STACKING CLASSIFICATION REPORT:")
print(classification_report(y_test, stack_pred))

print("STACKING CONFUSION MATRIX:")
print(confusion_matrix(y_test, stack_pred))

print("\n" + "=" * 60)
print("PIPELINE CONFIGURATION:")
print(f"  Target horizon: 5-day forward")
print(f"  Train-test split: 80/20 chronological per company")
print(f"  Sentiment: shifted by 1 day (no future leakage)")
print(f"  Cross-validation: TimeSeriesSplit(n_splits=5) per company")
print(f"  Features: {features}")
print("=" * 60)
