# src/xgboost_classifier.py
"""
XGBoost Classifier — predicts next-day AQI category per city.
    Categories: Good / Fair / Moderate / Poor / Hazardous

SHAP explainability runs after training.
Saves to vayu_clean.db:
    xgb_classifier_metrics   — per-city accuracy + F1
    xgb_classifier_shap      — mean |SHAP| per feature (for dashboard chart)
    xgb_classifier_report    — per-class precision/recall/F1 per city

Run: python src/xgboost_classifier.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
import warnings

from xgboost.interpret import shap_values
warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DB = os.path.join(ROOT, "data", "vayu_clean.db")

FEATURE_COLS = [
    "aqi", "pm25", "pm10", "no2", "o3",
    "temperature_c", "humidity_pct", "wind_speed_ms", "wind_deg",
    "rainfall_1h_mm", "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "aqi_roll_6h", "aqi_roll_24h"
]

TARGET_COL = "next_aqi_category"

# Fixed label order — used consistently everywhere
CATEGORIES = ["Good", "Fair", "Moderate", "Poor", "Hazardous"]


# ── Load ───────────────────────────────────────────────────────────────────

def load_features():
    conn = sqlite3.connect(CLEAN_DB)
    df   = pd.read_sql(
        "SELECT * FROM features ORDER BY city, timestamp",
        conn, parse_dates=["timestamp"]
    )
    conn.close()
    return df


# ── Label encoding ─────────────────────────────────────────────────────────

def encode_labels(y_series):
    """Encodes category strings to integers 0-4 in fixed CATEGORIES order."""
    le = LabelEncoder()
    le.classes_ = np.array(CATEGORIES)
    return le.transform(y_series), le


# ── Train + Evaluate (one city) ───────────────────────────────────────────

def run_classifier_for_city(city_df, city):
    df = city_df.copy().sort_values("timestamp").reset_index(drop=True)

    # Drop rows with missing target or features
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).reset_index(drop=True)

    # Keep only known categories
    df = df[df[TARGET_COL].isin(CATEGORIES)].reset_index(drop=True)

    if len(df) < 300:
        print(f"  [SKIP] Not enough data for {city} ({len(df)} rows).")
        return None, None, None

    X = df[FEATURE_COLS].values
    y_raw = df[TARGET_COL].values
    print("\n" + "="*50)
    print(city)
    print(pd.Series(y_raw[-48:]).value_counts())
    y, le = encode_labels(y_raw)

    # ── Train / test split — hold out last 48 hours ────────────────────
    split_idx = len(df) - 48
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Class weights — fixes the Good/Moderate imbalance ─────────────
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # ── Model ──────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(CATEGORIES),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # Internal validation split for early stopping
    val_size = int(len(X_train) * 0.1)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
    sw_tr       = sample_weights[:-val_size]

    model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # ── Evaluate ────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    accuracy  = round(float(accuracy_score(y_test, y_pred)), 4)
    f1_macro  = round(float(f1_score(y_test, y_pred, average="macro",
                                     zero_division=0)), 4)
    f1_weighted = round(float(f1_score(y_test, y_pred, average="weighted",
                                       zero_division=0)), 4)

    # Per-class report
    report = classification_report(
        y_test, y_pred,
        labels=list(range(len(CATEGORIES))),
        target_names=CATEGORIES,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "city":         city,
        "accuracy":     accuracy,
        "f1_macro":     f1_macro,
        "f1_weighted":  f1_weighted,
        "train_rows":   len(X_train),
        "test_rows":    len(X_test)
    }

    # ── Per-class report rows ───────────────────────────────────────────
    report_rows = []
    for cat in CATEGORIES:
        if cat in report:
            report_rows.append({
                "city":      city,
                "category":  cat,
                "precision": round(report[cat]["precision"], 4),
                "recall":    round(report[cat]["recall"], 4),
                "f1":        round(report[cat]["f1-score"], 4),
                "support":   int(report[cat]["support"])
            })

    # ── SHAP ────────────────────────────────────────────────────────────
    # Use a sample of training data for SHAP (full set is slow)
    shap_sample_size = min(2000, len(X_train))
    rng              = np.random.default_rng(42)
    sample_idx       = rng.choice(len(X_train), shap_sample_size, replace=False)
    X_shap           = X_train[sample_idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # shap_values shape: (n_classes, n_samples, n_features)
    # Mean absolute SHAP across all classes and all samples → one value per feature
    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[0] == len(CATEGORIES):
            # Old format: (classes, samples, features)
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))

        elif shap_values.shape[2] == len(CATEGORIES):
            # New format: (samples, features, classes)
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))

        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    else:
        # Binary classification
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_rows = [
        {
            "city":       city,
            "feature":    FEATURE_COLS[i],
            "mean_abs_shap": round(float(mean_abs_shap[i]), 6)
        }
        for i in range(len(FEATURE_COLS))
    ]
    shap_df = pd.DataFrame(shap_rows).sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    return metrics, report_rows, shap_df


# ── Save ───────────────────────────────────────────────────────────────────

def save_results(all_metrics, all_report_rows, all_shap_dfs):
    conn = sqlite3.connect(CLEAN_DB)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_sql("xgb_classifier_metrics", conn,
                      if_exists="replace", index=False)
    print(f"\n  Saved {len(metrics_df)} rows   → xgb_classifier_metrics")

    report_df = pd.DataFrame(all_report_rows)
    report_df.to_sql("xgb_classifier_report", conn,
                     if_exists="replace", index=False)
    print(f"  Saved {len(report_df)} rows   → xgb_classifier_report")

    shap_df = pd.concat(all_shap_dfs, ignore_index=True)
    shap_df.to_sql("xgb_classifier_shap", conn,
                   if_exists="replace", index=False)
    print(f"  Saved {len(shap_df)} rows   → xgb_classifier_shap")

    conn.close()
    return metrics_df, shap_df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_DB):
        print("ERROR: data/vayu_clean.db not found.")
        return

    print("=" * 60)
    print("VAYU — XGBoost Classifier + SHAP")
    print("  Predicts: Good / Fair / Moderate / Poor / Hazardous")
    print("=" * 60)

    df      = load_features()
    cities  = sorted(df["city"].unique())
    print(f"Cities    : {len(cities)}")
    print(f"Total rows: {len(df):,}")

    # Global class distribution
    dist = df[TARGET_COL].value_counts()
    print(f"\nGlobal label distribution:")
    for cat in CATEGORIES:
        count = dist.get(cat, 0)
        pct   = count / len(df) * 100
        print(f"  {cat:<12} {count:>8,}  ({pct:.1f}%)")

    all_metrics      = []
    all_report_rows  = []
    all_shap_dfs     = []

    for city in cities:
        print(f"\n[XGB Classifier] {city}")
        city_df = df[df["city"] == city].copy()

        result = run_classifier_for_city(city_df, city)
        if result[0] is None:
            continue

        metrics, report_rows, shap_df = result
        all_metrics.append(metrics)
        all_report_rows.extend(report_rows)
        all_shap_dfs.append(shap_df)

        print(f"  Accuracy   : {metrics['accuracy']}")
        print(f"  F1 macro   : {metrics['f1_macro']}")
        print(f"  F1 weighted: {metrics['f1_weighted']}")

        # Show top 3 SHAP features inline
        top3 = shap_df.head(3)["feature"].tolist()
        print(f"  Top SHAP   : {', '.join(top3)}")

    if not all_metrics:
        print("\nNo results generated.")
        return

    print("\n" + "=" * 60)
    metrics_df, shap_df = save_results(all_metrics, all_report_rows, all_shap_dfs)

    # ── Summary table ─────────────────────────────────────────────────
    print("\n── Per-City Classification Summary ──────────────────────")
    print(f"\n  {'City':<14} {'Accuracy':>10}  {'F1 Macro':>10}  {'F1 Weighted':>12}")
    print(f"  {'-'*52}")
    for _, row in metrics_df.sort_values("f1_macro", ascending=False).iterrows():
        print(f"  {row['city']:<14} {row['accuracy']:>10.4f}  "
              f"{row['f1_macro']:>10.4f}  {row['f1_weighted']:>12.4f}")
    print(f"  {'-'*52}")
    print(f"  {'Average':<14} {metrics_df['accuracy'].mean():>10.4f}  "
          f"{metrics_df['f1_macro'].mean():>10.4f}  "
          f"{metrics_df['f1_weighted'].mean():>12.4f}")

    # ── Global SHAP summary ───────────────────────────────────────────
    print("\n── Global SHAP Feature Importance (avg across all cities) ─")
    global_shap = (
        shap_df.groupby("feature")["mean_abs_shap"]
               .mean()
               .sort_values(ascending=False)
               .reset_index()
    )
    print(f"\n  {'Feature':<22} {'Mean |SHAP|':>14}")
    print(f"  {'-'*38}")
    print("\n── Global SHAP Feature Importance (avg across all cities) ─")
    print(f"\n  {'Feature':<22} {'Mean |SHAP|':>12}")
    print(f"  {'-'*38}")

    for _, row in global_shap.iterrows():
        print(f"  {row['feature']:<22} {row['mean_abs_shap']:>12.4f}")

    # Save global SHAP summary too — dashboard uses this for the overview chart
    global_shap.to_sql(
        "xgb_global_shap", sqlite3.connect(CLEAN_DB),
        if_exists="replace", index=False
    )
    print(f"\n  Saved global SHAP → xgb_global_shap")

    print("\n" + "=" * 60)
    print("Done. XGBoost Classifier + SHAP complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()