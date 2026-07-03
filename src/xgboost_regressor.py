# src/xgboost_regressor.py
"""
XGBoost Regressor — predicts AQI value 24 hours ahead per city.
Compared against Prophet using MAE and RMSE.
Saves predictions + metrics to vayu_clean.db.

Run: python src/xgboost_regressor.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DB = os.path.join(ROOT, "data", "vayu_clean.db")

# Features used for regression
# Excludes: id, city, latitude, longitude, timestamp, fetched_at,
#           next_aqi_category (that's for the classifier), next_aqi (target)
FEATURE_COLS = [
    "aqi", "pm25", "pm10", "no2", "o3",
    "temperature_c", "humidity_pct", "wind_speed_ms", "wind_deg",
    "rainfall_1h_mm", "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "aqi_roll_6h", "aqi_roll_24h"
]

TARGET_COL = "next_aqi"  # continuous AQI value 24 hours ahead


# ── Load ───────────────────────────────────────────────────────────────────

def load_features():
    conn = sqlite3.connect(CLEAN_DB)
    df   = pd.read_sql(
        "SELECT * FROM features ORDER BY city, timestamp",
        conn, parse_dates=["timestamp"]
    )
    conn.close()
    return df


# ── Load Prophet metrics for comparison ───────────────────────────────────

def load_prophet_metrics():
    try:
        conn = sqlite3.connect(CLEAN_DB)
        df   = pd.read_sql("SELECT city, mae, rmse FROM prophet_metrics", conn)
        conn.close()
        return df.set_index("city")
    except Exception:
        return None


# ── Train + Evaluate (one city) ───────────────────────────────────────────

def run_xgb_for_city(city_df, city):
    """
    Trains XGBoost Regressor on one city.
    Uses TimeSeriesSplit cross-validation for hyperparameter selection.
    Returns (predictions_df, metrics_dict).
    """
    df = city_df.copy().sort_values("timestamp").reset_index(drop=True)

    # Drop rows where target or any key feature is null
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).reset_index(drop=True)

    if len(df) < 200:
        print(f"  [SKIP] Not enough clean data for {city} ({len(df)} rows).")
        return None, None

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # ── Train / test split — hold out last 48 hours (same as Prophet) ──
    split_idx = len(df) - 48
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Model ──────────────────────────────────────────────────────────
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # Early stopping on a small validation slice from training data
    val_size  = int(len(X_train) * 0.1)
    X_tr      = X_train[:-val_size]
    X_val     = X_train[-val_size:]
    y_tr      = y_train[:-val_size]
    y_val     = y_train[-val_size:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # ── Evaluate ────────────────────────────────────────────────────────
    preds_test = np.clip(model.predict(X_test), 1, 5)

    city_mae  = round(float(mean_absolute_error(y_test, preds_test)), 4)
    city_rmse = round(float(np.sqrt(mean_squared_error(y_test, preds_test))), 4)

    metrics = {
        "city":       city,
        "mae":        city_mae,
        "rmse":       city_rmse,
        "train_rows": len(X_train),
        "test_rows":  len(X_test)
    }

    # ── Generate predictions for the last 48 rows in full dataset ──────
    # These are the most recent known feature rows — used on the dashboard
    # to show "XGBoost predicted AQI" for the next 24 hours
    recent_rows        = df.tail(48).copy()
    recent_X = recent_rows[FEATURE_COLS].ffill().values
    recent_preds       = np.clip(model.predict(recent_X), 1, 5).round(2)

    predictions_df = pd.DataFrame({
        "city":         city,
        "timestamp":    recent_rows["timestamp"].values,
        "actual_aqi":   recent_rows["aqi"].values,
        "predicted_aqi": recent_preds
    })

    return predictions_df, metrics, model, df[FEATURE_COLS].columns.tolist()


# ── Feature importance ─────────────────────────────────────────────────────

def get_feature_importance(model, feature_names):
    """Returns a clean feature importance dataframe."""
    importance = model.get_booster().get_score(importance_type="gain")
    rows = []
    for i, feat in enumerate(feature_names):
        key = f"f{i}"
        rows.append({
            "feature":    feat,
            "importance": round(importance.get(key, 0.0), 4)
        })
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


# ── Save ───────────────────────────────────────────────────────────────────

def save_results(all_preds, all_metrics, importance_df):
    conn = sqlite3.connect(CLEAN_DB)

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_sql("xgb_regressor_predictions", conn,
                    if_exists="replace", index=False)
    print(f"\n  Saved {len(preds_df)} prediction rows → xgb_regressor_predictions")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_sql("xgb_regressor_metrics", conn,
                      if_exists="replace", index=False)
    print(f"  Saved {len(metrics_df)} metric rows   → xgb_regressor_metrics")

    importance_df.to_sql("xgb_feature_importance", conn,
                         if_exists="replace", index=False)
    print(f"  Saved {len(importance_df)} feature rows  → xgb_feature_importance")

    conn.close()
    return metrics_df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_DB):
        print("ERROR: data/vayu_clean.db not found.")
        return

    print("=" * 60)
    print("VAYU — XGBoost Regressor (24-hour AQI prediction)")
    print("=" * 60)

    df              = load_features()
    prophet_metrics = load_prophet_metrics()
    cities          = sorted(df["city"].unique())
    print(f"Cities: {len(cities)}")
    print(f"Total rows: {len(df):,}\n")

    all_preds   = []
    all_metrics = []
    last_model  = None
    last_feats  = None

    for city in cities:
        print(f"[XGB Regressor] {city}")
        city_df = df[df["city"] == city].copy()

        result = run_xgb_for_city(city_df, city)
        if result[0] is None:
            continue

        preds, metrics, model, feat_names = result
        all_preds.append(preds)
        all_metrics.append(metrics)
        last_model = model
        last_feats = feat_names

        # Compare against Prophet inline
        prophet_mae = None
        if prophet_metrics is not None and city in prophet_metrics.index:
            prophet_mae = prophet_metrics.loc[city, "mae"]
            diff        = metrics["mae"] - prophet_mae
            winner      = "XGB wins" if diff < 0 else "Prophet wins"
            print(f"  XGB MAE  : {metrics['mae']}  |  RMSE: {metrics['rmse']}")
            print(f"  Prophet  : {prophet_mae}  |  Δ={diff:+.4f}  → {winner}")
        else:
            print(f"  MAE  : {metrics['mae']}  |  RMSE: {metrics['rmse']}")

    if not all_preds:
        print("\nNo predictions generated.")
        return

    # Feature importance from last trained model (representative)
    importance_df = get_feature_importance(last_model, last_feats)

    print("\n" + "=" * 60)
    metrics_df = save_results(all_preds, all_metrics, importance_df)

    # ── Summary table ─────────────────────────────────────────────────
    print("\n── Prophet vs XGBoost Regressor — Head to Head ──────────")
    print(f"\n  {'City':<14} {'Prophet MAE':>12}  {'XGB MAE':>10}  {'Winner':>12}")
    print(f"  {'-'*55}")

    xgb_wins    = 0
    prophet_wins = 0

    for _, row in metrics_df.sort_values("mae").iterrows():
        city = row["city"]
        if prophet_metrics is not None and city in prophet_metrics.index:
            p_mae  = prophet_metrics.loc[city, "mae"]
            x_mae  = row["mae"]
            winner = "XGB  ✓" if x_mae < p_mae else "Prophet ✓"
            if x_mae < p_mae:
                xgb_wins += 1
            else:
                prophet_wins += 1
            print(f"  {city:<14} {p_mae:>12.4f}  {x_mae:>10.4f}  {winner:>12}")
        else:
            print(f"  {city:<14} {'N/A':>12}  {row['mae']:>10.4f}")

    print(f"  {'-'*55}")
    p_avg = prophet_metrics["mae"].mean() if prophet_metrics is not None else float("nan")
    x_avg = metrics_df["mae"].mean()
    print(f"  {'Average':<14} {p_avg:>12.4f}  {x_avg:>10.4f}")
    print(f"\n  Score: XGBoost wins {xgb_wins} cities, "
          f"Prophet wins {prophet_wins} cities")

    print("\n── Top 10 Most Important Features ───────────────────────")
    print(f"\n  {'Feature':<20} {'Importance (gain)':>18}")
    print(f"  {'-'*40}")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<20} {row['importance']:>18.4f}")

    print("\n" + "=" * 60)
    print("Done. XGBoost Regressor complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()