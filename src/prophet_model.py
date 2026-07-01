# src/prophet_model.py
"""
Trains one Facebook Prophet model per city.
Forecasts AQI 48 hours ahead.
Includes weather regressors + per-city tuning for problem cities.
Saves forecasts + evaluation metrics to vayu_clean.db.

Run: python src/prophet_model.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DB = os.path.join(ROOT, "data", "vayu_clean.db")

# ── Per-city Prophet config ────────────────────────────────────────────────
# Higher changepoint_prior_scale = more flexible trend (better for erratic cities)
# multiplicative seasonality works better when variance scales with the level
CITY_CONFIG = {
    "Guwahati":   {"changepoint_prior_scale": 0.3, "seasonality_mode": "multiplicative"},
    "Shillong":   {"changepoint_prior_scale": 0.3, "seasonality_mode": "multiplicative"},
    "Patna":      {"changepoint_prior_scale": 0.15, "seasonality_mode": "multiplicative"},
    "Bhubaneswar":{"changepoint_prior_scale": 0.15, "seasonality_mode": "additive"},
}

DEFAULT_CONFIG = {
    "changepoint_prior_scale": 0.05,
    "seasonality_mode": "additive"
}

# Weather regressors to add as additional Prophet inputs
REGRESSORS = ["temperature_c", "wind_speed_ms", "humidity_pct"]


# ── Load ───────────────────────────────────────────────────────────────────

def load_features():
    conn = sqlite3.connect(CLEAN_DB)
    cols = ["city", "timestamp", "aqi"] + REGRESSORS
    df   = pd.read_sql(
        f"SELECT {', '.join(cols)} FROM features ORDER BY city, timestamp",
        conn,
        parse_dates=["timestamp"]
    )
    conn.close()
    return df


# ── Metrics ────────────────────────────────────────────────────────────────

def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ── Build model ────────────────────────────────────────────────────────────

def build_model(city):
    config = CITY_CONFIG.get(city, DEFAULT_CONFIG)
    model  = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=config["changepoint_prior_scale"],
        seasonality_mode=config["seasonality_mode"]
    )
    for reg in REGRESSORS:
        model.add_regressor(reg)
    return model


# ── Train + Forecast ───────────────────────────────────────────────────────

def run_prophet_for_city(city_df, city):
    # Prophet requires 'ds' and 'y' column names
    prophet_df = city_df.rename(columns={"timestamp": "ds", "aqi": "y"})

    # Fill any regressor nulls with column mean (rare edge case)
    for reg in REGRESSORS:
        prophet_df[reg] = prophet_df[reg].fillna(prophet_df[reg].mean())

    # ── Train/test split — hold out last 48 hours ──────────────────────
    split_idx = len(prophet_df) - 48
    train_df  = prophet_df.iloc[:split_idx]
    test_df   = prophet_df.iloc[split_idx:]

    if len(train_df) < 500:
        print(f"  [SKIP] Not enough data for {city} ({len(train_df)} rows).")
        return None, None

    # ── Evaluation pass ────────────────────────────────────────────────
    model_eval = build_model(city)
    model_eval.fit(train_df)

    future_eval   = model_eval.make_future_dataframe(periods=48, freq="h")

    # Add regressor values to the future evaluation frame
    # For the 48 held-out hours, use actual values from test_df
    regressor_history = prophet_df[["ds"] + REGRESSORS].copy()
    future_eval = future_eval.merge(regressor_history, on="ds", how="left")

    # For any still-missing regressor values, forward fill then use mean
    for reg in REGRESSORS:
        future_eval[reg] = (
            future_eval[reg]
            .ffill()
            .fillna(prophet_df[reg].mean())
        )

    forecast_eval = model_eval.predict(future_eval)
    predicted     = forecast_eval.tail(48)["yhat"].clip(1, 5).reset_index(drop=True)
    actual        = test_df["y"].reset_index(drop=True)

    metrics = {
        "city":       city,
        "mae":        round(mae(actual, predicted), 4),
        "rmse":       round(rmse(actual, predicted), 4),
        "train_rows": len(train_df),
        "test_rows":  len(test_df),
        "config":     str(CITY_CONFIG.get(city, DEFAULT_CONFIG))
    }

    # ── Full forecast pass — retrain on ALL data ───────────────────────
    model_full = build_model(city)
    model_full.fit(prophet_df)

    future_48 = model_full.make_future_dataframe(periods=48, freq="h")
    future_48 = future_48.merge(regressor_history, on="ds", how="left")

    for reg in REGRESSORS:
        future_48[reg] = (
            future_48[reg]
            .ffill()
            .fillna(prophet_df[reg].mean())
        )

    forecast_48 = model_full.predict(future_48)

    forecast_out = forecast_48.tail(48)[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    forecast_out["city"]       = city
    forecast_out["yhat"]       = forecast_out["yhat"].clip(1, 5).round(2)
    forecast_out["yhat_lower"] = forecast_out["yhat_lower"].clip(1, 5).round(2)
    forecast_out["yhat_upper"] = forecast_out["yhat_upper"].clip(1, 5).round(2)
    forecast_out = forecast_out.rename(columns={"ds": "timestamp"})

    return forecast_out, metrics


# ── Save ───────────────────────────────────────────────────────────────────

def save_results(all_forecasts, all_metrics):
    conn = sqlite3.connect(CLEAN_DB)

    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    forecasts_df.to_sql("prophet_forecasts", conn, if_exists="replace", index=False)
    print(f"\n  Saved {len(forecasts_df)} forecast rows → table: prophet_forecasts")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_sql("prophet_metrics", conn, if_exists="replace", index=False)
    print(f"  Saved {len(metrics_df)} metric rows   → table: prophet_metrics")

    conn.close()
    return metrics_df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_DB):
        print("ERROR: data/vayu_clean.db not found.")
        return

    print("=" * 55)
    print("VAYU — Prophet Forecasting (48-hour AQI)")
    print("  With weather regressors + per-city tuning")
    print("=" * 55)

    df     = load_features()
    cities = sorted(df["city"].unique())
    print(f"Cities: {len(cities)}")

    all_forecasts = []
    all_metrics   = []

    for city in cities:
        print(f"\n[Prophet] {city}")
        city_df = df[df["city"] == city].sort_values("timestamp").reset_index(drop=True)

        forecast, metrics = run_prophet_for_city(city_df, city)

        if forecast is not None:
            all_forecasts.append(forecast)
            all_metrics.append(metrics)
            print(f"  MAE  : {metrics['mae']}")
            print(f"  RMSE : {metrics['rmse']}")

    if not all_forecasts:
        print("\nNo forecasts generated.")
        return

    print("\n" + "=" * 55)
    metrics_df = save_results(all_forecasts, all_metrics)

    print("\n── Summary ───────────────────────────────────────")
    print(f"\n  {'City':<14} {'MAE':>8}  {'RMSE':>8}")
    print(f"  {'-'*35}")
    for _, row in metrics_df.sort_values("mae").iterrows():
        flag = " ← tuned" if row["city"] in CITY_CONFIG else ""
        print(f"  {row['city']:<14} {row['mae']:>8.4f}  {row['rmse']:>8.4f}{flag}")
    print(f"  {'-'*35}")
    print(f"  {'Average':<14} {metrics_df['mae'].mean():>8.4f}  "
          f"{metrics_df['rmse'].mean():>8.4f}")

    print("\n" + "=" * 55)
    print("Done.")
    print("=" * 55)


if __name__ == "__main__":
    main()