# src/prophet_model.py
"""
Trains one Facebook Prophet model per city.
Forecasts AQI 48 hours ahead.
Per-city config controls: changepoint scale, seasonality mode, regressor usage.

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

REGRESSORS = ["temperature_c", "wind_speed_ms", "humidity_pct"]

# ── Per-city config ────────────────────────────────────────────────────────
# use_regressors: False for cities where ffill hurts more than it helps
# changepoint_prior_scale: higher = more flexible trend
# seasonality_mode: multiplicative for high-variance/erratic cities
CITY_CONFIG = {
    "Delhi": {
        "changepoint_prior_scale": 0.1,
        "seasonality_mode":        "multiplicative",
        "use_regressors":          False   # regressors hurt Delhi, remove them
    },
    "Kolkata": {
        "changepoint_prior_scale": 0.1,
        "seasonality_mode":        "multiplicative",
        "use_regressors":          False
    },
    "Patna": {
        "changepoint_prior_scale": 0.2,
        "seasonality_mode":        "multiplicative",
        "use_regressors":          True
    },
    "Guwahati": {
        "changepoint_prior_scale": 0.5,   # very flexible trend
        "seasonality_mode":        "multiplicative",
        "use_regressors":          True,
        "extra_seasonalities": [
            {"name": "monthly", "period": 30.5, "fourier_order": 5},
        ]
    },
    "Shillong": {
        "changepoint_prior_scale": 0.5,
        "seasonality_mode":        "multiplicative",
        "use_regressors":          True,
        "extra_seasonalities": [
            {"name": "monthly", "period": 30.5, "fourier_order": 5},
        ]
    },
    "Bhubaneswar": {
        "changepoint_prior_scale": 0.15,
        "seasonality_mode":        "additive",
        "use_regressors":          True
    },
}

DEFAULT_CONFIG = {
    "changepoint_prior_scale": 0.05,
    "seasonality_mode":        "additive",
    "use_regressors":          True
}


# ── Load ───────────────────────────────────────────────────────────────────

def load_features():
    conn = sqlite3.connect(CLEAN_DB)
    cols = ["city", "timestamp", "aqi"] + REGRESSORS
    df   = pd.read_sql(
        f"SELECT {', '.join(cols)} FROM features ORDER BY city, timestamp",
        conn, parse_dates=["timestamp"]
    )
    conn.close()
    return df


# ── Metrics ────────────────────────────────────────────────────────────────

def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ── Build model ────────────────────────────────────────────────────────────

def build_model(city, use_regressors):
    config = CITY_CONFIG.get(city, DEFAULT_CONFIG)
    model  = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=config["changepoint_prior_scale"],
        seasonality_mode=config["seasonality_mode"]
    )
    # Extra seasonalities for erratic cities
    for s in config.get("extra_seasonalities", []):
        model.add_seasonality(
            name=s["name"],
            period=s["period"],
            fourier_order=s["fourier_order"]
        )
    if use_regressors:
        for reg in REGRESSORS:
            model.add_regressor(reg)
    return model


# ── Prepare future regressor frame ────────────────────────────────────────

def attach_regressors(future_df, history_df, city_df):
    """
    Attaches regressor values to a Prophet future dataframe.
    Known history rows get exact values.
    Future rows get a seasonal hourly average from the training data.
    """
    # Build hour-of-day seasonal averages from training history
    city_df = city_df.copy()
    city_df["hour"] = city_df["ds"].dt.hour

    hourly_avg = (
        city_df.groupby("hour")[REGRESSORS]
        .mean()
        .reset_index()
    )

    # Merge exact known values first
    future_df = future_df.merge(
        history_df[["ds"] + REGRESSORS],
        on="ds", how="left"
    )

    # For unknown future hours, use seasonal hour average instead of ffill/mean
    future_df["hour"] = future_df["ds"].dt.hour
    future_df = future_df.merge(hourly_avg, on="hour", how="left",
                                suffixes=("", "_avg"))

    for reg in REGRESSORS:
        avg_col = f"{reg}_avg"
        future_df[reg] = future_df[reg].fillna(future_df[avg_col])
        future_df.drop(columns=[avg_col], inplace=True)

    future_df.drop(columns=["hour"], inplace=True)
    return future_df


# ── Train + Forecast ───────────────────────────────────────────────────────

def run_prophet_for_city(city_df, city):
    config         = CITY_CONFIG.get(city, DEFAULT_CONFIG)
    use_regressors = config.get("use_regressors", True)

    prophet_df = city_df.rename(columns={"timestamp": "ds", "aqi": "y"})

    for reg in REGRESSORS:
        prophet_df[reg] = prophet_df[reg].fillna(prophet_df[reg].mean())

    split_idx = len(prophet_df) - 48
    train_df  = prophet_df.iloc[:split_idx].copy()
    test_df   = prophet_df.iloc[split_idx:].copy()

    if len(train_df) < 500:
        print(f"  [SKIP] Not enough data for {city}.")
        return None, None

    # ── Evaluation pass ────────────────────────────────────────────────
    model_eval = build_model(city, use_regressors)
    model_eval.fit(train_df)

    future_eval = model_eval.make_future_dataframe(periods=48, freq="h")

    if use_regressors:
        future_eval = attach_regressors(future_eval, prophet_df, train_df)

    forecast_eval = model_eval.predict(future_eval)
    predicted     = forecast_eval.tail(48)["yhat"].clip(1, 5).reset_index(drop=True)
    actual        = test_df["y"].reset_index(drop=True)

    metrics = {
        "city":            city,
        "mae":             round(mae(actual, predicted), 4),
        "rmse":            round(rmse(actual, predicted), 4),
        "train_rows":      len(train_df),
        "test_rows":       len(test_df),
        "use_regressors":  use_regressors,
        "changepoint":     config["changepoint_prior_scale"],
        "seasonality":     config["seasonality_mode"]
    }

    # ── Full forecast pass — retrain on ALL data ───────────────────────
    model_full = build_model(city, use_regressors)
    model_full.fit(prophet_df)

    future_48 = model_full.make_future_dataframe(periods=48, freq="h")

    if use_regressors:
        future_48 = attach_regressors(future_48, prophet_df, prophet_df)

    forecast_48  = model_full.predict(future_48)
    forecast_out = forecast_48.tail(48)[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()

    forecast_out["city"]       = city
    forecast_out["yhat"]       = forecast_out["yhat"].clip(1, 5).round(2)
    forecast_out["yhat_lower"] = forecast_out["yhat_lower"].clip(1, 5).round(2)
    forecast_out["yhat_upper"] = forecast_out["yhat_upper"].clip(1, 5).round(2)
    forecast_out               = forecast_out.rename(columns={"ds": "timestamp"})

    return forecast_out, metrics


# ── Save ───────────────────────────────────────────────────────────────────

def save_results(all_forecasts, all_metrics):
    conn         = sqlite3.connect(CLEAN_DB)
    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    forecasts_df.to_sql("prophet_forecasts", conn, if_exists="replace", index=False)
    print(f"\n  Saved {len(forecasts_df)} forecast rows → prophet_forecasts")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_sql("prophet_metrics", conn, if_exists="replace", index=False)
    print(f"  Saved {len(metrics_df)} metric rows   → prophet_metrics")

    conn.close()
    return metrics_df


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_DB):
        print("ERROR: data/vayu_clean.db not found.")
        return

    print("=" * 55)
    print("VAYU — Prophet Forecasting (48-hour AQI)")
    print("  Per-city config + smart regressor handling")
    print("=" * 55)

    df     = load_features()
    cities = sorted(df["city"].unique())
    print(f"Cities: {len(cities)}")

    all_forecasts = []
    all_metrics   = []

    for city in cities:
        print(f"\n[Prophet] {city}")
        city_df = (
            df[df["city"] == city]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        forecast, metrics = run_prophet_for_city(city_df, city)

        if forecast is not None:
            all_forecasts.append(forecast)
            all_metrics.append(metrics)
            reg_note = "regressors=ON" if metrics["use_regressors"] else "regressors=OFF"
            print(f"  MAE  : {metrics['mae']}  |  RMSE : {metrics['rmse']}")
            print(f"  Config: cps={metrics['changepoint']}  "
                  f"mode={metrics['seasonality']}  {reg_note}")

    if not all_forecasts:
        print("\nNo forecasts generated.")
        return

    print("\n" + "=" * 55)
    metrics_df = save_results(all_forecasts, all_metrics)

    print("\n── Summary ───────────────────────────────────────")
    print(f"\n  {'City':<14} {'MAE':>8}  {'RMSE':>8}  {'Note'}")
    print(f"  {'-'*50}")
    for _, row in metrics_df.sort_values("mae").iterrows():
        tuned   = " ← tuned"   if row["city"] in CITY_CONFIG else ""
        no_regs = " (no regs)" if not row["use_regressors"] else ""
        print(f"  {row['city']:<14} {row['mae']:>8.4f}  "
              f"{row['rmse']:>8.4f}  {tuned}{no_regs}")
    print(f"  {'-'*50}")
    print(f"  {'Average':<14} {metrics_df['mae'].mean():>8.4f}  "
          f"{metrics_df['rmse'].mean():>8.4f}")

    print("\n" + "=" * 55)
    print("Done.")
    print("=" * 55)


if __name__ == "__main__":
    main()