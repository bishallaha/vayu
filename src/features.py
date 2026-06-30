# src/features.py
"""
Reads from   : data/vayu_clean.db
Writes to    : data/vayu_clean.db  (new table: features)

Builds the final ML-ready dataset using an EXACT timestamp merge
(both AQI and weather are now genuine hourly series from the same grid).

Run: python src/features.py
"""

import os
import sqlite3
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DB = os.path.join(ROOT, "data", "vayu_clean.db")

# OpenWeather's own 1–5 AQI scale
AQI_CATEGORY_MAP = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Hazardous"
}


# ── Load ───────────────────────────────────────────────────────────────────

def load_clean_tables():
    conn = sqlite3.connect(CLEAN_DB)
    aqi = pd.read_sql("SELECT * FROM aqi_readings", conn, parse_dates=["timestamp"])
    wx  = pd.read_sql("SELECT * FROM weather_readings", conn, parse_dates=["timestamp"])
    conn.close()
    return aqi, wx


# ── Merge (exact timestamp match) ───────────────────────────────────────

def merge_aqi_weather(aqi, wx):
    print("\n── Merging AQI + Weather (exact timestamp match) " + "─" * 5)
    print(f"  AQI rows     : {len(aqi):,}")
    print(f"  Weather rows : {len(wx):,}")

    wx_cols = ["city", "timestamp", "temperature_c", "humidity_pct",
               "wind_speed_ms", "wind_deg", "rainfall_1h_mm"]

    merged = pd.merge(
        aqi, wx[wx_cols],
        on=["city", "timestamp"],
        how="inner"   # only keep hours that exist in BOTH tables — no approximation
    )

    dropped = len(aqi) - len(merged)
    coverage = (len(merged) / len(aqi) * 100) if len(aqi) > 0 else 0

    print(f"  Merged rows  : {len(merged):,}")
    print(f"  AQI rows dropped (no exact weather match) : {dropped:,}")
    print(f"  Match coverage : {coverage:.1f}%")

    return merged


# ── Time-based features ───────────────────────────────────────────────────

def add_time_features(df):
    print("\n── Adding time-based features " + "─" * 25)
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    print("  Added: hour, day_of_week, month, is_weekend")
    return df


# ── Lag features ──────────────────────────────────────────────────────────

def add_lag_features(df):
    print("\n── Adding lag features (per city) " + "─" * 20)
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    for lag_hours in [1, 3, 24]:
        col = f"aqi_lag_{lag_hours}h"
        df[col] = df.groupby("city")["aqi"].shift(lag_hours)

    print("  Added: aqi_lag_1h, aqi_lag_3h, aqi_lag_24h")
    return df


# ── Rolling averages ──────────────────────────────────────────────────────

def add_rolling_features(df):
    print("\n── Adding rolling averages (per city) " + "─" * 17)
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    df["aqi_roll_6h"] = (
        df.groupby("city")["aqi"]
          .transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    )
    df["aqi_roll_24h"] = (
        df.groupby("city")["aqi"]
          .transform(lambda x: x.rolling(window=24, min_periods=1).mean())
    )

    print("  Added: aqi_roll_6h, aqi_roll_24h")
    return df


# ── Target label ───────────────────────────────────────────────────────────

def add_target_label(df):
    print("\n── Creating next-day AQI target label " + "─" * 19)
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    df["next_aqi"] = df.groupby("city")["aqi"].shift(-24)  # next day's AQI (24 hours later)
    df["next_aqi_category"] = df["next_aqi"].map(AQI_CATEGORY_MAP)

    missing_target = df["next_aqi_category"].isnull().sum()
    print(f"  Added: next_aqi, next_aqi_category")
    print(f"  Rows without a target (last reading per city) : {missing_target:,}")

    return df


# ── Save ───────────────────────────────────────────────────────────────────

def save_features(df):
    conn = sqlite3.connect(CLEAN_DB)
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()
    print(f"\n  Saved {len(df):,} rows to table: features (in vayu_clean.db)")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_DB):
        print("ERROR: data/vayu_clean.db not found. Run clean.py first.")
        return

    print("=" * 55)
    print("VAYU — Feature Engineering Pipeline")
    print("=" * 55)

    aqi, wx = load_clean_tables()

    if aqi.empty:
        print("ERROR: aqi_readings table is empty. Run collect.py and clean.py first.")
        return

    df = merge_aqi_weather(aqi, wx)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target_label(df)

    before = len(df)
    df = df.dropna(subset=["next_aqi_category"]).reset_index(drop=True)
    print(f"\n  Dropped {before - len(df):,} rows with no target label")

    print(f"\n  Final feature set shape : {df.shape}")
    print(f"  Cities included : {df['city'].nunique()}")
    print(f"  Columns: {list(df.columns)}")

    save_features(df)

    print("\n" + "=" * 55)
    print("Done. Features table ready for modeling.")
    print("=" * 55)


if __name__ == "__main__":
    main()