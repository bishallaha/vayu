import os
import sys
import sqlite3
import pandas as pd

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DB   = os.path.join(ROOT, "data", "vayu.db")
CLEAN_DB = os.path.join(ROOT, "data", "vayu_clean.db")


#Helpers

def load_raw(table):
    conn = sqlite3.connect(RAW_DB)
    df   = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def write_clean_db(aqi_df, weather_df):
    """Overwrite vayu_clean.db with cleaned data. Recreates tables fresh each run."""
    conn = sqlite3.connect(CLEAN_DB)
    c    = conn.cursor()

    #AQI Table
    c.execute("DROP TABLE IF EXISTS aqi_readings")
    c.execute("""
        CREATE TABLE aqi_readings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            city       TEXT    NOT NULL,
            latitude   REAL,
            longitude  REAL,
            aqi        INTEGER,
            pm25       REAL,
            pm10       REAL,
            no2        REAL,
            o3         REAL,
            timestamp  TEXT    NOT NULL,
            fetched_at TEXT    NOT NULL,
            UNIQUE(city, timestamp)
        )
    """)

    #Weather table
    c.execute("DROP TABLE IF EXISTS weather_readings")
    c.execute("""
        CREATE TABLE weather_readings (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            city           TEXT NOT NULL,
            temperature_c  REAL,
            humidity_pct   REAL,
            wind_speed_ms  REAL,
            wind_deg       REAL,
            rainfall_1h_mm REAL,
            condition      TEXT,
            timestamp      TEXT NOT NULL,
            fetched_at     TEXT NOT NULL,
            UNIQUE(city, timestamp)
        )
    """)

    conn.commit()

    aqi_df.drop(columns=["id"], errors="ignore").to_sql(
        "aqi_readings", conn, if_exists="append", index=False
    )
    weather_df.drop(columns=["id"], errors="ignore").to_sql(
        "weather_readings", conn, if_exists="append", index=False
    )

    conn.close()


#AQI Cleaning
def clean_aqi(df):
    print("\n── Cleaning: aqi_readings " + "─" * 30)
    print(f"  Starting rows            : {len(df):,}")

    original = len(df)
    pollutants = ["pm25", "pm10", "no2", "o3"]

    # Step 1: Sort by city + timestamp
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    # Step 2: Remove exact duplicates (same city + same timestamp)
    before = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="first")
    print(f"  Exact duplicates removed : {before - len(df):,}")

    # Step 3: Remove rows where ALL four pollutants are null
    before = len(df)
    all_null = df[pollutants].isnull().all(axis=1)
    df = df[~all_null].reset_index(drop=True)
    print(f"  All pollutants null (dropped): {before - len(df):,}")

    # Step 4: Remove sensor error outliers
    before = len(df)
    bad = (
        (df["pm25"].notna() & ((df["pm25"] < 0) | (df["pm25"] > 999))) |
        (df["pm10"].notna() & ((df["pm10"] < 0) | (df["pm10"] > 999))) |
        (df["no2"].notna()  & ((df["no2"]  < 0) | (df["no2"]  > 500))) |
        (df["o3"].notna()   & ((df["o3"]   < 0) | (df["o3"]   > 500))) |
        (df["aqi"].notna()  & ((df["aqi"]  < 1) | (df["aqi"]  > 5)))
    )
    df = df[~bad].reset_index(drop=True)
    print(f"  Outliers removed         : {before - len(df):,}")

    # Step 5: Forward fill isolated gaps within each city (max 3 consecutive hours)
    null_before = int(df[pollutants].isnull().sum().sum())
    df[pollutants] = (
        df.groupby("city")[pollutants]
          .transform(lambda x: x.ffill(limit=3).bfill(limit=1))
    )
    null_after = int(df[pollutants].isnull().sum().sum())
    print(f"  Null cells filled (ffill): {null_before - null_after:,}")

    # Step 6: Drop any rows still missing ALL pollutants after fill
    before = len(df)
    df = df.dropna(subset=pollutants, how="all").reset_index(drop=True)
    print(f"  Still all-null (dropped) : {before - len(df):,}")

    print(f"  {'─'*40}")
    print(f"  Final rows               : {len(df):,}")
    print(f"  Total rows removed       : {original - len(df):,}")

    return df


#Weather Cleaning

def clean_weather(df):
    print("\n── Cleaning: weather_readings " + "─" * 25)
    print(f"  Starting rows            : {len(df):,}")

    # Step 1: Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["city", "timestamp"], keep="first")
    print(f"  Exact duplicates removed : {before - len(df):,}")

    # Step 2: Fill null rainfall with 0 (no report = no rain, valid)
    rain_nulls = int(df["rainfall_1h_mm"].isnull().sum())
    df["rainfall_1h_mm"] = df["rainfall_1h_mm"].fillna(0.0)
    print(f"  Rainfall null → 0        : {rain_nulls:,}")

    # Step 3: Remove rows with no temperature reading
    before = len(df)
    df = df.dropna(subset=["temperature_c"])
    print(f"  Missing temperature (dropped): {before - len(df):,}")

    # Step 4: Temperature range check for India (-5°C to 55°C)
    before = len(df)
    df = df[(df["temperature_c"] >= -5) & (df["temperature_c"] <= 55)]
    print(f"  Temp outliers removed    : {before - len(df):,}")

    print(f"  {'─'*40}")
    print(f"  Final rows               : {len(df):,}")

    return df.reset_index(drop=True)


#Main Function

def main():
    if not os.path.exists(RAW_DB):
        print("ERROR: data/vayu.db not found. Run collect.py first.")
        return

    print("=" * 55)
    print("VAYU — Data Cleaning Pipeline")
    print(f"  Source : data/vayu.db")
    print(f"  Output : data/vayu_clean.db")
    print("=" * 55)

    aqi_raw     = load_raw("aqi_readings")
    weather_raw = load_raw("weather_readings")

    if aqi_raw.empty:
        print("ERROR: aqi_readings table is empty. Run collect.py first.")
        return

    aqi_clean     = clean_aqi(aqi_raw)
    weather_clean = clean_weather(weather_raw)

    write_clean_db(aqi_clean, weather_clean)

    print(f"\n{'='*55}")
    print("Clean database saved to: data/vayu_clean.db")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()