#SQLite Database Setup

import sqlite3
import os

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "vayu.db"
)


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables if they don't already exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS aqi_readings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            city       TEXT NOT NULL,
            latitude   REAL,
            longitude  REAL,
            aqi        INTEGER,
            pm25       REAL,
            pm10       REAL,
            no2        REAL,
            o3         REAL,
            timestamp  TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            UNIQUE(city, timestamp)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS weather_readings (
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
    conn.close()
    print("Database ready: data/vayu.db")


def get_history_days():
    """
    First run (empty DB) → pull 180 days of history.
    All later runs      → pull last 7 days to fill any gaps.
    """
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM aqi_readings")
        count = c.fetchone()[0]
        conn.close()
        return 365 if count == 0 else 7
    except Exception:
        return 365  # table doesn't exist yet = first run


def insert_aqi_rows(rows):
    """Batch insert AQI rows. Duplicate city+timestamp is silently ignored."""
    if not rows:
        return 0
    conn = get_connection()
    c    = conn.cursor()
    inserted = 0
    for row in rows:
        try:
            c.execute("""
                INSERT OR IGNORE INTO aqi_readings
                    (city, latitude, longitude, aqi, pm25, pm10, no2, o3,
                     timestamp, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["city"], row["latitude"], row["longitude"],
                row["aqi"],  row["pm25"],    row["pm10"],
                row["no2"],  row["o3"],
                row["timestamp"], row["fetched_at"]
            ))
            if c.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"  [DB ERROR] {row.get('city', '?')}: {e}")
    conn.commit()
    conn.close()
    return inserted


def insert_weather_rows(rows):
    """Batch insert weather rows. Duplicate city+timestamp is silently ignored."""
    if not rows:
        return 0
    conn = get_connection()
    c    = conn.cursor()
    inserted = 0
    for row in rows:
        try:
            c.execute("""
                INSERT OR IGNORE INTO weather_readings
                    (city, temperature_c, humidity_pct, wind_speed_ms, wind_deg,
                     rainfall_1h_mm, condition, timestamp, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["city"],           row["temperature_c"],  row["humidity_pct"],
                row["wind_speed_ms"],  row["wind_deg"],        row["rainfall_1h_mm"],
                row["condition"],      row["timestamp"],       row["fetched_at"]
            ))
            if c.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"  [DB ERROR] {row.get('city', '?')}: {e}")
    conn.commit()
    conn.close()
    return inserted


def print_summary():
    """Print how many records each city has in both tables."""
    conn = get_connection()
    c    = conn.cursor()

    print("\n  AQI records per city:")
    c.execute("SELECT city, COUNT(*) FROM aqi_readings GROUP BY city ORDER BY city")
    for city, count in c.fetchall():
        print(f"    {city:<12} {count:>6} records")

    print("\n  Weather records per city:")
    c.execute("SELECT city, COUNT(*) FROM weather_readings GROUP BY city ORDER BY city")
    for city, count in c.fetchall():
        print(f"    {city:<12} {count:>6} records")

    conn.close()