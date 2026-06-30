# src/collect.py

from dotenv import load_dotenv
import os
import sys
import requests
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import (
    init_db, get_history_days,
    insert_aqi_rows, insert_weather_rows,
    print_summary
)

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OW_BASE          = "https://api.openweathermap.org/data/2.5"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

CITIES = {
    "Delhi":       (28.6139, 77.2090),
    "Mumbai":      (19.0760, 72.8777),
    "Kolkata":     (22.5726, 88.3639),
    "Bengaluru":   (12.9716, 77.5946),
    "Chennai":     (13.0827, 80.2707),
    "Hyderabad":   (17.3850, 78.4867),
    "Ahmedabad":   (23.0225, 72.5714),
    "Jaipur":      (26.9124, 75.7873),
    "Pune":        (18.5204, 73.8567),
    "Guwahati":    (26.1445, 91.7362),
    "Shillong":    (25.5788, 91.8933),
    "Bhubaneswar": (20.2961, 85.8245),
    "Patna":       (25.5941, 85.1376),
    "Bhopal":      (23.2599, 77.4126),
    "Chandigarh":  (30.7333, 76.7794),
}


def unix_to_iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ── AQI — OpenWeather Air Pollution API (unchanged) ────────────────────────

def fetch_aqi_for_city(city, lat, lng, days):
    end_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    fetched  = now_utc()

    try:
        r = requests.get(
            f"{OW_BASE}/air_pollution/history",
            params={"lat": lat, "lon": lng, "start": start_ts, "end": end_ts,
                     "appid": OPENWEATHER_API_KEY},
            timeout=30
        )
        r.raise_for_status()
        records = r.json().get("list", [])
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Timeout fetching AQI for {city}.")
        return []
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] No internet connection for {city}.")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"  [ERROR] HTTP error for {city} AQI: {e}")
        return []
    except Exception as e:
        print(f"  [ERROR] Unexpected error for {city} AQI: {e}")
        return []

    if not records:
        print(f"  No AQI records returned for {city}.")
        return []

    rows = []
    for rec in records:
        comp = rec.get("components", {})
        rows.append({
            "city": city, "latitude": lat, "longitude": lng,
            "aqi": rec.get("main", {}).get("aqi"),
            "pm25": comp.get("pm2_5"), "pm10": comp.get("pm10"),
            "no2": comp.get("no2"), "o3": comp.get("o3"),
            "timestamp": unix_to_iso(rec["dt"]), "fetched_at": fetched
        })
    return rows


# ── Weather — Open-Meteo (historical, hourly, free, no key) ───────────────

def fetch_weather_history(city, lat, lng, days):
    """Pulls hourly weather for the past `days` days. ~2 day lag for data availability."""
    end_date   = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    fetched    = now_utc()

    try:
        r = requests.get(
            OPENMETEO_ARCHIVE,
            params={
                "latitude": lat, "longitude": lng,
                "start_date": start_date, "end_date": end_date,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
                "wind_speed_unit": "ms",
                "timezone": "UTC"
            },
            timeout=30
        )
        r.raise_for_status()
        data = r.json().get("hourly", {})
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Timeout fetching weather history for {city}.")
        return []
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] No internet connection for {city}.")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"  [ERROR] HTTP error for {city} weather history: {e}")
        return []
    except Exception as e:
        print(f"  [ERROR] Unexpected error for {city} weather history: {e}")
        return []

    times = data.get("time", [])
    if not times:
        print(f"  No weather history returned for {city}.")
        return []

    rows = []
    for i, t in enumerate(times):
        rows.append({
            "city": city,
            "temperature_c":  data["temperature_2m"][i],
            "humidity_pct":   data["relative_humidity_2m"][i],
            "wind_speed_ms":  data["wind_speed_10m"][i],
            "wind_deg":       data["wind_direction_10m"][i],
            "rainfall_1h_mm": data["precipitation"][i] or 0.0,
            "condition":      None,
            "timestamp":      t.replace("T", " ") + ":00",
            "fetched_at":     fetched
        })
    return rows


def fetch_weather_current(city, lat, lng):
    """Pulls the current live weather snapshot."""
    fetched = now_utc()
    try:
        r = requests.get(
            OPENMETEO_FORECAST,
            params={
                "latitude": lat, "longitude": lng,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
                "wind_speed_unit": "ms",
                "timezone": "UTC"
            },
            timeout=15
        )
        r.raise_for_status()
        cur = r.json().get("current", {})
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Timeout fetching current weather for {city}.")
        return None
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] No internet connection for {city}.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"  [ERROR] HTTP error for {city} current weather: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Unexpected error for {city} current weather: {e}")
        return None

    if not cur:
        return None

    ts = cur.get("time", "").replace("T", " ")
    return {
        "city": city,
        "temperature_c":  cur.get("temperature_2m"),
        "humidity_pct":   cur.get("relative_humidity_2m"),
        "wind_speed_ms":  cur.get("wind_speed_10m"),
        "wind_deg":       cur.get("wind_direction_10m"),
        "rainfall_1h_mm": cur.get("precipitation") or 0.0,
        "condition":      None,
        "timestamp":      ts + ":00" if len(ts) == 16 else ts,
        "fetched_at":     fetched
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not OPENWEATHER_API_KEY:
        print("ERROR: OPENWEATHER_API_KEY not found in .env file.")
        return

    init_db()
    history_days = get_history_days()
    is_first_run = history_days == 180

    print("\n" + "=" * 55)
    print(f"VAYU — {'first run: ' + str(history_days) + ' days history' if is_first_run else f'updating last {history_days} days'}")
    print("=" * 55)

    total_aqi, total_wx = 0, 0

    for city, (lat, lng) in CITIES.items():
        print(f"\n[AQI] {city}")
        aqi_rows = fetch_aqi_for_city(city, lat, lng, history_days)
        if aqi_rows:
            inserted = insert_aqi_rows(aqi_rows)
            total_aqi += inserted
            print(f"  Fetched {len(aqi_rows)} — {inserted} new saved")
        time.sleep(1)

        print(f"[Weather History] {city}")
        wx_hist_rows = fetch_weather_history(city, lat, lng, history_days)
        if wx_hist_rows:
            inserted = insert_weather_rows(wx_hist_rows)
            total_wx += inserted
            print(f"  Fetched {len(wx_hist_rows)} — {inserted} new saved")
        time.sleep(1)

        print(f"[Weather Current] {city}")
        wx_cur = fetch_weather_current(city, lat, lng)
        if wx_cur:
            inserted = insert_weather_rows([wx_cur])
            total_wx += inserted
            print(f"  {wx_cur['temperature_c']}°C  Wind {wx_cur['wind_speed_ms']} m/s"
                  + ("  [saved]" if inserted else "  [duplicate]"))
        time.sleep(1)

    print("\n" + "=" * 55)
    print("Done.")
    print(f"  AQI rows inserted     : {total_aqi}")
    print(f"  Weather rows inserted : {total_wx}")
    print_summary()
    print("=" * 55)


if __name__ == "__main__":
    main()