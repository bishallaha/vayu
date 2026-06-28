from dotenv import load_dotenv
import os
import sys
import requests
import time
from datetime import datetime, timedelta, timezone

#Importing Database Helpers, i.e. sibling file database.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import (
    init_db, get_history_days,
    insert_aqi_rows, insert_weather_rows,
    print_summary
)

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OW_BASE             = "https://api.openweathermap.org/data/2.5"

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

#General mapping of city names to OpenWeatherMap names
OWM_CITY_MAP = {
    "Bengaluru": "Bangalore",
    "Kolkata":   "Calcutta",
}


#Helpers
def unix_to_iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


#AQI data fetching through OWM Air Pollution API

def fetch_aqi_for_city(city, lat, lng, days):
    """
    Pulls hourly AQI + pollutant history for the past `days` days.
    Endpoint: /data/2.5/air_pollution/history
    Returns a list of row dicts.
    """
    end_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    fetched  = now_utc()

    try:
        r = requests.get(
            f"{OW_BASE}/air_pollution/history",
            params={
                "lat":   lat,
                "lon":   lng,
                "start": start_ts,
                "end":   end_ts,
                "appid": OPENWEATHER_API_KEY
            },
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
        status = e.response.status_code if e.response is not None else "?"
        print(f"  [ERROR] HTTP {status} for {city} AQI: {e}")
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
            "city":       city,
            "latitude":   lat,
            "longitude":  lng,
            "aqi":        rec.get("main", {}).get("aqi"),
            "pm25":       comp.get("pm2_5"),   # note: API uses pm2_5
            "pm10":       comp.get("pm10"),
            "no2":        comp.get("no2"),
            "o3":         comp.get("o3"),
            "timestamp":  unix_to_iso(rec["dt"]),
            "fetched_at": fetched
        })

    return rows


#Weather data fetching through OWM Current Weather API
def fetch_weather_for_city(city):
    """
    Pulls current weather for a city.
    Endpoint: /data/2.5/weather
    Returns a row dict or None on failure.
    """
    owm_name = OWM_CITY_MAP.get(city, city)
    fetched  = now_utc()

    try:
        r = requests.get(
            f"{OW_BASE}/weather",
            params={
                "q":     f"{owm_name},IN",
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=15
        )
        r.raise_for_status()
        data = r.json()

        return {
            "city":           city,
            "temperature_c":  data.get("main", {}).get("temp"),
            "humidity_pct":   data.get("main", {}).get("humidity"),
            "wind_speed_ms":  data.get("wind", {}).get("speed"),
            "wind_deg":       data.get("wind", {}).get("deg"),
            "rainfall_1h_mm": data.get("rain", {}).get("1h", 0.0),
            "condition":      data.get("weather", [{}])[0].get("description"),
            "timestamp":      unix_to_iso(data.get("dt", 0)),
            "fetched_at":     fetched
        }

    except requests.exceptions.Timeout:
        print(f"  [ERROR] Timeout for {city} weather.")
        return None
    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] No internet connection for {city} weather.")
        return None
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 401:
            print(f"  [ERROR] Invalid API key.")
        elif status == 404:
            print(f"  [ERROR] City not found on OpenWeather: {owm_name},IN")
        else:
            print(f"  [ERROR] HTTP {status} for {city} weather: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Unexpected error for {city} weather: {e}")
        return None


def main():
    if not OPENWEATHER_API_KEY:
        print("ERROR: OPENWEATHER_API_KEY not found in .env file.")
        return

    # Initialise DB (creates tables if they don't exist)
    init_db()

    history_days = get_history_days()
    is_first_run = history_days == 180

    #AQI collection
    print("\n" + "=" * 55)
    if is_first_run:
        print(f"VAYU — first run: pulling {history_days} days of AQI history")
        print("  (This may take a minute — runs faster after this)")
    else:
        print(f"VAYU — updating: pulling last {history_days} days of AQI data")
    print("=" * 55)

    total_aqi = 0

    for city, (lat, lng) in CITIES.items():
        print(f"\n[AQI] {city}")
        rows = fetch_aqi_for_city(city, lat, lng, history_days)

        if rows:
            inserted = insert_aqi_rows(rows)
            total_aqi += inserted
            print(f"  Fetched {len(rows)} records — {inserted} new saved to DB")
        else:
            print(f"  Skipped.")

        time.sleep(1)  # be polite to the API

    #Weather collection
    print("\n" + "=" * 55)
    print("VAYU — collecting current weather")
    print("=" * 55)

    total_wx = 0

    for city in CITIES:
        print(f"\n[Weather] {city}")
        row = fetch_weather_for_city(city)

        if row:
            inserted = insert_weather_rows([row])
            total_wx += inserted
            print(
                f"  {row['temperature_c']}°C  |  "
                f"Wind {row['wind_speed_ms']} m/s  |  "
                f"Humidity {row['humidity_pct']}%  |  "
                f"Rain {row['rainfall_1h_mm']} mm"
                + ("  [saved]" if inserted else "  [duplicate, skipped]")
            )
        else:
            print(f"  Skipped.")

        time.sleep(1)

    #Summary
    print("\n" + "=" * 55)
    print("Done.")
    print(f"  AQI rows inserted this run     : {total_aqi}")
    print(f"  Weather rows inserted this run : {total_wx}")
    print_summary()
    print("=" * 55)


if __name__ == "__main__":
    main()