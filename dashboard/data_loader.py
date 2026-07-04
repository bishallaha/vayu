# dashboard/data_loader.py
import os, sqlite3
import pandas as pd
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DB = os.path.join(ROOT, "data", "vayu_clean.db")
OW_KEY   = os.getenv("OPENWEATHER_API_KEY")

CITY_COORDS = {
    "Ahmedabad":   (23.0225, 72.5714),
    "Bengaluru":   (12.9716, 77.5946),
    "Bhopal":      (23.2599, 77.4126),
    "Bhubaneswar": (20.2961, 85.8245),
    "Chandigarh":  (30.7333, 76.7794),
    "Chennai":     (13.0827, 80.2707),
    "Delhi":       (28.6139, 77.2090),
    "Guwahati":    (26.1445, 91.7362),
    "Hyderabad":   (17.3850, 78.4867),
    "Jaipur":      (26.9124, 75.7873),
    "Kolkata":     (22.5726, 88.3639),
    "Mumbai":      (19.0760, 72.8777),
    "Patna":       (25.5941, 85.1376),
    "Pune":        (18.5204, 73.8567),
    "Shillong":    (25.5788, 91.8933),
}

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Hazardous"}
AQI_COLORS = {1: "#00E400", 2: "#FFFF00", 3: "#FF7E00", 4: "#FF0000", 5: "#8F3F97"}

def _query(sql, params=()):
    conn = sqlite3.connect(CLEAN_DB)
    df   = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df

def get_all_cities_latest():
    return _query("""
        SELECT h.city, h.aqi, h.pm25, h.no2, h.o3, h.risk_score,
               h.risk_label, h.risk_level, h.aqi_label, h.aqi_color, h.timestamp
        FROM health_risk h
        INNER JOIN (
            SELECT city, MAX(timestamp) AS max_ts
            FROM health_risk GROUP BY city
        ) latest ON h.city = latest.city AND h.timestamp = latest.max_ts
        ORDER BY h.city
    """)

def get_latest_health_risk(city):
    df = _query(
        "SELECT * FROM health_risk WHERE city=? ORDER BY timestamp DESC LIMIT 1",
        (city,)
    )
    return df.iloc[0] if not df.empty else None

def get_prophet_forecast(city):
    return _query(
        "SELECT timestamp, yhat, yhat_lower, yhat_upper FROM prophet_forecasts WHERE city=? ORDER BY timestamp",
        (city,)
    )

def get_xgb_forecast(city):
    return _query(
        "SELECT timestamp, predicted_aqi FROM xgb_regressor_predictions WHERE city=? ORDER BY timestamp",
        (city,)
    )

def get_best_model(city):
    try:
        df = _query(
            "SELECT best_model, best_mae FROM best_model_selection WHERE city=?",
            (city,)
        )
        if not df.empty:
            return df.iloc[0]["best_model"], round(df.iloc[0]["best_mae"], 4)
    except Exception:
        pass
    return "prophet", None

def get_historical_aqi(city, limit_hours=720):
    return _query("""
        SELECT timestamp, aqi, pm25, pm10, no2, o3,
               temperature_c, wind_speed_ms, humidity_pct
        FROM features WHERE city=?
        ORDER BY timestamp DESC LIMIT ?
    """, (city, limit_hours))

def get_shap_data(city):
    return _query("""
        SELECT feature, mean_abs_shap FROM xgb_classifier_shap
        WHERE city=? ORDER BY mean_abs_shap DESC LIMIT 12
    """, (city,))

def get_model_metrics():
    try:
        p  = _query("SELECT city, mae AS prophet_mae, rmse AS prophet_rmse FROM prophet_metrics")
        xr = _query("SELECT city, mae AS xgb_reg_mae FROM xgb_regressor_metrics")
        xc = _query("SELECT city, accuracy, f1_macro FROM xgb_classifier_metrics")
        return p.merge(xr, on="city", how="outer").merge(xc, on="city", how="outer")
    except Exception:
        return pd.DataFrame()

def get_comparison_data(city1, city2):
    return _query("""
        SELECT h.city, h.aqi, h.pm25, h.no2, h.o3, h.risk_score,
               h.risk_label, h.aqi_label, h.aqi_color,
               h.general_advisory, h.flag_children, h.flag_elderly,
               h.flag_asthmatic, h.timestamp
        FROM health_risk h
        INNER JOIN (
            SELECT city, MAX(timestamp) AS max_ts
            FROM health_risk WHERE city IN (?,?)
            GROUP BY city
        ) latest ON h.city=latest.city AND h.timestamp=latest.max_ts
    """, (city1, city2))

def get_hourly_heatmap(city):
    return _query("""
        SELECT hour, day_of_week, AVG(aqi) AS avg_aqi
        FROM features WHERE city=?
        GROUP BY hour, day_of_week
        ORDER BY day_of_week, hour
    """, (city,))

def get_monthly_trend(city):
    return _query("""
        SELECT month, AVG(aqi) AS avg_aqi, AVG(pm25) AS avg_pm25
        FROM features WHERE city=?
        GROUP BY month ORDER BY month
    """, (city,))

def get_live_aqi(city):
    if city not in CITY_COORDS:
        return None
    lat, lon = CITY_COORDS[city]
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": OW_KEY},
            timeout=10
        )
        r.raise_for_status()
        d    = r.json()["list"][0]
        comp = d["components"]
        aqi  = d["main"]["aqi"]
        return {
            "aqi":       aqi,
            "pm25":      comp.get("pm2_5"),
            "pm10":      comp.get("pm10"),
            "no2":       comp.get("no2"),
            "o3":        comp.get("o3"),
            "aqi_label": AQI_LABELS.get(aqi, "Unknown"),
            "aqi_color": AQI_COLORS.get(aqi, "#888"),
            "timestamp": datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC"),
        }
    except Exception:
        return None

def get_yesterday_aqi(city):
    df = _query(
        "SELECT aqi FROM features WHERE city=? ORDER BY timestamp DESC LIMIT 48",
        (city,)
    )
    return float(df.iloc[-1]["aqi"]) if len(df) >= 24 else None