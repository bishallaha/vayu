# src/health_risk.py
"""
WHO-aligned Health Risk Index — 0 to 100 score per city per hour.
Combines AQI category + PM2.5 + NO2 into a single risk score.
Adds demographic advisory flags for children, elderly, asthmatic individuals.

Reads from  : data/vayu_clean.db  →  features table
Writes to   : data/vayu_clean.db  →  health_risk table

Run: python src/health_risk.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DB = os.path.join(ROOT, "data", "vayu_clean.db")

# ── WHO thresholds ─────────────────────────────────────────────────────────
# OpenWeather AQI scale: 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
# PM2.5 in µg/m³ — WHO 24-hour guideline: 15 µg/m³
# NO2 in µg/m³   — WHO annual guideline : 10 µg/m³

# AQI → base risk score (0–60 points)
AQI_BASE_SCORE = {
    1: 10,   # Good
    2: 25,   # Fair
    3: 42,   # Moderate
    4: 60,   # Poor
    5: 75,   # Very Poor / Hazardous
}

# AQI → label for display
AQI_LABEL = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Hazardous"
}

# AQI → hex color for map / dashboard
AQI_COLOR = {
    1: "#00C853",   # green
    2: "#FFD600",   # yellow
    3: "#FF6D00",   # orange
    4: "#D50000",   # red
    5: "#6A1B9A"    # purple
}

# PM2.5 contribution bands → additional points (0–15)
def pm25_score(pm25):
    if pd.isna(pm25):      return 0
    if pm25 <= 5:          return 0
    elif pm25 <= 15:       return 3
    elif pm25 <= 25:       return 6
    elif pm25 <= 50:       return 9
    elif pm25 <= 75:       return 12
    else:                  return 15

# NO2 contribution bands → additional points (0–10)
def no2_score(no2):
    if pd.isna(no2):       return 0
    if no2 <= 10:          return 0
    elif no2 <= 25:        return 2
    elif no2 <= 50:        return 4
    elif no2 <= 100:       return 6
    elif no2 <= 200:       return 8
    else:                  return 10

# O3 contribution bands → additional points (0–5)
def o3_score(o3):
    if pd.isna(o3):        return 0
    if o3 <= 60:           return 0
    elif o3 <= 100:        return 2
    elif o3 <= 140:        return 3
    else:                  return 5


# ── Risk label from final score ────────────────────────────────────────────

def risk_label(score):
    if score <= 20:   return "Low Risk"
    elif score <= 40: return "Moderate Risk"
    elif score <= 60: return "High Risk"
    elif score <= 75: return "Very High Risk"
    else:             return "Hazardous"


# ── Risk level 1–5 for map coloring ───────────────────────────────────────

def risk_level(score):
    if score <= 20:   return 1
    elif score <= 40: return 2
    elif score <= 60: return 3
    elif score <= 75: return 4
    else:             return 5


# ── Demographic flags ──────────────────────────────────────────────────────
# Thresholds based on WHO sensitivity guidelines for vulnerable groups

def flag_children(score, pm25):
    """
    Children (under 14) are more sensitive to PM2.5 and O3.
    Flag at lower thresholds than general population.
    """
    triggered = score >= 35 or (not pd.isna(pm25) and pm25 > 12)
    if not triggered:
        return False, None
    if score >= 70:
        return True, "Keep children indoors. Avoid all outdoor activity."
    elif score >= 50:
        return True, "Limit outdoor play for children. Avoid prolonged exertion."
    else:
        return True, "Sensitive children should reduce outdoor activity."


def flag_elderly(score, no2):
    """
    Elderly (65+) have reduced lung capacity and cardiovascular sensitivity.
    NO2 disproportionately affects this group.
    """
    triggered = score >= 30 or (not pd.isna(no2) and no2 > 25)
    if not triggered:
        return False, None
    if score >= 65:
        return True, "Elderly should remain indoors. Seek medical advice if symptomatic."
    elif score >= 45:
        return True, "Elderly should avoid outdoor activity and keep windows closed."
    else:
        return True, "Elderly with heart or lung conditions should reduce time outdoors."


def flag_asthmatic(score, pm25, o3):
    """
    Asthmatic individuals react strongly to PM2.5 and O3.
    PM2.5 triggers inflammation; O3 causes airway irritation.
    """
    triggered = (
        score >= 25 or
        (not pd.isna(pm25) and pm25 > 10) or
        (not pd.isna(o3)   and o3  > 60)
    )
    if not triggered:
        return False, None
    if score >= 60:
        return True, "Asthmatic individuals must stay indoors. Keep rescue inhaler accessible."
    elif score >= 40:
        return True, "Asthmatics should avoid outdoor activity. Use preventive inhaler if prescribed."
    else:
        return True, "Asthmatic individuals should limit outdoor exposure and monitor symptoms."


# ── General public advisory ────────────────────────────────────────────────

def general_advisory(score):
    if score <= 20:
        return "Air quality is good. Safe for all outdoor activities."
    elif score <= 40:
        return "Air quality is acceptable. Unusually sensitive individuals may experience discomfort."
    elif score <= 60:
        return "Sensitive groups should reduce prolonged outdoor exertion."
    elif score <= 75:
        return "Everyone should reduce outdoor activity. Sensitive groups should stay indoors."
    else:
        return "Hazardous air quality. Everyone should avoid all outdoor activity."


# ── Compute health risk for one row ───────────────────────────────────────

def compute_health_risk(row):
    aqi   = row.get("aqi")
    pm25  = row.get("pm25")
    no2   = row.get("no2")
    o3    = row.get("o3")

    # Base score from AQI (handles null aqi gracefully)
    base = AQI_BASE_SCORE.get(int(aqi), 42) if pd.notna(aqi) else 42

    # Add pollutant contributions
    total = base + pm25_score(pm25) + no2_score(no2) + o3_score(o3)
    total = int(min(total, 100))  # cap at 100

    # Demographic flags
    child_flag,    child_msg    = flag_children(total, pm25)
    elderly_flag,  elderly_msg  = flag_elderly(total, no2)
    asthma_flag,   asthma_msg   = flag_asthmatic(total, pm25, o3)

    return {
        "city":                  row["city"],
        "timestamp":             row["timestamp"],
        "aqi":                   aqi,
        "pm25":                  pm25,
        "no2":                   no2,
        "o3":                    o3,
        "risk_score":            total,
        "risk_percentage":       total,
        "risk_label":            risk_label(total),
        "risk_level":            risk_level(total),
        "aqi_label":             AQI_LABEL.get(int(aqi), "Unknown") if pd.notna(aqi) else "Unknown",
        "aqi_color":             AQI_COLOR.get(int(aqi), "#888888") if pd.notna(aqi) else "#888888",
        "general_advisory":      general_advisory(total),
        "flag_children":         int(child_flag),
        "flag_elderly":          int(elderly_flag),
        "flag_asthmatic":        int(asthma_flag),
        "advisory_children":     child_msg   or "",
        "advisory_elderly":      elderly_msg or "",
        "advisory_asthmatic":    asthma_msg  or "",
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_DB):
        print("ERROR: data/vayu_clean.db not found.")
        return

    print("=" * 60)
    print("VAYU — WHO-aligned Health Risk Index + Demographic Flags")
    print("=" * 60)

    conn = sqlite3.connect(CLEAN_DB)
    df   = pd.read_sql(
        """
        SELECT city, timestamp, aqi, pm25, pm10, no2, o3
        FROM   features
        ORDER  BY city, timestamp
        """,
        conn, parse_dates=["timestamp"]
    )
    conn.close()

    print(f"Rows loaded : {len(df):,}")
    print(f"Cities      : {df['city'].nunique()}")
    print("\nComputing health risk scores...")

    results = df.apply(compute_health_risk, axis=1, result_type="expand")

    conn = sqlite3.connect(CLEAN_DB)
    results.to_sql("health_risk", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nSaved {len(results):,} rows → health_risk table")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n── Average WHO-aligned Health Risk per City ─────────────────")
    print(f"\n  {'City':<14} {'Avg Risk %':>12}  {'Risk Level':>16}  "
        f"{'% Child Flag':>13}  {'% Elderly':>10}  {'% Asthma':>10}")
    print(f"  {'-'*80}")

    for city in sorted(results["city"].unique()):
        city_df     = results[results["city"] == city]
        avg_score   = city_df["risk_score"].mean()
        avg_label   = risk_label(avg_score)
        child_pct   = city_df["flag_children"].mean() * 100
        elderly_pct = city_df["flag_elderly"].mean() * 100
        asthma_pct  = city_df["flag_asthmatic"].mean() * 100

        print(f"  {city:<14} {avg_score:>11.1f}%  {avg_label:>16}  "
              f"{child_pct:>12.1f}%  {elderly_pct:>9.1f}%  {asthma_pct:>9.1f}%")

    # ── Risk distribution ─────────────────────────────────────────────
    print("\n── Overall Risk Score Distribution ──────────────────────")
    print(f"\n  {'Risk Label':<20} {'Count':>10}  {'%':>8}")
    print(f"  {'-'*42}")
    dist = results["risk_label"].value_counts()
    for label, count in dist.items():
        pct = count / len(results) * 100
        print(f"  {label:<20} {count:>10,}  {pct:>7.1f}%")

    # ── Sample advisory output ────────────────────────────────────────
    print("\n── Sample Advisory Output (latest reading, Delhi) ───────")
    delhi_latest = (
        results[results["city"] == "Delhi"]
        .sort_values("timestamp")
        .iloc[-1]
    )
    print(f"\n  City           : {delhi_latest['city']}")
    print(f"  Timestamp      : {delhi_latest['timestamp']}")
    print(f"  AQI            : {delhi_latest['aqi']} ({delhi_latest['aqi_label']})")
    print(f"  PM2.5          : {delhi_latest['pm25']} µg/m³")
    print(f"  Risk Score     : {delhi_latest['risk_score']} / 100")
    print(f"  Risk Label     : {delhi_latest['risk_label']}")
    print(f"\n  General        : {delhi_latest['general_advisory']}")
    if delhi_latest["flag_children"]:
        print(f"  Children ⚠     : {delhi_latest['advisory_children']}")
    if delhi_latest["flag_elderly"]:
        print(f"  Elderly ⚠      : {delhi_latest['advisory_elderly']}")
    if delhi_latest["flag_asthmatic"]:
        print(f"  Asthmatic ⚠    : {delhi_latest['advisory_asthmatic']}")

    print("\n" + "=" * 60)
    print("Done. Health Risk Index complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()