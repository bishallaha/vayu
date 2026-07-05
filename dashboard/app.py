# dashboard/app.py

import streamlit as st

st.set_page_config(
    page_title="VAYU — Air Quality Intelligence",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import sys, os, subprocess
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    get_all_cities_latest, get_latest_health_risk,
    get_prophet_forecast, get_xgb_forecast, get_best_model,
    get_historical_aqi, get_shap_data, get_model_metrics,
    get_comparison_data, get_hourly_heatmap, get_monthly_trend,
    get_live_aqi, get_yesterday_aqi,
    CITY_COORDS, AQI_LABELS, AQI_COLORS
)
from styles import VAYU_CSS

st.markdown(VAYU_CSS, unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────

CITIES     = sorted(CITY_COORDS.keys())
PAGES      = ["Overview", "City Comparison", "Historical Analytics"]

# ← Replace yourusername with your actual GitHub username
GITHUB_URL = "https://github.com/bishallaha/vayu"

FEATURE_NAMES = {
    "pm25":          "PM2.5 Concentration",
    "pm10":          "PM10 Concentration",
    "no2":           "Nitrogen Dioxide",
    "o3":            "Ozone Level",
    "aqi":           "Current AQI",
    "aqi_roll_6h":   "6-Hr AQI Average",
    "aqi_roll_24h":  "24-Hr AQI Average",
    "aqi_lag_1h":    "AQI — 1 Hour Prior",
    "aqi_lag_3h":    "AQI — 3 Hours Prior",
    "aqi_lag_24h":   "AQI — 24 Hours Prior",
    "temperature_c": "Temperature",
    "humidity_pct":  "Humidity",
    "wind_speed_ms": "Wind Speed",
    "wind_deg":      "Wind Direction",
    "rainfall_1h_mm":"Rainfall",
    "hour":          "Hour of Day",
    "day_of_week":   "Day of Week",
    "month":         "Month",
    "is_weekend":    "Weekend",
}

WORSENS  = {"pm25","pm10","no2","o3","aqi","aqi_lag_1h","aqi_lag_3h",
            "aqi_lag_24h","aqi_roll_6h","aqi_roll_24h","humidity_pct"}
IMPROVES = {"wind_speed_ms","rainfall_1h_mm"}

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
DAY_NAMES   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# ── Session state ──────────────────────────────────────────────────────────

if "city" not in st.session_state: st.session_state.city = "Delhi"
if "page" not in st.session_state: st.session_state.page = "Overview"

# ── Helpers ────────────────────────────────────────────────────────────────

def aqi_badge(label, color):
    tc = "#111" if color in ("#00E400","#FFFF00") else "#fff"
    return f'<span class="mc-badge" style="background:{color};color:{tc}">{label}</span>'

def risk_color(score):
    if score is None: return "#9CA3AF"
    if score <= 20:   return "#10B981"
    if score <= 40:   return "#F59E0B"
    if score <= 60:   return "#F97316"
    if score <= 75:   return "#DC2626"
    return "#7C3AED"

def sep():    st.markdown('<div class="sep"></div>',    unsafe_allow_html=True)
def sep_sm(): st.markdown('<div class="sep-sm"></div>', unsafe_allow_html=True)

def section_hd(title, subtitle=""):
    sub = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(f'<div class="shd"><h2>{title}</h2>{sub}</div>', unsafe_allow_html=True)

def metric_card(label, icon, value, delta_html, badge_html, accent=None):
    bar = f'<div class="mc-accent" style="background:{accent}"></div>' if accent else ""
    return f"""
    <div class="mc">
      {bar}
      <div class="mc-lbl">{icon}&nbsp;{label}</div>
      <div class="mc-val">{value}</div>
      <div class="mc-btm">{delta_html}{badge_html}</div>
    </div>"""

# ── Charts ─────────────────────────────────────────────────────────────────

CHART_BASE = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=0, r=0, t=8, b=0),
)
Y_AQI = dict(
    showgrid=True, gridcolor="#F3F4F6", title=None,
    range=[0.5, 5.5], tickvals=[1,2,3,4,5],
    ticktext=["Good","Fair","Moderate","Poor","Hazardous"],
    color="#9CA3AF", tickfont=dict(size=11, family="Inter"),
)
X_BASE = dict(showgrid=False, color="#9CA3AF", title=None,
              tickfont=dict(size=11, family="Inter"))


def chart_forecast(df, model_name):
    if df is None or df.empty: return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    has_ci = "yhat_lower" in df.columns
    ycol   = "yhat" if has_ci or "yhat" in df.columns else "predicted_aqi"

    fig = go.Figure()
    if has_ci:
        fig.add_trace(go.Scatter(
            x=list(df["timestamp"]) + list(df["timestamp"][::-1]),
            y=list(df["yhat_upper"]) + list(df["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[ycol],
        mode="lines", line=dict(color="#2563EB", width=2.5),
        name=f"{model_name} Forecast",
        hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI: %{y:.1f}<extra></extra>",
    ))
    fig.add_vline(x=datetime.now(timezone.utc), line_dash="dot",
                  line_color="#D1D5DB", line_width=1.5)

    fig.update_layout(
        **CHART_BASE, height=300,
        xaxis=X_BASE, yaxis=Y_AQI,
        legend=dict(orientation="h", y=-0.22, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, family="Inter")),
    )
    return fig


def chart_shap(shap_df):
    if shap_df is None or shap_df.empty: return None
    feats  = shap_df["feature"].tolist()
    vals   = shap_df["mean_abs_shap"].tolist()
    names  = [FEATURE_NAMES.get(f, f) for f in feats]
    colors = ["#DC2626" if f in WORSENS else "#2563EB" if f in IMPROVES else "#E5E7EB"
              for f in feats]

    fig = go.Figure(go.Bar(
        x=vals[::-1], y=names[::-1], orientation="h",
        marker_color=colors[::-1],
        hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_BASE, height=max(300, len(feats)*28),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, color="#374151",
                   tickfont=dict(size=12, family="Inter")),
        bargap=0.38,
    )
    return fig


def chart_historical(df):
    if df is None or df.empty: return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    peak_idx = df["aqi"].idxmax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        mode="lines", line=dict(color="#2563EB", width=1.8),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.04)",
        name="AQI Level",
        hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[df.loc[peak_idx, "timestamp"]], y=[df.loc[peak_idx, "aqi"]],
        mode="markers", marker=dict(color="#DC2626", size=7),
        name=f"Peak: {df.loc[peak_idx,'aqi']}",
        hovertemplate="Peak AQI: %{y}<extra></extra>",
    ))
    fig.add_hline(y=df["aqi"].mean(), line_dash="dot",
                  line_color="#D1D5DB", line_width=1,
                  annotation_text=f"Avg {df['aqi'].mean():.1f}",
                  annotation_font=dict(size=11, color="#9CA3AF"))

    fig.update_layout(
        **CHART_BASE, height=260,
        xaxis=X_BASE, yaxis=Y_AQI,
        legend=dict(orientation="h", y=-0.22, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, family="Inter")),
    )
    return fig


def chart_comparison(h1, h2, city1, city2):
    fig = go.Figure()
    for df, name, color in [(h1, city1, "#2563EB"), (h2, city2, "#DC2626")]:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["aqi"],
            mode="lines", line=dict(color=color, width=2),
            name=name,
            hovertemplate=f"<b>{name}</b> · %{{x|%d %b}}<br>AQI: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        **CHART_BASE, height=280,
        xaxis=X_BASE, yaxis=Y_AQI,
        legend=dict(orientation="h", y=-0.22, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, family="Inter")),
    )
    return fig


def chart_heatmap(hm):
    if hm is None or hm.empty: return None
    pivot = hm.pivot(index="day_of_week", columns="hour", values="avg_aqi").fillna(2)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[DAY_NAMES[d] for d in pivot.index],
        colorscale=[[0,"#00E400"],[0.25,"#FFFF00"],[0.5,"#FF7E00"],
                    [0.75,"#FF0000"],[1,"#8F3F97"]],
        zmin=1, zmax=5,
        hovertemplate="Hour: %{x}<br>Day: %{y}<br>AQI: %{z:.2f}<extra></extra>",
        colorbar=dict(tickvals=[1,2,3,4,5],
                      ticktext=["Good","Fair","Moderate","Poor","Haz."],
                      thickness=10, len=0.8,
                      tickfont=dict(size=10, family="Inter")),
    ))
    layout = CHART_BASE.copy()
    layout["margin"] = dict(l=0, r=0, t=0, b=0)

    fig.update_layout(
        **layout,
        height=260,
        xaxis=dict(
            tickfont=dict(size=10, family="Inter"),
            side="bottom"
        ),
        yaxis=dict(
            tickfont=dict(size=11, family="Inter")
        )
    )
    return fig


def chart_monthly(mt):
    if mt is None or mt.empty: return None
    mt = mt.copy()
    mt["month_name"] = mt["month"].map(MONTH_NAMES)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mt["month_name"], y=mt["avg_aqi"],
        name="Avg AQI", marker_color="#2563EB",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=mt["month_name"], y=mt["avg_pm25"],
        name="Avg PM2.5", mode="lines+markers",
        line=dict(color="#DC2626", width=2), yaxis="y2",
        hovertemplate="%{x}: %{y:.1f} µg/m³<extra></extra>",
    ))
    layout = CHART_BASE.copy()
    layout["margin"] = dict(l=0, r=0, t=0, b=0)

    fig.update_layout(
        **layout,
        height=260,
        xaxis=dict(
            showgrid=False,
            color="#9CA3AF"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            title=None,
            color="#9CA3AF",
            tickfont=dict(size=11, family="Inter"),
        ),
        yaxis2=dict(
            title=None,
            overlaying="y",
            side="right",
            color="#DC2626",
            showgrid=False,
            tickfont=dict(size=11, family="Inter"),
        ),
        legend=dict(
            orientation="h",
            y=-0.22,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, family="Inter"),
        ),
        bargap=0.35,
    )
    return fig


# ── Map ─────────────────────────────────────────────────────────────────────

def build_map(cities_df, selected):
    m = folium.Map(location=[22.5, 82.0], zoom_start=5,
                   tiles="CartoDB Positron", zoom_control=False,
                   scrollWheelZoom=True, attributionControl=False)

    for _, row in cities_df.iterrows():
        city     = row["city"]
        lat, lon = CITY_COORDS.get(city, (22, 82))
        aqi_i    = int(row["aqi"]) if pd.notna(row["aqi"]) else 2
        color    = AQI_COLORS.get(aqi_i, "#888")
        is_sel   = city == selected
        tc       = "#111" if aqi_i <= 2 else "#fff"

        folium.CircleMarker(
            location=[lat, lon],
            radius=22 if is_sel else 14,
            color="white", weight=3 if is_sel else 1.5,
            fill=True, fill_color=color, fill_opacity=0.9,
            tooltip=folium.Tooltip(
                f"<div style='font-family:Inter,sans-serif;min-width:110px'>"
                f"<b style='font-size:13px'>{city}</b><br>"
                f"<span style='font-size:12px'>AQI: {row['aqi_label']}</span><br>"
                f"<span style='font-size:12px'>Risk: {row['risk_score']:.0f}/100</span></div>",
                sticky=True
            ),
        ).add_to(m)

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=(f'<div style="font-family:Inter,sans-serif;font-size:9px;'
                      f'font-weight:700;color:{tc};text-align:center;'
                      f'width:44px;margin-left:-22px;margin-top:7px">'
                      f'{city[:3].upper()}</div>'),
                icon_size=(44, 16), icon_anchor=(22, -8),
            )
        ).add_to(m)

    return m


# ── Pipeline ────────────────────────────────────────────────────────────────

def run_refresh():
    scripts = [
        ("Collecting live data",          "src/collect.py"),
        ("Cleaning database",             "src/clean.py"),
        ("Engineering features",          "src/features.py"),
        ("Updating forecasts",            "src/prophet_model.py"),
        ("Recalculating health risk",     "src/health_risk.py"),
    ]
    bar = st.progress(0)
    msg = st.empty()
    ok  = True
    for i, (label, script) in enumerate(scripts):
        msg.markdown(f'<p style="font-size:13px;color:#6B7280">⏳ {label}…</p>',
                     unsafe_allow_html=True)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, script)],
            capture_output=True, text=True, timeout=600
        )
        bar.progress((i+1) / len(scripts))
        if r.returncode != 0:
            st.error(f"Pipeline failed at `{script}`.")
            ok = False; break
    msg.empty(); bar.empty()
    return ok


# ── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def c_cities():          return get_all_cities_latest()

@st.cache_data(ttl=3600)
def c_live(city):        return get_live_aqi(city)

@st.cache_data(ttl=3600)
def c_yesterday(city):   return get_yesterday_aqi(city)

@st.cache_data(ttl=3600)
def c_shap(city):        return get_shap_data(city)

@st.cache_data(ttl=3600)
def c_hist(city, hours): return get_historical_aqi(city, hours)

@st.cache_data(ttl=3600)
def c_forecast(city):
    model, mae_val = get_best_model(city)
    if model == "prophet":
        df = get_prophet_forecast(city)
    else:
        df = get_xgb_forecast(city)
        if not df.empty and "predicted_aqi" in df.columns:
            df = df.rename(columns={"predicted_aqi": "yhat"})
    return df, model.capitalize(), mae_val


# ── Header ──────────────────────────────────────────────────────────────────

def render_header():
    left, right = st.columns([5, 1])
    with left:
        st.markdown(
            f'<div class="vayu-brand">'
            f'<h1>VA<span>YU</span></h1>'
            f'<p class="vayu-brand-sub">Real-Time Air Quality Intelligence · India</p>'
            f'<a href="{GITHUB_URL}" target="_blank" class="vayu-github">⊹ View on GitHub</a>'
            f'</div>',
            unsafe_allow_html=True
        )
    with right:
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        if st.button("↻  Refresh", key="refresh_btn", use_container_width=True):
            with st.spinner("Updating pipeline…"):
                if run_refresh():
                    st.cache_data.clear()
                    st.success("Data updated successfully.")
                    st.rerun()


# ── Navigation row ───────────────────────────────────────────────────────────

def render_nav():
    nav_col, ts_col = st.columns([3, 2])
    with nav_col:
        page = st.selectbox(
            "Navigate to",
            PAGES,
            index=PAGES.index(st.session_state.page),
            key="page_nav",
        )
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
    with ts_col:
        ts = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
        st.markdown(
            f'<div style="padding-top:28px;text-align:right">'
            f'<span class="ts-pill">🕐 {ts}</span></div>',
            unsafe_allow_html=True
        )


# ── Overview ─────────────────────────────────────────────────────────────────

def render_overview():
    city = st.session_state.city

    # ── City selector ─────────────────────────────────────────────────────
    new_city = st.selectbox(
        "Select or search a city",
        CITIES,
        index=CITIES.index(city),
        key="city_ov",
    )
    if new_city != city:
        st.session_state.city = new_city
        st.rerun()

    sep_sm()

    # ── Fetch data ────────────────────────────────────────────────────────
    live    = c_live(city)
    db_risk = get_latest_health_risk(city)
    yest    = c_yesterday(city)

    aqi_val   = (live or {}).get("aqi")   or (db_risk["aqi"]      if db_risk is not None else None)
    aqi_label = (live or {}).get("aqi_label","") or (db_risk["aqi_label"] if db_risk is not None else "—")
    aqi_color = (live or {}).get("aqi_color","#888") or (db_risk["aqi_color"] if db_risk is not None else "#888")
    pm25      = (live or {}).get("pm25")  or (db_risk["pm25"]  if db_risk is not None else None)
    risk_sc   = int(db_risk["risk_score"]) if db_risk is not None else None

    # ── Alert ─────────────────────────────────────────────────────────────
    if aqi_val and aqi_val >= 4:
        st.markdown(f"""
        <div class="alert-banner">
          <span class="alert-icon">⚠️</span>
          <div>
            <p class="alert-title">Severe Air Pollution Alert — {city}</p>
            <p class="alert-body">Current AQI is classified as <b>{aqi_label}</b>.
               Avoid all outdoor activities. Sensitive groups must stay indoors.
               Keep windows closed and use air purification where available.</p>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── 4 Metric cards ────────────────────────────────────────────────────
    forecast_df, model_name, model_mae = c_forecast(city)

    if yest and aqi_val:
        diff = float(aqi_val) - float(yest)
        if   diff >  0.1: delta = f'<span class="mc-up">↑ {abs(diff):.1f} since yesterday</span>'
        elif diff < -0.1: delta = f'<span class="mc-down">↓ {abs(diff):.1f} since yesterday</span>'
        else:             delta = '<span class="mc-flat">→ Similar to yesterday</span>'
    else:
        delta = '<span class="mc-flat">Live reading</span>'

    if not forecast_df.empty:
        ycol  = "yhat" if "yhat" in forecast_df.columns else "predicted_aqi"
        f_max = forecast_df[ycol].max()
        f_min = forecast_df[ycol].min()
        if aqi_val and f_max > float(aqi_val) + 0.4:
            t_val, t_delta = "Worsening", '<span class="mc-up">↑ Expected to rise</span>'
        elif aqi_val and f_min < float(aqi_val) - 0.4:
            t_val, t_delta = "Improving",  '<span class="mc-down">↓ Expected to improve</span>'
        else:
            t_val, t_delta = "Stable",    '<span class="mc-flat">→ No major change</span>'
    else:
        t_val, t_delta = "—", ""

    rc = risk_color(risk_sc)

    c1, c2, c3, c4 = st.columns(4)
    card_data = [
        (c1, metric_card(
            "CURRENT AQI", "💨",
            str(aqi_val) if aqi_val else "—",
            delta,
            aqi_badge(aqi_label, aqi_color),
            accent=aqi_color)),
        (c2, metric_card(
            "ESTIMATED HEALTH RISK", "🫀",
            f"{risk_sc}%" if risk_sc is not None else "—",
            '<span class="mc-flat">WHO-aligned index</span>',
            f'<span class="mc-badge" style="background:{rc};color:white">'
            f'{db_risk["risk_label"] if db_risk is not None else ""}</span>',
            accent=rc)),
        (c3, metric_card(
            "48-HR FORECAST TREND", "📊",
            f'<span class="mc-val-sm">{t_val}</span>',
            t_delta,
            f'<span class="mc-badge" style="background:#EFF6FF;color:#2563EB">'
            f'{model_name}' + (f' · {model_mae:.3f}' if model_mae else '') + '</span>',
            accent="#2563EB")),
        (c4, metric_card(
            "PM2.5 LEVEL", "🌫️",
            f"{pm25:.1f}" if pm25 else "—",
            '<span class="mc-flat">µg/m³</span>',
            f'<span class="mc-badge" style="'
            f'background:{"#FEF2F2;color:#DC2626" if pm25 and pm25 > 15 else "#F0FDF4;color:#059669"}">'
            f'{"Above WHO limit" if pm25 and pm25 > 15 else "WHO Safe range"}</span>',
            accent="#DC2626" if pm25 and pm25 > 15 else "#10B981")),
    ]
    for col, html in card_data:
        with col:
            st.markdown(html, unsafe_allow_html=True)

    sep()

    # ── Map + Forecast ────────────────────────────────────────────────────
    map_col, fc_col = st.columns([6, 4], gap="large")

    with map_col:
        section_hd("Air Quality Map — India",
                   "15 monitored cities · Click any city to select it")
        cities_df = c_cities()
        m         = build_map(cities_df, city)
        result    = st_folium(m, height=390, use_container_width=True,
                              returned_objects=["last_object_clicked"])

        if result and result.get("last_object_clicked"):
            clat = result["last_object_clicked"].get("lat")
            clng = result["last_object_clicked"].get("lng")
            if clat and clng:
                nearest, bd = city, float("inf")
                for cn, (la, lo) in CITY_COORDS.items():
                    d = ((clat-la)**2 + (clng-lo)**2)**0.5
                    if d < bd: bd, nearest = d, cn
                if nearest != st.session_state.city:
                    st.session_state.city = nearest
                    st.rerun()

    with fc_col:
        mae_str = f" · MAE {model_mae:.3f}" if model_mae else ""
        section_hd("48-Hour Air Quality Forecast",
                   f"Model: {model_name}{mae_str}")
        fig = chart_forecast(forecast_df, model_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Forecast data not available for this city.")
        st.markdown(
            '<div class="fc-note"><p>Shaded band shows the model\'s confidence range. '
            'Best-performing model is automatically selected per city based on held-out '
            'test accuracy.</p></div>',
            unsafe_allow_html=True
        )

    sep()

    # ── Health Advisory ───────────────────────────────────────────────────
    section_hd("Today's Health Advisory",
               "Personalised guidance based on current AQI and WHO-aligned health risk score")

    if db_risk is not None:
        st.markdown(
            f'<div class="advisory-main">{db_risk["general_advisory"]}</div>',
            unsafe_allow_html=True
        )
        f1, f2, f3 = st.columns(3)

        def flag_block(triggered, label, icon, advice):
            if triggered:
                return (f'<div class="flag-card flag-alert">'
                        f'<p class="flag-alert-t">{icon} {label}</p>'
                        f'<p class="flag-alert-b">{advice}</p></div>')
            return (f'<div class="flag-card flag-ok">'
                    f'<p class="flag-ok-t">✓ {label} — Safe</p></div>')

        with f1:
            st.markdown(flag_block(
                db_risk["flag_children"], "Children (under 14)", "👶",
                db_risk.get("advisory_children", "")), unsafe_allow_html=True)
        with f2:
            st.markdown(flag_block(
                db_risk["flag_elderly"], "Elderly (65+)", "🧓",
                db_risk.get("advisory_elderly", "")), unsafe_allow_html=True)
        with f3:
            st.markdown(flag_block(
                db_risk["flag_asthmatic"], "Asthmatic", "🫁",
                db_risk.get("advisory_asthmatic", "")), unsafe_allow_html=True)
    else:
        st.info("Health advisory data unavailable for this city.")

    sep()

    # ── Explainability ────────────────────────────────────────────────────
    section_hd("Factors Driving Today's Prediction",
               "Red bars increase predicted AQI · Blue bars reduce it · Ranked by influence")

    shap_df      = c_shap(city)
    sh_col, f_col = st.columns([6, 4], gap="large")

    with sh_col:
        fig = chart_shap(shap_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Explainability data not available.")

    with f_col:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if not shap_df.empty:
            for _, row in shap_df.head(4).iterrows():
                feat   = row["feature"]
                name   = FEATURE_NAMES.get(feat, feat)
                score  = row["mean_abs_shap"]
                is_bad = feat in WORSENS
                color  = "#DC2626" if is_bad else ("#2563EB" if feat in IMPROVES else "#6B7280")
                impact = ("Increases predicted pollution" if is_bad
                          else "Helps reduce predicted pollution" if feat in IMPROVES
                          else "Moderate influence on prediction")
                st.markdown(f"""
                <div class="shap-fc">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <p class="shap-fc-name">{name}</p>
                    <span style="font-size:12px;font-weight:600;color:{color}">{score:.3f}</span>
                  </div>
                  <p class="shap-fc-desc">{impact}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown(
            '<div class="shap-note"><p>'
            '<b>How to read this</b> · Bar length shows each factor\'s influence on '
            'the AI prediction. Calculated using SHAP (Shapley values), a method from '
            'cooperative game theory applied to machine learning models.'
            '</p></div>',
            unsafe_allow_html=True
        )

    sep()

    # ── Historical Trends ─────────────────────────────────────────────────
    section_hd("Historical Air Quality Trends",
               "Track how pollution has evolved over time for the selected city")

    range_opts = {
    "7 days": 7 * 24,
    "30 days": 30 * 24,
    "90 days": 90 * 24,
    "1 year": 365 * 24,
    }

    if "hist_range" not in st.session_state:
        st.session_state.hist_range = "7 days"

    c1, c2, c3, c4 = st.columns(4)

    for col, label in zip(
        [c1, c2, c3, c4],
        range_opts.keys()
    ):
        active = st.session_state.hist_range == label

        with col:
            if st.button(
                label,
                key=f"hist_btn_{label}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.hist_range = label
                st.rerun()

    chosen = st.session_state.hist_range

    hist_df = c_hist(city, range_opts[chosen])
    fig     = chart_historical(hist_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if not hist_df.empty:
        vals = hist_df["aqi"]
        s1, s2, s3, s4 = st.columns(4)
        for col, label, value in [
            (s1, "Average AQI",   f"{vals.mean():.1f}"),
            (s2, "Peak AQI",      str(int(vals.max()))),
            (s3, "Lowest AQI",    str(int(vals.min()))),
            (s4, "% Hazardous",   f"{(vals >= 4).mean()*100:.1f}%"),
        ]:
            with col:
                st.markdown(
                    f'<div class="stat-mini">'
                    f'<p class="stat-mini-v">{value}</p>'
                    f'<p class="stat-mini-l">{label}</p></div>',
                    unsafe_allow_html=True
                )

    sep()

    # ── Live Monitoring Table ─────────────────────────────────────────────
    section_hd("Live Monitoring Table",
               "Latest readings across all 15 monitored Indian cities")

    cities_df = c_cities()
    disp = (cities_df
        .rename(columns={"city": "City", "aqi_label": "Category", "aqi": "AQI Level",
                         "pm25": "PM2.5", "no2": "NO2",
                         "risk_score": "Risk Score", "risk_label": "Risk Level"})
        [["City","AQI Level","Category","PM2.5","NO2","Risk Score","Risk Level"]]
        .set_index("City")
    )
    st.dataframe(disp, use_container_width=True, height=430)

    csv = disp.reset_index().to_csv(index=False)
    st.download_button(
        "⬇ Export as CSV", csv,
        file_name=f"vayu_live_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )


# ── City Comparison ──────────────────────────────────────────────────────────

def render_comparison():
    section_hd("City Comparison",
               "Compare air quality and health risk between any two monitored cities")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        city1 = st.selectbox("City A", CITIES, index=CITIES.index("Delhi"), key="cmp1")
    with col2:
        city2 = st.selectbox("City B", CITIES, index=CITIES.index("Mumbai"), key="cmp2")

    if city1 == city2:
        st.warning("Please select two different cities to compare.")
        return

    comp_df = get_comparison_data(city1, city2)
    if comp_df.empty:
        st.info("Comparison data not available.")
        return

    sep_sm()

    cc1, cc2 = st.columns(2, gap="large")
    cols = {city1: cc1, city2: cc2}

    for _, row in comp_df.iterrows():
        with cols.get(row["city"], cc1):
            rc = risk_color(row["risk_score"])
            tc = "#111" if row["aqi"] in [1, 2] else "#fff"
            st.markdown(f"""
            <div class="cmp-card">
              <p class="cmp-city-name">{row['city']}</p>
              <div class="cmp-aqi-block" style="background:{row['aqi_color']}">
                <p class="cmp-aqi-n" style="color:{tc}">{row['aqi']}</p>
                <p class="cmp-aqi-l" style="color:{tc}">{row['aqi_label']}</p>
              </div>
              <div class="cmp-stat-grid">
                <div class="cmp-stat">
                  <p class="cmp-stat-l">PM2.5</p>
                  <p class="cmp-stat-v">{row['pm25']:.1f}</p>
                </div>
                <div class="cmp-stat">
                  <p class="cmp-stat-l">NO2</p>
                  <p class="cmp-stat-v">{row['no2']:.1f}</p>
                </div>
              </div>
              <div class="cmp-risk" style="background:{rc}">
                <p class="cmp-risk-v">{row['risk_score']:.0f}%</p>
                <p class="cmp-risk-l">{row['risk_label']}</p>
              </div>
              <p class="cmp-advisory">{row['general_advisory']}</p>
            </div>""", unsafe_allow_html=True)

    sep()

    section_hd(f"{city1} vs {city2} — Past 30 Days",
               "Historical AQI trend comparison")

    h1 = get_historical_aqi(city1, 30*24)
    h2 = get_historical_aqi(city2, 30*24)
    if not h1.empty and not h2.empty:
        fig = chart_comparison(h1, h2, city1, city2)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Historical Analytics ──────────────────────────────────────────────────────

def render_analytics():
    section_hd("Historical Analytics",
               "Deep-dive into seasonal patterns, hourly cycles, and model performance")

    city = st.selectbox("Select City", CITIES,
                        index=CITIES.index(st.session_state.city),
                        key="analytics_city")

    sep_sm()

    tab1, tab2, tab3 = st.tabs(["Hourly Heatmap", "Monthly Trends", "Model Performance"])

    with tab1:
        st.markdown(
            '<p style="font-size:13px;color:#9CA3AF;margin-bottom:16px">'
            'Average AQI by hour of day and day of week. '
            'Darker color = more severe pollution.</p>',
            unsafe_allow_html=True
        )
        fig = chart_heatmap(get_hourly_heatmap(city))
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Heatmap data not available.")

    with tab2:
        st.markdown(
            '<p style="font-size:13px;color:#9CA3AF;margin-bottom:16px">'
            'Monthly average AQI (bars) and PM2.5 concentration (line).</p>',
            unsafe_allow_html=True
        )
        fig = chart_monthly(get_monthly_trend(city))
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Monthly data not available.")

    with tab3:
        st.markdown(
            '<p style="font-size:13px;color:#9CA3AF;margin-bottom:16px">'
            'Prophet and XGBoost forecasting performance across all 15 cities.</p>',
            unsafe_allow_html=True
        )
        mdf = get_model_metrics()
        if not mdf.empty:
            st.dataframe(mdf.round(4).set_index("city"),
                         use_container_width=True, height=460)
        else:
            st.info("Model metrics not available.")


# ── Footer ───────────────────────────────────────────────────────────────────

def render_footer():
    st.markdown(f"""
    <div class="vayu-footer">
      <span>VAYU · AI-Powered Air Quality Intelligence · India</span>
      <span>OpenWeather · Open-Meteo · {datetime.now().year}</span>
    </div>""", unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    render_header()
    st.markdown('<div class="sep-sm"></div>', unsafe_allow_html=True)

    render_nav()
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    page = st.session_state.get("page", "Overview")
    if page == "Overview":
        render_overview()
    elif page == "City Comparison":
        render_comparison()
    else:
        render_analytics()

    render_footer()


if __name__ == "__main__":
    main()