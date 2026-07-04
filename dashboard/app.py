# dashboard/app.py
import streamlit as st

st.set_page_config(
    page_title="VAYU — Air Quality Intelligence",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone

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

CITIES = sorted(CITY_COORDS.keys())

FEATURE_NAMES = {
    "pm25": "PM2.5 Concentration",
    "pm10": "PM10 Concentration",
    "no2": "Nitrogen Dioxide",
    "o3": "Ozone Level",
    "aqi": "Current AQI",
    "aqi_roll_6h": "6-Hour AQI Avg",
    "aqi_roll_24h": "24-Hour AQI Avg",
    "aqi_lag_1h": "AQI — 1 Hour Ago",
    "aqi_lag_3h": "AQI — 3 Hours Ago",
    "aqi_lag_24h": "AQI — 24 Hours Ago",
    "temperature_c": "Temperature",
    "humidity_pct": "Humidity",
    "wind_speed_ms": "Wind Speed",
    "wind_deg": "Wind Direction",
    "rainfall_1h_mm": "Rainfall",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
    "is_weekend": "Weekend",
}

# Features that worsen AQI (red) vs improve it (blue)
WORSENS = {"pm25","pm10","no2","o3","aqi","aqi_lag_1h","aqi_lag_3h",
           "aqi_lag_24h","aqi_roll_6h","aqi_roll_24h","humidity_pct","month"}
IMPROVES = {"wind_speed_ms","rainfall_1h_mm"}

MONTH_NAMES = {
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
}
DAY_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


# ── Session state ──────────────────────────────────────────────────────────

if "city" not in st.session_state:
    st.session_state.city = "Delhi"


# ── Helpers ────────────────────────────────────────────────────────────────

def aqi_badge_html(label, color):
    text_color = "#111" if color in ("#00E400","#FFFF00") else "#fff"
    return (
        f'<span class="metric-badge" '
        f'style="background:{color};color:{text_color}">'
        f'{label}</span>'
    )

def risk_color(score):
    if score is None: return "#9CA3AF"
    if score <= 20:   return "#10B981"
    if score <= 40:   return "#F59E0B"
    if score <= 60:   return "#F97316"
    if score <= 75:   return "#DC2626"
    return "#7C3AED"


# ── Components ─────────────────────────────────────────────────────────────

def metric_card(label, icon, value, delta_html, badge_html, accent=None):
    border = f"border-top: 3px solid {accent};" if accent else ""
    return f"""
    <div class="metric-card" style="{border}">
      <div class="metric-top">
        <span class="metric-label">{label}</span>
        <span class="metric-icon">{icon}</span>
      </div>
      <div class="metric-value-sm">{value}</div>
      <div class="metric-bottom">
        {delta_html}
        {badge_html}
      </div>
    </div>"""


def build_india_map(cities_df, selected_city):
    m = folium.Map(
        location=[22.5, 82.5],
        zoom_start=5,
        tiles="CartoDB Positron",
        zoom_control=False,
        scrollWheelZoom=True,
        attributionControl=False,
    )

    for _, row in cities_df.iterrows():
        city  = row["city"]
        lat, lon = CITY_COORDS.get(city, (22, 82))
        aqi_val = int(row["aqi"]) if pd.notna(row["aqi"]) else 2
        color   = AQI_COLORS.get(aqi_val, "#888")
        selected = (city == selected_city)

        folium.CircleMarker(
            location=[lat, lon],
            radius=20 if selected else 14,
            color="white",
            weight=3 if selected else 1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            tooltip=folium.Tooltip(
                f"<b style='font-family:Inter'>{city}</b><br>"
                f"AQI: <b>{row['aqi_label']}</b><br>"
                f"Risk: <b>{row['risk_score']:.0f}/100</b>",
                sticky=True
            ),
        ).add_to(m)

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-family:Inter;font-size:10px;font-weight:600;'
                     f'color:{"#111" if aqi_val <= 2 else "#fff"};'
                     f'text-align:center;margin-top:5px;'
                     f'width:40px;margin-left:-20px">{city[:3]}</div>',
                icon_size=(40, 20),
                icon_anchor=(20, -5),
            )
        ).add_to(m)

    return m


def forecast_chart(df, city, model_name):
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()

    if "yhat_lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=list(df["timestamp"]) + list(df["timestamp"][::-1]),
            y=list(df["yhat_upper"]) + list(df["yhat_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(37,99,235,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Range",
            hoverinfo="skip",
        ))
        y_col = "yhat"
    else:
        y_col = "predicted_aqi"

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[y_col],
        mode="lines",
        line=dict(color="#2563EB", width=2.5),
        name="Forecast",
        hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI Level: %{y:.1f}<extra></extra>",
    ))

    fig.add_vline(
        x=datetime.now(timezone.utc),
        line_dash="dash", line_color="#9CA3AF", line_width=1.5
    )

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=8, r=8, t=8, b=8),
        height=280,
        xaxis=dict(showgrid=False, color="#9CA3AF", title=None, tickfont=dict(size=11)),
        yaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            title=None, range=[0.5, 5.5],
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["Good","Fair","Moderate","Poor","Hazardous"],
            color="#9CA3AF", tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h", y=-0.18, x=0,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)"
        ),
        font=dict(family="Inter, sans-serif"),
        showlegend=True,
    )
    return fig


def shap_chart(shap_df):
    if shap_df.empty:
        return None

    features    = shap_df["feature"].tolist()
    values      = shap_df["mean_abs_shap"].tolist()
    disp_names  = [FEATURE_NAMES.get(f, f) for f in features]
    colors      = ["#DC2626" if f in WORSENS else
                   "#2563EB" if f in IMPROVES else
                   "#9CA3AF" for f in features]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=disp_names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:.3f}" for v in values[::-1]],
        textposition="outside",
        textfont=dict(size=11, color="#6B7280"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=8, r=40, t=8, b=8),
        height=max(280, len(features) * 30),
        xaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                   showticklabels=False, title=None, zeroline=False),
        yaxis=dict(showgrid=False, color="#374151",
                   tickfont=dict(size=12), title=None),
        font=dict(family="Inter, sans-serif"),
        bargap=0.35,
    )
    return fig


def historical_chart(df, city):
    if df.empty:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        mode="lines",
        line=dict(color="#2563EB", width=1.8),
        name="AQI Level",
        hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI: %{y}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.05)",
    ))

    # Highlight max
    max_idx = df["aqi"].idxmax()
    fig.add_trace(go.Scatter(
        x=[df.loc[max_idx, "timestamp"]],
        y=[df.loc[max_idx, "aqi"]],
        mode="markers",
        marker=dict(color="#DC2626", size=8, symbol="circle"),
        name=f"Peak: {df.loc[max_idx,'aqi']}",
        hovertemplate="<b>Peak AQI: %{y}</b><br>%{x|%d %b}<extra></extra>",
    ))

    fig.add_hline(
        y=df["aqi"].mean(), line_dash="dot",
        line_color="#9CA3AF", line_width=1,
        annotation_text=f"Avg: {df['aqi'].mean():.1f}",
        annotation_font=dict(size=11, color="#9CA3AF"),
    )

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=8, r=8, t=8, b=8),
        height=260,
        xaxis=dict(showgrid=False, color="#9CA3AF",
                   title=None, tickfont=dict(size=11)),
        yaxis=dict(
            showgrid=True, gridcolor="#F3F4F6",
            title=None, range=[0.5, 5.5],
            tickvals=[1,2,3,4,5],
            ticktext=["Good","Fair","Moderate","Poor","Hazardous"],
            color="#9CA3AF", tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h", y=-0.18,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)"
        ),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── Refresh pipeline ───────────────────────────────────────────────────────

def run_pipeline():
    root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts = [
        ("Collecting live data...",  "src/collect.py"),
        ("Cleaning database...",     "src/clean.py"),
        ("Engineering features...",  "src/features.py"),
        ("Updating forecasts...",    "src/prophet_model.py"),
        ("Recalculating health risk...", "src/health_risk.py"),
    ]
    bar    = st.progress(0)
    status = st.empty()
    for i, (msg, script) in enumerate(scripts):
        status.caption(f"⏳  {msg}")
        res = subprocess.run(
            [sys.executable, os.path.join(root, script)],
            capture_output=True, text=True, timeout=600
        )
        bar.progress((i + 1) / len(scripts))
        if res.returncode != 0:
            st.error(f"Pipeline failed at `{script}`:\n```\n{res.stderr[-500:]}\n```")
            return False
    status.empty()
    bar.empty()
    st.cache_data.clear()
    return True


# ── Page: Overview ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def cached_cities_latest():
    return get_all_cities_latest()

@st.cache_data(ttl=3600)
def cached_live_aqi(city):
    return get_live_aqi(city)

@st.cache_data(ttl=3600)
def cached_forecast(city):
    model, mae = get_best_model(city)
    if model == "prophet":
        return get_prophet_forecast(city), "Prophet", mae
    else:
        df = get_xgb_forecast(city)
        df = df.rename(columns={"predicted_aqi": "yhat"})
        return df, "XGBoost", mae

@st.cache_data(ttl=3600)
def cached_shap(city):
    return get_shap_data(city)

@st.cache_data(ttl=3600)
def cached_historical(city, hours):
    return get_historical_aqi(city, hours)


def render_overview():
    city = st.session_state.city

    # ── Header ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        st.markdown(
            '<p class="vayu-logo">VA<span>YU</span></p>'
            '<p class="vayu-tagline">Real-Time Air Quality Intelligence · India</p>',
            unsafe_allow_html=True
        )
    with c2:
        selected = st.selectbox(
            "Select City", CITIES,
            index=CITIES.index(city),
            key="city_selector",
            label_visibility="collapsed"
        )
        if selected != city:
            st.session_state.city = selected
            st.rerun()
    with c3:
        ts = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")
        st.markdown(
            f'<div style="text-align:right;padding-top:8px">'
            f'<span class="nav-timestamp">🕐 {ts}</span></div>',
            unsafe_allow_html=True
        )
        if st.button("↻ Refresh Data", use_container_width=True):
            with st.spinner("Running pipeline..."):
                if run_pipeline():
                    st.success("Data updated.")
                    st.rerun()

    st.markdown('<div class="vayu-divider"></div>', unsafe_allow_html=True)

    # ── Live data ──────────────────────────────────────────────────────────
    live      = cached_live_aqi(city)
    db_risk   = get_latest_health_risk(city)
    yesterday = get_yesterday_aqi(city)

    aqi_val   = live["aqi"]      if live else (db_risk["aqi"]      if db_risk is not None else None)
    aqi_label = live["aqi_label"]if live else (db_risk["aqi_label"]if db_risk is not None else "Unknown")
    aqi_color = live["aqi_color"]if live else (db_risk["aqi_color"]if db_risk is not None else "#888")
    pm25      = live["pm25"]     if live else (db_risk["pm25"]     if db_risk is not None else None)
    risk_sc   = int(db_risk["risk_score"]) if db_risk is not None else None

    # ── Alert banner ───────────────────────────────────────────────────────
    if aqi_val and aqi_val >= 4:
        st.markdown(f"""
        <div class="alert-banner">
          <span style="font-size:24px">⚠️</span>
          <div>
            <p class="alert-title">Severe Air Pollution Alert — {city}</p>
            <p class="alert-body">Current AQI is classified as <b>{aqi_label}</b>.
            Sensitive groups should remain indoors. Avoid all outdoor activity.
            Keep windows closed and use air purification if available.</p>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Hero cards ─────────────────────────────────────────────────────────
    forecast_df, model_name, model_mae = cached_forecast(city)

    # Delta vs yesterday
    if yesterday and aqi_val:
        diff = aqi_val - yesterday
        if diff > 0:
            delta_html = f'<span class="metric-delta-up">↑ {abs(diff):.1f} vs yesterday</span>'
        elif diff < 0:
            delta_html = f'<span class="metric-delta-down">↓ {abs(diff):.1f} vs yesterday</span>'
        else:
            delta_html = '<span class="metric-delta-flat">→ Same as yesterday</span>'
    else:
        delta_html = '<span class="metric-delta-flat">Live reading</span>'

    # Forecast trend
    if not forecast_df.empty:
        y_col        = "yhat" if "yhat" in forecast_df.columns else "predicted_aqi"
        forecast_max = forecast_df[y_col].max()
        forecast_min = forecast_df[y_col].min()
        if aqi_val and forecast_max > aqi_val + 0.5:
            trend_val  = f"Worsening"
            trend_icon = "📈"
            trend_delta = f'<span class="metric-delta-up">↑ Expected to rise</span>'
        elif aqi_val and forecast_min < aqi_val - 0.5:
            trend_val  = "Improving"
            trend_icon = "📉"
            trend_delta = f'<span class="metric-delta-down">↓ Expected to improve</span>'
        else:
            trend_val  = "Stable"
            trend_icon = "→"
            trend_delta = '<span class="metric-delta-flat">→ No major change</span>'
    else:
        trend_val, trend_icon, trend_delta = "N/A", "—", ""

    risk_badge = ""
    if risk_sc is not None:
        rc = risk_color(risk_sc)
        risk_badge = f'<span class="metric-badge" style="background:{rc};color:white">{db_risk["risk_label"]}</span>'

    cards = [
        metric_card(
            "Current AQI", "💨",
            str(aqi_val) if aqi_val else "—",
            delta_html,
            aqi_badge_html(aqi_label, aqi_color),
            accent=aqi_color
        ),
        metric_card(
            "Estimated Health Risk", "🫀",
            f"{risk_sc}%" if risk_sc is not None else "—",
            '<span class="metric-delta-flat">WHO-aligned index</span>',
            risk_badge,
            accent=risk_color(risk_sc)
        ),
        metric_card(
            "48-Hour Forecast Trend", trend_icon,
            trend_val,
            trend_delta,
            f'<span class="metric-badge" style="background:#EFF6FF;color:#2563EB">{model_name}</span>',
        ),
        metric_card(
            "PM2.5 Concentration", "🌫️",
            f"{pm25:.1f}" if pm25 else "—",
            '<span class="metric-delta-flat">µg/m³</span>',
            f'<span class="metric-badge" style="background:#F3F4F6;color:#374151">'
            f'{"Above WHO limit" if pm25 and pm25 > 15 else "Within range"}</span>',
        ),
    ]

    c1, c2, c3, c4 = st.columns(4)
    for col, card_html in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Map + Forecast ─────────────────────────────────────────────────────
    map_col, chart_col = st.columns([6, 4], gap="large")

    cities_df = cached_cities_latest()

    with map_col:
        st.markdown(
            '<p class="section-title">Air Quality Map — India</p>'
            '<p class="section-caption">All 15 monitored cities. Click a city to explore.</p>',
            unsafe_allow_html=True
        )
        with st.container():
            india_map   = build_india_map(cities_df, city)
            map_result  = st_folium(
                india_map,
                height=400,
                use_container_width=True,
                returned_objects=["last_object_clicked"],
            )

            # Handle map click → update selected city
            if map_result and map_result.get("last_object_clicked"):
                clat = map_result["last_object_clicked"].get("lat")
                clng = map_result["last_object_clicked"].get("lng")
                if clat and clng:
                    nearest, best_d = city, float("inf")
                    for c_name, (lat, lon) in CITY_COORDS.items():
                        d = ((clat - lat)**2 + (clng - lon)**2)**0.5
                        if d < best_d:
                            best_d, nearest = d, c_name
                    if nearest != st.session_state.city:
                        st.session_state.city = nearest
                        st.rerun()

    with chart_col:
        st.markdown(
            '<p class="section-title">48-Hour Air Quality Forecast</p>'
            f'<p class="section-caption">Model: {model_name}'
            + (f' · MAE {model_mae:.3f}' if model_mae else '') + '</p>',
            unsafe_allow_html=True
        )
        fig = forecast_chart(forecast_df, city, model_name)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Forecast not available for this city.")

    st.markdown('<div class="vayu-divider"></div>', unsafe_allow_html=True)

    # ── Health Recommendations ─────────────────────────────────────────────
    st.markdown(
        '<p class="section-title">Today\'s Health Advisory</p>'
        '<p class="section-caption">Personalised guidance based on current air quality.</p>',
        unsafe_allow_html=True
    )

    if db_risk is not None:
        st.markdown(
            f'<div class="advisory-main">{db_risk["general_advisory"]}</div>',
            unsafe_allow_html=True
        )

        rec_c1, rec_c2, rec_c3 = st.columns(3)

        def demo_flag(flag, label, icon, advice):
            if flag:
                return (
                    f'<div class="demo-flag flag-alert">'
                    f'{icon} <div><b>{label}</b><br>'
                    f'<span style="font-size:12px">{advice}</span></div></div>'
                )
            return (
                f'<div class="demo-flag flag-safe">'
                f'✓ <b>{label}</b> — Low risk</div>'
            )

        with rec_c1:
            st.markdown(
                demo_flag(
                    db_risk["flag_children"], "Children (under 14)", "👶",
                    db_risk.get("advisory_children","")
                ),
                unsafe_allow_html=True
            )
        with rec_c2:
            st.markdown(
                demo_flag(
                    db_risk["flag_elderly"], "Elderly (65+)", "🧓",
                    db_risk.get("advisory_elderly","")
                ),
                unsafe_allow_html=True
            )
        with rec_c3:
            st.markdown(
                demo_flag(
                    db_risk["flag_asthmatic"], "Asthmatic", "🫁",
                    db_risk.get("advisory_asthmatic","")
                ),
                unsafe_allow_html=True
            )
    else:
        st.info("Health advisory not available.")

    st.markdown('<div class="vayu-divider"></div>', unsafe_allow_html=True)

    # ── AI Explainability ──────────────────────────────────────────────────
    st.markdown(
        '<p class="section-title">Why the AI Predicted This</p>'
        '<p class="section-caption">Factors driving today\'s AQI category prediction. '
        'Red bars increase predicted pollution. Blue bars reduce it.</p>',
        unsafe_allow_html=True
    )

    shap_df = cached_shap(city)
    exp_c1, exp_c2 = st.columns([6, 4], gap="large")

    with exp_c1:
        fig_shap = shap_chart(shap_df)
        if fig_shap:
            st.plotly_chart(fig_shap, use_container_width=True,
                           config={"displayModeBar": False})

    with exp_c2:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if not shap_df.empty:
            top3 = shap_df.head(3)
            for _, row in top3.iterrows():
                feat   = row["feature"]
                name   = FEATURE_NAMES.get(feat, feat)
                score  = row["mean_abs_shap"]
                is_bad = feat in WORSENS
                color  = "#DC2626" if is_bad else "#2563EB" if feat in IMPROVES else "#9CA3AF"
                impact = "Major contributor to AQI" if is_bad else "Helps reduce AQI"
                st.markdown(f"""
                <div class="vayu-card" style="margin-bottom:12px;padding:16px 20px">
                  <div style="display:flex;align-items:center;justify-content:space-between">
                    <b style="font-size:14px;color:#111827">{name}</b>
                    <span style="font-size:12px;font-weight:600;color:{color}">{score:.3f}</span>
                  </div>
                  <p style="font-size:12px;color:#6B7280;margin:4px 0 0">{impact}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#F7F9FC;border-radius:12px;padding:16px;margin-top:4px">
          <p style="font-size:12px;color:#6B7280;margin:0;line-height:1.6">
          <b>How to read this:</b> Bar length shows how strongly each factor
          influenced the prediction. This is calculated using SHAP (Shapley values),
          a method from cooperative game theory applied to machine learning.
          </p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="vayu-divider"></div>', unsafe_allow_html=True)

    # ── Historical Trends ──────────────────────────────────────────────────
    st.markdown(
        '<p class="section-title">Historical Air Quality Trends</p>'
        '<p class="section-caption">Track how pollution has changed over time.</p>',
        unsafe_allow_html=True
    )

    day_options = {"7 days": 7*24, "30 days": 30*24, "90 days": 90*24, "1 year": 365*24}
    hist_tab    = st.radio(
        "Range", list(day_options.keys()),
        horizontal=True, label_visibility="collapsed", key="hist_range"
    )
    hist_df = cached_historical(city, day_options[hist_tab])
    fig_hist = historical_chart(hist_df, city)
    if fig_hist:
        st.plotly_chart(fig_hist, use_container_width=True,
                       config={"displayModeBar": False})

    # Quick stats
    if not hist_df.empty:
        s1, s2, s3, s4 = st.columns(4)
        vals = hist_df["aqi"]
        for col, label, val in [
            (s1, "Average AQI", f"{vals.mean():.1f}"),
            (s2, "Peak AQI",    f"{vals.max():.0f}"),
            (s3, "Lowest AQI",  f"{vals.min():.0f}"),
            (s4, "% Hazardous", f"{(vals >= 4).mean()*100:.1f}%"),
        ]:
            with col:
                st.markdown(f"""
                <div class="vayu-card" style="text-align:center;padding:16px">
                  <p style="font-size:22px;font-weight:700;color:#111827;margin:0">{val}</p>
                  <p style="font-size:12px;color:#9CA3AF;margin:4px 0 0">{label}</p>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="vayu-divider"></div>', unsafe_allow_html=True)

    # ── Live Data Table ────────────────────────────────────────────────────
    st.markdown(
        '<p class="section-title">Live Monitoring Table</p>'
        '<p class="section-caption">All 15 cities — latest readings.</p>',
        unsafe_allow_html=True
    )

    display_df = cities_df.copy()
    display_df["aqi_num"] = display_df["aqi"]
    display_df = display_df.rename(columns={
        "city":       "City",
        "aqi_label":  "AQI Category",
        "aqi_num":    "AQI Level",
        "pm25":       "PM2.5",
        "no2":        "NO2",
        "risk_score": "Risk Score",
        "risk_label": "Risk Level",
    })

    cols = ["City","AQI Level","AQI Category","PM2.5","NO2","Risk Score","Risk Level"]
    st.dataframe(
        display_df[cols].set_index("City"),
        use_container_width=True,
        height=450,
    )

    csv = display_df[cols].to_csv(index=False)
    st.download_button(
        "⬇ Export CSV", csv,
        file_name=f"vayu_aqi_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


# ── Page: City Comparison ──────────────────────────────────────────────────

def render_comparison():
    st.markdown(
        '<p class="section-title">City Comparison</p>'
        '<p class="section-caption">Compare air quality and health risk between two cities.</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        city1 = st.selectbox("City A", CITIES, index=CITIES.index("Delhi"), key="comp_city1")
    with col2:
        city2 = st.selectbox("City B", CITIES, index=CITIES.index("Mumbai"), key="comp_city2")

    if city1 == city2:
        st.warning("Please select two different cities.")
        return

    comp_df = get_comparison_data(city1, city2)
    if comp_df.empty:
        st.info("No data available.")
        return

    for _, row in comp_df.iterrows():
        with (col1 if row["city"] == city1 else col2):
            rc = risk_color(row["risk_score"])
            tc = "#111" if row["aqi"] in [1, 2] else "#fff"
            st.markdown(f"""
            <div class="vayu-card" style="text-align:center;margin-top:16px">
              <h3 style="margin:0 0 16px;font-size:20px">{row['city']}</h3>
              <div style="background:{row['aqi_color']};border-radius:12px;
                          padding:16px;margin-bottom:16px">
                <p style="font-size:42px;font-weight:700;margin:0;
                           color:{tc}">{row['aqi']}</p>
                <p style="font-size:14px;font-weight:600;margin:4px 0 0;
                           color:{tc}">{row['aqi_label']}</p>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;
                           margin-bottom:16px">
                <div style="background:#F7F9FC;border-radius:10px;padding:12px">
                  <p style="font-size:11px;color:#9CA3AF;margin:0">PM2.5</p>
                  <p style="font-size:18px;font-weight:600;margin:2px 0 0;color:#111">
                    {row['pm25']:.1f}</p>
                </div>
                <div style="background:#F7F9FC;border-radius:10px;padding:12px">
                  <p style="font-size:11px;color:#9CA3AF;margin:0">NO2</p>
                  <p style="font-size:18px;font-weight:600;margin:2px 0 0;color:#111">
                    {row['no2']:.1f}</p>
                </div>
              </div>
              <div style="background:{rc};border-radius:10px;padding:12px;margin-bottom:16px">
                <p style="font-size:11px;color:rgba(255,255,255,0.8);margin:0">Health Risk</p>
                <p style="font-size:24px;font-weight:700;margin:2px 0 0;color:white">
                  {row['risk_score']:.0f}%</p>
                <p style="font-size:12px;color:rgba(255,255,255,0.9);margin:2px 0 0">
                  {row['risk_label']}</p>
              </div>
              <p style="font-size:13px;color:#374151;text-align:left;
                         background:#F7F9FC;border-radius:10px;padding:12px;
                         line-height:1.6;margin:0">
                {row['general_advisory']}</p>
            </div>""", unsafe_allow_html=True)

    # Historical comparison chart
    st.markdown('<div class="vayu-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Historical AQI Comparison — 30 Days</p>',
                unsafe_allow_html=True)

    hist1 = get_historical_aqi(city1, 30*24)
    hist2 = get_historical_aqi(city2, 30*24)

    fig = go.Figure()
    for df, name, color in [(hist1, city1, "#2563EB"), (hist2, city2, "#DC2626")]:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["aqi"],
            mode="lines", line=dict(color=color, width=2),
            name=name,
            hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %H:%M}}<br>AQI: %{{y}}<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        height=280, margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(showgrid=False, color="#9CA3AF", title=None),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", title=None,
                   range=[0.5,5.5], tickvals=[1,2,3,4,5],
                   ticktext=["Good","Fair","Moderate","Poor","Hazardous"],
                   color="#9CA3AF"),
        legend=dict(orientation="h", y=-0.2, bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Page: Historical Analytics ─────────────────────────────────────────────

def render_analytics():
    st.markdown(
        '<p class="section-title">Historical Analytics</p>'
        '<p class="section-caption">Deep patterns in air quality across time and cities.</p>',
        unsafe_allow_html=True
    )

    city = st.selectbox("Select City", CITIES,
                        index=CITIES.index(st.session_state.city),
                        key="analytics_city")

    tab1, tab2, tab3 = st.tabs(["Hourly Heatmap", "Monthly Trends", "Model Performance"])

    with tab1:
        st.markdown('<p class="section-caption" style="margin-top:16px">'
                    'Average AQI by hour and day. Darker = worse pollution.</p>',
                    unsafe_allow_html=True)
        hm = get_hourly_heatmap(city)
        if not hm.empty:
            pivot = hm.pivot(index="day_of_week", columns="hour", values="avg_aqi").fillna(2)
            fig   = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"{h:02d}:00" for h in pivot.columns],
                y=[DAY_NAMES[d] for d in pivot.index],
                colorscale=[[0,"#00E400"],[0.25,"#FFFF00"],
                            [0.5,"#FF7E00"],[0.75,"#FF0000"],[1,"#8F3F97"]],
                zmin=1, zmax=5,
                hovertemplate="Hour: %{x}<br>Day: %{y}<br>Avg AQI: %{z:.2f}<extra></extra>",
                colorbar=dict(
                    tickvals=[1,2,3,4,5],
                    ticktext=["Good","Fair","Moderate","Poor","Hazardous"],
                    thickness=12, len=0.8,
                )
            ))
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                height=280, margin=dict(l=8,r=8,t=8,b=8),
                xaxis=dict(side="bottom", tickfont=dict(size=10)),
                yaxis=dict(tickfont=dict(size=11)),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with tab2:
        mt = get_monthly_trend(city)
        if not mt.empty:
            mt["month_name"] = mt["month"].map(MONTH_NAMES)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=mt["month_name"], y=mt["avg_aqi"],
                name="Average AQI",
                marker_color="#2563EB",
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=mt["month_name"], y=mt["avg_pm25"],
                name="Avg PM2.5",
                mode="lines+markers",
                line=dict(color="#DC2626", width=2),
                yaxis="y2",
            ))
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                height=280, margin=dict(l=8,r=8,t=8,b=8),
                xaxis=dict(showgrid=False, color="#9CA3AF"),
                yaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                           title="AQI Level", color="#9CA3AF"),
                yaxis2=dict(title="PM2.5 µg/m³", overlaying="y",
                            side="right", color="#DC2626",
                            showgrid=False),
                legend=dict(orientation="h", y=-0.2, bgcolor="rgba(0,0,0,0)"),
                font=dict(family="Inter, sans-serif"),
                bargap=0.35,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with tab3:
        metrics_df = get_model_metrics()
        if not metrics_df.empty:
            st.markdown('<p class="section-caption" style="margin-top:16px">'
                        'Prophet and XGBoost performance across all cities.</p>',
                        unsafe_allow_html=True)
            display = metrics_df.round(4).set_index("city")
            st.dataframe(display, use_container_width=True, height=450)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    tab1, tab2, tab3 = st.tabs([
        "🏠  Overview",
        "🔄  City Comparison",
        "📊  Historical Analytics",
    ])
    with tab1:
        render_overview()
    with tab2:
        render_comparison()
    with tab3:
        render_analytics()

    st.markdown(
        '<div class="vayu-footer">VAYU · AI-Powered Air Quality Intelligence · India · '
        'Built with OpenWeather & Open-Meteo · '
        f'{datetime.now().year}</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()