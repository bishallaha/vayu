# dashboard/styles.py

VAYU_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #F8FAFC !important;
    color: #111827 !important;
}

/* ── Hide Streamlit chrome ──────────────────────── */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
button[title="View fullscreen"] {
    display: none !important;
    visibility: hidden !important;
}

/* Remove Streamlit top header completely */
header {
    display: none !important;
}

[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

.stApp {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ── Main layout ────────────────────────────────── */
.block-container {
    padding-top: 40px !important;
    padding-left: 52px !important;
    padding-right: 52px !important;
    padding-bottom: 80px !important;
    max-width: 1340px !important;
}

/* ── Brand / Header ─────────────────────────────── */
.vayu-brand h1{
    font-family:"Brigends Expanded","Inter",sans-serif !important;
    font-size:54px;
    font-weight:400;
    letter-spacing:0;
    line-height:0.95;
    margin-bottom:-2px;
}
.vayu-brand h1 span { color: #2563EB; }
.vayu-brand-sub {
    font-size: 13px;
    color: #9CA3AF;
    margin-top:0px;
    margin-bottom:10px;
    font-weight: 400;
}
.vayu-github {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    font-weight: 500;
    color: #4B5563;
    text-decoration: none;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 5px 12px;
    transition: border-color 0.15s, color 0.15s;
    white-space: nowrap;
}
.vayu-github:hover { border-color: #2563EB; color: #2563EB; }

/* ── Timestamp pill ─────────────────────────────── */
.ts-pill {
    background: #F3F4F6;
    border-radius: 100px;
    padding: 5px 13px;
    font-size: 12px;
    color: #6B7280;
    font-weight: 400;
    white-space: nowrap;
    display: inline-block;
}

/* ── Separators ─────────────────────────────────── */
.sep    { height: 1px; background: #E5E7EB; margin: 36px 0; }
.sep-sm { height: 1px; background: #F3F4F6; margin: 20px 0; }

/* ── Alert banner ───────────────────────────────── */
.alert-banner {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid #DC2626;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 24px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.alert-icon  { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
.alert-title { font-size: 14px; font-weight: 600; color: #991B1B; margin: 0 0 2px; }
.alert-body  { font-size: 13px; color: #B91C1C; margin: 0; line-height: 1.55; }

/* ── Section header ─────────────────────────────── */
.shd      { margin: 0 0 18px; }
.shd h2   { font-size: 20px; font-weight: 700; color: #111827; margin: 0 0 2px; letter-spacing: -0.02em; }
.shd p    { font-size: 13px; color: #9CA3AF; margin: 0; font-weight: 400; }

/* ── Metric cards ───────────────────────────────── */
.mc {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 22px 20px 18px;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-height: 148px;
}
.mc:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.07); transform: translateY(-1px); }
.mc-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 14px 14px 0 0; }
.mc-lbl { font-size: 11px; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.08em; display: flex; align-items: center; gap: 5px; }
.mc-val { font-size: 42px; font-weight: 700; color: #111827; letter-spacing: -0.035em; line-height: 1; }
.mc-val-sm { font-size: 26px; font-weight: 700; color: #111827; letter-spacing: -0.02em; line-height: 1; }
.mc-btm { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 4px; margin-top: 2px; }
.mc-up   { font-size: 12px; font-weight: 500; color: #DC2626; }
.mc-down { font-size: 12px; font-weight: 500; color: #059669; }
.mc-flat { font-size: 12px; color: #9CA3AF; }
.mc-badge { font-size: 11px; font-weight: 600; padding: 2px 9px; border-radius: 100px; letter-spacing: 0.02em; }

/* ── Card container ─────────────────────────────── */
.vayu-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 22px;
}

/* ── Selectbox ──────────────────────────────────── */
.stSelectbox label {
    font-size:11px !important;
    font-weight:600 !important;
    color:#9CA3AF !important;
    text-transform:uppercase !important;
    letter-spacing:.08em !important;
    margin-bottom:5px !important;
}
.stSelectbox > div > div {
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    background: white !important;
    min-height:42px !important;
    padding:0 !important;
    font-size: 14px !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
}

/* ── Advisory ───────────────────────────────────── */
.advisory-main {
    font-size: 14px;
    color: #374151;
    line-height: 1.75;
    background: #EFF6FF;
    border-left: 3px solid #2563EB;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 16px;
}
.flag-card  { border-radius: 10px; padding: 13px 14px; }
.flag-alert { background: #FFF1F2; border: 1px solid #FECDD3; }
.flag-alert-t { font-weight: 600; color: #BE123C; font-size: 13px; margin: 0 0 4px; display: flex; align-items: center; gap: 5px; }
.flag-alert-b { color: #BE123C; font-size: 12px; margin: 0; line-height: 1.55; }
.flag-ok  { background: #F0FDF4; border: 1px solid #BBF7D0; }
.flag-ok-t { font-weight: 600; color: #166534; font-size: 13px; margin: 0; display: flex; align-items: center; gap: 5px; }

/* ── SHAP ───────────────────────────────────────── */
.shap-fc {
    background: #FAFAFA;
    border: 1px solid #F3F4F6;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.shap-fc-name { font-size: 13px; font-weight: 600; color: #111827; margin: 0 0 2px; }
.shap-fc-desc { font-size: 11px; color: #6B7280; margin: 0; }
.shap-note {
    background: #F8FAFC;
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 6px;
}
.shap-note p { font-size: 12px; color: #6B7280; line-height: 1.65; margin: 0; }

/* ── Mini stat cards ────────────────────────────── */
.stat-mini { background: #FAFAFA; border: 1px solid #E5E7EB; border-radius: 12px; padding: 16px; text-align: center; }
.stat-mini-v { font-size: 22px; font-weight: 700; color: #111827; letter-spacing: -0.02em; margin: 0 0 3px; line-height: 1; }
.stat-mini-l { font-size: 11px; color: #9CA3AF; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; margin: 0; }

/* ── Buttons ────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    border: 1px solid #E5E7EB !important;
    background: white !important;
    color: #374151 !important;
    padding: 7px 18px !important;
    transition: all 0.15s !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #F9FAFB !important;
    border-color: #9CA3AF !important;
    color: #111827 !important;
}

/* ── Radio (range picker) ───────────────────────── */
div[role="radiogroup"] {
    display: flex;
    gap: 4px;
    background: #F3F4F6;
    border-radius: 10px;
    padding: 4px;
    width: fit-content;
}
div[role="radiogroup"] > label {
    padding: 6px 16px !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    cursor: pointer;
    transition: all 0.12s;
    user-select: none;
}
div[role="radiogroup"] > label:hover { color: #374151 !important; }
div[role="radiogroup"] > label[data-checked="true"] {
    background: white !important;
    color: #111827 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #F3F4F6 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 7px 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #6B7280 !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #111827 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
.stTabs [data-baseweb="tab-border"]  { display: none !important; }
.stTabs [data-baseweb="tab-panel"]   { padding-top: 20px !important; }

/* ── Data table ─────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* ── Download button ────────────────────────────── */
.stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: 1px solid #E5E7EB !important;
    background: white !important;
    color: #374151 !important;
    padding: 5px 14px !important;
    margin-top: 10px;
}

/* ── Comparison city cards ──────────────────────── */
.cmp-card { background: white; border: 1px solid #E5E7EB; border-radius: 16px; padding: 24px; }
.cmp-city-name { font-size: 18px; font-weight: 700; color: #111827; margin: 0 0 16px; text-align: center; }
.cmp-aqi-block { border-radius: 12px; padding: 20px; margin-bottom: 16px; text-align: center; }
.cmp-aqi-n { font-size: 48px; font-weight: 700; line-height: 1; margin: 0 0 2px; }
.cmp-aqi-l { font-size: 14px; font-weight: 600; margin: 0; }
.cmp-stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
.cmp-stat { background: #F8FAFC; border-radius: 10px; padding: 11px; text-align: center; }
.cmp-stat-l { font-size: 10px; color: #9CA3AF; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 2px; }
.cmp-stat-v { font-size: 18px; font-weight: 700; color: #111827; margin: 0; }
.cmp-risk { border-radius: 10px; padding: 13px; text-align: center; margin-bottom: 14px; }
.cmp-risk-v { font-size: 28px; font-weight: 700; color: white; margin: 0 0 1px; }
.cmp-risk-l { font-size: 12px; color: rgba(255,255,255,0.85); margin: 0; }
.cmp-advisory { font-size: 12px; color: #374151; background: #F8FAFC; border-radius: 10px; padding: 12px; line-height: 1.65; margin: 0; }

/* ── Footer ─────────────────────────────────────── */
.vayu-footer {
    margin-top: 56px;
    padding-top: 20px;
    border-top: 1px solid #F3F4F6;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
.vayu-footer span { font-size: 12px; color: #D1D5DB; }

/* ── Forecast note box ──────────────────────────── */
.fc-note { background: #F8FAFC; border-radius: 10px; padding: 11px 14px; margin-top: 10px; }
.fc-note p { font-size: 12px; color: #6B7280; line-height: 1.65; margin: 0; }

/* ── Force dropdown arrow on Streamlit Selectboxes ───────────────── */

[data-baseweb="select"] svg {
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
    color: #6B7280 !important;
    width: 18px !important;
    height: 18px !important;
}

[data-baseweb="select"] {
    position: relative;
}

[data-baseweb="select"] > div {
    min-height:48px !important;
    height:48px !important;
    border-radius:10px !important;
    border:1px solid #E5E7EB !important;
    background:white !important;

    display:flex !important;
    align-items:center !important;
}

[data-baseweb="select"] span{
    display:flex !important;
    align-items:center !important;
    height:100% !important;
    font-size:15px !important;
    color:#111827 !important;
}

[data-baseweb="select"] input {
    font-size: 14px !important;
}

/* ── Mobile ─────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container {
        padding-left: 18px !important;
        padding-right: 18px !important;
        padding-top: 22px !important;
    }
    .vayu-brand h1 { font-size: 30px !important; }
    .mc-val { font-size: 34px !important; }
    .mc-val-sm { font-size: 22px !important; }
    .mc { min-height: 120px !important; padding: 16px !important; }
    .shd h2 { font-size: 17px !important; }
    .cmp-stat-grid { grid-template-columns: 1fr; }
}
@media (max-width: 480px) {
    .vayu-github { display: none; }
    .ts-pill { display: none; }
}

/* Historical range buttons */

div[data-testid="stButton"] > button {
    height: 48px;
    border-radius: 14px;
    border: 2px solid #D1D5DB;
    background: white;
    font-weight: 600;
    transition: all .2s ease;
}

/* ACTIVE */
div[data-testid="stButton"] > button[kind="primary"] {
    border-color: #FF5A5F !important;
    background: #FFF5F5 !important;
    color: #FF5A5F !important;
    opacity: 1;
}

/* INACTIVE */
div[data-testid="stButton"] > button[kind="secondary"] {
    border-color: #D1D5DB !important;
    background: white !important;
    color: rgba(17,24,39,.45) !important;
}

</style>
"""