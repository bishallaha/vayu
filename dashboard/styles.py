# dashboard/styles.py

VAYU_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #111827;
}

.main { background-color: #F7F9FC !important; }

section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1440px !important;
}

/* ── Navigation ───────────────────────────────── */
.vayu-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 24px 0;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 28px;
}
.vayu-logo {
    font-size: 26px;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1;
}
.vayu-logo span { color: #2563EB; }
.vayu-tagline {
    font-size: 12px;
    color: #9CA3AF;
    font-weight: 400;
    margin: 2px 0 0;
}
.nav-timestamp {
    font-size: 12px;
    color: #9CA3AF;
    background: #F3F4F6;
    padding: 4px 12px;
    border-radius: 100px;
}

/* ── Alert Banner ─────────────────────────────── */
.alert-banner {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid #DC2626;
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}
.alert-title {
    font-size: 15px;
    font-weight: 600;
    color: #991B1B;
    margin: 0 0 3px;
}
.alert-body {
    font-size: 13px;
    color: #B91C1C;
    margin: 0;
    line-height: 1.5;
}

/* ── Metric Cards ─────────────────────────────── */
.metric-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 1px solid #E5E7EB;
    height: 158px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    cursor: default;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.09);
}
.metric-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.metric-label {
    font-size: 12px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-icon { font-size: 18px; opacity: 0.5; }
.metric-value {
    font-size: 38px;
    font-weight: 700;
    color: #111827;
    line-height: 1;
    margin: 8px 0 4px;
}
.metric-value-sm {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    line-height: 1;
    margin: 8px 0 4px;
}
.metric-bottom {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.metric-delta-up   { font-size: 12px; font-weight: 500; color: #DC2626; }
.metric-delta-down { font-size: 12px; font-weight: 500; color: #10B981; }
.metric-delta-flat { font-size: 12px; font-weight: 500; color: #9CA3AF; }
.metric-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 100px;
    letter-spacing: 0.03em;
}

/* ── Section Headers ──────────────────────────── */
.section-title {
    font-size: 17px;
    font-weight: 600;
    color: #111827;
    margin: 0 0 4px;
}
.section-caption {
    font-size: 13px;
    color: #9CA3AF;
    margin: 0 0 16px;
}

/* ── Card Container ───────────────────────────── */
.vayu-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    border: 1px solid #E5E7EB;
}

/* ── Health Advisory ──────────────────────────── */
.advisory-main {
    font-size: 15px;
    color: #374151;
    line-height: 1.7;
    background: #F7F9FC;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    border-left: 3px solid #2563EB;
}
.demo-flag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 500;
    margin: 4px 4px 4px 0;
}
.flag-alert {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    color: #991B1B;
}
.flag-safe {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #166534;
}

/* ── SHAP Features ────────────────────────────── */
.shap-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #F9FAFB;
}
.shap-name {
    font-size: 13px;
    font-weight: 500;
    color: #374151;
    width: 170px;
    flex-shrink: 0;
}
.shap-desc {
    font-size: 11px;
    color: #9CA3AF;
    width: 170px;
    flex-shrink: 0;
    margin-top: 2px;
}
.shap-val {
    font-size: 12px;
    font-weight: 500;
    color: #6B7280;
    width: 55px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Divider ──────────────────────────────────── */
.vayu-divider {
    height: 1px;
    background: #E5E7EB;
    margin: 32px 0;
}

/* ── Buttons ──────────────────────────────────── */
.stButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 9px 22px !important;
    transition: background 0.2s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
}

/* ── Selectbox ────────────────────────────────── */
.stSelectbox label { font-size: 13px; font-weight: 500; color: #374151; }
.stSelectbox > div > div {
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    background: #FFFFFF !important;
    font-size: 14px !important;
}

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #F3F4F6;
    border-radius: 12px;
    padding: 4px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: #6B7280;
    padding: 7px 18px;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #111827 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Model performance table ──────────────────── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Footer ───────────────────────────────────── */
.vayu-footer {
    text-align: center;
    padding: 40px 0 16px;
    font-size: 12px;
    color: #D1D5DB;
}
</style>
"""