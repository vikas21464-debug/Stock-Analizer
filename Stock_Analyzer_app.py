# ============================================================
# Stock Breakout & Monthly Valuation Analyzer  (Pro UI)
# Streamlit + Plotly  |  by Vikas Dhiman
# ============================================================
import warnings, logging
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="Breakout & Valuation Analyzer", page_icon="📈", layout="wide")

# ------------------------------------------------------------
# Theme (Auto follows Streamlit's own theme; can be overridden)
# ------------------------------------------------------------
LIGHT = dict(
    mode="light", bg="#ffffff", side="#f4f7fa", card="#ffffff", border="#e3e9ef",
    ink="#13293d", muted="#6b7c8f", up="#0f9d8a", down="#e0475b",
    hero1="#13293d", hero2="#1f4b63", hero_sub="#bcd0de",
    dma={30: "#1976d2", 50: "#f9a825", 124: "#8e24aa", 200: "#e65100"}, car="#111111",
    buy_bg="#d9f3ee", buy_fg="#0a6b5d", sell_bg="#fde1e5", sell_fg="#a3243a", hold_bg="#e9eef3", hold_fg="#41566b",
    g_buy="#cfeee8", g_hold="#eef2f6", g_sell="#fbd9de", mk_buy="#2e7d32", mk_sell="#c62828", grid="#e8edf2",
)
DARK = dict(
    mode="dark", bg="#0e1621", side="#131e2b", card="#162232", border="#263548",
    ink="#e6edf5", muted="#8fa3b8", up="#26c6a2", down="#ff6b7f",
    hero1="#0b2a3d", hero2="#14566e", hero_sub="#a9c4d6",
    dma={30: "#64b5f6", 50: "#ffd54f", 124: "#ce93d8", 200: "#ff9e5e"}, car="#ffffff",
    buy_bg="#0f3d36", buy_fg="#5eead4", sell_bg="#4a1a24", sell_fg="#ff9aa9", hold_bg="#233447", hold_fg="#b6c7d8",
    g_buy="#12453d", g_hold="#1f2e3f", g_sell="#4d1f2a", mk_buy="#4ade80", mk_sell="#ff6b7f", grid="#22324a",
)


def detect_theme():
    try:
        t = st.context.theme.type          # Streamlit >= 1.46
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return st.get_option("theme.base") or "light"


with st.sidebar:
    theme_pref = st.radio("Appearance", ["Auto", "Light", "Dark"], horizontal=True,
                          help="Auto follows the Streamlit theme (menu ⋮ → Settings).")
mode = detect_theme() if theme_pref == "Auto" else theme_pref.lower()
P = DARK if mode == "dark" else LIGHT
UP, DOWN, INK, MUTED = P["up"], P["down"], P["ink"], P["muted"]

force_bg = f"""
.stApp, [data-testid="stHeader"] {{ background: {P['bg']}; color: {P['ink']}; }}
[data-testid="stSidebar"] {{ background: {P['side']}; }}
.stApp p, .stApp label, .stApp span, .stApp li, .stApp [data-testid="stMarkdownContainer"] {{ color: {P['ink']}; }}
.stApp [data-testid="stCaptionContainer"], .stApp small {{ color: {P['muted']}; }}
""" if theme_pref != "Auto" else ""

st.markdown(f"""
<style>
{force_bg}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {P['ink']}; letter-spacing: -0.01em; }}
.hero {{ background: linear-gradient(120deg, {P['hero1']} 0%, {P['hero2']} 100%); color: #fff;
        padding: 22px 28px; border-radius: 14px; margin-bottom: 18px; border: 1px solid {P['border']}; }}
.hero h1 {{ color: #fff; margin: 0; font-size: 1.7rem; }}
.hero p {{ margin: 4px 0 0; color: {P['hero_sub']}; font-size: .95rem; }}
.kpi {{ background: {P['card']}; border: 1px solid {P['border']}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
.kpi .l {{ color: {P['muted']}; font-size: .78rem; }}
.kpi .v {{ color: {P['ink']}; font-size: 1.45rem; font-weight: 700; margin-top: 2px; }}
.kpi .s {{ font-size: .8rem; margin-top: 2px; color: {P['muted']}; }}
.pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 600; font-size: .9rem; }}
.buy {{ background:{P['buy_bg']}; color:{P['buy_fg']}; }}
.sell {{ background:{P['sell_bg']}; color:{P['sell_fg']}; }}
.hold {{ background:{P['hold_bg']}; color:{P['hold_fg']}; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Logic
# ------------------------------------------------------------
def get_valuation_signal(r):
    if r <= 0.55: return "5th BUY"
    if r <= 0.64: return "4th BUY"
    if r <= 0.73: return "3rd BUY"
    if r <= 0.82: return "2nd BUY"
    if r <= 0.91: return "1st BUY"
    if r >= 1.45: return "5th Sell"
    if r >= 1.36: return "4th Sell"
    if r >= 1.27: return "3rd Sell"
    if r >= 1.18: return "2nd Sell"
    if r >= 1.09: return "1st Sell"
    return "HOLD"


def pill(sig):
    s = sig.upper()
    cls = "buy" if "BUY" in s else "sell" if "SELL" in s else "hold"
    return f'<span class="pill {cls}">{sig}</span>'


@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.sort_index()


def analyze_stock(ticker, as_of_date):
    start = as_of_date - timedelta(days=900)
    data = load_data(ticker, start.strftime("%Y-%m-%d"), (as_of_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    if data.empty:
        return None, "No data found for this stock/index."
    data = data.loc[data.index.normalize() <= pd.Timestamp(as_of_date)]
    if data.empty:
        return None, "No trading data on or before the selected date."
    if len(data) < 200:
        return None, "Not enough history to calculate the 200 DMA as of this date."

    close = data["Close"].squeeze()
    session = close.index[-1]
    cmp = float(close.iloc[-1])
    dma = {n: close.rolling(n).mean() for n in (30, 50, 124, 200)}
    ratio = close / dma[124]

    py, pm = (session.year - 1, 12) if session.month == 1 else (session.year, session.month - 1)
    prev = ratio[(ratio.index.year == py) & (ratio.index.month == pm)].dropna()
    if prev.empty:
        return None, "Unable to calculate the previous month's average CMP / 124 DMA."
    avg_ratio = float(prev.mean())

    # Signals on the 1st trading day of each month
    sig = pd.Series(index=close.index, dtype=object)
    for (yr, mo), g in close.groupby([close.index.year, close.index.month]):
        yy, mm = (yr - 1, 12) if mo == 1 else (yr, mo - 1)
        pr = ratio[(ratio.index.year == yy) & (ratio.index.month == mm)].dropna()
        sig.loc[g.index[0]] = get_valuation_signal(float(pr.mean())) if len(pr) >= 5 else "HOLD"

    last1y = data.tail(252)
    high_date = last1y["High"].squeeze().idxmax()
    car = close.loc[high_date:].expanding().mean()

    out = {
        "session": session, "cmp": cmp,
        "dma": {n: float(s.iloc[-1]) for n, s in dma.items()},
        "ratio_now": cmp / float(dma[124].iloc[-1]), "ratio_prev": avg_ratio,
        "signal": get_valuation_signal(avg_ratio), "val_month": f"{py}-{pm:02d}", "days_used": len(prev),
        "dist200": (cmp - float(dma[200].iloc[-1])) / float(dma[200].iloc[-1]) * 100,
        "ohlc": data.tail(252).copy(), "dma_series": {n: s.reindex(last1y.index) for n, s in dma.items()},
        "car": car, "high_date": high_date, "signals": sig.reindex(last1y.index),
        "car_status": None, "breakout": None,
    }
    if len(car) >= 10:
        out["car_status"] = "Positive" if car.tail(10).is_monotonic_increasing else "Negative"
        out["breakout"] = (cmp > out["dma"][30] and cmp > out["dma"][50] and cmp > out["dma"][200]
                           and out["car_status"] == "Positive")
    return out, None


# ------------------------------------------------------------
# Charts
# ------------------------------------------------------------
def build_chart(r, ticker, show):
    ohlc, idx = r["ohlc"], r["ohlc"].index
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=idx, open=ohlc["Open"], high=ohlc["High"], low=ohlc["Low"], close=ohlc["Close"],
        increasing_line_color=UP, increasing_fillcolor=UP,
        decreasing_line_color=DOWN, decreasing_fillcolor=DOWN, name="Price"), row=1, col=1)

    palette = P["dma"]
    for n in (30, 50, 124, 200):
        if show.get(f"{n} DMA"):
            s = r["dma_series"][n].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s, name=f"{n} DMA", mode="lines",
                                     line=dict(color=palette[n], width=1.6 if n == 200 else 1.2),
                                     hovertemplate=f"{n} DMA: %{{y:.2f}}<extra></extra>"), row=1, col=1)

    if show.get("CAR"):
        c = r["car"].reindex(idx).dropna()
        fig.add_trace(go.Scatter(x=c.index, y=c, name="CAR", mode="lines",
                                 line=dict(color=P["car"], width=2.4),
                                 hovertemplate="CAR: %{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_shape(type="line", x0=r["high_date"], x1=r["high_date"], yref="paper", y0=0, y1=1,
                      line=dict(color=MUTED, width=1, dash="dash"))
        fig.add_annotation(x=r["high_date"], yref="paper", y=1.02, text="52W high", showarrow=False,
                           font=dict(size=10, color=MUTED))

    if show.get("Signals"):
        sg = r["signals"].dropna()
        sg = sg[sg != "HOLD"]
        for key, color, sym, nm in (("BUY", P["mk_buy"], "triangle-up", "BUY signal"),
                                    ("SELL", P["mk_sell"], "triangle-down", "SELL signal")):
            m = sg[sg.str.upper().str.contains(key)]
            if not m.empty:
                fig.add_trace(go.Scatter(
                    x=m.index, y=ohlc.loc[m.index, "Low"] * 0.985, mode="markers", name=nm,
                    marker=dict(symbol=sym, size=13, color=color, line=dict(color=P["bg"], width=1)),
                    text=m.values, hovertemplate="%{text}<br>%{x|%d %b %Y}<extra></extra>"), row=1, col=1)

    if "Volume" in ohlc.columns:
        vcol = np.where(ohlc["Close"] >= ohlc["Open"], UP, DOWN)
        fig.add_trace(go.Bar(x=idx, y=ohlc["Volume"], marker_color=vcol, opacity=0.55,
                             name="Volume", showlegend=False), row=2, col=1)

    fig.update_layout(
        height=640, template="plotly_dark" if P["mode"] == "dark" else "plotly_white",
        paper_bgcolor=P["bg"], plot_bgcolor=P["bg"], font=dict(color=P["ink"]), hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.07, x=0), xaxis_rangeslider_visible=False,
        title=dict(text=f"{ticker.replace('.NS', '')} · 1Y daily", x=0, font=dict(size=16, color=P["ink"])),
        dragmode="pan")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showspikes=True, spikethickness=1, gridcolor=P["grid"])
    fig.update_yaxes(gridcolor=P["grid"])
    fig.update_xaxes(rangeselector=dict(buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(step="all", label="1Y")], bgcolor=P["card"], activecolor=P["border"], font=dict(color=P["ink"])), row=1, col=1)
    return fig


def ratio_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(valueformat=".3f", font=dict(size=30, color=P["ink"])),
        gauge=dict(axis=dict(range=[0.4, 1.6], tickcolor=P["muted"]), bar=dict(color=P["ink"], thickness=0.25), bgcolor=P["card"], bordercolor=P["border"],
                   steps=[dict(range=[0.4, 0.91], color=P["g_buy"]), dict(range=[0.91, 1.09], color=P["g_hold"]),
                          dict(range=[1.09, 1.6], color=P["g_sell"])])))
    fig.update_layout(height=210, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=P["ink"]))
    return fig


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown('<div class="hero"><h1>Breakout & Monthly Valuation Analyzer</h1>'
            '<p>Stocks and indices · Buy/Sell signals on the first trading day of each month · by Vikas Dhiman</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    quick = st.selectbox("Quick pick", ["—", "RELIANCE", "TCS", "INFY", "HDFCBANK", "^NSEI", "^NSEBANK", "^BSESN"])
    ticker_in = st.text_input("Ticker", value="RELIANCE" if quick == "—" else quick,
                              help="Type a name for NSE stocks. Use ^NSEI, ^NSEBANK, ^BSESN for indices.")
    as_of = st.date_input("As-of date", value=date.today(), max_value=date.today(),
                          help="Weekends and holidays use the previous trading session.")
    st.subheader("Chart layers")
    show = {k: st.checkbox(k, value=v) for k, v in
            {"30 DMA": True, "50 DMA": True, "124 DMA": False, "200 DMA": True, "CAR": True, "Signals": True}.items()}
    run = st.button("Analyze", type="primary", use_container_width=True)

if "ran" not in st.session_state:
    st.session_state.ran = False
if run:
    st.session_state.ran = True

if not st.session_state.ran:
    st.info("Pick a ticker and date in the sidebar, then select **Analyze**.")
    st.stop()

ticker = ticker_in.strip().upper()
if not ticker:
    st.error("Enter a stock or index name.")
    st.stop()
if not ticker.startswith("^") and not ticker.endswith(".NS"):
    ticker += ".NS"

with st.spinner(f"Loading {ticker}…"):
    r, err = analyze_stock(ticker, as_of)
if err:
    st.error(err)
    st.stop()

sess = r["session"].strftime("%d-%m-%Y")
if sess != as_of.strftime("%d-%m-%Y"):
    st.info(f"{as_of.strftime('%d-%m-%Y')} was not a trading day. Using session {sess}.")

k = st.columns(4)
kpi(k[0], "Last price", f"{r['cmp']:,.2f}", f"Session {sess}")
kpi(k[1], "Valuation signal", pill(r["signal"]), f"Based on {r['val_month']} ({r['days_used']} days)")
car_txt = r["car_status"] or "n/a"
kpi(k[2], "CAR status", car_txt, "Last 10 sessions" if r["car_status"] else "Under 10 days since 52W high")
if r["breakout"] is None:
    kpi(k[3], "Action", "n/a", "Not enough CAR data")
else:
    kpi(k[3], "Action", "🟢 Positive breakout" if r["breakout"] else "🔴 Avoid / Hold",
        f"{r['dist200']:+.2f}% from 200 DMA")

st.write("")
tab1, tab2, tab3 = st.tabs(["Chart", "Levels & valuation", "Signal history"])

with tab1:
    st.plotly_chart(build_chart(r, ticker, show), use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True})
    st.caption("Drag to pan, scroll to zoom, double-click to reset. Triangles mark the first trading day of a month.")

with tab2:
    a, b = st.columns([1, 1.3])
    with a:
        st.markdown("**CMP / 124 DMA now**")
        st.plotly_chart(ratio_gauge(r["ratio_now"]), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Previous-month average: **{r['ratio_prev']:.4f}**")
    with b:
        d = r["dma"]
        st.dataframe(pd.DataFrame({
            "Metric": ["30 DMA", "50 DMA", "124 DMA", "200 DMA", "Distance from 200 DMA",
                       "Previous-month avg CMP / 124 DMA", "Valuation month", "Trading days used"],
            "Value": [f"{d[30]:.2f}", f"{d[50]:.2f}", f"{d[124]:.2f}", f"{d[200]:.2f}",
                      f"{r['dist200']:.2f}%", f"{r['ratio_prev']:.4f}", r["val_month"], r["days_used"]]}),
            hide_index=True, use_container_width=True)

with tab3:
    sg = r["signals"].dropna().rename("Signal").to_frame()
    sg["Close"] = r["ohlc"].loc[sg.index, "Close"].round(2)
    sg.index = sg.index.strftime("%d-%m-%Y")
    sg.index.name = "First trading day"
    st.dataframe(sg.iloc[::-1], use_container_width=True)
    st.download_button("Download CSV", sg.to_csv().encode(), f"{ticker}_signals.csv", "text/csv")

st.caption("Data via Yahoo Finance · For research only, not investment advice.")
