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
.co {{ display:flex; align-items:center; gap:16px; background:{P['card']}; border:1px solid {P['border']};
       border-radius:14px; padding:14px 18px; margin-bottom:14px; }}
.co img, .co .ini {{ width:56px; height:56px; border-radius:12px; flex:none; }}
.co img {{ object-fit:contain; background:#fff; padding:6px; border:1px solid {P['border']}; box-sizing:border-box; }}
.co .ini {{ display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.2rem;
           color:#fff; background:linear-gradient(135deg,{P['hero1']},{P['hero2']}); }}
.co .n {{ color:{P['ink']}; font-size:1.35rem; font-weight:700; line-height:1.2; }}
.co .m {{ color:{P['muted']}; font-size:.85rem; margin-top:3px; }}
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


# ------------------------------------------------------------
# Logos: built-in domain map (fallback when Yahoo has no website) + several public sources,
# validated server-side and embedded as base64. Failures are never cached, so they retry.
# ------------------------------------------------------------
_DOMAINS = """
RELIANCE ril.com|TCS tcs.com|INFY infosys.com|HDFCBANK hdfcbank.com|ICICIBANK icicibank.com|SBIN sbi.co.in
BHARTIARTL airtel.in|ITC itcportal.com|LT larsentoubro.com|KOTAKBANK kotak.com|AXISBANK axisbank.com
HINDUNILVR hul.co.in|BAJFINANCE bajajfinance.in|BAJAJFINSV bajajfinserv.in|MARUTI marutisuzuki.com|M&M mahindra.com
SUNPHARMA sunpharma.com|TATAMOTORS tatamotors.com|TATASTEEL tatasteel.com|NTPC ntpc.co.in|POWERGRID powergrid.in
ONGC ongcindia.com|COALINDIA coalindia.in|ULTRACEMCO ultratechcement.com|ASIANPAINT asianpaints.com|TITAN titancompany.in
NESTLEIND nestle.in|WIPRO wipro.com|HCLTECH hcltech.com|TECHM techmahindra.com|LTIM ltimindtree.com
ADANIENT adanienterprises.com|ADANIPORTS adaniports.com|ADANIGREEN adanigreenenergy.com|ADANIPOWER adanipower.com
ADANIENSOL adanienergysolutions.com|ATGL adanigas.com|AWL awlagri.com|JSWSTEEL jsw.in|JSWENERGY jsw.in
HINDALCO hindalco.com|VEDL vedantalimited.com|GRASIM grasim.com|CIPLA cipla.com|DRREDDY drreddys.com
DIVISLAB divislabs.com|APOLLOHOSP apollohospitals.com|EICHERMOT eicher.in|BAJAJ-AUTO bajajauto.com
HEROMOTOCO heromotocorp.com|TVSMOTOR tvsmotor.com|BRITANNIA britannia.co.in|DABUR dabur.com|GODREJCP godrejcp.com
TATACONSUM tataconsumer.com|MARICO marico.com|INDUSINDBK indusind.com|BANKBARODA bankofbaroda.in|PNB pnbindia.in
CANBK canarabank.com|FEDERALBNK federalbank.co.in|IDFCFIRSTB idfcfirstbank.com|AUBANK aubank.in|BANDHANBNK bandhanbank.com
RBLBANK rblbank.com|YESBANK yesbank.in|IDBI idbibank.in|UNIONBANK unionbankofindia.co.in|BANKINDIA bankofindia.co.in
IOB iob.in|BPCL bharatpetroleum.in|IOC iocl.com|GAIL gailonline.com|HINDPETRO hindustanpetroleum.com
TATAPOWER tatapower.com|TORNTPOWER torrentpower.com|CESC cesc.co.in|NHPC nhpcindia.com|SJVN sjvn.nic.in
OIL oil-india.com|PETRONET petronetlng.com|IGL iglonline.net|MGL mahanagargas.com
HDFCLIFE hdfclife.com|SBILIFE sbilife.co.in|ICICIPRULI iciciprulife.com|ICICIGI icicilombard.com|LICI licindia.in
NIACL newindia.co.in|GICRE gicre.in|HDFCAMC hdfcfund.com|SBICARD sbicard.com|CHOLAFIN cholamandalam.com
SHRIRAMFIN shriramfinance.in|MUTHOOTFIN muthootfinance.com|MANAPPURAM manappuram.com|LICHSGFIN lichousing.com
PFC pfcindia.com|RECLTD recindia.nic.in|IRFC irfc.co.in|POONAWALLA poonawallafincorp.com|ABCAPITAL adityabirlacapital.com
ANGELONE angelone.in|CDSL cdslindia.com|BSE bseindia.com|MCX mcxindia.com|IEX iexindia.com|CAMS camsonline.com
KFINTECH kfintech.com|JIOFIN jiofinance.com|BAJAJHLDNG bajajholdings.in|TATAINVEST tatainvestment.com|MFSL maxfinancial.com
JINDALSTEL jindalsteelpower.com|SAIL sail.co.in|NMDC nmdc.co.in|HINDZINC hzlindia.com|NATIONALUM nalcoindia.com
HINDCOPPER hindustancopper.com|MOIL moil.nic.in|LUPIN lupin.com|AUROPHARMA aurobindo.com|TORNTPHARM torrentpharma.com
BIOCON biocon.com|ZYDUSLIFE zyduslife.com|MANKIND mankindpharma.com|MAXHEALTH maxhealthcare.in|GLENMARK glenmarkpharma.com
ALKEM alkem.com|IPCALAB ipcalabs.com|LAURUSLABS lauruslabs.com|ABBOTINDIA abbott.co.in|SYNGENE syngeneintl.com
PERSISTENT persistent.com|COFORGE coforge.com|MPHASIS mphasis.com|LTTS ltts.com|KPITTECH kpit.com|OFSS oracle.com
CYIENT cyient.com|ZENSARTECH zensar.com|BSOFT birlasoft.com|TATAELXSI tataelxsi.com|TATACOMM tatacommunications.com
ASHOKLEY ashokleyland.com|MOTHERSON motherson.com|BOSCHLTD bosch.in|MRF mrftyres.com|APOLLOTYRE apollotyres.com
BALKRISIND bkt-tires.com|EXIDEIND exideindustries.com|ESCORTS escortskubota.com|CUMMINSIND cummins.com|SONACOMS sonacomstar.com
DLF dlf.in|GODREJPROP godrejproperties.com|LODHA lodhagroup.com|OBEROIRLTY oberoirealty.com|PRESTIGE prestigeconstructions.com
PHOENIXLTD thephoenixmills.com|HAL hal-india.co.in|BEL bel-india.in|BHEL bhel.com|IRCTC irctc.co.in|CONCOR concorindia.com
ETERNAL eternal.com|ZOMATO eternal.com|NYKAA nykaa.com|PAYTM paytm.com|POLICYBZR policybazaar.com|DMART dmartindia.com
TRENT trentlimited.com|PIDILITIND pidilite.com|SIEMENS siemens.co.in|ABB abb.com|HAVELLS havells.com|VOLTAS voltas.com
POLYCAB polycab.com|KEI kei-ind.com|CROMPTON crompton.co.in|WHIRLPOOL whirlpoolindia.com|BLUESTARCO bluestarindia.com
DIXON dixoninfo.com|ASTRAL astralltd.com|SUPREMEIND supreme.co.in|INDIGO goindigo.in|DELHIVERY delhivery.com
NAUKRI infoedge.in|INFOEDGE infoedge.in|VBL varunbeverages.com|COLPAL colgatepalmolive.co.in|BERGEPAINT bergerpaints.com
PAGEIND pageindustries.com|JUBLFOOD jubilantfoodworks.com|DEVYANI devyani.in|MCDOWELL-N diageoindia.com
UBL unitedbreweries.com|RADICO radicokhaitan.com|AMBUJACEM ambujacement.com|ACC acclimited.com|SHREECEM shreecement.com
JKCEMENT jkcement.com|DALBHARAT dalmiabharat.com|RAMCOCEM ramcocements.in|UPL upl-ltd.com|PIIND piindustries.com
TATACHEM tatachemicals.com|SRF srf.com|DEEPAKNTR godeepak.com|COROMANDEL coromandel.biz|CHAMBLFERT chambal.in
SUNTV sunnetwork.in|ZEEL zee.com|PVRINOX pvrinox.com|IDEA myvi.in|VODAIDEA myvi.in
"""
DOMAINS = {}
for _chunk in _DOMAINS.replace("\n", "|").split("|"):
    _p = _chunk.split()
    if len(_p) == 2:
        DOMAINS[_p[0].upper()] = _p[1]


def _logo_sources(domain):
    return [
        f"https://t1.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=https://{domain}&size=256",
        f"https://www.google.com/s2/favicons?domain={domain}&sz=256",
        f"https://icons.duckduckgo.com/ip3/{domain}.ico",
        f"https://{domain}/apple-touch-icon.png",
        f"https://{domain}/favicon.ico",
    ]


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def _fetch_logo(domain, yahoo_logo=""):
    """data: URI of the first valid image. Raises when nothing works, so failures are NOT cached."""
    import base64, requests
    urls = ([yahoo_logo] if yahoo_logo else []) + (_logo_sources(domain) if domain else [])
    for u in urls:
        try:
            r = requests.get(u, timeout=4, headers={"User-Agent": "Mozilla/5.0"})
            ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
            if r.status_code == 200 and ct.startswith("image/") and len(r.content) > 800:
                return f"data:{ct};base64," + base64.b64encode(r.content).decode()
        except Exception:
            continue
    raise ValueError("no logo")


@st.cache_data(ttl=86400, show_spinner=False)
def _yahoo_info(ticker):
    """Raises on empty/failed responses so they are retried instead of cached."""
    info = yf.Ticker(ticker).info or {}
    if not (info.get("longName") or info.get("shortName")):
        raise ValueError("empty info")
    return info


def get_company(ticker):
    """Full name, logo (data URI) and profile line. Best effort; every step has a fallback."""
    from urllib.parse import urlparse
    try:
        info = _yahoo_info(ticker)
    except Exception:
        info = {}
    base = ticker.replace(".NS", "").lstrip("^")
    name = info.get("longName") or info.get("shortName") or base
    site = info.get("website") or ""
    domain = ""
    if site:
        domain = urlparse(site if "//" in site else "//" + site).netloc.replace("www.", "")
    domain = domain or DOMAINS.get(base.upper(), "")
    logo = ""
    if not ticker.startswith("^"):
        try:
            logo = _fetch_logo(domain, info.get("logo_url") or "")
        except Exception:
            logo = ""
    bits = [info.get("sector"), info.get("industry"), info.get("exchange")]
    return {"name": name, "logo": logo, "site": site, "meta": " · ".join(b for b in bits if b),
            "ok": bool(info)}


def company_header(ticker):
    c = get_company(ticker)
    import html as _h
    nm = _h.escape(c["name"])
    ini = _h.escape("".join(w[0] for w in c["name"].replace("&", " ").split()[:2]).upper() or "?")
    badge = (f'<img src="{_h.escape(c["logo"])}" alt="{nm} logo">') if c["logo"] else f'<div class="ini">{ini}</div>'
    sym = ticker.replace(".NS", "")
    meta = f'{sym}' + (f' · {_h.escape(c["meta"])}' if c["meta"] else "")
    st.markdown(f'<div class="co">{badge}<div><div class="n">{nm}</div><div class="m">{meta}</div></div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------
# Indices (Yahoo Finance tickers). Use "Custom ticker…" in the menu for anything not listed.
# ------------------------------------------------------------
INDICES = [  # (group, display name, Yahoo ticker)
    ("Broad market", "Nifty 50", "^NSEI"), ("Broad market", "Sensex", "^BSESN"),
    ("Broad market", "Nifty Next 50", "^NSMIDCP"), ("Broad market", "Nifty 100", "^CNX100"),
    ("Broad market", "Nifty 200", "^CNX200"), ("Broad market", "Nifty 500", "^CRSLDX"),
    ("Broad market", "Nifty Midcap 50", "^NSEMDCP50"), ("Broad market", "Nifty Midcap 100", "^CNXMIDCAP"),
    ("Broad market", "Nifty Smallcap 100", "^CNXSC"),
    ("Broad market", "BSE 100", "BSE-100.BO"), ("Broad market", "BSE 200", "BSE-200.BO"),
    ("Broad market", "BSE 500", "BSE-500.BO"), ("Broad market", "BSE Midcap", "BSE-MIDCAP.BO"),
    ("Broad market", "BSE Smallcap", "BSE-SMLCAP.BO"),
    ("Sector", "Nifty Bank", "^NSEBANK"), ("Sector", "Nifty Financial Services", "^CNXFIN"),
    ("Sector", "Nifty PSU Bank", "^CNXPSUBANK"), ("Sector", "Nifty IT", "^CNXIT"),
    ("Sector", "Nifty Auto", "^CNXAUTO"), ("Sector", "Nifty Pharma", "^CNXPHARMA"),
    ("Sector", "Nifty FMCG", "^CNXFMCG"), ("Sector", "Nifty Metal", "^CNXMETAL"),
    ("Sector", "Nifty Energy", "^CNXENERGY"), ("Sector", "Nifty Realty", "^CNXREALTY"),
    ("Sector", "Nifty Media", "^CNXMEDIA"), ("Sector", "Nifty Infrastructure", "^CNXINFRA"),
    ("Theme", "Nifty PSE", "^CNXPSE"), ("Theme", "Nifty Commodities", "^CNXCMDT"),
    ("Theme", "Nifty Consumption", "^CNXCONSUM"), ("Theme", "Nifty Services Sector", "^CNXSERVICE"),
    ("Theme", "Nifty MNC", "^CNXMNC"),
    ("Global", "S&P 500", "^GSPC"), ("Global", "Nasdaq Composite", "^IXIC"), ("Global", "Dow Jones", "^DJI"),
    ("Global", "Russell 2000", "^RUT"), ("Global", "FTSE 100", "^FTSE"), ("Global", "DAX", "^GDAXI"),
    ("Global", "CAC 40", "^FCHI"), ("Global", "Nikkei 225", "^N225"), ("Global", "Hang Seng", "^HSI"),
    ("Global", "Shanghai Composite", "000001.SS"), ("Global", "KOSPI", "^KS11"),
    ("Global", "ASX 200", "^AXJO"), ("Global", "Straits Times", "^STI"),
]
INDEX_MAP = {name: (group, tick) for group, name, tick in INDICES}
CUSTOM_INDEX = "Custom ticker…"


def index_header(name, ticker, group):
    import html as _h
    ini = _h.escape("".join(w[0] for w in name.replace("&", " ").split()[:3]).upper() or "IX")
    st.markdown(f'<div class="co"><div class="ini">{ini}</div><div><div class="n">{_h.escape(name)}</div>'
                f'<div class="m">{_h.escape(group)} index · {_h.escape(ticker)}</div></div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------
# Mutual funds: AMFI NAV data via mfapi.in (free, no key)
# ------------------------------------------------------------
MF_API = "https://api.mfapi.in/mf"
_UA = {"User-Agent": "Mozilla/5.0"}
_AMC_DOMAINS = [
    ("parag parikh", "ppfas.com"), ("sbi", "sbimf.com"), ("hdfc", "hdfcfund.com"),
    ("icici", "icicipruamc.com"), ("nippon", "mf.nipponindiaim.com"), ("axis", "axismf.com"),
    ("kotak", "kotakmf.com"), ("aditya birla", "mutualfund.adityabirlacapital.com"), ("uti", "utimf.com"),
    ("mirae", "miraeassetmf.co.in"), ("dsp", "dspim.com"), ("franklin", "franklintempletonindia.com"),
    ("tata", "tatamutualfund.com"), ("motilal", "motilaloswalmf.com"), ("canara", "canararobeco.com"),
    ("invesco", "invescomutualfund.com"), ("bandhan", "bandhanmf.com"), ("idfc", "bandhanmf.com"),
    ("hsbc", "assetmanagement.hsbc.co.in"), ("quantum", "quantumamc.com"), ("quant", "quantmutual.com"),
    ("pgim", "pgimindiamf.com"), ("sundaram", "sundarammutual.com"), ("mahindra", "mahindramanulife.com"),
    ("whiteoak", "whiteoakamc.com"), ("navi", "navimutualfund.com"), ("baroda", "barodabnpparibasmf.in"),
    ("union", "unionmf.com"), ("edelweiss", "edelweissmf.com"), ("samco", "samcomf.com"),
    ("groww", "growwmf.in"), ("iti", "itiamc.com"), ("bajaj", "bajajamc.com"), ("lic", "licmf.com"),
    ("jm financial", "jmfinancialmf.com"), ("360 one", "360.one"), ("helios", "heliosmf.in"),
    ("old bridge", "oldbridgemf.com"), ("trust", "trustmf.com"), ("nj", "njmutualfund.com"),
    ("zerodha", "zerodhafundhouse.com"), ("shriram", "shriramamc.in"), ("taurus", "taurusmutualfund.com"),
    ("principal", "principalindia.com"), ("sahara", "saharamutualfund.com"),
]


def amc_domain(text):
    t = f" {text.lower()} "
    for kw, dom in _AMC_DOMAINS:
        if f" {kw} " in t:
            return dom
    return ""


def is_growth_plan(name):
    n = name.lower()
    return "growth" in n and not any(w in n for w in ("idcw", "dividend", "bonus", "payout", "reinvest"))


@st.cache_data(ttl=3600, show_spinner=False)
def mf_search(q):
    import requests
    r = requests.get(f"{MF_API}/search", params={"q": q}, timeout=15, headers=_UA)
    r.raise_for_status()
    res = r.json()
    if not isinstance(res, list):
        raise ValueError("unexpected response")
    return [{"code": int(x["schemeCode"]), "name": str(x["schemeName"]).strip()}
            for x in res if x.get("schemeCode") and x.get("schemeName")]


def parse_nav(payload):
    """mfapi payload -> (DataFrame with Open/High/Low/Close = NAV, meta dict)."""
    meta = payload.get("meta") or {}
    df = pd.DataFrame(payload.get("data") or [])
    if df.empty or not {"date", "nav"} <= set(df.columns):
        raise ValueError("no NAV data")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df[df["nav"] > 0].drop_duplicates("date").set_index("date").sort_index()
    if df.empty:
        raise ValueError("no NAV data")
    nav = df["nav"].astype(float)
    return pd.DataFrame({"Open": nav, "High": nav, "Low": nav, "Close": nav}), meta


@st.cache_data(ttl=3600, show_spinner=False)
def load_fund(code):
    import requests
    r = requests.get(f"{MF_API}/{int(code)}", timeout=20, headers=_UA)
    r.raise_for_status()
    return parse_nav(r.json())


def analyze_fund(code, as_of_date):
    try:
        data, meta = load_fund(code)
    except Exception:
        return None, None, (f"Could not load NAV data for scheme code {code}. "
                            "Check the code, or try again in a minute.")
    data = data.loc[data.index >= pd.Timestamp(as_of_date - timedelta(days=900))]
    if data.empty:
        return None, meta, "This fund has no NAV data in the last 900 days."
    r, err = analyze_data(data, as_of_date, fund=True)
    if err and err.startswith("Not enough history"):
        err = "Not enough NAV history to calculate the 200 DMA as of this date (the fund needs about 200 trading days)."
    return r, meta, err


def fund_header(meta, code, fallback_name=""):
    import html as _h
    name = meta.get("scheme_name") or fallback_name or f"Scheme {code}"
    house = meta.get("fund_house") or ""
    logo = ""
    dom = amc_domain(house or name)
    if dom:
        try:
            logo = _fetch_logo(dom, "")
        except Exception:
            logo = ""
    ini = _h.escape("".join(w[0] for w in (house or name).replace("&", " ").split()[:2]).upper() or "MF")
    badge = f'<img src="{_h.escape(logo)}" alt="logo">' if logo else f'<div class="ini">{ini}</div>'
    bits = [house, meta.get("scheme_category") or meta.get("scheme_type"), f"Scheme code {code}"]
    st.markdown(f'<div class="co">{badge}<div><div class="n">{_h.escape(name)}</div>'
                f'<div class="m">{_h.escape(" · ".join(b for b in bits if b))}</div></div></div>',
                unsafe_allow_html=True)


def first_day_signals(close, ratio):
    """Valuation signal on the 1st trading day of each month, from the previous month's average CMP / 124 DMA."""
    sig = pd.Series(index=close.index, dtype=object)
    for (yr, mo), g in close.groupby([close.index.year, close.index.month]):
        yy, mm = (yr - 1, 12) if mo == 1 else (yr, mo - 1)
        pr = ratio[(ratio.index.year == yy) & (ratio.index.month == mm)].dropna()
        sig.loc[g.index[0]] = get_valuation_signal(float(pr.mean())) if len(pr) >= 5 else "HOLD"
    return sig


def monthly_view(data, as_of_date, years=5, fund=False):
    """Monthly candles from daily data. DMAs are the daily averages read at each month-end. No CAR."""
    data = data.loc[data.index.normalize() <= pd.Timestamp(as_of_date)].dropna(subset=["Close"])
    if len(data) < 200:
        return None, "Not enough history to build the monthly chart (about 200 trading days are needed)."
    close = data["Close"].squeeze()
    dma = {n: close.rolling(n).mean() for n in (30, 50, 124, 200)}
    sig = first_day_signals(close, close / dma[124])

    per = data.index.to_period("M")
    g = data.groupby(per)
    m = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(),
                      "Low": g["Low"].min(), "Close": g["Close"].last()})
    if "Volume" in data.columns:
        m["Volume"] = g["Volume"].sum()
    m.index = m.index.to_timestamp()                       # month start = candle position

    def by_month(series):
        out = series.groupby(series.index.to_period("M")).last()
        out.index = out.index.to_timestamp()
        return out

    sig_m = sig.dropna()
    sig_m.index = sig_m.index.to_period("M").to_timestamp()
    sig_m = sig_m[~sig_m.index.duplicated(keep="first")]

    cutoff = m.index[-1] - pd.DateOffset(months=years * 12 - 1)
    m = m.loc[cutoff:]
    return {"ohlc": m, "dma_series": {n: by_month(s).reindex(m.index) for n, s in dma.items()},
            "signals": sig_m.reindex(m.index), "fund": fund, "monthly": True,
            "session": close.index[-1], "months": len(m)}, None


def get_monthly(is_fund, ticker, fund_code, as_of_date, years):
    """Load enough daily history (plus 200-day warm-up) and build the monthly view."""
    span = timedelta(days=years * 366 + 330)
    if is_fund:
        try:
            data, _ = load_fund(fund_code)
        except Exception:
            return None, "Could not load NAV history for the monthly chart. Try again in a minute."
        data = data.loc[data.index >= pd.Timestamp(as_of_date - span)]
    else:
        start = as_of_date - span
        data = load_data(ticker, start.strftime("%Y-%m-%d"), (as_of_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        if data.empty:
            return None, "No data found for the monthly chart."
    return monthly_view(data, as_of_date, years, fund=is_fund)


def analyze_stock(ticker, as_of_date):
    start = as_of_date - timedelta(days=900)
    data = load_data(ticker, start.strftime("%Y-%m-%d"), (as_of_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    if data.empty:
        return None, "No data found for this stock/index."
    return analyze_data(data, as_of_date)


def analyze_data(data, as_of_date, fund=False):
    """Core analysis on a daily price (or NAV) table with Open/High/Low/Close columns."""
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
    sig = first_day_signals(close, ratio)

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
        "car_status": None, "breakout": None, "fund": fund,
    }
    if len(car) >= 10:
        out["car_status"] = "Positive" if car.tail(10).is_monotonic_increasing else "Negative"
        out["breakout"] = (cmp > out["dma"][30] and cmp > out["dma"][50] and cmp > out["dma"][200]
                           and out["car_status"] == "Positive")
    return out, None


# ------------------------------------------------------------
# Charts
# ------------------------------------------------------------
MIN_CANDLES = 15   # keeps the chart readable when the 52-week high is very recent


def build_chart(r, ticker, show, label=None, view="1Y"):
    ohlc = r["ohlc"]
    start = ohlc.index[0]
    if view == "HIGH":
        start = min(pd.Timestamp(r["high_date"]), ohlc.index[-min(MIN_CANDLES, len(ohlc))])
        ohlc = ohlc.loc[start:]
    idx = ohlc.index
    is_fund = bool(r.get("fund"))
    monthly = bool(r.get("monthly"))
    use_line = is_fund and not monthly      # monthly funds get real candles (first/high/low/last NAV of the month)
    has_vol = (not is_fund) and "Volume" in ohlc.columns and float(ohlc["Volume"].fillna(0).sum()) > 0   # many indices report no volume
    fig = make_subplots(rows=2 if has_vol else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22] if has_vol else [1.0], vertical_spacing=0.03)

    if use_line:
        fig.add_trace(go.Scatter(x=idx, y=ohlc["Close"], name="NAV", mode="lines",
                                 line=dict(color=P["ink"], width=1.8),
                                 hovertemplate="NAV: %{y:.4f}<extra></extra>"), row=1, col=1)
    else:
        fig.add_trace(go.Candlestick(
            x=idx, open=ohlc["Open"], high=ohlc["High"], low=ohlc["Low"], close=ohlc["Close"],
            increasing_line_color=UP, increasing_fillcolor=UP,
            decreasing_line_color=DOWN, decreasing_fillcolor=DOWN, name="Price"), row=1, col=1)

    palette = P["dma"]
    for n in (30, 50, 124, 200):
        if show.get(f"{n} DMA"):
            s = r["dma_series"][n].loc[start:].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s, name=f"{n} DMA", mode="lines",
                                     line=dict(color=palette[n], width=1.6 if n == 200 else 1.2),
                                     hovertemplate=f"{n} DMA: %{{y:.2f}}<extra></extra>"), row=1, col=1)

    if show.get("CAR") and not monthly:
        c = r["car"].reindex(idx).dropna()
        fig.add_trace(go.Scatter(x=c.index, y=c, name="CAR", mode="lines",
                                 line=dict(color=P["car"], width=2.4),
                                 hovertemplate="CAR: %{y:.2f}<extra></extra>"), row=1, col=1)
        fig.add_shape(type="line", x0=r["high_date"], x1=r["high_date"], yref="paper", y0=0, y1=1,
                      line=dict(color=MUTED, width=1, dash="dash"))
        fig.add_annotation(x=r["high_date"], yref="paper", y=0.99, yanchor="top", xanchor="left", text=" 52W high", showarrow=False,
                           font=dict(size=10, color=MUTED))

    if show.get("Signals"):
        sg = r["signals"].loc[start:].dropna()
        sg = sg[sg != "HOLD"]
        for key, color, sym, nm in (("BUY", P["mk_buy"], "triangle-up", "BUY signal"),
                                    ("SELL", P["mk_sell"], "triangle-down", "SELL signal")):
            m = sg[sg.str.upper().str.contains(key)]
            m = m[m.index.isin(ohlc.index)]
            if not m.empty:
                fig.add_trace(go.Scatter(
                    x=m.index, y=ohlc.loc[m.index, "Low"] * (0.97 if monthly else 0.985), mode="markers", name=nm,
                    marker=dict(symbol=sym, size=13, color=color, line=dict(color=P["bg"], width=1)),
                    text=m.values, hovertemplate="%{text}<br>%{x|%d %b %Y}<extra></extra>"), row=1, col=1)

    if has_vol:
        vcol = np.where(ohlc["Close"] >= ohlc["Open"], UP, DOWN)
        fig.add_trace(go.Bar(x=idx, y=ohlc["Volume"], marker_color=vcol, opacity=0.55,
                             name="Volume", showlegend=False), row=2, col=1)

    fig.update_layout(
        height=640, template="plotly_dark" if P["mode"] == "dark" else "plotly_white",
        paper_bgcolor=P["bg"], plot_bgcolor=P["bg"], font=dict(color=P["ink"]), hovermode="x unified", margin=dict(l=10, r=10, t=105, b=85),
        legend=dict(orientation="h", x=0, y=-0.09, xanchor="left", yanchor="top"),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"{label or ticker.replace('.NS', '')} · " + ("NAV " if is_fund else "") + (
                   f"daily since 52W high ({pd.Timestamp(r['high_date']):%d %b %Y}, {len(idx)} sessions)"
                   if view == "HIGH" else "monthly" if monthly else "1Y daily"), x=0, xanchor="left", y=0.975, yanchor="top",
                   font=dict(size=16, color=P["ink"])),
        dragmode="pan")
    fig.update_xaxes(rangebreaks=[] if monthly else [dict(bounds=["sat", "mon"])], showspikes=True, spikethickness=1, gridcolor=P["grid"])
    fig.update_yaxes(gridcolor=P["grid"])
    if monthly:
        btns = [dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All")]
        fig.update_xaxes(tickformat="%b %Y", hoverformat="%b %Y")
    else:
        btns = [dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All" if view == "HIGH" else "1Y")]
    fig.update_xaxes(rangeselector=dict(x=0, y=1.0, xanchor="left", yanchor="bottom", buttons=btns, bgcolor=P["card"], activecolor=P["border"], font=dict(color=P["ink"])), row=1, col=1)
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
st.markdown('<div class="hero"><h1>Monthly Valuation Analyzer</h1>'
            '<p>Stocks, indices and mutual funds · Buy/Sell signals on the first trading day of each month · by Vikas Dhiman</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Analyze", ["Stock", "Index", "Mutual fund"], horizontal=True)
    fund_code, fund_name = None, ""
    ticker_in, index_name, index_group = "", "", ""
    if mode == "Stock":
        quick = st.selectbox("Quick pick", ["—", "RELIANCE", "TCS", "INFY", "HDFCBANK"])
        ticker_in = st.text_input("Stock symbol", value="RELIANCE" if quick == "—" else quick,
                                  help="Type the NSE symbol, for example RELIANCE. Indices are in the Index menu.")
    elif mode == "Index":
        choice = st.selectbox("Index", [n for _, n, _ in INDICES] + [CUSTOM_INDEX],
                              help="Type in the box to search the list.")
        if choice == CUSTOM_INDEX:
            ticker_in = st.text_input("Yahoo Finance ticker", placeholder="e.g. ^NSEI or BSE-500.BO")
            index_name, index_group = ticker_in.strip().upper(), "Custom"
        else:
            index_group, ticker_in = INDEX_MAP[choice]
            index_name = choice
    else:
        mf_q = st.text_input("Search fund", placeholder="e.g. parag parikh flexi cap",
                             help="Type part of the scheme name and press Enter.")
        growth_only = st.checkbox("Growth plans only", value=True,
                                  help="Hides IDCW / dividend plans, whose NAV drops on payouts and distorts the signals.")
        code_in = st.text_input("Or enter AMFI scheme code", placeholder="e.g. 122639")
        if code_in.strip().isdigit():
            fund_code, fund_name = int(code_in.strip()), ""
        elif len(mf_q.strip()) >= 3:
            search_failed = False
            try:
                found = mf_search(mf_q.strip())
            except Exception:
                found, search_failed = [], True
                st.error("Fund search is unavailable right now. Try again, or enter a scheme code.")
            if growth_only:
                found = [x for x in found if is_growth_plan(x["name"])] or found
            if found:
                names = {x["code"]: x["name"] for x in found[:100]}
                fund_code = st.selectbox("Fund", list(names), format_func=lambda c: names[c])
                fund_name = names[fund_code]
            elif not search_failed:
                st.caption("No funds found. Try fewer or different words.")
        else:
            st.caption("Type at least 3 letters of the fund name.")
    as_of = st.date_input("As-of date", value=date.today(), max_value=date.today(),
                          help="Weekends and holidays use the previous trading session.")
    st.subheader("Chart range")
    timeframe = st.radio("Timeframe", ["Daily", "Monthly"], horizontal=True,
                         help="Monthly shows one candle per month and does not use the CAR line.")
    monthly = timeframe == "Monthly"
    view, monthly_years = "1Y", 5
    if monthly:
        monthly_years = {"3 years": 3, "5 years": 5, "10 years": 10}[
            st.selectbox("Monthly history", ["3 years", "5 years", "10 years"], index=1)]
    else:
        view_label = st.radio("Show", ["Last 1 year", "From 52-week high"], label_visibility="collapsed",
                              help="'From 52-week high' starts the chart at the latest 52-week high candle "
                                   "(the same candle where the CAR line begins).")
        view = "HIGH" if view_label.startswith("From") else "1Y"
    st.subheader("Chart layers")
    show = {k: st.checkbox(k, value=v) for k, v in
            {"30 DMA": True, "50 DMA": True, "124 DMA": False, "200 DMA": True, "CAR": True, "Signals": True}.items()}
    run = st.button("Analyze", type="primary", use_container_width=True)

if "ran" not in st.session_state:
    st.session_state.ran = False
if run:
    st.session_state.ran = True

if not st.session_state.ran:
    st.info("Choose Stock, Index or Mutual fund in the sidebar, pick a date, then select **Analyze**.")
    st.stop()

is_fund = mode == "Mutual fund"
is_index = mode == "Index"
meta = None
if not is_fund:
    ticker = ticker_in.strip().upper()
    if not ticker:
        st.error("Enter an index ticker." if is_index else "Enter a stock symbol.")
        st.stop()
    if not is_index and not ticker.startswith("^") and "." not in ticker:
        ticker += ".NS"
    with st.spinner(f"Loading {index_name if is_index else ticker}…"):
        r, err = analyze_stock(ticker, as_of)
    if err and is_index and err.startswith("No data found"):
        err = (f"Yahoo Finance has no data for {index_name} ({ticker}) right now. "
               "Try another index, or choose Custom ticker… and enter a different symbol.")
else:
    if not fund_code:
        st.info("Search for a mutual fund in the sidebar and pick one from the list.")
        st.stop()
    ticker = str(fund_code)
    with st.spinner(f"Loading NAV history for scheme {fund_code}…"):
        r, meta, err = analyze_fund(fund_code, as_of)
if err:
    st.error(err)
    st.stop()

unit = "NAV" if is_fund else ("Index" if is_index else "CMP")
price_label = "NAV" if is_fund else ("Index level" if is_index else "Last price")
full_fund_name = ((meta or {}).get("scheme_name") or fund_name or f"Scheme {fund_code}") if is_fund else ""
if is_fund:
    chart_label = full_fund_name if len(full_fund_name) <= 58 else full_fund_name[:55] + "…"
elif is_index:
    chart_label = index_name
else:
    chart_label = None

sess = r["session"].strftime("%d-%m-%Y")
if sess != as_of.strftime("%d-%m-%Y"):
    st.info(f"{as_of.strftime('%d-%m-%Y')} was not a trading day. Using session {sess}.")

if is_fund:
    fund_header(meta or {}, fund_code, fund_name)
elif is_index:
    index_header(index_name, ticker, index_group)
else:
    company_header(ticker)
k = st.columns(4)
kpi(k[0], price_label, f"{r['cmp']:,.4f}" if is_fund else f"{r['cmp']:,.2f}", f"Session {sess}")
kpi(k[1], "Valuation signal", pill(r["signal"]), f"Based on {r['val_month']} ({r['days_used']} days)")
car_txt = r["car_status"] or "n/a"
kpi(k[2], "CAR status", car_txt, "Last 10 sessions" if r["car_status"] else "Under 10 days since 52W high")
sig_up = r["signal"].upper()
action_txt = "🟢 Buy" if "BUY" in sig_up else "🔴 Sell" if "SELL" in sig_up else "🟡 Hold"
kpi(k[3], "Action", action_txt, f"{r['signal']} · {r['dist200']:+.2f}% from 200 DMA")

st.write("")
tab1, tab2, tab3 = st.tabs(["Chart", "Levels & valuation", "Signal history"])

with tab1:
    cfg = {"displaylogo": False, "scrollZoom": True}
    if monthly:
        with st.spinner("Building monthly candles…"):
            rm, errm = get_monthly(is_fund, ticker, fund_code, as_of, monthly_years)
        if errm:
            st.warning(errm)
        else:
            st.plotly_chart(build_chart(rm, ticker, show, label=chart_label), use_container_width=True,
                            config=cfg, key="monthly_chart")
            st.caption("One candle per month; the latest candle covers the month to date. Moving averages are the "
                       "daily averages read at each month-end. Triangles mark the monthly signal. "
                       "CAR is not used on this chart.")
    else:
        st.plotly_chart(build_chart(r, ticker, show, label=chart_label, view=view), use_container_width=True,
                        config=cfg, key="main_chart")
        st.caption("Drag to pan, scroll to zoom, double-click to reset. Triangles mark the first trading day of a month."
                   + (" The chart starts at the latest 52-week high." if view == "HIGH" else ""))

with tab2:
    a, b = st.columns([1, 1.3])
    with a:
        st.markdown(f"**{unit} / 124 DMA now**")
        st.plotly_chart(ratio_gauge(r["ratio_now"]), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Previous-month average: **{r['ratio_prev']:.4f}**")
    with b:
        d = r["dma"]
        st.dataframe(pd.DataFrame({
            "Metric": ["30 DMA", "50 DMA", "124 DMA", "200 DMA", "Distance from 200 DMA",
                       f"Previous-month avg {unit} / 124 DMA", "Valuation month", "Trading days used"],
            "Value": [f"{d[30]:.2f}", f"{d[50]:.2f}", f"{d[124]:.2f}", f"{d[200]:.2f}",
                      f"{r['dist200']:.2f}%", f"{r['ratio_prev']:.4f}", r["val_month"], r["days_used"]]}),
            hide_index=True, use_container_width=True)

with tab3:
    sg = r["signals"].dropna().rename("Signal").to_frame()
    sg["NAV" if is_fund else "Close"] = r["ohlc"].loc[sg.index, "Close"].round(4 if is_fund else 2)
    sg.index = sg.index.strftime("%d-%m-%Y")
    sg.index.name = "First trading day"
    st.dataframe(sg.iloc[::-1], use_container_width=True)
    st.download_button("Download CSV", sg.to_csv().encode(), f"{ticker.lstrip('^')}_signals.csv", "text/csv")

st.caption(("NAV data via mfapi.in (AMFI)" if is_fund else "Data via Yahoo Finance") + " · For research only, not investment advice.")
