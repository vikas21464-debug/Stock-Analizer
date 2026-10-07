# ============================================================
# SINGLE STOCK BREAKOUT & MONTHLY VALUATION ANALYZER
# Streamlit Web Version
# - Index support
# - Historical Buy/Sell signals on the first trading day of each month
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="Stock Breakout & Monthly Valuation Analyzer by Vikas Dhiman",
    page_icon="📈",
    layout="wide",
)

# ------------------------------------------------------------
# Valuation Signal
# ------------------------------------------------------------
def get_valuation_signal(ratio):
    if ratio <= 0.55:
        return "5th BUY"
    elif ratio <= 0.64:
        return "4th BUY"
    elif ratio <= 0.73:
        return "3rd BUY"
    elif ratio <= 0.82:
        return "2nd BUY"
    elif ratio <= 0.91:
        return "1st BUY"
    elif ratio >= 1.45:
        return "5th Sell"
    elif ratio >= 1.36:
        return "4th Sell"
    elif ratio >= 1.27:
        return "3rd Sell"
    elif ratio >= 1.18:
        return "2nd Sell"
    elif ratio >= 1.09:
        return "1st Sell"
    else:
        return "HOLD"


# ------------------------------------------------------------
# Candlestick Chart with historical Buy/Sell markers
# ------------------------------------------------------------
def plot_candlestick_chart(
    ohlc,
    ticker,
    as_of_label,
    dma_lines=None,
    car_series=None,
    high_date=None,
    signal_series=None,
):
    if ohlc is None or ohlc.empty:
        st.warning("Not enough data to draw the candlestick chart.")
        return

    ohlc = ohlc.copy().dropna(subset=["Open", "High", "Low", "Close"])
    if ohlc.empty:
        st.warning("Not enough data to draw the candlestick chart.")
        return

    dates = ohlc.index
    opens = ohlc["Open"].values
    highs = ohlc["High"].values
    lows = ohlc["Low"].values
    closes = ohlc["Close"].values

    fig, ax = plt.subplots(figsize=(14, 6))

    if len(dates) > 1:
        avg_gap = np.median(np.diff(mdates.date2num(dates)))
        width = avg_gap * 0.6
    else:
        width = 0.6

    # Candles
    for i, dt in enumerate(dates):
        x = mdates.date2num(dt)
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color = "#26a69a" if c >= o else "#ef5350"

        ax.plot([x, x], [l, h], color=color, linewidth=0.8, solid_capstyle="round")

        bottom = min(o, c)
        height = abs(c - o)
        if height < 1e-9:
            height = (h - l) * 0.01 or 0.01
        ax.add_patch(
            Rectangle(
                (x - width / 2, bottom),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
            )
        )

    # DMAs
    if dma_lines:
        styles = {
            "30 DMA": ("#1976d2", 1.2),
            "50 DMA": ("#f9a825", 1.2),
            "124 DMA": ("#8e24aa", 1.0),
            "200 DMA": ("#e65100", 1.4),
        }
        for label, series in dma_lines.items():
            if series is None or series.dropna().empty:
                continue
            aligned = series.reindex(dates).dropna()
            if aligned.empty:
                continue
            color, lw = styles.get(label, ("#607d8b", 1.0))
            ax.plot(
                mdates.date2num(aligned.index),
                aligned.values,
                label=label,
                color=color,
                linewidth=lw,
                alpha=0.9,
            )

    # CAR line
    if car_series is not None and not car_series.dropna().empty:
        car_aligned = car_series.reindex(dates).dropna()
        if not car_aligned.empty:
            ax.plot(
                mdates.date2num(car_aligned.index),
                car_aligned.values,
                label="CAR",
                color="#000000",
                linewidth=2.0,
                linestyle="-",
                alpha=0.95,
                zorder=5,
            )

    # 52-week high marker
    if high_date is not None:
        try:
            hd = pd.Timestamp(high_date)
            if dates[0] <= hd <= dates[-1]:
                ax.axvline(
                    mdates.date2num(hd),
                    color="#455a64",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label="52W High (CAR start)",
                )
        except Exception:
            pass

    # ---------- Historical Buy / Sell markers on the first trading day of each month ----------
    if signal_series is not None and not signal_series.empty:
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []

        for i, dt in enumerate(dates):
            sig = signal_series.get(dt, "HOLD")
            if sig == "HOLD" or pd.isna(sig):
                continue

            x = mdates.date2num(dt)
            y = lows[i] * 0.987          # slightly below the candle low

            if "BUY" in str(sig).upper():
                buy_x.append(x)
                buy_y.append(y)
            elif "SELL" in str(sig).upper():
                sell_x.append(x)
                sell_y.append(y)

        if buy_x:
            ax.scatter(buy_x, buy_y, marker="^", color="#2e7d32", s=50, zorder=6, label="Monthly Valuation BUY Signal")
        if sell_x:
            ax.scatter(sell_x, sell_y, marker="v", color="#c62828", s=50, zorder=6, label="Monthly Valuation SELL Signal")

    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        f"{ticker.replace('.NS', '')} — 1Y Daily Candlestick (as of {as_of_label})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(
        mdates.date2num(dates[0]) - width,
        mdates.date2num(dates[-1]) + width,
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------
# Core Analysis Function
# ------------------------------------------------------------
def analyze_stock(ticker, as_of_date=None):
    if as_of_date is None:
        as_of_date = date.today()

    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.date()

    start = as_of_date - timedelta(days=900)
    end = as_of_date + timedelta(days=1)

    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        return None, "No data found for this stock/index."

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.sort_index()
    data = data.loc[data.index.normalize() <= pd.Timestamp(as_of_date)]

    if data.empty:
        return None, "No trading data on or before the selected date."

    if len(data) < 200:
        return None, "Not enough historical data to calculate 200 DMA as of the selected date."

    close_prices = data["Close"].squeeze()
    if close_prices.empty:
        return None, "No closing-price data available."

    as_of_session = close_prices.index[-1]
    cmp = float(close_prices.iloc[-1])

    dma_30_series = close_prices.rolling(30).mean()
    dma_50_series = close_prices.rolling(50).mean()
    dma_124_series = close_prices.rolling(124).mean()
    dma_200_series = close_prices.rolling(200).mean()

    dma_30 = float(dma_30_series.iloc[-1])
    dma_50 = float(dma_50_series.iloc[-1])
    dma_124 = float(dma_124_series.iloc[-1])
    dma_200 = float(dma_200_series.iloc[-1])

    chart_ohlc = data[["Open", "High", "Low", "Close"]].tail(252).copy()
    chart_dmas = {
        "30 DMA": dma_30_series.reindex(chart_ohlc.index),
        "50 DMA": dma_50_series.reindex(chart_ohlc.index),
        "124 DMA": dma_124_series.reindex(chart_ohlc.index),
        "200 DMA": dma_200_series.reindex(chart_ohlc.index),
    }

    # ---------- Current Valuation Signal ----------
    current_year = as_of_session.year
    current_month = as_of_session.month

    if current_month == 1:
        previous_year = current_year - 1
        previous_month = 12
    else:
        previous_year = current_year
        previous_month = current_month - 1

    daily_ratio = close_prices / dma_124_series

    previous_month_mask = (
        (daily_ratio.index.year == previous_year)
        & (daily_ratio.index.month == previous_month)
    )
    previous_month_ratios = daily_ratio.loc[previous_month_mask].dropna()

    if previous_month_ratios.empty:
        return None, "Unable to calculate the previous month's average CMP / 124 DMA."

    monthly_average_ratio = float(previous_month_ratios.mean())
    valuation_signal = get_valuation_signal(monthly_average_ratio)

    # ---------- Historical signal restricted strictly to the first trading day of each month ----------
    signal_series = pd.Series(index=close_prices.index, dtype=object)
    signal_series[:] = "HOLD"

    temp_df = pd.DataFrame({'ratio': daily_ratio})
    temp_df['Year'] = temp_df.index.year
    temp_df['Month'] = temp_df.index.month

    first_days_of_months = temp_df.groupby(['Year', 'Month']).head(1).index

    for dt in first_days_of_months:
        yr = dt.year
        mo = dt.month
        if mo == 1:
            prev_yr = yr - 1
            prev_mo = 12
        else:
            prev_yr = yr
            prev_mo = mo - 1

        mask = (daily_ratio.index.year == prev_yr) & (daily_ratio.index.month == prev_mo)
        prev_ratios = daily_ratio.loc[mask].dropna()

        if len(prev_ratios) >= 5:
            avg = float(prev_ratios.mean())
            signal_series.loc[dt] = get_valuation_signal(avg)

    # Align to chart period
    chart_signals = signal_series.reindex(chart_ohlc.index)

    # CAR
    last_1y_data = data.tail(252)
    high_series = last_1y_data["High"].squeeze()
    high_date = high_series.idxmax()
    car_data = close_prices.loc[high_date:]
    car_series = car_data.expanding().mean()

    if len(car_data) < 10:
        return {
            "mode": "valuation_only",
            "Date": as_of_session.strftime("%d-%m-%Y"),
            "Selected Date": as_of_date.strftime("%d-%m-%Y"),
            "Stock": ticker.replace(".NS", ""),
            "Ticker": ticker,
            "Current CMP / 124 DMA": round(cmp / dma_124, 4),
            "Previous Month Avg CMP / 124 DMA": round(monthly_average_ratio, 4),
            "Valuation Signal": valuation_signal,
            "Valuation Month": f"{previous_year}-{previous_month:02d}",
            "Trading Days Used": len(previous_month_ratios),
            "chart_ohlc": chart_ohlc,
            "chart_dmas": chart_dmas,
            "chart_car": car_series,
            "high_date": high_date,
            "chart_signals": chart_signals,
        }, None

    last_10_car = car_series.tail(10)
    car_status = "Positive" if last_10_car.is_monotonic_increasing else "Negative"
    dist_200_dma = ((cmp - dma_200) / dma_200) * 100

    breakout = (
        (cmp > dma_30)
        and (cmp > dma_50)
        and (cmp > dma_200)
        and (car_status == "Positive")
    )
    action = "🟢 Positive Breakout" if breakout else "🔴 Avoid/Hold"

    return {
        "mode": "full",
        "Date": as_of_session.strftime("%d-%m-%Y"),
        "Selected Date": as_of_date.strftime("%d-%m-%Y"),
        "Stock": ticker.replace(".NS", ""),
        "Ticker": ticker,
        "CMP": round(cmp, 2),
        "30 DMA": round(dma_30, 2),
        "50 DMA": round(dma_50, 2),
        "124 DMA": round(dma_124, 2),
        "Current CMP / 124 DMA": round(cmp / dma_124, 4),
        "Previous Month Avg CMP / 124 DMA": round(monthly_average_ratio, 4),
        "Valuation Signal": valuation_signal,
        "200 DMA": round(dma_200, 2),
        "200 DMA Dist %": round(dist_200_dma, 2),
        "CAR Status": car_status,
        "Action": action,
        "Valuation Month": f"{previous_year}-{previous_month:02d}",
        "Trading Days Used": len(previous_month_ratios),
        "chart_ohlc": chart_ohlc,
        "chart_dmas": chart_dmas,
        "chart_car": car_series,
        "high_date": high_date,
        "chart_signals": chart_signals,
    }, None


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("📈 Stock Breakout & Monthly Valuation Analyzer by Vikas Dhiman")
st.caption("Supports Stocks + Indices • Monthly Valuation signals plotted on the first trading day of each month")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    stock_input = st.text_input(
        "Stock / Index Ticker",
        value="RELIANCE",
        placeholder="e.g. RELIANCE, TCS, ^NSEI, ^NSEBANK",
        help="For stocks just type name. For indices use ^NSEI, ^NSEBANK, ^BSESN",
    )

with col2:
    as_of = st.date_input(
        "As-of Date",
        value=date.today(),
        max_value=date.today(),
        help="Weekends/holidays will use the previous trading session.",
    )

with col3:
    st.write("")
    st.write("")
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

st.divider()

if analyze_btn:
    ticker = stock_input.strip().upper()

    if not ticker:
        st.error("Please enter a stock or index name.")
    else:
        if not ticker.startswith("^") and not ticker.endswith(".NS"):
            ticker = ticker + ".NS"

        with st.spinner(f"Analyzing {ticker} as of {as_of.strftime('%d-%m-%Y')} ..."):
            result, error = analyze_stock(ticker, as_of_date=as_of)

        if error:
            st.error(error)
        else:
            session_note = ""
            if result["Date"] != result["Selected Date"]:
                session_note = (
                    f"ℹ️ Selected {result['Selected Date']} was not a trading day. "
                    f"Using session {result['Date']}."
                )

            if result["mode"] == "valuation_only":
                st.subheader("📊 Valuation Signal")
                st.info(
                    "CAR status not shown because fewer than 10 trading days "
                    "have passed since the 52-week high. CAR line is still plotted."
                )
                if session_note:
                    st.info(session_note)

                c1, c2, c3 = st.columns(3)
                c1.metric("As-of Session", result["Date"])
                c2.metric("Symbol", result["Stock"])
                c3.metric("Valuation Signal", result["Valuation Signal"])

                st.write(f"**Current CMP / 124 DMA:** {result['Current CMP / 124 DMA']:.4f}")
                st.write(
                    f"**Previous Month Avg CMP / 124 DMA:** "
                    f"{result['Previous Month Avg CMP / 124 DMA']:.4f}"
                )
                st.caption(
                    f"Valuation month used: {result['Valuation Month']} "
                    f"| Trading days used: {result['Trading Days Used']}"
                )
            else:
                st.subheader("📊 Analysis Result")
                if session_note:
                    st.info(session_note)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("CMP", f"{result['CMP']:.2f}")
                m2.metric("Valuation Signal", result["Valuation Signal"])
                m3.metric("CAR Status", result["CAR Status"])
                m4.metric("Action", result["Action"])

                st.markdown("#### Key Levels")
                details = {
                    "Metric": [
                        "As-of Session",
                        "30 DMA",
                        "50 DMA",
                        "124 DMA",
                        "200 DMA",
                        "Current CMP / 124 DMA",
                        "Previous Month Avg CMP / 124 DMA",
                        "200 DMA Distance %",
                        "Valuation Month",
                        "Trading Days Used",
                    ],
                    "Value": [
                        result["Date"],
                        f"{result['30 DMA']:.2f}",
                        f"{result['50 DMA']:.2f}",
                        f"{result['124 DMA']:.2f}",
                        f"{result['200 DMA']:.2f}",
                        f"{result['Current CMP / 124 DMA']:.4f}",
                        f"{result['Previous Month Avg CMP / 124 DMA']:.4f}",
                        f"{result['200 DMA Dist %']:.2f}%",
                        result["Valuation Month"],
                        result["Trading Days Used"],
                    ],
                }
                st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)

                st.caption(
                    "📌 Valuation Signal is based on the average of every trading day's "
                    "CMP / 124 DMA from the **previous month** relative to the as-of date."
                )

            # Chart
            st.markdown("### 📈 1-Year Daily Candlestick Chart")
            st.caption(
                "Green ▲ = Monthly Valuation BUY Signal (first trading day of month) • Red ▼ = SELL Signal • Other days have no markers"
            )
            plot_candlestick_chart(
                result.get("chart_ohlc"),
                result.get("Ticker", ticker),
                result["Date"],
                dma_lines=result.get("chart_dmas"),
                car_series=result.get("chart_car"),
                high_date=result.get("high_date"),
                signal_series=result.get("chart_signals"),
            )

st.divider()
st.caption("Supports Stocks + Indices • Monthly Valuation signals on first trading day • Data via Yahoo Finance")
