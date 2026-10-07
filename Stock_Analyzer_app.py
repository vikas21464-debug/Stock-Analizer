# ============================================================
# SINGLE STOCK BREAKOUT & MONTHLY VALUATION ANALYZER
# Streamlit Web Version (with Index support + Signal on Chart)
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
    page_title="Stock Breakout & Valuation Analyzer by Vikas Dhiman",
    page_icon="📈",
    layout="wide",
)

# ------------------------------------------------------------
# Valuation Signal (exact same formula)
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
# Candlestick Chart (now shows Buy/Sell signal)
# ------------------------------------------------------------
def plot_candlestick_chart(
    ohlc,
    ticker,
    as_of_label,
    dma_lines=None,
    car_series=None,
    high_date=None,
    valuation_signal=None,
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

    # ---------- NEW: Show only BUY / SELL on chart ----------
    if valuation_signal and valuation_signal != "HOLD":
        is_buy = "BUY" in valuation_signal.upper()
        box_color = "#2e7d32" if is_buy else "#c62828"   # Green for BUY, Red for Sell
        text_color = "white"

        ax.text(
            0.98, 0.97,
            f"Valuation: {valuation_signal}",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color=text_color,
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=box_color,
                edgecolor=box_color,
                alpha=0.92,
            ),
            zorder=10,
        )

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

    chart_ohlc = data[["Open",
