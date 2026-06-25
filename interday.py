# =========================================================
# ULTRA AI INTRADAY STOCK PREDICTOR 
# =========================================================

# FEATURES:
# ---------------------------------------------------------
# RSI
# EMA
# MACD
# VWAP
# Bollinger Bands
# ATR
# ADX
# Stochastic RSI
# Supertrend
# Support / Resistance
# Multi Timeframe Analysis
# XGBoost AI
# Auto Learning
# Dynamic Stoploss
# Smart Trend Detection
# =========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import joblib

from ta.momentum import (
    RSIIndicator,
    StochasticOscillator
)

from ta.trend import (
    EMAIndicator,
    MACD,
    ADXIndicator
)

from ta.volatility import (
    BollingerBands,
    AverageTrueRange
)

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================================
# INPUT
# =========================================================

print("\n========= ULTRA AI INTRADAY STOCK PREDICTOR =========\n")

print("Examples:")
print("RELIANCE")
print("TCS")
print("INFY")
print("HDFCBANK")
print("ICICIBANK")
print("SBIN")
print("SAIL")
print("PNB")

company = input("\nEnter NSE Stock Name: ").upper()

stock = company + ".NS"

# =========================================================
# SETTINGS
# =========================================================

BUY_PROBABILITY = 60

MODEL_FILE = f"{company}_xgb_model.pkl"

# =========================================================
# DOWNLOAD DATA
# =========================================================

print("\nDownloading Market Data...\n")

try:

    # 5 Minute Data
    df5 = yf.download(
        stock,
        period="30d",
        interval="5m",
        auto_adjust=True,
        progress=False
    )

    # 15 Minute Data
    # Yahoo supports max 60d
    df15 = yf.download(
        stock,
        period="60d",
        interval="15m",
        auto_adjust=True,
        progress=False
    )

    # 1 Hour Data
    df1h = yf.download(
        stock,
        period="730d",
        interval="1h",
        auto_adjust=True,
        progress=False
    )

except Exception as e:

    print("Download Error:", e)
    exit()

# =========================================================
# EMPTY DATA CHECK
# =========================================================

if df5.empty:
    print("\n5 Minute Data Not Found")
    exit()

if df15.empty:
    print("\n15 Minute Data Not Found")
    exit()

if df1h.empty:
    print("\n1 Hour Data Not Found")
    exit()

# =========================================================
# FIX COLUMNS
# =========================================================

df5.columns = df5.columns.get_level_values(0)
df15.columns = df15.columns.get_level_values(0)
df1h.columns = df1h.columns.get_level_values(0)

# =========================================================
# TIME
# =========================================================

try:

    latest_time_ist = df5.index[-1].tz_convert(
        "Asia/Kolkata"
    )

except:

    latest_time_ist = df5.index[-1]

print("\nSYSTEM TIME:", datetime.datetime.now())

print("LATEST CANDLE TIME:", latest_time_ist)

# =========================================================
# MAIN DATA
# =========================================================

close = df5["Close"].squeeze()

open_price = df5["Open"].squeeze()

high = df5["High"].squeeze()

low = df5["Low"].squeeze()

volume = df5["Volume"].squeeze()

# =========================================================
# RSI
# =========================================================

df5["RSI"] = RSIIndicator(
    close=close,
    window=14
).rsi()

# =========================================================
# EMA
# =========================================================

df5["EMA9"] = EMAIndicator(
    close=close,
    window=9
).ema_indicator()

df5["EMA21"] = EMAIndicator(
    close=close,
    window=21
).ema_indicator()

df5["EMA50"] = EMAIndicator(
    close=close,
    window=50
).ema_indicator()

# =========================================================
# MACD
# =========================================================

macd = MACD(close=close)

df5["MACD"] = macd.macd()

df5["MACD_SIGNAL"] = macd.macd_signal()

# =========================================================
# VWAP
# =========================================================

df5["VWAP"] = (
    (df5["Close"] * df5["Volume"]).cumsum()
    / df5["Volume"].cumsum()
)

# =========================================================
# BOLLINGER BANDS
# =========================================================

bb = BollingerBands(close=close)

df5["BB_HIGH"] = bb.bollinger_hband()

df5["BB_LOW"] = bb.bollinger_lband()

df5["BB_MID"] = bb.bollinger_mavg()

# =========================================================
# ATR
# =========================================================

atr = AverageTrueRange(
    high=high,
    low=low,
    close=close
)

df5["ATR"] = atr.average_true_range()

# =========================================================
# ADX
# =========================================================

adx = ADXIndicator(
    high=high,
    low=low,
    close=close
)

df5["ADX"] = adx.adx()

# =========================================================
# STOCHASTIC RSI
# =========================================================

stoch = StochasticOscillator(
    high=high,
    low=low,
    close=close
)

df5["STOCH_K"] = stoch.stoch()

df5["STOCH_D"] = stoch.stoch_signal()

# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

df5["SUPPORT"] = low.rolling(20).min()

df5["RESISTANCE"] = high.rolling(20).max()

# =========================================================
# SUPERTREND FUNCTION
# =========================================================

def calculate_supertrend(
    df,
    period=10,
    multiplier=3
):

    hl2 = (df["High"] + df["Low"]) / 2

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=period
    ).average_true_range()

    upperband = hl2 + (
        multiplier * atr
    )

    lowerband = hl2 - (
        multiplier * atr
    )

    supertrend = [True]

    for i in range(1, len(df)):

        close_current = df["Close"].iloc[i]

        if close_current > upperband.iloc[i - 1]:

            supertrend.append(True)

        elif close_current < lowerband.iloc[i - 1]:

            supertrend.append(False)

        else:

            supertrend.append(
                supertrend[i - 1]
            )

    return pd.Series(
        supertrend,
        index=df.index
    )

# =========================================================
# SUPERTREND
# =========================================================

df5["SUPERTREND"] = calculate_supertrend(df5)

# =========================================================
# EXTRA FEATURES
# =========================================================

df5["PRICE_CHANGE"] = close.pct_change()

df5["VOL_AVG"] = volume.rolling(20).mean()

df5["HIGH_LOW_DIFF"] = high - low

df5["OPEN_CLOSE_DIFF"] = close - open_price

# =========================================================
# TARGET
# =========================================================

df5["TARGET"] = np.where(
    close.shift(-1) > close,
    1,
    0
)

# =========================================================
# CLEAN DATA
# =========================================================

df5.dropna(inplace=True)

# =========================================================
# FEATURES
# =========================================================

FEATURES = [

    "RSI",

    "EMA9",
    "EMA21",
    "EMA50",

    "MACD",
    "MACD_SIGNAL",

    "VWAP",

    "BB_HIGH",
    "BB_LOW",
    "BB_MID",

    "ATR",

    "ADX",

    "STOCH_K",
    "STOCH_D",

    "SUPPORT",
    "RESISTANCE",

    "PRICE_CHANGE",

    "VOL_AVG",

    "HIGH_LOW_DIFF",

    "OPEN_CLOSE_DIFF"
]

X = df5[FEATURES]

y = df5["TARGET"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# =========================================================
# AUTO LEARNING
# =========================================================

print("\nAUTO LEARNING SYSTEM ACTIVE")

if os.path.exists(MODEL_FILE):

    print("Loading Existing AI Model...")

    model = joblib.load(MODEL_FILE)

else:

    print("Training New AI Model...")

    model = XGBClassifier(

        n_estimators=400,

        learning_rate=0.03,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        MODEL_FILE
    )

# =========================================================
# DAILY RETRAIN
# =========================================================

current_hour = datetime.datetime.now().hour

if current_hour < 10:

    print("\nMorning Retraining Started...")

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("AI Retrained Successfully")

# =========================================================
# MODEL ACCURACY
# =========================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

# =========================================================
# LATEST PREDICTION
# =========================================================

latest_data = X.iloc[-1:]

prediction = model.predict(
    latest_data
)[0]

probabilities = model.predict_proba(
    latest_data
)[0]

up_probability = probabilities[1] * 100

down_probability = probabilities[0] * 100

# =========================================================
# MULTI TIMEFRAME ANALYSIS
# =========================================================

# -------------------------
# 15 MINUTE
# -------------------------

df15.dropna(inplace=True)

df15["EMA20"] = EMAIndicator(
    close=df15["Close"],
    window=20
).ema_indicator()

df15["EMA50"] = EMAIndicator(
    close=df15["Close"],
    window=50
).ema_indicator()

df15.dropna(inplace=True)

trend_15m = "SIDEWAYS"

if len(df15) > 0:

    if (
        df15["EMA20"].iloc[-1]
        >
        df15["EMA50"].iloc[-1]
    ):

        trend_15m = "BULLISH"

    else:

        trend_15m = "BEARISH"

# -------------------------
# 1 HOUR
# -------------------------

df1h.dropna(inplace=True)

df1h["EMA20"] = EMAIndicator(
    close=df1h["Close"],
    window=20
).ema_indicator()

df1h["EMA50"] = EMAIndicator(
    close=df1h["Close"],
    window=50
).ema_indicator()

df1h.dropna(inplace=True)

trend_1h = "SIDEWAYS"

if len(df1h) > 0:

    if (
        df1h["EMA20"].iloc[-1]
        >
        df1h["EMA50"].iloc[-1]
    ):

        trend_1h = "BULLISH"

    else:

        trend_1h = "BEARISH"

# =========================================================
# CURRENT VALUES
# =========================================================

current_price = float(
    close.iloc[-1]
)

current_open = float(
    open_price.iloc[-1]
)

latest_rsi = float(
    df5["RSI"].iloc[-1]
)

latest_adx = float(
    df5["ADX"].iloc[-1]
)

latest_atr = float(
    df5["ATR"].iloc[-1]
)

latest_vwap = float(
    df5["VWAP"].iloc[-1]
)

latest_support = float(
    df5["SUPPORT"].iloc[-1]
)

latest_resistance = float(
    df5["RESISTANCE"].iloc[-1]
)

# =========================================================
# CANDLE ANALYSIS
# =========================================================

if current_price > current_open:

    candle_signal = "BULLISH CANDLE"

else:

    candle_signal = "BEARISH CANDLE"

# =========================================================
# VWAP SIGNAL
# =========================================================

if current_price > latest_vwap:

    vwap_signal = "BULLISH"

else:

    vwap_signal = "BEARISH"

# =========================================================
# SUPERTREND SIGNAL
# =========================================================

if df5["SUPERTREND"].iloc[-1]:

    supertrend_signal = "BULLISH"

else:

    supertrend_signal = "BEARISH"

# =========================================================
# FINAL SIGNAL
# =========================================================

signal = "HOLD"

# STRONG BUY

if (

    prediction == 1

    and up_probability > BUY_PROBABILITY

    and trend_15m == "BULLISH"

    and trend_1h == "BULLISH"

    and current_price > latest_vwap

    and latest_rsi > 50

    and latest_adx > 20

    and supertrend_signal == "BULLISH"

):

    signal = "STRONG BUY"

# STRONG SELL

elif (

    prediction == 0

    and down_probability > BUY_PROBABILITY

    and trend_15m == "BEARISH"

    and trend_1h == "BEARISH"

    and current_price < latest_vwap

    and latest_rsi < 50

    and latest_adx > 20

    and supertrend_signal == "BEARISH"

):

    signal = "STRONG SELL"

elif up_probability > down_probability:

    signal = "BUY"

else:

    signal = "SELL"

# =========================================================
# ATR STOPLOSS
# =========================================================

if "BUY" in signal:

    target = current_price + (
        latest_atr * 2
    )

    stoploss = current_price - latest_atr

else:

    target = current_price - (
        latest_atr * 2
    )

    stoploss = current_price + latest_atr

# =========================================================
# OUTPUT
# =========================================================

print("\n========== AI ANALYSIS ==========")

print("STOCK:", stock)

print("CURRENT PRICE:", round(current_price, 2))

# =========================================================
# AI PROBABILITY
# =========================================================

print("\n========== AI PROBABILITY ==========")

print(
    f"UP PROBABILITY: "
    f"{up_probability:.2f}%"
)

print(
    f"DOWN PROBABILITY: "
    f"{down_probability:.2f}%"
)

# =========================================================
# INDICATORS
# =========================================================

print("\n========== INDICATORS ==========")

print("RSI:", round(latest_rsi, 2))

print("ADX:", round(latest_adx, 2))

print("ATR:", round(latest_atr, 2))

print("VWAP:", round(latest_vwap, 2))

print(
    "SUPPORT:",
    round(latest_support, 2)
)

print(
    "RESISTANCE:",
    round(latest_resistance, 2)
)

# =========================================================
# MULTI TIMEFRAME
# =========================================================

print("\n========== MULTI TIMEFRAME ==========")

print("15 MIN TREND:", trend_15m)

print("1 HOUR TREND:", trend_1h)

# =========================================================
# MARKET SIGNALS
# =========================================================

print("\n========== MARKET SIGNALS ==========")

print("VWAP SIGNAL:", vwap_signal)

print("SUPERTREND:", supertrend_signal)

print("CANDLE:", candle_signal)

# =========================================================
# FINAL SIGNAL
# =========================================================

print("\n========== FINAL SIGNAL ==========")

print("SIGNAL:", signal)

# =========================================================
# TRADE
# =========================================================

print("\n========== TRADE ==========")

print(
    f"ENTRY PRICE: "
    f"{current_price:.2f}"
)

print(
    f"TARGET: "
    f"{target:.2f}"
)

print(
    f"STOPLOSS: "
    f"{stoploss:.2f}"
)

# =========================================================
# ACCURACY
# =========================================================

print("\n========== MODEL ACCURACY ==========")

print(
    round(accuracy * 100, 2),
    "%"
)

# =========================================================
# MARKET CONDITIONS
# =========================================================

print("\n========== MARKET CONDITIONS ==========")

# RSI STATUS

if latest_rsi > 70:

    print("RSI STATUS: OVERBOUGHT")

elif latest_rsi < 30:

    print("RSI STATUS: OVERSOLD")

else:

    print("RSI STATUS: NORMAL")


if latest_adx > 25:

    print("TREND STRENGTH: STRONG")

else:

    print("TREND STRENGTH: WEAK")
latest_volume = volume.iloc[-1]

average_volume = df5["VOL_AVG"].iloc[-1]

if latest_volume > average_volume:

    print("VOLUME STATUS: HIGH")

else:

    print("VOLUME STATUS: LOW")


if current_price <= latest_support:

    print("PRICE NEAR SUPPORT")

if current_price >= latest_resistance:

    print("PRICE NEAR RESISTANCE")


bb_high = df5["BB_HIGH"].iloc[-1]

bb_low = df5["BB_LOW"].iloc[-1]

if current_price > bb_high:

    print("PRICE ABOVE BOLLINGER BAND")

elif current_price < bb_low:

    print("PRICE BELOW BOLLINGER BAND")

else:

    print("PRICE INSIDE BOLLINGER BAND")

print("\n========== END ==========\n")
