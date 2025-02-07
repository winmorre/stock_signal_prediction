import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD, SMAIndicator, EMAIndicator, WMAIndicator, AroonIndicator
import pandas as pd
from meta_heuristic_algo import mfo2
from meta_heuristic_algo.pso import PSO
from meta_heuristic_algo.pso_gpt import PSOGpt, pso_gpt
from sklearn.model_selection import train_test_split
from oversampling import oversample_with_mahakel

"""
A raising Moving average indicates the security is an uptrend
a declining moving average indicates it is a downtrend

With MACD, when it is positive then the short-term moving average is located above the long-term moving average
and is an indication of upward momentum. 

bull market is when the prices are rising or expected to rise
and bear market is when the prices are falling or expected to fall

in MACD, the golden cross is when the short-term moving average breaks above its long-term moving average.
This indicates a bull market


THe target output is calculated for [1,0] to represent the close price in the day t+1 will increase compared to the 
close price in the day t. 
A value [0,1] represents that the close price in the day t+1 will decrease compared to the closing price in the day t.

Window size can be 5,10,15,20,25,30,35

Calculate the target variable
- Calculate the absolute change in price converted to pips by multiplying by 10,000 
(since 1pip = 0.0001 in most currency pairs)
- Use a lambda function to apply the condition that if the price change in pip is greater than 20, it assigns 1 otherwise 0
- Clean up any NaN values that result from the difference calculation of shifting the signal

The target label is created by classifying the 5-day and 3-day average of percentage change of historical 
price changes as either positive (1) or negative (0) in most cases. Other forms of target labels are also considered.

Best solution = [1 0 1 1 1 0 1 1 1 1 0 1 0 1 1 0 0 1 1 1 1 0 0 1 1 1 1 0 0 1 0 0 1 0 1 0 0
 0 1 0 0 0 1 0 1 0 1 0 1 0 0 0 1 1 1 1 1 1 1 1 0 0 0 1 0 0 0 1 1 1 1 1 1 0
 1 0 0 1 1 0 1 1 1] 
Best indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 38, 39, 40, 43, 44, 47, 48, 49, 50, 51, 52, 53, 54, 59, 60, 61, 62, 63, 64, 65, 66, 67, 70, 77, 82}
"""

WINDOW1 = 3
WINDOW2 = 5
WINDOW3 = 8
WINDOW4 = 14
WINDOW5 = 21
WINDOW6 = 26

PIP = 10000
PIP_THRESHOLD = 20 * 0.0001
ADX_THRESHOLD = 25
WINDOWS = [WINDOW1, WINDOW2, WINDOW3, WINDOW4, WINDOW5, WINDOW6]


def create_target(x, classes, st_dev):
    """
    Returns the target label based on the value of x, the number of classes to be generated, and the standard deviation factor.

    Args:
    x (float): A numeric value representing the price movement.
    classes (int): An integer representing the number of target classes to be generated. Valid options are 2, 3, or 5.
    st_dev (float): A numeric value representing the standard deviation factor.

    Returns:
    int: The target label. The value of the label depends on the number of classes specified. If classes is 2, the function returns 1 if x > 0, and 0 otherwise. If classes is 3, the function returns 0 if x < -st_dev, 1 if -st_dev < x < st_dev, and 2 otherwise. If classes is 5, the function returns 0 if x < -2*st_dev, 1 if -2*st_dev < x < -st_dev, 2 if -st_dev < x < st_dev, 3 if st_dev < x < 2*st_dev, and 4 otherwise.
    """

    if classes == 2:
        if abs(x * PIP) > 20:
            return 1
        else:
            return 0

    if classes == 3:
        DEV_FACTOR = 0.7

        st_dev = st_dev * DEV_FACTOR
        if x < -st_dev:
            return 0
        elif st_dev > x > - st_dev:
            return 1
        else:
            return 2

    if classes == 5:
        DEV_FACTOR = 0.7
        st_dev = st_dev * DEV_FACTOR
        if x < -st_dev * 2:
            return 0
        elif -st_dev > x > -st_dev * 2:
            return 1
        elif st_dev > x > -st_dev:
            return 2
        elif st_dev < x < st_dev * 2:
            return 3
        else:
            return 4


def get_data():
    data = yf.download("^GSPC", start="2010-06-01", end="2024-06-01", rounding=True)
    data.round(2)
    data[["Open", "High", "Low", "Close", "Adj Close"]] = data[["Open", "High", "Low", "Close", "Adj Close"]] / (
            10 ** 4)
    return data


def compute_target_variable(dataset: pd.DataFrame):
    # we assume 1pip = 0.01
    dataset["target"] = dataset["Close"].pct_change().rolling(3).mean().shift(3).round(6)
    dataset["target"] = dataset["target"].fillna(dataset["target"].mean().round(6))
    dataset["Target"] = dataset["target"].shift(-1).apply(
        lambda x: create_target(x, classes=2, st_dev=dataset["target"].std()))
    dataset.drop("target", axis=1, inplace=True)
    return dataset


def compute_adx(data):
    for w in WINDOWS:
        adx_indicator = ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], fillna=False, window=w)
        data[f"{w}_adx"] = adx_indicator.adx().round(6)
        data[f"{w}_pos_di"] = adx_indicator.adx_pos().round(6)
        data[f"{w}_neg_di"] = adx_indicator.adx_neg().round(6)
    for w in WINDOWS:
        for i in range(1, len(data)):
            if data[f"{w}_pos_di"].iloc[i] >= data[f"{w}_neg_di"].iloc[i]:
                if data[f"{w}_adx"].iloc[i] > ADX_THRESHOLD:
                    data.at[data.index[i], f"{w}_ADX_Buy_Sell_Signal"] = 1  # for buy
                else:
                    data.at[data.index[i], f"{w}_ADX_Buy_Sell_Signal"] = 0
            else:
                if data[f"{w}_adx"].iloc[i] < ADX_THRESHOLD:
                    data.at[data.index[i], f"{w}_ADX_Buy_Sell_Signal"] = 0  # for buy
                else:
                    data.at[data.index[i], f"{w}_ADX_Buy_Sell_Signal"] = 1
        data[f"{w}_ADX_Buy_Sell_Signal"] = data[f"{w}_ADX_Buy_Sell_Signal"].fillna(0)
    return data


def compute_macd(data):
    macd_w1 = MACD(close=data["Close"], fillna=True)
    macd_6_19 = MACD(close=data["Close"], window_fast=6, window_slow=19, fillna=True)
    macd_19_39 = MACD(close=data["Close"], window_fast=19, window_slow=39, fillna=True)
    data["MACD_12_26"] = macd_w1.macd().round(6)
    data["MACD_6_19"] = macd_6_19.macd().round(6)
    data["MACD_19_39"] = macd_19_39.macd().round(6)

    data["MACD_Signal_12_26"] = macd_w1.macd_signal().round(6)
    data["MACD_Signal_6_19"] = macd_6_19.macd_signal().round(6)
    data["MACD_Signal_19_39"] = macd_19_39.macd_signal().round(6)

    data["MACD_diff_12_26"] = macd_w1.macd_diff().round(6)
    data["MACD_Diff_6_19"] = macd_6_19.macd_diff().round(6)
    data["MACD_diff_19_39"] = macd_19_39.macd_diff().round(6)

    data["MACD_Signal_12_26"] = data["MACD_Signal_12_26"].fillna(data["MACD_Signal_12_26"].mean().round(6))
    data["MACD_diff_12_26"] = data["MACD_diff_12_26"].fillna(data["MACD_diff_12_26"].mean().round(6))
    data["MACD_Signal_19_39"] = data["MACD_Signal_19_39"].fillna(data["MACD_Signal_19_39"].mean().round(6))
    data["MACD_diff_19_39"] = data["MACD_diff_19_39"].fillna(data["MACD_diff_19_39"].mean().round(6))
    data["MACD_12_26"] = data["MACD_12_26"].fillna(data["MACD_12_26"].mean().round(6))

    return data


def compute_ma(data):
    for w in WINDOWS:
        sma = SMAIndicator(close=data["Close"], window=w, fillna=True)
        ema = EMAIndicator(close=data["Close"], window=w, fillna=True)
        wma = WMAIndicator(close=data["Close"], window=w, fillna=True)
        data[f"{w}_sma"] = sma.sma_indicator().round(6)
        data[f"{w}_ema"] = ema.ema_indicator().round(6)
        data[f"{w}_wma"] = wma.wma().round(6)

    return data


def compute_arron(df: pd.DataFrame):
    for w in WINDOWS:
        aroon = AroonIndicator(high=df["High"], low=df["Low"], window=w, fillna=True)
        df[f"{w}_AROON_UP"] = aroon.aroon_up().round(6)
        df[f"{w}_AROON_DOWN"] = aroon.aroon_down().round(6)
        df[f"{w}_AROON"] = aroon.aroon_indicator().round(6)
    return df


def compute_rsi(df: pd.DataFrame):
    for w in WINDOWS:
        rsi = RSIIndicator(close=df["Close"], window=w, fillna=True)
        df[f"{w}_RSI"] = rsi.rsi().round(6)
    return df


def aroon_sma_strategy(df: pd.DataFrame):
    for i in range(1, len(df)):
        for w in WINDOWS:
            df[f"{w}_SMA_AROON_STR"] = 0
            if df[f"{w}_AROON_UP"].iloc[-1] > 50 >= df[f"{w}_AROON_UP"].iloc[-2] and df["Close"].iloc[-1] > \
                    df[f"{w}_sma"].iloc[-1]:
                df.at[df.index[i], f"{w}_SMA_AROON_STR"] = 1
            elif df[f"{w}_AROON_DOWN"].iloc[-1] > 50 >= df[f"{w}_AROON_DOWN"].iloc[-2] and df["Close"].iloc[-1] < \
                    df[f"{w}_sma"].iloc[-1]:
                df.at[df.index[i], f"{w}_SMA_AROON_STR"] = -1
    return df


def compute_indicators() -> pd.DataFrame:
    data = get_data()
    data = compute_target_variable(dataset=data)
    data = compute_adx(data)
    data = compute_macd(data)
    data = compute_ma(data)
    data = compute_rsi(data)
    data = compute_arron(data)

    return data


def generate_features():
    data = compute_indicators()
    data = pd.read_csv("./balanced_all_features.csv")
    data.drop(columns=["Volume"], axis=1, inplace=True)
    data.to_csv("all_features.csv")

    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]
    close_ = data["Close"]
    date_ = data["Date"]
    X = data.drop(["Target", "Open", "High", "Low", "Close", "Date"], axis=1)
    y = data["Target"]

    mfo_best_solution = mfo2.mfo(X, y, n_features=30, max_iter=50)
    pso_best_features = pso_gpt(n_particles=30, xdata=X, ydata=y, max_iter=50)

    print("PSO BEST FEATURES:  ", pso_best_features, "\n")

    mfo_selected_features = [i for i in range(len(mfo_best_solution)) if mfo_best_solution[i] == 1]
    x_data = X.iloc[:, mfo_selected_features]
    x_data.insert(1, "Date", date_, True)
    x_data.insert(2, "Open", open_, True)
    x_data.insert(3, "High", high, True)
    x_data.insert(4, "Low", low, True)
    x_data.insert(5, "Close", close_, True)
    x_data.insert(6, "Target", y, True)
    x_data.to_csv("mfo_selected_features.csv")
    print("Selected Features-1   ::  ", mfo_selected_features, " Count :: ", len(mfo_selected_features), "\n")
    pso_selected_features = [i for i in range(len(pso_best_features)) if pso_best_features[i] == 1]
    print("PSO SELECTED FEATURES :: ", pso_selected_features, " Count :: ", len(pso_best_features))

    pso_data = X.iloc[:, pso_selected_features]
    pso_data.insert(1, "Date", date_, True)
    pso_data.insert(2, "Open", open_, True)
    pso_data.insert(3, "High", high, True)
    pso_data.insert(4, "Low", low, True)
    pso_data.insert(5, "Close", close_, True)
    pso_data.insert(6, "Target", y, True)
    pso_data.to_csv("pso_selected_features.csv")


def write_to_file():
    data = compute_indicators()
    print("Number of columns:: ", data.shape[1])
    print("NUmber of rows:  ", data.shape[0])

    data.to_csv("data.csv")
    print("target_binary Count Valid  :::: ", data["Target"].value_counts(normalize=False), "\n")


def main():
    generate_features()


def count_target_class():
    df = pd.read_csv("./all_features.csv")
    print(df["Target"].value_counts(normalize=False))


if __name__ == "__main__":
    generate_features()
