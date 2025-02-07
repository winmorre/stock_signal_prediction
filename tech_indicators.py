import random
from tkinter import END
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import talib as tb
import os
import joblib
from collections import Counter
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import model_selection
import xgboost as xgb
from ta_patterns import create_signals
import matplotlib.pyplot as plt
import warnings
from sklearn import utils
import yfinance as yf
from collections import Counter
import pickle

model_names = ["svm", "knn", "rf", "gb", "xgb_model"]


def format_data(df, start_time, end_time, custom=True, trigrams=True, patterns=True, avg_days=5,
                additional=["O", "M", "V"]):
    """Takes input features and creates TA indicators, the 8-trigram scheme and Target labels.

    Parameters
    ----------
    df:
        Pandas DataFrame containing Open, High, Low, Close, Volume and Date columns.
    start_time:
        Start time in datetime format when the stock is purchased
    end_time:
        End time in datetime format when the stock is sold
    custom:
        Boolean Value to specify whether to create custom signals or not
    trigrams:
        Boolean value to specify whether to calculate 8 Trigrams or not
    patterns:
        Boolean Value to specify whether to calculate typical candlestick patterns or not
    avg_days:
        Integer denoting the number of days for which rolling average has to be calculated
    additional:
        List containing values O,M,V which specify which additional stock market indicators are to be calculated

    Returns
    ----------
    Pandas DataFrame with columns -
        Updated Open, High, Low and Closing Prices, Volume, Trigrams, Target and optionally Short Line Cdl, Long Line Cdl, Spinning Top and Closing Marubozu (if custom signals are required)
    """

    date_mask = (df["Date"] > start_time) & (df["Date"] <= end_time)
    df = df.loc[date_mask]

    short_ind = 5
    long_ind = 10

    # OVERLAP INDICATORS
    df["ma"] = tb.MA(df["Close"], timeperiod=short_ind)
    df["ema"] = tb.EMA(df["Close"], timeperiod=long_ind)
    df["dema"] = tb.DEMA(df["Close"], timeperiod=short_ind)
    df["kama"] = tb.KAMA(df["Close"], timeperiod=short_ind)
    df["sma"] = tb.SMA(df["Close"], timeperiod=long_ind)
    df["sar"] = tb.SAR(df["High"], df["Low"])

    # MOMENTUM INDICATORS
    df["adx"] = tb.ADX(df["High"], df["Low"],
                       df["Close"], timeperiod=long_ind)
    df["cci"] = tb.CCI(df["High"], df["Low"],
                       df["Close"], timeperiod=long_ind)
    df["apo"] = tb.APO(df["Close"], fastperiod=long_ind,
                       slowperiod=short_ind)
    df["bop"] = tb.BOP(df["Open"], df["High"], df["Low"], df["Close"])
    df["macd"], df["macdsignal"], df["macdhist"] = tb.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["mfi"] = tb.MFI(df["High"], df["Low"], df["Close"],
                       df["Volume"], timeperiod=long_ind)
    df["mom"] = tb.MOM(df["Close"], timeperiod=long_ind)
    df["rsi"] = tb.RSI(df["Close"], timeperiod=long_ind)

    # VOLUME INDICATORS
    df["ad"] = tb.AD(df["High"], df["Low"], df["Close"], df["Volume"])
    df["adosc"] = tb.ADOSC(df["High"], df["Low"], df["Close"],
                           df["Volume"], fastperiod=short_ind, slowperiod=long_ind)
    df["obv"] = tb.OBV(df["Close"], df["Volume"])
    df["trange"] = tb.TRANGE(df["High"], df["Low"], df["Close"])
    df["atr"] = tb.ATR(df["High"], df["Low"],
                       df["Close"], timeperiod=long_ind)
    df["natr"] = tb.NATR(df["High"], df["Low"],
                         df["Close"], timeperiod=long_ind)

    df.reset_index(drop=True, inplace=True)

    # 8 TRIGRAMS
    if trigrams == True:
        trigrams = []
        for i in range(1, len(df)):
            if (df.loc[i, "High"] > df.loc[i - 1, "High"]) & (df.loc[i, "Low"] < df.loc[i - 1, "Low"]) & (
                    df.loc[i, "Close"] > df.loc[i - 1, "Close"]):
                signal = 100  # "BullishHorn"
            elif (df.loc[i, "High"] > df.loc[i - 1, "High"]) & (df.loc[i, "Low"] < df.loc[i - 1, "Low"]) & (
                    df.loc[i, "Close"] < df.loc[i - 1, "Close"]):
                signal = -100  # "BearHorn"
            elif (df.loc[i, "High"] > df.loc[i - 1, "High"]) & (df.loc[i, "Low"] > df.loc[i - 1, "Low"]) & (
                    df.loc[i, "Close"] > df.loc[i - 1, "Close"]):
                signal = 100  # "BullishHigh"
            elif (df.loc[i, "High"] > df.loc[i - 1, "High"]) & (df.loc[i, "Low"] > df.loc[i - 1, "Low"]) & (
                    df.loc[i, "Close"] < df.loc[i - 1, "Close"]):
                signal = -100  # "BearHigh"
            elif (df.loc[i, "High"] < df.loc[i - 1, "High"]) & (df.loc[i, "Low"] < df.loc[i - 1, "Low"]) & (
                    df.loc[i, "Close"] > df.loc[i - 1, "Close"]):
                signal = 100  # "BullishLow"
            elif (df.loc[i, "High"] < df.loc[i - 1, "High"]) & (df.loc[i, "Low"] < df.loc[i - 1, "Low"]) & (
                    df.loc[i, "Close"] < df.loc[i - 1, "Close"]):
                signal = -100  # "BearLow"
            elif (df.loc[i, "High"] < df.loc[i - 1, "High"]) & (df.loc[i, "Low"] > df.loc[i - 1, "Low"]) & (
                    df.loc[i, "Close"] > df.loc[i - 1, "Close"]):
                signal = 100  # "BullishHarami"
            elif (df.loc[i, "High"] < df.loc[i - 1, "High"]) & (df.loc[i, "Low"] > df.loc[i - 1, "Low"]) & (
                    df.loc[i, "Close"] < df.loc[i - 1, "Close"]):
                signal = -100  # "BearHarami"
            else:
                signal = 0
            trigrams.append(signal)
    else:
        trigrams = [0] * (len(df.index) - 1)

    df.drop(df.index[0], inplace=True)
    df["trigrams"] = trigrams

    # TARGET
    df["target"] = df["Close"].pct_change().rolling(
        avg_days).mean().shift(avg_days)

    df.dropna(inplace=True)

    columns = ["Open", "High", "Volume", "Low",
               "trigrams", "target"]

    if custom == True:
        df = create_signals(data=df)
        columns = columns + ["shortLineCdl",
                             "longLineCdl", "spinningTop", "closingMarubozu"]

    if "O" in additional:
        columns = columns + ["ma", "ema", "dema", "kama", "sma", "sar"]
    if "M" in additional:
        columns = columns + ["adx", "cci", "apo",
                             "bop", "macd", "mfi", "mom", "rsi"]
    if "V" in additional:
        columns = columns + ["ad", "adosc", "obv", "trange", "atr", "natr"]

    df = df[columns]

    return df


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
        if x > 0:
            return 1
        else:
            return 0

    if classes == 3:
        DEV_FACTOR = 0.7

        st_dev = st_dev * DEV_FACTOR
        if x < -st_dev:
            return 0
        elif x < st_dev and x > - st_dev:
            return 1
        else:
            return 2

    if classes == 5:
        DEV_FACTOR = 0.7
        st_dev = st_dev * DEV_FACTOR
        if x < -st_dev * 2:
            return 0
        elif x < -st_dev and x > -st_dev * 2:
            return 1
        elif x < st_dev and x > -st_dev:
            return 2
        elif x > st_dev and x < st_dev * 2:
            return 3
        else:
            return 4


import os
import pandas as pd
import numpy as np
import talib as ta
from candle_patterns import cs_patterns_rest


def create_signals(data):
    """
    Creates technical trading signals based on candlestick charting patterns.

    Args:
        data (pandas.DataFrame): A dataframe of OHLCV (Open, High, Low, Close, Volume) data.

    Returns:
        pandas.DataFrame: A dataframe of OHLCV data with additional columns for each signal generated.
    """

    for signal in cs_patterns_rest:
        try:
            values = cs_patterns_rest[signal](
                data.Open, data.High, data.Low, data.Close)
            data[signal] = values
        except Exception as e:
            print(str(e))
    data = data.reset_index()

    return data


def fit_models(sample, models, cv=0, classes=2, fit=False):
    """
    Function used to fit models and evaluate their performance on a given dataset.

    Parameters
    ----------
    sample: pandas dataframe
        Pandas dataframe with the necessary features
    models: list
        List of ML models
    cv: int
        Number of cross validations
    classes: int
        Number of target classes
    fit: Bool
        Flag to indicate if the model has to be fitted or not

    Returns
    -------
    fitted_models : list
        List of fitted ML models
    X_test : year
        A feature matrix for Test set
    y_test : year
        Labels for Test set
    """

    ITER_SIZE = 5
    _df = pd.DataFrame()

    for col in sample.columns:

        _df[col] = sample[col]
        for i in range(ITER_SIZE):
            _df[f"{col}_{i}"] = sample[col].shift(periods=i + 1)
            if _df.shape[0] > 8000:
                raise Exception("Weird stuff going on")
        _df = _df.merge(_df, how="right")
    sample = _df

    X = sample.dropna().drop(["target"], axis=1)
    X = X.dropna().drop([f"target_{i}" for i in range(ITER_SIZE)], axis=1)

    sample.dropna(inplace=True)
    y = sample["target"].shift(-1).apply(lambda x: create_target(x, classes=classes, st_dev=sample["target"].std()))

    scaler = StandardScaler()
    pipeline = Pipeline(steps=[("scaler", scaler), ])
    X = pipeline.fit_transform(X)

    split_size = int(len(X) * 0.8)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]

    y_train = np.stack(y_train.values.tolist(), axis=0)
    y_test = np.stack(y_test.values.tolist(), axis=0)
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)
    X_train_lstm = X_train.reshape(X_train.shape[0], -1, ITER_SIZE + 1)
    X_test_lstm = X_test.reshape(X_test.shape[0], -1, ITER_SIZE + 1)
    if fit == True:
        fitted_models = []
        if len(models) == 0:

            svm_params = {
                "svc__C": [1],
                "svc__gamma": [0.1]
            }
            knn_params = {
                "knn__n_neighbors": [150],
                "knn__weights": ["distance"],
                "knn__algorithm": ["auto"],
                "knn__leaf_size": [1]

            }
            rf_params = {
                "rf__n_estimators": [9],
                "rf__criterion": ["gini"],
                "rf__min_samples_leaf": [5],
                "rf__max_depth": [1]
            }
            gb_params = {
                "gb__n_estimators": [1],
                "gb__max_features": [7],
                "gb__max_depth": [1]
            }
            xgb_params = {
                "xgb__n_estimators": [10],

                "xgb__max_depth": [3],
                "xgb__min_child_weight": [10],
                "xgb__gamma": [0],
                "xgb__learning_rate": [0.1],
                "xgb__seed": [27],
                "xgb__subsample": [0.65],
            }

            print("\tFitting Models...")

            svm, svm_best_params = iterate_models(
                SVC(), X_train, y_train, svm_params, cv)

            knn, knn_best_params = iterate_models(KNeighborsClassifier(), X_train,
                                                  y_train, knn_params, cv)

            rf, rf_best_params = iterate_models(RandomForestClassifier(),
                                                X_train, y_train, rf_params, cv)

            gb, gb_best_params = iterate_models(GradientBoostingClassifier(),
                                                X_train, y_train, gb_params, cv)
            xgb_model, xgb_best_params = iterate_models(xgb.XGBClassifier(),
                                                        X_train, y_train, xgb_params, cv)
            if cv != 0:
                print("SVM: ")
                print(svm_best_params)
                print("KNN: ")
                print(knn_best_params)
                print("RF: ")
                print(rf_best_params)
                print("GB: ")
                print(gb_best_params)
                print("XGB: ")
                print(xgb_best_params)

            fitted_models = [svm, knn, rf, gb, xgb_model]

        else:
            for i, model in enumerate(models):
                print(f"\tFitting Model_{model_names[i]}")
                if i == 5:
                    model.fit(X_train_lstm, y_train)
                else:
                    model.fit(X_train, y_train)
                fitted_models.append(model)
                filename = model_names[i] + ".sav"
                pickle.dump(model, open(
                    f"Lin_et_al_2021//ensemble_models//{classes}class//" + filename, 'wb'))
    else:
        fitted_models = [joblib.load(f"Lin_et_al_2021//ensemble_models//{classes}class//" + model) for model in
                         os.listdir(f"Lin_et_al_2021//ensemble_models//{classes}class//")]

    return fitted_models, X_test, y_test