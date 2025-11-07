import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
import sys

# Try to force an interactive backend commonly available.
# If this fails, we'll still try to show plots and provide a helpful message.
for preferred in ("TkAgg", "Qt5Agg", "MacOSX"):
    try:
        matplotlib.use(preferred, force=True)
        break
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import joblib

DATA_PATH = "Walmart.csv"   # put your file here


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    sample = pd.read_csv(path, nrows=5)
    date_cols = [c for c in sample.columns if "date" in c.lower() or "week" in c.lower()]
    if date_cols:
        df = pd.read_csv(path, parse_dates=[date_cols[0]])
        df.rename(columns={date_cols[0]: "Date"}, inplace=True)
    else:
        df = pd.read_csv(path)
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def clean_data(df):
    df = df.copy().drop_duplicates()
    sales_col = next((c for c in df.columns if "sale" in c.lower()), None)
    if not sales_col:
        raise ValueError("No column with 'sales' in the name was found.")
    df.rename(columns={sales_col: "Weekly_Sales"}, inplace=True)
    df["Weekly_Sales"] = pd.to_numeric(df["Weekly_Sales"], errors="coerce")
    df["Weekly_Sales"].fillna(df["Weekly_Sales"].median(), inplace=True)

    for col in list(df.columns):
        if "store" in col.lower() and col != "Store":
            df.rename(columns={col: "Store"}, inplace=True)
        if "dept" in col.lower() and col != "Dept":
            df.rename(columns={col: "Dept"}, inplace=True)

    df.sort_values(by=["Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def explore_and_show(df):
    print("Matplotlib backend:", matplotlib.get_backend())
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    ts = df.set_index("Date").resample("W")["Weekly_Sales"].sum()

    # 1) Total sales over time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts.index, ts.values, linewidth=1.5)
    ax.set_title("Total Weekly Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    plt.show(block=True)   # <--- blocking: window appears and stays until closed

    # 2) Top 10 stores (if available)
    if "Store" in df.columns:
        top_stores = df.groupby("Store")["Weekly_Sales"].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_stores.index.astype(str), y=top_stores.values, ax=ax)
        ax.set_title("Top 10 Stores by Total Sales")
        ax.set_xlabel("Store")
        ax.set_ylabel("Total Sales")
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.show(block=True)

    # 3) Seasonal decomposition if enough points
    if len(ts) >= 104:
        try:
            result = seasonal_decompose(ts, model="additive", period=52, extrapolate_trend='freq')
            # result.plot() creates subplots; show them
            result.plot()
            plt.show(block=True)
        except Exception as e:
            print("Seasonal decomposition error:", e)
    else:
        print("Not enough data for seasonal decomposition (need >=104 weekly points).")


def forecasting_and_show(df):
    ts = df.set_index("Date").resample("W")["Weekly_Sales"].sum()
    if len(ts) < 10:
        print("Too few data points for forecasting.")
        return

    def create_lags(series, lags=[1, 2, 3, 4, 52]):
        df_lag = pd.DataFrame({'y': series})
        for lag in lags:
            df_lag[f'lag_{lag}'] = df_lag['y'].shift(lag)
        df_lag["rolling_4"] = df_lag['y'].shift(1).rolling(4).mean()
        return df_lag.dropna()

    df_feat = create_lags(ts)
    X = df_feat.drop("y", axis=1)
    y = df_feat["y"]
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_series = pd.Series(preds, index=y_test.index)

    try:
        rmse = mean_squared_error(y_test, preds_series, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, preds_series))
    print(f"Forecast RMSE: {rmse:.2f}")

    # Plot actual vs predicted (show only)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test.values, label="Actual", linewidth=2)
    ax.plot(preds_series.index, preds_series.values, label="Predicted", linewidth=2)
    ax.set_title("Actual vs Predicted Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show(block=True)


def main():
    print("Running show-only script. Make sure you run this in a desktop environment.")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    explore_and_show(df)
    forecasting_and_show(df)
    print("Finished. All plots were displayed (no files saved).")


if __name__ == "__main__":
    main()
