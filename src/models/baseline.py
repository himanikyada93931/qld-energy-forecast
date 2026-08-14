"""Seasonal naive baseline: demand equals the same hour one week ago."""

import pandas as pd

WEEK_HOURS = 168


def predict(history: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    """Predict demand at a UTC timestamp using the value 168 hours earlier."""
    lookup = timestamp - pd.Timedelta(hours=WEEK_HOURS)

    if lookup not in history.index:
        raise KeyError(f"No observation at {lookup}")

    return float(history.loc[lookup, "demand_mw"])


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Predict every row from one week earlier. Returns actual vs predicted."""
    result = pd.DataFrame(index=df.index)
    result["actual"] = df["demand_mw"]
    result["predicted"] = df["demand_mw"].shift(WEEK_HOURS)
    result["error"] = result["predicted"] - result["actual"]

    return result.dropna()


def score(result: pd.DataFrame) -> dict:
    """Standard regression metrics."""
    error = result["error"]
    actual = result["actual"]

    return {
        "n": len(result),
        "MAE": error.abs().mean(),
        "RMSE": (error ** 2).mean() ** 0.5,
        "MAPE": (error.abs() / actual).mean() * 100,
        "bias": error.mean(),
    }