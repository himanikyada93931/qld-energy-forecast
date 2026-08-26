"""Train and compare models with a time-based split."""

import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import PROJECT_ROOT


TARGET = "demand_mw"
EXCLUDE = ["demand_mw", "price_aud_mwh"]

TEST_FRACTION = 0.25

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "gbm.pkl"


def split(df: pd.DataFrame):
    """Split by time, never randomly. Train on the past, test on the future."""
    cutoff = int(len(df) * (1 - TEST_FRACTION))

    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:]

    features = [c for c in df.columns if c not in EXCLUDE]

    return (train[features], train[TARGET],
            test[features], test[TARGET], features)


def evaluate(actual: pd.Series, predicted) -> dict:
    """Same metrics as the baseline, so the numbers are comparable."""
    error = pd.Series(predicted, index=actual.index) - actual

    return {
        "MAE": error.abs().mean(),
        "RMSE": (error ** 2).mean() ** 0.5,
        "MAPE": (error.abs() / actual).mean() * 100,
        "bias": error.mean(),
    }


def compare(df: pd.DataFrame) -> pd.DataFrame:
    """Train several models and score them all on the same test period."""
    X_train, y_train, X_test, y_test, features = split(df)

    print(f"Train: {len(X_train)} rows, {X_train.index.min()} to {X_train.index.max()}")
    print(f"Test:  {len(X_test)} rows, {X_test.index.min()} to {X_test.index.max()}")
    print(f"Features: {len(features)}\n")

    models = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(y_test, model.predict(X_test))
        print(f"{name:20} MAE {results[name]['MAE']:7.1f}   MAPE {results[name]['MAPE']:5.2f}%")

    return pd.DataFrame(results).T


def train_and_save(df: pd.DataFrame) -> dict:
    """Train the chosen model on all data and save it with its metadata."""
    X_train, y_train, X_test, y_test, features = split(df)

    model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42
    )
    model.fit(X_train, y_train)

    metrics = evaluate(y_test, model.predict(X_test))

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "features": features,
            "metrics": metrics,
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "train_rows": len(X_train),
        }, f)

    return metrics


def load_model() -> dict:
    """Load the saved model bundle."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)