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
    """Score on a held-out future window, then refit on every row before saving.

    The split exists to produce an honest accuracy figure: the model is scored
    on hours it has never seen. But a model shipped from that split has never
    seen the most recent quarter of history either, which is the part most like
    tomorrow. So the final model is refitted on 100% of the data with identical
    settings. The bundle records both windows, so /health can state exactly what
    was measured and what was shipped.
    """
    X_train, y_train, X_test, y_test, features = split(df)

    scorer = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42
    )
    scorer.fit(X_train, y_train)
    metrics = evaluate(y_test, scorer.predict(X_test))

    model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5, random_state=42
    )
    model.fit(df[features], df[TARGET])

    MODEL_DIR.mkdir(exist_ok=True)

    def window(index):
        return {"start": index.min().isoformat(),
                "end": index.max().isoformat(),
                "rows": int(len(index))}

    bundle = {
        "model": model,
        "features": features,
        "metrics": metrics,
        "evaluated_on": window(X_test.index),
        "trained_on": window(df.index),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "train_rows": int(len(df)),
    }

    print(f"Scored on  {bundle['evaluated_on']['rows']:>6} rows  "
          f"{bundle['evaluated_on']['start'][:10]} to {bundle['evaluated_on']['end'][:10]}")
    print(f"Shipped on {bundle['trained_on']['rows']:>6} rows  "
          f"{bundle['trained_on']['start'][:10]} to {bundle['trained_on']['end'][:10]}")

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    for path in (MODEL_PATH, MODEL_DIR / f"gbm_{stamp}.pkl"):
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    return metrics


def load_model() -> dict:
    """Load the saved model bundle."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)