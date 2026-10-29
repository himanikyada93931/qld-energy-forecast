"""Refit the model on everything in the database and save it to models/gbm.pkl.

Run this after new features are added or after a long stretch of new data:

    python -m scripts.retrain

It prints the score from the held-out future window, then saves a model refitted
on every row. The saved file has to be committed for the live service to use it.
"""

from src.features.build import build_features
from src.ingest.database import read_all
from src.models.train import train_and_save


def main() -> None:
    df = build_features(read_all())
    print(f"{len(df)} rows  {df.index.min()} -> {df.index.max()}\n")

    metrics = train_and_save(df)

    print()
    for name, value in metrics.items():
        print(f"{name:>6}: {value:.4f}")


if __name__ == "__main__":
    main()
