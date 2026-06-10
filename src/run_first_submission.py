from __future__ import annotations

from make_features import build_features
from train_baseline import train_and_predict


def main() -> None:
    print("Building first-submission baseline features...")
    build_features()
    print("\nTraining baseline models and writing submission...")
    train_and_predict()
    print("\nDone. Submission is ready.")


if __name__ == "__main__":
    main()
