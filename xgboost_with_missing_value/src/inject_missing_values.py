from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California housing regression dataset."""
    housing = fetch_california_housing(as_frame=True)
    return housing.data, housing.target  # type: ignore


def inject_missing_values(X: pd.DataFrame, missing_ratio: float, random_state: int = 42) -> pd.DataFrame:
    """
    Inject missing values randomly into the DataFrame.

    Args:
        X: Input DataFrame
        missing_ratio: Ratio of missing values to inject (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with missing values
    """
    np.random.seed(random_state)
    X_missing = X.copy()

    # Create a mask for missing values
    mask = np.random.rand(*X_missing.shape) < missing_ratio

    # Apply missing values
    X_missing = X_missing.mask(mask)

    return X_missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject missing values into California Housing dataset"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="Missing value ratio (0.0 to 1.0)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the dataset with missing values (CSV format)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate ratio
    if not 0.0 <= args.ratio <= 1.0:
        raise ValueError(f"ratio must be between 0.0 and 1.0, got {args.ratio}")

    # Load dataset
    print("Loading California Housing dataset...")
    X, y = load_dataset()
    print(f"Original dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Inject missing values
    print(f"Injecting missing values with ratio {args.ratio}...")
    X_missing = inject_missing_values(X, args.ratio, args.random_state)

    # Calculate statistics
    total_values = X_missing.size
    missing_count = X_missing.isnull().sum().sum()
    actual_ratio = missing_count / total_values

    print(f"Missing values injected: {missing_count} / {total_values} ({actual_ratio:.4f})")
    print("\nMissing values per column:")
    for col in X_missing.columns:
        col_missing = X_missing[col].isnull().sum()
        col_total = len(X_missing)
        col_ratio = col_missing / col_total
        print(f"  {col}: {col_missing} / {col_total} ({col_ratio:.4f})")

    # Combine X and y
    data_with_target = X_missing.copy()
    data_with_target['MedHouseVal'] = y

    # Save to CSV
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_with_target.to_csv(output_path, index=False)

    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    main()
