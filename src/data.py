"""Data loading, cleaning and feature engineering for the Kenya climate/food dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"

RENAME = {
    "Beans_Price_90KG": "Beans Price 90 KG",
    "Maize_Price_90KG": "Maize Price 90 KG",
    "Potatoes_Price_50KG": "Potatoes Price 50 KG",
    "Rice_Price_50KG": "Rice Price 50 KG",
    "Temperature - (Celsius)": "Temperature (Celsius)",
}

PRICE_COLUMNS = [
    "Beans Price 90 KG",
    "Maize Price 90 KG",
    "Potatoes Price 50 KG",
    "Rice Price 50 KG",
]
CLIMATE_COLUMNS = ["Temperature (Celsius)", "Rainfall (mm/year)"]
PCT_SUFFIX = " Pct Change"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the CSV, rename columns and add year-over-year price changes.

    The first year has no previous value, so its percentage change is set to 0
    instead of being filled with the column mean.
    """
    df = pd.read_csv(path or DATA_PATH)
    df = df.rename(columns=RENAME)
    df = df.sort_values("Year").reset_index(drop=True)

    for col in PRICE_COLUMNS:
        if col in df.columns:
            df[f"{col}{PCT_SUFFIX}"] = df[col].pct_change() * 100

    df = df.replace([np.inf, -np.inf], np.nan)
    pct_columns = [c for c in df.columns if c.endswith(PCT_SUFFIX)]
    if pct_columns:
        df[pct_columns] = df[pct_columns].fillna(0.0)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    return df


def pct_change_column(price_column: str) -> str:
    """Return the derived percentage-change column name for a price column."""
    return f"{price_column}{PCT_SUFFIX}"
