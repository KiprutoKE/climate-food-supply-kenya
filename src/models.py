"""Forecasting models: Gaussian Process, Random Forest and optional Prophet."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # Prophet is optional so the app still runs without it installed.
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    Prophet = None  # type: ignore[assignment]
    PROPHET_AVAILABLE = False


@dataclass
class ForecastResult:
    metrics: dict[str, float]
    forecast: pd.DataFrame
    residuals: pd.DataFrame | None = None


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": _safe_mape(y_true, y_pred),
    }


def _mean_metrics(folds: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in folds[0]:
        values = np.array([fold[key] for fold in folds], dtype=float)
        finite = values[np.isfinite(values)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
    return out


def _time_series_cv(model, X: pd.DataFrame, y: pd.Series, max_splits: int = 5) -> dict[str, float]:
    """Cross-validate with TimeSeriesSplit so future rows never train on the past test."""
    n = len(X)
    if n < 3:
        raise ValueError("At least 3 rows are required for time-series cross-validation.")
    n_splits = min(max_splits, n - 1)
    folds: list[dict[str, float]] = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        estimator = clone(model)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        prediction = estimator.predict(X.iloc[test_idx])
        folds.append(_metrics(y.iloc[test_idx], prediction))
    return _mean_metrics(folds)


def extrapolate(series, periods: int, method: str = "trend") -> np.ndarray:
    """Project a series into the future.

    ``trend`` fits a least-squares line over the whole history (more stable than
    using the last two points); ``constant`` repeats the latest value.
    """
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    if periods <= 0:
        return np.empty(0)
    if values.size == 0:
        return np.zeros(periods)
    if method == "constant" or values.size < 2:
        return np.full(periods, values[-1], dtype=float)
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    future_x = np.arange(values.size, values.size + periods, dtype=float)
    return slope * future_x + intercept


def build_future(data: pd.DataFrame, features, horizon: int, method: str = "trend") -> pd.DataFrame:
    last_year = int(data["Year"].max())
    years = np.arange(last_year + 1, last_year + horizon + 1)
    future = pd.DataFrame({"Year": years})
    for feature in features:
        future[feature] = extrapolate(data[feature], horizon, method)
    return future


def forecast_gpr(data, target: str, features, horizon: int = 5, method: str = "trend") -> ForecastResult:
    features = list(features)
    X, y = data[features], data[target]
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42,
            normalize_y=True,
        ),
    )
    metrics = _time_series_cv(model, X, y)
    model.fit(X, y)
    future = build_future(data, features, horizon, method)
    future["Predicted Price"] = model.predict(future[features])
    return ForecastResult(metrics, future[["Year", "Predicted Price"]])


def forecast_random_forest(data, target: str, features, horizon: int = 5, method: str = "trend") -> ForecastResult:
    features = list(features)
    X, y = data[features], data[target]
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    metrics = _time_series_cv(model, X, y)
    model.fit(X, y)
    future = build_future(data, features, horizon, method)
    future["Predicted Price"] = model.predict(future[features])
    return ForecastResult(metrics, future[["Year", "Predicted Price"]])


def _prophet_frame(data: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    frame = data[["Year", target, *features]].copy()
    frame["ds"] = pd.to_datetime(frame["Year"].astype(int).astype(str) + "-01-01")
    return frame.rename(columns={target: "y"})


def _prophet_model(features: list[str]):
    # Annual data has no within-year seasonality, so keep only the trend.
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    for feature in features:
        model.add_regressor(feature)
    return model


def _pad(values, length: int) -> list[float]:
    values = list(values)[:length]
    if values and len(values) < length:
        values += [values[-1]] * (length - len(values))
    return values


def forecast_prophet(data, target: str, features, horizon: int = 5, method: str = "trend") -> ForecastResult:
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed. Run `pip install -r requirements.txt`.")

    features = list(features)
    frame = _prophet_frame(data, target, features)
    holdout = min(3, max(1, len(frame) // 5))
    train, test = frame.iloc[:-holdout], frame.iloc[-holdout:]

    # Backtest on the last few years so reported metrics are out-of-sample.
    backtest = _prophet_model(features)
    backtest.fit(train[["ds", "y", *features]])
    future_bt = backtest.make_future_dataframe(periods=holdout, freq="YS")
    for feature in features:
        future_bt[feature] = _pad(list(train[feature]) + list(test[feature]), len(future_bt))
    pred_bt = backtest.predict(future_bt).tail(holdout)
    metrics = _metrics(test["y"].to_numpy(), pred_bt["yhat"].to_numpy())

    # Final model on all data.
    model = _prophet_model(features)
    model.fit(frame[["ds", "y", *features]])
    future = model.make_future_dataframe(periods=horizon, freq="YS")
    for feature in features:
        future[feature] = _pad(list(frame[feature]) + list(extrapolate(frame[feature], horizon, method)), len(future))
    forecast = model.predict(future).tail(horizon)[["ds", "yhat"]]
    forecast = forecast.rename(columns={"ds": "Year", "yhat": "Predicted Price"}).reset_index(drop=True)
    forecast["Year"] = pd.to_datetime(forecast["Year"]).dt.year

    in_sample = model.predict(frame[["ds", *features]])
    residuals = pd.DataFrame(
        {"Year": frame["Year"].to_numpy(), "Residual": frame["y"].to_numpy() - in_sample["yhat"].to_numpy()}
    )
    return ForecastResult(metrics, forecast, residuals)
