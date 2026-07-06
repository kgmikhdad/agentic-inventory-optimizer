from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_std(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _weighted_moving_average(values: np.ndarray, window: int = 28) -> float:
    values = values[-window:]
    if len(values) == 0:
        return 0.0
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.average(values, weights=weights))


def forecast_single_sku(sku_daily: pd.DataFrame, horizon_days: int = 30) -> dict:
    values = sku_daily.sort_values("date")["quantity_sold"].astype(float).to_numpy()
    recent_mean = _weighted_moving_average(values, window=28)
    long_mean = float(np.mean(values)) if len(values) else 0.0
    daily_forecast = max(0.0, 0.7 * recent_mean + 0.3 * long_mean)
    demand_std = _safe_std(values[-56:]) if len(values) else 0.0

    return {
        "daily_forecast": daily_forecast,
        "forecast_7d": daily_forecast * 7,
        "forecast_14d": daily_forecast * 14,
        "forecast_30d": daily_forecast * horizon_days,
        "daily_demand_std": demand_std,
        "model_name": "weighted_moving_average",
    }


def forecast_all_skus(daily_demand: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    rows = []
    for sku_id, sku_df in daily_demand.groupby("sku_id"):
        result = forecast_single_sku(sku_df, horizon_days=horizon_days)
        result["sku_id"] = sku_id
        rows.append(result)
    return pd.DataFrame(rows)
