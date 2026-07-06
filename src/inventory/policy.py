from __future__ import annotations

import math
import numpy as np
import pandas as pd

SERVICE_LEVEL_Z = {
    "low": 1.28,
    "medium": 1.65,
    "high": 2.05,
}


def calculate_safety_stock(daily_demand_std: float, lead_time_days: int, z_value: float) -> float:
    return max(0.0, z_value * daily_demand_std * math.sqrt(max(1, lead_time_days)))


def calculate_eoq(annual_demand: float, ordering_cost: float, holding_cost_per_unit: float) -> float:
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
        return 0.0
    return math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)


def classify_stockout_risk(current_stock: float, reorder_point: float) -> str:
    if current_stock <= 0.5 * reorder_point:
        return "High"
    if current_stock < reorder_point:
        return "Medium"
    return "Low"


def build_inventory_policy_table(
    forecasts: pd.DataFrame,
    inventory_master: pd.DataFrame,
    daily_demand: pd.DataFrame,
) -> pd.DataFrame:
    required_inventory_cols = {
        "sku_id", "current_stock", "lead_time_days", "unit_cost", "holding_cost_per_unit",
        "ordering_cost", "stockout_cost_per_unit", "min_order_qty", "max_order_qty",
        "pack_size", "storage_units_per_item", "priority",
    }
    missing = required_inventory_cols - set(inventory_master.columns)
    if missing:
        raise ValueError(f"Inventory master missing columns: {sorted(missing)}")

    df = inventory_master.merge(forecasts, on="sku_id", how="left")
    df["daily_forecast"] = df["daily_forecast"].fillna(0)
    df["daily_demand_std"] = df["daily_demand_std"].fillna(0)

    df["z_value"] = df["priority"].map(SERVICE_LEVEL_Z).fillna(1.65)
    df["forecast_lead_time_demand"] = df["daily_forecast"] * df["lead_time_days"]
    df["safety_stock"] = df.apply(
        lambda row: calculate_safety_stock(row["daily_demand_std"], int(row["lead_time_days"]), row["z_value"]),
        axis=1,
    )
    df["reorder_point"] = df["forecast_lead_time_demand"] + df["safety_stock"]
    df["annual_demand"] = df["daily_forecast"] * 365
    df["eoq"] = df.apply(
        lambda row: calculate_eoq(row["annual_demand"], row["ordering_cost"], row["holding_cost_per_unit"]),
        axis=1,
    )
    df["stockout_risk"] = df.apply(
        lambda row: classify_stockout_risk(row["current_stock"], row["reorder_point"]),
        axis=1,
    )
    df["action"] = np.where(df["current_stock"] < df["reorder_point"], "REORDER_NOW", "SAFE")
    df["target_stock_level"] = df["forecast_30d"] + df["safety_stock"]
    df["raw_required_qty"] = (df["target_stock_level"] - df["current_stock"]).clip(lower=0)
    return df
