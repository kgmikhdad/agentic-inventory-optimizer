from __future__ import annotations

import pandas as pd


def build_daily_demand(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales rows into complete daily SKU demand series."""
    required = {"date", "sku_id", "quantity_sold"}
    missing = required - set(sales.columns)
    if missing:
        raise ValueError(f"Sales data is missing required columns: {sorted(missing)}")

    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce").fillna(0)
    df = df[df["quantity_sold"] >= 0]

    aggregated = (
        df.groupby(["sku_id", "date"], as_index=False)["quantity_sold"]
        .sum()
        .sort_values(["sku_id", "date"])
    )

    completed = []
    for sku_id, sku_df in aggregated.groupby("sku_id"):
        full_idx = pd.date_range(sku_df["date"].min(), sku_df["date"].max(), freq="D")
        sku_full = sku_df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
        sku_full["sku_id"] = sku_id
        sku_full["quantity_sold"] = sku_full["quantity_sold"].fillna(0)
        completed.append(sku_full[["sku_id", "date", "quantity_sold"]])

    return pd.concat(completed, ignore_index=True)
