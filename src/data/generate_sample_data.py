from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_sample_data(output_dir: str | Path, n_skus: int = 40, n_days: int = 180, seed: int = 42) -> None:
    """Generate reproducible sample sales and inventory data for the demo."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    categories = ["medicine", "grocery", "electronics", "accessory", "spare_part"]
    suppliers = ["Supplier_A", "Supplier_B", "Supplier_C", "Supplier_D"]

    sales_rows = []
    inventory_rows = []

    for sku_idx in range(1, n_skus + 1):
        sku_id = f"SKU_{sku_idx:03d}"
        category = rng.choice(categories, p=[0.2, 0.25, 0.15, 0.25, 0.15])
        supplier_id = rng.choice(suppliers)

        base_demand = rng.uniform(3, 35)
        trend = rng.uniform(-0.03, 0.08)
        weekly_amp = rng.uniform(0.0, 0.35)
        intermittency = rng.uniform(0.0, 0.25)

        unit_cost = round(float(rng.uniform(40, 600)), 2)
        unit_price = round(unit_cost * float(rng.uniform(1.15, 1.75)), 2)
        lead_time_days = int(rng.integers(2, 12))
        current_stock = int(rng.integers(20, 450))
        holding_cost_per_unit = round(unit_cost * rng.uniform(0.005, 0.03), 2)
        ordering_cost = round(float(rng.uniform(100, 1200)), 2)
        stockout_cost_per_unit = round(unit_price * rng.uniform(0.15, 0.5), 2)
        min_order_qty = int(rng.choice([10, 20, 25, 50, 100]))
        max_order_qty = int(rng.choice([300, 500, 800, 1000]))
        pack_size = int(rng.choice([1, 5, 10, 20, 25]))
        storage_units_per_item = round(float(rng.uniform(0.5, 3.0)), 2)
        priority = rng.choice(["high", "medium", "low"], p=[0.25, 0.5, 0.25])

        for t, date in enumerate(dates):
            weekly = 1 + weekly_amp * np.sin(2 * np.pi * t / 7)
            trend_factor = max(0.2, 1 + trend * t / 10)
            lam = max(0.1, base_demand * weekly * trend_factor)
            quantity = rng.poisson(lam)
            if rng.random() < intermittency:
                quantity = 0
            sales_rows.append({
                "date": date.date().isoformat(),
                "sku_id": sku_id,
                "quantity_sold": int(quantity),
                "unit_price": unit_price,
                "category": category,
            })

        inventory_rows.append({
            "sku_id": sku_id,
            "category": category,
            "supplier_id": supplier_id,
            "current_stock": current_stock,
            "lead_time_days": lead_time_days,
            "unit_cost": unit_cost,
            "unit_price": unit_price,
            "holding_cost_per_unit": holding_cost_per_unit,
            "ordering_cost": ordering_cost,
            "stockout_cost_per_unit": stockout_cost_per_unit,
            "min_order_qty": min_order_qty,
            "max_order_qty": max_order_qty,
            "pack_size": pack_size,
            "storage_units_per_item": storage_units_per_item,
            "priority": priority,
        })

    pd.DataFrame(sales_rows).to_csv(output_dir / "sales_history.csv", index=False)
    pd.DataFrame(inventory_rows).to_csv(output_dir / "inventory_master.csv", index=False)
