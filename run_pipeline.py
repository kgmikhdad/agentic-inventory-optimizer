from pathlib import Path

import pandas as pd

from src.data.generate_sample_data import generate_sample_data
from src.data.preprocess import build_daily_demand
from src.forecasting.baseline_models import forecast_all_skus
from src.inventory.policy import build_inventory_policy_table
from src.optimization.optimizer import optimize_reorders
from src.agents.workflow import InventoryDecisionAgent

DATA_DIR = Path("data/sample")
REPORTS_DIR = Path("reports")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sales_path = DATA_DIR / "sales_history.csv"
    inventory_path = DATA_DIR / "inventory_master.csv"

    if not sales_path.exists() or not inventory_path.exists():
        print("Generating sample data...")
        generate_sample_data(DATA_DIR)

    sales = pd.read_csv(sales_path, parse_dates=["date"])
    inventory = pd.read_csv(inventory_path)

    daily_demand = build_daily_demand(sales)
    forecasts = forecast_all_skus(daily_demand, horizon_days=30)

    policy_table = build_inventory_policy_table(
        forecasts=forecasts,
        inventory_master=inventory,
        daily_demand=daily_demand,
    )

    optimized = optimize_reorders(
        policy_table=policy_table,
        budget=100000,
        capacity_units=12000,
    )

    agent = InventoryDecisionAgent(policy_docs_dir="docs")
    explanations = []
    for _, row in optimized.iterrows():
        explanation = agent.explain_recommendation(row.to_dict())
        explanations.append(explanation["answer"])

    optimized["agent_explanation"] = explanations
    out_path = REPORTS_DIR / "reorder_recommendations.csv"
    optimized.to_csv(out_path, index=False)

    print(f"Saved recommendations to {out_path}")
    print(optimized[["sku_id", "action", "recommended_order_qty", "stockout_risk"]].head(10))


if __name__ == "__main__":
    main()
