from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.generate_sample_data import generate_sample_data
from src.data.preprocess import build_daily_demand
from src.forecasting.baseline_models import forecast_all_skus
from src.inventory.policy import build_inventory_policy_table
from src.optimization.optimizer import optimize_reorders
from src.agents.workflow import InventoryDecisionAgent

st.set_page_config(page_title="Agentic Inventory Optimizer", page_icon="📦", layout="wide")

DATA_DIR = Path("data/sample")


@st.cache_data
def load_or_generate_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sales_path = DATA_DIR / "sales_history.csv"
    inventory_path = DATA_DIR / "inventory_master.csv"
    if not sales_path.exists() or not inventory_path.exists():
        generate_sample_data(DATA_DIR)
    sales = pd.read_csv(sales_path, parse_dates=["date"])
    inventory = pd.read_csv(inventory_path)
    return sales, inventory


@st.cache_data
def run_model_pipeline(budget: int, capacity_units: int):
    sales, inventory = load_or_generate_data()
    daily_demand = build_daily_demand(sales)
    forecasts = forecast_all_skus(daily_demand, horizon_days=30)
    policy_table = build_inventory_policy_table(forecasts, inventory, daily_demand)
    optimized = optimize_reorders(policy_table, budget=budget, capacity_units=capacity_units)
    return sales, inventory, daily_demand, forecasts, optimized


st.title("📦 Agentic Inventory and Reorder Optimization System")
st.caption("Demand forecasting + inventory policy + OR optimization + RAG-style agent explanation")

with st.sidebar:
    st.header("Scenario controls")
    budget = st.slider("Procurement budget", 5_000, 250_000, 100_000, step=1_000)
    capacity = st.slider("Warehouse capacity units", 1_000, 50_000, 12_000, step=500)
    st.write("Change these values to run what-if reorder scenarios.")

sales, inventory, daily_demand, forecasts, recommendations = run_model_pipeline(budget, capacity)
agent = InventoryDecisionAgent(policy_docs_dir="docs")

col1, col2, col3, col4 = st.columns(4)
col1.metric("SKUs", recommendations["sku_id"].nunique())
col2.metric("Reorder now", int((recommendations["action"] == "REORDER_NOW").sum()))
col3.metric("High risk", int((recommendations["stockout_risk"] == "High").sum()))
col4.metric("Total reorder cost", f"₹{recommendations['optimized_purchase_cost'].sum():,.0f}")

st.divider()
st.subheader("Optimized reorder recommendations")
view_cols = [
    "sku_id", "category", "current_stock", "lead_time_days", "forecast_lead_time_demand",
    "safety_stock", "reorder_point", "action", "stockout_risk", "recommended_order_qty",
    "optimized_purchase_cost",
]
st.dataframe(recommendations[view_cols], use_container_width=True)

st.divider()
st.subheader("Ask the inventory agent about a SKU")
selected_sku = st.selectbox("Select SKU", recommendations["sku_id"].tolist())
selected_row = recommendations[recommendations["sku_id"] == selected_sku].iloc[0].to_dict()

left, right = st.columns([1, 1])
with left:
    st.markdown("### SKU metrics")
    metrics = pd.DataFrame({
        "Metric": ["Current stock", "Forecast lead-time demand", "Safety stock", "Reorder point", "Recommended order quantity", "Stockout risk", "Action"],
        "Value": [
            selected_row["current_stock"], round(selected_row["forecast_lead_time_demand"], 2),
            round(selected_row["safety_stock"], 2), round(selected_row["reorder_point"], 2),
            selected_row["recommended_order_qty"], selected_row["stockout_risk"], selected_row["action"],
        ],
    })
    st.table(metrics)

with right:
    st.markdown("### Agent explanation")
    explanation = agent.explain_recommendation(selected_row)
    st.write(explanation["answer"])
    with st.expander("Retrieved policy context"):
        for item in explanation["retrieved_context"]:
            st.markdown(f"**{item['source']}**")
            st.write(item["text"])

st.divider()
st.subheader("Demand history")
sku_history = daily_demand[daily_demand["sku_id"] == selected_sku].sort_values("date")
st.line_chart(sku_history.set_index("date")["quantity_sold"])
