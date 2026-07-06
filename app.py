from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.workflow import InventoryDecisionAgent
from src.data.generate_sample_data import generate_sample_data
from src.data.preprocess import build_daily_demand
from src.forecasting.baseline_models import forecast_all_skus
from src.inventory.policy import build_inventory_policy_table
from src.optimization.optimizer import optimize_reorders
from src.rag.simple_rag import SimplePolicyRetriever

st.set_page_config(page_title="Agentic Inventory Optimizer", page_icon="📦", layout="wide")

DATA_DIR = Path("data/sample")
DOCS_DIR = Path("docs")


def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sales_path = DATA_DIR / "sales_history.csv"
    inventory_path = DATA_DIR / "inventory_master.csv"
    if not sales_path.exists() or not inventory_path.exists():
        generate_sample_data(DATA_DIR)
    sales = pd.read_csv(sales_path, parse_dates=["date"])
    inventory = pd.read_csv(inventory_path)
    return sales, inventory


def run_model(sales, inventory, budget, capacity, demand_shock, lead_time_shock):
    daily = build_daily_demand(sales)
    forecasts = forecast_all_skus(daily, horizon_days=30).copy()
    inventory = inventory.copy()
    for col in ["daily_forecast", "forecast_7d", "forecast_14d", "forecast_30d", "daily_demand_std"]:
        if col in forecasts.columns:
            forecasts[col] = forecasts[col] * demand_shock
    inventory["lead_time_days"] = (inventory["lead_time_days"] * lead_time_shock).round().clip(lower=1).astype(int)
    policy = build_inventory_policy_table(forecasts, inventory, daily)
    plan = optimize_reorders(policy, budget=budget, capacity_units=capacity)
    return daily, forecasts, policy, plan


def prompt_text(row, contexts, question):
    fields = ["sku_id", "category", "supplier_id", "priority", "current_stock", "lead_time_days", "forecast_lead_time_demand", "safety_stock", "reorder_point", "stockout_risk", "action", "recommended_order_qty"]
    numbers = "\n".join([f"- {field}: {row.get(field)}" for field in fields])
    rag = "\n\n".join([f"Source: {item.get('source')}\n{item.get('text')}" for item in contexts])
    return f"You are an inventory decision assistant. Use only the given numerical context and retrieved policy context. Do not invent order quantities.\n\nNUMERICAL CONTEXT:\n{numbers}\n\nRETRIEVED POLICY CONTEXT:\n{rag}\n\nUSER QUESTION:\n{question}\n\nANSWER:"


def local_answer(row, contexts, question):
    qty = int(row.get("recommended_order_qty", 0))
    if qty > 0:
        decision = f"Order {qty} units of {row['sku_id']}."
    elif row.get("action") == "REORDER_NOW":
        decision = f"{row['sku_id']} is below reorder point, but was not selected because of budget or capacity constraints."
    else:
        decision = f"Do not reorder {row['sku_id']} now."
    policy = " ".join([item.get("text", "")[:180] for item in contexts[:2]]) or "No matching policy context was retrieved."
    return f"""
**Question:** {question}

**Decision:** {decision}

**Reasoning:** Current stock is {row['current_stock']:.0f}. Reorder point is {row['reorder_point']:.1f}. Lead-time demand is {row['forecast_lead_time_demand']:.1f}. Safety stock is {row['safety_stock']:.1f}. Stockout risk is **{row['stockout_risk']}**.

**Policy context:** {policy}

**Control rule:** the optimizer decides the quantity. The language layer explains the decision.
"""


st.title("📦 Agentic Inventory and Reorder Optimization System")
st.caption("Forecasting + inventory policy + optimization + RAG + prompt lab + agent trace + purchase orders")

with st.sidebar:
    st.header("Scenario controls")
    budget = st.slider("Procurement budget", 5_000, 250_000, 100_000, step=1_000)
    capacity = st.slider("Warehouse capacity units", 1_000, 50_000, 12_000, step=500)
    demand_shock = st.slider("Demand shock multiplier", 0.50, 2.00, 1.00, step=0.05)
    lead_time_shock = st.slider("Lead-time shock multiplier", 0.50, 2.00, 1.00, step=0.05)

try:
    sales, inventory = load_data()
    daily, forecasts, policy, plan = run_model(sales, inventory, budget, capacity, demand_shock, lead_time_shock)
except Exception as error:
    st.error("The app failed while loading the model pipeline.")
    st.exception(error)
    st.stop()

agent = InventoryDecisionAgent("docs")
retriever = SimplePolicyRetriever("docs")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("SKUs", plan["sku_id"].nunique())
c2.metric("Reorder signals", int((plan["action"] == "REORDER_NOW").sum()))
c3.metric("Approved orders", int((plan["recommended_order_qty"] > 0).sum()))
c4.metric("High risk", int((plan["stockout_risk"] == "High").sum()))
c5.metric("Order cost", f"{plan['optimized_purchase_cost'].sum():,.0f}")

tabs = st.tabs(["Control Tower", "SKU Center", "RAG", "Prompt Lab", "Agent Trace", "Scenario", "Purchase Orders", "Data"])

with tabs[0]:
    st.subheader("Control Tower")
    view_cols = ["sku_id", "category", "supplier_id", "priority", "current_stock", "lead_time_days", "forecast_lead_time_demand", "safety_stock", "reorder_point", "action", "stockout_risk", "recommended_order_qty", "optimized_purchase_cost"]
    st.dataframe(plan[view_cols], use_container_width=True, height=420)
    st.download_button("Download recommendations CSV", plan[view_cols].to_csv(index=False), "reorder_recommendations.csv", "text/csv")
    left, right = st.columns(2)
    left.bar_chart(plan["stockout_risk"].value_counts())
    right.bar_chart(plan.groupby("supplier_id")["optimized_purchase_cost"].sum().sort_values(ascending=False))

with tabs[1]:
    st.subheader("SKU Command Center")
    sku = st.selectbox("Select SKU", plan["sku_id"].tolist(), key="sku_center")
    row = plan[plan["sku_id"] == sku].iloc[0].to_dict()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current stock", f"{row['current_stock']:.0f}")
    k2.metric("Reorder point", f"{row['reorder_point']:.1f}")
    k3.metric("Recommended qty", int(row["recommended_order_qty"]))
    k4.metric("Risk", row["stockout_risk"])
    st.line_chart(daily[daily["sku_id"] == sku].set_index("date")["quantity_sold"])
    st.markdown("### Agent explanation")
    st.write(agent.explain_recommendation(row)["answer"])

with tabs[2]:
    st.subheader("RAG Knowledge Base")
    rag_question = st.text_area("Ask a policy question", "Which policy applies to high-priority products with high stockout risk?")
    retrieved = retriever.retrieve(rag_question, top_k=5)
    for item in retrieved:
        with st.expander(f"{item['source']} | score={item.get('score', 0):.3f}"):
            st.write(item["text"])

with tabs[3]:
    st.subheader("LLM-style Prompt Lab")
    prompt_sku = st.selectbox("SKU for prompt", plan["sku_id"].tolist(), key="prompt_sku")
    row = plan[plan["sku_id"] == prompt_sku].iloc[0].to_dict()
    question = st.text_area("Business question", f"Should we reorder {prompt_sku}? Explain why.")
    contexts = retriever.retrieve(f"{row.get('category')} {row.get('priority')} {row.get('supplier_id')} stockout reorder", top_k=3)
    left, right = st.columns(2)
    left.markdown("### Prompt sent to LLM")
    left.code(prompt_text(row, contexts, question), language="text")
    right.markdown("### Local generated answer")
    right.write(local_answer(row, contexts, question))

with tabs[4]:
    st.subheader("Agent Workflow Trace")
    trace_sku = st.selectbox("SKU for trace", plan["sku_id"].tolist(), key="trace_sku")
    trace_row = plan[plan["sku_id"] == trace_sku].iloc[0]
    steps = [
        ("1. Read inventory", f"Current stock = {trace_row.current_stock}; lead time = {trace_row.lead_time_days} days."),
        ("2. Forecast demand", f"Lead-time demand = {trace_row.forecast_lead_time_demand:.1f}; 30-day forecast = {trace_row.forecast_30d:.1f}."),
        ("3. Apply inventory policy", f"Safety stock = {trace_row.safety_stock:.1f}; reorder point = {trace_row.reorder_point:.1f}; action = {trace_row.action}."),
        ("4. Optimize quantity", f"Recommended quantity = {int(trace_row.recommended_order_qty)}; order cost = {trace_row.optimized_purchase_cost:.2f}."),
        ("5. Retrieve policy", "RAG searches local Markdown documents in the docs folder."),
        ("6. Explain recommendation", "The agent explains the optimizer output without changing the quantity."),
    ]
    for title, detail in steps:
        st.markdown(f"### {title}")
        st.success(detail)

with tabs[5]:
    st.subheader("Scenario Simulator")
    alt_budget = st.number_input("Alternative budget", 1_000, 500_000, int(budget), step=1_000)
    alt_capacity = st.number_input("Alternative capacity", 100, 100_000, int(capacity), step=500)
    _, _, _, alt = run_model(sales, inventory, alt_budget, alt_capacity, demand_shock, lead_time_shock)
    comparison = pd.DataFrame([
        {"scenario": "current", "orders": int((plan.recommended_order_qty > 0).sum()), "cost": plan.optimized_purchase_cost.sum(), "units": plan.recommended_order_qty.sum()},
        {"scenario": "alternative", "orders": int((alt.recommended_order_qty > 0).sum()), "cost": alt.optimized_purchase_cost.sum(), "units": alt.recommended_order_qty.sum()},
    ])
    st.dataframe(comparison, use_container_width=True)

with tabs[6]:
    st.subheader("Purchase Order Generator")
    po = plan[plan["recommended_order_qty"] > 0].copy()
    if po.empty:
        st.warning("No purchase orders under the current constraints.")
    else:
        po["line_cost"] = po["recommended_order_qty"] * po["unit_cost"]
        po_cols = ["supplier_id", "sku_id", "category", "priority", "recommended_order_qty", "unit_cost", "line_cost", "pack_size", "stockout_risk"]
        st.dataframe(po[po_cols], use_container_width=True)
        st.download_button("Download purchase order CSV", po[po_cols].to_csv(index=False), "purchase_orders.csv", "text/csv")

with tabs[7]:
    st.subheader("Data and Validation")
    st.write("Sales rows:", len(sales))
    st.write("Inventory rows:", len(inventory))
    st.dataframe(sales.head(20), use_container_width=True)
    st.dataframe(inventory.head(20), use_container_width=True)
