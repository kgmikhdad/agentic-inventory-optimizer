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
    inv_path = DATA_DIR / "inventory_master.csv"
    if not sales_path.exists() or not inv_path.exists():
        generate_sample_data(DATA_DIR)
    return pd.read_csv(sales_path, parse_dates=["date"]), pd.read_csv(inv_path)


def compute_plan(sales, inventory, budget, capacity, demand_shock, lead_time_shock):
    daily = build_daily_demand(sales)
    forecasts = forecast_all_skus(daily, horizon_days=30)
    forecasts = forecasts.copy()
    inventory = inventory.copy()
    for col in ["daily_forecast", "forecast_7d", "forecast_14d", "forecast_30d", "daily_demand_std"]:
        if col in forecasts:
            forecasts[col] = forecasts[col] * demand_shock
    inventory["lead_time_days"] = (inventory["lead_time_days"] * lead_time_shock).round().clip(lower=1).astype(int)
    policy = build_inventory_policy_table(forecasts, inventory, daily)
    plan = optimize_reorders(policy, budget=budget, capacity_units=capacity)
    return daily, forecasts, policy, plan


def local_llm_answer(row, rag_context, question):
    qty = int(row.get("recommended_order_qty", 0))
    policy = " ".join([x.get("text", "")[:180] for x in rag_context[:2]]) or "No policy context retrieved."
    if qty > 0:
        decision = f"Order {qty} units of {row['sku_id']}."
    elif row.get("action") == "REORDER_NOW":
        decision = f"{row['sku_id']} is risky, but the optimizer did not allocate an order under current constraints."
    else:
        decision = f"Do not reorder {row['sku_id']} now."
    return f"""**Question:** {question}

**Decision:** {decision}

**Reasoning:** Current stock is {row['current_stock']:.0f}, reorder point is {row['reorder_point']:.1f}, lead-time demand is {row['forecast_lead_time_demand']:.1f}, and safety stock is {row['safety_stock']:.1f}. Stockout risk is **{row['stockout_risk']}**.

**Policy grounding:** {policy}

**Important rule:** the optimizer decides the order quantity; the language layer only explains the recommendation.
"""


def build_prompt(row, rag_context, question):
    numbers = "\n".join([f"- {k}: {row.get(k)}" for k in ["sku_id", "category", "supplier_id", "priority", "current_stock", "lead_time_days", "forecast_lead_time_demand", "safety_stock", "reorder_point", "stockout_risk", "action", "recommended_order_qty", "optimized_purchase_cost"]])
    context = "\n\n".join([f"SOURCE: {x.get('source')}\n{x.get('text')}" for x in rag_context])
    return f"You are an inventory decision-support assistant. Do not invent quantities.\n\nNUMBERS:\n{numbers}\n\nRETRIEVED POLICY CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nANSWER:"


st.title("📦 Agentic Inventory and Reorder Optimization System")
st.caption("Enhanced demo: forecasting, optimization, RAG search, prompt lab, agent trace, scenarios, and purchase orders")

with st.sidebar:
    st.header("Scenario controls")
    budget = st.slider("Budget", 5_000, 250_000, 100_000, 1_000)
    capacity = st.slider("Capacity units", 1_000, 50_000, 12_000, 500)
    demand_shock = st.slider("Demand shock", 0.50, 2.00, 1.00, 0.05)
    lead_time_shock = st.slider("Lead-time shock", 0.50, 2.00, 1.00, 0.05)

sales, inventory = load_data()
daily, forecasts, policy, plan = compute_plan(sales, inventory, budget, capacity, demand_shock, lead_time_shock)
agent = InventoryDecisionAgent("docs")
retriever = SimplePolicyRetriever("docs")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("SKUs", plan["sku_id"].nunique())
m2.metric("Reorder signals", int((plan["action"] == "REORDER_NOW").sum()))
m3.metric("Approved orders", int((plan["recommended_order_qty"] > 0).sum()))
m4.metric("High risk", int((plan["stockout_risk"] == "High").sum()))
m5.metric("Order cost", f"₹{plan['optimized_purchase_cost'].sum():,.0f}")

tabs = st.tabs(["Control Tower", "SKU Center", "RAG", "Prompt Lab", "Agent Trace", "Scenario", "Purchase Orders", "Data"])

with tabs[0]:
    st.subheader("Control Tower")
    cols = ["sku_id", "category", "supplier_id", "priority", "current_stock", "lead_time_days", "forecast_lead_time_demand", "safety_stock", "reorder_point", "action", "stockout_risk", "recommended_order_qty", "optimized_purchase_cost"]
    st.dataframe(plan[cols], use_container_width=True, height=430)
    st.download_button("Download recommendations", plan[cols].to_csv(index=False), "reorder_recommendations.csv", "text/csv")
    a, b = st.columns(2)
    a.bar_chart(plan["stockout_risk"].value_counts())
    b.bar_chart(plan.groupby("supplier_id")["optimized_purchase_cost"].sum().sort_values(ascending=False))

with tabs[1]:
    st.subheader("SKU Command Center")
    sku = st.selectbox("SKU", plan["sku_id"].tolist())
    row = plan[plan["sku_id"] == sku].iloc[0].to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current stock", f"{row['current_stock']:.0f}")
    c2.metric("Reorder point", f"{row['reorder_point']:.1f}")
    c3.metric("Recommended qty", int(row["recommended_order_qty"]))
    c4.metric("Risk", row["stockout_risk"])
    st.line_chart(daily[daily["sku_id"] == sku].set_index("date")["quantity_sold"])
    exp = agent.explain_recommendation(row)
    st.markdown("### Agent explanation")
    st.write(exp["answer"])

with tabs[2]:
    st.subheader("RAG Knowledge Base Search")
    q = st.text_area("Policy question", "Which policy applies to high-risk high-priority products with supplier constraints?")
    for r in retriever.retrieve(q, top_k=5):
        with st.expander(f"{r['source']} | score={r.get('score', 0):.3f}"):
            st.write(r["text"])

with tabs[3]:
    st.subheader("LLM-style Prompt Lab")
    sku2 = st.selectbox("SKU for prompt", plan["sku_id"].tolist(), key="prompt_sku")
    row2 = plan[plan["sku_id"] == sku2].iloc[0].to_dict()
    question = st.text_area("Business question", f"Should we reorder {sku2}? Explain why.")
    rag_context = retriever.retrieve(f"{row2.get('category')} {row2.get('priority')} {row2.get('supplier_id')} stockout reorder", top_k=3)
    left, right = st.columns(2)
    left.markdown("### Prompt")
    left.code(build_prompt(row2, rag_context, question), language="text")
    right.markdown("### Generated answer")
    right.write(local_llm_answer(row2, rag_context, question))

with tabs[4]:
    st.subheader("Agent Workflow Trace")
    sku3 = st.selectbox("SKU for trace", plan["sku_id"].tolist(), key="trace_sku")
    r = plan[plan["sku_id"] == sku3].iloc[0]
    trace = [
        ("1. Read inventory", f"Current stock={r.current_stock}, lead time={r.lead_time_days} days."),
        ("2. Forecast demand", f"Lead-time demand={r.forecast_lead_time_demand:.1f}, 30-day forecast={r.forecast_30d:.1f}."),
        ("3. Apply policy", f"Safety stock={r.safety_stock:.1f}, reorder point={r.reorder_point:.1f}, action={r.action}."),
        ("4. Optimize", f"Recommended quantity={int(r.recommended_order_qty)}, cost=₹{r.optimized_purchase_cost:.2f}."),
        ("5. Retrieve context", "RAG retrieves policy chunks from Markdown files in docs/."),
        ("6. Explain", "The agent explains optimizer output without changing the quantity."),
    ]
    for title, detail in trace:
        st.markdown(f"### {title}")
        st.success(detail)

with tabs[5]:
    st.subheader("Scenario Simulator")
    alt_budget = st.number_input("Alternative budget", 1_000, 500_000, int(budget), 1_000)
    alt_capacity = st.number_input("Alternative capacity", 100, 100_000, int(capacity), 500)
    _, _, _, alt = compute_plan(sales, inventory, alt_budget, alt_capacity, demand_shock, lead_time_shock)
    comp = pd.DataFrame([
        {"scenario": "current", "orders": int((plan.recommended_order_qty > 0).sum()), "cost": plan.optimized_purchase_cost.sum(), "units": plan.recommended_order_qty.sum()},
        {"scenario": "alternative", "orders": int((alt.recommended_order_qty > 0).sum()), "cost": alt.optimized_purchase_cost.sum(), "units": alt.recommended_order_qty.sum()},
    ])
    st.dataframe(comp, use_container_width=True)

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
    st.write("Sales rows:", len(sales), "Inventory rows:", len(inventory))
    st.dataframe(sales.head(20), use_container_width=True)
    st.dataframe(inventory.head(20), use_container_width=True)
