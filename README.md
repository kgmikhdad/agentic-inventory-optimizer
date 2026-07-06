# Agentic Inventory Optimizer

An AI-powered inventory and reorder optimization system using demand forecasting, RAG-style policy retrieval, agentic decision workflows, and operations research optimization.

## What it does

The system helps answer:

> Which products should I reorder, how much should I reorder, and why?

It combines:

1. SKU-level demand forecasting.
2. Safety stock, reorder point, and EOQ calculations.
3. OR-based reorder optimization under budget, capacity, MOQ, maximum order, and pack-size constraints.
4. RAG-style policy retrieval from local inventory, supplier, and service-level documents.
5. Agentic recommendation explanation.
6. Streamlit dashboard for demonstration.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Main project flow

```text
sales data + inventory master
        -> preprocessing
        -> demand forecasting
        -> inventory policy calculations
        -> OR optimization
        -> RAG policy retrieval
        -> agent explanation
        -> Streamlit dashboard
```

## Main outputs

Running `python run_pipeline.py` creates:

```text
reports/reorder_recommendations.csv
```

The dashboard shows:

- risky SKUs,
- reorder quantities,
- reorder point,
- safety stock,
- forecast lead-time demand,
- budget/capacity what-if analysis,
- policy-grounded agent explanation.

## Academic positioning

This project sits at the intersection of supply chain analytics, operations research, demand forecasting, RAG systems, and AI agents for decision support.

## Important note

The base version uses generated synthetic data so the system runs immediately. For final submission, add one real dataset experiment such as UCI Online Retail, M5 retail forecasting data, or a Kaggle inventory/sales dataset.
