# What You Should Do From Your Side

This repository is a complete runnable starter project. Your work is to make it stronger, personalize it, and prepare it for final submission.

## 1. GitHub setup

- Confirm the repository name is `agentic-inventory-optimizer`.
- Add the description: `AI-powered inventory and reorder optimization using forecasting, RAG, agents, and operations research.`
- Add topics: `supply-chain`, `inventory-optimization`, `demand-forecasting`, `rag`, `ai-agent`, `operations-research`, `streamlit`, `ortools`.
- Keep the repository public if you want to show it on your resume or LinkedIn.

## 2. Local setup

Run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## 3. Replace sample data later

The current project uses synthetic data. For final submission, add at least one real dataset experiment.

Recommended options:

- UCI Online Retail dataset.
- M5 retail forecasting dataset.
- Any store/inventory dataset from Kaggle.

Map the real dataset into:

```text
date, sku_id, quantity_sold, unit_price, category
```

Then create or simulate an inventory master file with:

```text
sku_id, current_stock, lead_time_days, unit_cost, holding_cost_per_unit,
ordering_cost, stockout_cost_per_unit, min_order_qty, max_order_qty,
pack_size, storage_units_per_item, priority
```

## 4. Strengthen forecasting

The current version uses weighted moving average. For a stronger final-year project, add one advanced model:

- AutoARIMA or ETS using StatsForecast,
- LightGBM/XGBoost using lag features,
- Prophet,
- Darts forecasting models.

Minimum evaluation table:

```text
Model | MAE | RMSE | sMAPE
Naive
Moving Average
Advanced Model
```

## 5. Strengthen optimization

Add one or more:

- supplier-level budget constraints,
- per-category service-level constraints,
- expiry constraints,
- multi-warehouse capacity,
- minimum fill-rate constraints,
- purchase-order batching by supplier.

## 6. Improve RAG

The current RAG module uses local TF-IDF retrieval so the project runs without paid APIs. For a stronger AI project, upgrade to:

- LlamaIndex + ChromaDB,
- LangChain + FAISS,
- OpenAI embeddings,
- local sentence-transformer embeddings.

Keep TF-IDF as fallback.

## 7. Final report sections

Your report should include abstract, problem statement, literature review, architecture, dataset, methodology, forecasting formulas, inventory-control formulas, OR optimization model, RAG and agent workflow, results, screenshots, limitations, future scope, and references.

## 8. Viva topics

Be ready to explain demand forecasting, safety stock, reorder point, EOQ, OR optimization, RAG, what makes the system agentic, and why the agent should explain decisions rather than invent reorder quantities.
