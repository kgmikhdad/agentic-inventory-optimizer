# Implementation Plan

## Objective

Build a practical agentic inventory and reorder optimization system that forecasts demand, identifies risky SKUs, optimizes reorder quantities, and explains decisions using policy retrieval.

## Phase 1: Data foundation

- Use the included synthetic data generator first.
- Later replace synthetic data with real sales history.
- Required sales columns: `date`, `sku_id`, `quantity_sold`, `unit_price`, `category`.
- Required inventory columns: `sku_id`, `current_stock`, `lead_time_days`, `unit_cost`, `holding_cost_per_unit`, `ordering_cost`, `stockout_cost_per_unit`, `min_order_qty`, `max_order_qty`, `pack_size`, `storage_units_per_item`, `priority`.

## Phase 2: Forecasting

- Start with weighted moving average.
- Evaluate with train/test split later.
- Extend with AutoARIMA, ETS, LightGBM, Prophet, or Darts.

## Phase 3: Inventory policy

- Calculate lead-time demand.
- Calculate safety stock.
- Calculate reorder point.
- Classify stockout risk.
- Create `REORDER_NOW` and `SAFE` actions.

## Phase 4: Optimization

Optimize reorder quantity under:

- budget constraint,
- capacity constraint,
- minimum order quantity,
- maximum order quantity,
- pack-size multiple.

## Phase 5: RAG and agent layer

- Store policy documents in `docs/`.
- Retrieve relevant policy chunks using `SimplePolicyRetriever`.
- Generate decision explanation using `InventoryDecisionAgent`.

## Phase 6: Dashboard

- Show summary metrics.
- Show reorder table.
- Show SKU-level explanation.
- Show demand history.
- Add what-if budget/capacity controls.

## Phase 7: Final report

Include problem statement, literature review, architecture, dataset, methodology, forecasting formulas, optimization model, RAG workflow, results, screenshots, limitations, future scope, and references.
