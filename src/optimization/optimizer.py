from __future__ import annotations

import math
import pandas as pd


def _round_up_to_pack(quantity: float, pack_size: int) -> int:
    pack_size = max(1, int(pack_size))
    if quantity <= 0:
        return 0
    return int(math.ceil(quantity / pack_size) * pack_size)


def _candidate_quantity(row: pd.Series) -> int:
    if row["action"] != "REORDER_NOW":
        return 0
    required = max(float(row["raw_required_qty"]), float(row["eoq"]))
    required = max(required, float(row["min_order_qty"]))
    required = min(required, float(row["max_order_qty"]))
    return _round_up_to_pack(required, int(row["pack_size"]))


def _greedy_optimize(policy_table: pd.DataFrame, budget: float, capacity_units: float) -> pd.DataFrame:
    df = policy_table.copy()
    df["candidate_order_qty"] = df.apply(_candidate_quantity, axis=1)
    df["candidate_purchase_cost"] = df["candidate_order_qty"] * df["unit_cost"]
    df["candidate_capacity_used"] = df["candidate_order_qty"] * df["storage_units_per_item"]

    shortage_gap = (df["reorder_point"] - df["current_stock"]).clip(lower=0)
    df["priority_score"] = (shortage_gap * df["stockout_cost_per_unit"]) / (df["candidate_purchase_cost"] + 1)

    remaining_budget = float(budget)
    remaining_capacity = float(capacity_units)
    selected = {sku: 0 for sku in df["sku_id"]}

    for _, row in df.sort_values("priority_score", ascending=False).iterrows():
        qty = int(row["candidate_order_qty"])
        if qty <= 0:
            continue
        cost = float(row["candidate_purchase_cost"])
        capacity = float(row["candidate_capacity_used"])
        if cost <= remaining_budget and capacity <= remaining_capacity:
            selected[row["sku_id"]] = qty
            remaining_budget -= cost
            remaining_capacity -= capacity

    df["recommended_order_qty"] = df["sku_id"].map(selected).fillna(0).astype(int)
    return df


def _ortools_optimize(policy_table: pd.DataFrame, budget: float, capacity_units: float) -> pd.DataFrame:
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return _greedy_optimize(policy_table, budget, capacity_units)

    df = policy_table.copy().reset_index(drop=True)
    df["candidate_order_qty"] = df.apply(_candidate_quantity, axis=1)

    model = cp_model.CpModel()
    variables = []
    scale = 100

    for idx, row in df.iterrows():
        max_qty = int(max(0, row["candidate_order_qty"]))
        pack_size = max(1, int(row["pack_size"]))
        var = model.NewIntVar(0, max_qty, f"q_{idx}")
        variables.append(var)
        multiplier = model.NewIntVar(0, max_qty // pack_size if pack_size else max_qty, f"m_{idx}")
        model.Add(var == multiplier * pack_size)
        if row["action"] != "REORDER_NOW":
            model.Add(var == 0)
        else:
            selected = model.NewBoolVar(f"selected_{idx}")
            model.Add(var == 0).OnlyEnforceIf(selected.Not())
            model.Add(var >= int(row["min_order_qty"])).OnlyEnforceIf(selected)

    model.Add(sum(int(row["unit_cost"] * scale) * variables[idx] for idx, row in df.iterrows()) <= int(budget * scale))
    model.Add(sum(int(row["storage_units_per_item"] * scale) * variables[idx] for idx, row in df.iterrows()) <= int(capacity_units * scale))

    objective_terms = []
    for idx, row in df.iterrows():
        shortage_gap = max(0.0, float(row["reorder_point"] - row["current_stock"]))
        benefit_per_unit = min(shortage_gap, float(row["candidate_order_qty"])) * float(row["stockout_cost_per_unit"])
        benefit_coeff = int(max(1, benefit_per_unit) * scale)
        cost_coeff = int((float(row["holding_cost_per_unit"]) + 0.01 * float(row["unit_cost"])) * scale)
        objective_terms.append((benefit_coeff - cost_coeff) * variables[idx])

    model.Maximize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _greedy_optimize(policy_table, budget, capacity_units)

    df["recommended_order_qty"] = [int(solver.Value(v)) for v in variables]
    return df


def optimize_reorders(policy_table: pd.DataFrame, budget: float, capacity_units: float) -> pd.DataFrame:
    optimized = _ortools_optimize(policy_table, budget, capacity_units)
    optimized["optimized_purchase_cost"] = optimized["recommended_order_qty"] * optimized["unit_cost"]
    optimized["optimized_capacity_used"] = optimized["recommended_order_qty"] * optimized["storage_units_per_item"]
    optimized["expected_stock_after_order"] = optimized["current_stock"] + optimized["recommended_order_qty"]
    optimized["post_order_gap_vs_reorder_point"] = optimized["expected_stock_after_order"] - optimized["reorder_point"]
    optimized["final_decision"] = optimized.apply(lambda row: "ORDER_APPROVED" if row["recommended_order_qty"] > 0 else "NO_ORDER", axis=1)
    sort_cols = [col for col in ["action", "stockout_risk", "recommended_order_qty"] if col in optimized.columns]
    ascending = [True if col != "recommended_order_qty" else False for col in sort_cols]
    return optimized.sort_values(sort_cols, ascending=ascending) if sort_cols else optimized
