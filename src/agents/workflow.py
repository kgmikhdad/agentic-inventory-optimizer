from __future__ import annotations

from typing import Dict, Any

from src.rag.simple_rag import SimplePolicyRetriever


class InventoryDecisionAgent:
    """Controlled agent workflow for inventory recommendations.

    The optimizer makes the numerical decision. This agent explains the decision,
    retrieves relevant policy context, and highlights risks.
    """

    def __init__(self, policy_docs_dir: str = "docs") -> None:
        self.retriever = SimplePolicyRetriever(policy_docs_dir)

    def explain_recommendation(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = (
            f"Inventory reorder policy for {row.get('category')} products with "
            f"{row.get('priority')} priority supplier {row.get('supplier_id')} service level "
            f"minimum order quantity pack size stockout risk"
        )
        context = self.retriever.retrieve(query, top_k=3)

        sku_id = row.get("sku_id")
        action = row.get("action")
        recommended_qty = int(row.get("recommended_order_qty", 0))
        current_stock = float(row.get("current_stock", 0))
        reorder_point = float(row.get("reorder_point", 0))
        lead_time_demand = float(row.get("forecast_lead_time_demand", 0))
        safety_stock = float(row.get("safety_stock", 0))
        risk = row.get("stockout_risk", "Unknown")
        category = row.get("category", "unknown")
        priority = row.get("priority", "unknown")
        supplier_id = row.get("supplier_id", "unknown supplier")

        if recommended_qty > 0:
            answer = (
                f"{sku_id} should be reordered now. The optimizer recommends ordering "
                f"{recommended_qty} units. Current stock is {current_stock:.0f}, which is below "
                f"the reorder point of {reorder_point:.1f}. Expected lead-time demand is "
                f"{lead_time_demand:.1f} units and safety stock is {safety_stock:.1f} units. "
                f"The stockout risk is {risk}. This item belongs to the {category} category "
                f"with {priority} priority and is supplied by {supplier_id}."
            )
        elif action == "REORDER_NOW":
            answer = (
                f"{sku_id} is below its reorder point, but no order was selected in the optimized plan. "
                f"This usually means the budget or warehouse capacity constraint forced the optimizer "
                f"to prioritize other SKUs with higher expected stockout impact. Current stock is "
                f"{current_stock:.0f}, reorder point is {reorder_point:.1f}, and stockout risk is {risk}."
            )
        else:
            answer = (
                f"{sku_id} does not need immediate reordering. Current stock is {current_stock:.0f}, "
                f"which is above the reorder point of {reorder_point:.1f}. Expected lead-time demand is "
                f"{lead_time_demand:.1f} units and stockout risk is {risk}."
            )

        if context:
            policy_summary = " ".join(item["text"] for item in context[:2])
            answer += f" Relevant policy context: {policy_summary[:500]}"

        return {"answer": answer, "retrieved_context": context}
