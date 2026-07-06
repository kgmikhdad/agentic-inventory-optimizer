from src.inventory.policy import calculate_safety_stock, calculate_eoq, classify_stockout_risk


def test_safety_stock_positive():
    value = calculate_safety_stock(daily_demand_std=10, lead_time_days=4, z_value=1.65)
    assert value > 0


def test_eoq_positive():
    value = calculate_eoq(annual_demand=1000, ordering_cost=500, holding_cost_per_unit=2)
    assert value > 0


def test_stockout_risk():
    assert classify_stockout_risk(10, 100) == "High"
    assert classify_stockout_risk(80, 100) == "Medium"
    assert classify_stockout_risk(120, 100) == "Low"
