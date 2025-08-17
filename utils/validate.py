# utils/validate.py
REQUIRED = [
    "created_at","actual_delivery_time",
    "total_items","subtotal","num_distinct_items",
    "min_item_price","max_item_price",
    "total_onshift_dashers","total_busy_dashers","total_outstanding_orders",
    "estimated_order_place_duration","estimated_store_to_consumer_driving_duration",
    "market_id","order_protocol"
]

def basic_check(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    return missing

