"""Sales forecasting per merchant and product.

Several model variants are backtested on each merchant's own history and the
best one per product makes the forecast (see docs/FORECASTING.md). Models
retrain on a schedule; between runs, the API nudges a forecast toward the
sales that have come in since (``nowcast``).
"""
