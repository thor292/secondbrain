"""Backtest the models, pick the best per product and store forecasts.

A system job on the admin connection, meant to run on a schedule (nightly):

    uv run python -m secondbrain.forecasting
    uv run python -m secondbrain.forecasting --under root.desiliving --horizon 56
"""

import argparse

import psycopg

from secondbrain.config import get_settings
from secondbrain.forecasting import backtest, store


def run(admin_database_url: str, under: str = "root", horizon: int = backtest.DEFAULT_HORIZON) -> dict:
    """Forecast every product with sales history under a brain; returns results per brain."""
    results = {}
    with psycopg.connect(admin_database_url) as conn:
        store.migrate(conn)
        for brain in store.series_brains(conn, under):
            sales = store.load_sales(conn, brain)
            if not sales:  # products registered, no sales yet
                continue
            forecasts = backtest.run(sales, store.load_promotions(conn, brain), horizon)
            cutoff = max(r["day"] for r in sales)
            for f in forecasts:
                store.save_forecast(conn, brain, f.sku, cutoff, f.model, f.reason, f.points, f.scores, f.quality)
            results[brain] = forecasts
    return results


def print_scores(results: dict) -> None:
    for brain, forecasts in results.items():
        models = list(dict.fromkeys(m for f in forecasts for m in f.scores))
        print(f"\n{brain}: average WAPE per model over the backtest windows (* = chosen)\n")
        print(f"{'product':<16}" + "".join(f"{m:>19}" for m in models) + "   MASE of chosen  grade")
        for f in forecasts:
            cells = "".join(_cell(f.scores.get(m, {}).get("wape")) + ("*" if m == f.model else " ") for m in models)
            mase = f.scores[f.model].get("mase")
            print(f"{f.sku:<16}{cells}   {mase if mase is None else round(mase, 2)!s:>15}  {f.quality.get('grade')}")


def _cell(wape: float | None) -> str:
    return f"{wape:>18.1%}" if wape is not None else f"{'-':>18}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--under", default="root", help="only brains in this subtree, e.g. root.desiliving")
    parser.add_argument("--horizon", type=int, default=backtest.DEFAULT_HORIZON, help="days to forecast")
    args = parser.parse_args()
    print_scores(run(get_settings().admin_database_url, args.under, args.horizon))


if __name__ == "__main__":
    main()
