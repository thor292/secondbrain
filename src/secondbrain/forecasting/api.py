"""Forecast endpoints. Reads run as the signed-in user, so row-level security
decides whose sales and forecasts they see."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request

from secondbrain.forecasting import nowcast, store
from secondbrain.service import SecondBrain, UnknownUser

UserId = Annotated[str, Header(alias="X-User-Id")]


def build_router() -> APIRouter:
    router = APIRouter(tags=["forecasting"])

    def brain_for(request: Request, user_id: str) -> SecondBrain:
        brain = request.app.state.brain
        if brain.store.get_user(user_id) is None:
            raise UnknownUser(user_id)
        return brain

    @router.get("/forecast/products")
    def products(request: Request, user_id: UserId):
        """Products whose sales this user may see."""
        with brain_for(request, user_id).store.as_user(user_id) as conn:
            return store.visible_products(conn)

    @router.get("/forecast")
    def forecast(
        request: Request,
        user_id: UserId,
        product: Annotated[str, Query(description="SKU or part of the product name")],
        horizon: Annotated[int, Query(ge=1, le=366)] = 28,
        history: Annotated[int, Query(ge=0, le=366, description="days of past sales to include")] = 28,
        brain: Annotated[str | None, Query(description="brain path, when a SKU exists in several")] = None,
    ):
        with brain_for(request, user_id).store.as_user(user_id) as conn:
            matches = store.find_products(conn, product, brain)
            if not matches:
                raise HTTPException(404, f"No product matching {product!r} that you can see")
            exact = [m for m in matches if m["sku"] == product]
            if len(exact) != 1 and len(matches) > 1:
                options = ", ".join(f"{m['sku']} ({m['brain']})" for m in matches)
                raise HTTPException(400, f"{product!r} matches several products: {options}")
            match = (exact or matches)[0]
            stored = store.latest_forecast(conn, match["brain"], match["sku"])
            if stored is None:
                raise HTTPException(404, f"No forecast for {match['sku']} yet; run python -m secondbrain.forecasting")
            cutoff = stored["cutoff"]
            sales = store.recent_sales(conn, match["brain"], match["sku"], cutoff - timedelta(days=history - 1))

        actuals = {s["day"]: s["units"] for s in sales if s["day"] > cutoff and s["in_stock"]}
        ahead, adjustment = nowcast.adjust(stored["points"], actuals)
        if horizon > len(ahead):
            raise HTTPException(400, f"Forecasts reach {len(ahead)} days ahead; ask for fewer")
        chosen = stored["backtest"][stored["model"]]
        return {
            "product": match,
            "model": stored["model"],
            "reason": stored["reason"],
            "trained_through": cutoff,
            "created_at": stored["created_at"],
            "backtest": stored["backtest"],
            "expected_error": {"wape": chosen["wape"], "mase": chosen["mase"]},
            "adjustment": vars(adjustment),
            "forecast": ahead[:horizon],
            "history": [{"day": s["day"], "units": s["units"], "in_stock": s["in_stock"]}
                        for s in sales if history],
        }

    return router
