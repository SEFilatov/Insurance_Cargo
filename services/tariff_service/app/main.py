from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException

from .models import HealthResponse, QuoteRequest, QuoteResponse, Decision
from .tariff import TariffConfig, TariffConfigError, load_config, assess, quote


def create_app() -> FastAPI:
    app = FastAPI(title="Cargo Insurance Tariff Engine", version="1.0.0")

    config_path = os.getenv("TARIFF_CONFIG_PATH", "/app/config/tariff_config.json")

    try:
        cfg: TariffConfig = load_config(config_path)
    except TariffConfigError as e:
        # Fail fast: misconfigured container should not serve traffic.
        raise RuntimeError(str(e))

    app.state.tariff_config = cfg

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        cfg2: TariffConfig = app.state.tariff_config
        return HealthResponse(status="ok", tariff_version=cfg2.version)

    @app.post("/quote", response_model=QuoteResponse)
    def post_quote(req: QuoteRequest) -> QuoteResponse:
        cfg2: TariffConfig = app.state.tariff_config

        decision_str, reasons = assess(
            cfg2,
            cargo_class_id=req.cargo_class_id,
            sum_insured_rub=req.sum_insured_rub,
            condition=req.condition.value,
            franchise_rub=req.franchise_rub,
            is_reefer=req.is_reefer,
            route_zone=req.route_zone,
        )

        public_breakdown = {
            "cargo_class_id": req.cargo_class_id,
            "condition": req.condition.value,
            "sum_insured_rub": req.sum_insured_rub,
            "franchise_rub": req.franchise_rub,
            "is_reefer": req.is_reefer,
            "route_zone": req.route_zone,
        }

        if decision_str == "DECLINE":
            return QuoteResponse(
                decision=Decision.DECLINE,
                premium_rub=None,
                min_premium_applied=None,
                tariff_version=cfg2.version,
                public_breakdown=public_breakdown,
                reasons=reasons,
            )

        if decision_str == "REFER":
            return QuoteResponse(
                decision=Decision.REFER,
                premium_rub=None,
                min_premium_applied=None,
                tariff_version=cfg2.version,
                public_breakdown=public_breakdown,
                reasons=reasons,
            )

        # AUTO_OK
        try:
            premium, min_applied = quote(
                cfg2,
                cargo_class_id=req.cargo_class_id,
                sum_insured_rub=req.sum_insured_rub,
                condition=req.condition.value,
                franchise_rub=req.franchise_rub,
                is_reefer=req.is_reefer,
                route_zone=req.route_zone,
            )
        except KeyError:
            # Shouldn't happen if assess() is correct, but just in case.
            raise HTTPException(status_code=400, detail="Unsupported tariff inputs")

        return QuoteResponse(
            decision=Decision.AUTO_OK,
            premium_rub=premium,
            min_premium_applied=min_applied,
            tariff_version=cfg2.version,
            public_breakdown=public_breakdown,
            reasons=reasons,
        )

    return app


app = create_app()
