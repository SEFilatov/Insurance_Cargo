from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
from typing import Any, Dict, Tuple


class TariffConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class TariffConfig:
    version: str
    auto_limit_rub: Decimal
    min_premium_rub: Decimal
    base_rates: Dict[str, Dict[str, Decimal]]  # cargo -> condition -> rate (fraction)
    k_franchise: Dict[str, Decimal]            # franchise_rub (string key) -> coeff
    k_reefer: Dict[str, Decimal]              # "true"/"false" -> coeff
    k_route: Dict[str, Decimal]               # route_zone -> coeff
    rounding_mode: str
    rounding_step_rub: int


def _d(x: Any) -> Decimal:
    try:
        return Decimal(str(x))
    except Exception as e:
        raise TariffConfigError(f"Cannot convert to Decimal: {x!r}") from e


def load_config(path: str) -> TariffConfig:
    if not os.path.exists(path):
        raise TariffConfigError(f"Tariff config not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    try:
        version = str(raw["version"])
        auto_limit_rub = _d(raw["auto_limit_rub"])
        min_premium_rub = _d(raw["min_premium_rub"])

        base_rates: Dict[str, Dict[str, Decimal]] = {}
        for cargo_id, by_cond in raw["base_rates"].items():
            base_rates[cargo_id] = {cond: _d(rate) for cond, rate in by_cond.items()}

        k_franchise = {str(k): _d(v) for k, v in raw["k_franchise"].items()}
        k_reefer = {str(k).lower(): _d(v) for k, v in raw["k_reefer"].items()}
        k_route = {str(k): _d(v) for k, v in raw["k_route"].items()}

        rounding = raw.get("rounding", {})
        rounding_mode = str(rounding.get("mode", "HALF_UP")).upper()
        rounding_step_rub = int(rounding.get("step_rub", 1))

    except KeyError as e:
        raise TariffConfigError(f"Missing required key in tariff config: {e}") from e

    if rounding_step_rub <= 0:
        raise TariffConfigError("rounding.step_rub must be > 0")

    return TariffConfig(
        version=version,
        auto_limit_rub=auto_limit_rub,
        min_premium_rub=min_premium_rub,
        base_rates=base_rates,
        k_franchise=k_franchise,
        k_reefer=k_reefer,
        k_route=k_route,
        rounding_mode=rounding_mode,
        rounding_step_rub=rounding_step_rub,
    )


def assess(cfg: TariffConfig, *, cargo_class_id: str, sum_insured_rub: Decimal,
           condition: str, franchise_rub: int, is_reefer: bool, route_zone: str) -> Tuple[str, list[str]]:
    reasons: list[str] = []

    # 1) Cargo whitelist + condition
    if cargo_class_id not in cfg.base_rates:
        return "DECLINE", ["CARGO_NOT_ELIGIBLE"]

    cond = str(condition).upper()
    if cond not in cfg.base_rates[cargo_class_id]:
        return "REFER", ["CONDITION_NOT_SUPPORTED"]

    # 2) Limit
    if _d(sum_insured_rub) > cfg.auto_limit_rub:
        return "REFER", ["LIMIT_EXCEEDED"]

    # 3) Exact buckets for franchise and route
    if str(franchise_rub) not in cfg.k_franchise:
        return "REFER", ["FRANCHISE_NOT_SUPPORTED"]

    if route_zone not in cfg.k_route:
        return "REFER", ["ROUTE_ZONE_NOT_SUPPORTED"]

    k_reefer_key = str(is_reefer).lower()
    if k_reefer_key not in cfg.k_reefer:
        return "REFER", ["REEFER_FLAG_NOT_SUPPORTED"]

    return "AUTO_OK", reasons


def _round_money(amount: Decimal, *, step_rub: int, mode: str) -> Decimal:
    # Round to nearest step (e.g., 1 rub, 10 rub, 100 rub)
    step = Decimal(step_rub)
    if step == 1:
        if mode == "CEIL":
            return amount.to_integral_value(rounding=ROUND_CEILING)
        return amount.to_integral_value(rounding=ROUND_HALF_UP)

    # Scale to step
    scaled = amount / step
    if mode == "CEIL":
        rounded = scaled.to_integral_value(rounding=ROUND_CEILING) * step
    else:
        rounded = scaled.to_integral_value(rounding=ROUND_HALF_UP) * step
    return rounded


def quote(cfg: TariffConfig, *, cargo_class_id: str, sum_insured_rub: Decimal,
          condition: str, franchise_rub: int, is_reefer: bool, route_zone: str) -> Tuple[Decimal, bool]:
    cond = str(condition).upper()
    base_rate = cfg.base_rates[cargo_class_id][cond]
    k_fr = cfg.k_franchise[str(franchise_rub)]
    k_ref = cfg.k_reefer[str(is_reefer).lower()]
    k_route = cfg.k_route[route_zone]

    premium_raw = _d(sum_insured_rub) * base_rate * k_fr * k_ref * k_route
    premium_rounded = _round_money(premium_raw, step_rub=cfg.rounding_step_rub, mode=cfg.rounding_mode)

    min_applied = False
    premium_final = premium_rounded
    if premium_final < cfg.min_premium_rub:
        premium_final = cfg.min_premium_rub
        min_applied = True

    return premium_final, min_applied
