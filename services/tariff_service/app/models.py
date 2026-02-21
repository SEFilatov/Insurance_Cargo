from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, condecimal, conint


class Condition(str, Enum):
    NEW = "NEW"
    USED = "USED"


class Decision(str, Enum):
    AUTO_OK = "AUTO_OK"
    REFER = "REFER"
    DECLINE = "DECLINE"


class QuoteRequest(BaseModel):
    cargo_class_id: str = Field(..., min_length=3, max_length=64)
    sum_insured_rub: condecimal(gt=0, max_digits=14, decimal_places=2) = Field(
        ...,
        description="Insured sum in RUB, supports kopecks",
    )
    condition: Condition
    franchise_rub: conint(ge=0, le=10**9)
    is_reefer: bool
    route_zone: str = Field(..., min_length=1, max_length=64)


class QuoteResponse(BaseModel):
    decision: Decision
    premium_rub: Optional[int] = None
    min_premium_applied: Optional[bool] = None
    tariff_version: str
    public_breakdown: Dict[str, Any]
    reasons: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    tariff_version: str
