from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Occupation = Literal["farmer", "student", "street_vendor", "artisan", "entrepreneur",
                     "salaried", "unemployed", "homemaker", "retired", "other"]


class Profile(BaseModel):
    """Citizen details. Every field is optional; missing ones give a 'maybe' result."""
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[Literal["male", "female", "other"]] = None
    residence: Optional[Literal["rural", "urban"]] = None
    state: Optional[str] = None
    category: Optional[Literal["general", "obc", "sc", "st"]] = None
    occupation: Optional[Occupation] = None
    annual_income: Optional[int] = Field(None, ge=0, description="Annual family income in INR")
    is_bpl: Optional[bool] = None
    owns_agri_land: Optional[bool] = None
    has_pucca_house: Optional[bool] = None
    has_bank_account: Optional[bool] = None
    is_income_tax_payer: Optional[bool] = None
    is_govt_employee: Optional[bool] = None
    has_lpg_connection: Optional[bool] = None
    is_pregnant_or_lactating: Optional[bool] = None
    has_girl_child_under_10: Optional[bool] = None


class SchemeOut(BaseModel):
    id: str
    name: str
    short_name: str
    ministry: str
    category: str
    summary: str
    benefits: str
    eligibility: str
    documents: list[str]
    how_to_apply: str
    url: str


class EligibilityOut(BaseModel):
    scheme: SchemeOut
    status: Literal["eligible", "maybe", "not_eligible"]
    met: list[str]
    failed: list[str]
    missing: list[str]


class CheckResponse(BaseModel):
    eligible: list[EligibilityOut]
    maybe: list[EligibilityOut]
    not_eligible_count: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    profile: Optional[Profile] = None
    k: int = Field(5, ge=1, le=10)


class Source(BaseModel):
    scheme_id: str
    scheme_name: str
    section: str
    text: str
    url: str
    score: float
    eligibility: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    mode: Literal["llm", "extractive"]
    sources: list[Source]
