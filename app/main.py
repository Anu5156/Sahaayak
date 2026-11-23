"""Sahaayak - government scheme eligibility assistant (FastAPI app)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .answer import answer
from .models import (AskRequest, AskResponse, CheckResponse, EligibilityOut, Profile,
                     SchemeOut, Source)
from .retrieval import HybridRetriever
from .rules import check_all, check_scheme

BASE = Path(__file__).resolve().parent
STATIC = BASE.parent / "static"


@lru_cache
def load_schemes() -> list[dict]:
    with open(BASE / "data" / "schemes.json", encoding="utf-8") as f:
        return json.load(f)["schemes"]


@lru_cache
def retriever() -> HybridRetriever:
    return HybridRetriever(load_schemes())


def scheme_by_id() -> dict[str, dict]:
    return {s["id"]: s for s in load_schemes()}


def to_scheme_out(s: dict) -> SchemeOut:
    return SchemeOut(**{k: s[k] for k in SchemeOut.model_fields})


app = FastAPI(
    title="Sahaayak API",
    description="Find government schemes you are eligible for, with clear reasons, "
                "and ask questions answered from official scheme information.",
    version="1.0.0",
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "schemes": len(load_schemes()), "retrieval": retriever().backend}


@app.get("/api/schemes", response_model=list[SchemeOut])
def list_schemes(category: str | None = None) -> list[SchemeOut]:
    schemes = load_schemes()
    if category:
        schemes = [s for s in schemes if s["category"].lower() == category.lower()]
    return [to_scheme_out(s) for s in schemes]


@app.get("/api/schemes/{scheme_id}", response_model=SchemeOut)
def get_scheme(scheme_id: str) -> SchemeOut:
    s = scheme_by_id().get(scheme_id)
    if not s:
        raise HTTPException(status_code=404, detail="Scheme not found")
    return to_scheme_out(s)


@app.post("/api/check", response_model=CheckResponse)
def check(profile: Profile) -> CheckResponse:
    lookup = scheme_by_id()
    results = check_all(load_schemes(), profile.model_dump())
    out = [EligibilityOut(scheme=to_scheme_out(lookup[r.scheme_id]), status=r.status,
                          met=r.met, failed=r.failed, missing=r.missing) for r in results]
    return CheckResponse(
        eligible=[o for o in out if o.status == "eligible"],
        maybe=[o for o in out if o.status == "maybe"],
        not_eligible_count=sum(o.status == "not_eligible" for o in out),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    hits = retriever().search(req.question, k=req.k)
    lookup = scheme_by_id()
    status = {}
    if req.profile:
        profile = {k: v for k, v in req.profile.model_dump().items() if v is not None}
        status = {h.chunk.scheme_id: check_scheme(lookup[h.chunk.scheme_id], profile).status
                  for h in hits}
    text, mode = answer(req.question, hits)
    sources = [Source(scheme_id=h.chunk.scheme_id, scheme_name=h.chunk.scheme_name,
                      section=h.chunk.title, text=h.chunk.text, url=h.chunk.url,
                      score=round(h.score, 3), eligibility=status.get(h.chunk.scheme_id))
               for h in hits]
    return AskResponse(answer=text, mode=mode, sources=sources)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
