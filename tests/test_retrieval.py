import pytest

from app.main import load_schemes
from app.retrieval import HybridRetriever, expand_query


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(load_schemes())


@pytest.mark.parametrize("query, expected", [
    ("free gas connection for women", "pmuy"),
    ("kisan ko paisa kaise milega", "pm-kisan"),
    ("scholarship for SC students", "post-matric-sc"),
    ("pension for old people", "ignoaps"),
    ("ghar banane ke liye madad", "pmay-g"),
    ("documents for ayushman card", "ab-pmjay"),
    ("loan for street vendor cart", "pm-svanidhi"),
    ("savings account for my daughter", "ssy"),
])
def test_top_hit(retriever, query, expected):
    assert retriever.search(query, k=1)[0].chunk.scheme_id == expected


def test_typo_tolerance(retriever):
    assert retriever.search("ujjwla gas", k=1)[0].chunk.scheme_id == "pmuy"


def test_scheme_filter(retriever):
    hits = retriever.search("insurance", k=5, scheme_ids={"pmsby"})
    assert hits and all(h.chunk.scheme_id == "pmsby" for h in hits)


def test_scores_are_fused_and_sorted(retriever):
    hits = retriever.search("life insurance", k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_synonym_expansion():
    assert "farmer" in expand_query("kisan yojana")
