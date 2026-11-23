from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["schemes"] >= 15


def test_list_and_get_scheme():
    schemes = client.get("/api/schemes").json()
    assert any(s["id"] == "pm-kisan" for s in schemes)
    assert client.get("/api/schemes/pm-kisan").json()["short_name"] == "PM-KISAN"
    assert client.get("/api/schemes/does-not-exist").status_code == 404


def test_filter_by_category():
    schemes = client.get("/api/schemes", params={"category": "insurance"}).json()
    assert schemes and all(s["category"] == "Insurance" for s in schemes)


def test_check_profile():
    profile = {"age": 34, "gender": "female", "residence": "rural", "occupation": "farmer",
               "is_bpl": True, "owns_agri_land": True, "has_bank_account": True,
               "is_income_tax_payer": False, "is_govt_employee": False,
               "has_lpg_connection": False, "has_pucca_house": False}
    data = client.post("/api/check", json=profile).json()
    eligible = {e["scheme"]["id"] for e in data["eligible"]}
    assert {"pm-kisan", "pmuy", "pmay-g", "apy"} <= eligible
    assert all(e["met"] for e in data["eligible"])


def test_check_rejects_bad_input():
    assert client.post("/api/check", json={"age": -5}).status_code == 422


def test_ask_returns_sources_and_eligibility():
    body = {"question": "free gas connection", "profile": {"gender": "male"}}
    data = client.post("/api/ask", json=body).json()
    assert data["mode"] == "extractive"
    assert data["sources"][0]["scheme_id"] == "pmuy"
    assert data["sources"][0]["eligibility"] == "not_eligible"
    assert "Ujjwala" in data["answer"]


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "Sahaayak" in r.text
