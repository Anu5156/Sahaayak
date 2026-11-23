from app.main import load_schemes
from app.rules import check_all, check_scheme, evaluate_rule

SCHEMES = {s["id"]: s for s in load_schemes()}


def test_simple_rule_pass_fail_unknown():
    rule = {"field": "age", "op": "gte", "value": 18, "reason": "18+"}
    assert evaluate_rule(rule, {"age": 20})[0] == "pass"
    assert evaluate_rule(rule, {"age": 10})[0] == "fail"
    assert evaluate_rule(rule, {})[0] == "unknown"


def test_any_group():
    rule = {"any": [{"field": "age", "op": "gte", "value": 70, "reason": "70+"},
                    {"field": "is_bpl", "op": "is_true", "reason": "BPL"}], "reason": "70+ or BPL"}
    assert evaluate_rule(rule, {"age": 72})[0] == "pass"
    assert evaluate_rule(rule, {"age": 30, "is_bpl": True})[0] == "pass"
    assert evaluate_rule(rule, {"age": 30, "is_bpl": False})[0] == "fail"
    assert evaluate_rule(rule, {"age": 30})[0] == "unknown"


def test_pm_kisan_farmer_eligible():
    profile = {"occupation": "farmer", "owns_agri_land": True,
               "is_income_tax_payer": False, "is_govt_employee": False}
    assert check_scheme(SCHEMES["pm-kisan"], profile).status == "eligible"


def test_pm_kisan_tax_payer_excluded():
    profile = {"occupation": "farmer", "owns_agri_land": True,
               "is_income_tax_payer": True, "is_govt_employee": False}
    result = check_scheme(SCHEMES["pm-kisan"], profile)
    assert result.status == "not_eligible"
    assert "must not be an income tax payer" in result.failed


def test_missing_fields_give_maybe():
    result = check_scheme(SCHEMES["apy"], {"age": 25})
    assert result.status == "maybe"
    assert len(result.missing) == 2


def test_apy_age_boundaries():
    base = {"has_bank_account": True, "is_income_tax_payer": False}
    assert check_scheme(SCHEMES["apy"], {**base, "age": 18}).status == "eligible"
    assert check_scheme(SCHEMES["apy"], {**base, "age": 40}).status == "eligible"
    assert check_scheme(SCHEMES["apy"], {**base, "age": 41}).status == "not_eligible"


def test_check_all_orders_eligible_first():
    results = check_all(load_schemes(), {"age": 72, "is_bpl": True, "has_bank_account": True})
    statuses = [r.status for r in results]
    assert statuses == sorted(statuses, key=["eligible", "maybe", "not_eligible"].index)
    assert "ignoaps" in [r.scheme_id for r in results if r.status == "eligible"]


def test_every_scheme_is_well_formed():
    required = {"id", "name", "short_name", "ministry", "category", "summary", "benefits",
                "eligibility", "documents", "how_to_apply", "url", "rules"}
    for s in load_schemes():
        assert required <= s.keys(), s["id"]
        assert s["url"].startswith("https://")
        for rule in s["rules"]:
            evaluate_rule(rule, {})  # must not raise
