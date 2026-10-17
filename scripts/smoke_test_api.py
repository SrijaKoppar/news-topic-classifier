"""API smoke test (runs the app in-process, no server needed).

Usage:  python -m scripts.smoke_test_api
"""
from fastapi.testclient import TestClient

from src.api import app

with TestClient(app) as client:  # the `with` runs the startup loading
    r = client.post("/predict", json={
        "headline": "Senate passes new budget bill after long debate",
        "short_description": "Lawmakers voted late on Thursday."})
    assert r.status_code == 200, r.text
    body = r.json()
    print("valid request   ->", body)
    assert set(body) == {"category", "confidence", "abstain"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert (body["category"] is None) == body["abstain"]

    r = client.post("/predict", json={"headline": "Quarterly earnings beat forecasts"})
    assert r.status_code == 200, "short_description must be optional"
    print("no description  ->", r.json())

    # Gibberish should be low confidence; print it, but don't assert on it.
    print("gibberish       ->", client.post("/predict", json={"headline": "zxq vlorp"}).json())

    assert client.post("/predict", json={"headline": "   "}).status_code == 422
    assert client.post("/predict", json={"short_description": "x"}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    print("invalid input   -> 422 for blank, missing and empty bodies")
print("ALL API CHECKS PASSED")