"""
API tests — exercise the FastAPI server against the offline demo pipeline.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402

client = TestClient(app)


def test_config_endpoint():
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "tickers" in data and data["tickers"]
    assert "presets" in data and "defaults" in data
    assert isinstance(data["finbert_available"], bool)


def test_run_demo():
    res = client.post(
        "/api/run",
        json={"mode": "demo", "tickers": ["AAPL", "MSFT"], "engine": "lexicon"},
    )
    assert res.status_code == 200, res.text
    data = res.json()

    # metrics
    m = data["metrics"]
    assert m["n_days"] > 0
    assert m["trades"] >= 0
    assert m["total_return"] is not None

    # perf arrays line up
    perf = data["perf"]
    n = len(perf["date"])
    assert n == m["n_days"]
    assert len(perf["equity"]) == n
    assert len(perf["bench_equity"]) == n
    assert len(perf["drawdown"]) == n

    # headlines scored with labels
    assert data["headlines"], "demo should always produce headlines"
    assert {"ticker", "ts", "headline", "sent_label", "sent_score"} <= set(data["headlines"][0])

    # topology payload per requested ticker
    assert set(data["topology"]) == {"AAPL", "MSFT"}
    aapl = data["topology"]["AAPL"]
    assert len(aapl["returns"]) > 10
    assert len(aapl["dates"]) == len(aapl["topo"])

    # meta reports the engine that actually ran
    assert data["meta"]["engine_used"] == "lexicon"
    assert data["meta"]["mode"] == "demo"


def test_run_demo_is_deterministic():
    body = {"mode": "demo", "tickers": ["NVDA"], "engine": "lexicon"}
    a = client.post("/api/run", json=body).json()
    b = client.post("/api/run", json=body).json()
    assert a["metrics"]["total_return"] == b["metrics"]["total_return"]
    assert a["perf"]["equity"] == b["perf"]["equity"]


def test_run_rejects_empty_tickers():
    res = client.post("/api/run", json={"mode": "demo", "tickers": []})
    assert res.status_code == 422


def test_run_rejects_bad_threshold():
    res = client.post("/api/run", json={"mode": "demo", "tickers": ["AAPL"], "threshold": 0.99})
    assert res.status_code == 422
