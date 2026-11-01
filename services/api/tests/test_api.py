from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)

ARTICLE = (
    "The city council of Naples approved a new metro line with 12 stations on 4 May 2024.\n"
    "Officials said the project will cost 1.2 billion euros and should be completed by 2030 according to the plan.\n"
)


async def fake_search(query, num=5, use_cache=True):
    return [
        {"title": "a", "link": "https://www.example-news.com/a", "snippet": "Naples council approved a metro line with 12 stations, officials confirmed on Monday."},
        {"title": "b", "link": "https://www.ansa.it/b", "snippet": "The new Naples metro line will have 12 stations and cost 1.2 billion euros, the council said."},
        {"title": "c", "link": "https://www.msn.com/c", "snippet": "Republished: Naples council approved a metro line with 12 stations."},
    ]


def test_health_and_source():
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/source/netflix.com").json()
    assert r["score"] == 50 and r["label"] == "MEDIUM"


def test_analyze_requires_input():
    assert client.post("/analyze", json={}).status_code == 400


def test_analyze_text_with_debate(monkeypatch):
    monkeypatch.setattr(main, "serpapi_search", fake_search)
    r = client.post("/analyze", json={"text": ARTICLE, "verifier": "debate", "enable_weighted_confidence": True})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["claims"], data
    c = data["claims"][0]
    assert c["verdict_source"] == "debate"
    assert c["debate"]["formal_argument"] is not None
    assert any(x["reason"] == "aggregator_or_syndication" for x in c["excluded_evidence"])
    assert c["adjusted_verdict"] in ("SUPPORTED", "REFUTED", "NEI")
    assert data["metadata"]["llm_usage"]["llm_calls"] == 3 * len(data["claims"])

    run_id = data["metadata"]["run_id"]
    saved = client.get(f"/runs/{run_id}").json()
    assert saved["response"]["claims"][0]["claim_text"] == c["claim_text"]
    assert client.get("/runs").json()["items"][0]["created_utc"]
    csv_text = client.get("/runs-export?format=csv").text
    assert csv_text.startswith("run_id,") and c["claim_text"][:20] in csv_text


def test_cors_allows_any_local_port():
    for origin in ("http://localhost:3001", "http://127.0.0.1:3005", "http://localhost:3000"):
        r = client.get("/health", headers={"Origin": origin})
        assert r.headers.get("access-control-allow-origin") == origin
    r = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in r.headers


def test_short_claim_is_analysed(monkeypatch):
    monkeypatch.setattr(main, "serpapi_search", fake_search)
    data = client.post("/analyze", json={"text": "Paris is capital of france", "verifier": "debate"}).json()
    assert [c["claim_text"] for c in data["claims"]] == ["Paris is capital of france"]
    assert data["final_misinformation_likelihood"] is not None


def test_no_claims_gives_no_score(monkeypatch):
    monkeypatch.setattr(main, "serpapi_search", fake_search)
    data = client.post("/analyze", json={"text": "hello ok", "verifier": "baseline"}).json()
    assert data["claims"] == [] and data["final_misinformation_likelihood"] is None
    assert data["metadata"]["warning"]
