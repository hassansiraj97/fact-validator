import asyncio

from app import llm
from app.credibility import compute_score
from app.evidence import LABELS_3, LABELS_4, filter_search_results, normalize_label
from app.text_utils import extract_claim_candidates, extract_domain
from app.verifiers import heuristic_verdict, llm_cot_sc, run_method


# ---------------- credibility ----------------

def test_platform_penalty_uses_domain_match_not_substring():
    assert compute_score("netflix.com").score == 50          # used to be penalised as "x.com"
    assert compute_score("fox.com").score == 50
    assert compute_score("x.com").score == 30
    assert compute_score("mobile.twitter.com").score == 30


def test_curated_subdomains_are_recognised():
    assert "curated_list" in compute_score("ec.europa.eu").reasons
    assert compute_score("news.bbc.co.uk").score == compute_score("bbc.co.uk").score == 80


def test_no_double_counting_tld_and_curated():
    assert compute_score("nih.gov").score == 85               # 50 + max(35, 35), not 100
    assert compute_score("epa.gov").score == 85               # TLD bonus only


# ---------------- text / claims ----------------

def test_archive_urls_are_unwrapped():
    assert extract_domain("https://web.archive.org/web/20201129/https://www.scoopertino.com/x") == "scoopertino.com"


def test_dates_are_kept_in_claims():
    text = ("The government announced on March 3, 2020 that 2,500 new hospital beds would be opened in the city by "
            "the end of the year.\nSubscribe to our newsletter for more news and updates every single day of the week.")
    claims = extract_claim_candidates(text, 5)
    assert any("March 3, 2020" in c for c in claims)
    assert not any("newsletter" in c.lower() for c in claims)


# ---------------- evidence filtering ----------------

def test_circular_and_syndicated_evidence_is_excluded():
    claim = "The new bridge across the river in Naples will be finished in 2027 according to the city council"
    raw = [
        {"link": "https://www.bbc.com/news/1", "snippet": "The city council said the bridge in Naples will open in 2027 after delays."},
        {"link": "https://www.msn.com/en-gb/news/x", "snippet": "The city council said the bridge in Naples will open in 2027 after delays."},
        {"link": "https://www.reuters.com/a", "snippet": claim + " officials said on Monday."},
        {"link": "https://www.ansa.it/b", "snippet": "Naples city council confirms bridge schedule, completion expected in 2027."},
    ]
    kept, excluded = filter_search_results(claim, raw, input_domain="bbc.com")
    reasons = {e["domain"]: e["excluded_reason"] for e in excluded}
    assert reasons["bbc.com"] == "same_site_as_input"
    assert reasons["msn.com"] == "aggregator_or_syndication"
    assert reasons["reuters.com"] == "near_verbatim_copy_of_claim"
    assert [e["domain"] for e in kept] == ["ansa.it"]


def test_label_normalisation():
    assert normalize_label("Refutes") == "REFUTED"
    assert normalize_label("Not Enough Evidence") == "NEI"
    assert normalize_label("Conflicting Evidence/Cherrypicking", LABELS_4) == "CONFLICTING"
    assert normalize_label("Conflicting Evidence/Cherrypicking", LABELS_3) == "NEI"
    assert normalize_label("banana") == "NEI"


# ---------------- verifiers ----------------

def _ev(snippet, domain="reuters.com", score=80, overlap=5):
    return {"snippet": snippet, "domain": domain, "base_domain": domain, "domain_score": score, "overlap": overlap, "url": f"https://{domain}"}


def test_refutation_requires_relevant_evidence():
    claim = "The vaccine contains microchips"
    unrelated = [_ev("This rumour about football transfers is false.", overlap=0)]
    assert heuristic_verdict(claim, unrelated).verdict == "NEI"
    related = [_ev("Claims that the vaccine contains microchips are false.", overlap=4)]
    assert heuristic_verdict(claim, related).verdict == "REFUTED"


def test_fallback_without_evidence_makes_no_llm_call():
    with llm.track_usage() as u:
        r = asyncio.run(run_method("debate", "Some claim", [], LABELS_3))
    assert r.verdict == "NEI" and r.source == "fallback_insufficient_evidence" and u.calls == 0


def test_low_credibility_evidence_filtered_for_debate_only_with_credibility():
    ev = [_ev("Some text about the claim here", domain="facebook.com", score=30)]
    r1 = asyncio.run(run_method("debate", "Some claim", ev, LABELS_3))
    r2 = asyncio.run(run_method("debate_no_cred", "Some claim", ev, LABELS_3))
    assert r1.source == "fallback_insufficient_evidence"
    assert r2.source == "debate_full"


def test_debate_makes_three_calls_and_skeptic_prompt_is_text():
    seen = []
    orig = llm._mock_reply

    def spy(msgs, jm, t):
        seen.append(msgs[-1]["content"])
        return orig(msgs, jm, t)

    llm._mock_reply = spy
    try:
        with llm.track_usage() as u:
            asyncio.run(run_method("debate", "Claim", [_ev("Evidence snippet text about the claim")], LABELS_3))
    finally:
        llm._mock_reply = orig
    assert u.calls == 3
    assert isinstance(seen[1], str) and seen[1].startswith("Role: SKEPTIC") and "('" not in seen[1][:5]


def test_self_consistency_uses_three_samples():
    with llm.track_usage() as u:
        r = asyncio.run(llm_cot_sc("Claim", [_ev("Evidence snippet text")], LABELS_3))
    assert u.calls == 3 and sum(r.debug["votes"].values()) == 3


def test_reasoning_model_request_settings(monkeypatch):
    from app import config

    seen = {}

    class FakeResp:
        class _C:
            class message:
                content = '{"verdict": "NEI", "confidence": 0.5}'
        choices = [_C]
        usage = None

    class FakeCompletions:
        async def create(self, **kw):
            seen.update(kw)
            return FakeResp()

    class FakeClient:
        class chat:
            completions = FakeCompletions()

    monkeypatch.setattr(llm, "_groq_client", lambda: FakeClient())
    out = asyncio.run(llm._groq_chat([{"role": "user", "content": "json"}], 0.2, 400, True, "openai/gpt-oss-20b", "t"))
    assert "NEI" in out
    assert seen["extra_body"] == {"reasoning_effort": config.GROQ_REASONING_EFFORT}
    assert seen["max_tokens"] >= 1200 and seen["response_format"] == {"type": "json_object"}
    assert config.GROQ_MODEL == "openai/gpt-oss-20b"


def _scripted_mock(monkeypatch, attack_type, judge_verdict, accepted):
    import json as _j

    def reply(msgs, jm, t):
        p = msgs[-1]["content"]
        if p.startswith("Role: PROVER"):
            return _j.dumps({"role": "support", "datum": "x", "warrant": "y", "claim": "z", "confidence": 0.5})
        if p.startswith("Role: SKEPTIC"):
            return _j.dumps({"role": "attack", "attack_type": attack_type, "datum": "x", "warrant": "y", "claim": "z", "confidence": 0.8})
        return _j.dumps({"verdict": judge_verdict, "confidence": 0.95, "accepted_arguments": accepted,
                         "defeated_arguments": ["support"], "summary": "s"})
    monkeypatch.setattr(llm, "_mock_reply", reply)


def test_missing_evidence_is_not_a_refutation(monkeypatch):
    _scripted_mock(monkeypatch, "undermining", "REFUTED", ["attack"])
    r = asyncio.run(run_method("debate", "Dr X said Y", [_ev("Unrelated text about Dr Z")], LABELS_3))
    assert r.verdict == "NEI" and r.confidence <= 0.7 and r.debug["consistency_rule"]


def test_rebuttal_can_refute(monkeypatch):
    _scripted_mock(monkeypatch, "rebuttal", "REFUTED", ["attack"])
    r = asyncio.run(run_method("debate", "The vaccine contains microchips", [_ev("Claims of microchips are false")], LABELS_3))
    assert r.verdict == "REFUTED" and r.debug["consistency_rule"] is None


def test_homework_sites_are_low_credibility_and_hypotheticals_rank_low():
    from app.text_utils import heuristic_claim_score
    for d in ("studocu.com", "www.coursehero.com", "quizlet.com"):
        assert compute_score(d).label == "LOW"
    real = "In 2019 a study in the journal Science found that the algorithm was used for 200 million patients in the US."
    hypo = "Now suppose the map-maker is handed a list of past crimes and where they took place, and asked to shade in areas."
    assert heuristic_claim_score(real) > heuristic_claim_score(hypo)


def test_rule_can_be_switched_off(monkeypatch):
    _scripted_mock(monkeypatch, "undermining", "REFUTED", ["attack"])
    r = asyncio.run(run_method("debate_no_rule", "Dr X said Y", [_ev("Unrelated text about Dr Z")], LABELS_3))
    assert r.verdict == "REFUTED"


def test_short_text_becomes_a_claim():
    assert extract_claim_candidates("Paris is capital of france", 6) == ["Paris is capital of france"]
    assert extract_claim_candidates("ok", 6) == []
