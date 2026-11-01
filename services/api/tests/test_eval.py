import json

from eval.datasets import load_averitec, load_jsonl, stratified_sample
from eval.metrics import accuracy, expected_calibration_error, macro_f1, mcnemar_exact, per_class_prf
from eval.run_eval import main as run_eval_main


def test_metrics_basic():
    gold = ["SUPPORTED", "REFUTED", "NEI", "REFUTED"]
    pred = ["SUPPORTED", "REFUTED", "REFUTED", "REFUTED"]
    labels = ["SUPPORTED", "REFUTED", "NEI"]
    assert accuracy(gold, pred) == 0.75
    prf = per_class_prf(gold, pred, labels)
    assert prf["REFUTED"]["precision"] == 2 / 3 and prf["REFUTED"]["recall"] == 1.0
    assert abs(macro_f1(gold, pred, labels) - (1.0 + 0.8 + 0.0) / 3) < 1e-9


def test_mcnemar_and_ece():
    a = [True] * 10 + [False] * 0
    b = [False] * 10
    assert mcnemar_exact(a, b)["p_value"] < 0.01
    assert mcnemar_exact(a, a)["p_value"] == 1.0
    assert expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    assert abs(expected_calibration_error([0.9, 0.9], [False, False]) - 0.9) < 1e-9


def test_averitec_loader_gold():
    exs = load_averitec(evidence="gold")
    assert len(exs) == 500
    assert {e.label for e in exs} == {"SUPPORTED", "REFUTED", "NEI", "CONFLICTING"}
    assert exs[0].evidence and exs[0].evidence[0]["domain"] == "scoopertino.com"
    s = stratified_sample(exs, 50, seed=1)
    assert len(s) == 50 and s == stratified_sample(exs, 50, seed=1)


def test_jsonl_loader(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"id": "a", "claim": "X is Y", "label": "SUPPORTS", "evidence": ["X is Y indeed"]}) + "\n")
    exs = load_jsonl(p)
    assert exs[0].label == "SUPPORTED" and exs[0].evidence[0]["snippet"] == "X is Y indeed"


def test_dry_run_end_to_end(tmp_path):
    out = tmp_path / "run"
    run_eval_main(["--dataset", "averitec", "--evidence", "gold", "--methods", "heuristic,llm_rag,debate",
                   "--limit", "12", "--dry-run", "--out", str(out)])
    assert (out / "report.md").exists() and (out / "results.csv").exists()
    assert "DRY RUN" in (out / "report.md").read_text()
    lines = (out / "predictions_debate.jsonl").read_text().strip().splitlines()
    assert len(lines) == 12
    # resumable: a second run does not duplicate predictions
    run_eval_main(["--dataset", "averitec", "--evidence", "gold", "--methods", "debate",
                   "--limit", "12", "--dry-run", "--out", str(out)])
    assert len((out / "predictions_debate.jsonl").read_text().strip().splitlines()) == 12


def test_knowledge_store_and_fever_loaders(tmp_path):
    from eval.datasets import load_fever

    # --- AVeriTeC knowledge store: one JSON line per document, url2text = list of sentences
    data = [{"claim": "The Eiffel Tower is 330 metres tall.", "label": "Supported", "questions": [], "claim_date": "01-01-2022"}]
    (tmp_path / "dev.json").write_text(json.dumps(data))
    ks = tmp_path / "ks"
    ks.mkdir()
    docs = [
        {"claim_id": 0, "url": "https://www.toureiffel.paris/en", "url2text": [
            "The Eiffel Tower is 330 metres tall since the new antenna was added.", "Paris is the capital of France and a big city."]},
        {"claim_id": 0, "url": "https://example.org/x", "url2text": ["Bananas are yellow fruits that grow in tropical regions."]},
    ]
    (ks / "0.json").write_text("\n".join(json.dumps(d) for d in docs))
    exs = load_averitec(tmp_path / "dev.json", evidence="knowledge_store", knowledge_store_dir=ks)
    assert exs[0].evidence and exs[0].evidence[0]["domain"] == "toureiffel.paris"
    assert "330 metres" in exs[0].evidence[0]["snippet"]

    # --- FEVER: official format + wiki-pages dump
    fever = [
        {"id": 1, "label": "SUPPORTS", "claim": "Naples is in Italy.", "evidence": [[[0, 0, "Naples", 0]]]},
        {"id": 2, "label": "NOT ENOUGH INFO", "claim": "Naples has 99 bridges.", "evidence": [[[0, None, None, None]]]},
    ]
    (tmp_path / "fever.jsonl").write_text("\n".join(json.dumps(r) for r in fever))
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    page = {"id": "Naples", "text": "", "lines": "0\tNaples is a city in Southern Italy -LRB- Campania -RRB- .\tItaly\n1\tIt is the regional capital of Campania and a major port city ."}
    (wiki / "wiki-001.jsonl").write_text(json.dumps(page))
    fx = load_fever(tmp_path / "fever.jsonl", evidence="gold", wiki_dir=wiki)
    sup = next(e for e in fx if e.label == "SUPPORTED")
    nei = next(e for e in fx if e.label == "NEI")
    assert "Naples is a city in Southern Italy (Campania)" in sup.evidence[0]["snippet"]
    assert nei.evidence, "NEI claims get randomly sampled sentences (RandomS)"
