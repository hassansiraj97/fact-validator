# Fact Validator API + evaluation

Backend of the Fact Validator thesis prototype (FastAPI) and the experimental framework
used for the evaluation chapter.

```
services/api/
  app/
    main.py          API routes (/analyze, /source, /runs, /runs-export, /health)
    config.py        paths + settings (.env)
    text_utils.py    URL/domain helpers, claim extraction
    retrieval.py     page download, text extraction, cached SerpAPI search
    evidence.py      labels, evidence filtering (independence), LLM evidence formatting
    credibility.py   rule-based source-credibility rubric
    llm.py           Groq client (+ offline mock), token/latency tracking, retries
    debate.py        Toulmin Prover-Skeptic-Judge debate (+ ablation variants)
    verifiers.py     heuristic baseline, single-LLM baselines, method registry
    storage.py       SQLite run history
  eval/
    run_eval.py      run experiments (resumable)
    report.py        paper-style result tables (report.md, results.csv)
    metrics.py       accuracy, macro-F1, per-class F1, ECE, McNemar, bootstrap CI
    datasets.py      AVeriTeC / FEVER / JSONL loaders
    bm25.py          BM25 retrieval for the knowledge-store setting
    validate_credibility.py   rubric vs. expert domain ratings (Lin et al., 2023)
    data/averitec/dev.json    AVeriTeC dev set (500 claims, included)
  tests/             pytest suite (runs offline with the mock LLM)
```

## 1. Setup (Windows, PowerShell)

```powershell
cd services\api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then put your SERPAPI_API_KEY and GROQ_API_KEY in .env
```

Run the tests (offline, no API keys needed):

```powershell
python -m pytest -q
```

Start the API: `uvicorn app.main:app --reload --port 8000` → docs at http://127.0.0.1:8000/docs

Start the web app (other terminal): `cd apps\web; npm install; npm run dev` → http://localhost:3000

## 2. How a URL is analysed

1. **Text extraction**: trafilatura extracts the article text.
2. **Claim detection**: sentences of 60–280 characters, with boilerplate and live-blog fragments removed, ranked by a check-worthiness heuristic (length, numbers, named entities, factual verbs).
3. **Evidence retrieval**: Google (SerpAPI) with the claim as the query. Results are cached in `data/search_cache/`. The following are **excluded** as non-independent:
   - results from the same site as the article
   - aggregators/syndication (msn, aol, yahoo, …)
   - near-verbatim copies of the claim
   - social media
   - more than 2 results per site
4. **Credibility**: rule-based 0–100 score per domain (`app/credibility.py`):
   - neutral prior of 50
   - plus the larger of the curated-list bonus and the institutional-TLD bonus
   - minus platform penalties (social media, blog hosts) and keyword penalties (e.g. "hoax")

   Labels: HIGH ≥ 80, MEDIUM ≥ 50, LOW < 50.
5. **Verdict**: the heuristic baseline always runs. With `verifier=debate`, the Toulmin debate decides. It falls back to the heuristic verdict when:
   - no credible evidence is left after filtering (no LLM call is made), or
   - the LLM fails.

   `verdict_source` says which component produced each verdict.
6. **Strict policy** (`enable_weighted_confidence`): re-decides each claim using only credible and relevant evidence from ≥ 2 independent sites.
7. **Misinformation likelihood**: the mean over claims of P(false), where REFUTED = confidence, SUPPORTED = 1 − confidence and NEI = 0.5.

## 3. Experiments

All methods are run on the **same claims with the same evidence**. They also share the same credibility filter and the same insufficient-evidence fallback.

| Name | Role | LLM calls / claim |
|---|---|---|
| `heuristic` | B1: keyword overlap + credibility rules | 0 |
| `llm_no_evidence` | B2: single LLM, own knowledge only | 1 |
| `llm_rag` | B3: single LLM with the evidence | 1 |
| `llm_cot_sc` | B4: chain-of-thought + self-consistency, 3 samples (compute-matched to the debate) | 3 |
| `debate` | **Proposed**: Toulmin Prover → Skeptic → Judge | 3 |
| `debate_no_skeptic` | Ablation: Prover → Judge | 2 |
| `debate_freeform` | Ablation: free-text arguments, no Toulmin structure | 3 |
| `debate_no_cred` | Ablation: no credibility scores or filtering | 3 |

### 3.1 Check the pipeline first (free, offline)

```powershell
python -m eval.run_eval --dataset averitec --evidence gold --methods all --limit 20 --dry-run
```

### 3.2 AVeriTeC (main dataset)

The dev set (500 claims) is already in `eval/data/averitec/dev.json`. Labels: SUPPORTED, REFUTED, NEI, CONFLICTING (Conflicting Evidence/Cherrypicking). Use `--labels 3` to merge CONFLICTING into NEI.

```powershell
# a) gold evidence (annotator Q&A pairs) - oracle setting, isolates the verification step
python -m eval.run_eval --dataset averitec --evidence gold --methods all

# b) knowledge-store evidence (realistic retrieval, reproducible)
python -m eval.run_eval --dataset averitec --evidence knowledge_store --methods heuristic,llm_rag,llm_cot_sc,debate
```

For (b), download `dev_knowledge_store.zip` (about 11.5 GB) from https://huggingface.co/chenxwh/AVeriTeC/tree/main/data_store/knowledge_store. Unzip it so that `eval/data/averitec/dev_knowledge_store/0.json … 499.json` exist. Each file holds one JSON object per line, `{"url": ..., "url2text": [sentences]}`, which is the same format the official baseline reads.

To try a few claims first, add `--limit 50` (a stratified sample).

### 3.3 FEVER (secondary dataset)

Download `shared_task_dev.jsonl` and the `wiki-pages` dump from https://fever.ai/dataset/fever.html into `eval/data/fever/`, then run:

```powershell
python -m eval.run_eval --dataset fever --data-path eval/data/fever/shared_task_dev.jsonl `
    --fever-wiki-dir eval/data/fever/wiki-pages --limit 1000 --methods all
```

NEI claims have no annotated evidence. Following Thorne et al. (2018, "RandomS"), they get randomly sampled Wikipedia sentences, so "no evidence" never reveals the label.

### 3.4 Outputs

Each run writes a folder in `eval/results/<dataset>_<evidence>/` with:

- `evidence.jsonl`: the exact evidence every method saw (useful for the appendix)
- `predictions_<method>.jsonl`: one line per claim, with the verdict, confidence, latency and tokens
- `metrics_<method>.json` and `results.csv`: all metrics
- `report.md`: four paper-style tables:
  - effectiveness: accuracy and macro-F1 with 95% bootstrap CIs, ECE, and the McNemar p-value vs the debate
  - per-class F1
  - efficiency: LLM calls, tokens, latency, fallback rate
  - the debate's confusion matrix
- `run_config.json`: model, settings and rubric version

Runs are **resumable**: if Groq rate limits or the run stops, run the same command again and only the missing claims are processed. Use `--price-in` and `--price-out` (USD per 1M tokens, from Groq's pricing page) to add cost to the efficiency table. Re-build the tables with `python -m eval.report eval/results/averitec_gold`.

**Rate limits:** the Groq free tier limits requests per minute. Keep `--concurrency 1` (the default) unless you have a paid tier. Retries with back-off are automatic.

### 3.5 Validate the credibility rubric

Download the aggregate domain ratings of Lin et al. (2023, *PNAS Nexus*), a CSV with columns `domain` and `pc1`, then run:

```powershell
python -m eval.validate_credibility path\to\domain_pc1.csv
```

This reports the Spearman correlation with the expert ratings, agreement of the LOW/MEDIUM/HIGH bands, and coverage of the curated list.

## 4. Notes for the thesis

- The **AVeriTeC score** of the shared task also checks the retrieved question-answer evidence (METEOR ≥ 0.25). This system does not generate QA pairs, so report **label accuracy and macro-F1**, as DebateCV (2025) does in its gold-evidence setting.
- The search setting (`--evidence search`) excludes fact-checking websites, because they would reveal the gold verdict. The knowledge-store setting is preferable for reproducibility.
- Every LLM call is made with temperature ≤ 0.2, except self-consistency sampling (0.7). Re-running gives very similar but not bit-identical results. The prediction files are the record of the experiment.
