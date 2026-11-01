# Changes: code review fixes + evaluation framework (October 2026)

## Critical fixes
- **Skeptic prompt bug** (`debate.py`). A stray comma turned the Skeptic's prompt into a Python tuple, so the model received a garbled string. The prompts are rewritten, and instructions are no longer embedded inside the JSON template.
- **Circular evidence**. Search results from the article's own site, from aggregators (msn, aol, yahoo, …) and near-verbatim copies of the claim are now excluded. Previously, BBC claims were "supported" by bbc.com, msn.com and aol.com copies of the same article. Excluded items are listed per claim in `excluded_evidence`.
- **API key removed from the source code**. The key is no longer printed to the console; keys are read from `services/api/.env` (see `.env.example`). The old SerpAPI key should be revoked and a new one generated.
- **`openai` added to `requirements.txt`**. Before this, the API could not start on a fresh install.
- **Insufficient-evidence fallback implemented**. When no credible evidence remains, the debate is not run (no LLM call) and the heuristic verdict is used. `verdict_source` records which component decided each claim.

## Logic fixes
- Credibility: platform penalties now match the domain exactly instead of a substring. Before, `netflix.com`, `fox.com` and `dropbox.com` were penalised as "x.com".
- Credibility: curated sources now match sub-domains, so `ec.europa.eu` and `news.bbc.co.uk` are recognised.
- Credibility: the TLD bonus and the curated-list bonus are no longer added together (the larger one is used).
- Credibility: cached scores now carry a rubric version, so a change in the rules takes effect immediately. The cache is keyed by full domain.
- `web.archive.org` links are unwrapped to the real source domain.
- Claim extraction no longer deletes dates. It also filters live-blog and boilerplate fragments, and the ranking adds named-entity and question cues.
- The heuristic REFUTED rule now requires the refuting snippet to be relevant to the claim (overlap ≥ 3).
- The debate now sees the most **relevant** evidence (ranked by relevance, with credibility as tie-breaker) instead of the 2 most credible. The default is 3 items, and it is configurable.
- The Judge prompt no longer anchors on `"confidence": 0.0`, and it defines NEI vs REFUTED explicitly: evidence that is silent on the claim is NEI. A 4th label, CONFLICTING, is available for AVeriTeC.
- By default, all claims are debated, not only the first 2.
- Misinformation likelihood now has a documented formula: the mean P(false) over claims. It uses confidences and the strict-policy verdict.
- Hard-coded paths are removed (`C:\Fact_Validator\...`, and `C:\Users\hassa\...` for the debug log). All paths are relative to the project. Debug logging is off by default (`FACT_VALIDATOR_DEBUG_LOG=1`).
- URL fetch errors are now reported (HTTP 422 with the reason) instead of silently returning no claims.
- CORS is restricted to the web app's origins.
- Claims are processed in parallel, limited to 3 at a time.
- `_init_.py` is renamed to `__init__.py`.
- Groq calls now retry with back-off on rate limits and record tokens and latency.

## Frontend fixes
- Per-source credibility scores are now shown. The UI used to read `score`, while the API sends `domain_score`.
- The debate's Toulmin arguments (Prover and Skeptic), the accepted/defeated arguments, `verdict_source` and the strict-policy verdict are now shown for each claim.
- The useless "Mode: live/snapshot" selector is replaced by a "Strict evidence policy" toggle.
- The saved-report page now shows the claims. It used to read the wrong level of the stored data.
- Run history shows the time and verifier, and each row links to its saved report.
- "Export CSV" now really exports a CSV (one row per claim).
- The API address can be configured with `NEXT_PUBLIC_API_BASE`.

## New: evaluation framework (`services/api/eval/`)
- Datasets:
  - AVeriTeC dev (500 claims, included), with gold, knowledge-store (BM25) or live search evidence
  - FEVER, with gold Wikipedia evidence (NEI claims use RandomS) or search
  - a generic JSONL format
- Methods:
  - B1 heuristic
  - B2 LLM without evidence
  - B3 LLM with evidence (RAG)
  - B4 chain-of-thought + self-consistency (compute-matched to the debate)
  - the proposed debate
  - ablations: no Skeptic, no Toulmin structure, no credibility
- Metrics:
  - accuracy and macro-F1 with 95% bootstrap CIs
  - per-class P/R/F1 and the confusion matrix
  - ECE
  - McNemar test vs the debate
  - LLM calls, tokens, latency, cost and fallback rate
- Resumable runs. Every method sees identical evidence, which is saved for the appendix. Results come out as paper-style tables (`report.md`, `results.csv`).
- `validate_credibility.py`: Spearman correlation of the rubric with expert domain ratings (Lin et al., 2023).
- `--dry-run`: an offline mock LLM, for checking the pipeline at no cost.

## Housekeeping
- Removed unused code: the SQLModel `db/` package, an unmounted duplicate `source_routes.py`, and a stray `.pyc`.
- `.gitignore` now covers `venv/`, databases, caches, logs and the large datasets.
- Added 21 offline tests (`python -m pytest -q` in `services/api`).
- Added README files.

## Still to do by hand
- Remove the files git already tracks but should not, from the project folder:
  ```
  git rm -r --cached services/api/venv services/api/app/__pycache__ data/fact_validator.db services/api/data/domain_cache.json
  git commit -m "Stop tracking venv, caches and databases"
  ```
- `infra/docker-compose.yml` (Postgres/Redis) is not used by the code. Either remove it or say in the thesis that it is reserved for future work.
- `apps/web_backup_16022026` is an old copy of the frontend. Delete it once you are happy with the new UI.
