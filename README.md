# Fact Validator

Master's thesis prototype for credibility-aware, explainable fact-checking. Claims are extracted from an article, checked against retrieved web evidence and source-credibility scores, and verified with a multi-agent LLM debate. In the debate, a Prover and a Skeptic exchange Toulmin-style arguments and a Judge decides.

- `services/api/`: FastAPI backend and the evaluation framework. See [services/api/README.md](services/api/README.md) for setup, how it works and how to run the experiments.
- `apps/web/`: Next.js frontend. Run `npm install` then `npm run dev`. Set `NEXT_PUBLIC_API_BASE` if the API is not at `http://127.0.0.1:8000`.
- [CHANGES.md](CHANGES.md): list of fixes and additions.
