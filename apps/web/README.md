# Fact Validator - web app

Next.js frontend for the Fact Validator API.

```powershell
npm install
npm run dev      # http://localhost:3000
```

The backend must be running (see `services/api/README.md`). If it is not at
`http://127.0.0.1:8000`, create `apps/web/.env.local` with:

```
NEXT_PUBLIC_API_BASE=http://your-host:8000
```

Pages:
- `/`: analyse a URL or text. Shows claims, verdicts, the debate's Toulmin arguments, evidence with credibility scores and run history.
- `/run/<id>`: saved report of a past run (printable, downloadable as JSON).
- `/source`: credibility score of a single domain, with the reasons.
