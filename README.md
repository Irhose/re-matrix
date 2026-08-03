# Re-Matrix

Causal reasoning for cancer immunology. Re-Matrix grounds every prediction in a curated corpus of principles (L0 axioms → L1 mechanistic pathways → L2 context modifiers → L3 known exceptions), simulates a causal chain, then runs a separate adversarial pass to find edge cases. Every claim is traceable to a source principle.

**Live app:** https://irhose.github.io/re-matrix/

## Features

- **7-stage pipeline**: document ingestion & principle extraction → query understanding → hierarchy-aware retrieval → thought experiment (causal simulation) → edge case generation (adversarial) → traceable report → feedback/refinement loop.
- **Two interfaces**: a browser app (`docs/`) that runs Stages 2–6 fully client-side against the Groq API, and a Python CLI that runs the complete pipeline including Stage 1 (ingestion) and Stage 7 (refinement).
- **Traceable reasoning**: every causal step cites the principle ID, its content, and the source document.

## Access

The web app is password-gated. Enter the access password to unlock it.

Browser requests go through a **Cloudflare Worker proxy** (`workers/proxy`) that holds the Groq API key **server-side** and enforces:

- the app password (checked server-side, not in the client),
- per-IP rate limiting (default 50 calls/hour per visitor),
- a global daily cap (default 1000 calls/day).

This means the key never reaches the browser, so scraping the page source or bypassing the gate cannot burn the Groq quota beyond the caps.

> **Rotation note.** An earlier version of the app bundled the key in `docs/scripts.js`, and that key is now in the public git history. **Rotate your Groq key** at console.groq.com and put the new key in the Worker secret (`wrangler secret put GROQ_API_KEY`).

## Web app (GitHub Pages)

No build step. The pipeline runs in the browser:

1. Open https://irhose.github.io/re-matrix/
2. Enter the access password.
3. Choose models under **API Configuration** (optional).
4. Enter a research query and run the pipeline.

The app calls `PROXY_URL` defined at the top of `docs/scripts.js` — set it to your deployed Worker URL (see `workers/proxy/README.md`).

`docs/` is served directly by GitHub Pages. Rebuild the corpus bundle with:

```bash
python -m cancer_immunology_reasoner.cli ingest
python -m cancer_immunology_reasoner.cli export-web
```

## Python CLI

Backend is **Groq** (set in `.env`). Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env   # then edit GROQ_API_KEY
```

### Stage 1 — Ingest corpus (extracts principles, builds FAISS index + dependency graph)

```bash
python -m cancer_immunology_reasoner.cli ingest
```

Requires local Ollama running with `nomic-embed-text` for embeddings (or set an OpenAI-compatible embedding backend).

### Stages 2–6 — Query the full pipeline

```bash
python -m cancer_immunology_reasoner.cli query "What happens if we block PD-1/PD-L1 in a cold tumor with low TMB?"
```

### Stage 7 — Refine / feedback loop

```bash
# Save a conversation, then refine it
python -m cancer_immunology_reasoner.cli query "..." --save data/conversations/q1.json
python -m cancer_immunology_reasoner.cli refine_conversation data/conversations/q1.json "Add context: tumor is heavily hypoxic" --step 3
```

## Deploying the API proxy (Cloudflare Worker)

The web app requires a deployed proxy. See **`workers/proxy/README.md`** for the full steps:

```bash
cd workers/proxy
npm install -D wrangler
npx wrangler login                                   # one-time browser auth
npx wrangler kv namespace create REMATRIX_KV         # paste ID into wrangler.jsonc
npx wrangler secret put GROQ_API_KEY                 # new (rotated) key
npx wrangler secret put APP_PASSWORD                 # app password
npx wrangler deploy
```

Then set `PROXY_URL` in `docs/scripts.js` to the deployed URL and push.

## Project layout

```
data/
  corpus/        # source PDFs
  index/         # principles.json, index.json (dependency graph), index.faiss
  conversations/ # saved reasoning state
docs/            # GitHub Pages web app (index.html, styles.css, scripts.js, data/principles.json)
workers/proxy/   # Cloudflare Worker: key server-side + password check + rate limits
src/cancer_immunology_reasoner/
  cli.py              # typer CLI (ingest / query / refine / export-web)
  ingestion.py        # Stage 1: PDF → chunks → principles (multi-principle extraction)
  retrieval.py        # Stage 3: FAISS index + hierarchy-aware dependency graph
  thought_experiment.py  # Stage 4
  edge_cases.py       # Stage 5
  report.py           # Stage 6
  pipeline.py         # orchestration + feedback loop (Stage 7)
  llm_client.py       # Groq / Ollama / OpenAI adapters
  config.py           # .env settings
.env.example       # template (copy to .env; never commit .env)
```

## Development

```bash
pip install -e .
python -m cancer_immunology_reasoner.cli --help
```

## Deploying changes to GitHub Pages

```bash
git add .
git commit -m "your change"
git push origin main    # Pages auto-deploys from /docs on main
```
