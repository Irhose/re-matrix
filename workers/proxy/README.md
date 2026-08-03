# Re-Matrix Groq Proxy (Cloudflare Worker)

Holds the Groq API key **server-side** so the browser app never sees it. Validates the app password and rate-limits per IP + a global daily cap, so a leaked password (or scraped client) cannot burn the Groq quota.

```
Browser ──/chat──▶ Worker ──/chat/completions──▶ api.groq.com
              (password)   (key + rate limit)
```

## Deploy (one-time)

Requires a Cloudflare account and Node 18+.

```bash
cd workers/proxy
npm init -y && npm install -D wrangler        # or: npm install -g wrangler

# 1. Login (opens browser)
npx wrangler login

# 2. Create the KV namespace for rate limiting, then paste the ID into wrangler.jsonc
npx wrangler kv namespace create REMATRIX_KV
#   -> copy the printed namespace "id" into wrangler.jsonc: "id": "..."
#   (also set the same id under "preview_id" if you want `wrangler dev`)

# 3. Set secrets (values are NOT stored in the repo)
npx wrangler secret put GROQ_API_KEY        # paste your Groq key
npx wrangler secret put APP_PASSWORD        # paste the app password (Contrari-matrix)

# 4. Deploy
npx wrangler deploy
```

The worker is then live at `https://re-matrix-proxy.<YOUR-SUBDOMAIN>.workers.dev`.
Test it:

```bash
curl -X POST https://re-matrix-proxy.<YOUR-SUBDOMAIN>.workers.dev/chat \
  -H "Content-Type: application/json" \
  -H "X-App-Password: Contrari-matrix" \
  -d '{"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":"ping"}],"response_format":{"type":"json_object"}}'
```

## Point the web app at it

In `docs/scripts.js`, set:

```js
const PROXY_URL = "https://re-matrix-proxy.<YOUR-SUBDOMAIN>.workers.dev";
```

Commit + push. GitHub Pages serves the updated app.

## Limits (tunable in wrangler.jsonc `vars`)

| Var | Default | Meaning |
|---|---|---|
| `RATE_LIMIT_PER_HOUR` | `50` | Max chat calls per IP per hour (~10 pipeline runs) |
| `RATE_LIMIT_GLOBAL_DAY` | `1000` | Max chat calls across all users per day |

## Security notes

- The Groq key exists only in the Worker secret + Groq's console. Never add it to a committed file.
- The app password is checked server-side. Anyone who knows it can *use* the app, but the per-IP + global caps bound the cost.
- Because the old key was previously committed to the repo history, **rotate the Groq key** at console.groq.com and put the *new* key in `wrangler secret put GROQ_API_KEY`.

## Local dev

```bash
npx wrangler dev
```
