// Re-Matrix Groq proxy — holds the Groq key server-side.
// Deploy: npx wrangler deploy  (after `wrangler login` + secrets + KV namespace)
//
// Endpoints:
//   POST /chat   -> proxy to Groq /chat/completions (password + rate-limited)
//   GET  /ping   -> connectivity/auth check for the Settings panel
//
// Secrets (set via `wrangler secret put ...`):
//   GROQ_API_KEY  -> your Groq key (never committed)
//   APP_PASSWORD  -> the app access password (never committed)
//
// KV namespace binding REMATRIX_KV is used for rate limiting.

const ALLOWED_MODELS = new Set([
  "llama-3.3-70b-versatile",
  "llama-3.1-8b-instant"
]);

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, X-App-Password",
  "Access-Control-Max-Age": "86400"
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS }
  });
}

function nowHour() {
  return Math.floor(Date.now() / 3600000);
}

function nowDay() {
  return new Date().toISOString().slice(0, 10);
}

async function isRateLimited(env, ip) {
  const perHour = parseInt(env.RATE_LIMIT_PER_HOUR || "50", 10);
  const perDay = parseInt(env.RATE_LIMIT_GLOBAL_DAY || "1000", 10);
  const kv = env.REMATRIX_KV;

  // Per-IP hourly cap: stops a single visitor scripting hundreds of runs.
  const ipKey = `rl:${ip}:${nowHour()}`;
  const ipCount = parseInt((await kv.get(ipKey)) || "0", 10);
  if (ipCount >= perHour) return true;
  await kv.put(ipKey, String(ipCount + 1), { expirationTtl: 3600 });

  // Global daily cap: stops distributed abuse (many IPs) from draining the quota.
  const dayKey = `g:${nowDay()}`;
  const dayCount = parseInt((await kv.get(dayKey)) || "0", 10);
  if (dayCount >= perDay) return true;
  await kv.put(dayKey, String(dayCount + 1), { expirationTtl: 90000 });

  return false;
}

async function handleChat(request, env) {
  // Password check
  const password = request.headers.get("X-App-Password") || "";
  if (!env.APP_PASSWORD || password !== env.APP_PASSWORD) {
    return json({ error: "Unauthorized: invalid access password." }, 401);
  }

  // Rate limiting
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  const limited = await isRateLimited(env, ip);
  if (limited) {
    return json({ error: "Rate limit exceeded. Please try again later." }, 429);
  }

  // Body
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Invalid JSON body." }, 400);
  }
  if (!body || !Array.isArray(body.messages) || body.messages.length === 0) {
    return json({ error: "Body must include non-empty messages." }, 400);
  }
  if (!ALLOWED_MODELS.has(body.model)) {
    return json({ error: `Model not allowed: ${body.model}` }, 400);
  }

  // Forward to Groq, retrying transient rate limits (429) with backoff.
  const payload = {
    model: body.model,
    temperature: typeof body.temperature === "number" ? body.temperature : 0.2,
    response_format: body.response_format || { type: "json_object" },
    messages: body.messages
  };
  const MAX_ATTEMPTS = 4;
  let resp = null;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + env.GROQ_API_KEY
      },
      body: JSON.stringify(payload)
    });
    if (resp.status !== 429 || attempt === MAX_ATTEMPTS) break;

    // Parse the suggested wait from Groq's message ("try again in 1.38s")
    // or the Retry-After header; fall back to exponential backoff.
    let wait = 1000 * Math.pow(2, attempt - 1);
    try {
      const j = await resp.json();
      const m = (j && j.error && j.error.message) || "";
      const ms = m.match(/try again in ([\d.]+)s/i);
      if (ms) wait = Math.ceil(parseFloat(ms[1]) * 1000);
    } catch { /* keep backoff */ }
    wait = Math.max(wait, 1000);
    await new Promise((r) => setTimeout(r, wait));
  }
  const text = await resp.text();
  return new Response(text, { status: resp.status, headers: { "Content-Type": "application/json", ...CORS_HEADERS } });
}

async function handlePing(request, env) {
  const password = request.headers.get("X-App-Password") || "";
  if (password !== env.APP_PASSWORD) {
    return json({ error: "Unauthorized" }, 401);
  }
  return json({ ok: true, models: [...ALLOWED_MODELS] });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }
    const url = new URL(request.url);
    if (url.pathname === "/chat") return handleChat(request, env);
    if (url.pathname === "/ping") return handlePing(request, env);
    return json({ error: "Not found. Use POST /chat or GET /ping." }, 404);
  }
};
