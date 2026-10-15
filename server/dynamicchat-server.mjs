import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir, readFile, readdir, rename, rm, rmdir, stat, unlink, writeFile } from "node:fs/promises";
import { existsSync, readdirSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const dataDir = path.resolve(process.env.DYNAMICCHAT_DATA_DIR ?? path.join(workspaceRoot, ".dynamicchat-data"));
const defaultObjectDir = path.join(dataDir, "objects");
const backupDir = path.join(dataDir, "backups");
const statePath = path.join(dataDir, "state.json");
const secretPath = path.join(dataDir, "dev-secrets.json");
// Neutral working directory for the claude CLI bridge. We spawn claude outside the project
// workspace so it does not auto-load the DynamicChat git repo context, CLAUDE.md, or any
// project-specific files that would re-activate the "Claude Code coding assistant" identity.
// codex and gemini continue to run from dataDir (gemini requires --skip-trust otherwise).
const claudeNeutralCwd = path.join(os.tmpdir(), "dynamicchat-claude-bridge");
// 4318 is inside a common Windows/Hyper-V excluded range (4287-4386) and fails with EACCES.
const port = Number(process.env.DYNAMICCHAT_API_PORT ?? 8788);
// NovelAI is migrating user/account routes off api.novelai.net onto the image host. api.novelai.net
// now answers a valid token with 400 "Please refresh NovelAI.net. If using a third-party tool, update
// to the image URL." so we hit the image host first and fall back to the legacy host only if the route
// is missing there (404/405).
const novelAiSubscriptionUrls = [
  "https://image.novelai.net/user/subscription",
  "https://api.novelai.net/user/subscription"
];
const novelAiGenerateImageUrl = "https://image.novelai.net/ai/generate-image";
const novelAiEncodeVibeUrl = "https://image.novelai.net/ai/encode-vibe";
const NOVELAI_UPSTREAM_TIMEOUT_MS = 180_000;
const rateLimitWindowMs = Number(process.env.DYNAMICCHAT_RATE_LIMIT_WINDOW_MS ?? 60_000);
const rateLimitMaxRequests = Number(process.env.DYNAMICCHAT_RATE_LIMIT_MAX ?? 180);
// Validated rather than read straight through, because both failure modes are silent and neither is
// exotic: "64MB" is NaN and every `> NaN` comparison is false, which disables the cap entirely, while an
// exported-but-empty variable is not nullish, so `Number("")` is 0 and every body is refused. A cap that
// can be switched off or jammed shut by a typo is worse than no cap; an unusable value falls back.
const configuredMaxBodyBytes = Number(process.env.DYNAMICCHAT_MAX_BODY_BYTES);
const maxJsonBodyBytes =
  Number.isFinite(configuredMaxBodyBytes) && configuredMaxBodyBytes > 0 ? configuredMaxBodyBytes : 64 * 1024 * 1024;
const rateLimitBuckets = new Map();
let stateStoreWriteQueue = Promise.resolve();
let secretStoreWriteQueue = Promise.resolve();
// Profile image cleanup (deleteReleasedProfileObjects, sweepAbandonedDraftProfileObjects); declared up here because
// the startup sweep runs before the rest of the module body is evaluated. Only files named the way the builder names
// a pick are ever deleted (createId("profile"), plus the NovelAI "_reference" canvas; story images are asset_*), and
// only folders named the way it names a new simulation's provisional id (createId("draft_sim")) are ever swept.
const PROFILE_OBJECT_FILE_PATTERN = /^profile_[A-Za-z0-9._-]+\.(?:png|jpg|jpeg|webp)$/u;
const DRAFT_PROFILE_DIRECTORY_PATTERN = /^draft_sim_[a-z0-9]+_[a-z0-9]+$/u;
const ABANDONED_DRAFT_PROFILE_AGE_MS = 7 * 24 * 60 * 60 * 1000;
// Canonical object key → number of saves in progress whose incoming state names it. A save that has read the store
// but not yet written it is invisible to a reference check, so its keys are pinned here until it finishes.
const inFlightProfileObjectKeys = new Map();

class PayloadTooLargeError extends Error {
  constructor() {
    super("Request body too large.");
    this.name = "PayloadTooLargeError";
  }
}

const corsHeaders = {
  "access-control-allow-origin": process.env.DYNAMICCHAT_CORS_ORIGIN ?? "*",
  "access-control-allow-methods": "GET,POST,PUT,PATCH,OPTIONS",
  "access-control-allow-headers": "content-type, authorization, x-dynamicchat-owner-id, x-dynamicchat-workspace-id, x-dynamicchat-project-id, x-dynamicchat-environment",
  "access-control-max-age": "86400"
};

await ensureStore();
// Create the neutral claude bridge directory if it doesn't exist yet. Errors are swallowed —
// a missing directory is not fatal; runCliAgentProcess falls back gracefully (spawn will fail
// with ENOENT and the bridge returns a clean 502 rather than crashing the server).
await mkdir(claudeNeutralCwd, { recursive: true }).catch(() => {});
// Before listening, so no save can land while the sweep decides what nothing references. Never throws; the price is
// that an imageStoragePath on a slow or unreachable share delays startup by as long as listing it takes.
await sweepAbandonedDraftProfileObjects();

const server = createServer(async (request, response) => {
  try {
    if (request.method === "OPTIONS") {
      send(response, 204);
      return;
    }

    if (!allowRequestByRateLimit(request, response)) {
      return;
    }

    const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);
    const route = matchRoute(request.method ?? "GET", url.pathname);

    if (!route) {
      sendJson(response, 404, { error: "Not found" });
      return;
    }

    await route.handler(request, response, route.params);
  } catch (error) {
    console.error("DynamicChat API request failed:", error);
    if (response.headersSent) {
      response.destroy();
      return;
    }
    // The generic 500 below says nothing on purpose, which would leave a caller whose image simply
    // outgrew the cap with no way to tell a size refusal from a server bug. This one case is safe to
    // name because the limit is the caller's own configuration, not a fact about the host. It reaches
    // the caller only when the request declared a content-length; a chunked body that overruns while
    // streaming has already lost its socket by the time we get here, so this write goes nowhere.
    if (error instanceof PayloadTooLargeError) {
      sendJson(response, 413, { error: "Request body too large.", maxBytes: maxJsonBodyBytes });
      return;
    }
    // The message is already on the operator's console above. Echoing it to the caller leaks
    // absolute filesystem paths (any escaping fs rejection carries one), so the wire gets a
    // constant — the same shape getObjectAsset has always used for its failures.
    sendJson(response, 500, { error: "Internal server error." });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`DynamicChat API listening on http://127.0.0.1:${port}`);
  console.log(`Data directory: ${dataDir}`);
  console.log("Dev warning: API secrets are stored in a local development secret file, not production encryption.");
});

function matchRoute(method, pathname) {
  const routes = [
    ["GET", /^\/health$/u, health],
    ["GET", /^\/novelai\/subscription$/u, getNovelAiSubscription],
    ["POST", /^\/novelai\/generate-image$/u, proxyNovelAiGenerateImage],
    ["POST", /^\/novelai\/encode-vibe$/u, proxyNovelAiEncodeVibe],
    ["POST", /^\/llm\/cli-agent$/u, proxyLlmCliAgent],
    ["POST", /^\/llm\/chat$/u, proxyLlmChat],
    ["POST", /^\/photoreal\/comfyui\/generate$/u, relayComfyUiGenerate],
    ["GET", /^\/photoreal\/comfyui\/status$/u, relayComfyUiStatus],
    ["POST", /^\/photoreal\/comfyui\/free$/u, relayComfyUiFree],
    ["POST", /^\/photoreal\/comfyui\/interrupt$/u, relayComfyUiInterrupt],
    ["GET", /^\/objects\/(.+)$/u, getObjectAsset],
    ["GET", /^\/simulations$/u, listSimulations],
    ["POST", /^\/simulations$/u, createSimulation],
    ["GET", /^\/simulations\/([^/]+)$/u, getSimulation],
    ["PUT", /^\/simulations\/([^/]+)\/state$/u, saveSimulationState],
    ["GET", /^\/personal-api-vault$/u, getPersonalApiVault],
    ["PUT", /^\/personal-api-vault$/u, savePersonalApiVault],
    ["POST", /^\/simulations\/([^/]+)\/prompt-modules$/u, createPromptModule],
    ["PATCH", /^\/prompt-modules\/([^/]+)$/u, updatePromptModule],
    ["POST", /^\/simulations\/([^/]+)\/chat\/turns$/u, createChatTurn],
    ["POST", /^\/simulations\/([^/]+)\/sessions\/reset$/u, resetSession],
    ["POST", /^\/simulations\/([^/]+)\/image-jobs$/u, createImageJob],
    ["POST", /^\/simulations\/([^/]+)\/assets$/u, persistImageAssets],
    ["GET", /^\/simulations\/([^/]+)\/assets$/u, listAssets],
    ["GET", /^\/simulations\/([^/]+)\/audit$/u, listAudit],
    ["POST", /^\/simulations\/([^/]+)\/backup$/u, createBackup],
    ["POST", /^\/simulations\/([^/]+)\/redactions$/u, createRedaction],
    ["POST", /^\/image-jobs\/([^/]+)\/cancel$/u, cancelImageJob],
    ["GET", /^\/image-jobs\/([^/]+)$/u, getImageJob]
  ];

  for (const [routeMethod, pattern, handler] of routes) {
    const match = pathname.match(pattern);
    if (routeMethod === method && match) {
      return { handler, params: match.slice(1).map(decodeURIComponent) };
    }
  }

  return undefined;
}

function allowRequestByRateLimit(request, response) {
  // Transport-derived only. The key used to fold in x-dynamicchat-owner-id, which the caller supplies
  // and which is in access-control-allow-headers: rotating it handed every request a fresh bucket
  // (measured: 400 requests with rotating ids drew 0 x 429 against a max of 180) and left a permanent
  // Map entry behind each one. The remote address is the one thing the caller cannot choose.
  const key = request.socket.remoteAddress ?? "local";
  const now = Date.now();
  const bucket = rateLimitBuckets.get(key) ?? { count: 0, resetAt: now + rateLimitWindowMs };

  if (now > bucket.resetAt) {
    bucket.count = 0;
    bucket.resetAt = now + rateLimitWindowMs;
  }

  bucket.count += 1;
  rateLimitBuckets.set(key, bucket);

  // Nothing else ever deletes from this Map. Keying on the remote address already bounds it in
  // practice, but a long-lived process still accumulates one entry per address that ever connected,
  // so sweep expired windows once the Map grows past a size no honest local client reaches.
  if (rateLimitBuckets.size > 1000) {
    for (const [existingKey, existingBucket] of rateLimitBuckets) {
      if (now > existingBucket.resetAt) {
        rateLimitBuckets.delete(existingKey);
      }
    }
  }

  if (bucket.count > rateLimitMaxRequests) {
    response.writeHead(429, {
      ...corsHeaders,
      "content-type": "application/json; charset=utf-8",
      "retry-after": String(Math.ceil((bucket.resetAt - now) / 1000))
    });
    response.end(JSON.stringify({ error: "DynamicChat API rate limit exceeded." }));
    return false;
  }

  return true;
}

async function health(_request, response) {
  // /health is a header-free simple GET reachable from any page the user is visiting, so it must not
  // describe the host. The absolute data directory carries the OS username and checkout location, and
  // the limiter configuration tells an attacker exactly what budget it has. Both are printed to the
  // operator's console at startup instead; nothing in src/, server/ or docs/ read them off the wire.
  sendJson(response, 200, { ok: true, service: "dynamicchat-api" });
}

async function getNovelAiSubscription(request, response) {
  const token = readBearerToken(request);
  if (!token) {
    sendJson(response, 400, { error: "NovelAI API token is required." });
    return;
  }

  let upstream;
  let text = "";
  for (let index = 0; index < novelAiSubscriptionUrls.length; index += 1) {
    upstream = await fetch(novelAiSubscriptionUrls[index], {
      method: "GET",
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": "application/json"
      }
    });
    text = await upstream.text();
    const routeMissing = upstream.status === 404 || upstream.status === 405;
    const hasFallback = index < novelAiSubscriptionUrls.length - 1;
    // Only fall through to the legacy host when the current host doesn't serve this route at all.
    if (routeMissing && hasFallback) {
      continue;
    }
    break;
  }

  if (!upstream.ok) {
    sendJson(response, upstream.status, {
      error: text || `NovelAI subscription request failed: ${upstream.status}`
    });
    return;
  }

  sendJson(response, 200, text ? JSON.parse(text) : {});
}

async function proxyNovelAiGenerateImage(request, response) {
  const token = readBearerToken(request);
  if (!token) {
    sendJson(response, 400, { error: "NovelAI API token is required." });
    return;
  }

  const payload = await readJsonBody(request);
  // Temporary timing probe: isolates whether a slow "first image" is NovelAI's upstream latency or our own
  // client/pipeline overhead. Reports the request body size (large vibe payloads inflate the first upload) and
  // the upstream round-trip in ms. Remove once the first-image delay is pinpointed.
  const requestBody = JSON.stringify(payload);
  const upstreamStartedAt = Date.now();
  let upstream;
  try {
    upstream = await fetch(novelAiGenerateImageUrl, {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": "application/json"
      },
      body: requestBody,
      // Mirrors the client-side ceiling. Without it a NovelAI request that never settles holds this handler
      // (and the browser's serialized image queue behind it) open for the life of the process.
      signal: AbortSignal.timeout(NOVELAI_UPSTREAM_TIMEOUT_MS)
    });
  } catch (error) {
    const timedOut = error?.name === "TimeoutError" || error?.name === "AbortError";
    sendJson(response, timedOut ? 504 : 502, {
      error: timedOut
        ? `NovelAI upstream timed out after ${Math.round(NOVELAI_UPSTREAM_TIMEOUT_MS / 1000)}s.`
        : `NovelAI upstream request failed: ${error?.message ?? "unknown error"}`
    });
    return;
  }
  const bytes = Buffer.from(await upstream.arrayBuffer());
  console.log(
    `[novelai-timing] generate-image status=${upstream.status} bodyKB=${Math.round(requestBody.length / 1024)} upstreamMs=${Date.now() - upstreamStartedAt}`
  );

  response.writeHead(upstream.status, {
    ...corsHeaders,
    "content-type": upstream.headers.get("content-type") ?? "application/octet-stream",
    "content-length": String(bytes.byteLength)
  });
  response.end(bytes);
}

// Hosts the LLM chat proxy is allowed to reach. DynamicChat runs in the browser, and virtually no inference
// vendor serves CORS headers for a web origin, so an open-model provider is unreachable without a relay.
// The list is an allow-list rather than an open relay: this server would otherwise be an SSRF hop into the
// user's own network. Self-hosted endpoints (Ollama, LM Studio, vLLM) do allow the page origin and are
// called directly by the client, so they are deliberately absent.
const LLM_PROXY_ALLOWED_HOSTS = new Set([
  "api.openai.com",
  "api.anthropic.com",
  "generativelanguage.googleapis.com",
  "api.deepseek.com",
  "api.moonshot.ai",
  "api.moonshot.cn",
  "dashscope-intl.aliyuncs.com",
  "dashscope.aliyuncs.com",
  "dashscope-us.aliyuncs.com",
  "api.z.ai",
  "open.bigmodel.cn",
  "openrouter.ai",
  "api.groq.com",
  "api.together.xyz",
  "api.fireworks.ai"
]);
const LLM_PROXY_TIMEOUT_MS = 300_000;
const LLM_PROXY_EXTRA_ENV_HOSTS = (process.env.DYNAMICCHAT_LLM_PROXY_HOSTS ?? "")
  .split(",")
  .map((host) => host.trim().toLowerCase())
  .filter(Boolean);

function isAllowedLlmProxyHost(hostname) {
  const host = hostname.toLowerCase();
  return LLM_PROXY_ALLOWED_HOSTS.has(host) || LLM_PROXY_EXTRA_ENV_HOSTS.includes(host);
}

/**
 * Relays one OpenAI-compatible chat request to a vendor the browser cannot reach directly.
 *
 * The API key travels in the Authorization header exactly as the client sent it; this server never stores it
 * for the proxied call. Streaming responses are piped through byte-for-byte so SSE still arrives incrementally.
 */
class LlmProxyRedirectError extends Error {}

/**
 * Follows redirects MANUALLY, re-checking the allow-list on every hop.
 *
 * With fetch's default redirect:"follow" the allow-list only ever covered the first URL, so any 30x from an
 * allowed host — an open redirect, a vendor's marketing hop, a hijacked path — would be followed to an
 * arbitrary destination, including the loopback interface or a cloud metadata endpoint. Since this route
 * answers any origin, that made the proxy an SSRF hop into the user's own network.
 */
async function fetchAllowedUpstream(url, method, headers, body, hop = 0) {
  if (hop > 3) {
    throw new LlmProxyRedirectError("LLM proxy: too many redirects.");
  }
  const response = await fetch(url.toString(), {
    method,
    headers,
    body,
    redirect: "manual",
    signal: AbortSignal.timeout(LLM_PROXY_TIMEOUT_MS)
  });
  if (response.status < 300 || response.status > 399) {
    return response;
  }
  const location = response.headers.get("location");
  if (!location) {
    return response;
  }
  const next = new URL(location, url);
  if (next.protocol !== "https:" || !isAllowedLlmProxyHost(next.hostname)) {
    throw new LlmProxyRedirectError(
      `LLM proxy refused a redirect to ${next.protocol}//${next.hostname}, which is not in the allow-list.`
    );
  }
  return fetchAllowedUpstream(next, method, headers, body, hop + 1);
}

async function proxyLlmChat(request, response) {
  const token = readBearerToken(request);
  let body;
  try {
    body = await readJsonBody(request);
  } catch (error) {
    // A size refusal is not a parse failure. Swallowing it here reports 400 "Invalid JSON body." for a
    // body that was well-formed and merely large — and this route carries the biggest payloads there
    // are (assembled prompts, vibe-transfer base64), so it is where the diagnostic matters most.
    if (error instanceof PayloadTooLargeError) {
      throw error;
    }
    sendJson(response, 400, { error: "Invalid JSON body." });
    return;
  }

  const targetUrl = typeof body.url === "string" ? body.url.trim() : "";
  if (!targetUrl) {
    sendJson(response, 400, { error: "url is required." });
    return;
  }

  let parsedUrl;
  try {
    parsedUrl = new URL(targetUrl);
  } catch {
    sendJson(response, 400, { error: "url is not a valid absolute URL." });
    return;
  }
  if (parsedUrl.protocol !== "https:") {
    sendJson(response, 400, { error: "Only https upstreams are proxied." });
    return;
  }
  if (!isAllowedLlmProxyHost(parsedUrl.hostname)) {
    sendJson(response, 403, {
      error: `Host ${parsedUrl.hostname} is not in the LLM proxy allow-list. Set DYNAMICCHAT_LLM_PROXY_HOSTS to add it.`
    });
    return;
  }

  const upstreamHeaders = { "content-type": "application/json" };
  if (token) {
    upstreamHeaders.authorization = `Bearer ${token}`;
  }
  // Provider-specific headers the preset declares (OpenRouter attribution, Anthropic's browser opt-in).
  // Hop-by-hop and host headers are never forwarded.
  if (body.headers && typeof body.headers === "object" && !Array.isArray(body.headers)) {
    for (const [key, value] of Object.entries(body.headers)) {
      const name = String(key).toLowerCase();
      if (typeof value === "string" && !["host", "content-length", "connection", "authorization"].includes(name)) {
        upstreamHeaders[name] = value;
      }
    }
  }

  // Key verification probes a listing endpoint with GET; generation posts a chat body. Everything else is
  // rejected so this cannot be used as a general-purpose request forwarder.
  const upstreamMethod = body.method === "GET" ? "GET" : "POST";
  const upstreamBody = upstreamMethod === "GET" ? undefined : JSON.stringify(body.body ?? {});
  let upstream;
  try {
    upstream = await fetchAllowedUpstream(parsedUrl, upstreamMethod, upstreamHeaders, upstreamBody);
  } catch (error) {
    if (error instanceof LlmProxyRedirectError) {
      sendJson(response, 403, { error: error.message });
      return;
    }
    const timedOut = error?.name === "TimeoutError" || error?.name === "AbortError";
    sendJson(response, timedOut ? 504 : 502, {
      error: timedOut
        ? `LLM upstream timed out after ${Math.round(LLM_PROXY_TIMEOUT_MS / 1000)}s.`
        : `LLM upstream request failed: ${error?.message ?? "unknown error"}`
    });
    return;
  }

  const contentType = upstream.headers.get("content-type") ?? "application/json";
  // Stream SSE straight through so the caller still sees tokens as they arrive; buffer everything else.
  if (upstream.body && contentType.includes("text/event-stream")) {
    response.writeHead(upstream.status, {
      ...corsHeaders,
      "content-type": contentType,
      "cache-control": "no-cache",
      connection: "keep-alive"
    });
    try {
      for await (const chunk of upstream.body) {
        response.write(Buffer.from(chunk));
      }
    } catch {
      /* client disconnected or upstream aborted mid-stream */
    }
    response.end();
    return;
  }

  const text = await upstream.text().catch(() => "");
  response.writeHead(upstream.status, {
    ...corsHeaders,
    "content-type": contentType,
    "content-length": String(Buffer.byteLength(text))
  });
  response.end(text);
}

async function proxyNovelAiEncodeVibe(request, response) {
  const token = readBearerToken(request);
  if (!token) {
    sendJson(response, 400, { error: "NovelAI API token is required." });
    return;
  }

  const payload = await readJsonBody(request);
  const upstream = await fetch(novelAiEncodeVibeUrl, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  const bytes = Buffer.from(await upstream.arrayBuffer());

  response.writeHead(upstream.status, {
    ...corsHeaders,
    "content-type": upstream.headers.get("content-type") ?? "application/octet-stream",
    "content-length": String(bytes.byteLength)
  });
  response.end(bytes);
}

// claude is used here as a plain single-shot completion engine, NOT for agentic coding. `claude --print`
// otherwise loads the full agent harness (global MCP servers, built-in tool defs, multi-turn loops) which
// inflates latency/usage. We strip the biggest cost — extended THINKING and CLAUDE.md/auto-memory — via env
// vars in runCliAgentProcess (those can never break the invocation).
//
// The hardcoded claude flags below are verified on claude CLI ≥ 2.1.202 (validated on 2.1.202 with exit 0
// and pure Korean prose output). They fully suppress the coding-agent identity so the model writes
// narrative prose instead of refusing or executing shell commands:
//   --tools=:                            disables all built-in tools (Bash, Computer Use, etc.).
//                                        Written as a single `--tools=` token (equals-with-no-value form)
//                                        so Windows cmd.exe shell:true cannot misparse it as consuming the
//                                        next flag (the two-token `--tools ""` form was eaten by cmd.exe).
//   --strict-mcp-config:                 blocks all MCP server connections.
//   --exclude-dynamic-system-prompt-sections: strips the cwd/env/memory/git injections that establish
//                                        "Claude Code coding assistant" identity.
//   --system-prompt-file {systemPromptFile}: passes the simulation system prompt via a temp file rather
//                                        than an argv string. Windows cmd.exe has an ~8191-char command-
//                                        line limit; a full DynamicChat systemPrompt (~8 KB) would hit
//                                        "The command line is too long." (exit 1). A temp file path is
//                                        always short. proxyLlmCliAgent writes the file before spawn and
//                                        deletes it after the process closes.
//
// If your installed claude is older than 2.1.x and rejects these flags (exit non-zero → HTTP 502),
// override the whole command via DYNAMICCHAT_CLI_CLAUDE, e.g.:
//   DYNAMICCHAT_CLI_CLAUDE="claude --print --model {model} --output-format text"
// The env override bypasses this default entirely (older-version escape hatch).
const CLI_AGENT_DEFAULT_COMMANDS = {
  // User turn is piped to stdin; system prompt goes via a temp file (--system-prompt-file, claude only).
  // {model} is replaced by resolveCliAgentArgv. {systemPromptFile} is replaced with the temp-file path.
  claude: ["claude", "--print", "--model", "{model}", "--tools=", "--strict-mcp-config", "--exclude-dynamic-system-prompt-sections", "--system-prompt-file", "{systemPromptFile}", "--output-format", "text"],
  codex: ["codex", "exec", "--model", "{model}", "-"],
  // --skip-trust: the bridge spawns gemini headlessly in the data dir (an untrusted workspace), which it otherwise
  // refuses with exit 55. Gemini reads the piped stdin as the prompt and runs non-interactively (no -p flag needed).
  gemini: ["gemini", "--skip-trust", "--model", "{model}"],
  // antigravity → the `agy` CLI (Google's terminal coding agent that replaced the individual gemini CLI).
  //
  // WITHOUT --print, `agy` opens an INTERACTIVE session: it ignores the piped stdin, prints no completion and
  // never exits, so every turn sat on a live-but-idle subprocess until the absolute ceiling below (9 minutes)
  // and the user saw a turn that simply never finished. --print takes the prompt as its own VALUE
  // (`--print='...'`) and does not read stdin, which a full simulation prompt cannot fit, so the prompt is fed
  // over stdin through the NDJSON input format instead. That path also emits incremental text_delta events,
  // which is what lets this agent stream like claude rather than going dark until the end.
  antigravity: [
    "agy",
    "--print=",
    // agy's own print-mode ceiling defaults to 5m, which is BELOW what a DynamicChat turn takes on a heavy
    // prompt (a 21-paragraph narrative measured 184s, and the annotation pass is the largest call of the
    // turn). Past it agy exits 1 with its reason on stdout, which the NDJSON parser discards — the caller saw
    // a bare "exited with code 1". Raised to sit just above the bridge's own absolute backstop so the bridge,
    // not agy, is what bounds a wedged run and reports it.
    "--print-timeout=16m",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--model",
    "{model}",
    "--disable-slash-commands"
  ]
};

// Streaming variants. `claude --output-format stream-json --verbose --include-partial-messages` emits
// token-level NDJSON deltas, which the bridge parses into plain-text chunks (parseClaudeStreamChunk).
// This makes the first byte arrive in ~1s instead of after the whole (possibly multi-minute) generation,
// so the client never trips its time-to-first-byte timeout, and the idle-timeout below keeps long-but-
// progressing generations alive instead of killing them at a hard ceiling. Agents without a stream entry
// fall back to their default command and have their raw stdout forwarded as-is.
const CLI_AGENT_STREAM_COMMANDS = {
  // Same narrative-hardening flags as the default command; see comment above for rationale.
  claude: ["claude", "--print", "--model", "{model}", "--tools=", "--strict-mcp-config", "--exclude-dynamic-system-prompt-sections", "--system-prompt-file", "{systemPromptFile}", "--output-format", "stream-json", "--verbose", "--include-partial-messages"],
  // agy needs no separate streaming argv: the NDJSON transport its prompt must travel over already carries
  // incremental text_delta events, so one command serves both callers.
  antigravity: CLI_AGENT_DEFAULT_COMMANDS.antigravity
};

// `agy` is a coding agent: its session exposes run_command, view_file, browser_* and the rest, and on a
// narrative prompt it will sometimes answer with a TOOL CALL instead of prose. In print mode nobody is there
// to approve it, so the call fails and agy ends the turn having emitted no text at all — the turn comes back
// empty. claude gets `--tools=` for this; agy has no equivalent flag, so the instruction is the only lever.
const ANTIGRAVITY_NO_TOOL_PREAMBLE =
  "Answer directly with text only. Do not call any tool, do not read or write files, do not run commands, and do not browse. This is a writing request, not a coding task.";

const CLI_AGENT_ENV_OVERRIDES = {
  claude: "DYNAMICCHAT_CLI_CLAUDE",
  codex: "DYNAMICCHAT_CLI_CODEX",
  gemini: "DYNAMICCHAT_CLI_GEMINI",
  antigravity: "DYNAMICCHAT_CLI_ANTIGRAVITY"
};

// Shown when the binary is missing or unauthenticated. "It exited with code 1" is not a thing a user can
// act on; the install command and the sign-in step are.
const CLI_AGENT_INSTALL_HINTS = {
  claude: "Install it with `npm i -g @anthropic-ai/claude-code`.",
  codex: "Install it with `npm i -g @openai/codex`.",
  gemini: "Install it with `npm i -g @google/gemini-cli`.",
  antigravity: "Install Google Antigravity and make sure `agy` is on PATH."
};

const CLI_AGENT_AUTH_HINTS = {
  claude: "Run `claude` once in a terminal and finish the sign-in.",
  codex: "Run `codex` once in a terminal and finish the sign-in.",
  gemini: "Run `gemini` once in a terminal to sign in, or set GEMINI_API_KEY in the environment.",
  antigravity: "Run `agy` once in a terminal and finish the sign-in."
};

// Kept below the client-side CLI ceiling (MAX_CLI_AGENT_REQUEST_TIMEOUT_MS = 300s in llmClient)
// so the bridge times out first and returns a clean 502 rather than the client aborting opaquely.
// This is the hard ceiling for the NON-streaming path (whole response buffered before we see anything).
const cliAgentTimeoutMs = Number(process.env.DYNAMICCHAT_CLI_TIMEOUT_MS ?? 540_000);
// Streaming path uses an inactivity (idle) timeout instead: as long as tokens keep flowing we never
// abort, so a long-but-healthy generation completes; we only give up after a real stall. A generous
// absolute backstop still bounds a truly wedged process.
const cliAgentIdleTimeoutMs = Number(process.env.DYNAMICCHAT_CLI_IDLE_TIMEOUT_MS ?? 120_000);
// How long the streaming path waits before committing to a 200 with no output yet. Must exceed the time a
// CLI takes to fail at startup (missing binary, expired or ineligible login) so that failure is still
// reportable as a 502; see the header-grace comment at its use site.
const CLI_AGENT_HEADER_GRACE_MS = Number(process.env.DYNAMICCHAT_CLI_HEADER_GRACE_MS ?? 20_000);
const cliAgentStreamMaxTimeoutMs = Number(process.env.DYNAMICCHAT_CLI_STREAM_MAX_MS ?? 900_000);

async function proxyLlmCliAgent(_request, response) {
  // The one relay that is NOT stateless-with-caller-credentials: it spawns this machine's own
  // subscription-authenticated CLI (see runCliAgentProcess, which deliberately does not pass --bare so the
  // signed-in OAuth session is used). A JSON POST with no custom headers is a simple request, so without
  // this any page the user is visiting could spend their Claude/Codex/Gemini subscription and read the
  // answer. The app's own calls carry a localhost origin and the eval harness sends none, so both pass.
  if (rejectNonLocalSecretRequest(_request, response)) {
    return;
  }

  let payload;
  try {
    payload = await readJsonBody(_request);
  } catch (error) {
    // Same reason as proxyLlmChat: a size refusal must not be reported as malformed JSON.
    if (error instanceof PayloadTooLargeError) {
      throw error;
    }
    sendJson(response, 400, { error: "Invalid JSON body." });
    return;
  }

  const agent = String(payload.agent ?? "").toLowerCase();
  if (!Object.prototype.hasOwnProperty.call(CLI_AGENT_DEFAULT_COMMANDS, agent)) {
    sendJson(response, 400, { error: `Unsupported CLI agent: ${agent || "(none)"}. Use claude, codex, gemini, or antigravity.` });
    return;
  }

  let model = sanitizeCliModel(payload.model);
  if (payload.model && !model) {
    sendJson(response, 400, { error: "Invalid model name for CLI agent." });
    return;
  }
  if (agent === "antigravity") {
    model = normalizeAntigravityModel(model);
  }

  const systemPrompt = typeof payload.systemPrompt === "string" ? payload.systemPrompt : "";
  const userPrompt = typeof payload.prompt === "string" ? payload.prompt : "";
  const fullPrompt = [systemPrompt.trim(), userPrompt.trim()].filter(Boolean).join("\n\n");
  if (!fullPrompt) {
    sendJson(response, 400, { error: "CLI agent prompt is empty." });
    return;
  }

  const wantStream = payload.stream === true;

  // For claude only: write the system prompt to a per-request temp file so it can be passed via
  // --system-prompt-file. This avoids the Windows cmd.exe command-line length limit (~8191 chars)
  // that causes "The command line is too long." (exit 1) when a full DynamicChat systemPrompt
  // (~8 KB) is embedded directly in the argv string. codex/gemini don't use this path.
  let sysPromptFilePath = "";
  if (agent === "claude" && systemPrompt) {
    sysPromptFilePath = path.join(
      claudeNeutralCwd,
      `sys_${Date.now().toString(36)}_${randomUUID()}.txt`
    );
    await writeFile(sysPromptFilePath, systemPrompt, "utf8");
  }
  // Called after the subprocess closes (on every exit path) to remove the temp file.
  const cleanupSysPromptFile = () => {
    if (sysPromptFilePath) {
      rm(sysPromptFilePath).catch(() => {});
      sysPromptFilePath = "";
    }
  };

  const [command, ...args] = resolveCliAgentArgv(agent, model, { stream: wantStream, prompt: fullPrompt }, sysPromptFilePath);

  // claude receives its system prompt via --system-prompt-file (above); its stdin carries only the
  // user turn so the two don't collide. Other agents (codex/gemini/antigravity) get the combined prompt via stdin.
  // antigravity reads stdin as NDJSON (one message per line), not as raw text — raw text is rejected with
  // `stream input message is missing the "event" field`.
  const stdinText =
    agent === "claude"
      ? (userPrompt.trim() || fullPrompt)
      : agent === "antigravity"
        ? `${JSON.stringify({
            event: "user",
            message: { role: "user", content: [{ type: "text", text: `${ANTIGRAVITY_NO_TOOL_PREAMBLE}\n\n${fullPrompt}` }] }
          })}\n`
        : fullPrompt;
  const spawnWithoutShell = agent === "antigravity";

  const describeMissingBinary = () =>
    `${agent} CLI not found (command: ${command}). ${CLI_AGENT_INSTALL_HINTS[agent] ?? "Install it and ensure it is on PATH."} Or set ${CLI_AGENT_ENV_OVERRIDES[agent]} to its full path.`;

  const describeSpawnError = (error) =>
    error?.code === "ENOENT"
      ? describeMissingBinary()
      : error instanceof Error
        ? error.message
        : `${agent} CLI agent failed.`;

  /**
   * Turns a non-zero exit into something the user can act on.
   *
   * The ENOENT branch above almost never fires in practice: the bridge spawns with `shell: true` on Windows,
   * so a missing binary is not a spawn failure at all — cmd.exe starts fine, prints "'gemini' is not
   * recognized as an internal or external command", and exits 1. The user was then told "gemini CLI exited
   * with code 1" with no hint that the tool simply is not installed. The shell's own wording is matched here
   * so the actionable message is produced on every platform.
   */
  const describeExitFailure = (code, stderr, stdout = "") => {
    // agy prints its own usage/startup errors to STDOUT, not stderr, and on the NDJSON path those lines are
    // discarded by the parser as unparseable. Falling back to stdout is what turns a bare "exited with code 1"
    // into the actual reason.
    const text = String(stderr ?? "").trim() || String(stdout ?? "").trim();
    if (/is not recognized as an internal or external command|command not found|No such file or directory/iu.test(text)) {
      return describeMissingBinary();
    }
    // Gemini and codex report an unconfigured account this way; without the hint it reads as a crash.
    if (/Please set an Auth method|not authenticated|Please login|GEMINI_API_KEY/iu.test(text)) {
      return `${agent} CLI is installed but not signed in. ${CLI_AGENT_AUTH_HINTS[agent] ?? "Run it once in a terminal and complete the sign-in."} Upstream said: ${text.slice(0, 300)}`;
    }
    // A model id the CLI has retired. Distinct from an outage: nothing recovers until the model is changed,
    // and the CLI already prints the list it does accept, so that list is worth forwarding verbatim.
    if (/is not recognized as a known model|invalid model selection|unknown model/iu.test(text)) {
      return `${agent} CLI rejected the selected model "${model || "(default)"}". Pick a different model for this call in the settings. Upstream said: ${text.slice(0, 400)}`;
    }
    // A signed-in account whose PLAN no longer covers this CLI. Distinct from "not signed in": no amount of
    // re-authenticating fixes it, so the actionable advice is to switch the backend, not to log in again.
    if (/IneligibleTier|no longer supported|migrate to the Antigravity/iu.test(text)) {
      return `${agent} CLI is signed in but its account tier no longer allows this client, so it cannot serve requests. Pick a different LLM backend in the settings. Upstream said: ${text.slice(0, 300)}`;
    }
    return `${agent} CLI exited with code ${code}.${text ? ` ${text.slice(0, 500)}` : ""}${text ? "" : " It produced no output."}`;
  };

  if (wantStream) {
    // Stream stdout to the client as it is produced so the UI can render text incrementally.
    let headersSent = false;
    const beginStream = () => {
      // response.headersSent guards against late stdout chunks arriving after the
      // response was already finalized (e.g. via sendJson on the timeout/error path).
      if (!headersSent && !response.headersSent && !response.writableEnded) {
        response.writeHead(200, { ...corsHeaders, "content-type": "text/plain; charset=utf-8", "cache-control": "no-cache" });
        headersSent = true;
      }
    };
    // claude and agy both stream token-level NDJSON; translate it to plain-text deltas so the client's
    // plain-text stream reader is unchanged. Other agents have their raw stdout forwarded verbatim (works
    // whether they emit incrementally or buffer — see the early-header grace timer below).
    const ndjsonStream = resolveCliStdoutParser(agent, wantStream);
    // Send headers before any stdout so an agent that buffers its whole response still establishes the
    // connection early and the client switches to an untimed body read instead of tripping its
    // time-to-first-byte timeout.
    //
    // The delay must outlast a STARTUP failure, not just a spawn failure. At 3s it did not: an unauthenticated
    // CLI takes a few seconds to reach its auth error (the gemini CLI reports IneligibleTierError at ~4.4s), so
    // the headers were already out and the failure could no longer be reported as a 502 — the caller got an
    // empty 200 and the real reason went to stderr and nowhere else. Every turn then lost its images with no
    // message anywhere. The client's own request timeout is at least 90s, so waiting here is free.
    const headerGraceTimer = setTimeout(beginStream, CLI_AGENT_HEADER_GRACE_MS);
    // Only the NDJSON agents are known to emit incrementally, so only they get the inactivity (idle)
    // timeout that resets on each delta. For the others we rely on the generous absolute backstop, since a
    // buffering agent is legitimately silent until the end and an idle timeout would kill a healthy run.
    const idleTimeoutMs = ndjsonStream ? cliAgentIdleTimeoutMs : undefined;
    // Temporary streaming probe: confirms whether the CLI emits text INCREMENTALLY (many chunks, first chunk
    // arrives early → early image dispatch can work) or buffers the whole response (1 chunk near the end → no
    // streaming, so image_cues-first vs assistant_text-first makes no difference to first-image latency).
    let streamedChunks = 0;
    let firstChunkMs = -1;
    const streamStartedAt = Date.now();
    let result;
    try {
      result = await runCliAgentProcess(command, args, stdinText, {
        idleTimeoutMs,
        maxTimeoutMs: cliAgentStreamMaxTimeoutMs,
        shell: spawnWithoutShell ? false : undefined,
        // claude spawns in a neutral directory outside the project to prevent it from loading
        // the DynamicChat git context or CLAUDE.md. codex/gemini stay in dataDir (gemini
        // requires --skip-trust and is already scoped; codex has no similar cwd restriction).
        cwd: agent === "claude" ? claudeNeutralCwd : undefined,
        onStdout: (chunk) => {
          beginStream();
          if (!headersSent || response.writableEnded) {
            return;
          }
          if (ndjsonStream) {
            const text = ndjsonStream.push(chunk);
            if (text) {
              if (firstChunkMs < 0) firstChunkMs = Date.now() - streamStartedAt;
              streamedChunks += 1;
              response.write(text);
            }
          } else {
            if (firstChunkMs < 0) firstChunkMs = Date.now() - streamStartedAt;
            streamedChunks += 1;
            response.write(chunk);
          }
        }
      });
      console.log(
        `[cli-stream-timing] agent=${agent} streamedChunks=${streamedChunks} firstChunkMs=${firstChunkMs} totalMs=${Date.now() - streamStartedAt}`
      );
    } catch (error) {
      clearTimeout(headerGraceTimer);
      cleanupSysPromptFile();
      if (!headersSent) {
        sendJson(response, 502, { error: describeSpawnError(error) });
        return;
      }
      response.end();
      return;
    }
    clearTimeout(headerGraceTimer);
    cleanupSysPromptFile();

    if (!headersSent) {
      // Process produced no stdout: surface an error (client checks response.ok before reading the stream).
      // A zero exit code is NOT proof of success here — the gemini CLI prints its auth failure and leaves,
      // and an empty 200 would be reported upstream as "the model said nothing" rather than "the CLI is not
      // signed in", which is the difference between a fixable message and a silent turn.
      if (result.code !== 0 || !result.stdout.trim()) {
        sendJson(response, 502, {
          error: describeExitFailure(result.code, result.stderr, result.stdout)
        });
        return;
      }
      beginStream();
    }
    // Flush any trailing text: a final delta with no newline, or — if no deltas ever streamed
    // (e.g. partial messages unsupported) — the complete assistant/result text as a one-shot fallback.
    if (ndjsonStream && !response.writableEnded) {
      const tail = ndjsonStream.flush();
      if (tail) {
        response.write(tail);
      }
    }
    response.end();
    return;
  }

  try {
    const result = await runCliAgentProcess(command, args, stdinText, {
      shell: spawnWithoutShell ? false : undefined,
      cwd: agent === "claude" ? claudeNeutralCwd : undefined
    });
    if (result.code !== 0) {
      sendJson(response, 502, { error: describeExitFailure(result.code, result.stderr, result.stdout) });
      return;
    }

    // agy answers in NDJSON on this path too, so the reply has to be translated back to plain text before it
    // reaches a caller that expects the model's answer (the annotation pass parses it as strict JSON).
    const stdoutParser = resolveCliStdoutParser(agent, false);
    const text = stdoutParser ? `${stdoutParser.push(result.stdout)}${stdoutParser.flush()}` : result.stdout;
    if (!text.trim()) {
      sendJson(response, 502, { error: describeExitFailure(result.code, result.stderr, result.stdout) });
      return;
    }

    sendJson(response, 200, { text, agent, model: model || undefined });
  } catch (error) {
    sendJson(response, 502, { error: describeSpawnError(error) });
  } finally {
    cleanupSysPromptFile();
  }
}

function sanitizeCliModel(value) {
  if (typeof value !== "string") {
    return "";
  }
  const trimmed = value.trim();
  // Only allow safe model identifiers and never let a value be parsed as a flag.
  return /^[A-Za-z0-9][A-Za-z0-9._:-]*$/u.test(trimmed) ? trimmed : "";
}

function normalizeAntigravityModel(model) {
  if (!model) return "gemini-3.7-flash-high";
  const exactMappings = {
    "gemini-3.7-flash": "gemini-3.7-flash-high",
    "gemini-3.7-pro": "gemini-3.1-pro-high",
    "gemini-3.1-pro": "gemini-3.1-pro-high",
    "gemini-3.6-flash": "gemini-3.6-flash-high",
    "gemini-3.5-flash": "gemini-3.5-flash-high",
    "gemini-2.0-flash": "gemini-3.7-flash-high",
    "gemini-flash-latest": "gemini-3.7-flash-high"
  };
  if (exactMappings[model]) {
    return exactMappings[model];
  }
  if (model.startsWith("gemini-") && !/-(high|medium|low)$/u.test(model)) {
    return `${model}-high`;
  }
  return model;
}

function resolveCliAgentArgv(agent, model, options = {}, systemPromptFile = "") {
  const override = process.env[CLI_AGENT_ENV_OVERRIDES[agent]]?.trim();
  // An explicit env override always wins; otherwise the streaming variant (if any) is used for streaming
  // requests so the agent emits incremental output instead of buffering the whole response.
  const template = override
    ? override.split(/\s+/u)
    : (options.stream && CLI_AGENT_STREAM_COMMANDS[agent]) || CLI_AGENT_DEFAULT_COMMANDS[agent];
  const argv = [];
  for (let index = 0; index < template.length; index += 1) {
    const token = template[index];
    if (token.includes("{prompt}")) {
      const promptText = typeof options.prompt === "string" ? options.prompt : "";
      if (!promptText) {
        const previous = argv[argv.length - 1];
        if (index > 0 && previous && previous.startsWith("-")) {
          argv.pop();
        }
        continue;
      }
      argv.push(promptText);
      continue;
    }
    if (token.includes("{model}")) {
      // No model selected: drop the placeholder AND the flag that introduces it (e.g. "--model"), otherwise the agent
      // sees a dangling flag with no value (e.g. `gemini --model` → "Not enough arguments following: model").
      if (!model) {
        const previous = argv[argv.length - 1];
        if (index > 0 && previous && previous.startsWith("-")) {
          argv.pop();
        }
        continue;
      }
      argv.push(token.replace(/\{model\}/gu, model));
      continue;
    }
    if (token.includes("{systemPromptFile}")) {
      // No system-prompt file: drop the placeholder AND the flag that introduces it
      // ("--system-prompt-file") so the agent is not called with a dangling flag and no value.
      if (!systemPromptFile) {
        const previous = argv[argv.length - 1];
        if (index > 0 && previous && previous.startsWith("-")) {
          argv.pop();
        }
        continue;
      }
      // File path is always short (just a temp filename); safe to pass as a regular argv token.
      argv.push(systemPromptFile);
      continue;
    }
    if (token.length > 0 || index === 0) {
      argv.push(token);
    }
  }
  return argv;
}

function runCliAgentProcess(command, args, stdinText, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      // Callers may supply a custom cwd (e.g. claudeNeutralCwd for the claude bridge).
      // Defaults to dataDir (the original behavior for codex and gemini).
      cwd: options.cwd ?? dataDir,
      shell: options.shell ?? process.platform === "win32",
      windowsHide: true,
      // Lean-completion env for the claude CLI (ignored by codex/gemini): disable extended thinking so it does
      // not burn thousands of hidden reasoning tokens (the main cause of ~14 tok/s, 466s turns), and skip
      // CLAUDE.md / auto-memory loading. Subscription OAuth auth is preserved (we do NOT use --bare).
      env: {
        ...process.env,
        MAX_THINKING_TOKENS: process.env.MAX_THINKING_TOKENS ?? "0",
        CLAUDE_CODE_DISABLE_CLAUDE_MDS: process.env.CLAUDE_CODE_DISABLE_CLAUDE_MDS ?? "1",
        CLAUDE_CODE_DISABLE_AUTO_MEMORY: process.env.CLAUDE_CODE_DISABLE_AUTO_MEMORY ?? "1"
      }
    });
    let stdout = "";
    let stderr = "";
    let settled = false;

    // Non-streaming callers pass no idleTimeoutMs and keep the original behavior: a single absolute
    // timeout (cliAgentTimeoutMs). Streaming callers pass an idle timeout that resets on every chunk so
    // a long-but-progressing generation is never killed, plus a generous absolute backstop.
    const idleTimeoutMs = options.idleTimeoutMs;
    const absoluteTimeoutMs = options.maxTimeoutMs ?? cliAgentTimeoutMs;
    let idleTimer;
    let absoluteTimer;
    const clearTimers = () => {
      clearTimeout(idleTimer);
      clearTimeout(absoluteTimer);
    };
    const fail = (message) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimers();
      child.kill("SIGKILL");
      reject(new Error(message));
    };
    const armIdleTimer = () => {
      if (!idleTimeoutMs) {
        return;
      }
      clearTimeout(idleTimer);
      idleTimer = setTimeout(
        () => fail(`CLI agent stalled: no output for ${Math.round(idleTimeoutMs / 1000)}s.`),
        idleTimeoutMs
      );
    };
    absoluteTimer = setTimeout(() => fail(`CLI agent timed out after ${absoluteTimeoutMs}ms.`), absoluteTimeoutMs);
    armIdleTimer();

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString("utf8");
      stdout += text;
      armIdleTimer();
      // Stop forwarding once the promise has settled (timeout/error) so we never
      // touch a response that may already have been finalized.
      if (!settled) {
        options.onStdout?.(text);
      }
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
      // stderr activity also counts as liveness (the agent is doing work, e.g. emitting progress logs).
      armIdleTimer();
    });
    child.on("error", (error) => {
      if (!settled) {
        settled = true;
        clearTimers();
        reject(error);
      }
    });
    child.on("close", (code) => {
      if (!settled) {
        settled = true;
        clearTimers();
        resolve({ code: code ?? 0, stdout: stdout.trim(), stderr: stderr.trim() });
      }
    });

    child.stdin.on("error", () => {
      // Ignore EPIPE if the agent closes stdin early; close/error handlers settle the promise.
    });
    child.stdin.end(stdinText);
  });
}

// Parses claude's `--output-format stream-json` NDJSON into plain text. `push` returns the incremental
// text to forward for each chunk; `flush` returns any trailing text once the process closes.
/**
 * Line-buffers a CLI's NDJSON stdout and turns it into plain text.
 *
 * Both streaming agents speak NDJSON but disagree on the envelope, so the buffering and line splitting live
 * here and only the two extractors differ per agent.
 */
function createNdjsonStreamTextParser({ extractDeltaText, extractCompleteText }) {
  let buffer = "";
  let emittedDelta = false;
  let completeText = "";

  const consumeLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return "";
    }
    let event;
    try {
      event = JSON.parse(trimmed);
    } catch {
      return "";
    }
    const delta = extractDeltaText(event);
    if (delta) {
      emittedDelta = true;
      return delta;
    }
    const complete = extractCompleteText(event);
    if (complete) {
      completeText = complete;
    }
    return "";
  };

  return {
    push(chunk) {
      buffer += chunk;
      let out = "";
      let newlineIndex;
      while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newlineIndex);
        buffer = buffer.slice(newlineIndex + 1);
        out += consumeLine(line);
      }
      return out;
    },
    flush() {
      let out = consumeLine(buffer);
      buffer = "";
      // If partial-message deltas never streamed, fall back to the one-shot complete text so the turn
      // still gets its full response (no incremental render, but no data loss).
      if (!emittedDelta && completeText) {
        out += completeText;
      }
      return out;
    }
  };
}

/**
 * Photoreal relay: POST /photoreal/comfyui/generate, GET .../status, POST .../free and POST .../interrupt.
 *
 * The photoreal backend renders through a ComfyUI server on this machine, and the browser cannot talk to it
 * directly: by default ComfyUI's origin middleware answers 403 to any loopback request whose Origin does not
 * match its Host — which is exactly what a fetch from the app's origin carries — and it sends no CORS headers
 * either (both change only with --enable-cors-header). So the app builds an API-format workflow and posts it
 * here, and this server does the round trip: upload the face references, queue the prompt, poll its history,
 * download the images.
 *
 * Every handler opens with rejectNonLocalSecretRequest (the rule is written above isLocalBrowserOrigin). Past
 * that the relay is deliberately narrow, because it is an unauthenticated local service issuing requests on a
 * caller's behalf: loopback ComfyUI only unless the operator names other hosts, ComfyUI's own fixed paths only,
 * no redirects followed, none of the browser's headers forwarded, allow-listed node types only, no image in a
 * graph but the references the request itself uploads (assertComfyUiImagesAreReferences), and no guidance scale below
 * 1 (assertComfyUiGuidanceAtLeastOne).
 */
const COMFYUI_DEFAULT_BASE_URL = "http://127.0.0.1:8188";
// Operator configuration: used when a request names no baseUrl, and its host is trusted without being listed.
const COMFYUI_CONFIGURED_BASE_URL = (process.env.DYNAMICCHAT_COMFYUI_URL ?? "").trim();
// ComfyUI on another machine (a GPU box on the LAN) is a real setup, but a free-form host on an unauthenticated
// relay is an SSRF hop into the user's network, so every non-loopback host is opt-in by name. An entry with a
// port ("192.168.0.10:8188") allows only that port.
const COMFYUI_ALLOWED_HOSTS = new Set(
  [...splitEnvList(process.env.DYNAMICCHAT_COMFYUI_HOSTS), readUrlHostname(COMFYUI_CONFIGURED_BASE_URL)]
    .filter(Boolean)
    .map((host) => host.toLowerCase())
);
// Every ceiling sits inside the client's. generatePhotorealImages waits timeoutSeconds + 30 s and sends
// timeoutMs = timeoutSeconds; the relay spends at most timeoutMs on upload + queue + render and at most
// COMFYUI_FETCH_GRACE_MS more downloading the result. So a slow render always comes back as this server's 504,
// with a reason and with the prompt taken back off the GPU, instead of the browser aborting blind while ComfyUI
// keeps rendering an image nobody will collect.
const COMFYUI_DEFAULT_TIMEOUT_MS = 300_000;
const COMFYUI_MIN_TIMEOUT_MS = 1_000;
const COMFYUI_MAX_TIMEOUT_MS = 900_000;
const COMFYUI_FETCH_GRACE_MS = 20_000;
const COMFYUI_REQUEST_TIMEOUT_MS = 30_000;
const COMFYUI_PROBE_TIMEOUT_MS = 5_000;
const COMFYUI_POLL_INTERVAL_MS = 400;
// Three history polls in a row that cannot even connect mean ComfyUI went away mid-job (on Windows a model load
// that runs out of system RAM kills the process outright). Sitting out the rest of a five-minute budget would
// only delay the same answer.
const COMFYUI_MAX_POLL_CONNECT_FAILURES = 3;
const COMFYUI_MAX_WORKFLOW_NODES = 300;
const COMFYUI_MAX_WORKFLOW_DEPTH = 32;
const COMFYUI_MAX_REFERENCES = 6;
const COMFYUI_MAX_REFERENCE_BYTES = 20 * 1024 * 1024;
const COMFYUI_MAX_OUTPUT_IMAGES = 16;
const COMFYUI_MAX_OUTPUT_IMAGE_BYTES = 64 * 1024 * 1024;
const COMFYUI_MAX_PROBE_NODES = 30;
const COMFYUI_MODEL_FOLDERS = ["diffusion_models", "text_encoders", "vae", "loras"];
const COMFYUI_CLIENT_ID = "dynamicchat-relay";
const COMFYUI_UPLOAD_SUBFOLDER = "dynamicchat";
const COMFYUI_REFERENCE_NAME_PATTERN = /^[A-Za-z0-9_.-]{1,64}$/u;
const COMFYUI_REFERENCE_PLACEHOLDER_PATTERN = /^%ref:([A-Za-z0-9_.-]{1,64})%$/u;
// ComfyUI issues uuid4 prompt ids. Anything else is not an id it handed out, and the value lands in a URL path.
const COMFYUI_PROMPT_ID_PATTERN = /^[A-Za-z0-9_-]{1,128}$/u;
// Node names land in an /object_info/<name> path. The leading character keeps "." and ".." out — URL parsing
// resolves those as dot segments even when percent-encoded — and the rest admits custom-node names such as
// "KSampler (Efficient)".
const COMFYUI_NODE_NAME_PATTERN = /^[A-Za-z0-9_][A-Za-z0-9_ .+()-]{0,99}$/u;
// The loader inputs whose value_not_in_list means "that model file is not installed", and where it belongs.
const COMFYUI_MODEL_INPUT_FOLDERS = {
  ckpt_name: "models/checkpoints",
  unet_name: "models/diffusion_models",
  clip_name: "models/text_encoders",
  clip_name1: "models/text_encoders",
  clip_name2: "models/text_encoders",
  clip_name3: "models/text_encoders",
  vae_name: "models/vae",
  lora_name: "models/loras"
};
// The class_types a relayed workflow may contain. ComfyUI's node registry is open-ended: custom node packs add
// nodes that run arbitrary Python (several popular ones exist precisely to evaluate a code string), and a
// workflow is just JSON naming them. This relay is what stands between "the app queues a render" and "a request
// runs code on this machine", so it takes the core nodes the built-in presets use plus the common loaders and
// samplers a hand-written custom workflow reaches for — and nothing else unless the operator names more in
// DYNAMICCHAT_COMFYUI_EXTRA_NODES (comma list; `*` turns the check off).
//
// LoRA loaders are deliberately not on it. A LoRA changes what the model draws without a word of it in the prompt, so
// the photoreal safety floor — which reads text — cannot see what one adds, and no built-in preset uses one. An
// operator who wants them names them in DYNAMICCHAT_COMFYUI_EXTRA_NODES like any other node.
const COMFYUI_ALLOWED_NODES = new Set([
  "CheckpointLoaderSimple",
  "UNETLoader",
  "CLIPLoader",
  "DualCLIPLoader",
  "TripleCLIPLoader",
  "VAELoader",
  "CLIPTextEncode",
  "ConditioningZeroOut",
  "ConditioningCombine",
  "ConditioningSetAreaPercentage",
  "KSampler",
  "KSamplerAdvanced",
  "SamplerCustom",
  "SamplerCustomAdvanced",
  "KSamplerSelect",
  "RandomNoise",
  "DisableNoise",
  "CFGGuider",
  "BasicGuider",
  "BasicScheduler",
  "ManualSigmas",
  "Flux2Scheduler",
  "ModelSamplingAuraFlow",
  "ModelSamplingSD3",
  "ModelSamplingFlux",
  "CFGNorm",
  "EmptyLatentImage",
  "EmptySD3LatentImage",
  "EmptyFlux2LatentImage",
  "VAEDecode",
  "VAEEncode",
  "VAEDecodeTiled",
  "VAEEncodeTiled",
  "LoadImage",
  "ImageScale",
  "ImageCrop",
  "ImageScaleBy",
  "ImageScaleToTotalPixels",
  "GetImageSize",
  "FluxKontextImageScale",
  "FluxKontextMultiReferenceLatentMethod",
  "ReferenceLatent",
  "TextEncodeQwenImageEdit",
  "TextEncodeQwenImageEditPlus",
  "TextEncodeQwenImage21",
  "QwenImage21Cache",
  "SaveImage",
  "PreviewImage",
  "PrimitiveString",
  "PrimitiveInt",
  "PrimitiveFloat",
  "UnetLoaderGGUF",
  "CLIPLoaderGGUF",
  "DualCLIPLoaderGGUF"
]);
const COMFYUI_EXTRA_NODES = new Set(splitEnvList(process.env.DYNAMICCHAT_COMFYUI_EXTRA_NODES));
const COMFYUI_NODE_CHECK_DISABLED = COMFYUI_EXTRA_NODES.has("*");

/**
 * A relay failure that already knows its HTTP answer. `kind` tells the callers that branch on the cause apart:
 * "rejected" (bad input or a refusal from ComfyUI), "unreachable", "timeout", "redirect", "execution" (ComfyUI ran
 * the prompt and it failed) and "aborted" (the browser went away; nothing is sent).
 */
class ComfyUiRelayError extends Error {
  constructor(status, message, details = {}, kind = "rejected") {
    super(message);
    this.name = "ComfyUiRelayError";
    this.status = status;
    this.details = details;
    this.kind = kind;
  }
}

async function relayComfyUiGenerate(request, response) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const body = await readComfyUiRelayBody(request, response);
  if (!body) {
    return;
  }

  const startedAt = Date.now();
  const jobLabel = typeof body.clientRequestId === "string" ? body.clientRequestId.replace(/[^\w.:-]/gu, "").slice(0, 80) || "-" : "-";
  // The browser gives up by closing the connection: a cancelled job, a closed tab, its own timeout. A response
  // "close" before end() is the signal that survives — the request's own "close" fires as soon as its body has
  // been read — and without it ComfyUI would keep the GPU on an image nobody will collect.
  const clientGone = new AbortController();
  response.on("close", () => {
    if (!response.writableEnded) {
      clientGone.abort();
    }
  });

  let baseUrl;
  let promptId;
  try {
    baseUrl = resolveComfyUiBaseUrl(body.baseUrl);
    assertRequestNotFromComfyUi(request, baseUrl);
    const workflow = validateComfyUiWorkflow(body.workflow);
    const outputNodeIds = readComfyUiOutputNodeIds(body.outputNodeIds, workflow);
    const timeoutMs = readComfyUiTimeoutMs(body.timeoutMs);
    const deadline = startedAt + timeoutMs;
    const references = readComfyUiReferences(body.references);
    const usedReferenceNames = collectComfyUiReferencePlaceholders(workflow, new Set(references.map((reference) => reference.name)));
    assertComfyUiImagesAreReferences(workflow);
    assertComfyUiGuidanceAtLeastOne(workflow);
    // Every used reference is read and checked before the first upload: a request that cannot be submitted must
    // not leave files behind in ComfyUI's input directory. References the workflow never names are not read.
    const usedReferences = references.filter((reference) => usedReferenceNames.has(reference.name));
    const objectRoots = usedReferences.some((reference) => reference.objectKey) ? await getRequestObjectRoots(request) : [];
    const referenceImages = [];
    for (const reference of usedReferences) {
      referenceImages.push({ name: reference.name, image: await loadComfyUiReferenceImage(reference, objectRoots) });
    }
    const uploadedNames = new Map();
    for (const { name, image } of referenceImages) {
      uploadedNames.set(name, await uploadComfyUiReference(baseUrl, name, image, deadline, clientGone.signal));
    }

    // Deliberately not tied to clientGone: once ComfyUI may have accepted the prompt, aborting the submit would
    // throw away the one id that lets us take it back off the queue.
    promptId = await queueComfyUiPrompt(baseUrl, substituteComfyUiReferences(workflow, uploadedNames), deadline);
    if (clientGone.signal.aborted) {
      throw new ComfyUiRelayError(499, "요청한 쪽이 연결을 끊었습니다.", {}, "aborted");
    }
    const entry = await waitForComfyUiPrompt(baseUrl, promptId, deadline, timeoutMs, clientGone.signal);
    const images = await fetchComfyUiOutputImages(baseUrl, entry.outputs, outputNodeIds);
    const elapsedMs = Date.now() - startedAt;
    console.log(`[comfyui-relay] generate job=${jobLabel} prompt=${promptId} images=${images.length} elapsedMs=${elapsedMs}`);
    sendJson(response, 200, { promptId, images, elapsedMs });
  } catch (error) {
    if (!(error instanceof ComfyUiRelayError)) {
      throw error;
    }
    if (promptId && (error.kind === "aborted" || error.kind === "timeout")) {
      await cancelComfyUiPrompt(baseUrl, promptId);
    }
    console.log(`[comfyui-relay] generate job=${jobLabel} prompt=${promptId ?? "-"} ${error.kind} ${error.status}: ${error.message}`);
    if (error.kind === "aborted") {
      return;
    }
    sendJson(response, error.status, { error: error.message, ...(promptId ? { promptId } : {}), ...error.details });
  }
}

async function relayComfyUiStatus(request, response) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  // Only the query is read, so the base is fixed rather than taken from a Host header that could be anything.
  const query = new URL(request.url ?? "/", "http://127.0.0.1").searchParams;
  let baseUrl;
  let nodeNames;
  try {
    baseUrl = resolveComfyUiBaseUrl(query.get("baseUrl") ?? undefined);
    assertRequestNotFromComfyUi(request, baseUrl);
    nodeNames = readComfyUiProbeNodeNames(query.get("nodes"));
  } catch (error) {
    if (!sendComfyUiRelayError(response, error)) {
      throw error;
    }
    return;
  }

  // A probe answers 200 either way: "is ComfyUI there?" has "no" as a normal answer, and the settings screen
  // shows the message rather than treating it as a failed request. A success carries no message — the client
  // composes one from the fields (it knows which of the probed nodes its presets need).
  let stats;
  try {
    const upstream = await comfyUiFetch(baseUrl, "/system_stats", { timeoutMs: COMFYUI_PROBE_TIMEOUT_MS });
    if (!upstream.ok) {
      await discardComfyUiBody(upstream);
      sendJson(response, 200, {
        ok: false,
        message: `ComfyUI 주소(${baseUrl})가 HTTP ${upstream.status}로 응답했습니다. ComfyUI 주소가 맞는지 확인하세요.`
      });
      return;
    }
    stats = await readComfyUiJson(upstream);
  } catch (error) {
    if (!(error instanceof ComfyUiRelayError)) {
      throw error;
    }
    sendJson(response, 200, { ok: false, message: error.message });
    return;
  }
  if (!isPlainObject(stats) || !isPlainObject(stats.system)) {
    sendJson(response, 200, {
      ok: false,
      message: `${baseUrl}의 /system_stats 응답이 ComfyUI 형식이 아닙니다. ComfyUI 주소가 맞는지 확인하세요.`
    });
    return;
  }

  const [models, nodes, queueRemaining] = await Promise.all([
    probeComfyUiModels(baseUrl),
    probeComfyUiNodes(baseUrl, nodeNames),
    probeComfyUiQueueRemaining(baseUrl)
  ]);
  const comfyuiVersion = typeof stats.system.comfyui_version === "string" ? stats.system.comfyui_version : undefined;
  // Picked field by field: system_stats also carries ComfyUI's argv, Python and RAM figures, none of which the
  // settings screen needs.
  const devices = (Array.isArray(stats.devices) ? stats.devices : [])
    .filter(isPlainObject)
    .slice(0, 16)
    .map((device) => ({
      name: readComfyUiDeviceName(device.name),
      vramTotalMb: bytesToMegabytes(device.vram_total),
      vramFreeMb: bytesToMegabytes(device.vram_free)
    }));
  sendJson(response, 200, { ok: true, comfyuiVersion, devices, models, nodes, queueRemaining });
}

async function relayComfyUiFree(request, response) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const body = await readComfyUiRelayBody(request, response);
  if (!body) {
    return;
  }
  try {
    const baseUrl = resolveComfyUiBaseUrl(body.baseUrl);
    assertRequestNotFromComfyUi(request, baseUrl);
    // Unloading the models is the point: the GPU is handed back to a local LLM that cannot share 16 GB with them.
    const upstream = await comfyUiFetch(baseUrl, "/free", { method: "POST", json: { unload_models: true, free_memory: true } });
    await discardComfyUiBody(upstream);
    if (!upstream.ok) {
      throw new ComfyUiRelayError(502, `ComfyUI가 메모리 해제 요청을 거부했습니다 (HTTP ${upstream.status}).`);
    }
    sendJson(response, 200, { ok: true });
  } catch (error) {
    if (!sendComfyUiRelayError(response, error)) {
      throw error;
    }
  }
}

async function relayComfyUiInterrupt(request, response) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const body = await readComfyUiRelayBody(request, response);
  if (!body) {
    return;
  }
  try {
    const baseUrl = resolveComfyUiBaseUrl(body.baseUrl);
    assertRequestNotFromComfyUi(request, baseUrl);
    // Required, not defaulted: an interrupt without a prompt id stops whatever ComfyUI is running, including
    // work the user queued from ComfyUI's own UI.
    const promptId = typeof body.promptId === "string" ? body.promptId.trim() : "";
    if (!COMFYUI_PROMPT_ID_PATTERN.test(promptId)) {
      throw new ComfyUiRelayError(400, "promptId가 필요합니다. 중단할 작업의 ComfyUI prompt_id를 보내세요.");
    }
    const removed = await comfyUiFetch(baseUrl, "/queue", { method: "POST", json: { delete: [promptId] } });
    await discardComfyUiBody(removed);
    const interrupted = await comfyUiFetch(baseUrl, "/interrupt", { method: "POST", json: { prompt_id: promptId } });
    await discardComfyUiBody(interrupted);
    if (!removed.ok || !interrupted.ok) {
      throw new ComfyUiRelayError(
        502,
        `ComfyUI가 작업 중단 요청을 거부했습니다 (queue HTTP ${removed.status}, interrupt HTTP ${interrupted.status}).`
      );
    }
    sendJson(response, 200, { ok: true, promptId });
  } catch (error) {
    if (!sendComfyUiRelayError(response, error)) {
      throw error;
    }
  }
}

async function readComfyUiRelayBody(request, response) {
  let body;
  try {
    body = await readJsonBody(request);
  } catch (error) {
    // Same reason as proxyLlmChat: a size refusal is not malformed JSON, and inline reference images make this
    // one of the routes most likely to meet the cap.
    if (error instanceof PayloadTooLargeError) {
      throw error;
    }
    sendJson(response, 400, { error: "요청 본문이 올바른 JSON이 아닙니다." });
    return undefined;
  }
  if (!isPlainObject(body)) {
    sendJson(response, 400, { error: "요청 본문은 JSON 객체여야 합니다." });
    return undefined;
  }
  return body;
}

function sendComfyUiRelayError(response, error) {
  if (!(error instanceof ComfyUiRelayError)) {
    return false;
  }
  sendJson(response, error.status, { error: error.message, ...error.details });
  return true;
}

/**
 * The ComfyUI origin a request may talk to, or a 400 saying why not.
 *
 * Only an origin survives. The relay appends its own fixed paths, so a base carrying a path, query or fragment is
 * refused rather than trimmed — quietly dropping "/comfy" would send requests somewhere the user did not
 * configure. Userinfo is refused because fetch turns it into an Authorization header nobody agreed to send.
 */
function resolveComfyUiBaseUrl(value) {
  if (value !== undefined && value !== null && typeof value !== "string") {
    throw new ComfyUiRelayError(400, "baseUrl은 문자열이어야 합니다.");
  }
  const raw = value?.trim() || COMFYUI_CONFIGURED_BASE_URL || COMFYUI_DEFAULT_BASE_URL;
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new ComfyUiRelayError(400, `ComfyUI 주소가 올바른 URL이 아닙니다: ${truncateText(raw, 200)}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new ComfyUiRelayError(400, `ComfyUI 주소는 http:// 또는 https:// 로 시작해야 합니다 (받은 값: ${url.protocol}).`);
  }
  if (url.username || url.password) {
    throw new ComfyUiRelayError(400, "ComfyUI 주소에 사용자 정보(아이디:비밀번호@)를 넣을 수 없습니다.");
  }
  if (url.pathname !== "/" || url.search || url.hash) {
    throw new ComfyUiRelayError(400, `ComfyUI 주소에는 경로나 쿼리를 붙일 수 없습니다. ${url.origin} 처럼 호스트와 포트만 적으세요.`);
  }
  const hostname = url.hostname.toLowerCase();
  if (!isLoopbackHostname(hostname) && !COMFYUI_ALLOWED_HOSTS.has(hostname) && !COMFYUI_ALLOWED_HOSTS.has(url.host.toLowerCase())) {
    throw new ComfyUiRelayError(
      400,
      `ComfyUI 주소 ${url.host}은(는) 허용되지 않습니다. 이 컴퓨터(127.0.0.1 · localhost) 밖의 ComfyUI를 쓰려면 DynamicChat API 서버의 DYNAMICCHAT_COMFYUI_HOSTS 환경변수에 호스트를 추가하세요.`
    );
  }
  const effectivePort = url.port ? Number(url.port) : url.protocol === "https:" ? 443 : 80;
  if (isLoopbackHostname(hostname) && effectivePort === port) {
    throw new ComfyUiRelayError(400, `ComfyUI 주소(${url.origin})가 DynamicChat API 서버 자신을 가리킵니다. ComfyUI의 포트(기본 8188)를 적으세요.`);
  }
  return url.origin;
}

/**
 * Refuses a page served by the very ComfyUI this request would drive. The deny list (isComfyUiBrowserOrigin) only
 * knows the ports it was told about; this catches a ComfyUI on any other port at the moment it is used. "The same
 * ComfyUI" is decided the way the deny list decides it, by isSameServerEndpoint — comparing origin strings let a page
 * on http://localhost:8190 drive the ComfyUI it was served from as http://127.0.0.1:8190. The body has the guard's
 * shape, since it is the same refusal.
 */
function assertRequestNotFromComfyUi(request, baseUrl) {
  const origin = parseServerEndpoint(request.headers.origin?.toString());
  const target = parseServerEndpoint(baseUrl);
  if (origin && target && isSameServerEndpoint(origin, target)) {
    throw new ComfyUiRelayError(403, "forbidden_origin", {
      message: `ComfyUI 웹 UI(${origin.origin})에서 온 요청은 받지 않습니다. DynamicChat 앱에서 다시 시도하세요.`
    });
  }
}

/**
 * Loopback by address, or by the one name that is loopback everywhere: localhost, 127.0.0.0/8, ::1 and IPv4-mapped
 * ::ffff:127.x.x.x. The name is canonicalised by the URL parser first, so every spelling of those (127.1, 0x7f.0.0.1,
 * [0:0::1], [::ffff:127.0.0.1]) is recognised as what it is. Deliberately not *.localhost or a trailing-dot
 * "localhost.": the relay also uses this for base URLs it fetches server-side, where such names can go to DNS.
 */
function isLoopbackHostname(hostname) {
  const canonical = canonicalHostname(hostname);
  return (
    canonical === "localhost" ||
    /^127(?:\.\d{1,3}){3}$/u.test(canonical) ||
    canonical === "[::1]" ||
    /^\[::ffff:7f[0-9a-f]{2}:[0-9a-f]{1,4}\]$/u.test(canonical)
  );
}

/** The URL parser's spelling of a hostname (lower case, dotted-decimal IPv4, compressed bracketed IPv6), or "". */
function canonicalHostname(hostname) {
  const raw = String(hostname ?? "").trim();
  if (!raw) {
    return "";
  }
  // A bare IPv6 literal ("::1") would read as a host and a port, so it is bracketed first.
  const host = raw.includes(":") && !raw.startsWith("[") ? `[${raw}]` : raw;
  try {
    return new URL(`http://${host}/`).hostname;
  } catch {
    return "";
  }
}

function readUrlHostname(value) {
  try {
    return value ? new URL(value).hostname : "";
  } catch {
    return "";
  }
}

function validateComfyUiWorkflow(value) {
  if (isPlainObject(value) && Array.isArray(value.nodes) && Array.isArray(value.links)) {
    throw new ComfyUiRelayError(400, "UI 형식으로 저장된 워크플로입니다. ComfyUI 메뉴의 'Export (API)'로 내보낸 API 형식 JSON을 쓰세요.");
  }
  if (!isPlainObject(value)) {
    throw new ComfyUiRelayError(400, "workflow는 ComfyUI API 형식(노드 id → { class_type, inputs })의 객체여야 합니다.");
  }
  const nodes = Object.entries(value);
  if (nodes.length === 0 || nodes.length > COMFYUI_MAX_WORKFLOW_NODES) {
    throw new ComfyUiRelayError(400, `워크플로 노드 수는 1-${COMFYUI_MAX_WORKFLOW_NODES}개여야 합니다 (받은 수: ${nodes.length}).`);
  }
  const rejected = new Set();
  for (const [nodeId, node] of nodes) {
    const label = truncateText(nodeId, 40);
    if (!isPlainObject(node) || typeof node.class_type !== "string" || !node.class_type.trim() || !isPlainObject(node.inputs)) {
      throw new ComfyUiRelayError(400, `워크플로 노드 ${label}이(가) { class_type, inputs } 형식이 아닙니다.`);
    }
    // Bounds every later walk over the node (_meta included), so none of them can be driven into a stack overflow.
    if (exceedsComfyUiDepth(node, COMFYUI_MAX_WORKFLOW_DEPTH)) {
      throw new ComfyUiRelayError(400, `워크플로 노드 ${label}의 입력이 너무 깊게 중첩되어 있습니다.`);
    }
    if (!COMFYUI_NODE_CHECK_DISABLED && !COMFYUI_ALLOWED_NODES.has(node.class_type) && !COMFYUI_EXTRA_NODES.has(node.class_type)) {
      rejected.add(truncateText(node.class_type, 100));
    }
  }
  if (rejected.size > 0) {
    const rejectedNodes = [...rejected];
    const loraNote = rejectedNodes.some((classType) => /lora/iu.test(classType))
      ? " LoRA 로더는 기본으로 허용하지 않습니다: LoRA가 더하는 내용은 프롬프트에 드러나지 않아 실사 안전 검사가 볼 수 없습니다."
      : "";
    throw new ComfyUiRelayError(
      400,
      `허용되지 않은 ComfyUI 노드: ${rejectedNodes.join(", ")}.${loraNote} 꼭 필요한 노드라면 DynamicChat API 서버의 DYNAMICCHAT_COMFYUI_EXTRA_NODES 환경변수에 이름을 넣고 서버를 다시 시작하세요.`,
      { rejectedNodes }
    );
  }
  return value;
}

/**
 * Every image a relayed workflow loads must be a reference this request uploaded. LoadImage reads any file in
 * ComfyUI's input folder by name, so `{ LoadImage: "photo.png" } -> PreviewImage` would hand back whatever sits there
 * (another app's upload, an earlier render) as this request's "render". So a LoadImage-family node, or any node whose
 * `image` input is a string (an IMAGE socket is always a link, so a string there is a file name), must hold exactly
 * `%ref:<name>%`. Runs after collectComfyUiReferencePlaceholders, which has already refused a placeholder naming no
 * reference of this request. Holds with the node check switched off.
 *
 * This is NOT what keeps a profile pick to images DynamicChat generated: a direct caller can upload any bytes as a
 * dataUrl reference and get them back through a %ref: pass-through, so a relay response proves nothing about where an
 * image came from (and a direct caller can drive ComfyUI itself anyway). That guarantee is the app's: it sends only
 * object keys of picked profile images as references, never bytes (comfyUiClient selectSendableReferences), and only
 * images its own graphs rendered reach the profile studio's pick.
 */
function assertComfyUiImagesAreReferences(workflow) {
  for (const [nodeId, node] of Object.entries(workflow)) {
    if (!/^LoadImage/u.test(node.class_type) && typeof node.inputs.image !== "string") {
      continue;
    }
    const image = node.inputs.image;
    if (typeof image !== "string" || !COMFYUI_REFERENCE_PLACEHOLDER_PATTERN.test(image)) {
      const received = typeof image === "string" ? `'${truncateText(image, 80)}'` : JSON.stringify(image ?? null).slice(0, 80);
      throw new ComfyUiRelayError(
        400,
        `${truncateText(node.class_type, 100)} 노드 ${truncateText(nodeId, 40)}의 image는 이 요청이 올린 참조 이미지(%ref:<이름>%)여야 합니다 (받은 값: ${received}). 릴레이는 ComfyUI 입력 폴더에 이미 있는 파일을 워크플로에 넣지 않습니다.`,
        { rejectedImageNodes: [nodeId] }
      );
    }
  }
}

// Guidance-scale inputs: `cfg` and ComfyUI's variants of it (cfg_scale, cfg_conds, cfg_cond2_negative, min_cfg,
// video_cfg, ...), plus the negative-prompt scales neg_scale and nag_scale. The same pattern as
// GUIDANCE_SCALE_INPUT_PATTERN in src/services/comfyWorkflows.ts, which the app's own graphs are built against.
const COMFYUI_GUIDANCE_INPUT_PATTERN = /^(?:[a-z0-9]+_)*cfg(?:_scale|_conds|_cond\d*_negative)?$|^(?:neg|nag)_scale$/iu;
const COMFYUI_MIN_GUIDANCE = 1;

/**
 * No relayed graph samples below CFG 1. A sampler mixes uncond + cfg × (cond − uncond) with the NEGATIVE as uncond, so
 * below 1 the picture moves toward the negative and at 0 it is the negative alone — the text the app's safety floor
 * deliberately does not read. The app never builds such a graph (PHOTOREAL_MIN_CFG in comfyWorkflows.ts); this holds
 * the same line for any graph posted here, with the node check switched off too. The value must be a number of at
 * least 1: ComfyUI casts a numeric string before sampling, so "0.5" counts as 0.5, and a link is refused because what
 * feeds it (a PrimitiveFloat, a math node) is not something the relay evaluates.
 */
function assertComfyUiGuidanceAtLeastOne(workflow) {
  const rejected = [];
  for (const [nodeId, node] of Object.entries(workflow)) {
    for (const [input, value] of Object.entries(node.inputs)) {
      if (!COMFYUI_GUIDANCE_INPUT_PATTERN.test(input)) {
        continue;
      }
      const scale = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : Number.NaN;
      if (!Number.isFinite(scale) || scale < COMFYUI_MIN_GUIDANCE) {
        rejected.push({ nodeId, classType: node.class_type, input, value });
      }
    }
  }
  if (rejected.length === 0) {
    return;
  }
  const [first] = rejected;
  const received = Array.isArray(first.value) ? "다른 노드의 출력(링크)" : truncateText(JSON.stringify(first.value ?? null), 40);
  throw new ComfyUiRelayError(
    400,
    `${truncateText(first.classType, 100)} 노드 ${truncateText(first.nodeId, 40)}의 ${truncateText(first.input, 40)}는 ${COMFYUI_MIN_GUIDANCE} 이상의 숫자여야 합니다 (받은 값: ${received}). CFG가 1보다 작으면 샘플러가 네거티브 쪽으로 그려 네거티브 문구가 그림이 되므로 릴레이는 이런 워크플로를 보내지 않습니다.`,
    { rejectedGuidanceInputs: rejected.map((entry) => `${entry.nodeId}.${entry.input}`) }
  );
}

function exceedsComfyUiDepth(value, remaining) {
  if (value === null || typeof value !== "object") {
    return false;
  }
  if (remaining <= 0) {
    return true;
  }
  return (Array.isArray(value) ? value : Object.values(value)).some((item) => exceedsComfyUiDepth(item, remaining - 1));
}

function readComfyUiOutputNodeIds(value, workflow) {
  if (!Array.isArray(value)) {
    return [];
  }
  return [...new Set(value.filter((nodeId) => typeof nodeId === "string" && Object.hasOwn(workflow, nodeId)))];
}

function readComfyUiTimeoutMs(value) {
  const requested = Number(value);
  if (value === undefined || value === null || value === "" || !Number.isFinite(requested)) {
    return COMFYUI_DEFAULT_TIMEOUT_MS;
  }
  return Math.min(COMFYUI_MAX_TIMEOUT_MS, Math.max(COMFYUI_MIN_TIMEOUT_MS, Math.round(requested)));
}

/**
 * Validates the reference list's shape without touching the disk. An objectKey is checked segment by segment
 * with isSafeObjectPathSegment before resolveObjectPath ever sees it (the containment check is the backstop, not
 * the gate — see isSafeObjectPathSegment), and only image extensions are accepted, so a reference can name
 * nothing in the object store but an image.
 */
function readComfyUiReferences(value) {
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new ComfyUiRelayError(400, "references는 배열이어야 합니다.");
  }
  if (value.length > COMFYUI_MAX_REFERENCES) {
    throw new ComfyUiRelayError(400, `참조 이미지는 한 번에 ${COMFYUI_MAX_REFERENCES}장까지 보낼 수 있습니다 (받은 수: ${value.length}).`);
  }
  const names = new Set();
  return value.map((entry) => {
    const name = isPlainObject(entry) ? entry.name : undefined;
    if (typeof name !== "string" || !COMFYUI_REFERENCE_NAME_PATTERN.test(name)) {
      throw new ComfyUiRelayError(400, `참조 이미지 이름이 올바르지 않습니다: ${truncateText(String(name), 80)}. 영문·숫자·_ . - 1-64자여야 합니다.`);
    }
    if (names.has(name)) {
      throw new ComfyUiRelayError(400, `참조 이미지 이름이 중복되었습니다: ${name}`);
    }
    names.add(name);
    if (entry.objectKey !== undefined && entry.objectKey !== null) {
      const segments = typeof entry.objectKey === "string" ? entry.objectKey.split("/") : [];
      if (
        segments.length !== 2 ||
        !segments.every(isSafeObjectPathSegment) ||
        !/\.(?:png|jpe?g|webp)$/iu.test(segments[1])
      ) {
        throw new ComfyUiRelayError(400, `참조 이미지 ${name}의 objectKey가 올바르지 않습니다: ${truncateText(String(entry.objectKey), 120)}`);
      }
      return { name, objectKey: entry.objectKey };
    }
    if (typeof entry.dataUrl === "string" && entry.dataUrl) {
      return { name, dataUrl: entry.dataUrl };
    }
    throw new ComfyUiRelayError(400, `참조 이미지 ${name}에 objectKey도 dataUrl도 없습니다.`);
  });
}

async function loadComfyUiReferenceImage(reference, objectRoots) {
  let bytes;
  if (reference.objectKey) {
    const objectFile = resolveExistingObjectFile(reference.objectKey, objectRoots);
    const info = objectFile ? await stat(objectFile.filePath) : undefined;
    if (!info?.isFile()) {
      throw new ComfyUiRelayError(400, `참조 이미지 ${reference.name}을(를) 찾을 수 없습니다 (${reference.objectKey}).`);
    }
    if (info.size > COMFYUI_MAX_REFERENCE_BYTES) {
      throw new ComfyUiRelayError(400, `참조 이미지 ${reference.name}이(가) 너무 큽니다 (최대 20MB).`);
    }
    bytes = await readFile(objectFile.filePath);
  } else {
    const prefix = /^data:image\/(?:png|jpeg|webp);base64,/u.exec(reference.dataUrl);
    if (!prefix) {
      throw new ComfyUiRelayError(400, `참조 이미지 ${reference.name}의 dataUrl은 data:image/png·jpeg·webp;base64, 형식이어야 합니다.`);
    }
    const encoded = reference.dataUrl.slice(prefix[0].length);
    // Measured before decoding, so an oversized image is refused without allocating it.
    if (Math.floor((encoded.length * 3) / 4) > COMFYUI_MAX_REFERENCE_BYTES) {
      throw new ComfyUiRelayError(400, `참조 이미지 ${reference.name}이(가) 너무 큽니다 (최대 20MB).`);
    }
    if (!/^[A-Za-z0-9+/]*={0,2}$/u.test(encoded)) {
      throw new ComfyUiRelayError(400, `참조 이미지 ${reference.name}의 base64 데이터가 올바르지 않습니다.`);
    }
    bytes = Buffer.from(encoded, "base64");
  }
  // The bytes decide the type, not the declared MIME or the extension: whatever lands in ComfyUI's input
  // directory is a real PNG, JPEG or WebP.
  const mimeType = sniffImageMimeType(bytes);
  if (!mimeType) {
    throw new ComfyUiRelayError(400, `참조 이미지 ${reference.name}이(가) PNG·JPEG·WebP 이미지가 아닙니다.`);
  }
  return { bytes, mimeType };
}

function sniffImageMimeType(bytes) {
  if (bytes.length >= 8 && bytes.readUInt32BE(0) === 0x89504e47 && bytes.readUInt32BE(4) === 0x0d0a1a0a) {
    return "image/png";
  }
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "image/jpeg";
  }
  if (bytes.length >= 12 && bytes.toString("latin1", 0, 4) === "RIFF" && bytes.toString("latin1", 8, 12) === "WEBP") {
    return "image/webp";
  }
  return undefined;
}

/**
 * Names the references the workflow uses, and refuses the placeholders the relay could not fill. One that
 * survived to ComfyUI would not be harmless: LoadImage would look for a file literally called "%ref:char_1%" and
 * fail the prompt with a validation error naming neither the reference nor the cause.
 */
function collectComfyUiReferencePlaceholders(workflow, referenceNames) {
  const used = new Set();
  const unresolved = new Set();
  visitComfyUiStrings(workflow, (text) => {
    const match = COMFYUI_REFERENCE_PLACEHOLDER_PATTERN.exec(text);
    if (match && referenceNames.has(match[1])) {
      used.add(match[1]);
    } else if (text.includes("%ref:")) {
      unresolved.add(truncateText(text, 80));
    }
  });
  if (unresolved.size > 0) {
    const unresolvedReferences = [...unresolved];
    throw new ComfyUiRelayError(
      400,
      `워크플로에 채울 수 없는 참조 자리표시자가 남아 있습니다: ${unresolvedReferences.join(", ")}. 같은 이름의 참조 이미지를 references에 보내세요.`,
      { unresolvedReferences }
    );
  }
  return used;
}

function visitComfyUiStrings(value, visit) {
  if (typeof value === "string") {
    visit(value);
  } else if (Array.isArray(value)) {
    value.forEach((item) => visitComfyUiStrings(item, visit));
  } else if (isPlainObject(value)) {
    Object.values(value).forEach((item) => visitComfyUiStrings(item, visit));
  }
}

function substituteComfyUiReferences(value, uploadedNames) {
  if (typeof value === "string") {
    const match = COMFYUI_REFERENCE_PLACEHOLDER_PATTERN.exec(value);
    return match && uploadedNames.has(match[1]) ? uploadedNames.get(match[1]) : value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => substituteComfyUiReferences(item, uploadedNames));
  }
  if (isPlainObject(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, substituteComfyUiReferences(item, uploadedNames)]));
  }
  return value;
}

/**
 * Uploads one reference into ComfyUI's input/dynamicchat folder and returns the name LoadImage must use.
 *
 * The file name carries a digest of the bytes. With a fixed "dc_<name>" and overwrite on, two jobs whose
 * char_1 is a different person would share one file, and since LoadImage reads it when the prompt EXECUTES, not
 * when it is queued, a job waiting behind another would render with the later upload's face. Identical bytes
 * still map to one file, so a profile image is stored once however many cuts use it. The name ComfyUI returns is
 * used as given rather than assumed.
 */
async function uploadComfyUiReference(baseUrl, name, image, deadline, signal) {
  const digest = createHash("sha256").update(image.bytes).digest("hex").slice(0, 16);
  const fileName = `dc_${name}_${digest}.${extensionForMimeType(image.mimeType)}`;
  const form = new FormData();
  form.append("image", new Blob([image.bytes], { type: image.mimeType }), fileName);
  form.append("type", "input");
  form.append("subfolder", COMFYUI_UPLOAD_SUBFOLDER);
  form.append("overwrite", "true");
  const upstream = await comfyUiFetch(baseUrl, "/upload/image", {
    method: "POST",
    form,
    timeoutMs: remainingComfyUiMs(deadline, COMFYUI_REQUEST_TIMEOUT_MS),
    signal
  });
  const uploaded = upstream.ok ? await readComfyUiJson(upstream) : await discardComfyUiBody(upstream);
  const uploadedName = typeof uploaded?.name === "string" ? uploaded.name.trim() : "";
  if (!upstream.ok || !uploadedName || uploadedName.length > 255) {
    throw new ComfyUiRelayError(502, `ComfyUI가 참조 이미지 업로드를 거부했습니다 (${name}: HTTP ${upstream.status}).`);
  }
  const subfolder = typeof uploaded.subfolder === "string" ? uploaded.subfolder.trim() : "";
  return subfolder ? `${subfolder}/${uploadedName}` : uploadedName;
}

async function queueComfyUiPrompt(baseUrl, workflow, deadline) {
  // No prompt_id is sent: ComfyUI expects a UUID there, and the client's job ids (createId) are not UUIDs.
  const upstream = await comfyUiFetch(baseUrl, "/prompt", {
    method: "POST",
    json: { prompt: workflow, client_id: COMFYUI_CLIENT_ID },
    timeoutMs: remainingComfyUiMs(deadline, COMFYUI_REQUEST_TIMEOUT_MS)
  });
  const payload = await readComfyUiJson(upstream);
  if (upstream.status === 400) {
    throw new ComfyUiRelayError(400, describeComfyUiPromptRejection(payload), {
      nodeErrors: summarizeComfyUiNodeErrors(payload?.node_errors),
      comfyError: summarizeComfyUiError(payload?.error)
    });
  }
  if (!upstream.ok) {
    throw new ComfyUiRelayError(502, `ComfyUI가 작업 등록에 실패했습니다 (HTTP ${upstream.status}).`);
  }
  const promptId = payload?.prompt_id;
  if (typeof promptId !== "string" || !COMFYUI_PROMPT_ID_PATTERN.test(promptId)) {
    throw new ComfyUiRelayError(502, "ComfyUI가 작업을 받았다는 응답(prompt_id)을 돌려주지 않았습니다.");
  }
  return promptId;
}

/**
 * Turns ComfyUI's 400 into a sentence a user can act on. The common case is a model file that is not installed
 * yet — ComfyUI reports it as value_not_in_list on the loader input, followed by the whole folder listing — so
 * that is named outright, file and folder, instead of surfacing "Value not in list".
 */
function describeComfyUiPromptRejection(payload) {
  const problems = [];
  for (const [nodeId, entry] of Object.entries(isPlainObject(payload?.node_errors) ? payload.node_errors : {})) {
    const classType = typeof entry?.class_type === "string" ? entry.class_type : `노드 ${nodeId}`;
    for (const failure of Array.isArray(entry?.errors) ? entry.errors : []) {
      problems.push(describeComfyUiNodeError(classType, failure));
    }
  }
  let summary;
  if (problems.length > 0) {
    summary = problems.slice(0, 4).join(" / ") + (problems.length > 4 ? ` 외 ${problems.length - 4}건` : "");
  } else {
    const error = isPlainObject(payload?.error) ? payload.error : {};
    summary = [error.message, error.details].filter((part) => typeof part === "string" && part.trim()).join(" — ");
    if (/does not exist|not found|missing_node/iu.test(`${error.type ?? ""} ${summary}`)) {
      summary += " (ComfyUI를 최신 버전으로 업데이트하거나 해당 커스텀 노드를 설치하세요)";
    }
  }
  return `ComfyUI가 워크플로를 거부했습니다: ${truncateText(summary || "검증 오류 (ComfyUI가 상세 내용을 보내지 않았습니다)", 700)}`;
}

function describeComfyUiNodeError(classType, failure) {
  const inputName = typeof failure?.extra_info?.input_name === "string" ? failure.extra_info.input_name : "";
  const received = readComfyUiReceivedValue(failure);
  if (received && failure?.type === "value_not_in_list" && COMFYUI_MODEL_INPUT_FOLDERS[inputName]) {
    return `모델 파일 '${received}'이(가) ComfyUI에 없습니다 (${classType}.${inputName}). ComfyUI의 ${COMFYUI_MODEL_INPUT_FOLDERS[inputName]} 폴더에 넣은 뒤 다시 시도하세요`;
  }
  if (received && classType === "LoadImage" && inputName === "image") {
    return `입력 이미지 '${received}'을(를) ComfyUI가 찾지 못했습니다 (LoadImage)`;
  }
  const message = [failure?.message, failure?.details].filter((part) => typeof part === "string" && part.trim()).join(": ");
  return `${classType}${inputName ? `.${inputName}` : ""}: ${truncateText(message || String(failure?.type ?? "알 수 없는 오류"), 200)}`;
}

function readComfyUiReceivedValue(failure) {
  const received = failure?.extra_info?.received_value;
  if (typeof received === "string" || typeof received === "number") {
    return truncateText(String(received), 200);
  }
  const quoted = typeof failure?.details === "string" ? /'([^']{1,200})' not in/u.exec(failure.details) : null;
  return quoted?.[1] ?? "";
}

// Handed back so the client can show more than the sentence. input_config is dropped: for a loader it is the
// complete list of model files on the user's disk, and this object can end up in a job's error and state.json.
function summarizeComfyUiNodeErrors(nodeErrors) {
  if (!isPlainObject(nodeErrors)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(nodeErrors)
      .slice(0, 50)
      .map(([nodeId, entry]) => [
        nodeId,
        {
          class_type: typeof entry?.class_type === "string" ? entry.class_type : undefined,
          errors: (Array.isArray(entry?.errors) ? entry.errors : []).slice(0, 10).map((failure) => ({
            type: typeof failure?.type === "string" ? failure.type : undefined,
            message: typeof failure?.message === "string" ? truncateText(failure.message, 300) : undefined,
            details: typeof failure?.details === "string" ? truncateText(failure.details, 300) : undefined,
            input_name: typeof failure?.extra_info?.input_name === "string" ? failure.extra_info.input_name : undefined,
            received_value: readComfyUiReceivedValue(failure) || undefined
          }))
        }
      ])
  );
}

function summarizeComfyUiError(error) {
  if (!isPlainObject(error)) {
    return undefined;
  }
  return {
    type: typeof error.type === "string" ? error.type : undefined,
    message: typeof error.message === "string" ? truncateText(error.message, 300) : undefined,
    details: typeof error.details === "string" ? truncateText(error.details, 500) : undefined
  };
}

/**
 * Polls /history/<id> until ComfyUI files the prompt there. A prompt only appears in history once it has
 * finished, successfully or not, so presence is completion; pending and running prompts are simply absent.
 */
async function waitForComfyUiPrompt(baseUrl, promptId, deadline, timeoutMs, signal) {
  let connectFailures = 0;
  for (;;) {
    if (signal.aborted) {
      throw new ComfyUiRelayError(499, "요청한 쪽이 연결을 끊었습니다.", {}, "aborted");
    }
    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      throw new ComfyUiRelayError(
        504,
        `ComfyUI가 ${Math.round(timeoutMs / 1000)}초 안에 이미지를 완성하지 못해 작업을 중단했습니다. 첫 생성은 모델을 VRAM에 올리느라 오래 걸릴 수 있으니 설정에서 대기 시간을 늘려 보세요.`,
        {},
        "timeout"
      );
    }
    try {
      const upstream = await comfyUiFetch(baseUrl, `/history/${encodeURIComponent(promptId)}`, {
        timeoutMs: Math.min(COMFYUI_PROBE_TIMEOUT_MS * 2, remaining),
        signal
      });
      connectFailures = 0;
      const history = upstream.ok ? await readComfyUiJson(upstream) : await discardComfyUiBody(upstream);
      const entry = isPlainObject(history) ? history[promptId] : undefined;
      if (isPlainObject(entry)) {
        if (entry.status?.status_str === "error") {
          throw new ComfyUiRelayError(500, describeComfyUiExecutionError(entry), {}, "execution");
        }
        if (entry.status?.completed === true || entry.status?.status_str === "success" || hasComfyUiOutputs(entry)) {
          return entry;
        }
      }
    } catch (error) {
      if (!(error instanceof ComfyUiRelayError)) {
        throw error;
      }
      if (error.kind === "unreachable") {
        connectFailures += 1;
        if (connectFailures >= COMFYUI_MAX_POLL_CONNECT_FAILURES) {
          throw error;
        }
      } else if (error.kind !== "timeout") {
        // A single slow poll is retried; the overall deadline above is what ends the wait.
        throw error;
      }
    }
    await sleepUnlessAborted(Math.min(COMFYUI_POLL_INTERVAL_MS, Math.max(0, deadline - Date.now())), signal);
  }
}

function hasComfyUiOutputs(entry) {
  return isPlainObject(entry.outputs) && Object.keys(entry.outputs).length > 0;
}

function describeComfyUiExecutionError(entry) {
  const messages = Array.isArray(entry.status?.messages) ? entry.status.messages : [];
  const findMessage = (type) => messages.find((message) => Array.isArray(message) && message[0] === type)?.[1];
  const failure = findMessage("execution_error");
  if (isPlainObject(failure)) {
    const nodeType = typeof failure.node_type === "string" && failure.node_type ? failure.node_type : "알 수 없는 노드";
    const exception = truncateText(String(failure.exception_message ?? "").trim(), 500) || String(failure.exception_type ?? "원인 미상");
    const outOfMemory = /out of memory|allocation on device|OutOfMemory/iu.test(`${failure.exception_type ?? ""} ${failure.exception_message ?? ""}`);
    return `ComfyUI 실행 오류: ${nodeType}: ${exception}${
      outOfMemory ? " (GPU 메모리 부족 — 해상도를 낮추거나 VRAM을 쓰는 다른 프로그램(로컬 LLM 등)을 내린 뒤 다시 시도하세요)" : ""
    }`;
  }
  const interrupted = findMessage("execution_interrupted");
  if (isPlainObject(interrupted)) {
    return `ComfyUI 실행 오류: 작업이 중단되었습니다 (${typeof interrupted.node_type === "string" ? interrupted.node_type : "interrupt"}).`;
  }
  return "ComfyUI 실행 오류: ComfyUI가 원인을 알려주지 않았습니다.";
}

/**
 * Downloads the finished images. Failures here answer 500, not 502: the client retries a 502 once on the
 * understanding that nothing was queued, and at this point the render has already happened — a retry would pay
 * for the whole cut again to meet, most likely, the same missing output.
 */
async function fetchComfyUiOutputImages(baseUrl, outputs, outputNodeIds) {
  const entries = collectComfyUiOutputImages(outputs, outputNodeIds);
  if (entries.length === 0) {
    throw new ComfyUiRelayError(500, "ComfyUI가 작업을 마쳤지만 결과 이미지가 없습니다. 워크플로에 PreviewImage 또는 SaveImage 노드가 있는지 확인하세요.");
  }
  const deadline = Date.now() + COMFYUI_FETCH_GRACE_MS;
  const images = [];
  for (const entry of entries) {
    const upstream = await comfyUiFetch(baseUrl, "/view", {
      query: { filename: entry.filename, subfolder: entry.subfolder, type: entry.type },
      timeoutMs: remainingComfyUiMs(deadline, COMFYUI_REQUEST_TIMEOUT_MS)
    });
    const declaredBytes = Number(upstream.headers.get("content-length"));
    if (!upstream.ok || (Number.isFinite(declaredBytes) && declaredBytes > COMFYUI_MAX_OUTPUT_IMAGE_BYTES)) {
      await discardComfyUiBody(upstream);
      throw new ComfyUiRelayError(500, `ComfyUI에서 결과 이미지를 받지 못했습니다 (${entry.filename}: HTTP ${upstream.status}).`);
    }
    const bytes = Buffer.from(await upstream.arrayBuffer());
    if (bytes.byteLength === 0 || bytes.byteLength > COMFYUI_MAX_OUTPUT_IMAGE_BYTES) {
      throw new ComfyUiRelayError(500, `ComfyUI 결과 이미지의 크기가 올바르지 않습니다 (${entry.filename}: ${bytes.byteLength} bytes).`);
    }
    const contentType = upstream.headers.get("content-type")?.split(";")[0].trim().toLowerCase() ?? "";
    images.push({
      ...entry,
      mimeType: sniffImageMimeType(bytes) ?? (contentType.startsWith("image/") ? contentType : mimeTypeForObjectKey(entry.filename)),
      base64: bytes.toString("base64")
    });
  }
  return images;
}

/**
 * The prompt's result images, each tagged with the node that produced it.
 *
 * The nodes the client names in outputNodeIds ARE the result, in that order and whatever folder ComfyUI wrote them
 * to: the built-in graphs end in PreviewImage (ComfyUI's temp folder, so a deleted cut does not live on in its
 * output folder), and a rule by folder alone would let any stray SaveImage elsewhere in a graph stand in for the
 * cut. Only when the named nodes produced nothing — or none were named — does the folder decide: saved ("output")
 * images first, previews ("temp") only when nothing was saved, since a preview next to a save is usually an
 * intermediate step.
 */
function collectComfyUiOutputImages(outputs, outputNodeIds) {
  if (!isPlainObject(outputs)) {
    return [];
  }
  const imagesOf = (nodeId, types) =>
    (Array.isArray(outputs[nodeId]?.images) ? outputs[nodeId].images : [])
      .filter((image) => isPlainObject(image) && types.includes(image.type) && typeof image.filename === "string" && image.filename)
      .map((image) => ({ nodeId, filename: image.filename, subfolder: typeof image.subfolder === "string" ? image.subfolder : "", type: image.type }));
  const declared = outputNodeIds.filter((nodeId) => Object.hasOwn(outputs, nodeId)).flatMap((nodeId) => imagesOf(nodeId, ["output", "temp"]));
  if (declared.length > 0) {
    return declared.slice(0, COMFYUI_MAX_OUTPUT_IMAGES);
  }
  const nodeIds = Object.keys(outputs);
  const saved = nodeIds.flatMap((nodeId) => imagesOf(nodeId, ["output"]));
  return (saved.length > 0 ? saved : nodeIds.flatMap((nodeId) => imagesOf(nodeId, ["temp"]))).slice(0, COMFYUI_MAX_OUTPUT_IMAGES);
}

/**
 * Takes a prompt back: off the pending queue first, then interrupted if it is already running. In that order
 * because the reverse leaves a gap — a prompt still pending when the (targeted, so no-op) interrupt arrives can
 * start before the delete lands and then render to the end. Best effort: every caller is already on a failure
 * path with nothing better to report.
 */
async function cancelComfyUiPrompt(baseUrl, promptId) {
  for (const [pathname, json] of [
    ["/queue", { delete: [promptId] }],
    ["/interrupt", { prompt_id: promptId }]
  ]) {
    try {
      await discardComfyUiBody(await comfyUiFetch(baseUrl, pathname, { method: "POST", json, timeoutMs: COMFYUI_PROBE_TIMEOUT_MS }));
    } catch {
      // Unreachable or slow: there is nothing left to take back, or no way to.
    }
  }
}

async function probeComfyUiModels(baseUrl) {
  const listings = await Promise.all(
    COMFYUI_MODEL_FOLDERS.map(async (folder) => {
      try {
        const upstream = await comfyUiFetch(baseUrl, `/models/${folder}`, { timeoutMs: COMFYUI_PROBE_TIMEOUT_MS });
        // 404 is ComfyUI saying it has no such folder registered, which for a probe means "nothing installed".
        const files = upstream.ok ? await readComfyUiJson(upstream) : await discardComfyUiBody(upstream);
        return [folder, Array.isArray(files) ? files.filter((file) => typeof file === "string").slice(0, 2000) : []];
      } catch (error) {
        if (!(error instanceof ComfyUiRelayError)) {
          throw error;
        }
        return [folder, []];
      }
    })
  );
  return Object.fromEntries(listings);
}

async function probeComfyUiNodes(baseUrl, nodeNames) {
  const results = await Promise.all(
    nodeNames.map(async (nodeName) => {
      try {
        const upstream = await comfyUiFetch(baseUrl, `/object_info/${encodeURIComponent(nodeName)}`, { timeoutMs: COMFYUI_PROBE_TIMEOUT_MS });
        // ComfyUI answers an unknown node with 200 and {}, so only a schema under the node's own name counts.
        const info = upstream.ok ? await readComfyUiJson(upstream) : await discardComfyUiBody(upstream);
        return [nodeName, isPlainObject(info) && isPlainObject(info[nodeName])];
      } catch (error) {
        if (!(error instanceof ComfyUiRelayError)) {
          throw error;
        }
        return [nodeName, false];
      }
    })
  );
  return Object.fromEntries(results);
}

async function probeComfyUiQueueRemaining(baseUrl) {
  try {
    // GET /prompt rather than /queue: the queue listing carries every queued workflow in full.
    const upstream = await comfyUiFetch(baseUrl, "/prompt", { timeoutMs: COMFYUI_PROBE_TIMEOUT_MS });
    const info = upstream.ok ? await readComfyUiJson(upstream) : await discardComfyUiBody(upstream);
    const remaining = info?.exec_info?.queue_remaining;
    return Number.isFinite(remaining) ? remaining : undefined;
  } catch (error) {
    if (!(error instanceof ComfyUiRelayError)) {
      throw error;
    }
    return undefined;
  }
}

function readComfyUiProbeNodeNames(value) {
  const names = [...new Set(String(value ?? "").split(",").map((name) => name.trim()).filter(Boolean))];
  if (names.length > COMFYUI_MAX_PROBE_NODES) {
    throw new ComfyUiRelayError(400, `nodes는 한 번에 ${COMFYUI_MAX_PROBE_NODES}개까지 확인할 수 있습니다 (받은 수: ${names.length}).`);
  }
  const invalid = names.filter((name) => !COMFYUI_NODE_NAME_PATTERN.test(name));
  if (invalid.length > 0) {
    throw new ComfyUiRelayError(400, `확인할 노드 이름이 올바르지 않습니다: ${invalid.slice(0, 5).map((name) => truncateText(name, 60)).join(", ")}`);
  }
  return names;
}

function readComfyUiDeviceName(value) {
  const name = typeof value === "string" ? value.trim() : "";
  // ComfyUI reports "cuda:0 NVIDIA GeForce RTX 4080 SUPER : cudaMallocAsync" — index, card, allocator.
  const card = /^(?:cuda|xpu|npu|mlu|musa):\d+\s+(.+?)\s+:\s+\S+$/u.exec(name)?.[1];
  return truncateText(card ?? name, 200) || "알 수 없는 장치";
}

function bytesToMegabytes(value) {
  const bytes = Number(value);
  return Number.isFinite(bytes) && bytes > 0 ? Math.round(bytes / (1024 * 1024)) : 0;
}

/**
 * One request to the configured ComfyUI.
 *
 * Only the fixed paths named in this file are ever requested, and the headers are built here from nothing: the
 * browser's Origin, Cookie and Sec-Fetch-* never reach ComfyUI (and ComfyUI would 403 a loopback request whose
 * Origin does not match its Host anyway). A 3xx is an error rather than a hop — following it would let whatever
 * answers on an allowed host point the relay at any other address.
 */
async function comfyUiFetch(baseUrl, pathname, options = {}) {
  const { method = "GET", json, form, query, timeoutMs = COMFYUI_REQUEST_TIMEOUT_MS, signal } = options;
  const url = new URL(pathname, baseUrl);
  for (const [key, value] of Object.entries(query ?? {})) {
    url.searchParams.set(key, String(value));
  }
  const timeout = AbortSignal.timeout(Math.max(1, timeoutMs));
  let upstream;
  try {
    upstream = await fetch(url, {
      method,
      headers: json === undefined ? undefined : { "content-type": "application/json" },
      body: json === undefined ? form : JSON.stringify(json),
      redirect: "manual",
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout
    });
  } catch {
    if (signal?.aborted) {
      throw new ComfyUiRelayError(499, "요청한 쪽이 연결을 끊었습니다.", {}, "aborted");
    }
    if (timeout.aborted) {
      throw new ComfyUiRelayError(504, `ComfyUI가 ${Math.ceil(timeoutMs / 1000)}초 동안 응답하지 않았습니다 (${pathname}).`, {}, "timeout");
    }
    throw new ComfyUiRelayError(502, `ComfyUI 서버에 연결할 수 없습니다 (${baseUrl}). ComfyUI를 실행했는지 확인하세요.`, {}, "unreachable");
  }
  if (upstream.status >= 300 && upstream.status < 400) {
    await discardComfyUiBody(upstream);
    throw new ComfyUiRelayError(
      502,
      `ComfyUI 주소(${baseUrl})가 리디렉션(HTTP ${upstream.status})으로 응답했습니다. 리디렉션은 따라가지 않으니 ComfyUI 주소를 확인하세요.`,
      {},
      "redirect"
    );
  }
  return upstream;
}

async function readComfyUiJson(upstream) {
  const text = await upstream.text().catch(() => "");
  if (!text.trim()) {
    return undefined;
  }
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

// An unread body pins its connection until garbage collection, so every response the relay does not need is
// cancelled explicitly.
async function discardComfyUiBody(upstream) {
  await upstream.body?.cancel().catch(() => undefined);
  return undefined;
}

function remainingComfyUiMs(deadline, ceilingMs) {
  return Math.max(1, Math.min(ceilingMs, deadline - Date.now()));
}

function sleepUnlessAborted(ms, signal) {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const finish = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", finish);
      resolve();
    };
    const timer = setTimeout(finish, ms);
    signal.addEventListener("abort", finish, { once: true });
  });
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function truncateText(value, maxLength) {
  const text = typeof value === "string" ? value : "";
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

function createClaudeStreamTextParser() {
  return createNdjsonStreamTextParser({
    extractDeltaText: (event) => {
      // With --include-partial-messages: {"type":"stream_event","event":{"type":"content_block_delta",
      // "delta":{"type":"text_delta","text":"..."}}}. Some builds surface the inner event at top level.
      const inner = event?.type === "stream_event" ? event.event : event;
      if (inner?.type === "content_block_delta" && inner?.delta?.type === "text_delta") {
        return typeof inner.delta.text === "string" ? inner.delta.text : "";
      }
      return "";
    },
    extractCompleteText: (event) => {
      if (event?.type === "result" && typeof event.result === "string") {
        return event.result;
      }
      if (event?.type === "assistant" && Array.isArray(event?.message?.content)) {
        return event.message.content
          .filter((block) => block?.type === "text" && typeof block.text === "string")
          .map((block) => block.text)
          .join("");
      }
      return "";
    }
  });
}

/**
 * Translates `agy --output-format stream-json` into plain text.
 *
 * Prose arrives as {"event":"step_update","step_update":{"step_type":"agent_response","text_delta":"..."}};
 * other step types (user_input, tool activity) carry no answer text and are skipped. The closing
 * {"event":"result","result":{"response":"..."}} repeats the whole answer and is the fallback when no delta
 * streamed. A failed run reports a non-SUCCESS status with an empty response — that is logged rather than
 * returned, so the bridge fails with "returned no content" instead of writing an error string into the turn.
 */
/**
 * The NDJSON parser for an agent's stdout, or undefined when it emits plain text.
 *
 * `wantStream` matters for claude only: it speaks NDJSON just for the streaming argv, and running the parser
 * over its plain-text reply would return nothing. agy has a single NDJSON argv, so its stdout ALWAYS needs
 * translating — including on the non-streaming path the annotation call takes, which otherwise hands the
 * caller a wall of raw NDJSON where a JSON answer was expected.
 */
function resolveCliStdoutParser(agent, wantStream) {
  if (agent === "antigravity") {
    return createAntigravityStreamTextParser();
  }
  if (agent === "claude" && wantStream) {
    return createClaudeStreamTextParser();
  }
  return undefined;
}

function createAntigravityStreamTextParser() {
  return createNdjsonStreamTextParser({
    extractDeltaText: (event) => {
      const step = event?.event === "step_update" ? event.step_update : undefined;
      if (step?.step_type !== "agent_response" || typeof step?.text_delta !== "string") {
        return "";
      }
      return step.text_delta;
    },
    extractCompleteText: (event) => {
      const result = event?.event === "result" ? event.result : undefined;
      if (!result) {
        return "";
      }
      if (typeof result.response === "string" && result.response) {
        return result.response;
      }
      if (result.status && result.status !== "SUCCESS") {
        console.error(`agy run failed (${result.status}): ${result.error ?? "no error text"}`);
      }
      return "";
    }
  });
}

async function getObjectAsset(_request, response, [objectKey]) {
  let objectRoots;
  try {
    objectRoots = await getRequestObjectRoots(_request);
  } catch (error) {
    console.error("getObjectAsset: failed to resolve object roots", error);
    sendJson(response, 500, { error: "Internal server error." });
    return;
  }

  const objectFile = resolveExistingObjectFile(objectKey, objectRoots);
  if (!objectFile) {
    sendJson(response, 404, { error: "Object not found" });
    return;
  }

  let bytes;
  try {
    bytes = await readFile(objectFile.filePath);
  } catch (error) {
    console.error("getObjectAsset: failed to read object file", error);
    sendJson(response, 500, { error: "Internal server error." });
    return;
  }
  response.writeHead(200, {
    ...corsHeaders,
    "content-type": mimeTypeForObjectKey(objectKey),
    "content-length": String(bytes.byteLength),
    "cache-control": "public, max-age=31536000, immutable"
  });
  response.end(bytes);
}

async function listSimulations(request, response) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const store = await readStateStore();
  const scope = readRequestScope(request);
  const simulations = Object.values(store.simulations ?? {}).filter((state) => state?.simulation?.ownerId === scope.ownerId);

  sendJson(response, 200, simulations.map((state) => redactResponseSecrets(stripAssetPayloadsFromState(state))));
}

async function createSimulation(request, response) {
  // POST /simulations is a PUT in disguise: it reaches the same upsertSimulationState sink as
  // saveSimulationState with no id-vs-URL check and no existence check, so an id that already exists is
  // overwritten wholesale — transcript included — and the extracted secrets are written to the vault
  // store. Guarding the state routes and leaving this one open left the write side of the same store open.
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const state = await readJsonBody(request);
  if (!assertRequestScope(response, request, state)) {
    return;
  }
  await upsertSimulationState(state);
  sendJson(response, 201, redactResponseSecrets(state));
}

async function getSimulation(_request, response, [simulationId]) {
  if (rejectNonLocalSecretRequest(_request, response)) {
    return;
  }
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestOwnerScope(response, _request, state)) {
    return;
  }

  sendJson(response, 200, redactResponseSecrets(stripAssetPayloadsFromState(state)));
}

async function saveSimulationState(request, response, [simulationId]) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const state = await readJsonBody(request);
  if (state?.simulation?.id !== simulationId) {
    sendJson(response, 400, { error: "Simulation ID mismatch" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  await upsertSimulationState(state);
  send(response, 204);
}

/**
 * Rejects a browser request that did not come from this machine's own app.
 *
 * The vault answers with DECODED provider keys and the NovelAI token (decodeSecretRecord base64-decodes
 * `encodedSecret` back to plaintext), the endpoint has no authentication, and `ownerId` falls back to
 * "local_user" — the very owner the app itself writes under. With access-control-allow-origin defaulting to
 * "*", a credential-less GET is a simple request that any page the user happens to be visiting could make
 * against 127.0.0.1 and then READ, lifting every stored API key.
 *
 * Origin is the right discriminator: browsers attach it to every cross-origin fetch and cannot be made to
 * forge it, while the app's own traffic carries either a localhost origin or (for non-browser callers such
 * as the eval scripts) none at all. DYNAMICCHAT_CORS_ORIGIN, when set, is honoured as an extra allowed
 * origin so a configured deployment keeps working.
 *
 * The rule for who gets it: every handler that touches the persisted state store or the secret store,
 * read or write, PLUS any handler that can spend something of the user's. Nothing narrower survives
 * contact — the transcript is readable through /simulations, through a single simulation, through its
 * assets and through its audit log, and writable through both POST /simulations and PUT .../state;
 * picking a subset just moves the door.
 *
 * Left off deliberately are /health, GET /objects/*, which needs an object key only a guarded route hands
 * out and is loaded by the app as an <img> subresource that sends no Origin at all, and the relays that
 * really are stateless and really do run on credentials the caller supplies in the request: /novelai/*
 * and POST /llm/chat. POST /llm/cli-agent is NOT in that set and IS guarded — it runs the host's own
 * signed-in CLI, so the cost of leaving it open is the user's subscription, not the caller's key. The four
 * /photoreal/comfyui/* routes are guarded for the same reason: there are no caller credentials in a ComfyUI
 * call, what they spend is this machine's GPU, and generate writes reference images into ComfyUI's input
 * directory.
 *
 * "Local" is not the same as "this app". A ComfyUI web UI is served from a loopback origin too, so
 * rejectNonLocalSecretRequest refuses ComfyUI origins (isComfyUiBrowserOrigin) before this check is consulted,
 * and a missing Origin is only trusted once the Host header has been checked — see there.
 */
function isLocalBrowserOrigin(origin) {
  if (!origin) {
    // No Origin header: a non-browser client (curl, the eval harness) or a same-origin GET. A cross-origin
    // fetch always carries one; a same-origin GET does not, and that includes a page whose own hostname has
    // been rebound to 127.0.0.1. rejectNonLocalSecretRequest has already refused that one by its Host header.
    return true;
  }
  const configured = process.env.DYNAMICCHAT_CORS_ORIGIN;
  if (configured && configured !== "*" && origin === configured) {
    return true;
  }
  try {
    const { hostname, protocol } = new URL(origin);
    return (protocol === "http:" || protocol === "https:") && (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]" || hostname === "::1");
  } catch {
    // Opaque origins ("null" from a sandboxed iframe or a file:// page) are not the local app.
    return false;
  }
}

/**
 * Loopback origins that are NOT this app: a ComfyUI web UI.
 *
 * ComfyUI serves its UI from http://127.0.0.1:8188, and every custom node installed into it may ship frontend
 * JavaScript that runs on that origin. Custom nodes install from a public registry in one click, so a malicious
 * or compromised one is a supply-chain event rather than a hypothetical, and its script passes
 * isLocalBrowserOrigin exactly as the app does: it could read the decoded keys off /personal-api-vault, the
 * transcripts off /simulations, or drive the CLI bridge. The photoreal backend makes running ComfyUI next to
 * DynamicChat the normal setup, which is why the list applies to every guarded route and not only to the relay.
 *
 * What is denied is a server — a host class and a port — not an origin string. A browser treats every spelling of
 * loopback as its own origin and ComfyUI answers on all of them, so a list of exact origins left each spelling but
 * the listed one open: ComfyUI prints and opens http://127.0.0.1:<port> even when DYNAMICCHAT_COMFYUI_URL says
 * localhost. So on a ComfyUI port, every loopback spelling (isLoopbackHostname) is denied under http and https
 * alike — ComfyUI keeps its port when started with --tls-keyfile. The ComfyUI ports are 8188, the port of
 * DYNAMICCHAT_COMFYUI_URL, and the port of every DYNAMICCHAT_DENIED_ORIGINS entry (comma list of origins or bare
 * host:port, the shape DYNAMICCHAT_COMFYUI_HOSTS uses) — the server cannot guess the port of a ComfyUI it has not
 * been told about. A non-loopback entry denies that host and port under either scheme. An entry that parses as
 * neither is reported on the console at startup instead of being dropped in silence. The list wins over
 * DYNAMICCHAT_CORS_ORIGIN.
 */
const COMFYUI_DEFAULT_PORT = Number(new URL(COMFYUI_DEFAULT_BASE_URL).port);
const COMFYUI_ORIGIN_DENY_LIST = buildComfyUiOriginDenyList();

function buildComfyUiOriginDenyList() {
  const loopbackPorts = new Set([COMFYUI_DEFAULT_PORT]);
  const endpoints = new Set();
  const add = (endpoint) => {
    if (endpoint.loopback) {
      loopbackPorts.add(endpoint.port);
    } else {
      endpoints.add(`${endpoint.hostname}:${endpoint.port}`);
    }
  };
  const configuredUrl = (process.env.DYNAMICCHAT_COMFYUI_URL ?? "").trim();
  if (configuredUrl) {
    const endpoint = parseServerEndpoint(configuredUrl);
    if (endpoint) {
      add(endpoint);
    } else {
      console.warn(`[security] DYNAMICCHAT_COMFYUI_URL=${JSON.stringify(configuredUrl)} is not an http(s) URL; its ComfyUI origin cannot be denied.`);
    }
  }
  for (const entry of splitEnvList(process.env.DYNAMICCHAT_DENIED_ORIGINS)) {
    const endpoint = parseServerEndpoint(/^[a-z][a-z\d+.-]*:\/\//iu.test(entry) ? entry : `http://${entry}`);
    if (endpoint) {
      add(endpoint);
    } else {
      console.warn(`[security] DYNAMICCHAT_DENIED_ORIGINS entry ${JSON.stringify(entry)} is neither an origin nor host:port; ignored.`);
    }
  }
  return { loopbackPorts, endpoints };
}

function isComfyUiBrowserOrigin(origin) {
  const endpoint = parseServerEndpoint(origin);
  if (!endpoint) {
    return false;
  }
  return endpoint.loopback
    ? COMFYUI_ORIGIN_DENY_LIST.loopbackPorts.has(endpoint.port)
    : COMFYUI_ORIGIN_DENY_LIST.endpoints.has(`${endpoint.hostname}:${endpoint.port}`);
}

/**
 * An origin or http(s) URL reduced to the server behind it: the canonical hostname, the effective port (80/443 when
 * the URL leaves it out) and whether that host is loopback. The scheme is dropped on purpose — see the deny list
 * above. Undefined for anything that is not an http(s) URL: "null" (opaque origins), file://, garbage.
 */
function parseServerEndpoint(value) {
  if (typeof value !== "string" || !value.trim()) {
    return undefined;
  }
  let url;
  try {
    url = new URL(value.trim());
  } catch {
    return undefined;
  }
  if ((url.protocol !== "http:" && url.protocol !== "https:") || !url.hostname) {
    return undefined;
  }
  return {
    origin: url.origin,
    hostname: url.hostname,
    port: url.port ? Number(url.port) : url.protocol === "https:" ? 443 : 80,
    loopback: isLoopbackHostname(url.hostname)
  };
}

/** One server: the same port, and either both loopback (in any spelling) or the same host. */
function isSameServerEndpoint(left, right) {
  return left.port === right.port && ((left.loopback && right.loopback) || left.hostname === right.hostname);
}

/**
 * The Host names a browser may legitimately have used to reach this server, beyond loopback.
 *
 * Origin alone cannot stop DNS rebinding. A page on http://attacker.example:8788 that rebinds its own name to
 * 127.0.0.1 talks to this server SAME-origin, and a same-origin GET carries no Origin header at all: it passed the
 * guard and read the decoded keys off GET /personal-api-vault, the transcripts off GET /simulations, and used GET
 * /photoreal/comfyui/status as a ComfyUI inventory and a loopback port scanner. What such a request cannot change is
 * its Host header — the browser sends the name the page was loaded from. This server listens on 127.0.0.1 only, so
 * the app reaches it as localhost, a 127.x address or [::1] (any port); any other name is a rebound page unless the
 * operator put a proxy in front on purpose and listed its name in DYNAMICCHAT_ALLOWED_HOSTS (comma list; `name`
 * allows any port, `name:port` only that one). Non-loopback IP literals are not exempt: they cannot be rebound, but
 * nothing reaches a 127.0.0.1 socket under one except through such a proxy, which is what the list is for.
 */
const ALLOWED_HOST_HEADERS = buildAllowedHostHeaders();

function buildAllowedHostHeaders() {
  const hosts = [];
  for (const entry of splitEnvList(process.env.DYNAMICCHAT_ALLOWED_HOSTS)) {
    // Lenient about a pasted URL: the scheme and any path are not part of a Host header.
    const parsed = parseHostHeader(entry.replace(/^[a-z][a-z\d+.-]*:\/\//iu, "").replace(/\/.*$/su, ""));
    if (parsed) {
      hosts.push(parsed);
    } else {
      console.warn(`[security] DYNAMICCHAT_ALLOWED_HOSTS entry ${JSON.stringify(entry)} is not a host or host:port; ignored.`);
    }
  }
  return hosts;
}

/** A Host header (or an allow-list entry) as { hostname, port }: canonical hostname, port undefined when absent. */
function parseHostHeader(value) {
  const match = /^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+)(?::(\d{1,5}))?$/u.exec(String(value ?? "").trim());
  const hostname = match ? canonicalHostname(match[1]) : "";
  const port = match?.[2] === undefined ? undefined : Number(match[2]);
  if (!hostname || (port !== undefined && port > 65535)) {
    return undefined;
  }
  return { hostname, port };
}

function isAllowedHostHeader(value) {
  if (value === undefined) {
    // Only a client without a browser omits Host (Node answers 400 to an HTTP/1.1 request that does), and such
    // a client is already trusted the way a missing Origin is.
    return true;
  }
  const host = parseHostHeader(value);
  if (!host) {
    return false;
  }
  return (
    isLoopbackHostname(host.hostname) ||
    ALLOWED_HOST_HEADERS.some((allowed) => allowed.hostname === host.hostname && (allowed.port === undefined || allowed.port === host.port))
  );
}

function rejectNonLocalSecretRequest(request, response) {
  if (!isAllowedHostHeader(request.headers.host)) {
    sendJson(response, 403, {
      error: "forbidden_host",
      message:
        "DynamicChat API는 이 컴퓨터 주소(127.0.0.1 · localhost)로 들어온 요청만 받습니다. 다른 이름으로 연결해야 한다면 API 서버의 DYNAMICCHAT_ALLOWED_HOSTS 환경변수에 그 이름을 추가하세요."
    });
    return true;
  }
  const origin = request.headers.origin?.toString();
  if (isComfyUiBrowserOrigin(origin)) {
    sendJson(response, 403, {
      error: "forbidden_origin",
      message: "ComfyUI 웹 UI에서 온 요청은 받지 않습니다. 이 데이터는 이 컴퓨터의 DynamicChat 앱에서만 접근할 수 있습니다."
    });
    return true;
  }
  if (isLocalBrowserOrigin(origin)) {
    return false;
  }
  sendJson(response, 403, {
    error: "forbidden_origin",
    message: "이 데이터는 이 컴퓨터의 DynamicChat 앱에서만 접근할 수 있습니다."
  });
  return true;
}

function splitEnvList(value) {
  return String(value ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

async function getPersonalApiVault(request, response) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const scope = readRequestScope(request);
  const secrets = await readSecretStore();
  const ownerSecrets = secrets.owners?.[scope.ownerId] ?? {};

  sendJson(response, 200, decodePersonalApiVault(ownerSecrets));
}

async function savePersonalApiVault(request, response) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const scope = readRequestScope(request);
  const vault = await readJsonBody(request);
  const secrets = await readSecretStore();
  const ownerSecrets = secrets.owners?.[scope.ownerId] ?? {};

  secrets.owners = {
    ...(secrets.owners ?? {}),
    [scope.ownerId]: {
      ...ownerSecrets,
      personalApiVault: encodePersonalApiVault(vault, ownerSecrets.personalApiVault),
      updatedAt: new Date().toISOString()
    }
  };

  await writeSecretStore(secrets);
  send(response, 204);
}

async function createPromptModule(request, response, [simulationId]) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const module = await readJsonBody(request);
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  state.modules = [...(state.modules ?? []), module];
  state.simulation.updatedAt = new Date().toISOString();
  appendAuditEvent(state, request, "state_saved", "prompt_module", module.id, { operation: "create_prompt_module" });
  await writeStateStore(store);
  sendJson(response, 201, module);
}

async function updatePromptModule(request, response, [moduleId]) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const patch = await readJsonBody(request);
  const store = await readStateStore();
  const { state, module } = findModule(store, moduleId);
  if (!state || !module) {
    sendJson(response, 404, { error: "Prompt module not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  const nextModule = { ...module, ...patch, updatedAt: new Date().toISOString() };
  state.modules = state.modules.map((candidate) => (candidate.id === moduleId ? nextModule : candidate));
  state.simulation.updatedAt = new Date().toISOString();
  appendAuditEvent(state, request, "state_saved", "prompt_module", moduleId, { operation: "update_prompt_module" });
  await writeStateStore(store);
  sendJson(response, 200, nextModule);
}

async function createChatTurn(_request, response) {
  sendJson(response, 501, { error: "Chat turn execution is still handled by the React runtime in this MVP server." });
}

async function resetSession(_request, response) {
  sendJson(response, 501, { error: "Session reset execution is tracked by AIN-19." });
}

async function createImageJob(request, response, [simulationId]) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const body = await readJsonBody(request);
  const now = new Date().toISOString();
  const job = {
    id: `imgjob_${Date.now().toString(36)}`,
    simulationId,
    sessionId: body.sessionId ?? "server_pending",
    turnId: body.turnId,
    status: "queued",
    reason: body.reason ?? "Manual image job requested",
    prompt: body.prompt ?? "",
    negativePrompt: body.negativePrompt ?? "",
    providerPayload: body.providerPayload ?? {},
    assetIds: [],
    contextNodeIds: body.contextNodeIds ?? [],
    createdAt: now
  };
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  state.imageJobs = [...(state.imageJobs ?? []), job];
  appendAuditEvent(state, request, "generation_job_created", "image_job", job.id, { status: job.status, source: "server" });
  await writeStateStore(store);
  sendJson(response, 201, job);
}

async function listAssets(request, response, [simulationId]) {
  // Answers with image asset metadata and, for requested ids, hydrated data URLs — the same simulation
  // content getSimulation guards, through a different door.
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestOwnerScope(response, request, state)) {
    return;
  }

  const objectRoot = await getRequestObjectRoot(request);
  const requestedAssetIds = readRequestedAssetIds(request);
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  const storedAssets = collectHydratedImageAssets(state).filter((asset) => !deletedAssetIds.has(asset.id));
  const storedAssetIds = new Set(storedAssets.map((asset) => asset.id));
  const recoveredAssets = requestedAssetIds.size === 0
    ? []
    : [...requestedAssetIds]
        .filter((assetId) => !deletedAssetIds.has(assetId))
        .filter((assetId) => !storedAssetIds.has(assetId))
        .map((assetId) => createRecoveredImageAsset(state, assetId));
  const allCandidates = [...storedAssets, ...recoveredAssets]
    .filter((asset) => !deletedAssetIds.has(asset.id))
    .filter((asset) => requestedAssetIds.size === 0 || requestedAssetIds.has(asset.id))
    .map(stripAssetDataUrl);

  const assets =
    requestedAssetIds.size === 0
      ? allCandidates
      : await hydrateAssetCollection(simulationId, allCandidates, objectRoot);
  appendAuditEvent(state, request, "asset_accessed", "simulation", simulationId, { assetCount: assets.length });
  void writeStateStore(store).catch(() => undefined);
  sendJson(response, 200, assets);
}

async function persistImageAssets(request, response, [simulationId]) {
  // This is the route the object-key traversal is reached through, and it writes files. The scope check
  // below is not a gate on its own — readRequestScope falls back to the stored ownerId, so a header-less
  // caller always satisfies it — so the origin guard is what keeps an arbitrary page off it.
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const body = await readJsonBody(request);
  const assets = Array.isArray(body?.assets) ? body.assets : [];
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (state && !assertRequestOwnerScope(response, request, state)) {
    return;
  }

  const objectRoot = await getRequestObjectRoot(request);
  const persistedAssets = await persistAssetObjects(simulationId, assets, objectRoot);
  sendJson(response, 200, persistedAssets.map(stripAssetDataUrl));
}

function readRequestedAssetIds(request) {
  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);
  return new Set(
    url.searchParams
      .getAll("ids")
      .flatMap((value) => value.split(","))
      .map((value) => value.trim())
      .filter(Boolean)
  );
}

async function cancelImageJob(request, response, [jobId]) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const store = await readStateStore();
  const { state, job } = findImageJob(store, jobId);
  if (!state || !job) {
    sendJson(response, 404, { error: "Image job not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  const nextJob = { ...job, status: "canceled", completedAt: new Date().toISOString() };
  state.imageJobs = state.imageJobs.map((candidate) => (candidate.id === jobId ? nextJob : candidate));
  appendAuditEvent(state, request, "generation_job_completed", "image_job", jobId, { status: "canceled" });
  await writeStateStore(store);
  sendJson(response, 200, nextJob);
}

async function getImageJob(request, response, [jobId]) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const store = await readStateStore();
  const { state, job } = findImageJob(store, jobId);
  if (!job) {
    sendJson(response, 404, { error: "Image job not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  sendJson(response, 200, job);
}

async function listAudit(request, response, [simulationId]) {
  // The audit log is not innocuous: appendSecretAuditEvents records api_secret_stored entries carrying
  // the secretRef ("dev:local_user:llm") plus provider and model, so this route names what the vault holds.
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  sendJson(response, 200, state.auditLog ?? []);
}

async function createBackup(request, response, [simulationId]) {
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  const backup = await writeBackupFile(store, `manual-${simulationId}`);
  appendAuditEvent(state, request, "backup_created", "backup", backup.id, {
    fileName: backup.fileName,
    simulationCount: Object.keys(store.simulations ?? {}).length
  });
  await writeStateStore(store);
  sendJson(response, 201, backup);
}

async function createRedaction(request, response, [simulationId]) {
  // The most destructive route on the server: applyRedaction drops memory events, rewrites prompt module
  // bodies to "[redacted]" and rm()s the backing object files, then the result is persisted. There is no
  // undo, so it needs the guard more than the read routes do.
  if (rejectNonLocalSecretRequest(request, response)) {
    return;
  }
  const body = await readJsonBody(request);
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  const result = await applyRedaction(state, request, body);
  await writeStateStore(store);
  sendJson(response, 200, result);
}

async function upsertSimulationState(inputState) {
  validateAppState(inputState);
  // Pinned before the store is read and unpinned only once it is written: in between, the profile images this save
  // names are in no store that a concurrent save's cleanup could check.
  const pinnedProfileObjectKeys = pinProfileObjectKeys(inputState);
  try {
    const store = await readStateStore();
    const secrets = await readSecretStore();
    const previous = store.simulations[inputState.simulation.id];
    const { state, secretPatch } = await extractPersistentState(inputState, secrets.owners[inputState.simulation.ownerId]);

    appendSecretAuditEvents(state, secretPatch);
    store.simulations[state.simulation.id] = state;
    if (secretPatch) {
      const ownerId = state.simulation.ownerId;
      secrets.owners[ownerId] = {
        ...(secrets.owners[ownerId] ?? {}),
        ...secretPatch
      };
      await writeSecretStore(secrets);
    }

    await writeStateStore(store);
    await deleteReleasedProfileObjects(previous, state, store, secrets);
  } finally {
    unpinProfileObjectKeys(pinnedProfileObjectKeys);
  }
}

async function extractPersistentState(inputState, existingSecrets = {}) {
  const state = structuredClone(inputState);
  const ownerId = state.simulation.ownerId;
  const updatedAt = new Date().toISOString();
  const secretPatch = {};

  if (state.llm?.apiKey) {
    const encodedSecret = Buffer.from(state.llm.apiKey, "utf8").toString("base64");
    if (existingSecrets.llm?.encodedSecret !== encodedSecret || existingSecrets.llm?.provider !== state.llm.provider || existingSecrets.llm?.model !== state.llm.model) {
      secretPatch.llm = {
      provider: state.llm.provider,
      model: state.llm.model,
      secretRef: `dev:${ownerId}:llm`,
      encodedSecret,
      updatedAt
      };
    }
    state.llm.apiKey = "";
  }

  if (state.novelAi?.apiKey) {
    const encodedSecret = Buffer.from(state.novelAi.apiKey, "utf8").toString("base64");
    if (existingSecrets.novelAi?.encodedSecret !== encodedSecret || existingSecrets.novelAi?.accountLabel !== state.novelAi.accountLabel) {
      secretPatch.novelAi = {
      provider: "novelai",
      accountLabel: state.novelAi.accountLabel,
      secretRef: `dev:${ownerId}:novelai`,
      encodedSecret,
      updatedAt
      };
    }
    state.novelAi.apiKey = "";
  }

  const objectRoot = getOwnerObjectRoot(existingSecrets);
  state.imageAssets = await persistAssetObjects(state.simulation.id, state.imageAssets ?? [], objectRoot);
  state.progressRuns = await Promise.all(
    (state.progressRuns ?? []).map(async (run) => ({
      ...run,
      imageAssets: await persistAssetObjects(run.simulationId ?? state.simulation.id, run.imageAssets ?? [], objectRoot)
    }))
  );
  return {
    state,
    secretPatch: Object.keys(secretPatch).length > 0 ? secretPatch : undefined
  };
}

async function persistAssetObjects(simulationId, assets, objectRoot = defaultObjectDir) {
  // Both halves of the object key arrive from the request. Without this, POSTing an asset id of
  // "../../../pwned" wrote the caller's bytes three levels above <data>/objects/ and mkdir created the
  // directories on the way. Refuse the write and hand the asset back unpersisted rather than throwing:
  // an id this shape is not something the app itself ever produces, and the caller still gets its
  // metadata echoed exactly as it does for an unparseable data URL.
  if (!isSafeObjectPathSegment(simulationId)) {
    return assets;
  }
  const simulationObjectDir = resolveObjectPath(simulationId, objectRoot);
  await mkdir(simulationObjectDir, { recursive: true });

  return Promise.all(
    assets.map(async (asset) => {
      if (!asset.dataUrl) {
        return attachExistingAssetObjectMetadata(simulationId, asset, objectRoot);
      }

      const parsed = parseDataUrl(asset.dataUrl);
      if (!parsed) {
        return asset;
      }

      if (!isSafeObjectPathSegment(asset.id)) {
        return asset;
      }

      const extension = extensionForMimeType(parsed.mimeType);
      const objectKey = `${simulationId}/${asset.id}.${extension}`;
      const filePath = resolveObjectPath(objectKey, objectRoot);
      await mkdir(path.dirname(filePath), { recursive: true });
      await writeFile(filePath, parsed.bytes);

      const { dataUrl: _dataUrl, ...metadataOnly } = asset;
      return {
        ...metadataOnly,
        objectKey,
        mimeType: asset.mimeType ?? parsed.mimeType
      };
    })
  );
}

function attachExistingAssetObjectMetadata(simulationId, asset, objectRoot = defaultObjectDir) {
  const objectFile = resolveAssetObjectFile(simulationId, asset, objectRoot);
  if (!objectFile) {
    return asset;
  }

  return {
    ...asset,
    objectKey: objectFile.objectKey,
    mimeType: asset.mimeType ?? mimeTypeForObjectKey(objectFile.objectKey)
  };
}

function stripAssetPayloadsFromState(state) {
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  return {
    ...state,
    visualProfiles: (state.visualProfiles ?? []).map((profile) => ({
      ...profile,
      referenceImageAssetIds: (profile.referenceImageAssetIds ?? []).filter((assetId) => !deletedAssetIds.has(assetId))
    })),
    messages: (state.messages ?? []).map((message) => filterMessageImageAssets(message, deletedAssetIds)),
    turnTraces: (state.turnTraces ?? []).map((trace) => filterTraceImageAssets(trace, deletedAssetIds)),
    imageAssets: (state.imageAssets ?? []).filter((asset) => !deletedAssetIds.has(asset.id)).map(stripAssetDataUrl),
    imageJobs: (state.imageJobs ?? []).map((job) => filterJobImageAssets(job, deletedAssetIds)),
    progressRuns: (state.progressRuns ?? []).map((run) => ({
      ...run,
      messages: (run.messages ?? []).map((message) => filterMessageImageAssets(message, deletedAssetIds)),
      turnTraces: (run.turnTraces ?? []).map((trace) => filterTraceImageAssets(trace, deletedAssetIds)),
      imageAssets: (run.imageAssets ?? []).filter((asset) => !deletedAssetIds.has(asset.id)).map(stripAssetDataUrl),
      imageJobs: (run.imageJobs ?? []).map((job) => filterJobImageAssets(job, deletedAssetIds))
    }))
  };
}

function stripAssetDataUrl(asset) {
  if (!asset?.dataUrl) {
    return asset;
  }

  const { dataUrl: _dataUrl, ...metadataOnly } = asset;
  return metadataOnly;
}

async function hydrateAssetDataUrls(state, objectRoot = defaultObjectDir) {
  const simulationId = state.simulation?.id;
  return {
    ...state,
    imageAssets: await hydrateAssetCollection(simulationId, state.imageAssets ?? [], objectRoot),
    progressRuns: await Promise.all(
      (state.progressRuns ?? []).map(async (run) => ({
        ...run,
        imageAssets: await hydrateAssetCollection(run.simulationId ?? simulationId, run.imageAssets ?? [], objectRoot)
      }))
    )
  };
}

function collectHydratedImageAssets(state) {
  const assetsById = new Map();
  for (const asset of [
    ...(state.imageAssets ?? []),
    ...(state.progressRuns ?? []).flatMap((run) => run.imageAssets ?? [])
  ]) {
    if (!asset?.id) {
      continue;
    }

    assetsById.set(asset.id, mergeHydratedAsset(assetsById.get(asset.id), asset));
  }

  return Array.from(assetsById.values());
}

function mergeHydratedAsset(existing, next) {
  if (!existing) {
    return next;
  }

  return {
    ...existing,
    ...next,
    dataUrl: existing.dataUrl ?? next.dataUrl,
    objectKey: existing.objectKey ?? next.objectKey,
    mimeType: existing.mimeType ?? next.mimeType
  };
}

function collectDeletedImageAssetIds(state) {
  return new Set(
    (state.redactionQueue ?? [])
      .filter((redaction) => redaction?.targetType === "image_asset" && redaction.status !== "failed")
      .map((redaction) => redaction.targetId)
      .filter(Boolean)
  );
}

function filterMessageImageAssets(message, deletedAssetIds) {
  return {
    ...message,
    imageAssetIds: (message.imageAssetIds ?? []).filter((assetId) => !deletedAssetIds.has(assetId))
  };
}

function filterTraceImageAssets(trace, deletedAssetIds) {
  const removedCount = (trace.imageAssetIds ?? []).filter((assetId) => deletedAssetIds.has(assetId)).length;
  return {
    ...trace,
    imageAssetIds: (trace.imageAssetIds ?? []).filter((assetId) => !deletedAssetIds.has(assetId)),
    metrics: trace.metrics
      ? {
          ...trace.metrics,
          imageAssetCount: Math.max(0, (trace.metrics.imageAssetCount ?? 0) - removedCount)
        }
      : trace.metrics
  };
}

function filterJobImageAssets(job, deletedAssetIds) {
  return {
    ...job,
    assetIds: (job.assetIds ?? []).filter((assetId) => !deletedAssetIds.has(assetId)),
    representativeAssetId: deletedAssetIds.has(job.representativeAssetId) ? undefined : job.representativeAssetId
  };
}

function createRecoveredImageAsset(state, assetId) {
  const imageJob = findStateImageJobForAsset(state, assetId);
  const now = new Date().toISOString();
  return {
    id: assetId,
    simulationId: state.simulation?.id,
    title: `Recovered image ${assetId}`,
    source: "generated",
    prompt: imageJob?.prompt ?? "",
    negativePrompt: imageJob?.negativePrompt ?? "",
    safetyLevel: state.imageProfile?.safetyLevel ?? "safe",
    characterIds: [],
    tags: ["recovered"],
    createdAt: imageJob?.completedAt ?? imageJob?.updatedAt ?? imageJob?.createdAt ?? state.simulation?.updatedAt ?? now,
    jobId: imageJob?.id,
    palette: ["#f4f0e8", "#d7c7aa", "#5f5046"],
    providerMetadata: imageJob?.providerPayload
  };
}

function findStateImageJobForAsset(state, assetId) {
  return [
    ...(state.imageJobs ?? []),
    ...(state.progressRuns ?? []).flatMap((run) => run.imageJobs ?? [])
  ].find((job) => (job.assetIds ?? []).includes(assetId) || job.representativeAssetId === assetId);
}

async function hydrateAssetCollection(simulationId, assets, objectRoot = defaultObjectDir) {
  return Promise.all(assets.map((asset) => hydrateAssetDataUrl(simulationId, asset, objectRoot)));
}

async function hydrateAssetDataUrl(simulationId, asset, objectRoot = defaultObjectDir) {
  if (asset.dataUrl) {
    return asset;
  }

  const objectFile = resolveAssetObjectFile(simulationId, asset, objectRoot);
  if (!objectFile) {
    return asset;
  }

  const bytes = await readFile(objectFile.filePath);
  const mimeType = asset.mimeType ?? mimeTypeForObjectKey(objectFile.objectKey);
  return {
    ...asset,
    objectKey: objectFile.objectKey,
    mimeType,
    dataUrl: `data:${mimeType};base64,${bytes.toString("base64")}`
  };
}

function resolveAssetObjectFile(simulationId, asset, objectRoot = defaultObjectDir) {
  const objectKey = resolveAssetObjectKey(simulationId, asset, objectRoot);
  return objectKey ? resolveExistingObjectFile(objectKey, getFallbackObjectRoots(objectRoot)) : undefined;
}

function resolveAssetObjectKey(simulationId, asset, objectRoot = defaultObjectDir) {
  if (asset.objectKey && objectKeyExists(asset.objectKey, objectRoot)) {
    return asset.objectKey;
  }

  const extensions = preferredAssetExtensions(asset);
  const scopedSimulationIds = Array.from(new Set([asset.simulationId, simulationId].filter(Boolean)));
  for (const scopedSimulationId of scopedSimulationIds) {
    for (const extension of extensions) {
      const candidate = `${scopedSimulationId}/${asset.id}.${extension}`;
      if (objectKeyExists(candidate, objectRoot)) {
        return candidate;
      }
    }
  }

  return findAssetObjectKeyById(asset.id, extensions, objectRoot);
}

function findAssetObjectKeyById(assetId, extensions, objectRoot = defaultObjectDir) {
  if (!assetId) {
    return undefined;
  }

  for (const root of getFallbackObjectRoots(objectRoot)) {
    if (!existsSync(root)) {
      continue;
    }

    for (const entry of readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }

      for (const extension of extensions) {
        const candidate = `${entry.name}/${assetId}.${extension}`;
        if (resolveExistingObjectFile(candidate, [root])) {
          return candidate;
        }
      }
    }
  }

  return undefined;
}

function objectKeyExists(objectKey, objectRoot = defaultObjectDir) {
  return Boolean(resolveExistingObjectFile(objectKey, getFallbackObjectRoots(objectRoot)));
}

function resolveExistingObjectFile(objectKey, objectRoots = [defaultObjectDir]) {
  for (const objectRoot of uniqueObjectRoots(...objectRoots)) {
    try {
      const filePath = resolveObjectPath(objectKey, objectRoot);
      if (existsSync(filePath)) {
        return { objectKey, objectRoot, filePath };
      }
    } catch {
      continue;
    }
  }

  return undefined;
}

function getFallbackObjectRoots(objectRoot = defaultObjectDir) {
  return uniqueObjectRoots(objectRoot, defaultObjectDir);
}

function uniqueObjectRoots(...values) {
  const roots = [];
  const seen = new Set();
  for (const value of values) {
    const normalized = normalizeImageStoragePath(value);
    if (!normalized) {
      continue;
    }

    const resolved = path.resolve(normalized);
    const key = process.platform === "win32" ? resolved.toLowerCase() : resolved;
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    roots.push(resolved);
  }

  return roots;
}

function preferredAssetExtensions(asset) {
  return Array.from(new Set([extensionForMimeType(asset.mimeType), "png", "jpg", "jpeg", "webp"].filter(Boolean)));
}

function mimeTypeForObjectKey(objectKey) {
  const extension = objectKey.split(".").at(-1)?.toLowerCase();
  if (extension === "jpg" || extension === "jpeg") {
    return "image/jpeg";
  }

  if (extension === "webp") {
    return "image/webp";
  }

  return "image/png";
}

async function writeBackupFile(store, reason) {
  const createdAt = new Date().toISOString();
  const id = `backup_${Date.now().toString(36)}`;
  const fileName = `${createdAt.replace(/[:.]/gu, "-")}_${reason}.json`;
  const filePath = path.join(backupDir, fileName);
  await mkdir(backupDir, { recursive: true });
  await writeFile(
    filePath,
    `${JSON.stringify(
      {
        id,
        reason,
        createdAt,
        schemaVersion: 1,
        store
      },
      null,
      2
    )}\n`
  );

  return {
    id,
    reason,
    createdAt,
    fileName
  };
}

async function applyRedaction(state, request, body) {
  const targetType = body.targetType;
  const targetId = body.targetId;
  const reason = body.reason ?? "operator redaction";
  const now = new Date().toISOString();
  const scope = readRequestScope(request, state);
  const redaction = {
    id: `redact_${Date.now().toString(36)}`,
    simulationId: state.simulation.id,
    ownerId: state.simulation.ownerId,
    scope,
    targetType,
    targetId,
    reason,
    neuralMapNodeIds: Array.isArray(body.neuralMapNodeIds) ? body.neuralMapNodeIds : [],
    status: "applied",
    createdAt: now,
    completedAt: now
  };

  if (targetType === "memory_event") {
    const memory = state.memoryEvents?.find((event) => event.id === targetId);
    const nodeIds = [targetId, memory?.neuralMapNodeId, ...redaction.neuralMapNodeIds].filter(Boolean);
    redaction.neuralMapNodeIds = [...new Set(nodeIds)];
    state.memoryEvents = (state.memoryEvents ?? []).filter((event) => event.id !== targetId);
    state.contextPacks = (state.contextPacks ?? []).map((pack) => ({
      ...pack,
      evidence: (pack.evidence ?? []).filter((item) => !redaction.neuralMapNodeIds.includes(item.nodeId))
    }));
    state.messages = (state.messages ?? []).map((message) => ({
      ...message,
      referencedNodeIds: (message.referencedNodeIds ?? []).filter((nodeId) => !redaction.neuralMapNodeIds.includes(nodeId))
    }));
    appendAuditEvent(state, request, "memory_redacted", "memory_event", targetId, { nodeIds: redaction.neuralMapNodeIds });
  } else if (targetType === "prompt_module") {
    state.modules = (state.modules ?? []).map((module) =>
      module.id === targetId
        ? {
            ...module,
            title: `${module.title} (redacted)`,
            body: "[redacted]",
            enabled: false,
            activationTags: [],
            tokenPolicy: "disabled",
            updatedAt: now
          }
        : module
    );
    state.contextPacks = (state.contextPacks ?? []).map((pack) => ({
      ...pack,
      evidence: (pack.evidence ?? []).filter((item) => item.nodeId !== targetId)
    }));
    appendAuditEvent(state, request, "prompt_module_redacted", "prompt_module", targetId, {});
  } else if (targetType === "image_asset") {
    const assets = collectImageAssetsById(state, targetId);
    const asset = assets[0];
    state.imageAssets = (state.imageAssets ?? []).filter((candidate) => candidate.id !== targetId);
    state.messages = (state.messages ?? []).map((message) => removeImageAssetIdFromMessage(message, targetId));
    state.turnTraces = (state.turnTraces ?? []).map((trace) => removeImageAssetIdFromTrace(trace, targetId));
    state.imageJobs = (state.imageJobs ?? []).map((job) => removeImageAssetIdFromJob(job, targetId));
    state.visualProfiles = (state.visualProfiles ?? []).map((profile) => ({
      ...profile,
      referenceImageAssetIds: (profile.referenceImageAssetIds ?? []).filter((assetId) => assetId !== targetId)
    }));
    state.progressRuns = (state.progressRuns ?? []).map((run) => ({
      ...run,
      imageAssets: (run.imageAssets ?? []).filter((candidate) => candidate.id !== targetId),
      messages: (run.messages ?? []).map((message) => removeImageAssetIdFromMessage(message, targetId)),
      turnTraces: (run.turnTraces ?? []).map((trace) => removeImageAssetIdFromTrace(trace, targetId)),
      imageJobs: (run.imageJobs ?? []).map((job) => removeImageAssetIdFromJob(job, targetId))
    }));

    const deletedObjectKeys = await deleteImageAssetObjectFiles(request, state.simulation.id, targetId, assets);
    if (deletedObjectKeys.length > 0) {
      redaction.neuralMapNodeIds = deletedObjectKeys;
    }
    appendAuditEvent(state, request, "image_asset_deleted", "image_asset", targetId, { objectKey: asset?.objectKey, objectKeys: deletedObjectKeys });
  } else {
    redaction.status = "failed";
    redaction.error = `Unsupported redaction target: ${targetType}`;
  }

  state.redactionQueue = [...(state.redactionQueue ?? []), redaction];
  state.simulation.updatedAt = now;
  return {
    redaction,
    state: redactResponseSecrets(state)
  };
}

function collectImageAssetsById(state, assetId) {
  return [
    ...(state.imageAssets ?? []),
    ...(state.progressRuns ?? []).flatMap((run) => run.imageAssets ?? [])
  ].filter((asset) => asset?.id === assetId);
}

function removeImageAssetIdFromMessage(message, assetId) {
  return {
    ...message,
    imageAssetIds: (message.imageAssetIds ?? []).filter((candidateId) => candidateId !== assetId)
  };
}

function removeImageAssetIdFromTrace(trace, assetId) {
  const removed = (trace.imageAssetIds ?? []).includes(assetId);
  return {
    ...trace,
    imageAssetIds: (trace.imageAssetIds ?? []).filter((candidateId) => candidateId !== assetId),
    metrics: trace.metrics
      ? {
          ...trace.metrics,
          imageAssetCount: Math.max(0, (trace.metrics.imageAssetCount ?? 0) - (removed ? 1 : 0))
        }
      : trace.metrics
  };
}

function removeImageAssetIdFromJob(job, assetId) {
  return {
    ...job,
    assetIds: (job.assetIds ?? []).filter((candidateId) => candidateId !== assetId),
    representativeAssetId: job.representativeAssetId === assetId ? undefined : job.representativeAssetId
  };
}

async function deleteImageAssetObjectFiles(request, simulationId, assetId, assets) {
  const roots = await getRequestObjectRoots(request);
  const objectKeys = new Set(
    assets
      .flatMap((asset) => [asset.objectKey, resolveAssetObjectKey(simulationId, asset, roots[0] ?? defaultObjectDir)])
      .filter(Boolean)
  );

  for (const objectRoot of roots) {
    for (const extension of ["png", "jpg", "jpeg", "webp"]) {
      const directKey = `${simulationId}/${assetId}.${extension}`;
      if (objectKeyExists(directKey, objectRoot)) {
        objectKeys.add(directKey);
      }
    }

    if (!existsSync(objectRoot)) {
      continue;
    }

    for (const entry of readdirSync(objectRoot, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }

      for (const extension of ["png", "jpg", "jpeg", "webp"]) {
        const candidateKey = `${entry.name}/${assetId}.${extension}`;
        if (objectKeyExists(candidateKey, objectRoot)) {
          objectKeys.add(candidateKey);
        }
      }
    }
  }

  const deletedObjectKeys = [];
  for (const objectKey of objectKeys) {
    await Promise.all(
      roots.map(async (objectRoot) => {
        try {
          await rm(resolveObjectPath(objectKey, objectRoot), { force: true });
          deletedObjectKeys.push(objectKey);
        } catch {
          // Invalid legacy object keys are ignored during redaction cleanup.
        }
      })
    );
  }

  return [...new Set(deletedObjectKeys)];
}

function resolveObjectPath(objectKey, objectRoot = defaultObjectDir) {
  const root = path.resolve(objectRoot);
  const filePath = path.resolve(root, objectKey);
  if (filePath !== root && !filePath.startsWith(`${root}${path.sep}`)) {
    throw new Error("Invalid object key.");
  }
  return filePath;
}

/**
 * Whitelists a single path segment before it is used to build an object path.
 *
 * resolveObjectPath is the containment backstop and catches anything that escapes the root, but it is
 * the wrong and only line of defence for a segment we are about to *create*: it happily accepts a
 * simulation id or asset id that merely burrows sideways into another simulation's directory, or that
 * carries a drive letter, a UNC prefix, or a NUL. Both are attacker-supplied — persistImageAssets takes
 * the id straight out of the request body and the simulation id out of the URL — so they get an
 * allow-list here as well as the containment check there. Neither replaces the other.
 */
function isSafeObjectPathSegment(value) {
  return typeof value === "string" && value !== "." && value !== ".." && /^[A-Za-z0-9._-]+$/u.test(value);
}

/**
 * Deletes the profile images a save stopped naming, once nothing else names them.
 *
 * A profile image (CharacterVisualProfile.profileImages) is not an ImageAsset, so image_asset redaction never reaches
 * it: before this, a face the creator cleared or replaced in the builder stayed on disk for good, still served by
 * GET /objects/<key>. A save is the one moment the server sees such a reference go away, so the release happens
 * here — after the new state is on disk, and only for keys the stored version named and the incoming one does not.
 *
 * "Nothing else" means no stored simulation of any owner, because copySimulation carries the source's object keys
 * into the copy rather than copying the files, and it includes image jobs that can still run, whose payload names
 * the face reference they will load. The store is read again for the check instead of trusting this save's earlier
 * read, and keys pinned by saves still in progress count too.
 *
 * A save older than the stored version (simulation.updatedAt) releases nothing. The app fires saves without waiting
 * for the previous one, so an older snapshot can land after a newer one, and diffing it against the newer stored
 * version would "release" the profile the newer save had just set — the one on the user's screen. Server-side
 * writes (a redaction) also move updatedAt, so a current save can occasionally look older; that only keeps files.
 *
 * Never throws: a failed cleanup must not fail the save it follows, and whatever was not deleted stays in place.
 */
async function deleteReleasedProfileObjects(previous, next, writtenStore, secrets) {
  if (!previous) {
    return;
  }
  try {
    const stillNamed = new Set(listProfileObjectKeys(next).map(canonicalObjectKey));
    const released = [...new Set(listProfileObjectKeys(previous))].filter(
      (objectKey) => isProfileObjectKey(objectKey) && !stillNamed.has(canonicalObjectKey(objectKey))
    );
    if (released.length === 0) {
      return;
    }
    const simulationId = next.simulation.id;
    if (!isSaveAtLeastAsRecent(next, previous)) {
      console.warn(`Profile image cleanup: kept ${released.join(", ")}; this save of ${simulationId} is older than the stored version.`);
      return;
    }

    const namedKeys = collectProfileObjectKeys([...listStoredSimulations(writtenStore), ...listStoredSimulations(await readStateStore())]);
    const objectRoots = getOwnerObjectRoots(secrets.owners?.[next.simulation.ownerId], secrets);
    for (const objectKey of released) {
      const canonicalKey = canonicalObjectKey(objectKey);
      if (namedKeys.has(canonicalKey) || inFlightProfileObjectKeys.has(canonicalKey)) {
        continue;
      }
      // Every root: the file sits in whichever root was current when it was picked, and GET /objects serves it from any.
      for (const objectRoot of objectRoots) {
        if (await removeProfileObjectFile(objectKey, objectRoot)) {
          console.log(`Profile image cleanup: deleted ${objectKey} from ${objectRoot}; ${simulationId} no longer uses it and nothing else does.`);
        }
      }
    }
  } catch (error) {
    console.error("Profile image cleanup after a save failed; the files were left in place:", error);
  }
}

/**
 * Startup sweep of the builder's provisional folders (objects/draft_sim_*), which no save ever releases.
 *
 * A new simulation's profile picks are stored under its draft's createId("draft_sim") id, since the simulation has no
 * id until 생성, and the simulation keeps those keys once created. A draft cancelled after a pick therefore leaves a
 * folder no state names, and a pick replaced before 생성 leaves a file no state ever named; neither passes through
 * a save that could release it.
 *
 * A folder is considered only once it and everything in it are older than ABANDONED_DRAFT_PROFILE_AGE_MS (a builder
 * page open right now is still writing into its folder), and then loses each profile image no stored simulation
 * names, by the same check a save uses. An abandoned draft goes entirely; a draft that became a simulation keeps
 * exactly the picks the simulation uses. Anything else in a folder — a stray file, a subfolder, a link — is left
 * alone and keeps the folder.
 *
 * Every object root is swept: the default one and each owner's personal-vault imageStoragePath, the list
 * getOwnerObjectRoots already gives GET /objects and image_asset redaction, and only folders matching the draft
 * pattern inside them are touched. A store that cannot be read stops the sweep before anything is deleted, since it
 * would name no references at all.
 */
async function sweepAbandonedDraftProfileObjects() {
  try {
    const namedKeys = collectProfileObjectKeys(listStoredSimulations(await readStateStore()));
    const objectRoots = getOwnerObjectRoots({}, await readSecretStore());
    const cutoff = Date.now() - ABANDONED_DRAFT_PROFILE_AGE_MS;
    for (const objectRoot of objectRoots) {
      let entries;
      try {
        entries = await readdir(objectRoot, { withFileTypes: true });
      } catch (error) {
        if (error?.code !== "ENOENT") {
          console.error(`Profile image cleanup could not list ${objectRoot}:`, error);
        }
        continue;
      }
      for (const entry of entries) {
        // Dirent types are lstat types: a junction or symlink wearing a draft's name is not a directory here.
        if (entry.isDirectory() && DRAFT_PROFILE_DIRECTORY_PATTERN.test(entry.name)) {
          await sweepDraftProfileDirectory(objectRoot, entry.name, namedKeys, cutoff);
        }
      }
    }
  } catch (error) {
    console.error("Profile image cleanup at startup stopped:", error);
  }
}

async function sweepDraftProfileDirectory(objectRoot, directoryName, namedKeys, cutoff) {
  try {
    const directory = resolveObjectPath(directoryName, objectRoot);
    const folder = await lstat(directory);
    if (!folder.isDirectory()) {
      return;
    }
    const entries = await Promise.all(
      (await readdir(directory)).map(async (name) => ({ name, stats: await lstat(path.join(directory, name)) }))
    );
    // mtime, the time stamp a write moves. The folder's own counts too: adding or removing a pick moves it.
    if (Math.max(folder.mtimeMs, ...entries.map(({ stats }) => stats.mtimeMs)) > cutoff) {
      return;
    }

    const removed = [];
    for (const { name, stats } of entries) {
      const objectKey = `${directoryName}/${name}`;
      if (!stats.isFile() || !isProfileObjectKey(objectKey) || namedKeys.has(canonicalObjectKey(objectKey))) {
        continue;
      }
      if (await removeProfileObjectFile(objectKey, objectRoot)) {
        removed.push(name);
      }
    }
    // rmdir only removes an empty folder, which is the point: whatever is still in it was not ours to delete.
    const emptied = removed.length === entries.length && (await rmdir(directory).then(() => true, () => false));
    if (emptied) {
      console.log(`Profile image cleanup: removed the abandoned draft folder ${directoryName}${removed.length > 0 ? ` (${removed.join(", ")})` : ""} from ${objectRoot}.`);
    } else if (removed.length > 0) {
      console.log(`Profile image cleanup: removed ${removed.map((name) => `${directoryName}/${name}`).join(", ")} from ${objectRoot}; no simulation uses them.`);
    }
  } catch (error) {
    console.error(`Profile image cleanup skipped ${directoryName} in ${objectRoot}:`, error);
  }
}

/**
 * Unlinks one profile image, or does nothing and says so with false. The file and its folder are both lstat-ed, never
 * stat-ed, so no link is followed: on Windows a directory junction reads as a link at its own path, but a path THROUGH
 * one reads as an ordinary file, so checking the file alone would delete whatever the junction points at. The key's
 * shape is re-checked here as well as by the callers, because this is the function that deletes.
 */
async function removeProfileObjectFile(objectKey, objectRoot) {
  if (!isProfileObjectKey(objectKey)) {
    return false;
  }
  let filePath;
  try {
    filePath = resolveObjectPath(objectKey, objectRoot);
  } catch {
    return false;
  }
  const [folder, file] = await Promise.all([lstat(path.dirname(filePath)).catch(() => undefined), lstat(filePath).catch(() => undefined)]);
  if (!folder?.isDirectory() || !file?.isFile()) {
    return false;
  }
  try {
    await unlinkWithRetry(filePath);
    return true;
  } catch (error) {
    if (error?.code !== "ENOENT") {
      console.error(`Profile image cleanup could not delete ${filePath}:`, error);
    }
    return false;
  }
}

// Windows answers EPERM/EBUSY for a moment while another process holds a file without delete sharing (an antivirus
// scan of a PNG written a second ago), the same transient renameWithRetry rides out for the state file.
async function unlinkWithRetry(filePath) {
  const maxAttempts = 4;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      await unlink(filePath);
      return;
    } catch (error) {
      const code = error?.code;
      if (attempt >= maxAttempts - 1 || (code !== "EPERM" && code !== "EACCES" && code !== "EBUSY")) {
        throw error;
      }

      await delay(25 * (attempt + 1));
    }
  }
}

/**
 * Every object key a simulation state names for a profile image: the profile records themselves, and the face
 * references of image jobs that can still run (queued, planning, generating, or a status this server does not know).
 * A finished job's payload is history — regenerating it re-plans from the current profile — so it pins nothing;
 * counting it would keep every face ever used in a cut on disk for good.
 */
function listProfileObjectKeys(state) {
  const objectKeys = [];
  for (const profile of Array.isArray(state?.visualProfiles) ? state.visualProfiles : []) {
    for (const image of isPlainObject(profile?.profileImages) ? Object.values(profile.profileImages) : []) {
      if (isPlainObject(image)) {
        objectKeys.push(image.objectKey, image.referenceObjectKey);
      }
    }
  }
  const runs = Array.isArray(state?.progressRuns) ? state.progressRuns : [];
  const jobs = [state?.imageJobs, ...runs.map((run) => run?.imageJobs)].flatMap((value) => (Array.isArray(value) ? value : []));
  for (const job of jobs) {
    if (!isPlainObject(job) || ["completed", "failed", "canceled"].includes(job.status)) {
      continue;
    }
    const payload = isPlainObject(job.providerPayload) ? job.providerPayload : {};
    for (const reference of [payload.photorealReferences, payload.references].flatMap((value) => (Array.isArray(value) ? value : []))) {
      if (isPlainObject(reference)) {
        objectKeys.push(reference.objectKey);
      }
    }
    if (isPlainObject(payload.characterReference)) {
      objectKeys.push(payload.characterReference.objectKey);
    }
  }
  return objectKeys.filter((objectKey) => typeof objectKey === "string" && objectKey.length > 0);
}

function collectProfileObjectKeys(states) {
  const namedKeys = new Set();
  for (const state of states) {
    for (const objectKey of listProfileObjectKeys(state)) {
      const canonicalKey = canonicalObjectKey(objectKey);
      if (canonicalKey) {
        namedKeys.add(canonicalKey);
      }
    }
  }
  return namedKeys;
}

/**
 * The form in which keys are compared when deciding whether anything still names a file. Looser than the keys this
 * server writes on purpose: a stored key spelled with a backslash or a dot segment, or (on Windows) in other letter
 * case, still reaches the same file through resolveObjectPath and so still counts. Nothing is deleted by this form;
 * a match only ever keeps a file.
 */
function canonicalObjectKey(objectKey) {
  if (typeof objectKey !== "string") {
    return undefined;
  }
  const normalized = path.posix.normalize(objectKey.trim().replace(/\\/gu, "/"));
  if (!normalized || normalized === "." || normalized === ".." || normalized.startsWith("../")) {
    return undefined;
  }
  return process.platform === "win32" ? normalized.toLowerCase() : normalized;
}

// The only keys ever deleted: exactly <folder>/<file>, both segments passing the write-side allow-list, the file named
// like a profile pick. A traversal-looking key fails here before it reaches resolveObjectPath.
function isProfileObjectKey(objectKey) {
  const segments = typeof objectKey === "string" ? objectKey.split("/") : [];
  return segments.length === 2 && segments.every(isSafeObjectPathSegment) && PROFILE_OBJECT_FILE_PATTERN.test(segments[1]);
}

// A store without a simulations map names no references, and treating that as "nothing is referenced" would delete
// every candidate, so it is an error instead.
function listStoredSimulations(store) {
  if (!isPlainObject(store) || !isPlainObject(store.simulations)) {
    throw new Error("The state store has no simulations map, so nothing can be shown to be unreferenced.");
  }
  return Object.values(store.simulations);
}

// An unreadable time stamp counts as not newer: retiring files on a save whose age cannot be told is the unsafe guess.
function isSaveAtLeastAsRecent(next, previous) {
  const nextTime = Date.parse(typeof next?.simulation?.updatedAt === "string" ? next.simulation.updatedAt : "");
  const previousTime = Date.parse(typeof previous?.simulation?.updatedAt === "string" ? previous.simulation.updatedAt : "");
  return Number.isFinite(nextTime) && Number.isFinite(previousTime) && nextTime >= previousTime;
}

function pinProfileObjectKeys(state) {
  const pinnedKeys = [...collectProfileObjectKeys([state])];
  for (const canonicalKey of pinnedKeys) {
    inFlightProfileObjectKeys.set(canonicalKey, (inFlightProfileObjectKeys.get(canonicalKey) ?? 0) + 1);
  }
  return pinnedKeys;
}

function unpinProfileObjectKeys(pinnedKeys) {
  for (const canonicalKey of pinnedKeys) {
    const count = (inFlightProfileObjectKeys.get(canonicalKey) ?? 0) - 1;
    if (count > 0) {
      inFlightProfileObjectKeys.set(canonicalKey, count);
    } else {
      inFlightProfileObjectKeys.delete(canonicalKey);
    }
  }
}

function parseDataUrl(dataUrl) {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/u);
  if (!match) {
    return undefined;
  }

  return {
    mimeType: match[1],
    bytes: Buffer.from(match[2], "base64")
  };
}

function extensionForMimeType(mimeType) {
  if (mimeType === "image/jpeg") {
    return "jpg";
  }

  if (mimeType === "image/webp") {
    return "webp";
  }

  return "png";
}

function redactResponseSecrets(state) {
  return {
    ...state,
    llm: state.llm ? { ...state.llm, apiKey: "" } : state.llm,
    novelAi: state.novelAi ? { ...state.novelAi, apiKey: "" } : state.novelAi
  };
}

function validateAppState(state) {
  if (!state?.simulation?.id || !state?.simulation?.ownerId) {
    throw new Error("Invalid DynamicChat AppState payload.");
  }
}

function readRequestScope(request, state) {
  const ownerId = request.headers["x-dynamicchat-owner-id"]?.toString() || state?.simulation?.ownerId || "local_user";
  const projectId = request.headers["x-dynamicchat-project-id"]?.toString() || state?.simulation?.id || "local_project";
  return {
    ownerId,
    workspaceId: request.headers["x-dynamicchat-workspace-id"]?.toString() || state?.security?.scope?.workspaceId || "local_workspace",
    projectId,
    environment: request.headers["x-dynamicchat-environment"]?.toString() || state?.security?.scope?.environment || "local"
  };
}

function assertRequestScope(response, request, state) {
  const scope = readRequestScope(request, state);
  if (state?.simulation?.ownerId && state.simulation.ownerId !== scope.ownerId) {
    sendJson(response, 403, { error: "Owner scope mismatch." });
    return false;
  }

  if (state?.security?.scope?.projectId && state.security.scope.projectId !== scope.projectId) {
    sendJson(response, 403, { error: "Project scope mismatch." });
    return false;
  }

  return true;
}

function assertRequestOwnerScope(response, request, state) {
  const scope = readRequestScope(request, state);
  if (state?.simulation?.ownerId && state.simulation.ownerId !== scope.ownerId) {
    sendJson(response, 403, { error: "Owner scope mismatch." });
    return false;
  }

  return true;
}

function appendSecretAuditEvents(state, secretPatch) {
  if (!secretPatch) {
    return;
  }

  if (secretPatch.llm) {
    appendAuditEventFromScope(state, state.security?.scope, "api_secret_stored", "api_secret", "llm", {
      provider: secretPatch.llm.provider,
      model: secretPatch.llm.model,
      secretRef: secretPatch.llm.secretRef
    });
  }

  if (secretPatch.novelAi) {
    appendAuditEventFromScope(state, state.security?.scope, "api_secret_stored", "api_secret", "novelai", {
      provider: secretPatch.novelAi.provider,
      accountLabel: secretPatch.novelAi.accountLabel,
      secretRef: secretPatch.novelAi.secretRef
    });
  }
}

function appendAuditEvent(state, request, action, resourceType, resourceId, metadata = {}) {
  appendAuditEventFromScope(state, readRequestScope(request, state), action, resourceType, resourceId, metadata);
}

function appendAuditEventFromScope(state, scope, action, resourceType, resourceId, metadata = {}) {
  const resolvedScope = scope ?? state.security?.scope ?? {
    ownerId: state.simulation.ownerId,
    workspaceId: "local_workspace",
    projectId: state.simulation.id,
    environment: "local"
  };
  state.auditLog = [
    ...(state.auditLog ?? []),
    {
      id: `audit_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      simulationId: state.simulation.id,
      ownerId: state.simulation.ownerId,
      scope: resolvedScope,
      action,
      resourceType,
      resourceId,
      metadata,
      createdAt: new Date().toISOString()
    }
  ].slice(-300);
}

function findModule(store, moduleId) {
  for (const state of Object.values(store.simulations)) {
    const module = state.modules?.find((candidate) => candidate.id === moduleId);
    if (module) {
      return { state, module };
    }
  }

  return {};
}

function findImageJob(store, jobId) {
  for (const state of Object.values(store.simulations)) {
    const job = state.imageJobs?.find((candidate) => candidate.id === jobId);
    if (job) {
      return { state, job };
    }
  }

  return {};
}

function decodePersonalApiVault(ownerSecrets) {
  const personalVault = ownerSecrets.personalApiVault ?? {};
  const llmByProvider = Object.fromEntries(
    Object.entries(personalVault.llmByProvider ?? {}).map(([provider, record]) => [
      provider,
      decodeSecretRecord(record, "개인 LLM 키가 저장되어 있습니다.")
    ])
  );

  return {
    llmByProvider,
    novelAi: decodeSecretRecord(personalVault.novelAi, "개인 NovelAI 토큰이 저장되어 있습니다."),
    imageStoragePath: normalizeImageStoragePath(personalVault.imageStoragePath),
    updatedAt: personalVault.updatedAt
  };
}

function encodePersonalApiVault(vault, existingVault = {}) {
  const now = new Date().toISOString();
  const providerNames = new Set([
    ...Object.keys(existingVault?.llmByProvider ?? {}),
    ...Object.keys(vault?.llmByProvider ?? {})
  ]);
  const llmByProvider = Object.fromEntries(
    [...providerNames].map((provider) => [
      provider,
      encodeSecretRecord(vault?.llmByProvider?.[provider], now, existingVault?.llmByProvider?.[provider])
    ])
  );

  return {
    llmByProvider,
    novelAi: encodeSecretRecord(vault?.novelAi, now, existingVault?.novelAi),
    imageStoragePath: normalizeImageStoragePath(vault?.imageStoragePath) || normalizeImageStoragePath(existingVault?.imageStoragePath),
    updatedAt: vault?.updatedAt ?? now
  };
}

async function getRequestObjectRoot(request) {
  return (await getRequestObjectRoots(request))[0] ?? defaultObjectDir;
}

async function getRequestObjectRoots(request) {
  const scope = readRequestScope(request);
  const secrets = await readSecretStore();
  return getOwnerObjectRoots(secrets.owners?.[scope.ownerId], secrets);
}

function getOwnerObjectRoot(ownerSecrets = {}) {
  return getOwnerObjectRoots(ownerSecrets)[0] ?? defaultObjectDir;
}

function getOwnerObjectRoots(ownerSecrets = {}, secrets = {}) {
  const knownOwnerRoots = Object.values(secrets.owners ?? {})
    .map((candidate) => normalizeImageStoragePath(candidate?.personalApiVault?.imageStoragePath))
    .filter(Boolean);
  return uniqueObjectRoots(normalizeImageStoragePath(ownerSecrets.personalApiVault?.imageStoragePath), defaultObjectDir, ...knownOwnerRoots);
}

function normalizeImageStoragePath(value) {
  return typeof value === "string" ? value.trim() : "";
}

function decodeSecretRecord(record, savedMessage) {
  const apiKey = record?.encodedSecret ? Buffer.from(record.encodedSecret, "base64").toString("utf8") : "";
  const registrationStatus = record?.registrationStatus ?? (apiKey ? "registered" : "idle");

  return {
    apiKey,
    registrationStatus,
    verifiedAt: record?.verifiedAt,
    verificationMessage: record?.verificationMessage ?? (apiKey ? savedMessage : ""),
    subscriptionTier: record?.subscriptionTier
  };
}

function encodeSecretRecord(record, updatedAt, existingRecord = {}) {
  const apiKey = typeof record?.apiKey === "string" ? record.apiKey.trim() : "";
  const encodedSecret = apiKey ? Buffer.from(apiKey, "utf8").toString("base64") : existingRecord?.encodedSecret ?? "";

  return {
    encodedSecret,
    registrationStatus: record?.registrationStatus ?? existingRecord?.registrationStatus ?? (encodedSecret ? "registered" : "idle"),
    verifiedAt: record?.verifiedAt ?? existingRecord?.verifiedAt,
    verificationMessage: record?.verificationMessage ?? existingRecord?.verificationMessage ?? "",
    subscriptionTier: record?.subscriptionTier ?? existingRecord?.subscriptionTier,
    updatedAt
  };
}

function readBearerToken(request) {
  return (request.headers.authorization ?? "").replace(/^Bearer\s+/iu, "").trim();
}

async function readJsonBody(request) {
  // Node's http server caps neither the body nor the number of chunks, so without this a single POST
  // buffers until the process dies (measured: a 64MB body was accepted in 318ms). The default is
  // deliberately generous — a full-resolution NovelAI PNG arrives here as a base64 data URL inside the
  // asset payload, which is the large legitimate case — and DYNAMICCHAT_MAX_BODY_BYTES raises it for
  // anyone whose images outgrow it. Declared content-length is checked first so an honest oversized
  // client is refused before we allocate anything, and that is the only path on which the caller gets
  // a readable 413 — see the streaming check below.
  const declared = Number(request.headers["content-length"] ?? 0);
  if (Number.isFinite(declared) && declared > maxJsonBodyBytes) {
    throw new PayloadTooLargeError();
  }

  const chunks = [];
  let total = 0;
  for await (const chunk of request) {
    total += chunk.length;
    if (total > maxJsonBodyBytes) {
      // A chunked request declares no length, so the only way to stop the flood is to drop the socket —
      // and that takes the response with it: the 413 the dispatcher then formats is written to a dead
      // connection and silently discarded. The caller sees a connection reset, not a status. Throwing
      // without destroying would not change that; leaving the `for await` runs the iterator's return(),
      // which destroys the stream anyway. The throw still exists so the operator log names the cause.
      request.destroy();
      throw new PayloadTooLargeError();
    }
    chunks.push(chunk);
  }

  const raw = Buffer.concat(chunks).toString("utf8").trim();
  return raw ? JSON.parse(raw) : {};
}

async function ensureStore() {
  await mkdir(dataDir, { recursive: true });
  await mkdir(defaultObjectDir, { recursive: true });
  await mkdir(backupDir, { recursive: true });
  if (!existsSync(statePath)) {
    await writeStateStore({ simulations: {} });
  }
  if (!existsSync(secretPath)) {
    await writeSecretStore({ owners: {} });
  }
}

async function readStateStore() {
  const raw = await readFile(statePath, "utf8");
  try {
    return JSON.parse(raw);
  } catch {
    const recovered = parseLeadingJsonObject(raw);
    if (!recovered) {
      throw new Error("DynamicChat state store is not valid JSON.");
    }
    await writeStateStore(recovered);
    return recovered;
  }
}

async function writeStateStore(store) {
  stateStoreWriteQueue = stateStoreWriteQueue
    .catch(() => undefined)
    .then(() => writeStateStoreNow(store));
  return stateStoreWriteQueue;
}

async function writeStateStoreNow(store) {
  await mkdir(path.dirname(statePath), { recursive: true });
  const tmpPath = `${statePath}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(tmpPath, `${JSON.stringify(store, null, 2)}\n`);
  await renameWithRetry(tmpPath, statePath);
}

async function renameWithRetry(sourcePath, targetPath) {
  const maxAttempts = 8;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      await rename(sourcePath, targetPath);
      return;
    } catch (error) {
      const code = error?.code;
      if (attempt >= maxAttempts - 1 || (code !== "EPERM" && code !== "EACCES" && code !== "EBUSY")) {
        throw error;
      }

      await delay(25 * (attempt + 1));
    }
  }
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function parseLeadingJsonObject(raw) {
  const endIndex = findLeadingJsonObjectEnd(raw);
  if (endIndex < 0) {
    return undefined;
  }

  try {
    return JSON.parse(raw.slice(0, endIndex));
  } catch {
    return undefined;
  }
}

function findLeadingJsonObjectEnd(raw) {
  const startIndex = raw.search(/\S/u);
  if (startIndex < 0 || raw[startIndex] !== "{") {
    return -1;
  }

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = startIndex; index < raw.length; index += 1) {
    const character = raw[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === "\"") {
        inString = false;
      }
      continue;
    }

    if (character === "\"") {
      inString = true;
    } else if (character === "{") {
      depth += 1;
    } else if (character === "}") {
      depth -= 1;
      if (depth === 0) {
        return index + 1;
      }
    }
  }

  return -1;
}

async function readSecretStore() {
  const raw = await readFile(secretPath, "utf8");
  try {
    return JSON.parse(raw);
  } catch {
    const recovered = parseLeadingJsonObject(raw);
    if (!recovered) {
      throw new Error(`DynamicChat secret store is not valid JSON and could not be recovered: ${secretPath}`);
    }
    await writeSecretStore(recovered);
    return recovered;
  }
}

async function writeSecretStore(store) {
  secretStoreWriteQueue = secretStoreWriteQueue
    .catch(() => undefined)
    .then(() => writeSecretStoreNow(store));
  return secretStoreWriteQueue;
}

async function writeSecretStoreNow(store) {
  await mkdir(path.dirname(secretPath), { recursive: true });
  const tmpPath = `${secretPath}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(tmpPath, `${JSON.stringify(store, null, 2)}\n`);
  await renameWithRetry(tmpPath, secretPath);
}

function send(response, statusCode) {
  response.writeHead(statusCode, corsHeaders);
  response.end();
}

function sendJson(response, statusCode, body) {
  response.writeHead(statusCode, {
    ...corsHeaders,
    "content-type": "application/json; charset=utf-8"
  });
  response.end(JSON.stringify(body));
}
