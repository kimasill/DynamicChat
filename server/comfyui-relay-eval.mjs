/**
 * ComfyUI relay evaluation (eval:comfyui-relay).
 *
 * The photoreal backend reaches the GPU only through the /photoreal/comfyui/* relay in dynamicchat-server.mjs,
 * and that relay is also a new door on an unauthenticated local server. Both halves are checked against the real
 * server process — spawned with a fresh data directory, a free port and a raised rate limit — talking to the mock
 * ComfyUI in mock-comfyui.mjs:
 *
 *   - the door: the origin guard on all four routes (hostile origins, ComfyUI's own origins in every loopback
 *     spelling and either scheme, the deny-list env in origin and host:port form), the Host check that stops a
 *     DNS-rebound page, base-URL validation, the node allow-list (LoRA loaders off unless the operator names them),
 *     images only from the request's own references (no file from ComfyUI's input folder), no guidance scale below
 *     1 (below 1 the negative becomes the picture), reference validation, the body cap
 *   - the round trip: reference upload + placeholder substitution (dataUrl and objectKey), history polling, image
 *     download (PreviewImage results from ComfyUI's temp folder, and which images count as the result), and the
 *     translation of each ComfyUI failure (node_errors, execution_error, a timeout that takes the prompt back, a
 *     browser that disconnects, an unreachable server, a redirect)
 *   - the client: generatePhotorealImages / probeComfyUi / freeComfyUiMemory from comfyUiClient.ts, end to end
 *     through the spawned server
 *   - the dev web server: vite.config.ts must not hand the data directory (keys, transcripts) to the ComfyUI
 *     origin, which the API's guard refuses — the same files are inside the folder Vite serves
 *
 * Four server instances run side by side because several checks are about environment variables, which the
 * server reads once at startup. No GPU, ComfyUI or network access is needed.
 *
 * Usage: node server/comfyui-relay-eval.mjs
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { createServer } from "vite";
import { MOCK_COMFYUI_GPU, MOCK_COMFYUI_VERSION, MOCK_COMFYUI_VRAM, createMockPng, startMockComfyUi } from "./mock-comfyui.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const serverEntry = path.join(workspaceRoot, "server", "dynamicchat-server.mjs");

const GENERATE = "/photoreal/comfyui/generate";
const STATUS = "/photoreal/comfyui/status";
const FREE = "/photoreal/comfyui/free";
const INTERRUPT = "/photoreal/comfyui/interrupt";
const HOSTILE_ORIGIN = "https://evil.example";
const APP_ORIGIN = "http://127.0.0.1:5173";
const COMFYUI_DEFAULT_ORIGINS = ["http://127.0.0.1:8188", "http://localhost:8188", "http://[::1]:8188", "https://127.0.0.1:8188"];
// The same server written the ways a browser can write it: each is its own origin to the browser.
const COMFYUI_ALIAS_ORIGINS = ["https://localhost:8188", "https://[::1]:8188", "http://127.0.0.2:8188", "http://[::ffff:127.0.0.1]:8188"];
// A loopback origin nothing denies by default; the "configured" instance lists it in DYNAMICCHAT_DENIED_ORIGINS.
const UNLISTED_LOCAL_ORIGIN = "http://127.0.0.1:7777";
// The rest of the configured instance's deny list: a host:port with no scheme (the DYNAMICCHAT_COMFYUI_HOSTS
// shape), an IPv6 one, and an entry that is neither, which must be reported rather than dropped in silence.
const DENIED_ORIGINS_ENV = `${UNLISTED_LOCAL_ORIGIN},localhost:7778,[::1]:7779,definitely not an origin`;
// The "aliases" instance trusts these Host names (a reverse proxy in front of the API) and nothing else.
const ALLOWED_HOSTS_ENV = "dynamicchat.test,proxy.dynamicchat.test:9443,not a host!";
const REBOUND_HOST = "rebind.attacker.example";
const LAN_COMFYUI = "http://192.168.0.10:8188";
const SAMPLE_PROMPT_ID = "3f2a9c1e-5b7d-4e8f-9a0b-1c2d3e4f5a6b";
// Read by the server at startup. Inherited values from the developer's shell would change what is being tested.
const SERVER_ENV_KEYS = [
  "DYNAMICCHAT_COMFYUI_URL",
  "DYNAMICCHAT_COMFYUI_HOSTS",
  "DYNAMICCHAT_COMFYUI_EXTRA_NODES",
  "DYNAMICCHAT_DENIED_ORIGINS",
  "DYNAMICCHAT_ALLOWED_HOSTS",
  "DYNAMICCHAT_CORS_ORIGIN",
  "DYNAMICCHAT_MAX_BODY_BYTES",
  "DYNAMICCHAT_RATE_LIMIT_WINDOW_MS"
];

const checks = [];
const skipped = [];
const cleanups = [];

try {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "dynamicchat-comfyui-relay-"));
  cleanups.push(() => rm(tempRoot, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }));
  const outputPng = createMockPng();
  const mock = await startMockComfyUi({ outputPng });
  cleanups.push(() => mock.close());

  const main = await startApiServer("main", tempRoot, {});
  cleanups.push(() => main.stop());
  const configured = await startApiServer("configured", tempRoot, {
    DYNAMICCHAT_COMFYUI_URL: mock.baseUrl,
    DYNAMICCHAT_COMFYUI_HOSTS: "192.168.0.10",
    DYNAMICCHAT_COMFYUI_EXTRA_NODES: "MyCustomNode,LoraLoader",
    DYNAMICCHAT_DENIED_ORIGINS: DENIED_ORIGINS_ENV
  });
  cleanups.push(() => configured.stop());
  const openNodes = await startApiServer("open-nodes", tempRoot, { DYNAMICCHAT_COMFYUI_EXTRA_NODES: "*" });
  cleanups.push(() => openNodes.stop());
  // The review's case: ComfyUI configured as localhost, while ComfyUI itself prints and opens 127.0.0.1.
  const aliases = await startApiServer("aliases", tempRoot, {
    DYNAMICCHAT_COMFYUI_URL: `http://localhost:${mock.port}`,
    DYNAMICCHAT_ALLOWED_HOSTS: ALLOWED_HOSTS_ENV
  });
  cleanups.push(() => aliases.stop());

  // A stored profile image, written where persistImageAssets would have put it.
  const referencePng = createMockPng({ width: 3, height: 4, rgb: [0x2a, 0x6f, 0x97] });
  const simulationId = "sim_relay_eval";
  const objectKey = `${simulationId}/profile_char_1.png`;
  await mkdir(path.join(main.dataDir, "objects", simulationId), { recursive: true });
  await writeFile(path.join(main.dataDir, "objects", objectKey), referencePng);
  await writeFile(path.join(main.dataDir, "objects", simulationId, "notes.json"), "{}");

  const context = { mock, main, configured, openNodes, aliases, outputPng, referencePng, objectKey, simulationId, tempRoot };
  await runGroup("existing routes", () => evaluateExistingRoutes(context));
  await runGroup("origin guard", () => evaluateOriginGuard(context));
  await runGroup("host header", () => evaluateHostHeader(context));
  await runGroup("base URL", () => evaluateBaseUrlValidation(context));
  await runGroup("node allow-list", () => evaluateNodeAllowList(context));
  await runGroup("lora loaders", () => evaluateLoraLoaders(context));
  await runGroup("image inputs", () => evaluateImageInputs(context));
  await runGroup("guidance", () => evaluateGuidanceFloor(context));
  await runGroup("references", () => evaluateReferenceValidation(context));
  await runGroup("generate", () => evaluateSuccessfulGeneration(context));
  await runGroup("ComfyUI failures", () => evaluateComfyUiFailures(context));
  await runGroup("cancellation", () => evaluateCancellation(context));
  await runGroup("unreachable", () => evaluateUnreachableUpstream(context));
  await runGroup("status", () => evaluateStatusEndpoint(context));
  await runGroup("free/interrupt", () => evaluateFreeAndInterrupt(context));
  await runGroup("body limits", () => evaluateBodyLimits(context));
  await runGroup("environment", () => evaluateEnvironmentConfiguration(context));
  await runGroup("client end to end", () => evaluateClientEndToEnd(context));
  await runGroup("dev web server", () => evaluateDevWebServer(context));
} catch (error) {
  assertCheck("setup", false, `eval setup failed: ${error?.stack ?? error}`);
} finally {
  for (const cleanup of cleanups.reverse()) {
    await Promise.resolve()
      .then(cleanup)
      .catch((error) => console.error(`cleanup failed: ${error?.message ?? error}`));
  }
}

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}`);
}
for (const note of skipped) {
  console.log(`SKIP ${note}`);
}

const failures = checks.filter((check) => check.level === "fail");
console.log("");
console.log(`ComfyUI relay checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

async function evaluateExistingRoutes({ main, referencePng, objectKey }) {
  const scope = "existing routes";
  const health = await api(main, "/health");
  assertCheck(scope, health.status === 200 && isDeepStrictEqual(health.json, { ok: true, service: "dynamicchat-api" }), "GET /health still answers 200 with its fixed body", describe(health));

  const preflight = await fetch(`${main.baseUrl}${GENERATE}`, {
    method: "OPTIONS",
    headers: { origin: APP_ORIGIN, "access-control-request-method": "POST", "access-control-request-headers": "content-type" }
  });
  assertCheck(
    scope,
    preflight.status === 204 && /POST/u.test(preflight.headers.get("access-control-allow-methods") ?? ""),
    "the relay's CORS preflight answers 204 with POST allowed",
    `status=${preflight.status}`
  );

  const vault = await api(main, "/personal-api-vault");
  const vaultFromApp = await api(main, "/personal-api-vault", { origin: APP_ORIGIN });
  assertCheck(scope, vault.status === 200 && vaultFromApp.status === 200 && isPlainObject(vault.json?.llmByProvider), "GET /personal-api-vault still answers the app and header-less callers", `${describe(vault)} / ${describe(vaultFromApp)}`);

  const vaultHostile = await api(main, "/personal-api-vault", { origin: HOSTILE_ORIGIN });
  const cliHostile = await api(main, "/llm/cli-agent", { method: "POST", body: { agent: "claude", prompt: "hello" }, origin: HOSTILE_ORIGIN });
  assertCheck(
    scope,
    vaultHostile.status === 403 && vaultHostile.json?.error === "forbidden_origin" && cliHostile.status === 403,
    "the existing guarded routes still refuse a hostile origin",
    `${describe(vaultHostile)} / ${describe(cliHostile)}`
  );

  const objectResponse = await fetch(`${main.baseUrl}/objects/${objectKey}`);
  const objectBytes = Buffer.from(await objectResponse.arrayBuffer());
  assertCheck(scope, objectResponse.status === 200 && objectBytes.equals(referencePng), "GET /objects/<key> still serves stored images", `status=${objectResponse.status}`);

  const missing = await api(main, "/photoreal/comfyui/unknown");
  assertCheck(scope, missing.status === 404, "unregistered paths still answer 404", describe(missing));
}

async function evaluateOriginGuard({ mock, main }) {
  const scope = "origin guard";
  const routes = createRelayRoutes(mock.baseUrl);
  mock.reset();
  for (const route of routes) {
    const result = await api(main, route.path, { method: route.method, body: route.body, origin: HOSTILE_ORIGIN });
    assertCheck(scope, result.status === 403 && result.json?.error === "forbidden_origin", `a hostile Origin is refused on ${route.label}`, describe(result));
  }
  for (const route of routes) {
    const result = await api(main, route.path, { method: route.method, body: route.body, origin: "http://127.0.0.1:8188" });
    assertCheck(
      scope,
      result.status === 403 && result.json?.error === "forbidden_origin" && /ComfyUI/u.test(result.json?.message ?? ""),
      `ComfyUI's web UI origin (http://127.0.0.1:8188) is refused on ${route.label}`,
      describe(result)
    );
  }
  assertCheck(scope, mock.received.length === 0, "a refused request never reaches ComfyUI", `ComfyUI saw ${mock.received.length} request(s)`);

  for (const origin of COMFYUI_DEFAULT_ORIGINS) {
    const vault = await api(main, "/personal-api-vault", { origin });
    assertCheck(scope, vault.status === 403 && /ComfyUI/u.test(vault.json?.message ?? ""), `the ComfyUI deny rule also covers /personal-api-vault: ${origin} → 403`, describe(vault));
  }
  // The ComfyUI message, not the generic one: these are denied as ComfyUI, by port, not merely as non-local.
  for (const origin of COMFYUI_ALIAS_ORIGINS) {
    const vault = await api(main, "/personal-api-vault", { origin });
    assertCheck(scope, vault.status === 403 && /ComfyUI/u.test(vault.json?.message ?? ""), `every loopback spelling and scheme on ComfyUI's port is ComfyUI: ${origin} → 403`, describe(vault));
  }

  const statusPath = `${STATUS}?baseUrl=${encodeURIComponent(mock.baseUrl)}`;
  const fromDrivenComfyUi = await api(main, statusPath, { origin: mock.baseUrl });
  assertCheck(
    scope,
    fromDrivenComfyUi.status === 403 && fromDrivenComfyUi.json?.error === "forbidden_origin",
    "a page served by the very ComfyUI a request would drive is refused, whatever its port",
    describe(fromDrivenComfyUi)
  );
  for (const origin of [`http://localhost:${mock.port}`, `https://127.0.0.1:${mock.port}`, `http://[::1]:${mock.port}`]) {
    const result = await api(main, statusPath, { origin });
    assertCheck(
      scope,
      result.status === 403 && result.json?.error === "forbidden_origin",
      `... and so is the same ComfyUI under another spelling or scheme (${origin} driving ${mock.baseUrl})`,
      describe(result)
    );
  }
  const aliasGenerate = await generate(
    main,
    { baseUrl: `http://localhost:${mock.port}`, workflow: createTextToImageWorkflow(), timeoutMs: 20_000 },
    { origin: mock.baseUrl }
  );
  assertCheck(
    scope,
    aliasGenerate.status === 403 && aliasGenerate.json?.error === "forbidden_origin" && mock.requests("POST", "/prompt").length === 0,
    `generate refuses ${mock.baseUrl} driving its own ComfyUI as localhost:${mock.port}, before anything is queued`,
    describe(aliasGenerate)
  );

  const noOrigin = await api(main, statusPath);
  assertCheck(scope, noOrigin.status === 200 && noOrigin.json?.ok === true, "a caller with no Origin (eval harness, curl) is allowed", describe(noOrigin));
  for (const origin of [APP_ORIGIN, "http://localhost:5173"]) {
    const result = await api(main, statusPath, { origin });
    assertCheck(scope, result.status === 200 && result.json?.ok === true, `the app's own origin (${origin}) is allowed`, describe(result));
  }
}

async function evaluateHostHeader({ mock, main, aliases }) {
  const scope = "host header";
  mock.reset();
  // What a page on http://rebind.attacker.example:<port> sends once its name resolves to 127.0.0.1: a same-origin
  // GET carries no Origin header, only the Host it was loaded from.
  const foreignHost = `${REBOUND_HOST}:${main.port}`;
  const reboundReads = [
    ["/personal-api-vault", "the decoded keys"],
    ["/simulations", "the transcripts"],
    [`${STATUS}?baseUrl=${encodeURIComponent(mock.baseUrl)}&nodes=UNETLoader`, "the ComfyUI inventory"],
    [`${STATUS}?baseUrl=${encodeURIComponent("http://127.0.0.1:1")}`, "a loopback port probe"]
  ];
  for (const [pathname, what] of reboundReads) {
    const result = await rawApi(main, pathname, { host: foreignHost });
    assertCheck(
      scope,
      result.status === 403 && result.json?.error === "forbidden_host",
      `a rebound page's same-origin GET (Host ${REBOUND_HOST}, no Origin) cannot read ${what}`,
      describe(result)
    );
  }
  const reboundPost = await rawApi(main, FREE, { method: "POST", host: foreignHost, body: { baseUrl: mock.baseUrl } });
  const portless = await rawApi(main, "/personal-api-vault", { host: REBOUND_HOST });
  assertCheck(
    scope,
    reboundPost.status === 403 && reboundPost.json?.error === "forbidden_host" && portless.status === 403,
    "a foreign Host is refused on POST too, and without a port",
    `${describe(reboundPost)} / ${describe(portless)}`
  );
  const withAppOrigin = await rawApi(main, "/personal-api-vault", { host: foreignHost, origin: APP_ORIGIN });
  assertCheck(
    scope,
    withAppOrigin.status === 403 && withAppOrigin.json?.error === "forbidden_host" && /DYNAMICCHAT_ALLOWED_HOSTS/u.test(withAppOrigin.json?.message ?? ""),
    "Host is checked before Origin, and the refusal names DYNAMICCHAT_ALLOWED_HOSTS",
    describe(withAppOrigin)
  );
  assertCheck(scope, mock.received.length === 0, "a refused Host never reaches ComfyUI (no inventory, no port probe)", `ComfyUI saw ${mock.received.length} request(s)`);

  for (const host of [`127.0.0.1:${main.port}`, `localhost:${main.port}`, `[::1]:${main.port}`, "127.0.0.1:5173", "localhost"]) {
    const result = await rawApi(main, "/personal-api-vault", { host });
    assertCheck(scope, result.status === 200, `a loopback Host is accepted, on any port (${host})`, describe(result));
  }
  const health = await rawApi(main, "/health", { host: foreignHost });
  assertCheck(scope, health.status === 200, "GET /health stays unguarded (the Host check belongs to the guard, not to every route)", describe(health));

  const allowedCases = [
    [`dynamicchat.test:${aliases.port}`, 200, "a name listed without a port is accepted on any port"],
    ["dynamicchat.test", 200, "... and without a port in the Host header"],
    ["proxy.dynamicchat.test:9443", 200, "a name listed as name:port is accepted on that port"],
    ["proxy.dynamicchat.test:9444", 403, "... and refused on any other"],
    [`${REBOUND_HOST}:${aliases.port}`, 403, "an unlisted name is still refused where others are listed"]
  ];
  for (const [host, expected, label] of allowedCases) {
    const result = await rawApi(aliases, "/personal-api-vault", { host });
    assertCheck(scope, result.status === expected, `DYNAMICCHAT_ALLOWED_HOSTS: ${label} (${host} → ${expected})`, describe(result));
  }
  const listedNameOnMain = await rawApi(main, "/personal-api-vault", { host: "dynamicchat.test" });
  assertCheck(scope, listedNameOnMain.status === 403, "a name is accepted only where it is listed", describe(listedNameOnMain));
  assertCheck(
    scope,
    aliases.output().includes('DYNAMICCHAT_ALLOWED_HOSTS entry "not a host!"'),
    "an allow-list entry that is not a host is reported at startup, not dropped in silence",
    aliases.output().slice(0, 400)
  );
}

async function evaluateBaseUrlValidation({ mock, main }) {
  const scope = "base URL";
  mock.reset();
  const cases = [
    [LAN_COMFYUI, /DYNAMICCHAT_COMFYUI_HOSTS/u, "a LAN host that is not allow-listed"],
    ["http://user:secret@127.0.0.1:8188", /사용자 정보/u, "userinfo in the base URL"],
    ["ftp://127.0.0.1:8188", /http/u, "an ftp:// base URL"],
    ["http://127.0.0.1:8188/comfy", /경로/u, "a base URL with a path"],
    ["not a url", /URL/u, "an unparseable base URL"],
    [`http://127.0.0.1:${main.port}`, /자신/u, "the API server's own address"],
    [42, /문자열/u, "a non-string base URL"]
  ];
  for (const [baseUrl, pattern, label] of cases) {
    const result = await api(main, GENERATE, { method: "POST", body: { baseUrl, workflow: createTextToImageWorkflow() } });
    assertCheck(scope, result.status === 400 && pattern.test(result.json?.error ?? ""), `generate refuses ${label} with 400`, describe(result));
  }
  const status = await api(main, `${STATUS}?baseUrl=${encodeURIComponent(LAN_COMFYUI)}`);
  assertCheck(scope, status.status === 400 && /DYNAMICCHAT_COMFYUI_HOSTS/u.test(status.json?.error ?? ""), "status refuses a host that is not allow-listed instead of probing it", describe(status));
  const free = await api(main, FREE, { method: "POST", body: { baseUrl: "http://user:secret@127.0.0.1:8188" } });
  assertCheck(scope, free.status === 400, "free validates the base URL the same way", describe(free));
}

async function evaluateNodeAllowList({ mock, main }) {
  const scope = "node allow-list";
  mock.reset();
  const python = createTextToImageWorkflow();
  python["99"] = { class_type: "ExecutePython", inputs: { code: "import os; os.system('calc')" } };
  const refused = await generate(main, { baseUrl: mock.baseUrl, workflow: python });
  assertCheck(
    scope,
    refused.status === 400 && refused.json?.error?.includes("ExecutePython") && isDeepStrictEqual(refused.json?.rejectedNodes, ["ExecutePython"]),
    "a disallowed class_type (ExecutePython) is refused with 400 naming it",
    describe(refused)
  );

  const custom = createTextToImageWorkflow();
  custom["40"] = { class_type: "MyCustomNode", inputs: {} };
  const customRefused = await generate(main, { baseUrl: mock.baseUrl, workflow: custom });
  assertCheck(scope, customRefused.status === 400 && customRefused.json?.rejectedNodes?.includes("MyCustomNode"), "an unlisted custom node is refused unless the operator names it", describe(customRefused));

  const shapes = [
    [{ nodes: [], links: [] }, /API/u, "a UI-format export (nodes + links arrays)"],
    [{}, /1-300/u, "an empty workflow"],
    [Object.fromEntries(Array.from({ length: 301 }, (_, index) => [String(index + 1), { class_type: "PrimitiveInt", inputs: { value: index } }])), /1-300/u, "a workflow with 301 nodes"],
    [{ "1": { class_type: "KSampler" } }, /class_type, inputs/u, "a node without inputs"],
    [{ "1": { class_type: "PrimitiveString", inputs: { value: nestArray(40) } } }, /중첩/u, "inputs nested 40 deep"],
    [{ "1": { class_type: "PrimitiveString", inputs: { value: "x" }, _meta: { title: nestArray(40) } } }, /중첩/u, "_meta nested 40 deep"],
    ["{\"1\":{}}", /객체/u, "a workflow sent as a string"]
  ];
  for (const [workflow, pattern, label] of shapes) {
    const result = await generate(main, { baseUrl: mock.baseUrl, workflow });
    assertCheck(scope, result.status === 400 && pattern.test(result.json?.error ?? ""), `${label} is refused with 400`, describe(result));
  }
  assertCheck(scope, mock.received.length === 0, "nothing that failed validation reached ComfyUI", `ComfyUI saw ${mock.received.length} request(s)`);
}

/**
 * A LoRA changes what the model draws with no word of it in the prompt, so the photoreal safety floor (which reads
 * text) cannot see it, and no built-in preset uses one: LoRA loaders are off by default and an operator names them in
 * DYNAMICCHAT_COMFYUI_EXTRA_NODES like any other node (the "configured" instance names LoraLoader).
 */
async function evaluateLoraLoaders({ mock, main, configured, openNodes }) {
  const scope = "lora loaders";
  const withLora = (classType) => {
    const workflow = createTextToImageWorkflow();
    workflow["11"] =
      classType === "LoraLoader"
        ? { class_type: classType, inputs: { model: ["2", 0], clip: ["3", 0], lora_name: "any_style.safetensors", strength_model: 1, strength_clip: 1 } }
        : { class_type: classType, inputs: { model: ["2", 0], lora_name: "any_style.safetensors", strength_model: 1 } };
    workflow["8"].inputs.model = ["11", 0];
    return workflow;
  };
  mock.reset();
  for (const classType of ["LoraLoader", "LoraLoaderModelOnly"]) {
    const refused = await generate(main, { baseUrl: mock.baseUrl, workflow: withLora(classType), timeoutMs: 20_000 });
    assertCheck(
      scope,
      refused.status === 400 &&
        isDeepStrictEqual(refused.json?.rejectedNodes, [classType]) &&
        /LoRA/u.test(refused.json?.error ?? "") &&
        /DYNAMICCHAT_COMFYUI_EXTRA_NODES/u.test(refused.json?.error ?? ""),
      `${classType} is refused by default, and the 400 says why (a LoRA is invisible to the text floor) and how to allow it`,
      describe(refused)
    );
  }
  assertCheck(scope, mock.requests("POST", "/prompt").length === 0, "a refused LoRA graph is never queued");

  const named = await generate(configured, { baseUrl: mock.baseUrl, workflow: withLora("LoraLoader"), timeoutMs: 20_000 });
  assertCheck(
    scope,
    named.status === 200 && mock.requests("POST", "/prompt")[0]?.body?.prompt?.["11"]?.class_type === "LoraLoader",
    "DYNAMICCHAT_COMFYUI_EXTRA_NODES=...,LoraLoader admits LoraLoader",
    describe(named)
  );
  const unnamed = await generate(configured, { baseUrl: mock.baseUrl, workflow: withLora("LoraLoaderModelOnly"), timeoutMs: 20_000 });
  assertCheck(scope, unnamed.status === 400 && unnamed.json?.rejectedNodes?.includes("LoraLoaderModelOnly"), "naming LoraLoader does not admit LoraLoaderModelOnly", describe(unnamed));
  mock.reset();
  const open = await generate(openNodes, { baseUrl: mock.baseUrl, workflow: withLora("LoraLoaderModelOnly"), timeoutMs: 20_000 });
  assertCheck(scope, open.status === 200, "with the node check off (*) a LoRA loader passes like any node", describe(open));
}

/**
 * Every image a relayed graph loads is a reference this request uploaded. A LoadImage naming a file would return
 * whatever sits in ComfyUI's input folder as the "render". Held whatever the node allow-list says. (It does not prove an
 * image was generated — a caller's own dataUrl reference passes through a %ref: loader — which is why the app sends
 * only object keys of picked profile images and never bytes; see photoreal-client-eval "references.picks-only".)
 */
async function evaluateImageInputs({ mock, main, openNodes, referencePng }) {
  const scope = "image inputs";
  mock.reset();
  const dataUrl = toDataUrl(referencePng);
  const passThrough = {
    "1": { class_type: "LoadImage", inputs: { image: "any_photo_on_disk.png" } },
    "2": { class_type: "PreviewImage", inputs: { images: ["1", 0] } }
  };
  const refused = await generate(main, { baseUrl: mock.baseUrl, workflow: passThrough, timeoutMs: 20_000 });
  assertCheck(
    scope,
    refused.status === 400 && /참조 이미지/u.test(refused.json?.error ?? "") && isDeepStrictEqual(refused.json?.rejectedImageNodes, ["1"]),
    "a LoadImage -> PreviewImage pass-through naming a file in ComfyUI's input folder is refused with 400",
    describe(refused)
  );

  const beside = createReferenceWorkflow(["char_1"]);
  beside["30"] = { class_type: "LoadImage", inputs: { image: "any_photo_on_disk.png" } };
  beside["31"] = { class_type: "PreviewImage", inputs: { images: ["30", 0] } };
  const besideResult = await generate(main, { baseUrl: mock.baseUrl, workflow: beside, references: [{ name: "char_1", dataUrl }], timeoutMs: 20_000 });
  assertCheck(
    scope,
    besideResult.status === 400 && isDeepStrictEqual(besideResult.json?.rejectedImageNodes, ["30"]),
    "a literal LoadImage is refused even beside a properly uploaded reference",
    describe(besideResult)
  );

  const linked = createTextToImageWorkflow();
  linked["30"] = { class_type: "PrimitiveString", inputs: { value: "any_photo_on_disk.png" } };
  linked["31"] = { class_type: "LoadImage", inputs: { image: ["30", 0] } };
  const linkedResult = await generate(main, { baseUrl: mock.baseUrl, workflow: linked, timeoutMs: 20_000 });
  assertCheck(scope, linkedResult.status === 400 && isDeepStrictEqual(linkedResult.json?.rejectedImageNodes, ["31"]), "a LoadImage whose file name arrives through a link is refused", describe(linkedResult));

  // With the node check off any loader may appear; where a face may come from does not change.
  for (const [node, label] of [
    [{ class_type: "LoadImageMask", inputs: { image: "mask.png", channel: "alpha" } }, "LoadImageMask with a file name"],
    [{ class_type: "SomeImageLoader", inputs: { image: "photo.png" } }, "a custom node with a string image input"]
  ]) {
    const workflow = createTextToImageWorkflow();
    workflow["30"] = node;
    const result = await generate(openNodes, { baseUrl: mock.baseUrl, workflow, timeoutMs: 20_000 });
    assertCheck(scope, result.status === 400 && isDeepStrictEqual(result.json?.rejectedImageNodes, ["30"]), `${label} is refused even with DYNAMICCHAT_COMFYUI_EXTRA_NODES=*`, describe(result));
  }
  assertCheck(
    scope,
    mock.requests("POST", "/upload/image").length === 0 && mock.requests("POST", "/prompt").length === 0,
    "a refused image input uploads nothing and queues nothing",
    `uploads=${mock.requests("POST", "/upload/image").length} prompts=${mock.requests("POST", "/prompt").length}`
  );

  const genuine = await generate(main, {
    baseUrl: mock.baseUrl,
    workflow: createReferenceWorkflow(["char_1"]),
    references: [{ name: "char_1", dataUrl }],
    timeoutMs: 20_000
  });
  assertCheck(scope, genuine.status === 200, "a LoadImage holding %ref:<name>% of an uploaded reference still renders", describe(genuine));
}

/**
 * No relayed graph samples below CFG 1 (red team S1-S3, R2). Below 1 the sampler draws the NEGATIVE — at 0 it is the
 * negative alone, so "a naked child" written on a graph's negative side became the picture at HTTP 200. Every
 * guidance-scale input (cfg and its variants, neg_scale, nag_scale) must be a number of at least 1, a numeric string
 * is read as the number ComfyUI casts it to, and a link is refused. Held with the node check off (*) as well.
 */
async function evaluateGuidanceFloor({ mock, main, openNodes }) {
  const scope = "guidance";
  mock.reset();
  const withSampler = (inputs) => {
    const workflow = createTextToImageWorkflow();
    workflow["8"].inputs = { ...workflow["8"].inputs, ...inputs };
    return workflow;
  };
  const negativeIsThePicture = () => {
    const workflow = createTextToImageWorkflow();
    workflow["11"] = { class_type: "CLIPTextEncode", inputs: { clip: ["3", 0], text: "the text a graph keeps on its negative side" } };
    workflow["8"].inputs = { ...workflow["8"].inputs, negative: ["11", 0], cfg: 0 };
    return workflow;
  };
  const refusedCases = [
    [main, negativeIsThePicture(), ["8.cfg"], "a KSampler at cfg 0 with text on its negative side (the R2 request)"],
    [main, withSampler({ cfg: 0.5 }), ["8.cfg"], "a KSampler cfg 0.5"],
    [main, withSampler({ cfg: "0.1" }), ["8.cfg"], "a KSampler cfg written as the string \"0.1\""],
    [main, withSampler({ cfg: "" }), ["8.cfg"], "a KSampler cfg that is an empty string"],
    [main, withSampler({ cfg: null }), ["8.cfg"], "a KSampler cfg of null"],
    [
      main,
      { ...withSampler({ cfg: ["20", 0] }), "20": { class_type: "PrimitiveFloat", inputs: { value: 0 } } },
      ["8.cfg"],
      "a KSampler cfg fed by a PrimitiveFloat link"
    ],
    [
      main,
      (() => {
        const workflow = createReferenceWorkflow([]);
        workflow["7"].inputs.cfg = 0.2;
        return workflow;
      })(),
      ["7.cfg"],
      "a CFGGuider cfg 0.2"
    ],
    [openNodes, withSampler({ cfg: 0 }), ["8.cfg"], "a KSampler cfg 0 with DYNAMICCHAT_COMFYUI_EXTRA_NODES=*"],
    [
      openNodes,
      { ...createTextToImageWorkflow(), "12": { class_type: "DualCFGGuider", inputs: { model: ["2", 0], cond1: ["5", 0], cond2: ["5", 0], negative: ["6", 0], cfg_conds: 0, cfg_cond2_negative: 0.9 } } },
      ["12.cfg_conds", "12.cfg_cond2_negative"],
      "a DualCFGGuider below 1 with the node check off"
    ],
    [
      openNodes,
      { ...createTextToImageWorkflow(), "12": { class_type: "PerpNegGuider", inputs: { model: ["2", 0], positive: ["5", 0], negative: ["6", 0], empty_conditioning: ["6", 0], cfg: 4, neg_scale: 0.5 } } },
      ["12.neg_scale"],
      "a PerpNegGuider neg_scale 0.5 with the node check off"
    ]
  ];
  for (const [server, workflow, rejected, label] of refusedCases) {
    const result = await generate(server, { baseUrl: mock.baseUrl, workflow, timeoutMs: 20_000 });
    assertCheck(
      scope,
      result.status === 400 && /1 이상의 숫자/u.test(result.json?.error ?? "") && /네거티브/u.test(result.json?.error ?? "") && isDeepStrictEqual(result.json?.rejectedGuidanceInputs, rejected),
      `${label} is refused with a Korean 400 naming ${rejected.join(", ")}`,
      describe(result)
    );
  }
  assertCheck(
    scope,
    mock.requests("POST", "/prompt").length === 0 && mock.requests("POST", "/upload/image").length === 0,
    "a graph refused for its guidance is never queued",
    `prompts=${mock.requests("POST", "/prompt").length}`
  );

  for (const [server, workflow, label] of [
    [main, withSampler({ cfg: 1 }), "cfg 1 (the Turbo presets' fixed value)"],
    [main, withSampler({ cfg: 7.5 }), "cfg 7.5"],
    [main, withSampler({ cfg: "4" }), "cfg written as the string \"4\""],
    [openNodes, { ...createTextToImageWorkflow(), "12": { class_type: "RenormCFG", inputs: { model: ["2", 0], cfg_trunc: 0, renorm_cfg: 1 } } }, "cfg_trunc 0 (a step bound, not a scale) with the node check off"]
  ]) {
    mock.reset();
    const result = await generate(server, { baseUrl: mock.baseUrl, workflow, timeoutMs: 20_000 });
    assertCheck(scope, result.status === 200 && mock.requests("POST", "/prompt").length === 1, `${label} still renders`, describe(result));
  }
}

async function evaluateReferenceValidation({ mock, main, referencePng, simulationId }) {
  const scope = "references";
  mock.reset();
  const workflow = createReferenceWorkflow(["char_1"]);
  const dataUrl = toDataUrl(referencePng);
  const cases = [
    [[{ name: "char_1", objectKey: "../x.png" }], /objectKey/u, "a traversal objectKey (../x.png)"],
    [[{ name: "char_1", objectKey: "..%2Fx" }], /objectKey/u, "an encoded traversal objectKey (..%2Fx)"],
    [[{ name: "char_1", objectKey: `${simulationId}/../../x.png` }], /objectKey/u, "a traversal hidden after a valid segment"],
    [[{ name: "char_1", objectKey: `${simulationId}/notes.json` }], /objectKey/u, "an objectKey that is not an image file"],
    [[{ name: "char_1", objectKey: `${simulationId}/missing.png` }], /찾을 수 없습니다/u, "an objectKey with no file behind it"],
    [[{ name: "char 1", dataUrl }], /이름/u, "a reference name outside [A-Za-z0-9_.-]"],
    [[{ name: "char_1", dataUrl: "data:image/gif;base64,R0lGODlhAQABAAAAACw=" }], /dataUrl/u, "a GIF data URL"],
    [[{ name: "char_1", dataUrl: toDataUrl(Buffer.from("definitely not an image")) }], /PNG·JPEG·WebP/u, "a data URL whose bytes are not an image"],
    [[{ name: "char_1" }], /objectKey도 dataUrl도/u, "a reference with neither objectKey nor dataUrl"],
    [[{ name: "char_1", dataUrl }, { name: "char_1", dataUrl }], /중복/u, "duplicate reference names"],
    [Array.from({ length: 7 }, (_, index) => ({ name: `char_${index + 1}`, dataUrl })), /6장/u, "seven references (the cap is six)"],
    ["char_1", /배열/u, "references that are not an array"]
  ];
  for (const [references, pattern, label] of cases) {
    const result = await generate(main, { baseUrl: mock.baseUrl, workflow, references });
    assertCheck(scope, result.status === 400 && pattern.test(result.json?.error ?? ""), `${label} is refused with 400`, describe(result));
  }

  const oversized = toDataUrl(Buffer.concat([referencePng, Buffer.alloc(21 * 1024 * 1024)]));
  const tooBig = await generate(main, { baseUrl: mock.baseUrl, workflow, references: [{ name: "char_1", dataUrl: oversized }] });
  assertCheck(scope, tooBig.status === 400 && /20MB/u.test(tooBig.json?.error ?? ""), "a reference over 20 MB decoded is refused with 400", describe(tooBig));

  const leftover = await generate(main, { baseUrl: mock.baseUrl, workflow: createReferenceWorkflow(["char_9"]) });
  assertCheck(
    scope,
    leftover.status === 400 && leftover.json?.unresolvedReferences?.includes("%ref:char_9%"),
    "a %ref:% placeholder with no matching reference is refused with 400",
    describe(leftover)
  );
  const embedded = createReferenceWorkflow(["char_1"]);
  embedded["20"].inputs.image = "faces/%ref:char_1%.png";
  const embeddedResult = await generate(main, { baseUrl: mock.baseUrl, workflow: embedded, references: [{ name: "char_1", dataUrl }] });
  assertCheck(scope, embeddedResult.status === 400 && /자리표시자/u.test(embeddedResult.json?.error ?? ""), "a %ref: placeholder inside a longer string is refused, not half-substituted", describe(embeddedResult));

  assertCheck(
    scope,
    mock.requests("POST", "/upload/image").length === 0 && mock.requests("POST", "/prompt").length === 0,
    "a refused reference uploads nothing and queues nothing",
    `uploads=${mock.requests("POST", "/upload/image").length} prompts=${mock.requests("POST", "/prompt").length}`
  );
}

async function evaluateSuccessfulGeneration({ mock, main, outputPng, referencePng, objectKey }) {
  const scope = "generate";
  mock.reset();
  const workflow = createReferenceWorkflow(["char_1"]);
  const result = await generate(
    main,
    {
      baseUrl: mock.baseUrl,
      workflow,
      outputNodeIds: ["13"],
      references: [{ name: "char_1", dataUrl: toDataUrl(referencePng), characterId: "char_eval" }],
      timeoutMs: 20_000,
      clientRequestId: "imgjob_relay_eval"
    },
    {
      origin: APP_ORIGIN,
      headers: { cookie: "session=abc123", "sec-fetch-site": "same-site", "sec-fetch-dest": "empty", referer: `${APP_ORIGIN}/run` }
    }
  );
  const promptId = [...mock.prompts.keys()][0];
  const image = result.json?.images?.[0];
  assertCheck(scope, result.status === 200, "generate with a dataUrl reference answers 200", describe(result));
  assertCheck(scope, typeof promptId === "string" && result.json?.promptId === promptId, "the response carries ComfyUI's prompt id", `${result.json?.promptId} vs ${promptId}`);
  assertCheck(
    scope,
    result.json?.images?.length === 1 && image?.mimeType === "image/png" && Buffer.from(image.base64 ?? "", "base64").equals(outputPng),
    "the rendered PNG comes back as base64, byte for byte",
    JSON.stringify({ count: result.json?.images?.length, mimeType: image?.mimeType })
  );
  assertCheck(
    scope,
    image?.nodeId === "13" && image?.type === "temp" && image?.subfolder === "" && /^ComfyUI_temp_[a-z]{5}_\d{5}_\.png$/u.test(image?.filename ?? ""),
    "the image keeps ComfyUI's filename/subfolder/type and names its node (a PreviewImage result: temp folder, ComfyUI's own name)",
    JSON.stringify({ nodeId: image?.nodeId, filename: image?.filename, subfolder: image?.subfolder, type: image?.type })
  );
  const views = mock.requests("GET", "/view");
  assertCheck(
    scope,
    views.length === 1 && views[0].query.type === "temp" && views[0].query.filename === image?.filename,
    "the result is downloaded from ComfyUI's temp folder (/view?type=temp), once",
    JSON.stringify(views.map((entry) => entry.query))
  );
  assertCheck(scope, Number.isFinite(result.json?.elapsedMs) && result.json.elapsedMs >= 0, "elapsedMs is reported", String(result.json?.elapsedMs));

  const upload = mock.requests("POST", "/upload/image")[0];
  const parts = upload?.body?.parts ?? [];
  const imagePart = parts.find((part) => part.name === "image");
  const field = (name) => parts.find((part) => part.name === name)?.value;
  assertCheck(
    scope,
    imagePart?.byteLength === referencePng.length && imagePart?.magic === "89504e470d0a1a0a" && /^multipart\/form-data; boundary=/u.test(upload?.headers["content-type"] ?? ""),
    "the reference reached ComfyUI as a multipart upload of the exact PNG bytes",
    JSON.stringify(parts)
  );
  assertCheck(scope, /^dc_char_1_[0-9a-f]{16}\.png$/u.test(imagePart?.filename ?? ""), "the uploaded file is named dc_<name>_<content digest>.png", imagePart?.filename);
  assertCheck(
    scope,
    field("type") === "input" && field("subfolder") === "dynamicchat" && field("overwrite") === "true",
    "the upload targets input/dynamicchat with overwrite on",
    JSON.stringify({ type: field("type"), subfolder: field("subfolder"), overwrite: field("overwrite") })
  );

  const queued = mock.requests("POST", "/prompt")[0]?.body ?? {};
  const submitted = queued.prompt ?? {};
  assertCheck(scope, submitted["20"]?.inputs?.image === `dynamicchat/${imagePart?.filename}`, "%ref:char_1% was replaced by dynamicchat/<the returned name>", submitted["20"]?.inputs?.image);
  assertCheck(scope, !JSON.stringify(submitted).includes("%ref:"), "no %ref: placeholder reached ComfyUI");
  assertCheck(
    scope,
    isDeepStrictEqual({ ...submitted, 20: undefined }, { ...workflow, 20: undefined }),
    "every other node is submitted exactly as sent"
  );
  assertCheck(
    scope,
    queued.client_id === "dynamicchat-relay" && !Object.hasOwn(queued, "prompt_id"),
    "the prompt is queued with client_id dynamicchat-relay and no prompt_id",
    JSON.stringify(Object.keys(queued))
  );
  assertCheck(scope, mock.requests("GET", `/history/${promptId}`).length >= 3, "history was polled until the prompt finished (two pending answers, then done)", String(mock.requests("GET", `/history/${promptId}`).length));

  const leaked = mock.received.filter(
    (entry) => entry.headers.origin || entry.headers.cookie || entry.headers.referer || entry.headers["sec-fetch-site"] || entry.headers["sec-fetch-dest"]
  );
  assertCheck(
    scope,
    leaked.length === 0 && mock.received.every((entry) => !entry.rejectedByOriginCheck),
    "none of the browser's Origin / Cookie / Referer / Sec-Fetch-* headers reached ComfyUI",
    JSON.stringify(leaked.map((entry) => ({ path: entry.path, headers: entry.headers })))
  );

  // objectKey reference, plus a second reference with different bytes.
  mock.reset();
  const secondPng = createMockPng({ rgb: [0x10, 0x20, 0x30] });
  const fromStore = await generate(main, {
    baseUrl: mock.baseUrl,
    workflow: createReferenceWorkflow(["char_1", "char_2"]),
    outputNodeIds: ["13"],
    references: [
      { name: "char_1", objectKey },
      { name: "char_2", dataUrl: toDataUrl(secondPng) }
    ],
    timeoutMs: 20_000
  });
  const storeWorkflow = mock.requests("POST", "/prompt")[0]?.body?.prompt ?? {};
  const uploadedNames = mock.requests("POST", "/upload/image").map((entry) => entry.body?.parts?.find((part) => part.name === "image")?.filename);
  assertCheck(scope, fromStore.status === 200 && fromStore.json?.images?.length === 1, "generate with an objectKey reference answers 200", describe(fromStore));
  assertCheck(
    scope,
    mock.uploads.get(`dynamicchat/${uploadedNames[0]}`)?.equals(referencePng) === true,
    "the objectKey reference uploaded the stored file's bytes",
    uploadedNames.join(", ")
  );
  assertCheck(
    scope,
    uploadedNames[0] === imagePart?.filename && uploadedNames[1] !== uploadedNames[0] && /^dc_char_2_[0-9a-f]{16}\.png$/u.test(uploadedNames[1] ?? ""),
    "identical bytes map to one upload name and different bytes to another (no cross-job overwrite)",
    uploadedNames.join(", ")
  );
  assertCheck(
    scope,
    storeWorkflow["20"]?.inputs?.image === `dynamicchat/${uploadedNames[0]}` && storeWorkflow["21"]?.inputs?.image === `dynamicchat/${uploadedNames[1]}`,
    "each placeholder gets its own reference's upload",
    JSON.stringify([storeWorkflow["20"]?.inputs?.image, storeWorkflow["21"]?.inputs?.image])
  );

  mock.reset();
  mock.setBehaviour("upload_rename");
  const renamed = await generate(main, {
    baseUrl: mock.baseUrl,
    workflow: createReferenceWorkflow(["char_1"]),
    references: [{ name: "char_1", dataUrl: toDataUrl(referencePng) }],
    timeoutMs: 20_000
  });
  const renamedWorkflow = mock.requests("POST", "/prompt")[0]?.body?.prompt ?? {};
  assertCheck(
    scope,
    renamed.status === 200 && renamedWorkflow["20"]?.inputs?.image === `dynamicchat/renamed_${imagePart?.filename}`,
    "the relay uses the name ComfyUI returned for an upload, not the one it sent",
    renamedWorkflow["20"]?.inputs?.image
  );

  mock.reset();
  const twoOutputs = createTextToImageWorkflow({ prefix: "DynamicChat/first" });
  twoOutputs["30"] = { class_type: "SaveImage", inputs: { images: ["9", 0], filename_prefix: "DynamicChat/second" } };
  twoOutputs["31"] = { class_type: "PreviewImage", inputs: { images: ["9", 0] } };
  const ordered = await generate(main, { baseUrl: mock.baseUrl, workflow: twoOutputs, outputNodeIds: ["30", "10"], timeoutMs: 20_000 });
  const orderedNames = (ordered.json?.images ?? []).map((entry) => entry.filename);
  assertCheck(
    scope,
    ordered.status === 200 && orderedNames.length === 2 && orderedNames[0]?.startsWith("second_") && orderedNames[1]?.startsWith("first_"),
    "saved images come back in outputNodeIds order, previews left out",
    JSON.stringify(orderedNames)
  );
  assertCheck(scope, mock.requests("GET", "/view").every((entry) => entry.query.type === "output"), "only saved (type output) images are downloaded when there are any");

  mock.reset();
  const previewOnly = createTextToImageWorkflow();
  previewOnly["10"] = { class_type: "PreviewImage", inputs: { images: ["9", 0] } };
  const preview = await generate(main, { baseUrl: mock.baseUrl, workflow: previewOnly, timeoutMs: 20_000 });
  assertCheck(
    scope,
    preview.status === 200 && preview.json?.images?.length === 1 && preview.json.images[0].type === "temp",
    "a workflow that ends in PreviewImage still returns its image",
    describe(preview)
  );

  // Which images are the result is decided by the node the client names, not by the folder: a built-in graph's
  // PreviewImage result must not be displaced by some other node that happens to save.
  const strayWorkflow = () => {
    const workflow = createTextToImageWorkflow();
    workflow["10"] = { class_type: "PreviewImage", inputs: { images: ["9", 0] } };
    workflow["30"] = { class_type: "SaveImage", inputs: { images: ["9", 0], filename_prefix: "DynamicChat/stray" } };
    return workflow;
  };
  mock.reset();
  const declared = await generate(main, { baseUrl: mock.baseUrl, workflow: strayWorkflow(), outputNodeIds: ["10"], timeoutMs: 20_000 });
  const declaredImages = declared.json?.images ?? [];
  assertCheck(
    scope,
    declared.status === 200 &&
      declaredImages.length === 1 &&
      declaredImages[0].nodeId === "10" &&
      declaredImages[0].type === "temp" &&
      mock.requests("GET", "/view").every((entry) => entry.query.type === "temp"),
    "the declared output node is the result even when it previews and another node saves (the saved image is not fetched)",
    JSON.stringify(declaredImages.map(({ nodeId, type, filename }) => ({ nodeId, type, filename })))
  );
  mock.reset();
  const undeclared = await generate(main, { baseUrl: mock.baseUrl, workflow: strayWorkflow(), timeoutMs: 20_000 });
  const undeclaredImages = undeclared.json?.images ?? [];
  assertCheck(
    scope,
    undeclared.status === 200 && undeclaredImages.length === 1 && undeclaredImages[0].nodeId === "30" && undeclaredImages[0].type === "output",
    "with no output node named, a saved image is preferred over a preview",
    JSON.stringify(undeclaredImages.map(({ nodeId, type, filename }) => ({ nodeId, type, filename })))
  );
  mock.reset();
  const emptyDeclared = await generate(main, { baseUrl: mock.baseUrl, workflow: strayWorkflow(), outputNodeIds: ["9"], timeoutMs: 20_000 });
  assertCheck(
    scope,
    emptyDeclared.status === 200 && emptyDeclared.json?.images?.length === 1 && emptyDeclared.json.images[0].nodeId === "30",
    "a named node that produced no image falls back to the same folder rule",
    describe(emptyDeclared)
  );
}

async function evaluateComfyUiFailures({ mock, main }) {
  const scope = "ComfyUI failures";
  mock.reset();
  mock.setBehaviour("node_errors");
  const rejected = await generate(main, { baseUrl: mock.baseUrl, workflow: createTextToImageWorkflow(), timeoutMs: 20_000 });
  const message = rejected.json?.error ?? "";
  assertCheck(scope, rejected.status === 400, "ComfyUI's 400 (node_errors) becomes a relay 400", describe(rejected));
  assertCheck(
    scope,
    message.startsWith("ComfyUI가 워크플로를 거부했습니다") &&
      message.includes("모델 파일 'z_image_turbo_int8_convrot.safetensors'") &&
      message.includes("models/diffusion_models"),
    "the Korean message names the missing model file and the folder it belongs in",
    message
  );
  assertCheck(
    scope,
    rejected.json?.nodeErrors?.["1"]?.errors?.[0]?.type === "value_not_in_list" &&
      rejected.json?.nodeErrors?.["1"]?.class_type === "UNETLoader" &&
      rejected.json?.comfyError?.type === "prompt_outputs_failed_validation",
    "nodeErrors and comfyError are handed back",
    JSON.stringify({ nodeErrors: rejected.json?.nodeErrors, comfyError: rejected.json?.comfyError })
  );
  assertCheck(scope, !rejected.text.includes("input_config"), "the loader's folder listing (input_config) is not passed back");
  assertCheck(scope, mock.received.every((entry) => !entry.path.startsWith("/history")), "a refused prompt is not polled");

  mock.reset();
  mock.setBehaviour("execution_error");
  const failed = await generate(main, { baseUrl: mock.baseUrl, workflow: createTextToImageWorkflow(), timeoutMs: 20_000 });
  assertCheck(
    scope,
    failed.status === 500 && (failed.json?.error ?? "").startsWith("ComfyUI 실행 오류: KSampler: Allocation on device"),
    "a history status_str error becomes 500 'ComfyUI 실행 오류: <node>: <exception>'",
    describe(failed)
  );
  assertCheck(scope, (failed.json?.error ?? "").includes("GPU 메모리 부족"), "an out-of-memory failure carries the VRAM hint", failed.json?.error);
  assertCheck(scope, failed.json?.promptId === [...mock.prompts.keys()][0], "the failed prompt's id is reported", describe(failed));

  // The client retries a 502 once, on the understanding that nothing was queued. A failure after the render
  // must therefore not be a 502, or the retry pays for the whole cut again.
  mock.reset();
  const noOutputs = createTextToImageWorkflow();
  delete noOutputs["10"];
  const empty = await generate(main, { baseUrl: mock.baseUrl, workflow: noOutputs, timeoutMs: 20_000 });
  assertCheck(
    scope,
    empty.status === 500 && /결과 이미지가 없습니다/u.test(empty.json?.error ?? ""),
    "a prompt that finishes without images answers 500, not a retryable 502",
    describe(empty)
  );
}

async function evaluateCancellation({ mock, main }) {
  const scope = "cancellation";
  mock.reset();
  mock.setBehaviour("never_completes");
  const startedAt = Date.now();
  const timedOut = await generate(main, { baseUrl: mock.baseUrl, workflow: createTextToImageWorkflow(), timeoutMs: 1_500 });
  const elapsedMs = Date.now() - startedAt;
  const promptId = [...mock.prompts.keys()][0];
  assertCheck(scope, timedOut.status === 504, "a prompt that does not finish within timeoutMs becomes 504", describe(timedOut));
  assertCheck(scope, elapsedMs >= 1_400 && elapsedMs < 8_000, "the 504 arrives at the requested timeout, not the five-minute default", `${elapsedMs}ms`);
  assertCheck(
    scope,
    timedOut.json?.promptId === promptId && /중단/u.test(timedOut.json?.error ?? ""),
    "the 504 names the prompt and says it was stopped",
    describe(timedOut)
  );
  const deleteIndex = mock.received.findIndex((entry) => entry.method === "POST" && entry.path === "/queue" && isDeepStrictEqual(entry.body, { delete: [promptId] }));
  const interruptIndex = mock.received.findIndex((entry) => entry.method === "POST" && entry.path === "/interrupt" && isDeepStrictEqual(entry.body, { prompt_id: promptId }));
  assertCheck(scope, interruptIndex >= 0, "the timed-out prompt is interrupted (POST /interrupt {prompt_id})");
  assertCheck(scope, deleteIndex >= 0, "the timed-out prompt is deleted from the queue (POST /queue {delete:[id]})");
  assertCheck(scope, deleteIndex >= 0 && deleteIndex < interruptIndex, "the queue delete goes first, so a still-pending prompt cannot start in between", `delete=${deleteIndex} interrupt=${interruptIndex}`);

  mock.reset();
  mock.setBehaviour("never_completes");
  const controller = new AbortController();
  const pending = fetch(`${main.baseUrl}${GENERATE}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ baseUrl: mock.baseUrl, workflow: createTextToImageWorkflow(), timeoutMs: 60_000 }),
    signal: controller.signal
  }).catch((error) => error);
  const queuedId = await waitFor(() => [...mock.prompts.keys()][0], 5_000);
  await waitFor(() => (queuedId ? mock.requests("GET", `/history/${queuedId}`).length > 0 : false), 5_000);
  controller.abort();
  await pending;
  const interrupted = await waitFor(
    () => mock.requests("POST", "/interrupt").find((entry) => entry.body?.prompt_id === queuedId),
    5_000
  );
  const removed = mock.requests("POST", "/queue").find((entry) => isDeepStrictEqual(entry.body, { delete: [queuedId] }));
  assertCheck(scope, Boolean(queuedId && interrupted), "a browser that disconnects mid-render gets its prompt interrupted", `prompt=${queuedId}`);
  assertCheck(scope, Boolean(queuedId && removed), "... and taken off the queue", `prompt=${queuedId}`);
}

async function evaluateUnreachableUpstream({ mock, main }) {
  const scope = "unreachable";
  const closedBase = `http://127.0.0.1:${await findFreePort()}`;
  const expected = `ComfyUI 서버에 연결할 수 없습니다 (${closedBase}). ComfyUI를 실행했는지 확인하세요.`;
  const down = await generate(main, { baseUrl: closedBase, workflow: createTextToImageWorkflow() });
  assertCheck(scope, down.status === 502 && down.json?.error === expected, "generate against a closed port → 502 with the Korean 'cannot connect' message", describe(down));
  const statusDown = await api(main, `${STATUS}?baseUrl=${encodeURIComponent(closedBase)}`);
  assertCheck(
    scope,
    statusDown.status === 200 && statusDown.json?.ok === false && statusDown.json?.message === expected,
    "status against a closed port → 200 { ok: false, message }",
    describe(statusDown)
  );
  const freeDown = await api(main, FREE, { method: "POST", body: { baseUrl: closedBase } });
  const interruptDown = await api(main, INTERRUPT, { method: "POST", body: { baseUrl: closedBase, promptId: SAMPLE_PROMPT_ID } });
  assertCheck(scope, freeDown.status === 502 && interruptDown.status === 502, "free and interrupt against a closed port → 502", `${describe(freeDown)} / ${describe(interruptDown)}`);

  mock.reset();
  mock.setBehaviour("upstream_down");
  const dropped = await generate(main, { baseUrl: mock.baseUrl, workflow: createTextToImageWorkflow() });
  assertCheck(scope, dropped.status === 502 && (dropped.json?.error ?? "").startsWith("ComfyUI 서버에 연결할 수 없습니다"), "a ComfyUI that drops the connection → 502", describe(dropped));

  mock.reset();
  mock.setBehaviour("redirect");
  const statusRedirect = await api(main, `${STATUS}?baseUrl=${encodeURIComponent(mock.baseUrl)}`);
  assertCheck(
    scope,
    statusRedirect.status === 200 && statusRedirect.json?.ok === false && /리디렉션/u.test(statusRedirect.json?.message ?? ""),
    "a 302 from /system_stats is reported as a problem, not followed",
    describe(statusRedirect)
  );
  const generateRedirect = await generate(main, { baseUrl: mock.baseUrl, workflow: createTextToImageWorkflow() });
  assertCheck(scope, generateRedirect.status === 502 && /리디렉션/u.test(generateRedirect.json?.error ?? ""), "a 302 from POST /prompt is an error, not followed", describe(generateRedirect));
  assertCheck(scope, mock.received.every((entry) => entry.path !== "/redirect-target"), "the redirect target is never requested");
}

async function evaluateStatusEndpoint({ mock, main }) {
  const scope = "status";
  mock.reset();
  const nodeNames = ["UNETLoader", "TextEncodeQwenImage21", "QwenImage21Cache", "NotARealNode", "KSampler (Efficient)"];
  const status = await api(main, `${STATUS}?baseUrl=${encodeURIComponent(mock.baseUrl)}&nodes=${encodeURIComponent(nodeNames.join(","))}`);
  const json = status.json ?? {};
  assertCheck(scope, status.status === 200 && json.ok === true && json.comfyuiVersion === MOCK_COMFYUI_VERSION, "status reports ok and the ComfyUI version", describe(status));
  assertCheck(
    scope,
    json.devices?.length === 1 &&
      json.devices[0].name === MOCK_COMFYUI_GPU &&
      json.devices[0].vramTotalMb === Math.round(MOCK_COMFYUI_VRAM.totalBytes / 1048576) &&
      json.devices[0].vramFreeMb === Math.round(MOCK_COMFYUI_VRAM.freeBytes / 1048576),
    "devices carry the card name (without ComfyUI's cuda:0 / allocator wrapping) and VRAM in MB",
    JSON.stringify(json.devices)
  );
  assertCheck(
    scope,
    isDeepStrictEqual(Object.keys(json.models ?? {}).sort(), ["diffusion_models", "loras", "text_encoders", "vae"]) &&
      json.models.diffusion_models.includes("qwen_image_2.1_turbo_int8_convrot.safetensors") &&
      json.models.vae.includes("ae.safetensors") &&
      isDeepStrictEqual(json.models.loras, []),
    "models lists the four folders with their files",
    JSON.stringify(json.models)
  );
  assertCheck(
    scope,
    isDeepStrictEqual(json.nodes, { UNETLoader: true, TextEncodeQwenImage21: true, QwenImage21Cache: true, NotARealNode: false, "KSampler (Efficient)": false }),
    "nodes maps each requested node to installed true/false",
    JSON.stringify(json.nodes)
  );
  assertCheck(scope, json.queueRemaining === 0, "queueRemaining is reported", String(json.queueRemaining));
  assertCheck(
    scope,
    isDeepStrictEqual(Object.keys(json).sort(), ["comfyuiVersion", "devices", "models", "nodes", "ok", "queueRemaining"]),
    "a successful probe has exactly the spec's fields (the client composes the message)",
    JSON.stringify(Object.keys(json))
  );
  assertCheck(scope, !/argv|python_version|ram_total/u.test(status.text), "system_stats is not passed through wholesale (no argv, Python or RAM details)");
  assertCheck(
    scope,
    mock.received.some((entry) => entry.path === "/object_info/KSampler%20(Efficient)"),
    "a node name with spaces is probed as one encoded path segment",
    JSON.stringify(mock.received.filter((entry) => entry.path.startsWith("/object_info")).map((entry) => entry.path))
  );

  mock.setModels("loras", null);
  const noLoras = await api(main, `${STATUS}?baseUrl=${encodeURIComponent(mock.baseUrl)}`);
  assertCheck(
    scope,
    noLoras.json?.ok === true && isDeepStrictEqual(noLoras.json?.models?.loras, []) && isDeepStrictEqual(noLoras.json?.nodes, {}),
    "a model folder ComfyUI does not know (404) is reported as [], and no nodes requested means an empty map",
    describe(noLoras)
  );
  mock.reset();

  const tooMany = await api(main, `${STATUS}?baseUrl=${encodeURIComponent(mock.baseUrl)}&nodes=${Array.from({ length: 31 }, (_, index) => `Node${index}`).join(",")}`);
  const traversal = await api(main, `${STATUS}?baseUrl=${encodeURIComponent(mock.baseUrl)}&nodes=${encodeURIComponent("../system_stats")}`);
  assertCheck(scope, tooMany.status === 400 && traversal.status === 400, "status refuses more than 30 nodes and node names that could move the probe path", `${describe(tooMany)} / ${describe(traversal)}`);
}

async function evaluateFreeAndInterrupt({ mock, main }) {
  const scope = "free/interrupt";
  mock.reset();
  const freed = await api(main, FREE, { method: "POST", body: { baseUrl: mock.baseUrl } });
  assertCheck(scope, freed.status === 200 && freed.json?.ok === true, "free answers 200 { ok: true }", describe(freed));
  assertCheck(
    scope,
    isDeepStrictEqual(mock.requests("POST", "/free")[0]?.body, { unload_models: true, free_memory: true }),
    "free is forwarded as POST /free { unload_models: true, free_memory: true }",
    JSON.stringify(mock.requests("POST", "/free")[0]?.body)
  );

  const stopped = await api(main, INTERRUPT, { method: "POST", body: { baseUrl: mock.baseUrl, promptId: SAMPLE_PROMPT_ID } });
  assertCheck(scope, stopped.status === 200 && stopped.json?.promptId === SAMPLE_PROMPT_ID, "interrupt answers 200 with the prompt id", describe(stopped));
  assertCheck(
    scope,
    isDeepStrictEqual(mock.requests("POST", "/interrupt")[0]?.body, { prompt_id: SAMPLE_PROMPT_ID }) &&
      isDeepStrictEqual(mock.requests("POST", "/queue")[0]?.body, { delete: [SAMPLE_PROMPT_ID] }),
    "interrupt is forwarded as POST /interrupt { prompt_id } plus POST /queue { delete: [id] }",
    JSON.stringify(mock.received.map((entry) => [entry.path, entry.body]))
  );

  const before = mock.received.length;
  const missingId = await api(main, INTERRUPT, { method: "POST", body: { baseUrl: mock.baseUrl } });
  const badId = await api(main, INTERRUPT, { method: "POST", body: { baseUrl: mock.baseUrl, promptId: "../free" } });
  assertCheck(
    scope,
    missingId.status === 400 && badId.status === 400 && mock.received.length === before,
    "interrupt without a valid promptId is refused before reaching ComfyUI (a bare /interrupt stops every job)",
    `${describe(missingId)} / ${describe(badId)}`
  );
}

async function evaluateBodyLimits({ main }) {
  const scope = "body limits";
  const tooLarge = await postWithDeclaredLength(main.port, GENERATE, 200 * 1024 * 1024);
  let parsed;
  try {
    parsed = JSON.parse(tooLarge.body);
  } catch {
    parsed = undefined;
  }
  assertCheck(scope, tooLarge.status === 413 && parsed?.error === "Request body too large.", "a body declared over the cap is refused with 413 before it is read", `status=${tooLarge.status} body=${tooLarge.body.slice(0, 200)}`);
  const malformed = await api(main, GENERATE, { method: "POST", body: "{not json" });
  assertCheck(scope, malformed.status === 400 && /JSON/u.test(malformed.json?.error ?? ""), "malformed JSON is a 400, not a 500", describe(malformed));
  const arrayBody = await api(main, FREE, { method: "POST", body: "[]" });
  assertCheck(scope, arrayBody.status === 400, "a JSON body that is not an object is a 400", describe(arrayBody));
}

async function evaluateEnvironmentConfiguration({ mock, main, configured, openNodes, aliases }) {
  const scope = "environment";
  mock.reset();
  const defaultStatus = await api(configured, STATUS);
  assertCheck(scope, defaultStatus.status === 200 && defaultStatus.json?.ok === true, "with DYNAMICCHAT_COMFYUI_URL set, status needs no baseUrl", describe(defaultStatus));
  const defaultGenerate = await generate(configured, { workflow: createTextToImageWorkflow(), timeoutMs: 20_000 });
  assertCheck(scope, defaultGenerate.status === 200 && defaultGenerate.json?.images?.length === 1, "with DYNAMICCHAT_COMFYUI_URL set, generate needs no baseUrl", describe(defaultGenerate));

  const configuredOrigin = await api(configured, "/personal-api-vault", { origin: mock.baseUrl });
  const sameOriginOnMain = await api(main, "/personal-api-vault", { origin: mock.baseUrl });
  assertCheck(
    scope,
    configuredOrigin.status === 403 && sameOriginOnMain.status === 200,
    "the origin of DYNAMICCHAT_COMFYUI_URL is denied on every guarded route (vault included), and only where it is configured",
    `${describe(configuredOrigin)} / ${describe(sameOriginOnMain)}`
  );
  const deniedVault = await api(configured, "/personal-api-vault", { origin: UNLISTED_LOCAL_ORIGIN });
  const deniedStatus = await api(configured, STATUS, { origin: UNLISTED_LOCAL_ORIGIN });
  const allowedOnMain = await api(main, "/personal-api-vault", { origin: UNLISTED_LOCAL_ORIGIN });
  assertCheck(
    scope,
    deniedVault.status === 403 && deniedStatus.status === 403 && allowedOnMain.status === 200,
    "DYNAMICCHAT_DENIED_ORIGINS denies a loopback origin that would otherwise pass",
    [deniedVault, deniedStatus, allowedOnMain].map(describe).join(" / ")
  );

  // What is denied is the server behind an entry, so every spelling of it is, scheme-less entries included.
  const aliasCases = [
    [configured, `http://localhost:${mock.port}`, `the configured ComfyUI (${mock.baseUrl}) opened as localhost`],
    [configured, `https://[::1]:${mock.port}`, "... or as https://[::1]"],
    [configured, "http://[::1]:7777", `a listed origin (${UNLISTED_LOCAL_ORIGIN}) written as [::1]`],
    [configured, "http://127.0.0.1:7778", "a scheme-less host:port entry (localhost:7778)"],
    [configured, "https://localhost:7779", "a scheme-less IPv6 entry ([::1]:7779) in another spelling and scheme"],
    [aliases, `http://127.0.0.1:${mock.port}`, `DYNAMICCHAT_COMFYUI_URL=http://localhost:${mock.port}, while ComfyUI prints and opens 127.0.0.1`],
    [aliases, `http://[::1]:${mock.port}`, "... or [::1]"]
  ];
  for (const [server, origin, label] of aliasCases) {
    const vault = await api(server, "/personal-api-vault", { origin });
    assertCheck(scope, vault.status === 403 && /ComfyUI/u.test(vault.json?.message ?? ""), `${label}: ${origin} → 403 on the "${server.label}" server`, describe(vault));
  }
  const unlistedElsewhere = await Promise.all(["http://127.0.0.1:7778", "https://localhost:7779"].map((origin) => api(main, "/personal-api-vault", { origin })));
  assertCheck(
    scope,
    unlistedElsewhere.every((result) => result.status === 200),
    "those ports are ordinary loopback origins where nothing lists them",
    unlistedElsewhere.map(describe).join(" / ")
  );
  assertCheck(
    scope,
    configured.output().includes('DYNAMICCHAT_DENIED_ORIGINS entry "definitely not an origin"'),
    "a DYNAMICCHAT_DENIED_ORIGINS entry that is neither an origin nor host:port is reported at startup, not dropped in silence",
    configured.output().slice(0, 400)
  );
  mock.reset();
  const aliasDefault = await generate(aliases, { workflow: createTextToImageWorkflow(), timeoutMs: 20_000 }, { origin: `http://127.0.0.1:${mock.port}` });
  const appOnAliases = await api(aliases, "/personal-api-vault", { origin: APP_ORIGIN });
  assertCheck(
    scope,
    aliasDefault.status === 403 && mock.requests("POST", "/prompt").length === 0 && appOnAliases.status === 200,
    "that page cannot queue a render through the default base URL either, while the app's own origin still passes",
    `${describe(aliasDefault)} / ${describe(appOnAliases)}`
  );

  const python = createTextToImageWorkflow();
  python["99"] = { class_type: "ExecutePython", inputs: { code: "print(1)" } };
  const lanListed = await generate(configured, { baseUrl: LAN_COMFYUI, workflow: python });
  const lanUnlisted = await generate(main, { baseUrl: LAN_COMFYUI, workflow: python });
  assertCheck(
    scope,
    lanListed.status === 400 && lanListed.json?.rejectedNodes?.includes("ExecutePython") && !/DYNAMICCHAT_COMFYUI_HOSTS/u.test(lanListed.json?.error ?? ""),
    "a host listed in DYNAMICCHAT_COMFYUI_HOSTS passes base-URL validation (the request gets as far as the node check)",
    describe(lanListed)
  );
  assertCheck(scope, lanUnlisted.status === 400 && /DYNAMICCHAT_COMFYUI_HOSTS/u.test(lanUnlisted.json?.error ?? ""), "the same host is refused where it is not listed", describe(lanUnlisted));

  mock.reset();
  const custom = createTextToImageWorkflow();
  custom["40"] = { class_type: "MyCustomNode", inputs: { strength: 0.5 } };
  const customListed = await generate(configured, { baseUrl: mock.baseUrl, workflow: custom, timeoutMs: 20_000 });
  assertCheck(
    scope,
    customListed.status === 200 && mock.requests("POST", "/prompt")[0]?.body?.prompt?.["40"]?.class_type === "MyCustomNode",
    "DYNAMICCHAT_COMFYUI_EXTRA_NODES admits the node it names",
    describe(customListed)
  );
  const pythonStillRefused = await generate(configured, { baseUrl: mock.baseUrl, workflow: python });
  assertCheck(scope, pythonStillRefused.status === 400, "naming one extra node does not open the list (ExecutePython still refused)", describe(pythonStillRefused));
  mock.reset();
  const pythonOpen = await generate(openNodes, { baseUrl: mock.baseUrl, workflow: python, timeoutMs: 20_000 });
  assertCheck(scope, pythonOpen.status === 200, "DYNAMICCHAT_COMFYUI_EXTRA_NODES=* turns the node check off", describe(pythonOpen));
  const lanOpen = await generate(openNodes, { baseUrl: LAN_COMFYUI, workflow: python });
  assertCheck(scope, lanOpen.status === 400 && /DYNAMICCHAT_COMFYUI_HOSTS/u.test(lanOpen.json?.error ?? ""), "turning the node check off leaves the host allow-list in force", describe(lanOpen));
}

async function evaluateClientEndToEnd({ mock, main, outputPng, referencePng, objectKey }) {
  const scope = "client end to end";
  if (!existsSync(path.join(workspaceRoot, "src", "services", "comfyUiClient.ts"))) {
    skipped.push(`${scope}: src/services/comfyUiClient.ts does not exist yet, so generatePhotorealImages was not run through the relay`);
    return;
  }
  const vite = await createServer({
    root: workspaceRoot,
    logLevel: "silent",
    server: { middlewareMode: true },
    appType: "custom"
  });
  try {
    let client;
    try {
      client = await vite.ssrLoadModule("/src/services/comfyUiClient.ts");
    } catch (error) {
      // The client is built in parallel with the modules it imports. A module that does not exist yet is a SKIP
      // (typecheck reports it too); anything else that stops it loading is a failure.
      const missingModule = /Failed to load url (\S+).*Does the file exist\?/su.exec(String(error?.message ?? ""));
      if (missingModule) {
        skipped.push(`${scope}: comfyUiClient.ts imports ${missingModule[1]}, which does not exist yet`);
        return;
      }
      throw error;
    }
    const { seedState, hydrateState } = await vite.ssrLoadModule("/src/data/seed.ts");
    const exported = ["generatePhotorealImages", "probeComfyUi", "freeComfyUiMemory", "interruptComfyUi", "getPhotorealRelayBase"];
    const missingExports = exported.filter((name) => typeof client[name] !== "function");
    assertCheck(scope, missingExports.length === 0, "comfyUiClient exports the spec's functions", `missing: ${missingExports.join(", ")}`);
    if (missingExports.length > 0) {
      return;
    }

    const relayBase = `${main.baseUrl}/photoreal/comfyui`;
    const state = createClientState(seedState, hydrateState, { baseUrl: mock.baseUrl, proxyUrl: relayBase });
    assertCheck(scope, client.getPhotorealRelayBase(state) === relayBase, "getPhotorealRelayBase honours photoreal.proxyUrl", client.getPhotorealRelayBase(state));

    const cue = {
      shouldGenerate: true,
      reason: "comfyui relay eval",
      characters: [],
      tags: ["1girl", "sitting", "window"],
      scene: "a quiet room by the window",
      frame: { shot: "upper_body", viewpoint: "front", visibleRegions: ["head", "torso"] },
      kind: "scene"
    };
    mock.reset();
    const result = await client.generatePhotorealImages({
      state,
      prompt: "A candid smartphone photograph of an adult woman reading by a window.",
      negativePrompt: "anime, illustration",
      cue,
      presetId: "z_image_turbo",
      references: [],
      jobId: "imgjob_client_e2e",
      seed: 1234
    });
    const queued = mock.requests("POST", "/prompt")[0]?.body?.prompt ?? {};
    assertCheck(
      scope,
      result?.dataUrls?.length === 1 && result.dataUrls[0] === `data:image/png;base64,${outputPng.toString("base64")}`,
      "generatePhotorealImages returns the rendered PNG as a data URL through the relay",
      JSON.stringify(result?.dataUrls?.map((url) => url.slice(0, 40)))
    );
    assertCheck(
      scope,
      typeof result?.payload?.promptId === "string" && result.payload.promptId === [...mock.prompts.keys()][0],
      "the result payload carries ComfyUI's prompt id",
      JSON.stringify(result?.payload?.promptId)
    );
    assertCheck(
      scope,
      Object.values(queued).some((node) => JSON.stringify(node?.inputs ?? {}).includes("reading by a window")),
      "the prompt text reached the submitted workflow",
      JSON.stringify(Object.values(queued).map((node) => node?.class_type))
    );
    const queuedOutputs = Object.values(queued).filter((node) => node?.class_type === "PreviewImage" || node?.class_type === "SaveImage");
    const clientViews = mock.requests("GET", "/view");
    assertCheck(
      scope,
      queuedOutputs.length === 1 && queuedOutputs[0].class_type === "PreviewImage" && clientViews.length === 1 && clientViews[0].query.type === "temp",
      "the built-in graph ends in PreviewImage, so nothing is written to ComfyUI's output folder, and its image comes back from the temp folder",
      JSON.stringify({ outputs: queuedOutputs.map((node) => node.class_type), views: clientViews.map((entry) => entry.query) })
    );
    assertCheck(scope, !JSON.stringify(result?.payload ?? {}).includes(outputPng.toString("base64")), "the result payload does not carry the image bytes");

    mock.reset();
    const withReference = await client.generatePhotorealImages({
      state,
      prompt: "The woman from <image1> sits by a window.",
      negativePrompt: "",
      cue,
      presetId: "qwen_image_2_1_turbo",
      references: [{ name: "char_1", objectKey, characterId: "char_eval" }],
      jobId: "imgjob_client_ref",
      seed: 99
    });
    const referenceWorkflow = mock.requests("POST", "/prompt")[0]?.body?.prompt ?? {};
    const loadImages = Object.values(referenceWorkflow).filter((node) => node?.class_type === "LoadImage");
    const uploadedName = [...mock.uploads.keys()][0];
    assertCheck(
      scope,
      withReference?.dataUrls?.length === 1 && loadImages.length === 1 && uploadedName && loadImages[0].inputs?.image === uploadedName,
      "a qwen_image_2_1_turbo cut with a face reference is uploaded and wired into LoadImage",
      JSON.stringify({ loadImages: loadImages.map((node) => node.inputs?.image), uploadedName })
    );
    assertCheck(
      scope,
      mock.uploads.get(uploadedName ?? "")?.equals(referencePng) === true && !JSON.stringify(withReference?.payload ?? {}).includes(referencePng.toString("base64")),
      "the stored profile bytes reach ComfyUI and are not copied into the payload"
    );

    const probe = await client.probeComfyUi(state);
    assertCheck(
      scope,
      probe?.ok === true && probe.comfyuiVersion === MOCK_COMFYUI_VERSION && probe.devices?.[0]?.name === MOCK_COMFYUI_GPU,
      "probeComfyUi reports the version and the card through the relay",
      JSON.stringify(probe)
    );

    mock.reset();
    mock.setBehaviour("node_errors");
    let failure;
    try {
      await client.generatePhotorealImages({ state, prompt: "x", negativePrompt: "", cue, presetId: "z_image_turbo", references: [], jobId: "imgjob_client_fail" });
    } catch (error) {
      failure = error;
    }
    assertCheck(
      scope,
      failure instanceof Error && failure.message.startsWith("실사 이미지 생성 실패: ") && failure.message.includes("ComfyUI가 워크플로를 거부했습니다"),
      "a relay 400 surfaces as '실사 이미지 생성 실패: <relay message>'",
      failure?.message ?? "no error thrown"
    );

    mock.reset();
    await client.freeComfyUiMemory(state);
    assertCheck(scope, mock.requests("POST", "/free").length === 1, "freeComfyUiMemory reaches ComfyUI's /free through the relay", String(mock.requests("POST", "/free").length));
  } finally {
    await vite.close();
  }
}

/**
 * vite.config.ts against the ComfyUI origin. The API guard refuses http://127.0.0.1:8188, but the default data
 * directory sits inside the folder the dev server serves, so the same keys and transcripts were one request away on
 * the web port. A real dev server is started with the project's config over a throwaway root that holds a canary
 * data directory — the developer's own .dynamicchat-data is never read, so a regression cannot print it.
 */
async function evaluateDevWebServer({ tempRoot }) {
  const scope = "dev web server";
  const root = await realpath(await mkdtemp(path.join(tempRoot, "web-root-")));
  const canary = `dynamicchat-dev-server-canary-${process.pid}`;
  const dataDir = path.join(root, ".dynamicchat-data");
  await mkdir(path.join(dataDir, "objects", "sim_eval"), { recursive: true });
  await writeFile(path.join(dataDir, "state.json"), JSON.stringify({ canary }));
  await writeFile(path.join(dataDir, "dev-secrets.json"), JSON.stringify({ owners: { local_user: { encodedSecret: canary } } }));
  await writeFile(path.join(dataDir, "objects", "sim_eval", "asset_eval.png"), canary);
  await writeFile(path.join(root, "index.html"), '<!doctype html><html><head><title>eval</title></head><body><div id="root"></div></body></html>');
  await writeFile(path.join(root, "served.txt"), "served");
  const port = await findFreePort();
  const web = await createServer({
    configFile: path.join(workspaceRoot, "vite.config.ts"),
    root,
    cacheDir: path.join(root, ".vite-cache"),
    logLevel: "silent",
    server: { host: "127.0.0.1", port, strictPort: true }
  });
  await web.listen();
  try {
    const base = `http://127.0.0.1:${port}`;
    const absoluteData = dataDir.replace(/\\/gu, "/");
    const comfyOrigin = "http://127.0.0.1:8188";
    const probes = [
      "/.dynamicchat-data/state.json",
      "/.dynamicchat-data/dev-secrets.json",
      "/.dynamicchat-data/objects/sim_eval/asset_eval.png",
      `/@fs/${absoluteData}/dev-secrets.json`,
      "/.dynamicchat-data/state.json?import",
      "/.dynamicchat-data/state.json?raw",
      "/%2Edynamicchat-data/state.json",
      "/./.dynamicchat-data/state.json"
    ];
    for (const pathname of probes) {
      const response = await fetch(`${base}${pathname}`, { headers: { origin: comfyOrigin, accept: "*/*" } });
      const body = await response.text();
      assertCheck(
        scope,
        (response.status === 403 || response.status === 404) && !body.includes(canary) && response.headers.get("access-control-allow-origin") === null,
        `the data directory is refused to ${comfyOrigin}: ${pathname} → 403/404, no CORS header`,
        `status=${response.status} acao=${response.headers.get("access-control-allow-origin")} canary=${body.includes(canary)}`
      );
    }
    const servedCrossOrigin = await fetch(`${base}/served.txt`, { headers: { origin: comfyOrigin } });
    await servedCrossOrigin.text();
    assertCheck(
      scope,
      servedCrossOrigin.status === 200 && servedCrossOrigin.headers.get("access-control-allow-origin") === null,
      "no response carries a CORS header, so nothing the dev server serves is readable from another origin",
      `status=${servedCrossOrigin.status} acao=${servedCrossOrigin.headers.get("access-control-allow-origin")}`
    );
    const served = await fetch(`${base}/served.txt`);
    const index = await fetch(`${base}/`, { headers: { accept: "text/html" } });
    const servedText = await served.text();
    const indexText = await index.text();
    assertCheck(
      scope,
      served.status === 200 && servedText === "served" && index.status === 200 && indexText.includes('<div id="root">'),
      "the rest of the project is still served to the app itself (the deny entry is specific to the data directory)",
      `served=${served.status} index=${index.status}`
    );
  } finally {
    await web.close();
  }
}

function createClientState(seedState, hydrateState, overrides) {
  let base;
  try {
    base = hydrateState(structuredClone(seedState));
  } catch {
    base = structuredClone(seedState);
  }
  return {
    ...base,
    photoreal: {
      ...(base.photoreal ?? {}),
      enabled: true,
      provider: "comfyui",
      requestMode: "proxy",
      workflowPreset: "z_image_turbo",
      referenceWorkflowPreset: "qwen_image_2_1_turbo",
      modelFiles: {},
      steps: 0,
      cfg: 0,
      width: 0,
      height: 0,
      seedFixed: false,
      stylePreset: "daily_look",
      stylePrompt: "",
      negativePrompt: "",
      faceReferenceEnabled: true,
      vramHandoff: "off",
      customWorkflow: "",
      timeoutSeconds: 60,
      registrationStatus: "idle",
      ...overrides
    }
  };
}

// The z_image_turbo graph, ending in SaveImage as a custom workflow may (the built-in one ends in PreviewImage):
// enough for the relay, which never interprets inputs beyond placeholders.
function createTextToImageWorkflow({ prefix = "DynamicChat/eval" } = {}) {
  return {
    "1": { class_type: "UNETLoader", inputs: { unet_name: "z_image_turbo_int8_convrot.safetensors", weight_dtype: "default" } },
    "2": { class_type: "ModelSamplingAuraFlow", inputs: { model: ["1", 0], shift: 3 } },
    "3": { class_type: "CLIPLoader", inputs: { clip_name: "qwen_3_4b_fp8_mixed.safetensors", type: "lumina2", device: "default" } },
    "4": { class_type: "VAELoader", inputs: { vae_name: "ae.safetensors" } },
    "5": { class_type: "CLIPTextEncode", inputs: { clip: ["3", 0], text: "A candid smartphone photograph of an adult woman by a window" } },
    "6": { class_type: "ConditioningZeroOut", inputs: { conditioning: ["5", 0] } },
    "7": { class_type: "EmptySD3LatentImage", inputs: { width: 1024, height: 1024, batch_size: 1 } },
    "8": {
      class_type: "KSampler",
      inputs: { model: ["2", 0], positive: ["5", 0], negative: ["6", 0], latent_image: ["7", 0], seed: 42, steps: 8, cfg: 1, sampler_name: "res_multistep", scheduler: "simple", denoise: 1 }
    },
    "9": { class_type: "VAEDecode", inputs: { samples: ["8", 0], vae: ["4", 0] } },
    "10": { class_type: "SaveImage", inputs: { images: ["9", 0], filename_prefix: prefix } }
  };
}

// The qwen_image_2_1_turbo graph with one LoadImage per reference, each holding its %ref:<name>% placeholder. It
// ends in PreviewImage like the built-in graph (createTextToImageWorkflow keeps SaveImage, as a custom one may).
function createReferenceWorkflow(referenceNames) {
  const workflow = {
    "1": { class_type: "UNETLoader", inputs: { unet_name: "qwen_image_2.1_turbo_int8_convrot.safetensors", weight_dtype: "default" } },
    "2": { class_type: "QwenImage21Cache", inputs: { model: ["1", 0], device: "auto", dtype: "default" } },
    "3": { class_type: "CLIPLoader", inputs: { clip_name: "qwen3vl_8b_int8_convrot.safetensors", type: "qwen_image", device: "default" } },
    "4": { class_type: "VAELoader", inputs: { vae_name: "qwen_image_2.1_vae_bf16.safetensors" } },
    "5": {
      class_type: "TextEncodeQwenImage21",
      inputs: { clip: ["3", 0], prompt: "The woman from <image1> reads by a window.", negative_prompt: "", vae: ["4", 0], resolution: 1024 }
    },
    "6": { class_type: "EmptyLatentImage", inputs: { width: 960, height: 1280, batch_size: 1 } },
    "7": { class_type: "CFGGuider", inputs: { model: ["2", 0], positive: ["5", 0], negative: ["5", 1], cfg: 1 } },
    "8": { class_type: "KSamplerSelect", inputs: { sampler_name: "euler" } },
    "9": { class_type: "RandomNoise", inputs: { noise_seed: 42 } },
    "10": { class_type: "ManualSigmas", inputs: { sigmas: "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0" } },
    "11": { class_type: "SamplerCustomAdvanced", inputs: { noise: ["9", 0], guider: ["7", 0], sampler: ["8", 0], sigmas: ["10", 0], latent_image: ["6", 0] } },
    "12": { class_type: "VAEDecode", inputs: { samples: ["11", 0], vae: ["4", 0] } },
    "13": { class_type: "PreviewImage", inputs: { images: ["12", 0] } }
  };
  referenceNames.forEach((name, index) => {
    const nodeId = String(20 + index);
    workflow[nodeId] = { class_type: "LoadImage", inputs: { image: `%ref:${name}%` } };
    workflow["5"].inputs[`images.image_${index + 1}`] = [nodeId, 0];
  });
  return workflow;
}

function createRelayRoutes(baseUrl) {
  return [
    { label: `POST ${GENERATE}`, method: "POST", path: GENERATE, body: { baseUrl, workflow: createTextToImageWorkflow(), timeoutMs: 20_000 } },
    { label: `GET ${STATUS}`, method: "GET", path: `${STATUS}?baseUrl=${encodeURIComponent(baseUrl)}` },
    { label: `POST ${FREE}`, method: "POST", path: FREE, body: { baseUrl } },
    { label: `POST ${INTERRUPT}`, method: "POST", path: INTERRUPT, body: { baseUrl, promptId: SAMPLE_PROMPT_ID } }
  ];
}

async function startApiServer(label, tempRoot, extraEnv) {
  const dataDir = path.join(tempRoot, label);
  await mkdir(dataDir, { recursive: true });
  const port = await findFreePort();
  const env = { ...process.env };
  for (const key of SERVER_ENV_KEYS) {
    delete env[key];
  }
  Object.assign(env, { DYNAMICCHAT_DATA_DIR: dataDir, DYNAMICCHAT_API_PORT: String(port), DYNAMICCHAT_RATE_LIMIT_MAX: "10000" }, extraEnv);
  const child = spawn(process.execPath, [serverEntry], { cwd: workspaceRoot, env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  let output = "";
  child.stdout.on("data", (chunk) => {
    output += chunk;
  });
  child.stderr.on("data", (chunk) => {
    output += chunk;
  });
  const exited = new Promise((resolve) => child.once("exit", resolve));
  const server = {
    label,
    port,
    dataDir,
    baseUrl: `http://127.0.0.1:${port}`,
    output: () => output,
    async stop() {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill();
        await Promise.race([exited, delay(5_000)]);
      }
    }
  };
  const startedAt = Date.now();
  while (Date.now() - startedAt < 20_000) {
    if (child.exitCode !== null) {
      throw new Error(`${label} server exited during startup (code ${child.exitCode}):\n${output}`);
    }
    try {
      const health = await fetch(`${server.baseUrl}/health`, { signal: AbortSignal.timeout(1_000) });
      await health.text();
      if (health.ok) {
        return server;
      }
    } catch {
      // Not listening yet.
    }
    await delay(100);
  }
  await server.stop();
  throw new Error(`${label} server did not answer /health within 20s:\n${output}`);
}

async function api(server, pathname, { method = "GET", body, origin, headers = {}, signal } = {}) {
  const requestHeaders = { ...headers };
  if (origin) {
    requestHeaders.origin = origin;
  }
  let payload;
  if (body !== undefined) {
    requestHeaders["content-type"] = "application/json";
    payload = typeof body === "string" ? body : JSON.stringify(body);
  }
  const response = await fetch(`${server.baseUrl}${pathname}`, { method, headers: requestHeaders, body: payload, signal });
  const text = await response.text();
  let json;
  try {
    json = text ? JSON.parse(text) : undefined;
  } catch {
    json = undefined;
  }
  return { status: response.status, json, text };
}

function generate(server, body, options = {}) {
  return api(server, GENERATE, { method: "POST", body, ...options });
}

// A request with a Host header of our choosing (fetch derives Host from the URL), as a DNS-rebound page sends it.
function rawApi(server, pathname, { method = "GET", host, origin, body } = {}) {
  return new Promise((resolve, reject) => {
    const headers = {};
    if (host !== undefined) {
      headers.host = host;
    }
    if (origin) {
      headers.origin = origin;
    }
    let payload;
    if (body !== undefined) {
      payload = JSON.stringify(body);
      headers["content-type"] = "application/json";
      headers["content-length"] = String(Buffer.byteLength(payload));
    }
    const request = http.request({ host: "127.0.0.1", port: server.port, method, path: pathname, headers }, (response) => {
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        let json;
        try {
          json = text ? JSON.parse(text) : undefined;
        } catch {
          json = undefined;
        }
        resolve({ status: response.statusCode, json, text });
      });
    });
    request.setTimeout(10_000, () => request.destroy(new Error(`no answer to ${method} ${pathname} within 10s`)));
    request.on("error", reject);
    request.end(payload);
  });
}

// Declares a body far over the cap and sends almost none of it: the 413 must come from the declared length alone.
function postWithDeclaredLength(port, pathname, declaredBytes) {
  return new Promise((resolve, reject) => {
    const request = http.request({
      host: "127.0.0.1",
      port,
      method: "POST",
      path: pathname,
      headers: { "content-type": "application/json", "content-length": String(declaredBytes) }
    });
    const timer = setTimeout(() => {
      request.destroy();
      reject(new Error("no answer to the oversized request within 10s"));
    }, 10_000);
    request.on("response", (response) => {
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => {
        clearTimeout(timer);
        request.destroy();
        resolve({ status: response.statusCode, body: Buffer.concat(chunks).toString("utf8") });
      });
    });
    request.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    request.write('{"workflow":');
  });
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.unref();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => resolve(port));
    });
  });
}

async function waitFor(read, timeoutMs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const value = read();
    if (value) {
      return value;
    }
    await delay(25);
  }
  return read() || undefined;
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function nestArray(depth) {
  let value = "leaf";
  for (let level = 0; level < depth; level += 1) {
    value = [value];
  }
  return value;
}

function toDataUrl(bytes) {
  return `data:image/png;base64,${bytes.toString("base64")}`;
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function describe(result) {
  return `status=${result?.status} body=${String(result?.text ?? "").slice(0, 300)}`;
}

async function runGroup(scope, evaluate) {
  try {
    await evaluate();
  } catch (error) {
    assertCheck(scope, false, `threw before finishing: ${error?.stack ?? error}`);
  }
}

function assertCheck(scope, passed, message, detail) {
  checks.push({
    level: passed ? "pass" : "fail",
    scope,
    message: passed || detail === undefined ? message : `${message} — ${detail}`
  });
}
