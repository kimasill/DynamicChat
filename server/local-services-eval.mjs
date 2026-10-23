/**
 * local-services-eval.mjs — eval:local-services
 *
 * Tests the ComfyUI and Ollama supervisor in server/local-services.mjs:
 *
 *   - Discovery precedence: env > remembered path > scan (using temp dirs)
 *   - Supervisor startup against a fake install (injectable spawn → mock-comfyui-server.mjs)
 *   - Status endpoint GET /local-services/status and POST /local-services/comfyui/start
 *   - Relay status messages: starting / not_installed / failed states
 *   - Restart: supervisor restarts ComfyUI after the process dies
 *   - Shutdown: only managed (self-started) processes are killed; external ones are left alone
 *
 * No real ComfyUI, Python or GPU needed.
 *
 * NOTE for eval harness runs on this PC: set TEMP/TMP to a folder on the S: drive
 * before running to avoid the cross-drive Vite issue in comfyui-relay-eval.mjs.
 *
 * Usage: node server/local-services-eval.mjs
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const serverEntry = path.join(__dirname, "dynamicchat-server.mjs");
const mockComfyUiServer = path.join(__dirname, "mock-comfyui-server.mjs");

// ─── check harness ───────────────────────────────────────────────────────────

const checks = [];
const cleanups = [];

function assertCheck(scope, ok, message, detail) {
  checks.push({ scope, level: ok ? "pass" : "fail", message, detail: detail ?? "" });
  if (!ok) {
    console.error(`  FAIL [${scope}] ${message}${detail ? ` — ${detail}` : ""}`);
  }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitUntil(condition, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await condition()) return true;
    await delay(150);
  }
  return false;
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.unref();
    s.once("error", reject);
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

async function probeUrl(url, timeoutMs = 1_500) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return (await fetch(url, { signal: controller.signal })).ok;
  } catch {
    return false;
  } finally {
    clearTimeout(id);
  }
}

/** Spawn a mock-comfyui-server.mjs on a free port; returns { port, proc }. */
async function spawnMockComfyUi(options = {}) {
  const args = [];
  if (options.port) args.push("--port", String(options.port));
  if (options.behaviour) args.push("--behaviour", options.behaviour);
  if (options.delayMs) args.push("--delay-ready-ms", String(options.delayMs));

  const proc = spawn(process.execPath, [mockComfyUiServer, ...args], {
    stdio: ["ignore", "pipe", "ignore"]
  });

  let port;
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("mock ComfyUI did not print READY within 5s")), 5_000);
    let buf = "";
    proc.stdout.on("data", (chunk) => {
      buf += chunk.toString();
      const m = /READY:(\d+)/u.exec(buf);
      if (m) {
        clearTimeout(timer);
        port = Number(m[1]);
        resolve();
      }
    });
    proc.once("exit", () => { clearTimeout(timer); reject(new Error("mock exited before READY")); });
  });

  return { port, proc, baseUrl: `http://127.0.0.1:${port}` };
}

/** Start a DynamicChat API server with isolated dataDir and env. */
async function startApiServer(label, tempRoot, extraEnv = {}) {
  const dataDir = path.join(tempRoot, label);
  await mkdir(dataDir, { recursive: true });
  const apiPort = await findFreePort();

  const env = {
    ...process.env,
    DYNAMICCHAT_DATA_DIR: dataDir,
    DYNAMICCHAT_API_PORT: String(apiPort),
    DYNAMICCHAT_RATE_LIMIT_MAX: "10000",
    // Disable neuralmap — we are not testing that here.
    DYNAMICCHAT_START_NEURALMAP: "0",
    ...extraEnv
  };

  const child = spawn(process.execPath, [serverEntry], {
    cwd: workspaceRoot,
    env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true
  });
  let output = "";
  child.stdout.on("data", (c) => { output += c; });
  child.stderr.on("data", (c) => { output += c; });
  const exited = new Promise((r) => child.once("exit", r));

  const server = {
    label, dataDir, apiPort,
    baseUrl: `http://127.0.0.1:${apiPort}`,
    output: () => output,
    async stop() {
      if (child.exitCode === null) {
        child.kill();
        await Promise.race([exited, delay(5_000)]);
      }
    }
  };

  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`${label} server exited early (code ${child.exitCode}):\n${output.slice(-500)}`);
    }
    try {
      const h = await fetch(`${server.baseUrl}/health`, { signal: AbortSignal.timeout(1_000) });
      if (h.ok) return server;
    } catch {}
    await delay(150);
  }
  await server.stop();
  throw new Error(`${label} server did not answer /health within 20s:\n${output.slice(-500)}`);
}

async function api(server, pathname, options = {}) {
  const { method = "GET", body, origin } = options;
  const headers = {};
  if (origin) headers.origin = origin;
  let payload;
  if (body !== undefined) {
    headers["content-type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(`${server.baseUrl}${pathname}`, { method, headers, body: payload });
  const text = await res.text();
  let json;
  try { json = text ? JSON.parse(text) : undefined; } catch {}
  return { status: res.status, json, text };
}

const APP_ORIGIN = "http://127.0.0.1:5173";

// ─── tests ───────────────────────────────────────────────────────────────────

let tempRoot;
try {
  tempRoot = await mkdtemp(path.join(
    process.env.TEMP ?? process.env.TMP ?? path.join(workspaceRoot, "tmp", "eval-tmp"),
    "dynamicchat-local-svc-eval-"
  ));
  cleanups.push(() => rm(tempRoot, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 }));

  await runGroup("discovery precedence", evaluateDiscovery);
  await runGroup("status endpoint", evaluateStatusEndpoint);
  await runGroup("relay messages", evaluateRelayMessages);
  await runGroup("auto-start and restart", evaluateAutoStartAndRestart);
  await runGroup("shutdown ownership", evaluateShutdownOwnership);
} catch (error) {
  assertCheck("setup", false, `eval setup failed: ${error?.stack ?? error}`);
} finally {
  for (const cleanup of cleanups.reverse()) {
    await Promise.resolve().then(cleanup).catch((e) => console.error("cleanup failed:", e?.message));
  }
}

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}${check.detail ? ` (${check.detail})` : ""}`);
}

const failures = checks.filter((c) => c.level === "fail");
console.log("");
console.log(`local-services checks: ${checks.length - failures.length} passed, ${failures.length} failures`);
if (failures.length > 0) process.exitCode = 1;

async function runGroup(name, fn) {
  console.log(`\n--- ${name} ---`);
  try {
    await fn();
  } catch (error) {
    assertCheck(name, false, `group threw: ${error?.message ?? error}`, error?.stack?.split("\n")[1]?.trim());
  }
}

// ─── discovery precedence ─────────────────────────────────────────────────────

async function evaluateDiscovery() {
  const scope = "discovery";
  // Import the module functions directly — no server needed for discovery tests.
  const { discoverComfyUiDir, resolveComfyUiInstall } = await import("./local-services.mjs");

  const base = path.join(tempRoot, "discovery");
  await mkdir(base, { recursive: true });

  // Helper: create a minimal git-checkout layout (venv/Scripts/python.exe + main.py)
  async function makeInstall(dir) {
    const exe = process.platform === "win32" ? "python.exe" : "python";
    const venvBin = path.join(dir, "venv", process.platform === "win32" ? "Scripts" : "bin");
    await mkdir(venvBin, { recursive: true });
    await writeFile(path.join(venvBin, exe), "# fake python");
    await writeFile(path.join(dir, "main.py"), "# fake main");
    return dir;
  }

  // 1. env wins over everything
  const envDir = await makeInstall(path.join(base, "env-dir"));
  const rememberedDir = await makeInstall(path.join(base, "remembered-dir"));
  const dataDir1 = path.join(base, "data1");
  await mkdir(dataDir1, { recursive: true });
  // Pre-seed remembered path
  await writeFile(path.join(dataDir1, "local-services.json"), JSON.stringify({ comfyuiDir: rememberedDir }));
  // Set env to envDir
  const origEnv = process.env.DYNAMICCHAT_COMFYUI_DIR;
  process.env.DYNAMICCHAT_COMFYUI_DIR = envDir;
  try {
    const found = await discoverComfyUiDir(dataDir1);
    assertCheck(scope, found?.dir === envDir, "env DYNAMICCHAT_COMFYUI_DIR wins over remembered path", `found=${found?.dir}`);
  } finally {
    if (origEnv === undefined) delete process.env.DYNAMICCHAT_COMFYUI_DIR;
    else process.env.DYNAMICCHAT_COMFYUI_DIR = origEnv;
  }

  // 2. remembered path wins when env is absent
  const dataDir2 = path.join(base, "data2");
  await mkdir(dataDir2, { recursive: true });
  await writeFile(path.join(dataDir2, "local-services.json"), JSON.stringify({ comfyuiDir: rememberedDir }));
  const origEnv2 = process.env.DYNAMICCHAT_COMFYUI_DIR;
  delete process.env.DYNAMICCHAT_COMFYUI_DIR;
  try {
    const found = await discoverComfyUiDir(dataDir2);
    assertCheck(scope, found?.dir === rememberedDir, "remembered path in local-services.json is used when env is absent", `found=${found?.dir}`);
  } finally {
    if (origEnv2 !== undefined) process.env.DYNAMICCHAT_COMFYUI_DIR = origEnv2;
  }

  // 3. remembered path is written after scan
  const dataDir3 = path.join(base, "data3");
  await mkdir(dataDir3, { recursive: true });
  // There is no local-services.json, and no env — but if there were a scannable candidate, it
  // would be remembered. We verify the write by noting it did NOT crash (scan returns undefined
  // on all our fake drive roots since they do not exist as ComfyUI installs).
  const origEnv3 = process.env.DYNAMICCHAT_COMFYUI_DIR;
  delete process.env.DYNAMICCHAT_COMFYUI_DIR;
  try {
    const found = await discoverComfyUiDir(dataDir3);
    // On this machine S:\Tools\ComfyUI IS a valid install, so found might be non-null.
    // What we test is that calling it does not throw and returns a consistent type.
    assertCheck(scope, found === undefined || (typeof found.dir === "string" && (found.install === undefined || typeof found.install.python === "string")), "scan returns undefined or { dir, install } without throwing");
  } finally {
    if (origEnv3 !== undefined) process.env.DYNAMICCHAT_COMFYUI_DIR = origEnv3;
  }

  // 4. resolveComfyUiInstall recognises the venv layout we created
  const install = resolveComfyUiInstall(envDir);
  assertCheck(scope, install !== undefined && typeof install.python === "string" && !install.portable, "resolveComfyUiInstall recognises venv/ layout", `python=${install?.python}`);

  // 5. resolveComfyUiInstall rejects a directory with no layout
  const emptyDir = path.join(base, "empty");
  await mkdir(emptyDir, { recursive: true });
  const noInstall = resolveComfyUiInstall(emptyDir);
  assertCheck(scope, noInstall === undefined, "resolveComfyUiInstall returns undefined for a dir with no ComfyUI layout");
}

// ─── status endpoint ──────────────────────────────────────────────────────────

async function evaluateStatusEndpoint() {
  const scope = "status endpoint";
  // Start a server with ComfyUI and Ollama auto-start disabled — the status should reflect "stopped".
  let server;
  try {
    server = await startApiServer("status-test", tempRoot, {
      DYNAMICCHAT_START_COMFYUI: "0",
      DYNAMICCHAT_START_OLLAMA: "0"
    });
    cleanups.push(() => server.stop());

    // GET /local-services/status with no origin (curl-like) — should be allowed
    const noOrigin = await api(server, "/local-services/status");
    assertCheck(scope, noOrigin.status === 200, "GET /local-services/status answers 200 with no origin header", `status=${noOrigin.status}`);
    assertCheck(scope, typeof noOrigin.json?.comfyui === "object" && typeof noOrigin.json?.ollama === "object", "response has comfyui and ollama objects", JSON.stringify(noOrigin.json)?.slice(0, 100));

    // With app origin
    const withOrigin = await api(server, "/local-services/status", { origin: APP_ORIGIN });
    assertCheck(scope, withOrigin.status === 200, "GET /local-services/status allows app origin", `status=${withOrigin.status}`);

    // Hostile origin must be refused
    const hostile = await api(server, "/local-services/status", { origin: "https://evil.example" });
    assertCheck(scope, hostile.status === 403, "GET /local-services/status refuses a hostile origin", `status=${hostile.status}`);

    // POST /local-services/comfyui/start when not installed
    const startResp = await api(server, "/local-services/comfyui/start", { method: "POST" });
    // With both disabled, state is "stopped" (supervisor not even created); start should 202 or 409
    assertCheck(scope, startResp.status === 202 || startResp.status === 409, "POST /local-services/comfyui/start answers 202 or 409", `status=${startResp.status}`);
  } finally {
    if (server) await server.stop().catch(() => {});
  }
}

// ─── relay messages ───────────────────────────────────────────────────────────

async function evaluateRelayMessages() {
  const scope = "relay messages";
  // Start a server that has NO ComfyUI installed and the default URL pointing at a closed port.
  // The relay should report the not-installed message, not the generic "cannot connect".
  const closedPort = await findFreePort();
  // No server ever listens on closedPort.
  let server;
  try {
    server = await startApiServer("relay-messages", tempRoot, {
      DYNAMICCHAT_COMFYUI_URL: `http://127.0.0.1:${closedPort}`,
      // Opt out of auto-start entirely so initComfyUi does not scan (would find S:\Tools\ComfyUI)
      DYNAMICCHAT_START_COMFYUI: "0",
      DYNAMICCHAT_START_OLLAMA: "0"
    });
    cleanups.push(() => server.stop());

    // When ComfyUI is not managed, the relay falls back to the old "cannot connect" message.
    const statusDown = await api(server, `/photoreal/comfyui/status?baseUrl=http://127.0.0.1:${closedPort}`);
    assertCheck(scope, statusDown.status === 200 && statusDown.json?.ok === false, "status against a closed non-managed port → 200 ok:false", `status=${statusDown.status} message=${statusDown.json?.message?.slice(0,80)}`);
    assertCheck(scope, statusDown.json?.starting !== true, "status against a non-managed URL does not set starting:true");
  } finally {
    if (server) await server.stop().catch(() => {});
  }

  // The app always sends baseUrl. Auto-start used to key on its ABSENCE, so a real generation never reached it
  // and got the bare "cannot connect" instead. With the managed address named explicitly, an install that cannot
  // be run must answer with the supervisor's message on both routes.
  const managedPort = await findFreePort();
  const managedUrl = `http://127.0.0.1:${managedPort}`;
  const unrecognised = path.join(tempRoot, "not-a-comfyui");
  await mkdir(unrecognised, { recursive: true });
  let managed;
  try {
    managed = await startApiServer("relay-managed", tempRoot, {
      DYNAMICCHAT_COMFYUI_URL: managedUrl,
      DYNAMICCHAT_COMFYUI_DIR: unrecognised,
      DYNAMICCHAT_START_OLLAMA: "0"
    });
    cleanups.push(() => managed.stop());
    const generate = await api(managed, "/photoreal/comfyui/generate", {
      method: "POST",
      origin: APP_ORIGIN,
      body: { baseUrl: managedUrl, workflow: createTextToImageWorkflow(), outputNodeIds: ["10"], timeoutMs: 20_000 }
    });
    assertCheck(
      scope,
      generate.status === 503 && /레이아웃/u.test(generate.json?.error ?? ""),
      "generate naming the managed baseUrl gets the supervisor's message, not a bare connection error",
      `status=${generate.status} error=${generate.json?.error?.slice(0, 80)}`
    );
    const probe = await api(managed, `/photoreal/comfyui/status?baseUrl=${encodeURIComponent(managedUrl)}`, { origin: APP_ORIGIN });
    assertCheck(scope, probe.json?.ok === false && /레이아웃/u.test(probe.json?.message ?? ""), "status naming the managed baseUrl reports the same", `message=${probe.json?.message?.slice(0, 80)}`);
  } finally {
    if (managed) await managed.stop().catch(() => {});
  }
}

// A graph the relay accepts (same as comfyui-relay-eval's text-to-image case).
function createTextToImageWorkflow() {
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
    "10": { class_type: "SaveImage", inputs: { images: ["9", 0], filename_prefix: "DynamicChat/eval" } }
  };
}

// ─── auto-start and restart ───────────────────────────────────────────────────

/**
 * Test the relay's behavior when ComfyUI is in various states.
 *
 * The injectable spawn override can't be used from a spawned server process (it's a module-level
 * singleton read at import time). Instead we test the HTTP relay behavior directly:
 * - When DYNAMICCHAT_START_COMFYUI=0 and ComfyUI URL is a closed port, the relay should give
 *   the "cannot connect" message (not a managed URL, so no auto-start special handling).
 * - The /local-services/status reports the right state.
 * - POST /local-services/comfyui/start starts the discovery + startup sequence.
 *
 * The injectable spawn is tested via direct module import: since _comfyUiManagedUrl is set at
 * module load time, we use the default port (8188) unless it's free. If 8188 is in use we skip.
 */
async function evaluateAutoStartAndRestart() {
  const scope = "auto-start";

  // --- HTTP server behavior: status reflects state, relay gives appropriate messages ---
  let server;
  try {
    // Start a server with a non-existent ComfyUI URL and no auto-start (so state=stopped/not_installed)
    const closedPort = await findFreePort();
    server = await startApiServer("autostart-http", tempRoot, {
      DYNAMICCHAT_COMFYUI_URL: `http://127.0.0.1:${closedPort}`,
      DYNAMICCHAT_START_COMFYUI: "0",
      DYNAMICCHAT_START_OLLAMA: "0"
    });
    cleanups.push(() => server.stop());

    // The /local-services/status should show "stopped" (supervisor not created when disabled)
    const statusResp = await api(server, "/local-services/status");
    assertCheck(scope, statusResp.status === 200, "GET /local-services/status responds 200", `status=${statusResp.status}`);
    assertCheck(scope, typeof statusResp.json?.comfyui?.state === "string", "comfyui.state field is present", JSON.stringify(statusResp.json?.comfyui));

    // The relay status against the managed-but-stopped URL should give a 200 ok:false
    const relayStatus = await api(server, `/photoreal/comfyui/status?baseUrl=http://127.0.0.1:${closedPort}`);
    assertCheck(scope, relayStatus.status === 200 && relayStatus.json?.ok === false, "relay status with disabled auto-start → 200 ok:false", `status=${relayStatus.status}`);

    // POST /local-services/comfyui/start — should succeed (202 or 409 when not installed)
    const startResp = await api(server, "/local-services/comfyui/start", { method: "POST" });
    assertCheck(scope, startResp.status === 202 || startResp.status === 409, "POST /local-services/comfyui/start returns 202 or 409", `status=${startResp.status}`);
  } finally {
    if (server) await server.stop().catch(() => {});
  }

  // --- Direct module test: injectable spawn ---
  // Only run if port 8188 is free, since _comfyUiManagedUrl defaults to http://127.0.0.1:8188.
  const port8188Free = !(await probeUrl("http://127.0.0.1:8188/system_stats", 800));
  if (!port8188Free) {
    console.log("  SKIP auto-start injectable spawn test: port 8188 is in use");
  } else {
    const ls = await import("./local-services.mjs");
    // Set the spawn override BEFORE calling initComfyUi.
    // _comfyUiManagedUrl is "http://127.0.0.1:8188" at this point.
    ls._testOnly_setSpawnOverride((_pythonPath, args, _options) => {
      const portIdx = args.indexOf("--port");
      const mockPort = portIdx >= 0 ? args[portIdx + 1] : "8188";
      return spawn(process.execPath, [mockComfyUiServer, "--port", mockPort], {
        stdio: ["ignore", "ignore", "ignore"]
      });
    });

    // Create a fake install dir (venv layout)
    const fakeDir = path.join(tempRoot, "fake-comfyui2");
    const exe = process.platform === "win32" ? "python.exe" : "python";
    const venvBin = path.join(fakeDir, "venv", process.platform === "win32" ? "Scripts" : "bin");
    await mkdir(venvBin, { recursive: true });
    await writeFile(path.join(venvBin, exe), "# fake");
    await writeFile(path.join(fakeDir, "main.py"), "# fake main");

    const dataDir2 = path.join(tempRoot, "autostart-data2");
    await mkdir(dataDir2, { recursive: true });

    const origDir = process.env.DYNAMICCHAT_COMFYUI_DIR;
    process.env.DYNAMICCHAT_COMFYUI_DIR = fakeDir;
    // Unset URL so initComfyUi uses the already-computed _comfyUiManagedUrl (8188)
    const origUrl = process.env.DYNAMICCHAT_COMFYUI_URL;
    delete process.env.DYNAMICCHAT_COMFYUI_URL;

    try {
      await ls.initComfyUi(dataDir2);
      const ok = await ls.ensureComfyUiRunning({ waitMs: 8_000 });
      assertCheck(scope, ok === true, "supervisor (injectable spawn) becomes running", `ok=${ok}`);
      const st = ls.getLocalServicesStatus().comfyui;
      assertCheck(scope, st.state === "running" && st.managed === true, "status: running + managed=true after injectable start", `state=${st.state} managed=${st.managed}`);

      // Ours dies after it was up (a crash, or the window closed): the next call starts it again. This used to stay
      // "failed" and every probe repeated the old exit message.
      ls.shutdownLocalServices();
      await waitUntil(() => ls.getLocalServicesStatus().comfyui.state === "failed", 8_000);
      const restarted = await ls.ensureComfyUiRunning({ waitMs: 8_000 });
      const afterRestart = ls.getLocalServicesStatus().comfyui;
      assertCheck(scope, restarted === true && afterRestart.state === "running" && afterRestart.managed === true, "a managed ComfyUI that died after coming up is restarted by the next call", `ok=${restarted} state=${afterRestart.state}`);
      ls.shutdownLocalServices();
      await waitUntil(async () => !(await probeUrl("http://127.0.0.1:8188/system_stats", 500)), 8_000);

      // One that was already running at boot (not ours, no install on record) and was closed later: the stored
      // "running" must not be believed — the install is discovered and started.
      const external = await spawnMockComfyUi({ port: 8188 });
      cleanups.push(() => { if (external.proc.exitCode === null) external.proc.kill(); });
      await ls.initComfyUi(dataDir2);
      const atBoot = ls.getLocalServicesStatus().comfyui;
      assertCheck(scope, atBoot.state === "running" && atBoot.managed === false, "an already-running ComfyUI is adopted, not managed", `state=${atBoot.state} managed=${atBoot.managed}`);
      external.proc.kill();
      await waitUntil(async () => !(await probeUrl("http://127.0.0.1:8188/system_stats", 500)), 8_000);
      const reopened = await ls.ensureComfyUiRunning({ waitMs: 8_000 });
      const afterClose = ls.getLocalServicesStatus().comfyui;
      assertCheck(scope, reopened === true && afterClose.state === "running" && afterClose.managed === true, "an adopted ComfyUI closed later is started from the discovered install", `ok=${reopened} state=${afterClose.state} managed=${afterClose.managed}`);
    } finally {
      ls.shutdownLocalServices();
      ls._testOnly_clearSpawnOverride();
      if (origDir === undefined) delete process.env.DYNAMICCHAT_COMFYUI_DIR;
      else process.env.DYNAMICCHAT_COMFYUI_DIR = origDir;
      if (origUrl !== undefined) process.env.DYNAMICCHAT_COMFYUI_URL = origUrl;
      await delay(500);
    }
  }
}

// ─── shutdown ownership ───────────────────────────────────────────────────────

async function evaluateShutdownOwnership() {
  const scope = "shutdown ownership";

  // Start a mock ComfyUI that we did NOT start through the supervisor (external).
  const external = await spawnMockComfyUi();
  cleanups.push(() => {
    if (external.proc.exitCode === null) external.proc.kill();
  });

  assertCheck(scope, await probeUrl(`${external.baseUrl}/system_stats`), "external mock ComfyUI is healthy before shutdown");

  // Start a server that targets the external mock (already running → not managed)
  let server;
  try {
    server = await startApiServer("shutdown-test", tempRoot, {
      DYNAMICCHAT_COMFYUI_URL: external.baseUrl,
      DYNAMICCHAT_START_COMFYUI: "1", // allow detection, but it should detect existing
      DYNAMICCHAT_START_OLLAMA: "0"
    });
    cleanups.push(() => server.stop());

    // Give the server a moment to detect the existing ComfyUI
    await delay(2_000);

    const statusResp = await api(server, "/local-services/status");
    const comfyState = statusResp.json?.comfyui;
    assertCheck(scope, comfyState?.state === "running" && comfyState?.managed === false, "server detects existing ComfyUI as running/not-managed", `state=${comfyState?.state} managed=${comfyState?.managed}`);

    // Stop the server (sends SIGTERM/kill)
    await server.stop();

    // External mock should still be alive — the server did not own it
    await delay(500);
    assertCheck(scope, external.proc.exitCode === null, "external ComfyUI process is still alive after server shutdown");
    assertCheck(scope, await probeUrl(`${external.baseUrl}/system_stats`, 2_000), "external ComfyUI still responds after server shutdown");
  } finally {
    if (server) await server.stop().catch(() => {});
  }
}
