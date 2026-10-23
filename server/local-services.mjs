/**
 * Local service supervisor — ComfyUI and Ollama.
 *
 * Discovery order for ComfyUI: env DYNAMICCHAT_COMFYUI_DIR → path remembered in
 * <dataDir>/local-services.json → shallow scan of common installation locations (existsSync
 * on fixed paths only, no recursion). A found path is written to local-services.json so later
 * starts skip the scan.
 *
 * Ownership rule: only processes this module started are killed on shutdown. A ComfyUI or
 * Ollama that was already running before the server started is never touched.
 *
 * Log output goes to <dataDir>/logs/comfyui.log and <dataDir>/logs/ollama.log, truncated on
 * each start (ComfyUI is very chatty). A one-line status is printed to the console instead:
 * "시작 중", "준비 완료 (Ns 경과)", "종료 (코드 N, Ns)".
 *
 * Opt-outs: DYNAMICCHAT_START_COMFYUI=0, DYNAMICCHAT_START_OLLAMA=0.
 */
import { execFile, spawn } from "node:child_process";
import { existsSync, openSync, closeSync, readFileSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";

// ─── constants ───────────────────────────────────────────────────────────────

const COMFYUI_STARTUP_POLL_MS = 2_000;
const COMFYUI_STARTUP_HARD_CEILING_MS = 180_000;
const COMFYUI_PROBE_TIMEOUT_MS = 3_000;
const OLLAMA_STARTUP_POLL_MS = 2_000;
const OLLAMA_STARTUP_MAX_MS = 60_000;
const OLLAMA_PROBE_TIMEOUT_MS = 1_500;

// Read once at module load — same source as the server's COMFYUI_CONFIGURED_BASE_URL.
const _comfyUiManagedUrl = (process.env.DYNAMICCHAT_COMFYUI_URL ?? "").trim() || "http://127.0.0.1:8188";
const _ollamaUrl = "http://127.0.0.1:11434";

// ─── ComfyUI install layout ─────────────────────────────────────────────────

/**
 * Same two-layout detection as dev-runner.mjs — both must agree on what counts as a valid
 * install, so moving the implementation here and having dev-runner re-export it keeps them in
 * sync without duplication.
 */
export function resolveComfyUiInstall(dir) {
  const exe = process.platform === "win32" ? "python.exe" : "python";
  const portable = {
    python: path.join(dir, "python_embeded", exe),
    main: path.join(dir, "ComfyUI", "main.py"),
    portable: true
  };
  if (existsSync(portable.python) && existsSync(portable.main)) return portable;
  const main = path.join(dir, "main.py");
  for (const venv of ["venv", ".venv"]) {
    const python = path.join(
      dir,
      venv,
      process.platform === "win32" ? "Scripts" : "bin",
      exe
    );
    if (existsSync(python) && existsSync(main)) return { python, main, portable: false };
  }
  return undefined;
}

// ─── discovery ───────────────────────────────────────────────────────────────

function buildScanCandidates() {
  const home = os.homedir();
  const candidates = [
    path.join(home, "ComfyUI"),
    path.join(home, "Documents", "ComfyUI"),
    path.join(home, "Desktop", "ComfyUI_windows_portable")
  ];
  if (process.platform === "win32") {
    // C–Z drive roots — existsSync on a non-existent drive returns false immediately.
    for (let code = 67; code <= 90; code++) {
      const root = `${String.fromCharCode(code)}:\\`;
      candidates.push(
        path.join(root, "ComfyUI"),
        path.join(root, "Tools", "ComfyUI"),
        path.join(root, "AI", "ComfyUI"),
        path.join(root, "ComfyUI_windows_portable"),
        path.join(root, "ComfyUI", "ComfyUI_windows_portable")
      );
    }
  }
  return candidates;
}

async function readRememberedComfyUiDir(dataDir) {
  try {
    const raw = await readFile(path.join(dataDir, "local-services.json"), "utf8");
    const data = JSON.parse(raw);
    return typeof data.comfyuiDir === "string" ? data.comfyuiDir : undefined;
  } catch {
    return undefined;
  }
}

async function saveRememberedComfyUiDir(dataDir, dir) {
  const p = path.join(dataDir, "local-services.json");
  try {
    let existing = {};
    try {
      existing = JSON.parse(await readFile(p, "utf8"));
    } catch {}
    await writeFile(p, `${JSON.stringify({ ...existing, comfyuiDir: dir }, null, 2)}\n`);
  } catch {}
}

/**
 * Returns `{ dir, install }` where install is the resolved layout (may be undefined if the dir
 * is forced by env but the layout is unrecognised), or undefined if nothing was found.
 */
export async function discoverComfyUiDir(dataDir) {
  // 1. env — honoured even when the layout is unrecognised, so the user sees a clear error.
  const envDir = process.env.DYNAMICCHAT_COMFYUI_DIR
    ? path.resolve(process.env.DYNAMICCHAT_COMFYUI_DIR)
    : undefined;
  if (envDir) {
    return { dir: envDir, install: resolveComfyUiInstall(envDir) };
  }
  // 2. remembered path
  const remembered = await readRememberedComfyUiDir(dataDir);
  if (remembered && existsSync(remembered)) {
    const install = resolveComfyUiInstall(remembered);
    if (install) return { dir: remembered, install };
  }
  // 3. shallow scan — fixed paths only, no recursion
  for (const candidate of buildScanCandidates()) {
    if (!existsSync(candidate)) continue;
    const install = resolveComfyUiInstall(candidate);
    if (install) {
      await saveRememberedComfyUiDir(dataDir, candidate);
      return { dir: candidate, install };
    }
  }
  return undefined;
}

// ─── health probes ───────────────────────────────────────────────────────────

async function probeHttp(url, timeoutMs) {
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

// ─── process tree kill ───────────────────────────────────────────────────────

function killTree(child) {
  if (!child || child.exitCode !== null || child.killed) return;
  if (process.platform === "win32") {
    // /T kills the entire process tree spawned from this pid.
    execFile("taskkill", ["/pid", String(child.pid), "/T", "/F"], () => {});
  } else {
    try {
      process.kill(-child.pid, "SIGKILL");
    } catch {}
    child.kill();
  }
}

// ─── supervisor ──────────────────────────────────────────────────────────────

/**
 * Tracks the lifecycle of one local service (ComfyUI or Ollama).
 * state: "stopped" | "starting" | "running" | "not_installed" | "failed"
 */
class ServiceSupervisor {
  constructor(label, url, logFile) {
    this.label = label;
    this.url = url;
    this.logFile = logFile;
    this.state = "stopped";
    this.managed = false; // true only if we spawned the process
    this.dir = undefined;
    this.message = undefined;
    this.startedAt = undefined;
    this.child = undefined;
    // Resolves when the current startup attempt finishes (true = healthy, false = failed/timeout).
    this._startPromise = undefined;
  }

  status() {
    return {
      state: this.state,
      managed: this.managed,
      url: this.url,
      ...(this.dir !== undefined ? { dir: this.dir } : {}),
      ...(this.message !== undefined ? { message: this.message } : {}),
      ...(this.logFile ? { logFile: this.logFile } : {})
    };
  }
}

// ─── module-level singletons ─────────────────────────────────────────────────

let _comfyUiSup = null;
let _ollamaSup = null;

// For testing: override the spawn function used to start ComfyUI. The override receives
// (pythonPath, args, spawnOptions) and must return a ChildProcess. Used by local-services-eval.mjs.
let _spawnOverride = null;
export function _testOnly_setSpawnOverride(fn) {
  _spawnOverride = fn;
}
export function _testOnly_clearSpawnOverride() {
  _spawnOverride = null;
}

// ─── ComfyUI ─────────────────────────────────────────────────────────────────

export async function initComfyUi(dataDir) {
  if (process.env.DYNAMICCHAT_START_COMFYUI === "0") {
    console.log("[local-services] DYNAMICCHAT_START_COMFYUI=0: ComfyUI 자동 시작 안 함");
    return;
  }
  const logFile = path.join(dataDir, "logs", "comfyui.log");
  _comfyUiSup = new ServiceSupervisor("comfyui", _comfyUiManagedUrl, logFile);
  // Kept for a later rediscovery: a ComfyUI that was already running at boot leaves no install on record.
  _comfyUiSup.dataDir = dataDir;

  if (await probeHttp(`${_comfyUiManagedUrl}/system_stats`, COMFYUI_PROBE_TIMEOUT_MS)) {
    console.log(`[comfyui] 이미 실행 중 (${_comfyUiManagedUrl})`);
    _comfyUiSup.state = "running";
    _comfyUiSup.managed = false;
    return;
  }

  const found = await discoverComfyUiDir(dataDir);
  if (!found) {
    _comfyUiSup.state = "not_installed";
    _comfyUiSup.message =
      "ComfyUI 설치 폴더를 찾지 못했습니다. ComfyUI를 설치하거나 DYNAMICCHAT_COMFYUI_DIR 환경변수로 폴더를 지정하세요.";
    // Not logged as an error — NovelAI-only setups never need ComfyUI.
    return;
  }
  if (!found.install) {
    _comfyUiSup.state = "not_installed";
    _comfyUiSup.dir = found.dir;
    _comfyUiSup.message = `${found.dir} 에서 ComfyUI를 찾았지만 레이아웃을 인식할 수 없습니다. python_embeded/+ComfyUI/main.py (포터블) 또는 venv/+main.py (git 체크아웃) 형식이어야 합니다.`;
    return;
  }
  _comfyUiSup.dir = found.dir;

  // Non-blocking: caller returns immediately, startup happens in the background.
  _doStartComfyUi(_comfyUiSup, found.install, COMFYUI_STARTUP_HARD_CEILING_MS).catch(() => {});
}

/**
 * Makes the loopback ComfyUI answer: returns true at once when it does, otherwise starts it — or restarts it, or
 * keeps watching one of ours that is still loading — and waits up to waitMs. waitMs 0 starts without waiting.
 *
 * The port is the truth, not the recorded state. Trusting the state left three dead ends: a ComfyUI that was
 * already running at boot and was closed later still read "running" and was never restarted; one of ours that
 * crashed read "failed" and every probe repeated that message; and one that came up after its startup wait
 * stayed "starting" for good. Installing ComfyUI after the server started is picked up here as well.
 */
export async function ensureComfyUiRunning({ waitMs = 120_000 } = {}) {
  if (!_comfyUiSup) return false;
  const sup = _comfyUiSup;

  if (await probeHttp(`${sup.url}/system_stats`, COMFYUI_PROBE_TIMEOUT_MS)) {
    if (sup.state !== "running") {
      sup.state = "running";
      sup.message = undefined;
    }
    return true;
  }

  if (!sup._startPromise) {
    if (sup.child && _isAlive(sup.child)) {
      // Ours, alive, still loading past the first watch: watch again rather than start a second copy.
      sup.state = "starting";
      _watchComfyUiStartup(sup, sup.child);
    } else {
      const install = sup.dir ? resolveComfyUiInstall(sup.dir) : undefined;
      const found = install ? { dir: sup.dir, install } : sup.dataDir ? await discoverComfyUiDir(sup.dataDir) : undefined;
      if (!found?.install) {
        sup.state = "not_installed";
        sup.managed = false;
        if (found?.dir) sup.dir = found.dir;
        sup.message = found
          ? `${found.dir} 에서 ComfyUI를 찾았지만 레이아웃을 인식할 수 없습니다. python_embeded/+ComfyUI/main.py (포터블) 또는 venv/+main.py (git 체크아웃) 형식이어야 합니다.`
          : "ComfyUI 설치 폴더를 찾지 못했습니다. ComfyUI를 설치하거나 DYNAMICCHAT_COMFYUI_DIR 환경변수로 폴더를 지정하세요.";
        return false;
      }
      sup.dir = found.dir;
      _doStartComfyUi(sup, found.install, COMFYUI_STARTUP_HARD_CEILING_MS).catch(() => {});
    }
  }

  if (!sup._startPromise || waitMs <= 0) return false;
  return Promise.race([
    sup._startPromise,
    new Promise((resolve) => setTimeout(() => resolve(false), waitMs))
  ]);
}

/** True when the given URL targets the loopback ComfyUI this module supervises. */
export function isComfyUiManagedUrl(url) {
  try {
    return new URL(url).origin === new URL(_comfyUiManagedUrl).origin;
  } catch {
    return false;
  }
}

/** Korean message for use in relay responses while ComfyUI is starting. */
export function getComfyUiStartingMessage() {
  if (!_comfyUiSup || _comfyUiSup.state !== "starting") return undefined;
  const elapsed = _comfyUiSup.startedAt
    ? Math.round((Date.now() - _comfyUiSup.startedAt) / 1000)
    : 0;
  return `ComfyUI를 시작하는 중입니다 (${elapsed}초 경과). 잠시 후 자동으로 연결됩니다.`;
}

export function getComfyUiNotInstalledMessage() {
  return (
    _comfyUiSup?.message ??
    "ComfyUI 설치 폴더를 찾지 못했습니다. ComfyUI를 설치하거나 DYNAMICCHAT_COMFYUI_DIR 환경변수로 폴더를 지정하세요."
  );
}

async function _doStartComfyUi(sup, install, hardCeilingMs) {
  const port = (() => {
    try {
      return new URL(sup.url).port || "8188";
    } catch {
      return "8188";
    }
  })();

  sup.state = "starting";
  sup.managed = true;
  sup.startedAt = Date.now();
  sup.message = undefined;

  // Opened BEFORE the first await so ensureComfyUiRunning can attach to it immediately even if
  // called right after initComfyUi returns.
  const watch = _openStartupWatch(sup);

  console.log(`[comfyui] 시작 중 (${sup.dir})`);

  await mkdir(path.dirname(sup.logFile), { recursive: true }).catch(() => {});

  let logFd;
  try {
    logFd = openSync(sup.logFile, "w");
  } catch {}

  const args = [
    ...(install.portable
      ? ["-s", install.main, "--windows-standalone-build"]
      : [install.main]),
    "--disable-auto-launch",
    "--listen",
    "127.0.0.1",
    "--port",
    port,
    "--preview-method",
    "none"
  ];

  let child;
  try {
    const spawnFn = _spawnOverride ?? spawn;
    child = spawnFn(install.python, args, {
      cwd: sup.dir,
      stdio: ["ignore", logFd ?? "ignore", logFd ?? "ignore"]
    });
  } catch (error) {
    try {
      if (logFd !== undefined) closeSync(logFd);
    } catch {}
    sup.state = "failed";
    sup.managed = false;
    sup.message = `ComfyUI를 시작할 수 없습니다: ${error.message}`;
    watch.finish(false);
    return false;
  }
  try {
    if (logFd !== undefined) closeSync(logFd);
  } catch {}

  sup.child = child;

  child.once("exit", (code) => {
    if (sup.child === child) {
      sup.child = undefined;
      if (sup.state === "starting" || sup.state === "running") {
        sup.state = "failed";
        const elapsed = Math.round((Date.now() - (sup.startedAt ?? Date.now())) / 1000);
        const lastLines = _readLastLines(sup.logFile, 10);
        const snippet = lastLines ? `\n마지막 로그:\n${lastLines}` : "";
        sup.message = `ComfyUI가 종료되었습니다 (종료 코드 ${code ?? 0}, ${elapsed}초 경과).${snippet}`;
        console.log(`[comfyui] 종료 (코드 ${code ?? 0}, ${elapsed}s)`);
      }
    }
  });

  return _pollComfyUiStartup(sup, child, watch, hardCeilingMs);
}

/**
 * The promise callers wait on while ComfyUI loads. `finish` clears it only if it is still the current one: a
 * watch that ends after a restart has opened a newer one must not take that one away from its waiters.
 */
function _openStartupWatch(sup) {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  sup._startPromise = promise;
  return {
    finish(result) {
      if (sup._startPromise === promise) sup._startPromise = undefined;
      resolve(result);
    }
  };
}

/** Watches one of our processes that outlived its first startup wait, without starting a second copy. */
function _watchComfyUiStartup(sup, child) {
  const watch = _openStartupWatch(sup);
  _pollComfyUiStartup(sup, child, watch, COMFYUI_STARTUP_HARD_CEILING_MS).catch(() => watch.finish(false));
}

async function _pollComfyUiStartup(sup, child, watch, hardCeilingMs) {
  const deadline = Date.now() + Math.min(hardCeilingMs, COMFYUI_STARTUP_HARD_CEILING_MS);
  let lastLogAt = Date.now();
  while (Date.now() < deadline && _isAlive(child)) {
    await _sleep(COMFYUI_STARTUP_POLL_MS);
    if (!_isAlive(child)) break;
    if (await probeHttp(`${sup.url}/system_stats`, COMFYUI_PROBE_TIMEOUT_MS)) {
      sup.state = "running";
      const elapsed = Math.round((Date.now() - sup.startedAt) / 1000);
      console.log(`[comfyui] 준비 완료 (${elapsed}초 경과)`);
      watch.finish(true);
      return true;
    }
    const now = Date.now();
    if (now - lastLogAt >= 10_000) {
      const elapsed = Math.round((now - sup.startedAt) / 1000);
      console.log(`[comfyui] 시작 중... (${elapsed}초 경과)`);
      lastLogAt = now;
    }
  }

  // Exited, or the ceiling passed while it is still loading (a slow custom-node load is normal):
  // the next ensureComfyUiRunning either restarts it or watches it again.
  watch.finish(false);
  return false;
}

function _isAlive(child) {
  return child.exitCode === null && child.signalCode === null;
}

// ─── Ollama ───────────────────────────────────────────────────────────────────

export async function initOllama(dataDir) {
  if (process.env.DYNAMICCHAT_START_OLLAMA === "0") {
    console.log("[local-services] DYNAMICCHAT_START_OLLAMA=0: Ollama 자동 시작 안 함");
    return;
  }
  const logFile = path.join(dataDir, "logs", "ollama.log");
  _ollamaSup = new ServiceSupervisor("ollama", _ollamaUrl, logFile);

  if (await probeHttp(_ollamaUrl, OLLAMA_PROBE_TIMEOUT_MS)) {
    console.log(`[ollama] 이미 실행 중 (${_ollamaUrl})`);
    _ollamaSup.state = "running";
    _ollamaSup.managed = false;
    return;
  }

  const binary = _findOllamaBinary();
  if (!binary) {
    _ollamaSup.state = "not_installed";
    // Not an error — NovelAI-only setups never need Ollama.
    return;
  }

  _startOllama(_ollamaSup, binary).catch(() => {});
}

function _findOllamaBinary() {
  if (process.platform === "win32") {
    const localAppData = process.env.LOCALAPPDATA;
    if (localAppData) {
      const p = path.join(localAppData, "Programs", "Ollama", "ollama.exe");
      if (existsSync(p)) return p;
    }
  }
  // On non-Windows, Ollama is typically in PATH, but we cannot check PATH cheaply without
  // spawning a shell. Return undefined to skip auto-start on platforms where we have no known
  // install location. The tray app already manages Ollama on macOS and Windows.
  return undefined;
}

async function _startOllama(sup, binary) {
  sup.state = "starting";
  sup.managed = true;
  sup.startedAt = Date.now();

  console.log("[ollama] 시작 중");

  await mkdir(path.dirname(sup.logFile), { recursive: true }).catch(() => {});

  let logFd;
  try {
    logFd = openSync(sup.logFile, "w");
  } catch {}

  let child;
  try {
    child = spawn(binary, ["serve"], {
      stdio: ["ignore", logFd ?? "ignore", logFd ?? "ignore"]
    });
  } catch (error) {
    try {
      if (logFd !== undefined) closeSync(logFd);
    } catch {}
    sup.state = "failed";
    sup.managed = false;
    sup.message = `Ollama를 시작할 수 없습니다: ${error.message}`;
    return;
  }
  try {
    if (logFd !== undefined) closeSync(logFd);
  } catch {}

  sup.child = child;
  child.once("exit", (code) => {
    if (sup.child === child) {
      sup.child = undefined;
      if (sup.state === "starting" || sup.state === "running") {
        sup.state = "failed";
        console.log(`[ollama] 종료 (코드 ${code ?? 0})`);
      }
    }
  });

  // Background poll — non-blocking, does not block initOllama's caller.
  _pollOllamaHealth(sup).catch(() => {});
}

async function _pollOllamaHealth(sup) {
  const deadline = Date.now() + OLLAMA_STARTUP_MAX_MS;
  while (Date.now() < deadline && sup.state === "starting") {
    await _sleep(OLLAMA_STARTUP_POLL_MS);
    if (await probeHttp(_ollamaUrl, OLLAMA_PROBE_TIMEOUT_MS)) {
      sup.state = "running";
      const elapsed = Math.round((Date.now() - (sup.startedAt ?? Date.now())) / 1000);
      console.log(`[ollama] 준비 완료 (${elapsed}초 경과)`);
      return;
    }
  }
}

// ─── combined init / status / shutdown ───────────────────────────────────────

/**
 * Called at API server startup. Discovers and starts ComfyUI and Ollama in the background —
 * this function itself awaits discovery (cheap) then fires startup non-blocking.
 */
export async function initLocalServices(dataDir) {
  await Promise.all([
    initComfyUi(dataDir).catch((e) => console.warn("[local-services] ComfyUI init 오류:", e?.message)),
    initOllama(dataDir).catch((e) => console.warn("[local-services] Ollama init 오류:", e?.message))
  ]);
}

export function getLocalServicesStatus() {
  const comfyui = _comfyUiSup
    ? _comfyUiSup.status()
    : { state: "stopped", managed: false, url: _comfyUiManagedUrl };
  const ollama = _ollamaSup
    ? _ollamaSup.status()
    : { state: "stopped", managed: false, url: _ollamaUrl };
  return { comfyui, ollama };
}

/** Kill managed children (ComfyUI, Ollama). Called from SIGINT/SIGTERM handlers. */
export function shutdownLocalServices() {
  for (const sup of [_comfyUiSup, _ollamaSup]) {
    if (sup?.child && sup.managed) {
      killTree(sup.child);
    }
  }
}

// ─── utilities ───────────────────────────────────────────────────────────────

function _sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function _readLastLines(file, n) {
  try {
    return readFileSync(file, "utf8")
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(-n)
      .join("\n");
  } catch {
    return undefined;
  }
}
