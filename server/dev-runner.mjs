import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const viteBin = path.join(workspaceRoot, "node_modules", "vite", "bin", "vite.js");
const neuralMapRoot = path.resolve(process.env.NEURALMAP_ROOT ?? path.join(workspaceRoot, "..", "NeuralMap"));
const neuralMapHealthUrl = process.env.NEURALMAP_HEALTH_URL ?? "http://127.0.0.1:4317/health";
const shouldStartNeuralMap = process.env.DYNAMICCHAT_START_NEURALMAP !== "0";

const children = [];

let shuttingDown = false;

registerChild(start("api", process.execPath, [path.join(workspaceRoot, "server", "dynamicchat-server.mjs")]));
await startNeuralMapIfNeeded();
registerChild(start("web", process.execPath, [viteBin, "--host", "127.0.0.1"]));

process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);

async function startNeuralMapIfNeeded() {
  if (!shouldStartNeuralMap) {
    console.log("[neuralmap] skipped by DYNAMICCHAT_START_NEURALMAP=0");
    return;
  }

  if (await isHealthy(neuralMapHealthUrl)) {
    console.log(`[neuralmap] existing API detected at ${neuralMapHealthUrl}`);
    return;
  }

  if (!existsSync(path.join(neuralMapRoot, "package.json"))) {
    console.log(`[neuralmap] sibling repo not found at ${neuralMapRoot}; using DynamicChat local fallback`);
    return;
  }

  // Default to the NeuralMap repo env so the API connects to its Postgres graph
  // DB (database mode) and runs precisely. Set DYNAMICCHAT_NEURALMAP_USE_REPO_ENV=0
  // to force the in-memory sample fallback (no DB required).
  const useRepoEnv = process.env.DYNAMICCHAT_NEURALMAP_USE_REPO_ENV !== "0";
  const env = { ...process.env };
  if (!useRepoEnv) {
    delete env.DATABASE_URL;
    delete env.NEURALMAP_DATABASE_ROUTES;
  }

  // Under WSL mirrored networking the WSL VM IP (172.16-31.x.x) is unreachable
  // from Windows -- the Postgres/Redis containers must be reached via localhost.
  // A stale user-level DATABASE_URL env var can override the repo .env, so
  // normalize any WSL-VM-IP host to 127.0.0.1 and pin it (static) here.
  if (useRepoEnv && process.platform === "win32") {
    const wslIpHost = /@172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+/u;
    for (const key of ["DATABASE_URL", "REDIS_URL"]) {
      if (env[key] && wslIpHost.test(env[key])) {
        env[key] = env[key].replace(wslIpHost, "@127.0.0.1");
      }
    }
    env.NEURALMAP_DATABASE_HOST_SOURCE = "static";
  }

  const pnpm = resolvePnpmInvocation();

  // Hold a WSL session open for the dev session. Without an attached wsl.exe,
  // WSL2 tears down its network relay seconds after the last session exits,
  // which silently drops Windows->WSL Postgres connectivity (NeuralMap then
  // falls back to sample data). This keepalive lives only as long as `pnpm dev`.
  // Disable with DYNAMICCHAT_WSL_KEEPALIVE=0.
  if (useRepoEnv && process.platform === "win32" && process.env.DYNAMICCHAT_WSL_KEEPALIVE !== "0") {
    const distro = process.env.NEURALMAP_WSL_DISTRO ?? "Ubuntu-24.04";
    registerChild(
      start("wsl-keepalive", "wsl.exe", ["-d", distro, "--exec", "sleep", "infinity"], { optional: true })
    );
  }

  // Best-effort: bring up the NeuralMap Postgres/Redis containers first so the
  // API has a graph DB to connect to. Non-fatal -- the API degrades gracefully
  // to local fallback if the DB never comes up. Disable with
  // DYNAMICCHAT_START_NEURALMAP_INFRA=0.
  if (useRepoEnv && process.env.DYNAMICCHAT_START_NEURALMAP_INFRA !== "0") {
    start("neuralmap-infra", pnpm.command, [...pnpm.args, "infra:up"], {
      cwd: neuralMapRoot,
      env,
      optional: true
    });
  }

  registerChild(
    start(
      "neuralmap",
      pnpm.command,
      [...pnpm.args, ...(useRepoEnv ? ["dev:api"] : ["--filter", "@neuralmap/api", "dev"])],
      {
        cwd: neuralMapRoot,
        env,
        optional: true
      }
    )
  );
}

function resolvePnpmInvocation() {
  if (process.platform !== "win32") {
    return { command: "pnpm", args: [] };
  }

  const appData = process.env.APPDATA;
  const pnpmCjs = appData ? path.join(appData, "npm", "node_modules", "pnpm", "bin", "pnpm.cjs") : undefined;
  if (pnpmCjs && existsSync(pnpmCjs)) {
    return { command: process.execPath, args: [pnpmCjs] };
  }

  return { command: "cmd.exe", args: ["/d", "/s", "/c", "pnpm"] };
}

async function isHealthy(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 800);
  try {
    const response = await fetch(url, { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeoutId);
  }
}

function start(label, command, args, options = {}) {
  let child;
  try {
    child = spawn(command, args, {
      cwd: options.cwd ?? workspaceRoot,
      env: options.env ?? process.env,
      stdio: ["inherit", "pipe", "pipe"]
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.log(`[${label}] failed to start: ${message}`);
    if (options.optional) {
      console.log(`[${label}] continuing with DynamicChat local fallback`);
      return undefined;
    }
    throw error;
  }

  child.stdout.on("data", (chunk) => writePrefixed(label, chunk));
  child.stderr.on("data", (chunk) => writePrefixed(label, chunk));
  child.once("exit", (code, signal) => {
    if (shuttingDown) {
      return;
    }

    const reason = signal ? `signal ${signal}` : `exit code ${code ?? 0}`;
    if (options.optional) {
      console.log(`[dev] optional ${label} process stopped with ${reason}; continuing with DynamicChat local fallback`);
      return;
    }

    shuttingDown = true;
    stopChildren();
    console.log(`[dev] ${child.spawnargs.join(" ")} stopped with ${reason}`);
    process.exit(code ?? (signal ? 1 : 0));
  });
  child.once("error", (error) => {
    console.log(`[${label}] failed to start: ${error.message}`);
    if (!options.optional && !shuttingDown) {
      shuttingDown = true;
      stopChildren();
      process.exit(1);
    }
  });
  return child;
}

function registerChild(child) {
  if (child) {
    children.push(child);
  }
}

function writePrefixed(label, chunk) {
  const text = chunk.toString();
  for (const line of text.split(/\r?\n/u)) {
    if (line.length > 0) {
      console.log(`[${label}] ${line}`);
    }
  }
}

function shutdown() {
  if (shuttingDown) {
    return;
  }

  shuttingDown = true;
  stopChildren();
}

function stopChildren() {
  for (const child of children) {
    if (!child.killed && child.exitCode === null) {
      child.kill();
    }
  }
}
