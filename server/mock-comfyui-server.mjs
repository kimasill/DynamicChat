/**
 * Standalone CLI entry point for mock-comfyui.mjs.
 *
 * Used by local-services-eval.mjs as the injectable spawn target so the ComfyUI supervisor
 * can be tested without a real Python/ComfyUI install.
 *
 * Usage: node mock-comfyui-server.mjs [--port N] [--behaviour NAME] [--delay-ready-ms N]
 *
 * Prints "READY:<port>" to stdout when listening.
 * --delay-ready-ms N: wait N ms before starting the HTTP server (simulates slow startup).
 * --behaviour NAME: set the initial mock behaviour (ok, never_completes, upstream_down, …).
 */
import { startMockComfyUi, MOCK_COMFYUI_BEHAVIOURS } from "./mock-comfyui.mjs";

const argv = process.argv.slice(2);

function argValue(flag) {
  const idx = argv.indexOf(flag);
  return idx >= 0 ? argv[idx + 1] : undefined;
}

const startPort = Number(argValue("--port") ?? 0) || 0;
const behaviour = argValue("--behaviour") ?? "ok";
const delayMs = Number(argValue("--delay-ready-ms") ?? 0) || 0;

if (delayMs > 0) {
  await new Promise((r) => setTimeout(r, delayMs));
}

if (behaviour !== "ok" && !MOCK_COMFYUI_BEHAVIOURS.includes(behaviour)) {
  process.stderr.write(`Unknown behaviour: ${behaviour}\n`);
  process.exit(1);
}

const mock = await startMockComfyUi({ port: startPort });
if (behaviour !== "ok") mock.setBehaviour(behaviour);

process.stdout.write(`READY:${mock.port}\n`);

async function stop() {
  await mock.close();
  process.exit(0);
}
process.once("SIGTERM", stop);
process.once("SIGINT", stop);
