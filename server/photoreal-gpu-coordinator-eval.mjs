/**
 * Local GPU hand-over between a local Ollama and the local photoreal ComfyUI (src/services/localGpuCoordinator.ts).
 *
 * The VRAM belongs to the machine, while photoreal settings belong to a simulation. These checks pin the cases the
 * per-state coordination got wrong:
 *
 *   - a turn in a simulation with photoreal OFF still frees the ComfyUI an earlier render left resident (through the
 *     relay directly, never through the client helper whose photoreal-enabled guard reads the turn's own state),
 *     and the next render unloads Ollama again;
 *   - a local-Ollama turn that is not coordinated (hand-over off) still records that Ollama loaded a model, and
 *     leaves ComfyUI marked resident so the next coordinated turn frees it;
 *   - the same on the Ollama side: a render unloads the models the turns before it actually ran on — another
 *     simulation's model, or any model at all when the render's own simulation uses a hosted LLM — not only the
 *     ones its own settings name;
 *   - a render whose cut is no longer wanted is skipped when its turn on the GPU comes, BEFORE Ollama is unloaded
 *     for it (`shouldRun`), so a cancelled cut never evicts the LLM it waited behind.
 *
 * No servers: every request goes through a recording fetch stub. Run: node server/photoreal-gpu-coordinator-eval.mjs
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");

const RELAY_A = "http://127.0.0.1:65531/photoreal/comfyui";
const COMFYUI_A = "http://127.0.0.1:8188";
const OLLAMA = "http://127.0.0.1:11434";
// What getPhotorealRelayBase falls back to in Node when a simulation sets no relay address.
const DEFAULT_RELAY = "http://127.0.0.1:8788/photoreal/comfyui";

const checks = [];
const vite = await createServer({
  root: workspaceRoot,
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom"
});

try {
  const gpu = await vite.ssrLoadModule("/src/services/localGpuCoordinator.ts");
  await runGroup("machine-keyed hand-back", () => evaluatePhotorealOffTurnFreesResidentComfyUi(gpu));
  await runGroup("machine-keyed hand-over", () => evaluateRenderUnloadsResidentOllama(gpu));
  await runGroup("lock", () => evaluateHandBackWaitsForRender(gpu));
  await runGroup("overlap", () => evaluateNothingRunsBesideALocalTask(gpu));
  await runGroup("uncoordinated turns", () => evaluateUncoordinatedTurns(gpu));
  await runGroup("render bookkeeping", () => evaluateRenderBookkeeping(gpu));
  await runGroup("shouldRun", () => evaluateShouldRun(gpu));
  await runGroup("free failures", () => evaluateFreeFailures(gpu));
  await runGroup("helpers", () => evaluateHelpers(gpu));
} catch (error) {
  assertCheck("setup", false, `eval setup failed: ${error instanceof Error ? `${error.message}\n${error.stack}` : String(error)}`);
} finally {
  await vite.close();
}

for (const check of checks) {
  console.log(`${check.passed ? "PASS" : "FAIL"} ${check.scope}: ${check.message}`);
}
const failures = checks.filter((check) => !check.passed);
console.log("");
console.log(`GPU coordinator checks: ${checks.length - failures.length} passed, ${failures.length} failures`);
if (failures.length > 0) {
  process.exitCode = 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------------------------------------------

/** The review's scenario: A renders, B (photoreal off) takes a turn, A renders again. */
async function evaluatePhotorealOffTurnFreesResidentComfyUi(gpu) {
  const scope = "machine-keyed hand-back";
  const stub = createFetchStub();
  const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
  const simA = createState();
  const simB = createState({ photoreal: { enabled: false } });
  assertCheck(scope, gpu.shouldCoordinateLocalGpu(simB) === false, "B on its own does not coordinate (photoreal off)");

  await coordinator.run("llm", simA, task(stub, "turn A"));
  await coordinator.run("image", simA, task(stub, "render A"));
  assertCheck(scope, coordinator.holder === "image", "after A's render the GPU holder is the image model", coordinator.holder);

  stub.clear();
  const answer = await coordinator.run("llm", simB, task(stub, "turn B"));
  const frees = stub.calls.filter((call) => call.kind === "free");
  assertCheck(scope, answer === "turn B", "B's turn runs");
  assertCheck(
    scope,
    sequence(stub) === "free,start:turn B,end:turn B",
    "B's turn frees the ComfyUI A's render left resident before it starts, although B has photoreal off",
    sequence(stub)
  );
  assertCheck(
    scope,
    frees.length === 1 && frees[0].url === `${RELAY_A}/free` && frees[0].body?.baseUrl === COMFYUI_A,
    "the free goes to the relay and ComfyUI that rendered (A's), not to anything B's settings name",
    JSON.stringify(frees)
  );
  assertCheck(scope, coordinator.holder === "llm", "after B's turn the holder is the LLM", coordinator.holder);

  stub.clear();
  await coordinator.run("llm", simB, task(stub, "turn B2"));
  assertCheck(scope, !stub.calls.some((call) => call.kind === "free"), "a freed ComfyUI is not freed again on the next turn", sequence(stub));

  stub.clear();
  await coordinator.run("image", simA, task(stub, "render A2"));
  const unloads = stub.calls.filter((call) => call.kind === "unload");
  assertCheck(
    scope,
    sequence(stub) === "unload,start:render A2,end:render A2" && unloads[0]?.model === "qwen3:14b" && unloads[0]?.keepAlive === 0,
    "back in A, the next render unloads the Ollama model B's turn loaded (it used to skip this: holder was still 'image')",
    sequence(stub)
  );
}

/**
 * The mirror image on the Ollama side: the model a turn left on the card is unloaded by the next render whatever
 * that render's own simulation names — another model, or a hosted LLM (nothing at all).
 */
async function evaluateRenderUnloadsResidentOllama(gpu) {
  const scope = "machine-keyed hand-over";
  const simA = createState();
  const hostedLlm = createState({ llm: { provider: "deepseek", baseUrl: "https://api.deepseek.com", model: "deepseek-chat" } });
  const otherModel = createState({ photoreal: { enabled: false }, llm: { model: "gemma3:12b" } });
  const handoffOff = createState({ photoreal: { vramHandoff: "off" } });

  // A's turn loads qwen3:14b; the next render belongs to a simulation whose LLM is hosted.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    assertCheck(scope, gpu.shouldCoordinateLocalGpu(hostedLlm) === false, "the hosted-LLM simulation does not coordinate on its own");
    await coordinator.run("llm", simA, task(stub, "turn A"));
    stub.clear();
    await coordinator.run("image", hostedLlm, task(stub, "render (hosted LLM)"));
    const unloads = stub.calls.filter((call) => call.kind === "unload");
    assertCheck(
      scope,
      sequence(stub) === "unload,start:render (hosted LLM),end:render (hosted LLM)" &&
        unloads.length === 1 &&
        unloads[0].model === "qwen3:14b" &&
        unloads[0].url === `${OLLAMA}/api/generate`,
      "a render whose own LLM is hosted still unloads the Ollama model A's turn left on the card",
      sequence(stub)
    );
    assertCheck(scope, coordinator.holder === "image", "…and holds the card afterwards", coordinator.holder);

    stub.clear();
    await coordinator.run("image", hostedLlm, task(stub, "second render"));
    assertCheck(scope, sequence(stub) === "start:second render,end:second render", "an unloaded model is not unloaded again by the next render", sequence(stub));
  }

  // The turn ran in another simulation on another model; the render's simulation names qwen3:14b.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("image", simA, task(stub, "warm-up render"));
    stub.clear();
    await coordinator.run("llm", otherModel, task(stub, "turn on gemma"));
    stub.clear();
    await coordinator.run("image", simA, task(stub, "render A"));
    const models = stub.calls.filter((call) => call.kind === "unload").map((call) => call.model);
    assertCheck(
      scope,
      models.join(",") === "gemma3:12b,qwen3:14b" && sequence(stub) === "unload,unload,start:render A,end:render A",
      "the render unloads the model the last turn really ran on (gemma3:12b) as well as its own (qwen3:14b), before it starts",
      `${models.join(",")} / ${sequence(stub)}`
    );
  }

  // Hand-over off left the model loaded through a render of its own; the next coordinated render still unloads it.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("llm", handoffOff, task(stub, "turn (hand-over off)"));
    await coordinator.run("image", handoffOff, task(stub, "render (hand-over off)"));
    assertCheck(
      scope,
      sequence(stub) === "start:turn (hand-over off),end:turn (hand-over off),start:render (hand-over off),end:render (hand-over off)",
      "with the hand-over off, a render after a local turn unloads nothing (the user said both fit)",
      sequence(stub)
    );
    assertCheck(scope, coordinator.holder === "image", "…and is still recorded as the holder", coordinator.holder);
    stub.clear();
    await coordinator.run("image", simA, task(stub, "render A"));
    assertCheck(
      scope,
      sequence(stub) === "unload,start:render A,end:render A",
      "the next render whose simulation hands over unloads that model, although the last task seen was a render",
      sequence(stub)
    );
  }

  // Nothing local was ever seen: a hosted-LLM render stays a plain pass-through.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("llm", hostedLlm, task(stub, "hosted turn"));
    await coordinator.run("image", hostedLlm, task(stub, "render"));
    assertCheck(
      scope,
      sequence(stub) === "start:hosted turn,end:hosted turn,start:render,end:render",
      "a hosted turn leaves no model to unload, so the render after it unloads nothing",
      sequence(stub)
    );
  }

  // The render waits for a turn that is still running before it unloads that turn's model.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("llm", simA, task(stub, "first turn"));
    stub.clear();
    const slowTurn = coordinator.run("llm", simA, task(stub, "slow turn", 60));
    await delay(5);
    const render = coordinator.run("image", hostedLlm, task(stub, "render (hosted LLM)"));
    await Promise.all([slowTurn, render]);
    assertCheck(
      scope,
      sequence(stub) === "start:slow turn,end:slow turn,unload,start:render (hosted LLM),end:render (hosted LLM)",
      "a hosted-LLM render queues behind the running local turn, then unloads its model — never beside it",
      sequence(stub)
    );
  }
}

/** The hand-back is coordinated work: B's turn queues behind a render still running, then frees it. */
async function evaluateHandBackWaitsForRender(gpu) {
  const scope = "lock";
  const stub = createFetchStub();
  const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
  const simA = createState();
  const simB = createState({ photoreal: { enabled: false } });

  await coordinator.run("image", simA, task(stub, "warm-up render"));
  stub.clear();
  const [render, turn] = await Promise.all([
    coordinator.run("image", simA, task(stub, "slow render", 60)),
    coordinator.run("llm", simB, task(stub, "turn B"))
  ]);
  assertCheck(
    scope,
    render === "slow render" && turn === "turn B" && sequence(stub) === "start:slow render,end:slow render,free,start:turn B,end:turn B",
    "a photoreal-off turn waits for the running render, then frees ComfyUI, then runs — never beside it",
    sequence(stub)
  );
}

/**
 * The verifier's scenario: nothing is resident yet, simulation B (photoreal off, local Ollama) is mid-turn —
 * auto-progress keeps running while the user is in 제작 — and a 실사 profile test starts. Deciding "coordinate?" by
 * what was resident let the turn run outside the lock, so the render found the lock free, unloaded the turn's model
 * mid-reply and rendered beside its reload (15.0-15.8 of 16.4 GB in use in the browser repro).
 */
async function evaluateNothingRunsBesideALocalTask(gpu) {
  const scope = "overlap";
  const simB = createState({ photoreal: { enabled: false } });
  const studio = createState();
  const hosted = createState({ llm: { provider: "deepseek", baseUrl: "https://api.deepseek.com", model: "deepseek-chat" } });

  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    const turn = coordinator.run("llm", simB, task(stub, "turn B", 60));
    await delay(10);
    await Promise.all([turn, coordinator.run("image", studio, task(stub, "studio render"))]);
    assertCheck(
      scope,
      sequence(stub) === "start:turn B,end:turn B,unload,start:studio render,end:studio render",
      "a render started during a photoreal-off simulation's local turn waits for it, then unloads, then renders",
      sequence(stub)
    );
  }

  // The mirror: a render in a simulation whose LLM is hosted, then a local turn from another simulation.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    const render = coordinator.run("image", hosted, task(stub, "render (hosted LLM)", 60));
    await delay(10);
    await Promise.all([render, coordinator.run("llm", simB, task(stub, "turn B"))]);
    assertCheck(
      scope,
      sequence(stub) === "start:render (hosted LLM),end:render (hosted LLM),free,start:turn B,end:turn B",
      "a local turn started during a hosted-LLM simulation's render waits for it, then frees ComfyUI, then runs",
      sequence(stub)
    );
  }

  // A fresh coordinator (a page reload) cannot know what an earlier session left on the card. Before its first task
  // the ComfyUI the state renders with is freed, the mirror of a first render unloading Ollama: measured, a 16k
  // qwen3:14b next to a leftover ComfyUI model timed out its annotation call after 157 s at 15.6 of 16.4 GB.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("llm", studio, task(stub, "first turn"));
    await coordinator.run("llm", studio, task(stub, "second turn"));
    assertCheck(
      scope,
      sequence(stub) === "free,start:first turn,end:first turn,start:second turn,end:second turn",
      "the first local turn frees the state's ComfyUI once; later turns with nothing rendered free nothing",
      sequence(stub)
    );
  }

  // Queuing a render whose simulation names no local model must not invent a hand-over to report.
  {
    const stub = createFetchStub();
    const events = [];
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch, onHandoff: (event) => events.push(event) });
    await coordinator.run("image", hosted, task(stub, "first render"));
    assertCheck(
      scope,
      sequence(stub) === "start:first render,end:first render" && events.length === 0,
      "a first render with a hosted LLM makes no request and posts no hand-over notice",
      `${sequence(stub)} events=${events.length}`
    );
  }
}

async function evaluateUncoordinatedTurns(gpu) {
  const scope = "uncoordinated turns";
  const simA = createState();
  const handoffOff = createState({ photoreal: { vramHandoff: "off" } });
  const hosted = createState({ llm: { provider: "deepseek", baseUrl: "https://api.deepseek.com", model: "deepseek-chat" } });

  // Hand-over off in this simulation: the user says both models fit, so nothing is unloaded for this turn — but
  // Ollama DID load its model, and ComfyUI is still resident.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("image", simA, task(stub, "render A"));
    stub.clear();
    await coordinator.run("llm", handoffOff, task(stub, "turn (hand-over off)"));
    assertCheck(scope, sequence(stub) === "start:turn (hand-over off),end:turn (hand-over off)", "a turn whose simulation turned the hand-over off frees nothing", sequence(stub));
    assertCheck(scope, coordinator.holder === "llm", "…but still records that Ollama loaded a model", coordinator.holder);

    stub.clear();
    await coordinator.run("llm", simA, task(stub, "turn A"));
    assertCheck(
      scope,
      sequence(stub) === "free,start:turn A,end:turn A",
      "the next coordinated turn still frees the ComfyUI that stayed resident (both were loaded at once)",
      sequence(stub)
    );

    stub.clear();
    await coordinator.run("llm", handoffOff, task(stub, "turn (hand-over off) 2"));
    await coordinator.run("image", simA, task(stub, "render A2"));
    assertCheck(
      scope,
      sequence(stub) === "start:turn (hand-over off) 2,end:turn (hand-over off) 2,unload,start:render A2,end:render A2",
      "after an uncoordinated local-Ollama turn the next render unloads Ollama",
      sequence(stub)
    );
  }

  // A fresh page: no render seen yet, photoreal off. Nothing to free; the turn is still noted.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("llm", createState({ photoreal: { enabled: false } }), task(stub, "first turn"));
    assertCheck(scope, sequence(stub) === "start:first turn,end:first turn" && coordinator.holder === "llm", "a first local turn with photoreal off makes no request and records the LLM", `${sequence(stub)} holder=${coordinator.holder}`);
  }

  // A hosted LLM does not load anything on this card.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("image", simA, task(stub, "render A"));
    stub.clear();
    await coordinator.run("llm", hosted, task(stub, "hosted turn"));
    assertCheck(
      scope,
      sequence(stub) === "start:hosted turn,end:hosted turn" && coordinator.holder === "image",
      "a hosted-LLM turn frees nothing and leaves the holder alone",
      `${sequence(stub)} holder=${coordinator.holder}`
    );
    stub.clear();
    await coordinator.run("llm", simA, task(stub, "turn A"));
    assertCheck(scope, sequence(stub) === "free,start:turn A,end:turn A", "the ComfyUI stays marked resident across it", sequence(stub));
  }
}

async function evaluateRenderBookkeeping(gpu) {
  const scope = "render bookkeeping";

  // A render while the LLM is hosted is not coordinated (nothing to hand over), but its model is on this card.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    const hostedLlm = createState({ llm: { provider: "deepseek", baseUrl: "https://api.deepseek.com", model: "deepseek-chat" } });
    await coordinator.run("image", hostedLlm, task(stub, "render (hosted LLM)"));
    assertCheck(
      scope,
      sequence(stub) === "start:render (hosted LLM),end:render (hosted LLM)" && coordinator.holder === "image",
      "a render with a hosted LLM unloads nothing but records the image model",
      `${sequence(stub)} holder=${coordinator.holder}`
    );
    stub.clear();
    await coordinator.run("llm", createState(), task(stub, "turn after switching to Ollama"));
    assertCheck(
      scope,
      sequence(stub) === "free,start:turn after switching to Ollama,end:turn after switching to Ollama",
      "switching the LLM to a local Ollama frees the ComfyUI that render loaded",
      sequence(stub)
    );
  }

  // A remote ComfyUI and mock mode do not touch this card.
  for (const [label, state] of [
    ["remote ComfyUI", createState({ photoreal: { baseUrl: "http://192.168.0.10:8188" } })],
    ["mock mode", createState({ photoreal: { requestMode: "mock" } })],
    ["photoreal off", createState({ photoreal: { enabled: false } })]
  ]) {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("image", state, task(stub, "render"));
    stub.clear();
    await coordinator.run("llm", createState({ photoreal: { enabled: false } }), task(stub, "turn"));
    assertCheck(
      scope,
      sequence(stub) === "start:turn,end:turn",
      `${label}: no ComfyUI on this card is remembered, so the next turn frees nothing`,
      sequence(stub)
    );
  }

  // No relay address in the simulation: the render is remembered with the default relay of the configured API base.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("image", createState({ photoreal: { proxyUrl: "", baseUrl: "" } }), task(stub, "render"));
    stub.clear();
    await coordinator.run("llm", createState({ photoreal: { enabled: false } }), task(stub, "turn"));
    const free = stub.calls.find((call) => call.kind === "free");
    assertCheck(
      scope,
      free?.url === `${DEFAULT_RELAY}/free` && free.body && !("baseUrl" in free.body),
      "an empty relay address frees through the default relay; an empty ComfyUI URL leaves the relay its default",
      JSON.stringify(free)
    );
  }
}

async function evaluateShouldRun(gpu) {
  const scope = "shouldRun";
  const simA = createState();

  // The render is queued behind a slow turn and its cut is cancelled while it waits.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    await coordinator.run("llm", simA, task(stub, "first turn"));
    stub.clear();
    let wanted = true;
    let askedAfter = "";
    const slowTurn = coordinator.run("llm", simA, task(stub, "slow turn", 60));
    const render = coordinator.run("image", simA, task(stub, "cancelled render"), {
      shouldRun: () => {
        askedAfter = stub.calls.at(-1)?.label ?? stub.calls.at(-1)?.kind ?? "";
        return wanted;
      },
      onSkip: () => "skipped"
    });
    await delay(15);
    wanted = false;
    const [turnResult, renderResult] = await Promise.all([slowTurn, render]);
    assertCheck(scope, turnResult === "slow turn" && renderResult === "skipped", "a render no longer wanted when its turn comes resolves with onSkip", `${turnResult} / ${renderResult}`);
    assertCheck(scope, askedAfter === "end:slow turn", "shouldRun is asked only once the GPU is free for the render", askedAfter);
    assertCheck(
      scope,
      sequence(stub) === "start:slow turn,end:slow turn",
      "a skipped render neither unloads Ollama nor starts",
      sequence(stub)
    );
    assertCheck(scope, coordinator.holder === "llm", "a skipped render leaves the holder as it was", coordinator.holder);

    stub.clear();
    const next = await coordinator.run("image", simA, task(stub, "next render"));
    assertCheck(scope, next === "next render" && sequence(stub) === "unload,start:next render,end:next render", "the lock moves on to the next task after a skip", sequence(stub));
  }

  // Without onSkip a skip is an AbortError.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    const outcome = await coordinator
      .run("image", simA, task(stub, "never"), { shouldRun: () => false })
      .then(() => "resolved", (error) => (error && error.name) || String(error));
    assertCheck(scope, outcome === "AbortError" && stub.calls.length === 0, "a skip without onSkip rejects with an AbortError and makes no request", `${outcome} ${sequence(stub)}`);
  }

  // Honoured on the uncoordinated path too.
  {
    const stub = createFetchStub();
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch });
    const hosted = createState({ llm: { provider: "deepseek", baseUrl: "https://api.deepseek.com", model: "deepseek-chat" } });
    const outcome = await coordinator.run("image", hosted, task(stub, "never"), { shouldRun: () => false, onSkip: () => "skipped" });
    assertCheck(
      scope,
      outcome === "skipped" && stub.calls.length === 0 && coordinator.holder === undefined,
      "an uncoordinated render is skipped the same way, without being recorded",
      `${outcome} ${sequence(stub)} holder=${coordinator.holder}`
    );
    const ran = await coordinator.run("image", hosted, task(stub, "ran"), { shouldRun: () => true, onSkip: () => "skipped" });
    assertCheck(scope, ran === "ran", "shouldRun true runs the task");
  }
}

async function evaluateFreeFailures(gpu) {
  const scope = "free failures";
  const simA = createState();
  const simB = createState({ photoreal: { enabled: false } });
  const stub = createFetchStub({ free: "throw" });
  const handoffs = [];
  const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: stub.fetch, onHandoff: (event) => handoffs.push(event) });
  await coordinator.run("image", simA, task(stub, "render A"));
  stub.clear();
  const answer = await coordinator.run("llm", simB, task(stub, "turn B"));
  assertCheck(scope, answer === "turn B" && sequence(stub) === "free,start:turn B,end:turn B", "an unreachable relay does not stop the turn", sequence(stub));
  const last = handoffs.at(-1);
  assertCheck(
    scope,
    last?.from === "image" && last?.to === "llm" && typeof last.detail === "string" && !last.detail.startsWith("ComfyUI 모델을 VRAM에서 내렸습니다"),
    "the hand-over notice does not claim a free that failed",
    JSON.stringify(last)
  );
  stub.clear();
  await coordinator.run("llm", simB, task(stub, "turn B2"));
  assertCheck(scope, sequence(stub) === "start:turn B2,end:turn B2", "a failed free is not retried on every later turn", sequence(stub));
}

async function evaluateHelpers(gpu) {
  const scope = "helpers";
  const target = gpu.readLocalComfyUiTarget(createState());
  assertCheck(scope, target?.relayBase === RELAY_A && target?.baseUrl === COMFYUI_A, "readLocalComfyUiTarget names the relay and the ComfyUI", JSON.stringify(target));
  assertCheck(scope, gpu.readLocalComfyUiTarget(createState({ photoreal: { enabled: false } })) === undefined, "no target with photoreal off");
  assertCheck(scope, gpu.readLocalComfyUiTarget(createState({ photoreal: { requestMode: "mock" } })) === undefined, "no target in mock mode");
  assertCheck(scope, gpu.readLocalComfyUiTarget(createState({ photoreal: { baseUrl: "http://10.0.0.5:8188" } })) === undefined, "no target for a ComfyUI on another machine");
  assertCheck(
    scope,
    gpu.shouldCoordinateLocalGpu(createState({ photoreal: { proxyUrl: undefined } })) === true,
    "shouldCoordinateLocalGpu does not depend on the relay field (an unhydrated state still coordinates)"
  );

  const stub = createFetchStub();
  const ok = await gpu.freeLocalComfyUi({ relayBase: RELAY_A, baseUrl: COMFYUI_A }, stub.fetch);
  assertCheck(scope, ok === true && stub.calls[0]?.url === `${RELAY_A}/free` && stub.calls[0]?.method === "POST", "freeLocalComfyUi posts to <relay>/free", JSON.stringify(stub.calls));
  const failing = createFetchStub({ free: "error" });
  assertCheck(scope, (await gpu.freeLocalComfyUi({ relayBase: RELAY_A }, failing.fetch)) === false, "an error answer is reported as not freed");
  const throwing = createFetchStub({ free: "throw" });
  assertCheck(scope, (await gpu.freeLocalComfyUi({ relayBase: RELAY_A }, throwing.fetch)) === false, "a network failure is reported as not freed, without throwing");
}

// ---------------------------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------------------------

/** The fields the coordinator reads, with a local Ollama and a local ComfyUI reached through relay A. */
function createState(overrides = {}) {
  return {
    simulation: { id: "sim_gpu_eval" },
    photoreal: {
      enabled: true,
      provider: "comfyui",
      requestMode: "proxy",
      baseUrl: COMFYUI_A,
      proxyUrl: RELAY_A,
      vramHandoff: "auto",
      ...overrides.photoreal
    },
    llm: { enabled: true, provider: "ollama", baseUrl: `${OLLAMA}/v1`, model: "qwen3:14b", ...overrides.llm },
    imageTagLlm: { enabled: false, provider: "ollama", baseUrl: `${OLLAMA}/v1`, model: "qwen3:4b", ...overrides.imageTagLlm }
  };
}

function createFetchStub(behaviour = {}) {
  const calls = [];
  const fetchImpl = async (input, init = {}) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const body = typeof init.body === "string" ? safeJson(init.body) : undefined;
    if (url.endsWith("/api/generate")) {
      calls.push({ kind: "unload", url, model: body?.model, keepAlive: body?.keep_alive, method: init.method });
      return new Response(JSON.stringify({ done: true, done_reason: "unload" }), { status: 200 });
    }
    if (url.endsWith("/free")) {
      calls.push({ kind: "free", url, body, method: init.method });
      if (behaviour.free === "throw") {
        throw new TypeError("fetch failed");
      }
      return new Response("{}", { status: behaviour.free === "error" ? 502 : 200 });
    }
    throw new TypeError(`unexpected fetch ${url}`);
  };
  return {
    calls,
    fetch: fetchImpl,
    clear() {
      calls.length = 0;
    }
  };
}

/** A task that records its start and end in the stub's call log, so hand-overs and tasks share one timeline. */
function task(stub, label, ms = 0) {
  return async () => {
    stub.calls.push({ kind: "start", label: `start:${label}` });
    await delay(ms);
    stub.calls.push({ kind: "end", label: `end:${label}` });
    return label;
  };
}

function sequence(stub) {
  return stub.calls.map((call) => call.label ?? call.kind).join(",");
}

async function runGroup(scope, run) {
  try {
    await run();
  } catch (error) {
    assertCheck(scope, false, `threw: ${error instanceof Error ? `${error.message}\n${error.stack}` : String(error)}`);
  }
}

function assertCheck(scope, passed, message, detail) {
  checks.push({ scope, passed: Boolean(passed), message: passed || detail === undefined ? message : `${message} — ${detail}` });
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}
