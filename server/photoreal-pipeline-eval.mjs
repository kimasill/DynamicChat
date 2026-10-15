/**
 * Photoreal pipeline evaluation (eval:photoreal).
 *
 * The routing eval proves the classifier and the router on hand-built inputs, and the prompt / client / relay
 * evals prove the translation and the transport. This one drives the seams between them through the real planner
 * and the real executor:
 *
 *   annotation cues -> planImageJobForCompletedTurn -> composed plan -> classifier -> router -> job (backend, routing)
 *                   -> reuse filter -> executeImageJob -> NovelAI | ComfyUI relay -> asset (backend, sceneCategory)
 *
 * plus the two services that only exist because of photoreal: the local-GPU hand-over (localGpuCoordinator) and the
 * creation screen's profile test generation (characterProfileStudio).
 *
 * No GPU, ComfyUI, NovelAI token or network: NovelAI, the object store and a stub LLM are answered by a fetch stub,
 * and the photoreal leg runs the real API server (spawned with a temp data dir and a free port) against
 * mock-comfyui.mjs, the same way comfyui-relay-eval.mjs does.
 *
 * Usage: node server/photoreal-pipeline-eval.mjs
 */
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { createMockPng, startMockComfyUi } from "./mock-comfyui.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const serverEntry = path.join(workspaceRoot, "server", "dynamicchat-server.mjs");

const SIMULATION_ID = "sim_photoreal_pipeline";
/** Where fetchObjectAsBase64 looks in Node (no localStorage → the default API base). Answered by the fetch stub. */
const DEFAULT_OBJECT_BASE = "http://127.0.0.1:8788/objects/";
const NOVELAI_ENDPOINT = "https://image.novelai.net/ai/generate-image";
const STUB_OLLAMA_ROOT = "http://127.0.0.1:65531";
const NOVELAI_DISABLED_TEXT = "NovelAI 연동이 꺼져 있어 이미지를 생성하지 않았습니다. 개인 설정에서 NovelAI를 활성화하세요.";
const NOVELAI_MOCK_TEXT = "NovelAI가 mock 모드입니다. 실제 이미지를 생성하려면 요청 모드를 proxy 또는 direct로 바꾸세요.";
const PHOTOREAL_DISABLED_TEXT = "실사 이미지(ComfyUI) 연동이 꺼져 있어 이미지를 생성하지 않았습니다. 설정에서 실사 이미지를 활성화하세요.";
const PHOTOREAL_MOCK_TEXT = "실사 이미지가 mock 모드입니다. 실제 이미지를 생성하려면 요청 모드를 proxy로 바꾸세요.";
const NOVELAI_SETUP_TEXT = "NovelAI 연동이 꺼져 있습니다. 개인 설정에서 켜면 이 컷부터 생성됩니다.";
const PHOTOREAL_SETUP_TEXT = "실사 이미지(ComfyUI) 연동이 꺼져 있습니다. 설정에서 켜면 이 컷부터 생성됩니다.";
// Read by the server at startup; inherited values from the developer's shell would change what is tested.
const SERVER_ENV_KEYS = [
  "DYNAMICCHAT_COMFYUI_URL",
  "DYNAMICCHAT_COMFYUI_HOSTS",
  "DYNAMICCHAT_COMFYUI_EXTRA_NODES",
  "DYNAMICCHAT_DENIED_ORIGINS",
  "DYNAMICCHAT_CORS_ORIGIN",
  "DYNAMICCHAT_MAX_BODY_BYTES",
  "DYNAMICCHAT_RATE_LIMIT_WINDOW_MS"
];

const FRAMES = {
  close_up: { shot: "close_up", viewpoint: "front", visibleRegions: ["head"] },
  upper_body: { shot: "upper_body", viewpoint: "front", visibleRegions: ["head", "torso"] },
  upper_body_behind: { shot: "upper_body", viewpoint: "from_behind", visibleRegions: ["head", "torso"] },
  hip_close_up: { shot: "close_up", viewpoint: "front", visibleRegions: ["hips"] },
  cowboy_shot: { shot: "cowboy_shot", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs"] },
  full_body: { shot: "full_body", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs", "feet"] },
  wide_shot: { shot: "wide_shot", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs", "feet"] }
};

const NAI_TOKEN_PATTERN = /masterpiece|best quality|very aesthetic|absurdres|\bartist:|\byear \d{4}\b|\brating:|score_\d|::|[{}[\]]/iu;
const HANGUL_PATTERN = /[가-힣]/u;

/** The routing fixture: one cut per category plus the two safety-floor locks, in sidecar order. */
const MIXED_CUES = [
  {
    label: "portrait",
    expected: "portrait",
    characters: ["char_yuna"],
    base: ["1girl", "solo", "indoors", "cafe"],
    prompts: [["char_yuna", "smile, looking at viewer, holding cup"]],
    frame: "upper_body",
    kind: "dialogue_face"
  },
  {
    label: "full_body",
    expected: "full_body",
    characters: ["char_yuna"],
    base: ["1girl", "solo", "outdoors", "street", "daytime"],
    prompts: [["char_yuna", "standing, hand in pocket"]],
    frame: "full_body",
    kind: "scene"
  },
  {
    label: "scenery",
    expected: "scenery",
    characters: [],
    base: ["no humans", "scenery", "cafe", "window", "sunlight"],
    prompts: [],
    frame: "wide_shot",
    kind: "scene"
  },
  {
    label: "object",
    expected: "object",
    characters: [],
    base: ["no humans", "coffee cup", "cake", "plate", "table", "close-up"],
    prompts: [],
    frame: "close_up",
    kind: "context"
  },
  {
    label: "group",
    expected: "group",
    characters: ["char_yuna", "char_minho"],
    base: ["1girl", "1boy", "indoors", "cafe"],
    prompts: [
      ["char_yuna", "smile, sitting"],
      ["char_minho", "sitting, talking"]
    ],
    frame: "upper_body",
    kind: "interaction"
  },
  {
    label: "action",
    expected: "action",
    characters: ["char_yuna"],
    base: ["1girl", "solo", "outdoors", "street"],
    prompts: [["char_yuna", "running, open mouth"]],
    frame: "full_body",
    kind: "action"
  },
  {
    label: "explicit",
    expected: "explicit",
    locked: true,
    characters: ["char_yuna", "char_minho"],
    base: ["1girl", "1boy", "bedroom"],
    prompts: [
      ["char_yuna", "nude, sex from behind, blush"],
      ["char_minho", "nude, sex from behind"]
    ],
    frame: "cowboy_shot",
    kind: "interaction"
  },
  {
    label: "school+suggestive",
    expected: "portrait",
    locked: true,
    characters: ["char_yuna"],
    base: ["1girl", "solo", "classroom"],
    // Authors its own posture on purpose. Cuts in one output are planned against a working state that carries
    // each earlier cut's pose forward (intra-output continuity), so without it this cut inherits the action
    // cut's "running" and is classified — correctly, from its composed tags — as action.
    prompts: [["char_yuna", "school uniform, cleavage, smile, sitting"]],
    frame: "upper_body",
    kind: "dialogue_face"
  }
];

const P = "photoreal";
const N = "novelai";
const ROUTING_EXPECTATIONS = [
  { label: "mode novelai", routing: { mode: "novelai" }, backends: [N, N, N, N, N, N, N, N] },
  { label: "mode mixed (default categories)", routing: { mode: "mixed" }, backends: [P, P, P, P, N, N, N, N] },
  {
    label: "mode mixed (group + action ticked)",
    routing: { mode: "mixed", photorealCategories: ["group", "action"] },
    backends: [N, N, N, N, P, P, N, N]
  },
  { label: "mode photoreal", routing: { mode: "photoreal" }, backends: [P, P, P, P, P, P, N, N] }
];

const checks = [];
const cleanups = [];
const realFetch = globalThis.fetch;

const vite = await createServer({
  root: workspaceRoot,
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom"
});

try {
  const modules = {
    seed: await vite.ssrLoadModule("/src/data/seed.ts"),
    orchestrator: await vite.ssrLoadModule("/src/services/imageOrchestrator.ts"),
    engine: await vite.ssrLoadModule("/src/services/simulationEngine.ts"),
    workflows: await vite.ssrLoadModule("/src/services/comfyWorkflows.ts"),
    gpu: await vite.ssrLoadModule("/src/services/localGpuCoordinator.ts"),
    studio: await vite.ssrLoadModule("/src/services/characterProfileStudio.ts"),
    photoPrompt: await vite.ssrLoadModule("/src/services/photorealPrompt.ts")
  };
  const fixtures = { referencePng: createMockPng({ width: 3, height: 4, rgb: [0x2a, 0x6f, 0x97] }), outputPng: createMockPng() };

  await runGroup("pipeline", () => evaluateMixedSidecars(modules));
  await runGroup("continuity", () => evaluateContinuityInjectedExplicit(modules));
  await runGroup("references", () => evaluateReferencePlanning(modules));
  await runGroup("floor", () => evaluatePhotoTextFloor(modules));
  await runGroup("reuse", () => evaluateReuseIsolation(modules));
  await runGroup("regenerate", () => evaluateRegeneration(modules));
  await runGroup("policy", () => evaluateCountLimitPolicy(modules));
  await runGroup("setup-block", () => evaluateSetupBlocks(modules));
  await runGroup("dry-run", () => evaluateDryRunExecution(modules));
  await runGroup("custom-workflow", () => evaluateCustomWorkflowFloor(modules));
  await runGroup("dispatch", () => evaluateDispatchFloor(modules));
  await runGroup("sent-graph", () => evaluateSentGraphFloor(modules));
  await runGroup("novelai-reference", () => evaluateNovelAiPreciseReference(modules, fixtures));
  await runGroup("gpu", () => evaluateLocalGpuCoordinator(modules));
  await runGroup("studio", () => evaluateProfileStudio(modules));
  await runGroup("trace", () => evaluateTurnTraceMetrics(modules));
  await runGroup("relay", () => evaluateRelayExecution(modules, fixtures));
} catch (error) {
  assertCheck("setup", false, `eval setup failed: ${describeError(error)}`);
} finally {
  globalThis.fetch = realFetch;
  for (const cleanup of cleanups.reverse()) {
    await Promise.resolve()
      .then(cleanup)
      .catch((error) => console.error(`cleanup failed: ${error?.message ?? error}`));
  }
  await vite.close();
}

for (const check of checks) {
  console.log(`${check.passed ? "PASS" : "FAIL"} ${check.scope}: ${check.message}`);
}
const failures = checks.filter((check) => !check.passed);
console.log("");
console.log(`Photoreal pipeline checks: ${checks.length - failures.length} passed, ${failures.length} failures`);
if (failures.length > 0) {
  process.exitCode = 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Planning through the turn pipeline
// ---------------------------------------------------------------------------------------------------------------

async function evaluateMixedSidecars({ seed, engine, workflows }) {
  for (const expectation of ROUTING_EXPECTATIONS) {
    const scope = `pipeline.${expectation.label}`;
    const state = createState(seed.seedState, { routing: expectation.routing });
    const plan = await planTurn(engine, state, MIXED_CUES.map(toDraft));
    const jobs = plan.imageJobs ?? [];
    const reused = plan.reusedAssetIds ?? [];

    assertCheck(scope, jobs.length + reused.length === MIXED_CUES.length, "every generating cue became a job or a reused asset", `jobs=${jobs.length} reused=${reused.length} cues=${MIXED_CUES.length}`);
    const cueIndexes = jobs.map((job) => job.providerPayload.cueIndex);
    assertCheck(scope, cueIndexes.every((value, index) => value === index), "cueIndex is the sidecar position, unchanged by routing", JSON.stringify(cueIndexes));

    const backends = jobs.map((job) => job.backend);
    assertCheck(scope, sameList(backends, expectation.backends), `backends per cue = [${expectation.backends.join(", ")}]`, `got [${backends.join(", ")}]`);
    const categories = jobs.map((job) => job.routing?.category);
    const expectedCategories = MIXED_CUES.map((cue) => cue.expected);
    assertCheck(scope, sameList(categories, expectedCategories), "the composed plan classifies into the expected category per cue", `got [${categories.join(", ")}]`);

    const malformed = jobs.filter(
      (job) =>
        !["novelai", "photoreal"].includes(job.backend) ||
        !job.routing ||
        job.routing.backend !== job.backend ||
        job.providerPayload.provider !== job.backend ||
        job.providerPayload.sceneCategory !== job.routing.category ||
        job.providerPayload.routing?.backend !== job.backend
    );
    assertCheck(scope, malformed.length === 0, "every job carries backend + routing, mirrored in the payload", malformed.map((job) => job.providerPayload.cueLabel).join(", "));

    const lockedIndexes = MIXED_CUES.flatMap((cue, index) => (cue.locked ? [index] : []));
    const lockedJobs = lockedIndexes.map((index) => jobs[index]).filter(Boolean);
    assertCheck(
      scope,
      lockedJobs.length === lockedIndexes.length && lockedJobs.every((job) => job.backend === "novelai" && job.routing.locked === true && /NovelAI 고정/u.test(job.routing.reason)),
      "explicit and school+suggestive cuts are locked to NovelAI with a reason",
      lockedJobs.map((job) => `${job.backend}/${job.routing?.locked}`).join(", ")
    );
    assertCheck(
      scope,
      jobs.every((job, index) => job.providerPayload.cue?.kind === MIXED_CUES[index].kind && job.routing.signals.includes(`kind:${MIXED_CUES[index].kind}`)),
      "the annotation kind travels into the job's cue and the routing signals"
    );

    const photoreal = jobs.filter((job) => job.backend === "photoreal");
    const size = workflows.resolvePhotorealOutputSize(state);
    const badPhotoreal = photoreal.filter(
      (job) =>
        job.providerPayload.promptFormat !== "natural-language" ||
        HANGUL_PATTERN.test(job.prompt) ||
        NAI_TOKEN_PATTERN.test(job.prompt) ||
        job.providerPayload.width !== size.width ||
        job.providerPayload.height !== size.height ||
        typeof job.providerPayload.novelAiPrompt !== "string" ||
        !job.providerPayload.photorealSections ||
        typeof job.providerPayload.presetId !== "string" ||
        "modelPreset" in job.providerPayload ||
        "promptGuidance" in job.providerPayload
    );
    assertCheck(scope, badPhotoreal.length === 0, "photoreal jobs carry an English photo prompt, the photo size, the preset and no NovelAI-only fields", badPhotoreal.map((job) => job.providerPayload.cueLabel).join(", "));
    const novelai = jobs.filter((job) => job.backend === "novelai");
    assertCheck(
      scope,
      novelai.every((job) => job.providerPayload.promptFormat === "novelai-tags" && job.providerPayload.width === state.imageProfile.width && typeof job.providerPayload.modelPreset === "string"),
      "NovelAI jobs keep the NovelAI tag plan and size"
    );
    if (expectation.routing.mode === "photoreal") {
      const portrait = jobs[0];
      const scenery = jobs[2];
      assertCheck(
        scope,
        portrait.providerPayload.presetId === "qwen_image_2_1_turbo" && portrait.prompt.includes("<image1>") && portrait.providerPayload.photorealReferences?.[0]?.objectKey === `${SIMULATION_ID}/profile_yuna.png`,
        "a cut with a profiled face uses the reference preset and names <image1>",
        `${portrait.providerPayload.presetId} / ${portrait.prompt.slice(0, 80)}`
      );
      assertCheck(
        scope,
        scenery.providerPayload.presetId === "z_image_turbo" && (scenery.providerPayload.photorealReferences ?? []).length === 0 && !/<image\d>/u.test(scenery.prompt),
        "a people-free cut uses the text-to-image preset with no references",
        scenery.providerPayload.presetId
      );
      const group = jobs[4];
      assertCheck(
        scope,
        group.providerPayload.photorealReferences?.length === 1 && group.providerPayload.photorealReferences[0].characterId === "char_yuna",
        "in a two-person cut only the character with a photoreal profile gets a reference slot",
        JSON.stringify(group.providerPayload.photorealReferences)
      );
    }
  }

  // Availability: photoreal preferred but switched off falls back to NovelAI, and says so.
  const fallbackState = createState(seed.seedState, { routing: { mode: "photoreal" }, photoreal: false });
  const fallbackPlan = await planTurn(engine, fallbackState, [toDraft(MIXED_CUES[0])]);
  const fallbackJob = fallbackPlan.imageJobs?.[0];
  assertCheck(
    "pipeline.availability",
    fallbackJob?.backend === "novelai" && fallbackJob.routing?.fallback === true,
    "photoreal mode with ComfyUI off routes to NovelAI as a marked fallback",
    `${fallbackJob?.backend} fallback=${fallbackJob?.routing?.fallback}`
  );
}

async function evaluateContinuityInjectedExplicit({ seed, engine }) {
  const scope = "continuity";
  const state = createState(seed.seedState, { routing: { mode: "photoreal" } });
  state.memoryEvents = [createStateMemoryEvent(state, "char_yuna", "InteractionTags", "sex from behind")];
  const closeUp = {
    label: "reaction close-up",
    characters: ["char_yuna"],
    base: ["close-up", "indoors"],
    prompts: [["char_yuna", "blush, open mouth"]],
    frame: "close_up",
    kind: "dialogue_face"
  };
  const plan = await planTurn(engine, state, [toDraft(closeUp)]);
  const job = plan.imageJobs?.[0];
  const composed = (job?.providerPayload.characterPrompts ?? []).map((entry) => entry.prompt).join(" | ");
  assertCheck(scope, /sex from behind/iu.test(composed), "premise: continuity injects the ongoing position into the authored face close-up", composed);
  assertCheck(
    scope,
    job?.backend === "novelai" && job.routing?.category === "explicit" && job.routing.locked === true && job.routing.signals.includes("explicit:sex from behind"),
    "the continuity-injected cut is classified explicit and locked to NovelAI even in photoreal mode",
    `${job?.backend} ${job?.routing?.category} ${job?.routing?.signals?.join(" | ")}`
  );
}

async function evaluateReferencePlanning({ seed, orchestrator }) {
  const scope = "references";
  const state = createState(seed.seedState, { routing: { mode: "photoreal" } });
  const portraitCue = toCue(MIXED_CUES[0]);
  const front = orchestrator.planImageJob(state, "turn_ref", portraitCue, [], true);
  const reference = front.providerPayload.photorealReferences?.[0];
  assertCheck(
    scope,
    front.providerPayload.photorealReferences?.length === 1 &&
      reference.name === "char_1" &&
      reference.characterId === "char_yuna" &&
      reference.width === 960 &&
      reference.height === 1280,
    "a front-facing cut plans the character's profile image as reference char_1, with its size for the head crop",
    JSON.stringify(front.providerPayload.photorealReferences)
  );
  assertCheck(scope, /the woman from <image1>/iu.test(front.prompt) && /Use <image1> only as the identity reference/u.test(front.prompt), "the prompt names the reference and closes with the identity-only instruction", front.prompt.slice(0, 120));
  assertCheck(scope, /Korean woman in her late twenties/u.test(front.prompt), "the creator's photoreal appearance text describes the person", front.prompt.slice(0, 160));

  const behind = orchestrator.planImageJob(state, "turn_ref", { ...portraitCue, frame: FRAMES.upper_body_behind }, [], true);
  assertCheck(
    scope,
    (behind.providerPayload.photorealReferences ?? []).length === 0 && !/<image\d>/u.test(behind.prompt) && behind.providerPayload.presetId === "z_image_turbo",
    "a from-behind cut sends no face reference and falls back to the text-to-image preset",
    `${behind.providerPayload.presetId} refs=${JSON.stringify(behind.providerPayload.photorealReferences)}`
  );
  const hips = orchestrator.planImageJob(state, "turn_ref", { ...portraitCue, frame: FRAMES.hip_close_up }, [], true);
  assertCheck(scope, (hips.providerPayload.photorealReferences ?? []).length === 0, "a body-region close-up without the head sends no face reference");

  const toggledOff = createState(seed.seedState, { routing: { mode: "photoreal" }, photorealOverrides: { faceReferenceEnabled: false } });
  const noToggle = orchestrator.planImageJob(toggledOff, "turn_ref", portraitCue, [], true);
  assertCheck(scope, (noToggle.providerPayload.photorealReferences ?? []).length === 0 && !/<image\d>/u.test(noToggle.prompt), "face references stay off when faceReferenceEnabled is false");

  // The edit preset holds at most three references and names them "image N".
  const editState = createState(seed.seedState, {
    routing: { mode: "photoreal" },
    photorealOverrides: { referenceWorkflowPreset: "qwen_image_edit_2511" },
    profileEveryone: true
  });
  const crowd = {
    shouldGenerate: true,
    reason: "four friends",
    characters: ["char_yuna", "char_minho", "char_sora", "char_jiwoo"],
    tags: [],
    baseTags: ["2girls", "2boys", "indoors", "living room"],
    characterPrompts: [
      { characterId: "char_yuna", prompt: "smile, sitting" },
      { characterId: "char_minho", prompt: "sitting, laughing" },
      { characterId: "char_sora", prompt: "standing, waving" },
      { characterId: "char_jiwoo", prompt: "sitting, holding phone" }
    ],
    scene: "current simulation scene",
    visualContext: "",
    frame: FRAMES.cowboy_shot
  };
  const capped = orchestrator.planImageJob(editState, "turn_ref", crowd, [], true);
  const names = (capped.providerPayload.photorealReferences ?? []).map((entry) => entry.name);
  assertCheck(
    scope,
    capped.backend === "photoreal" && capped.providerPayload.presetId === "qwen_image_edit_2511" && sameList(names, ["char_1", "char_2", "char_3"]),
    "references are capped at the preset's maxReferences (edit 2511: 3), in cue order",
    `${capped.backend} ${capped.providerPayload.presetId} ${names.join(",")}`
  );
  assertCheck(scope, /image 1/u.test(capped.prompt) && /image 3/u.test(capped.prompt) && !/image 4/u.test(capped.prompt) && !/<image/u.test(capped.prompt), "the edit preset's prompt uses 'image N' and never names a fourth reference", capped.prompt.slice(0, 200));

  // A male NPC attached to the heroine's id is not the heroine: neither her appearance nor her face is pinned on him.
  const npcCue = {
    shouldGenerate: true,
    reason: "npc",
    characters: ["char_yuna"],
    tags: [],
    baseTags: ["1boy", "solo", "street"],
    characterPrompts: [{ characterId: "char_yuna", prompt: "1boy, male soldier, short hair, armor" }],
    scene: "current simulation scene",
    visualContext: "",
    frame: FRAMES.upper_body
  };
  const npc = orchestrator.planImageJob(state, "turn_ref", npcCue, [], true);
  assertCheck(
    scope,
    npc.backend === "photoreal" && (npc.providerPayload.photorealReferences ?? []).length === 0 && !/Korean woman/u.test(npc.prompt),
    "a figure the composer refused to dress as the registered character gets neither her appearance text nor her face",
    npc.prompt.slice(0, 120)
  );
}

/**
 * Everything a photo prompt can carry is read by the safety floor, not only the cut's tags (review findings
 * safety:appearance-text-unclassified, style-text-unclassified, lexicon gaps, Unicode). Each probe must end locked to
 * NovelAI, or — for text the builder drops — never reach a photo prompt.
 */
async function evaluatePhotoTextFloor({ seed, orchestrator, photoPrompt }) {
  const scope = "floor";
  const portraitCue = toCue(MIXED_CUES[0]);
  const withYuna = (patch, options = {}) => {
    const state = createState(seed.seedState, { routing: { mode: "photoreal" }, ...options });
    state.visualProfiles = state.visualProfiles.map((profile) => (profile.characterId === "char_yuna" ? { ...profile, ...patch(profile) } : profile));
    return state;
  };
  const describe = (job) => `${job.backend}${job.routing?.locked ? "+locked" : ""} ${job.routing?.reason ?? ""} | ${job.prompt.slice(0, 120)}`;
  const isLockedNovelAi = (job) =>
    job.backend === "novelai" && job.routing?.locked === true && /NovelAI 고정/u.test(job.routing.reason) && job.providerPayload.promptFormat === "novelai-tags";

  // The creator's 실사 외형 replaces the adult noun in every photo cut of the character, so it is read like a tag.
  const appearances = [
    "A 15-year-old Korean girl with long black hair",
    "A petite teenage girl with twin tails",
    "一个十四岁的韩国少女",
    "A naked Korean woman with long black hair",
    "A 14-year-old Korean girl in a bikini",
    "A naked woman lying on a bed",
    "A teenage girl in lingerie, lying on a bed",
    "A little girl in a school uniform with a short skirt",
    "A Korean woman in her late twenties, 一个十四岁的韩国少女"
  ];
  for (const appearance of appearances) {
    for (const forced of [undefined, "photoreal"]) {
      const job = orchestrator.planImageJob(withYuna(() => ({ photorealPrompt: appearance })), "turn_floor", portraitCue, [], true, forced ? { backend: forced } : {});
      assertCheck(
        scope,
        isLockedNovelAi(job) && !job.prompt.includes(appearance) && !/一个|少女/u.test(job.prompt),
        `an appearance the floor refuses locks the cut to NovelAI${forced ? " (even when photoreal is forced)" : ""}: ${JSON.stringify(appearance)}`,
        describe(job)
      );
    }
  }
  const chinese = orchestrator.planImageJob(withYuna(() => ({ photorealPrompt: "一个十四岁的韩国少女" })), "turn_floor", portraitCue, [], true);
  assertCheck(
    scope,
    chinese.routing.signals.some((signal) => signal.startsWith("unverifiable-script:")) && /확인할 수 없는 문자/u.test(chinese.routing.reason),
    "text in another script is reported as unverifiable and says so in the lock reason",
    chinese.routing.reason
  );
  const korean = orchestrator.planImageJob(withYuna(() => ({ photorealPrompt: "검은 긴 머리의 20대 한국 여성" })), "turn_floor", portraitCue, [], true);
  assertCheck(
    scope,
    korean.backend === "photoreal" && !korean.routing.locked && !HANGUL_PATTERN.test(korean.prompt) && /long black hair/u.test(korean.prompt),
    "a Korean appearance keeps its documented fallback: not locked, left out, the identity tags describe the person",
    describe(korean)
  );
  const partKorean = orchestrator.planImageJob(
    withYuna(() => ({ photorealPrompt: "A Korean woman in her late twenties, 검은 긴 머리" })),
    "turn_floor",
    portraitCue,
    [],
    true
  );
  assertCheck(
    scope,
    partKorean.backend === "photoreal" && /a Korean woman in her late twenties/iu.test(partKorean.prompt) && !HANGUL_PATTERN.test(partKorean.prompt),
    "only the Korean segment of an appearance drops out; the English rest stays the subject",
    describe(partKorean)
  );
  const scarred = orchestrator.planImageJob(
    withYuna(() => ({ photorealPrompt: "A Korean woman in her late twenties with a minor scar on her chin" })),
    "turn_floor",
    portraitCue,
    [],
    true
  );
  assertCheck(scope, scarred.backend === "photoreal" && !scarred.routing.locked, "\"a minor scar\" is not a minor (false-friend guard)", describe(scarred));

  // The photoreal style opens and closes every photo prompt; it is read too, for the floor only.
  const cafeCue = { ...portraitCue, characterPrompts: [{ characterId: "char_yuna", prompt: "smile, sitting" }] };
  const styles = [
    { stylePreset: "custom", stylePrompt: "A nude boudoir photograph. Teenage model, 16 years old." },
    { stylePreset: "daily_look", stylePrompt: "fully naked subject, no clothes" },
    { stylePreset: "cinematic", stylePrompt: "soft light, ヌード" }
  ];
  for (const style of styles) {
    const job = orchestrator.planImageJob(createState(seed.seedState, { routing: { mode: "photoreal" }, photorealOverrides: style }), "turn_floor", cafeCue, [], true);
    assertCheck(
      scope,
      isLockedNovelAi(job) && !/boudoir|naked|teenage|ヌード/iu.test(job.prompt),
      `a style text the floor refuses locks every cut to NovelAI: ${JSON.stringify(style.stylePrompt)}`,
      describe(job)
    );
  }
  const nsfwStyleOnNovelAi = createState(seed.seedState, { routing: { mode: "novelai" }, photorealOverrides: styles[0] });
  const plainOnNovelAi = createState(seed.seedState, { routing: { mode: "novelai" } });
  const lockedNai = orchestrator.planImageJob(nsfwStyleOnNovelAi, "turn_floor", cafeCue, [], true);
  const plainNai = orchestrator.planImageJob(plainOnNovelAi, "turn_floor", cafeCue, [], true);
  assertCheck(
    scope,
    lockedNai.backend === "novelai" && lockedNai.prompt === plainNai.prompt && lockedNai.negativePrompt === plainNai.negativePrompt,
    "a NovelAI cut's prompt is untouched by the photo floor (only its routing record says locked)",
    `${lockedNai.prompt.slice(0, 80)} / ${plainNai.prompt.slice(0, 80)}`
  );

  // Tags the lexicons used to miss: inflections, synonyms, hyphenation, Unicode look-alikes, kana.
  const tagProbes = [
    "teenaged, bikini",
    "school-uniform, lingerie",
    "17 years of age",
    "aged fifteen",
    "sixth grader",
    "adolescent",
    "ｎｕｄｅ",
    "n\u0430ked",
    "nu\u200bde",
    "nu\u00adde",
    "ロリ, bikini",
    "-2::clothing::, standing"
  ];
  for (const tags of tagProbes) {
    const cue = { ...portraitCue, characterPrompts: [{ characterId: "char_yuna", prompt: `smile, ${tags}` }] };
    const job = orchestrator.planImageJob(createState(seed.seedState, { routing: { mode: "photoreal" } }), "turn_floor", cue, [], true);
    assertCheck(scope, isLockedNovelAi(job), `a cut tagged ${JSON.stringify(tags)} is locked to NovelAI`, describe(job));
  }
  const kanaOnNovelAi = orchestrator.planImageJob(
    createState(seed.seedState, { routing: { mode: "novelai" } }),
    "turn_floor",
    { ...portraitCue, characterPrompts: [{ characterId: "char_yuna", prompt: "smile, ロリ" }] },
    [],
    true
  );
  const kanaCaption = kanaOnNovelAi.providerPayload.characterPrompts?.[0]?.prompt ?? "";
  assertCheck(
    scope,
    kanaOnNovelAi.backend === "novelai" && kanaOnNovelAi.routing.locked === true && kanaCaption.includes("ロリ"),
    "NovelAI still receives the cut exactly as composed (kana included): the floor only decides where a cut may NOT go",
    kanaCaption
  );
  const fashion = orchestrator.planImageJob(
    createState(seed.seedState, { routing: { mode: "photoreal" } }),
    "turn_floor",
    { ...portraitCue, characterPrompts: [{ characterId: "char_yuna", prompt: "smile, nude lipstick, aged leather jacket" }] },
    [],
    true
  );
  assertCheck(scope, fashion.backend === "photoreal" && !fashion.routing.locked, "nude lipstick and an aged leather jacket stay photoreal (false-friend guard)", describe(fashion));

  // A profile image is the face of every later photo cut: its own prompt is re-read before it is used.
  const taintedReference = orchestrator.planImageJob(
    withYuna((profile) => ({ profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, prompt: "A 14-year-old Korean girl in a bikini" } } })),
    "turn_floor",
    portraitCue,
    [],
    true
  );
  assertCheck(
    scope,
    taintedReference.backend === "photoreal" &&
      (taintedReference.providerPayload.photorealReferences ?? []).length === 0 &&
      !/<image\d>/u.test(taintedReference.prompt) &&
      taintedReference.providerPayload.photorealReferenceRefusals?.[0]?.characterId === "char_yuna",
    "a profile image made from a prompt the floor refuses is never sent as a face reference, and the plan says why",
    JSON.stringify(taintedReference.providerPayload.photorealReferenceRefusals)
  );
  const kanaReference = orchestrator.planImageJob(
    withYuna((profile) => ({ profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, prompt: "1girl, ロリ" } } })),
    "turn_floor",
    portraitCue,
    [],
    true
  );
  assertCheck(scope, (kanaReference.providerPayload.photorealReferences ?? []).length === 0, "a profile prompt in another script is unverifiable: no reference");
  const koreanReference = orchestrator.planImageJob(
    withYuna((profile) => ({ profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, prompt: "검은 머리의 20대 여성" } } })),
    "turn_floor",
    portraitCue,
    [],
    true
  );
  // Changed deliberately (round 3): a Korean-only record used to keep its reference. The builder dropped that Korean
  // when the image was made, so the floor read nothing in it — and a record the floor reads nothing in vouches for
  // nothing. A record with English words beside the Korean still does.
  assertCheck(
    scope,
    (koreanReference.providerPayload.photorealReferences ?? []).length === 0 &&
      /읽을 수 있는 영어 단어가 없어/u.test(koreanReference.providerPayload.photorealReferenceRefusals?.[0]?.reason ?? ""),
    "a Korean-only profile prompt (dropped when the image was made, so the floor read nothing) no longer vouches for its image, and the plan says why",
    JSON.stringify(koreanReference.providerPayload.photorealReferenceRefusals)
  );
  const mixedReference = orchestrator.planImageJob(
    withYuna((profile) => ({ profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, prompt: "검은 머리, a Korean woman in her twenties" } } })),
    "turn_floor",
    portraitCue,
    [],
    true
  );
  assertCheck(scope, mixedReference.providerPayload.photorealReferences?.length === 1, "a record with English words beside Korean keeps its reference");
  // Red team S7: records the floor reads nothing in were accepted as long as they were not blank. Each probe is refused
  // the same way, at plan time, with no <imageN> in the prompt.
  for (const prompt of ["\u200b", "\u200b\u200b\u2060", "\u3164", ".", "\u2014", "\ud83d\ude42", "%%", "<image1>", "1girl", "1girl, 1boy", "image1", "%prompt%", "\u00a0\u3000", "~~ ** ##"]) {
    const job = orchestrator.planImageJob(
      withYuna((profile) => ({ profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, prompt } } })),
      "turn_floor",
      portraitCue,
      [],
      true
    );
    assertCheck(
      scope,
      (job.providerPayload.photorealReferences ?? []).length === 0 &&
        !/<image\d>/u.test(job.prompt) &&
        /(?:읽을 수 있는 영어 단어가 없어|프롬프트 기록이 없어)/u.test(job.providerPayload.photorealReferenceRefusals?.[0]?.reason ?? ""),
      `a profile record whose prompt is ${JSON.stringify(prompt)} says nothing the floor can read: no face reference, and the plan says why`,
      JSON.stringify(job.providerPayload.photorealReferenceRefusals)
    );
  }
  // A benign prompt vouches only for an image the user picked from generated candidates: the key must be a pick's.
  for (const objectKey of [`${SIMULATION_ID}/asset_cut_1.png`, `${SIMULATION_ID}/candidate.png`, "profile_yuna.png", `${SIMULATION_ID}/nested/profile_yuna.png`, `${SIMULATION_ID}/profile_yuna.gif`]) {
    const job = orchestrator.planImageJob(
      withYuna((profile) => ({ profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, objectKey } } })),
      "turn_floor",
      portraitCue,
      [],
      true
    );
    assertCheck(
      scope,
      (job.providerPayload.photorealReferences ?? []).length === 0 && /프로필로 고른 이미지가 아니어서/u.test(job.providerPayload.photorealReferenceRefusals?.[0]?.reason ?? ""),
      `a record with a clean prompt but the object key ${JSON.stringify(objectKey)} (not a picked profile image) is not a face reference`,
      JSON.stringify(job.providerPayload.photorealReferenceRefusals)
    );
  }
  // The image is vouched for only by its recorded prompt. hydrateState turns a missing one into "", which used to read
  // as "nothing to object to" and sent the image as <image1>.
  for (const [label, patch] of [
    ["missing", (photoreal) => { const { prompt: _prompt, ...rest } = photoreal; return rest; }],
    ["empty", (photoreal) => ({ ...photoreal, prompt: "" })],
    ["blank", (photoreal) => ({ ...photoreal, prompt: "   " })]
  ]) {
    const unrecorded = orchestrator.planImageJob(
      withYuna((profile) => ({ profileImages: { ...profile.profileImages, photoreal: patch(profile.profileImages.photoreal) } })),
      "turn_floor",
      portraitCue,
      [],
      true
    );
    assertCheck(
      scope,
      unrecorded.backend === "photoreal" &&
        (unrecorded.providerPayload.photorealReferences ?? []).length === 0 &&
        !/<image\d>/u.test(unrecorded.prompt) &&
        /프롬프트 기록이 없어/u.test(unrecorded.providerPayload.photorealReferenceRefusals?.[0]?.reason ?? ""),
      `a profile image whose recorded prompt is ${label} is not trusted as a face reference (fails closed), and the plan says why`,
      JSON.stringify(unrecorded.providerPayload.photorealReferenceRefusals)
    );
  }
  const hydrated = seed.hydrateState(
    withYuna((profile) => {
      const { prompt: _prompt, ...rest } = profile.profileImages.photoreal;
      return { profileImages: { ...profile.profileImages, photoreal: rest } };
    })
  );
  const hydratedImage = hydrated.visualProfiles.find((profile) => profile.characterId === "char_yuna")?.profileImages?.photoreal;
  assertCheck(
    scope,
    hydratedImage?.prompt === "" && orchestrator.screenPhotorealReferenceImage(hydrated, hydratedImage) !== undefined,
    "after hydrateState (missing prompt → \"\") the reference gate still refuses the record",
    JSON.stringify(hydratedImage)
  );

  // The final gate: the finished photo prompt is classified again, and a refused one becomes a locked NovelAI cut.
  const clean = orchestrator.planImageJob(createState(seed.seedState, { routing: { mode: "photoreal" } }), "turn_floor", portraitCue, [], true);
  assertCheck(
    scope,
    clean.backend === "photoreal" && orchestrator.screenFinishedPhotorealPrompt(createState(seed.seedState, { routing: { mode: "photoreal" } }), clean.prompt, clean.routing) === undefined,
    "a clean photo prompt passes the final gate untouched",
    describe(clean)
  );
  const gateState = createState(seed.seedState, { routing: { mode: "photoreal" } });
  const finished = [
    ["A candid photograph. A nude woman sitting on a bed.", "explicit"],
    ["A candid photograph. An adult woman, ロリ, smiling.", "portrait"],
    ["A candid photograph. A 16 years old model, smiling.", "portrait"],
    ["A candid photograph. An adult woman wearing a school uniform and lingerie.", "portrait"],
    ["A candid photograph. An adult woman lying on her back, wearing a bikini.", "explicit"]
  ];
  for (const [prompt, category] of finished) {
    const decision = orchestrator.screenFinishedPhotorealPrompt(gateState, prompt, clean.routing, "photoreal");
    assertCheck(
      scope,
      decision?.backend === "novelai" &&
        decision.locked === true &&
        decision.category === category &&
        decision.reason.startsWith("완성된 실사 프롬프트가 안전 기준에 걸려 NovelAI로 바꿨습니다") &&
        decision.signals.some((signal) => signal.startsWith("photo-prompt:")),
      `the final gate refuses a photo prompt the floor would lock: ${JSON.stringify(prompt)}`,
      decision ? `${decision.backend} ${decision.category} ${decision.reason}` : "passed"
    );
  }

  // NovelAI's undressing idiom is not turned into a photo negative; ordinary negative weights still are.
  const hat = orchestrator.planImageJob(
    withYuna(() => ({ photorealNegativePrompt: "clothing, glasses" })),
    "turn_floor",
    { ...portraitCue, characterPrompts: [{ characterId: "char_yuna", prompt: "smile, -1::hat::" }] },
    [],
    true
  );
  const negatives = hat.negativePrompt.split(/,\s*/u);
  // Whatever the one negative rule keeps of "hat, glasses" — not a list frozen here, since that rule is what decides.
  const keptAccessories = photoPrompt.sanitizePhotorealNegativePrompt("hat, glasses").split(/,\s*/u).filter(Boolean);
  assertCheck(
    scope,
    hat.backend === "photoreal" &&
      keptAccessories.every((term) => negatives.includes(term)) &&
      !negatives.includes("clothing") &&
      hat.negativePrompt === photoPrompt.sanitizePhotorealNegativePrompt(hat.negativePrompt),
    `a negative weight on a hat becomes a photo negative as the one negative rule keeps it (${keptAccessories.join(", ") || "nothing"}); body-covering clothing never does, from any source; the planned negative is already in its sent form`,
    hat.negativePrompt
  );
}

async function evaluateReuseIsolation({ seed, orchestrator, engine }) {
  const scope = "reuse";
  const state = createState(seed.seedState, { routing: { mode: "mixed" } });
  const portraitCue = toCue(MIXED_CUES[0]);
  const photorealJob = orchestrator.planImageJob(state, "turn_reuse", portraitCue, [], false);
  const novelAiJob = orchestrator.planImageJob(state, "turn_reuse", portraitCue, [], false, { backend: "novelai" });
  assertCheck(scope, photorealJob.backend === "photoreal" && novelAiJob.backend === "novelai", "premise: the same cut planned on both backends");

  const novelAiAsset = assetFromJob(novelAiJob, "asset_reuse_novelai", "novelai");
  const photorealAsset = assetFromJob(photorealJob, "asset_reuse_photoreal", "photoreal");
  const { backend: _ignored, ...legacyAsset } = { ...novelAiAsset, id: "asset_reuse_legacy" };
  const withAssets = (assets) => ({ ...state, imageAssets: assets });

  const naiForNai = orchestrator.findReusableImageAsset(withAssets([novelAiAsset]), novelAiJob);
  assertCheck(scope, naiForNai?.asset.id === "asset_reuse_novelai", "premise: a NovelAI image of the same cut is reusable for a NovelAI job");
  assertCheck(scope, !orchestrator.findReusableImageAsset(withAssets([novelAiAsset]), photorealJob), "a NovelAI image never satisfies a photoreal cut");
  assertCheck(scope, orchestrator.findReusableImageAsset(withAssets([photorealAsset]), photorealJob)?.asset.id === "asset_reuse_photoreal", "a photoreal image of the same cut is reusable for a photoreal job");
  assertCheck(scope, !orchestrator.findReusableImageAsset(withAssets([photorealAsset]), novelAiJob), "a photoreal image never satisfies a NovelAI cut");
  assertCheck(
    scope,
    orchestrator.findReusableImageAsset(withAssets([legacyAsset]), novelAiJob)?.asset.id === "asset_reuse_legacy" && !orchestrator.findReusableImageAsset(withAssets([legacyAsset]), photorealJob),
    "an asset from before photoreal (no backend) counts as NovelAI"
  );
  assertCheck(
    scope,
    !orchestrator.findReusableImageAsset(withAssets([novelAiAsset]), novelAiJob, { backend: "photoreal" }),
    "the backend option overrides the job's own (what the turn planner passes)"
  );
  const size = (await vite.ssrLoadModule("/src/services/comfyWorkflows.ts")).resolvePhotorealOutputSize(state);
  const wrongSize = {
    ...photorealAsset,
    id: "asset_reuse_wrong_size",
    providerMetadata: { ...photorealAsset.providerMetadata, width: size.width + 64, height: size.height + 64 }
  };
  assertCheck(
    scope,
    !orchestrator.findReusableImageAsset(withAssets([wrongSize]), photorealJob),
    "a photoreal image at a different size than the photoreal output size is not reused",
    `${size.width}x${size.height}`
  );
  assertCheck(
    scope,
    size.width !== state.imageProfile.width || size.height !== state.imageProfile.height
      ? !orchestrator.findReusableImageAsset(
          withAssets([{ ...photorealAsset, id: "asset_reuse_nai_size", providerMetadata: { ...photorealAsset.providerMetadata, width: state.imageProfile.width, height: state.imageProfile.height } }]),
          photorealJob
        )
      : true,
    "a photoreal image is measured against the photoreal size, not NovelAI's profile size",
    `${size.width}x${size.height} vs ${state.imageProfile.width}x${state.imageProfile.height}`
  );

  // Through the turn planner: the same NovelAI asset is reused in NovelAI mode, and ignored when the cut routes photoreal.
  const reuseCue = { ...MIXED_CUES[0], label: "reuse portrait" };
  const novelAiMode = createState(seed.seedState, { routing: { mode: "novelai" }, cadence: "balanced" });
  const novelAiPlan = await planTurn(engine, { ...novelAiMode, imageAssets: [novelAiAsset] }, [toDraft(reuseCue)]);
  assertCheck(
    scope,
    (novelAiPlan.imageJobs ?? []).length === 0 && (novelAiPlan.reusedAssetIds ?? []).includes("asset_reuse_novelai"),
    "turn planning in NovelAI mode reuses the matching NovelAI image",
    `jobs=${novelAiPlan.imageJobs?.length} reused=${novelAiPlan.reusedAssetIds?.join(",")}`
  );
  const mixedMode = createState(seed.seedState, { routing: { mode: "mixed" }, cadence: "balanced" });
  const mixedPlan = await planTurn(engine, { ...mixedMode, imageAssets: [novelAiAsset] }, [toDraft(reuseCue)]);
  assertCheck(
    scope,
    (mixedPlan.imageJobs ?? []).length === 1 && mixedPlan.imageJobs[0].backend === "photoreal" && (mixedPlan.reusedAssetIds ?? []).length === 0,
    "turn planning in mixed mode plans a photoreal job instead of reusing the NovelAI image",
    `jobs=${mixedPlan.imageJobs?.map((job) => job.backend).join(",")} reused=${mixedPlan.reusedAssetIds?.join(",")}`
  );
}

async function evaluateRegeneration({ seed, orchestrator }) {
  const scope = "regenerate";
  const portraitCue = toCue(MIXED_CUES[0]);
  const explicitCue = toCue(MIXED_CUES[6]);
  const novelAiMode = createState(seed.seedState, { routing: { mode: "novelai" } });
  const photorealMode = createState(seed.seedState, { routing: { mode: "photoreal" } });

  const forcedPhotoreal = orchestrator.planImageJob(novelAiMode, "turn_regen", portraitCue, [], true, { backend: "photoreal" });
  assertCheck(scope, forcedPhotoreal.backend === "photoreal" && /직접 지정/u.test(forcedPhotoreal.routing.reason), "regenerating with the photoreal backend in NovelAI mode plans photoreal", forcedPhotoreal.routing.reason);
  const forcedNovelAi = orchestrator.planImageJob(photorealMode, "turn_regen", portraitCue, [], true, { backend: "novelai" });
  assertCheck(scope, forcedNovelAi.backend === "novelai" && forcedNovelAi.providerPayload.promptFormat === "novelai-tags", "regenerating with the NovelAI backend in photoreal mode plans NovelAI");
  const kept = orchestrator.planImageJob(photorealMode, "turn_regen", portraitCue, [], true, { backend: forcedPhotoreal.backend });
  assertCheck(scope, kept.backend === "photoreal", "regenerating with the job's own backend keeps it");
  const lockedForce = orchestrator.planImageJob(photorealMode, "turn_regen", explicitCue, [], true, { backend: "photoreal" });
  assertCheck(
    scope,
    lockedForce.backend === "novelai" && lockedForce.routing.locked === true && /직접 지정해도 바뀌지 않습니다/u.test(lockedForce.routing.reason),
    "the safety lock beats a forced photoreal backend",
    lockedForce.routing.reason
  );
  const offForced = orchestrator.planImageJob(createState(seed.seedState, { routing: { mode: "novelai" }, photoreal: false }), "turn_regen", portraitCue, [], true, { backend: "photoreal" });
  assertCheck(
    scope,
    offForced.backend === "photoreal" && offForced.providerPayload.setupBlockedReason === PHOTOREAL_SETUP_TEXT,
    "a forced backend that is switched off is kept (no silent fallback) and waits with its setup message",
    `${offForced.backend} ${offForced.providerPayload.setupBlockedReason}`
  );
  const withKind = orchestrator.planImageJob(photorealMode, "turn_regen", portraitCue, [], true, { cueKind: "action" });
  assertCheck(scope, withKind.providerPayload.cue.kind === "action" && withKind.routing.signals.includes("kind:action"), "options.cueKind is recorded on the cue and in the routing signals");
  assertCheck(scope, withKind.routing.category === "portrait", "a bare 'action' kind does not make a static cut dynamic");
}

async function evaluateCountLimitPolicy({ seed, orchestrator }) {
  const scope = "policy";
  const portraitCue = toCue(MIXED_CUES[0]);
  const base = createState(seed.seedState, { routing: { mode: "mixed" } });
  const limited = (imageJobs) => ({
    ...base,
    novelAi: { ...base.novelAi, automationTermination: "count", countLimit: 1 },
    messages: [],
    imageJobs
  });
  const completedNovelAi = { ...orchestrator.planImageJob(base, "turn_done", portraitCue, [], true, { backend: "novelai" }), status: "completed" };
  const completedPhotoreal = { ...orchestrator.planImageJob(base, "turn_done", portraitCue, [], true, { backend: "photoreal" }), status: "completed" };

  const blockedNovelAi = orchestrator.planImageJob(limited([completedNovelAi]), "turn_policy", portraitCue, [], true, { backend: "novelai" });
  assertCheck(scope, blockedNovelAi.status === "failed" && /count limit/u.test(blockedNovelAi.error ?? ""), "the NovelAI count limit still stops a NovelAI job", blockedNovelAi.error);
  const photorealUnderLimit = orchestrator.planImageJob(limited([completedNovelAi]), "turn_policy", portraitCue, [], true, { backend: "photoreal" });
  assertCheck(scope, photorealUnderLimit.status === "queued" && photorealUnderLimit.providerPayload.policy.allowed === true, "the NovelAI count limit does not apply to a photoreal job", photorealUnderLimit.error);
  const novelAiAfterPhotoreal = orchestrator.planImageJob(limited([completedPhotoreal, { ...completedPhotoreal, id: "imgjob_other" }]), "turn_policy", portraitCue, [], true, { backend: "novelai" });
  assertCheck(scope, novelAiAfterPhotoreal.status === "queued", "completed photoreal jobs do not count toward the NovelAI limit", novelAiAfterPhotoreal.error);
}

async function evaluateSetupBlocks({ seed, orchestrator }) {
  const scope = "setup-block";
  const portraitCue = toCue(MIXED_CUES[0]);
  const both = createState(seed.seedState, { routing: { mode: "mixed" } });
  const novelAiOff = createState(seed.seedState, { routing: { mode: "mixed" }, novelAi: false });
  const photorealOff = createState(seed.seedState, { routing: { mode: "mixed" }, photoreal: false });

  const photorealJob = orchestrator.planImageJob(both, "turn_setup", portraitCue, [], false);
  assertCheck(scope, photorealJob.backend === "photoreal" && photorealJob.providerPayload.setupBlockedReason === undefined, "a photoreal cut with ComfyUI on is not blocked");
  const blockedPhotoreal = orchestrator.planImageJob(photorealOff, "turn_setup", portraitCue, [], false, { backend: "photoreal" });
  assertCheck(scope, blockedPhotoreal.providerPayload.setupBlockedReason === PHOTOREAL_SETUP_TEXT, "a photoreal cut with ComfyUI off waits with the photoreal setup text", blockedPhotoreal.providerPayload.setupBlockedReason);
  const lockedWithNovelAiOff = orchestrator.planImageJob(novelAiOff, "turn_setup", toCue(MIXED_CUES[6]), [], false);
  assertCheck(
    scope,
    lockedWithNovelAiOff.backend === "novelai" && lockedWithNovelAiOff.providerPayload.setupBlockedReason === NOVELAI_SETUP_TEXT,
    "a locked cut with NovelAI off stays NovelAI and waits with the NovelAI text (never promoted to photoreal)",
    `${lockedWithNovelAiOff.backend} ${lockedWithNovelAiOff.providerPayload.setupBlockedReason}`
  );
  assertCheck(scope, orchestrator.describeNovelAiSetupBlock(novelAiOff) === NOVELAI_SETUP_TEXT && orchestrator.describeNovelAiSetupBlock(both) === undefined, "describeNovelAiSetupBlock keeps its exact text");

  assertCheck(scope, orchestrator.shouldAutoRunImageJob(photorealJob, both) === true, "a queued photoreal job auto-runs while ComfyUI is on");
  assertCheck(scope, orchestrator.shouldAutoRunImageJob(photorealJob, novelAiOff) === true, "NovelAI being off does not hold back a photoreal job");
  assertCheck(scope, orchestrator.shouldAutoRunImageJob(photorealJob, photorealOff) === false, "ComfyUI being off holds back a photoreal job, re-derived from live state");
  const novelAiJob = orchestrator.planImageJob(both, "turn_setup", portraitCue, [], false, { backend: "novelai" });
  assertCheck(scope, orchestrator.shouldAutoRunImageJob(novelAiJob, novelAiOff) === false && orchestrator.shouldAutoRunImageJob(novelAiJob, photorealOff) === true, "a NovelAI job is held back by NovelAI's state only");
  const legacyJob = { ...novelAiJob, backend: undefined, providerPayload: { ...novelAiJob.providerPayload, provider: undefined } };
  assertCheck(scope, orchestrator.resolveImageJobBackend(legacyJob) === "novelai" && orchestrator.shouldAutoRunImageJob(legacyJob, novelAiOff) === false, "a job from before photoreal is read as NovelAI");
  assertCheck(scope, orchestrator.canUserRunQueuedImageJob(blockedPhotoreal) === true, "a setup-blocked photoreal cut offers the run-now control");
}

// ---------------------------------------------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------------------------------------------

async function evaluateDryRunExecution({ seed, orchestrator }) {
  const scope = "dry-run";
  const both = createState(seed.seedState, { routing: { mode: "mixed" } });
  const photorealJob = orchestrator.planImageJob(both, "turn_dry", toCue(MIXED_CUES[0]), [], true);

  const disabled = await orchestrator.executeImageJob({ ...both, photoreal: { ...both.photoreal, enabled: false } }, photorealJob);
  assertCheck(scope, disabled.job.status === "failed" && disabled.job.error === PHOTOREAL_DISABLED_TEXT, "photoreal off: the job fails with the photoreal 'switched off' text", disabled.job.error);
  assertCheck(scope, disabled.job.backend === "photoreal" && disabled.job.routing?.backend === "photoreal" && disabled.assets.length === 0, "a failed photoreal job keeps its backend and routing");
  const mock = await orchestrator.executeImageJob({ ...both, photoreal: { ...both.photoreal, requestMode: "mock" } }, photorealJob);
  assertCheck(scope, mock.job.status === "failed" && mock.job.error === PHOTOREAL_MOCK_TEXT, "photoreal mock: the job fails with the photoreal mock text", mock.job.error);

  const novelAiJob = orchestrator.planImageJob(both, "turn_dry", toCue(MIXED_CUES[0]), [], true, { backend: "novelai" });
  const novelAiDisabled = await orchestrator.executeImageJob({ ...both, novelAi: { ...both.novelAi, enabled: false } }, novelAiJob);
  assertCheck(scope, novelAiDisabled.job.error === NOVELAI_DISABLED_TEXT, "NovelAI off: the NovelAI text is byte-identical", novelAiDisabled.job.error);
  const novelAiMock = await orchestrator.executeImageJob({ ...both, novelAi: { ...both.novelAi, requestMode: "mock" } }, novelAiJob);
  assertCheck(scope, novelAiMock.job.error === NOVELAI_MOCK_TEXT, "NovelAI mock: the NovelAI text is byte-identical", novelAiMock.job.error);

  const aborted = new AbortController();
  aborted.abort();
  const canceled = await orchestrator.executeImageJob(both, novelAiJob, { signal: aborted.signal });
  assertCheck(scope, canceled.job.status === "canceled" && canceled.job.error === undefined && canceled.assets.length === 0, "an aborted request comes back canceled, not failed", `${canceled.job.status} ${canceled.job.error}`);
}

async function evaluateNovelAiPreciseReference({ seed, orchestrator }, { referencePng, outputPng }) {
  const scope = "novelai-reference";
  const referenceBase64 = referencePng.toString("base64");
  const configure = (modelPreset, overrides = {}) => {
    const state = createState(seed.seedState, { routing: { mode: "novelai" } });
    state.novelAi = {
      ...state.novelAi,
      enabled: true,
      requestMode: "direct",
      endpoint: NOVELAI_ENDPOINT,
      apiKey: "pst-eval",
      modelPreset,
      generationDelaySeconds: 0,
      characterReferenceEnabled: true,
      characterReferenceStrength: 0.7,
      characterReferenceFidelity: 0.8,
      characterReferenceStyleAware: true,
      vibeTransferEnabled: true,
      vibeTransferReferences: [
        {
          id: "vibe_eval",
          name: "vibe",
          image: "data:image/png;base64,AAAA",
          referenceStrength: 0.6,
          informationExtracted: 1,
          encodedVibe: "ZW5jb2RlZC12aWJl",
          encodedModel: modelPreset,
          encodedInformationExtracted: 1
        }
      ],
      ...overrides
    };
    return state;
  };
  const portraitCue = toCue(MIXED_CUES[0]);

  const off = configure("NAID4.5F", { characterReferenceEnabled: false });
  assertCheck(scope, !("characterReference" in orchestrator.planImageJob(off, "turn_nai", portraitCue, [], true).providerPayload), "no Precise Reference is planned while the opt-in is off");

  const state45 = configure("NAID4.5F");
  const job45 = orchestrator.planImageJob(state45, "turn_nai", portraitCue, [], true);
  assertCheck(
    scope,
    job45.backend === "novelai" &&
      job45.providerPayload.characterReference?.characterId === "char_yuna" &&
      job45.providerPayload.characterReference?.objectKey === `${SIMULATION_ID}/profile_yuna_nai_ref.png`,
    "the primary figure's letterboxed NovelAI profile is planned as {characterId, objectKey}",
    JSON.stringify(job45.providerPayload.characterReference)
  );
  assertCheck(scope, !JSON.stringify(job45.providerPayload).includes(referenceBase64), "the plan carries the object key, never the bytes");

  const novelAiBodies = [];
  const objectRequests = [];
  let objectStatus = 200;
  const stub = async (url, init) => {
    if (url.startsWith(DEFAULT_OBJECT_BASE)) {
      objectRequests.push(url);
      return objectStatus === 200
        ? new Response(referencePng, { status: 200, headers: { "content-type": "image/png", "content-length": String(referencePng.length) } })
        : new Response("missing", { status: objectStatus, headers: { "content-type": "text/plain" } });
    }
    if (url === NOVELAI_ENDPOINT) {
      novelAiBodies.push(JSON.parse(String(init?.body ?? "{}")));
      return new Response(outputPng, { status: 200, headers: { "content-type": "image/png" } });
    }
    return undefined;
  };

  await withFetchStub(stub, async () => {
    const result45 = await orchestrator.executeImageJob(state45, job45);
    const sent = novelAiBodies.at(-1)?.parameters ?? {};
    const stored = result45.job.providerPayload.parameters ?? {};
    const arrays = [
      "director_reference_images",
      "director_reference_descriptions",
      "director_reference_information_extracted",
      "director_reference_strength_values",
      "director_reference_secondary_strength_values"
    ];
    assertCheck(scope, result45.job.status === "completed" && result45.assets.length === 1 && result45.assets[0].backend === "novelai", "a V4.5 cut with a reference completes as a NovelAI asset", result45.job.error);
    assertCheck(scope, objectRequests.length === 1 && objectRequests[0].endsWith(`/objects/${SIMULATION_ID}/profile_yuna_nai_ref.png`), "the reference canvas is read from the object store by its key", objectRequests.join(", "));
    assertCheck(
      scope,
      arrays.every((key) => Array.isArray(sent[key]) && sent[key].length === 1) && sent.director_reference_images[0] === referenceBase64,
      "NovelAI receives all five director_reference arrays, one entry each, carrying the reference bytes",
      arrays.map((key) => `${key}:${Array.isArray(sent[key]) ? sent[key].length : "-"}`).join(" ")
    );
    assertCheck(
      scope,
      sent.director_reference_strength_values?.[0] === 0.7 &&
        sent.director_reference_secondary_strength_values?.[0] === 0.2 &&
        sent.director_reference_descriptions?.[0]?.caption?.base_caption === "character&style",
      "strength, inverted fidelity and style-aware come from the settings",
      JSON.stringify([sent.director_reference_strength_values, sent.director_reference_secondary_strength_values])
    );
    assertCheck(
      scope,
      !("reference_image_multiple" in sent) && sent.dynamicchat_vibe_transfer_skipped === "character_reference",
      "vibe transfer sits out a cut that carries a Precise Reference"
    );
    assertCheck(
      scope,
      typeof stored.director_reference_images?.[0] === "string" && stored.director_reference_images[0].startsWith("<omitted:") && !JSON.stringify(result45.job.providerPayload).includes(referenceBase64),
      "the stored payload records the reference's size, not its bytes",
      String(stored.director_reference_images?.[0])
    );
    assertCheck(
      scope,
      result45.job.providerPayload.provider === "novelai" && result45.job.providerPayload.routing?.backend === "novelai" && result45.job.backend === "novelai",
      "the result payload keeps provider and routing"
    );

    const state5 = configure("NAID5F");
    const job5 = orchestrator.planImageJob(state5, "turn_nai", portraitCue, [], true);
    const result5 = await orchestrator.executeImageJob(state5, job5);
    const sent5 = novelAiBodies.at(-1)?.parameters ?? {};
    assertCheck(
      scope,
      result5.job.status === "completed" && sent5.dynamicchat_character_reference_skipped === "model" && !("director_reference_images" in sent5) && !("reference_image_multiple" in sent5),
      "on V5 the reference is skipped (and flagged) and no vibe arrays are sent",
      JSON.stringify(Object.keys(sent5).filter((key) => /reference|vibe/u.test(key)))
    );

    objectStatus = 404;
    const missing = await orchestrator.executeImageJob(state45, orchestrator.planImageJob(state45, "turn_nai", portraitCue, [], true));
    const sentMissing = novelAiBodies.at(-1)?.parameters ?? {};
    assertCheck(
      scope,
      missing.job.status === "completed" && !("director_reference_images" in sentMissing) && (missing.job.policyWarnings ?? []).some((warning) => /레퍼런스 없이/u.test(warning)),
      "an unreadable reference degrades to generating without it, with a warning on the job",
      JSON.stringify(missing.job.policyWarnings)
    );
    objectStatus = 200;

    const before = objectRequests.length;
    const turnedOff = await orchestrator.executeImageJob({ ...state45, novelAi: { ...state45.novelAi, characterReferenceEnabled: false } }, job45);
    assertCheck(
      scope,
      turnedOff.job.status === "completed" && objectRequests.length === before && !("director_reference_images" in (novelAiBodies.at(-1)?.parameters ?? {})),
      "turning the opt-in off stops the reference for cuts planned while it was on"
    );

    const seeded = { ...job45, providerPayload: { ...job45.providerPayload, seed: 424242 } };
    await orchestrator.executeImageJob(state45, seeded);
    assertCheck(scope, novelAiBodies.at(-1)?.parameters?.seed === 424242, "a seed the planner asked for reaches NovelAI", String(novelAiBodies.at(-1)?.parameters?.seed));
  });
}

async function evaluateRelayExecution({ seed, orchestrator, studio, photoPrompt }, { referencePng, outputPng }) {
  const scope = "relay";
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), "dynamicchat-photoreal-pipeline-"));
  cleanups.push(() => rm(tempRoot, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 }));
  const mock = await startMockComfyUi({ outputPng });
  cleanups.push(() => mock.close());
  const server = await startApiServer(tempRoot);
  cleanups.push(() => server.stop());
  await mkdir(path.join(server.dataDir, "objects", SIMULATION_ID), { recursive: true });
  await writeFile(path.join(server.dataDir, "objects", SIMULATION_ID, "profile_yuna.png"), referencePng);

  const state = createState(seed.seedState, {
    routing: { mode: "photoreal" },
    photorealOverrides: { baseUrl: mock.baseUrl, proxyUrl: `${server.baseUrl}/photoreal/comfyui`, timeoutSeconds: 60 }
  });
  const outputDataUrl = `data:image/png;base64,${outputPng.toString("base64")}`;

  // A cut with a face reference: the profile bytes go to ComfyUI, the prompt names <image1>.
  mock.reset();
  const referenced = orchestrator.planImageJob(state, "turn_relay", toCue(MIXED_CUES[0]), [], false);
  const result = await orchestrator.executeImageJob(state, referenced);
  const asset = result.assets[0];
  assertCheck(scope, result.job.status === "completed", "a photoreal job completes through the relay", result.job.error);
  assertCheck(
    scope,
    asset?.backend === "photoreal" && asset.sceneCategory === "portrait" && asset.dataUrl === outputDataUrl && asset.mimeType === "image/png",
    "the asset is stamped photoreal + its category and carries the rendered PNG as a data URL",
    `${asset?.backend} ${asset?.sceneCategory} ${asset?.dataUrl?.slice(0, 40)}`
  );
  const payload = result.job.providerPayload;
  assertCheck(
    scope,
    payload.provider === "photoreal" &&
      payload.routing?.backend === "photoreal" &&
      payload.sceneCategory === "portrait" &&
      payload.mode === "provider-result" &&
      typeof payload.promptId === "string" &&
      !("workflow" in payload) &&
      !JSON.stringify(payload).includes(outputPng.toString("base64")),
    "the result payload keeps provider/routing, records the prompt id and holds neither the graph nor the image",
    JSON.stringify({ provider: payload.provider, mode: payload.mode, promptId: payload.promptId })
  );
  const queued = mock.requests("POST", "/prompt")[0]?.body?.prompt ?? {};
  const loadImages = Object.values(queued).filter((node) => node?.class_type === "LoadImage");
  const uploadedName = [...mock.uploads.keys()][0];
  assertCheck(
    scope,
    loadImages.length === 1 && uploadedName && loadImages[0].inputs?.image === uploadedName && mock.uploads.get(uploadedName)?.equals(referencePng),
    "the character's profile image is uploaded from the object store and wired into the graph",
    JSON.stringify({ loadImages: loadImages.map((node) => node.inputs?.image), uploadedName })
  );
  assertCheck(scope, JSON.stringify(queued).includes("<image1>"), "the submitted graph's prompt names <image1>");

  // The negative and the CFG as they reach ComfyUI: a job keeps the negative it was planned with, so the clean-up has
  // to happen on the way out, and the Turbo presets sample at CFG 1 whatever the setting says.
  mock.reset();
  const guided = createState(seed.seedState, {
    routing: { mode: "photoreal" },
    photorealOverrides: { baseUrl: mock.baseUrl, proxyUrl: `${server.baseUrl}/photoreal/comfyui`, timeoutSeconds: 60, cfg: 4, workflowPreset: "qwen_image_2_1", referenceWorkflowPreset: "qwen_image_2_1" }
  });
  const guidedJob = orchestrator.planImageJob(guided, "turn_relay", toCue(MIXED_CUES[0]), [], false);
  const steered = { ...guidedJob, negativePrompt: `${guidedJob.negativePrompt}, clothes, shirt, pleated skirt` };
  const steeredResult = await orchestrator.executeImageJob(guided, steered);
  const steeredGraph = mock.requests("POST", "/prompt")[0]?.body?.prompt ?? {};
  const steeredEncoder = Object.values(steeredGraph).find((node) => node?.class_type === "TextEncodeQwenImage21")?.inputs ?? {};
  const steeredSampler = Object.values(steeredGraph).find((node) => node?.class_type === "KSampler")?.inputs ?? {};
  assertCheck(
    scope,
    steeredResult.job.status === "completed" &&
      typeof steeredEncoder.negative_prompt === "string" &&
      steeredEncoder.negative_prompt === photoPrompt.sanitizePhotorealNegativePrompt(steered.negativePrompt) &&
      !/\b(?:clothes|shirt|skirt)\b/iu.test(steeredEncoder.negative_prompt) &&
      steeredSampler.cfg === 4,
    "a queued job's negative reaches ComfyUI exactly as the one negative rule leaves it (no body-covering clothing at CFG 4), and a non-Turbo preset keeps the user's CFG",
    `${steeredResult.job.error ?? ""} ${JSON.stringify(steeredEncoder.negative_prompt)} cfg=${steeredSampler.cfg}`
  );
  mock.reset();
  const turbo = createState(seed.seedState, {
    routing: { mode: "photoreal" },
    photorealOverrides: { baseUrl: mock.baseUrl, proxyUrl: `${server.baseUrl}/photoreal/comfyui`, timeoutSeconds: 60, cfg: 4 }
  });
  const turboJob = orchestrator.planImageJob(turbo, "turn_relay", toCue(MIXED_CUES[0]), [], false);
  const turboResult = await orchestrator.executeImageJob(turbo, turboJob);
  const turboGraph = mock.requests("POST", "/prompt")[0]?.body?.prompt ?? {};
  const turboGuider = Object.values(turboGraph).find((node) => node?.class_type === "CFGGuider")?.inputs ?? {};
  assertCheck(
    scope,
    turboResult.job.status === "completed" && turboJob.providerPayload.presetId === "qwen_image_2_1_turbo" && turboJob.providerPayload.cfg === 1 && turboGuider.cfg === 1,
    "with the CFG setting at 4, a Qwen-Image 2.1 Turbo cut is planned and sent at CFG 1",
    `${turboJob.providerPayload.presetId} planned=${turboJob.providerPayload.cfg} sent=${turboGuider.cfg} ${turboResult.job.error ?? ""}`
  );

  mock.reset();
  const behind = orchestrator.planImageJob(state, "turn_relay", { ...toCue(MIXED_CUES[0]), frame: FRAMES.upper_body_behind }, [], false);
  const behindResult = await orchestrator.executeImageJob(state, behind);
  assertCheck(scope, behindResult.job.status === "completed" && mock.uploads.size === 0, "a from-behind cut uploads no face", `uploads=${mock.uploads.size} ${behindResult.job.error ?? ""}`);

  // A cancel while ComfyUI is still working: the job is canceled, not failed, and ComfyUI is told to stop.
  mock.reset();
  mock.setBehaviour("never_completes");
  const controller = new AbortController();
  const pending = orchestrator.executeImageJob(state, orchestrator.planImageJob(state, "turn_relay", toCue(MIXED_CUES[2]), [], false), { signal: controller.signal });
  await waitFor(() => mock.requests("POST", "/prompt").length > 0, 10_000);
  controller.abort();
  const canceled = await pending;
  assertCheck(scope, canceled.job.status === "canceled" && canceled.job.error === undefined, "aborting a running photoreal job cancels it", `${canceled.job.status} ${canceled.job.error ?? ""}`);
  const interrupted = await waitFor(() => mock.requests("POST", "/interrupt").length > 0, 10_000);
  assertCheck(scope, Boolean(interrupted), "the relay interrupts ComfyUI when the client goes away");
  mock.reset();

  // The creation screen's candidates, end to end.
  const prompt = "A Korean woman in her late twenties with long straight black hair, wearing a cream knit cardigan";
  const jobs = studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "photoreal", prompt, variant: "portrait", count: 2 });
  const shown = [];
  const run = await studio.runCharacterProfileTestJobs(state, jobs, { onImage: (image, index) => shown.push({ image, index }) });
  assertCheck(
    scope,
    run.errors.length === 0 && run.assets.length === 2 && shown.map((entry) => entry.index).join(",") === "0,1" && run.assets.every((image) => image.backend === "photoreal" && image.dataUrl === outputDataUrl),
    "profile candidates render through the relay and are reported one by one",
    `${run.errors.join(" / ")} assets=${run.assets.length}`
  );
  assertCheck(scope, mock.uploads.size === 0, "portrait candidates are not drawn from the existing profile image");
  const seeds = mock.requests("POST", "/prompt").map((request) => findInputValue(request.body?.prompt, ["seed", "noise_seed"]));
  assertCheck(scope, seeds.length === 2 && seeds[0] !== seeds[1] && seeds.every((value) => value === jobs[seeds.indexOf(value)].providerPayload.seed), "each candidate is rendered with its own planned seed", JSON.stringify(seeds));
  const record = studio.createProfileImageRecord(run.assets[0], { backend: "photoreal", objectKey: `${SIMULATION_ID}/candidate.png`, prompt });
  assertCheck(
    scope,
    record.width === run.assets[0].providerMetadata.width && record.seed === jobs[0].providerPayload.seed && record.workflowPreset === jobs[0].providerPayload.presetId,
    "the picked candidate becomes a profile record with its size, seed and preset",
    JSON.stringify(record)
  );

  mock.reset();
  const check = studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "photoreal", prompt, variant: "reference_check", count: 1 });
  const checkRun = await studio.runCharacterProfileTestJobs(state, check);
  assertCheck(
    scope,
    checkRun.errors.length === 0 && checkRun.assets.length === 1 && [...mock.uploads.values()].some((bytes) => bytes.equals(referencePng)),
    "the face check sends the profile image as the reference",
    checkRun.errors.join(" / ")
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Local GPU hand-over
// ---------------------------------------------------------------------------------------------------------------

async function evaluateLocalGpuCoordinator({ seed, gpu }) {
  const scope = "gpu";
  const relayBase = "http://127.0.0.1:65530/photoreal/comfyui";
  const coordinated = createState(seed.seedState, {
    routing: { mode: "photoreal" },
    photorealOverrides: { vramHandoff: "auto", proxyUrl: relayBase, baseUrl: "http://127.0.0.1:8188" }
  });
  coordinated.llm = { ...coordinated.llm, enabled: true, provider: "ollama", baseUrl: "http://127.0.0.1:11434/v1", model: "qwen3:14b" };
  coordinated.imageTagLlm = { ...coordinated.imageTagLlm, enabled: false, provider: "ollama", baseUrl: "http://127.0.0.1:11434/v1", model: "qwen3:4b" };

  const variant = (patch) => {
    const next = structuredClone(coordinated);
    patch(next);
    return next;
  };
  assertCheck(scope, gpu.shouldCoordinateLocalGpu(coordinated) === true, "local Ollama + local ComfyUI through the relay with handoff auto coordinates");
  const negatives = [
    ["handoff off", (state) => (state.photoreal.vramHandoff = "off")],
    ["photoreal off", (state) => (state.photoreal.enabled = false)],
    ["photoreal mock", (state) => (state.photoreal.requestMode = "mock")],
    ["remote ComfyUI", (state) => (state.photoreal.baseUrl = "http://192.168.0.10:8188")],
    ["remote Ollama", (state) => (state.llm.baseUrl = "http://192.168.0.20:11434/v1")],
    ["hosted LLM", (state) => (state.llm.provider = "deepseek")],
    ["LLM disabled", (state) => (state.llm.enabled = false)]
  ];
  for (const [label, patch] of negatives) {
    assertCheck(scope, gpu.shouldCoordinateLocalGpu(variant(patch)) === false, `no coordination: ${label}`);
  }
  assertCheck(
    scope,
    gpu.shouldCoordinateLocalGpu(variant((state) => {
      state.llm.provider = "deepseek";
      state.imageTagLlm.enabled = true;
    })) === true,
    "a local Ollama tag model alone is enough to coordinate"
  );
  assertCheck(scope, gpu.shouldCoordinateLocalGpu(variant((state) => (state.photoreal.baseUrl = ""))) === true, "an empty ComfyUI URL means the loopback default");

  const roots = [
    ["http://127.0.0.1:11434/v1", "http://127.0.0.1:11434"],
    ["http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434"],
    ["http://localhost:11434/api", "http://localhost:11434"],
    ["http://127.0.0.1:11434/api/chat", "http://127.0.0.1:11434"],
    ["http://127.0.0.1:11434/v1/chat/completions", "http://127.0.0.1:11434"],
    ["http://[::1]:11434", "http://[::1]:11434"]
  ];
  for (const [baseUrl, expected] of roots) {
    const targets = gpu.readLocalOllamaTargets(variant((state) => (state.llm.baseUrl = baseUrl)));
    assertCheck(scope, targets.length === 1 && targets[0].apiRoot === expected && targets[0].model === "qwen3:14b", `Ollama root of ${baseUrl} is ${expected}`, JSON.stringify(targets));
  }
  const two = gpu.readLocalOllamaTargets(variant((state) => (state.imageTagLlm.enabled = true)));
  assertCheck(scope, two.length === 2 && two[1].model === "qwen3:4b", "narrative and tag models are both unloaded", JSON.stringify(two));
  const deduped = gpu.readLocalOllamaTargets(variant((state) => {
    state.imageTagLlm.enabled = true;
    state.imageTagLlm.model = "qwen3:14b";
    state.imageTagLlm.baseUrl = "http://127.0.0.1:11434/v1/";
  }));
  assertCheck(scope, deduped.length === 1, "the same model on the same server is unloaded once", JSON.stringify(deduped));

  const calls = [];
  let ollamaMode = "ok";
  const stub = async (url, init) => {
    const body = typeof init?.body === "string" ? safeJson(init.body) : undefined;
    if (url.endsWith("/api/generate")) {
      calls.push({ kind: "unload", model: body?.model, keepAlive: body?.keep_alive, url });
      if (ollamaMode === "throw") {
        throw new TypeError("fetch failed");
      }
      return new Response(JSON.stringify({ done: true, done_reason: "unload" }), { status: ollamaMode === "error" ? 500 : 200 });
    }
    if (url === `${relayBase}/free`) {
      calls.push({ kind: "free" });
      return new Response("{}", { status: 200 });
    }
    return undefined;
  };
  const unloadFetch = (input, init) => stubOrThrow(stub, input, init);

  const unloaded = await gpu.unloadOllamaModels(variant((state) => (state.imageTagLlm.enabled = true)), unloadFetch);
  assertCheck(
    scope,
    sameList(unloaded, ["qwen3:14b", "qwen3:4b"]) && calls.every((call) => call.kind === "unload" && call.keepAlive === 0 && call.url === "http://127.0.0.1:11434/api/generate"),
    "unloadOllamaModels posts {model, keep_alive: 0} to /api/generate per model",
    JSON.stringify(calls)
  );
  calls.length = 0;
  ollamaMode = "throw";
  assertCheck(scope, (await gpu.unloadOllamaModels(coordinated, unloadFetch)).length === 0, "an unreachable Ollama unloads nothing and does not throw");
  ollamaMode = "error";
  assertCheck(scope, (await gpu.unloadOllamaModels(coordinated, unloadFetch)).length === 0, "an Ollama error answer is not reported as unloaded");
  ollamaMode = "ok";
  calls.length = 0;

  await withFetchStub(stub, async () => {
    const handoffs = [];
    const coordinator = gpu.createLocalGpuCoordinator({ fetchImpl: unloadFetch, onHandoff: (event) => handoffs.push(event) });
    const task = (label, ms = 0) => async () => {
      calls.push({ kind: "start", label });
      await delay(ms);
      calls.push({ kind: "end", label });
      return label;
    };
    assertCheck(scope, coordinator.holder === undefined, "a new coordinator has no holder");
    await coordinator.run("llm", coordinated, task("turn 1"));
    // A fresh coordinator cannot know what an earlier session left on the card, so the first local turn frees the
    // state's ComfyUI once (the mirror of a first render unloading Ollama).
    assertCheck(scope, calls.map((call) => call.kind).join(",") === "free,start,end", "the first LLM turn frees the state's ComfyUI once", JSON.stringify(calls));
    calls.length = 0;
    await coordinator.run("image", coordinated, task("image 1"));
    assertCheck(
      scope,
      calls.map((call) => call.kind).join(",") === "unload,start,end" && coordinator.holder === "image",
      "LLM → image: the Ollama model is unloaded before the image job starts",
      JSON.stringify(calls)
    );
    calls.length = 0;
    await coordinator.run("image", coordinated, task("image 2"));
    assertCheck(scope, calls.map((call) => call.kind).join(",") === "start,end", "image → image: no hand-over", JSON.stringify(calls));
    calls.length = 0;
    await coordinator.run("llm", coordinated, task("turn 2"));
    assertCheck(
      scope,
      calls.map((call) => call.kind).join(",") === "free,start,end" && coordinator.holder === "llm",
      "image → LLM: ComfyUI is freed before the turn starts",
      JSON.stringify(calls)
    );
    assertCheck(
      scope,
      handoffs.length === 3 &&
        handoffs[0].from === "image" && handoffs[0].to === "llm" &&
        handoffs[1].from === "llm" && handoffs[1].to === "image" &&
        handoffs[2].from === "image" && handoffs[2].to === "llm" &&
        handoffs.every((event) => typeof event.detail === "string" && event.detail),
      "each hand-over is reported once with from/to and a message",
      JSON.stringify(handoffs)
    );

    // FIFO and no overlap: queued without awaiting, a slow first task must finish before the next one starts.
    calls.length = 0;
    const order = await Promise.all([
      coordinator.run("llm", coordinated, task("A", 60)),
      coordinator.run("llm", coordinated, task("B", 0)),
      coordinator.run("llm", coordinated, task("C", 10))
    ]);
    const sequence = calls.filter((call) => call.kind === "start" || call.kind === "end").map((call) => `${call.kind}:${call.label}`);
    assertCheck(
      scope,
      sameList(order, ["A", "B", "C"]) && sequence.join(",") === "start:A,end:A,start:B,end:B,start:C,end:C",
      "tasks run one at a time, in call order",
      sequence.join(",")
    );

    // A failed task releases the lock.
    calls.length = 0;
    const failed = await coordinator.run("image", coordinated, async () => {
      throw new Error("boom");
    }).then(() => "resolved", (error) => error.message);
    const after = await coordinator.run("image", coordinated, task("after failure"));
    assertCheck(scope, failed === "boom" && after === "after failure", "a rejected task rejects its own run and releases the lock for the next");

    // A hand-over that fails never blocks the task.
    calls.length = 0;
    ollamaMode = "throw";
    await coordinator.run("llm", coordinated, task("turn 3"));
    const survived = await coordinator.run("image", coordinated, task("image after failed unload"));
    assertCheck(scope, survived === "image after failed unload" && calls.some((call) => call.kind === "unload"), "an unload that fails still lets the image job run");
    ollamaMode = "ok";

    // Not coordinating: a plain pass-through, no lock and no calls.
    calls.length = 0;
    const plain = variant((state) => (state.photoreal.vramHandoff = "off"));
    const holderBefore = coordinator.holder;
    const passthrough = await coordinator.run("image", plain, task("plain"));
    assertCheck(
      scope,
      passthrough === "plain" && calls.map((call) => call.kind).join(",") === "start,end" && coordinator.holder === holderBefore,
      "without coordination run() just runs the task and leaves the holder alone",
      JSON.stringify(calls)
    );
    const thrown = await coordinator.run("llm", plain, () => {
      throw new Error("sync throw");
    }).then(() => "resolved", (error) => error.message);
    assertCheck(scope, thrown === "sync throw", "a synchronous throw comes back as a rejection");
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Creation screen profile tests (dry-run)
// ---------------------------------------------------------------------------------------------------------------

async function evaluateProfileStudio({ seed, studio, workflows }) {
  const scope = "studio";
  const state = createState(seed.seedState, { routing: { mode: "novelai" } });
  state.novelAi = { ...state.novelAi, characterReferenceEnabled: true };

  const naiJobs = studio.planCharacterProfileTestJobs(state, {
    characterId: "char_yuna",
    backend: "novelai",
    prompt: "1girl, short pink hair, green eyes, white blouse",
    negativePrompt: "glasses",
    variant: "portrait",
    count: 3
  });
  const naiCaption = naiJobs[0]?.providerPayload.characterPrompts?.[0];
  assertCheck(scope, naiJobs.length === 3 && naiJobs.every((job) => job.backend === "novelai" && job.turnId === "profile_test"), "three NovelAI candidates are planned", `${naiJobs.length}`);
  assertCheck(scope, new Set(naiJobs.map((job) => job.providerPayload.seed)).size === 3, "each candidate has its own seed");
  assertCheck(
    scope,
    /short pink hair/u.test(naiCaption?.prompt ?? "") && !/long black hair|pleated skirt|white knit sweater/u.test(naiCaption?.prompt ?? "") && naiCaption?.negativePrompt === "glasses",
    "the typed prompt is the whole description: saved identity and default outfit are not re-injected",
    naiCaption?.prompt
  );
  assertCheck(
    scope,
    /\b1girl\b/u.test(naiJobs[0].prompt) && /\bsolo\b/u.test(naiJobs[0].prompt) && /simple background/u.test(naiJobs[0].prompt) && /upper body/u.test(naiJobs[0].prompt),
    "the NovelAI portrait base caption has the subject, solo, a plain background and the crop",
    naiJobs[0].prompt
  );
  assertCheck(scope, naiJobs.every((job) => job.providerPayload.cue.kind === "profile" && job.routing.signals.includes("kind:profile")), "the cue kind is 'profile'");
  assertCheck(scope, !("characterReference" in naiJobs[0].providerPayload), "a portrait candidate never uses the existing profile as Precise Reference");

  const prose = "A Korean woman in her late twenties with shoulder-length black hair, wearing a cream knit cardigan";
  const proseJobs = studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "photoreal", prompt: prose, variant: "portrait", count: 1 });
  const proseJob = proseJobs[0];
  assertCheck(
    scope,
    proseJob.backend === "photoreal" && proseJob.prompt.includes("A Korean woman in her late twenties with shoulder-length black hair") && !NAI_TOKEN_PATTERN.test(proseJob.prompt),
    "a prose photoreal prompt is used as the appearance text",
    proseJob.prompt.slice(0, 160)
  );
  assertCheck(scope, (proseJob.providerPayload.photorealReferences ?? []).length === 0 && !/<image\d>/u.test(proseJob.prompt), "a portrait candidate is not drawn from the existing photoreal profile");
  const tagJob = studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "photoreal", prompt: "1girl, long black hair, white sweater, smile", variant: "portrait", count: 1 })[0];
  assertCheck(
    scope,
    /\ban adult woman\b/iu.test(tagJob.prompt) && /white sweater/u.test(tagJob.prompt) && !/1girl/u.test(tagJob.prompt),
    "a tag-list photoreal prompt is translated like any cut (adult noun, no Danbooru tokens)",
    tagJob.prompt.slice(0, 300)
  );
  const maleJob = studio.planCharacterProfileTestJobs(state, { characterId: "char_minho", backend: "photoreal", prompt: "A tall man in his thirties with short brown hair", variant: "portrait", count: 1 })[0];
  assertCheck(scope, /\b1boy\b/u.test(maleJob.providerPayload.novelAiPrompt ?? ""), "the subject tag follows the described gender", maleJob.providerPayload.novelAiPrompt);

  const check = studio.planCharacterProfileTestJobs(
    { ...state, photoreal: { ...state.photoreal, faceReferenceEnabled: false } },
    { characterId: "char_yuna", backend: "photoreal", prompt: prose, variant: "reference_check", count: 1 }
  )[0];
  assertCheck(
    scope,
    check.backend === "photoreal" &&
      check.providerPayload.cue.frame.shot === "cowboy_shot" &&
      check.providerPayload.photorealReferences?.[0]?.characterId === "char_yuna" &&
      /<image1>/u.test(check.prompt) &&
      /sitting/u.test(check.prompt),
    "the face check plans a different pose with the profile as <image1>, even with face references switched off globally",
    check.prompt.slice(0, 200)
  );
  assertThrowsKorean(scope, () => studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "novelai", prompt: prose, variant: "reference_check", count: 1 }), "the face check refuses NovelAI");
  assertThrowsKorean(scope, () => studio.planCharacterProfileTestJobs(state, { characterId: "char_minho", backend: "photoreal", prompt: prose, variant: "reference_check", count: 1 }), "the face check refuses a character without a photoreal profile");
  assertThrowsKorean(
    scope,
    () =>
      studio.planCharacterProfileTestJobs(
        { ...state, photoreal: { ...state.photoreal, workflowPreset: "z_image_turbo", referenceWorkflowPreset: "z_image_turbo" } },
        { characterId: "char_yuna", backend: "photoreal", prompt: prose, variant: "reference_check", count: 1 }
      ),
    "the face check refuses when no configured preset takes references"
  );
  assertThrowsKorean(scope, () => studio.planCharacterProfileTestJobs(state, { characterId: "char_nobody", backend: "photoreal", prompt: prose, variant: "portrait", count: 1 }), "an unknown character is refused");

  // The safety floor still applies, but a 실사 request it locks is refused instead of being planned on NovelAI: the
  // panel says a photoreal test is free, and quietly running it on NovelAI spent Anlas (review finding).
  const SAFETY_REFUSAL = "이 프롬프트는 실사 안전 기준(미성년·노출·성적 표현)에 걸려 실사로 생성할 수 없습니다. NovelAI로 바꾸어 테스트하세요.";
  const UNREADABLE_REFUSAL = "이 프롬프트에는 확인할 수 없는 문자(영문 외 문자)가 있어 실사로 생성할 수 없습니다. 영어로 적거나 NovelAI로 바꾸어 테스트하세요.";
  const refusedPrompts = [
    ["A 14-year-old Korean girl in a bikini", SAFETY_REFUSAL],
    ["A naked woman lying on a bed", SAFETY_REFUSAL],
    ["A teenage girl in lingerie, lying on a bed", SAFETY_REFUSAL],
    ["A little girl in a school uniform with a short skirt", SAFETY_REFUSAL],
    ["1girl, long black hair, student, swimsuit, beach", SAFETY_REFUSAL],
    ["1girl, school uniform, cleavage", SAFETY_REFUSAL],
    ["1girl, petite, teen, white dress", SAFETY_REFUSAL],
    ["一个十四岁的韩国少女", UNREADABLE_REFUSAL]
  ];
  const refusedVariants = [
    ...refusedPrompts.map(([prompt, expected]) => ({ prompt, expected, variant: "portrait" })),
    // The face check runs the typed prompt too (with the profile as reference): same refusal, no NovelAI fallback.
    { prompt: "A naked woman lying on a bed", expected: SAFETY_REFUSAL, variant: "reference_check" }
  ];
  for (const { prompt, expected, variant } of refusedVariants) {
    let message;
    let planned;
    try {
      planned = studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "photoreal", prompt, variant, count: 4 });
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
    assertCheck(
      scope,
      planned === undefined && message === expected,
      `a 실사 ${variant} test the floor locks is refused, never planned on NovelAI: ${JSON.stringify(prompt)}`,
      message ?? planned?.map((job) => `${job.backend}/${job.routing?.locked}`).join(",")
    );
  }
  const lockedOnNovelAi = studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "novelai", prompt: "1girl, school uniform, cleavage", variant: "portrait", count: 2 });
  assertCheck(
    scope,
    lockedOnNovelAi.length === 2 && lockedOnNovelAi.every((job) => job.backend === "novelai"),
    "the same prompt asked of NovelAI is planned on NovelAI as before (the photo floor never blocks NovelAI)",
    lockedOnNovelAi.map((job) => job.backend).join(",")
  );
  const koreanJob = studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "photoreal", prompt: "검은 긴 머리의 20대 여성", variant: "portrait", count: 1 })[0];
  assertCheck(
    scope,
    koreanJob?.backend === "photoreal" && !HANGUL_PATTERN.test(koreanJob.prompt) && /an adult woman/iu.test(koreanJob.prompt),
    "a Korean 실사 prompt keeps its documented fallback: not refused, left out of the photo prompt (an adult woman)",
    koreanJob?.prompt.slice(0, 160)
  );
  const styled = { ...state, photoreal: { ...state.photoreal, stylePreset: "custom", stylePrompt: "A nude boudoir photograph. Teenage model, 16 years old." } };
  let styleMessage;
  try {
    studio.planCharacterProfileTestJobs(styled, { characterId: "char_yuna", backend: "photoreal", prompt: "A Korean woman in her late twenties", variant: "portrait", count: 1 });
  } catch (error) {
    styleMessage = error instanceof Error ? error.message : String(error);
  }
  assertCheck(
    scope,
    typeof styleMessage === "string" && /실사 안전 기준/u.test(styleMessage) && /스타일 문장/u.test(styleMessage) && /NovelAI 고정/u.test(styleMessage),
    "a clean prompt under a style the floor locks is refused too, and the message points at the style text",
    styleMessage
  );
  const tainted = structuredClone(state);
  tainted.visualProfiles = tainted.visualProfiles.map((profile) =>
    profile.characterId === "char_yuna"
      ? { ...profile, profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, prompt: "A 14-year-old Korean girl in a bikini" } } }
      : profile
  );
  assertThrowsKorean(
    scope,
    () => studio.planCharacterProfileTestJobs(tainted, { characterId: "char_yuna", backend: "photoreal", prompt: prose, variant: "reference_check", count: 2 }),
    "the face check refuses a profile image whose own prompt the floor locks (it is never used as a reference)"
  );
  const unrecorded = structuredClone(state);
  unrecorded.visualProfiles = unrecorded.visualProfiles.map((profile) =>
    profile.characterId === "char_yuna" ? { ...profile, profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, prompt: "" } } } : profile
  );
  let unrecordedMessage;
  try {
    studio.planCharacterProfileTestJobs(unrecorded, { characterId: "char_yuna", backend: "photoreal", prompt: prose, variant: "reference_check", count: 1 });
  } catch (error) {
    unrecordedMessage = error instanceof Error ? error.message : String(error);
  }
  assertCheck(
    scope,
    typeof unrecordedMessage === "string" && /프롬프트 기록이 없어/u.test(unrecordedMessage),
    "the face check refuses a profile image with no recorded prompt",
    unrecordedMessage
  );
  assertCheck(
    scope,
    studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "novelai", prompt: "1girl", variant: "portrait", count: 9 }).length === 4 &&
      studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "novelai", prompt: "1girl", variant: "portrait", count: 0 }).length === 1,
    "the candidate count is clamped to 1-4"
  );

  const dry = await studio.runCharacterProfileTestJobs({ ...state, photoreal: { ...state.photoreal, enabled: false } }, proseJobs.concat(proseJobs));
  assertCheck(scope, dry.assets.length === 0 && sameList(dry.errors, [PHOTOREAL_DISABLED_TEXT]), "a dry-run backend reports its setup message once", JSON.stringify(dry.errors));
  const aborted = new AbortController();
  aborted.abort();
  let called = 0;
  const stopped = await studio.runCharacterProfileTestJobs(state, naiJobs, { signal: aborted.signal, onImage: () => (called += 1) });
  assertCheck(scope, stopped.assets.length === 0 && stopped.errors.length === 0 && called === 0, "an aborted run stops before the first request");

  const size = workflows.resolvePhotorealOutputSize(state);
  const photorealAsset = {
    id: "asset_candidate",
    mimeType: "image/png",
    providerMetadata: { width: size.width, height: size.height, seed: 77, presetId: "qwen_image_2_1_turbo" }
  };
  const first = studio.createProfileImageRecord(photorealAsset, { backend: "photoreal", objectKey: `${SIMULATION_ID}/p1.png`, referenceObjectKey: "ignored", prompt: prose, negativePrompt: " " });
  const fromEmptyBox = studio.createProfileImageRecord(
    { ...photorealAsset, prompt: "A candid photograph. An adult woman with long black hair, smiling." },
    { backend: "photoreal", objectKey: `${SIMULATION_ID}/p2.png`, prompt: "  " }
  );
  assertCheck(
    scope,
    fromEmptyBox.prompt === "A candid photograph. An adult woman with long black hair, smiling." && first.prompt === prose,
    "a pick made from an empty prompt box records the prompt the image was rendered from (the typed one otherwise), so the record can vouch for it",
    JSON.stringify(fromEmptyBox.prompt)
  );
  const second = studio.createProfileImageRecord(photorealAsset, { backend: "photoreal", objectKey: `${SIMULATION_ID}/p1.png`, prompt: prose });
  assertCheck(
    scope,
    /^profile_/u.test(first.assetId) &&
      first.assetId !== second.assetId &&
      first.width === size.width &&
      first.seed === 77 &&
      first.workflowPreset === "qwen_image_2_1_turbo" &&
      !("referenceObjectKey" in first) &&
      !("negativePrompt" in first) &&
      !("modelPreset" in first) &&
      typeof first.selectedAt === "string",
    "a photoreal profile record: fresh id per pick, size/seed/preset from the metadata, no NovelAI-only fields",
    JSON.stringify(first)
  );
  const novelAiAsset = { id: "asset_nai_candidate", mimeType: "image/png", providerMetadata: { width: 832, height: 1216, modelPreset: "NAID4.5F", parameters: { seed: 99 } } };
  const naiRecord = studio.createProfileImageRecord(novelAiAsset, {
    backend: "novelai",
    objectKey: `${SIMULATION_ID}/n1.png`,
    referenceObjectKey: `${SIMULATION_ID}/n1_ref.png`,
    prompt: "1girl",
    negativePrompt: "glasses"
  });
  assertCheck(
    scope,
    naiRecord.modelPreset === "NAID4.5F" && naiRecord.seed === 99 && naiRecord.referenceObjectKey === `${SIMULATION_ID}/n1_ref.png` && naiRecord.negativePrompt === "glasses" && !("workflowPreset" in naiRecord),
    "a NovelAI profile record keeps the model preset, the NovelAI seed and the letterboxed reference key",
    JSON.stringify(naiRecord)
  );

  // The builder persists the bytes first and then builds the record, so the record must name the file it stored and
  // carry the decoded size (the photoreal reference is head-cropped from it), not the size the request asked for.
  const pinned = studio.createProfileImageRecord(photorealAsset, {
    backend: "photoreal",
    objectKey: `${SIMULATION_ID}/profile_pick.png`,
    prompt: prose,
    assetId: "profile_pick",
    width: 1024.4,
    height: 1024
  });
  assertCheck(
    scope,
    pinned.assetId === "profile_pick" && pinned.width === 1024 && pinned.height === 1024 && pinned.seed === 77,
    "a record takes the id its bytes were stored under, and the decoded size over the metadata",
    JSON.stringify(pinned)
  );
  const unsized = studio.createProfileImageRecord(photorealAsset, {
    backend: "photoreal",
    objectKey: `${SIMULATION_ID}/profile_unsized.png`,
    prompt: prose,
    width: 0,
    height: Number.NaN
  });
  assertCheck(
    scope,
    unsized.width === size.width && unsized.height === size.height && /^profile_/u.test(unsized.assetId),
    "an unusable decoded size falls back to the metadata, and a missing id is minted",
    JSON.stringify(unsized)
  );

  // Progress: every candidate that ran reports as settled, a failed one included, so the panel's n/N never stalls.
  const settledCalls = [];
  await studio.runCharacterProfileTestJobs({ ...state, photoreal: { ...state.photoreal, enabled: false } }, proseJobs.concat(proseJobs), {
    onSettled: (settled, total) => settledCalls.push(`${settled}/${total}`)
  });
  assertCheck(scope, sameList(settledCalls, ["1/2", "2/2"]), "failed candidates still settle, in order", JSON.stringify(settledCalls));
  const abortedSettled = [];
  await studio.runCharacterProfileTestJobs(state, naiJobs, { signal: aborted.signal, onSettled: (settled) => abortedSettled.push(settled) });
  assertCheck(scope, abortedSettled.length === 0, "a cancelled run settles nothing");

  // The creation screen tests a character that only exists in a draft. The preview state keeps the DRAFT ids, so the
  // panel can name the character it edits; createStateFromDraft still mints new ones.
  const draft = createStudioDraft(seed.seedState);
  const preview = seed.createDraftPreviewState(draft);
  assertCheck(
    scope,
    sameList(
      preview.characters.map((character) => character.id),
      ["draft_char_mina", "draft_char_jun"]
    ) &&
      sameList(
        preview.visualProfiles.map((profile) => profile.characterId),
        ["draft_char_mina", "draft_char_jun"]
      ) &&
      preview.modules.filter((module) => module.kind === "character_prompt").every((module) => module.characterId?.startsWith("draft_char_")),
    "a draft preview state keeps the draft's character ids on characters, visual profiles and character modules",
    JSON.stringify(preview.characters.map((character) => character.id))
  );
  const created = seed.createStateFromDraft(draft);
  assertCheck(
    scope,
    created.characters.every((character) => !character.id.startsWith("draft_")) && created.visualProfiles.every((profile) => !profile.characterId.startsWith("draft_")),
    "createStateFromDraft still mints the simulation's own character ids"
  );
  const previewJob = studio.planCharacterProfileTestJobs(preview, {
    characterId: "draft_char_jun",
    backend: "photoreal",
    prompt: "A tall man in his thirties with short brown hair",
    variant: "portrait",
    count: 1
  })[0];
  assertCheck(
    scope,
    previewJob?.backend === "photoreal" &&
      sameList(previewJob.providerPayload.cue?.characters, ["draft_char_jun"]) &&
      previewJob.prompt.includes("A tall man in his thirties with short brown hair"),
    "the studio plans a draft character's candidates against the preview state",
    previewJob?.prompt.slice(0, 200)
  );
  const previewCheck = studio.planCharacterProfileTestJobs(preview, {
    characterId: "draft_char_mina",
    backend: "photoreal",
    prompt: "",
    variant: "reference_check",
    count: 2
  });
  const previewReference = previewCheck[0]?.providerPayload.photorealReferences?.[0];
  assertCheck(
    scope,
    previewCheck.length === 2 &&
      previewReference?.characterId === "draft_char_mina" &&
      previewReference?.objectKey === "draft_sim_eval/profile_mina.png" &&
      previewReference?.width === 1024 &&
      /<image1>/u.test(previewCheck[0].prompt),
    "a face check on a draft character uses the profile picked in the draft (key under the provisional id, with its size)",
    JSON.stringify(previewReference)
  );
}

/** A builder draft with two characters, one of whom already has a photoreal profile picked in the creation screen. */
function createStudioDraft(seedState) {
  const updatedAt = "2026-10-14T00:00:00.000Z";
  const character = (id, name, extra = {}) => ({
    id,
    name,
    role: "lead",
    summary: `${name} summary`,
    relationship: "",
    currentMood: "",
    visualPrompt: "1girl, black hair",
    negativeVisualPrompt: "",
    defaultOutfitPrompt: "",
    outfitPrompts: {},
    expressionPrompts: {},
    defaultSafetyLevel: "safe",
    ...extra
  });
  const characters = [
    character("draft_char_mina", "Mina", {
      photorealPrompt: "A Korean woman in her late twenties with shoulder-length black hair",
      profileImages: {
        photoreal: {
          backend: "photoreal",
          assetId: "profile_mina",
          objectKey: "draft_sim_eval/profile_mina.png",
          width: 1024,
          height: 1024,
          prompt: "A Korean woman in her late twenties with shoulder-length black hair",
          selectedAt: updatedAt
        }
      }
    }),
    character("draft_char_jun", "Jun", { visualPrompt: "1boy, short brown hair" })
  ];
  const createModule = (id, kind, title, extra = {}) => ({
    id,
    simulationId: "draft_simulation",
    kind,
    title,
    body: `${title} 본문`,
    enabled: true,
    priority: 80,
    activationTags: [kind],
    tokenPolicy: kind === "main_prompt" ? "always" : "rag",
    version: 1,
    updatedAt,
    ...extra
  });
  return {
    promptMode: "simulation",
    contentRating: "general",
    title: "프로필 스튜디오 초안",
    description: "",
    mainPrompt: "플레이어의 선택을 중심으로 진행한다.",
    characterName: "Mina",
    characterRole: "lead",
    characterSummary: "",
    characterRelationship: "",
    characterMood: "",
    worldLore: "",
    startSituationPrompt: "첫 장면.",
    visualPrompt: "",
    negativeVisualPrompt: "",
    realtimeImageEnabled: true,
    characters,
    imageScenePresets: [],
    modules: [
      createModule("draft_main", "main_prompt", "메인 규칙"),
      ...characters.map((entry) =>
        createModule(`draft_module_${entry.id}`, "character_prompt", `캐릭터: ${entry.name}`, { parentId: "draft_main", characterId: entry.id })
      )
    ],
    imageProfile: { ...structuredClone(seedState.imageProfile), id: "draft_image_profile", simulationId: "draft_simulation" },
    neuralMap: { ...seedState.neuralMap },
    relationshipMap: structuredClone(seedState.relationshipMap),
    llm: { ...seedState.llm },
    novelAi: structuredClone(seedState.novelAi),
    photoreal: { ...structuredClone(seedState.photoreal), enabled: true }
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Turn trace
// ---------------------------------------------------------------------------------------------------------------

async function evaluateTurnTraceMetrics({ seed, engine }) {
  const scope = "trace";
  const state = createState(seed.seedState, { routing: { mode: "mixed" }, cadence: "balanced" });
  state.llm = { ...state.llm, enabled: true, provider: "ollama", baseUrl: `${STUB_OLLAMA_ROOT}/v1`, model: "stub-model", apiKey: "", temperature: 0.7, maxTokens: 2000, contextTokens: 16384 };
  state.imageTagLlm = { ...state.imageTagLlm, enabled: false };
  const narrative =
    "유나는 창가 자리에 앉아 따뜻한 커피잔을 두 손으로 감쌌다. 맞은편의 민호가 웃으며 오늘 있었던 일을 이야기했고, 유나는 고개를 끄덕이며 미소 지었다. 오후의 햇살이 테이블 위로 부드럽게 내려앉았다.";
  const annotation = {
    image_cues: [
      {
        kind: "dialogue_face",
        placement: "after",
        anchor_text: "유나는 고개를 끄덕이며 미소 지었다.",
        should_generate: true,
        frame: { shot: "upper_body", viewpoint: "front", visible_regions: ["head", "torso"] },
        characters: ["char_yuna"],
        base_tags: ["1girl", "solo", "upper body", "indoors", "cafe"],
        character_prompts: [{ character_id: "char_yuna", prompt: "smile, holding cup, looking at viewer" }]
      },
      {
        kind: "interaction",
        placement: "after",
        anchor_text: "맞은편의 민호가 웃으며 오늘 있었던 일을 이야기했고",
        should_generate: true,
        frame: { shot: "upper_body", viewpoint: "front", visible_regions: ["head", "torso"] },
        characters: ["char_yuna", "char_minho"],
        base_tags: ["1girl", "1boy", "upper body", "indoors", "cafe"],
        character_prompts: [
          { character_id: "char_yuna", prompt: "smile, sitting" },
          { character_id: "char_minho", prompt: "sitting, laughing" }
        ]
      }
    ],
    state_events: []
  };
  const llmCalls = [];
  const stub = async (url, init) => {
    if (url !== `${STUB_OLLAMA_ROOT}/api/chat`) {
      return undefined;
    }
    const body = safeJson(String(init?.body ?? "{}")) ?? {};
    const system = String(body.messages?.[0]?.content ?? "");
    // Only the annotation pass is shown the narrative it annotates; the narrative pass is writing it.
    const isAnnotation = system.includes("따뜻한 커피잔을 두 손으로 감쌌다");
    llmCalls.push(isAnnotation ? "annotation" : "narrative");
    const content = isAnnotation ? JSON.stringify(annotation) : JSON.stringify({ assistant_text: narrative, memory_events: [] });
    return new Response(JSON.stringify({ message: { role: "assistant", content }, done: true }), { status: 200, headers: { "content-type": "application/json" } });
  };
  await withFetchStub(stub, async () => {
    const result = await engine.runSimulationTurn(state, "창가 자리에 앉아 민호와 커피를 마신다.", false, { deferMemoryIngest: true });
    const jobs = result.imageJobs ?? [];
    const photoreal = jobs.filter((job) => job.backend === "photoreal").length;
    assertCheck(scope, llmCalls.includes("narrative") && llmCalls.includes("annotation"), "the turn ran the narrative and the annotation pass against the stub", llmCalls.join(","));
    assertCheck(
      scope,
      jobs.length === 2 && photoreal === 1 && jobs.find((job) => job.backend === "novelai")?.routing?.category === "group",
      "a mixed-mode turn plans one photoreal portrait and one NovelAI group cut",
      jobs.map((job) => `${job.backend}:${job.routing?.category}`).join(", ")
    );
    assertCheck(
      scope,
      result.turnTrace.metrics.photorealJobCount === photoreal && result.turnTrace.metrics.imageJobCount === jobs.length,
      "TurnTrace.metrics.photorealJobCount counts the turn's photoreal jobs",
      JSON.stringify({ photorealJobCount: result.turnTrace.metrics.photorealJobCount, imageJobCount: result.turnTrace.metrics.imageJobCount })
    );
    assertCheck(scope, jobs.every((job) => typeof job.providerPayload.cue?.kind === "string"), "the annotation kind reaches every planned job");
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Fixtures and harness
// ---------------------------------------------------------------------------------------------------------------

/**
 * A custom ComfyUI workflow sends its own literal text with every cut, and nothing classified it: "A fully naked 14
 * year old girl. %prompt%" made every clean cut that picture (red team, channel A). The template's words are read by
 * the same floor as a finished photo prompt — in settings, when a cut is planned, when it is sent and in the studio.
 */
async function evaluateCustomWorkflowFloor({ seed, orchestrator, studio }) {
  const scope = "custom-workflow";
  const template = (text, extra = {}) =>
    JSON.stringify({
      "1": { class_type: "UNETLoader", inputs: { unet_name: "z_image_turbo_int8_convrot.safetensors", weight_dtype: "default" } },
      "2": { class_type: "CLIPLoader", inputs: { clip_name: "qwen_3_4b_fp8_mixed.safetensors", type: "lumina2", device: "default" } },
      "3": { class_type: "VAELoader", inputs: { vae_name: "ae.safetensors" } },
      "4": { class_type: "CLIPTextEncode", inputs: { clip: ["2", 0], text } },
      "5": { class_type: "ConditioningZeroOut", inputs: { conditioning: ["4", 0] } },
      "6": { class_type: "EmptySD3LatentImage", inputs: { width: "%width%", height: "%height%", batch_size: 1 } },
      "7": {
        class_type: "KSampler",
        inputs: { model: ["1", 0], seed: "%seed%", steps: 8, cfg: 1, sampler_name: "res_multistep", scheduler: "simple", positive: ["4", 0], negative: ["5", 0], latent_image: ["6", 0], denoise: 1 }
      },
      "8": { class_type: "VAEDecode", inputs: { samples: ["7", 0], vae: ["3", 0] } },
      "9": { class_type: "PreviewImage", inputs: { images: ["8", 0] } },
      ...extra
    });
  const stateFor = (customWorkflow, mode = "photoreal") =>
    createState(seed.seedState, {
      routing: { mode },
      photorealOverrides: { workflowPreset: "custom", referenceWorkflowPreset: "custom", customWorkflow, faceReferenceEnabled: false }
    });
  const clean = template("%prompt%");
  const explicit = template("A fully naked 14 year old girl. %prompt%");
  const shapes = [
    [explicit, "4", "explicit, minor-coded words before %prompt%"],
    [template("%prompt% She is completely naked."), "4", "words after %prompt%"],
    [template("A nude teenage girl on a bed"), "4", "no %prompt% at all, only its own words"],
    [template("a 14 %prompt% year old girl"), "4", "a phrase split around %prompt%"],
    [template("%prompt%, 나체"), "4", "Korean words (sent to the encoder as written, so unreadable to the floor)"],
    [template(["10", 0], { "10": { class_type: "PrimitiveString", inputs: { value: "A fully naked 14 year old girl" } } }), "10", "the words parked in a PrimitiveString linked into the encoder"]
  ];

  assertCheck(scope, orchestrator.validatePhotorealCustomWorkflow(stateFor(clean), clean).length === 0, "a template that only carries %prompt% is valid");
  for (const [workflow, nodeId, label] of shapes) {
    const errors = orchestrator.validatePhotorealCustomWorkflow(stateFor(workflow), workflow);
    assertCheck(
      scope,
      errors.some((error) => error.includes(`노드 "${nodeId}"`) && /실사 안전 기준/u.test(error)),
      `settings: a template with ${label} is invalid, naming node ${nodeId}`,
      JSON.stringify(errors)
    );
    const job = orchestrator.planImageJob(stateFor(workflow), "turn_custom", toCue(MIXED_CUES[0]), [], true);
    assertCheck(
      scope,
      job.backend === "novelai" && job.routing?.locked === true && job.routing.reason.includes(`노드 "${nodeId}"`) && job.providerPayload.promptFormat === "novelai-tags",
      `planning: a clean cut under that template becomes a locked NovelAI cut, and the reason names the node (${label})`,
      `${job.backend} ${job.routing?.reason}`
    );
  }
  const cleanJob = orchestrator.planImageJob(stateFor(clean), "turn_custom", toCue(MIXED_CUES[0]), [], true);
  assertCheck(scope, cleanJob.backend === "photoreal" && cleanJob.providerPayload.presetId === "custom", "the same cut under a clean template stays photoreal on the custom preset", `${cleanJob.backend} ${cleanJob.providerPayload.presetId}`);
  // File and list inputs are not text: a model file name does not invalidate a template (they never reach an encoder).
  const namedModel = template("%prompt%", { "1": { class_type: "UNETLoader", inputs: { unet_name: "nsfw_finetune_v2.safetensors", weight_dtype: "default" } } });
  assertCheck(scope, orchestrator.validatePhotorealCustomWorkflow(stateFor(namedModel), namedModel).length === 0, "a model file name is not screened as prompt text");
  // A negative says what to keep out. Words on the negative side are cleaned on the way to the sampler, not screened —
  // while the same words anywhere the picture can see them are.
  const negativeSide = template("%prompt%", {
    "5": { class_type: "CLIPTextEncode", inputs: { clip: ["2", 0], text: "nude, child, %negative_prompt%" } }
  });
  const negativeSideJob = orchestrator.planImageJob(stateFor(negativeSide), "turn_custom", toCue(MIXED_CUES[0]), [], true);
  assertCheck(
    scope,
    orchestrator.validatePhotorealCustomWorkflow(stateFor(negativeSide), negativeSide).length === 0 && negativeSideJob.backend === "photoreal",
    "\"nude, child\" on a template's negative side does not invalidate it (the cut stays photoreal)",
    `${negativeSideJob.backend} ${negativeSideJob.routing?.reason}`
  );
  const alsoPicture = template("%prompt%", {
    "5": { class_type: "CLIPTextEncode", inputs: { clip: ["2", 0], text: "nude, child" } },
    "10": { class_type: "ConditioningCombine", inputs: { conditioning_1: ["4", 0], conditioning_2: ["5", 0] } },
    "7": {
      class_type: "KSampler",
      inputs: { model: ["1", 0], seed: "%seed%", steps: 8, cfg: 1, sampler_name: "res_multistep", scheduler: "simple", positive: ["10", 0], negative: ["5", 0], latent_image: ["6", 0], denoise: 1 }
    }
  });
  assertCheck(
    scope,
    orchestrator.validatePhotorealCustomWorkflow(stateFor(alsoPicture), alsoPicture).some((error) => error.includes('노드 "5"') && /실사 안전 기준/u.test(error)),
    "the same words in a node that also reaches the picture (through ConditioningCombine into positive) are screened and refused"
  );

  let studioMessage;
  try {
    studio.planCharacterProfileTestJobs(stateFor(explicit), { characterId: "char_yuna", backend: "photoreal", prompt: "A Korean woman in her late twenties", variant: "portrait", count: 1 });
  } catch (error) {
    studioMessage = error instanceof Error ? error.message : String(error);
  }
  assertCheck(
    scope,
    typeof studioMessage === "string" && /사용자 워크플로/u.test(studioMessage) && /노드 "4"/u.test(studioMessage),
    "the profile studio refuses a 실사 test under that template (never planned on NovelAI), naming the workflow",
    studioMessage
  );

  // The template is read when the cut runs: a cut planned under a clean template is refused if the template changed since.
  const calls = [];
  const sent = await withFetchStub(
    async (url) => {
      calls.push(url);
      return new Response(JSON.stringify({ error: "should not be called" }), { status: 500, headers: { "content-type": "application/json" } });
    },
    () => orchestrator.executeImageJob(stateFor(explicit), cleanJob)
  );
  assertCheck(
    scope,
    sent.job.status === "failed" && /^실사 이미지 생성 실패: 사용자 워크플로/u.test(sent.job.error ?? "") && /노드 "4"/u.test(sent.job.error ?? "") && calls.length === 0,
    "execution: a cut planned under a clean template is refused once the template carries refused words, and nothing is sent",
    `${sent.job.status} ${sent.job.error} calls=${calls.length}`
  );
}

/**
 * The floor's last look, as a photoreal request leaves (red team, channels E and F). A job keeps the prompt and face
 * references it was planned with, and a queued job survives a reload, so what was clean under an older rule set — or
 * a reference whose profile has since lost its standing — would go out unchanged. A refused cut fails with a Korean
 * reason; it is not quietly rendered on NovelAI (that spends Anlas) and it keeps no references.
 */
async function evaluateDispatchFloor({ seed, orchestrator }) {
  const scope = "dispatch";
  const state = createState(seed.seedState, { routing: { mode: "photoreal" } });
  const planned = orchestrator.planImageJob(state, "turn_dispatch", toCue(MIXED_CUES[0]), [], true);
  assertCheck(
    scope,
    planned.backend === "photoreal" && planned.providerPayload.photorealReferences?.length === 1,
    "fixture: a clean photoreal cut planned with Yuna's face reference",
    `${planned.backend} ${JSON.stringify(planned.providerPayload.photorealReferences)}`
  );
  const outputBase64 = createMockPng().toString("base64");
  const run = async (runState, job) => {
    const calls = [];
    const result = await withFetchStub(
      async (url) => {
        calls.push(url);
        if (/\/photoreal\/comfyui\/generate$/u.test(url)) {
          return new Response(JSON.stringify({ promptId: "eval", images: [{ filename: "x.png", subfolder: "", type: "temp", mimeType: "image/png", base64: outputBase64 }], elapsedMs: 1 }), {
            status: 200,
            headers: { "content-type": "application/json" }
          });
        }
        return new Response(JSON.stringify({ error: "unexpected" }), { status: 500, headers: { "content-type": "application/json" } });
      },
      () => orchestrator.executeImageJob(runState, job)
    );
    return { result, calls };
  };

  const control = await run(state, planned);
  assertCheck(
    scope,
    control.result.job.status === "completed" && control.calls.length === 1 && /\/photoreal\/comfyui\/generate$/u.test(control.calls[0]),
    "control: the clean job passes the dispatch gate and reaches the relay once",
    `${control.result.job.status} ${control.result.job.error ?? ""} ${control.calls.join(",")}`
  );

  // A job planned under an older rule set: its prompt says what today's floor refuses.
  const appended = " She is 16 years old.";
  const stale = {
    ...planned,
    prompt: `${planned.prompt}${appended}`,
    providerPayload: {
      ...planned.providerPayload,
      promptVariants: (planned.providerPayload.promptVariants ?? []).map((variant) => ({ ...variant, prompt: `${variant.prompt}${appended}` }))
    }
  };
  const refused = await run(state, stale);
  assertCheck(
    scope,
    refused.result.job.status === "failed" && /^실사 이미지 생성 실패: /u.test(refused.result.job.error ?? "") && /안전 기준/u.test(refused.result.job.error ?? ""),
    "a queued job whose planned prompt the floor now locks fails with a Korean refusal",
    `${refused.result.job.status} ${refused.result.job.error}`
  );
  assertCheck(
    scope,
    refused.calls.length === 0 && refused.result.assets.length === 0 && refused.result.job.backend === "photoreal" && refused.result.job.routing?.backend === "photoreal",
    "nothing is sent — no relay request and no NovelAI request: the cut is not quietly re-rendered on NovelAI",
    `calls=${refused.calls.join(",")} backend=${refused.result.job.backend}`
  );
  assertCheck(
    scope,
    (refused.result.job.providerPayload.photorealReferences ?? []).length === 0 && refused.result.job.providerPayload.dispatchRefused === true,
    "the refused job keeps no face references",
    JSON.stringify(refused.result.job.providerPayload.photorealReferences)
  );

  // The face reference is checked against the character's profile as it is NOW.
  const withLiveProfile = (patch) => {
    const next = structuredClone(state);
    next.visualProfiles = next.visualProfiles.map((profile) =>
      profile.characterId === "char_yuna" ? { ...profile, profileImages: { ...profile.profileImages, photoreal: patch(profile.profileImages.photoreal) } } : profile
    );
    return next;
  };
  const referenceCases = [
    [withLiveProfile((image) => ({ ...image, prompt: "A 14-year-old Korean girl in a bikini" })), "whose recorded prompt the floor now locks"],
    [withLiveProfile((image) => ({ ...image, prompt: "" })), "whose recorded prompt is gone"],
    [withLiveProfile((image) => ({ ...image, objectKey: `${SIMULATION_ID}/profile_yuna_v2.png` })), "that is no longer the character's profile image"],
    [withLiveProfile(() => undefined), "whose character no longer has a photoreal profile"]
  ];
  for (const [liveState, label] of referenceCases) {
    const outcome = await run(liveState, planned);
    assertCheck(
      scope,
      outcome.result.job.status === "failed" &&
        /얼굴 참조 이미지를 쓸 수 없어/u.test(outcome.result.job.error ?? "") &&
        /유나/u.test(outcome.result.job.error ?? "") &&
        outcome.calls.length === 0 &&
        (outcome.result.job.providerPayload.photorealReferences ?? []).length === 0,
      `a planned face reference ${label} is never sent as <image1>: the cut fails, nothing is sent, the reference is dropped`,
      `${outcome.result.job.status} ${outcome.result.job.error} calls=${outcome.calls.length}`
    );
  }
}

/**
 * Red team round 2, structural: screening the pieces is not enough, so the graph as it would be SENT is built and read
 * at plan time and again as it leaves. Also here: a template that cannot build becomes a NovelAI cut when it is
 * planned (not a photoreal cut that fails later), no CFG below 1 reaches ComfyUI, a face reference is only a picked
 * profile image, and every photo negative — settings, character, the studio box — goes through the one negative rule.
 */
async function evaluateSentGraphFloor({ seed, orchestrator, studio, photoPrompt }) {
  const scope = "sent-graph";
  const template = (text, extra = {}) =>
    JSON.stringify({
      "1": { class_type: "UNETLoader", inputs: { unet_name: "z_image_turbo_int8_convrot.safetensors", weight_dtype: "default" } },
      "2": { class_type: "CLIPLoader", inputs: { clip_name: "qwen_3_4b_fp8_mixed.safetensors", type: "lumina2", device: "default" } },
      "3": { class_type: "VAELoader", inputs: { vae_name: "ae.safetensors" } },
      "4": { class_type: "CLIPTextEncode", inputs: { clip: ["2", 0], text } },
      "5": { class_type: "ConditioningZeroOut", inputs: { conditioning: ["4", 0] } },
      "6": { class_type: "EmptySD3LatentImage", inputs: { width: "%width%", height: "%height%", batch_size: 1 } },
      "7": {
        class_type: "KSampler",
        inputs: { model: ["1", 0], seed: "%seed%", steps: 8, cfg: 1, sampler_name: "res_multistep", scheduler: "simple", positive: ["4", 0], negative: ["5", 0], latent_image: ["6", 0], denoise: 1 }
      },
      "8": { class_type: "VAEDecode", inputs: { samples: ["7", 0], vae: ["3", 0] } },
      "9": { class_type: "PreviewImage", inputs: { images: ["8", 0] } },
      ...extra
    });
  const stateFor = (customWorkflow, overrides = {}) =>
    createState(seed.seedState, {
      routing: { mode: "photoreal" },
      photorealOverrides: { workflowPreset: "custom", referenceWorkflowPreset: "custom", customWorkflow, faceReferenceEnabled: false, ...overrides }
    });
  const cue = toCue(MIXED_CUES[0]);
  const outputBase64 = createMockPng().toString("base64");
  const dispatch = async (runState, job) => {
    const bodies = [];
    const result = await withFetchStub(
      async (url, init) => {
        if (/\/photoreal\/comfyui\/generate$/u.test(url)) {
          bodies.push(JSON.parse(String(init?.body ?? "{}")));
          return new Response(JSON.stringify({ promptId: "eval", images: [{ filename: "x.png", subfolder: "", type: "temp", mimeType: "image/png", base64: outputBase64 }], elapsedMs: 1 }), {
            status: 200,
            headers: { "content-type": "application/json" }
          });
        }
        return new Response(JSON.stringify({ error: "unexpected" }), { status: 500, headers: { "content-type": "application/json" } });
      },
      () => orchestrator.executeImageJob(runState, job)
    );
    return { result, bodies };
  };
  const studioError = (state, input) => {
    try {
      studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "photoreal", prompt: "A Korean woman in her late twenties", variant: "portrait", count: 1, ...input });
      return "";
    } catch (error) {
      return describeError(error);
    }
  };

  // The built graph, read directly: a word split over the pieces, and a phrase split over two encoders.
  const gateState = createState(seed.seedState, { routing: { mode: "photoreal" } });
  const built = (workflowJson) => JSON.parse(workflowJson.replaceAll('"%width%"', "1024").replaceAll('"%height%"', "1024").replaceAll('"%seed%"', "1"));
  const split = orchestrator.screenBuiltPhotorealWorkflow(gateState, built(template("A candid photograph of a woman, na ked, lying on a bed")));
  assertCheck(scope, split?.decision.locked === true && split.text?.nodeId === "4" && split.text.input === "text", "screenBuiltPhotorealWorkflow locks a built string the pieces never contained, and names its node", JSON.stringify(split?.text));
  const acrossNodes = built(
    template("A candid photograph. She is a girl", {
      "10": { class_type: "CLIPTextEncode", inputs: { clip: ["2", 0], text: "of twelve" } },
      "11": { class_type: "ConditioningCombine", inputs: { conditioning_1: ["4", 0], conditioning_2: ["10", 0] } },
      "7": {
        class_type: "KSampler",
        inputs: { model: ["1", 0], seed: 1, steps: 8, cfg: 1, sampler_name: "res_multistep", scheduler: "simple", positive: ["11", 0], negative: ["5", 0], latent_image: ["6", 0], denoise: 1 }
      }
    })
  );
  const acrossRefusal = orchestrator.screenBuiltPhotorealWorkflow(gateState, acrossNodes);
  // Locked either way: by the joined reading while "of twelve" passes alone (it does today), or by node 10 itself.
  assertCheck(scope, acrossRefusal?.decision.locked === true, "...and one read over every picture-side string together, for a phrase spread over encoders the sampler combines", JSON.stringify(acrossRefusal?.text ?? "joined"));
  const negativeOnly = built(template("A candid photograph of a woman by a window", { "5": { class_type: "CLIPTextEncode", inputs: { clip: ["2", 0], text: "nude, child, watermark" } } }));
  assertCheck(scope, orchestrator.screenBuiltPhotorealWorkflow(gateState, negativeOnly) === undefined, "the negative side is not read as the picture (a negative says \"nude\" to keep it out), and a clean picture passes");

  // Settings and planning for the phrase over two nodes: the template's literals are also read together.
  const twoNodeTemplate = template("%prompt% She is a girl", {
    "10": { class_type: "CLIPTextEncode", inputs: { clip: ["2", 0], text: "of twelve" } },
    "11": { class_type: "ConditioningCombine", inputs: { conditioning_1: ["4", 0], conditioning_2: ["10", 0] } },
    "7": {
      class_type: "KSampler",
      inputs: { model: ["1", 0], seed: "%seed%", steps: 8, cfg: 1, sampler_name: "res_multistep", scheduler: "simple", positive: ["11", 0], negative: ["5", 0], latent_image: ["6", 0], denoise: 1 }
    }
  });
  const twoNodeErrors = orchestrator.validatePhotorealCustomWorkflow(stateFor(twoNodeTemplate), twoNodeTemplate);
  const twoNodeJob = orchestrator.planImageJob(stateFor(twoNodeTemplate), "turn_sent", cue, [], true);
  assertCheck(
    scope,
    twoNodeErrors.some((error) => (/여러 노드/u.test(error) || error.includes('노드 "10"')) && /실사 안전 기준/u.test(error)) &&
      twoNodeJob.backend === "novelai" &&
      twoNodeJob.routing?.locked === true,
    "a template whose literals pass one by one but not together is invalid in settings, and its cuts are planned as locked NovelAI cuts",
    `${JSON.stringify(twoNodeErrors)} ${twoNodeJob.backend} ${twoNodeJob.routing?.reason}`
  );

  // Dispatch: only the graph as built shows the word. The job's prompt ends in a fragment (planned under an older
  // builder, or edited), the template's literal starts with the rest; each passes its own gate.
  const junctionTemplate = template("%prompt% ked, lying on a bed");
  const junctionState = stateFor(junctionTemplate);
  const junctionPlanned = orchestrator.planImageJob(junctionState, "turn_sent", cue, [], true);
  assertCheck(scope, junctionPlanned.backend === "photoreal", "fixture: a clean cut under a template whose literal starts with a fragment is photoreal", `${junctionPlanned.backend} ${junctionPlanned.routing?.reason}`);
  const fragmentPrompt = `${junctionPlanned.prompt.replace(/[.\s]+$/u, "")}, na`;
  const fragmentJob = {
    ...junctionPlanned,
    prompt: fragmentPrompt,
    providerPayload: { ...junctionPlanned.providerPayload, promptVariants: (junctionPlanned.providerPayload.promptVariants ?? []).map((variant) => ({ ...variant, prompt: fragmentPrompt })) }
  };
  const fragmentRun = await dispatch(junctionState, fragmentJob);
  assertCheck(
    scope,
    fragmentRun.result.job.status === "failed" &&
      /^실사 이미지 생성 실패: 보내려던 실사 그래프의 글\(노드 "4" \(CLIPTextEncode\)의 입력 "text"\)/u.test(fragmentRun.result.job.error ?? "") &&
      fragmentRun.bodies.length === 0 &&
      fragmentRun.result.job.providerPayload.dispatchRefused === true,
    "dispatch reads the graph that is about to leave: a word made only by the assembly is refused, nothing is sent, and the cut is not re-sent to NovelAI",
    `${fragmentRun.result.job.status} ${fragmentRun.result.job.error} sent=${fragmentRun.bodies.length}`
  );
  const junctionControl = await dispatch(junctionState, junctionPlanned);
  assertCheck(scope, junctionControl.result.job.status === "completed" && junctionControl.bodies.length === 1, "control: the same template with the planned prompt is sent", `${junctionControl.result.job.status} ${junctionControl.result.job.error ?? ""}`);

  // A template that cannot build is refused when the cut is PLANNED (red team: a LoadImage pass-through planned as a
  // photoreal cut and failed at execution). Each of these is a structural rule of round 3 or before.
  const unbuildable = [
    [JSON.stringify({ "1": { class_type: "LoadImage", inputs: { image: "any_photo_on_disk.png" } }, "2": { class_type: "PreviewImage", inputs: { images: ["1", 0] } } }), "a LoadImage pass-through"],
    [template("%prompt% She is a %steps% year old girl."), "a %steps% inside the picture's text"],
    [template("%prompt%", { "7": { class_type: "KSampler", inputs: { model: ["1", 0], seed: "%seed%", steps: 8, cfg: 0, sampler_name: "euler", scheduler: "simple", positive: ["4", 0], negative: ["5", 0], latent_image: ["6", 0], denoise: 1 } } }), "a KSampler cfg 0"],
    [template("%prompt%ked, lying on a bed"), "%prompt% glued to a word"],
    [template("%prompt%", { "20": { class_type: "LoadImage", inputs: { image: "%image1%" } } }), "a %image1% the cut has no face reference for"],
    ["", "an empty custom workflow"]
  ];
  for (const [workflowJson, label] of unbuildable) {
    const job = orchestrator.planImageJob(stateFor(workflowJson), "turn_sent", cue, [], true);
    assertCheck(
      scope,
      job.backend === "novelai" &&
        job.routing?.locked === true &&
        job.routing.signals.includes("photo-graph:unbuildable") &&
        /그래프를 만들 수 없어 NovelAI로 바꿨습니다/u.test(job.routing.reason) &&
        job.providerPayload.promptFormat === "novelai-tags",
      `planning: ${label} makes the cut a locked NovelAI cut with the reason on it, not a photoreal cut that fails later`,
      `${job.backend} ${job.routing?.reason}`
    );
  }
  const studioUnbuildable = studioError(stateFor(template("%prompt% She is a %steps% year old girl.")), {});
  assertCheck(scope, /실사 그래프를 만들 수 없어/u.test(studioUnbuildable) && /%steps%/u.test(studioUnbuildable), "the profile studio refuses a 실사 test under such a template in its own words (not as a safety lock)", studioUnbuildable);

  // CFG below 1 never reaches ComfyUI, whatever the stored setting says (S1: a negative at CFG 0.1 was the picture).
  const lowCfg = createState(seed.seedState, {
    routing: { mode: "photoreal" },
    photorealOverrides: { workflowPreset: "qwen_image_2_1", referenceWorkflowPreset: "qwen_image_2_1", cfg: 0.1, faceReferenceEnabled: false }
  });
  const lowCfgJob = orchestrator.planImageJob(lowCfg, "turn_sent", cue, [], true);
  const lowCfgRun = await dispatch(lowCfg, lowCfgJob);
  const lowCfgSampler = Object.values(lowCfgRun.bodies[0]?.workflow ?? {}).find((node) => node?.class_type === "KSampler")?.inputs ?? {};
  assertCheck(
    scope,
    lowCfgJob.backend === "photoreal" && lowCfgJob.providerPayload.cfg === 1 && lowCfgRun.result.job.status === "completed" && lowCfgSampler.cfg === 1,
    "a stored CFG of 0.1 is planned and sent as 1",
    `planned=${lowCfgJob.providerPayload.cfg} sent=${lowCfgSampler.cfg} ${lowCfgRun.result.job.error ?? ""}`
  );

  // Every photo negative goes through the one negative rule on its way: the studio box, a character's own, the
  // settings negative — planned in its sent form, and sent exactly as the rule leaves it.
  const negativeState = createState(seed.seedState, {
    routing: { mode: "photoreal" },
    photorealOverrides: { workflowPreset: "qwen_image_2_1", referenceWorkflowPreset: "qwen_image_2_1", cfg: 4, faceReferenceEnabled: false, negativePrompt: "clothes, adult, watermark" }
  });
  const steeringState = structuredClone(negativeState);
  steeringState.visualProfiles = steeringState.visualProfiles.map((profile) =>
    profile.characterId === "char_yuna" ? { ...profile, photorealNegativePrompt: "clothing, fully clothed, heavy makeup" } : profile
  );
  const storyJob = orchestrator.planImageJob(steeringState, "turn_sent", cue, [], true);
  const studioJobs = studio.planCharacterProfileTestJobs(negativeState, {
    characterId: "char_yuna",
    backend: "photoreal",
    prompt: "A Korean woman in her late twenties with long black hair",
    negativePrompt: "clothing, sweater, adult woman, blurry",
    variant: "portrait",
    count: 1
  });
  const faceCheckJobs = studio.planCharacterProfileTestJobs(negativeState, {
    characterId: "char_yuna",
    backend: "photoreal",
    prompt: "A Korean woman in her late twenties with long black hair",
    negativePrompt: "clothed, adult, jpeg artifacts",
    variant: "reference_check",
    count: 1
  });
  for (const [label, runState, job] of [
    ["a story cut with a settings and a character negative", steeringState, storyJob],
    ["a studio candidate with the negative box filled", negativeState, studioJobs[0]],
    ["a studio face check with the negative box filled", negativeState, faceCheckJobs[0]]
  ]) {
    const run = await dispatch(runState, job);
    const sentNegative = Object.values(run.bodies[0]?.workflow ?? {}).find((node) => node?.class_type === "TextEncodeQwenImage21")?.inputs?.negative_prompt;
    assertCheck(
      scope,
      job.backend === "photoreal" &&
        job.negativePrompt === photoPrompt.sanitizePhotorealNegativePrompt(job.negativePrompt) &&
        !/\b(?:cloth(?:es|ing)|clothed|sweater|adult)\b/iu.test(job.negativePrompt) &&
        run.result.job.status === "completed" &&
        sentNegative === photoPrompt.sanitizePhotorealNegativePrompt(job.negativePrompt),
      `${label}: the planned negative is already in its sent form, carries no safeguard word, and is sent exactly as the rule leaves it`,
      `${job.negativePrompt} | sent ${JSON.stringify(sentNegative)} ${run.result.job.error ?? ""}`
    );
  }

  // A face reference that is not a picked profile image is refused at dispatch too, and by the studio's face check.
  const referenced = createState(seed.seedState, { routing: { mode: "photoreal" } });
  const referencedJob = orchestrator.planImageJob(referenced, "turn_sent", cue, [], true);
  const notAPick = `${SIMULATION_ID}/asset_cut_9.png`;
  const swapped = structuredClone(referenced);
  swapped.visualProfiles = swapped.visualProfiles.map((profile) =>
    profile.characterId === "char_yuna" ? { ...profile, profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, objectKey: notAPick } } } : profile
  );
  const swappedJob = {
    ...referencedJob,
    providerPayload: {
      ...referencedJob.providerPayload,
      photorealReferences: (referencedJob.providerPayload.photorealReferences ?? []).map((reference) => ({ ...reference, objectKey: notAPick }))
    }
  };
  const swappedRun = await dispatch(swapped, swappedJob);
  assertCheck(
    scope,
    referencedJob.providerPayload.photorealReferences?.length === 1 &&
      swappedRun.result.job.status === "failed" &&
      /얼굴 참조 이미지를 쓸 수 없어/u.test(swappedRun.result.job.error ?? "") &&
      /프로필로 고른 이미지가 아니어서/u.test(swappedRun.result.job.error ?? "") &&
      swappedRun.bodies.length === 0,
    "a planned reference whose key (live and planned alike) names a story cut, not a pick, is refused as the cut leaves; nothing is sent",
    `${swappedRun.result.job.status} ${swappedRun.result.job.error}`
  );
  for (const [label, patch, pattern] of [
    ["whose key is not a pick", { objectKey: notAPick }, /생성한 후보 중에서 고른 이미지가 아니어서/u],
    ["whose recorded prompt is only \"1girl\"", { prompt: "1girl" }, /읽을 수 있는 영어 단어가 없어/u],
    ["whose recorded prompt is a zero-width space", { prompt: "\u200b" }, /읽을 수 있는 영어 단어가 없어/u]
  ]) {
    const checkState = structuredClone(referenced);
    checkState.visualProfiles = checkState.visualProfiles.map((profile) =>
      profile.characterId === "char_yuna" ? { ...profile, profileImages: { ...profile.profileImages, photoreal: { ...profile.profileImages.photoreal, ...patch } } } : profile
    );
    const message = studioError(checkState, { variant: "reference_check" });
    assertCheck(scope, pattern.test(message) && /새 후보를 생성해/u.test(message), `the studio's face check refuses a profile image ${label}, saying why`, message);
  }
}

function createState(seedState, { routing = { mode: "novelai" }, photoreal = true, novelAi = true, cadence = "image_progression", photorealOverrides = {}, profileEveryone = false } = {}) {
  const state = structuredClone(seedState);
  state.simulation = { ...state.simulation, id: SIMULATION_ID, realtimeImageEnabled: true, contentRating: "adult_19" };
  state.imageProfile = {
    ...state.imageProfile,
    simulationId: SIMULATION_ID,
    enabled: true,
    triggerMode: "realtime_auto",
    generationCadence: cadence,
    cooldownTurns: 0,
    countMin: 1,
    countMax: 1,
    userRules: "",
    backendRouting: {
      mode: routing.mode,
      photorealCategories: routing.photorealCategories ?? ["portrait", "full_body", "scenery", "object"]
    }
  };
  state.llm = { ...state.llm, enabled: false, provider: "mock", apiKey: "" };
  state.imageTagLlm = { ...state.imageTagLlm, enabled: false, provider: "mock", apiKey: "" };
  state.neuralMap = { ...state.neuralMap, enabled: false };
  state.novelAi = { ...state.novelAi, enabled: novelAi, requestMode: "proxy", modelPreset: "NAID4.5F", automationTermination: "unlimited", vibeTransferEnabled: false };
  state.photoreal = {
    ...state.photoreal,
    enabled: photoreal,
    requestMode: "proxy",
    baseUrl: "http://127.0.0.1:8188",
    proxyUrl: "",
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
    ...photorealOverrides
  };
  state.characters = [
    createCharacter("char_yuna", "유나"),
    createCharacter("char_minho", "민호"),
    createCharacter("char_sora", "소라"),
    createCharacter("char_jiwoo", "지우")
  ];
  state.visualProfiles = [
    createProfile("char_yuna", "유나", {
      positivePrompt: "1girl, long black hair, brown eyes",
      defaultOutfitPrompt: "white knit sweater, pleated skirt",
      photorealPrompt: "a Korean woman in her late twenties with long straight black hair and soft brown eyes",
      photorealNegativePrompt: "heavy makeup",
      profileImages: {
        photoreal: profileImage("photoreal", "profile_yuna", { width: 960, height: 1280 }),
        novelai: profileImage("novelai", "profile_yuna_nai", { referenceObjectKey: `${SIMULATION_ID}/profile_yuna_nai_ref.png` })
      }
    }),
    createProfile("char_minho", "민호", {
      positivePrompt: "1boy, short brown hair, black eyes",
      defaultOutfitPrompt: "navy suit",
      ...(profileEveryone ? { profileImages: { photoreal: profileImage("photoreal", "profile_minho") } } : {})
    }),
    createProfile("char_sora", "소라", {
      positivePrompt: "1girl, short pink hair",
      profileImages: { photoreal: profileImage("photoreal", "profile_sora") }
    }),
    createProfile("char_jiwoo", "지우", {
      positivePrompt: "1boy, black hair, glasses",
      profileImages: { photoreal: profileImage("photoreal", "profile_jiwoo") }
    })
  ];
  state.messages = [];
  state.memoryEvents = [];
  state.imageJobs = [];
  state.imageAssets = [];
  return state;
}

function createCharacter(id, name) {
  return { id, simulationId: SIMULATION_ID, name, role: "", summary: "", relationship: "", currentMood: "calm" };
}

function createProfile(characterId, displayName, overrides) {
  return {
    id: `visual_${characterId}`,
    simulationId: SIMULATION_ID,
    characterId,
    displayName,
    positivePrompt: "",
    negativePrompt: "",
    defaultOutfitPrompt: "",
    outfitPrompts: {},
    expressionPrompts: {},
    referenceImageAssetIds: [],
    defaultSafetyLevel: "safe",
    ...overrides
  };
}

function profileImage(backend, assetId, extra = {}) {
  return {
    backend,
    assetId,
    objectKey: `${SIMULATION_ID}/${assetId}.png`,
    mimeType: "image/png",
    prompt: "eval profile",
    selectedAt: "2026-10-14T00:00:00.000Z",
    ...extra
  };
}

function toCue(cue) {
  return {
    shouldGenerate: true,
    reason: cue.label,
    characters: cue.characters,
    tags: [],
    baseTags: cue.base,
    characterPrompts: cue.prompts.map(([characterId, prompt]) => ({ characterId, prompt })),
    scene: "current simulation scene",
    visualContext: "",
    frame: typeof cue.frame === "string" ? FRAMES[cue.frame] : cue.frame,
    kind: cue.kind
  };
}

function toDraft(cue) {
  return { ...toCue(cue), label: cue.label, placement: "after" };
}

async function planTurn(engine, state, drafts) {
  const userMessage = createMessage(state, "user", "장면을 이어간다.");
  const assistantMessage = createMessage(state, "assistant", "유나가 카페 창가에서 커피를 마시며 미소 짓는다.");
  return engine.planImageJobForCompletedTurn(
    { ...state, messages: [...state.messages, userMessage, assistantMessage] },
    {
      userMessage,
      assistantMessage,
      contextPack: {
        id: `ctx_${Math.random().toString(36).slice(2)}`,
        simulationId: state.simulation.id,
        sessionId: state.simulation.activeSessionId,
        source: "mock",
        objective: "photoreal pipeline eval",
        evidence: [],
        decisions: [],
        blockers: [],
        tokenBudget: 0,
        createdAt: new Date().toISOString()
      },
      promptModuleUsages: [],
      sidecar: { assistantText: assistantMessage.content, memoryEvents: [], imageCue: drafts[0], imageCues: drafts },
      sidecarTrace: {
        id: `sidecar_${Math.random().toString(36).slice(2)}`,
        simulationId: state.simulation.id,
        sessionId: state.simulation.activeSessionId,
        turnId: userMessage.id,
        source: "llm",
        status: "parsed",
        errors: [],
        createdAt: new Date().toISOString()
      },
      manualImage: false
    }
  );
}

function createMessage(state, role, content) {
  return {
    id: `msg_${role}_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    role,
    content,
    createdAt: new Date().toISOString(),
    referencedNodeIds: [],
    imageAssetIds: []
  };
}

function createStateMemoryEvent(state, actorId, stateType, value) {
  return {
    id: `memory_eval_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    actorId,
    actorName: actorId,
    content: `[State] ${actorId} ${stateType} = ${value}`,
    importance: 0.76,
    tags: ["memory-delta", "kind:state", `state:${stateType}`],
    sourceTurnId: `turn_eval_${Math.random().toString(36).slice(2)}`,
    createdAt: new Date().toISOString(),
    metadata: { memory_kind: "state", state_type: stateType, value, owner_id: actorId }
  };
}

/** A generated asset as executeImageJob would have stored it for `job`. */
function assetFromJob(job, id, backend) {
  const cue = job.providerPayload.cue;
  return {
    id,
    simulationId: SIMULATION_ID,
    title: "eval asset",
    source: "generated",
    prompt: job.prompt,
    negativePrompt: job.negativePrompt,
    safetyLevel: "safe",
    characterIds: cue.characters,
    tags: job.providerPayload.reuseTags,
    reuseTags: job.providerPayload.reuseTags,
    createdAt: new Date().toISOString(),
    jobId: job.id,
    palette: ["#213547", "#d5e1e8", "#f2b84b"],
    dataUrl: "data:image/png;base64,AA==",
    providerMetadata: job.providerPayload,
    representative: true,
    backend,
    sceneCategory: job.routing?.category
  };
}

/** Routes fetch through `handler` (returning a Response, or undefined to fall through to the real fetch). */
async function withFetchStub(handler, run) {
  globalThis.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const handled = await handler(url, init);
    return handled ?? realFetch(input, init);
  };
  try {
    return await run();
  } finally {
    globalThis.fetch = realFetch;
  }
}

async function stubOrThrow(handler, input, init) {
  const url = typeof input === "string" ? input : input.url;
  const handled = await handler(url, init);
  if (!handled) {
    throw new TypeError(`unexpected fetch ${url}`);
  }
  return handled;
}

async function startApiServer(tempRoot) {
  const dataDir = path.join(tempRoot, "api");
  await mkdir(dataDir, { recursive: true });
  const port = await findFreePort();
  const env = { ...process.env };
  for (const key of SERVER_ENV_KEYS) {
    delete env[key];
  }
  Object.assign(env, { DYNAMICCHAT_DATA_DIR: dataDir, DYNAMICCHAT_API_PORT: String(port), DYNAMICCHAT_RATE_LIMIT_MAX: "10000" });
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
    dataDir,
    baseUrl: `http://127.0.0.1:${port}`,
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
      throw new Error(`API server exited during startup (code ${child.exitCode}):\n${output}`);
    }
    try {
      const health = await realFetch(`${server.baseUrl}/health`, { signal: AbortSignal.timeout(1_000) });
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
  throw new Error(`API server did not answer /health within 20s:\n${output}`);
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

/** The first numeric input named one of `names` anywhere in an API-format graph. */
function findInputValue(workflow, names) {
  for (const node of Object.values(workflow ?? {})) {
    for (const name of names) {
      const value = node?.inputs?.[name];
      if (typeof value === "number") {
        return value;
      }
    }
  }
  return undefined;
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

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function sameList(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]);
}

function assertThrowsKorean(scope, run, message) {
  try {
    run();
    assertCheck(scope, false, message, "did not throw");
  } catch (error) {
    assertCheck(scope, error instanceof Error && HANGUL_PATTERN.test(error.message), message, error?.message);
  }
}

function describeError(error) {
  return error instanceof Error ? `${error.message}${error.stack ? `\n${error.stack.split("\n").slice(1, 4).join("\n")}` : ""}` : String(error);
}

async function runGroup(scope, evaluate) {
  try {
    await evaluate();
  } catch (error) {
    assertCheck(scope, false, `threw before finishing: ${describeError(error)}`);
  }
}

function assertCheck(scope, passed, message, detail) {
  checks.push({ scope, passed: Boolean(passed), message: passed || detail === undefined ? message : `${message} — ${detail}` });
}
