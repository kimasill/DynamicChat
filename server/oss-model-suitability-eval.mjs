/**
 * Open-model SUITABILITY measurement.
 *
 * The integration eval proves the wiring works. This one asks the different question: driven repeatedly,
 * does an open-weight model actually hold the two contracts a simulation depends on?
 *
 *   - narrative contract: Korean prose, no scaffolding leakage, enough length to be a turn
 *   - tag contract:       every generating cue commits to one crop, names its cast, and carries concrete
 *                         English NovelAI tags that survive into a prompt
 *
 * Reported as rates over N turns rather than pass/fail, because model output varies and a single sample
 * says nothing about whether a backend is dependable enough to run a long simulation on.
 *
 * Usage:
 *   OSS_EVAL_BASE_URL=http://127.0.0.1:11434/v1 OSS_EVAL_MODEL=qwen3-4b-16k:latest \
 *   OSS_EVAL_TURNS=5 node server/oss-model-suitability-eval.mjs
 */
import { createServer } from "vite";

const baseUrl = process.env.OSS_EVAL_BASE_URL ?? "http://127.0.0.1:11434/v1";
const model = process.env.OSS_EVAL_MODEL ?? "qwen3-4b-16k:latest";
const turns = Number(process.env.OSS_EVAL_TURNS ?? 5);
const apiKey = process.env.OSS_EVAL_API_KEY ?? "";
const provider = process.env.OSS_EVAL_PROVIDER ?? "ollama";

try {
  const probe = await fetch(`${baseUrl}/models`, {
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {},
    signal: AbortSignal.timeout(4000)
  });
  if (!probe.ok) throw new Error(`HTTP ${probe.status}`);
} catch (error) {
  console.log(`SKIP: ${baseUrl} 에 접속할 수 없습니다 (${error.message}). OSS_EVAL_BASE_URL/OSS_EVAL_API_KEY 를 지정하세요.`);
  process.exit(0);
}

const vite = await createServer({ server: { middlewareMode: true }, appType: "custom", logLevel: "error" });
const { seedState, hydrateState } = await vite.ssrLoadModule("/src/data/seed.ts");
// OSS_EVAL_SEED=womanlife runs against the shipped full-size simulation (8 characters, 39 modules,
// paragraph cadence, adult_19) instead of the small demo. The demo understates the load: the real prompt is
// several times larger and the cadence asks for many more cuts per turn, which is where a local model's
// context and instruction-following actually get tested.
const { womanLifeSeedState } = await vite.ssrLoadModule("/src/data/womanLifeSeed.ts");
const activeSeed = (process.env.OSS_EVAL_SEED ?? "demo") === "womanlife" ? womanLifeSeedState : seedState;
const { runSimulationTurn, planImageJobForCompletedTurn } = await vite.ssrLoadModule("/src/services/simulationEngine.ts");

const base = hydrateState(structuredClone(activeSeed));
const contextTokens = Number(process.env.OSS_EVAL_CONTEXT ?? 16384);
const llm = { ...base.llm, enabled: true, provider, baseUrl, model, apiKey, temperature: 0.8, maxTokens: 4000, contextTokens };
// The tag pass can run on its own model. Splitting it is the local answer to a cadence the narrative model
// cannot afford: the annotation call is pure structure, and a small instruct model holds that contract at a
// fraction of the latency — which is what makes several cuts per turn practical on one GPU.
const tagModel = process.env.OSS_EVAL_TAG_MODEL ?? model;
const tagContextTokens = Number(process.env.OSS_EVAL_TAG_CONTEXT ?? contextTokens);
let state = hydrateState({
  ...base,
  llm,
  imageTagLlm: { ...llm, enabled: true, model: tagModel, contextTokens: tagContextTokens },
  simulation: { ...base.simulation, realtimeImageEnabled: true },
  // The seed's own cast, so the roster matches the prompt modules the model is actually reading. On the demo
  // seed the saved appearance is overridden with known tags so identity injection can be asserted exactly;
  // the real simulation already has its own profiles and is left alone.
  characters: base.characters,
  visualProfiles:
    (process.env.OSS_EVAL_SEED ?? "demo") === "womanlife"
      ? base.visualProfiles
      : base.visualProfiles.map((profile) => ({
          ...profile,
          positivePrompt: "1girl, long black hair, green eyes, pale skin",
          defaultOutfitPrompt: "librarian uniform, pleated skirt, black thighhighs, brown loafers"
        })),
  memoryEvents: [],
  imageJobs: [],
  imageAssets: [],
  neuralMap: { ...base.neuralMap, enabled: false },
  novelAi: { ...base.novelAi, enabled: false, modelPreset: "NAID4.5C" },
  // Honour the seed's own cadence. This used to pin every run to "balanced", which silently replaced the
  // real simulation's high-density `paragraph` setting — so a measured 1.0 cue/turn read as the model
  // ignoring the cadence when in fact 1 cue is exactly what balanced asks for. OSS_EVAL_CADENCE overrides.
  imageProfile: {
    ...base.imageProfile,
    enabled: true,
    triggerMode: "realtime_auto",
    generationCadence: process.env.OSS_EVAL_CADENCE ?? base.imageProfile.generationCadence ?? "balanced",
    cooldownTurns: 0
  }
});
console.log(`seed: ${process.env.OSS_EVAL_SEED ?? "demo"}   cadence: ${state.imageProfile.generationCadence}`);

// Prompts must belong to the seed's own world. Driving the full-size simulation with the demo's library
// prompts ("서가 사이를 걸으며…") put the turn outside every scene rule the modules describe, and the scene
// cast guard then correctly refused to place anyone on stage — which showed up as cues with no character
// captions and read like a tag-pipeline defect rather than the harness feeding it the wrong world.
// Each set walks the same framing ladder (approach → reach up → look away → turn away → crouch → close-up)
// so the frame-consistency metrics stay comparable across seeds.
const PROMPTS_BY_SEED = {
  demo: [
    "서가 사이를 천천히 걸으며 그녀에게 말을 건다.",
    "책장 위쪽 선반으로 손을 뻗는다.",
    "창가로 걸어가 밖을 내다본다.",
    "그녀에게 등을 돌리고 문 쪽으로 걷는다.",
    "무릎을 굽혀 바닥에 떨어진 종이를 줍는다.",
    "그녀의 얼굴을 가까이서 바라본다."
  ],
  womanlife: [
    "거울 앞에 서서 오늘 입고 나갈 옷을 고른다.",
    "옷장 위 선반에서 가방을 꺼내려 손을 뻗는다.",
    "창가로 가서 커튼을 걷고 밖을 내다본다.",
    "휴대폰을 보다가 등을 돌리고 현관 쪽으로 걷는다.",
    "무릎을 굽혀 바닥에 떨어진 고지서를 줍는다.",
    "화장대 거울에 얼굴을 가까이 대고 화장을 고친다."
  ]
};
const prompts = PROMPTS_BY_SEED[process.env.OSS_EVAL_SEED ?? "demo"] ?? PROMPTS_BY_SEED.demo;

const rows = [];
console.log(`model: ${model} @ ${baseUrl}   turns: ${turns}\n`);

for (let index = 0; index < turns; index += 1) {
  const userText = prompts[index % prompts.length];
  const startedAt = Date.now();
  let turn;
  try {
    turn = await runSimulationTurn(state, userText, false, { deferImagePlanning: true, deferMemoryIngest: true });
  } catch (error) {
    rows.push({ index, error: error instanceof Error ? error.message : String(error) });
    continue;
  }
  const latencyMs = Date.now() - startedAt;

  const narrative = turn.assistantMessage.content ?? "";
  const cues = turn.sidecar?.imageCues ?? [];
  const generating = cues.filter((cue) => cue.shouldGenerate);
  const stateEvents = (turn.sidecar?.memoryEvents ?? []).filter((event) => event.stateType);

  const nextState = {
    ...state,
    messages: [...state.messages, turn.userMessage, turn.assistantMessage],
    memoryEvents: [...state.memoryEvents, ...turn.memoryEvents]
  };
  let plan;
  try {
    plan = await planImageJobForCompletedTurn(nextState, {
      userMessage: turn.userMessage,
      assistantMessage: turn.assistantMessage,
      contextPack: turn.contextPack,
      promptModuleUsages: turn.promptModuleUsages,
      sidecar: turn.sidecar,
      sidecarTrace: turn.sidecarTrace,
      manualImage: false
    });
  } catch (error) {
    plan = { imageJobs: [], imageCue: turn.imageCue };
  }
  const job = plan.imageJobs[0];
  const captions = (job?.providerPayload?.characterPrompts ?? []).map((entry) => entry.prompt).join(", ");
  const composed = `${job?.prompt ?? ""}, ${captions}`;

  rows.push({
    index,
    latencyMs,
    source: turn.sidecarTrace.source,
    annotationFailure: turn.annotationFailureReason,
    narrativeChars: narrative.length,
    korean: /[가-힣]/u.test(narrative),
    leak: /assistant_text|image_cues|state_events|base_tags|character_prompts/u.test(narrative),
    cues: cues.length,
    generating: generating.length,
    withFrame: generating.filter((cue) => Boolean(cue.frame)).length,
    // Measured on the FINAL prompt, not the cue's own tags: the crop reaches NovelAI either because the model
    // wrote the shot tag or because DynamicChat derived it from the declared frame. Either way is a success —
    // what matters is that the rendered image is cropped the way the cue decided.
    shotTagAuthored: generating.filter((cue) =>
      /close|face focus|upper body|cowboy|full body|wide shot|portrait/iu.test((cue.baseTags ?? []).join(", "))
    ).length,
    withCast: generating.filter((cue) => (cue.characterPrompts ?? []).length > 0).length,
    withTags: generating.filter(
      (cue) => (cue.baseTags ?? []).length > 0 || (cue.characterPrompts ?? []).some((p) => p.prompt.trim())
    ).length,
    stateEvents: stateEvents.length,
    job: Boolean(job),
    subjectCount: /1girl|1boy|\dgirls|\dboys/iu.test(composed),
    shotTagInPrompt: /close-?up|face focus|upper body|cowboy shot|full body|wide shot|portrait/iu.test(job?.prompt ?? ""),
    // Only meaningful where there IS a registered character and the crop shows the head — a scenery cut or a
    // deliberate hip/leg close-up has no hair or eyes to inject, and counting those as misses reported a
    // failure that was really "there is nothing to inject here".
    identityApplicable:
      (job?.providerPayload?.characterPrompts ?? []).some((entry) => entry.characterId) &&
      (job?.providerPayload?.cue?.frame?.visibleRegions ?? ["head"]).includes("head"),
    // Injection is asserted by comparing against the registered profile's OWN saved tags, so this works on
    // either seed instead of hard-coding the demo's appearance.
    identityInjected: (job?.providerPayload?.characterPrompts ?? [])
      .filter((entry) => entry.characterId)
      .every((entry) => {
        const profile = state.visualProfiles.find((candidate) => candidate.characterId === entry.characterId);
        const savedTags = (profile?.positivePrompt ?? "")
          .split(",")
          .map((tag) => tag.trim().toLowerCase())
          .filter(Boolean);
        return savedTags.length === 0 || savedTags.some((tag) => entry.prompt.toLowerCase().includes(tag));
      }),
    koreanInPrompt: /[가-힣]/u.test(composed)
  });

  state = hydrateState(nextState);
  const row = rows.at(-1);
  console.log(
    `turn ${index + 1}: ${(row.latencyMs / 1000).toFixed(1)}s  narrative=${row.narrativeChars}자  cues=${row.generating}` +
      `  frame=${row.withFrame}  cast=${row.withCast}  state=${row.stateEvents}  job=${row.job ? "yes" : "no"}` +
      (row.annotationFailure ? `  ⚠ ${row.annotationFailure.slice(0, 60)}` : "")
  );
}

await vite.close();

const ok = rows.filter((row) => !row.error);
const rate = (predicate) => (ok.length === 0 ? 0 : Math.round((ok.filter(predicate).length / ok.length) * 100));
const sum = (pick) => ok.reduce((total, row) => total + pick(row), 0);
const cueTotal = sum((row) => row.generating);

console.log("\n────────── 적합성 요약 ──────────");
console.log(`턴 성공률                : ${rate(() => true)}%  (${ok.length}/${rows.length})`);
console.log(`실제 모델 응답 (fallback 아님): ${rate((row) => row.source === "llm")}%`);
console.log(`한국어 서사              : ${rate((row) => row.korean)}%`);
console.log(`스캐폴딩 누출 없음        : ${rate((row) => !row.leak)}%`);
console.log(`평균 서사 길이            : ${ok.length ? Math.round(sum((row) => row.narrativeChars) / ok.length) : 0}자`);
console.log(`평균 턴 지연              : ${ok.length ? (sum((row) => row.latencyMs) / ok.length / 1000).toFixed(1) : 0}s`);
console.log(`--`);
console.log(`이미지 cue 생성된 턴       : ${rate((row) => row.generating > 0)}%`);
console.log(`턴당 평균 cue             : ${ok.length ? (cueTotal / ok.length).toFixed(1) : 0}`);
console.log(`cue 중 frame 명시          : ${cueTotal ? Math.round((sum((row) => row.withFrame) / cueTotal) * 100) : 0}%`);
console.log(`cue가 직접 구도 태그 작성  : ${cueTotal ? Math.round((sum((row) => row.shotTagAuthored) / cueTotal) * 100) : 0}%`);
console.log(`최종 프롬프트에 구도 태그  : ${rate((row) => row.shotTagInPrompt)}%   ← frame에서 보강`);
console.log(`cue 중 인물 캡션 포함      : ${cueTotal ? Math.round((sum((row) => row.withCast) / cueTotal) * 100) : 0}%`);
console.log(`cue 중 태그 있음           : ${cueTotal ? Math.round((sum((row) => row.withTags) / cueTotal) * 100) : 0}%`);
console.log(`--`);
console.log(`상태 이벤트 추출된 턴      : ${rate((row) => row.stateEvents > 0)}%`);
console.log(`이미지 작업까지 도달       : ${rate((row) => row.job)}%`);
console.log(`피사체 수 태그 존재        : ${rate((row) => row.subjectCount)}%`);
const identityRows = ok.filter((row) => row.identityApplicable);
console.log(
  `저장 외형 주입             : ${
    identityRows.length === 0 ? "해당 없음" : `${Math.round((identityRows.filter((row) => row.identityInjected).length / identityRows.length) * 100)}% (인물 포함 컷 ${identityRows.length}건 기준)`
  }`
);
console.log(`프롬프트에 한국어 없음      : ${rate((row) => !row.koreanInPrompt)}%`);
const failures = rows.filter((row) => row.error || row.annotationFailure);
if (failures.length > 0) {
  console.log(`\n실패/경고 ${failures.length}건:`);
  for (const row of failures) {
    console.log(`  turn ${row.index + 1}: ${row.error ?? row.annotationFailure}`);
  }
}
