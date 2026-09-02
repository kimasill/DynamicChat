/**
 * End-to-end run check: does a turn actually reach NovelAI, carrying the right prompt?
 *
 * The other evals each cover one leg. simulation-quality checks the modules, image-prompt-quality checks
 * prompt assembly against synthetic cues, oss-model-suitability measures how often a local model holds the
 * contract. None of them drives the whole chain, so the parts could each pass while the seam between two
 * of them was broken — which is where the defects have actually been.
 *
 * This runs REAL turns against a real local model and follows one all the way out:
 *
 *   user text -> narrative pass -> annotation pass -> image cues -> planned jobs -> queue -> NovelAI HTTP
 *
 * The last leg is deliberately expected to FAIL with an authentication error. Without a NovelAI token
 * that is as far as the chain can go, and a 401 from novelai.net is the proof that the request was
 * actually assembled and sent rather than short-circuited somewhere upstream. `mock` and `disabled` both
 * return an empty result from the same function, so "no image" on its own proves nothing.
 *
 * Requires a local OpenAI-compatible model (Ollama by default) and SKIPs cleanly without one.
 *
 *   OSS_EVAL_MODEL=qwen3:14b node server/end-to-end-run-eval.mjs
 *
 * Env: OSS_EVAL_BASE_URL, OSS_EVAL_MODEL, OSS_EVAL_PROVIDER, OSS_EVAL_CONTEXT, OSS_EVAL_SEED (demo|womanlife),
 *      E2E_TURNS, E2E_CADENCES (comma list), E2E_SKIP_DISPATCH=1 to stay off the network.
 */
import { createServer } from "vite";

const baseUrl = process.env.OSS_EVAL_BASE_URL ?? "http://127.0.0.1:11434/v1";
const model = process.env.OSS_EVAL_MODEL ?? "qwen3:14b";
const provider = process.env.OSS_EVAL_PROVIDER ?? "ollama";
const apiKey = process.env.OSS_EVAL_API_KEY ?? "";
const contextTokens = Number(process.env.OSS_EVAL_CONTEXT ?? 16384);
const seedName = process.env.OSS_EVAL_SEED ?? "demo";
const turnsPerCadence = Number(process.env.E2E_TURNS ?? 1);
const cadences = (process.env.E2E_CADENCES ?? "balanced,rich,image_progression").split(",").map((c) => c.trim()).filter(Boolean);
const skipDispatch = process.env.E2E_SKIP_DISPATCH === "1";

try {
  const probe = await fetch(`${baseUrl}/models`, {
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {},
    signal: AbortSignal.timeout(4000)
  });
  if (!probe.ok) throw new Error(`HTTP ${probe.status}`);
} catch (error) {
  console.log(`SKIP: ${baseUrl} 에 접속할 수 없습니다 (${error.message}). OSS_EVAL_BASE_URL 을 지정하세요.`);
  process.exit(0);
}

const checks = [];
function check(group, condition, label, detail) {
  checks.push({ group, ok: Boolean(condition), label, detail });
  if (!condition) console.log(`  FAIL ${group}: ${label}${detail ? ` — ${detail}` : ""}`);
}

const vite = await createServer({ server: { middlewareMode: true }, appType: "custom", logLevel: "error" });
const { seedState, hydrateState } = await vite.ssrLoadModule("/src/data/seed.ts");
const { womanLifeSeedState } = await vite.ssrLoadModule("/src/data/womanLifeSeed.ts");
const { runSimulationTurn, planImageJobForCompletedTurn } = await vite.ssrLoadModule("/src/services/simulationEngine.ts");
const { executeImageJob } = await vite.ssrLoadModule("/src/services/imageOrchestrator.ts");
// The frame contract is asserted with the app's OWN classifier rather than a list written here. A
// hand-written "footwear" regex got this wrong first time: thighhighs, stockings and socks are LEG
// garments, so a cowboy shot — which shows legs but not feet — keeps them correctly.
const { readTagBodyRegion, isGarmentTag } = await vite.ssrLoadModule("/src/services/imageFrame.ts");

const activeSeed = seedName === "womanlife" ? womanLifeSeedState : seedState;
const base = hydrateState(structuredClone(activeSeed));
const llm = { ...base.llm, enabled: true, provider, baseUrl, model, apiKey, temperature: 0.8, maxTokens: 4000, contextTokens };

// Known appearance and outfit on the demo seed so identity and outfit injection can be asserted exactly.
// The full simulation already carries its own profiles and is left alone.
const KNOWN_APPEARANCE = "1girl, long black hair, green eyes, pale skin";
const KNOWN_OUTFIT = "librarian uniform, pleated skirt, black thighhighs, brown loafers";
const visualProfiles =
  seedName === "womanlife"
    ? base.visualProfiles
    : base.visualProfiles.map((p) => ({ ...p, positivePrompt: KNOWN_APPEARANCE, defaultOutfitPrompt: KNOWN_OUTFIT }));

function freshState(cadence) {
  return hydrateState({
    ...base,
    llm,
    imageTagLlm: { ...llm, enabled: true },
    simulation: { ...base.simulation, realtimeImageEnabled: true },
    visualProfiles,
    messages: [],
    memoryEvents: [],
    imageJobs: [],
    imageAssets: [],
    neuralMap: { ...base.neuralMap, enabled: false },
    // enabled so jobs are planned; `direct` with no token so the dispatch leg reaches the real endpoint.
    novelAi: { ...base.novelAi, enabled: true, requestMode: "direct", apiKey: "", modelPreset: "NAID4.5C" },
    imageProfile: {
      ...base.imageProfile,
      enabled: true,
      triggerMode: "realtime_auto",
      generationCadence: cadence,
      cooldownTurns: 0
    }
  });
}

const PROMPTS = {
  demo: ["서가 사이를 천천히 걸으며 그녀에게 말을 건다.", "책장 위쪽 선반으로 손을 뻗는다.", "그녀의 얼굴을 가까이서 바라본다."],
  womanlife: ["거울 앞에 서서 오늘 입고 나갈 옷을 고른다.", "옷장 위 선반에서 가방을 꺼내려 손을 뻗는다.", "화장대 거울에 얼굴을 가까이 대고 화장을 고친다."]
};
const prompts = PROMPTS[seedName] ?? PROMPTS.demo;

/* The floors CLAUDE.md documents. balanced/sparse deliberately have none: one cut is the right answer
   there, and forcing more would manufacture images nobody asked for. */
const CADENCE_FLOOR = { image_progression: 10, paragraph: 6, rich: 2, balanced: 0, sparse: 0 };

const SHOT_TAG = /close-?up|face focus|upper body|cowboy shot|full body|wide shot|portrait/iu;
const SUBJECT_TAG = /\b(?:1girl|1boy|\dgirls|\dboys|solo)\b/iu;

let dispatched = null;

// Frame declaration is the MODEL's contribution and varies run to run, so it is reported as a rate
// with a floor rather than asserted per cut — a hard assert would make the eval flaky against
// ordinary model variance instead of catching a regression.
const FRAME_RATE_FLOOR = Number(process.env.E2E_FRAME_RATE_FLOOR ?? 0.8);

for (const cadence of cadences) {
  console.log(`\n────────── cadence: ${cadence} ──────────`);
  let state = freshState(cadence);
  const frameStats = { total: 0, declared: 0 };

  for (let index = 0; index < turnsPerCadence; index += 1) {
    const userText = prompts[index % prompts.length];
    const startedAt = Date.now();
    let turn;
    try {
      turn = await runSimulationTurn(state, userText, false, { deferImagePlanning: true, deferMemoryIngest: true });
    } catch (error) {
      check(cadence, false, `턴 ${index + 1} 실행`, error instanceof Error ? error.message : String(error));
      continue;
    }
    const latency = ((Date.now() - startedAt) / 1000).toFixed(1);
    const narrative = turn.assistantMessage.content ?? "";
    const cues = turn.sidecar?.imageCues ?? [];
    const generating = cues.filter((cue) => cue.shouldGenerate);

    check(cadence, turn.sidecarTrace.source === "llm", `턴 ${index + 1}: 실제 모델 응답 (fallback 아님)`, turn.sidecarTrace.source);
    check(cadence, /[가-힣]/u.test(narrative), `턴 ${index + 1}: 한국어 서사`);
    check(cadence, !/assistant_text|image_cues|state_events|base_tags|character_prompts/u.test(narrative), `턴 ${index + 1}: 스캐폴딩 누출 없음`);

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
      check(cadence, false, `턴 ${index + 1}: 이미지 작업 계획`, error instanceof Error ? error.message : String(error));
      plan = { imageJobs: [] };
    }
    const jobs = plan.imageJobs ?? [];
    // A cue can legitimately resolve to an EXISTING asset instead of a new job — the same cut already
    // rendered, so it is shown rather than paid for twice. What must never happen is a generating cue that
    // produces neither: that is an image the turn promised and will not deliver.
    const reused = plan.reusedAssetIds ?? [];

    const floor = CADENCE_FLOOR[cadence] ?? 0;
    if (cadence === "image_progression") {
      check(cadence, generating.length === floor, `턴 ${index + 1}: cue 수량 == ${floor} (image_progression은 정확히)`, `${generating.length}개`);
    } else if (floor > 0) {
      check(cadence, generating.length >= floor, `턴 ${index + 1}: cue 수량 >= ${floor}`, `${generating.length}개`);
    } else {
      check(cadence, generating.length >= 1, `턴 ${index + 1}: cue 최소 1개`, `${generating.length}개`);
    }
    // The quantity that actually reaches the provider is the JOB count, not the cue count. A cue that is
    // planned but produces no job is an image the user was promised and will not get.
    check(
      cadence,
      jobs.length + reused.length === generating.length,
      `턴 ${index + 1}: 생성 cue가 전부 큐 또는 재사용으로 이어짐`,
      `jobs=${jobs.length} reused=${reused.length} cues=${generating.length}`
    );

    console.log(
      `  turn ${index + 1}: ${latency}s  narrative=${narrative.length}자  cues=${generating.length}  jobs=${jobs.length}` +
        (reused.length ? `  reused=${reused.length}` : "")
    );

    for (const [jobIndex, job] of jobs.entries()) {
      const captions = (job.providerPayload?.characterPrompts ?? []).map((e) => e.prompt).join(", ");
      const composed = `${job.prompt ?? ""}, ${captions}`;
      const frame = job.providerPayload?.cue?.frame;
      const label = `턴 ${index + 1} 작업 ${jobIndex + 1}`;

      check(cadence, SHOT_TAG.test(job.prompt ?? ""), `${label}: 구도 태그 존재`, (job.prompt ?? "").slice(0, 60));
      check(cadence, SUBJECT_TAG.test(composed), `${label}: 피사체 수 태그 존재`);
      check(cadence, !/[가-힣]/u.test(composed), `${label}: 프롬프트에 한국어 없음`);
      // What must hold every time is the composition tag asserted above. When the model omits
      // `frame`, resolveImageCueFrame tries to infer one from the cut's own tags — and the cuts that
      // lack a frame also lack any composition tag (pure scene tags), so there is nothing to infer
      // and no code defect. The cost is that region gating is skipped for that one cut.
      frameStats.total += 1;
      if (frame?.shot) frameStats.declared += 1;

      const named = (job.providerPayload?.characterPrompts ?? []).filter((e) => e.characterId);
      const showsHead = (frame?.visibleRegions ?? ["head"]).includes("head");
      if (named.length && showsHead && seedName !== "womanlife") {
        const appearanceHit = named.every((e) =>
          KNOWN_APPEARANCE.split(",").map((t) => t.trim().toLowerCase()).some((t) => e.prompt.toLowerCase().includes(t))
        );
        check(cadence, appearanceHit, `${label}: 저장된 외형이 캡션에 주입됨`);
      }
      // Outfit continuity: the CURRENT outfit must appear whenever the crop shows a region it covers.
      const showsBody = (frame?.visibleRegions ?? []).some((r) => ["torso", "hips", "legs"].includes(r));
      if (named.length && showsBody && seedName !== "womanlife") {
        const outfitHit = named.some((e) => /uniform|skirt/iu.test(e.prompt));
        check(cadence, outfitHit, `${label}: 현재 의상이 캡션에 주입됨`, captions.slice(0, 70));
      }
      // Frame gating, stated as the code states it: every garment tag that reaches the provider must belong
      // to a region this crop actually shows. That covers the stored outfit as well as anything the model
      // wrote, which is the whole point of the frame system.
      if (frame) {
        const regions = frame.visibleRegions ?? [];
        const offenders = captions
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean)
          .filter((tag) => {
            if (!isGarmentTag(tag)) return false;
            const region = readTagBodyRegion(tag);
            return region !== undefined && !regions.includes(region);
          });
        check(
          cadence,
          offenders.length === 0,
          `${label}: 프레임이 가린 부위의 의상 태그 없음`,
          offenders.length ? `shot=${frame.shot} regions=[${regions.join("|")}] 위반=[${offenders.join(", ")}]` : ""
        );
      }
    }

    if (!dispatched && jobs.length) dispatched = { state: hydrateState(nextState), job: jobs[0] };
    state = hydrateState(nextState);
  }

  if (frameStats.total > 0) {
    const rate = frameStats.declared / frameStats.total;
    console.log(`  frame 선언율: ${Math.round(rate * 100)}% (${frameStats.declared}/${frameStats.total})`);
    check(
      cadence,
      rate >= FRAME_RATE_FLOOR,
      `frame 선언율 >= ${Math.round(FRAME_RATE_FLOOR * 100)}%`,
      `${Math.round(rate * 100)}% (${frameStats.declared}/${frameStats.total})`
    );
  }
}

/* ---------- the last leg: queue -> NovelAI ---------- */
console.log(`\n────────── 큐 → NovelAI 호출 ──────────`);
if (skipDispatch) {
  console.log("  E2E_SKIP_DISPATCH=1 — 네트워크 호출 건너뜀");
} else if (!dispatched) {
  check("dispatch", false, "디스패치할 작업이 없음");
} else {
  const { state, job } = dispatched;
  console.log(`  endpoint: ${state.novelAi.endpoint}   token: (없음)`);
  const started = Date.now();
  const result = await executeImageJob(state, job);
  const took = ((Date.now() - started) / 1000).toFixed(1);
  const error = result.job.error ?? "";
  console.log(`  status=${result.job.status}  ${took}s  error=${error.slice(0, 120)}`);

  check("dispatch", result.job.status === "failed", "토큰 없이 실행하면 작업이 실패로 끝난다", result.job.status);
  // The distinguishing evidence: a short-circuit upstream produces one of the two canned messages below,
  // while a real request that left the machine comes back carrying the provider's HTTP status.
  check(
    "dispatch",
    !/연동이 꺼져 있어|mock 모드입니다/u.test(error),
    "업스트림에서 단락되지 않고 실제 요청이 나갔다",
    error.slice(0, 90)
  );
  check("dispatch", /\b(401|402|403)\b|unauthor|credential|token/iu.test(error), "NovelAI가 인증 실패로 응답했다", error.slice(0, 90));
  check("dispatch", result.assets.length === 0, "이미지 자산은 생성되지 않았다 (토큰이 없으므로)");
}

await vite.close();

const failures = checks.filter((c) => !c.ok);
console.log("");
console.log(`End-to-end run checks: ${checks.length - failures.length} passed, ${failures.length} failures`);
if (failures.length) process.exitCode = 1;
