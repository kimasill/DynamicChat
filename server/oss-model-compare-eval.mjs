/**
 * Side-by-side comparison of local open-weight models.
 *
 * `eval:oss-suitability` answers "does this backend hold the contracts". This one answers the question that
 * actually decides which model to run: how much better is the WRITING, and what does it cost in latency?
 *
 * Structural compliance is easy to measure and easy to pass — a 4B model can emit valid JSON all day. Prose
 * quality is the part that degrades first and is not visible in a pass/fail. So alongside the contract
 * metrics this reports objective prose signals that separate a small model from a large one:
 *
 *   - 반복도       degenerate repetition (repeated 4-grams). The classic small-model failure: the same
 *                  clause recycled with slight variations until the budget runs out.
 *   - 어휘 다양성   type-token ratio over content words. A small model reaches for the same nouns.
 *   - 대사 비율     share of the turn that is spoken dialogue — simulation prose that is all narration and
 *                  no speech reads as a summary rather than a scene.
 *   - 문장 길이 분포 sentence-length variance. Uniform sentence length is a hallmark of weak generation.
 *
 * None of these is a quality score on its own; together they separate "structurally valid" from "worth
 * reading", and the printed samples let you judge the rest yourself.
 *
 * Usage:
 *   OSS_COMPARE_MODELS="qwen3:4b-instruct,qwen3:14b" OSS_COMPARE_TURNS=4 node server/oss-model-compare-eval.mjs
 */
import { createServer } from "vite";

const baseUrl = process.env.OSS_EVAL_BASE_URL ?? "http://127.0.0.1:11434/v1";
const provider = process.env.OSS_EVAL_PROVIDER ?? "ollama";
const apiKey = process.env.OSS_EVAL_API_KEY ?? "";
const turns = Number(process.env.OSS_COMPARE_TURNS ?? 4);
const contextTokens = Number(process.env.OSS_EVAL_CONTEXT ?? 16384);
const models = (process.env.OSS_COMPARE_MODELS ?? "qwen3:4b-instruct")
  .split(",")
  .map((entry) => entry.trim())
  .filter(Boolean);

try {
  const probe = await fetch(`${baseUrl}/models`, {
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : {},
    signal: AbortSignal.timeout(4000)
  });
  if (!probe.ok) throw new Error(`HTTP ${probe.status}`);
} catch (error) {
  console.log(`SKIP: ${baseUrl} 에 접속할 수 없습니다 (${error.message}).`);
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

const PROMPTS = [
  "서가 사이를 천천히 걸으며 그녀에게 말을 건다.",
  "그녀가 가리킨 책장 쪽으로 다가가 손을 뻗는다.",
  "창가로 걸어가 밖을 내다보며 조용히 묻는다.",
  "그녀에게 등을 돌리고 문 쪽으로 걷는다.",
  "무릎을 굽혀 바닥에 떨어진 종이를 줍는다.",
  "그녀의 얼굴을 가까이서 바라본다."
];

function createState(model) {
  const base = hydrateState(structuredClone(activeSeed));
  const llm = { ...base.llm, enabled: true, provider, baseUrl, model, apiKey, temperature: 0.8, maxTokens: 4000, contextTokens };
  return hydrateState({
    ...base,
    llm,
    imageTagLlm: { ...llm, enabled: true },
    simulation: { ...base.simulation, realtimeImageEnabled: true },
    // The seed's own character, so the roster matches the prompt modules the model is actually reading.
    // Its saved appearance is overridden with known tags so identity injection can be asserted exactly.
    characters: base.characters,
    visualProfiles: base.visualProfiles.map((profile) => ({
      ...profile,
      positivePrompt: "1girl, long black hair, green eyes, pale skin",
      defaultOutfitPrompt: "librarian uniform, pleated skirt, black thighhighs, brown loafers"
    })),
    memoryEvents: [],
    imageJobs: [],
    imageAssets: [],
    neuralMap: { ...base.neuralMap, enabled: false },
    novelAi: { ...base.novelAi, enabled: false, modelPreset: "NAID4.5C" },
    imageProfile: { ...base.imageProfile, enabled: true, triggerMode: "realtime_auto", generationCadence: "balanced", cooldownTurns: 0 }
  });
}

// ─── Prose signals ────────────────────────────────────────────────────────────────────────────────
// Korean has no whitespace-delimited morphology, so these operate on whitespace tokens with particles left
// attached. That is noisy in absolute terms but consistent BETWEEN models, which is all a comparison needs.

function tokenize(text) {
  return text
    .replace(/[.,!?"“”'‘’…—\-()[\]]/gu, " ")
    .split(/\s+/u)
    .map((token) => token.trim())
    .filter(Boolean);
}

/** Share of 4-gram positions that repeat elsewhere in the text — degenerate looping. */
function repetitionRatio(text) {
  const tokens = tokenize(text);
  if (tokens.length < 8) {
    return 0;
  }
  const seen = new Map();
  let repeated = 0;
  for (let index = 0; index + 4 <= tokens.length; index += 1) {
    const gram = tokens.slice(index, index + 4).join(" ");
    const count = (seen.get(gram) ?? 0) + 1;
    seen.set(gram, count);
    if (count > 1) {
      repeated += 1;
    }
  }
  return repeated / Math.max(1, tokens.length - 3);
}

/** Distinct tokens over total — how far the model reaches for vocabulary. */
function typeTokenRatio(text) {
  const tokens = tokenize(text);
  return tokens.length === 0 ? 0 : new Set(tokens).size / tokens.length;
}

/** Share of characters inside quotation marks. */
function dialogueRatio(text) {
  const quoted = [...text.matchAll(/[“"]([^”"]{1,400})[”"]/gu)].reduce((sum, match) => sum + match[1].length, 0);
  return text.length === 0 ? 0 : quoted / text.length;
}

function sentenceStats(text) {
  const sentences = text
    .split(/(?<=[.!?。…])\s+|\n+/u)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 1);
  if (sentences.length === 0) {
    return { count: 0, mean: 0, stdev: 0 };
  }
  const lengths = sentences.map((entry) => entry.length);
  const mean = lengths.reduce((sum, value) => sum + value, 0) / lengths.length;
  const variance = lengths.reduce((sum, value) => sum + (value - mean) ** 2, 0) / lengths.length;
  return { count: sentences.length, mean, stdev: Math.sqrt(variance) };
}

// ─── Run ──────────────────────────────────────────────────────────────────────────────────────────
const report = [];

for (const model of models) {
  console.log(`\n${"═".repeat(78)}\n▶ ${model}\n${"═".repeat(78)}`);
  let state = createState(model);
  const rows = [];

  for (let index = 0; index < turns; index += 1) {
    const userText = PROMPTS[index % PROMPTS.length];
    const startedAt = Date.now();
    let turn;
    try {
      turn = await runSimulationTurn(state, userText, false, { deferImagePlanning: true, deferMemoryIngest: true });
    } catch (error) {
      console.log(`  turn ${index + 1}: ERROR ${error instanceof Error ? error.message : String(error)}`);
      rows.push({ error: true });
      continue;
    }
    const latencyMs = Date.now() - startedAt;
    const narrative = turn.assistantMessage.content ?? "";
    const cues = (turn.sidecar?.imageCues ?? []).filter((cue) => cue.shouldGenerate);
    const stateEvents = (turn.sidecar?.memoryEvents ?? []).filter((event) => event.stateType);

    const nextState = {
      ...state,
      messages: [...state.messages, turn.userMessage, turn.assistantMessage],
      memoryEvents: [...state.memoryEvents, ...turn.memoryEvents]
    };
    let job;
    try {
      const plan = await planImageJobForCompletedTurn(nextState, {
        userMessage: turn.userMessage,
        assistantMessage: turn.assistantMessage,
        contextPack: turn.contextPack,
        promptModuleUsages: turn.promptModuleUsages,
        sidecar: turn.sidecar,
        sidecarTrace: turn.sidecarTrace,
        manualImage: false
      });
      job = plan.imageJobs[0];
    } catch {
      job = undefined;
    }

    const stats = sentenceStats(narrative);
    rows.push({
      latencyMs,
      chars: narrative.length,
      fallback: turn.sidecarTrace.source !== "llm",
      annotationFailure: turn.annotationFailureReason,
      repetition: repetitionRatio(narrative),
      ttr: typeTokenRatio(narrative),
      dialogue: dialogueRatio(narrative),
      sentences: stats.count,
      sentenceStdev: stats.stdev,
      cues: cues.length,
      withCast: cues.filter((cue) => (cue.characterPrompts ?? []).length > 0).length,
      stateEvents: stateEvents.length,
      job: Boolean(job),
      // Measured only over jobs that carry a registered character AND whose crop shows the head. A scenery
      // cut has no identity to inject, and a deliberate hip/leg close-up legitimately shows no hair or eyes
      // — counting either as a miss reported a failure that was really "there is nothing to inject here".
      registeredJob:
        (job?.providerPayload?.characterPrompts ?? []).some((entry) => entry.characterId) &&
        (job?.providerPayload?.cue?.frame?.visibleRegions ?? ["head"]).includes("head"),
      identityInjected: (job?.providerPayload?.characterPrompts ?? [])
        .filter((entry) => entry.characterId)
        .every((entry) => /long black hair/iu.test(entry.prompt))
    });

    if (rows.at(-1)?.registeredJob && !rows.at(-1)?.identityInjected) {
      console.log(
        `    ↳ 외형 미주입: frame=${JSON.stringify(job?.providerPayload?.cue?.frame)} caption=${(job?.providerPayload?.characterPrompts ?? [])
          .map((entry) => `${entry.characterId}:${entry.prompt.slice(0, 90)}`)
          .join(" | ")}`
      );
    }
    console.log(
      `  turn ${index + 1}: ${(latencyMs / 1000).toFixed(1)}s  ${narrative.length}자  ` +
        `반복 ${(repetitionRatio(narrative) * 100).toFixed(1)}%  어휘 ${(typeTokenRatio(narrative) * 100).toFixed(0)}%  ` +
        `대사 ${(dialogueRatio(narrative) * 100).toFixed(0)}%  cue ${cues.length}  state ${stateEvents.length}` +
        (turn.annotationFailureReason ? `  ⚠ ${turn.annotationFailureReason.slice(0, 50)}` : "")
    );

    if (index === 0) {
      console.log(`\n  ── 첫 턴 서사 ──\n${narrative.slice(0, 900).split("\n").map((line) => `  ${line}`).join("\n")}\n`);
    }
    state = hydrateState(nextState);
  }

  const ok = rows.filter((row) => !row.error);
  const avg = (pick) => (ok.length === 0 ? 0 : ok.reduce((sum, row) => sum + pick(row), 0) / ok.length);
  report.push({
    model,
    turns: ok.length,
    latency: avg((row) => row.latencyMs) / 1000,
    chars: avg((row) => row.chars),
    repetition: avg((row) => row.repetition) * 100,
    ttr: avg((row) => row.ttr) * 100,
    dialogue: avg((row) => row.dialogue) * 100,
    sentences: avg((row) => row.sentences),
    sentenceStdev: avg((row) => row.sentenceStdev),
    cues: avg((row) => row.cues),
    castRate: ok.length === 0 ? 0 : (ok.reduce((sum, row) => sum + row.withCast, 0) / Math.max(1, ok.reduce((sum, row) => sum + row.cues, 0))) * 100,
    stateRate: ok.length === 0 ? 0 : (ok.filter((row) => row.stateEvents > 0).length / ok.length) * 100,
    jobRate: ok.length === 0 ? 0 : (ok.filter((row) => row.job).length / ok.length) * 100,
    // Denominator is jobs that HAVE a registered character, not all jobs.
    identityRate: (() => {
      const withCharacter = ok.filter((row) => row.registeredJob);
      return withCharacter.length === 0 ? 100 : (withCharacter.filter((row) => row.identityInjected).length / withCharacter.length) * 100;
    })(),
    fallbacks: ok.filter((row) => row.fallback).length
  });
}

await vite.close();

console.log(`\n${"═".repeat(78)}\n비교 요약\n${"═".repeat(78)}`);
const columns = [
  ["모델", (row) => row.model, 22],
  ["지연", (row) => `${row.latency.toFixed(1)}s`, 7],
  // Latency alone conflates "slow model" with "long turn". Characters per second separates them, and is the
  // number that decides whether a partially-offloaded model is usable at all.
  ["자/초", (row) => (row.latency > 0 ? (row.chars / row.latency).toFixed(0) : "-"), 7],
  ["분량", (row) => `${Math.round(row.chars)}자`, 7],
  ["반복↓", (row) => `${row.repetition.toFixed(1)}%`, 7],
  ["어휘↑", (row) => `${row.ttr.toFixed(0)}%`, 6],
  ["대사", (row) => `${row.dialogue.toFixed(0)}%`, 6],
  ["문장", (row) => row.sentences.toFixed(1), 6],
  ["cue", (row) => row.cues.toFixed(1), 5],
  ["캐스팅", (row) => `${row.castRate.toFixed(0)}%`, 7],
  ["상태", (row) => `${row.stateRate.toFixed(0)}%`, 6],
  ["외형주입", (row) => `${row.identityRate.toFixed(0)}%`, 9]
];
console.log(columns.map(([head, , width]) => head.padEnd(width)).join(""));
console.log("─".repeat(columns.reduce((sum, [, , width]) => sum + width, 0)));
for (const row of report) {
  console.log(columns.map(([, pick, width]) => String(pick(row)).padEnd(width)).join(""));
}
console.log("\n반복↓ 은 낮을수록, 어휘↑ 는 높을수록 좋습니다.");
