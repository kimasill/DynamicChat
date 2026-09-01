/**
 * Where the turn's prompt budget actually goes.
 *
 * "The context is too small" is not an actionable finding on its own — the question is which block is large
 * and whether it is compressible. This prints the breakdown for a simulation seed so the lever is a measured
 * number rather than a guess.
 *
 * Usage:  node server/prompt-size-breakdown.mjs [demo|womanlife]
 */
import { createServer } from "vite";

const CHARS_PER_TOKEN = 2.6;
const tokens = (text) => Math.ceil((text ?? "").length / CHARS_PER_TOKEN);

const vite = await createServer({ server: { middlewareMode: true }, appType: "custom", logLevel: "error" });
const { seedState, hydrateState } = await vite.ssrLoadModule("/src/data/seed.ts");
const { womanLifeSeedState } = await vite.ssrLoadModule("/src/data/womanLifeSeed.ts");

const which = process.argv[2] ?? "womanlife";
const state = hydrateState(structuredClone(which === "womanlife" ? womanLifeSeedState : seedState));

const modules = state.modules ?? [];
const enabled = modules.filter((module) => module.enabled);
// `always` modules are in every prompt regardless of retrieval, so they are the floor the context must fit.
const alwaysOn = enabled.filter((module) => module.tokenPolicy === "always");
const retrieved = enabled.filter((module) => module.tokenPolicy !== "always");

const groupByKind = (list) => {
  const byKind = new Map();
  for (const module of list) {
    const current = byKind.get(module.kind) ?? { count: 0, chars: 0 };
    byKind.set(module.kind, { count: current.count + 1, chars: current.chars + module.body.length });
  }
  return [...byKind.entries()].sort((a, b) => b[1].chars - a[1].chars);
};

const transcript = (state.messages ?? []).slice(-12).map((message) => message.content ?? "").join("\n");
const personas = (state.characters ?? []).map((c) => `${c.summary ?? ""}${c.relationship ?? ""}${(c.traits ?? []).join(",")}`).join("\n");
const visualProfiles = (state.visualProfiles ?? [])
  .map((p) => `${p.positivePrompt ?? ""}${p.defaultOutfitPrompt ?? ""}${p.negativePrompt ?? ""}`)
  .join("\n");
const scenePresets = (state.imageScenePresets ?? [])
  .map((p) => `${p.label ?? ""}${p.tags ?? ""}${(p.keywords ?? []).join(",")}`)
  .join("\n");

const rows = [
  ["시뮬레이션 전제 (premise)", state.simulation?.premise ?? ""],
  ["시뮬레이션 설명", state.simulation?.description ?? ""],
  ["이미지 사용자 규정", state.imageProfile?.userRules ?? ""],
  ["항상 적용 모듈 (always)", alwaysOn.map((m) => m.body).join("\n")],
  ["검색 대상 모듈 (RAG 후보)", retrieved.map((m) => m.body).join("\n")],
  ["캐릭터 페르소나", personas],
  ["비주얼 프로파일", visualProfiles],
  ["장면 태그 프리셋", scenePresets],
  ["최근 대화 12개", transcript]
];

console.log(`seed: ${which}   캐릭터 ${(state.characters ?? []).length}명   모듈 ${modules.length}개 (활성 ${enabled.length}, 항상 ${alwaysOn.length})`);
console.log(`cadence: ${state.imageProfile?.generationCadence}   등급: ${state.simulation?.contentRating}\n`);

const width = Math.max(...rows.map(([label]) => label.length));
let foundationTokens = 0;
for (const [label, text] of rows) {
  const t = tokens(text);
  if (!label.startsWith("검색 대상")) {
    foundationTokens += t;
  }
  console.log(`${label.padEnd(width)}  ${String(text.length).padStart(7)}자  ${String(t).padStart(6)} 토큰`);
}
console.log(`\n항상 들어가는 합계(대략)     ${String(foundationTokens).padStart(6)} 토큰`);
console.log("(검색 대상 모듈은 턴마다 일부만 선택되므로 합계에서 제외)");

console.log("\n항상 적용 모듈 — 종류별:");
for (const [kind, info] of groupByKind(alwaysOn)) {
  console.log(`  ${kind.padEnd(18)} ${String(info.count).padStart(3)}개  ${String(info.chars).padStart(7)}자  ${String(Math.ceil(info.chars / CHARS_PER_TOKEN)).padStart(6)} 토큰`);
}

console.log("\n항상 적용 모듈 — 큰 것부터:");
for (const module of [...alwaysOn].sort((a, b) => b.body.length - a.body.length).slice(0, 10)) {
  console.log(`  ${String(module.body.length).padStart(6)}자  ${String(Math.ceil(module.body.length / CHARS_PER_TOKEN)).padStart(5)} 토큰  [${module.kind}] ${module.title}`);
}

// The blocks above are the creator's own content. The two below are what DynamicChat actually sends, which
// includes every rule block, cast guard and state summary the runtime generates on top of that content.
const { __buildNarrativePromptForTest, __buildAnnotationPromptForTest } = await vite.ssrLoadModule(
  "/src/services/llmClient.ts"
);
const selected = retrieved.slice(0, 12);
const narrative = __buildNarrativePromptForTest(state, { userText: "테스트", modules: selected, evidence: [] }, 2000);
const narrativeCompact = __buildNarrativePromptForTest(state, { userText: "테스트", modules: selected, evidence: [] }, 2000, "compact");

console.log("\n실제로 전송되는 프롬프트:");
console.log(`  서사 - 런타임 지시문      ${String(narrative.runtimeInstruction.length).padStart(7)}자  ${String(tokens(narrative.runtimeInstruction)).padStart(6)} 토큰`);
console.log(`  서사 - 컨텍스트 블록      ${String(narrative.contextBlock.length).padStart(7)}자  ${String(tokens(narrative.contextBlock)).padStart(6)} 토큰   (RAG 모듈 ${selected.length}개 포함)`);
console.log(`  서사 합계                ${String(narrative.runtimeInstruction.length + narrative.contextBlock.length).padStart(7)}자  ${String(tokens(narrative.runtimeInstruction) + tokens(narrative.contextBlock)).padStart(6)} 토큰`);
console.log(`  서사 합계 (compact)      ${String(narrativeCompact.runtimeInstruction.length + narrativeCompact.contextBlock.length).padStart(7)}자  ${String(tokens(narrativeCompact.runtimeInstruction) + tokens(narrativeCompact.contextBlock)).padStart(6)} 토큰`);
for (const tier of ["full", "compact", "minimal"]) {
  const built = __buildAnnotationPromptForTest(state, tier);
  console.log(`  주석 (${tier.padEnd(7)})         ${String(built.length).padStart(7)}자  ${String(tokens(built)).padStart(6)} 토큰`);
}

await vite.close();
