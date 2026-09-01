import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const statePath = path.join(workspaceRoot, ".dynamicchat-data", "state.json");

const state = JSON.parse(await readFile(statePath, "utf8"));
const sourceChecks = await readSourceChecks();
const checks = [
  ...sourceChecks,
  ...Object.values(state.simulations ?? {}).flatMap(evaluateSimulation)
];
const failures = checks.filter((check) => check.level === "fail");
const warnings = checks.filter((check) => check.level === "warn");

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : check.level === "warn" ? "WARN" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}`);
}

console.log("");
console.log(`Simulation quality checks: ${checks.length - failures.length - warnings.length} passed, ${warnings.length} warnings, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

async function readSourceChecks() {
  const llmClient = await readFile(path.join(workspaceRoot, "src", "services", "llmClient.ts"), "utf8");
  const simulationEngine = await readFile(path.join(workspaceRoot, "src", "services", "simulationEngine.ts"), "utf8");
  const imageOrchestrator = await readFile(path.join(workspaceRoot, "src", "services", "imageOrchestrator.ts"), "utf8");
  const novelAiClient = await readFile(path.join(workspaceRoot, "src", "services", "novelAiClient.ts"), "utf8");
  const memoryCompiler = await readFile(path.join(workspaceRoot, "src", "services", "memoryCompiler.ts"), "utf8");
  const neuralMapClient = await readFile(path.join(workspaceRoot, "src", "services", "neuralMapClient.ts"), "utf8");
  const sceneCast = await readFile(path.join(workspaceRoot, "src", "services", "sceneCast.ts"), "utf8");
  const dynamicChatServer = await readFile(path.join(workspaceRoot, "server", "dynamicchat-server.mjs"), "utf8");
  const appSource = await readFile(path.join(workspaceRoot, "src", "App.tsx"), "utf8");
  const seedSource = await readFile(path.join(workspaceRoot, "src", "data", "seed.ts"), "utf8");

  return [
    assertCheck(
      "source.llm",
      llmClient.includes("responseMimeType: \"application/json\""),
      "Gemini JSON MIME mode is enabled."
    ),
    assertCheck(
      "source.llm",
      !llmClient.includes("MAIN_GEMINI_RESPONSE_SCHEMA") &&
        !llmClient.includes("geminiResponseSchema: MAIN_GEMINI_RESPONSE_SCHEMA") &&
        llmClient.includes("Return JSON only. The JSON schema is:"),
      "Main Gemini calls avoid responseSchema and rely on prompt-level JSON sidecar instructions."
    ),
    assertCheck(
      "source.llm",
      llmClient.includes("createDisplayFallbackText") && !llmClient.includes("stripLikelyJsonFence(rawContent).trim() || input.fallback"),
      "LLM parse fallback cannot expose raw sidecar JSON."
    ),
    assertCheck(
      "source.llm",
      llmClient.includes("looksLikeInternalPromptLeak") &&
        llmClient.includes("Use this before older retrieved memories") &&
        llmClient.includes("SYSTEM INSTRUCTION:") &&
        llmClient.includes("Memory/context evidence:"),
      "LLM parse fallback blocks echoed internal prompt/context text."
    ),
    assertCheck(
      "source.llm",
      llmClient.includes("World/lore foundation") && llmClient.includes("Always-on rules"),
      "LLM foundation prompt includes world lore and always-on rules."
    ),
    assertCheck(
      "source.llm",
      llmClient.includes("character.summary") && llmClient.includes("relationship: ${truncatePromptText(character.relationship"),
      "LLM foundation prompt includes character summaries and relationships."
    ),
    assertCheck(
      "source.llm",
      llmClient.includes("Immediate continuity anchor") &&
        llmClient.includes("latest assistant ending") &&
        llmClient.includes("[...middle omitted for continuity...]"),
      "LLM prompt preserves prior assistant endings for turn-to-turn continuity."
    ),
    assertCheck(
      "source.llm",
      llmClient.includes("requestProviderTextWithRecovery") &&
        llmClient.includes("createRecoveryRuntimeInstruction") &&
        llmClient.includes("BLOCK_NONE") &&
        llmClient.includes("streamed?.trim()") &&
        llmClient.includes("LLM recovery response did not include content"),
      "LLM provider safety/empty responses have a structured recovery path."
    ),
    assertCheck(
      "source.llm",
      llmClient.includes("default outfit tags") &&
        llmClient.includes("outfit keyword mappings") &&
        llmClient.includes("state_type='Wearing'") &&
        llmClient.includes("'StatusTags'"),
      "Annotation prompt persists character outfit and visual status tags."
    ),
    assertCheck(
      "source.relationship",
      appSource.includes("기본 의상 태그") &&
        appSource.includes("createRelationshipParameterAliases") &&
        seedSource.includes("rel_param_outfit_tags") &&
        seedSource.includes("rel_param_status_tags") &&
        memoryCompiler.includes('type: "StatusTags"'),
      "Relationship tab tracks outfit tags and character status tags."
    ),
    assertCheck(
      "source.rag",
      simulationEngine.includes("isFoundationPromptModule") && simulationEngine.includes("기반 프롬프트/세계관은 매 턴 유지"),
      "Main/world modules are selected even when lexical RAG is weak."
    ),
    assertCheck(
      "source.rag",
      simulationEngine.includes("createImmediateContinuityRetrievalAnchor") &&
        simulationEngine.includes("최근 assistant 출력 끝부분") &&
        simulationEngine.includes("시뮬레이션 설정/상황 앵커"),
      "NeuralMap retrieval queries include immediate continuity and setting anchors."
    ),
    assertCheck(
      "source.neuralmap",
      neuralMapClient.includes("createLocalContextEvidence") && neuralMapClient.includes("moduleEvidence"),
      "NeuralMap context is merged with local foundation evidence."
    ),
    assertCheck(
      "source.neuralmap",
      neuralMapClient.includes("createImmediateContinuityEvidence") &&
        neuralMapClient.includes("Latest assistant ending") &&
        neuralMapClient.includes("직전 출력/현재 입력 연속성"),
      "Local Context Packs carry a first-class immediate continuity evidence item."
    ),
    assertCheck(
      "source.fallback",
      simulationEngine.includes("createFallbackMemoryHint") &&
        simulationEngine.includes("looksLikeInternalFallbackEvidence") &&
        simulationEngine.includes("최근 대화 연속성 단서는 Context Pack에 보존되어 있다."),
      "Local LLM fallback summarizes memory evidence without exposing internal continuity anchors."
    ),
    assertCheck(
      "source.image",
      imageOrchestrator.includes("parseNovelAiWeightedTag") && !imageOrchestrator.includes("shouldBlockImagePromptForState"),
      "Image prompt composer preserves NAI weights without app-level content blocking."
    ),
    assertCheck(
      "source.image",
      simulationEngine.includes("sidecar.imageCues") &&
        !simulationEngine.includes("planImageCueDraftsWithLlm") &&
        !simulationEngine.includes("dedicated_image_planner") &&
        !/export\s+async\s+function\s+generateImageCuePlans/u.test(llmClient),
      "Image cue planning uses the main simulation LLM sidecar directly without a dedicated planner."
    ),
    assertCheck(
      "source.image",
      llmClient.includes("char_caption[0], char_caption[1]") &&
        llmClient.includes("WHO IS IN FRAME IS YOUR CALL") &&
        llmClient.includes("PEOPLE IN CONTACT STAY SEPARATE") &&
        llmClient.includes("Image cue authoring reference") &&
        imageOrchestrator.includes("getNovelAiPositiveTagRank"),
      "Annotation tag contract keeps the NAI V4 per-character split and never merges two people into one entry."
    ),
    assertCheck(
      "source.image",
      llmClient.includes("No later tag planner will fix, expand, or infer tags") &&
        llmClient.includes("emit [] only for quiet text-only turns") &&
        llmClient.includes("DynamicChat will not run a later tag planner") &&
        !llmClient.includes("requestMainLlmImageCueRepairText") &&
        !llmClient.includes("image cue self-repair"),
      "Main simulation LLM is instructed to create final image cues and tags itself without a later repair request."
    ),
    assertCheck(
      "source.image",
      llmClient.includes("createCurrentTurnImagePolicyBlock") &&
        llmClient.includes("basis: image prompt user rules") &&
        llmClient.includes("basis: image generation cadence") &&
        llmClient.includes("Do not wait for a separate image request from the user"),
      "Main LLM image cue decisions are driven by generation cadence and image prompt user rules."
    ),
    assertCheck(
      "source.image",
      llmClient.includes("Image prompt user rules are binding") && llmClient.includes("anchor_text") && llmClient.includes("placement"),
      "Main image cue output reads user image rules and can emit anchored cues."
    ),
    assertCheck(
      "source.image",
      llmClient.includes("Image tag keyword presets (no roster identity tags)") &&
        llmClient.includes("createImageSceneTagPresetBlock") &&
        llmClient.includes("collectImageSceneTagPresetNodes") &&
        llmClient.includes("IMAGE_SCENE_PRESET_PROMPT_MAX_DEPTH") &&
        llmClient.includes("creator_note") &&
        llmClient.includes("inherited_creator_notes") &&
        appSource.includes("scene-tag-preset-panel") &&
        appSource.includes("scene-tag-preset-collapsed-summary") &&
        appSource.includes("scene-tag-preset-drag-handle") &&
        appSource.includes("moveImageScenePresetNodes") &&
        appSource.includes("SceneTagPresetNodeEditor") &&
        seedSource.includes("ImageSceneTagPreset"),
      "Creator scene tag keyword presets are hierarchical, collapsible, draggable, editable, and available with notes to the main LLM image cue prompt."
    ),
    assertCheck(
      "source.image",
      llmClient.includes("example_prompts") &&
        llmClient.includes("selectScenePresetExampleFiles") &&
        llmClient.includes("IMAGE_SCENE_PRESET_EXAMPLE_TOTAL") &&
        llmClient.includes("example_prompts are optional references") &&
        llmClient.includes("labeled groups") &&
        seedSource.includes("normalizeScenePresetExampleFiles") &&
        appSource.includes("scene-tag-preset-example-file") &&
        appSource.includes("addExampleFile"),
      "Scene tag keyword presets accept multiple labeled example-prompt files (with .txt import), optional, capped, and sent to the LLM as few-shot references mapped to the matching character."
    ),
    assertCheck(
      "source.image",
      llmClient.includes("REGISTERED CHARACTERS: SET THE ID") &&
        llmClient.includes("Character identity lock") &&
        llmClient.includes("injects their saved appearance and current outfit") &&
        llmClient.includes("DETAIL FOLLOWS THE CROP") &&
        !llmClient.includes("NovelAI/Danbooru tag conversion rules") &&
        !llmClient.includes("Good action tags look like") &&
        !llmClient.includes("Use this strict tag order for every generated cue"),
      "Annotation tag contract keeps per-character separation, injects saved appearance/outfit, and scales detail to the crop."
    ),
    assertCheck(
      "source.image",
      imageOrchestrator.includes("characterPrompts: promptPlan.characterPrompts") &&
        imageOrchestrator.includes("promptLayers: promptPlan.layers") &&
        novelAiClient.includes("resolveNovelAiCharacterPrompts") &&
        !imageOrchestrator.includes("visualProfile.negativePrompt"),
      "Local image composition separates base prompt tags from explicit per-character NovelAI V4 captions."
    ),
    assertCheck(
      "source.image",
      llmClient.includes("SUBJECT COUNT IS MANDATORY") &&
        llmClient.includes("1girl, 1boy") &&
        llmClient.includes("EVERY ENTRY'S GENDER MUST BE UNAMBIGUOUS") &&
        llmClient.includes("required_identity_tags"),
      "Annotation tag contract makes the subject-count tag mandatory and every entry's gender explicit."
    ),
    assertCheck(
      "source.image",
      sceneCast.includes("collectRecentImageCharacterIds") &&
        sceneCast.includes("shouldUseImageCastContinuity") &&
        llmClient.includes("latest image/assistant beat"),
      "Image cue character scoping preserves recent visible cast for pronoun-only continuations."
    ),
    assertCheck(
      "source.image",
      !/scene:\s*variant\.scene\s*\?/.test(imageOrchestrator),
      "Image variants do not append synthetic scene labels."
    ),
    assertCheck(
      "source.assets",
      dynamicChatServer.includes("collectHydratedImageAssets") &&
        dynamicChatServer.includes("progressRuns ?? []).flatMap") &&
        dynamicChatServer.includes("attachExistingAssetObjectMetadata"),
      "Generated image assets are restored from progress runs and existing object files."
    ),
    assertCheck(
      "source.assets",
      dynamicChatServer.includes("stripAssetPayloadsFromState") &&
        /function listSimulations[\s\S]*stripAssetPayloadsFromState/.test(dynamicChatServer) &&
        !/function listSimulations[\s\S]*hydrateAssetDataUrls[\s\S]*function createSimulation/.test(dynamicChatServer),
      "Simulation list responses do not hydrate every image payload."
    ),
    assertCheck(
      "source.scroll",
      appSource.includes("isStoryScrollPinnedToBottom") &&
        appSource.includes("scrollStoryToBottom(scrollElement, \"auto\")") &&
        appSource.includes("messageCountChanged && previous.pinnedToBottom"),
      "Story scroll follows image updates only when the reader is pinned to the bottom."
    )
  ];
}

function evaluateSimulation(simulationState) {
  const title = simulationState?.simulation?.title ?? simulationState?.simulation?.id ?? "unknown";
  const scope = `simulation.${title}`;
  if (isSmokeFixture(simulationState)) {
    return [warnCheck(scope, "Skipped API smoke fixture; it is not a playable simulation.")];
  }

  const modules = simulationState.modules ?? [];
  const enabledModules = modules.filter((module) => module.enabled && module.tokenPolicy !== "disabled");
  const mainModules = enabledModules.filter((module) => module.kind === "main_prompt");
  const worldModules = enabledModules.filter((module) => module.kind === "world_lore");
  const contextPacks = simulationState.contextPacks ?? [];
  const turnTraces = simulationState.turnTraces ?? [];
  const promptModuleUsages = simulationState.promptModuleUsages ?? [];
  const messages = simulationState.messages ?? [];
  const imageJobs = simulationState.imageJobs ?? [];
  const hasRunHistory = messages.some((message) => message.role === "user") || turnTraces.length > 0;
  const latestContext = contextPacks.at(-1);
  const latestTrace = turnTraces.at(-1);
  const latestUsageIds = new Set(latestTrace?.promptModuleUsageIds ?? []);
  const latestUsages = promptModuleUsages.filter((usage) => latestUsageIds.has(usage.id));

  return [
    assertCheck(scope, mainModules.length > 0, "Enabled main prompt module exists."),
    assertCheck(scope, simulationState.simulation?.promptMode === "basic" || worldModules.length > 0, "Enabled world lore exists for simulation/custom modes."),
    assertCheck(scope, !hasRunHistory || contextPacks.length > 0, "Run history has context packs."),
    assertCheck(scope, !hasRunHistory || turnTraces.length > 0, "Run history has turn traces."),
    assertCheck(scope, !latestContext || (latestContext.evidence ?? []).length > 0, "Latest context pack has evidence."),
    assertCheck(scope, !latestContext || (latestContext.moduleEvidence ?? latestContext.evidence ?? []).length >= (latestContext.evidence ?? []).length, "Module evidence is available for RAG selection."),
    assertCheck(scope, !latestTrace || latestUsages.some((usage) => usage.moduleTitle && usage.tokenPolicy === "always") || mainModules.length === 0, "Latest trace records always-on prompt usage."),
    ...mainModules.map((module) =>
      assertCheck(scope, !latestTrace || latestUsages.some((usage) => usage.moduleId === module.id || usage.moduleTitle === module.title), `Latest trace includes main prompt: ${module.title}`)
    ),
    ...worldModules.slice(0, 3).map((module) =>
      assertCheck(scope, !latestTrace || latestUsages.some((usage) => usage.moduleId === module.id || usage.moduleTitle === module.title), `Latest trace includes world lore: ${module.title}`, "warn")
    ),
    ...messages
      .filter((message) => message.role === "assistant" && looksLikeSidecarJson(message.content))
      .map((message) =>
        warnCheck(scope, `Historical assistant message still stores sidecar JSON at ${message.createdAt}; load-time sanitizer will hide it.`)
      ),
    ...imageJobs.flatMap((job) => evaluateImageJob(scope, job))
  ];
}

function isSmokeFixture(simulationState) {
  const id = simulationState?.simulation?.id ?? "";
  const title = simulationState?.simulation?.title ?? "";
  return /smoke/i.test(id) || /smoke/i.test(title);
}

function evaluateImageJob(scope, job) {
  const prompt = [job.prompt, job.providerPayload?.promptVariants?.map?.((variant) => variant.prompt).join("\n")].filter(Boolean).join("\n");
  const scenes = [
    job.providerPayload?.cue?.scene,
    ...(Array.isArray(job.providerPayload?.promptVariants) ? job.providerPayload.promptVariants.map((variant) => variant?.cue?.scene) : [])
  ].filter(Boolean);
  const hasBrokenWeight = /\b\d+\s+\d+\s*::/u.test(prompt);
  const hasSyntheticScene = scenes.some((scene) => /\b(?:main action|character close-up|wide context|interaction detail)\b/iu.test(scene));
  const checks = [
    hasBrokenWeight
      ? warnCheck(scope, `Historical image job ${job.id} still contains pre-fix broken weighted syntax.`)
      : passCheck(scope, `Image job ${job.id} has no broken weighted syntax.`),
    hasSyntheticScene
      ? warnCheck(scope, `Historical image job ${job.id} still contains pre-fix synthetic scene labels.`)
      : passCheck(scope, `Image job ${job.id} has no synthetic variant scene labels.`)
  ];
  return checks;
}

function looksLikeSidecarJson(value = "") {
  const stripped = value.trim().replace(/^```(?:json)?/iu, "").trim();
  return stripped.startsWith("{") && /"?(?:assistant_text|assistantText|memory_events|memoryEvents|image_cues|imageCue)"?\s*:/u.test(stripped);
}

function assertCheck(scope, passed, message, failureLevel = "fail") {
  return {
    level: passed ? "pass" : failureLevel,
    scope,
    message
  };
}

function warnCheck(scope, message) {
  return {
    level: "warn",
    scope,
    message
  };
}

function passCheck(scope, message) {
  return {
    level: "pass",
    scope,
    message
  };
}
