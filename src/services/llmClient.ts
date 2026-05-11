import type { AppState, AssistantImageCueDraft, AssistantSidecar, ContextEvidence, PromptModule } from "../types";
import { createImageUserRulesForContentRating, isAdultContentMode } from "./contentRating";
import {
  createCurrentCharacterImageStatePrompt,
  createCurrentSceneImageStateTags,
  IMAGE_STATE_TYPE_INSTRUCTION
} from "./imageStateTags";
import { createStructuredContextSummary } from "./memoryCompiler";
import { createSceneCastPromptBlock, inferCurrentSceneCharacterIds } from "./sceneCast";

interface OpenAiCompatibleChoice {
  message?: {
    content?: string | Array<{ type?: string; text?: string }>;
    refusal?: string;
  };
  text?: string;
  finish_reason?: string;
}

interface OpenAiCompatibleResponse {
  choices?: OpenAiCompatibleChoice[];
}

interface GeminiSafetyRating {
  category?: string;
  probability?: string;
  blocked?: boolean;
}

interface GeminiResponse {
  candidates?: Array<{
    finishReason?: string;
    finishMessage?: string;
    content?: {
      parts?: Array<{
        text?: string;
      }>;
    };
    safetyRatings?: GeminiSafetyRating[];
  }>;
  promptFeedback?: {
    blockReason?: string;
    blockReasonMessage?: string;
    safetyRatings?: GeminiSafetyRating[];
  };
}

interface ClaudeResponse {
  content?: Array<{
    type?: string;
    text?: string;
  }>;
}

interface ProviderTextOptions {
  model?: string;
  temperature?: number;
  geminiSafetyThreshold?: GeminiSafetyThreshold;
  onRawText?: (rawText: string) => void;
}

const MIN_OUTPUT_TOKENS = 512;
const INTERACTIVE_OUTPUT_TOKEN_CAP = 6000;
const MIN_LLM_REQUEST_TIMEOUT_MS = 45_000;
const MAX_LLM_REQUEST_TIMEOUT_MS = 180_000;
const MAIN_RULE_MAX_CHARS = 5200;
const SELECTED_MODULE_MAX_CHARS = 1800;
const RECENT_TRANSCRIPT_MESSAGE_CHARS = 800;
const CONTINUITY_ANCHOR_CHARS = 1800;
const CONTINUITY_PREVIOUS_USER_CHARS = 700;
const GEMINI_SAFETY_CATEGORIES = [
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY"
] as const;
type GeminiSafetyThreshold = "OFF" | "BLOCK_NONE";

const PROVIDER_BOILERPLATE_PATTERNS = [
  /모델에서\s*응답을\s*생성하지\s*못/u,
  /안전\s*정책으로\s*인해/u,
  /현재\s*내용이\s*안전\s*정책/u,
  /다른\s*방향으로\s*이야기를\s*이어/u,
  /내용을\s*조정(?:하여)?\s*다시\s*시도/u,
  /model\s+(?:could not|couldn't)\s+generate/iu,
  /safety\s+policy/iu,
  /blocked\s+by\s+(?:the\s+)?(?:safety|policy)/iu
] as const;

export interface PlannedImageCueDraft extends AssistantImageCueDraft {
}

export async function generateAssistantText(input: {
  state: AppState;
  userText: string;
  modules: PromptModule[];
  evidence: ContextEvidence[];
  fallback: string;
  onAssistantText?: (assistantText: string) => void;
}): Promise<{
  content: string;
  sidecar: AssistantSidecar;
  source: "mock" | "llm" | "fallback";
  sidecarStatus: "parsed" | "fallback" | "failed";
  sidecarErrors: string[];
  requestPreview?: string;
  rawPreview?: string;
  error?: string;
}> {
  const { state } = input;
  const outputTokenBudget = resolveInteractiveOutputTokenBudget(state.llm.maxTokens);
  const contextBlock = createContextBlock(state, input.userText, input.modules, input.evidence);
  const runtimeInstruction = createRuntimeInstruction(state, outputTokenBudget);
  const requestPreview = createRequestPreview(runtimeInstruction, contextBlock, input.userText);

  if (!state.llm.enabled || state.llm.provider === "mock" || !state.llm.apiKey.trim()) {
    return {
      content: input.fallback,
      sidecar: createFallbackSidecar(input.fallback),
      source: "mock",
      sidecarStatus: "fallback",
      sidecarErrors: ["Mock or unconfigured LLM used fallback sidecar."],
      requestPreview
    };
  }

  try {
    const emitAssistantText = createStreamingAssistantTextEmitter(input.onAssistantText);
    let rawContent = await requestProviderTextWithRecovery(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
      onRawText: emitAssistantText
    });
    if (!rawContent?.trim()) {
      throw new Error("LLM response did not include content.");
    }

    let parsed = parseAssistantSidecar(rawContent);
    if (parsed.sidecar && looksLikeProviderBoilerplateText(parsed.sidecar.assistantText)) {
      rawContent = await requestProviderRecoveryText(
        state,
        input,
        new Error("LLM returned provider boilerplate instead of a scene continuation."),
        outputTokenBudget,
        emitAssistantText
      );
      if (!rawContent?.trim()) {
        throw new Error("LLM response did not include content after scene-continuation retry.");
      }
      parsed = parseAssistantSidecar(rawContent);
      if (parsed.sidecar && looksLikeProviderBoilerplateText(parsed.sidecar.assistantText)) {
        throw new Error("LLM returned provider boilerplate instead of a scene continuation.");
      }
    }

    if (parsed.sidecar) {
      input.onAssistantText?.(parsed.sidecar.assistantText);
      return {
        content: parsed.sidecar.assistantText,
        sidecar: parsed.sidecar,
        source: "llm",
        sidecarStatus: "parsed",
        sidecarErrors: parsed.errors,
        requestPreview,
        rawPreview: rawContent.slice(0, 700)
      };
    }

    const fallbackText = createDisplayFallbackText(rawContent, "");
    if (fallbackText) {
      if (looksLikeProviderBoilerplateText(fallbackText)) {
        throw new Error("LLM returned provider boilerplate instead of a scene continuation.");
      }
      input.onAssistantText?.(fallbackText);
      return {
        content: fallbackText,
        sidecar: createFallbackSidecar(fallbackText, "Structured sidecar was unavailable; image cue will use main-turn fallback data."),
        source: "llm",
        sidecarStatus: "failed",
        sidecarErrors: parsed.errors.length > 0 ? parsed.errors : ["LLM sidecar parse failed."],
        requestPreview,
        rawPreview: rawContent.slice(0, 700)
      };
    }

    const fallbackContent = createLlmFailureFallbackContent(
      input.fallback,
      parsed.errors.length > 0 ? parsed.errors.join("; ") : "LLM sidecar parse failed."
    );
    input.onAssistantText?.(fallbackContent);
    return {
      content: fallbackContent,
      sidecar: createFallbackSidecar(fallbackContent),
      source: "fallback",
      sidecarStatus: "failed",
      sidecarErrors: parsed.errors.length > 0 ? parsed.errors : ["LLM sidecar parse failed."],
      requestPreview,
      rawPreview: rawContent.slice(0, 700)
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown LLM error";
    const fallbackContent = createLlmFailureFallbackContent(input.fallback, errorMessage);
    input.onAssistantText?.(fallbackContent);
    return {
      content: fallbackContent,
      sidecar: createFallbackSidecar(fallbackContent),
      source: "fallback",
      sidecarStatus: "fallback",
      sidecarErrors: [errorMessage],
      requestPreview,
      error: errorMessage
    };
  }
}

export async function generateImageCuePlans(input: {
  state: AppState;
  userText: string;
  assistantText: string;
  drafts: AssistantImageCueDraft[];
  reason: string;
}): Promise<{
  imageCues: AssistantImageCueDraft[];
  rawPreview?: string;
  errors: string[];
}> {
  const { state } = input;
  if (!state.llm.enabled || state.llm.provider === "mock" || !state.llm.apiKey.trim()) {
    return { imageCues: [], errors: ["Mock or unconfigured LLM cannot plan image cues."] };
  }

  const outputTokenBudget = Math.min(1600, resolveInteractiveOutputTokenBudget(state.llm.maxTokens));
  const runtimeInstruction = createImageCuePlannerRuntimeInstruction(state);
  const plannerModel = resolveImageCuePlannerModel(state);
  const requiresPlannerOutput = hasRequiredImagePlannerOutput(input.drafts) || hasExplicitImagePlannerRequest(input.userText);
  const maxAttempts = requiresPlannerOutput ? 2 : 1;
  let lastResult: { imageCues: AssistantImageCueDraft[]; rawPreview?: string; errors: string[] } = {
    imageCues: [],
    errors: []
  };

  try {
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      const retryInstruction =
        attempt > 0
          ? [
              "Previous image planner attempt produced no usable generated cue even though candidate hints/user rules/cadence require an image.",
              lastResult.errors.length > 0 ? `Previous errors: ${lastResult.errors.join(" | ")}` : undefined,
              "Correct it now by returning at least one should_generate=true cue with concrete English NovelAI tags from the current assistant text."
            ]
              .filter((item): item is string => Boolean(item))
              .join("\n")
          : undefined;
      const contextBlock = [
        createImageCuePlannerContextBlock(input),
        retryInstruction
      ]
        .filter((item): item is string => Boolean(item))
        .join("\n\n");
      const rawContent = await requestProviderTextWithRecovery(
        state,
        {
          userText:
            attempt > 0 && requiresPlannerOutput
              ? "Return corrected final image_cues JSON now. At least one cue must be should_generate=true with valid NovelAI tags unless the assistant text is visually impossible to illustrate."
              : "Return final image_cues JSON now.",
          modules: [],
          evidence: []
        },
        runtimeInstruction,
        contextBlock,
        outputTokenBudget,
        {
          model: plannerModel,
          temperature: Math.min(0.35, state.llm.temperature)
        }
      );
      if (!rawContent?.trim()) {
        lastResult = { imageCues: [], errors: ["Image cue planner response did not include content."] };
      } else {
        const jsonText = extractJsonObject(rawContent);
        if (!jsonText) {
          lastResult = {
            imageCues: [],
            rawPreview: rawContent.slice(0, 700),
            errors: ["No JSON object found in image cue planner response."]
          };
        } else {
          const parsed = JSON.parse(jsonText) as Record<string, unknown>;
          const errors: string[] = [];
          const imageCues = normalizeAssistantImageCueDrafts(parsed.image_cues ?? parsed.imageCues ?? parsed.image_cue, errors)
            .filter((cue) => hasCompleteImagePlannerCueDraft(cue, errors))
            .slice(0, 8);
          lastResult = {
            imageCues,
            rawPreview: rawContent.slice(0, 700),
            errors
          };
        }
      }

      if (!requiresPlannerOutput || lastResult.imageCues.some((cue) => cue.shouldGenerate) || attempt === maxAttempts - 1) {
        return lastResult;
      }
    }

    return lastResult;
  } catch (error) {
    return {
      imageCues: [],
      errors: [error instanceof Error ? error.message : "Image cue planning failed."]
    };
  }
}

function hasRequiredImagePlannerOutput(drafts: AssistantImageCueDraft[]): boolean {
  return drafts.some((draft) => {
    const extendedDraft = draft as AssistantImageCueDraft & {
      forceFreshImage?: boolean;
      forceImagePlanning?: boolean;
      plannerSource?: string;
    };
    return (
      draft.shouldGenerate ||
      extendedDraft.forceFreshImage === true ||
      extendedDraft.forceImagePlanning === true ||
      extendedDraft.plannerSource === "user_image_rules" ||
      extendedDraft.plannerSource === "image_generation_cadence"
    );
  });
}

function hasExplicitImagePlannerRequest(text: string): boolean {
  return /\b(?:image|picture|illustration|illustrate|draw|visualize|show me)\b|이미지|그림|일러스트|그려|보여줘|시각화/iu.test(text);
}

function hasCompleteImagePlannerCueDraft(cue: AssistantImageCueDraft, errors: string[]): boolean {
  if (!cue.shouldGenerate) {
    return Boolean(cue.suppressionReason);
  }

  const tags = cue.tags
    .map((tag) => tag.trim())
    .filter(Boolean)
    .filter((tag) => !isGenericOrAbstractPlannerTag(tag));
  const kind = (cue.kind ?? cue.cueType ?? "").toLowerCase();
  const minimumTagCount = kind === "dialogue_face" || kind === "body_detail" ? 5 : 6;
  const hasAnchor = Boolean(cue.anchorText?.trim());
  const hasScene = Boolean(normalizeImageSceneLabelForPlanner(cue.scene));
  const hasComposition = tags.some((tag) => /\b(?:close-up|close up|upper body|cowboy shot|full body|wide shot|medium shot|portrait|pov|from side|from behind|low angle|high angle|face focus|eye focus|body focus)\b/iu.test(tag));
  const hasActionOrExpression = tags.some((tag) => /\b(?:standing|sitting|walking|running|leaning|reaching|holding|grabbing|touching|hand|hands|wrist grab|microphone|open mouth|smile|surprised|angry|sad|blush|sweat|tears)\b/iu.test(tag));

  if (tags.length < minimumTagCount || !hasComposition || !hasActionOrExpression) {
    errors.push(`Dropped incomplete image planner cue '${cue.label || cue.kind || "unnamed"}': generated cues need structured NAI tags instead of sparse/local-looking tags.`);
    return false;
  }
  if (!hasAnchor) {
    errors.push(`Accepted image planner cue '${cue.label || cue.kind || "unnamed"}' without anchor_text; DynamicChat will place it using the nearest candidate cue hint.`);
  }
  if (!hasScene) {
    errors.push(`Accepted image planner cue '${cue.label || cue.kind || "unnamed"}' without a concrete English scene label; DynamicChat will use the candidate scene fallback.`);
  }

  return true;
}

function isGenericOrAbstractPlannerTag(tag: string): boolean {
  const normalized = tag.toLowerCase().replace(/[._-]+/gu, " ").trim();
  return /^(?:scene|current scene|generated scene|image|prompt|tag|tags|main action|facial expression|body detail|dialogue face|clear environment|context appropriate outfit|visible emotional reaction|props?|filming set|walking scene|microphone scene)$/iu.test(
    normalized
  );
}

function normalizeImageSceneLabelForPlanner(value: string | undefined): string | undefined {
  const normalized = value
    ?.trim()
    .replace(/[._-]+/gu, " ")
    .replace(/[^\p{L}\p{N} ]+/gu, " ")
    .replace(/\s+/gu, " ")
    .toLowerCase();
  if (!normalized || /[\u3131-\uD79D]/u.test(normalized)) {
    return undefined;
  }
  return /^(?:(?:current|generated|simulation|safe)\s+)*scene$/iu.test(normalized) ? undefined : normalized;
}

function createImageCuePlannerRuntimeInstruction(state: AppState): string {
  return [
    state.llm.systemPrompt,
    "You are DynamicChat's dedicated image cue and NovelAI tag planner.",
    "Return JSON only. Do not write assistant narrative.",
    "The simulation LLM has already written assistant_text. Your job is to read the whole context and produce the final image_cues: generation points, visible character ids, scene/action/body/dialogue targets, and concrete NovelAI tags.",
    "Candidate cue hints may come from legacy sidecar output, user image rules, manual requests, or cadence. Treat them as hints, not final truth: you may rewrite, split, drop, or add cues to match the actual visible beats.",
    "DynamicChat will not generate an image from local tag inference. If an image should exist, you must author the usable final tags yourself in image_cues.tags.",
    "If candidate cue hints contain should_generate=true, force_image_planning=true, planner_source=user_image_rules, or planner_source=image_generation_cadence, return at least one should_generate=true cue unless assistant_text is empty or genuinely impossible to illustrate. Do not return [] merely because the scene requires reasoning.",
    "If you decide not to generate, return a should_generate=false cue with a concrete suppression_reason. DynamicChat will respect that decision and will not fall back to local image tags.",
    "All image_cues with should_generate=true must include concrete English NovelAI tags in tags. visual_context may repeat or extend the same comma-separated tags, but it is not a substitute for an empty tags array. Sparse cues such as only `eye focus`, or repeated local-looking cues such as `wide shot, walking, microphone, open mouth, eye focus, smile, surprised`, are invalid unless they are expanded with the actual current character, location, pose/contact, outfit/state, and expression tags from this specific beat.",
    "For the first image of a run, there may be no stored image state yet. In that case, infer the best visual tags from the simulation premise, current user action, assistant_text, roster, and user image rules instead of returning an empty cue.",
    "Do not default to a character portrait. Choose the cue kind from the visible target: scene/context for full environment, action for the current beat, body_detail for an emphasized body/prop detail, dialogue_face for expression before speech, or interaction for multi-character interaction.",
    "User image rules are binding. If they request body detail, scene-wide cuts, every action beat, no reuse, fresh images, or dialogue-face cuts, reflect that in kind, label, placement, anchor_text, tags, and visual_context.",
    "Before writing each cue, privately choose: visible roster character ids, character count tag, camera/framing, background, base pose/action, physical/body/prop interaction details, clothing state, appearance/outfit, expression/effects. Then output only the final tags. If a visible roster character is selected, include that exact character id in characters and include a subject-count tag such as 1girl/1boy/2girls when the visible count is unambiguous.",
    "For action, interaction, or body_detail cues, background tags alone are a failure. Include pose/action and physical detail tags that define what the viewer sees: standing, sitting, leaning, arm up, hand on shoulder, wrist grab, holding prop, face focus, hands, chest focus, thigh focus, close-up, from side, pov, etc.",
    "If user image rules say to emphasize a body part when it appears in context, obey them as composition rules: use kind=body_detail, include the body part focus tag, a close framing tag, the relevant action/contact tag, and the visible character id when that body part belongs to a roster character.",
    "If the User persona block says source: existing_character, first-person narration, '나', and user stage directions/actions are actions by controlled_character_id. Include that id in characters when that character is visibly acting or speaking.",
    "For characters, include only exact visible roster character ids. Never leave characters empty when a roster character is the visible actor, speaker, target, or owner of an emphasized body detail. Leave characters empty only for pure environment/object cuts or non-roster people. If the assistant text uses pronouns such as she/he/그녀/그 and the Scene cast guard has exactly one active roster character, use that character id.",
    "Registered character visual profiles and outfit keyword mappings are authoritative. If a cue or current state matches an outfit keyword, expand the full mapped outfit tags in visual_context; never collapse a mapping such as `school uniform, dark grey pencil skirt, tight fit, necktie` down to only `school uniform`.",
    createNovelAiTagTransformInstruction(state),
    "Use compact current-state tags rather than storing or copying excessive micro-details. Prefer phase/state tags that explain the overall situation, then add only the visible pose/action/details needed for the image.",
    "For adult_19 mode, represent the current adult-only scene phase and visible composition with provider-compatible NovelAI tags when the current context requires it; otherwise use non-explicit composition and emotion tags.",
    "Return this JSON shape:",
    JSON.stringify({
      image_cues: [
        {
          label: "scene / action / body detail / dialogue face / interaction",
          kind: "scene | action | body_detail | dialogue_face | interaction | context",
          placement: "before | after | inline",
          anchor_text: "exact nearby assistant_text fragment",
          priority: 0.92,
          should_generate: true,
          reason: "why this image is useful",
          suppression_reason: "",
          characters: ["visible character ids only"],
          tags: ["1girl", "upper body", "looking at viewer", "classroom", "standing"],
          scene: "short English scene label",
          visual_context: "comma-separated English NAI tags"
        }
      ]
    })
  ].join("\n");
}

function createImageCuePlannerContextBlock(input: {
  state: AppState;
  userText: string;
  assistantText: string;
  drafts: AssistantImageCueDraft[];
  reason: string;
}): string {
  const { state } = input;
  return [
    "Dedicated image cue planner context. Return JSON only.",
    `Planner reason: ${input.reason}`,
    `Simulation: ${state.simulation.title}`,
    `Content rating: ${state.simulation.contentRating}`,
    state.simulation.description ? `Premise: ${state.simulation.description}` : undefined,
    "Scene cast guard:",
    createSceneCastPromptBlock(state, `${input.userText}\n${input.assistantText}`),
    "Existing character roster:",
    state.characters.map((character) => `- id: ${character.id} | name: ${character.name} | role: ${character.role} | state: ${character.currentMood}`).join("\n") || "(none)",
    "Registered visual profiles and outfit mappings:",
    createImageCueVisualProfileBlock(state) || "(none)",
    "Current image-state tags resolved for this planner:",
    createImageCueCurrentStateBlock(state) || "(none)",
    "User persona:",
    createUserPersonaBlock(state) || "(none)",
    "Image prompt user rules:",
    createImageUserRulesBlock(state) || "(none)",
    "Image generation cadence:",
    createImageGenerationCadenceBlock(state),
    "Structured simulation memory:",
    createStructuredContextSummary(state, { maxEvents: 6, maxStates: 14, currentText: `${input.userText}\n${input.assistantText}` }),
    "Current user action:",
    input.userText,
    "Assistant text to illustrate:",
    input.assistantText,
    "Candidate cue hints to rewrite or replace:",
    JSON.stringify({
      image_cues: input.drafts.map((draft) => {
        const extendedDraft = draft as AssistantImageCueDraft & {
          forceFreshImage?: boolean;
          forceImagePlanning?: boolean;
          plannerSource?: string;
        };
        return {
          label: draft.label,
          kind: draft.kind ?? draft.cueType,
          placement: draft.placement,
          anchor_text: draft.anchorText,
          should_generate: draft.shouldGenerate,
          force_fresh_image: extendedDraft.forceFreshImage,
          force_image_planning: extendedDraft.forceImagePlanning,
          planner_source: extendedDraft.plannerSource,
          reason: draft.reason,
          characters: draft.characters,
          tags: draft.tags,
          scene: draft.scene,
          visual_context: draft.visualContext
        };
      })
    })
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function createImageCueVisualProfileBlock(state: AppState): string {
  return state.characters
    .map((character) => {
      const visualProfile = state.visualProfiles.find((profile) => profile.characterId === character.id);
      if (!visualProfile) {
        return undefined;
      }
      const outfitMappings = formatOutfitMappingsForFoundation(visualProfile.outfitPrompts);
      return [
        `- character_id: ${character.id}`,
        `name: ${character.name}`,
        visualProfile.positivePrompt ? `visual_profile_tags: ${truncatePromptText(visualProfile.positivePrompt, 420, "visual profile")}` : undefined,
        visualProfile.defaultOutfitPrompt ? `default_outfit_tags: ${visualProfile.defaultOutfitPrompt}` : undefined,
        outfitMappings ? `outfit_keyword_mappings: ${outfitMappings}` : undefined
      ]
        .filter((item): item is string => Boolean(item))
        .join(" | ");
    })
    .filter((item): item is string => Boolean(item))
    .join("\n");
}

function createImageCueCurrentStateBlock(state: AppState): string {
  const characterLines = state.characters
    .map((character) => {
      const stateTags = createCurrentCharacterImageStatePrompt(state, character.id);
      return stateTags ? `- character_id: ${character.id} | current_state_tags: ${stateTags}` : undefined;
    })
    .filter((item): item is string => Boolean(item));
  const sceneTags = createCurrentSceneImageStateTags(state);
  return [
    sceneTags.length > 0 ? `- scene_state_tags: ${sceneTags.join(", ")}` : undefined,
    ...characterLines
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n");
}

function resolveImageCuePlannerModel(state: AppState): string | undefined {
  const model = state.llm.model.trim();
  if (!model || state.llm.provider === "mock" || state.llm.provider === "openai_compatible") {
    return model || undefined;
  }

  if (state.llm.provider === "gemini") {
    if (/flash/iu.test(model)) {
      return model;
    }
    if (/gemini-2\.5/iu.test(model)) {
      return "gemini-2.5-flash";
    }
    if (/gemini-1\.5/iu.test(model)) {
      return "gemini-1.5-flash";
    }
    return model;
  }

  if (state.llm.provider === "claude") {
    return /haiku/iu.test(model) ? model : "claude-3-5-haiku-latest";
  }

  if (state.llm.provider === "codex") {
    if (/gpt-4\.1(?:$|[^-])|gpt-4o|gpt-5|o3/iu.test(model)) {
      return "gpt-4.1-mini";
    }
    return model;
  }

  return model;
}

function createLlmFailureFallbackContent(fallback: string, reason: string): string {
  const summary = summarizeLlmFailureReason(reason);
  return fallback
    .replace(
      /::status\[LLM fallback: [^\]]*?적용 대상 모듈:\s*([^\]]*)\]/u,
      `::status[LLM 응답 fallback: ${summary}. 적용 대상 모듈: $1]`
    )
    .replace(
      "API 키, 모델, 또는 JSON sidecar 형식을 확인한 뒤 다시 입력하면 이 턴은 실제 LLM 규칙 적용 흐름으로 이어질 수 있다.",
      `${summary}. 설정과 모델 응답 상태를 확인한 뒤 다시 입력하면 실제 LLM 흐름으로 이어질 수 있다.`
    );
}

function summarizeLlmFailureReason(reason: string): string {
  const compact = reason.replace(/\s+/gu, " ").trim();
  if (/PROHIBITED_CONTENT/iu.test(compact)) {
    return "Gemini가 이 요청에 사용할 수 있는 본문을 반환하지 않았습니다";
  }
  if (/did not include (?:text )?content|did not include content|empty|content_filter|refusal/iu.test(compact)) {
    return "LLM provider가 빈 응답을 반환했습니다";
  }
  if (/sidecar parse|assistant_text is missing|image_cues is missing|invalid json|no json object/iu.test(compact)) {
    return "LLM 응답 JSON sidecar를 해석하지 못했습니다";
  }
  if (/request failed:\s*(\d+)/iu.test(compact)) {
    return `LLM API 요청이 HTTP ${compact.match(/request failed:\s*(\d+)/iu)?.[1]}로 실패했습니다`;
  }
  return compact ? `LLM provider 오류: ${compact.slice(0, 120)}` : "LLM provider가 사용할 수 있는 본문을 반환하지 않았습니다";
}

function looksLikeProviderBoilerplateText(value: string): boolean {
  const compact = value.replace(/\s+/gu, " ").trim();
  if (!compact) {
    return false;
  }

  const hasBoilerplateSignal = PROVIDER_BOILERPLATE_PATTERNS.some((pattern) => pattern.test(compact));
  if (!hasBoilerplateSignal) {
    return false;
  }

  return compact.length < 900 || /죄송|다시\s*시도|조정|생성|정책|policy|model|provider/iu.test(compact);
}

function sanitizeProviderRetryText(value: string, maxChars: number): string {
  const withoutBoilerplate = PROVIDER_BOILERPLATE_PATTERNS.reduce(
    (current, pattern) => current.replace(pattern, "[이전 오류 안내 생략]"),
    value
  );
  const redacted = withoutBoilerplate.replace(
    /\b(?:sex|sexual|explicit|nude|nudity|porn|rape|incest|underage|minor|self-harm|suicide)\b/giu,
    "[sensitive]"
  );
  const normalized = redacted.replace(/\s+/gu, " ").trim();
  return normalized.length > maxChars ? `${normalized.slice(0, maxChars)}...` : normalized;
}

function createContextBlock(state: AppState, currentUserText: string, modules: PromptModule[], evidence: ContextEvidence[]): string {
  const foundationText = createSimulationFoundationBlock(state, currentUserText);
  const immediateContinuityText = createImmediateContinuityBlock(state);
  const structuredMemoryText = createStructuredContextSummary(state, { maxEvents: 8, maxStates: 10, currentText: currentUserText });
  const recentTranscriptText = createRecentTranscriptBlock(state);
  const moduleText = modules
    .filter((module) => !(module.kind === "safety_policy" && isAdultContentMode(state)))
    .map((module) => formatActivePromptModule(module))
    .join("\n\n");
  const evidenceText = evidence
    .map((item) => `- (${Math.round(item.score * 100)}%) ${item.snippet}`)
    .join("\n");
  const personaText = createUserPersonaBlock(state);
  const imageUserRulesText = createImageUserRulesBlock(state);
  const imageCadenceText = createImageGenerationCadenceBlock(state);
  const relationshipMapRulesText = createRelationshipMapRulesBlock(state);
  const sceneCastText = createSceneCastPromptBlock(state, currentUserText);

  return [
    "Use the following DynamicChat context. Do not reveal internal IDs unless asked.",
    "Immediate continuity anchor:",
    immediateContinuityText || "(no previous assistant turn)",
    "Current scene cast guard:",
    sceneCastText,
    "Return JSON only. The JSON schema is:",
    JSON.stringify({
      assistant_text: "natural Korean response shown to the user",
      memory_events: [
        {
          memory_kind: "event | state | observation | belief | goal | relationship | open_thread | summary",
          event_type: "short stable event type, when memory_kind is event",
          state_type: `${IMAGE_STATE_TYPE_INSTRUCTION}, when memory_kind is state`,
          state_value: "current state value, when memory_kind is state",
          importance: 0.86,
          confidence: 0.92,
          tags: ["promise", "relationship", "scene-state"],
          content: "one concise durable delta, not the full assistant response",
          actor_id: "optional character id",
          actor_name: "optional character name",
          target_id: "optional target character/item/location id",
          observers: ["character ids who observed or heard this"]
        }
      ],
      image_cues: [
        {
          label: "optional high-level image hint; usually omit by returning []",
          kind: "scene | action | body_detail | dialogue_face | context | interaction",
          placement: "before | after | inline",
          anchor_text: "exact nearby assistant_text fragment used for placement",
          priority: 0.92,
          should_generate: true,
          reason: "short hint only; dedicated image planner makes final decision",
          suppression_reason: "why generation should be skipped, if any",
          characters: ["optional visible character ids from context when known"],
          tags: [],
          scene: "short visual scene label",
          visual_context: "optional short hint; final NAI tags are generated by the dedicated image planner"
        }
      ]
    }),
    "Match assistant_text to the runtime output length target and keep it complete: close the JSON object every time. Return at most 8 memory_events. For image_cues, default to []: a dedicated lightweight image planner reads the completed assistant_text and creates the final generation points and NovelAI tags after this response.",
    "Memory compiler rules: memory_events are structured simulation deltas only. Do not store the full assistant_text, style prose, atmosphere, repeated facts, or facts already present in Structured simulation memory.",
    "Separate actual events from current states. If a current state changes, output memory_kind='state' with state_type and state_value. If someone saw/heard/learned something, output observation or belief for that character only. Keep uncertain causes as belief/open_thread, not confirmed fact.",
    "Outfit and image-state continuity rules: character visual profiles may define default outfit tags and keyword outfit mappings. Do not spend simulation tokens creating final image prompts. Instead, store the changed current outfit as memory_kind='state', state_type='Wearing' with actor_id for that exact character. Store compact English NAI-style current visual state tags as memory_kind='state' when they matter for later consistency: state_type='StatusTags' for expression/condition, 'PoseTags' for pose, 'ActionTags' for current action, 'InteractionTags' or 'InteractionPhaseTags' for current interaction/overall phase, 'HeldItemTags' for held props, and scene-level 'SceneTags'/'ScenePhaseTags'/'CompositionTags'/'CameraTags'/'LightingTags' without actor_id when they describe the whole cut. Prefer compact phase/state tags over excessive micro-detail. The dedicated image planner will combine these states with character prompts, outfit mappings, user image rules, and assistant_text.",
    relationshipMapRulesText
      ? "Relationship/status map rules: when the rules below are active, update character states and relationships through memory_events instead of writing a long visible status block in assistant_text."
      : undefined,
    relationshipMapRulesText || undefined,
    "Simulation foundation:",
    foundationText,
    "User persona:",
    personaText || "(none)",
    "Image prompt user rules:",
    imageUserRulesText || "(none)",
    "Image generation cadence:",
    imageCadenceText,
    "Structured simulation memory:",
    structuredMemoryText,
    "Recent transcript:",
    recentTranscriptText || "(none)",
    "Selected prompt modules for this turn:",
    moduleText || "(none)",
    "Memory/context evidence:",
    evidenceText || "(none)"
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function createRelationshipMapRulesBlock(state: AppState): string {
  if (!state.relationshipMap?.enabled) {
    return "";
  }

  const parameters = (state.relationshipMap.parameters ?? [])
    .filter((parameter) => parameter.enabled && (parameter.title.trim() || parameter.rule.trim()))
    .sort((a, b) => b.priority - a.priority)
    .map((parameter) => `- ${parameter.title.trim()}: ${parameter.rule.trim() || "현재 진행에 맞게 짧고 안정적인 값으로 갱신한다."}`)
    .join("\n");

  return [
    state.relationshipMap.statusPrompt.trim(),
    parameters
      ? [
          "Configured character status parameters:",
          parameters,
          "For each configured parameter that changes or becomes newly relevant, emit a memory_events item with memory_kind='state', state_type exactly equal to the parameter title, state_value following that parameter rule, and content as a concise durable Korean state line. Set actor_id/actor_name to the affected character. Do not invent a value for a character when the scene gives no basis."
        ].join("\n")
      : undefined,
    "Outfit/status persistence: if clothing changes or the scene establishes a new outfit, also emit memory_kind='state', state_type='Wearing', state_value as English NovelAI outfit tags. If a character gains important visual state tags, emit state_type='StatusTags'. These records feed the relationship tab and later image prompts."
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function formatActivePromptModule(module: PromptModule): string {
  const body = truncatePromptText(module.body, SELECTED_MODULE_MAX_CHARS, "selected module excerpt");
  return [
    `# ${module.title}`,
    `kind: ${module.kind} | token_policy: ${module.tokenPolicy} | priority: ${module.priority}`,
    module.activationTags.length > 0 ? `activation_tags: ${module.activationTags.join(", ")}` : undefined,
    module.characterId ? `character_id: ${module.characterId}` : undefined,
    "body:",
    body
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n");
}

function resolveInteractiveOutputTokenBudget(maxTokens: number): number {
  const requested = Number.isFinite(maxTokens) ? maxTokens : 1600;
  return Math.min(INTERACTIVE_OUTPUT_TOKEN_CAP, Math.max(MIN_OUTPUT_TOKENS, requested));
}

function resolveLlmRequestTimeoutMs(outputTokenBudget: number): number {
  const tokenScaledTimeout = 30_000 + outputTokenBudget * 20;
  return Math.min(MAX_LLM_REQUEST_TIMEOUT_MS, Math.max(MIN_LLM_REQUEST_TIMEOUT_MS, tokenScaledTimeout));
}

function createContentRatingInstruction(state: AppState): string {
  if (isAdultContentMode(state)) {
    return "Content rating: 19+ adult-only mode. Treat adult-only material as involving adults, follow creator rules and user boundaries, and keep the response compatible with the selected LLM provider.";
  }

  return "Content rating: general. Use the configured LLM/provider behavior and creator rules.";
}

function createImageGenerationCadenceBlock(state: AppState): string {
  const cadence = state.imageProfile.generationCadence ?? "balanced";
  if (cadence === "sparse") {
    return "sparse: emit image_cues only for explicit image requests, major scene/location changes, or a visually decisive beat. Usually 0-1 cue.";
  }
  if (cadence === "rich") {
    return "rich: emit image_cues for visible action, emotion, outfit, expression, or location changes. Prefer 1-3 cues when the response has multiple visual beats.";
  }
  if (cadence === "paragraph") {
    return "paragraph: emit an anchored image_cue for each visually distinct assistant_text paragraph/block, up to 8 cues. Use placement and anchor_text so DynamicChat can attach each cut to its paragraph.";
  }

  return "balanced: emit one image_cue for the most important visual beat; use [] for quiet text-only turns. Usually 0-2 cues.";
}

function createImageCueTagContractInstruction(state: AppState): string {
  const adultExplicitInstruction = isAdultContentMode(state)
    ? "In adult_19 mode, emit direct visual tags only for material the selected LLM provider can return. If the current beat is provider-sensitive, use image_cues: [] and keep assistant_text as a non-explicit continuation."
    : "Do not invent sexual tags unless the current scene and content mode explicitly allow and require them.";
  return [
    "Image cue NAI tag contract: image_cues.tags must be the actual final NovelAI tags for the current visible beat, not summaries, labels, or prose.",
    "Never emit should_generate=true with empty image_cues.tags. If the current beat should be shown, provide concrete tags for actor count, visible action, pose, expression, location, camera/POV, and props. If you cannot provide those tags, set should_generate=false and explain the suppression_reason.",
    "Write 10-32 concise English NAI tags when an image is generated. Include subject count only when visible and unambiguous, e.g. 1girl, 1boy, 2girls; omit subject count when the visible count is uncertain.",
    "Tags should cover the current actor, visible action, pose, expression, gaze/POV/camera framing, location, important props/held items, and current outfit when the outfit is not already covered by the selected character profile. For a visible roster character, the characters array is what activates that character's prompt; do not rely on scene tags alone.",
    "Do not put artist, style, quality, resolution, or negative/undesired tags in image_cues.tags; those are configured separately by DynamicChat. Exclude tags like highres, absurdres, masterpiece, best quality, lowres, watermark.",
    adultExplicitInstruction,
    createNovelAiTagTransformInstruction(state),
    "Good action tags look like: 1girl, upper body, looking at viewer, classroom, indoors, standing, arm up, hand up, chair, holding notebook, school uniform, pleated skirt, open mouth, blush, sweat.",
    "Good body-detail tags look like: 1girl, close-up, body focus, hands, wrist grab, arm focus, from side, school uniform, blush, sweat.",
    "Bad tags are abstract, prose, production labels, or situation summaries: filming set, props, acting exercise, intimidation, abduction, fear, getting up, standing up from her seat, raising her hand eagerly, main action, facial expression, close up face, visible emotional reaction, situation specific clothing, character-close-up, wide context, clear environment, context appropriate outfit. A cue that only says `acting studio` or another background label is incomplete unless it also includes pose/action/framing/body details.",
    "image_cues.visual_context must also be comma-separated NAI tags or short tag phrases. Do not write a descriptive sentence there."
  ].join(" ");
}

function createNovelAiTagTransformInstruction(state: AppState): string {
  const adultInteractionInstruction = isAdultContentMode(state)
    ? "For adult-only explicit interaction beats, avoid emotional metaphors and use direct physical/provider-compatible Danbooru-style tags for the core position/act, body-contact details, clothing state, and visible physiological details only when the current context requires them."
    : "Do not add explicit body-contact or sexual-act tags outside adult_19 context.";
  return [
    "NovelAI/Danbooru tag conversion rules:",
    "Use comma-separated English tags only. Do not output names, natural-language sentences, verb clauses, production labels, or abstract descriptions.",
    "Decompose prose into visual tags: `standing up from her seat` becomes `standing, chair`; `raising her hand eagerly` becomes `arm up, hand up, smile`; `bright and confident smile` becomes `smile`.",
    "Convert abstract situations into visible anatomy, pose, contact, object, and expression tags. For example, `intimidation` becomes visible tags such as `looming`, `from below`, `open mouth`, `sweat` only if those are actually visible; `abduction` becomes tags such as `wrist grab`, `struggling`, `from behind` only if the scene visibly shows them; otherwise omit the abstract situation.",
    "Remove duplicate, synonymous, and contradictory tags. If the visible location is classroom/indoors, do not also output street/outdoors unless the scene truly shows both.",
    "Use this strict tag order for every generated cue: 1 character count; 2 perspective/framing; 3 background/environment; 4 base pose/action; 5 physical/prop/body interaction; 6 clothing state/outfit; 7 appearance; 8 expression/effects.",
    "Outfit keyword mappings are full prompts, not labels. When a character mapping says `교복: school uniform, dark grey pencil skirt, tight fit, necktie`, include every mapped tag that remains visible instead of only `school uniform`.",
    adultInteractionInstruction
  ].join(" ");
}

async function requestProviderText(
  state: AppState,
  input: {
    userText: string;
    modules: PromptModule[];
    evidence: ContextEvidence[];
  },
  runtimeInstruction: string,
  contextBlock: string,
  outputTokenBudget: number,
  options: ProviderTextOptions = {}
): Promise<string | undefined> {
  const baseUrl = state.llm.baseUrl.replace(/\/$/u, "");
  const model = options.model?.trim() || state.llm.model;
  const temperature = options.temperature ?? state.llm.temperature;
  const timeoutMs = resolveLlmRequestTimeoutMs(outputTokenBudget);

  if (state.llm.provider === "gemini") {
    const response = await fetchWithTimeout(`${baseUrl}/models/${encodeURIComponent(model)}${options.onRawText ? ":streamGenerateContent" : ":generateContent"}?key=${encodeURIComponent(state.llm.apiKey)}${options.onRawText ? "&alt=sse" : ""}`, {
      method: "POST",
      headers: {
        "content-type": "application/json"
      },
      body: JSON.stringify({
        systemInstruction: {
          parts: [{ text: runtimeInstruction }]
        },
        contents: [
          {
            role: "user",
            parts: [{ text: `${contextBlock}\n\nUser action:\n${input.userText}` }]
          }
        ],
        safetySettings: createGeminiSafetySettings(options.geminiSafetyThreshold ?? "OFF"),
        generationConfig: {
          temperature,
          maxOutputTokens: outputTokenBudget,
          responseMimeType: "application/json"
        }
      })
    }, timeoutMs);

    if (!response.ok) {
      if (options.onRawText) {
        return requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          onRawText: undefined
        });
      }
      throw new Error(`Gemini request failed: ${response.status}`);
    }

    if (options.onRawText) {
      return readGeminiStreamText(response, options.onRawText).catch(() =>
        requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          onRawText: undefined
        })
      );
    }

    const data = (await response.json()) as GeminiResponse;
    const candidate = data.candidates?.[0];
    const content = candidate?.content?.parts?.map((part) => part.text ?? "").join("").trim();
    if (!content) {
      throw new Error(createGeminiEmptyContentMessage(data));
    }

    return content;
  }

  if (state.llm.provider === "claude") {
    const response = await fetchWithTimeout(`${baseUrl}/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": state.llm.apiKey,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify({
        model,
        temperature,
        max_tokens: outputTokenBudget,
        stream: Boolean(options.onRawText),
        system: `${runtimeInstruction}\n\n${contextBlock}`,
        messages: [
          {
            role: "user",
            content: input.userText
          }
        ]
      })
    }, timeoutMs);

    if (!response.ok) {
      if (options.onRawText) {
        return requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          onRawText: undefined
        });
      }
      throw new Error(`Claude request failed: ${response.status}`);
    }

    if (options.onRawText) {
      return readClaudeStreamText(response, options.onRawText).catch(() =>
        requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          onRawText: undefined
        })
      );
    }

    const data = (await response.json()) as ClaudeResponse;
    return data.content?.map((part) => (part.type === "text" ? part.text ?? "" : "")).join("").trim();
  }

  const response = await fetchWithTimeout(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${state.llm.apiKey}`
    },
    body: JSON.stringify({
      model,
      temperature,
      max_tokens: outputTokenBudget,
      stream: Boolean(options.onRawText),
      messages: [
        {
          role: "system",
          content: runtimeInstruction
        },
        {
          role: "system",
          content: contextBlock
        },
        {
          role: "user",
          content: input.userText
        }
      ]
    })
  }, timeoutMs);

  if (!response.ok) {
    if (options.onRawText) {
      return requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
        ...options,
        onRawText: undefined
      });
    }
    throw new Error(`LLM request failed: ${response.status}`);
  }

  if (options.onRawText) {
    return readOpenAiCompatibleStreamText(response, options.onRawText).catch(() =>
      requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
        ...options,
        onRawText: undefined
      })
    );
  }

  const data = (await response.json()) as OpenAiCompatibleResponse;
  const choice = data.choices?.[0];
  const content = readOpenAiCompatibleChoiceText(choice);
  if (content) {
    return content;
  }
  if (choice?.finish_reason) {
    throw new Error(`LLM response did not include content (finish: ${choice.finish_reason}).`);
  }
  if (choice?.message?.refusal) {
    throw new Error("LLM response returned a refusal instead of content.");
  }
  return undefined;
}

function readOpenAiCompatibleChoiceText(choice: OpenAiCompatibleChoice | undefined): string | undefined {
  const messageContent = choice?.message?.content;
  if (typeof messageContent === "string") {
    return messageContent;
  }
  if (Array.isArray(messageContent)) {
    const text = messageContent
      .map((part) => (part.type === "text" ? part.text ?? "" : ""))
      .join("")
      .trim();
    if (text) {
      return text;
    }
  }
  return choice?.text;
}

function createStreamingAssistantTextEmitter(onAssistantText: ((assistantText: string) => void) | undefined): ((rawText: string) => void) | undefined {
  if (!onAssistantText) {
    return undefined;
  }

  let lastText = "";
  let lastEmitAt = 0;
  return (rawText) => {
    const assistantText = createDisplayFallbackText(rawText, "");
    if (!assistantText || assistantText.length < lastText.length || assistantText === lastText) {
      return;
    }

    const now = Date.now();
    if (now - lastEmitAt < 220 && assistantText.length - lastText.length < 80) {
      return;
    }

    lastText = assistantText;
    lastEmitAt = now;
    onAssistantText(assistantText);
  };
}

async function readGeminiStreamText(response: Response, onRawText: (rawText: string) => void): Promise<string | undefined> {
  return readSseProviderText(response, onRawText, (payload) => {
    const data = JSON.parse(payload) as GeminiResponse;
    return data.candidates?.[0]?.content?.parts?.map((part) => part.text ?? "").join("") ?? "";
  });
}

async function readClaudeStreamText(response: Response, onRawText: (rawText: string) => void): Promise<string | undefined> {
  return readSseProviderText(response, onRawText, (payload) => {
    const data = JSON.parse(payload) as {
      type?: string;
      delta?: {
        type?: string;
        text?: string;
      };
    };
    return data.type === "content_block_delta" && data.delta?.type === "text_delta"
      ? data.delta.text ?? ""
      : "";
  });
}

async function readOpenAiCompatibleStreamText(response: Response, onRawText: (rawText: string) => void): Promise<string | undefined> {
  return readSseProviderText(response, onRawText, (payload) => {
    const data = JSON.parse(payload) as {
      choices?: Array<{
        delta?: {
          content?: string | Array<{ type?: string; text?: string }>;
        };
        text?: string;
      }>;
    };
    const choice = data.choices?.[0];
    const content = choice?.delta?.content;
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return content.map((part) => (part.type === "text" ? part.text ?? "" : "")).join("");
    }
    return choice?.text ?? "";
  });
}

async function readSseProviderText(
  response: Response,
  onRawText: (rawText: string) => void,
  readPayloadText: (payload: string) => string
): Promise<string | undefined> {
  if (!response.body) {
    throw new Error("LLM stream response did not include a readable body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";

  const consumeEvent = (eventText: string) => {
    for (const payload of readSseDataPayloads(eventText)) {
      if (!payload || payload === "[DONE]") {
        continue;
      }

      const text = readPayloadText(payload);
      if (!text) {
        continue;
      }
      accumulated += text;
      onRawText(accumulated);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/u);
    buffer = events.pop() ?? "";
    events.forEach(consumeEvent);
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    consumeEvent(buffer);
  }

  return accumulated.trim() || undefined;
}

function readSseDataPayloads(eventText: string): string[] {
  const dataLines = eventText
    .split(/\r?\n/u)
    .map((line) => line.trimEnd())
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.replace(/^data:\s?/u, ""));

  if (dataLines.length > 0) {
    return [dataLines.join("\n").trim()];
  }

  const trimmed = eventText.trim();
  return trimmed.startsWith("{") || trimmed === "[DONE]" ? [trimmed] : [];
}

async function requestProviderTextWithRecovery(
  state: AppState,
  input: {
    userText: string;
    modules: PromptModule[];
    evidence: ContextEvidence[];
  },
  runtimeInstruction: string,
  contextBlock: string,
  outputTokenBudget: number,
  options: ProviderTextOptions = {}
): Promise<string | undefined> {
  try {
    return await requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, options);
  } catch (error) {
    if (state.llm.provider === "gemini" && isGeminiOffSafetySettingRejected(error)) {
      try {
        return await requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          geminiSafetyThreshold: "BLOCK_NONE"
        });
      } catch (blockNoneError) {
        if (!isProviderNonContentError(blockNoneError)) {
          throw blockNoneError;
        }
        return requestProviderRecoveryText(state, input, blockNoneError, outputTokenBudget, options.onRawText);
      }
    }

    if (isProviderNonContentError(error)) {
      return requestProviderRecoveryText(state, input, error, outputTokenBudget, options.onRawText);
    }

    throw error;
  }
}

async function requestProviderRecoveryText(
  state: AppState,
  input: {
    userText: string;
    modules: PromptModule[];
    evidence: ContextEvidence[];
  },
  originalError: unknown,
  outputTokenBudget: number,
  onRawText?: (rawText: string) => void
): Promise<string | undefined> {
  const recoveryInstruction = createRecoveryRuntimeInstruction(state, outputTokenBudget);
  const recoveryContext = createRecoveryContextBlock(state, input, originalError);
  const recoveryInput = {
    userText: "Continue the current turn now.",
    modules: [],
    evidence: []
  };

  try {
    return await requestProviderText(state, recoveryInput, recoveryInstruction, recoveryContext, Math.min(outputTokenBudget, 900), {
      temperature: 0.2,
      geminiSafetyThreshold: "OFF",
      onRawText
    });
  } catch (recoveryError) {
    if (state.llm.provider === "gemini" && isGeminiOffSafetySettingRejected(recoveryError)) {
      return requestProviderText(state, recoveryInput, recoveryInstruction, recoveryContext, Math.min(outputTokenBudget, 900), {
        temperature: 0.2,
        geminiSafetyThreshold: "BLOCK_NONE",
        onRawText
      });
    }

    const originalMessage = originalError instanceof Error ? originalError.message : String(originalError);
    const recoveryMessage = recoveryError instanceof Error ? recoveryError.message : String(recoveryError);
    throw new Error(`${originalMessage}; recovery request failed: ${recoveryMessage}`);
  }
}

function createGeminiSafetySettings(threshold: GeminiSafetyThreshold): Array<{ category: string; threshold: GeminiSafetyThreshold }> {
  return GEMINI_SAFETY_CATEGORIES.map((category) => ({ category, threshold }));
}

function createRuntimeInstruction(state: AppState, outputTokenBudget: number): string {
  const contentRatingInstruction = createContentRatingInstruction(state);
  const imageCadenceInstruction = createImageGenerationCadenceBlock(state);
  return [
    state.llm.systemPrompt,
    contentRatingInstruction,
    "High-priority roster/relationship-map guard: registered characters, relationship values, current moods, and stored status records are background reference only. Do not mention, summon, move, react with, or include a character solely because they exist in the roster or relationship map; require current-scene evidence from the recent transcript, current User action, or an active scene rule.",
    "DynamicChat execution order:",
    "1. Read the Simulation foundation first. Every Main rules block is creator-authored operating law for this turn.",
    "2. Read the Selected prompt modules for this turn after the foundation. Apply always modules and locally/neuralmap/manually selected modules as active rules, not as optional summaries.",
    "3. Continue the exact current scene from the Recent transcript and User action. assistant_text must be an actual Korean simulation continuation, not a meta acknowledgement that the next scene will continue.",
    "4. If creator rules define a response structure, status window, choice format, or required ending block, put that complete structure inside assistant_text every turn while keeping the outer response valid JSON.",
    "Always preserve the Simulation foundation below: title, premise, existing character roster, and main rules are active every turn.",
    "The Immediate continuity anchor is the handoff point for this turn. Continue from the latest assistant ending and the current User action before using older retrieved memories.",
    "For long previous assistant output, the ending/status/choice block is more important than the opening. Do not restart from the prior response opening when a later ending is available.",
    "Use retrieved prompt modules and memory evidence as additional active context. If they conflict with the foundation, preserve explicit user-authored main rules unless the user changes them.",
    "Treat character summaries, relationships, current moods, active scene rules, and situation-specific modules as concrete setting constraints, not flavor text.",
    "Treat Structured simulation memory as the compact truth/current-state view. Do not infer that every character knows every fact; respect observation and belief notes when deciding character knowledge.",
    "Do not replace the main cast or invent unrelated protagonists. Introduce a new character only when the user action or retrieved context clearly requires one.",
    "If a user persona is active, treat it as the user's controlled role, background, goals, tone, and boundaries. If the persona source is an existing character, the User action is that character's action/dialogue/choice in the current scene; do not also puppet that controlled character beyond the user's explicit input.",
    "Input notation rule: any user text enclosed by the literal `*(` and `)*` marker, such as `*(문 쪽으로 이동한다)*`, is an action, stage direction, or descriptive instruction. Do not treat that enclosed text as spoken dialogue or internal thought unless the user explicitly labels it that way.",
    createOutputLengthInstruction(outputTokenBudget),
    "Use the Recent transcript to continue the exact current scene, location, cast, and momentum. Do not jump to a stale or unrelated scene label.",
    "Default assistant_text to natural Korean prose with Markdown only when useful. DynamicChat effect blocks are optional: use ```scene, ```impact, ```whisper, ```sfx, ```status, ```choice, ```memory, ```letter or one-line ::impact[text] only when the creator/user rules explicitly need a breakout visual beat. Do not write visible labels such as 'SFX:', 'impact:', or 'status -' as prose; effect labels are parser hints, not user-facing text. Do not output raw HTML.",
    "Image planning separation: focus this request on simulation continuation and structured memory. A dedicated lightweight image planner runs after assistant_text is complete and owns final image_cues, visible character selection, outfit/profile lookup, scene/body/dialogue targeting, and NovelAI tags.",
    "For image_cues in this main response, use [] by default. Only include a high-level hint if the creator rules explicitly demand an anchored cut that cannot be inferred from assistant_text; keep tags empty or minimal and let the dedicated image planner rewrite it.",
    "Image prompt user rules are binding composition and tag-routing instructions. Follow explicit positive/negative NovelAI tag directives, but never turn rule labels, examples, or headings into visible objects.",
    `Image generation cadence setting is binding: ${imageCadenceInstruction}`,
    "When a character's current outfit, pose, action, interaction, expression, held item, physical state, whole-scene phase, or scene composition changes, put that delta in memory_events with memory_kind='state'. Use state_type='Wearing', 'StatusTags', 'PoseTags', 'ActionTags', 'InteractionTags', 'InteractionPhaseTags', 'HeldItemTags', 'PhysicalStateTags', 'SceneTags', 'ScenePhaseTags', 'CompositionTags', 'CameraTags', or 'LightingTags'. state_value must be comma-separated English NovelAI tags, preferably 3-8 compact tags that preserve the situation rather than every micro-detail. Set actor_id for character-specific state. Leave actor_id empty only for whole-scene state. These records are appended to later image prompts before any local fallback is considered.",
    "Keep the outer JSON valid and stop cleanly. Do not compress assistant_text below the requested output length unless the user explicitly asks for brevity or the scene naturally requires a short beat.",
    "Your response must be valid JSON only. Put the user-visible narrative in assistant_text. Put only structured memory deltas in memory_events. Put visual planning in image_cues. If assistant_text contains Markdown code fences, encode them as a JSON string value; never write Markdown outside the JSON object."
  ].join("\n");
}

function createRecoveryRuntimeInstruction(state: AppState, outputTokenBudget: number): string {
  return [
    "You are DynamicChat's resilient scene-continuation writer.",
    "The previous provider request returned no usable content or a boilerplate refusal. Write a provider-compatible continuation of the current scene instead of an error message.",
    "Do not mention the model, provider, filters, policies, blocked content, retry behavior, or generation failure in assistant_text.",
    "Respect the active creator rules and the current scene. If the exact user action cannot be continued directly, pivot to adjacent dialogue, aftermath, environmental action, or a clear choice point while preserving continuity.",
    "Do not quote or intensify provider-sensitive details from the transcript. Keep the response grounded in safe actions, emotions, decisions, and consequences.",
    createOutputLengthInstruction(Math.min(outputTokenBudget, 1600)),
    `Simulation title: ${state.simulation.title}`,
    `Content rating: ${state.simulation.contentRating}`,
    "Return exactly this JSON shape with no Markdown outside JSON:",
    JSON.stringify({
      assistant_text: "Korean scene continuation shown to the user",
      memory_events: [],
      image_cues: []
    })
  ].join("\n");
}

function createRecoveryContextBlock(
  state: AppState,
  input: {
    userText: string;
    modules: PromptModule[];
    evidence: ContextEvidence[];
  },
  originalError: unknown
): string {
  const reason = originalError instanceof Error ? originalError.message : String(originalError);
  const recentTranscript = state.messages
    .filter((message) => !(message.role === "assistant" && looksLikeProviderBoilerplateText(message.content)))
    .slice(-2)
    .map((message) => `${message.role}: ${sanitizeProviderRetryText(message.content, 260)}`)
    .join("\n");
  const moduleText = input.modules
    .filter((module) => module.kind !== "image_prompt_profile")
    .filter((module) => !(module.kind === "safety_policy" && isAdultContentMode(state)))
    .slice(0, 4)
    .map((module) => `# ${module.title}\n${truncatePromptText(sanitizeProviderRetryText(module.body, 900), 900, "retry module excerpt")}`)
    .join("\n\n");
  const evidenceText = input.evidence
    .slice(0, 4)
    .map((item) => `- ${sanitizeProviderRetryText(item.snippet, 220)}`)
    .join("\n");

  return [
    "Scene-continuation retry context. Use it to write the next assistant_text, not to explain the retry.",
    `Provider failure summary: ${summarizeLlmFailureReason(reason)}`,
    state.simulation.description ? `Premise: ${sanitizeProviderRetryText(state.simulation.description, 320)}` : undefined,
    `Current user action:\n${sanitizeProviderRetryText(input.userText, 420)}`,
    recentTranscript ? `Recent transcript summary:\n${recentTranscript}` : undefined,
    moduleText ? `Active creator rules:\n${moduleText}` : undefined,
    evidenceText ? `Memory/context evidence:\n${evidenceText}` : undefined,
    "Return JSON only. Use image_cues: [] for this retry."
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function isGeminiOffSafetySettingRejected(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /Gemini request failed:\s*400/iu.test(message);
}

function isProviderNonContentError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /PROHIBITED_CONTENT|SAFETY|BLOCK|blocked|content_filter|refusal|did not include (?:text )?content|did not include content|empty candidate|empty response|finish:\s*(?:SAFETY|BLOCKLIST|PROHIBITED_CONTENT|RECITATION|content_filter)/iu.test(
    message
  );
}

function createGeminiEmptyContentMessage(data: GeminiResponse): string {
  const promptFeedback = data.promptFeedback;
  const candidate = data.candidates?.[0];
  const details = [
    promptFeedback?.blockReason ? `prompt block: ${promptFeedback.blockReason}` : undefined,
    promptFeedback?.blockReasonMessage ? `prompt block message: ${promptFeedback.blockReasonMessage}` : undefined,
    candidate?.finishReason ? `finish: ${candidate.finishReason}` : undefined,
    candidate?.finishMessage ? `finish message: ${candidate.finishMessage}` : undefined,
    formatGeminiSafetyRatings("prompt safety", promptFeedback?.safetyRatings),
    formatGeminiSafetyRatings("candidate safety", candidate?.safetyRatings)
  ].filter((item): item is string => Boolean(item));

  return details.length > 0
    ? `Gemini response did not include text content (${details.join("; ")}).`
    : "Gemini response did not include text content.";
}

function formatGeminiSafetyRatings(
  label: string,
  ratings: GeminiSafetyRating[] | undefined
): string | undefined {
  const blockedRatings = ratings
    ?.filter((rating) => rating.blocked || rating.probability)
    .map((rating) => [rating.category, rating.probability, rating.blocked ? "blocked" : undefined].filter(Boolean).join(":"));

  return blockedRatings && blockedRatings.length > 0 ? `${label}: ${blockedRatings.join(", ")}` : undefined;
}

function createRequestPreview(runtimeInstruction: string, contextBlock: string, userText: string): string {
  return [
    "SYSTEM INSTRUCTION:",
    truncatePreview(runtimeInstruction, 1400),
    "CONTEXT BLOCK:",
    truncatePreview(contextBlock, 3600),
    "USER ACTION:",
    truncatePreview(userText, 700)
  ]
    .join("\n\n");
}

function truncatePreview(value: string, maxChars: number): string {
  return value.length > maxChars ? `${value.slice(0, maxChars)}\n[...preview truncated...]` : value;
}

function createSimulationFoundationBlock(state: AppState, currentUserText = ""): string {
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state, currentUserText));
  const activeCharacters = state.characters
    .filter((character) => activeCharacterIds.has(character.id))
    .map((character) => formatFoundationCharacterLine(state, character, "active"))
    .join("\n");
  const inactiveCharacters = state.characters
    .filter((character) => !activeCharacterIds.has(character.id))
    .map((character) => formatFoundationCharacterLine(state, character, "reference"))
    .join("\n");
  const mainRuleModules = state.modules
    .filter(
      (module) =>
        module.enabled &&
        module.tokenPolicy !== "disabled" &&
        module.kind === "main_prompt"
    )
    .sort((a, b) => b.priority - a.priority)
    .map((module) => formatFoundationModule(module.title, module.tokenPolicy, module.body, MAIN_RULE_MAX_CHARS))
    .join("\n\n");
  const worldLoreModules = state.modules
    .filter((module) => module.enabled && module.tokenPolicy !== "disabled" && module.kind === "world_lore")
    .sort((a, b) => b.priority - a.priority)
    .map((module) => formatFoundationModule(module.title, module.tokenPolicy, module.body, 1600))
    .join("\n\n");
  const alwaysOnModules = state.modules
    .filter(
      (module) =>
        module.enabled &&
        module.tokenPolicy === "always" &&
        !["main_prompt", "world_lore", "image_prompt_profile"].includes(module.kind) &&
        !(module.kind === "safety_policy" && isAdultContentMode(state))
    )
    .sort((a, b) => b.priority - a.priority)
    .map((module) => formatFoundationModule(module.title, module.tokenPolicy, module.body, 1200))
    .join("\n\n");

  return [
    `Simulation: ${state.simulation.title}`,
    `Prompt mode: ${state.simulation.promptMode}`,
    `Content rating: ${state.simulation.contentRating}`,
    state.simulation.description ? `Premise: ${state.simulation.description}` : undefined,
    createSceneCastPromptBlock(state, currentUserText),
    activeCharacters ? `Current scene character details:\n${activeCharacters}` : "Current scene character details: (none inferred)",
    inactiveCharacters
      ? `Registered character roster reference only, not active cast:\n${inactiveCharacters}`
      : "Registered character roster reference only, not active cast: (none)",
    mainRuleModules ? `Main rules:\n${mainRuleModules}` : "Main rules: (none)",
    worldLoreModules ? `World/lore foundation:\n${worldLoreModules}` : undefined,
    alwaysOnModules ? `Always-on rules:\n${alwaysOnModules}` : undefined
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function formatFoundationCharacterLine(
  state: AppState,
  character: AppState["characters"][number],
  mode: "active" | "reference"
): string {
  const visualProfile = mode === "active"
    ? state.visualProfiles.find((profile) => profile.characterId === character.id)
    : undefined;
  const outfitMappings = formatOutfitMappingsForFoundation(visualProfile?.outfitPrompts);
  const summaryLimit = mode === "active" ? 700 : 260;
  return [
    `- id: ${character.id}`,
    `name: ${character.name}`,
    character.role ? `role: ${character.role}` : undefined,
    character.summary ? `summary: ${truncatePromptText(character.summary, summaryLimit, "character summary")}` : undefined,
    mode === "active" && character.relationship ? `relationship: ${truncatePromptText(character.relationship, 420, "character relationship")}` : undefined,
    mode === "active" && character.currentMood ? `current mood: ${character.currentMood}` : undefined,
    visualProfile?.defaultOutfitPrompt ? `default outfit tags: ${visualProfile.defaultOutfitPrompt}` : undefined,
    outfitMappings ? `outfit keyword mappings: ${outfitMappings}` : undefined
  ]
    .filter((item): item is string => Boolean(item))
    .join(" | ");
}

function formatOutfitMappingsForFoundation(outfitPrompts: Record<string, string> | undefined): string | undefined {
  const entries = Object.entries(outfitPrompts ?? {})
    .map(([key, value]) => [key.trim(), value.trim()] as const)
    .filter(([key, value]) => key && value)
    .slice(0, 8);

  return entries.length > 0 ? entries.map(([key, value]) => `${key} => ${value}`).join("; ") : undefined;
}

function createImmediateContinuityBlock(state: AppState): string {
  const latestAssistant = findLatestUsableAssistantMessage(state);
  const latestUser = findLatestMessageByRole(state, "user");
  if (!latestAssistant && !latestUser) {
    return "";
  }

  return [
    "Highest priority for scene continuity. The next assistant_text must begin from this handoff point, then apply the current User action.",
    latestAssistant
      ? `Latest assistant ending:\n${createContinuityExcerpt(latestAssistant.content, CONTINUITY_ANCHOR_CHARS, "tail")}`
      : undefined,
    latestUser
      ? `Previous user action:\n${createContinuityExcerpt(latestUser.content, CONTINUITY_PREVIOUS_USER_CHARS, "balanced")}`
      : undefined
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function formatFoundationModule(title: string, tokenPolicy: string, body: string, maxChars: number): string {
  const excerpt = truncatePromptText(body, maxChars, "foundation excerpt");
  return `# ${title} (${tokenPolicy})\n${excerpt}`;
}

function createRecentTranscriptBlock(state: AppState): string {
  const recentMessages = state.messages
    .filter((message) => !(message.role === "assistant" && looksLikeProviderBoilerplateText(message.content)))
    .slice(-8);
  const latestAssistantId = [...recentMessages].reverse().find((message) => message.role === "assistant")?.id;
  return recentMessages
    .map((message) => {
      const isLatestAssistant = message.id === latestAssistantId;
      const content = createContinuityExcerpt(
        message.content,
        RECENT_TRANSCRIPT_MESSAGE_CHARS,
        isLatestAssistant ? "tail" : "balanced"
      );
      const roleLabel = isLatestAssistant ? `${message.role} (latest ending)` : message.role;
      return content ? `${roleLabel}: ${content}` : "";
    })
    .filter(Boolean)
    .join("\n");
}

function findLatestMessageByRole(state: AppState, role: AppState["messages"][number]["role"]): AppState["messages"][number] | undefined {
  return [...state.messages].reverse().find((message) => message.role === role);
}

function findLatestUsableAssistantMessage(state: AppState): AppState["messages"][number] | undefined {
  return [...state.messages]
    .reverse()
    .find((message) => message.role === "assistant" && !looksLikeProviderBoilerplateText(message.content));
}

function createContinuityExcerpt(value: string, maxChars: number, mode: "balanced" | "tail"): string {
  const normalized = value.replace(/\s+/gu, " ").trim();
  if (normalized.length <= maxChars) {
    return normalized;
  }

  if (mode === "tail") {
    return `[...earlier text omitted for continuity...] ${normalized.slice(-maxChars)}`;
  }

  const headChars = Math.floor(maxChars * 0.42);
  const tailChars = maxChars - headChars;
  return `${normalized.slice(0, headChars)} [...middle omitted for continuity...] ${normalized.slice(-tailChars)}`;
}

function createOutputLengthInstruction(maxTokens: number): string {
  if (maxTokens <= 1000) {
    return "Output length target: compact. Use the output token setting as a real length target, not only a hard ceiling. assistant_text should usually be 2-3 Korean paragraphs with concrete action, dialogue, and one clear consequence. Required status/choice blocks are extra and must not replace the narrative.";
  }

  if (maxTokens <= 1800) {
    return "Output length target: balanced. Use the output token setting as a real length target, not only a hard ceiling. assistant_text should usually be 4-6 Korean paragraphs with concrete action, dialogue, sensory detail, and visible consequences from the active main rules. Required status/choice blocks are extra and must not replace the narrative.";
  }

  if (maxTokens <= 3000) {
    return "Output length target: long. Use the output token setting as a real length target, not only a hard ceiling. assistant_text should usually be 7-10 Korean paragraphs, developing the scene through multiple beats of action, dialogue, reaction, and consequences before closing cleanly. Required status/choice blocks are extra and must not replace the narrative.";
  }

  if (maxTokens <= 4500) {
    return "Output length target: very long. Use the output token setting as a real length target, not only a hard ceiling. assistant_text should usually be 10-14 Korean paragraphs, with richer scene progression, character reactions, state changes, and a complete required status/choice block if creator rules ask for one.";
  }

  return "Output length target: extended. Use the output token setting as a real length target, not only a hard ceiling. assistant_text should usually be 14-18 Korean paragraphs, with substantial scene progression, dialogue, consequences, and complete required status/choice blocks, then stop cleanly and close the JSON.";
}

function truncatePromptText(value: string, maxChars: number, label: string): string {
  const normalized = value.trim();
  if (normalized.length <= maxChars) {
    return normalized;
  }

  return `${normalized.slice(0, maxChars)}\n[...${label} truncated for interactive latency...]`;
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    try {
      return await fetch(url, {
        ...init,
        signal: controller.signal
      });
    } catch (error) {
      if (controller.signal.aborted) {
        throw new Error(`LLM request timed out after ${Math.round(timeoutMs / 1000)}s.`);
      }
      throw error;
    }
  } finally {
    globalThis.clearTimeout(timeoutId);
  }
}

function createUserPersonaBlock(state: AppState): string {
  const persona = state.userPersona;
  if (!persona?.enabled) {
    return "";
  }

  const character = resolvePersonaCharacter(state);
  if (persona.source === "character" && character) {
    return [
      "source: existing_character",
      `controlled_character_id: ${character.id}`,
      `name: ${character.name}`,
      `image_cue_actor_rule: User actions, first-person narration, and '나' refer to controlled_character_id ${character.id}; include this id in image_cues.characters when this character is the visible actor or speaker.`,
      character.role.trim() ? `role: ${character.role.trim()}` : undefined,
      character.summary.trim() ? `background: ${character.summary.trim()}` : undefined,
      character.relationship.trim() ? `relationship_to_player_or_cast: ${character.relationship.trim()}` : undefined,
      character.currentMood.trim() ? `current_state: ${character.currentMood.trim()}` : undefined,
      persona.goals.trim() ? `player_goals_for_this_character: ${persona.goals.trim()}` : undefined,
      persona.style.trim() ? `input_style: ${persona.style.trim()}` : undefined,
      persona.boundaries.trim() ? `boundaries: ${persona.boundaries.trim()}` : undefined
    ]
      .filter((item): item is string => Boolean(item))
      .join("\n");
  }

  return [
    "source: custom_persona",
    persona.name.trim() ? `name: ${persona.name.trim()}` : undefined,
    persona.role.trim() ? `role: ${persona.role.trim()}` : undefined,
    persona.background.trim() ? `background: ${persona.background.trim()}` : undefined,
    persona.goals.trim() ? `goals: ${persona.goals.trim()}` : undefined,
    persona.style.trim() ? `style: ${persona.style.trim()}` : undefined,
    persona.boundaries.trim() ? `boundaries: ${persona.boundaries.trim()}` : undefined
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n");
}

function resolvePersonaCharacter(state: AppState): AppState["characters"][number] | undefined {
  const persona = state.userPersona;
  if (!persona?.enabled || persona.source !== "character" || !persona.characterId) {
    return undefined;
  }

  return state.characters.find((character) => character.id === persona.characterId);
}

function createImageUserRulesBlock(state: AppState): string {
  return createImageUserRulesForContentRating(state)
    .split(/\n+/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 40)
    .join("\n");
}

function parseAssistantSidecar(raw: string): { sidecar?: AssistantSidecar; errors: string[] } {
  const errors: string[] = [];
  const jsonText = extractJsonObject(raw);
  if (!jsonText) {
    return { errors: ["No JSON object found in LLM response."] };
  }

  try {
    const value = JSON.parse(jsonText) as Record<string, unknown>;
    const assistantText = readString(value.assistant_text) ?? readString(value.assistantText);
    if (!assistantText) {
      errors.push("assistant_text is missing or invalid.");
    }

    const memoryEventsValue = Array.isArray(value.memory_events)
      ? value.memory_events
      : Array.isArray(value.memoryEvents)
        ? value.memoryEvents
        : [];
    const memoryEvents = memoryEventsValue
      .map((item) => normalizeMemoryEventDraft(item, errors))
      .filter((item): item is AssistantSidecar["memoryEvents"][number] => Boolean(item));
    const imageCueValue = (value.image_cues ?? value.imageCue ?? value.image_cue) as unknown;
    const imageCues = normalizeAssistantImageCueDrafts(imageCueValue, errors);
    const imageCue = imageCues[0] ?? createNoImageCue("메인 LLM이 현재 문맥상 이미지 cue를 생략함");

    if (!assistantText) {
      return { errors };
    }

    return {
      sidecar: {
        assistantText,
        memoryEvents,
        imageCue,
        imageCues
      },
      errors
    };
  } catch (error) {
    return {
      errors: [error instanceof Error ? error.message : "Invalid JSON sidecar."]
    };
  }
}

function normalizeMemoryEventDraft(value: unknown, errors: string[]): AssistantSidecar["memoryEvents"][number] | undefined {
  if (!isRecord(value)) {
    errors.push("memory_events item is not an object.");
    return undefined;
  }

  const content = readString(value.content);
  if (!content) {
    errors.push("memory_events item missing content.");
    return undefined;
  }

  return {
    content,
    importance: clampNumber(readNumber(value.importance) ?? 0.7, 0, 1),
    tags: readStringArray(value.tags).slice(0, 8),
    actorId: readString(value.actor_id) ?? readString(value.actorId),
    actorName: readString(value.actor_name) ?? readString(value.actorName),
    memoryKind: normalizeMemoryKind(readString(value.memory_kind) ?? readString(value.memoryKind) ?? readString(value.kind)),
    eventType: readString(value.event_type) ?? readString(value.eventType),
    stateType: readString(value.state_type) ?? readString(value.stateType),
    stateValue: readString(value.state_value) ?? readString(value.stateValue) ?? readString(value.value),
    targetId: readString(value.target_id) ?? readString(value.targetId),
    observers: readStringArray(value.observers).slice(0, 8),
    confidence: clampNumber(readNumber(value.confidence) ?? 0.78, 0, 1)
  };
}

function normalizeMemoryKind(value: string | undefined): AssistantSidecar["memoryEvents"][number]["memoryKind"] {
  const normalized = value?.trim().toLowerCase().replace(/-/gu, "_");
  if (
    normalized === "event" ||
    normalized === "state" ||
    normalized === "observation" ||
    normalized === "belief" ||
    normalized === "goal" ||
    normalized === "relationship" ||
    normalized === "open_thread" ||
    normalized === "summary"
  ) {
    return normalized;
  }

  return undefined;
}

function normalizeImageCueDraft(value: unknown, errors: string[]): PlannedImageCueDraft | undefined {
  if (!isRecord(value)) {
    errors.push("image_cues is missing or invalid.");
    return undefined;
  }

  const reason = readString(value.reason) ?? "";
  const scene = readString(value.scene) ?? "current scene";
  const tags = readStringArray(value.tags);
  const visualContext = readString(value.visual_context) ?? readString(value.visualContext);
  const explicitShouldGenerate = readBoolean(value.should_generate) ?? readBoolean(value.shouldGenerate);
  const kind = readString(value.kind) ?? readString(value.type) ?? readString(value.cue_type) ?? readString(value.cueType);
  return {
    shouldGenerate: explicitShouldGenerate ?? Boolean(tags.length > 0 || visualContext),
    reason,
    characters: readStringArray(value.characters),
    tags,
    scene,
    suppressionReason: readString(value.suppression_reason) ?? readString(value.suppressionReason),
    visualContext,
    label: readString(value.label) ?? readString(value.name),
    kind,
    cueType: kind,
    placement: normalizeImageCuePlacement(readString(value.placement)),
    anchorText: readString(value.anchor_text) ?? readString(value.anchorText) ?? readString(value.anchor),
    priority: clampNumber(readNumber(value.priority) ?? 0.5, 0, 1)
  };
}

function createFallbackSidecar(
  assistantText: string,
  reason = "Structured sidecar fallback; image cue will use local main-turn fallback."
): AssistantSidecar {
  const imageCue = {
    shouldGenerate: false,
    reason,
    characters: [],
    tags: [],
    scene: "current scene",
    suppressionReason: "No structured image cue was available.",
    visualContext: assistantText.slice(0, 220)
  };
  return {
    assistantText,
    memoryEvents: [],
    imageCue,
    imageCues: [imageCue]
  };
}

function normalizeAssistantImageCueDrafts(value: unknown, errors: string[]): PlannedImageCueDraft[] {
  if (Array.isArray(value)) {
    return value
      .map((item) => normalizeImageCueDraft(item, errors))
      .filter((item): item is PlannedImageCueDraft => Boolean(item));
  }

  if (value === undefined || value === null) {
    return [];
  }

  const draft = normalizeImageCueDraft(value, errors);
  return draft ? [draft] : [];
}

function createNoImageCue(reason: string): PlannedImageCueDraft {
  return {
    shouldGenerate: false,
    reason,
    characters: [],
    tags: [],
    scene: "current scene",
    suppressionReason: "메인 LLM이 이미지 생성을 선택하지 않음"
  };
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter((value) => Boolean(value?.trim()))));
}

function extractJsonObject(raw: string): string | undefined {
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/iu)?.[1]?.trim();
  const candidate = fenced ?? raw.trim();
  const firstBrace = candidate.indexOf("{");
  const lastBrace = candidate.lastIndexOf("}");
  if (firstBrace < 0 || lastBrace <= firstBrace) {
    return undefined;
  }

  return candidate.slice(firstBrace, lastBrace + 1);
}

function stripLikelyJsonFence(raw: string): string {
  return raw.replace(/```(?:json)?/giu, "").replace(/```/gu, "");
}

function createDisplayFallbackText(raw: string, fallback: string): string {
  const assistantText =
    extractJsonStringField(raw, ["assistant_text", "assistantText"]) ??
    extractLooseJsonTextField(raw, ["assistant_text", "assistantText"]);
  if (assistantText) {
    return assistantText;
  }

  const stripped = stripLikelyJsonFence(raw).trim();
  if (!stripped || looksLikeSidecarJson(stripped)) {
    return fallback;
  }

  return stripped;
}

function looksLikeSidecarJson(value: string): boolean {
  const trimmed = value.trim();
  return (
    /^[{[]/u.test(trimmed) &&
    /"?(?:assistant_text|assistantText|memory_events|memoryEvents|image_cues|imageCue)"?\s*:/u.test(trimmed)
  );
}

function extractJsonStringField(raw: string, fieldNames: string[]): string | undefined {
  const text = stripLikelyJsonFence(raw);
  for (const fieldName of fieldNames) {
    const match = new RegExp(`"${fieldName}"\\s*:\\s*"`, "u").exec(text);
    if (!match) {
      continue;
    }

    const value = readPossiblyTruncatedJsonString(text, match.index + match[0].length);
    if (value.trim()) {
      return value.trim();
    }
  }

  return undefined;
}

function extractLooseJsonTextField(raw: string, fieldNames: string[]): string | undefined {
  const text = stripLikelyJsonFence(raw);
  for (const fieldName of fieldNames) {
    const match =
      new RegExp(`(?:\\\\?["'])${fieldName}(?:\\\\?["'])\\s*:\\s*`, "iu").exec(text) ??
      new RegExp(`\\b${fieldName}\\b\\s*:\\s*`, "iu").exec(text);
    if (!match) {
      continue;
    }

    const value = cleanLooseJsonTextValue(text.slice(match.index + match[0].length));
    if (value) {
      return value;
    }
  }

  return undefined;
}

function cleanLooseJsonTextValue(rawValue: string): string {
  const nextFieldMatch = /,\s*(?:\\?["'])(?:memory_events|memoryEvents|image_cues|imageCue|image_cue)(?:\\?["'])\s*:/iu.exec(rawValue);
  const boundedValue = nextFieldMatch ? rawValue.slice(0, nextFieldMatch.index) : rawValue;
  const trimmedValue = boundedValue.trim();
  const parsedValue = parseJsonStringLiteralIfComplete(trimmedValue);
  if (parsedValue) {
    return parsedValue.trim();
  }

  return trimmedValue
    .replace(/^\s*["']/u, "")
    .replace(/\s*[,}]\s*$/u, "")
    .replace(/\s*["']\s*$/u, "")
    .replace(/\\n/gu, "\n")
    .replace(/\\r/gu, "\r")
    .replace(/\\t/gu, "\t")
    .replace(/\\"/gu, "\"")
    .replace(/\\\\/gu, "\\")
    .trim();
}

function parseJsonStringLiteralIfComplete(value: string): string | undefined {
  if (!/^"/u.test(value)) {
    return undefined;
  }

  const endIndex = findCompleteJsonStringEnd(value, 0);
  if (endIndex < 0) {
    return undefined;
  }
  const remainder = value.slice(endIndex + 1).trim();
  if (remainder && !/^[,}]/u.test(remainder)) {
    return undefined;
  }

  try {
    return JSON.parse(value.slice(0, endIndex + 1)) as string;
  } catch {
    return undefined;
  }
}

function findCompleteJsonStringEnd(value: string, startIndex: number): number {
  let escaped = false;
  for (let index = startIndex + 1; index < value.length; index += 1) {
    const char = value[index];
    if (escaped) {
      escaped = false;
      continue;
    }

    if (char === "\\") {
      escaped = true;
      continue;
    }

    if (char === "\"") {
      return index;
    }
  }

  return -1;
}

function readPossiblyTruncatedJsonString(text: string, startIndex: number): string {
  let literal = "\"";
  let escaped = false;

  for (let index = startIndex; index < text.length; index += 1) {
    const char = text[index];
    if (escaped) {
      literal += `\\${char}`;
      escaped = false;
      continue;
    }

    if (char === "\\") {
      escaped = true;
      continue;
    }

    if (char === "\"") {
      literal += "\"";
      return decodeJsonStringLiteral(literal);
    }

    literal += char === "\n" ? "\\n" : char === "\r" ? "\\r" : char;
  }

  return decodeJsonStringLiteral(`${literal.replace(/\\$/u, "")}"`);
}

function decodeJsonStringLiteral(literal: string): string {
  try {
    return JSON.parse(literal) as string;
  } catch {
    return literal
      .slice(1, -1)
      .replace(/\\n/gu, "\n")
      .replace(/\\"/gu, "\"")
      .replace(/\\\\/gu, "\\");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter(Boolean);
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeImageCuePlacement(value: string | undefined): PlannedImageCueDraft["placement"] | undefined {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "before" || normalized === "after" || normalized === "inline") {
    return normalized;
  }

  return undefined;
}
