import type {
  AppState,
  AssistantImageCueDraft,
  AssistantSidecar,
  ContextEvidence,
  ImageCueCharacterPrompt,
  ImageSceneTagPresetNode,
  PromptModule
} from "../types";
import { createImageUserRulesForContentRating, isAdultContentMode } from "./contentRating";
import { IMAGE_STATE_TYPE_INSTRUCTION } from "./imageStateTags";
import { createStructuredContextSummary } from "./memoryCompiler";
import { createSceneCastPromptBlock, inferCurrentSceneCharacterIds } from "./sceneCast";
import {
  readStateMemoryKind,
  readStateMemoryOwnerId,
  readStateMemoryStateType,
  readStateMemoryValue
} from "./stateMemory";

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
  allowRecovery?: boolean;
  allowProviderSafeRecovery?: boolean;
  onRawText?: (rawText: string) => void;
}

const MIN_OUTPUT_TOKENS = 512;
const INTERACTIVE_OUTPUT_TOKEN_CAP = 8000;
const MIN_LLM_REQUEST_TIMEOUT_MS = 45_000;
const MAX_LLM_REQUEST_TIMEOUT_MS = 180_000;
const MAIN_RULE_MAX_CHARS = 5200;
const SELECTED_MODULE_MAX_CHARS = 1800;
const RECENT_TRANSCRIPT_MESSAGE_CHARS = 800;
const CONTINUITY_ANCHOR_CHARS = 1800;
const CONTINUITY_PREVIOUS_USER_CHARS = 700;
const SHORT_OUTPUT_CONTINUATION_MIN_CHARS = 360;
const IMAGE_PATTERN =
  /그려|보여|이미지|장면|모습|표정|빛|배경|의상|옷|복장|학교|교실|연습|훈련|무대|숙소|기숙|주방|거리|사무실|카페|병원|전투|여행/u;
const IMAGE_PROGRESSION_CUE_TARGET = 10;
const IMAGE_PROGRESSION_MIN_OUTPUT_TOKENS = 4200;
const IMAGE_SCENE_PRESET_PROMPT_LIMIT = 18;
const IMAGE_SCENE_PRESET_PROMPT_MAX_DEPTH = 5;
const IMAGE_SCENE_PRESET_SEARCH_NODE_LIMIT = 240;
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
const MALFORMED_SIDECAR_RECOVERY_MARKER = "Recovered assistant sidecar from malformed JSON.";

export interface PlannedImageCueDraft extends AssistantImageCueDraft {
}

export async function generateAssistantText(input: {
  state: AppState;
  userText: string;
  modules: PromptModule[];
  evidence: ContextEvidence[];
  fallback: string;
  manualImage?: boolean;
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
  const outputTokenBudget = resolveInteractiveOutputTokenBudget(state);
  const contextBlock = createContextBlock(state, input.userText, input.modules, input.evidence, {
    manualImage: input.manualImage,
    outputTokenBudget
  });
  const runtimeInstruction = createRuntimeInstruction(state, outputTokenBudget, {
    manualImage: input.manualImage
  });
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
      allowProviderSafeRecovery: true,
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
        { allowProviderSafeRecovery: true, onRawText: emitAssistantText }
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
      const sidecarCompletionErrors: string[] = [];
      if (shouldRepairShortAssistantText(state, input.userText, parsed.sidecar.assistantText, outputTokenBudget)) {
        const originalSidecar = parsed.sidecar;
        const repaired = await requestShortAssistantTextRepair(
          state,
          input,
          runtimeInstruction,
          contextBlock,
          outputTokenBudget,
          originalSidecar.assistantText,
          emitAssistantText
        ).catch((repairError: unknown) => {
          sidecarCompletionErrors.push(`short assistant_text rewrite failed: ${formatUnknownError(repairError)}`);
          return undefined;
        });
        if (repaired?.parsed.sidecar) {
          parsed = repaired.parsed;
          rawContent = repaired.rawContent;
        } else {
          const continued = await requestShortAssistantTextContinuation(
            state,
            input,
            runtimeInstruction,
            contextBlock,
            outputTokenBudget,
            originalSidecar.assistantText,
            emitAssistantText
          ).catch((continuationError: unknown) => {
            sidecarCompletionErrors.push(`short assistant_text continuation failed: ${formatUnknownError(continuationError)}`);
            return undefined;
          });
          if (continued?.parsed.sidecar) {
            const continuedSidecar = appendAssistantContinuationSidecar(originalSidecar, continued.parsed.sidecar);
            if (countVisibleTextChars(continuedSidecar.assistantText) > countVisibleTextChars(originalSidecar.assistantText)) {
              parsed = {
                sidecar: continuedSidecar,
                errors: uniqueStrings([...parsed.errors, ...continued.parsed.errors])
              };
              rawContent = `${rawContent}\n\n${continued.rawContent}`;
              sidecarCompletionErrors.length = 0;
            }
          }
        }

        if (shouldRepairShortAssistantText(state, input.userText, parsed.sidecar?.assistantText ?? "", outputTokenBudget)) {
          sidecarCompletionErrors.push("assistant_text remained shorter than the selected output target after repair attempts.");
        }
      }
      if (parsed.sidecar && shouldRepairAbruptAssistantText(state, input.userText, parsed.sidecar.assistantText, outputTokenBudget, parsed.errors)) {
        const originalSidecar = parsed.sidecar;
        const continued = await requestShortAssistantTextContinuation(
          state,
          input,
          runtimeInstruction,
          contextBlock,
          outputTokenBudget,
          originalSidecar.assistantText,
          emitAssistantText
        ).catch((continuationError: unknown) => {
          sidecarCompletionErrors.push(`cut-off assistant_text continuation failed: ${formatUnknownError(continuationError)}`);
          return undefined;
        });
        if (continued?.parsed.sidecar) {
          const continuedSidecar = appendAssistantContinuationSidecar(originalSidecar, continued.parsed.sidecar);
          if (countVisibleTextChars(continuedSidecar.assistantText) > countVisibleTextChars(originalSidecar.assistantText) + 80) {
            parsed = {
              sidecar: continuedSidecar,
              errors: uniqueStrings([...parsed.errors, ...continued.parsed.errors, "assistant_text cut-off was continued locally."])
            };
            rawContent = `${rawContent}\n\n${continued.rawContent}`;
            sidecarCompletionErrors.length = 0;
          }
        }

        if (shouldRepairAbruptAssistantText(state, input.userText, parsed.sidecar?.assistantText ?? "", outputTokenBudget, parsed.errors)) {
          sidecarCompletionErrors.push("assistant_text appears cut off before a complete handoff");
        }
      }
      if (parsed.sidecar && hasMalformedAssistantSidecarRecovery(parsed.errors)) {
        const previousSidecar = parsed.sidecar;
        const repaired = await requestAssistantSidecarCompletionRetry(
          state,
          input,
          runtimeInstruction,
          contextBlock,
          outputTokenBudget,
          previousSidecar,
          undefined,
          {
            forcePreserveAssistantText: true,
            extraIssues: ["memory_events and image_cues may have been dropped by malformed JSON"],
            rejectionReason:
              "Previous LLM output was malformed JSON. DynamicChat recovered the visible assistant_text, but structured memory_events/image_cues may have been lost."
          }
        ).catch((repairError: unknown) => {
          sidecarCompletionErrors.push(`malformed sidecar metadata repair failed: ${formatUnknownError(repairError)}`);
          return undefined;
        });
        if (repaired?.parsed.sidecar && hasAssistantSidecarMetadataImprovement(previousSidecar, repaired.parsed.sidecar)) {
          parsed = {
            sidecar: repaired.parsed.sidecar,
            errors: uniqueStrings([
              ...parsed.errors,
              ...repaired.parsed.errors,
              "malformed sidecar metadata was repaired without rewriting assistant_text."
            ])
          };
          rawContent = `${rawContent}\n\n${repaired.rawContent}`;
          sidecarCompletionErrors.length = 0;
        }
      }
      if (parsed.sidecar && shouldRetryIncompleteAssistantSidecar(state, input.userText, parsed.sidecar, outputTokenBudget, Boolean(input.manualImage))) {
        const previousSidecar = parsed.sidecar;
        const completed = await requestAssistantSidecarCompletionRetry(
          state,
          input,
          runtimeInstruction,
          contextBlock,
          outputTokenBudget,
          previousSidecar,
          emitAssistantText
        ).catch((completionError: unknown) => {
          sidecarCompletionErrors.push(`main sidecar completion retry failed: ${formatUnknownError(completionError)}`);
          return undefined;
        });
        if (
          completed?.parsed.sidecar &&
          shouldAcceptAssistantSidecarCompletionRetry(
            state,
            input.userText,
            previousSidecar,
            completed.parsed.sidecar,
            outputTokenBudget,
            Boolean(input.manualImage)
          )
        ) {
          parsed = completed.parsed;
          rawContent = completed.rawContent;
          sidecarCompletionErrors.length = 0;
        } else {
          sidecarCompletionErrors.push(createAssistantSidecarCompletionError(state, input.userText, parsed.sidecar, outputTokenBudget, Boolean(input.manualImage)));
        }
      }
      const sidecar = parsed.sidecar;
      if (!sidecar) {
        throw new Error("LLM sidecar became unavailable after short-output repair.");
      }

      input.onAssistantText?.(sidecar.assistantText);
      return {
        content: sidecar.assistantText,
        sidecar,
        source: "llm",
        sidecarStatus: "parsed",
        sidecarErrors: uniqueStrings([...parsed.errors, ...sidecarCompletionErrors]),
        requestPreview,
        rawPreview: rawContent.slice(0, 700)
      };
    }

    const fallbackText = createDisplayFallbackText(rawContent, "");
    if (fallbackText) {
      if (looksLikeProviderBoilerplateText(fallbackText)) {
        throw new Error("LLM returned provider boilerplate instead of a scene continuation.");
      }
      const initialFallbackSidecar = createFallbackSidecar(
        fallbackText,
        "Structured sidecar parse failed; main response completion retry may recover image tags."
      );
      let fallbackSidecar = initialFallbackSidecar;
      const fallbackRecoveryErrors: string[] = [];
      let fallbackMetadataImproved = false;
      const metadataRecovered = await requestAssistantSidecarCompletionRetry(
        state,
        input,
        runtimeInstruction,
        contextBlock,
        outputTokenBudget,
        fallbackSidecar,
        undefined,
        {
          forcePreserveAssistantText: true,
          extraIssues: ["memory_events and image_cues were unavailable after sidecar parse failure"],
          rejectionReason:
            "Previous LLM output was invalid JSON. DynamicChat recovered the visible assistant_text, but the structured sidecar metadata was unavailable."
        }
      ).catch((repairError: unknown) => {
        fallbackRecoveryErrors.push(`sidecar metadata repair failed: ${formatUnknownError(repairError)}`);
        return undefined;
      });
      if (metadataRecovered?.parsed.sidecar && hasAssistantSidecarMetadataImprovement(initialFallbackSidecar, metadataRecovered.parsed.sidecar)) {
        fallbackSidecar = metadataRecovered.parsed.sidecar;
        fallbackMetadataImproved = true;
        fallbackRecoveryErrors.push("sidecar parse failure metadata was recovered without rewriting assistant_text.");
      }
      if (
        fallbackMetadataImproved &&
        !shouldRetryIncompleteAssistantSidecar(state, input.userText, fallbackSidecar, outputTokenBudget, Boolean(input.manualImage))
      ) {
        input.onAssistantText?.(fallbackSidecar.assistantText);
        return {
          content: fallbackSidecar.assistantText,
          sidecar: fallbackSidecar,
          source: "llm",
          sidecarStatus: "parsed",
          sidecarErrors: uniqueStrings([
            ...parsed.errors,
            ...(metadataRecovered?.parsed.errors ?? []),
            ...fallbackRecoveryErrors
          ]),
          requestPreview,
          rawPreview: (metadataRecovered?.rawContent ?? rawContent).slice(0, 700)
        };
      }
      const recovered = await requestAssistantSidecarCompletionRetry(
        state,
        input,
        runtimeInstruction,
        contextBlock,
        outputTokenBudget,
        fallbackSidecar,
        emitAssistantText
      ).catch(() => undefined);
      if (
        recovered?.parsed.sidecar &&
        shouldAcceptAssistantSidecarCompletionRetry(state, input.userText, fallbackSidecar, recovered.parsed.sidecar, outputTokenBudget, Boolean(input.manualImage))
      ) {
        input.onAssistantText?.(recovered.parsed.sidecar.assistantText);
        return {
          content: recovered.parsed.sidecar.assistantText,
          sidecar: recovered.parsed.sidecar,
          source: "llm",
          sidecarStatus: "parsed",
          sidecarErrors: uniqueStrings([
            ...parsed.errors,
            ...fallbackRecoveryErrors,
            ...recovered.parsed.errors,
            "sidecar parse failure was recovered by a main response completion retry."
          ]),
          requestPreview,
          rawPreview: recovered.rawContent.slice(0, 700)
        };
      }
      if (fallbackMetadataImproved) {
        input.onAssistantText?.(fallbackSidecar.assistantText);
        return {
          content: fallbackSidecar.assistantText,
          sidecar: fallbackSidecar,
          source: "llm",
          sidecarStatus: "parsed",
          sidecarErrors: uniqueStrings([
            ...parsed.errors,
            ...fallbackRecoveryErrors,
            createAssistantSidecarCompletionError(state, input.userText, fallbackSidecar, outputTokenBudget, Boolean(input.manualImage))
          ]),
          requestPreview,
          rawPreview: (metadataRecovered?.rawContent ?? rawContent).slice(0, 700)
        };
      }
      input.onAssistantText?.(fallbackText);
      return {
        content: fallbackText,
        sidecar: fallbackSidecar,
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
    const fallbackContent = isProviderNonContentError(error)
      ? createProviderBlockedFallbackContent(state, input.userText)
      : createLlmFailureFallbackContent(input.fallback, errorMessage);
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

async function requestAssistantSidecarCompletionRetry(
  state: AppState,
  input: {
    userText: string;
    modules: PromptModule[];
    evidence: ContextEvidence[];
    manualImage?: boolean;
  },
  runtimeInstruction: string,
  contextBlock: string,
  outputTokenBudget: number,
  previousSidecar: AssistantSidecar,
  onRawText?: (rawText: string) => void,
  options: {
    forcePreserveAssistantText?: boolean;
    extraIssues?: string[];
    rejectionReason?: string;
  } = {}
): Promise<{ rawContent: string; parsed: ReturnType<typeof parseAssistantSidecar> } | undefined> {
  const minimumChars = resolveMinimumAssistantTextChars(outputTokenBudget);
  const issues = uniqueStrings([
    ...createAssistantSidecarCompletionIssues(state, input.userText, previousSidecar, outputTokenBudget, Boolean(input.manualImage)),
    ...(options.extraIssues ?? [])
  ]);
  const preserveAssistantText = Boolean(options.forcePreserveAssistantText) || shouldPreserveAssistantTextForSidecarRetry(issues);
  const requiresImageTags = shouldRequireMainImageTags(state, input.userText, previousSidecar.assistantText, Boolean(input.manualImage));
  const requiredCueCount = resolveRequiredGeneratedImageCueCount(state, input.userText, previousSidecar.assistantText, Boolean(input.manualImage));
  const assistantTextRequirement = isImageProgressionCadence(state)
    ? "assistant_text must be one compact Korean status line only; do not narrate the scene in visible prose."
    : `assistant_text must be complete and satisfy the selected output target${minimumChars > 0 ? `, at least about ${minimumChars} Korean characters unless the user explicitly asked for brevity` : ""}.`;
  const imageCueRequirement = isImageProgressionCadence(state)
    ? `image_cues must contain exactly ${IMAGE_PROGRESSION_CUE_TARGET} should_generate=true cuts with concrete final English NovelAI/Danbooru tags. Each cue advances the scene a little from the previous cue.`
    : requiredCueCount > 1
      ? `image_cues must include at least ${requiredCueCount} should_generate=true cue objects with concrete final English NovelAI/Danbooru tags, one anchored cue for each visually distinct dialogue/action/body/state visual beat in assistant_text. A long paragraph can require multiple cues; do not collapse this visual turn into only one or two summary cues.`
      : requiresImageTags
        ? "image_cues must include at least one should_generate=true cue with concrete final English NovelAI/Danbooru tags. Do not use [] and do not leave tags empty for this visual turn."
      : "image_cues may be [] only if this rewritten turn is truly quiet, administrative, or impossible to visualize.";
  const retryScopeInstruction = preserveAssistantText
    ? "Do not rewrite the rejected assistant_text. DynamicChat will keep the already displayed narrative and only use your corrected image_cues and memory_events for this retry. You may omit assistant_text; if you include it, it must exactly match the rejected assistant_text."
    : "Regenerate the whole main response JSON now. This is the simulation response itself, not a separate tag-only request.";
  const fieldOrderInstruction = preserveAssistantText
    ? "Return valid JSON with image_cues before memory_events. assistant_text is optional for this metadata-only retry."
    : "Write JSON fields in this exact order: assistant_text, image_cues, memory_events.";
  const completionInstruction = [
    runtimeInstruction,
    "",
    options.rejectionReason ?? "Previous LLM output was rejected by DynamicChat because the main sidecar was incomplete or malformed for this current turn.",
    issues.length > 0 ? `Detected incomplete fields: ${issues.join("; ")}` : undefined,
    retryScopeInstruction,
    fieldOrderInstruction,
    assistantTextRequirement,
    imageCueRequirement,
    "Put image_cues before memory_events so required image tags are not dropped near the end of the response.",
    "If the visible assistant_text or current user action changes clothing/outfit, memory_events must include memory_kind='state', state_type='Wearing', and state_value as comma-separated English NovelAI outfit tags for the affected character.",
    "If pose/action/status/held item/current visual state changed, include compact state memory_events with StatusTags, PoseTags, ActionTags, InteractionTags, HeldItemTags, PhysicalStateTags, or scene-level tags as appropriate.",
    "Keep memory_events last and compact: use 0-3 concise durable deltas, or [] if the budget is tight. Never sacrifice assistant_text completeness or required image_cues.tags to preserve memory metadata.",
    "Return valid JSON only."
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n");
  const completionContext = [
    contextBlock,
    "",
    "Rejected assistant_text preview:",
    truncatePreview(previousSidecar.assistantText, 1100),
    "",
    "Rejected image_cues summary:",
    summarizeAssistantImageCuesForRetry(previousSidecar)
  ].join("\n\n");
  const rawContent = await requestProviderTextWithRecovery(state, input, completionInstruction, completionContext, outputTokenBudget, {
    allowProviderSafeRecovery: false,
    onRawText: preserveAssistantText ? undefined : onRawText
  });
  if (!rawContent?.trim()) {
    return undefined;
  }

  const parsed = preserveAssistantText
    ? parseAssistantSidecarWithFallbackAssistantText(rawContent, previousSidecar.assistantText)
    : parseAssistantSidecar(rawContent);
  if (!parsed.sidecar || looksLikeProviderBoilerplateText(parsed.sidecar.assistantText)) {
    return undefined;
  }

  if (preserveAssistantText) {
    return {
      rawContent,
      parsed: {
        sidecar: mergeAssistantSidecarRetryMetadata(previousSidecar, parsed.sidecar),
        errors: parsed.errors
      }
    };
  }

  return { rawContent, parsed };
}

function shouldRetryIncompleteAssistantSidecar(
  state: AppState,
  userText: string,
  sidecar: AssistantSidecar,
  outputTokenBudget: number,
  manualImage = false
): boolean {
  return createAssistantSidecarCompletionIssues(state, userText, sidecar, outputTokenBudget, manualImage).length > 0;
}

function hasMalformedAssistantSidecarRecovery(errors: string[]): boolean {
  return errors.some((error) => error === MALFORMED_SIDECAR_RECOVERY_MARKER);
}

function createAssistantSidecarCompletionIssues(
  state: AppState,
  userText: string,
  sidecar: AssistantSidecar,
  outputTokenBudget: number,
  manualImage = false
): string[] {
  const issues: string[] = [];
  if (shouldRepairShortAssistantText(state, userText, sidecar.assistantText, outputTokenBudget)) {
    issues.push("assistant_text ended before the selected length target");
  } else if (shouldRepairAbruptAssistantText(state, userText, sidecar.assistantText, outputTokenBudget)) {
    issues.push("assistant_text appears cut off before a complete handoff");
  }
  if (isImageProgressionCadence(state) && shouldRequireMainImageTags(state, userText, sidecar.assistantText, manualImage)) {
    const generatedCueCount = countGeneratedImageCuesWithConcreteTags(sidecar);
    if (generatedCueCount < IMAGE_PROGRESSION_CUE_TARGET) {
      issues.push(`image_progression expected ${IMAGE_PROGRESSION_CUE_TARGET} generated image_cues with concrete tags, got ${generatedCueCount}`);
    }
  }
  const requiredCueCount = resolveRequiredGeneratedImageCueCount(state, userText, sidecar.assistantText, manualImage);
  if (!isImageProgressionCadence(state) && requiredCueCount > 1) {
    const generatedCueCount = countGeneratedImageCuesWithConcreteTags(sidecar);
    if (generatedCueCount < requiredCueCount) {
      issues.push(`high-density visual beat coverage expected at least ${requiredCueCount} generated image_cues with concrete tags, got ${generatedCueCount}`);
    }
  }
  if (shouldRequireMainImageTags(state, userText, sidecar.assistantText, manualImage) && !hasGeneratedImageCueTags(sidecar)) {
    issues.push("image_cues.tags was empty or too thin for a visual/image-required turn");
  }
  return issues;
}

function shouldPreserveAssistantTextForSidecarRetry(issues: string[]): boolean {
  return issues.length > 0 && !issues.some((issue) => issue.startsWith("assistant_text"));
}

function mergeAssistantSidecarRetryMetadata(base: AssistantSidecar, candidate: AssistantSidecar): AssistantSidecar {
  const baseCues = base.imageCues.length > 0 ? base.imageCues : [base.imageCue];
  const candidateCues = candidate.imageCues.length > 0 ? candidate.imageCues : [candidate.imageCue];
  const imageCues = hasGeneratedImageCueTags(candidate) ? candidateCues : baseCues;
  const imageCue =
    imageCues.find((cue) => cue.shouldGenerate && countConcreteImageTags(getAllImageCueDraftTags(cue)) >= 4) ??
    imageCues.find((cue) => cue.shouldGenerate) ??
    imageCues[0] ??
    base.imageCue;

  return {
    assistantText: base.assistantText,
    memoryEvents: uniqueAssistantMemoryEvents([...base.memoryEvents, ...candidate.memoryEvents]).slice(0, 8),
    imageCue,
    imageCues
  };
}

function hasAssistantSidecarMetadataImprovement(base: AssistantSidecar, candidate: AssistantSidecar): boolean {
  if (uniqueAssistantMemoryEvents(candidate.memoryEvents).length > uniqueAssistantMemoryEvents(base.memoryEvents).length) {
    return true;
  }

  if (countGeneratedImageCuesWithConcreteTags(candidate) > countGeneratedImageCuesWithConcreteTags(base)) {
    return true;
  }

  const baseCueCount = base.imageCues.length > 0 ? base.imageCues.length : 1;
  const candidateCueCount = candidate.imageCues.length > 0 ? candidate.imageCues.length : 1;
  return candidateCueCount > baseCueCount && candidate.imageCues.some((cue) => getAllImageCueDraftTags(cue).length > 0);
}

function uniqueAssistantMemoryEvents(events: AssistantSidecar["memoryEvents"]): AssistantSidecar["memoryEvents"] {
  const seen = new Set<string>();
  return events.filter((event) => {
    const key = [
      event.memoryKind ?? "",
      event.actorId ?? "",
      event.actorName ?? "",
      event.targetId ?? "",
      event.stateType ?? "",
      event.stateValue ?? "",
      event.content
    ]
      .map(normalizeAssistantMemoryEventKeyPart)
      .join("|");
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function normalizeAssistantMemoryEventKeyPart(value: string): string {
  return value.toLowerCase().replace(/\s+/gu, " ").replace(/[^\p{L}\p{N}:,_ -]+/gu, "").trim();
}

function shouldAcceptAssistantSidecarCompletionRetry(
  state: AppState,
  userText: string,
  previousSidecar: AssistantSidecar,
  candidateSidecar: AssistantSidecar,
  outputTokenBudget: number,
  manualImage = false
): boolean {
  if (looksLikeProviderBoilerplateText(candidateSidecar.assistantText)) {
    return false;
  }

  const preservesExistingAssistantText =
    candidateSidecar.assistantText === previousSidecar.assistantText &&
    shouldPreserveAssistantTextForSidecarRetry(
      createAssistantSidecarCompletionIssues(state, userText, previousSidecar, outputTokenBudget, manualImage)
    );
  const previousChars = countVisibleTextChars(previousSidecar.assistantText);
  const candidateChars = countVisibleTextChars(candidateSidecar.assistantText);
  if (!preservesExistingAssistantText && !isImageProgressionCadence(state) && candidateChars < Math.max(180, Math.floor(previousChars * 0.7))) {
    return false;
  }

  const previousWasShort = shouldRepairShortAssistantText(state, userText, previousSidecar.assistantText, outputTokenBudget);
  const candidateIsShort = shouldRepairShortAssistantText(state, userText, candidateSidecar.assistantText, outputTokenBudget);
  if (!preservesExistingAssistantText && previousWasShort && candidateIsShort && candidateChars <= previousChars + 120) {
    return false;
  }

  if (shouldRequireMainImageTags(state, userText, previousSidecar.assistantText, manualImage) && !hasGeneratedImageCueTags(candidateSidecar)) {
    return false;
  }
  const requiredCueCount = resolveRequiredGeneratedImageCueCount(state, userText, previousSidecar.assistantText, manualImage);
  if (requiredCueCount > 1 && countGeneratedImageCuesWithConcreteTags(candidateSidecar) < requiredCueCount) {
    return false;
  }
  if (
    isImageProgressionCadence(state) &&
    shouldRequireMainImageTags(state, userText, previousSidecar.assistantText, manualImage) &&
    countGeneratedImageCuesWithConcreteTags(candidateSidecar) < IMAGE_PROGRESSION_CUE_TARGET
  ) {
    return false;
  }

  return true;
}

function createAssistantSidecarCompletionError(
  state: AppState,
  userText: string,
  sidecar: AssistantSidecar,
  outputTokenBudget: number,
  manualImage = false
): string {
  const issues = createAssistantSidecarCompletionIssues(state, userText, sidecar, outputTokenBudget, manualImage);
  return issues.length > 0
    ? `main LLM sidecar remained incomplete after completion retry: ${issues.join("; ")}.`
    : "main LLM sidecar completion retry did not improve the current turn.";
}

function shouldRequireMainImageTags(state: AppState, userText: string, assistantText: string, manualImage = false): boolean {
  if (isImageProgressionCadence(state)) {
    return (
      state.imageProfile.enabled &&
      state.simulation.realtimeImageEnabled &&
      state.imageProfile.triggerMode !== "stored_only" &&
      state.imageProfile.triggerMode !== "manual" &&
      Boolean(assistantText.trim()) &&
      !looksLikeProviderBoilerplateText(assistantText)
    );
  }

  if (!hasRenderableAssistantText(assistantText)) {
    return false;
  }

  if (
    manualImage &&
    state.imageProfile.enabled &&
    state.simulation.realtimeImageEnabled &&
    state.imageProfile.triggerMode !== "stored_only"
  ) {
    return true;
  }

  const imageUserRules = createImageUserRulesForContentRating(state);
  const userRulePlan = analyzeImageCueUserRulePolicy(imageUserRules);
  if (userRulePlan.requiresGeneration) {
    return true;
  }

  if (
    !state.imageProfile.enabled ||
    !state.simulation.realtimeImageEnabled ||
    state.imageProfile.triggerMode === "stored_only" ||
    state.imageProfile.triggerMode === "manual"
  ) {
    return false;
  }

  const cadence = state.imageProfile.generationCadence ?? "balanced";
  if (cadence === "paragraph") {
    return true;
  }
  if (cadence === "rich" || cadence === "balanced") {
    return hasVisualCueText(`${userText}\n${assistantText}`);
  }
  if (cadence === "sparse") {
    return hasSparseMajorVisualCueText(`${userText}\n${assistantText}`);
  }

  return false;
}

function hasRenderableAssistantText(assistantText: string): boolean {
  const text = assistantText.trim();
  if (text.length < 24 || looksLikeProviderBoilerplateText(text)) {
    return false;
  }

  return !/(?:LLM fallback|fallback|설정과 모델 응답 상태|HTTP\s*400|API\s*요청|다시 입력하면 실제 LLM 흐름|모델 응답|오류가 발생|응답을 생성하지 못)/iu.test(text);
}

function hasGeneratedImageCueTags(sidecar: AssistantSidecar): boolean {
  return countGeneratedImageCuesWithConcreteTags(sidecar) > 0;
}

function countGeneratedImageCuesWithConcreteTags(sidecar: AssistantSidecar): number {
  const cues = sidecar.imageCues.length > 0 ? sidecar.imageCues : [sidecar.imageCue];
  return cues.filter((cue) => cue.shouldGenerate !== false && countConcreteImageTags(getAllImageCueDraftTags(cue)) >= 4).length;
}

function getAllImageCueDraftTags(cue: AssistantImageCueDraft): string[] {
  return [
    ...cue.tags,
    ...(cue.baseTags ?? []),
    ...(cue.characterPrompts ?? []).flatMap((prompt) => prompt.prompt.split(","))
  ];
}

function countConcreteImageTags(tags: string[]): number {
  return tags.filter((tag) => {
    const normalized = tag.trim().toLowerCase();
    return (
      normalized.length > 0 &&
      !/^(?:scene|current scene|generated scene|image|prompt|tag|visual|character|background|foreground)$/iu.test(normalized)
    );
  }).length;
}

function resolveRequiredGeneratedImageCueCount(
  state: AppState,
  userText: string,
  assistantText: string,
  manualImage = false
): number {
  if (!shouldRequireMainImageTags(state, userText, assistantText, manualImage)) {
    return 0;
  }
  if (isImageProgressionCadence(state)) {
    return IMAGE_PROGRESSION_CUE_TARGET;
  }
  if (state.imageProfile.generationCadence === "paragraph") {
    return Math.max(1, countRequiredVisualCueBeats(assistantText));
  }

  return 1;
}

function countRequiredVisualCueBeats(assistantText: string): number {
  const blocks = selectVisualAssistantTextBlocks(assistantText);
  if (blocks.length === 0) {
    return 0;
  }

  const beatFragments = uniqueStrings(
    [
      ...blocks,
      ...selectVisualCueBeatLines(assistantText),
      ...blocks.flatMap(splitVisualCueBeatFragments),
      ...extractDialogueCueBeatFragments(assistantText)
    ]
      .map(cleanImageCueAnchorText)
      .filter(isVisualCueBeatFragment)
      .map((block) => block.slice(0, 140))
  );
  const lengthBasedBeats = blocks.reduce((sum, block) => sum + Math.max(1, Math.ceil(block.length / 220)), 0);

  return Math.min(8, Math.max(blocks.length, beatFragments.length, lengthBasedBeats));
}

function selectVisualAssistantTextBlocks(assistantText: string): string[] {
  const trimmed = assistantText.trim();
  if (!trimmed) {
    return [];
  }

  const paragraphBlocks = trimmed
    .split(/\n{2,}/u)
    .map(cleanImageCueAnchorText)
    .filter(isVisualAssistantTextBlock);
  const blocks =
    paragraphBlocks.length > 1
      ? paragraphBlocks
      : trimmed
          .split(/\n+/u)
          .map(cleanImageCueAnchorText)
          .filter(isVisualAssistantTextBlock);

  return uniqueStrings(blocks);
}

function selectVisualCueBeatLines(assistantText: string): string[] {
  return assistantText
    .split(/\n+/u)
    .map(cleanImageCueAnchorText)
    .filter(isVisualCueBeatFragment);
}

function splitVisualCueBeatFragments(block: string): string[] {
  const normalized = cleanImageCueAnchorText(block)
    .replace(/([.!?。！？…]+)(\s*)/gu, "$1\n")
    .replace(/(["”」』])(\s*)/gu, "$1\n")
    .replace(/(\s+)(?=["“「『])/gu, "\n");
  const fragments = normalized
    .split(/\n+/u)
    .map(cleanImageCueAnchorText)
    .filter(isVisualCueBeatFragment);
  if (fragments.length > 1 || normalized.length <= 260) {
    return fragments;
  }

  return normalized.match(/.{1,220}/gu)?.map(cleanImageCueAnchorText).filter(isVisualCueBeatFragment) ?? fragments;
}

function extractDialogueCueBeatFragments(assistantText: string): string[] {
  const quoted = assistantText.match(/["“「『][^"”」』]{2,180}["”」』]/gu) ?? [];
  const speakerLines = assistantText.match(/^\s*\S.{0,20}[:：].{2,180}$/gmu) ?? [];
  return [...quoted, ...speakerLines].map(cleanImageCueAnchorText).filter(isVisualCueBeatFragment);
}

function cleanImageCueAnchorText(value: string): string {
  return value
    .replace(/```[a-zA-Z0-9_-]*\n?/gu, "")
    .replace(/```/gu, "")
    .replace(/^[:>#*\-\s]+/gmu, "")
    .replace(/\s+/gu, " ")
    .trim();
}

function isVisualAssistantTextBlock(value: string): boolean {
  if (value.length < 8) {
    return false;
  }

  return !/^(?:status|choice|memory|system|sfx|impact)\b|^::/iu.test(value);
}

function isVisualCueBeatFragment(value: string): boolean {
  return isVisualAssistantTextBlock(value) && hasDetailedVisualBeatText(value);
}

function hasDetailedVisualBeatText(text: string): boolean {
  return (
    hasVisualCueText(text) ||
    /표정|얼굴|눈|입|입술|고개|시선|어깨|팔|손|손가락|상체|허리|다리|무릎|발|자세|포즈|옷|의상|복장|교복|제복|드레스|셔츠|치마|바지|코트|젖|찢|흐트러|붉|떨|흔들|앉|서|눕|기어|바라|웃|울|찡그|숨|호흡|다가|물러|가까|멀어|문|벽|바닥|침대|소파|책상|의자|조명|카메라|시점|클로즈|배경|expression|face|eyes?|mouth|lips?|gaze|shoulder|arms?|hands?|fingers?|waist|legs?|knees?|feet|pose|outfit|clothes|uniform|dress|shirt|skirt|pants|coat|wet|torn|standing|sitting|lying|kneeling|crawling|looking|smiling|crying|breath|close-up|camera|background|lighting|bed|floor|chair|door|wall/iu.test(
      text
    )
  );
}

function summarizeAssistantImageCuesForRetry(sidecar: AssistantSidecar): string {
  const cues = sidecar.imageCues.length > 0 ? sidecar.imageCues : [sidecar.imageCue];
  if (cues.length === 0) {
    return "(none)";
  }

  return cues
    .slice(0, 8)
    .map((cue, index) => {
      const tags = cue.tags.length > 0 ? cue.tags.join(", ") : "(empty)";
      const cueKind = cue.kind ?? cue.cueType;
      return [
        `cue ${index + 1}`,
        `should_generate=${cue.shouldGenerate}`,
        cueKind ? `kind=${cueKind}` : undefined,
        cue.anchorText ? `anchor=${truncatePreview(cue.anchorText, 120)}` : undefined,
        `tags=${truncatePreview(tags, 320)}`
      ]
        .filter((item): item is string => Boolean(item))
        .join(" | ");
    })
    .join("\n");
}

function hasVisualCueText(text: string): boolean {
  return IMAGE_PATTERN.test(text) || hasActionBeatText(text) || hasDialogueText(text);
}

function hasSparseMajorVisualCueText(text: string): boolean {
  return /새로운\s*(?:장소|인물|복장|의상)|장소가\s*바뀌|무대|전투|위험|불빛|등장|문을\s*열|뛰|쓰러|피하|new\s+(?:location|character|outfit)|scene\s+change|battle|chase|explosion/iu.test(text);
}

function hasDialogueText(text: string): boolean {
  return /["“”「」『』]|^\s*\S.{0,20}[:：]|대사|말하|속삭|외치|신음|숨소리|voice|says?|said|whisper|moan/imu.test(text);
}

function hasActionBeatText(text: string): boolean {
  return /움직|손|시선|몸|걸음|다가|멈추|잡|놓|돌아|밀|당기|뻗|올리|숙이|기대|행동|동작|전투|action|gesture|looks?|steps?|reaches?|turns?|grabs?|holds?|leans?/iu.test(text);
}

async function requestShortAssistantTextRepair(
  state: AppState,
  input: {
    userText: string;
    modules: PromptModule[];
    evidence: ContextEvidence[];
  },
  runtimeInstruction: string,
  contextBlock: string,
  outputTokenBudget: number,
  previousAssistantText: string,
  onRawText?: (rawText: string) => void
): Promise<{ rawContent: string; parsed: ReturnType<typeof parseAssistantSidecar> } | undefined> {
  const minimumChars = resolveMinimumAssistantTextChars(outputTokenBudget);
  const repairInstruction = [
    runtimeInstruction,
    "",
    "Previous valid JSON was rejected by DynamicChat because assistant_text was far shorter than the selected output target.",
    `Regenerate the whole JSON response now. assistant_text must be at least about ${minimumChars} Korean characters unless the user explicitly asks for a brief reply.`,
    "The visible assistant_text is the product. Spend nearly all of the response budget there before writing metadata.",
    "Write JSON fields in this exact order: assistant_text, image_cues, memory_events.",
    "Keep sidecar metadata compact for this repair: image_cues should be 0-1 cue unless the user's image rules explicitly require more, and memory_events should be 0-3 concise deltas.",
    "Do not close assistant_text after only an opening beat. Continue the scene with the full target amount of narrative work while preserving required memory_events and image_cues."
  ].join("\n");
  const repairContext = [
    contextBlock,
    "",
    "Rejected short assistant_text preview:",
    truncatePreview(previousAssistantText, 700)
  ].join("\n\n");
  const rawContent = await requestProviderTextWithRecovery(state, input, repairInstruction, repairContext, outputTokenBudget, {
    allowProviderSafeRecovery: false,
    onRawText
  });
  if (!rawContent?.trim()) {
    return undefined;
  }

  const parsed = parseAssistantSidecar(rawContent);
  const repairedText = parsed.sidecar?.assistantText ?? "";
  if (
    !parsed.sidecar ||
    looksLikeProviderBoilerplateText(repairedText) ||
    repairedText.trim().length <= previousAssistantText.trim().length
  ) {
    return undefined;
  }

  return { rawContent, parsed };
}

async function requestShortAssistantTextContinuation(
  state: AppState,
  input: {
    userText: string;
    modules: PromptModule[];
    evidence: ContextEvidence[];
  },
  runtimeInstruction: string,
  contextBlock: string,
  outputTokenBudget: number,
  previousAssistantText: string,
  onRawText?: (rawText: string) => void
): Promise<{ rawContent: string; parsed: ReturnType<typeof parseAssistantSidecar> } | undefined> {
  const minimumChars = resolveMinimumAssistantTextChars(outputTokenBudget);
  const previousChars = countVisibleTextChars(previousAssistantText);
  const continuationMinChars = Math.max(SHORT_OUTPUT_CONTINUATION_MIN_CHARS, minimumChars - previousChars);
  const continuationInstruction = [
    runtimeInstruction,
    "",
    "The previous valid JSON ended assistant_text too early, while the current turn still needs more visible narrative.",
    "Write a continuation for the same current turn from the exact last sentence of the already displayed assistant_text.",
    "assistant_text must contain only the additional continuation paragraphs. Do not repeat or summarize the already displayed text.",
    `Write at least about ${continuationMinChars} additional Korean characters unless the current scene naturally reaches a clear handoff earlier.`,
    "Write JSON fields in this exact order: assistant_text, image_cues, memory_events.",
    "Keep sidecar metadata compact for this continuation: image_cues should be 0-1 cue unless the user's image rules explicitly require more, and memory_events should be 0-3 concise deltas.",
    "Return valid JSON only."
  ].join("\n");
  const continuationContext = [
    contextBlock,
    "",
    "Already displayed assistant_text opening. Continue after this text without repeating it:",
    truncatePreview(previousAssistantText, 1600)
  ].join("\n\n");
  const continuationInput = {
    userText: "Continue the same assistant response from the already displayed ending.",
    modules: input.modules,
    evidence: input.evidence
  };
  const rawContent = await requestProviderTextWithRecovery(
    state,
    continuationInput,
    continuationInstruction,
    continuationContext,
    outputTokenBudget,
    {
      allowProviderSafeRecovery: false,
      onRawText: createContinuationRawTextEmitter(previousAssistantText, onRawText)
    }
  );
  if (!rawContent?.trim()) {
    return undefined;
  }

  const parsed = parseAssistantSidecar(rawContent);
  const continuationText = parsed.sidecar?.assistantText ?? "";
  const combinedText = joinAssistantContinuationText(previousAssistantText, continuationText);
  if (!parsed.sidecar || countVisibleTextChars(combinedText) <= previousChars + 80) {
    return undefined;
  }

  return { rawContent, parsed };
}

function createContinuationRawTextEmitter(
  previousAssistantText: string,
  onRawText: ((rawText: string) => void) | undefined
): ((rawText: string) => void) | undefined {
  if (!onRawText) {
    return undefined;
  }

  return (rawText) => {
    const continuationText = createDisplayFallbackText(rawText, "");
    if (!continuationText) {
      return;
    }

    onRawText(JSON.stringify({
      assistant_text: joinAssistantContinuationText(previousAssistantText, continuationText)
    }));
  };
}

function appendAssistantContinuationSidecar(base: AssistantSidecar, continuation: AssistantSidecar): AssistantSidecar {
  const imageCues = uniqueAssistantImageCues([...base.imageCues, ...continuation.imageCues]).slice(0, 8);
  const imageCue =
    imageCues.find((cue) => cue.shouldGenerate && cue.tags.length > 0) ??
    imageCues.find((cue) => cue.shouldGenerate) ??
    imageCues[0] ??
    base.imageCue;

  return {
    assistantText: joinAssistantContinuationText(base.assistantText, continuation.assistantText),
    memoryEvents: [...base.memoryEvents, ...continuation.memoryEvents].slice(0, 8),
    imageCue,
    imageCues: imageCues.length > 0 ? imageCues : [imageCue]
  };
}

function uniqueAssistantImageCues(cues: AssistantImageCueDraft[]): AssistantImageCueDraft[] {
  const seen = new Set<string>();
  return cues.filter((cue) => {
    const key = [
      cue.kind ?? cue.cueType ?? "",
      cue.placement ?? "",
      cue.anchorText ?? "",
      cue.tags.join(","),
      cue.scene
    ].join("|");
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function joinAssistantContinuationText(previousAssistantText: string, continuationText: string): string {
  const previous = previousAssistantText.trimEnd();
  const continuation = removeRepeatedContinuationPrefix(previous, continuationText.trimStart());
  if (!previous) {
    return continuation;
  }
  if (!continuation) {
    return previous;
  }

  return `${previous}\n\n${continuation}`;
}

function removeRepeatedContinuationPrefix(previousAssistantText: string, continuationText: string): string {
  const previous = previousAssistantText.trim();
  let continuation = continuationText.trim();
  if (!previous || !continuation) {
    return continuation;
  }

  if (continuation.startsWith(previous)) {
    return continuation.slice(previous.length).trim();
  }

  const maxOverlap = Math.min(previous.length, continuation.length, 600);
  for (let length = maxOverlap; length >= 80; length -= 20) {
    const previousTail = previous.slice(-length);
    if (continuation.startsWith(previousTail)) {
      continuation = continuation.slice(previousTail.length).trim();
      break;
    }
  }

  return continuation;
}

function createImageCueVisualProfileBlock(state: AppState, currentUserText = ""): string {
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state, currentUserText));
  return state.characters
    .filter((character) => shouldIncludeCharacterScopedContext(state, character.id, activeCharacterIds))
    .map((character) => {
      const visualProfile = state.visualProfiles.find((profile) => profile.characterId === character.id);
      if (!visualProfile) {
        return undefined;
      }
      const outfitMappings = formatOutfitMappingsForFoundation(visualProfile.outfitPrompts);
      return [
        `- character_id: ${character.id}`,
        `name: ${character.name}`,
        visualProfile.positivePrompt ? `required_identity_tags: ${truncatePromptText(visualProfile.positivePrompt, 420, "visual profile")}` : undefined,
        visualProfile.positivePrompt ? `visual_profile_tags: ${truncatePromptText(visualProfile.positivePrompt, 420, "visual profile")}` : undefined,
        visualProfile.defaultOutfitPrompt ? `default_outfit_tags: ${visualProfile.defaultOutfitPrompt}` : undefined,
        outfitMappings ? `outfit_keyword_mappings: ${outfitMappings}` : undefined,
        "identity_rule: if this character appears in image_cues.characters, include required_identity_tags plus the visible outfit/state tags directly in image_cues.tags"
      ]
        .filter((item): item is string => Boolean(item))
        .join(" | ");
    })
    .filter((item): item is string => Boolean(item))
    .join("\n");
}

function createImageCueCurrentStateBlock(state: AppState, currentUserText = ""): string {
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state, currentUserText));
  return state.memoryEvents
    .filter(isCurrentStateMemoryEvent)
    .filter((event) => {
      const ownerId = readStateMemoryOwnerId(event);
      return !ownerId || shouldIncludeCharacterScopedContext(state, ownerId, activeCharacterIds);
    })
    .slice(-12)
    .map((event) => {
      const ownerId = readStateMemoryOwnerId(event);
      const owner = ownerId ? `character_id: ${ownerId}` : "scene";
      const stateType = readStateMemoryStateType(event);
      const value = readStateMemoryValue(event);
      return `- ${owner}${stateType ? ` | state_type: ${stateType}` : ""}${value ? ` | value: ${truncatePromptText(value, 180, "state value")}` : ""} | note: ${truncatePromptText(event.content, 260, "state note")}`;
    })
    .filter((item): item is string => Boolean(item))
    .join("\n");
}

function createImageSceneTagPresetBlock(state: AppState, currentUserText = ""): string {
  const contextText = [
    currentUserText,
    state.simulation.title,
    state.simulation.description,
    ...state.messages.slice(-4).map((message) => message.content),
    ...state.memoryEvents.slice(-8).map((event) => event.content)
  ]
    .join("\n")
    .toLowerCase();

  const counter = { count: 0 };
  const scoredPresets = collectImageSceneTagPresetNodes(state.imageScenePresets ?? [], contextText, [], [], [], 0, 0, counter)
    .filter((preset) => preset.branchScore > 0);
  if (scoredPresets.length === 0) {
    return "";
  }

  return scoredPresets
    .sort(
      (a, b) =>
        Number(b.branchScore > 0) - Number(a.branchScore > 0) ||
        b.branchScore - a.branchScore ||
        b.node.priority - a.node.priority ||
        a.depth - b.depth ||
        a.order - b.order
    )
    .slice(0, IMAGE_SCENE_PRESET_PROMPT_LIMIT)
    .map(({ node, path, inheritedTags, inheritedNotes, branchScore, depth }) => {
      const sceneTags = uniqueScenePresetTags(node.tags.filter((tag) => !looksLikeCharacterScenePresetTag(tag))).slice(0, 24);
      const parentTags = uniqueScenePresetTags(inheritedTags.filter((tag) => !looksLikeCharacterScenePresetTag(tag))).slice(0, 12);
      const parentNotes = uniqueScenePresetNotes(inheritedNotes).slice(-3);
      return [
        `- path: ${formatImageScenePresetPath(path)}`,
        `depth: ${depth + 1}`,
        `priority: ${node.priority}`,
        branchScore > 0 ? `matched_branch_score: ${branchScore}` : undefined,
        parentTags.length > 0 ? `inherited_base_tags: ${parentTags.join(", ")}` : undefined,
        sceneTags.length > 0 ? `base_scene_tags: ${sceneTags.join(", ")}` : undefined,
        parentNotes.length > 0 ? `inherited_creator_notes: ${parentNotes.map((note) => truncatePromptText(note, 120, "parent scene preset note")).join(" / ")}` : undefined,
        node.note.trim() ? `creator_note: ${truncatePromptText(node.note, 260, "scene preset note")}` : undefined,
        "rule: adapt this branch only if relevant. creator_note and inherited_creator_notes are author guidance, recommended variants, or wildcard usage hints. Scene/composition/shared-action/environment tags go to image_cues.base_tags; expression, exact pose, clothing, body-state, and identity-like tags go to the appropriate image_cues.character_prompts item."
      ]
        .filter((item): item is string => Boolean(item))
        .join(" | ");
    })
    .join("\n");
}

interface ScoredImageSceneTagPresetNode {
  node: ImageSceneTagPresetNode;
  path: string[];
  inheritedTags: string[];
  inheritedNotes: string[];
  branchScore: number;
  depth: number;
  order: number;
}

function collectImageSceneTagPresetNodes(
  nodes: ImageSceneTagPresetNode[],
  contextText: string,
  parentPath: string[],
  inheritedTags: string[],
  inheritedNotes: string[],
  depth: number,
  inheritedMatchScore: number,
  counter: { count: number }
): ScoredImageSceneTagPresetNode[] {
  if (counter.count >= IMAGE_SCENE_PRESET_SEARCH_NODE_LIMIT) {
    return [];
  }

  return nodes.flatMap((node, index) => {
    if (counter.count >= IMAGE_SCENE_PRESET_SEARCH_NODE_LIMIT || node.enabled === false) {
      return [];
    }

    counter.count += 1;
    const keyword = node.keyword.trim() || `scene-${index + 1}`;
    const path = [...parentPath, keyword];
    const ownScore = scoreImageScenePresetNode(node, contextText);
    const branchScore = Math.max(ownScore, inheritedMatchScore > 0 ? Math.max(1, inheritedMatchScore - 1) : 0);
    const currentTags = uniqueScenePresetTags(node.tags);
    const nextInheritedTags = uniqueScenePresetTags([...inheritedTags, ...currentTags]).slice(0, 32);
    const currentNote = node.note.trim();
    const nextInheritedNotes = uniqueScenePresetNotes([...inheritedNotes, currentNote]).slice(-6);
    const current: ScoredImageSceneTagPresetNode[] =
      keyword || currentTags.length > 0 || currentNote
        ? [
            {
              node,
              path,
              inheritedTags,
              inheritedNotes,
              branchScore,
              depth,
              order: counter.count
            }
          ]
        : [];
    const children =
      depth >= 8
        ? []
        : collectImageSceneTagPresetNodes(node.children ?? [], contextText, path, nextInheritedTags, nextInheritedNotes, depth + 1, branchScore, counter);
    return [...current, ...children];
  });
}

function scoreImageScenePresetNode(node: ImageSceneTagPresetNode, contextText: string): number {
  const keyword = node.keyword.trim().toLowerCase();
  const terms = new Set(
    [
      keyword,
      ...keyword.split(/[\s,;/|>]+/u),
      ...node.tags,
      ...node.note.split(/[\s,;/|>()[\]{}]+/u)
    ]
      .map((term) => term.trim().toLowerCase())
      .filter(isUsefulImageScenePresetSearchTerm)
  );

  let score = 0;
  for (const term of terms) {
    if (contextText.includes(term)) {
      score += term === keyword ? 4 : 1;
    }
  }
  return score;
}

function isUsefulImageScenePresetSearchTerm(term: string): boolean {
  return term.length >= 3 || /\p{Script=Hangul}/u.test(term);
}

function formatImageScenePresetPath(path: string[]): string {
  const visiblePath = path.slice(-IMAGE_SCENE_PRESET_PROMPT_MAX_DEPTH);
  return `${path.length > visiblePath.length ? "... > " : ""}${visiblePath.join(" > ")}`;
}

function uniqueScenePresetTags(tags: string[]): string[] {
  const seen = new Set<string>();
  return tags
    .map((tag) => tag.trim())
    .filter((tag) => {
      const key = tag.toLowerCase();
      if (!tag || seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

function uniqueScenePresetNotes(notes: string[]): string[] {
  const seen = new Set<string>();
  return notes
    .map((note) => note.trim())
    .filter((note) => {
      const key = note.toLowerCase();
      if (!note || seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

function looksLikeCharacterScenePresetTag(tag: string): boolean {
  return /\b(?:[1-6](?:girl|girls|boy|boys|other)|girl|girls|boy|boys|woman|women|man|men|female|male|hair|haircut|hairclip|eye\s*color|uniform|dress|skirt|shirt|coat|jacket|pants|shorts|stockings|shoes|boots|underwear|bra|panties)\b/iu.test(tag);
}

function isCurrentStateMemoryEvent(event: AppState["memoryEvents"][number]): boolean {
  if (readStateMemoryKind(event) === "state") {
    return true;
  }
  return Boolean(readStateMemoryStateType(event));
}

function shouldIncludeCharacterScopedContext(
  state: AppState,
  characterId: string,
  activeCharacterIds: ReadonlySet<string>
): boolean {
  return activeCharacterIds.has(characterId) || (state.characters.length === 1 && state.characters[0]?.id === characterId);
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

function createProviderBlockedFallbackContent(state: AppState, userText: string): string {
  const actionLine = sanitizeProviderRetryText(userText, 180) || "계속 진행";

  return [
    "::status[LLM fallback: 제공자가 현재 턴의 본문을 반환하지 않아 장면을 대체 작성하지 않았습니다. 적용 대상 모듈: provider-blocked]",
    `현재 입력 "${actionLine}"은 유지되었지만, 모델이 사용할 수 있는 본문을 반환하지 않아 이번 턴은 진행되지 않았습니다.`,
    "DynamicChat은 차단된 턴을 안전한 다른 장면으로 로컬 변환하지 않습니다. 모델, 콘텐츠 등급, 제공자 설정을 조정한 뒤 다시 생성하세요."
  ].join("\n\n");
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
  const redacted = withoutBoilerplate
  .replace(
    /\b(?:sex|sexual|explicit|nude|nudity|porn|rape|incest|underage|minor|self-harm|suicide)\b/giu,
    "[sensitive]"
  )
  .replace(
    /(?:성행위|성관계|성폭행|성폭력|강간|노골적|나체|누드|포르노|미성년|자해|자살|살해|고문|시체|유혈|납치|감금|협박|폭행|성욕|정액|애액|사정|절정|신음|교성|보지|자지|음경|질벽|질내|질\s*내부|자궁|하반신|가슴|유두|삽입|박히|박아|박혀|쑤시|쾌락|오르가즘|오줌|대소변|내장|살점|찢겨|뜯겨|절단|도끼|복부|목구멍|욕지거리|씨발|년아|시체|피비린내|피와|피를)/gu,
    "[sensitive]"
  );
  const normalized = redacted.replace(/\s+/gu, " ").trim();
  return normalized.length > maxChars ? `${normalized.slice(0, maxChars)}...` : normalized;
}

function createContextBlock(
  state: AppState,
  currentUserText: string,
  modules: PromptModule[],
  evidence: ContextEvidence[],
  options: { manualImage?: boolean; outputTokenBudget?: number } = {}
): string {
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
  const imageScenePresetText = createImageSceneTagPresetBlock(state, currentUserText);
  const imageCadenceText = createImageGenerationCadenceBlock(state, options.outputTokenBudget);
  const currentTurnImagePolicyText = createCurrentTurnImagePolicyBlock(state, options);
  const imageVisualProfileText = createImageCueVisualProfileBlock(state, currentUserText);
  const imageCurrentStateText = createImageCueCurrentStateBlock(state, currentUserText);
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
      assistant_text: isImageProgressionCadence(state)
        ? "one compact Korean status line shown to the user; the scene progresses through image_cues only"
        : "natural Korean response shown to the user",
      image_cues: [
        {
          label: "optional image cut label",
          kind: "scene | action | body_detail | dialogue_face | context | interaction",
          placement: "before | after | inline",
          anchor_text: "exact nearby assistant_text fragment used for placement",
          priority: 0.92,
          should_generate: true,
          reason: "why this exact cut should be generated now",
          suppression_reason: "why generation should be skipped, if any",
          characters: ["visible registered character ids"],
          tags: ["legacy flat final English NovelAI tags; prefer base_tags plus character_prompts"],
          base_tags: ["base prompt tags only: artist-free scene, camera, composition, location, props, action shared by the cut"],
          character_prompts: [
            {
              character_id: "registered visible character id, or omit for an unregistered single visible subject",
              prompt: "complete character prompt tags for this character: identity, expression, pose, outfit, held/body state",
              negative_prompt: "optional character-specific negative tags",
              center: { x: 0.5, y: 0.5 }
            }
          ],
          scene: "short visual scene label",
          visual_context: "comma-separated final NovelAI tags or short tag phrases"
        }
      ],
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
      ]
    }),
    "Current turn image generation policy:",
    currentTurnImagePolicyText,
    "Write JSON fields in this exact order: assistant_text, image_cues, memory_events. Put image_cues before memory_events so required image tags are not dropped near the end.",
    isImageProgressionCadence(state)
      ? `Image progression mode: keep assistant_text to one short Korean status line and use image_cues as the actual scene progression. Output exactly ${IMAGE_PROGRESSION_CUE_TARGET} should_generate=true cue objects when realtime image generation is active. Return at most 3 memory_events.`
      : "Match assistant_text to the runtime output length target and keep it complete: close the JSON object every time. Return at most 8 memory_events. You are the only image cue/tag author in this turn; DynamicChat will not run a later tag planner. For image_cues, emit [] only for quiet text-only turns; when an image should be generated, write the final usable NovelAI tags yourself.",
    "Memory compiler rules: memory_events are structured simulation deltas only. Do not store the full assistant_text, style prose, atmosphere, repeated facts, or facts already present in Structured simulation memory.",
    "Separate actual events from current states. Every memory_events item must include a concise content string; if you cannot write one, omit that memory event. If a current state changes, output memory_kind='state' with state_type and state_value. If someone saw/heard/learned something, output observation or belief for that character only. Keep uncertain causes as belief/open_thread, not confirmed fact.",
    "Memory graph role rules: split one visible beat into small deltas when needed: event for what happened, state for the affected character or scene, relationship for a relationship change, observation/belief for character-specific knowledge. Set actor_id to the acting or affected registered character, target_id to the relationship/observed target when known, and observers to registered characters who actually perceived it. target_id may refer to an off-stage relationship target, but that does not make the target present in assistant_text or image_cues.",
    createImageCueTagContractInstruction(state),
    "Outfit and image-state continuity rules: character visual profiles may define default outfit tags and keyword outfit mappings. If image_cues should generate an image, include the relevant visual profile, outfit, current state, pose, action, interaction, expression, prop, scene, camera, and lighting tags directly in image_cues.tags. Also store changed durable visual states in memory_events with memory_kind='state': use state_type='Wearing', 'StatusTags', 'PoseTags', 'ActionTags', 'InteractionTags', 'InteractionPhaseTags', 'HeldItemTags', 'PhysicalStateTags', 'SceneTags', 'ScenePhaseTags', 'CompositionTags', 'CameraTags', or 'LightingTags'. state_value must be comma-separated English NovelAI tags, preferably 3-8 compact tags. Wearing state_value must preserve the base outfit identity and garment details when the outfit is damaged, loosened, wet, dirty, or otherwise modified; write `police uniform, navy short dress, mini skirt, torn uniform` rather than only `torn uniform`. Set actor_id for character-specific state. Leave actor_id empty only for whole-scene state.",
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
    "Image tag keyword presets (no roster identity tags):",
    imageScenePresetText || "(none)",
    "Image generation cadence:",
    imageCadenceText,
    "Image cue authoring reference for current-scene visible characters only. Use these directly when writing image_cues.character_prompts, with only scene/composition/shared-action tags in image_cues.base_tags. local code will not append registered character prompt tags from character ids. Character identity lock: if a cue lists a character_id, copy that character's required_identity_tags and current visible outfit/state tags into that character_prompt item and do not borrow appearance tags from any other roster character:",
    imageVisualProfileText || "(none)",
    "Current image/visual state records for current-scene characters only:",
    imageCurrentStateText || "(none)",
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
    "Outfit/status persistence: if clothing changes or the scene establishes a new outfit, also emit memory_kind='state', state_type='Wearing', state_value as English NovelAI outfit tags. Preserve existing base outfit details when only condition/damage changes; append tags such as torn uniform, wet clothes, or dirty skirt instead of replacing `police uniform, navy short dress, mini skirt` with a generic label. If a character gains important visual state tags, emit state_type='StatusTags'. These records feed the relationship tab and later image prompts."
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

function resolveInteractiveOutputTokenBudget(state: AppState): number {
  const requested = Number.isFinite(state.llm.maxTokens) ? state.llm.maxTokens : 1600;
  const minimum = isImageProgressionCadence(state) ? IMAGE_PROGRESSION_MIN_OUTPUT_TOKENS : MIN_OUTPUT_TOKENS;
  return Math.min(INTERACTIVE_OUTPUT_TOKEN_CAP, Math.max(minimum, requested));
}

function resolveProviderOutputTokenBudget(state: AppState, outputTokenBudget: number): number {
  const sidecarOverhead = resolveStructuredSidecarOutputTokenOverhead(state, outputTokenBudget);
  return Math.min(INTERACTIVE_OUTPUT_TOKEN_CAP, Math.max(outputTokenBudget, outputTokenBudget + sidecarOverhead));
}

function resolveStructuredSidecarOutputTokenOverhead(state: AppState, outputTokenBudget: number): number {
  if (isImageProgressionCadence(state)) {
    return 0;
  }

  const realtimeMode =
    state.imageProfile.enabled &&
    state.simulation.realtimeImageEnabled &&
    state.imageProfile.triggerMode !== "stored_only";
  const userRulePlan = analyzeImageCueUserRulePolicy(createImageUserRulesForContentRating(state));
  const memoryOverhead = outputTokenBudget <= 1000 ? 160 : 320;
  if (!realtimeMode && !userRulePlan.requiresGeneration) {
    return memoryOverhead;
  }

  const cadence = state.imageProfile.generationCadence ?? "balanced";
  let imageCueOverhead = 0;
  if (cadence === "paragraph") {
    imageCueOverhead = resolveInitialParagraphImageCueTarget(outputTokenBudget).max * 220;
  } else if (cadence === "rich") {
    imageCueOverhead = 900;
  } else if (cadence === "balanced") {
    imageCueOverhead = 520;
  } else if (cadence === "sparse") {
    imageCueOverhead = 260;
  }

  if (userRulePlan.requiresGeneration) {
    const ruleDenseOverhead = userRulePlan.requirements.some((requirement) =>
      /action_beat|body_detail|dialogue_face|whole_scene/u.test(requirement)
    )
      ? 1200
      : 760;
    imageCueOverhead = Math.max(imageCueOverhead, ruleDenseOverhead);
  }

  return Math.min(2600, memoryOverhead + imageCueOverhead);
}

function resolveRecoveryOutputTokenBudget(outputTokenBudget: number): number {
  if (outputTokenBudget <= 1000) {
    return outputTokenBudget;
  }

  return Math.min(outputTokenBudget, Math.max(1600, Math.round(outputTokenBudget * 0.7)));
}

function resolveMinimumAssistantTextChars(outputTokenBudget: number): number {
  if (outputTokenBudget <= 1000) {
    return 0;
  }
  if (outputTokenBudget <= 1800) {
    return 450;
  }
  if (outputTokenBudget <= 3000) {
    return 1200;
  }
  if (outputTokenBudget <= 4500) {
    return 2000;
  }

  return 2800;
}

function countVisibleTextChars(value: string): number {
  return value.replace(/\s+/gu, "").length;
}

function shouldRepairShortAssistantText(
  state: AppState,
  userText: string,
  assistantText: string,
  outputTokenBudget: number
): boolean {
  if (isImageProgressionCadence(state)) {
    return false;
  }

  const minimumChars = resolveMinimumAssistantTextChars(outputTokenBudget);
  if (minimumChars <= 0) {
    return false;
  }

  if (userRequestedBriefAssistantText(state, userText)) {
    return false;
  }

  return countVisibleTextChars(assistantText) < minimumChars;
}

function shouldRepairAbruptAssistantText(
  state: AppState,
  userText: string,
  assistantText: string,
  outputTokenBudget: number,
  sidecarErrors: string[] = []
): boolean {
  if (isImageProgressionCadence(state) || !assistantText.trim()) {
    return false;
  }

  const minimumChars = resolveMinimumAssistantTextChars(outputTokenBudget);
  const malformedRecovery = hasMalformedAssistantSidecarRecovery(sidecarErrors);
  if (minimumChars <= 0 && !malformedRecovery) {
    return false;
  }

  if (userRequestedBriefAssistantText(state, userText) && !hasSeverelyAbruptAssistantTextEnding(assistantText)) {
    return false;
  }

  return hasAbruptAssistantTextEnding(assistantText);
}

function userRequestedBriefAssistantText(state: AppState, userText: string): boolean {
  const instructionText = `${userText}\n${state.llm.systemPrompt}`;
  return /\b(?:brief|concise|short|one sentence|single sentence|summary only)\b|짧게|간단히|간략히|한\s*문장|요약만|요약해/u.test(
    instructionText
  );
}

function hasAbruptAssistantTextEnding(assistantText: string): boolean {
  const text = assistantText.trim();
  if (text.length < 24) {
    return false;
  }

  if (hasSeverelyAbruptAssistantTextEnding(text)) {
    return true;
  }

  const lastLine = text.split(/\n+/u).map((line) => line.trim()).filter(Boolean).at(-1) ?? text;
  if (/[,:;([{「『“"'\-]$/u.test(lastLine)) {
    return true;
  }
  if (/(?:그리고|그러나|하지만|그러자|그런데|곧|다시|이어|또한|because|and|but|then)$/iu.test(lastLine)) {
    return true;
  }
  if (/(?:은|는|이|가|을|를|에|에서|으로|로|와|과|의|도|만|에게|한테|부터|까지|처럼|보다|고|며|면서|지만|는데|다가|채|듯|위해|때문에)$/u.test(lastLine)) {
    return true;
  }

  const lastSentence = readLastSentenceFragment(text);
  return lastSentence.length > 120 && !hasCompleteKoreanOrPunctuationEnding(lastSentence);
}

function hasSeverelyAbruptAssistantTextEnding(assistantText: string): boolean {
  const text = assistantText.trim();
  return hasUnclosedMarkdownFence(text) || hasUnclosedAssistantQuote(text) || /[([{「『“"']$/u.test(text);
}

function hasUnclosedMarkdownFence(text: string): boolean {
  return (text.match(/```/gu)?.length ?? 0) % 2 === 1;
}

function hasUnclosedAssistantQuote(text: string): boolean {
  const pairs: Array<[string, string]> = [
    ["“", "”"],
    ["「", "」"],
    ["『", "』"]
  ];
  if (pairs.some(([open, close]) => countStringOccurrences(text, open) > countStringOccurrences(text, close))) {
    return true;
  }

  return countUnescapedDoubleQuotes(text) % 2 === 1;
}

function countStringOccurrences(text: string, needle: string): number {
  return text.split(needle).length - 1;
}

function countUnescapedDoubleQuotes(text: string): number {
  let count = 0;
  let escaped = false;
  for (const char of text) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (char === "\"") {
      count += 1;
    }
  }
  return count;
}

function readLastSentenceFragment(text: string): string {
  const normalized = text.replace(/\s+/gu, " ").trim();
  const parts = normalized.split(/[.!?。！？…]+/u).map((part) => part.trim()).filter(Boolean);
  return parts.at(-1) ?? normalized;
}

function hasCompleteKoreanOrPunctuationEnding(text: string): boolean {
  return /[.!?。！？…)\]」』”']$|(?:다|요|죠|네|까|군|음|함|됨|임|였다|했다|한다|된다|있다|없다)$/u.test(text.trim());
}

function resolveLlmRequestTimeoutMs(outputTokenBudget: number): number {
  const tokenScaledTimeout = 30_000 + outputTokenBudget * 20;
  return Math.min(MAX_LLM_REQUEST_TIMEOUT_MS, Math.max(MIN_LLM_REQUEST_TIMEOUT_MS, tokenScaledTimeout));
}

function isImageProgressionCadence(state: AppState): boolean {
  return state.imageProfile.generationCadence === "image_progression";
}

function createContentRatingInstruction(state: AppState): string {
  if (isAdultContentMode(state)) {
    return "Content rating: 19+ adult-only mode. Treat adult-only material as involving adults, follow creator rules and user boundaries, and keep the response compatible with the selected LLM provider.";
  }

  return "Content rating: general. Use the configured LLM/provider behavior and creator rules.";
}

function createCurrentTurnImagePolicyBlock(state: AppState, options: { manualImage?: boolean; outputTokenBudget?: number } = {}): string {
  const imageUserRules = createImageUserRulesForContentRating(state);
  const userRulePlan = analyzeImageCueUserRulePolicy(imageUserRules);
  const realtimeMode =
    state.imageProfile.enabled &&
    state.simulation.realtimeImageEnabled &&
    state.imageProfile.triggerMode !== "stored_only";
  const cadence = state.imageProfile.generationCadence ?? "balanced";
  const initialCueTarget = createInitialImageCueCountInstruction(state, options.outputTokenBudget);
  if (options.manualImage && realtimeMode) {
    return [
      "basis: manual image request",
      `trigger_mode: ${state.imageProfile.triggerMode}`,
      `cadence: ${cadence}`,
      initialCueTarget,
      "Obligation: the user requested an image for this turn. Emit at least one should_generate=true image_cue with complete final English NovelAI/Danbooru tags.",
      "Do not leave image_cues empty and do not return a generated cue with empty tags/base_tags/character_prompts. DynamicChat will not create a local fallback image prompt."
    ]
      .filter((item): item is string => Boolean(item))
      .join("\n");
  }

  if (!realtimeMode && !userRulePlan.requiresGeneration) {
    return [
      "basis: image generation settings",
      `trigger_mode: ${state.imageProfile.triggerMode}`,
      `cadence: ${cadence}`,
      "Image generation is not automatic in the current settings. Emit image_cues only if image prompt user rules explicitly require a visual cut."
    ].join("\n");
  }

  if (cadence === "image_progression") {
    return [
      "basis: image generation cadence",
      `trigger_mode: ${state.imageProfile.triggerMode}`,
      "cadence: image_progression",
      userRulePlan.requirements.length > 0 ? `matched_rules: ${userRulePlan.requirements.join(", ")}` : undefined,
      `Obligation: image_cues is the visible scene progression. Emit exactly ${IMAGE_PROGRESSION_CUE_TARGET} should_generate=true cue objects for the current turn, each with complete final NovelAI/Danbooru tags.`,
      "assistant_text should be only one compact Korean status line. Do not write prose narration, dialogue blocks, explanations, or markdown status panels in assistant_text for this mode.",
      "Each cue must advance the current situation by a small step and vary framing naturally across wide/context shots, action beats, expression close-ups, detail close-ups, over-the-shoulder, POV/first-person, and other fitting camera angles."
    ]
      .filter((item): item is string => Boolean(item))
      .join("\n");
  }

  if (userRulePlan.requiresGeneration) {
    return [
      "basis: image prompt user rules",
      `trigger_mode: ${state.imageProfile.triggerMode}`,
      `cadence: ${cadence}`,
      initialCueTarget,
      userRulePlan.requirements.length > 0 ? `matched_rules: ${userRulePlan.requirements.join(", ")}` : undefined,
      "Obligation: image_cues must contain complete final NovelAI/Danbooru tags for the cuts required by the image prompt user rules. Do not wait for a separate image request from the user.",
      cadence === "paragraph"
        ? "High-density paragraph mode is active: before writing JSON, privately split the planned assistant_text into visual beats. Emit one anchored should_generate=true cue for each significant dialogue line, action/interaction, body-detail, pose/outfit/state change, camera/framing change, or location change, up to 8. A long paragraph can require multiple cues; do not collapse a multi-beat visual turn into only one or two summary cues."
        : "If the assistant_text has any visible scene/action/dialogue/body beat, include at least one should_generate=true cue with 10-32 concrete English tags. Use [] only when the current response is impossible to visualize or purely administrative."
    ]
      .filter((item): item is string => Boolean(item))
      .join("\n");
  }

  if (cadence === "paragraph") {
    return [
      "basis: image generation cadence",
      `trigger_mode: ${state.imageProfile.triggerMode}`,
      "cadence: paragraph",
      initialCueTarget,
      "Obligation: image_cues is a high-density cut list, not a single summary. Before writing JSON, privately split the planned assistant_text into visual beats and emit an anchored image_cue for each significant dialogue line, action/interaction, body-detail, pose/outfit/state change, camera/framing change, or location change, up to 8 cues. Long paragraphs can contain several cues. Each generated cue needs its own short anchor_text and final NovelAI/Danbooru tags. Use [] only for beats that are purely administrative or impossible to visualize."
    ]
      .filter((item): item is string => Boolean(item))
      .join("\n");
  }
  if (cadence === "rich") {
    return [
      "basis: image generation cadence",
      `trigger_mode: ${state.imageProfile.triggerMode}`,
      "cadence: rich",
      "Obligation: image_cues is a cut list, not a single summary. For visually meaningful assistant_text, emit separate image_cues for scene/action/body/dialogue/body-detail beats when they differ. Prefer 2-4 cues for multi-beat responses, each with distinct anchor_text and final NovelAI/Danbooru tags. Use [] only for purely administrative or recap-only turns."
    ].join("\n");
  }
  if (cadence === "balanced") {
    return [
      "basis: image generation cadence",
      `trigger_mode: ${state.imageProfile.triggerMode}`,
      "cadence: balanced",
      "Default obligation: autonomously emit one complete image_cue for the strongest visible beat when assistant_text contains concrete scene action, character pose/reaction, outfit/state change, dialogue-face moment, or a location transition. Use [] only for quiet planning, recap, or text-only turns."
    ].join("\n");
  }

  return [
    "basis: image generation cadence",
    `trigger_mode: ${state.imageProfile.triggerMode}`,
    "cadence: sparse",
    "Guidance: emit image_cues for major scene/location changes, a newly visible character, important outfit/state change, or a visually decisive beat. Use [] for ordinary dialogue, recap, or non-visual planning."
  ].join("\n");
}

function analyzeImageCueUserRulePolicy(userRules: string): { requiresGeneration: boolean; requirements: string[] } {
  const rules = userRules.toLowerCase();
  const requirements = [
    /항상[^\n]*(?:이미지|생성|그려)|모든\s*(?:assistant|응답|턴|문맥|장면)[^\n]*(?:이미지|생성|그려)|매\s*(?:턴|문맥|장면)[^\n]*(?:이미지|생성|그려)|every\s+(?:turn|response)[^\n]*(?:image|generate)/iu.test(rules)
      ? "always_generate"
      : undefined,
    /전체\s*장면|장면\s*전체|전체\s*컷|whole[-\s]?scene|establishing\s+(?:shot|image)|각\s*(?:문맥|장면|컷)|(?:문맥|장면|컷)마다|every\s+(?:scene|context|cut)/iu.test(rules)
      ? "whole_scene"
      : undefined,
    /행동\s*(?:비트|마다|컷|장면|부분)|행위\s*(?:비트|마다|컷|장면|부분)|액션\s*(?:비트|마다|컷|장면)|동작\s*(?:마다|컷|장면|부분)|action\s+beat|every\s+action|state\s+change|상태\s*변화/iu.test(rules)
      ? "action_beat"
      : undefined,
    /body\s+detail|body\s+before\s+dialogue|신체\s*(?:디테일|컷|묘사|부위|강조)|몸\s*(?:디테일|컷|묘사|강조)|손\s*(?:디테일|컷|강조)|눈\s*(?:디테일|컷|강조)|부위\s*(?:강조|컷)/iu.test(rules)
      ? "body_detail"
      : undefined,
    /dialogue\s+face|face\s+before\s+dialogue|before\s+dialogue|대사\s*(?:전|앞|직전|이전|마다)|말하기\s*전|발화\s*전|신음\s*(?:전|앞|직전|이전|마다)|소리\s*(?:전|앞|직전|이전)|표정\s*(?:컷|마다)/iu.test(rules)
      ? "dialogue_face"
      : undefined,
    /재사용\s*(?:금지|하지\s*마|하지\s*말|안\s*함|불가)|기존\s*이미지\s*(?:사용|재사용)\s*(?:금지|하지\s*마|하지\s*말)|새(?:로|로운)\s*(?:이미지|컷)|fresh\s+image|do\s+not\s+reuse|no\s+reuse|never\s+reuse|always\s+generate|매번\s*(?:새로\s*)?생성|각\s*(?:문맥|장면|컷)[^\n]*(?:생성|그려|이미지)|(?:문맥|장면|컷)마다[^\n]*(?:생성|그려|이미지)/iu.test(rules)
      ? "fresh_image"
      : undefined
  ].filter((item): item is string => Boolean(item));

  return {
    requiresGeneration: requirements.length > 0,
    requirements
  };
}

function createImageGenerationCadenceBlock(state: AppState, outputTokenBudget?: number): string {
  const cadence = state.imageProfile.generationCadence ?? "balanced";
  if (cadence === "image_progression") {
    return `image_progression: assistant_text is minimal; emit exactly ${IMAGE_PROGRESSION_CUE_TARGET} ordered image_cues as a tag-only scene progression. Each cue is one image prompt group with concrete final tags and a distinct small advancement or camera/framing shift.`;
  }
  if (cadence === "sparse") {
    return "sparse: emit image_cues only for major scene/location changes, newly visible characters, important outfit/state changes, or a visually decisive beat. Usually 0-1 cue.";
  }
  if (cadence === "rich") {
    return "rich: emit separate image_cues for each major visible action, pose/body-contact, outfit/exposure, expression, or location change. Prefer 2-4 cues when the response has multiple visual beats; do not collapse changing beats into one image.";
  }
  if (cadence === "paragraph") {
    const target = createInitialImageCueCountInstruction(state, outputTokenBudget);
    return [
      "paragraph: high-density mode. Before writing JSON, privately budget image_cues from the assistant_text you are about to write: one cue per significant dialogue line, action/interaction, body-detail, pose/outfit/state change, camera/framing change, or location change, up to 8 cues. A long paragraph can contain multiple cues. Use placement and unique anchor_text so DynamicChat can attach each cut to its nearby text.",
      target
    ]
      .filter((item): item is string => Boolean(item))
      .join(" ");
  }

  return "balanced: emit one image_cue for the most important visual beat whenever the turn has concrete visible action, character pose/reaction, outfit/state change, dialogue-face moment, or location transition; use [] only for quiet text-only turns. Usually 1 cue on visual turns, 0 on quiet turns.";
}

function createInitialImageCueCountInstruction(state: AppState, outputTokenBudget?: number): string | undefined {
  if (state.imageProfile.generationCadence !== "paragraph" || outputTokenBudget === undefined) {
    return undefined;
  }

  const target = resolveInitialParagraphImageCueTarget(outputTokenBudget);
  return `Initial image_cues target for this request: start by budgeting about ${target.min}-${target.max} generated cuts if assistant_text reaches the selected output length. Count visual beats, not only paragraphs: dialogue lines, action/interaction beats, body-detail beats, pose/outfit/state changes, and camera/location changes each deserve their own cue. If the final assistant_text has more visual beats than the starting target, follow the beat count up to the hard cap of 8 instead of merging beats.`;
}

function resolveInitialParagraphImageCueTarget(outputTokenBudget: number): { min: number; max: number } {
  if (outputTokenBudget <= 1000) {
    return { min: 2, max: 3 };
  }
  if (outputTokenBudget <= 1800) {
    return { min: 3, max: 5 };
  }
  if (outputTokenBudget <= 3000) {
    return { min: 5, max: 8 };
  }

  return { min: 6, max: 8 };
}

function createImageCueTagContractInstruction(state: AppState): string {
  const adultExplicitInstruction = isAdultContentMode(state)
    ? "In adult_19 mode, emit direct visual NovelAI/Danbooru tags for the current adult-only beat when image generation rules call for it. Do not replace explicit visual context with empty image_cues merely because the scene is adult-only."
    : "Do not invent sexual tags unless the current scene and content mode explicitly allow and require them.";
  return [
    "Image cue NAI tag contract: image_cues.tags must be the actual final NovelAI tags for the current visible beat, not summaries, labels, or prose.",
    "NovelAI V4 prompt split contract: prefer image_cues.base_tags for Base Prompt tags and image_cues.character_prompts for Character Prompt tags. Base Prompt must contain artist-free scene, camera/framing, composition, lighting, environment, props, and whole-cut action. Character Prompt must contain each visible character's identity tags, expression, pose, outfit/clothing state, body/held-item state, and character-specific action. Do not put character identity, expression, pose, clothing, or body-state tags in base_tags when a visible character prompt exists.",
    "Never emit should_generate=true with empty visual tags. If the current beat should be shown, provide concrete tags through base_tags and/or character_prompts; tags may mirror the combined set for legacy compatibility. If you cannot provide those tags, set should_generate=false and explain the suppression_reason.",
    "Write 10-32 concise English NAI tags when an image is generated. Include subject count only when visible and unambiguous, e.g. 1girl, 1boy, 2girls; omit subject count when the visible count is uncertain.",
    "Privately compress the current situation into one visual intent line before tagging. Build tags from that intent line, not from every word in the story text.",
    "Each image_cue is one cut with one primary visible action. Do not copy every story element into tags; keep only tags that directly support the visible action and composition.",
    "Tags should cover the current actor, registered character visual profile traits, visible action, pose, expression, gaze/POV/camera framing, location, important props/held items, current outfit, and current visual state across image_cues.base_tags and image_cues.character_prompts. For a visible roster character, characters is metadata for traces/reuse only; it no longer activates or appends any local character prompt. Include the character's prompt tags directly in image_cues.character_prompts and mirror them in image_cues.tags only for legacy compatibility.",
    "Character metadata rule: image_cues.characters records visible registered ids for trace/reuse only; it does not add prompt tags locally.",
    "Character identity lock: for every id in image_cues.characters, image_cues.character_prompts must include one matching character_id and its prompt must include that character's required_identity_tags from the current-scene authoring reference plus the visible current outfit/state tags. If the current beat uses pronouns, first-person, or a continuation without names, preserve the previous visible image cast unless the user clearly changes the scene. Never mix one roster character's hair, eye, outfit, or body tags with another character's id.",
    "Character ambiguity rule: when multiple roster characters could match the pronoun and the current scene evidence does not disambiguate, either choose the character already visible in the latest image/assistant beat or set should_generate=false with a suppression_reason instead of guessing the wrong character.",
    "For action, interaction, or body_detail cues, background tags alone are a failure. Include pose/action and physical detail tags that define what the viewer sees.",
    "If user image rules say to emphasize a body part, use kind=body_detail with close framing, the focal body/contact tag, and the visible character id when it belongs to a roster character.",
    "Do not put artist, style, quality, resolution, or negative/undesired tags in image_cues.tags; those are configured separately by DynamicChat. Exclude tags like highres, absurdres, masterpiece, best quality, lowres, watermark.",
    "Private image cue self-check: subject count tags must match visible roster ids; one selected roster character must not become 2girls/3girls; do not combine short hair plus long hair; do not output an emotion/voice stack such as jealous, shocked, pleasure, moan, whispering, dialogue.",
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
    ? "For adult-only explicit interaction beats, use one direct core act plus physical Danbooru-style tags for position, pose, body-contact details, clothing/exposure state, anatomy, visible fluids/effects, and expression only when the current context requires them. Do not add unrelated narrative actions."
    : "Do not add explicit body-contact or sexual-act tags outside adult_19 context.";
  return [
    "NovelAI/Danbooru tag conversion rules:",
    "Use comma-separated English tags only. Do not output names, natural-language sentences, verb clauses, production labels, or abstract descriptions.",
    "Decompose prose into visual tags: `standing up from her seat` becomes `standing, chair`; `raising her hand eagerly` becomes `arm up, hand up, smile`; `bright and confident smile` becomes `smile`.",
    "Do not output generic identity/anatomy inventory such as human, person, female, woman, forehead, eyebrows, nose, chin, neck, shoulders, collarbone, skin, face, or head. Prefer one subject-count tag plus a few salient focus/contact tags.",
    "Convert abstract situations into visible anatomy, pose, contact, object, and expression tags. For example, `intimidation` becomes visible tags such as `looming`, `from below`, `open mouth`, `sweat` only if those are actually visible; `abduction` becomes tags such as `wrist grab`, `struggling`, `from behind` only if the scene visibly shows them; otherwise omit the abstract situation.",
    "Do not emit abstract/role/location-stack tags such as trauma, psychological, character, role, acting, acting scene, audition, trauma operative, director, student, acting student, acting coach, academy, performance, professional, mood, atmosphere, tension, fear, desire, or arousal.",
    "For an acting/audition-style scene, tag the visible filming elements instead of the concept: camera, video camera, holding camera, microphone, script, clapperboard, stage lights, or spotlight, only when the object/light is actually visible.",
    "Remove duplicate, synonymous, and contradictory tags. If the visible location is classroom/indoors, do not also output street/outdoors unless the scene truly shows both.",
    "Use this strict tag order for every generated cue: 1 character count; 2 perspective/framing; 3 background/environment; 4 base pose/action; 5 physical/prop/body interaction; 6 clothing state/outfit; 7 appearance; 8 expression/effects.",
    "Outfit keyword mappings are full prompts, not labels. When a character mapping says `교복: school uniform, dark grey pencil skirt, tight fit, necktie`, include every mapped tag that remains visible instead of only `school uniform`.",
    "When modifying a stored outfit, preserve its base garments and add condition tags. If the current Wearing state is `police uniform, navy short dress, mini skirt` and the uniform is torn, use `police uniform, navy short dress, mini skirt, torn uniform`, not `torn uniform`.",
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
  const providerOutputTokenBudget = resolveProviderOutputTokenBudget(state, outputTokenBudget);
  const timeoutMs = resolveLlmRequestTimeoutMs(providerOutputTokenBudget);

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
          maxOutputTokens: providerOutputTokenBudget,
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
      const errorBody = await response.text().catch(() => "");
      throw new Error(`Gemini request failed: ${response.status}${formatProviderErrorBody(errorBody)}`);
    }

    if (options.onRawText) {
      const streamed = await readGeminiStreamText(response, options.onRawText).catch(() => undefined);
      if (streamed?.trim()) {
        return streamed;
      }
      return requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
        ...options,
        onRawText: undefined
      });
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
        max_tokens: providerOutputTokenBudget,
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
      const streamed = await readClaudeStreamText(response, options.onRawText).catch(() => undefined);
      if (streamed?.trim()) {
        return streamed;
      }
      return requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
        ...options,
        onRawText: undefined
      });
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
      max_tokens: providerOutputTokenBudget,
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
    const streamed = await readOpenAiCompatibleStreamText(response, options.onRawText).catch(() => undefined);
    if (streamed?.trim()) {
      return streamed;
    }
    return requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
      ...options,
      onRawText: undefined
    });
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

function formatProviderErrorBody(value: string): string {
  const compact = value.replace(/\s+/gu, " ").trim();
  return compact ? ` (${compact.slice(0, 260)})` : "";
}

function formatUnknownError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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
  const allowRecovery = options.allowRecovery !== false;
  const allowProviderSafeRecovery = options.allowProviderSafeRecovery === true;
  try {
    const content = await requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, options);
    if (content?.trim()) {
      return content;
    }
    throw new Error("LLM response did not include content.");
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
        if (!allowRecovery || (shouldUseProviderSafeRecovery(blockNoneError) && !allowProviderSafeRecovery)) {
          throw blockNoneError;
        }
        return requestProviderRecoveryText(state, input, blockNoneError, outputTokenBudget, {
          allowProviderSafeRecovery,
          onRawText: options.onRawText
        });
      }
    }

    if (isProviderNonContentError(error)) {
      if (!allowRecovery || (shouldUseProviderSafeRecovery(error) && !allowProviderSafeRecovery)) {
        throw error;
      }
      return requestProviderRecoveryText(state, input, error, outputTokenBudget, {
        allowProviderSafeRecovery,
        onRawText: options.onRawText
      });
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
  options: Pick<ProviderTextOptions, "allowProviderSafeRecovery" | "onRawText"> = {}
): Promise<string | undefined> {
  const allowProviderSafeRecovery = options.allowProviderSafeRecovery === true;
  const onRawText = options.onRawText;
  const useProviderSafeRecovery = allowProviderSafeRecovery && shouldUseProviderSafeRecovery(originalError);
  const recoveryInstruction = useProviderSafeRecovery
    ? createProviderSafeRecoveryRuntimeInstruction(state, outputTokenBudget)
    : createRecoveryRuntimeInstruction(state, outputTokenBudget);
  const recoveryContext = useProviderSafeRecovery
    ? createProviderSafeRecoveryContextBlock(state, input, originalError)
    : createRecoveryContextBlock(state, input, originalError);
  const recoveryInput = {
    userText: "Continue the current turn now.",
    modules: [],
    evidence: []
  };
  const recoveryOutputTokenBudget = resolveRecoveryOutputTokenBudget(outputTokenBudget);

  try {
    const recoveryContent = await requestProviderText(state, recoveryInput, recoveryInstruction, recoveryContext, recoveryOutputTokenBudget, {
      temperature: 0.2,
      geminiSafetyThreshold: "OFF",
      onRawText
    });
    if (recoveryContent?.trim()) {
      return recoveryContent;
    }
    throw new Error("LLM recovery response did not include content.");
  } catch (recoveryError) {
    if (state.llm.provider === "gemini" && isGeminiOffSafetySettingRejected(recoveryError)) {
      const blockNoneContent = await requestProviderText(state, recoveryInput, recoveryInstruction, recoveryContext, recoveryOutputTokenBudget, {
        temperature: 0.2,
        geminiSafetyThreshold: "BLOCK_NONE",
        onRawText
      });
      if (blockNoneContent?.trim()) {
        return blockNoneContent;
      }
      throw new Error("LLM recovery response did not include content after BLOCK_NONE retry.");
    }

    if (allowProviderSafeRecovery && !useProviderSafeRecovery && isProviderNonContentError(recoveryError)) {
      const providerSafeContent = await requestProviderText(
        state,
        recoveryInput,
        createProviderSafeRecoveryRuntimeInstruction(state, outputTokenBudget),
        createProviderSafeRecoveryContextBlock(state, input, recoveryError),
        recoveryOutputTokenBudget,
        {
          temperature: 0.15,
          geminiSafetyThreshold: "OFF",
          onRawText
        }
      );
      if (providerSafeContent?.trim()) {
        return providerSafeContent;
      }
      throw new Error("LLM provider-safe recovery response did not include content.");
    }

    const originalMessage = originalError instanceof Error ? originalError.message : String(originalError);
    const recoveryMessage = recoveryError instanceof Error ? recoveryError.message : String(recoveryError);
    throw new Error(`${originalMessage}; recovery request failed: ${recoveryMessage}`);
  }
}

function shouldUseProviderSafeRecovery(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /PROHIBITED_CONTENT|prompt block|SAFETY|BLOCK|content_filter|refusal/iu.test(message);
}

function createGeminiSafetySettings(threshold: GeminiSafetyThreshold): Array<{ category: string; threshold: GeminiSafetyThreshold }> {
  return GEMINI_SAFETY_CATEGORIES.map((category) => ({ category, threshold }));
}

function createRuntimeInstruction(state: AppState, outputTokenBudget: number, options: { manualImage?: boolean } = {}): string {
  const contentRatingInstruction = createContentRatingInstruction(state);
  const imageCadenceInstruction = createImageGenerationCadenceBlock(state, outputTokenBudget);
  const initialImageCueCountInstruction = createInitialImageCueCountInstruction(state, outputTokenBudget);
  return [
    state.llm.systemPrompt,
    contentRatingInstruction,
    "High-priority roster/relationship-map guard: registered characters, relationship values, current moods, and stored status records are background reference only. Do not mention, summon, move, react with, or include a character solely because they exist in the roster or relationship map; require current-scene evidence from the recent transcript, current User action, or an active scene rule.",
    "DynamicChat execution order:",
    "1. Read the Simulation foundation first. Every Main rules block is creator-authored operating law for this turn.",
    "2. Read the Selected prompt modules for this turn after the foundation. Apply always modules and locally/neuralmap/manually selected modules as active rules, not as optional summaries.",
    "3. Continue the exact current scene from the Recent transcript and User action. assistant_text must be an actual Korean simulation continuation, not a meta acknowledgement that the next scene will continue.",
    "4. If creator rules define a response structure, status window, choice format, or required ending block, put that complete structure inside assistant_text every turn while keeping the outer response valid JSON.",
    isImageProgressionCadence(state)
      ? "Image progression mode override: visible response structures, long status windows, prose narration, and choice blocks are replaced by the compact assistant_text line plus the ordered image_cues cut list. Persist only durable changes in memory_events."
      : undefined,
    "Always preserve the Simulation foundation below: title, premise, existing character roster, and main rules are active every turn.",
    "The Immediate continuity anchor is the handoff point for this turn. Continue from the latest assistant ending and the current User action before using older retrieved memories.",
    "For long previous assistant output, the ending/status/choice block is more important than the opening. Do not restart from the prior response opening when a later ending is available.",
    "Use retrieved prompt modules and memory evidence as additional active context. If they conflict with the foundation, preserve explicit user-authored main rules unless the user changes them.",
    "Treat character summaries, relationships, current moods, active scene rules, and situation-specific modules as concrete setting constraints, not flavor text.",
    "Treat Structured simulation memory as the compact truth/current-state view. Do not infer that every character knows every fact; respect observation and belief notes when deciding character knowledge.",
    "Do not replace the main cast or invent unrelated protagonists. Introduce a new character only when the user action or retrieved context clearly requires one.",
    "If a user persona is active, treat it as the user's controlled role, background, goals, tone, and boundaries. If the persona source is an existing character, the User action is that character's action/dialogue/choice in the current scene; do not also puppet that controlled character beyond the user's explicit input.",
    "Input notation rule: any user text enclosed by the literal `*(` and `)*` marker, such as `*(문 쪽으로 이동한다)*`, is an action, stage direction, or descriptive instruction. Do not treat that enclosed text as spoken dialogue or internal thought unless the user explicitly labels it that way.",
    createOutputLengthInstruction(outputTokenBudget, state),
    "Use the Recent transcript to continue the exact current scene, location, cast, and momentum. Do not jump to a stale or unrelated scene label.",
    "If a character appears only in off-stage roster references, stored status, relationship parameters, visual profile lists, or NeuralMap evidence, treat that as background continuity and do not include them in assistant_text or image_cues.characters for this turn.",
    isImageProgressionCadence(state)
      ? `Image progression assistant_text rule: write one short Korean status line only, such as \`이미지 진행 ${IMAGE_PROGRESSION_CUE_TARGET}컷.\` Do not write prose narration, dialogue, markdown, status windows, choices, explanations, or visible tag lists in assistant_text.`
      : "Default assistant_text to natural Korean prose with Markdown only when useful. DynamicChat effect blocks are optional: use ```scene, ```impact, ```whisper, ```sfx, ```status, ```choice, ```memory, ```letter or one-line ::impact[text] only when the creator/user rules explicitly need a breakout visual beat. Do not write visible labels such as 'SFX:', 'impact:', or 'status -' as prose; effect labels are parser hints, not user-facing text. Do not output raw HTML.",
    "Image cue ownership: this main response owns final image_cues. No later tag planner will fix, expand, or infer tags. When the current beat should be illustrated, write the complete NovelAI tags in image_cues.tags now.",
    options.manualImage && state.imageProfile.enabled && state.simulation.realtimeImageEnabled && state.imageProfile.triggerMode !== "stored_only"
      ? "Manual image request: this turn must include at least one should_generate=true image_cue with complete final NovelAI/Danbooru tags unless the current response is impossible to visualize. If impossible, set should_generate=false with a specific suppression_reason."
      : undefined,
    isImageProgressionCadence(state)
      ? `For image_cues in image_progression mode, emit exactly ${IMAGE_PROGRESSION_CUE_TARGET} ordered should_generate=true cuts. Each cue is one final NovelAI tag prompt group; advance the scene little by little and vary camera/framing only when it fits the current continuity.`
      : "For image_cues, use [] only for quiet text-only turns. If user image rules or cadence require cuts, emit the complete cut list now: one cue for balanced/sparse major beats, 2-4 cues for rich multi-beat responses, and in paragraph/high-density mode one anchored cue per significant dialogue/action/body/state visual beat, up to 8. Long paragraphs may contain multiple cues. Every generated cue must include concrete final tags, visible character metadata, placement, and a short anchor_text when useful.",
    initialImageCueCountInstruction,
    "Image prompt user rules are binding composition and tag-routing instructions. Follow explicit positive/negative NovelAI tag directives, but never turn rule labels, examples, or headings into visible objects.",
    "Scene tag keyword presets are creator-authored hierarchical references. Read each path as parent keyword > child keyword; inherited_base_tags are broad branch context and base_scene_tags are the selected node details. You may consult multiple matching branches in one cue, such as one branch for action/scene and another branch for expression/pose. If no branch matches, ignore the preset list. Put scene/composition/action/environment results in base_tags and character-specific expression/pose/outfit/body-state results in the matching character_prompts item.",
    `Image generation cadence setting is binding: ${imageCadenceInstruction}`,
    createImageCueTagContractInstruction(state),
    "When a character's current outfit, pose, action, interaction, expression, held item, physical state, whole-scene phase, or scene composition changes, put that delta in memory_events with memory_kind='state'. Use state_type='Wearing', 'StatusTags', 'PoseTags', 'ActionTags', 'InteractionTags', 'InteractionPhaseTags', 'HeldItemTags', 'PhysicalStateTags', 'SceneTags', 'ScenePhaseTags', 'CompositionTags', 'CameraTags', or 'LightingTags'. state_value must be comma-separated English NovelAI tags, preferably 3-8 compact tags that preserve the situation rather than every micro-detail. For Wearing, do not collapse a detailed current outfit into a generic modifier label; preserve base garments/colors/style and append damage/condition tags, e.g. `police uniform, navy short dress, mini skirt, torn uniform`. Set actor_id for character-specific state. Leave actor_id empty only for whole-scene state. These records also help your next main response write stable image_cues; DynamicChat does not append them to prompts locally.",
    "For memory_events, preserve graph roles carefully: relationship deltas need actor_id and target_id when both sides are known; observations and beliefs need observers or actor_id for the character who knows the fact; scene-wide facts should not be assigned to a random roster character.",
    "Keep the outer JSON valid and stop cleanly. If the budget is tight, reduce memory_events first; do not drop required image_cues.tags, and do not end assistant_text as only an opening beat.",
    "Your response must be valid JSON only. Put the user-visible narrative in assistant_text. Put visual planning in image_cues. Put only structured memory deltas in memory_events. If assistant_text contains Markdown code fences, encode them as a JSON string value; never write Markdown outside the JSON object."
  ].join("\n");
}

function createRecoveryRuntimeInstruction(state: AppState, outputTokenBudget: number): string {
  return [
    "You are DynamicChat's resilient scene-continuation writer.",
    "The previous provider request returned no usable content or a boilerplate refusal. Write a provider-compatible continuation of the current scene instead of an error message.",
    "Do not mention the model, provider, filters, policies, blocked content, retry behavior, or generation failure in assistant_text.",
    "Respect the active creator rules and the current scene. If the exact user action cannot be continued directly, pivot to adjacent dialogue, aftermath, environmental action, or a clear choice point while preserving continuity.",
    "Do not quote or intensify provider-sensitive details from the transcript. Keep the response grounded in safe actions, emotions, decisions, and consequences.",
    "Image cue ownership remains with this main recovery response. Follow image generation cadence and image prompt user rules; when the recovered assistant_text has a visible beat that should be illustrated, author final NovelAI/Danbooru tags in image_cues.tags.",
    createImageCueTagContractInstruction(state),
    createOutputLengthInstruction(resolveRecoveryOutputTokenBudget(outputTokenBudget)),
    "Write JSON fields in this exact order: assistant_text, image_cues, memory_events. Keep memory_events compact or [] if needed.",
    `Simulation title: ${state.simulation.title}`,
    `Content rating: ${state.simulation.contentRating}`,
    "Return exactly this JSON shape with no Markdown outside JSON:",
    JSON.stringify({
      assistant_text: "Korean scene continuation shown to the user",
      image_cues: [
        {
          label: "visible beat",
          kind: "scene",
          placement: "after",
          anchor_text: "nearby assistant_text fragment",
          priority: 0.9,
          should_generate: true,
          reason: "image generation cadence or user image rules call for this cut",
          suppression_reason: "",
          characters: [],
          tags: ["1girl", "upper body", "looking at viewer", "classroom", "school uniform", "open mouth"],
          scene: "current scene",
          visual_context: "comma-separated final NovelAI tags"
        }
      ],
      memory_events: []
    })
  ].join("\n");
}

function createProviderSafeRecoveryRuntimeInstruction(state: AppState, outputTokenBudget: number): string {
  return [
    "You are DynamicChat's sanitized scene-continuation writer.",
    "The prior request could not be served from its raw context, so DynamicChat is retrying with a sanitized continuity pack.",
    "Continue the current simulation turn as closely as the sanitized creator rules and current user action allow. Do not invent an unrelated safe scene or skip the selected character/action.",
    "If the exact raw context cannot be continued directly, move to the nearest compatible beat: point-of-view selection, positioning, dialogue, tactical decision, emotional reaction, movement, investigation, or a clear choice point.",
    "Do not reproduce, quote, intensify, or euphemistically restate sensitive body details, coercive acts, injuries, fluids, slurs, or graphic action from prior raw context.",
    "Do not mention the model, provider, filters, policies, blocked content, retry behavior, or generation failure in assistant_text.",
    "Preserve active creator constraints, selected character, current point of view, location, and immediate decision pressure from the sanitized context.",
    "assistant_text must be Korean narrative, 2-5 paragraphs, enough to let the next user input continue naturally.",
    "Image cue ownership remains with this sanitized recovery response. Emit image_cues only for provider-compatible visible beats with concrete NovelAI/Danbooru tags; use [] for text-only or sensitive raw beats.",
    createImageCueTagContractInstruction(state),
    createOutputLengthInstruction(Math.min(outputTokenBudget, 1800)),
    `Simulation title: ${state.simulation.title}`,
    "Return exactly this JSON shape with no Markdown outside JSON:",
    JSON.stringify({
      assistant_text: "Korean sanitized scene continuation shown to the user",
      image_cues: [
        {
          label: "provider-compatible visible beat",
          kind: "scene",
          placement: "after",
          anchor_text: "nearby assistant_text fragment",
          priority: 0.75,
          should_generate: true,
          reason: "current sanitized beat has a provider-compatible visual cut",
          suppression_reason: "",
          characters: [],
          tags: ["1girl", "upper body", "indoors", "looking at viewer", "serious expression"],
          scene: "current scene",
          visual_context: "comma-separated provider-compatible final NovelAI tags"
        }
      ],
      memory_events: []
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
    "Current turn image generation policy:",
    createCurrentTurnImagePolicyBlock(state),
    "Image cue authoring reference for current-scene visible characters only. Character identity lock applies: listed character ids require their required_identity_tags in image_cues.tags:",
    createImageCueVisualProfileBlock(state, input.userText) || "(none)",
    "Current image/visual state records for current-scene characters only:",
    createImageCueCurrentStateBlock(state, input.userText) || "(none)",
    "Return JSON only. Use image_cues: [] only when the image generation policy says this retry should be text-only."
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function createProviderSafeRecoveryContextBlock(
  state: AppState,
  input: {
    userText: string;
    modules: PromptModule[];
    evidence: ContextEvidence[];
  },
  originalError: unknown
): string {
  const reason = originalError instanceof Error ? originalError.message : String(originalError);
  const activeCharacterNames = inferCurrentSceneCharacterIds(state, input.userText)
    .map((id) => state.characters.find((character) => character.id === id)?.name)
    .filter((name): name is string => Boolean(name))
    .slice(0, 5)
    .join(", ");
  const latestUser = findLatestMessageByRole(state, "user");
  const recentTranscript = state.messages
    .filter((message) => !(message.role === "assistant" && looksLikeProviderBoilerplateText(message.content)))
    .slice(-3)
    .map((message) => `${message.role}: ${sanitizeProviderRetryText(message.content, 220)}`)
    .join("\n");
  const moduleText = input.modules
    .filter((module) => module.kind !== "image_prompt_profile")
    .filter((module) => !(module.kind === "safety_policy" && isAdultContentMode(state)))
    .slice(0, 5)
    .map((module) => `# ${module.title}\n${truncatePromptText(sanitizeProviderRetryText(module.body, 700), 700, "sanitized retry module excerpt")}`)
    .join("\n\n");
  const evidenceText = input.evidence
    .slice(0, 5)
    .map((item) => `- ${sanitizeProviderRetryText(item.snippet, 180)}`)
    .join("\n");

  return [
    "Provider-safe retry context. Use this minimal context instead of raw transcript text.",
    `Failure summary: ${summarizeLlmFailureReason(reason)}`,
    `Premise summary: ${sanitizeProviderRetryText(state.simulation.description || state.simulation.title, 220)}`,
    activeCharacterNames ? `Current visible/controlled names: ${activeCharacterNames}` : undefined,
    `Current user action intent:\n${sanitizeProviderRetryText(input.userText, 220) || "continue the current point of view"}`,
    latestUser && latestUser.content !== input.userText
      ? `Previous user action intent:\n${sanitizeProviderRetryText(latestUser.content, 180)}`
      : undefined,
    recentTranscript ? `Sanitized recent transcript:\n${recentTranscript}` : undefined,
    moduleText ? `Sanitized active creator rules:\n${moduleText}` : undefined,
    evidenceText ? `Sanitized memory/context evidence:\n${evidenceText}` : undefined,
    "Continuity hint: the prior raw scene may be provider-sensitive. Do not quote it. Preserve selected point of view and current scene intent from the sanitized context.",
    "Current turn image generation policy:",
    createCurrentTurnImagePolicyBlock(state),
    "Image cue authoring reference for current-scene visible characters only:",
    createImageCueVisualProfileBlock(state, input.userText) || "(none)",
    "Return JSON only. Use memory_events: [] if no durable provider-compatible delta is available."
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
  const inactiveCharacterCount = state.characters
    .filter((character) => !activeCharacterIds.has(character.id))
    .length;
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
        (module.kind !== "character_prompt" ||
          Boolean(module.characterId && shouldIncludeCharacterScopedContext(state, module.characterId, activeCharacterIds))) &&
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
    inactiveCharacterCount > 0
      ? `Registered off-stage roster reference only, not active cast and not visual/image subjects: ${inactiveCharacterCount} character(s) omitted from this current-turn prompt until current scene evidence activates them.`
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
  if (mode === "reference") {
    return [`- id: ${character.id}`, `name: ${character.name}`, "off_stage: true"].join(" | ");
  }

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
    visualProfile?.positivePrompt ? `visual profile tags: ${truncatePromptText(visualProfile.positivePrompt, 520, "visual profile")}` : undefined,
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

function createOutputLengthInstruction(maxTokens: number, state?: AppState): string {
  if (state && isImageProgressionCadence(state)) {
    return `Output mode: image progression. Do not spend the budget on visible prose. assistant_text must be one compact Korean status line, while image_cues must carry exactly ${IMAGE_PROGRESSION_CUE_TARGET} ordered tag prompt groups.`;
  }

  if (maxTokens <= 1000) {
    return "Output length target: compact. The token setting is both the budget ceiling and the intended amount of narrative work. assistant_text must usually be 2-3 Korean paragraphs with concrete action, dialogue, and one clear consequence. Do not collapse the turn into a one-sentence summary unless the user explicitly asks for a brief reply.";
  }

  if (maxTokens <= 1800) {
    return "Output length target: balanced. The token setting is both the budget ceiling and the intended amount of narrative work. assistant_text must usually be 4-6 Korean paragraphs with concrete action, dialogue, sensory detail, and visible consequences. Do not stop after setup; advance the scene through at least two beats unless the user explicitly asks for brevity.";
  }

  if (maxTokens <= 3000) {
    return "Output length target: long. The token setting is both the budget ceiling and the intended amount of narrative work. assistant_text must usually be 7-10 Korean paragraphs, developing the scene through multiple beats, dialogue, state changes, and a meaningful ending hook. Keep JSON valid, but do not shorten to a compact response just to be conservative.";
  }

  if (maxTokens <= 4500) {
    return "Output length target: very long. The token setting is both the budget ceiling and the intended amount of narrative work. assistant_text must usually be 10-14 Korean paragraphs, with richer scene progression, character reaction, state changes, and a complete status/choice block when creator rules require it. Keep JSON valid, but avoid short fallback-style narration.";
  }

  return "Output length target: extended. The token setting is both the budget ceiling and the intended amount of narrative work. assistant_text must usually be 12-18 Korean paragraphs, with substantial scene progression, dialogue, consequences, and all creator-required status/choice structure. Keep JSON valid, but never answer as a short summary unless the user explicitly requests it.";
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
      `image_cue_actor_rule: User actions, first-person narration, and '나' refer to controlled_character_id ${character.id}; include this id in image_cues.characters when this character is the visible actor or speaker, and include the character's visual/outfit/state tags directly in image_cues.tags.`,
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
    const parseError = error instanceof Error ? error.message : "Invalid JSON sidecar.";
    const recovered = parseMalformedAssistantSidecar(jsonText, parseError);
    if (recovered.sidecar) {
      return recovered;
    }

    return {
      errors: recovered.errors
    };
  }
}

function parseMalformedAssistantSidecar(raw: string, parseError: string): { sidecar?: AssistantSidecar; errors: string[] } {
  const errors = [parseError, MALFORMED_SIDECAR_RECOVERY_MARKER];
  const assistantText =
    extractJsonStringField(raw, ["assistant_text", "assistantText"]) ??
    extractLooseJsonTextField(raw, ["assistant_text", "assistantText"]);
  if (!assistantText) {
    return { errors };
  }

  const memoryEvents = extractRecoverableMemoryEventDrafts(raw, errors);
  const imageCues = extractRecoverableImageCueDrafts(raw, errors);
  const imageCue = imageCues[0] ?? createNoImageCue("Malformed LLM sidecar did not include a recoverable image cue");

  return {
    sidecar: {
      assistantText,
      memoryEvents,
      imageCue,
      imageCues
    },
    errors
  };
}

function parseAssistantSidecarWithFallbackAssistantText(
  raw: string,
  assistantText: string
): { sidecar?: AssistantSidecar; errors: string[] } {
  const parsed = parseAssistantSidecar(raw);
  if (parsed.sidecar) {
    return parsed;
  }

  const errors = uniqueStrings([
    ...parsed.errors,
    "Recovered sidecar metadata while preserving existing assistant_text."
  ]);
  const jsonText = extractJsonObject(raw);
  if (jsonText) {
    try {
      const value = JSON.parse(jsonText) as Record<string, unknown>;
      const metadataSidecar = createSidecarFromMetadataValue(value, assistantText, errors);
      if (metadataSidecar) {
        return {
          sidecar: metadataSidecar,
          errors
        };
      }
    } catch (error) {
      errors.push(error instanceof Error ? error.message : "Invalid metadata-only sidecar JSON.");
    }
  }

  const arrayMetadataSidecar = createSidecarFromMetadataArrayText(raw, assistantText, errors);
  if (arrayMetadataSidecar) {
    return {
      sidecar: arrayMetadataSidecar,
      errors
    };
  }

  const memoryEvents = extractRecoverableMemoryEventDrafts(raw, errors);
  const imageCues = extractRecoverableImageCueDrafts(raw, errors);
  if (memoryEvents.length === 0 && imageCues.length === 0) {
    return { errors };
  }

  return {
    sidecar: {
      assistantText,
      memoryEvents,
      imageCue: imageCues[0] ?? createNoImageCue("Metadata-only retry did not include a recoverable image cue"),
      imageCues
    },
    errors
  };
}

function createSidecarFromMetadataArrayText(
  raw: string,
  assistantText: string,
  errors: string[]
): AssistantSidecar | undefined {
  const stripped = stripLikelyJsonFence(raw).trim();
  if (!stripped.startsWith("[")) {
    return undefined;
  }

  try {
    const parsed = JSON.parse(stripped) as unknown;
    const imageCues = normalizeAssistantImageCueDrafts(parsed, errors);
    if (imageCues.length === 0) {
      return undefined;
    }

    return {
      assistantText,
      memoryEvents: [],
      imageCue: imageCues[0],
      imageCues
    };
  } catch (error) {
    errors.push(error instanceof Error ? error.message : "Invalid metadata-only image_cues array.");
    return undefined;
  }
}

function createSidecarFromMetadataValue(
  value: Record<string, unknown>,
  assistantText: string,
  errors: string[]
): AssistantSidecar | undefined {
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
  if (memoryEvents.length === 0 && imageCues.length === 0) {
    return undefined;
  }

  return {
    assistantText,
    memoryEvents,
    imageCue: imageCues[0] ?? createNoImageCue("Metadata-only retry did not include an image cue"),
    imageCues
  };
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
  const baseTags = readStringArray(value.base_tags).length > 0 ? readStringArray(value.base_tags) : readStringArray(value.baseTags);
  const characterPrompts = normalizeImageCueCharacterPrompts(value.character_prompts ?? value.characterPrompts, errors);
  const visualContext = readString(value.visual_context) ?? readString(value.visualContext);
  const explicitShouldGenerate = readBoolean(value.should_generate) ?? readBoolean(value.shouldGenerate);
  const kind = readString(value.kind) ?? readString(value.type) ?? readString(value.cue_type) ?? readString(value.cueType);
  return {
    shouldGenerate: explicitShouldGenerate ?? Boolean(tags.length > 0 || baseTags.length > 0 || characterPrompts.length > 0 || visualContext),
    reason,
    characters: readStringArray(value.characters),
    tags,
    baseTags,
    characterPrompts,
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

function normalizeImageCueCharacterPrompts(value: unknown, errors: string[]): ImageCueCharacterPrompt[] {
  if (!value) {
    return [];
  }

  if (Array.isArray(value)) {
    return value
      .map((item) => normalizeImageCueCharacterPrompt(item, errors))
      .filter((item): item is ImageCueCharacterPrompt => Boolean(item))
      .slice(0, 8);
  }

  if (isRecord(value)) {
    return Object.entries(value)
      .map(([characterId, promptValue]) => {
        if (typeof promptValue === "string") {
          return normalizeImageCueCharacterPrompt({ character_id: characterId, prompt: promptValue }, errors);
        }
        if (isRecord(promptValue)) {
          return normalizeImageCueCharacterPrompt({ character_id: characterId, ...promptValue }, errors);
        }
        return undefined;
      })
      .filter((item): item is ImageCueCharacterPrompt => Boolean(item))
      .slice(0, 8);
  }

  errors.push("image_cues.character_prompts is not an array or object.");
  return [];
}

function normalizeImageCueCharacterPrompt(value: unknown, errors: string[]): ImageCueCharacterPrompt | undefined {
  if (!isRecord(value)) {
    errors.push("image_cues.character_prompts item is not an object.");
    return undefined;
  }

  const tags = readStringArray(value.tags);
  const prompt =
    readString(value.prompt) ??
    readString(value.character_prompt) ??
    readString(value.characterPrompt) ??
    (tags.length > 0 ? tags.join(", ") : undefined);
  if (!prompt?.trim()) {
    return undefined;
  }

  return {
    characterId: readString(value.character_id) ?? readString(value.characterId) ?? readString(value.id),
    prompt: prompt.trim(),
    negativePrompt: readString(value.negative_prompt) ?? readString(value.negativePrompt),
    center: normalizeImageCueCharacterCenter(value.center)
  };
}

function normalizeImageCueCharacterCenter(value: unknown): ImageCueCharacterPrompt["center"] | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const x = readNumber(value.x);
  const y = readNumber(value.y);
  if (x === undefined || y === undefined) {
    return undefined;
  }

  return {
    x: clampNumber(x, 0, 1),
    y: clampNumber(y, 0, 1)
  };
}

function createFallbackSidecar(
  assistantText: string,
  reason = "Structured sidecar fallback; no LLM-authored image cue was available."
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

function extractRecoverableImageCueDrafts(raw: string, errors: string[]): PlannedImageCueDraft[] {
  const text = stripLikelyJsonFence(raw);
  const valueStart = findJsonFieldValueStart(text, ["image_cues", "imageCue", "image_cue"]);
  if (valueStart === undefined) {
    return [];
  }

  return extractJsonObjectFragmentsFromFieldValue(text.slice(valueStart))
    .map((fragment) => parseJsonObjectFragment(fragment))
    .filter((item): item is Record<string, unknown> => Boolean(item))
    .map((item) => normalizeImageCueDraft(item, errors))
    .filter((item): item is PlannedImageCueDraft => Boolean(item));
}

function extractRecoverableMemoryEventDrafts(raw: string, errors: string[]): AssistantSidecar["memoryEvents"] {
  const text = stripLikelyJsonFence(raw);
  const valueStart = findJsonFieldValueStart(text, ["memory_events", "memoryEvents"]);
  if (valueStart === undefined) {
    return [];
  }

  return extractJsonObjectFragmentsFromFieldValue(text.slice(valueStart))
    .map((fragment) => parseJsonObjectFragment(fragment))
    .filter((item): item is Record<string, unknown> => Boolean(item))
    .map((item) => normalizeMemoryEventDraft(item, errors))
    .filter((item): item is AssistantSidecar["memoryEvents"][number] => Boolean(item));
}

function findJsonFieldValueStart(text: string, fieldNames: string[]): number | undefined {
  for (const fieldName of fieldNames) {
    const match = new RegExp(`"${fieldName}"\\s*:\\s*`, "u").exec(text);
    if (match) {
      return match.index + match[0].length;
    }
  }

  return undefined;
}

function extractJsonObjectFragmentsFromFieldValue(valueText: string): string[] {
  const firstValueIndex = valueText.search(/\S/u);
  if (firstValueIndex < 0) {
    return [];
  }

  const firstChar = valueText[firstValueIndex];
  if (firstChar !== "[" && firstChar !== "{") {
    return [];
  }

  const valueKind = firstChar === "[" ? "array" : "object";
  const fragments: string[] = [];
  let objectStart = -1;
  let objectDepth = 0;
  let inString = false;
  let escaped = false;

  for (let index = firstValueIndex; index < valueText.length; index += 1) {
    if (objectDepth === 0) {
      if (valueKind === "object" && fragments.length > 0) {
        break;
      }
      if (valueKind === "array" && valueText[index] === "]") {
        break;
      }
      if (/^\s*,\s*"(?:memory_events|memoryEvents)"\s*:/u.test(valueText.slice(index))) {
        break;
      }
    }

    const char = valueText[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }

    if (char === "\"") {
      inString = true;
      continue;
    }

    if (char === "{") {
      if (objectDepth === 0) {
        objectStart = index;
      }
      objectDepth += 1;
      continue;
    }

    if (char === "}" && objectDepth > 0) {
      objectDepth -= 1;
      if (objectDepth === 0 && objectStart >= 0) {
        fragments.push(valueText.slice(objectStart, index + 1));
        objectStart = -1;
      }
    }
  }

  return fragments;
}

function parseJsonObjectFragment(fragment: string): Record<string, unknown> | undefined {
  const candidates = [
    fragment,
    fragment.replace(/,\s*([}\]])/gu, "$1")
  ];

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      if (isRecord(parsed)) {
        return parsed;
      }
    } catch {
      // Try the next conservative repair candidate.
    }
  }

  return undefined;
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
    if (looksLikeInternalPromptLeak(assistantText)) {
      return fallback;
    }
    return assistantText;
  }

  const stripped = stripLikelyJsonFence(raw).trim();
  if (!stripped || looksLikeSidecarJson(stripped) || looksLikeInternalPromptLeak(stripped)) {
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

function looksLikeInternalPromptLeak(value: string): boolean {
  return /(?:SYSTEM INSTRUCTION:|CONTEXT BLOCK:|USER ACTION:|Use the following DynamicChat context|Return JSON only\. The JSON schema is|Immediate continuity anchor:|#\s*Immediate Continuity Anchor|Use this before older retrieved memories|Current scene cast guard:|Simulation foundation:|Structured simulation memory:|Selected prompt modules for this turn:|Memory\/context evidence:)/iu.test(value);
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
      if (isLikelyJsonStringFieldTerminator(text, index)) {
        literal += "\"";
        return decodeJsonStringLiteral(literal);
      }

      literal += "\\\"";
      continue;
    }

    literal += char === "\n" ? "\\n" : char === "\r" ? "\\r" : char;
  }

  return decodeJsonStringLiteral(`${literal.replace(/\\$/u, "")}"`);
}

function isLikelyJsonStringFieldTerminator(text: string, quoteIndex: number): boolean {
  const remainder = text.slice(quoteIndex + 1);
  return (
    /^\s*\}\s*$/u.test(remainder) ||
    /^\s*,\s*(?:\\?["'])?[A-Za-z_][A-Za-z0-9_]*(?:\\?["'])?\s*:/u.test(remainder) ||
    /^\s*,\s*(?:$|(?:\\?["'])?$|(?:\\?["'])?(?:image_cues|imageCue|image_cue|memory_events|memoryEvents)[A-Za-z_]*(?:\\?["'])?\s*$)/u.test(
      remainder
    )
  );
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
  if (typeof value === "string") {
    return value
      .split(/[,;\n|]+/u)
      .map((item) => item.trim())
      .filter(Boolean);
  }

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
