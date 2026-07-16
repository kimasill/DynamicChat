import type {
  AppState,
  AssistantImageCueDraft,
  AssistantMemoryEventDraft,
  AssistantSidecar,
  ContextEvidence,
  ImageCueCharacterPrompt,
  ImageScenePresetExampleFile,
  ImageSceneTagPresetNode,
  PromptModule
} from "../types";
import { cliAgentKindForProvider, isCliAgentLlmProvider } from "../types";
import { getLlmCliAgentProxyUrl } from "./dynamicChatApi";
import { createImageUserRulesForContentRating, isAdultContentMode } from "./contentRating";
import { IMAGE_STATE_TYPE_INSTRUCTION } from "./imageStateTags";
import { createStructuredContextSummary, describeRelationshipParameterValue } from "./memoryCompiler";
import { createSceneCastPromptBlock, inferCurrentSceneCharacterIds } from "./sceneCast";
import {
  readStateMemoryKind,
  readStateMemoryOwnerId,
  readStateMemoryStateType,
  readStateMemoryTargetId,
  readStateMemoryValue,
  splitImageStateTagValue
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

class LlmRateLimitError extends Error {
  readonly retryAfterMs: number;

  constructor(retryAfterMs: number) {
    const retryAfterSeconds = Math.max(1, Math.ceil(retryAfterMs / 1000));
    super(`LLM request failed: 429 (retry after ${retryAfterSeconds}s)`);
    this.name = "LlmRateLimitError";
    this.retryAfterMs = retryAfterMs;
  }
}

const LLM_MIN_REQUEST_SPACING_MS = 1_800;
const LLM_RATE_LIMIT_DEFAULT_COOLDOWN_MS = 30_000;
const LLM_RATE_LIMIT_MIN_COOLDOWN_MS = 12_000;
const LLM_RATE_LIMIT_MAX_COOLDOWN_MS = 120_000;

class LlmProviderScheduler {
  private readonly chains = new Map<string, Promise<void>>();
  private readonly lastRequestAt = new Map<string, number>();
  private readonly circuitUntil = new Map<string, number>();

  private queueKey(provider: string, apiKey: string): string {
    const trimmedKey = apiKey.trim();
    return `${provider}:${trimmedKey.length > 0 ? trimmedKey.slice(-12) : "anonymous"}`;
  }

  isCoolingDown(provider: string, apiKey: string): boolean {
    return Date.now() < (this.circuitUntil.get(this.queueKey(provider, apiKey)) ?? 0);
  }

  remainingCooldownMs(provider: string, apiKey: string): number {
    return Math.max(0, (this.circuitUntil.get(this.queueKey(provider, apiKey)) ?? 0) - Date.now());
  }

  recordRateLimit(provider: string, apiKey: string, retryAfterMs: number): void {
    const cooldownMs = Math.min(
      LLM_RATE_LIMIT_MAX_COOLDOWN_MS,
      Math.max(LLM_RATE_LIMIT_MIN_COOLDOWN_MS, retryAfterMs || LLM_RATE_LIMIT_DEFAULT_COOLDOWN_MS)
    );
    const key = this.queueKey(provider, apiKey);
    const nextUntil = Date.now() + cooldownMs;
    const currentUntil = this.circuitUntil.get(key) ?? 0;
    this.circuitUntil.set(key, Math.max(currentUntil, nextUntil));
  }

  async run<T>(provider: string, apiKey: string, task: () => Promise<T>): Promise<T> {
    const key = this.queueKey(provider, apiKey);
    const previous = this.chains.get(key) ?? Promise.resolve();
    const current = previous
      .catch(() => undefined)
      .then(async () => {
        await this.waitForAvailableSlot(provider, apiKey);
        return task();
      });
    this.chains.set(
      key,
      current.then(
        () => undefined,
        () => undefined
      )
    );
    return current;
  }

  private async waitForAvailableSlot(provider: string, apiKey: string): Promise<void> {
    const key = this.queueKey(provider, apiKey);
    const now = Date.now();
    const circuitUntil = this.circuitUntil.get(key) ?? 0;
    if (now < circuitUntil) {
      throw new LlmRateLimitError(circuitUntil - now);
    }

    const lastRequestAt = this.lastRequestAt.get(key) ?? 0;
    const spacingDelayMs = Math.max(0, lastRequestAt + LLM_MIN_REQUEST_SPACING_MS - now);
    if (spacingDelayMs > 0) {
      await sleepMs(spacingDelayMs);
    }

    this.lastRequestAt.set(key, Date.now());
  }
}

const llmProviderScheduler = new LlmProviderScheduler();

const MIN_OUTPUT_TOKENS = 512;
const INTERACTIVE_OUTPUT_TOKEN_CAP = 8000;
const MIN_LLM_REQUEST_TIMEOUT_MS = 45_000;
const MAX_LLM_REQUEST_TIMEOUT_MS = 180_000;
// Local CLI agents (claude/codex/gemini) carry heavy fixed startup + agentic overhead and
// stream slowly, so they routinely exceed the ceiling that suits a rate-limited HTTP provider.
// Turn generation streams, and the server bridge allows a streaming run up to ~900s with a 120s
// inactivity (idle) timeout that ends a truly stuck process early. The client ceiling used to be
// 300s, which KILLED legitimately-long streaming generations on heavy (many-module) simulations at
// ~278s and forced a fallback even while the model was still producing tokens. Raise the client
// budget so a slow-but-still-streaming turn can finish; a real hang is still caught by the server's
// idle timeout, so the larger client ceiling is a backstop, not the primary abort. For NON-streaming
// CLI calls the bridge's own timeout (DYNAMICCHAT_CLI_TIMEOUT_MS, default 280s) still gives up first.
const MIN_CLI_AGENT_REQUEST_TIMEOUT_MS = 90_000;
const MAX_CLI_AGENT_REQUEST_TIMEOUT_MS = 600_000;
// Main-prompt modules are the creator's core operating law and must survive intact — silently cutting their
// tail (where style/format rules often live) makes the model look like it "isn't reading the main prompt".
// Kept far higher than lore/selected-module caps for that reason. The old 5200 truncated long creator prompts.
// (Interactive-latency risk historically came from MANY selected modules, not one long main prompt.)
const MAIN_RULE_MAX_CHARS = 16000;
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
const IMAGE_SCENE_PRESET_EXAMPLE_PER_FILE = 3;
const IMAGE_SCENE_PRESET_EXAMPLE_TOTAL = 12;
const IMAGE_SCENE_PRESET_EXAMPLE_MAX_LENGTH = 240;
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
  // When true, the narrative LLM does NOT author image_cues at all — a separate image-cue LLM call
  // (requestTurnImageCues) plans the visuals afterward. Keeps tag rules out of the narrative prompt.
  separateImageCues?: boolean;
  onAssistantText?: (assistantText: string) => void;
  onEarlyImageCues?: (cues: AssistantImageCueDraft[]) => void;
}): Promise<{
  content: string;
  sidecar: AssistantSidecar;
  source: "mock" | "llm" | "fallback";
  sidecarStatus: "parsed" | "fallback" | "failed";
  sidecarErrors: string[];
  requestPreview?: string;
  rawPreview?: string;
  error?: string;
  sidecarExpansion?: Promise<AssistantSidecar | undefined>;
}> {
  const { state } = input;
  const outputTokenBudget = resolveInteractiveOutputTokenBudget(state);
  const contextBlock = createContextBlock(state, input.userText, input.modules, input.evidence, {
    manualImage: input.manualImage,
    outputTokenBudget,
    omitImageAuthoring: input.separateImageCues
  });
  const runtimeInstruction = createRuntimeInstruction(state, outputTokenBudget, {
    manualImage: input.manualImage,
    modules: input.modules,
    omitImageAuthoring: input.separateImageCues
  });
  const requestPreview = createRequestPreview(runtimeInstruction, contextBlock, input.userText);

  const requiresApiKey = !isCliAgentLlmProvider(state.llm.provider);
  if (!state.llm.enabled || state.llm.provider === "mock" || (requiresApiKey && !state.llm.apiKey.trim())) {
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
    // Only extract image_cues mid-stream when the realtime pipeline is active: there the initial cue
    // array is stable (background completion only appends), so the cueIndex an early-dispatched cut gets
    // is the same one the post-turn pass would assign — which is exactly what the caller's dispatch-key
    // dedup relies on to never fire the paid provider twice for one cut.
    const emitEarlyImageCues =
      input.onEarlyImageCues && !input.separateImageCues && shouldPrioritizeImagePipeline(state, Boolean(input.manualImage))
        ? createEarlyImageCueEmitter(input.onEarlyImageCues)
        : undefined;
    const onPrimaryRawText = combineRawTextHandlers(emitAssistantText, emitEarlyImageCues);
    let sidecarExpansion: Promise<AssistantSidecar | undefined> | undefined;
    let rawContent = await requestProviderTextWithRecovery(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
      allowProviderSafeRecovery: true,
      onRawText: onPrimaryRawText
    });
    if (!rawContent?.trim()) {
      throw new Error("LLM response did not include content.");
    }

    let parsed = parseAssistantSidecar(rawContent);
    if (parsed.sidecar && looksLikeProviderBoilerplateText(parsed.sidecar.assistantText) && !isLlmProviderCoolingDown(state)) {
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
      const prioritizeImagePipeline = shouldPrioritizeImagePipeline(state, Boolean(input.manualImage));
      const skipSupplementalLlmCalls = isLlmProviderCoolingDown(state);
      // Local CLI agents (claude/codex/gemini) pay a fresh subprocess cold start + full
      // prompt re-processing on every call, so a prose-length repair/continuation is not a cheap
      // touch-up — it re-runs a full-budget generation and can double or triple turn latency.
      // The primary streamed call already runs at the user's configured token budget, so skip the
      // length-padding round-trips for these backends and let the single pass stand.
      const isCliAgentBackend = isCliAgentLlmProvider(state.llm.provider);
      const skipProseSupplementalLlmCalls = skipSupplementalLlmCalls || prioritizeImagePipeline || isCliAgentBackend;
      // The annotation call owns image_cues in this mode, so the narrative call is INSTRUCTED not to emit any —
      // which means every image-cue completeness check below is guaranteed to "fail" and fire a retry. That retry
      // re-generates the turn under a tag-focused prompt and can replace the visible narrative with its output:
      // exactly the tag/prose interference the separation exists to remove (and a second full CLI generation).
      // Image cues are validated where they are authored (requestTurnAnnotations), never here.
      const imageCuesOwnedElsewhere = Boolean(input.separateImageCues);
      if (skipSupplementalLlmCalls) {
        sidecarCompletionErrors.push(
          "LLM provider rate limit active; skipped supplemental completion/repair requests for this turn."
        );
      } else if (prioritizeImagePipeline) {
        sidecarCompletionErrors.push(
          "Realtime image pipeline active; skipped prose-length repair requests so image generation can start sooner."
        );
      } else if (isCliAgentBackend) {
        sidecarCompletionErrors.push(
          "CLI agent backend active; skipped prose-length repair/continuation and add-more-cuts/memory sidecar-completion round-trips to keep turn latency low (single streamed pass at the configured token budget). A single image-salvage retry is still allowed when a visual-required turn produced no LLM image tags."
        );
      }
      if (!skipProseSupplementalLlmCalls && shouldRepairShortAssistantText(state, input.userText, parsed.sidecar.assistantText, outputTokenBudget)) {
        const originalSidecar = parsed.sidecar;
        const repaired = await requestShortAssistantTextRepair(
          state,
          input,
          runtimeInstruction,
          contextBlock,
          outputTokenBudget,
          originalSidecar.assistantText,
          emitAssistantText,
          input.separateImageCues
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
            emitAssistantText,
            input.separateImageCues
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
      if (
        !skipProseSupplementalLlmCalls &&
        parsed.sidecar &&
        shouldRepairAbruptAssistantText(state, input.userText, parsed.sidecar.assistantText, outputTokenBudget, parsed.errors)
      ) {
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
      if (
        !skipSupplementalLlmCalls &&
        !imageCuesOwnedElsewhere &&
        parsed.sidecar &&
        hasMalformedAssistantSidecarRecovery(parsed.errors) &&
        // Only pay for a repair LLM call when the malformed JSON ACTUALLY lost the image tags on a turn that
        // needs an image. The local malformed-JSON recovery usually salvages the cues already, but this block
        // used to fire on EVERY malformed turn (the model often emits invalid JSON — unescaped quotes in the
        // long Korean narrative) — a full second generation of tens of seconds + tokens right before the first
        // image. Skip it when renderable cues survived; when they didn't, re-request ONLY the image_cues (slim).
        !hasGeneratedImageCueTags(parsed.sidecar) &&
        shouldRequireMainImageTags(state, input.userText, parsed.sidecar.assistantText, Boolean(input.manualImage))
      ) {
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
            imageCuesOnly: true,
            extraIssues: ["image_cues were dropped by malformed JSON"],
            rejectionReason:
              "Previous LLM output was malformed JSON and its image_cues were lost. Re-emit only the image_cues for the already-shown narrative."
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
      // When the realtime image pipeline is prioritized, don't block the turn on a second (slow)
      // LLM round-trip just to collect *more* image_cues — start generation with what we already
      // have. Only fall through to the completion retry when no renderable cue exists yet, since
      // otherwise there would be no image this turn.
      const hasRenderableImageCueAlready = parsed.sidecar ? hasGeneratedImageCueTags(parsed.sidecar) : false;
      const skipImageCueCountRetry = prioritizeImagePipeline && hasRenderableImageCueAlready;
      if (
        skipImageCueCountRetry &&
        !skipSupplementalLlmCalls &&
        !imageCuesOwnedElsewhere &&
        // CLI agents pay a FULL second generation per call (cold start + full re-process), so the background
        // image-cue EXPANSION roughly DOUBLES per-turn usage. The first pass already targets the full paragraph
        // cue count (createInitialImageCueCountInstruction → up to 6-8), so for CLI backends we accept whatever
        // it produced and never fire the expansion. (API backends keep it — the second call is cheap there.)
        !isCliAgentBackend &&
        parsed.sidecar &&
        shouldRetryIncompleteAssistantSidecar(state, input.userText, parsed.sidecar, outputTokenBudget, Boolean(input.manualImage), {
          imageMetadataOnly: prioritizeImagePipeline
        })
      ) {
        sidecarCompletionErrors.push(
          "Realtime image pipeline active; started image generation with available image_cues and is expanding the remaining cues in the background."
        );
        // Run the count-completion retry in the background so the caller can start the initial
        // images immediately and append the rest once the (slow) second round-trip resolves.
        const baseSidecar = parsed.sidecar;
        sidecarExpansion = requestAssistantSidecarCompletionRetry(
          state,
          input,
          runtimeInstruction,
          contextBlock,
          outputTokenBudget,
          baseSidecar,
          undefined,
          // Expansion only needs to add image cuts to the already-final narrative, and its result is consumed
          // for image_cues alone — so send the slim image-only instruction + context, not the full prompt again.
          { forcePreserveAssistantText: true, imageCuesOnly: true }
        )
          .then((completed) =>
            completed?.parsed.sidecar &&
            shouldAcceptAssistantSidecarCompletionRetry(
              state,
              input.userText,
              baseSidecar,
              completed.parsed.sidecar,
              outputTokenBudget,
              Boolean(input.manualImage)
            )
              ? completed.parsed.sidecar
              : undefined
          )
          .catch(() => undefined);
      }
      // CLI agents pay a full subprocess cold start + prompt re-processing on every call, so the
      // synchronous sidecar-completion retry is a second full-budget generation that blocks the turn
      // and roughly doubles latency. Speed-first: skip it for these backends — EXCEPT when the turn
      // visually requires an image but the first pass produced no renderable image tags. Image tags
      // must be LLM-authored (user rules/cadence only decide *whether* to draw, not the tag content),
      // so that one case is the sole safety net that keeps the turn from silently dropping its image.
      // We still allow at most this single salvage retry; "add more cuts" / memory-completeness
      // retries stay skipped for CLI agents.
      const cliImageTagsMissingForRequiredTurn =
        isCliAgentBackend &&
        !imageCuesOwnedElsewhere &&
        Boolean(parsed.sidecar) &&
        shouldRequireMainImageTags(state, input.userText, parsed.sidecar!.assistantText, Boolean(input.manualImage)) &&
        !hasGeneratedImageCueTags(parsed.sidecar!);
      if (
        !skipSupplementalLlmCalls &&
        // This retry is the one that can REWRITE the visible narrative (it streams through emitAssistantText and
        // replaces parsed on accept). In separated-cue mode its trigger is always a false alarm — the narrative was
        // told to emit no cues — so it must never run: the prose the user reads stays the clean narrative pass.
        !imageCuesOwnedElsewhere &&
        (!isCliAgentBackend || cliImageTagsMissingForRequiredTurn) &&
        !skipImageCueCountRetry &&
        parsed.sidecar &&
        shouldRetryIncompleteAssistantSidecar(state, input.userText, parsed.sidecar, outputTokenBudget, Boolean(input.manualImage), {
          imageMetadataOnly: prioritizeImagePipeline
        })
      ) {
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
        rawPreview: rawContent.slice(0, 700),
        sidecarExpansion
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
      const metadataRecovered =
        isLlmProviderCoolingDown(state)
          ? undefined
          : await requestAssistantSidecarCompletionRetry(
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
      const recovered = isLlmProviderCoolingDown(state)
        ? undefined
        : await requestAssistantSidecarCompletionRetry(
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
    const fallbackContent = isLlmRateLimitError(error)
      ? createRateLimitFallbackContent(input.fallback, error)
      : isProviderNonContentError(error)
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

// Slim instruction for the image-cut EXPANSION call only: the assistant_text is already final and the
// expansion result is consumed for image_cues alone (see planTurnImagesWithExpansion), so the full narrative
// runtime instruction (memory/persona/format rules) is dead weight a CLI subprocess would re-process. Keep the
// image tag contract so tag quality is unchanged.
function createImageCueExpansionInstruction(state: AppState): string {
  return [
    "You are DynamicChat's image-cut expander. The turn's assistant_text is ALREADY final and shown to the user; do NOT rewrite, restate, or change it. Produce ONLY the remaining image_cues for that same already-written text.",
    "Respond with valid JSON: image_cues first, then assistant_text (omit it or set it to an empty string), then memory_events as []. Durable state/memory was already saved this turn, so memory_events MUST be [].",
    createImageCueTagContractInstruction(state),
    "Return valid JSON only with no Markdown outside the JSON object."
  ].join("\n");
}

// Slim context for the image-cut EXPANSION call: only the data that drives image-tag quality (who is visible,
// their saved appearance + current outfit/pose/state, image user rules, scene presets, cadence/policy). The
// narrative context (foundation, selected narrative modules, transcript, structured memory, persona, evidence)
// is omitted because the narrative is already written and the expansion only adds image cuts.
function createImageCueExpansionContextBlock(
  state: AppState,
  currentUserText: string,
  options: { manualImage?: boolean; outputTokenBudget?: number }
): string {
  const sceneCastText = createSceneCastPromptBlock(state, currentUserText);
  const sceneBriefingText = createImageSceneBriefingBlock(state, currentUserText);
  const imageUserRulesText = createImageUserRulesBlock(state);
  const imageScenePresetText = createImageSceneTagPresetBlock(state, currentUserText);
  const imageVisualProfileText = createImageCueVisualProfileBlock(state, currentUserText);
  const imageCurrentStateText = createImageCueCurrentStateBlock(state, currentUserText);
  const imageCadenceText = createImageGenerationCadenceBlock(state, options.outputTokenBudget);
  const currentTurnImagePolicyText = createCurrentTurnImagePolicyBlock(state, options);
  return [
    "Image-cut expansion context. The narrative is already written; only add image_cues that match it.",
    "Current scene cast guard:",
    sceneCastText,
    sceneBriefingText ? "Current scene briefing (cast count, identities, current outfit/action/condition, who-acts-on-whom):" : undefined,
    sceneBriefingText || undefined,
    "Image prompt user rules:",
    imageUserRulesText || "(none)",
    "Image tag keyword presets (no roster identity tags):",
    imageScenePresetText || "(none)",
    "Image generation cadence:",
    imageCadenceText,
    "Current turn image generation policy:",
    currentTurnImagePolicyText,
    "Image cue authoring reference for current-scene visible characters only (DynamicChat injects each registered character's identity + current outfit; do not repeat them):",
    imageVisualProfileText || "(none)",
    "Ongoing scene/visual state for current-scene characters only (still-active pose/action/interaction/position and body/clothing-condition facts; keep them on every cut):",
    imageCurrentStateText || "(none)"
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
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
    imageCuesOnly?: boolean;
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
    : "Write JSON fields in this exact order: image_cues, assistant_text, memory_events.";
  // Image-cut expansion sends a slim image-only instruction + context instead of re-processing the full
  // narrative prompt; every other completion-retry path keeps the full runtime instruction + context.
  const effectiveBaseInstruction = options.imageCuesOnly ? createImageCueExpansionInstruction(state) : runtimeInstruction;
  const effectiveContextBlock = options.imageCuesOnly
    ? createImageCueExpansionContextBlock(state, input.userText, { manualImage: input.manualImage, outputTokenBudget })
    : contextBlock;
  const completionInstruction = [
    effectiveBaseInstruction,
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
    effectiveContextBlock,
    "",
    options.imageCuesOnly ? "Already-written assistant_text (do not change; add image_cues that match it):" : "Rejected assistant_text preview:",
    truncatePreview(previousSidecar.assistantText, 1100),
    "",
    options.imageCuesOnly ? "Image_cues already produced (continue from these; do not duplicate):" : "Rejected image_cues summary:",
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

function shouldPrioritizeImagePipeline(state: AppState, manualImage = false): boolean {
  if (!state.imageProfile.enabled || !state.simulation.realtimeImageEnabled) {
    return false;
  }
  if (state.imageProfile.triggerMode === "stored_only") {
    return false;
  }
  if (state.imageProfile.triggerMode === "manual" && !manualImage) {
    return false;
  }
  if (state.imageProfile.triggerMode === "realtime_confirm" && !manualImage) {
    return false;
  }

  const cadence = state.imageProfile.generationCadence ?? "balanced";
  return (
    manualImage ||
    cadence === "balanced" ||
    cadence === "rich" ||
    cadence === "paragraph" ||
    cadence === "image_progression"
  );
}

function shouldRetryIncompleteAssistantSidecar(
  state: AppState,
  userText: string,
  sidecar: AssistantSidecar,
  outputTokenBudget: number,
  manualImage = false,
  options: { imageMetadataOnly?: boolean } = {}
): boolean {
  return (
    createAssistantSidecarCompletionIssues(state, userText, sidecar, outputTokenBudget, manualImage, options).length > 0
  );
}

function hasMalformedAssistantSidecarRecovery(errors: string[]): boolean {
  return errors.some((error) => error === MALFORMED_SIDECAR_RECOVERY_MARKER);
}

function createAssistantSidecarCompletionIssues(
  state: AppState,
  userText: string,
  sidecar: AssistantSidecar,
  outputTokenBudget: number,
  manualImage = false,
  options: { imageMetadataOnly?: boolean } = {}
): string[] {
  const issues: string[] = [];
  if (!options.imageMetadataOnly) {
    if (shouldRepairShortAssistantText(state, userText, sidecar.assistantText, outputTokenBudget)) {
      issues.push("assistant_text ended before the selected length target");
    } else if (shouldRepairAbruptAssistantText(state, userText, sidecar.assistantText, outputTokenBudget)) {
      issues.push("assistant_text appears cut off before a complete handoff");
    }
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
  onRawText?: (rawText: string) => void,
  omitImageAuthoring?: boolean
): Promise<{ rawContent: string; parsed: ReturnType<typeof parseAssistantSidecar> } | undefined> {
  const minimumChars = resolveMinimumAssistantTextChars(outputTokenBudget);
  const repairInstruction = [
    runtimeInstruction,
    "",
    "Previous valid JSON was rejected by DynamicChat because assistant_text was far shorter than the selected output target.",
    `Regenerate the whole JSON response now. assistant_text must be at least about ${minimumChars} Korean characters unless the user explicitly asks for a brief reply.`,
    "The visible assistant_text is the product. Spend nearly all of the response budget there before writing metadata.",
    omitImageAuthoring
      ? "Write JSON fields in this exact order: assistant_text, then memory_events. Do NOT include image_cues."
      : "Write JSON fields in this exact order: image_cues, assistant_text, memory_events.",
    omitImageAuthoring
      ? "Keep memory_events compact for this repair: 0-3 concise semantic deltas (no memory_kind='state')."
      : "Keep sidecar metadata compact for this repair: image_cues should be 0-1 cue unless the user's image rules explicitly require more, and memory_events should be 0-3 concise deltas.",
    "Do not close assistant_text after only an opening beat. Continue the scene with the full target amount of narrative work."
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
  onRawText?: (rawText: string) => void,
  omitImageAuthoring?: boolean
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
    omitImageAuthoring
      ? "Write JSON fields in this exact order: assistant_text, then memory_events. Do NOT include image_cues."
      : "Write JSON fields in this exact order: image_cues, assistant_text, memory_events.",
    omitImageAuthoring
      ? "Keep memory_events compact for this continuation: 0-3 concise semantic deltas (no memory_kind='state')."
      : "Keep sidecar metadata compact for this continuation: image_cues should be 0-1 cue unless the user's image rules explicitly require more, and memory_events should be 0-3 concise deltas.",
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
        "identity_rule: set character_id on this character's character_prompt and write its current action, pose, expression, and interaction first. DynamicChat will inject required_identity_tags (hair/eyes/face/body) so you do not repeat them.",
        "outfit_rule: default_outfit_tags and outfit_keyword_mappings are REFERENCE ONLY. You decide the outfit/exposure tags for each cut based on the current composition and what should actually be visible. Adapt — do not paste the defaults verbatim. If the action is a state change (lifted/aside/torn/open/wet/removed/nude), write the explicit state tag; if a body region is the focus, name what is shown rather than reciting the full default outfit."
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

// Consolidated, authoring-ready snapshot of WHO is on-stage right now and, per character, their current
// outfit, ongoing action/pose, expression, body/clothing condition, and interaction target — joined in one
// place so the model can compose image_cues that reflect the cast count, who-is-who, outfits, actions, and
// who-acts-on-whom without having to cross-reference the scattered cast/profile/state blocks. Tag vocabulary
// still comes from the model + user rules + presets; this only surfaces existing persisted state.
function createImageSceneBriefingBlock(state: AppState, currentUserText = ""): string {
  const presentIds = inferCurrentSceneCharacterIds(state, currentUserText);
  if (presentIds.length === 0) {
    return "";
  }
  const nameById = new Map(state.characters.map((character) => [character.id, character.name] as const));
  // Latest current-state value per (ownerId, canonical state_type) — later events overwrite earlier ones.
  const latestStateValue = new Map<string, string>();
  const latestInteractionTarget = new Map<string, string>();
  for (const event of state.memoryEvents) {
    if (!isCurrentStateMemoryEvent(event)) {
      continue;
    }
    const ownerId = readStateMemoryOwnerId(event);
    const stateType = readStateMemoryStateType(event);
    const value = readStateMemoryValue(event);
    if (!ownerId || !stateType || !value) {
      continue;
    }
    latestStateValue.set(`${ownerId}::${stateType}`, value);
    if (/Interaction/u.test(stateType)) {
      const targetId = readStateMemoryTargetId(event);
      if (targetId) {
        latestInteractionTarget.set(ownerId, targetId);
      }
    }
  }
  const pick = (ownerId: string, types: string[]): string =>
    uniqueStrings(types.flatMap((type) => splitImageStateTagValue(latestStateValue.get(`${ownerId}::${type}`)))).join(", ");
  const lines = presentIds.map((id) => {
    const name = nameById.get(id) ?? id;
    const wearing = pick(id, ["Wearing"]);
    const doing = pick(id, ["PoseTags", "ActionTags", "InteractionTags", "InteractionPhaseTags"]);
    const expression = pick(id, ["ExpressionTags"]);
    const condition = pick(id, ["PhysicalStateTags", "BodyStateTags", "StatusTags"]);
    const held = pick(id, ["HeldItemTags"]);
    const targetId = latestInteractionTarget.get(id);
    const toward = targetId && nameById.has(targetId) ? nameById.get(targetId) : undefined;
    const segments = [
      `wearing: ${wearing || "(use saved/default outfit)"}`,
      doing ? `doing: ${doing}` : undefined,
      expression ? `expression: ${expression}` : undefined,
      condition ? `condition: ${condition}` : undefined,
      held ? `holding: ${held}` : undefined,
      toward ? `interacting_with: ${toward}` : undefined
    ].filter((segment): segment is string => Boolean(segment));
    return `- ${name} (character_id: ${id}) | ${segments.join(" | ")}`;
  });
  const header =
    `${presentIds.length} character(s) are present in the scene right now (listed below). These are CANDIDATES, not a mandatory cast for every cut. ` +
    "For EACH cut, first decide the camera/framing, then emit one character_prompts entry ONLY for the characters whose body is actually visible inside that frame, and set the subject-count base tag to that VISIBLE count (e.g. 1girl / 1boy / 2girls / 1girl 1boy). " +
    "A character who is present but NOT in the chosen frame must NOT get an entry in that cut — in particular the point-of-view/observer character whose eyes the shot looks through (e.g. when the cut shows what they are looking at) is usually off-frame, so do not add them just because they are on-stage. Match image_cues.characters to the entries you actually emit. " +
    "For each character you DO render, reflect their listed current outfit, ongoing action/pose, expression, condition, and interaction target — keep them unless this turn explicitly changes them. " +
    "Do not invent characters that are not listed and never merge two of them into one entry. When one character acts on another, " +
    "render the interaction from both sides (the actor's pose/hands and the target's reaction/contact point).";
  return [header, ...lines].join("\n");
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

  let exampleBudget = IMAGE_SCENE_PRESET_EXAMPLE_TOTAL;
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
      const exampleFiles = selectScenePresetExampleFiles(node.exampleFiles, exampleBudget);
      exampleBudget -= exampleFiles.reduce((sum, file) => sum + file.prompts.length, 0);
      return [
        `- path: ${formatImageScenePresetPath(path)}`,
        `depth: ${depth + 1}`,
        `priority: ${node.priority}`,
        branchScore > 0 ? `matched_branch_score: ${branchScore}` : undefined,
        parentTags.length > 0 ? `inherited_base_tags: ${parentTags.join(", ")}` : undefined,
        sceneTags.length > 0 ? `base_scene_tags: ${sceneTags.join(", ")}` : undefined,
        parentNotes.length > 0 ? `inherited_creator_notes: ${parentNotes.map((note) => truncatePromptText(note, 120, "parent scene preset note")).join(" / ")}` : undefined,
        node.note.trim() ? `creator_note: ${truncatePromptText(node.note, 260, "scene preset note")}` : undefined,
        exampleFiles.length > 0 ? `example_prompts: ${formatScenePresetExampleFiles(exampleFiles)}` : undefined,
        "rule: adapt this branch only if relevant. example_prompts are optional references; when none are provided, author tags normally. example_prompts are the creator's own finished NovelAI prompts for this keyword, optionally split into labeled groups (for example a per-character side/role like a female-side and a male-side group): use each group as a style/structure/tag-vocabulary reference for the matching character_prompt and adapt to the current scene, do not copy verbatim. Scene/composition/shared-action/environment tags go to image_cues.base_tags; expression, exact pose, clothing, body-state, and identity-like tags go to the appropriate image_cues.character_prompts item; do not copy character-identity tags from examples since registered character appearance is injected separately."
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

function selectScenePresetExampleFiles(
  exampleFiles: ImageScenePresetExampleFile[] | undefined,
  budget: number
): { label: string; prompts: string[] }[] {
  if (!Array.isArray(exampleFiles) || exampleFiles.length === 0 || budget <= 0) {
    return [];
  }

  let remaining = budget;
  const selected: { label: string; prompts: string[] }[] = [];
  for (const file of exampleFiles) {
    if (remaining <= 0) {
      break;
    }
    const cleaned = (file.prompts ?? []).map((line) => line.trim()).filter(Boolean);
    if (cleaned.length === 0) {
      continue;
    }
    const prompts = cleaned.slice(0, Math.min(IMAGE_SCENE_PRESET_EXAMPLE_PER_FILE, remaining));
    remaining -= prompts.length;
    selected.push({ label: typeof file.label === "string" ? file.label.trim() : "", prompts });
  }
  return selected;
}

function formatScenePresetExampleFiles(files: { label: string; prompts: string[] }[]): string {
  return files
    .map((file) => {
      const lines = file.prompts.map((line) => truncatePromptText(line, IMAGE_SCENE_PRESET_EXAMPLE_MAX_LENGTH, "scene preset example")).join(" || ");
      return file.label ? `[${file.label}] ${lines}` : lines;
    })
    .join(" ; ");
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

function isLlmRateLimitError(error: unknown): error is LlmRateLimitError {
  return error instanceof LlmRateLimitError;
}

function isLlmProviderCoolingDown(state: AppState): boolean {
  return llmProviderScheduler.isCoolingDown(state.llm.provider, state.llm.apiKey);
}

function isRateLimitHttpStatus(status: number): boolean {
  return status === 429;
}

function readRetryAfterMs(response: Response, fallbackMs = LLM_RATE_LIMIT_DEFAULT_COOLDOWN_MS): number {
  const header = response.headers.get("retry-after")?.trim();
  if (header) {
    const seconds = Number(header);
    if (Number.isFinite(seconds) && seconds >= 0) {
      return Math.min(LLM_RATE_LIMIT_MAX_COOLDOWN_MS, seconds * 1000);
    }
    const retryAt = Date.parse(header);
    if (Number.isFinite(retryAt)) {
      return Math.min(LLM_RATE_LIMIT_MAX_COOLDOWN_MS, Math.max(0, retryAt - Date.now()));
    }
  }

  return fallbackMs;
}

function createRateLimitErrorFromResponse(response: Response): LlmRateLimitError {
  return new LlmRateLimitError(readRetryAfterMs(response));
}

function recordProviderRateLimit(state: AppState, response: Response): void {
  llmProviderScheduler.recordRateLimit(state.llm.provider, state.llm.apiKey, readRetryAfterMs(response));
}

function throwProviderHttpError(state: AppState, response: Response, providerLabel: string, errorBody = ""): never {
  if (isRateLimitHttpStatus(response.status)) {
    recordProviderRateLimit(state, response);
    throw createRateLimitErrorFromResponse(response);
  }

  throw new Error(`${providerLabel} request failed: ${response.status}${formatProviderErrorBody(errorBody)}`);
}

function sleepMs(durationMs: number): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, durationMs);
  });
}

function createRateLimitFallbackContent(fallback: string, error: LlmRateLimitError): string {
  const retryAfterSeconds = Math.max(1, Math.ceil(error.retryAfterMs / 1000));
  const summary = `LLM API 할당량/요청 제한(HTTP 429)에 도달했습니다. 약 ${retryAfterSeconds}초 후 다시 시도하세요`;
  return createLlmFailureFallbackContent(fallback, summary).replace(
    "설정과 모델 응답 상태를 확인한 뒤 다시 입력하면 실제 LLM 흐름으로 이어질 수 있다.",
    `같은 턴에서 추가 LLM 보정 요청은 중단되었습니다. ${retryAfterSeconds}초 정도 기다린 뒤 다시 입력하면 실제 LLM 흐름으로 이어질 수 있습니다.`
  );
}

function summarizeLlmFailureReason(reason: string): string {
  const compact = reason.replace(/\s+/gu, " ").trim();
  if (/request failed:\s*429|HTTP\s*429|rate limit|RESOURCE_EXHAUSTED|quota exceeded|too many requests/iu.test(compact)) {
    const retryMatch = compact.match(/retry after\s*(\d+)s/iu);
    return retryMatch
      ? `LLM API 할당량/요청 제한(HTTP 429)에 도달했습니다. 약 ${retryMatch[1]}초 후 다시 시도하세요`
      : "LLM API 할당량/요청 제한(HTTP 429)에 도달했습니다. 잠시 후 다시 시도하세요";
  }
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

interface CreatorOutputFormatHints {
  requiresVisibleFormattedOutput: boolean;
  requiresMarkdownTables: boolean;
  requiresStatusBlock: boolean;
  // True only when the MAIN prompt itself explicitly asks for a visible status panel/table. This is the gate the
  // relationship-map path uses: incidental table/「상태」 mentions in world lore, presets, or selected modules must NOT
  // force a visible status panel into the narrative — only an explicit main-prompt directive does.
  requiresStatusBlockFromMain: boolean;
  // True when the MAIN prompt explicitly designs ANY visible format (status panel, table, effect block, or choice block).
  // When false, assistant_text is enforced as clean narrative prose: the sanitizer removes effect blocks, tables, and
  // status panels regardless of what incidental tables/keywords appear elsewhere in lore/presets/modules.
  requiresVisibleFormatFromMain: boolean;
}

function createContextBlock(
  state: AppState,
  currentUserText: string,
  modules: PromptModule[],
  evidence: ContextEvidence[],
  options: { manualImage?: boolean; outputTokenBudget?: number; omitImageAuthoring?: boolean } = {}
): string {
  const creatorOutputFormat = analyzeCreatorOutputFormat(state, modules);
  const foundationText = createSimulationFoundationBlock(state, currentUserText);
  const immediateContinuityText = createImmediateContinuityBlock(state);
  // Narrative audience: strips image-machinery state types and qualitatively bands relationship-map
  // values so the prose model never sees NovelAI tag vocabulary or raw numeric stat values.
  const structuredMemoryText = createStructuredContextSummary(state, { maxEvents: 8, maxStates: 10, currentText: currentUserText, audience: "narrative" });
  const recentTranscriptText = createRecentTranscriptBlock(state);
  const turnSelectedModules = modules.filter(
    (module) => !(module.kind === "safety_policy" && isAdultContentMode(state)) && !isFoundationContextModule(module)
  );
  const moduleText = turnSelectedModules.map((module) => formatActivePromptModule(module)).join("\n\n");
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
  const sceneBriefingText = createImageSceneBriefingBlock(state, currentUserText);

  const imageCueShape = {
    kind: "scene | action | body_detail | dialogue_face | context | interaction",
    placement: "before | after | inline",
    anchor_text: "a short phrase you WILL write verbatim in assistant_text, marking where this cut belongs (plan it here, then include that phrase in the narrative)",
    should_generate: true,
    characters: ["visible registered character ids"],
    base_tags: ["base prompt tags only: artist-free scene, camera, composition, location, props, action shared by the cut"],
    character_prompts: [
      {
        character_id: "registered visible character id — omit entirely for any figure not in the registered roster (unregistered NPC, enemy, bystander, crowd member); NEVER assign a registered character's id to a different person's caption (e.g. do not put a registered female character's id on a male aggressor's entry)",
        prompt: "this character's CURRENT action, pose, expression, interaction, framing/visible-body-state tags only — DynamicChat injects the saved appearance and current outfit, so do not restate identity or the established outfit; write a garment tag only when this turn changed it or the crop reveals a region",
        negative_prompt: "optional character-specific negative tags"
      }
    ]
  };

  // When the image-cue authoring is handled by a separate LLM call, drop every image section + the image_cues
  // schema/order rules from the narrative prompt so the narrative model focuses purely on prose + memory.
  const omit = options.omitImageAuthoring === true;

  return [
    "Use the following DynamicChat context. Do not reveal internal IDs unless asked.",
    "Immediate continuity anchor:",
    immediateContinuityText || "(no previous assistant turn)",
    omit ? undefined : "Current scene cast guard:",
    omit ? undefined : sceneCastText,
    omit || !sceneBriefingText ? undefined : "Current scene briefing (cast count, identities, current outfit/action/condition, who-acts-on-whom):",
    omit ? undefined : sceneBriefingText || undefined,
    "Return JSON only. The JSON schema is:",
    omit
      ? JSON.stringify({
          assistant_text: isImageProgressionCadence(state)
            ? "<one short Korean status line>"
            : "<the full Korean in-character narrative for this turn>",
          memory_events: [
            {
              // state kind is omitted — a separate annotation call records all state/visual deltas.
              memory_kind: "event | observation | belief | goal | relationship | open_thread | summary",
              event_type: "short stable event type, when memory_kind is event",
              importance: 0.86,
              confidence: 0.92,
              tags: ["promise", "relationship", "scene-event"],
              content: "one concise durable delta, not the full assistant response",
              actor_id: "optional character id",
              actor_name: "optional character name",
              target_id: "optional target character/item/location id",
              observers: ["character ids who observed or heard this"]
            }
          ]
        })
      : JSON.stringify({
      // Keep these placeholders as short, neutral SLOT markers, not instructional prose. An earlier version
      // embedded a long English directive inside this value ("THE STORY the user reads…"); the model copied that
      // English sentence straight into the displayed narrative (prompt leak / 문맥 간섭). The emphasis that
      // assistant_text must stay full and vivid lives in the instruction lines below and in the runtime system
      // prompt, so it is not lost by trimming the schema slot.
      // Narrative-first: assistant_text is authored BEFORE image_cues so the prose is never bent toward tag
      // thinking and each cue can anchor to text that already exists.
      assistant_text: isImageProgressionCadence(state)
        ? "<one short Korean status line>"
        : "<the full Korean in-character narrative for this turn>",
      image_cues: [imageCueShape],
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
    omit ? undefined : "Current turn image generation policy:",
    omit ? undefined : currentTurnImagePolicyText,
    omit
      ? "Write the JSON fields in this exact order: assistant_text first, then memory_events. assistant_text is the user-visible Korean simulation continuation. Do NOT output image_cues; a separate step plans all visuals. Do not emit memory_kind='state' in memory_events; a separate annotation step records all state/visual/outfit/pose deltas."
      : "Write the JSON fields in this exact order: assistant_text first, then image_cues, then memory_events. assistant_text is the user-visible Korean simulation continuation. After the narrative is written, author image_cues and set each cue's anchor_text to a short phrase that ACTUALLY appears in the assistant_text.",
    isImageProgressionCadence(state) && !omit
      ? `Image progression mode: keep assistant_text to one short Korean status line and use image_cues as the actual scene progression. Output exactly ${IMAGE_PROGRESSION_CUE_TARGET} should_generate=true cue objects when realtime image generation is active. Return at most 3 memory_events.`
      : omit
        ? "Match assistant_text to the runtime output length target and keep it complete: close the JSON object every time. Return at most 8 memory_events."
        : "Match assistant_text to the runtime output length target and keep it complete: close the JSON object every time. Return at most 8 memory_events. You are the only image cue/tag author in this turn; DynamicChat will not run a later tag planner. For image_cues, emit [] only for quiet text-only turns; when an image should be generated, write the final usable NovelAI tags yourself.",
    "Memory compiler rules: memory_events are structured simulation deltas only. Do not store the full assistant_text, style prose, atmosphere, repeated facts, or facts already present in Structured simulation memory.",
    omit
      // State extraction is the annotation call's job. Narrative emits only semantic kinds.
      ? "Separate actual events: every memory_events item must have a concise content string; omit the item if you cannot write one. Do not emit memory_kind='state' — a separate annotation step records all state/visual/outfit/pose/scene deltas. If someone saw/heard/learned something, output observation or belief for that character only. Keep uncertain causes as belief/open_thread, not confirmed fact."
      : "Separate actual events from current states. Every memory_events item must include a concise content string; if you cannot write one, omit that memory event. If a current state changes, output memory_kind='state' with state_type and state_value. If someone saw/heard/learned something, output observation or belief for that character only. Keep uncertain causes as belief/open_thread, not confirmed fact.",
    omit
      ? "Memory graph role rules: split one visible beat when needed — event for what happened, relationship for a relationship change, observation/belief for character-specific knowledge. Set actor_id to the acting character, target_id to the relationship/observed target when known, observers to characters who actually perceived it."
      : "Memory graph role rules: split one visible beat into small deltas when needed: event for what happened, state for the affected character or scene, relationship for a relationship change, observation/belief for character-specific knowledge. Set actor_id to the acting or affected registered character, target_id to the relationship/observed target when known, and observers to registered characters who actually perceived it. target_id may refer to an off-stage relationship target, but that does not make the target present in assistant_text or image_cues.",
    // The image-cue tag contract and the outfit/image-state memory rules are authored once in the runtime
    // (system) instruction; do NOT repeat them here. Re-sending the same multi-paragraph rule block in the
    // per-turn context doubled the prompt the CLI subprocess re-processes every call for zero quality gain.
    relationshipMapRulesText
      ? creatorOutputFormat.requiresStatusBlockFromMain
        ? "Relationship/status map rules: persist durable state/relationship changes through memory_events as below. The main prompt explicitly requires a visible status block in assistant_text every turn, so render that creator-required block and do not skip it."
        : "Relationship/status map rules: the relationship map is a SEPARATE tab fed by memory_events, not part of the visible reply. Update character states and relationships ONLY through memory_events. Do not print a status window, stat line, parameter list, or table in assistant_text — assistant_text is pure in-character narrative prose with no status panel appended. This holds even when world lore, presets, or other modules contain tables or mention 상태; only an explicit status-panel directive in the main prompt would change this."
      : undefined,
    relationshipMapRulesText || undefined,
    // createCreatorOutputFormatInstruction is already emitted once in the runtime (system) instruction.
    "Simulation foundation:",
    foundationText,
    "User persona:",
    personaText || "(none)",
    omit ? undefined : "Image prompt user rules:",
    omit ? undefined : imageUserRulesText || "(none)",
    omit ? undefined : "Image tag keyword presets (no roster identity tags):",
    omit ? undefined : imageScenePresetText || "(none)",
    omit ? undefined : "Image generation cadence:",
    omit ? undefined : imageCadenceText,
    omit ? undefined : "Image cue authoring reference for current-scene visible characters only (reference, do not copy verbatim). Keep only scene/composition/shared-action tags in image_cues.base_tags. For a registered character, set its character_id on a character_prompt and author its action/expression/pose plus any outfit change; DynamicChat injects that character's required_identity_tags and current outfit into the same entry, so do not repeat the saved appearance. Use this reference only to disambiguate which character is which and never to borrow another character's appearance:",
    omit ? undefined : imageVisualProfileText || "(none)",
    omit ? undefined : "Ongoing scene/visual state for current-scene characters only. These are the situation's still-active pose/action/interaction/position facts AND ongoing physical-detail facts — injuries, bandages, bruises, blood, bodily fluids, sweat/wetness, dirt, trembling, held props, and damaged/loosened/wet clothing state (who is doing what to whom right now, and what marks/conditions their body and clothing currently show). Treat them as continuity that PERSISTS across cuts and turns: every should_generate=true cue — including a close-up or a single body-region focus — must keep the still-active action/interaction/position tags AND the still-true body/clothing-damage detail tags from here, so the cut shows the character actually performing the ongoing action with the same visible condition, never standing idle and never silently healed or re-clothed. When you emit several cuts in one turn, keep these details consistent across all of them and only progress a detail in the direction the scene moves it (a fresh wound, more sweat, clothing torn further). Drop or change one of these only when the assistant_text or current user action explicitly ends, heals, or changes it this turn:",
    omit ? undefined : imageCurrentStateText || "(none)",
    "Structured simulation memory:",
    structuredMemoryText,
    "Recent transcript:",
    recentTranscriptText || "(none)",
    "Selected prompt modules for this turn:",
    // Modules are operating rules / world reference to APPLY through the scene, not source text to recite. Their
    // wording (including meta/community/system vocabulary in a module body) must shape what happens, but must not
    // be pasted verbatim into assistant_text as if it were narration — express the rule through in-character
    // action, description, and dialogue instead. This curbs module phrasing bleeding into the prose (문맥 간섭).
    moduleText
      ? "Apply each module below as an active rule/reference. Realize its intent through the scene; do NOT quote or paste the module's own wording into assistant_text as narration.\n\n" + moduleText
      : "(none)",
    "Memory/context evidence:",
    evidenceText || "(none)"
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function normalizeRelationshipParameterKey(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/gu, "").replace(/[^\p{L}\p{N}_:-]+/gu, "");
}

// Latest persisted value for a creator-defined relationship parameter, matched by its (canonicalized) state_type.
// Surfaced back to the LLM so it can continue/refresh the dynamic keyword instead of starting blank each turn.
function readLatestRelationshipParameterValue(state: AppState, title: string): string | undefined {
  const targetKey = normalizeRelationshipParameterKey(title);
  if (!targetKey) {
    return undefined;
  }

  for (let index = state.memoryEvents.length - 1; index >= 0; index -= 1) {
    const event = state.memoryEvents[index];
    const stateType = readStateMemoryStateType(event);
    if (stateType && normalizeRelationshipParameterKey(stateType) === targetKey) {
      const value = readStateMemoryValue(event);
      if (value) {
        return truncatePromptText(value, 120, "relationship parameter value");
      }
    }
  }

  return undefined;
}

function createRelationshipMapRulesBlock(state: AppState): string {
  if (!state.relationshipMap?.enabled) {
    return "";
  }

  const activeParameters = (state.relationshipMap.parameters ?? [])
    .filter((parameter) => parameter.enabled && (parameter.title.trim() || parameter.rule.trim()))
    .sort((a, b) => b.priority - a.priority);
  const parameters = activeParameters
    .map((parameter) => {
      const title = parameter.title.trim();
      const current = title ? readLatestRelationshipParameterValue(state, title) : undefined;
      // Qualitative descriptor only — raw numbers must not appear in the narrative call (they bleed
      // into prose as stat labels). Exact numeric values are provided to the annotation call instead.
      const qualDesc = current ? describeRelationshipParameterValue(current) : undefined;
      return `- ${title}: ${parameter.rule.trim() || "현재 진행에 맞게 짧고 안정적인 값으로 갱신한다."}${qualDesc ? ` (현재 상태: ${qualDesc})` : " (현재 상태: 아직 없음)"}`;
    })
    .join("\n");

  return [
    state.relationshipMap.statusPrompt.trim(),
    parameters
      ? [
          // Numeric current values and the state-emit obligation belong to the annotation call, not here.
          "Configured character status parameters (qualitative context; a separate annotation step records exact updated values):",
          parameters,
          // The relationship tab is the ONLY surface for these values. The visible narrative must stay clean prose.
          "Output channel separation: these parameters belong in the relationship tab only. Unless the creator's own main rules/modules explicitly require a visible status panel or table, do NOT print these parameters, their values, a stat line, a status window, or any table/list of them inside assistant_text — assistant_text stays pure in-character narrative prose."
        ]
          .filter((item): item is string => Boolean(item))
          .join("\n")
      : undefined
    // Outfit/status persistence duty (Wearing/StatusTags) has moved to requestTurnAnnotations.
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

// Provides the annotation call with relationship-parameter context including exact numeric current
// values and the per-turn emit obligation. Must NOT appear in the narrative call (digits bleed into prose).
function createAnnotationRelationshipParamsBlock(state: AppState): string {
  if (!state.relationshipMap?.enabled) {
    return "";
  }

  const activeParameters = (state.relationshipMap.parameters ?? [])
    .filter((parameter) => parameter.enabled && (parameter.title.trim() || parameter.rule.trim()))
    .sort((a, b) => b.priority - a.priority);
  const parameterTitles = activeParameters.map((p) => p.title.trim()).filter(Boolean);
  if (parameterTitles.length === 0) {
    return "";
  }

  const parameters = activeParameters
    .map((parameter) => {
      const title = parameter.title.trim();
      const current = title ? readLatestRelationshipParameterValue(state, title) : undefined;
      return `- ${title}: ${parameter.rule.trim() || "현재 진행에 맞게 짧고 안정적인 값으로 갱신한다."}${current ? ` (현재 값: ${current})` : " (현재 값: 아직 없음)"}`;
    })
    .join("\n");

  return [
    "Creator-defined relationship/status parameters — record CHANGED values as state_events this turn:",
    parameters,
    `Emit a state_events item for any of these parameters that CHANGED or became newly relevant this turn: ${parameterTitles.join(", ")}. Skip parameters whose value did not change.`,
    "state_type must be copied EXACTLY (verbatim, including Korean) from the parameter title above — do not translate, abbreviate, or substitute. state_value follows the parameter's rule as a concise durable Korean value. Set actor_id/actor_name to the affected character."
  ].join("\n");
}

function isFoundationContextModule(module: PromptModule): boolean {
  return module.kind === "main_prompt" || module.kind === "world_lore" || module.tokenPolicy === "always";
}

function collectCreatorRulePromptSources(state: AppState, modules: PromptModule[] = []): string {
  const foundationModules = state.modules.filter(
    (module) =>
      module.enabled &&
      module.tokenPolicy !== "disabled" &&
      (module.kind === "main_prompt" ||
        module.kind === "world_lore" ||
        (module.tokenPolicy === "always" && module.kind !== "image_prompt_profile"))
  );
  const merged = new Map<string, PromptModule>();
  for (const module of [...foundationModules, ...modules]) {
    merged.set(module.id, module);
  }
  return [...merged.values()].map((module) => `${module.title}\n${module.body}`).join("\n\n");
}

// Collect ONLY the creator's main prompt module(s). Used to decide whether a visible status panel was explicitly
// requested by the author — world lore / presets / selected modules routinely contain incidental tables and the word
// "상태" without intending a printed status window.
function collectMainPromptSources(state: AppState): string {
  return state.modules
    .filter((module) => module.enabled && module.tokenPolicy !== "disabled" && module.kind === "main_prompt")
    .map((module) => `${module.title}\n${module.body}`)
    .join("\n\n");
}

// Explicit request for a visible status panel/table (not a mere "상태" mention or an incidental pipe-table example).
// NOTE: "상태[^\n]{0,12}(?:출력|표시|표기|노출|보여|appended?)" was deliberately removed. That alternative matched prose
// directions such as "[심리 및 소통의 시각화]: 적들의 상태를 실시간 출력" — a creative instruction to vividly narrate
// character states in flowing prose, not to print a UI status panel. The remaining Korean branch ("상태\s*(?:창|표|…)")
// covers genuine panel nouns (상태창, 상태패널, etc.) without the false-positive risk.
const EXPLICIT_STATUS_PANEL_DIRECTIVE =
  /상태\s*(?:창|표|블록|윈도우|패널|보드)|status\s*(?:window|panel|block|board)|```\s*status|::status\b|\[status\]/iu;

// A PROHIBITION is not a design. "상태창을 출력하지 않는다" names the same nouns as "상태창을 출력한다", so a keyword test
// reads a ban as a request, flips the gate on, and disables the very sanitizer the author asked for. Drop the sentences
// that forbid a format before testing for one.
const FORMAT_DIRECTIVE_NEGATION =
  /하지\s*(?:않|말|마)|않는다|없이|금지|제외|배제|생략|삼가|말\s*것|안\s*된다|do\s*not|don't|never|avoid|without|no\s+(?:status|table|block)/iu;

function stripNegatedFormatDirectives(text: string): string {
  return text
    .split(/(?<=[.!?。])\s+|\n/u)
    .filter((sentence) => !FORMAT_DIRECTIVE_NEGATION.test(sentence))
    .join("\n");
}

// Any explicit visible-format directive the author put in the MAIN prompt: a status panel, a Markdown table (including a
// concrete pipe-table example), a DynamicChat effect block, or a choice block. This is the single gate for whether the
// runtime keeps the model's blocks/tables/status in assistant_text — incidental tables or "상태" mentions in world lore,
// presets, or retrieved modules never count, only the author's own main prompt.
function mainPromptRequiresVisibleFormat(state: AppState): boolean {
  const main = stripNegatedFormatDirectives(collectMainPromptSources(state));
  return (
    EXPLICIT_STATUS_PANEL_DIRECTIVE.test(main) ||
    /마크다운\s*표|markdown\s*table|gfm\s*table|pipe\s*table|표\s*(?:형식|로|를|출력|표시)|\|[^|\n]+\|[^|\n]+\|/iu.test(main) ||
    /```\s*(?:scene|impact|whisper|sfx|status|choice|memory|letter)|::(?:scene|impact|whisper|sfx|status|choice|memory|letter)\b|선택지\s*블록|choice\s*block/iu.test(main)
  );
}

function analyzeCreatorOutputFormat(state: AppState, modules: PromptModule[] = []): CreatorOutputFormatHints {
  const text = stripNegatedFormatDirectives(collectCreatorRulePromptSources(state, modules));
  const requiresMarkdownTables =
    /마크다운\s*표|markdown\s*table|gfm\s*table|pipe\s*table|표\s*(?:형식|로|를)|\|[^|\n]+\|[^|\n]+\|/iu.test(text);
  const requiresStatusBlock = EXPLICIT_STATUS_PANEL_DIRECTIVE.test(text);
  // Explicit, unambiguous request for a visible choice block (choices are not status, so they stay visible).
  const requiresChoiceBlock = /```\s*choice|::choice\b|선택지\s*블록/iu.test(text);
  // Weak signals (a bare "마크다운"/"markdown" mention, "출력 형식", "매 턴 상태를 추적" etc.) that only loosely imply
  // formatting. These commonly appear in prompts that just want state TRACKED, not a panel printed in the narrative.
  const looseVisibleSignal = /마크다운|markdown|출력\s*형식|응답\s*형식|매\s*턴.*(?:상태|선택지)/iu.test(text);
  // When a relationship map is active, status/state lives in its tab. Only explicit table/status/choice requests force
  // a visible block; loose mentions do NOT — otherwise an incidental "상태"/"마크다운" word makes the model dump a
  // status panel into the visible reply (and disables the deterministic sanitizer). Without a relationship map, keep
  // the looser behavior so prompts relying on it still get their formatting.
  const relationshipMapEnabled = Boolean(state.relationshipMap?.enabled);
  const requiresVisibleFormattedOutput =
    requiresMarkdownTables ||
    requiresStatusBlock ||
    requiresChoiceBlock ||
    (!relationshipMapEnabled && looseVisibleSignal);
  // The relationship-map path gates the visible status panel on the MAIN prompt alone: only when the author explicitly
  // asks for a status panel there do we both instruct the model to print it and stop the sanitizer from stripping it.
  const requiresStatusBlockFromMain = EXPLICIT_STATUS_PANEL_DIRECTIVE.test(collectMainPromptSources(state));
  const requiresVisibleFormatFromMain = mainPromptRequiresVisibleFormat(state);

  return {
    requiresVisibleFormattedOutput,
    requiresMarkdownTables,
    requiresStatusBlock,
    requiresStatusBlockFromMain,
    requiresVisibleFormatFromMain
  };
}

// Deterministic guard so the visible narrative stays clean prose even when the model ignores the prompt instruction.
// Unless the creator's MAIN prompt explicitly designs a visible format, assistant_text is enforced as narrative prose:
// DynamicChat effect blocks are unwrapped to their text (or dropped for status), Markdown tables are flattened to prose,
// and any status panel is removed. When the main prompt DOES design a format, the model's structure is left intact.
export function sanitizeAssistantNarrative(state: AppState, modules: PromptModule[], text: string): string {
  if (!text) {
    return text;
  }
  const hints = analyzeCreatorOutputFormat(state, modules);
  if (hints.requiresVisibleFormatFromMain) {
    // The author designed a visible status/table/choice format in the main prompt — keep that content, but never let it
    // arrive as a styled effect block: unwrap the blocks to plain text (nothing is deleted) so narration and dialogue
    // cannot end up trapped inside one. Tables carrying prose are flattened for the same reason, while genuine short
    // field/value status tables the author intended are preserved.
    return flattenMarkdownTablesToProse(stripDynamicEffectBlocks(text, true), true);
  }
  let out = stripDynamicEffectBlocks(text);
  out = flattenMarkdownTablesToProse(out);
  // relationshipMapEnabled=true forces the aggressive trailing table/stat-line removal: in clean-prose mode no status
  // structure is wanted regardless of whether the relationship tab is on.
  out = stripStatusPanelFromNarrative(out, true);
  return out;
}

// DynamicChat effect-block kinds the frontend renders as styled blocks, plus the loose aliases the model tends to use.
// Kept in sync with App.tsx (dynamicTextBlockKinds / dynamicTextBlockAliases).
const EFFECT_BLOCK_KINDS = new Set([
  "scene",
  "impact",
  "whisper",
  "sfx",
  "status",
  "choice",
  "memory",
  "letter",
  "big",
  "large",
  "shout",
  "small",
  "quiet",
  "note",
  "memo",
  "system"
]);
// Blocks whose entire content is removed (not unwrapped) in clean-prose mode: a status panel belongs in the relationship
// tab, and a choice block is a UI affordance rather than narrative prose.
const REMOVED_EFFECT_BLOCK_KINDS = new Set(["status", "choice", "system"]);

function normalizeEffectBlockKind(value: string): string | undefined {
  const normalized = value.trim().toLowerCase();
  return EFFECT_BLOCK_KINDS.has(normalized) ? normalized : undefined;
}

// Remove DynamicChat effect blocks from assistant_text. Fenced ```kind … ```, one-line ::kind[…], and ::kind … :: open
// blocks are recognized. status/choice/system blocks are dropped entirely; every other kind is unwrapped to its inner
// text so the narrative content survives as plain prose.
// keepStructuralContent=true: the creator's main prompt designs a visible status/choice format, so unwrap those blocks
// to plain text instead of deleting them — the author's content survives, but no prose or dialogue stays trapped in a
// styled block.
function stripDynamicEffectBlocks(text: string, keepStructuralContent = false): string {
  const isRemovedKind = (kind: string): boolean => !keepStructuralContent && REMOVED_EFFECT_BLOCK_KINDS.has(kind);
  const lines = text.replace(/\r\n/gu, "\n").split("\n");
  const out: string[] = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];

    const fenceKind = line.match(/^\s*```\s*([\p{L}\p{N}_-]+)\s*$/u);
    const fencedKind = fenceKind ? normalizeEffectBlockKind(fenceKind[1]) : undefined;
    if (fencedKind) {
      const blockLines: string[] = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/u.test(lines[index])) {
        blockLines.push(lines[index]);
        index += 1;
      }
      // index now sits on the closing fence (or end of input); the for-loop ++ skips it.
      if (!isRemovedKind(fencedKind)) {
        out.push(blockLines.join("\n").trim());
      }
      continue;
    }

    const shortDirective = line.match(/^\s*::([\p{L}\p{N}_-]+)\[(.*)\]\s*$/u);
    const shortKind = shortDirective ? normalizeEffectBlockKind(shortDirective[1]) : undefined;
    if (shortKind) {
      if (!isRemovedKind(shortKind)) {
        out.push(shortDirective![2].trim());
      }
      continue;
    }

    const openDirective = line.match(/^\s*::([\p{L}\p{N}_-]+)\s*$/u);
    const openKind = openDirective ? normalizeEffectBlockKind(openDirective[1]) : undefined;
    if (openKind) {
      const blockLines: string[] = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== "::") {
        blockLines.push(lines[index]);
        index += 1;
      }
      if (!isRemovedKind(openKind)) {
        out.push(blockLines.join("\n").trim());
      }
      continue;
    }

    out.push(line);
  }
  return out.join("\n").replace(/\n{3,}/gu, "\n\n").trim();
}

// Flatten every GFM-style Markdown table into plain prose lines. A table is a row of pipe cells immediately followed by a
// delimiter row (| --- | --- |). The header labels and the delimiter are dropped; each body row becomes one prose line:
// a short, unquoted first cell is treated as a speaker/label ("이름: 대사"), otherwise the non-empty cells are joined.
// keepDataTables=false (clean-prose mode): prose/dialogue tables are flattened to prose, pure status/data tables
// are dropped (they belong in the relationship tab). keepDataTables=true (the creator's main prompt designs a
// visible table format): still flatten any table that carries dialogue/narrative — that is always a misuse, the
// user never wants normal conversation trapped in a table — but PRESERVE genuine short field/value status tables
// the author intended.
function flattenMarkdownTablesToProse(text: string, keepDataTables = false): string {
  const lines = text.replace(/\r\n/gu, "\n").split("\n");
  const out: string[] = [];
  let inFence = false;
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^\s*```/u.test(line)) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    const next = lines[index + 1];
    const startsTable =
      !inFence && isPipeTableRow(line) && next !== undefined && isPipeTableDelimiter(next);
    if (!startsTable) {
      out.push(line);
      continue;
    }
    // Consume the header row, the delimiter row, and all contiguous body rows (keep the originals so a genuine
    // status/data table can be re-emitted untouched when keepDataTables is set).
    const headerLine = line;
    const delimiterLine = lines[index + 1];
    const headerCells = readPipeCells(line);
    index += 2;
    const bodyRows: string[][] = [];
    const bodyLines: string[] = [];
    while (index < lines.length && isPipeTableRow(lines[index]) && !isPipeTableDelimiter(lines[index])) {
      bodyRows.push(readPipeCells(lines[index]));
      bodyLines.push(lines[index]);
      index += 1;
    }
    index -= 1; // step back so the for-loop ++ lands on the first non-table line
    // A table that carries dialogue/narrative is flattened to prose so the text survives; a pure status/data table
    // (short label/value cells, no dialogue) is dropped (clean-prose mode) or kept verbatim (creator-format mode).
    const carriesProse = bodyRows.some((row) => row.some((cell) => isProseBearingCell(cell)));
    if (!carriesProse) {
      if (keepDataTables) {
        out.push(headerLine, delimiterLine, ...bodyLines);
      }
      continue;
    }
    for (const row of bodyRows) {
      const flattened = flattenTableRow(row, headerCells.length);
      if (flattened) {
        out.push(flattened);
      }
    }
  }
  return out.join("\n").replace(/\n{3,}/gu, "\n\n").trim();
}

// A table cell that reads like dialogue or narrative prose rather than a short status/data token: it contains quoted
// speech/thought or a long-ish phrase. Short values like "30", "교복", "분노" are data and do not count.
function isProseBearingCell(cell: string): boolean {
  const trimmed = cell.trim();
  if (/["“”「」『』']/u.test(trimmed)) {
    return true;
  }
  return trimmed.replace(/\s+/gu, "").length >= 14;
}

function readPipeCells(line: string): string[] {
  const trimmed = line.trim();
  const inner = trimmed.replace(/^\|/u, "").replace(/\|$/u, "");
  return inner.split("|").map((cell) => cell.trim());
}

function isPipeTableRow(line: string): boolean {
  const trimmed = line.trim();
  return trimmed.includes("|") && readPipeCells(line).some(Boolean);
}

function isPipeTableDelimiter(line: string): boolean {
  const cells = readPipeCells(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{2,}:?$/u.test(cell.replace(/\s+/gu, "")));
}

function flattenTableRow(cells: string[], columnCount: number): string {
  const nonEmpty = cells.filter(Boolean);
  if (nonEmpty.length === 0) {
    return "";
  }
  if (nonEmpty.length === 1) {
    return nonEmpty[0];
  }
  // Two-column rows are usually "speaker | line" or "label | value": render as "speaker: line" when the first cell looks
  // like a short label (no sentence punctuation, no quotes).
  if (columnCount === 2 && nonEmpty.length === 2 && /^[^"“”「」『』'.!?]{1,16}$/u.test(nonEmpty[0])) {
    return `${nonEmpty[0]}: ${nonEmpty[1]}`;
  }
  return nonEmpty.join(" ");
}

function stripStatusPanelFromNarrative(text: string, relationshipMapEnabled: boolean): string {
  let out = text.replace(/\r\n/gu, "\n");

  // Closed status fences: ```status ... ``` (space/case tolerant).
  out = out.replace(/^[ \t]*`{3,}[ \t]*status\b[^\n]*\n[\s\S]*?\n[ \t]*`{3,}[ \t]*$/gimu, "");
  // ::status[ ... ] single-line directive and ::status ... :: block.
  out = out.replace(/^[ \t]*::[ \t]*status[ \t]*\[[^\]]*\][ \t]*$/gimu, "");
  out = out.replace(/^[ \t]*::[ \t]*status[ \t]*\n[\s\S]*?\n[ \t]*::[ \t]*$/gimu, "");
  // [status] ... bracket directive line.
  out = out.replace(/^[ \t]*\[status\][^\n]*$/gimu, "");
  // Dangling, not-yet-closed status fence (e.g. while streaming): cut from the opener to the end.
  out = out.replace(/\n?[ \t]*`{3,}[ \t]*status\b[\s\S]*$/iu, "");

  if (relationshipMapEnabled) {
    // A trailing block introduced by a status-panel header ("## 상태창", "**현재 상태**", "【상태】", "[Status]" …)
    // through the end of the message. Only consumed when what follows reads like a panel, so creative mid-narrative
    // sections survive.
    out = stripTrailingStatusHeaderBlock(out);
    // A trailing GFM table is almost always a status panel here (narrative prose rarely ends in a table). Only consume
    // the contiguous table rows (header, delimiter, body) at the very end, not any prose that might follow.
    out = out.replace(
      /(?:\n|^)[ \t]*\|[^\n]*\|[ \t]*\n[ \t]*\|[ \t]*:?-{2,}[^\n]*\|?[ \t]*\n(?:[ \t]*\|[^\n]*\|?[ \t]*\n?)*$/u,
      ""
    );
    // A trailing run of emoji/stat lines ("❤️ 신뢰도 ...", "💪 체력 ...", "키: 값 | 키: 값").
    // Pipe lines are only treated as stat lines when they contain a colon before the pipe ("key: value | key: value").
    // This prevents "**이름** | "대사"" dialogue lines from being mistakenly stripped as a status panel.
    out = out.replace(
      /(?:\n[ \t]*(?:[\p{Extended_Pictographic}☀-➿][^\n]*|[^\n|]*:[^\n|]*\|[^\n]+)\s*){2,}$/u,
      ""
    );
  }

  return out.replace(/\n{3,}/gu, "\n\n").trim();
}

// Header line that introduces a status panel (markdown heading / bold / bracket / divider markers are tolerated, but the
// status keyword must carry a panel qualifier or be 상태창/스테이터스 — a bare "상태"/"현재 상태" in prose never matches).
// Note: no \b after a Korean keyword — JS word boundaries are ASCII-only and never trigger after a Korean syllable.
// Branch 1: a strong status keyword (상태 + panel qualifier, 스테이터스, or status window/panel/…); markers optional.
// Branch 2: a bare 「상태」 that is clearly delimited as a title by a heading/bold/bracket marker on both sides — this
//           lets 【상태】, [상태], **상태**, and "## 상태" match while a bare 상태 inside prose never does.
const STATUS_PANEL_HEADER_LINE =
  /^[ \t]*(?:(?:[#>*_=─━—~•▶◆■◼□●◇☆★『「【\[(]|\*\*)*[ \t#*_>]*(?:현재\s*)?(?:상태\s*(?:창|표|요약|정보|패널|윈도우|보드|업데이트)|스테이터스|status\s*(?:window|panel|board|update|summary))|(?:#{1,6}[ \t]*|\*\*[ \t]*|[【「『\[][ \t]*|[■◆▶●◇☆★][ \t]*)(?:현재\s*)?상태(?:\s*(?:창|표|요약|정보|패널|윈도우|보드|업데이트))?(?:[ \t]*[】」』\]]|\*\*|[ \t]*$))/iu;

// Strip a trailing status panel that begins with a recognizable status header and runs to the end of the message.
// Conservative: only removes the block when every content line after the header reads like panel structure
// (key:value rows, table/bullet/stat lines), so flowing narrative that merely follows such a header is left intact.
function stripTrailingStatusHeaderBlock(text: string): string {
  const lines = text.split("\n");
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    if (!STATUS_PANEL_HEADER_LINE.test(lines[index])) {
      continue;
    }
    const contentLines = lines.slice(index + 1).filter((line) => line.trim());
    const isPanelTail =
      contentLines.length === 0 || contentLines.every((line) => isStatusPanelLine(line));
    if (isPanelTail) {
      return lines.slice(0, index).join("\n");
    }
    // The nearest status header is followed by real prose — treat it as narrative, not a panel, and stop.
    return text;
  }
  return text;
}

// A single line that looks like part of a status panel rather than narrative prose.
function isStatusPanelLine(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) {
    return true;
  }
  // Horizontal rules / dividers used inside panels.
  if (/^[-=_*─━—~]{2,}$/u.test(trimmed)) {
    return true;
  }
  // Table rows, blockquotes, bullets, or emoji/symbol-led stat lines.
  if (/^(?:\||>|[-*•▶◆■◼□●◇☆★]|\p{Extended_Pictographic})/u.test(trimmed)) {
    return true;
  }
  // "라벨: 값" / "**라벨**: 값" style rows where the label is short (a stat key, not a sentence).
  if (/^\*{0,2}[^:：\n]{1,24}\*{0,2}\s*[:：]\s*\S/u.test(trimmed)) {
    return true;
  }
  return false;
}

function createCreatorOutputFormatInstruction(hints: CreatorOutputFormatHints): string | undefined {
  // Only bind the model to a visible structure when the MAIN prompt actually designs one. Incidental tables or 「상태」
  // mentions in lore/presets/modules must not turn on formatted output (it stays clean prose, enforced by the sanitizer).
  if (!hints.requiresVisibleFormatFromMain) {
    return undefined;
  }

  const lines = [
    "Creator output format (binding): Main rules and selected modules define the visible response structure. Follow that structure inside assistant_text every turn before ending the JSON object.",
    "Creator rules outrank generic brevity, relationship-map storage guidance, and default prose-only narration when they conflict."
  ];

  if (hints.requiresMarkdownTables || hints.requiresStatusBlock) {
    lines.push(
      "When the creator requires tables or a status block, render them as valid GitHub-Flavored Markdown inside assistant_text: use a header row, a delimiter row such as | --- | --- |, then data rows. Do not emit raw pipe text without the delimiter row, ASCII box drawing, or HTML tables.",
      "A table holds only short structured field/value data for the creator-required status/data block. Never put narration, scene description, or spoken dialogue inside table cells — those stay as ordinary prose lines outside the table.",
      "Status/choice blocks may use plain Markdown sections or ```status / ```choice fences, but tables inside them must still use valid GFM pipe-table syntax."
    );
  }

  if (hints.requiresStatusBlock) {
    lines.push(
      "Include the full creator-defined status/parameter block at the end of assistant_text when rules require it. Update values for the current turn instead of omitting the block because state is also stored in memory_events."
    );
  }

  return lines.join("\n");
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

function resolveLlmRequestTimeoutMs(outputTokenBudget: number, options: { isCliAgent?: boolean } = {}): number {
  if (options.isCliAgent) {
    // Cold start/agentic overhead (~120s) + slow per-token streaming (~60ms/token) on a heavy prompt; the
    // old 90s + 30ms/token capped at 300s killed big turns at ~278s mid-stream. The server idle timeout
    // protects against true hangs, so this can be generous.
    const cliScaledTimeout = 120_000 + outputTokenBudget * 60;
    return Math.min(
      MAX_CLI_AGENT_REQUEST_TIMEOUT_MS,
      Math.max(MIN_CLI_AGENT_REQUEST_TIMEOUT_MS, cliScaledTimeout)
    );
  }
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
      "Each cue must advance the current situation by a small step and vary framing naturally across cuts when it fits the continuity."
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
        : "If the assistant_text has any visible scene/action/dialogue/body beat, include at least one should_generate=true cue with concrete English tags. Use [] only when the current response is impossible to visualize or purely administrative."
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
    ? "In adult_19 mode, when the image prompt user rules or situational tag presets call for an explicit beat, emit the direct visual tags they describe; do not blank out image_cues merely because the scene is adult-only."
    : "Do not add sexual or explicit body-contact tags unless the active content mode and the user rules allow them.";
  return [
    // Format skeleton: enough structure for the NovelAI pipeline to function, no creative tag opinions.
    "Image cue tag format: image_cues.base_tags and every image_cues.character_prompts[].prompt must be final comma-separated English NovelAI tags, not prose, sentences, summaries, headings, or labels.",
    "Keep each image_cue COMPACT to save generation time: emit ONLY kind, placement, anchor_text, should_generate, characters, base_tags, and character_prompts. Do NOT emit label, reason, priority, scene, visual_context, or the legacy flat `tags` field — DynamicChat does not need them and they only slow the response. Use base_tags + per-character character_prompts as the single source of tags (do not also duplicate them into a flat `tags` field). Add suppression_reason ONLY when should_generate=false.",
    "Never emit should_generate=true with empty visual tags (empty base_tags and empty character_prompts). If you cannot produce concrete tags for a beat, set should_generate=false with a short suppression_reason.",
    "Do not put artist, style, quality, resolution, or negative/undesired tags in image_cues; DynamicChat adds those separately. Exclude tags like highres, absurdres, masterpiece, best quality, lowres, watermark.",
    // Structural V4 split + per-character separation (always enforced, even without user rules).
    "NovelAI V4 prompt split: image_cues.base_tags carries ONLY non-character tags (scene, location, environment, camera/framing, composition, lighting, props, weather, and whole-cut shared staging). Every visible human subject goes in image_cues.character_prompts, one separate entry per character. Never put a person tag in base_tags: subject count (1girl, 1boy), gender/body type (muscular man), anatomy, pose, expression, clothing, or per-character action must never appear in base_tags.",
    "One character_prompt per visible character: emit a separate entry for EACH human actually visible IN THIS CUT'S FRAME, including secondary, aggressor, background-but-visible, or unregistered characters. Never merge two characters into a single entry, and never describe a second character inside base_tags or inside another character's entry. List in image_cues.characters only the characters you render this cut (omit present-but-off-frame characters such as the observer the shot looks through), and include one matching character_prompts item for every id you list.",
    "Focus vs. incidental detail (tag budget): a cut has 1-2 focus subjects — the figure(s) the framing and action are actually about — and sometimes additional incidental, background, or crowd figures. The focus subject gets detail, but ONLY about whatever the chosen crop actually reveals for it — 'full detail' means thorough about what is in frame, NOT the whole body, face, and outfit every time. Match the detail to the framing: a face/expression close-up gets gaze/mouth/expression detail and skips body/pose/full-outfit tags; a from-behind or body-region crop gets the back/region/pose detail and skips facing-camera face tags; only a full-body shot warrants pose plus body-state plus visible outfit. Render incidental/background/crowd figures COMPACTLY: a subject-count plus role plus only the few tags the shot needs (e.g. `multiple boys, soldiers, surrounding, leering`), not a full per-character breakdown for each one. Do not spend detailed tags on figures the cut does not center on, and do not pile every possible state tag onto the focus subject either — an over-stuffed prompt where everything is described in full dilutes the subject and breaks the image. Each character_prompt should carry only the tags the chosen crop reveals for that character, ordered most-defining-first.",
    "Characters in physical contact STAY SEPARATE entries: two people touching, overlapping, grappling, embracing, carrying, pinning, or in a sexual position are still two character_prompts (char_caption[0], char_caption[1], ...), never one merged entry. Author the ACTIVE/doing participant's own body action in their entry (what their hands, hips, mouth, and body are doing) and the PASSIVE/receiving participant's own pose, body orientation, and reaction in THEIR entry — do not pack both characters' bodies and the whole interaction into a single character's entry and leave the other entry empty or appearance-only. The joint position/contact anchor (the tag naming the position itself) goes on each participant's entry per the user-rule subject/target convention so both figures actually perform the same interaction. If image_cues.characters lists two ids, image_cues.character_prompts MUST contain two non-empty matching entries; if you can only describe one body, fix the cue rather than merging.",
    "Within each character_prompt, order that character's current action/pose/interaction/expression tags first.",
    "Do NOT repeat injected or unchanged tags — this is the biggest token waste on multi-cut turns. DynamicChat already injects, into EVERY cut from saved state, each character's appearance/identity, current outfit, ongoing pose/action/interaction, and persisted physical condition (injuries, blood, bodily fluids, sweat, dirt, restraints). So a character_prompt must contain ONLY what is NEW or specific to THIS cut: the chosen framing/crop and this beat's changed action/expression/contact. Do not re-list the scene, the outfit, the wounds/blood/fluids, or the still-ongoing action that already holds — they are injected automatically. When a turn emits several cuts of the same ongoing scene, write each later cut as a SHORT delta from the previous one (only what the camera or the action changed), not a fresh full re-description; keep base_tags minimal and only restate a scene tag when that element actually changed. (New durable condition still goes into memory_events as PhysicalStateTags/Wearing so the next cut/turn keeps injecting it — record it once there, not in every cut.)",
    "Composition-first authoring: before choosing tags, decide the camera/framing (e.g., close up, cowboy shot, full body, over-the-shoulder, pov) and what the cut actually shows. Only emit tags for what is visible in that frame. Do not add clothing, accessories, jewelry, or background props the chosen composition would not show, and do not paste the saved default outfit when the cut is a close-up of a specific region.",
    "Framing is mandatory: every should_generate cue MUST commit to exactly one explicit shot/framing tag (e.g., close-up, face focus, upper body, cowboy shot, full body, wide shot, from behind, pov) so the crop is decided, not left ambiguous. Put the shared shot/framing in base_tags and any per-character viewpoint detail in that character's entry.",
    "Face/expression coherence: decide framing, viewpoint, and whether the face is shown autonomously from each character's actual pose and orientation — there is no fixed 'always show expression or always crop' rule. When the cut genuinely shows a character's face toward the viewer, include a matching expression (eyes/mouth/emotion) so it is not blank. When the pose means the face is not meaningfully visible (turned around, from behind, looking away, head outside the chosen crop, or obscured), do not force an expression or facing-camera face tags — instead use the viewpoint tags that fit (from behind, facing away, etc.). Match the tags to what the chosen composition actually reveals.",
    "Outfit follows context and composition, not chance: take the character's current Wearing state as the baseline and change it only when the narrative changed it (removed/added/torn/wet/displaced) or when the chosen crop only reveals part of it. Do not randomly swap, drop, or re-add garments between cuts that the story did not change; keep the established outfit stable turn to turn unless the scene altered it.",
    "Whole-situation first, then the focus: before authoring any cut, recall the ongoing scene action/interaction from the ongoing scene/visual state and the recent transcript, and build the cut on top of it. A cut that narrows to a moment or a body region is still the SAME ongoing situation, not a fresh neutral pose. Always pair the narrow focus with the still-active action/position/interaction tags so the composition stays coherent and the image does not collapse into a character standing idle while a detail floats in isolation. This applies to every action, not only sexual ones. Example (non-sexual): if a man is strangling a character and the next beat focuses on her body/neck, the cut still needs the interaction tags (e.g., strangling, hands on another's neck, choking, struggling) on the appropriate character_prompts plus the matching position — never emit just the body-region focus on a calmly standing figure. Example (sexual): if the scene action is vaginal penetration and you emphasize the hips, the cut still needs the insertion/position tags (e.g., sex from behind, vaginal, penis, penetration, hetero). Do not strand a focus shot away from the action it belongs to.",
    "Cross-turn action continuity: the situation does not reset between turns. Before writing this turn's cues, read the ongoing scene/visual state and the immediate continuity anchor for the action/position/interaction that was active at the END of the previous turn, and carry it into THIS turn's first cue unless the new assistant_text or current user action explicitly ends or changes it. A continuing physical action (e.g., still strangling, still pinned, still embracing, still running) must keep appearing in the tags across consecutive turns until it stops — do not silently drop it just because a new turn started. When such an ongoing physical action/interaction is present, also persist it as a memory_events state (state_type='ActionTags' or 'InteractionTags', state_value as the English action/interaction tags) so the next turn can continue it; do this even for actions whose wording is unusual, since only persisted state survives to the next turn.",
    "Registered vs unregistered characters: for a registered character, set character_id and author ONLY that character's current action, pose, expression, interaction, and the cut's framing/visible body state. DynamicChat injects that character's saved base appearance and current outfit (hair/eyes/face/body plus the current Wearing garments) into the same entry, so do NOT reproduce identity tags and do NOT restate the established outfit — that is exactly what keeps the character looking like the same person in the same clothes turn after turn. Only write a clothing tag yourself when THIS turn's narrative changed it (removed/torn/wet/displaced/clothed sex) or when the chosen close-up reveals a specific region; if the cut is fully nude, write the explicit nudity tag (nude/completely nude) and DynamicChat will skip the outfit injection. For an unregistered visible subject, omit character_id and author its full description (appearance, body, outfit, action) yourself.",
    "Unregistered NPC in interaction scenes: when an enemy, aggressor, bystander, crowd figure, or any person absent from the registered roster is physically interacting with a registered character (grabbing, restraining, attacking, embracing, penetrating, etc.), create two SEPARATE character_prompts entries — one entry for the registered character (carrying that character's character_id and describing only their own body/pose/reaction) and one entry for the unregistered participant (no character_id, full self-description: role, body, action). The registered character's entry must NOT be prefixed with the unregistered figure's role tags — each entry describes only its own subject. Never assign a registered character's id to the unregistered figure's entry, even when the unregistered figure is the active/doing participant.",
    "Visible-only clothing: name only the garments and body state that the chosen composition actually shows. If the character is fully nude in this cut, write the explicit nudity tag (e.g., nude, completely nude). If only the SAME garment changes condition (torn, wet, lifted, aside, clothed sex), write the partial-clothing tag plus the garment that is still on. If the cut is a close-up where the saved default outfit is not in frame, do not paste it in just to fill space.",
    "When the image prompt user rules define a subject/object action convention (for example a source/target/mutual scheme), apply it by writing each character's portion of the interaction inside that character's own character_prompt entry; keep a shared core pose/position tag only where the rules place it.",
    // Structural identity correctness.
    "Character identity lock: never mix one roster character's hair, eyes, outfit, or body tags into another character's entry, and do not borrow another character's required_identity_tags. Subject-count tags must match the visible subjects; a single character must not become 2girls/3girls.",
    "Character ambiguity rule: when a pronoun or unnamed continuation could match more than one roster character and the current scene does not disambiguate, reuse the character already visible in the latest image/assistant beat, or set should_generate=false with a suppression_reason instead of guessing.",
    // Content authority lives in user rules + situational presets, not in DynamicChat.
    "For which tags to use and how to phrase them, follow the image prompt user rules and the situational tag keyword presets. DynamicChat does not impose its own tag vocabulary, tag ordering, scene styling, or example tags beyond this structural split.",
    "Prefer concrete, render-able visual tags over abstract psychological or medical state words. NovelAI cannot draw a concept like `panic attack`, `hyperventilation`, or `anxiety`; express the same beat through the visible cues that show it (e.g. wide eyes, open mouth, gasping, trembling, tears, pale face, sweat). Name the abstract state only if the user rules or presets explicitly use it as a tag.",
    adultExplicitInstruction
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
  return llmProviderScheduler.run(state.llm.provider, state.llm.apiKey, () =>
    requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, options)
  );
}

async function requestProviderTextImmediate(
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
  const cliAgentKind = cliAgentKindForProvider(state.llm.provider);
  const timeoutMs = resolveLlmRequestTimeoutMs(providerOutputTokenBudget, {
    isCliAgent: Boolean(cliAgentKind)
  });

  if (cliAgentKind) {
    const bridgeUrl = baseUrl || getLlmCliAgentProxyUrl();
    const wantStream = Boolean(options.onRawText);
    const response = await fetchWithTimeout(bridgeUrl, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        agent: cliAgentKind,
        model,
        temperature,
        maxTokens: providerOutputTokenBudget,
        systemPrompt: runtimeInstruction,
        prompt: `${contextBlock}\n\nUser action:\n${input.userText}`,
        stream: wantStream
      })
    }, timeoutMs);

    if (!response.ok) {
      const errorBody = await response.text().catch(() => "");
      // A streaming attempt can fail because the agent's streaming flags are unsupported on this
      // install (e.g. an older claude without --include-partial-messages). Fall back once to the
      // buffered, non-streaming path so the turn still completes instead of hard-failing.
      if (wantStream) {
        return requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          onRawText: undefined
        });
      }
      throwProviderHttpError(state, response, `${cliAgentKind} CLI`, errorBody);
    }

    // Only consume the body as a stream when the bridge actually streamed it
    // (text/plain). A server that ignored `stream` and returned the JSON wrapper
    // ({text, agent, model}) must be parsed as JSON, otherwise the whole wrapper
    // leaks into the chat as raw text.
    const responseContentType = response.headers.get("content-type") ?? "";
    const isStreamingResponse = responseContentType.includes("text/plain") || responseContentType.includes("text/event-stream");
    if (wantStream && options.onRawText && isStreamingResponse) {
      try {
        return await readCliAgentStreamText(response, options.onRawText);
      } catch {
        // Streaming transport failed; retry once without streaming so the turn still completes.
        return requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          onRawText: undefined
        });
      }
    }

    const data = (await response.json()) as { text?: string; error?: string };
    if (data.error) {
      throw new Error(`${cliAgentKind} CLI agent error: ${data.error}`);
    }
    const content = data.text?.trim();
    if (!content) {
      throw new Error(`${cliAgentKind} CLI agent returned no content.`);
    }
    return content;
  }

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
      const errorBody = await response.text().catch(() => "");
      if (isRateLimitHttpStatus(response.status)) {
        throwProviderHttpError(state, response, "Gemini", errorBody);
      }
      if (options.onRawText) {
        return requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          onRawText: undefined
        });
      }
      throwProviderHttpError(state, response, "Gemini", errorBody);
    }

    if (options.onRawText) {
      const streamed = await readGeminiStreamText(response, options.onRawText).catch(() => undefined);
      if (streamed?.trim()) {
        return streamed;
      }
      return requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
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
      if (isRateLimitHttpStatus(response.status)) {
        throwProviderHttpError(state, response, "Claude");
      }
      if (options.onRawText) {
        return requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          onRawText: undefined
        });
      }
      throwProviderHttpError(state, response, "Claude");
    }

    if (options.onRawText) {
      const streamed = await readClaudeStreamText(response, options.onRawText).catch(() => undefined);
      if (streamed?.trim()) {
        return streamed;
      }
      return requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
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
    if (isRateLimitHttpStatus(response.status)) {
      throwProviderHttpError(state, response, "LLM");
    }
    if (options.onRawText) {
      return requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
        ...options,
        onRawText: undefined
      });
    }
    throwProviderHttpError(state, response, "LLM");
  }

  if (options.onRawText) {
    const streamed = await readOpenAiCompatibleStreamText(response, options.onRawText).catch(() => undefined);
    if (streamed?.trim()) {
      return streamed;
    }
    return requestProviderTextImmediate(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
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

function combineRawTextHandlers(
  ...handlers: Array<((rawText: string) => void) | undefined>
): ((rawText: string) => void) | undefined {
  const active = handlers.filter((handler): handler is (rawText: string) => void => Boolean(handler));
  if (active.length === 0) {
    return undefined;
  }
  if (active.length === 1) {
    return active[0];
  }
  return (rawText) => {
    for (const handler of active) {
      handler(rawText);
    }
  };
}

// Slices the complete value of the FIRST top-level array field `key` out of a partial JSON stream,
// using brace/bracket/string-aware scanning. Returns undefined until the array has closed.
function extractCompleteLeadingArray(raw: string, key: string): string | undefined {
  const unwrapped = unwrapCliAgentBridgeResponse(raw);
  const keyPattern = new RegExp(`["']?${key}["']?\\s*:\\s*\\[`, "u");
  const match = keyPattern.exec(unwrapped);
  if (!match) {
    return undefined;
  }
  const start = match.index + match[0].length - 1; // position of the opening '['
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < unwrapped.length; i++) {
    const char = unwrapped[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (inString) {
      if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
    } else if (char === "[") {
      depth += 1;
    } else if (char === "]") {
      depth -= 1;
      if (depth === 0) {
        return unwrapped.slice(start, i + 1);
      }
    }
  }
  return undefined;
}

// Slices the complete value of the FIRST top-level object field `key` out of a partial JSON stream, using
// brace/string-aware scanning. Returns undefined until the object has closed.
function extractCompleteLeadingObject(raw: string, key: string): string | undefined {
  const unwrapped = unwrapCliAgentBridgeResponse(raw);
  const keyPattern = new RegExp(`["']?${key}["']?\\s*:\\s*\\{`, "u");
  const match = keyPattern.exec(unwrapped);
  if (!match) {
    return undefined;
  }
  const start = match.index + match[0].length - 1; // position of the opening '{'
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < unwrapped.length; i++) {
    const char = unwrapped[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === "\\") {
      escaped = true;
      continue;
    }
    if (inString) {
      if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
    } else if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return unwrapped.slice(start, i + 1);
      }
    }
  }
  return undefined;
}

// The streaming opening cut: parse the leading `first_image_cue` object the instant it closes, so the first image
// dispatches while the narrative is still being written. parseAssistantSidecar folds it in as image cue index 0.
function extractEarlyFirstImageCueDrafts(raw: string): AssistantImageCueDraft[] | undefined {
  const objectSlice = extractCompleteLeadingObject(raw, "first_image_cue");
  if (!objectSlice) {
    return undefined;
  }
  try {
    // Non-empty placeholder assistant_text: parseAssistantSidecar drops the entire sidecar when assistant_text is
    // empty, and this streaming probe only needs the cue.
    const parsed = parseAssistantSidecar(`{"first_image_cue": ${objectSlice}, "assistant_text": "_"}`);
    const cues = parsed.sidecar?.imageCues ?? [];
    return cues.some((cue) => cue.shouldGenerate) ? cues : undefined;
  } catch {
    return undefined;
  }
}

// Backward-compatible fallback: if the model front-loads the full image_cues array instead of first_image_cue,
// dispatch as soon as that array closes.
function extractEarlyImageCueDrafts(raw: string): AssistantImageCueDraft[] | undefined {
  const arraySlice = extractCompleteLeadingArray(raw, "image_cues");
  if (!arraySlice) {
    return undefined;
  }
  try {
    // Non-empty placeholder assistant_text (see extractEarlyFirstImageCueDrafts): an empty value makes
    // parseAssistantSidecar return no sidecar, which silently disabled early dispatch entirely.
    const parsed = parseAssistantSidecar(`{"image_cues": ${arraySlice}, "assistant_text": "_"}`);
    const cues = parsed.sidecar?.imageCues ?? [];
    // Only dispatch early when there is at least one cut to actually render. Pass the full array (not just
    // the generating ones) so each cue keeps the same position-based cueIndex the post-turn pass assigns.
    return cues.some((cue) => cue.shouldGenerate) ? cues : undefined;
  } catch {
    return undefined;
  }
}

function createEarlyImageCueEmitter(onEarlyImageCues: (cues: AssistantImageCueDraft[]) => void): (rawText: string) => void {
  let fired = false;
  return (rawText) => {
    if (fired) {
      return;
    }
    const cues = extractEarlyFirstImageCueDrafts(rawText) ?? extractEarlyImageCueDrafts(rawText);
    if (cues) {
      fired = true;
      onEarlyImageCues(cues);
    }
  };
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

async function readCliAgentStreamText(response: Response, onRawText: (rawText: string) => void): Promise<string | undefined> {
  if (!response.body) {
    const text = (await response.text()).trim();
    if (text) {
      onRawText(text);
    }
    return text || undefined;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let accumulated = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    accumulated += decoder.decode(value, { stream: true });
    onRawText(accumulated);
  }
  accumulated += decoder.decode();
  return accumulated.trim() ? accumulated : undefined;
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
    if (isLlmRateLimitError(error)) {
      throw error;
    }

    if (state.llm.provider === "gemini" && isGeminiOffSafetySettingRejected(error)) {
      try {
        return await requestProviderText(state, input, runtimeInstruction, contextBlock, outputTokenBudget, {
          ...options,
          geminiSafetyThreshold: "BLOCK_NONE"
        });
      } catch (blockNoneError) {
        if (isLlmRateLimitError(blockNoneError)) {
          throw blockNoneError;
        }
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
  if (isLlmRateLimitError(originalError) || isLlmProviderCoolingDown(state)) {
    throw originalError instanceof Error ? originalError : new Error(String(originalError));
  }

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
    if (isLlmRateLimitError(recoveryError)) {
      throw recoveryError;
    }

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

// ── Separate image-cue LLM ──────────────────────────────────────────────────────────────────────────
// The narrative LLM no longer authors image tags (that tag-rule weight degraded the prose). This dedicated call
// receives the ALREADY-WRITTEN narrative + the full visual context (profiles, current state, scene cast, user
// rules, presets, cadence) and produces ONLY image_cues. It can run on a cheaper model (state.imageTagLlm).

function isTurnImageCueGenerationActive(state: AppState, manualImage: boolean): boolean {
  if (manualImage) {
    return true;
  }
  if (isImageProgressionCadence(state)) {
    return true;
  }
  if (!state.imageProfile.enabled || !state.simulation.realtimeImageEnabled) {
    return false;
  }
  const mode = state.imageProfile.triggerMode;
  // realtime + realtime_confirm generate cues every turn (confirm mode still needs cues ready to confirm).
  // stored_only and pure manual mode do not auto-generate, so skip the (paid) image-cue call there.
  return mode !== "stored_only" && mode !== "manual";
}

function resolveImageTagLlmState(state: AppState): AppState {
  return state.imageTagLlm?.enabled ? { ...state, llm: state.imageTagLlm } : state;
}

function createImageCueLlmInstruction(state: AppState, outputTokenBudget: number): string {
  return [
    "You are DynamicChat's image-cue planner. You are given the ALREADY-WRITTEN Korean narrative for this turn (assistant_text) plus the current visual state. Your only job is to output image_cues: the ordered list of NovelAI image cuts that illustrate that narrative.",
    "Do NOT rewrite, translate, summarize, continue, or comment on the narrative. Do NOT output assistant_text or memory_events. Output ONLY a JSON object with an image_cues array.",
    createContentRatingInstruction(state),
    isImageProgressionCadence(state)
      ? `Image progression mode: output exactly ${IMAGE_PROGRESSION_CUE_TARGET} ordered should_generate=true cue objects.`
      : "Place a cut at every major visual beat of the narrative INCLUDING the opening beat — dialogue lines, action/interaction beats, body-detail beats, pose/outfit/state changes, and camera/location changes each deserve their own cue. Set should_generate=false (with a short suppression_reason) only for a genuinely non-visual beat; use [] only if the whole turn is non-visual.",
    "Set each cue's anchor_text to a short phrase copied verbatim from the narrative where the cut belongs, so the image is placed at the right beat.",
    `Image generation cadence is binding: ${createImageGenerationCadenceBlock(state, outputTokenBudget)}`,
    createImageCueTagContractInstruction(state),
    createInitialImageCueCountInstruction(state, outputTokenBudget),
    "Return valid JSON only, no Markdown outside the JSON object."
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n");
}

function createImageCueLlmContext(
  state: AppState,
  input: { userText: string; assistantText: string; manualImage?: boolean; outputTokenBudget?: number }
): string {
  const sceneBriefingText = createImageSceneBriefingBlock(state, input.userText);
  const imageCueShape = {
    kind: "scene | action | body_detail | dialogue_face | context | interaction",
    placement: "before | after | inline",
    anchor_text: "a short phrase copied verbatim from the narrative, marking where this cut belongs",
    should_generate: true,
    characters: ["visible registered character ids"],
    base_tags: ["base prompt tags only: artist-free scene, camera, composition, location, props, action shared by the cut"],
    character_prompts: [
      {
        character_id: "registered visible character id — omit entirely for any figure not in the registered roster (unregistered NPC, enemy, bystander, crowd member); NEVER assign a registered character's id to a different person's caption (e.g. do not put a registered female character's id on a male aggressor's entry)",
        prompt: "this character's CURRENT action, pose, expression, interaction, framing/visible-body-state tags only — DynamicChat injects the saved appearance and current outfit, so do not restate identity or the established outfit; write a garment tag only when this turn changed it or the crop reveals a region",
        negative_prompt: "optional character-specific negative tags"
      }
    ]
  };
  return [
    "Plan image_cues for the following already-written turn. Do not change the narrative.",
    "Already-written narrative (assistant_text) — anchor your cuts to phrases in this exact text:",
    input.assistantText || "(empty)",
    "Current user action this turn:",
    input.userText || "(none)",
    "Current scene cast guard:",
    createSceneCastPromptBlock(state, input.userText),
    sceneBriefingText ? "Current scene briefing (cast count, identities, current outfit/action/condition, who-acts-on-whom):" : undefined,
    sceneBriefingText || undefined,
    "Current turn image generation policy:",
    createCurrentTurnImagePolicyBlock(state, { manualImage: input.manualImage, outputTokenBudget: input.outputTokenBudget }),
    "Image prompt user rules:",
    createImageUserRulesBlock(state) || "(none)",
    "Image tag keyword presets (no roster identity tags):",
    createImageSceneTagPresetBlock(state, input.userText) || "(none)",
    "Image cue authoring reference for current-scene visible characters only (reference, do not copy verbatim). Keep only scene/composition/shared-action tags in base_tags. For a registered character set its character_id on a character_prompt and author its action/expression/pose plus any outfit change; DynamicChat injects that character's required_identity_tags and current outfit, so do not repeat the saved appearance:",
    createImageCueVisualProfileBlock(state, input.userText) || "(none)",
    "Ongoing scene/visual state for current-scene characters only (still-active pose/action/interaction/position + physical-detail facts; persists across cuts and turns unless the narrative changed it):",
    createImageCueCurrentStateBlock(state, input.userText) || "(none)",
    "Return JSON only. The JSON schema is:",
    JSON.stringify({ image_cues: [imageCueShape] })
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function parseImageCuesResponse(raw: string): PlannedImageCueDraft[] {
  const errors: string[] = [];
  const jsonText = extractJsonObject(unwrapCliAgentBridgeResponse(raw));
  if (jsonText) {
    try {
      const value = JSON.parse(jsonText) as Record<string, unknown>;
      const cues = normalizeAssistantImageCueDrafts(value.image_cues ?? value.imageCues ?? value, errors);
      if (cues.length > 0) {
        return cues;
      }
    } catch {
      /* fall through to lenient recovery */
    }
  }
  return extractRecoverableImageCueDrafts(raw, errors);
}

// Authors the turn's image_cues with the dedicated image-cue LLM, given the already-written narrative.
// Returns [] when image generation is not active for this turn or the image LLM is unconfigured.
export async function requestTurnImageCues(
  state: AppState,
  input: { userText: string; assistantText: string; manualImage?: boolean }
): Promise<AssistantImageCueDraft[]> {
  // Temporary diagnostics: this function silently returned [] on every failure/gate, so a missing image had no
  // visible cause. Each return path now logs why. Filter the browser console by "[image-cue]".
  if (!isTurnImageCueGenerationActive(state, Boolean(input.manualImage))) {
    console.warn("[image-cue] skipped: turn image-cue generation not active", {
      imageProfileEnabled: state.imageProfile.enabled,
      realtimeImageEnabled: state.simulation.realtimeImageEnabled,
      triggerMode: state.imageProfile.triggerMode,
      manualImage: Boolean(input.manualImage)
    });
    return [];
  }
  const llmState = resolveImageTagLlmState(state);
  const requiresApiKey = !isCliAgentLlmProvider(llmState.llm.provider);
  if (!llmState.llm.enabled || llmState.llm.provider === "mock" || (requiresApiKey && !llmState.llm.apiKey.trim())) {
    console.warn("[image-cue] skipped: image-tag LLM disabled/unconfigured", {
      provider: llmState.llm.provider,
      enabled: llmState.llm.enabled,
      requiresApiKey,
      hasApiKey: Boolean(llmState.llm.apiKey.trim())
    });
    return [];
  }
  const outputTokenBudget = Math.min(resolveInteractiveOutputTokenBudget(state), 1800);
  const instruction = createImageCueLlmInstruction(state, outputTokenBudget);
  const context = createImageCueLlmContext(state, { ...input, outputTokenBudget });
  try {
    const raw = await requestProviderText(
      llmState,
      { userText: input.userText, modules: [], evidence: [] },
      instruction,
      context,
      outputTokenBudget,
      { temperature: llmState.llm.temperature }
    );
    if (!raw?.trim()) {
      console.warn("[image-cue] LLM returned empty content", { provider: llmState.llm.provider, model: llmState.llm.model });
      return [];
    }
    const cues = parseImageCuesResponse(raw);
    if (cues.length === 0) {
      console.warn("[image-cue] parsed 0 cues from LLM output", { rawPreview: raw.slice(0, 400) });
    } else {
      console.info(`[image-cue] parsed ${cues.length} cue(s)`, { shouldGenerate: cues.filter((c) => c.shouldGenerate).length });
    }
    return cues;
  } catch (error) {
    console.warn("[image-cue] LLM call failed", { provider: llmState.llm.provider, model: llmState.llm.model, error: error instanceof Error ? error.message : String(error) });
    return [];
  }
}

// Result of the post-narrative annotation call.
// state_events: memory_kind='state' deltas (Wearing/pose/scene + relationship params) — always extracted.
// image_cues:   authored only when isTurnImageCueGenerationActive is true.
export interface TurnAnnotationResult {
  imageCues: AssistantImageCueDraft[];
  stateEvents: AssistantMemoryEventDraft[];
}

// Instruction for the annotation LLM. State-event extraction runs every turn; image_cues authoring is
// conditional on includeImageCues so image-off turns still get Wearing/pose/param state extraction.
function createAnnotationLlmInstruction(state: AppState, outputTokenBudget: number, includeImageCues: boolean): string {
  return [
    includeImageCues
      ? "You are DynamicChat's annotation planner. Given the already-written Korean narrative, output BOTH state_events (required every turn) and image_cues (for this visual turn). Do NOT rewrite or comment on the narrative."
      : "You are DynamicChat's state extractor. Given the already-written Korean narrative, extract state_events that capture what changed this turn. Do NOT output image_cues and do NOT rewrite or comment on the narrative.",
    createContentRatingInstruction(state),
    includeImageCues
      ? [
          isImageProgressionCadence(state)
            ? `Image progression mode: output exactly ${IMAGE_PROGRESSION_CUE_TARGET} ordered should_generate=true image_cues.`
            : "For image_cues: place a cut at every major visual beat INCLUDING the opening beat. Set should_generate=false only for a genuinely non-visual beat.",
          "Set each cue's anchor_text to a short phrase copied verbatim from the narrative.",
          `Image generation cadence is binding: ${createImageGenerationCadenceBlock(state, outputTokenBudget)}`,
          createImageCueTagContractInstruction(state),
          createInitialImageCueCountInstruction(state, outputTokenBudget)
        ]
          .filter((item): item is string => Boolean(item))
          .join("\n")
      : undefined,
    "For state_events: capture ONLY what CHANGED this turn. Emit memory_kind='state' items for outfit changes (state_type='Wearing', state_value=English NovelAI tags), pose/action/interaction changes (state_type='PoseTags'/'ActionTags'/etc.), physical-state or scene changes, and any relationship-parameter changes. state_value for image-state types is comma-separated English tags (3-8 compact). Set actor_id for character-specific state; leave empty for whole-scene state. Do not re-emit unchanged state.",
    "Outfit/status persistence: if clothing changes, emit state_type='Wearing' preserving base garments and appending damage/condition tags (e.g. `police uniform, torn uniform`). If a character gains important visual state, emit the matching state_type.",
    "Return valid JSON only, no Markdown outside the JSON object."
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n");
}

// Context for the annotation LLM. Provides the narrative, scene cast, visual state, and (when active)
// image-cue authoring references. Always includes the relationship-parameter block with numeric values.
function createAnnotationLlmContext(
  state: AppState,
  input: { userText: string; assistantText: string; manualImage?: boolean; outputTokenBudget?: number },
  includeImageCues: boolean
): string {
  const sceneBriefingText = createImageSceneBriefingBlock(state, input.userText);
  const annotationRelParamsBlock = createAnnotationRelationshipParamsBlock(state);
  const stateEventShape: Record<string, unknown> = {
    memory_kind: "state",
    state_type: `${IMAGE_STATE_TYPE_INSTRUCTION} — or the verbatim Korean title of a creator-defined relationship parameter`,
    state_value: "English NovelAI tags (image state types) or Korean value (relationship params)",
    actor_id: "optional character id",
    actor_name: "optional character name",
    content: "짧은 한국어 상태 설명",
    importance: 0.8,
    confidence: 0.9,
    tags: ["state", "visual"]
  };
  const imageCueShape = {
    kind: "scene | action | body_detail | dialogue_face | context | interaction",
    placement: "before | after | inline",
    anchor_text: "a short phrase copied verbatim from the narrative",
    should_generate: true,
    characters: ["visible registered character ids"],
    base_tags: ["scene, camera, composition, location, props, shared-action tags"],
    character_prompts: [
      {
        character_id: "registered visible character id",
        prompt: "this character's CURRENT action, pose, expression, interaction, framing/body-state tags",
        negative_prompt: "optional"
      }
    ]
  };

  return [
    "Extract state_events" + (includeImageCues ? " and plan image_cues" : "") + " for the following already-written turn. Do not change the narrative.",
    "Already-written narrative (assistant_text):",
    input.assistantText || "(empty)",
    "Current user action this turn:",
    input.userText || "(none)",
    "Current scene cast guard:",
    createSceneCastPromptBlock(state, input.userText),
    sceneBriefingText ? "Current scene briefing (cast count, identities, current outfit/action/condition, who-acts-on-whom):" : undefined,
    sceneBriefingText || undefined,
    annotationRelParamsBlock || undefined,
    includeImageCues ? "Current turn image generation policy:" : undefined,
    includeImageCues ? createCurrentTurnImagePolicyBlock(state, { manualImage: input.manualImage, outputTokenBudget: input.outputTokenBudget }) : undefined,
    includeImageCues ? "Image prompt user rules:" : undefined,
    includeImageCues ? (createImageUserRulesBlock(state) || "(none)") : undefined,
    includeImageCues ? "Image tag keyword presets (no roster identity tags):" : undefined,
    includeImageCues ? (createImageSceneTagPresetBlock(state, input.userText) || "(none)") : undefined,
    includeImageCues ? "Image cue authoring reference for current-scene visible characters only:" : undefined,
    includeImageCues ? (createImageCueVisualProfileBlock(state, input.userText) || "(none)") : undefined,
    includeImageCues ? "Ongoing scene/visual state for current-scene characters only (still-active pose/action/interaction/position + physical-detail facts):" : undefined,
    includeImageCues ? (createImageCueCurrentStateBlock(state, input.userText) || "(none)") : undefined,
    "Return JSON only. The JSON schema is:",
    includeImageCues
      ? JSON.stringify({ state_events: [stateEventShape], image_cues: [imageCueShape] })
      : JSON.stringify({ state_events: [stateEventShape] })
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function parseAnnotationResponse(raw: string, includeImageCues: boolean): TurnAnnotationResult {
  const errors: string[] = [];
  const jsonText = extractJsonObject(unwrapCliAgentBridgeResponse(raw));
  if (jsonText) {
    try {
      const value = JSON.parse(jsonText) as Record<string, unknown>;
      const stateEventsRaw = Array.isArray(value.state_events)
        ? value.state_events
        : Array.isArray(value.stateEvents)
          ? value.stateEvents
          : [];
      const stateEvents = stateEventsRaw
        .map((item) => normalizeMemoryEventDraft(item, errors))
        .filter((item): item is AssistantMemoryEventDraft => Boolean(item))
        .filter((item) => item.memoryKind === "state");
      const imageCues = includeImageCues
        ? normalizeAssistantImageCueDrafts(value.image_cues ?? value.imageCues, errors)
        : [];
      return { stateEvents, imageCues };
    } catch {
      /* fall through to lenient recovery */
    }
  }
  // Lenient recovery: extract from raw text when JSON.parse fails.
  const text = stripLikelyJsonFence(raw);
  const stateStart = findJsonFieldValueStart(text, ["state_events", "stateEvents"]);
  const recoveredStateEvents = stateStart !== undefined
    ? extractJsonObjectFragmentsFromFieldValue(text.slice(stateStart))
        .map((fragment) => parseJsonObjectFragment(fragment))
        .filter((item): item is Record<string, unknown> => Boolean(item))
        .map((item) => normalizeMemoryEventDraft(item, errors))
        .filter((item): item is AssistantMemoryEventDraft => Boolean(item))
        .filter((item) => item.memoryKind === "state")
    : [];
  const recoveredImageCues = includeImageCues ? extractRecoverableImageCueDrafts(raw, errors) : [];
  return { stateEvents: recoveredStateEvents, imageCues: recoveredImageCues };
}

// Authors the turn's state_events (always) and image_cues (when isTurnImageCueGenerationActive) using
// the dedicated annotation LLM, given the already-written narrative.
//
// Runs synchronously after the narrative and before memory compilation:
//   - state_events (memory_kind='state') are merged into the narrative sidecar so memoryCompiler records
//     Wearing/pose/scene/relationship-param deltas in the same pass as semantic events.
//   - image_cues replace the narrative sidecar's imageCues for image planning — this call is the sole
//     tag author so the narrative LLM is completely free of tag vocabulary.
//
// Returns { stateEvents: [], imageCues: [] } when the annotation LLM is unconfigured (graceful degradation).
export async function requestTurnAnnotations(
  state: AppState,
  input: { userText: string; assistantText: string; manualImage?: boolean }
): Promise<TurnAnnotationResult> {
  const llmState = resolveImageTagLlmState(state);
  const requiresApiKey = !isCliAgentLlmProvider(llmState.llm.provider);
  if (!llmState.llm.enabled || llmState.llm.provider === "mock" || (requiresApiKey && !llmState.llm.apiKey.trim())) {
    console.warn("[annotation] skipped: annotation LLM disabled/unconfigured", {
      provider: llmState.llm.provider,
      enabled: llmState.llm.enabled,
      requiresApiKey,
      hasApiKey: Boolean(llmState.llm.apiKey.trim())
    });
    return { stateEvents: [], imageCues: [] };
  }
  const includeImageCues = isTurnImageCueGenerationActive(state, Boolean(input.manualImage));
  const outputTokenBudget = Math.min(resolveInteractiveOutputTokenBudget(state), 1800);
  const instruction = createAnnotationLlmInstruction(state, outputTokenBudget, includeImageCues);
  const context = createAnnotationLlmContext(state, { ...input, outputTokenBudget }, includeImageCues);
  try {
    const raw = await requestProviderText(
      llmState,
      { userText: input.userText, modules: [], evidence: [] },
      instruction,
      context,
      outputTokenBudget,
      { temperature: llmState.llm.temperature }
    );
    if (!raw?.trim()) {
      console.warn("[annotation] LLM returned empty content", { provider: llmState.llm.provider, model: llmState.llm.model });
      return { stateEvents: [], imageCues: [] };
    }
    const result = parseAnnotationResponse(raw, includeImageCues);
    console.info(`[annotation] parsed ${result.stateEvents.length} state event(s), ${result.imageCues.length} image cue(s)`, {
      includeImageCues,
      shouldGenerate: result.imageCues.filter((c) => c.shouldGenerate).length
    });
    return result;
  } catch (error) {
    console.warn("[annotation] LLM call failed", { provider: llmState.llm.provider, model: llmState.llm.model, error: error instanceof Error ? error.message : String(error) });
    return { stateEvents: [], imageCues: [] };
  }
}

function createRuntimeInstruction(
  state: AppState,
  outputTokenBudget: number,
  options: { manualImage?: boolean; modules?: PromptModule[]; omitImageAuthoring?: boolean } = {}
): string {
  const creatorOutputFormat = analyzeCreatorOutputFormat(state, options.modules ?? []);
  const contentRatingInstruction = createContentRatingInstruction(state);
  const imageCadenceInstruction = createImageGenerationCadenceBlock(state, outputTokenBudget);
  const initialImageCueCountInstruction = createInitialImageCueCountInstruction(state, outputTokenBudget);
  // When this turn cannot produce images (image profile off / stored-only / no realtime, and not a manual
  // request or image-progression turn), drop the large image-cue authoring rule block entirely — it is dead
  // weight that inflates the prompt and slows the LLM. A single short line tells the model to leave image_cues
  // empty instead. omitImageAuthoring forces this off entirely: a separate image-cue LLM call owns the tags, so
  // the narrative model must not see ANY tag rules (that style/quality interference is exactly what we're removing).
  const realtimeImageActive =
    state.imageProfile.enabled && state.simulation.realtimeImageEnabled && state.imageProfile.triggerMode !== "stored_only";
  const imageAuthoringActive =
    !options.omitImageAuthoring && (realtimeImageActive || Boolean(options.manualImage) || isImageProgressionCadence(state));
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
    "Long-run continuity (anti-degradation only): keep the established facts, character state, scene, and momentum from the transcript, and do not loop or repeat earlier turns as the log grows. HOW the scene reads — voice, tone, vividness, pacing, dialogue, intensity, explicitness — is governed solely by the creator's Main rules and modules; DynamicChat adds no style of its own and must not suppress, soften, or restrain the creator's. Match the creator's register exactly.",
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
    options.omitImageAuthoring
      ? "If a character appears only in off-stage roster references, stored status, relationship parameters, visual profile lists, or NeuralMap evidence, treat that as background continuity and do not include them in assistant_text for this turn."
      : "If a character appears only in off-stage roster references, stored status, relationship parameters, visual profile lists, or NeuralMap evidence, treat that as background continuity and do not include them in assistant_text or image_cues.characters for this turn.",
    isImageProgressionCadence(state)
      ? `Image progression assistant_text rule: write one short Korean status line only, such as \`이미지 진행 ${IMAGE_PROGRESSION_CUE_TARGET}컷.\` Do not write prose narration, dialogue, markdown, status windows, choices, explanations, or visible tag lists in assistant_text.`
      : creatorOutputFormat.requiresVisibleFormatFromMain
        ? "assistant_text must follow the creator-defined output format every turn: Korean scene prose plus any required Markdown status/choice/table blocks from Main rules. Use valid GFM pipe tables ONLY for an explicit structured status/data block the Main rules require; a table holds short field/value data, never narration or spoken dialogue. Do not wrap anything in DynamicChat effect blocks (```scene/```impact/```whisper/```sfx/```status/```choice/```memory/```letter or ::impact[text]) — they are not rendered; write required status/choice structure as plain Markdown instead. Do not replace a required status/table block with memory_events only. Do not output raw HTML."
        : "Default assistant_text to natural Korean prose with Markdown only when useful. DynamicChat effect blocks are off by default — do not output status windows, stat lines, choice menus, Markdown tables, or code-fence effect blocks (```scene/```impact/```status/```choice/```memory/```letter/```whisper/```sfx). Do not output raw HTML or visible parser labels such as 'SFX:' or 'status -'.",
    isImageProgressionCadence(state)
      ? undefined
      : "Paragraph formatting for assistant_text: separate paragraphs with a blank line — a literal double newline `\\n\\n` in the JSON string (write \"...문장.\\n\\n다음 문단...\") — so the reply is not one unbroken block and images can sit between beats. Use NATURAL paragraph lengths that fit the creator's style (a normal mix of multi-sentence narration and dialogue); do NOT force every sentence or every line onto its own paragraph, which makes the prose read like a clipped list.",
    createCreatorOutputFormatInstruction(creatorOutputFormat),
    imageAuthoringActive
      ? "Image cue ownership: this main response owns final image_cues. No later tag planner will fix, expand, or infer tags. When the current beat should be illustrated, write the complete NovelAI tags now in image_cues.base_tags + character_prompts."
      : options.omitImageAuthoring
        ? "Do NOT output image_cues or any image/visual tags — a separate step plans all visuals. Spend the whole budget on assistant_text and memory_events."
        : "Image generation is OFF for this turn: set image_cues to [] and spend the whole budget on assistant_text and memory_events.",
    // Narrative-first ordering: write assistant_text before image_cues so each cue can anchor to text that exists.
    imageAuthoringActive
      ? "JSON field order (narrative-first, mandatory): write assistant_text FIRST, then author image_cues, then memory_events. Set each generated cue's anchor_text to a short phrase that ACTUALLY appears in the assistant_text, placing a cut at every major beat INCLUDING the opening beat."
      : undefined,
    options.manualImage && state.imageProfile.enabled && state.simulation.realtimeImageEnabled && state.imageProfile.triggerMode !== "stored_only"
      ? "Manual image request: this turn must include at least one should_generate=true image_cue with complete final NovelAI/Danbooru tags unless the current response is impossible to visualize. If impossible, set should_generate=false with a specific suppression_reason."
      : undefined,
    !imageAuthoringActive
      ? undefined
      : isImageProgressionCadence(state)
        ? `For image_cues in image_progression mode, emit exactly ${IMAGE_PROGRESSION_CUE_TARGET} ordered should_generate=true cuts. Each cue is one final NovelAI tag prompt group; advance the scene little by little and vary camera/framing only when it fits the current continuity.`
        : "For image_cues, use [] only for quiet text-only turns. If user image rules or cadence require cuts, emit the complete cut list now: one cue for balanced/sparse major beats, 2-4 cues for rich multi-beat responses, and in paragraph/high-density mode one anchored cue per significant dialogue/action/body/state visual beat, up to 8. Long paragraphs may contain multiple cues. Every generated cue must include concrete final tags, visible character metadata, placement, and a short anchor_text when useful.",
    imageAuthoringActive ? initialImageCueCountInstruction : undefined,
    imageAuthoringActive
      ? "Image prompt user rules are binding composition and tag-routing instructions. Follow explicit positive/negative NovelAI tag directives, but never turn rule labels, examples, or headings into visible objects."
      : undefined,
    imageAuthoringActive
      ? "Scene tag keyword presets are creator-authored hierarchical references. Read each path as parent keyword > child keyword; inherited_base_tags are broad branch context and base_scene_tags are the selected node details. You may consult multiple matching branches in one cue, such as one branch for action/scene and another branch for expression/pose. If no branch matches, ignore the preset list. Put scene/composition/action/environment results in base_tags and character-specific expression/pose/outfit/body-state results in the matching character_prompts item."
      : undefined,
    imageAuthoringActive ? `Image generation cadence setting is binding: ${imageCadenceInstruction}` : undefined,
    imageAuthoringActive ? createImageCueTagContractInstruction(state) : undefined,
    // State/visual extraction has moved to requestTurnAnnotations (the annotation call that runs after the
    // narrative). The narrative model must not emit memory_kind='state' or see tag-category vocabulary —
    // both contaminate Korean prose with English annotation labels and numeric stat markers.
    options.omitImageAuthoring
      ? undefined
      : "Secondary state metadata — read only after assistant_text is fully written, and never let it influence how the prose reads: when a character's outfit, pose, action, interaction, expression, held item, physical state, whole-scene phase, or scene composition CHANGES this turn, record that delta in memory_events with memory_kind='state'. Use state_type='Wearing', 'StatusTags', 'PoseTags', 'ActionTags', 'InteractionTags', 'InteractionPhaseTags', 'HeldItemTags', 'PhysicalStateTags', 'SceneTags', 'ScenePhaseTags', 'CompositionTags', 'CameraTags', or 'LightingTags'. state_value is comma-separated English tags (3-8 compact tags that preserve the situation, not every micro-detail). For Wearing, keep base garments/colors/style and append damage/condition tags (e.g. `police uniform, navy short dress, torn uniform`). Set actor_id for character-specific state; leave it empty only for whole-scene state. Only emit a delta when something actually changed — do not re-list unchanged state.",
    "For memory_events, preserve graph roles carefully: relationship deltas need actor_id and target_id when both sides are known; observations and beliefs need observers or actor_id for the character who knows the fact; scene-wide facts should not be assigned to a random roster character.",
    options.omitImageAuthoring
      ? "Keep the outer JSON valid and stop cleanly. If the budget is tight, reduce memory_events first, and do not end assistant_text as only an opening beat."
      : "Keep the outer JSON valid and stop cleanly. If the budget is tight, reduce memory_events first; do not drop required image_cues base_tags/character_prompts, and do not end assistant_text as only an opening beat.",
    options.omitImageAuthoring
      ? "Your response must be valid JSON only with exactly two fields: assistant_text (the user-visible narrative) and memory_events (semantic deltas only — no memory_kind='state'). Do NOT include image_cues or state/visual tags. If assistant_text contains Markdown code fences, encode them as a JSON string value; never write Markdown outside the JSON object."
      : "Your response must be valid JSON only. Write the user-visible narrative in assistant_text first, then visual planning in image_cues, then only structured memory deltas in memory_events. If assistant_text contains Markdown code fences, encode them as a JSON string value; never write Markdown outside the JSON object."
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
    "Write JSON fields in this exact order: image_cues, assistant_text, memory_events. Keep memory_events compact or [] if needed.",
    `Simulation title: ${state.simulation.title}`,
    `Content rating: ${state.simulation.contentRating}`,
    "Return exactly this JSON shape with no Markdown outside JSON:",
    JSON.stringify({
      image_cues: [
        {
          label: "visible beat",
          kind: "scene",
          placement: "after",
          anchor_text: "assistant_text fragment you will write next",
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
      assistant_text: "Korean scene continuation shown to the user, written to match the planned image_cues",
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

  // The token setting is a CEILING and a rough target, NOT a quota to fill. Forcing length when the scene has
  // run out of new content makes the model pad with repeated phrases/loops — so every tier caps the length but
  // explicitly forbids padding/repetition and allows a clean early end when the beat is genuinely complete.
  const noPadding =
    " This is an UPPER bound and a rough target, not a quota to fill: write as much as the scene genuinely supports and end cleanly when the beat is complete. NEVER pad by repeating phrases or looping filler to reach the length. Padding means ECHOING wording — it does not mean atmosphere, sensory detail, or interiority: normal immersive prose in the creator's register is not padding.";
  if (maxTokens <= 1000) {
    return "Output length target: compact — up to about 2-3 Korean paragraphs with concrete action, dialogue, and one clear consequence." + noPadding;
  }

  if (maxTokens <= 1800) {
    return "Output length target: balanced — up to about 4-6 Korean paragraphs with concrete action, dialogue, sensory detail, and visible consequences." + noPadding;
  }

  if (maxTokens <= 3000) {
    return "Output length target: long — up to about 7-10 Korean paragraphs developing the scene through multiple beats, dialogue, and state changes." + noPadding;
  }

  if (maxTokens <= 4500) {
    return "Output length target: very long — up to about 10-14 Korean paragraphs with rich scene progression, character reaction, and state changes (plus a status/choice block when creator rules require it)." + noPadding;
  }

  return "Output length target: extended — up to about 12-18 Korean paragraphs with substantial scene progression, dialogue, and consequences (plus any creator-required status/choice structure)." + noPadding;
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

// Deterministic backstop against decoding-level repetition loops (e.g. "한 번에 한 번에 한 번에 …").
// Instruction-only guards do not reliably stop a model that has fallen into a degenerate loop, and once a
// looped reply is saved it is fed back through the recent-transcript block next turn and amplifies. Collapsing
// the loop here cleans the displayed text AND breaks that feedback chain, because the returned (collapsed)
// assistant_text is what gets persisted. The rule is intentionally conservative: it only collapses a short
// 1-3 word unit that repeats 3+ times back-to-back, which natural Korean prose effectively never does, so
// deliberate doubling ("정말, 정말") and non-adjacent reuse are left untouched.
function collapseDegenerateRepetition(text: string): string {
  if (!text || text.length < 12) {
    return text;
  }
  // Preserve paragraph structure (\n / \n\n) by collapsing line by line.
  return text.split("\n").map(collapseLineRepetition).join("\n");
}

// English context-scaffolding labels that the context block (createContextBlock) injects. A Korean
// in-character narrative never legitimately begins a line with one of these, so when the model echoes them
// back inside a VALID-JSON assistant_text they are unambiguous prompt leaks ("문맥 간섭"). looksLikeInternalPromptLeak
// only guards the JSON-parse-FAILED fallback path; this line-level stripper closes the gap for parsed sidecars
// without discarding the whole reply.
const INTERNAL_SCAFFOLDING_LINE_PATTERNS: RegExp[] = [
  /^\s*(?:SYSTEM INSTRUCTION|CONTEXT BLOCK|USER ACTION)\s*:/iu,
  /^\s*Use the following DynamicChat context/iu,
  /^\s*Return JSON only\b/iu,
  /^\s*Write JSON fields in this exact order/iu,
  /^\s*Immediate continuity anchor\s*:/iu,
  /^\s*Current scene (?:cast guard|briefing)\s*:/iu,
  /^\s*Simulation foundation\s*:/iu,
  /^\s*Structured simulation memory\s*:/iu,
  /^\s*Selected prompt modules for this turn\s*:/iu,
  /^\s*Memory\/context evidence\s*:/iu,
  /^\s*Recent transcript\s*:/iu,
  /^\s*User persona\s*:/iu,
  /^\s*Image (?:prompt user rules|tag keyword presets|generation cadence|cue authoring reference)\b/iu,
  /^\s*Current turn image generation policy\s*:/iu
];

function stripLeakedScaffolding(text: string): string {
  if (!text) {
    return text;
  }
  // Strip raw internal scene/session ids the context guard explicitly tells the model not to reveal.
  const withoutIds = text.replace(/scene:session_[A-Za-z0-9_:.-]+/gu, "").replace(/[ \t]{2,}/gu, " ");
  const kept = withoutIds
    .split("\n")
    .filter((line) => !INTERNAL_SCAFFOLDING_LINE_PATTERNS.some((pattern) => pattern.test(line)));
  const result = kept.join("\n").replace(/\n{3,}/gu, "\n\n").trim();
  // Safety: never let scaffolding-stripping gut a real reply. If almost nothing survived, the match was
  // probably wrong (or the whole reply was scaffolding, which the fallback path handles) — keep the original.
  if (result.length < Math.min(40, Math.floor(text.trim().length * 0.5))) {
    return text;
  }
  return result;
}

// Single chokepoint for cleaning a finalized assistant_text before it is displayed AND persisted: strip
// echoed internal context scaffolding, then collapse decoding-level repetition loops.
function sanitizeFinalAssistantText(text: string): string {
  return collapseDegenerateRepetition(stripLeakedScaffolding(text));
}

function collapseLineRepetition(line: string): string {
  const words = line.split(" ");
  if (words.length < 6) {
    return line;
  }
  const out: string[] = [];
  let index = 0;
  while (index < words.length) {
    let collapsed = false;
    for (let unitLen = 1; unitLen <= 3; unitLen += 1) {
      const unit = words.slice(index, index + unitLen);
      if (unit.length < unitLen || unit.some((word) => word.length === 0 || word.length > 12)) {
        continue;
      }
      let reps = 1;
      while (arraysEqual(words.slice(index + reps * unitLen, index + (reps + 1) * unitLen), unit)) {
        reps += 1;
      }
      if (reps >= 3) {
        out.push(...unit); // keep a single copy of the looped unit
        index += reps * unitLen;
        collapsed = true;
        break;
      }
    }
    if (!collapsed) {
      out.push(words[index]);
      index += 1;
    }
  }
  return out.join(" ");
}

function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

function parseAssistantSidecar(raw: string): { sidecar?: AssistantSidecar; errors: string[] } {
  const errors: string[] = [];
  const jsonText = extractJsonObject(unwrapCliAgentBridgeResponse(raw));
  if (!jsonText) {
    return { errors: ["No JSON object found in LLM response."] };
  }

  try {
    const value = JSON.parse(jsonText) as Record<string, unknown>;
    const assistantTextValue = readString(value.assistant_text) ?? readString(value.assistantText);
    const assistantText = assistantTextValue ? sanitizeFinalAssistantText(assistantTextValue) : assistantTextValue;
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
    const restImageCues = normalizeAssistantImageCueDrafts(imageCueValue, errors);
    // first_image_cue is the streaming-only opening cut emitted BEFORE assistant_text so the first image can be
    // dispatched mid-stream; image_cues (emitted after the narrative) holds the remaining cuts. Fold the opening
    // cut back in as cue index 0 so the rest of the pipeline keeps seeing one ordered list.
    const firstImageCueDraft = normalizeAssistantImageCueDrafts(
      (value.first_image_cue ?? value.firstImageCue) as unknown,
      errors
    )[0];
    const imageCues = prependFirstImageCueDraft(firstImageCueDraft, restImageCues);
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
  if (!assistantText || looksLikeSidecarJson(assistantText)) {
    return { errors };
  }

  const memoryEvents = extractRecoverableMemoryEventDrafts(raw, errors);
  const imageCues = prependFirstImageCueDraft(
    extractRecoverableFirstImageCueDraft(raw, errors),
    extractRecoverableImageCueDrafts(raw, errors)
  );
  const imageCue = imageCues[0] ?? createNoImageCue("Malformed LLM sidecar did not include a recoverable image cue");

  return {
    sidecar: {
      assistantText: sanitizeFinalAssistantText(assistantText),
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

function imageCueDraftSignature(draft: PlannedImageCueDraft): string {
  return [
    (draft.tags ?? []).join(","),
    (draft.baseTags ?? []).join(","),
    draft.anchorText ?? ""
  ]
    .join("|")
    .toLowerCase();
}

// Fold the streaming-only `first_image_cue` opening cut back in as cue index 0. Prepend only when it is a real
// generating cut; if the model also restated the same cut as image_cues[0], keep the list as-is to avoid a
// duplicate opening image.
function prependFirstImageCueDraft(
  first: PlannedImageCueDraft | undefined,
  rest: PlannedImageCueDraft[]
): PlannedImageCueDraft[] {
  if (!first || !first.shouldGenerate) {
    return rest;
  }
  if (rest.length > 0 && imageCueDraftSignature(first) === imageCueDraftSignature(rest[0])) {
    return rest;
  }
  return [first, ...rest];
}

function extractRecoverableFirstImageCueDraft(raw: string, errors: string[]): PlannedImageCueDraft | undefined {
  const text = stripLikelyJsonFence(raw);
  const valueStart = findJsonFieldValueStart(text, ["first_image_cue", "firstImageCue"]);
  if (valueStart === undefined) {
    return undefined;
  }

  return extractJsonObjectFragmentsFromFieldValue(text.slice(valueStart))
    .map((fragment) => parseJsonObjectFragment(fragment))
    .filter((item): item is Record<string, unknown> => Boolean(item))
    .map((item) => normalizeImageCueDraft(item, errors))
    .filter((item): item is PlannedImageCueDraft => Boolean(item))[0];
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

function unwrapCliAgentBridgeResponse(raw: string): string {
  // The local CLI-agent bridge returns the model output wrapped as
  // {"text": "<model output>", "agent": "...", "model": "..."}. If a server that
  // ignores `stream` returns this wrapper and it reaches the parser, unwrap it so
  // the inner sidecar is parsed instead of the wrapper leaking into the chat.
  const trimmed = raw.trim();
  if (!/^\{/u.test(trimmed) || !/"agent"\s*:/u.test(trimmed) || !/"text"\s*:/u.test(trimmed)) {
    return raw;
  }
  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (typeof parsed.text === "string" && (typeof parsed.agent === "string" || typeof parsed.model === "string")) {
      return parsed.text;
    }
  } catch {
    // Not a clean wrapper; fall through and parse the raw text as-is.
  }
  return raw;
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

function createDisplayFallbackText(rawInput: string, fallback: string): string {
  const raw = unwrapCliAgentBridgeResponse(rawInput);
  const assistantText =
    extractJsonStringField(raw, ["assistant_text", "assistantText"]) ??
    extractLooseJsonTextField(raw, ["assistant_text", "assistantText"]);
  if (assistantText) {
    if (looksLikeInternalPromptLeak(assistantText) || looksLikeSidecarJson(assistantText)) {
      return fallback;
    }
    return sanitizeFinalAssistantText(assistantText);
  }

  const stripped = stripLikelyJsonFence(raw).trim();
  if (!stripped || looksLikeSidecarJson(stripped) || looksLikeInternalPromptLeak(stripped)) {
    return fallback;
  }

  return sanitizeFinalAssistantText(stripped);
}

function looksLikeSidecarJson(value: string): boolean {
  const trimmed = value.trim();
  // Accept single quotes / unquoted keys too: models sometimes emit Python-style
  // dicts (e.g. `{'image_cues': [...]}`) that JSON.parse rejects. The single quote
  // sits between the key and the colon, so a `"?` optional double quote alone misses it.
  const hasSidecarKey =
    /["']?(?:assistant_text|assistantText|memory_events|memoryEvents|image_cues|imageCues|imageCue|image_cue)["']?\s*:/u.test(
      trimmed
    );
  if (!hasSidecarKey) {
    return false;
  }
  // Require an object/array brace near the start so plain prose that merely mentions
  // a field name is not suppressed, while still catching JSON with a short preamble.
  return /[{[]/u.test(trimmed.slice(0, 240));
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
    // Models sometimes emit invalid JSON escapes (e.g. a stray backslash before a
    // CJK character like `\까`). Drop backslashes that don't start a valid JSON
    // escape so the literal can be parsed instead of leaking the backslash.
    const repaired = literal.replace(/\\(?![\\"/bfnrtu])/gu, "");
    try {
      return JSON.parse(repaired) as string;
    } catch {
      return repaired
        .slice(1, -1)
        .replace(/\\n/gu, "\n")
        .replace(/\\r/gu, "\r")
        .replace(/\\t/gu, "\t")
        .replace(/\\"/gu, "\"")
        .replace(/\\\\/gu, "\\");
    }
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
