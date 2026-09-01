import type { AppState } from "../types";
import { getLlmProviderPreset } from "./llmProviders";

// Prompt sizing for the annotation (image-tag + state) pass.
//
// That pass builds by far the largest prompt of the turn — the tag contract, the cast guard, the scene
// briefing, the visual-profile reference, the ongoing visual state and the creator's scene-tag preset
// library all go in — and measured on a real simulation it runs to roughly 10k tokens. That is fine on a
// hosted model with a 128k window and impossible on a local one: Ollama pins every model to a 4096-token
// context regardless of what the weights support, so the request fails with a raw upstream 400 before the
// model ever sees it.
//
// Rather than permanently shrinking the prompt (which would cost quality on the backends that can afford
// it), the prompt is built at a DETAIL TIER chosen from the budget actually available. A large window keeps
// the full contract; a small one keeps every structural rule and drops the elaboration and the optional
// reference blocks.

/**
 * Characters per token for this prompt's Korean/English/tag mixture.
 *
 * Deliberately pessimistic. Korean runs closer to 1 token per character on most BPE vocabularies while
 * English prose runs 3.5-4; the annotation prompt is mostly English rules with Korean excerpts, and
 * underestimating the token count is the failure that produces a hard 400, so the estimate errs low.
 */
const CHARS_PER_TOKEN = 2.6;

/** Room left for the chat template, tokenizer drift and the response itself. */
const CONTEXT_SAFETY_MARGIN = 0.12;

export type PromptDetailTier = "full" | "compact" | "minimal";

export interface AnnotationPromptBudget {
  /** Usable context of the configured backend. */
  contextTokens: number;
  /** Tokens the prompt may occupy. */
  inputTokenBudget: number;
  /** Tokens reserved for the reply. */
  outputTokenBudget: number;
  /** How much elaboration the prompt can afford. */
  tier: PromptDetailTier;
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/** Usable context for the model that will actually serve this call. */
export function resolveContextTokens(state: AppState): number {
  const configured = state.llm.contextTokens;
  if (typeof configured === "number" && Number.isFinite(configured) && configured > 0) {
    return configured;
  }
  return getLlmProviderPreset(state.llm.provider).contextTokens;
}

/**
 * Splits the model's context between the prompt and the reply, and picks the detail tier that fits.
 *
 * The reply budget is honoured first — a truncated `image_cues` array is the failure that renders the wrong
 * cast, so it is the last thing to give up room. Whatever remains is the prompt's, and the tier follows.
 */
export function resolveAnnotationPromptBudget(state: AppState, requestedOutputTokens: number): AnnotationPromptBudget {
  const contextTokens = resolveContextTokens(state);
  const usable = Math.floor(contextTokens * (1 - CONTEXT_SAFETY_MARGIN));
  // Never let the reply take more than half the window; on a small context an oversized reply budget leaves
  // no room for the contract that tells the model what to write.
  const outputTokenBudget = Math.max(512, Math.min(requestedOutputTokens, Math.floor(usable / 2)));
  const inputTokenBudget = Math.max(512, usable - outputTokenBudget);

  return {
    contextTokens,
    inputTokenBudget,
    outputTokenBudget,
    tier: resolveTier(inputTokenBudget)
  };
}

function resolveTier(inputTokenBudget: number): PromptDetailTier {
  // A first guess only — the caller measures the built prompt and steps down if it does not fit (see
  // buildWithinBudget). Guessing from thresholds alone is brittle: the prompt's real size depends on how
  // many characters, presets and state events the creator's simulation has, which no constant can predict.
  if (inputTokenBudget >= 9_000) {
    return "full";
  }
  if (inputTokenBudget >= 3_000) {
    return "compact";
  }
  return "minimal";
}

const TIER_ORDER: PromptDetailTier[] = ["full", "compact", "minimal"];

/**
 * Builds the prompt at the richest detail tier that actually fits, measuring rather than assuming.
 *
 * Starts from the tier the budget suggests and steps down until the built prompt is within the input
 * budget. Returns the last attempt when even the smallest does not fit, so the caller can warn with a real
 * number instead of failing at the provider with an opaque 400.
 */
export function buildWithinBudget<T>(
  budget: AnnotationPromptBudget,
  build: (tier: PromptDetailTier) => T,
  measure: (built: T) => string
): { built: T; tier: PromptDetailTier; estimatedTokens: number; fits: boolean } {
  const startIndex = TIER_ORDER.indexOf(budget.tier);
  let last: { built: T; tier: PromptDetailTier; estimatedTokens: number } | undefined;

  for (let index = Math.max(0, startIndex); index < TIER_ORDER.length; index += 1) {
    const tier = TIER_ORDER[index];
    const built = build(tier);
    const estimatedTokens = estimateTokens(measure(built));
    last = { built, tier, estimatedTokens };
    if (estimatedTokens <= budget.inputTokenBudget) {
      return { ...last, fits: true };
    }
  }

  return { ...last!, fits: false };
}

/**
 * How many scene-tag preset entries the prompt can afford.
 *
 * The creator's preset library is the second-largest block and the most compressible: it is reference
 * material the model samples from, so fewer entries degrade the result gradually rather than breaking it.
 */
export function resolveScenePresetLimit(tier: PromptDetailTier, configuredLimit: number): number {
  if (tier === "full") {
    return configuredLimit;
  }
  return tier === "compact" ? Math.min(configuredLimit, 6) : 0;
}

/** Whether the tier can afford the optional per-character reference blocks. */
export function includesOptionalReferenceBlocks(tier: PromptDetailTier): boolean {
  return tier !== "minimal";
}

/**
 * Trims a prompt section to a character ceiling on a line boundary, so a truncated block never ends
 * mid-rule. Returns the section unchanged when it already fits.
 */
export function trimSectionToBudget(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  const lines = text.split("\n");
  const kept: string[] = [];
  let used = 0;
  for (const line of lines) {
    if (used + line.length + 1 > maxChars) {
      break;
    }
    kept.push(line);
    used += line.length + 1;
  }
  return kept.join("\n");
}
