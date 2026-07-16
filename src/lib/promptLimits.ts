import type { PromptModuleKind, TokenPolicy } from "../types";

// Soft per-turn excerpt budgets applied during prompt assembly (see compactPromptModuleForActiveContext in
// simulationEngine.ts). These are NOT hard input caps: a body longer than its budget is still saved in full,
// but from that point on each turn only the scene-relevant windows are excerpted into the LLM prompt. The
// simulation builder surfaces these numbers as "current / max" counters so a creator can see when a field
// will start being excerpted rather than injected verbatim.

// The main prompt carries the creator's core operating law, so it gets the largest budget.
export const MAX_MAIN_PROMPT_BODY_CHARS = 8000;
// World lore (and any other foundation module) excerpt budget.
export const MAX_FOUNDATION_MODULE_BODY_CHARS = 6000;
// Non-foundation RAG / scene / rule module bodies.
export const MAX_ACTIVE_MODULE_BODY_CHARS = 2400;

export function isFoundationPromptModuleKind(kind: PromptModuleKind): boolean {
  return kind === "main_prompt" || kind === "world_lore";
}

// Returns the soft excerpt budget applied to this module body during prompt assembly, or null when the body is
// kept verbatim (no budget applies, so no counter maximum should be shown). Keep in sync with
// compactPromptModuleForActiveContext in simulationEngine.ts.
export function promptModuleBodyCharLimit(kind: PromptModuleKind, tokenPolicy: TokenPolicy): number | null {
  if (isFoundationPromptModuleKind(kind)) {
    return kind === "main_prompt" ? MAX_MAIN_PROMPT_BODY_CHARS : MAX_FOUNDATION_MODULE_BODY_CHARS;
  }
  // Non-foundation always-on policies are injected verbatim, never excerpted.
  if (tokenPolicy === "always") {
    return null;
  }
  return MAX_ACTIVE_MODULE_BODY_CHARS;
}
