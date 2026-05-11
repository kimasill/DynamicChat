# DynamicChat Operational Inspector and Eval Notes

## Scope

AIN-23 adds a lightweight operator view for checking which context and image
decisions shaped each simulation turn.

## Data Captured

- `TurnTrace`: links user/assistant messages, Context Pack, prompt module usage,
  LLM sidecar trace, curated memory events, image cue, image job, image assets,
  and runtime metrics.
- `TurnTraceMetrics`: stores token budget, selected module count, context
  evidence count, estimated selected/context tokens, estimated RAG savings,
  latency, memory ingest count, and image count/cost when provider payloads
  include cost data.
- `EvaluationScenario`: defines memory recall, reset continuity, and image
  quality checks that the inspector can score from current local state.
- `ImageAsset.feedback`: stores the operator quality marker for generated
  assets.

## Inspector Surface

The simulation screen now includes an operations rail with filters for all
events, image jobs, suppressed image cues, continuity records, and warnings.
Search matches recent turn traces, memory events, Context Packs, image jobs,
handoffs, and continuity checks.

## Evaluation Rules

- Memory recall scenarios score expected signals against Context Pack evidence,
  stored memory events, and transcript text.
- Reset continuity scenarios use the latest continuity check after session
  handoff.
- Image quality scenarios score generated assets with saved feedback markers.

These checks are intentionally local and deterministic. They are meant to make
MVP debugging visible before a full server-side eval runner exists.
