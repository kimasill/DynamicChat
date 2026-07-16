import type { AppState, ChatMessage, ImageGenerationJob, SimulationProgressRun } from "../types";

type ProgressRunRuntime = Pick<
  AppState,
  | "simulation"
  | "activeProgressRunId"
  | "messages"
  | "memoryEvents"
  | "contextPacks"
  | "handoffs"
  | "continuityChecks"
  | "promptModuleUsages"
  | "sidecarTraces"
  | "turnTraces"
  | "imageAssets"
  | "imageJobs"
  | "selectedContextPackId"
>;

export function createActiveProgressRunSnapshot(
  state: ProgressRunRuntime,
  metadata?: Partial<SimulationProgressRun>
): SimulationProgressRun {
  const createdAt = metadata?.createdAt ?? state.messages[0]?.createdAt ?? state.simulation.createdAt;
  const updatedAt = getLatestProgressRunTimestamp(state) ?? metadata?.updatedAt ?? state.simulation.updatedAt;
  const activeSessionId = state.simulation.activeSessionId;
  const id = metadata?.id ?? state.activeProgressRunId ?? createDefaultProgressRunId(activeSessionId);

  return {
    id,
    simulationId: state.simulation.id,
    title: metadata?.title ?? "진행 1",
    activeSessionId,
    sessionIds: collectProgressRunSessionIds(state, metadata?.sessionIds),
    messages: state.messages,
    memoryEvents: state.memoryEvents,
    contextPacks: state.contextPacks,
    handoffs: state.handoffs,
    continuityChecks: state.continuityChecks,
    promptModuleUsages: state.promptModuleUsages,
    sidecarTraces: state.sidecarTraces,
    turnTraces: state.turnTraces,
    imageAssets: state.imageAssets,
    imageJobs: state.imageJobs,
    selectedContextPackId: state.selectedContextPackId,
    createdAt,
    updatedAt
  };
}

export function normalizeProgressRuns(state: ProgressRunRuntime, candidateRuns: SimulationProgressRun[] | undefined): SimulationProgressRun[] {
  const storedRuns = (candidateRuns ?? []).map((run) => normalizeProgressRun(run, state));
  const activeRunId =
    state.activeProgressRunId ??
    storedRuns.find((run) => run.activeSessionId === state.simulation.activeSessionId)?.id ??
    storedRuns[0]?.id ??
    createDefaultProgressRunId(state.simulation.activeSessionId);
  const existingActiveRun = storedRuns.find((run) => run.id === activeRunId);
  const activeRun = createActiveProgressRunSnapshot(
    {
      ...state,
      activeProgressRunId: activeRunId
    },
    existingActiveRun ?? {
      id: activeRunId,
      title: storedRuns.length > 0 ? createNextProgressRunTitle(storedRuns) : "진행 1"
    }
  );

  return upsertProgressRun(storedRuns, activeRun).sort(compareProgressRunsByUpdatedAt);
}

export function upsertProgressRun(runs: SimulationProgressRun[], nextRun: SimulationProgressRun): SimulationProgressRun[] {
  const exists = runs.some((run) => run.id === nextRun.id);
  return exists
    ? runs.map((run) => (run.id === nextRun.id ? nextRun : run))
    : [nextRun, ...runs];
}

export function createNextProgressRunTitle(runs: SimulationProgressRun[]): string {
  const maxIndex = runs.reduce((max, run) => {
    const match = /(?:새\s*)?진행\s*(\d+)/u.exec(run.title);
    return Math.max(max, match ? Number(match[1]) : 0);
  }, 0);

  return `새 진행 ${Math.max(maxIndex, runs.length) + 1}`;
}

export function createDefaultProgressRunId(activeSessionId: string): string {
  return `run_${activeSessionId.replace(/^session_?/u, "") || "default"}`;
}

function normalizeProgressRun(run: SimulationProgressRun, state: ProgressRunRuntime): SimulationProgressRun {
  const messages = normalizeRunMessages(run.messages);
  const activeSessionId = run.activeSessionId ?? messages.at(-1)?.sessionId ?? state.simulation.activeSessionId;

  return {
    ...run,
    simulationId: state.simulation.id,
    title: run.title?.trim() || "진행",
    activeSessionId,
    sessionIds: uniqueIds([...(run.sessionIds ?? []), ...messages.map((message) => message.sessionId), activeSessionId]),
    messages,
    memoryEvents: (run.memoryEvents ?? []).map((event) => ({
      ...event,
      tags: Array.isArray(event.tags) ? event.tags : []
    })),
    contextPacks: (run.contextPacks ?? []).map((pack) => ({
      ...pack,
      evidence: Array.isArray(pack.evidence) ? pack.evidence : [],
      decisions: Array.isArray(pack.decisions) ? pack.decisions : [],
      blockers: Array.isArray(pack.blockers) ? pack.blockers : []
    })),
    handoffs: run.handoffs ?? [],
    continuityChecks: run.continuityChecks ?? [],
    promptModuleUsages: run.promptModuleUsages ?? [],
    sidecarTraces: run.sidecarTraces ?? [],
    turnTraces: (run.turnTraces ?? []).map((trace) => ({
      ...trace,
      promptModuleUsageIds: Array.isArray(trace.promptModuleUsageIds) ? trace.promptModuleUsageIds : [],
      memoryEventIds: Array.isArray(trace.memoryEventIds) ? trace.memoryEventIds : [],
      imageAssetIds: Array.isArray(trace.imageAssetIds) ? trace.imageAssetIds : [],
      metrics: {
        ...trace.metrics,
        tokenBudget: trace.metrics?.tokenBudget ?? 0,
        selectedModuleCount: trace.metrics?.selectedModuleCount ?? 0,
        selectedModuleTokenEstimate: trace.metrics?.selectedModuleTokenEstimate ?? 0,
        contextEvidenceCount: trace.metrics?.contextEvidenceCount ?? 0,
        contextTokenEstimate: trace.metrics?.contextTokenEstimate ?? 0,
        ragTokenSavingsEstimate: trace.metrics?.ragTokenSavingsEstimate ?? 0,
        llmLatencyMs: trace.metrics?.llmLatencyMs ?? 0,
        llmRequestMs: trace.metrics?.llmRequestMs ?? trace.metrics?.llmLatencyMs ?? 0,
        retrievalLatencyMs: trace.metrics?.retrievalLatencyMs ?? 0,
        memoryIngestMs: trace.metrics?.memoryIngestMs ?? 0,
        turnLatencyMs: trace.metrics?.turnLatencyMs ?? trace.metrics?.llmLatencyMs ?? 0,
        memoryIngestCount: trace.metrics?.memoryIngestCount ?? 0,
        imageJobCount: trace.metrics?.imageJobCount ?? 0,
        imageAssetCount: trace.metrics?.imageAssetCount ?? 0
      }
    })),
    imageAssets: (run.imageAssets ?? []).map((asset) => ({
      ...asset,
      characterIds: Array.isArray(asset.characterIds) ? asset.characterIds : [],
      tags: Array.isArray(asset.tags) ? asset.tags : [],
      reuseTags: Array.isArray(asset.reuseTags) ? asset.reuseTags : undefined,
      palette: Array.isArray(asset.palette) && asset.palette.length === 3 ? asset.palette : ["#f4f0e8", "#d7c7aa", "#5f5046"]
    })),
    imageJobs: (run.imageJobs ?? []).map((job) => ({
      ...job,
      providerPayload:
        job.providerPayload && typeof job.providerPayload === "object" && !Array.isArray(job.providerPayload)
          ? job.providerPayload
          : {},
      assetIds: Array.isArray(job.assetIds) ? job.assetIds : [],
      contextNodeIds: Array.isArray(job.contextNodeIds) ? job.contextNodeIds : [],
      policyWarnings: Array.isArray(job.policyWarnings) ? job.policyWarnings : undefined
    })),
    selectedContextPackId: run.selectedContextPackId,
    createdAt: run.createdAt ?? messages[0]?.createdAt ?? state.simulation.createdAt,
    updatedAt: run.updatedAt ?? getLatestProgressRunTimestamp({ ...state, messages }) ?? state.simulation.updatedAt
  };
}

function normalizeRunMessages(messages: ChatMessage[] | undefined): ChatMessage[] {
  return (messages ?? []).map((message) => ({
    ...message,
    referencedNodeIds: Array.isArray(message.referencedNodeIds) ? message.referencedNodeIds : [],
    imageAssetIds: Array.isArray(message.imageAssetIds) ? message.imageAssetIds : []
  }));
}

function collectProgressRunSessionIds(state: ProgressRunRuntime, existingSessionIds: string[] | undefined): string[] {
  return uniqueIds([
    ...(existingSessionIds ?? []),
    state.simulation.activeSessionId,
    ...state.messages.map((message) => message.sessionId),
    ...state.memoryEvents.map((event) => event.sessionId),
    ...state.contextPacks.map((pack) => pack.sessionId),
    ...state.handoffs.flatMap((handoff) => [handoff.previousSessionId, handoff.nextSessionId]),
    ...state.continuityChecks.flatMap((check) => [check.previousSessionId, check.nextSessionId]),
    ...state.promptModuleUsages.map((usage) => usage.sessionId),
    ...state.sidecarTraces.map((trace) => trace.sessionId),
    ...state.turnTraces.map((trace) => trace.sessionId),
    ...state.imageJobs.map((job) => job.sessionId)
  ]);
}

function getLatestProgressRunTimestamp(state: Pick<ProgressRunRuntime, "messages" | "memoryEvents" | "contextPacks" | "imageJobs">): string | undefined {
  return [
    ...state.messages.map((item) => item.createdAt),
    ...state.memoryEvents.map((item) => item.createdAt),
    ...state.contextPacks.map((item) => item.createdAt),
    ...state.imageJobs.map((item) => item.updatedAt ?? item.completedAt ?? item.createdAt)
  ]
    .filter(Boolean)
    .sort()
    .at(-1);
}

function compareProgressRunsByUpdatedAt(a: SimulationProgressRun, b: SimulationProgressRun): number {
  return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
}

function uniqueIds(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))];
}

/**
 * Returns a slimmed copy of a ProgressRun suitable for long-term storage.
 *
 * Removes heavy per-run data that accumulates rapidly (NAI provider metadata /
 * request payloads) and clears debug-only trace arrays. Core content fields
 * (messages, memoryEvents, handoffs, imageAssets with objectKey, imageJobs with
 * assetIds, session IDs, timestamps) are preserved so the run remains usable
 * for replay and continuity purposes.
 */
export function slimProgressRunForStorage(run: SimulationProgressRun): SimulationProgressRun {
  return {
    ...run,
    // Debug / trace arrays — not needed in stored snapshots.
    contextPacks: [],
    continuityChecks: [],
    sidecarTraces: [],
    turnTraces: [],
    promptModuleUsages: [],
    // Strip NAI response metadata (~151 KB/asset). objectKey / mimeType / tags
    // / palette etc. are preserved for image display and vibe reuse.
    imageAssets: run.imageAssets.map(({ providerMetadata: _pm, ...rest }) => rest),
    // Strip NAI request payload (~89 KB/job). assetIds / status / prompt etc.
    // are preserved for asset linkage and display.
    imageJobs: run.imageJobs.map((job): ImageGenerationJob => ({ ...job, providerPayload: {} }))
  };
}
