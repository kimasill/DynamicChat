import { hydrateState } from "../data/seed";
import { createId } from "../lib/id";
import type { AppState, SimulationProgressRun } from "../types";
import {
  createActiveProgressRunSnapshot,
  createNextProgressRunTitle,
  upsertProgressRun
} from "./progressRuns";
import { createAuditEvent } from "./security";

export function createFreshSimulationRun(source: AppState, _library: AppState[] = []): AppState {
  const hydratedSource = hydrateState(source);
  const now = new Date().toISOString();
  const sessionId = createId("session");
  const progressRunId = createId("run");
  const syncedRuns = syncActiveProgressRun(hydratedSource);
  const openingMessage = createFreshRunOpeningMessage(hydratedSource, sessionId, now);
  const carriedAssets = hydratedSource.imageAssets.filter((asset) => asset.source === "stored");
  const freshRun: SimulationProgressRun = {
    id: progressRunId,
    simulationId: hydratedSource.simulation.id,
    title: createNextProgressRunTitle(syncedRuns),
    activeSessionId: sessionId,
    sessionIds: [sessionId],
    messages: [openingMessage],
    memoryEvents: [],
    contextPacks: [],
    handoffs: [],
    continuityChecks: [],
    promptModuleUsages: [],
    sidecarTraces: [],
    turnTraces: [],
    imageAssets: carriedAssets,
    imageJobs: [],
    selectedContextPackId: undefined,
    createdAt: now,
    updatedAt: now
  };
  const nextState = hydrateState({
    ...hydratedSource,
    activeProgressRunId: progressRunId,
    progressRuns: upsertProgressRun(syncedRuns, freshRun),
    simulation: {
      ...hydratedSource.simulation,
      title: createSimulationBaseTitle(hydratedSource.simulation.title),
      activeSessionId: sessionId,
      updatedAt: now
    },
    messages: [openingMessage],
    memoryEvents: [],
    contextPacks: [],
    handoffs: [],
    continuityChecks: [],
    promptModuleUsages: [],
    sidecarTraces: [],
    turnTraces: [],
    imageAssets: carriedAssets,
    imageJobs: [],
    selectedContextPackId: undefined
  });

  return {
    ...nextState,
    auditLog: [
      ...hydratedSource.auditLog,
      createAuditEvent(nextState, "state_saved", "simulation", nextState.simulation.id, {
        operation: "start_new_progress_run",
        sourceSimulationId: hydratedSource.simulation.id,
        sourceProgressRunId: hydratedSource.activeProgressRunId,
        sourceSessionId: hydratedSource.simulation.activeSessionId,
        nextProgressRunId: progressRunId,
        nextSessionId: sessionId
      })
    ]
  };
}

export function activateSimulationProgressRun(source: AppState, progressRunId: string): AppState {
  const hydratedSource = hydrateState(source);
  const syncedRuns = syncActiveProgressRun(hydratedSource);
  const selectedRun = syncedRuns.find((run) => run.id === progressRunId);
  if (!selectedRun || selectedRun.id === hydratedSource.activeProgressRunId) {
    return hydratedSource;
  }

  const now = new Date().toISOString();
  return hydrateState({
    ...hydratedSource,
    activeProgressRunId: selectedRun.id,
    progressRuns: syncedRuns,
    simulation: {
      ...hydratedSource.simulation,
      activeSessionId: selectedRun.activeSessionId,
      updatedAt: now
    },
    messages: selectedRun.messages,
    memoryEvents: selectedRun.memoryEvents,
    contextPacks: selectedRun.contextPacks,
    handoffs: selectedRun.handoffs,
    continuityChecks: selectedRun.continuityChecks,
    promptModuleUsages: selectedRun.promptModuleUsages,
    sidecarTraces: selectedRun.sidecarTraces,
    turnTraces: selectedRun.turnTraces,
    imageAssets: selectedRun.imageAssets,
    imageJobs: selectedRun.imageJobs,
    selectedContextPackId: selectedRun.selectedContextPackId
  });
}

export function deleteSimulationProgressRun(source: AppState, progressRunId: string): AppState {
  const hydratedSource = hydrateState(source);
  const syncedRuns = syncActiveProgressRun(hydratedSource);
  const deletedRun = syncedRuns.find((run) => run.id === progressRunId);
  if (!deletedRun || syncedRuns.length <= 1) {
    return hydratedSource;
  }

  const now = new Date().toISOString();
  const remainingRuns = syncedRuns.filter((run) => run.id !== progressRunId);
  const wasActiveRun = deletedRun.id === hydratedSource.activeProgressRunId;
  const replacementRun = wasActiveRun ? remainingRuns[0] : undefined;
  const nextState = hydrateState(
    replacementRun
      ? {
          ...hydratedSource,
          activeProgressRunId: replacementRun.id,
          progressRuns: remainingRuns,
          simulation: {
            ...hydratedSource.simulation,
            activeSessionId: replacementRun.activeSessionId,
            updatedAt: now
          },
          messages: replacementRun.messages,
          memoryEvents: replacementRun.memoryEvents,
          contextPacks: replacementRun.contextPacks,
          handoffs: replacementRun.handoffs,
          continuityChecks: replacementRun.continuityChecks,
          promptModuleUsages: replacementRun.promptModuleUsages,
          sidecarTraces: replacementRun.sidecarTraces,
          turnTraces: replacementRun.turnTraces,
          imageAssets: replacementRun.imageAssets,
          imageJobs: replacementRun.imageJobs,
          selectedContextPackId: replacementRun.selectedContextPackId
        }
      : {
          ...hydratedSource,
          progressRuns: remainingRuns,
          simulation: {
            ...hydratedSource.simulation,
            updatedAt: now
          }
        }
  );

  return {
    ...nextState,
    auditLog: [
      ...nextState.auditLog,
      createAuditEvent(nextState, "state_saved", "simulation", nextState.simulation.id, {
        operation: "delete_progress_run",
        deletedProgressRunId: progressRunId,
        deletedProgressRunTitle: deletedRun.title,
        nextProgressRunId: nextState.activeProgressRunId,
        deletedActiveRun: wasActiveRun
      })
    ]
  };
}

function createFreshRunOpeningMessage(
  source: AppState,
  sessionId: string,
  createdAt: string
): AppState["messages"][number] {
  const opening = source.messages.find((message) => message.role === "assistant") ?? source.messages[0];
  return {
    id: createId("msg"),
    simulationId: source.simulation.id,
    sessionId,
    role: "assistant",
    content: opening?.content || `${createSimulationBaseTitle(source.simulation.title)}의 첫 장면이 다시 시작됩니다.`,
    createdAt,
    referencedNodeIds: filterFreshRunOpeningReferences(source, opening?.referencedNodeIds ?? []),
    imageAssetIds: opening?.imageAssetIds.filter((assetId) =>
      source.imageAssets.some((asset) => asset.id === assetId && asset.source === "stored")
    ) ?? []
  };
}

function filterFreshRunOpeningReferences(source: AppState, referencedNodeIds: string[]): string[] {
  const staleRuntimeReferences = new Set([
    ...source.memoryEvents.flatMap((event) => [event.id, event.neuralMapNodeId]),
    ...source.contextPacks.flatMap((pack) => [
      pack.id,
      ...pack.evidence.map((item) => item.nodeId),
      ...Object.values(pack.sections ?? {}).flatMap((section) => section.map((item) => item.nodeId))
    ]),
    ...source.handoffs.flatMap((handoff) => [handoff.id, ...handoff.evidenceNodeIds]),
    ...source.continuityChecks.map((check) => check.id),
    ...source.turnTraces.map((trace) => trace.id),
    ...source.sidecarTraces.map((trace) => trace.id)
  ].filter((nodeId): nodeId is string => Boolean(nodeId)));

  return referencedNodeIds.filter((nodeId) => !staleRuntimeReferences.has(nodeId) && !looksLikeProgressScopedReference(nodeId));
}

function looksLikeProgressScopedReference(nodeId: string): boolean {
  return (
    /^(?:memory|ctx|handoff|trace|turn)_/iu.test(nodeId) ||
    /^simulation:[^:]+:(?:memory|event|session|scene|context)(?::|$)/iu.test(nodeId)
  );
}

function createSimulationBaseTitle(title: string): string {
  return title.replace(/\s*·\s*새 진행\s*\d+\s*$/u, "").trim() || title;
}

function syncActiveProgressRun(source: AppState): SimulationProgressRun[] {
  const existingActiveRun = source.progressRuns.find((run) => run.id === source.activeProgressRunId);
  return upsertProgressRun(
    source.progressRuns,
    createActiveProgressRunSnapshot(source, existingActiveRun)
  );
}
