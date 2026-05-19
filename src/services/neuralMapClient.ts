import { createId } from "../lib/id";
import type { AppState, ContextPack, MemoryEvent, NeuralMapSettings, SessionHandoff } from "../types";
import {
  SIMULATION_MEMORY_PROFILE_ID,
  createMemoryGraphNeuronId,
  createStructuredContextEvidence,
  type MemoryDelta,
  type MemoryDeltaRecord
} from "./memoryCompiler";
import { inferCurrentSceneCharacterIds } from "./sceneCast";
import { createScopeMetadata } from "./security";

interface NeuralMapSimulationContextResponse {
  pack?: {
    id: string;
    objective: string;
    session_id: string;
    token_budget: number;
    evidence: Array<{ node_id: string; snippet: string; score: number }>;
    decisions?: string[];
    blockers?: string[];
    created_at: string;
  };
}

interface NeuralMapContextPackResponse {
  id: string;
  objective: string;
  agent_id?: string;
  session_id: string;
  node_ids?: string[];
  evidence: NeuralMapEvidenceItem[];
  sections?: Record<string, NeuralMapEvidenceItem[]>;
  decisions?: string[];
  blockers?: string[];
  token_budget: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

interface NeuralMapEvidenceItem {
  node_id: string;
  snippet: string;
  score: number;
  scope?: PartialNeuralMapScope;
  properties?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

interface NeuralMapGraphDeltaResponse {
  accepted?: boolean;
  idempotency_key?: string;
  profile_id?: string;
  persistence?: {
    nodes?: number;
    edges?: number;
    mode?: string;
  };
  delta?: Record<string, unknown>;
}

interface NeuralMapGraphDeltaRequest {
  idempotency_key: string;
  profile_id: string;
  source: {
    system: string;
    run_id?: string;
    turn_id?: string;
    raw_ref?: string;
    metadata?: Record<string, unknown>;
  };
  scope: NeuralMapScope;
  upsert_neurons: NeuralMapGraphDeltaNeuron[];
  upsert_synapses: NeuralMapGraphDeltaSynapse[];
  temporal_operations: Array<{
    operation: "supersede_current";
    selector: {
      label?: string;
      ontology_type?: string;
      profile_id?: string;
      properties?: Record<string, unknown>;
    };
    valid_to: string;
    superseded_by: string;
  }>;
}

interface NeuralMapGraphDeltaNeuron {
  id: string;
  type?: string;
  labels?: string[];
  title: string;
  summary?: string;
  source_system?: string;
  trust_score?: number;
  freshness_score?: number;
  importance_score?: number;
  confidence?: number;
  lifecycle_status?: string;
  valid_from?: string;
  valid_to?: string | null;
  ontology?: {
    profile_id: string;
    type: string;
  };
  properties?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

interface NeuralMapGraphDeltaSynapse {
  id?: string;
  from: string;
  to: string;
  type: string;
  labels?: string[];
  weight?: number;
  confidence?: number;
  valid_from?: string;
  valid_to?: string | null;
  ontology?: {
    profile_id: string;
    type: string;
  };
  properties?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

interface NeuralMapScope {
  tenant_id: string;
  workspace_id: string;
  project_id: string;
  owner_scope: string;
}

type PartialNeuralMapScope = Partial<NeuralMapScope>;

interface NeuralMapHandoffPack {
  id?: string;
  current_status?: string;
  summary?: string;
  context_summary?: string;
  referenced_node_ids?: string[];
  evidence?: Array<{ node_id?: string; id?: string }>;
  created_at?: string;
}

type NeuralMapHandoffResponse = NeuralMapHandoffPack & {
  pack?: NeuralMapHandoffPack;
};

interface NeuralMapGraphQueryResponse {
  neighborhood?: {
    seed_node_ids?: string[];
    nodes?: Array<{
      id: string;
      type?: string;
      labels?: string[];
      title?: string;
      content_ref?: string;
      summary?: string;
      source_system?: string;
      importance_score?: number;
      lifecycle_status?: string;
      valid_from?: string;
      valid_to?: string | null;
      ontology?: {
        profile_id?: string;
        type?: string;
      };
      properties?: Record<string, unknown>;
      scope?: PartialNeuralMapScope;
      metadata?: Record<string, unknown>;
      created_at?: string;
      updated_at?: string;
    }>;
    edges?: Array<{
      id?: string;
      from: string;
      to: string;
      type?: string;
      labels?: string[];
      weight?: number;
      confidence?: number;
      lifecycle_status?: string;
      valid_from?: string;
      valid_to?: string | null;
      ontology?: {
        profile_id?: string;
        type?: string;
      };
      properties?: Record<string, unknown>;
      scope?: PartialNeuralMapScope;
      metadata?: Record<string, unknown>;
    }>;
    generated_at?: string;
  };
}

const NEURALMAP_REQUEST_TIMEOUT_MS = 2_500;
const NEURALMAP_GRAPH_REQUEST_TIMEOUT_MS = 8_000;
const LOCAL_CONTINUITY_ASSISTANT_CHARS = 1600;
const LOCAL_CONTINUITY_USER_CHARS = 700;

export interface NeuralMapLiveNode {
  id: string;
  type: string;
  labels?: string[];
  title: string;
  summary: string;
  importanceScore: number;
  sourceSystem?: string;
  kind?: string;
  ontology?: {
    profile_id?: string;
    type?: string;
  };
  properties?: Record<string, unknown>;
  scope?: PartialNeuralMapScope;
  metadata?: Record<string, unknown>;
}

export interface NeuralMapLiveEdge {
  id: string;
  from: string;
  to: string;
  type: string;
  labels?: string[];
  weight: number;
  confidence: number;
  ontology?: {
    profile_id?: string;
    type?: string;
  };
  properties?: Record<string, unknown>;
  scope?: PartialNeuralMapScope;
  metadata?: Record<string, unknown>;
}

export interface NeuralMapLiveGraph {
  source: "neuralmap" | "local";
  nodes: NeuralMapLiveNode[];
  edges: NeuralMapLiveEdge[];
  seedNodeIds: string[];
  generatedAt: string;
  error?: string;
}

export class NeuralMapClient {
  constructor(private readonly settings: NeuralMapSettings) {}

  async ingestEvent(event: MemoryEvent, state?: AppState): Promise<string | undefined> {
    if (!this.settings.enabled) {
      return event.neuralMapNodeId;
    }

    const response = await this.request<{ session_node_id?: string }>("/ingest/simulation-event", {
      method: "POST",
      body: JSON.stringify({
        ...(state ? createScopeMetadata(state) : {}),
        simulation_id: event.simulationId,
        session_id: event.sessionId,
        event_id: event.id,
        actor_id: event.actorId,
        actor_name: event.actorName,
        content: event.content,
        importance: event.importance,
        tags: event.tags,
        occurred_at: event.createdAt,
        previous_event_id: state?.memoryEvents.at(-1)?.id,
        progress_run_id: state?.activeProgressRunId,
        run_id: state?.activeProgressRunId,
        scope: state ? createNeuralMapScope(state) : undefined,
        participants: event.actorId
          ? [{ id: event.actorId, name: event.actorName, role: "actor" }]
          : undefined,
        metadata: {
          ...(state ? createScopeMetadata(state) : {}),
          ...(event.metadata ?? {}),
          agent_id: state ? createNeuralMapAgentId(state) : undefined,
          source: "dynamicchat",
          progress_run_id: state?.activeProgressRunId,
          run_id: state?.activeProgressRunId,
          source_turn_id: event.sourceTurnId
        }
      })
    });

    return response.session_node_id;
  }

  async applyMemoryDelta(delta: MemoryDelta, state: AppState): Promise<Map<string, string>> {
    const nodeIds = new Map(delta.upsertRecords.map((record) => [record.id, createMemoryGraphNeuronId(state, delta, record)]));
    if (!this.settings.enabled || delta.upsertRecords.length === 0) {
      return nodeIds;
    }

    const requestBody = createGraphDeltaRequest(state, delta);
    await this.request<NeuralMapGraphDeltaResponse>("/graph/deltas", {
      method: "POST",
      body: JSON.stringify(requestBody)
    }, NEURALMAP_GRAPH_REQUEST_TIMEOUT_MS);

    return nodeIds;
  }

  async getLiveGraph(state: AppState, query = ""): Promise<NeuralMapLiveGraph> {
    const graphQuery = createLiveGraphQuery(state, query);

    if (!this.settings.enabled) {
      return createLocalLiveGraph(state);
    }

    try {
      const response = await this.request<NeuralMapGraphQueryResponse>("/graph/query", {
        method: "POST",
        body: JSON.stringify({
          query: graphQuery,
          profile_id: SIMULATION_MEMORY_PROFILE_ID,
          top_k: 10,
          expand_hops: 2,
          min_edge_confidence: 0.2,
          filters: {
            simulation_id: state.simulation.id,
            progress_run_id: state.activeProgressRunId,
            run_id: state.activeProgressRunId
          },
          scope: createNeuralMapScope(state)
        })
      });
      const liveGraph = mergeRuntimeRecordsIntoGraph(
        sanitizeLiveGraphForDisplay(filterLiveGraphForDynamicChatScope(normalizeLiveGraphResponse(response), state), state),
        state
      );

      if (liveGraph.nodes.length === 0) {
        return createLocalLiveGraph(state, "NeuralMap graph query returned no nodes yet.");
      }

      return liveGraph;
    } catch (error) {
      return createLocalLiveGraph(
        state,
        error instanceof Error ? error.message : "Unknown NeuralMap graph query error"
      );
    }
  }

  async getSimulationContext(state: AppState, query: string): Promise<ContextPack> {
    if (!this.settings.enabled) {
      return createLocalContextPack(state, query);
    }

    try {
      const response = await this.request<NeuralMapContextPackResponse>("/context/compose", {
        method: "POST",
        body: JSON.stringify({
          objective: `Continue ${state.simulation.title}`,
          agent_id: createNeuralMapAgentId(state),
          session_id: state.simulation.activeSessionId,
          task_type: "simulation_turn",
          profile_id: SIMULATION_MEMORY_PROFILE_ID,
          query,
          token_budget: this.settings.tokenBudget,
          context_policy: {
            sections: ["current_scene", "canonical_state", "perspective_state", "relevant_history", "open_threads"],
            filters: {
              simulation_id: state.simulation.id,
              session_id: state.simulation.activeSessionId,
              active_session_id: state.simulation.activeSessionId,
              progress_run_id: state.activeProgressRunId,
              run_id: state.activeProgressRunId
            },
            ranking: {
              semantic_similarity: 0.35,
              graph_relevance: 0.25,
              recency: 0.15,
              importance: 0.2,
              active_state_bonus: 0.05
            }
          },
          scope: createNeuralMapScope(state)
        })
      }, NEURALMAP_GRAPH_REQUEST_TIMEOUT_MS);

      return createContextPackFromNeuralMapPack(state, response, query);
    } catch {
      return this.getLegacySimulationContext(state, query);
    }
  }

  private async getLegacySimulationContext(state: AppState, query: string): Promise<ContextPack> {
    try {
      const response = await this.request<NeuralMapSimulationContextResponse>("/simulation/context", {
        method: "POST",
        body: JSON.stringify({
          simulation_id: state.simulation.id,
          session_id: state.simulation.activeSessionId,
          progress_run_id: state.activeProgressRunId,
          run_id: state.activeProgressRunId,
          agent_id: createNeuralMapAgentId(state),
          query,
          token_budget: this.settings.tokenBudget,
          scope: createNeuralMapScope(state)
        })
      });

      if (!response.pack) {
        return createLocalContextPack(state, query);
      }

      const rawEvidence = response.pack.evidence.map((item) => ({
        nodeId: item.node_id,
        snippet: item.snippet,
        score: item.score,
        reason: "NeuralMap continuity evidence"
      }));
      const scopedEvidence = filterScopedContextEvidence(state, rawEvidence);
      const castScopedEvidence = scopeContextEvidenceForCurrentSceneCast(state, scopedEvidence, query);
      const localEvidence = createLocalContextEvidence(state);
      const displayEvidence = uniqueContextEvidence([
        ...filterPublicContextEvidence(state, castScopedEvidence),
        ...localEvidence
      ])
        .sort((a, b) => b.score - a.score)
        .slice(0, 12);
      const moduleEvidence = uniqueContextEvidence([...castScopedEvidence, ...localEvidence])
        .sort((a, b) => b.score - a.score)
        .slice(0, 24);

      return {
        id: response.pack.id,
        simulationId: state.simulation.id,
        sessionId: getSafeContextPackSessionId(state, response.pack.session_id),
        objective: response.pack.objective,
        tokenBudget: state.neuralMap.tokenBudget,
        evidence: displayEvidence,
        moduleEvidence,
        decisions: response.pack.decisions ?? [],
        blockers: response.pack.blockers ?? [],
        createdAt: response.pack.created_at,
        source: "neuralmap"
      };
    } catch {
      return createLocalContextPack(state, query);
    }
  }

  async upsertGraphDocument(
    state: AppState,
    document: {
      id: string;
      title: string;
      body: string;
      kind: string;
      tags: string[];
      importance: number;
    }
  ): Promise<boolean> {
    if (!this.settings.enabled) {
      return false;
    }

    try {
      await this.request<unknown>("/ingest/document", {
        method: "POST",
        body: JSON.stringify({
          id: document.id,
          title: document.title,
          uri: `dynamicchat://${state.simulation.id}/neural-editor/${encodeURIComponent(document.id)}`,
          body: document.body,
          scope: createNeuralMapScope(state),
          metadata: {
            ...createScopeMetadata(state),
            agent_id: createNeuralMapAgentId(state),
            source: "dynamicchat_neural_editor",
            document_id: document.id,
            simulation_id: state.simulation.id,
            session_id: state.simulation.activeSessionId,
            kind: document.kind,
            tags: document.tags,
            importance: document.importance,
            generated_record: true,
            mutable_runtime_record: true,
            updated_at: new Date().toISOString()
          }
        })
      });

      return true;
    } catch {
      return false;
    }
  }

  async createHandoff(state: AppState, nextSessionId: string, handoffEvent: MemoryEvent): Promise<SessionHandoff> {
    if (!this.settings.enabled) {
      return createLocalSessionHandoff(state, nextSessionId, handoffEvent);
    }

    try {
      const response = await this.request<NeuralMapHandoffResponse>("/context/handoff", {
        method: "POST",
        body: JSON.stringify({
          from_run_id: state.simulation.activeSessionId,
          to_session_id: nextSessionId,
          objective: createHandoffQuery(state),
          current_status: handoffEvent.content,
          open_loops: state.memoryEvents.slice(-3).map((event) => event.content.slice(0, 240)),
          constraints: [
            `simulation_id:${state.simulation.id}`,
            `progress_run_id:${state.activeProgressRunId}`,
            `handoff_event_id:${handoffEvent.id}`,
            handoffEvent.neuralMapNodeId ? `handoff_event_node_id:${handoffEvent.neuralMapNodeId}` : undefined
          ].filter((item): item is string => Boolean(item)),
          recommended_next_actions: [
            "Call /context/compose with profile_id=simulation-memory for the new session before the next narrative turn.",
            "Preserve recent relationship, promise, location, and unresolved clue state."
          ],
          scope: createNeuralMapScope(state)
        })
      });
      const pack = normalizeHandoffResponse(response);
      const evidenceNodeIds = readHandoffEvidenceNodeIds(pack);

      return {
        id: pack.id ?? createId("handoff"),
        simulationId: state.simulation.id,
        previousSessionId: state.simulation.activeSessionId,
        nextSessionId,
        summary: pack.summary ?? pack.context_summary ?? pack.current_status ?? handoffEvent.content,
        evidenceNodeIds: evidenceNodeIds.length > 0 ? evidenceNodeIds : [handoffEvent.neuralMapNodeId ?? handoffEvent.id],
        createdAt: pack.created_at ?? new Date().toISOString(),
        source: "neuralmap"
      };
    } catch (error) {
      return {
        ...createLocalSessionHandoff(state, nextSessionId, handoffEvent),
        error: error instanceof Error ? error.message : "Unknown NeuralMap handoff error"
      };
    }
  }

  private async request<T>(path: string, init: RequestInit, timeoutMs = NEURALMAP_REQUEST_TIMEOUT_MS): Promise<T> {
    const controller = new AbortController();
    const timeoutId = globalThis.setTimeout(() => controller.abort(), timeoutMs);
    let response: Response;
    try {
      response = await fetch(`${this.settings.baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
        headers: {
          "content-type": "application/json",
          ...(init.headers ?? {})
        }
      });
    } finally {
      globalThis.clearTimeout(timeoutId);
    }

    if (!response.ok) {
      throw new Error(`NeuralMap request failed: ${response.status}`);
    }

    return response.json() as Promise<T>;
  }

}

function createGraphDeltaRequest(state: AppState, delta: MemoryDelta): NeuralMapGraphDeltaRequest {
  const scope = createNeuralMapScope(state);
  const actorRefs = collectGraphActorRefs(state, delta);
  const sceneNode = createSceneNeuron(state, delta);
  const upsertNeurons = [
    ...[...actorRefs.values()].map((actor) => createCharacterNeuron(state, actor)),
    sceneNode,
    ...delta.upsertRecords.map((record) => createMemoryNeuron(state, delta, record, actorRefs))
  ];
  const upsertSynapses = delta.upsertRecords.flatMap((record) => createMemorySynapses(state, delta, record, actorRefs));
  const temporalOperations = delta.upsertRecords
    .filter((record) => record.kind === "state")
    .flatMap((record) => {
      const owner = getRecordOwnerRef(state, record, actorRefs);
      const stateType = record.stateType ?? "State";
      if (!owner) {
        return [];
      }

      return [
        {
          operation: "supersede_current" as const,
          selector: {
            label: "State",
            profile_id: SIMULATION_MEMORY_PROFILE_ID,
            properties: {
              owner_id: owner.nodeId,
              state_type: stateType,
              simulation_id: state.simulation.id,
              progress_run_id: state.activeProgressRunId,
              run_id: state.activeProgressRunId
            }
          },
          valid_to: delta.simTime,
          superseded_by: createMemoryGraphNeuronId(state, delta, record)
        }
      ];
    });

  return {
    idempotency_key: [
      "dynamicchat",
      graphIdPart(state.simulation.id),
      graphIdPart(state.simulation.activeSessionId),
      graphIdPart(delta.turnId),
      "simulation-memory-v1"
    ].join(":"),
    profile_id: SIMULATION_MEMORY_PROFILE_ID,
    source: {
      system: "dynamicchat",
      run_id: state.activeProgressRunId,
      turn_id: delta.turnId,
      raw_ref: `dynamicchat://${state.simulation.id}/sessions/${state.simulation.activeSessionId}/turns/${delta.turnId}`,
      metadata: {
        agent_id: createNeuralMapAgentId(state),
        simulation_id: state.simulation.id,
        session_id: state.simulation.activeSessionId,
        progress_run_id: state.activeProgressRunId,
        run_id: state.activeProgressRunId,
        scene_id: delta.sceneId,
        sim_time: delta.simTime
      }
    },
    scope,
    upsert_neurons: uniqueNeurons(upsertNeurons),
    upsert_synapses: uniqueSynapses(upsertSynapses),
    temporal_operations: temporalOperations
  };
}

interface GraphActorRef {
  nodeId: string;
  localId: string;
  name: string;
}

function collectGraphActorRefs(state: AppState, delta: MemoryDelta): Map<string, GraphActorRef> {
  const refs = new Map<string, GraphActorRef>();
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state));
  for (const character of state.characters.filter((candidate) => activeCharacterIds.has(candidate.id) || state.characters.length === 1)) {
    const ref = createGraphActorRef(state, character.id, character.name);
    refs.set(ref.localId, ref);
  }

  for (const record of delta.upsertRecords) {
    for (const ref of [
      createOptionalGraphActorRef(state, record.actorId, record.actorName),
      createOptionalGraphActorRef(state, record.ownerId),
      createOptionalCharacterGraphRef(state, record.targetId),
      ...(record.observers ?? []).map((observerId) => createOptionalCharacterGraphRef(state, observerId))
    ]) {
      if (ref) {
        refs.set(ref.localId, ref);
      }
    }
  }

  return refs;
}

function createGraphActorRef(state: AppState, id: string, name?: string): GraphActorRef {
  const character = state.characters.find(
    (candidate) =>
      candidate.id === id ||
      candidate.name === name ||
      candidate.name.toLowerCase() === name?.toLowerCase()
  );
  const localId = character?.id ?? id;
  const actorName = character?.name ?? name ?? id;
  return {
    nodeId: `simulation:${graphIdPart(state.simulation.id)}:person:${graphIdPart(localId)}`,
    localId,
    name: actorName
  };
}

function createOptionalGraphActorRef(state: AppState, id?: string, name?: string): GraphActorRef | undefined {
  if (!id && !name) {
    return undefined;
  }
  const fallbackId = id ?? name ?? "unknown";
  return createGraphActorRef(state, fallbackId, name);
}

function createOptionalCharacterGraphRef(state: AppState, id?: string, name?: string): GraphActorRef | undefined {
  if (!id && !name) {
    return undefined;
  }

  const character = state.characters.find(
    (candidate) =>
      candidate.id === id ||
      candidate.name === name ||
      candidate.name.toLowerCase() === name?.toLowerCase()
  );
  return character ? createGraphActorRef(state, character.id, character.name) : undefined;
}

function createCharacterNeuron(state: AppState, actor: GraphActorRef): NeuralMapGraphDeltaNeuron {
  const character = state.characters.find((candidate) => candidate.id === actor.localId);
  return {
    id: actor.nodeId,
    type: "Person",
    labels: ["Character", "Entity", "Actor"],
    title: actor.name,
    summary: [character?.role, character?.summary, character?.relationship, character?.currentMood]
      .filter(Boolean)
      .join(" / ") || `${actor.name} participates in ${state.simulation.title}.`,
    source_system: "runtime",
    trust_score: 0.86,
    freshness_score: 0.9,
    importance_score: 0.74,
    confidence: 0.9,
    ontology: {
      profile_id: SIMULATION_MEMORY_PROFILE_ID,
      type: "Character"
    },
    properties: {
      name: actor.name,
      local_character_id: actor.localId,
      simulation_id: state.simulation.id,
      session_id: state.simulation.activeSessionId
    },
    metadata: {
      ...createScopeMetadata(state),
      agent_id: createNeuralMapAgentId(state),
      source: "dynamicchat",
      kind: "simulation_character",
      graph_profile_id: SIMULATION_MEMORY_PROFILE_ID
    }
  };
}

function createSceneNeuron(state: AppState, delta: MemoryDelta): NeuralMapGraphDeltaNeuron {
  const sceneNodeId = createSceneNodeId(state, delta.sceneId);
  return {
    id: sceneNodeId,
    type: "Session",
    labels: ["Scene", "Entity", "Temporal"],
    title: `${state.simulation.title} scene`,
    summary: `Scene ${delta.sceneId} at ${delta.simTime}`,
    source_system: "runtime",
    trust_score: 0.78,
    freshness_score: 1,
    importance_score: 0.64,
    confidence: 0.78,
    valid_from: delta.simTime,
    ontology: {
      profile_id: SIMULATION_MEMORY_PROFILE_ID,
      type: "Scene"
    },
    properties: {
      session_id: state.simulation.activeSessionId,
      simulation_id: state.simulation.id,
      progress_run_id: state.activeProgressRunId,
      run_id: state.activeProgressRunId,
      scene_id: delta.sceneId,
      sim_time: delta.simTime
    },
    metadata: {
      ...createScopeMetadata(state),
      agent_id: createNeuralMapAgentId(state),
      source: "dynamicchat",
      kind: "simulation_scene",
      progress_run_id: state.activeProgressRunId,
      run_id: state.activeProgressRunId,
      graph_profile_id: SIMULATION_MEMORY_PROFILE_ID
    }
  };
}

function createMemoryNeuron(
  state: AppState,
  delta: MemoryDelta,
  record: MemoryDeltaRecord,
  actorRefs: ReadonlyMap<string, GraphActorRef>
): NeuralMapGraphDeltaNeuron {
  const ontologyType = getRecordOntologyType(record);
  const owner = getRecordOwnerRef(state, record, actorRefs);
  const holder = getRecordActorRef(state, record, actorRefs) ?? owner;
  const observer = getRecordObserverRefs(state, record, actorRefs)[0] ?? holder;
  const target = getRecordTargetRef(state, record, actorRefs);
  const stateType = record.stateType ?? "State";
  const stateValue = record.value ?? record.content;
  const properties: Record<string, unknown> = {
    simulation_id: state.simulation.id,
    session_id: state.simulation.activeSessionId,
    progress_run_id: state.activeProgressRunId,
    run_id: state.activeProgressRunId,
    scene_id: delta.sceneId,
    scene_node_id: createSceneNodeId(state, delta.sceneId),
    sim_time: delta.simTime,
    turn_id: delta.turnId,
    source_turn_id: delta.turnId,
    memory_record_id: record.id,
    memory_kind: record.kind,
    memory_layer: record.layer,
    content: record.content,
    event_type: record.eventType ?? getDefaultEventType(record),
    local_actor_id: record.actorId,
    local_owner_id: record.ownerId,
    local_target_id: record.targetId,
    target_character_node_id: target?.nodeId,
    observer_ids: record.observers,
    importance_reasons: record.importanceReasons
  };

  if (ontologyType === "State") {
    properties.owner_id = owner?.nodeId ?? "world";
    properties.state_type = stateType;
    properties.value = stateValue;
  }
  if (ontologyType === "Observation") {
    properties.observer_id = observer?.nodeId ?? "unknown";
    properties.target_id = target?.nodeId ?? record.targetId ?? createMemoryGraphNeuronId(state, delta, record);
  }
  if (ontologyType === "Belief") {
    properties.holder_id = holder?.nodeId ?? "unknown";
    properties.content = record.content;
  }

  return {
    id: createMemoryGraphNeuronId(state, delta, record),
    type: getRecordNodeType(record, ontologyType),
    labels: getRecordLabels(record, ontologyType),
    title: createRecordTitle(record, holder ?? owner),
    summary: record.content,
    source_system: "runtime",
    trust_score: record.confidence,
    freshness_score: 1,
    importance_score: record.importance,
    confidence: record.confidence,
    lifecycle_status: "active",
    valid_from: record.kind === "state" ? delta.simTime : undefined,
    valid_to: record.kind === "state" ? null : undefined,
    ontology: {
      profile_id: SIMULATION_MEMORY_PROFILE_ID,
      type: ontologyType
    },
    properties,
    metadata: {
      ...createScopeMetadata(state),
      agent_id: createNeuralMapAgentId(state),
      source: "dynamicchat",
      kind: record.kind === "open_thread" ? "simulation_open_loop" : `simulation_${record.kind}`,
      graph_profile_id: SIMULATION_MEMORY_PROFILE_ID,
      progress_run_id: state.activeProgressRunId,
      run_id: state.activeProgressRunId,
      memory_kind: record.kind,
      state_type: record.stateType,
      event_type: record.eventType,
      scene_id: delta.sceneId,
      sim_time: delta.simTime
    }
  };
}

function createMemorySynapses(
  state: AppState,
  delta: MemoryDelta,
  record: MemoryDeltaRecord,
  actorRefs: ReadonlyMap<string, GraphActorRef>
): NeuralMapGraphDeltaSynapse[] {
  const nodeId = createMemoryGraphNeuronId(state, delta, record);
  const sceneNodeId = createSceneNodeId(state, delta.sceneId);
  const actor = getRecordActorRef(state, record, actorRefs);
  const owner = getRecordOwnerRef(state, record, actorRefs);
  const target = getRecordTargetRef(state, record, actorRefs);
  const synapses: NeuralMapGraphDeltaSynapse[] = [];

  synapses.push(createSynapse({
    from: sceneNodeId,
    to: nodeId,
    type: "SCENE_HAS_MEMORY",
    confidence: record.confidence,
    weight: record.kind === "state" ? 0.72 : 0.9,
    validFrom: delta.simTime,
    properties: createBaseSynapseProperties(state, delta, record, {
      scene_node_id: sceneNodeId,
      memory_node_id: nodeId
    })
  }));

  for (const participant of collectSceneParticipantRefs(state, record, actorRefs)) {
    synapses.push(createSynapse({
      from: sceneNodeId,
      to: participant.ref.nodeId,
      type: "SCENE_PARTICIPANT",
      confidence: record.confidence,
      weight: participant.role === "observer" ? 0.72 : 0.82,
      validFrom: delta.simTime,
      properties: createBaseSynapseProperties(state, delta, record, {
        role: participant.role,
        local_character_id: participant.ref.localId,
        scene_node_id: sceneNodeId,
        memory_node_id: nodeId
      })
    }));
  }

  if (record.kind === "state" && owner) {
    const stateType = record.stateType ?? "State";
    synapses.push(createSynapse({
      from: owner.nodeId,
      to: nodeId,
      type: "HAS_CURRENT_STATE",
      confidence: record.confidence,
      weight: 1,
      validFrom: delta.simTime,
      properties: createBaseSynapseProperties(state, delta, record, {
        current_pointer_key: `${owner.nodeId}:${state.activeProgressRunId}:${stateType}`,
        owner_id: owner.nodeId,
        state_type: stateType
      })
    }));
  }

  const beliefHolder = actor ?? owner;
  if (record.kind === "belief" && beliefHolder) {
    synapses.push(createSynapse({
      from: beliefHolder.nodeId,
      to: nodeId,
      type: "BELIEVES",
      confidence: record.confidence,
      weight: 0.82,
      validFrom: delta.simTime,
      properties: createBaseSynapseProperties(state, delta, record, {
        source: "compiled_memory_delta",
        holder_id: beliefHolder.nodeId
      })
    }));
  }

  if (record.kind === "observation") {
    for (const observer of getRecordObserverRefs(state, record, actorRefs)) {
      synapses.push(createSynapse({
        from: observer.nodeId,
        to: nodeId,
        type: "OBSERVED",
        confidence: record.confidence,
        weight: 0.88,
        validFrom: delta.simTime,
        properties: createBaseSynapseProperties(state, delta, record, {
          method: "compiled_from_turn",
          observer_id: observer.nodeId
        })
      }));
    }
  }

  if (record.kind !== "state" && record.kind !== "belief" && record.kind !== "observation" && actor) {
    synapses.push(createSynapse({
      from: actor.nodeId,
      to: nodeId,
      type: "ACTOR_OF",
      confidence: record.confidence,
      weight: 0.78,
      validFrom: delta.simTime,
      properties: createBaseSynapseProperties(state, delta, record, {
        actor_id: actor.nodeId
      })
    }));
  }

  if (target) {
    synapses.push(createSynapse({
      from: target.nodeId,
      to: nodeId,
      type: "TARGET_OF",
      confidence: record.confidence,
      weight: 0.76,
      validFrom: delta.simTime,
      properties: createBaseSynapseProperties(state, delta, record, {
        target_id: target.nodeId,
        local_target_id: target.localId
      })
    }));
  }

  if (record.kind === "relationship" && actor && target) {
    synapses.push(createSynapse({
      from: actor.nodeId,
      to: target.nodeId,
      type: "RELATIONSHIP_TO",
      confidence: record.confidence,
      weight: Math.max(0.72, record.importance),
      validFrom: delta.simTime,
      properties: createBaseSynapseProperties(state, delta, record, {
        relationship_memory_node_id: nodeId,
        local_actor_id: actor.localId,
        local_target_id: target.localId,
        relationship_summary: record.content
      })
    }));
  }

  return synapses;
}

function createBaseSynapseProperties(
  state: AppState,
  delta: MemoryDelta,
  record: MemoryDeltaRecord,
  extra: Record<string, unknown> = {}
): Record<string, unknown> {
  return {
    simulation_id: state.simulation.id,
    session_id: state.simulation.activeSessionId,
    progress_run_id: state.activeProgressRunId,
    run_id: state.activeProgressRunId,
    source_turn_id: delta.turnId,
    scene_id: delta.sceneId,
    sim_time: delta.simTime,
    memory_record_id: record.id,
    memory_kind: record.kind,
    event_type: record.eventType ?? getDefaultEventType(record),
    ...extra
  };
}

function collectSceneParticipantRefs(
  state: AppState,
  record: MemoryDeltaRecord,
  actorRefs: ReadonlyMap<string, GraphActorRef>
): Array<{ ref: GraphActorRef; role: "actor" | "owner" | "observer" }> {
  const participants = [
    ...withParticipantRole(getRecordActorRef(state, record, actorRefs), "actor" as const),
    ...withParticipantRole(getRecordOwnerRef(state, record, actorRefs), "owner" as const),
    ...getRecordObserverRefs(state, record, actorRefs).map((ref) => ({ ref, role: "observer" as const }))
  ];
  const byId = new Map<string, { ref: GraphActorRef; role: "actor" | "owner" | "observer" }>();
  for (const participant of participants) {
    const existing = byId.get(participant.ref.localId);
    if (!existing || getParticipantRoleRank(participant.role) < getParticipantRoleRank(existing.role)) {
      byId.set(participant.ref.localId, participant);
    }
  }
  return [...byId.values()];
}

function withParticipantRole<T extends "actor" | "owner" | "observer">(
  ref: GraphActorRef | undefined,
  role: T
): Array<{ ref: GraphActorRef; role: T }> {
  return ref ? [{ ref, role }] : [];
}

function getParticipantRoleRank(role: "actor" | "owner" | "observer"): number {
  if (role === "actor") {
    return 0;
  }
  if (role === "owner") {
    return 1;
  }
  return 2;
}

function createSynapse(input: {
  from: string;
  to: string;
  type: string;
  confidence: number;
  weight: number;
  validFrom: string;
  properties: Record<string, unknown>;
}): NeuralMapGraphDeltaSynapse {
  return {
    id: `syn:${hashText(`${input.from}:${input.type}:${input.to}:${input.validFrom}`)}`,
    from: input.from,
    to: input.to,
    type: input.type,
    labels: [input.type],
    weight: input.weight,
    confidence: input.confidence,
    valid_from: input.validFrom,
    ontology: {
      profile_id: SIMULATION_MEMORY_PROFILE_ID,
      type: input.type
    },
    properties: input.properties,
    metadata: {
      source: "dynamicchat",
      graph_profile_id: SIMULATION_MEMORY_PROFILE_ID,
      synapse_type: input.type
    }
  };
}

function createContextPackFromNeuralMapPack(
  state: AppState,
  pack: NeuralMapContextPackResponse,
  query: string
): ContextPack {
  const sections = normalizeContextSections(state, pack.sections, query);
  const sectionSummaryEvidence = createSectionedContextEvidence(state, pack, sections);
  const rawEvidence = normalizeNeuralMapEvidenceItems(state, pack.evidence, "NeuralMap profile evidence");
  const neuralEvidence = uniqueContextEvidence([
    ...(sectionSummaryEvidence ? [sectionSummaryEvidence] : []),
    ...rawEvidence
  ]);
  const castScopedNeuralEvidence = scopeContextEvidenceForCurrentSceneCast(state, neuralEvidence, query);
  const localEvidence = createLocalContextEvidence(state);
  const displayEvidence = uniqueContextEvidence([
    ...filterPublicContextEvidence(state, castScopedNeuralEvidence),
    ...localEvidence
  ])
    .sort((a, b) => b.score - a.score)
    .slice(0, 12);
  const moduleEvidence = uniqueContextEvidence([...castScopedNeuralEvidence, ...localEvidence])
    .sort((a, b) => b.score - a.score)
    .slice(0, 24);

  return {
    id: pack.id,
    simulationId: state.simulation.id,
    sessionId: getSafeContextPackSessionId(state, pack.session_id),
    objective: pack.objective,
    tokenBudget: state.neuralMap.tokenBudget,
    evidence: displayEvidence,
    sections,
    moduleEvidence,
    decisions: pack.decisions ?? [],
    blockers: pack.blockers ?? [],
    createdAt: pack.created_at,
    source: "neuralmap"
  };
}

function normalizeContextSections(
  state: AppState,
  sections: NeuralMapContextPackResponse["sections"],
  currentText = ""
): ContextPack["sections"] {
  if (!sections) {
    return undefined;
  }

  const scopedSections = Object.entries(sections)
    .map(([section, evidence]) => [
      section,
      scopeContextEvidenceForCurrentSceneCast(
        state,
        normalizeNeuralMapEvidenceItems(state, evidence, `NeuralMap section: ${formatContextSectionName(section)}`),
        currentText
      )
    ] as const)
    .filter(([, evidence]) => evidence.length > 0);

  return scopedSections.length > 0 ? Object.fromEntries(scopedSections) : undefined;
}

function normalizeNeuralMapEvidenceItems(
  state: AppState,
  evidence: NeuralMapEvidenceItem[],
  reason: string
): ContextPack["evidence"] {
  return evidence
    .filter((item) => isScopedNeuralMapEvidenceItem(state, item))
    .map((item) => ({
      nodeId: item.node_id,
      snippet: item.snippet,
      score: item.score,
      reason
    }));
}

function getSafeContextPackSessionId(state: AppState, sessionId: string): string {
  return isAllowedRuntimeSessionId(state, sessionId) ? sessionId : state.simulation.activeSessionId;
}

function createSectionedContextEvidence(
  state: AppState,
  pack: NeuralMapContextPackResponse,
  sections: ContextPack["sections"]
): ContextPack["evidence"][number] | undefined {
  const entries = Object.entries(sections ?? {}).filter(([, evidence]) => evidence.length > 0);
  if (entries.length === 0) {
    return undefined;
  }

  const snippet = [
    "# NeuralMap Simulation Context",
    ...entries.flatMap(([section, evidence]) => [
      "",
      `## ${formatContextSectionName(section)}`,
      ...evidence.slice(0, 5).map((item) => `- ${item.snippet}`)
    ])
  ].join("\n");

  return {
    nodeId: `simulation:${state.simulation.id}:context:${pack.id}`,
    snippet,
    score: 0.98,
    reason: "NeuralMap sectioned simulation context"
  };
}

function formatContextSectionName(section: string): string {
  const known: Record<string, string> = {
    current_scene: "Current Scene",
    canonical_state: "Canonical State",
    perspective_state: "Character Knowledge",
    relevant_history: "Relevant History",
    open_threads: "Open Threads"
  };
  return known[section] ?? section.replace(/[_-]+/gu, " ").replace(/\b\w/gu, (letter) => letter.toUpperCase());
}

function scopeContextEvidenceForCurrentSceneCast(
  state: AppState,
  evidence: ContextPack["evidence"],
  currentText = ""
): ContextPack["evidence"] {
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state, currentText));

  return evidence.flatMap((item) => {
    const mentionedCharacterIds = getEvidenceMentionedCharacterIds(state, item);
    if (mentionedCharacterIds.length === 0) {
      return [item];
    }

    if (state.characters.length === 1 && mentionedCharacterIds.includes(state.characters[0].id)) {
      return [item];
    }

    const activeMentions = mentionedCharacterIds.filter((characterId) => activeCharacterIds.has(characterId));
    if (activeMentions.length === 0) {
      return [];
    }

    const offStageNames = mentionedCharacterIds
      .filter((characterId) => !activeCharacterIds.has(characterId))
      .map((characterId) => state.characters.find((character) => character.id === characterId)?.name)
      .filter((name): name is string => Boolean(name));

    return offStageNames.length > 0
      ? [
          {
            ...item,
            snippet: appendOffStageCharacterGuard(item.snippet, offStageNames)
          }
        ]
      : [item];
  });
}

function getEvidenceMentionedCharacterIds(
  state: AppState,
  item: ContextPack["evidence"][number]
): string[] {
  const evidenceText = [item.nodeId, item.reason, item.snippet].join("\n");
  return state.characters
    .filter((character) =>
      evidenceContainsCharacterTerm(evidenceText, character.id) ||
      evidenceContainsCharacterTerm(evidenceText, character.name) ||
      item.nodeId.endsWith(`:person:${graphIdPart(character.id)}`)
    )
    .map((character) => character.id);
}

function evidenceContainsCharacterTerm(text: string, term: string): boolean {
  const normalizedText = text.toLowerCase();
  const normalizedTerm = term.trim().toLowerCase();
  if (!normalizedTerm) {
    return false;
  }

  if (/^[a-z0-9_-]+$/iu.test(normalizedTerm)) {
    return new RegExp(`(^|[^\\p{L}\\p{N}_-])${escapeRegExp(normalizedTerm)}($|[^\\p{L}\\p{N}_-])`, "iu").test(
      normalizedText
    );
  }

  if (/^[\u3131-\uD79D]$/u.test(normalizedTerm)) {
    return new RegExp(
      `(^|[^\\p{L}\\p{N}_-])${escapeRegExp(normalizedTerm)}(?:은|는|이|가|을|를|와|과|도|만|에|로|의)?($|[^\\p{L}\\p{N}_-])`,
      "iu"
    ).test(normalizedText);
  }

  return normalizedText.includes(normalizedTerm);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

function appendOffStageCharacterGuard(snippet: string, characterNames: string[]): string {
  if (/Off-stage character reference only|무대 밖 인물 참조/iu.test(snippet)) {
    return snippet;
  }

  return [
    snippet,
    `[무대 밖 인물 참조 전용: ${characterNames.join(", ")}는 이 기록에 언급되지만 최근 transcript/current action의 현재 장면 등장 인물은 아니다.]`
  ].join("\n");
}

function getRecordOntologyType(record: MemoryDeltaRecord): "Event" | "State" | "Observation" | "Belief" {
  if (record.kind === "state") {
    return "State";
  }
  if (record.kind === "belief") {
    return "Belief";
  }
  if (record.kind === "observation") {
    return "Observation";
  }
  return "Event";
}

function getRecordNodeType(record: MemoryDeltaRecord, ontologyType: string): string {
  if (ontologyType === "State") {
    return "Artifact";
  }
  if (ontologyType === "Belief") {
    return "Decision";
  }
  if (record.kind === "summary") {
    return "Summary";
  }
  return "Task";
}

function getRecordLabels(record: MemoryDeltaRecord, ontologyType: string): string[] {
  const labels = [ontologyType];
  if (record.kind === "state") {
    labels.push("State", "TemporalFact");
  }
  if (record.kind === "observation") {
    labels.push("Perspective", "Observation", "Event");
  }
  if (record.kind === "belief") {
    labels.push("Perspective", "Belief", "Uncertain");
  }
  if (record.kind === "event" || record.kind === "goal" || record.kind === "relationship" || record.kind === "summary") {
    labels.push("Event", "Temporal");
  }
  if (record.kind === "open_thread") {
    labels.push("Event", "OpenThread", "Temporal");
  }
  return uniqueStrings(labels);
}

function createRecordTitle(record: MemoryDeltaRecord, actor: GraphActorRef | undefined): string {
  if (record.kind === "state") {
    return `${actor?.name ?? "World"} ${record.stateType ?? "State"}`;
  }
  if (record.kind === "observation") {
    return `${actor?.name ?? "Observer"} observation`;
  }
  if (record.kind === "belief") {
    return `${actor?.name ?? "Character"} belief`;
  }
  if (record.kind === "open_thread") {
    return "Open simulation thread";
  }
  return record.eventType ?? "Simulation event";
}

function getDefaultEventType(record: MemoryDeltaRecord): string {
  if (record.kind === "goal") {
    return "GoalUpdated";
  }
  if (record.kind === "relationship") {
    return "RelationshipUpdated";
  }
  if (record.kind === "open_thread") {
    return "OpenThread";
  }
  if (record.kind === "summary") {
    return "Summary";
  }
  return record.eventType ?? "SceneEvent";
}

function getRecordActorRef(
  state: AppState,
  record: MemoryDeltaRecord,
  actorRefs: ReadonlyMap<string, GraphActorRef>
): GraphActorRef | undefined {
  return record.actorId ? actorRefs.get(record.actorId) ?? createOptionalGraphActorRef(state, record.actorId, record.actorName) : undefined;
}

function getRecordOwnerRef(
  state: AppState,
  record: MemoryDeltaRecord,
  actorRefs: ReadonlyMap<string, GraphActorRef>
): GraphActorRef | undefined {
  const ownerId = record.ownerId ?? record.actorId;
  return ownerId ? actorRefs.get(ownerId) ?? createOptionalGraphActorRef(state, ownerId, record.actorName) : undefined;
}

function getRecordTargetRef(
  state: AppState,
  record: MemoryDeltaRecord,
  actorRefs: ReadonlyMap<string, GraphActorRef>
): GraphActorRef | undefined {
  return record.targetId ? actorRefs.get(record.targetId) ?? createOptionalCharacterGraphRef(state, record.targetId) : undefined;
}

function getRecordObserverRefs(
  state: AppState,
  record: MemoryDeltaRecord,
  actorRefs: ReadonlyMap<string, GraphActorRef>
): GraphActorRef[] {
  const observers = record.observers?.length ? record.observers : record.actorId ? [record.actorId] : [];
  return observers
    .map((observerId) => actorRefs.get(observerId) ?? createOptionalGraphActorRef(state, observerId))
    .filter((ref): ref is GraphActorRef => Boolean(ref));
}

function createSceneNodeId(state: AppState, sceneId: string): string {
  return `simulation:${graphIdPart(state.simulation.id)}:${sceneId.split(":").map(graphIdPart).join(":")}`;
}

function uniqueNeurons(neurons: NeuralMapGraphDeltaNeuron[]): NeuralMapGraphDeltaNeuron[] {
  const byId = new Map<string, NeuralMapGraphDeltaNeuron>();
  for (const neuron of neurons) {
    byId.set(neuron.id, { ...(byId.get(neuron.id) ?? {}), ...neuron });
  }
  return [...byId.values()];
}

function uniqueSynapses(synapses: NeuralMapGraphDeltaSynapse[]): NeuralMapGraphDeltaSynapse[] {
  const byId = new Map<string, NeuralMapGraphDeltaSynapse>();
  for (const synapse of synapses) {
    byId.set(synapse.id ?? `${synapse.from}:${synapse.type}:${synapse.to}`, synapse);
  }
  return [...byId.values()];
}

function graphIdPart(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}_-]+/gu, "-")
    .replace(/^-+|-+$/gu, "")
    .slice(0, 80) || "unknown";
}

function hashText(value: string): string {
  let hash = 5381;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 33) ^ value.charCodeAt(index);
  }
  return (hash >>> 0).toString(36);
}

function createNeuralMapScope(state: AppState): NeuralMapScope {
  const scope = state.security.scope;
  return {
    tenant_id: scope.ownerId,
    workspace_id: scope.workspaceId,
    project_id: scope.projectId,
    owner_scope: scope.ownerId
  };
}

function createNeuralMapAgentId(state: AppState): string {
  return `dynamicchat:${graphIdPart(state.simulation.id)}`;
}

function createLiveGraphQuery(state: AppState, query: string): string {
  const latestMessage = state.messages.at(-1)?.content;
  const latestMemory = state.memoryEvents.at(-1)?.content;
  const characterNames = state.characters.map((character) => character.name).join(", ");
  const personaLine = createUserPersonaLine(state);
  return [
    query,
    state.simulation.title,
    characterNames ? `characters: ${characterNames}` : undefined,
    personaLine ? `user persona: ${personaLine}` : undefined,
    latestMessage ? `latest message: ${latestMessage}` : undefined,
    latestMemory ? `latest memory: ${latestMemory}` : undefined
  ]
    .filter((item): item is string => Boolean(item?.trim()))
    .join("\n")
    .slice(0, 1200) || `simulation ${state.simulation.id}`;
}

function normalizeLiveGraphResponse(response: NeuralMapGraphQueryResponse): NeuralMapLiveGraph {
  const neighborhood = response.neighborhood;
  const nodes = (neighborhood?.nodes ?? []).map((node): NeuralMapLiveNode => ({
    id: node.id,
    type: node.type ?? "Document",
    labels: node.labels,
    title: node.title ?? node.id,
    summary: node.summary ?? node.content_ref ?? "",
    importanceScore: clampScore(node.importance_score),
    sourceSystem: node.source_system,
    kind:
      node.ontology?.type ??
      readMetadataText(node.properties, "memory_kind") ??
      readMetadataText(node.metadata, "memory_kind") ??
      readMetadataText(node.metadata, "kind"),
    ontology: node.ontology,
    properties: node.properties,
    scope: node.scope,
    metadata: node.metadata
  }));
  const edges = (neighborhood?.edges ?? []).map((edge): NeuralMapLiveEdge => ({
    id: edge.id ?? `edge:${edge.from}:${edge.type ?? "related_to"}:${edge.to}`,
    from: edge.from,
    to: edge.to,
    type: edge.ontology?.type ?? readMetadataText(edge.metadata, "synapse_type") ?? edge.type ?? "related_to",
    labels: edge.labels,
    weight: edge.weight ?? 1,
    confidence: clampScore(edge.confidence),
    ontology: edge.ontology,
    properties: edge.properties,
    scope: edge.scope,
    metadata: edge.metadata
  }));

  return {
    source: "neuralmap",
    nodes,
    edges,
    seedNodeIds: neighborhood?.seed_node_ids ?? [],
    generatedAt: neighborhood?.generated_at ?? new Date().toISOString()
  };
}

function filterLiveGraphForDynamicChatScope(graph: NeuralMapLiveGraph, state: AppState): NeuralMapLiveGraph {
  const nodes = graph.nodes.filter((node) => isDynamicChatScopedGraphNode(state, node));
  const visibleNodeIds = new Set(nodes.map((node) => node.id));

  return {
    ...graph,
    nodes,
    edges: graph.edges.filter((edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to) && scopeMatchesDynamicChat(state, edge.scope ?? readMetadataScope(edge.metadata))),
    seedNodeIds: graph.seedNodeIds.filter((nodeId) => visibleNodeIds.has(nodeId))
  };
}

function isDynamicChatScopedGraphNode(state: AppState, node: NeuralMapLiveNode): boolean {
  const metadataScope = readMetadataScope(node.metadata) ?? readMetadataScope(node.properties);

  if (!scopeMatchesDynamicChat(state, node.scope ?? metadataScope)) {
    return false;
  }

  if (
    !matchesRuntimeRecordScope(state, node.metadata, node.properties) ||
    hasForeignDynamicChatRuntimeReference(state, node.id, node.title, node.summary)
  ) {
    return false;
  }

  if (requiresCurrentProgressGraphScope(node) && !hasCurrentProgressGraphScope(state, node)) {
    return false;
  }

  if (hasExplicitScope(node.scope) || hasExplicitScope(metadataScope)) {
    return true;
  }

  if (isCurrentSimulationNode(state, node.id, node.metadata)) {
    return true;
  }

  if (isKnownRuntimeNodeId(state, node.id)) {
    return true;
  }

  if (
    isDynamicChatSource(node.sourceSystem, node.metadata) &&
    hasCurrentDynamicChatRuntimeReference(state, node.id, node.title, node.summary, ...readRuntimeReferenceValues(node.metadata, node.properties))
  ) {
    return true;
  }

  return false;
}

const PRIVATE_PROMPT_MODULE_KINDS = new Set([
  "main_prompt",
  "sub_prompt",
  "character_prompt",
  "world_lore",
  "scene_rule",
  "safety_policy",
  "style_guide",
  "image_prompt_profile"
]);

const PRIVATE_PROMPT_METADATA_SOURCES = new Set([
  "dynamicchat_prompt_module",
  "dynamicchat_prompt_module_chunk"
]);

function sanitizeLiveGraphForDisplay(graph: NeuralMapLiveGraph, state: AppState): NeuralMapLiveGraph {
  const nodes = graph.nodes.filter((node) => !isPrivatePromptGraphNode(state, node));
  const visibleNodeIds = new Set(nodes.map((node) => node.id));

  return {
    ...graph,
    nodes,
    edges: graph.edges.filter((edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to)),
    seedNodeIds: graph.seedNodeIds.filter((nodeId) => visibleNodeIds.has(nodeId))
  };
}

function mergeRuntimeRecordsIntoGraph(graph: NeuralMapLiveGraph, state: AppState): NeuralMapLiveGraph {
  const localRecords = createLocalLiveGraph(state);
  const hiddenNodeIds = createHiddenRuntimeNodeIds(state);
  const nodes = new Map(graph.nodes.filter((node) => !hiddenNodeIds.has(node.id)).map((node) => [node.id, node]));
  const edges = new Map(graph.edges.filter((edge) => !hiddenNodeIds.has(edge.from) && !hiddenNodeIds.has(edge.to)).map((edge) => [edge.id, edge]));

  for (const node of localRecords.nodes) {
    const existing = nodes.get(node.id);
    nodes.set(node.id, existing ? { ...node, ...existing } : node);
  }

  for (const edge of localRecords.edges) {
    edges.set(edge.id, edge);
  }

  const visibleNodeIds = new Set(nodes.keys());
  return {
    ...graph,
    nodes: [...nodes.values()],
    edges: [...edges.values()].filter((edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to)),
    seedNodeIds: uniqueStrings([...graph.seedNodeIds, ...localRecords.seedNodeIds]).filter((nodeId) => visibleNodeIds.has(nodeId))
  };
}

function createHiddenRuntimeNodeIds(state: AppState): Set<string> {
  return resolvePersonaCharacter(state) ? new Set([`persona:${state.simulation.id}`]) : new Set();
}

function filterPublicContextEvidence(state: AppState, evidence: ContextPack["evidence"]): ContextPack["evidence"] {
  return evidence.filter(
    (item) =>
      !isPrivatePromptNodeId(state, item.nodeId) &&
      !looksLikePromptModuleSnippet(item.snippet) &&
      !matchesPrivatePromptText(state, item.reason, item.snippet)
  );
}

function filterScopedContextEvidence(state: AppState, evidence: ContextPack["evidence"]): ContextPack["evidence"] {
  return evidence.filter((item) => isDynamicChatScopedEvidence(state, item));
}

function isDynamicChatScopedEvidence(state: AppState, item: ContextPack["evidence"][number]): boolean {
  if (hasForeignDynamicChatRuntimeReference(state, item.nodeId, item.reason, item.snippet)) {
    return false;
  }

  if (requiresCurrentProgressEvidenceScope(item.nodeId, item.reason, item.snippet)) {
    return isKnownRuntimeNodeId(state, item.nodeId) || hasCurrentDynamicChatRuntimeReference(state, item.nodeId, item.reason, item.snippet);
  }

  if (isCurrentSimulationNode(state, item.nodeId) || isKnownRuntimeNodeId(state, item.nodeId)) {
    return true;
  }

  if (looksLikeNeuralMapFrameworkRecord(item.reason, item.snippet)) {
    return false;
  }

  return hasCurrentDynamicChatRuntimeReference(state, item.nodeId, item.reason, item.snippet);
}

function isScopedNeuralMapEvidenceItem(state: AppState, item: NeuralMapEvidenceItem): boolean {
  const metadataScope = readMetadataScope(item.metadata) ?? readMetadataScope(item.properties);

  if (!scopeMatchesDynamicChat(state, item.scope ?? metadataScope)) {
    return false;
  }

  if (
    !matchesRuntimeRecordScope(state, item.metadata, item.properties) ||
    hasForeignDynamicChatRuntimeReference(state, item.node_id, item.snippet)
  ) {
    return false;
  }

  if (requiresCurrentProgressNeuralMapEvidenceScope(item) && !hasCurrentProgressNeuralMapEvidenceScope(state, item)) {
    return false;
  }

  if (
    isCurrentSimulationNode(state, item.node_id, item.metadata) ||
    isCurrentSimulationNode(state, item.node_id, item.properties) ||
    isKnownRuntimeNodeId(state, item.node_id)
  ) {
    return true;
  }

  if (looksLikeNeuralMapFrameworkRecord(item.snippet)) {
    return false;
  }

  return hasCurrentDynamicChatRuntimeReference(state, item.node_id, item.snippet, ...readRuntimeReferenceValues(item.metadata, item.properties));
}

function isCurrentSimulationNode(state: AppState, nodeId: string, metadata?: Record<string, unknown>): boolean {
  const simulationId = state.simulation.id;
  const metadataSimulationId =
    readMetadataText(metadata, "simulation_id") ??
    readMetadataText(metadata, "simulationId") ??
    readMetadataText(metadata, "project_id");

  return (
    metadataSimulationId === simulationId ||
    nodeId === simulationId ||
    nodeId.startsWith(`simulation:${simulationId}:`) ||
    nodeId.includes(`/${simulationId}/`) ||
    nodeId.includes(`dynamicchat://${simulationId}/`)
  );
}

function isKnownRuntimeNodeId(state: AppState, nodeId: string): boolean {
  if (!nodeId) {
    return false;
  }

  if (nodeId === `persona:${state.simulation.id}`) {
    return true;
  }

  return (
    state.memoryEvents.some((event) => nodeId === event.id || nodeId === event.neuralMapNodeId) ||
    state.contextPacks.some((pack) => nodeId === pack.id) ||
    state.handoffs.some((handoff) => nodeId === handoff.id) ||
    state.characters.some((character) => nodeId === character.id || nodeId.endsWith(`:person:${character.id}`)) ||
    state.modules.some((module) => nodeId === module.id)
  );
}

function isDynamicChatSource(sourceSystem: string | undefined, metadata: Record<string, unknown> | undefined): boolean {
  const source = readMetadataText(metadata, "source");
  const kind = readMetadataText(metadata, "kind");

  return (
    sourceSystem === "dynamicchat" ||
    source === "dynamicchat" ||
    Boolean(source?.startsWith("dynamicchat_")) ||
    Boolean(kind?.startsWith("simulation_")) ||
    kind === "user_persona" ||
    kind === "context_pack"
  );
}

function matchesRuntimeRecordScope(
  state: AppState,
  metadata?: Record<string, unknown>,
  properties?: Record<string, unknown>
): boolean {
  const simulationId = readRuntimeText(["simulation_id", "simulationId", "project_id"], metadata, properties);
  if (simulationId && simulationId !== state.simulation.id) {
    return false;
  }

  const sessionId = readRuntimeText(["session_id", "sessionId", "active_session_id", "activeSessionId"], metadata, properties);
  if (sessionId && !isAllowedRuntimeSessionId(state, sessionId)) {
    return false;
  }

  const runId = readRuntimeText(["progress_run_id", "run_id", "activeProgressRunId", "runId"], metadata, properties);
  if (runId && runId !== state.activeProgressRunId) {
    return false;
  }

  return true;
}

const PROGRESS_SCOPED_NODE_ID_PATTERN = /^simulation:[^:]+:(?:memory|event|session|scene|context)(?::|$)/iu;
const PROGRESS_SCOPED_KIND_PATTERN =
  /^(?:Event|State|Observation|Belief|Scene|simulation_(?:event|state|observation|belief|goal|relationship|open_loop|summary|scene)|context_pack|context_evidence)$/u;
const PROGRESS_SCOPED_EVIDENCE_PATTERN =
  /\b(?:memory|event|state|observation|belief|open thread|open loop|scene|context pack|recent history|relevant history|current scene|canonical state|character knowledge|최근 중요 이벤트|구조화된 시뮬레이션 현재 상태|NeuralMap continuity evidence|NeuralMap profile evidence|NeuralMap section)/iu;

function requiresCurrentProgressGraphScope(node: NeuralMapLiveNode): boolean {
  return (
    PROGRESS_SCOPED_NODE_ID_PATTERN.test(node.id) ||
    Boolean(node.kind && PROGRESS_SCOPED_KIND_PATTERN.test(node.kind)) ||
    Boolean(node.ontology?.type && PROGRESS_SCOPED_KIND_PATTERN.test(node.ontology.type)) ||
    PROGRESS_SCOPED_EVIDENCE_PATTERN.test(`${node.type}\n${node.title}\n${node.summary}`)
  );
}

function requiresCurrentProgressEvidenceScope(...values: string[]): boolean {
  const text = values.filter(Boolean).join("\n");
  return PROGRESS_SCOPED_NODE_ID_PATTERN.test(text) || PROGRESS_SCOPED_EVIDENCE_PATTERN.test(text);
}

function requiresCurrentProgressNeuralMapEvidenceScope(item: NeuralMapEvidenceItem): boolean {
  const kind = readRuntimeText(["kind", "memory_kind"], item.metadata, item.properties);
  const ontologyType = readRuntimeText(["ontology_type", "type"], item.metadata, item.properties);
  return (
    requiresCurrentProgressEvidenceScope(item.node_id, item.snippet) ||
    Boolean(kind && PROGRESS_SCOPED_KIND_PATTERN.test(kind)) ||
    Boolean(ontologyType && PROGRESS_SCOPED_KIND_PATTERN.test(ontologyType))
  );
}

function hasCurrentProgressGraphScope(state: AppState, node: NeuralMapLiveNode): boolean {
  return (
    isKnownRuntimeNodeId(state, node.id) ||
    hasCurrentRuntimeScopeMarker(state, node.metadata, node.properties, node.id, node.title, node.summary)
  );
}

function hasCurrentProgressNeuralMapEvidenceScope(state: AppState, item: NeuralMapEvidenceItem): boolean {
  return (
    isKnownRuntimeNodeId(state, item.node_id) ||
    hasCurrentRuntimeScopeMarker(state, item.metadata, item.properties, item.node_id, item.snippet)
  );
}

function hasCurrentRuntimeScopeMarker(
  state: AppState,
  metadata?: Record<string, unknown>,
  properties?: Record<string, unknown>,
  ...values: string[]
): boolean {
  const runId = readRuntimeText(["progress_run_id", "run_id", "activeProgressRunId", "runId"], metadata, properties);
  if (runId) {
    return runId === state.activeProgressRunId;
  }

  const sessionId = readRuntimeText(["session_id", "sessionId", "active_session_id", "activeSessionId"], metadata, properties);
  if (sessionId) {
    return isAllowedRuntimeSessionId(state, sessionId);
  }

  return hasCurrentDynamicChatRuntimeReference(state, ...values, ...readRuntimeReferenceValues(metadata, properties));
}

function hasForeignDynamicChatRuntimeReference(state: AppState, ...values: string[]): boolean {
  const simulationIds = extractRuntimeIds(values, /\bsim_[A-Za-z0-9_-]+\b/gu);
  if (simulationIds.some((simulationId) => simulationId !== state.simulation.id)) {
    return true;
  }

  const sessionIds = extractRuntimeIds(values, /\bsession_[A-Za-z0-9_-]+\b/gu);
  if (sessionIds.some((sessionId) => !isAllowedRuntimeSessionId(state, sessionId))) {
    return true;
  }

  const runIds = extractRuntimeIds(values, /\brun_[A-Za-z0-9_-]+\b/gu);
  return runIds.some((runId) => runId !== state.activeProgressRunId);
}

function hasCurrentDynamicChatRuntimeReference(state: AppState, ...values: string[]): boolean {
  const text = values.filter(Boolean).join("\n");
  if (!text.trim()) {
    return false;
  }

  if (text.includes(state.activeProgressRunId)) {
    return true;
  }

  return [...collectAllowedRuntimeSessionIds(state)].some((sessionId) => text.includes(sessionId));
}

function readRuntimeReferenceValues(
  ...records: Array<Record<string, unknown> | undefined>
): string[] {
  return records.flatMap((record) =>
    record
      ? [
          readMetadataText(record, "simulation_id"),
          readMetadataText(record, "simulationId"),
          readMetadataText(record, "project_id"),
          readMetadataText(record, "session_id"),
          readMetadataText(record, "sessionId"),
          readMetadataText(record, "active_session_id"),
          readMetadataText(record, "activeSessionId"),
          readMetadataText(record, "progress_run_id"),
          readMetadataText(record, "run_id"),
          readMetadataText(record, "activeProgressRunId"),
          readMetadataText(record, "runId"),
          readMetadataText(record, "raw_ref"),
          readMetadataText(record, "source_ref")
        ].filter((value): value is string => Boolean(value))
      : []
  );
}

function readRuntimeText(
  keys: string[],
  ...records: Array<Record<string, unknown> | undefined>
): string | undefined {
  for (const record of records) {
    for (const key of keys) {
      const value = readMetadataText(record, key);
      if (value) {
        return value;
      }
    }
  }

  return undefined;
}

function extractRuntimeIds(values: string[], pattern: RegExp): string[] {
  return uniqueStrings(values.flatMap((value) => value.match(pattern) ?? []));
}

function isAllowedRuntimeSessionId(state: AppState, sessionId: string | undefined): boolean {
  return Boolean(sessionId && collectAllowedRuntimeSessionIds(state).has(sessionId));
}

function collectAllowedRuntimeSessionIds(state: AppState): Set<string> {
  const activeRun = state.progressRuns.find((run) => run.id === state.activeProgressRunId);
  return new Set(
    uniqueStrings([
      state.simulation.activeSessionId,
      ...(activeRun?.sessionIds ?? []),
      ...state.messages.map((message) => message.sessionId),
      ...state.memoryEvents.map((event) => event.sessionId),
      ...state.contextPacks.map((pack) => pack.sessionId),
      ...state.handoffs.flatMap((handoff) => [handoff.previousSessionId, handoff.nextSessionId]),
      ...state.continuityChecks.flatMap((check) => [check.previousSessionId, check.nextSessionId]),
      ...state.promptModuleUsages.map((usage) => usage.sessionId),
      ...state.sidecarTraces.map((trace) => trace.sessionId),
      ...state.turnTraces.map((trace) => trace.sessionId),
      ...state.imageJobs.map((job) => job.sessionId)
    ])
  );
}

function hasDynamicChatTextAffinity(state: AppState, ...values: string[]): boolean {
  const text = values.join("\n").toLowerCase();
  const signals = uniqueStrings([
    state.simulation.id,
    state.simulation.title,
    state.simulation.activeSessionId,
    ...state.characters.flatMap((character) => [character.id, character.name]),
    ...state.memoryEvents.flatMap((event) => [event.id, event.actorName ?? "", ...event.tags]),
    state.userPersona.enabled ? state.userPersona.name : ""
  ])
    .map((signal) => signal.trim().toLowerCase())
    .filter((signal) => signal.length >= 3 || /[\u3131-\uD79D]{2,}/u.test(signal));

  return signals.some((signal) => text.includes(signal));
}

const NEURALMAP_FRAMEWORK_NOISE_PATTERN =
  /\b(?:NeuralMap Repository|Track [A-Z]:|Core retrieval|seed retrieval|graph expansion|compression policy|Context Pack Composer|Graph workbench|Drizzle Graph Schema|TypeScript monorepo|AIN-\d+|linear\.app\/aineuralmap|run trace foundation)\b/iu;

function looksLikeNeuralMapFrameworkRecord(...values: string[]): boolean {
  return NEURALMAP_FRAMEWORK_NOISE_PATTERN.test(values.join("\n"));
}

function scopeMatchesDynamicChat(state: AppState, scope: PartialNeuralMapScope | undefined): boolean {
  if (!hasExplicitScope(scope)) {
    return true;
  }

  const expected = createNeuralMapScope(state);
  return (["tenant_id", "workspace_id", "project_id", "owner_scope"] as const).every((key) => {
    const value = scope?.[key]?.trim();
    return !value || value === expected[key];
  });
}

function hasExplicitScope(scope: PartialNeuralMapScope | undefined): boolean {
  return Boolean(scope && Object.values(scope).some((value) => typeof value === "string" && value.trim().length > 0));
}

function readMetadataScope(metadata: Record<string, unknown> | undefined): PartialNeuralMapScope | undefined {
  const nested = metadata?.scope;
  const nestedScope = isPlainRecord(nested) ? readScopeFields(nested) : undefined;
  const legacyScope = readScopeFields(metadata);
  const scope = {
    ...(legacyScope ?? {}),
    ...(nestedScope ?? {})
  };

  return hasExplicitScope(scope) ? scope : undefined;
}

function readScopeFields(value: Record<string, unknown> | undefined): PartialNeuralMapScope | undefined {
  if (!value) {
    return undefined;
  }

  const scope: PartialNeuralMapScope = {};
  for (const key of ["tenant_id", "workspace_id", "project_id", "owner_scope"] as const) {
    const field = value[key];
    if (typeof field === "string" && field.trim()) {
      scope[key] = field.trim();
    }
  }

  return hasExplicitScope(scope) ? scope : undefined;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPrivatePromptGraphNode(state: AppState, node: NeuralMapLiveNode): boolean {
  return (
    isPrivatePromptNodeId(state, node.id) ||
    isPrivatePromptKind(node.kind) ||
    PRIVATE_PROMPT_METADATA_SOURCES.has(readMetadataText(node.metadata, "source") ?? "") ||
    isPrivatePromptNodeId(state, readMetadataText(node.metadata, "document_id") ?? "") ||
    isPrivatePromptNodeId(state, readMetadataText(node.metadata, "module_id") ?? "") ||
    matchesPrivatePromptText(state, node.title, node.summary) ||
    looksLikePromptModuleSnippet(node.summary)
  );
}

function isPrivatePromptNodeId(state: AppState, nodeId: string): boolean {
  if (!nodeId) {
    return false;
  }

  return state.modules.some((module) =>
    nodeId === module.id ||
    nodeId === `dynamicchat://${state.simulation.id}/prompt-modules/${module.id}` ||
    nodeId.startsWith(`${module.id}:chunk:`) ||
    nodeId.includes(`/prompt-modules/${module.id}`)
  );
}

function isPrivatePromptKind(kind: string | undefined): boolean {
  return Boolean(kind && PRIVATE_PROMPT_MODULE_KINDS.has(kind));
}

function matchesPrivatePromptText(state: AppState, title: string, summary: string): boolean {
  const normalizedTitle = title.trim().toLowerCase();
  return state.modules.some((module) => {
    if (normalizedTitle && normalizedTitle === module.title.trim().toLowerCase()) {
      return true;
    }

    const moduleHead = module.body.trim().slice(0, 96);
    const summaryHead = summary.trim().slice(0, 96);
    return (
      moduleHead.length >= 48 &&
      summaryHead.length >= 48 &&
      (summary.includes(moduleHead) || module.body.includes(summaryHead))
    );
  });
}

function looksLikePromptModuleSnippet(snippet: string): boolean {
  return /^\s*(Prompt module:|Prompt module manifest:|\[excerpted long prompt module:)/iu.test(snippet);
}

function createLocalLiveGraph(state: AppState, error?: string): NeuralMapLiveGraph {
  const nodes = new Map<string, NeuralMapLiveNode>();
  const edges = new Map<string, NeuralMapLiveEdge>();
  const sessionNodeId = `simulation:${state.simulation.id}:session:${state.simulation.activeSessionId}`;
  const latestContextPack = state.contextPacks.at(-1);

  upsertLiveNode(nodes, {
    id: sessionNodeId,
    type: "Session",
    title: state.simulation.title,
    summary: `현재 세션 ${state.simulation.activeSessionId}`,
    importanceScore: 0.82,
    sourceSystem: "dynamicchat",
    kind: "simulation_session"
  });

  for (const character of state.characters.slice(0, 6)) {
    const characterNodeId = createLocalCharacterNodeId(state, character.id);
    upsertLiveNode(nodes, {
      id: characterNodeId,
      type: "Person",
      title: character.name,
      summary: character.currentMood || character.summary,
      importanceScore: 0.72,
      sourceSystem: "dynamicchat",
      kind: "simulation_person"
    });
    upsertLiveEdge(edges, {
      id: `edge:${sessionNodeId}:mentions:${characterNodeId}`,
      from: sessionNodeId,
      to: characterNodeId,
      type: "mentions",
      weight: 0.58,
      confidence: 0.8
    });
  }

  if (state.userPersona?.enabled) {
    const personaCharacter = resolvePersonaCharacter(state);
    const personaNodeId = personaCharacter ? createLocalCharacterNodeId(state, personaCharacter.id) : `persona:${state.simulation.id}`;
    if (personaCharacter) {
      upsertLiveNode(nodes, {
        id: personaNodeId,
        type: "Person",
        title: personaCharacter.name,
        summary: createUserPersonaLine(state),
        importanceScore: 0.78,
        sourceSystem: "dynamicchat",
        kind: "simulation_person",
        metadata: {
          persona_source: "character",
          controlled_by_user: true
        }
      });
    } else {
      upsertLiveNode(nodes, {
        id: personaNodeId,
        type: "Person",
        title: state.userPersona.name || "사용자 페르소나",
        summary: createUserPersonaLine(state),
        importanceScore: 0.78,
        sourceSystem: "dynamicchat",
        kind: "user_persona"
      });
    }
    upsertLiveEdge(edges, {
      id: `edge:${sessionNodeId}:controlled-by:${personaNodeId}`,
      from: sessionNodeId,
      to: personaNodeId,
      type: "controlled_by",
      weight: 0.7,
      confidence: 0.84
    });
  }

  const recentMemories = state.memoryEvents.slice(-8);
  for (const event of recentMemories) {
    const eventNodeId = event.neuralMapNodeId ?? readMetadataText(event.metadata, "graph_neuron_id") ?? event.id;
    const memoryKind = readMetadataText(event.metadata, "memory_kind");
    const stateType = readMetadataText(event.metadata, "state_type");
    upsertLiveNode(nodes, {
      id: eventNodeId,
      type: memoryKind === "state" ? "Artifact" : memoryKind === "summary" ? "Summary" : "Task",
      title: stateType ? `${stateType} state` : event.tags[0] ? `${event.tags[0]} memory` : "Simulation event",
      summary: event.content,
      importanceScore: clampScore(event.importance),
      sourceSystem: event.neuralMapNodeId ? "neuralmap" : "dynamicchat",
      kind: memoryKind ? `simulation_${memoryKind}` : "simulation_event",
      metadata: event.metadata
    });
    upsertLiveEdge(edges, {
      id: `edge:${sessionNodeId}:references:${eventNodeId}`,
      from: sessionNodeId,
      to: eventNodeId,
      type: "references",
      weight: 0.82,
      confidence: 0.9
    });

    if (event.actorId) {
      const actorNodeId = createLocalCharacterNodeId(state, event.actorId);
      upsertLiveEdge(edges, {
        id: `edge:${eventNodeId}:mentions:${actorNodeId}`,
        from: eventNodeId,
        to: actorNodeId,
        type: "mentions",
        weight: 0.8,
        confidence: 0.86
      });
    }
  }

  if (latestContextPack) {
    upsertLiveNode(nodes, {
      id: latestContextPack.id,
      type: "Artifact",
      title: latestContextPack.source === "neuralmap" ? "NeuralMap Context Pack" : "Local Context Pack",
      summary: latestContextPack.objective,
      importanceScore: 0.88,
      sourceSystem: latestContextPack.source,
      kind: "context_pack"
    });
    upsertLiveEdge(edges, {
      id: `edge:${sessionNodeId}:references:${latestContextPack.id}`,
      from: sessionNodeId,
      to: latestContextPack.id,
      type: "references",
      weight: 0.72,
      confidence: 0.86
    });

    for (const evidence of filterPublicContextEvidence(state, filterScopedContextEvidence(state, latestContextPack.evidence)).slice(0, 8)) {
      upsertLiveNode(nodes, {
        id: evidence.nodeId,
        type: "Document",
        title: evidence.reason,
        summary: evidence.snippet,
        importanceScore: clampScore(evidence.score),
        sourceSystem: latestContextPack.source,
        kind: "context_evidence"
      });
      upsertLiveEdge(edges, {
        id: `edge:${latestContextPack.id}:references:${evidence.nodeId}`,
        from: latestContextPack.id,
        to: evidence.nodeId,
        type: "references",
        weight: evidence.score,
        confidence: Math.max(0.5, evidence.score)
      });
    }
  }

  return {
    source: "local",
    nodes: [...nodes.values()],
    edges: [...edges.values()].filter((edge) => nodes.has(edge.from) && nodes.has(edge.to)),
    seedNodeIds: recentMemories.map((event) => event.neuralMapNodeId ?? event.id).slice(-3),
    generatedAt: new Date().toISOString(),
    error
  };
}

function createLocalCharacterNodeId(state: AppState, characterId: string): string {
  return `simulation:${state.simulation.id}:person:${characterId}`;
}

function upsertLiveNode(nodes: Map<string, NeuralMapLiveNode>, node: NeuralMapLiveNode): void {
  const existing = nodes.get(node.id);
  nodes.set(node.id, existing ? { ...existing, ...node, importanceScore: Math.max(existing.importanceScore, node.importanceScore) } : node);
}

function upsertLiveEdge(edges: Map<string, NeuralMapLiveEdge>, edge: NeuralMapLiveEdge): void {
  edges.set(edge.id, edge);
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function readMetadataText(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" ? value : undefined;
}

function clampScore(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0.5;
}

function normalizeHandoffResponse(response: NeuralMapHandoffResponse): NeuralMapHandoffPack {
  return response.pack ?? response;
}

function readHandoffEvidenceNodeIds(pack: NeuralMapHandoffPack): string[] {
  return [
    ...(pack.referenced_node_ids ?? []),
    ...(pack.evidence?.map((item) => item.node_id ?? item.id).filter((item): item is string => Boolean(item)) ?? [])
  ];
}

function createHandoffQuery(state: AppState): string {
  const characters = state.characters
    .map((character) => `${character.name}: ${character.relationship}, ${character.currentMood}`)
    .join(" / ");
  const recentEvents = state.memoryEvents
    .slice(-5)
    .map((event) => event.content)
    .join(" / ");

  return [
    "세션 초기화를 위해 다음 LLM 세션에 이어져야 할 핵심 상태를 정리한다.",
    `캐릭터 상태: ${characters || "없음"}`,
    `최근 사건: ${recentEvents || "없음"}`
  ].join("\n");
}

function createLocalSessionHandoff(state: AppState, nextSessionId: string, handoffEvent: MemoryEvent): SessionHandoff {
  return {
    id: createId("handoff"),
    simulationId: state.simulation.id,
    previousSessionId: state.simulation.activeSessionId,
    nextSessionId,
    summary: handoffEvent.content,
    evidenceNodeIds: [handoffEvent.neuralMapNodeId ?? handoffEvent.id],
    createdAt: new Date().toISOString(),
    source: "mock"
  };
}

export function createLocalContextPack(state: AppState, query: string): ContextPack {
  void query;
  const evidence = createLocalContextEvidence(state).sort((a, b) => b.score - a.score).slice(0, 12);

  return {
    id: createId("ctx"),
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    objective: `Continue ${state.simulation.title}`,
    tokenBudget: state.neuralMap.tokenBudget,
    evidence,
    moduleEvidence: evidence,
    decisions: ["사용자의 선택을 우선하고, 장면 상태와 관계 변화를 유지한다."],
    blockers: [],
    createdAt: new Date().toISOString(),
    source: "mock"
  };
}

function createLocalContextEvidence(state: AppState): ContextPack["evidence"] {
  const immediateContinuityEvidence = createImmediateContinuityEvidence(state);
  const personaLine = createUserPersonaLine(state);
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state));
  const personaEvidence = personaLine
    ? [
        {
          nodeId: `persona:${state.simulation.id}`,
          snippet: personaLine,
          score: 0.9,
          reason: "사용자 페르소나"
        }
      ]
    : [];
  const structuredEvidence = createStructuredContextEvidence(state);
  const memoryEvidence = state.memoryEvents
    .filter((event) => memoryEventIsRelevantToCurrentSceneCast(state, event, activeCharacterIds))
    .slice(-5)
    .map((event) => ({
      nodeId: event.neuralMapNodeId ?? event.id,
      snippet: event.content,
      score: event.importance,
      reason: "최근 중요 이벤트"
    }));

  return uniqueContextEvidence([
    ...(immediateContinuityEvidence ? [immediateContinuityEvidence] : []),
    ...(structuredEvidence ? [structuredEvidence] : []),
    ...createLocalFoundationEvidence(state),
    ...personaEvidence,
    ...memoryEvidence
  ]);
}

function memoryEventIsRelevantToCurrentSceneCast(
  state: AppState,
  event: MemoryEvent,
  activeCharacterIds: ReadonlySet<string>
): boolean {
  const characterIds = getMemoryEventCharacterIds(state, event);
  if (characterIds.length === 0) {
    return true;
  }

  if (state.characters.length === 1 && characterIds.includes(state.characters[0].id)) {
    return true;
  }

  return characterIds.some((characterId) => activeCharacterIds.has(characterId));
}

function getMemoryEventCharacterIds(state: AppState, event: MemoryEvent): string[] {
  const knownCharacterIds = new Set(state.characters.map((character) => character.id));
  return uniqueStrings([
    event.actorId,
    readMetadataText(event.metadata, "owner_id"),
    readMetadataText(event.metadata, "target_id")
  ].filter((characterId): characterId is string => Boolean(characterId))).filter((characterId) => knownCharacterIds.has(characterId));
}

function createImmediateContinuityEvidence(state: AppState): ContextPack["evidence"][number] | undefined {
  const latestAssistant = findLatestMessageByRole(state, "assistant");
  const latestUser = findLatestMessageByRole(state, "user");
  if (!latestAssistant && !latestUser) {
    return undefined;
  }

  const snippet = [
    "# Immediate Continuity Anchor",
    "Use this before older retrieved memories. Continue from the latest assistant ending and the current user action.",
    latestUser
      ? `Current/last user action: ${createContinuityExcerpt(latestUser.content, LOCAL_CONTINUITY_USER_CHARS, "balanced")}`
      : undefined,
    latestAssistant
      ? `Latest assistant ending: ${createContinuityExcerpt(latestAssistant.content, LOCAL_CONTINUITY_ASSISTANT_CHARS, "tail")}`
      : undefined
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n");

  return {
    nodeId: `simulation:${state.simulation.id}:continuity-anchor:${state.simulation.activeSessionId}`,
    snippet,
    score: 0.99,
    reason: "직전 출력/현재 입력 연속성"
  };
}

function createLocalFoundationEvidence(state: AppState): ContextPack["evidence"] {
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state));
  const characterEvidence = state.characters.filter((character) => activeCharacterIds.has(character.id)).slice(0, 6).map((character) => ({
    nodeId: `simulation:${state.simulation.id}:character:${character.id}`,
    snippet: [
      `${character.name}: ${character.role}`,
      character.summary,
      character.relationship,
      character.currentMood
    ]
      .filter(Boolean)
      .join(" / "),
    score: 0.92,
    reason: "현재 장면에 언급된 캐릭터 설정"
  }));
  const worldEvidence = state.modules
    .filter((module) => module.enabled && module.tokenPolicy !== "disabled" && module.kind === "world_lore")
    .sort((a, b) => b.priority - a.priority)
    .slice(0, 3)
    .map((module) => ({
      nodeId: `simulation:${state.simulation.id}:module:${module.id}`,
      snippet: `${module.title}: ${module.body.slice(0, 420)}`,
      score: 0.86,
      reason: "시뮬레이션 세계관 기반 설정"
    }));
  const simulationEvidence = {
    nodeId: `simulation:${state.simulation.id}:foundation`,
    snippet: `${state.simulation.title}: ${state.simulation.description}`,
    score: 0.88,
    reason: "시뮬레이션 기본 설정"
  };

  return [simulationEvidence, ...characterEvidence, ...worldEvidence];
}

function uniqueContextEvidence(evidence: ContextPack["evidence"]): ContextPack["evidence"] {
  const seen = new Set<string>();
  return evidence.filter((item) => {
    const key = `${item.nodeId}:${item.snippet.slice(0, 120)}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function createUserPersonaLine(state: AppState): string {
  const persona = state.userPersona;
  if (!persona?.enabled) {
    return "";
  }

  const character = resolvePersonaCharacter(state);
  if (persona.source === "character" && character) {
    return [
      `시점 ${character.name}`,
      `캐릭터ID ${character.id}`,
      character.role.trim() ? `역할 ${character.role.trim()}` : undefined,
      character.summary.trim() ? `배경 ${character.summary.trim()}` : undefined,
      character.relationship.trim() ? `관계 ${character.relationship.trim()}` : undefined,
      character.currentMood.trim() ? `현재 ${character.currentMood.trim()}` : undefined,
      persona.goals.trim() ? `플레이목표 ${persona.goals.trim()}` : undefined,
      persona.style.trim() ? `입력방식 ${persona.style.trim()}` : undefined,
      persona.boundaries.trim() ? `경계 ${persona.boundaries.trim()}` : undefined
    ]
      .filter((item): item is string => Boolean(item))
      .join(" / ");
  }

  return [
    persona.name.trim() ? `이름 ${persona.name.trim()}` : undefined,
    persona.role.trim() ? `역할 ${persona.role.trim()}` : undefined,
    persona.background.trim() ? `배경 ${persona.background.trim()}` : undefined,
    persona.goals.trim() ? `목표 ${persona.goals.trim()}` : undefined,
    persona.style.trim() ? `성향 ${persona.style.trim()}` : undefined,
    persona.boundaries.trim() ? `경계 ${persona.boundaries.trim()}` : undefined
  ]
    .filter((item): item is string => Boolean(item))
    .join(" / ");
}

function findLatestMessageByRole(
  state: AppState,
  role: AppState["messages"][number]["role"]
): AppState["messages"][number] | undefined {
  return [...state.messages].reverse().find((message) => message.role === role);
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

function resolvePersonaCharacter(state: AppState): AppState["characters"][number] | undefined {
  const persona = state.userPersona;
  if (!persona?.enabled || persona.source !== "character" || !persona.characterId) {
    return undefined;
  }

  return state.characters.find((character) => character.id === persona.characterId);
}
