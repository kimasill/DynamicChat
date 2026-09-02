import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const vite = await createServer({
  root: workspaceRoot,
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom"
});

const checks = [];

try {
  const { seedState, hydrateState } = await vite.ssrLoadModule("/src/data/seed.ts");
  const { activateSimulationProgressRun, createFreshSimulationRun } = await vite.ssrLoadModule("/src/services/simulationRuns.ts");
  const { NeuralMapClient } = await vite.ssrLoadModule("/src/services/neuralMapClient.ts");
  const { createLocalContextPack } = await vite.ssrLoadModule("/src/services/neuralMapClient.ts");
  const { generateAssistantText } = await vite.ssrLoadModule("/src/services/llmClient.ts");
  const { runSimulationTurn } = await vite.ssrLoadModule("/src/services/simulationEngine.ts");
  const { createStructuredContextSummary } = await vite.ssrLoadModule("/src/services/memoryCompiler.ts");
  const { inferCurrentSceneCharacterIds } = await vite.ssrLoadModule("/src/services/sceneCast.ts");
  evaluateFreshSimulationRun(seedState, createFreshSimulationRun, activateSimulationProgressRun);
  await evaluateNeuralMapScopeFiltering(seedState, NeuralMapClient);
  await evaluateNeuralMapGraphDeltaRoles(seedState, NeuralMapClient);
  await evaluateNeuralMapCastFiltering(seedState, NeuralMapClient, runSimulationTurn);
  await evaluateContinuityAnchors(seedState, createLocalContextPack, generateAssistantText);
  await evaluateSceneCastGuard(seedState, createLocalContextPack, generateAssistantText, createStructuredContextSummary);
  evaluateImageCastContinuity(seedState, inferCurrentSceneCharacterIds);
  await evaluateUnconfiguredTurnHonesty(seedState, hydrateState, runSimulationTurn);
} finally {
  await vite.close();
}

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}`);
}

const failures = checks.filter((check) => check.level === "fail");
console.log("");
console.log(`Simulation run quality checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

function evaluateFreshSimulationRun(seedState, createFreshSimulationRun, activateSimulationProgressRun) {
  const source = {
    ...structuredClone(seedState),
    messages: [
      ...seedState.messages.map((message, index) =>
        index === 0
          ? {
              ...message,
              referencedNodeIds: [
                ...message.referencedNodeIds,
                "memory_eval_old",
                "simulation:sim_scope_current:event:memory_eval_old"
              ]
            }
          : message
      ),
      createUserMessage(seedState, "첫 진행에서 이미 선택한 행동"),
      createAssistantMessage(seedState, "첫 진행의 결과가 이어지고 있다.")
    ],
    memoryEvents: [
      {
        id: "memory_eval_old",
        simulationId: seedState.simulation.id,
        sessionId: seedState.simulation.activeSessionId,
        content: "이전 진행에서만 존재해야 하는 기억",
        importance: 0.8,
        tags: ["old-run"],
        createdAt: new Date().toISOString()
      }
    ],
    contextPacks: [
      {
        id: "ctx_eval_old",
        simulationId: seedState.simulation.id,
        sessionId: seedState.simulation.activeSessionId,
        objective: "old run context",
        tokenBudget: 1200,
        evidence: [],
        decisions: [],
        blockers: [],
        createdAt: new Date().toISOString(),
        source: "mock"
      }
    ],
    imageJobs: [
      {
        id: "imgjob_eval_old",
        simulationId: seedState.simulation.id,
        sessionId: seedState.simulation.activeSessionId,
        turnId: "msg_eval_user",
        status: "completed",
        reason: "old run image",
        prompt: "old prompt",
        negativePrompt: "",
        providerPayload: {},
        assetIds: [],
        contextNodeIds: [],
        createdAt: new Date().toISOString()
      }
    ]
  };

  const fresh = createFreshSimulationRun(source, [source]);
  const secondFresh = createFreshSimulationRun(fresh, [fresh]);
  const previousRun = fresh.progressRuns.find((run) => run.id !== fresh.activeProgressRunId);
  const activeRun = fresh.progressRuns.find((run) => run.id === fresh.activeProgressRunId);
  const restoredPreviousRun = previousRun ? activateSimulationProgressRun(fresh, previousRun.id) : undefined;

  assertCheck("run.fresh", fresh.simulation.id === source.simulation.id, "Fresh run stays inside the same simulation id.");
  assertCheck("run.fresh", fresh.simulation.title === source.simulation.title, "Fresh run does not alter the simulation title.");
  assertCheck("run.fresh", fresh.simulation.activeSessionId !== source.simulation.activeSessionId, "Fresh run receives a new session id.");
  assertCheck("run.fresh", fresh.activeProgressRunId !== source.activeProgressRunId, "Fresh run receives a new progress run id.");
  assertCheck("run.fresh", /새 진행 2$/u.test(activeRun?.title ?? ""), "Fresh run is labeled inside progress history.");
  assertCheck("run.fresh", /새 진행 3$/u.test(secondFresh.progressRuns.find((run) => run.id === secondFresh.activeProgressRunId)?.title ?? ""), "Additional fresh runs increment progress history labels.");
  assertCheck("run.fresh", fresh.progressRuns.length >= 2, "Fresh run preserves previous and active progress entries.");
  assertCheck("run.fresh", fresh.messages.length === 1 && fresh.messages[0]?.role === "assistant", "Fresh run starts from one opening assistant message.");
  assertCheck("run.fresh", fresh.messages.every((message) => message.sessionId === fresh.simulation.activeSessionId), "Fresh run messages target the new session.");
  assertCheck("run.fresh", !(fresh.messages[0]?.referencedNodeIds ?? []).includes("memory_eval_old"), "Fresh run opening does not inherit prior run memory references.");
  assertCheck("run.fresh", fresh.memoryEvents.length === 0, "Fresh run does not inherit previous run memories.");
  assertCheck("run.fresh", fresh.contextPacks.length === 0, "Fresh run does not inherit previous Context Packs.");
  assertCheck("run.fresh", fresh.imageJobs.length === 0, "Fresh run does not inherit previous image jobs.");
  assertCheck("run.fresh", fresh.turnTraces.length === 0 && fresh.sidecarTraces.length === 0, "Fresh run starts without turn traces.");
  assertCheck("run.fresh", fresh.modules.every((module) => module.simulationId === source.simulation.id), "Fresh run keeps the simulation prompt modules.");
  assertCheck("run.fresh", fresh.characters.every((character) => character.simulationId === source.simulation.id), "Fresh run keeps the simulation characters.");
  assertCheck("run.fresh", fresh.visualProfiles.every((profile) => profile.simulationId === source.simulation.id), "Fresh run keeps visual profiles on the same simulation.");
  assertCheck("run.fresh", fresh.imageAssets.every((asset) => asset.source === "stored"), "Fresh run carries only stored/reference assets.");
  assertCheck("run.fresh", previousRun?.messages.some((message) => message.content === "첫 진행에서 이미 선택한 행동"), "Previous progress stores old chat messages.");
  assertCheck("run.fresh", previousRun?.memoryEvents.some((event) => event.tags.includes("old-run")), "Previous progress stores old memories.");
  assertCheck("run.fresh", restoredPreviousRun?.activeProgressRunId === previousRun?.id, "A previous progress entry can be reopened.");
  assertCheck("run.fresh", restoredPreviousRun?.simulation.id === source.simulation.id, "Reopening a progress entry keeps the same simulation id.");
  assertCheck("run.fresh", fresh.auditLog.some((event) => event.metadata?.operation === "start_new_progress_run"), "Fresh run records a progress-run start audit event.");
}

async function evaluateNeuralMapScopeFiltering(seedState, NeuralMapClient) {
  const currentSimulationId = "sim_scope_current";
  const currentSessionId = "session_scope_current";
  const currentRunId = "run_scope_current";
  const previousRunId = "run_scope_previous";
  const previousSessionId = "session_scope_previous";
  const foreignSimulationId = "sim_scope_foreign";
  const foreignSessionId = "session_scope_foreign";
  const state = {
    ...structuredClone(seedState),
    simulation: {
      ...seedState.simulation,
      id: currentSimulationId,
      title: "Scope Current",
      activeSessionId: currentSessionId
    },
    security: {
      ...seedState.security,
      scope: {
        ownerId: "owner_scope",
        workspaceId: "workspace_scope",
        projectId: currentSimulationId,
        environment: "local"
      }
    },
    activeProgressRunId: currentRunId,
    progressRuns: [
      {
        id: currentRunId,
        simulationId: currentSimulationId,
        title: "진행 1",
        activeSessionId: currentSessionId,
        sessionIds: [currentSessionId],
        messages: [],
        memoryEvents: [],
        contextPacks: [],
        handoffs: [],
        continuityChecks: [],
        promptModuleUsages: [],
        sidecarTraces: [],
        turnTraces: [],
        imageAssets: [],
        imageJobs: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
    ],
    characters: [
      {
        id: "char_scope_current",
        simulationId: currentSimulationId,
        name: "SharedName",
        role: "주요 인물",
        summary: "현재 범위 캐릭터",
        relationship: "현재 진행 관계",
        currentMood: "차분함"
      }
    ],
    messages: [],
    memoryEvents: [],
    contextPacks: [],
    handoffs: [],
    continuityChecks: [],
    promptModuleUsages: [],
    sidecarTraces: [],
    turnTraces: [],
    imageJobs: [],
    neuralMap: {
      enabled: true,
      baseUrl: "https://neuralmap.invalid",
      tokenBudget: 1800
    }
  };
  const response = {
    id: "ctx_scope_remote",
    objective: "Continue Scope Current",
    session_id: foreignSessionId,
    evidence: [
      {
        node_id: `simulation:${foreignSimulationId}:memory:old`,
        snippet: `foreign memory from ${foreignSimulationId}`,
        score: 0.99
      },
      {
        node_id: "uuid-old-shared-character",
        snippet: "SharedName remembered an old run-only clue",
        score: 0.98
      },
      {
        node_id: `simulation:${currentSimulationId}:memory:previous-run`,
        snippet: `same simulation previous progress memory from ${previousRunId}`,
        score: 0.97,
        metadata: {
          simulation_id: currentSimulationId,
          session_id: previousSessionId,
          progress_run_id: previousRunId
        }
      },
      {
        node_id: `simulation:${currentSimulationId}:memory:current`,
        snippet: `current scoped memory for ${currentSimulationId}`,
        score: 0.96,
        metadata: {
          simulation_id: currentSimulationId,
          session_id: currentSessionId,
          progress_run_id: currentRunId
        }
      }
    ],
    sections: {
      current_scene: [
        {
          node_id: `simulation:${foreignSimulationId}:session:${foreignSessionId}`,
          snippet: `Persistent simulation session for ${foreignSimulationId}.`,
          score: 0.99
        },
        {
          node_id: `simulation:${currentSimulationId}:session:${currentSessionId}`,
          snippet: `Persistent simulation session for ${currentSimulationId}.`,
          score: 0.94,
          metadata: {
            simulation_id: currentSimulationId,
            session_id: currentSessionId,
            progress_run_id: currentRunId
          }
        }
      ],
      relevant_history: [
        {
          node_id: `simulation:${foreignSimulationId}:memory:old`,
          snippet: `Old run-only memory from ${foreignSessionId}`,
          score: 0.98
        },
        {
          node_id: `simulation:${currentSimulationId}:memory:current`,
          snippet: `Current scoped memory from ${currentSessionId}`,
          score: 0.96,
          metadata: {
            simulation_id: currentSimulationId,
            session_id: currentSessionId,
            progress_run_id: currentRunId
          }
        }
      ]
    },
    decisions: [],
    blockers: [],
    token_budget: 1800,
    created_at: new Date().toISOString()
  };
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => response
  });

  try {
    const pack = await new NeuralMapClient(state.neuralMap).getSimulationContext(state, "scope check");
    const combinedEvidence = [
      ...pack.evidence,
      ...(pack.moduleEvidence ?? []),
      ...Object.values(pack.sections ?? {}).flat()
    ]
      .map((item) => `${item.nodeId}\n${item.snippet}`)
      .join("\n");

    assertCheck("neuralmap.scope", pack.sessionId === currentSessionId, "Foreign Context Pack session id is normalized to the active session.");
    assertCheck("neuralmap.scope", !combinedEvidence.includes(foreignSimulationId), "Foreign simulation evidence is removed before section summaries are built.");
    assertCheck("neuralmap.scope", !combinedEvidence.includes(foreignSessionId), "Foreign session evidence is removed from Context Pack sections.");
    assertCheck("neuralmap.scope", !combinedEvidence.includes("old run-only clue"), "Ambiguous same-character remote evidence is not accepted without current scope markers.");
    assertCheck("neuralmap.scope", !combinedEvidence.includes(previousRunId), "Same-simulation evidence from another progress run is removed.");
    assertCheck("neuralmap.scope", combinedEvidence.includes(currentSimulationId), "Current simulation NeuralMap evidence is retained.");
  } finally {
    globalThis.fetch = previousFetch;
  }
}

async function evaluateNeuralMapGraphDeltaRoles(seedState, NeuralMapClient) {
  const simulationId = "sim_graph_roles";
  const sessionId = "session_graph_roles";
  const runId = "run_graph_roles";
  const now = new Date().toISOString();
  const state = {
    ...structuredClone(seedState),
    simulation: {
      ...seedState.simulation,
      id: simulationId,
      title: "Graph Role Eval",
      activeSessionId: sessionId
    },
    security: {
      ...seedState.security,
      scope: {
        ownerId: "owner_graph",
        workspaceId: "workspace_graph",
        projectId: simulationId,
        environment: "local"
      }
    },
    activeProgressRunId: runId,
    characters: [
      createEvalCharacter(simulationId, "char_mina", "미나", "관계 변화의 actor"),
      createEvalCharacter(simulationId, "char_sora", "소라", "관계 변화의 target")
    ],
    messages: [],
    memoryEvents: [],
    contextPacks: [],
    neuralMap: {
      enabled: true,
      baseUrl: "https://neuralmap.invalid",
      tokenBudget: 1800
    }
  };
  const delta = {
    id: "memdelta_graph_roles",
    turnId: "turn_graph_roles",
    simTime: now,
    sceneId: "scene:graph-role-room",
    warnings: [],
    upsertRecords: [
      createDeltaRecord("rec_event", "event", {
        content: "미나가 소라에게 오래된 약속을 확인했다.",
        actorId: "char_mina",
        actorName: "미나",
        targetId: "char_sora",
        eventType: "PromiseChecked",
        tags: ["promise"]
      }),
      createDeltaRecord("rec_relationship", "relationship", {
        content: "미나와 소라의 신뢰가 조금 회복되었다.",
        actorId: "char_mina",
        actorName: "미나",
        targetId: "char_sora",
        eventType: "RelationshipUpdated",
        tags: ["relationship", "trust"]
      }),
      createDeltaRecord("rec_state", "state", {
        content: "미나 Emotion: relieved",
        actorId: "char_mina",
        actorName: "미나",
        ownerId: "char_mina",
        stateType: "Emotion",
        value: "relieved",
        tags: ["state", "Emotion"]
      }),
      createDeltaRecord("rec_observation", "observation", {
        content: "미나는 소라가 약속을 숨기지 않는다고 관찰했다.",
        actorId: "char_mina",
        actorName: "미나",
        targetId: "char_sora",
        observers: ["char_mina"],
        eventType: "Observed",
        tags: ["observation"]
      })
    ]
  };
  const requests = [];
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    requests.push({
      url: String(url),
      body: JSON.parse(String(init?.body ?? "{}"))
    });
    return {
      ok: true,
      json: async () => ({ accepted: true })
    };
  };

  try {
    await new NeuralMapClient(state.neuralMap).applyMemoryDelta(delta, state);
    const request = requests.find((item) => item.url.includes("/graph/deltas"))?.body;
    const synapses = request?.upsert_synapses ?? [];
    const neurons = request?.upsert_neurons ?? [];
    // Graph node ids are RUN-SCOPED: createRunScopedSimNamespace (neuralMapClient.ts) builds
    // `simulation:<sim>:run:<runSlug>:…` so two progress runs of the same simulation cannot collide in
    // NeuralMap. These assertions were written before that scoping existed and kept the old
    // `simulation:<sim>:…` shape, so all five id-matching checks had been failing against a graph that was
    // in fact correct — the edge-type and edge-count checks beside them, which do not look at ids, passed
    // throughout. Built from the same two ids the state above is given, so the expectation moves with it.
    const runScopedNamespace = `simulation:${simulationId}:run:${runId}`;
    const sceneNodeId = `${runScopedNamespace}:scene:graph-role-room`;
    const minaNodeId = `${runScopedNamespace}:person:char_mina`;
    const soraNodeId = `${runScopedNamespace}:person:char_sora`;
    const edgeTypes = new Set(synapses.map((edge) => edge.type));

    assertCheck("neuralmap.graph", Boolean(request), "Graph delta request is sent to NeuralMap.");
    assertCheck("neuralmap.graph", neurons.some((node) => node.id === sceneNodeId && node.ontology?.type === "Scene"), "Graph delta includes a scene neuron.");
    assertCheck("neuralmap.graph", neurons.some((node) => node.id === minaNodeId) && neurons.some((node) => node.id === soraNodeId), "Graph delta includes actor and target character neurons.");
    assertCheck("neuralmap.graph", synapses.filter((edge) => edge.type === "SCENE_HAS_MEMORY").length === delta.upsertRecords.length, "Every memory record is linked to its scene.");
    assertCheck("neuralmap.graph", synapses.some((edge) => edge.type === "SCENE_PARTICIPANT" && edge.from === sceneNodeId && edge.to === minaNodeId), "Scene participant edge records the active actor/observer.");
    assertCheck("neuralmap.graph", synapses.some((edge) => edge.type === "TARGET_OF" && edge.from === soraNodeId), "Target character is linked to targeted memories.");
    assertCheck("neuralmap.graph", synapses.some((edge) => edge.type === "RELATIONSHIP_TO" && edge.from === minaNodeId && edge.to === soraNodeId), "Relationship deltas connect actor to target.");
    assertCheck("neuralmap.graph", edgeTypes.has("HAS_CURRENT_STATE") && edgeTypes.has("OBSERVED") && edgeTypes.has("ACTOR_OF"), "State, observation, and event role edges are preserved.");
  } finally {
    globalThis.fetch = previousFetch;
  }
}

/**
 * An unconfigured backend must fail the turn, never invent one.
 *
 * The seed used to ship `llm: { enabled: false, provider: "mock" }`, and requestAssistantTurn returned the
 * same locally-assembled placeholder for "mock" and for "not set up yet". The caller committed that to the
 * transcript as an ordinary assistant message, so a brand-new user's first turn was fabricated prose stored
 * in their simulation, warned about only by a toast that cleared itself. "Mock" stays a deliberate offline
 * demo; "not configured" now stops the turn and says what to fix.
 */
async function evaluateUnconfiguredTurnHonesty(seedState, hydrateState, runSimulationTurn) {
  assertCheck(
    "turn.honesty",
    seedState.llm.provider !== "mock" && seedState.llm.enabled === true,
    "A fresh install defaults to a real backend, not to the offline mock provider."
  );

  const attempt = async (llm) => {
    const state = hydrateState({
      ...seedState,
      llm: { ...seedState.llm, ...llm },
      memoryEvents: [],
      imageJobs: [],
      imageAssets: [],
      neuralMap: { ...seedState.neuralMap, enabled: false }
    });
    try {
      const turn = await runSimulationTurn(state, "창가로 걸어간다", false);
      return { wrote: true, text: turn?.assistantMessage?.content ?? "" };
    } catch (error) {
      return { wrote: false, message: String(error?.message ?? "") };
    }
  };

  const disabled = await attempt({ enabled: false, provider: "ollama" });
  assertCheck("turn.honesty", !disabled.wrote, "A disabled LLM fails the turn instead of writing a fabricated one.");
  assertCheck("turn.honesty", /설정/u.test(disabled.message ?? ""), "The failure says where to fix it.");

  const noKey = await attempt({ enabled: true, provider: "codex", apiKey: "", model: "gpt-4.1-mini" });
  assertCheck("turn.honesty", !noKey.wrote, "A hosted provider with no API key fails the turn instead of writing a fabricated one.");
  assertCheck("turn.honesty", /Ollama/u.test(noKey.message ?? ""), "The missing-key failure names the keyless local alternative.");

  const mock = await attempt({ enabled: false, provider: "mock", model: "mock-simulation-agent" });
  assertCheck("turn.honesty", mock.wrote, "The mock provider remains a working offline demo when deliberately selected.");
}

async function evaluateNeuralMapCastFiltering(seedState, NeuralMapClient, runSimulationTurn) {
  const simulationId = "sim_cast_filter";
  const sessionId = "session_cast_filter";
  const runId = "run_cast_filter";
  const now = new Date().toISOString();
  const characters = [
    createEvalCharacter(simulationId, "char_mina", "미나", "현재 복도 장면에서 사용자와 함께 있는 인물"),
    createEvalCharacter(simulationId, "char_sora", "소라", "관계도에는 있으나 현재 장면 밖에 있는 인물"),
    createEvalCharacter(simulationId, "char_jun", "준", "NeuralMap에는 있으나 현재 장면 밖에 있는 인물")
  ];
  const state = {
    ...structuredClone(seedState),
    simulation: {
      ...seedState.simulation,
      id: simulationId,
      title: "Cast Filtering Eval",
      description: "현재 장면은 복도에서 한 명과 이어진다.",
      activeSessionId: sessionId
    },
    security: {
      ...seedState.security,
      scope: {
        ownerId: "owner_cast",
        workspaceId: "workspace_cast",
        projectId: simulationId,
        environment: "local"
      }
    },
    activeProgressRunId: runId,
    progressRuns: [],
    llm: {
      ...seedState.llm,
      enabled: false,
      provider: "mock",
      apiKey: ""
    },
    neuralMap: {
      enabled: true,
      baseUrl: "https://neuralmap.invalid",
      tokenBudget: 1800
    },
    relationshipMap: {
      ...seedState.relationshipMap,
      enabled: true
    },
    characters,
    visualProfiles: characters.map((character) => ({
      id: `visual_${character.id}`,
      simulationId,
      characterId: character.id,
      displayName: character.name,
      positivePrompt: `${character.name} visual profile marker`,
      negativePrompt: "",
      defaultOutfitPrompt: `${character.name} outfit marker`,
      outfitPrompts: {},
      expressionPrompts: {},
      defaultSafetyLevel: "safe",
      updatedAt: now
    })),
    modules: [
      {
        id: "module_cast_main",
        simulationId,
        kind: "main_prompt",
        title: "메인 규칙",
        body: "현재 장면의 실제 등장 인물만 움직인다.",
        enabled: true,
        priority: 100,
        activationTags: ["core"],
        tokenPolicy: "always",
        version: 1,
        updatedAt: now
      },
      createEvalCharacterModule(simulationId, "module_cast_mina", "char_mina", "미나", "rag"),
      createEvalCharacterModule(simulationId, "module_cast_sora", "char_sora", "소라", "rag"),
      createEvalCharacterModule(simulationId, "module_cast_jun", "char_jun", "준", "always")
    ],
    messages: [
      createUserMessage(seedState, "미나에게 복도 끝을 확인해 달라고 한다."),
      createAssistantMessage(seedState, "미나는 복도 끝을 보고 돌아와, 아직 아무도 오지 않았다고 말한다.")
    ].map((message) => ({
      ...message,
      simulationId,
      sessionId
    })),
    memoryEvents: [
      createStateMemoryEvent({ simulationId, sessionId, runId, characterId: "char_mina", characterName: "미나", value: "quiet, standing" }),
      createStateMemoryEvent({ simulationId, sessionId, runId, characterId: "char_sora", characterName: "소라", value: "offstage, waiting" }),
      createStateMemoryEvent({ simulationId, sessionId, runId, characterId: "char_jun", characterName: "준", value: "offstage, waiting" })
    ],
    contextPacks: [],
    handoffs: [],
    continuityChecks: [],
    promptModuleUsages: [],
    sidecarTraces: [],
    turnTraces: [],
    imageJobs: []
  };
  const neuralMapResponse = {
    id: "ctx_cast_remote",
    objective: "Continue Cast Filtering Eval",
    session_id: sessionId,
    evidence: [
      {
        node_id: `simulation:${simulationId}:memory:mina`,
        snippet: "미나는 복도 문 앞에서 사용자의 다음 행동을 기다린다.",
        score: 0.98,
        metadata: { simulation_id: simulationId, session_id: sessionId, progress_run_id: runId }
      },
      {
        node_id: `simulation:${simulationId}:person:char_sora`,
        snippet: "소라는 관계도와 NeuralMap 상태에는 있지만 현재 복도 장면 밖에 있다.",
        score: 0.97,
        metadata: { simulation_id: simulationId, session_id: sessionId, progress_run_id: runId }
      },
      {
        node_id: `simulation:${simulationId}:person:char_jun`,
        snippet: "준은 항상 캐릭터 프롬프트가 있는 등록 인물이지만 현재 장면 밖에 있다.",
        score: 0.96,
        metadata: { simulation_id: simulationId, session_id: sessionId, progress_run_id: runId }
      }
    ],
    sections: {
      current_scene: [
        {
          node_id: `simulation:${simulationId}:memory:jun-state`,
          snippet: "준 상태: offstage, waiting.",
          score: 0.93,
          metadata: { simulation_id: simulationId, session_id: sessionId, progress_run_id: runId }
        }
      ]
    },
    decisions: [],
    blockers: [],
    token_budget: 1800,
    created_at: now
  };
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => neuralMapResponse
  });

  try {
    const currentAction = "문을 닫고 미나와 잠깐 기다린다.";
    const pack = await new NeuralMapClient(state.neuralMap).getSimulationContext(
      {
        ...state,
        messages: [...state.messages, createUserMessage(state, currentAction)]
      },
      currentAction
    );
    const combinedEvidence = [
      ...pack.evidence,
      ...(pack.moduleEvidence ?? []),
      ...Object.values(pack.sections ?? {}).flat()
    ]
      .map((item) => item.snippet)
      .join("\n");
    const turn = await runSimulationTurn(state, currentAction, false, {
      deferMemoryIngest: true,
      deferImagePlanning: true
    });
    const selectedModuleIds = turn.promptModuleUsages.map((usage) => usage.moduleId).join(",");
    const preview = turn.sidecarTrace.requestPreview ?? "";

    assertCheck("neuralmap.cast", combinedEvidence.includes("미나"), "NeuralMap context keeps active character evidence.");
    assertCheck("neuralmap.cast", !combinedEvidence.includes("소라") && !combinedEvidence.includes("준"), "NeuralMap context drops inactive character-only evidence.");
    assertCheck("neuralmap.cast", selectedModuleIds.includes("module_cast_mina"), "Active character prompt can still be selected.");
    assertCheck("neuralmap.cast", !selectedModuleIds.includes("module_cast_sora"), "Inactive character prompt is not activated by NeuralMap evidence.");
    assertCheck("neuralmap.cast", !selectedModuleIds.includes("module_cast_jun"), "Inactive always character prompt is not included as a current-turn module.");
    assertCheck("neuralmap.cast", !preview.includes("소라 visual profile marker") && !preview.includes("준 visual profile marker"), "LLM prompt omits inactive character visual profiles.");
    assertCheck("neuralmap.cast", !preview.includes("offstage, waiting"), "LLM prompt omits inactive character relationship/status records.");
  } finally {
    globalThis.fetch = previousFetch;
  }
}

async function evaluateContinuityAnchors(seedState, createLocalContextPack, generateAssistantText) {
  const longPreviousAssistant = [
    "OPENING_CONTEXT_SHOULD_NOT_BE_THE_ONLY_VISIBLE_PART",
    "장면의 도입부가 길게 이어진다. ".repeat(120),
    "LATEST_ASSISTANT_ENDING_LOCK: Aria가 은색 열쇠를 손바닥에 올려놓고, 선택지는 분수로 내려갈지 기록실 문을 잠글지로 좁혀진다.",
    "```choice\n1. 분수 아래로 간다\n2. 기록실 문을 잠근다\n```"
  ].join("\n");
  const state = {
    ...structuredClone(seedState),
    llm: {
      ...seedState.llm,
      enabled: false,
      provider: "mock",
      apiKey: ""
    },
    messages: [
      ...seedState.messages,
      createUserMessage(seedState, "Aria에게 은색 열쇠를 보여 달라고 한다."),
      createAssistantMessage(seedState, longPreviousAssistant)
    ]
  };
  const optimisticState = {
    ...state,
    messages: [...state.messages, createUserMessage(state, "1번을 선택한다.")]
  };
  const contextPack = createLocalContextPack(optimisticState, "continuity anchor check");
  const combinedEvidence = contextPack.evidence.map((item) => item.snippet).join("\n");
  const generation = await generateAssistantText({
    state,
    userText: "1번을 선택한다.",
    modules: state.modules,
    evidence: contextPack.evidence,
    fallback: "fallback"
  });
  const preview = generation.requestPreview ?? "";

  assertCheck("continuity.anchor", combinedEvidence.includes("LATEST_ASSISTANT_ENDING_LOCK"), "Local Context Pack includes the latest assistant ending.");
  assertCheck("continuity.anchor", combinedEvidence.includes("1번을 선택한다."), "Local Context Pack includes the current user action.");
  assertCheck("continuity.anchor", preview.includes("Immediate continuity anchor"), "LLM request preview has an immediate continuity section.");
  assertCheck("continuity.anchor", preview.includes("LATEST_ASSISTANT_ENDING_LOCK"), "LLM request preview preserves the prior assistant ending.");
}

async function evaluateSceneCastGuard(seedState, createLocalContextPack, generateAssistantText, createStructuredContextSummary) {
  const state = {
    ...structuredClone(seedState),
    llm: {
      ...seedState.llm,
      enabled: false,
      provider: "mock",
      apiKey: ""
    },
    simulation: {
      ...seedState.simulation,
      title: "Scene Cast Guard Eval",
      description: "등록된 캐릭터가 여럿 있지만 최근 장면에는 미나만 있다."
    },
    characters: [
      {
        id: "char_mina",
        simulationId: seedState.simulation.id,
        name: "미나",
        role: "현재 대화 중인 인물",
        summary: "최근 장면에서 사용자와 복도에 서 있다.",
        relationship: "사용자를 신뢰한다.",
        currentMood: "조용히 고개를 끄덕임"
      },
      {
        id: "char_sora",
        simulationId: seedState.simulation.id,
        name: "소라",
        role: "관계도에 등록된 동료",
        summary: "현재 장면 밖에 있는 인물.",
        relationship: "관계도만으로 불려 나오면 안 되는 테스트 인물.",
        currentMood: "대기 중"
      },
      {
        id: "char_jun",
        simulationId: seedState.simulation.id,
        name: "준",
        role: "관계도에 등록된 조력자",
        summary: "현재 장면 밖에 있는 인물.",
        relationship: "관계도만으로 불려 나오면 안 되는 테스트 인물.",
        currentMood: "대기 중"
      }
    ],
    messages: [
      createUserMessage(seedState, "미나에게 복도 끝을 확인해 달라고 한다."),
      createAssistantMessage(seedState, "미나는 복도 끝을 한 번 보고 돌아서서, 아직 아무도 오지 않았다고 낮게 말한다.")
    ],
    memoryEvents: [],
    contextPacks: []
  };
  const currentAction = "문을 닫고 잠깐 기다린다.";
  const optimisticState = {
    ...state,
    messages: [...state.messages, createUserMessage(state, currentAction)]
  };
  const summary = createStructuredContextSummary(optimisticState, { currentText: currentAction });
  const activeCharactersLine = summary.split("\n").find((line) => line.startsWith("- Active Characters:")) ?? "";
  const contextPack = createLocalContextPack(optimisticState, "scene cast guard");
  const combinedEvidence = contextPack.evidence.map((item) => `${item.reason}: ${item.snippet}`).join("\n");
  const generation = await generateAssistantText({
    state,
    userText: currentAction,
    modules: state.modules,
    evidence: contextPack.evidence,
    fallback: "fallback"
  });
  const preview = generation.requestPreview ?? "";

  assertCheck("scene.cast", activeCharactersLine.includes("미나"), "Structured context keeps recently mentioned characters active.");
  assertCheck("scene.cast", !activeCharactersLine.includes("소라") && !activeCharactersLine.includes("준"), "Structured context does not mark the whole roster as active.");
  assertCheck("scene.cast", combinedEvidence.includes("미나"), "Local Context Pack includes active character evidence.");
  assertCheck("scene.cast", !combinedEvidence.includes("소라") && !combinedEvidence.includes("준"), "Local Context Pack does not inject off-stage relationship-map characters as evidence.");
  assertCheck("scene.cast", preview.includes("relationship-map guard") || preview.includes("relationship map entries"), "LLM request preview includes the roster/relationship guard.");
}

function evaluateImageCastContinuity(seedState, inferCurrentSceneCharacterIds) {
  const state = {
    ...structuredClone(seedState),
    userPersona: {
      ...seedState.userPersona,
      enabled: false
    },
    characters: [
      {
        id: "char_mina",
        simulationId: seedState.simulation.id,
        name: "미나",
        role: "이전 장면 인물",
        summary: "이번 컷에는 나오지 않는다.",
        relationship: "",
        currentMood: ""
      },
      {
        id: "char_sora",
        simulationId: seedState.simulation.id,
        name: "소라",
        role: "최근 이미지에 나온 인물",
        summary: "이름이 반복되지 않아도 다음 컷의 주체로 유지되어야 한다.",
        relationship: "",
        currentMood: ""
      }
    ],
    imageAssets: [
      {
        id: "asset_eval_sora",
        simulationId: seedState.simulation.id,
        title: "최근 소라 컷",
        source: "generated",
        prompt: "1girl, black hair, hallway",
        negativePrompt: "",
        safetyLevel: seedState.imageProfile.safetyLevel,
        characterIds: ["char_sora"],
        tags: ["1girl", "black hair", "hallway"],
        createdAt: new Date().toISOString(),
        jobId: "imgjob_eval_sora",
        palette: ["#111111", "#eeeeee", "#777777"],
        reuseTags: ["1girl", "black hair", "hallway"]
      }
    ],
    imageJobs: [
      {
        id: "imgjob_eval_sora",
        simulationId: seedState.simulation.id,
        sessionId: seedState.simulation.activeSessionId,
        turnId: "msg_eval_assistant_image",
        status: "completed",
        reason: "최근 이미지 컷",
        prompt: "1girl, black hair, hallway",
        negativePrompt: "",
        providerPayload: {
          cue: {
            characters: ["char_sora"]
          }
        },
        assetIds: ["asset_eval_sora"],
        contextNodeIds: [],
        createdAt: new Date().toISOString(),
        completedAt: new Date().toISOString(),
        representativeAssetId: "asset_eval_sora"
      }
    ],
    turnTraces: [],
    messages: [
      createUserMessage(seedState, "그녀에게 고개를 끄덕인다."),
      {
        ...createAssistantMessage(seedState, "그녀는 복도 쪽으로 몸을 돌린 채 잠시 숨을 고른다."),
        id: "msg_eval_assistant_image",
        imageAssetIds: ["asset_eval_sora"]
      }
    ]
  };
  const activeIds = inferCurrentSceneCharacterIds(state, "그대로 이어간다.");
  const transitionIds = inferCurrentSceneCharacterIds(state, "장면 전환. 미나는 다른 방으로 이동한다.");

  assertCheck("scene.cast", activeIds.includes("char_sora"), "Recent generated image metadata keeps the visible character active for pronoun-only continuation.");
  assertCheck("scene.cast", !activeIds.includes("char_mina"), "Recent generated image metadata does not activate unrelated roster characters.");
  assertCheck("scene.cast", transitionIds.includes("char_mina") && !transitionIds.includes("char_sora"), "Explicit scene transition with a named character drops prior image-only cast continuity.");
}

function createEvalCharacter(simulationId, id, name, summary) {
  return {
    id,
    simulationId,
    name,
    role: "평가용 캐릭터",
    summary,
    relationship: "관계도 평가용 값",
    currentMood: "대기 중"
  };
}

function createEvalCharacterModule(simulationId, id, characterId, name, tokenPolicy) {
  return {
    id,
    simulationId,
    parentId: "module_cast_main",
    kind: "character_prompt",
    title: `캐릭터: ${name}`,
    body: `${name} 전용 캐릭터 프롬프트. 현재 장면 증거 없이 등장하면 안 된다.`,
    enabled: true,
    priority: 82,
    activationTags: [name.toLowerCase(), "character"],
    characterId,
    tokenPolicy,
    version: 1,
    updatedAt: new Date().toISOString()
  };
}

function createStateMemoryEvent({ simulationId, sessionId, runId, characterId, characterName, value }) {
  return {
    id: `memory_${characterId}_state`,
    simulationId,
    sessionId,
    actorId: characterId,
    actorName: characterName,
    content: `[State] ${characterName} StatusTags = ${value}`,
    importance: 0.78,
    tags: ["memory-delta", "kind:state", "state:StatusTags"],
    createdAt: new Date().toISOString(),
    metadata: {
      simulation_id: simulationId,
      session_id: sessionId,
      progress_run_id: runId,
      run_id: runId,
      memory_kind: "state",
      owner_id: characterId,
      state_type: "StatusTags",
      value
    }
  };
}

function createDeltaRecord(id, kind, patch) {
  return {
    id,
    kind,
    layer: kind === "summary" ? "semantic" : "episodic",
    content: patch.content,
    importance: patch.importance ?? 0.82,
    confidence: patch.confidence ?? 0.86,
    tags: patch.tags ?? [],
    actorId: patch.actorId,
    actorName: patch.actorName,
    ownerId: patch.ownerId,
    targetId: patch.targetId,
    stateType: patch.stateType,
    value: patch.value,
    eventType: patch.eventType,
    observers: patch.observers,
    importanceReasons: patch.importanceReasons ?? ["eval_graph_roles"]
  };
}

function createUserMessage(state, content) {
  return {
    id: "msg_eval_user",
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    role: "user",
    content,
    createdAt: new Date().toISOString(),
    referencedNodeIds: [],
    imageAssetIds: []
  };
}

function createAssistantMessage(state, content) {
  return {
    id: "msg_eval_assistant",
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    role: "assistant",
    content,
    createdAt: new Date().toISOString(),
    referencedNodeIds: [],
    imageAssetIds: []
  };
}

function assertCheck(scope, passed, message) {
  checks.push({
    level: passed ? "pass" : "fail",
    scope,
    message
  });
}
