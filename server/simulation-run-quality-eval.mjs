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
  const { seedState } = await vite.ssrLoadModule("/src/data/seed.ts");
  const { activateSimulationProgressRun, createFreshSimulationRun } = await vite.ssrLoadModule("/src/services/simulationRuns.ts");
  const { NeuralMapClient } = await vite.ssrLoadModule("/src/services/neuralMapClient.ts");
  const { createLocalContextPack } = await vite.ssrLoadModule("/src/services/neuralMapClient.ts");
  const { generateAssistantText } = await vite.ssrLoadModule("/src/services/llmClient.ts");
  const { createStructuredContextSummary } = await vite.ssrLoadModule("/src/services/memoryCompiler.ts");
  evaluateFreshSimulationRun(seedState, createFreshSimulationRun, activateSimulationProgressRun);
  await evaluateNeuralMapScopeFiltering(seedState, NeuralMapClient);
  await evaluateContinuityAnchors(seedState, createLocalContextPack, generateAssistantText);
  await evaluateSceneCastGuard(seedState, createLocalContextPack, generateAssistantText, createStructuredContextSummary);
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
