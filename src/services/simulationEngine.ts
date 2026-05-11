import { createId } from "../lib/id";
import type {
  AppState,
  AssistantImageCueDraft,
  AssistantMemoryEventDraft,
  AssistantSidecar,
  ChatMessage,
  ContinuityCheck,
  ContextPack,
  ImageCue,
  ImageGenerationCadence,
  MemoryEvent,
  PromptModule,
  PromptModuleUsage,
  SessionHandoff,
  SidecarTrace,
  TurnTrace,
  TurnResult
} from "../types";

import { findReusableImageAsset, pickStoredAsset, planImageJob, shouldPlanImageJob } from "./imageOrchestrator";
import { createCurrentImageCueStateTags } from "./imageStateTags";
import { generateAssistantText, generateImageCuePlans } from "./llmClient";
import {
  compileSimulationMemoryDelta,
  createStructuredContextSummary,
  memoryDeltaToEvents,
  type MemoryDelta
} from "./memoryCompiler";
import { NeuralMapClient } from "./neuralMapClient";
import { createImageUserRulesForContentRating, isAdultContentMode } from "./contentRating";
import { getCurrentSceneCharacters, inferCurrentSceneCharacterIds } from "./sceneCast";

const IMPORTANT_PATTERN = /기억|약속|관계|갈등|위험|비밀|장소|문제|목표|선택|상태|변화|단서/u;
const IMAGE_PATTERN =
  /그려|보여|이미지|장면|모습|표정|빛|배경|의상|옷|복장|학교|교실|연습|훈련|무대|숙소|기숙|주방|거리|사무실|카페|병원|전투|여행/u;
const MAX_ACTIVE_MODULE_BODY_CHARS = 2400;
const MODULE_EXCERPT_WINDOW_CHARS = 700;
const MODULE_EXCERPT_MAX_WINDOWS = 3;
const MAX_SELECTED_PROMPT_MODULES = 12;
const RETRIEVAL_RECENT_MESSAGE_CHARS = 520;
const RETRIEVAL_LATEST_ASSISTANT_CHARS = 1400;
const RETRIEVAL_SETTING_MODULE_CHARS = 360;

interface RunSimulationTurnOptions {
  deferImagePlanning?: boolean;
  deferMemoryIngest?: boolean;
  onAssistantText?: (snapshot: {
    userMessage: ChatMessage;
    assistantMessage: ChatMessage;
  }) => void;
  onMemoryIngested?: (snapshot: {
    turnId: string;
    memoryEvents: MemoryEvent[];
    memoryIngestMs: number;
  }) => void;
}

interface ImageRuleBackedCueDraft extends AssistantImageCueDraft {
  plannerSource?: "main_llm_sidecar" | "dedicated_image_planner" | "user_image_rules" | "image_generation_cadence";
  forceFreshImage?: boolean;
  forceImagePlanning?: boolean;
  forceLocalVisualTags?: boolean;
}

interface ImageUserRuleCuePlan {
  requiresGeneration: boolean;
  forceFresh: boolean;
  ignoreCooldown: boolean;
  requireWholeScene: boolean;
  requireActionBeat: boolean;
  requireBodyDetail: boolean;
  requireDialogueFace: boolean;
}

export interface CompletedTurnImagePlan {
  imageCue: ImageCue;
  imageJobs: ReturnType<typeof planImageJob>[];
  imageJob?: ReturnType<typeof planImageJob>;
  reusedAssetIds: string[];
}

export async function runSimulationTurn(
  state: AppState,
  userText: string,
  manualImage = false,
  options: RunSimulationTurnOptions = {}
): Promise<TurnResult> {
  const startedAt = Date.now();
  const userMessage: ChatMessage = {
    id: createId("msg"),
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    role: "user",
    content: userText,
    createdAt: new Date().toISOString(),
    referencedNodeIds: [],
    imageAssetIds: []
  };
  const optimisticState = {
    ...state,
    messages: [...state.messages, userMessage]
  };
  const personaAwareUserText = createPersonaAwareUserText(state, userText);
  const retrievalQuery = createTurnRetrievalQuery(state, personaAwareUserText, userText);
  const neuralMap = new NeuralMapClient(state.neuralMap);
  const retrievalStartedAt = Date.now();
  const contextPack = await neuralMap.getSimulationContext(optimisticState, retrievalQuery);
  const retrievalLatencyMs = Date.now() - retrievalStartedAt;
  const moduleSelections = selectRelevantModules(state, retrievalQuery, contextPack);
  const relevantModules = moduleSelections.map((selection) => selection.module);
  const promptModuleUsages = createPromptModuleUsages(
    state,
    userMessage.id,
    moduleSelections,
    contextPack.createdAt
  );
  const fallbackContent = createAssistantContent(state, userText, relevantModules, contextPack.evidence.map((item) => item.snippet));
  const assistantMessageBase: ChatMessage = {
    id: createId("msg"),
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    role: "assistant",
    content: "",
    createdAt: new Date().toISOString(),
    referencedNodeIds: contextPack.evidence.map((item) => item.nodeId),
    imageAssetIds: []
  };
  const llmStartedAt = Date.now();
  const assistantGeneration = await generateAssistantText({
    state,
    userText,
    modules: relevantModules,
    evidence: contextPack.evidence,
    fallback: fallbackContent,
    onAssistantText: options.onAssistantText
      ? (content) =>
          options.onAssistantText?.({
            userMessage,
            assistantMessage: {
              ...assistantMessageBase,
              content
            }
          })
      : undefined
  });
  const llmRequestMs = Date.now() - llmStartedAt;
  const assistantContent = assistantGeneration.content;
  const sidecarTrace = createSidecarTrace(state, userMessage.id, assistantGeneration);
  const compiledMemoryDelta = compileSimulationMemoryDelta({
    state,
    userText,
    assistantText: assistantContent,
    sidecar: assistantGeneration.sidecar,
    sourceTurnId: userMessage.id
  });
  const memoryEvents = shouldCurateAssistantMemory(assistantGeneration)
    ? memoryDeltaToEvents(state, compiledMemoryDelta)
    : [];
  const memoryIngestStartedAt = Date.now();
  let finalizedMemoryEvents = memoryEvents;
  let memoryIngestMs = 0;
  if (options.deferMemoryIngest) {
    if (memoryEvents.length > 0) {
      void commitMemoryDeltaToNeuralMap(neuralMap, state, compiledMemoryDelta, memoryEvents)
        .then((committedMemoryEvents) => {
          options.onMemoryIngested?.({
            turnId: userMessage.id,
            memoryEvents: committedMemoryEvents,
            memoryIngestMs: Date.now() - memoryIngestStartedAt
          });
        })
        .catch(() => undefined);
    }
  } else {
    finalizedMemoryEvents = await commitMemoryDeltaToNeuralMap(neuralMap, state, compiledMemoryDelta, memoryEvents);
    memoryIngestMs = Date.now() - memoryIngestStartedAt;
  }
  const assistantMessage: ChatMessage = {
    ...assistantMessageBase,
    content: assistantContent,
  };
  const nextStateForImage = {
    ...optimisticState,
    messages: [...optimisticState.messages, assistantMessage],
    memoryEvents: [...state.memoryEvents, ...finalizedMemoryEvents],
    contextPacks: [...state.contextPacks, contextPack]
  };
  const imagePlan: CompletedTurnImagePlan = options.deferImagePlanning
    ? {
        imageCue: createPendingImageCue(state, userText, assistantContent, manualImage),
        imageJobs: [],
        reusedAssetIds: []
      }
    : await planImageJobForCompletedTurn(nextStateForImage, {
        userMessage,
        assistantMessage,
        contextPack,
        promptModuleUsages,
        sidecar: assistantGeneration.sidecar,
        sidecarTrace,
        manualImage
      });
  const imageCue = imagePlan.imageCue;
  const jobs = imagePlan.imageJobs.length > 0 ? imagePlan.imageJobs : imagePlan.imageJob ? [imagePlan.imageJob] : [];
  const job = imagePlan.imageJob ?? jobs[0];
  const reusedAssetIds = imagePlan.reusedAssetIds ?? [];

  if (jobs.length > 0) {
    const turnTrace = createTurnTrace({
      state,
      userMessage,
      assistantMessage,
      contextPack,
      promptModuleUsages,
      selectedModules: relevantModules,
      sidecarTrace,
      memoryEvents: finalizedMemoryEvents,
      imageCue,
      imageJob: job,
      imageJobs: jobs,
      imageAssetIds: reusedAssetIds,
      latencyMs: llmRequestMs,
      retrievalLatencyMs,
      memoryIngestMs,
      turnLatencyMs: Date.now() - startedAt
    });
    return {
      userMessage,
      assistantMessage: {
        ...assistantMessage,
        imageAssetIds: reusedAssetIds
      },
      memoryEvents: finalizedMemoryEvents,
      contextPack,
      promptModuleUsages,
      sidecarTrace,
      sidecar: assistantGeneration.sidecar,
      imageCue,
      turnTrace,
      imageJob: job,
      imageJobs: jobs,
      imageAssets: []
    };
  }

  const storedAsset = reusedAssetIds.length > 0 ? undefined : pickStoredAsset(state, imageCue);
  const imageAssetIds = reusedAssetIds.length > 0 ? reusedAssetIds : storedAsset ? [storedAsset.id] : [];
  const turnTrace = createTurnTrace({
    state,
    userMessage,
    assistantMessage,
    contextPack,
    promptModuleUsages,
    selectedModules: relevantModules,
    sidecarTrace,
    memoryEvents: finalizedMemoryEvents,
    imageCue,
    imageJobs: jobs,
    imageAssetIds,
    latencyMs: llmRequestMs,
    retrievalLatencyMs,
    memoryIngestMs,
    turnLatencyMs: Date.now() - startedAt
  });
  return {
    userMessage,
    assistantMessage: {
      ...assistantMessage,
      imageAssetIds
    },
    memoryEvents: finalizedMemoryEvents,
    contextPack,
    promptModuleUsages,
    sidecarTrace,
    sidecar: assistantGeneration.sidecar,
    imageCue,
    turnTrace,
    imageJobs: jobs,
    imageAssets: []
  };
}

async function commitMemoryDeltaToNeuralMap(
  neuralMap: NeuralMapClient,
  state: AppState,
  delta: MemoryDelta,
  memoryEvents: MemoryEvent[]
): Promise<MemoryEvent[]> {
  if (memoryEvents.length === 0) {
    return [];
  }

  try {
    const committedNodeIds = await neuralMap.applyMemoryDelta(delta, state);
    return memoryEvents.map((memoryEvent) => {
      const recordId = readMemoryEventMetadataText(memoryEvent, "memory_record_id");
      const graphNeuronId = readMemoryEventMetadataText(memoryEvent, "graph_neuron_id");
      return {
        ...memoryEvent,
        neuralMapNodeId: graphNeuronId ?? (recordId ? committedNodeIds.get(recordId) : undefined) ?? memoryEvent.neuralMapNodeId
      };
    });
  } catch {
    return Promise.all(
      memoryEvents.map(async (memoryEvent) => {
        try {
          return {
            ...memoryEvent,
            neuralMapNodeId: await neuralMap.ingestEvent(memoryEvent, state)
          };
        } catch {
          return memoryEvent;
        }
      })
    );
  }
}

function readMemoryEventMetadataText(event: MemoryEvent, key: string): string | undefined {
  const value = event.metadata?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

export async function planImageJobForCompletedTurn(
  state: AppState,
  input: {
    userMessage: ChatMessage;
    assistantMessage: ChatMessage;
    contextPack: ContextPack;
    promptModuleUsages: PromptModuleUsage[];
    sidecar?: AssistantSidecar;
    sidecarTrace?: SidecarTrace;
    manualImage?: boolean;
  }
): Promise<CompletedTurnImagePlan> {
  const sidecar = input.sidecar ?? createFallbackImageSidecar(input.assistantMessage.content, input.manualImage);
  const baseDrafts = sidecar.imageCues.length > 0 ? sidecar.imageCues : [sidecar.imageCue];
  const initialDrafts = applyImageUserRuleCuePlan(
    state,
    input.userMessage.content,
    input.assistantMessage.content,
    baseDrafts,
    Boolean(input.manualImage)
  );
  const drafts = await planImageCueDraftsWithLlm(
    state,
    input.userMessage.content,
    input.assistantMessage.content,
    initialDrafts,
    input.sidecarTrace,
    Boolean(input.manualImage)
  );
  const includeLocalVisualTags = input.sidecarTrace
    ? input.sidecarTrace.source !== "llm" || input.sidecarTrace.status !== "parsed"
    : true;
  const trustLlmCharacterScope = input.sidecarTrace?.source === "llm" && input.sidecarTrace.status === "parsed";
  const imageCues = drafts.map((draft) =>
    planImageCue(
      state,
      input.userMessage.content,
      draft,
      input.assistantMessage.content,
      {
        allowLocalTrigger: false,
        includeLocalVisualTags:
          draft.plannerSource === "dedicated_image_planner"
            ? false
            : includeLocalVisualTags || draft.forceLocalVisualTags === true,
        trustLlmCharacterScope,
        manualImage: input.manualImage
      }
    )
  );
  const contextNodeIds = input.contextPack.evidence.map((item) => item.nodeId);
  const reusedAssetIds: string[] = [];
  const imageJobs = imageCues
    .map((imageCue, index) => {
      const draft = drafts[index];
      const canForcePlanning = shouldForceImagePlanningFromUserRules(state, draft, imageCue, Boolean(input.manualImage));
      if (!shouldPlanImageJob(state, imageCue, input.manualImage) && !canForcePlanning) {
        return undefined;
      }

      const plannedJob = addImageCuePlannerMetadata(
        planImageJob(state, input.assistantMessage.id, imageCue, contextNodeIds, input.manualImage),
        draft,
        index,
        state,
        input.sidecarTrace
      );
      const reuseMatch = input.manualImage
        ? undefined
        : findReusableImageAsset(state, plannedJob, { excludeAssetIds: reusedAssetIds });
      if (reuseMatch) {
        reusedAssetIds.push(reuseMatch.asset.id);
        return undefined;
      }

      return plannedJob;
    })
    .filter((job): job is ReturnType<typeof planImageJob> => Boolean(job));
  const primaryCueIndex =
    typeof imageJobs[0]?.providerPayload.cueIndex === "number" ? imageJobs[0].providerPayload.cueIndex : 0;
  const primaryImageCue =
    imageJobs.length > 0
      ? imageCues[primaryCueIndex] ?? imageCues[0]
      : imageCues[0] ?? createPendingImageCue(state, input.userMessage.content, input.assistantMessage.content, input.manualImage);

  return {
    imageCue: primaryImageCue,
    imageJobs,
    imageJob: imageJobs[0],
    reusedAssetIds
  };
}

async function planImageCueDraftsWithLlm(
  state: AppState,
  userText: string,
  assistantText: string,
  drafts: ImageRuleBackedCueDraft[],
  sidecarTrace: SidecarTrace | undefined,
  manualImage: boolean
): Promise<ImageRuleBackedCueDraft[]> {
  if (!shouldUseDedicatedImageCuePlanner(state, manualImage)) {
    return [];
  }

  const planner = await generateImageCuePlans({
    state,
    userText,
    assistantText,
    drafts,
    reason: [
      "Dedicated image planner owns final image cue selection and NovelAI tag generation for this completed simulation turn.",
      sidecarTrace ? `sidecar=${sidecarTrace.source}/${sidecarTrace.status}` : undefined
    ].filter(Boolean).join(" ")
  });

  if (planner.imageCues.length === 0) {
    return [];
  }

  const forceFreshImage = drafts.some((draft) => draft.forceFreshImage);
  const forceImagePlanning = drafts.some((draft) => draft.forceImagePlanning);
  const forceLocalVisualTags = drafts.some((draft) => draft.forceLocalVisualTags);

  return planner.imageCues.map((cue, cueIndex) => {
    const base = findImageCueHintForPlannerCue(cue, drafts, cueIndex);
    const plannerTags = sanitizeImageCueTags(cue.tags);
    const plannerHasOwnVisualTags = plannerTags.length > 0;
    return {
      ...(base ?? {}),
      ...cue,
      shouldGenerate: cue.shouldGenerate,
      reason: cue.reason || base?.reason || "전용 이미지 플래너가 현재 문맥을 이미지 cue로 선택함",
      characters: cue.characters.length > 0 ? cue.characters : base?.characters ?? [],
      tags: plannerTags,
      scene: cue.scene || base?.scene || "current simulation scene",
      visualContext: cue.visualContext ?? plannerTags.join(", "),
      suppressionReason: cue.suppressionReason ?? base?.suppressionReason,
      label: cue.label ?? base?.label,
      kind: cue.kind ?? base?.kind,
      cueType: cue.cueType ?? base?.cueType,
      placement: cue.placement ?? base?.placement,
      anchorText: cue.anchorText ?? base?.anchorText,
      priority: cue.priority ?? base?.priority,
      plannerSource: "dedicated_image_planner",
      forceFreshImage: forceFreshImage || base?.forceFreshImage,
      forceImagePlanning: forceImagePlanning || base?.forceImagePlanning,
      forceLocalVisualTags: plannerHasOwnVisualTags ? false : forceLocalVisualTags || base?.forceLocalVisualTags
    };
  });
}

function shouldUseDedicatedImageCuePlanner(state: AppState, manualImage: boolean): boolean {
  if (!state.imageProfile.enabled || !state.simulation.realtimeImageEnabled || state.imageProfile.triggerMode === "stored_only") {
    return false;
  }
  if (state.imageProfile.triggerMode === "manual" && !manualImage) {
    return false;
  }
  return state.llm.enabled && state.llm.provider !== "mock" && Boolean(state.llm.apiKey.trim());
}

function hasForcedImageCueDraft(drafts: ImageRuleBackedCueDraft[]): boolean {
  return drafts.some((draft) => draft.forceImagePlanning || draft.forceFreshImage || draft.plannerSource === "user_image_rules" || draft.plannerSource === "image_generation_cadence");
}

function findImageCueHintForPlannerCue(
  cue: AssistantImageCueDraft,
  drafts: ImageRuleBackedCueDraft[],
  cueIndex: number
): ImageRuleBackedCueDraft | undefined {
  const cueKind = cue.kind ?? cue.cueType;
  const sameKind = cueKind ? drafts.find((draft) => (draft.kind ?? draft.cueType) === cueKind) : undefined;
  if (sameKind) {
    return sameKind;
  }

  return drafts[cueIndex] ?? drafts.find((draft) => draft.shouldGenerate || draft.forceImagePlanning) ?? drafts[0];
}

function isGenericImageSceneTag(value: string): boolean {
  return /^(?:(?:current|generated|simulation|safe)\s+)*scene$/iu.test(value.trim());
}

function addImageCuePlannerMetadata(
  job: ReturnType<typeof planImageJob>,
  draft: ImageRuleBackedCueDraft,
  cueIndex: number,
  state: AppState,
  sidecarTrace?: SidecarTrace
): ReturnType<typeof planImageJob> {
  const cueKind = draft.kind ?? draft.cueType;
  const plannerSource = draft.plannerSource ?? "main_llm_sidecar";
  return {
    ...job,
    providerPayload: {
      ...job.providerPayload,
      cueIndex,
      cueLabel: draft.label,
      cueKind,
      cueType: cueKind,
      cuePlacement: draft.placement ?? "after",
      anchorText: draft.anchorText,
      cuePriority: draft.priority,
      forceFreshImage: draft.forceFreshImage === true,
      imageCuePlanner: {
        source: plannerSource,
        llmSource: sidecarTrace?.source ?? "fallback",
        status: sidecarTrace?.status ?? "fallback",
        model: state.llm.model,
        errors: sidecarTrace?.errors ?? []
      }
    }
  };
}

function applyImageUserRuleCuePlan(
  state: AppState,
  userText: string,
  assistantText: string,
  drafts: AssistantImageCueDraft[],
  manualImage: boolean
): ImageRuleBackedCueDraft[] {
  const cuePlan = analyzeImageUserRuleCuePlan(createImageUserRulesForContentRating(state));
  const cadence = resolveImageGenerationCadence(state);
  const currentDrafts = drafts.length > 0 ? drafts : [createFallbackImageSidecar(assistantText, manualImage).imageCue];
  const hasUsableCue = currentDrafts.some(
    (draft) =>
      draft.shouldGenerate ||
      draft.tags.length > 0 ||
      Boolean(draft.visualContext?.trim()) ||
      isSpecificImageCueKind(draft.kind ?? draft.cueType)
  );
  const cadenceRequiresGeneration = shouldImageCadenceRequireGeneration(cadence, userText, assistantText);
  const shouldReplaceNoImageCue = (cuePlan.requiresGeneration || cadenceRequiresGeneration) && !hasUsableCue;
  const preparedDrafts: ImageRuleBackedCueDraft[] = shouldReplaceNoImageCue
    ? []
    : currentDrafts.map((draft) => ({
        ...draft,
        shouldGenerate: draft.shouldGenerate || cuePlan.requiresGeneration || manualImage || cadenceRequiresGeneration,
        suppressionReason: draft.shouldGenerate || cuePlan.requiresGeneration || manualImage || cadenceRequiresGeneration ? undefined : draft.suppressionReason,
        reason:
          draft.reason ||
          (cuePlan.requiresGeneration
            ? "사용자 이미지 규칙이 현재 문맥 생성을 요구함"
            : cadenceRequiresGeneration
              ? "이미지 생성 밀도 설정이 현재 문맥 생성을 요구함"
              : manualImage
                ? "수동 이미지 생성 요청"
                : "메인 LLM 이미지 cue"),
        plannerSource: "main_llm_sidecar",
        forceFreshImage: cuePlan.forceFresh || cuePlan.ignoreCooldown || shouldImageCadenceForceFresh(cadence) || undefined,
        forceImagePlanning: cuePlan.ignoreCooldown || shouldImageCadenceForcePlanning(cadence) || undefined,
        forceLocalVisualTags: cuePlan.requiresGeneration || cadenceRequiresGeneration || undefined
      }));

  const withRuleCues = [...preparedDrafts];
  const textForDetection = `${userText}\n${assistantText}`;
  if ((cuePlan.requireWholeScene || (cuePlan.requiresGeneration && withRuleCues.length === 0)) && !hasImageCueKind(withRuleCues, "scene", "context")) {
    withRuleCues.push(createImageUserRuleCueDraft("scene", assistantText, cuePlan));
  }
  if (cuePlan.requireActionBeat && hasActionBeatText(textForDetection) && !hasImageCueKind(withRuleCues, "action")) {
    withRuleCues.push(createImageUserRuleCueDraft("action", assistantText, cuePlan));
  }
  if (cuePlan.requireBodyDetail && !hasImageCueKind(withRuleCues, "body_detail")) {
    withRuleCues.push(createImageUserRuleCueDraft("body_detail", assistantText, cuePlan));
  }
  if (cuePlan.requireDialogueFace && hasDialogueText(assistantText) && !hasImageCueKind(withRuleCues, "dialogue_face")) {
    withRuleCues.push(createImageUserRuleCueDraft("dialogue_face", assistantText, cuePlan));
  }

  return applyImageGenerationCadenceCuePlan(
    userText,
    assistantText,
    withRuleCues,
    cadence,
    cuePlan.requiresGeneration
  );
}

function resolveImageGenerationCadence(state: AppState): ImageGenerationCadence {
  const cadence = state.imageProfile.generationCadence;
  return cadence === "sparse" || cadence === "balanced" || cadence === "rich" || cadence === "paragraph"
    ? cadence
    : "balanced";
}

function shouldImageCadenceRequireGeneration(cadence: ImageGenerationCadence, userText: string, assistantText: string): boolean {
  if (cadence === "paragraph") {
    return Boolean(assistantText.trim());
  }
  if (cadence === "rich") {
    return hasVisualCueText(`${userText}\n${assistantText}`);
  }

  return false;
}

function shouldImageCadenceForcePlanning(cadence: ImageGenerationCadence): boolean {
  return cadence === "rich" || cadence === "paragraph";
}

function shouldImageCadenceForceFresh(cadence: ImageGenerationCadence): boolean {
  return cadence === "rich" || cadence === "paragraph";
}

function applyImageGenerationCadenceCuePlan(
  userText: string,
  assistantText: string,
  drafts: ImageRuleBackedCueDraft[],
  cadence: ImageGenerationCadence,
  hasUserRuleRequirement: boolean
): ImageRuleBackedCueDraft[] {
  const withCadenceCues = [...drafts];
  const cadenceRequiresGeneration = shouldImageCadenceRequireGeneration(cadence, userText, assistantText);

  if (cadence === "paragraph") {
    const anchors = selectVisualParagraphAnchors(assistantText);
    anchors.forEach((anchorText, index) => {
      if (hasDraftAnchor(withCadenceCues, anchorText)) {
        return;
      }
      withCadenceCues.push(createImageCadenceCueDraft(index === 0 ? "scene" : "action", assistantText, anchorText, index));
    });
  } else if (cadence === "rich" && cadenceRequiresGeneration) {
    if (!hasImageCueKind(withCadenceCues, "scene", "context")) {
      withCadenceCues.push(createImageCadenceCueDraft("scene", assistantText, selectImageCueAnchorText(assistantText, "scene"), 0));
    }
    if (hasActionBeatText(`${userText}\n${assistantText}`) && !hasImageCueKind(withCadenceCues, "action")) {
      withCadenceCues.push(createImageCadenceCueDraft("action", assistantText, selectImageCueAnchorText(assistantText, "action"), 1));
    }
    if (hasDialogueText(assistantText) && !hasImageCueKind(withCadenceCues, "dialogue_face")) {
      withCadenceCues.push(createImageCadenceCueDraft("dialogue_face", assistantText, selectImageCueAnchorText(assistantText, "dialogue_face"), 2));
    }
  }

  const normalizedDrafts = withCadenceCues.map((draft) => {
    if (!cadenceRequiresGeneration || draft.plannerSource === "user_image_rules") {
      return draft;
    }
    return {
      ...draft,
      shouldGenerate: draft.shouldGenerate || cadenceRequiresGeneration,
      suppressionReason: undefined,
      reason: draft.reason || "이미지 생성 밀도 설정이 현재 문맥 생성을 요구함",
      forceFreshImage: draft.forceFreshImage || shouldImageCadenceForceFresh(cadence) || undefined,
      forceImagePlanning: draft.forceImagePlanning || shouldImageCadenceForcePlanning(cadence) || undefined,
      forceLocalVisualTags: true
    };
  });
  const maxCueCount = hasUserRuleRequirement ? 8 : getImageCadenceMaxCueCount(cadence);
  return normalizedDrafts.slice(0, maxCueCount);
}

function getImageCadenceMaxCueCount(cadence: ImageGenerationCadence): number {
  if (cadence === "sparse") {
    return 1;
  }
  if (cadence === "rich") {
    return 4;
  }
  if (cadence === "paragraph") {
    return 8;
  }

  return 2;
}

function createImageCadenceCueDraft(
  kind: NonNullable<AssistantImageCueDraft["kind"]>,
  assistantText: string,
  anchorText: string | undefined,
  index: number
): ImageRuleBackedCueDraft {
  const normalizedKind = kind === "context" ? "scene" : kind;
  const placement = index === 0 || normalizedKind === "scene" ? "before" : "inline";
  const tags =
    normalizedKind === "dialogue_face"
      ? ["face focus"]
      : normalizedKind === "action"
        ? []
        : [];
  const visualContext =
    normalizedKind === "dialogue_face"
      ? "face focus"
      : normalizedKind === "action"
        ? undefined
        : undefined;

  return {
    shouldGenerate: true,
    reason: "이미지 생성 밀도 설정이 이 문맥의 별도 이미지를 요구함",
    characters: [],
    scene: "current simulation scene",
    plannerSource: "image_generation_cadence",
    forceFreshImage: true,
    forceImagePlanning: true,
    forceLocalVisualTags: true,
    anchorText,
    priority: Math.max(0.72, 0.9 - index * 0.03),
    kind: normalizedKind,
    label: normalizedKind === "dialogue_face" ? "dialogue face" : normalizedKind === "action" ? "action beat" : "scene establishing",
    placement,
    tags,
    visualContext
  };
}

function selectVisualParagraphAnchors(assistantText: string): string[] {
  const trimmed = assistantText.trim();
  if (!trimmed) {
    return [];
  }

  const blockCandidates = trimmed
    .split(/\n{2,}/u)
    .map(cleanImageAnchorText)
    .filter(Boolean);
  const candidates =
    blockCandidates.length > 1
      ? blockCandidates
      : trimmed
          .split(/\n+/u)
          .map(cleanImageAnchorText)
          .filter(Boolean);

  return uniqueStrings(candidates.filter((item) => item.length >= 8).map((item) => item.slice(0, 140))).slice(0, 8);
}

function cleanImageAnchorText(value: string): string {
  return value
    .replace(/```[a-zA-Z0-9_-]*\n?/gu, "")
    .replace(/```/gu, "")
    .replace(/^[:>#*\-\s]+/gmu, "")
    .replace(/\s+/gu, " ")
    .trim();
}

function hasDraftAnchor(drafts: ImageRuleBackedCueDraft[], anchorText: string): boolean {
  const normalizedAnchor = normalizeAnchorForComparison(anchorText);
  return drafts.some((draft) => normalizeAnchorForComparison(draft.anchorText ?? "").includes(normalizedAnchor.slice(0, 48)));
}

function normalizeAnchorForComparison(value: string): string {
  return value.replace(/\s+/gu, " ").trim().toLowerCase();
}

function analyzeImageUserRuleCuePlan(userRules: string): ImageUserRuleCuePlan {
  const rules = userRules.toLowerCase();
  const forceFresh =
    /재사용\s*(?:금지|하지\s*마|하지\s*말|안\s*함|불가)|기존\s*이미지\s*(?:사용|재사용)\s*(?:금지|하지\s*마|하지\s*말)|새(?:로|로운)\s*(?:이미지|컷)|fresh\s+image|do\s+not\s+reuse|no\s+reuse|never\s+reuse|always\s+generate|매번\s*(?:새로\s*)?생성|각\s*(?:문맥|장면|컷)[^\n]*(?:생성|그려|이미지)|(?:문맥|장면|컷)마다[^\n]*(?:생성|그려|이미지)/iu.test(rules);
  const requireWholeScene =
    /전체\s*장면|장면\s*전체|전체\s*컷|whole[-\s]?scene|establishing\s+(?:shot|image)|각\s*(?:문맥|장면|컷)|(?:문맥|장면|컷)마다|every\s+(?:scene|context|cut)/iu.test(rules);
  const requireActionBeat =
    /행동\s*(?:비트|마다|컷|장면|부분)|행위\s*(?:비트|마다|컷|장면|부분)|액션\s*(?:비트|마다|컷|장면)|동작\s*(?:마다|컷|장면|부분)|action\s+beat|every\s+action|state\s+change|상태\s*변화/iu.test(rules);
  const requireBodyDetail =
    /body\s+detail|body\s+before\s+dialogue|신체\s*(?:디테일|컷|묘사|부위|강조)|몸\s*(?:디테일|컷|묘사|강조)|손\s*(?:디테일|컷|강조)|눈\s*(?:디테일|컷|강조)|부위\s*(?:강조|컷)/iu.test(rules);
  const requireDialogueFace =
    /dialogue\s+face|face\s+before\s+dialogue|before\s+dialogue|대사\s*(?:전|앞|직전|이전|마다)|말하기\s*전|발화\s*전|신음\s*(?:전|앞|직전|이전|마다)|소리\s*(?:전|앞|직전|이전)|표정\s*(?:컷|마다)/iu.test(rules);
  const alwaysGenerate =
    /항상[^\n]*(?:이미지|생성|그려)|모든\s*(?:assistant|응답|턴|문맥|장면)[^\n]*(?:이미지|생성|그려)|매\s*(?:턴|문맥|장면)[^\n]*(?:이미지|생성|그려)|every\s+(?:turn|response)[^\n]*(?:image|generate)/iu.test(rules);
  const requiresGeneration =
    forceFresh || requireWholeScene || requireActionBeat || requireBodyDetail || requireDialogueFace || alwaysGenerate;

  return {
    requiresGeneration,
    forceFresh,
    ignoreCooldown: forceFresh || alwaysGenerate || requireWholeScene || requireActionBeat || requireBodyDetail || requireDialogueFace,
    requireWholeScene: requireWholeScene || alwaysGenerate,
    requireActionBeat,
    requireBodyDetail,
    requireDialogueFace
  };
}

function createImageUserRuleCueDraft(kind: NonNullable<AssistantImageCueDraft["kind"]>, assistantText: string, cuePlan: ImageUserRuleCuePlan): ImageRuleBackedCueDraft {
  const anchorText = selectImageCueAnchorText(assistantText, kind);
  const base = {
    shouldGenerate: true,
    reason: "사용자 이미지 규칙이 이 문맥의 별도 이미지를 요구함",
    characters: [],
    scene: "current simulation scene",
    plannerSource: "user_image_rules" as const,
    forceFreshImage: cuePlan.forceFresh || cuePlan.ignoreCooldown,
    forceImagePlanning: cuePlan.ignoreCooldown,
    forceLocalVisualTags: true,
    anchorText,
    priority: kind === "scene" ? 0.92 : 0.86,
    kind
  };

  if (kind === "action") {
    const tags = createLocalImageCueTags(anchorText ?? assistantText, undefined, { ...base, kind });
    return {
      ...base,
      label: "action beat",
      placement: "inline",
      tags
    };
  }
  if (kind === "body_detail") {
    const tags = createLocalImageCueTags(anchorText ?? assistantText, undefined, { ...base, kind });
    return {
      ...base,
      label: "body detail",
      placement: "inline",
      tags: tags.length > 0 ? tags : ["close-up", "hands"]
    };
  }
  if (kind === "dialogue_face") {
    const tags = createLocalImageCueTags(anchorText ?? assistantText, undefined, { ...base, kind });
    return {
      ...base,
      label: "dialogue face",
      placement: "before",
      tags: tags.length > 0 ? tags : ["close-up", "face focus", "open mouth"],
      visualContext: (tags.length > 0 ? tags : ["close-up", "face focus", "open mouth"]).join(", ")
    };
  }

  const tags = createLocalImageCueTags(anchorText ?? assistantText, undefined, { ...base, kind });
  return {
    ...base,
    label: "scene establishing",
    placement: "before",
    tags
  };
}

function shouldForceImagePlanningFromUserRules(
  state: AppState,
  draft: ImageRuleBackedCueDraft | undefined,
  cue: ImageCue,
  manualImage: boolean
): boolean {
  if (!draft?.forceImagePlanning || !cue.shouldGenerate) {
    return false;
  }
  if (!state.imageProfile.enabled || !state.simulation.realtimeImageEnabled || state.imageProfile.triggerMode === "stored_only") {
    return false;
  }
  if (state.imageProfile.triggerMode === "manual" && !manualImage) {
    return false;
  }
  return true;
}

function hasImageCueKind(drafts: ImageRuleBackedCueDraft[], ...kinds: string[]): boolean {
  return drafts.some((draft) => {
    const kind = draft.kind ?? draft.cueType;
    return kind ? kinds.includes(kind) : false;
  });
}

function isSpecificImageCueKind(kind: string | undefined): boolean {
  return Boolean(kind && !["context", "scene"].includes(kind));
}

function selectImageCueAnchorText(assistantText: string, kind: string): string | undefined {
  const lines = assistantText
    .split(/\n+/u)
    .map((line) => line.replace(/^[>*#\-\s]+/u, "").trim())
    .filter(Boolean);
  const dialogueLine = lines.find((line) => /["“”「」『』]|^\S.{0,20}[:：]|대사|말하|속삭|외치|신음|숨소리|voice|says?|said|whisper|moan/iu.test(line));
  const bodyLine = lines.find((line) => /손|손목|팔|어깨|가슴|허리|허벅지|다리|발|눈|입술|시선|body|hand|wrist|arm|shoulder|chest|waist|thigh|leg|feet|eye|lips/iu.test(line));
  const actionLine = lines.find((line) => /움직|손|시선|몸|걸음|다가|멈추|잡|놓|돌아|밀|당기|뻗|올리|숙이|기대|action|gesture|looks?|steps?|reaches?|turns?|grabs?|holds?|leans?/iu.test(line));
  const selected =
    kind === "dialogue_face"
      ? dialogueLine ?? lines[0]
      : kind === "body_detail"
        ? bodyLine ?? actionLine ?? dialogueLine ?? lines[0]
        : kind === "action"
          ? actionLine ?? bodyLine ?? dialogueLine ?? lines[0]
        : lines[0];
  return selected ? selected.slice(0, 120) : undefined;
}

function hasDialogueText(text: string): boolean {
  return /["“”「」『』]|^\s*\S.{0,20}[:：]|대사|말하|속삭|외치|신음|숨소리|voice|says?|said|whisper|moan/imu.test(text);
}

function hasActionBeatText(text: string): boolean {
  return /움직|손|시선|몸|걸음|다가|멈추|잡|놓|돌아|밀|당기|뻗|올리|숙이|기대|action|gesture|looks?|steps?|reaches?|turns?|grabs?|holds?|leans?/iu.test(text);
}

function hasVisualCueText(text: string): boolean {
  return IMAGE_PATTERN.test(text) || hasActionBeatText(text) || hasDialogueText(text);
}

function createFallbackImageSidecar(assistantText: string, manualImage = false): AssistantSidecar {
  const imageCue: AssistantImageCueDraft = {
    shouldGenerate: manualImage,
    reason: manualImage ? "수동 이미지 생성 요청" : "메인 LLM 이미지 cue가 없어 생성을 보류함",
    characters: [],
    tags: [],
    scene: "current simulation scene",
    suppressionReason: manualImage ? undefined : "메인 LLM sidecar에 image_cues가 없음",
    visualContext: assistantText.slice(0, 220)
  };
  return {
    assistantText,
    memoryEvents: [],
    imageCue,
    imageCues: [imageCue]
  };
}

function createTurnTrace({
  state,
  userMessage,
  assistantMessage,
  contextPack,
  promptModuleUsages,
  selectedModules,
  sidecarTrace,
  memoryEvents,
  imageCue,
  imageJob,
  imageJobs,
  imageAssetIds,
  latencyMs,
  retrievalLatencyMs,
  memoryIngestMs,
  turnLatencyMs
}: {
  state: AppState;
  userMessage: ChatMessage;
  assistantMessage: ChatMessage;
  contextPack: ContextPack;
  promptModuleUsages: PromptModuleUsage[];
  selectedModules: PromptModule[];
  sidecarTrace: SidecarTrace;
  memoryEvents: MemoryEvent[];
  imageCue: ImageCue;
  imageJob?: ReturnType<typeof planImageJob>;
  imageJobs?: ImageGenerationJobLike[];
  imageAssetIds: string[];
  latencyMs: number;
  retrievalLatencyMs: number;
  memoryIngestMs: number;
  turnLatencyMs: number;
}): TurnTrace {
  const selectedModuleIds = new Set(promptModuleUsages.map((usage) => usage.moduleId));
  const selectedModuleTokenEstimate = estimateTokens(
    selectedModules
      .filter((module) => selectedModuleIds.has(module.id))
      .map((module) => `${module.title}\n${module.body}`)
  );
  const enabledModuleTokenEstimate = estimateTokens(
    state.modules
      .filter((module) => module.enabled && module.tokenPolicy !== "disabled")
      .map((module) => `${module.title}\n${module.body}`)
  );
  const contextTokenEstimate = estimateTokens(contextPack.evidence.map((item) => item.snippet));

  return {
    id: createId("turntrace"),
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    turnId: userMessage.id,
    userMessageId: userMessage.id,
    assistantMessageId: assistantMessage.id,
    contextPackId: contextPack.id,
    promptModuleUsageIds: promptModuleUsages.map((usage) => usage.id),
    sidecarTraceId: sidecarTrace.id,
    memoryEventIds: memoryEvents.map((event) => event.id),
    imageCue,
    imageJobId: imageJob?.id,
    imageAssetIds,
    metrics: {
      tokenBudget: contextPack.tokenBudget,
      selectedModuleCount: promptModuleUsages.length,
      selectedModuleTokenEstimate,
      contextEvidenceCount: contextPack.evidence.length,
      contextTokenEstimate,
      ragTokenSavingsEstimate: Math.max(0, enabledModuleTokenEstimate - selectedModuleTokenEstimate),
      llmLatencyMs: latencyMs,
      llmRequestMs: latencyMs,
      retrievalLatencyMs,
      memoryIngestMs,
      turnLatencyMs,
      memoryIngestCount: memoryEvents.length,
      imageJobCount: imageJobs?.length ?? (imageJob ? 1 : 0),
      imageAssetCount: imageAssetIds.length,
      imageEstimatedCost: readImageCostEstimate(imageJob, imageJobs)
    },
    createdAt: new Date().toISOString()
  };
}

function estimateTokens(chunks: string[]): number {
  const textLength = chunks.join("\n").length;
  return Math.ceil(textLength / 4);
}

function readImageCostEstimate(imageJob?: ImageGenerationJobLike, imageJobs: ImageGenerationJobLike[] = []): number | undefined {
  const jobs = imageJobs.length > 0 ? imageJobs : imageJob ? [imageJob] : [];
  const total = jobs.reduce((sum, job) => {
    const raw = job.providerPayload.estimatedAnlas ?? job.providerPayload.estimatedCost;
    return sum + (typeof raw === "number" && Number.isFinite(raw) ? raw : 0);
  }, 0);
  return total > 0 ? total : undefined;
}

type ImageGenerationJobLike = {
  providerPayload: Record<string, unknown>;
};

export async function createResetSessionState(state: AppState): Promise<AppState> {
  const previousSessionId = state.simulation.activeSessionId;
  const newSessionId = createId("session");
  const handoffMemory: MemoryEvent = {
    id: createId("memory"),
    simulationId: state.simulation.id,
    sessionId: previousSessionId,
    content: createHandoffSummary(state),
    importance: 0.92,
    tags: ["handoff", "session-reset", "continuity"],
    createdAt: new Date().toISOString()
  };
  const neuralMap = new NeuralMapClient(state.neuralMap);
  let neuralMapNodeId: string | undefined;
  try {
    neuralMapNodeId = await neuralMap.ingestEvent(handoffMemory, state);
  } catch {
    neuralMapNodeId = undefined;
  }
  const finalizedHandoffMemory = {
    ...handoffMemory,
    neuralMapNodeId
  };
  const stateWithHandoffEvent = {
    ...state,
    memoryEvents: [...state.memoryEvents, finalizedHandoffMemory]
  };
  const handoff = await neuralMap.createHandoff(stateWithHandoffEvent, newSessionId, finalizedHandoffMemory);
  const resetBaseState = {
    ...stateWithHandoffEvent,
    simulation: {
      ...state.simulation,
      activeSessionId: newSessionId,
      updatedAt: new Date().toISOString()
    },
    handoffs: [...state.handoffs, handoff]
  };
  const contextPack = await neuralMap.getSimulationContext(resetBaseState, createResetContextQuery(state));
  const continuityCheck = validateContinuity(state, handoff, contextPack, newSessionId);
  const systemMessage: ChatMessage = {
    id: createId("msg"),
    simulationId: state.simulation.id,
    sessionId: newSessionId,
    role: "system",
    content:
      continuityCheck.status === "passed"
        ? "세션이 초기화되었습니다. handoff와 새 Context Pack에서 핵심 연속성 단서를 확인했습니다."
        : `세션이 초기화되었습니다. 연속성 점검 경고: ${continuityCheck.warnings.join(" / ")}`,
    createdAt: new Date().toISOString(),
    referencedNodeIds: [handoff.id, ...handoff.evidenceNodeIds, ...contextPack.evidence.map((item) => item.nodeId)],
    imageAssetIds: []
  };

  return {
    ...resetBaseState,
    messages: [...state.messages, systemMessage],
    contextPacks: [...state.contextPacks, contextPack],
    selectedContextPackId: contextPack.id,
    continuityChecks: [...state.continuityChecks, continuityCheck]
  };
}

function createHandoffSummary(state: AppState): string {
  const characterState = state.characters
    .map((character) => `${character.name}: ${character.summary.slice(0, 160)}`)
    .join(" / ");
  const recentMemories = state.memoryEvents
    .slice(-5)
    .map((event) => event.content)
    .join(" / ");

  return [
    `세션 핸드오프: ${state.simulation.title}`,
    characterState ? `캐릭터 현재 상태: ${characterState}` : undefined,
    recentMemories ? `최근 핵심 기억: ${recentMemories}` : undefined
  ]
    .filter(Boolean)
    .join("\n");
}

function createResetContextQuery(state: AppState): string {
  const characterNames = state.characters.map((character) => character.name).join(", ");
  return [
    "세션 초기화 직후 이어갈 핵심 기억, 관계, 장면 상태, 약속, 미해결 단서를 검색한다.",
    characterNames ? `관련 캐릭터: ${characterNames}` : undefined,
    state.memoryEvents.at(-1)?.content
  ]
    .filter(Boolean)
    .join("\n");
}

function validateContinuity(
  previousState: AppState,
  handoff: SessionHandoff,
  contextPack: ContextPack,
  nextSessionId: string
): ContinuityCheck {
  const evidenceText = [
    handoff.summary,
    ...contextPack.evidence.map((item) => item.snippet),
    ...previousState.memoryEvents.slice(-5).map((event) => event.content)
  ]
    .join("\n")
    .toLowerCase();
  const facts = previousState.characters.map((character) => {
    const promptTerms = [character.summary.slice(0, 32), character.summary.slice(32, 64)].filter(Boolean);
    return {
      label: `${character.name} 상태 연속성`,
      expected: character.summary.slice(0, 80),
      found: hasContinuitySignal(evidenceText, promptTerms.length > 0 ? promptTerms : [character.name]),
      evidence: findEvidenceSnippet(contextPack, promptTerms) ?? handoff.summary
    };
  });
  const memoryFacts = previousState.memoryEvents
    .slice(-3)
    .map((event) => ({
      label: event.tags.includes("handoff") ? "이전 handoff" : event.tags[0] ?? "최근 기억",
      expected: event.content.slice(0, 80),
      found: hasContinuitySignal(evidenceText, [event.content.slice(0, 32), ...event.tags]),
      evidence: findEvidenceSnippet(contextPack, [event.content.slice(0, 32), ...event.tags]) ?? handoff.summary
    }));
  const allFacts = [...facts, ...memoryFacts];
  const warnings = allFacts
    .filter((fact) => !fact.found)
    .map((fact) => `${fact.label} 단서 부족`);

  return {
    id: createId("continuity"),
    simulationId: previousState.simulation.id,
    previousSessionId: previousState.simulation.activeSessionId,
    nextSessionId,
    handoffId: handoff.id,
    status: warnings.length > 0 ? "warning" : "passed",
    checkedAt: new Date().toISOString(),
    facts: allFacts,
    warnings
  };
}

function hasContinuitySignal(text: string, terms: string[]): boolean {
  return terms
    .filter(Boolean)
    .some((term) => text.includes(term.toLowerCase().slice(0, 24)));
}

function findEvidenceSnippet(contextPack: ContextPack, terms: string[]): string | undefined {
  return contextPack.evidence.find((item) =>
    terms.filter(Boolean).some((term) => item.snippet.toLowerCase().includes(term.toLowerCase().slice(0, 24)))
  )?.snippet;
}

interface PromptModuleSelection {
  module: PromptModule;
  source: PromptModuleUsage["source"];
  reason: string;
  score: number;
}

function selectRelevantModules(state: AppState, userText: string, contextPack: ContextPack): PromptModuleSelection[] {
  const modules = state.modules.filter((module) => !(module.kind === "safety_policy" && isAdultContentMode(state)));
  const normalizedText = userText.toLowerCase();
  const queryTerms = createSelectionTerms(userText);
  const selectionEvidence = getModuleSelectionEvidence(contextPack);

  return modules
    .filter((module) => module.enabled && module.tokenPolicy !== "disabled")
    .map((module): PromptModuleSelection | undefined => {
      const character = findModuleCharacter(state, module);
      const tagMatch = module.activationTags.some((tag) => textContainsSelectionPhrase(normalizedText, tag));
      const titleMatch = normalizedText.includes(module.title.toLowerCase());
      const characterNameMatch = character ? textContainsSelectionPhrase(normalizedText, character.name) : false;
      const moduleSignalScore = scoreModuleSignalMatch(module, character, queryTerms);
      const bodySignalScore = scoreModuleBodyQueryMatch(module, queryTerms);
      const bodySignalMatch = hasStrongBodyQueryMatch(bodySignalScore, queryTerms);
      const lexicalSignalMatch = moduleSignalScore > 0 || bodySignalMatch;
      const directSignalMatch = tagMatch || titleMatch || characterNameMatch || lexicalSignalMatch;
      const neuralMapEvidence = findSupportedModuleEvidence(
        module,
        selectionEvidence,
        queryTerms,
        directSignalMatch,
        contextPack.source,
        character
      );

      if (module.tokenPolicy === "always") {
        return {
          module,
          source: "always",
          reason: "항상 포함 토큰 정책",
          score: 1
        };
      }

      if (isFoundationPromptModule(module)) {
        return {
          module,
          source: "local",
          reason: "기반 프롬프트/세계관은 매 턴 유지",
          score: 0.96
        };
      }

      if (module.tokenPolicy === "manual" && (tagMatch || titleMatch || characterNameMatch)) {
        return {
          module,
          source: "manual",
          reason: "사용자 입력이 수동 모듈 제목/태그를 언급함",
          score: 0.88
        };
      }

      if (module.tokenPolicy === "manual") {
        return undefined;
      }

      if (neuralMapEvidence) {
        return {
          module,
          source: contextPack.source === "neuralmap" ? "neuralmap" : "local",
          reason: neuralMapEvidence.reason,
          score: neuralMapEvidence.score
        };
      }

      if (characterNameMatch) {
        return {
          module,
          source: "local",
          reason: "입력이 연결된 캐릭터 이름과 일치",
          score: 0.82
        };
      }

      if (tagMatch || titleMatch || moduleSignalScore > 0) {
        return {
          module,
          source: "local",
          reason: "입력이 모듈 제목/태그와 일치",
          score: Math.min(0.82, 0.7 + moduleSignalScore * 0.04)
        };
      }

      if (bodySignalMatch) {
        return {
          module,
          source: "local",
          reason: "입력이 모듈 본문 단서와 일치",
          score: Math.min(0.78, 0.62 + bodySignalScore * 0.04)
        };
      }

      return undefined;
    })
    .filter((selection): selection is PromptModuleSelection => Boolean(selection))
    .sort((a, b) => b.module.priority + b.score * 100 - (a.module.priority + a.score * 100))
    .slice(0, MAX_SELECTED_PROMPT_MODULES)
    .map((selection) => ({
      ...selection,
      module: compactPromptModuleForActiveContext(selection.module, userText, contextPack)
    }));
}

function isFoundationPromptModule(module: PromptModule): boolean {
  return module.kind === "main_prompt" || module.kind === "world_lore";
}

function getModuleSelectionEvidence(contextPack: ContextPack): ContextPack["evidence"] {
  return contextPack.moduleEvidence ?? contextPack.evidence;
}

function findSupportedModuleEvidence(
  module: PromptModule,
  evidence: ContextPack["evidence"],
  queryTerms: Set<string>,
  directSignalMatch: boolean,
  contextSource: ContextPack["source"],
  character: AppState["characters"][number] | undefined
): ContextPack["evidence"][number] | undefined {
  const matches = evidence.filter((item) => evidenceMatchesModule(item, module, character));
  if (matches.length === 0) {
    return undefined;
  }

  return matches.find((item) =>
    evidenceSupportsModuleActivation(item, module, queryTerms, directSignalMatch, contextSource, character)
  );
}

function evidenceMatchesModule(
  item: ContextPack["evidence"][number],
  module: PromptModule,
  character: AppState["characters"][number] | undefined
): boolean {
  const snippet = item.snippet.toLowerCase();
  const snippetTerms = createSelectionTerms(snippet);
  const moduleSignals = createModuleSignalTerms(module, character);
  return (
    item.nodeId === module.id ||
    item.nodeId.includes(module.id) ||
    Boolean(character && (item.nodeId.includes(character.id) || snippet.includes(character.name.toLowerCase()))) ||
    snippet.includes(module.title.toLowerCase()) ||
    moduleSignals.some((term) => snippetTerms.has(term) || snippet.includes(term))
  );
}

function evidenceSupportsModuleActivation(
  item: ContextPack["evidence"][number],
  module: PromptModule,
  queryTerms: Set<string>,
  directSignalMatch: boolean,
  contextSource: ContextPack["source"],
  character: AppState["characters"][number] | undefined
): boolean {
  if (directSignalMatch) {
    return true;
  }

  if (contextSource === "neuralmap") {
    return true;
  }

  const isSelfDocument = item.nodeId === module.id || item.nodeId.includes(module.id);
  if (isSelfDocument) {
    return false;
  }

  const snippetTerms = createSelectionTerms(item.snippet);
  return createModuleSignalTerms(module, character).some((term) => snippetTerms.has(term) && queryTerms.has(term));
}

function scoreModuleSignalMatch(
  module: PromptModule,
  character: AppState["characters"][number] | undefined,
  queryTerms: Set<string>
): number {
  return createModuleSignalTerms(module, character).filter((term) => queryTerms.has(term)).length;
}

function scoreModuleBodyQueryMatch(module: PromptModule, queryTerms: Set<string>): number {
  if (queryTerms.size === 0) {
    return 0;
  }

  const bodyTerms = createSelectionTerms(`${module.title}\n${module.body}`);
  return Array.from(queryTerms).filter((term) => bodyTerms.has(term)).length;
}

function hasStrongBodyQueryMatch(score: number, queryTerms: Set<string>): boolean {
  if (score <= 0) {
    return false;
  }

  const minimumScore = queryTerms.size <= 1 ? 1 : 2;
  return score >= minimumScore;
}

function createModuleSignalTerms(
  module: PromptModule,
  character?: AppState["characters"][number]
): string[] {
  const signalSources = [
    ...module.activationTags,
    ...module.title.split(/[^\p{L}\p{N}_-]+/u),
    character?.name,
    character?.role
  ].filter((term): term is string => Boolean(term));

  return uniqueStrings(signalSources.flatMap((term) => Array.from(createSelectionTerms(term))));
}

const SELECTION_STOP_TERMS = new Set([
  "ask",
  "about",
  "continue",
  "what",
  "that",
  "this",
  "from",
  "only",
  "necessary",
  "active",
  "context",
  "next",
  "move",
  "summary",
  "summarize",
  "section",
  "archive",
  "unrelated",
  "distractor",
  "background",
  "noise",
  "policy",
  "minor",
  "procedural",
  "notes",
  "scene",
  "scenes",
  "계속",
  "진행",
  "시뮬레이션",
  "캐릭터",
  "인물",
  "장면",
  "현재",
  "다음",
  "대화",
  "응답",
  "상황",
  "행동",
  "선택",
  "설정",
  "모듈",
  "프롬프트",
  "시스템",
  "규칙",
  "기본",
  "주요",
  "기타",
  "필요",
  "부분",
  "그냥",
  "한다",
  "했다",
  "된다",
  "있는",
  "없는",
  "그리고",
  "그러나",
  "하지만",
  "대화",
  "대화한다",
  "진행한다",
  "확인한다",
  "이야기한다",
  "말한다",
  "본다",
  "보여준다"
]);

function createSelectionTerms(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^\p{L}\p{N}_-]+/u)
      .flatMap((term) => createSelectionTermVariants(term.trim()))
      .filter(isUsefulSelectionTerm)
  );
}

function createSelectionTermVariants(term: string): string[] {
  if (!term) {
    return [];
  }

  const stripped = stripKoreanParticle(term);
  return stripped === term ? [term] : [term, stripped];
}

function stripKoreanParticle(term: string): string {
  if (!/[\u3131-\uD79D]/u.test(term)) {
    return term;
  }

  const suffixes = [
    "에게서",
    "으로써",
    "으로서",
    "에게",
    "에서",
    "까지",
    "부터",
    "하고",
    "처럼",
    "만큼",
    "보다",
    "으로",
    "로",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "와",
    "과",
    "의",
    "도",
    "만",
    "에"
  ];

  const suffix = suffixes.find((candidate) => term.endsWith(candidate) && term.length - candidate.length >= 2);
  return suffix ? term.slice(0, -suffix.length) : term;
}

function isUsefulSelectionTerm(term: string): boolean {
  if (SELECTION_STOP_TERMS.has(term)) {
    return false;
  }

  const hasHangul = /[\u3131-\uD79D]/u.test(term);
  return hasHangul ? term.length >= 2 : term.length >= 3;
}

function textContainsSelectionPhrase(normalizedText: string, phrase: string | undefined): boolean {
  const normalizedPhrase = phrase?.trim().toLowerCase();
  return Boolean(normalizedPhrase && isUsefulSelectionPhrase(normalizedPhrase) && normalizedText.includes(normalizedPhrase));
}

function isUsefulSelectionPhrase(phrase: string): boolean {
  return phrase.length >= 3 || /[\u3131-\uD79D]{2,}/u.test(phrase);
}

function findModuleCharacter(state: AppState, module: PromptModule): AppState["characters"][number] | undefined {
  if (!module.characterId) {
    return undefined;
  }

  return state.characters.find((character) => character.id === module.characterId);
}

function compactPromptModuleForActiveContext(module: PromptModule, userText: string, contextPack: ContextPack): PromptModule {
  if (module.body.length <= MAX_ACTIVE_MODULE_BODY_CHARS) {
    return module;
  }

  const terms = createModuleExcerptTerms(module, userText, contextPack);
  const windows = findModuleExcerptWindows(module.body, terms);
  const excerptBody =
    windows.length > 0
      ? windows.join("\n\n[...]\n\n")
      : [
          module.body.slice(0, Math.floor(MAX_ACTIVE_MODULE_BODY_CHARS * 0.7)),
          module.body.slice(-Math.floor(MAX_ACTIVE_MODULE_BODY_CHARS * 0.25))
        ].join("\n\n[...]\n\n");

  return {
    ...module,
    body: [
      `[excerpted long prompt module: original ${module.body.length} chars, policy ${module.tokenPolicy}]`,
      excerptBody.slice(0, MAX_ACTIVE_MODULE_BODY_CHARS)
    ].join("\n")
  };
}

function createModuleExcerptTerms(module: PromptModule, userText: string, contextPack: ContextPack): string[] {
  const userTerms = userText
    .toLowerCase()
    .split(/[^\p{L}\p{N}_-]+/u)
    .filter(isUsefulSelectionTerm);
  const evidenceTerms = contextPack.evidence
    .flatMap((item) => item.snippet.toLowerCase().split(/[^\p{L}\p{N}_-]+/u))
    .filter(isUsefulSelectionTerm)
    .slice(0, 32);

  return uniqueStrings([
    ...module.activationTags.map((tag) => tag.toLowerCase()),
    ...module.title.toLowerCase().split(/[^\p{L}\p{N}_-]+/u),
    ...userTerms,
    ...evidenceTerms
  ]).filter(isUsefulSelectionTerm);
}

function findModuleExcerptWindows(body: string, terms: string[]): string[] {
  const lowerBody = body.toLowerCase();
  const ranges: Array<{ start: number; end: number }> = [];

  for (const term of terms) {
    const index = lowerBody.indexOf(term);
    if (index < 0) {
      continue;
    }

    const start = Math.max(0, index - Math.floor(MODULE_EXCERPT_WINDOW_CHARS / 2));
    const end = Math.min(body.length, start + MODULE_EXCERPT_WINDOW_CHARS);
    ranges.push({ start, end });
    if (ranges.length >= MODULE_EXCERPT_MAX_WINDOWS) {
      break;
    }
  }

  return mergeExcerptRanges(ranges)
    .map((range) => body.slice(range.start, range.end).trim())
    .filter(Boolean)
    .slice(0, MODULE_EXCERPT_MAX_WINDOWS);
}

function mergeExcerptRanges(ranges: Array<{ start: number; end: number }>): Array<{ start: number; end: number }> {
  const sorted = [...ranges].sort((a, b) => a.start - b.start);
  const merged: Array<{ start: number; end: number }> = [];

  for (const range of sorted) {
    const previous = merged.at(-1);
    if (!previous || range.start > previous.end + 80) {
      merged.push({ ...range });
    } else {
      previous.end = Math.max(previous.end, range.end);
    }
  }

  return merged;
}

function createPromptModuleUsages(
  state: AppState,
  turnId: string,
  selections: PromptModuleSelection[],
  createdAt: string
): PromptModuleUsage[] {
  return selections.map((selection) => ({
    id: createId("moduse"),
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    turnId,
    moduleId: selection.module.id,
    moduleTitle: selection.module.title,
    tokenPolicy: selection.module.tokenPolicy,
    source: selection.source,
    reason: selection.reason,
    score: selection.score,
    createdAt
  }));
}

function createAssistantContent(
  state: AppState,
  userText: string,
  modules: PromptModule[],
  evidence: string[]
): string {
  const character = resolvePersonaCharacter(state) ?? getCurrentSceneCharacters(state, userText)[0];
  const characterState = [character?.relationship, character?.currentMood].filter(Boolean).join(", ");
  const activeModules = modules.filter((module) => module.kind !== "image_prompt_profile" && module.kind !== "safety_policy");
  const activeGuidance = activeModules.map((module) => module.title).slice(0, 4).join(", ");
  const memoryHint = evidence[0] ?? "아직 강한 장기 기억은 없다.";
  const personaHint = createPersonaNarrationHint(state);
  const asksForImage = /그려|보여|이미지/u.test(userText);
  const fallbackNotice = createLocalFallbackNotice(activeGuidance);
  const statusBlock = createLocalFallbackStatusBlock(state, userText, activeModules);
  const actionSummary = userText.length > 80 ? `${userText.slice(0, 80)}...` : userText;

  if (asksForImage) {
    const sceneLine = character
      ? `${character.name}는 ${personaHint}현재 장면의 분위기와 인물의 상태를 확인한다.`
      : "현재 장면은 등록된 관계도 캐릭터를 자동으로 호출하지 않고, 최근 문맥에 드러난 인물과 상황만 확인한다.";
    return [
      fallbackNotice,
      `${sceneLine} 입력된 요청은 이미지 단서로 분류되지만, 실제 LLM 응답이 없어 장면 해석은 로컬 대체 흐름으로만 남긴다.`,
      `현재 입력: ${actionSummary}`,
      `참조 단서: ${memoryHint}`,
      statusBlock
    ]
      .filter(Boolean)
      .join("\n\n");
  }

  const continuityLine = character
    ? `${character.name}는 ${personaHint}장면의 연속성을 붙잡는다.`
    : "현재 장면은 최근 대화에 실제로 놓인 인물과 행동만 붙잡는다.";

  return [
    fallbackNotice,
    `현재 입력 "${actionSummary}"은 이번 턴의 사용자 행동으로 접수되었다. 다만 실제 LLM 응답이 생성되지 않아, 메인 프롬프트와 선택 모듈을 완전히 해석한 시뮬레이션 진행은 보류된다.`,
    `${continuityLine} ${characterState ? `현재 상태: ${characterState}. ` : ""}${activeGuidance ? `선택된 규칙: ${activeGuidance}. ` : ""}${memoryHint}`,
    "API 키, 모델, 또는 JSON sidecar 형식을 확인한 뒤 다시 입력하면 이 턴은 실제 LLM 규칙 적용 흐름으로 이어질 수 있다.",
    statusBlock
  ]
    .filter(Boolean)
    .join("\n\n");
}

function createLocalFallbackNotice(activeGuidance: string): string {
  return `::status[LLM fallback: 로컬 대체 진행입니다. 적용 대상 모듈: ${activeGuidance || "없음"}]`;
}

function createLocalFallbackStatusBlock(state: AppState, userText: string, modules: PromptModule[]): string {
  const statusRequired = modules.some((module) => /상태창|상태\s*요약|status|hud|출력\s*형식/iu.test(`${module.title}\n${module.body}`));
  if (!statusRequired) {
    return "";
  }

  const compactInput = userText.replace(/\s+/gu, " ").trim().slice(0, 90);
  return [
    "```status",
    `LLM: fallback`,
    `Simulation: ${state.simulation.title}`,
    `User action: ${compactInput || "(empty)"}`,
    `Selected modules: ${modules.map((module) => module.title).slice(0, 6).join(", ") || "(none)"}`,
    "State update: 실제 LLM 응답이 없어 수치/관계/상태 갱신은 적용하지 않음",
    "```"
  ].join("\n");
}

function createPersonaAwareUserText(state: AppState, userText: string): string {
  const personaContext = createPersonaContextLines(state);
  if (!personaContext) {
    return userText;
  }

  const heading = state.userPersona.source === "character" ? "사용자 조작 캐릭터 시점" : "사용자 페르소나";
  return `${userText}\n\n${heading}:\n${personaContext}`;
}

function createTurnRetrievalQuery(state: AppState, personaAwareUserText: string, rawUserText: string): string {
  const structuredMemory = createStructuredContextSummary(state, { maxEvents: 6, maxStates: 8, currentText: rawUserText });
  const immediateContinuity = createImmediateContinuityRetrievalAnchor(state);
  const settingAnchor = createRetrievalSettingAnchor(state, rawUserText);
  const recentTranscript = createRetrievalTranscript(state);

  return [
    settingAnchor ? `시뮬레이션 설정/상황 앵커:\n${settingAnchor}` : undefined,
    immediateContinuity ? `직전 출력 연속성 앵커:\n${immediateContinuity}` : undefined,
    `구조화된 현재 메모리:\n${structuredMemory}`,
    recentTranscript ? `최근 장면 표면 맥락:\n${recentTranscript}` : undefined,
    `현재 사용자 입력:\n${personaAwareUserText}`
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function createImmediateContinuityRetrievalAnchor(state: AppState): string {
  const latestAssistant = findLatestMessageByRole(state, "assistant");
  const latestUser = findLatestMessageByRole(state, "user");
  if (!latestAssistant && !latestUser) {
    return "";
  }

  return [
    "NeuralMap 검색은 오래된 기억보다 최근 assistant 출력 끝부분과 현재 장면을 우선해야 한다.",
    latestAssistant
      ? `최근 assistant 출력 끝부분:\n${createRetrievalExcerpt(latestAssistant.content, RETRIEVAL_LATEST_ASSISTANT_CHARS, "tail")}`
      : undefined,
    latestUser
      ? `직전 사용자 행동:\n${createRetrievalExcerpt(latestUser.content, RETRIEVAL_RECENT_MESSAGE_CHARS, "balanced")}`
      : undefined
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function createRetrievalSettingAnchor(state: AppState, rawUserText: string): string {
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state, rawUserText));
  const characters = state.characters
    .filter((character) => activeCharacterIds.has(character.id))
    .slice(0, 8)
    .map((character) =>
      [
        character.name,
        character.role,
        character.summary,
        character.relationship,
        character.currentMood
      ]
        .filter(Boolean)
        .join(" / ")
    )
    .filter(Boolean)
    .join("\n");
  const settingModules = state.modules
    .filter(
      (module) =>
        module.enabled &&
        module.tokenPolicy !== "disabled" &&
        (
          ["main_prompt", "world_lore", "scene_rule"].includes(module.kind) ||
          (module.kind === "character_prompt" && Boolean(module.characterId && activeCharacterIds.has(module.characterId)))
        )
    )
    .sort((a, b) => b.priority - a.priority)
    .slice(0, 8)
    .map((module) =>
      [
        `- ${module.title} (${module.kind}, ${module.tokenPolicy}, priority ${module.priority})`,
        module.activationTags.length > 0 ? `tags: ${module.activationTags.join(", ")}` : undefined,
        createRetrievalExcerpt(module.body, RETRIEVAL_SETTING_MODULE_CHARS, "balanced")
      ]
        .filter((item): item is string => Boolean(item))
        .join(" | ")
    )
    .join("\n");

  return [
    `${state.simulation.title}: ${state.simulation.description}`,
    characters ? `최근 문맥상 장면에 언급된 캐릭터 설정:\n${characters}` : "최근 문맥상 장면에 언급된 등록 캐릭터 없음. 관계도/로스터만으로 캐릭터를 검색하거나 호출하지 않는다.",
    settingModules ? `우선 설정/상황 모듈:\n${settingModules}` : undefined
  ]
    .filter((item): item is string => Boolean(item))
    .join("\n\n");
}

function createRetrievalTranscript(state: AppState): string {
  const recentMessages = state.messages.slice(-4);
  const latestAssistantId = [...recentMessages].reverse().find((message) => message.role === "assistant")?.id;
  return recentMessages
    .map((message) => {
      const isLatestAssistant = message.id === latestAssistantId;
      const content = createRetrievalExcerpt(
        message.content,
        isLatestAssistant ? RETRIEVAL_LATEST_ASSISTANT_CHARS : RETRIEVAL_RECENT_MESSAGE_CHARS,
        isLatestAssistant ? "tail" : "balanced"
      );
      const roleLabel = isLatestAssistant ? `${message.role} (latest ending)` : message.role;
      return content ? `${roleLabel}: ${content}` : "";
    })
    .filter(Boolean)
    .join("\n");
}

function findLatestMessageByRole(state: AppState, role: ChatMessage["role"]): ChatMessage | undefined {
  return [...state.messages].reverse().find((message) => message.role === role);
}

function createRetrievalExcerpt(value: string, maxChars: number, mode: "balanced" | "tail"): string {
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

function createPersonaContextLines(state: AppState): string {
  const persona = state.userPersona;
  if (!persona?.enabled) {
    return "";
  }

  const character = resolvePersonaCharacter(state);
  if (persona.source === "character" && character) {
    return [
      "입력 해석: 사용자의 입력은 아래 기존 캐릭터의 행동/대사/선택이다.",
      `캐릭터 ID: ${character.id}`,
      `이름/호칭: ${character.name}`,
      character.role.trim() ? `역할: ${character.role.trim()}` : undefined,
      character.summary.trim() ? `배경: ${character.summary.trim()}` : undefined,
      character.relationship.trim() ? `관계: ${character.relationship.trim()}` : undefined,
      character.currentMood.trim() ? `현재 상태: ${character.currentMood.trim()}` : undefined,
      persona.goals.trim() ? `플레이 목표: ${persona.goals.trim()}` : undefined,
      persona.style.trim() ? `입력 방식: ${persona.style.trim()}` : undefined,
      persona.boundaries.trim() ? `경계: ${persona.boundaries.trim()}` : undefined
    ]
      .filter((line): line is string => Boolean(line))
      .join("\n");
  }

  return [
    persona.name.trim() ? `이름/호칭: ${persona.name.trim()}` : undefined,
    persona.role.trim() ? `역할: ${persona.role.trim()}` : undefined,
    persona.background.trim() ? `배경: ${persona.background.trim()}` : undefined,
    persona.goals.trim() ? `목표: ${persona.goals.trim()}` : undefined,
    persona.style.trim() ? `행동/말투: ${persona.style.trim()}` : undefined,
    persona.boundaries.trim() ? `경계: ${persona.boundaries.trim()}` : undefined
  ]
    .filter((line): line is string => Boolean(line))
    .join("\n");
}

function createPersonaNarrationHint(state: AppState): string {
  const persona = state.userPersona;
  if (!persona?.enabled) {
    return "";
  }

  const character = resolvePersonaCharacter(state);
  if (character) {
    return "자신의 시점에서 ";
  }

  const name = persona.name.trim() || "당신";
  const role = persona.role.trim();
  if (!role || role.includes(name)) {
    return `${name}의 입장을 살피듯 `;
  }

  return `${name}의 ${role} 입장을 살피듯 `;
}

function resolvePersonaCharacter(state: AppState): AppState["characters"][number] | undefined {
  const persona = state.userPersona;
  if (!persona?.enabled || persona.source !== "character" || !persona.characterId) {
    return undefined;
  }

  return state.characters.find((character) => character.id === persona.characterId);
}

function createSidecarTrace(
  state: AppState,
  turnId: string,
  assistantGeneration: Awaited<ReturnType<typeof generateAssistantText>>
): SidecarTrace {
  return {
    id: createId("sidecar"),
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    turnId,
    source: assistantGeneration.source,
    status: assistantGeneration.sidecarStatus,
    errors: assistantGeneration.sidecarErrors,
    requestPreview: assistantGeneration.requestPreview,
    rawPreview: assistantGeneration.rawPreview,
    createdAt: new Date().toISOString()
  };
}

function shouldCurateAssistantMemory(assistantGeneration: Awaited<ReturnType<typeof generateAssistantText>>): boolean {
  return assistantGeneration.source === "llm" || assistantGeneration.sidecarStatus === "failed";
}

function createPendingImageCue(
  state: AppState,
  userText: string,
  assistantContent: string,
  manualImage = false
): ImageCue {
  const currentTurnText = `${userText}\n${assistantContent}`;
  return {
    shouldGenerate: manualImage,
    reason: manualImage ? "수동 이미지 생성 요청 대기" : "메인 LLM 이미지 cue 대기",
    characters: findMentionedCharacterIds(state, currentTurnText),
    tags: [],
    scene: "current simulation scene",
    suppressionReason: manualImage ? undefined : "메인 LLM 이미지 cue가 아직 적용되지 않음",
    visualContext: assistantContent.slice(0, 220)
  };
}

function planImageCue(
  state: AppState,
  userText: string,
  draft: AssistantImageCueDraft,
  assistantContent: string,
  options: { allowLocalTrigger?: boolean; includeLocalVisualTags?: boolean; trustLlmCharacterScope?: boolean; manualImage?: boolean } = {}
): ImageCue {
  const currentTurnText = `${userText}\n${assistantContent}`;
  const recentContext = createRecentImageCueContext(state);
  const combinedText = `${recentContext}\n${currentTurnText}`;
  const includeLocalVisualTags = options.includeLocalVisualTags ?? true;
  const visualInferenceText = includeLocalVisualTags
    ? createFocusedImageCueInferenceText(currentTurnText, recentContext, userText, draft)
    : currentTurnText;
  const focusedCueText = createFocusedImageCueText(currentTurnText, draft);
  const characterScopeText = createImageCueCharacterScopeText(currentTurnText, draft);
  const allowLocalTrigger = options.allowLocalTrigger ?? true;
  const rawResolvedCharacters = resolveCharacterIds(state, draft.characters);
  const mentionedCharacters = uniqueStrings([
    ...findMentionedCharacterIds(state, focusedCueText),
    ...findPersonaActionCharacterIds(state, userText, focusedCueText, characterScopeText)
  ]);
  const resolvedCharacters = filterImageCueCharactersForCurrentTurn(
    state,
    rawResolvedCharacters,
    mentionedCharacters,
    currentTurnText,
    characterScopeText,
    Boolean(options.trustLlmCharacterScope)
  );
  const alwaysVisual = state.modules.some(
    (module) =>
      module.enabled &&
      module.kind === "image_prompt_profile" &&
      /항상|모든 assistant|기본 이미지/u.test(module.body)
  );
  const hasVisualTrigger = allowLocalTrigger && IMAGE_PATTERN.test(combinedText);
  let shouldGenerate =
    draft.shouldGenerate ||
    Boolean(options.manualImage) ||
    (allowLocalTrigger && alwaysVisual) ||
    hasVisualTrigger;
  const scene = resolveImageScene(draft.scene);
  const defaultCharacters = shouldDropCharactersForExternalCue(
    state,
    uniqueStrings([...rawResolvedCharacters, ...mentionedCharacters]),
    characterScopeText
  )
    ? []
    : mentionedCharacters;
  const inferredPlannerCharacters =
    resolvedCharacters.length === 0 && defaultCharacters.length === 0 && isDedicatedPlannerDraft(draft)
      ? inferDedicatedPlannerCharacterIds(state, focusedCueText, characterScopeText, draft)
      : [];
  const characterIds = resolvedCharacters.length > 0 ? resolvedCharacters : defaultCharacters.length > 0 ? defaultCharacters : inferredPlannerCharacters;
  const localCueTags = includeLocalVisualTags ? createLocalImageCueTags(visualInferenceText, state, draft) : [];
  const plannerSubjectTags = isDedicatedPlannerDraft(draft) ? createPlannerSubjectCountTags(state, characterIds, draft.tags) : [];
  const tags = uniqueStrings([
    ...filterRosterNameCueTags(
      state,
      sanitizeImageCueTags([...plannerSubjectTags, ...draft.tags, ...localCueTags])
    )
  ].filter((tag): tag is string => Boolean(tag)));
  const stateBackedTags = createCurrentImageCueStateTags(state, characterIds);
  const visualContext = createImageCueVisualContext(
    state,
    visualInferenceText,
    scene,
    characterIds,
    draft.visualContext,
    includeLocalVisualTags,
    draft
  );
  const concreteVisualContextTags = sanitizeImageCueTags(visualContext.split(/[,;\n|]+/u))
    .filter((tag) => !isGenericImageSceneTag(tag));
  const hasConcreteLlmVisualInput =
    tags.length > 0 ||
    concreteVisualContextTags.length > 0 ||
    stateBackedTags.length > 0;
  if (shouldGenerate && !options.manualImage && !hasConcreteLlmVisualInput) {
    shouldGenerate = false;
  }

  return {
    shouldGenerate,
    reason: draft.reason || (shouldGenerate ? "메인 LLM이 현재 장면을 이미지 cue로 선택함" : "메인 LLM이 이미지 생성을 생략함"),
    characters: characterIds,
    tags,
    scene,
    suppressionReason: shouldGenerate
      ? draft.suppressionReason
      : draft.suppressionReason ?? "LLM image cue에 사용할 수 있는 태그/상태가 부족함",
    visualContext
  };
}

function isDedicatedPlannerDraft(draft: AssistantImageCueDraft): boolean {
  return (draft as ImageRuleBackedCueDraft).plannerSource === "dedicated_image_planner";
}

function inferDedicatedPlannerCharacterIds(
  state: AppState,
  focusedCueText: string,
  characterScopeText: string,
  draft: AssistantImageCueDraft
): string[] {
  const explicitCueText = [focusedCueText, characterScopeText, draft.scene, draft.visualContext, draft.tags.join(", ")]
    .filter(Boolean)
    .join("\n");
  if (!hasVisibleCharacterCueSignal(explicitCueText)) {
    return [];
  }

  const activeIds = inferCurrentSceneCharacterIds(state, explicitCueText);
  return activeIds.length === 1 ? activeIds : [];
}

function hasVisibleCharacterCueSignal(text: string): boolean {
  return /\b(?:\d+(?:girls?|boys?|others?)|girl|boy|woman|man|solo|upper body|cowboy shot|full body|portrait|face focus|eye focus|open mouth|smile|standing|sitting|walking|holding|grabbing|touching|microphone|hand|hands|body focus)\b|그녀|그|캐릭터|인물|얼굴|표정|손|몸|말하|속삭|걷|잡|쥐|들고/u.test(
    text
  );
}

function createPlannerSubjectCountTags(state: AppState, characterIds: string[], existingTags: string[]): string[] {
  if (existingTags.some((tag) => isImageSubjectCountTag(tag)) || characterIds.length === 0 || characterIds.length > 2) {
    return [];
  }

  const subjectKinds = characterIds.map((characterId) => detectCharacterSubjectKind(state, characterId));
  if (characterIds.length === 1) {
    return [`1${subjectKinds[0]}`];
  }
  const [first, second] = subjectKinds;
  if (first && first === second) {
    return [`2${first}s`];
  }
  return [];
}

function isImageSubjectCountTag(tag: string): boolean {
  return /^\d+(?:girls?|boys?|others?)$/iu.test(tag.trim());
}

function detectCharacterSubjectKind(state: AppState, characterId: string): "girl" | "boy" | "other" {
  const character = state.characters.find((candidate) => candidate.id === characterId);
  const visualProfile = state.visualProfiles.find((candidate) => candidate.characterId === characterId);
  const text = [
    character?.role,
    character?.summary,
    visualProfile?.displayName,
    visualProfile?.positivePrompt
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (/\b(?:girl|female|woman|schoolgirl|breasts?)\b|여학생|소녀|여자|여성/u.test(text)) {
    return "girl";
  }
  if (/\b(?:boy|male|man|schoolboy)\b|남학생|소년|남자|남성/u.test(text)) {
    return "boy";
  }
  return "other";
}

function resolveImageScene(draftScene: string | undefined): string {
  const normalizedDraftScene = normalizeImageSceneLabel(draftScene);
  if (normalizedDraftScene) {
    return normalizedDraftScene;
  }

  return "current simulation scene";
}

function createFocusedImageCueInferenceText(
  currentTurnText: string,
  recentContext: string,
  userText: string,
  draft: AssistantImageCueDraft
): string {
  const anchoredText = createAnchorFocusedImageCueText(currentTurnText, draft);
  const cueText = createFocusedImageCueText(currentTurnText, draft);

  if (anchoredText) {
    return cueText === currentTurnText ? anchoredText : `${anchoredText}\n${cueText}`;
  }

  if (recentContext && isDeicticImageRequest(userText)) {
    return `${recentContext}\n${cueText === currentTurnText ? currentTurnText : cueText}`;
  }

  if (hasSpecificImageCueDraft(draft) || hasCurrentTurnSceneSignal(currentTurnText)) {
    return cueText;
  }

  return currentTurnText;
}

function hasSpecificImageCueDraft(draft: AssistantImageCueDraft): boolean {
  return (
    sanitizeImageCueTags(draft.tags).length > 0 ||
    Boolean(draft.visualContext?.trim()) ||
    Boolean(normalizeImageSceneLabel(draft.scene))
  );
}

function createAnchorFocusedImageCueText(currentTurnText: string, draft: AssistantImageCueDraft): string | undefined {
  const anchor = cleanImageAnchorText(draft.anchorText ?? "");
  if (!anchor) {
    return undefined;
  }

  const lines = currentTurnText
    .split(/\n+/u)
    .map(cleanImageAnchorText)
    .filter(Boolean);
  const normalizedAnchor = normalizeAnchorForComparison(anchor);
  const directIndex = lines.findIndex((line) => normalizeAnchorForComparison(line).includes(normalizedAnchor));
  if (directIndex >= 0) {
    return selectAnchorNeighborLines(lines, directIndex).join("\n");
  }

  const anchorTerms = createAnchorSearchTerms(anchor);
  let bestIndex = -1;
  let bestScore = 0;
  lines.forEach((line, index) => {
    const normalizedLine = normalizeAnchorForComparison(line);
    const score = anchorTerms.reduce((sum, term) => (normalizedLine.includes(term) ? sum + Math.min(8, term.length) : sum), 0);
    if (score > bestScore) {
      bestIndex = index;
      bestScore = score;
    }
  });

  return bestIndex >= 0 && bestScore >= 4 ? selectAnchorNeighborLines(lines, bestIndex).join("\n") : anchor;
}

function selectAnchorNeighborLines(lines: string[], index: number): string[] {
  const start = Math.max(0, index - 1);
  const end = Math.min(lines.length, index + 2);
  return lines.slice(start, end);
}

function createAnchorSearchTerms(value: string): string[] {
  return uniqueStrings(
    value
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]+/gu, " ")
      .split(/\s+/u)
      .map((term) => term.trim())
      .filter((term) => term.length >= 2)
      .slice(0, 16)
  );
}

function hasCurrentTurnSceneSignal(text: string): boolean {
  void text;
  return false;
}

function isDeicticImageRequest(text: string): boolean {
  return /(?:이|현재|방금|지금)\s*(?:장면|상황|모습|컷)|this\s+scene|current\s+scene|show\s+(?:this|current)/iu.test(text);
}

function normalizeImageSceneLabel(value: string | undefined): string | undefined {
  const normalized = value
    ?.trim()
    .replace(/[._-]+/gu, " ")
    .replace(/[^\p{L}\p{N} ]+/gu, " ")
    .replace(/\s+/gu, " ")
    .toLowerCase();
  if (!normalized || /[\u3131-\uD79D]/u.test(normalized)) {
    return undefined;
  }

  if (
    [
      "current scene",
      "current simulation scene",
      "generated scene",
      "simulation scene",
      "scene",
      "main action",
      "character close up",
      "wide context",
      "interaction detail",
      "filming set",
      "getting up",
      "props"
    ].includes(normalized) ||
    /\b(?:chapter|episode|cut|shot)\s+\d+\b/iu.test(normalized)
  ) {
    return undefined;
  }

  return normalized.split(" ").length <= 4 ? normalized : undefined;
}

function createLocalImageCueTags(text: string, state?: AppState, draft?: Pick<AssistantImageCueDraft, "kind" | "cueType">): string[] {
  void state;
  const kind = normalizeImageCueKind(draft?.kind ?? draft?.cueType);
  return filterContradictoryLocalImageTags(uniqueStrings([
    ...selectCueFramingTags(kind, text),
    ...selectCueEnvironmentTags(text),
    ...selectCueWeatherAndLightingTags(text),
    ...selectCuePoseActionTags(text),
    ...selectCueBodyDetailTags(kind, text),
    ...selectCuePropTags(text),
    ...selectCueExpressionTags(kind, text)
  ])).slice(0, 18);
}

function normalizeImageCueKind(kind: string | undefined): string {
  const normalized = kind?.trim().toLowerCase().replace(/[\s-]+/gu, "_");
  if (normalized === "context") {
    return "scene";
  }
  return normalized ?? "";
}

function selectCueFramingTags(kind: string, text: string): string[] {
  const tags: string[] = [];
  if (/\b(?:pov|first person)\b|1인칭/u.test(text)) {
    tags.push("pov");
  }
  if (/측면|옆모습|\b(?:from side|side view)\b/iu.test(text)) {
    tags.push("from side");
  }
  if (/뒤에서|뒷모습|\b(?:from behind|back view)\b/iu.test(text)) {
    tags.push("from behind");
  }
  if (/낮은\s*각도|로우앵글|\b(?:low angle|from below)\b/iu.test(text)) {
    tags.push("low angle");
  }
  if (/높은\s*각도|하이앵글|\b(?:high angle|from above)\b/iu.test(text)) {
    tags.push("high angle");
  }
  if (kind === "dialogue_face") {
    tags.push("close-up", "face focus");
  } else if (kind === "body_detail") {
    tags.push("close-up", "body focus");
  } else if (kind === "scene") {
    tags.push("wide shot");
  } else if (kind === "action" || kind === "interaction") {
    tags.push("cowboy shot");
  }
  return tags;
}

function selectCueEnvironmentTags(text: string): string[] {
  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /archive|library|bookshelf|기록\s*보관|도서관|책장|서가/iu, tags: ["archive library", "bookshelf"] },
    { pattern: /classroom|school|교실|학교/iu, tags: ["classroom", "indoors"] },
    { pattern: /stage|performance|spotlight|무대|공연|스포트라이트/iu, tags: ["stage", "stage lights"] },
    { pattern: /hallway|corridor|복도/iu, tags: ["hallway", "indoors"] },
    { pattern: /bedroom|침실/iu, tags: ["bedroom", "indoors"] },
    { pattern: /kitchen|주방|부엌/iu, tags: ["kitchen", "indoors"] },
    { pattern: /street|alley|거리|골목/iu, tags: ["street", "outdoors"] },
    { pattern: /cafe|카페/iu, tags: ["cafe", "indoors"] },
    { pattern: /hospital|clinic|병원|진료실/iu, tags: ["hospital", "indoors"] },
    { pattern: /lab|laboratory|실험실|연구실/iu, tags: ["laboratory", "indoors"] },
    { pattern: /forest|woods|숲/iu, tags: ["forest", "outdoors"] },
    { pattern: /beach|바닷가|해변/iu, tags: ["beach", "outdoors"] },
    { pattern: /battlefield|전장/iu, tags: ["battlefield", "outdoors"] },
    { pattern: /room|방\b/iu, tags: ["room", "indoors"] }
  ];
  const matches = rules
    .map((rule) => {
      const match = rule.pattern.exec(text);
      return match ? { tags: rule.tags, index: match.index } : undefined;
    })
    .filter((match): match is { tags: string[]; index: number } => Boolean(match))
    .sort((a, b) => a.index - b.index);
  return matches[0]?.tags ?? [];
}

function selectCueWeatherAndLightingTags(text: string): string[] {
  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /rain|rainy|storm|비가|빗물|폭우/iu, tags: ["rain"] },
    { pattern: /snow|눈이|눈발/iu, tags: ["snow"] },
    { pattern: /night|밤|야간/iu, tags: ["night"] },
    { pattern: /daylight|sunlight|낮|햇빛/iu, tags: ["daylight"] },
    { pattern: /spotlight|스포트라이트/iu, tags: ["spotlight"] },
    { pattern: /neon|네온/iu, tags: ["neon lights"] },
    { pattern: /backlight|역광/iu, tags: ["backlighting"] }
  ];
  return uniqueStrings(rules.flatMap((rule) => (rule.pattern.test(text) ? rule.tags : [])));
}

function selectCuePoseActionTags(text: string): string[] {
  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /standing up|stands? up|일어나|몸을 일으/iu, tags: ["standing", "chair"] },
    { pattern: /\bstanding\b|서\s*있|선 채/iu, tags: ["standing"] },
    { pattern: /\bsitting\b|앉/iu, tags: ["sitting"] },
    { pattern: /\bkneeling\b|무릎/iu, tags: ["kneeling"] },
    { pattern: /\blying\b|누워/iu, tags: ["lying"] },
    { pattern: /\bleaning\b|기대|숙이/iu, tags: ["leaning forward"] },
    { pattern: /walks?|steps?|걸음|다가/iu, tags: ["walking"] },
    { pattern: /runs?|달리/iu, tags: ["running"] },
    { pattern: /danc|춤|안무/iu, tags: ["dancing"] },
    { pattern: /fight|combat|전투|싸움/iu, tags: ["dynamic action"] },
    { pattern: /reaches?|손을\s*뻗|팔을\s*뻗|뻗어/iu, tags: ["reaching out"] },
    { pattern: /raises?\s+(?:her|his|their)?\s*(?:hand|arm)|hand\s+up|arm\s+up|손을\s*들|팔을\s*올/iu, tags: ["arm up", "hand up"] },
    { pattern: /grabs?|움켜쥐|붙잡|잡아/iu, tags: ["grabbing"] },
    { pattern: /holds?|쥐|쥔|쥐고|들고|안고/iu, tags: ["holding"] },
    { pattern: /touch(?:es|ing)?|손을\s*대|만지/iu, tags: ["touching"] }
  ];
  return uniqueStrings(rules.flatMap((rule) => (rule.pattern.test(text) ? rule.tags : [])));
}

function selectCueBodyDetailTags(kind: string, text: string): string[] {
  const tags: string[] = [];
  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /손목|wrist/iu, tags: ["hands", "wrist grab"] },
    { pattern: /손|hand/iu, tags: ["hands"] },
    { pattern: /팔|arm/iu, tags: ["arm focus"] },
    { pattern: /어깨|shoulder/iu, tags: ["shoulder"] },
    { pattern: /가슴|chest|breast/iu, tags: ["chest focus"] },
    { pattern: /허리|waist|hip/iu, tags: ["hip focus"] },
    { pattern: /허벅지|thigh/iu, tags: ["thigh focus"] },
    { pattern: /다리|leg/iu, tags: ["leg focus"] },
    { pattern: /발|feet|foot/iu, tags: ["feet focus"] },
    { pattern: /눈|eye|시선/iu, tags: ["eye focus"] },
    { pattern: /입술|mouth|lip/iu, tags: ["mouth focus"] }
  ];
  tags.push(...rules.flatMap((rule) => (rule.pattern.test(text) ? rule.tags : [])));
  if (kind === "body_detail" && tags.length === 0) {
    tags.push("hands");
  }
  return uniqueStrings(tags);
}

function selectCuePropTags(text: string): string[] {
  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /key|열쇠/iu, tags: ["holding key"] },
    { pattern: /notebook|노트/iu, tags: ["holding notebook"] },
    { pattern: /\bbook\b|책\b/iu, tags: ["book"] },
    { pattern: /phone|휴대폰|스마트폰/iu, tags: ["phone"] },
    { pattern: /microphone|마이크/iu, tags: ["microphone"] },
    { pattern: /sword|검\b|칼\b/iu, tags: ["sword"] },
    { pattern: /gun|총\b/iu, tags: ["gun"] }
  ];
  return uniqueStrings(rules.flatMap((rule) => (rule.pattern.test(text) ? rule.tags : [])));
}

function selectCueExpressionTags(kind: string, text: string): string[] {
  const tags: string[] = [];
  if (kind === "dialogue_face" || /["“”「」『』]|말하|속삭|외치|신음|voice|says?|said|whisper|moan/iu.test(text)) {
    tags.push("open mouth");
  }
  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /smile|미소|웃/iu, tags: ["smile"] },
    { pattern: /tense|nervous|긴장|불안/iu, tags: ["tense expression"] },
    { pattern: /angry|frustrated|화난|분노|짜증/iu, tags: ["angry"] },
    { pattern: /sad|슬픈|울먹/iu, tags: ["sad"] },
    { pattern: /cry|tear|눈물|울/iu, tags: ["tears"] },
    { pattern: /blush|붉어|홍조/iu, tags: ["blush"] },
    { pattern: /sweat|땀/iu, tags: ["sweat"] },
    { pattern: /surpris|놀라|당황/iu, tags: ["surprised"] },
    { pattern: /look(?:ing)? at viewer|바라본|쳐다본|시선/iu, tags: ["looking at viewer"] }
  ];
  tags.push(...rules.flatMap((rule) => (rule.pattern.test(text) ? rule.tags : [])));
  return uniqueStrings(tags);
}

function filterContradictoryLocalImageTags(tags: string[]): string[] {
  const hasStage = tags.some((tag) => /\b(?:stage|spotlight|stage lights)\b/iu.test(tag));
  const hasStreet = tags.some((tag) => /\b(?:street|alley|outdoors)\b/iu.test(tag));
  const hasIndoor = tags.some((tag) => /\b(?:indoors|classroom|archive library|hallway|room|bedroom|kitchen|cafe|hospital|laboratory|practice room)\b/iu.test(tag));

  return tags.filter((tag) => {
    if (hasStage && hasStreet && /\b(?:street|alley|outdoors)\b/iu.test(tag)) {
      return false;
    }
    if (hasIndoor && /\b(?:street|alley|outdoors|forest|beach|battlefield)\b/iu.test(tag)) {
      return false;
    }
    return true;
  });
}

function createRecentImageCueContext(state: AppState): string {
  return state.messages
    .slice(-3)
    .map((message) => `${message.role}: ${message.content}`)
    .join("\n")
    .slice(-1000);
}

function sanitizeImageCueTags(tags: string[]): string[] {
  return uniqueStrings(
    tags
      .map((tag) =>
        tag
          .trim()
          .replace(/[._-]+/gu, " ")
          .replace(/[.!?。！？:：]+$/gu, "")
          .replace(/\s+/gu, " ")
      )
      .filter((tag) => tag.length > 0 && !/[\u3131-\uD79D]/u.test(tag))
      .filter((tag) => !isStaleSceneLabelTag(tag))
  );
}

function filterRosterNameCueTags(state: AppState, tags: string[]): string[] {
  const rosterTagNames = new Set(
    state.characters.flatMap((character) => [
      normalizeRosterCueTag(character.id),
      normalizeRosterCueTag(character.name)
    ])
  );
  return tags.filter((tag) => !containsRosterCueNameTag(rosterTagNames, tag));
}

function normalizeRosterCueTag(value: string): string {
  return value.toLowerCase().replace(/[._\s]+/gu, "-").trim();
}

function containsRosterCueNameTag(rosterTagNames: Set<string>, value: string): boolean {
  const normalized = normalizeRosterCueTag(value);
  if (rosterTagNames.has(normalized)) {
    return true;
  }

  const parts = new Set(normalized.split(/-+/u).filter(Boolean));
  return [...rosterTagNames].some((name) => {
    if (!name) {
      return false;
    }
    const nameParts = name.split(/-+/u).filter(Boolean);
    return nameParts.length > 0 && nameParts.every((part) => parts.has(part));
  });
}

function isStaleSceneLabelTag(tag: string): boolean {
  return /\b(?:commute|chapter|episode|cut|shot)\s+\d+\b/iu.test(tag) || /\bschool uniform commute\b/iu.test(tag);
}

function createImageCueVisualContext(
  state: AppState,
  combinedText: string,
  scene: string,
  characterIds: string[],
  assistantVisualContext?: string,
  includeLocalVisualTags = true,
  draft?: Pick<AssistantImageCueDraft, "kind" | "cueType">
): string {
  return [
    scene,
    ...createLocalImageCueTags(includeLocalVisualTags ? combinedText : "", state, draft),
    ...createCurrentImageCueStateTags(state, characterIds),
    ...sanitizeImageCueTags(assistantVisualContext ? assistantVisualContext.split(/[,;\n|]+/u) : [])
  ]
    .filter((item): item is string => Boolean(item?.trim()))
    .join(", ");
}

function createImageCueCharacterScopeText(currentTurnText: string, draft: AssistantImageCueDraft): string {
  return [
    draft.anchorText,
    draft.scene,
    draft.visualContext,
    draft.tags.join(", "),
    currentTurnText
  ]
    .filter((item): item is string => Boolean(item?.trim()))
    .join("\n");
}

function createFocusedImageCueText(currentTurnText: string, draft: AssistantImageCueDraft): string {
  const focused = [
    draft.anchorText,
    normalizeImageSceneLabel(draft.scene) ? draft.scene : undefined,
    draft.visualContext,
    draft.tags.join(", ")
  ]
    .filter((item): item is string => Boolean(item?.trim()))
    .join("\n");

  return focused || currentTurnText;
}

function findMentionedCharacterIds(state: AppState, text: string): string[] {
  const normalizedText = text.toLowerCase();
  const explicitCharacterIds = state.characters
      .filter((character) => {
        const normalizedName = character.name.toLowerCase();
        return Boolean(normalizedName && normalizedText.includes(normalizedName));
      })
      .map((character) => character.id);
  const personaCharacter = resolvePersonaCharacter(state);
  const personaIds =
    personaCharacter && hasFirstPersonSubjectCue(text) && !hasFirstPersonAsObjectOfExternalActor(text)
      ? [personaCharacter.id]
      : [];

  return uniqueStrings([...explicitCharacterIds, ...personaIds]);
}

function findPersonaActionCharacterIds(state: AppState, userText: string, focusedCueText: string, characterScopeText: string): string[] {
  const personaCharacter = resolvePersonaCharacter(state);
  if (!personaCharacter) {
    return [];
  }

  const userAction = userText.trim();
  if (!userAction) {
    return [];
  }

  const text = `${focusedCueText}\n${userText}`;
  if (hasFirstPersonAsObjectOfExternalActor(characterScopeText) || hasUnnamedExternalActorFocus(characterScopeText)) {
    return [];
  }

  if (!isLikelyPersonaCharacterAction(userAction, text)) {
    return [];
  }

  if (
    mentionsOtherRosterCharacter(state, personaCharacter.id, text) &&
    !hasFirstPersonSubjectCue(text) &&
    !isUserStageDirectionText(userAction)
  ) {
    return [];
  }

  return [personaCharacter.id];
}

function isLikelyPersonaCharacterAction(userText: string, cueText: string): boolean {
  if (hasFirstPersonSubjectCue(`${userText}\n${cueText}`) || isUserStageDirectionText(userText)) {
    return true;
  }

  return !isLikelyImageOnlyRequest(userText);
}

function isUserStageDirectionText(text: string): boolean {
  return /(?:\*\(|\)\*)|^\s*[\[(（(].+[\])）)]\s*$/u.test(text);
}

function isLikelyImageOnlyRequest(text: string): boolean {
  const normalized = text.trim();
  if (!normalized) {
    return true;
  }

  return /(?:이미지|그림|컷|장면|프롬프트|태그|생성|그려|보여줘|image|picture|prompt|tag|draw|generate|show)/iu.test(normalized) && !hasFirstPersonSubjectCue(normalized) && !isUserStageDirectionText(normalized);
}

function mentionsOtherRosterCharacter(state: AppState, personaCharacterId: string, text: string): boolean {
  const normalizedText = text.toLowerCase();
  return state.characters.some((character) => {
    if (character.id === personaCharacterId) {
      return false;
    }

    const normalizedName = character.name.toLowerCase().trim();
    return Boolean(normalizedName && normalizedText.includes(normalizedName));
  });
}

function filterImageCueCharactersForCurrentTurn(
  state: AppState,
  resolvedCharacters: string[],
  mentionedCharacters: string[],
  currentTurnText: string,
  characterScopeText: string = currentTurnText,
  trustLlmCharacterScope = false
): string[] {
  if (resolvedCharacters.length === 0) {
    return [];
  }

  if (trustLlmCharacterScope) {
    if (shouldDropCharactersForExternalCue(state, resolvedCharacters, characterScopeText)) {
      return [];
    }

    if (mentionedCharacters.length > 0) {
      const mentionedSet = new Set(mentionedCharacters);
      return resolvedCharacters.filter((characterId) => mentionedSet.has(characterId));
    }

    if (hasFullRosterSelectionWithoutGroupEvidence(state, resolvedCharacters, characterScopeText)) {
      return [];
    }

    return uniqueStrings(resolvedCharacters);
  }

  if (shouldDropCharactersForExternalCue(state, resolvedCharacters, characterScopeText)) {
    return [];
  }

  if (mentionedCharacters.length > 0) {
    const mentionedSet = new Set(mentionedCharacters);
    return resolvedCharacters.filter((characterId) => mentionedSet.has(characterId));
  }

  if (hasSoloExternalSubjectCue(characterScopeText)) {
    return [];
  }

  if (resolvedCharacters.length > 1) {
    return [];
  }

  const resolvedCharacter = state.characters.find((character) => character.id === resolvedCharacters[0]);
  return resolvedCharacter && hasCharacterEvidenceInCurrentTurn(resolvedCharacter, currentTurnText)
    ? resolvedCharacters
    : [];
}

function shouldDropCharactersForExternalCue(state: AppState, candidateCharacterIds: string[], text: string): boolean {
  if (
    !(
      hasUnnamedExternalActorFocus(text) ||
      hasClearUnnamedOutsiderImageFocus(text) ||
      hasUnregisteredSoloSubjectCue(text)
    )
  ) {
    return false;
  }

  return !hasResolvedCharacterActorEvidence(state, uniqueStrings(candidateCharacterIds), text);
}

function hasSoloExternalSubjectCue(text: string): boolean {
  return /혼자|홀로|단독|1인칭|일인칭|나\s*(?:혼자|만)|놈|녀석|사내|남자|남성|남학생|남자애|소년|낯선\s*(?:사람|인물)|모르는\s*(?:사람|인물|남자)|다른\s*(?:사람|인물)|선생|교사|직원|스태프|경비|감독|\b(?:solo|alone|single subject|first[-\s]?person|pov|guy|boy|man|male|stranger|outsider|teacher|staff|guard|director)\b/iu.test(text);
}

function hasUnregisteredSoloSubjectCue(text: string): boolean {
  const normalized = text.replace(/\s+/gu, " ");
  const soloSubject = "(?:혼자|홀로|단독|single subject|solo|alone)";
  const externalSubject = "(?:놈|녀석|사내|남자애|남학생|남자|남성|소년|낯선\\s*(?:사람|인물)|모르는\\s*(?:사람|인물|남자)|다른\\s*(?:사람|인물)|선생|교사|직원|스태프|경비|감독|1인칭|일인칭|first[-\\s]?person|pov|1boy|guy|boy|man|male|stranger|outsider|teacher|staff|guard|director)";
  return new RegExp(`(?:${soloSubject})[^.\\n]{0,70}(?:${externalSubject})|(?:${externalSubject})[^.\\n]{0,70}(?:${soloSubject})`, "iu").test(normalized);
}

function hasClearUnnamedOutsiderImageFocus(text: string): boolean {
  return /무명|이름\s*없는|낯선|처음\s*보는|외부인|다른\s*(?:사람|인물)|모르는\s*(?:사람|인물|남자|남성|소년)|선생|교사|직원|스태프|경비|감독|놈|녀석|사내|1인칭|일인칭|first[-\s]?person|pov|unnamed|unknown|stranger|outsider|teacher|staff|guard|director|lone unnamed|solo unnamed/iu.test(text);
}

function hasCharacterEvidenceInCurrentTurn(character: AppState["characters"][number], text: string): boolean {
  const normalizedText = text.toLowerCase();
  return [character.id, character.name]
    .map((value) => value.toLowerCase().trim())
    .some((value) => Boolean(value && normalizedText.includes(value)));
}

function hasUnnamedExternalActorFocus(text: string): boolean {
  return hasExternalSubjectActingCue(text) || hasFirstPersonAsObjectOfExternalActor(text);
}

function hasExternalSubjectActingCue(text: string): boolean {
  const normalized = text.replace(/\s+/gu, " ");
  const externalSubject = "(?:그\\s*)?(?:놈|녀석|사내|남자애|남학생|남자|남성|소년|낯선\\s*(?:사람|인물)|모르는\\s*(?:사람|인물|남자)|다른\\s*(?:사람|인물)|선생|교사|직원|스태프|경비|감독)|(?:guy|boy|man|male|stranger|outsider|teacher|staff|guard|director)";
  const actorVerb = "(?:웃|미소|말하|말했|말했다|말했다|지목|가리키|노려|쳐다|위협|협박|leering|leer|smil|speak|spoke|said|point|threaten)";
  return new RegExp(`(?:${externalSubject})(?:은|는|이|가|도)?[^.\\n]{0,90}${actorVerb}`, "iu").test(normalized);
}

function hasFirstPersonAsObjectOfExternalActor(text: string): boolean {
  const normalized = text.replace(/\s+/gu, " ");
  const firstPersonObject = "(?:나를|날|나에게|내게|나한테|내\\s*쪽으로|toward\\s+me|at\\s+me)";
  const externalSubject = "(?:그\\s*)?(?:놈|녀석|사내|남자애|남학생|남자|남성|소년|낯선\\s*(?:사람|인물)|모르는\\s*(?:사람|인물|남자)|다른\\s*(?:사람|인물)|선생|교사|직원|스태프|경비|감독)|(?:guy|boy|man|male|stranger|outsider|teacher|staff|guard|director)";
  const action = "(?:지목|가리키|노려|쳐다|위협|협박|비웃|웃|미소|말하|말했|말했다|point|stare|look|threaten|speak|spoke|said|smil|leer)";
  return new RegExp(`${firstPersonObject}[^.\\n]{0,90}(?:${action})?[^.\\n]{0,90}(?:${externalSubject})`, "iu").test(normalized);
}

function hasFullRosterSelectionWithoutGroupEvidence(state: AppState, resolvedCharacters: string[], text: string): boolean {
  if (state.characters.length < 3 || resolvedCharacters.length < state.characters.length) {
    return false;
  }

  if (/(?:전원|모두|다 같이|함께|세\s*명|네\s*명|전체|all|everyone|together|full cast|group shot|\b3girls\b|\b3boys\b)/iu.test(text)) {
    return false;
  }

  const mentionedIds = new Set(findMentionedCharacterIds(state, text));
  return mentionedIds.size < resolvedCharacters.length;
}

function hasFirstPersonSubjectCue(text: string): boolean {
  return /(?:^|[\s"'“”‘’([{])(?:나는|난|내가|나도|나\s*역시|내\s*(?:손|얼굴|몸|시선|입술|머리|눈)|i\s+(?:am|look|smile|speak|say|step|turn|reach)\b)/iu.test(text);
}

function hasResolvedCharacterActorEvidence(state: AppState, characterIds: string[], text: string): boolean {
  const actionPattern = /웃|미소|말하|말했|말했다|지목|가리키|노려|쳐다|위협|협박|leering|leer|smil|speak|spoke|said|point|threaten/iu;
  return characterIds.some((characterId) => {
    const character = state.characters.find((candidate) => candidate.id === characterId);
    const name = character?.name.trim();
    if (!name) {
      return false;
    }

    const index = text.toLowerCase().indexOf(name.toLowerCase());
    if (index < 0) {
      return false;
    }

    const segment = text.slice(index, index + 120);
    return actionPattern.test(segment) && !/(?:놈|녀석|사내|남자애|남학생|남자|남성|소년|\b(?:guy|boy|man|male)\b)/iu.test(segment.replace(name, ""));
  });
}

function curateMemoryEvents(
  state: AppState,
  sidecar: AssistantSidecar,
  assistantContent: string,
  sourceTurnId: string
): MemoryEvent[] {
  const drafts = sidecar.memoryEvents.length > 0 ? sidecar.memoryEvents : createFallbackMemoryDrafts(state, assistantContent);
  const now = new Date().toISOString();
  return drafts.slice(0, 5).map((draft) => {
    const actor = resolveActor(state, draft);
    return {
      id: createId("memory"),
      simulationId: state.simulation.id,
      sessionId: state.simulation.activeSessionId,
      actorId: actor?.id ?? draft.actorId,
      actorName: actor?.name ?? draft.actorName,
      content: draft.content.slice(0, 520),
      importance: clampNumber(draft.importance, 0, 1),
      tags: normalizeMemoryTags(draft.tags),
      sourceTurnId,
      createdAt: now
    };
  });
}

function createFallbackMemoryDrafts(state: AppState, assistantContent: string): AssistantMemoryEventDraft[] {
  const character = state.characters[0];
  const important = IMPORTANT_PATTERN.test(assistantContent);
  const drafts: AssistantMemoryEventDraft[] = [
    {
      actorId: character?.id,
      actorName: character?.name,
      content: assistantContent.slice(0, 360),
      importance: important ? 0.86 : 0.64,
      tags: important ? ["memory", "scene-state", "continuity", "curated-fallback"] : ["dialogue", "recent", "curated-fallback"]
    }
  ];

  if (character && assistantContent.includes(character.name)) {
    drafts.push({
      actorId: character.id,
      actorName: character.name,
      content: `${character.name} 현재 상태: ${[character.relationship, character.currentMood].filter(Boolean).join(", ") || "최근 장면에 반응함"}`,
      importance: 0.72,
      tags: ["character-state", "curated-fallback"]
    });
  }

  return drafts;
}

function resolveActor(state: AppState, draft: AssistantMemoryEventDraft) {
  return state.characters.find(
    (character) =>
      character.id === draft.actorId ||
      character.name === draft.actorName ||
      character.name.toLowerCase() === draft.actorName?.toLowerCase()
  );
}

function resolveCharacterIds(state: AppState, values: string[]): string[] {
  return uniqueStrings(
    values.flatMap((value) => {
      const normalized = normalizeCharacterLookupText(value);
      const character = state.characters.find(
        (candidate) =>
          normalizeCharacterLookupText(candidate.id) === normalized ||
          normalizeCharacterLookupText(candidate.name) === normalized ||
          state.visualProfiles.some(
            (profile) =>
              profile.characterId === candidate.id &&
              (normalizeCharacterLookupText(profile.id) === normalized ||
                normalizeCharacterLookupText(profile.displayName) === normalized)
          )
      );
      return character ? [character.id] : [];
    })
  );
}

function normalizeCharacterLookupText(value: string): string {
  return value.trim().toLowerCase().replace(/[._\s]+/gu, "-");
}

function normalizeMemoryTags(tags: string[]): string[] {
  const normalized = uniqueStrings(
    tags
      .map((tag) => tag.trim().toLowerCase().replace(/\s+/gu, "-"))
      .filter(Boolean)
  );
  return normalized.length > 0 ? normalized.slice(0, 8) : ["memory"];
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
