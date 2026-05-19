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
import { generateAssistantText } from "./llmClient";
import {
  compileSimulationMemoryDelta,
  createStructuredContextSummary,
  memoryDeltaToEvents,
  type MemoryDelta
} from "./memoryCompiler";
import { NeuralMapClient } from "./neuralMapClient";
import { createImageUserRulesForContentRating, isAdultContentMode } from "./contentRating";
import { characterIsReferencedInText, getCurrentSceneCharacters, inferCurrentSceneCharacterIds } from "./sceneCast";

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
const MISSING_MAIN_LLM_IMAGE_TAGS_SUPPRESSION_REASON = "메인 LLM image_cues.tags가 비어 있어 이미지 작업을 만들지 않음";
const IMAGE_PROGRESSION_CUE_TARGET = 10;

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
  plannerSource?: "main_llm_sidecar" | "user_image_rules" | "image_generation_cadence";
  forceFreshImage?: boolean;
  forceImagePlanning?: boolean;
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
  const moduleSelections = selectRelevantModules(state, retrievalQuery, contextPack, userText);
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
    manualImage,
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
  const drafts = requireMainLlmAuthoredImageTags(initialDrafts, Boolean(input.manualImage));
  const imageCues = drafts.map((draft) =>
    planImageCue(
      state,
      input.userMessage.content,
      draft,
      input.assistantMessage.content,
      {
        allowLocalTrigger: false,
        manualImage: input.manualImage
      }
    )
  );
  const contextNodeIds = input.contextPack.evidence.map((item) => item.nodeId);
  const reusedAssetIds: string[] = [];
  const imageJobs = imageCues
    .map((imageCue, index) => {
      const draft = drafts[index];
      if (isImageCueSuppressedForMissingMainLlmTags(imageCue)) {
        return undefined;
      }
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

function requireMainLlmAuthoredImageTags(
  drafts: ImageRuleBackedCueDraft[],
  manualImage: boolean
): ImageRuleBackedCueDraft[] {
  return drafts.map((draft) => {
    if (!isImageGenerationRequestedByDraft(draft, manualImage) || hasLlmAuthoredImageTags(draft)) {
      return draft;
    }

    return {
      ...draft,
      shouldGenerate: false,
      forceFreshImage: undefined,
      forceImagePlanning: false,
      suppressionReason: MISSING_MAIN_LLM_IMAGE_TAGS_SUPPRESSION_REASON,
      reason: draft.reason || MISSING_MAIN_LLM_IMAGE_TAGS_SUPPRESSION_REASON
    };
  });
}

function isImageGenerationRequestedByDraft(draft: ImageRuleBackedCueDraft, manualImage: boolean): boolean {
  return (
    draft.shouldGenerate ||
    manualImage ||
    draft.forceImagePlanning === true ||
    draft.plannerSource === "user_image_rules" ||
    draft.plannerSource === "image_generation_cadence"
  );
}

function hasLlmAuthoredImageTags(draft: Pick<AssistantImageCueDraft, "tags" | "baseTags" | "characterPrompts">): boolean {
  return (
    draft.tags.some((tag) => tag.trim().length > 0) ||
    (draft.baseTags ?? []).some((tag) => tag.trim().length > 0) ||
    (draft.characterPrompts ?? []).some((prompt) => prompt.prompt.trim().length > 0)
  );
}

function isImageCueSuppressedForMissingMainLlmTags(cue: ImageCue): boolean {
  return cue.suppressionReason?.startsWith(MISSING_MAIN_LLM_IMAGE_TAGS_SUPPRESSION_REASON) === true;
}

function isGenericImageSceneTag(value: string): boolean {
  const normalized = value.toLowerCase().replace(/[._-]+/gu, " ").trim();
  return /^(?:(?:current|generated|simulation|safe)\s+)*scene$/iu.test(normalized);
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
        mainModel: state.llm.model,
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
        forceImagePlanning: cuePlan.ignoreCooldown || shouldImageCadenceForcePlanning(cadence) || undefined
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
  return cadence === "sparse" || cadence === "balanced" || cadence === "rich" || cadence === "paragraph" || cadence === "image_progression"
    ? cadence
    : "balanced";
}

function shouldImageCadenceRequireGeneration(cadence: ImageGenerationCadence, userText: string, assistantText: string): boolean {
  if (cadence === "image_progression") {
    return Boolean(`${userText}\n${assistantText}`.trim());
  }
  if (cadence === "paragraph") {
    return Boolean(assistantText.trim());
  }
  if (cadence === "rich") {
    return hasVisualCueText(`${userText}\n${assistantText}`);
  }
  if (cadence === "balanced") {
    return hasVisualCueText(`${userText}\n${assistantText}`);
  }

  return false;
}

function shouldImageCadenceForcePlanning(cadence: ImageGenerationCadence): boolean {
  return cadence === "balanced" || cadence === "rich" || cadence === "paragraph" || cadence === "image_progression";
}

function shouldImageCadenceForceFresh(cadence: ImageGenerationCadence): boolean {
  return cadence === "rich" || cadence === "paragraph" || cadence === "image_progression";
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
    const richAnchors = selectRichVisualBeatAnchors(assistantText);
    if (richAnchors.length > 0) {
      richAnchors.forEach((anchorText, index) => {
        if (hasDraftAnchor(withCadenceCues, anchorText)) {
          return;
        }
        withCadenceCues.push(createImageCadenceCueDraft(inferRichVisualBeatKind(anchorText), assistantText, anchorText, index + 1));
      });
    } else if (hasActionBeatText(`${userText}\n${assistantText}`) && !hasImageCueKind(withCadenceCues, "action")) {
      withCadenceCues.push(createImageCadenceCueDraft("action", assistantText, selectImageCueAnchorText(assistantText, "action"), 1));
    }
    if (hasDialogueText(assistantText) && !hasImageCueKind(withCadenceCues, "dialogue_face")) {
      const anchorText = selectImageCueAnchorText(assistantText, "dialogue_face");
      if (!anchorText || !hasDraftAnchor(withCadenceCues, anchorText)) {
        withCadenceCues.push(createImageCadenceCueDraft("dialogue_face", assistantText, anchorText, 2));
      }
    }
  } else if (cadence === "balanced" && cadenceRequiresGeneration) {
    if (!hasImageCueKind(withCadenceCues, "scene", "context", "action", "dialogue_face", "body_detail", "interaction")) {
      withCadenceCues.push(createImageCadenceCueDraft("scene", assistantText, selectImageCueAnchorText(assistantText, "scene"), 0));
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
      forceImagePlanning: draft.forceImagePlanning || shouldImageCadenceForcePlanning(cadence) || undefined
    };
  });
  const maxCueCount = cadence === "image_progression" ? IMAGE_PROGRESSION_CUE_TARGET : hasUserRuleRequirement ? 8 : getImageCadenceMaxCueCount(cadence);
  return normalizedDrafts.slice(0, maxCueCount);
}

function getImageCadenceMaxCueCount(cadence: ImageGenerationCadence): number {
  if (cadence === "image_progression") {
    return IMAGE_PROGRESSION_CUE_TARGET;
  }
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

  return {
    shouldGenerate: true,
    reason: "이미지 생성 밀도 설정이 이 문맥의 별도 이미지를 요구함",
    characters: [],
    scene: "current simulation scene",
    plannerSource: "image_generation_cadence",
    forceFreshImage: true,
    forceImagePlanning: true,
    anchorText,
    priority: Math.max(0.72, 0.9 - index * 0.03),
    kind: normalizedKind,
    label: normalizedKind === "dialogue_face" ? "dialogue face" : normalizedKind === "action" ? "action beat" : "scene establishing",
    placement,
    tags: []
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

function selectRichVisualBeatAnchors(assistantText: string): string[] {
  const trimmed = assistantText.trim();
  if (!trimmed) {
    return [];
  }

  const blocks = trimmed
    .split(/\n{2,}|\n+/u)
    .map(cleanImageAnchorText)
    .filter((block) => block.length >= 8);
  const visualBlocks = blocks.filter((block) => hasVisualCueText(block));
  const candidates = visualBlocks.length > 0 ? visualBlocks : blocks;
  return uniqueStrings(candidates.map((item) => item.slice(0, 140))).slice(0, 4);
}

function inferRichVisualBeatKind(anchorText: string): NonNullable<AssistantImageCueDraft["kind"]> {
  if (/말하|속삭|외치|신음|숨소리|표정|눈|입|시선|dialogue|voice|says?|said|whisper|moan|face|eyes?|mouth/iu.test(anchorText)) {
    return "dialogue_face";
  }
  if (/손|손목|팔|어깨|가슴|허리|허벅지|다리|발|body|hand|wrist|arm|shoulder|chest|waist|thigh|leg|feet/iu.test(anchorText)) {
    return "body_detail";
  }
  if (/잡|놓|밀|당기|닿|기대|grabs?|holds?|touch(?:es|ing)?|push(?:es|ing)?|pull(?:s|ing)?|leans?/iu.test(anchorText)) {
    return "interaction";
  }
  return "action";
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
    anchorText,
    priority: kind === "scene" ? 0.92 : 0.86,
    kind
  };

  if (kind === "action") {
    return {
      ...base,
      label: "action beat",
      placement: "inline",
      tags: []
    };
  }
  if (kind === "body_detail") {
    return {
      ...base,
      label: "body detail",
      placement: "inline",
      tags: []
    };
  }
  if (kind === "dialogue_face") {
    return {
      ...base,
      label: "dialogue face",
      placement: "before",
      tags: []
    };
  }

  return {
    ...base,
    label: "scene establishing",
    placement: "before",
    tags: []
  };
}

function shouldForceImagePlanningFromUserRules(
  state: AppState,
  draft: ImageRuleBackedCueDraft | undefined,
  cue: ImageCue,
  manualImage: boolean
): boolean {
  if (!draft?.forceImagePlanning || (!cue.shouldGenerate && !draft.shouldGenerate && !manualImage)) {
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

function selectRelevantModules(
  state: AppState,
  userText: string,
  contextPack: ContextPack,
  currentSceneText = userText
): PromptModuleSelection[] {
  const modules = state.modules.filter((module) => !(module.kind === "safety_policy" && isAdultContentMode(state)));
  const normalizedText = userText.toLowerCase();
  const queryTerms = createSelectionTerms(userText);
  const selectionEvidence = getModuleSelectionEvidence(contextPack);
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state, currentSceneText));

  return modules
    .filter((module) => module.enabled && module.tokenPolicy !== "disabled")
    .map((module): PromptModuleSelection | undefined => {
      const character = findModuleCharacter(state, module);
      const tagMatch = module.activationTags.some((tag) => textContainsSelectionPhrase(normalizedText, tag));
      const titleMatch = normalizedText.includes(module.title.toLowerCase());
      const characterNameMatch = character ? characterIsReferencedInText(character, userText) : false;
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
      const directCharacterSignal = character ? characterNameMatch : tagMatch || titleMatch;

      if (
        module.kind === "character_prompt" &&
        character &&
        !shouldIncludeCharacterScopedContext(state, character.id, activeCharacterIds) &&
        !directCharacterSignal
      ) {
        return undefined;
      }

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
  if (module.kind === "character_prompt" && character) {
    return (
      item.nodeId === module.id ||
      item.nodeId.includes(module.id) ||
      item.nodeId.includes(character.id) ||
      characterIsReferencedInText(character, `${item.nodeId}\n${item.snippet}`)
    );
  }

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

function shouldIncludeCharacterScopedContext(
  state: AppState,
  characterId: string,
  activeCharacterIds: ReadonlySet<string>
): boolean {
  return activeCharacterIds.has(characterId) || (state.characters.length === 1 && state.characters[0]?.id === characterId);
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
  const memoryHint = createFallbackMemoryHint(evidence);
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

function createFallbackMemoryHint(evidence: string[]): string {
  for (const snippet of evidence) {
    const safeHint = sanitizeFallbackEvidenceHint(snippet);
    if (safeHint) {
      return safeHint;
    }
  }

  return "최근 대화 연속성 단서는 Context Pack에 보존되어 있다.";
}

function sanitizeFallbackEvidenceHint(value: string): string {
  const normalized = value.replace(/\r\n?/gu, "\n").trim();
  if (!normalized || looksLikeInternalFallbackEvidence(normalized)) {
    return "";
  }

  const compact = normalized
    .split("\n")
    .map((line) => line.replace(/^#+\s*/u, "").trim())
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/gu, " ")
    .trim();

  if (!compact || looksLikeInternalFallbackEvidence(compact)) {
    return "";
  }

  return compact.length > 180 ? `${compact.slice(0, 177).trimEnd()}...` : compact;
}

function looksLikeInternalFallbackEvidence(value: string): boolean {
  return /(?:#\s*Immediate Continuity Anchor|Use this before older retrieved memories|Immediate continuity anchor:|SYSTEM INSTRUCTION:|CONTEXT BLOCK:|USER ACTION:|Return JSON only\. The JSON schema is|Current scene cast guard:|Structured simulation memory:|Simulation foundation:|Selected prompt modules for this turn:|Memory\/context evidence:)/iu.test(value);
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
  void state;
  void userText;
  return {
    shouldGenerate: manualImage,
    reason: manualImage ? "수동 이미지 생성 요청 대기" : "메인 LLM 이미지 cue 대기",
    characters: [],
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
  options: { allowLocalTrigger?: boolean; manualImage?: boolean } = {}
): ImageCue {
  void userText;
  void assistantContent;
  const backedDraft = draft as ImageRuleBackedCueDraft;
  const generationRequested =
    draft.shouldGenerate ||
    Boolean(options.manualImage) ||
    Boolean(backedDraft.forceImagePlanning);
  const rawResolvedCharacters = resolveCharacterIds(state, draft.characters);
  const shouldGenerate = generationRequested;
  const scene = resolveImageScene(draft.scene);
  const characterIds = rawResolvedCharacters;
  const tags = sanitizeImageCueTags(draft.tags, state);
  const baseTags = draft.baseTags ? sanitizeImageCueTags(draft.baseTags, state) : undefined;
  const characterPrompts = (draft.characterPrompts ?? [])
    .map((prompt) => ({
      ...prompt,
      characterId: prompt.characterId ? resolveCharacterIds(state, [prompt.characterId])[0] ?? prompt.characterId : undefined,
      prompt: sanitizeImageCueTags(prompt.prompt.split(","), state).join(", "),
      negativePrompt: prompt.negativePrompt ? sanitizeImageCueTags(prompt.negativePrompt.split(","), state).join(", ") : undefined
    }))
    .filter((prompt) => prompt.prompt.trim());
  const visualContext = draft.visualContext?.trim() || tags.join(", ");

  return {
    shouldGenerate,
    reason: draft.reason || (shouldGenerate ? "메인 LLM이 현재 장면을 이미지 cue로 선택함" : "메인 LLM이 이미지 생성을 생략함"),
    characters: characterIds,
    tags,
    baseTags,
    characterPrompts,
    scene,
    suppressionReason: shouldGenerate
      ? draft.suppressionReason
      : draft.suppressionReason ?? "메인 LLM이 이미지 생성을 생략함",
    visualContext
  };
}

function resolveImageScene(draftScene: string | undefined): string {
  const normalizedDraftScene = normalizeImageSceneLabel(draftScene);
  if (normalizedDraftScene) {
    return normalizedDraftScene;
  }

  return "current simulation scene";
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

function sanitizeImageCueTags(tags: string[], state?: AppState): string[] {
  const sanitized = uniqueStrings(
    tags
      .map((tag) =>
        tag
          .trim()
          .replace(/[.!?。！？:：]+$/gu, "")
          .replace(/\s+/gu, " ")
      )
      .filter((tag) => tag.length > 0)
  );
  return state ? filterRosterNameCueTags(state, sanitized) : sanitized;
}

function filterRosterNameCueTags(state: AppState, tags: string[]): string[] {
  const rosterNames = createRosterCueTagNameSet(state);
  if (rosterNames.size === 0) {
    return tags;
  }

  return tags.filter((tag) => !rosterNames.has(normalizeRosterCueTag(tag)));
}

function createRosterCueTagNameSet(state: AppState): Set<string> {
  return new Set(
    state.characters.flatMap((character) => {
      const visualProfile = state.visualProfiles.find((profile) => profile.characterId === character.id);
      return [
        normalizeRosterCueTag(character.id),
        normalizeRosterCueTag(character.name),
        normalizeRosterCueTag(visualProfile?.id ?? ""),
        normalizeRosterCueTag(visualProfile?.displayName ?? "")
      ].filter(Boolean);
    })
  );
}

function normalizeRosterCueTag(value: string): string {
  return value
    .toLowerCase()
    .replace(/'s\b/gu, "")
    .replace(/[._\s-]+/gu, " ")
    .replace(/[^\p{L}\p{N} ]+/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
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
