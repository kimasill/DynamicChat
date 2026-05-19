import { createId } from "../lib/id";
import type { AppState, AssistantMemoryEventDraft, AssistantSidecar, ContextEvidence, MemoryEvent } from "../types";
import { characterIsReferencedInText, inferCurrentSceneCharacterIds } from "./sceneCast";
import {
  canonicalizeStateType,
  classifyImageStateTags,
  formatImageStateTypeLabel,
  isImageStateType,
  readStateMemoryKind,
  readStateMemoryOwnerId,
  readStateMemoryStateType,
  readStateMemoryValue,
  splitImageStateTagValue
} from "./stateMemory";

export type SimulationMemoryKind =
  | "event"
  | "state"
  | "observation"
  | "belief"
  | "goal"
  | "relationship"
  | "open_thread"
  | "summary";

type MemoryLayer = "semantic" | "episodic" | "procedural";

export interface MemoryDeltaRecord {
  id: string;
  kind: SimulationMemoryKind;
  layer: MemoryLayer;
  content: string;
  importance: number;
  confidence: number;
  tags: string[];
  actorId?: string;
  actorName?: string;
  ownerId?: string;
  targetId?: string;
  stateType?: string;
  value?: string;
  eventType?: string;
  observers?: string[];
  importanceReasons: string[];
}

export interface MemoryDelta {
  id: string;
  turnId: string;
  simTime: string;
  sceneId: string;
  upsertRecords: MemoryDeltaRecord[];
  warnings: string[];
}

const MAX_DELTA_RECORDS = 8;
const IMAGE_CUE_CHARACTER_STATE_TYPES = new Set<string>([
  "Wearing",
  "OutfitTags",
  "StatusTags",
  "ExpressionTags",
  "PoseTags",
  "ActionTags",
  "InteractionTags",
  "InteractionPhaseTags",
  "HeldItemTags",
  "PhysicalStateTags",
  "BodyStateTags",
  "PhysicalCondition",
  "Emotion"
]);
const IMPORTANT_EVENT_PATTERN = /기억|약속|관계|갈등|위험|비밀|장소|목표|상태|변화|단서|목격|알게|잃어|얻었|이동|도착|떠났|부상|아프|통증|고백|거짓말|계약|돈|채무|훈련|성장/u;
const OPEN_THREAD_PATTERN = /아직|모른|모름|확정되지|가능성|의심|수상|비밀|원인|해결되지|떡밥|추측/u;
const OBSERVATION_PATTERN = /목격|봤다|보았다|보는|본다|들었다|듣고|눈치챘|알아차렸/u;
const BELIEF_PATTERN = /믿|생각|추측|의심|확신|오해|알고 있|모른다/u;

const STATE_PATTERNS: Array<{
  type: string;
  label: string;
  pattern: RegExp;
  valuePattern?: RegExp;
  importance: number;
}> = [
  {
    type: "Wearing",
    label: "착용",
    pattern: /입었|입고|갈아입|착용|벗었|교복|후드티|코트|복장|의상|의상\s*태그|wearing|changed into|outfit tags?|clothing tags?/iu,
    valuePattern: /(?:의상\s*태그|outfit tags?|clothing tags?|wearing|착용|복장|의상)[:：\s]+([^\n.!?。]{2,90})/iu,
    importance: 0.64
  },
  {
    type: "StatusTags",
    label: "상태 태그",
    pattern: /상태\s*태그|캐릭터별\s*상태|표정\s*태그|컨디션\s*태그|status tags?|visual state tags?|condition tags?/iu,
    valuePattern: /(?:상태\s*태그|표정\s*태그|컨디션\s*태그|status tags?|visual state tags?|condition tags?)[:：\s]+([^\n.!?。]{2,90})/iu,
    importance: 0.62
  },
  {
    type: "Location",
    label: "위치",
    pattern: /도착|이동|들어갔|나갔|향했|머물|방|복도|교실|연습실|도서관|주방|카페|병원|무대|분수|시계탑|location|arrived|entered/iu,
    valuePattern: /(방|복도|교실|연습실|도서관|주방|카페|병원|무대|분수|시계탑|기숙사|오피스텔|거리|rooftop|library|classroom|stage|kitchen|hospital)/iu,
    importance: 0.72
  },
  {
    type: "PhysicalCondition",
    label: "신체 상태",
    pattern: /복통|배가 아프|아파|통증|부상|다쳤|열이|피곤|탈진|기침|fever|pain|injury|sick/iu,
    valuePattern: /(복통|배가 아픔|통증|부상|피로|탈진|열|기침|stomach pain|pain|injury|fever|fatigue)/iu,
    importance: 0.78
  },
  {
    type: "Emotion",
    label: "감정",
    pattern: /긴장|불안|분노|화가|기쁨|안도|슬픔|경계|기대|무서|겁|nervous|angry|relieved|sad|afraid/iu,
    valuePattern: /(긴장|불안|분노|기쁨|안도|슬픔|경계|기대|두려움|nervous|angry|relieved|sad|afraid)/iu,
    importance: 0.58
  },
  {
    type: "Goal",
    label: "목표",
    pattern: /목표|원한다|하려고|숨기고 싶|찾으려|해결하|약속|goal|wants|intends|promise/iu,
    importance: 0.72
  }
];

export function compileSimulationMemoryDelta(input: {
  state: AppState;
  userText: string;
  assistantText: string;
  sidecar: AssistantSidecar;
  sourceTurnId: string;
  createdAt?: string;
}): MemoryDelta {
  const createdAt = input.createdAt ?? new Date().toISOString();
  const simTime = inferSimulationTime(input.assistantText, createdAt);
  const sceneId = inferSceneId(input.state, input.userText, input.assistantText);
  const records: MemoryDeltaRecord[] = [];
  const warnings: string[] = [];

  records.push(...recordsFromSidecar(input.state, input.sidecar.memoryEvents, input.assistantText));
  records.push(...recordsFromImageCueDrafts(input.state, input.sidecar.imageCues.length > 0 ? input.sidecar.imageCues : [input.sidecar.imageCue], records));
  records.push(...extractStateRecords(input.state, input.userText, input.assistantText));
  records.push(...extractObservationAndBeliefRecords(input.state, input.userText, input.assistantText));

  if (records.length === 0) {
    const fallback = createFallbackEventRecord(input.state, input.userText, input.assistantText);
    if (fallback) {
      records.push(fallback);
    } else {
      warnings.push("No durable simulation delta detected for this turn.");
    }
  }

  const existingKeys = new Set(
    input.state.memoryEvents.slice(-60).map((event) => readMemoryKey(event) ?? normalizeMemoryText(event.content))
  );
  const seen = new Set<string>();
  const upsertRecords = records
    .filter((record) => record.content.trim().length > 0)
    .filter((record) => {
      const key = createMemoryRecordKey(record);
      if (seen.has(key) || existingKeys.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    })
    .sort((a, b) => b.importance - a.importance)
    .slice(0, MAX_DELTA_RECORDS);

  return {
    id: createId("memdelta"),
    turnId: input.sourceTurnId,
    simTime,
    sceneId,
    upsertRecords,
    warnings
  };
}

export function memoryDeltaToEvents(state: AppState, delta: MemoryDelta, createdAt = new Date().toISOString()): MemoryEvent[] {
  return delta.upsertRecords.map((record) => {
    const actor = resolveCharacter(state, record.actorId, record.actorName) ?? resolveCharacter(state, record.ownerId);
    const memoryKey = createMemoryRecordKey(record);
    const graphNeuronId = createMemoryGraphNeuronId(state, delta, record);
    const supersededState = record.kind === "state" ? findPreviousStateEvent(state, record) : undefined;
    return {
      id: createId("memory"),
      simulationId: state.simulation.id,
      sessionId: state.simulation.activeSessionId,
      actorId: actor?.id ?? record.actorId ?? record.ownerId,
      actorName: actor?.name ?? record.actorName,
      content: formatRecordContent(state, record).slice(0, 520),
      importance: clamp(record.importance, 0, 1),
      tags: normalizeTags([
        "memory-delta",
        `kind:${record.kind}`,
        `layer:${record.layer}`,
        record.eventType ? `event:${record.eventType}` : undefined,
        record.stateType ? `state:${record.stateType}` : undefined,
        ...record.tags
      ]),
      sourceTurnId: delta.turnId,
      createdAt,
      metadata: {
        simulation_id: state.simulation.id,
        session_id: state.simulation.activeSessionId,
        progress_run_id: state.activeProgressRunId,
        run_id: state.activeProgressRunId,
        memory_delta_id: delta.id,
        memory_record_id: record.id,
        memory_key: memoryKey,
        graph_profile_id: SIMULATION_MEMORY_PROFILE_ID,
        graph_neuron_id: graphNeuronId,
        memory_kind: record.kind,
        memory_layer: record.layer,
        event_type: record.eventType,
        state_type: record.stateType,
        value: record.value,
        owner_id: record.ownerId,
        target_id: record.targetId,
        observer_ids: record.observers,
        confidence: record.confidence,
        sim_time: delta.simTime,
        scene_id: delta.sceneId,
        valid_from: record.kind === "state" ? delta.simTime : undefined,
        valid_to: null,
        supersedes: supersededState?.id,
        previous_value: readString(supersededState?.metadata?.value),
        importance_reasons: record.importanceReasons
      }
    };
  });
}

export function createStructuredContextSummary(
  state: AppState,
  options: { maxEvents?: number; maxStates?: number; currentText?: string } = {}
): string {
  const maxEvents = options.maxEvents ?? 8;
  const maxStates = options.maxStates ?? 10;
  const activeCharacterIds = new Set(inferCurrentSceneCharacterIds(state, options.currentText ?? ""));
  const activeCharacterNames = state.characters
    .filter((character) => activeCharacterIds.has(character.id))
    .map((character) => character.name)
    .join(", ");
  const records = state.memoryEvents
    .map((event) => toReadableMemoryRecord(state, event))
    .filter((record): record is ReadableMemoryRecord => Boolean(record));
  const currentSceneRecords = records.filter((record) =>
    recordIsRelevantToCurrentSceneCast(state, record, activeCharacterIds)
  );
  const currentStates = latestByKey(
    currentSceneRecords.filter((record) => record.kind === "state"),
    (record) => `${record.ownerId ?? record.actorId ?? "world"}:${record.stateType ?? "State"}`
  ).slice(0, maxStates);
  const observations = currentSceneRecords.filter((record) => record.kind === "observation").slice(-6);
  const beliefs = currentSceneRecords.filter((record) => record.kind === "belief").slice(-6);
  const events = currentSceneRecords
    .filter((record) => record.kind === "event" || record.kind === "relationship" || record.kind === "goal")
    .slice(-maxEvents);
  const openThreads = currentSceneRecords.filter((record) => record.kind === "open_thread").slice(-5);
  const latestScene = [...records].reverse().find((record) => record.sceneId || record.simTime);

  return [
    "# Simulation Context",
    "",
    "## Current Scene",
    `- Time: ${latestScene?.simTime ?? "unknown"}`,
    `- Scene: ${latestScene?.sceneId ?? state.simulation.activeSessionId}`,
    `- Active Characters: ${activeCharacterNames || "none explicit in recent transcript/current action"}`,
    "- Cast Rule: registered roster, relationship map entries, and stored status values are reference data; they do not make a character present unless the current scene evidence says so.",
    "- Off-stage character states are omitted from this current-turn summary until that character is active in the recent transcript or current action.",
    "",
    "## Canonical World State",
    ...formatRecordLines(state, currentStates, "현재 상태 없음"),
    "",
    "## Character Knowledge",
    ...formatKnowledgeLines(state, observations, beliefs),
    "",
    "## Relevant Past Events",
    ...formatRecordLines(state, events, "관련 사건 없음"),
    "",
    "## Open Threads",
    ...formatRecordLines(state, openThreads, "미해결 사항 없음")
  ].join("\n");
}

export function createStructuredContextEvidence(state: AppState): ContextEvidence | undefined {
  const summary = createStructuredContextSummary(state);
  if (!state.memoryEvents.some((event) => event.tags.includes("memory-delta"))) {
    return undefined;
  }

  return {
    nodeId: `simulation:${state.simulation.id}:structured-context`,
    snippet: summary,
    score: 0.96,
    reason: "구조화된 시뮬레이션 현재 상태"
  };
}

export const SIMULATION_MEMORY_PROFILE_ID = "simulation-memory";

export function createMemoryGraphNeuronId(state: AppState, delta: MemoryDelta, record: MemoryDeltaRecord): string {
  return [
    "simulation",
    slugify(state.simulation.id),
    "memory",
    slugify(delta.turnId),
    slugify(record.kind),
    record.id
  ].join(":");
}

function recordsFromSidecar(state: AppState, drafts: AssistantMemoryEventDraft[], assistantText: string): MemoryDeltaRecord[] {
  return drafts
    .filter((draft) => !looksLikeRawAssistantOutput(draft.content, assistantText))
    .map((draft) => {
      const mentionedCharacters = findMentionedCharacters(state, draft.content);
      const actor = resolveCharacter(state, draft.actorId, draft.actorName) ?? mentionedCharacters[0];
      const target = resolveCharacter(state, draft.targetId) ?? mentionedCharacters.find((character) => character.id !== actor?.id);
      const inferred = inferRecordKind(draft);
      const stateType = canonicalizeStateType(draft.stateType ?? inferred.stateType);
      const ownerId = inferred.kind === "state" ? actor?.id ?? draft.actorId : undefined;
      const targetId = target?.id ?? draft.targetId;
      const stateValue = normalizeStateRecordValue(state, {
        ownerId,
        stateType,
        value: draft.stateValue ?? inferStateValue(draft.content, stateType)
      });
      return createRecord({
        state,
        kind: inferred.kind,
        layer: inferred.layer,
        content: draft.content,
        importance: draft.importance,
        confidence: draft.confidence ?? 0.78,
        actorId: actor?.id ?? draft.actorId,
        actorName: actor?.name ?? draft.actorName,
        ownerId,
        targetId,
        observers: normalizeObserverIds(state, draft.observers, actor?.id, inferred.kind),
        stateType,
        value: stateValue,
        eventType: draft.eventType ?? inferred.eventType,
        tags: draft.tags,
        importanceReasons: ["llm_memory_delta"]
      });
    });
}

function recordsFromImageCueDrafts(
  state: AppState,
  drafts: AssistantSidecar["imageCues"],
  existingRecords: MemoryDeltaRecord[]
): MemoryDeltaRecord[] {
  if (!state.relationshipMap.enabled) {
    return [];
  }

  const existingStateKeys = new Set(
    existingRecords
      .filter((record) => record.kind === "state" && record.stateType)
      .map((record) => `${record.ownerId ?? record.actorId ?? "scene"}:${canonicalizeStateType(record.stateType)}`)
  );
  const records: MemoryDeltaRecord[] = [];

  for (const draft of drafts) {
    if (draft.tags.length === 0 || (draft.shouldGenerate === false && !draft.visualContext?.trim())) {
      continue;
    }

    const characters = draft.characters
      .map((characterId) => resolveCharacter(state, characterId))
      .filter((character): character is AppState["characters"][number] => Boolean(character));
    if (characters.length === 0) {
      continue;
    }

    const groups = classifyImageStateTags(draft.tags);
    for (const character of characters) {
      for (const [rawStateType, tags] of Object.entries(groups)) {
        const stateType = canonicalizeStateType(rawStateType);
        if (!isImageStateType(stateType) || !IMAGE_CUE_CHARACTER_STATE_TYPES.has(stateType) || tags.length === 0) {
          continue;
        }

        const stateKey = `${character.id}:${stateType}`;
        if (existingStateKeys.has(stateKey)) {
          continue;
        }

        existingStateKeys.add(stateKey);
        const value = normalizeStateRecordValue(state, {
          ownerId: character.id,
          stateType,
          value: tags.join(", ")
        }) ?? tags.join(", ");
        records.push(
          createRecord({
            state,
            kind: "state",
            layer: "episodic",
            content: `${character.name} ${formatImageStateTypeLabel(stateType)}: ${value}`,
            importance: stateType === "Wearing" ? 0.66 : 0.58,
            confidence: 0.62,
            actorId: character.id,
            actorName: character.name,
            ownerId: character.id,
            stateType,
            value,
            tags: ["image-cue-state", stateType],
            importanceReasons: ["image_cue_tags_returned", "relationship_status_sync"]
          })
        );
      }
    }
  }

  return records.slice(0, 6);
}

function normalizeStateRecordValue(
  state: AppState,
  input: { ownerId?: string; stateType?: string; value?: string }
): string | undefined {
  if (!input.value?.trim()) {
    return undefined;
  }

  const stateType = canonicalizeStateType(input.stateType);
  if (stateType !== "Wearing" && stateType !== "OutfitTags") {
    return input.value.trim();
  }

  return preserveWearingStateDetails(state, input.ownerId, input.value);
}

function preserveWearingStateDetails(state: AppState, ownerId: string | undefined, value: string): string {
  const incomingTags = expandOutfitKeywordTags(state, ownerId, splitImageStateTagValue(value), value);
  if (incomingTags.length === 0) {
    return value.trim();
  }

  const baseTags = findStableOutfitBaseTags(state, ownerId);
  const shouldPreserveBase = shouldPreserveOutfitBase(value, incomingTags, baseTags);
  const tags = shouldPreserveBase ? uniquePromptTags([...baseTags, ...incomingTags]) : incomingTags;
  return limitStateTagValue(tags).join(", ");
}

function expandOutfitKeywordTags(state: AppState, ownerId: string | undefined, tags: string[], rawValue: string): string[] {
  const visualProfile = ownerId ? state.visualProfiles.find((profile) => profile.characterId === ownerId) : undefined;
  const matchingMappings = Object.entries(visualProfile?.outfitPrompts ?? {})
    .map(([key, prompt]) => [key.trim(), prompt.trim()] as const)
    .filter(([key, prompt]) => key && prompt && doesOutfitValueMentionKeyword(rawValue, tags, key));

  if (matchingMappings.length === 0) {
    return uniquePromptTags(tags);
  }

  const mappingTags = matchingMappings.flatMap(([, prompt]) => splitImageStateTagValue(prompt));
  const mappingKeys = matchingMappings.map(([key]) => key);
  const retainedTags = tags.filter((tag) => {
    if (mappingKeys.some((key) => doesOutfitTagMatchKeyword(tag, key))) {
      return isOutfitConditionTag(tag);
    }
    return true;
  });

  return uniquePromptTags([...mappingTags, ...retainedTags]);
}

function doesOutfitValueMentionKeyword(rawValue: string, tags: string[], keyword: string): boolean {
  return doesOutfitTagMatchKeyword(rawValue, keyword) || tags.some((tag) => doesOutfitTagMatchKeyword(tag, keyword));
}

function doesOutfitTagMatchKeyword(value: string, keyword: string): boolean {
  const normalizedValue = value.toLowerCase().replace(/[._-]+/gu, " ");
  const normalizedKeyword = keyword.toLowerCase().replace(/[._-]+/gu, " ").trim();
  if (!normalizedKeyword) {
    return false;
  }

  if (/^[a-z0-9 ]+$/iu.test(normalizedKeyword)) {
    return new RegExp(`(?:^|\\b)${escapeRegExp(normalizedKeyword)}(?:\\b|$)`, "iu").test(normalizedValue);
  }

  return normalizedValue.includes(normalizedKeyword);
}

function findStableOutfitBaseTags(state: AppState, ownerId: string | undefined): string[] {
  if (!ownerId) {
    return [];
  }

  const previousValue = state.memoryEvents
    .slice()
    .reverse()
    .find((event) => {
      const stateType = readStateMemoryStateType(event);
      return (
        readStateMemoryKind(event) === "state" &&
        readStateMemoryOwnerId(event) === ownerId &&
        (stateType === "Wearing" || stateType === "OutfitTags")
      );
    });
  const previousTags = previousValue ? splitImageStateTagValue(readStateMemoryValue(previousValue)) : [];
  if (previousTags.length > 0) {
    return previousTags;
  }

  const visualProfile = state.visualProfiles.find((profile) => profile.characterId === ownerId);
  return splitImageStateTagValue(visualProfile?.defaultOutfitPrompt);
}

function shouldPreserveOutfitBase(rawValue: string, incomingTags: string[], baseTags: string[]): boolean {
  if (incomingTags.length === 0 || baseTags.length === 0) {
    return false;
  }

  if (incomingTags.some((tag) => isSameOrBroaderOutfitTag(tag, baseTags))) {
    return true;
  }

  if (!isOutfitConditionTag(rawValue) && !incomingTags.some(isOutfitConditionTag)) {
    return false;
  }

  return !hasExplicitDifferentOutfitIdentity(incomingTags, baseTags);
}

function isSameOrBroaderOutfitTag(tag: string, baseTags: string[]): boolean {
  const normalizedTag = normalizeOutfitComparisonText(tag);
  if (!isGenericOutfitTag(normalizedTag)) {
    return baseTags.some((baseTag) => {
      const normalizedBase = normalizeOutfitComparisonText(baseTag);
      return normalizedBase === normalizedTag || normalizedBase.includes(normalizedTag) || normalizedTag.includes(normalizedBase);
    });
  }

  return baseTags.some((baseTag) => normalizeOutfitComparisonText(baseTag).includes(normalizedTag));
}

function hasExplicitDifferentOutfitIdentity(incomingTags: string[], baseTags: string[]): boolean {
  const baseText = normalizeOutfitComparisonText(baseTags.join(" "));
  const incomingIdentityWords = uniquePromptTags(
    incomingTags.flatMap((tag) => normalizeOutfitComparisonText(tag).match(OUTFIT_IDENTITY_WORD_PATTERN) ?? [])
  );
  return incomingIdentityWords.length > 0 && incomingIdentityWords.some((word) => !baseText.includes(word));
}

function isOutfitConditionTag(value: string): boolean {
  return OUTFIT_CONDITION_PATTERN.test(value);
}

function isGenericOutfitTag(value: string): boolean {
  const withoutCondition = normalizeOutfitComparisonText(value).replace(OUTFIT_CONDITION_WORD_PATTERN, " ").replace(/\s+/gu, " ").trim();
  return /^(?:uniform|dress|short dress|skirt|mini skirt|shirt|blouse|jacket|cardigan|coat|hoodie|pants|shorts|suit|clothes|outfit|shoes|boots|socks|hat|cap|gloves|mask)$/iu.test(
    withoutCondition
  );
}

function normalizeOutfitComparisonText(value: string): string {
  return value.toLowerCase().replace(/[{}[\]"'`]/gu, "").replace(/[._-]+/gu, " ").replace(/\s+/gu, " ").trim();
}

function limitStateTagValue(tags: string[]): string[] {
  const limited: string[] = [];
  for (const tag of tags) {
    const next = [...limited, tag];
    if (next.length > 12 || next.join(", ").length > 220) {
      break;
    }
    limited.push(tag);
  }
  return limited.length > 0 ? limited : tags.slice(0, 1);
}

function uniquePromptTags(values: string[]): string[] {
  const seen = new Set<string>();
  const tags: string[] = [];
  for (const value of values) {
    const normalized = value.trim().replace(/\s+/gu, " ");
    const key = normalizeOutfitComparisonText(normalized);
    if (!normalized || seen.has(key)) {
      continue;
    }
    seen.add(key);
    tags.push(normalized);
  }
  return tags;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

const OUTFIT_CONDITION_PATTERN =
  /\b(?:torn|ripped|damaged|shredded|frayed|ragged|cut|slashed|wet|soaked|drenched|dirty|muddy|stained|bloodstained|dusty|wrinkled|disheveled|loose|open|unbuttoned|unfastened|burnt|scorched|clothes lifted|shirt open|panties aside)\b/iu;
const OUTFIT_CONDITION_WORD_PATTERN =
  /\b(?:torn|ripped|damaged|shredded|frayed|ragged|cut|slashed|wet|soaked|drenched|dirty|muddy|stained|bloodstained|dusty|wrinkled|disheveled|loose|open|unbuttoned|unfastened|burnt|scorched)\b/giu;
const OUTFIT_IDENTITY_WORD_PATTERN =
  /\b(?:academy|business|casual|doctor|gothic|gym|hanbok|idol|kimono|lab|lolita|maid|military|nurse|office|police|rain|sailor|school|sports|stage|track|training|wedding)\b/giu;

function extractStateRecords(state: AppState, userText: string, assistantText: string): MemoryDeltaRecord[] {
  const text = `${userText}\n${assistantText}`;
  const actors = findMentionedCharacters(state, text);
  const owner = selectPrimaryActorForMemory(state, actors);

  return STATE_PATTERNS.flatMap((definition) => {
    if (!definition.pattern.test(text)) {
      return [];
    }

    if (!owner && definition.type !== "Location") {
      return [];
    }

    const value = definition.valuePattern?.exec(text)?.[1]?.trim() ?? extractDurableSentence(text, definition.pattern);
    if (!value) {
      return [];
    }

    return [
      createRecord({
        state,
        kind: definition.type === "Goal" ? "goal" : "state",
        layer: "episodic",
        content: `${owner?.name ?? "장면"} ${definition.label}: ${value}`,
        importance: definition.importance,
        confidence: 0.72,
        actorId: owner?.id,
        actorName: owner?.name,
        ownerId: owner?.id,
        stateType: definition.type,
        value,
        tags: [owner ? "current-state" : "scene-state", definition.type],
        importanceReasons: ["state_changed", "future_consistency_relevant"]
      })
    ];
  });
}

function extractObservationAndBeliefRecords(state: AppState, userText: string, assistantText: string): MemoryDeltaRecord[] {
  const text = `${userText}\n${assistantText}`;
  const actors = findMentionedCharacters(state, text);
  const observer = actors[0];
  const target = actors.find((character) => character.id !== observer?.id);
  const records: MemoryDeltaRecord[] = [];

  if (observer && OBSERVATION_PATTERN.test(text)) {
    records.push(
      createRecord({
        state,
        kind: "observation",
        layer: "episodic",
        content: `${observer.name} 관찰: ${extractDurableSentence(text, OBSERVATION_PATTERN)}`,
        importance: 0.76,
        confidence: 0.78,
        actorId: observer.id,
        actorName: observer.name,
        targetId: target?.id,
        observers: [observer.id],
        eventType: "Observed",
        tags: ["observation", "perspective"],
        importanceReasons: ["observed_by_character", "perspective_relevant"]
      })
    );
  }

  if (observer && BELIEF_PATTERN.test(text)) {
    records.push(
      createRecord({
        state,
        kind: "belief",
        layer: "episodic",
        content: `${observer.name} 믿음/추정: ${extractDurableSentence(text, BELIEF_PATTERN)}`,
        importance: 0.68,
        confidence: 0.65,
        actorId: observer.id,
        actorName: observer.name,
        ownerId: observer.id,
        targetId: target?.id,
        eventType: "BeliefUpdated",
        tags: ["belief", "perspective"],
        importanceReasons: ["character_knowledge_changed"]
      })
    );
  }

  if (OPEN_THREAD_PATTERN.test(text)) {
    records.push(
      createRecord({
        state,
        kind: "open_thread",
        layer: "episodic",
        content: extractDurableSentence(text, OPEN_THREAD_PATTERN),
        importance: 0.74,
        confidence: 0.66,
        actorId: observer?.id,
        actorName: observer?.name,
        eventType: "OpenThread",
        tags: ["open-thread", "unresolved"],
        importanceReasons: ["unresolved_future_relevance"]
      })
    );
  }

  return records;
}

function createFallbackEventRecord(state: AppState, userText: string, assistantText: string): MemoryDeltaRecord | undefined {
  const text = `${userText}\n${assistantText}`;
  if (!IMPORTANT_EVENT_PATTERN.test(text)) {
    return undefined;
  }

  const actor = findMentionedCharacters(state, text)[0] ?? state.characters[0];
  return createRecord({
    state,
    kind: "event",
    layer: "episodic",
    content: extractDurableSentence(text, IMPORTANT_EVENT_PATTERN),
    importance: 0.7,
    confidence: 0.62,
    actorId: actor?.id,
    actorName: actor?.name,
    eventType: "SceneEvent",
    tags: ["event", "compiled-fallback"],
    importanceReasons: ["important_keyword_detected"]
  });
}

function createRecord(input: {
  state: AppState;
  kind: SimulationMemoryKind;
  layer: MemoryLayer;
  content: string;
  importance: number;
  confidence: number;
  tags: string[];
  actorId?: string;
  actorName?: string;
  ownerId?: string;
  targetId?: string;
  stateType?: string;
  value?: string;
  eventType?: string;
  observers?: string[];
  importanceReasons: string[];
}): MemoryDeltaRecord {
  const content = cleanMemoryContent(input.content);
  return {
    id: createStableMemoryRecordId({
      kind: input.kind,
      layer: input.layer,
      content,
      actorId: input.actorId,
      ownerId: input.ownerId,
      targetId: input.targetId,
      stateType: input.stateType,
      value: input.value,
      eventType: input.eventType,
      observers: input.observers
    }),
    kind: input.kind,
    layer: input.layer,
    content,
    importance: clamp(input.importance, 0, 1),
    confidence: clamp(input.confidence, 0, 1),
    tags: normalizeTags(input.tags),
    actorId: input.actorId,
    actorName: input.actorName,
    ownerId: input.ownerId,
    targetId: input.targetId,
    stateType: input.stateType,
    value: input.value,
    eventType: input.eventType,
    observers: input.observers,
    importanceReasons: input.importanceReasons
  };
}

function inferRecordKind(draft: AssistantMemoryEventDraft): {
  kind: SimulationMemoryKind;
  layer: MemoryLayer;
  stateType?: string;
  eventType?: string;
} {
  const canonicalStateType = canonicalizeStateType(draft.stateType);
  if (draft.memoryKind) {
    return {
      kind: draft.memoryKind,
      layer: draft.memoryKind === "summary" ? "semantic" : "episodic",
      stateType: canonicalStateType,
      eventType: draft.eventType
    };
  }

  if (canonicalStateType) {
    return { kind: "state", layer: "episodic", stateType: canonicalStateType };
  }

  const text = `${draft.tags.join(" ")} ${draft.content}`.toLowerCase();
  if (/state|상태|wearing|location|physical|emotion|착용|위치|복통|감정/u.test(text)) {
    return { kind: "state", layer: "episodic", stateType: draft.stateType ?? inferStateType(text) };
  }
  if (/observ|목격|봤|들었|눈치/u.test(text)) {
    return { kind: "observation", layer: "episodic", eventType: "Observed" };
  }
  if (/belief|믿|의심|추측|생각/u.test(text)) {
    return { kind: "belief", layer: "episodic", eventType: "BeliefUpdated" };
  }
  if (/goal|목표|의도|약속/u.test(text)) {
    return { kind: "goal", layer: "episodic", eventType: "GoalUpdated" };
  }
  if (/open|unresolved|아직|미해결|원인/u.test(text)) {
    return { kind: "open_thread", layer: "episodic", eventType: "OpenThread" };
  }

  return { kind: "event", layer: "episodic", eventType: draft.eventType ?? "SceneEvent" };
}

function inferStateType(text: string): string | undefined {
  return STATE_PATTERNS.find((definition) => definition.pattern.test(text))?.type;
}

function inferStateValue(content: string, stateType: string | undefined): string | undefined {
  if (!stateType) {
    return undefined;
  }

  const definition = STATE_PATTERNS.find((item) => item.type === stateType);
  return definition?.valuePattern?.exec(content)?.[1]?.trim();
}

function looksLikeRawAssistantOutput(content: string, assistantText: string): boolean {
  const normalizedContent = normalizeMemoryText(content);
  const normalizedAssistant = normalizeMemoryText(assistantText);
  return normalizedContent.length > 260 && normalizedAssistant.startsWith(normalizedContent.slice(0, 180));
}

function cleanMemoryContent(value: string): string {
  return value
    .replace(/```[\s\S]*?```/gu, " ")
    .replace(/::[a-z_]+\[[^\]]*\]/giu, " ")
    .replace(/\s+/gu, " ")
    .trim()
    .slice(0, 420);
}

function extractDurableSentence(text: string, pattern: RegExp): string {
  const sentence =
    text
      .replace(/\r/gu, "\n")
      .split(/(?<=[.!?。！？])\s+|\n+/u)
      .map((line) => cleanMemoryContent(line))
      .find((line) => pattern.test(line) && line.length >= 6) ??
    cleanMemoryContent(text);

  return sentence.slice(0, 260);
}

function findMentionedCharacters(state: AppState, text: string): AppState["characters"] {
  return state.characters.filter((character) => characterIsReferencedInText(character, text));
}

function selectPrimaryActorForMemory(
  state: AppState,
  actors: AppState["characters"]
): AppState["characters"][number] | undefined {
  if (actors.length > 0) {
    return actors[0];
  }

  return state.characters.length === 1 ? state.characters[0] : undefined;
}

function normalizeObserverIds(
  state: AppState,
  observerIds: string[] | undefined,
  fallbackActorId: string | undefined,
  kind: SimulationMemoryKind
): string[] | undefined {
  const knownCharacterIds = new Set(state.characters.map((character) => character.id));
  const ids = [
    ...(observerIds ?? []),
    kind === "observation" && fallbackActorId ? fallbackActorId : undefined
  ].filter((id): id is string => Boolean(id && knownCharacterIds.has(id)));

  return ids.length > 0 ? [...new Set(ids)] : undefined;
}

function resolveCharacter(state: AppState, id?: string, name?: string) {
  return state.characters.find(
    (character) =>
      character.id === id ||
      character.name === name ||
      character.name.toLowerCase() === name?.toLowerCase()
  );
}

function findPreviousStateEvent(state: AppState, record: MemoryDeltaRecord): MemoryEvent | undefined {
  const ownerId = record.ownerId ?? record.actorId;
  if (!ownerId || !record.stateType) {
    return undefined;
  }

  return state.memoryEvents
    .slice()
    .reverse()
    .find((event) => {
      const metadata = event.metadata ?? {};
      return (
        readString(metadata.memory_kind) === "state" &&
        readString(metadata.owner_id) === ownerId &&
        readString(metadata.state_type) === record.stateType &&
        readString(metadata.valid_to) === undefined
      );
    });
}

function inferSimulationTime(assistantText: string, fallbackIso: string): string {
  const statusMatch = assistantText.match(/(?:Day\s*\d+|D\s*\+\s*\d+|[0-9]+일차|날짜|시간)[:\s|,-]*([^\n]{1,32})/iu);
  return statusMatch?.[0]?.trim() ?? fallbackIso;
}

function inferSceneId(state: AppState, userText: string, assistantText: string): string {
  const location =
    STATE_PATTERNS.find((definition) => definition.type === "Location")?.valuePattern?.exec(`${userText}\n${assistantText}`)?.[1] ??
    state.simulation.activeSessionId;
  return `scene:${state.simulation.activeSessionId}:${slugify(location)}`;
}

function formatRecordContent(state: AppState, record: MemoryDeltaRecord): string {
  const actorName = resolveCharacter(state, record.actorId, record.actorName)?.name ?? record.actorName;
  if (record.kind === "state") {
    const owner = resolveCharacter(state, record.ownerId ?? record.actorId, record.actorName)?.name ?? actorName ?? "대상";
    return `[State] ${owner} ${record.stateType ?? "State"} = ${record.value ?? record.content}`;
  }
  if (record.kind === "observation") {
    return `[Observation] ${actorName ?? "누군가"}: ${record.content}`;
  }
  if (record.kind === "belief") {
    return `[Belief] ${actorName ?? "누군가"}: ${record.content}`;
  }
  if (record.kind === "open_thread") {
    return `[OpenThread] ${record.content}`;
  }
  if (record.kind === "goal") {
    return `[Goal] ${actorName ?? "대상"}: ${record.content}`;
  }
  return `[Event] ${record.content}`;
}

interface ReadableMemoryRecord {
  kind: SimulationMemoryKind;
  content: string;
  actorId?: string;
  ownerId?: string;
  targetId?: string;
  actorName?: string;
  stateType?: string;
  value?: string;
  simTime?: string;
  sceneId?: string;
  importance: number;
}

function toReadableMemoryRecord(state: AppState, event: MemoryEvent): ReadableMemoryRecord | undefined {
  const metadata = event.metadata ?? {};
  const runtimeKind = readStateMemoryKind(event);
  const kind = readMemoryKind(metadata) ?? (isMemoryKind(runtimeKind) ? runtimeKind : inferKindFromTags(event.tags));
  const actor = resolveCharacter(state, event.actorId, event.actorName);
  return {
    kind,
    content: event.content,
    actorId: event.actorId,
    ownerId: readStateMemoryOwnerId(event) ?? event.actorId,
    targetId: readString(event.metadata?.target_id),
    actorName: actor?.name ?? event.actorName,
    stateType: readStateMemoryStateType(event),
    value: readStateMemoryValue(event),
    simTime: readString(metadata.sim_time),
    sceneId: readString(metadata.scene_id),
    importance: event.importance
  };
}

function recordIsRelevantToCurrentSceneCast(
  state: AppState,
  record: ReadableMemoryRecord,
  activeCharacterIds: ReadonlySet<string>
): boolean {
  const characterIds = getRecordCharacterIds(state, record);
  if (characterIds.length === 0) {
    return true;
  }

  if (state.characters.length === 1 && characterIds.includes(state.characters[0].id)) {
    return true;
  }

  return characterIds.some((characterId) => activeCharacterIds.has(characterId));
}

function getRecordCharacterIds(state: AppState, record: ReadableMemoryRecord): string[] {
  const knownCharacterIds = new Set(state.characters.map((character) => character.id));
  return [...new Set([record.ownerId, record.actorId, record.targetId].filter((id): id is string => Boolean(id)))].filter((id) =>
    knownCharacterIds.has(id)
  );
}

function formatRecordLines(state: AppState, records: ReadableMemoryRecord[], fallback: string): string[] {
  if (records.length === 0) {
    return [`- ${fallback}`];
  }

  return records.map((record) => {
    const owner = resolveCharacter(state, record.ownerId ?? record.actorId, record.actorName)?.name ?? record.actorName;
    const prefix = record.simTime ? `${record.simTime}: ` : "";
    if (record.kind === "state") {
      return `- ${owner ?? "대상"} ${record.stateType ?? "State"}: ${record.value ?? stripRecordPrefix(record.content)}`;
    }
    return `- ${prefix}${stripRecordPrefix(record.content)}`;
  });
}

function formatKnowledgeLines(state: AppState, observations: ReadableMemoryRecord[], beliefs: ReadableMemoryRecord[]): string[] {
  const byActor = new Map<string, string[]>();
  for (const record of [...observations, ...beliefs]) {
    const actor = resolveCharacter(state, record.actorId, record.actorName);
    const name = actor?.name ?? record.actorName ?? "Unknown";
    byActor.set(name, [...(byActor.get(name) ?? []), stripRecordPrefix(record.content)]);
  }

  if (byActor.size === 0) {
    return ["- 캐릭터별 관찰/믿음 기록 없음"];
  }

  return [...byActor.entries()].flatMap(([name, lines]) => [
    `### ${name}`,
    ...lines.slice(-4).map((line) => `- ${line}`)
  ]);
}

function latestByKey<T>(items: T[], keyOf: (item: T) => string): T[] {
  const map = new Map<string, T>();
  for (const item of items) {
    map.set(keyOf(item), item);
  }

  return [...map.values()];
}

function stripRecordPrefix(value: string): string {
  return value.replace(/^\[(?:Event|State|Observation|Belief|OpenThread|Goal)\]\s*/u, "");
}

function readMemoryKind(metadata: Record<string, unknown>): SimulationMemoryKind | undefined {
  const value = readString(metadata.memory_kind);
  return isMemoryKind(value) ? value : undefined;
}

function inferKindFromTags(tags: string[]): SimulationMemoryKind {
  const kindTag = tags.find((tag) => tag.startsWith("kind:"))?.slice("kind:".length);
  return isMemoryKind(kindTag) ? kindTag : "event";
}

function isMemoryKind(value: string | undefined): value is SimulationMemoryKind {
  return Boolean(value && ["event", "state", "observation", "belief", "goal", "relationship", "open_thread", "summary"].includes(value));
}

function readMemoryKey(event: MemoryEvent): string | undefined {
  return readString(event.metadata?.memory_key);
}

export function createMemoryRecordKey(record: MemoryDeltaRecord): string {
  return normalizeMemoryText(
    [
      record.kind,
      record.ownerId ?? record.actorId,
      record.stateType,
      record.value,
      record.eventType,
      record.content
    ]
      .filter(Boolean)
      .join(":")
  ).slice(0, 180);
}

function createStableMemoryRecordId(input: {
  kind: SimulationMemoryKind;
  layer: MemoryLayer;
  content: string;
  actorId?: string;
  ownerId?: string;
  targetId?: string;
  stateType?: string;
  value?: string;
  eventType?: string;
  observers?: string[];
}): string {
  const key = normalizeMemoryText(
    [
      input.kind,
      input.layer,
      input.actorId,
      input.ownerId,
      input.targetId,
      input.stateType,
      input.value,
      input.eventType,
      ...(input.observers ?? []),
      input.content
    ]
      .filter(Boolean)
      .join(":")
  );
  return `memrec_${hashText(key)}`;
}

function hashText(value: string): string {
  let hash = 5381;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 33) ^ value.charCodeAt(index);
  }
  return (hash >>> 0).toString(36);
}

function normalizeMemoryText(value: string): string {
  return value.toLowerCase().replace(/\s+/gu, " ").replace(/[^\p{L}\p{N}:_-]+/gu, "").trim();
}

function normalizeTags(tags: Array<string | undefined>): string[] {
  return [...new Set(tags.map((tag) => tag?.trim().toLowerCase().replace(/\s+/gu, "-")).filter((tag): tag is string => Boolean(tag)))].slice(0, 12);
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function slugify(value: string): string {
  return value.trim().toLowerCase().replace(/[^\p{L}\p{N}_-]+/gu, "-").replace(/^-+|-+$/gu, "").slice(0, 40) || "current";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}
