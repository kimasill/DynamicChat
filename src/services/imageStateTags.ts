import type { AppState } from "../types";

const CHARACTER_IMAGE_STATE_TYPES = [
  "StatusTags",
  "ExpressionTags",
  "PoseTags",
  "ActionTags",
  "InteractionTags",
  "InteractionPhaseTags",
  "HeldItemTags",
  "PhysicalStateTags",
  "BodyStateTags"
] as const;

const SCENE_IMAGE_STATE_TYPES = [
  "SceneTags",
  "ScenePhaseTags",
  "CompositionTags",
  "CameraTags",
  "LightingTags",
  "EnvironmentTags",
  "InteractionTags",
  "InteractionPhaseTags"
] as const;

const MAX_CHARACTER_IMAGE_STATE_TAGS = 18;
const MAX_SCENE_IMAGE_STATE_TAGS = 18;

export const IMAGE_STATE_TYPE_INSTRUCTION =
  "Wearing | StatusTags | ExpressionTags | PoseTags | ActionTags | InteractionTags | InteractionPhaseTags | HeldItemTags | PhysicalStateTags | BodyStateTags | SceneTags | ScenePhaseTags | CompositionTags | CameraTags | LightingTags | EnvironmentTags | Location | PhysicalCondition | Emotion | Goal";

export function createCurrentCharacterImageStatePrompt(state: AppState, characterId: string): string | undefined {
  const tags = selectCurrentImageStateTags(state, {
    ownerId: characterId,
    stateTypes: CHARACTER_IMAGE_STATE_TYPES,
    maxTags: MAX_CHARACTER_IMAGE_STATE_TAGS
  });
  return tags.length > 0 ? tags.join(", ") : undefined;
}

export function createCurrentSceneImageStateTags(state: AppState): string[] {
  return selectCurrentImageStateTags(state, {
    ownerId: undefined,
    stateTypes: SCENE_IMAGE_STATE_TYPES,
    maxTags: MAX_SCENE_IMAGE_STATE_TAGS
  });
}

export function createCurrentImageCueStateTags(state: AppState, characterIds: string[]): string[] {
  return uniqueStrings([
    ...characterIds.flatMap((characterId) =>
      selectCurrentImageStateTags(state, {
        ownerId: characterId,
        stateTypes: CHARACTER_IMAGE_STATE_TYPES,
        maxTags: 8
      })
    ),
    ...createCurrentSceneImageStateTags(state)
  ]).slice(0, 28);
}

function selectCurrentImageStateTags(
  state: AppState,
  options: {
    ownerId: string | undefined;
    stateTypes: readonly string[];
    maxTags: number;
  }
): string[] {
  const wantedTypes = new Set(options.stateTypes);
  const seenTypes = new Set<string>();
  const values: string[] = [];

  for (const event of [...state.memoryEvents].reverse()) {
    const metadata = event.metadata ?? {};
    const kind = readMetadataString(metadata, "memory_kind") ?? event.tags.find((tag) => tag.startsWith("kind:"))?.slice("kind:".length);
    const stateType = readMetadataString(metadata, "state_type") ?? event.tags.find((tag) => tag.startsWith("state:"))?.slice("state:".length);
    if (kind !== "state" || !stateType || !wantedTypes.has(stateType) || seenTypes.has(stateType)) {
      continue;
    }

    const ownerId = readMetadataString(metadata, "owner_id") ?? event.actorId;
    if (ownerId !== options.ownerId) {
      continue;
    }

    const value = readMetadataString(metadata, "value") ?? event.content;
    const tags = parseImageStateTags(value);
    if (tags.length === 0) {
      continue;
    }

    values.push(...tags);
    seenTypes.add(stateType);
    if (values.length >= options.maxTags) {
      break;
    }
  }

  return uniqueStrings(values).slice(0, options.maxTags);
}

function parseImageStateTags(value: string): string[] {
  const stripped = value
    .replace(/^\[(?:State|Event|Observation|Belief|OpenThread|Goal)\]\s*/u, "")
    .trim();
  const explicitValue =
    stripped.match(
      /(?:Wearing|OutfitTags|StatusTags|ExpressionTags|PoseTags|ActionTags|InteractionTags|InteractionPhaseTags|HeldItemTags|PhysicalStateTags|BodyStateTags|SceneTags|ScenePhaseTags|CompositionTags|CameraTags|LightingTags|EnvironmentTags)\s*(?:=|:|：)\s*(.+)$/iu
    )?.[1] ??
    stripped.match(/^[^=\n]{1,80}\s=\s(.+)$/u)?.[1];
  const cleaned = explicitValue?.trim() || stripped;
  return uniqueStrings(
    cleaned
      .split(/[,;\n|]+/u)
      .flatMap((tag) => normalizeImageStateTagParts(tag))
      .filter((tag): tag is string => Boolean(tag))
  );
}

function normalizeImageStateTagParts(value: string): string[] {
  const expanded = expandImageStateTagAlias(value);
  if (expanded) {
    return expanded
      .map((tag) => normalizeImageStateTag(tag))
      .filter((tag): tag is string => Boolean(tag));
  }

  const normalized = normalizeImageStateTag(value);
  return normalized ? [normalized] : [];
}

function expandImageStateTagAlias(value: string): string[] | undefined {
  const normalized = value
    .trim()
    .replace(/[.!?。！？]+$/gu, "")
    .replace(/[._-]+/gu, " ")
    .replace(/[^\p{L}\p{N}: ]+/gu, " ")
    .replace(/\s+/gu, " ")
    .toLowerCase();
  if (!normalized || /[\u3131-\uD79D]/u.test(normalized)) {
    return undefined;
  }

  const expressionMatch = normalized.match(/^(worried|tense|nervous|surprised|frustrated|angry|sad|happy|confident|defiant|evil)\s+expression$/u);
  if (expressionMatch?.[1]) {
    return expressionMatch[1] === "happy" ? ["smile"] : [expressionMatch[1]];
  }

  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /^(?:standing up from (?:her |his |their )?(?:seat|chair)|standing up|get(?:ting)? up)$/u, tags: ["standing", "chair"] },
    { pattern: /^(?:raising (?:her |his |their )?hand(?: eagerly)?|hand raised|raised hand)$/u, tags: ["arm up", "hand up"] },
    { pattern: /^bright and confident smile$/u, tags: ["smile", "confident"] },
    { pattern: /^classroom setting$/u, tags: ["classroom"] },
    { pattern: /^(?:close up face|close face|face close up|character close up)$/u, tags: ["close-up", "face focus"] },
    { pattern: /^(?:speaking|talking)$/u, tags: ["open mouth"] }
  ];
  const match = rules.find((rule) => rule.pattern.test(normalized));
  if (match) {
    return match.tags;
  }
  if (/^(?:filming set|props?|main action|facial expression|visible emotional reaction|situation specific clothing|wide context|clear environment|context appropriate outfit|dynamic pose|current scene|current simulation scene|generated scene|simulation scene|scene)$/u.test(normalized)) {
    return [];
  }
  return undefined;
}

function normalizeImageStateTag(value: string): string | undefined {
  const normalized = value
    .trim()
    .replace(/[.!?。！？]+$/gu, "")
    .replace(/\s+/gu, " ");
  if (!normalized || !/[\p{L}\p{N}:]/u.test(normalized) || /[\u3131-\uD79D]/u.test(normalized)) {
    return undefined;
  }
  const lower = normalized.toLowerCase();
  if (["current scene", "current simulation scene", "generated scene", "simulation scene", "scene"].includes(lower)) {
    return undefined;
  }
  return normalized;
}

function readMetadataString(metadata: Record<string, unknown>, key: string): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter((value) => Boolean(value?.trim()))));
}
