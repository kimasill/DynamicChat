import type { AppState, Character, ChatMessage, Id } from "../types";

interface SceneCastState {
  characters: Character[];
  messages: ChatMessage[];
  imageAssets?: AppState["imageAssets"];
  imageJobs?: AppState["imageJobs"];
  turnTraces?: AppState["turnTraces"];
  userPersona?: AppState["userPersona"];
}

interface InferSceneCastOptions {
  recentMessageLimit?: number;
  includePersonaCharacter?: boolean;
}

const DEFAULT_RECENT_MESSAGE_LIMIT = 6;

// A scene has a handful of people in it; a character-select menu, cast list, or roster recap names most of
// the roster at once. Those listings are reference data, so they must not put the whole roster on-stage —
// that floods the turn prompt with every character's full profile and buries the actual scene.
const ROSTER_LISTING_MIN_MENTIONS = 5;
const ROSTER_LISTING_ROSTER_RATIO = 0.6;

function messageLooksLikeRosterListing(state: SceneCastState, messageText: string): boolean {
  const rosterSize = state.characters.length;
  if (rosterSize < ROSTER_LISTING_MIN_MENTIONS) {
    return false;
  }

  const mentioned = state.characters.filter((character) =>
    characterIsReferencedInText(character, messageText)
  ).length;

  return (
    mentioned >= ROSTER_LISTING_MIN_MENTIONS &&
    mentioned >= Math.ceil(rosterSize * ROSTER_LISTING_ROSTER_RATIO)
  );
}

export function inferCurrentSceneCharacterIds(
  state: SceneCastState,
  currentText = "",
  options: InferSceneCastOptions = {}
): Id[] {
  const recentMessageLimit = options.recentMessageLimit ?? DEFAULT_RECENT_MESSAGE_LIMIT;
  const recentMessages = state.messages.slice(-recentMessageLimit);
  const sceneEvidenceTexts = recentMessages
    .map((message) => sanitizeSceneCastEvidenceText(message.content))
    .filter((messageText) => !messageLooksLikeRosterListing(state, messageText));
  const text = [sanitizeSceneCastEvidenceText(currentText), ...sceneEvidenceTexts].join("\n");
  const currentTextMentionedIds = state.characters
    .filter((character) => characterIsReferencedInText(character, sanitizeSceneCastEvidenceText(currentText)))
    .map((character) => character.id);
  const mentionedIds = state.characters
    .filter((character) => characterIsReferencedInText(character, text))
    .map((character) => character.id);
  const personaCharacterId =
    options.includePersonaCharacter === false ? undefined : resolvePersonaCharacterId(state);
  const imageContinuityIds = shouldUseImageCastContinuity(currentText, currentTextMentionedIds)
    ? collectRecentImageCharacterIds(state, recentMessages)
    : [];

  return uniqueStrings([
    ...(personaCharacterId ? [personaCharacterId] : []),
    ...mentionedIds,
    ...imageContinuityIds
  ]);
}

export function getCurrentSceneCharacters(
  state: SceneCastState,
  currentText = "",
  options: InferSceneCastOptions = {}
): Character[] {
  const activeIds = new Set(inferCurrentSceneCharacterIds(state, currentText, options));
  return state.characters.filter((character) => activeIds.has(character.id));
}

export function createSceneCastPromptBlock(state: SceneCastState, currentText = ""): string {
  const activeCharacters = getCurrentSceneCharacters(state, currentText);
  const activeNames = activeCharacters.map((character) => character.name).join(", ");
  const baseRule =
    "Registered roster, relationship map entries, current moods, and stored status values are reference data, not automatic evidence that a character is present in the current scene.";
  const activationRule =
    "Only include or mention a roster character when the recent transcript, current user action, selected scene rule, or explicit user instruction puts that character on-stage.";

  return activeNames
    ? `Current scene cast inferred from recent transcript/current action: ${activeNames}. ${baseRule} ${activationRule}`
    : `Current scene cast inferred from recent transcript/current action: none explicit. ${baseRule} ${activationRule}`;
}

export function characterIsReferencedInText(character: Character, text: string): boolean {
  return [character.name, character.id].some((term) => containsSceneCastTerm(text, term));
}

function resolvePersonaCharacterId(state: SceneCastState): Id | undefined {
  const persona = state.userPersona;
  if (!persona?.enabled || persona.source !== "character" || !persona.characterId) {
    return undefined;
  }

  return state.characters.some((character) => character.id === persona.characterId)
    ? persona.characterId
    : undefined;
}

function collectRecentImageCharacterIds(state: SceneCastState, recentMessages: ChatMessage[]): Id[] {
  const validCharacterIds = new Set(state.characters.map((character) => character.id));
  const recentMessageIds = new Set(recentMessages.map((message) => message.id));
  const recentImageAssetIds = new Set(recentMessages.flatMap((message) => message.imageAssetIds ?? []));
  const characterIds: Id[] = [];

  for (const trace of state.turnTraces ?? []) {
    if (recentMessageIds.has(trace.userMessageId) || recentMessageIds.has(trace.assistantMessageId)) {
      characterIds.push(...trace.imageCue.characters);
      trace.imageAssetIds.forEach((assetId) => recentImageAssetIds.add(assetId));
    }
  }

  for (const job of state.imageJobs ?? []) {
    if (!recentMessageIds.has(job.turnId)) {
      continue;
    }

    characterIds.push(...readImageJobCueCharacterIds(job.providerPayload));
    job.assetIds.forEach((assetId) => recentImageAssetIds.add(assetId));
  }

  for (const asset of state.imageAssets ?? []) {
    if (recentImageAssetIds.has(asset.id)) {
      characterIds.push(...asset.characterIds);
    }
  }

  return uniqueStrings(characterIds.filter((characterId) => validCharacterIds.has(characterId)));
}

function readImageJobCueCharacterIds(payload: Record<string, unknown> | undefined): Id[] {
  const cue = payload?.cue;
  if (!cue || typeof cue !== "object" || Array.isArray(cue)) {
    return [];
  }

  const characters = (cue as Record<string, unknown>).characters;
  return Array.isArray(characters)
    ? characters.filter((value): value is string => typeof value === "string" && Boolean(value.trim()))
    : [];
}

function shouldUseImageCastContinuity(currentText: string, currentTextMentionedIds: Id[]): boolean {
  if (currentTextMentionedIds.length === 0) {
    return true;
  }

  return !/(?:장면\s*전환|다음\s*날|다음날|며칠\s*뒤|몇\s*시간\s*뒤|한편|다른\s*곳|새(?:로운)?\s*장면|scene\s*change|cut\s*to|meanwhile)/iu.test(
    currentText
  );
}

function containsSceneCastTerm(text: string, term: string): boolean {
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

function sanitizeSceneCastEvidenceText(text: string): string {
  return text
    .replace(/```(?:status|stats?|relationship|relationships|memory|neuralmap|context|hud|choice)\b[\s\S]*?```/giu, "\n")
    .split(/\r?\n/u)
    .filter((line) => !looksLikeReferenceOnlySceneCastLine(line))
    .join("\n");
}

function looksLikeReferenceOnlySceneCastLine(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) {
    return false;
  }

  return (
    /^::(?:status|memory|choice)\[/iu.test(trimmed) ||
    /^(?:[-*]\s*)?(?:관계도|관계\/상태|상태창|상태\s*요약|캐릭터\s*상태|NeuralMap|Relationship(?:\s+Map)?|Status(?:\s+Window)?|HUD)\s*[:：-]/iu.test(trimmed) ||
    /^\|.*(?:관계|상태|캐릭터|인물|Relationship|Status|Character).*\|$/iu.test(trimmed)
  );
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}
