import type { AppState, Character, ChatMessage, Id } from "../types";

interface SceneCastState {
  characters: Character[];
  messages: ChatMessage[];
  userPersona?: AppState["userPersona"];
}

interface InferSceneCastOptions {
  recentMessageLimit?: number;
  includePersonaCharacter?: boolean;
}

const DEFAULT_RECENT_MESSAGE_LIMIT = 6;

export function inferCurrentSceneCharacterIds(
  state: SceneCastState,
  currentText = "",
  options: InferSceneCastOptions = {}
): Id[] {
  const recentMessageLimit = options.recentMessageLimit ?? DEFAULT_RECENT_MESSAGE_LIMIT;
  const text = [
    currentText,
    ...state.messages.slice(-recentMessageLimit).map((message) => message.content)
  ].join("\n");
  const mentionedIds = state.characters
    .filter((character) => characterIsReferencedInText(character, text))
    .map((character) => character.id);
  const personaCharacterId =
    options.includePersonaCharacter === false ? undefined : resolvePersonaCharacterId(state);

  return uniqueStrings([
    ...(personaCharacterId ? [personaCharacterId] : []),
    ...mentionedIds
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

  return normalizedText.includes(normalizedTerm);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}
