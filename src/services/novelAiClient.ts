import { unzipSync } from "fflate";

import type { AppState, CharacterVisualProfile, ImageCue, NovelAiModelPreset } from "../types";
import { getNovelAiGenerateProxyUrl } from "./dynamicChatApi";
import { createImageUserRulesForContentRating, isAdultContentMode } from "./contentRating";
import { createCurrentCharacterImageStatePrompt } from "./imageStateTags";
import { resolveNovelAiModelName } from "./novelAiModels";

interface NovelAiImageResult {
  dataUrls: string[];
  payload: Record<string, unknown>;
}

export async function generateNovelAiImages(input: {
  state: AppState;
  prompt: string;
  negativePrompt: string;
  cue: ImageCue;
  count: number;
}): Promise<NovelAiImageResult> {
  const { state } = input;
  if (!state.novelAi.enabled || state.novelAi.requestMode === "mock") {
    return {
      dataUrls: [],
      payload: createNovelAiPayload(input)
    };
  }

  const payload = createNovelAiPayload(input);
  const endpoint =
    state.novelAi.requestMode === "proxy"
      ? state.novelAi.proxyUrl.trim() || getNovelAiGenerateProxyUrl()
      : state.novelAi.endpoint;
  const headers: Record<string, string> = {
    "content-type": "application/json"
  };

  if (state.novelAi.apiKey.trim()) {
    headers.authorization = `Bearer ${normalizeApiToken(state.novelAi.apiKey)}`;
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`NovelAI request failed: ${response.status}`);
  }

  const contentType = response.headers.get("content-type") ?? "";
  const arrayBuffer = await response.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);

  if (contentType.includes("zip") || looksLikeZip(bytes)) {
    return {
      dataUrls: extractZipImages(bytes),
      payload
    };
  }

  const mimeType = contentType.includes("image/") ? contentType.split(";")[0] : "image/png";
  return {
    dataUrls: [bytesToDataUrl(bytes, mimeType)],
    payload
  };
}

function createNovelAiPayload(input: {
  state: AppState;
  prompt: string;
  negativePrompt: string;
  cue: ImageCue;
  count: number;
}): Record<string, unknown> {
  const { state } = input;
  const model = resolveNovelAiModelName(state.novelAi.modelPreset, state.imageProfile.model);
  const seed = state.novelAi.seedFixed ? state.novelAi.seed : state.novelAi.seed ?? Math.floor(Math.random() * 4_294_967_295);
  const imageUserRules = createImageUserRulesForContentRating(state);
  const baseParameters: Record<string, unknown> = {
    width: state.imageProfile.width,
    height: state.imageProfile.height,
    scale: state.imageProfile.promptGuidance,
    steps: state.imageProfile.steps,
    sampler: state.novelAi.sampler,
    noise_schedule: state.novelAi.noiseSchedule,
    n_samples: 1,
    seed,
    extra_noise_seed: seed,
    negative_prompt: input.negativePrompt,
    cfg_rescale: state.novelAi.cfgRescale,
    ucPreset: state.novelAi.ucPreset,
    qualityToggle: Boolean(state.imageProfile.qualityPrompt),
    params_version: 3,
    legacy: false,
    legacy_v3_extend: false,
    safety_level: state.imageProfile.safetyLevel,
    dynamicchat_scene: input.cue.scene,
    dynamicchat_tags: input.cue.tags,
    dynamicchat_visual_context: input.cue.visualContext,
    dynamicchat_user_rules: imageUserRules,
    dynamicchat_content_rating: state.simulation.contentRating,
    dynamicchat_adult_content_mode: isAdultContentMode(state),
    dynamicchat_payload_version: "dynamicchat-novelai-v1"
  };

  if (state.novelAi.varPlus) {
    baseParameters.skip_cfg_above_sigma = getVarPlusSigma(state.novelAi.modelPreset);
  }

  if (model.includes("nai-diffusion-4")) {
    const characterCaptions = createNovelAiCharacterCaptions(state, input.cue, "positive");
    const characterNegativeCaptions = createNovelAiCharacterCaptions(state, input.cue, "negative");
    const useCharacterCoords = characterCaptions.length > 1;
    baseParameters.autoSmea = true;
    baseParameters.prefer_brownian = true;
    baseParameters.use_coords = useCharacterCoords;
    baseParameters.legacy_uc = false;
    baseParameters.use_order = true;
    baseParameters.v4_prompt = {
      caption: {
        base_caption: input.prompt,
        char_captions: characterCaptions
      },
      use_coords: useCharacterCoords,
      use_order: true,
      legacy_uc: false
    };
    baseParameters.v4_negative_prompt = {
      caption: {
        base_caption: input.negativePrompt,
        char_captions: characterNegativeCaptions
      },
      use_coords: useCharacterCoords,
      use_order: false,
      legacy_uc: false
    };
  }

  return {
    action: "generate",
    input: input.prompt,
    model,
    parameters: baseParameters
  };
}

function getVarPlusSigma(preset: NovelAiModelPreset): number {
  return preset === "NAID4.5F" || preset === "NAID4.5C" ? 58 : 19;
}

function createNovelAiCharacterCaptions(
  state: AppState,
  cue: ImageCue,
  kind: "positive" | "negative"
): Array<{ char_caption: string; centers: Array<{ x: number; y: number }> }> {
  const characterIds = cue.characters.slice(0, 6);
  const centers = resolveCharacterCenters(characterIds.length);
  return characterIds
    .map((characterId, index) => {
      const visualProfile = state.visualProfiles.find((candidate) => candidate.characterId === characterId);
      const caption =
        kind === "positive"
          ? uniquePromptParts([
              detectCharacterCaptionSubjectTag(state, characterId, cue),
              sanitizeCharacterCaptionPositivePrompt(visualProfile?.positivePrompt),
              visualProfile ? selectCharacterOutfitPrompt(state, visualProfile, cue) : undefined,
              selectCharacterExpressionPrompt(cue),
              createCurrentCharacterImageStatePrompt(state, characterId)
            ]).join(", ")
          : uniquePromptParts([
              visualProfile?.negativePrompt
            ]).join(", ");

      return {
        char_caption: caption,
        centers: [centers[index] ?? { x: 0.5, y: 0.5 }]
      };
    })
    .filter((caption): caption is { char_caption: string; centers: Array<{ x: number; y: number }> } => Boolean(caption));
}

function detectCharacterCaptionSubjectTag(state: AppState, characterId: string, cue: ImageCue): "girl" | "boy" | "other" {
  const character = state.characters.find((candidate) => candidate.id === characterId);
  const visualProfile = state.visualProfiles.find((candidate) => candidate.characterId === characterId);
  const text = [
    character?.name,
    character?.role,
    character?.summary,
    visualProfile?.displayName,
    visualProfile?.positivePrompt,
    cue.tags.join(" ")
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (/\b(?:girl|female|woman|schoolgirl|breasts?)\b|여학생|소녀|여자|여성|가슴|[a-z]컵/u.test(text)) {
    return "girl";
  }
  if (/\b(?:boy|male|man|schoolboy)\b|남학생|소년|남자|남성/u.test(text)) {
    return "boy";
  }
  return "other";
}

function selectCharacterOutfitPrompt(state: AppState, profile: CharacterVisualProfile, cue: ImageCue): string | undefined {
  return (
    selectCurrentCharacterOutfitPrompt(state, profile.characterId, profile.outfitPrompts) ??
    selectMappedCharacterOutfitPrompt(profile.outfitPrompts, cue) ??
    extractExplicitOutfitPrompt(cue) ??
    inferCharacterDynamicOutfitPrompt(cue) ??
    normalizeCharacterOutfitPrompt(profile.defaultOutfitPrompt)
  );
}

function selectMappedCharacterOutfitPrompt(outfitPrompts: Record<string, string> | undefined, cue: ImageCue): string | undefined {
  const cueText = normalizeCharacterOutfitKeywordText(createCueText(cue));
  const rawCueText = normalizeCharacterOutfitKeywordText([cue.scene, cue.visualContext, cue.tags.join(" ")].filter(Boolean).join(" "));
  if (hasSchoolCharacterOutfitCue(rawCueText)) {
    const schoolPrompt = Object.entries(outfitPrompts ?? {}).find(([key]) => isSchoolCharacterOutfitKeyword(key))?.[1];
    const normalizedSchoolPrompt = normalizeCharacterOutfitPrompt(schoolPrompt);
    if (normalizedSchoolPrompt) {
      return normalizedSchoolPrompt;
    }
  }
  const matches = Object.entries(outfitPrompts ?? {})
    .map(([key, value], index) => {
      const prompt = normalizeCharacterOutfitPrompt(value);
      const score = prompt ? scoreCharacterOutfitKeywordMatch(key, cueText) : 0;
      return prompt && score > 0 ? { prompt, score, index } : undefined;
    })
    .filter((match): match is { prompt: string; score: number; index: number } => Boolean(match))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, 2)
    .map((match) => match.prompt);

  if (matches.length === 0 && hasSchoolCharacterOutfitCue(cueText)) {
    const schoolPrompt = Object.entries(outfitPrompts ?? {}).find(([key]) => isSchoolCharacterOutfitKeyword(key))?.[1];
    const normalizedSchoolPrompt = normalizeCharacterOutfitPrompt(schoolPrompt);
    if (normalizedSchoolPrompt) {
      return normalizedSchoolPrompt;
    }
  }

  return matches.length > 0 ? uniquePromptParts(matches).slice(0, 8).join(", ") : undefined;
}

function scoreCharacterOutfitKeywordMatch(key: string, cueText: string): number {
  const directScore = scoreTranslatedCharacterOutfitKeywordMatch(key, cueText);
  if (directScore > 0) {
    return directScore;
  }

  return key
    .split(/[,;\/|]+/u)
    .map((phrase) => phrase.trim())
    .filter(Boolean)
    .filter((phrase) => !/^(?:default|기본|base|fallback)$/iu.test(phrase))
    .reduce((score, phrase) => {
      const alias = createCharacterOutfitKeywordAliases(phrase).find((candidate) => cueText.includes(candidate));
      return alias ? Math.max(score, 1 + Math.min(3, alias.length / 8)) : score;
    }, 0);
}

function scoreTranslatedCharacterOutfitKeywordMatch(key: string, cueText: string): number {
  if (isSchoolCharacterOutfitKeyword(key) && hasSchoolCharacterOutfitCue(cueText)) {
    return 3;
  }
  const rules: Array<{ pattern: RegExp; cuePattern: RegExp; score: number }> = [
    { pattern: /학교|교복/u, cuePattern: /\b(?:school|classroom)\b|school uniform/u, score: 3 },
    { pattern: /연습|훈련|댄스/u, cuePattern: /\b(?:practice room|training|dance studio|workout)\b/u, score: 3 },
    { pattern: /무대|공연|아이돌/u, cuePattern: /\b(?:stage|performance|idol)\b/u, score: 3 },
    { pattern: /숙소|기숙|방|아파트|오피스텔/u, cuePattern: /\b(?:dormitory|dorm|apartment|room)\b/u, score: 2.6 },
    { pattern: /비|폭우|우산|우비/u, cuePattern: /\b(?:rain|rainy|storm|umbrella|raincoat)\b/u, score: 2.8 },
    { pattern: /겨울|눈|추운/u, cuePattern: /\b(?:winter|snow|cold)\b/u, score: 2.8 },
    { pattern: /전투|싸움|갑옷/u, cuePattern: /\b(?:battle|fight|combat|armor)\b/u, score: 2.8 },
    { pattern: /무도회|파티|정장|드레스/u, cuePattern: /\b(?:ball|party|formal|dress|suit)\b/u, score: 2.8 },
    { pattern: /실험실|연구실/u, cuePattern: /\b(?:lab|laboratory)\b/u, score: 2.7 },
    { pattern: /병원|진료/u, cuePattern: /\b(?:hospital|clinic)\b/u, score: 2.7 },
    { pattern: /카페|거리|외출/u, cuePattern: /\b(?:cafe|street|outing)\b/u, score: 2.4 }
  ];
  return rules.find((rule) => rule.pattern.test(key) && rule.cuePattern.test(cueText))?.score ?? 0;
}

function createCharacterOutfitKeywordAliases(phrase: string): string[] {
  const normalizedPhrase = normalizeCharacterOutfitKeywordText(phrase);
  return normalizedPhrase ? [normalizedPhrase] : [];
}

function hasSchoolCharacterOutfitCue(value: string): boolean {
  return /\b(?:school|classroom|student)\b|학교|교복|교실/u.test(value.normalize("NFC"));
}

function isSchoolCharacterOutfitKeyword(key: string): boolean {
  return /\b(?:school|classroom|student)\b|학교|교복|교실/iu.test(normalizeCharacterOutfitKeywordText(key));
}

function inferCharacterDynamicOutfitPrompt(cue: ImageCue): string | undefined {
  const context = createCueText(cue);
  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /교복|학교|교실|\b(?:school|classroom|student)\b/iu, tags: ["school uniform"] },
    { pattern: /연습실|댄스|훈련|\b(?:practice room|dance studio|training|workout)\b/iu, tags: ["training clothes", "sneakers"] },
    { pattern: /무대|공연|아이돌|\b(?:stage|performance|idol)\b/iu, tags: ["idol stage outfit"] },
    { pattern: /전투|싸움|갑옷|\b(?:battle|fight|combat|armor)\b/iu, tags: ["combat outfit"] },
    { pattern: /비|폭우|우산|우비|\b(?:rain|rainy|storm|umbrella|raincoat)\b/iu, tags: ["raincoat"] },
    { pattern: /겨울|눈|추운|\b(?:winter|snow|cold)\b/iu, tags: ["winter coat", "scarf"] },
    { pattern: /무도회|파티|정장|드레스|\b(?:ball|party|formal|dress|suit)\b/iu, tags: ["formal outfit"] },
    { pattern: /실험실|연구실|\b(?:lab|laboratory)\b/iu, tags: ["lab coat"] },
    { pattern: /병원|진료|\b(?:hospital|clinic)\b/iu, tags: ["medical coat"] },
    { pattern: /카페|거리|외출|\b(?:cafe|street|outing)\b/iu, tags: ["casual outfit"] },
    { pattern: /숙소|기숙|방|아파트|오피스텔|\b(?:dorm|apartment|bedroom|room)\b/iu, tags: ["casual clothes"] },
    { pattern: /잠옷|취침|침대|\b(?:pajamas|sleepwear|bed)\b/iu, tags: ["pajamas"] }
  ];
  const tags = uniquePromptParts(rules.flatMap((rule) => (rule.pattern.test(context) ? rule.tags : [])));
  return tags.length > 0 ? tags.slice(0, 6).join(", ") : undefined;
}

function selectCurrentCharacterOutfitPrompt(state: AppState, characterId: string, outfitPrompts?: Record<string, string>): string | undefined {
  const event = state.memoryEvents
    .slice()
    .reverse()
    .find((candidate) => {
      const metadata = candidate.metadata ?? {};
      const kind = readMetadataString(metadata, "memory_kind") ?? candidate.tags.find((tag) => tag.startsWith("kind:"))?.slice("kind:".length);
      const stateType = readMetadataString(metadata, "state_type") ?? candidate.tags.find((tag) => tag.startsWith("state:"))?.slice("state:".length);
      const ownerId = readMetadataString(metadata, "owner_id") ?? candidate.actorId;
      return kind === "state" && ownerId === characterId && Boolean(stateType && /^(?:Wearing|OutfitTags)$/iu.test(stateType));
    });

  const currentText = readMetadataString(event?.metadata, "value") ?? event?.content;
  if (!currentText) {
    return undefined;
  }

  const mappedCurrentOutfit = selectMappedCharacterOutfitPrompt(outfitPrompts, {
    shouldGenerate: true,
    reason: "current character outfit",
    characters: [characterId],
    tags: [],
    scene: currentText,
    visualContext: currentText
  });

  return mappedCurrentOutfit ?? normalizeCharacterOutfitPrompt(currentText);
}

function readMetadataString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeCharacterOutfitPrompt(value: string | undefined): string | undefined {
  const stripped = value
    ?.replace(/^\[(?:State|Event|Observation|Belief|OpenThread|Goal)\]\s*/u, "")
    .trim();
  const explicitValue = stripped?.match(/(?:Wearing|OutfitTags|착용|의상\s*태그|outfit tags?)\s*[=:：]\s*(.+)$/iu)?.[1]?.trim();
  const tags = uniquePromptParts([explicitValue || stripped]);
  return tags.length > 0 ? tags.slice(0, 8).join(", ") : undefined;
}

function normalizeCharacterOutfitKeywordText(value: string): string {
  return value
    .normalize("NFC")
    .toLowerCase()
    .replace(/[._-]+/gu, " ")
    .replace(/[^\p{L}\p{N} ]+/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
}

function selectCharacterExpressionPrompt(cue: ImageCue): string | undefined {
  const context = createCueText(cue);
  const rules: Array<{ pattern: RegExp; prompt: string }> = [
    { pattern: /relieved|smile|happy|joy|안도|웃|기쁨|행복/u, prompt: "soft smile" },
    { pattern: /tense|worried|anxious|fear|갈등|불안|걱정|긴장/u, prompt: "tense expression" },
    { pattern: /cry|teary|tear|울|눈물/u, prompt: "teary eyes" },
    { pattern: /angry|upset|분노|화남|짜증/u, prompt: "frustrated expression" },
    { pattern: /surprise|shock|놀람|당황/u, prompt: "surprised expression" }
  ];
  return rules.find((rule) => rule.pattern.test(context))?.prompt;
}

function sanitizeCharacterCaptionPositivePrompt(value: string | undefined): string | undefined {
  const parts = uniquePromptParts(value?.split(",") ?? []);
  const filtered = parts.filter((part) => !isStaleCharacterCaptionContext(part));
  return filtered.length > 0 ? filtered.join(", ") : undefined;
}

function isStaleCharacterCaptionContext(value: string): boolean {
  const normalized = value.toLowerCase().replace(/[._-]+/gu, " ").trim();
  return /\b(?:rain|rainy|storm|umbrella|raincoat|wet hair|wet clothes|damp|food|meal|hamburger|burger|rice|dining table|table|kitchen|restaurant|cafe|library|archive|bookshelf|street|room|hallway|classroom|stage|studio)\b/iu.test(
    normalized
  );
}

function createCueText(cue: ImageCue): string {
  return [cue.scene, cue.tags.join(" "), cue.visualContext].filter(Boolean).join(" ").toLowerCase();
}

function extractExplicitOutfitPrompt(cue: ImageCue): string | undefined {
  const rawContext = [cue.visualContext, cue.scene, cue.tags.join(", ")].filter(Boolean).join(" ");
  const match = rawContext.match(
    /(?:wearing|dressed in|outfit|clothing|costume|attire)[:\s]+([^.;\n]{3,90})/iu
  ) ?? rawContext.match(/(?:의상|복장|옷차림)[:：\s]+([^.\n]{2,90})/u);
  const value = match?.[1]?.trim();
  if (!value || /[\u3131-\uD79D]/u.test(value)) {
    return undefined;
  }

  const tags = uniquePromptParts([value]);
  return tags.length > 0 ? tags.slice(0, 5).join(", ") : undefined;
}

function resolveCharacterCenters(count: number): Array<{ x: number; y: number }> {
  if (count <= 1) {
    return [{ x: 0.5, y: 0.5 }];
  }

  return Array.from({ length: count }, (_, index) => ({
    x: roundCoordinate(0.2 + (0.6 * index) / Math.max(1, count - 1)),
    y: 0.5
  }));
}

function roundCoordinate(value: number): number {
  return Math.min(0.9, Math.max(0.1, Number(value.toFixed(2))));
}

function uniquePromptParts(values: Array<string | undefined>): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    for (const part of splitPromptParts(value ?? "")) {
      const normalized = part.trim().replace(/\s+/gu, " ");
      const key = normalized.toLowerCase();
      if (normalized && !seen.has(key)) {
        seen.add(key);
        result.push(normalized);
      }
    }
  }
  return result;
}

function splitPromptParts(value: string): string[] {
  const parts: string[] = [];
  let current = "";
  let inNovelAiEmphasis = false;

  for (let index = 0; index < value.length; index += 1) {
    const rest = value.slice(index);
    const opening = !inNovelAiEmphasis ? rest.match(/^(?:-?\d+(?:\.\d+)?\s*)?::/u)?.[0] : undefined;
    if (opening) {
      current += opening;
      index += opening.length - 1;
      inNovelAiEmphasis = true;
      continue;
    }

    if (inNovelAiEmphasis && rest.startsWith("::")) {
      current += "::";
      index += 1;
      inNovelAiEmphasis = false;
      continue;
    }

    const char = value[index];
    if (!inNovelAiEmphasis && /[,;\n]+/u.test(char)) {
      const trimmed = current.trim();
      if (trimmed) {
        parts.push(trimmed);
      }
      current = "";
      continue;
    }

    current += char;
  }

  const trimmed = current.trim();
  if (trimmed) {
    parts.push(trimmed);
  }

  return parts;
}

function normalizeApiToken(value: string): string {
  return value.trim().replace(/^Bearer\s+/iu, "").trim();
}

function extractZipImages(bytes: Uint8Array): string[] {
  const files = unzipSync(bytes);
  return Object.entries(files)
    .filter(([name]) => /\.(png|webp|jpg|jpeg)$/iu.test(name))
    .map(([name, data]) => {
      const mimeType = name.endsWith(".webp")
        ? "image/webp"
        : name.endsWith(".jpg") || name.endsWith(".jpeg")
          ? "image/jpeg"
          : "image/png";
      return bytesToDataUrl(data, mimeType);
    });
}

function looksLikeZip(bytes: Uint8Array): boolean {
  return bytes[0] === 0x50 && bytes[1] === 0x4b;
}

function bytesToDataUrl(bytes: Uint8Array, mimeType: string): string {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }

  return `data:${mimeType};base64,${btoa(binary)}`;
}
