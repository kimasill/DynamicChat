import { unzip } from "fflate";

import type { AppState, ImageCue, NovelAiModelPreset } from "../types";
import { getNovelAiEncodeVibeProxyUrl, getNovelAiGenerateProxyUrl } from "./dynamicChatApi";
import { createImageUserRulesForContentRating, isAdultContentMode, resolveEffectiveImageSafetyLevel } from "./contentRating";
import { bytesToBase64 } from "./imageReferences";
import { resolveNovelAiModelName } from "./novelAiModels";

interface NovelAiImageResult {
  dataUrls: string[];
  payload: Record<string, unknown>;
}

/**
 * A character's NovelAI profile image as a Precise Reference: the letterboxed reference canvas
 * (CharacterProfileImage.referenceObjectKey) plus the user's knobs from NovelAiApiSettings.characterReference*.
 */
export interface NovelAiCharacterReferenceInput {
  /** Raw base64 of the reference canvas. A data URL prefix is tolerated and stripped. */
  imageBase64: string;
  /** 0-1, sent as director_reference_strength_values. */
  strength: number;
  /** 0-1, sent inverted as director_reference_secondary_strength_values. */
  fidelity: number;
  /** true → base_caption "character&style", false → "character". */
  styleAware: boolean;
}

/** Fallbacks for a non-numeric knob, matching the documented defaults on NovelAiApiSettings. */
const CHARACTER_REFERENCE_DEFAULT_STRENGTH = 0.85;
const CHARACTER_REFERENCE_DEFAULT_FIDELITY = 1;

/**
 * Hard ceiling for one NovelAI generation. Without it a request that never settles (a dropped connection, or
 * NovelAI holding the socket open) leaves the image queue's promise chain pending forever, which silently
 * wedges every image for the rest of the session — the "images just stop appearing" failure.
 */
const NOVELAI_REQUEST_TIMEOUT_MS = 180_000;
/** Transient upstream conditions: concurrency limit, gateway hiccup, rate limit. */
const NOVELAI_RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);
const NOVELAI_MAX_ATTEMPTS = 3;
const NOVELAI_RETRY_BASE_DELAY_MS = 2_500;

class NovelAiRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryable: boolean
  ) {
    super(message);
    this.name = "NovelAiRequestError";
  }
}

export async function generateNovelAiImages(input: {
  state: AppState;
  prompt: string;
  negativePrompt: string;
  cue: ImageCue;
  count: number;
  signal?: AbortSignal;
  characterReference?: NovelAiCharacterReferenceInput;
}): Promise<NovelAiImageResult> {
  const { state } = input;
  const payload = createNovelAiPayload(input);
  // The reference canvas is a full PNG in base64. It has to reach NovelAI, but the payload returned from here
  // is copied into job.providerPayload and asset.providerMetadata, and from there into state.json for every
  // cut, so the returned copy only records how big it was.
  const storedPayload = omitCharacterReferenceImages(payload);
  // Disabled / mock is a DRY RUN: the assembled payload is returned with no images so the prompt pipeline
  // (and the prompt-quality evals) can be inspected without calling the provider. Nothing downstream should
  // reach here expecting pixels — shouldPlanImageJob refuses to plan a job while NovelAI is off, and says so.
  if (!state.novelAi.enabled || state.novelAi.requestMode === "mock") {
    return {
      dataUrls: [],
      payload: storedPayload
    };
  }

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

  const body = JSON.stringify(payload);
  let lastError: unknown;

  // NovelAI routinely answers a burst of requests with 429 (concurrent generation limit) or a transient 5xx.
  // A single attempt turned every one of those into a permanently failed job, which is the single largest
  // cause of an image "sometimes" not appearing. Authentication and prompt errors are NOT retried.
  for (let attempt = 1; attempt <= NOVELAI_MAX_ATTEMPTS; attempt += 1) {
    try {
      return await requestNovelAiImages(endpoint, headers, body, storedPayload, input.signal);
    } catch (error) {
      lastError = error;
      const retryable = error instanceof NovelAiRequestError ? error.retryable : isTransientNetworkError(error);
      if (!retryable || attempt === NOVELAI_MAX_ATTEMPTS || input.signal?.aborted) {
        throw error;
      }
      await delay(NOVELAI_RETRY_BASE_DELAY_MS * attempt);
    }
  }

  throw lastError instanceof Error ? lastError : new Error("NovelAI request failed.");
}

async function requestNovelAiImages(
  endpoint: string,
  headers: Record<string, string>,
  body: string,
  payload: Record<string, unknown>,
  signal?: AbortSignal
): Promise<NovelAiImageResult> {
  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body,
    signal: withRequestTimeout(signal)
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new NovelAiRequestError(
      `NovelAI request failed: ${response.status}${detail ? ` ${detail.slice(0, 500)}` : ""}`,
      response.status,
      NOVELAI_RETRYABLE_STATUS.has(response.status)
    );
  }

  const contentType = response.headers.get("content-type") ?? "";
  const arrayBuffer = await response.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);

  if (contentType.includes("zip") || looksLikeZip(bytes)) {
    return {
      dataUrls: await extractZipImages(bytes),
      payload
    };
  }

  const mimeType = contentType.includes("image/") ? contentType.split(";")[0] : "image/png";
  return {
    dataUrls: [bytesToDataUrl(bytes, mimeType)],
    payload
  };
}

/** Combines the caller's cancellation signal (if any) with the per-request timeout. */
function withRequestTimeout(signal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(NOVELAI_REQUEST_TIMEOUT_MS);
  if (!signal) {
    return timeout;
  }
  // AbortSignal.any is available in every browser that ships AbortSignal.timeout; guard anyway so an older
  // runtime degrades to timeout-only rather than throwing.
  return typeof AbortSignal.any === "function" ? AbortSignal.any([signal, timeout]) : timeout;
}

function isTransientNetworkError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  // A user-initiated cancel and a request timeout both surface as AbortError; neither should be retried
  // here (the timeout already consumed the full budget).
  if (error.name === "AbortError" || error.name === "TimeoutError") {
    return false;
  }
  return error.name === "TypeError" || /network|fetch failed|ECONNRESET|socket/iu.test(error.message);
}

function delay(durationMs: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, durationMs);
  });
}

const NOVELAI_ENCODE_VIBE_DIRECT_URL = "https://image.novelai.net/ai/encode-vibe";

/**
 * v4/v4.5 vibe transfer를 위해 참조 이미지를 NovelAI encode-vibe로 인코딩하고
 * 결과(encoded vibe)를 base64 문자열로 반환한다. (NovelAI 기준 1회 2 Anlas 과금)
 */
export async function encodeNovelAiVibe(input: {
  state: AppState;
  image: string;
  informationExtracted: number;
  signal?: AbortSignal;
}): Promise<string> {
  const { state } = input;
  const model = resolveNovelAiModelName(state.novelAi.modelPreset, state.imageProfile.model);
  const endpoint =
    state.novelAi.requestMode === "proxy"
      ? state.novelAi.proxyUrl.trim()
        ? state.novelAi.proxyUrl.replace(/\/generate-image\b.*$/u, "/encode-vibe")
        : getNovelAiEncodeVibeProxyUrl()
      : NOVELAI_ENCODE_VIBE_DIRECT_URL;
  const headers: Record<string, string> = {
    "content-type": "application/json"
  };

  if (state.novelAi.apiKey.trim()) {
    headers.authorization = `Bearer ${normalizeApiToken(state.novelAi.apiKey)}`;
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({
      image: stripImageDataUrlPrefix(input.image),
      model,
      // The official encode-vibe schema spells this field information_extracted, snake_case like every other
      // NovelAI image parameter. The camelCase key this used to send is not part of that schema.
      information_extracted: clampVibeValue(input.informationExtracted)
    }),
    signal: input.signal
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`NovelAI encode-vibe 요청 실패: ${response.status}${detail ? ` ${detail.slice(0, 500)}` : ""}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return bytesToBase64(new Uint8Array(arrayBuffer));
}

/**
 * Keeps NovelAI's emphasis syntax from swallowing a tag's trailing digits.
 *
 * NovelAI reads `<weight>::text::` as an emphasis group and, on the v4/v5 models, takes a number written
 * straight against `::` as a weight. So in `1.5::artist:bm94199::` the closer is read as part of a group
 * weighted ~94,199, the conditioning blows up, and every image comes back as pure noise — measured on
 * nai-diffusion-5-full: noise with the group as written, a clean image with `1.5::artist:bm94199 ::`.
 * Creators already wrote `pigeon666 ::` by hand for this reason; this does it for every closer, since the
 * space changes nothing else (NovelAI trims whitespace inside a tag).
 *
 * Only a CLOSING `::` is touched. An opener is the bare number that starts a group (`1.5::`, `-2::`) and
 * must keep its digits attached, or the group stops being weighted at all.
 */
export function protectEmphasisClosers(text: string): string {
  return text.replace(/(^|[,\n]|\s)(-?\d+(?:\.\d+)?)::([\s\S]*?)::/gu, (match, lead: string, weight: string, body: string) =>
    /\d$/u.test(body) ? `${lead}${weight}::${body} ::` : match
  );
}

function createNovelAiPayload(input: {
  state: AppState;
  prompt: string;
  negativePrompt: string;
  cue: ImageCue;
  count: number;
  characterReference?: NovelAiCharacterReferenceInput;
}): Record<string, unknown> {
  const { state } = input;
  const prompt = protectEmphasisClosers(input.prompt);
  const negativePrompt = protectEmphasisClosers(input.negativePrompt);
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
    negative_prompt: negativePrompt,
    cfg_rescale: state.novelAi.cfgRescale,
    ucPreset: state.novelAi.ucPreset,
    qualityToggle: Boolean(state.imageProfile.qualityPrompt),
    params_version: 3,
    legacy: false,
    legacy_v3_extend: false,
    safety_level: resolveEffectiveImageSafetyLevel(state),
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

  const characterReference = normalizeCharacterReference(input.characterReference);
  const supportsCharacterReference = isNovelAiV45Model(model);
  if (characterReference && !supportsCharacterReference) {
    // Precise Reference exists only on V4.5. On any other model the cut is drawn without the face reference,
    // and the payload records why so the inspector can explain a character that drifted.
    baseParameters.dynamicchat_character_reference_skipped = "model";
  }
  const attachedReference = supportsCharacterReference ? characterReference : undefined;

  if (state.novelAi.vibeTransferEnabled) {
    const vibeParameters = createVibeTransferParameters(state, model);
    if (vibeParameters) {
      if (isNovelAiV5Model(model)) {
        // V5 rejects vibe transfer, so the arrays never go out to a V5 model.
        baseParameters.dynamicchat_vibe_transfer_skipped = "model";
      } else if (attachedReference) {
        // NovelAI does not combine Vibe Transfer with Precise Reference in one request. The face reference is
        // what keeps the character recognisable, so it wins and the vibes sit this cut out.
        baseParameters.dynamicchat_vibe_transfer_skipped = "character_reference";
      } else {
        Object.assign(baseParameters, vibeParameters);
      }
    }
  }

  if (attachedReference) {
    Object.assign(baseParameters, createCharacterReferenceParameters(attachedReference));
  }

  if (model.includes("nai-diffusion-4") || model.includes("nai-diffusion-5")) {
    const cueCharacterPrompts = resolveNovelAiCharacterPrompts(input.cue);
    const characterCaptions: Array<{ char_caption: string; centers: Array<{ x: number; y: number }> }> = cueCharacterPrompts
      .map((characterPrompt, index, prompts) => ({
        char_caption: protectEmphasisClosers(characterPrompt.prompt),
        centers: [characterPrompt.center ?? createDefaultCharacterCenter(index, prompts.length)]
      }))
      .filter((caption) => caption.char_caption.trim());
    const characterNegativeCaptions: Array<{ char_caption: string; centers: Array<{ x: number; y: number }> }> = cueCharacterPrompts
      .map((characterPrompt, index, prompts) => ({
        char_caption: protectEmphasisClosers(characterPrompt.negativePrompt ?? ""),
        centers: [characterPrompt.center ?? createDefaultCharacterCenter(index, prompts.length)]
      }))
      .filter((caption) => caption.char_caption.trim());
    const useCharacterCoords = characterCaptions.length > 1;
    baseParameters.autoSmea = true;
    baseParameters.prefer_brownian = true;
    baseParameters.use_coords = useCharacterCoords;
    baseParameters.legacy_uc = false;
    baseParameters.use_order = true;
    baseParameters.v4_prompt = {
      caption: {
        base_caption: prompt,
        char_captions: characterCaptions
      },
      use_coords: useCharacterCoords,
      use_order: true,
      legacy_uc: false
    };
    baseParameters.v4_negative_prompt = {
      caption: {
        base_caption: negativePrompt,
        char_captions: characterNegativeCaptions
      },
      use_coords: useCharacterCoords,
      use_order: false,
      legacy_uc: false
    };
  }

  return {
    action: "generate",
    input: prompt,
    model,
    parameters: baseParameters
  };
}

/** The vibe arrays this model family takes, or undefined when no reference is usable. */
function createVibeTransferParameters(state: AppState, model: string): Record<string, unknown> | undefined {
  if (model.includes("nai-diffusion-4") || model.includes("nai-diffusion-5")) {
    // v4/v4.5/v5는 encode-vibe로 사전 인코딩한 vibe만 받는다. information extracted는 인코딩 단계에 반영된다.
    const references = state.novelAi.vibeTransferReferences.filter((reference) => reference.encodedVibe?.trim());
    if (references.length === 0) {
      return undefined;
    }
    return {
      reference_image_multiple: references.map((reference) => reference.encodedVibe!.trim()),
      reference_strength_multiple: references.map((reference) => clampVibeValue(reference.referenceStrength))
    };
  }

  // v3는 원본 이미지와 information extracted를 직접 전송한다.
  const references = state.novelAi.vibeTransferReferences.filter((reference) => stripImageDataUrlPrefix(reference.image));
  if (references.length === 0) {
    return undefined;
  }
  return {
    reference_image_multiple: references.map((reference) => stripImageDataUrlPrefix(reference.image)),
    reference_strength_multiple: references.map((reference) => clampVibeValue(reference.referenceStrength)),
    reference_information_extracted_multiple: references.map((reference) => clampVibeValue(reference.informationExtracted))
  };
}

function isNovelAiV45Model(model: string): boolean {
  return model === "nai-diffusion-4-5-full" || model === "nai-diffusion-4-5-curated";
}

function isNovelAiV5Model(model: string): boolean {
  return model.startsWith("nai-diffusion-5");
}

function normalizeCharacterReference(reference?: NovelAiCharacterReferenceInput): NovelAiCharacterReferenceInput | undefined {
  const imageBase64 = typeof reference?.imageBase64 === "string" ? stripImageDataUrlPrefix(reference.imageBase64) : "";
  if (!reference || !imageBase64) {
    return undefined;
  }
  return {
    imageBase64,
    // A NaN knob would otherwise clamp to 0 and silently send a reference with no effect while still costing Anlas.
    strength: clampUnitOr(reference.strength, CHARACTER_REFERENCE_DEFAULT_STRENGTH),
    fidelity: clampUnitOr(reference.fidelity, CHARACTER_REFERENCE_DEFAULT_FIDELITY),
    styleAware: reference.styleAware === true
  };
}

/**
 * Precise Reference is five parallel arrays with one entry per reference image, so they must always stay the
 * same length. information_extracted stays 1: Precise Reference has no such control, its knobs are strength
 * and fidelity (sent inverted as the secondary strength).
 */
function createCharacterReferenceParameters(reference: NovelAiCharacterReferenceInput): Record<string, unknown> {
  return {
    director_reference_images: [reference.imageBase64],
    director_reference_descriptions: [
      {
        caption: {
          base_caption: reference.styleAware ? "character&style" : "character",
          char_captions: []
        },
        legacy_uc: false
      }
    ],
    director_reference_information_extracted: [1],
    director_reference_strength_values: [reference.strength],
    director_reference_secondary_strength_values: [roundToHundredths(1 - reference.fidelity)]
  };
}

function omitCharacterReferenceImages(payload: Record<string, unknown>): Record<string, unknown> {
  const parameters = payload.parameters;
  if (!isRecord(parameters) || !Array.isArray(parameters.director_reference_images)) {
    return payload;
  }
  return {
    ...payload,
    parameters: {
      ...parameters,
      director_reference_images: parameters.director_reference_images.map((image) =>
        typeof image === "string" ? `<omitted:${countBase64Bytes(image)} bytes>` : image
      )
    }
  };
}

function countBase64Bytes(base64: string): number {
  const compact = base64.replace(/\s+/gu, "");
  const padding = compact.endsWith("==") ? 2 : compact.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((compact.length * 3) / 4) - padding);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function resolveNovelAiCharacterPrompts(cue: ImageCue): NonNullable<ImageCue["characterPrompts"]> {
  const explicitPrompts = cue.characterPrompts?.filter((prompt) => prompt.prompt.trim()) ?? [];
  if (explicitPrompts.length > 0) {
    return explicitPrompts;
  }

  const allTags = Array.from(new Set([...(cue.baseTags ?? []), ...(cue.tags ?? [])]));
  const characterTags = allTags
    .flatMap((tag) => tag.split(","))
    .map((tag) => tag.trim())
    .filter(isNovelAiCharacterPromptTag);
  if (characterTags.length === 0) {
    return [];
  }

  const uniqueTags = Array.from(new Set(characterTags));
  const characterIds = cue.characters.length > 0 ? cue.characters : [undefined];
  return characterIds.map((characterId, index) => ({
    characterId,
    prompt: uniqueTags.join(", "),
    center: createDefaultCharacterCenter(index, characterIds.length)
  }));
}

function isNovelAiCharacterPromptTag(tag: string): boolean {
  const normalized = tag.toLowerCase().replace(/[{}[\]"'`]/gu, "").replace(/[._-]+/gu, " ").trim();
  return /\b(?:hair|eyes?|twintails|twin tails|ponytail|braid|glasses|freckles|scar|beauty mark|horns?|tail|ears?|uniform|school uniform|skirt|pencil skirt|pleated skirt|necktie|ribbon|shirt|blouse|jacket|cardigan|dress|suit|coat|raincoat|sweater|pants|shorts|panties|bra|shoes|sneakers|boots|socks|thighhighs|naked|topless|bottomless|clothes lifted|panties aside|shirt open|tight fit|smile|open mouth|closed mouth|crying|tears?|blush|sweat|worried|tense|nervous|surprised|angry|sad|defiant|half closed eyes|closed eyes)\b/iu.test(normalized) ||
    /^(?:standing|sitting|lying|kneeling|crouching|leaning|bending|arm up|arms up|hand up|hands up|spread legs|legs apart|crossed arms|body focus|face focus|chest focus|thigh focus|leg focus|arm focus|back focus)$/iu.test(normalized);
}

function createDefaultCharacterCenter(index: number, total: number): { x: number; y: number } {
  return {
    x: total <= 1 ? 0.5 : (index + 1) / (total + 1),
    y: 0.5
  };
}

function stripImageDataUrlPrefix(image: string): string {
  return image.trim().replace(/^data:[^;]*;base64,/iu, "").trim();
}

function clampVibeValue(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

function clampUnitOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : fallback;
}

function roundToHundredths(value: number): number {
  return Math.round(value * 100) / 100;
}

function getVarPlusSigma(preset: NovelAiModelPreset): number {
  return preset === "NAID5F" || preset === "NAID5C" || preset === "NAID5" || preset === "NAID4.5F" || preset === "NAID4.5C" ? 58 : 19;
}

function normalizeApiToken(value: string): string {
  return value.trim().replace(/^Bearer\s+/iu, "").trim();
}

async function extractZipImages(bytes: Uint8Array): Promise<string[]> {
  const files = await unzipImages(bytes);
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

function unzipImages(bytes: Uint8Array): Promise<Record<string, Uint8Array>> {
  return new Promise((resolve, reject) => {
    unzip(bytes, (error, files) => {
      if (error) {
        reject(error);
        return;
      }

      resolve(files);
    });
  });
}

function looksLikeZip(bytes: Uint8Array): boolean {
  return bytes[0] === 0x50 && bytes[1] === 0x4b;
}

// Same bytes out as FileReader.readAsDataURL, but this also runs in Node, where FileReader does not exist and
// where the evals exercise the real request path against stub servers.
function bytesToDataUrl(bytes: Uint8Array, mimeType: string): string {
  return `data:${mimeType};base64,${bytesToBase64(bytes)}`;
}
