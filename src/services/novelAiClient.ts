import { unzip } from "fflate";

import type { AppState, ImageCue, NovelAiModelPreset } from "../types";
import { getNovelAiEncodeVibeProxyUrl, getNovelAiGenerateProxyUrl } from "./dynamicChatApi";
import { createImageUserRulesForContentRating, isAdultContentMode, resolveEffectiveImageSafetyLevel } from "./contentRating";
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
    const detail = await response.text().catch(() => "");
    throw new Error(`NovelAI request failed: ${response.status}${detail ? ` ${detail.slice(0, 500)}` : ""}`);
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
    dataUrls: [await bytesToDataUrl(bytes, mimeType)],
    payload
  };
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
      // NovelAI encode-vibe는 camelCase 필드를 사용한다.
      informationExtracted: clampVibeValue(input.informationExtracted)
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

  if (state.novelAi.vibeTransferEnabled) {
    if (model.includes("nai-diffusion-4")) {
      // v4/v4.5는 encode-vibe로 사전 인코딩한 vibe만 받는다. information extracted는 인코딩 단계에 반영된다.
      const references = state.novelAi.vibeTransferReferences.filter((reference) => reference.encodedVibe?.trim());
      if (references.length > 0) {
        baseParameters.reference_image_multiple = references.map((reference) => reference.encodedVibe!.trim());
        baseParameters.reference_strength_multiple = references.map((reference) => clampVibeValue(reference.referenceStrength));
      }
    } else {
      // v3는 원본 이미지와 information extracted를 직접 전송한다.
      const references = state.novelAi.vibeTransferReferences.filter((reference) => stripImageDataUrlPrefix(reference.image));
      if (references.length > 0) {
        baseParameters.reference_image_multiple = references.map((reference) => stripImageDataUrlPrefix(reference.image));
        baseParameters.reference_strength_multiple = references.map((reference) => clampVibeValue(reference.referenceStrength));
        baseParameters.reference_information_extracted_multiple = references.map((reference) => clampVibeValue(reference.informationExtracted));
      }
    }
  }

  if (model.includes("nai-diffusion-4")) {
    const cueCharacterPrompts = resolveNovelAiCharacterPrompts(input.cue);
    const characterCaptions: Array<{ char_caption: string; centers: Array<{ x: number; y: number }> }> = cueCharacterPrompts
      .map((prompt, index, prompts) => ({
        char_caption: prompt.prompt,
        centers: [prompt.center ?? createDefaultCharacterCenter(index, prompts.length)]
      }))
      .filter((caption) => caption.char_caption.trim());
    const characterNegativeCaptions: Array<{ char_caption: string; centers: Array<{ x: number; y: number }> }> = cueCharacterPrompts
      .map((prompt, index, prompts) => ({
        char_caption: prompt.negativePrompt ?? "",
        centers: [prompt.center ?? createDefaultCharacterCenter(index, prompts.length)]
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

function resolveNovelAiCharacterPrompts(cue: ImageCue): NonNullable<ImageCue["characterPrompts"]> {
  const explicitPrompts = cue.characterPrompts?.filter((prompt) => prompt.prompt.trim()) ?? [];
  if (explicitPrompts.length > 0) {
    return explicitPrompts;
  }

  const characterTags = cue.tags
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

function getVarPlusSigma(preset: NovelAiModelPreset): number {
  return preset === "NAID4.5F" || preset === "NAID4.5C" ? 58 : 19;
}

function normalizeApiToken(value: string): string {
  return value.trim().replace(/^Bearer\s+/iu, "").trim();
}

async function extractZipImages(bytes: Uint8Array): Promise<string[]> {
  const files = await unzipImages(bytes);
  return Promise.all(Object.entries(files)
    .filter(([name]) => /\.(png|webp|jpg|jpeg)$/iu.test(name))
    .map(([name, data]) => {
      const mimeType = name.endsWith(".webp")
        ? "image/webp"
        : name.endsWith(".jpg") || name.endsWith(".jpeg")
          ? "image/jpeg"
          : "image/png";
      return bytesToDataUrl(data, mimeType);
    }));
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

function bytesToBase64(bytes: Uint8Array): Promise<string> {
  return new Promise((resolve, reject) => {
    const arrayBuffer = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuffer).set(bytes);
    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const commaIndex = result.indexOf(",");
      commaIndex >= 0
        ? resolve(result.slice(commaIndex + 1))
        : reject(new Error("NovelAI encode-vibe 응답을 base64로 변환하지 못했습니다."));
    };
    reader.onerror = () => reject(reader.error ?? new Error("NovelAI encode-vibe 응답을 읽지 못했습니다."));
    reader.readAsDataURL(new Blob([arrayBuffer], { type: "application/octet-stream" }));
  });
}

function bytesToDataUrl(bytes: Uint8Array, mimeType: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const arrayBuffer = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(arrayBuffer).set(bytes);
    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      result ? resolve(result) : reject(new Error("NovelAI image payload could not be converted to a data URL."));
    };
    reader.onerror = () => reject(reader.error ?? new Error("NovelAI image payload could not be read."));
    reader.readAsDataURL(new Blob([arrayBuffer], { type: mimeType }));
  });
}
