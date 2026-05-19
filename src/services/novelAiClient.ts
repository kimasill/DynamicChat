import { unzip } from "fflate";

import type { AppState, ImageCue, NovelAiModelPreset } from "../types";
import { getNovelAiGenerateProxyUrl } from "./dynamicChatApi";
import { createImageUserRulesForContentRating, isAdultContentMode } from "./contentRating";
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
