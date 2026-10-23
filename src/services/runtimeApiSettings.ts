import type {
  ApiRegistrationStatus,
  CharacterProfileImage,
  CharacterProfileImages,
  ImageBackendId,
  LlmApiSettings,
  NovelAiApiSettings,
  NovelAiModelPreset,
  PhotorealApiSettings,
  PhotorealModelFiles,
  PhotorealStylePresetId,
  PhotorealWorkflowPresetId
} from "../types";
import { PHOTOREAL_MIN_CFG, snapPhotorealSide } from "./comfyWorkflows";

export function toShareableLlmSettings(settings: LlmApiSettings): LlmApiSettings {
  return {
    ...settings,
    apiKey: "",
    registrationStatus: "idle",
    verifiedAt: undefined,
    verificationMessage: ""
  };
}

export function toShareableNovelAiSettings(settings: NovelAiApiSettings): NovelAiApiSettings {
  return {
    ...settings,
    apiKey: "",
    registrationStatus: "idle",
    verifiedAt: undefined,
    verificationMessage: "",
    subscriptionTier: undefined,
    // Strip heavy base64 payloads: the raw image data URL and the pre-encoded vibe
    // blob can be several MB each and must not be included in server-bound state
    // bodies or simulation exports. The actual payloads live in the personal vault
    // and are re-injected at runtime by applyPersonalApiVault.
    vibeTransferReferences: settings.vibeTransferReferences.map(({ image: _image, encodedVibe: _encodedVibe, ...rest }) => ({
      ...rest,
      image: ""
    }))
  };
}

// Record-shaped so adding a member to the union without listing it here is a compile error, not a value that
// normalization silently resets to the default.
const PHOTOREAL_WORKFLOW_PRESET_IDS: Record<PhotorealWorkflowPresetId, true> = {
  qwen_image_2_1_turbo: true,
  qwen_image_2_1: true,
  z_image_turbo: true,
  qwen_image_edit_2511: true,
  custom: true
};
const PHOTOREAL_STYLE_PRESET_IDS: Record<PhotorealStylePresetId, true> = {
  instagram: true,
  daily_look: true,
  film_portrait: true,
  cinematic: true,
  studio_editorial: true,
  custom: true
};
const API_REGISTRATION_STATUSES: Record<ApiRegistrationStatus, true> = {
  idle: true,
  verifying: true,
  registered: true,
  failed: true
};
const NOVELAI_MODEL_PRESETS: Record<NovelAiModelPreset, true> = {
  NAID5F: true,
  NAID5C: true,
  NAID5: true,
  "NAID4.5F": true,
  "NAID4.5C": true,
  "NAID4.0F": true,
  "NAID4.0C": true,
  NAID3: true
};
const PHOTOREAL_MODEL_FILE_FIELDS: Array<keyof PhotorealModelFiles> = ["diffusionModel", "textEncoder", "vae"];
const PROFILE_IMAGE_BACKENDS: ImageBackendId[] = ["novelai", "photoreal"];

/**
 * Photoreal (local ComfyUI) settings every simulation starts from: off, so nothing changes for anyone who has not
 * set up ComfyUI. Defined here rather than in data/seed.ts because the built-in seeds need it while their modules
 * are still evaluating, and seed.ts imports womanLifeSeed.ts — a definition in seed.ts would be a circular import
 * that hands womanLifeSeed an uninitialised binding. Frozen (modelFiles included) because spreads share it: a stray
 * `settings.modelFiles[preset] = ...` would otherwise rewrite the default for every simulation created afterwards.
 */
export const DEFAULT_PHOTOREAL_SETTINGS: Readonly<PhotorealApiSettings> = Object.freeze({
  enabled: false,
  provider: "comfyui",
  requestMode: "proxy",
  baseUrl: "http://127.0.0.1:8188",
  proxyUrl: "",
  workflowPreset: "z_image_turbo",
  referenceWorkflowPreset: "qwen_image_2_1_turbo",
  modelFiles: Object.freeze({}),
  steps: 0,
  cfg: 0,
  width: 0,
  height: 0,
  seedFixed: false,
  // daily_look asked for a phone photo with lifted shadows and grain; people wanted the sharp, flattering
  // influencer-feed look the name promised, so new simulations start from that.
  stylePreset: "instagram",
  stylePrompt: "",
  appearancePrompt: "",
  negativePrompt: "",
  // Off until chosen: a profile image used as the reference held every later cut to that one photo's hair and
  // makeup, which read as "the character's look is frozen" rather than as a convenience.
  faceReferenceEnabled: false,
  vramHandoff: "auto",
  customWorkflow: "",
  timeoutSeconds: 300,
  registrationStatus: "idle"
});

/**
 * Coerces anything — a state saved before photoreal existed, a hand-edited state.json, a draft from an older
 * builder — into settings every consumer can trust without re-checking.
 *
 * This runs inside hydrateState, which App runs on EVERY state write, so it has to be idempotent and must not fight
 * a field the user is typing into: strings are kept verbatim (no trimming; an emptied field stays empty rather than
 * snapping back to the default) and only enums and numbers are corrected. The numeric clamps do apply immediately,
 * so a settings input for width/height/timeout needs a local draft committed on blur — bound directly, typing
 * "1024" would clamp to 512 at the first keystroke.
 */
export function normalizePhotorealSettings(candidate: unknown): PhotorealApiSettings {
  const source = isPlainRecord(candidate) ? candidate : {};
  const defaults = DEFAULT_PHOTOREAL_SETTINGS;
  const seed = normalizePhotorealSeed(source.seed);

  return {
    enabled: readBoolean(source.enabled, defaults.enabled),
    provider: "comfyui",
    requestMode: source.requestMode === "proxy" || source.requestMode === "mock" ? source.requestMode : defaults.requestMode,
    baseUrl: readString(source.baseUrl, defaults.baseUrl),
    proxyUrl: readString(source.proxyUrl, defaults.proxyUrl),
    workflowPreset: isPhotorealWorkflowPresetId(source.workflowPreset) ? source.workflowPreset : defaults.workflowPreset,
    referenceWorkflowPreset: isPhotorealWorkflowPresetId(source.referenceWorkflowPreset)
      ? source.referenceWorkflowPreset
      : defaults.referenceWorkflowPreset,
    modelFiles: normalizePhotorealModelFiles(source.modelFiles),
    // 0 means "the preset's own value" for steps and cfg, so 0 stays in range.
    steps: Math.round(clampNumber(source.steps, 0, 100, defaults.steps)),
    cfg: normalizePhotorealCfg(source.cfg, defaults.cfg),
    width: normalizePhotorealDimension(source.width),
    height: normalizePhotorealDimension(source.height),
    seedFixed: readBoolean(source.seedFixed, defaults.seedFixed),
    ...(seed === undefined ? {} : { seed }),
    stylePreset: isPhotorealStylePresetId(source.stylePreset) ? source.stylePreset : defaults.stylePreset,
    stylePrompt: readString(source.stylePrompt, defaults.stylePrompt),
    appearancePrompt: readString(source.appearancePrompt, defaults.appearancePrompt),
    negativePrompt: readString(source.negativePrompt, defaults.negativePrompt),
    faceReferenceEnabled: readBoolean(source.faceReferenceEnabled, defaults.faceReferenceEnabled),
    vramHandoff: source.vramHandoff === "auto" || source.vramHandoff === "off" ? source.vramHandoff : defaults.vramHandoff,
    customWorkflow: readString(source.customWorkflow, defaults.customWorkflow),
    // The relay waits at most 900 s, so a longer setting would only ever end in the server's timeout.
    timeoutSeconds: Math.round(clampNumber(source.timeoutSeconds, 30, 900, defaults.timeoutSeconds)),
    registrationStatus: isApiRegistrationStatus(source.registrationStatus) ? source.registrationStatus : defaults.registrationStatus,
    ...(typeof source.verifiedAt === "string" ? { verifiedAt: source.verifiedAt } : {}),
    ...(typeof source.verificationMessage === "string" ? { verificationMessage: source.verificationMessage } : {})
  };
}

/**
 * Photoreal settings hold no secret (ComfyUI runs locally without a key), so the shareable form only drops the
 * connection check: it describes this machine's ComfyUI at the time of the check, not the simulation being exported.
 */
export function toShareablePhotorealSettings(settings: PhotorealApiSettings): PhotorealApiSettings {
  const { verifiedAt: _verifiedAt, verificationMessage: _verificationMessage, ...rest } = settings;
  return {
    ...rest,
    modelFiles: { ...rest.modelFiles },
    registrationStatus: "idle"
  };
}

/**
 * The persisted form of a character's profile images: object-store references and their metadata, nothing else.
 * Built field by field instead of spread, so bytes that rode in on an entry (a candidate's data URL, a letterboxed
 * canvas) can never be written into state.json or the browser cache. An entry needs an asset id and an object key
 * that is a store path; anything else is dropped, and only the two known backends are kept — the key, not the
 * entry's own `backend` field, is what the pipeline reads, so the field is rewritten to match it.
 */
export function toShareableCharacterProfileImages(candidate: unknown): CharacterProfileImages | undefined {
  if (!isPlainRecord(candidate)) {
    return undefined;
  }

  const profileImages: CharacterProfileImages = {};
  for (const backend of PROFILE_IMAGE_BACKENDS) {
    const image = toShareableCharacterProfileImage(candidate[backend], backend);
    if (image) {
      profileImages[backend] = image;
    }
  }

  return Object.keys(profileImages).length > 0 ? profileImages : undefined;
}

function toShareableCharacterProfileImage(candidate: unknown, backend: ImageBackendId): CharacterProfileImage | undefined {
  if (!isPlainRecord(candidate)) {
    return undefined;
  }

  const { assetId, objectKey } = candidate;
  if (!isObjectStoreName(assetId) || !isObjectStoreKey(objectKey)) {
    return undefined;
  }

  return {
    backend,
    assetId,
    objectKey,
    ...(typeof candidate.mimeType === "string" && /^image\/[\w.+-]+$/u.test(candidate.mimeType) ? { mimeType: candidate.mimeType } : {}),
    ...(isPositiveNumber(candidate.width) ? { width: candidate.width } : {}),
    ...(isPositiveNumber(candidate.height) ? { height: candidate.height } : {}),
    ...(isObjectStoreKey(candidate.referenceObjectKey) ? { referenceObjectKey: candidate.referenceObjectKey } : {}),
    prompt: typeof candidate.prompt === "string" ? candidate.prompt : "",
    ...(backend === "photoreal" && typeof candidate.photoPrompt === "string" && candidate.photoPrompt.trim()
      ? { photoPrompt: candidate.photoPrompt }
      : {}),
    ...(typeof candidate.negativePrompt === "string" ? { negativePrompt: candidate.negativePrompt } : {}),
    ...(typeof candidate.seed === "number" && Number.isFinite(candidate.seed) && candidate.seed >= 0 ? { seed: candidate.seed } : {}),
    ...(isPhotorealWorkflowPresetId(candidate.workflowPreset) ? { workflowPreset: candidate.workflowPreset } : {}),
    ...(isNovelAiModelPreset(candidate.modelPreset) ? { modelPreset: candidate.modelPreset } : {}),
    selectedAt: typeof candidate.selectedAt === "string" ? candidate.selectedAt : ""
  };
}

// The server writes objects as `<simulationId>/<assetId>.<ext>`. A colon (data:, http:, C:), a backslash, a control
// character or an empty/dot segment never occurs in such a key, so rejecting them keeps inline bytes, remote URLs and
// path tricks out of a field the relay later resolves against the object store.
function isObjectStoreKey(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= 512 &&
    !/[:\\\u0000-\u001f\u007f]/u.test(value) &&
    value.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..")
  );
}

function isObjectStoreName(value: unknown): value is string {
  return isObjectStoreKey(value) && value.length <= 256 && !value.includes("/");
}

function normalizePhotorealModelFiles(candidate: unknown): PhotorealApiSettings["modelFiles"] {
  const modelFiles: PhotorealApiSettings["modelFiles"] = {};
  if (!isPlainRecord(candidate)) {
    return modelFiles;
  }

  for (const presetId of Object.keys(PHOTOREAL_WORKFLOW_PRESET_IDS) as PhotorealWorkflowPresetId[]) {
    const entry = candidate[presetId];
    if (!isPlainRecord(entry)) {
      continue;
    }

    const files: PhotorealModelFiles = {};
    for (const field of PHOTOREAL_MODEL_FILE_FIELDS) {
      const value = entry[field];
      if (typeof value === "string") {
        files[field] = value;
      }
    }
    if (Object.keys(files).length > 0) {
      modelFiles[presetId] = files;
    }
  }

  return modelFiles;
}

// 0 = derive the size from imageProfile's aspect ratio. Anything else is stored exactly as resolvePhotorealOutputSize
// will render it (multiples of 16, 512-2048 a side): the settings field shows the stored value as the size in effect,
// so a bound of its own here would put a size on screen that no cut is ever rendered at.
function normalizePhotorealDimension(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return 0;
  }

  return snapPhotorealSide(value);
}

/**
 * 0 (and anything at or below it) = the preset's own CFG; otherwise PHOTOREAL_MIN_CFG..20. Never between 0 and 1: a
 * guidance scale below 1 draws the negative prompt instead of keeping it out (see PHOTOREAL_MIN_CFG), so a stored
 * 0.5 — typed before the settings field refused it, or hand-edited into state.json — is held at 1 here, the weakest
 * guidance there is. The graph builder clamps again, whatever it is handed.
 */
function normalizePhotorealCfg(value: unknown, fallback: number): number {
  const cfg = clampNumber(value, 0, 20, fallback);
  return cfg > 0 ? Math.max(PHOTOREAL_MIN_CFG, cfg) : 0;
}

function normalizePhotorealSeed(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return undefined;
  }

  return Math.min(Math.floor(value), Number.MAX_SAFE_INTEGER);
}

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;
}

function readBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function readString(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function isPositiveNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function isPhotorealWorkflowPresetId(value: unknown): value is PhotorealWorkflowPresetId {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(PHOTOREAL_WORKFLOW_PRESET_IDS, value);
}

function isPhotorealStylePresetId(value: unknown): value is PhotorealStylePresetId {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(PHOTOREAL_STYLE_PRESET_IDS, value);
}

function isApiRegistrationStatus(value: unknown): value is ApiRegistrationStatus {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(API_REGISTRATION_STATUSES, value);
}

function isNovelAiModelPreset(value: unknown): value is NovelAiModelPreset {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(NOVELAI_MODEL_PRESETS, value);
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
