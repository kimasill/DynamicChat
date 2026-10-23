import type { AppState, PhotorealApiSettings, PhotorealModelFiles, PhotorealWorkflowPresetId } from "../types";
import { sanitizePhotorealNegativePrompt } from "./photorealPrompt";

// ComfyUI workflow presets for the photoreal backend.
//
// The graphs are built in code, in ComfyUI's API format ({ nodeId: { class_type, inputs } }, links as
// ["nodeId", outputIndex]), rather than shipped as JSON files: the shape changes with the cut (0-4 reference
// images add LoadImage nodes and encoder inputs), and every value that matters — model files, size, seed — is
// typed here instead of string-substituted into a template. Each graph mirrors the official Comfy-Org template
// for its model; only the deliberate differences are commented.
//
// Reference images never enter the graph as bytes. A LoadImage node carries the literal string `%ref:<name>%`;
// the DynamicChat server relay uploads the image to ComfyUI's input folder and substitutes the name ComfyUI
// returned. That keeps the graph small enough to store in the job payload and keeps the upload server-side.
//
// The built-in graphs end in PreviewImage, not SaveImage. SaveImage writes every cut into ComfyUI's output folder
// for good, where DynamicChat's image deletion cannot reach it — a deleted cut, or a profile candidate nobody picked,
// would live on there at full resolution with its prompt embedded. PreviewImage writes to ComfyUI's temp folder,
// which ComfyUI empties when it starts and when it shuts down, and the relay downloads from there just the same. The
// image DynamicChat keeps is the copy in its own object store. A custom workflow may still end in SaveImage.
//
// What reaches a graph is held to the photoreal floor's terms here, whoever built it. Every negative prompt is cleaned
// (sanitizePhotorealNegativePrompt) as it enters the graph: at a CFG above 1 a negative of "adult" or "breasts" steers
// the sampler toward a younger subject as surely as a positive does, and nothing upstream classifies negatives. No
// guidance scale is ever below 1 (PHOTOREAL_MIN_CFG): below 1 the sampler moves TOWARD the negative, so the negative —
// text the floor deliberately does not read — would become the picture. Every image a graph loads is a `%ref:`
// reference the relay uploads, never a file already in ComfyUI's input folder. And a custom template may only put text
// the floor never reads (the negative, the model file names, the numbers) where it cannot become the picture's prompt.
// Screening text needs the floor itself, which lives in the orchestrator: it reads the template's own literals
// (listCustomWorkflowLiteralTexts) and, because pieces that pass apart can be harmful once joined, every picture-side
// string of the graph as actually built (listPhotorealGraphPictureTexts).

export interface PhotorealPresetDownload {
  folder: "diffusion_models" | "text_encoders" | "vae" | "loras";
  file: string;
  url: string;
  sizeGb: number;
}

export interface PhotorealWorkflowPreset {
  id: PhotorealWorkflowPresetId;
  label: string;
  description: string;
  license: string;
  commercialUse: boolean;
  supportsReferences: boolean;
  maxReferences: number;
  referenceSyntax: "qwen21" | "edit_plus" | "none";
  defaults: { diffusionModel: string; textEncoder: string; vae: string; steps: number; cfg: number; sampler: string; scheduler: string };
  /**
   * The CFG a distilled model always samples at; the CFG setting does not apply to it. Distillation bakes guidance
   * into the weights, so a Turbo model above 1 only burns the image — and at exactly 1 ComfyUI never evaluates the
   * negative branch, which keeps a steering negative inert on these presets.
   */
  fixedCfg?: number;
  /** class_types the graph uses (for the /object_info probe). */
  requiredNodes: string[];
  downloads: PhotorealPresetDownload[];
  /** Korean. */
  vramNote: string;
}

const HF_QWEN_IMAGE_21 = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main";
const HF_Z_IMAGE_TURBO = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files";
const HF_QWEN_IMAGE = "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files";
const HF_QWEN_IMAGE_EDIT = "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files";

const QWEN21_TEXT_ENCODER_DOWNLOAD: PhotorealPresetDownload = {
  folder: "text_encoders",
  file: "qwen3vl_8b_int8_convrot.safetensors",
  url: `${HF_QWEN_IMAGE_21}/text_encoders/qwen3vl_8b_int8_convrot.safetensors`,
  sizeGb: 9.35
};
const QWEN21_VAE_DOWNLOAD: PhotorealPresetDownload = {
  folder: "vae",
  file: "qwen_image_2.1_vae_bf16.safetensors",
  url: `${HF_QWEN_IMAGE_21}/vae/qwen_image_2.1_vae_bf16.safetensors`,
  sizeGb: 0.68
};

const QWEN21_NODES = [
  "UNETLoader",
  "QwenImage21Cache",
  "CLIPLoader",
  "VAELoader",
  "TextEncodeQwenImage21",
  "EmptyLatentImage",
  "LoadImage",
  "ImageCrop",
  "VAEDecode",
  "PreviewImage"
];

export const PHOTOREAL_WORKFLOW_PRESETS: Record<PhotorealWorkflowPresetId, PhotorealWorkflowPreset> = {
  qwen_image_2_1_turbo: {
    id: "qwen_image_2_1_turbo",
    label: "Qwen-Image 2.1 Turbo (실사 + 얼굴 고정)",
    description:
      "8스텝 실사 생성과 얼굴 참조를 한 모델로 처리합니다. 캐릭터의 실사 프로필 이미지를 <image1>로 넣어 같은 얼굴을 유지합니다.",
    license: "Qwen Research (비상업)",
    commercialUse: false,
    supportsReferences: true,
    // The model takes up to 10; each reference adds prefix tokens and KV cache, and 4 is what a 16 GB card holds.
    maxReferences: 4,
    referenceSyntax: "qwen21",
    defaults: {
      diffusionModel: "qwen_image_2.1_turbo_int8_convrot.safetensors",
      textEncoder: "qwen3vl_8b_int8_convrot.safetensors",
      vae: "qwen_image_2.1_vae_bf16.safetensors",
      steps: 8,
      cfg: 1,
      sampler: "euler",
      scheduler: "simple"
    },
    fixedCfg: 1,
    requiredNodes: [...QWEN21_NODES, "CFGGuider", "KSamplerSelect", "RandomNoise", "ManualSigmas", "SamplerCustomAdvanced"],
    downloads: [
      {
        folder: "diffusion_models",
        file: "qwen_image_2.1_turbo_int8_convrot.safetensors",
        url: `${HF_QWEN_IMAGE_21}/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors`,
        sizeGb: 7.26
      },
      QWEN21_TEXT_ENCODER_DOWNLOAD,
      QWEN21_VAE_DOWNLOAD
    ],
    vramNote:
      "int8 모델 7.3GB + Qwen3-VL 인코더 9.4GB, 최대 약 15GB를 씁니다. 16GB 카드에서는 Ollama 모델과 동시에 올라가지 않으니 GPU 교대(auto)를 켜 두세요. 참조 이미지가 늘수록 메모리를 더 씁니다(최대 4장)."
  },
  qwen_image_2_1: {
    id: "qwen_image_2_1",
    label: "Qwen-Image 2.1 (25스텝, 고품질)",
    description: "Turbo와 같은 모델 계열의 원본입니다. 25스텝이라 3배가량 느리고 디테일이 조금 더 좋습니다. 얼굴 참조를 지원합니다.",
    license: "Qwen Research (비상업)",
    commercialUse: false,
    supportsReferences: true,
    maxReferences: 4,
    referenceSyntax: "qwen21",
    defaults: {
      diffusionModel: "qwen_image_2.1_int8_convrot.safetensors",
      textEncoder: "qwen3vl_8b_int8_convrot.safetensors",
      vae: "qwen_image_2.1_vae_bf16.safetensors",
      steps: 25,
      cfg: 1,
      sampler: "euler",
      scheduler: "simple"
    },
    requiredNodes: [...QWEN21_NODES, "KSampler"],
    downloads: [
      {
        folder: "diffusion_models",
        file: "qwen_image_2.1_int8_convrot.safetensors",
        url: `${HF_QWEN_IMAGE_21}/diffusion_models/qwen_image_2.1_int8_convrot.safetensors`,
        sizeGb: 7.26
      },
      QWEN21_TEXT_ENCODER_DOWNLOAD,
      QWEN21_VAE_DOWNLOAD
    ],
    vramNote: "Turbo와 메모리 사용량은 같고(최대 약 15GB) 시간은 25스텝만큼 더 걸립니다. 기본 설치에는 모델 파일이 없어 따로 받아야 합니다."
  },
  z_image_turbo: {
    id: "z_image_turbo",
    label: "Z-Image Turbo (가장 빠름, 얼굴 고정 없음)",
    description: "8스텝 실사 생성. 가장 빠르고 동아시아 인물 표현이 좋습니다. 참조 이미지를 받지 않아 얼굴 고정 컷에는 쓰이지 않습니다.",
    license: "Apache-2.0",
    commercialUse: true,
    supportsReferences: false,
    maxReferences: 0,
    referenceSyntax: "none",
    defaults: {
      diffusionModel: "z_image_turbo_int8_convrot.safetensors",
      textEncoder: "qwen_3_4b_fp8_mixed.safetensors",
      vae: "ae.safetensors",
      steps: 8,
      cfg: 1,
      sampler: "res_multistep",
      scheduler: "simple"
    },
    fixedCfg: 1,
    requiredNodes: [
      "UNETLoader",
      "ModelSamplingAuraFlow",
      "CLIPLoader",
      "VAELoader",
      "CLIPTextEncode",
      "ConditioningZeroOut",
      "EmptySD3LatentImage",
      "KSampler",
      "VAEDecode",
      "PreviewImage"
    ],
    downloads: [
      {
        folder: "diffusion_models",
        file: "z_image_turbo_int8_convrot.safetensors",
        url: `${HF_Z_IMAGE_TURBO}/diffusion_models/z_image_turbo_int8_convrot.safetensors`,
        sizeGb: 6.2
      },
      {
        folder: "text_encoders",
        file: "qwen_3_4b_fp8_mixed.safetensors",
        url: `${HF_Z_IMAGE_TURBO}/text_encoders/qwen_3_4b_fp8_mixed.safetensors`,
        sizeGb: 5.63
      },
      { folder: "vae", file: "ae.safetensors", url: `${HF_Z_IMAGE_TURBO}/vae/ae.safetensors`, sizeGb: 0.34 }
    ],
    vramNote: "int8 모델 6.2GB + 텍스트 인코더 5.6GB로 16GB 카드에 함께 올라갑니다. 1024px 한 장에 수 초."
  },
  qwen_image_edit_2511: {
    id: "qwen_image_edit_2511",
    label: "Qwen-Image-Edit 2511 (상업 가능, 얼굴 고정)",
    description: "Apache-2.0 라이선스의 참조 편집 모델(20B). 참조 이미지 1~3장을 'image 1'처럼 불러 같은 인물을 유지합니다.",
    license: "Apache-2.0",
    commercialUse: true,
    supportsReferences: true,
    maxReferences: 3,
    referenceSyntax: "edit_plus",
    defaults: {
      diffusionModel: "qwen_image_edit_2511_fp8mixed.safetensors",
      textEncoder: "qwen_2.5_vl_7b_fp8_scaled.safetensors",
      vae: "qwen_image_vae.safetensors",
      steps: 40,
      cfg: 4,
      sampler: "euler",
      scheduler: "simple"
    },
    requiredNodes: [
      "UNETLoader",
      "ModelSamplingAuraFlow",
      "CFGNorm",
      "CLIPLoader",
      "VAELoader",
      "LoadImage",
      "ImageCrop",
      "TextEncodeQwenImageEditPlus",
      "FluxKontextMultiReferenceLatentMethod",
      "EmptySD3LatentImage",
      "KSampler",
      "VAEDecode",
      "PreviewImage"
    ],
    downloads: [
      {
        folder: "diffusion_models",
        file: "qwen_image_edit_2511_fp8mixed.safetensors",
        url: `${HF_QWEN_IMAGE_EDIT}/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors`,
        sizeGb: 20.5
      },
      {
        folder: "text_encoders",
        file: "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        url: `${HF_QWEN_IMAGE}/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`,
        sizeGb: 9.38
      },
      { folder: "vae", file: "qwen_image_vae.safetensors", url: `${HF_QWEN_IMAGE}/vae/qwen_image_vae.safetensors`, sizeGb: 0.25 }
    ],
    vramNote:
      "20B 모델(fp8 20.5GB)이라 16GB 카드에서는 일부를 RAM으로 내려 돌립니다. 40스텝 기준 한 장에 수 분이 걸릴 수 있습니다. 기본 설치에는 모델 파일이 없습니다."
  },
  custom: {
    id: "custom",
    label: "사용자 워크플로 (API 형식 JSON)",
    description:
      "ComfyUI에서 'Export (API)'로 내보낸 워크플로에 %prompt%, %seed%, %width% 같은 자리표시자를 넣어 씁니다(SillyTavern과 같은 방식). 참조 이미지는 LoadImage의 %image1%~%image3%로 들어갑니다.",
    license: "워크플로와 모델의 라이선스를 따릅니다",
    commercialUse: false,
    supportsReferences: true,
    maxReferences: 3,
    // The graph's model is unknown, so the prompt does not name references in any model's syntax; the images still
    // reach the workflow through %image1%..%image3%.
    referenceSyntax: "none",
    defaults: { diffusionModel: "", textEncoder: "", vae: "", steps: 20, cfg: 4, sampler: "euler", scheduler: "simple" },
    requiredNodes: [],
    downloads: [],
    vramNote: "워크플로와 모델에 따라 다릅니다."
  }
};

/**
 * API-format input key of the n-th (1-based) reference image on TextEncodeQwenImage21.
 *
 * The node declares its images as an Autogrow input named "images" with template names image_1..image_16
 * (comfy_extras/nodes_qwen.py). ComfyUI flattens dynamic inputs into "<input>.<name>" keys
 * (comfy_api/latest/_io.py: finalize_prefix joins the prefix path with "."), and the official template's own
 * input slots read "images.image_1". Kept as the single place the name lives so a live /object_info check can
 * correct it without touching any graph.
 */
export const QWEN21_REFERENCE_INPUT_KEY = (index: number): string => `images.image_${index}`;

/** Qwen-Image 2.1 Turbo's distilled schedule, baked into the official template (8 steps). */
export const QWEN21_TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0";

/** Reference names the relay accepts (and that are safe inside `%ref:<name>%`). */
export const COMFY_REFERENCE_NAME_PATTERN = /^[A-Za-z0-9_.-]{1,64}$/u;

/** The literal a LoadImage node carries until the relay substitutes the uploaded file name. */
export function comfyReferencePlaceholder(name: string): string {
  return `%ref:${name}%`;
}

export interface ComfyReferenceCrop {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * The head-and-shoulders square a reference is cropped to before it reaches the encoder.
 *
 * Measured on Qwen-Image 2.1 Turbo against the live ComfyUI: given the WHOLE profile image, some seeds treat the
 * reference as the edit target and re-render it — same bedroom, same top and skirt, the new cut's pose and setting
 * ignored — and that happened most when the output canvas matched the reference's size. Cropping to the top-centred
 * square (face, hair, shoulders) removes what could be copied: 6/6 seeds then moved the person into the described
 * scene and outfit with the face intact, including the seed and canvas that had failed. Profile portraits are
 * generated upper-body and centred, which is what makes a fixed crop safe.
 *
 * Undefined when the size is unknown or too small to crop meaningfully; the reference is then sent whole.
 */
export function computeReferenceHeadCrop(width: number | undefined, height: number | undefined): ComfyReferenceCrop | undefined {
  if (!Number.isFinite(width) || !Number.isFinite(height) || !width || !height || width < 256 || height < 256) {
    return undefined;
  }
  const side = Math.floor(Math.min(width, height * 0.45));
  const x = Math.max(0, Math.floor((width - side) / 2));
  const y = Math.min(Math.max(0, Math.floor(height * 0.02)), Math.max(0, height - side));
  return { x, y, width: side, height: side };
}

export interface ComfyWorkflowParams {
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  seed: number;
  steps: number;
  cfg: number;
  files: Required<PhotorealModelFiles>;
  /** Reference ids; the graph uses the literal string `%ref:<name>%` in LoadImage.inputs.image — the server relay uploads and substitutes. */
  referenceNames: string[];
  /** Optional crop per reference name, applied in-graph (ImageCrop) before the encoder. See computeReferenceHeadCrop. */
  referenceCrops?: Record<string, ComfyReferenceCrop>;
  /**
   * "DynamicChat/<jobId>", for a custom workflow's %filename_prefix%. The built-in graphs do not use it: they end in
   * PreviewImage, which names its own temp files (see the note at the top of this file).
   */
  filenamePrefix: string;
}

export type ComfyApiWorkflow = Record<string, { class_type: string; inputs: Record<string, unknown>; _meta?: { title?: string } }>;

export interface BuiltComfyWorkflow {
  workflow: ComfyApiWorkflow;
  outputNodeIds: string[];
  presetId: PhotorealWorkflowPresetId;
  /** The references the graph actually loads, in slot order (capped at the preset's maxReferences). */
  usedReferences: string[];
}

/** Placeholders a custom workflow may use. The SillyTavern names are kept so its workflows work unchanged. */
export const COMFY_CUSTOM_PLACEHOLDERS: ReadonlyArray<{ name: string; description: string }> = [
  { name: "prompt", description: "실사 프롬프트 (영어 문장, 앞뒤에 글자를 바로 붙이지 않고)" },
  { name: "negative_prompt", description: "네거티브 프롬프트 (샘플러의 negative 쪽 노드에만)" },
  { name: "seed", description: "시드 (숫자 입력의 값 전체로만)" },
  { name: "steps", description: "스텝 수 (숫자 입력의 값 전체로만)" },
  { name: "scale", description: "CFG (SillyTavern 이름, 숫자 입력의 값 전체로만, 1 이상)" },
  { name: "cfg", description: "CFG (숫자 입력의 값 전체로만, 1 이상)" },
  { name: "width", description: "가로 픽셀 (숫자 입력의 값 전체로만)" },
  { name: "height", description: "세로 픽셀 (숫자 입력의 값 전체로만)" },
  { name: "denoise", description: "디노이즈 강도 (항상 1, 숫자 입력의 값 전체로만)" },
  { name: "clip_skip", description: "CLIP skip (항상 -1, 숫자 입력의 값 전체로만)" },
  { name: "sampler", description: "샘플러 이름" },
  { name: "scheduler", description: "스케줄러 이름" },
  { name: "model", description: "확산 모델 파일 (설정에서 지정했을 때만, 로더의 …_name 입력 값 전체로)" },
  { name: "text_encoder", description: "텍스트 인코더 파일 (설정에서 지정했을 때만, 로더의 …_name 입력 값 전체로)" },
  { name: "vae", description: "VAE 파일 (설정에서 지정했을 때만, 로더의 …_name 입력 값 전체로)" },
  { name: "filename_prefix", description: "저장 파일 이름 접두사" },
  { name: "image1", description: "참조 이미지 1 (LoadImage의 image 값 전체로)" },
  { name: "image2", description: "참조 이미지 2 (LoadImage의 image 값 전체로)" },
  { name: "image3", description: "참조 이미지 3 (LoadImage의 image 값 전체로)" }
];

const MAX_WORKFLOW_NODES = 300;
const CUSTOM_MAX_REFERENCES = 3;
const OUTPUT_CLASS_TYPES = new Set(["SaveImage", "PreviewImage"]);
const PLACEHOLDER_PATTERN = /%([A-Za-z_][A-Za-z0-9_]*)%/gu;
/** Tokens ComfyUI itself expands inside SaveImage's filename_prefix; not ours to resolve. */
const COMFY_FILENAME_TOKENS = new Set(["width", "height", "year", "month", "day", "hour", "minute", "second"]);
const SILLYTAVERN_AVATAR_PLACEHOLDERS = new Set(["user_avatar", "char_avatar"]);

interface NormalizedParams {
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  seed: number;
  steps: number;
  cfg: number;
  files: Required<PhotorealModelFiles>;
  references: string[];
  crops: Map<string, ComfyReferenceCrop>;
  filenamePrefix: string;
  preset: PhotorealWorkflowPreset;
}

type GraphBuild = Omit<BuiltComfyWorkflow, "presetId">;

const GRAPH_BUILDERS: Record<PhotorealWorkflowPresetId, (params: NormalizedParams, customWorkflowJson?: string) => GraphBuild> = {
  z_image_turbo: buildZImageTurboGraph,
  qwen_image_2_1_turbo: (params) => buildQwenImage21Graph(params, true),
  qwen_image_2_1: (params) => buildQwenImage21Graph(params, false),
  qwen_image_edit_2511: buildQwenImageEdit2511Graph,
  custom: buildCustomGraph
};

/**
 * Preset lookup for ids that come from stored state. Own properties only: a corrupted id such as "constructor"
 * must not resolve to something on Object.prototype.
 */
function findPreset(presetId: unknown): PhotorealWorkflowPreset | undefined {
  return typeof presetId === "string" && Object.prototype.hasOwnProperty.call(PHOTOREAL_WORKFLOW_PRESETS, presetId)
    ? PHOTOREAL_WORKFLOW_PRESETS[presetId as PhotorealWorkflowPresetId]
    : undefined;
}

export function buildComfyWorkflow(
  presetId: PhotorealWorkflowPresetId,
  params: ComfyWorkflowParams,
  customWorkflowJson?: string
): BuiltComfyWorkflow {
  const preset = findPreset(presetId);
  if (!preset) {
    throw new Error(`알 수 없는 실사 워크플로 프리셋입니다: ${String(presetId)}`);
  }
  const built = GRAPH_BUILDERS[preset.id](normalizeWorkflowParams(preset, params), customWorkflowJson);
  // Every graph is checked before it leaves, built-in ones included: a broken link or a stray placeholder is far
  // easier to read here than as a ComfyUI 400 relayed back through the server.
  const errors = validateComfyApiWorkflow(built.workflow);
  if (errors.length > 0) {
    throw new Error(`실사 워크플로가 올바르지 않습니다: ${errors.join(" / ")}`);
  }
  return { ...built, presetId };
}

function normalizeWorkflowParams(preset: PhotorealWorkflowPreset, params: ComfyWorkflowParams): NormalizedParams {
  const limit = preset.id === "custom" ? CUSTOM_MAX_REFERENCES : preset.supportsReferences ? preset.maxReferences : 0;
  const references: string[] = [];
  for (const name of Array.isArray(params.referenceNames) ? params.referenceNames : []) {
    if (typeof name !== "string" || !COMFY_REFERENCE_NAME_PATTERN.test(name)) {
      throw new Error(`참조 이미지 이름이 올바르지 않습니다: ${String(name)} (영문·숫자·._- 64자 이내)`);
    }
    if (!references.includes(name)) {
      references.push(name);
    }
  }
  const files = params.files ?? { diffusionModel: "", textEncoder: "", vae: "" };
  return {
    prompt: neutralizePlaceholders(typeof params.prompt === "string" ? params.prompt : ""),
    // Cleaned here rather than trusted from the planner: a job queued under an older rule set, or a caller that never
    // went through the planner, reaches the sampler through this line all the same.
    negativePrompt: neutralizePlaceholders(sanitizePhotorealNegativePrompt(typeof params.negativePrompt === "string" ? params.negativePrompt : "")),
    width: sanitizeDimension(params.width),
    height: sanitizeDimension(params.height),
    seed: sanitizeSeed(params.seed),
    steps: Number.isFinite(params.steps) && params.steps >= 1 ? Math.min(200, Math.round(params.steps)) : preset.defaults.steps,
    cfg: preset.fixedCfg ?? clampPhotorealCfg(params.cfg, preset.defaults.cfg),
    files: {
      diffusionModel: cleanFileName(files.diffusionModel) || preset.defaults.diffusionModel,
      textEncoder: cleanFileName(files.textEncoder) || preset.defaults.textEncoder,
      vae: cleanFileName(files.vae) || preset.defaults.vae
    },
    // The extra references are not an error: the planner caps by maxReferences too, and usedReferences tells the
    // caller exactly which images the graph loads.
    references: references.slice(0, limit),
    crops: sanitizeReferenceCrops(params.referenceCrops),
    filenamePrefix: sanitizeFilenamePrefix(params.filenamePrefix),
    preset
  };
}

function sanitizeReferenceCrops(value: ComfyWorkflowParams["referenceCrops"]): Map<string, ComfyReferenceCrop> {
  const crops = new Map<string, ComfyReferenceCrop>();
  if (!value || typeof value !== "object") {
    return crops;
  }
  for (const [name, crop] of Object.entries(value)) {
    if (!COMFY_REFERENCE_NAME_PATTERN.test(name) || !crop || typeof crop !== "object") {
      continue;
    }
    const x = Math.trunc(Number(crop.x));
    const y = Math.trunc(Number(crop.y));
    const width = Math.trunc(Number(crop.width));
    const height = Math.trunc(Number(crop.height));
    // ImageCrop clamps to the image itself, so only shape is checked here: non-negative origin, a usable size.
    if ([x, y, width, height].every(Number.isFinite) && x >= 0 && y >= 0 && width >= 64 && height >= 64 && width <= 16384 && height <= 16384) {
      crops.set(name, { x, y, width, height });
    }
  }
  return crops;
}

/**
 * The prompt is free text that ends up inside the graph. A "%word%" in it would read as an unresolved placeholder
 * (and get interpolated by a custom workflow), and "%ref:" would collide with the relay's reference substitution,
 * so the percent sign is split off from what follows it. The encoders read "% word%" the same way.
 */
function neutralizePlaceholders(text: string): string {
  return text.replace(/%(?=[A-Za-z_][A-Za-z0-9_]*%|ref:)/giu, "% ");
}

function sanitizeDimension(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 1024;
  }
  return snapPhotorealSide(value);
}

function sanitizeSeed(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(Math.abs(value)));
}

function cleanFileName(value: string | undefined): string {
  return typeof value === "string" ? value.trim() : "";
}

function sanitizeFilenamePrefix(value: string): string {
  // ComfyUI refuses a prefix that escapes its output folder; keep it to a plain relative path.
  const cleaned = (typeof value === "string" ? value : "")
    .replace(/\\/gu, "/")
    .split("/")
    .map((segment) => segment.replace(/[^A-Za-z0-9_.-]/gu, "_").replace(/^\.+$/u, ""))
    .filter(Boolean)
    .join("/");
  return cleaned || "DynamicChat/photoreal";
}

function link(nodeId: string, outputIndex = 0): [string, number] {
  return [nodeId, outputIndex];
}

function node(classType: string, inputs: Record<string, unknown>, title?: string): ComfyApiWorkflow[string] {
  return title ? { class_type: classType, inputs, _meta: { title } } : { class_type: classType, inputs };
}

/** Reference LoadImage nodes live at ids 20+ so the core graph keeps the same ids whatever the reference count. */
function referenceNodeId(index: number): string {
  return String(20 + index);
}

/** ImageCrop nodes for the references live at ids 40+, next to their loaders' 20+. */
function referenceCropNodeId(index: number): string {
  return String(40 + index);
}

function addReferenceLoaders(workflow: ComfyApiWorkflow, references: string[], crops?: Map<string, ComfyReferenceCrop>): void {
  references.forEach((name, index) => {
    workflow[referenceNodeId(index)] = node("LoadImage", { image: comfyReferencePlaceholder(name) }, `Reference ${index + 1}: ${name}`);
    const crop = crops?.get(name);
    if (crop) {
      workflow[referenceCropNodeId(index)] = node(
        "ImageCrop",
        { image: link(referenceNodeId(index)), width: crop.width, height: crop.height, x: crop.x, y: crop.y },
        `Reference ${index + 1}: head crop`
      );
    }
  });
}

/** What the encoder reads for reference `index`: its head crop when one was planned, else the loaded image. */
function referenceImageLink(params: NormalizedParams, index: number): [string, number] {
  return params.crops.has(params.references[index]) ? link(referenceCropNodeId(index)) : link(referenceNodeId(index));
}

function buildZImageTurboGraph(params: NormalizedParams): GraphBuild {
  const workflow: ComfyApiWorkflow = {
    "1": node("UNETLoader", { unet_name: params.files.diffusionModel, weight_dtype: "default" }, "Diffusion model"),
    "2": node("ModelSamplingAuraFlow", { model: link("1"), shift: 3 }),
    "3": node("CLIPLoader", { clip_name: params.files.textEncoder, type: "lumina2", device: "default" }, "Text encoder"),
    "4": node("VAELoader", { vae_name: params.files.vae }, "VAE"),
    "5": node("CLIPTextEncode", { clip: link("3"), text: params.prompt }, "Positive prompt"),
    // Z-Image Turbo is distilled for CFG 1, where the negative branch is never evaluated; the official template
    // zeroes the positive conditioning instead of encoding a negative prompt.
    "6": node("ConditioningZeroOut", { conditioning: link("5") }, "Negative (zeroed)"),
    "7": node("EmptySD3LatentImage", { width: params.width, height: params.height, batch_size: 1 }),
    "8": node(
      "KSampler",
      {
        model: link("2"),
        seed: params.seed,
        steps: params.steps,
        cfg: params.cfg,
        sampler_name: params.preset.defaults.sampler,
        scheduler: params.preset.defaults.scheduler,
        positive: link("5"),
        negative: link("6"),
        latent_image: link("7"),
        denoise: 1
      },
      "Sampler"
    ),
    "9": node("VAEDecode", { samples: link("8"), vae: link("4") }),
    "10": previewOutputNode(link("9"))
  };
  return { workflow, outputNodeIds: ["10"], usedReferences: [] };
}

/**
 * The output node of every built-in graph. PreviewImage takes only `images`: it writes to ComfyUI's temp folder under
 * a name of its own, so there is no filename_prefix to pass (see the note at the top of this file for why not
 * SaveImage).
 */
function previewOutputNode(images: [string, number]): ComfyApiWorkflow[string] {
  return node("PreviewImage", { images }, "Result (temp, not saved)");
}

function buildQwenImage21Graph(params: NormalizedParams, turbo: boolean): GraphBuild {
  const hasReferences = params.references.length > 0;
  const encoderInputs: Record<string, unknown> = {
    clip: link("3"),
    prompt: params.prompt,
    negative_prompt: params.negativePrompt,
    // Without references, vae stays connected and resolution stays 1024.
    // With references, ComfyUI's TextEncodeQwenImage21 (comfy_extras/nodes_qwen.py) sets
    // `keep_vision = len(ref_latents) == 0`: connecting `vae` drops Qwen3-VL's semantic vision tokens
    // and splices pixel-level 2D-RoPE VAE latents into the DiT, which forces the model to copy-paste the
    // exact head angle, expression, and pose of <image1> while spilling KV cache on 16 GB GPUs.
    // Omitting `vae` when references are present keeps `keep_vision = True` through Qwen3-VL's vision tower
    // at 512px, preserving facial identity and hairstyle while letting pose, expression, outfit, and scene
    // follow the prompt naturally.
    ...(hasReferences ? {} : { vae: link("4") }),
    resolution: hasReferences ? 512 : 1024
  };
  params.references.forEach((_name, index) => {
    encoderInputs[QWEN21_REFERENCE_INPUT_KEY(index + 1)] = referenceImageLink(params, index);
  });
  const workflow: ComfyApiWorkflow = {
    "1": node("UNETLoader", { unet_name: params.files.diffusionModel, weight_dtype: "default" }, "Diffusion model"),
    "2": node("QwenImage21Cache", { model: link("1"), device: "auto", dtype: "default" }),
    "3": node("CLIPLoader", { clip_name: params.files.textEncoder, type: "qwen_image", device: "default" }, "Text encoder"),
    "4": node("VAELoader", { vae_name: params.files.vae }, "VAE"),
    "5": node("TextEncodeQwenImage21", encoderInputs, "Prompt + references"),
    // The cut's own size, not the encoder's latent output (sized to reference 1): this is a new composition of the
    // referenced person, not a pixel-aligned edit of the reference.
    "6": node("EmptyLatentImage", { width: params.width, height: params.height, batch_size: 1 })
  };
  addReferenceLoaders(workflow, params.references, params.crops);
  if (turbo) {
    // Turbo's 8-step schedule is baked into the checkpoint's distillation, so steps are fixed by the sigmas.
    workflow["7"] = node("CFGGuider", { model: link("2"), positive: link("5", 0), negative: link("5", 1), cfg: params.cfg });
    workflow["8"] = node("KSamplerSelect", { sampler_name: params.preset.defaults.sampler });
    workflow["9"] = node("RandomNoise", { noise_seed: params.seed });
    workflow["10"] = node("ManualSigmas", { sigmas: QWEN21_TURBO_SIGMAS }, "Turbo schedule");
    workflow["11"] = node(
      "SamplerCustomAdvanced",
      { noise: link("9"), guider: link("7"), sampler: link("8"), sigmas: link("10"), latent_image: link("6") },
      "Sampler"
    );
    workflow["12"] = node("VAEDecode", { samples: link("11", 0), vae: link("4") });
    workflow["13"] = previewOutputNode(link("12"));
    return { workflow, outputNodeIds: ["13"], usedReferences: [...params.references] };
  }
  workflow["7"] = node(
    "KSampler",
    {
      model: link("2"),
      seed: params.seed,
      steps: params.steps,
      cfg: params.cfg,
      sampler_name: params.preset.defaults.sampler,
      scheduler: params.preset.defaults.scheduler,
      positive: link("5", 0),
      negative: link("5", 1),
      latent_image: link("6"),
      denoise: 1
    },
    "Sampler"
  );
  workflow["8"] = node("VAEDecode", { samples: link("7"), vae: link("4") });
  workflow["9"] = previewOutputNode(link("8"));
  return { workflow, outputNodeIds: ["9"], usedReferences: [...params.references] };
}

function buildQwenImageEdit2511Graph(params: NormalizedParams): GraphBuild {
  const references = params.references;
  const imageInputs: Record<string, unknown> = {};
  references.forEach((_name, index) => {
    imageInputs[`image${index + 1}`] = referenceImageLink(params, index);
  });
  const workflow: ComfyApiWorkflow = {
    "1": node("UNETLoader", { unet_name: params.files.diffusionModel, weight_dtype: "default" }, "Diffusion model"),
    "2": node("ModelSamplingAuraFlow", { model: link("1"), shift: 3.1 }),
    "3": node("CFGNorm", { model: link("2"), strength: 1 }),
    "4": node("CLIPLoader", { clip_name: params.files.textEncoder, type: "qwen_image", device: "default" }, "Text encoder"),
    "5": node("VAELoader", { vae_name: params.files.vae }, "VAE"),
    "6": node("TextEncodeQwenImageEditPlus", { clip: link("4"), prompt: params.prompt, vae: link("5"), ...imageInputs }, "Positive prompt"),
    // The template's negative is the same encoder with an empty prompt: the reference images sit in both branches,
    // so CFG amplifies what the instruction changes rather than pushing away from the referenced people.
    "7": node("TextEncodeQwenImageEditPlus", { clip: link("4"), prompt: "", vae: link("5"), ...imageInputs }, "Negative prompt"),
    "8": node("FluxKontextMultiReferenceLatentMethod", { conditioning: link("6"), reference_latents_method: "index_timestep_zero" }),
    "9": node("FluxKontextMultiReferenceLatentMethod", { conditioning: link("7"), reference_latents_method: "index_timestep_zero" }),
    // The official edit template samples on the VAE-encoded reference 1, i.e. on the reference's own canvas. That is
    // pixel-aligned EDITING, and it is exactly what made Qwen re-render the reference instead of the described cut
    // (measured on Qwen-Image 2.1, the same family). A story cut is a new composition of the person, so it samples on
    // its own size like every other preset; the references only carry identity.
    "10": node("EmptySD3LatentImage", { width: params.width, height: params.height, batch_size: 1 }),
    "11": node(
      "KSampler",
      {
        model: link("3"),
        seed: params.seed,
        steps: params.steps,
        cfg: params.cfg,
        sampler_name: params.preset.defaults.sampler,
        scheduler: params.preset.defaults.scheduler,
        positive: link("8"),
        negative: link("9"),
        latent_image: link("10"),
        denoise: 1
      },
      "Sampler"
    ),
    "12": node("VAEDecode", { samples: link("11"), vae: link("5") }),
    "13": previewOutputNode(link("12"))
  };
  addReferenceLoaders(workflow, references, params.crops);
  return { workflow, outputNodeIds: ["13"], usedReferences: [...references] };
}

function buildCustomGraph(params: NormalizedParams, customWorkflowJson?: string): GraphBuild {
  if (typeof customWorkflowJson !== "string" || !customWorkflowJson.trim()) {
    throw new Error("사용자 워크플로(JSON)가 비어 있습니다. 설정의 실사 이미지 항목에 ComfyUI API 형식 워크플로를 붙여 넣으세요.");
  }
  const references = params.references.slice(0, CUSTOM_MAX_REFERENCES);
  const slots = [1, 2, 3].filter((slot) => customWorkflowJson.includes(`%image${slot}%`));
  const missing = slots.filter((slot) => slot > references.length);
  if (missing.length > 0) {
    // Filling an empty slot with another person's face, or with nothing, would both be silently wrong.
    throw new Error(
      `사용자 워크플로가 ${missing.map((slot) => `%image${slot}%`).join(", ")} 자리표시자를 쓰지만 이 컷의 참조 이미지는 ${references.length}장입니다.`
    );
  }
  const values: Record<string, string | number> = {
    prompt: params.prompt,
    negative_prompt: params.negativePrompt,
    seed: params.seed,
    steps: params.steps,
    scale: params.cfg,
    cfg: params.cfg,
    width: params.width,
    height: params.height,
    denoise: 1,
    clip_skip: -1,
    sampler: params.preset.defaults.sampler,
    scheduler: params.preset.defaults.scheduler,
    filename_prefix: params.filenamePrefix
  };
  // Model files only when the creator set them: an empty string would pass the placeholder check and fail later
  // as a ComfyUI "value not in list" that does not say which field was missing.
  if (params.files.diffusionModel) {
    values.model = params.files.diffusionModel;
  }
  if (params.files.textEncoder) {
    values.text_encoder = params.files.textEncoder;
  }
  if (params.files.vae) {
    values.vae = params.files.vae;
  }
  references.forEach((name, index) => {
    values[`image${index + 1}`] = name;
  });
  // The settings screen runs the same check, but the template is read when the cut runs, so a template saved before
  // the rule existed (or edited outside the screen) is held to it here.
  const placementErrors = checkTemplatePlaceholderPlacement(parseTemplate(customWorkflowJson));
  if (placementErrors.length > 0) {
    throw new Error(`사용자 워크플로를 쓸 수 없습니다: ${placementErrors.join(" / ")}`);
  }
  const workflow = substituteComfyPlaceholders(customWorkflowJson, values);
  sanitizeNegativeSideText(workflow);
  return {
    workflow,
    outputNodeIds: findOutputNodeIds(workflow),
    usedReferences: slots.map((slot) => references[slot - 1])
  };
}

/**
 * Inputs that name a file or pick from a fixed list (model files, sampler, scheduler, dtype, device, a loader's
 * `type`, the output prefix, a LoadImage's file). ComfyUI checks such a value against what is installed or allowed
 * (value_not_in_list), so no free text reaches a model through one, while running a text rule over them would break
 * them ("index_timestep_zero" is not a negative prompt). `image` is held to its own rule (%ref: only).
 */
const NON_TEXT_INPUT_PATTERN =
  /(?:^|_)name\d*$|^(?:type|device|dtype|weight_dtype|sampler|scheduler|method|mode|crop|upscale_method|reference_latents_method|channel|filename_prefix|image|sigmas)$/u;
/** Model-file placeholders: creator-typed text that the safety floor never reads, so only a loader may receive it. */
const FILE_NAME_PLACEHOLDERS = new Set(["model", "text_encoder", "vae"]);
const FILE_NAME_INPUT_PATTERN = /(?:^|_)name\d*$/u;
/**
 * Placeholders that stand for a number. Their values are settings (steps, CFG, seed, size), which the floor never
 * reads as words, so inside a sentence they compose what screening cannot see: "She is a %steps% year old girl"
 * reads as "She is a year old girl" and is sent as "She is a 12 year old girl". They may only BE a number input's
 * value — the whole of it — where substitution leaves a JSON number and no text at all.
 */
const NUMERIC_PLACEHOLDERS = new Set(["seed", "steps", "scale", "cfg", "width", "height", "denoise", "clip_skip"]);
/** Inputs a numeric placeholder may fill: the sampler, size and CLIP-skip numbers, by their ComfyUI names. */
const NUMERIC_INPUT_PATTERN = /(?:^|_)(?:seed|steps|cfg|scale|guidance|width|height|denoise|clip_skip|stop_at_clip_layer)$/u;
/** Core primitive nodes whose `value` is a number (PrimitiveString's `value` is text, so it is not one of them). */
const NUMERIC_PRIMITIVE_CLASS_PATTERN = /^Primitive(?:Int|Float)$/u;
/**
 * Inputs that take a guidance scale, where a value below PHOTOREAL_MIN_CFG turns the negative into the picture:
 * `cfg` itself and the names ComfyUI gives its variants (cfg_scale, cfg_conds, cfg_cond2_negative, min_cfg,
 * video_cfg, ...), plus the two negative-prompt guidance scales (PerpNegGuider's neg_scale, NAG's nag_scale) that
 * invert the same way. Not cfg_trunc or restore_cfg_s_tmin, which are step and sigma bounds, not scales.
 */
const GUIDANCE_SCALE_INPUT_PATTERN = /^(?:[a-z0-9]+_)*cfg(?:_scale|_conds|_cond\d*_negative)?$|^(?:neg|nag)_scale$/iu;
/** The sampler/guider input that takes the conditioning a picture is pushed AWAY from (KSampler, CFGGuider, ...). */
const NEGATIVE_CONDITIONING_INPUT = "negative";
/**
 * The nodes whose `negative` input is only ever guidance — used up by the sampling, never handed on: ComfyUI's samplers
 * and guiders. A node that takes a `negative` and puts conditioning out again (ControlNetApplyAdvanced,
 * InpaintModelConditioning, InstructPixToPixConditioning, a custom node) can route it into a sampler's `positive`, so
 * on any other node an input named `negative` is read as the picture's own: the name alone proves nothing.
 */
const NEGATIVE_SINK_CLASS_PATTERN = /^(?:KSampler|KSamplerAdvanced|SamplerCustom|CFGGuider|DualCFGGuider|PerpNegGuider)$/u;
/**
 * Encoders that write the picture's conditioning and the negative one from separate inputs, with the output that
 * carries the negative. Their negative input is negative-side only while that output never feeds the picture side —
 * wired into `positive`, it is the picture's prompt like any other.
 */
const SPLIT_NEGATIVE_ENCODERS: Readonly<Record<string, { input: string; output: number }>> = {
  TextEncodeQwenImage21: { input: "negative_prompt", output: 1 }
};
// Not `%ref:` values: a template writes references as %image1%, so a %ref: in its text is read like any other words.
const ANY_PLACEHOLDER_PATTERN = /%[A-Za-z_][A-Za-z0-9_]*%/gu;

/** One literal string of a custom template, with every placeholder taken out. */
export interface ComfyWorkflowLiteralText {
  nodeId: string;
  classType: string;
  input: string;
  /** The literal with each placeholder replaced by a space: phrases split around a placeholder read as one. */
  text: string;
  /** The literal pieces between placeholders, each on its own. */
  fragments: string[];
}

/**
 * The text a custom template carries on its own — what it sends with every cut whatever the cut's prompt says, and
 * what the safety floor must therefore read. Every string input on the picture side except the file and list inputs
 * above, nested values included, links left out. Negative-side text is left to the negative clean-up instead (a
 * negative legitimately says "nude" or "child" to keep them out), which is safe only because the picture side is
 * worked out from the graph's ends rather than from input names (readPictureSide). Unparseable JSON yields nothing:
 * the template validator reports it, and such a template never builds.
 */
export function listCustomWorkflowLiteralTexts(workflowJson: string): ComfyWorkflowLiteralText[] {
  const workflow = parseTemplate(workflowJson);
  if (!workflow) {
    return [];
  }
  const pictureSide = readPictureSide(workflow);
  const literals: ComfyWorkflowLiteralText[] = [];
  for (const [nodeId, node] of Object.entries(workflow)) {
    if (!isPlainObject(node) || !isPlainObject(node.inputs)) {
      continue;
    }
    const classType = typeof node.class_type === "string" ? node.class_type : "";
    for (const [input, value] of Object.entries(node.inputs)) {
      if (NON_TEXT_INPUT_PATTERN.test(input) || isNegativeOnlyInput(pictureSide, classType, nodeId, input)) {
        continue;
      }
      visitTemplateStrings(value, (raw) => {
        const fragments = raw.split(ANY_PLACEHOLDER_PATTERN).map(collapseSpaces).filter(hasWordCharacter);
        const text = collapseSpaces(raw.replace(ANY_PLACEHOLDER_PATTERN, " "));
        if (fragments.length > 0 && hasWordCharacter(text)) {
          literals.push({ nodeId, classType, input, text, fragments });
        }
      });
    }
  }
  return literals;
}

/** One picture-side string of a built graph, where it sits. */
export interface ComfyWorkflowPictureText {
  nodeId: string;
  classType: string;
  input: string;
  text: string;
}

/**
 * Every string a BUILT graph can show the model — what is actually sent, after substitution — for the floor's last
 * reading (the orchestrator's screenBuiltPhotorealWorkflow). The same selection as listCustomWorkflowLiteralTexts:
 * picture side only (worked out from the links), file and list inputs left out. Screening the pieces that went in is
 * not enough, because the pieces meet here: a style ending in "na" and a template literal "%prompt%ked" each pass,
 * and only this string says "na.ked".
 */
export function listPhotorealGraphPictureTexts(workflow: ComfyApiWorkflow): ComfyWorkflowPictureText[] {
  if (!isPlainObject(workflow)) {
    return [];
  }
  const pictureSide = readPictureSide(workflow);
  const texts: ComfyWorkflowPictureText[] = [];
  for (const [nodeId, node] of Object.entries(workflow)) {
    if (!isPlainObject(node) || !isPlainObject(node.inputs)) {
      continue;
    }
    const classType = typeof node.class_type === "string" ? node.class_type : "";
    for (const [input, value] of Object.entries(node.inputs)) {
      if (NON_TEXT_INPUT_PATTERN.test(input) || isNegativeOnlyInput(pictureSide, classType, nodeId, input)) {
        continue;
      }
      visitTemplateStrings(value, (raw) => {
        if (raw.trim()) {
          texts.push({ nodeId, classType, input, text: raw });
        }
      });
    }
  }
  return texts;
}

function parseTemplate(workflowJson: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(workflowJson);
    return isPlainObject(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

/** Every string inside an input value, links (["id", n]) excepted. */
function visitTemplateStrings(value: unknown, visit: (text: string) => void): void {
  if (typeof value === "string") {
    visit(value);
  } else if (Array.isArray(value)) {
    if (!isLinkValue(value)) {
      value.forEach((item) => visitTemplateStrings(item, visit));
    }
  } else if (isPlainObject(value)) {
    Object.values(value).forEach((item) => visitTemplateStrings(item, visit));
  }
}

function isLinkValue(value: unknown[]): boolean {
  return value.length === 2 && (typeof value[0] === "string" || typeof value[0] === "number") && typeof value[1] === "number";
}

function collapseSpaces(text: string): string {
  return text.replace(/\s+/gu, " ").trim();
}

function hasWordCharacter(text: string): boolean {
  return /[\p{L}\p{N}]/u.test(text);
}

interface PictureSide {
  /** Nodes whose outputs reach the result other than through a `negative` input. */
  nodes: Set<string>;
  /** The same reach per output, as "nodeId:outputIndex". */
  outputs: Set<string>;
}

/**
 * Everything that can shape the picture: the graph walked back from its ends — every node no link consumes (the
 * output nodes, and anything left dangling, which is read as if it mattered) — through every input except a
 * sampler's or guider's `negative` (NEGATIVE_SINK_CLASS_PATTERN). Worked out from the ends rather than from inputs
 * named "positive" because a graph does not have to say "positive": BasicGuider takes `conditioning`, and a node can
 * feed a sampler's negative and the picture at once. A node off this side reaches the result only as what it is
 * steered away from.
 */
function readPictureSide(workflow: Record<string, unknown>): PictureSide {
  const consumed = new Set<string>();
  for (const node of Object.values(workflow)) {
    forEachLink(node, (_input, target) => consumed.add(target));
  }
  const nodes = new Set<string>();
  const outputs = new Set<string>();
  const pending = Object.keys(workflow).filter((nodeId) => !consumed.has(nodeId));
  while (pending.length > 0) {
    const nodeId = pending.pop() as string;
    if (nodes.has(nodeId) || !Object.prototype.hasOwnProperty.call(workflow, nodeId)) {
      continue;
    }
    nodes.add(nodeId);
    const node = workflow[nodeId];
    const negativeIsGuidance = isPlainObject(node) && typeof node.class_type === "string" && NEGATIVE_SINK_CLASS_PATTERN.test(node.class_type);
    forEachLink(node, (input, target, outputIndex) => {
      if (input !== NEGATIVE_CONDITIONING_INPUT || !negativeIsGuidance) {
        outputs.add(`${target}:${outputIndex}`);
        pending.push(target);
      }
    });
  }
  return { nodes, outputs };
}

function forEachLink(node: unknown, visit: (input: string, target: string, outputIndex: number) => void): void {
  if (!isPlainObject(node) || !isPlainObject(node.inputs)) {
    return;
  }
  for (const [input, value] of Object.entries(node.inputs)) {
    if (Array.isArray(value) && isLinkValue(value)) {
      visit(input, String(value[0]), Number(value[1]));
    }
  }
}

/** Whether an input of `nodeId` can only reach the result as negative conditioning (see readPictureSide). */
function isNegativeOnlyInput(pictureSide: PictureSide, classType: string, nodeId: string, input: string): boolean {
  const split = Object.prototype.hasOwnProperty.call(SPLIT_NEGATIVE_ENCODERS, classType) ? SPLIT_NEGATIVE_ENCODERS[classType] : undefined;
  if (split && split.input === input) {
    return !pictureSide.outputs.has(`${nodeId}:${split.output}`);
  }
  return !pictureSide.nodes.has(nodeId);
}

/** `value` with every string inside it (links excepted) passed through `map`. */
function mapTemplateStrings(value: unknown, map: (text: string) => string): unknown {
  if (typeof value === "string") {
    return map(value);
  }
  if (Array.isArray(value)) {
    return isLinkValue(value) ? value : value.map((item) => mapTemplateStrings(item, map));
  }
  if (isPlainObject(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, mapTemplateStrings(item, map)]));
  }
  return value;
}

/**
 * Where a custom template may put the placeholders whose text the safety floor never reads. %negative_prompt% (the
 * creator's negatives, which legitimately say "nude" or "child" to keep them OUT) only on the negative side — wired
 * into the picture's own encoder it would be an unread prompt. %model%, %text_encoder% and %vae% (file names typed in
 * settings) only as a whole loader file input. %seed%, %steps%, %cfg% and the other numbers only as the whole value of
 * a number input (NUMERIC_PLACEHOLDERS says why). And %prompt% never glued to a letter or digit: "%prompt%ked" after
 * a style ending in "na" sends a word that neither piece contains. Everything else a template sends is either the
 * screened %prompt% or its own literal text, which the orchestrator screens.
 */
function checkTemplatePlaceholderPlacement(workflow: Record<string, unknown> | undefined): string[] {
  if (!workflow) {
    return [];
  }
  const pictureSide = readPictureSide(workflow);
  const errors: string[] = [];
  for (const [nodeId, node] of Object.entries(workflow)) {
    if (!isPlainObject(node) || !isPlainObject(node.inputs)) {
      continue;
    }
    const classType = typeof node.class_type === "string" ? node.class_type : "?";
    for (const [input, value] of Object.entries(node.inputs)) {
      const where = `노드 "${nodeId}" (${classType})의 입력 "${input}"`;
      visitTemplateStrings(value, (text) => {
        for (const match of text.matchAll(PLACEHOLDER_PATTERN)) {
          const name = match[1];
          const whole = text === match[0];
          if (name === "negative_prompt" && !isNegativeOnlyInput(pictureSide, classType, nodeId, input)) {
            errors.push(`${where}: %negative_prompt%는 네거티브 쪽(샘플러의 negative로만 이어지는 노드)에만 쓸 수 있습니다.`);
          } else if (FILE_NAME_PLACEHOLDERS.has(name) && !(whole && FILE_NAME_INPUT_PATTERN.test(input))) {
            errors.push(`${where}: %${name}%는 모델 파일 입력(ckpt_name, unet_name 같은 …_name)의 값 전체로만 쓸 수 있습니다.`);
          } else if (NUMERIC_PLACEHOLDERS.has(name) && !NON_TEXT_INPUT_PATTERN.test(input) && !(whole && isNumericInput(classType, input))) {
            errors.push(
              `${where}: %${name}%는 숫자 입력(seed, steps, cfg, width 같은)의 값 전체로만 쓸 수 있습니다. 글 안에 넣으면 그 숫자가 프롬프트의 일부가 되어(예: "%steps%살") 안전 검사가 읽지 못한 문장이 됩니다.`
            );
          } else if (name === "prompt" && touchesWordCharacter(text, match.index ?? 0, match[0].length)) {
            errors.push(
              `${where}: %prompt% 바로 앞뒤에 글자나 숫자를 붙일 수 없습니다(띄어쓰기나 쉼표로 떼어 쓰세요). 붙여 쓰면 프롬프트 끝과 이어져 어느 쪽에도 없던 단어가 만들어집니다.`
            );
          }
        }
      });
    }
  }
  return errors;
}

function isNumericInput(classType: string, input: string): boolean {
  return NUMERIC_INPUT_PATTERN.test(input) || (input === "value" && NUMERIC_PRIMITIVE_CLASS_PATTERN.test(classType));
}

/**
 * Whether the placeholder at text[index, index + length) has a letter or digit right before or right after it.
 * Invisible format characters and combining marks in between do not separate anything (a zero-width space between
 * "na" and "ked" still reads as one word to the encoder), so they are looked through.
 */
function touchesWordCharacter(text: string, index: number, length: number): boolean {
  return /[\p{L}\p{N}][\p{Cf}\p{M}]*$/u.test(text.slice(0, index)) || /^[\p{Cf}\p{M}]*[\p{L}\p{N}]/u.test(text.slice(index + length));
}

/**
 * Cleans the text on a built custom graph's negative side the way the job's own negative was cleaned: the literal
 * words a template wraps around %negative_prompt% ("adult, %negative_prompt%") steer the sampler just the same.
 */
function sanitizeNegativeSideText(workflow: ComfyApiWorkflow): void {
  const pictureSide = readPictureSide(workflow);
  for (const [nodeId, node] of Object.entries(workflow)) {
    if (!isPlainObject(node) || !isPlainObject(node.inputs)) {
      continue;
    }
    for (const [input, value] of Object.entries(node.inputs)) {
      if (!NON_TEXT_INPUT_PATTERN.test(input) && isNegativeOnlyInput(pictureSide, node.class_type, nodeId, input)) {
        node.inputs[input] = mapTemplateStrings(value, (text) => neutralizePlaceholders(sanitizePhotorealNegativePrompt(text)));
      }
    }
  }
}

/**
 * SillyTavern-compatible placeholder substitution. A string value that is exactly "%name%" becomes the typed
 * value (so "%seed%" becomes a JSON number, which ComfyUI's INT inputs need); inside a longer string the value is
 * interpolated as text. `image1`..`image3` carry reference NAMES and become `%ref:<name>%`, which the server relay
 * resolves after uploading. Unknown placeholders are left in place for validateComfyApiWorkflow to report.
 */
export function substituteComfyPlaceholders(workflowJson: string, values: Record<string, string | number>): ComfyApiWorkflow {
  let parsed: unknown;
  try {
    parsed = JSON.parse(workflowJson);
  } catch (error) {
    throw new Error(`워크플로 JSON을 읽을 수 없습니다: ${error instanceof Error ? error.message : String(error)}`);
  }
  const resolved = new Map<string, string | number>();
  for (const [name, value] of Object.entries(values ?? {})) {
    if (typeof value !== "string" && !(typeof value === "number" && Number.isFinite(value))) {
      continue;
    }
    resolved.set(name, /^image\d+$/u.test(name) && typeof value === "string" && !/^%ref:.*%$/u.test(value) ? comfyReferencePlaceholder(value) : value);
  }
  return substituteValue(parsed, resolved) as ComfyApiWorkflow;
}

function substituteValue(value: unknown, values: Map<string, string | number>): unknown {
  if (typeof value === "string") {
    const whole = value.match(/^%([A-Za-z_][A-Za-z0-9_]*)%$/u);
    if (whole && values.has(whole[1])) {
      return values.get(whole[1]);
    }
    // Single pass: a substituted value is never re-scanned, so a prompt containing "%seed%" stays text.
    return value.replace(PLACEHOLDER_PATTERN, (match, name: string) => (values.has(name) ? String(values.get(name)) : match));
  }
  if (Array.isArray(value)) {
    return value.map((item) => substituteValue(item, values));
  }
  if (isPlainObject(value)) {
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      result[key] = substituteValue(item, values);
    }
    return result;
  }
  return value;
}

/** [] = ok. Messages are Korean: they are shown in the settings panel and in a failed job's error. */
export function validateComfyApiWorkflow(workflow: unknown): string[] {
  if (!isPlainObject(workflow)) {
    return ["워크플로가 JSON 객체가 아닙니다. ComfyUI에서 'Export (API)'로 내보낸 JSON을 넣어 주세요."];
  }
  if (Array.isArray(workflow.nodes) && Array.isArray(workflow.links)) {
    return ["ComfyUI 화면 저장 형식(nodes/links)입니다. ComfyUI 메뉴의 'Export (API)'로 내보낸 API 형식 JSON을 넣어 주세요."];
  }
  const wrapped = workflow.prompt;
  if (isPlainObject(wrapped) && Object.values(wrapped).some((value) => isPlainObject(value) && typeof value.class_type === "string")) {
    return ["/prompt 요청 본문 전체가 들어왔습니다. \"prompt\" 안의 노드 객체만 넣어 주세요."];
  }
  const entries = Object.entries(workflow);
  if (entries.length === 0) {
    return ["노드가 하나도 없습니다."];
  }
  const errors: string[] = [];
  if (entries.length > MAX_WORKFLOW_NODES) {
    errors.push(`노드가 너무 많습니다 (${entries.length}개, 최대 ${MAX_WORKFLOW_NODES}개).`);
  }
  let hasOutput = false;
  for (const [nodeId, value] of entries) {
    if (!isPlainObject(value)) {
      errors.push(`노드 "${nodeId}"가 객체가 아닙니다.`);
      continue;
    }
    const classType = value.class_type;
    if (typeof classType !== "string" || !classType.trim()) {
      errors.push(`노드 "${nodeId}"에 class_type이 없습니다.`);
      continue;
    }
    if (!isPlainObject(value.inputs)) {
      errors.push(`노드 "${nodeId}" (${classType})의 inputs가 객체가 아닙니다.`);
      continue;
    }
    if (OUTPUT_CLASS_TYPES.has(classType)) {
      hasOutput = true;
    }
    if (loadsImageFile(classType, value.inputs)) {
      // The only photoreal face reference is a profile image DynamicChat generated and the user picked, uploaded by
      // the relay for this request. A file name here would hand back whatever already sits in ComfyUI's input folder
      // — a pass-through graph returns it as the "rendered" cut, and a profile candidate as that face.
      const image = value.inputs.image;
      if (typeof image !== "string" || !COMFY_REFERENCE_VALUE_PATTERN.test(image)) {
        errors.push(
          `노드 "${nodeId}" (${classType})의 입력 "image"는 참조 이미지 자리표시자(%image1%~%image3%) 전체여야 합니다. ComfyUI 입력 폴더의 다른 파일은 쓰지 않습니다.`
        );
      }
    }
    for (const [inputName, inputValue] of Object.entries(value.inputs)) {
      const where = `노드 "${nodeId}" (${classType})의 입력 "${inputName}"`;
      if (GUIDANCE_SCALE_INPUT_PATTERN.test(inputName)) {
        errors.push(...checkGuidanceScale(where, inputValue));
      }
      if (Array.isArray(inputValue)) {
        errors.push(...checkLink(workflow, where, inputValue));
        continue;
      }
      errors.push(...checkPlaceholders(where, inputName, inputValue));
    }
  }
  if (!hasOutput) {
    errors.push("결과 이미지를 내보내는 PreviewImage 또는 SaveImage 노드가 없습니다.");
  }
  return [...new Set(errors)];
}

/** Stand-ins for every supported placeholder, so a template can be validated before any cut exists. */
const TEMPLATE_SAMPLE_VALUES: Record<string, string | number> = {
  prompt: "sample prompt",
  negative_prompt: "",
  seed: 0,
  steps: 20,
  scale: 4,
  cfg: 4,
  width: 1024,
  height: 1024,
  denoise: 1,
  clip_skip: -1,
  sampler: "euler",
  scheduler: "simple",
  model: "model.safetensors",
  text_encoder: "text_encoder.safetensors",
  vae: "vae.safetensors",
  filename_prefix: "DynamicChat/template_check",
  image1: "template_reference_1",
  image2: "template_reference_2",
  image3: "template_reference_3"
};

/**
 * Validates the custom workflow TEXT a creator pastes into settings. validateComfyApiWorkflow on the raw template
 * would report %prompt% and friends as unresolved — they are supposed to be there — so the supported placeholders
 * are filled with stand-ins first and only real problems remain: a UI-format export, a broken link, a missing
 * output, or a placeholder DynamicChat does not supply. An empty text is not an error here (nothing to check yet).
 *
 * Structure only. Whether the template's own words pass the safety floor is the orchestrator's
 * validatePhotorealCustomWorkflow, which adds that reading to these errors.
 */
export function validateCustomWorkflowTemplate(workflowJson: string): string[] {
  if (typeof workflowJson !== "string" || !workflowJson.trim()) {
    return [];
  }
  try {
    const built = validateComfyApiWorkflow(substituteComfyPlaceholders(workflowJson, TEMPLATE_SAMPLE_VALUES));
    return [...new Set([...built, ...checkTemplatePlaceholderPlacement(parseTemplate(workflowJson))])];
  } catch (error) {
    return [error instanceof Error ? error.message : String(error)];
  }
}

const COMFY_REFERENCE_VALUE_PATTERN = /^%ref:[A-Za-z0-9_.-]{1,64}%$/u;

/**
 * A node that reads an image file by name: the LoadImage family (LoadImage, LoadImageMask, LoadImageOutput), and any
 * node with a string `image` input — an IMAGE socket is always a link, so a string there is a file name.
 */
function loadsImageFile(classType: string, inputs: Record<string, unknown>): boolean {
  return /^LoadImage/u.test(classType) || typeof inputs.image === "string";
}

/**
 * A guidance scale must be a number of at least PHOTOREAL_MIN_CFG, written out — as a literal or as the number a
 * whole-value %cfg% became. A link is refused rather than followed: whatever feeds it (a PrimitiveFloat, a math node,
 * a custom node) cannot be shown here to stay at 1 or above. ComfyUI turns a numeric string into a float before it
 * samples, so "0.5" is read as the 0.5 it becomes. An unresolved placeholder is left to checkPlaceholders.
 */
function checkGuidanceScale(where: string, value: unknown): string[] {
  if (typeof value === "string" && /^%[A-Za-z_][A-Za-z0-9_]*%$/u.test(value)) {
    return [];
  }
  if (Array.isArray(value)) {
    return [`${where}: CFG는 숫자로 적거나 %cfg%로 받아야 합니다. 다른 노드의 출력(링크)으로 받으면 1 이상인지 확인할 수 없어 쓰지 않습니다.`];
  }
  const scale = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : Number.NaN;
  if (!Number.isFinite(scale) || scale < PHOTOREAL_MIN_CFG) {
    return [
      `${where}: CFG는 ${PHOTOREAL_MIN_CFG} 이상의 숫자여야 합니다 (받은 값: ${truncateForMessage(JSON.stringify(value) ?? String(value))}). 1보다 작으면 샘플러가 네거티브 쪽으로 그려 네거티브 문구가 그림이 됩니다.`
    ];
  }
  return [];
}

function truncateForMessage(text: string): string {
  return text.length > 40 ? `${text.slice(0, 40)}…` : text;
}

/** In API format every list-valued input is a link; a literal list has to be wrapped as {"__value__": [...]}. */
function checkLink(workflow: Record<string, unknown>, where: string, value: unknown[]): string[] {
  if (value.length !== 2 || (typeof value[0] !== "string" && typeof value[0] !== "number") || typeof value[1] !== "number") {
    return [`${where}: 링크는 ["노드 id", 출력 번호] 형태여야 합니다.`];
  }
  const errors: string[] = [];
  const [target, outputIndex] = value;
  if (typeof target === "number") {
    errors.push(`${where}: 링크의 노드 id는 문자열이어야 합니다 (["${target}", ${outputIndex}]).`);
  } else if (!Object.prototype.hasOwnProperty.call(workflow, target)) {
    errors.push(`${where}이(가) 없는 노드 "${target}"을(를) 가리킵니다.`);
  }
  if (!Number.isInteger(outputIndex) || outputIndex < 0) {
    errors.push(`${where}: 출력 번호 ${outputIndex}이(가) 올바르지 않습니다.`);
  }
  return errors;
}

function checkPlaceholders(where: string, inputName: string, value: unknown): string[] {
  const errors: string[] = [];
  const visit = (item: unknown) => {
    if (typeof item === "string") {
      const ref = item.match(/^%ref:([^%]*)%$/u);
      if (ref) {
        if (!COMFY_REFERENCE_NAME_PATTERN.test(ref[1])) {
          errors.push(`${where}: 참조 이미지 이름 "${ref[1]}"이(가) 올바르지 않습니다 (영문·숫자·._- 64자 이내).`);
        }
        return;
      }
      if (/%ref:/iu.test(item)) {
        errors.push(`${where}: 참조 이미지 자리표시자는 입력 값 전체여야 합니다 (예: LoadImage의 image 값으로 %image1%).`);
      }
      for (const match of item.matchAll(PLACEHOLDER_PATTERN)) {
        const name = match[1];
        if (inputName === "filename_prefix" && COMFY_FILENAME_TOKENS.has(name)) {
          continue;
        }
        errors.push(
          SILLYTAVERN_AVATAR_PLACEHOLDERS.has(name)
            ? `${where}: %${name}%는 SillyTavern 전용 자리표시자입니다. DynamicChat은 LoadImage의 %image1%~%image3%로 참조 이미지를 넣습니다.`
            : `${where}: 치환되지 않은 자리표시자 %${name}%가 남아 있습니다.`
        );
      }
      return;
    }
    if (Array.isArray(item)) {
      item.forEach(visit);
    } else if (isPlainObject(item)) {
      Object.values(item).forEach(visit);
    }
  };
  visit(value);
  return errors;
}

/**
 * A custom workflow's result nodes: its SaveImage nodes when it has any, else its PreviewImage nodes. The relay
 * returns exactly the images of the nodes named here, so a graph that saves its result and previews an intermediate
 * step comes back with the saved image only.
 */
function findOutputNodeIds(workflow: ComfyApiWorkflow): string[] {
  const ids = Object.keys(workflow);
  const save = ids.filter((id) => workflow[id]?.class_type === "SaveImage");
  return save.length > 0 ? save : ids.filter((id) => workflow[id]?.class_type === "PreviewImage");
}

export function resolvePhotorealModelFiles(
  settings: PhotorealApiSettings,
  presetId: PhotorealWorkflowPresetId
): Required<PhotorealModelFiles> {
  const preset = findPreset(presetId);
  const defaults = (preset ?? PHOTOREAL_WORKFLOW_PRESETS.custom).defaults;
  const overrides = (preset && settings?.modelFiles?.[preset.id]) || {};
  // An empty or whitespace-only override means "the preset's file" — the settings form saves cleared fields as "".
  return {
    diffusionModel: cleanFileName(overrides.diffusionModel) || defaults.diffusionModel,
    textEncoder: cleanFileName(overrides.textEncoder) || defaults.textEncoder,
    vae: cleanFileName(overrides.vae) || defaults.vae
  };
}

/**
 * The lowest guidance scale any photoreal graph samples at. A sampler mixes uncond + cfg × (cond − uncond), where
 * "uncond" is the NEGATIVE conditioning: below 1 the result moves toward the negative, and at 0 it is the negative
 * alone — measured by the red team, "a nude 12 yo girl, naked child" in the negative at CFG 0.1 was the subject. The
 * negative is the one text the floor does not read (it says "nude" to keep nudity out), so it is only safe while it
 * stays a push AWAY. At exactly 1 ComfyUI skips the negative branch altogether. Held in settings
 * (normalizePhotorealSettings), in every graph built here, in every custom template (checkGuidanceScale) and at the
 * relay for any graph posted to it.
 */
export const PHOTOREAL_MIN_CFG = 1;

/**
 * A guidance scale as it may be sampled at: a positive setting is kept within PHOTOREAL_MIN_CFG..30 — 0.5 becomes 1,
 * not the preset's own value, because a user who lowered it wanted the weakest guidance there is — and anything else
 * (0, a negative, NaN: "not set") is the preset's own value, which is never below 1.
 */
function clampPhotorealCfg(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.min(30, Math.max(PHOTOREAL_MIN_CFG, value)) : fallback;
}

/**
 * The CFG a cut on `presetId` is sampled at: a distilled preset's fixed value, else the setting (0 = the preset's
 * own) held to PHOTOREAL_MIN_CFG. The builder applies the same rule to whatever it is handed, so the number a plan
 * records is the one sent.
 */
export function resolvePhotorealCfg(settings: Pick<PhotorealApiSettings, "cfg"> | undefined, presetId: PhotorealWorkflowPresetId): number {
  const preset = findPreset(presetId) ?? PHOTOREAL_WORKFLOW_PRESETS.custom;
  if (preset.fixedCfg !== undefined) {
    return preset.fixedCfg;
  }
  return clampPhotorealCfg(settings?.cfg, preset.defaults.cfg);
}

/** Output sizes for an aspect-derived cut: about 1.0-1.3 megapixels, every side a multiple of 16. */
export const PHOTOREAL_OUTPUT_SIZES: ReadonlyArray<{ label: string; width: number; height: number }> = [
  { label: "1:1", width: 1024, height: 1024 },
  { label: "3:4", width: 960, height: 1280 },
  { label: "2:3", width: 832, height: 1248 },
  { label: "9:16", width: 768, height: 1344 },
  { label: "4:3", width: 1280, height: 960 },
  { label: "3:2", width: 1248, height: 832 },
  { label: "16:9", width: 1344, height: 768 }
];

/** The side range of every photoreal size: the stored setting, the resolved cut size and the built graph. */
export const PHOTOREAL_MIN_SIDE = 512;
export const PHOTOREAL_MAX_SIDE = 2048;

/**
 * Snaps one side onto the grid the photo models work on — 8x VAE downsampling then 2x2 patches means multiples of 16
 * — within PHOTOREAL_MIN_SIDE..PHOTOREAL_MAX_SIDE. The one rule for sizes: normalizePhotorealSettings stores what this
 * returns, so the settings field shows the size a cut is actually rendered at. (Its own 256 floor once let it show
 * 304 for a cut rendered at 512.) Both bounds are multiples of 16, so snapping and clamping commute.
 */
export function snapPhotorealSide(value: number): number {
  return Math.min(PHOTOREAL_MAX_SIDE, Math.max(PHOTOREAL_MIN_SIDE, Math.round(value / 16) * 16));
}

export function resolvePhotorealOutputSize(state: AppState): { width: number; height: number } {
  const settings = state.photoreal;
  const width = positiveNumber(settings?.width);
  const height = positiveNumber(settings?.height);
  const profileAspect = readAspect(state.imageProfile?.width, state.imageProfile?.height);
  if (width && height) {
    return { width: snapPhotorealSide(width), height: snapPhotorealSide(height) };
  }
  // One explicit side: the other follows the image profile's aspect, so a "width 1280" setting keeps the shape.
  if (width) {
    return { width: snapPhotorealSide(width), height: snapPhotorealSide(width / (profileAspect ?? 1)) };
  }
  if (height) {
    return { width: snapPhotorealSide(height * (profileAspect ?? 1)), height: snapPhotorealSide(height) };
  }
  // NovelAI's sizes (832x1216, ...) are not multiples of 16 in both sides and sit below a megapixel. The photo
  // models want both, so the cut keeps the profile's SHAPE at the nearest native bucket rather than its pixels.
  const aspect = profileAspect ?? 1;
  let best = PHOTOREAL_OUTPUT_SIZES[0];
  for (const size of PHOTOREAL_OUTPUT_SIZES) {
    if (Math.abs(Math.log(aspect / (size.width / size.height))) < Math.abs(Math.log(aspect / (best.width / best.height)))) {
      best = size;
    }
  }
  return { width: best.width, height: best.height };
}

function positiveNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

function readAspect(width: unknown, height: unknown): number | undefined {
  const w = positiveNumber(width);
  const h = positiveNumber(height);
  return w && h ? w / h : undefined;
}

const DEFAULT_PHOTOREAL_PRESET: PhotorealWorkflowPresetId = "qwen_image_2_1_turbo";

export function resolvePhotorealPresetForCut(settings: PhotorealApiSettings, hasReferences: boolean): PhotorealWorkflowPresetId {
  const referencePreset = findPreset(settings?.referenceWorkflowPreset);
  if (hasReferences && settings.faceReferenceEnabled && referencePreset?.supportsReferences && referencePreset.maxReferences > 0) {
    return referencePreset.id;
  }
  // A preset id this build does not know (state from a newer build) falls back to the shipped default.
  return findPreset(settings?.workflowPreset)?.id ?? DEFAULT_PHOTOREAL_PRESET;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
