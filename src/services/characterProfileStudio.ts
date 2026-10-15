import { createId } from "../lib/id";
import type {
  AppState,
  CharacterProfileImage,
  CharacterVisualProfile,
  Id,
  ImageAsset,
  ImageBackendId,
  ImageCue,
  ImageCueFrame,
  ImageGenerationJob,
  NovelAiModelPreset,
  PhotorealWorkflowPresetId
} from "../types";
import { PHOTOREAL_WORKFLOW_PRESETS } from "./comfyWorkflows";
import { executeImageJob, planImageJob, screenPhotorealReferenceImage, screenPhotorealText } from "./imageOrchestrator";
import type { PhotorealReferenceRefusal, PhotorealTextScreen } from "./imageOrchestrator";
import { sanitizePhotorealNegativePrompt } from "./photorealPrompt";
import { DEFAULT_PHOTOREAL_SETTINGS } from "./runtimeApiSettings";

// Test generation for a character's profile image, from the creation screen.
//
// The creator types a description, generates a few candidates on NovelAI or the photoreal backend, and picks one
// as the character's profile image — which from then on is also that backend's face reference for the character.
// This module is the whole pipeline behind that panel, so the App only wires buttons to it: it plans the
// candidates through the SAME planImageJob the story uses (routing, the safety floor, the photoreal prompt
// builder, the reference wiring), runs them through the same executeImageJob, and turns the chosen image into the
// record that gets persisted. Nothing here writes App state, and no profile image is ever made from anything but
// an image DynamicChat itself generated (there is deliberately no upload path).

export type CharacterProfileTestVariant = "portrait" | "reference_check";

/** The panel offers 1-4 candidates; each is its own request, so the cap is also a cost bound. */
const MAX_PROFILE_TEST_CANDIDATES = 4;
/** Candidate seeds stay in 0..2^31-1, which both backends reproduce exactly (see comfyUiClient). */
const PROFILE_SEED_MODULUS = 2 ** 31;
const PROFILE_TEST_TURN_ID = "profile_test";

/** A front-facing waist-up portrait: the face large and unobstructed, which is what a reference needs. */
const PORTRAIT_FRAME: ImageCueFrame = {
  shot: "upper_body",
  viewpoint: "front",
  angle: "eye_level",
  visibleRegions: ["head", "torso"]
};

/**
 * A deliberately DIFFERENT shot from the portrait the profile was picked from — wider, seated, head turned — so
 * the check shows whether the face survives a new pose, rather than re-rendering the reference itself.
 */
const REFERENCE_CHECK_FRAME: ImageCueFrame = {
  shot: "cowboy_shot",
  viewpoint: "front",
  angle: "eye_level",
  visibleRegions: ["head", "torso", "hips", "legs"]
};
const REFERENCE_CHECK_POSE_TAGS = ["sitting", "looking to the side"];

/** The face check's refusal, per reason the floor gives for not using a profile image (PhotorealReferenceRefusal). */
const REFERENCE_REFUSAL_MESSAGES: Record<PhotorealReferenceRefusal["kind"], string> = {
  not_picked: "이 실사 프로필 이미지는 DynamicChat이 생성한 후보 중에서 고른 이미지가 아니어서 얼굴 참조로 쓸 수 없습니다.",
  no_record: "이 실사 프로필 이미지에는 만든 프롬프트 기록이 없어 얼굴 참조로 쓸 수 없습니다.",
  unreadable_record: "이 실사 프로필 이미지의 프롬프트 기록에는 읽을 수 있는 영어 단어가 없어 얼굴 참조로 쓸 수 없습니다.",
  floor: "이 실사 프로필 이미지는 실사 안전 기준(미성년·노출·성적 표현)에 걸리는 프롬프트로 만들어져 얼굴 참조로 쓸 수 없습니다."
};

const NOVELAI_MODEL_PRESETS: ReadonlySet<string> = new Set<NovelAiModelPreset>([
  "NAID5F",
  "NAID5C",
  "NAID5",
  "NAID4.5F",
  "NAID4.5C",
  "NAID4.0F",
  "NAID4.0C",
  "NAID3"
]);

/**
 * Plans `count` candidate jobs for one character, each with its own seed.
 *
 * The typed prompt is the WHOLE description: the character's saved identity, default outfit, continuity state
 * and existing profile images are left out of the plan, so editing the hair colour in the panel changes the
 * hair colour instead of fighting the saved tags. For photoreal a prose prompt is taken as the appearance text
 * (what CharacterVisualProfile.photorealPrompt holds); a tag list is taken as tags, as for NovelAI.
 *
 * A photoreal request the safety floor locks is refused with an Error, not planned on NovelAI. A test the panel
 * labels free (it runs on this PC) must never quietly spend Anlas on the NovelAI account, and a face check
 * rendered without the face proves nothing. The floor reads the typed prompt (prose or tags), the photoreal style
 * and the finished photo prompt, like any story cut; a face check also re-reads the prompt the profile image was
 * made from, since that image becomes the reference.
 */
export function planCharacterProfileTestJobs(
  state: AppState,
  input: {
    characterId: Id;
    backend: ImageBackendId;
    prompt: string;
    negativePrompt?: string;
    variant: CharacterProfileTestVariant;
    count: number;
  }
): ImageGenerationJob[] {
  const character = state.characters.find((candidate) => candidate.id === input.characterId);
  const profile = state.visualProfiles.find((candidate) => candidate.characterId === input.characterId);
  if (!character && !profile) {
    throw new Error("테스트할 캐릭터를 찾을 수 없습니다. 캐릭터를 저장한 뒤 다시 시도하세요.");
  }
  const backend: ImageBackendId = input.backend === "photoreal" ? "photoreal" : "novelai";
  const referenceImage = profile?.profileImages?.photoreal;
  if (input.variant === "reference_check" && (backend !== "photoreal" || !referenceImage?.objectKey?.trim())) {
    throw new Error("얼굴 고정 테스트는 실사 프로필 이미지가 지정된 캐릭터에서만 할 수 있습니다. 먼저 실사 후보를 생성해 프로필로 지정하세요.");
  }
  const referenceRefusal = input.variant === "reference_check" ? screenPhotorealReferenceImage(state, referenceImage) : undefined;
  if (referenceRefusal) {
    // The planner would leave this face out of every photo cut; a check without it would be a portrait labelled
    // as a pass.
    throw new Error(`${REFERENCE_REFUSAL_MESSAGES[referenceRefusal.kind]} 새 후보를 생성해 프로필을 다시 지정하세요.`);
  }

  // An empty box falls back to what the character already has for that backend, so a reference check can be run
  // without retyping the appearance.
  const prompt =
    input.prompt.trim() ||
    (backend === "photoreal" ? profile?.photorealPrompt?.trim() : profile?.positivePrompt?.trim()) ||
    "";
  if (!prompt) {
    throw new Error("테스트 생성 프롬프트가 비어 있습니다. 캐릭터의 외형을 입력하세요.");
  }
  // Read as typed, before planning: a tag-shaped prompt goes through the tag composer, which drops CJK text
  // unread, so planning alone never sees "一个十四岁的韩国少女".
  const promptRefusal = backend === "photoreal" ? screenPhotorealText(state, prompt) : undefined;
  if (promptRefusal) {
    throw new Error(describePhotorealRefusal(promptRefusal));
  }
  const negativePrompt = input.negativePrompt?.trim() ?? "";
  const asAppearance = backend === "photoreal" && !looksLikeTagList(prompt);
  const promptTags = asAppearance ? [] : splitTags(prompt);
  const subjectTag = inferSubjectTag(promptTags, asAppearance ? prompt : "", splitTags(profile?.positivePrompt ?? ""));
  const poseTags = input.variant === "reference_check" ? REFERENCE_CHECK_POSE_TAGS : [];
  const promptSetsGaze = /\blooking\b/iu.test(prompt);

  const cue: ImageCue = {
    shouldGenerate: true,
    reason: input.variant === "reference_check" ? "캐릭터 얼굴 고정 테스트" : "캐릭터 프로필 테스트 생성",
    characters: [input.characterId],
    tags: [],
    baseTags: [
      subjectTag,
      "solo",
      ...(input.variant === "portrait" ? (promptSetsGaze ? [] : ["looking at viewer"]) : ["three-quarter view"]),
      // A plain backdrop isolates the character on NovelAI. Photoreal gets its backdrop from the style preset's
      // ambient setting instead (a clean cream room for daily_look), which reads as a real photo.
      ...(input.variant === "portrait" && backend === "novelai" ? ["simple background"] : [])
    ],
    characterPrompts: [
      {
        characterId: input.characterId,
        // A prose appearance lives on the profile (below); the caption then only needs the subject, so the
        // figure exists and is counted.
        prompt: (asAppearance ? [subjectTag, ...poseTags] : [prompt, ...poseTags]).join(", "),
        ...(backend === "novelai" && negativePrompt ? { negativePrompt } : {})
      }
    ],
    scene: "character profile",
    visualContext: "",
    frame: input.variant === "reference_check" ? REFERENCE_CHECK_FRAME : PORTRAIT_FRAME,
    kind: "profile"
  };

  const studioState = createStudioState(state, input.characterId, profile, character?.name ?? "", {
    backend,
    variant: input.variant,
    appearance: asAppearance ? prompt : undefined,
    negativePrompt
  });
  const count = Math.min(MAX_PROFILE_TEST_CANDIDATES, Math.max(1, Math.round(Number(input.count) || 1)));
  const seeds = createDistinctSeeds(count);
  const jobs = seeds.map((seed): ImageGenerationJob => {
    const job = planImageJob(studioState, PROFILE_TEST_TURN_ID, cue, [], true, { backend, cueKind: "profile" });
    return {
      ...job,
      providerPayload: {
        ...job.providerPayload,
        // executeImageJob sends a planned seed as-is, so the candidates differ and each can be reproduced.
        seed,
        profileTest: { characterId: input.characterId, variant: input.variant, requestedBackend: backend }
      }
    };
  });

  const first = jobs[0];
  if (backend === "photoreal" && first && first.backend !== "photoreal") {
    // The prompt itself passed above, so the floor tripped on the style text, the finished photo prompt or the
    // custom workflow's own text — or the custom workflow cannot be built for this test at all.
    if (first.routing?.signals?.includes("photo-graph:unbuildable")) {
      throw new Error(
        `사용자 워크플로로 이 테스트의 실사 그래프를 만들 수 없어 실사로 생성할 수 없습니다 (${first.routing.reason}). 설정의 사용자 워크플로를 고치거나 NovelAI로 바꾸어 테스트하세요.`
      );
    }
    throw new Error(describePhotorealRefusal(undefined, first.routing?.reason));
  }
  // A reference check that ends up without a reference would be an ordinary portrait mislabelled as a pass.
  if (input.variant === "reference_check" && first?.backend === "photoreal") {
    const references = first.providerPayload.photorealReferences;
    if (!Array.isArray(references) || references.length === 0) {
      throw new Error("얼굴 고정 테스트에는 얼굴 참조를 지원하는 워크플로(예: Qwen-Image 2.1)가 필요합니다. 설정의 '얼굴 고정용' 프리셋을 확인하세요.");
    }
  }
  return jobs;
}

/**
 * Runs the candidates one after another (a local GPU renders one image at a time anyway, and NovelAI rate-limits
 * bursts) and reports each image as soon as it exists. An abort stops the loop; a failed candidate does not.
 * Errors are collected once each, so four candidates failing the same way read as one message.
 *
 * `onSettled` fires after every candidate that ran to an end, image or not, so a progress count does not stall on
 * a candidate that failed. A cancelled candidate is not settled; the run simply stops there.
 */
export async function runCharacterProfileTestJobs(
  state: AppState,
  jobs: ImageGenerationJob[],
  options: {
    signal?: AbortSignal;
    onImage?: (asset: ImageAsset, index: number) => void;
    onSettled?: (settled: number, total: number) => void;
  } = {}
): Promise<{ assets: ImageAsset[]; errors: string[] }> {
  const assets: ImageAsset[] = [];
  const errors: string[] = [];
  for (const [index, job] of jobs.entries()) {
    if (options.signal?.aborted) {
      break;
    }
    const result = await executeImageJob(state, job, { signal: options.signal });
    if (result.job.status === "canceled") {
      break;
    }
    if (result.job.status === "completed" && result.assets.length > 0) {
      for (const asset of result.assets) {
        assets.push(asset);
        options.onImage?.(asset, index);
      }
    } else {
      const message = result.job.error?.trim() || "알 수 없는 오류로 후보 이미지를 만들지 못했습니다.";
      if (!errors.includes(message)) {
        errors.push(message);
      }
    }
    options.onSettled?.(index + 1, jobs.length);
  }
  return { assets, errors };
}

/**
 * The record stored as CharacterVisualProfile.profileImages[backend] once the bytes are persisted at
 * `objectKey`. A fresh id every time a profile is picked: an id never names two different images, so nothing that
 * cached the previous profile by id can show it for the new one.
 *
 * The caller that persists the bytes passes the id it stored them under (`assetId`), so the record names its own
 * file in the object store. That id must be minted for this pick, never reused from an earlier one. It also passes the
 * decoded image's `width`/`height`, which win over the provider metadata: the metadata describes the request, while
 * the photoreal reference is head-cropped from the stored size, so the size has to describe the bytes.
 *
 * `prompt` is what the panel held when the candidate was made. The safety floor reads it before the image may serve
 * as a face reference and trusts no record without one, so a candidate made from an empty box (the panel then falls
 * back to the saved appearance) records the prompt the image was actually rendered from instead.
 */
export function createProfileImageRecord(
  asset: ImageAsset,
  input: {
    backend: ImageBackendId;
    objectKey: string;
    referenceObjectKey?: string;
    prompt: string;
    negativePrompt?: string;
    assetId?: Id;
    width?: number;
    height?: number;
  }
): CharacterProfileImage {
  const metadata = asset.providerMetadata ?? {};
  const width = readPositiveNumber(input.width) ?? readMetadataNumber(metadata, "width");
  const height = readPositiveNumber(input.height) ?? readMetadataNumber(metadata, "height");
  const seed = readMetadataNumber(metadata, "seed");
  const negativePrompt = input.negativePrompt?.trim();
  const referenceObjectKey = input.referenceObjectKey?.trim();
  const presetId = metadata.presetId;
  const modelPreset = metadata.modelPreset;
  return {
    backend: input.backend,
    assetId: input.assetId?.trim() || createId("profile"),
    objectKey: input.objectKey,
    ...(asset.mimeType ? { mimeType: asset.mimeType } : {}),
    ...(width !== undefined && width > 0 ? { width } : {}),
    ...(height !== undefined && height > 0 ? { height } : {}),
    // Only NovelAI uses a letterboxed reference canvas (Precise Reference accepts three fixed sizes).
    ...(input.backend === "novelai" && referenceObjectKey ? { referenceObjectKey } : {}),
    prompt: input.prompt.trim() ? input.prompt : asset.prompt,
    ...(input.backend === "photoreal" && asset.prompt?.trim() ? { photoPrompt: asset.prompt } : {}),
    ...(negativePrompt ? { negativePrompt } : {}),
    ...(seed !== undefined ? { seed } : {}),
    ...(input.backend === "photoreal" && isPhotorealPresetId(presetId) ? { workflowPreset: presetId } : {}),
    ...(input.backend === "novelai" && typeof modelPreset === "string" && NOVELAI_MODEL_PRESETS.has(modelPreset)
      ? { modelPreset: modelPreset as NovelAiModelPreset }
      : {}),
    selectedAt: new Date().toISOString()
  };
}

/**
 * The state the candidates are planned against: the real one, with the target character reduced to what the
 * panel says. Saved identity and outfit tags would otherwise be re-injected on top of the typed prompt, persisted
 * continuity (a Wearing state, an ongoing pose) would dress the portrait in the middle of the story, and an
 * existing profile image would be sent as the face reference for the very candidates meant to replace it.
 */
function createStudioState(
  state: AppState,
  characterId: Id,
  profile: CharacterVisualProfile | undefined,
  displayName: string,
  options: {
    backend: ImageBackendId;
    variant: CharacterProfileTestVariant;
    appearance?: string;
    negativePrompt: string;
  }
): AppState {
  const base: CharacterVisualProfile = profile ?? {
    id: createId("visual"),
    simulationId: state.simulation.id,
    characterId,
    displayName,
    positivePrompt: "",
    negativePrompt: "",
    defaultOutfitPrompt: "",
    outfitPrompts: {},
    expressionPrompts: {},
    referenceImageAssetIds: [],
    defaultSafetyLevel: "safe"
  };
  const keepReference = options.variant === "reference_check" && base.profileImages?.photoreal;
  const studioProfile: CharacterVisualProfile = {
    ...base,
    positivePrompt: "",
    defaultOutfitPrompt: "",
    photorealPrompt: options.appearance,
    // Cleaned by the one negative rule as it enters, as the planner cleans it again: the box is free text, and a photo
    // negative may only say what to keep out, never what to steer toward.
    photorealNegativePrompt:
      options.backend === "photoreal" && options.negativePrompt
        ? sanitizePhotorealNegativePrompt(options.negativePrompt)
        : base.photorealNegativePrompt,
    profileImages: keepReference ? { photoreal: keepReference } : undefined
  };
  return {
    ...state,
    visualProfiles: [...state.visualProfiles.filter((candidate) => candidate.characterId !== characterId), studioProfile],
    memoryEvents: [],
    // The story's per-session NovelAI count limit guards automatic generation; a manual test in the creation
    // screen is not part of that budget.
    imageJobs: [],
    imageAssets: [],
    // A face check is meaningless without the face, whatever the story-wide toggle says.
    ...(options.variant === "reference_check"
      ? { photoreal: { ...DEFAULT_PHOTOREAL_SETTINGS, ...state.photoreal, faceReferenceEnabled: true } }
      : {})
  };
}

/**
 * Why a photoreal test cannot run, in the panel's words. `screen` is the typed prompt's own reading; without one the
 * prompt passed and the router's reason names what tripped instead (the style text, the finished photo prompt).
 */
function describePhotorealRefusal(screen: PhotorealTextScreen | undefined, routingReason?: string): string {
  if (!screen) {
    const reason = routingReason?.trim();
    return `실사 안전 기준(미성년·노출·성적 표현)에 걸려 실사로 생성할 수 없습니다${reason ? ` (${reason})` : ""}. 설정의 실사 스타일 문장이나 사용자 워크플로를 확인하거나 NovelAI로 바꾸어 테스트하세요.`;
  }
  const { classification } = screen;
  const unreadableOnly =
    classification.unverifiable &&
    classification.category !== "explicit" &&
    !classification.minorCoded &&
    !(classification.schoolCoded && classification.suggestive);
  return unreadableOnly
    ? "이 프롬프트에는 확인할 수 없는 문자(영문 외 문자)가 있어 실사로 생성할 수 없습니다. 영어로 적거나 NovelAI로 바꾸어 테스트하세요."
    : "이 프롬프트는 실사 안전 기준(미성년·노출·성적 표현)에 걸려 실사로 생성할 수 없습니다. NovelAI로 바꾸어 테스트하세요.";
}

/**
 * Whether a photoreal test prompt is a tag list rather than prose. Articles, pronouns and copulas never occur in
 * Danbooru tags and carry every sentence; subject counts and NovelAI weight syntax never occur in prose.
 */
function looksLikeTagList(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) {
    return false;
  }
  if (/(?:^|[\s,])\d+\+?\s*(?:girls?|boys?|others?)\b/iu.test(trimmed) || /::|[{}[\]]/u.test(trimmed)) {
    return true;
  }
  if (/\b(?:a|an|the|she|he|they|her|his|their|is|are|has|have|wearing|who)\b/iu.test(trimmed)) {
    return false;
  }
  if (/[.!?]\s+\S/u.test(trimmed)) {
    return false;
  }
  const segments = splitTags(trimmed);
  return segments.length > 1 || (segments[0]?.split(/\s+/u).length ?? 0) <= 3;
}

/**
 * One subject-count tag for the base caption: the typed tags' own, else the saved identity's, else the gender
 * the words imply, else 1girl. A profile is always one person.
 */
function inferSubjectTag(promptTags: string[], appearance: string, identityTags: string[]): string {
  for (const source of [promptTags, identityTags]) {
    const counted = source.map((tag) => tag.trim().toLowerCase()).find((tag) => /^1(?:girl|boy|other)$/u.test(tag));
    if (counted) {
      return counted;
    }
  }
  const words = [appearance, ...promptTags, ...identityTags].join(" ").toLowerCase();
  if (/\b(?:woman|women|girl|female|lady|she|her)\b/u.test(words)) {
    return "1girl";
  }
  if (/\b(?:man|men|boy|male|guy|gentleman|he|his)\b/u.test(words)) {
    return "1boy";
  }
  return "1girl";
}

function splitTags(value: string): string[] {
  return value
    .split(/[,\n]/u)
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function createDistinctSeeds(count: number): number[] {
  const seeds = new Set<number>();
  while (seeds.size < count) {
    seeds.add(Math.floor(Math.random() * PROFILE_SEED_MODULUS));
  }
  return [...seeds];
}

function readPositiveNumber(value: number | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.round(value) : undefined;
}

/** Top-level first (the photoreal result and the plan put it there), then NovelAI's `parameters`. */
function readMetadataNumber(metadata: Record<string, unknown>, key: string): number | undefined {
  const direct = metadata[key];
  if (typeof direct === "number" && Number.isFinite(direct) && direct >= 0) {
    return direct;
  }
  const parameters = metadata.parameters;
  if (parameters && typeof parameters === "object" && !Array.isArray(parameters)) {
    const nested = (parameters as Record<string, unknown>)[key];
    if (typeof nested === "number" && Number.isFinite(nested) && nested >= 0) {
      return nested;
    }
  }
  return undefined;
}

function isPhotorealPresetId(value: unknown): value is PhotorealWorkflowPresetId {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(PHOTOREAL_WORKFLOW_PRESETS, value);
}
