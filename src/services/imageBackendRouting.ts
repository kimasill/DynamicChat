import type {
  AppState,
  ImageBackendDecision,
  ImageBackendId,
  ImageBackendRouting,
  ImageBackendRoutingMode,
  ImageGenerationProfile,
  ImageSceneCategory
} from "../types";
import type { SceneClassification } from "./imageSceneClassifier";

// Per-cut backend routing: which renderer draws a cut, given its scene category and the simulation's routing
// mode.
//
// The safety floor lives here and nowhere else, so every caller inherits it: explicit cuts, minor-coded cuts,
// school-coded cuts with suggestive content and cuts carrying text the classifier cannot read are planned on
// NovelAI in every mode, including a manual regeneration that names the photoreal backend. When NovelAI is off such a cut stays on NovelAI and is
// setup-blocked like any other NovelAI cut; it is never promoted to photoreal to "make it work".
//
// Deliberately free of runtime imports from imageOrchestrator.ts: the orchestrator imports this module, and the
// reverse edge would make a cycle that vite's SSR loader (used by every eval) resolves to undefined.

// Frozen because seed state and every simulation created from it share this array by reference until the user
// edits it; resolveImageBackendRouting always returns a fresh copy, so nothing legitimate mutates it.
export const DEFAULT_PHOTOREAL_CATEGORIES: ImageSceneCategory[] = Object.freeze([
  "portrait",
  "full_body",
  "scenery",
  "object"
] as ImageSceneCategory[]) as ImageSceneCategory[];

export const IMAGE_SCENE_CATEGORY_OPTIONS: Array<{
  id: ImageSceneCategory;
  label: string;
  description: string;
  locked?: boolean;
}> = [
  {
    id: "portrait",
    label: "인물 단독 · 얼굴/상반신",
    description: "한 명이 얼굴이나 상반신으로 잡힌 정적인 컷"
  },
  {
    id: "full_body",
    label: "인물 단독 · 전신/의상",
    description: "한 명의 전신이나 의상이 보이는 정적인 컷 (카우보이 샷 이상)"
  },
  {
    id: "scenery",
    label: "배경 · 풍경",
    description: "인물이 없는 배경 · 실내 · 풍경 컷"
  },
  {
    id: "object",
    label: "사물 · 음식",
    description: "인물 없이 사물이나 음식이 주인공인 컷"
  },
  {
    id: "group",
    label: "2인 이상 · 대화/일상",
    description: "두 명 이상이 신체 접촉 없이 함께 있는 대화 · 일상 컷"
  },
  {
    id: "romance",
    label: "스킨십 (비노출)",
    description: "키스 · 포옹 · 손잡기처럼 노출 없는 신체 접촉이 있는 컷"
  },
  {
    id: "action",
    label: "액션 · 역동적 동작",
    description: "싸움 · 달리기 · 점프 · 추락 · 스포츠처럼 움직임이 큰 컷"
  },
  {
    id: "fantasy",
    label: "판타지 · 비인간 요소",
    description: "동물 귀 · 날개 · 뿔 · 꼬리 · 마법 · 몬스터 같은 비인간 요소가 있는 컷"
  },
  {
    id: "explicit",
    label: "성적 장면 — 항상 NovelAI",
    description: "노출이나 성적 묘사가 있는 컷. 안전 정책상 어떤 모드에서도 실사로 생성하지 않습니다.",
    locked: true
  }
];

const ROUTING_MODES: readonly ImageBackendRoutingMode[] = ["novelai", "mixed", "photoreal"];
const CATEGORY_ORDER: readonly ImageSceneCategory[] = IMAGE_SCENE_CATEGORY_OPTIONS.map((option) => option.id);

const MODE_LABELS: Record<ImageBackendRoutingMode, string> = {
  novelai: "NovelAI 전용 모드",
  mixed: "혼합 모드",
  photoreal: "전부 실사 모드"
};

const BACKEND_LABELS: Record<ImageBackendId, string> = {
  novelai: "NovelAI",
  photoreal: "실사"
};

// Same text as imageOrchestrator.describeNovelAiSetupBlock (an eval pins the two together). Duplicated rather
// than imported for the cycle reason above.
const NOVELAI_SETUP_BLOCK = "NovelAI 연동이 꺼져 있습니다. 개인 설정에서 켜면 이 컷부터 생성됩니다.";
const PHOTOREAL_SETUP_BLOCK = "실사 이미지(ComfyUI) 연동이 꺼져 있습니다. 설정에서 켜면 이 컷부터 생성됩니다.";

/**
 * The profile's routing, made safe to act on. A simulation saved before photoreal existed has none and keeps
 * its old behaviour (`novelai`). Unknown categories are dropped, `explicit` is stripped (it can never be
 * photoreal, so a stored tick on it would only mislead the settings screen), and the order is canonical.
 * An explicitly empty category list is kept: unticking everything is a legitimate choice.
 */
export function resolveImageBackendRouting(
  profile: Pick<ImageGenerationProfile, "backendRouting"> | null | undefined
): ImageBackendRouting {
  const raw: unknown = profile?.backendRouting;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { mode: "novelai", photorealCategories: [...DEFAULT_PHOTOREAL_CATEGORIES] };
  }
  const record = raw as Record<string, unknown>;
  return {
    mode: readRoutingMode(record.mode) ?? "novelai",
    photorealCategories: Array.isArray(record.photorealCategories)
      ? sanitizePhotorealCategories(record.photorealCategories)
      : [...DEFAULT_PHOTOREAL_CATEGORIES]
  };
}

export function isPhotorealBackendConfigured(state: AppState): boolean {
  return state.photoreal?.enabled === true;
}

export function isNovelAiBackendConfigured(state: AppState): boolean {
  return state.novelAi?.enabled === true;
}

/**
 * Routes one classified cut.
 *
 * 1. Safety floor: explicit, minor-coded, school-coded + suggestive, or unverifiable text → NovelAI, `locked`.
 *    Beats the forced backend, the mode and availability.
 * 2. Preference: the forced backend (manual "regenerate with the other backend"), else the mode.
 * 3. Availability: a preference for a backend that is off falls back to the other one when that one is on.
 *    Not for a forced backend (the user asked for that renderer; a setup block says why it is waiting), and
 *    never toward photoreal for a locked cut. With both off the preference stands, so the setup-block message
 *    names the backend the cut is actually waiting for.
 */
export function decideImageBackend(
  state: AppState,
  classification: SceneClassification,
  options: { forcedBackend?: ImageBackendId } = {}
): ImageBackendDecision {
  const routing = resolveImageBackendRouting(state.imageProfile);
  const category = classification.category;
  const signals = Array.isArray(classification.signals) ? [...classification.signals] : [];
  const forcedBackend = readBackendId(options.forcedBackend);
  const photorealOn = isPhotorealBackendConfigured(state);
  const novelAiOn = isNovelAiBackendConfigured(state);

  const lockReason = describeSafetyLock(classification);
  if (lockReason) {
    const notes = [
      forcedBackend === "photoreal" ? "실사로 직접 지정해도 바뀌지 않습니다" : undefined,
      novelAiOn ? undefined : "NovelAI 연동을 켜야 생성됩니다"
    ].filter((note): note is string => Boolean(note));
    return {
      backend: "novelai",
      category,
      mode: routing.mode,
      reason: [lockReason, ...notes].join(" · "),
      locked: true,
      fallback: false,
      signals
    };
  }

  const preferred: ImageBackendId =
    forcedBackend ??
    (routing.mode === "photoreal"
      ? "photoreal"
      : routing.mode === "mixed" && routing.photorealCategories.includes(category)
        ? "photoreal"
        : "novelai");

  let backend = preferred;
  let fallback = false;
  if (!forcedBackend) {
    if (preferred === "photoreal" && !photorealOn && novelAiOn) {
      backend = "novelai";
      fallback = true;
    } else if (preferred === "novelai" && !novelAiOn && photorealOn) {
      backend = "photoreal";
      fallback = true;
    }
  }

  const base = `${MODE_LABELS[routing.mode]}: '${describeImageSceneCategory(category)}' 장면 → ${BACKEND_LABELS[backend]}`;
  const reason = forcedBackend
    ? `${base} (직접 지정)`
    : fallback
      ? `${base} (${BACKEND_LABELS[preferred]} 연동이 꺼져 있어 대신 사용)`
      : base;

  return { backend, category, mode: routing.mode, reason, locked: false, fallback, signals };
}

/**
 * Why a cut planned on `backend` cannot run right now, or undefined when it can. The NovelAI text is exactly
 * describeNovelAiSetupBlock's, so cuts planned before and after photoreal existed read the same.
 */
export function describeImageBackendSetupBlock(state: AppState, backend: ImageBackendId): string | undefined {
  if (backend === "photoreal") {
    return isPhotorealBackendConfigured(state) ? undefined : PHOTOREAL_SETUP_BLOCK;
  }
  return isNovelAiBackendConfigured(state) ? undefined : NOVELAI_SETUP_BLOCK;
}

function describeSafetyLock(classification: SceneClassification): string | undefined {
  if (classification.category === "explicit") {
    return "성적 장면은 실사로 생성하지 않습니다 (NovelAI 고정)";
  }
  if (classification.minorCoded) {
    return "미성년자로 보일 수 있는 장면은 실사로 생성하지 않습니다 (NovelAI 고정)";
  }
  if (classification.schoolCoded && classification.suggestive) {
    return "학생 · 교복 요소와 노출 요소가 함께 있는 장면은 실사로 생성하지 않습니다 (NovelAI 고정)";
  }
  // Kana, Cyrillic, CJK and the like cannot be checked by any lexicon, while the photo encoders read them. Read as
  // `=== true` so a classification built before the flag existed still routes as it did.
  if (classification.unverifiable === true) {
    return "확인할 수 없는 문자(영문 외 문자)가 있는 장면은 실사로 생성하지 않습니다 (NovelAI 고정)";
  }
  return undefined;
}

function describeImageSceneCategory(category: ImageSceneCategory): string {
  return IMAGE_SCENE_CATEGORY_OPTIONS.find((option) => option.id === category)?.label ?? category;
}

function sanitizePhotorealCategories(values: unknown[]): ImageSceneCategory[] {
  const wanted = new Set(values.map(readCategoryId).filter((value): value is ImageSceneCategory => Boolean(value)));
  wanted.delete("explicit");
  return CATEGORY_ORDER.filter((category) => wanted.has(category));
}

function readCategoryId(value: unknown): ImageSceneCategory | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const normalized = value.trim().toLowerCase().replace(/[\s-]+/gu, "_");
  return CATEGORY_ORDER.find((category) => category === normalized);
}

function readRoutingMode(value: unknown): ImageBackendRoutingMode | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const normalized = value.trim().toLowerCase();
  return ROUTING_MODES.find((mode) => mode === normalized);
}

function readBackendId(value: unknown): ImageBackendId | undefined {
  return value === "novelai" || value === "photoreal" ? value : undefined;
}
