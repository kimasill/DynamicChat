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
// What renders photoreal is the user's setting, not the program's: every category — sexual scenes and nudity, dynamic
// action, hips-visible and full-body cuts — goes where the routing mode and the ticked categories send it. This is a
// local app whose owner sets the content rating, and an adult scene drawn as a photograph is theirs to choose.
//
// One line is not a setting: a sexual or suggestive cut whose subject reads as a minor — minor-coded, school-coded
// (a student, a school uniform), or carrying text the classifier cannot read, so its age cannot be checked — is never
// rendered photoreal (photorealFloorLock). It lives here and nowhere else, so every caller inherits it: the planner,
// a manual regeneration that names the photoreal backend, the finished-prompt, custom-workflow, built-graph and
// face-reference screens, the profile studio and the dispatch. When NovelAI is off such a cut stays on NovelAI and is
// setup-blocked like any other NovelAI cut; it is never promoted to photoreal to "make it work". Either half alone
// follows the routing: a child at a family dinner, or an adult in a sex scene.
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

/**
 * The one line photoreal never crosses, in the settings screen's words. Not a category and not a setting: it applies
 * in every mode and to every category (photorealFloorLock).
 */
export const PHOTOREAL_FLOOR_DESCRIPTION =
  "미성년으로 읽히는 인물(학생·교복 포함)이 성적·노출 장면에 있으면 실사로 생성하지 않습니다 (NovelAI)";

export const IMAGE_SCENE_CATEGORY_OPTIONS: Array<{
  id: ImageSceneCategory;
  label: string;
  description: string;
  /**
   * A row the user cannot tick. No category is locked: every one is the user's choice. Kept optional so a screen that
   * still reads it keeps compiling and simply shows every row as selectable.
   */
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
    description:
      "싸움 · 달리기 · 점프 · 추락 · 스포츠처럼 움직임이 큰 컷. 실사 모델은 역동적인 동작에 약해 팔다리·자세가 무너지기 쉬워 기본으로는 NovelAI가 그립니다. 실사로 보고 싶으면 체크하세요."
  },
  {
    id: "fantasy",
    label: "판타지 · 비인간 요소",
    description: "동물 귀 · 날개 · 뿔 · 꼬리 · 마법 · 몬스터 같은 비인간 요소가 있는 컷"
  },
  {
    id: "explicit",
    label: "성적 장면 · 노출",
    description:
      "성인 캐릭터의 노출이나 성적 묘사가 있는 컷 (목욕·온천처럼 노출이 암시된 컷과 노출 목적 의상 포함). 체크하면 실사로 그립니다. 미성년으로 읽히는 인물(학생·교복 포함)이 있는 컷은 체크와 상관없이 NovelAI로 그립니다."
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
 * its old behaviour (`novelai`). Unknown categories are dropped and the order is canonical; every known category is
 * kept, `explicit` included — it is the user's choice like the rest. An explicitly empty category list is kept:
 * unticking everything is a legitimate choice.
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
 * 1. The floor (photorealFloorLock): a sexual or suggestive cut whose subject reads as a minor, or whose age cannot
 *    be read → NovelAI, `locked`. Beats the forced backend, the mode and availability. Nothing else is locked.
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

  const lockReason = photorealFloorLock(classification);
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

/** What the floor reads of a classification: what the picture shows, and who it shows. */
export type PhotorealFloorReading = Pick<SceneClassification, "category" | "suggestive" | "minorCoded" | "schoolCoded" | "unverifiable">;

/**
 * Why a cut may never render photoreal, or undefined when the user's routing decides. The floor is a conjunction:
 * - the picture is sexual or suggestive — an `explicit` cut, or suggestive wardrobe and framing (lingerie, a bikini,
 *   an upskirt);
 * - and its subject reads as a minor (minorCoded), as a pupil (schoolCoded: a student, a school uniform written out or
 *   as its parts), or cannot be read at all (unverifiable: kana, Cyrillic, CJK — the photo encoders read them, no
 *   lexicon here can, so nothing can say the subject is an adult).
 * Each half alone is the user's to route: an adult in a sex scene renders photoreal if the routing says so, and so
 * does a child at a family dinner (the photo builder still words every subject as an adult). The flags are read as
 * `=== true`, so a classification built before a flag existed reads it as absent.
 */
export function photorealFloorLock(reading: PhotorealFloorReading): string | undefined {
  const sexual = reading.category === "explicit" || reading.suggestive === true;
  if (!sexual) {
    return undefined;
  }
  if (reading.minorCoded === true) {
    return "미성년으로 읽히는 인물이 성적·노출 장면에 있어 실사로 생성하지 않습니다 (NovelAI 고정)";
  }
  if (reading.schoolCoded === true) {
    return "학생 · 교복 요소가 성적·노출 장면에 있어 실사로 생성하지 않습니다 (NovelAI 고정)";
  }
  if (reading.unverifiable === true) {
    // Also reached by text that goes to the encoder as written (readVerbatimFloorText): there the scene itself is
    // unknown, so the reason names what is unknown rather than claiming the cut is sexual.
    return "확인할 수 없는 문자(영문 외 문자)가 있어 장면과 인물의 나이를 확인할 수 없으므로 실사로 생성하지 않습니다 — 그 글(사용자 워크플로 문구 등)을 영어로 바꾸면 실사로 생성됩니다 (NovelAI 고정)";
  }
  return undefined;
}

/**
 * The reading of text that reaches the photo encoder AS WRITTEN — a finished photo prompt, a custom workflow's own
 * literals, a built graph's strings. Everything the builder writes has had what it cannot read dropped, which is why
 * unreadable text alone does not lock a cut. Nothing drops it from these, and text no lexicon can read cannot be shown
 * to be anything but sexual: the encoders read "裸の少女" as a naked girl. So here it counts as sexual as well as
 * age-unknown, and trips the floor on its own.
 */
export function readVerbatimFloorText<T extends PhotorealFloorReading>(reading: T): T {
  return reading.unverifiable === true ? { ...reading, suggestive: true } : reading;
}

/**
 * Two readings of what ends up in ONE picture, as one: a text the photo model receives and the cut it is rendered
 * with, a face reference's record and the cut it is attached to. The floor's halves can sit in different texts — a
 * custom workflow that says "a 14-year-old girl" around a cut that is a sex scene, a profile image drawn from a
 * minor-coded prompt attached to a suggestive cut — and neither text alone trips it. The result keeps the first
 * reading's other fields; the category is explicit when either is.
 */
export function joinFloorReadings<T extends PhotorealFloorReading>(reading: T, other: PhotorealFloorReading | undefined): T {
  if (!other) {
    return reading;
  }
  return {
    ...reading,
    category: other.category === "explicit" ? "explicit" : reading.category,
    suggestive: reading.suggestive === true || other.suggestive === true || other.category === "explicit",
    minorCoded: reading.minorCoded === true || other.minorCoded === true,
    schoolCoded: reading.schoolCoded === true || other.schoolCoded === true,
    unverifiable: reading.unverifiable === true || other.unverifiable === true
  };
}

function describeImageSceneCategory(category: ImageSceneCategory): string {
  return IMAGE_SCENE_CATEGORY_OPTIONS.find((option) => option.id === category)?.label ?? category;
}

function sanitizePhotorealCategories(values: unknown[]): ImageSceneCategory[] {
  const wanted = new Set(values.map(readCategoryId).filter((value): value is ImageSceneCategory => Boolean(value)));
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
