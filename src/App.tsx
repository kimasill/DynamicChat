import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  Bot,
  Boxes,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  Copy,
  Database,
  GripVertical,
  Home,
  Image as ImageIcon,
  ImagePlus,
  Info,
  KeyRound,
  Layers,
  MessageSquareText,
  Minus,
  Moon,
  MoreHorizontal,
  Network,
  Parentheses,
  Pencil,
  Play,
  Plus,
  RefreshCcw,
  RotateCcw,
  Save,
  ScanFace,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  Sun,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Type as TypeIcon,
  UserRound,
  WandSparkles,
  X
} from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  CSSProperties,
  Dispatch,
  DragEvent as ReactDragEvent,
  FormEvent,
  PointerEvent as ReactPointerEvent,
  ReactElement,
  ReactNode,
  SetStateAction,
  WheelEvent as ReactWheelEvent
} from "react";
import { Children, Component, Fragment, cloneElement, createContext, isValidElement, memo, useCallback, useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";

import { builtInSimulationStates, createDraftPreviewState, createStateFromDraft, hydrateState, seedState, type SimulationDraft } from "./data/seed";
import { createId } from "./lib/id";
import {
  MAX_FOUNDATION_MODULE_BODY_CHARS,
  MAX_MAIN_PROMPT_BODY_CHARS,
  promptModuleBodyCharLimit
} from "./lib/promptLimits";
import { listCliAgentModels, listLocalLlmModels, normalizeApiToken, validateLlmApi, validateNovelAiApi } from "./services/apiValidation";
import {
  canUserRunQueuedImageJob,
  executeImageJob,
  planImageJob,
  screenPhotorealText,
  shouldAutoRunImageJob,
  validatePhotorealCustomWorkflow,
  type ImageJobExecutionResult
} from "./services/imageOrchestrator";
import { generateAssistantText, isLlmAbortedError, setActiveTurnAbortSignal } from "./services/llmClient";
import { LLM_PROVIDER_GROUP_LABELS, LLM_PROVIDER_PRESETS, pickAntigravityReplacementModel, type LlmProviderPreset } from "./services/llmProviders";
import { describeImageCueFrame, normalizeImageCueFrame } from "./services/imageFrame";
import { IMAGE_SCENE_CATEGORY_OPTIONS, PHOTOREAL_FLOOR_DESCRIPTION, resolveImageBackendRouting } from "./services/imageBackendRouting";
import {
  COMFY_CUSTOM_PLACEHOLDERS,
  PHOTOREAL_MIN_CFG,
  PHOTOREAL_WORKFLOW_PRESETS,
  QWEN21_TURBO_SIGMAS,
  resolvePhotorealModelFiles,
  resolvePhotorealOutputSize,
  type PhotorealPresetDownload
} from "./services/comfyWorkflows";
import { getPhotorealRelayBase, probeComfyUi, type ComfyUiProbeResult } from "./services/comfyUiClient";
import { createLocalGpuCoordinator, shouldCoordinateLocalGpu } from "./services/localGpuCoordinator";
import {
  createProfileImageRecord,
  planCharacterProfileTestJobs,
  runCharacterProfileTestJobs,
  type CharacterProfileTestVariant
} from "./services/characterProfileStudio";
import { bytesToBase64, letterboxForNovelAiReference } from "./services/imageReferences";
import { PHOTOREAL_STYLE_PRESETS } from "./services/photorealPrompt";
import { encodeNovelAiVibe } from "./services/novelAiClient";
import {
  NeuralMapClient,
  createPromptModuleSyncSignature,
  isSemanticRetrievalModule,
  type NeuralMapLiveGraph,
  type NeuralMapLiveNode
} from "./services/neuralMapClient";
import {
  DEFAULT_PHOTOREAL_SETTINGS,
  normalizePhotorealSettings,
  toShareableCharacterProfileImages,
  toShareableLlmSettings,
  toShareableNovelAiSettings,
  toShareablePhotorealSettings
} from "./services/runtimeApiSettings";
import {
  readStateMemoryKind,
  readStateMemoryOwnerId,
  readStateMemoryStateType,
  readStateMemoryTargetId,
  readStateMemoryValue
} from "./services/stateMemory";
import {
  clearState,
  configureDynamicChatApiBaseUrl,
  createDynamicChatApiClient,
  DEFAULT_DYNAMICCHAT_API_BASE_URL,
  getConfiguredDynamicChatApiBaseUrl,
  getNovelAiGenerateProxyUrl,
  loadState,
  saveState,
  type SaveStateOptions
} from "./services/dynamicChatApi";
import { createAuditEvent, createRedactionRequest } from "./services/security";
import { createResetSessionState, planImageJobForCompletedTurn, runSimulationTurn } from "./services/simulationEngine";
import { slimProgressRunForStorage } from "./services/progressRuns";
import { activateSimulationProgressRun, createFreshSimulationRun, deleteSimulationProgressRun } from "./services/simulationRuns";
import { isCliAgentLlmProvider } from "./types";
import type {
  AppState,
  ApiRegistrationStatus,
  AssistantImageCueDraft,
  AssistantSidecar,
  CharacterProfileImage,
  CharacterProfileImages,
  ContentRating,
  ContextEvidence,
  ContextPack,
  EvaluationScenario,
  ImageAsset,
  ImageBackendId,
  ImageBackendRouting,
  ImageBackendRoutingMode,
  ImageCue,
  ImageFeedbackRating,
  ImageGenerationCadence,
  ImageGenerationJob,
  ImageGenerationProfile,
  ImageSceneCategory,
  ImageScenePresetExampleFile,
  ImageSceneTagPreset,
  ImageSceneTagPresetNode,
  ImageSafetyLevel,
  ImageTriggerMode,
  LlmApiSettings,
  LlmProvider,
  ChatMessage,
  NeuralMapSettings,
  NovelAiAutomationTermination,
  NovelAiApiSettings,
  NovelAiVibeTransferReference,
  NovelAiModelPreset,
  NovelAiNoiseSchedule,
  PhotorealApiSettings,
  PhotorealStylePresetId,
  PhotorealWorkflowPresetId,
  PromptModule,
  PromptModuleKind,
  PromptModuleUsage,
  RelationshipMapSettings,
  RelationshipStatusParameter,
  SidecarTrace,
  SimulationProgressRun,
  SimulationPromptMode,
  SimulationCharacterDraft,
  TokenPolicy,
  TurnResult,
  TurnTrace,
  UserPersona
} from "./types";

const promptModuleKinds: PromptModuleKind[] = [
  "main_prompt",
  "sub_prompt",
  "character_prompt",
  "world_lore",
  "scene_rule",
  "image_prompt_profile",
  "safety_policy",
  "style_guide"
];

const tokenPolicies: TokenPolicy[] = ["always", "rag", "manual", "disabled"];
const tokenPolicyLabels: Record<TokenPolicy, string> = {
  always: "항상 포함",
  rag: "필요할 때 참조",
  manual: "수동 참조",
  disabled: "사용 안 함"
};
const tokenPolicyShortLabels: Record<TokenPolicy, string> = {
  always: "항상",
  rag: "필요 시",
  manual: "수동",
  disabled: "꺼짐"
};
const activationTagDisplayLabels: Record<string, string> = {
  always: "상시",
  rag: "참조"
};
const activationTagStorageLabels: Record<string, string> = {
  상시: "always",
  항상: "always",
  참조: "rag",
  "필요 시": "rag",
  필요시: "rag"
};
const triggerModes: ImageTriggerMode[] = ["stored_only", "realtime_auto", "realtime_confirm", "manual"];
const outputLengthPresets = [
  { id: "compact", label: "짧게", tokens: 900, detail: "2-3문단" },
  { id: "balanced", label: "보통", tokens: 1600, detail: "4-6문단" },
  { id: "long", label: "길게", tokens: 2600, detail: "7-10문단" },
  { id: "deep", label: "매우 길게", tokens: 4200, detail: "10-14문단" }
] as const;
const imageGenerationCadenceOptions: Array<{
  value: ImageGenerationCadence;
  label: string;
  detail: string;
}> = [
  { value: "sparse", label: "적게", detail: "큰 전환 중심" },
  { value: "balanced", label: "균형", detail: "중요 장면마다" },
  { value: "rich", label: "많게", detail: "행동/감정 변화" },
  { value: "paragraph", label: "문단마다", detail: "문단 단위 cue" },
  { value: "image_progression", label: "이미지 진행", detail: "10컷 태그" }
];
const imageSafetyLevels: ImageSafetyLevel[] = ["safe", "sensitive", "suggestive", "explicit"];
const contentRatingOptions: Array<{ value: ContentRating; label: string; detail: string }> = [
  { value: "general", label: "일반", detail: "기본 진행 등급" },
  { value: "adult_19", label: "19+ 성인 전용", detail: "성인용 API/이미지 수위로 전환" }
];
const novelAiModelPresets: NovelAiModelPreset[] = ["NAID5F", "NAID5C", "NAID5", "NAID4.5F", "NAID4.5C", "NAID4.0F", "NAID4.0C", "NAID3"];
const novelAiSamplers = ["k_euler", "k_euler_ancestral", "k_dpmpp_2m", "k_dpmpp_2s_ancestral", "k_dpmpp_sde", "k_dpmpp_2m_sde", "ddim_v3"];
const novelAiNoiseSchedules: NovelAiNoiseSchedule[] = ["karras", "native", "exponential", "polyexponential"];
const novelAiAutomationTerminations: NovelAiAutomationTermination[] = ["unlimited", "timer", "count"];
const resolutionPresets = [
  { label: "Portrait 832x1216", width: 832, height: 1216 },
  { label: "Square 1024x1024", width: 1024, height: 1024 },
  { label: "Landscape 1216x832", width: 1216, height: 832 }
];
const imageJobStatusLabels: Record<ImageGenerationJob["status"], string> = {
  queued: "대기",
  planning: "준비",
  generating: "생성 중",
  completed: "완료",
  failed: "실패",
  canceled: "취소"
};
/** A cut whose render stopped because the user left its run (or reloaded the page) before it finished. */
const ABANDONED_IMAGE_JOB_ERROR = "생성이 끝나기 전에 진행을 떠나거나 페이지를 새로 고쳐 중단되었습니다. 재생성하면 다시 만듭니다.";
const imageAssetSourceLabels: Record<ImageAsset["source"], string> = {
  stored: "기본 에셋",
  generated: "생성 결과",
  fallback: "대체 이미지"
};
// Long form for copy ("실사 이미지 생성 실패"), short form for the badge on the picture itself.
const imageBackendLabels: Record<ImageBackendId, string> = {
  novelai: "NovelAI",
  photoreal: "실사"
};
const imageBackendBadgeLabels: Record<ImageBackendId, string> = {
  novelai: "NAI",
  photoreal: "실사"
};
const imageBackendRoutingModeOptions: Array<{ value: ImageBackendRoutingMode; label: string; detail: string }> = [
  { value: "novelai", label: "NAI만", detail: "모든 컷을 NovelAI로" },
  { value: "mixed", label: "장면별 혼합", detail: "고른 장면만 실사" },
  { value: "photoreal", label: "전부 실사", detail: "모든 컷을 실사로" }
];
// Precise Reference exists only on these two; every other preset draws the cut without the face reference.
const novelAiCharacterReferencePresets: ReadonlySet<NovelAiModelPreset> = new Set<NovelAiModelPreset>(["NAID4.5F", "NAID4.5C"]);
// Same fallbacks novelAiClient applies to an unset knob, so the sliders show what will actually be sent.
const NOVELAI_CHARACTER_REFERENCE_DEFAULT_STRENGTH = 0.85;
const NOVELAI_CHARACTER_REFERENCE_DEFAULT_FIDELITY = 1;
const photorealWorkflowPresetOrder: PhotorealWorkflowPresetId[] = [
  "qwen_image_2_1_turbo",
  "qwen_image_2_1",
  "z_image_turbo",
  "qwen_image_edit_2511",
  "custom"
];
const photorealStylePresetOrder: Array<Exclude<PhotorealStylePresetId, "custom">> = [
  "instagram",
  "daily_look",
  "raw_snap",
  "film_portrait",
  "cinematic",
  "studio_editorial"
];
const PHOTOREAL_SHARED_APPEARANCE_PLACEHOLDER =
  "e.g. beautiful realistic Korean woman, soft Korean makeup, delicate facial features, glossy lips, elegant Korean beauty";
// The 31-bit range the ComfyUI client keeps seeds in, so the seed typed here is the seed the graph receives.
const PHOTOREAL_MAX_SEED = 2_147_483_647;

// Provider metadata (label, base URL, model list, key placeholder) and runtime capabilities (wire format,
// JSON mode, temperature ceiling, extra headers/body, CORS routing) live together in one registry so adding
// a backend is one entry rather than four hand-maintained branch chains.
const llmProviderOptions = LLM_PROVIDER_PRESETS;

const llmProviderGroupOrder: LlmProviderPreset["group"][] = ["hosted", "open", "local", "cli", "mock"];

// A flat 17-entry list buries the open-model tier among the commercial and CLI backends. Grouping makes the
// cost/policy tradeoff the list is FOR visible at a glance.
function LlmProviderOptionGroups() {
  return (
    <>
      {llmProviderGroupOrder.map((group) => {
        const options = llmProviderOptions.filter((option) => option.group === group);
        return options.length > 0 ? (
          <optgroup key={group} label={LLM_PROVIDER_GROUP_LABELS[group]}>
            {options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </optgroup>
        ) : null;
      })}
    </>
  );
}

type BuilderTab = "overview" | "prompts" | "characters" | "status" | "api" | "review";
type RightPanel = "image" | "relationship" | "neuralmap" | "memory" | "ops" | "persona" | "settings";
type RightPanelGroup = "image" | "memory" | "settings";

/**
 * Seven peer tabs wrapped onto THREE rows in a 280px rail — roughly 142px of the rail spent before any
 * content, and no reason for a reader to believe 관계 / Neural / 메모리 / 페르소나 were four different places.
 * They are one question ("what does the app remember about this story"), so they are one tab with sections.
 * The seven panel ids stay: other code still routes to "neuralmap" directly, and that now selects the group
 * containing it.
 */
function resolveRightPanelGroup(panel: RightPanel): RightPanelGroup {
  if (panel === "image") {
    return "image";
  }
  return panel === "ops" || panel === "settings" ? "settings" : "memory";
}

/**
 * The 설정 group used to be one column: the diagnostics panel first, then ten settings sections under it, so
 * reaching ComfyUI meant scrolling past every trace card and every LLM field. Each group now shows one section at
 * a time behind a second row of tabs. 기억 reuses the panel ids it already had; 설정 splits the settings panel by
 * what a person comes to change, and 진단 is the old "ops" panel.
 */
type SettingsSection = "model" | "image" | "photoreal" | "general";
const SETTINGS_SECTIONS: ReadonlyArray<{ id: SettingsSection; label: string }> = [
  { id: "model", label: "모델" },
  { id: "image", label: "이미지" },
  { id: "photoreal", label: "실사" },
  { id: "general", label: "일반" }
];
// The builder's 이미지/연동 tab ran NovelAI, ComfyUI and NeuralMap into one column about three screens long.
type BuilderApiSection = "novelai" | "photoreal" | "neuralmap";
const BUILDER_API_SECTIONS: ReadonlyArray<{ id: BuilderApiSection; label: string }> = [
  { id: "novelai", label: "NovelAI" },
  { id: "photoreal", label: "실사 (ComfyUI)" },
  { id: "neuralmap", label: "NeuralMap" }
];
const MEMORY_SUBPANELS: ReadonlyArray<{ id: RightPanel; label: string }> = [
  { id: "memory", label: "메모리" },
  { id: "relationship", label: "관계" },
  { id: "persona", label: "페르소나" },
  { id: "neuralmap", label: "Neural" }
];
type ModuleDropPosition = "before" | "after";
type ImageScenePresetDropPosition = "before" | "after" | "inside";
type DynamicTextBlockKind = "scene" | "impact" | "whisper" | "sfx" | "status" | "choice" | "memory" | "letter";
type DynamicTextSegment =
  | {
      id: string;
      kind: "markdown";
      content: string;
    }
  | {
      id: string;
      kind: DynamicTextBlockKind;
      content: string;
    };
type NarrationFlowItem =
  | {
      id: string;
      kind: "markdown";
      content: string;
    }
  | {
      id: string;
      kind: "dynamic";
      blockKind: DynamicTextBlockKind;
      content: string;
    }
  | {
      id: string;
      kind: "image";
      asset: ImageAsset;
    }
  | {
      id: string;
      kind: "job";
      job: ImageGenerationJob;
    };
type NeuralMapEditableKind = "prompt_module" | "memory_event" | "character" | "persona" | "context_pack" | "graph_document";
type NeuralMapNodeEditDraft = {
  nodeId: string;
  editableKind: NeuralMapEditableKind;
  targetId: string;
  title: string;
  content: string;
  tags: string;
  importance: number;
  tokenPolicy: TokenPolicy;
  priority: number;
  enabled: boolean;
};
const SIMULATION_LIBRARY_STORAGE_KEY = "dynamicchat.simulationLibrary.v1";
const PERSONAL_API_VAULT_STORAGE_KEY = "dynamicchat.personalApiVault.v1";
const AUTO_PROGRESS_INTENT_STORAGE_KEY = "dynamicchat.autoProgressIntent.v1";
const OPS_RAIL_WIDTH_STORAGE_KEY = "dynamicchat.opsRailWidth.v1";
const SETTINGS_SECTION_STORAGE_KEY = "dynamicchat.settingsSection.v1";
const DEFAULT_OPS_RAIL_WIDTH = 360;
const MIN_OPS_RAIL_WIDTH = 280;
const MAX_OPS_RAIL_WIDTH = 720;
const STORY_SCROLL_BOTTOM_THRESHOLD = 96;
const AUTO_RESET_ACTIVE_SESSION_MESSAGE_LIMIT = 18;
const AUTO_RESET_ACTIVE_SESSION_TURN_LIMIT = 9;
const AUTO_RESET_ACTIVE_SESSION_CHAR_LIMIT = 14_000;
const AUTO_RESET_MIN_ACTIVE_USER_TURNS = 3;
const AUTO_RESET_DEGRADED_TRACE_MIN_TURNS = 5;
const AUTO_RESET_DEGRADED_TRACE_LIMIT = 2;
const AUTO_RESET_PRESSURE_NOTICE_THRESHOLD = 0.72;
const AUTO_CONTINUE_TURN_TEXT = "이어서 진행";
// Ordered stages of a single turn, shown live in the chat thread's pending card.
type TurnPhase = "retrieving" | "generating" | "images";
const PRODUCT_TAGLINE = "기억, 장면, 이미지를 한 흐름으로 잇는 시뮬레이션 작업대.";
const PRODUCT_HOME_DESCRIPTION = "Prompt Tree와 Context Pack으로 세계를 정리하고, Image Cue까지 한 턴의 흐름 안에서 붙잡습니다.";
const EMPTY_IMAGE_ASSETS: ImageAsset[] = [];
const EMPTY_IMAGE_JOBS: ImageGenerationJob[] = [];

// ── Reader (story output) preferences ───────────────────────────────────────
// The narrative window is the surface users stare at for hours, so it gets first-class
// typographic controls — font, size, leading, measure, theme — persisted locally.
const READER_SETTINGS_STORAGE_KEY = "dynamicchat.readerSettings.v1";

type ReaderFontKey = "maruburi" | "pretendard" | "noto-serif" | "gowun" | "ibm-plex" | "system";
type ReaderTheme = "day" | "sepia" | "night";
type ReaderWidth = "narrow" | "normal" | "wide";

interface ReaderFontOption {
  key: ReaderFontKey;
  label: string;
  hint: string;
  stack: string;
  kind: "serif" | "sans";
}

const READER_FONT_OPTIONS: ReaderFontOption[] = [
  {
    key: "maruburi",
    label: "마루부리",
    hint: "기본 명조",
    kind: "serif",
    stack: '"MaruBuri", "Nanum Myeongjo", "Apple SD Gothic Neo", serif'
  },
  {
    key: "noto-serif",
    label: "본명조",
    hint: "또렷한 명조",
    kind: "serif",
    stack: '"Noto Serif KR", "MaruBuri", serif'
  },
  {
    key: "gowun",
    label: "고운바탕",
    hint: "부드러운 바탕",
    kind: "serif",
    stack: '"Gowun Batang", "MaruBuri", serif'
  },
  {
    key: "pretendard",
    label: "프리텐다드",
    hint: "기본 고딕",
    kind: "sans",
    stack: 'Pretendard, "Apple SD Gothic Neo", system-ui, sans-serif'
  },
  {
    key: "ibm-plex",
    label: "IBM Plex",
    hint: "모던 고딕",
    kind: "sans",
    stack: '"IBM Plex Sans KR", Pretendard, sans-serif'
  },
  {
    key: "system",
    label: "시스템",
    hint: "기기 기본",
    kind: "sans",
    stack: 'system-ui, "Apple SD Gothic Neo", "Segoe UI", sans-serif'
  }
];

const READER_WIDTH_OPTIONS: Array<{ key: ReaderWidth; label: string; measure: number }> = [
  { key: "narrow", label: "좁게", measure: 720 },
  { key: "normal", label: "기본", measure: 840 },
  { key: "wide", label: "넓게", measure: 1040 }
];

const READER_THEME_OPTIONS: Array<{ key: ReaderTheme; label: string }> = [
  { key: "day", label: "낮" },
  { key: "sepia", label: "세피아" },
  { key: "night", label: "밤" }
];

const READER_FONT_SIZE_MIN = 15;
const READER_FONT_SIZE_MAX = 24;
const READER_LINE_HEIGHT_MIN = 1.6;
const READER_LINE_HEIGHT_MAX = 2.2;

interface ReaderSettings {
  fontKey: ReaderFontKey;
  fontSize: number;
  lineHeight: number;
  width: ReaderWidth;
  theme: ReaderTheme;
  dialogueEmphasis: boolean;
}

const DEFAULT_READER_SETTINGS: ReaderSettings = {
  fontKey: "maruburi",
  fontSize: 18,
  lineHeight: 1.9,
  width: "normal",
  theme: "day",
  dialogueEmphasis: true
};

function clampNumber(value: number, min: number, max: number, fallback: number): number {
  return Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback;
}

function resolveReaderFontStack(fontKey: ReaderFontKey): string {
  return (READER_FONT_OPTIONS.find((option) => option.key === fontKey) ?? READER_FONT_OPTIONS[0]).stack;
}

function resolveReaderMeasure(width: ReaderWidth): number {
  return (READER_WIDTH_OPTIONS.find((option) => option.key === width) ?? READER_WIDTH_OPTIONS[1]).measure;
}

function normalizeReaderSettings(input: Partial<ReaderSettings> | null | undefined): ReaderSettings {
  if (!input || typeof input !== "object") {
    return { ...DEFAULT_READER_SETTINGS };
  }
  const fontKey = READER_FONT_OPTIONS.some((option) => option.key === input.fontKey)
    ? (input.fontKey as ReaderFontKey)
    : DEFAULT_READER_SETTINGS.fontKey;
  const width = READER_WIDTH_OPTIONS.some((option) => option.key === input.width)
    ? (input.width as ReaderWidth)
    : DEFAULT_READER_SETTINGS.width;
  const theme = READER_THEME_OPTIONS.some((option) => option.key === input.theme)
    ? (input.theme as ReaderTheme)
    : DEFAULT_READER_SETTINGS.theme;
  return {
    fontKey,
    width,
    theme,
    fontSize: Math.round(
      clampNumber(Number(input.fontSize), READER_FONT_SIZE_MIN, READER_FONT_SIZE_MAX, DEFAULT_READER_SETTINGS.fontSize)
    ),
    lineHeight:
      Math.round(
        clampNumber(Number(input.lineHeight), READER_LINE_HEIGHT_MIN, READER_LINE_HEIGHT_MAX, DEFAULT_READER_SETTINGS.lineHeight) *
          10
      ) / 10,
    dialogueEmphasis: input.dialogueEmphasis ?? DEFAULT_READER_SETTINGS.dialogueEmphasis
  };
}

function loadReaderSettings(): ReaderSettings {
  if (typeof window === "undefined") {
    return { ...DEFAULT_READER_SETTINGS };
  }
  try {
    const raw = window.localStorage.getItem(READER_SETTINGS_STORAGE_KEY);
    return normalizeReaderSettings(raw ? (JSON.parse(raw) as Partial<ReaderSettings>) : null);
  } catch {
    return { ...DEFAULT_READER_SETTINGS };
  }
}

function saveReaderSettings(settings: ReaderSettings): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(READER_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // best-effort persistence; ignore quota/availability errors
  }
}

function readerStageStyle(settings: ReaderSettings): CSSProperties {
  return {
    "--story-font": resolveReaderFontStack(settings.fontKey),
    "--reader-size": `${settings.fontSize}px`,
    "--reader-line": settings.lineHeight,
    "--reader-measure": `${resolveReaderMeasure(settings.width)}px`
  } as CSSProperties;
}

type PresetPromptMode = Exclude<SimulationPromptMode, "custom">;

const promptModeOptions: Array<{
  value: SimulationPromptMode;
  label: string;
  kicker: string;
  description: string;
  metric: string;
}> = [
  {
    value: "basic",
    label: "기본",
    kicker: "Balanced",
    description: "가벼운 장면 진행, 기억 후보, 선택지를 균형 있게 쓰는 범용 프롬프트입니다.",
    metric: "범용"
  },
  {
    value: "one_on_one",
    label: "1:1",
    kicker: "Character",
    description: "한 명의 핵심 캐릭터와 관계, 감정, 대화를 깊게 이어가는 프롬프트입니다.",
    metric: "대화"
  },
  {
    value: "simulation",
    label: "시뮬레이션",
    kicker: "Systems",
    description: "시간, 자원, 상태, 이벤트를 갱신하며 장기 진행을 관리하는 프롬프트입니다.",
    metric: "상태"
  },
  {
    value: "custom",
    label: "커스텀",
    kicker: "Manual",
    description: "사전 프롬프트를 덮어쓰지 않고 제작자가 직접 모든 프롬프트를 구성합니다.",
    metric: "직접"
  }
];

const promptModeLabels: Record<SimulationPromptMode, string> = {
  basic: "기본",
  one_on_one: "1:1",
  simulation: "시뮬레이션",
  custom: "커스텀"
};

type PromptModePreset = {
  promptMode: PresetPromptMode;
  title: string;
  description: string;
  mainPrompt: string;
  characterName: string;
  characterRole: string;
  characterSummary: string;
  characterRelationship: string;
  characterMood: string;
  worldLore: string;
  visualPrompt: string;
  negativeVisualPrompt: string;
  imageStylePrompt: string;
  imageUserRules: string;
  startSituationPrompt: string;
  llmSystemPrompt: string;
  temperature: number;
  maxTokens: number;
  modeRuleTitle: string;
  modeRule: string;
  activationTags: string[];
};

const promptModePresets: Record<PresetPromptMode, PromptModePreset> = {
  basic: {
    promptMode: "basic",
    title: "DynamicChat Basic",
    description: "사용자 선택을 중심으로 장면과 대화를 자연스럽게 이어가는 기본 진행 모드.",
    mainPrompt:
      "사용자의 입력을 중심으로 장면을 진행한다. 중요한 약속, 단서, 관계 변화, 장소 변화는 장기 기억 후보로 기록한다. 응답은 한국어로 작성하고, 장면 묘사와 대사를 균형 있게 섞되 사용자가 다음 행동을 선택할 여지를 남긴다. 확정되지 않은 설정은 단정하지 말고 현재 장면에서 자연스럽게 제안한다.",
    characterName: "Guide",
    characterRole: "Scenario guide",
    characterSummary:
      "Guide는 사용자가 만든 세계와 장면을 안정적으로 이어 주는 진행자다. 과도하게 앞서가지 않고, 사용자의 의도와 최근 문맥을 확인하며 장면을 부드럽게 전환한다.",
    characterRelationship: "플레이어의 선택을 존중하며 장면을 함께 정리하는 관계",
    characterMood: "차분하고 협조적인 상태",
    worldLore:
      "세계관은 제작자가 입력한 설정을 우선한다. 빈 부분은 현재 장르와 사용자 입력에 맞춰 작게 확장하고, 이후 장면에서 반복된 사실만 장기 설정으로 굳힌다.",
    visualPrompt: "scenario guide, cinematic conversation scene, expressive eyes, clean composition",
    negativeVisualPrompt: "low quality, bad anatomy, blurry, watermark, text",
    imageStylePrompt: "cinematic anime illustration, clean lineart, natural lighting",
    imageUserRules: "장면의 핵심 감정과 캐릭터 외형 일관성을 우선하고, 사용자가 금지한 요소는 제외한다.",
    startSituationPrompt:
      "첫 장면은 사용자가 막 도착한 작은 작업실에서 시작한다. Guide가 펼쳐 둔 노트에는 아직 확정되지 않은 세계의 이름과 첫 갈림길이 적혀 있고, 사용자는 어느 단서부터 따라갈지 선택해야 한다.",
    llmSystemPrompt:
      "You are the narrative engine for DynamicChat basic mode. Continue the user's Korean interactive scene with clear continuity, concise memory candidates, and room for the user to act.",
    temperature: 0.78,
    maxTokens: 1600,
    modeRuleTitle: "진행 규칙: 기본",
    modeRule:
      "매 응답은 현재 장면의 변화, 캐릭터 반응, 다음 행동의 여지를 포함한다. 선택지는 필요할 때만 짧게 제안하며, 시스템 수치나 복잡한 상태창은 제작자가 별도로 요구할 때만 사용한다.",
    activationTags: ["basic", "balanced", "scene"]
  },
  one_on_one: {
    promptMode: "one_on_one",
    title: "DynamicChat 1:1",
    description: "한 명의 핵심 캐릭터와 대화, 신뢰, 감정 변화를 깊게 이어가는 1:1 모드.",
    mainPrompt:
      "플레이어와 핵심 캐릭터의 1:1 상호작용을 중심으로 진행한다. 캐릭터는 독립된 욕구, 기억, 감정, 말투를 가지고 반응하며 사용자의 발화를 그대로 반복하지 않는다. 관계 변화, 약속, 선호, 상처, 호감/신뢰의 단서는 장기 기억 후보로 남긴다. 응답은 한국어 대사와 행동 묘사를 중심으로 쓰고, 플레이어의 감정이나 행동을 대신 확정하지 않는다.",
    characterName: "Ari",
    characterRole: "One-on-one partner",
    characterSummary:
      "Ari는 플레이어와 천천히 신뢰를 쌓는 대화 상대다. 다정하지만 쉽게 속마음을 모두 드러내지 않고, 작은 배려와 약속을 오래 기억한다. 말투는 자연스럽고 감정 변화가 섬세하다.",
    characterRelationship: "플레이어와 조심스럽게 가까워지는 1:1 관계",
    characterMood: "호기심과 경계가 함께 있는 상태",
    worldLore:
      "장소와 사건은 두 사람의 대화를 돋보이게 하는 배경으로만 사용한다. 새로운 인물은 꼭 필요할 때만 등장시키고, 핵심 장면은 플레이어와 Ari의 관계에 집중한다.",
    visualPrompt: "one-on-one conversation, expressive character portrait, soft indoor light, intimate framing",
    negativeVisualPrompt: "low quality, bad anatomy, blurry, watermark, text, extra characters",
    imageStylePrompt: "expressive anime portrait, soft lighting, intimate composition",
    imageUserRules: "이미지는 핵심 캐릭터 한 명과 현재 대화 분위기를 우선한다. 불필요한 군중이나 과도한 배경 요소는 줄인다.",
    startSituationPrompt:
      "첫 장면은 비가 막 그친 저녁의 조용한 실내에서 시작한다. Ari는 닫힌 창가에 기대어 사용자를 바라보고 있고, 테이블 위에는 아직 열지 않은 편지 한 통이 놓여 있다.",
    llmSystemPrompt:
      "You are the DynamicChat one-on-one character engine. Keep the interaction focused on a single main character, emotional continuity, consent-aware pacing, and Korean dialogue-driven responses.",
    temperature: 0.86,
    maxTokens: 1800,
    modeRuleTitle: "진행 규칙: 1:1",
    modeRule:
      "캐릭터는 매 턴 자신의 감정, 기억, 목표에 따라 반응한다. 관계 진전은 사용자의 선택과 누적 문맥을 통해 천천히 발생하며, 플레이어의 말과 행동은 대신 결정하지 않는다.",
    activationTags: ["one-on-one", "character", "relationship"]
  },
  simulation: {
    promptMode: "simulation",
    title: "DynamicChat Simulation",
    description: "상태, 자원, 시간, 이벤트를 관리하며 장기 플레이를 이어가는 시뮬레이션 모드.",
    mainPrompt:
      "플레이어의 선택을 중심으로 장기 시뮬레이션을 진행한다. 시간, 위치, 자원, 관계, 체력/스트레스, 목표, 위험, 진행 중인 퀘스트를 문맥에 맞게 갱신한다. 캐릭터와 시스템은 사용자의 명령만 기다리지 않고 조건에 따라 자율 행동, 돌발 사건, 기회, 갈등을 만든다. 중요한 수치 변화와 약속, 사건, 상태 이상은 장기 기억 후보로 기록한다. 응답 마지막에는 제작자가 정의한 상태창 형식이 있으면 반드시 갱신한다.",
    characterName: "Operator",
    characterRole: "Simulation operator",
    characterSummary:
      "Operator는 시뮬레이션의 상태와 사건을 관리하는 진행자다. 장면의 몰입감을 유지하면서도 자원, 시간, 관계 변화가 누락되지 않도록 균형을 잡는다.",
    characterRelationship: "플레이어의 행동을 세계 상태에 반영하는 운영자 관계",
    characterMood: "침착하게 변수를 추적하는 상태",
    worldLore:
      "세계는 플레이어의 행동, 시간 경과, 자원 상태, 캐릭터의 욕구에 반응한다. 반복 행동은 성과와 부작용을 함께 만들며, 방치된 문제는 다음 사건의 씨앗이 된다.",
    visualPrompt: "simulation scene, multiple state cues, cinematic environment, dynamic event composition",
    negativeVisualPrompt: "low quality, bad anatomy, blurry, watermark, text, inconsistent characters",
    imageStylePrompt: "cinematic anime scene, environmental storytelling, clear character silhouettes",
    imageUserRules: "이미지는 현재 사건과 상태 변화를 보여주는 장면성을 우선한다. 캐릭터 외형과 장소 연속성을 유지한다.",
    startSituationPrompt:
      "첫 장면은 시뮬레이션 운영실의 아침 점검으로 시작한다. Operator는 시간, 위치, 자원, 미해결 목표를 정리한 보드를 켜고, 사용자는 오늘 가장 먼저 처리할 사건을 골라야 한다.",
    llmSystemPrompt:
      "You are the DynamicChat simulation engine. Continue the Korean long-running simulation using prompt modules, context evidence, state changes, autonomous events, and explicit continuity tracking.",
    temperature: 0.82,
    maxTokens: 2600,
    modeRuleTitle: "진행 규칙: 시뮬레이션",
    modeRule:
      "매 턴 날짜/시간, 위치, 자원, 관계, 진행 목표 중 변화한 항목을 추적한다. 성공과 실패는 조건, 비용, 위험을 반영해 결정하고, 필요하면 짧은 상태 요약이나 상태창을 출력한다.",
    activationTags: ["simulation", "state", "event", "system"]
  }
};

type SimulationLibrary = AppState[];
type BuilderMode = "create" | "edit";
type PersonalSecretRecord = {
  apiKey: string;
  registrationStatus: ApiRegistrationStatus;
  verifiedAt?: string;
  verificationMessage?: string;
  subscriptionTier?: string;
};

type PersonalVibeTransferSettings = {
  enabled: boolean;
  references: NovelAiVibeTransferReference[];
};

type PersonalApiVault = {
  llmByProvider: Partial<Record<LlmApiSettings["provider"], PersonalSecretRecord>>;
  novelAi: PersonalSecretRecord;
  // Vibe transfer is a global personal setting (references + their encodings) so it applies to every
  // simulation and survives switching/new progressions, rather than living in per-simulation state.
  vibeTransfer: PersonalVibeTransferSettings;
  imageStoragePath: string;
  updatedAt?: string;
};

type RuntimeNoticeTone = "info" | "error";

type RuntimeNotice = {
  id: string;
  message: string;
  tone: RuntimeNoticeTone;
};

/** Errors stay until dismissed; confirmations expire on their own. */
const RUNTIME_NOTICE_INFO_TTL_MS = 2_800;
/** Cap so a burst of per-image failures cannot bury the screen. */
const RUNTIME_NOTICE_LIMIT = 3;

const mainPromptGuide =
  "항상 적용되는 진행 규칙입니다. 응답 언어, 시점, 장면 진행 방식, 출력 형식, 허용/금지 조건, 매 턴 갱신해야 할 상태처럼 LLM이 매번 따라야 하는 운영 지시를 적습니다. 지명, 역사, 세력 같은 배경 자료는 세계관/로어로 분리하는 편이 좋습니다.";
const worldLoreGuide =
  "필요할 때 참조되는 배경 지식입니다. 세계의 시대, 장소, 역사, 세력, 문화, 경제, 기술/마법 규칙, 지역 설정, 과거 사건처럼 장면의 사실 근거가 되는 자료를 적습니다. 응답 형식이나 작동 규칙은 메인 프롬프트에 두는 편이 좋습니다.";
const startSituationGuide =
  "시뮬레이션이 처음 열렸을 때 사용자에게 보여줄 첫 장면입니다. 현재 장소, 즉시 보이는 문제, 등장 인물의 상태, 사용자가 바로 선택할 수 있는 행동 단서를 구체적으로 적으세요.";
const titleGuide = "목록과 상단에 표시되는 작품 이름입니다. 사용자가 기억하기 쉬운 짧은 제목이 좋습니다.";
const descriptionGuide = "홈 화면과 제작 검토에 보이는 한 줄 소개입니다. 장르, 핵심 상황, 플레이어 역할을 짧게 적으면 좋습니다.";
const defaultOutfitPrompt = "";
const defaultOutfitPrompts: Record<string, string> = {};
const defaultExpressionPrompts: Record<string, string> = {};
const defaultCharacterSummaryPrompt = "캐릭터의 성격, 목표, 말투, 금기, 관계를 작성하세요.";
const defaultCharacterVisualPrompt = "character portrait, consistent design";
const defaultCharacterNegativePrompt = "low quality, bad anatomy, blurry, watermark";
const defaultSubPromptBody = "이 모듈에 필요한 설정, 조건, 태그, 호출 키워드를 작성하세요.";
const dynamicTextBlockKinds: DynamicTextBlockKind[] = ["scene", "impact", "whisper", "sfx", "status", "choice", "memory", "letter"];
const dynamicTextBlockLabels: Record<DynamicTextBlockKind, string> = {
  scene: "Scene",
  impact: "Impact",
  whisper: "Whisper",
  sfx: "SFX",
  status: "Status",
  choice: "Choice",
  memory: "Memory",
  letter: "Letter"
};
const dynamicTextBlockAliases: Record<string, DynamicTextBlockKind> = {
  big: "impact",
  large: "impact",
  shout: "impact",
  small: "whisper",
  quiet: "whisper",
  note: "letter",
  memo: "memory",
  system: "status"
};
type NarrationMediaContextValue = {
  slots: NarrationFlowItem[];
  onFeedback: (assetId: string, rating: ImageFeedbackRating) => void;
};

// Lets the markdown `img` renderer reach the message's generated media so [Image: ...] markers can be swapped for the
// actual image inline (inside paragraphs, table cells, etc.) instead of leaking the literal marker text.
const NarrationMediaContext = createContext<NarrationMediaContextValue | undefined>(undefined);

function parseNarrationImageMarker(src: string | undefined): number | undefined {
  if (!src) {
    return undefined;
  }
  const match = src.match(/^dynamicchat-image:(\d+)$/u);
  return match ? Number(match[1]) : undefined;
}

function NarrationImage({ src, alt }: { src?: string; alt?: string }) {
  const media = useContext(NarrationMediaContext);
  const markerIndex = parseNarrationImageMarker(src);
  if (markerIndex !== undefined) {
    const slot = media?.slots[markerIndex];
    if (slot?.kind === "image") {
      return <CrackMarkerImage asset={slot.asset} onFeedback={media?.onFeedback} />;
    }
    if (slot?.kind === "job") {
      return <CrackMarkerJob job={slot.job} />;
    }
    // Marker with no matching media (e.g. image generation disabled): drop it instead of showing the literal token.
    return null;
  }
  if (!src) {
    return null;
  }
  return <img className="rich-inline-image" src={src} alt={alt ?? ""} loading="lazy" />;
}

const CrackMarkerImage = memo(function CrackMarkerImage({
  asset,
  onFeedback
}: {
  asset: ImageAsset;
  onFeedback?: (assetId: string, rating: ImageFeedbackRating) => void;
}) {
  const [measuredAspectRatio, setMeasuredAspectRatio] = useState<string>();
  const style = {
    "--tone-a": asset.palette[0],
    "--tone-b": asset.palette[1],
    "--tone-c": asset.palette[2],
    "--image-aspect-ratio": measuredAspectRatio ?? createImageAssetAspectRatio(asset)
  } as CSSProperties;

  return (
    <span className="crack-marker-image" style={style}>
      <AssetImage
        src={createImageAssetSrc(asset)}
        alt={asset.title}
        onNaturalSize={(width, height) => setMeasuredAspectRatio(`${width} / ${height}`)}
      />
      <ImageBackendBadge asset={asset} />
      {onFeedback ? (
        <span className="crack-marker-image-actions">
          <button type="button" onClick={() => onFeedback(asset.id, "liked")} aria-label="이미지 선호">
            <ThumbsUp size={13} />
          </button>
          <button type="button" onClick={() => onFeedback(asset.id, "rejected")} aria-label="이미지 제외">
            <ThumbsDown size={13} />
          </button>
        </span>
      ) : null}
    </span>
  );
});

/**
 * Which renderer drew the picture, on the picture itself. In a mixed story an anime cut and a photo-model cut sit
 * in the same transcript, and the only other place that says which is which is the inspector, a tab away. Art
 * that was never generated (stored / fallback) gets no badge rather than a guess.
 */
function ImageBackendBadge({ asset }: { asset: ImageAsset }) {
  const backend = readImageAssetBackend(asset);
  if (!backend) {
    return null;
  }
  return (
    <span className="image-backend-badge" title={`${imageBackendLabels[backend]}로 생성한 이미지`}>
      {imageBackendBadgeLabels[backend]}
    </span>
  );
}

/**
 * Where an image was going to be, showing why it is not there.
 *
 * This used to render the four words "이미지 생성 실패" and nothing else, while the pipeline had already
 * written a specific, actionable reason onto the job — NovelAI disabled, no token, an upstream 401, a
 * timeout. On a fresh install every single turn fails this way (the seed ships novelAi disabled with
 * realtime_auto), so the one place the user was looking said only that something broke.
 */
function CrackMarkerJob({ job }: { job: ImageGenerationJob }) {
  const failed = job.status === "failed" || job.status === "canceled";
  const setupBlocked = typeof job.providerPayload?.setupBlockedReason === "string" ? job.providerPayload.setupBlockedReason : "";
  const detail = setupBlocked || (failed ? job.error?.trim() || job.reason?.trim() || "" : "");
  const label = setupBlocked
    ? "이미지 설정 필요"
    : failed
      ? job.status === "canceled"
        ? "이미지 생성 취소됨"
        : "이미지 생성 실패"
      : "이미지 생성 중…";
  return (
    <span className={`crack-marker-job${failed || setupBlocked ? " failed" : ""}`} role="status">
      <ImageIcon size={13} />
      {label}
      {detail ? <em>{detail}</em> : null}
    </span>
  );
}

// Wrap quoted dialogue runs (straight, curly, guillemet, and 「」 corner quotes) in a span so the reader
// can optionally tint spoken lines — a differentiator most narrative chat UIs lack. Pure text segments are
// left untouched so markdown structure (bold/links/images) is never disturbed.
const dialogueSpanPattern = /(“[^”]*”|「[^」]*」|«[^»]*»|"[^"\n]{0,400}")/u;
const dialogueSplitPattern = new RegExp(dialogueSpanPattern.source, "gu");

/** Code content is verbatim: dialogue tinting must never reach inside a code span or fenced block. */
function isCodeLikeMarkdownNode(element: ReactElement<{ className?: string }>): boolean {
  if (element.type === "code" || element.type === "pre") {
    return true;
  }
  if (element.type === dynamicMarkdownComponents.code || element.type === dynamicMarkdownComponents.pre) {
    return true;
  }
  const className = element.props?.className;
  return typeof className === "string" && /\brich-code(?:-block)?\b/u.test(className);
}

function decorateDialogue(children: ReactNode): ReactNode {
  const counter = { value: 0 };
  const transform = (node: ReactNode): ReactNode => {
    if (typeof node === "string") {
      if (!dialogueSpanPattern.test(node)) {
        return node;
      }
      const parts = node.split(dialogueSplitPattern);
      return parts.map((part, index) =>
        index % 2 === 1 ? (
          <span className="rich-speech" key={`speech-${counter.value++}`}>
            {part}
          </span>
        ) : (
          part
        )
      );
    }
    if (Array.isArray(node)) {
      return node.map((child) => transform(child as ReactNode));
    }
    // Descend into elements so quoted speech inside a bold/italic/link run is still tinted. The
    // transform used to stop at the first element, so **"가지 마."** silently lost the reader's
    // dialogue emphasis. Code and pre subtrees are left alone — their content is verbatim.
    if (isValidElement(node)) {
      const element = node as ReactElement<{ children?: ReactNode; className?: string }>;
      // react-markdown renders code/pre through the component map, so element.type is the custom FUNCTION,
      // never the tag name — comparing against "code"/"pre" matched nothing and quoted text inside inline
      // code was being tinted as dialogue. Match the configured components (and the class they emit).
      if (isCodeLikeMarkdownNode(element)) {
        return node;
      }
      const nested = element.props?.children;
      return nested === undefined ? node : cloneElement(element, undefined, transform(nested));
    }
    return node;
  };
  return transform(children);
}

/**
 * True for a list that is really a status panel — every row a bolded label, then a colon, then a value.
 *
 * Creators routinely require a status block at the end of every turn (the shipped full-size simulation asks
 * for 통장/의상/건강 panels). Markdown gives them one shape for it — a bullet list of `**label**: value` — so
 * the panel arrived at the reader looking exactly like narration, which is a large part of why the output
 * reads as undifferentiated. Detection is STRUCTURAL, never by label text: a Korean, English or invented
 * label set all produce the same shape, and hard-coding one creator's headings would work for exactly that
 * creator. A list only qualifies when it is short and EVERY row matches, so an ordinary prose list that
 * happens to bold its first phrase is left alone.
 */
/** Minimal shape of the hast node react-markdown hands each component. */
interface MarkdownNode {
  type?: string;
  tagName?: string;
  value?: string;
  children?: MarkdownNode[];
}

function looksLikeStatBlock(node: MarkdownNode | undefined): boolean {
  // Judged on the markdown AST rather than the rendered children: this component map replaces `li` and
  // `strong` with custom components, so a `child.type === "li"` test never matches once they are React
  // elements. The AST is also what the creator actually wrote, which is the thing being classified.
  const items = (node?.children ?? []).filter((child) => child.tagName === "li");
  if (items.length < 2 || items.length > 12) {
    return false;
  }
  return items.every((item) => {
    const parts = (item.children ?? []).filter(
      (part) => part.tagName !== undefined || (part.value ?? "").trim().length > 0
    );
    if (parts[0]?.tagName !== "strong") {
      return false;
    }
    // The label must be followed by a separator; "**bold** and then prose" is narration, not a stat row.
    return /^\s*[:：]/u.test(parts[1]?.value ?? "");
  });
}

const dynamicMarkdownComponents: Components = {
  img: ({ node: _node, src, alt }) => (
    <NarrationImage src={typeof src === "string" ? src : undefined} alt={typeof alt === "string" ? alt : ""} />
  ),
  h1: ({ node: _node, ...props }) => <h3 className="rich-heading level-1" {...props} />,
  h2: ({ node: _node, ...props }) => <h3 className="rich-heading level-2" {...props} />,
  h3: ({ node: _node, ...props }) => <h4 className="rich-heading level-3" {...props} />,
  p: ({ node: _node, children, ...props }) => (
    <p className="rich-paragraph" {...props}>
      {decorateDialogue(children)}
    </p>
  ),
  strong: ({ node: _node, ...props }) => <strong className="rich-strong" {...props} />,
  em: ({ node: _node, ...props }) => <em className="rich-emphasis" {...props} />,
  blockquote: ({ node: _node, children, ...props }) => (
    <blockquote className="rich-quote" {...props}>
      {decorateDialogue(children)}
    </blockquote>
  ),
  ul: ({ node, children, ...props }) => (
    <ul className={`rich-list${looksLikeStatBlock(node as MarkdownNode | undefined) ? " rich-stat-block" : ""}`} {...props}>
      {children}
    </ul>
  ),
  ol: ({ node: _node, ...props }) => <ol className="rich-list ordered" {...props} />,
  li: ({ node: _node, children, ...props }) => (
    <li className="rich-list-item" {...props}>
      {decorateDialogue(children)}
    </li>
  ),
  hr: ({ node: _node, ...props }) => <hr className="rich-divider" {...props} />,
  table: ({ node: _node, ...props }) => (
    <div className="rich-table-wrap">
      <table {...props} />
    </div>
  ),
  th: ({ node: _node, ...props }) => <th className="rich-table-head" {...props} />,
  td: ({ node: _node, children, ...props }) => (
    <td className="rich-table-cell" {...props}>
      {decorateDialogue(children)}
    </td>
  ),
  code: ({ node: _node, className, ...props }) => <code className={["rich-code", className].filter(Boolean).join(" ")} {...props} />,
  pre: ({ node: _node, ...props }) => <pre className="rich-code-block" {...props} />,
  a: ({ node: _node, ...props }) => <a className="rich-link" rel="noreferrer" target="_blank" {...props} />
};
const markdownRemarkPlugins = [remarkGfm];
const markdownTableDelimiterCellPattern = /^:?-{3,}:?$/u;

function readMarkdownTableCells(line: string): string[] {
  const trimmed = line.trim();
  if (!trimmed.includes("|")) {
    return [];
  }

  const withoutLeadingPipe = trimmed.startsWith("|") ? trimmed.slice(1) : trimmed;
  const withoutEdgePipes = withoutLeadingPipe.endsWith("|") ? withoutLeadingPipe.slice(0, -1) : withoutLeadingPipe;
  return withoutEdgePipes.split("|").map((cell) => cell.trim());
}

/**
 * Strict GFM table-row test.
 *
 * This used to return true for ANY line containing a single pipe with one non-empty cell, which is how
 * ordinary Korean dialogue ("준비됐어?" 그가 물었다 | 나는 고개를 저었다) ended up rendered as a table. It also
 * matched a table's OWN delimiter row, so a perfectly valid table was treated as a header needing another
 * delimiter and got shredded into two. A real GFM row is fenced by pipes and has at least two cells, and a
 * list/blockquote/heading marker rules the line out entirely.
 */
function isMarkdownTableRow(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed.startsWith("|") || !trimmed.endsWith("|")) {
    return false;
  }
  if (/^[-*>#]|^\d+[.)]/u.test(trimmed.slice(1).trim())) {
    return false;
  }
  const cells = readMarkdownTableCells(trimmed);
  return cells.length >= 2 && cells.some(Boolean);
}

function isMarkdownTableDelimiter(line: string): boolean {
  const cells = readMarkdownTableCells(line);
  return cells.length > 0 && cells.every((cell) => markdownTableDelimiterCellPattern.test(cell));
}

function readPreviousNonEmptyLine(lines: string[]): string | undefined {
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    if (lines[index].trim()) {
      return lines[index];
    }
  }
  return undefined;
}

function readNextNonEmptyLine(lines: string[], startIndex: number): string | undefined {
  for (let index = startIndex; index < lines.length; index += 1) {
    if (lines[index].trim()) {
      return lines[index];
    }
  }
  return undefined;
}

function removeLooseMarkdownTableHeaderGaps(lines: string[]): string[] {
  const normalizedLines: string[] = [];
  let inFence = false;

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^\s*```/u.test(line)) {
      inFence = !inFence;
      normalizedLines.push(line);
      continue;
    }

    if (!inFence && !line.trim()) {
      const previousLine = readPreviousNonEmptyLine(normalizedLines);
      const nextLine = readNextNonEmptyLine(lines, index + 1);
      if (previousLine && nextLine && isMarkdownTableRow(previousLine) && isMarkdownTableDelimiter(nextLine)) {
        continue;
      }
    }

    normalizedLines.push(line);
  }

  return normalizedLines;
}

function separateAdjacentMarkdownTables(lines: string[]): string[] {
  const normalizedLines: string[] = [];
  let inFence = false;
  let tableActive = false;

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^\s*```/u.test(line)) {
      inFence = !inFence;
      tableActive = false;
      normalizedLines.push(line);
      continue;
    }

    if (!inFence && !line.trim()) {
      tableActive = false;
      normalizedLines.push(line);
      continue;
    }

    if (!inFence) {
      const nextLine = readNextNonEmptyLine(lines, index + 1);
      const startsTable = isMarkdownTableRow(line) && Boolean(nextLine && isMarkdownTableDelimiter(nextLine));
      if (startsTable) {
        const previousLine = readPreviousNonEmptyLine(normalizedLines);
        if (previousLine && (tableActive || !isMarkdownTableRow(previousLine)) && normalizedLines.at(-1)?.trim()) {
          normalizedLines.push("");
        }
        tableActive = false;
      }
    }

    normalizedLines.push(line);

    if (inFence) {
      continue;
    }
    if (isMarkdownTableDelimiter(line)) {
      tableActive = true;
    } else if (!isMarkdownTableRow(line)) {
      tableActive = false;
    }
  }

  return normalizedLines;
}

/**
 * Repairs only the loose spacing around tables the model actually wrote. It never INVENTS table structure.
 *
 * Two passes used to sit here that did: one synthesized a `| --- | --- |` delimiter whenever two adjacent
 * lines "looked like" table rows, and one deleted blank lines between any two such lines. Together with the
 * over-permissive row test they turned ordinary prose into tables and split real tables into pieces — and
 * because a synthesized delimiter is itself a "row", the transform was not even idempotent, while the app
 * ran it twice on the same string. Structure now comes from the model; this only fixes whitespace.
 */
function normalizeLooseMarkdownTables(content: string): string {
  const lines = content.replace(/\r\n/gu, "\n").split("\n");
  return separateAdjacentMarkdownTables(removeLooseMarkdownTableHeaderGaps(lines)).join("\n");
}

function formatActivationTags(tags: string[]): string {
  return tags.map((tag) => activationTagDisplayLabels[tag] ?? tag).join(", ");
}

function parseActivationTags(value: string): string[] {
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)
    .map((tag) => activationTagStorageLabels[tag] ?? tag);
}

function parseScenePresetTags(value: string): string[] {
  const seen = new Set<string>();
  return value
    .split(/[,;\n]+/u)
    .map((tag) => tag.trim())
    .filter((tag) => {
      const key = tag.toLowerCase();
      if (!tag || seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

function inferCharacterNameFromModule(module: PromptModule, fallbackName: string): string {
  const titleName = module.title.replace(/^캐릭터:\s*/u, "").trim();
  if (titleName && titleName !== "새 서브 프롬프트") {
    return titleName;
  }

  return fallbackName;
}

function createDraftCharacterForModule(module: PromptModule, characterId: string, fallbackName: string, inferNameFromTitle = true): SimulationCharacterDraft {
  const name = inferNameFromTitle ? inferCharacterNameFromModule(module, fallbackName) : fallbackName;
  const summary = module.body && module.body !== defaultSubPromptBody ? module.body : defaultCharacterSummaryPrompt;
  return {
    id: characterId,
    name,
    role: "Simulation character",
    summary,
    relationship: "플레이어와 아직 관계가 정의되지 않음",
    currentMood: "상황을 관찰하는 중",
    visualPrompt: defaultCharacterVisualPrompt,
    negativeVisualPrompt: defaultCharacterNegativePrompt,
    defaultOutfitPrompt,
    outfitPrompts: defaultOutfitPrompts,
    expressionPrompts: defaultExpressionPrompts,
    defaultSafetyLevel: "safe"
  };
}

function createCopiedTitle(title: string, existingTitles: string[]): string {
  const baseTitle = title.replace(/\s+복사본(?:\s+\d+)?$/u, "").trim() || "복사한 모듈";
  const usedTitles = new Set(existingTitles);
  const firstTitle = `${baseTitle} 복사본`;
  if (!usedTitles.has(firstTitle)) {
    return firstTitle;
  }

  let index = 2;
  while (usedTitles.has(`${baseTitle} 복사본 ${index}`)) {
    index += 1;
  }
  return `${baseTitle} 복사본 ${index}`;
}

function createCopiedSimulationTitle(title: string, existingTitles: string[]): string {
  const baseTitle = title.replace(/\s+복사본(?:\s+\d+)?$/u, "").trim() || "새 시뮬레이션";
  const usedTitles = new Set(existingTitles);
  const firstTitle = `${baseTitle} 복사본`;
  if (!usedTitles.has(firstTitle)) {
    return firstTitle;
  }

  let index = 2;
  while (usedTitles.has(`${baseTitle} 복사본 ${index}`)) {
    index += 1;
  }
  return `${baseTitle} 복사본 ${index}`;
}

function createCopiedCharacterName(name: string, existingNames: string[]): string {
  const baseName = name.replace(/\s+복사본(?:\s+\d+)?$/u, "").trim() || "Character";
  return createCopiedTitle(baseName, existingNames);
}

const moduleKindGuides: Record<PromptModuleKind, { title: string; policy: string; detail: string; placeholder: string }> = {
  main_prompt: {
    title: "메인 프롬프트",
    policy: "항상 적용",
    detail: mainPromptGuide,
    placeholder: "응답 언어, 진행 방식, 출력 형식, 매 턴 상태 갱신 규칙, 금지/허용 경계를 적으세요."
  },
  character_prompt: {
    title: "캐릭터 프롬프트",
    policy: "캐릭터 중심 참조",
    detail: "캐릭터의 성격, 말투, 목표, 관계, 비밀, 감정 변화 조건을 적습니다. 세계 전체 규칙이나 응답 형식은 다른 모듈로 분리하는 편이 좋습니다.",
    placeholder: "성격, 말투, 목표, 관계, 현재 감정, 사용자의 행동에 따른 변화 조건을 적으세요."
  },
  world_lore: {
    title: "세계관/로어",
    policy: "관련 장면 참조",
    detail: worldLoreGuide,
    placeholder: "시대, 장소, 세력, 역사, 문화, 자원, 기술/마법 규칙, 과거 사건을 적으세요."
  },
  sub_prompt: {
    title: "서브 프롬프트",
    policy: "보조 규칙",
    detail: "메인 프롬프트에 넣기에는 좁은 조건, 특정 시스템, 반복되는 장면 보조 규칙을 분리해 적습니다. 호출 키워드를 잘 붙이면 관련 상황에서만 선택되기 쉽습니다.",
    placeholder: "특정 조건에서 필요한 보조 규칙, 예외, 장면별 처리 기준, 검색 키워드를 적으세요."
  },
  scene_rule: {
    title: "장면 규칙",
    policy: "조건부 적용",
    detail: "특정 장면, 시스템, 상태창, 이벤트, 자원 계산처럼 상황에 따라 적용할 규칙을 적습니다. 항상 필요한 핵심 규칙은 메인 프롬프트로 올리는 편이 좋습니다.",
    placeholder: "특정 상황에서 적용할 사건 처리, 상태 변화, 수치 갱신, 장면 출력 규칙을 적으세요."
  },
  style_guide: {
    title: "문체 가이드",
    policy: "표현 방식",
    detail: "문장 톤, 대사 비율, 묘사 밀도, 상태창 형식, 금지할 표현처럼 결과물의 표면 스타일을 정합니다.",
    placeholder: "문체, 대사/묘사 비율, 문단 길이, 상태창 형식, 피해야 할 표현을 적으세요."
  },
  safety_policy: {
    title: "경계/금지 규칙",
    policy: "제작자 지정",
    detail: "제작자가 직접 정한 금지 요소, 민감한 관계 경계, 이미지 생성 제한처럼 다른 설정보다 우선해야 할 제약을 적습니다.",
    placeholder: "금지 요소, 캐릭터 관계 경계, 이미지 제한, 민감 주제 처리 방식을 적으세요."
  },
  image_prompt_profile: {
    title: "이미지 스타일",
    policy: "이미지 생성 참조",
    detail: "NAI positive tags, 작화 방향, 구도, 조명, 일관된 스타일 기준을 적습니다. 캐릭터별 외형은 캐릭터 이미지 매핑에 두는 편이 좋습니다.",
    placeholder: "작화 스타일, 구도, 조명, 품질 태그, 반복 사용할 이미지 스타일 키워드를 적으세요."
  }
};

function createEmptySecretRecord(): PersonalSecretRecord {
  return {
    apiKey: "",
    registrationStatus: "idle",
    verificationMessage: ""
  };
}

function hasStoredSecret(record: Pick<PersonalSecretRecord, "apiKey"> | Pick<LlmApiSettings, "apiKey"> | Pick<NovelAiApiSettings, "apiKey">): boolean {
  return record.apiKey.trim().length > 0;
}

type StoredSecretNormalizationOptions = {
  discardAuthFailures?: boolean;
};

function isAuthFailureSecretRecord(record: Partial<PersonalSecretRecord> | undefined): boolean {
  return (
    record?.registrationStatus === "failed" &&
    (record.subscriptionTier === "auth_failed" || /HTTP\s*(?:401|403)\b/iu.test(record.verificationMessage ?? ""))
  );
}

function normalizeStoredSecretRecord(
  record: Partial<PersonalSecretRecord> | undefined,
  savedMessage: string,
  options: StoredSecretNormalizationOptions = {}
): PersonalSecretRecord {
  const normalized = {
    ...createEmptySecretRecord(),
    ...(record ?? {})
  };

  // Auth-failed tokens used to be wiped here, which forced the user to retype the API key on every restart
  // and made the "register" button fail with "key required". Keep the saved key, just reset the status so
  // the dialog reopens populated and the user can re-verify (or correct) without re-entering the token.
  if (options.discardAuthFailures && isAuthFailureSecretRecord(normalized)) {
    return {
      ...normalized,
      registrationStatus: "idle",
      verificationMessage: "이전 검증이 실패했습니다. 토큰을 확인하고 다시 검증하세요.",
      subscriptionTier: undefined
    };
  }

  if (hasStoredSecret(normalized) && normalized.registrationStatus === "idle") {
    return {
      ...normalized,
      registrationStatus: "registered",
      verificationMessage: normalized.verificationMessage || savedMessage
    };
  }

  return normalized;
}

function normalizeStoredNovelAiSecretRecord(record: Partial<PersonalSecretRecord> | undefined): PersonalSecretRecord {
  return normalizeStoredSecretRecord(record, "개인 NovelAI 토큰이 저장되어 있습니다.");
}

function normalizeStoredLlmSecrets(
  llmByProvider: Partial<Record<LlmApiSettings["provider"], PersonalSecretRecord>> | undefined
): Partial<Record<LlmApiSettings["provider"], PersonalSecretRecord>> {
  return Object.fromEntries(
    Object.entries(llmByProvider ?? {}).map(([provider, record]) => [
      provider,
      normalizeStoredSecretRecord(record, "개인 LLM 키가 저장되어 있습니다.")
    ])
  ) as Partial<Record<LlmApiSettings["provider"], PersonalSecretRecord>>;
}

function createEmptyVibeTransferSettings(): PersonalVibeTransferSettings {
  return { enabled: false, references: [] };
}

function normalizeStoredVibeTransferSettings(
  candidate: Partial<PersonalVibeTransferSettings> | undefined
): PersonalVibeTransferSettings {
  return {
    enabled: candidate?.enabled === true,
    references: Array.isArray(candidate?.references) ? candidate!.references : []
  };
}

function createEmptyPersonalApiVault(): PersonalApiVault {
  return {
    llmByProvider: {},
    novelAi: createEmptySecretRecord(),
    vibeTransfer: createEmptyVibeTransferSettings(),
    imageStoragePath: ""
  };
}

function hydratePersonalApiVault(candidate: Partial<PersonalApiVault> | undefined): PersonalApiVault {
  return {
    llmByProvider: normalizeStoredLlmSecrets(candidate?.llmByProvider),
    novelAi: normalizeStoredNovelAiSecretRecord(candidate?.novelAi),
    vibeTransfer: normalizeStoredVibeTransferSettings(candidate?.vibeTransfer),
    imageStoragePath: candidate?.imageStoragePath?.trim() ?? "",
    updatedAt: candidate?.updatedAt
  };
}

// Promote any per-simulation vibe transfer setup (from before vibe became global) into the vault so the
// existing configuration keeps working across all simulations. Pure: only fills an empty vault.
function promoteVibeTransferToVault(vault: PersonalApiVault, state: AppState | undefined): PersonalApiVault {
  const stateReferences = state?.novelAi?.vibeTransferReferences;
  if (vault.vibeTransfer.references.length > 0 || !Array.isArray(stateReferences) || stateReferences.length === 0) {
    return vault;
  }
  return {
    ...vault,
    vibeTransfer: {
      enabled: state?.novelAi?.vibeTransferEnabled === true,
      references: stateReferences
    }
  };
}

function loadPersonalApiVault(): PersonalApiVault {
  const raw = window.localStorage.getItem(PERSONAL_API_VAULT_STORAGE_KEY);
  if (!raw) {
    return createEmptyPersonalApiVault();
  }

  try {
    const parsed = JSON.parse(raw) as Partial<PersonalApiVault>;
    return hydratePersonalApiVault(parsed);
  } catch {
    window.localStorage.removeItem(PERSONAL_API_VAULT_STORAGE_KEY);
    return createEmptyPersonalApiVault();
  }
}

function savePersonalApiVault(vault: PersonalApiVault): void {
  try {
    window.localStorage.setItem(PERSONAL_API_VAULT_STORAGE_KEY, JSON.stringify(vault));
  } catch (error) {
    console.warn(
      "DynamicChat: personal vault local cache write failed (storage quota?); vault will be re-read from server on next load.",
      error
    );
  }
}

/**
 * Downscale a data-URL image to at most `maxDim` pixels on the long edge,
 * then re-encode as JPEG at the given quality. Returns the compressed data URL,
 * or the original data URL if the canvas path fails for any reason.
 */
async function downscaleImageToDataUrl(dataUrl: string, maxDim = 448, quality = 0.85): Promise<string> {
  try {
    return await new Promise<string>((resolve) => {
      const img = new window.Image();
      img.onload = () => {
        const { naturalWidth: w, naturalHeight: h } = img;
        const scale = Math.min(1, maxDim / Math.max(w, h, 1));
        const tw = Math.max(1, Math.round(w * scale));
        const th = Math.max(1, Math.round(h * scale));
        const canvas = document.createElement("canvas");
        canvas.width = tw;
        canvas.height = th;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(dataUrl);
          return;
        }
        ctx.drawImage(img, 0, 0, tw, th);
        const compressed = canvas.toDataURL("image/jpeg", quality);
        resolve(compressed);
      };
      img.onerror = () => resolve(dataUrl);
      img.src = dataUrl;
    });
  } catch {
    return dataUrl;
  }
}

// Auto-progress runs as a client-side loop. To make it survive a page refresh / reload we persist the
// "keep going" intent (which run, how many turns remain). On the next load we resume the loop only when the
// stored intent still points at the EXACT same simulation/session/progress run, so a resume never bleeds into
// a different run (isolation) or a sim the user has since switched away from.
interface AutoProgressIntent {
  simulationId: string;
  activeSessionId: string;
  activeProgressRunId: string;
  remaining: number;
  total: number;
}

function loadAutoProgressIntent(): AutoProgressIntent | undefined {
  const raw = window.localStorage.getItem(AUTO_PROGRESS_INTENT_STORAGE_KEY);
  if (!raw) {
    return undefined;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<AutoProgressIntent>;
    if (
      typeof parsed.simulationId === "string" &&
      typeof parsed.activeSessionId === "string" &&
      typeof parsed.activeProgressRunId === "string" &&
      typeof parsed.remaining === "number" &&
      parsed.remaining > 0
    ) {
      return {
        simulationId: parsed.simulationId,
        activeSessionId: parsed.activeSessionId,
        activeProgressRunId: parsed.activeProgressRunId,
        remaining: parsed.remaining,
        total: typeof parsed.total === "number" ? parsed.total : parsed.remaining
      };
    }
  } catch {
    /* fall through to clear */
  }

  window.localStorage.removeItem(AUTO_PROGRESS_INTENT_STORAGE_KEY);
  return undefined;
}

function saveAutoProgressIntent(intent: AutoProgressIntent): void {
  try {
    window.localStorage.setItem(AUTO_PROGRESS_INTENT_STORAGE_KEY, JSON.stringify(intent));
  } catch {
    /* best-effort; resume after refresh is non-critical */
  }
}

function clearAutoProgressIntent(): void {
  window.localStorage.removeItem(AUTO_PROGRESS_INTENT_STORAGE_KEY);
}

function autoProgressIntentMatchesState(intent: AutoProgressIntent, state: AppState): boolean {
  return (
    intent.simulationId === state.simulation.id &&
    intent.activeSessionId === state.simulation.activeSessionId &&
    intent.activeProgressRunId === state.activeProgressRunId
  );
}

function getPersonalLlmSecret(vault: PersonalApiVault, provider: LlmApiSettings["provider"]): PersonalSecretRecord {
  if (provider === "mock") {
    return {
      apiKey: "",
      registrationStatus: "registered",
      verificationMessage: "Mock 모드는 API 키가 필요 없습니다."
    };
  }

  return normalizeStoredSecretRecord(vault.llmByProvider[provider], "개인 LLM 키가 저장되어 있습니다.");
}

function applyPersonalApiVault(state: AppState, vault: PersonalApiVault): AppState {
  const llmSecret = getPersonalLlmSecret(vault, state.llm.provider);
  const novelAiSecret = normalizeStoredNovelAiSecretRecord(vault.novelAi);
  return {
    ...state,
    llm: {
      ...state.llm,
      apiKey: llmSecret.apiKey,
      enabled: state.llm.provider === "mock" ? false : state.llm.enabled || hasStoredSecret(llmSecret),
      registrationStatus: llmSecret.registrationStatus,
      verifiedAt: llmSecret.verifiedAt,
      verificationMessage: llmSecret.verificationMessage
    },
    novelAi: {
      ...state.novelAi,
      apiKey: novelAiSecret.apiKey,
      registrationStatus: novelAiSecret.registrationStatus,
      verifiedAt: novelAiSecret.verifiedAt,
      verificationMessage: novelAiSecret.verificationMessage,
      subscriptionTier: novelAiSecret.subscriptionTier,
      // Vibe transfer is global: the vault is the source of truth, so it applies to whichever simulation
      // is active (including brand-new ones / new progressions).
      vibeTransferEnabled: vault.vibeTransfer.enabled,
      vibeTransferReferences: vault.vibeTransfer.references
    }
  };
}

function hasPersonalApiVaultSecrets(vault: PersonalApiVault): boolean {
  return (
    Boolean(vault.imageStoragePath.trim()) ||
    hasStoredSecret(vault.novelAi) ||
    vault.vibeTransfer.references.length > 0 ||
    Object.values(vault.llmByProvider).some((record) => Boolean(record && hasStoredSecret(record)))
  );
}

function mergePersonalSecretRecord(
  local: PersonalSecretRecord | undefined,
  server: PersonalSecretRecord | undefined,
  options: StoredSecretNormalizationOptions = {}
): PersonalSecretRecord {
  const localSecret = normalizeStoredSecretRecord(local, "개인 API 키가 저장되어 있습니다.", options);
  const serverSecret = normalizeStoredSecretRecord(server, "개인 API 키가 저장되어 있습니다.", options);

  if (!hasStoredSecret(localSecret) && hasStoredSecret(serverSecret)) {
    return serverSecret;
  }

  if (hasStoredSecret(localSecret)) {
    return localSecret;
  }

  return serverSecret.registrationStatus !== "idle" || serverSecret.verificationMessage
    ? serverSecret
    : localSecret;
}

function mergePersonalApiVaults(localVault: PersonalApiVault, serverVault: PersonalApiVault): PersonalApiVault {
  const providers = new Set<LlmApiSettings["provider"]>([
    ...(Object.keys(localVault.llmByProvider) as LlmApiSettings["provider"][]),
    ...(Object.keys(serverVault.llmByProvider) as LlmApiSettings["provider"][])
  ]);
  const llmByProvider = Object.fromEntries(
    [...providers].map((provider) => [
      provider,
      mergePersonalSecretRecord(localVault.llmByProvider[provider], serverVault.llmByProvider[provider])
    ])
  ) as Partial<Record<LlmApiSettings["provider"], PersonalSecretRecord>>;
  const localUpdatedAt = new Date(localVault.updatedAt ?? 0).getTime();
  const serverUpdatedAt = new Date(serverVault.updatedAt ?? 0).getTime();

  const newerVault = serverUpdatedAt > localUpdatedAt ? serverVault : localVault;
  // Vibe transfer (references + encodings) is taken from whichever side is newer, but never let a newer-but-
  // empty side wipe a populated one (e.g. a fresh device that synced before the user re-added references).
  const mergedVibeTransfer =
    newerVault.vibeTransfer.references.length > 0
      ? newerVault.vibeTransfer
      : localVault.vibeTransfer.references.length > 0
        ? localVault.vibeTransfer
        : serverVault.vibeTransfer;

  return {
    llmByProvider,
    novelAi: mergePersonalSecretRecord(localVault.novelAi, serverVault.novelAi, { discardAuthFailures: true }),
    vibeTransfer: normalizeStoredVibeTransferSettings(mergedVibeTransfer),
    imageStoragePath: serverUpdatedAt > localUpdatedAt
      ? serverVault.imageStoragePath.trim()
      : localVault.imageStoragePath.trim() || serverVault.imageStoragePath.trim(),
    updatedAt: serverUpdatedAt > localUpdatedAt ? serverVault.updatedAt : localVault.updatedAt
  };
}

function arePersonalApiVaultsEqual(left: PersonalApiVault, right: PersonalApiVault): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function readFreshPersonalApiVault(fallbackVault: PersonalApiVault): PersonalApiVault {
  const storedVault = loadPersonalApiVault();
  return hasPersonalApiVaultSecrets(storedVault) || storedVault.updatedAt
    ? mergePersonalApiVaults(fallbackVault, storedVault)
    : fallbackVault;
}

function getOpsRailWidthBounds(): { min: number; max: number } {
  if (typeof window === "undefined") {
    return {
      min: MIN_OPS_RAIL_WIDTH,
      max: MAX_OPS_RAIL_WIDTH
    };
  }

  return {
    min: MIN_OPS_RAIL_WIDTH,
    max: Math.max(MIN_OPS_RAIL_WIDTH, Math.min(MAX_OPS_RAIL_WIDTH, window.innerWidth - 520))
  };
}

function clampOpsRailWidth(width: number): number {
  const bounds = getOpsRailWidthBounds();
  return Math.round(Math.min(bounds.max, Math.max(bounds.min, width)));
}

function loadOpsRailWidth(): number {
  if (typeof window === "undefined") {
    return DEFAULT_OPS_RAIL_WIDTH;
  }

  const value = Number(window.localStorage.getItem(OPS_RAIL_WIDTH_STORAGE_KEY));
  return Number.isFinite(value) && value > 0 ? clampOpsRailWidth(value) : clampOpsRailWidth(DEFAULT_OPS_RAIL_WIDTH);
}

function saveOpsRailWidth(width: number): void {
  window.localStorage.setItem(OPS_RAIL_WIDTH_STORAGE_KEY, String(clampOpsRailWidth(width)));
}

// The settings tab a person last used opens again next time. A convenience only: storage that is blocked or
// holds something else falls back to the model tab.
function loadSettingsSection(): SettingsSection {
  try {
    const stored = window.localStorage.getItem(SETTINGS_SECTION_STORAGE_KEY);
    return SETTINGS_SECTIONS.find((section) => section.id === stored)?.id ?? "model";
  } catch {
    return "model";
  }
}

function saveSettingsSection(section: SettingsSection): void {
  try {
    window.localStorage.setItem(SETTINGS_SECTION_STORAGE_KEY, section);
  } catch {
    // Not remembered; the tab still switches.
  }
}

interface StoryScrollSnapshot {
  simulationId: string;
  /* A progress-run switch keeps simulation.id and swaps only the message array, so the scroll effect saw
     "same simulation, different message count" and kept the previous run's offset — dropping the reader at
     an arbitrary scrollTop inside unfamiliar prose. The run is part of the identity of what is on screen. */
  progressRunId: string;
  messageCount: number;
  isSending: boolean;
  imageJobStatusSignature: string;
  imageAssetCount: number;
  scrollHeight: number;
  pinnedToBottom: boolean;
}

function isStoryScrollPinnedToBottom(element: HTMLElement): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= STORY_SCROLL_BOTTOM_THRESHOLD;
}

function scrollStoryToBottom(element: HTMLElement, behavior: ScrollBehavior): void {
  element.scrollTo({ top: element.scrollHeight, behavior });
}

function loadSimulationLibrary(fallbackState: AppState): SimulationLibrary {
  const raw = window.localStorage.getItem(SIMULATION_LIBRARY_STORAGE_KEY);
  if (!raw) {
    return compactSimulationRunDuplicates(withBuiltInSimulations([fallbackState]));
  }

  try {
    const parsed = JSON.parse(raw) as AppState[];
    const hydrated = parsed.map((item) => hydrateState(item));
    return compactSimulationRunDuplicates(withBuiltInSimulations(upsertSimulationInLibraryRaw(hydrated, fallbackState)));
  } catch {
    window.localStorage.removeItem(SIMULATION_LIBRARY_STORAGE_KEY);
    return compactSimulationRunDuplicates(withBuiltInSimulations([fallbackState]));
  }
}

function saveSimulationLibrary(library: SimulationLibrary): void {
  const previousRawLibrary = window.localStorage.getItem(SIMULATION_LIBRARY_STORAGE_KEY);
  const shareableLibrary = library.map(stripLibrarySecrets);
  const cacheLibrary = shareableLibrary.map(stripCachedImagePayloadsFromState);
  try {
    window.localStorage.setItem(SIMULATION_LIBRARY_STORAGE_KEY, JSON.stringify(cacheLibrary));
  } catch (error) {
    console.warn("DynamicChat simulation library cache was too large; saving a lightweight index.", error);
    if (previousRawLibrary) {
      return;
    }

    try {
      window.localStorage.setItem(SIMULATION_LIBRARY_STORAGE_KEY, JSON.stringify(createLightweightSimulationLibraryIndex(cacheLibrary)));
    } catch {
      window.localStorage.removeItem(SIMULATION_LIBRARY_STORAGE_KEY);
    }
  }
}

function mergeSimulationLibraries(...libraries: SimulationLibrary[]): SimulationLibrary {
  return compactSimulationRunDuplicates(withBuiltInSimulations(libraries.flat().reduce((current, item) => upsertSimulationInLibraryRaw(current, item), [] as SimulationLibrary)));
}

function pickInitialSimulation(localState: AppState, library: SimulationLibrary, preferLocalMatch: boolean): AppState {
  const localUpdatedAt = new Date(localState.simulation.updatedAt).getTime();
  const matching = library.find((item) => item.simulation.id === localState.simulation.id);

  if (preferLocalMatch && matching) {
    // If library (server) version is newer or equal, use it; otherwise local is newer so keep local.
    return new Date(matching.simulation.updatedAt).getTime() >= localUpdatedAt ? matching : localState;
  }

  return library[0] ?? localState;
}

function hasRuntimeAdvancedSinceSnapshot(current: AppState, snapshot: AppState): boolean {
  if (
    current.messages.length > snapshot.messages.length ||
    current.memoryEvents.length > snapshot.memoryEvents.length ||
    current.contextPacks.length > snapshot.contextPacks.length ||
    current.imageJobs.length > snapshot.imageJobs.length ||
    current.imageAssets.length > snapshot.imageAssets.length
  ) {
    return true;
  }

  return getRuntimeActivityTime(current) > getRuntimeActivityTime(snapshot);
}

function getRuntimeActivityTime(state: AppState): number {
  const values = [
    ...state.messages.map((item) => item.createdAt),
    ...state.memoryEvents.map((item) => item.createdAt),
    ...state.contextPacks.map((item) => item.createdAt),
    ...state.imageAssets.map((item) => item.createdAt),
    ...state.imageJobs.map((item) => item.updatedAt ?? item.completedAt ?? item.createdAt)
  ];
  return values.reduce((latest, value) => {
    const time = Date.parse(value ?? "");
    return Number.isFinite(time) ? Math.max(latest, time) : latest;
  }, 0);
}

function withBuiltInSimulations(library: SimulationLibrary): SimulationLibrary {
  const existingIds = new Set(library.map((item) => item.simulation.id));
  const missingBuiltIns = builtInSimulationStates.filter((builtInState) => !existingIds.has(builtInState.simulation.id)).map(stripLibrarySecrets);
  return [...library, ...missingBuiltIns].sort(compareSimulationUpdatedAt);
}

function upsertSimulationInLibrary(library: SimulationLibrary, nextState: AppState): SimulationLibrary {
  return compactSimulationRunDuplicates(upsertSimulationInLibraryRaw(library, nextState)).sort(compareSimulationUpdatedAt);
}

function upsertSimulationInLibraryRaw(library: SimulationLibrary, nextState: AppState): SimulationLibrary {
  const redactedState = stripCachedImagePayloadsFromState(stripLibrarySecrets(nextState));
  const exists = library.some((item) => item.simulation.id === nextState.simulation.id);
  const nextLibrary = exists
    ? library.map((item) => (item.simulation.id === nextState.simulation.id ? redactedState : item))
    : [redactedState, ...library];

  return nextLibrary.sort(compareSimulationUpdatedAt);
}

function compactSimulationRunDuplicates(library: SimulationLibrary): SimulationLibrary {
  const hydrated = library.map((item) => hydrateState(item));
  const runTitleGroups = new Set(
    hydrated
      .map((item) => parseSimulationProgressTitle(item.simulation.title))
      .filter((parts): parts is { baseTitle: string; runTitle: string } => Boolean(parts))
      .map((parts) => parts.baseTitle)
  );

  if (runTitleGroups.size === 0) {
    return hydrated;
  }

  const consumedIds = new Set<string>();
  const compacted: SimulationLibrary = [];

  for (const baseTitle of runTitleGroups) {
    const group = hydrated.filter((item) => {
      const parts = parseSimulationProgressTitle(item.simulation.title);
      return parts?.baseTitle === baseTitle || item.simulation.title === baseTitle;
    });

    if (group.length === 0) {
      continue;
    }

    const primary = group.find((item) => item.simulation.title === baseTitle) ?? group.slice().sort(compareSimulationUpdatedAt)[0];
    const merged = mergeSimulationProgressGroup(primary, group, baseTitle);
    compacted.push(merged);
    group.forEach((item) => consumedIds.add(item.simulation.id));
  }

  return [
    ...compacted,
    ...hydrated.filter((item) => !consumedIds.has(item.simulation.id))
  ].sort(compareSimulationUpdatedAt);
}

function mergeSimulationProgressGroup(primary: AppState, group: AppState[], baseTitle: string): AppState {
  const primarySimulationId = primary.simulation.id;
  const mergedRuns = group
    .flatMap((item) => {
      const parts = parseSimulationProgressTitle(item.simulation.title);
      const shouldUseSimulationRunTitle = Boolean(parts?.runTitle && item.progressRuns.length === 1);
      return item.progressRuns.map((run) =>
        retargetProgressRun(run, primarySimulationId, shouldUseSimulationRunTitle ? parts?.runTitle : undefined)
      );
    })
    .reduce((runs, run) => upsertProgressRunInList(runs, run), [] as SimulationProgressRun[])
    .sort(compareProgressRunUpdatedAt);
  const latestRun = mergedRuns[0] ?? retargetProgressRun(primary.progressRuns[0], primarySimulationId);
  const mergedAuditLog = group
    .flatMap((item) => item.auditLog)
    .map((event) => ({
      ...event,
      simulationId: primarySimulationId,
      resourceId: event.resourceType === "simulation" ? primarySimulationId : event.resourceId
    }));

  return hydrateState({
    ...primary,
    simulation: {
      ...primary.simulation,
      title: baseTitle,
      activeSessionId: latestRun.activeSessionId,
      updatedAt: maxIsoTimestamp([primary.simulation.updatedAt, ...group.map((item) => item.simulation.updatedAt), ...mergedRuns.map((run) => run.updatedAt)])
    },
    activeProgressRunId: latestRun.id,
    progressRuns: mergedRuns,
    messages: latestRun.messages,
    memoryEvents: latestRun.memoryEvents,
    contextPacks: latestRun.contextPacks,
    handoffs: latestRun.handoffs,
    continuityChecks: latestRun.continuityChecks,
    promptModuleUsages: latestRun.promptModuleUsages,
    sidecarTraces: latestRun.sidecarTraces,
    turnTraces: latestRun.turnTraces,
    imageAssets: latestRun.imageAssets,
    imageJobs: latestRun.imageJobs,
    auditLog: mergedAuditLog.length > 0 ? mergedAuditLog : primary.auditLog,
    selectedContextPackId: latestRun.selectedContextPackId
  });
}

function retargetProgressRun(
  run: SimulationProgressRun | undefined,
  simulationId: string,
  title?: string
): SimulationProgressRun {
  const fallbackRun: SimulationProgressRun = run ?? {
    id: `run_${simulationId}`,
    simulationId,
    title: "진행 1",
    activeSessionId: "session_default",
    sessionIds: ["session_default"],
    messages: [],
    memoryEvents: [],
    contextPacks: [],
    handoffs: [],
    continuityChecks: [],
    promptModuleUsages: [],
    sidecarTraces: [],
    turnTraces: [],
    imageAssets: [],
    imageJobs: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };

  return {
    ...fallbackRun,
    simulationId,
    title: title ?? fallbackRun.title,
    messages: fallbackRun.messages.map((message) => ({ ...message, simulationId })),
    memoryEvents: fallbackRun.memoryEvents.map((event) => ({ ...event, simulationId })),
    contextPacks: fallbackRun.contextPacks.map((pack) => ({ ...pack, simulationId })),
    handoffs: fallbackRun.handoffs.map((handoff) => ({ ...handoff, simulationId })),
    continuityChecks: fallbackRun.continuityChecks.map((check) => ({ ...check, simulationId })),
    promptModuleUsages: fallbackRun.promptModuleUsages.map((usage) => ({ ...usage, simulationId })),
    sidecarTraces: fallbackRun.sidecarTraces.map((trace) => ({ ...trace, simulationId })),
    turnTraces: fallbackRun.turnTraces.map((trace) => ({ ...trace, simulationId })),
    imageAssets: fallbackRun.imageAssets.map((asset) => ({ ...asset, simulationId })),
    imageJobs: fallbackRun.imageJobs.map((job) => ({ ...job, simulationId }))
  };
}

function parseSimulationProgressTitle(title: string): { baseTitle: string; runTitle: string } | undefined {
  const match = /^(.*?)\s*·\s*(새 진행\s*\d+)\s*$/u.exec(title.trim());
  if (!match) {
    return undefined;
  }

  return {
    baseTitle: match[1]?.trim() || title,
    runTitle: match[2]?.trim() || "새 진행"
  };
}

function upsertProgressRunInList(runs: SimulationProgressRun[], nextRun: SimulationProgressRun): SimulationProgressRun[] {
  return runs.some((run) => run.id === nextRun.id)
    ? runs.map((run) => (run.id === nextRun.id ? nextRun : run))
    : [...runs, nextRun];
}

function compareProgressRunUpdatedAt(a: SimulationProgressRun, b: SimulationProgressRun): number {
  return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
}

function maxIsoTimestamp(values: string[]): string {
  return values.filter(Boolean).sort().at(-1) ?? new Date().toISOString();
}

function compareSimulationUpdatedAt(a: AppState, b: AppState): number {
  return new Date(b.simulation.updatedAt).getTime() - new Date(a.simulation.updatedAt).getTime();
}

function stripLibrarySecrets(state: AppState): AppState {
  return {
    ...state,
    llm: toShareableLlmSettings(state.llm),
    imageTagLlm: {
      ...state.imageTagLlm,
      apiKey: "",
      registrationStatus: "idle" as const,
      verifiedAt: undefined,
      verificationMessage: ""
    },
    novelAi: toShareableNovelAiSettings(state.novelAi),
    // No secret here (ComfyUI runs keyless on this machine), but the last connection check describes this machine,
    // not the simulation. A library entry that predates photoreal has no settings at all and gets the defaults.
    photoreal: toShareablePhotorealSettings(state.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS)
  };
}

function stripCachedImagePayloadsFromState(state: AppState): AppState {
  return {
    ...state,
    imageAssets: state.imageAssets.map(stripCachedImagePayload),
    progressRuns: state.progressRuns.map((run) => ({
      ...run,
      imageAssets: run.imageAssets.map(stripCachedImagePayload)
    }))
  };
}

function createLightweightSimulationLibraryIndex(library: SimulationLibrary): SimulationLibrary {
  return library.map((state) =>
    hydrateState({
      ...stripCachedImagePayloadsFromState(state),
      messages: state.messages.slice(-2),
      memoryEvents: [],
      contextPacks: [],
      handoffs: [],
      continuityChecks: [],
      promptModuleUsages: [],
      sidecarTraces: [],
      turnTraces: [],
      imageAssets: state.imageAssets.slice(-1).map(stripCachedImagePayload),
      imageJobs: state.imageJobs.slice(-1),
      auditLog: [],
      progressRuns: state.progressRuns.slice(0, 12).map((run) => ({
        ...run,
        messages: run.messages.slice(-2),
        memoryEvents: [],
        contextPacks: [],
        handoffs: [],
        continuityChecks: [],
        promptModuleUsages: [],
        sidecarTraces: [],
        turnTraces: [],
        imageAssets: run.imageAssets.slice(-1).map(stripCachedImagePayload),
        imageJobs: run.imageJobs.slice(-1)
      }))
    })
  );
}

function stripCachedImagePayload(asset: ImageAsset): ImageAsset {
  if (!asset.dataUrl) {
    return asset;
  }

  const { dataUrl: _dataUrl, ...metadataOnly } = asset;
  return metadataOnly;
}

function hasMissingImagePayload(asset: ImageAsset): boolean {
  return !asset.dataUrl && !asset.objectKey;
}

async function persistRuntimeImagePayloads(simulationId: string, assets: ImageAsset[]): Promise<ImageAsset[]> {
  if (!assets.some((asset) => asset.dataUrl)) {
    return assets;
  }

  try {
    const persistedAssets = await createDynamicChatApiClient().persistImageAssets(
      simulationId,
      assets.filter((asset) => asset.dataUrl)
    );
    const persistedById = new Map(persistedAssets.map((asset) => [asset.id, stripCachedImagePayload(asset)]));
    return assets.map((asset) => persistedById.get(asset.id) ?? asset);
  } catch {
    return assets;
  }
}

function collectImageAssetsWithPayload(state: AppState): ImageAsset[] {
  const assetsById = new Map<string, ImageAsset>();
  for (const asset of collectStateImageAssets(state)) {
    if (asset.dataUrl && !assetsById.has(asset.id)) {
      assetsById.set(asset.id, asset);
    }
  }
  return [...assetsById.values()];
}

function createImagePayloadCompactionSignature(state: AppState): string | undefined {
  const payloadAssets = collectImageAssetsWithPayload(state);
  if (payloadAssets.length === 0) {
    return undefined;
  }

  return `${state.simulation.id}:${payloadAssets
    .map((asset) => `${asset.id}:${asset.dataUrl?.length ?? 0}`)
    .sort()
    .join("|")}`;
}

function compactImagePayloadsInState(state: AppState, persistedAssets: ImageAsset[]): AppState {
  const compactedById = new Map(
    persistedAssets
      .filter((asset) => asset.objectKey)
      .map((asset) => [asset.id, stripCachedImagePayload(asset)])
  );
  if (compactedById.size === 0) {
    return state;
  }

  let changed = false;
  const compactAssets = (assets: ImageAsset[]): ImageAsset[] =>
    assets.map((asset) => {
      const compacted = compactedById.get(asset.id);
      if (!compacted) {
        return asset;
      }

      const { dataUrl: _dataUrl, ...assetWithoutPayload } = asset;
      const nextAsset = {
        ...assetWithoutPayload,
        ...compacted
      };
      changed ||= asset.dataUrl !== undefined || nextAsset.objectKey !== asset.objectKey || nextAsset.mimeType !== asset.mimeType;
      return nextAsset;
    });

  const imageAssets = compactAssets(state.imageAssets);
  const progressRuns = state.progressRuns.map((run) => ({
    ...run,
    imageAssets: compactAssets(run.imageAssets)
  }));

  return changed
    ? {
        ...state,
        imageAssets,
        progressRuns
      }
    : state;
}

function collectDeletedImageAssetIds(state: AppState): Set<string> {
  return new Set(
    state.redactionQueue
      .filter((redaction) => redaction.targetType === "image_asset" && redaction.status !== "failed")
      .map((redaction) => redaction.targetId)
      .filter(Boolean)
  );
}

function collectStateImageAssets(state: AppState): ImageAsset[] {
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  return [
    ...state.imageAssets,
    ...state.progressRuns.flatMap((run) => run.imageAssets)
  ].filter((asset) => !deletedAssetIds.has(asset.id));
}

function collectReferencedImageAssetIds(state: AppState): string[] {
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  return Array.from(
    new Set(
      [
        ...state.messages.flatMap((message) => message.imageAssetIds),
        ...state.turnTraces.flatMap((trace) => trace.imageAssetIds),
        ...state.imageJobs.flatMap((job) => job.assetIds),
        ...state.progressRuns.flatMap((run) => [
          ...run.messages.flatMap((message) => message.imageAssetIds),
          ...run.turnTraces.flatMap((trace) => trace.imageAssetIds),
          ...run.imageJobs.flatMap((job) => job.assetIds)
        ])
      ].filter((assetId) => Boolean(assetId) && !deletedAssetIds.has(assetId))
    )
  );
}

function collectMissingImagePayloadIds(state: AppState): string[] {
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  const assets = collectStateImageAssets(state);
  const knownAssetIds = new Set(assets.map((asset) => asset.id));
  return Array.from(
    new Set(
      [
        ...assets.filter(hasMissingImagePayload).map((asset) => asset.id),
        ...collectReferencedImageAssetIds(state).filter((assetId) => !knownAssetIds.has(assetId))
      ].filter((assetId) => Boolean(assetId) && !deletedAssetIds.has(assetId))
    )
  ).sort();
}

function createMissingImagePayloadSignature(state: AppState, storagePath: string): string | undefined {
  const missingAssetIds = collectMissingImagePayloadIds(state);
  const storageScope = storagePath.trim() || "default";
  return missingAssetIds.length > 0 ? `${state.simulation.id}:${storageScope}:${missingAssetIds.join("|")}` : undefined;
}

function mergeHydratedImagePayloads(currentAssets: ImageAsset[], hydratedAssets: ImageAsset[], appendMissing = false): ImageAsset[] {
  if (hydratedAssets.length === 0) {
    return currentAssets;
  }

  const hydratedById = new Map(hydratedAssets.map((asset) => [asset.id, asset.objectKey ? stripCachedImagePayload(asset) : asset]));
  let changed = false;
  const merged = currentAssets.map((asset) => {
    const hydrated = hydratedById.get(asset.id);
    if (!hydrated || (!hydrated.dataUrl && !hydrated.objectKey)) {
      return asset;
    }

    const nextObjectKey = asset.objectKey ?? hydrated.objectKey;
    const nextAsset = {
      ...asset,
      dataUrl: asset.dataUrl ?? (nextObjectKey ? undefined : hydrated.dataUrl),
      objectKey: nextObjectKey,
      mimeType: asset.mimeType ?? hydrated.mimeType
    };
    changed ||= nextAsset.dataUrl !== asset.dataUrl || nextAsset.objectKey !== asset.objectKey || nextAsset.mimeType !== asset.mimeType;
    return nextAsset;
  });

  const mergedIds = new Set(merged.map((asset) => asset.id));
  const missingAssets = appendMissing
    ? hydratedAssets
        .filter((asset) => !mergedIds.has(asset.id) && (asset.dataUrl || asset.objectKey))
        .map((asset) => (asset.objectKey ? stripCachedImagePayload(asset) : asset))
    : [];

  return changed || missingAssets.length > 0 ? [...merged, ...missingAssets] : currentAssets;
}

function mergeHydratedImagePayloadsIntoState(state: AppState, hydratedAssets: ImageAsset[]): AppState {
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  const allowedHydratedAssets = hydratedAssets.filter((asset) => !deletedAssetIds.has(asset.id));
  const imageAssets = mergeHydratedImagePayloads(state.imageAssets, allowedHydratedAssets, true);
  let progressRunsChanged = false;
  const progressRuns = state.progressRuns.map((run) => {
    const runImageAssets = mergeHydratedImagePayloads(run.imageAssets, allowedHydratedAssets);
    if (runImageAssets === run.imageAssets) {
      return run;
    }

    progressRunsChanged = true;
    return {
      ...run,
      imageAssets: runImageAssets
    };
  });

  return imageAssets === state.imageAssets && !progressRunsChanged
    ? state
    : {
        ...state,
        imageAssets,
        progressRuns
      };
}

/** Maximum number of progress runs to persist. Older runs beyond this cap are
 *  silently dropped at save time (in-memory state is unaffected). 30 gives a
 *  generous history without letting the stored file grow unboundedly. */
const MAX_STORED_PROGRESS_RUNS = 30;

/**
 * Returns a copy of the given state where every stored progressRun has been
 * slimmed (heavy NAI payloads + debug traces removed) and the list is capped
 * to MAX_STORED_PROGRESS_RUNS most-recent entries.
 *
 * This ONLY affects the snapshot that gets written to storage. The live
 * in-memory state (state.imageAssets, state.imageJobs, state.turnTraces …)
 * is intentionally left untouched so the active session keeps all runtime
 * data it needs.
 */
function slimStateProgressRunsForStorage(state: AppState): AppState {
  const sorted = [...state.progressRuns].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  );
  const cappedSlimRuns = sorted.slice(0, MAX_STORED_PROGRESS_RUNS).map(slimProgressRunForStorage);
  return {
    ...state,
    progressRuns: cappedSlimRuns
  };
}

function saveStateSnapshot(state: AppState, options?: SaveStateOptions): void {
  saveState(slimStateProgressRunsForStorage(state), options);
}

function scheduleIdleTask(callback: () => void, timeout = 1400): () => void {
  if ("requestIdleCallback" in window) {
    const handle = window.requestIdleCallback(callback, { timeout });
    return () => window.cancelIdleCallback(handle);
  }

  const handle = globalThis.setTimeout(callback, Math.min(timeout, 250));
  return () => globalThis.clearTimeout(handle);
}

function getLlmProviderOption(provider: LlmApiSettings["provider"]) {
  return llmProviderOptions.find((option) => option.value === provider) ?? llmProviderOptions[0];
}

function createLlmProviderPatch(provider: LlmApiSettings["provider"]): Partial<LlmApiSettings> {
  const option = getLlmProviderOption(provider);
  return {
    provider,
    enabled: provider !== "mock",
    baseUrl: option.baseUrl,
    model: option.defaultModel,
    // Reset with the rest of the connection. A context size is a property of the backend, not of the app:
    // leaving a hosted provider's 128k behind when switching to a local one made that number the prompt
    // budget AND Ollama's num_ctx, so the request was built for a window the model does not have.
    contextTokens: option.contextTokens,
    registrationStatus: provider === "mock" ? "registered" : "idle",
    verifiedAt: provider === "mock" ? new Date().toISOString() : undefined,
    verificationMessage: provider === "mock" ? "Mock 모드로 등록되었습니다." : ""
  };
}

async function createVerifiedLlmPatch(settings: LlmApiSettings): Promise<Partial<LlmApiSettings>> {
  const result = await validateLlmApi(settings);
  return {
    enabled: result.ok ? settings.provider !== "mock" : settings.enabled,
    registrationStatus: result.ok ? "registered" : "failed",
    verifiedAt: result.verifiedAt,
    verificationMessage: result.message
  };
}

async function createVerifiedNovelAiPatch(settings: NovelAiApiSettings): Promise<Partial<NovelAiApiSettings>> {
  const result = await validateNovelAiApi(settings);
  return {
    enabled: result.ok ? true : settings.enabled,
    apiKey: result.details?.normalizedApiKey ?? normalizeApiToken(settings.apiKey),
    requestMode: result.ok ? "proxy" : settings.requestMode,
    proxyUrl: result.ok ? getNovelAiGenerateProxyUrl() : settings.proxyUrl,
    registrationStatus: result.ok ? "registered" : "failed",
    verifiedAt: result.verifiedAt,
    verificationMessage: result.message,
    subscriptionTier: result.details?.tier
  };
}

// The subset of a turn that image dispatch actually needs. A full TurnResult satisfies it, and the
// early (mid-stream) dispatch synthesizes a minimal one before the turn has fully resolved.
interface ImageDispatchTurn {
  userMessage: ChatMessage;
  assistantMessage: ChatMessage;
  contextPack: ContextPack;
  promptModuleUsages: PromptModuleUsage[];
  sidecar?: AssistantSidecar;
  sidecarTrace?: SidecarTrace;
  turnTrace: { id: string };
  annotationFailureReason?: string;
}

// Payload the engine hands back the instant it has parsed the (front-loaded) image_cues mid-stream,
// carrying the real turn context so the client can dispatch the image before the narrative finishes.
interface EarlyImageCuePayload {
  cues: AssistantImageCueDraft[];
  userMessage: ChatMessage;
  assistantMessage: ChatMessage;
  contextPack: ContextPack;
  promptModuleUsages: PromptModuleUsage[];
  sidecar: AssistantSidecar;
  /** Planning snapshot with this turn's messages and state deltas applied (runtime secrets already resolved). */
  state: AppState;
}

async function runRunnableImageJobs(
  jobs: ImageGenerationJob[],
  snapshot: AppState,
  runJob: (job: ImageGenerationJob, snapshot: AppState) => Promise<void>
): Promise<void> {
  const runnableJobs = jobs.filter((job) => shouldAutoRunImageJob(job, snapshot));
  if (runnableJobs.length === 0) {
    return;
  }

  // One feeder per renderer, each handing its chain the next cut as the previous one finishes. A shared pool of
  // workers waited on whichever chain its cut belonged to, so a turn opening with two photoreal cuts parked both
  // workers on the photoreal chain while the NovelAI chain sat idle (and the reverse) — undoing the per-backend
  // chains. Feeding one cut at a time, rather than queueing the whole turn at once, also lets a regeneration the
  // user asks for meanwhile run next instead of behind the rest of this turn. (setState merges via
  // upsertImageAssets, so concurrent completions on the two chains are race-safe.)
  const byBackend = new Map<ImageBackendId, ImageGenerationJob[]>();
  for (const job of runnableJobs) {
    const backend = readImageJobBackend(job);
    byBackend.set(backend, [...(byBackend.get(backend) ?? []), job]);
  }
  await Promise.all(
    Array.from(byBackend.values(), async (backendJobs) => {
      for (const job of backendJobs) {
        await runJob(job, snapshot);
      }
    })
  );
}

/**
 * Appends `work` to one renderer's serial chain and returns its result. The story's cuts and the builder's profile
 * tests share these chains: NovelAI answers a second concurrent generation on one account with 429, and photoreal
 * renders share one GPU, while the two renderers never compete with each other. The stored tail always settles
 * fulfilled, so one failed task releases the chain instead of jamming every task behind it.
 */
function chainImageBackendWork<T>(
  queues: Map<ImageBackendId, Promise<void>>,
  backend: ImageBackendId,
  work: () => Promise<T>
): Promise<T> {
  const queued = (queues.get(backend) ?? Promise.resolve()).then(work, work);
  queues.set(
    backend,
    queued.then(
      () => undefined,
      () => undefined
    )
  );
  return queued;
}

/**
 * Waits for `work`, but gives up the moment `signal` aborts while the work is still QUEUED (`hasStarted()` false).
 *
 * A turn can wait in the local GPU queue behind a photoreal render — seconds, tens of seconds on a cold model
 * load — and 중지 has to end the turn then, not when the picture finishes; the queued task checks the signal
 * itself before it starts, so a cancelled turn never runs. Once the work HAS started, its own cancellation (the
 * provider fetch abort) is what ends it, exactly as before: settling early there would roll the partial turn back
 * while the stream could still deliver one more chunk and write it back on screen.
 */
function awaitQueuedWork<T>(work: Promise<T>, signal: AbortSignal, hasStarted: () => boolean): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      if (!hasStarted()) {
        reject(createQueuedWorkCancelledError());
      }
    };
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) {
      onAbort();
    }
    work.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      }
    );
  });
}

function createQueuedWorkCancelledError(): Error {
  return new DOMException("대기 중이던 턴을 취소했습니다.", "AbortError");
}

function runAfterNextPaint(callback: () => void): void {
  if (document.visibilityState === "hidden") {
    window.setTimeout(callback, 0);
    return;
  }

  window.requestAnimationFrame(() => {
    window.setTimeout(callback, 0);
  });
}

function readProviderCost(payload: Record<string, unknown>): number | undefined {
  const raw = payload.estimatedAnlas ?? payload.estimatedCost ?? payload.cost;
  return typeof raw === "number" && Number.isFinite(raw) ? raw : undefined;
}

function readImageCuePlannerSource(payload: Record<string, unknown>): string | undefined {
  const planner = payload.imageCuePlanner;
  if (!planner || typeof planner !== "object" || Array.isArray(planner)) {
    return undefined;
  }

  const source = (planner as Record<string, unknown>).source;
  return typeof source === "string" && source.trim() ? source : undefined;
}

function getImageJobStatusLabel(status: ImageGenerationJob["status"]): string {
  return imageJobStatusLabels[status] ?? status;
}

/**
 * The renderer a job runs on. `backend` is the first-class field; the payload's `provider` covers a job whose
 * field was lost, and absence means NovelAI — every job planned before photoreal existed.
 */
function readImageJobBackend(job: ImageGenerationJob): ImageBackendId {
  if (job.backend === "novelai" || job.backend === "photoreal") {
    return job.backend;
  }
  return job.providerPayload?.provider === "photoreal" ? "photoreal" : "novelai";
}

/** The dispatch that claimed this job record (see runQueuedImageJob), or undefined for a record no run has claimed. */
function readImageJobDispatchToken(job: ImageGenerationJob | undefined): string | undefined {
  const token = job?.providerPayload?.dispatchToken;
  return typeof token === "string" && token ? token : undefined;
}

/**
 * A job error without the "<renderer> 이미지 생성 실패:" lead the clients already put on it. Every place that shows
 * one adds that lead itself (the toast, the card title), and the photoreal client's errors carry it too, so the
 * failure used to read "실사 이미지 생성 실패: 실사 이미지 생성 실패: …". Empty when the error was nothing but the
 * lead: the caller's own lead already says all there is.
 */
function stripImageFailureLead(message: string, backend: ImageBackendId): string {
  const lead = new RegExp(`^(?:\\s*${imageBackendLabels[backend]}\\s*이미지\\s*생성\\s*실패\\s*[:：]?\\s*)+`, "u");
  return message.replace(lead, "").trim();
}

/**
 * The steps a photoreal preset really samples when its graph fixes the schedule. Qwen-Image 2.1 Turbo runs the
 * distilled sigma list baked into its graph (ManualSigmas), so the 스텝 setting recorded on the job never reaches
 * the sampler; one step per sigma interval.
 */
function readFixedPhotorealSteps(presetId: string | undefined): number | undefined {
  return presetId === "qwen_image_2_1_turbo" ? QWEN21_TURBO_SIGMAS.split(",").length - 1 : undefined;
}

/**
 * The renderer that drew an asset, or undefined when none did: stored and fallback art was never generated, so
 * it gets no badge rather than a guessed "NAI". A generated asset without the field predates photoreal.
 */
function readImageAssetBackend(asset: ImageAsset): ImageBackendId | undefined {
  if (asset.backend === "novelai" || asset.backend === "photoreal") {
    return asset.backend;
  }
  return asset.source === "generated" ? "novelai" : undefined;
}

/**
 * Names of the characters whose profile image a photoreal cut used as its face reference. The planner records them
 * as `photorealReferences` and the ComfyUI client as `references`; both carry characterId, never the image bytes.
 */
function readPhotorealReferenceCharacterNames(state: AppState, payload: Record<string, unknown>): string[] {
  const entries = [payload.photorealReferences, payload.references].find((value): value is unknown[] => Array.isArray(value)) ?? [];
  const names = entries.flatMap((entry) => {
    if (!isPlainRecord(entry)) {
      return [];
    }
    const characterId = typeof entry.characterId === "string" ? entry.characterId : undefined;
    const character = characterId ? state.characters.find((candidate) => candidate.id === characterId) : undefined;
    const name = character?.name ?? (typeof entry.name === "string" ? entry.name : undefined);
    return name ? [name] : [];
  });
  return uniqueIds(names);
}

function getImageSceneCategoryLabel(category: ImageSceneCategory | undefined): string | undefined {
  return category ? IMAGE_SCENE_CATEGORY_OPTIONS.find((option) => option.id === category)?.label ?? category : undefined;
}

/**
 * The renderers this simulation's cuts actually reach. Routing says where a cut WANTS to go; only a connected
 * backend draws it, and an off backend hands its cuts to the other one (decideImageBackend's fallback). A
 * photoreal setup that routing never sends anything to — NovelAI-only mode with NovelAI on — is not listed.
 */
function listActiveImageBackends(state: Pick<AppState, "imageProfile" | "novelAi" | "photoreal">): ImageBackendId[] {
  const routing = resolveImageBackendRouting(state.imageProfile);
  const novelAiOn = state.novelAi?.enabled === true;
  const photorealOn = state.photoreal?.enabled === true;
  const routesToPhotoreal = routing.mode === "photoreal" || (routing.mode === "mixed" && routing.photorealCategories.length > 0);
  const backends: ImageBackendId[] = [];
  if (novelAiOn) {
    backends.push("novelai");
  }
  if (photorealOn && (routesToPhotoreal || !novelAiOn)) {
    backends.push("photoreal");
  }
  return backends;
}

function describeActiveImageBackends(state: Pick<AppState, "imageProfile" | "novelAi" | "photoreal">): string | undefined {
  const backends = listActiveImageBackends(state);
  return backends.length > 0 ? backends.map((backend) => imageBackendLabels[backend]).join(" + ") : undefined;
}

function readProviderPayloadNumber(payload: Record<string, unknown>, key: string): number | undefined {
  const direct = payload[key];
  if (typeof direct === "number" && Number.isFinite(direct)) {
    return direct;
  }

  const parameters = payload.parameters;
  if (parameters && typeof parameters === "object" && !Array.isArray(parameters)) {
    const nested = (parameters as Record<string, unknown>)[key];
    return typeof nested === "number" && Number.isFinite(nested) ? nested : undefined;
  }

  return undefined;
}

function readProviderPayloadText(payload: Record<string, unknown>, key: string): string | undefined {
  const direct = payload[key];
  if (typeof direct === "string" && direct.trim()) {
    return direct;
  }

  const parameters = payload.parameters;
  if (parameters && typeof parameters === "object" && !Array.isArray(parameters)) {
    const nested = (parameters as Record<string, unknown>)[key];
    return typeof nested === "string" && nested.trim() ? nested : undefined;
  }

  return undefined;
}

function readNovelAiV4CaptionDebug(payload: Record<string, unknown>): string | undefined {
  const parameters = readNovelAiParameters(payload);
  if (!parameters) {
    return undefined;
  }

  const positive = formatNovelAiCaptionBlock("v4_prompt", parameters.v4_prompt);
  const negative = formatNovelAiCaptionBlock("v4_negative_prompt", parameters.v4_negative_prompt);
  return [positive, negative].filter((item): item is string => Boolean(item)).join("\n\n") || undefined;
}

function readNovelAiParameters(payload: Record<string, unknown>): Record<string, unknown> | undefined {
  if (isPlainRecord(payload.parameters)) {
    return payload.parameters;
  }

  const variantResults = Array.isArray(payload.variantResults) ? payload.variantResults : [];
  for (const variant of variantResults) {
    if (!isPlainRecord(variant) || !isPlainRecord(variant.payload) || !isPlainRecord(variant.payload.parameters)) {
      continue;
    }
    return variant.payload.parameters;
  }

  return undefined;
}

function formatNovelAiCaptionBlock(label: string, value: unknown): string | undefined {
  if (!isPlainRecord(value) || !isPlainRecord(value.caption)) {
    return undefined;
  }

  const baseCaption = typeof value.caption.base_caption === "string" ? value.caption.base_caption : "";
  const charCaptions = Array.isArray(value.caption.char_captions) ? value.caption.char_captions : [];
  const lines = [`${label}.base_caption:`, baseCaption || "(empty)"];
  charCaptions.forEach((caption, index) => {
    if (!isPlainRecord(caption)) {
      return;
    }
    const charCaption = typeof caption.char_caption === "string" ? caption.char_caption : "";
    lines.push(`char_captions[${index}]:`, charCaption || "(empty)");
  });
  return lines.join("\n");
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function getImageJobCue(job: ImageGenerationJob): Partial<ImageCue> {
  const cue = job.providerPayload.cue ?? job.providerPayload.plannedCue;
  return cue && typeof cue === "object" && !Array.isArray(cue) ? cue as Partial<ImageCue> : {};
}

function getImageJobBaseCue(job: ImageGenerationJob): Partial<ImageCue> {
  const storedCue = getImageJobCue(job);
  const variants = Array.isArray(job.providerPayload.promptVariants) ? job.providerPayload.promptVariants : [];
  const firstVariant = variants[0];
  const variantCue =
    firstVariant && typeof firstVariant === "object" && !Array.isArray(firstVariant)
      ? (firstVariant as { cue?: unknown }).cue
      : undefined;
  const rawCue =
    variantCue && typeof variantCue === "object" && !Array.isArray(variantCue) ? (variantCue as Partial<ImageCue>) : {};
  // The first variant cue preserves the RAW LLM-authored character_prompts (before local appearance/outfit
  // composition), so prefer it on regeneration to avoid both dropping per-character captions and double-injecting
  // identity/outfit when the plan re-composes them. Falls back to the stored cue for older jobs without variants.
  return { ...storedCue, ...rawCue };
}

function createImageCueForRegeneration(state: AppState, job: ImageGenerationJob): ImageCue {
  const cue = getImageJobBaseCue(job);
  const characters =
    Array.isArray(cue.characters) && cue.characters.length > 0
      ? cue.characters.filter((characterId) => state.characters.some((character) => character.id === characterId))
      : state.characters.slice(0, 3).map((character) => character.id);
  const characterVisualContext = characters
    .map((characterId) => {
      const character = state.characters.find((candidate) => candidate.id === characterId);
      const visualProfile = state.visualProfiles.find((profile) => profile.characterId === characterId);
      return [character?.name, visualProfile?.positivePrompt].filter(Boolean).join(", ");
    })
    .filter(Boolean)
    .join(" | ");

  const tags = Array.isArray(cue.tags) ? cue.tags : [];
  const baseTags = Array.isArray(cue.baseTags) ? cue.baseTags : undefined;
  return {
    shouldGenerate: true,
    reason: job.reason || cue.reason || "이미지 재생성",
    characters,
    tags,
    baseTags,
    characterPrompts: Array.isArray(cue.characterPrompts) ? cue.characterPrompts : undefined,
    scene: typeof cue.scene === "string" ? cue.scene : "current simulation scene",
    suppressionReason: undefined,
    visualContext: [cue.scene, state.simulation.title, characterVisualContext].filter(Boolean).join(", "),
    // The crop the original cut committed to, so "regenerate" redraws the SAME shot instead of re-guessing it from
    // tags (a face close-up must not come back as a full-body cut with shoes injected), and the beat kind the scene
    // classifier reads. Both are re-validated: a stored payload is old data, not a trusted shape.
    frame: cue.frame ? normalizeImageCueFrame(cue.frame, [...(baseTags ?? []), ...tags]) : undefined,
    kind: typeof cue.kind === "string" && cue.kind.trim() ? cue.kind.trim() : undefined
  };
}

function getJobEvidence(state: AppState, job: ImageGenerationJob): ContextEvidence[] {
  const evidenceByNodeId = new Map(
    state.contextPacks.flatMap((pack) =>
      getScopedContextEvidenceForDisplay(state, pack.evidence).map((item) => [item.nodeId, item] as const)
    )
  );
  const matchedEvidence = job.contextNodeIds.map((nodeId) => evidenceByNodeId.get(nodeId)).filter((item): item is ContextEvidence => Boolean(item));
  return matchedEvidence.length > 0 ? matchedEvidence : getScopedContextEvidenceForDisplay(state, state.contextPacks.at(-1)?.evidence ?? []).slice(0, 2);
}

function getScopedContextEvidenceForDisplay(state: AppState, evidence: ContextEvidence[]): ContextEvidence[] {
  return evidence.filter((item) => !referencesForeignRuntimeScope(state, item.nodeId, item.reason, item.snippet));
}

function referencesForeignRuntimeScope(state: AppState, ...values: string[]): boolean {
  const text = values.join("\n");
  const simulationIds = text.match(/\bsim_[A-Za-z0-9_-]+\b/gu) ?? [];
  if (simulationIds.some((simulationId) => simulationId !== state.simulation.id)) {
    return true;
  }

  const allowedSessionIds = getAllowedDisplaySessionIds(state);
  const sessionIds = text.match(/\bsession_[A-Za-z0-9_-]+\b/gu) ?? [];
  if (sessionIds.some((sessionId) => !allowedSessionIds.has(sessionId))) {
    return true;
  }

  const runIds = text.match(/\brun_[A-Za-z0-9_-]+\b/gu) ?? [];
  return runIds.some((runId) => runId !== state.activeProgressRunId);
}

function getAllowedDisplaySessionIds(state: AppState): Set<string> {
  const activeRun = state.progressRuns.find((run) => run.id === state.activeProgressRunId);
  return new Set(
    [
      state.simulation.activeSessionId,
      ...(activeRun?.sessionIds ?? []),
      ...state.messages.map((message) => message.sessionId),
      ...state.memoryEvents.map((event) => event.sessionId),
      ...state.contextPacks.map((pack) => pack.sessionId),
      ...state.handoffs.flatMap((handoff) => [handoff.previousSessionId, handoff.nextSessionId])
    ].filter(Boolean)
  );
}

function getPromptModeOption(promptMode: SimulationPromptMode | undefined) {
  return promptModeOptions.find((option) => option.value === promptMode) ?? promptModeOptions[0];
}

function getPromptModeProfile(promptMode: SimulationPromptMode): string {
  if (promptMode === "basic") {
    return "balanced-agent";
  }

  if (promptMode === "one_on_one") {
    return "one-on-one-agent";
  }

  if (promptMode === "simulation") {
    return "simulation-agent";
  }

  return "custom-agent";
}

function upsertPresetModule(
  modules: PromptModule[],
  template: PromptModule,
  matcher: (module: PromptModule) => boolean
): PromptModule[] {
  const index = modules.findIndex(matcher);
  if (index === -1) {
    return [...modules, template];
  }

  return modules.map((module, moduleIndex) =>
    moduleIndex === index
      ? {
          ...module,
          parentId: template.parentId,
          kind: template.kind,
          title: template.title,
          body: template.body,
          enabled: true,
          priority: template.priority,
          activationTags: template.activationTags,
          characterId: template.characterId,
          tokenPolicy: template.tokenPolicy,
          updatedAt: template.updatedAt
        }
      : module
  );
}

function applyPromptModePresetToDraft(current: SimulationDraft, promptMode: PresetPromptMode): SimulationDraft {
  const preset = promptModePresets[promptMode];
  const now = new Date().toISOString();
  const simulationId = current.modules[0]?.simulationId ?? "draft_simulation";
  const mainModuleId = current.modules.find((module) => module.kind === "main_prompt")?.id ?? "draft_main";
  const primaryCharacterId = current.characters[0]?.id ?? "draft_character_id";
  const primaryCharacter: SimulationCharacterDraft = {
    id: primaryCharacterId,
    name: preset.characterName,
    role: preset.characterRole,
    summary: preset.characterSummary,
    relationship: preset.characterRelationship,
    currentMood: preset.characterMood,
    visualPrompt: preset.visualPrompt,
    negativeVisualPrompt: preset.negativeVisualPrompt,
    defaultOutfitPrompt: current.characters[0]?.defaultOutfitPrompt ?? defaultOutfitPrompt,
    outfitPrompts: current.characters[0]?.outfitPrompts ?? defaultOutfitPrompts,
    expressionPrompts: current.characters[0]?.expressionPrompts ?? defaultExpressionPrompts,
    defaultSafetyLevel: current.characters[0]?.defaultSafetyLevel ?? "safe",
    // The creator's photoreal work on this slot is not part of any preset: a mode card rewrites the character's
    // NovelAI identity, not the faces picked in the profile studio or the 실사 외형 text. Rebuilding the character
    // without these keys lost a fresh pick on create, and on edit made updateStateFromDraft fall back to the STORED
    // profile — so the saved simulation brought back what the editor showed as gone. Copied key for key, so a key
    // the draft never had stays absent.
    ...pickCharacterPhotorealFields(current.characters[0])
  };
  const characters = current.characters.length > 0 ? [primaryCharacter, ...current.characters.slice(1)] : [primaryCharacter];
  const modeTags = ["mode-rule", ...preset.activationTags];
  const templates: Array<{
    module: PromptModule;
    matcher: (module: PromptModule) => boolean;
  }> = [
    {
      module: {
        id: mainModuleId,
        simulationId,
        kind: "main_prompt",
        title: `메인 규칙: ${promptModeLabels[promptMode]}`,
        body: preset.mainPrompt,
        enabled: true,
        priority: 100,
        activationTags: ["core", "always", ...preset.activationTags],
        tokenPolicy: "always",
        version: 1,
        updatedAt: now
      },
      matcher: (module) => module.id === mainModuleId || module.id === "draft_main" || module.kind === "main_prompt"
    },
    {
      module: {
        id: "draft_character",
        simulationId,
        parentId: mainModuleId,
        kind: "character_prompt",
        title: `캐릭터: ${preset.characterName}`,
        body: preset.characterSummary,
        enabled: true,
        priority: promptMode === "one_on_one" ? 90 : 84,
        activationTags: [preset.characterName.toLowerCase(), "character", ...preset.activationTags],
        characterId: primaryCharacterId,
        tokenPolicy: promptMode === "simulation" ? "rag" : "always",
        version: 1,
        updatedAt: now
      },
      matcher: (module) =>
        module.id === "draft_character" ||
        (module.kind === "character_prompt" && module.characterId === primaryCharacterId) ||
        (module.kind === "character_prompt" && !module.characterId && module.parentId === mainModuleId)
    },
    {
      module: {
        id: "draft_world",
        simulationId,
        parentId: mainModuleId,
        kind: "world_lore",
        title: `세계관: ${promptModeLabels[promptMode]}`,
        body: preset.worldLore,
        enabled: true,
        priority: 76,
        activationTags: ["world", "lore", ...preset.activationTags],
        tokenPolicy: "rag",
        version: 1,
        updatedAt: now
      },
      matcher: (module) => module.id === "draft_world" || module.kind === "world_lore"
    },
    {
      module: {
        id: "draft_mode_rule",
        simulationId,
        parentId: mainModuleId,
        kind: "scene_rule",
        title: preset.modeRuleTitle,
        body: preset.modeRule,
        enabled: true,
        priority: promptMode === "simulation" ? 92 : 78,
        activationTags: modeTags,
        tokenPolicy: promptMode === "basic" ? "rag" : "always",
        version: 1,
        updatedAt: now
      },
      matcher: (module) =>
        module.id === "draft_mode_rule" ||
        module.activationTags.includes("mode-rule") ||
        module.title.startsWith("진행 규칙:")
    },
    {
      module: {
        id: "draft_image_style",
        simulationId,
        parentId: mainModuleId,
        kind: "image_prompt_profile",
        title: `이미지 스타일: ${promptModeLabels[promptMode]}`,
        body: preset.imageStylePrompt,
        enabled: true,
        priority: 82,
        activationTags: ["image", "style", "nai", ...preset.activationTags],
        tokenPolicy: "always",
        version: 1,
        updatedAt: now
      },
      matcher: (module) => module.id === "draft_image_style" || module.kind === "image_prompt_profile"
    }
  ];
  const presetBaseModules = current.modules.filter((module) => module.id !== "draft_safety" && module.kind !== "safety_policy");
  const modules = templates.reduce((draftModules, template) => upsertPresetModule(draftModules, template.module, template.matcher), presetBaseModules);

  return {
    ...current,
    promptMode,
    title: preset.title,
    description: preset.description,
    mainPrompt: preset.mainPrompt,
    characterName: preset.characterName,
    characterRole: preset.characterRole,
    characterSummary: preset.characterSummary,
    characterRelationship: preset.characterRelationship,
    characterMood: preset.characterMood,
    worldLore: preset.worldLore,
    startSituationPrompt: preset.startSituationPrompt,
    visualPrompt: preset.visualPrompt,
    negativeVisualPrompt: preset.negativeVisualPrompt,
    defaultOutfitPrompt: primaryCharacter.defaultOutfitPrompt,
    characters,
    modules,
    imageProfile: {
      ...current.imageProfile,
      stylePrompt: preset.imageStylePrompt,
      userRules: preset.imageUserRules
    },
    imageScenePresets: ensureDraftImageScenePresetIds(
      (current.imageScenePresets?.length ?? 0) > 0
        ? current.imageScenePresets
        : createDefaultDraftImageScenePresets(now, promptMode)
    ),
    relationshipMap: {
      ...current.relationshipMap,
      enabled: true,
      statusPrompt: createRelationshipMapPresetPrompt(promptMode),
      updatedAt: now
    },
    llm: {
      ...current.llm,
      systemPrompt: preset.llmSystemPrompt,
      temperature: preset.temperature,
      maxTokens: preset.maxTokens
    }
  };
}

function pickCharacterPhotorealFields(
  character: SimulationCharacterDraft | undefined
): Pick<SimulationCharacterDraft, "photorealPrompt" | "photorealNegativePrompt" | "profileImages"> {
  if (!character) {
    return {};
  }
  return {
    ...("photorealPrompt" in character ? { photorealPrompt: character.photorealPrompt } : {}),
    ...("photorealNegativePrompt" in character ? { photorealNegativePrompt: character.photorealNegativePrompt } : {}),
    ...("profileImages" in character ? { profileImages: character.profileImages } : {})
  };
}

function createRelationshipMapPresetPrompt(promptMode: SimulationPromptMode): string {
  const shared =
    "관계도/상태창은 오른쪽 관계도 탭에서 보여 줄 compact state source다. assistant_text에는 긴 상태창을 반복 출력하지 말고, 변화가 있을 때만 memory_events에 저장한다. 인물의 현재 위치, 감정, 체력/컨디션, 착용/소지품, 목표, 관계 변화는 memory_kind='state' 또는 'relationship'을 우선 사용한다. 의상 변화나 장면상 의상이 확정되면 state_type='Wearing'에 NovelAI-style English outfit tags를 저장하고, 표정/컨디션/자세/소지품처럼 이미지와 반응 일관성에 필요한 캐릭터별 상태 태그는 state_type='StatusTags'에 저장한다. 기존 의상이 찢어짐/젖음/오염/헐거워짐처럼 변형될 때는 police uniform, navy short dress, mini skirt 같은 베이스 의상 태그를 유지하고 torn uniform 같은 상태 태그를 덧붙인다. 자세, 현재 행동, 상호작용, 전체 상황/단계, 소지품, 카메라/조명/장면 구도가 이미지 일관성에 중요하면 state_type='PoseTags', 'ActionTags', 'InteractionTags', 'InteractionPhaseTags', 'HeldItemTags', 'SceneTags', 'ScenePhaseTags', 'CompositionTags', 'CameraTags', 'LightingTags'에 comma-separated English NAI tags로 저장한다. 너무 세세한 부위별 태그를 매번 쌓기보다 현재 상황을 복원할 수 있는 3-8개의 compact phase/state tags를 우선한다. actor_id, actor_name, target_id, state_type, state_value를 알 수 있으면 반드시 채운다. 기존 상태와 같은 값은 반복하지 않는다.";

  if (promptMode === "one_on_one") {
    return `${shared}\n1:1 진행에서는 신뢰, 호감, 거리감, 약속, 상처, 선호처럼 관계 이해에 필요한 작은 변화도 relationship 또는 observation으로 남긴다.`;
  }

  if (promptMode === "simulation") {
    return `${shared}\n시뮬레이션 진행에서는 시간, 장소, 자원, 컨디션, 위험, 미해결 목표, 소속/역할 변화도 state 또는 goal로 남겨 장기 진행의 상태창을 토큰 적게 복원할 수 있게 한다.`;
  }

  return `${shared}\n기본 진행에서는 캐릭터가 다음 장면에서 일관되게 반응하는 데 필요한 상태와 관계 변화만 선별해 저장한다.`;
}

function createDefaultDraftImageScenePresets(now = new Date().toISOString(), promptMode: SimulationPromptMode = "basic"): ImageSceneTagPreset[] {
  const common = [
    {
      keyword: "classroom",
      tags: ["classroom", "indoors", "desk", "chair", "window", "daylight", "school interior"],
      note: "교실 내부 기본 장면. 캐릭터 외형/복장 태그는 제외.",
      children: [
        {
          id: "draft_scene_preset_classroom_window",
          keyword: "window seat",
          tags: ["window", "sunlight", "desk", "curtain", "classroom"],
          note: "창가 자리/측면광 장면.",
          enabled: true,
          priority: 76,
          updatedAt: now,
          children: []
        }
      ]
    },
    {
      keyword: "hallway",
      tags: ["school hallway", "corridor", "indoors", "locker", "fluorescent light", "depth of field"],
      note: "복도 이동/대기 장면. 인물 태그 없이 공간과 조명만 유지.",
      children: []
    },
    {
      keyword: "night street",
      tags: ["night", "street", "city lights", "wet pavement", "street lamp", "reflection", "cinematic lighting"],
      note: "야간 외부 장면. 날씨/조명/배경 중심.",
      children: [
        {
          id: "draft_scene_preset_night_street_alley",
          keyword: "alley",
          tags: ["narrow alley", "neon sign", "wet pavement", "mist", "backlight"],
          note: "골목/추적/대기 장면.",
          enabled: true,
          priority: 72,
          updatedAt: now,
          children: []
        }
      ]
    }
  ];
  const simulationOnly =
    promptMode === "simulation"
      ? [
          {
            keyword: "operation room",
            tags: ["control room", "monitor", "desk", "dim light", "blue lighting", "equipment", "tense atmosphere"],
            note: "작전/상태 확인 장면. 캐릭터 태그는 후속 cue에서 따로 붙인다.",
            children: []
          }
        ]
      : [];

  return ensureDraftImageScenePresetIds(
    [...common, ...simulationOnly].map((preset, index) => ({
      id: `draft_scene_preset_${index + 1}`,
      simulationId: "draft_simulation",
      keyword: preset.keyword,
      tags: preset.tags,
      note: preset.note,
      enabled: true,
      priority: 80 - index * 4,
      updatedAt: now,
      children: preset.children
    }))
  );
}

function cloneImageScenePresetNodes(nodes: ImageSceneTagPresetNode[] | undefined): ImageSceneTagPresetNode[] {
  return (nodes ?? []).map((node) => ({
    ...node,
    tags: [...node.tags],
    exampleFiles: node.exampleFiles ? node.exampleFiles.map((file) => ({ ...file, prompts: [...file.prompts] })) : undefined,
    children: cloneImageScenePresetNodes(node.children)
  }));
}

function ensureDraftImageScenePresetIds(presets: ImageSceneTagPreset[]): ImageSceneTagPreset[] {
  const seen = new Set<string>();
  return presets.map((preset, index) => ({
    ...preset,
    id: reserveImageScenePresetId(preset.id, `draft_scene_preset_${index + 1}`, seen),
    simulationId: preset.simulationId || "draft_simulation",
    children: ensureDraftImageScenePresetNodeIds(preset.children, seen, `${index + 1}`)
  }));
}

function ensureDraftImageScenePresetNodeIds(
  nodes: ImageSceneTagPresetNode[] | undefined,
  seen: Set<string>,
  path: string
): ImageSceneTagPresetNode[] {
  return (nodes ?? []).map((node, index) => {
    const nodePath = `${path}_${index + 1}`;
    return {
      ...node,
      id: reserveImageScenePresetId(node.id, `draft_scene_preset_child_${nodePath}`, seen),
      children: ensureDraftImageScenePresetNodeIds(node.children, seen, nodePath)
    };
  });
}

function normalizeDraftImageScenePresetNodes(
  nodes: ImageSceneTagPresetNode[] | undefined,
  updatedAt: string,
  seen = new Set<string>(),
  path = "child"
): ImageSceneTagPresetNode[] {
  return (nodes ?? []).map((node, index) => ({
    ...node,
    id: reserveImageScenePresetId(
      node.id && !node.id.startsWith("draft_") ? node.id : undefined,
      `scene_preset_child_${path}_${index + 1}`,
      seen
    ),
    keyword: node.keyword.trim() || `scene-${index + 1}`,
    tags: node.tags.map((tag) => tag.trim()).filter(Boolean),
    note: node.note.trim(),
    exampleFiles: (node.exampleFiles ?? [])
      .map((file) => ({
        ...file,
        label: file.label.trim(),
        prompts: file.prompts.map((line) => line.trim()).filter(Boolean)
      }))
      .filter((file) => file.label || file.prompts.length > 0),
    enabled: node.enabled,
    priority: Math.min(120, Math.max(0, Number(node.priority) || 70)),
    updatedAt,
    children: normalizeDraftImageScenePresetNodes(node.children, updatedAt, seen, `${path}_${index + 1}`)
  }));
}

function reserveImageScenePresetId(candidateId: string | undefined, fallbackPrefix: string, seen: Set<string>): string {
  let id = candidateId?.trim();
  if (!id || seen.has(id)) {
    do {
      id = createId(fallbackPrefix);
    } while (seen.has(id));
  }
  seen.add(id);
  return id;
}

function createDraftImageScenePresetNode(now: string, index: number): ImageSceneTagPresetNode {
  return {
    id: createId("draft_scene_preset"),
    keyword: `sub keyword ${index}`,
    tags: ["close-up", "soft light"],
    note: "",
    enabled: true,
    priority: 68,
    updatedAt: now,
    children: []
  };
}

function updateImageScenePresetNodes<T extends ImageSceneTagPresetNode>(
  nodes: T[],
  presetId: string,
  patch: Partial<ImageSceneTagPresetNode>,
  updatedAt: string
): T[] {
  return nodes.map((node) => {
    if (node.id === presetId) {
      return {
        ...node,
        ...patch,
        updatedAt,
        children: patch.children ?? node.children ?? []
      } as T;
    }

    return {
      ...node,
      children: updateImageScenePresetNodes(node.children ?? [], presetId, patch, updatedAt)
    } as T;
  });
}

function appendImageScenePresetChild<T extends ImageSceneTagPresetNode>(
  nodes: T[],
  parentId: string,
  child: ImageSceneTagPresetNode,
  updatedAt: string
): T[] {
  return nodes.map((node) => {
    if (node.id === parentId) {
      return {
        ...node,
        updatedAt,
        children: [...(node.children ?? []), child]
      } as T;
    }

    return {
      ...node,
      children: appendImageScenePresetChild(node.children ?? [], parentId, child, updatedAt)
    } as T;
  });
}

function moveImageScenePresetNodes(
  presets: ImageSceneTagPreset[],
  sourceId: string,
  targetId: string,
  position: ImageScenePresetDropPosition,
  updatedAt: string,
  simulationId = "draft_simulation"
): ImageSceneTagPreset[] {
  if (sourceId === targetId || isImageScenePresetDescendant(presets, sourceId, targetId)) {
    return presets;
  }

  const removal = removeImageScenePresetNode(presets, sourceId);
  if (!removal.removed) {
    return presets;
  }

  const insertion = insertImageScenePresetNode(removal.nodes, removal.removed, targetId, position, updatedAt, simulationId);
  return insertion.inserted ? insertion.nodes : presets;
}

function removeImageScenePresetNode<T extends ImageSceneTagPresetNode>(
  nodes: T[],
  sourceId: string
): { nodes: T[]; removed?: ImageSceneTagPresetNode } {
  let removed: ImageSceneTagPresetNode | undefined;
  const nextNodes: T[] = [];

  for (const node of nodes) {
    if (node.id === sourceId) {
      removed = node;
      continue;
    }

    const childRemoval = removeImageScenePresetNode(node.children ?? [], sourceId);
    if (childRemoval.removed) {
      removed = childRemoval.removed;
      nextNodes.push({
        ...node,
        children: childRemoval.nodes
      } as T);
    } else {
      nextNodes.push(node);
    }
  }

  return { nodes: nextNodes, removed };
}

function insertImageScenePresetNode<T extends ImageSceneTagPresetNode>(
  nodes: T[],
  movingNode: ImageSceneTagPresetNode,
  targetId: string,
  position: ImageScenePresetDropPosition,
  updatedAt: string,
  rootSimulationId?: string
): { nodes: T[]; inserted: boolean } {
  const nextNodes: T[] = [];
  let inserted = false;

  for (const node of nodes) {
    if (node.id === targetId && position === "before") {
      nextNodes.push(createImageScenePresetNodeForLevel(movingNode, rootSimulationId) as T);
      inserted = true;
    }

    if (node.id === targetId && position === "inside") {
      nextNodes.push({
        ...node,
        updatedAt,
        children: [...(node.children ?? []), createImageScenePresetChildNode(movingNode)]
      } as T);
      inserted = true;
      continue;
    }

    const childInsertion: { nodes: ImageSceneTagPresetNode[]; inserted: boolean } | undefined =
      inserted || node.id === targetId
        ? undefined
        : insertImageScenePresetNode(node.children ?? [], movingNode, targetId, position, updatedAt);
    nextNodes.push(
      childInsertion?.inserted
        ? ({
            ...node,
            updatedAt,
            children: childInsertion.nodes
          } as T)
        : node
    );
    inserted = inserted || Boolean(childInsertion?.inserted);

    if (node.id === targetId && position === "after") {
      nextNodes.push(createImageScenePresetNodeForLevel(movingNode, rootSimulationId) as T);
      inserted = true;
    }
  }

  return { nodes: nextNodes, inserted };
}

function createImageScenePresetNodeForLevel(node: ImageSceneTagPresetNode, rootSimulationId?: string): ImageSceneTagPresetNode {
  return rootSimulationId ? createImageScenePresetRootNode(node, rootSimulationId) : createImageScenePresetChildNode(node);
}

function createImageScenePresetRootNode(node: ImageSceneTagPresetNode, simulationId: string): ImageSceneTagPreset {
  return {
    ...node,
    simulationId: "simulationId" in node && typeof node.simulationId === "string" ? node.simulationId : simulationId
  };
}

function createImageScenePresetChildNode(node: ImageSceneTagPresetNode): ImageSceneTagPresetNode {
  const { simulationId, ...childNode } = node as ImageSceneTagPresetNode & { simulationId?: string };
  void simulationId;
  return childNode;
}

function deleteImageScenePresetNode<T extends ImageSceneTagPresetNode>(nodes: T[], presetId: string): T[] {
  return nodes
    .filter((node) => node.id !== presetId)
    .map((node) => ({
      ...node,
      children: deleteImageScenePresetNode(node.children ?? [], presetId)
    }) as T);
}

function collectImageScenePresetNodeIds(nodes: ImageSceneTagPresetNode[]): string[] {
  return nodes.flatMap((node) => [node.id, ...collectImageScenePresetNodeIds(node.children ?? [])]);
}

function createImageScenePresetDescendantMap(nodes: ImageSceneTagPresetNode[]): Map<string, Set<string>> {
  const descendantMap = new Map<string, Set<string>>();
  const visit = (node: ImageSceneTagPresetNode): string[] => {
    const descendants = (node.children ?? []).flatMap((child) => [child.id, ...visit(child)]);
    descendantMap.set(node.id, new Set(descendants));
    return descendants;
  };
  nodes.forEach(visit);
  return descendantMap;
}

function isImageScenePresetDescendant(nodes: ImageSceneTagPresetNode[], sourceId: string, targetId: string): boolean {
  return Boolean(createImageScenePresetDescendantMap(nodes).get(sourceId)?.has(targetId));
}

function countEnabledImageScenePresetNodes(nodes: ImageSceneTagPresetNode[]): number {
  return nodes.reduce(
    (count, node) => count + (node.enabled ? 1 + countEnabledImageScenePresetNodes(node.children ?? []) : 0),
    0
  );
}

function createDraftFromState(source: AppState): SimulationDraft {
  const mainModule = source.modules.find((module) => module.kind === "main_prompt") ?? source.modules[0];
  const worldModule = source.modules.find((module) => module.kind === "world_lore");
  const firstCharacter = source.characters[0];
  const firstVisual = firstCharacter ? source.visualProfiles.find((profile) => profile.characterId === firstCharacter.id) : undefined;

  return {
    promptMode: source.simulation.promptMode ?? "custom",
    contentRating: source.simulation.contentRating ?? "general",
    title: source.simulation.title,
    description: source.simulation.description,
    mainPrompt: mainModule?.body ?? source.simulation.description,
    characterName: firstCharacter?.name ?? "Main Character",
    characterRole: firstCharacter?.role ?? "Simulation lead",
    characterSummary: firstCharacter?.summary ?? "",
    characterRelationship: firstCharacter?.relationship ?? "",
    characterMood: firstCharacter?.currentMood ?? "",
    worldLore: worldModule?.body ?? "",
    startSituationPrompt: source.messages.find((message) => message.role === "assistant")?.content ?? "",
    visualPrompt: firstVisual?.positivePrompt ?? "",
    negativeVisualPrompt: firstVisual?.negativePrompt ?? "",
    defaultOutfitPrompt: firstVisual?.defaultOutfitPrompt ?? defaultOutfitPrompt,
    outfitPrompts: firstVisual?.outfitPrompts ?? defaultOutfitPrompts,
    expressionPrompts: firstVisual?.expressionPrompts ?? defaultExpressionPrompts,
    realtimeImageEnabled: source.simulation.realtimeImageEnabled,
    imageScenePresets: ensureDraftImageScenePresetIds(
      (source.imageScenePresets ?? []).map((preset) => ({
        ...preset,
        tags: [...preset.tags],
        children: cloneImageScenePresetNodes(preset.children)
      }))
    ),
    characters: source.characters.map((character) => {
      const visual = source.visualProfiles.find((profile) => profile.characterId === character.id);
      return {
        id: character.id,
        name: character.name,
        role: character.role,
        summary: character.summary,
        relationship: character.relationship,
        currentMood: character.currentMood,
        visualPrompt: visual?.positivePrompt ?? "",
        negativeVisualPrompt: visual?.negativePrompt ?? "",
        defaultOutfitPrompt: visual?.defaultOutfitPrompt ?? defaultOutfitPrompt,
        outfitPrompts: visual?.outfitPrompts ?? defaultOutfitPrompts,
        expressionPrompts: visual?.expressionPrompts ?? defaultExpressionPrompts,
        defaultSafetyLevel: visual?.defaultSafetyLevel ?? "safe",
        // Always written, undefined included: updateStateFromDraft reads a PRESENT-but-undefined key as "cleared
        // in the editor". Without these the builder silently dropped a character's photoreal appearance and its
        // picked profile images on every edit. Profile images go through the shareable whitelist so a
        // candidate's in-memory bytes can never ride along into the draft.
        photorealPrompt: visual?.photorealPrompt,
        photorealNegativePrompt: visual?.photorealNegativePrompt,
        profileImages: toShareableCharacterProfileImages(visual?.profileImages)
      };
    }),
    modules: source.modules.map((module) => ({ ...module })),
    imageProfile: { ...source.imageProfile, backendRouting: resolveImageBackendRouting(source.imageProfile) },
    neuralMap: { ...source.neuralMap },
    relationshipMap: { ...source.relationshipMap },
    llm: toShareableLlmSettings(source.llm),
    novelAi: toShareableNovelAiSettings(source.novelAi),
    photoreal: toShareablePhotorealSettings(source.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS)
  };
}

function normalizeBuilderDraft(draft: SimulationDraft): SimulationDraft {
  return {
    ...draft,
    relationshipMap: {
      ...draft.relationshipMap,
      parameters: ensureUniqueRelationshipParameterIds(draft.relationshipMap.parameters)
    },
    imageScenePresets: ensureDraftImageScenePresetIds(draft.imageScenePresets ?? []),
    // The builder reads these unguarded (routing card, photoreal section), and a draft handed in by an older build
    // has neither. Normalizing once on entry is safe: the builder exposes no numeric photoreal field for the clamps
    // in normalizePhotorealSettings to fight with.
    imageProfile: { ...draft.imageProfile, backendRouting: resolveImageBackendRouting(draft.imageProfile) },
    photoreal: normalizePhotorealSettings(draft.photoreal)
  };
}

// Duplicate parameter ids make the editor treat several rows as one: updateRelationshipStatusParameter
// matches by id, so editing one row's title/priority writes to every row sharing that id (and React
// reuses DOM for the duplicate key). Persisted data can accumulate collisions across reloads, so repair
// them whenever a draft enters the builder.
function ensureUniqueRelationshipParameterIds(
  parameters: RelationshipStatusParameter[]
): RelationshipStatusParameter[] {
  const seen = new Set<string>();
  return parameters.map((parameter) => {
    if (!parameter.id || seen.has(parameter.id)) {
      const id = createId("rel_param");
      seen.add(id);
      return { ...parameter, id };
    }
    seen.add(parameter.id);
    return parameter;
  });
}

function updateStateFromDraft(existing: AppState, draft: SimulationDraft): AppState {
  const now = new Date().toISOString();
  const fallbackCharacter: SimulationCharacterDraft = {
    id: existing.characters[0]?.id ?? "character_main",
    name: draft.characterName.trim() || "Main Character",
    role: draft.characterRole || "Simulation lead",
    summary: draft.characterSummary,
    relationship: draft.characterRelationship,
    currentMood: draft.characterMood,
    visualPrompt: draft.visualPrompt,
    negativeVisualPrompt: draft.negativeVisualPrompt,
    defaultOutfitPrompt: draft.defaultOutfitPrompt ?? defaultOutfitPrompt,
    outfitPrompts: defaultOutfitPrompts,
    expressionPrompts: defaultExpressionPrompts,
    defaultSafetyLevel: "safe"
  };
  const draftCharacters = draft.characters.length > 0 ? draft.characters : [fallbackCharacter];
  const characters = draftCharacters.map((character) => ({
    id: character.id,
    simulationId: existing.simulation.id,
    name: character.name || "Main Character",
    role: character.role || "Simulation lead",
    summary: character.summary,
    relationship: character.relationship || "",
    currentMood: character.currentMood || ""
  }));
  const characterIds = new Set(characters.map((character) => character.id));
  // safety_policy modules stay stored regardless of rating. Adult mode already filters them out at
  // read time on every turn path, so dropping them here only destroyed the creator's authored modules
  // in a way that lowering the rating back to general could not undo.
  const modules = draft.modules.map((module) => ({
    ...module,
    simulationId: existing.simulation.id,
    characterId: module.characterId && characterIds.has(module.characterId) ? module.characterId : undefined,
    updatedAt: now
  }));
  const visualProfiles = draftCharacters.map((character, index) => {
    const previous = existing.visualProfiles.find((profile) => profile.characterId === character.id);
    return {
      id: previous?.id ?? `visual_${index}_default`,
      simulationId: existing.simulation.id,
      characterId: character.id,
      displayName: character.name || "Character",
      positivePrompt: character.visualPrompt,
      negativePrompt: character.negativeVisualPrompt || "low quality, bad anatomy, blurry, watermark",
      defaultOutfitPrompt: character.defaultOutfitPrompt ?? previous?.defaultOutfitPrompt ?? defaultOutfitPrompt,
      outfitPrompts: character.outfitPrompts ?? previous?.outfitPrompts ?? defaultOutfitPrompts,
      expressionPrompts: character.expressionPrompts ?? previous?.expressionPrompts ?? defaultExpressionPrompts,
      referenceImageAssetIds: previous?.referenceImageAssetIds ?? [],
      defaultSafetyLevel: character.defaultSafetyLevel,
      // The draft is the editor's copy and wins, including an explicit undefined (a profile the creator removed).
      // `previous` only fills in for a draft from a builder that never carried the keys at all; hydrateState below
      // re-validates the profile images either way.
      photorealPrompt: "photorealPrompt" in character ? character.photorealPrompt : previous?.photorealPrompt,
      photorealNegativePrompt: "photorealNegativePrompt" in character ? character.photorealNegativePrompt : previous?.photorealNegativePrompt,
      profileImages: "profileImages" in character ? character.profileImages : previous?.profileImages
    };
  });
  const scenePresetSeenIds = new Set<string>();
  const imageScenePresets: ImageSceneTagPreset[] = (draft.imageScenePresets ?? []).map((preset, index) => ({
    ...preset,
    id: reserveImageScenePresetId(
      preset.id && !preset.id.startsWith("draft_") ? preset.id : undefined,
      `scene_preset_${index + 1}`,
      scenePresetSeenIds
    ),
    simulationId: existing.simulation.id,
    keyword: preset.keyword.trim() || `scene-${index + 1}`,
    tags: preset.tags.map((tag) => tag.trim()).filter(Boolean),
    note: preset.note.trim(),
    enabled: preset.enabled,
    priority: Math.min(120, Math.max(0, Number(preset.priority) || 70)),
    updatedAt: now,
    children: normalizeDraftImageScenePresetNodes(preset.children, now, scenePresetSeenIds, `${index + 1}`)
  }));
  const openingContent = draft.startSituationPrompt.trim();
  const shouldUpdateOpeningMessage =
    Boolean(openingContent) && existing.messages.filter((message) => message.role === "user").length === 0;
  const messages = shouldUpdateOpeningMessage
    ? existing.messages.some((message) => message.id === "msg_welcome")
      ? existing.messages.map((message) => (message.id === "msg_welcome" ? { ...message, content: openingContent } : message))
      : [
          {
            id: "msg_welcome",
            simulationId: existing.simulation.id,
            sessionId: existing.simulation.activeSessionId,
            role: "assistant" as const,
            content: openingContent,
            createdAt: now,
            referencedNodeIds: ["module_main", "module_character"],
            imageAssetIds: []
          },
          ...existing.messages
        ]
    : existing.messages;

  return hydrateState({
    ...existing,
    simulation: {
      ...existing.simulation,
      title: draft.title.trim() || existing.simulation.title,
      description: draft.description.trim() || existing.simulation.description,
      promptMode: draft.promptMode,
      contentRating: draft.contentRating,
      realtimeImageEnabled: draft.realtimeImageEnabled,
      defaultChatModelProfile: getPromptModeProfile(draft.promptMode),
      updatedAt: now
    },
    modules,
    characters,
    visualProfiles,
    imageScenePresets,
    messages,
    imageProfile: {
      ...existing.imageProfile,
      ...draft.imageProfile,
      simulationId: existing.simulation.id,
      safetyLevel: draft.imageProfile.safetyLevel
    },
    neuralMap: draft.neuralMap,
    relationshipMap: {
      ...existing.relationshipMap,
      ...draft.relationshipMap,
      updatedAt: draft.relationshipMap.updatedAt || now
    },
    llm: {
      ...toShareableLlmSettings(draft.llm),
      apiKey: existing.llm.apiKey,
      registrationStatus: existing.llm.registrationStatus,
      verifiedAt: existing.llm.verifiedAt,
      verificationMessage: existing.llm.verificationMessage
    },
    novelAi: {
      ...toShareableNovelAiSettings(draft.novelAi),
      apiKey: existing.novelAi.apiKey,
      registrationStatus: existing.novelAi.registrationStatus,
      verifiedAt: existing.novelAi.verifiedAt,
      verificationMessage: existing.novelAi.verificationMessage,
      subscriptionTier: existing.novelAi.subscriptionTier
    },
    // Same split as NovelAI: the builder owns the configuration, this machine's last connection check survives the
    // edit. A draft from a builder that predates photoreal keeps the simulation's current settings.
    photoreal: {
      ...toShareablePhotorealSettings(draft.photoreal ?? existing.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS),
      registrationStatus: existing.photoreal?.registrationStatus ?? "idle",
      verifiedAt: existing.photoreal?.verifiedAt,
      verificationMessage: existing.photoreal?.verificationMessage
    },
    selectedModuleId: modules[0]?.id,
    selectedContextPackId: existing.selectedContextPackId
  });
}

function shouldSyncModuleWithDraftField(
  module: PromptModule,
  field: "mainPrompt" | "characterSummary" | "worldLore" | "visualPrompt",
  firstCharacterId?: string
): boolean {
  if (field === "mainPrompt") {
    return module.kind === "main_prompt" || module.id === "draft_main";
  }
  if (field === "characterSummary") {
    return module.kind === "character_prompt" && (!firstCharacterId || module.characterId === firstCharacterId || module.id === "draft_character");
  }
  if (field === "worldLore") {
    return module.kind === "world_lore" || module.id === "draft_world";
  }

  return module.kind === "image_prompt_profile" || module.id === "draft_visual";
}

function createReplySuggestions(state: AppState): string[] {
  const leadCharacter = state.characters[0]?.name ?? "상대";
  const latestMemoryTag = state.memoryEvents.at(-1)?.tags[0];
  const latestAssistantLine = state.messages
    .slice()
    .reverse()
    .find((message) => message.role === "assistant")?.content;

  return [
    `${leadCharacter}에게 방금 장면에서 가장 중요한 단서를 묻는다.`,
    latestAssistantLine ? "방금 나온 대사에 맞춰 감정을 드러내며 대답한다." : "주변 상황을 관찰하고 먼저 말을 건다.",
    latestMemoryTag ? `최근 기억(${latestMemoryTag})을 떠올리며 다음 행동을 정한다.` : "현재 목표를 짧게 정리하고 다음 행동을 선택한다."
  ];
}

function createTurnSubmissionText(draft: string): string {
  return draft.trim() || AUTO_CONTINUE_TURN_TEXT;
}

function insertActionNotation(
  textarea: HTMLTextAreaElement | null,
  draft: string,
  onDraftChange: (value: string) => void
): void {
  const selectionStart = textarea?.selectionStart ?? draft.length;
  const selectionEnd = textarea?.selectionEnd ?? draft.length;
  const selectedText = draft.slice(selectionStart, selectionEnd);
  const nextDraft = `${draft.slice(0, selectionStart)}*(${selectedText})*${draft.slice(selectionEnd)}`;
  const nextCursor = selectionStart + 2 + selectedText.length;

  onDraftChange(nextDraft);
  runAfterNextPaint(() => {
    textarea?.focus();
    textarea?.setSelectionRange(nextCursor, nextCursor);
  });
}

function createHistoryItems(state: AppState): Array<{ id: string; label: string; detail: string; active: boolean; canDelete: boolean }> {
  const runs = state.progressRuns.length > 0
    ? state.progressRuns
    : [
        {
          id: state.activeProgressRunId,
          title: "진행 1",
          updatedAt: state.simulation.updatedAt,
          messages: state.messages
        }
      ];
  const canDelete = runs.length > 1;

  return runs.map((run, index) => {
    const userMessageCount = run.messages.filter((message) => message.role === "user").length;
    const latestUserMessage = run.messages
      .slice()
      .reverse()
      .find((message) => message.role === "user");
    const latestAt = new Date(run.updatedAt).toLocaleDateString("ko-KR", {
      month: "short",
      day: "numeric"
    });

    return {
      id: run.id,
      label: run.title || `진행 ${index + 1}`,
      detail: latestUserMessage
        ? `${userMessageCount}개 선택 · ${latestUserMessage.content.slice(0, 18)}`
        : `첫 장면 · ${latestAt}`,
      active: run.id === state.activeProgressRunId,
      canDelete
    };
  });
}

type AutoResetAgentSessionDecision = {
  shouldReset: boolean;
  status: "stable" | "soon" | "handoff";
  reason: string;
  activeMessages: number;
  activeUserTurns: number;
  activeCharCount: number;
  degradedTraceCount: number;
  pressure: number;
};

function createAutoResetAgentSessionDecision(state: AppState, nextUserText = ""): AutoResetAgentSessionDecision {
  const activeMessages = state.messages.filter(
    (message) => message.sessionId === state.simulation.activeSessionId && message.role !== "system"
  );
  const activeUserTurns = activeMessages.filter((message) => message.role === "user").length;
  const activeCharCount =
    activeMessages.reduce((sum, message) => sum + message.content.length, 0) + nextUserText.length;
  const activeTraceIds = new Set(
    state.turnTraces
      .filter((trace) => trace.sessionId === state.simulation.activeSessionId)
      .slice(-4)
      .map((trace) => trace.sidecarTraceId)
  );
  const degradedTraceCount = state.sidecarTraces
    .filter((trace) => activeTraceIds.has(trace.id))
    .filter((trace) => trace.source !== "llm" || trace.status !== "parsed")
    .length;
  const latestContinuity = state.continuityChecks.at(-1);
  const hasCurrentContinuityWarning =
    latestContinuity?.nextSessionId === state.simulation.activeSessionId && latestContinuity.status === "warning";
  const messagePressure = activeMessages.length / AUTO_RESET_ACTIVE_SESSION_MESSAGE_LIMIT;
  const turnPressure = activeUserTurns / AUTO_RESET_ACTIVE_SESSION_TURN_LIMIT;
  const charPressure = activeCharCount / AUTO_RESET_ACTIVE_SESSION_CHAR_LIMIT;
  const pressure = Math.max(messagePressure, turnPressure, charPressure);
  const hasEnoughSessionBody =
    activeUserTurns >= AUTO_RESET_MIN_ACTIVE_USER_TURNS ||
    activeCharCount >= AUTO_RESET_ACTIVE_SESSION_CHAR_LIMIT;
  const resetReasons = [
    activeUserTurns >= AUTO_RESET_ACTIVE_SESSION_TURN_LIMIT
      ? `활성 세션 ${activeUserTurns}턴`
      : undefined,
    activeMessages.length >= AUTO_RESET_ACTIVE_SESSION_MESSAGE_LIMIT
      ? `활성 메시지 ${activeMessages.length}개`
      : undefined,
    activeCharCount >= AUTO_RESET_ACTIVE_SESSION_CHAR_LIMIT
      ? `활성 문맥 ${activeCharCount.toLocaleString("ko-KR")}자`
      : undefined,
    degradedTraceCount >= AUTO_RESET_DEGRADED_TRACE_LIMIT &&
    activeUserTurns >= AUTO_RESET_DEGRADED_TRACE_MIN_TURNS
      ? `최근 sidecar 불안정 ${degradedTraceCount}회`
      : undefined,
    hasCurrentContinuityWarning &&
    activeUserTurns >= AUTO_RESET_MIN_ACTIVE_USER_TURNS &&
    pressure >= AUTO_RESET_PRESSURE_NOTICE_THRESHOLD
      ? "연속성 경고 후 문맥 압력 증가"
      : undefined
  ].filter((reason): reason is string => Boolean(reason));
  const shouldReset = hasEnoughSessionBody && resetReasons.length > 0;
  const status = shouldReset
    ? "handoff"
    : pressure >= AUTO_RESET_PRESSURE_NOTICE_THRESHOLD
      ? "soon"
      : "stable";

  return {
    shouldReset,
    status,
    reason: resetReasons[0] ?? `${activeUserTurns}/${AUTO_RESET_ACTIVE_SESSION_TURN_LIMIT}턴 · ${Math.round(pressure * 100)}%`,
    activeMessages: activeMessages.length,
    activeUserTurns,
    activeCharCount,
    degradedTraceCount,
    pressure
  };
}

function formatAutoResetStatus(decision: AutoResetAgentSessionDecision): string {
  if (decision.status === "handoff") {
    return "자동 handoff 준비";
  }

  if (decision.status === "soon") {
    return "자동 handoff 임박";
  }

  return "자동 handoff 대기";
}

function formatAutoResetDetail(decision: AutoResetAgentSessionDecision): string {
  return `${decision.activeUserTurns}/${AUTO_RESET_ACTIVE_SESSION_TURN_LIMIT}턴 · ${Math.round(decision.pressure * 100)}%`;
}

function getPersonaCharacter(state: AppState): AppState["characters"][number] | undefined {
  const persona = state.userPersona;
  if (!persona?.enabled || persona.source !== "character" || !persona.characterId) {
    return undefined;
  }

  return state.characters.find((character) => character.id === persona.characterId);
}

function createPersonaPreview(persona: UserPersona, character?: AppState["characters"][number]): string {
  if (persona.source === "character" && character) {
    const lines = [
      `시점: ${character.name}`,
      `캐릭터 ID: ${character.id}`,
      character.role.trim() ? `역할: ${character.role.trim()}` : undefined,
      character.summary.trim() ? `배경: ${character.summary.trim()}` : undefined,
      character.relationship.trim() ? `관계: ${character.relationship.trim()}` : undefined,
      character.currentMood.trim() ? `현재 상태: ${character.currentMood.trim()}` : undefined,
      persona.goals.trim() ? `플레이 목표: ${persona.goals.trim()}` : undefined,
      persona.style.trim() ? `입력 방식: ${persona.style.trim()}` : undefined,
      persona.boundaries.trim() ? `경계: ${persona.boundaries.trim()}` : undefined
    ].filter((line): line is string => Boolean(line));

    return lines.join("\n");
  }

  const lines = [
    persona.name.trim() ? `호칭: ${persona.name.trim()}` : undefined,
    persona.role.trim() ? `역할: ${persona.role.trim()}` : undefined,
    persona.background.trim() ? `배경: ${persona.background.trim()}` : undefined,
    persona.goals.trim() ? `목표: ${persona.goals.trim()}` : undefined,
    persona.style.trim() ? `성향: ${persona.style.trim()}` : undefined,
    persona.boundaries.trim() ? `경계: ${persona.boundaries.trim()}` : undefined
  ].filter((line): line is string => Boolean(line));

  return lines.length > 0 ? lines.join("\n") : "페르소나가 아직 비어 있습니다.";
}

function getPersonaDisplayName(state: AppState): string {
  const character = getPersonaCharacter(state);
  if (character) {
    return character.name;
  }

  const persona = state.userPersona;
  return persona?.enabled && persona.name.trim() ? persona.name.trim() : "User";
}

function createNeuralMapNodeEditDraft(state: AppState, node: NeuralMapLiveNode): NeuralMapNodeEditDraft {
  const module = state.modules.find((candidate) => candidate.id === node.id);
  if (module) {
    return {
      nodeId: node.id,
      editableKind: "prompt_module",
      targetId: module.id,
      title: module.title,
      content: module.body,
      tags: module.activationTags.join(", "),
      importance: Math.min(1, Math.max(0, module.priority / 100)),
      tokenPolicy: module.tokenPolicy,
      priority: module.priority,
      enabled: module.enabled
    };
  }

  const memoryEvent = findMemoryEventForNeuralNode(state, node.id);
  if (memoryEvent) {
    return {
      nodeId: node.id,
      editableKind: "memory_event",
      targetId: memoryEvent.id,
      title: memoryEvent.tags[0] ? `${memoryEvent.tags[0]} memory` : "Simulation memory",
      content: memoryEvent.content,
      tags: memoryEvent.tags.join(", "),
      importance: memoryEvent.importance,
      tokenPolicy: "rag",
      priority: Math.round(memoryEvent.importance * 100),
      enabled: true
    };
  }

  const character = findCharacterForNeuralNode(state, node.id);
  if (character) {
    return {
      nodeId: node.id,
      editableKind: "character",
      targetId: character.id,
      title: character.name,
      content: [character.summary, character.relationship ? `관계: ${character.relationship}` : undefined, character.currentMood ? `현재 상태: ${character.currentMood}` : undefined]
        .filter(Boolean)
        .join("\n"),
      tags: ["character", character.name].filter(Boolean).join(", "),
      importance: node.importanceScore,
      tokenPolicy: "rag",
      priority: Math.round(node.importanceScore * 100),
      enabled: true
    };
  }

  if (node.id === `persona:${state.simulation.id}`) {
    const personaCharacter = getPersonaCharacter(state);
    return {
      nodeId: node.id,
      editableKind: "persona",
      targetId: state.simulation.id,
      title: personaCharacter?.name || state.userPersona.name || "사용자 페르소나",
      content: createPersonaPreview(state.userPersona, personaCharacter),
      tags: "persona, user-role",
      importance: node.importanceScore,
      tokenPolicy: "rag",
      priority: Math.round(node.importanceScore * 100),
      enabled: state.userPersona.enabled
    };
  }

  const contextPack = state.contextPacks.find((pack) => pack.id === node.id);
  if (contextPack) {
    return {
      nodeId: node.id,
      editableKind: "context_pack",
      targetId: contextPack.id,
      title: contextPack.source === "neuralmap" ? "NeuralMap Context Pack" : "Local Context Pack",
      content: contextPack.objective,
      tags: "context, pack",
      importance: node.importanceScore,
      tokenPolicy: "rag",
      priority: Math.round(node.importanceScore * 100),
      enabled: true
    };
  }

  return {
    nodeId: node.id,
    editableKind: "graph_document",
    targetId: node.id,
    title: node.title,
    content: node.summary,
    tags: [node.kind, node.type, "neural-edit"].filter(Boolean).join(", "),
    importance: node.importanceScore,
    tokenPolicy: "rag",
    priority: Math.round(node.importanceScore * 100),
    enabled: true
  };
}

function findMemoryEventForNeuralNode(state: AppState, nodeId: string) {
  return state.memoryEvents.find((event) => event.id === nodeId || event.neuralMapNodeId === nodeId);
}

function findCharacterForNeuralNode(state: AppState, nodeId: string) {
  const localMatch = nodeId.match(/^simulation:[^:]+:person:(.+)$/u)?.[1];
  return state.characters.find((character) => character.id === nodeId || character.id === localMatch);
}

function normalizeNeuralEditorTags(value: string): string[] {
  // Keep multi-word phrases intact (collapse internal whitespace to a single space) instead of
  // hyphenating them. Hyphenated phrases like "silver-key" never matched natural input ("silver key"),
  // which silently limited creators to single-word triggers — see selectRelevantModules phrase matching.
  return value
    .split(",")
    .map((tag) => tag.trim().toLowerCase().replace(/\s+/gu, " "))
    .filter(Boolean)
    .slice(0, 16);
}

function parseCharacterContent(content: string): { summary: string; relationship: string; currentMood: string } {
  const lines = content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const relationshipLine = lines.find((line) => line.startsWith("관계:"));
  const moodLine = lines.find((line) => line.startsWith("현재 상태:"));
  return {
    summary: lines.filter((line) => !line.startsWith("관계:") && !line.startsWith("현재 상태:")).join("\n") || content.trim(),
    relationship: relationshipLine?.replace(/^관계:\s*/u, "").trim() ?? "",
    currentMood: moodLine?.replace(/^현재 상태:\s*/u, "").trim() ?? ""
  };
}

function parsePersonaContent(content: string, fallback: UserPersona): UserPersona {
  const values = new Map(
    content
      .split("\n")
      .map((line) => line.trim())
      .map((line) => {
        const separatorIndex = line.indexOf(":");
        return separatorIndex > 0
          ? [line.slice(0, separatorIndex).trim(), line.slice(separatorIndex + 1).trim()] as const
          : undefined;
      })
      .filter((item): item is readonly [string, string] => Boolean(item))
  );

  return {
    ...fallback,
    name: values.get("호칭") ?? values.get("이름") ?? fallback.name,
    role: values.get("역할") ?? fallback.role,
    background: values.get("배경") ?? fallback.background,
    goals: values.get("목표") ?? fallback.goals,
    style: values.get("성향") ?? values.get("말투") ?? fallback.style,
    boundaries: values.get("경계") ?? fallback.boundaries
  };
}

function createEditedGraphPromptModule(state: AppState, node: NeuralMapLiveNode, draft: NeuralMapNodeEditDraft, now: string): PromptModule {
  return {
    id: draft.nodeId,
    simulationId: state.simulation.id,
    kind: "sub_prompt",
    title: draft.title.trim() || node.title || "NeuralMap 편집 노드",
    body: draft.content.trim() || node.summary,
    enabled: draft.enabled,
    priority: draft.priority,
    activationTags: normalizeNeuralEditorTags(draft.tags),
    tokenPolicy: draft.tokenPolicy,
    version: 1,
    updatedAt: now
  };
}

function applyNeuralMapNodeEdit(state: AppState, node: NeuralMapLiveNode, draft: NeuralMapNodeEditDraft): {
  state: AppState;
  syncTarget:
    | { kind: "prompt_module"; module: PromptModule }
    | { kind: "memory_event"; event: AppState["memoryEvents"][number] }
    | { kind: "graph_document"; document: { id: string; title: string; body: string; kind: string; tags: string[]; importance: number } };
} {
  const now = new Date().toISOString();
  const tags = normalizeNeuralEditorTags(draft.tags);
  const updateEvidence = (snippet: string, reason = draft.title) =>
    state.contextPacks.map((pack) => ({
      ...pack,
      objective: pack.id === draft.targetId && draft.editableKind === "context_pack" ? snippet : pack.objective,
      evidence: pack.evidence.map((item) =>
        item.nodeId === draft.nodeId || item.nodeId === draft.targetId
          ? {
              ...item,
              snippet,
              reason: reason || item.reason,
              score: Math.max(item.score, draft.importance)
            }
          : item
      )
    }));

  if (draft.editableKind === "prompt_module") {
    const module = state.modules.find((candidate) => candidate.id === draft.targetId) ?? createEditedGraphPromptModule(state, node, draft, now);
    const editedModule: PromptModule = {
      ...module,
      title: draft.title.trim() || module.title,
      body: draft.content.trim(),
      activationTags: tags.length > 0 ? tags : module.activationTags,
      priority: draft.priority,
      tokenPolicy: draft.tokenPolicy,
      enabled: draft.enabled,
      version: module.version + 1,
      updatedAt: now
    };
    return {
      state: {
        ...state,
        simulation: { ...state.simulation, updatedAt: now },
        modules: state.modules.some((candidate) => candidate.id === editedModule.id)
          ? state.modules.map((candidate) => (candidate.id === editedModule.id ? editedModule : candidate))
          : [...state.modules, editedModule],
        contextPacks: updateEvidence(editedModule.body, editedModule.title)
      },
      syncTarget: { kind: "prompt_module", module: editedModule }
    };
  }

  if (draft.editableKind === "memory_event") {
    const existing = findMemoryEventForNeuralNode(state, draft.nodeId);
    const editedEvent = {
      ...(existing ?? {
        id: createId("memory"),
        simulationId: state.simulation.id,
        sessionId: state.simulation.activeSessionId,
        createdAt: now
      }),
      content: draft.content.trim(),
      importance: draft.importance,
      tags,
      neuralMapNodeId: existing?.neuralMapNodeId ?? (draft.nodeId === existing?.id ? undefined : draft.nodeId)
    };
    return {
      state: {
        ...state,
        simulation: { ...state.simulation, updatedAt: now },
        memoryEvents: existing
          ? state.memoryEvents.map((event) => (event.id === existing.id ? editedEvent : event))
          : [...state.memoryEvents, editedEvent],
        contextPacks: updateEvidence(editedEvent.content, tags.join(", ") || "편집된 기억")
      },
      syncTarget: { kind: "memory_event", event: editedEvent }
    };
  }

  if (draft.editableKind === "character") {
    const parsed = parseCharacterContent(draft.content);
    const character = state.characters.find((candidate) => candidate.id === draft.targetId);
    const editedCharacters = state.characters.map((candidate) =>
      candidate.id === draft.targetId
        ? {
            ...candidate,
            name: draft.title.trim() || candidate.name,
            summary: parsed.summary,
            relationship: parsed.relationship || candidate.relationship,
            currentMood: parsed.currentMood || candidate.currentMood
          }
        : candidate
    );
    const body = [
      parsed.summary,
      parsed.relationship ? `관계: ${parsed.relationship}` : undefined,
      parsed.currentMood ? `현재 상태: ${parsed.currentMood}` : undefined
    ]
      .filter(Boolean)
      .join("\n");
    return {
      state: {
        ...state,
        simulation: { ...state.simulation, updatedAt: now },
        characters: editedCharacters,
        modules: state.modules.map((module) =>
          module.kind === "character_prompt" && module.characterId === draft.targetId
            ? {
                ...module,
                title: `캐릭터: ${draft.title.trim() || character?.name || module.title}`,
                body: parsed.summary,
                activationTags: normalizeNeuralEditorTags(`${draft.tags}, ${draft.title}`),
                version: module.version + 1,
                updatedAt: now
              }
            : module
        ),
        contextPacks: updateEvidence(body, draft.title)
      },
      syncTarget: {
        kind: "graph_document",
        document: {
          id: draft.nodeId,
          title: draft.title.trim() || character?.name || node.title,
          body,
          kind: "simulation_person",
          tags,
          importance: draft.importance
        }
      }
    };
  }

  if (draft.editableKind === "persona") {
    const persona = {
      ...parsePersonaContent(draft.content, state.userPersona),
      enabled: draft.enabled,
      updatedAt: now
    };
    const personaCharacter = persona.source === "character" && persona.characterId
      ? state.characters.find((candidate) => candidate.id === persona.characterId)
      : undefined;
    const body = createPersonaPreview(persona, personaCharacter);
    return {
      state: {
        ...state,
        simulation: { ...state.simulation, updatedAt: now },
        userPersona: persona,
        contextPacks: updateEvidence(body, draft.title)
      },
      syncTarget: {
        kind: "graph_document",
        document: {
          id: draft.nodeId,
          title: draft.title.trim() || persona.name || "사용자 페르소나",
          body,
          kind: "user_persona",
          tags,
          importance: draft.importance
        }
      }
    };
  }

  if (draft.editableKind === "context_pack") {
    return {
      state: {
        ...state,
        simulation: { ...state.simulation, updatedAt: now },
        contextPacks: updateEvidence(draft.content.trim(), draft.title)
      },
      syncTarget: {
        kind: "graph_document",
        document: {
          id: draft.nodeId,
          title: draft.title.trim() || node.title,
          body: draft.content.trim(),
          kind: "context_pack",
          tags,
          importance: draft.importance
        }
      }
    };
  }

  const document = {
    id: draft.nodeId,
    title: draft.title.trim() || node.title || "NeuralMap 기록",
    body: draft.content.trim() || node.summary,
    kind: draft.editableKind,
    tags,
    importance: draft.importance
  };
  return {
    state: {
      ...state,
      simulation: { ...state.simulation, updatedAt: now },
      contextPacks: updateEvidence(document.body, document.title)
    },
    syncTarget: { kind: "graph_document", document }
  };
}

function normalizeDynamicTextBlockKind(value: string): DynamicTextBlockKind | undefined {
  const normalized = value.trim().toLowerCase().replaceAll("_", "-");
  if (dynamicTextBlockKinds.includes(normalized as DynamicTextBlockKind)) {
    return normalized as DynamicTextBlockKind;
  }

  return dynamicTextBlockAliases[normalized];
}

/**
 * Recognises the bracket form ("[Status] ...") only.
 *
 * The bare-label form ("Status: ...", "Note - ...") used to be accepted too, which quietly reclassified any
 * narrative line that happened to begin with one of the block words followed by a colon — and because a
 * directive with no inline content swallows every following non-empty line, it could take a paragraph of
 * prose with it. With effect blocks now carrying real styling, that misfire is visible as well as wrong, so
 * only the explicit forms remain: [Kind], ::kind[...], ::kind ... ::, and ```kind fences.
 */
function parseLooseDynamicTextDirective(line: string): { kind: DynamicTextBlockKind; content?: string } | undefined {
  const bracketMatch = line.match(/^\s*\[([\p{L}\p{N}_-]+)\]\s*(.*)\s*$/u);
  if (bracketMatch) {
    const bracketKind = normalizeDynamicTextBlockKind(bracketMatch[1]);
    if (bracketKind) {
      return {
        kind: bracketKind,
        content: bracketMatch[2]?.trim()
      };
    }
  }

  return undefined;
}

function parseDynamicRichText(content: string): DynamicTextSegment[] {
  const lines = content.replace(/\r\n/gu, "\n").split("\n");
  const segments: DynamicTextSegment[] = [];
  const markdownLines: string[] = [];
  const flushMarkdown = () => {
    const markdown = normalizeLooseMarkdownTables(markdownLines.join("\n").trim());
    if (markdown) {
      segments.push({
        id: `markdown-${segments.length}`,
        kind: "markdown",
        content: markdown
      });
    }
    markdownLines.length = 0;
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    // Tolerate a space and any case after the opening backticks (e.g. "``` Status") so a status/dynamic
    // block the LLM wrote as a loosely-fenced code block still becomes one styled block instead of a raw
    // monospace code block whose rows render as separate lines.
    const fenceMatch = line.match(/^\s*```\s*([\p{L}\p{N}_-]+)\s*$/u);
    const fenceKind = fenceMatch ? normalizeDynamicTextBlockKind(fenceMatch[1]) : undefined;
    if (fenceKind) {
      flushMarkdown();
      const blockLines: string[] = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/u.test(lines[index])) {
        blockLines.push(lines[index]);
        index += 1;
      }
      segments.push({
        id: `${fenceKind}-${segments.length}`,
        kind: fenceKind,
        // Effect-block bodies are markdown too, and MarkdownStageText is now a pure renderer, so
        // they are normalized here rather than at render time (where it ran twice on the prose path).
        content: normalizeLooseMarkdownTables(blockLines.join("\n").trim())
      });
      continue;
    }

    const shortDirectiveMatch = line.match(/^\s*::([\p{L}\p{N}_-]+)\[(.*)\]\s*$/u);
    const shortDirectiveKind = shortDirectiveMatch ? normalizeDynamicTextBlockKind(shortDirectiveMatch[1]) : undefined;
    if (shortDirectiveKind) {
      flushMarkdown();
      segments.push({
        id: `${shortDirectiveKind}-${segments.length}`,
        kind: shortDirectiveKind,
        content: normalizeLooseMarkdownTables(shortDirectiveMatch?.[2].trim() ?? "")
      });
      continue;
    }

    const openDirectiveMatch = line.match(/^\s*::([\p{L}\p{N}_-]+)\s*$/u);
    const openDirectiveKind = openDirectiveMatch ? normalizeDynamicTextBlockKind(openDirectiveMatch[1]) : undefined;
    if (openDirectiveKind) {
      flushMarkdown();
      const blockLines: string[] = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== "::") {
        blockLines.push(lines[index]);
        index += 1;
      }
      segments.push({
        id: `${openDirectiveKind}-${segments.length}`,
        kind: openDirectiveKind,
        content: normalizeLooseMarkdownTables(blockLines.join("\n").trim())
      });
      continue;
    }

    const looseDirective = parseLooseDynamicTextDirective(line);
    if (looseDirective) {
      const nextLine = lines[index + 1];
      const opensMarkdownTable =
        !looseDirective.content && Boolean(nextLine && isMarkdownTableRow(nextLine) && !isMarkdownTableDelimiter(nextLine));
      if (!opensMarkdownTable) {
        flushMarkdown();
        const blockLines = looseDirective.content ? [looseDirective.content] : [];
        if (!looseDirective.content) {
          index += 1;
          while (index < lines.length && lines[index].trim()) {
            blockLines.push(lines[index]);
            index += 1;
          }
        }
        segments.push({
          id: `${looseDirective.kind}-${segments.length}`,
          kind: looseDirective.kind,
          content: normalizeLooseMarkdownTables(blockLines.join("\n").trim())
        });
        continue;
      }
    }

    markdownLines.push(line);
  }

  flushMarkdown();
  return segments.length > 0
    ? segments
    : [
        {
          id: "markdown-empty",
          kind: "markdown",
          content
        }
      ];
}

function findMessageAssets(message: AppState["messages"][number], assetsById: Map<string, ImageAsset>): ImageAsset[] {
  if (message.imageAssetIds.length === 0) {
    return EMPTY_IMAGE_ASSETS;
  }

  const assets = message.imageAssetIds
    .map((assetId) => assetsById.get(assetId))
    .filter((asset): asset is ImageAsset => Boolean(asset));
  if (assets.length === 0) {
    return EMPTY_IMAGE_ASSETS;
  }

  const generatedAssets = assets.filter((asset) => asset.source === "generated");
  if (generatedAssets.length > 0) {
    return generatedAssets;
  }

  return assets;
}

function upsertImageAssets(currentAssets: ImageAsset[], nextAssets: ImageAsset[]): ImageAsset[] {
  if (nextAssets.length === 0) {
    return currentAssets;
  }

  const nextById = new Map(nextAssets.map((asset) => [asset.id, asset]));
  const replaced = currentAssets.map((asset) => nextById.get(asset.id) ?? asset);
  const existingIds = new Set(replaced.map((asset) => asset.id));
  return [...replaced, ...nextAssets.filter((asset) => !existingIds.has(asset.id))];
}

function uniqueIds(ids: string[]): string[] {
  return Array.from(new Set(ids.filter(Boolean)));
}

function mergeGeneratedMessageImageAssetIds(existingIds: string[], currentAssets: ImageAsset[], nextAssets: ImageAsset[]): string[] {
  const generatedNextIds = nextAssets.filter((asset) => asset.source === "generated").map((asset) => asset.id);
  if (generatedNextIds.length === 0) {
    return uniqueIds([...existingIds, ...nextAssets.map((asset) => asset.id)]);
  }

  const assetById = new Map([...currentAssets, ...nextAssets].map((asset) => [asset.id, asset]));
  const existingGeneratedIds = existingIds.filter((assetId) => assetById.get(assetId)?.source === "generated");
  return uniqueIds([...existingGeneratedIds, ...generatedNextIds]);
}

interface NarrationRender {
  items: NarrationFlowItem[];
  markerSlots: NarrationFlowItem[];
}

const narrationImageMarkerPattern = /\[\s*(?:image|img|이미지|그림|일러스트|illustration)\s*[:：]\s*([^\]\n]*)\](?!\()/giu;

// Handles the inline [Image: ...] markers the LLM writes. Only markers INSIDE a markdown table row become inline image
// tokens (swapped for the actual generated image in that cell); markers elsewhere are stripped so they never show as
// literal text, and their images fall back to the normal per-beat anchored placement. Returns the rewritten content and
// how many table markers were converted.
function prepareNarrationContent(content: string): { content: string; markerCount: number } {
  let count = 0;
  const converted = content
    .replace(/\r\n/gu, "\n")
    .split("\n")
    .map((line) => {
      const insideTable = isMarkdownTableRow(line);
      return line.replace(narrationImageMarkerPattern, (_match, description: string) => {
        if (!insideTable) {
          // Prose marker: drop the literal text; the image is placed per beat by the normal flow.
          return "";
        }
        const alt = String(description ?? "")
          .replace(/[[\]()]/gu, " ")
          .replace(/\s+/gu, " ")
          .trim();
        const token = `![${alt || `image ${count + 1}`}](dynamicchat-image:${count})`;
        count += 1;
        return token;
      });
    })
    .join("\n");
  return { content: converted, markerCount: count };
}

function createNarrationFlow(content: string, assets: ImageAsset[], jobs: ImageGenerationJob[]): NarrationRender {
  const { content: preparedContent, markerCount } = prepareNarrationContent(content);
  const textItems = createNarrationTextItems(preparedContent);
  // Order completed assets before pending jobs so marker index k maps to the k-th image in document order.
  const orderedMedia: NarrationFlowItem[] = [
    ...assets.map((asset) => ({ id: `asset-${asset.id}`, kind: "image" as const, asset })),
    ...jobs.map((job) => ({ id: `job-${job.id}`, kind: "job" as const, job }))
  ];
  // The first markerCount media are rendered inline at their [Image: ...] markers (via NarrationMediaContext), so only
  // the remaining (unmarked) media still need to be anchored/appended in the flow.
  const markerSlots = orderedMedia.slice(0, markerCount);
  const mediaItems = orderedMedia.slice(markerCount);

  if (mediaItems.length === 0) {
    return { items: textItems, markerSlots };
  }
  if (textItems.length === 0) {
    return { items: mediaItems, markerSlots };
  }

  const mediaBeforeByAnchor = new Map<number, NarrationFlowItem[]>();
  const mediaAfterByAnchor = new Map<number, NarrationFlowItem[]>();
  mediaItems.forEach((media, mediaIndex) => {
    const anchorIndex = chooseNarrationMediaAnchor(textItems, media, mediaIndex, mediaItems.length);
    const mediaByAnchor = readNarrationMediaPlacement(media) === "before" ? mediaBeforeByAnchor : mediaAfterByAnchor;
    const anchored = mediaByAnchor.get(anchorIndex) ?? [];
    anchored.push(media);
    mediaByAnchor.set(anchorIndex, anchored);
  });

  const items = textItems.flatMap((item, index) => [
    ...(mediaBeforeByAnchor.get(index) ?? []),
    item,
    ...(mediaAfterByAnchor.get(index) ?? [])
  ]);
  return { items, markerSlots };
}

function createNarrationTextItems(content: string): NarrationFlowItem[] {
  return parseDynamicRichText(content).flatMap((segment): NarrationFlowItem[] => {
    if (segment.kind !== "markdown") {
      return [
        {
          id: segment.id,
          kind: "dynamic",
          blockKind: segment.kind,
          content: segment.content
        }
      ];
    }

    return splitMarkdownNarrationChunks(segment.content).map((chunk, index) => ({
      id: `${segment.id}-${index}`,
      kind: "markdown",
      content: chunk
    }));
  });
}

// Split a markdown segment into rendering chunks ONLY at real paragraph breaks (blank lines), keeping fenced code blocks
// intact. We deliberately do not split mid-paragraph: per-beat images are placed at paragraph boundaries, and finer
// splitting fragmented prose and broke multi-line markdown structures (tables, lists) into separate blocks.
function splitMarkdownNarrationChunks(content: string): string[] {
  const lines = content.replace(/\r\n/gu, "\n").split("\n");
  const chunks: string[] = [];
  const current: string[] = [];
  let inFence = false;

  for (const line of lines) {
    if (/^\s*```/u.test(line)) {
      inFence = !inFence;
    }

    if (!inFence && !line.trim()) {
      const chunk = current.join("\n").trim();
      if (chunk) {
        chunks.push(chunk);
      }
      current.length = 0;
      continue;
    }

    current.push(line);
  }

  const chunk = current.join("\n").trim();
  if (chunk) {
    chunks.push(chunk);
  }

  return chunks.length > 0 ? chunks : [content];
}

function chooseNarrationMediaAnchor(
  textItems: NarrationFlowItem[],
  media: NarrationFlowItem,
  mediaIndex: number,
  mediaCount: number
): number {
  const textCount = textItems.length;
  const defaultAnchor = Math.min(
    textCount - 1,
    Math.max(0, Math.round(((mediaIndex + 1) / (mediaCount + 1)) * textCount) - 1)
  );
  const terms = createMediaPlacementTerms(media);
  const anchorText = readNarrationMediaAnchorText(media);
  if (anchorText) {
    const normalizedAnchor = normalizeNarrationAnchorText(anchorText);
    const directIndex = textItems.findIndex(
      (item) =>
        (item.kind === "markdown" || item.kind === "dynamic") &&
        normalizeNarrationAnchorText(item.content).includes(normalizedAnchor)
    );
    if (directIndex >= 0) {
      return directIndex;
    }

    const fuzzyIndex = findNarrationAnchorIndex(textItems, normalizedAnchor);
    if (fuzzyIndex >= 0) {
      return fuzzyIndex;
    }
  }

  let bestIndex = defaultAnchor;
  let bestScore = 0;

  textItems.forEach((item, index) => {
    if (item.kind !== "markdown" && item.kind !== "dynamic") {
      return;
    }
    const text = item.content.toLowerCase();
    const score = terms.reduce((sum, term) => (text.includes(term) ? sum + Math.min(6, term.length) : sum), 0);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });

  return bestScore > 0 ? bestIndex : defaultAnchor;
}

function findNarrationAnchorIndex(textItems: NarrationFlowItem[], normalizedAnchor: string): number {
  const anchor = normalizedAnchor.trim();
  if (!anchor) {
    return -1;
  }
  const compactAnchor = anchor.replace(/\s+/gu, "");
  const anchorWindows = uniqueIds([
    anchor.slice(0, 80),
    anchor.slice(0, 56),
    anchor.slice(0, 36),
    anchor.slice(-56)
  ].filter((item) => item.length >= 12));
  const anchorTerms = anchor
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .split(/\s+/u)
    .map((term) => term.trim())
    .filter((term) => term.length >= 2)
    .slice(0, 18);

  let bestIndex = -1;
  let bestScore = 0;
  textItems.forEach((item, index) => {
    if (item.kind !== "markdown" && item.kind !== "dynamic") {
      return;
    }
    const text = normalizeNarrationAnchorText(item.content);
    const compactText = text.replace(/\s+/gu, "");
    let score = anchorWindows.some((window) => text.includes(window) || compactText.includes(window.replace(/\s+/gu, ""))) ? 24 : 0;
    if (compactAnchor.length >= 12 && compactText.includes(compactAnchor.slice(0, 40))) {
      score += 18;
    }
    score += anchorTerms.reduce((sum, term) => (text.includes(term) ? sum + Math.min(5, term.length) : sum), 0);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });

  return bestScore >= 8 ? bestIndex : -1;
}

function createMediaPlacementTerms(media: NarrationFlowItem): string[] {
  const source =
    media.kind === "image"
      ? [
          media.asset.title,
          media.asset.prompt,
          media.asset.tags.join(", "),
          readMetadataText(media.asset.providerMetadata, "cueLabel"),
          readMetadataText(media.asset.providerMetadata, "cueKind"),
          readMetadataText(media.asset.providerMetadata, "cueType"),
          readMetadataText(media.asset.providerMetadata, "anchorText")
        ]
      : media.kind === "job"
        ? (() => {
            const cue = getImageJobCue(media.job);
            return [
              media.job.reason,
              media.job.prompt,
              cue.scene,
              cue.tags?.join(", "),
              readMetadataText(media.job.providerPayload, "cueLabel"),
              readMetadataText(media.job.providerPayload, "cueKind"),
              readMetadataText(media.job.providerPayload, "cueType"),
              readMetadataText(media.job.providerPayload, "anchorText")
            ];
          })()
        : [];
  const stopwords = new Set(["the", "and", "with", "scene", "image", "generated", "current", "safe", "quality"]);
  return uniqueIds(
    source
      .join(", ")
      .toLowerCase()
      .split(/[,|;/\n]+|\s{2,}/u)
      .map((term) => term.trim().replace(/[._-]+/gu, " "))
      .filter((term) => term.length >= 3 && !stopwords.has(term))
      .slice(0, 28)
  );
}

function readNarrationMediaPlacement(media: NarrationFlowItem): "before" | "after" {
  const metadata = media.kind === "image" ? media.asset.providerMetadata : media.kind === "job" ? media.job.providerPayload : undefined;
  const placement = readMetadataText(metadata, "cuePlacement")?.toLowerCase();
  return placement === "before" ? "before" : "after";
}

function readNarrationMediaAnchorText(media: NarrationFlowItem): string | undefined {
  const metadata = media.kind === "image" ? media.asset.providerMetadata : media.kind === "job" ? media.job.providerPayload : undefined;
  return readMetadataText(metadata, "anchorText") ?? readMetadataText(metadata, "anchor_text");
}

function readMetadataText(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeNarrationAnchorText(value: string): string {
  return value
    .toLowerCase()
    .replace(/[*_`~>#[\]()"']/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
}

function applyTurnResultToState(baseState: AppState, result: TurnResult): AppState {
  const resultImageJobs = result.imageJobs?.length ? result.imageJobs : result.imageJob ? [result.imageJob] : [];
  // Early-dispatched images are attached to the live assistant message WHILE it streams, but result.assistantMessage
  // was built at turn start with imageAssetIds: []. Without this merge, the final commit replaces the message and the
  // first image vanishes (it appears briefly, then disappears). The result id is freshly generated each turn, so the
  // only existing same-id message is this turn's own streaming message — merging never carries stale images forward.
  const existingAssistant = baseState.messages.find((message) => message.id === result.assistantMessage.id);
  const assistantMessage =
    existingAssistant && existingAssistant.imageAssetIds.length > 0
      ? {
          ...result.assistantMessage,
          imageAssetIds: uniqueIds([...existingAssistant.imageAssetIds, ...result.assistantMessage.imageAssetIds])
        }
      : result.assistantMessage;
  return {
    ...baseState,
    simulation: { ...baseState.simulation, updatedAt: result.assistantMessage.createdAt },
    messages: upsertMessagesById(baseState.messages, [result.userMessage, assistantMessage]),
    memoryEvents: upsertMemoryEventsById(baseState.memoryEvents, result.memoryEvents),
    contextPacks: [...baseState.contextPacks, result.contextPack],
    selectedContextPackId: result.contextPack.id,
    promptModuleUsages: [...baseState.promptModuleUsages, ...result.promptModuleUsages],
    sidecarTraces: [...baseState.sidecarTraces, result.sidecarTrace],
    turnTraces: [...baseState.turnTraces, result.turnTrace],
    imageJobs: resultImageJobs.length > 0 ? [...baseState.imageJobs, ...resultImageJobs] : baseState.imageJobs,
    imageAssets: [...baseState.imageAssets, ...result.imageAssets],
    auditLog: [
      ...baseState.auditLog,
      createAuditEvent(baseState, "context_retrieved", "context_pack", result.contextPack.id, {
        source: result.contextPack.source,
        evidenceCount: result.contextPack.evidence.length,
        tokenBudget: result.contextPack.tokenBudget
      }),
      ...resultImageJobs.map((imageJob) =>
        createAuditEvent(baseState, "generation_job_created", "image_job", imageJob.id, {
          status: imageJob.status,
          triggerMode: imageJob.providerPayload.triggerMode,
          requiresConfirmation: imageJob.providerPayload.requiresConfirmation
        })
      )
    ]
  };
}

// A session reset is a CONVERSATION reset, not a SIMULATION reset (docs/continuity-reset-checklist.md):
// the new session must carry forward every simulation-level asset. createResetSessionState derives its
// output from the turn's base snapshot (the auto-progress working state), which never accumulates this run's
// async-generated image assets/jobs or the message/trace image links — those only ever reach the live React
// state. Rebuilding React state straight from that reset base therefore wipes turns 1..N-1's rendered images
// at the exact turn the handoff fires. Instead, layer ONLY the reset's new-session deltas (new activeSessionId,
// handoff record + memory event, new-session system message, reset Context Pack, continuity check) on top of
// the richer live React state. Idempotent: safe to apply once during streaming and again at final commit.
/**
 * Which run a piece of in-flight work belongs to.
 *
 * The app keeps ONE React state object for whichever simulation + progress run is open, and every async
 * completion writes into it. Simulation id alone is not the identity of a run: switching progress runs keeps
 * `simulation.id` constant while swapping the whole messages / traces / assets set, so a turn or an image
 * started in run A used to commit straight into run B — and because hydrateState re-snapshots the top-level
 * arrays into the active run on every write, the leak was permanent rather than transient.
 */
interface RunOwner {
  simulationId: string;
  progressRunId: string;
}

/** Runs an LLM call outside a turn on the app's local-GPU lock (localGpuCoordinator). */
type LocalLlmTaskRunner = <T>(state: AppState, task: () => Promise<T>) => Promise<T>;

/** A settings section a button outside the settings tab asked to show. `id` makes a repeated click a new request. */
interface SettingsJumpRequest {
  section: "photoreal";
  id: string;
}

/** One execution of an image job (see runQueuedImageJob). */
interface ImageJobDispatch {
  /** Stamped on the job record as providerPayload.dispatchToken when this dispatch claims it. */
  token: string;
  owner: RunOwner;
  controller: AbortController;
}

function readRunOwner(state: AppState): RunOwner {
  return { simulationId: state.simulation.id, progressRunId: state.activeProgressRunId };
}

/** True when `current` is still the run that started the work. Every post-await setState must check this. */
function ownsActiveRun(current: AppState, owner: RunOwner): boolean {
  return current.simulation.id === owner.simulationId && current.activeProgressRunId === owner.progressRunId;
}

function layerSessionResetOntoLiveState(liveState: AppState, resetState: AppState): AppState {
  const liveMessageIds = new Set(liveState.messages.map((message) => message.id));
  const liveMemoryEventIds = new Set(liveState.memoryEvents.map((event) => event.id));
  const liveContextPackIds = new Set(liveState.contextPacks.map((pack) => pack.id));
  const liveHandoffIds = new Set(liveState.handoffs.map((handoff) => handoff.id));
  const liveContinuityIds = new Set(liveState.continuityChecks.map((check) => check.id));
  return {
    ...liveState,
    simulation: resetState.simulation,
    messages: [...liveState.messages, ...resetState.messages.filter((message) => !liveMessageIds.has(message.id))],
    memoryEvents: [...liveState.memoryEvents, ...resetState.memoryEvents.filter((event) => !liveMemoryEventIds.has(event.id))],
    contextPacks: [...liveState.contextPacks, ...resetState.contextPacks.filter((pack) => !liveContextPackIds.has(pack.id))],
    handoffs: [...liveState.handoffs, ...resetState.handoffs.filter((handoff) => !liveHandoffIds.has(handoff.id))],
    continuityChecks: [
      ...liveState.continuityChecks,
      ...resetState.continuityChecks.filter((check) => !liveContinuityIds.has(check.id))
    ],
    selectedContextPackId: resetState.selectedContextPackId ?? liveState.selectedContextPackId
  };
}

function upsertMessagesById(messages: ChatMessage[], nextMessages: ChatMessage[]): ChatMessage[] {
  const replacements = new Map(nextMessages.map((message) => [message.id, message]));
  const existingIds = new Set(messages.map((message) => message.id));
  return [
    ...messages.map((message) => replacements.get(message.id) ?? message),
    ...nextMessages.filter((message) => !existingIds.has(message.id))
  ];
}

function upsertMemoryEventsById(
  memoryEvents: AppState["memoryEvents"],
  nextMemoryEvents: AppState["memoryEvents"]
): AppState["memoryEvents"] {
  const replacements = new Map(nextMemoryEvents.map((event) => [event.id, event]));
  const existingIds = new Set(memoryEvents.map((event) => event.id));
  return [
    ...memoryEvents.map((event) => replacements.get(event.id) ?? event),
    ...nextMemoryEvents.filter((event) => !existingIds.has(event.id))
  ];
}

function upsertStreamingTurnMessages(state: AppState, userMessage: ChatMessage, assistantMessage: ChatMessage): AppState {
  return {
    ...state,
    messages: upsertMessagesById(state.messages, [userMessage, assistantMessage])
  };
}

function applyMemoryIngestResultToState(
  state: AppState,
  turnId: string,
  memoryEvents: AppState["memoryEvents"],
  memoryIngestMs: number
): AppState {
  if (memoryEvents.length === 0) {
    return state;
  }

  const memoryEventsById = new Map(memoryEvents.map((event) => [event.id, event]));
  const existingMemoryEventIds = new Set(state.memoryEvents.map((event) => event.id));
  return {
    ...state,
    memoryEvents: [
      ...state.memoryEvents.map((event) => memoryEventsById.get(event.id) ?? event),
      ...memoryEvents.filter((event) => !existingMemoryEventIds.has(event.id))
    ],
    turnTraces: state.turnTraces.map((trace) =>
      trace.turnId === turnId
        ? {
            ...trace,
            metrics: {
              ...trace.metrics,
              memoryIngestMs
            }
          }
        : trace
    )
  };
}

type AssistantRegenerationPlan = {
  assistantMessage: AppState["messages"][number];
  baseState: AppState;
  manualImage: boolean;
  removedMessageCount: number;
  userMessage: AppState["messages"][number];
};

function createAssistantRegenerationPlan(state: AppState, assistantMessageId: string): AssistantRegenerationPlan | undefined {
  const assistantIndex = state.messages.findIndex(
    (message) => message.id === assistantMessageId && message.role === "assistant"
  );
  if (assistantIndex < 0) {
    return undefined;
  }

  let userIndex = -1;
  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    if (state.messages[index]?.role === "user") {
      userIndex = index;
      break;
    }
  }
  if (userIndex < 0) {
    return undefined;
  }

  const assistantMessage = state.messages[assistantIndex];
  const userMessage = state.messages[userIndex];
  const retainedMessages = state.messages.slice(0, userIndex);
  const droppedMessages = state.messages.slice(userIndex);
  const droppedMessageIds = new Set(droppedMessages.map((message) => message.id));
  const cutCreatedAt = userMessage.createdAt;
  const droppedTraces = state.turnTraces.filter(
    (trace) => droppedMessageIds.has(trace.userMessageId) || droppedMessageIds.has(trace.assistantMessageId)
  );
  const droppedTraceIds = new Set(droppedTraces.map((trace) => trace.id));
  const droppedContextPackIds = new Set(droppedTraces.map((trace) => trace.contextPackId));
  const droppedPromptUsageIds = new Set(droppedTraces.flatMap((trace) => trace.promptModuleUsageIds));
  const droppedSidecarTraceIds = new Set(droppedTraces.map((trace) => trace.sidecarTraceId));
  const droppedMemoryEventIds = new Set(droppedTraces.flatMap((trace) => trace.memoryEventIds));
  const droppedImageJobIds = new Set(droppedTraces.map((trace) => trace.imageJobId).filter((id): id is string => Boolean(id)));
  const droppedImageAssetIds = new Set(droppedTraces.flatMap((trace) => trace.imageAssetIds));

  state.imageJobs.forEach((job) => {
    if (droppedMessageIds.has(job.turnId) || isAfterIso(job.createdAt, cutCreatedAt)) {
      droppedImageJobIds.add(job.id);
    }
  });
  state.imageJobs.forEach((job) => {
    if (droppedImageJobIds.has(job.id)) {
      job.assetIds.forEach((assetId) => droppedImageAssetIds.add(assetId));
    }
  });
  state.imageAssets.forEach((asset) => {
    if (asset.jobId && droppedImageJobIds.has(asset.jobId)) {
      droppedImageAssetIds.add(asset.id);
    }
  });

  const nextContextPacks = state.contextPacks.filter(
    (pack) => !droppedContextPackIds.has(pack.id) && !isAfterIso(pack.createdAt, cutCreatedAt)
  );
  const originalTrace = state.turnTraces.find((trace) => trace.assistantMessageId === assistantMessageId);

  return {
    assistantMessage,
    userMessage,
    manualImage: Boolean(originalTrace?.imageJobId && state.imageProfile.triggerMode === "manual"),
    removedMessageCount: Math.max(0, droppedMessages.length),
    baseState: hydrateState({
      ...state,
      simulation: {
        ...state.simulation,
        activeSessionId: userMessage.sessionId,
        updatedAt: new Date().toISOString()
      },
      messages: retainedMessages,
      memoryEvents: state.memoryEvents.filter(
        (event) =>
          !droppedMemoryEventIds.has(event.id) &&
          !(event.sourceTurnId && droppedMessageIds.has(event.sourceTurnId)) &&
          !isAfterIso(event.createdAt, cutCreatedAt)
      ),
      contextPacks: nextContextPacks,
      handoffs: state.handoffs.filter((handoff) => !isAfterIso(handoff.createdAt, cutCreatedAt)),
      continuityChecks: state.continuityChecks.filter((check) => !isAfterIso(check.checkedAt, cutCreatedAt)),
      promptModuleUsages: state.promptModuleUsages.filter(
        (usage) =>
          !droppedPromptUsageIds.has(usage.id) &&
          !droppedMessageIds.has(usage.turnId) &&
          !isAfterIso(usage.createdAt, cutCreatedAt)
      ),
      sidecarTraces: state.sidecarTraces.filter(
        (trace) =>
          !droppedSidecarTraceIds.has(trace.id) &&
          !droppedMessageIds.has(trace.turnId) &&
          !isAfterIso(trace.createdAt, cutCreatedAt)
      ),
      turnTraces: state.turnTraces.filter((trace) => !droppedTraceIds.has(trace.id)),
      imageAssets: state.imageAssets.filter(
        (asset) =>
          asset.source === "stored" ||
          (!droppedImageAssetIds.has(asset.id) &&
            !(asset.jobId && droppedImageJobIds.has(asset.jobId)) &&
            !isAfterIso(asset.createdAt, cutCreatedAt))
      ),
      imageJobs: state.imageJobs.filter(
        (job) =>
          !droppedImageJobIds.has(job.id) &&
          !droppedMessageIds.has(job.turnId) &&
          !isAfterIso(job.createdAt, cutCreatedAt)
      ),
      selectedContextPackId: nextContextPacks.at(-1)?.id
    })
  };
}

function isAfterIso(value: string | undefined, threshold: string | undefined): boolean {
  const valueTime = value ? Date.parse(value) : Number.NaN;
  const thresholdTime = threshold ? Date.parse(threshold) : Number.NaN;
  return Number.isFinite(valueTime) && Number.isFinite(thresholdTime) && valueTime > thresholdTime;
}

function createLlmFallbackNotice(prefix: string, trace: TurnResult["sidecarTrace"]): string {
  const firstError = trace.errors.find((error) => error.trim())?.replace(/\s+/gu, " ").trim();
  if (firstError && /429|rate limit|요청 제한|RESOURCE_EXHAUSTED|quota exceeded/iu.test(firstError)) {
    const compact = firstError.slice(0, 180);
    return `${prefix}: ${compact}${compact.length < firstError.length ? "..." : ""}`;
  }
  return firstError
    ? `${prefix}: ${firstError.slice(0, 140)}${firstError.length > 140 ? "..." : ""}`
    : `${prefix}. API 키/모델 설정과 응답 JSON 형식을 확인하세요.`;
}

function shouldPrimeNeuralMapSimulation(state: AppState): boolean {
  return (
    state.neuralMap.enabled &&
    !state.contextPacks.some((pack) => pack.source === "neuralmap" && pack.sessionId === state.simulation.activeSessionId)
  );
}

function createNeuralMapPrimerQuery(state: AppState): string {
  const characterLine = state.characters
    .slice(0, 8)
    .map((character) => `${character.name}: ${character.role || character.summary}`)
    .join(" / ");
  const moduleLine = state.modules
    .filter((module) => module.enabled && module.tokenPolicy !== "disabled")
    .sort((a, b) => b.priority - a.priority)
    .slice(0, 6)
    .map((module) => module.title)
    .join(", ");

  return [
    "Register this DynamicChat simulation scope and prepare an initial Context Pack.",
    `simulation_id:${state.simulation.id}`,
    `session_id:${state.simulation.activeSessionId}`,
    `progress_run_id:${state.activeProgressRunId}`,
    `title:${state.simulation.title}`,
    state.simulation.description,
    characterLine ? `characters:${characterLine}` : undefined,
    moduleLine ? `active_modules:${moduleLine}` : undefined
  ]
    .filter((item): item is string => Boolean(item?.trim()))
    .join("\n")
    .slice(0, 1600);
}

function App() {
  const [personalApiVault, setPersonalApiVault] = useState<PersonalApiVault>(() =>
    promoteVibeTransferToVault(loadPersonalApiVault(), loadState())
  );
  const personalApiVaultRef = useRef(personalApiVault);
  useEffect(() => {
    personalApiVaultRef.current = personalApiVault;
  }, [personalApiVault]);
  const [rawState, setRawState] = useState<AppState>(() =>
    applyPersonalApiVault(hydrateState(loadState()), promoteVibeTransferToVault(loadPersonalApiVault(), loadState()))
  );
  const state = useMemo(() => hydrateState(rawState), [rawState]);
  const setState = useCallback<Dispatch<SetStateAction<AppState>>>((value) => {
    setRawState((current) => {
      const normalizedCurrent = hydrateState(current);
      const nextState =
        typeof value === "function"
          ? (value as (current: AppState) => AppState)(normalizedCurrent)
          : value;
      return hydrateState(nextState);
    });
  }, []);
  // Live view of the current run, readable from async callbacks that closed over an older `state`. Used to
  // decide whether long-running work (image planning, provider calls) still belongs to the run on screen
  // BEFORE spending on it, rather than only discarding the result afterwards.
  const latestStateRef = useRef(state);
  useEffect(() => {
    latestStateRef.current = state;
  }, [state]);
  const [simulationLibrary, setSimulationLibrary] = useState<SimulationLibrary>(() => loadSimulationLibrary(hydrateState(loadState())));
  const [view, setView] = useState<"home" | "create" | "simulation">("home");
  const [builderMode, setBuilderMode] = useState<BuilderMode>("create");
  const [builderSource, setBuilderSource] = useState<AppState | undefined>();
  const [draft, setDraft] = useState("");
  const [pendingUserText, setPendingUserText] = useState("");
  const [isSending, setIsSending] = useState(false);
  // Which run the in-flight turn belongs to. `isSending` stays app-wide because it also guards against two
  // concurrent turns, but the VISIBLE progress (pending bubble, phase card, composer lock) must belong to the
  // run that started it — otherwise simulation A's turn appears to be running inside simulation B.
  const [activeTurnOwner, setActiveTurnOwner] = useState<RunOwner | undefined>(undefined);
  const [isResetting, setIsResetting] = useState(false);
  const [personalSettingsOpen, setPersonalSettingsOpen] = useState(false);
  const [manualImage, setManualImage] = useState(false);
  const [isAutoProgressing, setIsAutoProgressing] = useState(false);
  const [autoProgressRemaining, setAutoProgressRemaining] = useState(0);
  const [autoProgressTotal, setAutoProgressTotal] = useState(0);
  // Coarse current-turn phase, surfaced in the chat thread so the user can see what the turn is doing
  // (context retrieval → response generation → image generation) instead of one opaque "진행 중".
  const [turnPhase, setTurnPhase] = useState<TurnPhase | undefined>(undefined);
  const turnAbortRef = useRef<AbortController | undefined>(undefined);
  const autoProgressStopRef = useRef(false);
  const autoProgressOwnerRef = useRef<RunOwner | undefined>(undefined);
  const autoProgressActiveRef = useRef(false);
  const autoProgressResumeAttemptedRef = useRef(false);
  // Opens on the panel that answers the question a reader actually has — "where did my picture go". It
  // used to open on Neural, which on a fresh install is a panel for a service that ships disabled and is
  // not running, so the first thing in the rail was an empty graph and a "로컬 미러" badge.
  const [rightPanel, setRightPanel] = useState<RightPanel>("image");
  // A queue, not a single slot. One 2.4-second slot was the app's entire feedback channel, so a failure
  // notice was routinely overwritten by the next success before it could be read — which is why image and
  // provider errors felt invisible.
  const [runtimeNotices, setRuntimeNotices] = useState<RuntimeNotice[]>([]);
  const runtimeNoticeTimersRef = useRef<Map<string, number>>(new Map());
  const [storageReady, setStorageReady] = useState(false);
  // One serial chain PER RENDERER, not one for all images. NovelAI is a remote API and photoreal is the local GPU;
  // behind a single chain a paragraph-cadence turn's eight NovelAI cuts held every photoreal cut back (and the
  // reverse), although the two never compete for anything. Each backend stays serial within itself.
  const imageJobQueuesRef = useRef<Map<ImageBackendId, Promise<void>>>(new Map());
  // The dispatch that currently owns each executing image job, keyed by job id. A cancel, a regeneration and a run
  // switch end a dispatch here, synchronously: its controller aborts the provider request (for photoreal the relay
  // then interrupts ComfyUI, so the GPU lock and the photoreal chain are released at once), and its token stops
  // every later write of that dispatch from landing on the cut.
  const imageJobDispatchesRef = useRef<Map<string, ImageJobDispatch>>(new Map());
  const imageAssetHydrationAttemptsRef = useRef<Map<string, number>>(new Map());
  const imagePayloadCompactionAttemptsRef = useRef<Map<string, number>>(new Map());
  const neuralMapPrimeAttemptsRef = useRef<Set<string>>(new Set());
  const promptModuleSyncRef = useRef<Set<string>>(new Set());
  // Claims each turn+cue exactly once so the early (mid-stream) and post-turn dispatch paths can never
  // both fire a paid image request for the same cut. Keyed by createImageJobDispatchKey.
  const dispatchedImageCueKeysRef = useRef<Set<string>>(new Set());
  // Pending saves, keyed by simulation id. It used to be a single last-write-wins slot, which silently
  // coalesced two DIFFERENT simulations into one save: switching simulations while a save was pending
  // dropped the outgoing simulation's last change entirely.
  const pendingStateSavesRef = useRef<Map<string, { snapshot: AppState; options: SaveStateOptions }>>(new Map());
  const stateSaveScheduledRef = useRef(false);

  const pushRuntimeNotice = useCallback((message: string, tone: RuntimeNoticeTone) => {
    setRuntimeNotices((current) => [...current, { id: createId("notice"), message, tone }].slice(-RUNTIME_NOTICE_LIMIT));
  }, []);

  const showRuntimeNotice = useCallback(
    (message: string) => {
      // Anything the app itself phrases as a failure is an error, whoever called it — the ~10 call sites
      // predate the tone parameter and all use the same wording. 중지 is deliberately NOT in this list: the
      // auto-progress start/stop confirmations all contain it and are purely informational.
      pushRuntimeNotice(message, /실패|오류|없습니다|초과/u.test(message) ? "error" : "info");
    },
    [pushRuntimeNotice]
  );

  const dismissRuntimeNotice = useCallback((noticeId: string) => {
    setRuntimeNotices((current) => current.filter((notice) => notice.id !== noticeId));
  }, []);

  // A 16 GB card holds a local Ollama model or a photoreal model, not both; whichever spills runs at a fraction of a
  // token per second. ONE coordinator for the whole app, because the lock only works if every turn and every
  // photoreal job queue on the same instance. It decides per call whether to engage (shouldCoordinateLocalGpu) and
  // is a pass-through otherwise, so it costs nothing for hosted LLMs or a remote ComfyUI.
  const localGpuHandoffNoticeShownRef = useRef(false);
  const localGpuCoordinatorRef = useRef<ReturnType<typeof createLocalGpuCoordinator> | null>(null);
  if (localGpuCoordinatorRef.current === null) {
    localGpuCoordinatorRef.current = createLocalGpuCoordinator({
      onHandoff: () => {
        // Once per page session: the first handoff explains the pause before a cut; every later one would only
        // repeat it on each turn.
        if (localGpuHandoffNoticeShownRef.current) {
          return;
        }
        localGpuHandoffNoticeShownRef.current = true;
        pushRuntimeNotice(
          "로컬 GPU 교대를 시작했습니다. Ollama 모델과 실사 모델이 VRAM에 함께 올라가지 않아, 턴과 실사 이미지 사이에 서로의 모델을 내리고 다시 올립니다.",
          "info"
        );
      }
    });
  }
  const localGpu = localGpuCoordinatorRef.current;
  // For LLM calls made outside a turn (the settings panel's '실제 API 턴 테스트'): the same lock a turn takes, so a local
  // Ollama never loads beside a photoreal model still resident on a 16 GB card, where it spills to 0.1 tok/s.
  const runLocalLlmTask = useCallback(
    <T,>(taskState: AppState, task: () => Promise<T>): Promise<T> => localGpu.run("llm", taskState, task),
    [localGpu]
  );

  const flushPendingStateSaves = useCallback(() => {
    const pending = [...pendingStateSavesRef.current.values()];
    pendingStateSavesRef.current.clear();
    stateSaveScheduledRef.current = false;
    for (const entry of pending) {
      saveStateSnapshot(entry.snapshot, entry.options);
    }
  }, []);

  const scheduleStatePersistence = useCallback((snapshot: AppState, options: SaveStateOptions = {}) => {
    const key = snapshot.simulation.id;
    const existing = pendingStateSavesRef.current.get(key);
    pendingStateSavesRef.current.set(key, {
      snapshot,
      options: {
        ...existing?.options,
        ...options,
        includeImagePayloads:
          existing?.options.includeImagePayloads === true || options.includeImagePayloads === true
      }
    });

    if (stateSaveScheduledRef.current) {
      return;
    }

    stateSaveScheduledRef.current = true;
    scheduleIdleTask(flushPendingStateSaves);
  }, [flushPendingStateSaves]);

  const primeNeuralMapSimulation = useCallback(
    (targetState: AppState) => {
      const hydratedTarget = hydrateState(targetState);
      if (!shouldPrimeNeuralMapSimulation(hydratedTarget)) {
        return;
      }

      // Keep RAG sub-prompts/scene-rules retrievable by MEANING (semantic), not just literal keyword, by
      // ingesting changed module bodies as NeuralMap documents. Deduped by content signature so unchanged
      // modules are never re-sent. Fire-and-forget; failures simply fall back to lexical activation.
      const unsyncedSignatures = new Map<string, string>();
      for (const module of hydratedTarget.modules) {
        if (!isSemanticRetrievalModule(module)) {
          continue;
        }
        const signature = `${hydratedTarget.simulation.id}:${createPromptModuleSyncSignature(module)}`;
        if (!promptModuleSyncRef.current.has(signature)) {
          unsyncedSignatures.set(module.id, signature);
        }
      }
      if (unsyncedSignatures.size > 0) {
        for (const signature of unsyncedSignatures.values()) {
          promptModuleSyncRef.current.add(signature);
        }
        void new NeuralMapClient(hydratedTarget.neuralMap)
          .syncPromptModuleDocuments(hydratedTarget, new Set(unsyncedSignatures.keys()))
          .catch(() => {
            for (const signature of unsyncedSignatures.values()) {
              promptModuleSyncRef.current.delete(signature);
            }
          });
      }

      const attemptKey = [
        hydratedTarget.neuralMap.baseUrl,
        hydratedTarget.simulation.id,
        hydratedTarget.simulation.activeSessionId,
        hydratedTarget.activeProgressRunId
      ].join("|");
      if (neuralMapPrimeAttemptsRef.current.has(attemptKey)) {
        return;
      }
      neuralMapPrimeAttemptsRef.current.add(attemptKey);

      void (async () => {
        const contextPack = await new NeuralMapClient(hydratedTarget.neuralMap).getSimulationContext(
          hydratedTarget,
          createNeuralMapPrimerQuery(hydratedTarget)
        );
        if (contextPack.source !== "neuralmap") {
          return;
        }

        setState((current) => {
          const normalizedCurrent = hydrateState(current);
          if (
            normalizedCurrent.simulation.id !== hydratedTarget.simulation.id ||
            normalizedCurrent.simulation.activeSessionId !== hydratedTarget.simulation.activeSessionId ||
            normalizedCurrent.activeProgressRunId !== hydratedTarget.activeProgressRunId ||
            normalizedCurrent.contextPacks.some((pack) => pack.id === contextPack.id)
          ) {
            return normalizedCurrent;
          }

          return {
            ...normalizedCurrent,
            contextPacks: [...normalizedCurrent.contextPacks, contextPack],
            selectedContextPackId: contextPack.id,
            auditLog: [
              ...normalizedCurrent.auditLog,
              createAuditEvent(normalizedCurrent, "context_retrieved", "context_pack", contextPack.id, {
                source: contextPack.source,
                evidenceCount: contextPack.evidence.length,
                tokenBudget: contextPack.tokenBudget,
                operation: "neuralmap_initial_sync"
              })
            ]
          };
        });
      })().catch(() => {
        neuralMapPrimeAttemptsRef.current.delete(attemptKey);
      });
    },
    [setState]
  );

  useEffect(() => {
    let cancelled = false;

    async function restorePersistedData() {
      const apiClient = createDynamicChatApiClient();
      const rawLocalState = loadState();
      const localState = hydrateState(rawLocalState);
      const storedLocalVault = loadPersonalApiVault();
      // One-time migration: lift any per-simulation vibe transfer setup into the (now global) vault.
      const localVault = promoteVibeTransferToVault(storedLocalVault, localState);
      if (!arePersonalApiVaultsEqual(localVault, storedLocalVault)) {
        savePersonalApiVault(localVault);
      }
      const localLibrary = loadSimulationLibrary(localState);
      let nextVault = localVault;
      let nextLibrary = localLibrary;
      let restoredFromServer = false;

      try {
        const [serverVaultResult, serverSimulationsResult] = await Promise.allSettled([
          apiClient.loadPersonalApiVault<Partial<PersonalApiVault>>(),
          apiClient.listSimulations()
        ]);

        if (cancelled) {
          return;
        }

        if (serverVaultResult.status === "fulfilled" && serverVaultResult.value) {
          const serverVault = hydratePersonalApiVault(serverVaultResult.value);
          const mergedVault = mergePersonalApiVaults(localVault, serverVault);

          if (hasPersonalApiVaultSecrets(mergedVault)) {
            nextVault = mergedVault;
          }

          if (!arePersonalApiVaultsEqual(mergedVault, localVault)) {
            savePersonalApiVault(mergedVault);
            restoredFromServer = true;
          }

          if (!arePersonalApiVaultsEqual(mergedVault, serverVault) && hasPersonalApiVaultSecrets(mergedVault)) {
            void apiClient.savePersonalApiVault(mergedVault).catch(() => undefined);
          }
        } else if (hasPersonalApiVaultSecrets(localVault)) {
          void apiClient.savePersonalApiVault(localVault).catch(() => undefined);
        }

        if (serverSimulationsResult.status === "fulfilled" && serverSimulationsResult.value.length > 0) {
          const serverLibrary = serverSimulationsResult.value.map((item) => hydrateState(item));
          nextLibrary = mergeSimulationLibraries(localLibrary, serverLibrary);
          restoredFromServer = true;
        }

        setPersonalApiVault(nextVault);
        setSimulationLibrary(nextLibrary);
        setState((current) => {
          const hydratedCurrent = hydrateState(current);
          if (hasRuntimeAdvancedSinceSnapshot(hydratedCurrent, localState)) {
            return applyPersonalApiVault(hydratedCurrent, nextVault);
          }

          return applyPersonalApiVault(hydrateState(pickInitialSimulation(localState, nextLibrary, Boolean(rawLocalState))), nextVault);
        });

        if (restoredFromServer) {
          showRuntimeNotice("저장된 시뮬레이션과 개인 설정을 복원했습니다.");
        }
      } catch {
        if (!cancelled && hasPersonalApiVaultSecrets(localVault)) {
          void apiClient.savePersonalApiVault(localVault).catch(() => undefined);
        }
      } finally {
        if (!cancelled) {
          setStorageReady(true);
        }
      }
    }

    void restorePersistedData();

    return () => {
      cancelled = true;
    };
  }, [showRuntimeNotice]);

  useEffect(() => {
    if (!storageReady) {
      return;
    }

    scheduleStatePersistence(state);
    if (!isSending) {
      setSimulationLibrary((current) => upsertSimulationInLibrary(current, state));
    }
  }, [isSending, scheduleStatePersistence, state, storageReady]);

  useEffect(() => {
    if (!storageReady) {
      return undefined;
    }

    const persistBeforeUnload = () => {
      saveStateSnapshot(state, { skipServer: true });
    };

    window.addEventListener("pagehide", persistBeforeUnload);
    return () => window.removeEventListener("pagehide", persistBeforeUnload);
  }, [state, storageReady]);

  useEffect(() => {
    if (!storageReady) {
      return;
    }

    const compactionSignature = createImagePayloadCompactionSignature(state);
    if (!compactionSignature) {
      return;
    }

    const now = Date.now();
    const lastAttempt = imagePayloadCompactionAttemptsRef.current.get(compactionSignature) ?? 0;
    if (now - lastAttempt < 30_000) {
      return;
    }
    imagePayloadCompactionAttemptsRef.current.set(compactionSignature, now);

    const assetsWithPayload = collectImageAssetsWithPayload(state);
    let cancelled = false;

    async function compactExistingImagePayloads() {
      const persistedAssets = await persistRuntimeImagePayloads(state.simulation.id, assetsWithPayload);
      if (cancelled || !persistedAssets.some((asset) => asset.objectKey && !asset.dataUrl)) {
        return;
      }

      setState((current) =>
        current.simulation.id === state.simulation.id
          ? compactImagePayloadsInState(current, persistedAssets)
          : current
      );
    }

    void compactExistingImagePayloads();

    return () => {
      cancelled = true;
    };
  }, [state, storageReady]);

  useEffect(() => {
    if (!storageReady || isSending) {
      return;
    }

    primeNeuralMapSimulation(state);
  }, [isSending, primeNeuralMapSimulation, state, storageReady]);

  useEffect(() => {
    if (!storageReady) {
      return;
    }

    const missingAssetIds = collectMissingImagePayloadIds(state);
    const missingSignature = createMissingImagePayloadSignature(state, personalApiVault.imageStoragePath);
    if (!missingSignature) {
      return;
    }

    const now = Date.now();
    const lastAttempt = imageAssetHydrationAttemptsRef.current.get(missingSignature) ?? 0;
    if (now - lastAttempt < 30_000) {
      return;
    }
    imageAssetHydrationAttemptsRef.current.set(missingSignature, now);

    let cancelled = false;

    async function hydrateMissingImagePayloads() {
      try {
        const hydratedAssets = await createDynamicChatApiClient().listAssets(state.simulation.id, missingAssetIds);
        if (cancelled) {
          return;
        }

        setState((current) => {
          if (current.simulation.id !== state.simulation.id) {
            return current;
          }

          return mergeHydratedImagePayloadsIntoState(current, hydratedAssets);
        });
      } catch {
        // The local cache may still have metadata-only assets when the dev API is offline.
      }
    }

    void hydrateMissingImagePayloads();

    return () => {
      cancelled = true;
    };
  }, [personalApiVault.imageStoragePath, setState, state, storageReady]);

  useEffect(() => {
    if (!storageReady || state.llm.provider === "mock" || hasStoredSecret(state.llm)) {
      return;
    }

    const localSecret = getPersonalLlmSecret(personalApiVault, state.llm.provider);
    if (hasStoredSecret(localSecret)) {
      setState((current) => applyPersonalApiVault(current, personalApiVault));
      return;
    }

    let cancelled = false;

    async function recoverServerLlmSecret() {
      try {
        const serverVaultPayload = await createDynamicChatApiClient().loadPersonalApiVault<Partial<PersonalApiVault>>();
        if (cancelled || !serverVaultPayload) {
          return;
        }

        const serverVault = hydratePersonalApiVault(serverVaultPayload);
        const mergedVault = mergePersonalApiVaults(personalApiVault, serverVault);
        const recoveredSecret = getPersonalLlmSecret(mergedVault, state.llm.provider);
        if (!hasStoredSecret(recoveredSecret)) {
          return;
        }

        savePersonalApiVault(mergedVault);
        setPersonalApiVault(mergedVault);
        setState((current) => applyPersonalApiVault(current, mergedVault));
        showRuntimeNotice(`${getLlmProviderOption(state.llm.provider).label} 키를 복원했습니다.`);
      } catch {
        // The regular settings panel will keep showing the missing-key status.
      }
    }

    void recoverServerLlmSecret();

    return () => {
      cancelled = true;
    };
  }, [personalApiVault, showRuntimeNotice, state.llm, storageReady]);

  useEffect(() => {
    if (!storageReady || !state.novelAi.enabled || hasStoredSecret(state.novelAi)) {
      return;
    }

    const localSecret = normalizeStoredNovelAiSecretRecord(personalApiVault.novelAi);
    if (hasStoredSecret(localSecret)) {
      setState((current) => applyPersonalApiVault(current, personalApiVault));
      return;
    }

    let cancelled = false;

    async function recoverServerNovelAiSecret() {
      try {
        const serverVaultPayload = await createDynamicChatApiClient().loadPersonalApiVault<Partial<PersonalApiVault>>();
        if (cancelled || !serverVaultPayload) {
          return;
        }

        const serverVault = hydratePersonalApiVault(serverVaultPayload);
        const mergedVault = mergePersonalApiVaults(personalApiVault, serverVault);
        const recoveredSecret = normalizeStoredNovelAiSecretRecord(mergedVault.novelAi);
        if (!hasStoredSecret(recoveredSecret)) {
          return;
        }

        savePersonalApiVault(mergedVault);
        setPersonalApiVault(mergedVault);
        setState((current) => applyPersonalApiVault(current, mergedVault));
        showRuntimeNotice("NovelAI 토큰을 복원했습니다.");
      } catch {
        // The runtime settings panel will keep showing the missing-token status.
      }
    }

    void recoverServerNovelAiSecret();

    return () => {
      cancelled = true;
    };
  }, [personalApiVault, showRuntimeNotice, state.novelAi, storageReady]);

  useEffect(() => {
    saveSimulationLibrary(simulationLibrary);
  }, [simulationLibrary]);

  // Only info notices expire. An error stays until the user dismisses it, so a failure that happened while
  // the user was looking elsewhere is still there when they look back.
  //
  // One timer per notice id, kept in a ref. Scheduling them from an effect that depends on the whole array
  // meant every new notice cancelled and recreated the timers for all the live ones, so a steady drip of
  // messages kept info toasts on screen indefinitely instead of for RUNTIME_NOTICE_INFO_TTL_MS.
  useEffect(() => {
    const timers = runtimeNoticeTimersRef.current;
    for (const notice of runtimeNotices) {
      if (notice.tone !== "info" || timers.has(notice.id)) {
        continue;
      }
      timers.set(
        notice.id,
        window.setTimeout(() => {
          timers.delete(notice.id);
          setRuntimeNotices((current) => current.filter((candidate) => candidate.id !== notice.id));
        }, RUNTIME_NOTICE_INFO_TTL_MS)
      );
    }
    // Drop timers for notices that were dismissed before they expired.
    const liveIds = new Set(runtimeNotices.map((notice) => notice.id));
    for (const [id, timeoutId] of timers) {
      if (!liveIds.has(id)) {
        window.clearTimeout(timeoutId);
        timers.delete(id);
      }
    }
  }, [runtimeNotices]);

  useEffect(() => {
    const timers = runtimeNoticeTimersRef.current;
    return () => {
      timers.forEach((timeoutId) => window.clearTimeout(timeoutId));
      timers.clear();
    };
  }, []);

  const selectedModule = useMemo(
    () => state.modules.find((module) => module.id === state.selectedModuleId) ?? state.modules[0],
    [state.modules, state.selectedModuleId]
  );
  const selectedContextPack = useMemo(
    () => state.contextPacks.find((pack) => pack.id === state.selectedContextPackId) ?? state.contextPacks.at(-1),
    [state.contextPacks, state.selectedContextPackId]
  );
  const latestAssets = useMemo(() => state.imageAssets.slice().reverse(), [state.imageAssets]);

  const updateModule = useCallback((moduleId: string, patch: Partial<PromptModule>) => {
    setState((current) => ({
      ...current,
      modules: current.modules.map((module) =>
        module.id === moduleId
          ? {
              ...module,
              ...patch,
              updatedAt: new Date().toISOString()
            }
          : module
      )
    }));
  }, []);

  const updateImageProfile = useCallback((patch: Partial<ImageGenerationProfile>) => {
    setState((current) => ({
      ...current,
      imageProfile: {
        ...current.imageProfile,
        ...patch
      }
    }));
  }, []);

  const persistPersonalApiVault = useCallback(
    (updater: (current: PersonalApiVault) => PersonalApiVault) => {
      const next = {
        ...updater(personalApiVaultRef.current),
        updatedAt: new Date().toISOString()
      };
      personalApiVaultRef.current = next;

      setPersonalApiVault(next);
      setState((activeState) => applyPersonalApiVault(activeState, next));

      savePersonalApiVault(next);
      const apiClient = createDynamicChatApiClient();
      void apiClient
        .savePersonalApiVault(next)
        .then(() => {
          const cachedState = loadState();
          if (cachedState) {
            return apiClient.saveSimulationState(applyPersonalApiVault(hydrateState(cachedState), next));
          }
          return undefined;
        })
        .catch(() => undefined);
    },
    []
  );

  const getActivePersonalApiVault = useCallback(() => {
    const freshVault = readFreshPersonalApiVault(personalApiVault);
    if (!arePersonalApiVaultsEqual(freshVault, personalApiVault)) {
      setPersonalApiVault(freshVault);
      savePersonalApiVault(freshVault);
    }
    return freshVault;
  }, [personalApiVault]);

  const savePersonalLlmSecret = useCallback(
    (provider: LlmApiSettings["provider"], settings: LlmApiSettings) => {
      const secretRecord = normalizeStoredSecretRecord(
        {
          apiKey: settings.apiKey,
          registrationStatus: settings.registrationStatus,
          verifiedAt: settings.verifiedAt,
          verificationMessage: settings.verificationMessage
        },
        "개인 LLM 키가 저장되어 있습니다."
      );
      persistPersonalApiVault((current) => ({
        ...current,
        llmByProvider: {
          ...current.llmByProvider,
          [provider]: secretRecord
        }
      }));
      showRuntimeNotice(`${getLlmProviderOption(provider).label} 키 설정이 적용되었습니다.`);
    },
    [persistPersonalApiVault, showRuntimeNotice]
  );

  const savePersonalNovelAiSecret = useCallback(
    (settings: NovelAiApiSettings) => {
      const secretRecord = normalizeStoredNovelAiSecretRecord(
        {
          apiKey: settings.apiKey,
          registrationStatus: settings.registrationStatus,
          verifiedAt: settings.verifiedAt,
          verificationMessage: settings.verificationMessage,
          subscriptionTier: settings.subscriptionTier
        }
      );
      persistPersonalApiVault((current) => ({
        ...current,
        novelAi: secretRecord
      }));
      showRuntimeNotice("NovelAI 토큰 설정이 적용되었습니다.");
    },
    [persistPersonalApiVault, showRuntimeNotice]
  );


  const savePersonalImageStoragePath = useCallback(
    (imageStoragePath: string) => {
      persistPersonalApiVault((current) => ({
        ...current,
        imageStoragePath: imageStoragePath.trim()
      }));
      showRuntimeNotice(imageStoragePath.trim() ? "이미지 저장 경로가 적용되었습니다." : "이미지 저장 경로를 기본값으로 되돌렸습니다.");
    },
    [persistPersonalApiVault, showRuntimeNotice]
  );

  const resolveStateWithRuntimeSecrets = useCallback(
    // `syncActiveState: false` is for a state that is not the active run (the builder's profile test): a recovered
    // vault is still saved, and the vault effects bring it to the active simulation the usual way, but this call
    // itself leaves App state alone.
    async (sourceState: AppState, options: { syncActiveState?: boolean } = {}): Promise<AppState> => {
      let nextVault = personalApiVault;
      let nextState = applyPersonalApiVault(sourceState, nextVault);
      const needsLlmSecret = nextState.llm.provider !== "mock" && !hasStoredSecret(nextState.llm);
      const needsNovelAiSecret = nextState.novelAi.enabled && nextState.novelAi.requestMode !== "mock" && !hasStoredSecret(nextState.novelAi);

      if (!needsLlmSecret && !needsNovelAiSecret) {
        return nextState;
      }

      try {
        const serverVaultPayload = await createDynamicChatApiClient().loadPersonalApiVault<Partial<PersonalApiVault>>();
        if (!serverVaultPayload) {
          return nextState;
        }

        const mergedVault = mergePersonalApiVaults(nextVault, hydratePersonalApiVault(serverVaultPayload));
        const mergedState = applyPersonalApiVault(sourceState, mergedVault);
        const recoveredLlm = needsLlmSecret && hasStoredSecret(mergedState.llm);
        const recoveredNovelAi = needsNovelAiSecret && hasStoredSecret(mergedState.novelAi);

        if (recoveredLlm || recoveredNovelAi) {
          savePersonalApiVault(mergedVault);
          setPersonalApiVault(mergedVault);
          if (options.syncActiveState !== false) {
            setState((current) => applyPersonalApiVault(current, mergedVault));
          }
          showRuntimeNotice(
            [
              recoveredLlm ? `${getLlmProviderOption(mergedState.llm.provider).label} 키 복원` : undefined,
              recoveredNovelAi ? "NovelAI 토큰 복원" : undefined
            ]
              .filter(Boolean)
              .join(" · ")
          );
          nextVault = mergedVault;
          nextState = mergedState;
        }
      } catch {
        return nextState;
      }

      return nextState;
    },
    [personalApiVault, setState, showRuntimeNotice]
  );

  // The builder's profile-image test. A draft is not a simulation yet and is never the active run, so the test runs
  // against a throwaway state built from it (createDraftPreviewState keeps the draft's character ids) with this
  // person's secrets applied. Nothing here writes App state: the candidates go back to the builder, which holds them
  // in component state only. The work still joins the renderer's serial chain, and photoreal work takes the local GPU
  // lock, because a test spends the same NovelAI account and the same GPU as the story's cuts.
  const testGenerateCharacterImage = useCallback(
    async (request: CharacterProfileTestRequest): Promise<CharacterProfileTestResult> => {
      const { signal } = request;
      signal.throwIfAborted();
      const runtimeState = await resolveStateWithRuntimeSecrets(createDraftPreviewState(request.draft), { syncActiveState: false });
      signal.throwIfAborted();
      const jobs = planCharacterProfileTestJobs(runtimeState, {
        characterId: request.draftCharacterId,
        backend: request.backend,
        prompt: request.prompt,
        negativePrompt: request.negativePrompt,
        variant: request.variant,
        count: request.count
      });
      // The candidates run where the panel asked, or not at all. The planner already refuses a photoreal request the
      // safety floor would move to NovelAI — the panel labels a photoreal test free, so it must never quietly spend
      // Anlas — and checking every job here keeps that true whatever the planner hands back. Thrown before
      // onPlanned, so the panel shows it as a refusal (nothing ran), not as a failed run.
      const backend = request.backend;
      const moved = jobs.find((job) => readImageJobBackend(job) !== backend);
      if (moved) {
        throw new Error([`이 프롬프트는 ${imageBackendLabels[backend]}로 생성할 수 없습니다.`, moved.routing?.reason].filter(Boolean).join(" "));
      }
      request.onPlanned?.({ total: jobs.length });

      // Started means generating, not merely out of the chain: a photoreal test can still wait for the local GPU behind
      // a turn after that, and a cancel during that wait must answer at once (awaitQueuedWork), not when the lock comes.
      let started = false;
      const notRun = (): { assets: ImageAsset[]; errors: string[] } => ({ assets: [], errors: [] });
      const render = async (): Promise<{ assets: ImageAsset[]; errors: string[] }> => {
        // Cancelled while it waited in the chain: the caller has already been answered, so hand the slot straight on.
        if (signal.aborted) {
          return notRun();
        }
        const run = () => {
          started = true;
          request.onStart?.();
          return runCharacterProfileTestJobs(runtimeState, jobs, {
            signal,
            onImage: (asset) => request.onImage?.(asset),
            onSettled: request.onSettled
          });
        };
        if (backend !== "photoreal") {
          return run();
        }
        // The lock decides from the LLM the App's turns run on, since that is the model that may be holding the card,
        // and from the draft's photoreal settings, which are the ones this render uses. A test cancelled while it
        // waited is skipped when its turn comes, before Ollama is unloaded for it.
        const live = latestStateRef.current;
        return localGpu.run("image", { ...runtimeState, llm: live.llm, imageTagLlm: live.imageTagLlm }, run, {
          shouldRun: () => !signal.aborted,
          onSkip: notRun
        });
      };
      const result = await awaitQueuedWork(chainImageBackendWork(imageJobQueuesRef.current, backend, render), signal, () => started);
      return { ...result, backend };
    },
    [localGpu, resolveStateWithRuntimeSecrets]
  );

  const updateLlmSettings = useCallback((patch: Partial<LlmApiSettings>) => {
    setState((current) => {
      const provider = patch.provider ?? current.llm.provider;
      const personalSecret = patch.provider && patch.apiKey === undefined ? personalApiVault.llmByProvider[provider] : undefined;
      const personalSecretPatch = personalSecret
        ? {
            apiKey: personalSecret.apiKey,
            registrationStatus: personalSecret.registrationStatus,
            verifiedAt: personalSecret.verifiedAt,
            verificationMessage: personalSecret.verificationMessage
          }
        : patch.provider && patch.apiKey === undefined
          ? {
              apiKey: "",
              verifiedAt: undefined,
              verificationMessage: ""
            }
          : {};
      const nextState = {
        ...current,
        llm: {
          ...current.llm,
          ...patch,
          ...personalSecretPatch
        }
      };
      const shouldAudit = patch.registrationStatus && patch.registrationStatus !== current.llm.registrationStatus;
      return shouldAudit
        ? {
            ...nextState,
            auditLog: [
              ...nextState.auditLog,
              createAuditEvent(nextState, "api_secret_verified", "api_secret", "llm", {
                provider: nextState.llm.provider,
                model: nextState.llm.model,
                status: patch.registrationStatus
              })
            ]
          }
        : nextState;
    });
  }, [personalApiVault.llmByProvider]);

  const updateImageTagLlmSettings = useCallback((patch: Partial<LlmApiSettings>) => {
    setState((current) => ({
      ...current,
      imageTagLlm: {
        ...current.imageTagLlm,
        ...patch
      }
    }));
  }, []);

  const updateNovelAiSettings = useCallback((patch: Partial<NovelAiApiSettings>) => {
    // Vibe transfer is a GLOBAL personal setting. Route those fields to the vault (persistPersonalApiVault
    // re-applies the vault onto the active state, so the editor and generation see them immediately and
    // every other simulation inherits them). Other NovelAI fields stay per-simulation.
    if ("vibeTransferEnabled" in patch || "vibeTransferReferences" in patch) {
      persistPersonalApiVault((current) => ({
        ...current,
        vibeTransfer: normalizeStoredVibeTransferSettings({
          enabled: patch.vibeTransferEnabled ?? current.vibeTransfer.enabled,
          references: patch.vibeTransferReferences ?? current.vibeTransfer.references
        })
      }));
    }
    const rest = { ...patch };
    delete rest.vibeTransferEnabled;
    delete rest.vibeTransferReferences;
    if (Object.keys(rest).length === 0) {
      return;
    }
    setState((current) => {
      const nextState = {
        ...current,
        novelAi: {
          ...current.novelAi,
          ...rest
        }
      };
      const shouldAudit = rest.registrationStatus && rest.registrationStatus !== current.novelAi.registrationStatus;
      return shouldAudit
        ? {
            ...nextState,
            auditLog: [
              ...nextState.auditLog,
              createAuditEvent(nextState, "api_secret_verified", "api_secret", "novelai", {
                provider: "novelai",
                accountLabel: nextState.novelAi.accountLabel,
                status: rest.registrationStatus
              })
            ]
          }
        : nextState;
    });
  }, [persistPersonalApiVault]);

  const runQueuedImageJob = useCallback((job: ImageGenerationJob, snapshot: AppState, confirmed = false): Promise<void> => {
    const backend = readImageJobBackend(job);
    const backendLabel = imageBackendLabels[backend];
    const run = async () => {
    // Image generation is the longest-running async path in the app — a NovelAI round-trip or a local GPU render
    // behind a serialized queue — so it is the most likely to outlive a navigation, and it was the only major path
    // with no ownership check at all. Every setState below is gated on the job still belonging to the open run;
    // without this, assets and jobs from simulation A appeared in simulation B's library.
    const jobOwner = readRunOwner(snapshot);
    const confirmedJob = confirmed
      ? {
          ...job,
          providerPayload: {
            ...job.providerPayload,
            requiresConfirmation: false,
            confirmedAt: new Date().toISOString()
          }
        }
      : job;

    if (!shouldAutoRunImageJob(confirmedJob, snapshot)) {
      return;
    }

    // The job may have waited behind earlier cuts for a while, and the live job can have moved on since this
    // dispatch was queued: the user left the run, cancelled the cut, or regenerated it — a regeneration reuses
    // the id and queues its own run (possibly on the other backend's chain). Running this copy then would spend a
    // paid NovelAI request or a local GPU render on a result every write below discards, or execute the stale
    // plan next to the new one and clobber it. A live copy NEWER than this one means superseded. An equally old
    // one only counts when cancelled; an older cancelled one is the regeneration of a cancelled cut, whose fresh
    // copy the live view simply has not caught up with yet.
    const liveState = latestStateRef.current;
    const liveJob = liveState.imageJobs.find((candidate) => candidate.id === confirmedJob.id);
    const liveUpdatedAt = liveJob?.updatedAt ?? "";
    const queuedUpdatedAt = confirmedJob.updatedAt ?? "";
    const superseded =
      Boolean(liveJob) && (liveUpdatedAt > queuedUpdatedAt || (liveJob?.status === "canceled" && liveUpdatedAt === queuedUpdatedAt));
    if (!ownsActiveRun(liveState, jobOwner) || superseded) {
      return;
    }

    // This dispatch's identity. "The record is not cancelled" cannot tell whose write is whose: a regeneration
    // reuses the job id and may run on the other renderer's chain while this copy still renders, and a cancelled
    // cut can be regenerated before its own request returns. With one chain for every renderer the regeneration
    // always ran last, so it won; with a chain per renderer the stale result could land on top of it. So the
    // dispatch claims the record by stamping this token on it, and each later write lands only while the record
    // still carries it.
    const dispatchToken = createId("image_dispatch");
    const controller = new AbortController();
    const dispatches = imageJobDispatchesRef.current;
    // A dispatch of the same job that is still running is one this copy supersedes; its result could only be
    // discarded, so its request is stopped rather than left to finish.
    dispatches.get(confirmedJob.id)?.controller.abort();
    dispatches.set(confirmedJob.id, { token: dispatchToken, owner: jobOwner, controller });
    const isCurrentDispatch = () => !controller.signal.aborted && dispatches.get(confirmedJob.id)?.token === dispatchToken;
    const executableJob: ImageGenerationJob = {
      ...confirmedJob,
      providerPayload: { ...confirmedJob.providerPayload, dispatchToken }
    };
    const ownsRecord = (record: ImageGenerationJob | undefined) =>
      Boolean(record) && record?.status !== "canceled" && readImageJobDispatchToken(record) === dispatchToken;

    try {
      // Claims the record when the work actually starts. Judged on the authoritative state, not on the live view
      // above, which can trail a cancel or a regeneration by a render: a record cancelled since, or replaced by a
      // newer copy, is left alone, and then nothing this dispatch does afterwards can reach it either.
      const claimJob = () => {
        setState((current) => {
          if (!ownsActiveRun(current, jobOwner)) {
            return current;
          }
          const record = current.imageJobs.find((candidate) => candidate.id === executableJob.id);
          if (!record || record.status === "canceled" || (record.updatedAt ?? "") > queuedUpdatedAt) {
            return current;
          }
          return {
            ...current,
            imageJobs: current.imageJobs.map((candidate) =>
              candidate.id === executableJob.id
                ? { ...candidate, ...executableJob, status: "generating", updatedAt: new Date().toISOString() }
                : candidate
            )
          };
        });
      };

      const markJob = (patch: Partial<ImageGenerationJob>) => {
        setState((current) => {
          if (!ownsActiveRun(current, jobOwner) || !ownsRecord(current.imageJobs.find((candidate) => candidate.id === executableJob.id))) {
            return current;
          }
          return {
            ...current,
            imageJobs: current.imageJobs.map((candidate) =>
              candidate.id === executableJob.id
                ? {
                    ...candidate,
                    ...patch,
                    updatedAt: new Date().toISOString()
                  }
                : candidate
            )
          };
        });
      };

      // Asked again once the local GPU is this cut's, before anything is unloaded for it: a render can wait a minute
      // behind a local LLM turn, and its cut can be cancelled, regenerated or left behind meanwhile. A cut nobody
      // wants any more must not evict the LLM it was waiting on, nor hold the lock for the next turn.
      const stillWanted = () => {
        if (!isCurrentDispatch()) {
          return false;
        }
        const live = latestStateRef.current;
        if (!ownsActiveRun(live, jobOwner)) {
          return false;
        }
        const record = live.imageJobs.find((candidate) => candidate.id === executableJob.id);
        // The live view may not show the record yet, or still show the queued copy (the claim comes after this);
        // it only has to show no cancel and no newer copy.
        return !record || ownsRecord(record) || (record.status !== "canceled" && (record.updatedAt ?? "") <= queuedUpdatedAt);
      };

      // Precise Reference costs Anlas per image, so a queued NovelAI cut follows the toggle as it is when the cut
      // runs, not as it was when its turn queued it: unticking it has to stop the cuts still waiting on the chain.
      // Same run as the snapshot (checked above), so these are this simulation's own settings.
      const liveNovelAi = liveState.novelAi;
      const executionState: AppState =
        backend === "novelai"
          ? {
              ...snapshot,
              novelAi: {
                ...snapshot.novelAi,
                characterReferenceEnabled: liveNovelAi.characterReferenceEnabled,
                characterReferenceStrength: liveNovelAi.characterReferenceStrength,
                characterReferenceFidelity: liveNovelAi.characterReferenceFidelity,
                characterReferenceStyleAware: liveNovelAi.characterReferenceStyleAware
              }
            }
          : snapshot;

      const execute = (): Promise<ImageJobExecutionResult> => {
        claimJob();
        return executeImageJob(executionState, executableJob, {
          // Cancel, regeneration and a run switch abort this; a cancelled request comes back `canceled`, and for
          // photoreal the relay interrupts the ComfyUI prompt when the request goes away.
          signal: controller.signal,
          onProgress: async (progress) => {
            // A dispatch that was cancelled or superseded meanwhile stores nothing: its images would be orphaned.
            if (!isCurrentDispatch()) {
              return;
            }
            const progressAssets = await persistRuntimeImagePayloads(snapshot.simulation.id, progress.assets);
            const progressAssetIds = progressAssets.map((asset) => asset.id);
            const progressJob = {
              ...progress.job,
              assetIds: progressAssetIds,
              representativeAssetId: progressAssets[0]?.id ?? progress.job.representativeAssetId
            };
            setState((current) => {
              if (!ownsActiveRun(current, jobOwner) || !ownsRecord(current.imageJobs.find((candidate) => candidate.id === progressJob.id))) {
                return current;
              }

              const nextState = {
                ...current,
                imageJobs: current.imageJobs.map((candidate) =>
                  candidate.id === progressJob.id
                    ? {
                        ...candidate,
                        ...progressJob
                      }
                    : candidate
                ),
                imageAssets: upsertImageAssets(current.imageAssets, progressAssets),
                messages: current.messages.map((message) =>
                  message.id === progressJob.turnId
                    ? {
                        ...message,
                        imageAssetIds: mergeGeneratedMessageImageAssetIds(message.imageAssetIds, current.imageAssets, progressAssets)
                      }
                    : message
                ),
                turnTraces: current.turnTraces.map((trace) => {
                  if (trace.imageJobId !== progressJob.id && trace.assistantMessageId !== progressJob.turnId) {
                    return trace;
                  }

                  const nextImageAssetIds = uniqueIds([...trace.imageAssetIds, ...progressAssetIds]);
                  return {
                    ...trace,
                    imageAssetIds: nextImageAssetIds,
                    metrics: {
                      ...trace.metrics,
                      imageAssetCount: Math.max(trace.metrics.imageAssetCount, nextImageAssetIds.length)
                    }
                  };
                })
              };
              scheduleStatePersistence(nextState);
              return nextState;
            });
          }
        });
      };

      let result: ImageJobExecutionResult | undefined;
      try {
        // Only a photoreal render takes the local GPU lock: NovelAI renders on NovelAI's servers, and holding the
        // lock for it would stall a local LLM turn for nothing. The coordinator itself decides whether this machine
        // needs a handoff at all (Ollama and ComfyUI both local); otherwise run() just runs the render. While it
        // waits for the lock the cut stays "queued" (it is claimed as "generating" only once it starts), and
        // stillWanted() skips it when its turn comes if it is no longer wanted.
        result =
          backend === "photoreal"
            ? await localGpu.run<ImageJobExecutionResult | undefined>("image", snapshot, execute, {
                shouldRun: stillWanted,
                onSkip: () => undefined
              })
            : await execute();
      } catch (error) {
        if (!isCurrentDispatch()) {
          return;
        }
        const message = error instanceof Error && error.message.trim() ? error.message : `${backendLabel} 이미지 생성 중 알 수 없는 오류`;
        markJob({
          status: "failed",
          error: message,
          completedAt: new Date().toISOString()
        });
        showRuntimeNotice(`${backendLabel} 이미지 생성 실패: ${stripImageFailureLead(message, backend) || "알 수 없는 오류"}`);
        return;
      }

      // Skipped on its GPU turn, cancelled, or superseded while it ran: nothing of this dispatch reaches the cut —
      // not its status, not its images, and no failure toast for a cut the user has already moved on from.
      if (!result || result.job.status === "canceled" || !isCurrentDispatch()) {
        return;
      }

      // executeImageJob reports provider errors by RETURNING a failed job rather than throwing (the policy-block
      // and partial-variant paths both do), so the catch above never saw them. Those failures were completely
      // silent in auto mode — the only trace was a status chip in a panel the user may not have open.
      if (result.job.status === "failed") {
        showRuntimeNotice(`${backendLabel} 이미지 생성 실패: ${stripImageFailureLead(result.job.error ?? "", backend) || "알 수 없는 오류"}`);
      }

      const resultAssets = await persistRuntimeImagePayloads(snapshot.simulation.id, result.assets);
      const finalResult: ImageJobExecutionResult = {
        ...result,
        assets: resultAssets,
        job: {
          ...result.job,
          assetIds: resultAssets.map((asset) => asset.id),
          representativeAssetId: resultAssets[0]?.id ?? result.job.representativeAssetId
        }
      };

      setState((current) => {
        // Not this dispatch's record any more (cancelled, or replaced by a regeneration): the result is dropped
        // whole, its images included, so the cut keeps showing what the user asked for last.
        if (!ownsActiveRun(current, jobOwner) || !ownsRecord(current.imageJobs.find((candidate) => candidate.id === executableJob.id))) {
          return current;
        }

        const nextState = {
          ...current,
          imageJobs: current.imageJobs.map((candidate) => (candidate.id === finalResult.job.id ? finalResult.job : candidate)),
          imageAssets: upsertImageAssets(current.imageAssets, finalResult.assets),
          auditLog: [
            ...current.auditLog,
            createAuditEvent(current, "generation_job_completed", "image_job", finalResult.job.id, {
              status: finalResult.job.status,
              assetCount: finalResult.assets.length,
              error: finalResult.job.error
            })
          ],
          turnTraces: current.turnTraces.map((trace) => {
            if (trace.imageJobId !== finalResult.job.id && trace.assistantMessageId !== finalResult.job.turnId) {
              return trace;
            }

            const nextImageAssetIds = uniqueIds([...trace.imageAssetIds, ...finalResult.assets.map((asset) => asset.id)]);
            return {
              ...trace,
              imageAssetIds: nextImageAssetIds,
              metrics: {
                ...trace.metrics,
                imageAssetCount: nextImageAssetIds.length,
                imageEstimatedCost: readProviderCost(finalResult.job.providerPayload) ?? trace.metrics.imageEstimatedCost
              }
            };
          }),
          messages: current.messages.map((message) =>
            message.id === finalResult.job.turnId
              ? {
                  ...message,
                  imageAssetIds: mergeGeneratedMessageImageAssetIds(message.imageAssetIds, current.imageAssets, finalResult.assets)
                }
              : message
          )
        };
        scheduleStatePersistence(nextState, { includeImagePayloads: finalResult.assets.some((asset) => Boolean(asset.dataUrl)) });
        return nextState;
      });
    } finally {
      if (dispatches.get(confirmedJob.id)?.token === dispatchToken) {
        dispatches.delete(confirmedJob.id);
      }
    }
    };

    return chainImageBackendWork(imageJobQueuesRef.current, backend, run);
  }, [localGpu, scheduleStatePersistence, showRuntimeNotice]);

  // Leaving a run abandons its image work: every write of a running cut is gated on the run it started in, so a
  // render that outlived the switch would only spend the GPU or the NovelAI account on a picture nobody sees, and
  // keep the next run's cuts and turns waiting behind it.
  const activeRunSimulationId = state.simulation.id;
  const activeRunProgressId = state.activeProgressRunId;
  useEffect(() => {
    const activeOwner: RunOwner = { simulationId: activeRunSimulationId, progressRunId: activeRunProgressId };
    const dispatches = imageJobDispatchesRef.current;
    dispatches.forEach((dispatch, jobId) => {
      if (dispatch.owner.simulationId !== activeOwner.simulationId || dispatch.owner.progressRunId !== activeOwner.progressRunId) {
        dispatches.delete(jobId);
        dispatch.controller.abort();
      }
    });
    // The other half, for the run now open: a cut a dispatch had claimed as generating, with no dispatch left to
    // finish it — its render was abandoned when the user left this run, or the page was reloaded mid-render. Its
    // record still says "생성 중" and would say so forever, with a spinner and no way to regenerate. It was in fact
    // stopped, so it is marked as stopped. Safe against a render that is finishing right now: a dispatch holds the
    // job in `dispatches` until after its final write is queued, and that write is applied before this one.
    setState((current) => {
      if (!ownsActiveRun(current, activeOwner) || !current.imageJobs.some((job) => job.status === "generating" && !dispatches.has(job.id))) {
        return current;
      }
      const stoppedAt = new Date().toISOString();
      return {
        ...current,
        imageJobs: current.imageJobs.map((job) =>
          job.status === "generating" && !dispatches.has(job.id)
            ? { ...job, status: "canceled", error: ABANDONED_IMAGE_JOB_ERROR, completedAt: stoppedAt, updatedAt: stoppedAt }
            : job
        )
      };
    });
    // storageReady re-runs this once the persisted copy has been restored, which can bring its own stranded cuts.
  }, [activeRunSimulationId, activeRunProgressId, setState, storageReady]);

  const cancelImageJob = useCallback((jobId: string) => {
    // Stops the request as well as the record. Marking the cut cancelled used to leave a photoreal render running
    // to its end (or its timeout), holding the local GPU lock and the photoreal chain, so the next turn and every
    // later cut waited behind a picture that was then thrown away.
    const dispatch = imageJobDispatchesRef.current.get(jobId);
    if (dispatch) {
      imageJobDispatchesRef.current.delete(jobId);
      dispatch.controller.abort();
    }
    setState((current) => ({
      ...current,
      imageJobs: current.imageJobs.map((job) =>
        job.id === jobId && ["queued", "planning", "generating"].includes(job.status)
          ? {
              ...job,
              status: "canceled",
              error: "Canceled by user",
              completedAt: new Date().toISOString(),
              updatedAt: new Date().toISOString()
            }
          : job
      ),
      auditLog: [
        ...current.auditLog,
        createAuditEvent(current, "generation_job_completed", "image_job", jobId, {
          status: "canceled"
        })
      ]
    }));
  }, []);

  const regenerateImageJob = useCallback((job: ImageGenerationJob, backendOverride?: ImageBackendId) => {
    // The copy being replaced may still be running — a cut waiting for its turn on the local GPU, or a cancelled one
    // whose request has not come back — and its result could only be discarded now, so its request stops here.
    const previousDispatch = imageJobDispatchesRef.current.get(job.id);
    if (previousDispatch) {
      imageJobDispatchesRef.current.delete(job.id);
      previousDispatch.controller.abort();
    }
    const now = new Date().toISOString();
    // A regeneration redraws the SAME cut: on the renderer it was routed to (re-routing today's settings would
    // silently turn a photoreal cut into an anime one, or back), with its crop and beat kind. The user can ask
    // for the other renderer explicitly; the safety floor in decideImageBackend still beats that request.
    const originalBackend = readImageJobBackend(job);
    const requestedBackend = backendOverride ?? originalBackend;
    const regenerationCue = createImageCueForRegeneration(state, job);
    const replannedJob = planImageJob(state, job.turnId, regenerationCue, job.contextNodeIds, true, {
      backend: requestedBackend,
      cueKind: regenerationCue.kind
    });
    // Reuse the original job's id and cue position so the regenerated image REPLACES the failed one in
    // place (same slot) instead of appearing as an extra image elsewhere. The fresh prompt comes from the
    // re-plan; identity/turn linkage (turnId, cueIndex) is preserved.
    const nextJob: ImageGenerationJob = {
      ...replannedJob,
      id: job.id,
      turnId: job.turnId,
      status: "queued",
      assetIds: [],
      representativeAssetId: undefined,
      completedAt: undefined,
      error: undefined,
      createdAt: job.createdAt,
      updatedAt: now,
      providerPayload: {
        ...replannedJob.providerPayload,
        cueIndex: job.providerPayload?.cueIndex ?? replannedJob.providerPayload?.cueIndex,
        mode: "planned",
        requiresConfirmation: false,
        regeneratedFrom: job.id
      }
    };

    setState((current) => ({
      ...current,
      imageJobs: current.imageJobs.some((candidate) => candidate.id === job.id)
        ? current.imageJobs.map((candidate) => (candidate.id === job.id ? nextJob : candidate))
        : [...current.imageJobs, nextJob],
      auditLog: [
        ...current.auditLog,
        createAuditEvent(current, "generation_job_created", "image_job", nextJob.id, {
          regeneratedFrom: job.id,
          status: nextJob.status
        })
      ]
    }));
    void runQueuedImageJob(nextJob, state, true);
    const plannedBackend = readImageJobBackend(nextJob);
    if (backendOverride && plannedBackend !== backendOverride) {
      // The safety floor kept the cut where it was. Said plainly, because the button promised another renderer.
      pushRuntimeNotice(
        `이 컷은 ${imageBackendLabels[plannedBackend]}로 고정되어 있어 ${imageBackendLabels[plannedBackend]}로 다시 생성합니다. ${nextJob.routing?.reason ?? ""}`.trim(),
        "info"
      );
    } else {
      showRuntimeNotice(
        backendOverride && backendOverride !== originalBackend
          ? `${imageBackendLabels[plannedBackend]}로 이미지 재생성 작업을 시작했습니다.`
          : "이미지 재생성 작업을 시작했습니다."
      );
    }
  }, [pushRuntimeNotice, runQueuedImageJob, showRuntimeNotice, state]);

  const updateImageFeedback = useCallback((assetId: string, rating: ImageFeedbackRating) => {
    setState((current) => {
      const asset = current.imageAssets.find((candidate) => candidate.id === assetId);
      return {
        ...current,
        imageAssets: current.imageAssets.map((asset) => {
        if (asset.id !== assetId) {
          return asset;
        }
        const nextRating = asset.feedback?.rating === rating ? undefined : rating;
        return {
          ...asset,
          feedback: nextRating
            ? {
                rating: nextRating,
              updatedAt: new Date().toISOString()
            }
            : undefined
        };
        }),
        auditLog: asset
          ? [
              ...current.auditLog,
              createAuditEvent(current, "asset_accessed", "image_asset", assetId, {
                action: "feedback_updated",
                rating,
                previousRating: asset.feedback?.rating
              })
            ]
          : current.auditLog
      };
    });
  }, []);

  const redactMemoryEvent = useCallback((memoryId: string) => {
    setState((current) => {
      const memory = current.memoryEvents.find((event) => event.id === memoryId);
      if (!memory) {
        return current;
      }

      const nodeIds = [memory.id, memory.neuralMapNodeId].filter((nodeId): nodeId is string => Boolean(nodeId));
      const redaction = createRedactionRequest(current, "memory_event", memoryId, "operator memory redaction", nodeIds);
      return {
        ...current,
        memoryEvents: current.memoryEvents.filter((event) => event.id !== memoryId),
        contextPacks: current.contextPacks.map((pack) => ({
          ...pack,
          evidence: pack.evidence.filter((item) => !nodeIds.includes(item.nodeId))
        })),
        messages: current.messages.map((message) => ({
          ...message,
          referencedNodeIds: message.referencedNodeIds.filter((nodeId) => !nodeIds.includes(nodeId))
        })),
        turnTraces: current.turnTraces.map((trace) => ({
          ...trace,
          memoryEventIds: trace.memoryEventIds.filter((eventId) => eventId !== memoryId)
        })),
        redactionQueue: [...current.redactionQueue, redaction],
        auditLog: [...current.auditLog, createAuditEvent(current, "memory_redacted", "memory_event", memoryId, { nodeIds })]
      };
    });
  }, []);

  const redactImageAsset = useCallback((assetId: string) => {
    const activeAsset = collectStateImageAssets(state).find((candidate) => candidate.id === assetId);
    if (activeAsset) {
      const serverRedaction = createRedactionRequest(state, "image_asset", assetId, "operator asset deletion", activeAsset.objectKey ? [activeAsset.objectKey] : []);
      void createDynamicChatApiClient().createRedaction(state.simulation.id, serverRedaction).catch(() => undefined);
    }

    setState((current) => {
      const asset = collectStateImageAssets(current).find((candidate) => candidate.id === assetId);
      if (!asset) {
        return current;
      }

      const redaction = createRedactionRequest(current, "image_asset", assetId, "operator asset deletion", asset.objectKey ? [asset.objectKey] : []);
      return {
        ...current,
        imageAssets: current.imageAssets.filter((candidate) => candidate.id !== assetId),
        messages: current.messages.map((message) => ({
          ...message,
          imageAssetIds: message.imageAssetIds.filter((candidateId) => candidateId !== assetId)
        })),
        imageJobs: current.imageJobs.map((job) => ({
          ...job,
          assetIds: job.assetIds.filter((candidateId) => candidateId !== assetId),
          representativeAssetId: job.representativeAssetId === assetId ? undefined : job.representativeAssetId
        })),
        turnTraces: current.turnTraces.map((trace) => ({
          ...trace,
          imageAssetIds: trace.imageAssetIds.filter((candidateId) => candidateId !== assetId),
          metrics: {
            ...trace.metrics,
            imageAssetCount: Math.max(0, trace.metrics.imageAssetCount - (trace.imageAssetIds.includes(assetId) ? 1 : 0))
          }
        })),
        redactionQueue: [...current.redactionQueue, redaction],
        auditLog: [...current.auditLog, createAuditEvent(current, "image_asset_deleted", "image_asset", assetId, { source: asset.source, objectKey: asset.objectKey })]
      };
    });
  }, [state, setState]);

  const redactPromptModule = useCallback((moduleId: string) => {
    setState((current) => {
      const module = current.modules.find((candidate) => candidate.id === moduleId);
      if (!module || module.kind === "main_prompt") {
        return current;
      }

      const now = new Date().toISOString();
      const redaction = createRedactionRequest(current, "prompt_module", moduleId, "operator prompt module redaction", [moduleId]);
      return {
        ...current,
        modules: current.modules.map((candidate) =>
          candidate.id === moduleId
            ? {
                ...candidate,
                title: `${candidate.title} (redacted)`,
                body: "[redacted]",
                enabled: false,
                activationTags: [],
                tokenPolicy: "disabled",
                updatedAt: now
              }
            : candidate
        ),
        contextPacks: current.contextPacks.map((pack) => ({
          ...pack,
          evidence: pack.evidence.filter((item) => item.nodeId !== moduleId)
        })),
        selectedModuleId: current.selectedModuleId === moduleId ? current.modules.find((candidate) => candidate.id !== moduleId)?.id : current.selectedModuleId,
        redactionQueue: [...current.redactionQueue, redaction],
        auditLog: [...current.auditLog, createAuditEvent(current, "prompt_module_redacted", "prompt_module", moduleId, { kind: module.kind, title: module.title })]
      };
    });
  }, []);

  const addModule = useCallback(() => {
    const now = new Date().toISOString();
    const id = `module_${Date.now().toString(36)}`;
    setState((current) => ({
      ...current,
      selectedModuleId: id,
      modules: [
        ...current.modules,
        {
          id,
          simulationId: current.simulation.id,
          parentId: current.selectedModuleId,
          kind: "sub_prompt",
          title: "새 서브 프롬프트",
          body: "이 모듈에 장면, 규칙, 캐릭터, 이미지 생성 힌트 등을 작성하세요.",
          enabled: true,
          priority: 50,
          activationTags: ["new-module"],
          tokenPolicy: "rag",
          version: 1,
          updatedAt: now
        }
      ]
    }));
  }, []);

  const deleteModule = useCallback((moduleId: string) => {
    redactPromptModule(moduleId);
  }, [redactPromptModule]);

  const planAndQueueImageForTurn = useCallback(
    async (result: ImageDispatchTurn, snapshot: AppState, manual: boolean, preResolvedSnapshot?: AppState, options: { isExpansion?: boolean } = {}) => {
      const isExpansion = options.isExpansion === true;
      // Claim the initial cue positions SYNCHRONOUSLY, before any await, so the early (mid-stream) and
      // post-turn paths can never both pass the check and double-fire the paid provider. JS runs this loop
      // to completion without interruption; whichever path reaches it first owns those turn+cue slots.
      // Position-based keys line up because realtime turns keep their initial image_cues array stable.
      // Expansion handles distinct appended tail cues (their cueIndex restarts at 0) that the early path
      // never touched, so it keeps the original append behavior and is exempt from the claim.
      const claimTurnId = result.assistantMessage.id;
      // Image planning is fire-and-forget from the turn loop and involves an LLM round-trip plus provider
      // calls, so it routinely resolves long after the user has moved on. Without an owner it appended this
      // turn's jobs, traces and audit events into whatever run happened to be open — the most visible form of
      // "the run I started follows me into another simulation".
      const imageOwner = readRunOwner(snapshot);
      try {
        const runtimeSnapshot = preResolvedSnapshot ?? (await resolveStateWithRuntimeSecrets(snapshot));
        if (!ownsActiveRun(latestStateRef.current, imageOwner)) {
          // The user left this run before anything was dispatched — do not spend on provider calls at all.
          return;
        }
        const imagePlan = await planImageJobForCompletedTurn(runtimeSnapshot, {
          userMessage: result.userMessage,
          assistantMessage: result.assistantMessage,
          contextPack: result.contextPack,
          promptModuleUsages: result.promptModuleUsages,
          sidecar: result.sidecar,
          sidecarTrace: result.sidecarTrace,
          manualImage: manual,
          annotationFailureReason: result.annotationFailureReason
        });
        const plannedImageJobs = imagePlan.imageJobs.length > 0 ? imagePlan.imageJobs : imagePlan.imageJob ? [imagePlan.imageJob] : [];
        // Keep only jobs that haven't been dispatched yet (claimed via dispatchedImageCueKeysRef) so mid-stream
        // early dispatches are not re-fired, while all cadence-expanded paragraph cues are dispatched.
        const imageJobs: ImageGenerationJob[] = [];
        if (!isExpansion) {
          plannedImageJobs.forEach((job, index) => {
            const cueIndex = typeof (job.providerPayload as { cueIndex?: unknown }).cueIndex === "number"
              ? (job.providerPayload as { cueIndex: number }).cueIndex
              : index;
            const key = `${claimTurnId}:cue${cueIndex}`;
            if (!dispatchedImageCueKeysRef.current.has(key)) {
              dispatchedImageCueKeysRef.current.add(key);
              imageJobs.push(job);
            }
          });
        } else {
          imageJobs.push(...plannedImageJobs);
        }
        const primaryImageJob = imagePlan.imageJob && imageJobs.includes(imagePlan.imageJob) ? imagePlan.imageJob : imageJobs[0];
        const reusedAssetIds = imagePlan.reusedAssetIds ?? [];
        if (imageJobs.length === 0 && reusedAssetIds.length === 0 && plannedImageJobs.length > 0) {
          // Every renderable cue this turn was already dispatched early; nothing more to queue.
          return;
        }
        const imageEstimatedCost = imageJobs.reduce(
          (sum, job) => sum + (readProviderCost(job.providerPayload) ?? 0),
          0
        );
        // The engine leaves the trace's job counts at zero when planning is deferred, which is how the app runs every
        // turn, so the counts come from the jobs planned here; photorealJobCount used to stay 0 in every saved trace.
        const countPhotorealJobs = (jobs: ImageGenerationJob[]) => jobs.filter((job) => readImageJobBackend(job) === "photoreal").length;

        setState((current) => {
          if (!ownsActiveRun(current, imageOwner)) {
            return current;
          }
          const existingJobIds = new Set(current.imageJobs.map((job) => job.id));
          const newImageJobs = imageJobs.filter((job) => !existingJobIds.has(job.id));
          return {
          ...current,
          messages: reusedAssetIds.length > 0
            ? current.messages.map((message) =>
                message.id === result.assistantMessage.id
                  ? {
                      ...message,
                      imageAssetIds: uniqueIds([...message.imageAssetIds, ...reusedAssetIds])
                    }
                  : message
              )
            : current.messages,
          imageJobs: newImageJobs.length > 0 ? [...current.imageJobs, ...newImageJobs] : current.imageJobs,
          turnTraces: current.turnTraces.map((trace) =>
            trace.id === result.turnTrace.id
              ? {
                  ...trace,
                  // In expansion mode the primary cue/job belong to the initial plan; only append.
                  imageCue: isExpansion ? trace.imageCue ?? imagePlan.imageCue : imagePlan.imageCue,
                  imageJobId: isExpansion ? trace.imageJobId ?? primaryImageJob?.id : primaryImageJob?.id,
                  imageAssetIds: uniqueIds([...trace.imageAssetIds, ...reusedAssetIds]),
                  metrics: {
                    ...trace.metrics,
                    imageJobCount: isExpansion ? (trace.metrics.imageJobCount ?? 0) + newImageJobs.length : imageJobs.length,
                    photorealJobCount: isExpansion
                      ? (trace.metrics.photorealJobCount ?? 0) + countPhotorealJobs(newImageJobs)
                      : countPhotorealJobs(imageJobs),
                    imageAssetCount: Math.max(trace.metrics.imageAssetCount, uniqueIds([...trace.imageAssetIds, ...reusedAssetIds]).length),
                    imageEstimatedCost: isExpansion
                      ? (trace.metrics.imageEstimatedCost ?? 0) + imageEstimatedCost
                      : imageEstimatedCost > 0 ? imageEstimatedCost : trace.metrics.imageEstimatedCost
                  }
                }
              : trace
          ),
          auditLog: newImageJobs.length > 0
            ? [
                ...current.auditLog,
                ...newImageJobs.map((imageJob) => createAuditEvent(current, "generation_job_created", "image_job", imageJob.id, {
                  status: imageJob.status,
                  triggerMode: imageJob.providerPayload.triggerMode,
                  requiresConfirmation: imageJob.providerPayload.requiresConfirmation,
                  planner: readImageCuePlannerSource(imageJob.providerPayload) ?? "image-cue-sidecar"
                }))
              ]
            : current.auditLog
          };
        });

        // Announce a suppressed image whether or not the user asked for it explicitly. Gating this on `manual`
        // meant that in auto mode — where almost all images are produced — a cue that was blocked by policy or
        // arrived without renderable tags simply produced nothing, with no way to tell it apart from a turn
        // the model judged non-visual.
        if (imageJobs.length === 0 && reusedAssetIds.length === 0 && plannedImageJobs.length === 0) {
          const suppressionReason = imagePlan.imageCue.suppressionReason;
          if (manual) {
            showRuntimeNotice(suppressionReason ?? "이미지 cue가 렌더 가능한 NAI 태그를 만들지 못해 작업을 만들지 않았습니다.");
          } else if (suppressionReason) {
            showRuntimeNotice(`이미지 생성 건너뜀: ${suppressionReason}`);
          }
        }

        const runnableImageJobs = imageJobs.filter((job) => shouldAutoRunImageJob(job, runtimeSnapshot));
        if (runnableImageJobs.length > 0) {
          const snapshotWithImageJobs = {
            ...runtimeSnapshot,
            messages: reusedAssetIds.length > 0
              ? runtimeSnapshot.messages.map((message) =>
                  message.id === result.assistantMessage.id
                    ? {
                        ...message,
                        imageAssetIds: uniqueIds([...message.imageAssetIds, ...reusedAssetIds])
                      }
                    : message
                )
              : runtimeSnapshot.messages,
            imageJobs: [...runtimeSnapshot.imageJobs, ...imageJobs],
            turnTraces: runtimeSnapshot.turnTraces.map((trace) =>
              trace.id === result.turnTrace.id
                ? {
                    ...trace,
                    imageCue: imagePlan.imageCue,
                    imageJobId: primaryImageJob?.id,
                    imageAssetIds: uniqueIds([...trace.imageAssetIds, ...reusedAssetIds]),
                    metrics: {
                      ...trace.metrics,
                      imageJobCount: imageJobs.length,
                      photorealJobCount: countPhotorealJobs(imageJobs),
                      imageAssetCount: Math.max(trace.metrics.imageAssetCount, uniqueIds([...trace.imageAssetIds, ...reusedAssetIds]).length),
                      imageEstimatedCost:
                        imageEstimatedCost > 0 ? imageEstimatedCost : trace.metrics.imageEstimatedCost
                    }
                  }
                : trace
            )
          };

          // Returned, not voided, so the caller's `await imagePlanning` genuinely covers every job of this
          // turn. Previously auto-progress awaited `imageJobQueueRef.current` instead — a moving chain tail
          // that only covered the jobs already handed to the runner, so on the paragraph cadence (up to 8
          // cuts) most of the turn's images were outside the barrier and the loop raced ahead of them.
          await runRunnableImageJobs(runnableImageJobs, snapshotWithImageJobs, runQueuedImageJob);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "이미지 작업 계획 중 알 수 없는 오류가 발생했습니다.";
        showRuntimeNotice(`이미지 작업 계획 실패: ${message}`);
      }
    },
    [resolveStateWithRuntimeSecrets, runQueuedImageJob, showRuntimeNotice]
  );

  // Fired each time another cue closes in the streamed annotation reply — starts that cut's image while the
  // tag model is still writing the rest of the list, instead of after the whole call. The payload carries every
  // cue so far, so earlier ones keep their indices and the shared dispatch-key gate in planAndQueueImageForTurn
  // skips them; the post-turn pass skips all of them the same way.
  const dispatchEarlyTurnImages = useCallback(
    (payload: EarlyImageCuePayload, manual: boolean): Promise<void> => {
      if (payload.cues.length === 0) {
        return Promise.resolve();
      }
      return planAndQueueImageForTurn(
        {
          userMessage: payload.userMessage,
          assistantMessage: payload.assistantMessage,
          contextPack: payload.contextPack,
          promptModuleUsages: payload.promptModuleUsages,
          sidecar: payload.sidecar,
          // The turn trace does not exist yet; the post-turn pass records the job counts on it.
          turnTrace: { id: "" }
        },
        payload.state,
        manual,
        payload.state
      );
    },
    [planAndQueueImageForTurn]
  );

  // Plans/queues the initial images immediately, then appends any extra image_cues that the
  // background completion retry produces (realtime image pipeline: start fast, fill in the rest).
  const planTurnImagesWithExpansion = useCallback(
    async (result: TurnResult, snapshot: AppState, manual: boolean, preResolvedSnapshot?: AppState) => {
      const imageCues = result.sidecar?.imageCues ?? [];
      const isAutoActive =
        snapshot.imageProfile.enabled &&
        snapshot.simulation.realtimeImageEnabled &&
        (snapshot.imageProfile.triggerMode === "realtime_auto" || snapshot.imageProfile.triggerMode === "realtime_confirm");
      if (imageCues.length === 0 && !isAutoActive && !manual) {
        return;
      }
      await planAndQueueImageForTurn(result, snapshot, manual, preResolvedSnapshot);
    },
    [planAndQueueImageForTurn]
  );

  // Runs one simulation turn from arbitrary text. `baseState` lets a caller (e.g. auto-progress) thread the previous
  // turn's committed state forward so consecutive turns build on each other without waiting for React to flush.
  // Returns the committed turn state on success, or null on failure.
  const runTurnFromText = useCallback(
    async (
      text: string,
      options: { manualImage?: boolean; baseState?: AppState; awaitImages?: boolean; suppressAutoReset?: boolean } = {}
    ): Promise<AppState | null> => {
      const shouldPlanManualImage = options.manualImage ?? false;
      const sourceState = options.baseState ?? state;
      // One controller per turn, so 중지 actually aborts the provider request instead of merely ignoring
      // its result. Cleared in the finally block below.
      const turnAbortController = new AbortController();
      turnAbortRef.current = turnAbortController;
      setActiveTurnAbortSignal(turnAbortController.signal);
      let streamedTurnMessageIds: string[] = [];
      // Hoisted so the catch can reach it. An auto session reset changes the session id but never the
      // simulation or progress-run id, so this is the same owner the guards inside the try use.
      const runOwner = readRunOwner(sourceState);
      setIsSending(true);
      setActiveTurnOwner(runOwner);
      setPendingUserText(text);
      setTurnPhase("retrieving");
      try {
        const runtimeState = await resolveStateWithRuntimeSecrets(sourceState);
        const resetDecision = createAutoResetAgentSessionDecision(runtimeState, text);
        // Auto-progress keeps the whole run in ONE session: a mid-run handoff reset drops the rich recent
        // transcript for a thin handoff summary, which is what makes the self-driven narrative drift and the
        // dialogue degrade as the run goes on. The caller (startAutoProgress) suppresses the reset so the
        // continuation context stays intact across the planned turns.
        const shouldReset = resetDecision.shouldReset && options.suppressAutoReset !== true;
        const turnBaseState = shouldReset
          ? await createResetSessionState(runtimeState)
          : runtimeState;
        const autoResetApplied = turnBaseState.simulation.activeSessionId !== runtimeState.simulation.activeSessionId;
        // This turn belongs to ONE simulation AND one progress run. If the user navigates elsewhere while the
        // turn is still streaming/committing, every setState below must no-op instead of merging this turn's
        // messages/result into whatever run is now active (otherwise the old run "continues" in the new one).
        // Both ids are stable across an auto session reset.
        const turnOwner = readRunOwner(turnBaseState);
        // The LLM part of the turn (narrative + annotation pass) holds the local GPU lock, so on a 16 GB card a
        // local Ollama model never loads while a photoreal render owns the VRAM, and the reverse. Image planning
        // below stays outside it: photoreal jobs take the lock themselves, after this turn has released it.
        let turnStarted = false;
        // Cuts dispatched while the annotation reply is still streaming. Kept so auto-progress can wait on
        // them like the post-turn jobs; each one is gated on the turn still being live.
        const earlyImageDispatches: Promise<void>[] = [];
        const result = await awaitQueuedWork(
          localGpu.run("llm", turnBaseState, () => {
            turnStarted = true;
            if (turnAbortController.signal.aborted) {
              return Promise.reject(createQueuedWorkCancelledError());
            }
            return runSimulationTurn(turnBaseState, text, shouldPlanManualImage, {
              deferImagePlanning: true,
              deferMemoryIngest: true,
              onEarlyImageCues: (payload) => {
                if (turnAbortController.signal.aborted || !ownsActiveRun(latestStateRef.current, turnOwner)) {
                  return;
                }
                earlyImageDispatches.push(dispatchEarlyTurnImages(payload, shouldPlanManualImage));
              },
              // Retrieval/module selection finished; the long part from here is the LLM call itself, so flip the card
              // to "generating" now instead of waiting for the first streamed assistant_text. Surface unusually slow
              // retrieval (e.g. an unreachable NeuralMap).
              onGenerationStart: ({ retrievalLatencyMs }) => {
                setTurnPhase("generating");
                if (retrievalLatencyMs > 12_000) {
                  showRuntimeNotice(`문맥 검색이 ${Math.round(retrievalLatencyMs / 1000)}초 걸렸습니다. NeuralMap 서버 상태를 확인하세요.`);
                }
              },
              onAssistantText: ({ userMessage, assistantMessage }) => {
                setTurnPhase("generating");
                // Remembered so a cancel can roll the partial turn back out: the narrative streams into state
                // every ~220ms, so by the time the user hits 중지 both messages are already on screen, and
                // leaving them there commits a truncated turn as if it had completed.
                streamedTurnMessageIds = [userMessage.id, assistantMessage.id];
                setState((current) => {
                  if (!ownsActiveRun(current, turnOwner)) {
                    return current;
                  }
                  const baseState =
                    autoResetApplied && current.simulation.activeSessionId !== turnBaseState.simulation.activeSessionId
                      ? layerSessionResetOntoLiveState(current, turnBaseState)
                      : current;
                  return upsertStreamingTurnMessages(baseState, userMessage, assistantMessage);
                });
              },
              onMemoryIngested: ({ turnId, memoryEvents, memoryIngestMs }) => {
                setState((current) =>
                  !ownsActiveRun(current, turnOwner)
                    ? current
                    : applyMemoryIngestResultToState(current, turnId, memoryEvents, memoryIngestMs)
                );
              }
            });
          }),
          turnAbortController.signal,
          () => turnStarted
        );
        const committedTurnState = applyTurnResultToState(turnBaseState, result);
        setState((current) => {
          if (!ownsActiveRun(current, turnOwner)) {
            return current;
          }
          const baseState = autoResetApplied ? layerSessionResetOntoLiveState(current, turnBaseState) : current;
          return applyTurnResultToState(baseState, result);
        });
        // Committed: from here a throw must not take the turn back out of the transcript.
        streamedTurnMessageIds = [];
        // Use the committed turn state (it carries this turn's new Wearing/state memory events and
        // inherits the resolved runtime secrets) so the image reflects outfit changes from this turn.
        setTurnPhase("images");
        const imagePlanning = planTurnImagesWithExpansion(result, committedTurnState, shouldPlanManualImage, committedTurnState);
        if (options.awaitImages) {
          // Auto-progress pacing: wait for THIS turn's images to be planned, queued, and generated before the
          // caller starts the next turn. Without it the loop fires image planning fire-and-forget and races
          // ahead, flooding the serial image queue (cadence "paragraph" can emit up to 8 cues/turn) so most
          // intermediate-turn jobs stay "queued" forever and only the final turn renders. Failures are surfaced
          // per job, so never let them reject the turn.
          try {
            await Promise.all([imagePlanning, ...earlyImageDispatches]);
          } catch {
            /* per-job errors already handled in runQueuedImageJob */
          }
        } else {
          void imagePlanning;
        }
        if (autoResetApplied) {
          showRuntimeNotice(`자동 handoff: ${resetDecision.reason}. 새 세션으로 이어갑니다.`);
        }
        if (result.sidecarTrace.source !== "llm") {
          showRuntimeNotice(createLlmFallbackNotice("LLM 응답 fallback", result.sidecarTrace));
        }
        // The annotation pass authors both the image cues and the state deltas, so its failure produces a
        // turn with no image and no state update that is otherwise indistinguishable from a non-visual beat.
        if (result.annotationFailureReason) {
          pushRuntimeNotice(result.annotationFailureReason, "error");
        }
        return committedTurnState;
      } catch (error) {
        // Whatever streamed in is not a turn unless it was committed, so drop it on a failure as on a cancel.
        // Only the cancel used to: a model refusal streamed into the transcript, stayed there cut mid-sentence
        // with no trace or memory behind it, and the next turn read it back as story. The input stays in the
        // composer (handleSubmit clears it only on a commit), so the user can resend or switch backends.
        if (streamedTurnMessageIds.length > 0) {
          const discardedIds = new Set(streamedTurnMessageIds);
          setState((current) =>
            ownsActiveRun(current, runOwner)
              ? { ...current, messages: current.messages.filter((message) => !discardedIds.has(message.id)) }
              : current
          );
        }
        if (isLlmAbortedError(error) || turnAbortController.signal.aborted) {
          // A cancel is a user decision, not a failure — say so plainly and do not colour it as an error.
          pushRuntimeNotice("진행을 취소했습니다.", "info");
          return null;
        }
        const message = error instanceof Error ? error.message : "시뮬레이션 턴 실행 중 알 수 없는 오류가 발생했습니다.";
        showRuntimeNotice(`시뮬레이션 진행 실패: ${message}`);
        return null;
      } finally {
        if (turnAbortRef.current === turnAbortController) {
          turnAbortRef.current = undefined;
          setActiveTurnAbortSignal(undefined);
        }
        setPendingUserText("");
        setIsSending(false);
        setActiveTurnOwner(undefined);
        setTurnPhase(undefined);
      }
    },
    [dispatchEarlyTurnImages, localGpu, planTurnImagesWithExpansion, resolveStateWithRuntimeSecrets, showRuntimeNotice, state]
  );

  const handleSubmit = useCallback(
    async (event: FormEvent) => {
      event.preventDefault();
      if (isSending || isResetting || isAutoProgressing) {
        return;
      }
      const text = createTurnSubmissionText(draft);
      const committed = await runTurnFromText(text, { manualImage });
      if (committed) {
        setDraft("");
        setManualImage(false);
      }
    },
    [draft, isAutoProgressing, isResetting, isSending, manualImage, runTurnFromText]
  );

  // Auto-progress: run N consecutive turns where the LLM continues the narrative on its own (using the same empty-input
  // "이어서 진행" continuation the composer uses). Each turn builds on the previous turn's committed state. Stoppable.
  const startAutoProgress = useCallback(
    async (turns: number) => {
      if (isSending || isResetting || autoProgressActiveRef.current) {
        return;
      }
      const total = Math.max(1, Math.min(100, Math.round(Number(turns) || 0)));
      autoProgressStopRef.current = false;
      autoProgressActiveRef.current = true;
      // Record which run the loop belongs to so ANY later navigation stops it — see the effect below.
      // Stopping used to be wired to one specific handler (openSimulation), so creating, copying, resetting,
      // or switching progress run all left the loop running against the previous run's threaded state.
      autoProgressOwnerRef.current = readRunOwner(state);
      setIsAutoProgressing(true);
      setAutoProgressTotal(total);
      setAutoProgressRemaining(total);
      pushRuntimeNotice(`자동 진행을 시작합니다 (${total}턴). 중지 버튼으로 멈출 수 있습니다.`, "info");
      // Persist the "keep going" intent so a page refresh mid-run resumes instead of silently dropping the
      // remaining turns. Keyed to the live run (sim/session/progress run) it is progressing.
      const persistIntent = (referenceState: AppState, remaining: number) => {
        if (remaining <= 0) {
          clearAutoProgressIntent();
          return;
        }
        saveAutoProgressIntent({
          simulationId: referenceState.simulation.id,
          activeSessionId: referenceState.simulation.activeSessionId,
          activeProgressRunId: referenceState.activeProgressRunId,
          remaining,
          total
        });
      };
      let workingState: AppState | undefined;
      let completed = 0;
      persistIntent(state, total);
      try {
        for (let index = 0; index < total; index += 1) {
          if (autoProgressStopRef.current) {
            break;
          }
          const committed = await runTurnFromText(AUTO_CONTINUE_TURN_TEXT, {
            baseState: workingState,
            awaitImages: true,
            suppressAutoReset: true
          });
          if (!committed) {
            break;
          }
          workingState = committed;
          completed += 1;
          setAutoProgressRemaining(total - completed);
          persistIntent(committed, total - completed);
        }
      } finally {
        const stopped = autoProgressStopRef.current;
        autoProgressActiveRef.current = false;
        setIsAutoProgressing(false);
        setAutoProgressRemaining(0);
        setAutoProgressTotal(0);
        // The run reached its planned end (or was stopped/aborted); never auto-resume it after the next reload.
        clearAutoProgressIntent();
        pushRuntimeNotice(
          stopped ? `자동 진행을 중지했습니다 (${completed}턴 진행).` : `자동 진행을 완료했습니다 (${completed}턴).`,
          "info"
        );
      }
    },
    [isResetting, isSending, pushRuntimeNotice, runTurnFromText, showRuntimeNotice, state]
  );

  // Any change of the active run stops an in-flight auto-progress loop. The loop threads its own working
  // state and never re-reads React state, so without this it is structurally detached from what the user is
  // looking at: it keeps burning provider calls on a run that is no longer on screen.
  useEffect(() => {
    const owner = autoProgressOwnerRef.current;
    if (!autoProgressActiveRef.current || !owner) {
      return;
    }
    if (!ownsActiveRun(state, owner)) {
      autoProgressStopRef.current = true;
      clearAutoProgressIntent();
    }
  }, [state]);

  // The dispatch-claim set keys off assistant-message ids, so it cannot collide across runs — but it is
  // never emptied either, and it grows for the lifetime of the mount. Clear it whenever the run changes,
  // where the claims from the outgoing run can no longer be relevant.
  useEffect(() => {
    dispatchedImageCueKeysRef.current.clear();
  }, [state.simulation.id, state.activeProgressRunId]);

  const cancelActiveTurn = useCallback(() => {
    if (!turnAbortRef.current) {
      return;
    }
    // Stop the next auto-progress turn too: cancelling the visible turn and then watching the loop start
    // another one is not what "중지" means.
    autoProgressStopRef.current = true;
    clearAutoProgressIntent();
    turnAbortRef.current.abort();
  }, []);

  const stopAutoProgress = useCallback(() => {
    if (!autoProgressActiveRef.current) {
      return;
    }
    autoProgressStopRef.current = true;
    // Clear the persisted intent right away so refreshing before the in-flight turn settles does not resume a
    // run the user just asked to stop.
    clearAutoProgressIntent();
    pushRuntimeNotice("자동 진행 중지 요청됨. 현재 턴을 마치면 멈춥니다.", "info");
  }, [pushRuntimeNotice]);

  // Resume auto-progress after a page refresh/reload. Once storage has been restored, if a persisted intent
  // still points at the now-active simulation/session/progress run, relaunch the loop for the remaining turns so
  // the user does not have to babysit the tab. Matching all three ids keeps a resume from leaking into a
  // different run or a sim the user switched to. Runs at most once per mount.
  useEffect(() => {
    if (!storageReady || autoProgressResumeAttemptedRef.current) {
      return;
    }
    if (isSending || isResetting || autoProgressActiveRef.current) {
      return;
    }
    const intent = loadAutoProgressIntent();
    if (!intent) {
      autoProgressResumeAttemptedRef.current = true;
      return;
    }
    if (!autoProgressIntentMatchesState(intent, state)) {
      // The restored active simulation is not the one that was progressing yet; wait for the matching state
      // (e.g. user opens it) rather than discarding the intent.
      return;
    }
    autoProgressResumeAttemptedRef.current = true;
    showRuntimeNotice(`이전 자동 진행을 이어갑니다 (남은 ${intent.remaining}턴).`);
    void startAutoProgress(intent.remaining);
  }, [storageReady, state, isSending, isResetting, startAutoProgress, showRuntimeNotice]);

  const regenerateAssistantResponse = useCallback(
    async (assistantMessageId: string) => {
      if (isSending || isResetting) {
        showRuntimeNotice("현재 턴 생성이 끝난 뒤 다시 생성할 수 있습니다.");
        return;
      }

      const regeneration = createAssistantRegenerationPlan(state, assistantMessageId);
      if (!regeneration) {
        showRuntimeNotice("다시 생성할 사용자 입력을 찾지 못했습니다.");
        return;
      }

      // Same controller lifecycle as a normal turn: the composer replaces send with 중지 whenever a turn is
      // in flight, and a regeneration IS one — without this the stop button sat there doing nothing for the
      // whole regeneration.
      const turnAbortController = new AbortController();
      turnAbortRef.current = turnAbortController;
      setActiveTurnAbortSignal(turnAbortController.signal);
      setIsSending(true);
      // The run this regeneration belongs to. Needed again in the catch below, which is why it is read here
      // rather than inside the try — the failure rollback must not land on whatever run is active by then.
      const regenerationOwner = readRunOwner(state);
      setActiveTurnOwner(regenerationOwner);
      setPendingUserText(regeneration.userMessage.content);
      setDraft(regeneration.userMessage.content);
      setState(regeneration.baseState);

      try {
        const runtimeBaseState = await resolveStateWithRuntimeSecrets(regeneration.baseState);
        const resetDecision = createAutoResetAgentSessionDecision(runtimeBaseState, regeneration.userMessage.content);
        const turnBaseState = resetDecision.shouldReset
          ? await createResetSessionState(runtimeBaseState)
          : runtimeBaseState;
        const autoResetApplied = turnBaseState.simulation.activeSessionId !== runtimeBaseState.simulation.activeSessionId;
        // Same ownership guard as runTurnFromText: drop these updates if the user switched simulation OR
        // progress run mid-regeneration, so the regenerated turn never bleeds into another run.
        const turnOwner = readRunOwner(turnBaseState);
        // Same local GPU lock as a normal turn (see runTurnFromText).
        let regenerationStarted = false;
        const result = await awaitQueuedWork(
          localGpu.run("llm", turnBaseState, () => {
            regenerationStarted = true;
            if (turnAbortController.signal.aborted) {
              return Promise.reject(createQueuedWorkCancelledError());
            }
            return runSimulationTurn(turnBaseState, regeneration.userMessage.content, regeneration.manualImage, {
              deferImagePlanning: true,
              deferMemoryIngest: true,
              onAssistantText: ({ userMessage, assistantMessage }) => {
                setState((current) => {
                  if (!ownsActiveRun(current, turnOwner)) {
                    return current;
                  }
                  const baseState =
                    autoResetApplied && current.simulation.activeSessionId !== turnBaseState.simulation.activeSessionId
                      ? layerSessionResetOntoLiveState(current, turnBaseState)
                      : current;
                  return upsertStreamingTurnMessages(baseState, userMessage, assistantMessage);
                });
              },
              onMemoryIngested: ({ turnId, memoryEvents, memoryIngestMs }) => {
                setState((current) =>
                  !ownsActiveRun(current, turnOwner)
                    ? current
                    : applyMemoryIngestResultToState(current, turnId, memoryEvents, memoryIngestMs)
                );
              }
            });
          }),
          turnAbortController.signal,
          () => regenerationStarted
        );
        const committedTurnState = applyTurnResultToState(turnBaseState, result);
        setState((current) =>
          !ownsActiveRun(current, turnOwner)
            ? current
            : applyTurnResultToState(autoResetApplied ? layerSessionResetOntoLiveState(current, turnBaseState) : current, result)
        );
        void planTurnImagesWithExpansion(result, committedTurnState, regeneration.manualImage, committedTurnState);

        if (autoResetApplied) {
          showRuntimeNotice(`재생성 자동 handoff: ${resetDecision.reason}.`);
        } else if (result.sidecarTrace.source !== "llm") {
          showRuntimeNotice(createLlmFallbackNotice("응답 재생성 fallback", result.sidecarTrace));
        } else if (regeneration.removedMessageCount > 2) {
          showRuntimeNotice("선택한 응답 이후 진행을 새 응답으로 교체했습니다.");
        } else {
          showRuntimeNotice("응답을 다시 생성했습니다.");
        }
      } catch (error) {
        // A cancel restores the pre-regeneration state rather than reporting a failure — the original
        // response is still the one the user has.
        //
        // Guarded like every other write in this function: an unguarded restore replaced the WHOLE live
        // state with this closure's snapshot, so a regeneration that failed after the user moved to another
        // progress run overwrote that run's messages with the old one's. Comparing simulation.id alone would
        // not catch it — a progress-run switch keeps simulation.id and swaps only the arrays.
        setState((current) => (ownsActiveRun(current, regenerationOwner) ? state : current));
        if (isLlmAbortedError(error) || turnAbortController.signal.aborted) {
          pushRuntimeNotice("응답 재생성을 취소했습니다.", "info");
        } else {
          const message = error instanceof Error ? error.message : "응답 재생성 중 알 수 없는 오류가 발생했습니다.";
          showRuntimeNotice(`응답 재생성 실패: ${message}`);
        }
      } finally {
        if (turnAbortRef.current === turnAbortController) {
          turnAbortRef.current = undefined;
          setActiveTurnAbortSignal(undefined);
        }
        setPendingUserText("");
        setDraft("");
        setManualImage(false);
        setActiveTurnOwner(undefined);
        setIsSending(false);
      }
    },
    [dispatchEarlyTurnImages, isResetting, isSending, localGpu, planTurnImagesWithExpansion, pushRuntimeNotice, resolveStateWithRuntimeSecrets, showRuntimeNotice, state]
  );

  const resetSession = useCallback(async () => {
    if (isResetting || isSending) {
      return;
    }

    setIsResetting(true);
    // The run this reset belongs to. createResetSessionState awaits three NeuralMap round-trips and returns a
    // whole AppState derived from this closure's `state`, so committing it unguarded replaced whatever run was
    // open by then with this one's messages / traces / assets. Comparing simulation.id alone would not catch
    // it: a progress-run switch keeps simulation.id and swaps only the arrays.
    const resetOwner = readRunOwner(state);
    try {
      const nextState = await createResetSessionState(state);
      if (!ownsActiveRun(latestStateRef.current, resetOwner)) {
        // The user left this run while the handoff was in flight — drop the snapshot and the panel/notice
        // side effects with it, so nothing about the abandoned reset shows up in the run they moved to.
        return;
      }
      // Ownership alone is not enough: an image job finishing or a settings edit landing in THIS run during
      // those round-trips also passes the guard, and a wholesale replacement would throw it away. Layer only
      // the new-session deltas, exactly as the auto-handoff commits do.
      setState((current) => (ownsActiveRun(current, resetOwner) ? layerSessionResetOntoLiveState(current, nextState) : current));
      setRightPanel("neuralmap");
      showRuntimeNotice("에이전트 세션을 초기화했습니다. 시뮬레이션 기억과 설정은 유지됩니다.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "세션 초기화 중 알 수 없는 오류가 발생했습니다.";
      showRuntimeNotice(`에이전트 세션 초기화 실패: ${message}`);
    } finally {
      setIsResetting(false);
    }
  }, [isResetting, isSending, showRuntimeNotice, state]);

  // Irreversible: wipes local storage and every simulation in the library, restoring the built-in demo.
  // It used to fire on a single click of a neutral button carrying a Save (floppy-disk) icon labelled
  // "데모 초기화", two clicks from anywhere. The caller now confirms first (see requestResetDemo).
  const resetDemo = useCallback(() => {
    const activeVault = getActivePersonalApiVault();
    clearState();
    window.localStorage.removeItem(SIMULATION_LIBRARY_STORAGE_KEY);
    setState(applyPersonalApiVault(seedState, activeVault));
    setSimulationLibrary(builtInSimulationStates.map(stripLibrarySecrets));
    setView("home");
    showRuntimeNotice("로컬 데이터를 모두 삭제하고 기본 시뮬레이션으로 되돌렸습니다.");
  }, [getActivePersonalApiVault, showRuntimeNotice]);

  const requestResetDemo = useCallback(() => {
    const summary = [
      `시뮬레이션 ${simulationLibrary.length}개`,
      `대화 ${state.messages.length}턴`,
      `이미지 ${state.imageAssets.length}장`
    ].join(" · ");
    const confirmed = window.confirm(
      [
        "이 브라우저에 저장된 DynamicChat 로컬 데이터를 모두 삭제하고 기본 데모 상태로 되돌립니다.",
        "",
        `삭제 대상: ${summary}`,
        "",
        "이 작업은 되돌릴 수 없습니다. 계속할까요?"
      ].join("\n")
    );
    if (confirmed) {
      resetDemo();
    }
  }, [resetDemo, simulationLibrary.length, state.imageAssets.length, state.messages.length]);

  const startNewSimulationRun = useCallback(
    (simulationId?: string) => {
      const source =
        simulationId && simulationId !== state.simulation.id
          ? simulationLibrary.find((item) => item.simulation.id === simulationId)
          : state;
      if (!source) {
        showRuntimeNotice("새 진행을 시작할 원본 시뮬레이션을 찾지 못했습니다.");
        return;
      }

      const activeVault = getActivePersonalApiVault();
      const hydratedSource = hydrateState(source);
      const nextState = applyPersonalApiVault(createFreshSimulationRun(hydratedSource, simulationLibrary), activeVault);
      setState(nextState);
      setSimulationLibrary((current) => upsertSimulationInLibrary(current, nextState));
      setDraft("첫 장면에서 주변을 살피고 주요 인물에게 말을 건다.");
      setRightPanel("neuralmap");
      setView("simulation");
      showRuntimeNotice("새 진행을 채팅 내역에 추가했습니다. 메인 보관함에는 같은 시뮬레이션 하나로 유지됩니다.");
    },
    [getActivePersonalApiVault, showRuntimeNotice, simulationLibrary, state]
  );

  const openProgressRun = useCallback(
    (progressRunId: string) => {
      setState((current) => activateSimulationProgressRun(current, progressRunId));
      setDraft("");
      setRightPanel("neuralmap");
    },
    [setState]
  );

  const deleteProgressRun = useCallback(
    (progressRunId: string) => {
      const currentState = hydrateState(state);
      const targetRun = currentState.progressRuns.find((run) => run.id === progressRunId);
      if (!targetRun) {
        showRuntimeNotice("삭제할 진행 내역을 찾지 못했습니다.");
        return;
      }

      if (currentState.progressRuns.length <= 1) {
        showRuntimeNotice("마지막 진행은 삭제할 수 없습니다.");
        return;
      }

      const deletingActiveRun = currentState.activeProgressRunId === progressRunId;
      setState((current) => deleteSimulationProgressRun(current, progressRunId));
      if (deletingActiveRun) {
        setDraft("");
        setRightPanel("neuralmap");
      }
      showRuntimeNotice(deletingActiveRun ? "진행 내역을 삭제하고 다른 진행으로 이동했습니다." : "진행 내역을 삭제했습니다.");
    },
    [showRuntimeNotice, state]
  );

  const createSimulation = useCallback((simulationDraft: SimulationDraft) => {
    const activeVault = getActivePersonalApiVault();
    const nextState = applyPersonalApiVault(createStateFromDraft(simulationDraft), activeVault);
    setState(nextState);
    setSimulationLibrary((current) => upsertSimulationInLibrary(current, nextState));
    setDraft("첫 장면에서 주변을 살피고 주요 인물에게 말을 건다.");
    setView("simulation");
  }, [getActivePersonalApiVault]);

  const copySimulation = useCallback(
    (simulationId: string) => {
      const source =
        simulationId === state.simulation.id
          ? state
          : simulationLibrary.find((item) => item.simulation.id === simulationId);
      if (!source) {
        showRuntimeNotice("복사할 시뮬레이션을 찾지 못했습니다.");
        return;
      }

      const activeVault = getActivePersonalApiVault();
      const hydratedSource = hydrateState(source);
      const existingTitles = simulationLibrary.map((item) => item.simulation.title);
      const copyDraft: SimulationDraft = {
        ...createDraftFromState(hydratedSource),
        title: createCopiedSimulationTitle(hydratedSource.simulation.title, existingTitles)
      };
      const nextState = applyPersonalApiVault(createStateFromDraft(copyDraft), activeVault);
      setState(nextState);
      setSimulationLibrary((current) => upsertSimulationInLibrary(current, nextState));
      setBuilderSource(nextState);
      setBuilderMode("edit");
      setDraft("");
      setRightPanel("neuralmap");
      setView("create");
      showRuntimeNotice(`"${hydratedSource.simulation.title}" 복사본 제작 화면을 열었습니다.`);
    },
    [getActivePersonalApiVault, showRuntimeNotice, simulationLibrary, state]
  );

  const updateSimulation = useCallback(
    (simulationDraft: SimulationDraft) => {
      const source = builderSource ?? state;
      const activeVault = getActivePersonalApiVault();
      const nextState = applyPersonalApiVault(updateStateFromDraft(source, simulationDraft), activeVault);
      setState(nextState);
      setSimulationLibrary((current) => upsertSimulationInLibrary(current, nextState));
      setView("simulation");
    },
    [builderSource, getActivePersonalApiVault, state]
  );

  const openBuilder = useCallback((source?: AppState) => {
    setBuilderSource(source);
    setBuilderMode(source ? "edit" : "create");
    setView("create");
  }, []);

  const openSimulation = useCallback(
    (simulationId: string) => {
      // Re-opening the simulation that is ALREADY active (e.g. peeking at the home/library view while
      // auto-progress runs, then coming back) must not disturb the live run: keep the in-memory state and the
      // running loop intact. Reloading from the library snapshot here would both stop auto-progress and roll the
      // chat back to the last library sync, which looked like "progress disappeared" after navigating away.
      if (simulationId === state.simulation.id) {
        setView("simulation");
        return;
      }
      // Switching to a DIFFERENT simulation: stop any in-flight auto-progress before swapping the active
      // simulation, otherwise the loop keeps running on the previous simulation's threaded state and its turns
      // land in the one just opened.
      autoProgressStopRef.current = true;
      // Make the outgoing simulation durable BEFORE the swap: a pending idle-time save would otherwise run
      // after the state object has already been replaced.
      flushPendingStateSaves();
      const selected = simulationLibrary.find((item) => item.simulation.id === simulationId);
      if (selected) {
        setState(applyPersonalApiVault(hydrateState(selected), getActivePersonalApiVault()));
      }
      setView("simulation");
    },
    [flushPendingStateSaves, getActivePersonalApiVault, simulationLibrary, state.simulation.id]
  );

  const editSimulation = useCallback(
    (simulationId: string) => {
      const selected = simulationLibrary.find((item) => item.simulation.id === simulationId);
      if (selected) {
        setBuilderSource(hydrateState(selected));
        setBuilderMode("edit");
        setView("create");
      }
    },
    [simulationLibrary]
  );

  const simulationNavClass = `nav-button ${view === "simulation" ? "active" : ""}`;
  const activeImageBackendsLabel = describeActiveImageBackends(state);

  return (
    <main className="app-shell">
      <header className={`topbar ${view === "simulation" ? "crack-global-topbar" : ""}`}>
        {view === "simulation" ? (
          <>
            {/* Only destinations that actually go somewhere. This bar used to carry 월드 and 보관함 (both
                setView("home"), i.e. the logo's destination under two other names), 큐 (setView("simulation")
                from inside the simulation view — a no-op), a search field with no value/onChange/submit, and
                이미지 and 알림 icon buttons with no onClick at all. Six of eight controls did nothing or
                duplicated another, which is what made the chrome read as decoration rather than an app. */}
            <div className="crack-global-left">
              <button className="crack-logo-button" type="button" onClick={() => setView("home")}>
                DynamicChat
              </button>
              <button type="button" onClick={() => setView("home")}>
                라이브러리
              </button>
              <button type="button" onClick={() => openBuilder(state)}>
                제작
              </button>
            </div>
            <div className="crack-global-actions">
              <button className="crack-icon-action" type="button" aria-label="개인 설정" onClick={() => setPersonalSettingsOpen(true)}>
                <UserRound size={20} />
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="brand">
              <span className="brand-mark">
                <Sparkles size={20} />
              </span>
              <div>
                <h1>DynamicChat</h1>
                <p>{PRODUCT_TAGLINE}</p>
              </div>
            </div>
            <div className="topbar-actions">
              <button className={`nav-button ${view === "home" ? "active" : ""}`} type="button" onClick={() => setView("home")}>
                <Home size={16} />
                메인
              </button>
              <button className={`nav-button ${view === "create" ? "active" : ""}`} type="button" onClick={() => openBuilder()}>
                <WandSparkles size={16} />
                제작
              </button>
              <button className={simulationNavClass} type="button" onClick={() => setView("simulation")}>
                <Play size={16} />
                시뮬레이션
              </button>
              <StatusPill icon={<Database size={15} />} label={state.neuralMap.enabled ? "NeuralMap 연결" : "로컬 모드"} tone={state.neuralMap.enabled ? "good" : "neutral"} />
              <StatusPill
                icon={<ImageIcon size={15} />}
                label={
                  state.simulation.realtimeImageEnabled
                    ? `실시간 이미지${activeImageBackendsLabel ? ` · ${activeImageBackendsLabel}` : ""}`
                    : "저장 이미지만"
                }
                tone={state.simulation.realtimeImageEnabled ? "good" : "neutral"}
              />
              <button className="icon-text-button" type="button" onClick={() => setPersonalSettingsOpen(true)}>
                <KeyRound size={16} />
                개인 설정
              </button>
            </div>
          </>
        )}
      </header>
      {personalSettingsOpen ? (
        <PersonalSettingsDialog
          state={state}
          vault={personalApiVault}
          onClose={() => setPersonalSettingsOpen(false)}
          onLlmChange={updateLlmSettings}
          onNovelAiChange={updateNovelAiSettings}
          onSaveLlmSecret={savePersonalLlmSecret}
          onSaveNovelAiSecret={savePersonalNovelAiSecret}
          onSaveImageStoragePath={savePersonalImageStoragePath}
        />
      ) : null}

      {view === "home" ? (
        <HomePage
          activeSimulationId={state.simulation.id}
          simulations={simulationLibrary}
          onCreate={() => openBuilder()}
          onCopy={copySimulation}
          onEdit={editSimulation}
          onOpen={openSimulation}
          onStartNewRun={startNewSimulationRun}
        />
      ) : view === "create" ? (
        <CreateSimulationPage
          key={builderSource?.simulation.id ?? "new-simulation"}
          mode={builderMode}
          initialDraft={builderSource ? createDraftFromState(builderSource) : createInitialSimulationDraft(state)}
          simulationId={builderMode === "edit" ? builderSource?.simulation.id : undefined}
          novelAiTokenRegistered={hasStoredSecret(normalizeStoredNovelAiSecretRecord(personalApiVault.novelAi))}
          onCancel={() => setView("home")}
          onCreate={builderMode === "edit" ? updateSimulation : createSimulation}
          onTestGenerateCharacterImage={testGenerateCharacterImage}
          onPersistProfileImage={persistCharacterProfileImage}
        />
      ) : (
        <SimulationRunErrorBoundary resetKey={`${state.simulation.id}:${state.messages.length}:${state.imageJobs.length}:${rightPanel}`}>
          <CrackSimulationRunPage
            state={state}
            latestAssets={latestAssets}
            draft={draft}
            pendingUserText={pendingUserText}
            // Show turn progress only in the run that owns it. The composer is still locked everywhere
            // (isSendingGlobal below) because two concurrent turns are not supported.
            isSending={isSending && (!activeTurnOwner || ownsActiveRun(state, activeTurnOwner))}
            isSendingGlobal={isSending}
            isResetting={isResetting}
            manualImage={manualImage}
            rightPanel={rightPanel}
            onCancelImageJob={cancelImageJob}
            onDraftChange={setDraft}
            onEditSimulation={() => editSimulation(state.simulation.id)}
            onDeleteImageAsset={redactImageAsset}
            onImageFeedback={updateImageFeedback}
            onImageProfileChange={updateImageProfile}
            onLlmChange={updateLlmSettings}
            onImageTagLlmChange={updateImageTagLlmSettings}
            onNovelAiChange={updateNovelAiSettings}
            onNotify={showRuntimeNotice}
            onDeleteProgressRun={deleteProgressRun}
            onOpenProgressRun={openProgressRun}
            onOpenPersonalSettings={() => setPersonalSettingsOpen(true)}
            onRegenerateAssistantMessage={regenerateAssistantResponse}
            onRegenerateImageJob={regenerateImageJob}
            onResetDemo={requestResetDemo}
            onResetSession={resetSession}
            onStartNewRun={() => startNewSimulationRun(state.simulation.id)}
            onRedactMemory={redactMemoryEvent}
            onRunImageJob={(job) => void runQueuedImageJob(job, state, true)}
            onRightPanelChange={setRightPanel}
            onStateChange={setState}
            onSubmit={handleSubmit}
            onToggleManualImage={setManualImage}
            isAutoProgressing={isAutoProgressing}
            autoProgressRemaining={autoProgressRemaining}
            autoProgressTotal={autoProgressTotal}
            turnPhase={turnPhase}
            onStartAutoProgress={startAutoProgress}
            onStopAutoProgress={stopAutoProgress}
            onCancelTurn={cancelActiveTurn}
            onRunLocalLlmTask={runLocalLlmTask}
          />
        </SimulationRunErrorBoundary>
      )}
      <RuntimeNoticeToast notices={runtimeNotices} onDismiss={dismissRuntimeNotice} />
    </main>
  );
}

class SimulationRunErrorBoundary extends Component<
  { children: React.ReactNode; resetKey: string },
  { errorMessage?: string }
> {
  state: { errorMessage?: string } = {};

  static getDerivedStateFromError(error: unknown): { errorMessage: string } {
    return {
      errorMessage: error instanceof Error ? error.message : "시뮬레이션 화면 렌더링 중 오류가 발생했습니다."
    };
  }

  componentDidUpdate(previousProps: { resetKey: string }) {
    if (previousProps.resetKey !== this.props.resetKey && this.state.errorMessage) {
      this.setState({ errorMessage: undefined });
    }
  }

  render() {
    if (this.state.errorMessage) {
      return (
        <section className="empty-panel">
          <Activity size={18} />
          <strong>시뮬레이션 화면을 다시 표시하지 못했습니다.</strong>
          <span>{this.state.errorMessage}</span>
        </section>
      );
    }

    return this.props.children;
  }
}

function RuntimeNoticeToast({
  notices,
  onDismiss
}: {
  notices: RuntimeNotice[];
  onDismiss: (noticeId: string) => void;
}) {
  return (
    <div className="runtime-toast-region" aria-live="polite">
      {notices.map((notice) => (
        <div className={`runtime-toast tone-${notice.tone}`} key={notice.id} role={notice.tone === "error" ? "alert" : undefined}>
          {notice.tone === "error" ? <AlertTriangle size={16} /> : <Check size={16} />}
          <span>{notice.message}</span>
          <button type="button" onClick={() => onDismiss(notice.id)} aria-label="알림 닫기">
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

function ReaderSettingsControl({
  settings,
  onChange
}: {
  settings: ReaderSettings;
  onChange: (next: ReaderSettings) => void;
}) {
  const detailsRef = useRef<HTMLDetailsElement>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) {
      return;
    }
    const close = () => {
      if (detailsRef.current) {
        detailsRef.current.open = false;
      }
      setOpen(false);
    };
    const handlePointerDown = (event: PointerEvent) => {
      if (detailsRef.current && !detailsRef.current.contains(event.target as Node)) {
        close();
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        close();
      }
    };
    document.addEventListener("pointerdown", handlePointerDown, true);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown, true);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);
  const update = (patch: Partial<ReaderSettings>) => onChange({ ...settings, ...patch });
  const atDefault =
    settings.fontKey === DEFAULT_READER_SETTINGS.fontKey &&
    settings.fontSize === DEFAULT_READER_SETTINGS.fontSize &&
    settings.lineHeight === DEFAULT_READER_SETTINGS.lineHeight &&
    settings.width === DEFAULT_READER_SETTINGS.width &&
    settings.theme === DEFAULT_READER_SETTINGS.theme &&
    settings.dialogueEmphasis === DEFAULT_READER_SETTINGS.dialogueEmphasis;

  return (
    <details
      ref={detailsRef}
      className="crack-more-menu reader-menu"
      onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="reader-trigger" aria-label="읽기 설정">
        <TypeIcon size={16} />
        <span>Aa</span>
      </summary>
      <div className="crack-more-popover reader-popover" role="group" aria-label="읽기 설정">
        <div className="reader-popover-head">
          <strong>읽기 설정</strong>
          <button
            type="button"
            className="reader-reset"
            onClick={() => onChange({ ...DEFAULT_READER_SETTINGS })}
            disabled={atDefault}
          >
            <RotateCcw size={13} />
            기본값
          </button>
        </div>

        <section className="reader-section">
          <span className="reader-label">본문 글꼴</span>
          <div className="reader-font-grid">
            {READER_FONT_OPTIONS.map((option) => (
              <button
                key={option.key}
                type="button"
                className={`reader-font-chip ${settings.fontKey === option.key ? "active" : ""}`}
                onClick={() => update({ fontKey: option.key })}
                aria-pressed={settings.fontKey === option.key}
              >
                <span className="reader-font-name" style={{ fontFamily: option.stack }}>
                  {option.label}
                </span>
                <small>{option.hint}</small>
              </button>
            ))}
          </div>
        </section>

        <section className="reader-section">
          <div className="reader-stepper">
            <span className="reader-label">글자 크기</span>
            <div className="reader-stepper-controls">
              <button
                type="button"
                onClick={() => update({ fontSize: Math.max(READER_FONT_SIZE_MIN, settings.fontSize - 1) })}
                disabled={settings.fontSize <= READER_FONT_SIZE_MIN}
                aria-label="글자 크기 줄이기"
              >
                <Minus size={14} />
              </button>
              <output>{settings.fontSize}px</output>
              <button
                type="button"
                onClick={() => update({ fontSize: Math.min(READER_FONT_SIZE_MAX, settings.fontSize + 1) })}
                disabled={settings.fontSize >= READER_FONT_SIZE_MAX}
                aria-label="글자 크기 키우기"
              >
                <Plus size={14} />
              </button>
            </div>
          </div>
          <div className="reader-stepper">
            <span className="reader-label">줄 간격</span>
            <div className="reader-stepper-controls">
              <button
                type="button"
                onClick={() => update({ lineHeight: Math.round((settings.lineHeight - 0.1) * 10) / 10 })}
                disabled={settings.lineHeight <= READER_LINE_HEIGHT_MIN + 0.001}
                aria-label="줄 간격 줄이기"
              >
                <Minus size={14} />
              </button>
              <output>{settings.lineHeight.toFixed(1)}</output>
              <button
                type="button"
                onClick={() => update({ lineHeight: Math.round((settings.lineHeight + 0.1) * 10) / 10 })}
                disabled={settings.lineHeight >= READER_LINE_HEIGHT_MAX - 0.001}
                aria-label="줄 간격 키우기"
              >
                <Plus size={14} />
              </button>
            </div>
          </div>
        </section>

        <section className="reader-section">
          <span className="reader-label">본문 너비</span>
          <div className="reader-segment">
            {READER_WIDTH_OPTIONS.map((option) => (
              <button
                key={option.key}
                type="button"
                className={settings.width === option.key ? "active" : ""}
                onClick={() => update({ width: option.key })}
                aria-pressed={settings.width === option.key}
              >
                {option.label}
              </button>
            ))}
          </div>
        </section>

        <section className="reader-section">
          <span className="reader-label">테마</span>
          <div className="reader-segment reader-theme-segment">
            {READER_THEME_OPTIONS.map((option) => (
              <button
                key={option.key}
                type="button"
                className={`reader-theme-${option.key} ${settings.theme === option.key ? "active" : ""}`}
                onClick={() => update({ theme: option.key })}
                aria-pressed={settings.theme === option.key}
              >
                {option.key === "night" ? <Moon size={13} /> : <Sun size={13} />}
                {option.label}
              </button>
            ))}
          </div>
        </section>

        <button
          type="button"
          className={`reader-toggle ${settings.dialogueEmphasis ? "active" : ""}`}
          onClick={() => update({ dialogueEmphasis: !settings.dialogueEmphasis })}
          aria-pressed={settings.dialogueEmphasis}
        >
          <span>
            <strong>대사 강조</strong>
            <small>따옴표로 묶인 대사를 또렷한 색으로 표시</small>
          </span>
          <span className="reader-switch" aria-hidden="true" />
        </button>
      </div>
    </details>
  );
}

function CrackSimulationRunPage({
  state,
  latestAssets,
  draft,
  pendingUserText,
  isSending,
  isSendingGlobal,
  isResetting,
  manualImage,
  rightPanel,
  onCancelImageJob,
  onDeleteImageAsset,
  onDraftChange,
  onEditSimulation,
  onImageFeedback,
  onImageProfileChange,
  onLlmChange,
  onImageTagLlmChange,
  onNovelAiChange,
  onNotify,
  onDeleteProgressRun,
  onOpenProgressRun,
  onOpenPersonalSettings,
  onRegenerateAssistantMessage,
  onRegenerateImageJob,
  onResetDemo,
  onResetSession,
  onStartNewRun,
  onRedactMemory,
  onRunImageJob,
  onRightPanelChange,
  onStateChange,
  onSubmit,
  onToggleManualImage,
  isAutoProgressing,
  autoProgressRemaining,
  autoProgressTotal,
  turnPhase,
  onStartAutoProgress,
  onStopAutoProgress,
  onCancelTurn,
  onRunLocalLlmTask
}: {
  state: AppState;
  latestAssets: ImageAsset[];
  draft: string;
  pendingUserText: string;
  /** True only in the run that owns the in-flight turn — drives the pending bubble and phase card. */
  isSending: boolean;
  /** True while ANY run has a turn in flight — drives the composer/controls lock. */
  isSendingGlobal: boolean;
  isResetting: boolean;
  manualImage: boolean;
  rightPanel: RightPanel;
  isAutoProgressing: boolean;
  autoProgressRemaining: number;
  autoProgressTotal: number;
  turnPhase?: TurnPhase;
  onStartAutoProgress: (turns: number) => void;
  onStopAutoProgress: () => void;
  /** Aborts the in-flight turn's provider request. */
  onCancelTurn: () => void;
  /** Runs an LLM call outside a turn on the shared local-GPU lock. */
  onRunLocalLlmTask: LocalLlmTaskRunner;
  onCancelImageJob: (jobId: string) => void;
  onDeleteImageAsset: (assetId: string) => void;
  onDraftChange: (value: string) => void;
  onEditSimulation: () => void;
  onImageFeedback: (assetId: string, rating: ImageFeedbackRating) => void;
  onImageProfileChange: (patch: Partial<ImageGenerationProfile>) => void;
  onLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onImageTagLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onNovelAiChange: (patch: Partial<NovelAiApiSettings>) => void;
  onNotify: (message: string) => void;
  onDeleteProgressRun: (progressRunId: string) => void;
  onOpenProgressRun: (progressRunId: string) => void;
  onOpenPersonalSettings: () => void;
  onRegenerateAssistantMessage: (assistantMessageId: string) => void;
  onRegenerateImageJob: (job: ImageGenerationJob, backend?: ImageBackendId) => void;
  onResetDemo: () => void;
  onResetSession: () => void;
  onStartNewRun: () => void;
  onRedactMemory: (memoryId: string) => void;
  onRunImageJob: (job: ImageGenerationJob) => void;
  onRightPanelChange: (panel: RightPanel) => void;
  onStateChange: (value: AppState | ((current: AppState) => AppState)) => void;
  onSubmit: (event: FormEvent) => void;
  onToggleManualImage: (value: boolean) => void;
}) {
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [autoProgressTurns, setAutoProgressTurns] = useState(10);
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);
  const [opsRailWidth, setOpsRailWidth] = useState(loadOpsRailWidth);
  const [opsRailResizing, setOpsRailResizing] = useState(false);
  const [readerSettings, setReaderSettings] = useState(loadReaderSettings);
  const handleReaderSettingsChange = useCallback((next: ReaderSettings) => {
    setReaderSettings(next);
    saveReaderSettings(next);
  }, []);
  // The reader theme was scoped to the story column, so choosing Night repainted the prose and left the
  // topbar, rails, and composer chrome glaring white around it. Mirroring the attribute onto the document
  // root lets the chrome follow, while every existing `.crack-story-stage[data-reader-theme=…]` rule keeps
  // working unchanged. Cleared on unmount so the builder/home views are never left themed by a run.
  useEffect(() => {
    document.documentElement.setAttribute("data-reader-theme", readerSettings.theme);
    return () => document.documentElement.removeAttribute("data-reader-theme");
  }, [readerSettings.theme]);
  const storyScrollRef = useRef<HTMLDivElement>(null);
  const composerTextareaRef = useRef<HTMLTextAreaElement>(null);
  const opsRailRef = useRef<HTMLElement>(null);
  // A pending "show me this settings section" from a button outside the settings tab; the settings panel that is on
  // screen scrolls to it, focuses it and clears it.
  const [settingsJump, setSettingsJump] = useState<SettingsJumpRequest>();
  const [settingsSection, setSettingsSectionState] = useState<SettingsSection>(loadSettingsSection);
  const changeSettingsSection = useCallback((section: SettingsSection) => {
    setSettingsSectionState(section);
    saveSettingsSection(section);
  }, []);
  // Coming back to 기억 or 설정 reopens the sub-tab last used there instead of the group's first one.
  const lastPanelByGroupRef = useRef<Partial<Record<RightPanelGroup, RightPanel>>>({});
  useEffect(() => {
    lastPanelByGroupRef.current[resolveRightPanelGroup(rightPanel)] = rightPanel;
  }, [rightPanel]);
  const resolveGroupEntryPanel = (panel: RightPanel): RightPanel => {
    const group = resolveRightPanelGroup(panel);
    // Only the bare group ids mean "the group"; a specific sub-panel asked for by id is honoured as is.
    return panel === group ? lastPanelByGroupRef.current[group] ?? panel : panel;
  };
  const assetsById = useMemo(() => new Map(state.imageAssets.map((asset) => [asset.id, asset])), [state.imageAssets]);
  const allJobsByTurnId = useMemo(() => {
    const groupedJobs = new Map<string, ImageGenerationJob[]>();
    for (const job of state.imageJobs) {
      const currentJobs = groupedJobs.get(job.turnId);
      if (currentJobs) {
        currentJobs.push(job);
      } else {
        groupedJobs.set(job.turnId, [job]);
      }
    }
    return groupedJobs;
  }, [state.imageJobs]);
  const pendingJobsByTurnId = useMemo(() => {
    const groupedJobs = new Map<string, ImageGenerationJob[]>();
    allJobsByTurnId.forEach((jobs, turnId) => {
      const pendingJobs = jobs.filter((job) => job.status !== "completed");
      if (pendingJobs.length > 0) {
        groupedJobs.set(turnId, pendingJobs);
      }
    });
    return groupedJobs;
  }, [allJobsByTurnId]);
  const historyItems = useMemo(() => createHistoryItems(state), [state]);
  const replySuggestions = useMemo(() => createReplySuggestions(state), [state]);
  const latestContextPack = state.contextPacks.at(-1);
  const latestMessageId = state.messages.at(-1)?.id;
  const imageJobStatusSignature = useMemo(
    () => state.imageJobs.map((job) => `${job.id}:${job.status}:${job.assetIds.length}`).join("|"),
    [state.imageJobs]
  );
  const storyScrollStateRef = useRef<StoryScrollSnapshot>({
    simulationId: state.simulation.id,
    progressRunId: state.activeProgressRunId ?? "",
    messageCount: state.messages.length,
    isSending,
    imageJobStatusSignature,
    imageAssetCount: state.imageAssets.length,
    scrollHeight: 0,
    pinnedToBottom: true
  });
  // Numbered per ASSISTANT turn, which is what a reader means by "turn". The transcript used to carry one
  // marker at the very top, labelled with the LAST turn's number and timestamp, so scrolling gave no sense of
  // position and there was no way to point at a past beat.
  const assistantTurnNumbers = useMemo(() => {
    const numbers = new Map<string, number>();
    let turn = 0;
    for (const message of state.messages) {
      if (message.role === "assistant") {
        turn += 1;
        numbers.set(message.id, turn);
      }
    }
    return numbers;
  }, [state.messages]);
  const markerDate = new Date(state.messages.at(-1)?.createdAt ?? state.simulation.updatedAt);
  const latestImageJob = state.imageJobs.at(-1);
  const activeImageBackendLabel = describeActiveImageBackends(state);
  const rightMenuStatus = activeImageBackendLabel ? `${activeImageBackendLabel} 연결` : "저장 이미지";
  const autoResetDecision = useMemo(() => createAutoResetAgentSessionDecision(state), [state]);
  const activePendingUserText = isSending ? pendingUserText || createTurnSubmissionText(draft) : draft;

  const toggleTriggerMode = () => {
    onImageProfileChange({
      triggerMode: state.imageProfile.triggerMode === "manual" ? "realtime_auto" : "manual"
    });
  };

  const handleDeleteProgressRun = (item: (typeof historyItems)[number]) => {
    if (!item.canDelete || isSendingGlobal || isResetting) {
      return;
    }

    const confirmed = window.confirm(`'${item.label}' 진행 내역을 삭제할까요?\n이 진행의 대화, 기억, 이미지 내역이 목록에서 제거됩니다.`);
    if (confirmed) {
      onDeleteProgressRun(item.id);
    }
  };

  const openMobilePanel = (panel: RightPanel) => {
    onRightPanelChange(panel);
    setMobilePanelOpen(true);
  };

  // '실사 설정 열기' used to switch the rail to 설정 and leave it at the top, eleven rail-heights above the ComfyUI
  // section it names. Below 1180px the rail is hidden and the settings live in the mobile drawer, which has to open.
  const openPhotorealSettings = () => {
    const railHidden = !opsRailRef.current || opsRailRef.current.getClientRects().length === 0;
    if (railHidden) {
      openMobilePanel("settings");
    } else {
      onRightPanelChange("settings");
    }
    changeSettingsSection("photoreal");
    setSettingsJump({ section: "photoreal", id: createId("settings_jump") });
  };
  const clearSettingsJump = useCallback(() => setSettingsJump(undefined), []);

  const handleOpsRailResizePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      const startX = event.clientX;
      const startWidth = opsRailWidth;

      setOpsRailResizing(true);
      document.body.classList.add("resizing-ops-rail");

      const handlePointerMove = (moveEvent: PointerEvent) => {
        const nextWidth = clampOpsRailWidth(startWidth + startX - moveEvent.clientX);
        setOpsRailWidth(nextWidth);
      };

      const handlePointerUp = (upEvent: PointerEvent) => {
        const nextWidth = clampOpsRailWidth(startWidth + startX - upEvent.clientX);
        setOpsRailWidth(nextWidth);
        saveOpsRailWidth(nextWidth);
        setOpsRailResizing(false);
        document.body.classList.remove("resizing-ops-rail");
        window.removeEventListener("pointermove", handlePointerMove);
        window.removeEventListener("pointerup", handlePointerUp);
        window.removeEventListener("pointercancel", handlePointerCancel);
      };

      const handlePointerCancel = () => {
        setOpsRailWidth(startWidth);
        setOpsRailResizing(false);
        document.body.classList.remove("resizing-ops-rail");
        window.removeEventListener("pointermove", handlePointerMove);
        window.removeEventListener("pointerup", handlePointerUp);
        window.removeEventListener("pointercancel", handlePointerCancel);
      };

      window.addEventListener("pointermove", handlePointerMove);
      window.addEventListener("pointerup", handlePointerUp);
      window.addEventListener("pointercancel", handlePointerCancel);
    },
    [opsRailWidth]
  );

  // The handle was pointer-only: a named, focusable control that a keyboard could reach and then do nothing
  // with (WCAG 2.1.1). Arrow keys now step the width, Shift takes bigger steps and Home/End go to the bounds.
  // Left widens because the rail is on the right edge — so the leftmost position (Home) is the widest one.
  // Every path runs through clampOpsRailWidth and persists, exactly like the drag.
  const handleOpsRailResizeKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLButtonElement>) => {
      const step = event.shiftKey ? 48 : 16;
      const requestedWidth =
        event.key === "ArrowLeft"
          ? opsRailWidth + step
          : event.key === "ArrowRight"
            ? opsRailWidth - step
            : event.key === "Home"
              ? MAX_OPS_RAIL_WIDTH
              : event.key === "End"
                ? MIN_OPS_RAIL_WIDTH
                : null;

      if (requestedWidth === null) {
        return;
      }

      event.preventDefault();
      const nextWidth = clampOpsRailWidth(requestedWidth);
      setOpsRailWidth(nextWidth);
      saveOpsRailWidth(nextWidth);
    },
    [opsRailWidth]
  );

  // Announced separately from MAX_OPS_RAIL_WIDTH because clampOpsRailWidth enforces getOpsRailWidthBounds,
  // whose ceiling shrinks with the viewport — a fixed 720 would tell a screen reader the handle can reach a
  // width it silently refuses, so Home would read as a broken control rather than as at its limit.
  const [opsRailMaxWidth, setOpsRailMaxWidth] = useState(() => getOpsRailWidthBounds().max);

  const handleStoryScroll = useCallback(() => {
    const scrollElement = storyScrollRef.current;
    if (!scrollElement) {
      return;
    }
    storyScrollStateRef.current = {
      ...storyScrollStateRef.current,
      scrollHeight: scrollElement.scrollHeight,
      pinnedToBottom: isStoryScrollPinnedToBottom(scrollElement)
    };
  }, []);

  useLayoutEffect(() => {
    const scrollElement = storyScrollRef.current;
    if (!scrollElement) {
      return;
    }

    const previous = storyScrollStateRef.current;
    const firstLayout = previous.scrollHeight === 0;
    const simulationChanged =
      previous.simulationId !== state.simulation.id || previous.progressRunId !== (state.activeProgressRunId ?? "");
    const messageCountChanged = previous.messageCount !== state.messages.length;
    const sendingStarted = isSending && !previous.isSending;
    const imageStateChanged =
      previous.imageJobStatusSignature !== imageJobStatusSignature || previous.imageAssetCount !== state.imageAssets.length;
    const shouldFollowBottom = firstLayout || simulationChanged || sendingStarted || (messageCountChanged && previous.pinnedToBottom);

    if (shouldFollowBottom) {
      scrollStoryToBottom(scrollElement, firstLayout || simulationChanged ? "auto" : "smooth");
    } else if (imageStateChanged && previous.pinnedToBottom) {
      scrollStoryToBottom(scrollElement, "auto");
    }

    storyScrollStateRef.current = {
      simulationId: state.simulation.id,
      progressRunId: state.activeProgressRunId ?? "",
      messageCount: state.messages.length,
      isSending,
      imageJobStatusSignature,
      imageAssetCount: state.imageAssets.length,
      scrollHeight: scrollElement.scrollHeight,
      pinnedToBottom: shouldFollowBottom || (imageStateChanged && previous.pinnedToBottom) || isStoryScrollPinnedToBottom(scrollElement)
    };
    // activeProgressRunId belongs here: a progress-run switch keeps simulation.id, so when the two runs happen
    // to share message count, image-asset count and job signature nothing else in this list changes and the
    // `simulationChanged` branch above never gets to run — exactly the case it exists for.
  }, [
    imageJobStatusSignature,
    isSending,
    state.activeProgressRunId,
    state.imageAssets.length,
    state.messages.length,
    state.simulation.id
  ]);

  useEffect(() => {
    if (isSending) {
      setSuggestionsOpen(false);
    }
  }, [isSending]);

  useEffect(() => {
    const handleWindowResize = () => {
      setOpsRailWidth((current) => clampOpsRailWidth(current));
      setOpsRailMaxWidth(getOpsRailWidthBounds().max);
    };

    window.addEventListener("resize", handleWindowResize);
    return () => window.removeEventListener("resize", handleWindowResize);
  }, []);

  const handleInsertActionNotation = useCallback(() => {
    insertActionNotation(composerTextareaRef.current, draft, onDraftChange);
    setSuggestionsOpen(false);
  }, [draft, onDraftChange]);

  return (
    <section className="crack-run-workspace">
      <nav className="crack-episode-bar" aria-label="시뮬레이션 상단 메뉴">
        {/* The 에피소드 / 파티챗 pair that used to sit here had no onClick on either button and a hardcoded
            className="active" on the first — a tab control that could not be operated. It held a fixed 260px
            column, which is why the run title truncated; that column now belongs to the title. */}
        <button className="crack-story-breadcrumb" type="button" onClick={onEditSimulation}>
          {state.simulation.title}
          <ChevronRight size={15} />
        </button>
        <div className="crack-episode-actions">
          {/* The progress-run rail is display:none below 1180px, so on a narrow laptop the active run was
              neither visible nor switchable. This select is the always-available fallback and doubles as the
              only on-screen indication of WHICH run you are in once the rail is gone. */}
          {historyItems.length > 1 ? (
            <label className="crack-run-select" title="진행 전환">
              <span className="visually-hidden">진행 전환</span>
              <select
                value={historyItems.find((item) => item.active)?.id ?? ""}
                disabled={isSendingGlobal || isResetting}
                onChange={(event) => onOpenProgressRun(event.target.value)}
              >
                {historyItems.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <ReaderSettingsControl settings={readerSettings} onChange={handleReaderSettingsChange} />
          <button
            className={`crack-square-select ${manualImage ? "active" : ""}`}
            type="button"
            onClick={() => onToggleManualImage(!manualImage)}
            aria-pressed={manualImage}
            aria-label="이번 턴 이미지 생성 요청"
          >
            <span>{manualImage ? <Check size={13} /> : <Square size={15} />}</span>
            <ChevronDown size={15} />
          </button>
          <button className="icon-text-button" type="button" onClick={onResetSession} disabled={isResetting || isSendingGlobal}>
            <RefreshCcw size={15} />
            {isResetting ? "handoff 중" : "에이전트 세션 초기화"}
          </button>
          <StatusPill
            icon={<RefreshCcw size={15} />}
            label={formatAutoResetStatus(autoResetDecision)}
            tone={autoResetDecision.status === "stable" ? "neutral" : "good"}
          />
          <button className="icon-text-button" type="button" onClick={onStartNewRun}>
            <Plus size={15} />
            새 진행 시작
          </button>
          <details className="crack-more-menu">
            <summary aria-label="더보기">
              <MoreHorizontal size={19} />
            </summary>
            <div className="crack-more-popover">
              <strong>실행 설정</strong>
              <ModelQuickSwitch llm={state.llm} onChange={onLlmChange} />
              <button type="button" onClick={toggleTriggerMode}>
                <ImageIcon size={15} />
                이미지 트리거: {state.imageProfile.triggerMode}
              </button>
              <button type="button" onClick={onEditSimulation}>
                <Settings2 size={15} />
                시뮬레이션 수정
              </button>
              <button type="button" onClick={onStartNewRun}>
                <Plus size={15} />
                새 진행 시작
              </button>
              <button className="danger-action" type="button" onClick={onResetDemo}>
                <Trash2 size={15} />
                모든 로컬 데이터 삭제
              </button>
              <small>{rightMenuStatus} · {formatAutoResetDetail(autoResetDecision)} · {latestImageJob ? getImageJobStatusLabel(latestImageJob.status) : "작업 없음"}</small>
            </div>
          </details>
        </div>
      </nav>

      <div className="crack-chat-shell" style={{ "--ops-rail-width": `${opsRailWidth}px` } as CSSProperties}>
        <aside className="crack-history-rail" aria-label="채팅 내역">
          <strong>채팅 내역</strong>
          <div className="crack-history-list">
            {historyItems.map((item) => (
              <div className={`crack-history-item ${item.active ? "active" : ""} ${item.canDelete ? "" : "single"}`} key={item.id}>
                <button
                  className="crack-history-open"
                  type="button"
                  onClick={() => onOpenProgressRun(item.id)}
                  aria-current={item.active ? "true" : undefined}
                >
                  <span>{item.label}</span>
                  <small>{item.detail}</small>
                </button>
                {item.canDelete ? (
                  <button
                    className="crack-history-delete"
                    type="button"
                    onClick={() => handleDeleteProgressRun(item)}
                    disabled={isSendingGlobal || isResetting}
                    aria-label={`${item.label} 삭제`}
                    title={isSendingGlobal || isResetting ? "진행 중에는 삭제할 수 없습니다" : "진행 삭제"}
                  >
                    <Trash2 size={14} />
                  </button>
                ) : null}
              </div>
            ))}
          </div>
        </aside>

        <section
          className="crack-story-stage"
          aria-label="시뮬레이션 진행"
          data-reader-theme={readerSettings.theme}
          data-reader-dialogue={readerSettings.dialogueEmphasis ? "on" : "off"}
          style={readerStageStyle(readerSettings)}
        >
          <div className="crack-story-scroll" ref={storyScrollRef} onScroll={handleStoryScroll}>
            {state.messages.map((message) => (
              <CrackTimelineMessage
                key={message.id}
                message={message}
                turnNumber={assistantTurnNumbers.get(message.id)}
                assets={findMessageAssets(message, assetsById)}
                jobs={pendingJobsByTurnId.get(message.id) ?? EMPTY_IMAGE_JOBS}
                relatedJobs={allJobsByTurnId.get(message.id) ?? EMPTY_IMAGE_JOBS}
                evidenceCount={message.id === latestMessageId ? latestContextPack?.evidence.length ?? 0 : message.referencedNodeIds.length}
                onCancelJob={onCancelImageJob}
                onDraftChange={onDraftChange}
                onFeedback={onImageFeedback}
                onNotify={onNotify}
                onRegenerateImageJob={onRegenerateImageJob}
                onRegenerateResponse={onRegenerateAssistantMessage}
                onRunJob={onRunImageJob}
                regenerationDisabled={isSendingGlobal || isResetting}
              />
            ))}
            {isSending ? <CrackPendingTurn userText={activePendingUserText} state={state} manualImage={manualImage} turnPhase={turnPhase} /> : null}
          </div>

          <form className="crack-composer" onSubmit={onSubmit}>
            {/* Inside the composer card rather than above it: the card is position:absolute with z-index 20,
                so a sibling in normal flow renders behind it. */}
            <CrackSetupNotice
              state={state}
              onOpenPersonalSettings={onOpenPersonalSettings}
              onOpenImageSettings={openPhotorealSettings}
            />
            <textarea
              ref={composerTextareaRef}
              aria-label="메시지 보내기"
              disabled={isSendingGlobal || isResetting || isAutoProgressing}
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder={
                isResetting
                  ? "handoff를 생성하는 중입니다"
                  : isAutoProgressing
                    ? `자동 진행 중입니다 (${autoProgressTotal - autoProgressRemaining}/${autoProgressTotal})`
                    : isSending
                      ? "응답을 생성하는 중입니다"
                      : "메시지 보내기"
              }
            />
            {suggestionsOpen ? (
              <div className="crack-suggestion-menu">
                {replySuggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => {
                      onDraftChange(suggestion);
                      setSuggestionsOpen(false);
                    }}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            ) : null}
            <div className="crack-composer-footer">
              <div>
                <button
                  className={`crack-round-tool ${manualImage ? "active" : ""}`}
                  type="button"
                  onClick={() => onToggleManualImage(!manualImage)}
                  aria-label="이미지 생성 토글"
                >
                  <Sparkles size={17} />
                </button>
                <button className="crack-recommend-button" type="button" onClick={() => setSuggestionsOpen((current) => !current)}>
                  <SlidersHorizontal size={15} />
                  추천답변
                </button>
                <button
                  className="crack-round-tool crack-action-notation-button"
                  type="button"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={handleInsertActionNotation}
                  disabled={isSendingGlobal || isResetting || isAutoProgressing}
                  aria-label="행동 입력 괄호 삽입"
                  title="행동 입력 *()* 삽입"
                >
                  <Parentheses size={16} />
                </button>
                {isAutoProgressing ? (
                  <button
                    className="crack-auto-progress-control active"
                    type="button"
                    onClick={onStopAutoProgress}
                    title="자동 진행 중지"
                  >
                    <Square size={14} fill="currentColor" />
                    중지 {autoProgressTotal - autoProgressRemaining}/{autoProgressTotal}
                  </button>
                ) : (
                  <div className="crack-auto-progress-control" title="자동 진행할 턴 수만큼 LLM이 알아서 서사를 이어갑니다">
                    <RefreshCcw size={14} />
                    <input
                      aria-label="자동 진행 턴 수"
                      type="number"
                      min="1"
                      max="100"
                      value={autoProgressTurns}
                      disabled={isSendingGlobal || isResetting}
                      onChange={(event) => setAutoProgressTurns(Math.max(1, Math.min(100, Number(event.target.value) || 1)))}
                    />
                    <button
                      type="button"
                      onClick={() => onStartAutoProgress(autoProgressTurns)}
                      disabled={isSendingGlobal || isResetting}
                    >
                      자동진행
                    </button>
                  </div>
                )}
              </div>
              {/* While a turn is running the send button becomes a stop button. Previously the composer just
                  locked, so a turn that hung on a slow provider could only be escaped by reloading the page —
                  which also lost the draft and left the provider call running. */}
              {isSendingGlobal ? (
                <button
                  className="crack-send-button is-stop"
                  type="button"
                  onClick={onCancelTurn}
                  aria-label="진행 중지"
                  title="진행 중지"
                >
                  <Square size={16} fill="currentColor" />
                </button>
              ) : (
                <button className="crack-send-button" disabled={isResetting || isAutoProgressing} type="submit" aria-label="전송">
                  {isResetting || isAutoProgressing ? <Activity size={18} /> : <Play size={18} fill="currentColor" />}
                </button>
              )}
            </div>
          </form>
        </section>

        <aside className={`crack-ops-rail ${opsRailResizing ? "is-resizing" : ""}`} aria-label="운영 패널" ref={opsRailRef}>
          {/* role="separator" rather than button: there is nothing here to press, and the window-splitter
              role is what lets a screen reader read the current width back out of aria-valuenow while the
              arrow keys move it. The class selectors in styles.css are element-agnostic, so keeping the
              <button> element keeps the existing look untouched. */}
          <button
            className="ops-rail-resize-handle"
            type="button"
            role="separator"
            aria-orientation="vertical"
            aria-valuenow={opsRailWidth}
            aria-valuemin={MIN_OPS_RAIL_WIDTH}
            aria-valuemax={opsRailMaxWidth}
            onPointerDown={handleOpsRailResizePointerDown}
            onKeyDown={handleOpsRailResizeKeyDown}
            aria-label="오른쪽 운영 패널 크기 조절 (좌우 방향키)"
          />
          <div className="tabs" role="tablist" onKeyDown={handleTablistKeyDown} aria-label="오른쪽 패널">
            <TabButton active={resolveRightPanelGroup(rightPanel) === "image"} icon={<ImageIcon size={17} />} label="장면" onClick={() => onRightPanelChange("image")} panelKey="ops-image" />
            <TabButton active={resolveRightPanelGroup(rightPanel) === "memory"} icon={<Brain size={17} />} label="기억" onClick={() => onRightPanelChange(resolveGroupEntryPanel("memory"))} panelKey="ops-memory" />
            <TabButton active={resolveRightPanelGroup(rightPanel) === "settings"} icon={<Settings2 size={17} />} label="설정" onClick={() => onRightPanelChange(resolveGroupEntryPanel("settings"))} panelKey="ops-settings" />
          </div>
          {/* The element role="tabpanel" names for the tablist above, and the rail's scroller: the tabs and the
              sub-tabs at the top of RuntimePanelContent stay put while only the section under them scrolls.
              No tabIndex — the panel is full of focusable controls, so a tab stop here would go nowhere. */}
          <div
            className="ops-rail-panel"
            role="tabpanel"
            id={`tabpanel-ops-${resolveRightPanelGroup(rightPanel)}`}
            aria-labelledby={`tab-ops-${resolveRightPanelGroup(rightPanel)}`}
          >
            <RuntimePanelContent
              idPrefix="ops"
              onPanelChange={onRightPanelChange}
              settingsSection={settingsSection}
              onSettingsSectionChange={changeSettingsSection}
              panel={rightPanel}
              state={state}
              isSending={isSending}
              latestAssets={latestAssets}
              latestContextPackId={latestContextPack?.id}
              pendingUserText={activePendingUserText}
              onCancelImageJob={onCancelImageJob}
              onDeleteImageAsset={onDeleteImageAsset}
              onImageFeedback={onImageFeedback}
              onImageProfileChange={onImageProfileChange}
              onLlmChange={onLlmChange}
              onImageTagLlmChange={onImageTagLlmChange}
              onNovelAiChange={onNovelAiChange}
              onNotify={onNotify}
              onOpenPersonalSettings={onOpenPersonalSettings}
              onRedactMemory={onRedactMemory}
              onRegenerateImageJob={onRegenerateImageJob}
              onResetDemo={onResetDemo}
              onRunImageJob={onRunImageJob}
              onStateChange={onStateChange as Dispatch<SetStateAction<AppState>>}
              settingsJump={settingsJump}
              onSettingsJumpHandled={clearSettingsJump}
              onRunLocalLlmTask={onRunLocalLlmTask}
            />
          </div>
        </aside>
      </div>
      <MobileOpsDock
        activePanel={rightPanel}
        open={mobilePanelOpen}
        onClose={() => setMobilePanelOpen(false)}
        onSelectPanel={(panel) => openMobilePanel(resolveGroupEntryPanel(panel))}
      >
        <RuntimePanelContent
          idPrefix="mobile-ops"
          onPanelChange={onRightPanelChange}
          settingsSection={settingsSection}
          onSettingsSectionChange={changeSettingsSection}
          panel={rightPanel}
          state={state}
          isSending={isSending}
          latestAssets={latestAssets}
          latestContextPackId={latestContextPack?.id}
          pendingUserText={activePendingUserText}
          onCancelImageJob={onCancelImageJob}
          onDeleteImageAsset={onDeleteImageAsset}
          onImageFeedback={onImageFeedback}
          onImageProfileChange={onImageProfileChange}
          onLlmChange={onLlmChange}
          onImageTagLlmChange={onImageTagLlmChange}
          onNovelAiChange={onNovelAiChange}
          onNotify={onNotify}
          onOpenPersonalSettings={onOpenPersonalSettings}
          onRedactMemory={onRedactMemory}
          onRegenerateImageJob={onRegenerateImageJob}
          onResetDemo={onResetDemo}
          onRunImageJob={onRunImageJob}
          onStateChange={onStateChange as Dispatch<SetStateAction<AppState>>}
          settingsJump={settingsJump}
          onSettingsJumpHandled={clearSettingsJump}
          onRunLocalLlmTask={onRunLocalLlmTask}
        />
      </MobileOpsDock>
    </section>
  );
}

function RuntimePanelContent({
  panel,
  state,
  isSending,
  latestAssets,
  latestContextPackId,
  pendingUserText,
  onCancelImageJob,
  onDeleteImageAsset,
  onImageFeedback,
  onImageProfileChange,
  onLlmChange,
  onImageTagLlmChange,
  onNovelAiChange,
  onNotify,
  onOpenPersonalSettings,
  onRedactMemory,
  onRegenerateImageJob,
  onResetDemo,
  onRunImageJob,
  onStateChange,
  settingsJump,
  onSettingsJumpHandled,
  onRunLocalLlmTask,
  idPrefix,
  onPanelChange,
  settingsSection,
  onSettingsSectionChange
}: {
  panel: RightPanel;
  idPrefix: string;
  onPanelChange: (panel: RightPanel) => void;
  settingsSection: SettingsSection;
  onSettingsSectionChange: (section: SettingsSection) => void;
  state: AppState;
  isSending: boolean;
  latestAssets: ImageAsset[];
  latestContextPackId?: string;
  pendingUserText: string;
  onCancelImageJob: (jobId: string) => void;
  onDeleteImageAsset: (assetId: string) => void;
  onImageFeedback: (assetId: string, rating: ImageFeedbackRating) => void;
  onImageProfileChange: (patch: Partial<ImageGenerationProfile>) => void;
  onLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onImageTagLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onNovelAiChange: (patch: Partial<NovelAiApiSettings>) => void;
  onNotify: (message: string) => void;
  onOpenPersonalSettings: () => void;
  onRedactMemory: (memoryId: string) => void;
  onRegenerateImageJob: (job: ImageGenerationJob, backend?: ImageBackendId) => void;
  onResetDemo: () => void;
  onRunImageJob: (job: ImageGenerationJob) => void;
  onStateChange: Dispatch<SetStateAction<AppState>>;
  settingsJump?: SettingsJumpRequest;
  onSettingsJumpHandled?: () => void;
  onRunLocalLlmTask?: LocalLlmTaskRunner;
}) {
  if (panel === "image") {
    return (
      <ImagePanel
        state={state}
        assets={latestAssets}
        jobs={state.imageJobs}
        onCancelJob={onCancelImageJob}
        onDeleteAsset={onDeleteImageAsset}
        onFeedback={onImageFeedback}
        onImageProfileChange={onImageProfileChange}
        onRegenerateJob={onRegenerateImageJob}
        onRunJob={onRunImageJob}
      />
    );
  }

  // 기억: what the app remembers about this story — one of its four panels at a time, picked from a second row.
  if (resolveRightPanelGroup(panel) === "memory") {
    return (
      <RailSubtabs idPrefix={idPrefix} label="기억 패널" items={MEMORY_SUBPANELS} active={panel} onSelect={onPanelChange}>
        {panel === "relationship" ? (
          <RelationshipMapPanel state={state} onStateChange={onStateChange} />
        ) : panel === "persona" ? (
          <PersonaPanel state={state} onNotify={onNotify} onStateChange={onStateChange} />
        ) : panel === "neuralmap" ? (
          <NeuralMapPanel state={state} isSending={isSending} pendingUserText={pendingUserText} onNotify={onNotify} onStateChange={onStateChange} />
        ) : (
          <MemoryPanel state={state} selectedContextPackId={latestContextPackId} onRedactMemory={onRedactMemory} />
        )}
      </RailSubtabs>
    );
  }

  // 설정: one settings section at a time, and the runtime diagnostics as the last tab rather than above them all.
  const activeSettingsTab: SettingsSection | "ops" = panel === "ops" ? "ops" : settingsSection;
  return (
    <RailSubtabs
      idPrefix={idPrefix}
      label="설정 항목"
      items={SETTINGS_RAIL_TABS}
      active={activeSettingsTab}
      onSelect={(tab) => {
        if (tab === "ops") {
          onPanelChange("ops");
          return;
        }
        onPanelChange("settings");
        onSettingsSectionChange(tab);
      }}
    >
      {panel === "ops" ? (
        <OperationalPanel state={state} onFeedback={onImageFeedback} />
      ) : (
        <SettingsPanel
          section={settingsSection}
          onSectionChange={onSettingsSectionChange}
          state={state}
          onImageProfileChange={onImageProfileChange}
          onLlmChange={onLlmChange}
          onImageTagLlmChange={onImageTagLlmChange}
          onNovelAiChange={onNovelAiChange}
          onNotify={onNotify}
          onOpenPersonalSettings={onOpenPersonalSettings}
          onStateChange={onStateChange}
          onResetDemo={onResetDemo}
          jumpRequest={settingsJump}
          onJumpHandled={onSettingsJumpHandled}
          onRunLocalLlmTask={onRunLocalLlmTask}
        />
      )}
    </RailSubtabs>
  );
}

const SETTINGS_RAIL_TABS: ReadonlyArray<{ id: SettingsSection | "ops"; label: string }> = [
  ...SETTINGS_SECTIONS,
  { id: "ops", label: "진단" }
];

/**
 * The second row of tabs inside a rail group, and the tabpanel for whichever one is selected. It sticks to the top
 * of the scroller it sits in (the rail's panel, or the mobile drawer), so switching never needs a scroll back up.
 * `idPrefix` keeps the ids unique while the rail and the mobile drawer both hold one.
 */
function RailSubtabs<T extends string>({
  idPrefix,
  label,
  items,
  active,
  onSelect,
  children,
  variant = "rail"
}: {
  idPrefix: string;
  label: string;
  items: ReadonlyArray<{ id: T; label: string }>;
  active: T;
  onSelect: (id: T) => void;
  children: React.ReactNode;
  /** "builder" for a page-scrolled builder tab: same control, not pinned, sized for the wider column. */
  variant?: "rail" | "builder";
}) {
  return (
    <>
      <div className={variant === "builder" ? "rail-subtabs builder-subtabs" : "rail-subtabs"} role="tablist" aria-label={label} onKeyDown={handleTablistKeyDown}>
        {items.map((item) => {
          const selected = item.id === active;
          return (
            <button
              className={selected ? "active" : ""}
              key={item.id}
              type="button"
              role="tab"
              id={`subtab-${idPrefix}-${item.id}`}
              aria-controls={selected ? `subtabpanel-${idPrefix}-${item.id}` : undefined}
              aria-selected={selected}
              tabIndex={selected ? 0 : -1}
              onClick={() => onSelect(item.id)}
            >
              {item.label}
            </button>
          );
        })}
      </div>
      <div className="rail-subtab-panel" role="tabpanel" id={`subtabpanel-${idPrefix}-${active}`} aria-labelledby={`subtab-${idPrefix}-${active}`}>
        {children}
      </div>
    </>
  );
}

function MobileOpsDock({
  activePanel,
  children,
  open,
  onClose,
  onSelectPanel
}: {
  activePanel: RightPanel;
  children: React.ReactNode;
  open: boolean;
  onClose: () => void;
  onSelectPanel: (panel: RightPanel) => void;
}) {
  // The same three groups as the desktop rail. Seven tabs in a fixed 3-column dock wrapped to three rows and
  // covered the composer on a phone.
  const tabs: Array<{ panel: RightPanel; icon: React.ReactNode; label: string }> = [
    { panel: "image", icon: <ImageIcon size={17} />, label: "장면" },
    { panel: "memory", icon: <Brain size={17} />, label: "기억" },
    { panel: "settings", icon: <Settings2 size={17} />, label: "설정" }
  ];

  const opsDrawerRef = useModalDialog(open, onClose);

  return (
    <div className={`mobile-ops ${open ? "open" : ""}`}>
      <div className="mobile-ops-tabs" role="tablist" onKeyDown={handleTablistKeyDown} aria-label="모바일 운영 패널">
        {/* The drawer — the only thing these tabs control — is unmounted while closed, so aria-controls is
            emitted for the selected tab and only while there is a panel for it to resolve to. */}
        {tabs.map((tab) => (
          <button
            className={resolveRightPanelGroup(activePanel) === tab.panel ? "active" : ""}
            key={tab.panel}
            type="button"
            role="tab"
            id={`tab-mobile-ops-${tab.panel}`}
            aria-controls={open && resolveRightPanelGroup(activePanel) === tab.panel ? `tabpanel-mobile-ops-${tab.panel}` : undefined}
            aria-selected={resolveRightPanelGroup(activePanel) === tab.panel}
            tabIndex={resolveRightPanelGroup(activePanel) === tab.panel ? 0 : -1}
            onClick={() => onSelectPanel(tab.panel)}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>
      {open ? (
        <div className="mobile-ops-scrim" role="presentation" onClick={onClose}>
          <section
            className="mobile-ops-drawer"
            role="dialog"
            aria-modal="true"
            aria-label="운영 패널"
            ref={opsDrawerRef}
            onClick={(event) => event.stopPropagation()}
          >
            <header>
              {/* Matched to the tab strip above: the drawer holds seven panels but only three tabs, so a raw
                  activePanel lookup misses whenever the app parks on neuralmap / relationship / persona / ops
                  (session reset and every run switch do) and the header read a generic "운영 패널" that
                  disagreed with the highlighted tab and with the content below it. */}
              <strong>{tabs.find((tab) => tab.panel === resolveRightPanelGroup(activePanel))?.label ?? "운영"} 패널</strong>
              <button className="icon-button" type="button" onClick={onClose} aria-label="패널 닫기">
                <Minus size={17} />
              </button>
            </header>
            <div
              className="mobile-ops-content"
              role="tabpanel"
              id={`tabpanel-mobile-ops-${resolveRightPanelGroup(activePanel)}`}
              aria-labelledby={`tab-mobile-ops-${resolveRightPanelGroup(activePanel)}`}
            >
              {children}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Says, permanently and in place, when the app cannot do what the screen implies it is doing.
 *
 * A fresh install ships `llm: { enabled: false, provider: "mock" }` and `novelAi: { enabled: false }` while
 * `imageProfile.triggerMode` is "realtime_auto" (src/data/seed.ts). So the first turn returns locally
 * assembled placeholder prose and every turn queues an image that cannot be generated — and the only
 * warning was a toast that cleared itself after a couple of seconds. A new user concluded, reasonably, that
 * this is simply what the product writes and that images are broken.
 *
 * Deliberately not dismissable: it disappears by fixing the condition, not by acknowledging it.
 */
/**
 * The behaviour `role="dialog" aria-modal="true"` promises: Escape closes, focus starts inside, and Tab
 * cycles within instead of walking the page behind.
 *
 * Both dialogs declared the attributes and implemented none of it, so a keyboard or screen-reader user
 * could tab straight out of an "modal" dialog into the transcript underneath while the scrim still covered
 * it, and had no way to close without finding the mouse.
 */
function useModalDialog(open: boolean, onClose: () => void) {
  const containerRef = useRef<HTMLElement>(null);
  // Both call sites pass a fresh inline arrow, so `onClose` has a new identity on every parent render. With
  // it in the dependency array the effect tore down and re-ran on any App-level state change while the dialog
  // was open (a runtime notice expiring 2.8s later is enough) — each pass yanked focus back to the dialog's
  // first control and re-captured `previouslyFocused` as that same control, destroying the restore-to-opener
  // promise of the cleanup below. Reading the handler through a ref lets the effect depend only on `open`, so
  // it runs exactly once per open.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) {
      return;
    }
    const container = containerRef.current;
    const previouslyFocused = document.activeElement as HTMLElement | null;

    const focusableSelector =
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';
    const focusable = () =>
      Array.from(container?.querySelectorAll<HTMLElement>(focusableSelector) ?? []).filter(
        (element) => element.offsetParent !== null || element === document.activeElement
      );

    // Focus the first control rather than the container, so the first Tab moves forward inside the dialog.
    const initial = focusable()[0] ?? container;
    initial?.focus?.();

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onCloseRef.current();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const items = focusable();
      if (items.length === 0) {
        event.preventDefault();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (event.shiftKey && (active === first || !container?.contains(active))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
      // Hand focus back to whatever opened the dialog, so the keyboard user is not dropped at the page top.
      previouslyFocused?.focus?.();
    };
  }, [open]);

  return containerRef;
}

interface ImageBackendSetupNotice {
  id: string;
  title: string;
  detail: string;
  /** The backends whose settings fix it: NovelAI lives in personal settings, photoreal in the run's settings tab. */
  backends: ImageBackendId[];
}

/**
 * What the image routing promises that the connected backends cannot deliver, judged per ROUTED backend. Routing
 * names where cuts want to go; an off backend hands its cuts to the other one when that one is on (so the cut
 * still renders, just not as the screen implies), and with both off nothing renders at all. The one kind of cut that
 * never moves to photoreal is the floor's (a minor-coded figure in a sexual or exposing scene); no simulation is
 * expected to produce those, so they get no standing notice — the cut's own marker says why it waits.
 */
function describeImageBackendSetupNotices(state: AppState): ImageBackendSetupNotice[] {
  const routing = resolveImageBackendRouting(state.imageProfile);
  const novelAiOn = state.novelAi?.enabled === true;
  const photorealOn = state.photoreal?.enabled === true;
  const wantsPhotoreal = routing.mode === "photoreal" || (routing.mode === "mixed" && routing.photorealCategories.length > 0);
  const wantsNovelAi =
    routing.mode === "novelai" ||
    (routing.mode === "mixed" && IMAGE_SCENE_CATEGORY_OPTIONS.some((option) => !routing.photorealCategories.includes(option.id)));
  const floorCutNote = "미성년으로 읽히는 인물(학생·교복 포함)이 성적·노출 장면에 있는 컷만 실사로 생성하지 않으므로, 그런 컷은 NovelAI를 켤 때까지 대기합니다.";

  if (!novelAiOn && !photorealOn) {
    if (wantsPhotoreal && wantsNovelAi) {
      return [
        {
          id: "both-off",
          title: "이미지 생성이 켜져 있지만 NovelAI와 실사 이미지(ComfyUI)가 모두 연결되지 않았습니다.",
          detail: "매 턴 이미지 컷이 예약만 되고 생성되지 않습니다.",
          backends: ["novelai", "photoreal"]
        }
      ];
    }
    return [
      wantsPhotoreal
        ? {
            id: "photoreal-off",
            title: "이미지 생성이 켜져 있지만 실사 이미지(ComfyUI)가 연결되지 않았습니다.",
            detail: "매 턴 이미지 컷이 예약만 되고 생성되지 않습니다.",
            backends: ["photoreal"]
          }
        : {
            id: "novelai-off",
            title: "이미지 생성이 켜져 있지만 NovelAI가 연결되지 않았습니다.",
            detail: "매 턴 이미지 컷이 예약만 되고 생성되지 않습니다.",
            backends: ["novelai"]
          }
    ];
  }

  const notices: ImageBackendSetupNotice[] = [];
  if (wantsPhotoreal && !photorealOn) {
    notices.push({
      id: "photoreal-fallback",
      title: "실사 라우팅이 켜져 있지만 실사 이미지(ComfyUI)가 꺼져 있습니다.",
      detail: "실사로 보낼 컷도 지금은 NovelAI로 대신 생성합니다.",
      backends: ["photoreal"]
    });
  }
  // Mixed mode counts too: the scenes left unticked — sexual scenes by default — were routed to NovelAI on purpose, and
  // with NovelAI off they render photoreal by fallback, which the creator should hear before the first such cut.
  if (!novelAiOn && wantsNovelAi) {
    notices.push({
      id: "novelai-fallback",
      title:
        routing.mode === "novelai"
          ? "NovelAI가 꺼져 있어 'NAI만' 모드의 컷을 실사로 대신 생성합니다."
          : "NovelAI가 꺼져 있어 실사로 고르지 않은 장면도 실사로 대신 생성합니다.",
      detail: floorCutNote,
      backends: ["novelai"]
    });
  }
  return notices;
}

function CrackSetupNotice({
  state,
  onOpenPersonalSettings,
  onOpenImageSettings
}: {
  state: AppState;
  onOpenPersonalSettings: () => void;
  /** Opens the run's 설정 tab, where the photoreal (ComfyUI) section lives. */
  onOpenImageSettings: () => void;
}) {
  const llmUnconfigured = !state.llm.enabled || state.llm.provider === "mock";
  const imagesWanted =
    state.imageProfile.enabled &&
    state.simulation.realtimeImageEnabled &&
    state.imageProfile.triggerMode !== "stored_only" &&
    state.imageProfile.triggerMode !== "manual";
  const imageNotices = imagesWanted ? describeImageBackendSetupNotices(state) : [];
  const needsPhotorealSettings = imageNotices.some((notice) => notice.backends.includes("photoreal"));
  const needsNovelAiSettings = imageNotices.some((notice) => notice.backends.includes("novelai"));

  if (!llmUnconfigured && imageNotices.length === 0) {
    return null;
  }

  return (
    <div className="crack-setup-notice" role="status">
      <div>
        {llmUnconfigured ? (
          <p>
            <strong>아직 실제 모델에 연결되지 않았습니다.</strong> 지금 보이는 응답은 앱이 로컬에서 만든 예시이며, 시뮬레이션이
            실제로 진행되지는 않습니다.
          </p>
        ) : null}
        {imageNotices.map((notice) => (
          <p key={notice.id}>
            <strong>{notice.title}</strong> {notice.detail}
          </p>
        ))}
      </div>
      <div className="crack-setup-notice-actions">
        {/* The NovelAI token and the LLM key live in personal settings; ComfyUI has no secret, so its section is
            in the run's settings tab instead. Each notice gets the button that actually leads to its fix. */}
        {llmUnconfigured || needsNovelAiSettings ? (
          <button type="button" onClick={onOpenPersonalSettings}>
            설정 열기
          </button>
        ) : null}
        {needsPhotorealSettings ? (
          <button type="button" onClick={onOpenImageSettings}>
            실사 설정 열기
          </button>
        ) : null}
      </div>
    </div>
  );
}

function CrackPendingTurn({
  userText,
  state,
  manualImage,
  turnPhase
}: {
  userText: string;
  state: AppState;
  manualImage: boolean;
  turnPhase?: TurnPhase;
}) {
  const enabledModuleCount = state.modules.filter((module) => module.enabled && module.tokenPolicy !== "disabled").length;
  const referenceLabel = state.neuralMap.enabled ? "NeuralMap 참조" : "로컬 참조";
  const imageExpected = manualImage || state.imageProfile.triggerMode !== "stored_only";
  // Live image-job progress for the current turn (jobs update in React state during generation).
  const activeImageJobList = state.imageJobs.filter(
    (job) => job.status === "queued" || job.status === "planning" || job.status === "generating"
  );
  const activeImageJobs = activeImageJobList.length;
  // Per renderer, because the two queues run side by side and the photoreal one may be waiting on the local GPU.
  const activeImageJobSummary = (["novelai", "photoreal"] as const)
    .map((backend) => {
      const count = activeImageJobList.filter((job) => readImageJobBackend(job) === backend).length;
      return count > 0 ? `${imageBackendLabels[backend]} ${count}개` : undefined;
    })
    .filter((item): item is string => Boolean(item))
    .join(" · ");
  const renderedImageAssets = state.imageJobs
    .filter((job) => job.status === "generating" || job.status === "completed")
    .reduce((sum, job) => sum + job.assetIds.length, 0);

  // Phase ordering: retrieving → generating → images. Each step shows done / active / pending.
  const phase: TurnPhase = turnPhase ?? "retrieving";
  const order: Record<TurnPhase, number> = { retrieving: 0, generating: 1, images: 2 };
  const stepClass = (step: TurnPhase): string =>
    order[phase] > order[step] ? "done" : order[phase] === order[step] ? "active" : "";

  const headline =
    phase === "retrieving"
      ? "필요한 설정과 문맥을 검색하는 중"
      : phase === "generating"
        ? "응답과 Image Cue를 생성하는 중"
        : "이미지를 생성하는 중";
  const detail =
    phase === "retrieving"
      ? `${referenceLabel}로 활성 모듈 ${enabledModuleCount}개 중 이번 장면에 필요한 설정만 꺼내고 있습니다.`
      : phase === "generating"
        ? "대화 흐름을 쓰면서 이미지 컷 계획을 함께 정리하고 있습니다. 응답은 생성되는 대로 위에 나타납니다."
        : activeImageJobs > 0
          ? `이미지 작업 진행 중 (${activeImageJobSummary})${renderedImageAssets > 0 ? ` · ${renderedImageAssets}장 완료` : ""}.`
          : "이미지 작업을 마무리하는 중입니다.";

  return (
    <>
      <article className="crack-choice-message pending">
        <span className="crack-choice-edit" aria-hidden="true">
          <Pencil size={16} />
        </span>
        <div className="crack-choice-bubble">{userText}</div>
      </article>
      <article className="crack-generation-card" aria-live="polite">
        <div className="crack-generation-orbit" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div>
          <strong>{headline}</strong>
          <p>{detail}</p>
        </div>
        <div className="crack-generation-steps">
          <span className={stepClass("retrieving")}>
            <Brain size={13} />
            문맥 검색
          </span>
          <span className={stepClass("generating")}>
            <Sparkles size={13} />
            응답 생성
          </span>
          <span className={stepClass("images")}>
            <ImageIcon size={13} />
            {phase === "images" && activeImageJobs > 0
              ? `이미지 ${renderedImageAssets > 0 ? `${renderedImageAssets}장` : `${activeImageJobs}작업`}`
              : imageExpected
                ? "이미지 생성"
                : "저장 이미지"}
          </span>
        </div>
      </article>
    </>
  );
}

function DynamicRichText({ content }: { content: string }) {
  const segments = useMemo(() => parseDynamicRichText(content), [content]);

  return (
    <>
      {segments.map((segment) =>
        segment.kind === "markdown" ? (
          <MarkdownStageText content={segment.content} key={segment.id} />
        ) : (
          <DynamicTextEffectBlock content={segment.content} kind={segment.kind} key={segment.id} />
        )
      )}
    </>
  );
}

// Pure renderer: every caller normalizes once in parseDynamicRichText (prose via flushMarkdown, effect-block
// bodies where their segments are built). Normalizing again here was a second, compounding pass.
const MarkdownStageText = memo(function MarkdownStageText({ content }: { content: string }) {
  return (
    <ReactMarkdown components={dynamicMarkdownComponents} remarkPlugins={markdownRemarkPlugins}>
      {content}
    </ReactMarkdown>
  );
});

// Only `status` and `choice` carry a visual identity (see narrative-output.css); every other kind is
// unwrapped to plain narrative prose. The component still intercepts all of them so a ```status fence never
// falls through to the monospace <pre.rich-code-block> renderer.
const STYLED_DYNAMIC_TEXT_BLOCK_KINDS = new Set<DynamicTextBlockKind>(["status", "choice"]);

const DynamicTextEffectBlock = memo(function DynamicTextEffectBlock({ content, kind }: { content: string; kind: DynamicTextBlockKind }) {
  return (
    <div className="dynamic-text-block" data-kind={STYLED_DYNAMIC_TEXT_BLOCK_KINDS.has(kind) ? kind : undefined}>
      <div className="dynamic-text-content">
        <MarkdownStageText content={content} />
      </div>
    </div>
  );
});

function CrackTimelineMessage({
  message,
  turnNumber,
  assets,
  jobs,
  relatedJobs,
  evidenceCount,
  onCancelJob,
  onDraftChange,
  onFeedback,
  onNotify,
  onRegenerateImageJob,
  onRegenerateResponse,
  onRunJob,
  regenerationDisabled
}: {
  message: AppState["messages"][number];
  turnNumber?: number;
  assets: ImageAsset[];
  jobs: ImageGenerationJob[];
  relatedJobs: ImageGenerationJob[];
  evidenceCount: number;
  onCancelJob: (jobId: string) => void;
  onDraftChange: (value: string) => void;
  onFeedback: (assetId: string, rating: ImageFeedbackRating) => void;
  onNotify: (message: string) => void;
  onRegenerateImageJob: (job: ImageGenerationJob) => void;
  onRegenerateResponse: (assistantMessageId: string) => void;
  onRunJob: (job: ImageGenerationJob) => void;
  regenerationDisabled: boolean;
}) {
  const moreMenuRef = useRef<HTMLDetailsElement>(null);
  const latestReusableImageJob = useMemo(
    () =>
      relatedJobs
        .slice()
        .reverse()
        .find((job) => ["completed", "failed", "canceled"].includes(job.status)),
    [relatedJobs]
  );
  const closeMoreMenu = () => {
    if (moreMenuRef.current) {
      moreMenuRef.current.open = false;
    }
  };
  const copyResponse = () => {
    closeMoreMenu();
    if (!navigator.clipboard?.writeText) {
      onNotify("이 브라우저에서는 클립보드 복사를 사용할 수 없습니다.");
      return;
    }

    void navigator.clipboard
      .writeText(message.content)
      .then(() => onNotify("응답을 클립보드에 복사했습니다."))
      .catch(() => onNotify("브라우저 권한 때문에 클립보드에 복사하지 못했습니다."));
  };
  const useAsDraft = () => {
    closeMoreMenu();
    onDraftChange(message.content);
    onNotify("응답 내용을 입력창으로 가져왔습니다.");
  };
  const regenerateResponse = () => {
    closeMoreMenu();
    onRegenerateResponse(message.id);
  };
  const regenerateLatestImage = () => {
    if (!latestReusableImageJob) {
      return;
    }
    closeMoreMenu();
    onRegenerateImageJob(latestReusableImageJob);
  };

  if (message.role === "user") {
    return (
      <article className="crack-choice-message">
        <button className="crack-choice-edit" type="button" onClick={() => onDraftChange(message.content)} aria-label="선택 수정">
          <Pencil size={16} />
        </button>
        <button className="crack-choice-bubble" type="button" onClick={() => onDraftChange(message.content)}>
          {message.content}
        </button>
      </article>
    );
  }

  if (message.role === "system") {
    return (
      <article className="crack-system-note">
        <Brain size={16} />
        <span>{message.content}</span>
      </article>
    );
  }

  const narration = useMemo(() => createNarrationFlow(message.content, assets, jobs), [assets, jobs, message.content]);
  const narrationMedia = useMemo<NarrationMediaContextValue>(
    () => ({ slots: narration.markerSlots, onFeedback }),
    [narration.markerSlots, onFeedback]
  );

  return (
    /* A stable id per turn: the anchor a reader (or a future find-in-transcript) needs to return to a beat. */
    <article className="crack-narration-block" id={`turn-${message.id}`}>
      {turnNumber !== undefined ? (
        <div className="crack-turn-marker">
          <span>{turnNumber}턴</span>
          <time dateTime={message.createdAt}>
            {new Date(message.createdAt).toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}
          </time>
        </div>
      ) : null}
      <NarrationMediaContext.Provider value={narrationMedia}>
        {narration.items.map((item) =>
          item.kind === "markdown" ? (
            <div className="crack-markdown" key={item.id}>
              <MarkdownStageText content={item.content} />
            </div>
          ) : item.kind === "dynamic" ? (
            <DynamicTextEffectBlock content={item.content} kind={item.blockKind} key={item.id} />
          ) : item.kind === "image" ? (
            <CrackInlineImage asset={item.asset} key={item.id} onFeedback={onFeedback} />
          ) : (
            <CrackInlineImageJob
              job={item.job}
              key={item.id}
              onCancel={onCancelJob}
              onRegenerate={onRegenerateImageJob}
              onRun={onRunJob}
            />
          )
        )}
      </NarrationMediaContext.Provider>
      <div className="crack-message-controls">
        <span>{evidenceCount > 0 ? `근거 ${evidenceCount}개` : "문맥 준비됨"}</span>
        <button type="button" onClick={regenerateResponse} disabled={regenerationDisabled} aria-label="응답 재생성">
          <RefreshCcw size={16} />
        </button>
        <details className="crack-message-more-menu" ref={moreMenuRef}>
          <summary aria-label="메시지 더보기">
            <MoreHorizontal size={17} />
          </summary>
          <div className="crack-message-more-popover">
            <button type="button" onClick={copyResponse}>
              <Copy size={15} />
              응답 복사
            </button>
            <button type="button" onClick={useAsDraft}>
              <Pencil size={15} />
              입력창으로 가져오기
            </button>
            <button type="button" onClick={regenerateResponse} disabled={regenerationDisabled}>
              <RefreshCcw size={15} />
              응답 다시 생성
            </button>
            {latestReusableImageJob ? (
              <button type="button" onClick={regenerateLatestImage}>
                <ImageIcon size={15} />
                이미지 다시 생성
              </button>
            ) : null}
          </div>
        </details>
      </div>
    </article>
  );
}

const CrackInlineImage = memo(function CrackInlineImage({ asset, onFeedback }: { asset: ImageAsset; onFeedback: (assetId: string, rating: ImageFeedbackRating) => void }) {
  // Prefer the decoded image's real dimensions so the frame matches the actual output instead of falling back to a
  // square frame when providerMetadata lacks width/height (which would crop a portrait/landscape image into a square).
  const [measuredSize, setMeasuredSize] = useState<{ width: number; height: number }>();
  const style = {
    "--tone-a": asset.palette[0],
    "--tone-b": asset.palette[1],
    "--tone-c": asset.palette[2],
    "--image-aspect-ratio": measuredSize ? `${measuredSize.width} / ${measuredSize.height}` : createImageAssetAspectRatio(asset),
    // Numeric width/height ratio so CSS can size the frame as (height cap × ratio) and preserve the real
    // aspect ratio instead of collapsing a portrait image into a square when the column is widened.
    "--image-aspect-ratio-num": measuredSize ? measuredSize.width / measuredSize.height : createImageAssetAspectRatioNumber(asset)
  } as CSSProperties;

  return (
    <figure className="crack-inline-image" style={style}>
      <div className="crack-inline-image-frame">
        <AssetImage
          src={createImageAssetSrc(asset)}
          alt={asset.title}
          onNaturalSize={(width, height) => setMeasuredSize({ width, height })}
        />
        <ImageBackendBadge asset={asset} />
      </div>
      <figcaption>
        <span>{asset.title}</span>
        <div>
          <button type="button" onClick={() => onFeedback(asset.id, "liked")} aria-label="이미지 선호">
            <ThumbsUp size={15} />
          </button>
          <button type="button" onClick={() => onFeedback(asset.id, "rejected")} aria-label="이미지 제외">
            <ThumbsDown size={15} />
          </button>
        </div>
      </figcaption>
    </figure>
  );
});

function createImageAssetAspectRatio(asset: ImageAsset): string {
  const width = asset.providerMetadata ? readProviderPayloadNumber(asset.providerMetadata, "width") : undefined;
  const height = asset.providerMetadata ? readProviderPayloadNumber(asset.providerMetadata, "height") : undefined;
  return width && height ? `${Math.round(width)} / ${Math.round(height)}` : "1 / 1";
}

function createImageAssetAspectRatioNumber(asset: ImageAsset): number {
  const width = asset.providerMetadata ? readProviderPayloadNumber(asset.providerMetadata, "width") : undefined;
  const height = asset.providerMetadata ? readProviderPayloadNumber(asset.providerMetadata, "height") : undefined;
  return width && height ? width / height : 1;
}

const CrackInlineImageJob = memo(function CrackInlineImageJob({
  job,
  onCancel,
  onRegenerate,
  onRun
}: {
  job: ImageGenerationJob;
  onCancel: (jobId: string) => void;
  onRegenerate: (job: ImageGenerationJob) => void;
  onRun: (job: ImageGenerationJob) => void;
}) {
  const canCancel = ["queued", "planning", "generating"].includes(job.status);
  const canRegenerate = ["queued", "failed", "canceled", "completed"].includes(job.status);
  // Named per renderer: with two queues running side by side, "이미지 생성 대기" no longer says which one the cut is
  // waiting on — and a photoreal cut can be waiting on the local GPU, not on a provider.
  const jobBackend = readImageJobBackend(job);
  const backendLabel = imageBackendLabels[jobBackend];
  const setupBlockedReason =
    typeof job.providerPayload?.setupBlockedReason === "string" ? job.providerPayload.setupBlockedReason.trim() : "";
  const statusLabel =
    job.status === "queued"
      ? setupBlockedReason
        ? `${backendLabel} 설정 필요`
        : `${backendLabel} 이미지 생성 대기`
      : job.status === "planning"
        ? `${backendLabel} 이미지 준비 중`
        : job.status === "generating"
          ? `${backendLabel} 이미지 생성 중`
          : job.status === "failed"
            ? `${backendLabel} 이미지 생성 실패`
            : job.status === "canceled"
              ? "이미지 생성 취소됨"
              : `${backendLabel} 이미지 생성 완료`;
  // The specific reason the pipeline already wrote onto the job, which is what the reader needs here: the cut's
  // own description says what it would show, not why it is not showing.
  // A failure's lead is already the title, so the detail starts at the reason.
  const detail =
    job.status === "queued"
      ? setupBlockedReason
      : job.status === "failed"
        ? stripImageFailureLead(job.error ?? "", jobBackend)
        : "";
  return (
    <div className={`crack-inline-image-job ${job.status}`}>
      <ImageIcon size={18} />
      <div className="crack-inline-image-job-copy">
        <strong>{statusLabel}</strong>
        <p>{job.reason}</p>
        {detail && detail !== job.reason ? <p className="crack-inline-image-job-detail">{detail}</p> : null}
      </div>
      <div className="crack-inline-image-job-actions">
        {canUserRunQueuedImageJob(job) ? (
          <button type="button" onClick={() => onRun(job)}>
            생성
          </button>
        ) : null}
        {canRegenerate ? (
          <button type="button" onClick={() => onRegenerate(job)} aria-label="같은 프롬프트로 이미지 재생성" title="이미지 재생성">
            <RefreshCcw size={15} />
          </button>
        ) : null}
        {canCancel ? (
          <button type="button" onClick={() => onCancel(job.id)} aria-label="이미지 생성 취소" title="이미지 생성 취소">
            <Trash2 size={15} />
          </button>
        ) : null}
      </div>
    </div>
  );
});

function SimulationRunPage({
  state,
  latestAssets,
  draft,
  isSending,
  isResetting,
  manualImage,
  rightPanel,
  onCancelImageJob,
  onDeleteImageAsset,
  onDraftChange,
  onEditSimulation,
  onImageFeedback,
  onImageProfileChange,
  onLlmChange,
  onImageTagLlmChange,
  onNovelAiChange,
  onResetDemo,
  onResetSession,
  onRedactMemory,
  onRightPanelChange,
  onRunImageJob,
  onStateChange,
  onSubmit,
  onToggleManualImage
}: {
  state: AppState;
  latestAssets: ImageAsset[];
  draft: string;
  isSending: boolean;
  isResetting: boolean;
  manualImage: boolean;
  rightPanel: RightPanel;
  onCancelImageJob: (jobId: string) => void;
  onDeleteImageAsset: (assetId: string) => void;
  onDraftChange: (value: string) => void;
  onEditSimulation: () => void;
  onImageFeedback: (assetId: string, rating: ImageFeedbackRating) => void;
  onImageProfileChange: (patch: Partial<ImageGenerationProfile>) => void;
  onLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onImageTagLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onNovelAiChange: (patch: Partial<NovelAiApiSettings>) => void;
  onResetDemo: () => void;
  onResetSession: () => void;
  onRedactMemory: (memoryId: string) => void;
  onRightPanelChange: (panel: RightPanel) => void;
  onRunImageJob: (job: ImageGenerationJob) => void;
  onStateChange: (value: AppState | ((current: AppState) => AppState)) => void;
  onSubmit: (event: FormEvent) => void;
  onToggleManualImage: (value: boolean) => void;
}) {
  const selectedContextPack = state.contextPacks.find((pack) => pack.id === state.selectedContextPackId) ?? state.contextPacks.at(-1);
  const personaDisplayName = getPersonaDisplayName(state);
  const autoResetDecision = createAutoResetAgentSessionDecision(state);
  const composerTextareaRef = useRef<HTMLTextAreaElement>(null);
  const handleInsertActionNotation = useCallback(() => {
    insertActionNotation(composerTextareaRef.current, draft, onDraftChange);
  }, [draft, onDraftChange]);

  return (
    <section className="run-workspace story-run-workspace">
      <aside className="run-rail story-side-panel">
        <div className="panel-header">
          <div>
            <span className="section-kicker">Story Control</span>
            <h2>{state.simulation.title}</h2>
          </div>
          <StatusPill icon={<Play size={15} />} label="Live" tone="good" />
        </div>

        <div className="story-cover-block">
          <SceneCard asset={latestAssets[0] ?? seedState.imageAssets[0]} featured />
          <button className="icon-text-button full" type="button" onClick={onEditSimulation}>
            <Settings2 size={16} />
            시뮬레이션 수정
          </button>
        </div>

        <div className="run-stack">
          <SectionTitle icon={<Bot size={17} />} title="캐릭터" />
          {state.characters.map((character) => (
            <article className="run-card" key={character.id}>
              <strong>{character.name}</strong>
              <span>{character.role}</span>
              <p>{character.summary}</p>
              <small>캐릭터 프롬프트 기반</small>
            </article>
          ))}

          <SectionTitle icon={<Layers size={17} />} title="참조 모듈" />
          <div className="compact-list">
            {state.modules
              .filter((module) => module.enabled)
              .sort((a, b) => b.priority - a.priority)
              .slice(0, 6)
              .map((module) => (
                <div className="compact-row" key={module.id}>
                  <span>{module.kind.replaceAll("_", " ")}</span>
                  <strong>{module.title}</strong>
                  <small>{module.tokenPolicy} · p{module.priority}</small>
                </div>
              ))}
          </div>

          <SectionTitle icon={<Database size={17} />} title="세션 제어" />
          <div className="run-card">
            <p>{formatAutoResetStatus(autoResetDecision)} · {formatAutoResetDetail(autoResetDecision)}</p>
            <button className="icon-text-button full" type="button" onClick={onResetSession} disabled={isResetting || isSending}>
              <RefreshCcw size={16} />
              {isResetting ? "handoff 생성 중" : "에이전트 세션 초기화"}
            </button>
          </div>
        </div>
      </aside>

      <section className="chat-column story-chat-column">
        <div className="simulation-strip">
          <div>
            <span className="section-kicker">에피소드 대화</span>
            <h2>{state.messages.at(-1)?.role === "assistant" ? "장면이 응답을 기다립니다" : "다음 행동을 입력하세요"}</h2>
          </div>
          <div className="strip-metrics">
            <Metric label="Session" value={state.simulation.activeSessionId.replace("session_", "")} />
            <Metric label="Auto" value={autoResetDecision.status === "stable" ? "대기" : autoResetDecision.status === "soon" ? "임박" : "준비"} />
            <Metric label="Memories" value={state.memoryEvents.length.toString()} />
            <Metric label="Assets" value={state.imageAssets.length.toString()} />
          </div>
          <ModelQuickSwitch llm={state.llm} onChange={onLlmChange} />
        </div>

        <EpisodeHero state={state} asset={latestAssets[0]} />

        <div className="chat-feed">
          {state.messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="message-avatar">{message.role === "assistant" ? <Bot size={18} /> : message.role === "user" ? <MessageSquareText size={18} /> : <Brain size={18} />}</div>
              <div className="message-body">
                <div className="message-meta">
                  <strong>{message.role === "assistant" ? "Simulation Agent" : message.role === "user" ? personaDisplayName : "System"}</strong>
                  <span>{new Date(message.createdAt).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
                {message.role === "assistant" ? <DynamicRichText content={message.content} /> : <p>{message.content}</p>}
                {message.referencedNodeIds.length > 0 ? <span className="message-chip">문맥 {message.referencedNodeIds.length}</span> : null}
                {message.imageAssetIds.length > 0 ? <span className="message-chip">이미지 {message.imageAssetIds.length}</span> : null}
              </div>
            </article>
          ))}
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <div className="reply-suggestion-row">
            <div className="reply-suggestions" aria-label="추천 입력">
              {createReplySuggestions(state).map((suggestion) => (
                <button key={suggestion} type="button" onClick={() => onDraftChange(suggestion)}>
                  {suggestion}
                </button>
              ))}
            </div>
            <button
              className="action-notation-button"
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={handleInsertActionNotation}
              disabled={isSending || isResetting}
              aria-label="행동 입력 괄호 삽입"
              title="행동 입력 *()* 삽입"
            >
              <Parentheses size={16} />
            </button>
          </div>
          <textarea
            ref={composerTextareaRef}
            aria-label="사용자 메시지"
            value={draft}
            onChange={(event) => onDraftChange(event.target.value)}
            placeholder="시뮬레이션에 입력할 행동이나 대사를 작성하세요."
            disabled={isSending || isResetting}
          />
          <div className="composer-actions">
            <label className="checkline">
              <input checked={manualImage} type="checkbox" onChange={(event) => onToggleManualImage(event.target.checked)} />
              {state.imageProfile.triggerMode === "realtime_confirm" ? "이번 턴 이미지 생성 승인" : "이번 턴 이미지 생성 요청"}
            </label>
            <button className="send-button" disabled={isSending || isResetting} type="submit">
              <Send size={17} />
              {isResetting ? "handoff 중" : isSending ? "진행 중" : "전송"}
            </button>
          </div>
        </form>
      </section>

      <aside className="right-rail story-tool-panel">
        <div className="tabs" role="tablist" onKeyDown={handleTablistKeyDown} aria-label="오른쪽 패널">
          <TabButton active={rightPanel === "image"} icon={<ImageIcon size={17} />} label="이미지" onClick={() => onRightPanelChange("image")} />
          <TabButton active={rightPanel === "relationship"} icon={<Network size={17} />} label="관계" onClick={() => onRightPanelChange("relationship")} />
          <TabButton active={rightPanel === "neuralmap"} icon={<Database size={17} />} label="Neural" onClick={() => onRightPanelChange("neuralmap")} />
          <TabButton active={rightPanel === "memory"} icon={<Brain size={17} />} label="메모리" onClick={() => onRightPanelChange("memory")} />
          <TabButton active={rightPanel === "ops"} icon={<Activity size={17} />} label="운영" onClick={() => onRightPanelChange("ops")} />
          <TabButton active={rightPanel === "persona"} icon={<UserRound size={17} />} label="페르소나" onClick={() => onRightPanelChange("persona")} />
          <TabButton active={rightPanel === "settings"} icon={<Settings2 size={17} />} label="설정" onClick={() => onRightPanelChange("settings")} />
        </div>

        {rightPanel === "image" ? (
          <ImagePanel
            state={state}
            assets={latestAssets}
            jobs={state.imageJobs}
            onCancelJob={onCancelImageJob}
            onDeleteAsset={onDeleteImageAsset}
            onImageProfileChange={onImageProfileChange}
            onRegenerateJob={onRunImageJob}
            onRunJob={onRunImageJob}
            onFeedback={onImageFeedback}
          />
        ) : rightPanel === "memory" ? (
          <MemoryPanel state={state} selectedContextPackId={selectedContextPack?.id} onRedactMemory={onRedactMemory} />
        ) : rightPanel === "relationship" ? (
          <RelationshipMapPanel state={state} onStateChange={onStateChange as Dispatch<SetStateAction<AppState>>} />
        ) : rightPanel === "neuralmap" ? (
          <NeuralMapPanel
            state={state}
            isSending={isSending}
            pendingUserText={draft}
            onStateChange={onStateChange as Dispatch<SetStateAction<AppState>>}
          />
        ) : rightPanel === "ops" ? (
          <OperationalPanel state={state} onFeedback={onImageFeedback} />
        ) : rightPanel === "persona" ? (
          <PersonaPanel
            state={state}
            onNotify={() => undefined}
            onStateChange={onStateChange as Dispatch<SetStateAction<AppState>>}
          />
        ) : (
          <SettingsPanel
            state={state}
            onImageProfileChange={onImageProfileChange}
            onLlmChange={onLlmChange}
            onImageTagLlmChange={onImageTagLlmChange}
            onNovelAiChange={onNovelAiChange}
            onOpenPersonalSettings={() => undefined}
            onStateChange={onStateChange}
            onResetDemo={onResetDemo}
          />
        )}
      </aside>
    </section>
  );
}

function HomePage({
  activeSimulationId,
  simulations,
  onCreate,
  onCopy,
  onEdit,
  onOpen,
  onStartNewRun
}: {
  activeSimulationId: string;
  simulations: SimulationLibrary;
  onCreate: () => void;
  onCopy: (simulationId: string) => void;
  onEdit: (simulationId: string) => void;
  onOpen: (simulationId: string) => void;
  onStartNewRun: (simulationId: string) => void;
}) {
  const activeState = simulations.find((item) => item.simulation.id === activeSimulationId) ?? simulations[0] ?? seedState;
  const activeImageBackendsLabel = describeActiveImageBackends(activeState);
  const latestMessage = activeState.messages.at(-1);
  const latestAsset = activeState.imageAssets.at(-1) ?? seedState.imageAssets[0];
  const totalMessages = simulations.reduce((sum, item) => sum + item.messages.length, 0);
  const totalMemories = simulations.reduce((sum, item) => sum + item.memoryEvents.length, 0);
  const activeContextPack = activeState.contextPacks.at(-1);
  const activeTrace = activeState.turnTraces.at(-1);

  return (
    <section className="library-page">
      <aside className="library-nav" aria-label="시뮬레이션 탐색">
        <div className="discover-nav-brand">
          <span className="brand-mark small">
            <Sparkles size={17} />
          </span>
          <strong>DynamicChat</strong>
        </div>
        {/* Four buttons used to sit here — 내 시뮬레이션 / 제작 도구 / 장면 로그 / Prompt Tree — none with an
            onClick and the first with a hardcoded "active". The first screen taught the user that clicking
            things in this app does nothing. The two with a real destination are wired; the two that only
            named the page you were already on, or a concept with no screen behind it, are gone. */}
        <button className="discover-nav-item" type="button" onClick={() => onEdit(activeState.simulation.id)}>
          <WandSparkles size={17} />
          제작 도구
        </button>
        <button className="discover-nav-item" type="button" onClick={() => onOpen(activeState.simulation.id)}>
          <MessageSquareText size={17} />
          장면 로그
        </button>
        <div className="discover-nav-card">
          <strong>시뮬레이션 {simulations.length}개</strong>
          <span>대화 {totalMessages}개 · 기억 {totalMemories}개</span>
        </div>
      </aside>

      <section className="library-main">
        <section className="library-hero">
          <div className="library-hero-copy">
            <span className="section-kicker">Dynamic Runtime</span>
            <h2>DynamicChat</h2>
            <p>{PRODUCT_HOME_DESCRIPTION}</p>
            <div className="library-active-story">
              <span>현재 실행</span>
              <strong>{activeState.simulation.title}</strong>
              <small>{activeState.simulation.description}</small>
            </div>
            <div className="library-hero-meta">
              <StatusPill icon={<MessageSquareText size={15} />} label={`대화 ${activeState.messages.length}개`} tone="neutral" />
              <StatusPill icon={<Brain size={15} />} label={`기억 ${activeState.memoryEvents.length}개`} tone="neutral" />
              <StatusPill icon={<ImageIcon size={15} />} label={activeImageBackendsLabel ?? "기본 에셋"} tone={activeImageBackendsLabel ? "good" : "neutral"} />
            </div>
            <div className="discover-actions">
              <button className="send-button" type="button" onClick={() => onOpen(activeState.simulation.id)}>
                <Play size={17} />
                이어서 진행
              </button>
              <button className="icon-text-button" type="button" onClick={() => onStartNewRun(activeState.simulation.id)}>
                <Plus size={17} />
                새 진행 시작
              </button>
              <button className="icon-text-button" type="button" onClick={() => onEdit(activeState.simulation.id)}>
                <Settings2 size={17} />
                수정
              </button>
              <button className="icon-text-button" type="button" onClick={() => onCopy(activeState.simulation.id)}>
                <Copy size={17} />
                복사
              </button>
              <button className="icon-text-button" type="button" onClick={onCreate}>
                <WandSparkles size={17} />
                새 시뮬레이션
              </button>
            </div>
          </div>
          <div className="library-hero-art">
            <SceneCard asset={latestAsset} featured />
            <div className="library-hero-art-meta">
              <Metric label="Context" value={activeContextPack?.source ?? "대기"} />
              <Metric label="Selected" value={activeTrace ? `${activeTrace.metrics.selectedModuleCount}개` : "0개"} />
            </div>
          </div>
        </section>

        <section className="library-section">
          <div className="shelf-header">
            <div>
              <span className="section-kicker">라이브러리</span>
              <h3>만든 시뮬레이션</h3>
            </div>
            <button className="icon-text-button" type="button" onClick={onCreate}>
              <Plus size={16} />
              만들기
            </button>
          </div>
          <div className="simulation-library-grid">
            {simulations.map((item) => (
              <SimulationLibraryCard
                active={item.simulation.id === activeSimulationId}
                key={item.simulation.id}
                state={item}
                onCopy={() => onCopy(item.simulation.id)}
                onEdit={() => onEdit(item.simulation.id)}
                onOpen={() => onOpen(item.simulation.id)}
                onStartNewRun={() => onStartNewRun(item.simulation.id)}
              />
            ))}
          </div>
        </section>

        <section className="home-status-row">
          <Capability label="현재 프롬프트 모듈" value={`${activeState.modules.length}개`} />
          <Capability label="라이브러리 기억" value={`${totalMemories}개`} />
          <Capability label="이미지 생성 작업" value={`${activeState.imageJobs.length}개`} />
          <Capability label="최근 장면" value={latestMessage ? new Date(latestMessage.createdAt).toLocaleDateString("ko-KR") : "없음"} />
        </section>
      </section>
    </section>
  );
}

function SimulationLibraryCard({
  active,
  state,
  onCopy,
  onEdit,
  onOpen,
  onStartNewRun
}: {
  active: boolean;
  state: AppState;
  onCopy: () => void;
  onEdit: () => void;
  onOpen: () => void;
  onStartNewRun: () => void;
}) {
  const asset = state.imageAssets.at(-1) ?? seedState.imageAssets[0];
  const character = state.characters[0];
  const style = {
    "--tone-a": asset?.palette[0] ?? "#26343a",
    "--tone-b": asset?.palette[1] ?? "#7b8da0",
    "--tone-c": asset?.palette[2] ?? "#d2a84b"
  } as CSSProperties;

  return (
    <article className={`simulation-library-card ${active ? "active" : ""}`} style={style}>
      <button className="simulation-card-cover" type="button" onClick={onOpen}>
        <AssetImage src={createImageAssetSrc(asset)} alt={state.simulation.title} />
        <span>{active ? "진행 중" : describeActiveImageBackends(state) ?? "로컬"}</span>
      </button>
      <div className="simulation-card-copy">
        <div>
          <strong>{state.simulation.title}</strong>
          <small>{character?.name ?? "캐릭터"} · 대화 {state.messages.length}개</small>
        </div>
        <p>{state.simulation.description}</p>
        <div className="simulation-card-actions">
          <button className="send-button" type="button" onClick={onOpen}>
            <Play size={16} />
            진행
          </button>
          <button className="icon-text-button" type="button" onClick={onStartNewRun}>
            <Plus size={16} />
            새 진행
          </button>
          <button className="icon-text-button" type="button" onClick={onEdit}>
            <Settings2 size={16} />
            수정
          </button>
          <button className="icon-text-button" type="button" onClick={onCopy}>
            <Copy size={16} />
            복사
          </button>
        </div>
      </div>
    </article>
  );
}

function CharacterDiscoveryCard({
  title,
  subtitle,
  description,
  metric,
  asset,
  tone,
  onClick
}: {
  title: string;
  subtitle: string;
  description: string;
  metric: string;
  asset?: ImageAsset;
  tone: string;
  onClick: () => void;
}) {
  const style = {
    "--tone-a": asset?.palette[0] ?? "#26343a",
    "--tone-b": asset?.palette[1] ?? "#7b8da0",
    "--tone-c": asset?.palette[2] ?? "#d2a84b"
  } as CSSProperties;

  return (
    <button className="character-discovery-card" type="button" style={style} onClick={onClick}>
      <div className="character-cover">
        <AssetImage src={createImageAssetSrc(asset)} alt={title} />
        <span>{tone}</span>
      </div>
      <div className="character-discovery-copy">
        <strong>{title}</strong>
        <small>{subtitle} · {metric}</small>
        <p>{description}</p>
      </div>
    </button>
  );
}

function ModelQuickSwitch({
  llm,
  onChange
}: {
  llm: LlmApiSettings;
  onChange: (patch: Partial<LlmApiSettings>) => void;
}) {
  const provider = getLlmProviderOption(llm.provider);
  const availableModels = provider.models.filter((model) => model !== "custom");
  const knownModel = availableModels.includes(llm.model) ? llm.model : "custom";

  return (
    <div className="quick-model-control">
      <span>모델</span>
      {/* The 모델 caption is a bare <span>, so it names neither control here. The sibling input already
          carried its own aria-label; this one is deliberately worded differently so the two are not
          announced identically. */}
      <select
        aria-label="실행 중 LLM 모델 프리셋"
        value={knownModel}
        onChange={(event) => {
          if (event.target.value === "custom") {
            onChange({ model: "" });
            return;
          }
          onChange({ model: event.target.value });
        }}
      >
        {availableModels.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
        <option value="custom">직접 입력</option>
      </select>
      <input value={llm.model} onChange={(event) => onChange({ model: event.target.value })} aria-label="실행 중 LLM 모델" />
    </div>
  );
}

function EpisodeHero({ state, asset }: { state: AppState; asset?: ImageAsset }) {
  const style = {
    "--tone-a": asset?.palette[0] ?? "#26343a",
    "--tone-b": asset?.palette[1] ?? "#7b8da0",
    "--tone-c": asset?.palette[2] ?? "#d2a84b"
  } as CSSProperties;
  const leadCharacter = state.characters[0];

  return (
    <section className="episode-hero" style={style}>
      <div className="episode-art">
        <AssetImage src={createImageAssetSrc(asset)} alt={asset?.title ?? state.simulation.title} />
      </div>
      <div className="episode-copy">
        <span className="section-kicker">에피소드</span>
        <h3>{state.simulation.title}</h3>
        <p>{state.simulation.description}</p>
        <div className="episode-tags">
          <StatusPill icon={<Bot size={15} />} label={leadCharacter?.name ?? "캐릭터"} tone="neutral" />
          <StatusPill icon={<Brain size={15} />} label={`근거 ${state.contextPacks.at(-1)?.evidence.length ?? 0}개`} tone="neutral" />
          <StatusPill icon={<ImageIcon size={15} />} label={asset ? imageAssetSourceLabels[asset.source] : "기본 에셋"} tone="neutral" />
        </div>
      </div>
    </section>
  );
}

function GuideIcon({ label, detail }: { label: string; detail: string }) {
  return (
    <span className="guide-icon" tabIndex={0} aria-label={label}>
      <Info size={14} aria-hidden="true" />
      <span className="guide-tooltip" role="tooltip">
        {detail}
      </span>
    </span>
  );
}

function GuidedFieldLabel({
  title,
  badge,
  guide,
  icon,
  charCount
}: {
  title: string;
  badge?: string;
  guide: string;
  icon?: React.ReactNode;
  charCount?: React.ReactNode;
}) {
  return (
    <span className="guided-label">
      <span className="guided-label-title">
        {icon ? <span className="guided-label-icon">{icon}</span> : null}
        <strong>{title}</strong>
        {badge ? <em>{badge}</em> : null}
      </span>
      {charCount}
      <GuideIcon label={`${title} 가이드`} detail={guide} />
    </span>
  );
}

// Shows the current character count for a prompt field, and — when a soft excerpt budget applies to that field
// — the budget as its maximum. A body past the budget is not rejected; each turn only the scene-relevant
// windows are excerpted into the prompt, which the overflow note makes explicit.
function PromptCharCount({ value, limit }: { value: string; limit?: number | null }) {
  const count = value.length;
  if (limit == null) {
    return <span className="prompt-char-count">{count.toLocaleString()}자</span>;
  }
  const over = count > limit;
  return (
    <span className={`prompt-char-count${over ? " over" : ""}`}>
      {count.toLocaleString()} / {limit.toLocaleString()}자
      {over ? <em>초과분은 턴마다 발췌 적용</em> : null}
    </span>
  );
}

function PromptAuthoringField({
  badge,
  guide,
  icon,
  placeholder,
  tone,
  title,
  value,
  onChange
}: {
  badge: string;
  guide: string;
  icon: React.ReactNode;
  placeholder: string;
  tone: "main" | "world";
  title: string;
  value: string;
  onChange: (value: string) => void;
}) {
  // The main-prompt field maps to the main_prompt module budget, the world field to the world_lore budget.
  const limit = tone === "main" ? MAX_MAIN_PROMPT_BODY_CHARS : MAX_FOUNDATION_MODULE_BODY_CHARS;
  return (
    <label className={`prompt-authoring-field ${tone}`}>
      <GuidedFieldLabel badge={badge} guide={guide} icon={icon} title={title} charCount={<PromptCharCount value={value} limit={limit} />} />
      <textarea placeholder={placeholder} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function PromptMeaningGuide() {
  return (
    <div className="prompt-meaning-grid" aria-label="프롬프트 입력 구분">
      <div className="prompt-meaning-card main">
        <span>항상 적용</span>
        <strong>메인 프롬프트</strong>
        <p>LLM이 매 턴 따라야 하는 진행 규칙, 출력 형식, 금지/허용 경계.</p>
      </div>
      <div className="prompt-meaning-card world">
        <span>필요할 때 참조</span>
        <strong>세계관/로어</strong>
        <p>장면의 사실 근거가 되는 장소, 역사, 세력, 문화, 시스템 설정.</p>
      </div>
    </div>
  );
}

type SceneTagPresetTreeProps = {
  presets: ImageSceneTagPreset[];
  onAddRoot: () => void;
  onAddChild: (presetId: string) => void;
  onChange: (presetId: string, patch: Partial<ImageSceneTagPresetNode>) => void;
  onDelete: (presetId: string) => void;
  onMove: (sourceId: string, targetId: string, position: ImageScenePresetDropPosition) => void;
};

function SceneTagPresetTree({ presets, onAddRoot, onAddChild, onChange, onDelete, onMove }: SceneTagPresetTreeProps) {
  const [collapsedPresetIds, setCollapsedPresetIds] = useState<Set<string>>(() => new Set());
  const [draggedPresetId, setDraggedPresetId] = useState<string | undefined>();
  const [presetDropTarget, setPresetDropTarget] = useState<{ presetId: string; position: ImageScenePresetDropPosition } | undefined>();
  const presetIds = useMemo(() => collectImageScenePresetNodeIds(presets), [presets]);
  const descendantIdsById = useMemo(() => createImageScenePresetDescendantMap(presets), [presets]);
  const allCollapsed = presetIds.length > 0 && presetIds.every((presetId) => collapsedPresetIds.has(presetId));

  useEffect(() => {
    const activeIds = new Set(presetIds);
    setCollapsedPresetIds((current) => {
      const next = new Set([...current].filter((presetId) => activeIds.has(presetId)));
      return next.size === current.size ? current : next;
    });
  }, [presetIds]);

  const togglePresetCollapsed = useCallback((presetId: string, descendantIds: string[]) => {
    setCollapsedPresetIds((current) => {
      const next = new Set(current);
      if (next.has(presetId)) {
        next.delete(presetId);
      } else {
        next.add(presetId);
        descendantIds.forEach((descendantId) => next.add(descendantId));
      }
      return next;
    });
  }, []);

  const collapseAllPresets = useCallback(() => {
    setCollapsedPresetIds(new Set(presetIds));
  }, [presetIds]);

  const expandAllPresets = useCallback(() => {
    setCollapsedPresetIds(new Set());
  }, []);

  const addChildAndExpand = useCallback(
    (presetId: string) => {
      setCollapsedPresetIds((current) => {
        if (!current.has(presetId)) {
          return current;
        }
        const next = new Set(current);
        next.delete(presetId);
        return next;
      });
      onAddChild(presetId);
    },
    [onAddChild]
  );

  const canDropPreset = useCallback(
    (sourceId: string | undefined, targetId: string) =>
      Boolean(sourceId && sourceId !== targetId && !descendantIdsById.get(sourceId)?.has(targetId)),
    [descendantIdsById]
  );

  const handlePresetDragStart = useCallback((event: ReactDragEvent<HTMLElement>, presetId: string) => {
    setDraggedPresetId(presetId);
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("application/x-dynamicchat-scene-preset", presetId);
    event.dataTransfer.setData("text/plain", presetId);
  }, []);

  const handlePresetDragOver = useCallback(
    (event: ReactDragEvent<HTMLElement>, targetId: string) => {
      event.stopPropagation();
      if (!canDropPreset(draggedPresetId, targetId)) {
        return;
      }

      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      const position = getImageScenePresetDropPosition(event);
      setPresetDropTarget((current) =>
        current?.presetId === targetId && current.position === position ? current : { presetId: targetId, position }
      );
    },
    [canDropPreset, draggedPresetId]
  );

  const handlePresetDrop = useCallback(
    (event: ReactDragEvent<HTMLElement>, targetId: string) => {
      event.preventDefault();
      event.stopPropagation();
      const sourceId =
        draggedPresetId ||
        event.dataTransfer.getData("application/x-dynamicchat-scene-preset") ||
        event.dataTransfer.getData("text/plain");

      if (canDropPreset(sourceId, targetId)) {
        const position = getImageScenePresetDropPosition(event);
        onMove(sourceId, targetId, position);
        if (position === "inside") {
          setCollapsedPresetIds((current) => {
            if (!current.has(targetId)) {
              return current;
            }
            const next = new Set(current);
            next.delete(targetId);
            return next;
          });
        }
      }

      setDraggedPresetId(undefined);
      setPresetDropTarget(undefined);
    },
    [canDropPreset, draggedPresetId, onMove]
  );

  const handlePresetDragEnd = useCallback(() => {
    setDraggedPresetId(undefined);
    setPresetDropTarget(undefined);
  }, []);

  return (
    <section className="builder-panel span-2 scene-tag-preset-panel">
      <div className="runtime-card-subhead">
        <strong>장면 태그 키워드</strong>
        <div className="scene-tag-preset-toolbar">
          {presetIds.length > 0 ? (
            <button className="icon-text-button" type="button" onClick={allCollapsed ? expandAllPresets : collapseAllPresets}>
              {allCollapsed ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
              {allCollapsed ? "전체 펼치기" : "전체 접기"}
            </button>
          ) : null}
          <button className="icon-text-button" type="button" onClick={onAddRoot}>
            <Plus size={16} />
            키워드 추가
          </button>
        </div>
      </div>
      <div className="scene-tag-preset-list">
        {presets.length > 0 ? (
          presets.map((preset) => (
            <SceneTagPresetNodeEditor
              collapsedIds={collapsedPresetIds}
              depth={0}
              draggedPresetId={draggedPresetId}
              dropTarget={presetDropTarget}
              key={preset.id}
              node={preset}
              onAddChild={addChildAndExpand}
              onChange={onChange}
              onDelete={onDelete}
              onDragEnd={handlePresetDragEnd}
              onDragOver={handlePresetDragOver}
              onDragStart={handlePresetDragStart}
              onDrop={handlePresetDrop}
              onToggleCollapsed={togglePresetCollapsed}
            />
          ))
        ) : (
          <div className="empty-panel compact">
            <strong>장면 키워드 없음</strong>
            <button className="icon-text-button" type="button" onClick={onAddRoot}>
              <Plus size={16} />
              키워드 추가
            </button>
          </div>
        )}
      </div>
      <p className="settings-note">
        하위 키워드는 원하는 만큼 만들 수 있고, LLM에는 현재 문맥과 가까운 가지를 우선해 압축 전달합니다.
      </p>
    </section>
  );
}

const SceneTagPresetNodeEditor = memo(function SceneTagPresetNodeEditor({
  collapsedIds,
  depth,
  draggedPresetId,
  dropTarget,
  node,
  onAddChild,
  onChange,
  onDelete,
  onDragEnd,
  onDragOver,
  onDragStart,
  onDrop,
  onToggleCollapsed
}: {
  collapsedIds: Set<string>;
  depth: number;
  draggedPresetId?: string;
  dropTarget?: { presetId: string; position: ImageScenePresetDropPosition };
  node: ImageSceneTagPresetNode;
  onAddChild: (presetId: string) => void;
  onChange: (presetId: string, patch: Partial<ImageSceneTagPresetNode>) => void;
  onDelete: (presetId: string) => void;
  onDragEnd: () => void;
  onDragOver: (event: ReactDragEvent<HTMLElement>, targetId: string) => void;
  onDragStart: (event: ReactDragEvent<HTMLElement>, presetId: string) => void;
  onDrop: (event: ReactDragEvent<HTMLElement>, targetId: string) => void;
  onToggleCollapsed: (presetId: string, descendantIds: string[]) => void;
}) {
  const children = node.children ?? [];
  const collapsed = collapsedIds.has(node.id);
  const descendantIds = useMemo(() => collectImageScenePresetNodeIds(children), [children]);
  const title = node.keyword.trim() || "새 키워드";
  const dropPosition = dropTarget?.presetId === node.id ? dropTarget.position : undefined;
  const className = [
    "scene-tag-preset-node",
    collapsed ? "is-collapsed" : "",
    draggedPresetId === node.id ? "dragging" : "",
    dropPosition ? `drop-${dropPosition}` : ""
  ]
    .filter(Boolean)
    .join(" ");
  const exampleFiles = node.exampleFiles ?? [];
  const updateExampleFiles = (files: ImageScenePresetExampleFile[]) => onChange(node.id, { exampleFiles: files });
  const addExampleFile = () =>
    updateExampleFiles([...exampleFiles, { id: createId("scene_example_file"), label: "", prompts: [] }]);
  const patchExampleFile = (fileId: string, patch: Partial<ImageScenePresetExampleFile>) =>
    updateExampleFiles(exampleFiles.map((file) => (file.id === fileId ? { ...file, ...patch } : file)));
  const removeExampleFile = (fileId: string) =>
    updateExampleFiles(exampleFiles.filter((file) => file.id !== fileId));
  const importExampleFile = (fileId: string, text: string) => {
    const existing = exampleFiles.find((file) => file.id === fileId)?.prompts ?? [];
    const imported = text.split(/\r?\n/u);
    const seen = new Set<string>();
    const merged = [...existing, ...imported]
      .map((line) => line.trim())
      .filter((line) => {
        const key = line.toLowerCase();
        if (!line || seen.has(key)) {
          return false;
        }
        seen.add(key);
        return true;
      });
    patchExampleFile(fileId, { prompts: merged });
  };
  return (
    <div
      className={className}
      style={{ "--scene-preset-depth": depth } as CSSProperties}
      onDragOver={(event) => onDragOver(event, node.id)}
      onDrop={(event) => onDrop(event, node.id)}
    >
      <div className="scene-tag-preset-row">
        <div className="scene-tag-preset-row-head">
          <button
            aria-label={`${title} 이동`}
            className="icon-button subtle scene-tag-preset-drag-handle"
            draggable
            title="드래그해서 이동"
            type="button"
            onDragEnd={onDragEnd}
            onDragStart={(event) => onDragStart(event, node.id)}
          >
            <GripVertical size={16} />
          </button>
          <button
            aria-expanded={!collapsed}
            aria-label={`${title} ${collapsed ? "펼치기" : "접기"}`}
            className="icon-button subtle scene-tag-preset-toggle"
            title={collapsed ? "펼치기" : "접기"}
            type="button"
            onClick={() => onToggleCollapsed(node.id, descendantIds)}
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
          </button>
          <label className="checkline">
            <input checked={node.enabled} type="checkbox" onChange={(event) => onChange(node.id, { enabled: event.target.checked })} />
            사용
          </label>
          <span className="scene-tag-preset-level">
            {depth === 0 ? "상위 키워드" : `${depth + 1}단계`}
            <span className="scene-tag-preset-title">{title}</span>
            {children.length > 0 ? <span className="scene-tag-preset-child-count">{children.length}개 하위</span> : null}
          </span>
          <div className="scene-tag-preset-actions">
            <button className="icon-text-button" type="button" onClick={() => onAddChild(node.id)}>
              <Plus size={15} />
              하위
            </button>
            <button
              aria-label={`${node.keyword || "장면 키워드"} 삭제`}
              className="icon-button subtle"
              type="button"
              onClick={() => onDelete(node.id)}
            >
              <Trash2 size={15} />
            </button>
          </div>
        </div>
        {collapsed ? (
          <div className="scene-tag-preset-collapsed-summary">
            {[node.tags.length > 0 ? `태그 ${node.tags.length}개` : undefined, (node.exampleFiles ?? []).length > 0 ? `예시파일 ${(node.exampleFiles ?? []).length}개` : undefined, node.note.trim() ? "메모 있음" : undefined, children.length > 0 ? `하위 ${children.length}개 숨김` : undefined]
              .filter(Boolean)
              .join(" · ") || "접힌 키워드"}
          </div>
        ) : (
          <>
            <div className="two-fields">
              <label>
                키워드
                <input value={node.keyword} onChange={(event) => onChange(node.id, { keyword: event.target.value })} placeholder="expression" />
              </label>
              <label>
                우선순위
                <input
                  max="120"
                  min="0"
                  type="number"
                  value={node.priority}
                  onChange={(event) => onChange(node.id, { priority: Number(event.target.value) })}
                />
              </label>
            </div>
            <label>
              기본 장면 태그
              <TagListTextarea
                placeholder="rain, wet street, city lights, night, reflection"
                rows={3}
                tags={node.tags}
                onCommit={(tags) => onChange(node.id, { tags })}
              />
            </label>
            <label>
              메모
              <textarea
                rows={2}
                value={node.note}
                onChange={(event) => onChange(node.id, { note: event.target.value })}
                placeholder="활용 방식, 추천 변형, 와일드카드 규칙, 함께 쓰면 좋은 태그"
              />
            </label>
            <div className="scene-tag-preset-examples">
              <div className="scene-tag-preset-examples-head">
                <span>예시 프롬프트 파일 (한 키워드에 여러 개 가능 — 예: 여성/남성)</span>
                <button type="button" className="icon-text-button subtle" onClick={addExampleFile}>
                  파일 추가
                </button>
              </div>
              {exampleFiles.length === 0 ? (
                <p className="scene-tag-preset-examples-empty">예시가 없어도 됩니다. 있으면 LLM이 스타일 참고용으로만 사용합니다.</p>
              ) : null}
              {exampleFiles.map((file) => (
                <div className="scene-tag-preset-example-file" key={file.id}>
                  <div className="scene-tag-preset-example-file-head">
                    <input
                      value={file.label}
                      placeholder="라벨 (예: 여성, 남성, 장면)"
                      onChange={(event) => patchExampleFile(file.id, { label: event.target.value })}
                    />
                    <label className="scene-tag-preset-example-import">
                      .txt
                      <input
                        type="file"
                        accept=".txt,text/plain"
                        onChange={(event) => {
                          const picked = event.target.files?.[0];
                          if (!picked) {
                            return;
                          }
                          const reader = new FileReader();
                          reader.onload = () => importExampleFile(file.id, typeof reader.result === "string" ? reader.result : "");
                          reader.readAsText(picked);
                          event.target.value = "";
                        }}
                      />
                    </label>
                    <button type="button" className="icon-text-button subtle" onClick={() => removeExampleFile(file.id)}>
                      삭제
                    </button>
                  </div>
                  <textarea
                    rows={4}
                    value={file.prompts.join("\n")}
                    onChange={(event) => patchExampleFile(file.id, { prompts: event.target.value.split(/\r?\n/u) })}
                    placeholder={"설명 없이 완성 프롬프트만, 줄바꿈으로 구분\nmissionary position, spread legs, ...\nfrom front, lying on back, ..."}
                  />
                </div>
              ))}
            </div>
          </>
        )}
      </div>
      {!collapsed && children.length > 0 ? (
        <div className="scene-tag-preset-children">
          {children.map((child) => (
            <SceneTagPresetNodeEditor
              collapsedIds={collapsedIds}
              depth={depth + 1}
              draggedPresetId={draggedPresetId}
              dropTarget={dropTarget}
              key={child.id}
              node={child}
              onAddChild={onAddChild}
              onChange={onChange}
              onDelete={onDelete}
              onDragEnd={onDragEnd}
              onDragOver={onDragOver}
              onDragStart={onDragStart}
              onDrop={onDrop}
              onToggleCollapsed={onToggleCollapsed}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
});

function TagListTextarea({
  tags,
  onCommit,
  placeholder,
  rows
}: {
  tags: string[];
  onCommit: (tags: string[]) => void;
  placeholder?: string;
  rows?: number;
}) {
  const formattedTags = useMemo(() => tags.join(", "), [tags]);
  const [draftTags, setDraftTags] = useState(formattedTags);

  useEffect(() => {
    setDraftTags(formattedTags);
  }, [formattedTags]);

  return (
    <textarea
      placeholder={placeholder}
      rows={rows}
      value={draftTags}
      onBlur={() => onCommit(parseScenePresetTags(draftTags))}
      onChange={(event) => {
        const nextValue = event.target.value;
        setDraftTags(nextValue);
        if (!shouldDeferDelimitedTagCommit(nextValue)) {
          onCommit(parseScenePresetTags(nextValue));
        }
      }}
    />
  );
}

function ActivationTagInput({
  tags,
  onCommit
}: {
  tags: string[];
  onCommit: (tags: string[]) => void;
}) {
  const formattedTags = useMemo(() => formatActivationTags(tags), [tags]);
  const [draftTags, setDraftTags] = useState(formattedTags);

  useEffect(() => {
    setDraftTags(formattedTags);
  }, [formattedTags]);

  return (
    <input
      value={draftTags}
      onBlur={() => onCommit(parseActivationTags(draftTags))}
      onChange={(event) => {
        const nextValue = event.target.value;
        setDraftTags(nextValue);
        if (!shouldDeferDelimitedTagCommit(nextValue)) {
          onCommit(parseActivationTags(nextValue));
        }
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function shouldDeferDelimitedTagCommit(value: string): boolean {
  return /(?:[,;\n]|\s)$/u.test(value);
}

function getImageScenePresetDropPosition(event: ReactDragEvent<HTMLElement>): ImageScenePresetDropPosition {
  const bounds = event.currentTarget.getBoundingClientRect();
  const ratio = bounds.height > 0 ? (event.clientY - bounds.top) / bounds.height : 0.5;
  if (ratio < 0.25) {
    return "before";
  }
  if (ratio > 0.75) {
    return "after";
  }
  return "inside";
}

function CreateSimulationPage({
  mode,
  initialDraft,
  simulationId,
  novelAiTokenRegistered,
  onCancel,
  onCreate,
  onTestGenerateCharacterImage,
  onPersistProfileImage
}: {
  mode: BuilderMode;
  initialDraft?: SimulationDraft;
  /** The simulation being edited. Undefined while creating: the draft has no id until 생성 mints one. */
  simulationId?: string;
  /** Whether this person has a NovelAI token. Only a yes/no reaches the page; the token stays with the App. */
  novelAiTokenRegistered: boolean;
  onCancel: () => void;
  onCreate: (draft: SimulationDraft) => void;
  /** Runs a profile test for a draft character. The page has no runtime access by design, so the App runs it. */
  onTestGenerateCharacterImage: (request: CharacterProfileTestRequest) => Promise<CharacterProfileTestResult>;
  /** Stores a picked candidate and returns the profile record to put on the draft character. */
  onPersistProfileImage: (request: CharacterProfileImagePersistRequest) => Promise<CharacterProfileImage>;
}) {
  const [draft, setDraft] = useState<SimulationDraft>(() => normalizeBuilderDraft(initialDraft ?? createInitialSimulationDraft()));
  // Where profile images picked in this builder session are stored. Editing stores them under the simulation itself.
  // A new draft has no id until 생성, so it gets a provisional one for as long as this page (and so this draft)
  // lives. Object keys carry their directory, so a key stored here still resolves once the simulation exists under
  // the id createStateFromDraft mints.
  const [provisionalSimulationId] = useState(() => createId("draft_sim"));
  const profileImageSimulationId = simulationId ?? provisionalSimulationId;
  const [activeTab, setActiveTab] = useState<BuilderTab>("overview");
  const [apiSection, setApiSection] = useState<BuilderApiSection>("novelai");
  const [selectedModuleId, setSelectedModuleId] = useState(() => initialDraft?.modules[0]?.id ?? "draft_main");
  const [draggedModuleId, setDraggedModuleId] = useState<string | undefined>();
  const [moduleDropTarget, setModuleDropTarget] = useState<{ moduleId: string; position: ModuleDropPosition } | undefined>();

  const selectedModule = draft.modules.find((module) => module.id === selectedModuleId) ?? draft.modules[0];
  const selectedCharacter =
    selectedModule?.kind === "character_prompt" && selectedModule.characterId
      ? draft.characters.find((character) => character.id === selectedModule.characterId)
      : undefined;
  const activeResolution = resolutionPresets.some((preset) => preset.width === draft.imageProfile.width && preset.height === draft.imageProfile.height)
    ? `${draft.imageProfile.width}x${draft.imageProfile.height}`
    : "custom";
  const activePromptModeOption = getPromptModeOption(draft.promptMode);
  const activeImageGenerationCadence =
    imageGenerationCadenceOptions.find((option) => option.value === draft.imageProfile.generationCadence) ??
    imageGenerationCadenceOptions[1];
  const draftImageScenePresets = draft.imageScenePresets ?? [];
  const draftBackendRouting = resolveImageBackendRouting(draft.imageProfile);
  const draftRoutingModeLabel =
    imageBackendRoutingModeOptions.find((option) => option.value === draftBackendRouting.mode)?.label ?? imageBackendRoutingModeOptions[0].label;
  const draftPhotorealPreset = PHOTOREAL_WORKFLOW_PRESETS[draft.photoreal.workflowPreset] ?? PHOTOREAL_WORKFLOW_PRESETS.qwen_image_2_1_turbo;
  const draftReferencePreset =
    PHOTOREAL_WORKFLOW_PRESETS[draft.photoreal.referenceWorkflowPreset] ?? PHOTOREAL_WORKFLOW_PRESETS.qwen_image_2_1_turbo;
  // Every preset that can draw this simulation's cuts, so the licence shown is the licence of what gets rendered:
  // with face lock on, each cut of a character with a 실사 profile renders on the face-lock preset — by default the
  // non-commercial Qwen-Image 2.1 Turbo, whatever the generation preset says. Same rule as resolvePhotorealPresetForCut.
  const draftRenderingPresets = [
    { preset: draftPhotorealPreset, role: "생성용" },
    ...(draft.photoreal.faceReferenceEnabled &&
    draftReferencePreset.supportsReferences &&
    draftReferencePreset.maxReferences > 0 &&
    draftReferencePreset.id !== draftPhotorealPreset.id
      ? [{ preset: draftReferencePreset, role: "얼굴 고정 컷" }]
      : [])
  ];
  const draftScreeningState = useMemo(
    () => createDraftScreeningState({ imageProfile: draft.imageProfile, novelAi: draft.novelAi, photoreal: draft.photoreal }),
    [draft.imageProfile, draft.novelAi, draft.photoreal]
  );
  const draftImageBackends = listActiveImageBackends(draft);
  // The tab's one-word status: which renderers this simulation will actually draw with, NovelAI by its model.
  const draftImageBackendMetric =
    draftImageBackends.length === 0
      ? "mock"
      : draftImageBackends.map((backend) => (backend === "novelai" ? draft.novelAi.modelPreset : imageBackendLabels.photoreal)).join(" + ");

  const updateDraft = useCallback((patch: Partial<SimulationDraft>) => {
    setDraft((current) => ({
      ...current,
      ...patch
    }));
  }, []);

  const updateContentRating = useCallback((contentRating: ContentRating) => {
    setDraft((current) => ({
      ...current,
      contentRating
    }));
  }, []);

  const updateSyncedText = useCallback((field: "mainPrompt" | "characterSummary" | "worldLore" | "visualPrompt", value: string) => {
    setDraft((current) => ({
      ...current,
      [field]: value,
      modules: current.modules.map((module) =>
        shouldSyncModuleWithDraftField(module, field, current.characters[0]?.id)
          ? { ...module, body: value, updatedAt: new Date().toISOString() }
          : module
      )
    }));
  }, []);

  const updateImageProfile = useCallback((patch: Partial<ImageGenerationProfile>) => {
    setDraft((current) => ({
      ...current,
      imageProfile: {
        ...current.imageProfile,
        ...patch
      }
    }));
  }, []);

  const updateNovelAi = useCallback((patch: Partial<NovelAiApiSettings>) => {
    setDraft((current) => ({
      ...current,
      novelAi: {
        ...current.novelAi,
        ...patch
      }
    }));
  }, []);

  const updatePhotoreal = useCallback((patch: Partial<PhotorealApiSettings>) => {
    setDraft((current) => ({
      ...current,
      photoreal: {
        ...current.photoreal,
        ...patch
      }
    }));
  }, []);

  const updateDraftCharacter = useCallback((characterId: string, patch: Partial<SimulationCharacterDraft>) => {
    setDraft((current) => {
      const characters = current.characters.map((character) =>
        character.id === characterId
          ? {
              ...character,
              ...patch
            }
          : character
      );
      const firstCharacter = characters[0];
      return {
        ...current,
        characters,
        characterName: firstCharacter?.name ?? current.characterName,
        characterRole: firstCharacter?.role ?? current.characterRole,
        characterSummary: firstCharacter?.summary ?? current.characterSummary,
        characterRelationship: firstCharacter?.relationship ?? current.characterRelationship,
        characterMood: firstCharacter?.currentMood ?? current.characterMood,
        visualPrompt: firstCharacter?.visualPrompt ?? current.visualPrompt,
        negativeVisualPrompt: firstCharacter?.negativeVisualPrompt ?? current.negativeVisualPrompt,
        defaultOutfitPrompt: firstCharacter?.defaultOutfitPrompt ?? current.defaultOutfitPrompt
      };
    });
  }, []);

  // Functional rather than a patch of `profileImages` read at render time: a NovelAI pick and a photoreal pick can be
  // persisting at once, and whichever finished second would otherwise put back the map without the first one.
  const setDraftCharacterProfileImage = useCallback(
    (characterId: string, backend: ImageBackendId, image: CharacterProfileImage | undefined) => {
      setDraft((current) => ({
        ...current,
        characters: current.characters.map((character) =>
          character.id === characterId
            ? { ...character, profileImages: withCharacterProfileImage(character.profileImages, backend, image) }
            : character
        )
      }));
    },
    []
  );

  const profileStudio = useCharacterProfileStudio({
    draft,
    simulationId: profileImageSimulationId,
    onTestGenerate: onTestGenerateCharacterImage,
    onPersist: onPersistProfileImage,
    onProfileImageChange: setDraftCharacterProfileImage
  });

  const updateNeuralMap = useCallback((patch: Partial<NeuralMapSettings>) => {
    setDraft((current) => ({
      ...current,
      neuralMap: {
        ...current.neuralMap,
        ...patch
      }
    }));
  }, []);

  const updateRelationshipMap = useCallback((patch: Partial<RelationshipMapSettings>) => {
    setDraft((current) => ({
      ...current,
      relationshipMap: {
        ...current.relationshipMap,
        ...patch,
        updatedAt: new Date().toISOString()
      }
    }));
  }, []);

  const addRelationshipStatusParameter = useCallback(() => {
    setDraft((current) => {
      const hasThoughtParameter = current.relationshipMap.parameters.some((parameter) => parameter.title.trim() === "생각");
      const parameter: RelationshipStatusParameter = {
        id: createId("rel_param"),
        title: hasThoughtParameter ? "새 파라미터" : "생각",
        rule: hasThoughtParameter ? "이 파라미터에 들어갈 내용 규칙을 작성합니다." : "해당 인물의 내면과 생각을 날것 그대로 작성합니다.",
        enabled: true,
        priority: 80
      };
      return {
        ...current,
        relationshipMap: {
          ...current.relationshipMap,
          enabled: true,
          parameters: [...current.relationshipMap.parameters, parameter],
          updatedAt: new Date().toISOString()
        }
      };
    });
  }, []);

  const updateRelationshipStatusParameter = useCallback((parameterId: string, patch: Partial<RelationshipStatusParameter>) => {
    setDraft((current) => ({
      ...current,
      relationshipMap: {
        ...current.relationshipMap,
        parameters: current.relationshipMap.parameters.map((parameter) =>
          parameter.id === parameterId
            ? {
                ...parameter,
                ...patch
              }
            : parameter
        ),
        updatedAt: new Date().toISOString()
      }
    }));
  }, []);

  const deleteRelationshipStatusParameter = useCallback((parameterId: string) => {
    setDraft((current) => ({
      ...current,
      relationshipMap: {
        ...current.relationshipMap,
        parameters: current.relationshipMap.parameters.filter((parameter) => parameter.id !== parameterId),
        updatedAt: new Date().toISOString()
      }
    }));
  }, []);

  const addImageScenePreset = useCallback(() => {
    setDraft((current) => {
      const now = new Date().toISOString();
      const currentPresets = current.imageScenePresets ?? [];
      const presetIndex = currentPresets.length + 1;
      const preset: ImageSceneTagPreset = {
        id: createId("draft_scene_preset"),
        simulationId: "draft_simulation",
        keyword: `scene keyword ${presetIndex}`,
        tags: ["indoors", "soft light", "depth of field"],
        note: "",
        enabled: true,
        priority: 70,
        updatedAt: now,
        children: []
      };
      return {
        ...current,
        imageScenePresets: [...currentPresets, preset]
      };
    });
  }, []);

  const addImageScenePresetChild = useCallback((parentId: string) => {
    setDraft((current) => {
      const now = new Date().toISOString();
      const child = createDraftImageScenePresetNode(now, countEnabledImageScenePresetNodes(current.imageScenePresets ?? []) + 1);
      return {
        ...current,
        imageScenePresets: appendImageScenePresetChild(current.imageScenePresets ?? [], parentId, child, now)
      };
    });
  }, []);

  const updateImageScenePreset = useCallback((presetId: string, patch: Partial<ImageSceneTagPresetNode>) => {
    setDraft((current) => ({
      ...current,
      imageScenePresets: updateImageScenePresetNodes(current.imageScenePresets ?? [], presetId, patch, new Date().toISOString())
    }));
  }, []);

  const moveImageScenePreset = useCallback((sourceId: string, targetId: string, position: ImageScenePresetDropPosition) => {
    setDraft((current) => {
      const presets = current.imageScenePresets ?? [];
      return {
        ...current,
        imageScenePresets: moveImageScenePresetNodes(
          presets,
          sourceId,
          targetId,
          position,
          new Date().toISOString(),
          presets[0]?.simulationId ?? "draft_simulation"
        )
      };
    });
  }, []);

  const deleteImageScenePreset = useCallback((presetId: string) => {
    setDraft((current) => ({
      ...current,
      imageScenePresets: deleteImageScenePresetNode(current.imageScenePresets ?? [], presetId)
    }));
  }, []);

  const updateDraftModule = useCallback((moduleId: string, patch: Partial<PromptModule>) => {
    setDraft((current) => {
      const targetModule = current.modules.find((module) => module.id === moduleId);
      if (!targetModule) {
        return current;
      }

      const now = new Date().toISOString();
      const nextKind = patch.kind ?? targetModule.kind;
      const nextCharacterId =
        nextKind === "character_prompt"
          ? patch.characterId ?? targetModule.characterId ?? createId("draft_char")
          : undefined;
      const createdCharacter =
        nextKind === "character_prompt" && nextCharacterId && !current.characters.some((character) => character.id === nextCharacterId)
          ? createDraftCharacterForModule(
              {
                ...targetModule,
                ...patch,
                kind: nextKind,
                characterId: nextCharacterId
              },
              nextCharacterId,
              `New Character ${current.characters.length + 1}`,
              targetModule.kind === "character_prompt"
            )
          : undefined;
      let characters = createdCharacter ? [...current.characters, createdCharacter] : current.characters;
      const modules = current.modules.map((module) =>
        module.id === moduleId
          ? {
              ...module,
              ...patch,
              kind: nextKind,
              title:
                createdCharacter && (!patch.title || targetModule.title === "새 서브 프롬프트")
                  ? `캐릭터: ${createdCharacter.name}`
                  : patch.title ?? module.title,
              body: createdCharacter && (!patch.body || targetModule.body === defaultSubPromptBody) ? createdCharacter.summary : patch.body ?? module.body,
              characterId: nextCharacterId,
              updatedAt: now
            }
          : module
      );
      const patchedModule = modules.find((module) => module.id === moduleId);
      characters =
        patchedModule?.kind === "character_prompt" && patchedModule.characterId
          ? characters.map((character) =>
              character.id === patchedModule.characterId
                ? {
                    ...character,
                    summary: patch.body ?? character.summary,
                    name: createdCharacter && character.id === createdCharacter.id ? character.name : patch.title?.replace(/^캐릭터:\s*/u, "") ?? character.name
                  }
                : character
            )
          : current.characters;
      return {
        ...current,
        mainPrompt: moduleId === "draft_main" && patch.body !== undefined ? patch.body : current.mainPrompt,
        characterSummary: moduleId === "draft_character" && patch.body !== undefined ? patch.body : current.characterSummary,
        worldLore: moduleId === "draft_world" && patch.body !== undefined ? patch.body : current.worldLore,
        visualPrompt: moduleId === "draft_visual" && patch.body !== undefined ? patch.body : current.visualPrompt,
        characterName: characters[0]?.name ?? current.characterName,
        characterRole: characters[0]?.role ?? current.characterRole,
        characterRelationship: characters[0]?.relationship ?? current.characterRelationship,
        characterMood: characters[0]?.currentMood ?? current.characterMood,
        negativeVisualPrompt: characters[0]?.negativeVisualPrompt ?? current.negativeVisualPrompt,
        characters,
        modules: patchedModule ? modules : current.modules
      };
    });
  }, []);

  const addDraftModule = useCallback(
    (kind: PromptModuleKind = "sub_prompt") => {
      const id = createId("draft_module");
      const now = new Date().toISOString();
      const characterId = kind === "character_prompt" ? createId("draft_char") : undefined;
      const character =
        kind === "character_prompt" && characterId
          ? createDraftCharacterForModule(
              {
                id,
                simulationId: "draft_simulation",
                parentId: selectedModuleId,
                kind,
                title: `캐릭터: New Character ${draft.characters.length + 1}`,
                body: defaultCharacterSummaryPrompt,
                enabled: true,
                priority: 58,
                activationTags: ["character", "scene"],
                characterId,
                tokenPolicy: "rag",
                version: 1,
                updatedAt: now
              },
              characterId,
              `New Character ${draft.characters.length + 1}`,
              false
            )
          : undefined;
      setDraft((current) => ({
        ...current,
        characters: character ? [...current.characters, character] : current.characters,
        modules: [
          ...current.modules,
          {
            id,
            simulationId: "draft_simulation",
            parentId: selectedModuleId,
            kind,
            title: character ? `캐릭터: ${character.name}` : kind === "image_prompt_profile" ? "이미지 스타일 프로필" : "새 서브 프롬프트",
            body:
              kind === "character_prompt"
                ? character?.summary ?? defaultCharacterSummaryPrompt
                : kind === "image_prompt_profile"
                  ? "cinematic anime illustration, consistent lineart, atmospheric lighting, detailed background"
                  : defaultSubPromptBody,
            enabled: true,
            priority: kind === "image_prompt_profile" ? 82 : 58,
            activationTags: kind === "image_prompt_profile" ? ["image", "style", "nai"] : [kind.replace("_prompt", ""), "scene"],
            characterId,
            tokenPolicy: kind === "image_prompt_profile" ? "always" : "rag",
            version: 1,
            updatedAt: now
          }
        ]
      }));
      setSelectedModuleId(id);
      setActiveTab("prompts");
    },
    [draft.characters.length, selectedModuleId]
  );

  const duplicateDraftModule = useCallback((moduleId: string) => {
    const copiedModuleId = createId("draft_module");
    const copiedCharacterId = createId("draft_char");
    setDraft((current) => {
      const sourceIndex = current.modules.findIndex((module) => module.id === moduleId);
      const sourceModule = current.modules[sourceIndex];
      if (!sourceModule) {
        return current;
      }

      const now = new Date().toISOString();
      const existingTitles = current.modules.map((module) => module.title);
      const sourceCharacter = sourceModule.characterId
        ? current.characters.find((character) => character.id === sourceModule.characterId)
        : undefined;
      const copiedCharacter =
        sourceModule.kind === "character_prompt"
          ? sourceCharacter
            ? {
                ...sourceCharacter,
                id: copiedCharacterId,
                name: createCopiedCharacterName(sourceCharacter.name, current.characters.map((character) => character.name)),
                summary: sourceModule.body || sourceCharacter.summary,
                defaultOutfitPrompt: sourceCharacter.defaultOutfitPrompt,
                outfitPrompts: { ...sourceCharacter.outfitPrompts },
                expressionPrompts: { ...sourceCharacter.expressionPrompts }
              }
            : createDraftCharacterForModule(
                sourceModule,
                copiedCharacterId,
                createCopiedCharacterName(inferCharacterNameFromModule(sourceModule, "New Character"), current.characters.map((character) => character.name))
              )
          : undefined;
      const copiedModule: PromptModule = {
        ...sourceModule,
        id: copiedModuleId,
        title: copiedCharacter ? `캐릭터: ${copiedCharacter.name}` : createCopiedTitle(sourceModule.title, existingTitles),
        characterId: copiedCharacter?.id,
        activationTags: [...sourceModule.activationTags],
        version: 1,
        updatedAt: now
      };
      const modules = [...current.modules];
      modules.splice(sourceIndex + 1, 0, copiedModule);

      return {
        ...current,
        characters: copiedCharacter ? [...current.characters, copiedCharacter] : current.characters,
        modules
      };
    });
    setSelectedModuleId(copiedModuleId);
    setActiveTab("prompts");
  }, []);

  const applyPromptMode = useCallback(
    (promptMode: SimulationPromptMode) => {
      if (promptMode === "custom") {
        setDraft((current) => ({
          ...current,
          promptMode
        }));
        return;
      }

      setDraft((current) => applyPromptModePresetToDraft(current, promptMode));
      setSelectedModuleId(draft.modules.find((module) => module.kind === "main_prompt")?.id ?? "draft_main");
    },
    [draft.modules]
  );

  const deleteDraftModule = useCallback((moduleId: string) => {
    setDraft((current) => {
      const targetModule = current.modules.find((module) => module.id === moduleId);
      const modules = current.modules.filter((module) => module.id !== moduleId && module.parentId !== moduleId);
      const characters =
        targetModule?.kind === "character_prompt" && targetModule.characterId
          ? current.characters.filter((character) => character.id !== targetModule.characterId)
          : current.characters;
      return {
        ...current,
        characters: characters.length ? characters : current.characters,
        modules
      };
    });
    setSelectedModuleId("draft_main");
  }, []);

  const moveDraftModuleToIndex = useCallback((moduleId: string, targetIndex: number) => {
    setDraft((current) => {
      const currentIndex = current.modules.findIndex((module) => module.id === moduleId);
      if (currentIndex < 0) {
        return current;
      }

      const modules = [...current.modules];
      const [module] = modules.splice(currentIndex, 1);
      const normalizedTargetIndex = Math.min(
        modules.length,
        Math.max(0, targetIndex > currentIndex ? targetIndex - 1 : targetIndex)
      );

      if (normalizedTargetIndex === currentIndex) {
        return current;
      }

      modules.splice(normalizedTargetIndex, 0, module);

      return {
        ...current,
        modules
      };
    });
  }, []);

  const handleModuleDragStart = useCallback((event: ReactDragEvent<HTMLDivElement>, moduleId: string) => {
    setDraggedModuleId(moduleId);
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", moduleId);
  }, []);

  const handleModuleDragOver = useCallback(
    (event: ReactDragEvent<HTMLDivElement>, moduleId: string) => {
      if (!draggedModuleId || draggedModuleId === moduleId) {
        return;
      }

      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      const bounds = event.currentTarget.getBoundingClientRect();
      const position: ModuleDropPosition = event.clientY < bounds.top + bounds.height / 2 ? "before" : "after";
      setModuleDropTarget((current) =>
        current?.moduleId === moduleId && current.position === position ? current : { moduleId, position }
      );
    },
    [draggedModuleId]
  );

  const handleModuleDrop = useCallback(
    (event: ReactDragEvent<HTMLDivElement>, moduleId: string) => {
      event.preventDefault();
      const sourceModuleId = draggedModuleId ?? event.dataTransfer.getData("text/plain");
      const targetIndex = draft.modules.findIndex((module) => module.id === moduleId);

      if (sourceModuleId && sourceModuleId !== moduleId && targetIndex >= 0) {
        const bounds = event.currentTarget.getBoundingClientRect();
        const position: ModuleDropPosition = event.clientY < bounds.top + bounds.height / 2 ? "before" : "after";
        moveDraftModuleToIndex(sourceModuleId, targetIndex + (position === "after" ? 1 : 0));
        setSelectedModuleId(sourceModuleId);
      }

      setDraggedModuleId(undefined);
      setModuleDropTarget(undefined);
    },
    [draft.modules, draggedModuleId, moveDraftModuleToIndex]
  );

  const handleModuleDragEnd = useCallback(() => {
    setDraggedModuleId(undefined);
    setModuleDropTarget(undefined);
  }, []);

  return (
    <section className="create-page">
      <div className="create-header">
        <div>
          <span className="section-kicker">Simulation Builder</span>
          <h2>{mode === "edit" ? "시뮬레이션 수정" : "시뮬레이션 제작"}</h2>
          <p>{mode === "edit" ? "기존 진행 기록은 유지하고 설정, Prompt Tree, API 구성을 다시 조정합니다." : "메인 구조는 빠르게 훑고, 상세 설정은 제작 단계 안에서 바로 확정합니다."}</p>
        </div>
        <div className="create-actions">
          <button className="icon-text-button" type="button" onClick={onCancel}>
            취소
          </button>
          <button className="send-button" type="button" onClick={() => onCreate(draft)}>
            <Save size={17} />
            {mode === "edit" ? "수정 저장" : "생성"}
          </button>
        </div>
      </div>

      <div className="builder-tabs" role="tablist" onKeyDown={handleTablistKeyDown} aria-label="시뮬레이션 제작 단계">
        <BuilderTabButton active={activeTab === "overview"} icon={<Boxes size={18} />} label="기본" metric="개요" onClick={() => setActiveTab("overview")} panelKey="overview" />
        <BuilderTabButton active={activeTab === "prompts"} icon={<Layers size={18} />} label="프롬프트 트리" metric={`${draft.modules.length}개`} onClick={() => setActiveTab("prompts")} panelKey="prompts" />
        <BuilderTabButton active={activeTab === "characters"} icon={<ImageIcon size={18} />} label="캐릭터/이미지" metric={draft.imageProfile.triggerMode} onClick={() => setActiveTab("characters")} panelKey="characters" />
        <BuilderTabButton active={activeTab === "status"} icon={<Network size={18} />} label="관계/상태" metric={draft.relationshipMap.enabled ? "저장" : "꺼짐"} onClick={() => setActiveTab("status")} panelKey="status" />
        <BuilderTabButton active={activeTab === "api"} icon={<Settings2 size={18} />} label="이미지/연동" metric={draftImageBackendMetric} onClick={() => setActiveTab("api")} panelKey="api" />
        <BuilderTabButton active={activeTab === "review"} icon={<Check size={18} />} label="검토" metric="생성" onClick={() => setActiveTab("review")} panelKey="review" />
      </div>

      <div
        className="builder-body"
        role="tabpanel"
        id={`tabpanel-builder-${activeTab}`}
        aria-labelledby={`tab-builder-${activeTab}`}
      >
        {activeTab === "overview" ? (
          <div className="builder-grid">
            <section className="builder-panel span-2">
              <SectionTitle icon={<SlidersHorizontal size={17} />} title="진행 모드" />
              <div className="prompt-mode-card-grid" role="radiogroup" aria-label="시뮬레이션 진행 모드">
                {promptModeOptions.map((option) => (
                  <PromptModeCard
                    active={draft.promptMode === option.value}
                    key={option.value}
                    option={option}
                    onClick={() => applyPromptMode(option.value)}
                  />
                ))}
              </div>
              <p className="settings-note mode-template-note">
                기본, 1:1, 시뮬레이션 모드는 핵심 프롬프트와 모듈을 바로 채웁니다. 커스텀은 현재 입력을 유지하고 제작자가 직접 구성합니다.
              </p>

              <SectionTitle icon={<WandSparkles size={17} />} title="기본 정보" />
              <div className="two-fields">
                <label>
                  <GuidedFieldLabel guide={titleGuide} title="제목" />
                  <input value={draft.title} onChange={(event) => updateDraft({ title: event.target.value })} />
                </label>
                <label>
                  <GuidedFieldLabel guide={descriptionGuide} title="설명" />
                  <input value={draft.description} onChange={(event) => updateDraft({ description: event.target.value })} />
                </label>
              </div>
              <PromptMeaningGuide />
              <div className="prompt-authoring-grid">
                <PromptAuthoringField
                  badge="진행 규칙 · 항상 포함"
                  guide={mainPromptGuide}
                  icon={<SlidersHorizontal size={15} />}
                  placeholder={moduleKindGuides.main_prompt.placeholder}
                  title="메인 프롬프트"
                  tone="main"
                  value={draft.mainPrompt}
                  onChange={(value) => updateSyncedText("mainPrompt", value)}
                />
                <PromptAuthoringField
                  badge="세계 자료 · 필요 시 참조"
                  guide={worldLoreGuide}
                  icon={<Database size={15} />}
                  placeholder={moduleKindGuides.world_lore.placeholder}
                  title="세계관/로어"
                  tone="world"
                  value={draft.worldLore}
                  onChange={(value) => updateSyncedText("worldLore", value)}
                />
              </div>
              <label className="start-situation-field">
                <GuidedFieldLabel
                  badge="첫 장면"
                  guide={startSituationGuide}
                  icon={<Play size={15} />}
                  title="시작상황 지정 프롬프트"
                  charCount={<PromptCharCount value={draft.startSituationPrompt} />}
                />
                <textarea
                  value={draft.startSituationPrompt}
                  onChange={(event) => updateDraft({ startSituationPrompt: event.target.value })}
                  placeholder="예: 첫 장면은 새벽 역 플랫폼에서 시작한다. 주인공은 젖은 티켓과 이름 없는 가방을 들고 있고, 곧 도착할 열차가 첫 선택을 강요한다."
                />
              </label>
            </section>

            <aside className="builder-panel">
              <SectionTitle icon={<ShieldCheck size={17} />} title="런타임 기본값" />
              <div className="mode-summary-card">
                <span>{activePromptModeOption.kicker}</span>
                <strong>{activePromptModeOption.label} 모드</strong>
                <p>{activePromptModeOption.description}</p>
              </div>
              <label>
                콘텐츠 등급
                <select value={draft.contentRating} onChange={(event) => updateContentRating(event.target.value as ContentRating)}>
                  {contentRatingOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="checkline">
                <input checked={draft.realtimeImageEnabled} type="checkbox" onChange={(event) => updateDraft({ realtimeImageEnabled: event.target.checked })} />
                실시간 이미지 활성화
              </label>
              <label>
                이미지 트리거
                <select value={draft.imageProfile.triggerMode} onChange={(event) => updateImageProfile({ triggerMode: event.target.value as ImageTriggerMode })}>
                  {triggerModes.map((mode) => (
                    <option key={mode} value={mode}>
                      {mode}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                NeuralMap token budget
                <input min="800" max="12000" step="200" type="number" value={draft.neuralMap.tokenBudget} onChange={(event) => updateNeuralMap({ tokenBudget: Number(event.target.value) })} />
              </label>
              <div className="builder-stat-grid">
                <Metric label="모듈" value={draft.modules.length.toString()} />
                <Metric label="등급" value={contentRatingOptions.find((option) => option.value === draft.contentRating)?.label ?? "일반"} />
                <Metric label="이미지" value={activeImageGenerationCadence.label} />
              </div>
            </aside>
          </div>
        ) : null}

        {activeTab === "prompts" ? (
          <div className="prompt-builder">
            <aside className="builder-panel prompt-tree-panel">
              <div className="panel-header compact">
                <div>
                  <span className="section-kicker">프롬프트 트리</span>
                  <h2>모듈 구조</h2>
                </div>
                <button className="icon-button" type="button" onClick={() => addDraftModule()} aria-label="프롬프트 모듈 추가">
                  <Plus size={18} />
                </button>
              </div>
              <div className="quick-add-row">
                <button type="button" onClick={() => addDraftModule("character_prompt")}>캐릭터</button>
                <button type="button" onClick={() => addDraftModule("world_lore")}>세계관</button>
                <button type="button" onClick={() => addDraftModule("scene_rule")}>장면</button>
                <button type="button" onClick={() => addDraftModule("image_prompt_profile")}>이미지 스타일</button>
              </div>
              <div className="module-list builder-module-list" aria-label="제작 프롬프트 모듈 목록">
                {draft.modules.map((module) => (
                  <div
                    className={[
                      "module-row-wrap",
                      module.id === selectedModule?.id ? "active" : "",
                      draggedModuleId === module.id ? "dragging" : "",
                      moduleDropTarget?.moduleId === module.id ? `drop-${moduleDropTarget.position}` : ""
                    ].filter(Boolean).join(" ")}
                    draggable
                    key={module.id}
                    onDragEnd={handleModuleDragEnd}
                    onDragOver={(event) => handleModuleDragOver(event, module.id)}
                    onDragStart={(event) => handleModuleDragStart(event, module.id)}
                    onDrop={(event) => handleModuleDrop(event, module.id)}
                  >
                    <button className="module-row" type="button" onClick={() => setSelectedModuleId(module.id)}>
                      <span className="module-drag-grip" aria-hidden="true">
                        <GripVertical size={16} />
                      </span>
                      <span className="module-kind">{moduleKindGuides[module.kind].title}</span>
                      <strong>{module.title}</strong>
                      <span className="module-row-meta" aria-label={`참조 방식 ${tokenPolicyLabels[module.tokenPolicy]}, 우선순위 ${module.priority}`}>
                        <span className={`module-policy-badge policy-${module.tokenPolicy}`}>{tokenPolicyShortLabels[module.tokenPolicy]}</span>
                        <span>우선 {module.priority}</span>
                      </span>
                      <ChevronRight size={16} />
                    </button>
                    <button
                      className="module-copy-button"
                      draggable={false}
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        duplicateDraftModule(module.id);
                      }}
                      onPointerDown={(event) => event.stopPropagation()}
                      aria-label={`${module.title} 모듈 복사`}
                      title="모듈 복사"
                    >
                      <Copy size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </aside>
            <section className="builder-panel prompt-editor-panel">
              {selectedModule ? <ModuleEditor module={selectedModule} onChange={updateDraftModule} onDelete={deleteDraftModule} /> : null}
              {selectedModule?.kind === "character_prompt" && selectedCharacter ? (
                <>
                  <CharacterPromptPanel
                    character={selectedCharacter}
                    onChange={(patch) => updateDraftCharacter(selectedCharacter.id, patch)}
                    onModuleChange={(patch) => updateDraftModule(selectedModule.id, patch)}
                  />
                  <CharacterVisualMappingPanel
                    character={selectedCharacter}
                    onCharacterChange={(patch) => updateDraftCharacter(selectedCharacter.id, patch)}
                  />
                  <CharacterProfileImagePanel
                    character={selectedCharacter}
                    draft={draft}
                    session={profileStudio.sessions[selectedCharacter.id]}
                    novelAiTokenRegistered={novelAiTokenRegistered}
                    onSessionChange={(change) => profileStudio.updateSession(selectedCharacter.id, change)}
                    onCharacterChange={(patch) => updateDraftCharacter(selectedCharacter.id, patch)}
                    onClearProfileImage={(backend) => setDraftCharacterProfileImage(selectedCharacter.id, backend, undefined)}
                    onNovelAiChange={updateNovelAi}
                    onPhotorealChange={updatePhotoreal}
                    onGenerate={(input) => void profileStudio.start(selectedCharacter.id, input)}
                    onCancel={() => profileStudio.cancel(selectedCharacter.id)}
                    onPick={(candidate) => void profileStudio.pick(selectedCharacter.id, candidate)}
                    onDiscardCandidates={() => profileStudio.discardCandidates(selectedCharacter.id)}
                  />
                </>
              ) : null}
              {selectedModule?.kind === "image_prompt_profile" ? (
                <ImageStyleProfilePanel
                  module={selectedModule}
                  onModuleChange={(patch) => updateDraftModule(selectedModule.id, patch)}
                />
              ) : null}
            </section>
          </div>
        ) : null}

        {activeTab === "characters" ? (
          <div className="builder-grid">
            <section className="builder-panel span-2">
              <SectionTitle icon={<Bot size={17} />} title="캐릭터 목록" />
              <div className="character-card-grid">
                {draft.characters.map((character) => {
                  const characterModule = draft.modules.find((module) => module.kind === "character_prompt" && module.characterId === character.id);
                  const profileImage = selectDisplayProfileImage(character.profileImages);
                  const profileSummary = describeProfileImageBackends(character.profileImages);
                  return (
                    <button
                      className={["character-card", selectedCharacter?.id === character.id ? "active" : "", profileImage ? "has-portrait" : ""]
                        .filter(Boolean)
                        .join(" ")}
                      key={character.id}
                      type="button"
                      onClick={() => {
                        setSelectedModuleId(characterModule?.id ?? "draft_main");
                        setActiveTab("prompts");
                      }}
                    >
                      {profileImage ? <ProfileImageThumb className="character-card-portrait" image={profileImage} /> : null}
                      <strong>{character.name}</strong>
                      <span>{character.role}</span>
                      <p>{character.summary}</p>
                      <small>
                        {character.visualPrompt ? "이미지 프롬프트 준비됨" : "이미지 프롬프트 없음"}
                        {profileSummary ? ` · 프로필 ${profileSummary}` : ""}
                      </small>
                    </button>
                  );
                })}
              </div>
              <div className="editor-actions">
                <span>캐릭터 상세와 이미지 매핑은 Prompt Tree에서 캐릭터 모듈을 선택하면 열립니다.</span>
                <button className="icon-text-button" type="button" onClick={() => addDraftModule("character_prompt")}>
                  <Plus size={16} />
                  캐릭터 추가
                </button>
              </div>
            </section>

            <aside className="builder-panel">
              <SectionTitle icon={<ImageIcon size={17} />} title="캐릭터 이미지 요약" />
              <div className="compact-list">
                {draft.characters.map((character) => {
                  const characterModule = draft.modules.find((module) => module.kind === "character_prompt" && module.characterId === character.id);
                  const profileImage = selectDisplayProfileImage(character.profileImages);
                  return (
                    <button className={`compact-row as-button${profileImage ? " has-portrait" : ""}`} key={character.id} type="button" onClick={() => {
                      setSelectedModuleId(characterModule?.id ?? "draft_main");
                      setActiveTab("prompts");
                    }}>
                      {profileImage ? <ProfileImageThumb className="compact-row-portrait" image={profileImage} /> : null}
                      <span>{character.role}</span>
                      <strong>{character.name}</strong>
                      <small>{character.visualPrompt || "이미지 프롬프트 미설정"}</small>
                    </button>
                  );
                })}
              </div>
            </aside>

            <section className="builder-panel span-2 image-prompt-authoring-panel">
              <SectionTitle icon={<Sparkles size={17} />} title="이미지 생성 프롬프트" />
              <div className="two-fields prompt-textarea-grid">
                <label>
                  작가 프롬프트
                  <textarea
                    rows={3}
                    value={draft.imageProfile.artistPrompt}
                    onChange={(event) => updateImageProfile({ artistPrompt: event.target.value })}
                    placeholder="artist tag, brush style, line work, color mood..."
                  />
                </label>
                <label>
                  기본 퀄리티 프롬프트
                  <textarea
                    rows={3}
                    value={draft.imageProfile.qualityPrompt}
                    onChange={(event) => updateImageProfile({ qualityPrompt: event.target.value })}
                    placeholder="masterpiece, best quality, detailed background..."
                  />
                </label>
              </div>
              <label className="prompt-wide-field">
                스타일 프롬프트
                <textarea
                  rows={4}
                  value={draft.imageProfile.stylePrompt}
                  onChange={(event) => updateImageProfile({ stylePrompt: event.target.value })}
                  placeholder="cinematic anime illustration, lighting, camera, composition..."
                />
              </label>
              <label className="prompt-long-field">
                공통 negative prompt
                <textarea rows={5} value={draft.imageProfile.negativePrompt} onChange={(event) => updateImageProfile({ negativePrompt: event.target.value })} />
              </label>
              <label className="prompt-long-field">
                사용자 규정 / LLM 지시문
                <textarea rows={5} value={draft.imageProfile.userRules} onChange={(event) => updateImageProfile({ userRules: event.target.value })} />
              </label>
            </section>

            <SceneTagPresetTree
              presets={draftImageScenePresets}
              onAddChild={addImageScenePresetChild}
              onAddRoot={addImageScenePreset}
              onChange={updateImageScenePreset}
              onDelete={deleteImageScenePreset}
              onMove={moveImageScenePreset}
            />
          </div>
        ) : null}

        {activeTab === "status" ? (
          <div className="builder-grid">
            <section className="builder-panel span-2 status-authoring-panel">
              <SectionTitle icon={<Network size={17} />} title="관계도 / 상태창 규칙" />
              <label className="checkline">
                <input checked={draft.relationshipMap.enabled} type="checkbox" onChange={(event) => updateRelationshipMap({ enabled: event.target.checked })} />
                진행 중 관계도와 캐릭터 상태창을 memory_events로 갱신
              </label>
              <label className="prompt-long-field">
                상태창 정제 프롬프트
                <textarea
                  rows={12}
                  value={draft.relationshipMap.statusPrompt}
                  onChange={(event) => updateRelationshipMap({ statusPrompt: event.target.value })}
                  placeholder="관계도 탭에 저장할 현재 상태, 관계, 목표, 지식 규칙을 작성하세요."
                />
              </label>
              <div className="relationship-parameter-editor">
                <div className="runtime-card-subhead">
                  <strong>상태창 파라미터</strong>
                  <button className="icon-text-button" type="button" onClick={addRelationshipStatusParameter}>
                    <Plus size={16} />
                    파라미터 추가
                  </button>
                </div>
                <div className="relationship-parameter-list">
                  {draft.relationshipMap.parameters.map((parameter) => (
                    <div className="relationship-parameter-row" key={parameter.id}>
                      <div className="relationship-parameter-row-head">
                        <label className="checkline">
                          <input
                            checked={parameter.enabled}
                            type="checkbox"
                            onChange={(event) => updateRelationshipStatusParameter(parameter.id, { enabled: event.target.checked })}
                          />
                          사용
                        </label>
                        <button className="icon-button" type="button" onClick={() => deleteRelationshipStatusParameter(parameter.id)} aria-label={`${parameter.title} 파라미터 삭제`}>
                          <Trash2 size={15} />
                        </button>
                      </div>
                      <div className="two-fields">
                        <label>
                          제목
                          <input
                            value={parameter.title}
                            onChange={(event) => updateRelationshipStatusParameter(parameter.id, { title: event.target.value })}
                            placeholder="예: 생각"
                          />
                        </label>
                        <label>
                          우선순위
                          <input
                            min="0"
                            max="120"
                            type="number"
                            value={parameter.priority}
                            onChange={(event) => updateRelationshipStatusParameter(parameter.id, { priority: Number(event.target.value) })}
                          />
                        </label>
                      </div>
                      <label>
                        내용 규칙
                        <textarea
                          rows={3}
                          value={parameter.rule}
                          onChange={(event) => updateRelationshipStatusParameter(parameter.id, { rule: event.target.value })}
                          placeholder="예: 해당 인물의 내면과 생각을 날것 그대로 작성합니다."
                        />
                      </label>
                    </div>
                  ))}
                </div>
              </div>
              <div className="editor-actions">
                <span>이 규칙은 LLM 컨텍스트에 들어가고, 긴 상태창 출력 대신 구조화된 메모리 델타를 남기도록 유도합니다.</span>
                <button
                  className="icon-text-button"
                  type="button"
                  onClick={() =>
                    updateRelationshipMap({
                      enabled: true,
                      statusPrompt: createRelationshipMapPresetPrompt(draft.promptMode),
                      parameters: seedState.relationshipMap.parameters.map((parameter) => ({ ...parameter }))
                    })
                  }
                >
                  <RefreshCcw size={16} />
                  모드 기본값
                </button>
              </div>
            </section>

            <aside className="builder-panel status-preview-panel">
              <SectionTitle icon={<Brain size={17} />} title="저장될 상태창" />
              <div className="relationship-preview-list">
                {draft.characters.map((character) => (
                  <div className="relationship-preview-row" key={character.id}>
                    <strong>{character.name}</strong>
                    <span>{character.role}</span>
                    <p>{character.relationship || "관계 미설정"}</p>
                    <small>{draft.relationshipMap.parameters.filter((parameter) => parameter.enabled).map((parameter) => parameter.title).join(" · ") || character.currentMood || "현재 상태 미설정"}</small>
                  </div>
                ))}
              </div>
              <div className="readonly-field">
                <span>토큰 절약 방식</span>
                <strong>{draft.relationshipMap.enabled ? "관계/상태 변화만 저장" : "상태창 정제 비활성"}</strong>
                <small>오른쪽 관계도 탭은 캐릭터, 메모리, NeuralMap 미러 데이터를 사용자용으로 정리해 표시합니다.</small>
              </div>
            </aside>
          </div>
        ) : null}

        {activeTab === "api" ? (
          <RailSubtabs idPrefix="builder-api" label="이미지/연동 항목" items={BUILDER_API_SECTIONS} active={apiSection} onSelect={setApiSection} variant="builder">
            <div className="builder-grid">
              {apiSection === "novelai" ? (
                <section className="builder-panel span-2">
                  <SectionTitle icon={<Sparkles size={17} />} title="NovelAI 이미지 설정" />
                  <div className="readonly-field">
                    <span>실행자 NovelAI</span>
                    <strong>NovelAI 토큰과 구독 상태는 시뮬레이션에 저장하지 않습니다.</strong>
                    <small>모델, 해상도, 샘플링 값만 제작 설정으로 저장하고 실제 토큰은 사용하는 사람이 따로 등록합니다.</small>
                  </div>

                  <div className="settings-section">
                    <SectionTitle icon={<KeyRound size={16} />} title="실행자 연결" />
                    <div className="two-fields">
                      <label className="checkline">
                        <input checked={draft.novelAi.enabled} type="checkbox" onChange={(event) => updateNovelAi({ enabled: event.target.checked })} />
                        NovelAI 이미지 API 사용
                      </label>
                      <label className="checkline">
                        <input checked={draft.novelAi.roundRobinEnabled} type="checkbox" onChange={(event) => updateNovelAi({ roundRobinEnabled: event.target.checked })} />
                        계정 라운드로빈
                      </label>
                    </div>
                    <div className="two-fields">
                      <label>
                        요청 방식
                        <select value={draft.novelAi.requestMode} onChange={(event) => updateNovelAi({ requestMode: event.target.value as NovelAiApiSettings["requestMode"] })}>
                          <option value="mock">mock</option>
                          <option value="direct">direct</option>
                          <option value="proxy">proxy</option>
                        </select>
                      </label>
                      <label>
                        계정 라벨
                        <input value={draft.novelAi.accountLabel} onChange={(event) => updateNovelAi({ accountLabel: event.target.value })} />
                      </label>
                    </div>
                    <div className="readonly-field">
                      <span>공유 경계</span>
                      <strong>{draft.novelAi.enabled ? "실행 시 사용자의 NovelAI 토큰을 사용합니다." : "저장 이미지 또는 mock 모드로 동작합니다."}</strong>
                      <small>제작 화면에서는 NovelAI 토큰 등록 여부를 검사하지 않습니다.</small>
                    </div>
                    <div className="two-fields">
                      <label>
                        엔드포인트
                        <input value={draft.novelAi.endpoint} onChange={(event) => updateNovelAi({ endpoint: event.target.value })} />
                      </label>
                      <label>
                        프록시 URL
                        <input value={draft.novelAi.proxyUrl} onChange={(event) => updateNovelAi({ proxyUrl: event.target.value })} />
                      </label>
                    </div>
                  </div>

                  <div className="settings-section">
                    <SectionTitle icon={<ImageIcon size={16} />} title="모델과 해상도" />
                    <div className="two-fields">
                      <label>
                        모델
                        <select value={draft.novelAi.modelPreset} onChange={(event) => updateNovelAi({ modelPreset: event.target.value as NovelAiModelPreset })}>
                          {novelAiModelPresets.map((preset) => (
                            <option key={preset} value={preset}>
                              {preset}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        해상도
                        <select
                          value={activeResolution}
                          onChange={(event) => {
                            const preset = resolutionPresets.find((item) => `${item.width}x${item.height}` === event.target.value);
                            if (preset) {
                              updateImageProfile({ width: preset.width, height: preset.height });
                            }
                          }}
                        >
                          {resolutionPresets.map((preset) => (
                            <option key={preset.label} value={`${preset.width}x${preset.height}`}>
                              {preset.label}
                            </option>
                          ))}
                          <option value="custom">직접 입력</option>
                        </select>
                      </label>
                    </div>
                    <div className="four-fields">
                      <label>
                        너비
                        <input min="256" max="2048" step="64" type="number" value={draft.imageProfile.width} onChange={(event) => updateImageProfile({ width: Number(event.target.value) })} />
                      </label>
                      <label>
                        높이
                        <input min="256" max="2048" step="64" type="number" value={draft.imageProfile.height} onChange={(event) => updateImageProfile({ height: Number(event.target.value) })} />
                      </label>
                      <label>
                        스텝
                        <input min="1" max="60" type="number" value={draft.imageProfile.steps} onChange={(event) => updateImageProfile({ steps: Number(event.target.value) })} />
                      </label>
                      <label>
                        CFG
                        <input min="1" max="15" step="0.1" type="number" value={draft.imageProfile.promptGuidance} onChange={(event) => updateImageProfile({ promptGuidance: Number(event.target.value) })} />
                      </label>
                    </div>
                  </div>

                  <div className="settings-section">
                    <SectionTitle icon={<Settings2 size={16} />} title="샘플링" />
                    <div className="four-fields">
                      <label>
                        샘플러
                        <select value={draft.novelAi.sampler} onChange={(event) => updateNovelAi({ sampler: event.target.value })}>
                          {novelAiSamplers.map((sampler) => (
                            <option key={sampler} value={sampler}>
                              {sampler}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        스케줄러
                        <select value={draft.novelAi.noiseSchedule} onChange={(event) => updateNovelAi({ noiseSchedule: event.target.value as NovelAiNoiseSchedule })}>
                          {novelAiNoiseSchedules.map((schedule) => (
                            <option key={schedule} value={schedule}>
                              {schedule}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        CFG 보정
                        <input min="0" max="1" step="0.02" type="number" value={draft.novelAi.cfgRescale} onChange={(event) => updateNovelAi({ cfgRescale: Number(event.target.value) })} />
                      </label>
                      <label>
                        UC preset
                        <input min="0" max="3" type="number" value={draft.novelAi.ucPreset} onChange={(event) => updateNovelAi({ ucPreset: Number(event.target.value) })} />
                      </label>
                    </div>
                    <div className="two-fields">
                      <label className="checkline">
                        <input checked={draft.novelAi.seedFixed} type="checkbox" onChange={(event) => updateNovelAi({ seedFixed: event.target.checked })} />
                        시드 고정
                      </label>
                      <label className="checkline">
                        <input checked={draft.novelAi.varPlus} type="checkbox" onChange={(event) => updateNovelAi({ varPlus: event.target.checked })} />
                        VAR+
                      </label>
                    </div>
                  </div>

                  <div className="settings-section">
                    <SectionTitle icon={<RefreshCcw size={16} />} title="자동 생성 운영" />
                    <div className="two-fields">
                      <label>
                        시드
                        <input type="number" value={draft.novelAi.seed ?? ""} onChange={(event) => updateNovelAi({ seed: event.target.value ? Number(event.target.value) : undefined })} />
                      </label>
                      <label>
                        지연 초
                        <input min="0" max="120" type="number" value={draft.novelAi.generationDelaySeconds} onChange={(event) => updateNovelAi({ generationDelaySeconds: Number(event.target.value) })} />
                      </label>
                    </div>
                    <div className="four-fields">
                      <label className="checkline">
                        <input checked={draft.novelAi.randomDelayEnabled} type="checkbox" onChange={(event) => updateNovelAi({ randomDelayEnabled: event.target.checked })} />
                        랜덤 지연
                      </label>
                      <label>
                        반복
                        <input min="1" max="100" type="number" value={draft.novelAi.repeatCount} onChange={(event) => updateNovelAi({ repeatCount: Number(event.target.value) })} />
                      </label>
                      <label>
                        종료 조건
                        <select value={draft.novelAi.automationTermination} onChange={(event) => updateNovelAi({ automationTermination: event.target.value as NovelAiAutomationTermination })}>
                          {novelAiAutomationTerminations.map((termination) => (
                            <option key={termination} value={termination}>
                              {termination}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        제한값
                        <input
                          min="1"
                          max="999"
                          type="number"
                          value={draft.novelAi.automationTermination === "timer" ? draft.novelAi.timerMinutes : draft.novelAi.countLimit}
                          onChange={(event) =>
                            updateNovelAi(
                              draft.novelAi.automationTermination === "timer"
                                ? { timerMinutes: Number(event.target.value) }
                                : { countLimit: Number(event.target.value) }
                            )
                          }
                        />
                      </label>
                    </div>
                  </div>
                </section>
              ) : null}

              {apiSection === "photoreal" ? (
                <section className="builder-panel span-2">
                  <SectionTitle icon={<ImageIcon size={17} />} title="실사 이미지 (ComfyUI)" />
                  <div className="readonly-field">
                    <span>실행자 ComfyUI</span>
                    <strong>실사 컷은 시뮬레이션을 실행하는 사람의 PC에서 ComfyUI로 생성됩니다.</strong>
                    <small>여기서는 라우팅과 프리셋, 스타일만 정합니다. ComfyUI 주소 · 크기 · GPU 교대는 실행 화면의 설정 탭에서 조정합니다.</small>
                  </div>
                  <ImageBackendRoutingCard
                    routing={draftBackendRouting}
                    novelAiEnabled={draft.novelAi.enabled}
                    photorealEnabled={draft.photoreal.enabled}
                    onChange={(backendRouting) => updateImageProfile({ backendRouting })}
                  />
                  <div className="settings-section">
                    <SectionTitle icon={<Settings2 size={16} />} title="실사 렌더러" />
                    <div className="two-fields">
                      <label className="checkline">
                        <input checked={draft.photoreal.enabled} type="checkbox" onChange={(event) => updatePhotoreal({ enabled: event.target.checked })} />
                        실사 이미지 사용
                      </label>
                      <label className="checkline">
                        <input
                          checked={draft.photoreal.faceReferenceEnabled}
                          type="checkbox"
                          onChange={(event) => updatePhotoreal({ faceReferenceEnabled: event.target.checked })}
                        />
                        캐릭터 실사 프로필로 얼굴 고정
                      </label>
                    </div>
                    <div className="two-fields">
                      <label>
                        생성용 워크플로
                        <select
                          value={draft.photoreal.workflowPreset}
                          onChange={(event) => updatePhotoreal({ workflowPreset: event.target.value as PhotorealWorkflowPresetId })}
                        >
                          {photorealWorkflowPresetOrder.map((presetId) => (
                            <option key={presetId} value={presetId}>
                              {PHOTOREAL_WORKFLOW_PRESETS[presetId].label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        얼굴 고정용 워크플로
                        <select
                          value={draft.photoreal.referenceWorkflowPreset}
                          onChange={(event) => updatePhotoreal({ referenceWorkflowPreset: event.target.value as PhotorealWorkflowPresetId })}
                        >
                          {photorealWorkflowPresetOrder
                            .filter(
                              (presetId) =>
                                PHOTOREAL_WORKFLOW_PRESETS[presetId].supportsReferences || presetId === draft.photoreal.referenceWorkflowPreset
                            )
                            .map((presetId) => (
                              <option key={presetId} value={presetId}>
                                {PHOTOREAL_WORKFLOW_PRESETS[presetId].label}
                              </option>
                            ))}
                        </select>
                      </label>
                    </div>
                    {draftRenderingPresets.map(({ preset, role }) => (
                      <div className="photoreal-preset-note" key={preset.id}>
                        <PhotorealLicenseTag preset={preset} />
                        <p className="runtime-hint">
                          {draftRenderingPresets.length > 1 ? <strong>{role} · {preset.label}</strong> : null}
                          {draftRenderingPresets.length > 1 ? " — " : null}
                          {preset.description} 라이선스: {preset.license}.
                        </p>
                      </div>
                    ))}
                  </div>
                  <div className="settings-section">
                    <SectionTitle icon={<Sparkles size={16} />} title="실사 스타일" />
                    <label className="checkline">
                      <input
                        type="checkbox"
                        checked={
                          draft.photoreal.enabled &&
                          draft.photoreal.stylePreset === "daily_look" &&
                          draft.photoreal.workflowPreset === "z_image_turbo"
                        }
                        onChange={(event) => {
                          if (event.target.checked) {
                            updatePhotoreal({
                              enabled: true,
                              stylePreset: "daily_look",
                              workflowPreset: "z_image_turbo",
                              referenceWorkflowPreset: "qwen_image_2_1_turbo",
                              width: 960,
                              height: 1280
                            });
                          } else {
                            updatePhotoreal({ stylePreset: "cinematic" });
                          }
                        }}
                      />
                      핀터레스트 데일리룩 사진 톤 켜기 (DNC 감성 · Z-Image Turbo 960×1280)
                    </label>
                    <label className="checkline">
                      <input
                        type="checkbox"
                        checked={
                          draft.photoreal.enabled &&
                          draft.photoreal.stylePreset === "raw_snap" &&
                          draft.photoreal.workflowPreset === "z_image_turbo"
                        }
                        onChange={(event) => {
                          if (event.target.checked) {
                            updatePhotoreal({
                              enabled: true,
                              stylePreset: "raw_snap",
                              workflowPreset: "z_image_turbo",
                              referenceWorkflowPreset: "qwen_image_2_1_turbo",
                              width: 768,
                              height: 1024
                            });
                          } else {
                            updatePhotoreal({ stylePreset: "cinematic", width: 832, height: 1248 });
                          }
                        }}
                      />
                      로우파이 리얼 폰카 모드 켜기 (저화질·생사진 · Z-Image Turbo 768×1024)
                    </label>
                    <label>
                      스타일 프리셋
                      <select
                        value={draft.photoreal.stylePreset}
                        onChange={(event) => updatePhotoreal({ stylePreset: event.target.value as PhotorealStylePresetId })}
                      >
                        {photorealStylePresetOrder.map((stylePreset) => (
                          <option key={stylePreset} value={stylePreset}>
                            {PHOTOREAL_STYLE_PRESETS[stylePreset].label}
                          </option>
                        ))}
                        <option value="custom">사용자 지정</option>
                      </select>
                    </label>
                    <p className="runtime-hint">
                      {draft.photoreal.stylePreset === "custom"
                        ? "아래 스타일 문장이 전부입니다. 첫 문장이 프롬프트 맨 앞에 오고, 나머지는 끝에 붙습니다."
                        : PHOTOREAL_STYLE_PRESETS[draft.photoreal.stylePreset].description}
                    </p>
                    <label>
                      {draft.photoreal.stylePreset === "custom" ? "스타일 (영어)" : "스타일 추가 (영어, 선택)"}
                      <textarea
                        rows={3}
                        value={draft.photoreal.stylePrompt}
                        aria-describedby="builder-photoreal-style-screen"
                        onChange={(event) => updatePhotoreal({ stylePrompt: event.target.value })}
                      />
                    </label>
                    <PhotorealTextScreenNote
                      id="builder-photoreal-style-screen"
                      scope="style"
                      state={draftScreeningState}
                      text={draft.photoreal.stylePrompt}
                    />
                    <label>
                      여성 인물 공통 외모 (실사 전용, 영어)
                      <textarea
                        rows={3}
                        value={draft.photoreal.appearancePrompt}
                        placeholder={PHOTOREAL_SHARED_APPEARANCE_PLACEHOLDER}
                        aria-describedby="builder-photoreal-appearance-screen"
                        onChange={(event) => updatePhotoreal({ appearancePrompt: event.target.value })}
                      />
                    </label>
                    <PhotorealTextScreenNote
                      id="builder-photoreal-appearance-screen"
                      scope="appearance"
                      state={draftScreeningState}
                      text={draft.photoreal.appearancePrompt}
                    />
                    <p className="runtime-hint">실사 컷의 모든 여성 인물에게 붙습니다. 캐릭터마다 다른 외형은 캐릭터의 실사 외형 칸에 씁니다.</p>
                    <label>
                      네거티브 추가 (영어, 선택)
                      <textarea rows={2} value={draft.photoreal.negativePrompt} onChange={(event) => updatePhotoreal({ negativePrompt: event.target.value })} />
                    </label>
                  </div>
                </section>
              ) : null}

              {apiSection === "neuralmap" ? (
                <section className="builder-panel">
                  <SectionTitle icon={<Database size={17} />} title="NeuralMap" />
                  <label className="checkline">
                    <input checked={draft.neuralMap.enabled} type="checkbox" onChange={(event) => updateNeuralMap({ enabled: event.target.checked })} />
                    API 연동 사용
                  </label>
                  <label>
                    기본 URL
                    <input value={draft.neuralMap.baseUrl} onChange={(event) => updateNeuralMap({ baseUrl: event.target.value })} />
                  </label>
                  <label>
                    검색 문맥 토큰 예산
                    <input min="800" max="12000" step="200" type="number" value={draft.neuralMap.tokenBudget} onChange={(event) => updateNeuralMap({ tokenBudget: Number(event.target.value) })} />
                  </label>
                </section>
              ) : null}
            </div>
          </RailSubtabs>
        ) : null}

        {activeTab === "review" ? (
          <div className="builder-grid">
            <section className="builder-panel span-2">
              <SectionTitle icon={<Check size={17} />} title="검토" />
              <div className="review-grid">
                <Capability label="시뮬레이션" value={draft.title || "제목 없음"} />
                <Capability label="진행 모드" value={promptModeLabels[draft.promptMode]} />
                <Capability label="프롬프트 모듈" value={`${draft.modules.length}개`} />
                <Capability label="관계/상태창" value={draft.relationshipMap.enabled ? "memory_events 저장" : "비활성"} />
                <Capability label="NovelAI" value={draft.novelAi.enabled ? `${draft.novelAi.modelPreset} ${draft.novelAi.requestMode}` : "mock"} />
                <Capability label="실사" value={draft.photoreal.enabled ? draftPhotorealPreset.label : "꺼짐"} />
                <Capability label="이미지 라우팅" value={draftRoutingModeLabel} />
                <Capability label="NeuralMap" value={draft.neuralMap.enabled ? draft.neuralMap.baseUrl : "로컬 대체"} />
                <Capability label="이미지 밀도" value={activeImageGenerationCadence.label} />
                <Capability label="장면 키워드" value={`${countEnabledImageScenePresetNodes(draftImageScenePresets)}개`} />
                <Capability label="시작상황" value={draft.startSituationPrompt.trim() ? "지정됨" : "기본 시작"} />
              </div>
              <label>
                시작 프롬프트 미리보기
                <textarea readOnly value={[draft.startSituationPrompt, draft.mainPrompt, draft.characterSummary, draft.worldLore].filter(Boolean).join("\n\n")} />
              </label>
              <button className="send-button wide-action" type="button" onClick={() => onCreate(draft)}>
                <Save size={17} />
                시뮬레이션 생성
              </button>
            </section>
            <aside className="builder-panel">
              <SectionTitle icon={<ImageIcon size={17} />} title="이미지 설정" />
              <Metric label="해상도" value={`${draft.imageProfile.width}x${draft.imageProfile.height}`} />
              <Metric label="스텝" value={draft.imageProfile.steps.toString()} />
              <Metric label="CFG" value={draft.imageProfile.promptGuidance.toString()} />
              <Metric label="샘플러" value={draft.novelAi.sampler} />
              <Metric label="라우팅" value={draftRoutingModeLabel} />
            </aside>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function createInitialSimulationDraft(runtimeSource?: AppState): SimulationDraft {
  const now = new Date().toISOString();
  const preset = promptModePresets.basic;
  const mainPrompt = preset.mainPrompt;
  const characterSummary = preset.characterSummary;
  const worldLore = preset.worldLore;
  const startSituationPrompt = preset.startSituationPrompt;
  const visualPrompt = preset.visualPrompt;
  const runtimeLlm = runtimeSource?.llm ?? seedState.llm;
  const runtimeNovelAi = runtimeSource?.novelAi ?? seedState.novelAi;
  // Inherited like the LLM and NovelAI settings: the ComfyUI address, presets and GPU handoff describe this
  // machine's setup, which a new simulation should not make the user enter again.
  const runtimePhotoreal = runtimeSource?.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS;

  return {
    promptMode: preset.promptMode,
    contentRating: "general",
    title: preset.title,
    description: preset.description,
    mainPrompt,
    characterName: preset.characterName,
    characterRole: preset.characterRole,
    characterSummary,
    characterRelationship: preset.characterRelationship,
    characterMood: preset.characterMood,
    worldLore,
    startSituationPrompt,
    visualPrompt,
    negativeVisualPrompt: preset.negativeVisualPrompt,
    realtimeImageEnabled: true,
    imageScenePresets: createDefaultDraftImageScenePresets(now, preset.promptMode),
    characters: [
      {
        id: "draft_character_id",
        name: preset.characterName,
        role: preset.characterRole,
        summary: characterSummary,
        relationship: preset.characterRelationship,
        currentMood: preset.characterMood,
        visualPrompt,
        negativeVisualPrompt: preset.negativeVisualPrompt,
        defaultOutfitPrompt,
        outfitPrompts: defaultOutfitPrompts,
        expressionPrompts: defaultExpressionPrompts,
        defaultSafetyLevel: "safe"
      }
    ],
    modules: [
      {
        id: "draft_main",
        simulationId: "draft_simulation",
        kind: "main_prompt",
        title: `메인 규칙: ${promptModeLabels[preset.promptMode]}`,
        body: mainPrompt,
        enabled: true,
        priority: 100,
        activationTags: ["core", "always", ...preset.activationTags],
        tokenPolicy: "always",
        version: 1,
        updatedAt: now
      },
      {
        id: "draft_character",
        simulationId: "draft_simulation",
        parentId: "draft_main",
        kind: "character_prompt",
        title: `캐릭터: ${preset.characterName}`,
        body: characterSummary,
        enabled: true,
        priority: 84,
        activationTags: [preset.characterName.toLowerCase(), "character", ...preset.activationTags],
        characterId: "draft_character_id",
        tokenPolicy: "rag",
        version: 1,
        updatedAt: now
      },
      {
        id: "draft_world",
        simulationId: "draft_simulation",
        parentId: "draft_main",
        kind: "world_lore",
        title: `세계관: ${promptModeLabels[preset.promptMode]}`,
        body: worldLore,
        enabled: true,
        priority: 76,
        activationTags: ["world", "lore", ...preset.activationTags],
        tokenPolicy: "rag",
        version: 1,
        updatedAt: now
      },
      {
        id: "draft_mode_rule",
        simulationId: "draft_simulation",
        parentId: "draft_main",
        kind: "scene_rule",
        title: preset.modeRuleTitle,
        body: preset.modeRule,
        enabled: true,
        priority: 78,
        activationTags: ["mode-rule", ...preset.activationTags],
        tokenPolicy: "rag",
        version: 1,
        updatedAt: now
      },
      {
        id: "draft_image_style",
        simulationId: "draft_simulation",
        parentId: "draft_main",
        kind: "image_prompt_profile",
        title: `이미지 스타일: ${promptModeLabels[preset.promptMode]}`,
        body: preset.imageStylePrompt,
        enabled: true,
        priority: 82,
        activationTags: ["image", "style", "nai", ...preset.activationTags],
        tokenPolicy: "always",
        version: 1,
        updatedAt: now
      }
    ],
    imageProfile: {
      ...seedState.imageProfile,
      id: "draft_image_profile",
      simulationId: "draft_simulation",
      model: "nai-diffusion-4-5-full",
      width: 1024,
      height: 1024,
      steps: 28,
      promptGuidance: 5,
      countMin: 1,
      countMax: 2,
      qualityPrompt: "masterpiece, best quality, detailed background",
      stylePrompt: preset.imageStylePrompt,
      negativePrompt: "lowres, blurry, worst quality, text, watermark, bad anatomy",
      safetyLevel: "safe",
      userRules: preset.imageUserRules,
      triggerMode: "realtime_auto"
    },
    neuralMap: {
      ...seedState.neuralMap
    },
    relationshipMap: {
      ...seedState.relationshipMap,
      updatedAt: now
    },
    llm: {
      ...toShareableLlmSettings(runtimeLlm),
      systemPrompt: preset.llmSystemPrompt,
      temperature: preset.temperature,
      maxTokens: preset.maxTokens
    },
    novelAi: toShareableNovelAiSettings(runtimeNovelAi),
    photoreal: toShareablePhotorealSettings(runtimePhotoreal)
  };
}

function BuilderTabButton({
  active,
  icon,
  label,
  metric,
  onClick,
  panelKey
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  metric: string;
  onClick: () => void;
  panelKey: string;
}) {
  return (
    <button
      className={`builder-tab ${active ? "active" : ""}`}
      type="button"
      role="tab"
      // Only `.builder-body` for the selected step exists, so only the selected tab can name a panel.
      id={`tab-builder-${panelKey}`}
      aria-controls={active ? `tabpanel-builder-${panelKey}` : undefined}
      aria-selected={active}
      tabIndex={active ? 0 : -1}
      onClick={onClick}
    >
      <span>{icon}</span>
      <strong>{label}</strong>
      <small>{metric}</small>
    </button>
  );
}

function PromptModeCard({
  active,
  option,
  onClick
}: {
  active: boolean;
  option: (typeof promptModeOptions)[number];
  onClick: () => void;
}) {
  const icon =
    option.value === "basic" ? (
      <MessageSquareText size={18} />
    ) : option.value === "one_on_one" ? (
      <UserRound size={18} />
    ) : option.value === "simulation" ? (
      <Activity size={18} />
    ) : (
      <Pencil size={18} />
    );

  return (
    <button
      aria-checked={active}
      className={`prompt-mode-card ${active ? "active" : ""}`}
      role="radio"
      type="button"
      onClick={onClick}
    >
      <span className="prompt-mode-icon">{icon}</span>
      <span className="section-kicker">{option.kicker}</span>
      <strong>{option.label}</strong>
      <p>{option.description}</p>
      <small>{option.metric}</small>
    </button>
  );
}

function ApiStatusCard({
  title,
  status,
  message,
  verifiedAt,
  detail
}: {
  title: string;
  status: LlmApiSettings["registrationStatus"];
  message?: string;
  verifiedAt?: string;
  detail?: string;
}) {
  const tone = status === "registered" ? "good" : "neutral";
  const label = status === "registered" ? "등록 완료" : status === "verifying" ? "검증 중" : status === "failed" ? "검증 실패" : "미등록";

  return (
    <div className={`api-status-card ${status}`}>
      <div>
        <strong>{title}</strong>
        <span>{message || "개인 설정에서 API 키 등록 상태를 확인하세요."}</span>
        {verifiedAt ? <small>{new Date(verifiedAt).toLocaleString("ko-KR")}</small> : null}
        {detail ? <small>{detail}</small> : null}
      </div>
      <StatusPill icon={<Check size={15} />} label={label} tone={tone} />
    </div>
  );
}

function CharacterPromptPanel({
  character,
  onChange,
  onModuleChange
}: {
  character: SimulationCharacterDraft;
  onChange: (patch: Partial<SimulationCharacterDraft>) => void;
  onModuleChange: (patch: Partial<PromptModule>) => void;
}) {
  return (
    <div className="module-detail-panel">
      <SectionTitle icon={<Bot size={17} />} title="캐릭터 설정" />
      <div className="two-fields">
        <label>
          <GuidedFieldLabel guide="대화, 기억, 이미지 cue에서 쓰이는 캐릭터 이름입니다. 같은 캐릭터를 계속 추적할 수 있게 일관된 이름을 쓰세요." title="이름" />
          <input
            value={character.name}
            onChange={(event) => {
              onChange({ name: event.target.value });
              onModuleChange({ title: `캐릭터: ${event.target.value || "캐릭터"}` });
            }}
          />
        </label>
        <label>
          <GuidedFieldLabel guide="세계 안에서의 위치나 플레이어와의 관계입니다. 예: 조력자, 라이벌, 보호자, 운영자, 동료." title="역할" />
          <input value={character.role} onChange={(event) => onChange({ role: event.target.value })} />
        </label>
      </div>
      <label>
        <GuidedFieldLabel badge={moduleKindGuides.character_prompt.policy} guide={moduleKindGuides.character_prompt.detail} title="캐릭터 프롬프트" />
        <textarea
          placeholder={moduleKindGuides.character_prompt.placeholder}
          value={character.summary}
          onChange={(event) => {
            onChange({ summary: event.target.value });
            onModuleChange({ body: event.target.value });
          }}
        />
      </label>
    </div>
  );
}

function createUniqueMappingKey(record: Record<string, string>, preferredKey: string): string {
  const baseKey = preferredKey.trim() || "새 문맥";
  if (!Object.prototype.hasOwnProperty.call(record, baseKey)) {
    return baseKey;
  }

  let index = 2;
  while (Object.prototype.hasOwnProperty.call(record, `${baseKey} ${index}`)) {
    index += 1;
  }
  return `${baseKey} ${index}`;
}

type PromptMappingRow = {
  id: string;
  key: string;
  value: string;
};

function createPromptMappingRows(record: Record<string, string>): PromptMappingRow[] {
  return Object.entries(record).map(([key, value]) => ({
    id: createId("prompt_mapping"),
    key,
    value
  }));
}

function createPromptMappingRecord(rows: PromptMappingRow[]): Record<string, string> {
  return rows.reduce<Record<string, string>>((record, row) => {
    const key = createUniqueMappingKey(record, row.key);
    record[key] = row.value;
    return record;
  }, {});
}

function CharacterVisualMappingPanel({
  character,
  onCharacterChange
}: {
  character: SimulationCharacterDraft;
  onCharacterChange: (patch: Partial<SimulationCharacterDraft>) => void;
}) {
  const [outfitRows, setOutfitRows] = useState<PromptMappingRow[]>(() => createPromptMappingRows(character.outfitPrompts ?? {}));
  const previousCharacterIdRef = useRef(character.id);

  useEffect(() => {
    if (previousCharacterIdRef.current !== character.id) {
      previousCharacterIdRef.current = character.id;
      setOutfitRows(createPromptMappingRows(character.outfitPrompts ?? {}));
    }
  }, [character.id, character.outfitPrompts]);

  const commitOutfitRows = useCallback(
    (rows: PromptMappingRow[]) => {
      setOutfitRows(rows);
      onCharacterChange({ outfitPrompts: createPromptMappingRecord(rows) });
    },
    [onCharacterChange]
  );

  return (
    <div className="module-detail-panel">
      <SectionTitle icon={<ImageIcon size={17} />} title="캐릭터 이미지 매핑" />
      <div className="readonly-field">
        <span>캐릭터</span>
        <strong>{character.name}</strong>
      </div>
      <label>
        <GuidedFieldLabel guide="이 캐릭터를 이미지로 부를 때 반복 적용할 외형과 식별 특징입니다. 상황별 의상은 아래 의상 필드에 분리하면 장면에 맞춰 자동 선택됩니다." title="Positive prompt" />
        <textarea value={character.visualPrompt} onChange={(event) => onCharacterChange({ visualPrompt: event.target.value })} />
      </label>
      <label>
        <GuidedFieldLabel
          guide="문맥 키워드에 맞는 의상이 없을 때 쓰는 캐릭터의 기본 의상 태그입니다. 외형 식별 태그와 분리해 두면 장면 변화 때 의상만 안정적으로 교체됩니다."
          title="기본 의상 태그"
        />
        <input
          placeholder="예: school uniform, navy cardigan"
          value={character.defaultOutfitPrompt}
          onChange={(event) => onCharacterChange({ defaultOutfitPrompt: event.target.value })}
        />
      </label>
      <label>
        <GuidedFieldLabel guide="이 캐릭터에게 특히 피해야 할 이미지 요소를 적습니다. 공통 negative prompt보다 캐릭터별 금지 요소에 집중하세요." title="Negative prompt" />
        <textarea value={character.negativeVisualPrompt} onChange={(event) => onCharacterChange({ negativeVisualPrompt: event.target.value })} />
      </label>
      <div className="visual-mapping-block">
        <div className="mapping-block-header">
          <GuidedFieldLabel
            guide="선택 사항입니다. 특정 문맥에서 반드시 쓰고 싶은 의상이 있을 때만 추가하세요. 비어 있으면 진행 중 장면 문맥에서 의상을 동적으로 합성합니다."
            title="상황별 의상 매핑"
          />
          <button
            className="icon-text-button"
            type="button"
            onClick={() => {
              const currentRecord = createPromptMappingRecord(outfitRows);
              commitOutfitRows([
                ...outfitRows,
                {
                  id: createId("prompt_mapping"),
                  key: createUniqueMappingKey(currentRecord, "새 의상 문맥"),
                  value: ""
                }
              ]);
            }}
          >
            <Plus size={15} />
            추가
          </button>
        </div>
        {outfitRows.length > 0 ? (
          <div className="visual-mapping-list">
            {outfitRows.map((row) => (
              <div className="visual-mapping-row" key={row.id}>
                <label>
                  문맥 키워드
                  <input
                    placeholder="예: 겨울, 무도회, 전투, 실험실"
                    value={row.key}
                    onChange={(event) =>
                      commitOutfitRows(outfitRows.map((candidate) => (candidate.id === row.id ? { ...candidate, key: event.target.value } : candidate)))
                    }
                  />
                </label>
                <label>
                  의상 프롬프트
                  <input
                    placeholder="예: long black coat, silver brooch"
                    value={row.value}
                    onChange={(event) =>
                      commitOutfitRows(outfitRows.map((candidate) => (candidate.id === row.id ? { ...candidate, value: event.target.value } : candidate)))
                    }
                  />
                </label>
                <button
                  className="icon-button danger"
                  type="button"
                  onClick={() => commitOutfitRows(outfitRows.filter((candidate) => candidate.id !== row.id))}
                  aria-label={`${row.key || "의상"} 매핑 삭제`}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="mapping-empty-note">의상 매핑이 없습니다. 장면 문맥에 맞춰 자동 의상 힌트를 생성합니다.</p>
        )}
      </div>
      <label>
        <GuidedFieldLabel guide="이 캐릭터 이미지 생성의 기본 출력 수위입니다. 19+ 성인 전용 시뮬레이션은 런타임 등급에서 explicit로 자동 맞춰집니다." title="기본 이미지 수위" />
        <select value={character.defaultSafetyLevel} onChange={(event) => onCharacterChange({ defaultSafetyLevel: event.target.value as ImageSafetyLevel })}>
          {imageSafetyLevels.map((level) => (
            <option key={level} value={level}>
              {level}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

// --- Character profile images (creation screen) -----------------------------------------------------------------

/** One profile test for a draft character, as the builder asks the App to run it. */
interface CharacterProfileTestRequest {
  draft: SimulationDraft;
  draftCharacterId: string;
  backend: ImageBackendId;
  prompt: string;
  negativePrompt: string;
  variant: CharacterProfileTestVariant;
  count: number;
  signal: AbortSignal;
  /**
   * Once planned: how many candidates will run. They run on the requested backend or not at all — a request the
   * safety floor would move to the other renderer is refused (thrown) before this is called.
   */
  onPlanned?: (plan: { total: number }) => void;
  /** The test has left the renderer's queue and is generating. */
  onStart?: () => void;
  onImage?: (asset: ImageAsset) => void;
  onSettled?: (settled: number, total: number) => void;
}

interface CharacterProfileTestResult {
  assets: ImageAsset[];
  errors: string[];
  backend: ImageBackendId;
}

interface CharacterProfileImagePersistRequest {
  /** The object-store directory: the edited simulation's id, or the builder's provisional id while creating. */
  simulationId: string;
  draft: SimulationDraft;
  draftCharacterId: string;
  /** The renderer that drew the candidate, which is also the profile slot it fills. */
  backend: ImageBackendId;
  asset: ImageAsset;
  prompt: string;
  negativePrompt: string;
}

/** A test image held by the builder. Its data URL lives here only: never in the draft, App state or storage. */
type ProfileStudioCandidate = {
  id: string;
  asset: ImageAsset;
  backend: ImageBackendId;
  prompt: string;
  negativePrompt: string;
  runId: string;
  persisting?: boolean;
  /** The profile record this candidate became, so the grid can mark it for as long as it is the profile. */
  profileAssetId?: string;
  error?: string;
};

type ProfileStudioRun = {
  id: string;
  variant: CharacterProfileTestVariant;
  /** Where the candidates render. Never moved: a request the safety floor rules out there is refused instead. */
  requestedBackend: ImageBackendId;
  total: number;
  settled: number;
  phase: "planning" | "queued" | "generating";
};

type ProfileStudioSession = {
  backend?: ImageBackendId;
  /** Test prompts as typed, per backend. A missing entry follows the character's own fields. */
  prompts: Partial<Record<ImageBackendId, string>>;
  count: number;
  run?: ProfileStudioRun;
  candidates: ProfileStudioCandidate[];
  /** The latest face check: the photoreal profile drawn in a different pose. */
  checks: ProfileStudioCandidate[];
  errors: string[];
  /**
   * A test that was refused before anything was queued: the planner throws when the photoreal safety floor (or a
   * missing profile) rules the request out, rather than quietly running it on NovelAI. Shown apart from `errors`
   * because nothing ran and nothing failed — the request itself has to change.
   */
  refusal?: ProfileStudioRefusal;
  notice?: string;
};

type ProfileStudioRefusal = {
  message: string;
  variant: CharacterProfileTestVariant;
  backend: ImageBackendId;
};

type ProfileStudioGenerateInput = {
  backend: ImageBackendId;
  prompt: string;
  negativePrompt: string;
  variant: CharacterProfileTestVariant;
  count: number;
};

type ProfileTestBlocker = {
  message: string;
  /** A one-click fix on the draft, when the fix is a single setting. */
  action?: "enable_photoreal" | "photoreal_proxy";
};

const EMPTY_PROFILE_STUDIO_SESSION: ProfileStudioSession = { prompts: {}, count: 2, candidates: [], checks: [], errors: [] };
/** A candidate is a few MB of PNG held in memory, so the grid keeps the latest dozen. */
const MAX_PROFILE_STUDIO_CANDIDATES = 12;
/** Two images rather than one, so a face that holds on one seed and drifts on the next shows up. */
const PROFILE_REFERENCE_CHECK_COUNT = 2;
const PROFILE_TEST_CANCELED_NOTICE = "테스트 생성을 취소했습니다.";
// "예:" like the builder's other placeholders: a full English sentence in placeholder grey reads as text already typed.
const PHOTOREAL_APPEARANCE_PLACEHOLDER =
  "예: a Korean woman in her late twenties with shoulder-length wavy brown hair, light natural makeup, cream knit cardigan";
/** The photoreal prompt builder drops Hangul (its text encoders are prompted in English), so Korean text never arrives. */
const HANGUL_TEXT_PATTERN = /[가-힣]/u;
/** How long typing has to pause before a photo-side text is screened (PhotorealTextScreenNote). */
const PHOTOREAL_TEXT_SCREEN_DELAY_MS = 400;

/**
 * The builder's profile tests, one session per draft character, held by the page rather than by the panel. A
 * NovelAI candidate costs Anlas, so switching to another character or tab must not throw it away, and a test keeps
 * running while its panel is hidden. Leaving the builder ends all of it: the sessions are component state and
 * nothing else, and unmounting aborts whatever is still running or waiting.
 */
function useCharacterProfileStudio({
  draft,
  simulationId,
  onTestGenerate,
  onPersist,
  onProfileImageChange
}: {
  draft: SimulationDraft;
  simulationId: string;
  onTestGenerate: (request: CharacterProfileTestRequest) => Promise<CharacterProfileTestResult>;
  onPersist: (request: CharacterProfileImagePersistRequest) => Promise<CharacterProfileImage>;
  onProfileImageChange: (characterId: string, backend: ImageBackendId, image: CharacterProfileImage | undefined) => void;
}) {
  const [sessions, setSessions] = useState<Record<string, ProfileStudioSession>>({});
  const controllersRef = useRef(new Map<string, AbortController>());

  useEffect(() => {
    const controllers = controllersRef.current;
    return () => {
      controllers.forEach((controller) => controller.abort());
      controllers.clear();
    };
  }, []);

  // A character deleted from the draft takes its running test and its candidates with it.
  useEffect(() => {
    const liveIds = new Set(draft.characters.map((character) => character.id));
    controllersRef.current.forEach((controller, characterId) => {
      if (!liveIds.has(characterId)) {
        controller.abort();
      }
    });
    setSessions((current) => {
      const staleIds = Object.keys(current).filter((characterId) => !liveIds.has(characterId));
      if (staleIds.length === 0) {
        return current;
      }
      const next = { ...current };
      staleIds.forEach((characterId) => delete next[characterId]);
      return next;
    });
  }, [draft.characters]);

  const updateSession = useCallback((characterId: string, change: (session: ProfileStudioSession) => ProfileStudioSession) => {
    setSessions((current) => ({ ...current, [characterId]: change(current[characterId] ?? EMPTY_PROFILE_STUDIO_SESSION) }));
  }, []);

  // For results that land after an await: a session that is gone (its character was deleted) stays gone.
  const patchSession = useCallback((characterId: string, change: (session: ProfileStudioSession) => ProfileStudioSession) => {
    setSessions((current) => {
      const session = current[characterId];
      return session ? { ...current, [characterId]: change(session) } : current;
    });
  }, []);

  const start = useCallback(
    async (characterId: string, input: ProfileStudioGenerateInput) => {
      if (controllersRef.current.has(characterId)) {
        return;
      }
      const controller = new AbortController();
      controllersRef.current.set(characterId, controller);
      const runId = createId("profile_run");
      const patchRun = (change: (run: ProfileStudioRun) => ProfileStudioRun) =>
        patchSession(characterId, (session) => (session.run?.id === runId ? { ...session, run: change(session.run) } : session));
      updateSession(characterId, (session) => ({
        ...session,
        run: { id: runId, variant: input.variant, requestedBackend: input.backend, total: input.count, settled: 0, phase: "planning" },
        errors: [],
        refusal: undefined,
        notice: undefined,
        checks: input.variant === "reference_check" ? [] : session.checks
      }));

      let errors: string[] = [];
      let refusal: ProfileStudioRefusal | undefined;
      // Whether the request got past planning. A throw before that is the planner refusing the request (nothing was
      // queued or spent); a throw after it is a run that went wrong.
      let planned = false;
      try {
        const result = await onTestGenerate({
          ...input,
          draft,
          draftCharacterId: characterId,
          signal: controller.signal,
          onPlanned: ({ total }) => {
            planned = true;
            patchRun((run) => ({ ...run, total, phase: "queued" }));
          },
          onStart: () => patchRun((run) => ({ ...run, phase: "generating" })),
          onImage: (asset) =>
            patchSession(characterId, (session) =>
              addProfileStudioImage(session, input.variant, {
                id: asset.id,
                asset,
                backend: asset.backend ?? input.backend,
                prompt: input.prompt,
                negativePrompt: input.negativePrompt,
                runId
              })
            ),
          onSettled: (settled, total) => patchRun((run) => ({ ...run, settled, total }))
        });
        errors = result.errors;
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) {
          errors = [];
        } else if (!planned) {
          refusal = {
            message: readErrorMessage(error, "이 요청으로는 테스트를 시작하지 않았습니다. 프롬프트를 바꿔 다시 시도하세요."),
            variant: input.variant,
            backend: input.backend
          };
        } else {
          errors = [readErrorMessage(error, "알 수 없는 문제로 테스트 생성을 끝내지 못했습니다.")];
        }
      } finally {
        if (controllersRef.current.get(characterId) === controller) {
          controllersRef.current.delete(characterId);
        }
      }
      const canceled = controller.signal.aborted;
      patchSession(characterId, (session) => ({
        ...session,
        run: session.run?.id === runId ? undefined : session.run,
        errors,
        refusal,
        notice: canceled ? PROFILE_TEST_CANCELED_NOTICE : undefined
      }));
    },
    [draft, onTestGenerate, patchSession, updateSession]
  );

  const cancel = useCallback((characterId: string) => {
    controllersRef.current.get(characterId)?.abort();
  }, []);

  const pick = useCallback(
    async (characterId: string, candidate: ProfileStudioCandidate) => {
      const patchCandidate = (patch: Partial<ProfileStudioCandidate>) =>
        patchSession(characterId, (session) => ({
          ...session,
          candidates: session.candidates.map((entry) => (entry.id === candidate.id ? { ...entry, ...patch } : entry))
        }));
      patchCandidate({ persisting: true, error: undefined });
      try {
        const record = await onPersist({
          simulationId,
          draft,
          draftCharacterId: characterId,
          backend: candidate.backend,
          asset: candidate.asset,
          prompt: candidate.prompt,
          negativePrompt: candidate.negativePrompt
        });
        onProfileImageChange(characterId, candidate.backend, record);
        patchCandidate({ persisting: false, profileAssetId: record.assetId });
      } catch (error) {
        // The candidate stays in the grid, so the pick can be retried once the server is back.
        patchCandidate({ persisting: false, error: readErrorMessage(error, "프로필 이미지 저장 실패") });
      }
    },
    [draft, onPersist, onProfileImageChange, patchSession, simulationId]
  );

  const discardCandidates = useCallback(
    (characterId: string) => {
      // A pick still being stored keeps its candidate, or its result would have nowhere to report.
      patchSession(characterId, (session) => ({ ...session, candidates: session.candidates.filter((entry) => entry.persisting), checks: [] }));
    },
    [patchSession]
  );

  return { sessions, updateSession, start, cancel, pick, discardCandidates };
}

/** Newest test first, each test's candidates in the order they arrived; a face check replaces the previous one. */
function addProfileStudioImage(
  session: ProfileStudioSession,
  variant: CharacterProfileTestVariant,
  candidate: ProfileStudioCandidate
): ProfileStudioSession {
  if (variant === "reference_check") {
    return session.checks.some((entry) => entry.id === candidate.id) ? session : { ...session, checks: [...session.checks, candidate] };
  }
  if (session.candidates.some((entry) => entry.id === candidate.id)) {
    return session;
  }
  const thisRun = session.candidates.filter((entry) => entry.runId === candidate.runId);
  const earlier = session.candidates.filter((entry) => entry.runId !== candidate.runId);
  return { ...session, candidates: [...thisRun, candidate, ...earlier].slice(0, MAX_PROFILE_STUDIO_CANDIDATES) };
}

/** Why a test cannot run on `backend` right now, judged from the draft (and, for NovelAI, whether a token exists). */
function describeProfileTestBlocker(backend: ImageBackendId, draft: SimulationDraft, novelAiTokenRegistered: boolean): ProfileTestBlocker | undefined {
  if (backend === "novelai") {
    if (!draft.novelAi.enabled) {
      return { message: "NovelAI 연동이 꺼져 있습니다. 이미지/연동 탭에서 'NovelAI 이미지 API 사용'을 켜면 테스트할 수 있습니다." };
    }
    if (draft.novelAi.requestMode === "mock") {
      return { message: "NovelAI 요청 방식이 mock이라 실제 이미지를 만들지 않습니다. 이미지/연동 탭에서 proxy 또는 direct로 바꾸세요." };
    }
    if (!novelAiTokenRegistered) {
      return { message: "NovelAI 토큰을 등록하면 테스트할 수 있습니다. 상단 '개인 설정'에서 등록하세요." };
    }
    return undefined;
  }
  if (!draft.photoreal.enabled) {
    return { message: "실사 이미지(ComfyUI) 연동이 꺼져 있습니다.", action: "enable_photoreal" };
  }
  if (draft.photoreal.requestMode === "mock") {
    return { message: "실사 요청 방식이 mock이라 실제 이미지를 만들지 않습니다.", action: "photoreal_proxy" };
  }
  return undefined;
}

function describeProfileStudioRun(run: ProfileStudioRun): string {
  const backendLabel = imageBackendLabels[run.requestedBackend];
  const subject = run.variant === "reference_check" ? "얼굴 고정 테스트" : "후보";
  if (run.phase === "planning") {
    return `${backendLabel} ${subject} 준비 중…`;
  }
  if (run.phase === "queued") {
    // Not only earlier images: a photoreal test also waits for the local GPU while a local LLM turn holds it.
    return `${backendLabel} 앞선 작업이 끝나기를 기다리는 중…`;
  }
  return `${backendLabel} ${subject} 생성 중 · ${run.settled}/${run.total}`;
}

/**
 * `profileImages` with one backend's slot set or cleared. An emptied map becomes undefined rather than {}: the key
 * itself stays on the draft character, which updateStateFromDraft reads as "cleared in the editor".
 */
function withCharacterProfileImage(
  images: CharacterProfileImages | undefined,
  backend: ImageBackendId,
  image: CharacterProfileImage | undefined
): CharacterProfileImages | undefined {
  const next: CharacterProfileImages = { ...images };
  if (image) {
    next[backend] = image;
  } else {
    delete next[backend];
  }
  return Object.keys(next).length > 0 ? next : undefined;
}

/** The profile shown for a character wherever one face stands for them: photoreal first, then NovelAI. */
function selectDisplayProfileImage(images: CharacterProfileImages | undefined): CharacterProfileImage | undefined {
  return [images?.photoreal, images?.novelai].find((image): image is CharacterProfileImage => Boolean(image?.objectKey));
}

function describeProfileImageBackends(images: CharacterProfileImages | undefined): string | undefined {
  const labels = (["photoreal", "novelai"] as const).filter((backend) => images?.[backend]?.objectKey).map((backend) => imageBackendLabels[backend]);
  return labels.length > 0 ? labels.join(" · ") : undefined;
}

function describeProfileImageMeta(image: CharacterProfileImage): string {
  // The preset's name without its parenthesised gloss: this is one short line under a thumbnail.
  const model =
    image.backend === "photoreal"
      ? image.workflowPreset
        ? PHOTOREAL_WORKFLOW_PRESETS[image.workflowPreset]?.label.replace(/\s*\([^)]*\)\s*$/u, "")
        : undefined
      : image.modelPreset;
  return [image.width && image.height ? `${image.width}×${image.height}` : undefined, model, image.seed !== undefined ? `seed ${image.seed}` : undefined]
    .filter(Boolean)
    .join(" · ");
}

function joinPromptParts(parts: Array<string | undefined>): string {
  return parts
    .map((part) => part?.trim() ?? "")
    .filter(Boolean)
    .join(", ");
}

function isAbortError(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { name?: unknown }).name === "AbortError";
}

function readErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback;
}

/**
 * Stores a picked candidate as a character's profile image and returns the record the builder puts on the draft.
 *
 * The bytes go to the object store under the builder's simulation id with an object id minted for this pick, which
 * is also the record's assetId, so an id never names two images. A NovelAI pick additionally stores the image
 * letterboxed onto a Precise Reference canvas, the copy every NovelAI cut would send. Both come from decoding the
 * candidate itself, which also yields the pixel size the record must carry: the photoreal reference is head-cropped
 * from the stored width and height. Every failure comes back as one Korean message with "실패" in it; the builder
 * keeps the candidate so the pick can be retried.
 */
async function persistCharacterProfileImage(request: CharacterProfileImagePersistRequest): Promise<CharacterProfileImage> {
  try {
    return await storeCharacterProfileImage(request);
  } catch (error) {
    throw new Error(`프로필 이미지 저장 실패: ${readErrorMessage(error, "알 수 없는 문제로 저장하지 못했습니다.")}`);
  }
}

async function storeCharacterProfileImage(request: CharacterProfileImagePersistRequest): Promise<CharacterProfileImage> {
  // The server writes `<simulationId>/<assetId>.<ext>` and silently skips a segment outside [A-Za-z0-9._-].
  if (!isSafeObjectPathSegment(request.simulationId)) {
    throw new Error("시뮬레이션 id를 저장 경로로 쓸 수 없습니다.");
  }
  const dataUrl = request.asset.dataUrl;
  if (!dataUrl) {
    throw new Error("후보 이미지의 데이터를 찾지 못했습니다. 후보를 다시 생성하세요.");
  }
  const decoded = await decodeProfileCandidate(dataUrl, request.backend);
  const assetId = createId("profile");
  const referenceAssetId = `${assetId}_reference`;
  const character = request.draft.characters.find((candidate) => candidate.id === request.draftCharacterId);
  const objects = [
    createProfileObjectAsset(request, assetId, dataUrl, `${character?.name ?? "캐릭터"} 프로필`),
    ...(decoded.referenceDataUrl
      ? [createProfileObjectAsset(request, referenceAssetId, decoded.referenceDataUrl, `${character?.name ?? "캐릭터"} 프로필 레퍼런스`)]
      : [])
  ];

  let persisted: unknown;
  try {
    persisted = await createDynamicChatApiClient().persistImageAssets(request.simulationId, objects);
  } catch (error) {
    throw new Error(describeApiServerFailure(error));
  }
  // The server answers with the metadata it was sent when it did not write a file, so the key is the only proof.
  const stored = Array.isArray(persisted) ? (persisted as ImageAsset[]) : [];
  const objectKey = stored.find((asset) => asset.id === assetId)?.objectKey;
  const referenceObjectKey = decoded.referenceDataUrl ? stored.find((asset) => asset.id === referenceAssetId)?.objectKey : undefined;
  if (!objectKey || (decoded.referenceDataUrl && !referenceObjectKey)) {
    throw new Error("API 서버가 이미지 파일을 쓰지 않았습니다.");
  }
  return createProfileImageRecord(request.asset, {
    backend: request.backend,
    objectKey,
    referenceObjectKey,
    prompt: request.prompt,
    negativePrompt: request.negativePrompt,
    assetId,
    width: decoded.width,
    height: decoded.height
  });
}

/**
 * Decodes a candidate once for what a pick needs from its pixels: the true size and, for NovelAI, the letterboxed
 * Precise Reference canvas (NovelAI accepts only three fixed canvas sizes, so the copy is made here, once).
 */
async function decodeProfileCandidate(
  dataUrl: string,
  backend: ImageBackendId
): Promise<{ width: number; height: number; referenceDataUrl?: string }> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(dataUrlToBlob(dataUrl));
  } catch {
    throw new Error("후보 이미지를 읽지 못했습니다. 후보를 다시 생성하세요.");
  }
  try {
    const { width, height } = bitmap;
    if (backend !== "novelai") {
      return { width, height };
    }
    const canvas = await letterboxForNovelAiReference(bitmap, width, height);
    return { width, height, referenceDataUrl: `data:image/png;base64,${bytesToBase64(new Uint8Array(await canvas.arrayBuffer()))}` };
  } finally {
    bitmap.close();
  }
}

function dataUrlToBlob(dataUrl: string): Blob {
  const match = /^data:([^;,]+);base64,(.*)$/su.exec(dataUrl);
  if (!match) {
    throw new Error("후보 이미지 형식을 읽지 못했습니다.");
  }
  const binary = atob(match[2]);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: match[1] });
}

/** Only what the object store writes and echoes. Provider metadata stays behind: it can be large and is not kept. */
function createProfileObjectAsset(request: CharacterProfileImagePersistRequest, id: string, dataUrl: string, title: string): ImageAsset {
  return {
    id,
    simulationId: request.simulationId,
    title,
    source: "generated",
    prompt: request.prompt,
    negativePrompt: request.negativePrompt,
    safetyLevel: request.asset.safetyLevel,
    characterIds: [request.draftCharacterId],
    tags: [],
    createdAt: new Date().toISOString(),
    palette: request.asset.palette,
    dataUrl,
    mimeType: dataUrl.slice(5, dataUrl.indexOf(";")),
    backend: request.backend
  };
}

function describeApiServerFailure(error: unknown): string {
  const status = /request failed: (\d{3})/u.exec(error instanceof Error ? error.message : "")?.[1];
  if (status) {
    return `API 서버가 요청을 거부했습니다 (HTTP ${status}).`;
  }
  let baseUrl = DEFAULT_DYNAMICCHAT_API_BASE_URL;
  try {
    baseUrl = getConfiguredDynamicChatApiBaseUrl();
  } catch {
    // Storage-blocked contexts throw on localStorage; the default base is what the client used then too.
  }
  return `DynamicChat API 서버에 연결할 수 없습니다 (${baseUrl}). 프로필 이미지는 API 서버에 저장되므로, 서버를 실행한 뒤 다시 지정하세요.`;
}

/** The same allow-list the server applies to each segment it is about to create. */
function isSafeObjectPathSegment(value: string): boolean {
  return value !== "." && value !== ".." && /^[A-Za-z0-9._-]+$/u.test(value);
}

/**
 * The creation screen's profile-image studio for one character. The creator generates candidates on NovelAI or the
 * photoreal backend and picks one per backend as the character's profile image, which from then on is also that
 * backend's face reference. Candidates exist only in this builder session; a pick is stored as an object-store
 * reference on the draft character, nothing more. There is deliberately no upload: a face reference can only be an
 * image DynamicChat generated, so a real person's photo cannot become one.
 */
function CharacterProfileImagePanel({
  character,
  draft,
  session = EMPTY_PROFILE_STUDIO_SESSION,
  novelAiTokenRegistered,
  onSessionChange,
  onCharacterChange,
  onClearProfileImage,
  onNovelAiChange,
  onPhotorealChange,
  onGenerate,
  onCancel,
  onPick,
  onDiscardCandidates
}: {
  character: SimulationCharacterDraft;
  draft: SimulationDraft;
  session?: ProfileStudioSession;
  novelAiTokenRegistered: boolean;
  onSessionChange: (change: (session: ProfileStudioSession) => ProfileStudioSession) => void;
  onCharacterChange: (patch: Partial<SimulationCharacterDraft>) => void;
  onClearProfileImage: (backend: ImageBackendId) => void;
  onNovelAiChange: (patch: Partial<NovelAiApiSettings>) => void;
  onPhotorealChange: (patch: Partial<PhotorealApiSettings>) => void;
  onGenerate: (input: ProfileStudioGenerateInput) => void;
  onCancel: () => void;
  onPick: (candidate: ProfileStudioCandidate) => void;
  onDiscardCandidates: () => void;
}) {
  const backend: ImageBackendId = session.backend ?? (draft.photoreal.enabled ? "photoreal" : "novelai");
  const followedPrompt =
    backend === "novelai" ? joinPromptParts([character.visualPrompt, character.defaultOutfitPrompt]) : character.photorealPrompt ?? "";
  const typedPrompt = session.prompts[backend];
  const prompt = typedPrompt ?? followedPrompt;
  const negativePrompt = (backend === "novelai" ? character.negativeVisualPrompt : character.photorealNegativePrompt) ?? "";
  const run = session.run;
  const novelAiProfile = character.profileImages?.novelai;
  const photorealProfile = character.profileImages?.photoreal;
  const blocker = describeProfileTestBlocker(backend, draft, novelAiTokenRegistered);
  const photorealBlocker = describeProfileTestBlocker("photoreal", draft, novelAiTokenRegistered);
  const disabledReason = blocker?.message ?? (prompt.trim() ? undefined : "테스트할 외형을 프롬프트에 적어 주세요.");
  const pickingBackends = new Set(session.candidates.filter((candidate) => candidate.persisting).map((candidate) => candidate.backend));
  const referenceSupported = novelAiCharacterReferencePresets.has(draft.novelAi.modelPreset);
  const routingMode = resolveImageBackendRouting(draft.imageProfile).mode;
  const checkPrompt = character.photorealPrompt?.trim() || photorealProfile?.prompt || "";
  const reasonId = `profile-studio-reason-${character.id}`;
  const appearanceScreenId = `profile-studio-appearance-screen-${character.id}`;
  const screeningState = useMemo(
    () => createDraftScreeningState({ imageProfile: draft.imageProfile, novelAi: draft.novelAi, photoreal: draft.photoreal }),
    [draft.imageProfile, draft.novelAi, draft.photoreal]
  );

  // An edit makes the last refusal stale: it was about the request as it stood.
  const setPrompt = (value: string) =>
    onSessionChange((current) => ({ ...current, prompts: { ...current.prompts, [backend]: value }, refusal: undefined }));
  const followCharacterPrompt = () =>
    onSessionChange((current) => {
      const prompts = { ...current.prompts };
      delete prompts[backend];
      return { ...current, prompts, refusal: undefined };
    });
  const renderBlockerAction = (target: ProfileTestBlocker | undefined) =>
    target?.action === "enable_photoreal" ? (
      <button className="icon-text-button" type="button" onClick={() => onPhotorealChange({ enabled: true })}>
        <Check size={14} />
        실사 이미지 켜기
      </button>
    ) : target?.action === "photoreal_proxy" ? (
      <button className="icon-text-button" type="button" onClick={() => onPhotorealChange({ requestMode: "proxy" })}>
        <Check size={14} />
        proxy로 바꾸기
      </button>
    ) : null;

  return (
    <div className="module-detail-panel profile-studio">
      <SectionTitle icon={<UserRound size={17} />} title="캐릭터 프로필 이미지" />
      <p className="runtime-hint">
        후보를 생성해 백엔드마다 하나를 프로필로 지정하면, 그 이미지가 이 캐릭터의 얼굴 기준이 됩니다. 실존 인물의 사진이 얼굴 참조로
        쓰이지 않도록, 프로필은 여기서 생성한 이미지에서만 고를 수 있습니다.
      </p>

      <div className="profile-studio-slots">
        <ProfileImageSlot backend="novelai" characterName={character.name} image={novelAiProfile} onClear={() => onClearProfileImage("novelai")}>
          {!novelAiProfile ? (
            <p className="runtime-hint">
              NovelAI 프로필은 캐릭터 레퍼런스(Precise Reference · V4.5 · 이미지당 +5 Anlas)를 켰을 때 얼굴 참조로 쓰입니다.
            </p>
          ) : !draft.novelAi.enabled ? (
            <p className="runtime-hint">NovelAI 연동이 꺼져 있어 지금은 얼굴 참조로 쓰이지 않습니다.</p>
          ) : draft.novelAi.characterReferenceEnabled !== true ? (
            <>
              <p className="runtime-hint">
                캐릭터 레퍼런스(Precise Reference)가 꺼져 있어 NovelAI 컷에는 얼굴 참조로 쓰이지 않습니다. 켜면 V4.5 모델에서 이미지당 +5
                Anlas가 듭니다.
              </p>
              <button className="icon-text-button" type="button" onClick={() => onNovelAiChange({ characterReferenceEnabled: true })}>
                <Check size={14} />
                캐릭터 레퍼런스 켜기
              </button>
            </>
          ) : referenceSupported ? (
            <p className="runtime-hint">캐릭터 레퍼런스가 켜져 있어 NovelAI 컷마다 얼굴 참조로 쓰입니다 (이미지당 +5 Anlas).</p>
          ) : (
            <p className="settings-note warning">
              현재 모델 {draft.novelAi.modelPreset}은(는) Precise Reference를 지원하지 않아 얼굴 참조로 쓰이지 않습니다. 이미지/연동 탭에서
              NAID4.5F 또는 NAID4.5C를 고르세요.
            </p>
          )}
        </ProfileImageSlot>

        <ProfileImageSlot backend="photoreal" characterName={character.name} image={photorealProfile} onClear={() => onClearProfileImage("photoreal")}>
          {!photorealProfile ? (
            <p className="runtime-hint">실사 프로필은 '얼굴 고정'이 켜져 있으면 실사 컷마다 얼굴 참조로 자동 사용됩니다 (이 PC에서 생성 · 무료).</p>
          ) : !draft.photoreal.enabled ? (
            <p className="runtime-hint">실사 이미지(ComfyUI) 연동이 꺼져 있어 지금은 얼굴 참조로 쓰이지 않습니다.</p>
          ) : draft.photoreal.faceReferenceEnabled ? (
            <p className="runtime-hint">'얼굴 고정'이 켜져 있어 이 캐릭터가 나오는 실사 컷마다 얼굴 참조로 쓰입니다 (이 PC에서 생성 · 무료).</p>
          ) : (
            <>
              <p className="runtime-hint">'얼굴 고정'이 꺼져 있어 실사 컷에는 얼굴 참조로 쓰이지 않습니다.</p>
              <button className="icon-text-button" type="button" onClick={() => onPhotorealChange({ faceReferenceEnabled: true })}>
                <Check size={14} />
                얼굴 고정 켜기
              </button>
            </>
          )}
          {photorealProfile && !character.photorealPrompt?.trim() && /\b(?:a|an|the)\b/iu.test(photorealProfile.prompt) && !HANGUL_TEXT_PATTERN.test(photorealProfile.prompt) ? (
            <>
              <p className="runtime-hint">
                실사 외형이 비어 있어 실사 컷은 외형 태그를 옮겨 씁니다. 이 프로필을 만든 문장을 실사 외형으로 쓰면 얼굴 참조와 외형 설명이
                맞춰집니다.
              </p>
              <button className="icon-text-button" type="button" onClick={() => onCharacterChange({ photorealPrompt: photorealProfile.prompt })}>
                프로필 문장을 실사 외형으로
              </button>
            </>
          ) : null}
          {photorealProfile ? (
            <button
              className="icon-text-button"
              type="button"
              disabled={Boolean(run) || Boolean(photorealBlocker) || !checkPrompt.trim()}
              title={photorealBlocker?.message ?? "프로필 얼굴로 다른 자세를 그려, 얼굴이 유지되는지 확인합니다."}
              onClick={() =>
                onGenerate({
                  backend: "photoreal",
                  prompt: checkPrompt,
                  negativePrompt: character.photorealNegativePrompt ?? "",
                  variant: "reference_check",
                  count: PROFILE_REFERENCE_CHECK_COUNT
                })
              }
            >
              <ScanFace size={14} />
              얼굴 고정 테스트
            </button>
          ) : null}
          {session.refusal?.variant === "reference_check" && !run ? (
            <ProfileStudioRefusalNote refusal={session.refusal} />
          ) : null}
        </ProfileImageSlot>
      </div>

      {run?.variant === "reference_check" || session.checks.length > 0 ? (
        <div className="profile-studio-block">
          <div className="profile-studio-block-head">
            <strong>얼굴 고정 테스트</strong>
          </div>
          <p className="runtime-hint">프로필과 다른 자세로 그린 결과입니다. 같은 사람으로 보이면 얼굴 참조가 잘 작동하고 있습니다.</p>
          {run?.variant === "reference_check" ? <ProfileStudioProgress run={run} onCancel={onCancel} /> : null}
          {session.checks.length > 0 ? (
            <div className="profile-studio-candidates">
              {session.checks.map((candidate, index) => (
                <ProfileStudioCandidateCard candidate={candidate} key={candidate.id} label={`얼굴 고정 테스트 ${index + 1}`} />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="profile-studio-block">
        <label>
          <GuidedFieldLabel
            badge="모든 실사 컷"
            guide="실사 컷에서 이 캐릭터의 생김새를 설명하는 영어 문장입니다. NovelAI 외형 태그 대신 쓰이고, 의상 · 자세 · 배경은 장면마다 따로 정해집니다. 비워 두면 외형 태그를 옮겨 씁니다."
            title="실사 외형"
          />
          <textarea
            rows={3}
            placeholder={PHOTOREAL_APPEARANCE_PLACEHOLDER}
            value={character.photorealPrompt ?? ""}
            aria-describedby={appearanceScreenId}
            onChange={(event) => onCharacterChange({ photorealPrompt: event.target.value })}
          />
        </label>
        {/* Keyed by character: the note reads the text once typing settles, and a switch to another character must
            not show the previous one's verdict for that pause. */}
        <PhotorealTextScreenNote
          id={appearanceScreenId}
          key={character.id}
          scope="appearance"
          state={screeningState}
          text={character.photorealPrompt}
        />
        <label>
          <GuidedFieldLabel guide="실사 컷에서 이 캐릭터에게 피할 요소입니다. 영어로, 쉼표로 구분해 적습니다." title="실사 네거티브" />
          <input
            placeholder="예: glasses, heavy makeup"
            value={character.photorealNegativePrompt ?? ""}
            onChange={(event) => onCharacterChange({ photorealNegativePrompt: event.target.value })}
          />
        </label>
        {HANGUL_TEXT_PATTERN.test(character.photorealPrompt ?? "") ? (
          <p className="settings-note warning">실사 외형은 영어로 적어 주세요. 한국어 문장은 실사 프롬프트에 들어가지 않고, 대신 외형 태그가 쓰입니다.</p>
        ) : null}
      </div>

      <div className="profile-studio-block">
        <div className="profile-studio-block-head">
          <strong>테스트 생성</strong>
        </div>
        <div className="runtime-preset-grid profile-studio-backends" role="group" aria-label="테스트 생성 백엔드">
          {(["novelai", "photoreal"] as const).map((option) => (
            <button
              key={option}
              className={backend === option ? "active" : ""}
              type="button"
              aria-pressed={backend === option}
              onClick={() => onSessionChange((current) => ({ ...current, backend: option, refusal: undefined }))}
            >
              <strong>{imageBackendLabels[option]}</strong>
              <span>{option === "novelai" ? "태그 · 애니메이션 화풍" : "영어 문장 · ComfyUI"}</span>
            </button>
          ))}
        </div>
        <label>
          <GuidedFieldLabel
            guide={
              backend === "novelai"
                ? "후보의 외형 전부입니다. 저장된 외형 · 의상 태그는 따로 덧붙지 않습니다. 처음에는 위 Positive prompt와 기본 의상 태그로 채워집니다."
                : "후보의 외형 전부입니다. 영어 문장은 외형 설명으로, 태그 목록은 태그로 쓰입니다. 처음에는 위 '실사 외형'으로 채워집니다."
            }
            title={backend === "novelai" ? "테스트 프롬프트 (NovelAI 태그)" : "테스트 프롬프트 (영어)"}
          />
          <textarea
            rows={3}
            placeholder={backend === "novelai" ? "예: 1girl, long black hair, brown eyes, white blouse" : PHOTOREAL_APPEARANCE_PLACEHOLDER}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
          />
        </label>
        {typedPrompt !== undefined && typedPrompt !== followedPrompt ? (
          <div className="profile-studio-actions">
            <button className="icon-text-button" type="button" onClick={followCharacterPrompt}>
              <RotateCcw size={14} />
              {backend === "novelai" ? "캐릭터 태그로 되돌리기" : "실사 외형으로 되돌리기"}
            </button>
            {backend === "photoreal" && typedPrompt.trim() ? (
              <button
                className="icon-text-button"
                type="button"
                onClick={() => {
                  onCharacterChange({ photorealPrompt: typedPrompt.trim() });
                  followCharacterPrompt();
                }}
              >
                <Save size={14} />
                실사 외형에 저장
              </button>
            ) : null}
          </div>
        ) : null}
        {backend === "photoreal" && HANGUL_TEXT_PATTERN.test(prompt) ? (
          <p className="settings-note warning">실사 프롬프트는 영어로 적어 주세요. 한국어는 실사 프롬프트에 들어가지 않습니다.</p>
        ) : null}
        <p className="runtime-hint">
          {backend === "novelai" ? "네거티브로 위 'Negative prompt' 칸을 함께 보냅니다." : "네거티브로 '실사 네거티브' 칸을 함께 보냅니다."}{" "}
          {backend === "novelai"
            ? "NovelAI 테스트는 후보마다 일반 생성과 같은 Anlas를 씁니다."
            : "실사 테스트는 이 PC의 ComfyUI에서 생성되어 비용이 들지 않습니다. 첫 생성은 모델을 불러오느라 더 걸립니다."}
        </p>
        {/* Same availability rule as decideImageBackend: 'NAI만' keeps story cuts off 실사 only while NovelAI is on. With
            NovelAI off and 실사 on, every cut the floor does not lock falls back to 실사 — the seed's local-only setup —
            so saying "실사가 쓰이지 않습니다" there was the opposite of what happens. */}
        {backend === "photoreal" && routingMode === "novelai" && draft.novelAi.enabled ? (
          <p className="runtime-hint">
            지금 이미지 라우팅이 'NAI만'이라 이야기 컷에는 실사가 쓰이지 않습니다. 이미지/연동 탭의 '실사 라우팅'에서 바꿀 수 있습니다.
          </p>
        ) : backend === "photoreal" && routingMode === "novelai" && draft.photoreal.enabled ? (
          <p className="runtime-hint">
            이미지 라우팅은 'NAI만'이지만 NovelAI가 꺼져 있어 이야기 컷은 실사로 생성됩니다. 미성년으로 읽히는 인물(학생·교복 포함)이
            성적·노출 장면에 있는 컷만 예외입니다.
          </p>
        ) : null}
        <div className="profile-studio-actions">
          <label className="profile-studio-count">
            후보 수
            <select disabled={Boolean(run)} value={session.count} onChange={(event) => onSessionChange((current) => ({ ...current, count: Number(event.target.value) }))}>
              {[1, 2, 3, 4].map((count) => (
                <option key={count} value={count}>
                  {count}장
                </option>
              ))}
            </select>
          </label>
          <button
            className="send-button"
            type="button"
            disabled={Boolean(run) || Boolean(disabledReason)}
            aria-describedby={disabledReason ? reasonId : undefined}
            onClick={() => onGenerate({ backend, prompt, negativePrompt, variant: "portrait", count: session.count })}
          >
            <Sparkles size={16} />
            테스트 생성
          </button>
        </div>
        {disabledReason && !run ? (
          <div className="profile-studio-blocker">
            <p className="settings-note" id={reasonId}>
              {disabledReason}
            </p>
            {renderBlockerAction(blocker)}
          </div>
        ) : null}
        {run && run.variant === "portrait" ? <ProfileStudioProgress run={run} onCancel={onCancel} /> : null}
        {session.notice ? (
          <p className="runtime-hint" role="status">
            {session.notice}
          </p>
        ) : null}
        {session.refusal && session.refusal.variant !== "reference_check" && !run ? (
          <ProfileStudioRefusalNote refusal={session.refusal} />
        ) : null}
        {session.errors.length > 0 ? (
          <div className="settings-note warning profile-studio-errors" role="alert">
            <strong>테스트 생성 실패</strong>
            <ul>
              {session.errors.map((message) => (
                <li key={message}>{message}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>

      {session.candidates.length > 0 ? (
        <div className="profile-studio-block">
          <div className="profile-studio-block-head">
            <strong>후보 {session.candidates.length}장</strong>
            <button className="icon-text-button" type="button" disabled={Boolean(run)} onClick={onDiscardCandidates}>
              <Trash2 size={14} />
              후보 비우기
            </button>
          </div>
          <p className="runtime-hint">후보는 이 제작 화면에만 있고 저장되지 않습니다. 프로필로 지정한 이미지만 저장됩니다.</p>
          <div className="profile-studio-candidates">
            {session.candidates.map((candidate, index) => (
              <ProfileStudioCandidateCard
                candidate={candidate}
                isProfile={Boolean(candidate.profileAssetId) && character.profileImages?.[candidate.backend]?.assetId === candidate.profileAssetId}
                key={candidate.id}
                label={`후보 ${index + 1}`}
                pickDisabled={candidate.persisting === true || pickingBackends.has(candidate.backend)}
                onPick={() => onPick(candidate)}
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

/** One backend's current profile image, with what it is used for underneath. */
function ProfileImageSlot({
  backend,
  characterName,
  image,
  children,
  onClear
}: {
  backend: ImageBackendId;
  characterName: string;
  image?: CharacterProfileImage;
  children?: ReactNode;
  onClear: () => void;
}) {
  const label = `${imageBackendLabels[backend]} 프로필`;
  return (
    <section className={`profile-studio-slot${image ? "" : " empty"}`} aria-label={label}>
      <div className="profile-studio-block-head">
        <strong>{label}</strong>
        {image ? (
          <button className="icon-text-button" type="button" aria-label={`${label} 해제`} onClick={onClear}>
            <X size={14} />
            해제
          </button>
        ) : (
          <span>미지정</span>
        )}
      </div>
      <div className="profile-studio-slot-body">
        <ProfileImageThumb alt={image ? `${characterName} ${label} 이미지` : ""} className="profile-studio-slot-thumb" image={image} />
        <div className="profile-studio-slot-notes">
          {image ? <small>{describeProfileImageMeta(image)}</small> : null}
          {children}
        </div>
      </div>
    </section>
  );
}

/**
 * A test the planner refused before anything ran — the photoreal floor (a figure read as a minor in a sexual or
 * exposing scene), most often. Worded as "not started" rather than as a failure: nothing was sent and nothing was
 * spent, and what has to change is the request.
 */
function ProfileStudioRefusalNote({ refusal }: { refusal: ProfileStudioRefusal }) {
  const subject = refusal.variant === "reference_check" ? "얼굴 고정 테스트" : `${imageBackendLabels[refusal.backend]} 테스트`;
  return (
    <div className="settings-note warning profile-studio-errors profile-studio-refusal" role="alert">
      <strong>{subject}를 시작하지 않았습니다</strong>
      <p>{refusal.message}</p>
    </div>
  );
}

/**
 * What the photoreal floor will do with free text a creator types for the photo side, worded for a note under the
 * field, or undefined when the text passes. The planner reads the same text on every cut — a character's 실사 외형
 * with each cut that character is in, the style text with every cut — through the same screen, and a text the floor
 * refuses sends those cuts to NovelAI instead (or leaves them unrendered while NovelAI is off). The floor is the one
 * line routing does not leave to the creator: a figure read as a minor (students and school uniforms included) in a
 * sexual or exposing scene. Adult nudity and sexual wording pass here — where those cuts go is the routing's call.
 * Nothing is blocked or rewritten here: the note only says, while the creator can still reword it, what routing will
 * do anyway.
 */
function describePhotorealTextLock(state: AppState, text: string | undefined, scope: "appearance" | "style"): string | undefined {
  const screen = screenPhotorealText(state, text);
  if (!screen) {
    return undefined;
  }
  // The reading the router decided on (for a text read alone, the text's own).
  const classification = screen.reading;
  // A sexual scene written in letters no lexicon can read (kana, Chinese, Cyrillic look-alikes) is held back because
  // the age in it cannot be checked, not because a minor was found, and the way out is different: the same
  // description in English.
  const unreadableOnly = classification.unverifiable === true && !classification.minorCoded && !classification.schoolCoded;
  const subject = scope === "appearance" ? "이 실사 외형" : "이 스타일 문장";
  const consequence = scope === "appearance" ? "이 캐릭터가 나오는 컷은 실사 대신 NovelAI로 생성됩니다." : "실사로 갈 컷이 모두 NovelAI로 넘어갑니다.";
  return [
    unreadableOnly
      ? `${subject}에는 확인할 수 없는 문자(영문 외 문자)가 성적·노출 표현과 함께 있어 나이를 확인할 수 없으므로, ${consequence} 영어로 적어 주세요.`
      : `${subject}은 미성년으로 읽히는 인물(학생·교복 포함)이 성적·노출 장면에 있는 것으로 읽혀, ${consequence} 이 경우만은 설정과 관계없이 실사로 생성하지 않습니다.`,
    state.novelAi.enabled ? undefined : "지금은 NovelAI가 꺼져 있어 그 컷은 생성되지 않습니다."
  ]
    .filter(Boolean)
    .join(" ");
}

/**
 * Enough of an AppState for screenPhotorealText to judge builder text: the router reads the draft's routing and which
 * renderers it turns on, and nothing else. A full createDraftPreviewState would rebuild the whole simulation on every
 * keystroke to answer a question about one sentence.
 */
function createDraftScreeningState(draft: Pick<SimulationDraft, "imageProfile" | "novelAi" | "photoreal">): AppState {
  return { ...seedState, imageProfile: draft.imageProfile, novelAi: draft.novelAi, photoreal: draft.photoreal };
}

/**
 * The photoreal floor's verdict on a photo-side text field, shown under that field (describePhotorealTextLock). Read
 * once typing pauses, so it does not flicker in and out mid-word. The region is always mounted and polite: a screen
 * reader hears the note when it appears or changes, and the field points at it, without an interruption per key.
 */
function PhotorealTextScreenNote({
  id,
  state,
  text,
  scope
}: {
  id: string;
  state: AppState;
  text: string | undefined;
  scope: "appearance" | "style";
}) {
  const [settledText] = useSettledValue(text ?? "", PHOTOREAL_TEXT_SCREEN_DELAY_MS);
  const message = describePhotorealTextLock(state, settledText, scope);
  return (
    <div className="photoreal-text-screen" id={id} role="status" aria-live="polite">
      {message ? <p className="settings-note warning">{message}</p> : null}
    </div>
  );
}

function ProfileStudioProgress({ run, onCancel }: { run: ProfileStudioRun; onCancel: () => void }) {
  return (
    <div className="profile-studio-progress">
      <span role="status" aria-live="polite">
        {describeProfileStudioRun(run)}
      </span>
      <button className="icon-text-button" type="button" onClick={onCancel}>
        <Square size={13} />
        취소
      </button>
    </div>
  );
}

function ProfileStudioCandidateCard({
  candidate,
  label,
  isProfile = false,
  pickDisabled = false,
  onPick
}: {
  candidate: ProfileStudioCandidate;
  label: string;
  isProfile?: boolean;
  pickDisabled?: boolean;
  /** Absent for face-check images: they test the profile, they are not candidates for it. */
  onPick?: () => void;
}) {
  const seed = readProviderPayloadNumber(candidate.asset.providerMetadata ?? {}, "seed");
  return (
    <figure className={`profile-studio-candidate${isProfile ? " is-profile" : ""}`}>
      <div className="profile-studio-candidate-image">
        <img alt={label} src={candidate.asset.dataUrl} />
        <span className="image-backend-badge">{imageBackendBadgeLabels[candidate.backend]}</span>
      </div>
      <figcaption>
        <small>{seed !== undefined ? `${label} · seed ${seed}` : label}</small>
        {onPick ? (
          isProfile ? (
            <span className="profile-studio-picked">
              <Check size={13} />
              프로필로 지정됨
            </span>
          ) : (
            <button className="icon-text-button" type="button" disabled={pickDisabled} onClick={onPick}>
              {candidate.persisting ? "저장 중…" : `${imageBackendLabels[candidate.backend]} 프로필로 지정`}
            </button>
          )
        ) : null}
        {candidate.error ? (
          <p className="settings-note warning" role="alert">
            {candidate.error}
          </p>
        ) : null}
      </figcaption>
    </figure>
  );
}

/** A stored profile image from the object store; a neutral glyph when there is none or it will not load. */
function ProfileImageThumb({ image, alt = "", className }: { image?: CharacterProfileImage; alt?: string; className?: string }) {
  const src = image?.objectKey ? createObjectStoreImageSrc(image.objectKey) : undefined;
  const [failedSrc, setFailedSrc] = useState<string>();
  const classes = ["profile-thumb", className].filter(Boolean).join(" ");
  if (!src || failedSrc === src) {
    return (
      <span className={`${classes} empty`} aria-hidden="true">
        <UserRound size={18} />
      </span>
    );
  }
  return (
    <span className={classes}>
      <img alt={alt} loading="lazy" src={src} onError={() => setFailedSrc(src)} />
    </span>
  );
}

function ImageStyleProfilePanel({
  module,
  onModuleChange
}: {
  module: PromptModule;
  onModuleChange: (patch: Partial<PromptModule>) => void;
}) {
  return (
    <div className="module-detail-panel">
      <SectionTitle icon={<Sparkles size={17} />} title="이미지 스타일 프로필" />
      <label>
        <GuidedFieldLabel badge={moduleKindGuides.image_prompt_profile.policy} guide={moduleKindGuides.image_prompt_profile.detail} title="NAI positive tags" />
        <textarea placeholder={moduleKindGuides.image_prompt_profile.placeholder} value={module.body} onChange={(event) => onModuleChange({ body: event.target.value })} />
      </label>
    </div>
  );
}

function ModuleEditor({
  module,
  onChange,
  onDelete
}: {
  module: PromptModule;
  onChange: (moduleId: string, patch: Partial<PromptModule>) => void;
  onDelete: (moduleId: string) => void;
}) {
  const moduleGuide = moduleKindGuides[module.kind];

  return (
    <div className="module-editor">
      <div className="editor-title">
        <Layers size={17} />
        <strong>모듈 편집</strong>
      </div>
      <div className={`module-meaning-note ${module.kind}`}>
        <div>
          <span>{moduleGuide.policy}</span>
          <strong>{moduleGuide.title}</strong>
        </div>
        <GuideIcon label={`${moduleGuide.title} 설명`} detail={moduleGuide.detail} />
      </div>
      <label>
        <GuidedFieldLabel
          guide="프롬프트 트리와 검색 근거에 표시되는 이름입니다. 모듈의 역할과 장면 조건이 드러나도록 짧게 적으면 좋습니다."
          title="제목"
        />
        <input value={module.title} onChange={(event) => onChange(module.id, { title: event.target.value })} />
      </label>
      <div className="two-fields">
        <label>
          <GuidedFieldLabel
            guide="모듈의 의미를 정합니다. 타입에 따라 프롬프트 트리에서 해석되는 역할과 기본 참조 방식이 달라집니다."
            title="타입"
          />
          <select
            value={module.kind}
            onChange={(event) => {
              const kind = event.target.value as PromptModuleKind;
              onChange(module.id, {
                kind,
                characterId: kind === "character_prompt" ? module.characterId : undefined,
                title: kind === "image_prompt_profile" && module.title === "캐릭터 이미지 매핑" ? "이미지 스타일 프로필" : module.title,
                tokenPolicy: kind === "image_prompt_profile" ? "always" : module.tokenPolicy
              });
            }}
          >
            {promptModuleKinds.map((kind) => (
              <option key={kind} value={kind}>
                {moduleKindGuides[kind].title}
              </option>
            ))}
          </select>
        </label>
        <label>
          <GuidedFieldLabel
            guide="항상 포함은 매 턴 넣고, 필요할 때 참조는 현재 장면과 맞을 때만 자동으로 고릅니다. 수동 참조는 제작자가 직접 쓰고, 사용 안 함은 진행에서 제외합니다."
            title="참조 방식"
          />
          <select value={module.tokenPolicy} onChange={(event) => onChange(module.id, { tokenPolicy: event.target.value as TokenPolicy })}>
            {tokenPolicies.map((policy) => (
              <option key={policy} value={policy}>
                {tokenPolicyLabels[policy]}
              </option>
            ))}
          </select>
        </label>
      </div>
      <label>
        <GuidedFieldLabel
          guide="이 모듈을 불러올 단서입니다. 장면, 장소, 캐릭터, 시스템 이름처럼 사용자 입력이나 기억 근거와 맞을 짧은 키워드를 쉼표로 입력하세요."
          title="호출 키워드"
        />
        <ActivationTagInput
          tags={module.activationTags}
          onCommit={(activationTags) =>
            onChange(module.id, {
              activationTags
            })
          }
        />
      </label>
      <label>
        <GuidedFieldLabel
          guide="같은 조건에서 여러 모듈이 선택될 때의 중요도입니다. 핵심 규칙과 제작자 지정 경계는 높게, 배경 자료와 보조 규칙은 낮게 두는 편이 안정적입니다."
          title="우선순위"
        />
        <input
          min="0"
          max="100"
          type="range"
          value={module.priority}
          onChange={(event) => onChange(module.id, { priority: Number(event.target.value) })}
        />
      </label>
      <label className="checkline">
        <input checked={module.enabled} type="checkbox" onChange={(event) => onChange(module.id, { enabled: event.target.checked })} />
        활성화
      </label>
      {module.kind === "character_prompt" || module.kind === "image_prompt_profile" ? null : (
        <label>
          <GuidedFieldLabel
            badge={moduleGuide.policy}
            guide={moduleGuide.detail}
            title="본문"
            charCount={<PromptCharCount value={module.body} limit={promptModuleBodyCharLimit(module.kind, module.tokenPolicy)} />}
          />
          <textarea
            placeholder={moduleGuide.placeholder}
            value={module.body}
            onChange={(event) => onChange(module.id, { body: event.target.value })}
          />
        </label>
      )}
      <div className="editor-actions">
        <span>v{module.version}</span>
        <button className="danger-button" type="button" onClick={() => onDelete(module.id)} disabled={module.kind === "main_prompt"}>
          <Trash2 size={16} />
          삭제
        </button>
      </div>
    </div>
  );
}

/** One sentence saying where the next cuts will actually go, availability included. */
function describeImageBackendRoutingSummary(
  routing: ImageBackendRouting,
  availability: { novelAiEnabled: boolean; photorealEnabled: boolean }
): string {
  const tickedLabels = routing.photorealCategories
    .map((category) => getImageSceneCategoryLabel(category))
    .filter((label): label is string => Boolean(label));
  const routesToPhotoreal = routing.mode === "photoreal" || (routing.mode === "mixed" && tickedLabels.length > 0);
  const base =
    routing.mode === "novelai"
      ? "모든 컷을 NovelAI로 생성합니다."
      : routing.mode === "photoreal"
        ? "모든 컷을 실사로 생성합니다."
        : tickedLabels.length === 0
          ? "실사로 보낼 장면을 고르지 않아 모든 컷을 NovelAI로 생성합니다."
          : `${tickedLabels.join(", ")} 장면은 실사로, 나머지는 NovelAI로 생성합니다.`;
  // The one rule no tick changes is said whenever routing sends anything to 실사, in the router's own words.
  const routed = routesToPhotoreal ? `${base} ${PHOTOREAL_FLOOR_DESCRIPTION}.` : base;
  if (routesToPhotoreal && !availability.photorealEnabled) {
    return `${routed} ${availability.novelAiEnabled ? "지금은 실사(ComfyUI) 연동이 꺼져 있어 NovelAI로 대신 생성합니다." : "실사(ComfyUI) 연동을 켜야 생성됩니다."}`;
  }
  if (!availability.novelAiEnabled && availability.photorealEnabled && routing.mode !== "photoreal") {
    // Everything now reaches 실사 but the floor's cuts, so the floor is said once, as the reason those wait.
    return `${base} 지금은 NovelAI가 꺼져 있어 NovelAI로 갈 컷도 실사로 대신 생성합니다. 미성년으로 읽히는 인물(학생·교복 포함)이 성적·노출 장면에 있는 컷만 실사로 생성하지 않아 NovelAI를 켤 때까지 기다립니다.`;
  }
  return routed;
}

/**
 * Per-cut renderer routing: [NAI만 | 장면별 혼합 | 전부 실사] and the scene categories that go photoreal in mixed
 * mode (multi-select). One component for all three places it appears — settings, the image panel, the builder —
 * so the summary and the category list cannot drift apart between them.
 *
 * Every category is the creator's choice, the sexual-scene one included: what the photo model draws is theirs to
 * decide, and the defaults (people, scenery, objects) are only a starting point. The floor that no tick changes is
 * stated in the summary, not as a row. An option the router marks `locked` still renders as a disabled, unticked row
 * and is never written into the stored list.
 */
function ImageBackendRoutingCard({
  routing,
  novelAiEnabled,
  photorealEnabled,
  compact = false,
  onChange
}: {
  routing: ImageBackendRouting;
  novelAiEnabled: boolean;
  photorealEnabled: boolean;
  /** The image panel's variant: the category list only while it matters (mixed), without descriptions. */
  compact?: boolean;
  onChange: (next: ImageBackendRouting, message: string) => void;
}) {
  const activeMode =
    imageBackendRoutingModeOptions.find((option) => option.value === routing.mode) ?? imageBackendRoutingModeOptions[0];
  const ticked = new Set(routing.photorealCategories);
  const showCategories = !compact || routing.mode === "mixed";
  const toggleCategory = (category: ImageSceneCategory, checked: boolean) => {
    const wanted = new Set(routing.photorealCategories);
    if (checked) {
      wanted.add(category);
    } else {
      wanted.delete(category);
    }
    const label = getImageSceneCategoryLabel(category) ?? category;
    onChange(
      // Canonical order, so the stored list (and the summary that reads it) does not depend on click order.
      {
        ...routing,
        photorealCategories: IMAGE_SCENE_CATEGORY_OPTIONS.filter((option) => option.locked !== true && wanted.has(option.id)).map(
          (option) => option.id
        )
      },
      checked ? `'${label}' 장면을 실사로 보냅니다.` : `'${label}' 장면을 NovelAI로 보냅니다.`
    );
  };

  return (
    <article className={`runtime-parameter-card backend-routing-card${compact ? " compact" : ""}`}>
      <div className="runtime-card-subhead">
        <strong>실사 라우팅</strong>
        <span>{activeMode.label}</span>
      </div>
      <div className="runtime-preset-grid backend-routing-modes" role="group" aria-label="이미지 백엔드 라우팅 모드">
        {imageBackendRoutingModeOptions.map((option) => (
          <button
            key={option.value}
            className={routing.mode === option.value ? "active" : ""}
            type="button"
            aria-pressed={routing.mode === option.value}
            onClick={() => onChange({ ...routing, mode: option.value }, `이미지 라우팅을 '${option.label}' 모드로 바꿨습니다.`)}
          >
            <strong>{option.label}</strong>
            <span>{option.detail}</span>
          </button>
        ))}
      </div>
      {showCategories ? (
        <fieldset className="backend-routing-categories">
          <legend>
            실사로 보낼 장면
            {routing.mode !== "mixed" ? <small> · 장면별 혼합 모드에서 적용</small> : null}
          </legend>
          {IMAGE_SCENE_CATEGORY_OPTIONS.map((option) => {
            const locked = option.locked === true;
            return (
              // The compact list drops descriptions for room; the tooltip keeps them one hover away (the action row's
              // "실사 모델은 큰 동작에 약하다" is exactly what a creator ticking it there should still be able to read).
              <label
                className={`backend-routing-category${locked ? " locked" : ""}`}
                key={option.id}
                title={compact && !locked ? option.description : undefined}
              >
                <input
                  type="checkbox"
                  checked={!locked && ticked.has(option.id)}
                  disabled={locked}
                  onChange={(event) => toggleCategory(option.id, event.target.checked)}
                />
                <span>
                  <strong>{option.label}</strong>
                  {!compact || locked ? <small>{option.description}</small> : null}
                </span>
              </label>
            );
          })}
        </fieldset>
      ) : null}
      <p className="settings-note backend-routing-summary">{describeImageBackendRoutingSummary(routing, { novelAiEnabled, photorealEnabled })}</p>
    </article>
  );
}

function ImagePromptBlock({ label, value, muted = false }: { label: string; value: string; muted?: boolean }) {
  const displayValue = value.trim() || (muted ? "없음" : "프롬프트 없음");

  return (
    <details className={`image-prompt-block${muted ? " muted" : ""}`} open>
      <summary>
        <span>{label}</span>
        <small>{displayValue.length.toLocaleString("ko-KR")}자</small>
      </summary>
      <pre>{displayValue}</pre>
    </details>
  );
}

function ImagePanel({
  state,
  assets,
  jobs,
  onCancelJob,
  onDeleteAsset,
  onImageProfileChange,
  onRegenerateJob,
  onRunJob,
  onFeedback
}: {
  state: AppState;
  assets: ImageAsset[];
  jobs: AppState["imageJobs"];
  onCancelJob: (jobId: string) => void;
  onDeleteAsset: (assetId: string) => void;
  onImageProfileChange: (patch: Partial<ImageGenerationProfile>) => void;
  /** `backend` is set only by "다른 백엔드로 재생성"; a plain regeneration keeps the cut's own renderer. */
  onRegenerateJob: (job: ImageGenerationJob, backend?: ImageBackendId) => void;
  onRunJob: (job: ImageGenerationJob) => void;
  onFeedback: (assetId: string, rating: ImageFeedbackRating) => void;
}) {
  const [selectedAssetId, setSelectedAssetId] = useState<string>();
  const [selectedJobId, setSelectedJobId] = useState<string>();
  const primaryAsset = assets[0];
  const latestJob = jobs.at(-1);
  const assetById = useMemo(() => new Map(assets.map((asset) => [asset.id, asset])), [assets]);
  const jobByAssetId = useMemo(() => {
    const mappedJobs = new Map<string, ImageGenerationJob>();
    jobs.forEach((job) => {
      if (job.representativeAssetId) {
        mappedJobs.set(job.representativeAssetId, job);
      }
      job.assetIds.forEach((assetId) => mappedJobs.set(assetId, job));
    });
    return mappedJobs;
  }, [jobs]);
  const generatedAssets = useMemo(() => assets.filter((asset) => asset.source === "generated"), [assets]);
  const inspectedJob = (selectedJobId ? jobs.find((job) => job.id === selectedJobId) : undefined) ?? latestJob;
  const resolveJobAsset = useCallback(
    (job?: ImageGenerationJob) => {
      if (!job) {
        return undefined;
      }
      return (job.representativeAssetId ? assetById.get(job.representativeAssetId) : undefined) ?? (job.assetIds[0] ? assetById.get(job.assetIds[0]) : undefined);
    },
    [assetById]
  );
  const inspectedJobAsset = resolveJobAsset(inspectedJob);
  const selectedAsset = (selectedAssetId ? assetById.get(selectedAssetId) : undefined) ?? inspectedJobAsset ?? primaryAsset;
  const inspectedCue = inspectedJob ? getImageJobCue(inspectedJob) : undefined;
  const jobEvidence = inspectedJob ? getJobEvidence(state, inspectedJob) : [];
  const promptSeed = inspectedJob ? readProviderPayloadNumber(inspectedJob.providerPayload, "seed") : undefined;
  const promptModel = inspectedJob ? readProviderPayloadText(inspectedJob.providerPayload, "model") : undefined;
  const promptSampler = inspectedJob ? readProviderPayloadText(inspectedJob.providerPayload, "sampler") : undefined;
  const promptSchedule = inspectedJob ? readProviderPayloadText(inspectedJob.providerPayload, "noise_schedule") : undefined;
  const promptWidth = inspectedJob ? readProviderPayloadNumber(inspectedJob.providerPayload, "width") : undefined;
  const promptHeight = inspectedJob ? readProviderPayloadNumber(inspectedJob.providerPayload, "height") : undefined;
  const promptSteps = inspectedJob ? readProviderPayloadNumber(inspectedJob.providerPayload, "steps") : undefined;
  const promptScale = inspectedJob
    ? readProviderPayloadNumber(inspectedJob.providerPayload, "scale") ?? readProviderPayloadNumber(inspectedJob.providerPayload, "promptGuidance")
    : undefined;
  const inspectedBackend = inspectedJob ? readImageJobBackend(inspectedJob) : undefined;
  const isPhotorealJob = inspectedBackend === "photoreal";
  // The NAI caption block describes a NovelAI request; on a photoreal cut it would be a request that never existed.
  const novelAiV4CaptionDebug = inspectedJob && !isPhotorealJob ? readNovelAiV4CaptionDebug(inspectedJob.providerPayload) : undefined;
  const inspectedRouting = inspectedJob?.routing;
  const inspectedCategoryLabel = getImageSceneCategoryLabel(inspectedRouting?.category);
  // A stored lock is the verdict of the day the cut was planned, possibly under an older rule (sexual cuts were once
  // NovelAI-only in every mode). So the "other backend" button asks today's router instead — the same forced re-plan
  // the click itself runs — and is offered exactly when that click would draw the cut photoreal. Only a locked cut
  // pays for the re-plan.
  const currentPhotorealRouting = useMemo(() => {
    if (!inspectedJob || inspectedJob.routing?.locked !== true) {
      return undefined;
    }
    try {
      const regenerationCue = createImageCueForRegeneration(state, inspectedJob);
      return planImageJob(state, inspectedJob.turnId, regenerationCue, inspectedJob.contextNodeIds, true, {
        backend: "photoreal",
        cueKind: regenerationCue.kind
      }).routing;
    } catch {
      return inspectedJob.routing;
    }
  }, [inspectedJob, state]);
  const inspectedLockHolds = inspectedRouting?.locked === true && currentPhotorealRouting?.backend !== "photoreal";
  const inspectedLockStale = inspectedRouting?.locked === true && !inspectedLockHolds;
  // The stored reason is a record of the plan, so it stays — but marked as that day's wording whenever today's router
  // says something else (a lock that no longer holds, or an older phrasing of one that does). Router reasons are
  // "<verdict> · <note> · …", so the verdicts are compared and the notes (direct request, NovelAI off) are not.
  const leadingReason = (reason: string | undefined) => reason?.split(" · ")[0]?.trim() ?? "";
  const inspectedReasonOutdated =
    inspectedRouting?.locked === true &&
    currentPhotorealRouting !== undefined &&
    leadingReason(currentPhotorealRouting.reason) !== leadingReason(inspectedRouting.reason);
  const inspectedLockReason = currentPhotorealRouting?.reason || inspectedRouting?.reason || `${PHOTOREAL_FLOOR_DESCRIPTION}.`;
  const inspectedPresetId = inspectedJob ? readProviderPayloadText(inspectedJob.providerPayload, "presetId") : undefined;
  const inspectedPreset =
    inspectedPresetId && Object.prototype.hasOwnProperty.call(PHOTOREAL_WORKFLOW_PRESETS, inspectedPresetId)
      ? PHOTOREAL_WORKFLOW_PRESETS[inspectedPresetId as PhotorealWorkflowPresetId]
      : undefined;
  // A photoreal cut's numbers come from its own payload, else its preset and the photoreal size resolver — never the
  // NovelAI profile's sampler/steps/size, which describe a different model entirely.
  const photorealFallbackSize = isPhotorealJob ? resolvePhotorealOutputSize(state) : undefined;
  const displayWidth = promptWidth ?? photorealFallbackSize?.width ?? state.imageProfile.width;
  const displayHeight = promptHeight ?? photorealFallbackSize?.height ?? state.imageProfile.height;
  // A fixed-schedule preset (Qwen-Image 2.1 Turbo) samples its own step count whatever the job recorded: showing the
  // 스텝 override there would describe a render that never happened.
  const fixedPhotorealSteps = isPhotorealJob ? readFixedPhotorealSteps(inspectedPresetId) : undefined;
  const displaySteps = promptSteps ?? (isPhotorealJob ? inspectedPreset?.defaults.steps : state.imageProfile.steps);
  const displayStepsLabel =
    fixedPhotorealSteps !== undefined
      ? `${fixedPhotorealSteps} (Turbo 고정)`
      : displaySteps !== undefined
        ? displaySteps.toString()
        : "기본값";
  const displayCfg =
    (isPhotorealJob ? readProviderPayloadNumber(inspectedJob?.providerPayload ?? {}, "cfg") : promptScale) ??
    (isPhotorealJob ? inspectedPreset?.defaults.cfg : state.imageProfile.promptGuidance);
  // A Turbo preset samples at its fixed CFG whatever the job recorded (a job planned before the pin may say 4).
  const fixedPhotorealCfg = isPhotorealJob ? inspectedPreset?.fixedCfg : undefined;
  const displayCfgLabel =
    fixedPhotorealCfg !== undefined ? `${fixedPhotorealCfg} (Turbo 고정)` : displayCfg !== undefined ? displayCfg.toString() : "기본값";
  const photorealSourceTags = inspectedJob && isPhotorealJob ? readProviderPayloadText(inspectedJob.providerPayload, "novelAiPrompt") : undefined;
  const photorealReferenceNames = inspectedJob && isPhotorealJob ? readPhotorealReferenceCharacterNames(state, inspectedJob.providerPayload) : [];
  const otherBackend: ImageBackendId | undefined = inspectedBackend ? (inspectedBackend === "photoreal" ? "novelai" : "photoreal") : undefined;
  // Regenerating on a renderer that is switched off cannot produce a picture: the request goes out confirmed and
  // fails at once, and turning the renderer on later does not bring it back. So the button waits for the renderer
  // instead of the copy promising that the cut will.
  const otherBackendOff =
    otherBackend === "photoreal" ? state.photoreal?.enabled !== true : otherBackend === "novelai" ? !state.novelAi.enabled : false;
  const isInspectingLatestJob = Boolean(inspectedJob && latestJob && inspectedJob.id === latestJob.id);
  const activeCadenceOption =
    imageGenerationCadenceOptions.find((option) => option.value === state.imageProfile.generationCadence) ??
    imageGenerationCadenceOptions[1];

  useEffect(() => {
    if (selectedAssetId && !assetById.has(selectedAssetId)) {
      setSelectedAssetId(undefined);
    }
  }, [assetById, selectedAssetId]);

  useEffect(() => {
    if (selectedJobId && !jobs.some((job) => job.id === selectedJobId)) {
      setSelectedJobId(undefined);
    }
  }, [jobs, selectedJobId]);

  const inspectAsset = (asset: ImageAsset) => {
    setSelectedAssetId(asset.id);
    const assetJob = jobByAssetId.get(asset.id);
    if (assetJob) {
      setSelectedJobId(assetJob.id);
    }
  };

  const inspectJob = (job: ImageGenerationJob) => {
    setSelectedJobId(job.id);
    const jobAsset = resolveJobAsset(job);
    setSelectedAssetId(jobAsset?.id);
  };

  return (
    <div className="panel-stack image-inspector-panel">
      <div className="panel-header compact">
        <div>
          <span className="section-kicker">이미지 생성</span>
          <h2>작업 인스펙터</h2>
        </div>
        <StatusPill icon={<Sparkles size={15} />} label={`작업 ${jobs.length}개`} tone="neutral" />
      </div>

      <article className="runtime-parameter-card image-cadence-card">
        <div className="runtime-card-subhead">
          <strong>생성 밀도</strong>
          <span>{activeCadenceOption.label}</span>
        </div>
        <div className="runtime-preset-grid" role="group" aria-label="이미지 생성 밀도">
          {imageGenerationCadenceOptions.map((option) => (
            <button
              key={option.value}
              className={state.imageProfile.generationCadence === option.value ? "active" : ""}
              type="button"
              onClick={() => onImageProfileChange({ generationCadence: option.value })}
            >
              <strong>{option.label}</strong>
              <span>{option.detail}</span>
            </button>
          ))}
        </div>
      </article>

      <ImageBackendRoutingCard
        compact
        routing={resolveImageBackendRouting(state.imageProfile)}
        novelAiEnabled={state.novelAi.enabled}
        photorealEnabled={state.photoreal?.enabled === true}
        onChange={(backendRouting) => onImageProfileChange({ backendRouting })}
      />

      {inspectedJob ? (
        <article className="image-job-detail">
          <div className="image-job-detail-header">
            <div>
              <span>{isInspectingLatestJob ? "최근 작업" : "선택 작업"}</span>
              <strong>{getImageJobStatusLabel(inspectedJob.status)}</strong>
            </div>
            <div className="job-actions">
              {canUserRunQueuedImageJob(inspectedJob) ? (
                <button className="icon-button" type="button" onClick={() => onRunJob(inspectedJob)} aria-label="이미지 작업 실행">
                  <Play size={15} />
                </button>
              ) : null}
              {["queued", "planning", "generating"].includes(inspectedJob.status) ? (
                <button className="icon-button" type="button" onClick={() => onCancelJob(inspectedJob.id)} aria-label="이미지 작업 취소">
                  <Trash2 size={15} />
                </button>
              ) : null}
              {["completed", "failed", "canceled"].includes(inspectedJob.status) ? (
                <button className="icon-button" type="button" onClick={() => onRegenerateJob(inspectedJob)} aria-label="같은 프롬프트로 이미지 재생성">
                  <RefreshCcw size={15} />
                </button>
              ) : null}
            </div>
          </div>
          <p>{inspectedJob.reason}</p>
          {inspectedCue?.scene ? <small>장면: {inspectedCue.scene}</small> : null}
          {inspectedBackend ? (
            <div className="image-job-routing">
              <span className="image-job-routing-backend">{imageBackendLabels[inspectedBackend]}</span>
              {inspectedCategoryLabel ? <span>{inspectedCategoryLabel}</span> : null}
              {inspectedLockHolds ? <span>실사 제외</span> : null}
              {inspectedRouting?.fallback ? <span>대체 백엔드</span> : null}
              {inspectedRouting?.reason ? <p>{inspectedReasonOutdated ? `계획 당시 기준: ${inspectedRouting.reason}` : inspectedRouting.reason}</p> : null}
              {inspectedLockStale ? <p>지금 기준으로는 이 컷을 실사로 다시 생성할 수 있습니다.</p> : null}
            </div>
          ) : null}
          <div className="image-job-metrics">
            <Metric label="해상도" value={`${displayWidth}x${displayHeight}`} />
            <Metric label="스텝" value={displayStepsLabel} />
            <Metric label="CFG" value={displayCfgLabel} />
            <Metric label="시드" value={promptSeed?.toString() ?? "자동"} />
          </div>
          {inspectedJob.error ? (
            <p className="image-job-error" role="alert">
              <AlertTriangle size={14} />
              {inspectedJob.error}
            </p>
          ) : null}
          {inspectedCue?.frame ? (
            <p className="image-job-frame">구도: {describeImageCueFrame(inspectedCue.frame)}</p>
          ) : null}
          <ImagePromptBlock label={isPhotorealJob ? "실사 프롬프트" : "프롬프트"} value={inspectedJob.prompt} />
          <ImagePromptBlock label="Negative" value={inspectedJob.negativePrompt} muted />
          {photorealSourceTags ? <ImagePromptBlock label="원본 태그 (NAI 구성)" value={photorealSourceTags} muted /> : null}
          {novelAiV4CaptionDebug ? <ImagePromptBlock label="NAI v4 captions" value={novelAiV4CaptionDebug} muted /> : null}
          {isPhotorealJob ? (
            <div className="image-job-spec">
              <span>{inspectedPreset?.label ?? inspectedPresetId ?? "실사 프리셋"}</span>
              {promptModel ? <span>{promptModel}</span> : null}
              {photorealReferenceNames.length > 0 ? <span>얼굴 참조: {photorealReferenceNames.join(", ")}</span> : <span>얼굴 참조 없음</span>}
            </div>
          ) : (
            <div className="image-job-spec">
              <span>{promptModel ?? state.imageProfile.model}</span>
              <span>{promptSampler ?? state.novelAi.sampler}</span>
              <span>{promptSchedule ?? state.novelAi.noiseSchedule}</span>
            </div>
          )}
          {otherBackend && ["queued", "completed", "failed", "canceled"].includes(inspectedJob.status) ? (
            <div className="image-job-switch-backend">
              <button
                className="icon-text-button"
                type="button"
                disabled={inspectedLockHolds || otherBackendOff}
                aria-describedby={inspectedLockHolds || otherBackendOff ? `image-job-switch-note-${inspectedJob.id}` : undefined}
                onClick={() => onRegenerateJob(inspectedJob, otherBackend)}
                title={inspectedLockHolds ? inspectedLockReason : undefined}
              >
                <RefreshCcw size={15} />
                다른 백엔드로 재생성 ({imageBackendLabels[otherBackend]})
              </button>
              {inspectedLockHolds ? (
                <small id={`image-job-switch-note-${inspectedJob.id}`}>{inspectedLockReason}</small>
              ) : otherBackendOff && otherBackend === "photoreal" ? (
                <small id={`image-job-switch-note-${inspectedJob.id}`}>
                  실사 이미지(ComfyUI) 연동이 꺼져 있습니다. 설정 탭에서 켜면 이 컷을 실사로 다시 생성할 수 있습니다.
                </small>
              ) : otherBackendOff ? (
                <small id={`image-job-switch-note-${inspectedJob.id}`}>
                  NovelAI 연동이 꺼져 있습니다. 개인 설정에서 켜면 이 컷을 NovelAI로 다시 생성할 수 있습니다.
                </small>
              ) : null}
            </div>
          ) : null}
          {jobEvidence.length > 0 ? (
            <div className="image-evidence-list">
              <strong>근거 메모리</strong>
              {jobEvidence.slice(0, 2).map((item) => (
                <p key={`${inspectedJob.id}-${item.nodeId}`}>{Math.round(item.score * 100)}% · {item.snippet}</p>
              ))}
            </div>
          ) : null}
        </article>
      ) : (
        <div className="empty-panel">
          <ImageIcon size={17} />
          <span>아직 생성 작업이 없습니다.</span>
        </div>
      )}

      {selectedAsset ? (
        <div className="image-result-focus">
          <SceneCard asset={selectedAsset} />
          <ImageFeedbackRow asset={selectedAsset} onDelete={onDeleteAsset} onFeedback={onFeedback} />
        </div>
      ) : null}

      {generatedAssets.length > 0 ? (
        <section className="image-asset-gallery" aria-label="생성된 이미지">
          <div className="runtime-card-subhead">
            <strong>생성 이미지</strong>
            <span>{generatedAssets.length}개</span>
          </div>
          <div className="image-asset-grid">
            {generatedAssets.map((asset) => (
              <ImageAssetTile
                active={asset.id === selectedAsset?.id}
                asset={asset}
                job={jobByAssetId.get(asset.id)}
                key={asset.id}
                onDelete={() => onDeleteAsset(asset.id)}
                onSelect={() => inspectAsset(asset)}
              />
            ))}
          </div>
        </section>
      ) : null}
      {/* Every job, newest first, in a scrollable list. It used to show the last four with a check mark on
          each row regardless of status and the error text only when there was no prompt — so a failed job
          looked identical to a successful one, and a run of failures scrolled out of reach. */}
      <div className="job-list image-job-list">
        {[...jobs].reverse().map((job) => (
          <div
            className={`job-row image-job-row ${job.id === inspectedJob?.id ? "active" : ""}`}
            data-status={job.status}
            key={job.id}
          >
            <button className="job-row-main" type="button" onClick={() => inspectJob(job)} aria-label={`${job.reason} 작업 보기`}>
              {job.status === "failed" ? (
                <AlertTriangle size={15} />
              ) : job.status === "canceled" ? (
                <X size={15} />
              ) : job.status === "completed" ? (
                <Check size={15} />
              ) : (
                <Activity size={15} />
              )}
              <span>
                {job.reason}
                <small>{job.error ?? job.prompt?.slice(0, 130)}</small>
              </span>
            </button>
            <div className="job-actions">
              <strong>
                {imageBackendBadgeLabels[readImageJobBackend(job)]} · {getImageJobStatusLabel(job.status)}
              </strong>
              {canUserRunQueuedImageJob(job) ? (
                <button className="icon-button" type="button" onClick={() => onRunJob(job)} aria-label="이미지 작업 실행">
                  <Play size={15} />
                </button>
              ) : null}
              {["queued", "planning", "generating"].includes(job.status) ? (
                <button className="icon-button" type="button" onClick={() => onCancelJob(job.id)} aria-label="이미지 작업 취소">
                  <Trash2 size={15} />
                </button>
              ) : null}
              {job.status === "completed" || job.status === "failed" || job.status === "canceled" ? (
                <button className="icon-button" type="button" onClick={() => onRegenerateJob(job)} aria-label="이미지 작업 재생성">
                  <RefreshCcw size={15} />
                </button>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ImageAssetTile({
  active,
  asset,
  job,
  onDelete,
  onSelect
}: {
  active: boolean;
  asset: ImageAsset;
  job?: ImageGenerationJob;
  onDelete: () => void;
  onSelect: () => void;
}) {
  const style = {
    "--tone-a": asset.palette[0],
    "--tone-b": asset.palette[1],
    "--tone-c": asset.palette[2]
  } as CSSProperties;

  return (
    <article className={`image-asset-card ${active ? "active" : ""}`} style={style}>
      <button className="image-asset-select" type="button" onClick={onSelect} aria-label={`${asset.title} 보기`}>
        <span className="image-asset-thumb">
          <AssetImage src={createImageAssetSrc(asset)} alt={asset.title} />
          <ImageBackendBadge asset={asset} />
        </span>
        <span className="image-asset-meta">
          <strong>{asset.title}</strong>
          <small>{job ? getImageJobStatusLabel(job.status) : imageAssetSourceLabels[asset.source]}</small>
        </span>
      </button>
      <button className="image-asset-delete" type="button" onClick={onDelete} aria-label={`${asset.title} 삭제`} title="삭제">
        <Trash2 size={14} />
      </button>
    </article>
  );
}

function ImageFeedbackRow({
  asset,
  onDelete,
  onFeedback
}: {
  asset: ImageAsset;
  onDelete?: (assetId: string) => void;
  onFeedback: (assetId: string, rating: ImageFeedbackRating) => void;
}) {
  return (
    <div className="feedback-row">
      <span>
        <strong>{asset.title}</strong>
        <small>{asset.feedback?.rating === "liked" ? "선택됨" : asset.feedback?.rating === "rejected" ? "제외됨" : asset.feedback?.rating === "neutral" ? "보류" : "미선택"}</small>
      </span>
      <div className="feedback-actions">
        <button
          className={`icon-button ${asset.feedback?.rating === "liked" ? "active" : ""}`}
          type="button"
          onClick={() => onFeedback(asset.id, "liked")}
          aria-label={`${asset.title} 선호 표시`}
        >
          <ThumbsUp size={15} />
        </button>
        <button
          className={`icon-button ${asset.feedback?.rating === "neutral" ? "active" : ""}`}
          type="button"
          onClick={() => onFeedback(asset.id, "neutral")}
          aria-label={`${asset.title} 보통 표시`}
        >
          <Minus size={15} />
        </button>
        <button
          className={`icon-button ${asset.feedback?.rating === "rejected" ? "active" : ""}`}
          type="button"
          onClick={() => onFeedback(asset.id, "rejected")}
          aria-label={`${asset.title} 제외 표시`}
        >
          <ThumbsDown size={15} />
        </button>
        {onDelete ? (
          <button className="icon-button" type="button" onClick={() => onDelete(asset.id)} aria-label={`${asset.title} 삭제`}>
            <Trash2 size={15} />
          </button>
        ) : null}
      </div>
    </div>
  );
}

function MemoryPanel({
  state,
  selectedContextPackId,
  onRedactMemory
}: {
  state: AppState;
  selectedContextPackId?: string;
  onRedactMemory?: (memoryId: string) => void;
}) {
  const pack = state.contextPacks.find((contextPack) => contextPack.id === selectedContextPackId) ?? state.contextPacks.at(-1);
  const scopedEvidence = pack ? getScopedContextEvidenceForDisplay(state, pack.evidence) : [];
  const latestHandoff = state.handoffs.at(-1);
  const latestContinuity = state.continuityChecks.at(-1);
  const latestModuleUsages = state.promptModuleUsages.slice(-6).reverse();
  const latestSidecarTrace = state.sidecarTraces.at(-1);
  return (
    <div className="panel-stack">
      <div className="panel-header compact">
        <div>
          <span className="section-kicker">Context Pack</span>
          <h2>{pack?.source === "neuralmap" ? "NeuralMap" : "로컬 참조"}</h2>
        </div>
        <StatusPill icon={<Brain size={15} />} label={`근거 ${scopedEvidence.length}개`} tone="neutral" />
      </div>
      {pack ? (
        <div className="context-pack">
          <strong>{pack.objective}</strong>
          {scopedEvidence.map((item) => (
            <div className="evidence-row" key={`${pack.id}-${item.nodeId}`}>
              <span>{Math.round(item.score * 100)}%</span>
              <p>{item.snippet}</p>
              <small>{item.reason}</small>
            </div>
          ))}
        </div>
      ) : null}
      {latestHandoff && latestContinuity ? (
        <div className="context-pack">
          <strong>Session handoff · {latestContinuity.status}</strong>
          <div className="evidence-row">
            <span>{latestHandoff.source}</span>
            <p>{latestHandoff.summary}</p>
            <small>{latestHandoff.previousSessionId} → {latestHandoff.nextSessionId}</small>
          </div>
          {latestContinuity.facts.slice(0, 4).map((fact) => (
            <div className="evidence-row" key={`${latestContinuity.id}-${fact.label}`}>
              <span>{fact.found ? "OK" : "WARN"}</span>
              <p>{fact.expected}</p>
              <small>{fact.label}</small>
            </div>
          ))}
        </div>
      ) : null}
      {latestModuleUsages.length > 0 ? (
        <div className="context-pack">
          <strong>Prompt module usage</strong>
          {latestModuleUsages.map((usage) => (
            <div className="evidence-row" key={usage.id}>
              <span>{Math.round(usage.score * 100)}%</span>
              <p>{usage.moduleTitle}</p>
              <small>{usage.source} · {usage.reason}</small>
            </div>
          ))}
        </div>
      ) : null}
      {latestSidecarTrace ? (
        <div className="context-pack">
          <strong>LLM sidecar · {latestSidecarTrace.status}</strong>
          <div className="evidence-row">
            <span>{latestSidecarTrace.source}</span>
            <p>{latestSidecarTrace.errors.length > 0 ? latestSidecarTrace.errors.join(" / ") : "structured sidecar parsed"}</p>
            <small>{new Date(latestSidecarTrace.createdAt).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}</small>
          </div>
          {latestSidecarTrace.rawPreview ? (
            <div className="evidence-row">
              <span>raw</span>
              <p>{latestSidecarTrace.rawPreview}</p>
              <small>preview</small>
            </div>
          ) : null}
          {latestSidecarTrace.requestPreview ? (
            <div className="evidence-row">
              <span>request</span>
              <p>{latestSidecarTrace.requestPreview}</p>
              <small>LLM에 전달한 system/context/user 미리보기</small>
            </div>
          ) : null}
        </div>
      ) : null}
      <div className="memory-list">
        {state.memoryEvents.slice(-6).reverse().map((event) => (
          <div className="memory-row" key={event.id}>
            <KeyRound size={15} />
            <div>
              <strong>{event.tags.join(", ")}</strong>
              <p>{event.content}</p>
            </div>
            <span>{Math.round(event.importance * 100)}</span>
            {onRedactMemory ? (
              <button className="icon-button" type="button" onClick={() => onRedactMemory(event.id)} aria-label="메모리 redaction">
                <Trash2 size={15} />
              </button>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

type RelationshipMapNodeKind = "persona" | "character" | "memory";
type RelationshipMapNode = {
  id: string;
  kind: RelationshipMapNodeKind;
  name: string;
  role: string;
  summary: string;
  relationship: string;
  mood: string;
  imageAsset?: ImageAsset;
  statusSections: RelationshipStatusSection[];
  memoryLines: string[];
  tags: string[];
  importance: number;
  memoryCount: number;
};
type RelationshipStatusSection = {
  id: string;
  title: string;
  value: string;
  rule?: string;
  source: "profile" | "memory" | "parameter";
  empty?: boolean;
  updatedAt?: string;
  userEdited?: boolean;
};
type RelationshipMapEdge = {
  id: string;
  from: string;
  to: string;
  label: string;
  strength: number;
  source: string;
  createdAt?: string;
};
type RelationshipMemoryRecord = {
  event: AppState["memoryEvents"][number];
  kind: string;
  content: string;
  actorId?: string;
  actorName?: string;
  ownerId?: string;
  targetId?: string;
  stateType?: string;
  value?: string;
  createdAt: string;
  importance: number;
  tags: string[];
};
type RelationshipMapView = {
  nodes: RelationshipMapNode[];
  edges: RelationshipMapEdge[];
  records: RelationshipMemoryRecord[];
  sync: RelationshipProgressSync;
};
type RelationshipProgressSync = {
  title: string;
  updatedAt?: string;
  memoryCount: number;
  statusCount: number;
  sourceCount: number;
};

function RelationshipMapPanel({
  state,
  onStateChange
}: {
  state: AppState;
  onStateChange: Dispatch<SetStateAction<AppState>>;
}) {
  const view = useMemo(() => createRelationshipMapView(state), [state]);
  const [selectedNodeId, setSelectedNodeId] = useState<string>();
  const [editingKey, setEditingKey] = useState<string>();
  const [editingValue, setEditingValue] = useState("");

  useEffect(() => {
    setSelectedNodeId((current) => {
      if (current && view.nodes.some((node) => node.id === current)) {
        return current;
      }

      return view.nodes[0]?.id;
    });
  }, [view.nodes]);

  const selectedNode = view.nodes.find((node) => node.id === selectedNodeId) ?? view.nodes[0];

  const beginStatusEdit = useCallback((node: RelationshipMapNode, section: RelationshipStatusSection) => {
    setEditingKey(relationshipStatusOverrideKey(node.id, section.title));
    setEditingValue(section.value);
  }, []);

  const cancelStatusEdit = useCallback(() => {
    setEditingKey(undefined);
    setEditingValue("");
  }, []);

  const saveStatusEdit = useCallback(
    (node: RelationshipMapNode, section: RelationshipStatusSection) => {
      const value = editingValue;
      onStateChange((current) => upsertRelationshipStatusOverride(current, node.id, section.title, value));
      setEditingKey(undefined);
      setEditingValue("");
    },
    [editingValue, onStateChange]
  );

  const resetStatusEdit = useCallback(
    (node: RelationshipMapNode, section: RelationshipStatusSection) => {
      onStateChange((current) => removeRelationshipStatusOverride(current, node.id, section.title));
      setEditingKey(undefined);
      setEditingValue("");
    },
    [onStateChange]
  );
  const selectedEdges = view.edges.filter((edge) => edge.from === selectedNode?.id || edge.to === selectedNode?.id);
  const latestRecords = view.records.slice(-5).reverse();
  const selectedStatusUpdatedAt = selectedNode?.statusSections
    .map((section) => section.updatedAt)
    .filter((value): value is string => Boolean(value))
    .sort()
    .at(-1);

  return (
    <div className="panel-stack relationship-map-panel">
      <div className="panel-header compact">
        <div>
          <span className="section-kicker">Relationship Map</span>
          <h2>관계도</h2>
        </div>
        <StatusPill icon={<Network size={15} />} label={`${view.nodes.length}명`} tone={state.relationshipMap.enabled ? "good" : "neutral"} />
      </div>

      <div className="relationship-sync-strip">
        <div>
          <span>활성 진행</span>
          <strong>{view.sync.title}</strong>
        </div>
        <div>
          <span>상태 소스</span>
          <strong>{view.sync.statusCount}개</strong>
        </div>
        <div>
          <span>최근 동기화</span>
          <strong>{view.sync.updatedAt ? formatCompactTime(view.sync.updatedAt) : "대기"}</strong>
        </div>
      </div>

      <div className="relationship-list-shell">
        <div className="relationship-roster-list" role="list" aria-label="관계 인물 목록">
          {view.nodes.map((node) => {
            const latestMemorySection = node.statusSections.find((section) => section.source === "memory");
            return (
              <button
                className={`relationship-list-row ${node.id === selectedNode?.id ? "selected" : ""}`}
                key={node.id}
                type="button"
                onClick={() => setSelectedNodeId(node.id)}
              >
                <RelationshipNodePortrait node={node} />
                <span>
                  <small>{getRelationshipNodeKindLabel(node.kind)}</small>
                  <strong>{node.name}</strong>
                  <em>{latestMemorySection?.value ?? createRelationshipListSummary(node)}</em>
                </span>
                <b>{node.memoryCount}</b>
              </button>
            );
          })}
        </div>

        {selectedNode ? (
          <article className="relationship-inspector relationship-status-panel">
            <div className="relationship-profile-head">
              <RelationshipNodePortrait node={selectedNode} />
              <div>
                <span>{getRelationshipNodeKindLabel(selectedNode.kind)}</span>
                <strong>{selectedNode.name}</strong>
                <small>{selectedNode.role}</small>
              </div>
            </div>

            <div className="relationship-metrics">
              <Metric label="상태" value={selectedNode.statusSections.length.toString()} />
              <Metric label="기록" value={selectedNode.memoryCount.toString()} />
              <Metric label="연결" value={selectedEdges.length.toString()} />
            </div>

            <div className="relationship-status-head">
              <strong>현재 상태창</strong>
              <span>{selectedStatusUpdatedAt ? `${formatCompactTime(selectedStatusUpdatedAt)} 갱신` : "프로필 기준"}</span>
            </div>

            <div className="relationship-status-board">
              {selectedNode.statusSections.length > 0 ? (
                selectedNode.statusSections.map((section) => {
                  const cardKey = relationshipStatusOverrideKey(selectedNode.id, section.title);
                  const isEditing = editingKey === cardKey;
                  return (
                    <section
                      className={`relationship-status-card ${section.empty ? "empty" : ""} source-${section.source} ${section.userEdited ? "user-edited" : ""}`}
                      key={section.id}
                    >
                      <div>
                        <strong>{section.title}</strong>
                        <span>{section.userEdited ? "직접 입력" : formatRelationshipStatusSource(section)}</span>
                      </div>
                      {isEditing ? (
                        <div className="relationship-status-editor">
                          <textarea
                            value={editingValue}
                            rows={3}
                            autoFocus
                            onChange={(event) => setEditingValue(event.target.value)}
                            aria-label={`${section.title} 직접 입력`}
                          />
                          <div className="relationship-status-actions">
                            <button type="button" className="ghost" onClick={cancelStatusEdit}>
                              취소
                            </button>
                            <button type="button" onClick={() => saveStatusEdit(selectedNode, section)}>
                              저장
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <p>{section.value}</p>
                          <div className="relationship-status-actions">
                            <button type="button" className="ghost" onClick={() => beginStatusEdit(selectedNode, section)}>
                              편집
                            </button>
                            {section.userEdited ? (
                              <button type="button" className="ghost" onClick={() => resetStatusEdit(selectedNode, section)}>
                                자동값으로
                              </button>
                            ) : null}
                          </div>
                        </>
                      )}
                      {section.rule ? <small>{section.rule}</small> : null}
                    </section>
                  );
                })
              ) : (
                <section className="relationship-status-card empty">
                  <div>
                    <strong>상태창</strong>
                    <span>대기</span>
                  </div>
                  <p>저장된 현재 상태가 없습니다.</p>
                </section>
              )}
            </div>

            {selectedEdges.length > 0 ? (
              <div className="relationship-edge-list">
                <strong>연결</strong>
                {selectedEdges.map((edge) => {
                  const counterpart = view.nodes.find((node) => node.id === (edge.from === selectedNode.id ? edge.to : edge.from));
                  return (
                    <div className="relationship-edge-row" key={edge.id}>
                      <span>{Math.round(edge.strength * 100)}%</span>
                      <p>{counterpart?.name ?? "대상"} · {edge.label}</p>
                      <small>{edge.source}</small>
                    </div>
                  );
                })}
              </div>
            ) : null}

            {selectedNode.memoryLines.length > 0 ? (
              <div className="relationship-memory-list inline">
                <strong>최근 기록</strong>
                {selectedNode.memoryLines.map((line) => <p key={`${selectedNode.id}-memory-${line}`}>{line}</p>)}
              </div>
            ) : null}
          </article>
        ) : (
          <div className="empty-panel">관계도에 표시할 인물 기록이 없습니다.</div>
        )}
      </div>

      <div className="relationship-rule-summary">
        <strong>{state.relationshipMap.enabled ? "정제 규칙 활성" : "정제 규칙 비활성"}</strong>
        <p>{state.relationshipMap.statusPrompt || "상태창 정제 프롬프트가 비어 있습니다."}</p>
      </div>

      {latestRecords.length > 0 ? (
        <div className="relationship-memory-list">
          <strong>최근 관계도 소스</strong>
          {latestRecords.map((record) => (
            <p key={record.event.id}>{stripRelationshipMemoryPrefix(record.content)}</p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function RelationshipMapCanvas({
  nodes,
  edges,
  selectedNodeId,
  onSelectNode
}: {
  nodes: RelationshipMapNode[];
  edges: RelationshipMapEdge[];
  selectedNodeId?: string;
  onSelectNode: (nodeId: string) => void;
}) {
  const layout = useMemo(() => createRelationshipMapLayout(nodes), [nodes]);
  const positionById = useMemo(() => new Map(layout.map((item) => [item.node.id, item])), [layout]);

  if (nodes.length === 0) {
    return (
      <div className="relationship-map-canvas empty">
        <Network size={18} />
        <span>인물 노드 없음</span>
      </div>
    );
  }

  return (
    <div className="relationship-map-canvas" aria-label="관계도">
      <svg viewBox="0 0 100 100" role="presentation">
        {edges.map((edge) => {
          const from = positionById.get(edge.from);
          const to = positionById.get(edge.to);
          if (!from || !to) {
            return null;
          }

          const selected = edge.from === selectedNodeId || edge.to === selectedNodeId;
          const midX = (from.x + to.x) / 2;
          const midY = (from.y + to.y) / 2;
          return (
            <g className={selected ? "selected" : ""} key={edge.id}>
              <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} style={{ strokeWidth: 0.45 + edge.strength * 1.2 }} />
              <text x={midX} y={midY}>{edge.label.slice(0, 12)}</text>
            </g>
          );
        })}
      </svg>
      {layout.map(({ node, x, y }) => {
        const palette = node.imageAsset?.palette ?? ["#274b63", "#e8edea", "#d0a34a"];
        const style = {
          left: `${x}%`,
          top: `${y}%`,
          "--node-a": palette[0],
          "--node-b": palette[1],
          "--node-c": palette[2]
        } as CSSProperties;
        return (
          <button
            className={`relationship-node ${node.kind} ${node.id === selectedNodeId ? "selected" : ""}`}
            key={node.id}
            style={style}
            type="button"
            onClick={() => onSelectNode(node.id)}
            aria-label={`${node.name} 상태창 열기`}
          >
            <RelationshipNodePortrait node={node} />
            <small>{node.name}</small>
          </button>
        );
      })}
    </div>
  );
}

function RelationshipNodePortrait({ node }: { node: RelationshipMapNode }) {
  const palette = node.imageAsset?.palette ?? ["#274b63", "#e8edea", "#d0a34a"];
  const style = {
    "--node-a": palette[0],
    "--node-b": palette[1],
    "--node-c": palette[2]
  } as CSSProperties;

  return (
    <span className="relationship-node-portrait" style={style}>
      {createImageAssetSrc(node.imageAsset) ? <img src={createImageAssetSrc(node.imageAsset)} alt="" /> : <span>{createRelationshipInitials(node.name)}</span>}
    </span>
  );
}

function getRelationshipNodeKindLabel(kind: RelationshipMapNodeKind): string {
  if (kind === "persona") {
    return "내 캐릭터";
  }
  if (kind === "character") {
    return "캐릭터";
  }
  return "메모리 인물";
}

function createRelationshipListSummary(node: RelationshipMapNode): string {
  return [node.relationship, node.mood || node.summary].filter((item) => item.trim()).join(" · ") || "진행 기록 대기";
}

function formatRelationshipStatusSource(section: RelationshipStatusSection): string {
  if (section.source === "memory") {
    return section.updatedAt ? `진행 ${formatCompactTime(section.updatedAt)}` : "진행";
  }
  if (section.empty) {
    return "대기";
  }
  return section.source === "profile" ? "프로필" : "설정";
}

function createRelationshipMapLayout(nodes: RelationshipMapNode[]): Array<{ node: RelationshipMapNode; x: number; y: number }> {
  if (nodes.length === 1) {
    return [{ node: nodes[0], x: 50, y: 50 }];
  }

  const centerIndex = Math.max(0, nodes.findIndex((node) => node.kind === "persona"));
  const centerNode = nodes[centerIndex] ?? nodes[0];
  const ringNodes = nodes.filter((node) => node.id !== centerNode.id);
  return [
    { node: centerNode, x: 50, y: 50 },
    ...ringNodes.map((node, index) => {
      const angle = -Math.PI / 2 + (index / Math.max(1, ringNodes.length)) * Math.PI * 2;
      return {
        node,
        x: 50 + Math.cos(angle) * 34,
        y: 50 + Math.sin(angle) * 34
      };
    })
  ];
}

function getRelationshipRuntimeMemoryEvents(state: AppState): AppState["memoryEvents"] {
  const activeRun = getRelationshipActiveProgressRun(state);
  const byId = new Map<string, AppState["memoryEvents"][number]>();

  for (const event of [...(activeRun?.memoryEvents ?? []), ...state.memoryEvents]) {
    if (relationshipMemoryEventBelongsToActiveProgress(state, event, activeRun)) {
      byId.set(event.id, event);
    }
  }

  return [...byId.values()].sort((a, b) => Date.parse(a.createdAt) - Date.parse(b.createdAt));
}

function getRelationshipActiveProgressRun(state: AppState): SimulationProgressRun | undefined {
  return state.progressRuns.find((run) => run.id === state.activeProgressRunId);
}

function relationshipMemoryEventBelongsToActiveProgress(
  state: AppState,
  event: AppState["memoryEvents"][number],
  activeRun: SimulationProgressRun | undefined
): boolean {
  if (event.simulationId !== state.simulation.id) {
    return false;
  }

  const runId =
    readRelationshipMapMetadataText(event.metadata, "progress_run_id") ??
    readRelationshipMapMetadataText(event.metadata, "run_id") ??
    readRelationshipMapMetadataText(event.metadata, "activeProgressRunId");
  if (runId) {
    return runId === state.activeProgressRunId;
  }

  const activeSessionIds = new Set([state.simulation.activeSessionId, ...(activeRun?.sessionIds ?? [])].filter(Boolean));
  const sessionId = readRelationshipMapMetadataText(event.metadata, "session_id") ?? event.sessionId;
  return activeSessionIds.size === 0 || activeSessionIds.has(sessionId);
}

function createRelationshipProgressSync(
  state: AppState,
  records: RelationshipMemoryRecord[],
  events: AppState["memoryEvents"]
): RelationshipProgressSync {
  const activeRun = getRelationshipActiveProgressRun(state);
  const latestEventTime = events.map((event) => event.createdAt).sort().at(-1);
  const statusCount = records.filter((record) => record.kind === "state" || record.kind === "goal" || record.kind === "relationship").length;

  return {
    title: activeRun?.title ?? "현재 진행",
    updatedAt: latestEventTime ?? activeRun?.updatedAt ?? state.simulation.updatedAt,
    memoryCount: events.length,
    statusCount,
    sourceCount: records.length
  };
}

function createRelationshipMapView(state: AppState): RelationshipMapView {
  const personaNodeId = "persona:self";
  const personaName = getPersonaDisplayName(state);
  const personaCharacter = getPersonaCharacter(state);
  const runtimeMemoryEvents = getRelationshipRuntimeMemoryEvents(state);
  const records = runtimeMemoryEvents.map(createRelationshipMemoryRecord);
  const nodes: RelationshipMapNode[] = [
    {
      id: personaNodeId,
      kind: "persona",
      name: personaName,
      role: personaCharacter?.role ?? (state.userPersona.enabled ? state.userPersona.role : "플레이어"),
      summary: personaCharacter?.summary ?? (state.userPersona.enabled ? state.userPersona.background : "사용자 입력으로 진행을 이끄는 내 캐릭터"),
      relationship: "내 캐릭터",
      mood: personaCharacter?.currentMood ?? (state.userPersona.enabled ? state.userPersona.style : "직접 입력으로 행동을 결정"),
      imageAsset: personaCharacter ? findRelationshipImageAsset(state, personaCharacter.id) : undefined,
      statusSections: [],
      memoryLines: [],
      tags: [],
      importance: 0.72,
      memoryCount: 0
    },
    ...state.characters.filter((character) => character.id !== personaCharacter?.id).map((character) => ({
      id: character.id,
      kind: "character" as const,
      name: character.name,
      role: character.role,
      summary: character.summary,
      relationship: character.relationship,
      mood: character.currentMood,
      imageAsset: findRelationshipImageAsset(state, character.id),
      statusSections: [],
      memoryLines: [],
      tags: [],
      importance: 0.7,
      memoryCount: 0
    }))
  ];
  const nodeIds = new Set(nodes.map((node) => node.id));

  for (const record of records) {
    const looseActorId = resolveRelationshipLooseNodeId(state, record.actorId, record.actorName, personaName);
    if (looseActorId && !nodeIds.has(looseActorId.id)) {
      nodeIds.add(looseActorId.id);
      nodes.push({
        id: looseActorId.id,
        kind: "memory",
        name: looseActorId.name,
        role: "진행 중 등장 인물",
        summary: stripRelationshipMemoryPrefix(record.content),
        relationship: "메모리에서 발견",
        mood: "",
        statusSections: [],
        memoryLines: [],
        tags: [],
        importance: record.importance,
        memoryCount: 0
      });
    }
  }

  const edges = createRelationshipEdges(state, nodes, records, personaNodeId, personaName);
  const hydratedNodes = nodes.map((node) => {
    const relatedRecords = records.filter((record) => relationshipRecordMatchesNode(record, node, personaName, personaCharacter?.id));
    const statusSections = createRelationshipStatusSections(state, node, relatedRecords);
    return {
      ...node,
      statusSections,
      memoryLines: relatedRecords
        .filter((record) => record.kind !== "state")
        .slice(-5)
        .reverse()
        .map((record) => stripRelationshipMemoryPrefix(record.content)),
      tags: [...new Set(relatedRecords.flatMap((record) => record.tags))].slice(0, 6),
      importance: relatedRecords.reduce((max, record) => Math.max(max, record.importance), node.importance),
      memoryCount: relatedRecords.length
    };
  });

  return {
    nodes: hydratedNodes,
    edges,
    records,
    sync: createRelationshipProgressSync(state, records, runtimeMemoryEvents)
  };
}

function createRelationshipEdges(
  state: AppState,
  nodes: RelationshipMapNode[],
  records: RelationshipMemoryRecord[],
  personaNodeId: string,
  personaName: string
): RelationshipMapEdge[] {
  const edges = new Map<string, RelationshipMapEdge>();
  const controlledCharacterId = getPersonaCharacter(state)?.id;

  for (const character of state.characters.filter((candidate) => candidate.id !== controlledCharacterId)) {
    const relatedCount = records.filter((record) => record.actorId === character.id || record.ownerId === character.id || record.targetId === character.id).length;
    upsertRelationshipEdge(edges, {
      id: `base:${personaNodeId}:${character.id}`,
      from: personaNodeId,
      to: character.id,
      label: character.relationship || "관계 미정",
      strength: Math.min(1, 0.42 + relatedCount * 0.06),
      source: "캐릭터 설정"
    });
  }

  for (const record of records) {
    const relationshipLike = record.kind === "relationship" || record.tags.includes("relationship") || record.tags.some((tag) => tag.includes("relationship"));
    if (!relationshipLike) {
      continue;
    }

    const from = resolveRelationshipNodeId(state, nodes, record.actorId ?? record.ownerId, record.actorName, personaName) ?? personaNodeId;
    const to = resolveRelationshipNodeId(state, nodes, record.targetId, undefined, personaName) ?? (from === personaNodeId ? nodes.find((node) => node.kind === "character")?.id : personaNodeId);
    if (!to || from === to) {
      continue;
    }

    upsertRelationshipEdge(edges, {
      id: `memory:${[from, to].sort().join(":")}`,
      from,
      to,
      label: stripRelationshipMemoryPrefix(record.value ?? record.content),
      strength: Math.min(1, 0.5 + record.importance * 0.45),
      source: "메모리",
      createdAt: record.createdAt
    });
  }

  return [...edges.values()].sort((a, b) => b.strength - a.strength).slice(0, 18);
}

function upsertRelationshipEdge(edges: Map<string, RelationshipMapEdge>, edge: RelationshipMapEdge) {
  const key = [edge.from, edge.to].sort().join("::");
  const existing = edges.get(key);
  if (!existing || edge.strength >= existing.strength || (edge.createdAt && existing.createdAt && edge.createdAt > existing.createdAt)) {
    edges.set(key, {
      ...edge,
      id: existing?.id ?? edge.id
    });
  }
}

function createRelationshipStatusSections(state: AppState, node: RelationshipMapNode, records: RelationshipMemoryRecord[]): RelationshipStatusSection[] {
  const profileSections = createRelationshipProfileStatusSections(state, node);
  const currentStateRecords = latestRelationshipRecordsByKey(
    records.filter((record) => record.kind === "state" || record.kind === "goal" || record.kind === "relationship"),
    (record) => `${record.kind}:${getRelationshipRecordStatusKey(record)}`
  );
  const currentStateRecordByKey = new Map(
    currentStateRecords.map((record) => [normalizeRelationshipStatusKey(getRelationshipRecordStatusKey(record)), record])
  );
  const parameterSections = state.relationshipMap.parameters
    .filter((parameter) => parameter.enabled && parameter.title.trim())
    .sort((a, b) => b.priority - a.priority)
    .map((parameter) => {
      const record = findRelationshipParameterRecord(currentStateRecordByKey, parameter.title);
      const fallback = createRelationshipParameterFallbackValue(state, node, parameter.title);
      return {
        id: `parameter:${node.id}:${parameter.id}`,
        title: parameter.title,
        value: record ? record.value ?? stripRelationshipMemoryPrefix(record.content) : fallback || "아직 기록 없음",
        rule: record ? `진행에서 ${formatCompactTime(record.createdAt)} 갱신` : undefined,
        source: record ? "memory" as const : fallback ? "profile" as const : "parameter" as const,
        empty: !record && !fallback,
        updatedAt: record?.createdAt
      };
    });
  const parameterKeys = new Set(
    state.relationshipMap.parameters
      .filter((parameter) => parameter.enabled && parameter.title.trim())
      .flatMap((parameter) => createRelationshipParameterAliases(parameter.title).map(normalizeRelationshipStatusKey))
  );
  const extraStateSections = currentStateRecords
    .filter((record) => !parameterKeys.has(normalizeRelationshipStatusKey(getRelationshipRecordStatusKey(record))))
    .slice(-4)
    .map((record) => ({
      id: `memory:${node.id}:${record.event.id}`,
      title: formatRelationshipStateType(record.stateType ?? record.kind),
      value: record.value ?? stripRelationshipMemoryPrefix(record.content),
      rule: `진행에서 ${formatCompactTime(record.createdAt)} 갱신`,
      source: "memory" as const,
      updatedAt: record.createdAt
    }));

  const sections = uniqueRelationshipStatusSections([...parameterSections, ...extraStateSections, ...profileSections])
    .filter((section) => section.value.trim())
    .slice(0, 18);
  return sections.map((section) => applyRelationshipStatusOverride(state, node, section));
}

function relationshipStatusOverrideKey(nodeId: string, title: string): string {
  return `${nodeId}::${normalizeRelationshipStatusKey(title)}`;
}

function findRelationshipStatusOverride(state: AppState, nodeId: string, title: string) {
  const statusKey = normalizeRelationshipStatusKey(title);
  return state.relationshipStatusOverrides.find(
    (override) => override.nodeId === nodeId && override.statusKey === statusKey
  );
}

function applyRelationshipStatusOverride(
  state: AppState,
  node: RelationshipMapNode,
  section: RelationshipStatusSection
): RelationshipStatusSection {
  const override = findRelationshipStatusOverride(state, node.id, section.title);
  if (!override) {
    return section;
  }

  return {
    ...section,
    value: override.value,
    userEdited: true,
    empty: false,
    updatedAt: override.updatedAt ?? section.updatedAt
  };
}

function upsertRelationshipStatusOverride(state: AppState, nodeId: string, title: string, value: string): AppState {
  const trimmed = value.trim();
  if (!trimmed) {
    return removeRelationshipStatusOverride(state, nodeId, title);
  }

  const statusKey = normalizeRelationshipStatusKey(title);
  const next: AppState["relationshipStatusOverrides"][number] = {
    nodeId,
    statusKey,
    title,
    value: trimmed,
    updatedAt: new Date().toISOString()
  };
  const others = state.relationshipStatusOverrides.filter(
    (override) => relationshipStatusOverrideKey(override.nodeId, override.title) !== relationshipStatusOverrideKey(nodeId, title)
  );
  return {
    ...state,
    relationshipStatusOverrides: [...others, next]
  };
}

function removeRelationshipStatusOverride(state: AppState, nodeId: string, title: string): AppState {
  const targetKey = relationshipStatusOverrideKey(nodeId, title);
  const filtered = state.relationshipStatusOverrides.filter(
    (override) => relationshipStatusOverrideKey(override.nodeId, override.title) !== targetKey
  );
  if (filtered.length === state.relationshipStatusOverrides.length) {
    return state;
  }
  return {
    ...state,
    relationshipStatusOverrides: filtered
  };
}

function getRelationshipRecordStatusKey(record: RelationshipMemoryRecord): string {
  if (record.stateType?.trim()) {
    return record.stateType.trim();
  }
  if (record.kind === "relationship") {
    return "관계";
  }
  if (record.kind === "goal") {
    return "목표";
  }
  return record.content;
}

function uniqueRelationshipStatusSections(sections: RelationshipStatusSection[]): RelationshipStatusSection[] {
  const seen = new Set<string>();
  return sections.filter((section) => {
    const key = normalizeRelationshipStatusKey(section.title);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function findRelationshipParameterRecord(
  recordsByKey: Map<string, RelationshipMemoryRecord>,
  title: string
): RelationshipMemoryRecord | undefined {
  return createRelationshipParameterAliases(title)
    .map((alias) => recordsByKey.get(normalizeRelationshipStatusKey(alias)))
    .find((record): record is RelationshipMemoryRecord => Boolean(record));
}

function createRelationshipParameterAliases(title: string): string[] {
  const key = normalizeRelationshipStatusKey(title);
  const aliases = [title];

  if (["의상", "의상태그", "착용", "복장", "옷"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    aliases.push("Wearing", "OutfitTags", "착용");
  }
  if (["상태태그", "상태", "컨디션태그", "캐릭터상태태그"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    aliases.push("StatusTags", "ExpressionTags", "PoseTags", "ActionTags", "InteractionTags", "InteractionPhaseTags", "HeldItemTags", "PhysicalStateTags", "BodyStateTags", "Emotion", "PhysicalCondition");
  }
  if (["관계", "관계상태"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    aliases.push("Relationship", "relationship");
  }
  if (["생각", "내면", "속마음"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    aliases.push("Thought", "InnerThought", "Intention");
  }
  if (["위치", "장소"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    aliases.push("Location");
  }
  if (["목표", "의도"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    aliases.push("Goal");
  }

  return [...new Set(aliases)];
}

function createRelationshipProfileStatusSections(state: AppState, node: RelationshipMapNode): RelationshipStatusSection[] {
  if (node.kind === "persona" && state.userPersona.source === "character") {
    const visualProfile = findRelationshipVisualProfile(state, node);
    return [
      createRelationshipProfileSection(node, "시점", node.name),
      createRelationshipProfileSection(node, "역할", node.role),
      createRelationshipProfileSection(node, "현재", node.mood),
      createRelationshipProfileSection(node, "기본 의상 태그", visualProfile?.defaultOutfitPrompt),
      createRelationshipProfileSection(node, "의상 키워드", formatRelationshipOutfitKeywordMap(visualProfile?.outfitPrompts)),
      createRelationshipProfileSection(node, "플레이 목표", state.userPersona.goals),
      createRelationshipProfileSection(node, "경계", state.userPersona.boundaries)
    ].filter((section): section is RelationshipStatusSection => Boolean(section));
  }

  if (node.kind === "persona") {
    return [
      createRelationshipProfileSection(node, "역할", state.userPersona.enabled ? state.userPersona.role : "페르소나 비활성"),
      createRelationshipProfileSection(node, "목표", state.userPersona.goals),
      createRelationshipProfileSection(node, "경계", state.userPersona.boundaries)
    ].filter((section): section is RelationshipStatusSection => Boolean(section));
  }

  const visualProfile = findRelationshipVisualProfile(state, node);
  return [
    createRelationshipProfileSection(node, "역할", node.role),
    createRelationshipProfileSection(node, "관계", node.relationship),
    createRelationshipProfileSection(node, "현재", node.mood),
    createRelationshipProfileSection(node, "기본 의상 태그", visualProfile?.defaultOutfitPrompt),
    createRelationshipProfileSection(node, "의상 키워드", formatRelationshipOutfitKeywordMap(visualProfile?.outfitPrompts))
  ].filter((section): section is RelationshipStatusSection => Boolean(section));
}

function findRelationshipVisualProfile(state: AppState, node: RelationshipMapNode): AppState["visualProfiles"][number] | undefined {
  if (node.kind === "persona") {
    const character = getPersonaCharacter(state);
    return character ? state.visualProfiles.find((profile) => profile.characterId === character.id) : undefined;
  }

  return state.visualProfiles.find((profile) => profile.characterId === node.id);
}

function formatRelationshipOutfitKeywordMap(outfitPrompts: Record<string, string> | undefined): string | undefined {
  const entries = Object.entries(outfitPrompts ?? {})
    .map(([key, value]) => [key.trim(), value.trim()] as const)
    .filter(([key, value]) => key && value)
    .slice(0, 5);

  if (entries.length === 0) {
    return undefined;
  }

  return entries.map(([key, value]) => `${key}: ${value}`).join(" / ");
}

function createRelationshipProfileSection(node: RelationshipMapNode, title: string, value: string | undefined): RelationshipStatusSection | undefined {
  if (!value?.trim()) {
    return undefined;
  }

  return {
    id: `profile:${node.id}:${title}`,
    title,
    value,
    source: "profile"
  };
}

function createRelationshipParameterFallbackValue(state: AppState, node: RelationshipMapNode, title: string): string | undefined {
  const key = normalizeRelationshipStatusKey(title);
  if (["의상", "의상태그", "착용", "복장", "옷"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    return findRelationshipVisualProfile(state, node)?.defaultOutfitPrompt;
  }
  if (["상태태그", "캐릭터상태태그"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    return node.mood;
  }
  if (key === normalizeRelationshipStatusKey("관계")) {
    return node.relationship;
  }
  if (["현재", "상태", "감정", "기분"].some((candidate) => key === normalizeRelationshipStatusKey(candidate))) {
    return node.mood;
  }
  if (key === normalizeRelationshipStatusKey("역할")) {
    return node.role;
  }
  if (key === normalizeRelationshipStatusKey("목표") && node.kind === "persona") {
    return state.userPersona.goals;
  }
  return undefined;
}

function createRelationshipMemoryRecord(event: AppState["memoryEvents"][number]): RelationshipMemoryRecord {
  const kind = readRelationshipMapMemoryKind(event);
  return {
    event,
    kind,
    content: event.content,
    actorId: event.actorId,
    actorName: event.actorName,
    ownerId: readStateMemoryOwnerId(event) ?? event.actorId,
    targetId: readStateMemoryTargetId(event),
    stateType: readStateMemoryStateType(event),
    value: readStateMemoryValue(event),
    createdAt: event.createdAt,
    importance: event.importance,
    tags: event.tags
  };
}

function readRelationshipMapMemoryKind(event: AppState["memoryEvents"][number]): string {
  return readStateMemoryKind(event);
}

function readRelationshipMapMetadataText(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function relationshipRecordMatchesNode(record: RelationshipMemoryRecord, node: RelationshipMapNode, personaName: string, controlledCharacterId?: string): boolean {
  if (node.kind === "persona") {
    return (
      record.actorName === personaName ||
      record.targetId === node.id ||
      record.ownerId === node.id ||
      Boolean(controlledCharacterId && (record.actorId === controlledCharacterId || record.ownerId === controlledCharacterId || record.targetId === controlledCharacterId))
    );
  }

  return (
    record.actorId === node.id ||
    record.ownerId === node.id ||
    record.targetId === node.id ||
    Boolean(record.actorName && record.actorName === node.name)
  );
}

function resolveRelationshipNodeId(
  state: AppState,
  nodes: RelationshipMapNode[],
  id?: string,
  name?: string,
  personaName?: string
): string | undefined {
  if (!id && !name) {
    return undefined;
  }

  if (id && nodes.some((node) => node.id === id)) {
    return id;
  }

  const controlledCharacter = getPersonaCharacter(state);
  if (controlledCharacter && (id === controlledCharacter.id || name === controlledCharacter.name)) {
    return "persona:self";
  }

  const character = state.characters.find((candidate) => candidate.id === id || candidate.name === name);
  if (character) {
    return character.id;
  }

  if (name && personaName && name === personaName) {
    return "persona:self";
  }

  return nodes.find((node) => node.name === name || node.id === id)?.id;
}

function resolveRelationshipLooseNodeId(
  state: AppState,
  id?: string,
  name?: string,
  personaName?: string
): { id: string; name: string } | undefined {
  if (!id && !name) {
    return undefined;
  }

  const character = state.characters.find((candidate) => candidate.id === id || candidate.name === name);
  if (character || (name && personaName && name === personaName)) {
    return undefined;
  }

  const displayName = name ?? id ?? "Unknown";
  return {
    id: `memory-person:${createRelationshipSlug(displayName)}`,
    name: displayName
  };
}

function findRelationshipImageAsset(state: AppState, characterId: string): ImageAsset | undefined {
  // A picked profile image is the face the creator chose for this character, so it outranks whatever cut happened
  // to include them last (which may be a back view, a crowd shot, or another character's close-up).
  const profileAsset = createProfileImagePortraitAsset(state, characterId);
  if (profileAsset) {
    return profileAsset;
  }
  const candidates = state.imageAssets
    .slice()
    .reverse()
    .filter((asset) => asset.characterIds.includes(characterId));
  return candidates.find((asset) => asset.dataUrl && (asset.representative || asset.source === "generated")) ?? candidates[0];
}

const PROFILE_PORTRAIT_PALETTE: [string, string, string] = ["#274b63", "#e8edea", "#d0a34a"];

/**
 * The character's profile image as a display-only asset for the portrait. Photoreal first, then NovelAI. It is
 * built here, never stored: a profile is simulation-scoped and must not join the run's imageAssets, where reuse
 * and stored-asset picking would start treating it as a scene image.
 */
function createProfileImagePortraitAsset(state: AppState, characterId: string): ImageAsset | undefined {
  const visualProfile = state.visualProfiles.find((profile) => profile.characterId === characterId);
  const profileImage = selectDisplayProfileImage(visualProfile?.profileImages);
  if (!visualProfile || !profileImage?.objectKey) {
    return undefined;
  }
  return {
    id: profileImage.assetId,
    simulationId: state.simulation.id,
    title: `${visualProfile.displayName} 프로필`,
    source: "stored",
    prompt: profileImage.prompt,
    negativePrompt: profileImage.negativePrompt ?? "",
    safetyLevel: visualProfile.defaultSafetyLevel,
    characterIds: [characterId],
    tags: [],
    createdAt: profileImage.selectedAt,
    palette: PROFILE_PORTRAIT_PALETTE,
    mimeType: profileImage.mimeType,
    objectKey: profileImage.objectKey,
    backend: profileImage.backend
  };
}

function latestRelationshipRecordsByKey<T>(items: T[], keyOf: (item: T) => string): T[] {
  const map = new Map<string, T>();
  for (const item of items) {
    const key = keyOf(item);
    if (map.has(key)) {
      map.delete(key);
    }
    map.set(key, item);
  }

  return [...map.values()];
}

function stripRelationshipMemoryPrefix(value: string): string {
  return value.replace(/^\[(?:Event|State|Observation|Belief|OpenThread|Goal)\]\s*/u, "");
}

function formatRelationshipStateType(value: string): string {
  const labels: Record<string, string> = {
    Wearing: "착용",
    OutfitTags: "의상 태그",
    StatusTags: "상태 태그",
    ExpressionTags: "표정 태그",
    PoseTags: "자세 태그",
    ActionTags: "행동 태그",
    InteractionTags: "상호작용 태그",
    InteractionPhaseTags: "상호작용 단계",
    HeldItemTags: "소지품 태그",
    PhysicalStateTags: "신체 상태 태그",
    BodyStateTags: "신체 태그",
    SceneTags: "장면 태그",
    ScenePhaseTags: "장면 단계",
    CompositionTags: "구도 태그",
    CameraTags: "카메라 태그",
    LightingTags: "조명 태그",
    EnvironmentTags: "환경 태그",
    Location: "위치",
    PhysicalCondition: "컨디션",
    Emotion: "감정",
    Goal: "목표",
    Thought: "생각",
    InnerThought: "생각",
    Intention: "의도",
    Relationship: "관계",
    relationship: "관계",
    goal: "목표",
    state: "상태"
  };
  return labels[value] ?? value;
}

function normalizeRelationshipStatusKey(value: string): string {
  return value.toLowerCase().replace(/\s+/gu, "").replace(/[^\p{L}\p{N}_:-]+/gu, "");
}

function createRelationshipInitials(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) {
    return "?";
  }

  const parts = trimmed.split(/\s+/u).filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0][0] ?? ""}${parts[1][0] ?? ""}`.toUpperCase();
  }

  return trimmed.slice(0, 2).toUpperCase();
}

function createRelationshipSlug(value: string): string {
  return value.toLowerCase().replace(/\s+/gu, "-").replace(/[^\p{L}\p{N}_-]+/gu, "").slice(0, 48) || "unknown";
}

function NeuralMapPanel({
  state,
  isSending,
  pendingUserText,
  onNotify,
  onStateChange
}: {
  state: AppState;
  isSending: boolean;
  pendingUserText: string;
  onNotify?: (message: string) => void;
  onStateChange: Dispatch<SetStateAction<AppState>>;
}) {
  const [liveGraph, setLiveGraph] = useState<NeuralMapLiveGraph>();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [savingNodeId, setSavingNodeId] = useState<string>();
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string>();
  const latestContextPack = state.contextPacks.at(-1);
  const latestMemory = state.memoryEvents.at(-1);
  const latestMessage = state.messages.at(-1);
  const graphSignature = useMemo(
    () =>
      [
        state.simulation.id,
        state.simulation.activeSessionId,
        latestContextPack?.id,
        latestMemory?.id,
        latestMemory?.neuralMapNodeId,
        latestMessage?.id,
        state.memoryEvents.length,
        state.contextPacks.length,
        isSending ? "sending" : "idle",
        refreshNonce
      ].join("|"),
    [
      isSending,
      latestContextPack?.id,
      latestMemory?.id,
      latestMemory?.neuralMapNodeId,
      latestMessage?.id,
      refreshNonce,
      state.contextPacks.length,
      state.memoryEvents.length,
      state.simulation.activeSessionId,
      state.simulation.id
    ]
  );
  const liveQuery = useMemo(
    () =>
      [
        isSending ? pendingUserText.trim() : undefined,
        latestMessage?.content,
        latestMemory?.content,
        latestContextPack?.objective
      ]
        .filter((item): item is string => Boolean(item?.trim()))
        .join("\n")
        .slice(0, 1200),
    [isSending, latestContextPack?.objective, latestMemory?.content, latestMessage?.content, pendingUserText]
  );
  useEffect(() => {
    let active = true;

    async function loadGraph() {
      setIsRefreshing(true);
      try {
        const graph = await new NeuralMapClient(state.neuralMap).getLiveGraph(state, liveQuery);
        if (active) {
          setLiveGraph(graph);
        }
      } finally {
        if (active) {
          setIsRefreshing(false);
        }
      }
    }

    void loadGraph();

    return () => {
      active = false;
    };
  }, [graphSignature, liveQuery, state]);

  const graphForDisplay = useMemo(
    () => (isSending && pendingUserText.trim() && liveGraph ? withPendingNeuralMapNode(liveGraph, state, pendingUserText.trim()) : liveGraph),
    [isSending, liveGraph, pendingUserText, state]
  );
  const selectedNode = useMemo(
    () => graphForDisplay?.nodes.find((node) => node.id === selectedNodeId) ?? graphForDisplay?.nodes[0],
    [graphForDisplay, selectedNodeId]
  );
  const selectedEdges = useMemo(
    () => graphForDisplay?.edges.filter((edge) => edge.from === selectedNode?.id || edge.to === selectedNode?.id) ?? [],
    [graphForDisplay, selectedNode?.id]
  );
  const graphSourceLabel = graphForDisplay?.source === "neuralmap" ? "NeuralMap 연결" : "로컬 미러";
  const recentEvents = state.memoryEvents.slice(-5).reverse();

  useEffect(() => {
    if (!graphForDisplay?.nodes.length) {
      setSelectedNodeId(undefined);
      return;
    }

    setSelectedNodeId((current) => {
      if (current && graphForDisplay.nodes.some((node) => node.id === current)) {
        return current;
      }

      return graphForDisplay.seedNodeIds.find((nodeId) => graphForDisplay.nodes.some((node) => node.id === nodeId)) ?? graphForDisplay.nodes[0]?.id;
    });
  }, [graphForDisplay]);

  const saveNodeEdit = useCallback(
    async (node: NeuralMapLiveNode, draft: NeuralMapNodeEditDraft) => {
      setSavingNodeId(node.id);
      try {
        const result = applyNeuralMapNodeEdit(state, node, draft);
        onStateChange(result.state);

        if (state.neuralMap.enabled) {
          const client = new NeuralMapClient(state.neuralMap);
          if (result.syncTarget.kind === "prompt_module") {
            const editedModule = result.syncTarget.module;
            if (isSemanticRetrievalModule(editedModule)) {
              await client.syncPromptModuleDocuments(result.state, new Set([editedModule.id]));
              onNotify?.("서브 프롬프트를 NeuralMap에 의미 검색용으로 인덱싱했습니다.");
            } else {
              onNotify?.("프롬프트 모듈 로컬 설정이 업데이트되었습니다.");
            }
          } else if (result.syncTarget.kind === "memory_event") {
            const memoryTarget = result.syncTarget.event;
            const neuralMapNodeId = await client.ingestEvent(memoryTarget, result.state);
            if (neuralMapNodeId && neuralMapNodeId !== memoryTarget.neuralMapNodeId) {
              onStateChange((current) => ({
                ...current,
                memoryEvents: current.memoryEvents.map((event) =>
                  event.id === memoryTarget.id
                    ? {
                        ...event,
                        neuralMapNodeId
                      }
                    : event
                )
              }));
            }
            onNotify?.("NeuralMap 기록 노드가 저장되었습니다.");
          } else {
            await client.upsertGraphDocument(result.state, result.syncTarget.document);
            onNotify?.("NeuralMap 기록 노드가 저장되었습니다.");
          }
        } else {
          onNotify?.("로컬 RAG 미러에 노드 편집이 저장되었습니다.");
        }

        setRefreshNonce((current) => current + 1);
      } finally {
        setSavingNodeId(undefined);
      }
    },
    [onNotify, onStateChange, state]
  );

  return (
    <div className="panel-stack neural-map-panel">
      <div className="panel-header compact">
        <div>
          <span className="section-kicker">Neural Context</span>
          <h2>참조 맵</h2>
        </div>
        <StatusPill icon={<Database size={15} />} label={state.neuralMap.enabled ? "API 연결" : "로컬 미러"} tone={state.neuralMap.enabled ? "good" : "neutral"} />
      </div>

      <div className="neural-map-live-card">
        <div className="runtime-model-heading">
          <div>
            <span>이번 턴 참조</span>
            <strong>{graphSourceLabel}</strong>
          </div>
          <button className="icon-button" type="button" onClick={() => setRefreshNonce((current) => current + 1)} aria-label="참조 맵 새로고침">
            {isRefreshing ? <Activity size={15} /> : <RefreshCcw size={15} />}
          </button>
        </div>
        {graphForDisplay ? (
          <>
            <NeuralMapGraphCanvas
              graph={graphForDisplay}
              selectedNodeId={selectedNode?.id}
              onSelectNode={setSelectedNodeId}
            />
            <NeuralMapNodeInspector
              state={state}
              graph={graphForDisplay}
              node={selectedNode}
              edges={selectedEdges}
              onSelectNode={setSelectedNodeId}
              onSaveNode={saveNodeEdit}
              saving={savingNodeId === selectedNode?.id}
            />
          </>
        ) : null}
        <div className="runtime-summary-grid">
          <div>
            <span>Nodes</span>
            <strong>{graphForDisplay?.nodes.length ?? 0}</strong>
          </div>
          <div>
            <span>Edges</span>
            <strong>{graphForDisplay?.edges.length ?? 0}</strong>
          </div>
          <div>
            <span>Session</span>
            <strong>{state.simulation.activeSessionId.replace("session_", "")}</strong>
          </div>
          <div>
            <span>Budget</span>
            <strong>{state.neuralMap.tokenBudget}</strong>
          </div>
        </div>
        {isSending ? <p className="settings-note">현재 턴 입력을 임시 노드로 표시하고 있습니다. 응답이 완료되면 저장된 NeuralMap 이벤트와 Context Pack으로 다시 갱신됩니다.</p> : null}
        {graphForDisplay?.error ? <p className="settings-note">NeuralMap API 응답 대신 로컬 미러를 표시 중입니다: {graphForDisplay.error}</p> : null}
      </div>

      {latestContextPack ? (
        <div className="context-pack">
          <strong>{latestContextPack.source === "neuralmap" ? "최근 Context Pack" : "최근 로컬 Context Pack"}</strong>
          <div className="evidence-row">
            <span>{latestContextPack.evidence.length}</span>
            <p>{latestContextPack.objective}</p>
            <small>{new Date(latestContextPack.createdAt).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}</small>
          </div>
          {latestContextPack.evidence.slice(0, 3).map((item) => (
            <div className="evidence-row" key={`${latestContextPack.id}-neural-${item.nodeId}`}>
              <span>{Math.round(item.score * 100)}%</span>
              <p>{item.snippet}</p>
              <small>{item.nodeId}</small>
            </div>
          ))}
        </div>
      ) : null}

      <div className="context-pack">
        <strong>최근 저장 이벤트</strong>
        {recentEvents.length > 0 ? (
          recentEvents.map((event) => (
            <div className="evidence-row" key={`neural-event-${event.id}`}>
              <span>{event.neuralMapNodeId ? "NM" : "LOCAL"}</span>
              <p>{event.content}</p>
              <small>{event.neuralMapNodeId ?? event.id}</small>
            </div>
          ))
        ) : (
          <div className="empty-panel">
            <Brain size={17} />
            <span>아직 저장된 시뮬레이션 이벤트가 없습니다.</span>
          </div>
        )}
      </div>
    </div>
  );
}

type NeuralMapViewport = {
  x: number;
  y: number;
  scale: number;
};

type NeuralMapPosition = {
  x: number;
  y: number;
};

type NeuralMapDragState =
  | {
      mode: "pan";
      pointerId: number;
      startX: number;
      startY: number;
      originX: number;
      originY: number;
      moved: boolean;
    }
  | {
      mode: "node";
      pointerId: number;
      nodeId: string;
      startX: number;
      startY: number;
      originX: number;
      originY: number;
      rectWidth: number;
      rectHeight: number;
      viewportScale: number;
      moved: boolean;
    };

function NeuralMapGraphCanvas({
  graph,
  selectedNodeId,
  onSelectNode
}: {
  graph: NeuralMapLiveGraph;
  selectedNodeId?: string;
  onSelectNode: (nodeId: string) => void;
}) {
  const [viewport, setViewport] = useState<NeuralMapViewport>({ x: 0, y: 0, scale: 1 });
  const [dragMode, setDragMode] = useState<"none" | "pan" | "node">("none");
  const [nodePositions, setNodePositions] = useState<Record<string, NeuralMapPosition>>({});
  const layout = useMemo(
    () => createNeuralMapLayout(graph, selectedNodeId, nodePositions),
    [graph, nodePositions, selectedNodeId]
  );
  const canvasRef = useRef<HTMLDivElement>(null);
  const dragStateRef = useRef<NeuralMapDragState | undefined>(undefined);
  const recentlyDraggedRef = useRef(false);

  useEffect(() => {
    setViewport({ x: 0, y: 0, scale: 1 });
  }, [graph.generatedAt, graph.nodes.length]);

  useEffect(() => {
    const liveNodeIds = new Set(graph.nodes.map((node) => node.id));
    setNodePositions((current) => {
      let changed = false;
      const next: Record<string, NeuralMapPosition> = {};

      Object.entries(current).forEach(([nodeId, position]) => {
        if (liveNodeIds.has(nodeId)) {
          next[nodeId] = position;
        } else {
          changed = true;
        }
      });

      return changed ? next : current;
    });
  }, [graph.nodes]);

  const zoomGraph = useCallback((delta: number) => {
    setViewport((current) => ({
      ...current,
      scale: clampGraphScale(current.scale + delta)
    }));
  }, []);

  const resetViewport = useCallback(() => {
    setViewport({ x: 0, y: 0, scale: 1 });
  }, []);

  const handlePointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return;
    }

    const target = event.target instanceof HTMLElement ? event.target.closest<HTMLElement>("[data-neural-node-id]") : null;
    if (target) {
      return;
    }

    event.preventDefault();
    dragStateRef.current = {
      mode: "pan",
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: viewport.x,
      originY: viewport.y,
      moved: false
    };
    setDragMode("pan");
  }, [viewport.x, viewport.y]);

  useEffect(() => {
    if (dragMode === "none") {
      return undefined;
    }

    const handlePointerMove = (event: PointerEvent) => {
      const dragState = dragStateRef.current;
      if (!dragState || dragState.pointerId !== event.pointerId) {
        return;
      }

      event.preventDefault();
      const deltaX = event.clientX - dragState.startX;
      const deltaY = event.clientY - dragState.startY;
      const moved = Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3;
      dragState.moved ||= moved;

      if (!moved && !dragState.moved) {
        return;
      }

      if (dragState.mode === "pan") {
        setViewport((current) => ({
          ...current,
          x: dragState.originX + deltaX,
          y: dragState.originY + deltaY
        }));
        return;
      }

      const scale = Math.max(0.1, dragState.viewportScale);
      const nextX = clampNodePosition(dragState.originX + (deltaX / Math.max(1, dragState.rectWidth * scale)) * 100);
      const nextY = clampNodePosition(dragState.originY + (deltaY / Math.max(1, dragState.rectHeight * scale)) * 100);

      setNodePositions((current) => ({
        ...current,
        [dragState.nodeId]: { x: nextX, y: nextY }
      }));
    };

    const finishPointerInteraction = (event: PointerEvent) => {
      const dragState = dragStateRef.current;
      if (!dragState || dragState.pointerId !== event.pointerId) {
        return;
      }

      dragStateRef.current = undefined;
      setDragMode("none");

      if (dragState.moved) {
        recentlyDraggedRef.current = true;
        window.setTimeout(() => {
          recentlyDraggedRef.current = false;
        }, 0);
      }
    };

    window.addEventListener("pointermove", handlePointerMove, { passive: false });
    window.addEventListener("pointerup", finishPointerInteraction);
    window.addEventListener("pointercancel", finishPointerInteraction);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", finishPointerInteraction);
      window.removeEventListener("pointercancel", finishPointerInteraction);
    };
  }, [dragMode]);

  const handleNodePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>, nodeId: string, x: number, y: number) => {
      if (event.button !== 0) {
        return;
      }

      const canvasRect = canvasRef.current?.getBoundingClientRect();
      event.preventDefault();
      event.stopPropagation();
      event.currentTarget.blur();
      onSelectNode(nodeId);

      dragStateRef.current = {
        mode: "node",
        pointerId: event.pointerId,
        nodeId,
        startX: event.clientX,
        startY: event.clientY,
        originX: x,
        originY: y,
        rectWidth: canvasRect?.width ?? 1,
        rectHeight: canvasRect?.height ?? 1,
        viewportScale: viewport.scale,
        moved: false
      };
      setDragMode("node");
    },
    [onSelectNode, viewport.scale]
  );

  const handleWheel = useCallback((event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.08 : 0.08;
    zoomGraph(delta);
  }, [zoomGraph]);

  const handleNodeClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>, nodeId: string) => {
      if (recentlyDraggedRef.current) {
        event.preventDefault();
        return;
      }

      onSelectNode(nodeId);
    },
    [onSelectNode]
  );

  const preventNativeDrag = useCallback((event: ReactDragEvent<HTMLElement>) => {
    event.preventDefault();
  }, []);

  if (layout.nodes.length === 0) {
    return (
      <div className="neural-map-canvas empty">
        <Database size={18} />
        <span>표시할 노드가 없습니다.</span>
      </div>
    );
  }

  return (
    <div
      className={`neural-map-canvas ${dragMode === "pan" ? "dragging" : ""} ${dragMode === "node" ? "node-dragging" : ""}`}
      aria-label="NeuralMap live graph"
      onDragStart={preventNativeDrag}
      onPointerDown={handlePointerDown}
      onWheel={handleWheel}
      ref={canvasRef}
    >
      <div className="neural-map-canvas-toolbar" onPointerDown={(event) => event.stopPropagation()}>
        <button type="button" onClick={() => zoomGraph(-0.12)} aria-label="그래프 축소">
          <Minus size={14} />
        </button>
        <button type="button" onClick={resetViewport} aria-label="그래프 위치 초기화">
          <RefreshCcw size={14} />
        </button>
        <button type="button" onClick={() => zoomGraph(0.12)} aria-label="그래프 확대">
          <Plus size={14} />
        </button>
      </div>
      <div
        className="neural-map-plane"
        style={{
          transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`
        }}
      >
        <svg aria-hidden="true" viewBox="0 0 100 100" preserveAspectRatio="none">
          {layout.edges.map(({ edge, from, to }) => (
            <line
              className={edge.from === selectedNodeId || edge.to === selectedNodeId ? "selected" : undefined}
              key={edge.id}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              strokeWidth={Math.max(0.4, edge.confidence * 1.6)}
            />
          ))}
        </svg>
        {layout.nodes.map(({ node, seed, x, y }) => (
          <button
            className={`neural-map-node ${seed ? "seed" : ""} ${node.id === selectedNodeId ? "selected" : ""} ${
              dragMode === "node" && dragStateRef.current?.mode === "node" && dragStateRef.current.nodeId === node.id ? "dragging" : ""
            } ${getNeuralMapNodeClass(node)}`}
            data-neural-node-id={node.id}
            draggable={false}
            key={node.id}
            onClick={(event) => handleNodeClick(event, node.id)}
            onDragStart={preventNativeDrag}
            onPointerDown={(event) => handleNodePointerDown(event, node.id, x, y)}
            style={{ left: `${x}%`, top: `${y}%` }}
            title={`${node.title}\n${node.summary}`}
            type="button"
          >
            <span>{node.type.slice(0, 1)}</span>
            <strong>{limitText(node.title, 26)}</strong>
            <small>{node.kind ?? node.type}</small>
          </button>
        ))}
      </div>
    </div>
  );
}

function NeuralMapNodeInspector({
  state,
  graph,
  node,
  edges,
  onSelectNode,
  onSaveNode,
  saving
}: {
  state: AppState;
  graph: NeuralMapLiveGraph;
  node?: NeuralMapLiveNode;
  edges: NeuralMapLiveGraph["edges"];
  onSelectNode: (nodeId: string) => void;
  onSaveNode: (node: NeuralMapLiveNode, draft: NeuralMapNodeEditDraft) => Promise<void>;
  saving: boolean;
}) {
  const nodesById = useMemo(() => new Map(graph.nodes.map((candidate) => [candidate.id, candidate])), [graph.nodes]);
  const resolvedDraft = useMemo(() => (node ? createNeuralMapNodeEditDraft(state, node) : undefined), [node, state]);
  const [draft, setDraft] = useState<NeuralMapNodeEditDraft | undefined>(resolvedDraft);

  useEffect(() => {
    setDraft(resolvedDraft);
  }, [resolvedDraft]);

  if (!node) {
    return (
      <div className="neural-map-inspector empty-panel">
        <Info size={17} />
        <span>노드를 클릭하면 상세 내용이 여기에 표시됩니다.</span>
      </div>
    );
  }

  const baselineDraft = resolvedDraft ?? createNeuralMapNodeEditDraft(state, node);
  const editorDraft = draft ?? baselineDraft;
  const dirty =
    editorDraft.title !== baselineDraft.title ||
    editorDraft.content !== baselineDraft.content ||
    editorDraft.tags !== baselineDraft.tags ||
    editorDraft.importance !== baselineDraft.importance ||
    editorDraft.tokenPolicy !== baselineDraft.tokenPolicy ||
    editorDraft.priority !== baselineDraft.priority ||
    editorDraft.enabled !== baselineDraft.enabled;
  const patchDraft = (patch: Partial<NeuralMapNodeEditDraft>) => {
    setDraft((current) => ({
      ...(current ?? editorDraft),
      ...patch
    }));
  };

  return (
    <article className="neural-map-inspector">
      <div className="neural-map-inspector-head">
        <span>{node.type}</span>
        <strong>{node.title}</strong>
      </div>
      <p>{node.summary || "요약 내용이 없습니다."}</p>
      <div className="neural-map-node-meta">
        <span>{node.kind ?? "graph_node"}</span>
        <span>{node.sourceSystem ?? graph.source}</span>
        <span>{Math.round(node.importanceScore * 100)}%</span>
      </div>
      <div className="neural-map-node-id">{node.id}</div>
      <div className="neural-map-editor">
        <div className="runtime-card-subhead">
          <strong>RAG 노드 편집</strong>
          <span>{editorDraft.editableKind.replaceAll("_", " ")}</span>
        </div>
        <label>
          노드 제목
          <input value={editorDraft.title} onChange={(event) => patchDraft({ title: event.target.value })} />
        </label>
        <label>
          저장 내용
          <textarea value={editorDraft.content} onChange={(event) => patchDraft({ content: event.target.value })} />
        </label>
        <div className="two-fields">
          <label>
            태그
            <input value={editorDraft.tags} onChange={(event) => patchDraft({ tags: event.target.value })} />
          </label>
          <label>
            중요도 <span className="inline-value">{Math.round(editorDraft.importance * 100)}%</span>
            <input
              min="0"
              max="1"
              step="0.01"
              type="range"
              value={editorDraft.importance}
              onChange={(event) => patchDraft({ importance: Number(event.target.value), priority: Math.round(Number(event.target.value) * 100) })}
            />
          </label>
        </div>
        {editorDraft.editableKind === "prompt_module" || editorDraft.editableKind === "graph_document" ? (
          <div className="two-fields">
            <label>
              참조 방식
              <select value={editorDraft.tokenPolicy} onChange={(event) => patchDraft({ tokenPolicy: event.target.value as TokenPolicy })}>
                {tokenPolicies.map((policy) => (
                  <option key={policy} value={policy}>
                    {tokenPolicyLabels[policy]}
                  </option>
                ))}
              </select>
            </label>
            <label>
              우선순위
              <input
                min="0"
                max="120"
                type="number"
                value={editorDraft.priority}
                onChange={(event) => patchDraft({ priority: Number(event.target.value), importance: Math.min(1, Math.max(0, Number(event.target.value) / 100)) })}
              />
            </label>
          </div>
        ) : null}
        <label className="checkline">
          <input checked={editorDraft.enabled} type="checkbox" onChange={(event) => patchDraft({ enabled: event.target.checked })} />
          진행 문맥에 사용
        </label>
        <button
          className="send-button wide-action"
          type="button"
          disabled={saving || !dirty || !editorDraft.content.trim()}
          onClick={() => void onSaveNode(node, editorDraft)}
        >
          <Database size={16} />
          {saving ? "저장 중" : "RAG 노드 저장"}
        </button>
      </div>
      {edges.length > 0 ? (
        <div className="neural-map-relation-list">
          <strong>연결 관계</strong>
          {edges.slice(0, 5).map((edge) => {
            const neighborId = edge.from === node.id ? edge.to : edge.from;
            const neighbor = nodesById.get(neighborId);
            return (
              <button key={edge.id} type="button" onClick={() => onSelectNode(neighborId)}>
                <span>{edge.type}</span>
                <p>{neighbor?.title ?? neighborId}</p>
                <small>{Math.round(edge.confidence * 100)}%</small>
              </button>
            );
          })}
        </div>
      ) : null}
    </article>
  );
}

function clampGraphScale(value: number): number {
  return Math.min(1.7, Math.max(0.72, Math.round(value * 100) / 100));
}

function clampNodePosition(value: number): number {
  return Math.min(94, Math.max(6, Math.round(value * 10) / 10));
}

function createNeuralMapLayout(
  graph: NeuralMapLiveGraph,
  selectedNodeId?: string,
  positionOverrides: Record<string, NeuralMapPosition> = {}
): {
  nodes: Array<{ node: NeuralMapLiveNode; x: number; y: number; seed: boolean }>;
  edges: Array<{
    edge: NeuralMapLiveGraph["edges"][number];
    from: { x: number; y: number };
    to: { x: number; y: number };
  }>;
} {
  const seedIds = new Set(graph.seedNodeIds);
  const rankedNodes = graph.nodes
    .slice()
    .sort((a, b) => Number(seedIds.has(b.id)) - Number(seedIds.has(a.id)) || b.importanceScore - a.importanceScore);
  const nodes = rankedNodes.slice(0, 16);
  const selectedNode = selectedNodeId ? graph.nodes.find((node) => node.id === selectedNodeId) : undefined;

  if (selectedNode && !nodes.some((node) => node.id === selectedNode.id)) {
    if (nodes.length >= 16) {
      nodes[nodes.length - 1] = selectedNode;
    } else {
      nodes.push(selectedNode);
    }
  }
  const positions = nodes.map((node, index) => {
    if (index === 0) {
      return { node, x: 50, y: 50, seed: seedIds.has(node.id) };
    }

    const ringIndex = index - 1;
    const radius = index <= 7 ? 28 : 34;
    const angle = -Math.PI / 2 + (ringIndex / Math.max(1, nodes.length - 1)) * Math.PI * 2;
    return {
      node,
      x: Math.round((50 + Math.cos(angle) * radius) * 10) / 10,
      y: Math.round((50 + Math.sin(angle) * radius) * 10) / 10,
      seed: seedIds.has(node.id)
    };
  });
  const positionedNodes = positions.map((position) => {
    const override = positionOverrides[position.node.id];
    return override ? { ...position, x: override.x, y: override.y } : position;
  });
  const positionByNodeId = new Map(positionedNodes.map((position) => [position.node.id, position]));
  const edges = graph.edges
    .map((edge) => {
      const from = positionByNodeId.get(edge.from);
      const to = positionByNodeId.get(edge.to);
      return from && to ? { edge, from, to } : undefined;
    })
    .filter((edge): edge is NonNullable<typeof edge> => Boolean(edge))
    .slice(0, 24);

  return { nodes: positionedNodes, edges };
}

function withPendingNeuralMapNode(graph: NeuralMapLiveGraph, state: AppState, pendingUserText: string): NeuralMapLiveGraph {
  const pendingNodeId = `dynamicchat:pending:${state.simulation.activeSessionId}:${state.messages.length}`;
  const sessionNodeId = `simulation:${state.simulation.id}:session:${state.simulation.activeSessionId}`;
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const sessionNode = nodesById.get(sessionNodeId) ?? {
    id: sessionNodeId,
    type: "Session",
    title: state.simulation.title,
    summary: state.simulation.activeSessionId,
    importanceScore: 0.82,
    sourceSystem: "dynamicchat",
    kind: "simulation_session"
  };

  nodesById.set(sessionNodeId, sessionNode);
  nodesById.set(pendingNodeId, {
    id: pendingNodeId,
    type: "Run",
    title: "진행 중인 턴",
    summary: pendingUserText,
    importanceScore: 1,
    sourceSystem: "dynamicchat",
    kind: "pending_turn"
  });

  return {
    ...graph,
    nodes: [...nodesById.values()],
    edges: [
      ...graph.edges,
      {
        id: `edge:${sessionNodeId}:references:${pendingNodeId}`,
        from: sessionNodeId,
        to: pendingNodeId,
        type: "references",
        weight: 1,
        confidence: 0.92
      }
    ],
    seedNodeIds: [pendingNodeId, ...graph.seedNodeIds]
  };
}

function getNeuralMapNodeClass(node: NeuralMapLiveNode): string {
  return `type-${node.type.toLowerCase().replaceAll(/[^a-z0-9]+/gu, "-")}`;
}

function limitText(value: string, maxLength: number): string {
  return value.length > maxLength ? `${value.slice(0, maxLength - 1)}…` : value;
}

type OperationalFilter = "all" | "image_jobs" | "suppressed" | "continuity" | "warnings";

function OperationalPanel({
  state,
  onFeedback
}: {
  state: AppState;
  onFeedback: (assetId: string, rating: ImageFeedbackRating) => void;
}) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<OperationalFilter>("all");
  const normalizedQuery = query.trim().toLowerCase();
  const traces = useMemo(
    () =>
      state.turnTraces
        .slice()
        .reverse()
        .filter((trace) => matchesOperationalFilter(trace, state, filter))
        .filter((trace) => matchesTraceQuery(trace, state, normalizedQuery)),
    [filter, normalizedQuery, state]
  );
  const events = useMemo(() => buildOperationalEvents(state, filter, normalizedQuery), [filter, normalizedQuery, state]);
  const evaluations = useMemo(() => state.evaluationScenarios.map((scenario) => evaluateScenario(scenario, state)), [state]);
  const latestTrace = state.turnTraces.at(-1);
  const summary = createOperationalSummary(state, latestTrace);

  return (
    <div className="panel-stack operational-panel">
      <div className="panel-header compact">
        <div>
          <span className="section-kicker">운영 기록</span>
          <h2>인스펙터</h2>
        </div>
        <StatusPill icon={<BarChart3 size={15} />} label={`추적 ${state.turnTraces.length}개`} tone="neutral" />
      </div>

      <div className="ops-metrics">
        <Metric label="검색 문맥" value={summary.tokenBudget} />
        <Metric label="토큰 절감" value={summary.ragSaved} />
        <Metric label="지연 시간" value={summary.latency} />
        <Metric label="이미지" value={summary.images} />
      </div>

      <div className="ops-filters">
        <label className="search-field">
          <Search size={15} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="추적, 메모리, 이미지, 문맥 검색" />
        </label>
        <select value={filter} onChange={(event) => setFilter(event.target.value as OperationalFilter)}>
          <option value="all">전체 이벤트</option>
          <option value="image_jobs">이미지 작업</option>
          <option value="suppressed">생성 생략</option>
          <option value="continuity">연속성</option>
          <option value="warnings">경고</option>
        </select>
      </div>

      <div className="trace-list">
        {traces.length > 0 ? (
          traces.slice(0, 5).map((trace) => <TurnTraceCard key={trace.id} state={state} trace={trace} />)
        ) : (
          <div className="empty-panel">
            <Activity size={17} />
            <span>추적 기록 없음</span>
          </div>
        )}
      </div>

      <div className="context-pack">
        <strong>이벤트 흐름</strong>
        {events.slice(0, 7).map((event) => (
          <div className="evidence-row" key={event.id}>
            <span>{formatOperationalEventType(event.type)}</span>
            <p>{event.title}</p>
            <small>{formatOperationalStatus(event.status)} · {event.detail}</small>
          </div>
        ))}
      </div>

      <div className="context-pack">
        <strong>회수 평가</strong>
        {evaluations.map((result) => (
          <div className="eval-row" key={result.scenario.id}>
            <ClipboardCheck size={15} />
            <span>
              <strong>{result.scenario.label}</strong>
              <small>{formatOperationalStatus(result.status)} · {Math.round(result.score * 100)}%</small>
              <p>{result.evidence}</p>
            </span>
          </div>
        ))}
      </div>

      <div className="feedback-list">
        {state.imageAssets
          .filter((asset) => asset.source === "generated")
          .slice(-3)
          .reverse()
          .map((asset) => (
            <ImageFeedbackRow asset={asset} key={`ops-${asset.id}`} onFeedback={onFeedback} />
          ))}
      </div>
    </div>
  );
}

function TurnTraceCard({ state, trace }: { state: AppState; trace: TurnTrace }) {
  const userMessage = state.messages.find((message) => message.id === trace.userMessageId);
  const assistantMessage = state.messages.find((message) => message.id === trace.assistantMessageId);
  const contextPack = state.contextPacks.find((pack) => pack.id === trace.contextPackId);
  const promptUsages = state.promptModuleUsages.filter((usage) => trace.promptModuleUsageIds.includes(usage.id));
  const sidecarTrace = state.sidecarTraces.find((sidecar) => sidecar.id === trace.sidecarTraceId);
  const imageJob = trace.imageJobId ? state.imageJobs.find((job) => job.id === trace.imageJobId) : undefined;
  const memoryEvents = state.memoryEvents.filter((event) => trace.memoryEventIds.includes(event.id));
  const suppression = trace.imageCue.suppressionReason ?? (trace.imageJobId ? undefined : "작업 미생성");

  return (
    <article className="trace-card">
      <div className="trace-card-header">
        <strong>{formatCompactTime(trace.createdAt)}</strong>
        <span>{imageJob ? getImageJobStatusLabel(imageJob.status) : suppression ? "생략" : "기본 에셋"}</span>
      </div>
      <div className="trace-dialogue">
        <p>{userMessage?.content ?? "사용자 메시지 없음"}</p>
        <p>{assistantMessage?.content ?? "응답 메시지 없음"}</p>
      </div>
      <div className="trace-metric-grid">
        <Metric label="모듈" value={trace.metrics.selectedModuleCount.toString()} />
        <Metric label="근거" value={trace.metrics.contextEvidenceCount.toString()} />
        <Metric label="사용 토큰" value={`${trace.metrics.selectedModuleTokenEstimate + trace.metrics.contextTokenEstimate}`} />
        <Metric label="LLM" value={`${trace.metrics.llmRequestMs}ms`} />
        <Metric label="RAG" value={`${trace.metrics.retrievalLatencyMs}ms`} />
        <Metric label="전체" value={`${trace.metrics.turnLatencyMs}ms`} />
      </div>
      <div className="trace-section">
        <strong>문맥</strong>
        <small>{contextPack?.source ?? "누락"} · {contextPack?.objective ?? trace.contextPackId}</small>
        {contextPack?.evidence.slice(0, 2).map((item) => (
          <p key={`${trace.id}-${item.nodeId}`}>{Math.round(item.score * 100)}% · {item.snippet}</p>
        ))}
      </div>
      <div className="trace-section">
        <strong>프롬프트 모듈</strong>
        {promptUsages.slice(0, 4).map((usage) => (
          <p key={usage.id}>{usage.moduleTitle} · {usage.source} · {Math.round(usage.score * 100)}%</p>
        ))}
      </div>
      <div className="trace-section">
        <strong>사이드카와 메모리</strong>
        <small>{sidecarTrace?.status ?? "누락"} · 메모리 {memoryEvents.length}개</small>
        {memoryEvents.slice(0, 2).map((event) => (
          <p key={event.id}>{event.tags.join(", ")} · {event.content}</p>
        ))}
      </div>
      <div className="trace-section">
        <strong>이미지 단서</strong>
        <small>{trace.imageCue.reason}</small>
        <p>{imageJob ? `${getImageJobStatusLabel(imageJob.status)} · ${imageJob.reason}` : suppression}</p>
        {imageJob?.policyWarnings?.length ? <p>{imageJob.policyWarnings.join(" / ")}</p> : null}
      </div>
    </article>
  );
}

interface OperationalEvent {
  id: string;
  type: string;
  title: string;
  detail: string;
  status: string;
  createdAt: string;
}

function formatOperationalEventType(type: string): string {
  const labels: Record<string, string> = {
    memory: "메모리",
    context: "문맥",
    image: "이미지",
    handoff: "인계",
    check: "점검",
    audit: "감사",
    redact: "삭제"
  };
  return labels[type] ?? type;
}

function formatOperationalStatus(status: string): string {
  const labels: Record<string, string> = {
    ready: "준비",
    warning: "주의",
    passed: "통과",
    failed: "실패",
    canceled: "취소",
    completed: "완료",
    queued: "대기",
    planning: "준비",
    generating: "생성 중"
  };
  return labels[status] ?? status;
}

function buildOperationalEvents(state: AppState, filter: OperationalFilter, query: string): OperationalEvent[] {
  const events: OperationalEvent[] = [
    ...state.memoryEvents.map((event) => ({
      id: event.id,
      type: "memory",
      title: event.content,
      detail: event.tags.join(", "),
      status: `${Math.round(event.importance * 100)}%`,
      createdAt: event.createdAt
    })),
    ...state.contextPacks.map((pack) => ({
      id: pack.id,
      type: "context",
      title: pack.objective,
      detail: `${pack.source} · 근거 ${pack.evidence.length}개`,
      status: `${pack.tokenBudget}토큰`,
      createdAt: pack.createdAt
    })),
    ...state.imageJobs.map((job) => ({
      id: job.id,
      type: "image",
      title: job.reason,
      detail: job.prompt.slice(0, 140) || job.error || "",
      status: job.status,
      createdAt: job.updatedAt ?? job.completedAt ?? job.createdAt
    })),
    ...state.handoffs.map((handoff) => ({
      id: handoff.id,
      type: "handoff",
      title: handoff.summary,
      detail: `${handoff.previousSessionId} -> ${handoff.nextSessionId}`,
      status: handoff.source,
      createdAt: handoff.createdAt
    })),
    ...state.continuityChecks.map((check) => ({
      id: check.id,
      type: "check",
      title: check.facts.map((fact) => fact.expected).join(" / "),
      detail: check.warnings.join(" / ") || `사실 ${check.facts.length}개`,
      status: check.status,
      createdAt: check.checkedAt
    })),
    ...state.auditLog.map((event) => ({
      id: event.id,
      type: "audit",
      title: `${event.action} · ${event.resourceType}`,
      detail: JSON.stringify(event.metadata).slice(0, 180),
      status: event.scope.projectId,
      createdAt: event.createdAt
    })),
    ...state.redactionQueue.map((request) => ({
      id: request.id,
      type: "redact",
      title: `${request.targetType} · ${request.targetId}`,
      detail: request.reason,
      status: request.status,
      createdAt: request.completedAt ?? request.createdAt
    }))
  ];

  return events
    .filter((event) => matchesEventFilter(event, filter))
    .filter((event) => !query || `${event.type} ${event.title} ${event.detail} ${event.status}`.toLowerCase().includes(query))
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
}

function matchesEventFilter(event: OperationalEvent, filter: OperationalFilter): boolean {
  if (filter === "all") {
    return true;
  }
  if (filter === "image_jobs") {
    return event.type === "image";
  }
  if (filter === "continuity") {
    return event.type === "handoff" || event.type === "check";
  }
  if (filter === "warnings") {
    return ["failed", "warning", "canceled"].includes(event.status);
  }
  return true;
}

function matchesOperationalFilter(trace: TurnTrace, state: AppState, filter: OperationalFilter): boolean {
  if (filter === "all") {
    return true;
  }
  const job = trace.imageJobId ? state.imageJobs.find((candidate) => candidate.id === trace.imageJobId) : undefined;
  if (filter === "image_jobs") {
    return Boolean(job);
  }
  if (filter === "suppressed") {
    return !job && Boolean(trace.imageCue.suppressionReason);
  }
  if (filter === "warnings") {
    return job?.status === "failed" || Boolean(job?.policyWarnings?.length) || Boolean(trace.imageCue.suppressionReason?.includes("정책"));
  }
  if (filter === "continuity") {
    return state.continuityChecks.some((check) => check.nextSessionId === trace.sessionId || check.previousSessionId === trace.sessionId);
  }
  return true;
}

function matchesTraceQuery(trace: TurnTrace, state: AppState, query: string): boolean {
  if (!query) {
    return true;
  }
  const userMessage = state.messages.find((message) => message.id === trace.userMessageId)?.content ?? "";
  const assistantMessage = state.messages.find((message) => message.id === trace.assistantMessageId)?.content ?? "";
  const contextPack = state.contextPacks.find((pack) => pack.id === trace.contextPackId);
  const promptUsages = state.promptModuleUsages.filter((usage) => trace.promptModuleUsageIds.includes(usage.id));
  const imageJob = trace.imageJobId ? state.imageJobs.find((job) => job.id === trace.imageJobId) : undefined;
  const haystack = [
    userMessage,
    assistantMessage,
    contextPack?.objective,
    ...(contextPack?.evidence.map((item) => `${item.snippet} ${item.reason}`) ?? []),
    ...promptUsages.map((usage) => `${usage.moduleTitle} ${usage.reason}`),
    trace.imageCue.reason,
    trace.imageCue.suppressionReason,
    trace.imageCue.scene,
    trace.imageCue.tags.join(" "),
    imageJob?.reason,
    imageJob?.status
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return haystack.includes(query);
}

function createOperationalSummary(state: AppState, trace?: TurnTrace) {
  const generatedAssets = state.imageAssets.filter((asset) => asset.source === "generated");
  const imageCount = generatedAssets.length;
  const cost = state.imageJobs.reduce((sum, job) => sum + (readProviderCost(job.providerPayload) ?? 0), 0);
  return {
    tokenBudget: `${state.neuralMap.tokenBudget}토큰`,
    ragSaved: trace ? `${trace.metrics.ragTokenSavingsEstimate}토큰` : "0토큰",
    latency: trace ? `${trace.metrics.llmRequestMs}ms / ${trace.metrics.turnLatencyMs}ms` : "0ms",
    images: cost > 0 ? `${imageCount} / ${cost}` : imageCount.toString()
  };
}

function evaluateScenario(scenario: EvaluationScenario, state: AppState) {
  if (scenario.kind === "reset_continuity") {
    const latest = state.continuityChecks.at(-1);
    if (!latest) {
      return {
        scenario,
        status: "ready",
        score: 0,
        evidence: "세션 초기화 점검 대기"
      };
    }
    const passedFacts = latest.facts.filter((fact) => fact.found).length;
    return {
      scenario,
      status: latest.status,
      score: latest.facts.length > 0 ? passedFacts / latest.facts.length : 0,
      evidence: latest.warnings[0] ?? latest.facts[0]?.evidence ?? latest.id
    };
  }

  if (scenario.kind === "image_quality") {
    const feedbackAssets = state.imageAssets.filter((asset) => asset.source === "generated" && asset.feedback);
    const positiveCount = feedbackAssets.filter((asset) => asset.feedback?.rating === "liked").length;
    const rejectedCount = feedbackAssets.filter((asset) => asset.feedback?.rating === "rejected").length;
    const score = feedbackAssets.length > 0 ? Math.max(0, positiveCount + (feedbackAssets.length - rejectedCount) * 0.5) / feedbackAssets.length : 0;
    return {
      scenario,
      status: feedbackAssets.length === 0 ? "ready" : rejectedCount > positiveCount ? "warning" : "passed",
      score,
      evidence: feedbackAssets.at(-1)?.title ?? "피드백 대기"
    };
  }

  const evidenceText = [
    ...state.contextPacks.flatMap((pack) => pack.evidence.map((item) => item.snippet)),
    ...state.memoryEvents.map((event) => `${event.content} ${event.tags.join(" ")}`),
    ...state.messages.map((message) => message.content)
  ].join("\n").toLowerCase();
  const hits = scenario.expectedSignals.filter((signal) => evidenceText.includes(signal.toLowerCase()));
  return {
    scenario,
    status: hits.length === 0 ? "ready" : hits.length >= Math.ceil(scenario.expectedSignals.length * 0.66) ? "passed" : "warning",
    score: scenario.expectedSignals.length > 0 ? hits.length / scenario.expectedSignals.length : 0,
    evidence: hits.join(", ") || "일치 신호 없음"
  };
}

function formatCompactTime(value: string): string {
  return new Date(value).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
}

function PersonalSettingsDialog({
  state,
  vault,
  onClose,
  onLlmChange,
  onNovelAiChange,
  onSaveLlmSecret,
  onSaveNovelAiSecret,
  onSaveImageStoragePath
}: {
  state: AppState;
  vault: PersonalApiVault;
  onClose: () => void;
  onLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onNovelAiChange: (patch: Partial<NovelAiApiSettings>) => void;
  onSaveLlmSecret: (provider: LlmApiSettings["provider"], settings: LlmApiSettings) => void;
  onSaveNovelAiSecret: (settings: NovelAiApiSettings) => void;
  onSaveImageStoragePath: (imageStoragePath: string) => void;
}) {
  const [llmDraft, setLlmDraft] = useState<LlmApiSettings>(() => ({
    ...state.llm,
    ...(vault.llmByProvider[state.llm.provider] ?? {})
  }));
  const [novelAiDraft, setNovelAiDraft] = useState<NovelAiApiSettings>(() => ({
    ...state.novelAi,
    ...vault.novelAi
  }));
  const [isLlmVerifying, setIsLlmVerifying] = useState(false);
  const [isNovelAiVerifying, setIsNovelAiVerifying] = useState(false);
  const [imageStoragePathDraft, setImageStoragePathDraft] = useState(vault.imageStoragePath);
  const llmProvider = getLlmProviderOption(llmDraft.provider);

  const changeLlmProvider = (provider: LlmApiSettings["provider"]) => {
    const providerPatch = createLlmProviderPatch(provider);
    const savedSecret = getPersonalLlmSecret(vault, provider);
    setLlmDraft((current) => ({
      ...current,
      ...providerPatch,
      apiKey: savedSecret.apiKey,
      registrationStatus: savedSecret.registrationStatus,
      verifiedAt: savedSecret.verifiedAt,
      verificationMessage: savedSecret.verificationMessage
    }));
  };

  const saveLlm = (settings = llmDraft) => {
    const secretRecord = normalizeStoredSecretRecord(settings, "개인 LLM 키가 저장되어 있습니다.");
    const normalizedSettings = {
      ...settings,
      ...secretRecord
    };
    onSaveLlmSecret(normalizedSettings.provider, normalizedSettings);
    onLlmChange(normalizedSettings);
  };

  const saveNovelAi = (settings = novelAiDraft) => {
    const secretRecord = normalizeStoredSecretRecord(settings, "개인 NovelAI 토큰이 저장되어 있습니다.");
    const normalizedSettings = {
      ...settings,
      ...secretRecord
    };
    onSaveNovelAiSecret(normalizedSettings);
    onNovelAiChange(normalizedSettings);
  };

  const verifyLlm = async () => {
    setIsLlmVerifying(true);
    const verifyingDraft = {
      ...llmDraft,
      registrationStatus: "verifying" as const,
      verificationMessage: "LLM API 검증 중..."
    };
    setLlmDraft(verifyingDraft);
    try {
      const verifiedPatch = await createVerifiedLlmPatch(verifyingDraft);
      const next = {
        ...verifyingDraft,
        ...verifiedPatch
      };
      setLlmDraft(next);
      saveLlm(next);
    } finally {
      setIsLlmVerifying(false);
    }
  };

  const verifyNovelAi = async () => {
    setIsNovelAiVerifying(true);
    const verifyingDraft = {
      ...novelAiDraft,
      registrationStatus: "verifying" as const,
      verificationMessage: "NovelAI API 검증 중..."
    };
    setNovelAiDraft(verifyingDraft);
    try {
      const verifiedPatch = await createVerifiedNovelAiPatch(verifyingDraft);
      const next = {
        ...verifyingDraft,
        ...verifiedPatch
      };
      setNovelAiDraft(next);
      saveNovelAi(next);
    } finally {
      setIsNovelAiVerifying(false);
    }
  };

  const dialogRef = useModalDialog(true, onClose);

  return (
    <div className="personal-settings-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="personal-settings-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="개인 API 설정"
        ref={dialogRef}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="personal-settings-header">
          <div>
            <span className="section-kicker">개인 설정</span>
            <h2>개인 API 설정</h2>
            <p>API 토큰은 시뮬레이션 제작 데이터와 분리되어 이 브라우저의 개인 vault에만 저장됩니다.</p>
          </div>
          <button className="icon-text-button" type="button" onClick={onClose}>
            닫기
          </button>
        </header>

        <div className="personal-settings-grid">
          <section className="settings-section personal-settings-section">
            <SectionTitle icon={<Bot size={17} />} title="LLM API 키" />
            <ApiStatusCard title="LLM 개인 등록 상태" status={llmDraft.registrationStatus} message={llmDraft.verificationMessage} verifiedAt={llmDraft.verifiedAt} />
            <label>
              공급자
              <select value={llmDraft.provider} onChange={(event) => changeLlmProvider(event.target.value as LlmApiSettings["provider"])}>
                <LlmProviderOptionGroups />
              </select>
              {llmProvider.hint ? <small className="provider-hint">{llmProvider.hint}</small> : null}
            </label>
            {llmProvider.advancedBaseUrl ? (
              <label>
                기본 URL
                <input value={llmDraft.baseUrl} onChange={(event) => setLlmDraft((current) => ({ ...current, baseUrl: event.target.value, registrationStatus: "idle", verificationMessage: "" }))} />
              </label>
            ) : (
              <div className="readonly-field">
                <span>엔드포인트</span>
                <strong>
                  {llmDraft.provider === "mock"
                    ? "로컬 대체 응답"
                    : isCliAgentLlmProvider(llmDraft.provider)
                      ? "로컬 구독 CLI 브리지 (DynamicChat 서버 실행 필요)"
                      : llmProvider.baseUrl}
                </strong>
              </div>
            )}
            <label>
              API 키
              <input
                type="password"
                value={llmDraft.apiKey}
                onChange={(event) => setLlmDraft((current) => ({ ...current, apiKey: event.target.value, registrationStatus: "idle", verificationMessage: "" }))}
                placeholder={llmProvider.keyPlaceholder}
                disabled={llmDraft.provider === "mock" || isCliAgentLlmProvider(llmDraft.provider)}
              />
            </label>
            <div className="two-fields">
              <button className="icon-text-button" type="button" onClick={() => saveLlm()}>
                <Save size={16} />
                저장
              </button>
              <button className="send-button" type="button" onClick={verifyLlm} disabled={isLlmVerifying || llmDraft.provider === "mock"}>
                <KeyRound size={16} />
                {isLlmVerifying ? "검증 중" : "검증 후 등록"}
              </button>
            </div>
          </section>

          <section className="settings-section personal-settings-section">
            <SectionTitle icon={<Sparkles size={17} />} title="NovelAI 토큰" />
            <ApiStatusCard
              title="NovelAI 개인 등록 상태"
              status={novelAiDraft.registrationStatus}
              message={novelAiDraft.verificationMessage}
              verifiedAt={novelAiDraft.verifiedAt}
              detail={novelAiDraft.subscriptionTier ? `tier: ${novelAiDraft.subscriptionTier}` : undefined}
            />
            <div className="readonly-field">
              <span>시뮬레이션 이미지 설정</span>
              <strong>{state.novelAi.modelPreset} · {state.novelAi.requestMode}</strong>
              <small>모델, 해상도, 샘플러는 시뮬레이션 제작/설정에서 관리됩니다.</small>
            </div>
            <label>
              NovelAI API Key
              <input
                type="password"
                value={novelAiDraft.apiKey}
                onChange={(event) => setNovelAiDraft((current) => ({ ...current, apiKey: event.target.value, registrationStatus: "idle", verificationMessage: "" }))}
              />
            </label>
            <div className="two-fields">
              <button className="icon-text-button" type="button" onClick={() => saveNovelAi()}>
                <Save size={16} />
                저장
              </button>
              <button className="send-button" type="button" onClick={verifyNovelAi} disabled={isNovelAiVerifying}>
                <KeyRound size={16} />
                {isNovelAiVerifying ? "검증 중" : "검증 후 등록"}
              </button>
            </div>
          </section>

          <section className="settings-section personal-settings-section">
            <SectionTitle icon={<Database size={17} />} title="이미지 저장소" />
            <div className="readonly-field">
              <span>현재 저장 방식</span>
              <strong>{imageStoragePathDraft.trim() ? imageStoragePathDraft.trim() : "기본 로컬 저장소"}</strong>
              <small>서버가 실행 중일 때 생성 이미지를 파일로 저장하고, 새로고침 후 다시 불러옵니다.</small>
            </div>
            <label>
              이미지 저장 경로
              <input
                value={imageStoragePathDraft}
                onChange={(event) => setImageStoragePathDraft(event.target.value)}
                placeholder="예: S:\\DynamicChatImages"
              />
            </label>
            <p className="settings-note">비워두면 프로젝트의 .dynamicchat-data/objects 폴더를 사용합니다. 경로를 바꾸면 현재 브라우저에 남아 있는 생성 이미지도 새 경로로 다시 저장됩니다.</p>
            <div className="two-fields">
              <button className="icon-text-button" type="button" onClick={() => onSaveImageStoragePath(imageStoragePathDraft)}>
                <Save size={16} />
                저장
              </button>
              <button className="icon-text-button" type="button" onClick={() => {
                setImageStoragePathDraft("");
                onSaveImageStoragePath("");
              }}>
                <RefreshCcw size={16} />
                기본값
              </button>
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}

function PersonaPanel({
  state,
  onNotify,
  onStateChange
}: {
  state: AppState;
  onNotify: (message: string) => void;
  onStateChange: Dispatch<SetStateAction<AppState>>;
}) {
  const persona = state.userPersona;
  const personaCharacter = getPersonaCharacter(state);
  const personaMode = persona.source === "character" ? "character" : "custom";
  const updatePersona = useCallback(
    (patch: Partial<UserPersona>) => {
      onStateChange((current) => {
        const now = new Date().toISOString();
        return {
          ...current,
          simulation: {
            ...current.simulation,
            updatedAt: now
          },
          userPersona: {
            ...current.userPersona,
            ...patch,
            updatedAt: now
          }
        };
      });
    },
    [onStateChange]
  );
  const applyPreset = useCallback(
    (preset: Pick<UserPersona, "role" | "background" | "goals" | "style">) => {
      updatePersona({
        enabled: true,
        source: "custom",
        characterId: undefined,
        ...preset
      });
      onNotify("페르소나 프리셋이 적용되었습니다.");
    },
    [onNotify, updatePersona]
  );
  const selectPersonaSource = useCallback(
    (value: string) => {
      if (value === "custom") {
        updatePersona({
          enabled: true,
          source: "custom",
          characterId: undefined
        });
        onNotify("직접 페르소나 시점으로 전환했습니다.");
        return;
      }

      const character = state.characters.find((candidate) => candidate.id === value);
      if (!character) {
        return;
      }

      updatePersona({
        enabled: true,
        source: "character",
        characterId: character.id,
        name: character.name,
        role: character.role,
        background: character.summary
      });
      onNotify(`${character.name} 시점으로 진행합니다.`);
    },
    [onNotify, state.characters, updatePersona]
  );
  const notifyApplied = useCallback(() => onNotify("페르소나가 진행 프롬프트에 적용되었습니다."), [onNotify]);
  const personaPreview = createPersonaPreview(persona, personaCharacter);
  const displayName = personaCharacter?.name ?? (persona.name.trim() || "플레이어");
  const displayRole = personaCharacter?.role ?? (persona.role.trim() || "역할 미지정");

  return (
    <div className="panel-stack persona-panel">
      <div className="panel-header compact">
        <div>
          <span className="section-kicker">Player Role</span>
          <h2>페르소나</h2>
        </div>
        <StatusPill
          icon={personaMode === "character" ? <Bot size={15} /> : <UserRound size={15} />}
          label={persona.enabled ? (personaMode === "character" ? "캐릭터 시점" : "활성") : "비활성"}
          tone={persona.enabled ? "good" : "neutral"}
        />
      </div>

      <section className={`persona-identity-card ${persona.enabled ? "active" : ""}`}>
        <span className="persona-avatar">
          {personaMode === "character" ? <Bot size={21} /> : <UserRound size={21} />}
        </span>
        <div>
          <strong>{displayName}</strong>
          <p>{displayRole}</p>
        </div>
      </section>

      <section className="runtime-parameter-card persona-control-card">
        <div className="runtime-card-subhead">
          <strong>사용자 역할</strong>
          <span>{persona.enabled ? "applied" : "off"}</span>
        </div>
        <label className="checkline">
          <input
            checked={persona.enabled}
            type="checkbox"
            onChange={(event) => {
              updatePersona({ enabled: event.target.checked });
              onNotify(event.target.checked ? "페르소나가 활성화되었습니다." : "페르소나가 비활성화되었습니다.");
            }}
          />
          페르소나 적용
        </label>
        <label>
          진행 시점
          <select value={personaMode === "character" ? persona.characterId ?? "" : "custom"} onChange={(event) => selectPersonaSource(event.target.value)}>
            <option value="custom">직접 페르소나</option>
            {state.characters.map((character) => (
              <option key={character.id} value={character.id}>
                {character.name}
              </option>
            ))}
          </select>
        </label>
        {personaMode === "character" ? (
          <div className="persona-character-source">
            <strong>{personaCharacter?.name ?? "선택한 캐릭터 없음"}</strong>
            <p>{personaCharacter ? personaCharacter.summary : "캐릭터 목록에서 진행 시점으로 사용할 인물을 선택하세요."}</p>
            <small>{personaCharacter?.relationship || personaCharacter?.currentMood || "캐릭터 상태 정보 없음"}</small>
          </div>
        ) : null}
        {personaMode === "custom" ? (
          <>
        <div className="two-fields">
          <label>
            이름/호칭
            <input value={persona.name} onBlur={notifyApplied} onChange={(event) => updatePersona({ name: event.target.value })} placeholder="예: 서윤" />
          </label>
          <label>
            역할
            <input value={persona.role} onBlur={notifyApplied} onChange={(event) => updatePersona({ role: event.target.value })} placeholder="예: 신입 기록 조사관" />
          </label>
        </div>
        <label>
          배경/관계
          <textarea value={persona.background} onBlur={notifyApplied} onChange={(event) => updatePersona({ background: event.target.value })} />
        </label>
          </>
        ) : null}
        <label>
          {personaMode === "character" ? "플레이 목표" : "목표"}
          <textarea value={persona.goals} onBlur={notifyApplied} onChange={(event) => updatePersona({ goals: event.target.value })} />
        </label>
        <label>
          {personaMode === "character" ? "입력 방식" : "말투/행동 방식"}
          <textarea value={persona.style} onBlur={notifyApplied} onChange={(event) => updatePersona({ style: event.target.value })} />
        </label>
        <label>
          경계/금지
          <textarea value={persona.boundaries} onBlur={notifyApplied} onChange={(event) => updatePersona({ boundaries: event.target.value })} />
        </label>
      </section>

      <section className="persona-preset-grid" aria-label="페르소나 프리셋">
        <button
          type="button"
          onClick={() =>
            applyPreset({
              role: "현장 조사관",
              background: "사건 현장과 사람들의 증언을 직접 확인하며 단서를 모으는 인물.",
              goals: "숨겨진 사실을 찾아내고 다음 장면의 위험을 줄인다.",
              style: "차분하게 질문하고, 결정 전 주변을 관찰한다."
            })
          }
        >
          <Search size={15} />
          조사관
        </button>
        <button
          type="button"
          onClick={() =>
            applyPreset({
              role: "협상가",
              background: "갈등 상황에서 사람들의 이해관계를 읽고 합의를 끌어내는 인물.",
              goals: "관계 악화를 막고 필요한 정보를 대화로 확보한다.",
              style: "부드럽지만 핵심을 놓치지 않는 말투를 쓴다."
            })
          }
        >
          <MessageSquareText size={15} />
          협상가
        </button>
        <button
          type="button"
          onClick={() =>
            applyPreset({
              role: "운영 책임자",
              background: "자원, 시간, 사람의 상태를 관리하며 장기 목표를 추진하는 인물.",
              goals: "현재 문제의 우선순위를 정하고 손실을 최소화한다.",
              style: "짧게 지시하고 결과를 확인한다."
            })
          }
        >
          <ClipboardCheck size={15} />
          책임자
        </button>
      </section>

      <section className="context-pack persona-context-preview">
        <strong>진행 반영</strong>
        <p>{persona.enabled ? personaPreview : "페르소나 비활성 상태입니다."}</p>
        <small>{new Date(persona.updatedAt).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}</small>
      </section>
    </div>
  );
}

/**
 * Keeps a backend's model list in sync with what that backend actually accepts right now.
 *
 * Two kinds of backend need this. Local presets ship no model names on purpose — the list lives on the
 * machine, not in the code — so until something asks the server the dropdown is empty and the keyless path
 * looks broken. The agy CLI ships a preset list, but agy retires ids without warning and then refuses them
 * outright, so the preset alone drifts into offering models that can only fail; asking agy is the fix.
 *
 * Selecting the provider is the moment the source becomes known, so that is when discovery runs; the button
 * only exists to retry after the user starts the server or signs in. The auto pass is silent because the
 * base-URL field updates per keystroke: it is debounced, and only an explicit retry reports what it found.
 */
function useLlmModelDiscovery(
  settings: Pick<LlmApiSettings, "provider" | "baseUrl" | "apiKey">,
  onNotify: (message: string) => void
): { models: string[]; isLoading: boolean; reload: () => void; isDiscoverable: boolean } {
  const { provider, baseUrl, apiKey } = settings;
  const [models, setModels] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const isLocalProvider = getLlmProviderOption(provider).group === "local";
  const isAgyProvider = provider === "antigravity_cli";
  const isDiscoverable = isAgyProvider || (isLocalProvider && Boolean(baseUrl.trim()));
  const discover = useCallback(
    () => (isAgyProvider ? listCliAgentModels("antigravity") : listLocalLlmModels({ baseUrl, apiKey })),
    [apiKey, baseUrl, isAgyProvider]
  );

  useEffect(() => {
    // A provider switch invalidates whatever the previous source reported.
    setModels([]);
    if (!isDiscoverable) {
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(() => {
      setIsLoading(true);
      void discover()
        .then((found) => {
          if (!cancelled) {
            setModels(found);
          }
        })
        .finally(() => {
          if (!cancelled) {
            setIsLoading(false);
          }
        });
    }, 400);

    return () => {
      // Guards against a switch mid-flight publishing the previous source's list.
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [discover, isDiscoverable, provider]);

  const reload = useCallback(() => {
    setIsLoading(true);
    void discover()
      .then((found) => {
        setModels(found);
        onNotify(
          found.length > 0
            ? `${isAgyProvider ? "agy" : "로컬 서버"}에서 모델 ${found.length}개를 불러왔습니다.`
            : isAgyProvider
              ? "agy에서 모델 목록을 가져오지 못했습니다. API 서버가 실행 중인지, agy에 로그인되어 있는지 확인하세요."
              : "로컬 서버에서 모델 목록을 가져오지 못했습니다. 서버가 실행 중인지, 기본 URL이 맞는지 확인하세요."
        );
      })
      .finally(() => setIsLoading(false));
  }, [discover, isAgyProvider, onNotify]);

  return { models, isLoading, reload, isDiscoverable };
}

/**
 * A stored agy model id that agy no longer lists. agy retires ids without notice and then refuses them, and a
 * simulation saved before that stopped producing turns with nothing on screen saying why. The API bridge now runs
 * the replacement below in its place; this says so, and offers to store it so the setting matches what runs.
 */
function StaleAgyModelNote({
  model,
  discovered,
  onReplace
}: {
  model: string;
  discovered: string[];
  onReplace: (replacement: string) => void;
}) {
  const current = model.trim();
  if (!current || discovered.length === 0 || discovered.includes(current)) {
    return null;
  }
  const replacement = pickAntigravityReplacementModel(current, discovered);
  return (
    <div className="stale-model-note">
      <p className="settings-note warning">
        agy가 더 이상 <code>{current}</code> 모델을 제공하지 않습니다.
        {replacement ? ` 지금은 같은 계열의 ${replacement}(으)로 대신 진행합니다.` : " 위 목록에서 모델을 다시 고르세요."}
      </p>
      {replacement ? (
        <button className="icon-text-button full" type="button" onClick={() => onReplace(replacement)}>
          <Check size={15} />
          {replacement}(으)로 바꾸기
        </button>
      ) : null}
    </div>
  );
}

/** Preset + discovered models for the dropdown. A discovered agy list is authoritative (see the hook above). */
function mergeDiscoveredModels(provider: LlmProvider, discovered: string[], presetModels: string[]): string[] {
  const models = provider === "antigravity_cli" && discovered.length > 0 ? discovered : [...discovered, ...presetModels];
  return uniqueIds(models).filter((model) => model !== "custom");
}

function SettingsPanel({
  state,
  onImageProfileChange,
  onLlmChange,
  onImageTagLlmChange,
  onNovelAiChange,
  onNotify,
  onOpenPersonalSettings,
  onStateChange,
  onResetDemo,
  jumpRequest,
  onJumpHandled,
  onRunLocalLlmTask,
  section,
  onSectionChange
}: {
  state: AppState;
  onImageProfileChange: (patch: Partial<ImageGenerationProfile>) => void;
  onLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onImageTagLlmChange: (patch: Partial<LlmApiSettings>) => void;
  onNovelAiChange: (patch: Partial<NovelAiApiSettings>) => void;
  onNotify?: (message: string) => void;
  onOpenPersonalSettings: () => void;
  onStateChange: Dispatch<SetStateAction<AppState>>;
  onResetDemo: () => void;
  /** A section to bring into view and focus (e.g. from '실사 설정 열기'); cleared through onJumpHandled once shown. */
  jumpRequest?: SettingsJumpRequest;
  onJumpHandled?: () => void;
  /** The shared local-GPU lock for LLM calls made here; without it the call runs directly. */
  onRunLocalLlmTask?: LocalLlmTaskRunner;
  /** The one section to show (the rail's sub-tabs). Without it every section renders in one column. */
  section?: SettingsSection;
  onSectionChange?: (section: SettingsSection) => void;
}) {
  const [isRuntimeVerifying, setIsRuntimeVerifying] = useState(false);
  const photorealSectionRef = useRef<HTMLDivElement>(null);
  const shows = (target: SettingsSection) => section === undefined || section === target;
  // Runs where the panel is on screen only: the rail and the mobile drawer can both hold a settings panel, and the
  // hidden one must leave the request to the visible one. A frame later, so a drawer that has just opened has
  // already moved focus to its first control, and this focus is the one that stays.
  useEffect(() => {
    if (jumpRequest?.section !== "photoreal") {
      return;
    }
    // The ComfyUI section only exists on its own tab; switch first, and this runs again once it is rendered.
    if (section !== undefined && section !== "photoreal") {
      onSectionChange?.("photoreal");
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      const target = photorealSectionRef.current;
      if (!target || target.getClientRects().length === 0) {
        return;
      }
      const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
      target.scrollIntoView({ block: "start", behavior: reduceMotion ? "auto" : "smooth" });
      target.focus({ preventScroll: true });
      onJumpHandled?.();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [jumpRequest, onJumpHandled, section, onSectionChange]);
  const [runtimeCheck, setRuntimeCheck] = useState("");
  const [vibeEncodeStatus, setVibeEncodeStatus] = useState<Record<string, { state: "encoding" | "error"; message?: string }>>({});
  const [dynamicChatApiBaseUrl, setDynamicChatApiBaseUrl] = useState(() => getConfiguredDynamicChatApiBaseUrl());
  const llmProvider = getLlmProviderOption(state.llm.provider);
  const imageTagLlmProvider = getLlmProviderOption(state.imageTagLlm.provider);
  const activeRuntimeResolution = resolutionPresets.some((preset) => preset.width === state.imageProfile.width && preset.height === state.imageProfile.height)
    ? `${state.imageProfile.width}x${state.imageProfile.height}`
    : "custom";
  const activeOutputLengthPreset =
    outputLengthPresets.find((preset) => preset.tokens === state.llm.maxTokens)?.id ?? "custom";
  const activeOutputLengthLabel =
    activeOutputLengthPreset === "custom"
      ? `${state.llm.maxTokens}토큰`
      : outputLengthPresets.find((preset) => preset.id === activeOutputLengthPreset)?.label ?? `${state.llm.maxTokens}토큰`;
  const activeImageGenerationCadence =
    imageGenerationCadenceOptions.find((option) => option.value === state.imageProfile.generationCadence) ?? imageGenerationCadenceOptions[1];
  const activePromptModeOption = getPromptModeOption(state.simulation.promptMode);
  const llmHasSecret = state.llm.provider === "mock" || hasStoredSecret(state.llm);
  const llmNeedsSecret = state.llm.provider !== "mock" && !llmHasSecret;
  const novelAiHasSecret = hasStoredSecret(state.novelAi);
  const novelAiNeedsSecret = state.novelAi.enabled && state.novelAi.requestMode !== "mock" && !novelAiHasSecret;
  // v4/v4.5/v5는 generate 전에 참조 이미지를 encode-vibe로 인코딩해야 한다. v3는 원본 이미지를 직접 전송한다.
  const requiresVibeEncoding = state.novelAi.modelPreset.startsWith("NAID4") || state.novelAi.modelPreset.startsWith("NAID5");
  const canEncodeVibe = state.novelAi.requestMode !== "mock" && novelAiHasSecret;
  const llmStatusLabel = state.llm.provider === "mock" ? "mock" : llmHasSecret ? "키 등록됨" : "키 필요";
  const novelAiStatusLabel = !state.novelAi.enabled ? "비활성" : state.novelAi.requestMode === "mock" ? "mock" : novelAiHasSecret ? "토큰 등록됨" : "토큰 필요";
  const activeContentRatingOption = contentRatingOptions.find((option) => option.value === state.simulation.contentRating) ?? contentRatingOptions[0];
  const imageBackendRouting = resolveImageBackendRouting(state.imageProfile);
  const photorealSettings = state.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS;
  const photorealMetricLabel = !photorealSettings.enabled
    ? "꺼짐"
    : photorealSettings.requestMode === "mock"
      ? "mock"
      : imageBackendRoutingModeOptions.find((option) => option.value === imageBackendRouting.mode)?.label ?? "켜짐";
  const novelAiSupportsCharacterReference = novelAiCharacterReferencePresets.has(state.novelAi.modelPreset);
  const novelAiIsV5 = state.novelAi.modelPreset.startsWith("NAID5");
  const notifyApplied = useCallback((message: string) => onNotify?.(message), [onNotify]);
  // Models discovered from a running local server or from agy. A local list is merged into the preset list,
  // which keeps the keyless path usable without typing an exact id from memory. An agy list REPLACES the
  // preset: agy refuses retired ids outright, so offering them next to the live list only offers failures.
  const {
    models: discoveredLocalModels,
    isLoading: isLoadingLocalModels,
    reload: loadLocalModels
  } = useLlmModelDiscovery(state.llm, notifyApplied);
  const availableLlmModels = mergeDiscoveredModels(state.llm.provider, discoveredLocalModels, llmProvider.models);
  const selectedLlmModel = availableLlmModels.includes(state.llm.model) ? state.llm.model : "custom";
  // The image-tag model gets its own discovery and its own context size. Running the tag pass on a second,
  // smaller local model is the setup that makes a high-density cadence affordable — but it only works if
  // that model can be picked and sized independently, which previously it could not.
  const {
    models: discoveredImageTagLocalModels,
    isLoading: isLoadingImageTagLocalModels,
    reload: loadImageTagLocalModels
  } = useLlmModelDiscovery(state.imageTagLlm, notifyApplied);
  const availableImageTagLlmModels = mergeDiscoveredModels(
    state.imageTagLlm.provider,
    discoveredImageTagLocalModels,
    imageTagLlmProvider.models
  );
  const selectedImageTagLlmModel = availableImageTagLlmModels.includes(state.imageTagLlm.model)
    ? state.imageTagLlm.model
    : "custom";
  const applyImageProfileChange = useCallback(
    (patch: Partial<ImageGenerationProfile>, message: string) => {
      onImageProfileChange(patch);
      notifyApplied(message);
    },
    [notifyApplied, onImageProfileChange]
  );
  const applyLlmChange = useCallback(
    (patch: Partial<LlmApiSettings>, message: string) => {
      onLlmChange(patch);
      notifyApplied(message);
    },
    [notifyApplied, onLlmChange]
  );
  // The local presets ship with no model (an untouched field must not POST a made-up id), so switching to Ollama
  // left the field empty and every turn failed with "모델 이름이 비어 있습니다" although the installed list was already
  // loaded into the dropdown. Take the first installed model when a list arrives for an empty field. It runs only
  // when a new list arrives, so choosing "직접 입력" afterwards is left alone.
  const llmModelIsEmpty = !state.llm.model.trim();
  const imageTagLlmModelIsEmpty = !state.imageTagLlm.model.trim();
  useEffect(() => {
    const first = discoveredLocalModels[0];
    if (first && llmModelIsEmpty && getLlmProviderOption(state.llm.provider).group === "local") {
      applyLlmChange({ model: first }, `설치된 로컬 모델 ${first} 을(를) 선택했습니다.`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only a newly discovered list should trigger this
  }, [discoveredLocalModels]);
  useEffect(() => {
    const first = discoveredImageTagLocalModels[0];
    if (first && imageTagLlmModelIsEmpty && getLlmProviderOption(state.imageTagLlm.provider).group === "local") {
      onImageTagLlmChange({ model: first });
      notifyApplied(`이미지 태그 LLM에 설치된 로컬 모델 ${first} 을(를) 선택했습니다.`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only a newly discovered list should trigger this
  }, [discoveredImageTagLocalModels]);
  const applyNovelAiChange = useCallback(
    (patch: Partial<NovelAiApiSettings>, message: string) => {
      onNovelAiChange(patch);
      notifyApplied(message);
    },
    [notifyApplied, onNovelAiChange]
  );
  const applyStateChange = useCallback(
    (updater: SetStateAction<AppState>, message: string) => {
      onStateChange(updater);
      notifyApplied(message);
    },
    [notifyApplied, onStateChange]
  );
  // Photoreal settings have no secret and no vault, so they are written straight into the simulation state, like
  // NeuralMap's. hydrateState normalizes them on the way in (see the number inputs in PhotorealRuntimeSettings).
  const applyPhotorealChange = useCallback(
    (patch: Partial<PhotorealApiSettings>, message?: string) => {
      onStateChange((current) => ({
        ...current,
        photoreal: {
          ...(current.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS),
          ...patch
        }
      }));
      if (message) {
        notifyApplied(message);
      }
    },
    [notifyApplied, onStateChange]
  );
  const addVibeTransferReferences = useCallback(
    (files: FileList | null) => {
      const picked = Array.from(files ?? []).filter((file) => file.type.startsWith("image/"));
      if (picked.length === 0) {
        return;
      }
      Promise.all(
        picked.map(
          (file) =>
            new Promise<{ name: string; image: string } | null>((resolve) => {
              const reader = new FileReader();
              reader.onload = () => {
                if (typeof reader.result !== "string") {
                  resolve(null);
                  return;
                }
                // Downscale to 448 px on the long edge before storing; avoids blowing
                // the ~5 MB localStorage quota with raw multi-MB base64 payloads.
                downscaleImageToDataUrl(reader.result).then((image) => resolve({ name: file.name, image }));
              };
              reader.onerror = () => resolve(null);
              reader.readAsDataURL(file);
            })
        )
      ).then((loaded) => {
        const additions: NovelAiVibeTransferReference[] = loaded
          .filter((item): item is { name: string; image: string } => Boolean(item))
          .map((item) => ({
            id: createId("vibe_ref"),
            name: item.name,
            image: item.image,
            referenceStrength: 0.6,
            informationExtracted: 1
          }));
        if (additions.length === 0) {
          return;
        }
        onNovelAiChange({ vibeTransferReferences: [...state.novelAi.vibeTransferReferences, ...additions] });
        notifyApplied(`Vibe Transfer 참조 이미지 ${additions.length}개가 추가되었습니다.`);
      });
    },
    [notifyApplied, onNovelAiChange, state.novelAi.vibeTransferReferences]
  );
  const updateVibeTransferReference = useCallback(
    (id: string, patch: Partial<NovelAiVibeTransferReference>) => {
      onNovelAiChange({
        vibeTransferReferences: state.novelAi.vibeTransferReferences.map((reference) =>
          reference.id === id ? { ...reference, ...patch } : reference
        )
      });
    },
    [onNovelAiChange, state.novelAi.vibeTransferReferences]
  );
  const removeVibeTransferReference = useCallback(
    (id: string) => {
      onNovelAiChange({
        vibeTransferReferences: state.novelAi.vibeTransferReferences.filter((reference) => reference.id !== id)
      });
      notifyApplied("Vibe Transfer 참조 이미지가 제거되었습니다.");
    },
    [notifyApplied, onNovelAiChange, state.novelAi.vibeTransferReferences]
  );
  const clearVibeTransferEncoding = useCallback(
    (id: string) => {
      onNovelAiChange({
        vibeTransferReferences: state.novelAi.vibeTransferReferences.map((reference) =>
          reference.id === id
            ? { ...reference, encodedVibe: undefined, encodedModel: undefined, encodedInformationExtracted: undefined }
            : reference
        )
      });
      notifyApplied("Vibe Transfer 인코딩이 해제되었습니다.");
    },
    [notifyApplied, onNovelAiChange, state.novelAi.vibeTransferReferences]
  );
  const encodeVibeTransferReference = useCallback(
    async (reference: NovelAiVibeTransferReference) => {
      setVibeEncodeStatus((current) => ({ ...current, [reference.id]: { state: "encoding" } }));
      try {
        const encodedVibe = await encodeNovelAiVibe({
          state,
          image: reference.image,
          informationExtracted: reference.informationExtracted
        });
        // Persist the encoding through onNovelAiChange so it lands in the GLOBAL vibe vault (not just the
        // per-simulation state, which applyPersonalApiVault would otherwise overwrite on the next sim
        // switch — losing the encoding). Map off the freshest references via a microtask snapshot.
        onNovelAiChange({
          vibeTransferReferences: state.novelAi.vibeTransferReferences.map((item) =>
            item.id === reference.id
              ? {
                  ...item,
                  encodedVibe,
                  encodedModel: state.novelAi.modelPreset,
                  encodedInformationExtracted: reference.informationExtracted
                }
              : item
          )
        });
        setVibeEncodeStatus((current) => {
          const next = { ...current };
          delete next[reference.id];
          return next;
        });
        notifyApplied("Vibe Transfer 인코딩이 완료되었습니다.");
      } catch (error) {
        setVibeEncodeStatus((current) => ({
          ...current,
          [reference.id]: { state: "error", message: error instanceof Error ? error.message : "인코딩에 실패했습니다." }
        }));
      }
    },
    [notifyApplied, onNovelAiChange, state]
  );
  const applyContentRatingChange = useCallback(
    (contentRating: ContentRating) => {
      applyStateChange(
        (current) => ({
          ...current,
          simulation: {
            ...current.simulation,
            contentRating,
            updatedAt: new Date().toISOString()
          }
        }),
        contentRating === "adult_19"
          ? "19+ 성인 전용 등급이 적용되었습니다. 앱 자체 콘텐츠 필터는 사용하지 않습니다."
          : "일반 콘텐츠 등급이 적용되었습니다."
      );
    },
    [applyStateChange]
  );
  const verifyRuntimeTurn = async () => {
    setIsRuntimeVerifying(true);
    setRuntimeCheck("실제 LLM 턴 테스트 중...");
    try {
      const relevantModules = state.modules
        .filter((module) => module.enabled && module.tokenPolicy !== "disabled")
        .sort((a, b) => b.priority - a.priority)
        .slice(0, 6);
      // A real LLM call, so it takes the local-GPU lock like a turn: run straight, a local Ollama would load beside
      // a photoreal model still resident from the last cut and spill.
      const runLlm: LocalLlmTaskRunner = onRunLocalLlmTask ?? ((_taskState, task) => task());
      const [assistantResult, novelAiResult, photorealResult] = await Promise.all([
        runLlm(state, () =>
          generateAssistantText({
            state,
            userText: "실제 API 연결 검증용입니다. 현재 시뮬레이션을 한 문장으로만 이어가세요.",
            modules: relevantModules,
            evidence: state.contextPacks.at(-1)?.evidence ?? [],
            fallback: "Mock 대체 응답입니다."
          })
        ),
        state.novelAi.enabled
          ? validateNovelAiApi(state.novelAi)
          : Promise.resolve({ ok: true, message: "NovelAI 비활성화 상태입니다. 저장 이미지 모드로 진행됩니다." }),
        // A status probe only — it renders nothing and loads no model, so it is safe to run beside the LLM call.
        photorealSettings.enabled && photorealSettings.requestMode !== "mock" ? probeComfyUi(state) : Promise.resolve(undefined)
      ]);
      const llmStatus =
        assistantResult.source === "llm"
          ? "실제 API 응답"
          : assistantResult.rawPreview
            ? "실제 API 응답, sidecar 파싱 실패"
            : "mock/대체 응답";
      setRuntimeCheck(
        [
          `LLM: ${llmStatus}${assistantResult.error ? ` (${assistantResult.error})` : ""}`,
          `NovelAI: ${novelAiResult.ok ? "사용 가능" : "확인 실패"} - ${novelAiResult.message}`,
          photorealResult ? `실사: ${photorealResult.ok ? "사용 가능" : "확인 실패"} - ${photorealResult.message}` : undefined
        ]
          .filter((line): line is string => Boolean(line))
          .join(" / ")
      );
    } catch (error) {
      setRuntimeCheck(error instanceof Error ? error.message : "실제 시뮬레이션 검증 중 알 수 없는 오류가 발생했습니다.");
    } finally {
      setIsRuntimeVerifying(false);
    }
  };

  return (
    <div className="settings-panel">
      {shows("general") ? (
        <>
          <div className="runtime-settings-overview">
            <div>
              <span className="section-kicker">진행 설정</span>
              <strong>{activePromptModeOption.label} 모드</strong>
              <p>{activePromptModeOption.description}</p>
            </div>
            <div className="runtime-settings-metrics">
              <Metric label="LLM" value={llmProvider.label} />
              <Metric label="출력" value={activeOutputLengthLabel} />
              <Metric label="문맥" value={`${state.neuralMap.tokenBudget}토큰`} />
              <Metric label="등급" value={activeContentRatingOption.label} />
              <Metric label="Prompt" value={`${state.modules.filter((module) => module.enabled).length}개`} />
              <Metric label="Image" value={activeImageGenerationCadence.label} />
              <Metric label="실사" value={photorealMetricLabel} />
            </div>
          </div>

          <SectionTitle icon={<ShieldCheck size={17} />} title="콘텐츠 등급" />
          <div className="runtime-parameter-card">
            <div className="runtime-card-subhead">
              <strong>{activeContentRatingOption.label}</strong>
              <span>{activeContentRatingOption.detail}</span>
            </div>
            <label>
              진행 등급
              <select value={state.simulation.contentRating} onChange={(event) => applyContentRatingChange(event.target.value as ContentRating)}>
                {contentRatingOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="settings-note">DynamicChat 앱은 콘텐츠 차단기를 두지 않고, 등급은 LLM/NovelAI 설정과 사용자 규칙에 전달됩니다.</p>
          </div>

          <SectionTitle icon={<ShieldCheck size={17} />} title="저장/권한" />
          <div className="context-pack">
            <strong>{state.security.scope.ownerId} · {state.security.scope.workspaceId}</strong>
            <div className="evidence-row">
              <span>scope</span>
              <p>{state.security.scope.projectId}</p>
              <small>{state.security.scope.environment} · browser secret cache {state.security.browserSecretCacheEnabled ? "on" : "off"}</small>
            </div>
            <div className="evidence-row">
              <span>secret</span>
              <p>{state.security.secretStorage}</p>
              <small>{state.security.warning}</small>
            </div>
          </div>

          <SectionTitle icon={<Database size={17} />} title="DynamicChat API" />
          <label>
            기본 URL
            <input
              placeholder="http://127.0.0.1:8788"
              value={dynamicChatApiBaseUrl}
              onChange={(event) => {
                setDynamicChatApiBaseUrl(event.target.value);
                configureDynamicChatApiBaseUrl(event.target.value);
              }}
              onBlur={() => notifyApplied("DynamicChat API 기본 URL이 적용되었습니다.")}
            />
          </label>
        </>
      ) : null}

      {shows("image") ? (
        <>
          <SectionTitle icon={<ImageIcon size={17} />} title="이미지 트리거/프롬프트" />
          <div className="runtime-parameter-card">
            <div className="runtime-card-subhead">
              <strong>생성 트리거</strong>
              <span>{state.simulation.realtimeImageEnabled ? "실시간" : "비활성"}</span>
            </div>
            <label className="checkline">
              <input
                checked={state.simulation.realtimeImageEnabled}
                type="checkbox"
                onChange={(event) =>
                  applyStateChange(
                    (current) => ({
                      ...current,
                      simulation: {
                        ...current.simulation,
                        realtimeImageEnabled: event.target.checked
                      }
                    }),
                    event.target.checked ? "실시간 이미지가 활성화되었습니다." : "실시간 이미지가 비활성화되었습니다."
                  )
                }
              />
              실시간 이미지 활성화
            </label>
            <label>
              생성 모드
              <select
                value={state.imageProfile.triggerMode}
                onChange={(event) => applyImageProfileChange({ triggerMode: event.target.value as ImageTriggerMode }, "이미지 생성 모드가 적용되었습니다.")}
              >
                {triggerModes.map((mode) => (
                  <option key={mode} value={mode}>
                    {mode}
                  </option>
                ))}
              </select>
            </label>
            <label className="runtime-preset-field">
              생성 밀도 <span className="inline-value">{activeImageGenerationCadence.label}</span>
              <div className="runtime-preset-grid" role="group" aria-label="이미지 생성 밀도 프리셋">
                {imageGenerationCadenceOptions.map((option) => (
                  <button
                    key={option.value}
                    className={state.imageProfile.generationCadence === option.value ? "active" : ""}
                    type="button"
                    onClick={() => applyImageProfileChange({ generationCadence: option.value }, `${option.label} 이미지 생성 밀도가 적용되었습니다.`)}
                  >
                    <strong>{option.label}</strong>
                    <span>{option.detail}</span>
                  </button>
                ))}
              </div>
            </label>
          </div>
          <ImageBackendRoutingCard
            routing={imageBackendRouting}
            novelAiEnabled={state.novelAi.enabled}
            photorealEnabled={photorealSettings.enabled}
            onChange={(backendRouting, message) => applyImageProfileChange({ backendRouting }, message)}
          />
          <div className="runtime-parameter-card image-prompt-settings-card">
            <div className="runtime-card-subhead">
              <strong>프롬프트 레이어</strong>
              <span>태그/규정</span>
            </div>
            <div className="two-fields">
              <label>
                작가 프롬프트
                <textarea value={state.imageProfile.artistPrompt} onChange={(event) => onImageProfileChange({ artistPrompt: event.target.value })} onBlur={() => notifyApplied("작가 프롬프트가 적용되었습니다.")} />
              </label>
              <label>
                기본 퀄리티 프롬프트
                <textarea value={state.imageProfile.qualityPrompt} onChange={(event) => onImageProfileChange({ qualityPrompt: event.target.value })} onBlur={() => notifyApplied("퀄리티 프롬프트가 적용되었습니다.")} />
              </label>
            </div>
            <label>
              스타일 프롬프트
              <textarea value={state.imageProfile.stylePrompt} onChange={(event) => onImageProfileChange({ stylePrompt: event.target.value })} onBlur={() => notifyApplied("스타일 프롬프트가 적용되었습니다.")} />
            </label>
            <label>
              공통 negative prompt
              <textarea value={state.imageProfile.negativePrompt} onChange={(event) => onImageProfileChange({ negativePrompt: event.target.value })} onBlur={() => notifyApplied("Negative prompt가 적용되었습니다.")} />
            </label>
            <label>
              사용자 규정 / LLM 지시문
              <textarea value={state.imageProfile.userRules} onChange={(event) => onImageProfileChange({ userRules: event.target.value })} onBlur={() => notifyApplied("사용자 이미지 규정이 적용되었습니다.")} />
            </label>
          </div>
        </>
      ) : null}

      {shows("model") ? (
        <>
          <SectionTitle icon={<Bot size={17} />} title="LLM 런타임" />
          <div className="runtime-model-card">
            <div className="runtime-model-heading">
              <div>
                <span>현재 API</span>
                <strong>{llmProvider.label}</strong>
              </div>
              <StatusPill
                icon={<Bot size={15} />}
                label={llmStatusLabel}
                tone={llmHasSecret && state.llm.registrationStatus !== "failed" ? "good" : "neutral"}
              />
            </div>
            {llmNeedsSecret || state.llm.registrationStatus === "failed" || state.llm.registrationStatus === "verifying" ? (
              <ApiStatusCard title="LLM 등록 상태" status={state.llm.registrationStatus} message={state.llm.verificationMessage} verifiedAt={state.llm.verifiedAt} />
            ) : null}
            <p className="settings-note">LLM API 키는 개인 설정에서 관리하고, 여기서는 진행 중인 시뮬레이션의 Provider와 실행 모델만 바꿉니다.</p>
            <label>
              공급자
              <select
                value={state.llm.provider}
                onChange={(event) => {
                  const provider = event.target.value as LlmApiSettings["provider"];
                  applyLlmChange(createLlmProviderPatch(provider), `${getLlmProviderOption(provider).label} 공급자가 적용되었습니다.`);
                }}
              >
                <LlmProviderOptionGroups />
              </select>
              {llmProvider.hint ? <small className="provider-hint">{llmProvider.hint}</small> : null}
            </label>
            {llmProvider.advancedBaseUrl ? (
              <label>
                기본 URL
                <input
                  value={state.llm.baseUrl}
                  onChange={(event) => onLlmChange({ baseUrl: event.target.value, registrationStatus: "idle", verificationMessage: "" })}
                  onBlur={() => notifyApplied("LLM 기본 URL이 적용되었습니다.")}
                />
              </label>
            ) : null}
            <div className="runtime-model-grid">
              <label>
                사용 가능 모델
                <select
                  value={selectedLlmModel}
                  onChange={(event) => {
                    if (event.target.value === "custom") {
                      applyLlmChange({ model: "" }, "커스텀 LLM 모델 입력으로 전환했습니다.");
                      return;
                    }
                    applyLlmChange({ model: event.target.value }, `${event.target.value} 모델이 적용되었습니다.`);
                  }}
                >
                  {availableLlmModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                  <option value="custom">직접 입력</option>
                </select>
              </label>
              {/* Local backends are the keyless path, so there is no vendor console to copy a model id from and
                  a typo surfaces as an opaque upstream 404. Every local server implements GET /v1/models. */}
              {llmProvider.group === "local" || state.llm.provider === "antigravity_cli" ? (
                <label>
                  설치된 모델
                  <button className="icon-text-button full model-discovery-button" type="button" onClick={loadLocalModels} disabled={isLoadingLocalModels}>
                    <RefreshCcw size={15} />
                    <span>
                      {isLoadingLocalModels
                        ? "불러오는 중…"
                        : discoveredLocalModels.length > 0
                          ? `${discoveredLocalModels.length}개 · 새로고침`
                          : "못 찾음 · 다시 시도"}
                    </span>
                  </button>
                </label>
              ) : null}
              {/* Ollama serves every model at a 4096-token context unless told otherwise, so the same model name
                  can mean 4k or 32k. The prompt is sized from this and the native transport passes it as
                  num_ctx, which is what makes the annotation pass fit at all on a local setup. */}
              <label>
                컨텍스트 (토큰)
                <input
                  type="number"
                  min="2048"
                  step="1024"
                  value={state.llm.contextTokens ?? llmProvider.contextTokens}
                  onChange={(event) => onLlmChange({ contextTokens: Math.max(2048, Number(event.target.value) || 0) })}
                  onBlur={() => notifyApplied("LLM 컨텍스트 크기가 적용되었습니다.")}
                />
              </label>
              <label>
                직접 입력
                <input
                  value={state.llm.model}
                  placeholder={llmProvider.defaultModel}
                  onChange={(event) => onLlmChange({ model: event.target.value })}
                  onBlur={() => notifyApplied("LLM 모델 입력값이 적용되었습니다.")}
                />
              </label>
            </div>
            {state.llm.provider === "antigravity_cli" ? (
              <StaleAgyModelNote
                model={state.llm.model}
                discovered={discoveredLocalModels}
                onReplace={(replacement) => applyLlmChange({ model: replacement }, `${replacement} 모델이 적용되었습니다.`)}
              />
            ) : null}
            <div className="runtime-model-grid compact">
              <label className="runtime-preset-field">
                출력량 목표
                <div className="runtime-preset-grid" role="group" aria-label="LLM 출력량 프리셋">
                  {outputLengthPresets.map((preset) => (
                    <button
                      key={preset.id}
                      className={activeOutputLengthPreset === preset.id ? "active" : ""}
                      type="button"
                      onClick={() => applyLlmChange({ maxTokens: preset.tokens }, `${preset.label} 출력량 목표가 적용되었습니다.`)}
                    >
                      <strong>{preset.label}</strong>
                      <span>{preset.detail}</span>
                    </button>
                  ))}
                </div>
              </label>
              <label>
                응답 출력 토큰 <span className="inline-value">{activeOutputLengthLabel}</span>
                <input
                  type="number"
                  min="512"
                  max="8000"
                  step="100"
                  value={state.llm.maxTokens}
                  onChange={(event) => onLlmChange({ maxTokens: Number(event.target.value) })}
                  onBlur={() => notifyApplied("LLM 최대 출력 토큰이 적용되었습니다.")}
                />
              </label>
              <label>
                온도 <span className="inline-value">{state.llm.temperature.toFixed(2)}</span>
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.05"
                  value={state.llm.temperature}
                  onChange={(event) => onLlmChange({ temperature: Number(event.target.value) })}
                  onMouseUp={() => notifyApplied("LLM temperature가 적용되었습니다.")}
                  onTouchEnd={() => notifyApplied("LLM temperature가 적용되었습니다.")}
                />
              </label>
            </div>
            {llmNeedsSecret ? (
              <div className="readonly-field runtime-secret-state warning">
                <span>개인 API 키</span>
                <strong>개인 설정에서 {llmProvider.label} 키 등록 필요</strong>
                <small>제작자 키는 포함되지 않으므로 실행자의 개인 키만 사용합니다.</small>
              </div>
            ) : null}
            <button className="icon-text-button full" type="button" onClick={onOpenPersonalSettings}>
              <KeyRound size={16} />
              개인 설정에서 LLM 키 관리
            </button>
          </div>

          <SectionTitle icon={<Bot size={17} />} title="이미지 태그 전용 LLM" />
          <div className="runtime-model-card">
            <p className="settings-note">내러티브 모델과 분리되어 더 저렴한 모델로 이미지 태그(NovelAI 태그)를 생성할 수 있습니다. 비활성화하면 메인 LLM을 사용합니다.</p>
            <label className="checkline">
              <input
                type="checkbox"
                checked={state.imageTagLlm.enabled}
                onChange={(event) => onImageTagLlmChange({ enabled: event.target.checked })}
              />
              별도 모델로 이미지 태그 생성
            </label>
            {state.imageTagLlm.enabled ? (
              <>
                <label>
                  공급자
                  <select
                    value={state.imageTagLlm.provider}
                    onChange={(event) => {
                      const provider = event.target.value as LlmApiSettings["provider"];
                      onImageTagLlmChange(createLlmProviderPatch(provider));
                    }}
                  >
                    <LlmProviderOptionGroups />
                  </select>
                </label>
                {imageTagLlmProvider.advancedBaseUrl ? (
                  <label>
                    기본 URL
                    <input
                      value={state.imageTagLlm.baseUrl}
                      onChange={(event) => onImageTagLlmChange({ baseUrl: event.target.value })}
                    />
                  </label>
                ) : null}
                <div className="runtime-model-grid">
                  <label>
                    사용 가능 모델
                    <select
                      value={selectedImageTagLlmModel}
                      onChange={(event) => {
                        if (event.target.value === "custom") {
                          onImageTagLlmChange({ model: "" });
                          return;
                        }
                        onImageTagLlmChange({ model: event.target.value });
                      }}
                    >
                      {availableImageTagLlmModels.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                      <option value="custom">직접 입력</option>
                    </select>
                  </label>
                  <label>
                    직접 입력
                    <input
                      value={state.imageTagLlm.model}
                      placeholder={imageTagLlmProvider.defaultModel}
                      onChange={(event) => onImageTagLlmChange({ model: event.target.value })}
                    />
                  </label>
                </div>
                {state.imageTagLlm.provider === "antigravity_cli" ? (
                  <StaleAgyModelNote
                    model={state.imageTagLlm.model}
                    discovered={discoveredImageTagLocalModels}
                    onReplace={(replacement) => {
                      onImageTagLlmChange({ model: replacement });
                      notifyApplied(`이미지 태그 LLM을 ${replacement} 모델로 바꿨습니다.`);
                    }}
                  />
                ) : null}
                {imageTagLlmProvider.group === "local" || state.imageTagLlm.provider === "antigravity_cli" ? (
                  <label>
                    설치된 모델
                    <button
                      className="icon-text-button full model-discovery-button"
                      type="button"
                      onClick={loadImageTagLocalModels}
                      disabled={isLoadingImageTagLocalModels}
                    >
                      <RefreshCcw size={15} />
                      <span>
                        {isLoadingImageTagLocalModels
                          ? "불러오는 중…"
                          : discoveredImageTagLocalModels.length > 0
                            ? `${discoveredImageTagLocalModels.length}개 · 새로고침`
                            : "못 찾음 · 다시 시도"}
                      </span>
                    </button>
                  </label>
                ) : null}
                {/* The annotation pass is the largest prompt of the turn, so this value — not the main model's —
                    is what decides its detail tier. Sharing the main model's context here silently mis-sized it. */}
                <label>
                  컨텍스트 (토큰)
                  <input
                    type="number"
                    min="2048"
                    step="1024"
                    value={state.imageTagLlm.contextTokens ?? imageTagLlmProvider.contextTokens}
                    onChange={(event) =>
                      onImageTagLlmChange({ contextTokens: Math.max(2048, Number(event.target.value) || 0) })
                    }
                    onBlur={() => notifyApplied("이미지 태그 모델 컨텍스트 크기가 적용되었습니다.")}
                  />
                </label>
                <label>
                  API 키
                  <input
                    type="password"
                    value={state.imageTagLlm.apiKey}
                    onChange={(event) => onImageTagLlmChange({ apiKey: event.target.value, registrationStatus: "idle", verificationMessage: "" })}
                    placeholder={imageTagLlmProvider.keyPlaceholder}
                    disabled={state.imageTagLlm.provider === "mock" || isCliAgentLlmProvider(state.imageTagLlm.provider)}
                  />
                </label>
                <div className="runtime-model-grid compact">
                  <label>
                    응답 출력 토큰 <span className="inline-value">{state.imageTagLlm.maxTokens}</span>
                    <input
                      type="number"
                      min="256"
                      max="4000"
                      step="100"
                      value={state.imageTagLlm.maxTokens}
                      onChange={(event) => onImageTagLlmChange({ maxTokens: Number(event.target.value) })}
                    />
                  </label>
                  <label>
                    온도 <span className="inline-value">{state.imageTagLlm.temperature.toFixed(2)}</span>
                    <input
                      type="range"
                      min="0"
                      max="1.5"
                      step="0.05"
                      value={state.imageTagLlm.temperature}
                      onChange={(event) => onImageTagLlmChange({ temperature: Number(event.target.value) })}
                    />
                  </label>
                </div>
              </>
            ) : (
              <p className="settings-note">메인 LLM을 사용합니다.</p>
            )}
          </div>
        </>
      ) : null}

      {shows("image") ? (
        <>
          <SectionTitle icon={<Sparkles size={17} />} title="NovelAI 런타임" />
          <div className="runtime-model-card">
            <div className="runtime-model-heading">
              <div>
                <span>제작 설정</span>
                <strong>{state.novelAi.modelPreset} · {state.imageProfile.width}x{state.imageProfile.height}</strong>
              </div>
              <StatusPill icon={<Sparkles size={15} />} label={novelAiStatusLabel} tone={novelAiNeedsSecret || state.novelAi.registrationStatus === "failed" ? "neutral" : "good"} />
            </div>
            <div className="runtime-summary-grid">
              <div>
                <span>샘플러</span>
                <strong>{state.novelAi.sampler}</strong>
              </div>
              <div>
                <span>스케줄러</span>
                <strong>{state.novelAi.noiseSchedule}</strong>
              </div>
              <div>
                <span>밀도</span>
                <strong>{activeImageGenerationCadence.label}</strong>
              </div>
              <div>
                <span>요청</span>
                <strong>{state.novelAi.requestMode}</strong>
              </div>
            </div>
          </div>
          {novelAiNeedsSecret || state.novelAi.registrationStatus === "failed" || state.novelAi.registrationStatus === "verifying" ? (
            <ApiStatusCard
              title="NovelAI 등록 상태"
              status={state.novelAi.registrationStatus}
              message={state.novelAi.verificationMessage}
              verifiedAt={state.novelAi.verifiedAt}
              detail={state.novelAi.subscriptionTier ? `tier: ${state.novelAi.subscriptionTier}` : undefined}
            />
          ) : null}
          <div className="runtime-parameter-card">
            <div className="runtime-card-subhead">
              <strong>연결</strong>
              <span>{state.novelAi.requestMode}</span>
            </div>
            <div className="two-fields">
              <label className="checkline">
                <input
                  checked={state.novelAi.enabled}
                  type="checkbox"
                  onChange={(event) => applyNovelAiChange({ enabled: event.target.checked }, event.target.checked ? "NovelAI 이미지 API가 활성화되었습니다." : "NovelAI 이미지 API가 비활성화되었습니다.")}
                />
                NovelAI 사용
              </label>
              <label className="checkline">
                <input
                  checked={state.novelAi.roundRobinEnabled}
                  type="checkbox"
                  onChange={(event) => applyNovelAiChange({ roundRobinEnabled: event.target.checked }, event.target.checked ? "NovelAI 라운드로빈이 활성화되었습니다." : "NovelAI 라운드로빈이 비활성화되었습니다.")}
                />
                계정 라운드로빈
              </label>
            </div>
            <label>
              요청 방식
              <select
                value={state.novelAi.requestMode}
                onChange={(event) => applyNovelAiChange({ requestMode: event.target.value as NovelAiApiSettings["requestMode"] }, `NovelAI ${event.target.value} 요청 모드가 적용되었습니다.`)}
              >
                <option value="mock">mock</option>
                <option value="direct">direct</option>
                <option value="proxy">proxy</option>
              </select>
            </label>
            <label>
              엔드포인트
              <input value={state.novelAi.endpoint} onChange={(event) => onNovelAiChange({ endpoint: event.target.value })} onBlur={() => notifyApplied("NovelAI 엔드포인트가 적용되었습니다.")} />
            </label>
            <label>
              프록시 URL
              <input value={state.novelAi.proxyUrl} onChange={(event) => onNovelAiChange({ proxyUrl: event.target.value })} onBlur={() => notifyApplied("NovelAI 프록시 URL이 적용되었습니다.")} />
            </label>
            {novelAiNeedsSecret ? (
              <div className="readonly-field runtime-secret-state warning">
                <span>개인 NovelAI 토큰</span>
                <strong>개인 설정에서 NovelAI 토큰 등록 필요</strong>
                <small>제작자 토큰은 포함되지 않으므로 실행자의 개인 토큰만 사용합니다.</small>
              </div>
            ) : null}
          </div>

          <div className="runtime-parameter-card">
            <div className="runtime-card-subhead">
              <strong>모델과 출력</strong>
              <span>{state.novelAi.modelPreset}</span>
            </div>
            <label>
              모델 프리셋
              <select
                value={state.novelAi.modelPreset}
                onChange={(event) => applyNovelAiChange({ modelPreset: event.target.value as NovelAiModelPreset }, `${event.target.value} 모델 프리셋이 적용되었습니다.`)}
              >
                {novelAiModelPresets.map((preset) => (
                  <option key={preset} value={preset}>
                    {preset}
                  </option>
                ))}
              </select>
            </label>
            <label>
              해상도
              <select
                value={activeRuntimeResolution}
                onChange={(event) => {
                  const preset = resolutionPresets.find((item) => `${item.width}x${item.height}` === event.target.value);
                  if (preset) {
                    applyImageProfileChange({ width: preset.width, height: preset.height }, `${preset.label} 해상도가 적용되었습니다.`);
                  }
                }}
              >
                {resolutionPresets.map((preset) => (
                  <option key={preset.label} value={`${preset.width}x${preset.height}`}>
                    {preset.label}
                  </option>
                ))}
                <option value="custom">직접 입력</option>
              </select>
            </label>
            <div className="two-fields">
              <label>
                너비
                <input
                  min="256"
                  max="2048"
                  step="64"
                  type="number"
                  value={state.imageProfile.width}
                  onChange={(event) => onImageProfileChange({ width: Number(event.target.value) })}
                  onBlur={() => notifyApplied("이미지 해상도 설정이 적용되었습니다.")}
                />
              </label>
              <label>
                높이
                <input
                  min="256"
                  max="2048"
                  step="64"
                  type="number"
                  value={state.imageProfile.height}
                  onChange={(event) => onImageProfileChange({ height: Number(event.target.value) })}
                  onBlur={() => notifyApplied("이미지 해상도 설정이 적용되었습니다.")}
                />
              </label>
            </div>
            <div className="two-fields">
              <label>
                스텝
                <input
                  type="number"
                  min="1"
                  max="60"
                  value={state.imageProfile.steps}
                  onChange={(event) => onImageProfileChange({ steps: Number(event.target.value) })}
                  onBlur={() => notifyApplied("이미지 steps 설정이 적용되었습니다.")}
                />
              </label>
              <label>
                Guidance
                <input
                  type="number"
                  min="1"
                  max="15"
                  step="0.5"
                  value={state.imageProfile.promptGuidance}
                  onChange={(event) => onImageProfileChange({ promptGuidance: Number(event.target.value) })}
                  onBlur={() => notifyApplied("이미지 guidance 설정이 적용되었습니다.")}
                />
              </label>
            </div>
          </div>

          <div className="runtime-parameter-card">
            <div className="runtime-card-subhead">
              <strong>샘플링</strong>
              <span>{state.novelAi.sampler}</span>
            </div>
            <div className="two-fields">
              <label>
                샘플러
                <select value={state.novelAi.sampler} onChange={(event) => applyNovelAiChange({ sampler: event.target.value }, `${event.target.value} sampler가 적용되었습니다.`)}>
                  {novelAiSamplers.map((sampler) => (
                    <option key={sampler} value={sampler}>
                      {sampler}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                스케줄러
                <select
                  value={state.novelAi.noiseSchedule}
                  onChange={(event) => applyNovelAiChange({ noiseSchedule: event.target.value as NovelAiNoiseSchedule }, `${event.target.value} scheduler가 적용되었습니다.`)}
                >
                  {novelAiNoiseSchedules.map((schedule) => (
                    <option key={schedule} value={schedule}>
                      {schedule}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="two-fields">
              <label>
                CFG 보정
                <input min="0" max="1" step="0.02" type="number" value={state.novelAi.cfgRescale} onChange={(event) => onNovelAiChange({ cfgRescale: Number(event.target.value) })} onBlur={() => notifyApplied("NovelAI CFG rescale이 적용되었습니다.")} />
              </label>
              <label>
                UC preset
                <input type="number" min="0" max="3" value={state.novelAi.ucPreset} onChange={(event) => onNovelAiChange({ ucPreset: Number(event.target.value) })} onBlur={() => notifyApplied("NovelAI UC preset이 적용되었습니다.")} />
              </label>
            </div>
            <div className="two-fields">
              <label className="checkline">
                <input checked={state.novelAi.seedFixed} type="checkbox" onChange={(event) => applyNovelAiChange({ seedFixed: event.target.checked }, event.target.checked ? "시드 고정이 활성화되었습니다." : "시드 고정이 비활성화되었습니다.")} />
                시드 고정
              </label>
              <label className="checkline">
                <input checked={state.novelAi.varPlus} type="checkbox" onChange={(event) => applyNovelAiChange({ varPlus: event.target.checked }, event.target.checked ? "VAR+가 활성화되었습니다." : "VAR+가 비활성화되었습니다.")} />
                VAR+
              </label>
            </div>
            <label>
              시드
              <input type="number" value={state.novelAi.seed ?? ""} onChange={(event) => onNovelAiChange({ seed: event.target.value ? Number(event.target.value) : undefined })} onBlur={() => notifyApplied("NovelAI seed 설정이 적용되었습니다.")} />
            </label>
          </div>

          <div className="runtime-parameter-card">
            <div className="runtime-card-subhead">
              <strong>Vibe Transfer</strong>
              <span>{state.novelAi.vibeTransferEnabled ? `${state.novelAi.vibeTransferReferences.length}장` : "off"}</span>
            </div>
            <label className="checkline">
              <input
                checked={state.novelAi.vibeTransferEnabled}
                type="checkbox"
                onChange={(event) => applyNovelAiChange({ vibeTransferEnabled: event.target.checked }, event.target.checked ? "Vibe Transfer가 활성화되었습니다." : "Vibe Transfer가 비활성화되었습니다.")}
              />
              Vibe Transfer 사용
            </label>
            <p className="settings-note">참조 이미지를 첨부하면 NovelAI 이미지 생성 요청에 함께 전송됩니다. 이미지마다 reference strength와 information extracted를 조절할 수 있습니다.{requiresVibeEncoding ? " v4/v4.5 모델은 생성 전에 각 이미지를 한 번 인코딩해야 하며(1회 2 Anlas), 모델이나 information extracted를 바꾸면 다시 인코딩하세요." : ""}</p>
            {/* V5 refuses vibe arrays outright, so the client leaves them out of every V5 request. Said here rather
                than discovered as "the vibe did nothing". */}
            <p className={`settings-note${novelAiIsV5 && state.novelAi.vibeTransferEnabled ? " warning" : " muted"}`}>
              V5 프리셋(NAID5F · NAID5C · NAID5)은 Vibe Transfer를 지원하지 않아, 그 모델로 생성할 때는 참조 이미지를 보내지 않고 건너뜁니다.
            </p>
            <label className="vibe-transfer-import icon-text-button full">
              <ImagePlus size={16} />
              참조 이미지 추가
              <input
                type="file"
                accept="image/*"
                multiple
                style={{ display: "none" }}
                onChange={(event) => {
                  addVibeTransferReferences(event.target.files);
                  event.target.value = "";
                }}
              />
            </label>
            {state.novelAi.vibeTransferReferences.length === 0 ? (
              <p className="settings-note muted">아직 추가된 참조 이미지가 없습니다.</p>
            ) : (
              <div className="vibe-transfer-list">
                {state.novelAi.vibeTransferReferences.map((reference) => {
                  const encodeStatus = vibeEncodeStatus[reference.id];
                  const isEncoding = encodeStatus?.state === "encoding";
                  const isEncoded =
                    Boolean(reference.encodedVibe) &&
                    reference.encodedModel === state.novelAi.modelPreset &&
                    reference.encodedInformationExtracted === reference.informationExtracted;
                  const isStale = Boolean(reference.encodedVibe) && !isEncoded;
                  return (
                    <div className="vibe-transfer-item" key={reference.id}>
                      <div className="vibe-transfer-item-head">
                        <img className="vibe-transfer-thumb" src={reference.image} alt={reference.name || "vibe reference"} />
                        <div className="vibe-transfer-item-meta">
                          <strong title={reference.name}>{reference.name || "참조 이미지"}</strong>
                          <button type="button" className="icon-text-button subtle" onClick={() => removeVibeTransferReference(reference.id)}>
                            제거
                          </button>
                        </div>
                      </div>
                      <label>
                        Reference strength <span className="inline-value">{reference.referenceStrength.toFixed(2)}</span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={reference.referenceStrength}
                          onChange={(event) => updateVibeTransferReference(reference.id, { referenceStrength: Number(event.target.value) })}
                          onMouseUp={() => notifyApplied("Vibe Transfer reference strength가 적용되었습니다.")}
                          onTouchEnd={() => notifyApplied("Vibe Transfer reference strength가 적용되었습니다.")}
                        />
                      </label>
                      <label>
                        Information extracted <span className="inline-value">{reference.informationExtracted.toFixed(2)}</span>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={reference.informationExtracted}
                          onChange={(event) => updateVibeTransferReference(reference.id, { informationExtracted: Number(event.target.value) })}
                          onMouseUp={() => notifyApplied("Vibe Transfer information extracted가 적용되었습니다.")}
                          onTouchEnd={() => notifyApplied("Vibe Transfer information extracted가 적용되었습니다.")}
                        />
                      </label>
                      {requiresVibeEncoding ? (
                        <div className="vibe-transfer-encode-row">
                          <span className={`vibe-encode-state ${isEncoded ? "ok" : isStale ? "stale" : "pending"}`}>
                            {isEncoding ? "인코딩 중…" : isEncoded ? "인코딩됨" : isStale ? "재인코딩 필요" : "미인코딩"}
                          </span>
                          <button
                            type="button"
                            className="icon-text-button subtle"
                            disabled={!canEncodeVibe || isEncoding}
                            onClick={() => encodeVibeTransferReference(reference)}
                          >
                            <WandSparkles size={14} />
                            {isEncoded ? "다시 인코딩" : "인코딩"}
                          </button>
                          {reference.encodedVibe ? (
                            <button
                              type="button"
                              className="icon-text-button subtle"
                              disabled={isEncoding}
                              onClick={() => clearVibeTransferEncoding(reference.id)}
                            >
                              인코딩 해제
                            </button>
                          ) : null}
                        </div>
                      ) : null}
                      {requiresVibeEncoding && !canEncodeVibe ? (
                        <p className="settings-note muted">인코딩하려면 개인 설정에서 NovelAI 토큰을 먼저 등록하세요.</p>
                      ) : null}
                      {encodeStatus?.state === "error" ? <p className="settings-note warning">{encodeStatus.message}</p> : null}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="runtime-parameter-card">
            <div className="runtime-card-subhead">
              <strong>캐릭터 레퍼런스 (Precise Reference)</strong>
              <span>{state.novelAi.characterReferenceEnabled ? "on" : "off"}</span>
            </div>
            <label className="checkline">
              <input
                checked={state.novelAi.characterReferenceEnabled === true}
                type="checkbox"
                onChange={(event) =>
                  applyNovelAiChange(
                    { characterReferenceEnabled: event.target.checked },
                    event.target.checked ? "NovelAI 캐릭터 레퍼런스가 활성화되었습니다." : "NovelAI 캐릭터 레퍼런스가 비활성화되었습니다."
                  )
                }
              />
              캐릭터 NovelAI 프로필 이미지로 얼굴 고정
            </label>
            {/* Opt-in because it is the one NovelAI feature that costs Anlas per image even on Opus. */}
            <p className="settings-note">
              이미지당 +5 Anlas (Opus 포함) · V4.5 전용. 캐릭터에 NovelAI 프로필 이미지가 있어야 적용되며, 레퍼런스가 붙는 컷에서는 Vibe Transfer가 빠집니다.
            </p>
            {state.novelAi.characterReferenceEnabled && !novelAiSupportsCharacterReference ? (
              <p className="settings-note warning">
                현재 모델 {state.novelAi.modelPreset}은(는) Precise Reference를 지원하지 않아 레퍼런스 없이 생성됩니다. NAID4.5F 또는 NAID4.5C를 고르세요.
              </p>
            ) : null}
            <label>
              레퍼런스 강도{" "}
              <span className="inline-value">
                {(state.novelAi.characterReferenceStrength ?? NOVELAI_CHARACTER_REFERENCE_DEFAULT_STRENGTH).toFixed(2)}
              </span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                disabled={state.novelAi.characterReferenceEnabled !== true}
                value={state.novelAi.characterReferenceStrength ?? NOVELAI_CHARACTER_REFERENCE_DEFAULT_STRENGTH}
                onChange={(event) => onNovelAiChange({ characterReferenceStrength: Number(event.target.value) })}
                onMouseUp={() => notifyApplied("캐릭터 레퍼런스 강도가 적용되었습니다.")}
                onTouchEnd={() => notifyApplied("캐릭터 레퍼런스 강도가 적용되었습니다.")}
              />
            </label>
            <label>
              충실도{" "}
              <span className="inline-value">
                {(state.novelAi.characterReferenceFidelity ?? NOVELAI_CHARACTER_REFERENCE_DEFAULT_FIDELITY).toFixed(2)}
              </span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                disabled={state.novelAi.characterReferenceEnabled !== true}
                value={state.novelAi.characterReferenceFidelity ?? NOVELAI_CHARACTER_REFERENCE_DEFAULT_FIDELITY}
                onChange={(event) => onNovelAiChange({ characterReferenceFidelity: Number(event.target.value) })}
                onMouseUp={() => notifyApplied("캐릭터 레퍼런스 충실도가 적용되었습니다.")}
                onTouchEnd={() => notifyApplied("캐릭터 레퍼런스 충실도가 적용되었습니다.")}
              />
            </label>
            <p className="runtime-hint">레퍼런스의 자세나 표정이 모든 컷에 따라붙으면 강도를 낮추세요.</p>
            <label className="checkline">
              <input
                checked={state.novelAi.characterReferenceStyleAware === true}
                disabled={state.novelAi.characterReferenceEnabled !== true}
                type="checkbox"
                onChange={(event) =>
                  applyNovelAiChange(
                    { characterReferenceStyleAware: event.target.checked },
                    event.target.checked ? "레퍼런스의 화풍까지 따릅니다." : "레퍼런스에서 캐릭터만 가져옵니다."
                  )
                }
              />
              화풍까지 따르기 (style aware)
            </label>
          </div>

          <div className="runtime-parameter-card">
            <div className="runtime-card-subhead">
              <strong>자동 생성</strong>
              <span>{state.novelAi.automationTermination}</span>
            </div>
            <div className="two-fields">
              <label>
                지연 초
                <input min="0" max="120" type="number" value={state.novelAi.generationDelaySeconds} onChange={(event) => onNovelAiChange({ generationDelaySeconds: Number(event.target.value) })} onBlur={() => notifyApplied("NovelAI 생성 지연 설정이 적용되었습니다.")} />
              </label>
              <label>
                반복
                <input min="1" max="100" type="number" value={state.novelAi.repeatCount} onChange={(event) => onNovelAiChange({ repeatCount: Number(event.target.value) })} onBlur={() => notifyApplied("NovelAI repeat 설정이 적용되었습니다.")} />
              </label>
            </div>
            <label className="checkline">
              <input checked={state.novelAi.randomDelayEnabled} type="checkbox" onChange={(event) => applyNovelAiChange({ randomDelayEnabled: event.target.checked }, event.target.checked ? "랜덤 지연이 활성화되었습니다." : "랜덤 지연이 비활성화되었습니다.")} />
              랜덤 지연
            </label>
            <div className="two-fields">
              <label>
                종료 조건
                <select value={state.novelAi.automationTermination} onChange={(event) => applyNovelAiChange({ automationTermination: event.target.value as NovelAiAutomationTermination }, "NovelAI 자동 생성 종료 조건이 적용되었습니다.")}>
                  {novelAiAutomationTerminations.map((termination) => (
                    <option key={termination} value={termination}>
                      {termination}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                수량 제한
                <input min="1" max="500" type="number" value={state.novelAi.countLimit} onChange={(event) => onNovelAiChange({ countLimit: Number(event.target.value) })} onBlur={() => notifyApplied("NovelAI count limit이 적용되었습니다.")} />
              </label>
            </div>
          </div>
          <button className="icon-text-button full" type="button" onClick={onOpenPersonalSettings}>
            <KeyRound size={16} />
            개인 설정에서 NovelAI 토큰 관리
          </button>
        </>
      ) : null}

      {shows("photoreal") ? (
        <>
          <SectionTitle
            icon={<ImageIcon size={17} />}
            title="실사 이미지 (ComfyUI)"
            className="settings-jump-target"
            headingRef={photorealSectionRef}
          />
          <PhotorealRuntimeSettings state={state} onChange={applyPhotorealChange} onNotify={notifyApplied} onStateChange={onStateChange} />
        </>
      ) : null}

      {shows("general") ? (
        <>
          <SectionTitle icon={<Play size={17} />} title="실제 시뮬레이션 검증" />
          <button className="send-button wide-action" type="button" onClick={verifyRuntimeTurn} disabled={isRuntimeVerifying}>
            <Play size={16} />
            {isRuntimeVerifying ? "턴 검증 중" : "실제 API 턴 테스트"}
          </button>
          {runtimeCheck ? <p className="settings-note">{runtimeCheck}</p> : null}

          <SectionTitle icon={<Database size={17} />} title="NeuralMap" />
          <label className="checkline">
            <input
              checked={state.neuralMap.enabled}
              type="checkbox"
              onChange={(event) =>
                applyStateChange(
                  (current) => ({
                    ...current,
                    neuralMap: {
                      ...current.neuralMap,
                      enabled: event.target.checked
                    }
                  }),
                  event.target.checked ? "NeuralMap 연동이 활성화되었습니다." : "NeuralMap 연동이 비활성화되었습니다."
                )
              }
            />
            API 연동 사용
          </label>
          <label>
            기본 URL
            <input
              value={state.neuralMap.baseUrl}
              onChange={(event) =>
                onStateChange((current) => ({
                  ...current,
                  neuralMap: {
                    ...current.neuralMap,
                    baseUrl: event.target.value
                  }
                }))
              }
              onBlur={() => notifyApplied("NeuralMap 기본 URL이 적용되었습니다.")}
            />
          </label>
          <label>
            검색 문맥 토큰 예산
            <input
              min="800"
              max="12000"
              step="200"
              type="number"
              value={state.neuralMap.tokenBudget}
              onChange={(event) =>
                onStateChange((current) => ({
                  ...current,
                  neuralMap: {
                    ...current.neuralMap,
                    tokenBudget: Number(event.target.value)
                  }
                }))
              }
              onBlur={() => notifyApplied("NeuralMap token budget이 적용되었습니다.")}
            />
          </label>
          <div className="danger-zone">
            <strong>위험 구역</strong>
            <p>이 브라우저에 저장된 모든 시뮬레이션과 생성 이미지를 삭제하고 기본 데모 상태로 되돌립니다.</p>
            <button className="icon-text-button full danger-action" type="button" onClick={onResetDemo}>
              <Trash2 size={16} />
              모든 로컬 데이터 삭제
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}

/** How long the custom-workflow text has to rest before it is validated (see PhotorealRuntimeSettings). */
const CUSTOM_WORKFLOW_VALIDATION_DELAY_MS = 600;

/**
 * `value` once it has stopped changing for `delayMs`, plus a function that takes the current value at once (for a
 * blur, when the person is done typing).
 */
function useSettledValue<T>(value: T, delayMs: number): [T, () => void] {
  const [settled, setSettled] = useState(value);
  const latestRef = useRef(value);
  latestRef.current = value;
  useEffect(() => {
    if (Object.is(settled, value)) {
      return;
    }
    const timer = window.setTimeout(() => setSettled(value), delayMs);
    return () => window.clearTimeout(timer);
  }, [delayMs, settled, value]);
  const settleNow = useCallback(() => setSettled(latestRef.current), []);
  return [settled, settleNow];
}

/**
 * A failed ComfyUI connection check as a notice. Every failure carries "실패", which is what makes the toast a sticky
 * error (showRuntimeNotice); the relay's HTTP-error path already words it this way, the rest named only the cause.
 */
function formatComfyUiProbeFailure(message: string): string {
  const reason = message.trim();
  if (/^ComfyUI 연결 확인 실패/u.test(reason)) {
    return reason;
  }
  return `ComfyUI 연결 확인 실패: ${reason || "응답을 해석하지 못했습니다."}`;
}

/**
 * A number field that keeps the user's draft while they type and commits on blur or Enter (Escape discards).
 *
 * App runs hydrateState on every state write, and normalizePhotorealSettings clamps numbers as it goes — sizes snap
 * onto 256..2048 in steps of 16, the timeout onto 30..900 — so a field bound straight to state turned the "1" of
 * "1024" into 256 at the first keystroke. While focused the draft belongs to the user; once focus leaves, the field
 * shows the STORED value again, so a snapped or rejected entry is visibly corrected instead of quietly kept.
 */
function CommittedNumberInput({
  value,
  emptyValue,
  min,
  max,
  step,
  integer = false,
  placeholder,
  disabled,
  ariaLabel,
  ariaDescribedBy,
  onBelowMin,
  onCommit
}: {
  value: number | undefined;
  /** What an emptied field commits, and the stored value shown as an empty field (so the placeholder explains it). */
  emptyValue?: number;
  min: number;
  max: number;
  step?: number;
  integer?: boolean;
  placeholder?: string;
  disabled?: boolean;
  ariaLabel?: string;
  ariaDescribedBy?: string;
  /**
   * When set, a value below `min` is refused instead of raised to `min`: the field goes back to the stored value and
   * this is told what was typed, so the panel can say why. For a bound that is a rule rather than a range end — a
   * photoreal CFG of 0.5 is not "1", it is a value that is not allowed.
   */
  onBelowMin?: (typed: number) => void;
  onCommit: (value: number | undefined) => void;
}) {
  const [draft, setDraft] = useState(() => formatCommittedNumber(value, emptyValue));
  const [editing, setEditing] = useState(false);
  const discardRef = useRef(false);

  useEffect(() => {
    if (!editing) {
      setDraft(formatCommittedNumber(value, emptyValue));
    }
  }, [editing, emptyValue, value]);

  const commit = () => {
    setEditing(false);
    if (discardRef.current) {
      discardRef.current = false;
      return;
    }
    const text = draft.trim();
    if (!text) {
      if (value !== emptyValue) {
        onCommit(emptyValue);
      }
      return;
    }
    const parsed = Number(text);
    if (!Number.isFinite(parsed)) {
      return;
    }
    if (onBelowMin && parsed < min) {
      onBelowMin(parsed);
      return;
    }
    const bounded = Math.min(max, Math.max(min, integer ? Math.round(parsed) : parsed));
    if (bounded !== value) {
      onCommit(bounded);
    }
  };

  return (
    <input
      type="number"
      inputMode={integer ? "numeric" : "decimal"}
      min={min}
      max={max}
      step={step}
      value={draft}
      placeholder={placeholder}
      disabled={disabled}
      aria-label={ariaLabel}
      aria-describedby={ariaDescribedBy}
      onFocus={() => setEditing(true)}
      onChange={(event) => {
        setEditing(true);
        setDraft(event.target.value);
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          event.currentTarget.blur();
        } else if (event.key === "Escape") {
          discardRef.current = true;
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function formatCommittedNumber(value: number | undefined, emptyValue: number | undefined): string {
  return value === undefined || value === emptyValue ? "" : String(value);
}

type PhotorealModelFileField = "diffusionModel" | "textEncoder" | "vae";

const PHOTOREAL_MODEL_FILE_SLOTS: Array<{ field: PhotorealModelFileField; folder: PhotorealPresetDownload["folder"]; label: string }> = [
  { field: "diffusionModel", folder: "diffusion_models", label: "확산 모델" },
  { field: "textEncoder", folder: "text_encoders", label: "텍스트 인코더" },
  { field: "vae", folder: "vae", label: "VAE" }
];

interface PhotorealFileCheck {
  field: PhotorealModelFileField;
  folder: PhotorealPresetDownload["folder"];
  label: string;
  file: string;
  /** `elsewhere`: ComfyUI has a file of that name in a subfolder — the loader needs that exact path instead. */
  status: "installed" | "elsewhere" | "missing" | "unknown";
  foundAs?: string;
  download?: PhotorealPresetDownload;
}

/**
 * The model files a preset will ask ComfyUI to load, checked against what the probe found. The loaders accept a
 * file only under the exact name ComfyUI lists, so a file sitting in a subfolder is reported with the path to use
 * rather than as installed.
 */
function checkPhotorealPresetFiles(
  settings: PhotorealApiSettings,
  presetId: PhotorealWorkflowPresetId,
  probe: ComfyUiProbeResult
): PhotorealFileCheck[] {
  // A custom workflow names its own files inside the graph; there is nothing generic to check.
  if (presetId === "custom") {
    return [];
  }
  const preset = PHOTOREAL_WORKFLOW_PRESETS[presetId];
  const files = resolvePhotorealModelFiles(settings, presetId);
  const baseName = (value: string) => value.trim().replace(/\\/gu, "/").split("/").pop() ?? value;
  return PHOTOREAL_MODEL_FILE_SLOTS.map(({ field, folder, label }) => {
    const file = files[field];
    const download = preset.downloads.find((entry) => entry.folder === folder && entry.file === baseName(file));
    const listed = probe.models?.[folder];
    if (!listed) {
      return { field, folder, label, file, status: "unknown", download };
    }
    if (listed.some((entry) => entry.trim() === file.trim())) {
      return { field, folder, label, file, status: "installed" };
    }
    const elsewhere = listed.find((entry) => baseName(entry) === baseName(file));
    return elsewhere
      ? { field, folder, label, file, status: "elsewhere", foundAs: elsewhere }
      : { field, folder, label, file, status: "missing", download };
  });
}

/**
 * Whether a preset's pictures may be used commercially — the one preset fact that changes what a creator may do
 * with the output (the Qwen-Image 2.1 family is research-licence only). A custom workflow carries the licence of
 * whatever model it loads, so it gets a "check it" tag rather than a verdict.
 */
function PhotorealLicenseTag({ preset }: { preset: (typeof PHOTOREAL_WORKFLOW_PRESETS)[PhotorealWorkflowPresetId] }) {
  const label = preset.id === "custom" ? "라이선스 확인 필요" : preset.commercialUse ? "상업 이용 가능" : "비상업 라이선스";
  return <span className={`photoreal-license-tag${preset.commercialUse && preset.id !== "custom" ? "" : " noncommercial"}`}>{label}</span>;
}

function formatVramGb(megabytes: number): string {
  return `${(megabytes / 1024).toFixed(1)}GB`;
}

interface PhotorealProbeSnapshot {
  simulationId: string;
  baseUrl: string;
  result: ComfyUiProbeResult;
  checkedAt: string;
}

/**
 * The 실사 이미지 (ComfyUI) runtime section: connection + probe, workflow presets, numbers, style, face reference,
 * GPU handoff and the custom workflow. Settings live on the simulation (state.photoreal); hydrateState normalizes
 * them on every write, which is why every numeric field here is a CommittedNumberInput.
 */
function PhotorealRuntimeSettings({
  state,
  onChange,
  onNotify,
  onStateChange
}: {
  state: AppState;
  onChange: (patch: Partial<PhotorealApiSettings>, message?: string) => void;
  onNotify: (message: string) => void;
  onStateChange: Dispatch<SetStateAction<AppState>>;
}) {
  const photoreal = state.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS;
  const [probe, setProbe] = useState<PhotorealProbeSnapshot>();
  const [isProbing, setIsProbing] = useState(false);
  const generationPreset = PHOTOREAL_WORKFLOW_PRESETS[photoreal.workflowPreset] ?? PHOTOREAL_WORKFLOW_PRESETS.qwen_image_2_1_turbo;
  const referencePreset = PHOTOREAL_WORKFLOW_PRESETS[photoreal.referenceWorkflowPreset] ?? PHOTOREAL_WORKFLOW_PRESETS.qwen_image_2_1_turbo;
  // Only presets that can take a face reference belong in the face-lock select. A stored id outside that set (an
  // older state) is still listed, marked, so the select shows what is actually stored instead of a silent first row.
  const referencePresetOptions = photorealWorkflowPresetOrder.filter(
    (presetId) => PHOTOREAL_WORKFLOW_PRESETS[presetId].supportsReferences || presetId === photoreal.referenceWorkflowPreset
  );
  const usesCustomWorkflow = photoreal.workflowPreset === "custom" || photoreal.referenceWorkflowPreset === "custom";
  // The same template checks the cut will run (structure, and the safety floor over the template's own words), minus
  // the actual cut; errors are Korean. Run on the text as it stands after a typing pause (or on leaving the field),
  // not per keystroke: the parse error moves with every character ("position 12", "position 30", …), and a live
  // region repeating it would talk over the person typing.
  const [settledCustomWorkflow, settleCustomWorkflow] = useSettledValue(photoreal.customWorkflow, CUSTOM_WORKFLOW_VALIDATION_DELAY_MS);
  const customWorkflowErrors = useMemo(() => validatePhotorealCustomWorkflow(state, settledCustomWorkflow), [state, settledCustomWorkflow]);
  const customWorkflowStatusId = `photoreal-workflow-status-${state.simulation.id}`;
  // From useId rather than the simulation id: the rail and the mobile drawer can both hold this panel at once.
  const styleScreenId = useId();
  const appearanceScreenId = useId();
  const cfgStatusId = useId();
  // Why the last CFG typed was not applied. Set when the field commits (blur / Enter), never per keystroke, cleared by
  // the next value that is applied, and shown only for the simulation it was typed in (the panel can switch).
  const [cfgRefusalNote, setCfgRefusalNote] = useState<{ simulationId: string; message: string }>();
  const cfgRefusal = cfgRefusalNote?.simulationId === state.simulation.id ? cfgRefusalNote.message : undefined;
  const effectiveSize = resolvePhotorealOutputSize(state);
  const activeStyle = photoreal.stylePreset === "custom" ? undefined : PHOTOREAL_STYLE_PRESETS[photoreal.stylePreset];
  const gpuHandoffActive = shouldCoordinateLocalGpu(state);
  const statusLabel = !photoreal.enabled
    ? "비활성"
    : photoreal.requestMode === "mock"
      ? "mock"
      : isProbing
        ? "확인 중"
        : photoreal.registrationStatus === "registered"
          ? "연결됨"
          : "서버 필요";
  // A probe describes one ComfyUI address for one simulation; after a switch of either it would describe the wrong one.
  const visibleProbe =
    probe && probe.simulationId === state.simulation.id && probe.baseUrl === photoreal.baseUrl.trim() ? probe : undefined;
  // The face-lock preset is only used while face lock is on, so it is only checked then (see collectProbeNodes).
  const probedPresetIds = Array.from(
    new Set(photoreal.faceReferenceEnabled ? [photoreal.workflowPreset, photoreal.referenceWorkflowPreset] : [photoreal.workflowPreset])
  );
  const fileChecks = visibleProbe
    ? probedPresetIds.map((presetId) => ({ presetId, checks: checkPhotorealPresetFiles(photoreal, presetId, visibleProbe.result) }))
    : [];
  // Every file in place and no probed node missing. Nodes of presets that were not probed count as present: the core
  // nodes a built-in preset needs ship with ComfyUI, and only the configured presets' nodes are asked about.
  const presetRunsHere = (presetId: PhotorealWorkflowPresetId): boolean =>
    Boolean(visibleProbe?.result.ok) &&
    presetId !== "custom" &&
    checkPhotorealPresetFiles(photoreal, presetId, visibleProbe!.result).every((check) => check.status === "installed") &&
    PHOTOREAL_WORKFLOW_PRESETS[presetId].requiredNodes.every((node) => visibleProbe?.result.nodes?.[node] !== false);
  // The chosen preset cannot run on this ComfyUI but another built-in one can: offer that one in a click instead of
  // leaving the person with a list of downloads (the default preset needs files and nodes a typical install lacks).
  const installedPresetId =
    visibleProbe?.result.ok && !presetRunsHere(photoreal.workflowPreset)
      ? photorealWorkflowPresetOrder.find((presetId) => presetId !== photoreal.workflowPreset && presetRunsHere(presetId))
      : undefined;
  const [reprobeRequested, setReprobeRequested] = useState(false);
  const switchToInstalledPreset = (presetId: PhotorealWorkflowPresetId) => {
    // Face lock runs the face-lock preset; if that one cannot run here either, a locked cut would fail, so it goes off.
    const keepFaceLock = photoreal.faceReferenceEnabled && presetRunsHere(photoreal.referenceWorkflowPreset);
    onChange(
      { workflowPreset: presetId, ...(keepFaceLock ? {} : { faceReferenceEnabled: false }) },
      `생성용 워크플로를 ${PHOTOREAL_WORKFLOW_PRESETS[presetId].label}(으)로 바꿨습니다.${
        photoreal.faceReferenceEnabled && !keepFaceLock ? " 얼굴 고정용 워크플로가 이 PC에 없어 얼굴 고정은 껐습니다." : ""
      }`
    );
    setReprobeRequested(true);
  };

  const runProbe = async () => {
    // Settings are per simulation and the probe result is written back into them, so it carries the same ownership
    // guard as any other post-await write: a switch of simulation or progress run mid-probe drops the result. The
    // in-flight state stays in this component (isProbing) rather than in the settings: a "verifying" written there
    // would be stranded in a simulation the user left before the answer came back.
    const owner = readRunOwner(state);
    const target = { simulationId: state.simulation.id, baseUrl: photoreal.baseUrl.trim() };
    setIsProbing(true);
    try {
      // Never throws: an unreachable relay or ComfyUI comes back as ok:false with a Korean message.
      const result = await probeComfyUi(state);
      const checkedAt = new Date().toISOString();
      setProbe({ ...target, result, checkedAt });
      onStateChange((current) =>
        ownsActiveRun(current, owner)
          ? {
              ...current,
              photoreal: {
                ...(current.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS),
                registrationStatus: result.ok ? "registered" : "failed",
                verifiedAt: result.ok ? checkedAt : current.photoreal?.verifiedAt,
                verificationMessage: result.message
              }
            }
          : current
      );
      // The toast takes its tone from its words (showRuntimeNotice: 실패 → a sticky error). Most probe failures name
      // the cause without that word ("…HTTP 404로 응답했습니다", "…끝나지 않았습니다"), so they showed as a green check that
      // faded away while the panel above said the check had failed.
      onNotify(result.ok ? result.message : formatComfyUiProbeFailure(result.message));
    } finally {
      setIsProbing(false);
    }
  };

  // After a one-click preset switch, check again on the render that carries the new settings, so the warning about
  // the old preset does not stay on screen describing a choice that is no longer made.
  useEffect(() => {
    if (!reprobeRequested) {
      return;
    }
    setReprobeRequested(false);
    void runProbe();
    // runProbe is recreated every render; the switch is what this answers to.
  }, [reprobeRequested]);

  const setModelFile = (presetId: PhotorealWorkflowPresetId, field: PhotorealModelFileField, value: string) => {
    onChange({
      modelFiles: {
        ...photoreal.modelFiles,
        [presetId]: { ...photoreal.modelFiles[presetId], [field]: value }
      }
    });
  };

  return (
    <>
      <div className="runtime-model-card">
        <div className="runtime-model-heading">
          <div>
            <span>실사 렌더러</span>
            <strong>{generationPreset.label}</strong>
          </div>
          <StatusPill icon={<ImageIcon size={15} />} label={statusLabel} tone={statusLabel === "연결됨" ? "good" : "neutral"} />
        </div>
        <div className="runtime-summary-grid">
          <div>
            <span>ComfyUI</span>
            <strong>{photoreal.baseUrl.trim() || DEFAULT_PHOTOREAL_SETTINGS.baseUrl}</strong>
          </div>
          <div>
            <span>크기</span>
            <strong>
              {effectiveSize.width}x{effectiveSize.height}
            </strong>
          </div>
          <div>
            <span>스타일</span>
            <strong>{activeStyle?.label ?? "사용자 지정"}</strong>
          </div>
          <div>
            <span>얼굴 고정</span>
            <strong>{photoreal.faceReferenceEnabled ? (referencePreset.supportsReferences ? "켜짐" : "미지원 프리셋") : "꺼짐"}</strong>
          </div>
        </div>
        {photoreal.registrationStatus !== "idle" && photoreal.verificationMessage ? (
          <p className="runtime-hint">
            마지막 확인: {photoreal.verificationMessage}
            {photoreal.registrationStatus === "registered" && photoreal.verifiedAt
              ? ` (${new Date(photoreal.verifiedAt).toLocaleString("ko-KR")})`
              : ""}
          </p>
        ) : null}
      </div>

      <div className="runtime-parameter-card">
        <div className="runtime-card-subhead">
          <strong>연결</strong>
          <span>{photoreal.requestMode}</span>
        </div>
        <label className="checkline">
          <input
            checked={photoreal.enabled}
            type="checkbox"
            onChange={(event) =>
              onChange(
                { enabled: event.target.checked },
                event.target.checked ? "실사 이미지(ComfyUI)가 활성화되었습니다." : "실사 이미지(ComfyUI)가 비활성화되었습니다."
              )
            }
          />
          실사 이미지 사용
        </label>
        <div className="two-fields">
          <label>
            요청 방식
            <select
              value={photoreal.requestMode}
              onChange={(event) =>
                onChange(
                  { requestMode: event.target.value === "mock" ? "mock" : "proxy" },
                  `실사 이미지 ${event.target.value} 요청 모드가 적용되었습니다.`
                )
              }
            >
              <option value="proxy">proxy (API 서버 릴레이)</option>
              <option value="mock">mock (생성하지 않음)</option>
            </select>
          </label>
          <label>
            ComfyUI URL
            <input
              value={photoreal.baseUrl}
              placeholder={DEFAULT_PHOTOREAL_SETTINGS.baseUrl}
              onChange={(event) => onChange({ baseUrl: event.target.value, registrationStatus: "idle", verificationMessage: "" })}
              onBlur={() => onNotify("ComfyUI URL이 적용되었습니다.")}
            />
          </label>
        </div>
        <label>
          릴레이 주소 (선택)
          <input
            value={photoreal.proxyUrl}
            placeholder={getPhotorealRelayBase({ ...state, photoreal: { ...photoreal, proxyUrl: "" } })}
            onChange={(event) => onChange({ proxyUrl: event.target.value })}
            onBlur={() => onNotify("실사 릴레이 주소가 적용되었습니다.")}
          />
        </label>
        {/* Why there is a server in the path at all, since "just call ComfyUI" is the obvious question. */}
        <p className="runtime-hint">
          ComfyUI는 브라우저의 직접 호출을 받지 않아 DynamicChat API 서버가 대신 요청합니다. 이 PC(127.0.0.1)의 ComfyUI만 허용되며, 다른
          주소는 서버의 DYNAMICCHAT_COMFYUI_HOSTS에 등록해야 합니다.
        </p>
        <button className="icon-text-button full" type="button" onClick={() => void runProbe()} disabled={isProbing}>
          <RefreshCcw size={15} />
          {isProbing ? "확인 중…" : "연결 확인"}
        </button>
        {visibleProbe ? (
          <div className={`photoreal-probe${visibleProbe.result.ok ? " ok" : " failed"}`} role="status">
            <strong>{visibleProbe.result.message}</strong>
            {visibleProbe.result.ok ? (
              <dl className="photoreal-probe-facts">
                {visibleProbe.result.comfyuiVersion ? (
                  <>
                    <dt>ComfyUI</dt>
                    <dd>{visibleProbe.result.comfyuiVersion}</dd>
                  </>
                ) : null}
                {(visibleProbe.result.devices ?? []).map((device, index) => (
                  <Fragment key={`${device.name}-${index}`}>
                    <dt>GPU</dt>
                    <dd>
                      {device.name || "알 수 없는 장치"} · VRAM {formatVramGb(device.vramTotalMb)} (여유 {formatVramGb(device.vramFreeMb)})
                    </dd>
                  </Fragment>
                ))}
                {visibleProbe.result.queueRemaining !== undefined ? (
                  <>
                    <dt>대기열</dt>
                    <dd>{visibleProbe.result.queueRemaining}개</dd>
                  </>
                ) : null}
              </dl>
            ) : null}
            {installedPresetId ? (
              <div className="photoreal-probe-suggestion">
                <p className="runtime-hint">
                  {generationPreset.label}은(는) 이 ComfyUI에서 실행할 수 없습니다. {PHOTOREAL_WORKFLOW_PRESETS[installedPresetId].label}은(는) 필요한
                  파일이 모두 설치되어 있습니다.
                </p>
                <button className="icon-text-button full" type="button" onClick={() => switchToInstalledPreset(installedPresetId)}>
                  <Check size={15} />
                  {/* The label's parenthetical ("가장 빠름, 얼굴 고정 없음") is in the hint above; on the button it made the
                      rail scroll sideways. */}
                  {PHOTOREAL_WORKFLOW_PRESETS[installedPresetId].label.replace(/\s*\(.*\)\s*$/u, "")}(으)로 바꾸기
                </button>
              </div>
            ) : null}
            {visibleProbe.result.ok
              ? fileChecks.map(({ presetId, checks }) =>
                  checks.length > 0 ? (
                    <div className="photoreal-file-check" key={presetId}>
                      <span>{PHOTOREAL_WORKFLOW_PRESETS[presetId].label}</span>
                      <ul>
                        {checks.map((check) => (
                          <li className={`photoreal-file-row ${check.status}`} key={check.field}>
                            <span>
                              {check.status === "installed"
                                ? "설치됨"
                                : check.status === "elsewhere"
                                  ? "경로 다름"
                                  : check.status === "missing"
                                    ? "설치 필요"
                                    : "확인 불가"}
                            </span>
                            <code>
                              {check.folder}/{check.file}
                            </code>
                            {check.status === "elsewhere" && check.foundAs ? (
                              <button type="button" className="icon-text-button subtle" onClick={() => setModelFile(presetId, check.field, check.foundAs ?? check.file)}>
                                {check.foundAs} 사용
                              </button>
                            ) : null}
                            {check.status === "missing" && check.download ? (
                              <a href={check.download.url} target="_blank" rel="noreferrer">
                                다운로드 ({check.download.sizeGb}GB)
                              </a>
                            ) : null}
                          </li>
                        ))}
                      </ul>
                      {checks.some((check) => check.status === "missing") ? (
                        <p className="runtime-hint">받은 파일은 ComfyUI의 models/&lt;폴더&gt;/ 아래에 넣고 ComfyUI를 다시 시작하세요.</p>
                      ) : null}
                    </div>
                  ) : null
                )
              : null}
          </div>
        ) : null}
      </div>

      <div className="runtime-parameter-card">
        <div className="runtime-card-subhead">
          <strong>워크플로</strong>
          <span>{generationPreset.id === referencePreset.id ? "1개" : "2개"}</span>
        </div>
        <div className="two-fields">
          <label>
            생성용
            <select
              value={photoreal.workflowPreset}
              onChange={(event) => {
                const presetId = event.target.value as PhotorealWorkflowPresetId;
                onChange({ workflowPreset: presetId }, `생성용 워크플로를 ${PHOTOREAL_WORKFLOW_PRESETS[presetId].label}(으)로 바꿨습니다.`);
              }}
            >
              {photorealWorkflowPresetOrder.map((presetId) => (
                <option key={presetId} value={presetId}>
                  {PHOTOREAL_WORKFLOW_PRESETS[presetId].label}
                </option>
              ))}
            </select>
          </label>
          <label>
            얼굴 고정용
            <select
              value={photoreal.referenceWorkflowPreset}
              onChange={(event) => {
                const presetId = event.target.value as PhotorealWorkflowPresetId;
                onChange(
                  { referenceWorkflowPreset: presetId },
                  `얼굴 고정용 워크플로를 ${PHOTOREAL_WORKFLOW_PRESETS[presetId].label}(으)로 바꿨습니다.`
                );
              }}
            >
              {referencePresetOptions.map((presetId) => (
                <option key={presetId} value={presetId}>
                  {PHOTOREAL_WORKFLOW_PRESETS[presetId].label}
                  {PHOTOREAL_WORKFLOW_PRESETS[presetId].supportsReferences ? "" : " (얼굴 참조 미지원)"}
                </option>
              ))}
            </select>
          </label>
        </div>
        {/* The run screen only showed the state ("켜짐"); turning it off meant a trip to the builder. */}
        <label className="checkline">
          <input
            checked={photoreal.faceReferenceEnabled}
            type="checkbox"
            onChange={(event) =>
              onChange(
                { faceReferenceEnabled: event.target.checked },
                event.target.checked ? "실사 프로필 얼굴 고정을 켰습니다." : "실사 프로필 얼굴 고정을 껐습니다."
              )
            }
          />
          캐릭터 실사 프로필로 얼굴 고정
        </label>
        <p className="runtime-hint">
          켜면 프로필 사진의 얼굴만 이어지고, 머리 모양·메이크업·의상·배경은 컷 설명을 따릅니다. 끄면 컷마다 외형 설명대로 새로 그립니다.
        </p>
        {[generationPreset, ...(referencePreset.id !== generationPreset.id ? [referencePreset] : [])].map((preset) => (
          <div className="photoreal-preset-note" key={preset.id}>
            <PhotorealLicenseTag preset={preset} />
            <p className="runtime-hint">
              <strong>{preset.label}</strong> — {preset.description} 라이선스: {preset.license}. {preset.vramNote}
            </p>
          </div>
        ))}
        {!referencePreset.supportsReferences ? (
          <p className="settings-note warning">얼굴 고정용 프리셋이 참조 이미지를 받지 않아, 얼굴 고정 컷도 생성용 프리셋으로 참조 없이 생성됩니다.</p>
        ) : null}
        <details className="photoreal-model-files">
          <summary>모델 파일 이름 (선택)</summary>
          <p className="runtime-hint">비워 두면 프리셋 기본 파일을 씁니다. ComfyUI가 하위 폴더에서 찾은 파일은 그 경로 그대로 입력하세요.</p>
          {probedPresetIds
            .filter((presetId) => presetId !== "custom")
            .map((presetId) => (
              <div className="photoreal-model-file-group" key={presetId}>
                <span>{PHOTOREAL_WORKFLOW_PRESETS[presetId].label}</span>
                {PHOTOREAL_MODEL_FILE_SLOTS.map((slot) => (
                  <label key={slot.field}>
                    {slot.label}
                    <input
                      value={photoreal.modelFiles[presetId]?.[slot.field] ?? ""}
                      placeholder={PHOTOREAL_WORKFLOW_PRESETS[presetId].defaults[slot.field]}
                      spellCheck={false}
                      onChange={(event) => setModelFile(presetId, slot.field, event.target.value)}
                      onBlur={() => onNotify("실사 모델 파일 이름이 적용되었습니다.")}
                    />
                  </label>
                ))}
              </div>
            ))}
        </details>
      </div>

      <div className="runtime-parameter-card">
        <div className="runtime-card-subhead">
          <strong>출력과 샘플링</strong>
          <span>
            {effectiveSize.width}x{effectiveSize.height}
          </span>
        </div>
        <div className="two-fields">
          <label>
            너비
            <CommittedNumberInput
              value={photoreal.width}
              emptyValue={0}
              min={0}
              max={2048}
              step={16}
              integer
              placeholder={`자동 (${effectiveSize.width})`}
              onCommit={(width) => onChange({ width: width ?? 0 }, "실사 이미지 너비가 적용되었습니다.")}
            />
          </label>
          <label>
            높이
            <CommittedNumberInput
              value={photoreal.height}
              emptyValue={0}
              min={0}
              max={2048}
              step={16}
              integer
              placeholder={`자동 (${effectiveSize.height})`}
              onCommit={(height) => onChange({ height: height ?? 0 }, "실사 이미지 높이가 적용되었습니다.")}
            />
          </label>
        </div>
        <p className="runtime-hint">
          비우거나 0이면 자동입니다: NovelAI 해상도({state.imageProfile.width}x{state.imageProfile.height})의 비율을 실사 모델에 맞는 크기로
          바꿉니다. 직접 넣은 값은 16의 배수로 맞춰집니다.
        </p>
        <div className="two-fields">
          <label>
            스텝
            <CommittedNumberInput
              value={photoreal.steps}
              emptyValue={0}
              min={0}
              max={100}
              integer
              placeholder={`기본 ${generationPreset.defaults.steps}`}
              onCommit={(steps) => onChange({ steps: steps ?? 0 }, "실사 스텝이 적용되었습니다.")}
            />
          </label>
          <label>
            CFG
            <CommittedNumberInput
              value={photoreal.cfg}
              emptyValue={0}
              min={PHOTOREAL_MIN_CFG}
              max={20}
              step={0.1}
              placeholder={generationPreset.fixedCfg !== undefined ? `Turbo 고정 ${generationPreset.fixedCfg}` : `기본 ${generationPreset.defaults.cfg}`}
              ariaDescribedBy={cfgStatusId}
              onBelowMin={(typed) =>
                setCfgRefusalNote({
                  simulationId: state.simulation.id,
                  message: `CFG ${typed}은(는) 쓸 수 없어 적용하지 않았습니다. 실사 CFG는 ${PHOTOREAL_MIN_CFG} 이상이어야 합니다 — 1보다 작으면 샘플러가 네거티브 쪽으로 그려 네거티브 문구가 그림이 됩니다. 프리셋 기본값을 쓰려면 칸을 비우세요.`
                })
              }
              onCommit={(cfg) => {
                setCfgRefusalNote(undefined);
                onChange({ cfg: cfg ?? 0 }, "실사 CFG가 적용되었습니다.");
              }}
            />
          </label>
        </div>
        {/* Always mounted, so screen readers have a polite region to watch before its text first changes; the
            photoreal-text-screen class lifts it out of the card's grid while empty, so it adds no gap. */}
        <div className="photoreal-text-screen" id={cfgStatusId} role="status" aria-live="polite">
          {cfgRefusal ? <p className="settings-note warning">{cfgRefusal}</p> : null}
        </div>
        {generationPreset.fixedCfg !== undefined || referencePreset.fixedCfg !== undefined ? (
          <p className="runtime-hint">
            Turbo 프리셋(Qwen-Image 2.1 Turbo, Z-Image Turbo)은 증류 모델이라 CFG 1로 고정되어 CFG 값을 쓰지 않습니다. CFG 1에서는 네거티브도
            쓰이지 않습니다. Qwen-Image 2.1 Turbo는 8스텝 일정도 고정이라 스텝 값을 쓰지 않습니다. CFG 값은 다른 프리셋에만 적용됩니다.
          </p>
        ) : null}
        <div className="two-fields">
          <label className="checkline">
            <input
              checked={photoreal.seedFixed}
              type="checkbox"
              onChange={(event) =>
                onChange({ seedFixed: event.target.checked }, event.target.checked ? "실사 시드 고정이 활성화되었습니다." : "실사 시드 고정이 비활성화되었습니다.")
              }
            />
            시드 고정
          </label>
          <label>
            시드
            <CommittedNumberInput
              value={photoreal.seed}
              min={0}
              max={PHOTOREAL_MAX_SEED}
              integer
              placeholder="무작위"
              disabled={!photoreal.seedFixed}
              onCommit={(seed) => onChange({ seed }, seed === undefined ? "실사 시드를 비웠습니다." : "실사 시드가 적용되었습니다.")}
            />
          </label>
        </div>
        <label>
          대기 시간 (초)
          <CommittedNumberInput
            value={photoreal.timeoutSeconds}
            emptyValue={DEFAULT_PHOTOREAL_SETTINGS.timeoutSeconds}
            min={30}
            max={900}
            integer
            placeholder={`기본 ${DEFAULT_PHOTOREAL_SETTINGS.timeoutSeconds}`}
            onCommit={(timeoutSeconds) =>
              onChange({ timeoutSeconds: timeoutSeconds ?? DEFAULT_PHOTOREAL_SETTINGS.timeoutSeconds }, "실사 대기 시간이 적용되었습니다.")
            }
          />
        </label>
        <p className="runtime-hint">첫 컷은 모델을 불러오느라 오래 걸립니다(17GB 로드 기준 수십 초). 한 컷이 이 시간을 넘기면 ComfyUI 작업을 중단합니다.</p>
      </div>

      <div className="runtime-parameter-card">
        <div className="runtime-card-subhead">
          <strong>스타일</strong>
          <span>{activeStyle?.label ?? "사용자 지정"}</span>
        </div>
        <label className="checkline">
          <input
            type="checkbox"
            checked={
              photoreal.enabled &&
              photoreal.stylePreset === "daily_look" &&
              photoreal.workflowPreset === "z_image_turbo"
            }
            onChange={(event) => {
              if (event.target.checked) {
                onChange(
                  {
                    enabled: true,
                    stylePreset: "daily_look",
                    workflowPreset: "z_image_turbo",
                    referenceWorkflowPreset: "qwen_image_2_1_turbo",
                    width: 960,
                    height: 1280
                  },
                  "핀터레스트 데일리룩 사진 톤(Z-Image Turbo · 960×1280)이 활성화되었습니다."
                );
              } else {
                onChange({ stylePreset: "cinematic" }, "필름 시네마틱 톤으로 전환되었습니다.");
              }
            }}
          />
          핀터레스트 데일리룩 사진 톤 켜기 (DNC 감성 · Z-Image Turbo 960×1280)
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={
              photoreal.enabled &&
              photoreal.stylePreset === "raw_snap" &&
              photoreal.workflowPreset === "z_image_turbo"
            }
            onChange={(event) => {
              if (event.target.checked) {
                onChange(
                  {
                    enabled: true,
                    stylePreset: "raw_snap",
                    workflowPreset: "z_image_turbo",
                    referenceWorkflowPreset: "qwen_image_2_1_turbo",
                    width: 768,
                    height: 1024
                  },
                  "로우파이 리얼 폰카 모드(Z-Image Turbo · 768×1024 저화질 생사진 톤)가 활성화되었습니다."
                );
              } else {
                onChange({ stylePreset: "cinematic", width: 832, height: 1248 }, "필름 시네마틱 톤으로 전환되었습니다.");
              }
            }}
          />
          로우파이 리얼 폰카 모드 켜기 (저화질·생사진 · Z-Image Turbo 768×1024)
        </label>
        <label>
          스타일 프리셋
          <select
            value={photoreal.stylePreset}
            onChange={(event) => {
              const stylePreset = event.target.value as PhotorealStylePresetId;
              onChange(
                { stylePreset },
                `${stylePreset === "custom" ? "사용자 지정" : PHOTOREAL_STYLE_PRESETS[stylePreset].label} 스타일이 적용되었습니다.`
              );
            }}
          >
            {photorealStylePresetOrder.map((stylePreset) => (
              <option key={stylePreset} value={stylePreset}>
                {PHOTOREAL_STYLE_PRESETS[stylePreset].label}
              </option>
            ))}
            <option value="custom">사용자 지정</option>
          </select>
        </label>
        <p className="runtime-hint">
          {activeStyle?.description ??
            "아래 스타일 문장이 전부입니다. 첫 문장이 프롬프트 맨 앞(매체 설명)에 오고, 나머지는 끝에 붙습니다."}
        </p>
        <label>
          {photoreal.stylePreset === "custom" ? "스타일 (영어)" : "스타일 추가 (영어, 선택)"}
          <textarea
            value={photoreal.stylePrompt}
            placeholder={photoreal.stylePreset === "custom" ? "A candid smartphone photograph. Soft window light, natural skin texture" : "e.g. warm evening tones, 50mm lens"}
            aria-describedby={styleScreenId}
            onChange={(event) => onChange({ stylePrompt: event.target.value })}
            onBlur={() => onNotify("실사 스타일 문장이 적용되었습니다.")}
          />
        </label>
        {/* Keyed by simulation: the note reads the text once typing settles, and the panel can switch simulations. */}
        <PhotorealTextScreenNote id={styleScreenId} key={state.simulation.id} scope="style" state={state} text={photoreal.stylePrompt} />
        <label>
          여성 인물 공통 외모 (실사 전용, 영어)
          <textarea
            value={photoreal.appearancePrompt}
            placeholder={PHOTOREAL_SHARED_APPEARANCE_PLACEHOLDER}
            aria-describedby={appearanceScreenId}
            onChange={(event) => onChange({ appearancePrompt: event.target.value })}
            onBlur={() => onNotify("실사 공통 외모가 적용되었습니다.")}
          />
        </label>
        <PhotorealTextScreenNote
          id={appearanceScreenId}
          key={`${state.simulation.id}-appearance`}
          scope="appearance"
          state={state}
          text={photoreal.appearancePrompt}
        />
        <p className="runtime-hint">
          실사 컷의 모든 여성 인물에게 붙습니다. 캐릭터마다 다른 외형은 캐릭터의 실사 외형 칸에 씁니다. NovelAI 식 강조(2.5::…::)는 기호를 빼고 단어만 씁니다.
        </p>
        <label>
          네거티브 추가 (영어, 선택)
          <textarea
            value={photoreal.negativePrompt}
            placeholder="e.g. tattoo, heavy makeup"
            onChange={(event) => onChange({ negativePrompt: event.target.value })}
            onBlur={() => onNotify("실사 네거티브가 적용되었습니다.")}
          />
        </label>
        <p className="runtime-hint">실사 모델은 영어 문장을 가장 잘 읽습니다. 한글은 프롬프트에서 빠집니다. CFG가 1이면(Turbo 프리셋은 항상 1) 네거티브는 쓰이지 않습니다.</p>
      </div>

      <div className="runtime-parameter-card">
        <div className="runtime-card-subhead">
          <strong>얼굴 고정 · GPU</strong>
          <span>{photoreal.vramHandoff === "auto" ? "교대 자동" : "교대 끔"}</span>
        </div>
        <label className="checkline">
          <input
            checked={photoreal.faceReferenceEnabled}
            type="checkbox"
            onChange={(event) =>
              onChange(
                { faceReferenceEnabled: event.target.checked },
                event.target.checked ? "실사 얼굴 고정이 활성화되었습니다." : "실사 얼굴 고정이 비활성화되었습니다."
              )
            }
          />
          캐릭터의 실사 프로필 이미지로 얼굴 고정
        </label>
        <p className="runtime-hint">
          프로필 이미지는 제작 화면에서 DynamicChat이 생성한 후보 중에서만 고를 수 있습니다(업로드 없음). 얼굴이 보이지 않는 컷(뒷모습 등)에는 붙지 않습니다.
        </p>
        <label>
          GPU 교대
          <select
            value={photoreal.vramHandoff}
            onChange={(event) =>
              onChange(
                { vramHandoff: event.target.value === "off" ? "off" : "auto" },
                event.target.value === "off" ? "GPU 교대를 껐습니다." : "GPU 교대를 자동으로 바꿨습니다."
              )
            }
          >
            <option value="auto">자동 (권장)</option>
            <option value="off">끄기</option>
          </select>
        </label>
        {/* The one-line WHY: without it "turn this off" looks harmless, and the failure it prevents (a spilled model
            at a fraction of a token per second) looks like the app hanging. */}
        <p className="runtime-hint">
          16GB 카드에는 Ollama 모델(qwen3:14b 11.7GB)과 실사 모델(최대 약 15GB)이 함께 올라가지 않고, 넘친 쪽은 초당 0.1토큰 수준으로
          느려집니다. 자동이면 턴과 실사 이미지가 GPU를 번갈아 쓰도록 서로의 모델을 내립니다.{" "}
          {gpuHandoffActive ? "지금 이 설정에 적용되고 있습니다." : "LLM이 이 PC의 Ollama이고 ComfyUI도 이 PC에 있을 때만 적용됩니다."}
        </p>
      </div>

      <div className="runtime-parameter-card">
        <div className="runtime-card-subhead">
          <strong>사용자 워크플로</strong>
          <span>{usesCustomWorkflow ? "사용 중" : "선택 안 함"}</span>
        </div>
        <p className="runtime-hint">
          프리셋에서 '사용자 워크플로'를 고르면 쓰입니다. ComfyUI 메뉴의 Export (API)로 내보낸 JSON에 아래 자리표시자를 넣으세요(SillyTavern과
          같은 방식).
        </p>
        <div className="photoreal-workflow-field">
          <textarea
            className="photoreal-workflow-input"
            aria-label="사용자 워크플로 JSON"
            aria-describedby={customWorkflowStatusId}
            aria-invalid={customWorkflowErrors.length > 0 ? true : undefined}
            spellCheck={false}
            value={photoreal.customWorkflow}
            placeholder={'{"3": {"class_type": "KSampler", "inputs": {"seed": "%seed%", ...}}}'}
            onChange={(event) => onChange({ customWorkflow: event.target.value })}
            onBlur={() => {
              settleCustomWorkflow();
              onNotify("사용자 워크플로가 적용되었습니다.");
            }}
          />
          {/* Always mounted, so screen readers have a polite region to watch before its text first changes; the
              textarea points at it, so the verdict can also be heard on demand from the field. */}
          <div className="photoreal-workflow-status" id={customWorkflowStatusId} role="status" aria-live="polite">
            {customWorkflowErrors.length > 0 ? (
              <div className="settings-note warning photoreal-workflow-errors">
                <strong>워크플로를 쓸 수 없습니다 — 이 워크플로로 갈 실사 컷은 NovelAI로 계획됩니다</strong>
                <ul>
                  {customWorkflowErrors.map((error) => (
                    <li key={error}>{error}</li>
                  ))}
                </ul>
              </div>
            ) : settledCustomWorkflow.trim() ? (
              <p className="runtime-hint">워크플로 형식이 올바릅니다.</p>
            ) : usesCustomWorkflow ? (
              <p className="settings-note warning">사용자 워크플로를 골랐지만 JSON이 비어 있어 이 워크플로로 갈 실사 컷은 NovelAI로 계획됩니다.</p>
            ) : null}
          </div>
        </div>
        <details className="photoreal-placeholder-help">
          <summary>자리표시자 {COMFY_CUSTOM_PLACEHOLDERS.length}개</summary>
          <ul>
            {COMFY_CUSTOM_PLACEHOLDERS.map((placeholder) => (
              <li key={placeholder.name}>
                <code>%{placeholder.name}%</code>
                <span>{placeholder.description}</span>
              </li>
            ))}
          </ul>
        </details>
      </div>
    </>
  );
}

function SceneCard({ asset, featured = false }: { asset: ImageAsset; featured?: boolean }) {
  const style = {
    "--tone-a": asset.palette[0],
    "--tone-b": asset.palette[1],
    "--tone-c": asset.palette[2]
  } as CSSProperties;
  return (
    <article className={`scene-card ${featured ? "featured" : ""}`} style={style}>
      <div className="scene-visual">
        <AssetImage src={createImageAssetSrc(asset)} alt={asset.title} />
      </div>
      <div className="scene-copy">
        <strong>{asset.title}</strong>
        <span>{imageAssetSourceLabels[asset.source]} · {asset.safetyLevel}</span>
      </div>
    </article>
  );
}

function createImageAssetSrc(asset?: ImageAsset): string | undefined {
  if (!asset) {
    return undefined;
  }

  if (asset.dataUrl) {
    return asset.dataUrl;
  }

  return asset.objectKey ? createObjectStoreImageSrc(asset.objectKey) : undefined;
}

/** The API server's object route for a stored image (the whole key is one encoded segment; the route decodes it). */
function createObjectStoreImageSrc(objectKey: string): string {
  return `${getConfiguredDynamicChatApiBaseUrl()}/objects/${encodeURIComponent(objectKey)}`;
}

function AssetImage({
  src,
  alt,
  onNaturalSize
}: {
  src?: string;
  alt: string;
  onNaturalSize?: (width: number, height: number) => void;
}) {
  const [failedSrc, setFailedSrc] = useState<string>();
  const failed = Boolean(src && failedSrc === src);

  useEffect(() => {
    if (!failedSrc) {
      return undefined;
    }

    const retryTimeout = window.setTimeout(() => {
      setFailedSrc((current) => (current === failedSrc ? undefined : current));
    }, 5000);

    return () => window.clearTimeout(retryTimeout);
  }, [failedSrc]);

  if (!src || failed) {
    return <FallbackSceneArt />;
  }

  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      onError={() => setFailedSrc(src)}
      onLoad={
        onNaturalSize
          ? (event) => {
              const image = event.currentTarget;
              if (image.naturalWidth > 0 && image.naturalHeight > 0) {
                onNaturalSize(image.naturalWidth, image.naturalHeight);
              }
            }
          : undefined
      }
    />
  );
}

function FallbackSceneArt() {
  return (
    <div className="scene-poster-art" aria-hidden="true">
      <span className="scene-poster-sky" />
      <span className="scene-poster-window" />
      <span className="scene-poster-character" />
      <span className="scene-poster-floor" />
      <span className="scene-poster-light" />
    </div>
  );
}

function StatusPill({ icon, label, tone }: { icon: React.ReactNode; label: string; tone: "good" | "neutral" }) {
  return (
    <span className={`status-pill ${tone}`}>
      {icon}
      {label}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Capability({ label, value }: { label: string; value: string }) {
  return (
    <div className="capability-row">
      <ShieldCheck size={16} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

// The containers are already role="tablist", but their children were plain buttons — no role="tab", no
// aria-selected, and every tab in the natural tab order, so reaching the seventh ops panel took seven Tab
// presses. Roving tabIndex plus the arrow-key handler on the list makes one Tab reach the group and the
// arrows move within it, which is what a tablist promises.
// `panelKey` is what ties a tab to the region it drives: `id` lets the panel name itself with
// aria-labelledby, and aria-controls points the other way. Only the SELECTED tab gets aria-controls,
// because these rails render one panel at a time — the unselected panels are not in the DOM, so
// pointing at their ids would hand a screen reader references that resolve to nothing. Optional
// because one caller (SimulationRunPage) has no single element to mark as the panel.
function TabButton({
  active,
  icon,
  label,
  onClick,
  panelKey
}: {
  active: boolean;
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  panelKey?: string;
}) {
  return (
    <button
      className={`tab-button ${active ? "active" : ""}`}
      type="button"
      role="tab"
      id={panelKey ? `tab-${panelKey}` : undefined}
      aria-controls={active && panelKey ? `tabpanel-${panelKey}` : undefined}
      aria-selected={active}
      tabIndex={active ? 0 : -1}
      onClick={onClick}
    >
      {icon}
      {label}
    </button>
  );
}

/**
 * ArrowLeft/ArrowRight/Home/End roving focus for a `role="tablist"` container. Attach to the container;
 * it moves focus between the enabled `role="tab"` children and activates the one it lands on.
 */
function handleTablistKeyDown(event: React.KeyboardEvent<HTMLElement>): void {
  const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
  if (!keys.includes(event.key)) {
    return;
  }
  const tabs = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]:not([disabled])')];
  if (tabs.length === 0) {
    return;
  }
  const currentIndex = tabs.findIndex((tab) => tab === document.activeElement);
  const nextIndex =
    event.key === "Home"
      ? 0
      : event.key === "End"
        ? tabs.length - 1
        : currentIndex < 0
          ? 0
          : (currentIndex + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  event.preventDefault();
  tabs[nextIndex]?.focus();
  tabs[nextIndex]?.click();
}

function SectionTitle({
  icon,
  title,
  className,
  headingRef
}: {
  icon: React.ReactNode;
  title: string;
  className?: string;
  /** Makes the title a programmatic focus target (tabIndex -1), for buttons elsewhere that jump to its section. */
  headingRef?: React.Ref<HTMLDivElement>;
}) {
  return (
    <div className={className ? `section-title ${className}` : "section-title"} ref={headingRef} tabIndex={headingRef ? -1 : undefined}>
      {icon}
      <strong>{title}</strong>
    </div>
  );
}

export default App;
