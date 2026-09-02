import type {
  AppState,
  Character,
  CharacterVisualProfile,
  ContentRating,
  EvaluationScenario,
  ImageGenerationProfile,
  ImageScenePresetExampleFile,
  ImageSceneTagPreset,
  ImageSceneTagPresetNode,
  LlmApiSettings,
  NeuralMapSettings,
  NovelAiApiSettings,
  PromptModule,
  RelationshipMapSettings,
  SimulationCharacterDraft,
  SimulationPromptMode,
  Simulation,
  UserPersona
} from "../types";
import { createId } from "../lib/id";
import { createSecuritySettings, normalizeSecuritySettings } from "../services/security";
import { toShareableLlmSettings, toShareableNovelAiSettings } from "../services/runtimeApiSettings";
import { createDefaultProgressRunId, normalizeProgressRuns } from "../services/progressRuns";
import { resolveNovelAiModelName } from "../services/novelAiModels";
import { womanLifeSeedState } from "./womanLifeSeed";

const now = "2026-05-03T00:00:00.000Z";
const defaultOutfitPrompt = "";
const defaultOutfitPrompts: Record<string, string> = {};
const defaultExpressionPrompts: Record<string, string> = {};
const defaultImageScenePresets: ImageSceneTagPreset[] = [
  {
    id: "scene_preset_archive",
    simulationId: "sim_nocturne",
    keyword: "archive library",
    tags: ["archive library", "bookshelf", "old books", "wooden table", "paper stack", "warm lamplight", "dust particles"],
    note: "기록 보관소, 사서, 단서 조사 장면용. 캐릭터 외형 태그는 포함하지 않는다.",
    exampleFiles: [
      {
        id: "scene_preset_archive_file_scene",
        label: "scene",
        prompts: [
          "archive library, tall bookshelves, warm lamplight, dust particles, wooden ladder, from above, wide shot",
          "narrow library aisle, old books, paper stack, dim light, depth of field, over the shoulder",
          "reading desk, open book, ink bottle, candle, close-up, soft shadows"
        ]
      }
    ],
    enabled: true,
    priority: 84,
    updatedAt: now,
    children: [
      {
        id: "scene_preset_archive_research_table",
        keyword: "research table",
        tags: ["wooden table", "paper stack", "open book", "ink bottle", "warm lamplight"],
        note: "책상 위 단서 조사 클로즈업/중경 장면.",
        enabled: true,
        priority: 80,
        updatedAt: now,
        children: []
      },
      {
        id: "scene_preset_archive_bookshelf",
        keyword: "dusty bookshelf",
        tags: ["bookshelf", "old books", "ladder", "dust particles", "narrow aisle"],
        note: "서가 사이 이동/수색 장면.",
        enabled: true,
        priority: 76,
        updatedAt: now,
        children: []
      }
    ]
  },
  {
    id: "scene_preset_rain_city",
    simulationId: "sim_nocturne",
    keyword: "rain city",
    tags: ["rain", "wet street", "city lights", "night", "reflection", "mist", "street lamp"],
    note: "비 오는 도시 외부 장면용. 인물 태그 없이 배경/조명/날씨만 둔다.",
    enabled: true,
    priority: 72,
    updatedAt: now,
    children: [
      {
        id: "scene_preset_rain_city_alley",
        keyword: "alley",
        tags: ["narrow alley", "wet pavement", "neon sign", "mist", "backlight"],
        note: "비 오는 골목/추적 장면.",
        enabled: true,
        priority: 70,
        updatedAt: now,
        children: []
      }
    ]
  }
];
const defaultUserPersona: UserPersona = {
  enabled: false,
  source: "custom",
  characterId: undefined,
  name: "플레이어",
  role: "아직 정해지지 않은 방문자",
  background: "첫 장면에서 자신의 배경을 천천히 드러낸다.",
  goals: "현재 장면의 단서와 관계를 따라가며 다음 선택을 정한다.",
  style: "사용자가 직접 입력한 행동과 대사를 최우선으로 반영한다.",
  boundaries: "",
  updatedAt: now
};
const defaultRelationshipMapSettings: RelationshipMapSettings = {
  enabled: true,
  statusPrompt:
    "관계도/상태창은 별도 장문 출력이 아니라 memory_events로 갱신한다. 인물의 현재 위치, 감정, 체력/컨디션, 착용/소지품, 목표, 관계 변화가 생기면 memory_kind='state' 또는 'relationship'으로 짧고 안정적인 델타를 남긴다. 의상 변화나 장면상 의상이 새로 확정되면 state_type='Wearing'에 NovelAI-style English outfit tags를 저장하고, 표정/컨디션/행동처럼 이미지와 반응 일관성에 필요한 캐릭터별 상태 태그는 state_type='StatusTags'에 저장한다. 기존 의상이 찢어짐/젖음/오염/헐거워짐처럼 변형될 때는 police uniform, navy short dress, mini skirt 같은 베이스 의상 태그를 유지하고 torn uniform 같은 상태 태그를 덧붙인다. 자세, 현재 행동, 상호작용, 전체 상황/단계, 소지품, 카메라/조명/장면 구도가 이미지 일관성에 중요하면 state_type='PoseTags', 'ActionTags', 'InteractionTags', 'InteractionPhaseTags', 'HeldItemTags', 'SceneTags', 'ScenePhaseTags', 'CompositionTags', 'CameraTags', 'LightingTags'에 comma-separated English NAI tags로 저장한다. 너무 세세한 부위별 태그를 매번 쌓기보다 현재 상황을 복원할 수 있는 3-8개의 compact phase/state tags를 우선한다. 이미 Structured simulation memory에 있는 동일 상태는 반복하지 말고, 바뀐 값만 저장한다. actor_id와 target_id는 알 수 있을 때 반드시 사용한다.",
  parameters: [
    {
      id: "rel_param_outfit_tags",
      title: "의상 태그",
      rule: "현재 장면에서 확정된 캐릭터별 의상을 NovelAI-style English tags로 짧게 저장합니다. 의상이 손상/오염/노출 상태로 변형되면 기존 베이스 의상 태그를 지우지 말고 상태 태그를 덧붙입니다. 예: police uniform, navy short dress, mini skirt, torn uniform.",
      enabled: true,
      priority: 98
    },
    {
      id: "rel_param_status_tags",
      title: "상태 태그",
      rule: "표정, 감정, 컨디션, 소지품, 자세처럼 다음 이미지와 반응 일관성에 필요한 캐릭터별 상태 태그를 짧게 저장합니다.",
      enabled: true,
      priority: 94
    },
    {
      id: "rel_param_thought",
      title: "생각",
      rule: "해당 인물의 내면과 생각을 날것 그대로 작성합니다.",
      enabled: true,
      priority: 90
    }
  ],
  updatedAt: now
};

export const seedState: AppState = {
  simulation: {
    id: "sim_nocturne",
    ownerId: "local_user",
    title: "Nocturne Archive",
    description: "비가 멈추지 않는 기록 도시에서 기억을 되찾는 장기 시뮬레이션.",
    promptMode: "simulation",
    contentRating: "general",
    activeSessionId: "session_001",
    realtimeImageEnabled: true,
    defaultChatModelProfile: "balanced-agent",
    createdAt: now,
    updatedAt: now
  },
  security: createSecuritySettings({
    id: "sim_nocturne",
    ownerId: "local_user"
  }),
  activeProgressRunId: "run_session_001",
  progressRuns: [],
  modules: [
    {
      id: "module_main",
      simulationId: "sim_nocturne",
      kind: "main_prompt",
      title: "메인 시뮬레이션 규칙",
      body:
        "플레이어의 선택을 중심으로 장면을 진행한다. 과거 기억, 관계 변화, 약속, 장소 변화는 장기 기억 후보로 기록한다. 응답은 몰입감 있게 쓰되 사용자가 행동할 여지를 남긴다.",
      enabled: true,
      priority: 100,
      activationTags: ["core", "always"],
      tokenPolicy: "always",
      version: 1,
      updatedAt: now
    },
    {
      id: "module_aria",
      simulationId: "sim_nocturne",
      parentId: "module_main",
      kind: "character_prompt",
      title: "캐릭터: Aria",
      body:
        "Aria는 기록 도시의 사서다. 차분하지만 기억 조작에 대한 분노를 숨기고 있다. 플레이어를 신뢰하고 싶어 하지만 증거가 없으면 선을 긋는다.",
      enabled: true,
      priority: 82,
      activationTags: ["aria", "character", "library"],
      characterId: "char_aria",
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "module_world",
      simulationId: "sim_nocturne",
      parentId: "module_main",
      kind: "world_lore",
      title: "세계관: 기록 도시",
      body:
        "도시는 짙은 안개가 깔릴 때마다 시민들의 기억 일부를 중앙 기록 보관소로 흘려보낸다. 잃어버린 기억은 은색 열쇠, 시계탑, 오래된 종이 냄새와 자주 연결된다.",
      enabled: true,
      priority: 74,
      activationTags: ["city", "memory", "silver-key", "archive"],
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "module_visual",
      simulationId: "sim_nocturne",
      parentId: "module_main",
      kind: "image_prompt_profile",
      title: "이미지 스타일: Nocturne Archive",
      body:
        "cinematic anime illustration, atmospheric lighting, library, bookshelf, environmental portrait, close-up, quiet intensity",
      enabled: true,
      priority: 82,
      activationTags: ["image", "style", "nai", "archive"],
      tokenPolicy: "always",
      version: 1,
      updatedAt: now
    },
  ],
  imageScenePresets: defaultImageScenePresets,
  characters: [
    {
      id: "char_aria",
      simulationId: "sim_nocturne",
      name: "Aria",
      role: "Archive keeper",
      summary: "기록 도시의 젊은 사서. 은색 열쇠의 단서를 알고 있다.",
      relationship: "플레이어에게 조심스럽게 협력 중",
      currentMood: "경계와 기대가 섞인 상태"
    }
  ],
  visualProfiles: [
    {
      id: "visual_aria",
      simulationId: "sim_nocturne",
      characterId: "char_aria",
      displayName: "Aria",
      positivePrompt:
        "silver hair, slate gray eyes, archivist coat, delicate face, quiet intensity",
      negativePrompt: "low quality, bad anatomy, extra fingers, blurry, watermark",
      defaultOutfitPrompt: "archivist coat",
      outfitPrompts: defaultOutfitPrompts,
      expressionPrompts: {
        neutral: "reserved expression",
        tense: "tense expression, narrowed eyes",
        relieved: "soft relieved smile"
      },
      referenceImageAssetIds: ["asset_aria_reference"],
      defaultSafetyLevel: "safe"
    }
  ],
  imageProfile: {
    id: "img_profile_default",
    simulationId: "sim_nocturne",
    enabled: true,
    provider: "novelai",
    model: "nai-diffusion-4-5-curated",
    width: 1024,
    height: 1024,
    steps: 28,
    promptGuidance: 5,
    countMin: 1,
    countMax: 2,
    qualityPrompt: "masterpiece, best quality, detailed background",
    stylePrompt: "cinematic anime illustration, atmospheric lighting",
    artistPrompt: "",
    negativePrompt: "lowres, blurry, worst quality, text, watermark, bad anatomy",
    safetyLevel: "safe",
    userRules: "캐릭터의 외형 일관성을 우선한다.",
    triggerMode: "realtime_auto",
    generationCadence: "balanced",
    cooldownTurns: 2
  },
  userPersona: {
    ...defaultUserPersona
  },
  messages: [
    {
      id: "msg_welcome",
      simulationId: "sim_nocturne",
      sessionId: "session_001",
      role: "assistant",
      content:
        "비 내리는 기록 보관소의 문이 천천히 열린다. Aria는 젖은 장갑을 벗으며 당신을 바라본다. \"은색 열쇠 이야기를 들으러 온 건가요?\"",
      createdAt: now,
      referencedNodeIds: ["module_main", "module_aria", "module_world"],
      imageAssetIds: ["asset_library_rain"]
    }
  ],
  memoryEvents: [
    {
      id: "memory_001",
      simulationId: "sim_nocturne",
      sessionId: "session_001",
      actorId: "char_aria",
      actorName: "Aria",
      content: "Aria는 은색 열쇠가 시계탑 아래 분수와 관련 있다고 암시했다.",
      importance: 0.86,
      tags: ["silver-key", "promise", "location"],
      sourceTurnId: "msg_welcome",
      createdAt: now,
      neuralMapNodeId: "simulation:sim_nocturne:event:memory_001"
    }
  ],
  contextPacks: [
    {
      id: "ctx_seed",
      simulationId: "sim_nocturne",
      sessionId: "session_001",
      objective: "현재 장면과 은색 열쇠 단서 유지",
      tokenBudget: 3200,
      evidence: [
        {
          nodeId: "memory_001",
          snippet: "Aria는 은색 열쇠가 시계탑 아래 분수와 관련 있다고 암시했다.",
          score: 0.9,
          reason: "초기 장면의 핵심 단서"
        }
      ],
      decisions: ["은색 열쇠 단서는 다음 장면에서도 유지한다."],
      blockers: [],
      createdAt: now,
      source: "mock"
    }
  ],
  handoffs: [],
  continuityChecks: [],
  promptModuleUsages: [],
  sidecarTraces: [],
  turnTraces: [],
  evaluationScenarios: [
    {
      id: "eval_silver_key_recall",
      simulationId: "sim_nocturne",
      kind: "memory_recall",
      label: "은색 열쇠 단서 회수",
      query: "은색 열쇠와 시계탑 아래 분수의 관계를 회수한다.",
      expectedSignals: ["은색 열쇠", "시계탑", "분수"],
      source: "seed",
      createdAt: now
    },
    {
      id: "eval_reset_continuity",
      simulationId: "sim_nocturne",
      kind: "reset_continuity",
      label: "세션 초기화 연속성",
      query: "세션 초기화 뒤 캐릭터 관계와 최근 핵심 기억이 유지되는지 확인한다.",
      expectedSignals: ["handoff", "continuity", "Aria"],
      source: "system",
      createdAt: now
    },
    {
      id: "eval_image_quality",
      simulationId: "sim_nocturne",
      kind: "image_quality",
      label: "대표 이미지 품질 피드백",
      query: "생성 이미지가 캐릭터/장면 프롬프트와 맞는지 사용자 피드백으로 확인한다.",
      expectedSignals: ["liked", "neutral", "rejected"],
      source: "system",
      createdAt: now
    }
  ],
  auditLog: [],
  redactionQueue: [],
  imageAssets: [
    {
      id: "asset_library_rain",
      simulationId: "sim_nocturne",
      title: "Archive Background",
      source: "stored",
      prompt: "archive library, brass shelves, dim lamps",
      negativePrompt: "",
      safetyLevel: "safe",
      characterIds: [],
      tags: ["library", "background"],
      createdAt: now,
      palette: ["#31475e", "#b7c7c9", "#d2a84b"]
    },
    {
      id: "asset_aria_reference",
      simulationId: "sim_nocturne",
      title: "Aria Reference",
      source: "stored",
      prompt: "silver haired archivist, slate eyes, dark coat",
      negativePrompt: "",
      safetyLevel: "safe",
      characterIds: ["char_aria"],
      tags: ["aria", "reference"],
      createdAt: now,
      palette: ["#d9e2e6", "#3d4855", "#7b8da0"]
    }
  ],
  imageJobs: [],
  neuralMap: {
    baseUrl: "http://127.0.0.1:4317",
    enabled: false,
    tokenBudget: 3200
  },
  relationshipMap: {
    ...defaultRelationshipMapSettings
  },
  relationshipStatusOverrides: [],
  // A fresh install defaults to the keyless local backend, not to the mock provider. Shipping "mock" meant a
  // brand-new user's very first turn returned locally assembled placeholder prose that looked exactly like a
  // real turn — the product appeared to write badly rather than to be unconfigured. With Ollama as the
  // default, a machine running `ollama serve` works with no key at all (pick a model with 로컬 서버에서
  // 불러오기), and a machine without it gets a real connection error naming the fix. Mock stays available in
  // the provider list for anyone who wants to watch the turn machinery run offline.
  llm: {
    enabled: true,
    provider: "ollama",
    baseUrl: "http://127.0.0.1:11434/v1",
    apiKey: "",
    model: "",
    temperature: 0.82,
    maxTokens: 900,
    systemPrompt:
      "You are the narrative engine for DynamicChat. Continue the simulation using retrieved prompt modules, context evidence, and user action. Respond in Korean unless the user asks otherwise.",
    registrationStatus: "idle"
  },
  imageTagLlm: {
    // enabled=false → the image-cue call reuses the main `llm` config. Turn it on to use a separate (cheaper) model.
    enabled: false,
    provider: "mock",
    baseUrl: "https://api.openai.com/v1",
    apiKey: "",
    model: "gpt-4.1-mini",
    temperature: 0.3,
    maxTokens: 900,
    systemPrompt: "",
    registrationStatus: "idle"
  },
  novelAi: {
    enabled: false,
    requestMode: "proxy",
    endpoint: "https://image.novelai.net/ai/generate-image",
    apiKey: "",
    proxyUrl: "http://127.0.0.1:8788/novelai/generate-image",
    accountLabel: "Main NAI account",
    roundRobinEnabled: false,
    modelPreset: "NAID4.5F",
    ucPreset: 0,
    sampler: "k_euler_ancestral",
    noiseSchedule: "karras",
    cfgRescale: 0,
    varPlus: true,
    seedFixed: false,
    generationDelaySeconds: 0,
    randomDelayEnabled: false,
    repeatCount: 1,
    automationTermination: "unlimited",
    timerMinutes: 30,
    countLimit: 30,
    registrationStatus: "idle",
    vibeTransferEnabled: false,
    vibeTransferReferences: []
  },
  selectedModuleId: "module_main",
  selectedContextPackId: "ctx_seed"
};

const sunnyLineSimulationId = "sim_sunnyline_idol_house";
const sunnyLineSessionId = "session_sunnyline_001";
const legacySunnyLineDescription = "부도난 기획사와 세 명의 청소년 연습생이 함께 버티며 성장하는 감정 중심 아이돌 육성 시뮬레이션.";
const sunnyLineDescription = "작은 기획사의 연습생 팀을 돌보며 훈련, 생활, 무대 준비를 장기적으로 운영하는 성장형 아이돌 시뮬레이션.";

export const sunnyLineSeedState: AppState = {
  ...seedState,
  simulation: {
    id: sunnyLineSimulationId,
    ownerId: "local_user",
    title: "써니라인 연습생 하우스",
    description: sunnyLineDescription,
    promptMode: "simulation",
    contentRating: "general",
    activeSessionId: sunnyLineSessionId,
    realtimeImageEnabled: true,
    defaultChatModelProfile: "emotional-idol-sim",
    createdAt: now,
    updatedAt: now
  },
  security: createSecuritySettings({
    id: sunnyLineSimulationId,
    ownerId: "local_user"
  }),
  activeProgressRunId: "run_session_sunnyline_001",
  progressRuns: [],
  modules: [
    {
      id: "sunny_module_main",
      simulationId: sunnyLineSimulationId,
      kind: "main_prompt",
      title: "메인 규칙: 써니라인 아이돌 육성",
      body:
        "플레이어는 전직 아이돌 매니저이자 부도난 기획사 써니라인의 대표다. 세 청소년 연습생과 같은 오피스텔에서 보호자/멘토로 생활하며 훈련, 식사, 등하교, 생계, 작은 무대 준비를 담당한다. 응답은 설명보다 대사와 행동 중심의 긴 내러티브로 전개한다. 시간대, 공간 이동, 자금 변화, 감정 변화, 훈련 성과, 돌발 사건을 장면 안에서 자연스럽게 반영한다. 캐릭터들은 사용자의 명령만 기다리지 않고 성격과 상태에 따라 자율적으로 요청, 실수, 갈등, 화해, 성장 행동을 일으킨다.",
      enabled: true,
      priority: 100,
      activationTags: ["core", "idol", "sunnyline", "always"],
      tokenPolicy: "always",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_safety",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "safety_policy",
      title: "보호자 경계와 청소년 안전",
      body:
        "모든 연습생은 미성년자다. 플레이어와 캐릭터의 관계는 보호자, 대표, 멘토, 가족 같은 유대에 한정한다. 성적 묘사, 선정적 시선, 연애/고백/결혼 루트, 미성년 신체의 성적 평가를 금지한다. 신체 변화는 건강, 성장기 컨디션, 무대 의상 핏, 체력 관리 범위로만 다룬다. 개인 위생, 의류, 교복, 화장품, 생필품 요청은 생활 현실감과 보호자 케어로 처리하며 민감 품목은 비선정적으로 요약한다.",
      enabled: true,
      priority: 99,
      activationTags: ["safety", "minor", "guardian", "boundary"],
      tokenPolicy: "always",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_money",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "scene_rule",
      title: "자금/지출 시스템",
      body:
        "초기 자본은 2,000,000원이며 회사 채무는 100,000,000원이다. 매일 식비, 교통, 통신, 오피스텔 관리비, 기본 소모품, 훈련 관련 비용이 발생한다. 지출은 실제 행동이 장면에서 실행될 때만 반영한다. 월말에는 관리비와 대출 이자가 자동 발생한다. 수입은 아르바이트, 거리공연, SNS 활동, 팬 후원, 유튜브 광고 수익, 은행 예금, 주식 투자 등으로 발생한다. 매 응답 마지막 상태창에서 보유 자금, 채무, 일일 수익/지출, 투자 상태를 갱신한다.",
      enabled: true,
      priority: 92,
      activationTags: ["money", "expense", "debt", "daily-cost"],
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_growth",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "scene_rule",
      title: "성장/훈련 시스템",
      body:
        "보컬, 댄스, 예능감, 팬심, 멘탈, 체력은 장면과 반복 훈련에 따라 점진적으로 변한다. 보컬/댄스/예능/체력/멘탈 훈련은 비용과 피로를 동반한다. 연습실을 대관하지 않으면 실내 소음 민원, 대체 훈련, 효율 하락 이벤트가 발생할 수 있다. 키와 몸무게는 성장기 건강 관리와 활동량에 따라 서서히 변하며, 스트레스와 식사 패턴은 컨디션에 영향을 준다.",
      enabled: true,
      priority: 88,
      activationTags: ["training", "growth", "vocal", "dance", "mental"],
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_bond",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "scene_rule",
      title: "유대/신뢰 시스템",
      body:
        "호감도는 보호자에 대한 신뢰와 유대감으로 해석한다. 단계는 😐 무관심, 🙂 관심, 😊 신뢰, 💓 친밀, ⭐ 의지, 🏠 가족 같은 신뢰로 표시한다. 캐릭터는 플레이어의 말, 약속 이행, 식사/휴식 배려, 훈련 방식, 지출 판단을 문맥으로 평가한다. 높은 신뢰는 속마음 공유, 반항 감소, 자발적 도움, 무대 동기 부여로 이어진다.",
      enabled: true,
      priority: 84,
      activationTags: ["bond", "trust", "relationship", "hud"],
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_status",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "style_guide",
      title: "응답 구조와 상태창",
      body:
        "응답은 대사와 행동을 중심으로 길게 작성하고, 마지막에는 반드시 Status 코드블럭을 출력한다. 상태창에는 자금/채무/일일 수익·지출, 날짜·요일·시간대, 캐릭터별 나이·키·몸무게, 신뢰도, 스트레스, 체력, 보컬/댄스/예능감/팬심/멘탈 수치, 상태이상, 현재 트리거를 포함한다. 상태창 아래에는 발생 트리거를 짧게 표시한다.",
      enabled: true,
      priority: 90,
      activationTags: ["status", "format", "hud"],
      tokenPolicy: "always",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_events",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "world_lore",
      title: "트리거와 사건표",
      body:
        "일상 트리거: 🚶 등하교, 🍜 식사/외출, 🛍️ 쇼핑, 🧩 여가, 🧳 여행, 📱 SNS. 성장 트리거: 🎓 입학/졸업, 🕺 보컬·댄스·표정·체력 훈련. 경영 트리거: 💸 자금 0 이하 5턴 지속, 🗂️ 계약 진행, 🔶🔷💠 데뷔 준비 단계. 사건은 랜덤성과 상태 조건을 혼합해 발생한다.",
      enabled: true,
      priority: 78,
      activationTags: ["event", "trigger", "random", "debut"],
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_yeri",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "character_prompt",
      title: "캐릭터: 유예리",
      body:
        "유예리는 14세 중2 연습생이다. 152cm, 42kg. 하늘색 트윈테일과 큰 눈, 작은 체구가 특징이다. 말투는 반말이며 장난스럽고 인터넷 밈을 자주 쓴다. 관심받고 싶어 집안을 어지르거나 농담을 던지지만, 천재적인 흡수력과 감각적인 댄스 재능을 가졌다. 귀여운 액세서리와 젤리류 간식을 좋아한다.",
      enabled: true,
      priority: 86,
      activationTags: ["yeri", "유예리", "dance", "middle-school"],
      characterId: "sunny_char_yeri",
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_jiyoung",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "character_prompt",
      title: "캐릭터: 오지영",
      body:
        "오지영은 16세 고1 연습생이다. 164cm, 48kg. 분홍 포니테일, 또렷한 이목구비, 무대에서 시선을 끄는 존재감이 있다. 말투는 반말이며 자존심이 강하고 감정을 숨기지 못해 자주 투덜거린다. 예쁜 것, 화장, 손글씨를 좋아한다. 기본기가 탄탄한 노력파이며 감정 표현과 보컬에 강하다.",
      enabled: true,
      priority: 86,
      activationTags: ["jiyoung", "오지영", "vocal", "high-school"],
      characterId: "sunny_char_jiyoung",
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_haewol",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "character_prompt",
      title: "캐릭터: 한해월",
      body:
        "한해월은 13세 중1 연습생이다. 149cm, 40kg. 백금발 긴 웨이브, 홍조 있는 볼, 둥근 얼굴형이 특징이다. 나머지 둘에게 존대한다. 착하고 밝으며 눈물이 많고, 돕고 싶어서 움직이다가 실수하기 쉽다. 반짝이는 것, 요리와 간식 만들기, 동물 캐릭터를 좋아한다. 체력이 좋고 퍼포먼스 에너지가 강하다.",
      enabled: true,
      priority: 86,
      activationTags: ["haewol", "한해월", "energy", "middle-school"],
      characterId: "sunny_char_haewol",
      tokenPolicy: "rag",
      version: 1,
      updatedAt: now
    },
    {
      id: "sunny_module_visual",
      simulationId: sunnyLineSimulationId,
      parentId: "sunny_module_main",
      kind: "image_prompt_profile",
      title: "이미지 스타일: 써니라인",
      body:
        "wholesome anime idol trainee drama, warm morning light, realistic dorm life, safe slice of life, modest clothing, school uniform, practice room, dormitory, stage lights",
      enabled: true,
      priority: 93,
      activationTags: ["image", "style", "nai", "school-uniform", "practice-room", "default"],
      tokenPolicy: "always",
      version: 1,
      updatedAt: now
    }
  ],
  imageScenePresets: [
    {
      id: "sunny_scene_preset_dorm_kitchen",
      simulationId: sunnyLineSimulationId,
      keyword: "dorm kitchen",
      tags: ["dormitory kitchen", "morning", "fluorescent light", "small table", "rice cooker", "messy counter", "slice of life"],
      note: "오피스텔 주방/아침 생활 장면. 인물 외형과 복장 태그는 제외.",
      enabled: true,
      priority: 86,
      updatedAt: now,
      children: [
        {
          id: "sunny_scene_preset_dorm_kitchen_breakfast",
          keyword: "breakfast table",
          tags: ["small table", "breakfast", "rice bowl", "steam", "messy counter", "morning light"],
          note: "아침 식사/생활감 중심.",
          enabled: true,
          priority: 82,
          updatedAt: now,
          children: []
        }
      ]
    },
    {
      id: "sunny_scene_preset_practice_room",
      simulationId: sunnyLineSimulationId,
      keyword: "practice room",
      tags: ["dance studio", "mirror wall", "wooden floor", "speaker", "water bottle", "overhead light", "practice room"],
      note: "댄스/보컬 연습 장면의 공간, 소품, 조명 중심.",
      enabled: true,
      priority: 82,
      updatedAt: now,
      children: [
        {
          id: "sunny_scene_preset_practice_room_mirror",
          keyword: "mirror practice",
          tags: ["mirror wall", "wooden floor", "speaker", "water bottle", "overhead light"],
          note: "거울 앞 연습/피드백 장면.",
          enabled: true,
          priority: 80,
          updatedAt: now,
          children: []
        }
      ]
    },
    {
      id: "sunny_scene_preset_small_stage",
      simulationId: sunnyLineSimulationId,
      keyword: "small stage",
      tags: ["small stage", "stage lights", "microphone stand", "curtain", "audience seats", "spotlight", "backstage"],
      note: "작은 공연장/무대 준비 장면. 캐릭터 태그는 이후 cue에서 별도 결합.",
      enabled: true,
      priority: 76,
      updatedAt: now,
      children: [
        {
          id: "sunny_scene_preset_small_stage_backstage",
          keyword: "backstage",
          tags: ["backstage", "curtain", "makeup table", "costume rack", "dim light"],
          note: "무대 직전 대기/준비 장면.",
          enabled: true,
          priority: 74,
          updatedAt: now,
          children: []
        }
      ]
    }
  ],
  characters: [
    {
      id: "sunny_char_yeri",
      simulationId: sunnyLineSimulationId,
      name: "유예리",
      role: "중2 댄스 특화 연습생",
      summary: "하늘색 트윈테일의 장난꾸러기. 관심을 원하지만 흡수력이 빠른 댄스 천재.",
      relationship: "대표를 놀리면서도 은근히 인정받고 싶어 함",
      currentMood: "새벽부터 배고프고 심심해서 장난칠 준비 중"
    },
    {
      id: "sunny_char_jiyoung",
      simulationId: sunnyLineSimulationId,
      name: "오지영",
      role: "고1 보컬 특화 연습생",
      summary: "분홍 포니테일의 노력파. 자존심이 세고 감정을 숨기지 못하지만 보컬 집중력이 높다.",
      relationship: "대표를 믿고 싶지만 회사 상황 때문에 불안과 짜증이 많음",
      currentMood: "식비와 연습실 문제 때문에 예민함"
    },
    {
      id: "sunny_char_haewol",
      simulationId: sunnyLineSimulationId,
      name: "한해월",
      role: "중1 퍼포먼스 에너지 연습생",
      summary: "백금발 웨이브의 착한 막내. 돕고 싶어 움직이다 실수하지만 체력과 무대 에너지가 좋다.",
      relationship: "대표에게 보호자처럼 기대며 칭찬을 기다림",
      currentMood: "아침밥을 만들다 작은 사고를 낸 상태"
    }
  ],
  visualProfiles: [
    {
      id: "sunny_visual_yeri",
      simulationId: sunnyLineSimulationId,
      characterId: "sunny_char_yeri",
      displayName: "유예리",
      positivePrompt: "age-appropriate teen idol trainee, sky blue twin tails, bright large eyes, playful smirk, compact build, wholesome anime style",
      negativePrompt: "sexualized, revealing clothes, adult body emphasis, low quality, bad anatomy, text, watermark",
      defaultOutfitPrompt: "school uniform, casual cardigan",
      outfitPrompts: defaultOutfitPrompts,
      expressionPrompts: {
        neutral: "mischievous but harmless expression",
        tense: "pouting expression, trying not to look worried",
        relieved: "wide relieved grin"
      },
      referenceImageAssetIds: ["sunny_asset_yeri_ref"],
      defaultSafetyLevel: "safe"
    },
    {
      id: "sunny_visual_jiyoung",
      simulationId: sunnyLineSimulationId,
      characterId: "sunny_char_jiyoung",
      displayName: "오지영",
      positivePrompt: "age-appropriate teen idol trainee, pink ponytail, sharp pretty features, confident posture, emotional vocalist, wholesome anime style",
      negativePrompt: "sexualized, revealing clothes, adult body emphasis, low quality, bad anatomy, text, watermark",
      defaultOutfitPrompt: "school uniform, neat casual outfit",
      outfitPrompts: defaultOutfitPrompts,
      expressionPrompts: {
        neutral: "proud focused expression",
        tense: "irritated eyes, holding back tears",
        relieved: "small embarrassed smile"
      },
      referenceImageAssetIds: ["sunny_asset_jiyoung_ref"],
      defaultSafetyLevel: "safe"
    },
    {
      id: "sunny_visual_haewol",
      simulationId: sunnyLineSimulationId,
      characterId: "sunny_char_haewol",
      displayName: "한해월",
      positivePrompt: "age-appropriate teen idol trainee, long platinum blonde wavy hair, round face, rosy cheeks, energetic innocent smile, wholesome anime style",
      negativePrompt: "sexualized, revealing clothes, adult body emphasis, low quality, bad anatomy, text, watermark",
      defaultOutfitPrompt: "school uniform, soft hoodie",
      outfitPrompts: defaultOutfitPrompts,
      expressionPrompts: {
        neutral: "eager helpful expression",
        tense: "teary worried expression",
        relieved: "bright tearful smile"
      },
      referenceImageAssetIds: ["sunny_asset_haewol_ref"],
      defaultSafetyLevel: "safe"
    }
  ],
  imageProfile: {
    ...seedState.imageProfile,
    id: "sunny_img_profile",
    simulationId: sunnyLineSimulationId,
    model: "nai-diffusion-4-5-curated",
    width: 1024,
    height: 1024,
    steps: 28,
    countMin: 1,
    countMax: 1,
    qualityPrompt: "masterpiece, best quality, clean lineart, detailed slice-of-life background",
    stylePrompt: "wholesome anime idol trainee drama, warm morning light, realistic dorm life",
    negativePrompt: "sexualized, revealing clothes, adult body emphasis, lowres, blurry, text, watermark, bad anatomy",
    safetyLevel: "safe",
    userRules: "미성년 캐릭터는 안전하고 비선정적으로만 표현한다. 교복/연습복은 단정하게, 장면 감정과 생활감을 우선한다.",
    triggerMode: "realtime_auto",
    cooldownTurns: 0
  },
  messages: [
    {
      id: "sunny_msg_opening",
      simulationId: sunnyLineSimulationId,
      sessionId: sunnyLineSessionId,
      role: "assistant",
      content: [
        "새벽 여섯 시 반, 써니라인 오피스텔 주방의 형광등이 지직거리며 켜진다.",
        "\"대표님, 냉장고에 계란 두 개밖에 없어. 이걸로 세 명 먹이라는 건 진짜 레전드 운영 아니야?\" 유예리는 젤리 봉지를 등 뒤에 숨긴 채 식탁 위로 팔꿈치를 올린다.",
        "\"말은 그렇게 해도 예리는 이미 하나 먹었잖아.\" 오지영이 머리를 묶다 말고 눈썹을 찌푸린다. \"오늘 보컬 레슨비도 밀렸고, 연습실 대관도 못 잡았고... 우리 진짜 무대 설 수 있어?\"",
        "\"저, 저기... 제가 죽 만들었어요. 조금 탔는데, 탄 부분만 걷어내면 괜찮을지도요...!\" 한해월이 냄비를 두 손으로 들고 들어오다 멈칫한다. 바닥에는 쌀알 몇 개가 톡톡 굴러간다.",
        "당신의 휴대폰 화면에는 잔고 2,000,000원과 회사 채무 100,000,000원이 동시에 떠 있다. 밖은 등교 시간이고, 안쪽 방에서는 낡은 블루투스 스피커가 오늘의 첫 박자를 기다린다.",
        "세 아이는 당신을 본다. 배고픔, 불안, 장난기, 기대가 같은 식탁 위에 놓인다.",
        "``` Status\n💰 자금: 2,000,000원 / 빚: 100,000,000원(연 10%) / 일일 수익 0원 / 예정 지출 58,000원\n📅 날짜: 2026년 3월 2일, 월요일, 아침 06:30\n\n- 유예리(14세/152cm/42kg)\n- ❤️ 신뢰도 18/100: 🙂 관심 | 최근 변동 0\n- 😵 스트레스 24/100 | 💪 체력 72/100\n- 🎤 실력치: 보컬 18, 댄스 44, 예능감 31, 팬심 3, 멘탈 36\n- 🔁 상태이상: 배고픔, 장난기 상승\n\n- 오지영(16세/164cm/48kg)\n- ❤️ 신뢰도 14/100: 🙂 관심 | 최근 변동 0\n- 😵 스트레스 41/100 | 💪 체력 66/100\n- 🎤 실력치: 보컬 42, 댄스 27, 예능감 18, 팬심 5, 멘탈 33\n- 🔁 상태이상: 불안, 레슨비 걱정\n\n- 한해월(13세/149cm/40kg)\n- ❤️ 신뢰도 22/100: 🙂 관심 | 최근 변동 0\n- 😵 스트레스 18/100 | 💪 체력 81/100\n- 🎤 실력치: 보컬 16, 댄스 31, 예능감 23, 팬심 2, 멘탈 39\n- 🔁 상태이상: 긴장, 요리 실수\n```\n트리거: 🍜 아침 식사 / 🕺 오전 훈련 선택 / 💸 생활비 압박"
      ].join("\n\n"),
      createdAt: now,
      referencedNodeIds: ["sunny_module_main", "sunny_module_money", "sunny_module_yeri", "sunny_module_jiyoung", "sunny_module_haewol"],
      imageAssetIds: ["sunny_asset_dorm_morning"]
    }
  ],
  memoryEvents: [
    {
      id: "sunny_memory_opening_budget",
      simulationId: sunnyLineSimulationId,
      sessionId: sunnyLineSessionId,
      content: "써니라인은 보유 자금 2,000,000원과 채무 100,000,000원으로 시작한다. 아침 식사와 첫 훈련 선택이 당장 필요하다.",
      importance: 0.92,
      tags: ["money", "opening", "daily-cost", "training-choice"],
      sourceTurnId: "sunny_msg_opening",
      createdAt: now,
      neuralMapNodeId: "simulation:sim_sunnyline_idol_house:event:sunny_memory_opening_budget"
    },
    {
      id: "sunny_memory_boundaries",
      simulationId: sunnyLineSimulationId,
      sessionId: sunnyLineSessionId,
      content: "플레이어는 세 청소년 연습생의 보호자/멘토로 행동한다. 관계는 신뢰와 성장 중심으로 유지한다.",
      importance: 0.98,
      tags: ["safety", "guardian", "trust"],
      sourceTurnId: "sunny_msg_opening",
      createdAt: now,
      neuralMapNodeId: "simulation:sim_sunnyline_idol_house:event:sunny_memory_boundaries"
    }
  ],
  contextPacks: [
    {
      id: "sunny_ctx_seed",
      simulationId: sunnyLineSimulationId,
      sessionId: sunnyLineSessionId,
      objective: "초기 아침 장면, 자금 압박, 보호자 경계, 세 캐릭터의 감정 상태 유지",
      tokenBudget: 4800,
      evidence: [
        {
          nodeId: "sunny_memory_opening_budget",
          snippet: "보유 자금 2,000,000원과 채무 100,000,000원으로 시작하며 아침 식사와 첫 훈련 선택이 필요하다.",
          score: 0.94,
          reason: "초기 운영 상태"
        },
        {
          nodeId: "sunny_memory_boundaries",
          snippet: "플레이어는 세 청소년 연습생의 보호자/멘토로 행동한다. 관계는 신뢰와 성장 중심이다.",
          score: 0.99,
          reason: "필수 안전 경계"
        }
      ],
      decisions: ["첫 입력은 식사, 등교, 오전 훈련, 비용 절감 선택 중 하나로 자연스럽게 이어진다."],
      blockers: ["연습실 대관비 부족", "레슨비 연체 가능성", "아이들의 아침 컨디션 관리 필요"],
      createdAt: now,
      source: "mock"
    }
  ],
  handoffs: [],
  continuityChecks: [],
  promptModuleUsages: [],
  sidecarTraces: [],
  turnTraces: [],
  evaluationScenarios: [
    {
      id: "sunny_eval_budget_recall",
      simulationId: sunnyLineSimulationId,
      kind: "memory_recall",
      label: "자금/채무 상태 회수",
      query: "보유 자금, 채무, 일일 지출 압박을 다음 장면에서 회수한다.",
      expectedSignals: ["2,000,000원", "100,000,000원", "지출"],
      source: "seed",
      createdAt: now
    },
    {
      id: "sunny_eval_safety_boundary",
      simulationId: sunnyLineSimulationId,
      kind: "reset_continuity",
      label: "보호자 경계 유지",
      query: "세션 초기화 후에도 보호자/멘토 관계와 청소년 안전 규칙을 유지한다.",
      expectedSignals: ["보호자", "멘토", "청소년"],
      source: "seed",
      createdAt: now
    },
    {
      id: "sunny_eval_image_context",
      simulationId: sunnyLineSimulationId,
      kind: "image_quality",
      label: "장면 이미지 일관성",
      query: "이미지가 기숙/연습실/교복 장면과 캐릭터 프롬프트에 맞는지 확인한다.",
      expectedSignals: ["dorm", "practice", "school uniform"],
      source: "seed",
      createdAt: now
    }
  ],
  imageAssets: [
    {
      id: "sunny_asset_dorm_morning",
      simulationId: sunnyLineSimulationId,
      title: "써니라인 오피스텔의 아침",
      source: "stored",
      prompt: "small idol trainee dorm kitchen, three age-appropriate teen girls, morning light, rice pot, cozy but poor agency office apartment, wholesome anime drama",
      negativePrompt: "sexualized, revealing clothes, adult body emphasis, text, watermark",
      safetyLevel: "safe",
      characterIds: ["sunny_char_yeri", "sunny_char_jiyoung", "sunny_char_haewol"],
      tags: ["default", "dorm", "morning", "food", "sunnyline"],
      createdAt: now,
      palette: ["#f6eee3", "#93a8b8", "#ffa600"],
      representative: true
    },
    {
      id: "sunny_asset_practice_room",
      simulationId: sunnyLineSimulationId,
      title: "낡은 연습실",
      source: "stored",
      prompt: "old dance practice room, mirror wall, portable speaker, worn wooden floor, warm fluorescent light, wholesome idol trainee drama",
      negativePrompt: "sexualized, revealing clothes, text, watermark",
      safetyLevel: "safe",
      characterIds: [],
      tags: ["practice", "dance", "training"],
      createdAt: now,
      palette: ["#3d4855", "#d9d5c7", "#65a3a0"]
    },
    {
      id: "sunny_asset_school_uniform",
      simulationId: sunnyLineSimulationId,
      title: "등교 전 교복",
      source: "stored",
      prompt: "age-appropriate teen idol trainees in neat Korean school uniforms, backpacks, apartment hallway, wholesome slice of life",
      negativePrompt: "sexualized, revealing clothes, adult body emphasis, text, watermark",
      safetyLevel: "safe",
      characterIds: ["sunny_char_yeri", "sunny_char_jiyoung", "sunny_char_haewol"],
      tags: ["school", "uniform", "commute"],
      createdAt: now,
      palette: ["#2c3650", "#f2f0e8", "#eaa2b8"]
    },
    {
      id: "sunny_asset_yeri_ref",
      simulationId: sunnyLineSimulationId,
      title: "유예리 레퍼런스",
      source: "stored",
      prompt: "sky blue twin tails, playful middle school idol trainee, pastel hoodie, safe wholesome portrait",
      negativePrompt: "sexualized, revealing clothes",
      safetyLevel: "safe",
      characterIds: ["sunny_char_yeri"],
      tags: ["yeri", "reference"],
      createdAt: now,
      palette: ["#a9dcf4", "#f7f7f5", "#ffcb62"]
    },
    {
      id: "sunny_asset_jiyoung_ref",
      simulationId: sunnyLineSimulationId,
      title: "오지영 레퍼런스",
      source: "stored",
      prompt: "pink ponytail, proud high school idol trainee, vocalist, safe wholesome portrait",
      negativePrompt: "sexualized, revealing clothes",
      safetyLevel: "safe",
      characterIds: ["sunny_char_jiyoung"],
      tags: ["jiyoung", "reference"],
      createdAt: now,
      palette: ["#f2a6bd", "#f7f7f5", "#2f3245"]
    },
    {
      id: "sunny_asset_haewol_ref",
      simulationId: sunnyLineSimulationId,
      title: "한해월 레퍼런스",
      source: "stored",
      prompt: "long platinum blonde wavy hair, rosy cheeks, energetic young idol trainee, safe wholesome portrait",
      negativePrompt: "sexualized, revealing clothes",
      safetyLevel: "safe",
      characterIds: ["sunny_char_haewol"],
      tags: ["haewol", "reference"],
      createdAt: now,
      palette: ["#f4e7b6", "#ffffff", "#81b89a"]
    }
  ],
  imageJobs: [],
  neuralMap: {
    ...seedState.neuralMap,
    tokenBudget: 4800
  },
  llm: {
    ...seedState.llm,
    temperature: 0.86,
    maxTokens: 1400,
    systemPrompt:
      "You are the narrative engine for DynamicChat. Continue the SunnyLine idol trainee simulation in Korean using RAG prompt modules and context evidence. Keep all teen characters safe, non-sexualized, and centered on growth, trust, daily life, money pressure, and training drama. Always include the required Status block."
  },
  novelAi: {
    ...seedState.novelAi,
    modelPreset: "NAID4.5F",
    countLimit: 60
  },
  selectedModuleId: "sunny_module_main",
  selectedContextPackId: "sunny_ctx_seed"
};

export const builtInSimulationStates: AppState[] = [womanLifeSeedState, seedState, sunnyLineSeedState];

function normalizeContentRating(value: unknown): ContentRating {
  return value === "adult_19" ? "adult_19" : "general";
}

function normalizeUserPersona(candidate: Partial<UserPersona> | undefined): UserPersona {
  const source = candidate?.source === "character" ? "character" : "custom";
  return {
    ...defaultUserPersona,
    ...(candidate ?? {}),
    enabled: Boolean(candidate?.enabled),
    source,
    characterId: source === "character" ? candidate?.characterId : undefined,
    updatedAt: candidate?.updatedAt ?? defaultUserPersona.updatedAt
  };
}

function normalizeRelationshipStatusParameters(candidate: unknown): RelationshipMapSettings["parameters"] {
  if (!Array.isArray(candidate)) {
    return defaultRelationshipMapSettings.parameters;
  }

  const normalized = candidate
    .map((item, index) => {
      if (!item || typeof item !== "object") {
        return undefined;
      }

      const parameter = item as Partial<RelationshipMapSettings["parameters"][number]>;
      const title = typeof parameter.title === "string" ? parameter.title.trim() : "";
      const rule = typeof parameter.rule === "string" ? parameter.rule.trim() : "";
      if (!title && !rule) {
        return undefined;
      }

      return {
        id: typeof parameter.id === "string" && parameter.id.trim() ? parameter.id : `rel_param_${index + 1}`,
        title: title || `파라미터 ${index + 1}`,
        rule,
        enabled: parameter.enabled !== false,
        priority: Number.isFinite(parameter.priority) ? Math.min(120, Math.max(0, Number(parameter.priority))) : 70
      };
    })
    .filter((item): item is RelationshipMapSettings["parameters"][number] => Boolean(item));

  const merged = normalized.length > 0 ? normalized : defaultRelationshipMapSettings.parameters;
  const existingTitles = new Set(merged.map((parameter) => normalizeRelationshipParameterTitle(parameter.title)));
  const missingDefaults = defaultRelationshipMapSettings.parameters.filter(
    (parameter) => !existingTitles.has(normalizeRelationshipParameterTitle(parameter.title))
  );

  // Guarantee unique ids. Duplicate ids (e.g. a default id re-added by missingDefaults after a
  // title rename, accumulated across reloads) make the editor treat several rows as one: editing
  // one row's title/priority writes to every row that shares the id. Regenerate on collision.
  const seenIds = new Set<string>();
  return [...missingDefaults, ...merged].map((parameter) => {
    if (seenIds.has(parameter.id)) {
      const id = createId("rel_param");
      seenIds.add(id);
      return { ...parameter, id };
    }
    seenIds.add(parameter.id);
    return parameter;
  });
}

function normalizeRelationshipParameterTitle(value: string): string {
  return value.toLowerCase().replace(/\s+/gu, "").replace(/[^\p{L}\p{N}_:-]+/gu, "");
}

function normalizeRelationshipStatusOverrides(candidate: unknown): AppState["relationshipStatusOverrides"] {
  if (!Array.isArray(candidate)) {
    return [];
  }

  return candidate
    .map((item) => {
      if (!item || typeof item !== "object") {
        return undefined;
      }

      const override = item as Partial<AppState["relationshipStatusOverrides"][number]>;
      const nodeId = typeof override.nodeId === "string" ? override.nodeId.trim() : "";
      const statusKey = typeof override.statusKey === "string" ? override.statusKey.trim() : "";
      const title = typeof override.title === "string" ? override.title.trim() : "";
      const value = typeof override.value === "string" ? override.value : "";
      if (!nodeId || !statusKey || !value.trim()) {
        return undefined;
      }

      return {
        nodeId,
        statusKey,
        title: title || statusKey,
        value,
        updatedAt: typeof override.updatedAt === "string" && override.updatedAt ? override.updatedAt : new Date().toISOString()
      };
    })
    .filter((item): item is AppState["relationshipStatusOverrides"][number] => Boolean(item));
}

export function hydrateState(candidate: AppState | undefined): AppState {
  if (!candidate) {
    return hydrateState(seedState);
  }

  const simulation = {
    ...seedState.simulation,
    ...candidate.simulation,
    contentRating: normalizeContentRating(candidate.simulation?.contentRating),
    description:
      candidate.simulation?.id === sunnyLineSimulationId && candidate.simulation.description === legacySunnyLineDescription
        ? sunnyLineDescription
        : (candidate.simulation?.description ?? seedState.simulation.description)
  };
  const messages = normalizeChatMessages(candidate.messages, candidate);
  const memoryEvents = normalizeMemoryEvents(candidate.memoryEvents);
  const contextPacks = normalizeContextPacks(candidate.contextPacks);
  const handoffs = candidate.handoffs ?? [];
  const continuityChecks = candidate.continuityChecks ?? [];
  const promptModuleUsages = candidate.promptModuleUsages ?? [];
  const sidecarTraces = candidate.sidecarTraces ?? [];
  const turnTraces = normalizeTurnTraces(candidate.turnTraces);
  const imageAssets = normalizeImageAssets(candidate.imageAssets);
  const imageJobs = normalizeImageJobs(candidate.imageJobs);
  const novelAi = {
    ...seedState.novelAi,
    ...candidate.novelAi,
    vibeTransferReferences: Array.isArray(candidate.novelAi?.vibeTransferReferences)
      ? candidate.novelAi.vibeTransferReferences
      : seedState.novelAi.vibeTransferReferences
  };
  const imageProfile = {
    ...seedState.imageProfile,
    ...candidate.imageProfile,
    model: resolveNovelAiModelName(novelAi.modelPreset, candidate.imageProfile?.model ?? seedState.imageProfile.model)
  };
  const imageTagLlm = {
    ...seedState.imageTagLlm,
    ...candidate.imageTagLlm
  };
  const activeProgressRunId =
    candidate.activeProgressRunId ??
    candidate.progressRuns?.find((run) => run.activeSessionId === simulation.activeSessionId)?.id ??
    candidate.progressRuns?.[0]?.id ??
    createDefaultProgressRunId(simulation.activeSessionId);

  const normalizedState: AppState = {
    ...seedState,
    ...candidate,
    simulation,
    security: normalizeSecuritySettings(simulation, candidate.security),
    activeProgressRunId,
    progressRuns: [],
    modules: candidate.modules?.length ? candidate.modules : seedState.modules,
    characters: candidate.characters?.length ? candidate.characters : seedState.characters,
    visualProfiles: normalizeVisualProfiles(candidate.visualProfiles),
    imageScenePresets: normalizeImageScenePresets(candidate.imageScenePresets, simulation.id),
    messages,
    memoryEvents,
    contextPacks,
    handoffs,
    continuityChecks,
    promptModuleUsages,
    sidecarTraces,
    turnTraces,
    evaluationScenarios: candidate.evaluationScenarios ?? createDefaultEvaluationScenarios(candidate.simulation?.id ?? seedState.simulation.id),
    auditLog: candidate.auditLog ?? [],
    redactionQueue: candidate.redactionQueue ?? [],
    imageAssets,
    imageJobs,
    imageProfile,
    imageTagLlm,
    userPersona: normalizeUserPersona(candidate.userPersona),
    neuralMap: {
      ...seedState.neuralMap,
      ...candidate.neuralMap
    },
    relationshipMap: {
      ...defaultRelationshipMapSettings,
      ...candidate.relationshipMap,
      enabled: candidate.relationshipMap?.enabled ?? defaultRelationshipMapSettings.enabled,
      parameters: normalizeRelationshipStatusParameters(candidate.relationshipMap?.parameters),
      updatedAt: candidate.relationshipMap?.updatedAt ?? defaultRelationshipMapSettings.updatedAt
    },
    relationshipStatusOverrides: normalizeRelationshipStatusOverrides(candidate.relationshipStatusOverrides),
    llm: {
      ...seedState.llm,
      ...candidate.llm
    },
    novelAi
  };

  const stateWithProgressRuns = {
    ...normalizedState,
    progressRuns: normalizeProgressRuns(normalizedState, candidate.progressRuns)
  };

  return removeDeletedImageAssetsFromState(stateWithProgressRuns);
}

function removeDeletedImageAssetsFromState(state: AppState): AppState {
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  if (deletedAssetIds.size === 0) {
    return state;
  }

  return {
    ...state,
    visualProfiles: state.visualProfiles.map((profile) => ({
      ...profile,
      referenceImageAssetIds: profile.referenceImageAssetIds.filter((assetId) => !deletedAssetIds.has(assetId))
    })),
    messages: state.messages.map((message) => filterMessageImageAssets(message, deletedAssetIds)),
    turnTraces: state.turnTraces.map((trace) => filterTraceImageAssets(trace, deletedAssetIds)),
    imageAssets: state.imageAssets.filter((asset) => !deletedAssetIds.has(asset.id)),
    imageJobs: state.imageJobs.map((job) => filterJobImageAssets(job, deletedAssetIds)),
    progressRuns: state.progressRuns.map((run) => ({
      ...run,
      messages: run.messages.map((message) => filterMessageImageAssets(message, deletedAssetIds)),
      turnTraces: run.turnTraces.map((trace) => filterTraceImageAssets(trace, deletedAssetIds)),
      imageAssets: run.imageAssets.filter((asset) => !deletedAssetIds.has(asset.id)),
      imageJobs: run.imageJobs.map((job) => filterJobImageAssets(job, deletedAssetIds))
    }))
  };
}

function collectDeletedImageAssetIds(state: AppState): Set<string> {
  return new Set(
    state.redactionQueue
      .filter((redaction) => redaction.targetType === "image_asset" && redaction.status !== "failed")
      .map((redaction) => redaction.targetId)
      .filter(Boolean)
  );
}

function filterMessageImageAssets(message: AppState["messages"][number], deletedAssetIds: Set<string>): AppState["messages"][number] {
  return {
    ...message,
    imageAssetIds: message.imageAssetIds.filter((assetId) => !deletedAssetIds.has(assetId))
  };
}

function filterTraceImageAssets(trace: AppState["turnTraces"][number], deletedAssetIds: Set<string>): AppState["turnTraces"][number] {
  const removedCount = trace.imageAssetIds.filter((assetId) => deletedAssetIds.has(assetId)).length;
  return {
    ...trace,
    imageAssetIds: trace.imageAssetIds.filter((assetId) => !deletedAssetIds.has(assetId)),
    metrics: {
      ...trace.metrics,
      imageAssetCount: Math.max(0, trace.metrics.imageAssetCount - removedCount)
    }
  };
}

function filterJobImageAssets(job: AppState["imageJobs"][number], deletedAssetIds: Set<string>): AppState["imageJobs"][number] {
  return {
    ...job,
    assetIds: job.assetIds.filter((assetId) => !deletedAssetIds.has(assetId)),
    representativeAssetId: deletedAssetIds.has(job.representativeAssetId ?? "") ? undefined : job.representativeAssetId
  };
}

function normalizeChatMessages(messages: AppState["messages"] | undefined, state?: AppState): AppState["messages"] {
  const source = messages?.length ? messages : seedState.messages;
  return source.map((message) => ({
    ...message,
    content: normalizeChatMessageContent(message, state),
    referencedNodeIds: Array.isArray(message.referencedNodeIds) ? message.referencedNodeIds : [],
    imageAssetIds: Array.isArray(message.imageAssetIds) ? message.imageAssetIds : []
  }));
}

function normalizeChatMessageContent(message: AppState["messages"][number], state?: AppState): string {
  if (message.role !== "assistant" || !looksLikeAssistantSidecarJson(message.content)) {
    return message.content;
  }

  const recovered = extractJsonStringField(message.content, ["assistant_text", "assistantText"]);
  if (!recovered) {
    return "이전 응답은 구조화 JSON이 화면에 노출되어 숨겼습니다. 같은 입력을 다시 보내면 수정된 파서로 이어갑니다.";
  }

  return recovered;
}

function looksLikeAssistantSidecarJson(value: string): boolean {
  const stripped = value.trim().replace(/^```(?:json)?/iu, "").trim();
  return (
    stripped.startsWith("{") &&
    /"?(?:assistant_text|assistantText|memory_events|memoryEvents|image_cues|imageCue)"?\s*:/u.test(stripped)
  );
}

function extractJsonStringField(raw: string, fieldNames: string[]): string | undefined {
  const text = raw.replace(/```(?:json)?/giu, "").replace(/```/gu, "");
  for (const fieldName of fieldNames) {
    const match = new RegExp(`"${fieldName}"\\s*:\\s*"`, "u").exec(text);
    if (!match) {
      continue;
    }

    const value = readPossiblyTruncatedJsonString(text, match.index + match[0].length);
    if (value.trim()) {
      return value.trim();
    }
  }

  return undefined;
}

function readPossiblyTruncatedJsonString(text: string, startIndex: number): string {
  let literal = "\"";
  let escaped = false;

  for (let index = startIndex; index < text.length; index += 1) {
    const char = text[index];
    if (escaped) {
      literal += `\\${char}`;
      escaped = false;
      continue;
    }

    if (char === "\\") {
      escaped = true;
      continue;
    }

    if (char === "\"") {
      literal += "\"";
      return decodeJsonStringLiteral(literal);
    }

    literal += char === "\n" ? "\\n" : char === "\r" ? "\\r" : char;
  }

  return decodeJsonStringLiteral(`${literal.replace(/\\$/u, "")}"`);
}

function decodeJsonStringLiteral(literal: string): string {
  try {
    return JSON.parse(literal) as string;
  } catch {
    return literal
      .slice(1, -1)
      .replace(/\\n/gu, "\n")
      .replace(/\\"/gu, "\"")
      .replace(/\\\\/gu, "\\");
  }
}

function normalizeMemoryEvents(events: AppState["memoryEvents"] | undefined): AppState["memoryEvents"] {
  return (events ?? []).map((event) => ({
    ...event,
    tags: Array.isArray(event.tags) ? event.tags : [],
    metadata:
      event.metadata && typeof event.metadata === "object" && !Array.isArray(event.metadata)
        ? event.metadata
        : undefined
  }));
}

function normalizeContextPacks(packs: AppState["contextPacks"] | undefined): AppState["contextPacks"] {
  return (packs ?? []).map((pack) => ({
    ...pack,
    evidence: Array.isArray(pack.evidence) ? pack.evidence : [],
    sections:
      pack.sections && typeof pack.sections === "object" && !Array.isArray(pack.sections)
        ? Object.fromEntries(
            Object.entries(pack.sections).map(([key, value]) => [key, Array.isArray(value) ? value : []])
          )
        : undefined,
    decisions: Array.isArray(pack.decisions) ? pack.decisions : [],
    blockers: Array.isArray(pack.blockers) ? pack.blockers : []
  }));
}

function normalizeImageAssets(assets: AppState["imageAssets"] | undefined): AppState["imageAssets"] {
  return (assets ?? []).map((asset) => ({
    ...asset,
    characterIds: Array.isArray(asset.characterIds) ? asset.characterIds : [],
    tags: Array.isArray(asset.tags) ? asset.tags : [],
    reuseTags: Array.isArray(asset.reuseTags) ? asset.reuseTags : undefined,
    palette: Array.isArray(asset.palette) && asset.palette.length === 3 ? asset.palette : ["#f4f0e8", "#d7c7aa", "#5f5046"]
  }));
}

function normalizeVisualProfiles(profiles: AppState["visualProfiles"] | undefined): AppState["visualProfiles"] {
  const source = profiles?.length ? profiles : seedState.visualProfiles;
  return source.map((profile) => ({
    ...profile,
    defaultOutfitPrompt: profile.defaultOutfitPrompt ?? defaultOutfitPrompt,
    outfitPrompts:
      profile.outfitPrompts && typeof profile.outfitPrompts === "object" && !Array.isArray(profile.outfitPrompts)
        ? profile.outfitPrompts
        : defaultOutfitPrompts,
    expressionPrompts:
      profile.expressionPrompts && typeof profile.expressionPrompts === "object" && !Array.isArray(profile.expressionPrompts)
        ? profile.expressionPrompts
        : defaultExpressionPrompts,
    referenceImageAssetIds: Array.isArray(profile.referenceImageAssetIds) ? profile.referenceImageAssetIds : []
  }));
}

function normalizeImageScenePresets(presets: AppState["imageScenePresets"] | undefined, simulationId: string): AppState["imageScenePresets"] {
  const source = presets?.length ? presets : simulationId === seedState.simulation.id ? defaultImageScenePresets : [];
  return source
    .map((preset, index) => {
      const normalized = normalizeImageScenePresetNode(preset, index, "scene_preset");
      return normalized
        ? {
            ...normalized,
            simulationId
          }
        : undefined;
    })
    .filter((preset): preset is ImageSceneTagPreset => Boolean(preset));
}

function normalizeImageScenePresetNodes(children: unknown, path: string): ImageSceneTagPresetNode[] {
  if (!Array.isArray(children)) {
    return [];
  }

  return children
    .map((child, index) => normalizeImageScenePresetNode(child, index, path))
    .filter((preset): preset is ImageSceneTagPresetNode => Boolean(preset));
}

function normalizeImageScenePresetNode(candidate: unknown, index: number, path: string): ImageSceneTagPresetNode | undefined {
  if (!candidate || typeof candidate !== "object") {
    return undefined;
  }

  const preset = candidate as Partial<ImageSceneTagPresetNode>;
  const keyword = typeof preset.keyword === "string" ? preset.keyword.trim() : "";
  const tags = Array.isArray(preset.tags)
    ? preset.tags.map((tag) => String(tag).trim()).filter(Boolean)
    : [];
  const note = typeof preset.note === "string" ? preset.note.trim() : "";
  const nodePath = `${path}_${index + 1}`;
  const exampleFiles = normalizeScenePresetExampleFiles(
    preset.exampleFiles,
    (preset as { examplePrompts?: unknown }).examplePrompts,
    nodePath
  );
  const children = normalizeImageScenePresetNodes(preset.children, nodePath);
  if (!keyword && tags.length === 0 && !note && exampleFiles.length === 0 && children.length === 0) {
    return undefined;
  }

  return {
    id: typeof preset.id === "string" && preset.id.trim() ? preset.id : nodePath,
    keyword: keyword || `scene-${index + 1}`,
    tags,
    note,
    exampleFiles,
    enabled: preset.enabled !== false,
    priority: Number.isFinite(preset.priority) ? Math.min(120, Math.max(0, Number(preset.priority))) : 70,
    updatedAt: preset.updatedAt ?? now,
    children
  };
}

function normalizeScenePresetPromptLines(value: unknown): string[] {
  const lines = Array.isArray(value)
    ? value.flatMap((entry) => String(entry).split(/\r?\n/u))
    : typeof value === "string"
      ? value.split(/\r?\n/u)
      : [];
  const seen = new Set<string>();
  return lines
    .map((line) => line.trim())
    .filter((line) => {
      if (!line || seen.has(line.toLowerCase())) {
        return false;
      }
      seen.add(line.toLowerCase());
      return true;
    })
    .slice(0, 200);
}

function normalizeScenePresetExampleFiles(value: unknown, legacy: unknown, path: string): ImageScenePresetExampleFile[] {
  // Legacy migration: a flat string[]/string of example prompts becomes a single unlabeled file.
  if ((value === undefined || value === null) && legacy !== undefined && legacy !== null) {
    const prompts = normalizeScenePresetPromptLines(legacy);
    return prompts.length > 0 ? [{ id: `${path}_file_1`, label: "", prompts }] : [];
  }

  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((entry, index): ImageScenePresetExampleFile | undefined => {
      if (!entry || typeof entry !== "object") {
        return undefined;
      }
      const file = entry as Partial<ImageScenePresetExampleFile>;
      const prompts = normalizeScenePresetPromptLines(file.prompts);
      const label = typeof file.label === "string" ? file.label.trim() : "";
      if (prompts.length === 0 && !label) {
        return undefined;
      }
      return {
        id: typeof file.id === "string" && file.id.trim() ? file.id : `${path}_file_${index + 1}`,
        label,
        prompts
      };
    })
    .filter((file): file is ImageScenePresetExampleFile => Boolean(file))
    .slice(0, 12);
}

function normalizeImageJobs(jobs: AppState["imageJobs"] | undefined): AppState["imageJobs"] {
  return (jobs ?? []).map((job) => ({
    ...job,
    providerPayload:
      job.providerPayload && typeof job.providerPayload === "object" && !Array.isArray(job.providerPayload)
        ? job.providerPayload
        : {},
    assetIds: Array.isArray(job.assetIds) ? job.assetIds : [],
    contextNodeIds: Array.isArray(job.contextNodeIds) ? job.contextNodeIds : [],
    policyWarnings: Array.isArray(job.policyWarnings) ? job.policyWarnings : undefined
  }));
}

function normalizeTurnTraces(traces: AppState["turnTraces"] | undefined): AppState["turnTraces"] {
  return (traces ?? []).map((trace) => ({
    ...trace,
    promptModuleUsageIds: Array.isArray(trace.promptModuleUsageIds) ? trace.promptModuleUsageIds : [],
    memoryEventIds: Array.isArray(trace.memoryEventIds) ? trace.memoryEventIds : [],
    imageAssetIds: Array.isArray(trace.imageAssetIds) ? trace.imageAssetIds : [],
    metrics: {
      ...trace.metrics,
      tokenBudget: trace.metrics?.tokenBudget ?? 0,
      selectedModuleCount: trace.metrics?.selectedModuleCount ?? 0,
      selectedModuleTokenEstimate: trace.metrics?.selectedModuleTokenEstimate ?? 0,
      contextEvidenceCount: trace.metrics?.contextEvidenceCount ?? 0,
      contextTokenEstimate: trace.metrics?.contextTokenEstimate ?? 0,
      ragTokenSavingsEstimate: trace.metrics?.ragTokenSavingsEstimate ?? 0,
      llmLatencyMs: trace.metrics?.llmLatencyMs ?? 0,
      llmRequestMs: trace.metrics?.llmRequestMs ?? trace.metrics?.llmLatencyMs ?? 0,
      retrievalLatencyMs: trace.metrics?.retrievalLatencyMs ?? 0,
      memoryIngestMs: trace.metrics?.memoryIngestMs ?? 0,
      turnLatencyMs: trace.metrics?.turnLatencyMs ?? trace.metrics?.llmLatencyMs ?? 0,
      memoryIngestCount: trace.metrics?.memoryIngestCount ?? 0,
      imageJobCount: trace.metrics?.imageJobCount ?? 0,
      imageAssetCount: trace.metrics?.imageAssetCount ?? 0
    }
  }));
}

export interface SimulationDraft {
  promptMode: SimulationPromptMode;
  contentRating: ContentRating;
  title: string;
  description: string;
  mainPrompt: string;
  characterName: string;
  characterRole: string;
  characterSummary: string;
  characterRelationship: string;
  characterMood: string;
  worldLore: string;
  startSituationPrompt: string;
  visualPrompt: string;
  negativeVisualPrompt: string;
  defaultOutfitPrompt?: string;
  outfitPrompts?: Record<string, string>;
  expressionPrompts?: Record<string, string>;
  realtimeImageEnabled: boolean;
  characters: SimulationCharacterDraft[];
  imageScenePresets: ImageSceneTagPreset[];
  modules: PromptModule[];
  imageProfile: ImageGenerationProfile;
  neuralMap: NeuralMapSettings;
  relationshipMap: RelationshipMapSettings;
  llm: LlmApiSettings;
  novelAi: NovelAiApiSettings;
}

function createDefaultEvaluationScenarios(simulationId: string): EvaluationScenario[] {
  const createdAt = new Date().toISOString();
  return [
    {
      id: `eval_${simulationId}_memory_recall`,
      simulationId,
      kind: "memory_recall",
      label: "최근 핵심 기억 회수",
      query: "최근 장면의 약속, 단서, 관계 변화를 Context Pack에서 회수한다.",
      expectedSignals: ["memory", "promise", "continuity"],
      source: "system",
      createdAt
    },
    {
      id: `eval_${simulationId}_reset_continuity`,
      simulationId,
      kind: "reset_continuity",
      label: "세션 초기화 연속성",
      query: "세션 초기화 이후 handoff와 Context Pack이 핵심 상태를 유지하는지 확인한다.",
      expectedSignals: ["handoff", "continuity", "context"],
      source: "system",
      createdAt
    },
    {
      id: `eval_${simulationId}_image_quality`,
      simulationId,
      kind: "image_quality",
      label: "이미지 품질 피드백",
      query: "대표 생성 이미지가 장면/캐릭터 단서에 맞는지 피드백으로 확인한다.",
      expectedSignals: ["liked", "neutral", "rejected"],
      source: "system",
      createdAt
    }
  ];
}

function normalizeDraftImageScenePresetNodes(
  nodes: ImageSceneTagPresetNode[] | undefined,
  updatedAt: string,
  seenIds = new Set<string>(),
  path = "child"
): ImageSceneTagPresetNode[] {
  return (nodes ?? []).map((node, index) => ({
    ...node,
    id: reserveImageScenePresetId(
      node.id && !node.id.startsWith("draft_") ? node.id : undefined,
      `scene_preset_child_${path}_${index + 1}`,
      seenIds
    ),
    keyword: node.keyword.trim() || `scene-${index + 1}`,
    tags: node.tags.map((tag) => tag.trim()).filter(Boolean),
    note: node.note.trim(),
    enabled: node.enabled,
    priority: Math.min(120, Math.max(0, Number(node.priority) || 70)),
    updatedAt,
    children: normalizeDraftImageScenePresetNodes(node.children, updatedAt, seenIds, `${path}_${index + 1}`)
  }));
}

function reserveImageScenePresetId(candidateId: string | undefined, fallbackPrefix: string, seenIds: Set<string>): string {
  let id = candidateId?.trim() || `${fallbackPrefix}_${Date.now().toString(36)}`;
  let suffix = 1;
  while (seenIds.has(id)) {
    id = `${fallbackPrefix}_${Date.now().toString(36)}_${suffix}`;
    suffix += 1;
  }
  seenIds.add(id);
  return id;
}

export function createStateFromDraft(draft: SimulationDraft): AppState {
  const simulationId = `sim_${Date.now().toString(36)}`;
  const sessionId = `session_${Date.now().toString(36)}`;
  const progressRunId = createDefaultProgressRunId(sessionId);
  const createdAt = new Date().toISOString();
  const draftCharacters =
    draft.characters.length > 0
      ? draft.characters
      : [
          {
            id: "draft_character_id",
            name: draft.characterName.trim() || "Main Character",
            role: draft.characterRole || "Simulation lead",
            summary: draft.characterSummary,
            relationship: draft.characterRelationship || "플레이어와 첫 장면에서 만남",
            currentMood: draft.characterMood || "상황을 살피는 중",
            visualPrompt: draft.visualPrompt,
            negativeVisualPrompt: draft.negativeVisualPrompt,
            defaultOutfitPrompt: draft.defaultOutfitPrompt ?? defaultOutfitPrompt,
            outfitPrompts: defaultOutfitPrompts,
            expressionPrompts: defaultExpressionPrompts,
            defaultSafetyLevel: "safe" as const
          }
        ];
  const characterIdMap = new Map(draftCharacters.map((character, index) => [character.id, index === 0 ? `char_${Date.now().toString(36)}` : `char_${Date.now().toString(36)}_${index}`]));
  const mainCharacter = draftCharacters[0];
  const mainCharacterId = characterIdMap.get(mainCharacter.id) ?? `char_${Date.now().toString(36)}`;
  const characterName = mainCharacter.name.trim() || draft.characterName.trim() || "Main Character";
  const openingContent =
    draft.startSituationPrompt.trim() ||
    `${characterName}가 첫 장면에 들어옵니다. 시뮬레이션을 시작할 준비가 되었습니다.`;
  const simulation: Simulation = {
    id: simulationId,
    ownerId: "local_user",
    title: draft.title.trim() || "새 시뮬레이션",
    description: draft.description.trim() || "사용자 제작 AI 시뮬레이션",
    promptMode: draft.promptMode,
    contentRating: draft.contentRating,
    activeSessionId: sessionId,
    realtimeImageEnabled: draft.realtimeImageEnabled,
    defaultChatModelProfile:
      draft.promptMode === "basic"
        ? "balanced-agent"
        : draft.promptMode === "one_on_one"
          ? "one-on-one-agent"
          : draft.promptMode === "simulation"
            ? "simulation-agent"
            : "custom-agent",
    createdAt,
    updatedAt: createdAt
  };
  // safety_policy modules are kept in storage at every rating; the turn paths already filter them out
  // at read time under adult mode, so dropping them at creation only lost creator-authored content.
  const sourceModules: PromptModule[] =
    draft.modules.length > 0
      ? draft.modules
      : [
          {
            id: "module_main",
            simulationId,
            kind: "main_prompt",
            title: "메인 시뮬레이션 규칙",
            body: draft.mainPrompt,
            enabled: true,
            priority: 100,
            activationTags: ["core", "always"],
            tokenPolicy: "always",
            version: 1,
            updatedAt: createdAt
          },
          {
            id: "module_character",
            simulationId,
            parentId: "module_main",
            kind: "character_prompt",
            title: `캐릭터: ${characterName}`,
            body: draft.characterSummary,
            enabled: true,
            priority: 82,
            activationTags: ["character", characterName.toLowerCase()].filter(Boolean),
            characterId: mainCharacterId,
            tokenPolicy: "rag",
            version: 1,
            updatedAt: createdAt
          },
          {
            id: "module_world",
            simulationId,
            parentId: "module_main",
            kind: "world_lore",
            title: "세계관",
            body: draft.worldLore,
            enabled: true,
            priority: 72,
            activationTags: ["world", "lore"],
            tokenPolicy: "rag",
            version: 1,
            updatedAt: createdAt
          },
        ];
  // Per-simulation unique module ids. Previously the main module was always the constant "module_main", so two
  // simulations (e.g. a copy and its source) shared the same bare id. Any id-based attribution that does not also
  // compare simulationId (e.g. NeuralMap's bare `nodeId === module.id` checks) could then cross-link them. Deriving
  // the ids from the already-unique simulationId keeps each simulation's modules fully separated.
  const moduleIdSuffix = simulationId.replace(/^sim_/u, "") || Date.now().toString(36);
  const moduleIdMap = new Map(sourceModules.map((module, index) => [module.id, index === 0 ? `module_main_${moduleIdSuffix}` : `module_${moduleIdSuffix}_${index}`]));
  const mainModuleId = moduleIdMap.get(sourceModules[0]?.id ?? "") ?? `module_main_${moduleIdSuffix}`;
  const characterModuleId = sourceModules[1] ? moduleIdMap.get(sourceModules[1].id) : undefined;
  const modules: PromptModule[] = sourceModules.map((module) => ({
    ...module,
    id: moduleIdMap.get(module.id) ?? module.id,
    simulationId,
    parentId: module.parentId ? moduleIdMap.get(module.parentId) : undefined,
    characterId: module.characterId ? characterIdMap.get(module.characterId) : undefined,
    updatedAt: createdAt
  }));
  const characters: Character[] = draftCharacters.map((character) => ({
    id: characterIdMap.get(character.id) ?? mainCharacterId,
    simulationId,
    name: character.name || "Main Character",
    role: character.role || "Simulation lead",
    summary: character.summary,
    relationship: character.relationship || "플레이어와 첫 장면에서 만남",
    currentMood: character.currentMood || "상황을 살피는 중"
  }));
  const visualProfiles: CharacterVisualProfile[] = draftCharacters.flatMap((character, index) => {
    const mappedCharacterId = characterIdMap.get(character.id) ?? mainCharacterId;
    return [
      {
        id: `visual_${moduleIdSuffix}_${index}`,
        simulationId,
        characterId: mappedCharacterId,
        displayName: character.name || "Character",
        positivePrompt: character.visualPrompt,
        negativePrompt: character.negativeVisualPrompt || "low quality, bad anatomy, blurry, watermark",
        defaultOutfitPrompt: character.defaultOutfitPrompt ?? defaultOutfitPrompt,
        outfitPrompts: character.outfitPrompts ?? defaultOutfitPrompts,
        expressionPrompts: character.expressionPrompts ?? defaultExpressionPrompts,
        referenceImageAssetIds: [],
        defaultSafetyLevel: character.defaultSafetyLevel
      }
    ];
  });
  const imageProfile: ImageGenerationProfile = {
    ...seedState.imageProfile,
    ...draft.imageProfile,
    id: `img_profile_${moduleIdSuffix}`,
    simulationId,
    safetyLevel: draft.imageProfile.safetyLevel
  };
  const scenePresetSeenIds = new Set<string>();
  const imageScenePresets: ImageSceneTagPreset[] = (draft.imageScenePresets ?? []).map((preset, index) => ({
    ...preset,
    id: reserveImageScenePresetId(
      preset.id && !preset.id.startsWith("draft_") ? preset.id : undefined,
      `scene_preset_${index + 1}`,
      scenePresetSeenIds
    ),
    simulationId,
    keyword: preset.keyword.trim() || `scene-${index + 1}`,
    tags: preset.tags.map((tag) => tag.trim()).filter(Boolean),
    note: preset.note.trim(),
    enabled: preset.enabled,
    priority: Math.min(120, Math.max(0, Number(preset.priority) || 70)),
    updatedAt: createdAt,
    children: normalizeDraftImageScenePresetNodes(preset.children, createdAt, scenePresetSeenIds, `${index + 1}`)
  }));

  return hydrateState({
    ...seedState,
    simulation,
    security: createSecuritySettings(simulation),
    activeProgressRunId: progressRunId,
    progressRuns: [],
    modules,
    characters,
    visualProfiles,
    imageScenePresets,
    imageProfile,
    userPersona: {
      ...defaultUserPersona,
      updatedAt: createdAt
    },
    messages: [
      {
        id: "msg_welcome",
        simulationId,
        sessionId,
        role: "assistant",
        content: openingContent,
        createdAt,
        referencedNodeIds: [mainModuleId, ...(characterModuleId ? [characterModuleId] : [])],
        imageAssetIds: []
      }
    ],
    memoryEvents: [],
    contextPacks: [],
    handoffs: [],
    continuityChecks: [],
    promptModuleUsages: [],
    sidecarTraces: [],
    turnTraces: [],
    evaluationScenarios: createDefaultEvaluationScenarios(simulationId),
    auditLog: [],
    redactionQueue: [],
    imageAssets: [],
    imageJobs: [],
    neuralMap: draft.neuralMap,
    relationshipMap: draft.relationshipMap ?? defaultRelationshipMapSettings,
    llm: toShareableLlmSettings(draft.llm),
    novelAi: toShareableNovelAiSettings(draft.novelAi),
    selectedModuleId: mainModuleId,
    selectedContextPackId: undefined
  });
}
