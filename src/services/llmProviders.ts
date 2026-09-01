import type { LlmApiSettings, LlmProvider } from "../types";

// Single source of truth for every LLM backend DynamicChat can talk to.
//
// Before this file, provider knowledge was split three ways: a display-only array in App.tsx, a hand-written
// if-chain in llmClient.requestProviderTextImmediate, and another in apiValidation. Adding a backend meant
// editing all three and hoping the quirks matched. Presets now carry BOTH the UI metadata and the runtime
// capabilities (wire format, JSON mode, parameter limits, extra headers/body, CORS routing), so the request
// builder can stay generic and a new backend is one entry.

/** How the request/response is shaped on the wire. */
export type LlmTransport = "openai" | "gemini" | "claude" | "cli" | "mock" | "ollama";

/** Whether the backend can be asked for guaranteed JSON, and how. */
export type LlmJsonMode = "json_object" | "none";

/** Which body field carries the output-token ceiling. */
export type LlmTokenParam = "max_tokens" | "max_completion_tokens";

export interface LlmProviderPreset {
  value: LlmProvider;
  label: string;
  /** Grouping for the provider dropdown. */
  group: "mock" | "hosted" | "open" | "local" | "cli";
  /** One-line Korean hint shown under the provider select. */
  hint?: string;
  baseUrl: string;
  defaultModel: string;
  models: string[];
  keyPlaceholder: string;
  /** Show the editable base-URL field (self-hosted / region-specific endpoints). */
  advancedBaseUrl: boolean;
  transport: LlmTransport;
  /**
   * Route the call through the DynamicChat API server instead of calling the vendor from the page.
   * Most inference vendors serve no CORS headers for browser origins, so a direct fetch fails before it
   * is ever sent. The server proxy forwards the request (streaming included) and returns the response.
   */
  requiresProxy: boolean;
  jsonMode: LlmJsonMode;
  /** Hard ceiling for the temperature slider — Anthropic rejects >1, several open backends reject >2. */
  maxTemperature: number;
  /** Some reasoning models 400 on any explicit temperature. */
  sendTemperature: boolean;
  tokenParam: LlmTokenParam;
  /**
   * Usable context window in tokens, used to size the prompt so it fits.
   *
   * Local servers are the reason this exists: Ollama defaults every model to a 4096-token context no matter
   * what the weights support, and the annotation prompt alone is larger than that. Without a budget the turn
   * fails with a raw upstream 400 that the user has no way to act on.
   */
  contextTokens: number;
  /**
   * Whether an API key is required before the backend can be used. Local servers (Ollama, LM Studio, a
   * self-hosted vLLM) accept unauthenticated requests, and the runtime's "no key -> fall back to mock" gate
   * made them silently unusable: the turn returned a fallback and no request was ever sent.
   */
  requiresApiKey: boolean;
  /** Extra request headers (OpenRouter attribution, Anthropic browser opt-in, …). */
  extraHeaders?: Record<string, string>;
  /** Extra top-level body fields — thinking toggles and the like. Never a new code branch. */
  extraBody?: Record<string, unknown>;
  /**
   * Models that spend their whole budget on chain-of-thought before emitting an answer token.
   * Used to widen the key-verification probe and to explain an empty completion instead of retrying it.
   */
  reasoningModelPattern?: RegExp;
}

const OPEN_MODEL_DEFAULTS = {
  transport: "openai",
  requiresProxy: true,
  requiresApiKey: true,
  jsonMode: "json_object",
  maxTemperature: 2,
  sendTemperature: true,
  tokenParam: "max_tokens",
  // Every hosted open-model endpoint in this list serves at least 128k; the conservative floor keeps the
  // prompt well inside it without needing a per-model table.
  contextTokens: 128_000
} as const satisfies Partial<LlmProviderPreset>;

export const LLM_PROVIDER_PRESETS: LlmProviderPreset[] = [
  {
    value: "mock",
    label: "Mock",
    group: "mock",
    hint: "외부 호출 없이 시뮬레이션 흐름만 확인합니다.",
    baseUrl: "",
    defaultModel: "mock-simulation-agent",
    models: ["mock-simulation-agent"],
    keyPlaceholder: "API 키 필요 없음",
    advancedBaseUrl: false,
    transport: "mock",
    requiresProxy: false,
    jsonMode: "none",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: false,
    contextTokens: 128_000
  },

  // ─── Hosted commercial ────────────────────────────────────────────────────────────────────────
  {
    value: "codex",
    label: "OpenAI",
    group: "hosted",
    baseUrl: "https://api.openai.com/v1",
    defaultModel: "gpt-5-mini",
    models: ["gpt-5", "gpt-5-mini", "o4-mini", "gpt-4.1", "gpt-4.1-mini"],
    keyPlaceholder: "sk-...",
    advancedBaseUrl: false,
    transport: "openai",
    requiresProxy: false,
    jsonMode: "json_object",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: true,
    contextTokens: 128_000,
    // o-series / gpt-5 reasoning models reject max_tokens and a non-default temperature.
    reasoningModelPattern: /^(?:o\d|gpt-5)/iu
  },
  {
    value: "gemini",
    label: "Gemini",
    group: "hosted",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta",
    defaultModel: "gemini-3.7-flash",
    models: [
      "gemini-3.7-flash",
      "gemini-3.7-pro",
      "gemini-3.1-pro-preview",
      "gemini-3-pro-preview",
      "gemini-3-flash-preview",
      "gemini-3.1-flash-lite-preview",
      "gemini-2.5-pro",
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
      "gemini-2.0-flash",
      "gemini-flash-latest"
    ],
    keyPlaceholder: "AIza...",
    advancedBaseUrl: false,
    transport: "gemini",
    requiresProxy: false,
    jsonMode: "json_object",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: true,
    contextTokens: 1_000_000
  },
  {
    value: "claude",
    label: "Claude",
    group: "hosted",
    baseUrl: "https://api.anthropic.com/v1",
    defaultModel: "claude-sonnet-4-6",
    models: [
      "claude-opus-4-7",
      "claude-sonnet-4-6",
      "claude-haiku-4-5-20251001",
      "claude-3-7-sonnet-latest",
      "claude-3-5-haiku-latest"
    ],
    keyPlaceholder: "sk-ant-...",
    advancedBaseUrl: false,
    transport: "claude",
    requiresProxy: false,
    jsonMode: "none",
    // Anthropic rejects temperature > 1. The UI slider reaches 1.5, so clamp at the request boundary.
    maxTemperature: 1,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: true,
    contextTokens: 200_000,
    // Without this Anthropic refuses requests whose Origin is a browser page.
    extraHeaders: { "anthropic-dangerous-direct-browser-access": "true" }
  },

  // ─── Open-weight / open-source model providers ────────────────────────────────────────────────
  // Cheaper per token than the hosted commercial tier and far less likely to refuse in-fiction adult
  // or violent narrative, which is what makes them usable for long simulation runs.
  {
    value: "deepseek",
    label: "DeepSeek",
    group: "open",
    hint: "장문 서술과 상황 추론에 강하고 단가가 낮습니다. 시뮬레이션 본문 모델로 권장.",
    baseUrl: "https://api.deepseek.com/v1",
    defaultModel: "deepseek-v4-flash",
    models: ["deepseek-v4-flash", "deepseek-v4-pro", "custom"],
    keyPlaceholder: "sk-...",
    advancedBaseUrl: true,
    ...OPEN_MODEL_DEFAULTS,
    // V4 made thinking a request parameter rather than a model name. DynamicChat needs one JSON object per
    // call, not a chain of thought, so thinking is off by default — leaving it on burns the whole output
    // budget on reasoning tokens and returns an empty `content`.
    extraBody: { thinking: { type: "disabled" } }
  },
  {
    value: "moonshot",
    label: "Kimi (Moonshot)",
    group: "open",
    hint: "1M 컨텍스트. 긴 진행 기록을 통째로 유지해야 하는 시뮬레이션에 적합.",
    baseUrl: "https://api.moonshot.ai/v1",
    defaultModel: "kimi-k2.6",
    models: ["kimi-k3", "kimi-k2.6", "kimi-k2.5", "kimi-k2.7-code", "custom"],
    keyPlaceholder: "sk-...",
    advancedBaseUrl: true,
    ...OPEN_MODEL_DEFAULTS
  },
  {
    value: "dashscope",
    label: "Qwen (DashScope)",
    group: "open",
    hint: "국제 엔드포인트 기본값. 중국 리전은 base URL을 직접 바꾸세요.",
    baseUrl: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    defaultModel: "qwen-plus",
    models: ["qwen3-max", "qwen-max", "qwen-plus", "qwen-turbo", "custom"],
    keyPlaceholder: "sk-...",
    advancedBaseUrl: true,
    ...OPEN_MODEL_DEFAULTS,
    // Qwen thinking models stream reasoning deltas and require stream=true when thinking is on. This app
    // wants one JSON object, so thinking stays off.
    extraBody: { enable_thinking: false }
  },
  {
    value: "zhipu",
    label: "GLM (Z.ai / Zhipu)",
    group: "open",
    hint: "한국어 서술 품질이 안정적입니다. 중국 리전은 open.bigmodel.cn 으로 변경.",
    baseUrl: "https://api.z.ai/api/paas/v4",
    defaultModel: "glm-4.6",
    models: ["glm-4.6", "glm-4.5", "custom"],
    keyPlaceholder: "...",
    advancedBaseUrl: true,
    ...OPEN_MODEL_DEFAULTS
  },
  {
    value: "openrouter",
    label: "OpenRouter",
    group: "open",
    hint: "한 키로 여러 오픈 모델을 씁니다. 모델 id는 제공자/모델 형식으로 직접 입력하세요.",
    baseUrl: "https://openrouter.ai/api/v1",
    defaultModel: "deepseek/deepseek-v4-flash",
    models: [
      "deepseek/deepseek-v4-flash",
      "deepseek/deepseek-v4-pro",
      "moonshotai/kimi-k2.6",
      "qwen/qwen3-max",
      "z-ai/glm-4.6",
      "meta-llama/llama-3.3-70b-instruct",
      "mistralai/mistral-large",
      "custom"
    ],
    keyPlaceholder: "sk-or-...",
    advancedBaseUrl: true,
    ...OPEN_MODEL_DEFAULTS,
    // Attribution headers; OpenRouter shows them on the app leaderboard and uses them for abuse triage.
    extraHeaders: { "HTTP-Referer": "https://github.com/kimasill/DynamicChat", "X-Title": "DynamicChat" }
  },
  {
    value: "groq",
    label: "Groq",
    group: "open",
    hint: "오픈 웨이트 모델을 매우 빠르게 서빙합니다. 이미지 태그 보조 모델로 적합.",
    baseUrl: "https://api.groq.com/openai/v1",
    defaultModel: "llama-3.3-70b-versatile",
    models: ["llama-3.3-70b-versatile", "qwen-2.5-32b", "custom"],
    keyPlaceholder: "gsk_...",
    advancedBaseUrl: true,
    ...OPEN_MODEL_DEFAULTS
  },
  {
    value: "together",
    label: "Together AI",
    group: "open",
    baseUrl: "https://api.together.xyz/v1",
    defaultModel: "deepseek-ai/DeepSeek-V3",
    models: ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct-Turbo", "custom"],
    keyPlaceholder: "...",
    advancedBaseUrl: true,
    ...OPEN_MODEL_DEFAULTS
  },

  // ─── Local / self-hosted ──────────────────────────────────────────────────────────────────────
  // No API key, no per-token cost, no vendor policy at all. Local servers usually allow the page origin,
  // so these do NOT need the proxy.
  {
    value: "ollama",
    label: "Ollama (로컬)",
    group: "local",
    hint: "ollama serve 실행 후 사용. 키 불필요. 설치된 모델을 자동으로 불러옵니다.",
    baseUrl: "http://127.0.0.1:11434/v1",
    // Empty, not "custom": that string is the UI's free-text sentinel, and shipping it as the default meant
    // an untouched field POSTed {"model":"custom"} and got an opaque upstream 404 instead of the
    // "모델 이름이 비어 있습니다" guidance the empty-model guard exists to give.
    defaultModel: "",
    models: ["custom"],
    keyPlaceholder: "로컬 서버 — 키 불필요",
    advancedBaseUrl: true,
    // Ollama's OpenAI-compatible endpoint silently IGNORES num_ctx and serves every model with a
    // 4096-token context, which the annotation prompt cannot fit. Its native /api/chat endpoint honours it,
    // so the local preset speaks that instead and passes the configured context window through.
    transport: "ollama",
    requiresProxy: false,
    jsonMode: "json_object",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: false,
    // Raised from Ollama's 4096 default because the native transport sets num_ctx explicitly. Lower it in
    // the settings if the machine cannot hold the KV cache for a longer window.
    contextTokens: 16_384
  },
  {
    value: "lmstudio",
    label: "LM Studio (로컬)",
    group: "local",
    hint: "LM Studio 로컬 서버(기본 1234 포트).",
    baseUrl: "http://127.0.0.1:1234/v1",
    defaultModel: "",
    models: ["custom"],
    keyPlaceholder: "로컬 서버 — 키 불필요",
    advancedBaseUrl: true,
    transport: "openai",
    requiresProxy: false,
    jsonMode: "none",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: false,
    contextTokens: 8_192
  },
  {
    value: "openai_compatible",
    label: "직접 입력 (OpenAI 호환)",
    group: "local",
    hint: "vLLM, llama.cpp, KoboldCpp 등 OpenAI 호환 엔드포인트.",
    baseUrl: "http://127.0.0.1:8000/v1",
    defaultModel: "local-model",
    models: ["local-model", "custom"],
    keyPlaceholder: "provider key",
    advancedBaseUrl: true,
    transport: "openai",
    requiresProxy: false,
    jsonMode: "none",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: false,
    contextTokens: 8_192
  },

  // ─── Subscription CLI bridges ─────────────────────────────────────────────────────────────────
  {
    value: "claude_cli",
    label: "Claude 구독 CLI",
    group: "cli",
    baseUrl: "",
    defaultModel: "sonnet",
    models: ["sonnet", "opus", "haiku", "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    keyPlaceholder: "구독 CLI — API 키 불필요",
    advancedBaseUrl: false,
    transport: "cli",
    requiresProxy: false,
    jsonMode: "none",
    maxTemperature: 1,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: false,
    contextTokens: 200_000
  },
  {
    value: "codex_cli",
    label: "Codex 구독 CLI",
    group: "cli",
    baseUrl: "",
    defaultModel: "gpt-5-codex",
    models: ["gpt-5-codex", "gpt-5", "gpt-5-mini", "o4-mini"],
    keyPlaceholder: "구독 CLI — API 키 불필요",
    advancedBaseUrl: false,
    transport: "cli",
    requiresProxy: false,
    jsonMode: "none",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: false,
    contextTokens: 128_000
  },
  {
    value: "gemini_cli",
    label: "Gemini 구독 CLI",
    group: "cli",
    baseUrl: "",
    defaultModel: "gemini-3.7-flash",
    models: [
      "gemini-3.7-flash",
      "gemini-3.7-pro",
      "gemini-3.1-pro-preview",
      "gemini-3-pro-preview",
      "gemini-3-flash-preview",
      "gemini-3.1-flash-lite-preview",
      "gemini-2.5-pro",
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
      "gemini-2.0-flash"
    ],
    keyPlaceholder: "구독 CLI — API 키 불필요",
    advancedBaseUrl: false,
    transport: "cli",
    requiresProxy: false,
    jsonMode: "none",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: false,
    contextTokens: 1_000_000
  },
  {
    value: "antigravity_cli",
    label: "Antigravity CLI (agy)",
    group: "cli",
    baseUrl: "",
    defaultModel: "gemini-3.7-flash-high",
    models: [
      "gemini-3.7-flash-high",
      "gemini-3.7-flash-medium",
      "gemini-3.7-flash-low",
      "gemini-3.6-flash-high",
      "gemini-3.6-flash-medium",
      "gemini-3.6-flash-low",
      "gemini-3.5-flash-high",
      "gemini-3.5-flash-medium",
      "gemini-3.5-flash-low",
      "gemini-3.1-pro-high",
      "gemini-3.1-pro-low",
      "claude-sonnet-4-6",
      "claude-opus-4-6-thinking",
      "gpt-oss-120b-medium"
    ],
    keyPlaceholder: "구독 CLI — API 키 불필요 (agy 로그인)",
    advancedBaseUrl: false,
    transport: "cli",
    requiresProxy: false,
    jsonMode: "none",
    maxTemperature: 2,
    sendTemperature: true,
    tokenParam: "max_tokens",
    requiresApiKey: false,
    contextTokens: 1_000_000
  }
];

export const LLM_PROVIDER_GROUP_LABELS: Record<LlmProviderPreset["group"], string> = {
  mock: "테스트",
  hosted: "상용 API",
  open: "오픈 모델 API",
  local: "로컬 / 직접 입력",
  cli: "구독 CLI"
};

const PRESET_BY_PROVIDER = new Map(LLM_PROVIDER_PRESETS.map((preset) => [preset.value, preset] as const));

export function getLlmProviderPreset(provider: LlmProvider): LlmProviderPreset {
  return PRESET_BY_PROVIDER.get(provider) ?? LLM_PROVIDER_PRESETS[0];
}

/** True when the configured model is one whose whole budget can go to hidden reasoning tokens. */
export function isReasoningModel(settings: Pick<LlmApiSettings, "provider" | "model">): boolean {
  const pattern = getLlmProviderPreset(settings.provider).reasoningModelPattern;
  return Boolean(pattern && pattern.test(settings.model.trim()));
}

/** Clamp a user-configured temperature into what the selected backend actually accepts. */
export function clampProviderTemperature(provider: LlmProvider, temperature: number): number {
  const { maxTemperature } = getLlmProviderPreset(provider);
  if (!Number.isFinite(temperature)) {
    return 0.8;
  }
  return Math.min(Math.max(temperature, 0), maxTemperature);
}

/**
 * True when the annotation/image-cue passes should ask the backend for guaranteed JSON.
 * Those passes are parsed as strict JSON; open-weight models are much less disciplined than the hosted
 * tier without it, and a malformed response degrades into cues with missing character prompts.
 */
export function supportsJsonResponseFormat(provider: LlmProvider): boolean {
  return getLlmProviderPreset(provider).jsonMode === "json_object";
}
