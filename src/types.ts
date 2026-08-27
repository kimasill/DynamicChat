export type Id = string;

export type PromptModuleKind =
  | "main_prompt"
  | "sub_prompt"
  | "character_prompt"
  | "world_lore"
  | "scene_rule"
  | "image_prompt_profile"
  | "safety_policy"
  | "style_guide";

export type TokenPolicy = "always" | "rag" | "manual" | "disabled";

export type ContentRating = "general" | "adult_19";

export type ImageSafetyLevel = "safe" | "sensitive" | "suggestive" | "explicit";

export type ImageTriggerMode = "stored_only" | "realtime_auto" | "realtime_confirm" | "manual";

export type ImageGenerationCadence = "sparse" | "balanced" | "rich" | "paragraph" | "image_progression";

export type ImageJobStatus = "queued" | "planning" | "generating" | "completed" | "failed" | "canceled";

export type LlmProvider =
  | "mock"
  | "codex"
  | "gemini"
  | "claude"
  | "openai_compatible"
  | "claude_cli"
  | "codex_cli"
  | "gemini_cli"
  | "antigravity_cli";

export const CLI_AGENT_LLM_PROVIDERS: readonly LlmProvider[] = ["claude_cli", "codex_cli", "gemini_cli", "antigravity_cli"];

export function isCliAgentLlmProvider(provider: LlmProvider): boolean {
  return CLI_AGENT_LLM_PROVIDERS.includes(provider);
}

// The kind is the bridge agent id, NOT the binary name. `antigravity` → the `agy` CLI (Google's
// terminal coding agent that replaced the individual gemini CLI); the command lives server-side.
export function cliAgentKindForProvider(provider: LlmProvider): "claude" | "codex" | "gemini" | "antigravity" | undefined {
  switch (provider) {
    case "claude_cli":
      return "claude";
    case "codex_cli":
      return "codex";
    case "gemini_cli":
      return "gemini";
    case "antigravity_cli":
      return "antigravity";
    default:
      return undefined;
  }
}

export type ApiRegistrationStatus = "idle" | "verifying" | "registered" | "failed";

export type NovelAiRequestMode = "mock" | "direct" | "proxy";

export type NovelAiModelPreset = "NAID5F" | "NAID5C" | "NAID5" | "NAID4.5F" | "NAID4.5C" | "NAID4.0F" | "NAID4.0C" | "NAID3";

export type NovelAiNoiseSchedule = "karras" | "native" | "exponential" | "polyexponential";

export type NovelAiAutomationTermination = "unlimited" | "timer" | "count";

export type DeploymentEnvironment = "local" | "development" | "production";

export type SimulationPromptMode = "basic" | "one_on_one" | "simulation" | "custom";

export interface TenantScope {
  ownerId: Id;
  workspaceId: Id;
  projectId: Id;
  environment: DeploymentEnvironment;
}

export interface SecuritySettings {
  localUserId: Id;
  scope: TenantScope;
  secretStorage: "server_dev_store" | "external_vault";
  browserSecretCacheEnabled: boolean;
  warning: string;
  warningAcceptedAt?: string;
}

export type AuditAction =
  | "context_retrieved"
  | "generation_job_created"
  | "generation_job_completed"
  | "api_secret_verified"
  | "api_secret_stored"
  | "memory_redacted"
  | "prompt_module_redacted"
  | "image_asset_deleted"
  | "asset_accessed"
  | "backup_created"
  | "state_saved";

export type AuditResourceType =
  | "context_pack"
  | "image_job"
  | "api_secret"
  | "memory_event"
  | "prompt_module"
  | "image_asset"
  | "simulation"
  | "backup";

export interface AuditEvent {
  id: Id;
  simulationId: Id;
  ownerId: Id;
  scope: TenantScope;
  action: AuditAction;
  resourceType: AuditResourceType;
  resourceId: Id;
  metadata: Record<string, unknown>;
  createdAt: string;
}

export type RedactionTargetType = "memory_event" | "prompt_module" | "image_asset" | "neuralmap_record";

export interface RedactionRequest {
  id: Id;
  simulationId: Id;
  ownerId: Id;
  scope: TenantScope;
  targetType: RedactionTargetType;
  targetId: Id;
  reason: string;
  neuralMapNodeIds: Id[];
  status: "queued" | "applied" | "failed";
  createdAt: string;
  completedAt?: string;
  error?: string;
}

export interface Simulation {
  id: Id;
  ownerId: Id;
  title: string;
  description: string;
  premise?: string;
  promptMode: SimulationPromptMode;
  contentRating: ContentRating;
  activeSessionId: Id;
  realtimeImageEnabled: boolean;
  defaultChatModelProfile: string;
  createdAt: string;
  updatedAt: string;
}

export interface PromptModule {
  id: Id;
  simulationId: Id;
  parentId?: Id;
  kind: PromptModuleKind;
  title: string;
  name?: string;
  body: string;
  enabled: boolean;
  priority: number;
  activationTags: string[];
  characterId?: Id;
  tokenPolicy: TokenPolicy;
  version: number;
  updatedAt: string;
}

export interface Character {
  id: Id;
  simulationId?: Id;
  name: string;
  role: string;
  summary: string;
  relationship: string;
  currentMood: string;
  traits?: string[];
  description?: string;
}

export interface SimulationCharacterDraft {
  id: Id;
  name: string;
  role: string;
  summary: string;
  relationship: string;
  currentMood: string;
  visualPrompt: string;
  negativeVisualPrompt: string;
  defaultOutfitPrompt: string;
  outfitPrompts: Record<string, string>;
  expressionPrompts: Record<string, string>;
  defaultSafetyLevel: ImageSafetyLevel;
}

export interface CharacterVisualProfile {
  id: Id;
  simulationId: Id;
  characterId: Id;
  displayName: string;
  positivePrompt: string;
  negativePrompt: string;
  defaultOutfitPrompt: string;
  outfitPrompts: Record<string, string>;
  expressionPrompts: Record<string, string>;
  referenceImageAssetIds: Id[];
  defaultSafetyLevel: ImageSafetyLevel;
}

export interface ImageScenePresetExampleFile {
  id: Id;
  label: string;
  prompts: string[];
}

export interface ImageSceneTagPresetNode {
  id: Id;
  keyword: string;
  tags: string[];
  note: string;
  exampleFiles?: ImageScenePresetExampleFile[];
  enabled: boolean;
  priority: number;
  updatedAt: string;
  children: ImageSceneTagPresetNode[];
}

export interface ImageSceneTagPreset extends ImageSceneTagPresetNode {
  simulationId: Id;
}

export interface ImageGenerationProfile {
  id: Id;
  simulationId: Id;
  enabled: boolean;
  provider: "novelai";
  model: string;
  width: number;
  height: number;
  steps: number;
  promptGuidance: number;
  countMin: number;
  countMax: number;
  qualityPrompt: string;
  stylePrompt: string;
  artistPrompt: string;
  negativePrompt: string;
  safetyLevel: ImageSafetyLevel;
  userRules: string;
  triggerMode: ImageTriggerMode;
  generationCadence: ImageGenerationCadence;
  cooldownTurns: number;
}

export interface ChatMessage {
  id: Id;
  simulationId: Id;
  sessionId: Id;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
  referencedNodeIds: Id[];
  imageAssetIds: Id[];
}

export interface SimulationProgressRun {
  id: Id;
  simulationId: Id;
  title: string;
  activeSessionId: Id;
  sessionIds: Id[];
  messages: ChatMessage[];
  memoryEvents: MemoryEvent[];
  contextPacks: ContextPack[];
  handoffs: SessionHandoff[];
  continuityChecks: ContinuityCheck[];
  promptModuleUsages: PromptModuleUsage[];
  sidecarTraces: SidecarTrace[];
  turnTraces: TurnTrace[];
  imageAssets: ImageAsset[];
  imageJobs: ImageGenerationJob[];
  selectedContextPackId?: Id;
  createdAt: string;
  updatedAt: string;
}

export type UserPersonaSource = "custom" | "character";

export interface UserPersona {
  enabled: boolean;
  source: UserPersonaSource;
  characterId?: Id;
  name: string;
  role: string;
  background: string;
  goals: string;
  style: string;
  boundaries: string;
  updatedAt: string;
}

export interface MemoryEvent {
  id: Id;
  simulationId: Id;
  sessionId: Id;
  actorId?: Id;
  actorName?: string;
  content: string;
  importance: number;
  tags: string[];
  sourceTurnId?: Id;
  createdAt: string;
  neuralMapNodeId?: Id;
  metadata?: Record<string, unknown>;
}

export interface ContextEvidence {
  nodeId: Id;
  snippet: string;
  score: number;
  reason: string;
}

export interface ContextPack {
  id: Id;
  simulationId: Id;
  sessionId: Id;
  objective: string;
  tokenBudget: number;
  evidence: ContextEvidence[];
  sections?: Record<string, ContextEvidence[]>;
  moduleEvidence?: ContextEvidence[];
  decisions: string[];
  blockers: string[];
  createdAt: string;
  source: "mock" | "neuralmap";
}

export interface SessionHandoff {
  id: Id;
  simulationId: Id;
  previousSessionId: Id;
  nextSessionId: Id;
  summary: string;
  evidenceNodeIds: Id[];
  createdAt: string;
  source: "mock" | "neuralmap";
  error?: string;
}

export interface ContinuityFactCheck {
  label: string;
  expected: string;
  found: boolean;
  evidence: string;
}

export interface ContinuityCheck {
  id: Id;
  simulationId: Id;
  previousSessionId: Id;
  nextSessionId: Id;
  handoffId: Id;
  status: "passed" | "warning";
  checkedAt: string;
  facts: ContinuityFactCheck[];
  warnings: string[];
}

export interface PromptModuleUsage {
  id: Id;
  simulationId: Id;
  sessionId: Id;
  turnId: Id;
  moduleId: Id;
  moduleTitle: string;
  tokenPolicy: TokenPolicy;
  source: "always" | "manual" | "neuralmap" | "local";
  reason: string;
  score: number;
  createdAt: string;
}

export interface AssistantMemoryEventDraft {
  content: string;
  importance: number;
  tags: string[];
  actorId?: Id;
  actorName?: string;
  memoryKind?: "event" | "state" | "observation" | "belief" | "goal" | "relationship" | "open_thread" | "summary";
  eventType?: string;
  stateType?: string;
  stateValue?: string;
  targetId?: Id;
  observers?: Id[];
  confidence?: number;
}

export interface AssistantImageCueDraft {
  shouldGenerate: boolean;
  reason: string;
  characters: Id[];
  tags: string[];
  baseTags?: string[];
  characterPrompts?: ImageCueCharacterPrompt[];
  scene: string;
  suppressionReason?: string;
  visualContext?: string;
  label?: string;
  kind?: string;
  placement?: "before" | "after" | "inline";
  anchorText?: string;
  cueType?: string;
  priority?: number;
}

export interface AssistantSidecar {
  assistantText: string;
  memoryEvents: AssistantMemoryEventDraft[];
  imageCue: AssistantImageCueDraft;
  imageCues: AssistantImageCueDraft[];
}

export interface SidecarTrace {
  id: Id;
  simulationId: Id;
  sessionId: Id;
  turnId: Id;
  source: "llm" | "mock" | "fallback";
  status: "parsed" | "fallback" | "failed";
  errors: string[];
  requestPreview?: string;
  rawPreview?: string;
  createdAt: string;
}

export interface TurnTraceMetrics {
  tokenBudget: number;
  selectedModuleCount: number;
  selectedModuleTokenEstimate: number;
  contextEvidenceCount: number;
  contextTokenEstimate: number;
  ragTokenSavingsEstimate: number;
  llmLatencyMs: number;
  llmRequestMs: number;
  retrievalLatencyMs: number;
  memoryIngestMs: number;
  turnLatencyMs: number;
  memoryIngestCount: number;
  imageJobCount: number;
  imageAssetCount: number;
  imageEstimatedCost?: number;
}

export interface TurnTrace {
  id: Id;
  simulationId: Id;
  sessionId: Id;
  turnId: Id;
  userMessageId: Id;
  assistantMessageId: Id;
  contextPackId: Id;
  promptModuleUsageIds: Id[];
  sidecarTraceId: Id;
  memoryEventIds: Id[];
  imageCue: ImageCue;
  imageJobId?: Id;
  imageAssetIds: Id[];
  metrics: TurnTraceMetrics;
  createdAt: string;
}

export type EvaluationScenarioKind = "memory_recall" | "reset_continuity" | "image_quality";

export interface EvaluationScenario {
  id: Id;
  simulationId: Id;
  kind: EvaluationScenarioKind;
  label: string;
  query: string;
  expectedSignals: string[];
  source: "seed" | "user" | "system";
  createdAt: string;
}

export type ImageFeedbackRating = "liked" | "neutral" | "rejected";

export interface ImageAssetFeedback {
  rating: ImageFeedbackRating;
  note?: string;
  updatedAt: string;
}

export interface ImageCue {
  shouldGenerate: boolean;
  reason: string;
  characters: Id[];
  tags: string[];
  baseTags?: string[];
  characterPrompts?: ImageCueCharacterPrompt[];
  scene: string;
  suppressionReason?: string;
  visualContext?: string;
}

export interface ImageCueCharacterPrompt {
  characterId?: Id;
  prompt: string;
  negativePrompt?: string;
  center?: {
    x: number;
    y: number;
  };
}

export interface ImageAsset {
  id: Id;
  simulationId: Id;
  title: string;
  source: "stored" | "generated" | "fallback";
  prompt: string;
  negativePrompt: string;
  safetyLevel: ImageSafetyLevel;
  characterIds: Id[];
  tags: string[];
  createdAt: string;
  jobId?: Id;
  palette: [string, string, string];
  dataUrl?: string;
  mimeType?: string;
  objectKey?: string;
  providerMetadata?: Record<string, unknown>;
  reuseTags?: string[];
  representative?: boolean;
  feedback?: ImageAssetFeedback;
}

export interface ImageGenerationJob {
  id: Id;
  simulationId: Id;
  sessionId: Id;
  turnId: Id;
  status: ImageJobStatus;
  reason: string;
  prompt: string;
  negativePrompt: string;
  providerPayload: Record<string, unknown>;
  assetIds: Id[];
  contextNodeIds: Id[];
  createdAt: string;
  updatedAt?: string;
  completedAt?: string;
  error?: string;
  policyWarnings?: string[];
  representativeAssetId?: Id;
}

export interface NeuralMapSettings {
  baseUrl: string;
  enabled: boolean;
  tokenBudget: number;
}

export interface RelationshipStatusParameter {
  id: Id;
  title: string;
  rule: string;
  enabled: boolean;
  priority: number;
}

export interface RelationshipMapSettings {
  enabled: boolean;
  statusPrompt: string;
  parameters: RelationshipStatusParameter[];
  updatedAt: string;
}

// A user-authored value for a relationship status card. The system keeps auto-computing the underlying status from
// the profile/memory/parameters; when an override exists for the same node + status key, the override is shown instead
// until the user resets it.
export interface RelationshipStatusOverride {
  nodeId: Id;
  statusKey: string;
  title: string;
  value: string;
  updatedAt: string;
}

export interface LlmApiSettings {
  enabled: boolean;
  provider: LlmProvider;
  baseUrl: string;
  apiKey: string;
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  registrationStatus: ApiRegistrationStatus;
  verifiedAt?: string;
  verificationMessage?: string;
}

export interface NovelAiVibeTransferReference {
  id: Id;
  name: string;
  /** 전체 data URL (data:image/...;base64,...). NovelAI 요청 시 prefix를 제거하고 전송한다. */
  image: string;
  /** Reference strength (0~1). 참조 이미지의 영향력. */
  referenceStrength: number;
  /** Information extracted (0~1). 참조 이미지에서 추출하는 정보량. */
  informationExtracted: number;
  /** v4/v4.5에서 encode-vibe로 사전 인코딩한 vibe 데이터 (base64). 인코딩 전에는 비어 있다. */
  encodedVibe?: string;
  /** encodedVibe를 만들 때 사용한 모델 프리셋. 현재 프리셋과 다르면 재인코딩이 필요하다. */
  encodedModel?: NovelAiModelPreset;
  /** encodedVibe를 만들 때 사용한 information extracted 값. */
  encodedInformationExtracted?: number;
}

export interface NovelAiApiSettings {
  enabled: boolean;
  requestMode: NovelAiRequestMode;
  endpoint: string;
  apiKey: string;
  proxyUrl: string;
  accountLabel: string;
  roundRobinEnabled: boolean;
  modelPreset: NovelAiModelPreset;
  ucPreset: number;
  sampler: string;
  noiseSchedule: NovelAiNoiseSchedule;
  cfgRescale: number;
  varPlus: boolean;
  seedFixed: boolean;
  generationDelaySeconds: number;
  randomDelayEnabled: boolean;
  repeatCount: number;
  automationTermination: NovelAiAutomationTermination;
  timerMinutes: number;
  countLimit: number;
  registrationStatus: ApiRegistrationStatus;
  verifiedAt?: string;
  verificationMessage?: string;
  subscriptionTier?: string;
  seed?: number;
  vibeTransferEnabled: boolean;
  vibeTransferReferences: NovelAiVibeTransferReference[];
}

export interface AppState {
  simulation: Simulation;
  security: SecuritySettings;
  activeProgressRunId: Id;
  progressRuns: SimulationProgressRun[];
  modules: PromptModule[];
  characters: Character[];
  visualProfiles: CharacterVisualProfile[];
  imageScenePresets: ImageSceneTagPreset[];
  imageProfile: ImageGenerationProfile;
  userPersona: UserPersona;
  messages: ChatMessage[];
  memoryEvents: MemoryEvent[];
  contextPacks: ContextPack[];
  handoffs: SessionHandoff[];
  continuityChecks: ContinuityCheck[];
  promptModuleUsages: PromptModuleUsage[];
  sidecarTraces: SidecarTrace[];
  turnTraces: TurnTrace[];
  evaluationScenarios: EvaluationScenario[];
  auditLog: AuditEvent[];
  redactionQueue: RedactionRequest[];
  imageAssets: ImageAsset[];
  imageJobs: ImageGenerationJob[];
  neuralMap: NeuralMapSettings;
  relationshipMap: RelationshipMapSettings;
  relationshipStatusOverrides: RelationshipStatusOverride[];
  llm: LlmApiSettings;
  // Dedicated LLM for authoring image_cues / NovelAI tags, separated from the narrative LLM so the narrative model
  // is never polluted by tag rules and can use a stronger model while image tags use a cheaper one. When
  // `enabled` is false the image-cue call falls back to the main `llm` config.
  imageTagLlm: LlmApiSettings;
  novelAi: NovelAiApiSettings;
  selectedModuleId?: Id;
  selectedContextPackId?: Id;
  updatedAt?: string;
}

export interface TurnResult {
  userMessage: ChatMessage;
  assistantMessage: ChatMessage;
  memoryEvents: MemoryEvent[];
  contextPack: ContextPack;
  promptModuleUsages: PromptModuleUsage[];
  sidecarTrace: SidecarTrace;
  sidecar?: AssistantSidecar;
  imageCue: ImageCue;
  turnTrace: TurnTrace;
  imageJob?: ImageGenerationJob;
  imageJobs?: ImageGenerationJob[];
  imageAssets: ImageAsset[];
  // Resolves with an expanded sidecar when image_cues are completed in the background
  // (realtime image pipeline started generation with the initial cues for speed).
  sidecarExpansion?: Promise<AssistantSidecar | undefined>;
}
