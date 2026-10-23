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
  // Open-weight / open-source model endpoints. All speak the OpenAI chat-completions shape; their
  // per-provider quirks (thinking toggles, JSON mode, temperature ceilings, extra headers, CORS) live in
  // LLM_PROVIDER_PRESETS rather than in request-builder branches.
  | "deepseek"
  | "moonshot"
  | "dashscope"
  | "zhipu"
  | "openrouter"
  | "groq"
  | "together"
  | "ollama"
  | "lmstudio"
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
  /** Mirrors CharacterVisualProfile.photorealPrompt so the creation screen can edit it. */
  photorealPrompt?: string;
  photorealNegativePrompt?: string;
  /** Mirrors CharacterVisualProfile.profileImages; picked from test generations in the creation screen. */
  profileImages?: CharacterProfileImages;
}

/**
 * Which renderer draws a cut. `novelai` is the anime pipeline driven by Danbooru tags; `photoreal` is an
 * open-weight photo model (Qwen-Image / Z-Image) run by a local ComfyUI server and driven by natural language.
 */
export type ImageBackendId = "novelai" | "photoreal";

/**
 * A character's chosen profile image for one backend. It is both the face shown for the character and the face
 * REFERENCE that backend uses to keep the character recognisable across cuts (NovelAI: Precise Reference;
 * photoreal: multi-reference editing). Profile images are only ever picked from images DynamicChat generated —
 * there is deliberately no upload path, so a real person's photo cannot become a photoreal face reference.
 *
 * Not an ImageAsset: assets are per progress run and feed reuse/stored-asset picking, while a profile is
 * simulation-scoped and must never be attached to a turn as a scene image. The bytes live in the object store.
 */
export interface CharacterProfileImage {
  backend: ImageBackendId;
  /** Object id of the image (the file name in the object store). Never reused for different bytes. */
  assetId: Id;
  /** Object-store key, served at GET /objects/<key>. */
  objectKey?: string;
  mimeType?: string;
  width?: number;
  height?: number;
  /**
   * NovelAI only: the same image letterboxed onto 1024x1536, 1536x1024 or 1472x1472 with opaque black padding,
   * which is the only shape NovelAI Precise Reference accepts. Prepared once when the profile is picked.
   */
  referenceObjectKey?: string;
  prompt: string;
  /**
   * Photoreal only: the photo prompt the image was actually rendered from (style, camera and subject sentences), as
   * the safety floor screened it. Kept beside `prompt` (what the user typed, which the "adopt as appearance" button
   * reuses) because a typed prompt in Korean is dropped from the photo prompt and so cannot vouch for the image.
   */
  photoPrompt?: string;
  negativePrompt?: string;
  seed?: number;
  /** Photoreal only: the workflow preset that produced it. */
  workflowPreset?: PhotorealWorkflowPresetId;
  /** NovelAI only: the model preset that produced it. */
  modelPreset?: NovelAiModelPreset;
  selectedAt: string;
}

export type CharacterProfileImages = Partial<Record<ImageBackendId, CharacterProfileImage>>;

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
  /**
   * Natural-language appearance for the photoreal model ("a Korean woman in her late twenties with shoulder-length
   * black hair ..."). Replaces the NovelAI identity tags in photoreal cuts; outfit and pose still come from the cut.
   */
  photorealPrompt?: string;
  photorealNegativePrompt?: string;
  /** Chosen profile image per backend; also that backend's face reference. */
  profileImages?: CharacterProfileImages;
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

/**
 * The scene category the backend router puts every cut into. Exactly one per cut, decided deterministically by
 * code from the COMPOSED prompt (after identity/outfit/continuity injection) — never by the LLM, and never from the
 * authored tags alone, because continuity can inject an ongoing sex position into a cut that was authored as a
 * plain face close-up.
 */
export type ImageSceneCategory =
  /** Sexual content or nudity. Locked to NovelAI: never rendered photoreal, in any routing mode. */
  | "explicit"
  /** Dynamic motion: fighting, running, jumping, falling, sports. */
  | "action"
  /** Non-human or supernatural traits: animal ears, wings, horns, tails, magic, monsters. */
  | "fantasy"
  /** Non-explicit intimacy between people: kissing, hugging, holding hands, cuddling. */
  | "romance"
  /** Two or more people without intimate contact: conversation, daily life together. */
  | "group"
  /** One person, face or upper-body crop, static. */
  | "portrait"
  /** One person, cowboy / full-body / wide crop, static (outfit shots). */
  | "full_body"
  /** No people: landscape, interior, establishing shot. */
  | "scenery"
  /** No people, an object or food is the subject. */
  | "object";

/**
 * `novelai`: every cut is anime (the pre-photoreal behaviour, and the default for existing simulations).
 * `mixed`: cuts whose category is in `photorealCategories` render photoreal, the rest NovelAI.
 * `photoreal`: every cut renders photoreal except the safety-locked categories.
 */
export type ImageBackendRoutingMode = "novelai" | "mixed" | "photoreal";

export interface ImageBackendRouting {
  mode: ImageBackendRoutingMode;
  /** Categories rendered photoreal in `mixed` mode. `explicit` is ignored even if present. */
  photorealCategories: ImageSceneCategory[];
}

/** Why the router sent a cut where it did. Stored first-class on the job so it survives stored-run slimming. */
export interface ImageBackendDecision {
  backend: ImageBackendId;
  category: ImageSceneCategory;
  mode: ImageBackendRoutingMode;
  /** Short Korean explanation shown in the inspector. */
  reason: string;
  /** The safety floor forced NovelAI (explicit content, or minor-coded subjects). Cannot be overridden. */
  locked?: boolean;
  /** The preferred backend was not available, so the other one was used. */
  fallback?: boolean;
  /** The tags/fields that decided the category, for the inspector and the evals. */
  signals: string[];
}

export type PhotorealWorkflowPresetId =
  /** Qwen-Image 2.1 Turbo (8 steps): text-to-image AND up to 10 face references in one 7B model. Qwen research licence (non-commercial). */
  | "qwen_image_2_1_turbo"
  /** Qwen-Image 2.1 base (25 steps): same model family, slower, slightly higher quality. */
  | "qwen_image_2_1"
  /** Z-Image Turbo (8 steps, Apache-2.0): fastest photoreal text-to-image; no reference input. */
  | "z_image_turbo"
  /** Qwen-Image-Edit 2511 (Apache-2.0, 20B): licence-safe face-reference editor, 1-3 references. */
  | "qwen_image_edit_2511"
  /** A user-supplied API-format ComfyUI workflow with %placeholders%. */
  | "custom";

export type PhotorealStylePresetId = "instagram" | "daily_look" | "film_portrait" | "cinematic" | "studio_editorial" | "custom";

/** `proxy`: through the DynamicChat API server relay (ComfyUI refuses cross-origin browser calls). `mock`: dry run. */
export type PhotorealRequestMode = "proxy" | "mock";

export interface PhotorealModelFiles {
  diffusionModel?: string;
  textEncoder?: string;
  vae?: string;
}

export interface PhotorealApiSettings {
  enabled: boolean;
  provider: "comfyui";
  requestMode: PhotorealRequestMode;
  /** ComfyUI base URL. The relay only accepts loopback hosts unless DYNAMICCHAT_COMFYUI_HOSTS lists the host. */
  baseUrl: string;
  /** Overrides the relay endpoint (like NovelAiApiSettings.proxyUrl). Empty = `${API base}/photoreal/comfyui`. */
  proxyUrl: string;
  /** Workflow for cuts with no face reference. */
  workflowPreset: PhotorealWorkflowPresetId;
  /** Workflow for cuts whose characters have a photoreal profile image. Falls back to `workflowPreset` when it cannot take references. */
  referenceWorkflowPreset: PhotorealWorkflowPresetId;
  /** Model file overrides per preset; an empty field means the preset's default file. */
  modelFiles: Partial<Record<PhotorealWorkflowPresetId, PhotorealModelFiles>>;
  /** 0 = preset default. */
  steps: number;
  /** 0 = preset default. */
  cfg: number;
  /** 0 = derived from imageProfile's aspect ratio at about 1.2 megapixels. */
  width: number;
  height: number;
  seedFixed: boolean;
  seed?: number;
  stylePreset: PhotorealStylePresetId;
  /** Appended to the preset's style text; the whole style when stylePreset is `custom`. */
  stylePrompt: string;
  /**
   * Photoreal only: how every woman in a photo cut looks, added to her own photorealPrompt ("beautiful realistic
   * Korean woman, soft korean makeup, glossy lips"). Written once per simulation instead of into each character.
   */
  appearancePrompt: string;
  negativePrompt: string;
  /** Use characters' photoreal profile images as face references. Off unless chosen: it fixes a character's look. */
  faceReferenceEnabled: boolean;
  /**
   * Hand the GPU back and forth between a local LLM (Ollama) and local ComfyUI. A 16 GB card cannot hold
   * qwen3:14b (11.7 GB) and a photoreal model (13-15 GB) at once, and a spilled LLM runs at 0.1 tok/s.
   */
  vramHandoff: "auto" | "off";
  /** API-format ComfyUI workflow JSON for the `custom` preset. */
  customWorkflow: string;
  /** Per-image wait ceiling in seconds, including the first model load. */
  timeoutSeconds: number;
  registrationStatus: ApiRegistrationStatus;
  verifiedAt?: string;
  verificationMessage?: string;
}

export interface ImageGenerationProfile {
  id: Id;
  simulationId: Id;
  enabled: boolean;
  provider: "novelai";
  /** Per-cut backend routing. Absent on simulations saved before photoreal existed (= NovelAI only). */
  backendRouting?: ImageBackendRouting;
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

/** Which crop the cut uses. Decided before any other tag; everything else is filtered against it. */
export type ImageCueShot = "close_up" | "face_focus" | "upper_body" | "cowboy_shot" | "full_body" | "wide_shot";

/** Where the camera stands relative to the subject. Governs whether the face is renderable at all. */
export type ImageCueViewpoint = "front" | "side" | "from_behind" | "pov" | "over_the_shoulder";

/** Body regions a crop can show. A tag whose region is absent never reaches NovelAI. */
export type ImageCueBodyRegion = "head" | "torso" | "hips" | "legs" | "feet";

/**
 * The cut's composition, authored by the image-tag LLM before it picks any tag.
 *
 * This exists so frame consistency stops being an unverifiable claim about the model's private reasoning.
 * With it, DynamicChat can drop what the crop cannot show — footwear in a face close-up, an expression on a
 * figure facing away — both in the authored caption and in the appearance/outfit/continuity tags it injects
 * locally, which were previously frame-blind by construction.
 */
export interface ImageCueFrame {
  shot: ImageCueShot;
  viewpoint: ImageCueViewpoint;
  angle?: "eye_level" | "from_above" | "from_below" | "dutch_angle";
  visibleRegions: ImageCueBodyRegion[];
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
  frame?: ImageCueFrame;
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
  /** How many of this turn's image jobs were routed to the photoreal backend. */
  photorealJobCount?: number;
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
  frame?: ImageCueFrame;
  /** The annotation pass's beat kind (scene | action | body_detail | dialogue_face | context | interaction). Free text. */
  kind?: string;
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
  /** Renderer that drew this image. Absent = NovelAI (every asset made before photoreal existed). */
  backend?: ImageBackendId;
  sceneCategory?: ImageSceneCategory;
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
  /** Renderer this job runs on. Absent = NovelAI (every job planned before photoreal existed). */
  backend?: ImageBackendId;
  /** The router's decision, kept first-class because stored runs drop providerPayload. */
  routing?: ImageBackendDecision;
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
  /**
   * Usable context window of the configured model, in tokens. Overrides the provider preset's default.
   *
   * Matters most for local backends: Ollama serves every model with a 4096-token context unless told
   * otherwise, so the same model name can mean 4k or 32k depending on how it was loaded. The prompt is
   * sized from this, and the Ollama transport also passes it as num_ctx.
   */
  contextTokens?: number;
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
  /**
   * NovelAI Precise Reference from the characters' NovelAI profile images. V4.5 models only (NAID4.5F/C); NovelAI
   * charges 5 Anlas per reference per image even on Opus, so it is opt-in. Cannot be combined with Vibe Transfer in
   * one request — when a reference is attached the vibes are dropped for that request.
   */
  characterReferenceEnabled?: boolean;
  /** 0-1. Lower it if the reference's pose or expression leaks into every cut. Default 0.85. */
  characterReferenceStrength?: number;
  /** 0-1. Sent inverted as director_reference_secondary_strength_values (fidelity 1 → 0). Default 1. */
  characterReferenceFidelity?: number;
  /** true → base_caption "character&style" (carries the reference's rendering of the character); false → "character". */
  characterReferenceStyleAware?: boolean;
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
  /** Photoreal backend (local ComfyUI). Routing lives in imageProfile.backendRouting. */
  photoreal: PhotorealApiSettings;
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
  /**
   * Set when the annotation pass failed rather than deciding the turn was non-visual. That call authors
   * BOTH the image cues and the state deltas, so its failure is invisible on screen — the turn just quietly
   * produces no image and no state update. Also set when only the cut list came back empty on a cadence that
   * demands one (the tag model declined and the main-model fallback could not recover it). Surfaced to the
   * user by the caller.
   */
  annotationFailureReason?: string;
}
