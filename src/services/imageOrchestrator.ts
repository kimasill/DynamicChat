import { createId } from "../lib/id";
import type {
  AppState,
  CharacterVisualProfile,
  ImageAsset,
  ImageCue,
  ImageGenerationProfile,
  ImageGenerationJob,
  PromptModule
} from "../types";
import { generateNovelAiImages } from "./novelAiClient";
import { resolveNovelAiModelName } from "./novelAiModels";
import {
  createImageUserRulesForContentRating,
  isAdultContentMode
} from "./contentRating";
import {
  createCurrentCharacterImageStatePrompt,
  createCurrentSceneImageStateTags
} from "./imageStateTags";

interface ImagePolicyResult {
  allowed: boolean;
  warnings: string[];
  blockedReason?: string;
}

interface ImagePromptLayers {
  quality: string[];
  style: string[];
  artist: string[];
  imageProfiles: string[];
  characters: string[];
  context: string[];
  userRules: string[];
}

interface ImagePromptPlan {
  prompt: string;
  negativePrompt: string;
  positiveTags: string[];
  negativeTags: string[];
  layers: ImagePromptLayers;
  userRuleInstructions: string[];
}

interface ImagePromptVariant {
  label: string;
  prompt: string;
  tags: string[];
  cue: ImageCue;
}

interface PromptTagSplit {
  positive: string[];
  negative: string[];
}

export interface ImageReuseMatch {
  asset: ImageAsset;
  score: number;
  sharedTags: string[];
  targetTags: string[];
}

export interface ImageJobExecutionResult {
  job: ImageGenerationJob;
  assets: ImageAsset[];
}

export interface ImageJobProgressResult {
  job: ImageGenerationJob;
  asset: ImageAsset;
  assets: ImageAsset[];
}

export function shouldGenerateImage(state: AppState, cue: ImageCue, manual = false): boolean {
  if (!state.imageProfile.enabled || !state.simulation.realtimeImageEnabled) {
    return false;
  }
  if (state.imageProfile.triggerMode === "stored_only") {
    return false;
  }
  if (state.imageProfile.triggerMode === "manual") {
    return manual;
  }
  if (state.imageProfile.triggerMode === "realtime_confirm") {
    return cue.shouldGenerate && manual;
  }

  const recentCompletedJobs = state.imageJobs.filter((job) => job.status === "completed").slice(-1);
  const turnCountSinceLastJob = state.messages.length - (recentCompletedJobs.length > 0 ? findTurnIndex(state, recentCompletedJobs[0].turnId) : 0);
  return cue.shouldGenerate && turnCountSinceLastJob >= resolveImageGenerationCooldownTurns(state.imageProfile);
}

export function shouldPlanImageJob(state: AppState, cue: ImageCue, manual = false): boolean {
  if (!state.imageProfile.enabled || !state.simulation.realtimeImageEnabled) {
    return false;
  }
  if (state.imageProfile.triggerMode === "stored_only") {
    return false;
  }
  if (manual) {
    return true;
  }
  if (state.imageProfile.triggerMode === "manual") {
    return false;
  }
  if (state.imageProfile.triggerMode === "realtime_confirm") {
    return cue.shouldGenerate;
  }

  return shouldGenerateImage(state, cue, manual);
}

export function shouldAutoRunImageJob(job: ImageGenerationJob): boolean {
  const requiresConfirmation = Boolean(job.providerPayload.requiresConfirmation);
  const allowed = (job.providerPayload.policy as ImagePolicyResult | undefined)?.allowed !== false;
  return job.status === "queued" && !requiresConfirmation && allowed;
}

export function planImageJob(
  state: AppState,
  turnId: string,
  cue: ImageCue,
  contextNodeIds: string[],
  manual = false,
  options: { count?: number } = {}
): ImageGenerationJob {
  const scopedCue = normalizeImageCueCharacterScope(state, cue);
  const promptPlan = createImagePromptPlan(state, scopedCue);
  const imageUserRules = createImageUserRulesForContentRating(state);
  const prompt = promptPlan.prompt;
  const negativePrompt = promptPlan.negativePrompt;
  const jobId = createId("imgjob");
  const requestedCount = options.count ?? 1;
  const count = Math.min(8, Math.max(1, Math.round(requestedCount)));
  const promptVariants = createImagePromptVariants(state, promptPlan, scopedCue, count);
  const reuseTags = createReusableImageTags(scopedCue, promptPlan, promptVariants);
  const policy = validateImagePolicy(state, scopedCue, count);
  const requiresConfirmation = state.imageProfile.triggerMode === "realtime_confirm" && !manual;
  const resolvedModel = resolveNovelAiModelName(state.novelAi.modelPreset, state.imageProfile.model);
  const providerPayload: Record<string, unknown> = {
    provider: "novelai",
    payloadVersion: "dynamicchat-image-plan-v1",
    mode: "planned",
    model: resolvedModel,
    modelPreset: state.novelAi.modelPreset,
    width: state.imageProfile.width,
    height: state.imageProfile.height,
    steps: state.imageProfile.steps,
    promptGuidance: state.imageProfile.promptGuidance,
    count,
    contentRating: state.simulation.contentRating,
    adultContentMode: isAdultContentMode(state),
    safetyLevel: state.imageProfile.safetyLevel,
    triggerMode: state.imageProfile.triggerMode,
    generationCadence: state.imageProfile.generationCadence,
    requiresConfirmation,
    policy,
    userRules: imageUserRules,
    userRulesMode: "composer_instructions",
    userRuleInstructions: promptPlan.userRuleInstructions,
    promptFormat: "novelai-tags",
    promptLayers: promptPlan.layers,
    positiveTags: promptPlan.positiveTags,
    negativeTags: promptPlan.negativeTags,
    reuseTags,
    promptVariants: promptVariants.map((variant) => ({
      label: variant.label,
      prompt: variant.prompt,
      tags: variant.tags,
      cue: {
        scene: variant.cue.scene,
        tags: variant.cue.tags,
        visualContext: variant.cue.visualContext,
        characters: variant.cue.characters
      }
    })),
    contextTags: promptPlan.layers.context,
    cue: {
      shouldGenerate: scopedCue.shouldGenerate,
      scene: scopedCue.scene,
      tags: scopedCue.tags,
      characters: scopedCue.characters,
      visualContext: scopedCue.visualContext,
      suppressionReason: scopedCue.suppressionReason
    }
  };

  return {
    id: jobId,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    turnId,
    status: policy.allowed ? "queued" : "failed",
    reason: policy.blockedReason ?? scopedCue.reason,
    prompt,
    negativePrompt,
    providerPayload,
    assetIds: [],
    contextNodeIds,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    completedAt: policy.allowed ? undefined : new Date().toISOString(),
    error: policy.blockedReason,
    policyWarnings: policy.warnings
  };
}

function resolveImageGenerationCooldownTurns(profile: ImageGenerationProfile): number {
  const baseCooldown = Number.isFinite(profile.cooldownTurns) ? Math.max(0, Math.round(profile.cooldownTurns)) : 0;
  if (profile.generationCadence === "rich" || profile.generationCadence === "paragraph") {
    return 0;
  }
  if (profile.generationCadence === "sparse") {
    return Math.max(2, baseCooldown);
  }

  return baseCooldown;
}

function createImagePromptVariants(state: AppState, promptPlan: ImagePromptPlan, cue: ImageCue, count: number): ImagePromptVariant[] {
  if (count <= 1) {
    return [
      {
        label: "main",
        prompt: promptPlan.prompt,
        tags: promptPlan.positiveTags,
        cue
      }
    ];
  }

  const variantLayers = createContextVariantLayers(cue);
  return Array.from({ length: count }, (_, index) => {
    const variant = variantLayers[index % variantLayers.length];
    const tags = orderNovelAiPositiveTags(
      state,
      cue,
      uniqueStrings([
        ...promptPlan.positiveTags,
        ...variant.tags
      ]).filter((tag) => isNovelAiWeightedTag(tag) || !shouldRouteToNegativePrompt(tag))
    );
    const variantCue: ImageCue = {
      ...cue,
      tags: uniqueStrings([...cue.tags, ...variant.tags]),
      visualContext: uniqueStrings([cue.visualContext, variant.visualContext].filter((item): item is string => Boolean(item))).join(", ")
    };

    return {
      label: variant.label,
      prompt: tags.join(", "),
      tags,
      cue: variantCue
    };
  });
}

function createContextVariantLayers(cue: ImageCue): Array<{ label: string; visualContext: string; tags: string[] }> {
  const cueText = createCueText(cue);
  const variants: Array<{ label: string; visualContext: string; tags: string[] }> = [
    {
      label: "llm-cue",
      visualContext: cue.visualContext ?? "",
      tags: []
    }
  ];

  if (hasFaceVariantSignal(cueText)) {
    variants.push({
      label: "expression-closeup",
      visualContext: "close-up, face focus",
      tags: ["close-up", "face focus"]
    });
  }

  if (hasBodyDetailVariantSignal(cueText)) {
    const bodyFocusTags = selectBodyDetailVariantTags(cueText);
    variants.push({
      label: "body-action-detail",
      visualContext: bodyFocusTags.join(", "),
      tags: bodyFocusTags
    });
  }

  if (hasActionVariantSignal(cueText) || cue.characters.length > 0) {
    variants.push({
      label: "character-action",
      visualContext: "cowboy shot",
      tags: ["cowboy shot"]
    });
  }

  if (hasEnvironmentVariantSignal(cueText)) {
    variants.push({
      label: "establishing-view",
      visualContext: "wide shot",
      tags: ["wide shot"]
    });
  }

  if (hasActionVariantSignal(cueText)) {
    variants.push({
      label: "side-action",
      visualContext: "from side",
      tags: ["from side"]
    });
  }

  variants.push({
    label: "upper-body",
    visualContext: "upper body",
    tags: ["upper body"]
  });

  const seen = new Set<string>();
  return variants.filter((variant) => {
    const key = variant.tags.join(",");
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function hasFaceVariantSignal(text: string): boolean {
  return /\b(?:face|face focus|expression|open mouth|closed mouth|speaking|talking|dialogue|smile|evil smile|crying|tears?|tearing|blush|sweat|worried|tense|nervous|surprised|angry|sad|leering)\b/iu.test(
    text
  );
}

function hasBodyDetailVariantSignal(text: string): boolean {
  return /\b(?:hand|hands|arm|wrist|chest|breast|thigh|leg|feet|foot|body focus|grabbing|holding|touching|hand on|wrist grab)\b/iu.test(
    text
  );
}

function selectBodyDetailVariantTags(text: string): string[] {
  if (/\b(?:hand|hands|wrist|holding|grabbing|touching|hand on|wrist grab)\b/iu.test(text)) {
    return ["close-up", "hands"];
  }
  if (/\b(?:arm)\b/iu.test(text)) {
    return ["close-up", "arm focus"];
  }
  if (/\b(?:chest|breast)\b/iu.test(text)) {
    return ["close-up", "chest focus"];
  }
  if (/\b(?:thigh|leg)\b/iu.test(text)) {
    return ["close-up", "thigh focus"];
  }
  if (/\b(?:feet|foot)\b/iu.test(text)) {
    return ["close-up", "feet focus"];
  }
  return ["close-up", "body focus"];
}

function hasActionVariantSignal(text: string): boolean {
  return /\b(?:standing|sitting|lying|kneeling|crouching|walking|running|leaning|bending|reaching|holding|grabbing|touching|hugging|kissing|fighting|dancing|pointing|arm up|hand up|wrist grab)\b/iu.test(
    text
  );
}

function hasEnvironmentVariantSignal(text: string): boolean {
  return /\b(?:classroom|indoors|outdoors|stage|stage lights|practice room|dance studio|library|archive|bookshelf|hallway|street|room|bedroom|apartment|kitchen|cafe|restaurant|hospital|laboratory|forest|beach|battlefield|rain|snow|night|daylight)\b/iu.test(
    text
  );
}

function readPromptVariants(job: ImageGenerationJob, fallbackCue: ImageCue, count: number): ImagePromptVariant[] {
  const rawVariants = Array.isArray(job.providerPayload.promptVariants) ? job.providerPayload.promptVariants : [];
  const variants = rawVariants
    .map((item, index): ImagePromptVariant | undefined => {
      if (!item || typeof item !== "object") {
        return undefined;
      }

      const value = item as Record<string, unknown>;
      const prompt = typeof value.prompt === "string" && value.prompt.trim() ? value.prompt : undefined;
      if (!prompt) {
        return undefined;
      }

      const cueValue = value.cue && typeof value.cue === "object" ? (value.cue as Partial<ImageCue>) : {};
      return {
        label: typeof value.label === "string" ? value.label : `variant-${index + 1}`,
        prompt,
        tags: Array.isArray(value.tags) ? value.tags.filter((tag): tag is string => typeof tag === "string") : [],
        cue: {
          ...fallbackCue,
          scene: typeof cueValue.scene === "string" ? cueValue.scene : fallbackCue.scene,
          tags: Array.isArray(cueValue.tags) ? cueValue.tags.filter((tag): tag is string => typeof tag === "string") : fallbackCue.tags,
          characters: Array.isArray(cueValue.characters) ? cueValue.characters.filter((character): character is string => typeof character === "string") : fallbackCue.characters,
          visualContext: typeof cueValue.visualContext === "string" ? cueValue.visualContext : fallbackCue.visualContext
        }
      };
    })
    .filter((variant): variant is ImagePromptVariant => Boolean(variant))
    .slice(0, Math.min(8, Math.max(1, count)));

  return variants.length > 0 ? variants : [{ label: "main", prompt: job.prompt, tags: [], cue: fallbackCue }];
}

export async function executeImageJob(
  state: AppState,
  job: ImageGenerationJob,
  options: {
    onProgress?: (progress: ImageJobProgressResult) => void;
  } = {}
): Promise<ImageJobExecutionResult> {
  const policy = job.providerPayload.policy as ImagePolicyResult | undefined;
  if (policy?.allowed === false) {
    return {
      job: {
        ...job,
        status: "failed",
        error: policy.blockedReason ?? "Image policy blocked this job.",
        completedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      assets: []
    };
  }

  const cue = cueFromJob(job);
  const count = Number(job.providerPayload.count ?? 1);
  const promptVariants = readPromptVariants(job, cue, count);
  let dataUrls: string[] = [];
  let error: string | undefined;
  let providerPayload: Record<string, unknown> = {
    ...job.providerPayload,
    mode: "provider-result-pending"
  };
  const progressiveAssets: ImageAsset[] = [];

  try {
    if (promptVariants.length > 1) {
      const variantResults = [];
      for (const [variantIndex, variant] of promptVariants.entries()) {
        await waitForSequentialNovelAiSlot(state, variantIndex);
        const result = await generateNovelAiImages({
          state,
          prompt: variant.prompt,
          negativePrompt: job.negativePrompt,
          cue: variant.cue,
          count: 1
        });
        dataUrls.push(...result.dataUrls.slice(0, 1));
        variantResults.push({
          label: variant.label,
          prompt: variant.prompt,
          cue: variant.cue,
          payload: result.payload
        });
        const partialProviderPayload = {
          ...job.providerPayload,
          mode: "provider-result-partial",
          dynamicchatJobId: job.id,
          dynamicchatPayloadVersion: "dynamicchat-image-result-v1",
          policy,
          sequentialRequests: true,
          requestedCount: count,
          requestCount: variantResults.length,
          variantResults: [...variantResults],
          plannedPromptFormat: job.providerPayload.promptFormat,
          plannedPromptLayers: job.providerPayload.promptLayers,
          plannedPositiveTags: job.providerPayload.positiveTags,
          plannedNegativeTags: job.providerPayload.negativeTags,
          plannedCue: job.providerPayload.cue
        };
        const partialAsset = createGeneratedAsset(
          state,
          job.id,
          variant.prompt,
          job.negativePrompt,
          variant.cue,
          progressiveAssets.length,
          result.dataUrls[0],
          partialProviderPayload
        );
        progressiveAssets.push(partialAsset);
        options.onProgress?.({
          job: {
            ...job,
            status: "generating",
            providerPayload: partialProviderPayload,
            assetIds: progressiveAssets.map((asset) => asset.id),
            representativeAssetId: progressiveAssets[0]?.id,
            updatedAt: new Date().toISOString()
          },
          asset: partialAsset,
          assets: [...progressiveAssets]
        });
      }
      providerPayload = {
        ...job.providerPayload,
        mode: "provider-result",
        dynamicchatJobId: job.id,
        dynamicchatPayloadVersion: "dynamicchat-image-result-v1",
        policy,
        sequentialRequests: true,
        requestedCount: count,
        requestCount: variantResults.length,
        variantResults,
        plannedPromptFormat: job.providerPayload.promptFormat,
        plannedPromptLayers: job.providerPayload.promptLayers,
        plannedPositiveTags: job.providerPayload.positiveTags,
        plannedNegativeTags: job.providerPayload.negativeTags,
        plannedCue: job.providerPayload.cue
      };
    } else if (count > 1) {
      const variant = promptVariants[0] ?? { label: "main", prompt: job.prompt, tags: [], cue };
      const variantResults = [];
      for (let requestIndex = 0; requestIndex < count; requestIndex += 1) {
        await waitForSequentialNovelAiSlot(state, requestIndex);
        const result = await generateNovelAiImages({
          state,
          prompt: variant.prompt,
          negativePrompt: job.negativePrompt,
          cue: variant.cue,
          count: 1
        });
        dataUrls.push(...result.dataUrls.slice(0, 1));
        variantResults.push({
          label: `${variant.label}-${requestIndex + 1}`,
          prompt: variant.prompt,
          cue: variant.cue,
          payload: result.payload
        });
        const partialProviderPayload = {
          ...job.providerPayload,
          mode: "provider-result-partial",
          dynamicchatJobId: job.id,
          dynamicchatPayloadVersion: "dynamicchat-image-result-v1",
          policy,
          sequentialRequests: true,
          requestedCount: count,
          requestCount: variantResults.length,
          variantResults: [...variantResults],
          plannedPromptFormat: job.providerPayload.promptFormat,
          plannedPromptLayers: job.providerPayload.promptLayers,
          plannedPositiveTags: job.providerPayload.positiveTags,
          plannedNegativeTags: job.providerPayload.negativeTags,
          plannedCue: job.providerPayload.cue
        };
        const partialAsset = createGeneratedAsset(
          state,
          job.id,
          variant.prompt,
          job.negativePrompt,
          variant.cue,
          progressiveAssets.length,
          result.dataUrls[0],
          partialProviderPayload
        );
        progressiveAssets.push(partialAsset);
        options.onProgress?.({
          job: {
            ...job,
            status: "generating",
            providerPayload: partialProviderPayload,
            assetIds: progressiveAssets.map((asset) => asset.id),
            representativeAssetId: progressiveAssets[0]?.id,
            updatedAt: new Date().toISOString()
          },
          asset: partialAsset,
          assets: [...progressiveAssets]
        });
      }
      providerPayload = {
        ...job.providerPayload,
        mode: "provider-result",
        dynamicchatJobId: job.id,
        dynamicchatPayloadVersion: "dynamicchat-image-result-v1",
        policy,
        sequentialRequests: true,
        requestedCount: count,
        requestCount: variantResults.length,
        variantResults,
        plannedPromptFormat: job.providerPayload.promptFormat,
        plannedPromptLayers: job.providerPayload.promptLayers,
        plannedPositiveTags: job.providerPayload.positiveTags,
        plannedNegativeTags: job.providerPayload.negativeTags,
        plannedCue: job.providerPayload.cue
      };
    } else {
      const result = await generateNovelAiImages({
        state,
        prompt: job.prompt,
        negativePrompt: job.negativePrompt,
        cue,
        count: 1
      });
      dataUrls = result.dataUrls;
      providerPayload = {
        ...job.providerPayload,
        ...result.payload,
        dynamicchatJobId: job.id,
        dynamicchatPayloadVersion: "dynamicchat-image-result-v1",
        policy,
        sequentialRequests: true,
        requestedCount: count,
        requestCount: 1,
        plannedPromptFormat: job.providerPayload.promptFormat,
        plannedPromptLayers: job.providerPayload.promptLayers,
        plannedPositiveTags: job.providerPayload.positiveTags,
        plannedNegativeTags: job.providerPayload.negativeTags,
        plannedCue: job.providerPayload.cue
      };
    }
  } catch (caught) {
    error = caught instanceof Error ? caught.message : "Unknown NovelAI error";
  }

  const assetCount = Math.max(promptVariants.length > 1 ? promptVariants.length : count, dataUrls.length || 1);
  const assets = error
    ? []
    : progressiveAssets.length > 0
      ? progressiveAssets.slice(0, assetCount).map((asset) => ({
          ...asset,
          providerMetadata: providerPayload,
          representative: asset.id === progressiveAssets[0]?.id
        }))
      : Array.from({ length: assetCount }, (_, index) => {
        const variant = promptVariants[index % promptVariants.length];
        return createGeneratedAsset(
          state,
          job.id,
          variant?.prompt ?? job.prompt,
          job.negativePrompt,
          variant?.cue ?? cue,
          index,
          dataUrls[index],
          providerPayload
        );
      });
  const representativeAssetId = assets[0]?.id;

  return {
    job: {
      ...job,
      status: error ? "failed" : "completed",
      providerPayload,
      assetIds: assets.map((asset) => asset.id),
      representativeAssetId,
      completedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      error
    },
    assets
  };
}

async function waitForSequentialNovelAiSlot(state: AppState, requestIndex: number): Promise<void> {
  if (requestIndex <= 0) {
    return;
  }

  const baseDelayMs = Math.max(0, Number(state.novelAi.generationDelaySeconds) || 0) * 1000;
  if (baseDelayMs <= 0) {
    return;
  }

  const multiplier = state.novelAi.randomDelayEnabled ? 0.75 + Math.random() * 0.5 : 1;
  await delay(Math.round(baseDelayMs * multiplier));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => globalThis.setTimeout(resolve, ms));
}

export function pickStoredAsset(state: AppState, cue: ImageCue): ImageAsset | undefined {
  const characterMatches = state.imageAssets.filter((asset) => hasReusableCharacterScopeMatch(cue, asset));
  if (characterMatches.length > 0) {
    return characterMatches[characterMatches.length - 1];
  }

  if (cue.characters.length > 0) {
    return undefined;
  }

  return state.imageAssets.find((asset) => cue.tags.some((tag) => asset.tags.includes(tag))) ?? state.imageAssets[0];
}

export function findReusableImageAsset(
  state: AppState,
  job: ImageGenerationJob,
  options: { excludeAssetIds?: string[]; threshold?: number } = {}
): ImageReuseMatch | undefined {
  if (shouldBypassImageReuse(state, job)) {
    return undefined;
  }

  const cue = cueFromJob(job);
  const targetTags = getReusableTagsFromJob(job, cue);
  if (targetTags.length < 4) {
    return undefined;
  }

  const excludedIds = new Set(options.excludeAssetIds ?? []);
  const threshold = options.threshold ?? 0.82;
  const candidates = state.imageAssets
    .filter((asset) =>
      asset.simulationId === state.simulation.id &&
      asset.source === "generated" &&
      !excludedIds.has(asset.id) &&
      asset.feedback?.rating !== "rejected"
    )
    .map((asset) => scoreReusableImageAsset(asset, cue, targetTags))
    .filter((match): match is ImageReuseMatch => match !== undefined && match.score >= threshold)
    .sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }
      const likedDelta = Number(b.asset.feedback?.rating === "liked") - Number(a.asset.feedback?.rating === "liked");
      if (likedDelta !== 0) {
        return likedDelta;
      }
      return new Date(b.asset.createdAt).getTime() - new Date(a.asset.createdAt).getTime();
    });

  return candidates[0];
}

function shouldBypassImageReuse(state: AppState, job: ImageGenerationJob): boolean {
  if (job.providerPayload.forceFreshImage === true) {
    return true;
  }

  const userRules = createImageUserRulesForContentRating(state);
  return /재사용\s*(?:금지|하지\s*마|하지\s*말|안\s*함|불가)|기존\s*이미지\s*(?:사용|재사용)\s*(?:금지|하지\s*마|하지\s*말)|새(?:로|로운)\s*(?:이미지|컷)|fresh\s+image|do\s+not\s+reuse|no\s+reuse|never\s+reuse|always\s+generate|매번\s*(?:새로\s*)?생성|각\s*(?:문맥|장면|컷)[^\n]*(?:생성|그려|이미지)|(?:문맥|장면|컷)마다[^\n]*(?:생성|그려|이미지)|dialogue\s+face|body\s+detail|action\s+beat|대사\s*(?:전|앞|직전|이전|마다)/iu.test(
    userRules
  );
}

export function getReusableTagsFromJob(job: ImageGenerationJob, fallbackCue?: ImageCue): string[] {
  const cue = fallbackCue ?? cueFromJob(job);
  const payload = job.providerPayload;
  return uniqueReusableTags([
    ...readStringArray(payload.reuseTags),
    ...readStringArray(payload.contextTags),
    ...readReusableTagsFromPromptLayers(payload.promptLayers),
    ...readReusableTagsFromPromptLayers(payload.plannedPromptLayers),
    ...readCueReusableTags(payload.cue),
    ...readCueReusableTags(payload.plannedCue),
    ...createReusableTagsFromCue(cue)
  ]);
}

function scoreReusableImageAsset(asset: ImageAsset, cue: ImageCue, targetTags: string[]): ImageReuseMatch | undefined {
  const assetTags = getReusableTagsFromAsset(asset);
  if (assetTags.length < 4) {
    return undefined;
  }

  if (!hasReusableCharacterScopeMatch(cue, asset)) {
    return undefined;
  }

  const targetScene = resolveReusableScene(cue.scene, targetTags);
  const assetScene = resolveReusableScene(readAssetCueScene(asset), assetTags);
  if (targetScene && assetScene && targetScene !== assetScene) {
    return undefined;
  }
  if (targetScene && !assetScene && !assetTags.includes(targetScene)) {
    return undefined;
  }
  if (!hasRequiredReusableVisualStateMatch(targetTags, assetTags)) {
    return undefined;
  }

  const assetTagSet = new Set(assetTags);
  const sharedTags = targetTags.filter((tag) => assetTagSet.has(tag));
  const sharedDistinctiveTags = sharedTags.filter(isDistinctiveReusableTag);
  if (sharedDistinctiveTags.length < 2) {
    return undefined;
  }

  const criticalTargetTags = targetTags.filter(isCriticalReusableTag);
  if (criticalTargetTags.length > 0) {
    const sharedCriticalTags = criticalTargetTags.filter((tag) => assetTagSet.has(tag));
    const requiredCriticalTags = Math.max(1, Math.ceil(criticalTargetTags.length * 0.75));
    if (sharedCriticalTags.length < requiredCriticalTags) {
      return undefined;
    }
  }

  const minimumSharedTags = Math.max(4, Math.min(7, Math.ceil(Math.min(targetTags.length, assetTags.length) * 0.65)));
  if (sharedTags.length < minimumSharedTags) {
    return undefined;
  }

  const recall = sharedTags.length / targetTags.length;
  const precision = sharedTags.length / assetTags.length;
  const compactCoverage = sharedTags.length / Math.min(targetTags.length, assetTags.length);
  const characterScore =
    cue.characters.length === 0
      ? 0.04
      : asset.characterIds.length === 0
        ? 0
        : Math.min(1, cue.characters.filter((characterId) => asset.characterIds.includes(characterId)).length / cue.characters.length) * 0.1;
  const sceneScore = targetScene && assetScene && targetScene === assetScene ? 0.08 : 0;
  const likedBonus = asset.feedback?.rating === "liked" ? 0.03 : 0;
  const score = Math.min(1, compactCoverage * 0.48 + precision * 0.22 + recall * 0.15 + characterScore + sceneScore + likedBonus);

  return {
    asset,
    score,
    sharedTags,
    targetTags
  };
}

function hasReusableCharacterScopeMatch(cue: ImageCue, asset: ImageAsset): boolean {
  const targetCharacters = uniqueStrings(cue.characters);
  const assetCharacters = uniqueStrings(asset.characterIds);
  const metadataCharacterScopes = readAssetCueCharacterScopes(asset);
  if (targetCharacters.length === 0) {
    return assetCharacters.length === 0 && metadataCharacterScopes.every((scope) => scope.length === 0);
  }
  if (!hasSameReusableCharacterSet(targetCharacters, assetCharacters)) {
    return false;
  }

  return metadataCharacterScopes.every((scope) => scope.length === 0 || hasSameReusableCharacterSet(targetCharacters, scope));
}

function hasSameReusableCharacterSet(left: string[], right: string[]): boolean {
  if (left.length === 0 || right.length === 0 || left.length !== right.length) {
    return false;
  }

  const rightSet = new Set(right);
  return left.every((characterId) => rightSet.has(characterId));
}

function readAssetCueCharacterScopes(asset: ImageAsset): string[][] {
  return [
    readCueObject(asset.providerMetadata?.cue),
    readCueObject(asset.providerMetadata?.plannedCue)
  ].map((cue) => uniqueStrings(readStringArray(cue?.characters)));
}

function hasRequiredReusableVisualStateMatch(targetTags: string[], assetTags: string[]): boolean {
  const requiredTags = targetTags.filter(isRequiredReusableVisualStateTag);
  if (requiredTags.length === 0) {
    return true;
  }

  const assetTagSet = new Set(assetTags);
  return requiredTags.every((tag) => assetTagSet.has(tag));
}

function getReusableTagsFromAsset(asset: ImageAsset): string[] {
  return uniqueReusableTags([
    ...(Array.isArray(asset.reuseTags) ? asset.reuseTags : []),
    ...asset.tags,
    ...readStringArray(asset.providerMetadata?.reuseTags),
    ...readStringArray(asset.providerMetadata?.contextTags),
    ...readStringArray(asset.providerMetadata?.plannedPositiveTags),
    ...readStringArray(asset.providerMetadata?.positiveTags),
    ...readReusableTagsFromPromptLayers(asset.providerMetadata?.promptLayers),
    ...readReusableTagsFromPromptLayers(asset.providerMetadata?.plannedPromptLayers),
    ...readCueReusableTags(asset.providerMetadata?.cue),
    ...readCueReusableTags(asset.providerMetadata?.plannedCue)
  ]);
}

function readAssetCueScene(asset: ImageAsset): string | undefined {
  const cue = readCueObject(asset.providerMetadata?.cue) ?? readCueObject(asset.providerMetadata?.plannedCue);
  return typeof cue?.scene === "string" ? cue.scene : undefined;
}

function readCueObject(value: unknown): Partial<ImageCue> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Partial<ImageCue>) : undefined;
}

function resolveReusableScene(scene: string | undefined, tags: string[]): string | undefined {
  const sceneTag = scene ? normalizeSceneReusableTag(scene) : undefined;
  return sceneTag ?? tags.find(isSceneReusableTag);
}

function normalizeSceneReusableTag(value: string): string | undefined {
  const tag = normalizeReusableTag(value);
  return tag && isSceneReusableTag(tag) ? tag : undefined;
}

function isSceneReusableTag(tag: string): boolean {
  return /\b(?:archive library|library|bookshelf|clock tower|fountain|practice room|dance studio|stage|dormitory|apartment|kitchen|school interior|classroom|hallway|office|cafe|restaurant|street|hospital|lab|forest|beach|battlefield|public scene|commute|performance|room)\b/iu.test(
    tag
  );
}

function isDistinctiveReusableTag(tag: string): boolean {
  return !/^(?:girl|boy|girls|boys|other|others|\d+(?:girls?|boys?|others?)|medium shot|wide shot|close up|close-up|portrait|environment|facial expression|dynamic pose|dramatic|ambient|cinematic lighting|warm light|low light|smile|worried expression|tense expression|nervous expression|surprised expression|defiant expression)$/iu.test(
    tag
  );
}

function isCriticalReusableTag(tag: string): boolean {
  return /\b(?:standing|sitting|lying|kneeling|crouching|walking|running|reaching|holding|grabbing|hugging|kissing|fighting|dancing|sleeping|crying|pointing|looking at viewer|from behind|pov|low angle|high angle|close up|body detail|face focus|hands|climax|afterglow|intimate|conversation|argument|battle|performance|training|ceremony|classroom|archive library|library|stage|bedroom|room|hallway|street|hospital|laboratory|rain|snow|night|daylight)\b/iu.test(
    tag
  );
}

function isRequiredReusableVisualStateTag(tag: string): boolean {
  return isPoseOrActionReusableTag(tag) || isInteractionOrBodyReusableTag(tag) || isExpressionOrConditionReusableTag(tag) || isClothingStateReusableTag(tag) || isCameraStateReusableTag(tag);
}

function isPoseOrActionReusableTag(tag: string): boolean {
  return /\b(?:standing|sitting|lying|kneeling|crouching|walking|running|leaning|bending|reaching|arm up|arms up|hand up|hands up|spread legs|legs apart|thighs apart|crossed arms|hands on hips|dancing|fighting|sleeping|jumping|turning|looking back)\b/iu.test(
    tag
  );
}

function isInteractionOrBodyReusableTag(tag: string): boolean {
  return /\b(?:holding|grabbing|touching|hugging|kissing|pointing|wrist grab|hand on|hands on|holding hands|holding notebook|holding key|microphone|book|phone|weapon|sword|gun|hands|face focus|body focus|chest focus|breast focus|thigh focus|leg focus|feet focus|arm focus|back focus|pov hands)\b/iu.test(
    tag
  );
}

function isExpressionOrConditionReusableTag(tag: string): boolean {
  return /\b(?:smile|evil smile|soft smile|open mouth|closed mouth|crying|tears?|tearing|blush|sweat|sweating|worried|worried expression|tense|tense expression|nervous|nervous expression|surprised|surprised expression|angry|sad|frustrated|defiant|leering|half-closed eyes|closed eyes)\b/iu.test(
    tag
  );
}

function isClothingStateReusableTag(tag: string): boolean {
  return /\b(?:school uniform|uniform|training clothes|practice clothes|casual clothes|idol stage outfit|dress|jacket|cardigan|coat|raincoat|pajamas|naked|topless|bottomless|clothes lifted|panties aside|shirt open|skirt|pencil skirt|pleated skirt|necktie|ribbon|tight fit)\b/iu.test(
    tag
  );
}

function isCameraStateReusableTag(tag: string): boolean {
  return /\b(?:looking at viewer|looking away|from side|from behind|pov|low angle|high angle|dutch angle|close up|close-up|upper body|cowboy shot|full body|wide shot|medium shot|over the shoulder|over-the-shoulder)\b/iu.test(
    tag
  );
}

function createImagePromptPlan(state: AppState, cue: ImageCue): ImagePromptPlan {
  const scopedCue = normalizeImageCueCharacterScope(state, cue);
  const profile = state.imageProfile;
  const imageUserRules = createImageUserRulesForContentRating(state);
  const visualProfiles = getVisualProfiles(state, scopedCue);
  const usesNovelAiV4CharacterCaptions = state.novelAi.modelPreset !== "NAID3";
  const imageProfileModules = getImagePromptProfileModules(state);
  const visualPrompts = visualProfiles.map((visualProfile) => {
    const expression = selectExpressionPrompt(visualProfile, scopedCue);
    const outfit = selectOutfitPrompt(state, visualProfile, scopedCue);
    const currentImageState = createCurrentCharacterImageStatePrompt(state, visualProfile.characterId);
    return [visualProfile.positivePrompt, outfit, expression, currentImageState].filter(Boolean).join(", ");
  });
  const qualityTags = splitPromptField(profile.qualityPrompt);
  const styleTags = splitPromptField(profile.stylePrompt);
  const artistTags = splitPromptField(profile.artistPrompt);
  const rawImageProfileTags = splitPromptTagGroups(imageProfileModules.map((module) => module.body));
  const characterTags = splitPromptTagGroups(visualPrompts);
  const contextTags = uniqueStrings([...createContextTags(scopedCue), ...createCurrentSceneImageStateTags(state)]);
  const imageProfileTags = {
    positive: filterImageProfileTagsForCue(rawImageProfileTags.positive, contextTags),
    negative: rawImageProfileTags.negative
  };
  const userRuleTags = splitUserRuleTagDirectives(imageUserRules);
  const userRuleInstructions = createUserRuleInstructions(imageUserRules);
  const layers: ImagePromptLayers = {
    quality: qualityTags.positive,
    style: styleTags.positive,
    artist: artistTags.positive,
    imageProfiles: imageProfileTags.positive,
    characters: characterTags.positive,
    context: contextTags,
    userRules: userRuleTags.positive
  };
  const generatedCandidates = uniqueStrings([
    ...layers.imageProfiles,
    ...layers.context,
    ...layers.userRules,
    ...(usesNovelAiV4CharacterCaptions ? [] : layers.characters)
  ]);
  const positiveCandidates = uniqueStrings([
    ...layers.artist,
    ...layers.quality,
    ...layers.style,
    ...generatedCandidates
  ]);
  const reroutedNegativeTags = uniqueStrings(
    positiveCandidates
      .filter((tag) => !isNovelAiWeightedTag(tag) && shouldRouteToNegativePrompt(tag))
      .map(normalizeNegativePromptTag)
  );
  const generatedTags = orderNovelAiPositiveTags(
    state,
    scopedCue,
    filterRosterNameTags(
      state,
      uniqueStrings(
        applyUserRulePositiveConstraints(
          generatedCandidates
            .filter((tag) => isSubjectTagAllowedForCue(scopedCue, tag))
            .filter((tag) => isNovelAiWeightedTag(tag) || !shouldRouteToNegativePrompt(tag)),
          imageUserRules
        )
      )
    )
  );
  const positiveTags = uniqueStrings([
    ...layers.artist,
    ...layers.quality,
    ...layers.style,
    ...generatedTags
  ]);
  const negativeTags = uniqueStrings([
    ...qualityTags.negative,
    ...styleTags.negative,
    ...artistTags.negative,
    ...imageProfileTags.negative,
    ...(usesNovelAiV4CharacterCaptions ? [] : characterTags.negative),
    ...userRuleTags.negative,
    ...reroutedNegativeTags,
    ...promptToTags(profile.negativePrompt),
    ...(usesNovelAiV4CharacterCaptions ? [] : visualProfiles.flatMap((visualProfile) => promptToTags(visualProfile.negativePrompt)))
  ].map(normalizeNegativePromptTag)).filter((tag) => tag && !positiveTags.includes(tag));

  return {
    prompt: positiveTags.join(", "),
    negativePrompt: negativeTags.join(", "),
    positiveTags,
    negativeTags,
    layers,
    userRuleInstructions
  };
}

function splitPromptTagGroups(values: string[]): PromptTagSplit {
  return values.reduce<PromptTagSplit>(
    (result, value) => {
      const split = splitPromptField(value);
      result.positive.push(...split.positive);
      result.negative.push(...split.negative);
      return result;
    },
    { positive: [], negative: [] }
  );
}

function splitPromptField(value: string | undefined): PromptTagSplit {
  const tags = promptToTags(value);
  return {
    positive: uniqueStrings(tags.filter((tag) => isNovelAiWeightedTag(tag) || !shouldRouteToNegativePrompt(tag))),
    negative: uniqueStrings(
      tags
        .filter((tag) => !isNovelAiWeightedTag(tag) && shouldRouteToNegativePrompt(tag))
        .map(normalizeNegativePromptTag)
    )
  };
}

function getVisualProfiles(state: AppState, cue: ImageCue): CharacterVisualProfile[] {
  return state.visualProfiles.filter((profile) => cue.characters.includes(profile.characterId));
}

function normalizeImageCueCharacterScope(state: AppState, cue: ImageCue): ImageCue {
  if (cue.characters.length === 0) {
    return cue;
  }

  const cueText = createCueText(cue);
  const mentionedCharacterIds = findCueMentionedCharacterIds(state, cueText);
  if (shouldDropCharactersForExternalCue(state, cue.characters, cueText)) {
    return { ...cue, characters: [] };
  }

  if (mentionedCharacterIds.length > 0) {
    const mentioned = new Set(mentionedCharacterIds);
    const characters = cue.characters.filter((characterId) => mentioned.has(characterId));
    return characters.length === cue.characters.length ? cue : { ...cue, characters };
  }

  if (hasFullRosterSelectionWithoutGroupEvidence(state, cue.characters, cueText)) {
    return { ...cue, characters: [] };
  }

  if (hasSoloOrExternalSubjectCue(cueText)) {
    return { ...cue, characters: [] };
  }

  return cue;
}

function shouldDropCharactersForExternalCue(state: AppState, characterIds: string[], cueText: string): boolean {
  if (
    !(
      hasUnnamedExternalActorFocus(cueText) ||
      hasClearUnnamedOutsiderImageFocus(cueText) ||
      hasUnregisteredSoloSubjectCue(cueText)
    )
  ) {
    return false;
  }

  return !hasResolvedCharacterActorEvidence(state, uniqueStrings(characterIds), cueText);
}

function findCueMentionedCharacterIds(state: AppState, cueText: string): string[] {
  const normalizedText = cueText.toLowerCase();
  return uniqueStrings(
    state.characters
      .filter((character) => {
        const name = character.name.toLowerCase().trim();
        return Boolean(name && normalizedText.includes(name));
      })
      .map((character) => character.id)
  );
}

function hasSoloOrExternalSubjectCue(cueText: string): boolean {
  return /혼자|홀로|단독|1인칭|일인칭|놈|녀석|사내|남자|남성|남학생|남자애|소년|낯선\s*(?:사람|인물)|모르는\s*(?:사람|인물|남자)|다른\s*(?:사람|인물)|선생|교사|직원|스태프|경비|감독|\b(?:solo|alone|single subject|first[-\s]?person|pov|1boy|guy|boy|man|male|stranger|outsider|teacher|staff|guard|director)\b/iu.test(cueText);
}

function hasUnregisteredSoloSubjectCue(cueText: string): boolean {
  const normalized = cueText.replace(/\s+/gu, " ");
  const soloSubject = "(?:혼자|홀로|단독|single subject|solo|alone)";
  const externalSubject = "(?:놈|녀석|사내|남자애|남학생|남자|남성|소년|낯선\\s*(?:사람|인물)|모르는\\s*(?:사람|인물|남자)|다른\\s*(?:사람|인물)|선생|교사|직원|스태프|경비|감독|1인칭|일인칭|first[-\\s]?person|pov|1boy|guy|boy|man|male|stranger|outsider|teacher|staff|guard|director)";
  return new RegExp(`(?:${soloSubject})[^.\\n]{0,70}(?:${externalSubject})|(?:${externalSubject})[^.\\n]{0,70}(?:${soloSubject})`, "iu").test(normalized);
}

function hasClearUnnamedOutsiderImageFocus(text: string): boolean {
  return /무명|이름\s*없는|낯선|처음\s*보는|외부인|다른\s*(?:사람|인물)|모르는\s*(?:사람|인물|남자|남성|소년)|선생|교사|직원|스태프|경비|감독|놈|녀석|사내|1인칭|일인칭|first[-\s]?person|pov|unnamed|unknown|stranger|outsider|teacher|staff|guard|director|lone unnamed|solo unnamed/iu.test(text);
}

function hasUnnamedExternalActorFocus(text: string): boolean {
  return hasExternalSubjectActingCue(text) || hasFirstPersonAsObjectOfExternalActor(text);
}

function hasExternalSubjectActingCue(text: string): boolean {
  const normalized = text.replace(/\s+/gu, " ");
  const externalSubject = "(?:그\\s*)?(?:놈|녀석|사내|남자애|남학생|남자|남성|소년|낯선\\s*(?:사람|인물)|모르는\\s*(?:사람|인물|남자)|다른\\s*(?:사람|인물)|선생|교사|직원|스태프|경비|감독)|(?:guy|boy|man|male|stranger|outsider|teacher|staff|guard|director)";
  const actorVerb = "(?:웃|미소|말하|말했|말했다|지목|가리키|노려|쳐다|위협|협박|leering|leer|smil|speak|spoke|said|point|threaten)";
  return new RegExp(`(?:${externalSubject})(?:은|는|이|가|도)?[^.\\n]{0,90}${actorVerb}`, "iu").test(normalized);
}

function hasFirstPersonAsObjectOfExternalActor(text: string): boolean {
  const normalized = text.replace(/\s+/gu, " ");
  const firstPersonObject = "(?:나를|날|나에게|내게|나한테|내\\s*쪽으로|toward\\s+me|at\\s+me)";
  const externalSubject = "(?:그\\s*)?(?:놈|녀석|사내|남자애|남학생|남자|남성|소년|낯선\\s*(?:사람|인물)|모르는\\s*(?:사람|인물|남자)|다른\\s*(?:사람|인물)|선생|교사|직원|스태프|경비|감독)|(?:guy|boy|man|male|stranger|outsider|teacher|staff|guard|director)";
  const action = "(?:지목|가리키|노려|쳐다|위협|협박|비웃|웃|미소|말하|말했|말했다|point|stare|look|threaten|speak|spoke|said|smil|leer)";
  return new RegExp(`${firstPersonObject}[^.\\n]{0,90}(?:${action})?[^.\\n]{0,90}(?:${externalSubject})`, "iu").test(normalized);
}

function hasFullRosterSelectionWithoutGroupEvidence(state: AppState, characterIds: string[], cueText: string): boolean {
  if (state.characters.length < 3 || characterIds.length < state.characters.length) {
    return false;
  }

  if (/(?:전원|모두|다 같이|함께|세\s*명|네\s*명|전체|all|everyone|together|full cast|group shot|\b3girls\b|\b3boys\b)/iu.test(cueText)) {
    return false;
  }

  const mentionedIds = new Set(findCueMentionedCharacterIds(state, cueText));
  return mentionedIds.size < characterIds.length;
}

function hasResolvedCharacterActorEvidence(state: AppState, characterIds: string[], text: string): boolean {
  const actionPattern = /웃|미소|말하|말했|말했다|지목|가리키|노려|쳐다|위협|협박|leering|leer|smil|speak|spoke|said|point|threaten/iu;
  return characterIds.some((characterId) => {
    const character = state.characters.find((candidate) => candidate.id === characterId);
    const name = character?.name.trim();
    if (!name) {
      return false;
    }

    const index = text.toLowerCase().indexOf(name.toLowerCase());
    if (index < 0) {
      return false;
    }

    const segment = text.slice(index, index + 120);
    return actionPattern.test(segment) && !/(?:놈|녀석|사내|남자애|남학생|남자|남성|소년|\b(?:guy|boy|man|male)\b)/iu.test(segment.replace(name, ""));
  });
}

function isSubjectTagAllowedForCue(cue: ImageCue, tag: string): boolean {
  void cue;
  void tag;
  return true;
}

function getImagePromptProfileModules(state: AppState): PromptModule[] {
  return state.modules.filter(
    (module) => module.enabled && module.kind === "image_prompt_profile" && module.tokenPolicy !== "disabled"
  );
}

function selectOutfitPrompt(state: AppState, profile: CharacterVisualProfile, cue: ImageCue): string | undefined {
  const currentOutfit = selectCurrentOutfitPrompt(state, profile.characterId, profile.outfitPrompts);
  if (currentOutfit) {
    return currentOutfit;
  }

  const mappedOutfit = selectMappedOutfitPrompt(profile.outfitPrompts, cue);
  if (mappedOutfit) {
    return mappedOutfit;
  }

  const explicitOutfit = extractExplicitOutfitPrompt(cue);
  if (explicitOutfit) {
    return explicitOutfit;
  }

  const dynamicOutfit = inferDynamicOutfitPrompt(cue);
  if (dynamicOutfit) {
    return dynamicOutfit;
  }

  return normalizeOutfitPrompt(profile.defaultOutfitPrompt);
}

function selectExpressionPrompt(_profile: CharacterVisualProfile, cue: ImageCue): string | undefined {
  return inferDynamicExpressionPrompt(cue);
}

function selectMappedOutfitPrompt(outfitPrompts: Record<string, string> | undefined, cue: ImageCue): string | undefined {
  const cueText = createOutfitCueText(cue);
  const rawCueText = normalizeOutfitKeywordText([cue.scene, cue.visualContext, cue.tags.join(" ")].filter(Boolean).join(" "));
  if (hasSchoolOutfitCue(rawCueText)) {
    const schoolPrompt = Object.entries(outfitPrompts ?? {}).find(([key]) => isSchoolOutfitKeyword(key))?.[1];
    const normalizedSchoolPrompt = normalizeOutfitPrompt(schoolPrompt);
    if (normalizedSchoolPrompt) {
      return normalizedSchoolPrompt;
    }
  }
  const matches = Object.entries(outfitPrompts ?? {})
    .map(([key, value], index) => {
      const prompt = normalizeOutfitPrompt(value);
      if (!prompt) {
        return undefined;
      }

      const score = scoreOutfitKeywordMatch(key, cueText);
      return score > 0 ? { prompt, score, index } : undefined;
    })
    .filter((match): match is { prompt: string; score: number; index: number } => Boolean(match))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, 2)
    .map((match) => match.prompt);

  if (matches.length === 0 && hasSchoolOutfitCue(cueText)) {
    const schoolPrompt = Object.entries(outfitPrompts ?? {}).find(([key]) => isSchoolOutfitKeyword(key))?.[1];
    const normalizedSchoolPrompt = normalizeOutfitPrompt(schoolPrompt);
    if (normalizedSchoolPrompt) {
      return normalizedSchoolPrompt;
    }
  }

  return matches.length > 0 ? uniqueStrings(matches.flatMap((prompt) => promptToTags(prompt))).slice(0, 8).join(", ") : undefined;
}

function scoreOutfitKeywordMatch(key: string, cueText: string): number {
  const directScore = scoreTranslatedOutfitKeywordMatch(key, cueText);
  if (directScore > 0) {
    return directScore;
  }

  return key
    .split(/[,;\/|]+/u)
    .map((phrase) => phrase.trim())
    .filter(Boolean)
    .filter((phrase) => !/^(?:default|기본|base|fallback)$/iu.test(phrase))
    .reduce((score, phrase) => {
      const alias = createOutfitKeywordAliases(phrase).find((candidate) => cueText.includes(candidate));
      return alias ? Math.max(score, 1 + Math.min(3, alias.length / 8)) : score;
    }, 0);
}

function scoreTranslatedOutfitKeywordMatch(key: string, cueText: string): number {
  if (isSchoolOutfitKeyword(key) && hasSchoolOutfitCue(cueText)) {
    return 3;
  }
  const rules: Array<{ pattern: RegExp; cuePattern: RegExp; score: number }> = [
    { pattern: /학교|교복/u, cuePattern: /\b(?:school|classroom)\b|school uniform/u, score: 3 },
    { pattern: /연습|훈련|댄스/u, cuePattern: /\b(?:practice room|training|dance studio|workout)\b/u, score: 3 },
    { pattern: /무대|공연|아이돌/u, cuePattern: /\b(?:stage|performance|idol)\b/u, score: 3 },
    { pattern: /숙소|기숙|방|아파트|오피스텔/u, cuePattern: /\b(?:dormitory|dorm|apartment|room)\b/u, score: 2.6 },
    { pattern: /비|폭우|우산|우비/u, cuePattern: /\b(?:rain|rainy|storm|umbrella|raincoat)\b/u, score: 2.8 },
    { pattern: /겨울|눈|추운/u, cuePattern: /\b(?:winter|snow|cold)\b/u, score: 2.8 },
    { pattern: /전투|싸움|갑옷/u, cuePattern: /\b(?:battle|fight|combat|armor)\b/u, score: 2.8 },
    { pattern: /무도회|파티|정장|드레스/u, cuePattern: /\b(?:ball|party|formal|dress|suit)\b/u, score: 2.8 },
    { pattern: /실험실|연구실/u, cuePattern: /\b(?:lab|laboratory)\b/u, score: 2.7 },
    { pattern: /병원|진료/u, cuePattern: /\b(?:hospital|clinic)\b/u, score: 2.7 },
    { pattern: /카페|거리|외출/u, cuePattern: /\b(?:cafe|street|outing)\b/u, score: 2.4 }
  ];
  return rules.find((rule) => rule.pattern.test(key) && rule.cuePattern.test(cueText))?.score ?? 0;
}

function createOutfitKeywordAliases(phrase: string): string[] {
  const normalizedPhrase = normalizeOutfitKeywordText(phrase);
  return normalizedPhrase ? [normalizedPhrase] : [];
}

function hasSchoolOutfitCue(value: string): boolean {
  return /\b(?:school|classroom|student)\b|학교|교복|교실/u.test(value.normalize("NFC"));
}

function isSchoolOutfitKeyword(key: string): boolean {
  return /\b(?:school|classroom|student)\b|학교|교복|교실/iu.test(normalizeOutfitKeywordText(key));
}

function createOutfitCueText(cue: ImageCue): string {
  return normalizeOutfitKeywordText([cue.scene, cue.visualContext, cue.tags.join(" ")].filter(Boolean).join(" "));
}

function normalizeOutfitKeywordText(value: string): string {
  return value
    .normalize("NFC")
    .toLowerCase()
    .replace(/[._-]+/gu, " ")
    .replace(/[^\p{L}\p{N} ]+/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
}

function selectCurrentOutfitPrompt(state: AppState, characterId: string, outfitPrompts?: Record<string, string>): string | undefined {
  const currentOutfit = state.memoryEvents
    .slice()
    .reverse()
    .find((event) => {
      const metadata = event.metadata ?? {};
      const kind = readMetadataString(metadata, "memory_kind") ?? event.tags.find((tag) => tag.startsWith("kind:"))?.slice("kind:".length);
      const stateType = readMetadataString(metadata, "state_type") ?? readStateTypeFromTags(event.tags);
      const ownerId = readMetadataString(metadata, "owner_id") ?? event.actorId;
      return (
        kind === "state" &&
        ownerId === characterId &&
        Boolean(stateType && /^(?:Wearing|OutfitTags)$/iu.test(stateType))
      );
    });

  if (!currentOutfit) {
    return undefined;
  }

  const currentText = readMetadataString(currentOutfit.metadata, "value") ?? currentOutfit.content;
  const mappedCurrentOutfit = selectMappedOutfitPrompt(outfitPrompts, {
    shouldGenerate: true,
    reason: "current character outfit",
    characters: [characterId],
    tags: [],
    scene: currentText,
    visualContext: currentText
  });

  return mappedCurrentOutfit ?? normalizeOutfitPrompt(currentText);
}

function readStateTypeFromTags(tags: string[]): string | undefined {
  const stateTag = tags.find((tag) => tag.startsWith("state:"));
  return stateTag?.slice("state:".length);
}

function readMetadataString(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeOutfitPrompt(value: string | undefined): string | undefined {
  const tags = promptToTags(cleanOutfitPromptText(value));
  return tags.length > 0 ? tags.slice(0, 8).join(", ") : undefined;
}

function cleanOutfitPromptText(value: string | undefined): string | undefined {
  const stripped = value
    ?.replace(/^\[(?:State|Event|Observation|Belief|OpenThread|Goal)\]\s*/u, "")
    .trim();
  if (!stripped) {
    return undefined;
  }

  const explicitValue = stripped.match(/(?:Wearing|OutfitTags|착용|의상\s*태그|outfit tags?)\s*[=:：]\s*(.+)$/iu)?.[1]?.trim();
  return explicitValue || stripped;
}

function extractExplicitOutfitPrompt(cue: ImageCue): string | undefined {
  const rawContext = [cue.visualContext, cue.scene, cue.tags.join(", ")].filter(Boolean).join(" ");
  const match = rawContext.match(
    /(?:wearing|dressed in|outfit|clothing|costume|attire)[:\s]+([^.;\n]{3,90})/iu
  ) ?? rawContext.match(/(?:의상|복장|옷차림)[:：\s]+([^.\n]{2,90})/u);
  const value = match?.[1]?.trim();
  if (!value) {
    return undefined;
  }
  const tags = promptToTags(value);
  return tags.length > 0 ? tags.slice(0, 5).join(", ") : undefined;
}

function inferDynamicOutfitPrompt(cue: ImageCue): string | undefined {
  const context = createCueText(cue);
  const rules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /교복|학교|교실|\b(?:school|classroom|student)\b/iu, tags: ["school uniform"] },
    { pattern: /연습실|댄스|훈련|\b(?:practice room|dance studio|training|workout)\b/iu, tags: ["training clothes", "sneakers"] },
    { pattern: /무대|공연|아이돌|\b(?:stage|performance|idol)\b/iu, tags: ["idol stage outfit"] },
    { pattern: /전투|싸움|갑옷|\b(?:battle|fight|combat|armor)\b/iu, tags: ["combat outfit"] },
    { pattern: /비|폭우|우산|우비|\b(?:rain|rainy|storm|umbrella|raincoat)\b/iu, tags: ["raincoat"] },
    { pattern: /겨울|눈|추운|\b(?:winter|snow|cold)\b/iu, tags: ["winter coat", "scarf"] },
    { pattern: /무도회|파티|정장|드레스|\b(?:ball|party|formal|dress|suit)\b/iu, tags: ["formal outfit"] },
    { pattern: /실험실|연구실|\b(?:lab|laboratory)\b/iu, tags: ["lab coat"] },
    { pattern: /병원|진료|\b(?:hospital|clinic)\b/iu, tags: ["medical coat"] },
    { pattern: /카페|거리|외출|\b(?:cafe|street|outing)\b/iu, tags: ["casual outfit"] },
    { pattern: /숙소|기숙|방|아파트|오피스텔|\b(?:dorm|apartment|bedroom|room)\b/iu, tags: ["casual clothes"] },
    { pattern: /잠옷|취침|침대|\b(?:pajamas|sleepwear|bed)\b/iu, tags: ["pajamas"] }
  ];
  const tags = uniqueStrings(rules.flatMap((rule) => (rule.pattern.test(context) ? rule.tags : [])));
  return tags.length > 0 ? tags.slice(0, 6).join(", ") : undefined;
}

function inferDynamicExpressionPrompt(cue: ImageCue): string | undefined {
  const context = createCueText(cue);
  const rules: Array<{ pattern: RegExp; prompt: string }> = [
    { pattern: /relieved|smile|happy|joy|안도|웃|기쁨|행복/u, prompt: "soft smile" },
    { pattern: /tense|worried|anxious|fear|갈등|불안|걱정|긴장/u, prompt: "tense expression" },
    { pattern: /cry|teary|tear|울|눈물/u, prompt: "teary eyes" },
    { pattern: /angry|upset|분노|화남|짜증/u, prompt: "frustrated expression" },
    { pattern: /surprise|shock|놀람|당황/u, prompt: "surprised expression" }
  ];
  return rules.find((rule) => rule.pattern.test(context))?.prompt ?? undefined;
}

function createContextTags(cue: ImageCue): string[] {
  return uniqueStrings([
    ...contextTextToTags(cue.scene),
    ...cue.tags.flatMap((tag) => contextTextToTags(tag)),
    ...contextTextToTags(cue.visualContext)
  ]).filter((tag) => !shouldRouteToNegativePrompt(tag));
}

function filterImageProfileTagsForCue(tags: string[], contextTags: string[]): string[] {
  const activeEnvironmentGroups = uniqueStrings(contextTags.flatMap((tag) => {
    const group = getEnvironmentTagGroup(tag);
    return group ? [group] : [];
  }));
  if (activeEnvironmentGroups.length === 0) {
    return tags;
  }

  const activeGroups = new Set(activeEnvironmentGroups);
  return tags.filter((tag) => {
    const group = getEnvironmentTagGroup(tag);
    return !group || activeGroups.has(group);
  });
}

function getEnvironmentTagGroup(tag: string): string | undefined {
  const normalized = stripNovelAiTagWeight(tag).replace(/[._-]+/gu, " ").trim();
  if (/\b(?:archive library|library|bookshelf|archive)\b/iu.test(normalized)) {
    return "library";
  }
  if (/\b(?:classroom|school desk|chalkboard|blackboard)\b/iu.test(normalized)) {
    return "classroom";
  }
  if (/\b(?:stage|stage lights|spotlight|performance)\b/iu.test(normalized)) {
    return "stage";
  }
  if (/\b(?:street|alley|road|outdoors)\b/iu.test(normalized)) {
    return "street";
  }
  if (/\b(?:practice room|dance studio)\b/iu.test(normalized)) {
    return "practice";
  }
  if (/\b(?:bedroom|room|dormitory|apartment)\b/iu.test(normalized)) {
    return "room";
  }
  if (/\b(?:kitchen)\b/iu.test(normalized)) {
    return "kitchen";
  }
  if (/\b(?:cafe|restaurant)\b/iu.test(normalized)) {
    return "cafe";
  }
  if (/\b(?:hospital|clinic)\b/iu.test(normalized)) {
    return "hospital";
  }
  if (/\b(?:lab|laboratory)\b/iu.test(normalized)) {
    return "laboratory";
  }
  if (/\b(?:forest|woods)\b/iu.test(normalized)) {
    return "forest";
  }
  if (/\b(?:beach)\b/iu.test(normalized)) {
    return "beach";
  }
  if (/\b(?:battlefield)\b/iu.test(normalized)) {
    return "battlefield";
  }
  return undefined;
}

function filterRosterNameTags(state: AppState, tags: string[]): string[] {
  const rosterTagNames = new Set(
    state.characters.flatMap((character) => [
      normalizeRosterTag(character.id),
      normalizeRosterTag(character.name)
    ])
  );
  return tags.filter((tag) => !containsRosterNameTag(rosterTagNames, stripNovelAiTagWeight(tag)));
}

function normalizeRosterTag(value: string): string {
  return value.toLowerCase().replace(/[._\s]+/gu, "-").trim();
}

function containsRosterNameTag(rosterTagNames: Set<string>, value: string): boolean {
  const normalized = normalizeRosterTag(value);
  if (rosterTagNames.has(normalized)) {
    return true;
  }

  const parts = new Set(normalized.split(/-+/u).filter(Boolean));
  return [...rosterTagNames].some((name) => {
    if (!name) {
      return false;
    }
    const nameParts = name.split(/-+/u).filter(Boolean);
    return nameParts.length > 0 && nameParts.every((part) => parts.has(part));
  });
}

function filterContextConditionedTagsForCue(tags: string[], cue: ImageCue): string[] {
  void cue;
  return tags;
}

function createUserRuleInstructions(userRules: string): string[] {
  return userRules
    .split(/\n+/u)
    .map((line) => line.replace(/^[-*•\d.)\s]+/u, "").trim())
    .filter(Boolean)
    .slice(0, 32);
}

function splitUserRuleTagDirectives(userRules: string): PromptTagSplit {
  return userRules.split(/\n+/u).reduce<PromptTagSplit>(
    (result, rawLine) => {
      const directive = readUserRuleTagDirective(rawLine);
      if (!directive) {
        return result;
      }

      const tags = promptToTags(directive.value);
      if (directive.kind === "negative") {
        result.negative.push(...tags.map(normalizeNegativePromptTag));
        return result;
      }

      result.positive.push(
        ...tags.filter((tag) => isNovelAiWeightedTag(tag) || !shouldRouteToNegativePrompt(tag))
      );
      result.negative.push(
        ...tags
          .filter((tag) => !isNovelAiWeightedTag(tag) && shouldRouteToNegativePrompt(tag))
          .map(normalizeNegativePromptTag)
      );
      return result;
    },
    { positive: [], negative: [] }
  );
}

function readUserRuleTagDirective(line: string): { kind: "positive" | "negative"; value: string } | undefined {
  const trimmed = line.trim().replace(/^[-*•\d.)\s]+/u, "").trim();
  const negative = trimmed.match(
    /^(?:negative|undesired|uc|avoid|exclude|ban(?:ned)?|금지|제외|네거티브|부정|원치\s*않는\s*내용|언디자이어드)(?:\s*(?:tags?|prompt|태그|프롬프트|수칙))?\s*[:：-]\s*(.+)$/iu
  );
  if (negative?.[1]?.trim()) {
    return { kind: "negative", value: negative[1].trim() };
  }

  const positive = trimmed.match(
    /^(?:positive|prompt|tags?|include|always|must\s*include|nai\s*tags?|포지티브|프롬프트|태그|토큰|포함|항상|반드시)(?:\s*(?:tags?|prompt|태그|프롬프트|수칙))?\s*[:：-]\s*(.+)$/iu
  );
  if (positive?.[1]?.trim()) {
    return { kind: "positive", value: positive[1].trim() };
  }

  return undefined;
}

function applyUserRulePositiveConstraints(tags: string[], userRules: string): string[] {
  void userRules;
  const promptTags = tags.flatMap((tag) => normalizePromptTagParts(tag));
  return uniqueStrings(promptTags);
}

function promptToTags(value: string | undefined): string[] {
  if (!value) {
    return [];
  }

  const rawParts = splitPromptTags(value);
  const parts = rawParts.length > 0 ? rawParts : [value.trim()];

  return uniqueStrings(
    parts.flatMap((part) => {
      return normalizePromptTagParts(part);
    })
  );
}

function splitPromptTags(value: string): string[] {
  const parts: string[] = [];
  let current = "";
  let inNovelAiEmphasis = false;

  for (let index = 0; index < value.length; index += 1) {
    const rest = value.slice(index);
    const opening = !inNovelAiEmphasis ? rest.match(/^(?:-?\d+(?:\.\d+)?\s*)?::/u)?.[0] : undefined;
    if (opening) {
      current += opening;
      index += opening.length - 1;
      inNovelAiEmphasis = true;
      continue;
    }

    if (inNovelAiEmphasis && rest.startsWith("::")) {
      current += "::";
      index += 1;
      inNovelAiEmphasis = false;
      continue;
    }

    const char = value[index];
    if (!inNovelAiEmphasis && /[,;\n|]/u.test(char)) {
      const trimmed = current.trim();
      if (trimmed) {
        parts.push(trimmed);
      }
      current = "";
      continue;
    }

    current += char;
  }

  const trimmed = current.trim();
  if (trimmed) {
    parts.push(trimmed);
  }

  return parts;
}

function normalizePromptTag(value: string): string | undefined {
  const trimmed = value.trim();
  const weighted = parseNovelAiWeightedTag(trimmed);
  if (weighted) {
    return weighted.normalized;
  }

  const normalized = value
    .trim()
    .replace(/[.!?。！？]+$/gu, "")
    .replace(/\s+/gu, " ");
  if (!normalized) {
    return undefined;
  }
  if (!/[\p{L}\p{N}:]/u.test(normalized)) {
    return undefined;
  }
  if (/[\u3131-\uD79D]/u.test(normalized)) {
    return undefined;
  }
  const lower = normalized.toLowerCase();
  if (
    ["current scene", "current simulation scene", "generated scene", "simulation scene", "scene"].includes(lower) ||
    isStaleContextSceneTag(lower)
  ) {
    return undefined;
  }
  return normalized;
}

function normalizePromptTagParts(value: string): string[] {
  const expanded = expandPromptTagAlias(value);
  if (expanded) {
    return expanded.flatMap((part) => {
      const normalized = normalizePromptTag(part);
      return normalized ? [normalized] : [];
    });
  }

  const normalized = normalizePromptTag(value);
  return normalized ? [normalized] : [];
}

function expandPromptTagAlias(value: string): string[] | undefined {
  const normalized = value
    .trim()
    .replace(/[.!?。！？]+$/gu, "")
    .replace(/[._-]+/gu, " ")
    .replace(/[^\p{L}\p{N}: ]+/gu, " ")
    .replace(/\s+/gu, " ")
    .toLowerCase();
  if (!normalized || /[\u3131-\uD79D]/u.test(normalized)) {
    return undefined;
  }

  const expressionMatch = normalized.match(/^(worried|tense|nervous|surprised|frustrated|angry|sad|happy|confident|defiant|evil)\s+expression$/u);
  if (expressionMatch?.[1]) {
    return expressionMatch[1] === "happy" ? ["smile"] : [expressionMatch[1]];
  }

  const directRules: Array<{ pattern: RegExp; tags: string[] }> = [
    { pattern: /^(?:standing up from (?:her |his |their )?(?:seat|chair)|standing up)$/u, tags: ["standing", "chair"] },
    { pattern: /^(?:getting up|getting up from (?:a |the )?(?:seat|chair))$/u, tags: ["standing", "chair"] },
    { pattern: /^(?:raising (?:her |his |their )?hand(?: eagerly)?|hand raised|raised hand)$/u, tags: ["arm up", "hand up"] },
    { pattern: /^raising (?:her |his |their )?arm(?: eagerly)?$/u, tags: ["arm up"] },
    { pattern: /^bright and confident smile$/u, tags: ["smile", "confident"] },
    { pattern: /^confident smile$/u, tags: ["smile", "confident"] },
    { pattern: /^classroom setting$/u, tags: ["classroom"] },
    { pattern: /^school classroom$/u, tags: ["classroom", "indoors"] },
    { pattern: /^(?:other students?(?: blurred)? in (?:the )?background|students? blurred in (?:the )?background)$/u, tags: ["blurred background"] },
    { pattern: /^(?:close up face|close face|face close up|character close up)$/u, tags: ["close-up", "face focus"] },
    { pattern: /^(?:speaking|talking)$/u, tags: ["open mouth"] },
    { pattern: /^(?:looking forward)$/u, tags: ["looking at viewer"] },
    { pattern: /^(?:standing pose)$/u, tags: ["standing"] }
  ];
  const direct = directRules.find((rule) => rule.pattern.test(normalized));
  if (direct) {
    return direct.tags;
  }

  if (/^(?:filming set|props?|main action|facial expression|visible emotional reaction|situation specific clothing|wide context|clear environment|context appropriate outfit|dynamic pose|current scene|current simulation scene|generated scene|simulation scene|scene)$/u.test(normalized)) {
    return [];
  }

  return undefined;
}

function contextTextToTags(value: string | undefined): string[] {
  if (!value) {
    return [];
  }

  const parts = value
    .split(/[,;\n|]+/u)
    .map((part) => part.trim())
    .filter(Boolean);

  return uniqueStrings(
    parts.flatMap((part) => {
      return normalizeContextTagParts(part);
    })
  );
}

function normalizeContextTagParts(value: string): string[] {
  const expanded = normalizePromptTagParts(value);
  return uniqueStrings(
    expanded.flatMap((part) => {
      const normalized = normalizeContextTag(part);
      return normalized ? [normalized] : [];
    })
  );
}

function normalizeContextTag(value: string): string | undefined {
  const trimmed = value.trim();
  if (!trimmed || /[\u3131-\uD79D]/u.test(trimmed)) {
    return undefined;
  }

  const normalized = trimmed
    .replace(/[._-]+/gu, " ")
    .replace(/[^\p{L}\p{N}: ]+/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (!normalized) {
    return undefined;
  }
  const lower = normalized.toLowerCase();
  if (["current scene", "current simulation scene", "generated scene", "simulation scene"].includes(lower)) {
    return undefined;
  }

  return normalized.toLowerCase();
}

function isStaleContextSceneTag(tag: string): boolean {
  return /\b(?:commute|chapter|episode|cut|shot)\s+\d+\b/iu.test(tag) || /\bschool uniform commute\b/iu.test(tag);
}

function shouldRouteToNegativePrompt(tag: string): boolean {
  return isTechnicalNegativeTag(tag);
}

function isTechnicalNegativeTag(tag: string): boolean {
  const weighted = parseNovelAiWeightedTag(tag);
  const normalized = (weighted?.tag ?? normalizeNegativePromptTag(tag)).toLowerCase();
  return /\b(?:lowres|blurry|worst quality|bad quality|bad anatomy|bad hands|bad face|deformed|distorted|disfigured|mutated|mutation|ugly|extra fingers|extra limbs|missing limbs|missing fingers|text|watermark|logo|caption|subtitles|artist name|signature|jpeg artifacts|scan artifacts|duplicate|chibi)\b/iu.test(
    normalized
  );
}

function normalizeNegativePromptTag(tag: string): string {
  const trimmed = tag.trim();
  const weighted = parseNovelAiWeightedTag(trimmed);
  if (weighted) {
    return weighted.normalized;
  }
  const value = trimmed.replace(/^-+\s*/u, "");
  return normalizePromptTag(value) ?? value.replace(/\s+/gu, " ").trim();
}

function parseNovelAiWeightedTag(value: string): { weight?: number; tag: string; normalized: string } | undefined {
  const trimmed = value.trim();
  const weighted = trimmed.match(/^(-?\d+(?:\.\d+)?)\s*::([\s\S]*)$/u);
  if (weighted?.[2]?.trim()) {
    const weightText = weighted[1];
    const body = weighted[2];
    const hasClosing = body.trimEnd().endsWith("::");
    const tag = (hasClosing ? body.trimEnd().slice(0, -2) : body).trim();
    if (!tag) {
      return undefined;
    }
    return {
      weight: Number(weightText),
      tag,
      normalized: hasClosing ? trimmed : `${weightText}::${tag} ::`
    };
  }

  if (trimmed.startsWith("::")) {
    const body = trimmed.slice(2);
    const hasClosing = body.trimEnd().endsWith("::");
    if (!hasClosing) {
      return undefined;
    }
    const tag = body.trimEnd().slice(0, -2).trim();
    if (!tag) {
      return undefined;
    }
    return {
      tag,
      normalized: trimmed
    };
  }

  return undefined;
}

function isNovelAiWeightedTag(tag: string): boolean {
  return Boolean(parseNovelAiWeightedTag(tag));
}

function orderNovelAiPositiveTags(state: AppState, cue: ImageCue, tags: string[]): string[] {
  void state;
  void cue;
  const candidates = uniqueStrings(tags.flatMap((tag) => normalizePromptTagParts(tag)));
  const filtered = candidates.filter((tag) => !shouldDropContradictoryPromptTag(tag, candidates));
  const datasetTags = filtered.filter(isNovelAiDatasetTag);
  const artistTags = filtered.filter(isArtistPromptTag);
  const qualityTags = filtered.filter(isNovelAiQualityOrAestheticTag);
  const visualTags = filtered
    .filter((tag) => !isNovelAiDatasetTag(tag) && !isArtistPromptTag(tag) && !isNovelAiQualityOrAestheticTag(tag))
    .map((tag, index) => ({ tag, index, rank: getNovelAiPositiveTagRank(tag) }))
    .sort((a, b) => a.rank - b.rank || a.index - b.index)
    .map((item) => item.tag);

  return uniqueStrings([...datasetTags, ...artistTags, ...visualTags, ...qualityTags]);
}

function isSubjectCountTag(tag: string): boolean {
  return /^\d+(?:girls?|boys?|others?)$/iu.test(tag.trim());
}

function isNovelAiDatasetTag(tag: string): boolean {
  return /^(?:fur dataset|background dataset)$/iu.test(stripNovelAiTagWeight(tag));
}

function isNovelAiQualityOrAestheticTag(tag: string): boolean {
  return /^(?:masterpiece|best quality|amazing quality|great quality|normal quality|bad quality|worst quality|top aesthetic|very aesthetic|aesthetic|displeasing|very displeasing)$/iu.test(
    stripNovelAiTagWeight(tag)
  );
}

function getNovelAiPositiveTagRank(tag: string): number {
  const normalized = stripNovelAiTagWeight(tag);
  if (isSubjectCountTag(normalized)) {
    return 0;
  }
  if (/\b(?:looking at viewer|looking away|looking back|pov|from side|from behind|front view|side view|low angle|high angle|dutch angle|close up|close-up|face focus|upper body|cowboy shot|full body|portrait|wide shot|medium shot|over the shoulder|over-the-shoulder|fisheye|depth of field)\b/iu.test(normalized)) {
    return 1;
  }
  if (/\b(?:classroom|indoors|outdoors|school desk|desk|chair|chalkboard|blackboard|bed|bedroom|room|hallway|street|alley|library|archive|bookshelf|stage|spotlight|practice room|dance studio|dormitory|apartment|kitchen|cafe|restaurant|hospital|clinic|lab|laboratory|forest|beach|battlefield|rain|snow|night|daylight|window|door|blurred background|background)\b/iu.test(normalized)) {
    return 2;
  }
  if (/\b(?:standing|sitting|lying|kneeling|crouching|walking|running|leaning|bending|reaching|arm up|arms up|hand up|hands up|spread legs|legs apart|thighs apart|from above|from below)\b/iu.test(normalized)) {
    return 3;
  }
  if (/\b(?:holding|grabbing|touching|hand on|hands on|hugging|kissing|fighting|dancing|microphone|notebook|book|pen|phone|weapon|sword|gun|hands|breasts?|chest|hips?|thighs?|legs?|feet|mouth|tongue|penis|pussy|vagina|penetration)\b/iu.test(normalized)) {
    return 4;
  }
  if (/\b(?:hair|eyes?|twintails|twin tails|ponytail|braid|glasses|freckles|scar|beauty mark|horns?|tail|ears?|uniform|school uniform|skirt|pencil skirt|pleated skirt|necktie|ribbon|shirt|blouse|jacket|cardigan|dress|suit|coat|raincoat|sweater|pants|shorts|panties|bra|shoes|sneakers|boots|socks|thighhighs|naked|topless|bottomless|clothes lifted|panties aside|shirt open|tight fit)\b/iu.test(normalized)) {
    return 5;
  }
  if (/\b(?:smile|confident|worried|tense|nervous|surprised|frustrated|angry|sad|defiant|evil smile|leering|crying|tears?|tearing|blush|sweat|open mouth|half-closed eyes|closed eyes|light|lighting|spotlight|glow|motion blur|skin detail)\b/iu.test(normalized)) {
    return 6;
  }
  return 7;
}

function shouldDropContradictoryPromptTag(tag: string, candidates: string[]): boolean {
  const normalized = stripNovelAiTagWeight(tag);
  const candidateText = candidates.map(stripNovelAiTagWeight).join(", ");
  const hasIndoorAnchor = /\b(?:classroom|indoors|school desk|chalkboard|blackboard|bedroom|room|hallway|library|archive|bookshelf|hospital|clinic|lab|laboratory|dormitory|apartment|kitchen|cafe|restaurant)\b/iu.test(candidateText);
  const hasOutdoorAnchor = /\b(?:outdoors|street|alley|forest|beach|battlefield)\b/iu.test(candidateText);
  if (hasIndoorAnchor && /\b(?:outdoors|street|alley|forest|beach|battlefield)\b/iu.test(normalized)) {
    return true;
  }
  if (!hasIndoorAnchor && hasOutdoorAnchor && /\b(?:indoors|classroom|school desk|chalkboard|blackboard|bedroom|room|hallway|library|archive|bookshelf|hospital|clinic|lab|laboratory|dormitory|apartment|kitchen)\b/iu.test(normalized)) {
    return true;
  }
  return false;
}

function stripNovelAiTagWeight(tag: string): string {
  return parseNovelAiWeightedTag(tag)?.tag.toLowerCase().trim() ?? tag.toLowerCase().trim();
}

function deriveTagsFromText(value: string | undefined): string[] {
  void value;
  return [];
}

function createReusableImageTags(cue: ImageCue, promptPlan: ImagePromptPlan, promptVariants: ImagePromptVariant[]): string[] {
  void promptVariants;
  return uniqueReusableTags([
    ...createReusableTagsFromCue(cue),
    ...promptPlan.layers.context
  ]);
}

function createReusableTagsFromCue(cue: ImageCue): string[] {
  return filterContextConditionedTagsForCue(uniqueReusableTags([
    ...cue.tags,
    ...contextTextToTags(cue.scene),
    ...contextTextToTags(cue.visualContext)
  ]), cue);
}

function readReusableTagsFromPromptLayers(value: unknown): string[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }

  const layers = value as Partial<ImagePromptLayers>;
  return uniqueReusableTags([
    ...readStringArray(layers.context),
    ...readStringArray(layers.userRules)
  ]);
}

function readCueReusableTags(value: unknown): string[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }

  const cue = value as Partial<ImageCue>;
  return uniqueReusableTags([
    ...readStringArray(cue.tags),
    ...contextTextToTags(typeof cue.scene === "string" ? cue.scene : undefined),
    ...contextTextToTags(typeof cue.visualContext === "string" ? cue.visualContext : undefined)
  ]);
}

function readStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0) : [];
}

function uniqueReusableTags(values: string[]): string[] {
  return uniqueStrings(
    values
      .map(normalizeReusableTag)
      .filter((tag): tag is string => Boolean(tag))
  ).slice(0, 48);
}

function normalizeReusableTag(value: string): string | undefined {
  const weighted = parseNovelAiWeightedTag(value);
  const tag = (weighted?.tag ?? value)
    .toLowerCase()
    .replace(/[{}[\]"'`]/gu, "")
    .replace(/[._-]+/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (!tag || /[\u3131-\uD79D]/u.test(tag)) {
    return undefined;
  }
  if (isTechnicalNegativeTag(tag) || isNovelAiQualityOrAestheticTag(tag) || isNovelAiDatasetTag(tag) || isArtistPromptTag(tag)) {
    return undefined;
  }
  if (/^(?:highres|absurdres|intricate body details|highly detailed skin|detailed background|sharp focus)$/iu.test(tag)) {
    return undefined;
  }
  if (/^(?:current|generated|simulation|safe)?\s*scene$/iu.test(tag)) {
    return undefined;
  }
  if (isGenericReusableTag(tag)) {
    return undefined;
  }
  if (tag.length < 2 || tag.length > 48) {
    return undefined;
  }

  return tag;
}

function isGenericReusableTag(tag: string): boolean {
  return /^(?:dramatic|ambient|environment|medium shot|wide shot|close up|close-up|portrait|over the shoulder shot|over-the-shoulder shot|clear character interaction|visible facial reaction|facial expression|visible emotional reaction|visible emotional response|current action beat|whole current scene|situation specific clothing|situation appropriate clothing|context appropriate outfit|hands and posture visible|close up face|close-up face|close up body detail|close-up body detail|generated image|filming set|movie set|set|props?|prop interaction|getting up|clear environment|clear location|visible character interaction)$/iu.test(
    tag
  );
}

function isArtistPromptTag(tag: string): boolean {
  return /\bartist(?::|_|\s)|\b(?:by|style of)\s+[a-z0-9_()-]+/iu.test(stripNovelAiTagWeight(tag));
}

function createCueText(cue: ImageCue): string {
  return [cue.scene, cue.tags.join(" "), cue.visualContext].filter(Boolean).join(" ").toLowerCase();
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

function createGeneratedAsset(
  state: AppState,
  jobId: string,
  prompt: string,
  negativePrompt: string,
  cue: ImageCue,
  index: number,
  dataUrl?: string,
  providerMetadata?: Record<string, unknown>
): ImageAsset {
  const palettes: Array<[string, string, string]> = [
    ["#213547", "#d5e1e8", "#f2b84b"],
    ["#2c2438", "#c9a9d8", "#68b0ab"],
    ["#153b3d", "#d7e9dc", "#c36f55"]
  ];
  const reuseTags = uniqueReusableTags([
    ...createReusableTagsFromCue(cue),
    ...readStringArray(providerMetadata?.reuseTags),
    ...readStringArray(providerMetadata?.contextTags),
    ...readReusableTagsFromPromptLayers(providerMetadata?.promptLayers),
    ...readReusableTagsFromPromptLayers(providerMetadata?.plannedPromptLayers)
  ]);

  return {
    id: createId("asset"),
    simulationId: state.simulation.id,
    title: createGeneratedAssetTitle(cue, index),
    source: "generated",
    prompt,
    negativePrompt,
    safetyLevel: state.imageProfile.safetyLevel,
    characterIds: cue.characters,
    tags: reuseTags.length > 0 ? reuseTags : cue.tags,
    createdAt: new Date().toISOString(),
    jobId,
    palette: palettes[index % palettes.length],
    dataUrl,
    mimeType: dataUrl ? dataUrl.slice(5, dataUrl.indexOf(";")) : undefined,
    providerMetadata,
    reuseTags,
    representative: index === 0
  };
}

function createGeneratedAssetTitle(cue: ImageCue, index: number): string {
  const scene = cue.scene && !/^(?:current|generated|simulation|safe) scene/iu.test(cue.scene)
    ? cue.scene.replace(/\s+/gu, " ").trim()
    : "Generated image";
  return `${scene} #${index + 1}`;
}

function findTurnIndex(state: AppState, turnId: string): number {
  const index = state.messages.findIndex((message) => message.id === turnId);
  return index >= 0 ? index : 0;
}

function validateImagePolicy(state: AppState, cue: ImageCue, count: number): ImagePolicyResult {
  const warnings: string[] = [];
  const completedJobs = state.imageJobs.filter((job) => job.status === "completed").length;
  const requestedTotal = completedJobs + count;

  if (state.novelAi.automationTermination === "count" && requestedTotal > state.novelAi.countLimit) {
    return {
      allowed: false,
      warnings,
      blockedReason: `이미지 생성 count limit 초과: ${requestedTotal}/${state.novelAi.countLimit}`
    };
  }

  return {
    allowed: true,
    warnings
  };
}

function cueFromJob(job: ImageGenerationJob): ImageCue {
  const cue = job.providerPayload.cue as Partial<ImageCue> | undefined;
  return {
    shouldGenerate: cue?.shouldGenerate ?? true,
    reason: job.reason,
    characters: Array.isArray(cue?.characters) ? cue.characters : [],
    tags: Array.isArray(cue?.tags) ? cue.tags : [],
    scene: typeof cue?.scene === "string" ? cue.scene : "generated scene",
    suppressionReason: typeof cue?.suppressionReason === "string" ? cue.suppressionReason : undefined,
    visualContext: typeof cue?.visualContext === "string" ? cue.visualContext : undefined
  };
}
