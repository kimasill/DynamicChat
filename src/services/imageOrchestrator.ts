import { createId } from "../lib/id";
import type {
  AppState,
  ImageAsset,
  ImageCue,
  ImageCueCharacterPrompt,
  ImageGenerationProfile,
  ImageGenerationJob,
  PromptModule
} from "../types";
import { generateNovelAiImages } from "./novelAiClient";
import { resolveNovelAiModelName } from "./novelAiModels";
import {
  createImageUserRulesForContentRating,
  isAdultContentMode,
  resolveEffectiveImageSafetyLevel
} from "./contentRating";
import {
  canonicalizeStateType,
  classifyImageStateTags,
  readStateMemoryOwnerId,
  readStateMemoryStateType,
  readStateMemoryValue
} from "./stateMemory";
import { inferCurrentSceneCharacterIds } from "./sceneCast";
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
  characterPrompts: ImageCueCharacterPrompt[];
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
  const scopedCue = cue;
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
    safetyLevel: resolveEffectiveImageSafetyLevel(state),
    triggerMode: state.imageProfile.triggerMode,
    generationCadence: state.imageProfile.generationCadence,
    requiresConfirmation,
    policy,
    userRules: imageUserRules,
    userRulesMode: "composer_instructions",
    userRuleInstructions: promptPlan.userRuleInstructions,
    promptFormat: "novelai-tags",
    promptLayers: promptPlan.layers,
    characterPrompts: promptPlan.characterPrompts,
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
        baseTags: variant.cue.baseTags,
        characterPrompts: variant.cue.characterPrompts,
        visualContext: variant.cue.visualContext,
        characters: variant.cue.characters
      }
    })),
    contextTags: promptPlan.layers.context,
    cue: {
      shouldGenerate: scopedCue.shouldGenerate,
      scene: scopedCue.scene,
      tags: scopedCue.tags,
      baseTags: scopedCue.baseTags,
      characterPrompts: promptPlan.characterPrompts,
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
  if (profile.generationCadence === "rich" || profile.generationCadence === "paragraph" || profile.generationCadence === "image_progression") {
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
      baseTags: cue.baseTags ? uniqueStrings([...cue.baseTags, ...variant.tags]) : undefined,
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
  return [
    {
      label: "llm-cue",
      visualContext: cue.visualContext ?? "",
      tags: []
    }
  ];
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
          baseTags: Array.isArray(cueValue.baseTags) ? cueValue.baseTags.filter((tag): tag is string => typeof tag === "string") : fallbackCue.baseTags,
          characterPrompts: Array.isArray(cueValue.characterPrompts)
            ? cueValue.characterPrompts.filter((prompt): prompt is ImageCueCharacterPrompt => isImageCueCharacterPrompt(prompt))
            : fallbackCue.characterPrompts,
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
    onProgress?: (progress: ImageJobProgressResult) => void | Promise<void>;
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
        assertNovelAiResultHasImages(result.dataUrls, variant.label);
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
        await options.onProgress?.({
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
        assertNovelAiResultHasImages(result.dataUrls, `${variant.label}-${requestIndex + 1}`);
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
        await options.onProgress?.({
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
      assertNovelAiResultHasImages(result.dataUrls);
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

function assertNovelAiResultHasImages(dataUrls: string[], label = "main"): void {
  if (dataUrls.length === 0) {
    throw new Error(`NovelAI response contained no image files for ${label}.`);
  }
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
  const targetTags = createReusableTagsFromCue(cue);
  const candidates = state.imageAssets
    .filter((asset) => asset.simulationId === state.simulation.id && asset.source === "stored" && asset.feedback?.rating !== "rejected")
    .map((asset) => scoreStoredImageAsset(asset, cue, targetTags))
    .filter((match): match is ImageReuseMatch => Boolean(match))
    .sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }
      return new Date(b.asset.createdAt).getTime() - new Date(a.asset.createdAt).getTime();
    });

  return candidates[0]?.asset;
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
      asset.feedback?.rating !== "rejected" &&
      // Only reuse an asset whose rendered size matches the currently configured resolution; otherwise reusing an older
      // asset would silently ignore a changed width/height (e.g. serving a square 1024² image after switching to 832×1216).
      assetMatchesConfiguredResolution(asset, state)
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

function assetMatchesConfiguredResolution(asset: ImageAsset, state: AppState): boolean {
  const targetWidth = Math.round(Number(state.imageProfile.width));
  const targetHeight = Math.round(Number(state.imageProfile.height));
  if (!Number.isFinite(targetWidth) || !Number.isFinite(targetHeight) || targetWidth <= 0 || targetHeight <= 0) {
    return true;
  }

  const dimensions = readAssetDimensions(asset);
  if (dimensions.width === undefined || dimensions.height === undefined) {
    // Unknown asset size: allow reuse and let the semantic match decide. Real generated assets always carry their
    // rendered size (so a genuine resolution change is still caught below); only legacy/dimensionless assets land
    // here, and blocking them would silently defeat reuse for the common case.
    return true;
  }

  return Math.round(dimensions.width) === targetWidth && Math.round(dimensions.height) === targetHeight;
}

function readAssetDimensions(asset: ImageAsset): { width?: number; height?: number } {
  const metadata = asset.providerMetadata;
  if (!metadata) {
    return {};
  }
  return {
    width: readProviderMetadataNumber(metadata, "width"),
    height: readProviderMetadataNumber(metadata, "height")
  };
}

function readProviderMetadataNumber(payload: Record<string, unknown>, key: string): number | undefined {
  const direct = payload[key];
  if (typeof direct === "number" && Number.isFinite(direct)) {
    return direct;
  }

  const parameters = payload.parameters;
  if (parameters && typeof parameters === "object" && !Array.isArray(parameters)) {
    const nested = (parameters as Record<string, unknown>)[key];
    if (typeof nested === "number" && Number.isFinite(nested)) {
      return nested;
    }
  }

  return undefined;
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

  const targetScene = resolveReusableScene(cue.scene, targetTags);
  const assetScene = resolveReusableScene(readAssetCueScene(asset), assetTags);
  if (!hasStrictReusableSemanticMatch(asset, cue, targetTags, assetTags)) {
    return undefined;
  }
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

function scoreStoredImageAsset(asset: ImageAsset, cue: ImageCue, targetTags: string[]): ImageReuseMatch | undefined {
  const assetTags = getReusableTagsFromAsset(asset);
  if (assetTags.length === 0) {
    return undefined;
  }
  if (!hasStrictReusableSemanticMatch(asset, cue, targetTags, assetTags)) {
    return undefined;
  }

  const targetScene = resolveReusableScene(cue.scene, targetTags);
  const assetScene = resolveReusableScene(readAssetCueScene(asset), assetTags);
  if (targetScene && assetScene && targetScene !== assetScene) {
    return undefined;
  }

  const assetTagSet = new Set(assetTags);
  const sharedTags = targetTags.filter((tag) => assetTagSet.has(tag));
  const sceneMatched = Boolean(targetScene && (assetScene === targetScene || assetTagSet.has(targetScene)));
  const characterMatched = cue.characters.length > 0 && asset.characterIds.length > 0;
  if (sharedTags.length === 0 && !sceneMatched && !characterMatched) {
    return undefined;
  }

  const denominator = Math.max(1, Math.min(targetTags.length || assetTags.length, assetTags.length));
  const score = Math.min(
    1,
    sharedTags.length / denominator +
      (sceneMatched ? 0.18 : 0) +
      (characterMatched ? 0.12 : 0) +
      (asset.representative ? 0.03 : 0)
  );

  return {
    asset,
    score,
    sharedTags,
    targetTags
  };
}

function hasStrictReusableSemanticMatch(
  asset: ImageAsset,
  cue: ImageCue,
  targetTags: string[],
  assetTags: string[]
): boolean {
  return (
    hasReusableCharacterScopeMatch(cue, asset) &&
    hasReusableSubjectCountMatch(cue, asset, targetTags, assetTags) &&
    hasReusableCharacterTagMatch(targetTags, assetTags) &&
    hasReusableActionTagMatch(targetTags, assetTags)
  );
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

interface ReusableSubjectCountSignature {
  total?: number;
  girls?: number;
  boys?: number;
  others?: number;
  conflict: boolean;
}

function hasReusableSubjectCountMatch(cue: ImageCue, asset: ImageAsset, targetTags: string[], assetTags: string[]): boolean {
  const targetSignature = createReusableSubjectCountSignature(targetTags, cue.characters.length);
  const assetSignature = createReusableSubjectCountSignature(assetTags, asset.characterIds.length);
  if (targetSignature.conflict || assetSignature.conflict) {
    return false;
  }

  const targetHasSubject =
    targetSignature.total !== undefined || cue.characters.length > 0 || targetTags.some(isHumanSubjectReusableTag);
  const assetHasSubject =
    assetSignature.total !== undefined || asset.characterIds.length > 0 || assetTags.some(isHumanSubjectReusableTag);
  if (!targetHasSubject && !assetHasSubject) {
    return true;
  }
  if (targetSignature.total === undefined || assetSignature.total === undefined) {
    return false;
  }
  if (targetSignature.total !== assetSignature.total) {
    return false;
  }

  const targetGenderedCount = (targetSignature.girls ?? 0) + (targetSignature.boys ?? 0) + (targetSignature.others ?? 0);
  const assetGenderedCount = (assetSignature.girls ?? 0) + (assetSignature.boys ?? 0) + (assetSignature.others ?? 0);
  if (targetGenderedCount > 0 && assetGenderedCount > 0) {
    return (
      (targetSignature.girls ?? 0) === (assetSignature.girls ?? 0) &&
      (targetSignature.boys ?? 0) === (assetSignature.boys ?? 0) &&
      (targetSignature.others ?? 0) === (assetSignature.others ?? 0)
    );
  }

  return true;
}

function createReusableSubjectCountSignature(tags: string[], characterCount: number): ReusableSubjectCountSignature {
  let girls = 0;
  let boys = 0;
  let others = 0;
  let totalFromTags: number | undefined;

  for (const tag of tags) {
    const normalized = stripNovelAiTagWeight(tag).replace(/\s+/gu, " ").trim();
    if (/^(?:no humans?|empty scene|background only)$/iu.test(normalized)) {
      totalFromTags = mergeReusableSubjectTotal(totalFromTags, 0);
      continue;
    }

    const counted = normalized.match(/^(\d+)\s*(girls?|boys?|others?|people|persons?|humans?|characters?)$/iu);
    if (counted) {
      const count = Number(counted[1]);
      const kind = counted[2].toLowerCase();
      if (/^girls?$/u.test(kind)) {
        girls += count;
      } else if (/^boys?$/u.test(kind)) {
        boys += count;
      } else if (/^others?$/u.test(kind)) {
        others += count;
      }
      totalFromTags = mergeReusableSubjectTotal(totalFromTags, count, true);
      continue;
    }

    const namedCount = readNamedReusableSubjectCount(normalized);
    if (namedCount !== undefined) {
      totalFromTags = mergeReusableSubjectTotal(totalFromTags, namedCount);
    }
  }

  const genderedTotal = girls + boys + others;
  const resolvedTagTotal = genderedTotal > 0 ? genderedTotal : totalFromTags;
  const total = resolvedTagTotal ?? (characterCount > 0 ? characterCount : undefined);
  const conflict =
    (resolvedTagTotal !== undefined && characterCount > 0 && resolvedTagTotal !== characterCount) ||
    (totalFromTags !== undefined && genderedTotal > 0 && totalFromTags !== genderedTotal);

  return {
    total,
    girls: girls > 0 ? girls : undefined,
    boys: boys > 0 ? boys : undefined,
    others: others > 0 ? others : undefined,
    conflict
  };
}

function mergeReusableSubjectTotal(current: number | undefined, next: number, additive = false): number {
  if (current === undefined) {
    return next;
  }

  return additive ? current + next : current;
}

function readNamedReusableSubjectCount(tag: string): number | undefined {
  if (/^(?:solo|single person|single character)$/iu.test(tag)) {
    return 1;
  }
  if (/^(?:duo|pair|couple|two people|two characters)$/iu.test(tag)) {
    return 2;
  }
  if (/^(?:trio|three people|three characters)$/iu.test(tag)) {
    return 3;
  }

  return undefined;
}

function hasReusableCharacterTagMatch(targetTags: string[], assetTags: string[]): boolean {
  const targetCharacterTags = targetTags.filter(isCharacterIdentityReusableTag);
  if (targetCharacterTags.length === 0) {
    return true;
  }

  const assetTagSet = new Set(assetTags);
  return targetCharacterTags.every((tag) => assetTagSet.has(tag));
}

function hasReusableActionTagMatch(targetTags: string[], assetTags: string[]): boolean {
  const targetActionTags = targetTags.filter(isActionReusableTag);
  const assetActionTags = assetTags.filter(isActionReusableTag);
  if (targetActionTags.length === 0) {
    return assetActionTags.length === 0;
  }
  if (assetActionTags.length === 0) {
    return false;
  }

  return hasSameReusableTagSet(targetActionTags, assetActionTags);
}

function hasSameReusableTagSet(left: string[], right: string[]): boolean {
  const uniqueLeft = uniqueStrings(left);
  const uniqueRight = uniqueStrings(right);
  if (uniqueLeft.length !== uniqueRight.length) {
    return false;
  }

  const rightSet = new Set(uniqueRight);
  return uniqueLeft.every((tag) => rightSet.has(tag));
}

function isHumanSubjectReusableTag(tag: string): boolean {
  return (
    /^(?:solo|duo|pair|couple|trio|group|crowd|multiple girls?|multiple boys?|multiple people|no humans?|empty scene|background only|\d+\s*(?:girls?|boys?|others?|people|persons?|humans?|characters?))$/iu.test(
      tag
    ) ||
    isCharacterIdentityReusableTag(tag) ||
    isClothingStateReusableTag(tag) ||
    isExpressionOrConditionReusableTag(tag) ||
    isActionReusableTag(tag)
  );
}

function isCharacterIdentityReusableTag(tag: string): boolean {
  return /\b(?:hair|eyes?|twintails|twin tails|ponytail|braid|drill hair|bob cut|bangs|ahoge|glasses|freckles|scar|beauty mark|mole|horns?|tail|animal ears?|cat ears?|fox ears?|elf ears?|wings?|halo|skin|complexion|tattoo|makeup|fangs?)\b/iu.test(
    tag
  );
}

function isActionReusableTag(tag: string): boolean {
  return isPoseOrActionReusableTag(tag) || /\b(?:holding|grabbing|touching|hugging|kissing|pointing|wrist grab|hand on|hands on|holding hands|holding notebook|holding key|holding phone|holding microphone|holding book|holding weapon|reaching out|taking|pushing|pulling|opening|closing|writing|reading|eating|drinking|playing instrument|singing|microphone|book|phone|weapon|sword|gun)\b/iu.test(
    tag
  );
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
  const scopedCue = cue;
  const profile = state.imageProfile;
  const imageUserRules = createImageUserRulesForContentRating(state);
  const imageProfileModules = getImagePromptProfileModules(state);
  const qualityTags = splitPromptField(profile.qualityPrompt);
  const styleTags = splitPromptField(profile.stylePrompt);
  const artistTags = splitPromptField(profile.artistPrompt);
  const rawImageProfileTags = splitPromptTagGroups(imageProfileModules.map((module) => module.body));
  const characterPrompts = createCueCharacterPrompts(state, scopedCue);
  const contextTags = createContextTags(state, scopedCue, characterPrompts.length > 0);
  const characterPromptTags = uniqueStrings(characterPrompts.flatMap((prompt) => promptToTags(prompt.prompt)));
  const imageProfileTags = {
    positive: rawImageProfileTags.positive,
    negative: rawImageProfileTags.negative
  };
  const userRuleTags = splitUserRuleTagDirectives(imageUserRules);
  const userRuleInstructions = createUserRuleInstructions(imageUserRules);
  const layers: ImagePromptLayers = {
    quality: qualityTags.positive,
    style: styleTags.positive,
    artist: artistTags.positive,
    imageProfiles: imageProfileTags.positive,
    characters: characterPromptTags,
    context: contextTags,
    userRules: userRuleTags.positive
  };
  const generatedCandidates = uniqueStrings([
    ...layers.imageProfiles,
    ...layers.context,
    ...layers.userRules
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
    uniqueStrings(
      applyUserRulePositiveConstraints(
        generatedCandidates
          .filter((tag) => isNovelAiWeightedTag(tag) || !shouldRouteToNegativePrompt(tag)),
        imageUserRules
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
    ...userRuleTags.negative,
    ...reroutedNegativeTags,
    ...promptToTags(profile.negativePrompt)
  ].map(normalizeNegativePromptTag)).filter((tag) => tag && !positiveTags.includes(tag));

  return {
    prompt: positiveTags.join(", "),
    negativePrompt: negativeTags.join(", "),
    positiveTags,
    negativeTags,
    layers,
    characterPrompts,
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

function getImagePromptProfileModules(state: AppState): PromptModule[] {
  return state.modules.filter(
    (module) => module.enabled && module.kind === "image_prompt_profile" && module.tokenPolicy !== "disabled"
  );
}

function createCueCharacterPrompts(state: AppState, cue: ImageCue): ImageCueCharacterPrompt[] {
  const explicitPrompts = (cue.characterPrompts ?? [])
    .map((prompt, index) => normalizeCueCharacterPrompt(prompt, index))
    .filter((prompt): prompt is ImageCueCharacterPrompt => Boolean(prompt));
  const allCueTags = uniqueStrings([...(cue.baseTags ?? []), ...(cue.tags ?? [])]);
  const legacyCharacterTags = uniqueStrings(allCueTags.flatMap((tag) => promptToTags(tag)).filter(isCharacterPromptTag));

  if (explicitPrompts.length > 0) {
    const composed = explicitPrompts.map((prompt, index) =>
      composeCharacterPrompt(state, prompt, index === 0 ? legacyCharacterTags : [])
    );
    return rebalanceCharacterCenters(composed.filter((prompt) => prompt.prompt.trim()));
  }

  if (legacyCharacterTags.length === 0) {
    return [];
  }

  const characterIds = cue.characters.length > 0 ? cue.characters : [undefined];
  return characterIds.map((characterId, index) => ({
    characterId,
    prompt: legacyCharacterTags.join(", "),
    center: createDefaultCharacterCenter(index, characterIds.length)
  }));
}

// Anti-runaway backstop only. The LLM is the tag author and the prompt instructions own focus/background
// tiering; this cap exists solely so a single pathological caption cannot balloon to the point that NovelAI
// drops or smears the subject. It is set well above any healthy caption (a detailed focus subject runs
// ~25-35 authored tags before local outfit/identity injection), trims from the TAIL (the contract orders the
// defining action/pose/expression tags first, so overflow is the least important), and never merges or drops
// a character. Injected outfit/identity tags are added after this and are never trimmed.
const MAX_AUTHORED_CHARACTER_PROMPT_TAGS = 45;

function composeCharacterPrompt(
  state: AppState,
  prompt: ImageCueCharacterPrompt,
  extraLeadingTags: string[]
): ImageCueCharacterPrompt {
  const llmTags = promptToTags(prompt.prompt).slice(0, MAX_AUTHORED_CHARACTER_PROMPT_TAGS);
  // Only the saved IDENTITY (hair/eyes/face/body) is locally injected so the visible character stays the
  // right person. Outfit/exposure tags are authored by the LLM based on composition and current action —
  // the stored default outfit and outfit_keyword_mappings are surfaced to the LLM as reference only, never
  // re-injected here, because forced injection conflicted with action-specific framing (e.g. the saved
  // garment leaking into a fully nude close-up, or a default uniform overriding a state-changed outfit).
  const { identity } = resolveRegisteredCharacterTags(state, prompt.characterId);

  // Cross-gender subject identity guard: the image-cue LLM sometimes mis-attaches a registered
  // character's id to an UNREGISTERED figure of the opposite gender — most commonly a male NPC
  // (enemy, aggressor, bystander) described with male-subject tags while the attached character_id
  // belongs to a registered female/futanari character. When that happens, injecting the saved
  // female identity (1girl, hair colour, futanari, etc.) and outfit onto the male NPC's caption
  // produces a garbled cross-gender image. Guard: if llmTags carry a clear male-subject assertion
  // AND the registered identity carries a clear female-subject assertion (or vice versa), the LLM
  // effectively described a different, unregistered person — skip identity, outfit, and continuity
  // injection entirely so llmTags alone reach NovelAI and the unregistered figure renders correctly.
  // The guard fires only when BOTH sides carry an explicit gendered signal; neutral or ambiguous
  // captions (no gender signal on either side) fall through to the normal injection path.
  if (
    (hasMaleSubjectAssertion(llmTags) && hasFemaleSubjectAssertion(identity)) ||
    (hasFemaleSubjectAssertion(llmTags) && hasMaleSubjectAssertion(identity))
  ) {
    const composedTags = uniqueStrings([...llmTags, ...extraLeadingTags]);
    return { ...prompt, prompt: composedTags.join(", ") };
  }

  // Position + physical-detail continuity: fill the persisted ongoing posture (pose/action/interaction) and physical
  // details (injury, blood, bodily fluids, sweat/wetness, held prop) that this cut omitted, so the figure keeps
  // performing the scene's action and keeps its established body/clothing-damage state instead of silently resetting.
  // Any explicit posture or per-detail tag the LLM wrote always wins (see resolvePersistedCharacterContinuityTags).
  const continuityTags = prompt.characterId
    ? resolvePersistedCharacterContinuityTags(state, prompt.characterId, llmTags)
    : [];
  // Outfit continuity: inject the character's CURRENT outfit (latest Wearing state, falling back to the saved
  // default outfit) so clothing stays the same across turns instead of drifting whenever the LLM re-describes it.
  // Skipped only when this cut is explicitly fully nude. Wins of state-changed outfits are preserved because the
  // current Wearing memory — not the static default — is the source. uniqueStrings dedups any overlap with llmTags.
  const outfit =
    prompt.characterId && !assertsFullNudity(llmTags)
      ? resolveCurrentCharacterOutfit(state, prompt.characterId)
      : [];
  // Gender/subject tags lead the caption. Identity used to be appended last, so in a long caption (up to 45
  // authored tags plus continuity and outfit) the one tag that tells NovelAI this figure is a man or a woman
  // sat at the very end, where it is weakest — a reliable way to get a male caption rendered as a second
  // female. The rest of the identity (hair/eyes/face/body) stays at the tail as before.
  const { subject: identitySubject, rest: identityRest } = splitSubjectIdentityTags(identity);
  const composedTags = uniqueStrings([
    ...identitySubject,
    ...llmTags,
    ...continuityTags,
    ...extraLeadingTags,
    ...outfit,
    ...identityRest
  ]);
  return {
    ...prompt,
    prompt: composedTags.join(", ")
  };
}

/** Splits saved identity tags into the subject/gender assertions and everything else. */
function splitSubjectIdentityTags(identity: string[]): { subject: string[]; rest: string[] } {
  const subject: string[] = [];
  const rest: string[] = [];
  for (const tag of identity) {
    if (hasMaleSubjectAssertion([tag]) || hasFemaleSubjectAssertion([tag])) {
      subject.push(tag);
    } else {
      rest.push(tag);
    }
  }
  return { subject, rest };
}

function resolveCurrentCharacterOutfit(state: AppState, characterId: string): string[] {
  // Latest in-scene Wearing state wins; fall back to the registered default outfit so a registered character
  // is never rendered in a random or missing outfit.
  for (let index = state.memoryEvents.length - 1; index >= 0; index -= 1) {
    const event = state.memoryEvents[index];
    if (readStateMemoryOwnerId(event) !== characterId) {
      continue;
    }
    if (canonicalizeStateType(readStateMemoryStateType(event)) !== "Wearing") {
      continue;
    }
    const value = readStateMemoryValue(event);
    if (value) {
      return promptToTags(value);
    }
    break;
  }
  const profile = state.visualProfiles.find((candidate) => candidate.characterId === characterId);
  return profile?.defaultOutfitPrompt ? promptToTags(profile.defaultOutfitPrompt) : [];
}

function assertsFullNudity(tags: string[]): boolean {
  return tags.some((tag) =>
    /\b(?:completely nude|fully nude|stark naked|totally naked|naked|nude|nakedness|bare body|no clothes|undressed|nothing)\b/iu.test(
      stripNovelAiTagWeight(tag)
    )
  );
}

/**
 * Extracts individual lowercase text fragments from a single tag, stripping NAI weight syntax
 * and splitting composite weighted bodies so that checks can operate on bare terms.
 * Examples:
 *   "male soldier"      → ["male soldier"]
 *   "1.5::futanari::"   → ["futanari"]
 *   "3::slender, skinny::" → ["slender", "skinny"]
 */
function extractInnerTagTexts(tag: string): string[] {
  const inner = stripNovelAiTagWeight(tag); // already lowercased by stripNovelAiTagWeight
  return inner
    .split(/[,;]\s*/u)
    .map((part) => part.trim())
    .filter(Boolean);
}

/**
 * Returns true when the given tags carry a clear male-subject assertion.
 * Uses word-boundary matching so "female" is never mis-detected as a male signal.
 * Handles both plain tags ("male soldier") and NAI-weighted tags ("1.5::1boy::").
 */
function hasMaleSubjectAssertion(tags: string[]): boolean {
  return tags.some((tag) =>
    extractInnerTagTexts(tag).some((text) =>
      /\b(?:male|1boy|2boys|3boys|4boys|boys?|man|men|male\s+focus|male\s+reader|bishounen|guy)\b/iu.test(text)
    )
  );
}

/**
 * Returns true when the given tags carry a clear female-subject assertion, including futanari
 * (treated as female body per the image-pipeline identity rules).
 * Handles both plain tags and NAI-weighted tags.
 */
function hasFemaleSubjectAssertion(tags: string[]): boolean {
  return tags.some((tag) =>
    extractInnerTagTexts(tag).some((text) =>
      /\b(?:1girl|2girls|3girls|4girls|girls?|woman|women|female|futanari|futa)\b/iu.test(text)
    )
  );
}

function resolveRegisteredCharacterTags(
  state: AppState,
  characterId?: string
): { identity: string[] } {
  if (!characterId) {
    return { identity: [] };
  }

  const profile = state.visualProfiles.find((candidate) => candidate.characterId === characterId);
  if (!profile) {
    return { identity: [] };
  }

  return { identity: promptToTags(profile.positivePrompt) };
}

function rebalanceCharacterCenters(prompts: ImageCueCharacterPrompt[]): ImageCueCharacterPrompt[] {
  if (prompts.length <= 1) {
    return prompts;
  }

  return prompts.map((prompt, index) => ({
    ...prompt,
    center: prompt.center ?? createDefaultCharacterCenter(index, prompts.length)
  }));
}

function normalizeCueCharacterPrompt(prompt: ImageCueCharacterPrompt, index: number): ImageCueCharacterPrompt | undefined {
  const tags = promptToTags(prompt.prompt);
  if (tags.length === 0) {
    return undefined;
  }

  return {
    characterId: prompt.characterId,
    prompt: tags.join(", "),
    negativePrompt: prompt.negativePrompt ? promptToTags(prompt.negativePrompt).join(", ") : undefined,
    center: prompt.center ?? createDefaultCharacterCenter(index, 1)
  };
}

function createDefaultCharacterCenter(index: number, total: number): { x: number; y: number } {
  return {
    x: total <= 1 ? 0.5 : (index + 1) / (total + 1),
    y: 0.5
  };
}

function createContextTags(state: AppState, cue: ImageCue, hasCharacterPrompts = false): string[] {
  const baseSource = uniqueStrings([...(cue.baseTags ?? []), ...(cue.tags ?? [])]);
  const contextTags = uniqueStrings(baseSource.flatMap((tag) => promptToTags(tag)))
    .filter((tag) => !shouldRouteToNegativePrompt(tag))
    .filter((tag) => !hasCharacterPrompts || !isCharacterPromptTag(tag));

  // Scene continuity: only when this cut establishes NO scene/place itself, fall back to the last persisted
  // scene/location/environment so the established setting carries instead of silently resetting. An explicit
  // scene in the cut always wins, so a real scene change is never blocked.
  const groups = classifyImageStateTags(contextTags);
  const cutEstablishesScene = Boolean(groups.SceneTags?.length || groups.EnvironmentTags?.length || groups.Location?.length);
  if (cutEstablishesScene) {
    return contextTags;
  }

  const persistedScene = resolvePersistedSceneTags(state);
  return persistedScene.length > 0 ? uniqueStrings([...contextTags, ...persistedScene]) : contextTags;
}

// Last persisted scene-owned scene/location/environment tags, newest first per type. Used only as a fallback when a
// cut does not establish its own scene, so an ongoing setting persists across cuts/turns.
function resolvePersistedSceneTags(state: AppState): string[] {
  const wanted = new Set(["SceneTags", "EnvironmentTags", "Location"]);
  const found = new Map<string, string[]>();
  for (let index = state.memoryEvents.length - 1; index >= 0 && found.size < wanted.size; index -= 1) {
    const event = state.memoryEvents[index];
    if (readStateMemoryOwnerId(event)) {
      continue;
    }
    const stateType = canonicalizeStateType(readStateMemoryStateType(event));
    if (!stateType || !wanted.has(stateType) || found.has(stateType)) {
      continue;
    }
    const value = readStateMemoryValue(event);
    if (value) {
      found.set(stateType, promptToTags(value));
    }
  }
  return uniqueStrings([...found.values()].flat());
}

// Posture state types describe a single, mutually-exclusive body position. If THIS cut authored any posture tag
// (pose / action / interaction) the figure's position is being set explicitly, so none of the persisted posture is
// re-injected and a real position change is never overridden.
const POSTURE_CONTINUITY_STATE_TYPES = ["InteractionTags", "ActionTags", "PoseTags"] as const;
// Detail state types are additive, independent ongoing physical facts — an injury, blood, bodily fluids, sweat/wetness,
// or a held prop. Each persists on its own across cuts/turns until the scene changes it, so each is carried per-type and
// only when this cut authored no tag of that same type (an explicit change to one detail never drops the others).
const DETAIL_CONTINUITY_STATE_TYPES = ["PhysicalStateTags", "BodyStateTags", "HeldItemTags"] as const;

// Persisted ongoing posture + physical-detail tags for one character, used to fill a cut that omits them so the figure
// keeps the scene's established position and visible body/clothing-damage details (the "floating body part on a neutral
// standing figure", or "the bleeding lip vanished in the close-up" cases). Explicit cut tags always win: posture is
// skipped entirely when the cut authored any posture; each detail type is skipped when the cut authored that type.
function resolvePersistedCharacterContinuityTags(state: AppState, characterId: string, cutTags: string[]): string[] {
  const cutGroups = classifyImageStateTags(cutTags);
  const cutHasPosture = POSTURE_CONTINUITY_STATE_TYPES.some((stateType) => (cutGroups[stateType]?.length ?? 0) > 0);
  const wanted = new Set<string>([
    ...(cutHasPosture ? [] : POSTURE_CONTINUITY_STATE_TYPES),
    ...DETAIL_CONTINUITY_STATE_TYPES.filter((stateType) => (cutGroups[stateType]?.length ?? 0) === 0)
  ]);
  if (wanted.size === 0) {
    return [];
  }
  return resolvePersistedCharacterStateTags(state, characterId, wanted);
}

// Last persisted value of each wanted state type for one character, newest first per type.
function resolvePersistedCharacterStateTags(state: AppState, characterId: string, wanted: Set<string>): string[] {
  const found = new Map<string, string[]>();
  for (let index = state.memoryEvents.length - 1; index >= 0 && found.size < wanted.size; index -= 1) {
    const event = state.memoryEvents[index];
    if (readStateMemoryOwnerId(event) !== characterId) {
      continue;
    }
    const stateType = canonicalizeStateType(readStateMemoryStateType(event));
    if (!stateType || !wanted.has(stateType) || found.has(stateType)) {
      continue;
    }
    const value = readStateMemoryValue(event);
    if (value) {
      found.set(stateType, promptToTags(value));
    }
  }
  return uniqueStrings([...found.values()].flat());
}

function isCharacterPromptTag(tag: string): boolean {
  const normalized = stripNovelAiTagWeight(tag).replace(/[._-]+/gu, " ").trim();
  return (
    isCharacterIdentityReusableTag(normalized) ||
    isPoseCharacterPromptTag(normalized) ||
    isBodyFocusCharacterPromptTag(normalized) ||
    isExpressionOrConditionReusableTag(normalized) ||
    isClothingStateReusableTag(normalized) ||
    /\b(?:wearing|outfit|expression|face focus|body focus|eyes?|mouth|smile|crying|blush|sweat)\b/iu.test(normalized)
  );
}

function isPoseCharacterPromptTag(tag: string): boolean {
  return /^(?:standing|sitting|lying|kneeling|crouching|leaning|bending|arm up|arms up|hand up|hands up|spread legs|legs apart|thighs apart|crossed arms|hands on hips|looking back)$/iu.test(tag);
}

function isBodyFocusCharacterPromptTag(tag: string): boolean {
  return /^(?:face focus|body focus|chest focus|breast focus|thigh focus|leg focus|feet focus|arm focus|back focus|pov hands)$/iu.test(tag);
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
  const normalized = normalizePromptTag(value);
  return normalized ? [normalized] : [];
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
  const filtered = candidates;
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

function stripNovelAiTagWeight(tag: string): string {
  return parseNovelAiWeightedTag(tag)?.tag.toLowerCase().trim() ?? tag.toLowerCase().trim();
}

function createReusableImageTags(cue: ImageCue, promptPlan: ImagePromptPlan, promptVariants: ImagePromptVariant[]): string[] {
  void promptVariants;
  return uniqueReusableTags([
    ...createReusableTagsFromCue(cue),
    ...promptPlan.layers.context,
    ...promptPlan.layers.characters
  ]);
}

function createReusableTagsFromCue(cue: ImageCue): string[] {
  return filterContextConditionedTagsForCue(uniqueReusableTags([
    ...cue.tags,
    ...(cue.baseTags ?? []),
    ...(cue.characterPrompts ?? []).flatMap((prompt) => prompt.prompt.split(","))
  ]), cue);
}

function readReusableTagsFromPromptLayers(value: unknown): string[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return [];
  }

  const layers = value as Partial<ImagePromptLayers>;
  return uniqueReusableTags([
    ...readStringArray(layers.context),
    ...readStringArray(layers.characters),
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
    ...readStringArray(cue.baseTags),
    ...(Array.isArray(cue.characterPrompts) ? cue.characterPrompts.flatMap((prompt) => prompt.prompt.split(",")) : [])
  ]);
}

function readStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string" && item.trim().length > 0) : [];
}

function isImageCueCharacterPrompt(value: unknown): value is ImageCueCharacterPrompt {
  return Boolean(
    value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      typeof (value as ImageCueCharacterPrompt).prompt === "string" &&
      (value as ImageCueCharacterPrompt).prompt.trim()
  );
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
    safetyLevel: resolveEffectiveImageSafetyLevel(state),
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
  const normalized = (cue.scene ?? "").replace(/\s+/gu, " ").trim();
  // Treat DynamicChat's internal placeholder scene labels as "no real scene name" so they never leak into a
  // visible caption. The old regex only caught "current scene" and missed "current simulation scene" (the
  // actual default), so titles surfaced as "current simulation scene #1" in the chat.
  const isPlaceholder =
    !normalized ||
    /^(?:(?:current|generated|safe)(?:\s+simulation)?\s+scene|simulation\s+scene|generated\s+image|scene)$/iu.test(
      normalized
    );
  const scene = isPlaceholder ? "Generated image" : normalized;
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
    baseTags: Array.isArray(cue?.baseTags) ? cue.baseTags : undefined,
    characterPrompts: Array.isArray(cue?.characterPrompts) ? cue.characterPrompts.filter(isImageCueCharacterPrompt) : undefined,
    scene: typeof cue?.scene === "string" ? cue.scene : "generated scene",
    suppressionReason: typeof cue?.suppressionReason === "string" ? cue.suppressionReason : undefined,
    visualContext: typeof cue?.visualContext === "string" ? cue.visualContext : undefined
  };
}
