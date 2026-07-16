import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const vite = await createServer({
  root: workspaceRoot,
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom"
});

const checks = [];

try {
  const { seedState } = await vite.ssrLoadModule("/src/data/seed.ts");
  const { planImageJobForCompletedTurn, runSimulationTurn } = await vite.ssrLoadModule("/src/services/simulationEngine.ts");
  const { findReusableImageAsset, getReusableTagsFromJob, pickStoredAsset, planImageJob } = await vite.ssrLoadModule("/src/services/imageOrchestrator.ts");
  const { generateNovelAiImages } = await vite.ssrLoadModule("/src/services/novelAiClient.ts");
  const { compileSimulationMemoryDelta, memoryDeltaToEvents } = await vite.ssrLoadModule("/src/services/memoryCompiler.ts");

  await evaluateRecentContextTagDetection(seedState, runSimulationTurn);
  evaluateNovelAiWeightingAndUserRules(seedState, planImageJob);
  await evaluateReusableImageMatching(seedState, planImageJob, findReusableImageAsset, getReusableTagsFromJob, pickStoredAsset, planImageJobForCompletedTurn);
  await evaluateImageUserRuleCuePlanning(seedState, planImageJobForCompletedTurn);
  await evaluateImageGenerationCadence(seedState, planImageJobForCompletedTurn);
  await evaluateAnchoredRulePromptDiversity(seedState, planImageJobForCompletedTurn);
  evaluateAdultContentRating(seedState, planImageJob);
  await evaluateSceneSpecificity(seedState, runSimulationTurn);
  await evaluateNoisyCueTagFiltering(seedState, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateImageCueCharacterScoping(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateLlmCharacterScopedOutfits(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages);
  evaluateWearingStateDetailPreservation(seedState, compileSimulationMemoryDelta, memoryDeltaToEvents);
  await evaluateStructuredCueDisambiguation(seedState, planImageJob, planImageJobForCompletedTurn);
  await evaluatePersonaCharacterImageCueScoping(seedState, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateLlmImageStateTagCarryover(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateImageDetailStateContinuity(seedState, planImageJobForCompletedTurn);
  await evaluateLlmNaiTagPreservation(seedState, planImageJob, generateNovelAiImages);
  evaluateNameAndBodyInventoryCleanup(seedState, planImageJob);
  await evaluateActorTargetCharacterDisambiguation(seedState, planImageJob, planImageJobForCompletedTurn);
  await evaluateFallbackImageCueSuppression(seedState, runSimulationTurn);
  await evaluateNovelAiV4Payload(seedState, planImageJob, generateNovelAiImages);
  await evaluateRegisteredCharacterPromptInjection(seedState, planImageJob, generateNovelAiImages);
  await evaluateNovelAiVibeTransferPayload(seedState, planImageJob, generateNovelAiImages);
} finally {
  await vite.close();
}

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}`);
}

const failures = checks.filter((check) => check.level === "fail");
console.log("");
console.log(`Image prompt quality checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

async function evaluateRecentContextTagDetection(seedState, runSimulationTurn) {
  const state = createPlayableState(seedState, {
    messages: [
      createAssistantMessage(
        seedState,
        "Aria stands inside a quiet archive library, silver hair visible, brass key pendant beside tall bookshelves."
      )
    ]
  });
  const result = await runSimulationTurn(state, "이 장면을 이미지로 보여줘", true);
  const job = result.imageJob;
  const prompt = job?.prompt ?? "";
  const characterLayer = job?.providerPayload.promptLayers?.characters?.join(", ") ?? "";
  const cueText = [result.imageCue.scene, result.imageCue.tags.join(", "), result.imageCue.visualContext].join(", ");

  assertCheck("context.library", result.imageCue.scene === "current simulation scene", "Recent transcript does not locally infer an archive-library scene.");
  assertCheck("context.library", !/library|bookshelf/iu.test(cueText), "Deictic image requests do not locally convert transcript context into tags.");
  assertCheck("context.library", !/\b(?:rain|food|table|dining table)\b/iu.test(prompt), "Archive context does not auto-inject old rain/food/table tags.");
  assertCheck("context.library", !/silver hair/iu.test(characterLayer), "Character profiles are not added without an LLM-provided character id.");
  assertCheck("context.library", !/silver hair/iu.test(prompt), "NovelAI V4 base prompt keeps character prompt tags out of the generated scene prompt.");
  assertCheck("context.library", !/\b1(?:girl|boy|other)\b/iu.test(prompt), "Subject count tags are not inferred by app-side character mapping.");
}

async function evaluateReusableImageMatching(seedState, planImageJob, findReusableImageAsset, getReusableTagsFromJob, pickStoredAsset, planImageJobForCompletedTurn) {
  const state = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      artistPrompt: "::artist:sample_artist ::",
      qualityPrompt: "masterpiece, best quality, highly detailed skin"
    }
  });
  // Real generated assets carry their rendered size; reuse is gated on the asset size matching the configured
  // resolution (assetMatchesConfiguredResolution), so stamp the seed resolution onto every reuse fixture.
  const reuseDims = { width: state.imageProfile.width, height: state.imageProfile.height };
  const cue = createCue(state, {
    scene: "archive library",
    tags: ["archive library", "bookshelf", "silver hair", "brass key"],
    visualContext: "archive library, tall bookshelf, silver hair, brass key"
  });
  const imageAssets = [
    {
      id: "asset_reuse_library",
      simulationId: state.simulation.id,
      title: "Reusable archive cut",
      source: "generated",
      prompt: "archive library, bookshelf, silver hair, brass key",
      negativePrompt: "",
      safetyLevel: "safe",
      characterIds: cue.characters,
      tags: ["archive library", "bookshelf", "silver hair", "brass key"],
      reuseTags: ["archive library", "bookshelf", "silver hair", "brass key"],
      createdAt: new Date().toISOString(),
      palette: ["#213547", "#d5e1e8", "#f2b84b"],
      dataUrl: "data:image/png;base64,AA==",
      providerMetadata: { ...reuseDims },
      representative: true
    },
    {
      id: "asset_reuse_stage",
      simulationId: state.simulation.id,
      title: "Different stage cut",
      source: "generated",
      prompt: "stage, stage lights, dancing",
      negativePrompt: "",
      safetyLevel: "safe",
      characterIds: cue.characters,
      tags: ["stage", "stage lights", "dancing"],
      reuseTags: ["stage", "stage lights", "dancing"],
      createdAt: new Date(Date.now() + 1).toISOString(),
      palette: ["#213547", "#d5e1e8", "#f2b84b"],
      dataUrl: "data:image/png;base64,AA==",
      providerMetadata: { ...reuseDims }
    }
  ];
  const reuseState = { ...state, imageAssets };
  const job = planImageJob(reuseState, "turn_reuse", cue, [], false);
  const match = findReusableImageAsset(reuseState, job);
  const differentSceneJob = planImageJob(
    reuseState,
    "turn_reuse_different_scene",
    createCue(state, {
      scene: "stage performance",
      tags: ["stage", "stage lights", "silver hair", "brass key"],
      visualContext: "stage performance, stage lights, silver hair, brass key"
    }),
    [],
    false
  );
  const differentSceneMatch = findReusableImageAsset(reuseState, differentSceneJob);
  const wrongCharacterMatch = findReusableImageAsset(
    {
      ...reuseState,
      imageAssets: [
        {
          ...imageAssets[0],
          id: "asset_reuse_wrong_character",
          characterIds: ["char_not_in_cue"],
          createdAt: new Date(Date.now() + 2).toISOString()
        }
      ]
    },
    job
  );
  const unscopedCueJob = planImageJob(
    reuseState,
    "turn_reuse_unscoped",
    createCue(state, {
      characters: [],
      scene: "archive library",
      tags: ["archive library", "bookshelf", "silver hair", "brass key"],
      visualContext: "archive library, tall bookshelf, silver hair, brass key"
    }),
    [],
    false
  );
  const unscopedCharacterMatch = findReusableImageAsset(reuseState, unscopedCueJob);
  const actionCue = createCue(state, {
    scene: "archive library",
    tags: ["archive library", "bookshelf", "standing", "holding key", "worried expression"],
    visualContext: "archive library, bookshelf, standing, holding key, worried expression"
  });
  const actionReusableAsset = {
    ...imageAssets[0],
    id: "asset_reuse_action_exact",
    tags: ["1girl", "archive library", "bookshelf", "silver hair", "standing", "holding key", "worried expression"],
    reuseTags: ["1girl", "archive library", "bookshelf", "silver hair", "standing", "holding key", "worried expression"],
    providerMetadata: {
      ...reuseDims,
      cue: {
        characters: actionCue.characters,
        scene: "archive library",
        tags: ["1girl", "silver hair", "standing", "holding key", "worried expression"],
        visualContext: "1girl, archive library, bookshelf, silver hair, standing, holding key, worried expression"
      }
    }
  };
  const countedActionCue = {
    ...actionCue,
    tags: ["1girl", "archive library", "bookshelf", "silver hair", "standing", "holding key", "worried expression"],
    visualContext: "1girl, archive library, bookshelf, silver hair, standing, holding key, worried expression"
  };
  const actionJob = planImageJob({ ...reuseState, imageAssets: [actionReusableAsset] }, "turn_reuse_action_exact", countedActionCue, [], false);
  const exactActionMatch = findReusableImageAsset({ ...reuseState, imageAssets: [actionReusableAsset] }, actionJob);
  const mismatchedPoseMatch = findReusableImageAsset(
    {
      ...reuseState,
      imageAssets: [
        {
          ...actionReusableAsset,
          id: "asset_reuse_pose_mismatch",
          tags: ["1girl", "archive library", "bookshelf", "silver hair", "sitting", "holding key", "worried expression"],
          reuseTags: ["1girl", "archive library", "bookshelf", "silver hair", "sitting", "holding key", "worried expression"],
          providerMetadata: {
            ...reuseDims,
            cue: {
              characters: countedActionCue.characters,
              scene: "archive library",
              tags: ["1girl", "silver hair", "sitting", "holding key", "worried expression"],
              visualContext: "1girl, archive library, bookshelf, silver hair, sitting, holding key, worried expression"
            }
          }
        }
      ]
    },
    actionJob
  );
  const mismatchedStatusMatch = findReusableImageAsset(
    {
      ...reuseState,
      imageAssets: [
        {
          ...actionReusableAsset,
          id: "asset_reuse_status_mismatch",
          tags: ["1girl", "archive library", "bookshelf", "silver hair", "standing", "holding key", "smile"],
          reuseTags: ["1girl", "archive library", "bookshelf", "silver hair", "standing", "holding key", "smile"],
          providerMetadata: {
            ...reuseDims,
            cue: {
              characters: countedActionCue.characters,
              scene: "archive library",
              tags: ["1girl", "silver hair", "standing", "holding key", "smile"],
              visualContext: "1girl, archive library, bookshelf, silver hair, standing, holding key, smile"
            }
          }
        }
      ]
    },
    actionJob
  );
  const mismatchedCountMatch = findReusableImageAsset(
    {
      ...reuseState,
      imageAssets: [
        {
          ...actionReusableAsset,
          id: "asset_reuse_count_mismatch",
          tags: ["2girls", "archive library", "bookshelf", "silver hair", "standing", "holding key", "worried expression"],
          reuseTags: ["2girls", "archive library", "bookshelf", "silver hair", "standing", "holding key", "worried expression"],
          characterIds: countedActionCue.characters,
          providerMetadata: {
            ...reuseDims,
            cue: {
              characters: countedActionCue.characters,
              scene: "archive library",
              tags: ["2girls", "silver hair", "standing", "holding key", "worried expression"],
              visualContext: "2girls, archive library, bookshelf, silver hair, standing, holding key, worried expression"
            }
          }
        }
      ]
    },
    actionJob
  );
  const mismatchedCharacterTagMatch = findReusableImageAsset(
    {
      ...reuseState,
      imageAssets: [
        {
          ...actionReusableAsset,
          id: "asset_reuse_character_tag_mismatch",
          tags: ["1girl", "archive library", "bookshelf", "black hair", "standing", "holding key", "worried expression"],
          reuseTags: ["1girl", "archive library", "bookshelf", "black hair", "standing", "holding key", "worried expression"],
          providerMetadata: {
            ...reuseDims,
            cue: {
              characters: countedActionCue.characters,
              scene: "archive library",
              tags: ["1girl", "black hair", "standing", "holding key", "worried expression"],
              visualContext: "1girl, archive library, bookshelf, black hair, standing, holding key, worried expression"
            }
          }
        }
      ]
    },
    actionJob
  );
  const extraActionMatch = findReusableImageAsset(
    {
      ...reuseState,
      imageAssets: [
        {
          ...actionReusableAsset,
          id: "asset_reuse_extra_action",
          tags: ["1girl", "archive library", "bookshelf", "silver hair", "standing", "holding key", "holding phone", "worried expression"],
          reuseTags: ["1girl", "archive library", "bookshelf", "silver hair", "standing", "holding key", "holding phone", "worried expression"],
          providerMetadata: {
            ...reuseDims,
            cue: {
              characters: countedActionCue.characters,
              scene: "archive library",
              tags: ["1girl", "silver hair", "standing", "holding key", "holding phone", "worried expression"],
              visualContext: "1girl, archive library, bookshelf, silver hair, standing, holding key, holding phone, worried expression"
            }
          }
        }
      ]
    },
    actionJob
  );
  const mismatchedMetadataCharacterMatch = findReusableImageAsset(
    {
      ...reuseState,
      imageAssets: [
        {
          ...actionReusableAsset,
          id: "asset_reuse_metadata_character_mismatch",
          providerMetadata: {
            ...reuseDims,
            cue: {
              characters: ["char_not_in_cue"],
              scene: "archive library",
              tags: ["standing", "holding key", "worried expression"],
              visualContext: "archive library, bookshelf, standing, holding key, worried expression"
            }
          }
        }
      ]
    },
    actionJob
  );
  const exactStoredAsset = {
    ...actionReusableAsset,
    id: "asset_stored_action_exact",
    source: "stored",
    createdAt: new Date(Date.now() + 3).toISOString()
  };
  const staleStoredAsset = {
    ...exactStoredAsset,
    id: "asset_stored_stale_action",
    tags: ["1girl", "archive library", "bookshelf", "silver hair", "sitting", "holding key", "worried expression"],
    reuseTags: ["1girl", "archive library", "bookshelf", "silver hair", "sitting", "holding key", "worried expression"],
    providerMetadata: {
      ...reuseDims,
      cue: {
        characters: countedActionCue.characters,
        scene: "archive library",
        tags: ["1girl", "silver hair", "sitting", "holding key", "worried expression"],
        visualContext: "1girl, archive library, bookshelf, silver hair, sitting, holding key, worried expression"
      }
    }
  };
  const storedActionMatch = pickStoredAsset({ ...reuseState, imageAssets: [staleStoredAsset, exactStoredAsset] }, countedActionCue);
  const staleStoredOnlyMatch = pickStoredAsset({ ...reuseState, imageAssets: [staleStoredAsset] }, countedActionCue);
  const reusableTags = getReusableTagsFromJob(job);
  const userMessage = createUserMessage(state, "기록 보관소 장면을 다시 보여줘");
  const assistantMessage = createAssistantMessage(state, "The archive library returns, with silver hair and brass key by the bookshelf.");
  const sidecarCue = {
    shouldGenerate: true,
    reason: "reuse eval",
    characters: cue.characters,
    tags: cue.tags,
    scene: cue.scene,
    visualContext: cue.visualContext
  };
  const autoReusePlan = await planImageJobForCompletedTurn(
    {
      ...reuseState,
      imageProfile: {
        ...reuseState.imageProfile,
        triggerMode: "realtime_auto",
        cooldownTurns: 0
      },
      messages: [...reuseState.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: sidecarCue,
        imageCues: [sidecarCue]
      },
      sidecarTrace: createParsedLlmSidecarTrace(state, userMessage.id),
      manualImage: false
    }
  );

  assertCheck("image.reuse", reusableTags.includes("archive library") && reusableTags.includes("bookshelf"), "Image jobs store reusable context tags.");
  assertCheck("image.reuse", !reusableTags.some((tag) => /\b(?:rain|food|table|dining table)\b/iu.test(tag)), "Reusable context tags do not auto-inject old rain/food/table tags.");
  assertCheck("image.reuse", !reusableTags.some((tag) => /artist|masterpiece|best quality/iu.test(tag)), "Reusable tags exclude artist and quality tags.");
  assertCheck("image.reuse", match?.asset.id === "asset_reuse_library", "Matching generated image is selected for high-overlap context reuse.");
  assertCheck("image.reuse", !differentSceneMatch, "Generated images are not reused for a different scene with only character/prop overlap.");
  assertCheck("image.reuse", !wrongCharacterMatch, "Generated images are not reused when the character scope does not match exactly.");
  assertCheck("image.reuse", !unscopedCharacterMatch, "Unscoped image cues do not reuse character-scoped generated images.");
  assertCheck("image.reuse", exactActionMatch?.asset.id === "asset_reuse_action_exact", "Exact character, pose, and status tags can still reuse a generated image.");
  assertCheck("image.reuse", !mismatchedPoseMatch, "Generated images are not reused when required pose/action tags differ.");
  assertCheck("image.reuse", !mismatchedStatusMatch, "Generated images are not reused when required expression/status tags differ.");
  assertCheck("image.reuse", !mismatchedCountMatch, "Generated images are not reused when the visible person count differs.");
  assertCheck("image.reuse", !mismatchedCharacterTagMatch, "Generated images are not reused when character identity tags differ.");
  assertCheck("image.reuse", !extraActionMatch, "Generated images are not reused when the stored action has extra unmatched action tags.");
  assertCheck("image.reuse", !mismatchedMetadataCharacterMatch, "Generated image reuse rejects provider metadata character-scope mismatches.");
  assertCheck("image.reuse", storedActionMatch?.id === "asset_stored_action_exact", "Stored image fallback only selects assets with matching character, count, and action tags.");
  assertCheck("image.reuse", !staleStoredOnlyMatch, "Stored image fallback rejects stale assets whose action tags do not match.");
  assertCheck("image.reuse", autoReusePlan.imageJobs.length === 0 && autoReusePlan.reusedAssetIds.includes("asset_reuse_library"), "Image planning reuses a high-overlap generated asset instead of queuing NovelAI.");
}

async function evaluateImageUserRuleCuePlanning(seedState, planImageJobForCompletedTurn) {
  const state = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto",
      cooldownTurns: 99,
      userRules: [
        "각 문맥마다 새 이미지 생성. 기존 이미지는 재사용하지 않는다.",
        "대사 전 얼굴 컷을 별도 이미지로 만든다."
      ].join("\n")
    }
  });
  const existingAsset = {
    id: "asset_existing_archive",
    simulationId: state.simulation.id,
    title: "Existing archive cut",
    source: "generated",
    prompt: "archive library, rain, silver hair, brass key",
    negativePrompt: "",
    safetyLevel: "safe",
    characterIds: [state.characters[0].id],
    tags: ["archive library", "rain", "silver hair", "brass key"],
    reuseTags: ["archive library", "rain", "silver hair", "brass key"],
    createdAt: new Date().toISOString(),
    palette: ["#213547", "#d5e1e8", "#f2b84b"],
    dataUrl: "data:image/png;base64,AA=="
  };
  const userMessage = createUserMessage(state, "비 오는 기록 보관소 장면을 이어가");
  const assistantMessage = createAssistantMessage(
    state,
    [
      "Aria lifts the brass key beside the rain-soaked bookshelf.",
      "\"문을 열게요.\""
    ].join("\n")
  );
  const noImageCue = {
    shouldGenerate: false,
    reason: "llm skipped image cues",
    characters: [],
    tags: [],
    scene: "current scene",
    suppressionReason: "llm skipped"
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      imageAssets: [existingAsset],
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: noImageCue,
        imageCues: []
      },
      manualImage: false
    }
  );
  const cueKinds = plan.imageJobs.map((job) => job.providerPayload.cueKind);
  const plannerSources = plan.imageJobs.map((job) => job.providerPayload.imageCuePlanner?.source);

  assertCheck("image.user_rules", plan.imageJobs.length === 0, "User image rules do not synthesize local NAI tags when the LLM omits image_cues.");
  assertCheck("image.user_rules", cueKinds.length === 0, "Rule-backed local cue hints stay non-rendered without LLM-authored tags.");
  assertCheck("image.user_rules", plan.reusedAssetIds.length === 0, "Fresh-image user rules prevent generated asset reuse.");
  assertCheck("image.user_rules", plan.imageJobs.every((job) => job.providerPayload.forceFreshImage === true), "Rule-backed jobs are marked as fresh image jobs.");
  assertCheck("image.user_rules", plannerSources.every((source) => source === "user_image_rules"), "Rule-backed jobs record user-image-rule planner source.");
}

async function evaluateImageGenerationCadence(seedState, planImageJobForCompletedTurn) {
  const paragraphState = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto",
      generationCadence: "paragraph",
      cooldownTurns: 99,
      userRules: "캐릭터의 외형 일관성을 우선한다."
    }
  });
  const paragraphUserMessage = createUserMessage(paragraphState, "문단마다 현재 장면을 이어가");
  const paragraphAssistantMessage = createAssistantMessage(
    paragraphState,
    [
      "Aria steps between the archive shelves while rain streaks down the tall window.",
      "",
      "She raises the brass key, and blue light spills across her wet gloves.",
      "",
      "\"지금이에요.\" Her expression tightens before she touches the locked door."
    ].join("\n")
  );
  const noImageCue = {
    shouldGenerate: false,
    reason: "llm skipped image cues",
    characters: [],
    tags: [],
    scene: "current scene",
    suppressionReason: "llm skipped"
  };
  const paragraphPlan = await planImageJobForCompletedTurn(
    {
      ...paragraphState,
      messages: [...paragraphState.messages, paragraphUserMessage, paragraphAssistantMessage]
    },
    {
      userMessage: paragraphUserMessage,
      assistantMessage: paragraphAssistantMessage,
      contextPack: createContextPack(paragraphState),
      promptModuleUsages: [],
      sidecar: {
        assistantText: paragraphAssistantMessage.content,
        memoryEvents: [],
        imageCue: noImageCue,
        imageCues: []
      },
      manualImage: false
    }
  );

  assertCheck("image.cadence", paragraphPlan.imageJobs.length === 0, "Paragraph cadence does not synthesize local NAI tags when the LLM omits image_cues.");
  assertCheck("image.cadence", paragraphPlan.imageJobs.every((job) => job.providerPayload.generationCadence === "paragraph"), "Image jobs record the active generation cadence.");
  assertCheck("image.cadence", paragraphPlan.imageJobs.every((job) => job.providerPayload.forceFreshImage === true), "Paragraph cadence jobs bypass cooldown/reuse as fresh cuts.");
  assertCheck("image.cadence", paragraphPlan.imageJobs.every((job) => job.providerPayload.imageCuePlanner?.source === "image_generation_cadence"), "Paragraph cadence records cadence planner source only on LLM-tagged jobs.");

  const paragraphLlmCues = [
    {
      kind: "scene",
      anchorText: "archive shelves while rain streaks",
      tags: ["1girl", "wide shot", "archive library", "standing", "rain", "wet gloves", "school uniform", "black hair"]
    },
    {
      kind: "action",
      anchorText: "raises the brass key",
      tags: ["1girl", "upper body", "archive library", "holding key", "blue glow", "wet gloves", "school uniform", "serious expression"]
    },
    {
      kind: "dialogue_face",
      anchorText: "지금이에요",
      tags: ["1girl", "close-up", "archive library", "open mouth", "looking at viewer", "blue glow", "school uniform", "tense expression"]
    }
  ].map((cue, index) => ({
    shouldGenerate: true,
    reason: `paragraph cue ${index + 1}`,
    characters: [paragraphState.characters[0].id],
    scene: "archive library",
    visualContext: cue.tags.join(", "),
    placement: index === 0 ? "before" : "inline",
    priority: 0.92 - index * 0.03,
    label: `paragraph ${index + 1}`,
    ...cue
  }));
  const paragraphLlmPlan = await planImageJobForCompletedTurn(
    {
      ...paragraphState,
      messages: [...paragraphState.messages, paragraphUserMessage, paragraphAssistantMessage]
    },
    {
      userMessage: paragraphUserMessage,
      assistantMessage: paragraphAssistantMessage,
      contextPack: createContextPack(paragraphState),
      promptModuleUsages: [],
      sidecar: {
        assistantText: paragraphAssistantMessage.content,
        memoryEvents: [],
        imageCue: paragraphLlmCues[0],
        imageCues: paragraphLlmCues
      },
      sidecarTrace: createParsedLlmSidecarTrace(paragraphState, paragraphAssistantMessage.id),
      manualImage: false
    }
  );

  assertCheck("image.cadence", paragraphLlmPlan.imageJobs.length === 3, "Paragraph cadence turns LLM-authored paragraph cues into separate image jobs.");
  assertCheck("image.cadence", paragraphLlmPlan.imageJobs.every((job) => job.providerPayload.imageCuePlanner?.source === "main_llm_sidecar"), "Paragraph cadence keeps LLM-authored cue source on generated jobs.");
  assertCheck("image.cadence", paragraphLlmPlan.imageJobs.every((job) => job.providerPayload.forceFreshImage === true), "Paragraph LLM-authored jobs bypass reuse/cooldown as fresh paragraph cuts.");
  assertCheck("image.cadence", paragraphLlmPlan.imageJobs.map((job) => job.providerPayload.cueIndex).join(",") === "0,1,2", "Paragraph LLM-authored jobs preserve cue order and cue indexes.");
  assertCheck(
    "image.cadence",
    paragraphLlmPlan.imageJobs.every((job) => {
      const characterLayer = job.providerPayload.promptLayers?.characters?.join(", ") ?? "";
      return /archive library/iu.test(job.prompt) && !/school uniform/iu.test(job.prompt) && /school uniform/iu.test(characterLayer);
    }),
    "Paragraph LLM-authored jobs split base scene tags and character outfit tags into the correct prompt layers."
  );

  const richState = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto",
      generationCadence: "rich",
      cooldownTurns: 99,
      userRules: "캐릭터의 외형 일관성을 우선한다."
    }
  });
  const richUserMessage = createUserMessage(richState, "주요 행동마다 장면을 이어가");
  const richAssistantMessage = createAssistantMessage(
    richState,
    [
      "Aria pushes through the archive door into a rain-blue reading room.",
      "She crouches beside the fallen brass key and reaches for it with gloved fingers.",
      "The lock flashes; she turns toward the viewer and whispers, \"지금이에요.\"",
      "Then she steps under the skylight as blue sparks scatter over her uniform."
    ].join("\n")
  );
  const richCues = [
    {
      kind: "scene",
      anchorText: "rain-blue reading room",
      tags: ["1girl", "wide shot", "reading room", "rain", "standing", "school uniform", "black hair", "blue lighting"]
    },
    {
      kind: "body_detail",
      anchorText: "gloved fingers",
      tags: ["1girl", "close-up", "hand focus", "gloves", "reaching out", "brass key", "school uniform", "blue lighting"]
    },
    {
      kind: "dialogue_face",
      anchorText: "지금이에요",
      tags: ["1girl", "close-up", "looking at viewer", "open mouth", "parted lips", "school uniform", "blue lighting", "serious expression"]
    },
    {
      kind: "action",
      anchorText: "blue sparks scatter",
      tags: ["1girl", "upper body", "skylight", "standing", "blue sparks", "school uniform", "black hair", "dramatic lighting"]
    }
  ].map((cue, index) => ({
    shouldGenerate: true,
    reason: `rich cue ${index + 1}`,
    characters: [richState.characters[0].id],
    scene: index === 0 ? "reading room" : "archive library",
    visualContext: cue.tags.join(", "),
    placement: index === 0 ? "before" : "inline",
    priority: 0.94 - index * 0.04,
    label: `rich ${index + 1}`,
    ...cue
  }));
  const richPlan = await planImageJobForCompletedTurn(
    {
      ...richState,
      messages: [...richState.messages, richUserMessage, richAssistantMessage]
    },
    {
      userMessage: richUserMessage,
      assistantMessage: richAssistantMessage,
      contextPack: createContextPack(richState),
      promptModuleUsages: [],
      sidecar: {
        assistantText: richAssistantMessage.content,
        memoryEvents: [],
        imageCue: richCues[0],
        imageCues: richCues
      },
      sidecarTrace: createParsedLlmSidecarTrace(richState, richAssistantMessage.id),
      manualImage: false
    }
  );

  assertCheck("image.cadence", richPlan.imageJobs.length === 4, "Rich cadence keeps separate LLM-authored jobs for major visual beats.");
  assertCheck("image.cadence", richPlan.imageJobs.every((job) => job.providerPayload.generationCadence === "rich"), "Rich cadence jobs record the active generation density.");
  assertCheck("image.cadence", richPlan.imageJobs.every((job) => job.providerPayload.forceFreshImage === true), "Rich cadence jobs bypass reuse/cooldown as fresh high-density cuts.");
  assertCheck("image.cadence", richPlan.imageJobs.map((job) => job.providerPayload.cueKind).join(",") === "scene,body_detail,dialogue_face,action", "Rich cadence preserves distinct cue kinds for scene/body/dialogue/action beats.");
  assertCheck(
    "image.cadence",
    richPlan.imageJobs.every((job) => Array.isArray(job.providerPayload.positiveTags) && job.providerPayload.positiveTags.length >= 8),
    "Rich cadence jobs carry complete LLM-authored NAI tag sets."
  );

  const progressionState = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto",
      generationCadence: "image_progression",
      cooldownTurns: 99,
      userRules: "캐릭터의 외형 일관성을 우선한다."
    }
  });
  const progressionUserMessage = createUserMessage(progressionState, "이미지 진행 버전으로 이어가");
  const progressionAssistantMessage = createAssistantMessage(progressionState, "이미지 진행 10컷.");
  const progressionNoCuePlan = await planImageJobForCompletedTurn(
    {
      ...progressionState,
      messages: [...progressionState.messages, progressionUserMessage, progressionAssistantMessage]
    },
    {
      userMessage: progressionUserMessage,
      assistantMessage: progressionAssistantMessage,
      contextPack: createContextPack(progressionState),
      promptModuleUsages: [],
      sidecar: {
        assistantText: progressionAssistantMessage.content,
        memoryEvents: [],
        imageCue: noImageCue,
        imageCues: []
      },
      sidecarTrace: createParsedLlmSidecarTrace(progressionState, progressionAssistantMessage.id),
      manualImage: false
    }
  );
  const progressionKinds = ["scene", "action", "dialogue_face", "body_detail", "interaction"];
  const progressionFrames = ["wide shot", "medium shot", "upper body", "close-up", "over-the-shoulder", "pov"];
  const progressionCues = Array.from({ length: 12 }, (_, index) => {
    const kind = progressionKinds[index % progressionKinds.length];
    const frame = progressionFrames[index % progressionFrames.length];
    const action = index % 4 === 0 ? "standing" : index % 4 === 1 ? "holding key" : index % 4 === 2 ? "reaching out" : "looking at viewer";
    const expression = index % 3 === 0 ? "serious expression" : index % 3 === 1 ? "open mouth" : "determined expression";
    const tags = ["1girl", frame, "archive library", action, "blue glow", "school uniform", "black hair", expression];
    return {
      shouldGenerate: true,
      reason: `progression cue ${index + 1}`,
      characters: [progressionState.characters[0].id],
      tags,
      scene: "archive progression",
      visualContext: tags.join(", "),
      kind,
      placement: index === 0 ? "before" : "inline",
      anchorText: `progression beat ${index + 1}`,
      priority: 0.95 - index * 0.01,
      label: `progression ${index + 1}`
    };
  });
  const progressionPlan = await planImageJobForCompletedTurn(
    {
      ...progressionState,
      messages: [...progressionState.messages, progressionUserMessage, progressionAssistantMessage]
    },
    {
      userMessage: progressionUserMessage,
      assistantMessage: progressionAssistantMessage,
      contextPack: createContextPack(progressionState),
      promptModuleUsages: [],
      sidecar: {
        assistantText: progressionAssistantMessage.content,
        memoryEvents: [],
        imageCue: progressionCues[0],
        imageCues: progressionCues
      },
      sidecarTrace: createParsedLlmSidecarTrace(progressionState, progressionAssistantMessage.id),
      manualImage: false
    }
  );

  assertCheck("image.cadence", progressionNoCuePlan.imageJobs.length === 0, "Image progression mode does not synthesize local NAI tags when the LLM omits image_cues.");
  assertCheck("image.cadence", progressionPlan.imageJobs.length === 10, "Image progression mode turns 10 LLM-authored cue groups into 10 image jobs.");
  assertCheck("image.cadence", progressionPlan.imageJobs.every((job) => job.providerPayload.generationCadence === "image_progression"), "Image progression jobs record the active generation cadence.");
  assertCheck("image.cadence", progressionPlan.imageJobs.every((job) => job.providerPayload.forceFreshImage === true), "Image progression jobs bypass cooldown/reuse as fresh cuts.");
  assertCheck("image.cadence", progressionPlan.imageJobs.map((job) => job.providerPayload.cueIndex).join(",") === Array.from({ length: 10 }, (_, index) => index).join(","), "Image progression jobs preserve all cue indexes in order.");
  assertCheck(
    "image.cadence",
    progressionPlan.imageJobs.every((job) => {
      const characterLayer = job.providerPayload.promptLayers?.characters?.join(", ") ?? "";
      return /archive library/iu.test(job.prompt) && !/school uniform/iu.test(job.prompt) && /school uniform/iu.test(characterLayer);
    }),
    "Image progression jobs split each LLM-authored tag group into base and character prompt layers."
  );

  const sparseState = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto",
      generationCadence: "sparse",
      cooldownTurns: 0,
      userRules: "캐릭터의 외형 일관성을 우선한다."
    }
  });
  const sparseUserMessage = createUserMessage(sparseState, "장면을 이어가");
  const sparseAssistantMessage = createAssistantMessage(sparseState, "Aria moves through the rain-lit archive and raises the brass key.");
  const sparseCues = ["scene", "action", "dialogue_face"].map((kind) => ({
    shouldGenerate: true,
    reason: `${kind} cue`,
    characters: [sparseState.characters[0].id],
    tags: ["archive library", "rain", kind],
    scene: "archive library",
    visualContext: `archive library, rain, ${kind}`,
    kind
  }));
  const sparsePlan = await planImageJobForCompletedTurn(
    {
      ...sparseState,
      messages: [...sparseState.messages, sparseUserMessage, sparseAssistantMessage]
    },
    {
      userMessage: sparseUserMessage,
      assistantMessage: sparseAssistantMessage,
      contextPack: createContextPack(sparseState),
      promptModuleUsages: [],
      sidecar: {
        assistantText: sparseAssistantMessage.content,
        memoryEvents: [],
        imageCue: sparseCues[0],
        imageCues: sparseCues
      },
      manualImage: false
    }
  );

  assertCheck("image.cadence", sparsePlan.imageJobs.length === 1, "Sparse cadence caps LLM image_cues to one job when user rules do not require more.");
}

async function evaluateAnchoredRulePromptDiversity(seedState, planImageJobForCompletedTurn) {
  const state = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto",
      generationCadence: "paragraph",
      cooldownTurns: 99,
      userRules: [
        "각 문맥마다 새 이미지 생성. 기존 이미지는 재사용하지 않는다.",
        "대사나 신음 전에는 얼굴 컷을 넣는다.",
        "신체 부위 강조와 행위 부분에는 별도 컷을 넣는다."
      ].join("\n")
    }
  });
  const userMessage = createUserMessage(state, "무대에서 거리까지 이어지는 장면을 문맥별 이미지로 보여줘");
  const assistantMessage = createAssistantMessage(
    state,
    [
      "무대 조명이 켜지고 Aria가 마이크를 쥔 채 팔을 들어 올린다.",
      "",
      "그녀는 손목을 붙잡고 숨을 고르며 고개를 돌린다.",
      "",
      "\"지금 시작할게요,\" Aria가 속삭인다.",
      "",
      "잠시 뒤, 비 내리는 거리로 걸어나간다."
    ].join("\n")
  );
  const noImageCue = {
    shouldGenerate: false,
    reason: "llm skipped image cues",
    characters: [],
    tags: [],
    scene: "current scene",
    suppressionReason: "llm skipped"
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: noImageCue,
        imageCues: []
      },
      manualImage: false
    }
  );
  const jobs = plan.imageJobs;
  const prompts = jobs.map((job) => job.prompt);
  const byKind = new Map(jobs.map((job) => [job.providerPayload.cueKind, job]));
  const actionJob =
    jobs.find((job) => /stage|stage lights|microphone|arm up|hand up/iu.test(job.prompt)) ??
    jobs.find((job) => job.providerPayload.cueKind === "action");
  const actionPrompt = String(actionJob?.prompt ?? "");
  const bodyPrompt = String(byKind.get("body_detail")?.prompt ?? "");
  const dialoguePrompt = String(byKind.get("dialogue_face")?.prompt ?? "");
  const dialogueAnchor = String(byKind.get("dialogue_face")?.providerPayload.anchorText ?? "");

  assertCheck("image.anchored_rules", jobs.length === 0, "Rule and paragraph planning does not create local-tagged image jobs without LLM cues.");
  assertCheck("image.anchored_rules", new Set(prompts).size <= 1, "No local prompt variants are fabricated for anchored hints.");
  assertCheck("image.anchored_rules", !/stage|stage lights|arm up|hand up|microphone/iu.test(actionPrompt), "Action cue prompt is not locally synthesized from assistant prose.");
  assertCheck("image.anchored_rules", !/\bstreet\b/iu.test(actionPrompt), "Stage action cue does not inherit later street tags from the same assistant turn.");
  assertCheck("image.anchored_rules", !/close-up|body focus|hands|wrist grab/iu.test(bodyPrompt), "Body-detail tags are not locally synthesized.");
  assertCheck("image.anchored_rules", !/close-up|face focus|open mouth/iu.test(dialoguePrompt), "Dialogue-face tags are not locally synthesized.");
  assertCheck("image.anchored_rules", !byKind.get("dialogue_face"), "Dialogue-face cue is not rendered without LLM tags.");
  assertCheck("image.anchored_rules", !dialogueAnchor, "Dialogue-face anchor is not emitted on a suppressed local hint.");
}

function evaluateNovelAiWeightingAndUserRules(seedState, planImageJob) {
  const state = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      artistPrompt: "::artist:bm94199 ::",
      userRules: [
        "Positive tags: 1.5::rain, night ::, -1::hat ::, {{dramatic lighting}}",
        "Negative tags: lowres, watermark, logo"
      ].join("\n")
    }
  });
  const cue = createCue(state, {
    scene: "archive library",
    visualContext: "rain, night, brass key, no hat visible"
  });
  const job = planImageJob(state, "turn_weighting", cue, [], true);
  const variantCue = createCue(state, {
    scene: "stage performance",
    tags: ["1girl", "stage", "stage lights", "standing", "holding microphone", "open mouth"],
    visualContext: "1girl, stage, stage lights, standing, holding microphone, open mouth"
  });
  const variantJob = planImageJob(state, "turn_prompt_variants", variantCue, [], true, { count: 3 });
  const weightedControlState = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      artistPrompt: "2::artist:eriol_s2 ::, -4.0::artist collaboration ::, 0.6::jeneral::, -2.0:: upscaled ::, -2.0::simple illustration ::",
      qualityPrompt: "masterpiece, best quality",
      stylePrompt: "cinematic anime illustration"
    }
  });
  const weightedControlJob = planImageJob(
    weightedControlState,
    "turn_weighted_controls",
    createCue(weightedControlState, {
      scene: "stage",
      tags: ["spotlight"],
      visualContext: "spotlight"
    }),
    [],
    true
  );
  const variantPrompts = Array.isArray(variantJob.providerPayload.promptVariants)
    ? variantJob.providerPayload.promptVariants.map((variant) => String(variant.prompt ?? ""))
    : [];
  const artistIndex = variantJob.prompt.indexOf("::artist:bm94199 ::");
  const firstSceneIndex = variantJob.prompt.search(/\bstage\b/iu);
  const weightedPrompt = weightedControlJob.prompt;
  const artistLayerIndex = weightedPrompt.indexOf("2::artist:eriol_s2 ::");
  const qualityLayerIndex = weightedPrompt.indexOf("masterpiece");
  const styleLayerIndex = weightedPrompt.indexOf("cinematic anime illustration");
  const spotlightIndex = weightedPrompt.indexOf("spotlight");

  assertCheck("nai.weighting", job.prompt.includes("1.5::rain, night ::"), "Numeric emphasis with comma remains intact.");
  assertCheck("nai.weighting", job.prompt.includes("-1::hat ::"), "Negative numeric emphasis stays in the prompt for targeted NAI removal.");
  assertCheck("nai.weighting", job.prompt.includes("::artist:bm94199 ::"), "Artist emphasis tag preserves the user-provided closing-space syntax.");
  assertCheck("nai.weighting", artistIndex >= 0 && firstSceneIndex >= 0 && artistIndex < firstSceneIndex, "Artist prompt stays before generated scene/action tags.");
  assertCheck("nai.weighting", variantPrompts.length === 3 && new Set(variantPrompts).size === 1, "Counted image jobs do not fabricate local prompt variants.");
  assertCheck("nai.weighting", variantPrompts.every((prompt) => /stage/iu.test(prompt) && /standing|holding microphone/iu.test(prompt)), "Prompt variants preserve the core scene and action tags.");
  assertCheck(
    "nai.weighting",
    artistLayerIndex >= 0 &&
      qualityLayerIndex > artistLayerIndex &&
      styleLayerIndex > qualityLayerIndex &&
      spotlightIndex > styleLayerIndex,
    "Positive prompt layers are ordered as artist, quality, style, generated tags."
  );
  assertCheck("nai.weighting", /2::artist:eriol_s2 ::.*-4\.0::artist collaboration ::.*0\.6::jeneral::.*-2\.0:: upscaled ::.*-2\.0::simple illustration ::/isu.test(weightedPrompt), "Artist prompt preserves the user's exact weighted tag order.");
  assertCheck("nai.weighting", !/\b\d+\s+\d+\s*::/u.test(job.prompt), "Weighted syntax is not mangled into broken token fragments.");
  assertCheck("nai.weighting", /lowres/iu.test(job.negativePrompt) && /watermark/iu.test(job.negativePrompt), "Explicit negative user-rule tags enter Undesired Content.");
  assertCheck("nai.weighting", Array.isArray(job.providerPayload.userRuleInstructions), "User image rules are preserved as composer instructions.");
}

function evaluateAdultContentRating(seedState, planImageJob) {
  const generalState = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      safetyLevel: "explicit",
      userRules: "Positive tags: nude, cleavage, nsfw"
    }
  });
  const generalCue = createCue(generalState, {
    scene: "private room",
    visualContext: "single adult character, private room"
  });
  const generalJob = planImageJob(generalState, "turn_general_content_rating", generalCue, [], true);

  assertCheck("rating.general", generalJob.status === "queued", "General mode does not apply DynamicChat app-level content blocking.");
  assertCheck("rating.general", /nude|cleavage|nsfw/iu.test(generalJob.prompt), "General mode preserves explicit positive tags for provider/API handling.");

  const adultState = createPlayableState(seedState, {
    simulation: {
      ...seedState.simulation,
      contentRating: "adult_19"
    },
    imageProfile: {
      ...seedState.imageProfile,
      safetyLevel: "explicit",
      userRules: [
        "캐릭터의 외형 일관성을 우선하고, 노골적 수위는 생성하지 않는다.",
        "Positive tags: 1.2::cleavage ::, nude, nsfw, revealing clothes",
        "Negative tags: lowres, watermark"
      ].join("\n")
    },
    characters: [
      {
        id: "char_adult",
        simulationId: seedState.simulation.id,
        name: "Mira",
        role: "adult protagonist",
        summary: "adult woman in a private apartment scene",
        relationship: "partner",
        currentMood: "confident"
      }
    ],
    visualProfiles: [
      {
        id: "visual_adult",
        simulationId: seedState.simulation.id,
        characterId: "char_adult",
        displayName: "Mira",
        positivePrompt: "adult woman, black hair, private apartment",
        negativePrompt: "bad hands",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "explicit"
      }
    ]
  });
  const adultCue = createCue(adultState, {
    characters: ["char_adult"],
    scene: "private adult apartment",
    visualContext: "adult woman, confident expression"
  });
  const adultJob = planImageJob(adultState, "turn_adult_rating", adultCue, [], true);

  assertCheck("rating.adult19", adultJob.status === "queued", "19+ adult-only state permits adult-only explicit image planning.");
  assertCheck("rating.adult19", /cleavage|nude|nsfw|revealing clothes/iu.test(adultJob.prompt), "19+ adult-only state keeps explicit positive user-rule tags in the NAI prompt.");
  assertCheck("rating.adult19", !/\b(?:cleavage|nude|nsfw|revealing clothes)\b/iu.test(adultJob.negativePrompt), "19+ adult-only state does not reroute explicit positive tags into Undesired Content.");
  assertCheck("rating.adult19", !/\bcensor\b/iu.test(adultJob.negativePrompt), "19+ adult-only state no longer applies removed adult scene focus negative tags.");
  assertCheck("rating.adult19", !/노골적\s*수위는\s*생성하지\s*않는다/u.test(String(adultJob.providerPayload.userRules ?? "")), "19+ adult-only state removes DynamicChat's built-in general image limiter from user-rule instructions.");

  const campusState = createPlayableState(seedState, {
    simulation: {
      ...seedState.simulation,
      contentRating: "adult_19",
      title: "Adult Campus Scene"
    },
    characters: [
      {
        id: "char_campus_adult",
        simulationId: seedState.simulation.id,
        name: "Yuna",
        role: "adult university student",
        summary: "adult college student in a campus apartment scene",
        relationship: "partner",
        currentMood: "confident"
      }
    ],
    visualProfiles: [],
    imageProfile: {
      ...seedState.imageProfile,
      safetyLevel: "explicit",
      userRules: "Positive tags: nude, cleavage"
    }
  });
  const campusCue = createCue(campusState, {
    characters: ["char_campus_adult"],
    scene: "adult campus apartment",
    visualContext: "adult university student, campus apartment"
  });
  const campusJob = planImageJob(campusState, "turn_adult_campus_rating", campusCue, [], true);

  assertCheck("rating.adult19", campusJob.status === "queued", "19+ adult-only state does not apply app-level student/campus content blocking.");
  assertCheck("rating.adult19", /nude|cleavage/iu.test(campusJob.prompt), "19+ adult-only state keeps explicit positive tags in adult campus scenes.");
}

function evaluateSceneSpecificity(seedState, runSimulationTurn) {
  const trainingState = createPlayableState(seedState, {
    messages: [
      createAssistantMessage(seedState, "The team reviews vocal training goals, budget pressure, and daily schedule at the office table.")
    ]
  });
  const practiceState = createPlayableState(seedState, {
    messages: [
      createAssistantMessage(seedState, "The trainees stand in a rented practice room, facing a wall mirror on the wooden floor.")
    ]
  });

  return Promise.all([
    runSimulationTurn(trainingState, "훈련 계획 장면을 이미지로 보여줘", true).then((result) => {
      const prompt = result.imageJob?.prompt ?? "";
      assertCheck("context.training", result.imageCue.scene !== "practice room", "Generic training does not fabricate a practice-room scene.");
      assertCheck("context.training", !/dance studio/iu.test(prompt), "Generic training does not inject dance studio tags.");
    }),
    runSimulationTurn(practiceState, "연습실 거울 앞 장면을 이미지로 보여줘", true).then((result) => {
      const prompt = result.imageJob?.prompt ?? "";
      const cueText = [result.imageCue.scene, result.imageCue.tags.join(", "), result.imageCue.visualContext].join(", ");
      assertCheck("context.practice", result.imageCue.scene === "current simulation scene", "Korean practice-room wording does not locally resolve a scene.");
      assertCheck("context.practice", !/practice room|dance studio|mirror/iu.test(`${cueText}, ${prompt}`), "Korean practice-room wording does not locally contribute practice-room or mirror tags.");
    })
  ]);
}

async function evaluateNoisyCueTagFiltering(seedState, planImageJobForCompletedTurn, generateNovelAiImages) {
  const state = createPlayableState(seedState, {
    modules: seedState.modules.filter((module) => module.kind !== "image_prompt_profile"),
    imageProfile: {
      ...seedState.imageProfile,
      qualityPrompt: "masterpiece, best quality",
      stylePrompt: "anime illustration",
      userRules: ""
    },
    messages: [
      createAssistantMessage(seedState, "Old scene: rain-soaked archive library, food on a table, and a confrontation."),
      createAssistantMessage(seedState, "The old scene closes."),
      createAssistantMessage(seedState, "The cast moves on.")
    ]
  });
  const userMessage = createUserMessage(state, "학교 복도 장면을 이미지로 보여줘");
  const assistantMessage = createAssistantMessage(state, "Hana walks through a quiet school hallway with a stable training plan and a notebook in hand.");
  const noisyCue = {
    shouldGenerate: true,
    reason: "noise filter eval",
    characters: [state.characters[0]?.id].filter(Boolean),
    tags: ["terror", "rain", "confrontation", "food", "table", "dramatic", "-"],
    scene: "current scene",
    visualContext: "terror, rain, confrontation, food, table, dramatic, -, stable training plan"
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: noisyCue,
        imageCues: [noisyCue]
      },
      manualImage: true
    }
  );
  const job = plan.imageJob;
  const combinedTags = [
    plan.imageCue.tags.join(", "),
    plan.imageCue.visualContext ?? "",
    job?.prompt ?? "",
    Array.isArray(job?.providerPayload.contextTags) ? job.providerPayload.contextTags.join(", ") : "",
    Array.isArray(job?.providerPayload.promptVariants)
      ? job.providerPayload.promptVariants.map((variant) => [variant.prompt, variant.cue?.tags?.join(", "), variant.cue?.visualContext].filter(Boolean).join(", ")).join(", ")
      : ""
  ].join(", ");

  assertCheck("context.noise", plan.imageCue.scene === "current simulation scene", "Current Korean scene wording is not locally converted into a school scene.");
  assertCheck("context.noise", /\b(?:terror|rain|confrontation|food|table|dramatic)\b/iu.test(combinedTags), "LLM-provided cue tags are preserved instead of being locally vetoed.");
  assertCheck("context.noise", !/\bdining table\b/iu.test(combinedTags), "Training/stable text does not locally synthesize extra dining-table tags beyond the cue.");
  assertCheck("context.noise", !/\bschool\b/iu.test(combinedTags), "Current visible scene tags are not inferred unless the sidecar provides them.");

  if (job) {
    const payloadResult = await generateNovelAiImages({
      state: {
        ...state,
        novelAi: {
          ...state.novelAi,
          modelPreset: "NAID4.5C"
        }
      },
      prompt: job.prompt,
      negativePrompt: job.negativePrompt,
      cue: plan.imageCue,
      count: 1
    });
    const v4Prompt = payloadResult.payload.parameters?.v4_prompt ?? {};
    const v4CaptionText = [
      v4Prompt.caption?.base_caption ?? "",
      ...(v4Prompt.caption?.char_captions ?? []).map((caption) => caption?.char_caption ?? "")
    ].join(", ");
    assertCheck("context.noise", /\b(?:rain|food)\b/iu.test(v4CaptionText), "NovelAI V4 base caption preserves LLM-provided cue tags.");
  }
}

async function evaluateImageCueCharacterScoping(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages) {
  const state = createPlayableState(seedState, {
    memoryEvents: [],
    characters: [
      {
        id: "char_a",
        simulationId: seedState.simulation.id,
        name: "Ari",
        role: "girl",
        summary: "female cast member",
        relationship: "cast",
        currentMood: "idle"
      },
      {
        id: "char_b",
        simulationId: seedState.simulation.id,
        name: "Beni",
        role: "girl",
        summary: "female cast member",
        relationship: "cast",
        currentMood: "idle"
      },
      {
        id: "char_c",
        simulationId: seedState.simulation.id,
        name: "Ciel",
        role: "girl",
        summary: "female cast member",
        relationship: "cast",
        currentMood: "idle"
      }
    ],
    visualProfiles: [
      {
        id: "visual_a",
        simulationId: seedState.simulation.id,
        characterId: "char_a",
        displayName: "Ari",
        positivePrompt: "red hair, green eyes, girl",
        negativePrompt: "bad hands",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_b",
        simulationId: seedState.simulation.id,
        characterId: "char_b",
        displayName: "Beni",
        positivePrompt: "blue hair, brown eyes, girl",
        negativePrompt: "bad anatomy",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_c",
        simulationId: seedState.simulation.id,
        characterId: "char_c",
        displayName: "Ciel",
        positivePrompt: "blonde hair, gray eyes, girl",
        negativePrompt: "low quality",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ]
  });
  const userMessage = createUserMessage(state, "혼자 있는 남자 장면을 이미지로 보여줘");
  const assistantMessage = createAssistantMessage(state, "A lone unnamed boy stands in first-person view near the hallway door.");
  const rosterCue = {
    shouldGenerate: true,
    reason: "character scope eval",
    characters: ["char_a", "char_b", "char_c"],
    tags: ["boy", "solo", "hallway", "ari", "beni", "ciel"],
    scene: "hallway",
    visualContext: "solo boy, first-person POV, hallway, ari, beni, ciel"
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: rosterCue,
        imageCues: [rosterCue]
      },
      manualImage: true
    }
  );
  const job = plan.imageJob;
  const directJob = planImageJob(
    state,
    "turn_direct_scope",
    {
      ...rosterCue,
      tags: ["3girls", "solo", "boy", "hallway"],
      visualContext: "3girls example tag, solo boy, first-person POV, hallway"
    },
    [],
    true
  );

  assertCheck("context.characters", plan.imageCue.characters.join(",") === "char_a,char_b,char_c", "LLM-provided character ids are preserved without local target rewriting.");
  assertCheck("context.characters", !/\b(?:ari|beni|ciel)\b/iu.test(plan.imageCue.tags.join(", ")), "Roster character names are not carried as image cue tags.");
  assertCheck("context.characters", !/\b3girls\b|red hair|blue hair|blonde hair/iu.test(job?.prompt ?? ""), "Solo outsider prompt does not include the configured three-girl roster.");
  assertCheck("context.characters", !/\b(?:ari|beni|ciel)\b/iu.test(directJob.prompt), "Final image job planning still strips roster-name tags from raw cue tags.");
  assertCheck("context.characters", !directJob.providerPayload.positiveTags?.some?.((tag) => /^(?:ari|beni|ciel)$/iu.test(String(tag))), "Roster names are removed from final positive tag layers.");
  assertCheck("context.characters", Array.isArray(directJob.providerPayload.cue?.characters) && directJob.providerPayload.cue.characters.length === 3, "Final image job payload preserves LLM-provided character ids.");

  if (job) {
    const payloadResult = await generateNovelAiImages({
      state: {
        ...state,
        novelAi: {
          ...state.novelAi,
          modelPreset: "NAID4.5C"
        }
      },
      prompt: job.prompt,
      negativePrompt: job.negativePrompt,
      cue: plan.imageCue,
      count: 1
    });
    const charCaptions = payloadResult.payload.parameters?.v4_prompt?.caption?.char_captions ?? [];
    assertCheck("context.characters", charCaptions.length === 0, "Solo outsider V4 payload does not create local captions from LLM-provided character ids.");
  }
}

async function evaluateLlmCharacterScopedOutfits(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages) {
  const state = createPlayableState(seedState, {
    memoryEvents: [],
    characters: [
      {
        id: "char_a",
        simulationId: seedState.simulation.id,
        name: "Ari",
        role: "girl performer",
        summary: "female cast member with red hair",
        relationship: "cast",
        currentMood: "focused"
      },
      {
        id: "char_b",
        simulationId: seedState.simulation.id,
        name: "Beni",
        role: "girl performer",
        summary: "female cast member with blue hair",
        relationship: "cast",
        currentMood: "focused"
      },
      {
        id: "char_c",
        simulationId: seedState.simulation.id,
        name: "Ciel",
        role: "girl performer",
        summary: "female cast member with blonde hair",
        relationship: "cast",
        currentMood: "offstage"
      }
    ],
    visualProfiles: [
      {
        id: "visual_a",
        simulationId: seedState.simulation.id,
        characterId: "char_a",
        displayName: "Ari",
        positivePrompt: "red hair, green eyes, girl",
        negativePrompt: "bad hands",
        defaultOutfitPrompt: "red casual cardigan",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_b",
        simulationId: seedState.simulation.id,
        characterId: "char_b",
        displayName: "Beni",
        positivePrompt: "blue hair, brown eyes, girl",
        negativePrompt: "bad anatomy",
        defaultOutfitPrompt: "blue casual cardigan",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_c",
        simulationId: seedState.simulation.id,
        characterId: "char_c",
        displayName: "Ciel",
        positivePrompt: "blonde hair, gray eyes, girl",
        negativePrompt: "low quality",
        defaultOutfitPrompt: "gray casual cardigan",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ]
  });
  const userMessage = createUserMessage(state, "둘이 무대에 서는 장면을 이미지로 보여줘");
  const assistantMessage = createAssistantMessage(state, "They step into the spotlight together, one reaching forward while the other steadies the microphone.");
  const llmCue = {
    shouldGenerate: true,
    reason: "LLM selected exactly two visible roster characters",
    characters: ["char_a", "char_b"],
    tags: ["stage lights", "microphone"],
    scene: "stage performance",
    visualContext: "two roster performers, stage lights, microphone"
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: llmCue,
        imageCues: [llmCue]
      },
      sidecarTrace: createParsedLlmSidecarTrace(state, userMessage.id),
      manualImage: true
    }
  );
  const layerText = plan.imageJob?.providerPayload.promptLayers?.characters?.join(", ") ?? "";

  assertCheck("context.llm_characters", plan.imageCue.characters.join(",") === "char_a,char_b", "Parsed LLM image cue keeps the exact visible character ids even when the prose omits names.");
  assertCheck("context.llm_characters", !/red hair|blue hair/iu.test(layerText), "Selected LLM character ids do not pull configured visual profile tags locally.");
  assertCheck("context.llm_characters", !/blonde hair/iu.test(layerText), "Unselected roster character tags are not added locally.");

  if (plan.imageJob) {
    const payloadResult = await generateNovelAiImages({
      state: {
        ...state,
        novelAi: {
          ...state.novelAi,
          modelPreset: "NAID4.5C"
        }
      },
      prompt: plan.imageJob.prompt,
      negativePrompt: plan.imageJob.negativePrompt,
      cue: plan.imageCue,
      count: 1
    });
    const charCaptions = payloadResult.payload.parameters?.v4_prompt?.caption?.char_captions ?? [];
    const captionText = charCaptions.map((caption) => caption?.char_caption ?? "").join("\n");
    assertCheck("context.llm_characters", charCaptions.length === 0, "NovelAI V4 does not create local captions from LLM-selected visible characters.");
    assertCheck("context.llm_characters", !/red hair|blue hair|blonde hair/iu.test(captionText), "V4 captions do not inject configured character tags.");
  }

  const outfitState = {
    ...state,
    memoryEvents: [
      createStateMemoryEvent(state, "char_a", "Wearing", "black performance jacket, silver belt"),
      createStateMemoryEvent(state, "char_b", "Wearing", "white ribbon dress, blue sash")
    ]
  };
  const outfitCue = createCue(outfitState, {
    characters: ["char_a", "char_b"],
    tags: ["stage lights", "microphone"],
    scene: "stage performance",
    visualContext: "two roster performers, stage lights, microphone"
  });
  const outfitJob = planImageJob(outfitState, "turn_llm_outfit_tags", outfitCue, [], true);
  const characterTags = outfitJob.providerPayload.promptLayers?.characters ?? [];
  assertCheck("context.llm_outfit", !characterTags.some((tag) => /red hair|black performance jacket/iu.test(String(tag))), "Ari character layer no longer injects configured or Wearing memory tags locally.");
  assertCheck("context.llm_outfit", !characterTags.some((tag) => /blue hair|white ribbon dress/iu.test(String(tag))), "Beni character layer no longer injects configured or Wearing memory tags locally.");
  assertCheck("context.llm_outfit", !characterTags.some((tag) => /gray casual cardigan|blonde hair/iu.test(String(tag))), "Outfit application does not pull tags for an unselected roster character.");

  const outfitPayload = await generateNovelAiImages({
    state: outfitState,
    prompt: outfitJob.prompt,
    negativePrompt: outfitJob.negativePrompt,
    cue: outfitCue,
    count: 1
  });
  const captions = outfitPayload.payload.parameters?.v4_prompt?.caption?.char_captions ?? [];
  assertCheck("context.llm_outfit", captions.length === 0 || !/red hair|black performance jacket/isu.test(captions[0]?.char_caption ?? ""), "Ari V4 caption is not locally created from configured or Wearing tags.");
  assertCheck("context.llm_outfit", captions.length === 0 || !/blue hair|white ribbon dress/isu.test(captions[1]?.char_caption ?? ""), "Beni V4 caption is not locally created from configured or Wearing tags.");
}

function evaluateWearingStateDetailPreservation(seedState, compileSimulationMemoryDelta, memoryDeltaToEvents) {
  const state = createPlayableState(seedState, {
    characters: [
      {
        id: "char_uniform",
        simulationId: seedState.simulation.id,
        name: "Mira",
        role: "officer",
        summary: "uniformed protagonist",
        relationship: "player-facing cast",
        currentMood: "tense"
      }
    ],
    visualProfiles: [
      {
        id: "visual_uniform",
        simulationId: seedState.simulation.id,
        characterId: "char_uniform",
        displayName: "Mira",
        positivePrompt: "black hair, gray eyes, girl",
        negativePrompt: "",
        defaultOutfitPrompt: "police uniform, navy short dress, mini skirt",
        outfitPrompts: {
          school: "school uniform, dark grey pencil skirt, tight fit, necktie"
        },
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ],
    memoryEvents: [
      createStateMemoryEvent(seedState, "char_uniform", "Wearing", "police uniform, navy short dress, mini skirt")
    ]
  });

  const damagedDelta = compileSimulationMemoryDelta({
    state,
    userText: "계속",
    assistantText: "미라의 유니폼 자락이 찢어진 채로 장면이 이어진다.",
    sourceTurnId: "turn_eval_wearing_damage",
    sidecar: {
      assistantText: "미라의 유니폼 자락이 찢어진 채로 장면이 이어진다.",
      imageCue: createCue(state, { shouldGenerate: false, tags: [], characters: [] }),
      imageCues: [],
      memoryEvents: [
        {
          memoryKind: "state",
          stateType: "Wearing",
          stateValue: "torn uniform",
          content: "Mira's uniform is torn.",
          importance: 0.86,
          confidence: 0.9,
          tags: ["outfit"],
          actorId: "char_uniform",
          actorName: "Mira"
        }
      ]
    }
  });
  const damagedValue = memoryDeltaToEvents(state, damagedDelta)[0]?.metadata?.value ?? "";
  assertCheck("memory.wearing", /police uniform/iu.test(String(damagedValue)), "Damaged Wearing state preserves the registered/previous base uniform.");
  assertCheck("memory.wearing", /navy short dress/iu.test(String(damagedValue)) && /mini skirt/iu.test(String(damagedValue)), "Damaged Wearing state keeps garment detail tags.");
  assertCheck("memory.wearing", /torn uniform/iu.test(String(damagedValue)), "Damaged Wearing state appends the new condition tag.");

  const keywordState = {
    ...state,
    memoryEvents: []
  };
  const keywordDelta = compileSimulationMemoryDelta({
    state: keywordState,
    userText: "교복으로 바뀐다",
    assistantText: "미라는 교복 차림으로 복도에 선다.",
    sourceTurnId: "turn_eval_wearing_keyword",
    sidecar: {
      assistantText: "미라는 교복 차림으로 복도에 선다.",
      imageCue: createCue(keywordState, { shouldGenerate: false, tags: [], characters: [] }),
      imageCues: [],
      memoryEvents: [
        {
          memoryKind: "state",
          stateType: "Wearing",
          stateValue: "school",
          content: "Mira is wearing the school outfit.",
          importance: 0.86,
          confidence: 0.9,
          tags: ["outfit"],
          actorId: "char_uniform",
          actorName: "Mira"
        }
      ]
    }
  });
  const keywordValue = memoryDeltaToEvents(keywordState, keywordDelta)[0]?.metadata?.value ?? "";
  assertCheck("memory.wearing", /school uniform/iu.test(String(keywordValue)), "Wearing keyword labels expand to mapped outfit prompts.");
  assertCheck("memory.wearing", /dark grey pencil skirt/iu.test(String(keywordValue)) && /necktie/iu.test(String(keywordValue)), "Wearing keyword expansion keeps mapped detail tags.");

  const removalState = createPlayableState(seedState, {
    characters: [
      {
        id: "char_uniform",
        simulationId: seedState.simulation.id,
        name: "Mira",
        role: "officer",
        summary: "uniformed protagonist",
        relationship: "player-facing cast",
        currentMood: "tense"
      }
    ],
    visualProfiles: [
      {
        id: "visual_uniform",
        simulationId: seedState.simulation.id,
        characterId: "char_uniform",
        displayName: "Mira",
        positivePrompt: "black hair, gray eyes, girl",
        negativePrompt: "",
        defaultOutfitPrompt: "police uniform, navy short dress, mini skirt, metal chastity cage",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ],
    memoryEvents: [
      createStateMemoryEvent(seedState, "char_uniform", "Wearing", "police uniform, navy short dress, mini skirt, metal chastity cage")
    ]
  });
  const removalDelta = compileSimulationMemoryDelta({
    state: removalState,
    userText: "케이지를 벗긴다",
    assistantText: "미라의 다리 사이에서 metal chastity cage를 벗겨 바닥에 내려놓는다.",
    sourceTurnId: "turn_eval_wearing_removal",
    sidecar: {
      assistantText: "미라의 다리 사이에서 metal chastity cage를 벗겨 바닥에 내려놓는다.",
      imageCue: createCue(removalState, { shouldGenerate: false, tags: [], characters: [] }),
      imageCues: [],
      memoryEvents: [
        {
          memoryKind: "state",
          stateType: "Wearing",
          stateValue: "police uniform, navy short dress, mini skirt",
          content: "The metal chastity cage is removed from Mira.",
          importance: 0.88,
          confidence: 0.9,
          tags: ["outfit"],
          actorId: "char_uniform",
          actorName: "Mira"
        }
      ]
    }
  });
  const removalValue = memoryDeltaToEvents(removalState, removalDelta)[0]?.metadata?.value ?? "";
  assertCheck("memory.wearing", !/chastity cage/iu.test(String(removalValue)), "Removed garment drops out of the Wearing state instead of being re-merged from the base.");
  assertCheck("memory.wearing", /police uniform/iu.test(String(removalValue)) && /mini skirt/iu.test(String(removalValue)), "Removal keeps the remaining outfit the LLM still lists.");
}

async function evaluateActorTargetCharacterDisambiguation(seedState, planImageJob, planImageJobForCompletedTurn) {
  const state = createPlayableState(seedState, {
    userPersona: {
      ...seedState.userPersona,
      enabled: true,
      source: "character",
      characterId: "char_akane",
      name: "아카네 리제"
    },
    characters: [
      {
        id: "char_akane",
        simulationId: seedState.simulation.id,
        name: "아카네 리제",
        role: "user-controlled protagonist",
        summary: "first-person controlled character",
        relationship: "player persona",
        currentMood: "alert"
      },
      {
        id: "char_minsel",
        simulationId: seedState.simulation.id,
        name: "민설",
        role: "girl classmate",
        summary: "female classmate with light blonde twin tails",
        relationship: "nearby classmate",
        currentMood: "watching"
      }
    ],
    visualProfiles: [
      {
        id: "visual_akane",
        simulationId: seedState.simulation.id,
        characterId: "char_akane",
        displayName: "아카네 리제",
        positivePrompt: "black hair, red eyes, girl",
        negativePrompt: "bad anatomy",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_minsel",
        simulationId: seedState.simulation.id,
        characterId: "char_minsel",
        displayName: "민설",
        positivePrompt: "light blonde hair, twin tails, girl",
        negativePrompt: "bad anatomy",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ]
  });
  const userMessage = createUserMessage(state, "나를 지목했던 놈이 기분 나쁜 미소를 지으며 말했다.");
  const assistantMessage = createAssistantMessage(state, "나를 지목했던 놈이 기분 나쁜 미소를 지으며 말했다. 골목의 공기가 싸늘하게 굳었다.");
  const wrongRosterCue = {
    shouldGenerate: true,
    reason: "actor-target disambiguation eval",
    characters: ["char_minsel"],
    tags: ["leering", "evil smile", "bloodshot eyes", "dark alley", "threatening", "smile"],
    scene: "dark alley",
    visualContext: "나를 지목했던 놈이 기분 나쁜 미소를 지으며 말했다, face focus, dark alley"
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: wrongRosterCue,
        imageCues: [wrongRosterCue]
      },
      manualImage: true
    }
  );
  const job = plan.imageJob;
  const directJob = planImageJob(
    state,
    "turn_actor_target_direct",
    {
      shouldGenerate: true,
      reason: "actor-target direct eval",
      characters: ["char_minsel"],
      tags: ["leering", "evil smile", "dark alley", "smile"],
      scene: "dark alley",
      visualContext: "나를 지목했던 놈이 기분 나쁜 미소를 지으며 말했다"
    },
    [],
    true
  );
  const jobCharacterLayer = job?.providerPayload.promptLayers?.characters?.join(", ") ?? "";
  const directJobCharacterLayer = directJob.providerPayload.promptLayers?.characters?.join(", ") ?? "";

  assertCheck("context.actor_target", plan.imageCue.characters.join(",") === "char_minsel", "Unnamed external actor cue preserves the LLM-provided character id without local target rewriting.");
  assertCheck("context.actor_target", !/\b(?:light blonde hair|twin tails|twintails)\b/iu.test(job?.prompt ?? ""), "Wrong target character visual tags are not applied to the unnamed male actor.");
  assertCheck("context.actor_target", !/\b1boy\b/iu.test(job?.prompt ?? ""), "Unnamed Korean male actor cue does not infer 1boy locally.");
  assertCheck("context.actor_target", !/\bmale student\b/iu.test(job?.prompt ?? ""), "Unnamed Korean male actor cue does not infer gender/subject tags locally.");
  assertCheck("context.actor_target", /\b(?:evil smile|leering)\b/iu.test(jobCharacterLayer), "Expression tags stay attached to the external actor character prompt.");
  assertCheck("context.actor_target", !/\b(?:light blonde hair|twin tails|twintails)\b/iu.test(directJob.prompt), "Direct image job planning also strips wrong target character visuals for external actors.");
  assertCheck("context.actor_target", /\b(?:evil smile|leering)\b/iu.test(directJobCharacterLayer), "Direct image job planning keeps external actor expression tags in the character layer.");
  assertCheck("context.actor_target", Array.isArray(directJob.providerPayload.cue?.characters) && directJob.providerPayload.cue.characters.join(",") === "char_minsel", "Direct image job payload preserves LLM-provided character ids.");
}

async function evaluateStructuredCueDisambiguation(seedState, planImageJob, planImageJobForCompletedTurn) {
  const state = createPlayableState(seedState, {
    memoryEvents: [],
    characters: [
      {
        id: "char_a",
        simulationId: seedState.simulation.id,
        name: "Ari",
        role: "girl performer",
        summary: "female cast member with red hair",
        relationship: "cast",
        currentMood: "focused"
      },
      {
        id: "char_b",
        simulationId: seedState.simulation.id,
        name: "Beni",
        role: "girl performer",
        summary: "female cast member with blue hair",
        relationship: "cast",
        currentMood: "watching"
      },
      {
        id: "char_c",
        simulationId: seedState.simulation.id,
        name: "Ciel",
        role: "girl performer",
        summary: "female cast member with blonde hair",
        relationship: "cast",
        currentMood: "offstage"
      }
    ],
    visualProfiles: [
      {
        id: "visual_a",
        simulationId: seedState.simulation.id,
        characterId: "char_a",
        displayName: "Ari",
        positivePrompt: "red hair, green eyes, girl",
        negativePrompt: "bad hands",
        defaultOutfitPrompt: "red casual cardigan",
        outfitPrompts: { school: "registered navy uniform, pleated skirt" },
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_b",
        simulationId: seedState.simulation.id,
        characterId: "char_b",
        displayName: "Beni",
        positivePrompt: "blue hair, brown eyes, girl",
        negativePrompt: "bad anatomy",
        defaultOutfitPrompt: "blue casual cardigan",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_c",
        simulationId: seedState.simulation.id,
        characterId: "char_c",
        displayName: "Ciel",
        positivePrompt: "blonde hair, gray eyes, girl",
        negativePrompt: "low quality",
        defaultOutfitPrompt: "gray casual cardigan",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ]
  });

  const userMessage = createUserMessage(state, "아리가 마이크를 잡고 말하는 장면을 이미지로 보여줘.");
  const assistantMessage = createAssistantMessage(state, "Ari grips the microphone and speaks while the other performers wait off camera.");
  const noisyRosterCue = {
    shouldGenerate: true,
    reason: "LLM copied broad roster and generic production tags",
    characters: ["char_a", "char_b", "char_c"],
    tags: ["filming set", "getting up", "props", "acting scene", "audition", "director", "student", "academy", "microphone", "camera", "speaking"],
    scene: "filming set",
    visualContext: "Ari speaking, acting scene, audition, director, student, academy, microphone, camera",
    anchorText: "Ari grips the microphone and speaks."
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: noisyRosterCue,
        imageCues: [noisyRosterCue]
      },
      sidecarTrace: createParsedLlmSidecarTrace(state, userMessage.id),
      manualImage: true
    }
  );
  const combinedPrompt = [
    plan.imageCue.tags.join(", "),
    plan.imageCue.visualContext ?? "",
    plan.imageJob?.prompt ?? "",
    plan.imageJob?.providerPayload.promptVariants?.map((variant) => [variant.prompt, variant.cue?.tags?.join(", "), variant.cue?.visualContext].filter(Boolean).join(", ")).join(", ") ?? ""
  ].join(", ");
  const finalPromptText = [
    plan.imageJob?.prompt ?? "",
    plan.imageJob?.providerPayload.promptVariants?.map((variant) => variant.prompt).join(", ") ?? ""
  ].join(", ");

  assertCheck("context.structured_cue", plan.imageCue.characters.join(",") === "char_a,char_b,char_c", "LLM-provided character ids are not locally narrowed.");
  assertCheck("context.structured_cue", /\b(?:getting up|props?)\b/iu.test(finalPromptText), "LLM-provided placeholder cue tags are passed through without local cleanup.");
  assertCheck("context.structured_cue", /microphone|camera|open mouth/iu.test(combinedPrompt), "Concrete object/action tags remain after NAI tag normalization.");

  const outsiderCue = createCue(state, {
    characters: ["char_a"],
    tags: ["teacher", "speaking"],
    scene: "classroom",
    visualContext: "teacher speaking to viewer"
  });
  const outsiderJob = planImageJob(state, "turn_teacher_outsider", outsiderCue, [], true);
  assertCheck("context.structured_cue", outsiderJob.providerPayload.cue?.characters?.join(",") === "char_a", "Non-roster speaker cue preserves the LLM-provided character id.");
  assertCheck("context.structured_cue", !/red hair/iu.test(outsiderJob.providerPayload.promptLayers?.characters?.join(", ") ?? ""), "Selected character visual prompt is not locally applied when a character id is present.");

  const outfitState = {
    ...state,
    memoryEvents: [createStateMemoryEvent(state, "char_a", "Wearing", "school")]
  };
  const outfitCue = createCue(outfitState, {
    characters: ["char_a"],
    tags: ["microphone"],
    scene: "stage performance",
    visualContext: "Ari speaking into microphone"
  });
  const outfitJob = planImageJob(outfitState, "turn_current_outfit_mapping", outfitCue, [], true);
  const characterLayer = outfitJob.providerPayload.promptLayers?.characters?.join(", ") ?? "";
  assertCheck("context.structured_cue", !/registered navy uniform|red casual cardigan/iu.test(characterLayer), "Current/default outfit tags are not locally resolved into the character layer.");
}

async function evaluateLlmNaiTagPreservation(seedState, planImageJob, generateNovelAiImages) {
  const state = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      artistPrompt: "",
      qualityPrompt: "masterpiece, best quality",
      stylePrompt: "",
      userRules: "Positive tags: dutch angle\nNegative tags: watermark"
    },
    characters: [
      {
        id: "char_pose",
        simulationId: seedState.simulation.id,
        name: "Mina",
        role: "visible roster character",
        summary: "A girl in the classroom scene.",
        relationship: "",
        currentMood: "worried"
      }
    ],
    visualProfiles: [
      {
        id: "visual_pose",
        simulationId: seedState.simulation.id,
        characterId: "char_pose",
        displayName: "Mina",
        positivePrompt: "pink hair, green eyes, petite girl",
        negativePrompt: "low quality",
        defaultOutfitPrompt: "",
        outfitPrompts: {
          school: "school uniform, dark grey pencil skirt, tight fit, necktie"
        },
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ]
  });
  const cue = {
    shouldGenerate: true,
    reason: "LLM returned concrete NAI tags",
    characters: ["char_pose"],
    tags: ["1girl", "standing up from chair", "looking at viewer", "worried expression", "holding notebook", "classroom", "school uniform", "facial expression", "street"],
    scene: "classroom",
    visualContext: "1girl, standing up from chair, looking at viewer, worried expression, holding notebook, classroom setting, school uniform, visible emotional reaction, situation specific clothing"
  };
  const job = planImageJob(state, "turn_llm_nai_tags", cue, [], true);
  const prompt = job.prompt;

  assertCheck("context.nai_tags", /\b1girl\b/iu.test(prompt), "LLM-provided subject count tag is preserved in the NAI prompt.");
  assertCheck("context.nai_tags", /standing up from chair/iu.test(prompt), "Natural action phrases are passed through without local decomposition.");
  assertCheck("context.nai_tags", /looking at viewer/iu.test(prompt), "Concrete gaze/POV tag is preserved.");
  assertCheck("context.nai_tags", /holding notebook/iu.test(prompt), "Held-item/action tag is preserved.");
  assertCheck("context.nai_tags", /dutch angle/iu.test(prompt), "Positive image user-rule tag is included.");
  assertCheck("context.nai_tags", /watermark/iu.test(job.negativePrompt), "Negative image user-rule tag is included in Undesired Content.");
  assertCheck("context.nai_tags", /street/iu.test(prompt), "Contradictory LLM cue scene tags are not locally filtered.");
  const characterLayer = job.providerPayload.promptLayers?.characters?.join(", ") ?? "";
  assertCheck("context.nai_tags", /school uniform/iu.test(characterLayer) && /worried expression|facial expression/iu.test(characterLayer), "Character outfit and expression tags are kept in the character prompt layer.");
  assertCheck("context.nai_tags", !/pink hair|dark grey pencil skirt|necktie/iu.test(characterLayer), "Registered visual and outfit keyword mappings are not locally expanded.");

  const payloadResult = await generateNovelAiImages({
    state,
    prompt: job.prompt,
    negativePrompt: job.negativePrompt,
    cue,
    count: 1
  });
  const baseCaption = payloadResult.payload.parameters?.v4_prompt?.caption?.base_caption ?? "";
  const charCaption = payloadResult.payload.parameters?.v4_prompt?.caption?.char_captions?.[0]?.char_caption ?? "";
  assertCheck(
    "context.nai_tags",
    /1girl/iu.test(baseCaption) && /looking at viewer/iu.test(baseCaption) && /classroom/iu.test(baseCaption) && /standing up from chair/iu.test(baseCaption),
    "NovelAI V4 base caption keeps base LLM cue tags without local decomposition."
  );
  assertCheck("context.nai_tags", /school uniform|worried expression/iu.test(charCaption), "NovelAI V4 character caption receives LLM character-related tags.");
  assertCheck("context.nai_tags", !/pink hair|dark grey pencil skirt|necktie/isu.test(charCaption), "NovelAI V4 character caption does not inject configured identity or outfit tags locally.");
}

function evaluateNameAndBodyInventoryCleanup(seedState, planImageJob) {
  const state = createPlayableState(seedState, {
    userPersona: {
      ...seedState.userPersona,
      enabled: true,
      source: "custom",
      name: "Yang Woojeong",
      role: "POV owner",
      updatedAt: new Date().toISOString()
    },
    imageProfile: {
      ...seedState.imageProfile,
      artistPrompt: "",
      qualityPrompt: "",
      stylePrompt: "",
      userRules: ""
    }
  });
  const noisyCue = {
    shouldGenerate: true,
    reason: "LLM over-expanded visible subject tags",
    characters: [],
    tags: [
      "1girl",
      "yang woojeong s pov",
      "close up",
      "looking at viewer",
      "upper body",
      "indoors",
      "holding head",
      "open mouth",
      "mouth",
      "chest",
      "disheveled hair",
      "red eyes",
      "sweat",
      "sweat drop",
      "blush",
      "grimacing",
      "bangs",
      "forehead",
      "eyebrows",
      "nose",
      "lips",
      "chin",
      "neck",
      "shoulders",
      "collarbone",
      "skin",
      "face",
      "head",
      "human",
      "person",
      "female",
      "woman"
    ],
    scene: "indoors",
    visualContext: "1girl, Yang Woojeong's POV, close up, holding head, open mouth, forehead, eyebrows, nose, lips, chin, neck, shoulders, collarbone, skin, face, head, human, person, female, woman"
  };
  const job = planImageJob(state, "turn_name_body_inventory_cleanup", noisyCue, [], true);
  const prompt = job.prompt;
  const characterLayer = job.providerPayload.promptLayers?.characters?.join(", ") ?? "";
  const promptTags = new Set(`${prompt}, ${characterLayer}`.split(",").map((tag) => tag.trim().toLowerCase()));
  const anatomyTags = ["forehead", "eyebrows", "nose", "lips", "chin", "neck", "shoulders", "collarbone", "skin", "face", "head", "mouth", "chest"];

  assertCheck("context.tag_cleanup", /\b1girl\b/iu.test(prompt) && /yang woojeong s pov/iu.test(prompt), "LLM subject and POV tags pass through without local rewriting.");
  assertCheck("context.tag_cleanup", /\byang\b|\bwoojeong\b/iu.test(prompt), "Persona names in LLM tags are not locally removed.");
  assertCheck("context.tag_cleanup", /\b(?:human|person|female|woman)\b/iu.test(prompt), "Generic human/person labels in LLM tags are not locally removed.");
  assertCheck("context.tag_cleanup", anatomyTags.every((tag) => promptTags.has(tag)), "Bulk anatomy inventory is preserved across base and character prompt layers.");
  assertCheck("context.tag_cleanup", /holding head|open mouth|disheveled hair|red eyes|sweat|blush/iu.test(`${prompt}, ${characterLayer}`), "Salient pose/expression/appearance tags remain after pass-through.");
}

async function evaluatePersonaCharacterImageCueScoping(seedState, planImageJobForCompletedTurn, generateNovelAiImages) {
  const state = createPlayableState(seedState, {
    userPersona: {
      ...seedState.userPersona,
      enabled: true,
      source: "character",
      characterId: "char_player",
      name: "Player",
      updatedAt: new Date().toISOString()
    },
    characters: [
      {
        id: "char_player",
        simulationId: seedState.simulation.id,
        name: "Akane",
        role: "user-controlled heroine",
        summary: "The current playable character.",
        relationship: "",
        currentMood: "focused"
      },
      {
        id: "char_other",
        simulationId: seedState.simulation.id,
        name: "Rize",
        role: "other roster character",
        summary: "Another character in the scene.",
        relationship: "",
        currentMood: "watching"
      }
    ],
    visualProfiles: [
      {
        id: "visual_player",
        simulationId: seedState.simulation.id,
        characterId: "char_player",
        displayName: "Akane",
        positivePrompt: "red hair, amber eyes, girl",
        negativePrompt: "low quality",
        defaultOutfitPrompt: "black hoodie, pleated skirt",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_other",
        simulationId: seedState.simulation.id,
        characterId: "char_other",
        displayName: "Rize",
        positivePrompt: "silver hair, blue eyes, girl",
        negativePrompt: "low quality",
        defaultOutfitPrompt: "white cardigan",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ]
  });
  const userMessage = createUserMessage(state, "*(의자에서 일어나 노트를 움켜쥔다)*");
  const assistantMessage = createAssistantMessage(state, "나는 의자에서 몸을 일으키며 노트를 세게 움켜쥔다.");
  const cue = {
    shouldGenerate: true,
    reason: "controlled persona action cue",
    characters: [],
    tags: ["1girl", "standing up from chair", "holding notebook", "determined expression"],
    scene: "classroom",
    visualContext: "1girl, standing up from chair, holding notebook, determined expression, classroom"
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: cue,
        imageCues: [cue]
      },
      sidecarTrace: createParsedLlmSidecarTrace(state, userMessage.id),
      manualImage: true
    }
  );

  assertCheck("context.persona_actor", plan.imageCue.characters.length === 0, "User-controlled character persona is not locally inferred when image_cues.characters is empty.");
  assertCheck("context.persona_actor", !plan.imageCue.characters.includes("char_other"), "Persona action cue does not pull unrelated roster characters.");

  if (plan.imageJob) {
    const payloadResult = await generateNovelAiImages({
      state,
      prompt: plan.imageJob.prompt,
      negativePrompt: plan.imageJob.negativePrompt,
      cue: plan.imageCue,
      count: 1
    });
    const charCaption = payloadResult.payload.parameters?.v4_prompt?.caption?.char_captions?.[0]?.char_caption ?? "";
    assertCheck("context.persona_actor", !/red hair|silver hair|black hoodie|white cardigan/iu.test(charCaption), "NovelAI V4 character caption does not inject roster visuals without an LLM-provided character id.");
  }
}

async function evaluateLlmImageStateTagCarryover(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages) {
  const state = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto"
    },
    userPersona: {
      ...seedState.userPersona,
      enabled: true,
      source: "character",
      characterId: "char_player_state",
      name: "Player",
      updatedAt: new Date().toISOString()
    },
    characters: [
      {
        id: "char_player_state",
        simulationId: seedState.simulation.id,
        name: "Akane",
        role: "user-controlled heroine",
        summary: "The current playable character.",
        relationship: "",
        currentMood: "focused"
      }
    ],
    visualProfiles: [
      {
        id: "visual_player_state",
        simulationId: seedState.simulation.id,
        characterId: "char_player_state",
        displayName: "Akane",
        positivePrompt: "red hair, amber eyes, girl",
        negativePrompt: "low quality",
        defaultOutfitPrompt: "black hoodie, pleated skirt",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ],
    memoryEvents: [
      createStateMemoryEvent(seedState, "char_player_state", "ActionTags", "standing up from chair, holding notebook"),
      createStateMemoryEvent(seedState, "char_player_state", "PoseTags", "standing pose, leaning forward"),
      createSceneStateMemoryEvent(seedState, "SceneTags", "classroom, low angle shot")
    ]
  });

  const userMessage = createUserMessage(state, "*(의자에서 일어나 노트를 움켜쥔다)*");
  const assistantMessage = createAssistantMessage(state, "나는 의자에서 몸을 일으키며 노트를 세게 움켜쥔다.");
  const sparseCue = {
    shouldGenerate: true,
    reason: "LLM asked for an image but relied on current image state memory",
    characters: [],
    tags: [],
    scene: "current simulation scene",
    visualContext: ""
  };
  const plan = await planImageJobForCompletedTurn(
    {
      ...state,
      messages: [...state.messages, userMessage, assistantMessage]
    },
    {
      userMessage,
      assistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: assistantMessage.content,
        memoryEvents: [],
        imageCue: sparseCue,
        imageCues: [sparseCue]
      },
      sidecarTrace: createParsedLlmSidecarTrace(state, userMessage.id),
      manualImage: false
    }
  );

  assertCheck("context.image_state", !plan.imageJob, "Sparse generated image cues are suppressed instead of being rescued by local image state tags.");
  assertCheck("context.image_state", plan.imageCue.characters.length === 0, "Sparse persona cue does not locally resolve the controlled character as actor.");
  assertCheck("context.image_state", !/standing up from chair|holding notebook/iu.test(plan.imageCue.visualContext ?? ""), "Current character image state tags do not enter the planned cue visual context.");

  if (plan.imageJob) {
    const characterLayer = plan.imageJob.providerPayload.promptLayers?.characters?.join(", ") ?? "";
    assertCheck("context.image_state", !/\bstanding\b|leaning forward|holding notebook/iu.test(characterLayer), "Current character image state tags do not enter the character prompt layer.");
    assertCheck("context.image_state", !/classroom|low angle shot/iu.test(plan.imageJob.prompt), "Current scene image state tags do not enter the base prompt.");

    const payloadResult = await generateNovelAiImages({
      state,
      prompt: plan.imageJob.prompt,
      negativePrompt: plan.imageJob.negativePrompt,
      cue: plan.imageCue,
      count: 1
    });
    const charCaption = payloadResult.payload.parameters?.v4_prompt?.caption?.char_captions?.[0]?.char_caption ?? "";
    assertCheck(
      "context.image_state",
      !/\bstanding\b|leaning forward|holding notebook/iu.test(charCaption),
      "NovelAI V4 character caption does not include local current image state tags."
    );
  }

  const blankUserMessage = createUserMessage(state, "이미지 생성");
  const blankAssistantMessage = createAssistantMessage(state, "장면은 아직 시각적으로 확정되지 않았다.");
  const blankPlan = await planImageJobForCompletedTurn(
    {
      ...state,
      memoryEvents: [],
      messages: [...state.messages, blankUserMessage, blankAssistantMessage]
    },
    {
      userMessage: blankUserMessage,
      assistantMessage: blankAssistantMessage,
      contextPack: createContextPack(state),
      promptModuleUsages: [],
      sidecar: {
        assistantText: blankAssistantMessage.content,
        memoryEvents: [],
        imageCue: sparseCue,
        imageCues: [sparseCue]
      },
      sidecarTrace: createParsedLlmSidecarTrace(state, blankUserMessage.id),
      manualImage: false
    }
  );
  assertCheck("context.image_state", !blankPlan.imageJob, "Parsed LLM cue with no tags and no image state is suppressed instead of generating a generic image.");
}

async function evaluateImageDetailStateContinuity(seedState, planImageJobForCompletedTurn) {
  const characters = [
    {
      id: "char_detail",
      simulationId: seedState.simulation.id,
      name: "Mei",
      role: "heroine",
      summary: "lead",
      relationship: "",
      currentMood: "tense"
    }
  ];
  const visualProfiles = [
    {
      id: "visual_detail",
      simulationId: seedState.simulation.id,
      characterId: "char_detail",
      displayName: "Mei",
      positivePrompt: "black hair, red eyes, girl",
      negativePrompt: "low quality",
      defaultOutfitPrompt: "white blouse, blue skirt",
      outfitPrompts: {},
      expressionPrompts: {},
      referenceImageAssetIds: [],
      defaultSafetyLevel: "safe"
    }
  ];
  const baseOverrides = {
    imageProfile: { ...seedState.imageProfile, triggerMode: "realtime_auto", cooldownTurns: 0 },
    relationshipMap: { ...seedState.relationshipMap, enabled: true },
    characters,
    visualProfiles
  };
  const state = createPlayableState(seedState, {
    ...baseOverrides,
    memoryEvents: [
      createStateMemoryEvent({ ...seedState, characters }, "char_detail", "InteractionTags", "straddling another"),
      createStateMemoryEvent({ ...seedState, characters }, "char_detail", "PhysicalStateTags", "bloody lip, bruised cheek"),
      createStateMemoryEvent({ ...seedState, characters }, "char_detail", "HeldItemTags", "holding notebook")
    ]
  });

  const userMessage = createUserMessage(state, "그 순간을 클로즈업으로 보여줘");
  const assistantMessage = createAssistantMessage(state, "메이가 입을 벌리며 얼굴을 붉힌다.");
  const composedFor = (plan, jobIndex = 0) => {
    const job = jobIndex === 0 ? plan.imageJob : plan.imageJobs[jobIndex];
    return job?.providerPayload?.characterPrompts?.find((entry) => entry.characterId === "char_detail")?.prompt ?? "";
  };
  const planTurn = (cues, planState) =>
    planImageJobForCompletedTurn(
      { ...planState, messages: [...planState.messages, userMessage, assistantMessage] },
      {
        userMessage,
        assistantMessage,
        contextPack: createContextPack(planState),
        promptModuleUsages: [],
        sidecar: {
          assistantText: assistantMessage.content,
          memoryEvents: [],
          imageCue: cues[0],
          imageCues: cues
        },
        sidecarTrace: createParsedLlmSidecarTrace(planState, userMessage.id),
        manualImage: false
      }
    );

  const closeUpCue = {
    shouldGenerate: true,
    reason: "close-up beat",
    characters: ["char_detail"],
    tags: [],
    baseTags: ["close-up", "indoors"],
    characterPrompts: [{ characterId: "char_detail", prompt: "blush, open mouth, face focus" }],
    scene: "current simulation scene",
    visualContext: ""
  };
  const plan = await planTurn([closeUpCue], state);
  const composed = composedFor(plan);
  assertCheck("context.detail_continuity", /bloody lip/iu.test(composed), "Persisted physical-detail tags carry into a close-up cut that omitted them.");
  assertCheck("context.detail_continuity", /bruised cheek/iu.test(composed), "All persisted physical-detail tags carry, not just the first one.");
  assertCheck("context.detail_continuity", /holding notebook/iu.test(composed), "Persisted held-item tags carry into a cut that omitted them.");
  assertCheck("context.detail_continuity", /straddling/iu.test(composed), "Persisted interaction posture carries into a cut with no posture of its own.");

  const changedCue = {
    ...closeUpCue,
    characterPrompts: [{ characterId: "char_detail", prompt: "blood on arm, gritting teeth, face focus" }]
  };
  const changedPlan = await planTurn([changedCue], state);
  const changedComposed = composedFor(changedPlan);
  assertCheck("context.detail_continuity", /blood on arm/iu.test(changedComposed), "Explicitly authored new physical detail is kept on the cut.");
  assertCheck("context.detail_continuity", !/bloody lip/iu.test(changedComposed), "A cut that authors its own physical detail does not re-inject the stale persisted physical detail.");

  const progressionState = createPlayableState(seedState, { ...baseOverrides, memoryEvents: [] });
  const cutOne = {
    shouldGenerate: true,
    reason: "establishing cut",
    characters: ["char_detail"],
    tags: [],
    baseTags: ["medium shot", "indoors"],
    characterPrompts: [{ characterId: "char_detail", prompt: "torn blouse, blood on lip, straddling another" }],
    scene: "current simulation scene",
    visualContext: ""
  };
  const cutTwo = {
    shouldGenerate: true,
    reason: "close-up cut",
    characters: ["char_detail"],
    tags: [],
    baseTags: ["close-up", "indoors"],
    characterPrompts: [{ characterId: "char_detail", prompt: "open mouth, face focus" }],
    scene: "current simulation scene",
    visualContext: ""
  };
  const progressionPlan = await planTurn([cutOne, cutTwo], progressionState);
  const secondComposed = composedFor(progressionPlan, 1);
  assertCheck("context.detail_continuity", progressionPlan.imageJobs.length === 2, "Both cuts in one output are planned as separate jobs.");
  assertCheck("context.detail_continuity", /blood on lip/iu.test(secondComposed), "A detail introduced in an earlier cut carries into a later cut in the same output.");
  assertCheck("context.detail_continuity", /torn blouse/iu.test(secondComposed), "An outfit change in an earlier cut carries into a later cut in the same output.");
  assertCheck("context.detail_continuity", /straddling/iu.test(secondComposed), "A posture established in an earlier cut carries into a later cut in the same output.");
}

async function evaluateFallbackImageCueSuppression(seedState, runSimulationTurn) {
  const visualState = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto"
    },
    messages: [
      createAssistantMessage(seedState, "The idol team is preparing a stage performance under bright stage lights.")
    ]
  });
  const visualResult = await runSimulationTurn(visualState, "무대 장면을 이어가", false);

  assertCheck("fallback.image_cue", !visualResult.imageJob, "Unconfigured fallback turns do not auto-plan image jobs from keyword matches.");
  assertCheck("fallback.image_cue", visualResult.imageCue.shouldGenerate === false, "Fallback visual context waits for main sidecar cues instead of keyword triggering.");
  assertCheck("fallback.image_cue", !/Fallback sidecar used/iu.test(visualResult.imageCue.reason), "Fallback sidecar reason is not exposed as an image-generation reason.");
  assertCheck("fallback.image_cue", !/Fallback sidecar used/iu.test(visualResult.imageCue.visualContext ?? ""), "Fallback sidecar text is not leaked into the image cue.");

  const quietState = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      triggerMode: "realtime_auto"
    },
    messages: [
      createAssistantMessage(seedState, "The team quietly reviews the budget ledger and next deadline.")
    ]
  });
  const quietResult = await runSimulationTurn(quietState, "계획을 계속 정리해", false);

  assertCheck("fallback.image_cue", !quietResult.imageJob, "Non-visual fallback turns remain suppressed.");
  assertCheck("fallback.image_cue", quietResult.imageCue.shouldGenerate === false, "Fallback without visual scene keywords does not force image generation.");
}

async function evaluateNovelAiV4Payload(seedState, planImageJob, generateNovelAiImages) {
  const state = createPlayableState(seedState, {
    memoryEvents: [],
    characters: [
      {
        id: "char_rin",
        simulationId: seedState.simulation.id,
        name: "Rin",
        role: "girl protagonist",
        summary: "female lead with red hair",
        relationship: "ally",
        currentMood: "focused"
      },
      {
        id: "char_kai",
        simulationId: seedState.simulation.id,
        name: "Kai",
        role: "boy rival",
        summary: "male rival with blue hair",
        relationship: "rival",
        currentMood: "tense"
      }
    ],
    visualProfiles: [
      {
        id: "visual_rin",
        simulationId: seedState.simulation.id,
        characterId: "char_rin",
        displayName: "Rin",
        positivePrompt: "red hair, green eyes, school jacket",
        negativePrompt: "bad hands",
        outfitPrompts: {
          "학교": "red academy blazer, pleated skirt",
          practice: "red practice hoodie, track pants"
        },
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      },
      {
        id: "visual_kai",
        simulationId: seedState.simulation.id,
        characterId: "char_kai",
        displayName: "Kai",
        positivePrompt: "blue hair, gray eyes, dark uniform",
        negativePrompt: "bad anatomy",
        outfitPrompts: {
          school: "navy academy blazer, pressed slacks",
          "연습": "navy practice jersey, sneakers"
        },
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe"
      }
    ]
  });
  const cue = createCue(state, {
    characters: ["char_rin", "char_kai"],
    tags: ["side-by-side", "tense expression", "school uniform"],
    scene: "school interior",
    visualContext: "side-by-side confrontation, hallway light"
  });
  const job = planImageJob(state, "turn_v4", cue, [], true);
  const result = await generateNovelAiImages({
    state,
    prompt: job.prompt,
    negativePrompt: job.negativePrompt,
    cue,
    count: 1
  });
  const parameters = result.payload.parameters ?? {};
  const v4Prompt = parameters.v4_prompt ?? {};
  const v4Negative = parameters.v4_negative_prompt ?? {};
  const charCaptions = v4Prompt.caption?.char_captions ?? [];
  const negativeCharCaptions = v4Negative.caption?.char_captions ?? [];

  assertCheck("nai.v4", !/\b1(?:girl|boy|other)\b/iu.test(job.prompt), "Mixed-character prompt does not infer subject count tags locally.");
  assertCheck("nai.v4", charCaptions.length === 2, "V4 payload creates one character caption per visible character when LLM tags contain character-related details.");
  assertCheck("nai.v4", negativeCharCaptions.length === 0, "V4 payload does not create character-specific negative captions from visible character ids.");
  assertCheck("nai.v4", parameters.use_coords === true && v4Prompt.use_coords === true, "V4 multi-character payload enables coordinate nudges when character captions are present.");
  assertCheck("nai.v4", charCaptions.every((caption) => /tense expression|school uniform/iu.test(caption?.char_caption ?? "")), "V4 character captions receive LLM-authored character tags.");
  assertCheck("nai.v4", !/girl, red hair/iu.test(charCaptions[0]?.char_caption ?? ""), "First character caption does not inject character-specific visual tags.");
  assertCheck("nai.v4", !/boy, blue hair/iu.test(charCaptions[1]?.char_caption ?? ""), "Second character caption does not inject character-specific visual tags.");
  assertCheck("nai.v4", !/red academy blazer, pleated skirt/iu.test(charCaptions[0]?.char_caption ?? ""), "Korean school outfit mapping is not applied locally to V4 character captions.");
  assertCheck("nai.v4", !/navy academy blazer, pressed slacks/iu.test(charCaptions[1]?.char_caption ?? ""), "English school outfit mapping is not applied locally to V4 character captions.");
  assertCheck(
    "nai.v4",
    Array.isArray(job.providerPayload.promptLayers?.characters) &&
      !job.providerPayload.promptLayers.characters.some((tag) => /red academy blazer|navy academy blazer/iu.test(String(tag))),
    "Character outfit mappings are not locally recorded in prompt layers."
  );
}

async function evaluateNovelAiVibeTransferPayload(seedState, planImageJob, generateNovelAiImages) {
  const encodedVibe = "ENCODEDVIBEDATA==";
  const rawImage = "data:image/png;base64,QUJD"; // stripped -> "QUJD"

  // --- v4/v4.5: encode된 vibe만 reference_image_multiple로 전송 ---
  const v4State = createPlayableState(seedState, {
    novelAi: {
      ...seedState.novelAi,
      enabled: false,
      modelPreset: "NAID4.5F",
      vibeTransferEnabled: true,
      vibeTransferReferences: [
        {
          id: "vibe_ref_1",
          name: "ref-1.png",
          image: rawImage,
          referenceStrength: 0.6,
          informationExtracted: 1,
          encodedVibe,
          encodedModel: "NAID4.5F",
          encodedInformationExtracted: 1
        }
      ]
    }
  });
  const v4Cue = createCue(v4State, { tags: ["smile"], scene: "studio" });
  const v4Job = planImageJob(v4State, "turn_vibe_v4", v4Cue, [], true);
  const v4Payload = (await generateNovelAiImages({ state: v4State, prompt: v4Job.prompt, negativePrompt: v4Job.negativePrompt, cue: v4Cue, count: 1 })).payload;
  const v4Params = v4Payload.parameters ?? {};

  assertCheck(
    "nai.vibe",
    Array.isArray(v4Params.reference_image_multiple) && v4Params.reference_image_multiple[0] === encodedVibe,
    "V4 vibe transfer sends the encoded vibe in parameters.reference_image_multiple."
  );
  assertCheck(
    "nai.vibe",
    Array.isArray(v4Params.reference_strength_multiple) && v4Params.reference_strength_multiple[0] === 0.6,
    "V4 vibe transfer sends reference_strength_multiple parallel to the encoded vibe."
  );
  assertCheck(
    "nai.vibe",
    v4Params.reference_information_extracted_multiple === undefined,
    "V4 vibe transfer omits reference_information_extracted_multiple (baked into the encode step)."
  );

  // --- v4: 아직 인코딩되지 않은 참조는 전송하지 않음 (raw 이미지로 NAI가 거부되는 것을 방지) ---
  const v4Unencoded = createPlayableState(seedState, {
    novelAi: {
      ...seedState.novelAi,
      enabled: false,
      modelPreset: "NAID4.5F",
      vibeTransferEnabled: true,
      vibeTransferReferences: [
        { id: "vibe_ref_2", name: "ref-2.png", image: rawImage, referenceStrength: 0.6, informationExtracted: 1 }
      ]
    }
  });
  const v4UnencodedCue = createCue(v4Unencoded, { tags: ["smile"] });
  const v4UnencodedJob = planImageJob(v4Unencoded, "turn_vibe_v4_raw", v4UnencodedCue, [], true);
  const v4UnencodedParams =
    (await generateNovelAiImages({ state: v4Unencoded, prompt: v4UnencodedJob.prompt, negativePrompt: v4UnencodedJob.negativePrompt, cue: v4UnencodedCue, count: 1 })).payload.parameters ?? {};
  assertCheck(
    "nai.vibe",
    v4UnencodedParams.reference_image_multiple === undefined,
    "V4 vibe transfer does not send un-encoded references (encode required before generation)."
  );

  // --- v3: 원본 이미지 base64 + information extracted 직접 전송 ---
  const v3State = createPlayableState(seedState, {
    novelAi: {
      ...seedState.novelAi,
      enabled: false,
      modelPreset: "NAID3",
      vibeTransferEnabled: true,
      vibeTransferReferences: [
        { id: "vibe_ref_3", name: "ref-3.png", image: rawImage, referenceStrength: 0.5, informationExtracted: 0.8 }
      ]
    }
  });
  const v3Cue = createCue(v3State, { tags: ["smile"] });
  const v3Job = planImageJob(v3State, "turn_vibe_v3", v3Cue, [], true);
  const v3Params = (await generateNovelAiImages({ state: v3State, prompt: v3Job.prompt, negativePrompt: v3Job.negativePrompt, cue: v3Cue, count: 1 })).payload.parameters ?? {};
  assertCheck(
    "nai.vibe",
    Array.isArray(v3Params.reference_image_multiple) && v3Params.reference_image_multiple[0] === "QUJD",
    "V3 vibe transfer sends the raw image base64 (data URL prefix stripped) in reference_image_multiple."
  );
  assertCheck(
    "nai.vibe",
    Array.isArray(v3Params.reference_information_extracted_multiple) && v3Params.reference_information_extracted_multiple[0] === 0.8,
    "V3 vibe transfer sends reference_information_extracted_multiple."
  );

  // --- vibe transfer 비활성 시 어떤 reference 파라미터도 추가하지 않음 ---
  const offState = createPlayableState(seedState, {
    novelAi: {
      ...seedState.novelAi,
      enabled: false,
      modelPreset: "NAID4.5F",
      vibeTransferEnabled: false,
      vibeTransferReferences: [
        { id: "vibe_ref_4", name: "ref-4.png", image: rawImage, referenceStrength: 0.6, informationExtracted: 1, encodedVibe }
      ]
    }
  });
  const offCue = createCue(offState, { tags: ["smile"] });
  const offJob = planImageJob(offState, "turn_vibe_off", offCue, [], true);
  const offParams = (await generateNovelAiImages({ state: offState, prompt: offJob.prompt, negativePrompt: offJob.negativePrompt, cue: offCue, count: 1 })).payload.parameters ?? {};
  assertCheck(
    "nai.vibe",
    offParams.reference_image_multiple === undefined,
    "Disabled vibe transfer adds no reference image parameters even when an encoded reference exists."
  );
}

async function evaluateRegisteredCharacterPromptInjection(seedState, planImageJob, generateNovelAiImages) {
  const state = createPlayableState(seedState, {
    characters: [
      {
        id: "char_f",
        simulationId: seedState.simulation.id,
        name: "Mira",
        role: "girl protagonist",
        summary: "registered female lead",
        relationship: "",
        currentMood: "tense"
      }
    ],
    visualProfiles: [
      {
        id: "visual_f",
        simulationId: seedState.simulation.id,
        characterId: "char_f",
        displayName: "Mira",
        positivePrompt: "pink rolled long hair, police style, futanari",
        negativePrompt: "bad hands",
        defaultOutfitPrompt: "navy crop top, peaked cap",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "explicit"
      }
    ],
    memoryEvents: [createStateMemoryEvent({ characters: [{ id: "char_f", name: "Mira" }], simulation: seedState.simulation }, "char_f", "Wearing", "torn police uniform, navy short dress")]
  });
  const cue = createCue(state, {
    characters: ["char_f"],
    tags: ["bedroom", "indoors"],
    scene: "bedroom",
    visualContext: "bedroom, indoors",
    characterPrompts: [
      { characterId: "char_f", prompt: "missionary position, tears, wide eyed" },
      { prompt: "muscular man, brute, smashing" }
    ]
  });
  const job = planImageJob(state, "turn_inject", cue, [], true);
  const composed = Array.isArray(job.providerPayload.characterPrompts) ? job.providerPayload.characterPrompts : [];
  const female = composed[0]?.prompt ?? "";
  const male = composed[1]?.prompt ?? "";

  assertCheck("context.inject", composed.length === 2, "Each visible character (registered + unregistered) gets its own separate character prompt entry.");
  assertCheck("context.inject", /missionary position/iu.test(female) && /pink rolled long hair/iu.test(female), "Registered character caption injects the saved base appearance alongside the LLM action.");
  assertCheck("context.inject", female.indexOf("missionary position") < female.indexOf("pink rolled long hair"), "LLM action/expression tags are ordered before the injected saved appearance.");
  assertCheck("context.inject", /torn police uniform/iu.test(female) && !/navy crop top/iu.test(female), "Current stored Wearing outfit is injected and overrides the default outfit.");
  assertCheck("context.inject", /muscular man/iu.test(male) && !/pink rolled long hair/iu.test(male), "Unregistered second character keeps its own caption and never borrows another character's appearance.");
  assertCheck("context.inject", !/muscular man|pink rolled long hair/iu.test(job.prompt), "Per-character tags stay out of the base prompt.");

  const payloadCue = { ...cue, characterPrompts: job.providerPayload.cue?.characterPrompts ?? composed };
  const result = await generateNovelAiImages({ state, prompt: job.prompt, negativePrompt: job.negativePrompt, cue: payloadCue, count: 1 });
  const charCaptions = result.payload.parameters?.v4_prompt?.caption?.char_captions ?? [];
  assertCheck("context.inject", charCaptions.length === 2, "NovelAI V4 payload carries one char_caption per visible character including the injected registered character.");
  assertCheck("context.inject", /pink rolled long hair/iu.test(charCaptions[0]?.char_caption ?? "") && /torn police uniform/iu.test(charCaptions[0]?.char_caption ?? ""), "NovelAI V4 char_caption includes the injected saved appearance and current outfit.");
}

function createPlayableState(seedState, overrides = {}) {
  const state = structuredClone(seedState);
  Object.assign(state, overrides);
  state.simulation = {
    ...state.simulation,
    realtimeImageEnabled: true,
    ...(overrides.simulation ?? {})
  };
  state.imageProfile = {
    ...seedState.imageProfile,
    countMin: 1,
    countMax: 1,
    triggerMode: "manual",
    enabled: true,
    ...(overrides.imageProfile ?? {})
  };
  state.llm = {
    ...seedState.llm,
    enabled: false,
    provider: "mock",
    apiKey: "",
    ...(overrides.llm ?? {})
  };
  state.novelAi = {
    ...seedState.novelAi,
    enabled: false,
    modelPreset: "NAID4.5C",
    ...(overrides.novelAi ?? {})
  };
  state.neuralMap = {
    ...seedState.neuralMap,
    enabled: false,
    ...(overrides.neuralMap ?? {})
  };
  state.messages = overrides.messages ?? state.messages;
  state.characters = overrides.characters ?? state.characters;
  state.visualProfiles = overrides.visualProfiles ?? state.visualProfiles;
  state.imageJobs = [];
  state.imageAssets = [];
  return state;
}

function createAssistantMessage(state, content) {
  return {
    id: `msg_eval_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    role: "assistant",
    content,
    createdAt: new Date().toISOString(),
    referencedNodeIds: [],
    imageAssetIds: []
  };
}

function createUserMessage(state, content) {
  return {
    id: `msg_eval_user_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    role: "user",
    content,
    createdAt: new Date().toISOString(),
    referencedNodeIds: [],
    imageAssetIds: []
  };
}

function createContextPack(state) {
  return {
    id: `ctx_eval_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    source: "mock",
    objective: "image reuse eval",
    query: "image reuse eval",
    evidence: [],
    tokenBudget: 0,
    createdAt: new Date().toISOString()
  };
}

function createParsedLlmSidecarTrace(state, turnId) {
  return {
    id: `sidecar_eval_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    turnId,
    source: "llm",
    status: "parsed",
    errors: [],
    createdAt: new Date().toISOString()
  };
}

function createStateMemoryEvent(state, actorId, stateType, value) {
  const actor = state.characters.find((character) => character.id === actorId);
  return {
    id: `memory_eval_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    actorId,
    actorName: actor?.name,
    content: `[State] ${actor?.name ?? actorId} ${stateType} = ${value}`,
    importance: 0.76,
    tags: ["memory-delta", "kind:state", `state:${stateType}`],
    sourceTurnId: `turn_eval_${Math.random().toString(36).slice(2)}`,
    createdAt: new Date().toISOString(),
    metadata: {
      memory_kind: "state",
      state_type: stateType,
      value,
      owner_id: actorId
    }
  };
}

function createSceneStateMemoryEvent(state, stateType, value) {
  return {
    id: `memory_eval_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    content: `[State] scene ${stateType} = ${value}`,
    importance: 0.76,
    tags: ["memory-delta", "kind:state", `state:${stateType}`],
    sourceTurnId: `turn_eval_${Math.random().toString(36).slice(2)}`,
    createdAt: new Date().toISOString(),
    metadata: {
      memory_kind: "state",
      state_type: stateType,
      value
    }
  };
}

function createCue(state, overrides = {}) {
  return {
    shouldGenerate: true,
    reason: "image prompt quality eval",
    characters: [state.characters[0]?.id].filter(Boolean),
    tags: [],
    scene: "current simulation scene",
    visualContext: "",
    ...overrides
  };
}

function assertCheck(scope, passed, message) {
  checks.push({
    level: passed ? "pass" : "fail",
    scope,
    message
  });
}
