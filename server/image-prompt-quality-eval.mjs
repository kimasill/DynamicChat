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
  const { findReusableImageAsset, getReusableTagsFromJob, planImageJob } = await vite.ssrLoadModule("/src/services/imageOrchestrator.ts");
  const { generateNovelAiImages } = await vite.ssrLoadModule("/src/services/novelAiClient.ts");

  await evaluateRecentContextTagDetection(seedState, runSimulationTurn);
  evaluateNovelAiWeightingAndUserRules(seedState, planImageJob);
  await evaluateReusableImageMatching(seedState, planImageJob, findReusableImageAsset, getReusableTagsFromJob, planImageJobForCompletedTurn);
  await evaluateImageUserRuleCuePlanning(seedState, planImageJobForCompletedTurn);
  await evaluateImageGenerationCadence(seedState, planImageJobForCompletedTurn);
  await evaluateAnchoredRulePromptDiversity(seedState, planImageJobForCompletedTurn);
  evaluateAdultContentRating(seedState, planImageJob);
  await evaluateSceneSpecificity(seedState, runSimulationTurn);
  await evaluateNoisyCueTagFiltering(seedState, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateImageCueCharacterScoping(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateLlmCharacterScopedOutfits(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateStructuredCueDisambiguation(seedState, planImageJob, planImageJobForCompletedTurn);
  await evaluatePersonaCharacterImageCueScoping(seedState, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateLlmImageStateTagCarryover(seedState, planImageJob, planImageJobForCompletedTurn, generateNovelAiImages);
  await evaluateLlmNaiTagPreservation(seedState, planImageJob, generateNovelAiImages);
  await evaluateActorTargetCharacterDisambiguation(seedState, planImageJob, planImageJobForCompletedTurn);
  await evaluateFallbackImageCueSuppression(seedState, runSimulationTurn);
  await evaluateNovelAiV4Payload(seedState, planImageJob, generateNovelAiImages);
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
  assertCheck("context.library", /library|bookshelf/iu.test(cueText), "Deictic first image requests can use the immediate scene context.");
  assertCheck("context.library", !/\b(?:rain|food|table|dining table)\b/iu.test(prompt), "Archive context does not auto-inject old rain/food/table tags.");
  assertCheck("context.library", /silver hair/iu.test(characterLayer), "Visible character profile remains in the character prompt layer.");
  assertCheck("context.library", !/silver hair/iu.test(prompt), "NovelAI V4 base prompt keeps character prompt tags out of the generated scene prompt.");
  assertCheck("context.library", !/\b1(?:girl|boy|other)\b/iu.test(prompt), "Subject count tags are not inferred by app-side character mapping.");
}

async function evaluateReusableImageMatching(seedState, planImageJob, findReusableImageAsset, getReusableTagsFromJob, planImageJobForCompletedTurn) {
  const state = createPlayableState(seedState, {
    imageProfile: {
      ...seedState.imageProfile,
      artistPrompt: "::artist:sample_artist ::",
      qualityPrompt: "masterpiece, best quality, highly detailed skin"
    }
  });
  const cue = createCue(state, {
    scene: "archive library",
    tags: ["bookshelf", "silver hair", "brass key"],
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
      dataUrl: "data:image/png;base64,AA=="
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
    tags: ["archive library", "bookshelf", "standing", "holding key", "worried expression"],
    reuseTags: ["archive library", "bookshelf", "standing", "holding key", "worried expression"],
    providerMetadata: {
      cue: {
        characters: actionCue.characters,
        scene: "archive library",
        tags: ["standing", "holding key", "worried expression"],
        visualContext: "archive library, bookshelf, standing, holding key, worried expression"
      }
    }
  };
  const actionJob = planImageJob({ ...reuseState, imageAssets: [actionReusableAsset] }, "turn_reuse_action_exact", actionCue, [], false);
  const exactActionMatch = findReusableImageAsset({ ...reuseState, imageAssets: [actionReusableAsset] }, actionJob);
  const mismatchedPoseMatch = findReusableImageAsset(
    {
      ...reuseState,
      imageAssets: [
        {
          ...actionReusableAsset,
          id: "asset_reuse_pose_mismatch",
          tags: ["archive library", "bookshelf", "sitting", "holding key", "worried expression"],
          reuseTags: ["archive library", "bookshelf", "sitting", "holding key", "worried expression"],
          providerMetadata: {
            cue: {
              characters: actionCue.characters,
              scene: "archive library",
              tags: ["sitting", "holding key", "worried expression"],
              visualContext: "archive library, bookshelf, sitting, holding key, worried expression"
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
          tags: ["archive library", "bookshelf", "standing", "holding key", "smile"],
          reuseTags: ["archive library", "bookshelf", "standing", "holding key", "smile"],
          providerMetadata: {
            cue: {
              characters: actionCue.characters,
              scene: "archive library",
              tags: ["standing", "holding key", "smile"],
              visualContext: "archive library, bookshelf, standing, holding key, smile"
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
  assertCheck("image.reuse", !mismatchedMetadataCharacterMatch, "Generated image reuse rejects provider metadata character-scope mismatches.");
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

  assertCheck("image.user_rules", plan.imageJobs.length >= 2, "User image rules add required scene/dialogue cues when the LLM omits image_cues.");
  assertCheck("image.user_rules", cueKinds.includes("scene") && cueKinds.includes("dialogue_face"), "Rule-backed cue planning preserves scene and dialogue-face cue kinds.");
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

  assertCheck("image.cadence", paragraphPlan.imageJobs.length >= 3, "Paragraph cadence creates image jobs for multiple assistant paragraphs when the LLM omits image_cues.");
  assertCheck("image.cadence", paragraphPlan.imageJobs.every((job) => job.providerPayload.generationCadence === "paragraph"), "Image jobs record the active generation cadence.");
  assertCheck("image.cadence", paragraphPlan.imageJobs.every((job) => job.providerPayload.forceFreshImage === true), "Paragraph cadence jobs bypass cooldown/reuse as fresh cuts.");
  assertCheck("image.cadence", paragraphPlan.imageJobs.some((job) => job.providerPayload.imageCuePlanner?.source === "image_generation_cadence"), "Paragraph cadence records cadence planner source.");

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

  assertCheck("image.anchored_rules", jobs.length >= 4, "Rule and paragraph planning creates multiple anchored image jobs.");
  assertCheck("image.anchored_rules", new Set(prompts).size > 1, "Anchored image jobs do not reuse one identical prompt for every cut.");
  assertCheck("image.anchored_rules", /stage|stage lights/iu.test(actionPrompt) && /arm up|hand up|microphone/iu.test(actionPrompt), "Action cue prompt keeps the stage action and prop tags.");
  assertCheck("image.anchored_rules", !/\bstreet\b/iu.test(actionPrompt), "Stage action cue does not inherit later street tags from the same assistant turn.");
  assertCheck("image.anchored_rules", /close-up|body focus/iu.test(bodyPrompt) && /hands|wrist grab/iu.test(bodyPrompt), "Body-detail cue uses close framing and body/contact tags.");
  assertCheck("image.anchored_rules", /close-up|face focus/iu.test(dialoguePrompt) && /open mouth/iu.test(dialoguePrompt), "Dialogue-face cue uses face and speech-expression tags.");
  assertCheck("image.anchored_rules", byKind.get("dialogue_face")?.providerPayload.cuePlacement === "before", "Dialogue-face cue is placed before its anchor text.");
  assertCheck("image.anchored_rules", /시작할게요|속삭/iu.test(dialogueAnchor), "Dialogue-face cue anchors to the dialogue line, not a random paragraph.");
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
      tags: ["eye focus"],
      visualContext: "eye focus"
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
  const eyeFocusIndex = weightedPrompt.indexOf("eye focus");

  assertCheck("nai.weighting", job.prompt.includes("1.5::rain, night ::"), "Numeric emphasis with comma remains intact.");
  assertCheck("nai.weighting", job.prompt.includes("-1::hat ::"), "Negative numeric emphasis stays in the prompt for targeted NAI removal.");
  assertCheck("nai.weighting", job.prompt.includes("::artist:bm94199 ::"), "Artist emphasis tag preserves the user-provided closing-space syntax.");
  assertCheck("nai.weighting", artistIndex >= 0 && firstSceneIndex >= 0 && artistIndex < firstSceneIndex, "Artist prompt stays before generated scene/action tags.");
  assertCheck("nai.weighting", variantPrompts.length === 3 && new Set(variantPrompts).size === 3, "Counted image jobs receive distinct prompt variants instead of repeating the same prompt.");
  assertCheck("nai.weighting", variantPrompts.every((prompt) => /stage/iu.test(prompt) && /standing|holding microphone/iu.test(prompt)), "Prompt variants preserve the core scene and action tags.");
  assertCheck(
    "nai.weighting",
    artistLayerIndex >= 0 &&
      qualityLayerIndex > artistLayerIndex &&
      styleLayerIndex > qualityLayerIndex &&
      eyeFocusIndex > styleLayerIndex,
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

  assertCheck("context.characters", plan.imageCue.characters.length === 0, "Solo outsider context drops copied full-roster image cue characters.");
  assertCheck("context.characters", !/\b(?:ari|beni|ciel)\b/iu.test(plan.imageCue.tags.join(", ")), "Roster character names are not carried as image cue tags.");
  assertCheck("context.characters", !/\b3girls\b|red hair|blue hair|blonde hair/iu.test(job?.prompt ?? ""), "Solo outsider prompt does not include the configured three-girl roster.");
  assertCheck("context.characters", !/red hair|blue hair|blonde hair|\b(?:ari|beni|ciel)\b/iu.test(directJob.prompt), "Final image job planning strips copied roster character prompts and roster-name tags even when a raw cue bypasses simulation planning.");
  assertCheck("context.characters", !directJob.providerPayload.positiveTags?.some?.((tag) => /^(?:ari|beni|ciel)$/iu.test(String(tag))), "Roster names are removed from final positive tag layers.");
  assertCheck("context.characters", Array.isArray(directJob.providerPayload.cue?.characters) && directJob.providerPayload.cue.characters.length === 0, "Final image job payload stores the scoped empty character list.");

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
    assertCheck("context.characters", charCaptions.length === 0, "Solo outsider V4 payload does not write character captions for absent roster characters.");
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
  assertCheck("context.llm_characters", /red hair/iu.test(layerText) && /blue hair/iu.test(layerText), "Selected LLM character ids pull their configured visual profile tags.");
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
    assertCheck("context.llm_characters", charCaptions.length === 2, "NovelAI V4 creates captions only for the LLM-selected visible characters.");
    assertCheck("context.llm_characters", /red hair/iu.test(captionText) && /blue hair/iu.test(captionText) && !/blonde hair/iu.test(captionText), "V4 captions use configured tags for selected characters only.");
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
  const redHairIndex = characterTags.findIndex((tag) => /red hair/iu.test(String(tag)));
  const ariOutfitIndex = characterTags.findIndex((tag) => /black performance jacket/iu.test(String(tag)));
  const blueHairIndex = characterTags.findIndex((tag) => /blue hair/iu.test(String(tag)));
  const beniOutfitIndex = characterTags.findIndex((tag) => /white ribbon dress/iu.test(String(tag)));

  assertCheck("context.llm_outfit", redHairIndex >= 0 && ariOutfitIndex > redHairIndex, "Ari's LLM Wearing tags are appended after Ari's configured character prompt tags.");
  assertCheck("context.llm_outfit", blueHairIndex >= 0 && beniOutfitIndex > blueHairIndex, "Beni's LLM Wearing tags are appended after Beni's configured character prompt tags.");
  assertCheck("context.llm_outfit", !characterTags.some((tag) => /gray casual cardigan|blonde hair/iu.test(String(tag))), "Outfit application does not pull tags for an unselected roster character.");

  const outfitPayload = await generateNovelAiImages({
    state: outfitState,
    prompt: outfitJob.prompt,
    negativePrompt: outfitJob.negativePrompt,
    cue: outfitCue,
    count: 1
  });
  const captions = outfitPayload.payload.parameters?.v4_prompt?.caption?.char_captions ?? [];
  assertCheck("context.llm_outfit", /red hair.*black performance jacket/isu.test(captions[0]?.char_caption ?? ""), "Ari V4 caption keeps configured tags before LLM outfit tags.");
  assertCheck("context.llm_outfit", /blue hair.*white ribbon dress/isu.test(captions[1]?.char_caption ?? ""), "Beni V4 caption keeps configured tags before LLM outfit tags.");
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

  assertCheck("context.actor_target", plan.imageCue.characters.length === 0, "Unnamed external actor drops wrongly selected roster character from planned image cue.");
  assertCheck("context.actor_target", !/\b(?:light blonde hair|twin tails|twintails)\b/iu.test(job?.prompt ?? ""), "Wrong target character visual tags are not applied to the unnamed male actor.");
  assertCheck("context.actor_target", !/\b1boy\b/iu.test(job?.prompt ?? ""), "Unnamed Korean male actor cue does not infer 1boy locally.");
  assertCheck("context.actor_target", !/\bmale student\b/iu.test(job?.prompt ?? ""), "Unnamed Korean male actor cue does not infer gender/subject tags locally.");
  assertCheck("context.actor_target", /\b(?:evil smile|leering)\b/iu.test(job?.prompt ?? ""), "Expression tags stay attached to the external actor cue.");
  assertCheck("context.actor_target", !/\b(?:light blonde hair|twin tails|twintails)\b/iu.test(directJob.prompt), "Direct image job planning also strips wrong target character visuals for external actors.");
  assertCheck("context.actor_target", Array.isArray(directJob.providerPayload.cue?.characters) && directJob.providerPayload.cue.characters.length === 0, "Direct image job payload stores no roster character for unnamed external actors.");
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
    tags: ["filming set", "getting up", "props", "microphone", "speaking"],
    scene: "filming set",
    visualContext: "Ari speaking, filming set, props",
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

  assertCheck("context.structured_cue", plan.imageCue.characters.join(",") === "char_a", "Focused LLM cue keeps only the speaking/action-subject roster character.");
  assertCheck("context.structured_cue", !/\b(?:filming set|getting up|props?)\b/iu.test(finalPromptText), "LLM-provided placeholder cue tags are removed before the NAI prompt.");
  assertCheck("context.structured_cue", /microphone|open mouth/iu.test(combinedPrompt), "Concrete object/action tags remain after NAI tag normalization.");

  const outsiderCue = createCue(state, {
    characters: ["char_a"],
    tags: ["teacher", "speaking"],
    scene: "classroom",
    visualContext: "teacher speaking to viewer"
  });
  const outsiderJob = planImageJob(state, "turn_teacher_outsider", outsiderCue, [], true);
  assertCheck("context.structured_cue", outsiderJob.providerPayload.cue?.characters?.length === 0, "Non-roster speaker/action-subject drops wrongly selected roster character.");
  assertCheck("context.structured_cue", !/red hair/iu.test(outsiderJob.prompt), "Non-roster speaker cue does not include a selected character visual prompt.");

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
  assertCheck("context.structured_cue", /registered navy uniform/iu.test(characterLayer), "Current Wearing memory is resolved through the character's registered outfit mapping.");
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
  assertCheck("context.nai_tags", /\bstanding\b/iu.test(prompt) && /\bchair\b/iu.test(prompt) && !/standing up from chair/iu.test(prompt), "Natural action phrases are decomposed into NAI tags.");
  assertCheck("context.nai_tags", /looking at viewer/iu.test(prompt), "Concrete gaze/POV tag is preserved.");
  assertCheck("context.nai_tags", /holding notebook/iu.test(prompt), "Held-item/action tag is preserved.");
  assertCheck("context.nai_tags", /dutch angle/iu.test(prompt), "Positive image user-rule tag is included.");
  assertCheck("context.nai_tags", /watermark/iu.test(job.negativePrompt), "Negative image user-rule tag is included in Undesired Content.");
  assertCheck("context.nai_tags", !/facial expression|visible emotional reaction|situation specific clothing|street/iu.test(prompt), "Abstract cue tags and contradictory scene tags are filtered from the NAI prompt.");
  const characterLayer = job.providerPayload.promptLayers?.characters?.join(", ") ?? "";
  assertCheck("context.nai_tags", /school uniform/iu.test(characterLayer) && /dark grey pencil skirt/iu.test(characterLayer) && /tight fit/iu.test(characterLayer) && /necktie/iu.test(characterLayer), "Registered outfit keyword mappings are expanded beyond the generic clothing tag.");

  const payloadResult = await generateNovelAiImages({
    state,
    prompt: job.prompt,
    negativePrompt: job.negativePrompt,
    cue,
    count: 1
  });
  const baseCaption = payloadResult.payload.parameters?.v4_prompt?.caption?.base_caption ?? "";
  const charCaption = payloadResult.payload.parameters?.v4_prompt?.caption?.char_captions?.[0]?.char_caption ?? "";
  assertCheck("context.nai_tags", /1girl.*looking at viewer.*classroom.*standing/isu.test(baseCaption), "NovelAI V4 base caption keeps ordered subject/gaze/environment/action tags from the LLM cue.");
  assertCheck("context.nai_tags", /pink hair.*school uniform.*dark grey pencil skirt.*tight fit.*necktie/isu.test(charCaption), "NovelAI V4 character caption keeps identity plus full registered outfit mapping.");
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

  assertCheck("context.persona_actor", plan.imageCue.characters.join(",") === "char_player", "User-controlled character persona is selected when the user action is the visible image actor.");
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
    assertCheck("context.persona_actor", /red hair/iu.test(charCaption) && !/silver hair/iu.test(charCaption), "NovelAI V4 character caption uses the controlled persona character prompt only.");
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

  assertCheck("context.image_state", plan.imageJob, "LLM-owned image state tags can rescue a sparse generated image cue.");
  assertCheck("context.image_state", plan.imageCue.characters.join(",") === "char_player_state", "Sparse persona cue still resolves the controlled character as actor.");
  assertCheck("context.image_state", /standing up from chair|holding notebook/iu.test(plan.imageCue.visualContext ?? ""), "Current character image state tags enter the planned cue visual context.");

  if (plan.imageJob) {
    const characterLayer = plan.imageJob.providerPayload.promptLayers?.characters?.join(", ") ?? "";
    assertCheck("context.image_state", /\bstanding\b|leaning forward|holding notebook/iu.test(characterLayer), "Current character image state tags enter the character prompt layer.");
    assertCheck("context.image_state", /classroom|low angle shot/iu.test(plan.imageJob.prompt), "Current scene image state tags enter the base prompt.");

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
      /red hair/iu.test(charCaption) && /\bstanding\b/iu.test(charCaption) && /leaning forward|holding notebook/iu.test(charCaption),
      "NovelAI V4 character caption includes LLM-owned current image state tags."
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
  assertCheck("nai.v4", charCaptions.length === 2, "V4 payload includes one character caption per visible character.");
  assertCheck("nai.v4", negativeCharCaptions.length === 2, "V4 payload includes character-specific negative captions.");
  assertCheck("nai.v4", parameters.use_coords === true && v4Prompt.use_coords === true, "V4 multi-character payload enables coordinate nudges.");
  assertCheck("nai.v4", /girl, red hair/iu.test(charCaptions[0]?.char_caption ?? ""), "First character caption keeps character-specific visual tags.");
  assertCheck("nai.v4", /boy, blue hair/iu.test(charCaptions[1]?.char_caption ?? ""), "Second character caption keeps character-specific visual tags.");
  assertCheck("nai.v4", /red academy blazer, pleated skirt/iu.test(charCaptions[0]?.char_caption ?? ""), "Korean school outfit mapping is applied to V4 character captions.");
  assertCheck("nai.v4", /navy academy blazer, pressed slacks/iu.test(charCaptions[1]?.char_caption ?? ""), "English school outfit mapping is applied to V4 character captions.");
  assertCheck(
    "nai.v4",
    Array.isArray(job.providerPayload.promptLayers?.characters) &&
      job.providerPayload.promptLayers.characters.some((tag) => /red academy blazer/iu.test(String(tag))) &&
        job.providerPayload.promptLayers.characters.some((tag) => /navy academy blazer/iu.test(String(tag))),
    "Character outfit mappings are recorded in prompt layers."
  );
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
