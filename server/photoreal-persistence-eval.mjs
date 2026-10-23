import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
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

// Spelled out instead of read from the module under test, so changing a default is a deliberate edit in two places
// rather than something this eval agrees with by construction.
const EXPECTED_PHOTOREAL_DEFAULTS = {
  enabled: false,
  provider: "comfyui",
  requestMode: "proxy",
  baseUrl: "http://127.0.0.1:8188",
  proxyUrl: "",
  workflowPreset: "qwen_image_2_1_turbo",
  referenceWorkflowPreset: "qwen_image_2_1_turbo",
  modelFiles: {},
  steps: 0,
  cfg: 0,
  width: 0,
  height: 0,
  seedFixed: false,
  stylePreset: "instagram",
  stylePrompt: "",
  appearancePrompt: "",
  negativePrompt: "",
  faceReferenceEnabled: false,
  vramHandoff: "auto",
  customWorkflow: "",
  timeoutSeconds: 300,
  registrationStatus: "idle"
};
const INLINE_BYTES = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk";

try {
  let modules;
  try {
    modules = {
      seed: await vite.ssrLoadModule("/src/data/seed.ts"),
      runtimeApiSettings: await vite.ssrLoadModule("/src/services/runtimeApiSettings.ts"),
      progressRuns: await vite.ssrLoadModule("/src/services/progressRuns.ts"),
      simulationRuns: await vite.ssrLoadModule("/src/services/simulationRuns.ts"),
      dynamicChatApi: await vite.ssrLoadModule("/src/services/dynamicChatApi.ts"),
      routing: await vite.ssrLoadModule("/src/services/imageBackendRouting.ts"),
      workflows: await vite.ssrLoadModule("/src/services/comfyWorkflows.ts")
    };
  } catch (error) {
    assertCheck("load", false, `Modules under test load through Vite SSR: ${describeError(error)}`);
  }

  if (modules) {
    await runGroup("defaults", () => evaluateDefaultsAndLegacyHydration(modules));
    await runGroup("photoreal.normalize", () => evaluatePhotorealSettingsNormalization(modules));
    await runGroup("routing.hydrate", () => evaluateBackendRoutingHydration(modules));
    await runGroup("visual-profiles", () => evaluateVisualProfileNormalization(modules));
    await runGroup("draft", () => evaluateCreateStateFromDraft(modules));
    await runGroup("jobs-assets", () => evaluateJobAndAssetBackendPersistence(modules));
    await runGroup("api-boundary", () => evaluatePersistenceBoundary(modules));
  }
} finally {
  await vite.close();
}

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}`);
}

const failures = checks.filter((check) => check.level === "fail");
console.log("");
console.log(`Photoreal persistence checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

function evaluateDefaultsAndLegacyHydration({ seed, runtimeApiSettings, routing }) {
  const { DEFAULT_PHOTOREAL_SETTINGS, builtInSimulationStates, hydrateState, seedState } = seed;
  const expectedRouting = { mode: "novelai", photorealCategories: ["portrait", "full_body", "scenery", "object"] };

  assertCheck(
    "defaults",
    isDeepStrictEqual(DEFAULT_PHOTOREAL_SETTINGS, EXPECTED_PHOTOREAL_DEFAULTS),
    "DEFAULT_PHOTOREAL_SETTINGS matches the spec defaults field for field."
  );
  assertCheck(
    "defaults",
    Object.isFrozen(DEFAULT_PHOTOREAL_SETTINGS) && Object.isFrozen(DEFAULT_PHOTOREAL_SETTINGS.modelFiles),
    "DEFAULT_PHOTOREAL_SETTINGS is frozen, modelFiles included."
  );
  assertCheck(
    "defaults",
    DEFAULT_PHOTOREAL_SETTINGS === runtimeApiSettings.DEFAULT_PHOTOREAL_SETTINGS &&
      seed.normalizePhotorealSettings === runtimeApiSettings.normalizePhotorealSettings,
    "seed.ts re-exports the one definition of the photoreal defaults and normalizer."
  );
  assertCheck(
    "defaults",
    isDeepStrictEqual(routing.DEFAULT_PHOTOREAL_CATEGORIES, expectedRouting.photorealCategories),
    "DEFAULT_PHOTOREAL_CATEGORIES is portrait / full_body / scenery / object."
  );
  assertCheck(
    "defaults",
    isDeepStrictEqual(seedState.photoreal, EXPECTED_PHOTOREAL_DEFAULTS),
    "seedState.photoreal holds the default photoreal settings."
  );
  assertCheck(
    "defaults",
    seedState.photoreal !== DEFAULT_PHOTOREAL_SETTINGS && !Object.isFrozen(seedState.photoreal) && !Object.isFrozen(seedState.photoreal.modelFiles),
    "seedState.photoreal is an editable copy, not the frozen default."
  );
  assertCheck(
    "defaults",
    isDeepStrictEqual(seedState.imageProfile.backendRouting, expectedRouting),
    "seedState routes every cut to NovelAI, with the default categories ready for mixed mode."
  );

  for (const builtIn of builtInSimulationStates) {
    const hydrated = hydrateState(builtIn);
    assertCheck(
      "defaults",
      isDeepStrictEqual(hydrated.photoreal, EXPECTED_PHOTOREAL_DEFAULTS) && hydrated.imageProfile.backendRouting?.mode === "novelai",
      `Built-in "${builtIn.simulation.title}" hydrates with photoreal off and NovelAI-only routing.`
    );
  }

  // A state saved before photoreal existed: no `photoreal`, no `imageProfile.backendRouting`.
  const legacy = cloneJson(seedState);
  delete legacy.photoreal;
  delete legacy.imageProfile.backendRouting;
  legacy.imageJobs = [createJob(seedState, { id: "job_legacy" })];
  legacy.imageAssets = [createAsset(seedState, { id: "asset_legacy" })];
  const hydratedLegacy = hydrateState(legacy);
  assertCheck(
    "defaults",
    isDeepStrictEqual(hydratedLegacy.photoreal, EXPECTED_PHOTOREAL_DEFAULTS),
    "An old state without photoreal hydrates to the default photoreal settings."
  );
  assertCheck(
    "defaults",
    isDeepStrictEqual(hydratedLegacy.imageProfile.backendRouting, expectedRouting),
    "An old state without backendRouting resolves to NovelAI-only routing."
  );
  assertCheck(
    "defaults",
    isDeepStrictEqual(hydrateState(undefined).photoreal, EXPECTED_PHOTOREAL_DEFAULTS),
    "hydrateState(undefined) carries the photoreal defaults."
  );

  // Throws (strict mode) if hydration handed out the frozen default's modelFiles.
  hydratedLegacy.photoreal.modelFiles.z_image_turbo = { diffusionModel: "edited.safetensors" };
  assertCheck(
    "defaults",
    isDeepStrictEqual(DEFAULT_PHOTOREAL_SETTINGS.modelFiles, {}) && isDeepStrictEqual(hydrateState(legacy).photoreal.modelFiles, {}),
    "Editing one hydrated state's modelFiles leaks into neither the default nor the next hydration."
  );

  const legacyProfile = hydratedLegacy.visualProfiles[0];
  assertCheck(
    "defaults",
    !("photorealPrompt" in legacyProfile) && !("photorealNegativePrompt" in legacyProfile) && !("profileImages" in legacyProfile),
    "Old visual profiles gain no photoreal keys."
  );
  const legacyJob = hydratedLegacy.imageJobs.find((job) => job.id === "job_legacy");
  const legacyAsset = hydratedLegacy.imageAssets.find((asset) => asset.id === "asset_legacy");
  assertCheck(
    "defaults",
    Boolean(legacyJob) && !("backend" in legacyJob) && !("routing" in legacyJob),
    "Old image jobs gain no backend/routing keys (absent still means NovelAI)."
  );
  assertCheck(
    "defaults",
    Boolean(legacyAsset) && !("backend" in legacyAsset) && !("sceneCategory" in legacyAsset),
    "Old image assets gain no backend/sceneCategory keys."
  );
}

function evaluatePhotorealSettingsNormalization({ seed, runtimeApiSettings, workflows }) {
  const { normalizePhotorealSettings, toShareablePhotorealSettings } = runtimeApiSettings;

  for (const [label, candidate] of [["undefined", undefined], ["null", null], ["a string", "comfyui"], ["an array", []], ["a number", 42]]) {
    assertCheck(
      "photoreal.normalize",
      isDeepStrictEqual(normalizePhotorealSettings(candidate), EXPECTED_PHOTOREAL_DEFAULTS),
      `A non-object candidate (${label}) normalizes to the defaults.`
    );
  }

  const malformed = {
    enabled: "yes",
    provider: "a1111",
    requestMode: "direct",
    baseUrl: 8188,
    proxyUrl: null,
    workflowPreset: "sdxl_turbo",
    referenceWorkflowPreset: "z_image_turbo",
    modelFiles: {
      z_image_turbo: { diffusionModel: "z_custom.safetensors", textEncoder: 5, vae: "ae.safetensors", lora: "extra.safetensors" },
      qwen_image_2_1: "qwen.safetensors",
      flux_dev: { diffusionModel: "flux.safetensors" },
      qwen_image_edit_2511: {}
    },
    steps: 500,
    cfg: Number.NaN,
    width: 1000,
    height: -64,
    seedFixed: 1,
    seed: 12.7,
    stylePreset: "anime",
    stylePrompt: ["soft light"],
    appearancePrompt: { glossy: true },
    negativePrompt: 3,
    faceReferenceEnabled: "false",
    vramHandoff: "always",
    customWorkflow: { nodes: [] },
    timeoutSeconds: 5,
    registrationStatus: "done",
    verifiedAt: 1700000000,
    verificationMessage: { ok: true },
    apiKey: "not-a-photoreal-field"
  };
  const normalized = normalizePhotorealSettings(malformed);
  const expectations = [
    ["enabled", false],
    ["provider", "comfyui"],
    ["requestMode", "proxy"],
    ["baseUrl", "http://127.0.0.1:8188"],
    ["proxyUrl", ""],
    ["workflowPreset", "qwen_image_2_1_turbo"],
    ["referenceWorkflowPreset", "z_image_turbo"],
    ["modelFiles", { z_image_turbo: { diffusionModel: "z_custom.safetensors", vae: "ae.safetensors" } }],
    ["steps", 100],
    ["cfg", 0],
    ["width", 1008],
    ["height", 0],
    ["seedFixed", false],
    ["seed", 12],
    ["stylePreset", "instagram"],
    ["stylePrompt", ""],
    ["appearancePrompt", ""],
    ["negativePrompt", ""],
    ["faceReferenceEnabled", false],
    ["vramHandoff", "auto"],
    ["customWorkflow", ""],
    ["timeoutSeconds", 30],
    ["registrationStatus", "idle"]
  ];
  for (const [field, expected] of expectations) {
    assertCheck(
      "photoreal.normalize",
      isDeepStrictEqual(normalized[field], expected),
      `Malformed ${field} becomes ${JSON.stringify(expected)} (got ${JSON.stringify(normalized[field])}).`
    );
  }
  assertCheck(
    "photoreal.normalize",
    !("verifiedAt" in normalized) && !("verificationMessage" in normalized) && !("apiKey" in normalized),
    "Non-string verification fields and unknown keys are dropped, not carried."
  );

  const clampCases = [
    [{ steps: -3 }, "steps", 0],
    [{ steps: 25.6 }, "steps", 26],
    [{ cfg: 99 }, "cfg", 20],
    [{ cfg: -1 }, "cfg", 0],
    [{ cfg: 4.5 }, "cfg", 4.5],
    // Below 1 the sampler draws the negative prompt (PHOTOREAL_MIN_CFG): a stored 0 < cfg < 1 is held at 1, never kept.
    [{ cfg: 0.1 }, "cfg", 1],
    [{ cfg: 0.999 }, "cfg", 1],
    [{ cfg: 1e-9 }, "cfg", 1],
    [{ cfg: 1 }, "cfg", 1],
    [{ cfg: 0 }, "cfg", 0],
    [{ width: 100 }, "width", 512],
    [{ width: 300 }, "width", 512],
    [{ width: 520 }, "width", 528],
    [{ width: 5000 }, "width", 2048],
    [{ width: 0 }, "width", 0],
    [{ width: 1024 }, "width", 1024],
    [{ width: "1024" }, "width", 0],
    [{ height: 1240 }, "height", 1248],
    [{ height: Number.POSITIVE_INFINITY }, "height", 0],
    [{ timeoutSeconds: 5000 }, "timeoutSeconds", 900],
    [{ timeoutSeconds: Number.NaN }, "timeoutSeconds", 300],
    [{ timeoutSeconds: 120 }, "timeoutSeconds", 120],
    [{ seed: -1 }, "seed", undefined]
  ];
  for (const [candidate, field, expected] of clampCases) {
    const value = normalizePhotorealSettings(candidate)[field];
    assertCheck(
      "photoreal.normalize",
      Object.is(value, expected),
      `${field} ${formatValue(candidate[field])} → ${formatValue(expected)} (got ${formatValue(value)}).`
    );
  }

  // The settings field shows the stored size as the size in effect, so it must be the size a cut renders at:
  // normalizing must land exactly where resolvePhotorealOutputSize does, for every input, at both bounds.
  const { resolvePhotorealOutputSize } = workflows;
  const mismatches = [];
  for (const side of [1, 100, 255, 256, 300, 496, 504, 511, 512, 513, 520, 1000, 1023, 2040, 2047, 2048, 2049, 5000]) {
    const stored = normalizePhotorealSettings({ width: side, height: side });
    const rendered = resolvePhotorealOutputSize({ photoreal: stored, imageProfile: { width: 832, height: 1216 } });
    if (stored.width !== rendered.width || stored.height !== rendered.height || normalizePhotorealSettings(stored).width !== stored.width) {
      mismatches.push(`${side}: stored ${stored.width}x${stored.height}, rendered ${rendered.width}x${rendered.height}`);
    }
  }
  assertCheck(
    "photoreal.normalize",
    mismatches.length === 0,
    `A stored photoreal width/height is exactly the size the cut renders at, and stays put when normalized again (${mismatches.join("; ") || "1..5000 checked"}).`
  );

  const typing = normalizePhotorealSettings({
    ...EXPECTED_PHOTOREAL_DEFAULTS,
    baseUrl: "",
    stylePrompt: "soft window light, ",
    negativePrompt: " watermark",
    customWorkflow: "{\n  \"3\": "
  });
  assertCheck(
    "photoreal.normalize",
    typing.baseUrl === "" && typing.stylePrompt === "soft window light, " && typing.negativePrompt === " watermark" && typing.customWorkflow === "{\n  \"3\": ",
    "Text fields stay verbatim (an emptied URL stays empty, trailing spaces survive), so normalizing on every write never fights typing."
  );

  const valid = {
    ...EXPECTED_PHOTOREAL_DEFAULTS,
    enabled: true,
    requestMode: "mock",
    baseUrl: "http://127.0.0.1:8189",
    proxyUrl: "http://127.0.0.1:9000/photoreal/comfyui",
    workflowPreset: "z_image_turbo",
    referenceWorkflowPreset: "qwen_image_edit_2511",
    modelFiles: {
      qwen_image_2_1_turbo: { diffusionModel: "q.safetensors", textEncoder: "t.safetensors", vae: "v.safetensors" },
      custom: { vae: "" }
    },
    steps: 12,
    cfg: 1.5,
    width: 832,
    height: 1248,
    seedFixed: true,
    seed: 4242,
    stylePreset: "custom",
    stylePrompt: "editorial",
    appearancePrompt: "glossy lips",
    negativePrompt: "blur",
    faceReferenceEnabled: false,
    vramHandoff: "off",
    customWorkflow: "{}",
    timeoutSeconds: 600,
    registrationStatus: "registered",
    verifiedAt: "2026-10-14T00:00:00.000Z",
    verificationMessage: "ComfyUI 연결됨"
  };
  assertCheck("photoreal.normalize", isDeepStrictEqual(normalizePhotorealSettings(valid), valid), "Valid settings pass through unchanged.");
  assertCheck(
    "photoreal.normalize",
    isDeepStrictEqual(normalizePhotorealSettings(normalized), normalized) && isDeepStrictEqual(normalizePhotorealSettings(normalizePhotorealSettings(valid)), valid),
    "Normalization is idempotent (it runs on every state write)."
  );

  const hydrated = seed.hydrateState({ ...cloneJson(seed.seedState), photoreal: malformed });
  assertCheck(
    "photoreal.normalize",
    isDeepStrictEqual(hydrated.photoreal, normalized),
    "hydrateState applies the same normalization to state.photoreal."
  );

  // A state.json written before the CFG floor (or edited by hand) with a CFG below 1 comes back at 1 — on load, on
  // every later write, and in what the builder would sample at for a preset that takes the setting.
  const belowOne = seed.hydrateState({ ...cloneJson(seed.seedState), photoreal: { ...EXPECTED_PHOTOREAL_DEFAULTS, workflowPreset: "qwen_image_2_1", cfg: 0.1 } });
  assertCheck(
    "photoreal.normalize",
    belowOne.photoreal.cfg === 1 &&
      seed.hydrateState(belowOne).photoreal.cfg === 1 &&
      workflows.resolvePhotorealCfg(belowOne.photoreal, "qwen_image_2_1") === 1 &&
      workflows.PHOTOREAL_MIN_CFG === 1,
    `A hydrated CFG of 0.1 is stored as 1 and resolves to 1 (got ${belowOne.photoreal.cfg}).`
  );
  const unclamped = [-5, -0.5, 0, 0.01, 0.5, 0.99, 1, 1.5, 7, 20, 25, Number.NaN, Number.POSITIVE_INFINITY].filter((cfg) => {
    const stored = normalizePhotorealSettings({ cfg }).cfg;
    return stored !== 0 && stored < 1;
  });
  assertCheck("photoreal.normalize", unclamped.length === 0, `No stored CFG is ever between 0 and 1 (0 = the preset's own) — ${unclamped.join(", ") || "13 values checked"}.`);

  const shareable = toShareablePhotorealSettings(valid);
  const { verifiedAt: _verifiedAt, verificationMessage: _verificationMessage, ...unverified } = valid;
  assertCheck(
    "photoreal.normalize",
    isDeepStrictEqual(shareable, { ...unverified, registrationStatus: "idle" }),
    "toShareablePhotorealSettings resets the connection check and keeps every other field."
  );
  assertCheck(
    "photoreal.normalize",
    !("verifiedAt" in shareable) && !("verificationMessage" in shareable) && shareable.modelFiles !== valid.modelFiles,
    "The verification fields are dropped (not set to undefined) and modelFiles is copied, not shared."
  );
}

function evaluateBackendRoutingHydration({ seed, routing }) {
  const hydrateRouting = (backendRouting) => {
    const candidate = cloneJson(seed.seedState);
    candidate.imageProfile.backendRouting = backendRouting;
    return seed.hydrateState(candidate).imageProfile.backendRouting;
  };

  const mixed = hydrateRouting({ mode: "mixed", photorealCategories: ["portrait", "explicit", "selfie", "scenery", "object"] });
  assertCheck("routing.hydrate", mixed?.mode === "mixed", "A mixed routing keeps its mode through hydration.");
  assertCheck(
    "routing.hydrate",
    ["portrait", "scenery", "object"].every((category) => mixed?.photorealCategories?.includes(category)),
    "A mixed routing keeps its known categories."
  );
  assertCheck(
    "routing.hydrate",
    Array.isArray(mixed?.photorealCategories) && !mixed.photorealCategories.includes("explicit") && !mixed.photorealCategories.includes("selfie"),
    "Hydration strips `explicit` and unknown categories from photorealCategories."
  );

  const unknownMode = hydrateRouting({ mode: "everything", photorealCategories: ["portrait"] });
  assertCheck("routing.hydrate", unknownMode?.mode === "novelai", "An unknown routing mode falls back to NovelAI-only.");

  const notAnObject = hydrateRouting("photoreal");
  assertCheck(
    "routing.hydrate",
    notAnObject?.mode === "novelai" && isDeepStrictEqual(notAnObject.photorealCategories, [...routing.DEFAULT_PHOTOREAL_CATEGORIES]),
    "A non-object routing resolves to the default routing."
  );

  const photoreal = hydrateRouting({ mode: "photoreal", photorealCategories: ["group"] });
  assertCheck("routing.hydrate", photoreal?.mode === "photoreal", "A photoreal routing keeps its mode through hydration.");

  const candidate = cloneJson(seed.seedState);
  candidate.imageProfile.backendRouting = { mode: "mixed", photorealCategories: ["portrait", "explicit", "romance"] };
  const once = seed.hydrateState(candidate);
  assertCheck(
    "routing.hydrate",
    isDeepStrictEqual(seed.hydrateState(once).imageProfile.backendRouting, once.imageProfile.backendRouting),
    "Routing hydration is idempotent."
  );
}

function evaluateVisualProfileNormalization({ seed }) {
  const state = cloneJson(seed.seedState);
  const base = state.visualProfiles[0];
  const novelAiImage = createProfileImage("novelai", "asset_nai_a", "sim_photoreal_eval/asset_nai_a.png", {
    backend: "photoreal",
    referenceObjectKey: "sim_photoreal_eval/asset_nai_a_reference.png",
    mimeType: "image/png",
    width: 832,
    height: 1216,
    negativePrompt: "lowres",
    seed: 1234,
    modelPreset: "NAID4.5F",
    dataUrl: INLINE_BYTES,
    base64: INLINE_BYTES.slice(22),
    imageBase64: INLINE_BYTES.slice(22)
  });
  state.visualProfiles = [
    {
      ...base,
      id: "visual_a",
      characterId: "char_a",
      photorealPrompt: "a Korean woman in her late twenties with shoulder-length black hair, ",
      photorealNegativePrompt: 42,
      profileImages: {
        novelai: novelAiImage,
        photoreal: { assetId: "asset_pr_a", prompt: "missing its object key" },
        midjourney: createProfileImage("photoreal", "asset_mj_a", "sim_photoreal_eval/asset_mj_a.png")
      }
    },
    {
      ...base,
      id: "visual_b",
      characterId: "char_b",
      photorealPrompt: "",
      profileImages: {
        photoreal: createProfileImage("photoreal", "asset_pr_d", "sim_photoreal_eval/asset_pr_d.webp", {
          workflowPreset: "qwen_image_2_1_turbo",
          modelPreset: "NAID9",
          mimeType: "text/html",
          width: -5,
          seed: "7",
          prompt: 12,
          selectedAt: 99
        }),
        novelai: "not an entry"
      }
    },
    { ...base, id: "visual_c", characterId: "char_c", profileImages: ["photoreal"] }
  ];
  const hydrated = seed.hydrateState(state);
  const profileA = hydrated.visualProfiles.find((profile) => profile.id === "visual_a");
  const profileB = hydrated.visualProfiles.find((profile) => profile.id === "visual_b");
  const profileC = hydrated.visualProfiles.find((profile) => profile.id === "visual_c");

  assertCheck(
    "visual-profiles",
    profileA?.photorealPrompt === "a Korean woman in her late twenties with shoulder-length black hair, " && profileB?.photorealPrompt === "",
    "photorealPrompt strings are kept verbatim, an empty one included."
  );
  assertCheck("visual-profiles", Boolean(profileA) && !("photorealNegativePrompt" in profileA), "A non-string photorealNegativePrompt is dropped.");
  assertCheck(
    "visual-profiles",
    isDeepStrictEqual(Object.keys(profileA?.profileImages ?? {}), ["novelai"]),
    "Only well-formed entries under known backends survive (missing objectKey and unknown backend dropped)."
  );
  assertCheck(
    "visual-profiles",
    isDeepStrictEqual(profileA?.profileImages?.novelai, {
      backend: "novelai",
      assetId: "asset_nai_a",
      objectKey: "sim_photoreal_eval/asset_nai_a.png",
      mimeType: "image/png",
      width: 832,
      height: 1216,
      referenceObjectKey: "sim_photoreal_eval/asset_nai_a_reference.png",
      prompt: "1girl, black hair",
      negativePrompt: "lowres",
      seed: 1234,
      modelPreset: "NAID4.5F",
      selectedAt: "2026-10-14T00:00:00.000Z"
    }),
    "A good entry keeps every known field, its backend rewritten to the key, and loses dataUrl/base64 riders."
  );
  assertCheck(
    "visual-profiles",
    isDeepStrictEqual(profileB?.profileImages, {
      photoreal: {
        backend: "photoreal",
        assetId: "asset_pr_d",
        objectKey: "sim_photoreal_eval/asset_pr_d.webp",
        prompt: "",
        workflowPreset: "qwen_image_2_1_turbo",
        selectedAt: ""
      }
    }),
    "Optional fields with the wrong type are dropped; required text defaults to an empty string."
  );
  assertCheck("visual-profiles", Boolean(profileC) && !("profileImages" in profileC), "A profileImages array is dropped entirely.");
  assertCheck(
    "visual-profiles",
    !JSON.stringify(hydrated.visualProfiles).includes("base64"),
    "No image bytes remain anywhere in the hydrated visual profiles."
  );

  const rejectedKeys = [
    ["a data URL", INLINE_BYTES],
    ["a remote URL", "https://example.com/face.png"],
    ["a traversal", "sim_photoreal_eval/../../secrets/face.png"],
    ["an absolute path", "/etc/face.png"],
    ["a drive path", "C:\\Users\\face.png"],
    ["an empty key", ""],
    ["a non-string", 7]
  ];
  for (const [label, objectKey] of rejectedKeys) {
    const keyed = seed.hydrateState({
      ...cloneJson(seed.seedState),
      visualProfiles: [{ ...base, profileImages: { photoreal: createProfileImage("photoreal", "asset_pr_key", objectKey) } }]
    });
    assertCheck("visual-profiles", !("profileImages" in keyed.visualProfiles[0]), `A profile image whose objectKey is ${label} is dropped.`);
  }
  for (const [label, assetId] of [["a path", "sim/asset"], ["a data URL", INLINE_BYTES], ["empty", ""]]) {
    const keyed = seed.hydrateState({
      ...cloneJson(seed.seedState),
      visualProfiles: [{ ...base, profileImages: { photoreal: createProfileImage("photoreal", assetId, "sim_photoreal_eval/asset_ok.png") } }]
    });
    assertCheck("visual-profiles", !("profileImages" in keyed.visualProfiles[0]), `A profile image whose assetId is ${label} is dropped.`);
  }
  const draftKeyed = seed.hydrateState({
    ...cloneJson(seed.seedState),
    visualProfiles: [{ ...base, profileImages: { photoreal: createProfileImage("photoreal", "asset_m1_2", "draft_simulation/asset_m1_2.png") } }]
  });
  assertCheck(
    "visual-profiles",
    draftKeyed.visualProfiles[0].profileImages?.photoreal?.objectKey === "draft_simulation/asset_m1_2.png",
    "A key under a draft's provisional simulation id is a valid store path and is kept."
  );

  assertCheck(
    "visual-profiles",
    isDeepStrictEqual(seed.hydrateState(hydrated).visualProfiles, hydrated.visualProfiles),
    "Visual profile normalization is idempotent."
  );
}

function evaluateCreateStateFromDraft({ seed }) {
  const minaImages = {
    photoreal: createProfileImage("photoreal", "asset_pr_mina", "sim_draft_photoreal/asset_pr_mina.png", {
      workflowPreset: "qwen_image_2_1_turbo"
    }),
    novelai: createProfileImage("novelai", "asset_nai_mina", "sim_draft_photoreal/asset_nai_mina.png", {
      referenceObjectKey: "sim_draft_photoreal/asset_nai_mina_reference.png",
      modelPreset: "NAID4.5F"
    })
  };
  const yunaImages = {
    photoreal: createProfileImage("photoreal", "asset_pr_yuna", "sim_draft_photoreal/asset_pr_yuna.png", { dataUrl: INLINE_BYTES })
  };
  const draft = createDraft(seed.seedState, [
    createDraftCharacter("draft_char_mina", "Mina", {
      photorealPrompt: "a Korean woman in her late twenties with shoulder-length black hair",
      photorealNegativePrompt: "heavy makeup",
      profileImages: minaImages
    }),
    createDraftCharacter("draft_char_yuna", "Yuna", {
      photorealPrompt: "a woman in her early thirties with a short bob",
      profileImages: yunaImages
    })
  ]);
  const state = seed.createStateFromDraft(draft);
  const [mina, yuna] = state.characters;
  const minaProfile = state.visualProfiles.find((profile) => profile.characterId === mina?.id);
  const yunaProfile = state.visualProfiles.find((profile) => profile.characterId === yuna?.id);

  assertCheck(
    "draft",
    state.characters.length === 2 && mina?.name === "Mina" && yuna?.name === "Yuna" && mina.id !== yuna.id,
    "Both draft characters become characters of the new simulation."
  );
  assertCheck(
    "draft",
    state.characters.every((character) => !character.id.startsWith("draft_")) &&
      state.visualProfiles.every((profile) => !profile.characterId.startsWith("draft_")) &&
      state.modules.every((module) => !module.characterId?.startsWith("draft_")),
    "No character, visual profile or module keeps a draft character id."
  );
  assertCheck("draft", state.visualProfiles.length === 2, "One visual profile per draft character.");
  assertCheck(
    "draft",
    minaProfile?.photorealPrompt === "a Korean woman in her late twenties with shoulder-length black hair" &&
      minaProfile?.photorealNegativePrompt === "heavy makeup",
    "Mina's photoreal appearance lands on the profile of her NEW id."
  );
  assertCheck(
    "draft",
    yunaProfile?.photorealPrompt === "a woman in her early thirties with a short bob" && !("photorealNegativePrompt" in yunaProfile),
    "Yuna's photoreal appearance lands on her own new profile, without borrowing Mina's negative."
  );
  assertCheck(
    "draft",
    minaProfile?.profileImages?.photoreal?.objectKey === "sim_draft_photoreal/asset_pr_mina.png" &&
      minaProfile?.profileImages?.novelai?.objectKey === "sim_draft_photoreal/asset_nai_mina.png" &&
      minaProfile?.profileImages?.novelai?.referenceObjectKey === "sim_draft_photoreal/asset_nai_mina_reference.png",
    "Mina's profile images keep their object keys exactly (both backends, NovelAI reference canvas included)."
  );
  assertCheck(
    "draft",
    yunaProfile?.profileImages?.photoreal?.objectKey === "sim_draft_photoreal/asset_pr_yuna.png" &&
      !("novelai" in (yunaProfile?.profileImages ?? {})) &&
      !("dataUrl" in (yunaProfile?.profileImages?.photoreal ?? {})),
    "Yuna's profile image keeps its object key and loses the data URL it was drafted with."
  );
  assertCheck(
    "draft",
    !JSON.stringify(state.visualProfiles).includes("base64"),
    "No image bytes are carried from the draft into the new simulation."
  );
  assertCheck(
    "draft",
    state.photoreal.enabled === true &&
      state.photoreal.stylePreset === "film_portrait" &&
      state.photoreal.width === 1008 &&
      state.photoreal.height === 1248 &&
      state.photoreal.registrationStatus === "registered",
    "The draft's photoreal settings are copied (normalized: width 1000 → 1008)."
  );
  assertCheck(
    "draft",
    state.imageProfile.backendRouting?.mode === "mixed" && !state.imageProfile.backendRouting.photorealCategories.includes("explicit"),
    "The draft's routing is kept and `explicit` cannot be routed photoreal."
  );

  const { photoreal: _photoreal, ...legacyDraft } = draft;
  assertCheck(
    "draft",
    isDeepStrictEqual(seed.createStateFromDraft(legacyDraft).photoreal, EXPECTED_PHOTOREAL_DEFAULTS),
    "A draft from a builder that predates photoreal creates a simulation with the default photoreal settings."
  );
}

function evaluateJobAndAssetBackendPersistence({ seed, progressRuns, simulationRuns }) {
  const routing = {
    backend: "photoreal",
    category: "portrait",
    mode: "mixed",
    reason: "혼합 모드: '인물 단독 · 얼굴/상반신' 장면 → 실사",
    signals: ["subjects:1", 5, "frame:upper_body"],
    fallback: false
  };
  const candidate = cloneJson(seed.seedState);
  candidate.imageJobs = [
    createJob(seed.seedState, {
      id: "job_photoreal",
      assetIds: ["asset_photoreal"],
      providerPayload: { provider: "photoreal", presetId: "qwen_image_2_1_turbo", workflow: { 1: { class_type: "UNETLoader", inputs: {} } } },
      backend: "photoreal",
      routing
    }),
    createJob(seed.seedState, {
      id: "job_bad",
      backend: "dalle",
      routing: { backend: "photoreal", category: "selfie", mode: "mixed", reason: "x", signals: [] }
    }),
    createJob(seed.seedState, { id: "job_locked", backend: "novelai", routing: { backend: "novelai", category: "explicit", mode: "photoreal", locked: true } })
  ];
  candidate.imageAssets = [
    createAsset(seed.seedState, {
      id: "asset_photoreal",
      jobId: "job_photoreal",
      backend: "photoreal",
      sceneCategory: "portrait",
      providerMetadata: { provider: "photoreal", workflowSummary: { nodeCount: 11 } }
    }),
    createAsset(seed.seedState, { id: "asset_bad", backend: 3, sceneCategory: "nsfw" })
  ];
  const state = seed.hydrateState(candidate);
  const job = state.imageJobs.find((item) => item.id === "job_photoreal");
  const badJob = state.imageJobs.find((item) => item.id === "job_bad");
  const lockedJob = state.imageJobs.find((item) => item.id === "job_locked");
  const asset = state.imageAssets.find((item) => item.id === "asset_photoreal");
  const badAsset = state.imageAssets.find((item) => item.id === "asset_bad");
  const expectedRouting = { ...routing, signals: ["subjects:1", "frame:upper_body"] };

  assertCheck("jobs-assets", job?.backend === "photoreal", "hydrateState keeps job.backend.");
  assertCheck("jobs-assets", isDeepStrictEqual(job?.routing, expectedRouting), "hydrateState keeps a well-formed job.routing (non-string signals filtered).");
  assertCheck(
    "jobs-assets",
    Boolean(badJob) && !("backend" in badJob) && !("routing" in badJob),
    "An unknown backend and a routing with an unknown category are dropped (the job reads as NovelAI)."
  );
  assertCheck(
    "jobs-assets",
    isDeepStrictEqual(lockedJob?.routing, { backend: "novelai", category: "explicit", mode: "photoreal", reason: "", locked: true, signals: [] }),
    "A locked decision keeps `locked`; missing reason/signals are coerced instead of losing the decision."
  );
  assertCheck("jobs-assets", asset?.backend === "photoreal" && asset?.sceneCategory === "portrait", "hydrateState keeps asset.backend and asset.sceneCategory.");
  assertCheck(
    "jobs-assets",
    Boolean(badAsset) && !("backend" in badAsset) && !("sceneCategory" in badAsset),
    "An unknown asset backend / scene category is dropped."
  );

  const activeRun = state.progressRuns.find((run) => run.id === state.activeProgressRunId);
  assertCheck(
    "jobs-assets",
    activeRun?.imageJobs.find((item) => item.id === "job_photoreal")?.backend === "photoreal" &&
      activeRun?.imageAssets.find((item) => item.id === "asset_photoreal")?.sceneCategory === "portrait",
    "The active progress run snapshot carries the backend fields."
  );

  const slim = progressRuns.slimProgressRunForStorage(activeRun);
  const slimJob = slim.imageJobs.find((item) => item.id === "job_photoreal");
  const slimAsset = slim.imageAssets.find((item) => item.id === "asset_photoreal");
  assertCheck(
    "jobs-assets",
    isDeepStrictEqual(slimJob?.providerPayload, {}) && slimJob?.backend === "photoreal" && isDeepStrictEqual(slimJob?.routing, expectedRouting),
    "slimProgressRunForStorage drops providerPayload but keeps job.backend and job.routing."
  );
  assertCheck(
    "jobs-assets",
    Boolean(slimAsset) && !("providerMetadata" in slimAsset) && slimAsset.backend === "photoreal" && slimAsset.sceneCategory === "portrait",
    "slimProgressRunForStorage drops providerMetadata but keeps asset.backend and asset.sceneCategory."
  );

  // Store the slimmed run as a past run, reload, then switch back to it.
  const stored = { ...cloneJson(slim), id: "run_stored_photoreal", title: "진행 1" };
  const reloaded = seed.hydrateState({
    ...cloneJson(state),
    activeProgressRunId: "run_active_after_reload",
    progressRuns: [stored],
    imageJobs: [],
    imageAssets: []
  });
  const reloadedRun = reloaded.progressRuns.find((run) => run.id === "run_stored_photoreal");
  assertCheck(
    "jobs-assets",
    reloadedRun?.imageJobs.find((item) => item.id === "job_photoreal")?.backend === "photoreal" &&
      isDeepStrictEqual(reloadedRun?.imageJobs.find((item) => item.id === "job_photoreal")?.routing, expectedRouting) &&
      reloadedRun?.imageAssets.find((item) => item.id === "asset_photoreal")?.backend === "photoreal",
    "A stored run keeps the backend fields through a JSON round trip and hydration."
  );
  const activated = simulationRuns.activateSimulationProgressRun(reloaded, "run_stored_photoreal");
  assertCheck(
    "jobs-assets",
    activated.activeProgressRunId === "run_stored_photoreal" &&
      activated.imageJobs.find((item) => item.id === "job_photoreal")?.backend === "photoreal" &&
      isDeepStrictEqual(activated.imageJobs.find((item) => item.id === "job_photoreal")?.routing, expectedRouting) &&
      activated.imageAssets.find((item) => item.id === "asset_photoreal")?.sceneCategory === "portrait",
    "Switching back to the stored run restores photoreal jobs and assets with their backend."
  );
}

async function evaluatePersistenceBoundary({ seed, dynamicChatApi }) {
  const defaultBase = dynamicChatApi.DEFAULT_DYNAMICCHAT_API_BASE_URL;
  assertCheck("api-boundary", typeof globalThis.window === "undefined", "The eval runs in plain Node (no window).");
  assertCheck(
    "api-boundary",
    dynamicChatApi.getPhotorealRelayBaseUrl() === "http://127.0.0.1:8788/photoreal/comfyui",
    "getPhotorealRelayBaseUrl() works in Node and points at the default API server's relay."
  );
  assertCheck(
    "api-boundary",
    dynamicChatApi.getConfiguredDynamicChatApiBaseUrl() === defaultBase &&
      dynamicChatApi.getNovelAiGenerateProxyUrl() === `${defaultBase}/novelai/generate-image` &&
      typeof dynamicChatApi.createDynamicChatApiClient().saveSimulationState === "function",
    "The other base-URL readers are Node-safe too."
  );

  const capture = await startCaptureServer();
  try {
    const client = dynamicChatApi.createDynamicChatApiClient(capture.baseUrl);
    const state = seed.hydrateState(cloneJson(seed.seedState));
    // Set after hydration on purpose: the boundary guard exists for states that skipped it.
    state.visualProfiles = state.visualProfiles.map((profile, index) =>
      index === 0
        ? {
            ...profile,
            profileImages: {
              photoreal: createProfileImage("photoreal", "asset_pr_boundary", "sim_nocturne/asset_pr_boundary.png", { dataUrl: INLINE_BYTES }),
              novelai: createProfileImage("novelai", "asset_nai_boundary", INLINE_BYTES)
            }
          }
        : profile
    );
    state.llm = { ...state.llm, apiKey: "sk-must-not-leave" };

    await client.saveSimulationState(state);
    await client.createSimulation(state);

    for (const [label, method] of [["saveSimulationState", "PUT"], ["createSimulation", "POST"]]) {
      const request = capture.requests.find((item) => item.method === method);
      assertCheck("api-boundary", Boolean(request), `${label} reached the capture server.`);
      if (!request) {
        continue;
      }

      const body = JSON.parse(request.body);
      const images = body.visualProfiles?.[0]?.profileImages;
      assertCheck("api-boundary", !request.body.includes("base64"), `${label}: no profile image bytes in the request body.`);
      assertCheck(
        "api-boundary",
        images?.photoreal?.objectKey === "sim_nocturne/asset_pr_boundary.png" && !("dataUrl" in images.photoreal),
        `${label}: the profile keeps its object-key reference without the data URL.`
      );
      assertCheck("api-boundary", Boolean(images) && !("novelai" in images), `${label}: an entry whose objectKey is inline bytes is dropped.`);
      assertCheck(
        "api-boundary",
        body.llm?.apiKey === "" && body.visualProfiles?.length === state.visualProfiles.length,
        `${label}: existing secret redaction still applies and no profile is lost.`
      );
    }
  } finally {
    await capture.close();
  }
}

function createDraft(seedState, characters) {
  const updatedAt = "2026-10-14T00:00:00.000Z";
  const createModule = (id, kind, title, extra = {}) => ({
    id,
    simulationId: "draft_simulation",
    kind,
    title,
    body: `${title} 본문`,
    enabled: true,
    priority: 80,
    activationTags: [kind],
    tokenPolicy: kind === "main_prompt" ? "always" : "rag",
    version: 1,
    updatedAt,
    ...extra
  });

  return {
    promptMode: "simulation",
    contentRating: "general",
    title: "실사 프로필 초안",
    description: "프로필 이미지 이관 검사",
    mainPrompt: "플레이어의 선택을 중심으로 진행한다.",
    characterName: characters[0]?.name ?? "Main Character",
    characterRole: "lead",
    characterSummary: "",
    characterRelationship: "",
    characterMood: "",
    worldLore: "",
    startSituationPrompt: "첫 장면.",
    visualPrompt: "",
    negativeVisualPrompt: "",
    realtimeImageEnabled: true,
    characters,
    imageScenePresets: [],
    modules: [
      createModule("draft_main", "main_prompt", "메인 규칙"),
      ...characters.map((character) =>
        createModule(`draft_module_${character.id}`, "character_prompt", `캐릭터: ${character.name}`, {
          parentId: "draft_main",
          characterId: character.id
        })
      )
    ],
    imageProfile: {
      ...cloneJson(seedState.imageProfile),
      id: "draft_image_profile",
      simulationId: "draft_simulation",
      backendRouting: { mode: "mixed", photorealCategories: ["portrait", "full_body", "explicit"] }
    },
    neuralMap: { ...seedState.neuralMap },
    relationshipMap: cloneJson(seedState.relationshipMap),
    llm: { ...seedState.llm },
    novelAi: cloneJson(seedState.novelAi),
    photoreal: {
      ...EXPECTED_PHOTOREAL_DEFAULTS,
      enabled: true,
      stylePreset: "film_portrait",
      width: 1000,
      height: 1248,
      registrationStatus: "registered",
      verifiedAt: "2026-10-14T00:00:00.000Z"
    }
  };
}

function createDraftCharacter(id, name, extra = {}) {
  return {
    id,
    name,
    role: "lead",
    summary: `${name} summary`,
    relationship: "",
    currentMood: "",
    visualPrompt: "1girl, black hair",
    negativeVisualPrompt: "",
    defaultOutfitPrompt: "",
    outfitPrompts: {},
    expressionPrompts: {},
    defaultSafetyLevel: "safe",
    ...extra
  };
}

function createProfileImage(backend, assetId, objectKey, extra = {}) {
  return {
    backend,
    assetId,
    objectKey,
    prompt: "1girl, black hair",
    selectedAt: "2026-10-14T00:00:00.000Z",
    ...extra
  };
}

function createJob(seedState, overrides) {
  return {
    id: "job_eval",
    simulationId: seedState.simulation.id,
    sessionId: seedState.simulation.activeSessionId,
    turnId: "msg_welcome",
    status: "completed",
    reason: "eval",
    prompt: "1girl, black hair",
    negativePrompt: "",
    providerPayload: {},
    assetIds: [],
    contextNodeIds: [],
    createdAt: "2026-10-14T00:00:00.000Z",
    ...overrides
  };
}

function createAsset(seedState, overrides) {
  return {
    id: "asset_eval",
    simulationId: seedState.simulation.id,
    title: "eval asset",
    source: "generated",
    prompt: "1girl, black hair",
    negativePrompt: "",
    safetyLevel: "safe",
    characterIds: [],
    tags: [],
    createdAt: "2026-10-14T00:00:00.000Z",
    palette: ["#f4f0e8", "#d7c7aa", "#5f5046"],
    objectKey: `${seedState.simulation.id}/${overrides.id ?? "asset_eval"}.png`,
    ...overrides
  };
}

function startCaptureServer() {
  return new Promise((resolve, reject) => {
    const requests = [];
    const server = http.createServer((request, response) => {
      let body = "";
      request.setEncoding("utf8");
      request.on("data", (chunk) => {
        body += chunk;
      });
      request.on("end", () => {
        requests.push({ method: request.method, url: request.url, body });
        if (request.method === "PUT") {
          response.writeHead(204);
          response.end();
          return;
        }
        response.writeHead(200, { "content-type": "application/json" });
        response.end(body || "{}");
      });
    });
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({
        baseUrl: `http://127.0.0.1:${port}`,
        requests,
        close: () =>
          new Promise((done) => {
            // fetch keeps connections alive; without this close() waits out the keep-alive timeout.
            server.closeAllConnections();
            server.close(() => done());
          })
      });
    });
  });
}

async function runGroup(scope, evaluate) {
  try {
    await evaluate();
  } catch (error) {
    assertCheck(scope, false, `Group threw before finishing: ${describeError(error)}`);
  }
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

// JSON.stringify prints NaN and Infinity as null, which would make a clamp label lie about its input.
function formatValue(value) {
  return typeof value === "number" && !Number.isFinite(value) ? String(value) : (JSON.stringify(value) ?? String(value));
}

function describeError(error) {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function assertCheck(scope, passed, message) {
  checks.push({
    level: passed ? "pass" : "fail",
    scope,
    message
  });
}
