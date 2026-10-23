import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { zipSync } from "fflate";
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
const originalFetch = globalThis.fetch;
const hadOffscreenCanvas = Object.prototype.hasOwnProperty.call(globalThis, "OffscreenCanvas");
const originalOffscreenCanvas = globalThis.OffscreenCanvas;

// What the stub relay / stub NovelAI hand back as "the rendered image": a 1x1 PNG. Round-tripped through
// Buffer so the fixture is canonical base64 and byte-exact comparisons cannot trip over padding bits.
const PNG_BASE64 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
  "base64"
).toString("base64");
// Stand-in for a letterboxed Precise Reference canvas / a profile image. Long and distinctive, so a copy that
// leaks into a stored payload is impossible to miss.
const REFERENCE_BASE64 = Buffer.from("DYNAMICCHAT-PROFILE-REFERENCE|".repeat(64)).toString("base64");
const REFERENCE_BYTES = Buffer.from(REFERENCE_BASE64, "base64").byteLength;
const PROMPT = "A candid, true-to-life smartphone photograph. The woman from <image1> stands by a cream wardrobe mirror, smiling.";
const NEGATIVE_PROMPT = "anime, cartoon, illustration, watermark";
const SEED_LIMIT = 2 ** 31;

// Spec §6 defaults, used when the seed predates (or lacks) state.photoreal so this eval does not depend on it.
const PHOTOREAL_DEFAULTS = {
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
  stylePreset: "daily_look",
  stylePrompt: "",
  negativePrompt: "",
  faceReferenceEnabled: true,
  vramHandoff: "auto",
  customWorkflow: "",
  timeoutSeconds: 300,
  registrationStatus: "idle"
};

const PROFILE_REFERENCES = [
  // objectKey AND dataUrl: the relay must get only the object key (it reads the file itself).
  { name: "char_1", objectKey: "sim_eval/profile_a.png", characterId: "char_a", dataUrl: `data:image/png;base64,${REFERENCE_BASE64}` },
  // dataUrl only: bytes that were never a picked profile image. Never sent — the reference is dropped.
  { name: "char_2", dataUrl: `data:image/png;base64,${REFERENCE_BASE64}`, characterId: "char_b" }
];

const ENCODED_VIBE = {
  id: "vibe_eval",
  name: "vibe.png",
  image: "data:image/png;base64,QUJD",
  referenceStrength: 0.6,
  informationExtracted: 1,
  encodedVibe: "ENCODEDVIBEDATA==",
  encodedModel: "NAID4.5F",
  encodedInformationExtracted: 1
};

let relay;

try {
  const modules = await loadModules({
    seed: "/src/data/seed.ts",
    comfy: "/src/services/comfyUiClient.ts",
    workflows: "/src/services/comfyWorkflows.ts",
    novelAi: "/src/services/novelAiClient.ts",
    references: "/src/services/imageReferences.ts"
  });
  relay = await startStubRelay();

  await runGroup("photoreal.dry-run", () => evaluatePhotorealDryRun(modules));
  await runGroup("photoreal.workflow-safety", () => evaluatePhotorealWorkflowSafety(modules));
  await runGroup("photoreal.what-is-sent", () => evaluateWhatIsSent(modules, relay));
  await runGroup("photoreal.seed", () => evaluatePhotorealSeedPolicy(modules));
  await runGroup("photoreal.relay", () => evaluatePhotorealRelayRequest(modules, relay));
  await runGroup("photoreal.relay-errors", () => evaluatePhotorealRelayErrors(modules, relay));
  await runGroup("photoreal.abort", () => evaluatePhotorealAbort(modules, relay));
  await runGroup("photoreal.probe", () => evaluateComfyUiProbe(modules, relay));
  await runGroup("photoreal.best-effort", () => evaluateBestEffortCalls(modules, relay));
  await runGroup("nai.precise-reference", () => evaluatePreciseReferencePayload(modules));
  await runGroup("nai.request", () => evaluateNovelAiRequestBody(modules));
  await runGroup("nai.encode-vibe", () => evaluateEncodeVibeRequest(modules));
  await runGroup("references", () => evaluateImageReferences(modules));
} finally {
  globalThis.fetch = originalFetch;
  restoreOffscreenCanvas();
  await relay?.close();
  await vite.close();
}

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}`);
}

const failures = checks.filter((check) => check.level === "fail");
console.log("");
console.log(`Photoreal client checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

// ---------------------------------------------------------------------------------------------------------
// (a) Dry run: disabled and mock never touch the network, and the payload says exactly what would be sent.
// ---------------------------------------------------------------------------------------------------------
async function evaluatePhotorealDryRun(modules) {
  const { seedState } = requireModule(modules, "seed");
  const { generatePhotorealImages } = requireModule(modules, "comfy");
  const { buildComfyWorkflow, resolvePhotorealModelFiles, resolvePhotorealOutputSize, resolvePhotorealPresetForCut, PHOTOREAL_WORKFLOW_PRESETS } =
    requireModule(modules, "workflows");
  const fetchProbe = installForbiddenFetch();

  const disabledState = createState(seedState, { photoreal: { enabled: false } });
  const disabled = await generatePhotorealImages({
    state: disabledState,
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    cue: createCue(),
    presetId: "qwen_image_2_1_turbo",
    references: PROFILE_REFERENCES,
    jobId: "job_eval_dry"
  });
  const payload = disabled.payload;
  const workflow = payload.workflow ?? {};
  const preset = PHOTOREAL_WORKFLOW_PRESETS.qwen_image_2_1_turbo;
  const expected = buildComfyWorkflow("qwen_image_2_1_turbo", {
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    width: payload.width,
    height: payload.height,
    seed: payload.seed,
    steps: preset.defaults.steps,
    cfg: preset.defaults.cfg,
    files: resolvePhotorealModelFiles(disabledState.photoreal, "qwen_image_2_1_turbo"),
    referenceNames: ["char_1"],
    filenamePrefix: "DynamicChat/job_eval_dry"
  });
  const loadImageInputs = Object.values(workflow)
    .filter((node) => node?.class_type === "LoadImage")
    .map((node) => node.inputs?.image);
  const outputSize = resolvePhotorealOutputSize(disabledState);

  assertCheck("photoreal.dry-run", Array.isArray(disabled.dataUrls) && disabled.dataUrls.length === 0, "Disabled photoreal returns no images (dry run).");
  assertCheck(
    "photoreal.dry-run",
    payload.provider === "photoreal" && payload.promptFormat === "natural-language" && payload.presetId === "qwen_image_2_1_turbo",
    "Dry-run payload names provider 'photoreal', natural-language prompting and the preset."
  );
  assertCheck(
    "photoreal.dry-run",
    JSON.stringify(workflow) === JSON.stringify(expected.workflow),
    "Dry-run payload carries exactly the graph buildComfyWorkflow builds from the resolved settings."
  );
  // Changed deliberately (round 3): a reference with bytes but no picked-profile object key used to be sent as a
  // dataUrl. It is now dropped — the app only ever sends object keys of picked profile images.
  assertCheck(
    "photoreal.dry-run",
    JSON.stringify(workflow).includes("smiling") && loadImageInputs.includes("%ref:char_1%") && !loadImageInputs.includes("%ref:char_2%") && loadImageInputs.length === 1,
    "Built graph holds the prompt and wires the picked profile image as a %ref:<name>% LoadImage; a dataUrl-only reference is not wired."
  );
  assertCheck(
    "photoreal.dry-run",
    JSON.stringify(payload.references) === JSON.stringify([{ name: "char_1", objectKey: "sim_eval/profile_a.png", characterId: "char_a" }]),
    "Payload references are {name, objectKey, characterId} only, and only for object keys."
  );
  assertCheck(
    "photoreal.dry-run",
    !JSON.stringify(payload).includes(REFERENCE_BASE64) && !JSON.stringify(payload).includes("data:image"),
    "Payload never carries reference bytes (no data URL, no base64)."
  );
  assertCheck(
    "photoreal.dry-run",
    payload.width === outputSize.width && payload.height === outputSize.height,
    `Payload size is resolvePhotorealOutputSize (${outputSize.width}x${outputSize.height}).`
  );
  assertCheck("photoreal.dry-run", isSeed(payload.seed), `Payload exposes the seed used (${payload.seed}), a 31-bit integer.`);
  assertCheck(
    "photoreal.dry-run",
    Array.isArray(payload.outputNodeIds) && payload.outputNodeIds.length > 0 && payload.outputNodeIds.every((id) => workflow[id]),
    "Payload outputNodeIds point at nodes in the graph."
  );
  assertCheck(
    "photoreal.dry-run",
    payload.workflowSummary?.nodeCount === Object.keys(workflow).length && payload.workflowSummary?.classTypes?.includes("PreviewImage"),
    "Payload workflowSummary counts the nodes and lists PreviewImage (temp output) among the class types."
  );

  const mock = await generatePhotorealImages({
    state: createState(seedState, { photoreal: { enabled: true, requestMode: "mock" } }),
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    cue: createCue(),
    presetId: "qwen_image_2_1_turbo",
    references: PROFILE_REFERENCES,
    jobId: "job_eval_mock"
  });
  assertCheck(
    "photoreal.dry-run",
    mock.dataUrls.length === 0 && mock.payload.provider === "photoreal" && isRecord(mock.payload.workflow) && isSeed(mock.payload.seed),
    "Mock mode is a dry run too: built graph and seed, no images."
  );

  const zImage = await generatePhotorealImages({
    state: disabledState,
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    cue: createCue(),
    presetId: "z_image_turbo",
    references: PROFILE_REFERENCES,
    jobId: "job_eval_zimage"
  });
  assertCheck(
    "photoreal.dry-run",
    zImage.payload.presetId === "z_image_turbo" &&
      Array.isArray(zImage.payload.references) &&
      zImage.payload.references.length === 0 &&
      !JSON.stringify(zImage.payload.workflow).includes("%ref:"),
    "A preset without reference inputs (Z-Image) drops the references instead of shipping unused faces."
  );

  const unknownPreset = await generatePhotorealImages({
    state: disabledState,
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    cue: createCue(),
    presetId: "not_a_preset",
    references: PROFILE_REFERENCES,
    jobId: "job_eval_unknown"
  });
  assertCheck(
    "photoreal.dry-run",
    unknownPreset.payload.presetId === resolvePhotorealPresetForCut(disabledState.photoreal, true),
    "An unknown preset id resolves through resolvePhotorealPresetForCut instead of crashing."
  );

  const customError = await captureError(() =>
    generatePhotorealImages({
      state: createState(seedState, { photoreal: { enabled: false, workflowPreset: "custom", customWorkflow: "{ this is not json" } }),
      prompt: PROMPT,
      negativePrompt: NEGATIVE_PROMPT,
      cue: createCue(),
      presetId: "custom",
      references: [],
      jobId: "job_eval_custom"
    })
  );
  assertCheck(
    "photoreal.dry-run",
    customError instanceof Error && customError.message.startsWith("실사 이미지 생성 실패"),
    `An unusable custom workflow fails with the Korean photoreal prefix (${customError?.message?.slice(0, 80) ?? "no error"}).`
  );

  assertCheck("photoreal.dry-run", fetchProbe.calls === 0, "Disabled, mock and failed builds never call fetch.");
}

// ---------------------------------------------------------------------------------------------------------
// (a2) What a graph may carry, whoever built it (red-team channels A-D): Turbo presets sample at CFG 1 whatever the
//      setting says, every negative is cleaned as it enters the graph, an image is only ever a %ref: reference the
//      relay uploads, and a custom template keeps the text the floor never reads off the picture's side.
// ---------------------------------------------------------------------------------------------------------
async function evaluatePhotorealWorkflowSafety(modules) {
  const scope = "photoreal.workflow-safety";
  const { seedState } = requireModule(modules, "seed");
  const { generatePhotorealImages } = requireModule(modules, "comfy");
  const workflows = requireModule(modules, "workflows");
  const fetchProbe = installForbiddenFetch();
  const dryRun = (photoreal, presetId, negativePrompt = NEGATIVE_PROMPT) =>
    generatePhotorealImages({
      state: createState(seedState, { photoreal: { enabled: false, ...photoreal } }),
      prompt: PROMPT,
      negativePrompt,
      cue: createCue(),
      presetId,
      references: [],
      jobId: "job_eval_safety"
    });
  const nodesOf = (workflow, classType) => Object.values(workflow ?? {}).filter((node) => node?.class_type === classType);

  // D: a distilled preset at CFG > 1 burns the image, and at exactly 1 the negative is never evaluated.
  const qwenTurbo = (await dryRun({ cfg: 4 }, "qwen_image_2_1_turbo")).payload;
  const zTurbo = (await dryRun({ cfg: 4 }, "z_image_turbo")).payload;
  const qwenBase = (await dryRun({ cfg: 4 }, "qwen_image_2_1")).payload;
  assertCheck(
    scope,
    nodesOf(qwenTurbo.workflow, "CFGGuider")[0]?.inputs?.cfg === 1 && qwenTurbo.cfg === 1,
    "Qwen-Image 2.1 Turbo samples at CFG 1 with the setting at 4, and the payload records the 1 that was sent."
  );
  assertCheck(scope, nodesOf(zTurbo.workflow, "KSampler")[0]?.inputs?.cfg === 1 && zTurbo.cfg === 1, "Z-Image Turbo samples at CFG 1 with the setting at 4.");
  assertCheck(scope, nodesOf(qwenBase.workflow, "KSampler")[0]?.inputs?.cfg === 4 && qwenBase.cfg === 4, "A preset that is not distilled (Qwen-Image 2.1) keeps the user's CFG 4.");
  assertCheck(
    scope,
    workflows.resolvePhotorealCfg({ cfg: 6 }, "qwen_image_2_1_turbo") === 1 &&
      workflows.resolvePhotorealCfg({ cfg: 6 }, "z_image_turbo") === 1 &&
      workflows.resolvePhotorealCfg({ cfg: 6 }, "qwen_image_edit_2511") === 6 &&
      workflows.resolvePhotorealCfg({ cfg: 0 }, "qwen_image_edit_2511") === 4 &&
      workflows.resolvePhotorealCfg({ cfg: 6 }, "custom") === 6,
    "resolvePhotorealCfg: fixed 1 for the Turbo presets, the setting (0 = the preset's own) for the rest, custom included."
  );
  const handed = workflows.buildComfyWorkflow("qwen_image_2_1_turbo", {
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    width: 1024,
    height: 1024,
    seed: 1,
    steps: 8,
    cfg: 7.5,
    files: workflows.resolvePhotorealModelFiles({ modelFiles: {} }, "qwen_image_2_1_turbo"),
    referenceNames: [],
    filenamePrefix: "DynamicChat/job_eval_safety"
  });
  assertCheck(scope, nodesOf(handed.workflow, "CFGGuider")[0]?.inputs?.cfg === 1, "The builder pins the Turbo CFG whatever its caller hands it (7.5 in, 1 out).");

  // D: the job's negative is cleaned as it enters the graph — a queued job keeps the negative it was planned with.
  const steering = "anime, clothes, shirt, pleated skirt, watermark";
  const cleaned = (await dryRun({ cfg: 4 }, "qwen_image_2_1", steering)).payload;
  const sentNegative = String(nodesOf(cleaned.workflow, "TextEncodeQwenImage21")[0]?.inputs?.negative_prompt ?? "");
  assertCheck(
    scope,
    /\banime\b/u.test(sentNegative) && /\bwatermark\b/u.test(sentNegative) && !/\b(?:clothes|shirt|skirt)\b/iu.test(sentNegative),
    `A negative that names body-covering clothing reaches the graph without it; the unwanted things stay (${JSON.stringify(sentNegative)}).`
  );

  const template = (overrides = {}) =>
    JSON.stringify({
      "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "%model%" } },
      "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "%prompt%" } },
      "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "%negative_prompt%" } },
      "4": { class_type: "EmptyLatentImage", inputs: { width: "%width%", height: "%height%", batch_size: 1 } },
      "5": {
        class_type: "KSampler",
        inputs: { model: ["1", 0], seed: "%seed%", steps: "%steps%", cfg: "%cfg%", sampler_name: "%sampler%", scheduler: "%scheduler%", positive: ["2", 0], negative: ["3", 0], latent_image: ["4", 0], denoise: 1 }
      },
      "6": { class_type: "VAEDecode", inputs: { samples: ["5", 0], vae: ["1", 2] } },
      "7": { class_type: "SaveImage", inputs: { images: ["6", 0], filename_prefix: "%filename_prefix%" } },
      ...overrides
    });
  const customSettings = (workflow) => ({ workflowPreset: "custom", customWorkflow: workflow, cfg: 5, modelFiles: { custom: { diffusionModel: "photo.safetensors" } } });
  assertCheck(scope, workflows.validateCustomWorkflowTemplate(template()).length === 0, "A template with %negative_prompt% on the negative side and %model% in its loader validates.");

  const negativeLiteral = template({ "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "clothes, shirt, %negative_prompt%" } } });
  const customRun = (await dryRun(customSettings(negativeLiteral), "custom", "anime")).payload;
  assertCheck(
    scope,
    customRun.workflow?.["3"]?.inputs?.text === "anime" && String(customRun.workflow?.["2"]?.inputs?.text).includes("cream wardrobe mirror") && customRun.workflow?.["5"]?.inputs?.cfg === 5,
    `A custom template's own negative words are cleaned like the job's ("clothes, shirt, %negative_prompt%" → ${JSON.stringify(customRun.workflow?.["3"]?.inputs?.text)}), its picture side is untouched, and it keeps the user's CFG.`
  );

  const negativeIntoPicture = template({ "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "%prompt%, %negative_prompt%" } } });
  const negativeErrors = workflows.validateCustomWorkflowTemplate(negativeIntoPicture);
  assertCheck(
    scope,
    negativeErrors.some((error) => /노드 "2" \(CLIPTextEncode\)/u.test(error) && /%negative_prompt%/u.test(error)),
    `%negative_prompt% wired into the picture's own encoder is invalid — the negative is text the floor never reads (${negativeErrors[0] ?? "no error"}).`
  );
  const negativeIntoPictureRun = await captureError(() => dryRun(customSettings(negativeIntoPicture), "custom"));
  assertCheck(
    scope,
    negativeIntoPictureRun instanceof Error && negativeIntoPictureRun.message.startsWith("실사 이미지 생성 실패") && /%negative_prompt%/u.test(negativeIntoPictureRun.message),
    "...and such a template is refused when the cut runs, not only in settings."
  );
  const modelIntoPicture = template({ "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "%prompt% %model%" } } });
  assertCheck(
    scope,
    workflows.validateCustomWorkflowTemplate(modelIntoPicture).some((error) => /%model%/u.test(error) && /_name/u.test(error)),
    "%model% (a file name typed in settings) is only accepted as a whole loader file input."
  );
  const qwenStyle = JSON.stringify({
    "1": { class_type: "UNETLoader", inputs: { unet_name: "qwen_image_2.1_turbo_int8_convrot.safetensors", weight_dtype: "default" } },
    "3": { class_type: "CLIPLoader", inputs: { clip_name: "qwen3vl_8b_int8_convrot.safetensors", type: "qwen_image", device: "default" } },
    "5": { class_type: "TextEncodeQwenImage21", inputs: { clip: ["3", 0], prompt: "%prompt%", negative_prompt: "%negative_prompt%", resolution: 1024 } },
    "6": { class_type: "EmptyLatentImage", inputs: { width: "%width%", height: "%height%", batch_size: 1 } },
    "7": { class_type: "KSampler", inputs: { model: ["1", 0], seed: "%seed%", steps: "%steps%", cfg: "%cfg%", sampler_name: "euler", scheduler: "simple", positive: ["5", 0], negative: ["5", 1], latent_image: ["6", 0], denoise: 1 } },
    "8": { class_type: "PreviewImage", inputs: { images: ["7", 0] } }
  });
  assertCheck(
    scope,
    workflows.validateCustomWorkflowTemplate(qwenStyle).length === 0,
    "An encoder that outputs both conditionings may take %negative_prompt% in its own negative_prompt input."
  );
  const swapped = JSON.parse(qwenStyle);
  swapped["7"].inputs.positive = ["5", 1];
  swapped["7"].inputs.negative = ["5", 0];
  assertCheck(
    scope,
    workflows.validateCustomWorkflowTemplate(JSON.stringify(swapped)).some((error) => /노드 "5"/u.test(error) && /%negative_prompt%/u.test(error)),
    "...but not once that encoder's negative output is wired into the sampler's positive: the side is read from the links, not the input's name."
  );
  // A graph does not have to say "positive": BasicGuider takes `conditioning`. The node below also feeds a sampler's
  // negative, which once made it look negative-only.
  const guided = template({
    "10": { class_type: "BasicGuider", inputs: { model: ["1", 0], conditioning: ["3", 0] } },
    "11": { class_type: "RandomNoise", inputs: { noise_seed: "%seed%" } },
    "12": { class_type: "KSamplerSelect", inputs: { sampler_name: "euler" } },
    "13": { class_type: "BasicScheduler", inputs: { model: ["1", 0], scheduler: "simple", steps: 8, denoise: 1 } },
    "14": { class_type: "SamplerCustomAdvanced", inputs: { noise: ["11", 0], guider: ["10", 0], sampler: ["12", 0], sigmas: ["13", 0], latent_image: ["4", 0] } },
    "15": { class_type: "VAEDecode", inputs: { samples: ["14", 0], vae: ["1", 2] } },
    "16": { class_type: "PreviewImage", inputs: { images: ["15", 0] } }
  });
  assertCheck(
    scope,
    workflows.validateCustomWorkflowTemplate(guided).some((error) => /노드 "3"/u.test(error) && /%negative_prompt%/u.test(error)),
    "%negative_prompt% in a node that reaches the picture through BasicGuider is invalid, though the node also feeds a sampler's negative."
  );

  // B: no file from ComfyUI's input folder ever enters a graph. A pass-through template would return it as the cut,
  // and as a profile candidate it would become a photoreal face reference.
  const passThrough = JSON.stringify({
    "1": { class_type: "LoadImage", inputs: { image: "any_photo_on_disk.png" } },
    "2": { class_type: "PreviewImage", inputs: { images: ["1", 0] } }
  });
  const passThroughErrors = workflows.validateCustomWorkflowTemplate(passThrough);
  assertCheck(
    scope,
    passThroughErrors.some((error) => /노드 "1" \(LoadImage\)/u.test(error) && /%image1%/u.test(error)),
    `A LoadImage pass-through template is invalid, naming the node (${passThroughErrors[0] ?? "no error"}).`
  );
  const passThroughRun = await captureError(() => dryRun(customSettings(passThrough), "custom"));
  assertCheck(
    scope,
    passThroughRun instanceof Error && passThroughRun.message.startsWith("실사 이미지 생성 실패") && /LoadImage/u.test(passThroughRun.message),
    "...and it never builds when the cut runs."
  );
  const imageShapes = [
    [{ "8": { class_type: "PrimitiveString", inputs: { value: "photo.png" } }, "9": { class_type: "LoadImage", inputs: { image: ["8", 0] } } }, "a LoadImage whose file name comes through a link"],
    [{ "9": { class_type: "LoadImageMask", inputs: { image: "mask.png", channel: "alpha" } } }, "a LoadImageMask with a literal file"],
    [{ "9": { class_type: "LoadImage", inputs: {} } }, "a LoadImage with no image at all"],
    [{ "9": { class_type: "SomeCustomLoader", inputs: { image: "photo.png" } } }, "any node with a string image input (an IMAGE socket is always a link)"]
  ];
  for (const [nodes, label] of imageShapes) {
    assertCheck(scope, workflows.validateCustomWorkflowTemplate(template(nodes)).some((error) => /노드 "9"/u.test(error) && /image/u.test(error)), `Invalid: ${label}.`);
  }
  assertCheck(
    scope,
    workflows.validateCustomWorkflowTemplate(template({ "9": { class_type: "LoadImage", inputs: { image: "%image1%" } } })).length === 0,
    "A LoadImage that takes %image1% (a reference the relay uploads) stays valid."
  );

  // A: the template's own words, found for the floor (the orchestrator screens them); file and list inputs are not text,
  // and negative-side text is cleaned instead of screened (a negative says "nude" to keep it out).
  const literals = workflows.listCustomWorkflowLiteralTexts(
    template({
      "1": { class_type: "UNETLoader", inputs: { unet_name: "some_model_v2.safetensors", weight_dtype: "fp8_e4m3fn" } },
      "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "A fully naked 14 year old girl. %prompt%" } },
      "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "nude, child, %negative_prompt%" } },
      "8": { class_type: "PrimitiveString", inputs: { value: { __value__: ["hidden words"] } } },
      "9": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "a 14 %prompt% year old" } }
    })
  );
  const byNode = Object.fromEntries(literals.map((literal) => [literal.nodeId, literal]));
  assertCheck(
    scope,
    byNode["2"]?.input === "text" &&
      byNode["2"]?.text === "A fully naked 14 year old girl." &&
      byNode["8"]?.text === "hidden words" &&
      byNode["9"]?.text === "a 14 year old" &&
      sameStrings(byNode["9"]?.fragments, ["a 14", "year old"]) &&
      !literals.some((literal) => ["1", "3", "4", "5", "7"].includes(literal.nodeId)),
    `listCustomWorkflowLiteralTexts finds every picture-side literal (nested ones, dangling ones and phrases split around %prompt% included) and skips file names, enums, placeholders and the negative side (${JSON.stringify(literals.map((literal) => [literal.nodeId, literal.text]))}).`
  );

  assertCheck(scope, fetchProbe.calls === 0, "None of this touched the network.");
}

function sameStrings(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]);
}

// ---------------------------------------------------------------------------------------------------------
// (a3) What is SENT (red team round 2, structural S1-S7). No guidance scale is ever below 1 — below 1 the sampler
//      draws the negative, the one text the floor does not read. Numbers fill only whole number inputs — inside a
//      sentence "%steps%" composes "a 12 year old" after screening. %prompt% is never glued to a word. The caller sees
//      the exact outgoing graph before it leaves. And a face reference is only ever a picked profile image's key.
// ---------------------------------------------------------------------------------------------------------
async function evaluateWhatIsSent(modules, relay) {
  const { seedState } = requireModule(modules, "seed");
  const comfy = requireModule(modules, "comfy");
  const workflows = requireModule(modules, "workflows");
  const fetchProbe = installForbiddenFetch();
  const nodesOf = (workflow, classType) => Object.values(workflow ?? {}).filter((node) => node?.class_type === classType);
  const dryRun = (photoreal, presetId, extra = {}) =>
    comfy.generatePhotorealImages({
      state: createState(seedState, { photoreal: { enabled: false, ...photoreal } }),
      prompt: PROMPT,
      negativePrompt: NEGATIVE_PROMPT,
      cue: createCue(),
      presetId,
      references: [],
      jobId: "job_eval_sent",
      ...extra
    });
  const customSettings = (workflow, extra = {}) => ({ workflowPreset: "custom", customWorkflow: workflow, cfg: 5, modelFiles: { custom: { diffusionModel: "photo.safetensors" } }, ...extra });
  const sampler = (inputs = {}) => ({
    class_type: "KSampler",
    inputs: { model: ["1", 0], seed: "%seed%", steps: "%steps%", cfg: "%cfg%", sampler_name: "euler", scheduler: "simple", positive: ["2", 0], negative: ["3", 0], latent_image: ["4", 0], denoise: 1, ...inputs }
  });
  const template = (overrides = {}) =>
    JSON.stringify({
      "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "%model%" } },
      "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "%prompt%" } },
      "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "%negative_prompt%" } },
      "4": { class_type: "EmptyLatentImage", inputs: { width: "%width%", height: "%height%", batch_size: 1 } },
      "5": sampler(),
      "6": { class_type: "VAEDecode", inputs: { samples: ["5", 0], vae: ["1", 2] } },
      "7": { class_type: "PreviewImage", inputs: { images: ["6", 0] } },
      ...overrides
    });
  const guided = (guider, extra = {}) =>
    template({
      "10": guider,
      "11": { class_type: "RandomNoise", inputs: { noise_seed: "%seed%" } },
      "12": { class_type: "KSamplerSelect", inputs: { sampler_name: "euler" } },
      "13": { class_type: "BasicScheduler", inputs: { model: ["1", 0], scheduler: "simple", steps: "%steps%", denoise: 1 } },
      "14": { class_type: "SamplerCustomAdvanced", inputs: { noise: ["11", 0], guider: ["10", 0], sampler: ["12", 0], sigmas: ["13", 0], latent_image: ["4", 0] } },
      "6": { class_type: "VAEDecode", inputs: { samples: ["14", 0], vae: ["1", 2] } },
      ...extra
    });

  // ---- S1-S3: the guidance floor.
  let scope = "photoreal.cfg-floor";
  assertCheck(
    scope,
    workflows.PHOTOREAL_MIN_CFG === 1 && Object.values(workflows.PHOTOREAL_WORKFLOW_PRESETS).every((preset) => preset.defaults.cfg >= 1 && (preset.fixedCfg ?? 1) >= 1),
    "PHOTOREAL_MIN_CFG is 1, and no preset's own CFG (default or fixed) is below it."
  );
  const resolvedBelow = [0.01, 0.1, 0.5, 0.999].flatMap((cfg) => ["qwen_image_2_1", "qwen_image_edit_2511", "custom"].map((presetId) => workflows.resolvePhotorealCfg({ cfg }, presetId)));
  assertCheck(scope, resolvedBelow.every((cfg) => cfg === 1), `resolvePhotorealCfg holds a setting below 1 at 1 on every preset that takes the setting (${resolvedBelow.join(",")}).`);
  assertCheck(
    scope,
    workflows.resolvePhotorealCfg({ cfg: 0 }, "qwen_image_2_1") === 1 &&
      workflows.resolvePhotorealCfg({ cfg: -2 }, "custom") === 4 &&
      workflows.resolvePhotorealCfg({ cfg: Number.NaN }, "qwen_image_edit_2511") === 4 &&
      workflows.resolvePhotorealCfg({ cfg: 99 }, "custom") === 30 &&
      workflows.resolvePhotorealCfg({ cfg: 1.5 }, "custom") === 1.5,
    "0, a negative and NaN still mean the preset's own CFG; 1.5 is kept; the top stays 30."
  );
  for (const presetId of ["qwen_image_2_1", "qwen_image_edit_2511", "qwen_image_2_1_turbo", "z_image_turbo"]) {
    const built = workflows.buildComfyWorkflow(presetId, {
      prompt: PROMPT,
      negativePrompt: NEGATIVE_PROMPT,
      width: 1024,
      height: 1024,
      seed: 1,
      steps: 8,
      cfg: 0.1,
      files: workflows.resolvePhotorealModelFiles({ modelFiles: {} }, presetId),
      referenceNames: [],
      filenamePrefix: "DynamicChat/job_eval_cfg"
    });
    const scales = Object.values(built.workflow).flatMap((node) => (node.inputs.cfg === undefined ? [] : [node.inputs.cfg]));
    assertCheck(scope, scales.length > 0 && scales.every((value) => value === 1), `${presetId}: handed CFG 0.1, the builder samples at 1 (${scales.join(",")}).`);
  }
  const lowSetting = (await dryRun({ cfg: 0.1 }, "qwen_image_2_1")).payload;
  assertCheck(
    scope,
    lowSetting.cfg === 1 && nodesOf(lowSetting.workflow, "KSampler")[0]?.inputs?.cfg === 1,
    `A settings CFG of 0.1 that never went through hydrate (red team S1) is sent — and recorded — as 1 (sent ${nodesOf(lowSetting.workflow, "KSampler")[0]?.inputs?.cfg}).`
  );
  const viaPlaceholder = (await dryRun(customSettings(template(), { cfg: 0.5 }), "custom")).payload;
  assertCheck(scope, viaPlaceholder.workflow?.["5"]?.inputs?.cfg === 1, `%cfg% in a custom template becomes 1 with the setting at 0.5 (got ${viaPlaceholder.workflow?.["5"]?.inputs?.cfg}).`);

  const refusedScales = [
    [template({ "5": sampler({ cfg: 0 }) }), "5", "a KSampler cfg literal 0 (red team S3: the negative-side literal became the picture)"],
    [template({ "5": sampler({ cfg: 0.5 }) }), "5", "a KSampler cfg literal 0.5"],
    [template({ "5": sampler({ cfg: "0.5" }) }), "5", "a KSampler cfg written as the string \"0.5\" (ComfyUI casts it to 0.5)"],
    [template({ "5": sampler({ cfg: ["20", 0] }), "20": { class_type: "PrimitiveFloat", inputs: { value: 0 } } }), "5", "a KSampler cfg fed by a link (a PrimitiveFloat holding 0)"],
    [template({ "5": sampler({ cfg: "loud" }) }), "5", "a KSampler cfg that is not a number"],
    [template({ "5": sampler({ cfg: "%clip_skip%" }) }), "5", "a KSampler cfg filled by %clip_skip% (always -1)"],
    [template({ "5": { ...sampler({ cfg: 0.9 }), class_type: "KSamplerAdvanced" } }), "5", "a KSamplerAdvanced cfg 0.9"],
    [guided({ class_type: "CFGGuider", inputs: { model: ["1", 0], positive: ["2", 0], negative: ["3", 0], cfg: 0.2 } }), "10", "a CFGGuider cfg 0.2"],
    [guided({ class_type: "DualCFGGuider", inputs: { model: ["1", 0], cond1: ["2", 0], cond2: ["2", 0], negative: ["3", 0], cfg_conds: 0, cfg_cond2_negative: 0 } }), "10", "a DualCFGGuider cfg_conds 0"],
    [
      guided(
        { class_type: "PerpNegGuider", inputs: { model: ["1", 0], positive: ["2", 0], negative: ["3", 0], empty_conditioning: ["15", 0], cfg: 4, neg_scale: 0.5 } },
        { "15": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "" } } }
      ),
      "10",
      "a PerpNegGuider neg_scale 0.5"
    ],
    [template({ "5": { class_type: "SamplerCustom", inputs: { model: ["1", 0], add_noise: true, noise_seed: "%seed%", cfg: 0, positive: ["2", 0], negative: ["3", 0], sampler: ["12", 0], sigmas: ["13", 0], latent_image: ["4", 0] } }, "12": { class_type: "KSamplerSelect", inputs: { sampler_name: "euler" } }, "13": { class_type: "BasicScheduler", inputs: { model: ["1", 0], scheduler: "simple", steps: 8, denoise: 1 } } }), "5", "a SamplerCustom cfg 0"]
  ];
  for (const [workflow, nodeId, label] of refusedScales) {
    const errors = workflows.validateCustomWorkflowTemplate(workflow);
    const run = await captureError(() => dryRun(customSettings(workflow), "custom"));
    assertCheck(
      scope,
      errors.some((error) => error.includes(`노드 "${nodeId}"`) && /CFG/u.test(error)) && run instanceof Error && run.message.startsWith("실사 이미지 생성 실패") && /CFG/u.test(run.message),
      `Refused in settings and never built: ${label} (${errors.find((error) => /CFG/u.test(error)) ?? "no error"}).`
    );
  }
  for (const [workflow, label] of [
    [template({ "5": sampler({ cfg: 1 }) }), "cfg 1"],
    [template({ "5": sampler({ cfg: 7.5 }) }), "cfg 7.5"],
    [template({ "5": sampler({ cfg: "%scale%" }) }), "cfg %scale% (the SillyTavern name)"],
    [guided({ class_type: "CFGGuider", inputs: { model: ["1", 0], positive: ["2", 0], negative: ["3", 0], cfg: "%cfg%" } }), "a CFGGuider taking %cfg%"],
    [template({ "20": { class_type: "CLIPTextEncodeLumina2", inputs: { clip: ["1", 1], text: "%prompt%", cfg_trunc: 0, renorm_cfg: 1 } } }), "cfg_trunc 0 (a step bound, not a scale)"]
  ]) {
    const errors = workflows.validateCustomWorkflowTemplate(workflow).filter((error) => /CFG/u.test(error));
    assertCheck(scope, errors.length === 0, `Valid: ${label} (${errors[0] ?? "ok"}).`);
  }

  // ---- S4: numbers only as whole number inputs.
  scope = "photoreal.template-numbers";
  const inText = [
    ["%prompt% She is a %steps% year old girl.", "%steps%"],
    ["%prompt% The girl is aged %cfg%.", "%cfg%"],
    ["%prompt% She was born in 20%steps%.", "%steps%"],
    ["%prompt% A %seed%yo girl in a bikini.", "%seed%"],
    ["%prompt% Aged %scale%, %width% by %height%.", "%scale%"],
    ["%prompt% %denoise% %clip_skip%", "%denoise%"],
    ["%steps%", "%steps%"]
  ];
  for (const [text, placeholder] of inText) {
    const workflow = template({ "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text } } });
    const errors = workflows.validateCustomWorkflowTemplate(workflow);
    const run = await captureError(() => dryRun(customSettings(workflow, { steps: 12 }), "custom"));
    assertCheck(
      scope,
      errors.some((error) => error.includes('노드 "2"') && error.includes(placeholder) && /숫자 입력/u.test(error)) && run instanceof Error && run.message.includes(placeholder),
      `A number placeholder in the picture's text is refused in settings and never built: ${JSON.stringify(text)}.`
    );
  }
  for (const [nodes, label] of [
    [{ "20": { class_type: "PrimitiveString", inputs: { value: "%seed%" } } }, "a PrimitiveString whose value is %seed%"],
    [{ "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "blurry, %steps%, %negative_prompt%" } } }, "%steps% inside the negative's text (any string, not only the picture's)"]
  ]) {
    assertCheck(scope, workflows.validateCustomWorkflowTemplate(template(nodes)).some((error) => /숫자 입력/u.test(error)), `Refused: ${label}.`);
  }
  const numeric = template({
    "5": { ...sampler({ cfg: "%scale%" }), class_type: "KSampler" },
    "20": { class_type: "RandomNoise", inputs: { noise_seed: "%seed%" } },
    "21": { class_type: "CLIPSetLastLayer", inputs: { clip: ["1", 1], stop_at_clip_layer: "%clip_skip%" } },
    "22": { class_type: "PrimitiveInt", inputs: { value: "%steps%" } },
    "23": { class_type: "PrimitiveFloat", inputs: { value: "%cfg%" } },
    "24": { class_type: "BasicScheduler", inputs: { model: ["1", 0], scheduler: "simple", steps: "%steps%", denoise: "%denoise%" } },
    "7": { class_type: "SaveImage", inputs: { images: ["6", 0], filename_prefix: "DynamicChat_%seed%_%width%" } }
  });
  const numericErrors = workflows.validateCustomWorkflowTemplate(numeric);
  assertCheck(
    scope,
    numericErrors.length === 0,
    `Valid: every number placeholder as the whole value of a number input (seed, noise_seed, steps, cfg, denoise, width, height, stop_at_clip_layer, PrimitiveInt/Float value), and inside a file-name prefix (${numericErrors[0] ?? "ok"}).`
  );
  const builtNumeric = (await dryRun(customSettings(numeric, { steps: 12, seedFixed: true, seed: 7 }), "custom")).payload.workflow ?? {};
  assertCheck(
    scope,
    builtNumeric["5"]?.inputs?.seed === 7 && builtNumeric["22"]?.inputs?.value === 12 && builtNumeric["20"]?.inputs?.noise_seed === 7 && typeof builtNumeric["5"]?.inputs?.steps === "number",
    "...and each becomes a JSON number in the built graph."
  );

  // ---- S5: %prompt% never glued to a word.
  scope = "photoreal.prompt-junction";
  for (const text of ["%prompt%ked, lying on a bed", "na%prompt%", "%prompt%\u200bked", "%prompt%2", "x%prompt% y"]) {
    const errors = workflows.validateCustomWorkflowTemplate(template({ "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text } } }));
    assertCheck(scope, errors.some((error) => error.includes('노드 "2"') && /%prompt% 바로 앞뒤/u.test(error)), `Refused: ${JSON.stringify(text)} (a word made across the boundary).`);
  }
  for (const text of ["%prompt% ked, lying on a bed", "%prompt%, by a window", "(%prompt%:1.1)", "%prompt%.", "A photo. %prompt%"]) {
    const errors = workflows.validateCustomWorkflowTemplate(template({ "2": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text } } }));
    assertCheck(scope, !errors.some((error) => /%prompt% 바로 앞뒤/u.test(error)), `Valid: ${JSON.stringify(text)} (separated by a space or punctuation).`);
  }

  // ---- The outgoing graph, as the caller's last look sees it.
  scope = "photoreal.sent-graph";
  const seen = [];
  const inspected = await dryRun({}, "qwen_image_2_1_turbo", { inspectWorkflow: (built) => seen.push(built), seed: 31 });
  assertCheck(
    scope,
    seen.length === 1 && JSON.stringify(seen[0].workflow) === JSON.stringify(inspected.payload.workflow),
    "inspectWorkflow sees exactly the graph the dry run reports, once."
  );
  const planned = comfy.buildPhotorealRequestWorkflow({
    state: createState(seedState, { photoreal: { enabled: false } }),
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    presetId: "qwen_image_2_1_turbo",
    references: [],
    jobId: "job_eval_sent",
    seed: 31
  });
  assertCheck(scope, JSON.stringify(planned.workflow) === JSON.stringify(inspected.payload.workflow), "buildPhotorealRequestWorkflow (the planner's no-network build) yields the same graph as the request.");
  const stopped = await captureError(() =>
    dryRun({}, "qwen_image_2_1_turbo", {
      inspectWorkflow: () => {
        throw new Error("refused by the caller");
      }
    })
  );
  assertCheck(scope, stopped instanceof Error && stopped.message === "refused by the caller", "A throwing inspectWorkflow stops a dry run and its error reaches the caller unchanged.");
  const unbuildable = await captureError(async () =>
    comfy.buildPhotorealRequestWorkflow({
      state: createState(seedState, { photoreal: { enabled: false, ...customSettings(template({ "5": sampler({ cfg: 0 }) })) } }),
      prompt: PROMPT,
      negativePrompt: NEGATIVE_PROMPT,
      presetId: "custom",
      references: [],
      jobId: "job_eval_sent",
      seed: 0
    })
  );
  assertCheck(scope, unbuildable instanceof Error && unbuildable.message.startsWith("실사 이미지 생성 실패") && /CFG/u.test(unbuildable.message), "buildPhotorealRequestWorkflow refuses an unbuildable template with the Korean prefix.");
  assertCheck(scope, fetchProbe.calls === 0, "None of the above touched the network.");

  globalThis.fetch = originalFetch;
  relay.reset();
  const relayed = [];
  await comfy.generatePhotorealImages({
    state: createRelayState(seedState, relay),
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    cue: createCue(),
    presetId: "qwen_image_2_1_turbo",
    references: PROFILE_REFERENCES,
    jobId: "job_eval_inspect",
    inspectWorkflow: (built) => relayed.push(JSON.stringify(built.workflow))
  });
  const posted = relay.requestsTo("/photoreal/comfyui/generate")[0]?.body?.workflow;
  assertCheck(scope, relayed.length === 1 && relayed[0] === JSON.stringify(posted), "With a live relay, inspectWorkflow sees exactly the graph that is posted.");
  relay.reset();
  const refusedLive = await captureError(() =>
    comfy.generatePhotorealImages({
      state: createRelayState(seedState, relay),
      prompt: PROMPT,
      negativePrompt: NEGATIVE_PROMPT,
      cue: createCue(),
      presetId: "qwen_image_2_1_turbo",
      references: PROFILE_REFERENCES,
      jobId: "job_eval_inspect_refused",
      inspectWorkflow: () => {
        throw new Error("refused by the caller");
      }
    })
  );
  assertCheck(scope, refusedLive?.message === "refused by the caller" && relay.requestsTo("/photoreal/comfyui/generate").length === 0, "A throwing inspectWorkflow means nothing is posted to the relay.");

  // The picture side of a built graph, as the floor reads it: the prompt; never the negative, file names or lists.
  const qwen = workflows.buildComfyWorkflow("qwen_image_2_1", {
    prompt: PROMPT,
    negativePrompt: "nude, child",
    width: 1024,
    height: 1024,
    seed: 1,
    steps: 8,
    cfg: 4,
    files: workflows.resolvePhotorealModelFiles({ modelFiles: {} }, "qwen_image_2_1"),
    referenceNames: ["char_1"],
    filenamePrefix: "DynamicChat/job_eval_texts"
  });
  const qwenTexts = workflows.listPhotorealGraphPictureTexts(qwen.workflow);
  assertCheck(
    scope,
    qwenTexts.length === 1 && qwenTexts[0].nodeId === "5" && qwenTexts[0].input === "prompt" && qwenTexts[0].text === PROMPT,
    `listPhotorealGraphPictureTexts on a built Qwen-Image 2.1 graph is exactly its prompt (not the negative, file names, the sampler, the %ref: loader) — ${JSON.stringify(qwenTexts.map((text) => `${text.nodeId}.${text.input}`))}.`
  );
  const twoEncoders = JSON.parse(template({
    "4": { class_type: "EmptyLatentImage", inputs: { width: 1024, height: 1024, batch_size: 1 } },
    "8": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "of twelve" } },
    "9": { class_type: "ConditioningCombine", inputs: { conditioning_1: ["2", 0], conditioning_2: ["8", 0] } },
    "5": sampler({ positive: ["9", 0], cfg: 4, seed: 1, steps: 8 }),
    "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "nude, child" } }
  }));
  const twoTexts = workflows.listPhotorealGraphPictureTexts(twoEncoders);
  assertCheck(
    scope,
    sameStrings(twoTexts.map((text) => text.nodeId).sort(), ["2", "8"]),
    `Every encoder whose conditioning reaches the picture is read, the negative's is not (${JSON.stringify(twoTexts.map((text) => text.nodeId))}).`
  );

  // A `negative` input is negative-only on a sampler or guider, which uses it up. A node that takes a `negative` and
  // puts conditioning out again (ControlNetApplyAdvanced) can hand it to the sampler's `positive` — there the
  // "negative" text is the picture, so it is read as the picture's own.
  scope = "photoreal.negative-side";
  const passThrough = (negativeText) =>
    template({
      "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: negativeText } },
      "20": {
        class_type: "ControlNetApplyAdvanced",
        inputs: { positive: ["2", 0], negative: ["3", 0], control_net: ["21", 0], image: ["22", 0], strength: 1, start_percent: 0, end_percent: 1 }
      },
      "21": { class_type: "ControlNetLoader", inputs: { control_net_name: "pose.safetensors" } },
      "22": { class_type: "EmptyImage", inputs: { width: 64, height: 64, batch_size: 1, color: 0 } },
      "5": sampler({ positive: ["20", 1], negative: ["20", 0] })
    });
  assertCheck(
    scope,
    workflows.validateCustomWorkflowTemplate(passThrough("%negative_prompt%")).some((error) => error.includes('노드 "3"') && /%negative_prompt%/u.test(error)),
    "%negative_prompt% that reaches the sampler's positive through a conditioning pass-through (ControlNetApplyAdvanced's negative output) is on the picture side, and refused"
  );
  const passedLiterals = workflows.listCustomWorkflowLiteralTexts(passThrough("a naked child"));
  const passedTexts = workflows.listPhotorealGraphPictureTexts(JSON.parse(passThrough("a naked child")));
  assertCheck(
    scope,
    passedLiterals.some((literal) => literal.nodeId === "3") && passedTexts.some((text) => text.nodeId === "3"),
    "...and its literal words are read by the floor as the picture's (template literals and built-graph texts alike)"
  );
  const sinkOnly = JSON.parse(template({ "3": { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "a naked child" } } }));
  assertCheck(
    scope,
    !workflows.listPhotorealGraphPictureTexts(sinkOnly).some((text) => text.nodeId === "3"),
    "while the same words wired only into a KSampler's negative stay negative-side (a negative says what to keep out)"
  );

  // ---- S7 / relay: a face reference is only ever a picked profile image, by its object key.
  scope = "references.picks-only";
  const accepted = ["sim_eval/profile_a.png", "draft_sim_x1_y2/profile_mg1_3.webp", "sim/profile_a.jpg", "sim/profile_a.jpeg", "sim/profile_a_reference.png"];
  const refused = ["sim_eval/asset_cut.png", "sim_eval/candidate.png", "profile_a.png", "a/b/profile_a.png", "../profile_a.png", "sim_eval/profile_a.gif", "sim_eval/profile a.png", "sim_eval/..", ".//profile_a.png", "", undefined, 42, "data:image/png;base64,AAAA", "C:/sim/profile_a.png"];
  assertCheck(
    scope,
    accepted.every((key) => comfy.isPickedProfileObjectKey(key)) && refused.every((key) => !comfy.isPickedProfileObjectKey(key)),
    `isPickedProfileObjectKey takes <simulation>/profile_*.png|jpg|jpeg|webp only (wrongly ${[...accepted.filter((key) => !comfy.isPickedProfileObjectKey(key)), ...refused.filter((key) => comfy.isPickedProfileObjectKey(key))].join(", ") || "none"}).`
  );
  const notPicks = await dryRun({}, "qwen_image_2_1_turbo", {
    references: [
      { name: "char_1", objectKey: "sim_eval/asset_cut.png", characterId: "char_a" },
      { name: "char_2", objectKey: "../profile_b.png", characterId: "char_b" },
      { name: "char_3", objectKey: "sim_eval/profile_c.png", characterId: "char_c", dataUrl: "data:image/png;base64,QUJD" }
    ]
  });
  const loaded = nodesOf(notPicks.payload.workflow, "LoadImage").map((node) => node.inputs.image);
  assertCheck(
    scope,
    sameStrings(loaded, ["%ref:char_3%"]) && JSON.stringify(notPicks.payload.references) === JSON.stringify([{ name: "char_3", objectKey: "sim_eval/profile_c.png", characterId: "char_c" }]),
    `References whose key is not a picked profile image (a story cut, a traversal) are dropped; a pick is sent by key alone (${JSON.stringify(loaded)}).`
  );
}

async function evaluatePhotorealSeedPolicy(modules) {
  const { seedState } = requireModule(modules, "seed");
  const { generatePhotorealImages } = requireModule(modules, "comfy");
  installForbiddenFetch();
  const run = (photoreal, seed) =>
    generatePhotorealImages({
      state: createState(seedState, { photoreal: { enabled: false, ...photoreal } }),
      prompt: PROMPT,
      negativePrompt: NEGATIVE_PROMPT,
      cue: createCue(),
      presetId: "qwen_image_2_1_turbo",
      references: [],
      jobId: "job_eval_seed",
      seed
    }).then((result) => result.payload);

  const fixed = await run({ seedFixed: true, seed: 424242 });
  assertCheck("photoreal.seed", fixed.seed === 424242, "seedFixed uses state.photoreal.seed.");
  assertCheck(
    "photoreal.seed",
    JSON.stringify(fixed.workflow).includes("424242"),
    "The seed reported in the payload is the one wired into the graph."
  );

  const folded = [await run({ seedFixed: true, seed: 2 ** 32 + 7 }), await run({ seedFixed: true, seed: 2 ** 32 + 7 })];
  assertCheck(
    "photoreal.seed",
    folded.every((payload) => isSeed(payload.seed)) && folded[0].seed === folded[1].seed,
    `An out-of-range fixed seed is folded into 0..2^31-1 and stays reproducible (${folded[0].seed}).`
  );

  const explicit = await run({ seedFixed: true, seed: 424242 }, 99);
  assertCheck("photoreal.seed", explicit.seed === 99, "An explicit per-call seed wins over the fixed seed.");

  const randomSeeds = [];
  for (let index = 0; index < 6; index += 1) {
    randomSeeds.push((await run({ seedFixed: false })).seed);
  }
  assertCheck(
    "photoreal.seed",
    randomSeeds.every(isSeed) && new Set(randomSeeds).size > 1,
    "Unfixed seeds are random 31-bit integers."
  );

  const fixedWithoutSeed = await run({ seedFixed: true, seed: undefined });
  assertCheck("photoreal.seed", isSeed(fixedWithoutSeed.seed), "seedFixed without a stored seed still produces a valid seed.");
}

// ---------------------------------------------------------------------------------------------------------
// (b) Real mode against a stub relay that speaks the spec §4 contract.
// ---------------------------------------------------------------------------------------------------------
async function evaluatePhotorealRelayRequest(modules, relay) {
  const { seedState } = requireModule(modules, "seed");
  const { generatePhotorealImages, getPhotorealRelayBase } = requireModule(modules, "comfy");
  const state = createRelayState(seedState, relay);

  relay.reset();
  const result = await generatePhotorealImages({
    state,
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    cue: createCue(),
    presetId: "qwen_image_2_1_turbo",
    references: PROFILE_REFERENCES,
    jobId: "job_eval_real"
  });
  const generateRequests = relay.requestsTo("/photoreal/comfyui/generate");
  const sent = generateRequests[0]?.body ?? {};
  const sentWorkflow = isRecord(sent.workflow) ? sent.workflow : {};

  assertCheck(
    "photoreal.relay",
    generateRequests.length === 1 && generateRequests[0].method === "POST" && /application\/json/u.test(generateRequests[0].headers["content-type"] ?? ""),
    "One JSON POST to <relay>/generate."
  );
  assertCheck("photoreal.relay", sent.baseUrl === "http://127.0.0.1:8188", "Request names the configured ComfyUI baseUrl.");
  assertCheck(
    "photoreal.relay",
    Object.values(sentWorkflow).some((node) => node?.class_type === "PreviewImage") && JSON.stringify(sentWorkflow).includes("smiling"),
    "Request carries the built API-format graph."
  );
  assertCheck(
    "photoreal.relay",
    Array.isArray(sent.outputNodeIds) && sent.outputNodeIds.length > 0 && sent.outputNodeIds.every((id) => sentWorkflow[id]),
    "Request outputNodeIds exist in the graph."
  );
  // Changed deliberately (round 3): the dataUrl-only reference used to go out as bytes.
  assertCheck(
    "photoreal.relay",
    JSON.stringify(sent.references) === JSON.stringify([{ name: "char_1", objectKey: "sim_eval/profile_a.png" }]) &&
      !JSON.stringify(sent).includes("data:image") &&
      !JSON.stringify(sent).includes(REFERENCE_BASE64),
    "References go out as {name, objectKey} only: bytes riding on a reference are never sent, and a reference with no picked-profile key is dropped."
  );
  assertCheck("photoreal.relay", sent.timeoutMs === 120_000, "timeoutMs is timeoutSeconds * 1000 (the relay times out first).");
  assertCheck("photoreal.relay", sent.clientRequestId === "job_eval_real", "clientRequestId is the job id.");
  assertCheck(
    "photoreal.relay",
    result.dataUrls.length === 1 && result.dataUrls[0] === `data:image/png;base64,${PNG_BASE64}`,
    "Relay images come back as data:image/png;base64 URLs."
  );
  assertCheck(
    "photoreal.relay",
    result.payload.mode === "provider-result" && result.payload.promptId === "prompt_eval_1" && result.payload.elapsedMs === 1234,
    "Result payload records promptId, elapsedMs and mode 'provider-result'."
  );
  assertCheck(
    "photoreal.relay",
    !("workflow" in result.payload) && result.payload.workflowSummary?.nodeCount === Object.keys(sentWorkflow).length,
    "Result payload drops the full graph and keeps workflowSummary."
  );
  assertCheck(
    "photoreal.relay",
    !JSON.stringify(result.payload).includes(REFERENCE_BASE64) && !JSON.stringify(result.payload).includes("data:image"),
    "Result payload (stored as providerMetadata) carries no image bytes."
  );
  assertCheck(
    "photoreal.relay",
    isSeed(result.payload.seed) && JSON.stringify(sentWorkflow).includes(String(result.payload.seed)),
    "Result payload's seed is the one the graph was sent with."
  );

  relay.reset();
  relay.script({ status: 200, body: { promptId: "prompt_eval_mime", images: [{ mimeType: "text/html", base64: PNG_BASE64 }], elapsedMs: 5 } });
  const oddMime = await generatePhotorealImages({
    state,
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    cue: createCue(),
    presetId: "qwen_image_2_1_turbo",
    references: [],
    jobId: "job_eval_mime"
  });
  assertCheck(
    "photoreal.relay",
    oddMime.dataUrls[0] === `data:image/png;base64,${PNG_BASE64}`,
    "A non-image mimeType from the relay never becomes the data URL type (falls back to image/png)."
  );

  const withActionSuffix = { ...state, photoreal: { ...state.photoreal, proxyUrl: `${relay.url}/photoreal/comfyui/generate/` } };
  assertCheck(
    "photoreal.relay",
    getPhotorealRelayBase(withActionSuffix) === `${relay.url}/photoreal/comfyui`,
    "A pasted …/generate proxy URL is normalised back to the relay base."
  );
  const withoutOverride = { ...state, photoreal: { ...state.photoreal, proxyUrl: "  " } };
  assertCheck(
    "photoreal.relay",
    getPhotorealRelayBase(withoutOverride) === "http://127.0.0.1:8788/photoreal/comfyui",
    "With no proxyUrl the relay base is the configured API base + /photoreal/comfyui, and reading it does not throw in Node."
  );
}

async function evaluatePhotorealRelayErrors(modules, relay) {
  const { seedState } = requireModule(modules, "seed");
  const { generatePhotorealImages } = requireModule(modules, "comfy");
  const state = createRelayState(seedState, relay);
  const run = (overrides = {}) =>
    captureError(() =>
      generatePhotorealImages({
        state: overrides.state ?? state,
        prompt: PROMPT,
        negativePrompt: NEGATIVE_PROMPT,
        cue: createCue(),
        presetId: "qwen_image_2_1_turbo",
        references: PROFILE_REFERENCES,
        jobId: "job_eval_error"
      })
    );

  const relayErrors = [
    [400, "ComfyUI가 워크플로를 거부했습니다: UNETLoader: 모델 파일 qwen_image_2.1_turbo_int8_convrot.safetensors 이(가) 없습니다."],
    [500, "ComfyUI 실행 오류: KSampler: CUDA out of memory"],
    [504, "ComfyUI가 120초 안에 끝나지 않아 작업을 중단했습니다."]
  ];
  for (const [status, message] of relayErrors) {
    relay.reset();
    relay.script({ status, body: { error: message } });
    const error = await run();
    assertCheck(
      "photoreal.relay-errors",
      error instanceof Error && error.message.startsWith("실사 이미지 생성 실패") && error.message.includes(message),
      `Relay ${status} surfaces as '실사 이미지 생성 실패: <server error>'.`
    );
    assertCheck("photoreal.relay-errors", relay.requestsTo("/photoreal/comfyui/generate").length === 1, `Relay ${status} is not retried.`);
  }

  relay.reset();
  relay.script(
    { status: 502, body: { error: "ComfyUI 서버에 연결할 수 없습니다 (http://127.0.0.1:8188). ComfyUI를 실행했는지 확인하세요." } },
    "ok"
  );
  const recovered = await generatePhotorealImages({
    state,
    prompt: PROMPT,
    negativePrompt: NEGATIVE_PROMPT,
    cue: createCue(),
    presetId: "qwen_image_2_1_turbo",
    references: [],
    jobId: "job_eval_retry"
  });
  assertCheck(
    "photoreal.relay-errors",
    recovered.dataUrls.length === 1 && relay.requestsTo("/photoreal/comfyui/generate").length === 2,
    "A 502 (ComfyUI unreachable) is retried once and the retry's image is returned."
  );

  relay.reset();
  relay.script({ status: 503, body: { error: "ComfyUI가 바쁩니다." } }, { status: 503, body: { error: "ComfyUI가 바쁩니다." } }, "ok");
  const persistent = await run();
  assertCheck(
    "photoreal.relay-errors",
    persistent instanceof Error && persistent.message.startsWith("실사 이미지 생성 실패") && relay.requestsTo("/photoreal/comfyui/generate").length === 2,
    "A persistent 503 is retried exactly once, then fails with the Korean prefix."
  );

  relay.reset();
  relay.script({ status: 200, body: { promptId: "prompt_eval_empty", images: [], elapsedMs: 10 } });
  const empty = await run();
  assertCheck(
    "photoreal.relay-errors",
    empty instanceof Error && empty.message.startsWith("실사 이미지 생성 실패") && empty.message.includes("SaveImage"),
    "A 200 with no images fails and points at the missing SaveImage output."
  );

  relay.reset();
  relay.script({ status: 200, raw: "<html>proxy</html>" });
  const garbage = await run();
  assertCheck(
    "photoreal.relay-errors",
    garbage instanceof Error && garbage.message.startsWith("실사 이미지 생성 실패"),
    "A 200 that is not JSON fails with the Korean prefix."
  );

  relay.reset();
  const missingRoute = await run({ state: { ...state, photoreal: { ...state.photoreal, proxyUrl: `${relay.url}/no-such-relay` } } });
  assertCheck(
    "photoreal.relay-errors",
    missingRoute instanceof Error && missingRoute.message.startsWith("실사 이미지 생성 실패") && missingRoute.message.includes("릴레이 경로"),
    "A bare 404 explains that the relay route itself is missing (old API server or wrong proxy URL)."
  );

  relay.reset();
  const relayNotFound = "참조 이미지 char_1을(를) 찾을 수 없습니다 (sim_eval/profile_a.png).";
  relay.script({ status: 404, body: { error: relayNotFound } });
  const specific404 = await run();
  assertCheck(
    "photoreal.relay-errors",
    specific404 instanceof Error && specific404.message.includes(relayNotFound) && !specific404.message.includes("릴레이 경로"),
    "A 404 that carries the relay's own message keeps that message."
  );

  relay.reset();
  relay.script({ status: 500, raw: "<html><body><h1>Internal Server Error</h1></body></html>" });
  const htmlError = await run();
  assertCheck(
    "photoreal.relay-errors",
    htmlError instanceof Error && htmlError.message.startsWith("실사 이미지 생성 실패") && htmlError.message.includes("HTTP 500") && !htmlError.message.includes("<"),
    "An HTML error page from something in between becomes 'HTTP 500', not markup in the toast."
  );

  const closedPort = await findClosedPort();
  const unreachable = await run({ state: { ...state, photoreal: { ...state.photoreal, proxyUrl: `http://127.0.0.1:${closedPort}/photoreal/comfyui` } } });
  assertCheck(
    "photoreal.relay-errors",
    unreachable instanceof Error && unreachable.message.startsWith("실사 이미지 생성 실패") && unreachable.message.includes("DynamicChat API 서버"),
    "An unreachable relay fails with the prefix and names the DynamicChat API server."
  );
}

async function evaluatePhotorealAbort(modules, relay) {
  const { seedState } = requireModule(modules, "seed");
  const { generatePhotorealImages } = requireModule(modules, "comfy");
  const state = createRelayState(seedState, relay);

  relay.reset();
  relay.script("hang");
  const arrived = relay.nextGenerate();
  const controller = new AbortController();
  const startedAt = Date.now();
  const pending = captureError(() =>
    generatePhotorealImages({
      state,
      prompt: PROMPT,
      negativePrompt: NEGATIVE_PROMPT,
      cue: createCue(),
      presetId: "qwen_image_2_1_turbo",
      references: [],
      jobId: "job_eval_abort",
      signal: controller.signal
    })
  );
  await Promise.race([arrived, sleep(5_000)]);
  controller.abort();
  const aborted = await pending;
  const elapsedMs = Date.now() - startedAt;
  const disconnected = await relay.waitForDisconnect(2_000);

  assertCheck(
    "photoreal.abort",
    aborted instanceof Error && aborted.name === "AbortError",
    `Aborting the signal rejects with the caller's AbortError (${aborted?.name ?? "resolved"}), not a wrapped failure.`
  );
  assertCheck("photoreal.abort", elapsedMs < 5_000, `The abort takes effect immediately (${elapsedMs} ms).`);
  assertCheck("photoreal.abort", disconnected, "The relay sees the request closed, so it can interrupt ComfyUI.");
  assertCheck("photoreal.abort", relay.requestsTo("/photoreal/comfyui/generate").length === 1, "An aborted request is not retried.");

  relay.reset();
  const preAborted = new AbortController();
  preAborted.abort();
  const early = await captureError(() =>
    generatePhotorealImages({
      state,
      prompt: PROMPT,
      negativePrompt: NEGATIVE_PROMPT,
      cue: createCue(),
      presetId: "qwen_image_2_1_turbo",
      references: [],
      jobId: "job_eval_preabort",
      signal: preAborted.signal
    })
  );
  assertCheck(
    "photoreal.abort",
    early instanceof Error && early.name === "AbortError" && relay.requestsTo("/photoreal/comfyui/generate").length === 0,
    "An already-aborted signal rejects without sending anything."
  );
}

// ---------------------------------------------------------------------------------------------------------
// (c) probeComfyUi / free / interrupt against the stub relay.
// ---------------------------------------------------------------------------------------------------------
async function evaluateComfyUiProbe(modules, relay) {
  const { seedState } = requireModule(modules, "seed");
  const { probeComfyUi } = requireModule(modules, "comfy");
  const { PHOTOREAL_WORKFLOW_PRESETS } = requireModule(modules, "workflows");
  // Probing works before photoreal is enabled: it is how the user checks the setup first.
  // Face lock on, so both presets are in play (it is off by default now; the face-lock-off case is checked below).
  const state = createRelayState(seedState, relay, {
    enabled: false,
    workflowPreset: "z_image_turbo",
    referenceWorkflowPreset: "qwen_image_2_1_turbo",
    faceReferenceEnabled: true
  });
  const requiredNodes = Array.from(
    new Set([...PHOTOREAL_WORKFLOW_PRESETS.z_image_turbo.requiredNodes, ...PHOTOREAL_WORKFLOW_PRESETS.qwen_image_2_1_turbo.requiredNodes])
  );

  relay.reset();
  relay.setStatus({
    status: 200,
    body: {
      ok: true,
      comfyuiVersion: "0.3.75",
      devices: [{ name: "cuda:0 NVIDIA GeForce RTX 4080 SUPER", vramTotalMb: 16376, vramFreeMb: 15012, extra: "ignored" }],
      models: {
        diffusion_models: ["qwen_image_2.1_turbo_int8_convrot.safetensors", 42],
        text_encoders: ["qwen3vl_8b_int8_convrot.safetensors"],
        vae: ["qwen_image_2.1_vae_bf16.safetensors"],
        loras: []
      },
      nodes: Object.fromEntries(requiredNodes.map((node) => [node, true])),
      queueRemaining: 0
    }
  });
  const probe = await probeComfyUi(state);
  const statusRequest = relay.requestsTo("/photoreal/comfyui/status")[0];
  const queriedNodes = (statusRequest?.query.nodes ?? "").split(",").filter(Boolean);

  assertCheck("photoreal.probe", probe.ok === true && probe.comfyuiVersion === "0.3.75", "Probe reports ok and the ComfyUI version.");
  assertCheck(
    "photoreal.probe",
    probe.devices?.length === 1 && probe.devices[0].vramTotalMb === 16376 && probe.devices[0].vramFreeMb === 15012 && !("extra" in probe.devices[0]),
    "Probe devices carry name and VRAM figures only."
  );
  assertCheck(
    "photoreal.probe",
    JSON.stringify(probe.models?.diffusion_models) === JSON.stringify(["qwen_image_2.1_turbo_int8_convrot.safetensors"]) &&
      Array.isArray(probe.models?.loras) &&
      probe.models.loras.length === 0,
    "Probe models list file names per folder (non-strings dropped)."
  );
  assertCheck(
    "photoreal.probe",
    requiredNodes.every((node) => probe.nodes?.[node] === true) && probe.queueRemaining === 0,
    "Probe passes the node availability map and queue length through."
  );
  assertCheck("photoreal.probe", /연결되었습니다/u.test(probe.message), `Probe composes a Korean success message (${probe.message}).`);
  assertCheck("photoreal.probe", statusRequest?.query.baseUrl === "http://127.0.0.1:8188", "Probe asks about the configured ComfyUI baseUrl.");
  assertCheck(
    "photoreal.probe",
    requiredNodes.length > 0 && requiredNodes.every((node) => queriedNodes.includes(node)) && queriedNodes.length <= 30,
    `Probe asks for the requiredNodes of both configured presets (${queriedNodes.length} nodes, cap 30).`
  );

  relay.reset();
  await probeComfyUi(state, "http://localhost:8190");
  assertCheck(
    "photoreal.probe",
    relay.requestsTo("/photoreal/comfyui/status")[0]?.query.baseUrl === "http://localhost:8190",
    "An explicit baseUrl argument overrides the stored one (testing an unsaved field)."
  );

  // Face lock off: the face-lock preset never runs, so its nodes are not asked about. Asking made a Z-Image-only
  // ComfyUI report Qwen-Image 2.1's nodes as missing with face lock switched off.
  relay.reset();
  await probeComfyUi(
    createRelayState(seedState, relay, {
      enabled: true,
      workflowPreset: "z_image_turbo",
      referenceWorkflowPreset: "qwen_image_2_1_turbo",
      faceReferenceEnabled: false
    })
  );
  const unlockedNodes = (relay.requestsTo("/photoreal/comfyui/status")[0]?.query.nodes ?? "").split(",").filter(Boolean);
  const qwenOnlyNodes = PHOTOREAL_WORKFLOW_PRESETS.qwen_image_2_1_turbo.requiredNodes.filter(
    (node) => !PHOTOREAL_WORKFLOW_PRESETS.z_image_turbo.requiredNodes.includes(node)
  );
  assertCheck(
    "photoreal.probe",
    qwenOnlyNodes.length > 0 && qwenOnlyNodes.every((node) => !unlockedNodes.includes(node)) &&
      PHOTOREAL_WORKFLOW_PRESETS.z_image_turbo.requiredNodes.every((node) => unlockedNodes.includes(node)),
    `With face lock off only the generation preset's nodes are probed (${unlockedNodes.length} nodes).`
  );

  relay.reset();
  relay.setStatus({ status: 200, body: { ok: true, comfyuiVersion: "0.3.40", nodes: { TextEncodeQwenImage21: false, SaveImage: true } } });
  const outdated = await probeComfyUi(state);
  assertCheck(
    "photoreal.probe",
    outdated.ok === true && outdated.nodes?.TextEncodeQwenImage21 === false && outdated.message.includes("TextEncodeQwenImage21"),
    "A missing node is named in the message so the user knows to update ComfyUI."
  );

  relay.reset();
  const downMessage = "ComfyUI 서버에 연결할 수 없습니다 (http://127.0.0.1:8188). ComfyUI를 실행했는지 확인하세요.";
  relay.setStatus({ status: 200, body: { ok: false, message: downMessage } });
  const down = await probeComfyUi(state);
  assertCheck("photoreal.probe", down.ok === false && down.message === downMessage, "ComfyUI down: ok false with the relay's Korean message.");

  relay.reset();
  relay.setStatus({ status: 403, body: { error: "Forbidden origin" } });
  const forbidden = await captureError(() => probeComfyUi(state));
  assertCheck(
    "photoreal.probe",
    !(forbidden instanceof Error) && forbidden?.ok === false && forbidden.message.includes("Forbidden origin"),
    "A relay 403 becomes ok:false with the reason, never a throw."
  );

  relay.reset();
  relay.setStatus({ status: 200, raw: "<html>not json</html>" });
  const garbage = await captureError(() => probeComfyUi(state));
  assertCheck("photoreal.probe", !(garbage instanceof Error) && garbage?.ok === false, "A non-JSON relay answer becomes ok:false, never a throw.");

  const closedPort = await findClosedPort();
  const unreachable = await captureError(() =>
    probeComfyUi({ ...state, photoreal: { ...state.photoreal, proxyUrl: `http://127.0.0.1:${closedPort}/photoreal/comfyui` } })
  );
  assertCheck(
    "photoreal.probe",
    !(unreachable instanceof Error) && unreachable?.ok === false && unreachable.message.includes("DynamicChat API 서버"),
    "An unreachable relay becomes ok:false naming the DynamicChat API server, never a throw."
  );

  relay.reset();
  relay.setStatus({ status: 200, body: { ok: true } });
  const customState = createRelayState(seedState, relay, {
    workflowPreset: "custom",
    referenceWorkflowPreset: "custom",
    customWorkflow: JSON.stringify({
      1: { class_type: "MyCustomSampler", inputs: {} },
      2: { class_type: "SaveImage", inputs: { images: ["1", 0], filename_prefix: "x" } }
    })
  });
  await probeComfyUi(customState);
  const customNodes = (relay.requestsTo("/photoreal/comfyui/status")[0]?.query.nodes ?? "").split(",");
  assertCheck("photoreal.probe", customNodes.includes("MyCustomSampler"), "A custom preset's own class types are probed too.");
}

async function evaluateBestEffortCalls(modules, relay) {
  const { seedState } = requireModule(modules, "seed");
  const { freeComfyUiMemory, interruptComfyUi } = requireModule(modules, "comfy");
  const state = createRelayState(seedState, relay);

  relay.reset();
  const freed = await freeComfyUiMemory(state);
  const freeRequest = relay.requestsTo("/photoreal/comfyui/free")[0];
  assertCheck(
    "photoreal.best-effort",
    freed === undefined && freeRequest?.method === "POST" && freeRequest.body?.baseUrl === "http://127.0.0.1:8188",
    "freeComfyUiMemory posts {baseUrl} to <relay>/free."
  );

  relay.reset();
  await interruptComfyUi(state, "prompt_eval_9");
  const interruptRequest = relay.requestsTo("/photoreal/comfyui/interrupt")[0];
  assertCheck(
    "photoreal.best-effort",
    interruptRequest?.body?.baseUrl === "http://127.0.0.1:8188" && interruptRequest.body.promptId === "prompt_eval_9",
    "interruptComfyUi posts {baseUrl, promptId} to <relay>/interrupt."
  );

  relay.reset();
  await interruptComfyUi(state, "   ");
  await freeComfyUiMemory({ ...state, photoreal: { ...state.photoreal, enabled: false } });
  await freeComfyUiMemory({ ...state, photoreal: { ...state.photoreal, requestMode: "mock" } });
  assertCheck(
    "photoreal.best-effort",
    relay.requests.length === 0,
    "No prompt id, or a disabled/mock backend, sends nothing (nothing was ever queued)."
  );

  relay.reset();
  relay.setFree({ status: 500, body: { error: "boom" } });
  const failedFree = await captureError(() => freeComfyUiMemory(state));
  relay.setFree({ status: 200, body: { ok: true } });
  const closedPort = await findClosedPort();
  const offline = { ...state, photoreal: { ...state.photoreal, proxyUrl: `http://127.0.0.1:${closedPort}/photoreal/comfyui` } };
  const offlineFree = await captureError(() => freeComfyUiMemory(offline));
  const offlineInterrupt = await captureError(() => interruptComfyUi(offline, "prompt_eval_10"));
  assertCheck(
    "photoreal.best-effort",
    ![failedFree, offlineFree, offlineInterrupt].some((outcome) => outcome instanceof Error),
    "free / interrupt never throw (relay error or relay unreachable)."
  );
}

// ---------------------------------------------------------------------------------------------------------
// (d) NovelAI: Precise Reference payload, vibe exclusion, V5 skip, encode-vibe schema.
// ---------------------------------------------------------------------------------------------------------
async function evaluatePreciseReferencePayload(modules) {
  const { seedState } = requireModule(modules, "seed");
  const { generateNovelAiImages } = requireModule(modules, "novelAi");
  const fetchProbe = installForbiddenFetch();
  const reference = { imageBase64: REFERENCE_BASE64, strength: 0.7, fidelity: 0.75, styleAware: true };
  const parametersFor = async (modelPreset, characterReference, novelAi = {}) =>
    (
      await generateNovelAiImages({
        state: createState(seedState, {
          novelAi: { enabled: false, modelPreset, vibeTransferEnabled: true, vibeTransferReferences: [ENCODED_VIBE], ...novelAi }
        }),
        prompt: "1girl, smile, bedroom",
        negativePrompt: "lowres",
        cue: createCue(),
        count: 1,
        characterReference
      })
    ).payload;

  const v45Payload = await parametersFor("NAID4.5F", reference);
  const v45 = v45Payload.parameters ?? {};
  const directorArrays = [
    v45.director_reference_images,
    v45.director_reference_descriptions,
    v45.director_reference_information_extracted,
    v45.director_reference_strength_values,
    v45.director_reference_secondary_strength_values
  ];
  assertCheck(
    "nai.precise-reference",
    directorArrays.every((array) => Array.isArray(array) && array.length === 1),
    "V4.5 + reference: all five director_reference_* arrays present with equal length."
  );
  assertCheck(
    "nai.precise-reference",
    JSON.stringify(v45.director_reference_descriptions?.[0]) ===
      JSON.stringify({ caption: { base_caption: "character&style", char_captions: [] }, legacy_uc: false }),
    "styleAware → base_caption 'character&style', empty char_captions, legacy_uc false."
  );
  assertCheck(
    "nai.precise-reference",
    v45.director_reference_information_extracted?.[0] === 1 && v45.director_reference_strength_values?.[0] === 0.7,
    "information_extracted is 1 and strength is passed through."
  );
  assertCheck(
    "nai.precise-reference",
    v45.director_reference_secondary_strength_values?.[0] === 0.25,
    "Secondary strength is 1 - fidelity (fidelity 0.75 → 0.25)."
  );
  assertCheck(
    "nai.precise-reference",
    v45.reference_image_multiple === undefined &&
      v45.reference_strength_multiple === undefined &&
      v45.reference_information_extracted_multiple === undefined &&
      v45.dynamicchat_vibe_transfer_skipped === "character_reference",
    "Vibe arrays are dropped when a reference is attached, even with encoded vibes enabled."
  );
  assertCheck(
    "nai.precise-reference",
    v45.director_reference_images?.[0] === `<omitted:${REFERENCE_BYTES} bytes>` && !JSON.stringify(v45Payload).includes(REFERENCE_BASE64),
    `Returned payload replaces the reference base64 with '<omitted:${REFERENCE_BYTES} bytes>'.`
  );

  const plain = (await parametersFor("NAID4.5F", { ...reference, styleAware: false })).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    plain.director_reference_descriptions?.[0]?.caption?.base_caption === "character",
    "styleAware false → base_caption 'character'."
  );

  const lowFidelity = (await parametersFor("NAID4.5F", { ...reference, fidelity: 0.333 })).parameters ?? {};
  const fullFidelity = (await parametersFor("NAID4.5F", { ...reference, fidelity: 1 })).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    lowFidelity.director_reference_secondary_strength_values?.[0] === 0.67 && fullFidelity.director_reference_secondary_strength_values?.[0] === 0,
    "Secondary strength is rounded to two decimals (0.333 → 0.67, 1 → 0)."
  );

  const curated = (await parametersFor("NAID4.5C", reference)).parameters ?? {};
  assertCheck("nai.precise-reference", Array.isArray(curated.director_reference_images), "NAID4.5C (curated) takes the reference too.");

  const dataUrlReference = (await parametersFor("NAID4.5F", { ...reference, imageBase64: `data:image/png;base64,${REFERENCE_BASE64}` })).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    dataUrlReference.director_reference_images?.[0] === `<omitted:${REFERENCE_BYTES} bytes>`,
    "A data URL reference is accepted (prefix stripped)."
  );

  const nanStrength = (await parametersFor("NAID4.5F", { ...reference, strength: Number.NaN })).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    nanStrength.director_reference_strength_values?.[0] === 0.85,
    "A non-numeric strength falls back to the 0.85 default instead of a useless 0."
  );

  const v5 = (await parametersFor("NAID5F", reference)).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    v5.dynamicchat_character_reference_skipped === "model" && !Object.keys(v5).some((key) => key.startsWith("director_reference_")),
    "NAID5F skips the reference with parameters.dynamicchat_character_reference_skipped = 'model'."
  );
  assertCheck(
    "nai.precise-reference",
    v5.reference_image_multiple === undefined && v5.reference_strength_multiple === undefined,
    "NAID5F never receives vibe arrays."
  );
  const v5NoReference = (await parametersFor("NAID5F", undefined)).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    v5NoReference.reference_image_multiple === undefined && v5NoReference.dynamicchat_vibe_transfer_skipped === "model",
    "NAID5F skips encoded vibes even without a reference, and says why."
  );

  const v40 = (await parametersFor("NAID4.0F", reference)).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    v40.dynamicchat_character_reference_skipped === "model" &&
      v40.director_reference_images === undefined &&
      v40.reference_image_multiple?.[0] === ENCODED_VIBE.encodedVibe,
    "NAID4.0F skips the reference but still sends its vibes."
  );

  const v3 = (await parametersFor("NAID3", reference)).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    v3.dynamicchat_character_reference_skipped === "model" && Array.isArray(v3.reference_information_extracted_multiple),
    "NAID3 skips the reference and keeps its raw-image vibes."
  );

  const v45NoReference = (await parametersFor("NAID4.5F", undefined)).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    v45NoReference.reference_image_multiple?.[0] === ENCODED_VIBE.encodedVibe &&
      v45NoReference.director_reference_images === undefined &&
      v45NoReference.dynamicchat_vibe_transfer_skipped === undefined &&
      v45NoReference.dynamicchat_character_reference_skipped === undefined,
    "Without a reference V4.5 behaves as before: vibes sent, no director arrays, no skip flags."
  );

  const emptyReference = (await parametersFor("NAID4.5F", { ...reference, imageBase64: "   " })).parameters ?? {};
  assertCheck(
    "nai.precise-reference",
    emptyReference.director_reference_images === undefined && emptyReference.reference_image_multiple?.[0] === ENCODED_VIBE.encodedVibe,
    "An empty reference image is treated as no reference (vibes keep working)."
  );

  assertCheck("nai.precise-reference", fetchProbe.calls === 0, "Disabled NovelAI dry runs never call fetch.");
}

async function evaluateNovelAiRequestBody(modules) {
  const { seedState } = requireModule(modules, "seed");
  const { generateNovelAiImages } = requireModule(modules, "novelAi");
  const reference = { imageBase64: REFERENCE_BASE64, strength: 0.7, fidelity: 0.75, styleAware: true };
  const state = createState(seedState, {
    novelAi: {
      enabled: true,
      requestMode: "proxy",
      proxyUrl: "http://novelai.stub.invalid/novelai/generate-image",
      apiKey: "pst-eval-token",
      modelPreset: "NAID4.5F",
      vibeTransferEnabled: true,
      vibeTransferReferences: [ENCODED_VIBE]
    }
  });
  const pngBytes = Buffer.from(PNG_BASE64, "base64");
  const captured = [];
  globalThis.fetch = async (url, init) => {
    captured.push({ url: String(url), headers: init?.headers ?? {}, body: JSON.parse(String(init?.body ?? "{}")) });
    return new Response(pngBytes, { status: 200, headers: { "content-type": "image/png" } });
  };

  const result = await generateNovelAiImages({
    state,
    prompt: "1girl, smile, bedroom",
    negativePrompt: "lowres",
    cue: createCue(),
    count: 1,
    characterReference: reference
  });
  const sent = captured[0]?.body?.parameters ?? {};
  const sentArrays = [
    sent.director_reference_images,
    sent.director_reference_descriptions,
    sent.director_reference_information_extracted,
    sent.director_reference_strength_values,
    sent.director_reference_secondary_strength_values
  ];

  assertCheck(
    "nai.request",
    captured.length === 1 && captured[0].url === "http://novelai.stub.invalid/novelai/generate-image" && captured[0].headers.authorization === "Bearer pst-eval-token",
    "Real mode posts once to the proxy URL with the bearer token."
  );
  assertCheck(
    "nai.request",
    sent.director_reference_images?.[0] === REFERENCE_BASE64,
    "The request body actually sent carries the full reference base64."
  );
  assertCheck(
    "nai.request",
    sentArrays.every((array) => Array.isArray(array) && array.length === 1) && sent.reference_image_multiple === undefined,
    "The sent body has five equal-length director arrays and no vibe arrays."
  );
  assertCheck(
    "nai.request",
    result.payload.parameters?.director_reference_images?.[0] === `<omitted:${REFERENCE_BYTES} bytes>` &&
      !JSON.stringify(result.payload).includes(REFERENCE_BASE64),
    "The payload returned after a real request omits the base64."
  );
  assertCheck(
    "nai.request",
    result.dataUrls.length === 1 && result.dataUrls[0] === `data:image/png;base64,${PNG_BASE64}`,
    "A PNG response decodes to a data URL in Node (no FileReader)."
  );

  const zip = zipSync({ "image_0.png": new Uint8Array(pngBytes) });
  globalThis.fetch = async () => new Response(zip, { status: 200, headers: { "content-type": "application/x-zip-compressed" } });
  const zipped = await generateNovelAiImages({ state, prompt: "1girl", negativePrompt: "lowres", cue: createCue(), count: 1 });
  assertCheck(
    "nai.request",
    zipped.dataUrls.length === 1 && zipped.dataUrls[0] === `data:image/png;base64,${PNG_BASE64}`,
    "A zipped NovelAI response still unpacks to PNG data URLs."
  );
}

async function evaluateEncodeVibeRequest(modules) {
  const { seedState } = requireModule(modules, "seed");
  const { encodeNovelAiVibe } = requireModule(modules, "novelAi");
  const state = createState(seedState, {
    novelAi: {
      enabled: true,
      requestMode: "proxy",
      proxyUrl: "http://novelai.stub.invalid/novelai/generate-image",
      apiKey: "pst-eval-token",
      modelPreset: "NAID4.5F"
    }
  });
  const encodedBytes = new Uint8Array([1, 2, 3, 250, 251, 252, 0, 128]);
  let captured;
  globalThis.fetch = async (url, init) => {
    captured = { url: String(url), body: JSON.parse(String(init?.body ?? "{}")) };
    return new Response(encodedBytes, { status: 200, headers: { "content-type": "application/octet-stream" } });
  };

  const encoded = await encodeNovelAiVibe({ state, image: "data:image/png;base64,QUJD", informationExtracted: 1.4 });
  assertCheck(
    "nai.encode-vibe",
    captured?.body?.information_extracted === 1 && !("informationExtracted" in (captured?.body ?? {})),
    "encode-vibe sends information_extracted (snake_case, clamped to 1), never informationExtracted."
  );
  assertCheck(
    "nai.encode-vibe",
    captured?.body?.image === "QUJD" && captured.body.model === "nai-diffusion-4-5-full",
    "encode-vibe sends the stripped image and the model name."
  );
  assertCheck(
    "nai.encode-vibe",
    captured?.url === "http://novelai.stub.invalid/novelai/encode-vibe",
    "encode-vibe derives its proxy URL from the generate-image proxy URL."
  );
  assertCheck(
    "nai.encode-vibe",
    encoded === Buffer.from(encodedBytes).toString("base64"),
    "The encoded vibe comes back as base64 in Node (no FileReader)."
  );
}

// ---------------------------------------------------------------------------------------------------------
// imageReferences: canvas choice, object fetch, letterboxing (with a recording OffscreenCanvas stand-in).
// ---------------------------------------------------------------------------------------------------------
async function evaluateImageReferences(modules) {
  const { pickNovelAiReferenceCanvas, fetchObjectAsBase64, letterboxForNovelAiReference, bytesToBase64 } = requireModule(modules, "references");

  const sizeOf = (width, height) => {
    const canvas = pickNovelAiReferenceCanvas(width, height);
    return `${canvas.width}x${canvas.height}`;
  };
  assertCheck(
    "references",
    sizeOf(832, 1216) === "1024x1536" && sizeOf(1216, 832) === "1536x1024" && sizeOf(1024, 1024) === "1472x1472",
    "Portrait, landscape and square images map to 1024x1536, 1536x1024 and 1472x1472."
  );
  assertCheck(
    "references",
    sizeOf(900, 1000) === "1472x1472" && sizeOf(1000, 1600) === "1024x1536" && sizeOf(1600, 900) === "1536x1024",
    "The closest aspect ratio wins (9:10 → square, 5:8 → portrait, 16:9 → landscape)."
  );
  assertCheck(
    "references",
    [sizeOf(0, 0), sizeOf(Number.NaN, 5), sizeOf(-1, 10)].every((size) => size === "1472x1472"),
    "Unusable sizes fall back to the square canvas."
  );

  const randomBytes = new Uint8Array(100_003).map((_, index) => (index * 7919 + 13) % 256);
  assertCheck(
    "references",
    bytesToBase64(randomBytes) === Buffer.from(randomBytes).toString("base64") && bytesToBase64(new Uint8Array()) === "",
    "bytesToBase64 matches Node's encoder across chunk boundaries."
  );

  const objectBytes = Buffer.from(PNG_BASE64, "base64");
  const requested = [];
  let respond = () => new Response(objectBytes, { status: 200, headers: { "content-type": "image/png" } });
  globalThis.fetch = async (url) => {
    requested.push(String(url));
    return respond();
  };

  const fetched = await fetchObjectAsBase64("sim_eval/profile_a.png");
  assertCheck(
    "references",
    fetched === PNG_BASE64 && requested[0] === "http://127.0.0.1:8788/objects/sim_eval/profile_a.png",
    "fetchObjectAsBase64 GETs <API base>/objects/<key> (segments encoded, slash kept) and returns base64."
  );

  requested.length = 0;
  const unsafeKeys = ["../simulations", "sim_eval/../../secrets.json", "sim_eval/profile a.png", "sim_eval//x.png", "", "sim_eval\\x.png"];
  const unsafeResults = [];
  for (const key of unsafeKeys) {
    unsafeResults.push(await fetchObjectAsBase64(key));
  }
  assertCheck(
    "references",
    unsafeResults.every((value) => value === undefined) && requested.length === 0,
    "Keys that could climb out of /objects/ (.., empty or odd segments) are refused without a request."
  );

  respond = () => new Response(JSON.stringify({ error: "Object not found" }), { status: 404, headers: { "content-type": "application/json" } });
  const missing = await fetchObjectAsBase64("sim_eval/missing.png");
  respond = () => new Response("<html></html>", { status: 200, headers: { "content-type": "text/html" } });
  const notImage = await fetchObjectAsBase64("sim_eval/profile_a.png");
  respond = () => new Response(new Uint8Array(), { status: 200, headers: { "content-type": "image/png" } });
  const emptyBody = await fetchObjectAsBase64("sim_eval/profile_a.png");
  respond = () => {
    throw new TypeError("fetch failed");
  };
  const offline = await captureError(() => fetchObjectAsBase64("sim_eval/profile_a.png"));
  assertCheck(
    "references",
    missing === undefined && notImage === undefined && emptyBody === undefined && offline === undefined,
    "404, a non-image body, an empty body and a network failure all resolve to undefined (never throw)."
  );

  restoreOffscreenCanvas();
  if (!hadOffscreenCanvas) {
    const outcome = await captureError(() => letterboxForNovelAiReference({ width: 832, height: 1216 }, 832, 1216));
    assertCheck(
      "references",
      outcome instanceof Error && /브라우저/u.test(outcome.message),
      "letterboxForNovelAiReference rejects (does not throw synchronously) where no canvas exists."
    );
  }

  const portraitLog = installRecordingOffscreenCanvas();
  const portraitBlob = await letterboxForNovelAiReference({ width: 832, height: 1216 }, 832, 1216);
  const fillIndex = portraitLog.calls.findIndex(([name]) => name === "fillRect");
  const drawIndex = portraitLog.calls.findIndex(([name]) => name === "drawImage");
  const drawCall = portraitLog.calls[drawIndex] ?? [];
  assertCheck(
    "references",
    portraitLog.canvas?.width === 1024 && portraitLog.canvas?.height === 1536 && portraitLog.contextOptions?.alpha === false,
    "Letterboxing a portrait image uses an opaque 1024x1536 canvas."
  );
  assertCheck(
    "references",
    portraitLog.calls.some(([name, value]) => name === "fillStyle" && value === "#000000") &&
      JSON.stringify(portraitLog.calls[fillIndex]) === JSON.stringify(["fillRect", 0, 0, 1024, 1536]) &&
      fillIndex < drawIndex,
    "The whole canvas is filled black before the image is drawn (opaque black padding)."
  );
  assertCheck(
    "references",
    JSON.stringify(drawCall.slice(2)) === JSON.stringify([0, 19, 1024, 1497]),
    `The image is scaled to fit and centred (drawImage at ${JSON.stringify(drawCall.slice(2))}).`
  );
  assertCheck(
    "references",
    portraitBlob instanceof Blob && portraitBlob.type === "image/png" && portraitLog.blobOptions?.type === "image/png",
    "Letterboxing returns a PNG Blob."
  );

  const landscapeLog = installRecordingOffscreenCanvas();
  await letterboxForNovelAiReference({ naturalWidth: 1600, naturalHeight: 900 }, 0, 0);
  const landscapeDraw = landscapeLog.calls.find(([name]) => name === "drawImage") ?? [];
  assertCheck(
    "references",
    landscapeLog.canvas?.width === 1536 &&
      landscapeLog.canvas?.height === 1024 &&
      JSON.stringify(landscapeDraw.slice(2)) === JSON.stringify([0, 80, 1536, 864]),
    "Without explicit sizes the image's natural size is used (16:9 → 1536x1024, letterboxed top and bottom)."
  );
  restoreOffscreenCanvas();
}

// ---------------------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------------------
async function loadModules(paths) {
  const modules = {};
  const failed = [];
  for (const [key, modulePath] of Object.entries(paths)) {
    try {
      modules[key] = await vite.ssrLoadModule(modulePath);
    } catch (error) {
      failed.push(`${modulePath}: ${error?.message ?? error}`);
    }
  }
  assertCheck(
    "load",
    failed.length === 0,
    failed.length === 0 ? "Every module under test loads in Node through vite.ssrLoadModule." : `Modules failed to load: ${failed.join(" | ")}`
  );
  return modules;
}

function requireModule(modules, key) {
  if (!modules[key]) {
    throw new Error(`module '${key}' is not available (see the load check)`);
  }
  return modules[key];
}

async function runGroup(scope, evaluate) {
  try {
    await evaluate();
  } catch (error) {
    assertCheck(scope, false, `threw before finishing: ${error?.stack ?? error}`);
  } finally {
    globalThis.fetch = originalFetch;
    restoreOffscreenCanvas();
  }
}

function assertCheck(scope, condition, message) {
  checks.push({ scope, level: condition ? "pass" : "fail", message });
  return condition;
}

function createState(seedState, overrides = {}) {
  const base = structuredClone(seedState);
  return {
    ...base,
    imageProfile: { ...base.imageProfile, ...(overrides.imageProfile ?? {}) },
    novelAi: { ...base.novelAi, ...(overrides.novelAi ?? {}) },
    photoreal: { ...PHOTOREAL_DEFAULTS, ...(base.photoreal ?? {}), ...(overrides.photoreal ?? {}) }
  };
}

function createRelayState(seedState, relay, photoreal = {}) {
  return createState(seedState, {
    photoreal: {
      enabled: true,
      requestMode: "proxy",
      proxyUrl: `${relay.url}/photoreal/comfyui`,
      baseUrl: "http://127.0.0.1:8188",
      timeoutSeconds: 120,
      ...photoreal
    }
  });
}

function createCue(overrides = {}) {
  return {
    shouldGenerate: true,
    reason: "photoreal client eval",
    characters: [],
    tags: ["1girl", "smile", "mirror selfie"],
    scene: "bedroom mirror",
    kind: "scene",
    ...overrides
  };
}

function installForbiddenFetch() {
  const probe = { calls: 0 };
  globalThis.fetch = async () => {
    probe.calls += 1;
    throw new Error("this path must not touch the network");
  };
  return probe;
}

function installRecordingOffscreenCanvas() {
  const log = { calls: [] };
  globalThis.OffscreenCanvas = class RecordingOffscreenCanvas {
    constructor(width, height) {
      this.width = width;
      this.height = height;
      log.canvas = { width, height };
    }

    getContext(kind, options) {
      log.contextKind = kind;
      log.contextOptions = options;
      return {
        set fillStyle(value) {
          log.calls.push(["fillStyle", value]);
        },
        get fillStyle() {
          return "#000000";
        },
        imageSmoothingEnabled: false,
        imageSmoothingQuality: "low",
        fillRect: (...args) => log.calls.push(["fillRect", ...args]),
        drawImage: (...args) => log.calls.push(["drawImage", ...args])
      };
    }

    async convertToBlob(options) {
      log.blobOptions = options;
      return new Blob([new Uint8Array([137, 80, 78, 71])], { type: options?.type ?? "" });
    }
  };
  return log;
}

function restoreOffscreenCanvas() {
  if (hadOffscreenCanvas) {
    globalThis.OffscreenCanvas = originalOffscreenCanvas;
  } else {
    delete globalThis.OffscreenCanvas;
  }
}

async function captureError(run) {
  try {
    return await run();
  } catch (error) {
    return error;
  }
}

function isSeed(value) {
  return Number.isInteger(value) && value >= 0 && value < SEED_LIMIT;
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sleep(durationMs) {
  return new Promise((resolve) => setTimeout(resolve, durationMs));
}

async function findClosedPort() {
  const server = http.createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

/**
 * Speaks the relay contract of spec §4 from the client's side: /generate answers with base64 images or the
 * scripted error, /status with the configured probe, /free and /interrupt with ok. Every request is recorded.
 */
async function startStubRelay() {
  const requests = [];
  let script = [];
  let statusReply = { status: 200, body: { ok: true } };
  let freeReply = { status: 200, body: { ok: true } };
  let generateWaiters = [];
  let disconnects = 0;

  const server = http.createServer(async (request, response) => {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    const raw = await readRequestBody(request);
    let body;
    try {
      body = raw ? JSON.parse(raw) : undefined;
    } catch {
      body = raw;
    }
    const record = { method: request.method, path: url.pathname, query: Object.fromEntries(url.searchParams), headers: request.headers, body };
    requests.push(record);

    if (request.method === "POST" && url.pathname === "/photoreal/comfyui/generate") {
      const step = script.length > 0 ? script.shift() : "ok";
      const waiters = generateWaiters;
      generateWaiters = [];
      waiters.forEach((resolve) => resolve(record));
      if (step === "hang") {
        response.on("close", () => {
          if (!response.writableFinished) {
            disconnects += 1;
          }
        });
        return;
      }
      if (step === "ok") {
        sendReply(response, {
          status: 200,
          body: {
            promptId: "prompt_eval_1",
            images: [{ filename: "DynamicChat_00001_.png", subfolder: "DynamicChat", type: "output", mimeType: "image/png", base64: PNG_BASE64 }],
            elapsedMs: 1234
          }
        });
        return;
      }
      sendReply(response, step);
      return;
    }
    if (request.method === "GET" && url.pathname === "/photoreal/comfyui/status") {
      sendReply(response, statusReply);
      return;
    }
    if (request.method === "POST" && url.pathname === "/photoreal/comfyui/free") {
      sendReply(response, freeReply);
      return;
    }
    if (request.method === "POST" && url.pathname === "/photoreal/comfyui/interrupt") {
      sendReply(response, { status: 200, body: { ok: true } });
      return;
    }
    sendReply(response, { status: 404, body: { error: "Not found" } });
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();

  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    reset() {
      requests.length = 0;
      script = [];
      disconnects = 0;
      statusReply = { status: 200, body: { ok: true } };
      freeReply = { status: 200, body: { ok: true } };
    },
    script(...steps) {
      script = steps;
    },
    setStatus(reply) {
      statusReply = reply;
    },
    setFree(reply) {
      freeReply = reply;
    },
    requestsTo(pathname) {
      return requests.filter((record) => record.path === pathname);
    },
    nextGenerate() {
      return new Promise((resolve) => generateWaiters.push(resolve));
    },
    async waitForDisconnect(timeoutMs) {
      const deadline = Date.now() + timeoutMs;
      while (disconnects === 0 && Date.now() < deadline) {
        await sleep(25);
      }
      return disconnects > 0;
    },
    async close() {
      server.closeAllConnections?.();
      await new Promise((resolve) => server.close(resolve));
    }
  };
}

function sendReply(response, reply) {
  if (reply.raw !== undefined) {
    response.writeHead(reply.status, { "content-type": "text/html" });
    response.end(reply.raw);
    return;
  }
  response.writeHead(reply.status, { "content-type": "application/json" });
  response.end(JSON.stringify(reply.body ?? {}));
}

function readRequestBody(request) {
  return new Promise((resolve) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    request.on("error", () => resolve(""));
  });
}
