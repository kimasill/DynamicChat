import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// Photoreal prompt builder + ComfyUI workflow builder.
//
// Both modules are pure, so everything here runs offline: no ComfyUI, no model files, no keys. The workflow
// half re-checks every built graph with a validator written independently of validateComfyApiWorkflow, because
// a graph that only agrees with its own validator proves nothing — the point is that a live ComfyUI would accept
// it (links resolve, numbers are numbers, every placeholder resolved, each reference loaded exactly once).

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const vite = await createServer({
  root: workspaceRoot,
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom"
});

const checks = [];
const HANGUL = /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/u;
// Declared before the run below: the evaluate* groups execute during module evaluation.
const COMFY_FILENAME_TOKENS = new Set(["width", "height", "year", "month", "day", "hour", "minute", "second"]);
const EMPTY_STYLE = { lead: "", tail: "", negative: "" };
const SHOT_REGIONS = {
  close_up: ["head"],
  face_focus: ["head"],
  upper_body: ["head", "torso"],
  cowboy_shot: ["head", "torso", "hips", "legs"],
  full_body: ["head", "torso", "hips", "legs", "feet"],
  wide_shot: ["head", "torso", "hips", "legs", "feet"]
};
const NUMERIC_INPUTS = new Set(["seed", "noise_seed", "width", "height", "steps", "cfg", "batch_size", "denoise", "shift", "resolution", "strength"]);
const REF_PATTERN = /^%ref:([A-Za-z0-9_.-]{1,64})%$/u;
const QWEN21_TURBO_TYPES = ["UNETLoader", "QwenImage21Cache", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21", "EmptyLatentImage", "CFGGuider", "KSamplerSelect", "RandomNoise", "ManualSigmas", "SamplerCustomAdvanced", "VAEDecode", "PreviewImage"];
const EDIT_TYPES = ["UNETLoader", "ModelSamplingAuraFlow", "CFGNorm", "CLIPLoader", "VAELoader", "TextEncodeQwenImageEditPlus", "FluxKontextMultiReferenceLatentMethod", "KSampler", "VAEDecode", "PreviewImage"];

try {
  let promptModule;
  let comfyModule;
  try {
    promptModule = await vite.ssrLoadModule("/src/services/photorealPrompt.ts");
    comfyModule = await vite.ssrLoadModule("/src/services/comfyWorkflows.ts");
  } catch (error) {
    assertCheck("load", false, `photorealPrompt.ts / comfyWorkflows.ts load: ${describeError(error)}`);
  }

  if (promptModule) {
    runGroup("prompt.nai-tokens", () => evaluateNaiTokenDropping(promptModule));
    runGroup("prompt.photo-sanitize", () => evaluatePhotoSanitizing(promptModule));
    runGroup("prompt.korean", () => evaluateKoreanDropping(promptModule));
    runGroup("prompt.scripts", () => evaluateScriptsAndCanonicalText(promptModule));
    runGroup("prompt.adult-nouns", () => evaluateAdultNouns(promptModule));
    runGroup("prompt.appearance", () => evaluateAppearanceReplacesIdentity(promptModule));
    runGroup("prompt.camera", () => evaluateCameraSentences(promptModule));
    runGroup("prompt.garments", () => evaluateGarmentsAsGiven(promptModule));
    runGroup("prompt.framed-garments", () => evaluateFramedGarmentsAndPhrases(promptModule));
    runGroup("prompt.multi-person", () => evaluateMultiPerson(promptModule));
    runGroup("prompt.references", () => evaluateReferences(promptModule));
    runGroup("prompt.style", () => evaluateStylePresets(promptModule));
    runGroup("prompt.look", () => evaluateLookAndBackground(promptModule));
    runGroup("prompt.negative", () => evaluateNegativeMerge(promptModule));
    runGroup("prompt.phrasing", () => evaluateNaturalPhrasing(promptModule));
    runGroup("prompt.length", () => evaluatePathologicalInput(promptModule));
  }

  if (comfyModule) {
    runGroup("comfy.presets", () => evaluatePresetMetadata(comfyModule));
    runGroup("comfy.static-validator", () => evaluateStaticValidatorSelfTest(comfyModule));
    runGroup("comfy.z-image", () => evaluateZImageTurboGraph(comfyModule));
    runGroup("comfy.qwen21-turbo", () => evaluateQwen21TurboGraphs(comfyModule));
    runGroup("comfy.qwen21", () => evaluateQwen21BaseGraph(comfyModule));
    runGroup("comfy.edit-2511", () => evaluateQwenEdit2511Graphs(comfyModule));
    runGroup("comfy.custom", () => evaluateCustomWorkflow(comfyModule));
    runGroup("comfy.substitute", () => evaluatePlaceholderSubstitution(comfyModule));
    runGroup("comfy.validate", () => evaluateValidatorRejections(comfyModule));
    runGroup("comfy.model-files", () => evaluateModelFiles(comfyModule));
    runGroup("comfy.output-size", () => evaluateOutputSize(comfyModule));
    runGroup("comfy.preset-for-cut", () => evaluatePresetForCut(comfyModule));
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
console.log(`Photoreal prompt checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

// ---------------------------------------------------------------------------------------------------------------
// photorealPrompt.ts
// ---------------------------------------------------------------------------------------------------------------

function dailyLook(promptModule) {
  return promptModule.resolvePhotorealStyle({ stylePreset: "daily_look", stylePrompt: "", negativePrompt: "" });
}

function build(promptModule, input) {
  return promptModule.buildPhotorealPrompt({
    characters: [],
    contextTags: [],
    style: dailyLook(promptModule),
    referenceSyntax: "none",
    ...input
  });
}

function evaluateNaiTokenDropping(promptModule) {
  const plan = build(promptModule, {
    characters: [
      {
        tags: [
          "1girl",
          "masterpiece",
          "best quality",
          "amazing quality",
          "very aesthetic",
          "absurdres",
          "newest",
          "artist:x",
          "year 2024",
          "rating:general",
          "score_9",
          "score_8_up",
          "1.5::smile::",
          "{{white blouse}}",
          "[[pleated skirt]]",
          "anime screencap",
          "illustration",
          "official art",
          "by wlop"
        ],
        identityTags: ["1girl"]
      }
    ],
    contextTags: ["cafe", "2::window::"]
  });
  const { prompt, droppedTags } = plan;
  assertCheck("prompt.nai-tokens", !/masterpiece|best quality|amazing quality|very aesthetic|absurdres|newest/iu.test(prompt), "Quality/aesthetic tokens never reach the photo prompt.");
  assertCheck("prompt.nai-tokens", !/artist|wlop/iu.test(prompt), "artist:x and `by <artist>` tags are dropped.");
  assertCheck("prompt.nai-tokens", !/2024|\byear\b/iu.test(prompt), "`year 2024` is dropped.");
  assertCheck("prompt.nai-tokens", !/rating|\bgeneral\b/iu.test(prompt), "`rating:general` is dropped.");
  assertCheck("prompt.nai-tokens", !/score/iu.test(prompt), "score_9 / score_8_up are dropped.");
  assertCheck("prompt.nai-tokens", !/\banime\b|illustration|official art/iu.test(prompt), "Anime/illustration style tokens are dropped from the positive prompt.");
  assertCheck("prompt.nai-tokens", !/::|[{}[\]]/u.test(prompt), "No NovelAI weight syntax (::, {}, []) survives.");
  assertCheck("prompt.nai-tokens", /\bsmiling\b/u.test(prompt), "A weighted tag keeps its content: 1.5::smile:: renders as \"smiling\".");
  assertCheck("prompt.nai-tokens", /white blouse/u.test(prompt) && /pleated skirt/u.test(prompt), "{{tag}} / [[tag]] emphasis keeps the tag.");
  assertCheck("prompt.nai-tokens", /\bwindow\b/u.test(plan.settingSentence), "A weighted context tag (2::window::) keeps its content in the setting.");
  const expectedDropped = ["masterpiece", "best quality", "artist:x", "year 2024", "rating:general", "score_9", "anime screencap", "by wlop"];
  const missing = expectedDropped.filter((tag) => !droppedTags.includes(tag));
  assertCheck("prompt.nai-tokens", missing.length === 0, `droppedTags lists the removed NAI-only tokens (missing: ${missing.join(", ") || "none"}).`);
  assertCheck("prompt.nai-tokens", !droppedTags.includes("1.5::smile::") && !droppedTags.includes("{{white blouse}}"), "Weight syntax whose tag survived is not reported as dropped.");

  const noText = promptModule.buildPhotorealPrompt({
    characters: [{ tags: ["1girl", "no text", "no watermark"], identityTags: [] }],
    contextTags: [],
    style: EMPTY_STYLE,
    referenceSyntax: "none"
  });
  assertCheck("prompt.nai-tokens", !/no text|watermark/iu.test(noText.prompt), "\"no text\" / \"no watermark\" never appear in the positive prompt.");
  assertCheck("prompt.nai-tokens", splitList(noText.negativePrompt).includes("text") && splitList(noText.negativePrompt).includes("watermark"), "\"no text\" stays as a negative: \"text\" and \"watermark\" join the negative prompt.");
}

function evaluateKoreanDropping(promptModule) {
  const plan = build(promptModule, {
    characters: [{ appearance: "검은 머리의 20대 여성", tags: ["1girl", "웃는 얼굴", "교복", "smile"], identityTags: ["1girl"] }],
    contextTags: ["카페", "night"]
  });
  assertCheck("prompt.korean", !HANGUL.test(plan.prompt), "Korean tags never reach the prompt.");
  assertCheck("prompt.korean", ["웃는 얼굴", "교복", "카페"].every((tag) => plan.droppedTags.includes(tag)), "Hangul tags are reported in droppedTags.");
  assertCheck("prompt.korean", plan.droppedTags.includes("검은 머리의 20대 여성") && /an adult woman/iu.test(plan.prompt), "A Korean appearance text is dropped and the subject falls back to the adult noun.");
  assertCheck("prompt.korean", /\bsmiling\b/u.test(plan.prompt) && /at night/u.test(plan.prompt), "English tags next to Korean ones survive.");
  const style = promptModule.resolvePhotorealStyle({ stylePreset: "daily_look", stylePrompt: "필름 느낌, soft film grain", negativePrompt: "흐림, blurry" });
  assertCheck("prompt.korean", !HANGUL.test(style.tail) && /soft film grain$/u.test(style.tail), "Korean segments of the style text are dropped; English ones are appended.");
  assertCheck("prompt.korean", !HANGUL.test(style.negative) && splitList(style.negative).includes("blurry"), "Korean segments of the negative text are dropped.");
}

/**
 * Review finding safety:no-unicode-normalization: the builder reads canonical text (fullwidth, zero-width and
 * accented spellings become the plain word), drops every token in a script other than Latin — tags, appearance and
 * style segments alike — and keeps NovelAI's undressing idiom out of the photo negative.
 */
function evaluateScriptsAndCanonicalText(promptModule) {
  const scope = "prompt.scripts";
  const plan = build(promptModule, {
    characters: [{ tags: ["1girl", "ｗｈｉｔｅ ｂｌｏｕｓｅ", "smi\u200ble", "ロリ", "n\u0430ked", "一个十四岁的韩国少女", "\u03bdude", "café latte"], identityTags: ["1girl"] }],
    contextTags: ["ヌード", "cafe"]
  });
  assertCheck(scope, /wearing a white blouse/u.test(plan.prompt) && /\bsmiling\b/u.test(plan.prompt), `Fullwidth and zero-width-split tags render as the plain words (${plan.subjectSentences[0]}).`);
  assertCheck(scope, !/[^\x00-\x7F]/u.test(plan.prompt), `Nothing outside ASCII reaches the photo prompt: kana, Cyrillic and Greek look-alikes, CJK are dropped, accents folded (${plan.prompt.slice(0, 200)}).`);
  assertCheck(
    scope,
    ["ロリ", "n\u0430ked", "一个十四岁的韩国少女", "\u03bdude", "ヌード"].every((tag) => plan.droppedTags.includes(tag)) && /cafe latte/u.test(plan.prompt),
    `Dropped scripts are reported in droppedTags; "café latte" survives as "cafe latte" (${plan.droppedTags.join(", ")}).`
  );

  const mixed = build(promptModule, {
    characters: [{ appearance: "A Korean woman in her late twenties, 一个十四岁的韩国少女, in a sheer orange top", tags: ["1girl", "smile"], identityTags: ["1girl"] }]
  });
  assertCheck(
    scope,
    mixed.prompt.includes("A Korean woman in her late twenties, in an orange top") && !/一个|sheer/u.test(mixed.prompt) && mixed.droppedTags.includes("一个十四岁的韩国少女"),
    `An appearance loses only its unreadable segment, and a sheer qualifier in prose loses the word (${mixed.subjectSentences[0]}).`
  );
  const koreanLead = build(promptModule, {
    // As composed: the saved identity is in the cut's tags too, and is rendered only when no appearance replaces it.
    characters: [{ appearance: "검은 머리의 20대 여성, long black hair", tags: ["1girl", "short brown hair", "smile"], identityTags: ["1girl", "short brown hair"] }]
  });
  assertCheck(
    scope,
    /^An adult woman with short brown hair, smiling\.$/u.test(koreanLead.subjectSentences[0] ?? ""),
    `An appearance whose subject segment is dropped falls back to the identity tags as a whole, never to a subjectless list (${koreanLead.subjectSentences[0]}).`
  );
  const traits = build(promptModule, {
    characters: [{ appearance: "A tall Korean woman, futanari, emaciated frame", tags: ["1girl"], identityTags: ["1girl"] }]
  });
  assertCheck(scope, /A tall Korean woman/u.test(traits.prompt) && !/futanari|emaciated/u.test(traits.prompt), `Body-identity traits and eating-disorder builds are dropped from an appearance as from tags (${traits.subjectSentences[0]}).`);

  const style = promptModule.resolvePhotorealStyle({ stylePreset: "custom", stylePrompt: "A moody 35mm photograph. Teal grade, ヌード, heavy grain, ｆｉｌｍ ｌｏｏｋ.", negativePrompt: "" });
  assertCheck(scope, style.lead === "A moody 35mm photograph" && style.tail === "Teal grade, heavy grain, film look", `Style segments in other scripts are dropped; fullwidth text is canonical (tail "${style.tail}").`);

  const undress = promptModule.buildPhotorealPrompt({
    characters: [{ tags: ["1girl", "-2::clothing::", "-1::shirt, skirt::", "-1::hat::", "smile"], identityTags: [] }],
    contextTags: [],
    style: { lead: "", tail: "", negative: "blurry" },
    extraNegative: "clothes, glasses, ｃｌｏｔｈｉｎｇ, pants, ヌード",
    referenceSyntax: "none"
  });
  const negatives = splitList(undress.negativePrompt);
  assertCheck(
    scope,
    negatives.includes("hat") && negatives.includes("glasses") && negatives.includes("blurry") && !negatives.some((term) => /cloth|shirt|skirt|pants|ヌード/u.test(term)),
    `Body-covering clothing never becomes a photo negative, from a negative weight or a typed negative; accessories still do (${undress.negativePrompt}).`
  );
  assertCheck(scope, !/clothing|shirt|skirt|\bhat\b/u.test(undress.prompt), "Negatively weighted tags are still kept out of the positive prompt.");

  const screened = promptModule.selectPhotoTextForScreening("필름 느낌, soft film grain. ヌード, ｎｕｄｅ");
  assertCheck(
    scope,
    JSON.stringify(screened) === JSON.stringify(["soft film grain", "ヌード", "nude"]),
    `The screen reads every segment but the Korean ones, canonical (${JSON.stringify(screened)}).`
  );
  const mixedKorean = promptModule.selectPhotoTextForScreening("긴 머리의 nude 여성, A naked woman 〮");
  assertCheck(
    scope,
    mixedKorean.includes("nude") && mixedKorean.some((segment) => /^A naked woman\b/u.test(segment)) && !mixedKorean.some((segment) => /[가-힯〮〯]/u.test(segment)),
    `A segment that mixes Hangul with Latin words is screened without the Hangul, never skipped (${JSON.stringify(mixedKorean)}).`
  );
  const presetTexts = promptModule.describePhotorealStyleForScreening({ stylePreset: "studio_editorial", stylePrompt: "warm tones" });
  assertCheck(
    scope,
    presetTexts.includes("A high-end fashion editorial photograph") && presetTexts.includes("a seamless studio backdrop") && presetTexts.includes("warm tones"),
    `The style screen covers the preset's lead, tail and ambient defaults plus the creator's text (${presetTexts.length} segments).`
  );
}

function evaluateAdultNouns(promptModule) {
  const subject = (input) => build(promptModule, input).subjectSentences.join(" ");
  const oneGirl = subject({ characters: [{ tags: ["1girl", "smile"], identityTags: ["1girl"] }] });
  assertCheck("prompt.adult-nouns", /^An adult woman\b/u.test(oneGirl), `1girl renders as "an adult woman" (${oneGirl}).`);
  const oneBoy = subject({ characters: [{ tags: ["1boy"], identityTags: [] }] });
  assertCheck("prompt.adult-nouns", /^An adult man\b/u.test(oneBoy), `1boy renders as "an adult man" (${oneBoy}).`);
  const v4Caption = subject({ characters: [{ tags: ["girl", "smile"], identityTags: [] }] });
  assertCheck("prompt.adult-nouns", /^An adult woman\b/u.test(v4Caption), "A NovelAI V4 char caption's bare \"girl\" is an adult woman too.");
  const twoGirls = subject({ contextTags: ["2girls", "cafe"] });
  assertCheck("prompt.adult-nouns", /^Two adult women\b/u.test(twoGirls), `2girls renders as "two adult women" (${twoGirls}).`);
  const mixed = subject({ contextTags: ["1girl", "1boy", "cafe"] });
  assertCheck("prompt.adult-nouns", /^An adult woman and an adult man\b/u.test(mixed), `1girl + 1boy renders as "an adult woman and an adult man" (${mixed}).`);
  const several = subject({ contextTags: ["multiple girls", "street"] });
  assertCheck("prompt.adult-nouns", /^Several adult women\b/u.test(several), `"multiple girls" renders as "several adult women" (${several}).`);
  const mature = subject({ characters: [{ tags: ["mature female"], identityTags: [] }] });
  assertCheck("prompt.adult-nouns", /^A mature woman\b/u.test(mature), "\"mature female\" keeps its age as \"a mature woman\".");
  const all = [oneGirl, oneBoy, v4Caption, twoGirls, mixed, several].join(" ");
  assertCheck("prompt.adult-nouns", !/\b(?:girls?|boys?)\b/iu.test(all), "No subject is ever called a girl or a boy.");

  const coded = build(promptModule, {
    characters: [{ tags: ["1girl", "schoolgirl", "child", "age-appropriate teen idol trainee", "school uniform"], identityTags: [] }]
  });
  assertCheck("prompt.adult-nouns", !/schoolgirl|child|teen/iu.test(coded.prompt), "Minor- and student-coded words are dropped from the photo prompt.");
  assertCheck("prompt.adult-nouns", /an adult woman/iu.test(coded.prompt) && /school uniform/u.test(coded.prompt), "The subject stays an adult woman; the school uniform is kept as clothing.");
  assertCheck("prompt.adult-nouns", ["schoolgirl", "child", "age-appropriate teen idol trainee"].every((tag) => coded.droppedTags.includes(tag)), "Dropped minor/student words are reported in droppedTags.");

  // Prose gets the same scrub as tags: an appearance rendered "lolita, flat-chested, childish body" and "a high
  // school student" verbatim before.
  const prose = build(promptModule, {
    characters: [{ appearance: "A Korean woman with long black hair, a high school student, lolita, childish body, 12 years young, in a beige coat", tags: ["1girl", "smile"], identityTags: ["1girl"] }]
  });
  assertCheck(
    "prompt.adult-nouns",
    /A Korean woman with long black hair, in a beige coat/u.test(prose.prompt) && !/student|lolita|childish|years young/iu.test(prose.prompt),
    `Minor- and student-coded segments of an appearance are dropped like tags (${prose.subjectSentences[0]}).`
  );
  const coldStart = build(promptModule, {
    characters: [{ appearance: "A Korean high school student with long black hair", tags: ["1girl", "smile"], identityTags: ["1girl"] }]
  });
  assertCheck("prompt.adult-nouns", /^An adult woman, smiling\.$/u.test(coldStart.subjectSentences[0] ?? ""), `An appearance whose subject is a student falls back to the adult noun (${coldStart.subjectSentences[0]}).`);
  const kept = build(promptModule, { characters: [{ tags: ["1girl", "elementary school teacher", "college student", "baby blue sweater", "baby carrots"], identityTags: [] }] });
  assertCheck(
    "prompt.adult-nouns",
    /elementary school teacher/u.test(kept.prompt) && /college student/u.test(kept.prompt) && /baby blue sweater/u.test(kept.prompt),
    `Adults who work with children, college students and fashion words are kept (${kept.subjectSentences[0]}).`
  );
}

function evaluateAppearanceReplacesIdentity(promptModule) {
  const identity = ["1girl", "long black hair", "brown eyes", "slim"];
  const appearance = "a Korean woman in her late twenties with a sleek chin-length bob";
  const tags = [...identity, "smile", "white blouse", "pencil skirt", "sitting", "crossed legs"];
  const plan = build(promptModule, {
    characters: [{ appearance, tags, identityTags: identity }],
    contextTags: ["long black hair", "cafe"]
  });
  assertCheck("prompt.appearance", plan.prompt.includes("A Korean woman in her late twenties with a sleek chin-length bob"), "The creator's appearance text is the subject description.");
  assertCheck("prompt.appearance", !/long black hair|brown eyes|\bslim\b/u.test(plan.prompt), "Registered identity tags are removed when an appearance is given (also from the shared caption).");
  assertCheck("prompt.appearance", !/adult woman/u.test(plan.prompt), "No generic adult noun is added next to the creator's own description.");
  assertCheck("prompt.appearance", /\bsmiling\b/u.test(plan.prompt) && /\bsitting\b/u.test(plan.prompt) && /legs crossed/u.test(plan.prompt), "Expression and pose tags of the cut remain.");
  assertCheck("prompt.appearance", /wearing a white blouse and a pencil skirt/u.test(plan.prompt), "Outfit tags of the cut remain as \"wearing ...\".");
  const control = build(promptModule, { characters: [{ tags, identityTags: identity }] });
  assertCheck("prompt.appearance", /long black hair/u.test(control.prompt) && /brown eyes/u.test(control.prompt), "Without an appearance, the identity tags describe the person.");
  const multiSentence = build(promptModule, {
    characters: [{ appearance: "A Korean woman in her late twenties. She has a slim build.", tags: ["1girl", "smile", "white blouse"], identityTags: ["1girl"] }]
  });
  assertCheck("prompt.appearance", /She has a slim build\. She is smiling, wearing a white blouse\./u.test(multiSentence.prompt), "A multi-sentence appearance stays prose and the cut continues in a new sentence.");
}

function evaluateCameraSentences(promptModule) {
  const person = [{ tags: ["1girl", "smile"], identityTags: [] }];
  const camera = (frame, extra = {}) => build(promptModule, { characters: person, frame, ...extra }).cameraSentence;
  // Crops that show the head also say the face is in view: the default look comes from daily-look references where
  // the face is routinely cropped off or hidden behind a phone, and one QA cut lost the face exactly that way.
  const shots = {
    close_up: "Tight close-up of the face, the face filling most of the frame",
    face_focus: "Head-and-shoulders portrait, with the face fully visible",
    upper_body: "Medium close-up from the waist up, with the face fully visible",
    cowboy_shot: "Medium shot from mid-thigh up, with the face fully visible",
    full_body: "Full-length shot showing the whole body from head to toe",
    wide_shot: "Wide shot with the subject small in the environment"
  };
  for (const [shot, phrase] of Object.entries(shots)) {
    const sentence = camera({ shot, viewpoint: "front", visibleRegions: SHOT_REGIONS[shot] });
    assertCheck("prompt.camera", sentence === phrase, `frame.shot ${shot} → "${phrase}" (got "${sentence}").`);
  }
  const viewpoints = { side: "in profile", from_behind: "seen from behind", pov: "first-person point of view", over_the_shoulder: "over-the-shoulder shot" };
  for (const [viewpoint, phrase] of Object.entries(viewpoints)) {
    const sentence = camera({ shot: "upper_body", viewpoint, visibleRegions: SHOT_REGIONS.upper_body });
    const face = viewpoint === "from_behind" ? "" : ", with the face fully visible";
    assertCheck("prompt.camera", sentence === `Medium close-up from the waist up${face}, ${phrase}`, `viewpoint ${viewpoint} → "${phrase}" (got "${sentence}").`);
  }
  const angles = { from_above: "high angle", from_below: "low angle", dutch_angle: "tilted dutch angle", eye_level: "eye-level" };
  for (const [angle, phrase] of Object.entries(angles)) {
    const sentence = camera({ shot: "full_body", viewpoint: "front", angle, visibleRegions: SHOT_REGIONS.full_body });
    assertCheck("prompt.camera", sentence.endsWith(`, ${phrase}`), `angle ${angle} → "${phrase}" (got "${sentence}").`);
  }
  const bodyCloseUp = camera({ shot: "close_up", viewpoint: "front", visibleRegions: ["torso"] });
  assertCheck("prompt.camera", bodyCloseUp === "Tight close-up of the upper body", `A body-region close-up is not described as a face close-up (got "${bodyCloseUp}").`);
  const inferred = camera(undefined, { contextTags: ["cowboy shot", "from side", "cafe"] });
  assertCheck("prompt.camera", /^Medium shot from mid-thigh up(?:, with the face fully visible)?, in profile$/u.test(inferred), `With no frame, the crop is inferred from composition tags (got "${inferred}").`);
  const none = build(promptModule, { characters: person, contextTags: ["cafe"] });
  assertCheck("prompt.camera", none.cameraSentence === "" && !/close-up|medium shot|full-length/iu.test(none.prompt), "No frame and no composition tags → the camera sentence is omitted.");
  const framed = build(promptModule, {
    characters: person,
    contextTags: ["upper body", "from side", "cafe"],
    frame: { shot: "upper_body", viewpoint: "side", visibleRegions: SHOT_REGIONS.upper_body }
  });
  assertCheck("prompt.camera", !/upper body|from side/u.test(framed.settingSentence) && framed.prompt.includes("Medium close-up from the waist up, with the face fully visible, in profile."), "Composition tags in the caption are expressed once, by the camera sentence.");
  // Measured: the crop stated after the people and the room lost to them; leading with it held the close-up.
  const ordered = build(promptModule, { characters: person, contextTags: ["cafe"], frame: { shot: "close_up", viewpoint: "front", visibleRegions: ["head"] } });
  const leadEnd = ordered.prompt.indexOf(". ") + 2;
  assertCheck("prompt.camera", ordered.prompt.slice(leadEnd).startsWith("Tight close-up of the face"), "The camera sentence comes right after the style lead, before the subject.");
  const behind = camera({ shot: "upper_body", viewpoint: "from_behind", visibleRegions: ["head", "torso"] });
  assertCheck("prompt.camera", !/face/u.test(behind), "A figure seen from behind is never told to show its face.");
}

function evaluateFramedGarmentsAndPhrases(promptModule) {
  // QA on the shipped adult seed: a face close-up whose caption listed 'navy skinnies' rendered waist-up on 3/3 seeds.
  const closeUp = build(promptModule, {
    characters: [{ tags: ["1girl", "sad", "ivory off shoulder sweater", "silver necklace", "navy skinnies", "white sneakers"], identityTags: [] }],
    frame: { shot: "close_up", viewpoint: "front", visibleRegions: ["head"] }
  });
  assertCheck("prompt.framed-garments", !/skinnies|sneakers/u.test(closeUp.prompt) && closeUp.droppedTags.includes("navy skinnies"), `Garments the crop cannot show are dropped from the photo prompt (${closeUp.subjectSentences[0]}).`);
  const full = build(promptModule, {
    characters: [{ tags: ["1girl", "ivory sweater", "navy skinnies", "white sneakers"], identityTags: [] }],
    frame: { shot: "full_body", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs", "feet"] }
  });
  assertCheck("prompt.framed-garments", /navy skinnies/u.test(full.prompt) && /white sneakers/u.test(full.prompt), "A full-body crop keeps every garment.");
  const action = build(promptModule, {
    characters: [{ tags: ["1girl", "trembling hands reaching toward a brown leather handle", "cold brown leather handle in trembling fingers", "ivory sweater"], identityTags: [] }],
    frame: { shot: "upper_body", viewpoint: "front", visibleRegions: ["head", "torso"] }
  });
  assertCheck("prompt.framed-garments", !/wearing [^,.]*(?:reaching|trembling)/u.test(action.prompt) && /reaching toward a brown leather handle/u.test(action.prompt) && /wearing an ivory sweater/u.test(action.prompt), `A whole action written as one tag stays an action, not something worn (${action.subjectSentences[0]}).`);
  const worn = build(promptModule, { characters: [{ tags: ["1girl", "wearing white blouse", "white blouse", "wearing jeans"], identityTags: [] }], frame: { shot: "cowboy_shot", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs"] } });
  assertCheck("prompt.framed-garments", (worn.prompt.match(/\bwearing\b/gu) ?? []).length === 1 && /wearing a white blouse and jeans/u.test(worn.prompt), `Tags that already say "wearing" are not printed twice (${worn.subjectSentences[0]}).`);
  const side = build(promptModule, { characters: [{ tags: ["1girl", "side", "smile"], identityTags: [] }], contextTags: ["side", "cafe"] });
  assertCheck("prompt.framed-garments", !/\bside\b/u.test(side.settingSentence) && !/includes side/u.test(side.prompt), "A bare viewpoint word never becomes a prop of the scene.");
}

function evaluateGarmentsAsGiven(promptModule) {
  const plan = build(promptModule, {
    characters: [{ tags: ["1girl", "white blouse", "smile"], identityTags: ["1girl"] }],
    frame: { shot: "upper_body", viewpoint: "front", visibleRegions: ["head", "torso"] }
  });
  assertCheck("prompt.garments", /wearing a white blouse/u.test(plan.prompt), "The given garment renders as \"wearing a white blouse\".");
  assertCheck("prompt.garments", !/skirt|shoes|socks|thigh|boots|sneakers|pants|jeans/iu.test(plan.prompt), "No garment is re-added that the frame-filtered plan did not carry.");
  assertCheck("prompt.garments", (plan.prompt.match(/\bwearing\b/gu) ?? []).length === 1, "Exactly one \"wearing\" clause for one garment.");
  const renamed = build(promptModule, { characters: [{ tags: ["1girl", "black thighhighs", "serafuku", "no shoes"], identityTags: [] }] });
  assertCheck("prompt.garments", /black thigh-high socks/u.test(renamed.prompt) && !/thighhighs/u.test(renamed.prompt), "NovelAI garment words are translated (thighhighs → thigh-high socks).");
  assertCheck("prompt.garments", /without shoes/u.test(renamed.prompt), "\"no shoes\" renders as \"without shoes\".");
  const underscored = build(promptModule, { characters: [{ tags: ["1girl", "white_shirt", "denim_shorts"], identityTags: [] }] });
  assertCheck("prompt.garments", /wearing a white shirt and denim shorts/u.test(underscored.prompt), "Underscores become spaces and garments join naturally.");
}

function evaluateMultiPerson(promptModule) {
  const plan = build(promptModule, {
    characters: [
      { tags: ["1girl", "smile", "white dress"], identityTags: [], center: { x: 0.7, y: 0.5 } },
      { tags: ["1boy", "black suit"], identityTags: [], center: { x: 0.3, y: 0.5 } }
    ]
  });
  assertCheck("prompt.multi-person", /On the left, an adult man[^;]*black suit; on the right, an adult woman[^.]*white dress\./u.test(plan.prompt), `Distinct centers place people left/right by center.x (${plan.subjectSentences.join(" ")}).`);
  assertCheck("prompt.multi-person", plan.subjectSentences.length === 2, "One subject clause per person.");
  const sameCenters = build(promptModule, {
    characters: [
      { tags: ["1girl", "smile"], identityTags: [], center: { x: 0.5, y: 0.5 } },
      { tags: ["1boy", "frown"], identityTags: [], center: { x: 0.5, y: 0.5 } }
    ]
  });
  assertCheck("prompt.multi-person", /First, an adult woman, smiling\. Second, an adult man, frowning\./u.test(sameCenters.prompt), "Equal centers fall back to \"First, ... Second, ...\" in cue order.");
  const noCenters = build(promptModule, {
    characters: [
      { tags: ["1boy"], identityTags: [] },
      { tags: ["1girl"], identityTags: [] }
    ]
  });
  assertCheck("prompt.multi-person", /First, an adult man\. Second, an adult woman\./u.test(noCenters.prompt), "Missing centers fall back to ordinals without inventing a layout.");
  const three = build(promptModule, {
    characters: [
      { tags: ["1girl"], identityTags: [], center: { x: 0.5, y: 0.5 } },
      { tags: ["1boy"], identityTags: [], center: { x: 0.8, y: 0.5 } },
      { tags: ["1other"], identityTags: [], center: { x: 0.2, y: 0.5 } }
    ]
  });
  assertCheck("prompt.multi-person", /On the left, an adult person; in the middle, an adult woman; on the right, an adult man\./u.test(three.prompt), "Three people read left / middle / right.");
  const group = build(promptModule, {
    characters: [
      { tags: ["1girl"], identityTags: [], center: { x: 0.3, y: 0.5 } },
      { tags: ["1girl"], identityTags: [], center: { x: 0.7, y: 0.5 } }
    ],
    contextTags: ["2girls", "school uniform", "classroom"]
  });
  assertCheck("prompt.multi-person", /Both of them are wearing school uniforms\./u.test(group.prompt), "Shared caption detail with several people becomes a group sentence, not the first person's.");
  const single = build(promptModule, { characters: [{ tags: ["1girl"], identityTags: [] }], contextTags: ["smile", "holding cup", "cafe"] });
  assertCheck("prompt.multi-person", /^An adult woman, smiling, holding a cup\.$/u.test(single.subjectSentences[0] ?? ""), `With one person, character tags in the shared caption join that person (${single.subjectSentences[0]}).`);
}

/**
 * What an anime model stylises and a photograph renders literally. Found on the shipped adult seed during QA: its
 * saved identities carry 'futanari' and 'anorexic', and a default outfit carries a 'translucent tactical bodysuit'.
 */
function evaluatePhotoSanitizing(promptModule) {
  const style = dailyLook(promptModule);
  const plan = promptModule.buildPhotorealPrompt({
    characters: [{ tags: ["1girl", "futanari", "anorexic", "slender", "smile", "translucent tactical bodysuit", "see-through", "sheer white blouse", "metal collar"], identityTags: [] }],
    contextTags: ["cafe"],
    style,
    referenceSyntax: "none"
  });
  assertCheck("prompt.photo-sanitize", !/futa|intersex|dickgirl/iu.test(plan.prompt) && plan.droppedTags.includes("futanari"), "Body-identity traits never reach the photo prompt.");
  assertCheck("prompt.photo-sanitize", !/anorexi|emaciated|skeletal/iu.test(plan.prompt) && plan.droppedTags.includes("anorexic") && /slender/u.test(plan.prompt), "Eating-disorder body terms are dropped; ordinary build words stay.");
  assertCheck("prompt.photo-sanitize", /tactical bodysuit/u.test(plan.prompt) && /white blouse/u.test(plan.prompt) && !/translucent|see-?through|sheer|transparent/iu.test(plan.prompt), `Sheer qualifiers are removed and the garment stays, opaque (${plan.subjectSentences[0]}).`);
  // The patterns are module-level regexes; a global flag would make .test() alternate between calls.
  const again = promptModule.buildPhotorealPrompt({
    characters: [{ tags: ["1girl", "translucent tactical bodysuit"], identityTags: [] }], contextTags: [], style, referenceSyntax: "none"
  });
  const third = promptModule.buildPhotorealPrompt({
    characters: [{ tags: ["1girl", "translucent tactical bodysuit"], identityTags: [] }], contextTags: [], style, referenceSyntax: "none"
  });
  assertCheck("prompt.photo-sanitize", again.prompt === third.prompt && !/translucent/u.test(third.prompt), "Repeated builds sanitize identically (no regex state carried between calls).");
}

function evaluateReferences(promptModule) {
  const keepFor = promptModule.createKeepIdentitySentence;
  const woman = { tags: ["1girl", "smile"], identityTags: ["1girl"], referenceIndex: 1 };
  const qwen = build(promptModule, { characters: [woman], referenceSyntax: "qwen21" });
  const keepOne = keepFor(["<image1>"]);
  assertCheck("prompt.references", qwen.prompt.includes("The woman from <image1>"), "qwen21 names the referenced person \"The woman from <image1>\".");
  assertCheck("prompt.references", qwen.prompt.endsWith(keepOne), "qwen21 ends with the keep-identity sentence naming <image1>.");
  // The wording measured on the live ComfyUI: the reference is identity-only and the BACKGROUND comes from the
  // description. Without the background clause Qwen-Image 2.1 kept the profile photo's room behind every cut.
  assertCheck(
    "prompt.references",
    keepOne.startsWith("Use <image1> only as the identity reference") && /background and lighting entirely from this description/u.test(keepOne),
    "The keep-identity sentence is identity-only and hands the background to the description (live-verified wording)."
  );
  const two = build(promptModule, {
    referenceSyntax: "qwen21",
    characters: [
      { ...woman, center: { x: 0.25, y: 0.5 } },
      { tags: ["1boy"], identityTags: ["1boy"], referenceIndex: 2, center: { x: 0.75, y: 0.5 } }
    ]
  });
  const keepTwo = keepFor(["<image1>", "<image2>"]);
  assertCheck("prompt.references", /the woman from <image1>.*the man from <image2>/u.test(two.prompt), "Each referenced person gets their own <imageN> token.");
  assertCheck("prompt.references", two.prompt.split(keepTwo).length === 2 && keepTwo.startsWith("Use <image1> and <image2> only as identity references"), "The keep-identity sentence names both tokens and appears exactly once.");
  const edit = build(promptModule, { characters: [woman], referenceSyntax: "edit_plus" });
  assertCheck("prompt.references", edit.prompt.includes("The woman from image 1") && !edit.prompt.includes("<image") && edit.prompt.endsWith(keepFor(["image 1"])), "edit_plus names the reference \"image 1\" (no <image1> token) in the subject and the closing sentence.");
  const none = build(promptModule, { characters: [woman], referenceSyntax: "none" });
  assertCheck("prompt.references", !none.prompt.includes("<image") && !/image 1/u.test(none.prompt) && !/identity reference/u.test(none.prompt), "referenceSyntax none produces neither a reference token nor the keep-identity sentence.");
  const unreferenced = build(promptModule, { characters: [{ tags: ["1girl"], identityTags: [] }], referenceSyntax: "qwen21" });
  assertCheck("prompt.references", !unreferenced.prompt.includes("<image") && !/identity reference/u.test(unreferenced.prompt), "A cut without referenceIndex never mentions reference images.");
}

// The look people asked for: a shared appearance on every woman, NovelAI emphasis read as words, places with a
// qualifier recognised, no daylight at night, and a face reference that does not freeze hair and makeup.
function evaluateLookAndBackground(promptModule) {
  const scope = "prompt.look";
  const shared = "+2.5:: beautiful realistic Korean woman, soft korean makeup, glossy lips ::";
  const woman = { tags: ["1girl", "long brown hair", "smile"], identityTags: ["1girl"] };
  const solo = build(promptModule, { characters: [woman], womanAppearance: shared });
  assertCheck(
    scope,
    /An adult woman, beautiful realistic Korean woman, soft korean makeup, glossy lips with long brown hair/u.test(solo.prompt),
    `The shared appearance follows the noun, before her own features (got "${solo.subjectSentences[0]}").`
  );
  assertCheck(scope, !/::|\+2\.5/u.test(solo.prompt), "Emphasis syntax in the shared appearance is read as words; no '::' or '+2.5' reaches the prompt.");
  const man = build(promptModule, { characters: [{ tags: ["1boy", "smile"], identityTags: ["1boy"] }], womanAppearance: shared });
  assertCheck(scope, !/Korean woman|glossy lips/u.test(man.prompt), "A man never gets the women's shared appearance.");
  const pair = build(promptModule, {
    characters: [
      { ...woman, center: { x: 0.25, y: 0.5 } },
      { tags: ["1boy"], identityTags: ["1boy"], center: { x: 0.75, y: 0.5 } }
    ],
    womanAppearance: "soft korean makeup"
  });
  assertCheck(scope, /On the left, an adult woman, soft korean makeup/u.test(pair.prompt) && !/an adult man, soft korean makeup/u.test(pair.prompt), "In a mixed cut only the woman carries it.");
  const own = build(promptModule, {
    characters: [{ ...woman, appearance: "A Korean woman with sleek black hair" }],
    womanAppearance: "glossy lips"
  });
  assertCheck(scope, /A Korean woman with sleek black hair, glossy lips/u.test(own.prompt), "With her own appearance the shared one is added after it.");
  const referenced = build(promptModule, { characters: [{ ...woman, referenceIndex: 1 }], referenceSyntax: "qwen21", womanAppearance: "glossy lips" });
  assertCheck(scope, /The woman from <image1>, glossy lips/u.test(referenced.prompt), "A referenced woman keeps the shared appearance after the reference phrase.");
  const minorCoded = build(promptModule, { characters: [woman], womanAppearance: "beautiful, schoolgirl" });
  assertCheck(scope, !/schoolgirl/u.test(minorCoded.prompt) && /beautiful/u.test(minorCoded.prompt), "A student-coded segment of the shared appearance is dropped like any appearance text.");

  const weightedAppearance = build(promptModule, { characters: [{ ...woman, appearance: "+1.5:: a Korean woman with long black hair ::, {glossy lips}, (freckles:1.2)" }] });
  assertCheck(
    scope,
    /a Korean woman with long black hair, glossy lips, freckles/iu.test(weightedAppearance.prompt) && !/::|[{}]|:1\.2/u.test(weightedAppearance.prompt),
    "Emphasis in a character's own appearance is read as words."
  );
  const weightedTag = build(promptModule, { characters: [{ tags: ["1girl", "+1.5::glossy lips::"], identityTags: [] }] });
  assertCheck(scope, /glossy lips/u.test(weightedTag.prompt) && !/\+1\.5|::/u.test(weightedTag.prompt), "A '+N::' weighted tag is read as its words.");
  const negativeGroup = build(promptModule, { characters: [{ ...woman, appearance: "A Korean woman, -1:: heavy makeup ::" }] });
  assertCheck(scope, !/heavy makeup/u.test(negativeGroup.prompt), "A negatively weighted group in free text is left out.");

  const style = dailyLook(promptModule);
  const person = [{ tags: ["1girl"], identityTags: [] }];
  const cafe = promptModule.buildPhotorealPrompt({ characters: person, contextTags: ["cafe interior", "potted plant"], style, referenceSyntax: "none" });
  assertCheck(scope, /Set in a cafe interior/u.test(cafe.prompt) && !/cream and white tones/u.test(cafe.prompt), `"cafe interior" is a place, so the preset's default room is not added (got "${cafe.settingSentence}").`);
  const base = promptModule.buildPhotorealPrompt({ characters: person, contextTags: ["military base", "hangar interior"], style, referenceSyntax: "none" });
  assertCheck(scope, /military base/u.test(base.settingSentence) && /hangar interior/u.test(base.settingSentence) && !/cream and white tones/u.test(base.prompt), "Military places are places.");
  const night = promptModule.buildPhotorealPrompt({ characters: person, contextTags: ["city street", "night"], style, referenceSyntax: "none" });
  assertCheck(scope, !/daylight/u.test(night.prompt), "A night cut with no lighting tag gets no default daylight.");

  const keep = promptModule.createKeepIdentitySentence(["<image1>"]);
  assertCheck(
    scope,
    /face and facial features/u.test(keep) && /hairstyle, makeup/u.test(keep) && !/hairstyle exactly the same/u.test(keep),
    "The face reference keeps the face only; hairstyle and makeup come from the description."
  );
  assertCheck(scope, promptModule.PHOTOREAL_STYLE_PRESETS.instagram && !/grain|lifted shadows/u.test(promptModule.PHOTOREAL_STYLE_PRESETS.instagram.tail), "The instagram preset asks for a clean, sharp look (no grain, no lifted shadows).");
  const screened = promptModule.describePhotorealStyleForScreening({ stylePreset: "instagram", stylePrompt: "", appearancePrompt: "glossy lips" });
  assertCheck(scope, screened.includes("glossy lips"), "The safety floor reads the shared appearance with the style.");
}

function evaluateAmbientDefaults(promptModule) {
  // The preset's light and room are defaults, not facts: they may only fill what the cut left unsaid.
  const style = promptModule.resolvePhotorealStyle({ stylePreset: "daily_look", stylePrompt: "", negativePrompt: "" });
  const person = [{ tags: ["1girl", "smile"], identityTags: [] }];
  const bare = promptModule.buildPhotorealPrompt({ characters: person, contextTags: [], style, referenceSyntax: "none" });
  assertCheck("prompt.ambient", bare.prompt.includes("Soft diffused natural daylight.") && /cream and white tones/u.test(bare.prompt), "A cut with no light and no place gets the preset's default daylight and room.");
  const neon = promptModule.buildPhotorealPrompt({ characters: person, contextTags: ["outdoors", "city street", "night", "neon lights"], style, referenceSyntax: "none" });
  assertCheck("prompt.ambient", !/daylight|cream and white tones/u.test(neon.prompt) && /neon lights/u.test(neon.prompt), "A neon-lit night street keeps its own light and place; no daylight room is added.");
  const lit = promptModule.buildPhotorealPrompt({ characters: person, contextTags: ["candlelight"], style, referenceSyntax: "none" });
  assertCheck("prompt.ambient", !/daylight/u.test(lit.prompt) && /cream and white tones/u.test(lit.prompt), "Own lighting suppresses only the default light; the default room still fills the missing place.");
  const placed = promptModule.buildPhotorealPrompt({ characters: person, contextTags: ["cafe"], style, referenceSyntax: "none" });
  assertCheck("prompt.ambient", /Soft diffused natural daylight\./u.test(placed.prompt) && !/cream and white tones/u.test(placed.prompt), "Own place suppresses only the default room; the default light still applies.");
  const empty = promptModule.buildPhotorealPrompt({ characters: [], contextTags: ["no humans", "scenery"], style, referenceSyntax: "none" });
  assertCheck("prompt.ambient", !/cream and white tones/u.test(empty.prompt), "A people-free cut never gets the preset's portrait room.");
}

function evaluateStylePresets(promptModule) {
  evaluateAmbientDefaults(promptModule);
  const presets = promptModule.PHOTOREAL_STYLE_PRESETS;
  const ids = Object.keys(presets);
  assertCheck("prompt.style", ["daily_look", "film_portrait", "cinematic", "studio_editorial"].every((id) => ids.includes(id)) && !ids.includes("custom"), "All four named style presets exist (custom is not a preset).");
  assertCheck(
    "prompt.style",
    presets.daily_look.lead === "A candid, true-to-life smartphone photograph in the style of a Korean and Chinese daily-look fashion post" && presets.daily_look.label === "데일리룩 · 핀터레스트/샤오홍슈",
    "daily_look carries the spec's lead and Korean label."
  );
  const person = [{ tags: ["1girl", "smile"], identityTags: [] }];
  for (const id of ids) {
    const preset = presets[id];
    const style = promptModule.resolvePhotorealStyle({ stylePreset: id, stylePrompt: "", negativePrompt: "" });
    const plan = promptModule.buildPhotorealPrompt({ characters: person, contextTags: [], style, referenceSyntax: "none" });
    assertCheck(
      "prompt.style",
      HANGUL.test(preset.label) && preset.description.length > 0 && plan.prompt.startsWith(`${preset.lead}.`) && plan.prompt.endsWith(`${preset.tail}.`),
      `${id}: Korean label, and the prompt opens with its lead and closes with its tail.`
    );
    assertCheck("prompt.style", splitList(plan.negativePrompt).includes("anime") && !HANGUL.test(plan.negativePrompt), `${id}: the negative keeps the photo-model away from anime.`);
  }
  const custom = promptModule.resolvePhotorealStyle({ stylePreset: "custom", stylePrompt: "A moody 35mm photograph. Teal and orange grade, heavy grain.", negativePrompt: "blurry" });
  assertCheck("prompt.style", custom.lead === "A moody 35mm photograph" && custom.tail === "Teal and orange grade, heavy grain" && custom.negative === "blurry", "custom: lead/tail/negative come from stylePrompt and negativePrompt only.");
  const customPlan = promptModule.buildPhotorealPrompt({ characters: person, contextTags: [], style: custom, referenceSyntax: "none" });
  assertCheck("prompt.style", customPlan.prompt.startsWith("A moody 35mm photograph. An adult woman") && customPlan.prompt.endsWith("Teal and orange grade, heavy grain.") && !/smartphone/u.test(customPlan.prompt), "custom: no preset text leaks in.");
  const emptyCustom = promptModule.resolvePhotorealStyle({ stylePreset: "custom", stylePrompt: "", negativePrompt: "" });
  const emptyPlan = promptModule.buildPhotorealPrompt({ characters: person, contextTags: [], style: emptyCustom, referenceSyntax: "none" });
  assertCheck("prompt.style", emptyPlan.prompt === "An adult woman, smiling.", `custom with no text leaves only the scene (got "${emptyPlan.prompt}").`);
  const appended = promptModule.resolvePhotorealStyle({ stylePreset: "cinematic", stylePrompt: "35mm film grain", negativePrompt: "freckles, Anime" });
  assertCheck("prompt.style", appended.tail.endsWith(", 35mm film grain") && appended.tail.startsWith(presets.cinematic.tail), "A preset's stylePrompt is appended to its tail.");
  assertCheck("prompt.style", splitList(appended.negative).includes("freckles") && (appended.negative.match(/anime/giu) ?? []).length === 1, "negativePrompt is appended to the preset negative, case-insensitively deduped.");
  const unknown = promptModule.resolvePhotorealStyle({ stylePreset: "vaporwave", stylePrompt: "", negativePrompt: "" });
  const inherited = promptModule.resolvePhotorealStyle({ stylePreset: "constructor", stylePrompt: "", negativePrompt: "" });
  assertCheck("prompt.style", unknown.lead === presets.instagram.lead && inherited.lead === presets.instagram.lead, "An unknown preset id (or a prototype name) falls back to the default look (instagram).");
}

function evaluateNegativeMerge(promptModule) {
  const plan = promptModule.buildPhotorealPrompt({
    characters: [{ tags: ["1girl", "-1::hat::"], identityTags: [] }],
    contextTags: [],
    style: { lead: "", tail: "", negative: "blurry, lowres, watermark" },
    extraNegative: "LOWRES, text, , watermark, 1.3::jpeg artifacts::",
    referenceSyntax: "none"
  });
  assertCheck(
    "prompt.negative",
    plan.negativePrompt === "blurry, lowres, watermark, text, jpeg artifacts, hat",
    `style negative + extraNegative merge, dedupe case-insensitively and strip weights (got "${plan.negativePrompt}").`
  );
  assertCheck("prompt.negative", !/\bhat\b/u.test(plan.prompt), "A NovelAI negative-weight tag is removed from the positive prompt.");
  const empty = promptModule.buildPhotorealPrompt({ characters: [{ tags: ["1girl"], identityTags: [] }], contextTags: [], style: EMPTY_STYLE, referenceSyntax: "none" });
  assertCheck("prompt.negative", empty.negativePrompt === "", "No negatives in → empty negative out (no stray commas).");

  // Red-team round 3: a negative that names what keeps the subject covered, safe or adult steers the sampler toward
  // the opposite at any CFG above 1, from a typed negative or from a cut's negative weight alike.
  const steering = promptModule.buildPhotorealPrompt({
    characters: [{ tags: ["1girl", "smile", "-2::apparel::", "-1::adult::", "-1::mature female::", "-1::glasses::"], identityTags: [] }],
    contextTags: [],
    style: { lead: "", tail: "", negative: "blurry, lowres, watermark" },
    extraNegative: "covered, censored, modest, fabric, apparel, wardrobe, sfw, safe for work, adult, mature, grown woman, aged, wrinkles, woman, tall, large breasts, coat, tights, towel, extra fingers",
    referenceSyntax: "none"
  });
  assertCheck(
    "prompt.negative",
    steering.negativePrompt === "blurry, lowres, watermark, extra fingers, glasses",
    `Covering, safety, adulthood and adult-build negatives are dropped; quality terms and accessories stay (got "${steering.negativePrompt}").`
  );
  const sanitized = promptModule.sanitizePhotorealNegativePrompt("anime, 1.2::sfw::, Clothing, nsfw, child, (fully clothed:1.3), hat");
  assertCheck("prompt.negative", sanitized === "anime, hat", `sanitizePhotorealNegativePrompt applies the same rule to a finished negative (got "${sanitized}").`);

  // Red-team round 4: the photo negative is an allow-list. A deny-list let 70 of 75 steering negatives through — every
  // garment, every language, every spelling — so only quality, artefact, medium and look terms (and a few worn extras)
  // are kept. Plurals, case, hyphens and underscores are one entry; anything not in the vocabulary is dropped.
  const presetNegatives = Object.values(promptModule.PHOTOREAL_STYLE_PRESETS).map((preset) => preset.negative);
  const presetLost = presetNegatives.filter((negative) => promptModule.sanitizePhotorealNegativePrompt(negative) !== negative);
  assertCheck("prompt.negative", presetLost.length === 0, `Every preset's negative survives the allow-list intact (changed: ${presetLost.join(" | ") || "none"}).`);
  const variants = promptModule.sanitizePhotorealNegativePrompt("Low-Res, JPEG_Artifacts, Extra Fingers, mutated-hands, poorly drawn faces, Out Of Frame, artefacts, (worst quality:1.4), {blurry}");
  assertCheck(
    "prompt.negative",
    variants === "Low-Res, JPEG Artifacts, Extra Fingers, mutated-hands, poorly drawn faces, Out Of Frame, artefacts, worst quality, blurry",
    `Close variants of quality terms (case, hyphen, underscore, plural, spelling, weights) are kept (got "${variants}").`
  );
  const steeringNegatives = "cl0thing, ropa, Kleidung, vêtements, adulthood, breasts, curves, knitwear, denim, turtleneck, long sleeves, in her twenties, 25yo, makeup, high heels, pubic hair, nude, naked child, loli, extra breasts, deformed clothing, censor bar, pixelation";
  const steeringKept = promptModule.sanitizePhotorealNegativePrompt(steeringNegatives);
  assertCheck("prompt.negative", steeringKept === "", `Nothing outside the vocabulary survives: garments in any language or spelling, adulthood, body words, safety words (kept "${steeringKept}").`);
  assertCheck("prompt.negative", ["blurry", "heavy makeup", "glasses", "tattoos"].every(promptModule.isPhotoNegativeQualityTerm) && !["makeup", "clothes", "hands", "skin"].some(promptModule.isPhotoNegativeQualityTerm), "A defect or look term is in the vocabulary; the bare thing it qualifies is not.");
}

function evaluateNaturalPhrasing(promptModule) {
  const subject = (tags) => build(promptModule, { characters: [{ tags: ["1girl", ...tags], identityTags: [] }] }).subjectSentences[0] ?? "";
  assertCheck("prompt.phrasing", /long black hair in a ponytail/u.test(subject(["long hair", "black hair", "ponytail"])), "Hair tags merge: long hair + black hair + ponytail → \"long black hair in a ponytail\".");
  assertCheck("prompt.phrasing", /blushing slightly/u.test(subject(["blush"])), "blush → \"blushing slightly\".");
  assertCheck("prompt.phrasing", /with tears in her eyes/u.test(subject(["crying"])), "crying → \"with tears in her eyes\".");
  assertCheck("prompt.phrasing", /looking at the camera/u.test(subject(["looking at viewer"])) && !/viewer/u.test(subject(["looking at viewer"])), "looking at viewer → \"looking at the camera\".");
  assertCheck("prompt.phrasing", /holding a phone/u.test(subject(["holding phone"])) && /sitting on a chair/u.test(subject(["sitting on chair"])), "Articles come back: holding phone → \"holding a phone\", sitting on chair → \"sitting on a chair\".");
  assertCheck("prompt.phrasing", /with a hand on her hip/u.test(subject(["hand on hip"])), "Possessives follow the subject: \"with a hand on her hip\".");
  const man = build(promptModule, { characters: [{ tags: ["1boy", "crying", "arms crossed"], identityTags: [] }] }).subjectSentences[0] ?? "";
  assertCheck("prompt.phrasing", /with tears in his eyes/u.test(man) && /with his arms crossed/u.test(man), "Possessives follow a male subject (his).");
  const empty = build(promptModule, { contextTags: ["no humans", "cafe", "coffee cup", "cake", "window light"] });
  assertCheck("prompt.phrasing", empty.subjectSentences.length === 0 && !/adult|woman|\bman\b/u.test(empty.prompt), "A no-humans cut has no subject sentence and no person noun.");
  assertCheck("prompt.phrasing", /A coffee cup and a cake in a cafe, lit by window light\. No people are present\./u.test(empty.settingSentence), `The objects lead a no-humans setting, and the empty frame is stated (${empty.settingSentence}).`);
  const setting = build(promptModule, { characters: [{ tags: ["1girl"], identityTags: [] }], contextTags: ["bedroom", "indoors", "night", "white curtains", "dim lighting", "rain"] }).settingSentence;
  assertCheck("prompt.phrasing", setting === "Set in a bedroom at night in the rain, with white curtains, with dim lighting.", `The setting sentence places, times, lists props and lighting (got "${setting}").`);
}

function evaluatePathologicalInput(promptModule) {
  const tags = [];
  for (let index = 0; index < 120; index += 1) {
    const kind = index % 6;
    tags.push(
      kind === 0
        ? `한국어 태그 ${index}`
        : kind === 1
          ? `1.${index % 9}::unusual decorative detail number ${index}::`
          : kind === 2
            ? `{{ornate background element ${index}}}`
            : kind === 3
              ? `artist:someone_${index}`
              : `long descriptive background phrase number ${index} with several extra words`
    );
  }
  const plan = build(promptModule, {
    referenceSyntax: "qwen21",
    characters: [
      { tags: ["1girl", "smile", "white blouse", ...tags.slice(0, 60)], identityTags: [], referenceIndex: 1, center: { x: 0.2, y: 0.5 } },
      { tags: ["1boy", "black suit", ...tags.slice(60)], identityTags: [], center: { x: 0.8, y: 0.5 } }
    ],
    contextTags: tags,
    frame: { shot: "cowboy_shot", viewpoint: "front", visibleRegions: SHOT_REGIONS.cowboy_shot }
  });
  const max = promptModule.PHOTOREAL_PROMPT_MAX_CHARS;
  assertCheck("prompt.length", max === 2400 && plan.prompt.length <= max, `A 120-tag caption stays within ${max} characters (got ${plan.prompt.length}).`);
  assertCheck("prompt.length", !HANGUL.test(plan.prompt) && !/::|[{}[\]]|artist/u.test(plan.prompt), "The capped prompt is English only, with no NovelAI syntax.");
  assertCheck("prompt.length", plan.prompt.endsWith(promptModule.createKeepIdentitySentence(["<image1>"])), "The reference sentence survives trimming intact, at the end.");
  assertCheck("prompt.length", /On the left, the woman from <image1>/u.test(plan.prompt) && /on the right, an adult man/u.test(plan.prompt), "Both subjects survive trimming.");
  assertCheck("prompt.length", plan.prompt.startsWith(dailyLook(promptModule).lead) && /Medium shot from mid-thigh up[,.]/u.test(plan.prompt), "The style lead and the camera sentence survive trimming.");
  assertCheck("prompt.length", Array.isArray(plan.trimmedTags) && plan.trimmedTags.length > 0 && plan.trimmedTags.every((tag) => !HANGUL.test(tag)), "Trimmed tags are reported (only kept-language tags are ever trimmed).");
  const hugeStyle = { lead: `A photograph ${"with an extremely long custom style ".repeat(80)}`, tail: "film grain ".repeat(100), negative: "" };
  const clamped = promptModule.buildPhotorealPrompt({ characters: [{ tags: ["1girl"], identityTags: [], referenceIndex: 1 }], contextTags: [], style: hugeStyle, referenceSyntax: "qwen21" });
  assertCheck("prompt.length", clamped.prompt.length <= max && clamped.prompt.endsWith(promptModule.createKeepIdentitySentence(["<image1>"])), `An oversized custom style is hard-clamped and the reference sentence is still appended (${clamped.prompt.length}).`);
}

// ---------------------------------------------------------------------------------------------------------------
// comfyWorkflows.ts
// ---------------------------------------------------------------------------------------------------------------

function baseParams(comfyModule, presetId, overrides = {}) {
  const preset = comfyModule.PHOTOREAL_WORKFLOW_PRESETS[presetId];
  return {
    prompt: "A candid smartphone photograph. An adult woman, smiling.",
    negativePrompt: "anime, cartoon",
    width: 832,
    height: 1248,
    seed: 123456789,
    steps: preset.defaults.steps,
    cfg: preset.defaults.cfg,
    files: { ...comfyModule.resolvePhotorealModelFiles({ modelFiles: {} }, presetId) },
    referenceNames: [],
    filenamePrefix: "DynamicChat/job_eval_1",
    ...overrides
  };
}

/**
 * Independent static validator: what ComfyUI's /prompt validation would reject, checked without ComfyUI. Every
 * graph the builder produces goes through this, then through the module's own validator. Split from the
 * assertions so evaluateStaticValidatorSelfTest can prove it actually catches each defect.
 */
/**
 * `outputClassType` is the node the graph must end in: PreviewImage for every built-in graph (results go to ComfyUI's
 * temp folder, which DynamicChat's image deletion does not have to reach), whatever the custom workflow uses.
 */
function inspectGraph(comfyModule, built, { presetId, expectedClassTypes, forbiddenClassTypes = [], references = [], outputClassType = "PreviewImage" }) {
  const workflow = built.workflow;
  const ids = Object.keys(workflow);
  const structural = [];
  const badNumbers = [];
  const leftovers = [];
  const refHits = new Map();
  const visitStrings = (value, where, onString) => {
    if (typeof value === "string") {
      onString(value, where);
    } else if (Array.isArray(value)) {
      value.forEach((item) => visitStrings(item, where, onString));
    } else if (value && typeof value === "object") {
      Object.values(value).forEach((item) => visitStrings(item, where, onString));
    }
  };
  for (const id of ids) {
    const node = workflow[id];
    if (!node || typeof node !== "object" || typeof node.class_type !== "string" || !node.class_type || !node.inputs || typeof node.inputs !== "object" || Array.isArray(node.inputs)) {
      structural.push(`node ${id} lacks class_type/inputs`);
      continue;
    }
    for (const [key, value] of Object.entries(node.inputs)) {
      if (Array.isArray(value)) {
        if (value.length !== 2 || typeof value[0] !== "string" || !Number.isInteger(value[1]) || value[1] < 0) {
          structural.push(`${id}.${key} is not a ["id", n] link`);
        } else if (!workflow[value[0]]) {
          structural.push(`${id}.${key} → missing node ${value[0]}`);
        }
        continue;
      }
      if (NUMERIC_INPUTS.has(key) && typeof value !== "number") {
        badNumbers.push(`${node.class_type}.${key}=${JSON.stringify(value)}`);
      }
      visitStrings(value, `${node.class_type}.${key}`, (text, where) => {
        const ref = text.match(REF_PATTERN);
        if (ref) {
          const hits = refHits.get(ref[1]) ?? [];
          hits.push(where);
          refHits.set(ref[1], hits);
          return;
        }
        if (text.includes("%ref:")) {
          leftovers.push(`${where}: stray %ref:`);
        }
        for (const match of text.matchAll(/%([A-Za-z_][A-Za-z0-9_]*)%/gu)) {
          // SaveImage expands these itself (folder_paths.get_save_image_path), so they are not left for anyone.
          if (where === "SaveImage.filename_prefix" && COMFY_FILENAME_TOKENS.has(match[1])) {
            continue;
          }
          leftovers.push(`${where}: %${match[1]}%`);
        }
      });
    }
  }
  const refProblems = [
    ...references.flatMap((name) => {
      const hits = refHits.get(name) ?? [];
      return hits.length === 1 && hits[0] === "LoadImage.image" ? [] : [`${name}: ${hits.length} hit(s) ${hits.join("/")}`];
    }),
    ...[...refHits.keys()].filter((name) => !references.includes(name)).map((name) => `unexpected ${name}`)
  ];
  const badOutputs = built.outputNodeIds.length === 0 ? ["none"] : built.outputNodeIds.filter((id) => workflow[id]?.class_type !== outputClassType);
  const present = new Set(ids.map((id) => workflow[id]?.class_type));
  const missing = expectedClassTypes.filter((classType) => !present.has(classType));
  const forbidden = forbiddenClassTypes.filter((classType) => present.has(classType));
  const required = new Set(comfyModule.PHOTOREAL_WORKFLOW_PRESETS[presetId]?.requiredNodes ?? []);
  const unprobed = presetId === "custom" ? [] : [...present].filter((classType) => !required.has(classType));
  return { structural, badNumbers, leftovers, refProblems, badOutputs, missing, forbidden, unprobed };
}

function staticGraphChecks(comfyModule, scope, built, options) {
  const report = inspectGraph(comfyModule, built, options);
  const references = options.references ?? [];
  assertCheck(scope, report.structural.length === 0, `every node has class_type + inputs and every link ["id", n] resolves (${report.structural.join("; ") || "ok"}).`);
  assertCheck(scope, report.badNumbers.length === 0, `seed/width/height/steps/cfg/... are JSON numbers, not strings (${report.badNumbers.join(", ") || "ok"}).`);
  assertCheck(scope, report.leftovers.length === 0, `no unresolved %placeholder% is left (${report.leftovers.join(", ") || "ok"}).`);
  assertCheck(scope, report.refProblems.length === 0, `each reference appears as %ref:<name>% exactly once, in a LoadImage image input (${report.refProblems.join("; ") || `${references.length} ok`}).`);
  assertCheck(scope, report.badOutputs.length === 0, `outputNodeIds ${JSON.stringify(built.outputNodeIds)} exist and are ${options.outputClassType ?? "PreviewImage"} nodes.`);
  assertCheck(scope, report.missing.length === 0 && report.forbidden.length === 0, `expected node types present (missing: ${report.missing.join(", ") || "none"}; unexpected: ${report.forbidden.join(", ") || "none"}).`);
  assertCheck(scope, report.unprobed.length === 0, `every node type is listed in the preset's requiredNodes for the /object_info probe (${report.unprobed.join(", ") || "ok"}).`);
  const ownErrors = comfyModule.validateComfyApiWorkflow(built.workflow);
  assertCheck(scope, ownErrors.length === 0, `validateComfyApiWorkflow accepts it (${ownErrors.join(" / ") || "ok"}).`);
  assertCheck(scope, built.presetId === options.presetId && JSON.stringify(built.usedReferences) === JSON.stringify(references), `presetId and usedReferences report what was built (${JSON.stringify(built.usedReferences)}).`);
}

/** The static validator above is only evidence if it fails on broken graphs; break one graph each way and look. */
function evaluateStaticValidatorSelfTest(comfyModule) {
  const options = { presetId: "qwen_image_2_1_turbo", expectedClassTypes: QWEN21_TURBO_TYPES, references: ["char_1"] };
  const good = comfyModule.buildComfyWorkflow("qwen_image_2_1_turbo", baseParams(comfyModule, "qwen_image_2_1_turbo", { referenceNames: ["char_1"] }));
  const clean = inspectGraph(comfyModule, good, options);
  assertCheck("comfy.static-validator", Object.values(clean).every((problems) => problems.length === 0), "self-test: a correct graph reports no problem at all.");
  const broken = (mutate) => {
    const copy = structuredClone(good);
    mutate(copy);
    return inspectGraph(comfyModule, copy, options);
  };
  const idOf = (classType) => findNodes(good.workflow, classType)[0].id;
  assertCheck("comfy.static-validator", broken((copy) => { copy.workflow[idOf("CFGGuider")].inputs.model = ["404", 0]; }).structural.length === 1, "self-test: a dangling link is caught.");
  assertCheck("comfy.static-validator", broken((copy) => { copy.workflow[idOf("RandomNoise")].inputs.noise_seed = "42"; }).badNumbers.length === 1, "self-test: a string seed is caught.");
  assertCheck("comfy.static-validator", broken((copy) => { copy.workflow[idOf("TextEncodeQwenImage21")].inputs.prompt = "%prompt%"; }).leftovers.length === 1, "self-test: a leftover placeholder is caught.");
  assertCheck("comfy.static-validator", broken((copy) => { copy.workflow["99"] = { class_type: "LoadImage", inputs: { image: "%ref:char_1%" } }; }).refProblems.length === 1, "self-test: a reference loaded twice is caught.");
  assertCheck("comfy.static-validator", broken((copy) => { copy.workflow["99"] = { class_type: "LoadImage", inputs: { image: "%ref:char_9%" } }; }).refProblems.length === 1, "self-test: an unplanned reference is caught.");
  assertCheck("comfy.static-validator", broken((copy) => { copy.outputNodeIds = [idOf("CFGGuider")]; }).badOutputs.length === 1, "self-test: an output id that is not the graph's output node is caught.");
  assertCheck("comfy.static-validator", broken((copy) => { delete copy.workflow[idOf("ManualSigmas")]; }).missing.length === 1, "self-test: a missing expected node type is caught.");
  assertCheck("comfy.static-validator", broken((copy) => { copy.workflow["98"] = { class_type: "UpscaleModelLoader", inputs: { model_name: "x" } }; }).unprobed.length === 1, "self-test: a node type the probe would not check is caught.");
}

function findNodes(workflow, classType) {
  return Object.entries(workflow)
    .filter(([, node]) => node.class_type === classType)
    .map(([id, node]) => ({ id, node }));
}

function onlyNode(workflow, classType) {
  const nodes = findNodes(workflow, classType);
  if (nodes.length !== 1) {
    throw new Error(`expected exactly one ${classType}, found ${nodes.length}`);
  }
  return nodes[0];
}

function loadImageIdFor(workflow, name) {
  return findNodes(workflow, "LoadImage").find(({ node }) => node.inputs.image === `%ref:${name}%`)?.id;
}

function sameLink(value, id, index = 0) {
  return Array.isArray(value) && value[0] === id && value[1] === index;
}

function evaluatePresetMetadata(comfyModule) {
  const presets = comfyModule.PHOTOREAL_WORKFLOW_PRESETS;
  const ids = ["qwen_image_2_1_turbo", "qwen_image_2_1", "z_image_turbo", "qwen_image_edit_2511", "custom"];
  assertCheck("comfy.presets", ids.every((id) => presets[id]?.id === id) && Object.keys(presets).length === ids.length, "Every PhotorealWorkflowPresetId has a preset whose id matches its key.");
  for (const id of ids.filter((presetId) => presetId !== "custom")) {
    const preset = presets[id];
    const files = new Set(preset.downloads.map((download) => download.file));
    assertCheck(
      "comfy.presets",
      [preset.defaults.diffusionModel, preset.defaults.textEncoder, preset.defaults.vae].every((file) => files.has(file)) &&
        preset.downloads.every((download) => /^https:\/\/huggingface\.co\/Comfy-Org\//u.test(download.url) && download.url.endsWith(`/${download.file}`) && download.sizeGb > 0) &&
        HANGUL.test(preset.label) &&
        HANGUL.test(preset.vramNote),
      `${id}: the three default files are in its download list (Comfy-Org URLs, sizes), with Korean label and VRAM note.`
    );
  }
  const turbo = presets.qwen_image_2_1_turbo;
  assertCheck("comfy.presets", turbo.referenceSyntax === "qwen21" && turbo.maxReferences === 4 && turbo.supportsReferences && turbo.license === "Qwen Research (비상업)" && !turbo.commercialUse, "qwen_image_2_1_turbo: qwen21 references, max 4, non-commercial Qwen licence.");
  assertCheck("comfy.presets", turbo.defaults.diffusionModel === "qwen_image_2.1_turbo_int8_convrot.safetensors" && turbo.defaults.textEncoder === "qwen3vl_8b_int8_convrot.safetensors" && turbo.defaults.vae === "qwen_image_2.1_vae_bf16.safetensors", "qwen_image_2_1_turbo defaults to the int8_convrot files.");
  const zImage = presets.z_image_turbo;
  assertCheck("comfy.presets", !zImage.supportsReferences && zImage.maxReferences === 0 && zImage.referenceSyntax === "none" && zImage.license === "Apache-2.0" && zImage.commercialUse, "z_image_turbo: no references, Apache-2.0.");
  assertCheck("comfy.presets", zImage.defaults.diffusionModel === "z_image_turbo_int8_convrot.safetensors" && zImage.defaults.textEncoder === "qwen_3_4b_fp8_mixed.safetensors" && zImage.defaults.vae === "ae.safetensors", "z_image_turbo defaults to int8 + fp8_mixed + ae.");
  const edit = presets.qwen_image_edit_2511;
  assertCheck("comfy.presets", edit.referenceSyntax === "edit_plus" && edit.maxReferences === 3 && edit.license === "Apache-2.0" && edit.defaults.steps === 40 && edit.defaults.cfg === 4, "qwen_image_edit_2511: edit_plus, max 3, Apache-2.0, 40 steps / cfg 4.");
  assertCheck("comfy.presets", presets.qwen_image_2_1.defaults.steps === 25 && presets.qwen_image_2_1.defaults.diffusionModel === "qwen_image_2.1_int8_convrot.safetensors", "qwen_image_2_1: 25 steps on the base int8 file.");
  assertCheck("comfy.presets", comfyModule.QWEN21_REFERENCE_INPUT_KEY(1) === "images.image_1" && comfyModule.QWEN21_REFERENCE_INPUT_KEY(4) === "images.image_4", "QWEN21_REFERENCE_INPUT_KEY names the Autogrow inputs images.image_N.");
  // The relay refuses any class_type outside its default allow-list (spec §4) unless the operator widens it, so a
  // built-in preset that needed one more node would fail on every machine with a default configuration. Graphs are
  // separately checked to use only their preset's requiredNodes, which closes the chain.
  const relayDefaults = new Set([
    "CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "VAELoader", "LoraLoader", "LoraLoaderModelOnly",
    "CLIPTextEncode", "ConditioningZeroOut", "ConditioningCombine", "ConditioningSetAreaPercentage", "KSampler", "KSamplerAdvanced", "SamplerCustom",
    "SamplerCustomAdvanced", "KSamplerSelect", "RandomNoise", "DisableNoise", "CFGGuider", "BasicGuider", "BasicScheduler", "ManualSigmas",
    "Flux2Scheduler", "ModelSamplingAuraFlow", "ModelSamplingSD3", "ModelSamplingFlux", "CFGNorm", "EmptyLatentImage", "EmptySD3LatentImage",
    "EmptyFlux2LatentImage", "VAEDecode", "VAEEncode", "VAEDecodeTiled", "VAEEncodeTiled", "LoadImage", "ImageScale", "ImageCrop", "ImageScaleBy",
    "ImageScaleToTotalPixels", "GetImageSize", "FluxKontextImageScale", "FluxKontextMultiReferenceLatentMethod", "ReferenceLatent",
    "TextEncodeQwenImageEdit", "TextEncodeQwenImageEditPlus", "TextEncodeQwenImage21", "QwenImage21Cache", "SaveImage", "PreviewImage",
    "PrimitiveString", "PrimitiveInt", "PrimitiveFloat", "UnetLoaderGGUF", "CLIPLoaderGGUF", "DualCLIPLoaderGGUF"
  ]);
  const blocked = ids.flatMap((id) => presets[id].requiredNodes.filter((node) => !relayDefaults.has(node)).map((node) => `${id}:${node}`));
  assertCheck("comfy.presets", blocked.length === 0, `Every built-in preset node passes the relay's default allow-list (${blocked.join(", ") || "ok"}).`);
}

function evaluateZImageTurboGraph(comfyModule) {
  const params = baseParams(comfyModule, "z_image_turbo", { referenceNames: ["char_1"] });
  const built = comfyModule.buildComfyWorkflow("z_image_turbo", params);
  staticGraphChecks(comfyModule, "comfy.z-image", built, {
    presetId: "z_image_turbo",
    expectedClassTypes: ["UNETLoader", "ModelSamplingAuraFlow", "CLIPLoader", "VAELoader", "CLIPTextEncode", "ConditioningZeroOut", "EmptySD3LatentImage", "KSampler", "VAEDecode", "PreviewImage"],
    forbiddenClassTypes: ["LoadImage", "SaveImage"],
    references: []
  });
  const workflow = built.workflow;
  const sampler = onlyNode(workflow, "KSampler").node.inputs;
  assertCheck("comfy.z-image", sampler.sampler_name === "res_multistep" && sampler.scheduler === "simple" && sampler.steps === 8 && sampler.cfg === 1 && sampler.denoise === 1 && sampler.seed === params.seed, "KSampler: res_multistep / simple, 8 steps, cfg 1, denoise 1, the job's seed.");
  const shift = onlyNode(workflow, "ModelSamplingAuraFlow").node.inputs.shift;
  const clip = onlyNode(workflow, "CLIPLoader").node.inputs;
  assertCheck("comfy.z-image", shift === 3 && clip.type === "lumina2" && clip.device === "default", "ModelSamplingAuraFlow shift 3; CLIPLoader type lumina2.");
  const encode = onlyNode(workflow, "CLIPTextEncode");
  const zero = onlyNode(workflow, "ConditioningZeroOut");
  assertCheck("comfy.z-image", encode.node.inputs.text === params.prompt && sameLink(zero.node.inputs.conditioning, encode.id) && sameLink(sampler.positive, encode.id) && sameLink(sampler.negative, zero.id), "The prompt is encoded once; the negative is the zeroed positive.");
  const latent = onlyNode(workflow, "EmptySD3LatentImage").node.inputs;
  const unet = onlyNode(workflow, "UNETLoader").node.inputs;
  assertCheck("comfy.z-image", latent.width === 832 && latent.height === 1248 && latent.batch_size === 1 && unet.unet_name === "z_image_turbo_int8_convrot.safetensors" && unet.weight_dtype === "default", "Latent uses the job size; the loader uses the preset file.");
  const output = onlyNode(workflow, "PreviewImage");
  assertCheck(
    "comfy.z-image",
    sameLink(output.node.inputs.images, onlyNode(workflow, "VAEDecode").id) && !("filename_prefix" in output.node.inputs),
    "The result goes to a PreviewImage (ComfyUI's temp folder), never a SaveImage: a deleted cut must not live on in ComfyUI's output folder."
  );
}

function evaluateQwen21TurboGraphs(comfyModule) {
  const keyFor = comfyModule.QWEN21_REFERENCE_INPUT_KEY;
  const noRefs = comfyModule.buildComfyWorkflow("qwen_image_2_1_turbo", baseParams(comfyModule, "qwen_image_2_1_turbo"));
  staticGraphChecks(comfyModule, "comfy.qwen21-turbo", noRefs, {
    presetId: "qwen_image_2_1_turbo",
    expectedClassTypes: QWEN21_TURBO_TYPES,
    forbiddenClassTypes: ["LoadImage", "KSampler", "SaveImage"],
    references: []
  });
  const workflow = noRefs.workflow;
  const encoder = onlyNode(workflow, "TextEncodeQwenImage21");
  assertCheck("comfy.qwen21-turbo", !Object.keys(encoder.node.inputs).some((key) => key.startsWith("images.")), "0 references: the encoder has no images.* inputs.");
  assertCheck(
    "comfy.qwen21-turbo",
    encoder.node.inputs.prompt.startsWith("A candid") && encoder.node.inputs.negative_prompt === "anime, cartoon" && encoder.node.inputs.resolution === 1024 && Array.isArray(encoder.node.inputs.vae),
    "TextEncodeQwenImage21 gets prompt, negative_prompt, vae and resolution 1024."
  );
  const guider = onlyNode(workflow, "CFGGuider").node.inputs;
  assertCheck("comfy.qwen21-turbo", sameLink(guider.positive, encoder.id, 0) && sameLink(guider.negative, encoder.id, 1) && guider.cfg === 1, "CFGGuider takes the encoder's positive [enc,0] and negative [enc,1] at cfg 1.");
  const cache = onlyNode(workflow, "QwenImage21Cache");
  assertCheck("comfy.qwen21-turbo", sameLink(guider.model, cache.id) && cache.node.inputs.device === "auto" && cache.node.inputs.dtype === "default", "The guider samples the QwenImage21Cache model (device auto).");
  assertCheck("comfy.qwen21-turbo", onlyNode(workflow, "ManualSigmas").node.inputs.sigmas === "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0", "ManualSigmas carries Turbo's 8-step schedule verbatim.");
  const sca = onlyNode(workflow, "SamplerCustomAdvanced");
  const decode = onlyNode(workflow, "VAEDecode").node.inputs;
  assertCheck(
    "comfy.qwen21-turbo",
    onlyNode(workflow, "RandomNoise").node.inputs.noise_seed === 123456789 && onlyNode(workflow, "KSamplerSelect").node.inputs.sampler_name === "euler" && sameLink(decode.samples, sca.id, 0),
    "RandomNoise uses the seed, KSamplerSelect euler, VAEDecode decodes [sca,0]."
  );
  assertCheck("comfy.qwen21-turbo", onlyNode(workflow, "CLIPLoader").node.inputs.type === "qwen_image", "CLIPLoader type qwen_image.");

  const oneRef = comfyModule.buildComfyWorkflow("qwen_image_2_1_turbo", baseParams(comfyModule, "qwen_image_2_1_turbo", { referenceNames: ["char_1"] }));
  staticGraphChecks(comfyModule, "comfy.qwen21-turbo", oneRef, {
    presetId: "qwen_image_2_1_turbo",
    expectedClassTypes: [...QWEN21_TURBO_TYPES, "LoadImage"],
    forbiddenClassTypes: ["SaveImage"],
    references: ["char_1"]
  });
  const oneEncoder = onlyNode(oneRef.workflow, "TextEncodeQwenImage21").node.inputs;
  assertCheck("comfy.qwen21-turbo", sameLink(oneEncoder[keyFor(1)], loadImageIdFor(oneRef.workflow, "char_1")), "1 reference: images.image_1 is wired to the LoadImage carrying %ref:char_1%.");

  const threeRefs = comfyModule.buildComfyWorkflow("qwen_image_2_1_turbo", baseParams(comfyModule, "qwen_image_2_1_turbo", { referenceNames: ["char_1", "char_2", "char_3"] }));
  staticGraphChecks(comfyModule, "comfy.qwen21-turbo", threeRefs, {
    presetId: "qwen_image_2_1_turbo",
    expectedClassTypes: [...QWEN21_TURBO_TYPES, "LoadImage"],
    forbiddenClassTypes: ["SaveImage"],
    references: ["char_1", "char_2", "char_3"]
  });
  const threeEncoder = onlyNode(threeRefs.workflow, "TextEncodeQwenImage21").node.inputs;
  assertCheck(
    "comfy.qwen21-turbo",
    [1, 2, 3].every((slot) => sameLink(threeEncoder[keyFor(slot)], loadImageIdFor(threeRefs.workflow, `char_${slot}`))) && !(keyFor(4) in threeEncoder),
    "3 references: images.image_1..3 follow the reference order, and no fourth slot is wired."
  );
  const sixRefs = comfyModule.buildComfyWorkflow("qwen_image_2_1_turbo", baseParams(comfyModule, "qwen_image_2_1_turbo", { referenceNames: ["a", "b", "c", "d", "e", "f"] }));
  assertCheck("comfy.qwen21-turbo", sixRefs.usedReferences.join(",") === "a,b,c,d" && findNodes(sixRefs.workflow, "LoadImage").length === 4, "More references than maxReferences are capped at 4, and usedReferences says which.");
  let invalidName = "";
  try {
    comfyModule.buildComfyWorkflow("qwen_image_2_1_turbo", baseParams(comfyModule, "qwen_image_2_1_turbo", { referenceNames: ["../../etc/passwd"] }));
  } catch (error) {
    invalidName = describeError(error);
  }
  assertCheck("comfy.qwen21-turbo", /참조 이미지 이름/u.test(invalidName), "A reference name the relay would reject (path characters) fails the build.");
  const quirky = comfyModule.buildComfyWorkflow("qwen_image_2_1_turbo", baseParams(comfyModule, "qwen_image_2_1_turbo", { prompt: "50%off% sale, %ref:char_1% sign", width: 1000, height: 1501, seed: -7.5 }));
  const quirkyEncoder = onlyNode(quirky.workflow, "TextEncodeQwenImage21").node.inputs.prompt;
  const quirkyLatent = onlyNode(quirky.workflow, "EmptyLatentImage").node.inputs;
  assertCheck(
    "comfy.qwen21-turbo",
    !/%off%|%ref:/u.test(quirkyEncoder) && quirkyLatent.width === 1008 && quirkyLatent.height === 1504 && onlyNode(quirky.workflow, "RandomNoise").node.inputs.noise_seed === 7,
    "Prompt text cannot fake a placeholder; sizes round to 16 and the seed becomes a non-negative integer."
  );
}

function evaluateQwen21BaseGraph(comfyModule) {
  const built = comfyModule.buildComfyWorkflow("qwen_image_2_1", baseParams(comfyModule, "qwen_image_2_1", { referenceNames: ["char_1"] }));
  staticGraphChecks(comfyModule, "comfy.qwen21", built, {
    presetId: "qwen_image_2_1",
    expectedClassTypes: ["UNETLoader", "QwenImage21Cache", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21", "EmptyLatentImage", "KSampler", "VAEDecode", "PreviewImage", "LoadImage"],
    forbiddenClassTypes: ["SamplerCustomAdvanced", "ManualSigmas", "CFGGuider", "SaveImage"],
    references: ["char_1"]
  });
  const workflow = built.workflow;
  const sampler = onlyNode(workflow, "KSampler").node.inputs;
  const encoder = onlyNode(workflow, "TextEncodeQwenImage21");
  assertCheck(
    "comfy.qwen21",
    sampler.steps === 25 && sampler.cfg === 1 && sampler.sampler_name === "euler" && sampler.scheduler === "simple" && sameLink(sampler.positive, encoder.id, 0) && sameLink(sampler.negative, encoder.id, 1),
    "Plain KSampler: 25 steps, cfg 1, euler/simple, on the encoder's positive/negative outputs."
  );
  assertCheck("comfy.qwen21", onlyNode(workflow, "UNETLoader").node.inputs.unet_name === "qwen_image_2.1_int8_convrot.safetensors", "The base preset loads the non-turbo diffusion file.");
}

function evaluateQwenEdit2511Graphs(comfyModule) {
  const noRefs = comfyModule.buildComfyWorkflow("qwen_image_edit_2511", baseParams(comfyModule, "qwen_image_edit_2511"));
  staticGraphChecks(comfyModule, "comfy.edit-2511", noRefs, {
    presetId: "qwen_image_edit_2511",
    expectedClassTypes: [...EDIT_TYPES, "EmptySD3LatentImage"],
    forbiddenClassTypes: ["LoadImage", "VAEEncode", "FluxKontextImageScale", "SaveImage"],
    references: []
  });
  const workflow = noRefs.workflow;
  const encoders = findNodes(workflow, "TextEncodeQwenImageEditPlus");
  const positive = encoders.find(({ node }) => node.inputs.prompt.length > 0);
  const negative = encoders.find(({ node }) => node.inputs.prompt === "");
  assertCheck("comfy.edit-2511", encoders.length === 2 && positive && negative && !encoders.some(({ node }) => "image1" in node.inputs), "0 references: positive + empty-prompt negative encoders, no image inputs.");
  const methods = findNodes(workflow, "FluxKontextMultiReferenceLatentMethod");
  const sampler = onlyNode(workflow, "KSampler").node.inputs;
  const methodFor = (encoderId) => methods.find(({ node }) => sameLink(node.inputs.conditioning, encoderId));
  assertCheck(
    "comfy.edit-2511",
    methods.length === 2 && methods.every(({ node }) => node.inputs.reference_latents_method === "index_timestep_zero") && sameLink(sampler.positive, methodFor(positive.id)?.id) && sameLink(sampler.negative, methodFor(negative.id)?.id),
    "Both conditionings pass through FluxKontextMultiReferenceLatentMethod(index_timestep_zero) into the sampler."
  );
  const norm = onlyNode(workflow, "CFGNorm");
  const aura = onlyNode(workflow, "ModelSamplingAuraFlow").node.inputs;
  assertCheck("comfy.edit-2511", aura.shift === 3.1 && norm.node.inputs.strength === 1 && sameLink(sampler.model, norm.id), "UNET → ModelSamplingAuraFlow(3.1) → CFGNorm(1) → KSampler.");
  assertCheck("comfy.edit-2511", sampler.steps === 40 && sampler.cfg === 4 && sampler.sampler_name === "euler" && sampler.scheduler === "simple" && sampler.denoise === 1, "KSampler 40 steps, cfg 4, euler/simple, denoise 1.");
  const latent = onlyNode(workflow, "EmptySD3LatentImage");
  assertCheck("comfy.edit-2511", sameLink(sampler.latent_image, latent.id) && latent.node.inputs.width === 832 && latent.node.inputs.height === 1248, "Without references the latent is EmptySD3LatentImage at the job size.");

  const twoRefs = comfyModule.buildComfyWorkflow("qwen_image_edit_2511", baseParams(comfyModule, "qwen_image_edit_2511", { referenceNames: ["char_1", "char_2"] }));
  staticGraphChecks(comfyModule, "comfy.edit-2511", twoRefs, {
    presetId: "qwen_image_edit_2511",
    expectedClassTypes: [...EDIT_TYPES, "LoadImage", "EmptySD3LatentImage"],
    // The template's VAEEncode(FluxKontextImageScale(image1)) latent made the reference the edit target; a story cut
    // is a new composition, so the latent is the cut's own size (measured failure on the same model family).
    forbiddenClassTypes: ["FluxKontextImageScale", "VAEEncode", "SaveImage"],
    references: ["char_1", "char_2"]
  });
  const refWorkflow = twoRefs.workflow;
  const loadOne = loadImageIdFor(refWorkflow, "char_1");
  const loadTwo = loadImageIdFor(refWorkflow, "char_2");
  const refEncoders = findNodes(refWorkflow, "TextEncodeQwenImageEditPlus");
  assertCheck(
    "comfy.edit-2511",
    refEncoders.length === 2 && refEncoders.every(({ node }) => sameLink(node.inputs.image1, loadOne) && sameLink(node.inputs.image2, loadTwo) && !("image3" in node.inputs) && Array.isArray(node.inputs.vae)),
    "Both encoders get image1 = reference 1, image2 = reference 2, no image3, and the VAE."
  );
  const refLatent = onlyNode(refWorkflow, "EmptySD3LatentImage");
  assertCheck(
    "comfy.edit-2511",
    sameLink(onlyNode(refWorkflow, "KSampler").node.inputs.latent_image, refLatent.id) && refLatent.node.inputs.width === 832 && refLatent.node.inputs.height === 1248,
    "With references the latent is still EmptySD3LatentImage at the cut's size (no pixel-aligned edit of reference 1)."
  );
  const fourRefs = comfyModule.buildComfyWorkflow("qwen_image_edit_2511", baseParams(comfyModule, "qwen_image_edit_2511", { referenceNames: ["a", "b", "c", "d"] }));
  assertCheck("comfy.edit-2511", fourRefs.usedReferences.join(",") === "a,b,c" && findNodes(fourRefs.workflow, "LoadImage").length === 3, "Edit 2511 caps references at its 3 image slots.");
  evaluateReferenceHeadCrops(comfyModule);
}

function evaluateReferenceHeadCrops(comfyModule) {
  const crop = comfyModule.computeReferenceHeadCrop;
  const square = crop(1024, 1024);
  assertCheck("comfy.head-crop", square && square.width === 634 && square.height === 634 && square.x === 195 && square.y === 20, `1024x1024 profile → top-centred 634px square (${JSON.stringify(square)}).`);
  const portrait = crop(960, 1280);
  assertCheck("comfy.head-crop", portrait && portrait.width === 793 && portrait.x === 83 && portrait.y === 25 && portrait.y + portrait.height <= 1280, `960x1280 profile → 793px square from the top (${JSON.stringify(portrait)}).`);
  const wide = crop(1344, 768);
  assertCheck("comfy.head-crop", wide && wide.width === 476 && wide.x === 434 && wide.y + wide.height <= 768, `A landscape profile still crops a centred square inside the image (${JSON.stringify(wide)}).`);
  assertCheck("comfy.head-crop", crop(undefined, 1024) === undefined && crop(200, 300) === undefined && crop(NaN, NaN) === undefined, "Unknown or tiny sizes give no crop (the reference is sent whole).");

  for (const presetId of ["qwen_image_2_1_turbo", "qwen_image_edit_2511"]) {
    const built = comfyModule.buildComfyWorkflow(
      presetId,
      baseParams(comfyModule, presetId, { referenceNames: ["char_1", "char_2"], referenceCrops: { char_1: crop(960, 1280) } })
    );
    const crops = findNodes(built.workflow, "ImageCrop");
    const loadOne = loadImageIdFor(built.workflow, "char_1");
    const loadTwo = loadImageIdFor(built.workflow, "char_2");
    const encoderType = presetId === "qwen_image_2_1_turbo" ? "TextEncodeQwenImage21" : "TextEncodeQwenImageEditPlus";
    const encoders = findNodes(built.workflow, encoderType);
    const firstKey = presetId === "qwen_image_2_1_turbo" ? comfyModule.QWEN21_REFERENCE_INPUT_KEY(1) : "image1";
    const secondKey = presetId === "qwen_image_2_1_turbo" ? comfyModule.QWEN21_REFERENCE_INPUT_KEY(2) : "image2";
    assertCheck(
      "comfy.head-crop",
      crops.length === 1 && sameLink(crops[0].node.inputs.image, loadOne) && crops[0].node.inputs.width === 793 && crops[0].node.inputs.x === 83 && crops[0].node.inputs.y === 25,
      `${presetId}: one ImageCrop on reference 1 with the planned box.`
    );
    assertCheck(
      "comfy.head-crop",
      encoders.length > 0 && encoders.every(({ node }) => sameLink(node.inputs[firstKey], crops[0]?.id) && sameLink(node.inputs[secondKey], loadTwo)),
      `${presetId}: the encoder reads the CROPPED reference 1 and the uncropped reference 2.`
    );
    assertCheck("comfy.head-crop", comfyModule.validateComfyApiWorkflow(built.workflow).length === 0, `${presetId}: the cropped graph validates.`);
  }
  const bogus = comfyModule.buildComfyWorkflow(
    "qwen_image_2_1_turbo",
    baseParams(comfyModule, "qwen_image_2_1_turbo", { referenceNames: ["char_1"], referenceCrops: { char_1: { x: -5, y: 0, width: 10, height: 10 }, "../evil": { x: 0, y: 0, width: 500, height: 500 } } })
  );
  assertCheck("comfy.head-crop", findNodes(bogus.workflow, "ImageCrop").length === 0, "A malformed crop (negative origin, tiny box, bad name) is ignored rather than wired.");
}

/** SillyTavern's shipped default workflow, plus a reference loader and an interpolated string. */
function sillyTavernWorkflow() {
  return {
    3: {
      class_type: "KSampler",
      inputs: {
        cfg: "%scale%",
        denoise: "%denoise%",
        latent_image: ["5", 0],
        model: ["4", 0],
        negative: ["7", 0],
        positive: ["6", 0],
        sampler_name: "%sampler%",
        scheduler: "%scheduler%",
        seed: "%seed%",
        steps: "%steps%"
      }
    },
    4: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "%model%" } },
    5: { class_type: "EmptyLatentImage", inputs: { batch_size: 1, height: "%height%", width: "%width%" } },
    6: { class_type: "CLIPTextEncode", inputs: { clip: ["4", 1], text: "%prompt%" } },
    7: { class_type: "CLIPTextEncode", inputs: { clip: ["4", 1], text: "%negative_prompt%" } },
    8: { class_type: "VAEDecode", inputs: { samples: ["3", 0], vae: ["4", 2] } },
    9: { class_type: "SaveImage", inputs: { filename_prefix: "SillyTavern_%year%", images: ["8", 0] } },
    10: { class_type: "LoadImage", inputs: { image: "%image1%" } },
    // A number placeholder may only be a whole value: written into text it would compose words the floor never read
    // ("a %steps% year old girl"), so only %prompt% is interpolated here (see the refusal checked below).
    11: { class_type: "PrimitiveString", inputs: { value: "for %prompt%" } }
  };
}

function evaluateCustomWorkflow(comfyModule) {
  const params = baseParams(comfyModule, "custom", {
    files: { diffusionModel: "realistic_checkpoint.safetensors", textEncoder: "", vae: "" },
    referenceNames: ["char_1"],
    steps: 24,
    cfg: 5.5,
    seed: 42
  });
  const built = comfyModule.buildComfyWorkflow("custom", params, JSON.stringify(sillyTavernWorkflow()));
  staticGraphChecks(comfyModule, "comfy.custom", built, {
    presetId: "custom",
    expectedClassTypes: ["KSampler", "CheckpointLoaderSimple", "EmptyLatentImage", "CLIPTextEncode", "VAEDecode", "SaveImage", "LoadImage"],
    // A custom workflow is the creator's own graph and keeps the output node it was written with.
    outputClassType: "SaveImage",
    references: ["char_1"]
  });
  const workflow = built.workflow;
  const sampler = workflow["3"].inputs;
  assertCheck("comfy.custom", sampler.seed === 42 && sampler.steps === 24 && sampler.cfg === 5.5 && sampler.denoise === 1, "Typed substitution: %seed%/%steps%/%scale%/%denoise% become JSON numbers.");
  assertCheck("comfy.custom", workflow["5"].inputs.width === 832 && workflow["5"].inputs.height === 1248, "%width%/%height% become numbers.");
  assertCheck("comfy.custom", sampler.sampler_name === "euler" && sampler.scheduler === "simple" && workflow["4"].inputs.ckpt_name === "realistic_checkpoint.safetensors", "%sampler%/%scheduler%/%model% resolve to strings.");
  assertCheck("comfy.custom", workflow["6"].inputs.text === params.prompt && workflow["7"].inputs.text === params.negativePrompt, "%prompt%/%negative_prompt% become the prompt texts.");
  assertCheck("comfy.custom", workflow["11"].inputs.value === `for ${params.prompt}`, "Inside a longer string %prompt% is interpolated as text.");
  let numberInText = "";
  try {
    comfyModule.buildComfyWorkflow("custom", params, JSON.stringify({ ...sillyTavernWorkflow(), 11: { class_type: "PrimitiveString", inputs: { value: "seed %seed% for %prompt%" } } }));
  } catch (error) {
    numberInText = describeError(error);
  }
  assertCheck("comfy.custom", /%seed%/u.test(numberInText), `A number placeholder inside text is refused rather than interpolated (${numberInText || "built"}).`);
  assertCheck("comfy.custom", workflow["10"].inputs.image === "%ref:char_1%", "%image1% becomes %ref:char_1% for the relay.");
  assertCheck("comfy.custom", workflow["9"].inputs.filename_prefix === "SillyTavern_%year%" && JSON.stringify(built.outputNodeIds) === JSON.stringify(["9"]), "ComfyUI's own %year% filename token is left alone; the SaveImage is the output node.");

  const failures = {
    missingSlot: () => comfyModule.buildComfyWorkflow("custom", baseParams(comfyModule, "custom", { referenceNames: ["char_1"] }), JSON.stringify({ ...sillyTavernWorkflow(), 12: { class_type: "LoadImage", inputs: { image: "%image2%" } } })),
    uiFormat: () => comfyModule.buildComfyWorkflow("custom", baseParams(comfyModule, "custom"), JSON.stringify({ nodes: [{ id: 1, type: "KSampler" }], links: [], version: 0.4 })),
    empty: () => comfyModule.buildComfyWorkflow("custom", baseParams(comfyModule, "custom"), "  "),
    noModel: () => comfyModule.buildComfyWorkflow("custom", baseParams(comfyModule, "custom", { referenceNames: ["char_1"] }), JSON.stringify(sillyTavernWorkflow())),
    badJson: () => comfyModule.buildComfyWorkflow("custom", baseParams(comfyModule, "custom"), "{ not json")
  };
  const messages = Object.fromEntries(
    Object.entries(failures).map(([name, run]) => {
      try {
        run();
        return [name, ""];
      } catch (error) {
        return [name, describeError(error)];
      }
    })
  );
  assertCheck("comfy.custom", /%image2%/u.test(messages.missingSlot) && /1장/u.test(messages.missingSlot), `A workflow that needs more reference slots than the cut has fails with a clear message (${messages.missingSlot}).`);
  assertCheck("comfy.custom", /API/u.test(messages.uiFormat), `A UI-format export is rejected with a hint to export the API format (${messages.uiFormat}).`);
  assertCheck("comfy.custom", /비어 있습니다/u.test(messages.empty), "An empty custom workflow is rejected.");
  assertCheck("comfy.custom", /%model%/u.test(messages.noModel), `%model% with no model file set stays unresolved and is reported (${messages.noModel}).`);
  assertCheck("comfy.custom", /JSON/u.test(messages.badJson), "Malformed JSON is reported as such.");

  // The settings panel validates the raw template, where %prompt% and friends are supposed to still be present.
  const templateErrors = comfyModule.validateCustomWorkflowTemplate(JSON.stringify(sillyTavernWorkflow()));
  assertCheck("comfy.custom", templateErrors.length === 0, `validateCustomWorkflowTemplate accepts a template whose placeholders DynamicChat supplies (${templateErrors.join(" / ") || "ok"}).`);
  const unsupported = comfyModule.validateCustomWorkflowTemplate(JSON.stringify({ ...sillyTavernWorkflow(), 12: { class_type: "PrimitiveString", inputs: { value: "%lora_strength%" } } }));
  assertCheck("comfy.custom", unsupported.length === 1 && /%lora_strength%/u.test(unsupported[0]), "validateCustomWorkflowTemplate flags a placeholder DynamicChat does not supply.");
  assertCheck(
    "comfy.custom",
    comfyModule.validateCustomWorkflowTemplate("").length === 0 && comfyModule.validateCustomWorkflowTemplate("{ broken").length === 1 && /API/u.test(comfyModule.validateCustomWorkflowTemplate(JSON.stringify({ nodes: [], links: [] }))[0] ?? ""),
    "validateCustomWorkflowTemplate: empty text is not an error; bad JSON and UI-format exports are."
  );
  assertCheck(
    "comfy.custom",
    comfyModule.COMFY_CUSTOM_PLACEHOLDERS.map((entry) => entry.name).join(",") === "prompt,negative_prompt,seed,steps,scale,cfg,width,height,denoise,clip_skip,sampler,scheduler,model,text_encoder,vae,filename_prefix,image1,image2,image3",
    "COMFY_CUSTOM_PLACEHOLDERS documents every supported placeholder (SillyTavern names included)."
  );
}

function evaluatePlaceholderSubstitution(comfyModule) {
  const template = JSON.stringify({
    1: { class_type: "KSampler", inputs: { seed: "%seed%", steps: "%steps%", note: "%foo%", label: "run %seed%" } },
    2: { class_type: "LoadImage", inputs: { image: "%image1%" } },
    3: { class_type: "LoadImage", inputs: { image: "%image2%" } },
    4: { class_type: "CLIPTextEncode", inputs: { text: "%prompt%" } }
  });
  const result = comfyModule.substituteComfyPlaceholders(template, { seed: 7, steps: "9", prompt: "a %seed% literal", image1: "char_a", image2: "%ref:char_b%" });
  assertCheck("comfy.substitute", result["1"].inputs.seed === 7 && typeof result["1"].inputs.seed === "number", "An exact \"%seed%\" value becomes the number 7.");
  assertCheck("comfy.substitute", result["1"].inputs.steps === "9", "The value's own type is kept (a string stays a string).");
  assertCheck("comfy.substitute", result["1"].inputs.note === "%foo%", "Unknown placeholders are left in place for the validator to report.");
  assertCheck("comfy.substitute", result["1"].inputs.label === "run 7", "Interpolation inside a longer string.");
  assertCheck("comfy.substitute", result["2"].inputs.image === "%ref:char_a%" && result["3"].inputs.image === "%ref:char_b%", "image1..3 carry reference names and become %ref:<name>% (idempotent for an already-wrapped value).");
  assertCheck("comfy.substitute", result["4"].inputs.text === "a %seed% literal", "A substituted value is never re-scanned (single pass).");
  let parseError = "";
  try {
    comfyModule.substituteComfyPlaceholders("[unterminated", {});
  } catch (error) {
    parseError = describeError(error);
  }
  assertCheck("comfy.substitute", /JSON/u.test(parseError), "Invalid JSON throws a readable error.");
}

function minimalValidWorkflow() {
  return {
    1: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "model.safetensors" } },
    2: { class_type: "CLIPTextEncode", inputs: { clip: ["1", 1], text: "a photo" } },
    3: { class_type: "EmptyLatentImage", inputs: { width: 1024, height: 1024, batch_size: 1 } },
    4: { class_type: "KSampler", inputs: { model: ["1", 0], positive: ["2", 0], negative: ["2", 0], latent_image: ["3", 0], seed: 1, steps: 20, cfg: 5, sampler_name: "euler", scheduler: "normal", denoise: 1 } },
    5: { class_type: "VAEDecode", inputs: { samples: ["4", 0], vae: ["1", 2] } },
    6: { class_type: "SaveImage", inputs: { images: ["5", 0], filename_prefix: "x_%year%-%month%" } }
  };
}

function evaluateValidatorRejections(comfyModule) {
  const validate = comfyModule.validateComfyApiWorkflow;
  assertCheck("comfy.validate", validate(minimalValidWorkflow()).length === 0, "A well-formed API workflow (with ComfyUI's own %year% filename token) validates.");
  const ui = validate({ nodes: [{ id: 3, type: "KSampler" }], links: [[1, 4, 0, 3, 0, "MODEL"]], groups: [], version: 0.4 });
  assertCheck("comfy.validate", ui.length === 1 && /API/u.test(ui[0]), "A UI-format export (nodes + links arrays) is rejected with an export hint.");
  const noOutput = minimalValidWorkflow();
  delete noOutput["6"];
  assertCheck("comfy.validate", validate(noOutput).some((error) => /SaveImage/u.test(error)), "A graph with no SaveImage/PreviewImage output is rejected.");
  const preview = minimalValidWorkflow();
  preview["6"] = { class_type: "PreviewImage", inputs: { images: ["5", 0] } };
  assertCheck("comfy.validate", validate(preview).length === 0, "PreviewImage counts as an output node.");
  const dangling = minimalValidWorkflow();
  dangling["4"].inputs.model = ["99", 0];
  assertCheck("comfy.validate", validate(dangling).some((error) => /"99"/u.test(error)), "A dangling link [\"99\", 0] is rejected naming the missing node.");
  const numericLink = minimalValidWorkflow();
  numericLink["4"].inputs.model = [1, 0];
  assertCheck("comfy.validate", validate(numericLink).some((error) => /문자열/u.test(error)), "A numeric link id ([1, 0]) is rejected — ComfyUI only resolves string ids.");
  const placeholder = minimalValidWorkflow();
  placeholder["2"].inputs.text = "%prompt%";
  assertCheck("comfy.validate", validate(placeholder).some((error) => /%prompt%/u.test(error)), "An unresolved %prompt% placeholder is rejected.");
  const avatar = minimalValidWorkflow();
  avatar["7"] = { class_type: "LoadImage", inputs: { image: "%char_avatar%" } };
  assertCheck("comfy.validate", validate(avatar).some((error) => /SillyTavern/u.test(error) && /%image1%/u.test(error)), "SillyTavern's %char_avatar% gets a specific hint toward %image1%.");
  const strayRef = minimalValidWorkflow();
  strayRef["7"] = { class_type: "LoadImage", inputs: { image: "input/%ref:char_1%.png" } };
  assertCheck("comfy.validate", validate(strayRef).some((error) => /전체/u.test(error)), "A %ref:% inside a longer string is rejected (the relay only substitutes whole values).");
  const badRefName = minimalValidWorkflow();
  badRefName["7"] = { class_type: "LoadImage", inputs: { image: "%ref:../secret%" } };
  assertCheck("comfy.validate", validate(badRefName).some((error) => /이름/u.test(error)), "A %ref:% with a path-like name is rejected.");
  const missingClass = minimalValidWorkflow();
  delete missingClass["3"].class_type;
  assertCheck("comfy.validate", validate(missingClass).some((error) => /class_type/u.test(error)), "A node without class_type is rejected.");
  const badInputs = minimalValidWorkflow();
  badInputs["3"].inputs = "width=1024";
  assertCheck("comfy.validate", validate(badInputs).some((error) => /inputs/u.test(error)), "A node whose inputs is not an object is rejected.");
  assertCheck("comfy.validate", validate("not a workflow").length > 0 && validate([]).length > 0 && validate(null).length > 0 && validate({}).length > 0, "Non-objects, arrays and empty objects are rejected.");
  assertCheck("comfy.validate", validate({ prompt: minimalValidWorkflow(), client_id: "x" }).some((error) => /prompt/u.test(error)), "A whole /prompt request body is recognised and rejected with a hint.");
  const huge = {};
  for (let index = 0; index < 301; index += 1) {
    huge[String(index)] = { class_type: "SaveImage", inputs: { filename_prefix: "x" } };
  }
  assertCheck("comfy.validate", validate(huge).some((error) => /300/u.test(error)), "More than 300 nodes is rejected (the relay's limit).");
}

function evaluateModelFiles(comfyModule) {
  const resolve = comfyModule.resolvePhotorealModelFiles;
  const presets = comfyModule.PHOTOREAL_WORKFLOW_PRESETS;
  const defaults = ["qwen_image_2_1_turbo", "qwen_image_2_1", "z_image_turbo", "qwen_image_edit_2511"].every((id) => {
    const files = resolve({ modelFiles: {} }, id);
    return files.diffusionModel === presets[id].defaults.diffusionModel && files.textEncoder === presets[id].defaults.textEncoder && files.vae === presets[id].defaults.vae;
  });
  assertCheck("comfy.model-files", defaults, "No overrides → every built-in preset resolves to its default files.");
  const settings = { modelFiles: { qwen_image_2_1_turbo: { diffusionModel: " qwen_image_2.1_turbo_bf16.safetensors ", textEncoder: "   ", vae: "" } } };
  const overridden = resolve(settings, "qwen_image_2_1_turbo");
  assertCheck("comfy.model-files", overridden.diffusionModel === "qwen_image_2.1_turbo_bf16.safetensors", "An override replaces the preset file (trimmed).");
  assertCheck("comfy.model-files", overridden.textEncoder === "qwen3vl_8b_int8_convrot.safetensors" && overridden.vae === "qwen_image_2.1_vae_bf16.safetensors", "Blank or whitespace-only overrides fall back to the preset file.");
  assertCheck("comfy.model-files", resolve(settings, "qwen_image_2_1").diffusionModel === "qwen_image_2.1_int8_convrot.safetensors", "One preset's override never leaks into another preset.");
  const custom = resolve({ modelFiles: {} }, "custom");
  assertCheck("comfy.model-files", custom.diffusionModel === "" && custom.textEncoder === "" && custom.vae === "", "custom has no default files (its placeholders stay unresolved until set).");
  const built = comfyModule.buildComfyWorkflow("z_image_turbo", baseParams(comfyModule, "z_image_turbo", { files: resolve({ modelFiles: { z_image_turbo: { vae: "my_ae.safetensors" } } }, "z_image_turbo") }));
  assertCheck("comfy.model-files", onlyNode(built.workflow, "VAELoader").node.inputs.vae_name === "my_ae.safetensors", "The resolved override reaches the built graph.");
}

function evaluateOutputSize(comfyModule) {
  const size = (photoreal, profile = { width: 832, height: 1216 }) =>
    comfyModule.resolvePhotorealOutputSize({ photoreal: { width: 0, height: 0, ...photoreal }, imageProfile: profile });
  const cases = [
    ["explicit 1000x1500 rounds to multiples of 16", size({ width: 1000, height: 1500 }), [1008, 1504]],
    ["explicit 300x4000 clamps to 512..2048", size({ width: 300, height: 4000 }), [512, 2048]],
    ["explicit 1024x1024 is kept", size({ width: 1024, height: 1024 }), [1024, 1024]],
    ["aspect 832x1216 → 2:3 832x1248", size({}, { width: 832, height: 1216 }), [832, 1248]],
    ["aspect 1216x832 → 3:2 1248x832", size({}, { width: 1216, height: 832 }), [1248, 832]],
    ["aspect 1024x1024 → 1:1 1024x1024", size({}, { width: 1024, height: 1024 }), [1024, 1024]],
    ["aspect 720x1280 → 9:16 768x1344", size({}, { width: 720, height: 1280 }), [768, 1344]],
    ["aspect 1920x1080 → 16:9 1344x768", size({}, { width: 1920, height: 1080 }), [1344, 768]],
    ["aspect 900x1200 → 3:4 960x1280", size({}, { width: 900, height: 1200 }), [960, 1280]],
    ["aspect 1200x900 → 4:3 1280x960", size({}, { width: 1200, height: 900 }), [1280, 960]],
    ["an unusable profile size falls back to 1:1", size({}, { width: 0, height: Number.NaN }), [1024, 1024]],
    ["one explicit side keeps the profile's aspect", size({ width: 1280 }, { width: 832, height: 1216 }), [1280, 1872]]
  ];
  for (const [label, actual, [width, height]] of cases) {
    assertCheck("comfy.output-size", actual.width === width && actual.height === height && actual.width % 16 === 0 && actual.height % 16 === 0, `${label} (got ${actual.width}x${actual.height}).`);
  }
  assertCheck("comfy.output-size", comfyModule.resolvePhotorealOutputSize({ imageProfile: { width: 832, height: 1216 } }).width === 832, "A state saved before photoreal existed (no state.photoreal) still resolves.");
}

function evaluatePresetForCut(comfyModule) {
  const pick = comfyModule.resolvePhotorealPresetForCut;
  const settings = { workflowPreset: "z_image_turbo", referenceWorkflowPreset: "qwen_image_2_1_turbo", faceReferenceEnabled: true };
  assertCheck("comfy.preset-for-cut", pick(settings, true) === "qwen_image_2_1_turbo", "References + face reference on + a reference-capable preset → referenceWorkflowPreset.");
  assertCheck("comfy.preset-for-cut", pick(settings, false) === "z_image_turbo", "No references → workflowPreset.");
  assertCheck("comfy.preset-for-cut", pick({ ...settings, faceReferenceEnabled: false }, true) === "z_image_turbo", "Face reference off → workflowPreset even with references.");
  assertCheck("comfy.preset-for-cut", pick({ ...settings, referenceWorkflowPreset: "z_image_turbo", workflowPreset: "qwen_image_2_1" }, true) === "qwen_image_2_1", "A reference preset that cannot take references falls back to workflowPreset.");
  assertCheck("comfy.preset-for-cut", pick({ ...settings, referenceWorkflowPreset: "qwen_image_edit_2511" }, true) === "qwen_image_edit_2511", "Edit 2511 is a valid reference preset.");
  assertCheck("comfy.preset-for-cut", pick({ ...settings, referenceWorkflowPreset: "custom" }, true) === "custom", "A custom workflow can be the reference preset.");
  assertCheck("comfy.preset-for-cut", pick({ ...settings, workflowPreset: "flux_dev_unknown" }, false) === "qwen_image_2_1_turbo", "An unknown workflowPreset falls back to the shipped default.");
  // Ids come from stored state; inherited Object.prototype names must not count as presets.
  assertCheck(
    "comfy.preset-for-cut",
    pick({ ...settings, workflowPreset: "constructor", referenceWorkflowPreset: "__proto__" }, true) === "qwen_image_2_1_turbo",
    "Prototype names (constructor, __proto__) are not presets."
  );
  let unknownBuild = "";
  try {
    comfyModule.buildComfyWorkflow("constructor", baseParams(comfyModule, "z_image_turbo"));
  } catch (error) {
    unknownBuild = describeError(error);
  }
  const protoFiles = comfyModule.resolvePhotorealModelFiles({ modelFiles: {} }, "toString");
  assertCheck(
    "comfy.preset-for-cut",
    /알 수 없는/u.test(unknownBuild) && protoFiles.diffusionModel === "" && protoFiles.vae === "",
    "buildComfyWorkflow refuses an unknown id; resolvePhotorealModelFiles treats it as having no default files."
  );
}

// ---------------------------------------------------------------------------------------------------------------

function splitList(value) {
  return String(value ?? "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function runGroup(scope, evaluate) {
  try {
    evaluate();
  } catch (error) {
    assertCheck(scope, false, `threw: ${describeError(error)}`);
  }
}

function describeError(error) {
  return error instanceof Error ? error.message : String(error);
}

function assertCheck(scope, passed, message) {
  checks.push({
    level: passed ? "pass" : "fail",
    scope,
    message
  });
}
