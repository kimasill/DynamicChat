import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// The photoreal safety floor against a corpus, end to end.
//
// photoreal-routing-eval pins the classifier's rules one by one. This suite asks the question the floor exists to
// answer: does any of this text reach a photo model? Every probe of every bypass class two red teams demonstrated
// (English inflections and compounds, Danbooru NSFW tags, stated minor ages in any phrasing, school + suggestive,
// other Latin-script languages, leetspeak and in-word punctuation, Unicode disguises, a bare chest by gender,
// NovelAI's negative-weight idioms) goes through the REAL planner in all-photoreal mode — as a cut tag, inside a
// character's photoreal appearance and inside a custom style — and through the screens a creator meets (the free-
// text screen and the profile studio). It must lock to NovelAI or be refused, and never be printed into a photo
// prompt. The other half is the cost: a plain SFW corpus and the false friends this floor was taught to tell apart
// must stay photoreal, and the shipped adult seed must not lock its ordinary cuts.
//
// The corpus lives in photoreal-safety-corpus.json. It is a sample of each class, not the lexicon: a lexical floor
// cannot be complete, and the rules are written for the class, so a new spelling of a covered class belongs here.
// Round 4 added the classes a third red team found (garments that expose by design, decimal and contextual ages,
// paraphrases, broken-up and symbol-masked words, pictographs, implied nudity, school without the word, the child-body
// combination, a bare chest in a global text, the Latin-script long tail), each with probes beyond the red team's own
// strings, and the photo negative's allow-list.

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const vite = await createServer({
  root: workspaceRoot,
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom"
});

const SIM = "sim_safety_corpus";
const FRAMES = {
  upper_body: { shot: "upper_body", viewpoint: "front", visibleRegions: ["head", "torso"] },
  cowboy_shot: { shot: "cowboy_shot", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs"] },
  full_body: { shot: "full_body", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs", "feet"] },
  close_up: { shot: "close_up", viewpoint: "front", visibleRegions: ["head"] }
};
const APPEARANCE_LEAD = "A Korean woman in her late twenties with long black hair";

const checks = [];
const corpusSizes = [];

try {
  let modules;
  try {
    modules = {
      seed: await vite.ssrLoadModule("/src/data/seed.ts"),
      womanLife: await vite.ssrLoadModule("/src/data/womanLifeSeed.ts"),
      orchestrator: await vite.ssrLoadModule("/src/services/imageOrchestrator.ts"),
      studio: await vite.ssrLoadModule("/src/services/characterProfileStudio.ts"),
      classifier: await vite.ssrLoadModule("/src/services/imageSceneClassifier.ts"),
      photo: await vite.ssrLoadModule("/src/services/photorealPrompt.ts")
    };
  } catch (error) {
    assertCheck("load", false, `module load: ${describeError(error)}`);
  }
  if (modules) {
    const corpus = JSON.parse(await readFile(path.join(__dirname, "photoreal-safety-corpus.json"), "utf8"));
    const lab = createLab(modules);
    runGroup("floor.explicit", () => evaluateMustLock(lab, "floor.explicit", corpus.explicit));
    runGroup("floor.danbooru", () => evaluateMustLock(lab, "floor.danbooru", corpus.danbooruExplicit));
    runGroup("floor.minor", () => evaluateMustLock(lab, "floor.minor", corpus.minor));
    runGroup("floor.multilingual", () => evaluateMustLock(lab, "floor.multilingual", corpus.multilingual));
    runGroup("floor.leet", () => evaluateMustLock(lab, "floor.leet", corpus.leet));
    runGroup("floor.unicode", () => evaluateMustLock(lab, "floor.unicode", corpus.unicode));
    runGroup("floor.student", () => evaluateStudentIdentity(lab, corpus.studentIdentity));
    runGroup("floor.school", () => evaluateSchoolSuggestive(lab, corpus.school, corpus.schoolSuggestive));
    runGroup("floor.danbooru-graded", () => evaluateGradedDanbooru(lab, corpus.danbooruSuggestive, corpus.danbooruAmbiguous));
    runGroup("floor.bare-chest", () => evaluateBareChest(lab, corpus.bareChestFemale, corpus.bareChestMale));
    runGroup("floor.negative-weights", () => evaluateNegativeWeights(lab, corpus.negativeWeights));
    runGroup("floor.hangul", () => evaluateHangulSegments(lab));
    runGroup("floor.garments", () => evaluateMustLock(lab, "floor.garments", corpus.exposingGarments));
    runGroup("floor.garments", () => evaluateExposingSeedOutfits(lab));
    runGroup("floor.decimal-ages", () => evaluateMustLock(lab, "floor.decimal-ages", corpus.decimalAges));
    runGroup("floor.age-forms", () => evaluateMustLock(lab, "floor.age-forms", corpus.ageForms));
    runGroup("floor.nudity-paraphrase", () => evaluateMustLock(lab, "floor.nudity-paraphrase", corpus.nudityParaphrases));
    runGroup("floor.act-paraphrase", () => evaluateMustLock(lab, "floor.act-paraphrase", corpus.actParaphrases));
    runGroup("floor.obfuscation", () => evaluateMustLock(lab, "floor.obfuscation", corpus.obfuscation));
    runGroup("floor.emoji", () => evaluateMustLock(lab, "floor.emoji", corpus.emoji));
    runGroup("floor.emoji", () => evaluatePictographStripping(lab));
    runGroup("floor.implied-nudity", () => evaluateMustLock(lab, "floor.implied-nudity", corpus.impliedNudity));
    runGroup("floor.implied-nudity", () => evaluateMustPass(lab, "floor.implied-nudity", corpus.impliedNudityScenery, "a place, a towel or a tub of ice cream is not bathing"));
    runGroup("floor.school-composite", () => evaluateSchoolComposite(lab, corpus.schoolComposite, corpus.schoolCompositeClean));
    runGroup("floor.child-body", () => evaluateMustLock(lab, "floor.child-body", corpus.childBody));
    runGroup("floor.child-body", () => evaluateMustPass(lab, "floor.child-body", corpus.adultPetite, "adult petite fashion, one descriptor or no swimwear"));
    runGroup("floor.global-bare-chest", () => evaluateGlobalBareChest(lab, corpus.globalBareChest));
    runGroup("floor.multilingual-long-tail", () => evaluateMustLock(lab, "floor.multilingual-long-tail", corpus.multilingualLongTail));
    runGroup("negatives", () => evaluateNegativeSanitizer(lab, corpus.negativeDrop, corpus.negativeKeep));
    runGroup("negatives", () => evaluateNegativeAllowList(lab, corpus.negativeSteering));
    runGroup("sfw.plain", () => evaluatePlainSfw(lab));
    runGroup("sfw.false-friends", () => evaluateFalseFriendFixes(lab, corpus.falsePositiveFixes, corpus.latinFolds));
    runGroup("sfw.womanlife", () => evaluateWomanLifeSeed(lab));
    runGroup("sfw.camera-style", () => evaluateCameraStyle(lab));
    runGroup("reference.korean-pick", () => evaluateKoreanPickRecord(lab, modules.studio));
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
console.log(`Corpus sizes: ${corpusSizes.map(({ name, size }) => `${name} ${size}`).join(", ")}`);
console.log(`Photoreal safety corpus checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Lab: the real planner, screens and studio over a minimal photoreal-on state
// ---------------------------------------------------------------------------------------------------------------

function createLab(modules) {
  const { seed, orchestrator, studio, classifier, photo } = modules;

  /** All-photoreal routing, both backends on, the default daily-look style, a face reference on the profile. */
  function createState({ appearance = APPEARANCE_LEAD, stylePreset = "daily_look", stylePrompt = "", negativePrompt = "", characterNegative = "", identity = "1girl, long black hair, brown eyes", mode = "photoreal" } = {}) {
    const state = structuredClone(seed.seedState);
    state.simulation = { ...state.simulation, id: SIM, realtimeImageEnabled: true, contentRating: "adult_19" };
    state.imageProfile = {
      ...state.imageProfile,
      simulationId: SIM,
      enabled: true,
      userRules: "",
      backendRouting: { mode, photorealCategories: ["portrait", "full_body", "scenery", "object", "group", "romance", "action", "fantasy"] }
    };
    state.novelAi = { ...state.novelAi, enabled: true, requestMode: "proxy" };
    state.photoreal = {
      ...state.photoreal,
      enabled: true,
      requestMode: "mock",
      stylePreset,
      stylePrompt,
      negativePrompt,
      faceReferenceEnabled: true,
      vramHandoff: "off",
      customWorkflow: ""
    };
    state.characters = [{ id: "char_yuna", simulationId: SIM, name: "유나", role: "", summary: "", relationship: "", currentMood: "calm" }];
    state.visualProfiles = [
      {
        id: "visual_char_yuna",
        simulationId: SIM,
        characterId: "char_yuna",
        displayName: "유나",
        positivePrompt: identity,
        negativePrompt: "",
        defaultOutfitPrompt: "",
        outfitPrompts: {},
        expressionPrompts: {},
        referenceImageAssetIds: [],
        defaultSafetyLevel: "safe",
        photorealPrompt: appearance,
        photorealNegativePrompt: characterNegative,
        profileImages: {
          photoreal: {
            backend: "photoreal",
            assetId: "profile_yuna",
            objectKey: `${SIM}/profile_yuna.png`,
            mimeType: "image/png",
            prompt: APPEARANCE_LEAD,
            width: 960,
            height: 1280,
            selectedAt: "2026-10-14T00:00:00.000Z"
          }
        }
      }
    ];
    state.messages = [];
    state.memoryEvents = [];
    state.imageJobs = [];
    state.imageAssets = [];
    return state;
  }

  function createCue(prompt, { base = ["1girl", "solo", "indoors"], frame = FRAMES.cowboy_shot } = {}) {
    return {
      shouldGenerate: true,
      reason: "safety corpus",
      characters: ["char_yuna"],
      tags: [],
      baseTags: base,
      characterPrompts: [{ characterId: "char_yuna", prompt }],
      scene: "safety corpus",
      visualContext: "",
      frame,
      kind: "scene"
    };
  }

  function plan(state, cue) {
    return orchestrator.planImageJob(state, "turn_safety", cue, [], true);
  }

  /** The planned job's verdict: locked to NovelAI, or the photo prompt it would send, and the caption it was routed on. */
  function verdict(job) {
    return {
      locked: job.routing?.locked === true && job.backend === "novelai",
      photoPrompt: job.backend === "photoreal" ? job.prompt : "",
      negativePrompt: job.backend === "photoreal" ? job.negativePrompt : "",
      composedCaption: (job.providerPayload?.characterPrompts ?? []).map((entry) => entry.prompt).join(", "),
      signals: job.routing?.signals ?? []
    };
  }

  return {
    ...modules,
    classifier,
    photo,
    createState,
    createCue,
    plan,
    verdict,
    // Full body, so the frame filter never removes the probe before it is routed.
    tagCut: (probe, options) => verdict(plan(createState(), createCue(`standing, smile, ${probe}`, { frame: FRAMES.full_body, ...options }))),
    appearanceCut: (probe) => verdict(plan(createState({ appearance: `${APPEARANCE_LEAD}, ${probe}` }), createCue("sitting, smile", { base: ["1girl", "solo", "cafe"] }))),
    styleCut: (probe) =>
      verdict(plan(createState({ stylePreset: "custom", stylePrompt: `A candid smartphone photograph, ${probe}. Natural light` }), createCue("sitting, smile", { base: ["1girl", "solo", "cafe"] }))),
    screen: (text) => orchestrator.screenPhotorealText(createState(), text),
    studio: (prompt) => {
      try {
        const jobs = studio.planCharacterProfileTestJobs(createState(), { characterId: "char_yuna", backend: "photoreal", prompt, negativePrompt: "", variant: "portrait", count: 1 });
        return { refused: false, ...verdict(jobs[0]) };
      } catch (error) {
        return { refused: true, message: describeError(error) };
      }
    },
    lockReason: (classification) =>
      classification.category === "explicit"
        ? "explicit"
        : classification.minorCoded
          ? "minor"
          : classification.schoolCoded && classification.suggestive
            ? "school+suggestive"
            : classification.unverifiable
              ? "unverifiable"
              : undefined
  };
}

/** Whether the probe's own words were printed into a photo prompt, read the way an encoder would read them. */
function printed(lab, photoPrompt, probe) {
  if (!photoPrompt) {
    return false;
  }
  const prompt = lab.classifier.normalizeSceneTag(photoPrompt);
  const marker = lab.classifier.normalizeSceneTag(probe);
  return marker.length >= 3 && prompt.includes(marker);
}

// ---------------------------------------------------------------------------------------------------------------
// Must lock
// ---------------------------------------------------------------------------------------------------------------

/**
 * Every probe, in every place a photo prompt can carry it, ends locked to NovelAI or refused: a cut tag, a character's
 * photoreal appearance, a custom photoreal style, the free-text screen and the profile studio. A cut tag the NovelAI
 * composer already removed (it drops emoji-like symbols before anything is routed) never reaches either model's
 * prompt; it counts as held, and is reported apart so the count stays honest.
 */
function evaluateMustLock(lab, scope, probes) {
  corpusSizes.push({ name: scope.replace(/^floor\./u, ""), size: probes.length });
  const paths = {
    tag: (probe) => lab.tagCut(probe),
    appearance: (probe) => lab.appearanceCut(probe),
    style: (probe) => lab.styleCut(probe)
  };
  const placed = { tag: "as a cut tag", appearance: "inside a photoreal appearance", style: "inside a custom style" };
  for (const [name, run] of Object.entries(paths)) {
    const leaked = [];
    const removed = [];
    for (const probe of probes) {
      const result = run(probe);
      if (result.locked) {
        continue;
      }
      const inCaption = name === "tag" && (result.composedCaption.includes(probe) || printed(lab, result.composedCaption, probe));
      if (name === "tag" && !inCaption && !printed(lab, result.photoPrompt, probe)) {
        removed.push(probe);
        continue;
      }
      leaked.push(`${probe}${printed(lab, result.photoPrompt, probe) ? " (printed)" : ""}`);
    }
    assertCheck(
      scope,
      leaked.length === 0,
      `${probes.length - leaked.length - removed.length}/${probes.length} lock to NovelAI ${placed[name]} in all-photoreal mode${removed.length ? `, ${removed.length} never reach a prompt (${removed.join(" | ")})` : ""} (leaked: ${leaked.join(" | ") || "none"}).`
    );
  }
  const unscreened = probes.filter((probe) => !lab.screen(`A woman, ${probe}.`));
  assertCheck(scope, unscreened.length === 0, `${probes.length - unscreened.length}/${probes.length} are refused by the free-text screen (missed: ${unscreened.join(" | ") || "none"}).`);
  const studioPassed = probes.filter((probe) => !lab.studio(`${APPEARANCE_LEAD}, ${probe}`).refused);
  assertCheck(scope, studioPassed.length === 0, `${probes.length - studioPassed.length}/${probes.length} are refused by the profile studio's photoreal test (passed: ${studioPassed.join(" | ") || "none"}).`);
}

/**
 * A student identity is school-coded in a tag (the builder drops the word and the subject stays an adult), and
 * minor-coded in free text, where nothing is left to make the subject an adult.
 */
function evaluateStudentIdentity(lab, probes) {
  corpusSizes.push({ name: "student-identity", size: probes.length });
  const printedInTag = probes.filter((probe) => printed(lab, lab.tagCut(probe).photoPrompt, probe));
  assertCheck("floor.student", printedInTag.length === 0, `A student identity in a cut tag never reaches the photo prompt (printed: ${printedInTag.join(" | ") || "none"}).`);
  const printedInAppearance = probes.filter((probe) => printed(lab, lab.appearanceCut(probe).photoPrompt, probe));
  assertCheck(
    "floor.student",
    printedInAppearance.length === 0,
    `A student identity in a photoreal appearance is dropped from the photo prompt like the tag (printed: ${printedInAppearance.join(" | ") || "none"}).`
  );
  const unrefused = probes.filter((probe) => !lab.studio(`A Korean ${probe} with long black hair`).refused || !lab.screen(`A Korean ${probe} with long black hair.`));
  assertCheck("floor.student", unrefused.length === 0, `As free text it is minor-coded: the studio and the screen refuse it (passed: ${unrefused.join(" | ") || "none"}).`);
  const lockedStyles = probes.filter((probe) => lab.styleCut(probe).locked);
  assertCheck("floor.student", lockedStyles.length === probes.length, `In a custom style it locks the cut (${lockedStyles.length}/${probes.length}).`);
}

function evaluateSchoolSuggestive(lab, schoolTerms, suggestiveTerms) {
  corpusSizes.push({ name: "school x suggestive", size: schoolTerms.length * suggestiveTerms.length });
  const leaked = [];
  for (const school of schoolTerms) {
    for (const suggestive of suggestiveTerms) {
      if (!lab.tagCut(`${school}, ${suggestive}`, { frame: FRAMES.full_body }).locked) {
        leaked.push(`${school} + ${suggestive}`);
      }
    }
  }
  assertCheck("floor.school", leaked.length === 0, `Every school term with every suggestive term locks (leaked: ${leaked.join(" | ") || "none"}).`);
  const classify = (tags) => lab.classifier.classifyImageScene({ characterTagSets: [["1girl", ...tags]], contextTags: ["indoors"], registeredCharacterCount: 1 });
  const notCoded = schoolTerms.filter((school) => !classify([school]).schoolCoded);
  const lockedAlone = schoolTerms.filter((school) => lab.lockReason(classify([school])));
  assertCheck("floor.school", notCoded.length === 0, `Each term is school-coded (missed: ${notCoded.join(" | ") || "none"}).`);
  assertCheck("floor.school", lockedAlone.length === 0, `Alone, none locks: adults in a school setting are ordinary fiction (locked: ${lockedAlone.join(" | ") || "none"}).`);
  const disguised = classify(["plaid pleated skirt", "ribbon tie", "white blouse"]);
  assertCheck("floor.school", disguised.schoolCoded, `A plaid skirt with a ribbon tie is a school uniform without the word (${disguised.signals.join(" | ")}).`);
  assertCheck("floor.school", lab.tagCut("plaid pleated skirt, ribbon tie, lingerie", { frame: FRAMES.full_body }).locked, "…and locks with a suggestive term.");
}

/** Danbooru tags graded suggestive (adults may render them) or ambiguous postures (explicit only with a suggestive tag). */
function evaluateGradedDanbooru(lab, suggestiveTags, ambiguousTags) {
  corpusSizes.push({ name: "danbooru suggestive", size: suggestiveTags.length }, { name: "danbooru ambiguous", size: ambiguousTags.length });
  const classify = (tags) => lab.classifier.classifyImageScene({ characterTagSets: [["1girl", ...tags]], contextTags: ["indoors"], registeredCharacterCount: 1 });
  const notSuggestive = suggestiveTags.filter((tag) => {
    const result = classify([tag]);
    return !result.suggestive || result.category === "explicit";
  });
  assertCheck("floor.danbooru-graded", notSuggestive.length === 0, `Suggestive Danbooru tags are suggestive, not explicit (wrong: ${notSuggestive.join(" | ") || "none"}).`);
  const unlockedWithSchool = suggestiveTags.filter((tag) => !lab.lockReason(classify(["school uniform", tag])));
  assertCheck("floor.danbooru-graded", unlockedWithSchool.length === 0, `Each locks next to a school uniform (leaked: ${unlockedWithSchool.join(" | ") || "none"}).`);
  const explicitAlone = ambiguousTags.filter((tag) => classify([tag]).category === "explicit");
  const notExplicitWithLingerie = ambiguousTags.filter((tag) => classify([tag, "lingerie"]).category !== "explicit");
  assertCheck("floor.danbooru-graded", explicitAlone.length === 0, `Ambiguous postures are not explicit alone (flagged: ${explicitAlone.join(" | ") || "none"}).`);
  assertCheck("floor.danbooru-graded", notExplicitWithLingerie.length === 0, `…and are explicit next to lingerie (missed: ${notExplicitWithLingerie.join(" | ") || "none"}).`);
}

/** A bare chest is topless on a woman and ordinary on a man; with nobody named it is at least suggestive. */
function evaluateBareChest(lab, femaleProbes, maleProbes) {
  corpusSizes.push({ name: "bare chest", size: femaleProbes.length + maleProbes.length });
  const leaked = femaleProbes.filter((probe) => !lab.tagCut(probe).locked || !lab.screen(`${probe}.`));
  assertCheck("floor.bare-chest", leaked.length === 0, `On a woman it locks, as a tag and as free text (leaked: ${leaked.join(" | ") || "none"}).`);
  const locked = maleProbes.filter((probe) => {
    const asTag = lab.classifier.classifyImageScene({ characterTagSets: [["1boy", probe]], contextTags: ["beach"], registeredCharacterCount: 1 });
    return lab.lockReason(asTag) || lab.screen(`${probe}.`);
  });
  assertCheck("floor.bare-chest", locked.length === 0, `On a man it does not (locked: ${locked.join(" | ") || "none"}).`);
  const nobody = lab.classifier.classifyImageScene({ characterTagSets: [["shirtless"]], contextTags: ["beach"], registeredCharacterCount: 1 });
  assertCheck("floor.bare-chest", nobody.suggestive && nobody.category !== "explicit", `With nobody named it is suggestive (${nobody.category}, suggestive ${nobody.suggestive}).`);
  const styled = lab.styleCut("shirtless");
  assertCheck("floor.bare-chest", styled.locked, "A style that says \"shirtless\" over a woman's cut locks: the gender comes from the cut when the style names nobody.");
  const twoPeople = lab.classifier.classifyImageScene({
    characterTagSets: [],
    contextTags: [],
    safetyTexts: ["On the left, an adult man, shirtless; on the right, an adult woman, smiling."],
    registeredCharacterCount: 0
  });
  assertCheck("floor.bare-chest", twoPeople.category !== "explicit", `In a finished prompt the bare chest belongs to the person it is written about (${twoPeople.category}).`);
}

/** NovelAI's negative-weight idioms read as what they ask NovelAI for, and never become a photo negative. */
function evaluateNegativeWeights(lab, probes) {
  corpusSizes.push({ name: "negative weights", size: probes.length });
  const leaked = probes.filter((probe) => !lab.tagCut(probe).locked);
  assertCheck("floor.negative-weights", leaked.length === 0, `Undressing, uncensoring and age-down weights lock the cut (leaked: ${leaked.join(" | ") || "none"}).`);
  const built = lab.photo.buildPhotorealPrompt({
    characters: [{ tags: ["1girl", "smile", ...probes, "-1::hat::"], identityTags: [] }],
    contextTags: [],
    style: { lead: "", tail: "", negative: "blurry" },
    referenceSyntax: "none"
  });
  const negatives = built.negativePrompt.split(/,\s*/u);
  const carried = probes.map((probe) => lab.classifier.normalizeSceneTag(probe)).filter((term) => negatives.includes(term));
  assertCheck(
    "floor.negative-weights",
    carried.length === 0 && negatives.includes("hat") && negatives.includes("blurry"),
    `None of them becomes a photo negative; an accessory still does (${built.negativePrompt}).`
  );
}

/** A Hangul code point in a segment no longer hides the Latin words around it from the screen. */
function evaluateHangulSegments(lab) {
  const toneMark = lab.photo.selectPhotoTextForScreening("A naked woman 〮 with long black hair");
  assertCheck("floor.hangul", toneMark.some((segment) => /naked/u.test(segment)), `A Hangul tone mark does not skip the segment (${JSON.stringify(toneMark)}).`);
  const mixed = lab.photo.selectPhotoTextForScreening("긴 머리의 nude 여성, soft light");
  assertCheck("floor.hangul", mixed.includes("nude") && !mixed.some((segment) => /[가-힣]/u.test(segment)), `A mixed segment is screened without its Korean (${JSON.stringify(mixed)}).`);
  assertCheck("floor.hangul", Boolean(lab.screen("A naked woman 〮")) && Boolean(lab.screen("긴 머리의 nude 여성")), "Both are refused by the screen.");
  const korean = lab.screen("검은 머리의 20대 여성, 카페에 앉아 있다");
  assertCheck("floor.hangul", korean === undefined, "An all-Korean appearance still passes (the builder drops it and falls back to the saved tags).");
}

/**
 * Free text that must stay photoreal on every path the must-lock probes take: as a cut tag, inside a photoreal
 * appearance, through the free-text screen and the profile studio. The cost side of a class.
 */
function evaluateMustPass(lab, scope, probes, what) {
  corpusSizes.push({ name: `${scope.replace(/^floor\./u, "")} (clean)`, size: probes.length });
  const locked = probes.filter((probe) => lab.tagCut(probe).locked || lab.appearanceCut(probe).locked || lab.screen(`A woman, ${probe}.`) || lab.studio(`${APPEARANCE_LEAD}, ${probe}`).refused);
  assertCheck(scope, locked.length === 0, `${probes.length - locked.length}/${probes.length} stay photoreal: ${what} (locked: ${locked.join(" | ") || "none"}).`);
}

/**
 * A photographer's style line: the style screen hands it over one comma segment at a time, so "f/8" arrives without
 * "35mm" and "ISO 100" beside it and used to read as a stated age ("f/8" = female, 8), locking every cut.
 */
function evaluateCameraStyle(lab) {
  const scope = "sfw.camera-style";
  for (const stylePrompt of ["Shot on a Sony A7, 35mm, f/8, ISO 100. Natural window light.", "Leica Q2, f/14, 28mm, soft daylight", "50mm lens, f/11, film grain"]) {
    const job = lab.verdict(lab.plan(lab.createState({ stylePreset: "custom", stylePrompt }), lab.createCue("sitting, smile", { base: ["1girl", "solo", "cafe"] })));
    assertCheck(scope, !job.locked && Boolean(job.photoPrompt), `a camera style with an aperture stays photoreal: ${stylePrompt}`, job.signals.join(" | "));
  }
  const job = lab.verdict(lab.plan(lab.createState({ stylePreset: "custom", stylePrompt: "A candid photo, f/14, girl in a bikini" }), lab.createCue("sitting, smile", { base: ["1girl", "solo", "cafe"] })));
  assertCheck(scope, job.locked, "without camera words, f/14 next to a girl is still an age", job.signals.join(" | "));
}

/**
 * A studio pick typed in Korean (this app's language): the photo prompt drops Korean, so the typed text cannot vouch
 * for the image — the rendered photo prompt recorded beside it (photoPrompt) does. A tainted typed prompt still
 * refuses the reference even when the rendered one is clean, and a tainted rendered one refuses a Korean typed one.
 */
function evaluateKoreanPickRecord(lab, studio) {
  const scope = "reference.korean-pick";
  const state = lab.createState();
  const record = (typed, rendered) =>
    studio.createProfileImageRecord(
      { id: "asset_pick", prompt: rendered, mimeType: "image/png", providerMetadata: { width: 960, height: 1280 } },
      { backend: "photoreal", objectKey: `${SIM}/profile_pick.png`, prompt: typed, assetId: "profile_pick" }
    );
  const typed = "긴 검은 머리의 20대 후반 한국 여성, 미소";
  const [job] = studio.planCharacterProfileTestJobs(state, { characterId: "char_yuna", backend: "photoreal", prompt: typed, negativePrompt: "", variant: "portrait", count: 1 });
  const korean = record(typed, job.prompt);
  assertCheck(scope, typeof korean.photoPrompt === "string" && korean.photoPrompt === job.prompt, "the pick records the photo prompt it was rendered from");
  assertCheck(scope, lab.orchestrator.screenPhotorealReferenceImage(state, korean) === undefined, "a Korean-typed pick vouches for its image through that photo prompt");
  assertCheck(
    scope,
    lab.orchestrator.screenPhotorealReferenceImage(state, { objectKey: korean.objectKey, prompt: typed })?.kind === "unreadable_record",
    "a Korean-typed record without the photo prompt is still unreadable"
  );
  assertCheck(
    scope,
    lab.orchestrator.screenPhotorealReferenceImage(state, record("A Korean girl who is 14, long black hair", "A candid photo of a Korean woman in her late twenties."))?.kind === "floor",
    "a tainted typed prompt refuses the reference even beside a clean photo prompt"
  );
  assertCheck(
    scope,
    lab.orchestrator.screenPhotorealReferenceImage(state, record("긴 검은 머리", "A candid photo. A Korean girl, 12 years old."))?.kind === "floor",
    "a tainted photo prompt refuses a Korean-typed pick"
  );
}

/**
 * The shipped adult seed dresses most of its cast in "open-crotch" garments. A garment cut to expose the body is
 * explicit, so every cut that shows the hips of such a character locks; a crop that hides them still renders
 * photoreal, and no photo prompt ever names the garment.
 */
function evaluateExposingSeedOutfits(lab) {
  const base = lab.seed.hydrateState(structuredClone(lab.womanLife.womanLifeSeedState));
  const state = lab.seed.hydrateState({
    ...base,
    photoreal: { ...base.photoreal, enabled: true, requestMode: "mock" },
    novelAi: { ...base.novelAi, enabled: true },
    imageProfile: { ...base.imageProfile, backendRouting: { mode: "mixed", photorealCategories: ["portrait", "full_body"] } }
  });
  const exposing = state.visualProfiles.filter((profile) => /open.?crotch/iu.test(String(profile.defaultOutfitPrompt ?? "")));
  const plan = (profile, frame) =>
    lab.plan(state, {
      shouldGenerate: true,
      reason: "safety corpus",
      scene: "safety corpus",
      tags: [],
      baseTags: ["1girl", "solo", "indoors", "bedroom", "mirror"],
      frame,
      kind: "scene",
      characters: [profile.characterId],
      characterPrompts: [{ characterId: profile.characterId, prompt: "standing, looking at mirror, smile" }]
    });
  const unlocked = exposing.filter((profile) => !plan(profile, FRAMES.full_body).routing?.locked);
  assertCheck(
    "floor.garments",
    exposing.length >= 8 && unlocked.length === 0,
    `Every women's-life character whose saved outfit is an open-crotch garment locks its full-body mirror cut (${exposing.length - unlocked.length}/${exposing.length}; unlocked: ${unlocked.map((profile) => profile.displayName).join(", ") || "none"}).`
  );
  const portraits = exposing.map((profile) => plan(profile, FRAMES.upper_body));
  const leaked = portraits.filter((job) => job.backend === "photoreal" && /crotch/iu.test(job.prompt));
  assertCheck(
    "floor.garments",
    portraits.some((job) => job.backend === "photoreal") && leaked.length === 0,
    `An upper-body cut hides the garment and still renders photoreal, and no photo prompt names it (${portraits.filter((job) => job.backend === "photoreal").length}/${portraits.length} photoreal, ${leaked.length} naming it).`
  );
}

/**
 * Pictographs: the floor counts the sexual, child and school ones (must-lock above); every other one is stripped from
 * the photo prompt, which the encoders would otherwise read, and a cut made of nothing else still renders.
 */
function evaluatePictographStripping(lab) {
  const pictograph = /\p{Extended_Pictographic}/u;
  const built = lab.photo.buildPhotorealPrompt({
    characters: [{ appearance: "A Korean woman 🙂 with long black hair ☕", tags: ["1girl", "smile", "🌸 hairclip"], identityTags: ["1girl"] }],
    contextTags: ["cafe ✨"],
    style: { lead: "A candid photograph 📸", tail: "Soft light 🌤️", negative: "blurry 🚫" },
    referenceSyntax: "none"
  });
  assertCheck(
    "floor.emoji",
    !pictograph.test(built.prompt) && !pictograph.test(built.negativePrompt) && /long black hair/u.test(built.prompt) && /hairclip/u.test(built.prompt),
    `Pictographs never reach the photo prompt or its negative; the words around them stay (${built.prompt.slice(0, 160)}).`
  );
  const plain = lab.appearanceCut("🙂 ☕");
  assertCheck("floor.emoji", !plain.locked && plain.photoPrompt && !pictograph.test(plain.photoPrompt), "An ordinary emoji neither locks a cut nor reaches its photo prompt.");
  const screened = lab.photo.selectPhotoTextForScreening("A Korean woman, 🍆💦, 🙂");
  assertCheck(
    "floor.emoji",
    screened.includes("🍆💦") && !screened.includes("🙂"),
    `The screen keeps an emoji-only segment the floor reads and drops one it does not (${JSON.stringify(screened)}).`
  );
}

/**
 * A school uniform or school life written without the word "school" (a short pleated skirt with a ribbon tie, a
 * blazer or knee socks; prom, detention, the nurse's office, homeroom, a yearbook) locks next to a suggestive term,
 * and an office outfit or a detention centre does not.
 */
function evaluateSchoolComposite(lab, combos, clean) {
  corpusSizes.push({ name: "school composite", size: combos.length + clean.length });
  const leaked = combos.filter((combo) => !lab.tagCut(combo, { frame: FRAMES.full_body }).locked);
  assertCheck("floor.school-composite", leaked.length === 0, `${combos.length - leaked.length}/${combos.length} school-without-the-word combinations lock (leaked: ${leaked.join(" | ") || "none"}).`);
  const locked = clean.filter((combo) => lab.tagCut(combo, { frame: FRAMES.full_body }).locked);
  assertCheck("floor.school-composite", locked.length === 0, `Office wear and a detention centre do not (locked: ${locked.join(" | ") || "none"}).`);
  // A cut tag of emoji never reaches routing (the NovelAI composer drops it), so the school pictograph is read as text.
  assertCheck("floor.school-composite", Boolean(lab.screen("A woman in lingerie, 🏫.")), "A school pictograph next to lingerie is refused by the free-text screen.");
}

/**
 * A style or other text laid over a cut has no subject of its own: its bare chest belongs to the cut's figure. Over a
 * woman's cut it locks whatever the sentence compares itself to; over a man's cut it is ordinary.
 */
function evaluateGlobalBareChest(lab, probes) {
  corpusSizes.push({ name: "global bare chest", size: probes.length });
  const leaked = probes.filter((probe) => !lab.styleCut(probe).locked);
  assertCheck("floor.global-bare-chest", leaked.length === 0, `In a custom style over a woman's cut every one locks (leaked: ${leaked.join(" | ") || "none"}).`);
  const lockedOnMan = probes.filter((probe) =>
    lab.lockReason(
      lab.classifier.classifyImageScene({ characterTagSets: [["1boy", "smile"]], contextTags: ["beach"], safetyTexts: [`A candid smartphone photograph. ${probe}.`], registeredCharacterCount: 1 })
    )
  );
  assertCheck("floor.global-bare-chest", lockedOnMan.length === 0, `Over a man's cut none does (locked: ${lockedOnMan.join(" | ") || "none"}).`);
}

// ---------------------------------------------------------------------------------------------------------------
// Negative prompts
// ---------------------------------------------------------------------------------------------------------------

/**
 * The photo negative is an allow-list of quality, artefact, medium and look terms. Everything a red team used to steer
 * the sampler (garments in any language or spelling, adulthood, body words, safety words) is dropped, and every
 * preset's own negative survives intact.
 */
function evaluateNegativeAllowList(lab, steering) {
  corpusSizes.push({ name: "negative steering", size: steering.length });
  const kept = lab.photo.sanitizePhotorealNegativePrompt(steering.join(", "));
  assertCheck("negatives", kept === "", `Nothing outside the vocabulary survives (kept: ${kept || "none"}).`);
  const presets = Object.entries(lab.photo.PHOTOREAL_STYLE_PRESETS).filter(([, preset]) => lab.photo.sanitizePhotorealNegativePrompt(preset.negative) !== preset.negative);
  assertCheck("negatives", presets.length === 0, `Every preset's negative survives intact (changed: ${presets.map(([id]) => id).join(", ") || "none"}).`);
  const job = lab.verdict(lab.plan(lab.createState({ negativePrompt: steering.join(", "), characterNegative: "denim, Kleidung, 25yo, breasts" }), lab.createCue("standing, smile")));
  const daily = lab.photo.PHOTOREAL_STYLE_PRESETS.daily_look.negative;
  assertCheck(
    "negatives",
    !job.locked && job.negativePrompt === daily,
    `A planned photo job under those settings and character negatives carries exactly the preset's negative (${job.negativePrompt}).`
  );
}

function evaluateNegativeSanitizer(lab, dropTerms, keepTerms) {
  corpusSizes.push({ name: "negatives", size: dropTerms.length + keepTerms.length });
  const sanitized = lab.photo.sanitizePhotorealNegativePrompt([...dropTerms, ...keepTerms].join(", ")).split(/,\s*/u);
  const kept = dropTerms.filter((term) => sanitized.includes(lab.classifier.normalizeSceneTag(term)) || sanitized.includes(term));
  const lost = keepTerms.filter((term) => !sanitized.includes(term));
  assertCheck("negatives", kept.length === 0, `Clothing, covering, safety, adulthood and adult-build terms are dropped (kept: ${kept.join(" | ") || "none"}).`);
  assertCheck("negatives", lost.length === 0, `Quality terms, accessories and the safety negatives that keep a picture clean stay (lost: ${lost.join(" | ") || "none"}).`);

  // The same holds wherever a negative enters a planned job: the settings, the character, a cut's negative weight.
  const state = lab.createState({ negativePrompt: dropTerms.join(", "), characterNegative: "apparel, fabric, covered, sfw, adult, mature" });
  const job = lab.plan(state, lab.createCue("standing, smile, -1::apparel::"));
  const planned = lab.verdict(job);
  const negatives = planned.negativePrompt.split(/,\s*/u);
  const reached = dropTerms.map((term) => lab.classifier.normalizeSceneTag(term)).filter((term) => negatives.includes(term));
  assertCheck("negatives", !planned.locked ? reached.length === 0 : true, `A planned photo job's negative carries none of them (reached: ${reached.join(" | ") || "none"}).`);
  const clean = lab.verdict(lab.plan(lab.createState({ negativePrompt: dropTerms.join(", "), characterNegative: "apparel, fabric, covered, sfw, adult, mature" }), lab.createCue("standing, smile")));
  const cleanNegatives = clean.negativePrompt.split(/,\s*/u);
  assertCheck(
    "negatives",
    clean.photoPrompt && dropTerms.every((term) => !cleanNegatives.includes(lab.classifier.normalizeSceneTag(term))) && cleanNegatives.includes("watermark"),
    `A clean cut under an anti-safety negative still renders photoreal, with only the quality negatives left (${clean.negativePrompt}).`
  );
}

// ---------------------------------------------------------------------------------------------------------------
// SFW: what the floor must not cost
// ---------------------------------------------------------------------------------------------------------------

/** The red team's plain SFW sweep: ordinary daily-life cuts and sentences, no deliberately chosen trap words. */
function evaluatePlainSfw(lab) {
  const garments = ["white t-shirt", "black hoodie", "denim jacket", "beige trench coat", "knit cardigan", "striped shirt", "linen dress", "pleated skirt", "wide-leg jeans", "cargo pants", "leather jacket", "puffer jacket", "turtleneck sweater", "polo shirt", "blazer", "maxi dress", "shorts", "sneakers", "loafers", "ankle boots", "sandals", "baseball cap", "beanie", "tote bag", "backpack", "sunglasses", "wristwatch", "silver earrings", "scarf", "raincoat"];
  const places = ["cafe", "office", "subway station", "bookstore", "park", "beach", "kitchen", "living room", "airport", "hotel lobby", "rooftop bar", "convenience store", "street market", "museum", "library", "gym", "train", "bus stop", "restaurant", "bakery", "flower shop", "laundromat", "supermarket", "art gallery", "riverside", "mountain trail", "campsite", "ski resort", "amusement park", "aquarium"];
  const actions = ["holding a coffee cup", "reading a book", "looking at phone", "laughing", "walking", "sitting on a bench", "typing on laptop", "carrying shopping bags", "waiting for the train", "eating noodles", "drinking iced americano", "taking a selfie", "hailing a taxi", "stretching", "jogging", "cooking pasta", "watering plants", "playing guitar", "painting on canvas", "browsing clothes", "checking a map", "pushing a shopping cart", "tying shoelaces", "leaning on railing", "writing in a notebook"];
  const subjects = ["A woman in her late twenties", "A man in his thirties", "A young office worker", "A college student", "Two friends", "A couple", "A barista", "A tourist", "A businesswoman", "A retired man", "A father and his adult son", "A mother and her adult daughter", "A group of coworkers", "A woman with short hair", "A man with glasses"];
  const scenes = [
    "sipping coffee by the window of a quiet cafe", "browsing paperbacks in a cozy bookstore", "waiting on the subway platform during rush hour", "laughing over dinner at a Korean barbecue restaurant",
    "walking a golden retriever through the park", "typing on a laptop in a coworking space", "shopping for vegetables at a street market", "taking photos of cherry blossoms along the river",
    "checking in at a hotel front desk", "stretching on a yoga mat in a sunny studio", "cooking pasta in a small apartment kitchen", "reading a newspaper on a park bench",
    "carrying grocery bags up the stairs", "hiking a mountain trail in autumn", "sitting on the beach at sunset", "riding the bus home in the rain",
    "choosing flowers at a flower shop", "watching the city lights from a rooftop bar", "folding laundry at a laundromat", "painting a canvas in an art studio",
    "practicing guitar in the living room", "playing tennis on a clay court", "eating ramen at a counter seat", "jogging along the Han River in the morning",
    "giving a presentation in a meeting room", "studying for exams in the library", "looking at paintings in a museum", "pushing a cart through the supermarket",
    "boarding a plane at the airport gate", "warming up by a campfire"
  ];
  const tagCuts = Array.from({ length: 170 }, (_, i) => [garments[i % garments.length], garments[(i * 7 + 3) % garments.length], places[(i * 3) % places.length], actions[(i * 5) % actions.length]].join(", "));
  const sentences = Array.from({ length: 165 }, (_, i) => `${subjects[i % subjects.length]} wearing a ${garments[(i * 11) % garments.length]}, ${scenes[(i * 7) % scenes.length]}.`);
  corpusSizes.push({ name: "plain SFW", size: tagCuts.length + sentences.length });

  const lockedCuts = tagCuts.filter((tags) => lab.verdict(lab.plan(lab.createState({ appearance: "" }), lab.createCue(tags, { base: ["1girl", "solo"] }))).locked);
  assertCheck("sfw.plain", lockedCuts.length === 0, `${tagCuts.length - lockedCuts.length}/${tagCuts.length} plain SFW tag cuts render photoreal through the planner (locked: ${lockedCuts.join(" | ") || "none"}).`);
  const lockedSentences = sentences.filter((sentence) => {
    const classification = lab.classifier.classifyImageScene({ characterTagSets: [["1girl"]], contextTags: [], safetyTexts: [sentence], registeredCharacterCount: 1 });
    return lab.lockReason(classification) || lab.screen(sentence);
  });
  assertCheck("sfw.plain", lockedSentences.length === 0, `${sentences.length - lockedSentences.length}/${sentences.length} plain SFW sentences pass the floor and the screen (locked: ${lockedSentences.join(" | ") || "none"}).`);
  const refusedInStudio = sentences.filter((_, index) => index % 5 === 0).filter((sentence) => lab.studio(sentence).refused);
  assertCheck("sfw.plain", refusedInStudio.length === 0, `A sample of them passes the profile studio (refused: ${refusedInStudio.join(" | ") || "none"}).`);
}

/**
 * The false friends this floor was taught to tell apart, each excepted only for the one neighbour that makes it
 * something else ("baby carrots", "a 12-year-old whisky", "an elementary school teacher", "Node.js", "nude nails"),
 * and the Latin letters NFD does not decompose.
 */
function evaluateFalseFriendFixes(lab, probes, latinFolds) {
  corpusSizes.push({ name: "false friends", size: probes.length + latinFolds.length });
  const locked = probes.filter((probe) => {
    const asTag = lab.classifier.classifyImageScene({ characterTagSets: [[probe]], contextTags: ["indoors"], registeredCharacterCount: 1 });
    const asProse = lab.classifier.classifyImageScene({ characterTagSets: [], contextTags: [], safetyTexts: [probe], registeredCharacterCount: 0 });
    return lab.lockReason(asTag) || lab.lockReason(asProse);
  });
  assertCheck("sfw.false-friends", locked.length === 0, `${probes.length - locked.length}/${probes.length} false friends pass as a tag and as free text (locked: ${locked.join(" | ") || "none"}).`);
  const unverifiable = latinFolds.filter((probe) => lab.classifier.hasUnverifiableScript(probe) || lab.screen(`A morning walk near ${probe}.`));
  assertCheck("sfw.false-friends", unverifiable.length === 0, `Latin letters without a decomposition fold and stay verifiable (locked: ${unverifiable.join(" | ") || "none"}).`);
  const folded = lab.classifier.canonicalizeSceneText("Øresund, Königstraße, Kadıköy, Łódź, Đà Nẵng");
  assertCheck("sfw.false-friends", folded === "Oresund, Konigstrasse, Kadikoy, Lodz, Da Nang", `They fold to the letters an encoder reads (${folded}).`);
  const guarded = ["nude heels", "nude stockings", "baby oil on a baby", "a 12-year-old cat girl", "an elementary school teacher and an elementary school student", "a girl aged 12 years single malt"];
  const unguarded = guarded.filter((probe) => !lab.lockReason(lab.classifier.classifyImageScene({ characterTagSets: [[probe]], contextTags: [], registeredCharacterCount: 1 })));
  assertCheck("sfw.false-friends", unguarded.length === 0, `Each exception covers its one neighbour, never the rest of the term (leaked: ${unguarded.join(" | ") || "none"}).`);
}

/**
 * The shipped adult seed (여성의 삶). Every character in ordinary cuts: a cut may lock only when the character's own
 * saved identity or outfit is explicit (that text is injected into every cut), never because of the floor's reading
 * of ordinary vocabulary. Its futanari identity trait in particular must not lock a clothed cut.
 */
function evaluateWomanLifeSeed(lab) {
  const base = lab.seed.hydrateState(structuredClone(lab.womanLife.womanLifeSeedState));
  const state = lab.seed.hydrateState({
    ...base,
    photoreal: { ...base.photoreal, enabled: true, requestMode: "mock" },
    novelAi: { ...base.novelAi, enabled: true },
    imageProfile: { ...base.imageProfile, backendRouting: { mode: "photoreal", photorealCategories: [] } }
  });
  const cuts = [
    ["cafe portrait", "smile, sitting, holding cup", ["1girl", "solo", "indoors", "cafe"], FRAMES.upper_body],
    ["street full body", "standing, walking, looking at viewer", ["1girl", "solo", "outdoors", "street", "daytime"], FRAMES.full_body],
    ["face close-up", "looking at viewer, calm", ["1girl", "solo", "indoors"], FRAMES.close_up],
    ["office", "standing, holding papers, serious", ["1girl", "solo", "indoors", "office"], FRAMES.cowboy_shot],
    ["bedroom mirror", "standing, looking at mirror, holding clothes", ["1girl", "solo", "indoors", "bedroom", "mirror"], FRAMES.cowboy_shot]
  ];
  let planned = 0;
  const unexplained = [];
  let locked = 0;
  for (const profile of state.visualProfiles) {
    const savedExplicit = [profile.positivePrompt, profile.defaultOutfitPrompt]
      .flatMap((text) => String(text ?? "").split(","))
      .some((tag) => lab.classifier.isExplicitSceneTag(tag));
    for (const [label, prompt, baseTags, frame] of cuts) {
      const job = lab.plan(state, {
        shouldGenerate: true,
        reason: "safety corpus",
        scene: "safety corpus",
        tags: [],
        baseTags,
        frame,
        kind: "scene",
        characters: [profile.characterId],
        characterPrompts: [{ characterId: profile.characterId, prompt }]
      });
      planned += 1;
      if (job.routing?.locked) {
        locked += 1;
        if (!savedExplicit) {
          unexplained.push(`${profile.displayName} / ${label}: ${job.routing.signals.join(" | ")}`);
        }
      }
    }
  }
  corpusSizes.push({ name: "women's-life cuts", size: planned });
  assertCheck(
    "sfw.womanlife",
    unexplained.length === 0,
    `${planned - locked}/${planned} ordinary cuts of the shipped adult seed render photoreal; every locked one belongs to a character whose saved identity or outfit is itself explicit (unexplained: ${unexplained.join(" || ") || "none"}).`
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------------------------

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
