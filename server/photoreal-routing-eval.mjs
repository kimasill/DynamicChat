import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// Scene classification and backend routing for the photoreal backend.
//
// The router is the safety floor (explicit, minor-coded and school-coded + suggestive cuts never render
// photoreal), so most of this suite is about the floor holding: in every routing mode, with every availability
// combination, and when a manual regeneration names the photoreal backend outright. The rest pins the category
// matrix and its precedence, which decides what "mixed" mode actually sends where.

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const vite = await createServer({
  root: workspaceRoot,
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom"
});

// ---------------------------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------------------------

const FRAMES = {
  close_up: { shot: "close_up", viewpoint: "front", visibleRegions: ["head"] },
  face_focus: { shot: "face_focus", viewpoint: "front", visibleRegions: ["head"] },
  upper_body: { shot: "upper_body", viewpoint: "front", visibleRegions: ["head", "torso"] },
  cowboy_shot: { shot: "cowboy_shot", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs"] },
  full_body: { shot: "full_body", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs", "feet"] },
  wide_shot: { shot: "wide_shot", viewpoint: "front", visibleRegions: ["head", "torso", "hips", "legs", "feet"] }
};

/** A classifier input shaped like planImageJob builds it from a composed prompt plan. */
function cut({ figures = [], context = [], frame, registered, kind } = {}) {
  return {
    cueKind: kind,
    characterTagSets: figures,
    contextTags: context,
    frame: typeof frame === "string" ? FRAMES[frame] : frame,
    registeredCharacterCount: registered ?? figures.length
  };
}

/** Classifications reused by the routing groups. Built through the real classifier, not hand-written. */
/**
 * Body-identity traits (futanari and synonyms) live in a character's saved identity and are injected into every
 * cut. Measured on the shipped adult seed: treating them as explicit pinned 16/16 cuts of a real turn to NovelAI,
 * clothed cafe portraits included. They are identity; what a cut SHOWS is still decided by the explicit lexicon.
 */
function evaluateIdentityTraits(classifier, routing, seed) {
  const { classifyImageScene } = classifier;
  const portrait = classifyImageScene({
    characterTagSets: [["1girl, futanari, long hair, smile, sitting, holding cup, white blouse"]],
    contextTags: ["indoors", "cafe", "upper body"],
    frame: { shot: "upper_body", viewpoint: "front", visibleRegions: ["head", "torso"] },
    registeredCharacterCount: 1
  });
  assertCheck("identity-traits", portrait.category === "portrait", `A clothed cafe portrait of a character whose identity carries 'futanari' is a portrait, not explicit (got ${portrait.category}).`);
  assertCheck("identity-traits", portrait.signals.includes("identity-trait:futanari"), "The trait is reported as an identity-trait signal for the inspector.");
  for (const [label, tags] of [
    ["nude", "1girl, futanari, nude, lying on bed"],
    ["erection", "1girl, futanari, erection, standing"],
    ["penis", "1girl, futanari, penis, panties"],
    ["sex act", "1girl, futanari, sex from behind"]
  ]) {
    const result = classifyImageScene({ characterTagSets: [[tags]], contextTags: ["bedroom"], registeredCharacterCount: 1 });
    assertCheck("identity-traits", result.category === "explicit", `With a trait character, a cut that SHOWS ${label} is still explicit (got ${result.category}).`);
  }
  for (const synonym of ["futa", "dickgirl", "newhalf", "intersex"]) {
    const result = classifyImageScene({ characterTagSets: [[`1girl, ${synonym}, smile`]], contextTags: ["park"], registeredCharacterCount: 1 });
    assertCheck("identity-traits", result.category !== "explicit", `'${synonym}' alone is an identity trait, not explicit content.`);
  }
  const sheer = classifyImageScene({ characterTagSets: [["1girl, translucent bodysuit, standing"]], contextTags: ["street"], registeredCharacterCount: 1 });
  assertCheck("identity-traits", sheer.suggestive, "A translucent garment marks the cut suggestive (school + suggestive stays locked).");
  const schoolSheer = classifyImageScene({ characterTagSets: [["1girl, school uniform, sheer blouse"]], contextTags: ["classroom"], registeredCharacterCount: 1 });
  const state = structuredClone(seed);
  state.photoreal = { ...(state.photoreal ?? {}), enabled: true, requestMode: "mock" };
  state.novelAi = { ...state.novelAi, enabled: true };
  state.imageProfile = { ...state.imageProfile, backendRouting: { mode: "photoreal", photorealCategories: [] } };
  const decision = routing.decideImageBackend(state, schoolSheer);
  assertCheck("identity-traits", decision.backend === "novelai" && decision.locked === true, "School uniform + a sheer blouse is locked to NovelAI even in all-photoreal mode.");
  const traitDecision = routing.decideImageBackend(state, portrait);
  assertCheck("identity-traits", traitDecision.backend === "photoreal" && !traitDecision.locked, "In all-photoreal mode the trait character's clothed portrait renders photoreal.");
}

function createClassifications(classifyImageScene) {
  return {
    portrait: classifyImageScene(
      cut({
        figures: [["1girl", "black hair", "long hair", "smile", "looking at viewer", "white sweater"]],
        context: ["1girl", "solo", "indoors", "window light"],
        frame: "upper_body"
      })
    ),
    action: classifyImageScene(
      cut({
        figures: [["1girl", "running", "track jacket", "open mouth"]],
        context: ["1girl", "solo", "city street", "motion blur"],
        frame: "full_body"
      })
    ),
    explicit: classifyImageScene(
      cut({
        figures: [["1girl", "blush", "open mouth", "sex from behind"], ["1boy", "hetero"]],
        context: ["1girl", "1boy", "bedroom"],
        frame: "close_up"
      })
    ),
    minor: classifyImageScene(
      cut({ figures: [["1girl", "loli", "smile", "sundress"]], context: ["1girl", "solo", "park"], frame: "upper_body" })
    ),
    school: classifyImageScene(
      cut({
        figures: [["1girl", "school uniform", "serafuku", "smile", "holding book"]],
        context: ["1girl", "solo", "classroom"],
        frame: "upper_body"
      })
    ),
    schoolSuggestive: classifyImageScene(
      cut({
        figures: [["1girl", "school uniform", "skirt lift", "panties", "blush"]],
        context: ["1girl", "solo", "classroom"],
        frame: "cowboy_shot"
      })
    )
  };
}

const PHOTOREAL_SETTINGS = {
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

/**
 * A cloned seed with routing and availability set explicitly. state.photoreal is always written here — the seed
 * may not carry a default yet, and a test that leaned on one would pass or fail for the wrong reason.
 */
function createRoutingState(seedState, { mode, categories, photoreal = false, novelAi = false } = {}) {
  const state = structuredClone(seedState);
  state.novelAi = { ...state.novelAi, enabled: novelAi };
  state.photoreal = { ...PHOTOREAL_SETTINGS, enabled: photoreal };
  const imageProfile = { ...state.imageProfile };
  if (mode === undefined) {
    delete imageProfile.backendRouting;
  } else {
    imageProfile.backendRouting = {
      mode,
      photorealCategories: categories ?? ["portrait", "full_body", "scenery", "object"]
    };
  }
  state.imageProfile = imageProfile;
  return state;
}

const AVAILABILITY = [
  { label: "P on/N on", photoreal: true, novelAi: true },
  { label: "P on/N off", photoreal: true, novelAi: false },
  { label: "P off/N on", photoreal: false, novelAi: true },
  { label: "P off/N off", photoreal: false, novelAi: false }
];

const NOVELAI_SETUP_TEXT = "NovelAI 연동이 꺼져 있습니다. 개인 설정에서 켜면 이 컷부터 생성됩니다.";
const PHOTOREAL_SETUP_TEXT = "실사 이미지(ComfyUI) 연동이 꺼져 있습니다. 설정에서 켜면 이 컷부터 생성됩니다.";

// ---------------------------------------------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------------------------------------------

function evaluateTagNormalization({ normalizeSceneTag }) {
  const cases = [
    ["1.5::sex_from_behind::", "sex from behind", "NovelAI numeric weight and underscores"],
    ["{{Nude}}", "nude", "NovelAI brace emphasis and case"],
    ["[[loli]]", "loli", "NovelAI bracket de-emphasis"],
    ["(cleavage:1.2)", "cleavage", "Stable Diffusion `(tag:weight)` emphasis"],
    ["  School_Uniform ", "school uniform", "whitespace, underscores and case"],
    ["-1::nsfw::", "nsfw", "negative weight (the photoreal builder strips weights, so it would arrive as nsfw)"],
    ["::upper body::", "upper body", "weightless `::tag::` emphasis"],
    ["1.2::{nude}::", "nude", "nested weight syntax"],
    ["hatsune miku (cosplay)", "hatsune miku (cosplay)", "a bracketed qualifier inside a tag is kept"],
    // Canonical text: what a photo encoder reads as the plain word is the plain word to the lexicons too.
    ["ｎｕｄｅ", "nude", "fullwidth letters (NFKC)"],
    ["nu\u200bde", "nude", "a zero-width space inside a word"],
    ["nu\u00adde", "nude", "a soft hyphen inside a word"],
    ["lo\u200dli\ufe0f", "loli", "a zero-width joiner and a variation selector"],
    ["n\u0336u\u0336d\u0336e\u0336", "nude", "strike-through combining marks"],
    ["school-uniform", "school uniform", "hyphens read as spaces"],
    ["－１::school－uniform::", "school uniform", "a fullwidth negative weight and fullwidth hyphen"],
    ["Café", "cafe", "accents folded"],
    ["15 y.o.", "15 y.o", "closing punctuation dropped, inner dots kept"],
    // A mark on a digit, a letter-like symbol NFKC leaves alone, a Latin letter NFD does not decompose: each is the
    // plain character to an encoder, so each is the plain character here.
    ["1⃣2⃣ years old", "12 years old", "keycap marks on digits"],
    ["1️⃣4️⃣ years old", "14 years old", "emoji keycaps"],
    ["12́ years old", "12 years old", "an accent on a digit"],
    ["1̶2̶ years old", "12 years old", "strike-through on digits"],
    ["\u{1F17D}\u{1F170}\u{1F17A}\u{1F174}\u{1F173}", "naked", "negative squared capitals"],
    ["\u{1F15D}\u{1F150}\u{1F15A}\u{1F154}\u{1F153}", "naked", "negative circled capitals"],
    ["\u{1F1F3}\u{1F1FA}\u{1F1E9}\u{1F1EA}", "nude", "regional indicators"],
    ["Øresund Königstraße Kadıköy Łódź Đà Nẵng", "oresund konigstrasse kadikoy lodz da nang", "Latin letters without a decomposition"],
    ["naked 〮", "naked", "a stray Hangul tone mark"],
    ["niña", "ninya", "the ñ that separates a girl from the name Nina folds to \"ny\""],
    ["Nina", "nina", "the name stays the name"],
    ["jalapeño", "jalapeno", "every other ñ folds to n"]
  ];
  for (const [input, expected, label] of cases) {
    const actual = normalizeSceneTag(input);
    assertCheck("normalize", actual === expected, `${label}: ${JSON.stringify(input)} → ${JSON.stringify(actual)}`);
  }
}

/**
 * Review finding safety:lexicon-inflection-synonym-gaps: inflections, hyphenations and synonyms the lexicons missed.
 * Each must be caught, and the false friends next to them must not be.
 */
function evaluateLexiconGaps({ isExplicitSceneTag, isMinorCodedSceneTag, classifyImageScene }) {
  const explicit = [
    "nudes", "stripping", "striptease", "stripper", "intercourse", "making love", "lovemaking", "genitals", "genitalia", "dick",
    "boobs", "titties", "boobs out", "without clothes", "without any clothing", "no clothing", "wearing nothing", "sexual assault",
    "sexually assaulted", "non-consensual", "nonconsensual", "sexual-act", "strap-on", "orgasms", "ejaculates", "skinny dipping"
  ];
  const missedExplicit = explicit.filter((tag) => !isExplicitSceneTag(tag));
  assertCheck("lexicon.gaps", missedExplicit.length === 0, `Explicit inflections and synonyms are explicit (missed: ${missedExplicit.join(", ") || "none"}).`);

  const minor = [
    "teenaged", "teens", "teenage", "underaged", "under-age", "adolescent", "adolescents", "juvenile", "minor", "minors", "a minor",
    "she is a minor", "minor girl", "middle-schooler", "middle schoolers", "elementary-schooler", "elementary student", "sixth grader",
    "6th grade", "10th grader", "grade 6", "jk", "jc", "js", "jk uniform", "17 years of age", "fifteen years of age", "age: 15", "age 15",
    "aged fifteen", "age: fifteen", "at the age of 16", "under the age of 18", "a 25 year old and a 15 year old"
  ];
  const missedMinor = minor.filter((tag) => !isMinorCodedSceneTag(tag));
  assertCheck("lexicon.gaps", missedMinor.length === 0, `Minor-coded inflections, nouns, grades and stated ages are recognised (missed: ${missedMinor.join(", ") || "none"}).`);

  // Decided per case: "minor" codes a person only as a noun; "high school" stays school-coded (spec §2.3: adults cast
  // as high school students are the genre's norm), so a reunion of adults is not minor-coded — it locks only with a
  // suggestive term, like any school-coded cut.
  const friends = [
    "minor details", "a minor injury", "the minor key", "minor scar", "a sixth sense", "aged leather", "aged leather jacket",
    "high school reunion photo of adults", "color grade", "first-person view", "age 18+", "aged 30", "a 25-year-old woman", "jkt48 poster"
  ];
  const wronglyMinor = friends.filter((tag) => isMinorCodedSceneTag(tag));
  assertCheck("lexicon.gaps", wronglyMinor.length === 0, `Adjective "minor", "sixth sense", "aged leather", adult ages are not minor-coded (flagged: ${wronglyMinor.join(", ") || "none"}).`);
  const notExplicit = [
    "nude lipstick", "nude-colored heels", "nude make-up", "boob tube", "strap on her shoulder", "bag strap on shoulder", "grapes",
    "rapeseed field", "comic strip", "strip of paper", "striped shirt", "penetrating gaze", "sexy", "-1::hat::"
  ];
  const wronglyExplicit = notExplicit.filter((tag) => isExplicitSceneTag(tag));
  assertCheck("lexicon.gaps", wronglyExplicit.length === 0, `Fashion colours, straps, stripes and look-alikes are not explicit (flagged: ${wronglyExplicit.join(", ") || "none"}).`);

  const floor = (tags) => classifyImageScene(cut({ figures: [["1girl", ...tags]], context: ["1girl", "solo", "indoors"], frame: "cowboy_shot" }));
  for (const tags of [["school-uniform", "lingerie"], ["school-girl", "lingerie"], ["school uniform", "bralette"], ["school uniform", "swim suit"], ["sailor-fuku", "bathing suit"]]) {
    const result = floor(tags);
    assertCheck("lexicon.gaps", result.schoolCoded && result.suggestive, `School + suggestive is caught through hyphens and spaced spellings: ${tags.join(", ")} (school ${result.schoolCoded}, suggestive ${result.suggestive}).`);
  }
  const reunion = floor(["high school reunion photo of adults"]);
  assertCheck("lexicon.gaps", reunion.schoolCoded && !reunion.minorCoded && !reunion.suggestive, "A high school reunion of adults is school-coded only: not minor-coded, not locked on its own.");
}

/**
 * Review finding safety:no-unicode-normalization. Fullwidth letters, zero-width characters and soft hyphens fold to
 * the plain word; letters no lexicon can read (kana, Cyrillic look-alikes, CJK, Hangul) are unverifiable and lock the
 * cut in every mode, a forced photoreal regeneration included.
 */
function evaluateUnverifiableScript({ classifyImageScene }, { decideImageBackend }, seedState) {
  const solo = (tags) => classifyImageScene(cut({ figures: [["1girl", ...tags]], context: ["1girl", "solo", "beach"], frame: "full_body" }));
  for (const tag of ["ｎｕｄｅ", "nu\u200bde", "nu\u00adde", "ｎ\u200bｕｄｅ"]) {
    const result = solo([tag]);
    assertCheck("unicode", result.category === "explicit" && !result.unverifiable, `${JSON.stringify(tag)} folds to "nude" and is explicit (${result.category}).`);
  }
  assertCheck("unicode", solo(["ｌｏｌｉ"]).minorCoded && solo(["lo\u200bli"]).minorCoded, "Fullwidth and zero-width-split \"loli\" are minor-coded.");
  const unreadable = [
    ["n\u0430ked", "Cyrillic \u0430"],
    ["l\u043eli", "Cyrillic \u043e"],
    ["ロリ", "katakana"],
    ["ヌード", "katakana"],
    ["ﾇｰﾄﾞ", "halfwidth katakana"],
    ["一个十四岁的韩国少女", "Chinese"],
    ["누드", "Hangul"],
    ["ɴᴜᴅᴇ", "IPA small capitals"],
    ["\u03bdude", "Greek \u03bd"]
  ];
  for (const [tag, label] of unreadable) {
    const result = solo([tag, "bikini"]);
    assertCheck(
      "unicode",
      result.unverifiable && result.signals.some((signal) => signal.startsWith("unverifiable-script:")),
      `${label} (${JSON.stringify(tag)}) is unverifiable (${result.signals.join(" | ")}).`
    );
  }
  const kana = solo(["ロリ", "bikini"]);
  for (const mode of ["novelai", "mixed", "photoreal"]) {
    for (const availability of AVAILABILITY) {
      const state = createRoutingState(seedState, { mode, categories: ["portrait", "full_body", "scenery", "object", "group", "romance", "action", "fantasy"], photoreal: availability.photoreal, novelAi: availability.novelAi });
      const outcomes = [undefined, "photoreal"].map((forcedBackend) => decideImageBackend(state, kana, forcedBackend ? { forcedBackend } : undefined));
      assertCheck(
        "unicode",
        outcomes.every((decision) => decision.backend === "novelai" && decision.locked === true && /확인할 수 없는 문자/u.test(decision.reason)),
        `An unverifiable cut stays NovelAI + locked in ${mode} mode, ${availability.label}, unforced and forced photoreal (${outcomes.map((decision) => `${decision.backend}${decision.locked ? "+locked" : ""}`).join(", ")}).`
      );
    }
  }
  const accented = solo(["café latte", "Pokémon t-shirt"]);
  assertCheck("unicode", !accented.unverifiable, "Accented Latin letters fold to a–z and stay verifiable (café, Pokémon).");
}

/**
 * Free text read for the floor only (safetyTexts): the photoreal style and finished photo prompts. Prose is split
 * into sentences, comma segments and clauses, so the probe sentences lock; the category is never decided by it.
 */
async function evaluateSafetyTexts({ classifyImageScene }, { decideImageBackend }, seedState) {
  const prose = (text) => classifyImageScene({ characterTagSets: [["1girl"]], contextTags: ["cafe"], safetyTexts: [text], frame: FRAMES.upper_body, registeredCharacterCount: 1 });
  const locked = (result) => result.category === "explicit" || result.minorCoded || (result.schoolCoded && result.suggestive) || result.unverifiable;
  const probes = [
    "A 14-year-old Korean girl in a bikini",
    "A naked woman lying on a bed",
    "A teenage girl in lingerie, lying on a bed",
    "A little girl in a school uniform with a short skirt",
    "一个十四岁的韩国少女",
    "A nude boudoir photograph. Teenage model, 16 years old.",
    "fully naked subject, no clothes",
    "She is lying on her back and wearing lingerie.",
    "She is a minor."
  ];
  for (const text of probes) {
    const result = prose(text);
    assertCheck("safety-texts", locked(result), `Prose the floor must refuse is refused: ${JSON.stringify(text)} (${result.category}; ${result.signals.join(" | ")}).`);
  }
  for (const text of ["A Korean woman in her late twenties with a minor scar.", "A cinematic film still. Anamorphic lens, atmospheric depth."]) {
    const result = prose(text);
    assertCheck("safety-texts", !locked(result) && result.category === "portrait", `Ordinary prose passes and leaves the category alone: ${JSON.stringify(text)} (${result.category}).`);
  }

  // The style vocabulary names a smartphone, a photograph and a room; read as tags it turned an empty street into an
  // object shot. As safety-only text it decides nothing about the category.
  const { describePhotorealStyleForScreening } = await vite.ssrLoadModule("/src/services/photorealPrompt.ts");
  const dailyLook = describePhotorealStyleForScreening({ stylePreset: "daily_look", stylePrompt: "" });
  assertCheck("safety-texts", dailyLook.some((text) => /smartphone photograph/u.test(text)) && dailyLook.some((text) => /cream and white tones/u.test(text)), "The screened style text includes the preset lead and its ambient defaults.");
  for (const context of [["no humans", "street", "night"], ["no humans", "bedroom"], ["no humans", "coffee cup", "cake"]]) {
    const plain = classifyImageScene(cut({ context }));
    const styled = classifyImageScene({ ...cut({ context }), safetyTexts: dailyLook });
    assertCheck("safety-texts", plain.category === styled.category && !locked(styled), `The daily_look style never changes a cut's category: ${context.join(", ")} → ${plain.category} / ${styled.category}.`);
  }
  const state = createRoutingState(seedState, { mode: "photoreal", photoreal: true, novelAi: true });
  const styledCut = classifyImageScene({ ...cut({ figures: [["1girl", "smile", "sitting"]], context: ["1girl", "solo", "cafe"], frame: "upper_body" }), safetyTexts: describePhotorealStyleForScreening({ stylePreset: "custom", stylePrompt: "A nude boudoir photograph. Teenage model, 16 years old." }) });
  const decision = decideImageBackend(state, styledCut, { forcedBackend: "photoreal" });
  assertCheck("safety-texts", decision.backend === "novelai" && decision.locked, `A clean cafe cut under an explicit, minor-coded custom style is locked (${decision.reason}).`);
  const korean = describePhotorealStyleForScreening({ stylePreset: "daily_look", stylePrompt: "필름 느낌, soft film grain, ヌード" });
  assertCheck(
    "safety-texts",
    korean.includes("soft film grain") && korean.includes("ヌード") && !korean.some((text) => /[가-힣]/u.test(text)),
    "Korean style segments are left out of the screen (the builder drops them, as the settings screen says); any other script stays in to lock."
  );
}

/** Review finding safety:negative-weight-undress: NovelAI's undressing idiom reads as undressed, other negatives do not. */
function evaluateNegativeWeights({ classifyImageScene, isExplicitSceneTag }) {
  const undress = ["-2::clothing::", "-1::clothes::", "-1::shirt::", "-1::white shirt, pleated skirt::", "-1.5::school uniform::", "-1::bra::"];
  const missed = undress.filter((tag) => !isExplicitSceneTag(tag));
  assertCheck("negative-weight", missed.length === 0, `A negative weight on body-covering clothing is read as undressed (missed: ${missed.join(", ") || "none"}).`);
  const harmless = ["-1::hat::", "-1::glasses::", "-1::jacket::", "-1::shoes::", "1.2::white shirt::", "-1::text::"];
  const flagged = harmless.filter((tag) => isExplicitSceneTag(tag));
  assertCheck("negative-weight", flagged.length === 0, `A negative weight on an accessory, or a positive one on clothing, is not (flagged: ${flagged.join(", ") || "none"}).`);
  const grouped = classifyImageScene(cut({ figures: [["1girl, smile, -1::shirt, skirt::, standing"]], context: ["1girl", "solo"], frame: "full_body" }));
  assertCheck(
    "negative-weight",
    grouped.category === "explicit" && grouped.signals.some((signal) => signal.includes("skirt weighted down")),
    `A negative weight group weights down every tag inside it (${grouped.signals.join(" | ")}).`
  );
}

/**
 * Red-team round 3: the floor reads each term in its disguise readings too (leetspeak, in-word punctuation, masked
 * letters, split letters and fragments, stretched letters), states of undress and ages in any phrasing, a bare chest
 * by gender, and student identities in free text. Readings only ever add: the canonical term is still read, so
 * "1girl" and adult ages stay what they were. photoreal-safety-corpus-eval runs the full corpus end to end.
 */
function evaluateFloorReadings({ classifyImageScene, isExplicitSceneTag, isMinorCodedSceneTag, isPhotoNegativeSafeguardTerm }) {
  const disguised = ["n4ked", "nak3d", "$ex", "p0rn", "nu/de", "na·ked", "nak'd", "n*ked", "p*ssy", "n a k e d", "na ked", "nu de", "nuuuude", "blow-job", "blow job", "cream-pie", "deep-throating"];
  const missedDisguised = disguised.filter((tag) => !isExplicitSceneTag(tag));
  assertCheck("floor.readings", missedDisguised.length === 0, `Disguised spellings read as the word (missed: ${missedDisguised.join(", ") || "none"}).`);
  const undressed = ["seminude", "half-naked", "in the buff", "not wearing anything", "clothes off", "wearing only a smile", "nothing but stockings", "coitus", "jerking off", "uncensored", "wardrobe malfunction", "womb tattoo", "x-ray"];
  const missedUndressed = undressed.filter((tag) => !isExplicitSceneTag(tag));
  assertCheck("floor.readings", missedUndressed.length === 0, `Undress, acts and Danbooru's explicit tags in every phrasing are explicit (missed: ${missedUndressed.join(", ") || "none"}).`);
  const graded = ["wearing only an oversized sweater", "nothing but trouble", "naked cake", "food porn", "summa cum laude", "oral presentation", "nude nails", "Moby Dick", "banana cream pie", "riding a jet ski"];
  const wronglyExplicit = graded.filter((tag) => isExplicitSceneTag(tag));
  assertCheck("floor.readings", wronglyExplicit.length === 0, `A garment worn alone, and the false friends each guard names, are not explicit (flagged: ${wronglyExplicit.join(", ") || "none"}).`);

  const ages = ["a girl of twelve", "she's 15", "who is 14", "12 years", "12 years young", "17-year-olds", "XII years old", "on her 13th birthday", "under 18", "born in 2014", "looks 12", "14", "niña", "chica de 14 años", "14-jähriges Mädchen", "12⃣ years old"];
  const missedAges = ages.filter((tag) => !isMinorCodedSceneTag(tag));
  assertCheck("floor.readings", missedAges.length === 0, `Ages under 18 are read in any phrasing and language (missed: ${missedAges.join(", ") || "none"}).`);
  const notAges = ["1girl", "25 years old", "she is 25", "twelve years after graduation", "for 12 years", "aged 24 months", "a 12-year-old whisky", "her 16-year-old cat naps on the sofa", "twenty-one years old", "21.5 years old", "18th birthday", "under 21", "Nina", "it's 12 o'clock"];
  const wronglyMinor = notAges.filter((tag) => isMinorCodedSceneTag(tag));
  assertCheck("floor.readings", wronglyMinor.length === 0, `Adult ages, durations, quantities and an age owned by a thing are not minor-coded (flagged: ${wronglyMinor.join(", ") || "none"}).`);
  const owned = ["a 12-year-old cat girl", "a girl aged 12 years single malt", "a 12-year-old girl's cat"];
  const missedOwned = owned.filter((tag) => !isMinorCodedSceneTag(tag));
  assertCheck("floor.readings", missedOwned.length === 0, `A thing owns an age only when no person can (missed: ${missedOwned.join(", ") || "none"}).`);

  const solo = (figure, safetyTexts) => classifyImageScene({ characterTagSets: figure ? [figure] : [], contextTags: [], safetyTexts, registeredCharacterCount: figure ? 1 : 0 });
  assertCheck("floor.readings", solo(["1girl", "shirtless"]).category === "explicit" && solo(["1boy", "shirtless"]).category !== "explicit", "A bare chest is explicit on a woman and not on a man.");
  const student = solo(["1girl", "high school student"]);
  const studentProse = solo(undefined, ["A Korean high school student with long black hair."]);
  assertCheck("floor.readings", student.schoolCoded && !student.minorCoded && studentProse.minorCoded, "A student identity is school-coded in a tag and minor-coded in free text.");

  assertCheck("floor.readings", isExplicitSceneTag("-1::sfw::") && isExplicitSceneTag("-2::apparel::") && isMinorCodedSceneTag("-1::adult::") && isMinorCodedSceneTag("-1::mature female::"), "Weighted down, a safeguard word uncensors, a covering garment undresses and an adulthood word ages down.");
  assertCheck("floor.readings", !isMinorCodedSceneTag("-1::large breasts::") && !isExplicitSceneTag("-1::hat::"), "An adult build or an accessory weighted down locks nothing.");
  const safeguards = ["apparel", "fabric", "covered body", "sfw", "censored", "modest", "adult", "mature female", "woman", "wrinkles", "tall", "large breasts", "jacket", "towel"];
  const notSafeguards = ["blurry", "lowres", "watermark", "hat", "glasses", "shoes", "nsfw", "nude", "child", "anime"];
  assertCheck(
    "floor.readings",
    safeguards.every((term) => isPhotoNegativeSafeguardTerm(term)) && notSafeguards.every((term) => !isPhotoNegativeSafeguardTerm(term)),
    `Safeguard terms never become a photo negative; quality terms and accessories do (${[...safeguards.filter((term) => !isPhotoNegativeSafeguardTerm(term)), ...notSafeguards.filter((term) => isPhotoNegativeSafeguardTerm(term))].join(", ") || "all as expected"}).`
  );
}

/**
 * The real-data check from the review: every tag-shaped string in the shipped simulations' image fields (identity,
 * outfit, scene presets, style modules) run through the classifier. None may be unverifiable, and every minor-coded
 * one must be a genuine age marker (the shipped idol drama's teen trainees), so a lexicon change that starts flagging
 * ordinary vocabulary fails here.
 */
async function evaluateSeedVocabulary({ isMinorCodedSceneTag, classifyImageScene }) {
  const seed = await vite.ssrLoadModule("/src/data/seed.ts");
  const tags = new Set();
  const seen = new WeakSet();
  const walk = (value, key = "") => {
    if (typeof value === "string") {
      // Image fields only: a module's activationTags are retrieval keywords and never reach an image prompt.
      if (/(?:prompts?|tags|keyword|userRules)$/iu.test(key) && key !== "activationTags") {
        for (const raw of value.split(/[,\n;|]/u)) {
          const tag = raw.trim();
          // The composer drops Hangul and CJK tags before anything is classified.
          if (tag && !/[\u3131-\uD79D]/u.test(tag) && tag.split(/\s+/u).length <= 6) {
            tags.add(tag);
          }
        }
      }
      return;
    }
    if (!value || typeof value !== "object" || seen.has(value)) {
      return;
    }
    seen.add(value);
    for (const [childKey, child] of Object.entries(value)) {
      walk(child, Array.isArray(value) ? key : childKey);
    }
  };
  seed.builtInSimulationStates.forEach((state) => walk(state));
  const list = [...tags];
  const unreadable = list.filter((tag) => classifyImageScene(cut({ figures: [[tag]], registered: 1 })).unverifiable);
  assertCheck("seed-vocabulary", list.length > 500 && unreadable.length === 0, `No shipped image tag is unverifiable (${list.length} tags; flagged: ${unreadable.join(", ") || "none"}).`);
  const minor = list.filter((tag) => isMinorCodedSceneTag(tag));
  const unexpected = minor.filter((tag) => !/\bteen\b|\bmiddle[- ]school\b/iu.test(tag));
  assertCheck("seed-vocabulary", unexpected.length === 0, `Every minor-coded shipped tag is a genuine age marker (${minor.length} minor-coded; unexpected: ${unexpected.join(", ") || "none"}).`);
}

function evaluateTagLexicons({ isExplicitSceneTag, isMinorCodedSceneTag }) {
  const explicitTags = ["sex from behind", "1.3::missionary::", "nothing", "no clothes", "{{nude}}", "undressed", "after sex", "fellatio"];
  const missedExplicit = explicitTags.filter((tag) => !isExplicitSceneTag(tag));
  assertCheck("lexicon", missedExplicit.length === 0, `Unambiguous explicit tags are explicit on their own (missed: ${missedExplicit.join(", ") || "none"}).`);

  // What a tag model writes when it ignores the canonical vocabulary. The floor holds on what is sent.
  const vulgarTags = ["blowjob", "creampie", "fucking", "porn", "titjob", "masturbating", "1.2::deepthroat::", "nude, lipstick"];
  const missedVulgar = vulgarTags.filter((tag) => !isExplicitSceneTag(tag));
  assertCheck("lexicon", missedVulgar.length === 0, `Non-canonical explicit spellings are explicit too (missed: ${missedVulgar.join(", ") || "none"}).`);

  const fashion = ["nude lipstick", "nude-colored heels", "nude color dress", "nude makeup", "hetero couple"];
  const fashionFlagged = fashion.filter((tag) => isExplicitSceneTag(tag));
  assertCheck("lexicon", fashionFlagged.length === 0, `"nude" as a cosmetic or named colour, and "hetero couple", are not explicit (flagged: ${fashionFlagged.join(", ") || "none"}).`);
  assertCheck(
    "lexicon",
    isExplicitSceneTag("nude heels") && isExplicitSceneTag("nude stockings"),
    `"nude" before a garment stays explicit: a photo model can read "nude heels" as a nude woman in heels.`
  );
  assertCheck(
    "lexicon",
    isExplicitSceneTag("nude lipstick on a naked woman") && isExplicitSceneTag("nude makeup, completely nude"),
    "The colour exemption covers that one word only; the rest of the tag is still read."
  );

  const ambiguous = ["riding", "on back", "on stomach", "all fours", "spread legs", "bent over", "legs up", "lap pillow", "spooning"];
  const flagged = ambiguous.filter((tag) => isExplicitSceneTag(tag));
  assertCheck("lexicon", flagged.length === 0, `Ambiguous posture words are NOT explicit alone (flagged: ${flagged.join(", ") || "none"}).`);

  const notExplicit = ["partially undressed", "sexy", "bathing suit", "lingerie", "same-sex couple", "heterochromia", "huge breasts", "choker"];
  const wronglyExplicit = notExplicit.filter((tag) => isExplicitSceneTag(tag));
  assertCheck("lexicon", wronglyExplicit.length === 0, `Suggestive or look-alike tags are not explicit (flagged: ${wronglyExplicit.join(", ") || "none"}).`);

  assertCheck("lexicon", isExplicitSceneTag("showering") && isExplicitSceneTag("taking a bath"), "Activities that imply nudity (showering, bathing) are explicit.");

  const minorTags = [
    "loli", "shota", "child", "children", "kid", "toddler", "baby", "infant", "little girl", "little boy", "preteen",
    "elementary school", "elementary schooler", "middle school", "middle schooler", "kindergarten", "underage",
    "aged down", "female child", "male child", "[[loli]]", "15 years old", "16yo", "aged 14", "fifteen-year-old"
  ];
  const missedMinor = minorTags.filter((tag) => !isMinorCodedSceneTag(tag));
  assertCheck("lexicon", missedMinor.length === 0, `Minor-coded tags are recognised (missed: ${missedMinor.join(", ") || "none"}).`);

  const notMinor = [
    "school uniform", "high school student", "1girl", "lolita fashion", "baby blue sweater", "baby bangs", "baby tee",
    "childhood friend", "young woman", "20 years old", "aged 30", "adult"
  ];
  const wronglyMinor = notMinor.filter((tag) => isMinorCodedSceneTag(tag));
  assertCheck("lexicon", wronglyMinor.length === 0, `Adult, school-only and fashion tags are not minor-coded (flagged: ${wronglyMinor.join(", ") || "none"}).`);

  assertCheck(
    "lexicon",
    isMinorCodedSceneTag("child in a baby blue dress"),
    "A fashion exemption on one word never exempts the rest of the tag (`child in a baby blue dress` stays minor-coded)."
  );
}

function evaluateCategoryFixtures({ classifyImageScene }) {
  const fixtures = [
    {
      label: "hetero sex scene with continuity position tags",
      expected: "explicit",
      input: cut({
        figures: [
          ["1girl", "long hair", "black hair", "brown eyes", "nude", "blush", "on back", "missionary", "spread legs", "sweat"],
          ["1boy", "short hair", "hetero", "sex", "penis"]
        ],
        context: ["1girl", "1boy", "indoors", "bedroom", "bed", "night", "dim lighting"],
        frame: "cowboy_shot"
      })
    },
    {
      label: "solo sprint through a neon street",
      expected: "action",
      input: cut({
        figures: [["1girl", "brown hair", "ponytail", "running", "track jacket", "pleated skirt", "open mouth"]],
        context: ["1girl", "solo", "city street", "night", "motion blur", "neon lights"],
        frame: "full_body"
      })
    },
    {
      label: "two-figure sword fight",
      expected: "action",
      input: cut({
        figures: [
          ["1girl", "silver hair", "holding sword", "fighting stance", "sword fight", "determined"],
          ["1boy", "armor", "katana", "attacking"]
        ],
        context: ["1girl", "1boy", "battlefield", "sparks", "dutch angle"],
        frame: "wide_shot"
      })
    },
    {
      label: "cat-eared girl in a sweater",
      expected: "fantasy",
      input: cut({
        figures: [["1girl", "cat ears", "cat tail", "white hair", "smile", "looking at viewer", "sweater"]],
        context: ["1girl", "solo", "indoors", "window light"],
        frame: "upper_body"
      })
    },
    {
      label: "elf whose fantasy traits come only from the injected identity",
      expected: "fantasy",
      input: cut({
        figures: [["1girl", "pointy ears", "elf", "blonde hair", "green eyes", "reading", "smile", "white dress"]],
        context: ["1girl", "solo", "library", "bookshelf"],
        frame: "upper_body"
      })
    },
    {
      label: "couple hugging and kissing at sunset",
      expected: "romance",
      input: cut({
        figures: [
          ["1girl", "long hair", "closed eyes", "kiss", "hug", "blush"],
          ["1boy", "short hair", "hug", "kiss"]
        ],
        context: ["1girl", "1boy", "couple", "sunset", "outdoors", "beach"],
        frame: "upper_body"
      })
    },
    {
      label: "two friends talking over coffee",
      expected: "group",
      input: cut({
        figures: [
          ["1girl", "brown hair", "smile", "sitting", "holding cup", "cardigan"],
          ["1girl", "black hair", "short hair", "talking", "sitting", "coffee cup"]
        ],
        context: ["2girls", "cafe", "table", "indoors", "daylight"],
        frame: "upper_body"
      })
    },
    {
      label: "solo upper-body portrait",
      expected: "portrait",
      input: cut({
        figures: [["1girl", "black hair", "long hair", "brown eyes", "smile", "looking at viewer", "white sweater", "earrings"]],
        context: ["1girl", "solo", "indoors", "window light", "soft lighting"],
        frame: "upper_body"
      })
    },
    {
      label: "daily-look mirror selfie, full body",
      expected: "full_body",
      input: cut({
        figures: [["1girl", "black hair", "mirror selfie", "holding phone", "crop top", "miniskirt", "thighhighs", "sneakers", "standing"]],
        context: ["1girl", "solo", "bedroom", "mirror", "indoors"],
        frame: "full_body"
      })
    },
    {
      label: "rainy street at night, nobody in it",
      expected: "scenery",
      input: cut({ context: ["no humans", "street", "night", "rain", "puddle", "neon lights"] })
    },
    {
      label: "no humans, cafe, coffee cup, cake",
      expected: "object",
      input: cut({ context: ["no humans", "cafe", "coffee cup", "cake"] })
    },
    {
      label: "letter and envelope on a desk",
      expected: "object",
      input: cut({ context: ["no humans", "desk", "letter", "envelope", "window light"] })
    },
    {
      label: "library interior that happens to show a book",
      expected: "scenery",
      input: cut({ context: ["no humans", "library", "bookshelf", "book", "indoors", "sunlight"] })
    },
    {
      label: "plated dessert with food focus",
      expected: "object",
      input: cut({ context: ["no humans", "food focus", "strawberry", "parfait", "table", "restaurant", "indoors"] })
    }
  ];

  for (const fixture of fixtures) {
    const result = classifyImageScene(fixture.input);
    assertCheck(
      "category",
      result.category === fixture.expected,
      `${fixture.label} → ${result.category} (expected ${fixture.expected}; signals ${result.signals.join(" | ")})`
    );
  }

  const explicitResult = classifyImageScene(fixtures[0].input);
  assertCheck(
    "category",
    explicitResult.signals[0]?.startsWith("explicit:") && explicitResult.signals.includes("subjects:2"),
    `Signals lead with the deciding tag and record the subject count (${explicitResult.signals.join(" | ")}).`
  );
}

function evaluatePrecedence({ classifyImageScene }) {
  const kissDuringFight = classifyImageScene(
    cut({
      figures: [["1girl", "red hair", "kiss", "fighting", "punching", "angry"], ["1boy", "kiss", "struggling"]],
      context: ["1girl", "1boy", "alley", "motion blur"],
      frame: "upper_body"
    })
  );
  assertCheck("precedence", kissDuringFight.category === "action", `A kiss during a fight is action, not romance (${kissDuringFight.category}).`);

  // The authored caption is a plain face close-up; continuity injected the ongoing position into the composed tags.
  const injected = classifyImageScene(
    cut({
      figures: [["1girl", "black hair", "blush", "open mouth", "half-closed eyes", "sweat", "sex from behind"]],
      context: ["close-up", "1girl", "indoors"],
      frame: "close_up"
    })
  );
  assertCheck(
    "precedence",
    injected.category === "explicit" && injected.signals.includes("explicit:sex from behind"),
    `A face close-up whose composed tags carry an injected 'sex from behind' is explicit (${injected.category}; ${injected.signals.join(" | ")}).`
  );

  const holdingHands = classifyImageScene(
    cut({
      figures: [["1girl", "long hair", "smile", "holding hands", "walking"], ["1girl", "short hair", "holding hands", "blush"]],
      context: ["2girls", "street", "outdoors", "day"],
      frame: "full_body"
    })
  );
  assertCheck(
    "precedence",
    holdingHands.category === "romance" && holdingHands.humanCount === 2,
    `Two girls holding hands is romance, not group (${holdingHands.category}, ${holdingHands.humanCount} people).`
  );

  const solo = (frame, context = ["1girl", "solo", "indoors"]) =>
    classifyImageScene(cut({ figures: [["1girl", "black hair", "smile", "white blouse"]], context, frame })).category;
  const portraitShots = ["close_up", "face_focus", "upper_body"].map((shot) => `${shot}=${solo(shot)}`);
  assertCheck(
    "precedence",
    portraitShots.every((entry) => entry.endsWith("=portrait")),
    `close_up / face_focus / upper_body frames are portrait (${portraitShots.join(", ")}).`
  );
  const fullShots = ["cowboy_shot", "full_body", "wide_shot"].map((shot) => `${shot}=${solo(shot)}`);
  assertCheck(
    "precedence",
    fullShots.every((entry) => entry.endsWith("=full_body")),
    `cowboy_shot / full_body / wide_shot frames are full_body (${fullShots.join(", ")}).`
  );
  assertCheck("precedence", solo(undefined, ["1girl", "solo", "full body", "street"]) === "full_body", "With no frame, a 'full body' composition tag makes the cut full_body.");
  assertCheck("precedence", solo(undefined, ["1girl", "solo", "upper body", "street"]) === "portrait", "With no frame, an 'upper body' composition tag makes the cut portrait.");
  assertCheck("precedence", solo(undefined, ["1girl", "solo", "street"]) === "portrait", "With no frame and no composition tag the cut defaults to portrait.");
  assertCheck("precedence", solo("close_up", ["1girl", "solo", "full body"]) === "portrait", "A declared frame outranks a contradictory composition tag.");

  const fantasyKiss = classifyImageScene(
    cut({
      figures: [["1girl", "cat ears", "kiss"], ["1girl", "fox ears", "kiss"]],
      context: ["2girls", "indoors"],
      frame: "upper_body"
    })
  );
  assertCheck("precedence", fantasyKiss.category === "fantasy", `Fantasy outranks romance (${fantasyKiss.category}).`);

  const nudeElf = classifyImageScene(cut({ figures: [["1girl", "elf", "pointy ears", "nude", "forest"]], context: ["1girl", "solo"] }));
  assertCheck("precedence", nudeElf.category === "explicit", `Explicit outranks fantasy (${nudeElf.category}).`);

  const battleMage = classifyImageScene(
    cut({ figures: [["1girl", "witch", "magic circle", "fighting", "casting spell"]], context: ["1girl", "solo", "ruins"] })
  );
  assertCheck("precedence", battleMage.category === "action", `Action outranks fantasy (${battleMage.category}).`);

  const reachingBeat = classifyImageScene(
    cut({
      kind: "action",
      figures: [["1girl", "reaching", "standing on tiptoes", "cardigan"]],
      context: ["1girl", "solo", "library", "bookshelf"],
      frame: "full_body"
    })
  );
  assertCheck(
    "precedence",
    reachingBeat.category === "full_body" && reachingBeat.signals.includes("kind:action"),
    `A bare cue kind "action" without a dynamic tag is not action (${reachingBeat.category}; ${reachingBeat.signals.join(" | ")}).`
  );

  const povHands = classifyImageScene(cut({ figures: [["1girl", "smile", "holding hands", "pov"]], context: ["1girl", "solo", "street"] }));
  assertCheck("precedence", povHands.category === "romance", `A pair tag (POV holding hands) is romance even with one rendered figure (${povHands.category}).`);

  const soloHug = classifyImageScene(cut({ figures: [["1girl", "hug", "smile"]], context: ["1girl", "solo", "bedroom"], frame: "upper_body" }));
  assertCheck("precedence", soloHug.category === "portrait", `"hug" needs a second person to be romance (${soloHug.category}).`);
}

function evaluateSubjectCounting({ classifyImageScene }) {
  const baseOnly = classifyImageScene(cut({ context: ["2girls", "1boy", "classroom", "talking"] }));
  assertCheck("subjects", baseOnly.humanCount === 3 && baseOnly.category === "group", `Base-caption counts add up across kinds: 2girls + 1boy = ${baseOnly.humanCount} (${baseOnly.category}).`);

  const noDoubleCount = classifyImageScene(
    cut({ figures: [["1girl", "smile"], ["1girl", "frown"]], context: ["2girls", "park"], frame: "upper_body" })
  );
  assertCheck("subjects", noDoubleCount.humanCount === 2, `Per-figure 1girl identity tags are not added to the base caption's 2girls (${noDoubleCount.humanCount}).`);

  const registered = classifyImageScene(cut({ figures: [["1girl", "smile"]], context: ["indoors"], registered: 3 }));
  assertCheck("subjects", registered.humanCount === 3 && registered.category === "group", `The registered cast counts when the composer drops a figure (${registered.humanCount}, ${registered.category}).`);

  const crowd = classifyImageScene(cut({ context: ["6+girls", "stage", "idol"] }));
  assertCheck("subjects", crowd.humanCount === 6, `"6+girls" counts six (${crowd.humanCount}).`);

  const multiple = classifyImageScene(cut({ context: ["multiple girls", "beach"] }));
  assertCheck("subjects", multiple.humanCount === 2 && multiple.category === "group", `"multiple girls" counts at least two (${multiple.humanCount}, ${multiple.category}).`);

  const establishing = classifyImageScene(cut({ context: ["no humans", "scenery", "city", "night"], registered: 1 }));
  assertCheck(
    "subjects",
    establishing.category === "scenery",
    `A cue that lists a character but composes no figure and says "no humans" is scenery (${establishing.category}).`
  );

  const frozenInput = Object.freeze({
    characterTagSets: Object.freeze([Object.freeze(["1girl", "1.2::nude::"])]),
    contextTags: Object.freeze(["bedroom"]),
    registeredCharacterCount: 1
  });
  let threw = false;
  let frozenResult;
  try {
    frozenResult = classifyImageScene(frozenInput);
  } catch {
    threw = true;
  }
  assertCheck("subjects", !threw && frozenResult?.category === "explicit", "Classification never mutates its input and reads weighted tags (frozen input, `1.2::nude::`).");
}

function evaluateFalseFriends({ classifyImageScene }) {
  const soloCategory = (tags) =>
    classifyImageScene(cut({ figures: [["1girl", "smile", ...tags]], context: ["1girl", "solo", "outdoors"], frame: "upper_body" })).category;

  const actionFriends = [
    "falling petals", "running shoes", "swimming pool", "fruit punch", "combat boots", "tight combat shirt", "boxing gloves",
    "shooting star", "falling asleep", "choker"
  ];
  const misread = actionFriends.filter((tag) => soloCategory([tag]) === "action");
  assertCheck("false-friends", misread.length === 0, `Look-alikes are not action (${misread.join(", ") || "none misread"}).`);

  const fantasyFriends = [
    "ponytail", "twin tails", "fairy lights", "wing collar", "cat ear headphones", "mermaid dress", "dragon print", "witch hat",
    "tail lights", "chicken wings", "halo braid", "angel sleeves"
  ];
  const misreadFantasy = fantasyFriends.filter((tag) => soloCategory([tag]) === "fantasy");
  assertCheck("false-friends", misreadFantasy.length === 0, `Costume, props and hairstyles are not fantasy (${misreadFantasy.join(", ") || "none misread"}).`);

  const bunny = classifyImageScene(
    cut({ figures: [["1girl", "playboy bunny", "rabbit ears", "fake animal ears", "leotard"]], context: ["1girl", "solo", "casino"], frame: "cowboy_shot" })
  );
  assertCheck(
    "false-friends",
    bunny.category === "full_body" && bunny.suggestive,
    `A bunny costume's ears are wardrobe, not a fantasy trait (${bunny.category}, suggestive ${bunny.suggestive}).`
  );

  const costumeScope = classifyImageScene(
    cut({
      figures: [["1girl", "fake animal ears", "rabbit ears", "dress"], ["1girl", "fox ears", "fox tail", "kimono"]],
      context: ["2girls", "festival"],
      frame: "full_body"
    })
  );
  assertCheck("false-friends", costumeScope.category === "fantasy", `One figure's costume ears do not hide another figure's real fox ears (${costumeScope.category}).`);

  const soloPoses = classifyImageScene(
    cut({
      figures: [["1girl", "hugging own legs", "sitting"], ["1girl", "blowing kiss", "standing"]],
      context: ["2girls", "rooftop"],
      frame: "full_body"
    })
  );
  assertCheck("false-friends", soloPoses.category === "group", `Self-hugs and blown kisses are not romance between the two figures (${soloPoses.category}).`);

  const settingObject = classifyImageScene(cut({ context: ["no humans", "coffee shop", "coffee table", "sofa", "indoors"] }));
  assertCheck("false-friends", settingObject.category === "scenery", `A coffee shop and a coffee table are a setting, not an object shot (${settingObject.category}).`);
}

function evaluateAmbiguousPostures({ classifyImageScene }) {
  const sleeping = classifyImageScene(
    cut({ figures: [["1girl", "on back", "sleeping", "closed eyes", "pajamas"]], context: ["1girl", "solo", "bed", "bedroom"], frame: "full_body" })
  );
  assertCheck("posture", sleeping.category !== "explicit", `Lying on one's back asleep is not explicit (${sleeping.category}).`);

  const sitting = classifyImageScene(
    cut({ figures: [["1girl", "spread legs", "sitting", "chair", "jeans", "hoodie"]], context: ["1girl", "solo", "room"], frame: "full_body" })
  );
  assertCheck("posture", sitting.category !== "explicit", `Sitting with spread legs in jeans is not explicit (${sitting.category}).`);

  const lapPillow = classifyImageScene(
    cut({ figures: [["1girl", "lap pillow", "smile", "skirt"], ["1boy", "lying", "closed eyes"]], context: ["1girl", "1boy", "park", "grass"], frame: "full_body" })
  );
  assertCheck("posture", lapPillow.category === "romance", `A lap pillow between two people is romance, not explicit (${lapPillow.category}).`);

  const cycling = classifyImageScene(
    cut({ figures: [["1girl", "riding bicycle", "bikini", "sunglasses"]], context: ["1girl", "solo", "beach", "summer"], frame: "full_body" })
  );
  assertCheck("posture", cycling.category !== "explicit", `"riding bicycle" is a vehicle, not a posture, even next to a bikini (${cycling.category}).`);

  const yoga = classifyImageScene(
    cut({ figures: [["1girl", "all fours", "yoga mat", "sportswear"]], context: ["1girl", "solo", "indoors"], frame: "full_body" })
  );
  assertCheck("posture", yoga.category !== "explicit", `All fours on a yoga mat is not explicit (${yoga.category}).`);

  const combos = [
    { tags: ["on back", "lingerie", "bed"], signal: "explicit:on back+lingerie" },
    { tags: ["all fours", "bikini", "beach"], signal: "explicit:all fours+bikini" },
    { tags: ["bent over", "panties", "skirt"], signal: "explicit:bent over+panties" },
    { tags: ["spread legs", "cleavage", "sitting"], signal: "explicit:spread legs+cleavage" }
  ];
  for (const combo of combos) {
    const result = classifyImageScene(cut({ figures: [["1girl", ...combo.tags]], context: ["1girl", "solo"], frame: "full_body" }));
    assertCheck(
      "posture",
      result.category === "explicit" && result.signals.includes(combo.signal),
      `An ambiguous posture with a suggestive tag is explicit: ${combo.tags.join(", ")} → ${result.category} (${result.signals.join(" | ")}).`
    );
  }
}

function evaluateSafetyFloorFlags({ classifyImageScene }) {
  const flags = (figure, context = ["1girl", "solo"], frame = "upper_body") =>
    classifyImageScene(cut({ figures: [figure], context, frame }));

  const loli = flags(["1girl", "loli", "smile"]);
  assertCheck("floor.flags", loli.minorCoded && loli.signals.includes("minor:loli"), `loli is minor-coded (${loli.signals.join(" | ")}).`);
  assertCheck("floor.flags", flags(["female child", "sundress"], ["1girl", "park"]).minorCoded, "female child is minor-coded.");
  assertCheck("floor.flags", flags(["1girl", "{{loli}}"]).minorCoded, "A weighted minor tag ({{loli}}) is still minor-coded.");
  assertCheck("floor.flags", flags(["1girl", "14 years old"]).minorCoded && !flags(["1girl", "24 years old"]).minorCoded, "A stated age under 18 is minor-coded; 24 is not.");
  assertCheck("floor.flags", flags([], ["no humans", "kindergarten", "classroom"]).minorCoded, "A minor-coded setting tag codes the cut even without a figure.");

  const uniform = flags(["1girl", "school uniform", "serafuku", "smile"], ["1girl", "solo", "classroom"]);
  assertCheck(
    "floor.flags",
    uniform.schoolCoded && !uniform.suggestive && !uniform.minorCoded,
    `School uniform alone is school-coded but not suggestive (school ${uniform.schoolCoded}, suggestive ${uniform.suggestive}).`
  );

  const uniformLewd = flags(["1girl", "school uniform", "skirt lift", "panties"], ["1girl", "solo", "classroom"], "cowboy_shot");
  assertCheck("floor.flags", uniformLewd.schoolCoded && uniformLewd.suggestive, "School uniform + skirt lift + panties is school-coded and suggestive.");

  const schoolSwimsuit = flags(["1girl", "school swimsuit"], ["1girl", "solo", "pool"], "full_body");
  assertCheck("floor.flags", schoolSwimsuit.schoolCoded && schoolSwimsuit.suggestive, "A school swimsuit is school-coded and suggestive on its own.");

  const collegeBikini = flags(["1girl", "college student", "bikini"], ["1girl", "solo", "beach"], "full_body");
  assertCheck("floor.flags", !collegeBikini.schoolCoded && collegeBikini.suggestive, "A college student is adult-coded: bikini is suggestive but not school-coded.");

  const weighted = flags(["1girl", "(school uniform:1.1)", "[[panties]]"], ["1girl", "solo"], "cowboy_shot");
  assertCheck("floor.flags", weighted.schoolCoded && weighted.suggestive, "Weighted school and suggestive tags still count.");

  const bedCombo = flags(["1girl", "bare shoulders", "school uniform"], ["1girl", "solo", "lying on bed"], "full_body");
  const bareOnly = flags(["1girl", "bare shoulders", "school uniform"], ["1girl", "solo", "classroom"]);
  assertCheck(
    "floor.flags",
    bedCombo.suggestive && !bareOnly.suggestive,
    `"bare shoulders" is suggestive only together with lying on a bed (bed ${bedCombo.suggestive}, classroom ${bareOnly.suggestive}).`
  );

  const explicit = flags(["1girl", "nude"], ["1girl", "solo"]);
  assertCheck("floor.flags", explicit.suggestive, "An explicit cut also reports suggestive.");
}

// ---------------------------------------------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------------------------------------------

function evaluateRoutingResolution({ DEFAULT_PHOTOREAL_CATEGORIES, IMAGE_SCENE_CATEGORY_OPTIONS, resolveImageBackendRouting }) {
  assertCheck(
    "routing.resolve",
    JSON.stringify(DEFAULT_PHOTOREAL_CATEGORIES) === JSON.stringify(["portrait", "full_body", "scenery", "object"]),
    `DEFAULT_PHOTOREAL_CATEGORIES is portrait, full_body, scenery, object (${DEFAULT_PHOTOREAL_CATEGORIES.join(", ")}).`
  );
  assertCheck("routing.resolve", Object.isFrozen(DEFAULT_PHOTOREAL_CATEGORIES), "The shared default category list is frozen.");

  const missing = resolveImageBackendRouting({});
  assertCheck(
    "routing.resolve",
    missing.mode === "novelai" && JSON.stringify(missing.photorealCategories) === JSON.stringify(DEFAULT_PHOTOREAL_CATEGORIES),
    `A profile saved before photoreal (no backendRouting) routes as NovelAI-only with default categories (${missing.mode}).`
  );
  missing.photorealCategories.push("group");
  assertCheck(
    "routing.resolve",
    missing.photorealCategories !== DEFAULT_PHOTOREAL_CATEGORIES && DEFAULT_PHOTOREAL_CATEGORIES.length === 4,
    "Resolved categories are a fresh copy; editing them never touches the shared default."
  );
  assertCheck("routing.resolve", resolveImageBackendRouting(undefined).mode === "novelai" && resolveImageBackendRouting(null).mode === "novelai", "A missing profile resolves to NovelAI-only.");

  const sanitized = resolveImageBackendRouting({
    backendRouting: { mode: "mixed", photorealCategories: ["object", "bogus", "explicit", "portrait", "object", " Full-Body ", 7] }
  });
  assertCheck(
    "routing.resolve",
    sanitized.mode === "mixed" && JSON.stringify(sanitized.photorealCategories) === JSON.stringify(["portrait", "full_body", "object"]),
    `Unknown categories are dropped, explicit is stripped, duplicates collapse, order is canonical (${sanitized.photorealCategories.join(", ")}).`
  );

  const badMode = resolveImageBackendRouting({ backendRouting: { mode: "turbo", photorealCategories: ["scenery"] } });
  assertCheck("routing.resolve", badMode.mode === "novelai" && badMode.photorealCategories.join() === "scenery", `An unknown mode falls back to novelai and keeps valid categories (${badMode.mode}).`);

  const casing = resolveImageBackendRouting({ backendRouting: { mode: " Photoreal ", photorealCategories: "portrait" } });
  assertCheck(
    "routing.resolve",
    casing.mode === "photoreal" && JSON.stringify(casing.photorealCategories) === JSON.stringify(DEFAULT_PHOTOREAL_CATEGORIES),
    `Mode is read case-insensitively; a non-array category list falls back to the default (${casing.mode}).`
  );

  const empty = resolveImageBackendRouting({ backendRouting: { mode: "mixed", photorealCategories: [] } });
  assertCheck("routing.resolve", empty.photorealCategories.length === 0, "An explicitly empty category list stays empty (the user unticked everything).");

  const onlyExplicit = resolveImageBackendRouting({ backendRouting: { mode: "mixed", photorealCategories: ["explicit"] } });
  assertCheck("routing.resolve", onlyExplicit.photorealCategories.length === 0, "A stored tick on explicit alone resolves to nothing photoreal.");

  const expectedOptions = [
    ["portrait", "인물 단독 · 얼굴/상반신"],
    ["full_body", "인물 단독 · 전신/의상"],
    ["scenery", "배경 · 풍경"],
    ["object", "사물 · 음식"],
    ["group", "2인 이상 · 대화/일상"],
    ["romance", "스킨십 (비노출)"],
    ["action", "액션 · 역동적 동작"],
    ["fantasy", "판타지 · 비인간 요소"],
    ["explicit", "성적 장면 — 항상 NovelAI"]
  ];
  assertCheck(
    "routing.resolve",
    JSON.stringify(IMAGE_SCENE_CATEGORY_OPTIONS.map((option) => [option.id, option.label])) === JSON.stringify(expectedOptions),
    "Category options list all nine categories in the spec order with the spec's Korean labels."
  );
  const locked = IMAGE_SCENE_CATEGORY_OPTIONS.filter((option) => option.locked).map((option) => option.id);
  assertCheck("routing.resolve", locked.join() === "explicit", `Only the explicit option is locked (${locked.join(", ")}).`);
  assertCheck(
    "routing.resolve",
    IMAGE_SCENE_CATEGORY_OPTIONS.every((option) => /[가-힣]/u.test(option.description)),
    "Every option carries a Korean description."
  );
}

function evaluateBackendAvailability({ isNovelAiBackendConfigured, isPhotorealBackendConfigured }, seedState) {
  assertCheck("routing.available", isPhotorealBackendConfigured(createRoutingState(seedState, { photoreal: true })), "photoreal.enabled true → photoreal configured.");
  assertCheck("routing.available", !isPhotorealBackendConfigured(createRoutingState(seedState, { photoreal: false })), "photoreal.enabled false → photoreal not configured.");
  const legacy = createRoutingState(seedState);
  delete legacy.photoreal;
  assertCheck("routing.available", !isPhotorealBackendConfigured(legacy), "A state with no photoreal settings is not photoreal-configured (and does not throw).");
  assertCheck(
    "routing.available",
    isNovelAiBackendConfigured(createRoutingState(seedState, { novelAi: true })) && !isNovelAiBackendConfigured(createRoutingState(seedState, { novelAi: false })),
    "NovelAI configured follows novelAi.enabled."
  );
}

function evaluateDecisionTable({ decideImageBackend }, { classifyImageScene }, seedState) {
  const classifications = createClassifications(classifyImageScene);
  const ALL_SAFE = ["portrait", "full_body", "scenery", "object", "group", "romance", "action", "fantasy"];
  // [label, routing mode, photoreal on, NovelAI on, classification, forced backend, backend, locked, fallback]
  const table = [
    ["novelai mode keeps a portrait on NovelAI", "novelai", true, true, "portrait", undefined, "novelai", false, false],
    ["novelai mode with NovelAI off falls back to photoreal", "novelai", true, false, "portrait", undefined, "photoreal", false, true],
    ["novelai mode with photoreal off stays NovelAI", "novelai", false, true, "portrait", undefined, "novelai", false, false],
    ["novelai mode with both off keeps NovelAI", "novelai", false, false, "portrait", undefined, "novelai", false, false],
    ["mixed mode sends a ticked portrait photoreal", "mixed", true, true, "portrait", undefined, "photoreal", false, false],
    ["mixed mode portrait with NovelAI off stays photoreal", "mixed", true, false, "portrait", undefined, "photoreal", false, false],
    ["mixed mode portrait with photoreal off falls back to NovelAI", "mixed", false, true, "portrait", undefined, "novelai", false, true],
    ["mixed mode portrait with both off keeps photoreal", "mixed", false, false, "portrait", undefined, "photoreal", false, false],
    ["mixed mode keeps an unticked action cut on NovelAI", "mixed", true, true, "action", undefined, "novelai", false, false],
    ["mixed mode action with NovelAI off falls back to photoreal", "mixed", true, false, "action", undefined, "photoreal", false, true],
    ["photoreal mode sends a portrait photoreal", "photoreal", true, true, "portrait", undefined, "photoreal", false, false],
    ["photoreal mode sends action photoreal regardless of ticks", "photoreal", true, true, "action", undefined, "photoreal", false, false],
    ["photoreal mode with photoreal off falls back to NovelAI", "photoreal", false, true, "portrait", undefined, "novelai", false, true],
    ["photoreal mode with both off keeps photoreal", "photoreal", false, false, "portrait", undefined, "photoreal", false, false],
    ["forced photoreal overrides novelai mode", "novelai", true, true, "portrait", "photoreal", "photoreal", false, false],
    ["forced photoreal never falls back when photoreal is off", "novelai", false, true, "portrait", "photoreal", "photoreal", false, false],
    ["forced NovelAI overrides photoreal mode", "photoreal", true, true, "portrait", "novelai", "novelai", false, false],
    ["forced NovelAI never falls back when NovelAI is off", "photoreal", true, false, "portrait", "novelai", "novelai", false, false],
    ["school uniform without suggestive tags is not locked", "photoreal", true, true, "school", undefined, "photoreal", false, false],
    ["explicit with NovelAI off is never promoted to photoreal", "photoreal", true, false, "explicit", undefined, "novelai", true, false],
    ["minor-coded beats a forced photoreal regeneration", "photoreal", true, true, "minor", "photoreal", "novelai", true, false],
    ["school + suggestive is locked in mixed mode", "mixed", true, true, "schoolSuggestive", undefined, "novelai", true, false],
    ["school + suggestive beats a forced photoreal regeneration", "novelai", true, true, "schoolSuggestive", "photoreal", "novelai", true, false]
  ];

  for (const [label, mode, photoreal, novelAi, key, forcedBackend, backend, locked, fallback] of table) {
    const state = createRoutingState(seedState, { mode, categories: mode === "mixed" ? undefined : ALL_SAFE, photoreal, novelAi });
    const decision = decideImageBackend(state, classifications[key], forcedBackend ? { forcedBackend } : undefined);
    assertCheck(
      "routing.decide",
      decision.backend === backend && Boolean(decision.locked) === locked && Boolean(decision.fallback) === fallback,
      `${label}: ${decision.backend}${decision.locked ? " locked" : ""}${decision.fallback ? " fallback" : ""} (expected ${backend}${locked ? " locked" : ""}${fallback ? " fallback" : ""})`
    );
  }

  const tickedAction = decideImageBackend(
    createRoutingState(seedState, { mode: "mixed", categories: ["portrait", "action"], photoreal: true, novelAi: true }),
    classifications.action
  );
  assertCheck("routing.decide", tickedAction.backend === "photoreal", `Mixed mode follows the ticks: action ticked → photoreal (${tickedAction.backend}).`);

  const storedExplicitTick = decideImageBackend(
    createRoutingState(seedState, { mode: "mixed", categories: ["explicit", "portrait"], photoreal: true, novelAi: true }),
    classifications.explicit
  );
  assertCheck(
    "routing.decide",
    storedExplicitTick.backend === "novelai" && storedExplicitTick.locked === true,
    "A stored tick on explicit (hand-edited state) cannot route an explicit cut photoreal."
  );

  const legacyState = createRoutingState(seedState, { mode: undefined, photoreal: true, novelAi: true });
  const legacy = decideImageBackend(legacyState, classifications.portrait);
  assertCheck(
    "routing.decide",
    legacy.backend === "novelai" && legacy.mode === "novelai",
    `A simulation saved before photoreal keeps rendering NovelAI even with photoreal enabled (${legacy.backend}, mode ${legacy.mode}).`
  );

  const decided = decideImageBackend(createRoutingState(seedState, { mode: "mixed", photoreal: true, novelAi: true }), classifications.portrait);
  assertCheck(
    "routing.decide",
    decided.category === "portrait" &&
      decided.mode === "mixed" &&
      JSON.stringify(decided.signals) === JSON.stringify(classifications.portrait.signals) &&
      decided.signals !== classifications.portrait.signals,
    "The decision records category, mode and a copy of the classification signals."
  );
}

function evaluateSafetyLockMatrix({ decideImageBackend, describeImageBackendSetupBlock }, { classifyImageScene }, seedState) {
  const { explicit } = createClassifications(classifyImageScene);
  for (const mode of ["novelai", "mixed", "photoreal"]) {
    for (const availability of AVAILABILITY) {
      const state = createRoutingState(seedState, {
        mode,
        categories: ["portrait", "full_body", "scenery", "object", "group", "romance", "action", "fantasy", "explicit"],
        photoreal: availability.photoreal,
        novelAi: availability.novelAi
      });
      const outcomes = [undefined, "photoreal", "novelai"].map((forcedBackend) =>
        decideImageBackend(state, explicit, forcedBackend ? { forcedBackend } : undefined)
      );
      const held = outcomes.every((decision) => decision.backend === "novelai" && decision.locked === true && !decision.fallback);
      assertCheck(
        "routing.floor",
        held,
        `Explicit stays NovelAI + locked in ${mode} mode, ${availability.label}, unforced / forced photoreal / forced NovelAI (${outcomes
          .map((decision) => `${decision.backend}${decision.locked ? "+locked" : ""}`)
          .join(", ")}).`
      );
    }
  }

  const blockedState = createRoutingState(seedState, { mode: "photoreal", photoreal: true, novelAi: false });
  const blocked = decideImageBackend(blockedState, explicit, { forcedBackend: "photoreal" });
  assertCheck(
    "routing.floor",
    describeImageBackendSetupBlock(blockedState, blocked.backend) === NOVELAI_SETUP_TEXT,
    "A locked cut with NovelAI off is setup-blocked with the NovelAI message, not quietly moved to photoreal."
  );

  const neitherState = createRoutingState(seedState, { mode: "mixed", photoreal: false, novelAi: false });
  const neither = decideImageBackend(neitherState, createClassifications(classifyImageScene).portrait);
  assertCheck(
    "routing.floor",
    describeImageBackendSetupBlock(neitherState, neither.backend) === PHOTOREAL_SETUP_TEXT,
    "With both backends off, the setup block names the backend the cut is actually waiting for (photoreal)."
  );
}

function evaluateDecisionReasons({ decideImageBackend }, { classifyImageScene }, seedState) {
  const classifications = createClassifications(classifyImageScene);
  const mixed = decideImageBackend(createRoutingState(seedState, { mode: "mixed", photoreal: true, novelAi: true }), classifications.portrait);
  assertCheck(
    "routing.reason",
    mixed.reason === "혼합 모드: '인물 단독 · 얼굴/상반신' 장면 → 실사",
    `The reason names mode, category and backend: ${mixed.reason}`
  );

  const explicit = decideImageBackend(
    createRoutingState(seedState, { mode: "photoreal", photoreal: true, novelAi: true }),
    classifications.explicit,
    { forcedBackend: "photoreal" }
  );
  assertCheck(
    "routing.reason",
    explicit.reason.startsWith("성적 장면은 실사로 생성하지 않습니다 (NovelAI 고정)") && explicit.reason.includes("직접 지정"),
    `The explicit lock explains itself and that the manual pick was not applied: ${explicit.reason}`
  );

  const minor = decideImageBackend(createRoutingState(seedState, { mode: "photoreal", photoreal: true, novelAi: false }), classifications.minor);
  assertCheck(
    "routing.reason",
    minor.reason.includes("미성년자") && minor.reason.includes("NovelAI 고정") && minor.reason.includes("NovelAI 연동을 켜야"),
    `The minor lock explains itself and why the cut waits on NovelAI: ${minor.reason}`
  );

  const fallback = decideImageBackend(createRoutingState(seedState, { mode: "mixed", photoreal: false, novelAi: true }), classifications.portrait);
  assertCheck(
    "routing.reason",
    fallback.reason.includes("→ NovelAI") && fallback.reason.includes("실사 연동이 꺼져"),
    `A fallback says which backend was off: ${fallback.reason}`
  );

  const forced = decideImageBackend(createRoutingState(seedState, { mode: "novelai", photoreal: true, novelAi: true }), classifications.portrait, {
    forcedBackend: "photoreal"
  });
  assertCheck("routing.reason", forced.reason.includes("(직접 지정)") && forced.reason.includes("→ 실사"), `A forced backend is labelled as a manual pick: ${forced.reason}`);
}

async function evaluateSetupBlockTexts({ describeImageBackendSetupBlock }, seedState) {
  const offState = createRoutingState(seedState, { photoreal: false, novelAi: false });
  const onState = createRoutingState(seedState, { photoreal: true, novelAi: true });

  assertCheck("setup-block", describeImageBackendSetupBlock(offState, "novelai") === NOVELAI_SETUP_TEXT, "NovelAI off → the NovelAI setup-block text.");
  assertCheck("setup-block", describeImageBackendSetupBlock(onState, "novelai") === undefined, "NovelAI on → no setup block.");
  assertCheck("setup-block", describeImageBackendSetupBlock(offState, "photoreal") === PHOTOREAL_SETUP_TEXT, "Photoreal off → the photoreal setup-block text.");
  assertCheck("setup-block", describeImageBackendSetupBlock(onState, "photoreal") === undefined, "Photoreal on → no setup block.");

  const legacy = createRoutingState(seedState, { novelAi: true });
  delete legacy.photoreal;
  assertCheck("setup-block", describeImageBackendSetupBlock(legacy, "photoreal") === PHOTOREAL_SETUP_TEXT, "A state with no photoreal settings reads as photoreal off.");

  // Loaded here rather than up front: a sibling's in-progress edit of the orchestrator must cost this one check,
  // not the whole report.
  const { describeNovelAiSetupBlock } = await vite.ssrLoadModule("/src/services/imageOrchestrator.ts");
  assertCheck(
    "setup-block",
    describeImageBackendSetupBlock(offState, "novelai") === describeNovelAiSetupBlock(offState) &&
      describeImageBackendSetupBlock(onState, "novelai") === describeNovelAiSetupBlock(onState),
    "The NovelAI text is exactly describeNovelAiSetupBlock's, on and off."
  );
}

/**
 * The reason classification runs on the COMPOSED prompt: the real composer injects a persisted sex position
 * into a cut the model authored as a plain face close-up. Classifying the authored caption would miss it.
 */
async function evaluateComposedContinuity({ classifyImageScene }, seedState) {
  const { planImageJob } = await vite.ssrLoadModule("/src/services/imageOrchestrator.ts");
  const characters = [
    {
      id: "char_mei",
      simulationId: seedState.simulation.id,
      name: "Mei",
      role: "heroine",
      summary: "lead",
      relationship: "",
      currentMood: "flustered"
    }
  ];
  const visualProfiles = [
    {
      id: "visual_mei",
      simulationId: seedState.simulation.id,
      characterId: "char_mei",
      displayName: "Mei",
      positivePrompt: "1girl, black hair, brown eyes",
      negativePrompt: "",
      defaultOutfitPrompt: "white blouse, blue skirt",
      outfitPrompts: {},
      expressionPrompts: {},
      referenceImageAssetIds: [],
      defaultSafetyLevel: "safe"
    }
  ];
  const state = createRoutingState(seedState, { mode: "photoreal", photoreal: true, novelAi: true });
  state.characters = characters;
  state.visualProfiles = visualProfiles;
  state.memoryEvents = [createStateMemoryEvent(state, "char_mei", "Mei", "InteractionTags", "sex from behind")];

  const cue = {
    shouldGenerate: true,
    reason: "reaction close-up",
    characters: ["char_mei"],
    tags: [],
    baseTags: ["close-up", "indoors"],
    characterPrompts: [{ characterId: "char_mei", prompt: "blush, open mouth, face focus" }],
    scene: "current simulation scene",
    visualContext: "",
    frame: FRAMES.close_up
  };
  const job = planImageJob(state, "turn_photoreal_eval", cue, [], true);
  const composedPrompts = Array.isArray(job.providerPayload?.characterPrompts) ? job.providerPayload.characterPrompts : [];
  const composedText = composedPrompts.map((prompt) => prompt.prompt).join(" | ");
  assertCheck(
    "composed",
    /sex from behind/iu.test(composedText),
    `Premise: the composer injects the persisted position into the face close-up (${composedText || "no character prompts"}).`
  );

  const contextTags = Array.isArray(job.providerPayload?.contextTags)
    ? job.providerPayload.contextTags
    : job.providerPayload?.promptLayers?.context ?? [];
  const composed = classifyImageScene({
    characterTagSets: composedPrompts.map((prompt) => [prompt.prompt]),
    contextTags,
    frame: cue.frame,
    registeredCharacterCount: cue.characters.length
  });
  assertCheck(
    "composed",
    composed.category === "explicit" && composed.signals.includes("explicit:sex from behind"),
    `The composed cut classifies explicit (${composed.category}; ${composed.signals.join(" | ")}).`
  );

  const authoredOnly = classifyImageScene({
    characterTagSets: cue.characterPrompts.map((prompt) => [prompt.prompt]),
    contextTags: cue.baseTags,
    frame: cue.frame,
    registeredCharacterCount: 1
  });
  assertCheck(
    "composed",
    authoredOnly.category === "portrait",
    `The authored caption alone would have read as a harmless portrait (${authoredOnly.category}) — why routing classifies the composed prompt.`
  );
}

async function evaluateModuleBoundaries() {
  const routingSource = await readFile(path.join(workspaceRoot, "src/services/imageBackendRouting.ts"), "utf8");
  const classifierSource = await readFile(path.join(workspaceRoot, "src/services/imageSceneClassifier.ts"), "utf8");
  assertCheck(
    "boundaries",
    !/from\s+["']\.\/imageOrchestrator["']/u.test(routingSource) && !/from\s+["']\.\/imageOrchestrator["']/u.test(classifierSource),
    "Neither module imports imageOrchestrator (it imports them; the reverse edge would be a cycle)."
  );
  assertCheck(
    "boundaries",
    ![routingSource, classifierSource].some((source) => /from\s+["']node:|\bBuffer\b|\bprocess\.|require\(/u.test(source)),
    "Both modules are browser-safe: no node: imports, Buffer or process."
  );
  assertCheck(
    "boundaries",
    !/import\s+(?!type\b)[^;]*from\s+["']\.\/imageSceneClassifier["']/u.test(routingSource),
    "The router imports only the classification TYPE, so loading it never pulls the lexicons into seed.ts."
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------------------------

let seedStatePromise;
function loadSeedState() {
  seedStatePromise ??= vite.ssrLoadModule("/src/data/seed.ts").then((module) => module.seedState);
  return seedStatePromise;
}

function createStateMemoryEvent(state, actorId, actorName, stateType, value) {
  return {
    id: `memory_eval_${Math.random().toString(36).slice(2)}`,
    simulationId: state.simulation.id,
    sessionId: state.simulation.activeSessionId,
    actorId,
    actorName,
    content: `[State] ${actorName} ${stateType} = ${value}`,
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

async function runGroup(scope, evaluate) {
  try {
    await evaluate();
  } catch (error) {
    assertCheck(scope, false, `Group threw before finishing: ${describeError(error)}`);
  }
}

function describeError(error) {
  return error instanceof Error ? `${error.message}${error.stack ? `\n${error.stack.split("\n").slice(1, 4).join("\n")}` : ""}` : String(error);
}

function assertCheck(scope, passed, message) {
  checks.push({
    level: passed ? "pass" : "fail",
    scope,
    message
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------------------------

// Last in the file: top-level await runs the groups immediately, and the fixture constants above must already
// be initialised when it does.
const checks = [];

try {
  const classifier = await vite.ssrLoadModule("/src/services/imageSceneClassifier.ts");
  const routing = await vite.ssrLoadModule("/src/services/imageBackendRouting.ts");

  await runGroup("normalize", () => evaluateTagNormalization(classifier));
  await runGroup("lexicon", () => evaluateTagLexicons(classifier));
  await runGroup("lexicon.gaps", () => evaluateLexiconGaps(classifier));
  await runGroup("unicode", async () => evaluateUnverifiableScript(classifier, routing, await loadSeedState()));
  await runGroup("safety-texts", async () => evaluateSafetyTexts(classifier, routing, await loadSeedState()));
  await runGroup("negative-weight", () => evaluateNegativeWeights(classifier));
  await runGroup("floor.readings", () => evaluateFloorReadings(classifier));
  await runGroup("seed-vocabulary", () => evaluateSeedVocabulary(classifier));
  await runGroup("category", () => evaluateCategoryFixtures(classifier));
  await runGroup("precedence", () => evaluatePrecedence(classifier));
  await runGroup("subjects", () => evaluateSubjectCounting(classifier));
  await runGroup("false-friends", () => evaluateFalseFriends(classifier));
  await runGroup("posture", () => evaluateAmbiguousPostures(classifier));
  await runGroup("floor.flags", () => evaluateSafetyFloorFlags(classifier));
  await runGroup("routing.resolve", () => evaluateRoutingResolution(routing));
  await runGroup("routing.available", async () => evaluateBackendAvailability(routing, await loadSeedState()));
  await runGroup("routing.decide", async () => evaluateDecisionTable(routing, classifier, await loadSeedState()));
  await runGroup("routing.floor", async () => evaluateSafetyLockMatrix(routing, classifier, await loadSeedState()));
  await runGroup("routing.reason", async () => evaluateDecisionReasons(routing, classifier, await loadSeedState()));
  await runGroup("setup-block", async () => evaluateSetupBlockTexts(routing, await loadSeedState()));
  await runGroup("composed", async () => evaluateComposedContinuity(classifier, await loadSeedState()));
  await runGroup("identity-traits", async () => evaluateIdentityTraits(classifier, routing, await loadSeedState()));
  await runGroup("boundaries", () => evaluateModuleBoundaries());
} catch (error) {
  assertCheck("load", false, `Classifier/routing modules failed to load: ${describeError(error)}`);
} finally {
  await vite.close();
}

for (const check of checks) {
  const icon = check.level === "pass" ? "PASS" : "FAIL";
  console.log(`${icon} ${check.scope}: ${check.message}`);
}

const failures = checks.filter((check) => check.level === "fail");
console.log("");
console.log(`Photoreal routing checks: ${checks.length - failures.length} passed, ${failures.length} failures`);

if (failures.length > 0) {
  process.exitCode = 1;
}
