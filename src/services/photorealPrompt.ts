import type {
  Id,
  ImageCueBodyRegion,
  ImageCueFrame,
  ImageCueShot,
  ImageCueViewpoint,
  PhotorealApiSettings,
  PhotorealStylePresetId
} from "../types";
import { inferImageCueFrameFromTags, isGarmentTag, readTagBodyRegion } from "./imageFrame";
import {
  IDENTITY_ANATOMY_TRAIT_PATTERN,
  canonicalizeSceneText,
  hasFloorPictograph,
  hasUnverifiableScript,
  isMinorOrStudentCodedSceneText,
  isPhotoNegativeSafeguardTerm,
  splitSceneSentences,
  stripHangul,
  stripPictographs
} from "./imageSceneClassifier";

// Photoreal prompt builder.
//
// NovelAI is driven by Danbooru tags. Qwen-Image and Z-Image are driven by an LLM text encoder (Qwen3-VL /
// Qwen3) that reads prose, and sending them the NovelAI caption verbatim goes wrong in two ways: the quality /
// artist / rating vocabulary means nothing to a photo model (or pulls it toward anime, which is where those
// tokens come from), and a flat comma list leaves the encoder guessing which hair colour belongs to which of two
// people. This module re-renders the SAME composed plan — identity, outfit, continuity and frame filtering are
// already applied upstream — as one English paragraph: a sentence per person, one for the setting, one for the
// camera. It never adds a garment, a prop or a body detail the plan did not carry; it translates, and it drops
// what has no meaning in a photograph.

export interface PhotorealCharacterInput {
  characterId?: Id;
  /** CharacterVisualProfile.photorealPrompt (creator NL text) — wins over identity tags. */
  appearance?: string;
  /** Composed tags for this figure in this cut. */
  tags: string[];
  /** The registered identity tags; removed from `tags` when `appearance` is given. */
  identityTags: string[];
  /** 1-based index of this figure's face reference image, if any. */
  referenceIndex?: number;
  center?: { x: number; y: number };
}

export interface PhotorealStyle {
  lead: string;
  /** The photographic LOOK (grade, grain, skin, lens). Always rendered. */
  tail: string;
  negative: string;
  /**
   * The preset's default light, rendered only when the cut states no lighting of its own. A preset that always
   * said "soft window daylight" contradicted every night street and neon-lit bar the story walked into.
   */
  ambientLight?: string;
  /** The preset's default surroundings, rendered only when the cut names no place or background. */
  ambientSetting?: string;
}

export interface PhotorealPromptInput {
  characters: PhotorealCharacterInput[];
  /** promptPlan.layers.context (+ user-rule positives). NOT the quality/style/artist layers. */
  contextTags: string[];
  frame?: ImageCueFrame;
  style: PhotorealStyle;
  /** photoreal.negativePrompt + per-character photorealNegativePrompt. */
  extraNegative?: string;
  /** qwen21 → "<image1>", edit_plus → "image 1", none → references are not named in the prompt. */
  referenceSyntax: "qwen21" | "edit_plus" | "none";
}

export interface PhotorealPromptPlan {
  prompt: string;
  negativePrompt: string;
  subjectSentences: string[];
  settingSentence: string;
  cameraSentence: string;
  /**
   * Tags removed because they mean nothing (or the wrong thing) in a photograph: NAI quality/artist/year/rating/
   * score tokens, Danbooru bookkeeping, text in a script other than Latin (Hangul, kana, Cyrillic ...), minor- or
   * student-coded words; and segments of the appearance or style text dropped for the same reasons. Weight syntax
   * whose tag survived (`1.5::smile::` → "smiling") is not listed.
   */
  droppedTags: string[];
  /** Tags left out only because the prompt hit PHOTOREAL_PROMPT_MAX_CHARS, least important first. */
  trimmedTags?: string[];
}

/**
 * Hard ceiling for the positive prompt. Both encoders accept more, but past roughly this length the last
 * sentences stop steering the picture and only dilute the subject — and the style tail and the reference
 * sentence sit at the end, where they would be the first casualties of a silent truncation.
 */
export const PHOTOREAL_PROMPT_MAX_CHARS = 1600;

/**
 * Appended once when any figure is tied to a reference image. Measured against Qwen-Image 2.1 on the live ComfyUI:
 * the softer "keep the face ...; change only the pose, expression, outfit, setting and lighting as described" kept
 * the REFERENCE's room behind the new pose (a cafe cut came back in the profile photo's white apartment). Saying
 * that the reference is identity-only and that the background comes from the description replaced the room while
 * the face held. The generic form is what the builder falls back to; `createKeepIdentitySentence` names the tokens.
 */
export const PHOTOREAL_KEEP_IDENTITY_SENTENCE =
  "Use each reference image only as the identity reference for that person: keep their face, facial features and hairstyle exactly the same, but take the pose, expression, outfit, background and lighting entirely from this description, not from the reference images.";

/** The keep-identity sentence naming the actual tokens ("<image1>", "image 1"), so the instruction is unambiguous. */
export function createKeepIdentitySentence(tokens: string[]): string {
  const unique = uniqueStrings(tokens.filter(Boolean));
  if (unique.length === 0) {
    return PHOTOREAL_KEEP_IDENTITY_SENTENCE;
  }
  if (unique.length === 1) {
    return `Use ${unique[0]} only as the identity reference: keep the person's face, facial features and hairstyle exactly the same, but take the pose, expression, outfit, background and lighting entirely from this description, not from the reference image.`;
  }
  const listed = `${unique.slice(0, -1).join(", ")} and ${unique[unique.length - 1]}`;
  return `Use ${listed} only as identity references: keep each person's face, facial features and hairstyle exactly the same as in their own reference, but take the poses, expressions, outfits, background and lighting entirely from this description, not from the reference images.`;
}

/** Shared photo-model negatives: what an anime-trained caption drags in, plus the usual anatomy failures. */
const PHOTO_NEGATIVE_BASE =
  "anime, cartoon, illustration, 3d render, cgi, painting, plastic skin, oversaturated, watermark, text, logo, extra fingers, deformed hands, distorted face";

export const PHOTOREAL_STYLE_PRESETS: Record<
  Exclude<PhotorealStylePresetId, "custom">,
  { label: string; description: string } & PhotorealStyle
> = {
  daily_look: {
    label: "데일리룩 · 핀터레스트/샤오홍슈",
    description:
      "스마트폰으로 찍은 듯한 일상 패션 사진. 부드러운 자연광, 크림·화이트 톤 실내, 대비가 낮은 파스텔 보정, 결이 살아 있는 피부.",
    lead: "A candid, true-to-life smartphone photograph in the style of a Korean and Chinese daily-look fashion post",
    tail:
      "Gentle pastel color grade with lifted shadows and low contrast, natural skin texture with visible pores, subtle phone-camera grain, sharp focus on the subject, realistic proportions",
    ambientLight: "Soft diffused natural daylight",
    ambientSetting: "a clean, minimal room with cream and white tones",
    negative:
      "anime, cartoon, illustration, 3d render, cgi, painting, doll, plastic skin, airbrushed skin, oversaturated, harsh flash, heavy hdr, watermark, text, logo, extra fingers, deformed hands, distorted face"
  },
  film_portrait: {
    label: "필름 인물 사진",
    description: "35mm 필름 인물 사진. Kodak Portra 400 색감, 자연광, 얕은 심도와 부드러운 보케, 고운 필름 입자.",
    lead: "A 35mm film portrait photograph",
    tail: "Kodak Portra 400 tones, shallow depth of field with soft bokeh, fine film grain, natural skin texture, realistic proportions",
    ambientLight: "Natural light",
    negative: `${PHOTO_NEGATIVE_BASE}, airbrushed skin, digital sharpening`
  },
  cinematic: {
    label: "시네마틱 스틸",
    description: "영화의 한 장면 같은 스틸. 아나모픽 렌즈, 광원이 분명한 무드 조명, 공기 원근감, 필름 컬러 그레이딩.",
    lead: "A cinematic film still",
    tail: "Anamorphic lens, atmospheric depth, filmic color grade, natural skin texture, realistic proportions",
    ambientLight: "Motivated moody lighting",
    negative: `${PHOTO_NEGATIVE_BASE}, flat lighting, subtitles`
  },
  studio_editorial: {
    label: "스튜디오 화보",
    description: "고급 패션 화보. 소프트박스 스튜디오 조명, 이음매 없는 배경지, 선명한 디테일.",
    lead: "A high-end fashion editorial photograph",
    tail: "Crisp detail, natural skin texture, realistic proportions",
    ambientLight: "Softbox studio lighting",
    ambientSetting: "a seamless studio backdrop",
    negative: `${PHOTO_NEGATIVE_BASE}, airbrushed skin, cluttered background`
  }
};

export function resolvePhotorealStyle(
  settings: Pick<PhotorealApiSettings, "stylePreset" | "stylePrompt" | "negativePrompt">
): PhotorealStyle {
  const extraStyle = sanitizePhotoText(settings.stylePrompt);
  if (settings.stylePreset === "custom") {
    // A custom style is the creator's whole description of the medium. Its first sentence leads the prompt (the
    // encoder weighs the opening most, and that is where a preset says "a smartphone photograph"); anything after
    // it closes the prompt the way a preset's tail does.
    const { lead, tail } = splitLeadSentence(extraStyle);
    return { lead, tail, negative: mergeNegativePrompts(settings.negativePrompt) };
  }
  const preset = resolveStylePreset(settings.stylePreset);
  return {
    lead: preset.lead,
    tail: [stripTerminalPunctuation(preset.tail), stripTerminalPunctuation(extraStyle)].filter(Boolean).join(", "),
    negative: mergeNegativePrompts(preset.negative, settings.negativePrompt),
    ambientLight: preset.ambientLight,
    ambientSetting: preset.ambientSetting
  };
}

/**
 * The segments of a creator's free text (a photoreal appearance, a style, the prompt a profile image was made
 * from) as the safety floor must read them: every segment, with its Korean taken out. The builder drops Korean, and
 * the settings and builder screens tell the creator so ("한글은 프롬프트에서 빠집니다"), so a Korean appearance keeps
 * that documented fallback to the saved tags instead of locking every cut. Whatever else a segment carries is read:
 * skipping a segment because it held Hangul let one Hangul mark carry "A naked woman" past the floor. Every other
 * script stays in for the classifier to lock on: the builder drops it too, but nothing in an English-only field
 * should be written in kana or Cyrillic, and nothing here can check what it says. Segmentation is the builder's own
 * (splitSceneSentences), so whatever the builder keeps is a segment that was read. A segment of emoji alone is kept
 * when one of them is a pictograph the floor reads ("🍆💦"): the builder strips pictographs, but a typed one says what
 * the creator asked for. Any other symbol-only segment is not text, and still yields no segment.
 */
export function selectPhotoTextForScreening(text: string | undefined): string[] {
  return splitSceneSentences(text)
    .flat()
    .map((segment) => (KOREAN_SCRIPT_PATTERN.test(segment) ? stripHangul(segment) : segment))
    .filter((segment) => /[\p{L}\p{N}]/u.test(segment) || hasFloorPictograph(segment));
}

/**
 * Everything the style puts into a photo prompt — the preset's lead, tail and ambient defaults, and the creator's
 * stylePrompt (which is all of a custom style) — as the safety floor must read it (see selectPhotoTextForScreening).
 * The ambient defaults are listed although a cut that names its own light or place does not render them: reading
 * more than is sent costs nothing, reading less is how a style slipped past the floor.
 */
export function describePhotorealStyleForScreening(settings: Pick<PhotorealApiSettings, "stylePreset" | "stylePrompt">): string[] {
  const preset = settings.stylePreset === "custom" ? undefined : resolveStylePreset(settings.stylePreset);
  const presetTexts = preset ? [preset.lead, preset.tail, preset.ambientLight, preset.ambientSetting] : [];
  return [...presetTexts, settings.stylePrompt].flatMap((text) => selectPhotoTextForScreening(text));
}

/**
 * An unknown id (state written by a newer build, or hand-edited) falls back to the default look, not to none. Own
 * properties only, so a corrupted id like "constructor" cannot resolve to something on Object.prototype.
 */
function resolveStylePreset(stylePreset: PhotorealApiSettings["stylePreset"]): PhotorealStyle {
  return Object.prototype.hasOwnProperty.call(PHOTOREAL_STYLE_PRESETS, stylePreset)
    ? PHOTOREAL_STYLE_PRESETS[stylePreset as Exclude<PhotorealStylePresetId, "custom">]
    : PHOTOREAL_STYLE_PRESETS.daily_look;
}

export function buildPhotorealPrompt(input: PhotorealPromptInput): PhotorealPromptPlan {
  const sink: TagSink = { dropped: [], negative: [] };
  const referenceSyntax = input.referenceSyntax === "qwen21" || input.referenceSyntax === "edit_plus" ? input.referenceSyntax : "none";
  const order = { next: 0 };
  const setting = createSettingModel();
  const camera: CameraModel = { lens: [], compositionTags: [] };

  const figures = (Array.isArray(input.characters) ? input.characters : [])
    .filter((character): character is PhotorealCharacterInput => Boolean(character) && typeof character === "object")
    .map((character) => buildFigureModel(character, referenceSyntax, setting, camera, sink, order));

  // Character-level tags the model left in the shared caption. With exactly one person they can only be about
  // that person, so they join that person's sentence. With several they cannot be attributed, and gluing them onto
  // whoever happens to be first mislabels them — they become a sentence about the whole group instead.
  const contextFigure = createFigureModel(undefined);
  const detailTarget = figures.length === 1 ? figures[0] : contextFigure;
  let contextNoHumans = false;
  for (const tag of cleanTagList(input.contextTags, sink)) {
    if (detailTarget.identityKeys?.has(tag.key)) {
      continue;
    }
    if (absorbContextTag(setting, camera, contextFigure, detailTarget, tag, order) === "no_humans") {
      contextNoHumans = true;
    }
  }
  for (const figure of [...figures, contextFigure]) {
    finalizeFigure(figure, order);
  }

  const groupFigure = figures.length >= 2 && figureHasDetail(contextFigure) ? contextFigure : undefined;
  // No registered figure, but the caption still describes people (count tags, or an expression with no owner):
  // the context becomes the subject, and the adult-noun rule applies to it like to anyone else.
  const contextSubject =
    figures.length === 0 && !contextNoHumans && (countTotal(contextFigure.counts) > 0 || figureHasDetail(contextFigure))
      ? contextFigure
      : undefined;
  const hasHumans = figures.length > 0 || contextSubject !== undefined;
  // A people-free cut says so out loud: every style lead describes a fashion photograph, which an encoder will
  // happily populate with a model unless the prompt states that the frame is empty.
  const declareNoPeople =
    !hasHumans && (contextNoHumans || setting.places.length + setting.props.length + setting.extras.length > 0);

  const frame: CameraFrame | undefined = input.frame ?? inferFrameFromTags(camera.compositionTags);
  // A photo model renders every garment it is told about, and widens the shot to show them: a face close-up whose
  // caption still listed jeans came back waist-up on 3/3 seeds (measured on Qwen-Image 2.1). NovelAI's frame filter
  // deliberately keeps AUTHORED garments; the photo prompt cannot afford that, so what the crop hides is dropped.
  const visibleRegions =
    frame?.shot && Array.isArray(frame.visibleRegions) && frame.visibleRegions.length > 0 ? new Set(frame.visibleRegions) : undefined;
  if (visibleRegions) {
    for (const figure of [...figures, contextFigure]) {
      figure.clothing = figure.clothing.filter((item) => {
        const region = readTagBodyRegion(item.source);
        if (region && !visibleRegions.has(region)) {
          sink.dropped.push(item.source);
          return false;
        }
        return true;
      });
    }
  }
  const cameraPhrases = describeCamera(frame, hasHumans);
  const lead = stripTerminalPunctuation(sanitizePhotoText(input.style?.lead, sink.dropped));
  // The preset's default light and surroundings only fill what the cut left unsaid: a cut that names its own
  // lighting or place keeps it, instead of being told it is also in a cream-white room in soft daylight.
  const lightingKnown = keptPhrases(setting.lighting).length > 0;
  const placeKnown =
    keptPhrases(setting.places).length > 0 ||
    keptPhrases(setting.backgrounds).length > 0 ||
    keptPhrases(setting.indoorOutdoor).some((phrase) => phrase !== "indoors");
  const ambientLight = lightingKnown ? "" : stripTerminalPunctuation(sanitizePhotoText(input.style?.ambientLight, sink.dropped));
  const ambientSetting =
    placeKnown || !hasHumans ? "" : stripTerminalPunctuation(sanitizePhotoText(input.style?.ambientSetting, sink.dropped));
  const tail = stripTerminalPunctuation(sanitizePhotoText(input.style?.tail, sink.dropped));
  const layout = orderFiguresForLayout(figures);
  const referenceTokens = referenceSyntax !== "none" ? figures.map((figure) => figure.referenceToken ?? "").filter(Boolean) : [];
  const referenceSentence = referenceTokens.length > 0 ? createKeepIdentitySentence(referenceTokens) : "";

  const render = (): RenderedPrompt => {
    const subjectSentences = renderSubjectSentences(layout, contextSubject, groupFigure);
    const renderedSetting = renderSettingSentence(setting, hasHumans, declareNoPeople);
    const settingSentence = ambientSetting
      ? joinPromptParts([renderedSetting, `${capitalizeFirst(renderedSetting ? `the surroundings are ${ambientSetting}` : `set in ${ambientSetting}`)}.`])
      : renderedSetting;
    const cameraSentence = [...cameraPhrases, ...keptPhrases(camera.lens)].join(", ");
    // The crop leads, right after the medium. Stated after the people and the room it lost to them: the same
    // face close-up came back as a medium shot until the camera sentence moved up front (measured, 3/3 seeds).
    const body = joinPromptParts([
      lead ? `${capitalizeFirst(lead)}.` : "",
      cameraSentence ? `${capitalizeFirst(cameraSentence)}.` : "",
      ...subjectSentences,
      settingSentence,
      ambientLight ? `${capitalizeFirst(ambientLight)}.` : "",
      tail ? `${capitalizeFirst(tail)}.` : ""
    ]);
    return { body, subjectSentences, settingSentence, cameraSentence };
  };

  const trimmedTags: string[] = [];
  const budget = PHOTOREAL_PROMPT_MAX_CHARS - (referenceSentence ? referenceSentence.length + 1 : 0);
  let rendered = render();
  if (rendered.body.length > budget) {
    // Least important detail goes first (unknown background extras, then props, then per-person extras ...), with
    // a re-render after every removal so exactly as much is cut as the budget requires and no more. Only what is
    // actually rendered is in the pool, so trimmedTags never reports a cut that shortened nothing.
    const renderedFigures = [...figures, ...(groupFigure ? [groupFigure] : []), ...(contextSubject ? [contextSubject] : [])];
    const trimmable = [...Object.values(setting).flat(), ...camera.lens, ...renderedFigures.flatMap(figureItems)];
    while (rendered.body.length > budget) {
      const victim = pickTrimVictim(trimmable);
      if (!victim) {
        break;
      }
      victim.kept = false;
      trimmedTags.push(victim.source);
      rendered = render();
    }
  }

  return {
    prompt: joinPromptParts([clampToLength(rendered.body, budget), referenceSentence]),
    negativePrompt: mergeNegativePrompts(
      input.style?.negative,
      input.extraNegative,
      // What the plan itself marked unwanted (a NovelAI negative weight, "no text") is unwanted in a photo too.
      sink.negative.join(", ")
    ),
    subjectSentences: rendered.subjectSentences,
    settingSentence: rendered.settingSentence,
    cameraSentence: rendered.cameraSentence,
    droppedTags: uniqueStrings(sink.dropped),
    trimmedTags
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Tag cleanup: NovelAI syntax, NAI-only vocabulary, text in other scripts.
// ---------------------------------------------------------------------------------------------------------------

interface CleanTag {
  /** Readable form: weights and brackets stripped, underscores turned into spaces. */
  text: string;
  /** Lowercase form every lexicon is matched against. */
  key: string;
  /** What the plan carried, reported in droppedTags / trimmedTags. */
  source: string;
}

interface TagSink {
  dropped: string[];
  negative: string[];
}

/** Korean: dropped from photo text like every other non-Latin script, but by contract rather than for safety. */
const KOREAN_SCRIPT_PATTERN = /\p{Script=Hangul}/u;

/** Tokens that only mean something to an anime tag model, or to Danbooru's bookkeeping. */
const NAI_ONLY_TAG_PATTERNS: RegExp[] = [
  /^(?:masterpiece|best quality|amazing quality|great quality|good quality|normal quality|high quality|top quality|very aesthetic|aesthetic|top aesthetic|absurdres|incredibly absurdres|highres|newest|recent|very awa)$/u,
  /^year \d{4}$/u,
  /^rating\s*[:_ ]/u,
  /^score[ _]?\d/u,
  /^source[ _](?:anime|pony|furry|cartoon|manga)$/u,
  /^artist\s*:/u,
  /^anime\b/u,
  /\b(?:anime|manga|cartoon)\s+(?:style|coloring|colouring|screencap|screenshot|art|aesthetic|illustration)\b/u,
  /^(?:manga|illustration|2d|3d|cel shading|cel-shading|cel shaded|flat color|lineart|line art|sketch|comic|chibi|oekaki|pixel art|cartoon|toon|digital art|concept art|official art|game cg|key visual|promotional art|cover art)$/u,
  /\((?:medium|style)\)$/u,
  /^(?:commentary|commentary request|english commentary|translated|translation request|bad id|bad pixiv id|md5 mismatch|third-party edit|scan|dated|copyright name|character name|speech bubble|english text|censored|mosaic censoring|bar censor|border|letterboxed|sfw|nsfw|solo|solo focus|male focus|female focus|sweatdrop|multiple views|split screen|spoken heart|musical note)$/u
];

/** Technical defects a tag caption sometimes leaks into the positive side; in a photo prompt they belong in the negative. */
const TECHNICAL_NEGATIVE_PATTERN =
  /^(?:text|watermark|logo|signature|artist name|username|jpeg artifacts|lowres|blurry|worst quality|bad quality|low quality|bad anatomy|bad hands|bad face|extra fingers|extra limbs|missing fingers|deformed|distorted|disfigured|mutated|ugly|duplicate)$/u;
const NEGATED_TECHNICAL_PATTERN = /^no (text|watermarks?|logos?|signatures?|artist name|username|speech bubbles?)$/u;

// Words the photoreal side never renders — anything that codes the person as a minor, and student identities that
// would put the subject back into a school-age frame — are the classifier's own (isMinorOrStudentCodedSceneText).
// The router already locks minor-coded cuts to NovelAI; this is the same floor applied to the words themselves, so a
// photo prompt can only ever describe an adult whatever reached it. School GARMENTS are not affected — "school
// uniform" is clothing, and the subject stays an adult.
//
// Body-identity traits (futanari and synonyms) are routed as identity, not as explicit content, which is only safe
// because the photo prompt never carries them: the photo shows the clothed person the rest of the caption describes.

/**
 * Garment qualifiers a photo model renders literally. An anime model stylises a "translucent bodysuit"; a photo
 * model shows the body through it — nudity the router never classified, because the cut is a clothed portrait. The
 * garment stays, opaque: "translucent tactical bodysuit" → "tactical bodysuit". A bare "see-through" tag with no
 * garment is dropped. (Found on the shipped adult seed's default outfits during QA.)
 */
const SHEER_QUALIFIER_PATTERN = /\b(?:see-?through|see through|translucent|transparent|sheer)\s+/u;
const BARE_SHEER_TAG_PATTERN = /^(?:see-?through|see through|translucent|transparent|sheer)(?: (?:clothes|clothing|fabric|material))?$/u;
/** The same qualifier inside prose, with the article in front of it so "a sheer orange top" can become "an orange top". */
const SHEER_QUALIFIER_IN_TEXT_PATTERN = /\b(?:(a|an)\s+)?(?:see-?through|see through|translucent|transparent|sheer)\s+(?=(\p{L}+))/giu;

/**
 * Body descriptors that a photograph turns into a realistic depiction of an eating disorder. Stylised in anime;
 * not something the photo side should ever render. The rest of the build ("slender", "skinny") is untouched.
 */
const EATING_DISORDER_BODY_PATTERN = /\b(?:anorexic|anorexia|bulimic|bulimia|emaciated|skeletal|skin and bones|underweight)\b/u;

/** Danbooru emoticon tags, mapped onto the expression they stand for. Other punctuation-only tags are dropped. */
const EMOTICON_TAGS: Record<string, string> = {
  ":d": "big smile",
  ":)": "smile",
  "^_^": "smile",
  "^^": "smile",
  ";d": "wink",
  ";)": "wink",
  ":o": "open mouth",
  ":p": "tongue out",
  ";p": "tongue out",
  xd: "laughing",
  ">_<": "closed eyes",
  v: "peace sign"
};

function cleanTagList(raw: readonly unknown[] | undefined, sink: TagSink): CleanTag[] {
  const kept: CleanTag[] = [];
  const seen = new Set<string>();
  for (const value of Array.isArray(raw) ? raw : []) {
    if (typeof value !== "string") {
      continue;
    }
    const parts = expandWeightedTag(value);
    if (parts.length === 0 && value.trim()) {
      // Nothing but pictographs: stripped, and reported like any other tag the photo prompt leaves out.
      sink.dropped.push(canonicalizeSceneText(value).trim());
    }
    for (const part of parts) {
      const tag = cleanSingleTag(part, sink);
      if (tag && !seen.has(tag.key)) {
        seen.add(tag.key);
        kept.push(tag);
      }
    }
  }
  return kept;
}

interface WeightedPart {
  body: string;
  source: string;
  weight?: number;
}

/**
 * Splits NovelAI weight syntax off a tag. `1.5::smile::` keeps `smile`; a weighted group `1.2::black hair, long
 * hair::` becomes two tags; a negative weight marks the tag unwanted. A caption split naively on commas can also
 * hand over the halves `1.2::black hair` and `long hair::`, so each marker is stripped on its own.
 */
function expandWeightedTag(raw: string): WeightedPart[] {
  // Canonical first (NFKC, invisible characters out, accents folded), as the classifier read it: a fullwidth
  // "ｎｕｄｅ" or a zero-width-split "nu\u200bde" is the plain word to the encoder, so it is the plain word here too.
  const source = canonicalizeSceneText(raw).trim();
  // Pictographs never reach a photo prompt: the encoders read "👙" as a bikini, and the floor has already read and
  // counted the ones that mean something (stripPictographs).
  let body = stripPictographs(source);
  let weight: number | undefined;
  const weighted = body.match(/^(-?\d+(?:\.\d+)?)\s*::/u);
  if (weighted) {
    weight = Number(weighted[1]);
    body = body.slice(weighted[0].length);
  } else if (body.startsWith("::")) {
    body = body.slice(2);
  }
  body = body.replace(/\s*::\s*$/u, "");
  return body
    .split(/[,;\n|]/u)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => ({ body: part, source, weight }));
}

function cleanSingleTag(part: WeightedPart, sink: TagSink): CleanTag | undefined {
  const text = normalizeTagText(part.body);
  const key = text.toLowerCase();
  const source = part.source;
  if (!key) {
    return undefined;
  }
  if (hasUnverifiableScript(text)) {
    // Letters outside a–z: Korean, kana, CJK, Cyrillic or Greek look-alikes. The encoders read most of them, and
    // nothing here can check what such a tag would add to the picture (the router locks a cut that carries one;
    // this keeps it out of any prompt regardless). Korean, the common case, is noise to them at best.
    sink.dropped.push(source);
    return undefined;
  }
  if (part.weight !== undefined && part.weight <= 0) {
    sink.dropped.push(source);
    // "Less of this" is unwanted in a photo too, except for what keeps the subject covered, safe and adult: a
    // negative weight on clothing, on "sfw" or on "adult" is NovelAI's undressing or age-down idiom (`-2::apparel::`,
    // `-1::adult::`), and the same photo negative steers the sampler there at any CFG above 1. The classifier locks
    // such a cut; the term just goes.
    if (part.weight < 0 && !isPhotoNegativeSafeguardTerm(text)) {
      sink.negative.push(text);
    }
    return undefined;
  }
  const negated = key.match(NEGATED_TECHNICAL_PATTERN);
  if (negated) {
    sink.dropped.push(source);
    sink.negative.push(negated[1].replace(/s$/u, ""));
    return undefined;
  }
  if (TECHNICAL_NEGATIVE_PATTERN.test(key)) {
    sink.dropped.push(source);
    sink.negative.push(text);
    return undefined;
  }
  if (NAI_ONLY_TAG_PATTERNS.some((pattern) => pattern.test(key)) || isByArtistTag(key)) {
    sink.dropped.push(source);
    return undefined;
  }
  // The floor locks a minor-coded cut before it gets here, and whatever still arrives is never worded as a minor or
  // a student.
  if (isMinorOrStudentCodedSceneText(key)) {
    sink.dropped.push(source);
    return undefined;
  }
  if (IDENTITY_ANATOMY_TRAIT_PATTERN.test(key) || EATING_DISORDER_BODY_PATTERN.test(key) || BARE_SHEER_TAG_PATTERN.test(key)) {
    sink.dropped.push(source);
    return undefined;
  }
  if (SHEER_QUALIFIER_PATTERN.test(key)) {
    const opaque = text.replace(new RegExp(SHEER_QUALIFIER_PATTERN.source, "giu"), "").trim();
    if (!/\p{L}{2,}/u.test(opaque)) {
      sink.dropped.push(source);
      return undefined;
    }
    return { text: opaque, key: opaque.toLowerCase(), source };
  }
  if (/^(?:side|front|back|profile)$/u.test(key)) {
    // A bare viewpoint word leaked from a composition tag; the camera sentence already states the viewpoint.
    sink.dropped.push(source);
    return undefined;
  }
  const worn = text.match(/^(?:wearing|dressed in|clad in)\s+(.+)$/iu);
  if (worn && /\p{L}{2,}/u.test(worn[1])) {
    // The rendered sentence supplies its own "wearing"; keeping the tag's would print it twice.
    const garment = worn[1].trim();
    return { text: garment, key: garment.toLowerCase(), source };
  }
  const emoticon = EMOTICON_TAGS[key];
  if (emoticon) {
    return { text: emoticon, key: emoticon, source };
  }
  if (!/\p{L}{2,}/u.test(text)) {
    // Punctuation-only tags (":3", "o_o") are anime face shorthand with no photographic meaning.
    sink.dropped.push(source);
    return undefined;
  }
  return { text, key, source };
}

function normalizeTagText(value: string): string {
  let text = value.replace(/\\([()])/gu, "$1").replace(/[{}[\]]/gu, "");
  // Stable-Diffusion style emphasis — `(tag:1.2)`, `((tag))` — wraps the whole tag.
  for (let guard = 0; guard < 4; guard += 1) {
    const wrapped = text.trim().match(/^\((.*?)(?::\s*-?\d+(?:\.\d+)?)?\)$/u);
    if (!wrapped) {
      break;
    }
    text = wrapped[1];
  }
  return collapseWhitespace(text.replace(/_/gu, " ")).replace(/[.!?。]+$/u, "");
}

/** `by <artist>` tags. Location phrases ("by the window", "by window") are kept. */
function isByArtistTag(key: string): boolean {
  const match = key.match(/^by\s+(.+)$/u);
  if (!match) {
    return false;
  }
  const rest = match[1];
  if (/^(?:the|a|an|her|his|their|my|your)\b/u.test(rest)) {
    return false;
  }
  return !(
    PLACE_PATTERN.test(rest) ||
    PROP_PATTERN.test(rest) ||
    /\b(?:window|door|wall|sea|ocean|lake|river|fire|fireplace|pool|road|bed|desk|table|tree|shore)\b/u.test(rest)
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Lexicons. Every lookup runs on CleanTag.key (lowercase, weights stripped, underscores → spaces).
// ---------------------------------------------------------------------------------------------------------------

type Gender = "female" | "male" | "other";
type PronounKey = Gender | "plural";

const COUNT_TAG_PATTERN = /^(\d+)(\+)?\s*(girls?|boys?|others?)$/u;
const MULTIPLE_TAG_PATTERN = /^multiple (girls|boys|others)$/u;
const NO_HUMANS_PATTERN = /^(?:no humans?|no people|nobody|no one)$/u;
const GENDER_WORD_PATTERN =
  /^(?:(mature|young|old|elderly|middle-aged|adult)\s+)?(girl|woman|female|lady|boy|man|male|guy|gentleman|person|other)$/u;

const COMPOSITION_PATTERN =
  /^(?:close-up|close up|closeup|extreme close-up|portrait|upper body|lower body|cowboy shot|full body|wide shot|medium shot|long shot|face focus|head shot|headshot|bust shot|from side|from behind|from above|from below|high angle|low angle|dutch angle|pov|pov hands|over the shoulder|over-the-shoulder|profile|profile view|side view|front view|rear view|back view|three-quarter view|straight-on|feet out of frame|head out of frame|out of frame|cropped legs|cropped torso|cropped|establishing shot|bird's-eye view|birds-eye view|aerial view|scenery|landscape|panorama)$/u;

const LENS_PHRASES: Record<string, string> = {
  "depth of field": "shallow depth of field",
  "shallow depth of field": "shallow depth of field",
  bokeh: "soft bokeh",
  "blurry background": "softly blurred background",
  "blurry foreground": "softly blurred foreground",
  "motion blur": "motion blur",
  fisheye: "fisheye lens",
  "wide-angle": "wide-angle lens",
  "wide angle lens": "wide-angle lens",
  telephoto: "telephoto lens",
  "film grain": "film grain",
  "chromatic aberration": "slight chromatic aberration",
  vignette: "soft vignette",
  vignetting: "soft vignette",
  "long exposure": "long exposure",
  "tilt-shift": "tilt-shift lens",
  "soft focus": "soft focus",
  "sharp focus": "sharp focus"
};

const EXPRESSION_PHRASES: Record<string, string> = {
  smile: "smiling",
  smiling: "smiling",
  "light smile": "smiling softly",
  "slight smile": "smiling softly",
  "soft smile": "smiling softly",
  "gentle smile": "smiling gently",
  "big smile": "smiling broadly",
  "happy smile": "smiling happily",
  grin: "grinning",
  grinning: "grinning",
  "evil smile": "smiling mischievously",
  smirk: "smirking",
  smug: "looking smug",
  laughing: "laughing",
  laugh: "laughing",
  happy: "looking happy",
  blush: "blushing slightly",
  blushing: "blushing slightly",
  "light blush": "blushing faintly",
  "full-face blush": "blushing deeply",
  "nose blush": "blushing slightly",
  crying: "with tears in {pron} eyes",
  "crying with eyes open": "with tears in {pron} eyes",
  tears: "with tears in {pron} eyes",
  "teary eyes": "with teary eyes",
  "tearing up": "with tears welling up in {pron} eyes",
  "streaming tears": "with tears streaming down {pron} face",
  sad: "looking sad",
  angry: "looking angry",
  annoyed: "looking annoyed",
  surprised: "looking surprised",
  shocked: "looking shocked",
  embarrassed: "looking embarrassed",
  nervous: "looking nervous",
  worried: "looking worried",
  scared: "looking scared",
  confused: "looking confused",
  sleepy: "looking sleepy",
  tired: "looking tired",
  bored: "looking bored",
  determined: "looking determined",
  thinking: "looking thoughtful",
  serious: "with a serious expression",
  expressionless: "with a neutral expression",
  "blank stare": "with a blank stare",
  pout: "pouting",
  pouting: "pouting",
  frown: "frowning",
  frowning: "frowning",
  "furrowed brow": "with a furrowed brow",
  "open mouth": "with {pron} mouth open",
  "closed mouth": "with {pron} mouth closed",
  "parted lips": "with parted lips",
  "closed eyes": "with {pron} eyes closed",
  "eyes closed": "with {pron} eyes closed",
  "half-closed eyes": "with half-closed eyes",
  "one eye closed": "winking",
  wink: "winking",
  "wide-eyed": "wide-eyed",
  "narrowed eyes": "with narrowed eyes",
  "tongue out": "sticking {pron} tongue out",
  "biting lip": "biting {pron} lip",
  "lip biting": "biting {pron} lip",
  "clenched teeth": "with clenched teeth",
  "seductive smile": "with a seductive smile",
  "naughty face": "with a playful expression",
  "looking at viewer": "looking at the camera",
  "looking at camera": "looking at the camera",
  "eye contact": "looking at the camera",
  "facing viewer": "facing the camera",
  "looking away": "looking away",
  "looking back": "looking back over {pron} shoulder",
  "looking down": "looking down",
  "looking up": "looking up",
  "looking to the side": "looking to the side",
  "looking aside": "looking to the side",
  "looking afar": "gazing into the distance",
  "looking into the distance": "gazing into the distance",
  "looking at another": "looking at the other person",
  "looking at each other": "looking at each other",
  "looking at phone": "looking at {pron} phone"
};

/** Free-form expressions ("energetic innocent smile", "playful smirk") read as a noun phrase. */
const EXPRESSION_NOUN_PATTERN = /\b(?:smile|smirk|grin|frown|pout|expression|gaze|stare|glare)$/u;

const POSE_PHRASES: Record<string, string> = {
  standing: "standing",
  sitting: "sitting",
  seated: "seated",
  lying: "lying down",
  "lying down": "lying down",
  "on back": "lying on {pron} back",
  "on stomach": "lying on {pron} stomach",
  "on side": "lying on {pron} side",
  kneeling: "kneeling",
  squatting: "squatting",
  crouching: "crouching",
  walking: "walking",
  running: "running",
  jumping: "jumping",
  leaning: "leaning",
  "leaning forward": "leaning forward",
  "leaning back": "leaning back",
  "arms crossed": "with {pron} arms crossed",
  "crossed arms": "with {pron} arms crossed",
  "hand on hip": "with a hand on {pron} hip",
  "hands on hips": "with {pron} hands on {pron} hips",
  "arms up": "with {pron} arms raised",
  "arms raised": "with {pron} arms raised",
  "arm up": "with one arm raised",
  "hand up": "with one hand raised",
  "arms behind back": "with {pron} arms behind {pron} back",
  "hands in pockets": "with {pron} hands in {pron} pockets",
  "hand in pocket": "with a hand in {pron} pocket",
  "head tilt": "tilting {pron} head",
  "head rest": "resting {pron} head on {pron} hand",
  "peace sign": "making a peace sign",
  "v sign": "making a peace sign",
  "double v": "making peace signs with both hands",
  waving: "waving",
  salute: "saluting",
  "thumbs up": "giving a thumbs up",
  pointing: "pointing",
  selfie: "taking a selfie",
  "mirror selfie": "taking a mirror selfie",
  "taking picture": "taking a photo",
  "taking photo": "taking a photo",
  "crossed legs": "with {pron} legs crossed",
  "legs crossed": "with {pron} legs crossed",
  "knees up": "with {pron} knees drawn up",
  "hugging own legs": "hugging {pron} knees",
  stretching: "stretching",
  contrapposto: "standing in contrapposto",
  "all fours": "on all fours",
  "bent over": "bending over",
  "fighting stance": "in a fighting stance",
  "holding hands": "holding hands",
  hug: "hugging",
  hugging: "hugging",
  kiss: "kissing",
  kissing: "kissing",
  cuddling: "cuddling",
  embrace: "embracing"
};

const POSE_VERB_PATTERN =
  /^(?:standing|sitting|lying|kneeling|squatting|crouching|leaning|walking|running|jumping|stretching|bending|reaching|turning|posing|dancing|holding|carrying|drinking|eating|reading|writing|typing|sleeping|waving|hugging|kissing|pointing|touching|grabbing|playing|riding|cooking|shopping|smoking|singing|resting|lounging|relaxing|adjusting|tying|fixing|brushing|applying|pulling|lifting|wiping|covering|cupping|clasping|hiding|facing|peeking|sipping|using|wearing|taking|making|putting|opening|closing|cleaning|washing|stepping|climbing)\b/u;
const POSE_LIMB_PATTERN =
  /^(?:hands?|arms?|legs?|fingers?|knees?|feet|foot|head)\s+(?:on|in|behind|between|over|under|up|down|together|apart|crossed|raised|out|around|to|at|near|against)\b/u;
const POSE_NOUN_PATTERN = /\b(?:posture|pose|stance)$/u;
/** A phrase that is already a location relative to something ("by the window", "on bed"). */
const PREPOSITIONAL_PHRASE_PATTERN =
  /^(?:by|on|in|at|near|under|behind|beside|next to|against|inside|outside|over|across|around|between|along)\s+\S/u;
/** "skirt hold", "shirt lift": the garment is what the hands are doing something to. */
const GARMENT_ACTION_PATTERN = /^(.+?)\s+(hold|lift|pull|tug|grab)$/u;
const GARMENT_ACTION_VERBS: Record<string, string> = {
  hold: "holding",
  lift: "lifting",
  pull: "pulling",
  tug: "tugging",
  grab: "grabbing"
};

const GARMENT_PATTERN =
  /\b(?:shirt|t-shirt|tee|blouse|top|crop top|tank top|tube top|camisole|sweater|jumper|cardigan|hoodie|sweatshirt|turtleneck|knit|knitwear|jacket|blazer|coat|parka|vest|dress|sundress|gown|bodycon|skirt|miniskirt|shorts|hot pants|pants|trousers|jeans|leggings|tights|pantyhose|stockings|thighhighs|thigh-highs|thigh highs|kneehighs|kneehigh|socks|shoes|sneakers|boots|heels|sandals|loafers|slippers|flats|uniform|serafuku|gakuran|suit|tuxedo|kimono|yukata|hanbok|qipao|cheongsam|apron|pajamas|pyjamas|nightgown|robe|bathrobe|swimsuit|bikini|lingerie|bra|panties|underwear|overalls|jumpsuit|romper|bodysuit|leotard|cape|cloak|scarf|muffler|gloves|mittens|hat|cap|beanie|beret|headband|hairband|hair ornament|hairclip|hair clip|hairpin|hair ribbon|hair bow|hair flower|scrunchie|ribbon|bow|bowtie|necktie|tie|choker|necklace|pendant|earrings?|bracelet|ring|wristwatch|watch|glasses|eyewear|sunglasses|mask|belt|bag|handbag|purse|backpack|tote|clutch|collar|sleeves?|sleeveless|strapless|off-shoulder|off shoulder|neckline|denim|leather|lace|frills?|ruffles?|clothes|clothing|outfit|attire|costume|sportswear|loungewear|skinnies|skinny jeans|thong)\b/u;

const NAI_GARMENT_RENAMES: Record<string, string> = {
  thighhighs: "thigh-high socks",
  "thigh highs": "thigh-high socks",
  kneehighs: "knee-high socks",
  kneehigh: "knee-high socks",
  serafuku: "sailor-collar school uniform",
  gakuran: "high-collared school uniform",
  "hot pants": "short shorts",
  miniskirt: "mini skirt"
};

const BARE_PHRASES: Record<string, string> = {
  barefoot: "barefoot",
  "bare shoulders": "with bare shoulders",
  "bare legs": "with bare legs",
  "bare arms": "with bare arms",
  "bare back": "with a bare back",
  "bare midriff": "with a bare midriff",
  midriff: "with a bare midriff",
  navel: "with {pron} navel showing",
  collarbone: "with {pron} collarbone showing",
  cleavage: "showing cleavage"
};

/** Objects a person holds or has with them; in a person's own tags they stay with that person. */
const HANDHELD_PATTERN =
  /\b(?:phone|smartphone|cellphone|mobile phone|cup|mug|teacup|book|notebook|umbrella|bouquet|flowers?|microphone|camera|guitar|gift|present|letter|envelope|teddy bear|stuffed toy|ice cream|drink|bottle|shopping bag|tablet|fan)$/u;

const HAIR_LENGTH_WORDS = new Set(["very", "long", "short", "medium", "shoulder-length", "absurdly", "extremely", "chin-length", "waist-length"]);
const HAIR_TEXTURE_WORDS = new Set([
  "wavy", "curly", "straight", "messy", "wet", "damp", "shiny", "fluffy", "silky", "tousled", "frizzy", "voluminous",
  "thick", "sleek", "glossy", "permed", "disheveled"
]);
const HAIR_STYLE_PHRASES: Record<string, string> = {
  ponytail: "a ponytail",
  "high ponytail": "a high ponytail",
  "low ponytail": "a low ponytail",
  "side ponytail": "a side ponytail",
  "braided ponytail": "a braided ponytail",
  twintails: "twin tails",
  "twin tails": "twin tails",
  "low twintails": "low twin tails",
  pigtails: "pigtails",
  braid: "a braid",
  "single braid": "a braid",
  "twin braids": "twin braids",
  braids: "braids",
  "french braid": "a French braid",
  "hair bun": "a bun",
  bun: "a bun",
  "single hair bun": "a bun",
  "double bun": "double buns",
  "messy bun": "a messy bun",
  "bob cut": "a bob cut",
  "hime cut": "a hime cut",
  "pixie cut": "a pixie cut",
  updo: "an updo"
};
const HAIR_STYLE_SUFFIX_PATTERN = /^(.*?)\s*\b(twintails|twin tails|ponytail|braids?|bun|pigtails|bob cut)$/u;
const HAIR_DETAIL_PHRASES: Record<string, string> = {
  bangs: "bangs",
  "blunt bangs": "blunt bangs",
  "swept bangs": "swept bangs",
  "side-swept bangs": "side-swept bangs",
  "parted bangs": "parted bangs",
  "see-through bangs": "see-through bangs",
  "curtain bangs": "curtain bangs",
  "hair between eyes": "strands falling between {pron} eyes",
  sidelocks: "sidelocks",
  "hair over one eye": "hair covering one eye",
  "hair over shoulder": "hair draped over {pron} shoulder",
  "hair behind ear": "hair tucked behind {pron} ear",
  "hair down": "{pron} hair worn down",
  "hair up": "{pron} hair pinned up",
  ahoge: "a stray strand of hair sticking up"
};
/** Words that make "<prefix> hair" an action ("hand in own hair") instead of a description. */
const HAIR_NON_DESCRIPTIVE_WORDS = /\b(?:in|on|with|own|hand|hands|holding|touching|playing|adjusting|grabbing|pulling|tucking|brushing|fixing)\b/u;

const EYE_DESCRIPTION_PATTERN =
  /^(?:black|brown|dark brown|light brown|blue|light blue|dark blue|green|dark green|grey|gray|hazel|amber|purple|violet|golden|gold|yellow|pink|red|aqua|teal|silver|orange)$|\b(?:bright|large|big|round|almond|narrow|droopy|sharp|tsurime|tareme|sparkling|glowing|doe)\b/u;

const BODY_PATTERN =
  /\b(?:skin|tan|tanned|tanlines|freckles|mole|moles|beauty mark|scars?|tattoos?|makeup|lipstick|eyeshadow|eyeliner|mascara|nail polish|nails|fingernails|eyelashes|eyebrows|lips|teeth|fangs?|dimples|slim|slender|skinny|petite|curvy|plump|chubby|muscular|toned|athletic|tall|short|breasts|flat chest|abs|waist|hips|legs|thighs|figure|build|physique|body|ears|horns?|tail|wings|halo|cheeks|features|face)\b/u;
/** Body descriptions that are singular countable nouns and read better with an article. */
const BODY_ARTICLE_PATTERN = /\b(?:build|figure|physique|face|body|frame|tattoo|scar|mole|beauty mark|tail|halo|waist)$/u;
/**
 * One-word build tags ("slender", "skinny", "tall"). NovelAI lists them as separate tags; in prose they are
 * adjectives of one noun, so they merge into "a slender and tall figure" instead of dangling after the hair.
 * "small" / "tiny" are deliberately absent: "a small figure" reads as a child's proportions to a photo model.
 */
const BODY_ADJECTIVE_PATTERN =
  /^(?:slim|slender|skinny|petite|curvy|plump|chubby|muscular|toned|athletic|tall|thin|fit|willowy|voluptuous|lean|stocky|lanky)$/u;

const PLACE_PATTERN =
  /\b(?:bedroom|living room|kitchen|bathroom|dining room|classroom|library|office|cafe|café|coffee shop|restaurant|bar|izakaya|street|alley|alleyway|park|beach|forest|woods|rooftop|train interior|train|subway|bus interior|car interior|convenience store|store|shop|boutique|shopping mall|mall|hallway|corridor|staircase|elevator|gym|swimming pool|pool|school|campus|hospital|hotel room|hotel|fitting room|dressing room|changing room|studio|photo studio|balcony|garden|city|cityscape|town|village|countryside|field|meadow|mountains?|lake|riverside|seaside|ocean|harbor|pier|bridge|station|train station|platform|airport|apartment|room|house|home|shrine|temple|church|museum|aquarium|amusement park|stage|nightclub|club|karaoke|laundromat|bookstore|bakery|market|parking lot|crosswalk|sidewalk|road|highway|desert|courtyard|terrace|porch|veranda|attic|basement|closet|locker room|infirmary|cafeteria|workplace|lobby|lounge|onsen|hot spring|sauna|bathhouse)$/u;
/** Places one is ON rather than IN. */
const PLACE_ON_PATTERN = /\b(?:street|rooftop|beach|train|subway|balcony|stage|bridge|road|sidewalk|crosswalk|highway|platform|pier|terrace|porch|veranda|campus)$/u;

const TIME_PHRASES: Record<string, string> = {
  night: "at night",
  nighttime: "at night",
  "at night": "at night",
  midnight: "at midnight",
  day: "during the day",
  daytime: "during the day",
  morning: "in the morning",
  afternoon: "in the afternoon",
  evening: "in the evening",
  sunset: "at sunset",
  sunrise: "at sunrise",
  dusk: "at dusk",
  dawn: "at dawn",
  twilight: "at twilight",
  "golden hour": "during golden hour",
  "blue hour": "during the blue hour"
};

const WEATHER_PHRASES: Record<string, string> = {
  rain: "in the rain",
  raining: "in the rain",
  rainy: "on a rainy day",
  "heavy rain": "in heavy rain",
  snow: "in the snow",
  snowing: "in falling snow",
  snowy: "on a snowy day",
  fog: "in the fog",
  foggy: "in the fog",
  mist: "in the mist",
  cloudy: "under a cloudy sky",
  overcast: "under an overcast sky",
  sunny: "on a sunny day",
  "clear sky": "under a clear sky",
  "blue sky": "under a blue sky",
  storm: "in a storm",
  thunderstorm: "in a thunderstorm",
  windy: "on a windy day",
  wind: "on a windy day"
};

const INDOOR_OUTDOOR_PHRASES: Record<string, string> = {
  indoors: "indoors",
  inside: "indoors",
  interior: "indoors",
  outdoors: "outdoors",
  outside: "outdoors",
  exterior: "outdoors"
};

/**
 * Lighting is matched on the tag's HEAD noun, so "light brown hair" and "light smile" fall through to the person
 * lexicons, and "twilight" is caught by the time lexicon first. Sources read "lit by ..."; effects read "with ...".
 */
const LIGHTING_PATTERN =
  /(?:light|lights|lighting|backlighting|backlit|sunlight|moonlight|candlelight|lamplight|daylight|firelight|shadows?|lens flare|sunbeams?|rays|glare|bloom|neon)$/u;
const LIGHTING_EXCLUSION_PATTERN = /\b(?:hair|eyes|smile|blush|skin|highlights)\b/u;
const LIGHT_SOURCE_PATTERN =
  /\b(?:sunlight|moonlight|candlelight|lamplight|lamp light|window light|natural light|daylight|street ?lights?|neon lights?|fluorescent lights?|fairy lights|string lights|firelight|fire light|sunbeams?|sun rays|light rays|god rays|crepuscular rays|ring light|spotlight|stage lights?|city lights|lantern light|screen light)$/u;

const BACKGROUND_PATTERN = /^(?:simple|plain|white|black|grey|gray|gradient|pink|blue|beige|cream|dark|light|solid)(?: \w+)? background$/u;

const PROP_PATTERN =
  /\b(?:bed|pillows?|blanket|sheets|window|windows|curtains?|table|desk|chair|sofa|couch|mirror|door|wall|floor|carpet|rug|bookshelf|shelf|shelves|plants?|potted plant|flowers?|bouquet|vase|cup|mug|coffee|tea|teacup|glass|wine glass|bottle|book|books|notebook|laptop|computer|phone|smartphone|cellphone|tablet|lamp|candle|clock|television|tv|counter|sink|bathtub|stairs|railing|fence|tree|trees|cherry blossoms|leaves|grass|sky|clouds?|stars?|starry sky|night sky|moon|sun|sea|water|waves|river|building|buildings|skyscraper|car|bicycle|bus|umbrella|food|cake|dessert|bread|meal|plate|dish|bowl|fruit|ice cream|drink|letter|envelope|gift|present|box|shopping bag|cat|dog|teddy bear|stuffed toy|poster|painting|picture frame|sign|signboard|vending machine|bench|lockers?|blackboard|whiteboard|piano|guitar|microphone|camera|keyboard|headphones)$/u;

const PRONOUNS: Record<PronounKey, string> = { female: "her", male: "his", other: "their", plural: "their" };

// ---------------------------------------------------------------------------------------------------------------
// Models: what one person / the setting / the camera say, as trimmable phrase items.
// ---------------------------------------------------------------------------------------------------------------

/** Lower ranks are trimmed first when the prompt is over budget. */
const RANK_SETTING_EXTRA = 0;
const RANK_SETTING_PROP = 1;
const RANK_FIGURE_EXTRA = 2;
const RANK_LENS = 3;
const RANK_LIGHTING = 4;
const RANK_FIGURE_DETAIL = 5;
const RANK_TIME = 6;
const RANK_FIGURE_CORE = 8;
const RANK_PLACE = 9;
/** How many leading items of each per-person group are core detail (trimmed last). */
const CORE_ITEMS_PER_GROUP = 2;

interface PhraseItem {
  /** Rendered English; may contain the `{pron}` slot, filled per person at render time. */
  phrase: string;
  source: string;
  rank: number;
  order: number;
  kept: boolean;
}

interface SubjectCounts {
  female: number;
  male: number;
  other: number;
  /** "multiple girls": a group of unknown size. */
  several: Gender[];
  /** "6+girls": at least that many. */
  crowd: Gender[];
}

interface HairParts {
  words: string[];
  styles: string[];
  details: string[];
  sources: string[];
}

interface FigureModel {
  counts: SubjectCounts;
  gender?: Gender;
  modifier?: string;
  /** The creator's appearance as one clause (sentence breaks folded into commas), for multi-person sentences. */
  appearance?: string;
  /** The same text as the creator wrote it, sentence by sentence, for a single-person cut. */
  appearanceSentences?: string[];
  /** Registered identity, filtered out of every tag source when the creator's appearance text replaces it. */
  identityKeys?: Set<string>;
  referenceToken?: string;
  center?: { x: number; y: number };
  hair: HairParts;
  /** Build adjectives, merged into one "a slender and tall figure" feature when the figure is finalized. */
  build: { words: string[]; sources: string[] };
  features: PhraseItem[];
  expression: PhraseItem[];
  pose: PhraseItem[];
  clothing: PhraseItem[];
  bare: PhraseItem[];
  extras: PhraseItem[];
}

interface SettingModel {
  indoorOutdoor: PhraseItem[];
  places: PhraseItem[];
  times: PhraseItem[];
  lighting: PhraseItem[];
  backgrounds: PhraseItem[];
  props: PhraseItem[];
  extras: PhraseItem[];
}

interface CameraModel {
  lens: PhraseItem[];
  /** Composition tags found anywhere in the plan; only used to infer the crop when no frame was given. */
  compositionTags: string[];
}

/** ImageCueFrame, except that an inferred frame may know the viewpoint or angle without knowing the crop. */
interface CameraFrame {
  shot?: ImageCueShot;
  viewpoint: ImageCueViewpoint;
  angle?: ImageCueFrame["angle"];
  visibleRegions: ImageCueBodyRegion[];
}

interface RenderedPrompt {
  body: string;
  subjectSentences: string[];
  settingSentence: string;
  cameraSentence: string;
}

function createItem(phrase: string, source: string, rank: number, order: { next: number }): PhraseItem {
  order.next += 1;
  return { phrase, source, rank, order: order.next, kept: true };
}

function createFigureModel(character: PhotorealCharacterInput | undefined): FigureModel {
  return {
    counts: { female: 0, male: 0, other: 0, several: [], crowd: [] },
    center: character?.center,
    hair: { words: [], styles: [], details: [], sources: [] },
    build: { words: [], sources: [] },
    features: [],
    expression: [],
    pose: [],
    clothing: [],
    bare: [],
    extras: []
  };
}

function createSettingModel(): SettingModel {
  return { indoorOutdoor: [], places: [], times: [], lighting: [], backgrounds: [], props: [], extras: [] };
}

function buildFigureModel(
  character: PhotorealCharacterInput,
  referenceSyntax: PhotorealPromptInput["referenceSyntax"],
  setting: SettingModel,
  camera: CameraModel,
  sink: TagSink,
  order: { next: number }
): FigureModel {
  const figure = createFigureModel(character);
  const tags = cleanTagList(character.tags, sink);
  const identity = cleanTagList(character.identityTags, { dropped: [], negative: [] });
  // Gender and head count come from every source, identity included: the identity tags are often the only place
  // "1girl" is written, and the creator's appearance prose replaces how the person LOOKS, not who they are.
  for (const tag of tags) {
    readSubjectTag(figure, tag.key);
  }
  if (!figure.gender && countTotal(figure.counts) === 0) {
    for (const tag of identity) {
      readSubjectTag(figure, tag.key);
    }
  }

  // The creator's appearance goes in segment by segment: what cannot be sent is dropped and the rest is kept as
  // written. Its first segment names the person ("A Korean woman in her late twenties"); without it the rest is a
  // list of features with no subject, so the whole appearance drops out and the registered identity tags describe
  // the person instead — which is also what a wholly Korean appearance does, as the builder screen tells the creator.
  const appearanceDropped: string[] = [];
  const sanitizedAppearance = sanitizePhotoSentences(character.appearance, appearanceDropped);
  const subjectDropped = appearanceDropped.includes(splitSceneSentences(character.appearance)[0]?.[0] ?? "");
  const appearanceSentences = subjectDropped ? [] : sanitizedAppearance;
  sink.dropped.push(...(subjectDropped ? splitSceneSentences(character.appearance).flat() : appearanceDropped));
  if (appearanceSentences.length > 0) {
    figure.appearanceSentences = appearanceSentences;
    figure.appearance = appearanceSentences
      .map((sentence, index) => (index === 0 ? sentence : sentence.replace(/^(She|He|They|Her|His|Their|The|A|An)\b/u, (word) => word.toLowerCase())))
      .join(", ");
    figure.gender = figure.gender ?? readGenderFromText(appearanceSentences.join(" "));
    figure.identityKeys = new Set(identity.map((tag) => tag.key));
  }

  for (const tag of tags) {
    if (!figure.identityKeys?.has(tag.key)) {
      absorbFigureTag(figure, setting, camera, tag, order);
    }
  }

  const index =
    typeof character.referenceIndex === "number" && Number.isInteger(character.referenceIndex) && character.referenceIndex >= 1
      ? character.referenceIndex
      : undefined;
  if (index !== undefined && referenceSyntax !== "none") {
    figure.referenceToken = referenceSyntax === "qwen21" ? `<image${index}>` : `image ${index}`;
  }
  return figure;
}

function isSubjectTagKey(key: string): boolean {
  return COUNT_TAG_PATTERN.test(key) || MULTIPLE_TAG_PATTERN.test(key) || GENDER_WORD_PATTERN.test(key);
}

/** Subject-count and gender tags decide the noun; they are never rendered as text themselves. */
function readSubjectTag(figure: FigureModel, key: string): boolean {
  const count = key.match(COUNT_TAG_PATTERN);
  if (count) {
    const gender = readGenderWord(count[3]);
    figure.counts[gender] += Number(count[1]);
    if (count[2]) {
      figure.counts.crowd.push(gender);
    }
    return true;
  }
  const multiple = key.match(MULTIPLE_TAG_PATTERN);
  if (multiple) {
    figure.counts.several.push(readGenderWord(multiple[1]));
    return true;
  }
  const word = key.match(GENDER_WORD_PATTERN);
  if (word) {
    figure.gender = figure.gender ?? readGenderWord(word[2]);
    if (word[1]) {
      figure.modifier = figure.modifier ?? word[1];
    }
    return true;
  }
  return false;
}

function readGenderWord(word: string): Gender {
  if (/^(?:girls?|woman|women|female|lady)$/u.test(word)) {
    return "female";
  }
  if (/^(?:boys?|man|men|male|guy|gentleman)$/u.test(word)) {
    return "male";
  }
  return "other";
}

function readGenderFromText(text: string): Gender | undefined {
  const lower = text.toLowerCase();
  if (/\b(?:woman|women|girl|lady|female|she|her)\b/u.test(lower)) {
    return "female";
  }
  if (/\b(?:man|men|boy|guy|male|gentleman|he|his)\b/u.test(lower)) {
    return "male";
  }
  return undefined;
}

function absorbFigureTag(figure: FigureModel, setting: SettingModel, camera: CameraModel, tag: CleanTag, order: { next: number }): void {
  const { key } = tag;
  if (isSubjectTagKey(key) || NO_HUMANS_PATTERN.test(key)) {
    return;
  }
  if (COMPOSITION_PATTERN.test(key)) {
    camera.compositionTags.push(key);
    return;
  }
  if (LENS_PHRASES[key]) {
    camera.lens.push(createItem(LENS_PHRASES[key], tag.source, RANK_LENS, order));
    return;
  }
  const place = readPrepositionalPlace(key);
  if (place) {
    setting.places.push(createItem(place, tag.source, RANK_PLACE, order));
    return;
  }
  if (absorbPersonDetail(figure, tag, order)) {
    return;
  }
  if (HANDHELD_PATTERN.test(key)) {
    figure.extras.push(createItem(`with ${withArticle(insertArticles(tag.text.toLowerCase()))}`, tag.source, RANK_FIGURE_EXTRA, order));
    return;
  }
  if (absorbSettingTag(setting, tag, order)) {
    return;
  }
  figure.extras.push(createItem(tag.text, tag.source, RANK_FIGURE_EXTRA, order));
}

/**
 * Context tags: subject counts feed the shared subject, the setting and camera claim what they can, per-person
 * detail goes to `detailTarget` (the one figure, or the group), and leftovers become background extras.
 */
function absorbContextTag(
  setting: SettingModel,
  camera: CameraModel,
  contextFigure: FigureModel,
  detailTarget: FigureModel,
  tag: CleanTag,
  order: { next: number }
): "no_humans" | "consumed" {
  const { key } = tag;
  if (NO_HUMANS_PATTERN.test(key)) {
    return "no_humans";
  }
  if (readSubjectTag(contextFigure, key)) {
    return "consumed";
  }
  if (COMPOSITION_PATTERN.test(key)) {
    camera.compositionTags.push(key);
    return "consumed";
  }
  if (LENS_PHRASES[key]) {
    camera.lens.push(createItem(LENS_PHRASES[key], tag.source, RANK_LENS, order));
    return "consumed";
  }
  if (
    INDOOR_OUTDOOR_PHRASES[key] ||
    TIME_PHRASES[key] ||
    WEATHER_PHRASES[key] ||
    (LIGHTING_PATTERN.test(key) && !LIGHTING_EXCLUSION_PATTERN.test(key))
  ) {
    absorbSettingTag(setting, tag, order);
    return "consumed";
  }
  const place = readPrepositionalPlace(key);
  if (place) {
    setting.places.push(createItem(place, tag.source, RANK_PLACE, order));
    return "consumed";
  }
  if (absorbPersonDetail(detailTarget, tag, order, { skipBody: true })) {
    return "consumed";
  }
  if (absorbSettingTag(setting, tag, order)) {
    return "consumed";
  }
  if (BODY_PATTERN.test(key)) {
    detailTarget.features.push(createItem(withBodyArticle(tag.text), tag.source, RANK_FIGURE_DETAIL, order));
    return "consumed";
  }
  setting.extras.push(createItem(tag.text, tag.source, RANK_SETTING_EXTRA, order));
  return "consumed";
}

function absorbPersonDetail(figure: FigureModel, tag: CleanTag, order: { next: number }, options: { skipBody?: boolean } = {}): boolean {
  const { key, text, source } = tag;
  const expression = EXPRESSION_PHRASES[key];
  if (expression) {
    figure.expression.push(createItem(expression, source, RANK_FIGURE_DETAIL, order));
    return true;
  }
  if (/^looking at\s+/u.test(key)) {
    figure.expression.push(createItem(insertArticles(key), source, RANK_FIGURE_DETAIL, order));
    return true;
  }
  const pose = readPosePhrase(key);
  if (pose) {
    figure.pose.push(createItem(pose, source, RANK_FIGURE_DETAIL, order));
    return true;
  }
  if (readHairTag(figure.hair, key, source)) {
    return true;
  }
  if (readEyesTag(key)) {
    figure.features.push(createItem(key === "heterochromia" ? "heterochromatic eyes" : key, source, RANK_FIGURE_DETAIL, order));
    return true;
  }
  if (EXPRESSION_NOUN_PATTERN.test(key) && !GARMENT_PATTERN.test(key)) {
    figure.expression.push(createItem(`with ${withArticle(key)}`, source, RANK_FIGURE_DETAIL, order));
    return true;
  }
  const bare = BARE_PHRASES[key];
  if (bare) {
    figure.bare.push(createItem(bare, source, RANK_FIGURE_DETAIL, order));
    return true;
  }
  if (BODY_ADJECTIVE_PATTERN.test(key)) {
    if (!figure.build.words.includes(key)) {
      figure.build.words.push(key);
    }
    figure.build.sources.push(source);
    return true;
  }
  if ((GARMENT_PATTERN.test(key) || isGarmentTag(key)) && !isActionPhraseTag(key)) {
    figure.clothing.push(createItem(readGarmentPhrase(key), source, RANK_FIGURE_DETAIL, order));
    return true;
  }
  if (!options.skipBody && BODY_PATTERN.test(key)) {
    figure.features.push(createItem(withBodyArticle(text), source, RANK_FIGURE_DETAIL, order));
    return true;
  }
  return false;
}

function absorbSettingTag(setting: SettingModel, tag: CleanTag, order: { next: number }): boolean {
  const { key, source } = tag;
  const indoorOutdoor = INDOOR_OUTDOOR_PHRASES[key];
  if (indoorOutdoor) {
    setting.indoorOutdoor.push(createItem(indoorOutdoor, source, RANK_TIME, order));
    return true;
  }
  const time = TIME_PHRASES[key] ?? WEATHER_PHRASES[key];
  if (time) {
    setting.times.push(createItem(time, source, RANK_TIME, order));
    return true;
  }
  if (LIGHTING_PATTERN.test(key) && !LIGHTING_EXCLUSION_PATTERN.test(key)) {
    setting.lighting.push(createItem(key === "backlit" ? "backlighting" : key, source, RANK_LIGHTING, order));
    return true;
  }
  if (BACKGROUND_PATTERN.test(key)) {
    const color = key.replace(/\s*background$/u, "").replace(/^(?:simple|plain|solid)\s*/u, "");
    setting.backgrounds.push(createItem(`against a plain ${color ? `${color} ` : ""}background`, source, RANK_LIGHTING, order));
    return true;
  }
  if (PLACE_PATTERN.test(key)) {
    setting.places.push(createItem(key, source, RANK_PLACE, order));
    return true;
  }
  if (PROP_PATTERN.test(key)) {
    setting.props.push(createItem(insertArticles(key), source, RANK_SETTING_PROP, order));
    return true;
  }
  return false;
}

function readHairTag(hair: HairParts, key: string, source: string): boolean {
  const style = HAIR_STYLE_PHRASES[key];
  if (style) {
    hair.styles.push(style);
    hair.sources.push(source);
    return true;
  }
  const detail = HAIR_DETAIL_PHRASES[key];
  if (detail) {
    hair.details.push(detail);
    hair.sources.push(source);
    return true;
  }
  // "black hair", and the longer authored form "pastel silver hair with light purple tint".
  const described = key.match(/^(.+?)\s+hair(?:\s+with\s+(.+))?$/u);
  if (described && !GARMENT_PATTERN.test(key) && !HAIR_NON_DESCRIPTIVE_WORDS.test(described[1])) {
    addHairWords(hair, described[1]);
    if (described[2]) {
      hair.details.push(described[2]);
    }
    hair.sources.push(source);
    return true;
  }
  // "sky blue twin tails", "pink ponytail": a colour folded into the style tag.
  const styled = key.match(HAIR_STYLE_SUFFIX_PATTERN);
  if (styled && styled[1] && !GARMENT_PATTERN.test(styled[1]) && !HAIR_NON_DESCRIPTIVE_WORDS.test(styled[1])) {
    addHairWords(hair, styled[1]);
    hair.styles.push(HAIR_STYLE_PHRASES[styled[2]] ?? styled[2]);
    hair.sources.push(source);
    return true;
  }
  return false;
}

/** Adds "long platinum blonde wavy" as ordered words, keeping multi-word colours together. */
function addHairWords(hair: HairParts, prefix: string): void {
  let colour: string[] = [];
  const push = (word: string) => {
    if (!hair.words.includes(word)) {
      hair.words.push(word);
    }
  };
  const flush = () => {
    if (colour.length > 0) {
      push(colour.join(" "));
      colour = [];
    }
  };
  for (const word of prefix.split(/\s+/u).filter(Boolean)) {
    if (HAIR_LENGTH_WORDS.has(word) || HAIR_TEXTURE_WORDS.has(word)) {
      flush();
      push(word);
    } else {
      colour.push(word);
    }
  }
  flush();
}

function readEyesTag(key: string): boolean {
  if (key === "heterochromia") {
    return true;
  }
  // "closed eyes" and the other expressions were matched before this; what is left is a description.
  const match = key.match(/^(.+?)\s+eyes$/u);
  return Boolean(match && EYE_DESCRIPTION_PATTERN.test(match[1]));
}

function readPosePhrase(key: string): string | undefined {
  const fixed = POSE_PHRASES[key];
  if (fixed) {
    return fixed;
  }
  const garmentAction = key.match(GARMENT_ACTION_PATTERN);
  if (garmentAction && GARMENT_PATTERN.test(garmentAction[1])) {
    return `${GARMENT_ACTION_VERBS[garmentAction[2]]} {pron} ${garmentAction[1]}`;
  }
  if (POSE_VERB_PATTERN.test(key) || PREPOSITIONAL_PHRASE_PATTERN.test(key)) {
    // "on bed", "against wall": in a person's tags a bare location phrase says where that person IS.
    return insertArticles(key);
  }
  if (POSE_LIMB_PATTERN.test(key)) {
    return `with ${insertArticles(possessivizeLimb(key))}`;
  }
  if (POSE_NOUN_PATTERN.test(key)) {
    return `with ${withArticle(key)}`;
  }
  return undefined;
}

/** "in the classroom", "at cafe": a place tag that already carries its preposition. Returns the bare place. */
function readPrepositionalPlace(key: string): string | undefined {
  const match = key.match(/^(?:in|on|at|inside)\s+(?:(?:the|a|an)\s+)?(.+)$/u);
  return match && PLACE_PATTERN.test(match[1]) && !PROP_PATTERN.test(match[1]) ? match[1] : undefined;
}

/** "hand on own chest" → "a hand on {pron} chest"; "hands together" → "{pron} hands together". */
function possessivizeLimb(phrase: string): string {
  return phrase.replace(/^(hand|arm|leg|finger|knee|foot)\b/u, "a $1").replace(/^(hands|arms|legs|fingers|knees|feet|head)\b/u, "{pron} $1");
}

/**
 * A tag model sometimes writes a whole action as one tag ("trembling hands reaching toward a brown leather
 * handle"). It mentions a material, so the garment lexicon claimed it and the photo prompt read "wearing a trembling
 * hands reaching toward ...". Three or more words with an action verb or a directional preposition is a beat of
 * the scene, not something worn; it stays in the sentence as a plain clause.
 */
const ACTION_PHRASE_PATTERN =
  /\b(?:reaching|holding|dropping|grabbing|pulling|lifting|touching|gripping|clutching|trembling|shaking|falling|carrying|adjusting|tugging|fiddling|fidgeting|waving|pointing|covering|pressing|pushing|squeezing|hugging|clinging|leaning|stretching|picking|putting|taking|opening|closing|hanging|toward|towards|into|onto|across|through)\b/u;

function isActionPhraseTag(key: string): boolean {
  return key.split(/\s+/u).length >= 3 && ACTION_PHRASE_PATTERN.test(key);
}

function readGarmentPhrase(key: string): string {
  if (/^no\s+/u.test(key)) {
    return `without ${key.slice(3)}`;
  }
  return (
    NAI_GARMENT_RENAMES[key] ??
    key
      .replace(/\b(thighhighs|thigh highs|kneehighs)\b/u, (word) => NAI_GARMENT_RENAMES[word] ?? word)
      .replace(/\bown\s+/u, "{pron} ")
  );
}

/** Merges the hair words into one phrase and promotes the leading items of each group to core detail. */
function finalizeFigure(figure: FigureModel, order: { next: number }): void {
  if (figure.build.words.length > 0) {
    figure.features.unshift(
      createItem(withArticle(`${naturalJoin(figure.build.words)} figure`), figure.build.sources.join(", "), RANK_FIGURE_DETAIL, order)
    );
  }
  if (figure.hair.sources.length > 0) {
    // "long hair" + "black hair" + "ponytail" → "long black hair in a ponytail": one phrase, first among the
    // features, because hair is what identifies a person at a glance and NovelAI tags spread it over three tags.
    figure.features.unshift(createItem(renderHair(figure.hair), figure.hair.sources.join(", "), RANK_FIGURE_CORE, order));
  }
  for (const group of [figure.features, figure.expression, figure.pose, figure.clothing]) {
    group.forEach((item, index) => {
      if (index < CORE_ITEMS_PER_GROUP) {
        item.rank = Math.max(item.rank, RANK_FIGURE_CORE);
      }
    });
  }
}

function renderHair(hair: HairParts): string {
  const lengths = hair.words.filter((word) => HAIR_LENGTH_WORDS.has(word));
  const textures = hair.words.filter((word) => HAIR_TEXTURE_WORDS.has(word));
  const colours = hair.words.filter((word) => !HAIR_LENGTH_WORDS.has(word) && !HAIR_TEXTURE_WORDS.has(word));
  const described = [...lengths, ...textures, ...colours].join(" ");
  const styles = uniqueStrings(hair.styles);
  const details = uniqueStrings(hair.details);
  let phrase = described ? `${described} hair` : styles.length > 0 ? "hair" : "";
  if (styles.length > 0) {
    phrase = `${phrase} in ${naturalJoin(styles)}`;
  }
  if (details.length > 0) {
    phrase = phrase ? `${phrase} with ${naturalJoin(details)}` : naturalJoin(details);
  }
  return phrase.trim();
}

function figureHasDetail(figure: FigureModel): boolean {
  return figureItems(figure).length > 0;
}

function figureItems(figure: FigureModel): PhraseItem[] {
  return [...figure.features, ...figure.expression, ...figure.pose, ...figure.clothing, ...figure.bare, ...figure.extras];
}

function countTotal(counts: SubjectCounts): number {
  return counts.female + counts.male + counts.other + counts.several.length;
}

// ---------------------------------------------------------------------------------------------------------------
// Rendering.
// ---------------------------------------------------------------------------------------------------------------

const ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth"];

/**
 * People are laid out where NovelAI would have put them (center.x, 0 = left). When the plan gives no usable
 * positions — or two people share one — the cue's own order is kept and the sentences use ordinals, rather than
 * inventing a left/right the plan never decided.
 */
function orderFiguresForLayout(figures: FigureModel[]): { figures: FigureModel[]; positional: boolean } {
  if (figures.length < 2) {
    return { figures, positional: false };
  }
  const xs = figures.map((figure) => figure.center?.x);
  const usable = xs.every((x) => typeof x === "number" && Number.isFinite(x));
  const distinct = usable && new Set(xs.map((x) => Math.round((x ?? 0) * 100))).size === figures.length;
  if (!distinct) {
    return { figures, positional: false };
  }
  return { figures: [...figures].sort((a, b) => (a.center?.x ?? 0) - (b.center?.x ?? 0)), positional: true };
}

function positionPrefix(index: number, total: number): string {
  if (total === 2) {
    return index === 0 ? "On the left" : "on the right";
  }
  if (total === 3) {
    return ["On the left", "in the middle", "on the right"][index];
  }
  const ordinal = (ORDINALS[index] ?? `number ${index + 1}`).toLowerCase();
  return index === 0 ? `${capitalizeFirst(ordinal)} from the left` : `${ordinal} from the left`;
}

function renderSubjectSentences(
  layout: { figures: FigureModel[]; positional: boolean },
  contextSubject: FigureModel | undefined,
  groupFigure: FigureModel | undefined
): string[] {
  const { figures, positional } = layout;
  const sentences: string[] = [];
  if (figures.length === 1) {
    sentences.push(renderStandaloneFigure(figures[0]));
  } else if (figures.length > 1 && positional) {
    // "On the left, ...; on the right, ...": one sentence, one clause per person, each clause its own element.
    figures.forEach((figure, index) => {
      const clause = `${positionPrefix(index, figures.length)}, ${lowercaseLeadingArticle(renderFigureClause(figure))}`;
      sentences.push(`${clause}${index < figures.length - 1 ? ";" : "."}`);
    });
  } else if (figures.length > 1) {
    figures.forEach((figure, index) => {
      sentences.push(`${ORDINALS[index] ?? `Number ${index + 1}`}, ${lowercaseLeadingArticle(renderFigureClause(figure))}.`);
    });
  } else if (contextSubject) {
    sentences.push(`${capitalizeFirst(renderFigureClause(contextSubject))}.`);
  }
  if (groupFigure && figures.length > 1) {
    const who = figures.length === 2 ? "Both of them" : "All of them";
    const features = keptPhrases(groupFigure.features).map((phrase) => fillPronouns(phrase, "plural"));
    const groups = renderFigureGroups(groupFigure, "plural", true);
    if (features.length > 0 && groups.length > 0) {
      sentences.push(`${who} have ${naturalJoin(features)} and are ${groups.join(", ")}.`);
    } else if (groups.length > 0) {
      sentences.push(`${who} are ${groups.join(", ")}.`);
    } else if (features.length > 0) {
      sentences.push(`${who} have ${naturalJoin(features)}.`);
    }
  }
  return sentences;
}

/**
 * A cut with one person keeps a multi-sentence appearance as the creator wrote it and continues in new sentences
 * ("She is smiling, ..."); folding three sentences of prose into one comma chain would bury the cut's own detail.
 */
function renderStandaloneFigure(figure: FigureModel): string {
  const sentences = figure.appearanceSentences ?? [];
  if (sentences.length < 2) {
    return `${capitalizeFirst(renderFigureClause(figure))}.`;
  }
  const pronoun = dominantGender(figure);
  const [subject, be, have] = pronoun === "female" ? ["She", "is", "has"] : pronoun === "male" ? ["He", "is", "has"] : ["They", "are", "have"];
  const first = figure.referenceToken ? `${renderReferencePhrase(figure)}, ${lowercaseLeadingArticle(sentences[0])}` : sentences[0];
  const parts = [first, ...sentences.slice(1)].map((sentence) => `${capitalizeFirst(sentence)}.`);
  const features = keptPhrases(figure.features);
  const groups = renderFigureGroups(figure, pronoun, false);
  if (features.length > 0) {
    parts.push(`${subject} ${have} ${fillPronouns(naturalJoin(features), pronoun)}.`);
  }
  if (groups.length > 0) {
    parts.push(`${subject} ${be} ${groups.join(", ")}.`);
  }
  return parts.join(" ");
}

function renderFigureClause(figure: FigureModel): string {
  const plural = isPluralSubject(figure);
  const pronoun: PronounKey = plural ? "plural" : dominantGender(figure);
  const features = keptPhrases(figure.features);
  const head = `${renderSubjectPhrase(figure)}${features.length > 0 ? ` with ${naturalJoin(features)}` : ""}`;
  return fillPronouns([head, ...renderFigureGroups(figure, pronoun, plural)].join(", "), pronoun);
}

function renderFigureGroups(figure: FigureModel, pronoun: PronounKey, plural: boolean): string[] {
  const clothing = keptPhrases(figure.clothing);
  const wearing = clothing.filter((phrase) => !/^without\s/u.test(phrase)).map((phrase) => (plural ? pluralizeLastWord(phrase) : withArticle(phrase)));
  const without = clothing.filter((phrase) => /^without\s/u.test(phrase));
  return [
    naturalJoin(dropPrefixDuplicates(keptPhrases(figure.expression))),
    naturalJoin(dropPrefixDuplicates(keptPhrases(figure.pose))),
    wearing.length > 0 ? `wearing ${naturalJoin(wearing)}` : "",
    naturalJoin([...without, ...keptPhrases(figure.bare)]),
    keptPhrases(figure.extras).join(", ")
  ]
    .filter(Boolean)
    .map((group) => fillPronouns(group, pronoun));
}

/** "the woman from <image1>": the reference names the person, so no other noun is needed. */
function renderReferencePhrase(figure: FigureModel): string {
  const gender = dominantGender(figure);
  const noun = gender === "female" ? "woman" : gender === "male" ? "man" : "person";
  return `the ${noun} from ${figure.referenceToken ?? ""}`;
}

function renderSubjectPhrase(figure: FigureModel): string {
  if (figure.referenceToken) {
    return `${renderReferencePhrase(figure)}${figure.appearance ? `, ${lowercaseLeadingArticle(figure.appearance)}` : ""}`;
  }
  if (figure.appearance) {
    return figure.appearance;
  }
  return describeSubjectCounts(figure.counts, figure.modifier) ?? singularAdultNoun(figure.gender ?? "other", figure.modifier);
}

function describeSubjectCounts(counts: SubjectCounts, modifier?: string): string | undefined {
  const parts: string[] = [];
  for (const gender of ["female", "male", "other"] as const) {
    const total = counts[gender];
    if (total === 1) {
      parts.push(singularAdultNoun(gender, modifier));
    } else if (total > 1) {
      parts.push(`${counts.crowd.includes(gender) ? "a group of" : numberWord(total)} adult ${pluralNoun(gender)}`);
    } else if (counts.several.includes(gender)) {
      parts.push(`several adult ${pluralNoun(gender)}`);
    }
  }
  return parts.length > 0 ? naturalJoin(parts) : undefined;
}

/** Adults by default (safety floor §2.4): "1girl" is "an adult woman", never "a girl". */
function singularAdultNoun(gender: Gender, modifier?: string): string {
  const noun = gender === "female" ? "woman" : gender === "male" ? "man" : "person";
  switch (modifier) {
    case "mature":
      return `a mature ${noun}`;
    case "young":
      return `a young adult ${noun}`;
    case "old":
    case "elderly":
      return `an elderly ${noun}`;
    case "middle-aged":
      return `a middle-aged ${noun}`;
    default:
      return `an adult ${noun}`;
  }
}

function pluralNoun(gender: Gender): string {
  return gender === "female" ? "women" : gender === "male" ? "men" : "people";
}

function numberWord(value: number): string {
  return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"][value] ?? String(value);
}

function isPluralSubject(figure: FigureModel): boolean {
  if (figure.referenceToken || figure.appearance) {
    return false;
  }
  return figure.counts.female + figure.counts.male + figure.counts.other > 1 || figure.counts.several.length > 0;
}

function dominantGender(figure: FigureModel): Gender {
  if (figure.counts.female > 0 && figure.counts.male === 0) {
    return "female";
  }
  if (figure.counts.male > 0 && figure.counts.female === 0) {
    return "male";
  }
  return figure.gender ?? (figure.appearance ? readGenderFromText(figure.appearance) : undefined) ?? "other";
}

function fillPronouns(text: string, pronoun: PronounKey): string {
  return text.replace(/\{pron\}/gu, PRONOUNS[pronoun]);
}

function renderSettingSentence(setting: SettingModel, hasHumans: boolean, declareNoPeople: boolean): string {
  const places = keptPhrases(setting.places);
  const indoorOutdoor = keptPhrases(setting.indoorOutdoor).filter((phrase) => !(places.length > 0 && phrase === "indoors"));
  const times = keptPhrases(setting.times);
  const described = [...keptPhrases(setting.props).map(withArticle), ...keptPhrases(setting.extras)];
  // "by the window" is already a clause of its own; listing it after "with" would read "with by the window".
  const things = described.filter((phrase) => !PREPOSITIONAL_PHRASE_PATTERN.test(phrase));
  const positioned = described.filter((phrase) => PREPOSITIONAL_PHRASE_PATTERN.test(phrase));
  const lighting = keptPhrases(setting.lighting);
  const sources = lighting.filter((phrase) => LIGHT_SOURCE_PATTERN.test(phrase));
  const effects = lighting.filter((phrase) => !LIGHT_SOURCE_PATTERN.test(phrase));
  const placePhrase = naturalJoin([...places.map(renderPlace), ...indoorOutdoor]);
  const where = [placePhrase, ...times].filter(Boolean).join(" ");
  const tailClauses = [
    ...positioned,
    ...keptPhrases(setting.backgrounds),
    sources.length > 0 ? `lit by ${naturalJoin(sources.map((source) => (/\b(?:ring light|spotlight)$/u.test(source) ? withArticle(source) : source)))}` : "",
    effects.length > 0 ? `with ${naturalJoin(effects)}` : ""
  ];

  let clauses: string[];
  if (hasHumans) {
    clauses = [
      where ? `set ${where}` : things.length > 0 ? `the scene includes ${naturalJoin(things)}` : "",
      where && things.length > 0 ? `with ${naturalJoin(things)}` : "",
      ...tailClauses
    ];
  } else {
    // Without people the objects and the place ARE the subject, so they lead the sentence.
    const head =
      things.length > 0
        ? `${naturalJoin(things)}${placePhrase ? ` ${placePhrase}` : ""}`
        : places.length > 0
          ? `a view of ${naturalJoin(places.map(withArticle))}${indoorOutdoor.length > 0 ? ` ${indoorOutdoor.join(" ")}` : ""}`
          : indoorOutdoor.join(" ");
    clauses = [[head, ...times].filter(Boolean).join(" "), ...tailClauses];
  }
  const sentence = clauses.filter(Boolean).join(", ");
  return joinPromptParts([sentence ? `${capitalizeFirst(sentence)}.` : "", declareNoPeople ? "No people are present." : ""]);
}

function renderPlace(place: string): string {
  return `${PLACE_ON_PATTERN.test(place) ? "on" : "in"} ${withArticle(place)}`;
}

function inferFrameFromTags(tags: string[]): CameraFrame | undefined {
  if (tags.length === 0) {
    return undefined;
  }
  const inferred = inferImageCueFrameFromTags(tags);
  const haystack = tags.join(", ");
  const viewpoint = readViewpoint(haystack);
  const angle: CameraFrame["angle"] = /\b(?:from above|high angle|bird'?s-eye view|aerial view)\b/u.test(haystack)
    ? "from_above"
    : /\b(?:from below|low angle)\b/u.test(haystack)
      ? "from_below"
      : /\bdutch angle\b/u.test(haystack)
        ? "dutch_angle"
        : undefined;
  if (inferred) {
    return { ...inferred, viewpoint: viewpoint ?? inferred.viewpoint, angle };
  }
  // No crop tag but a viewpoint or an angle: describe what is known and leave the crop to the model.
  return viewpoint || angle ? { viewpoint: viewpoint ?? "front", angle, visibleRegions: [] } : undefined;
}

function readViewpoint(haystack: string): ImageCueViewpoint | undefined {
  if (/\b(?:from behind|rear view|back view)\b/u.test(haystack)) {
    return "from_behind";
  }
  if (/\bover[ -]the[ -]shoulder\b/u.test(haystack)) {
    return "over_the_shoulder";
  }
  if (/\bpov\b/u.test(haystack)) {
    return "pov";
  }
  if (/\b(?:from side|side view|profile|profile view)\b/u.test(haystack)) {
    return "side";
  }
  return undefined;
}

const SHOT_PHRASES: Record<ImageCueShot, string> = {
  close_up: "Tight close-up of the face",
  face_focus: "Head-and-shoulders portrait",
  upper_body: "Medium close-up from the waist up",
  cowboy_shot: "Medium shot from mid-thigh up",
  full_body: "Full-length shot showing the whole body from head to toe",
  wide_shot: "Wide shot with the subject small in the environment"
};

/** The same crops with nobody in them: there is no face to close in on and no subject to keep small. */
const SCENE_SHOT_PHRASES: Record<ImageCueShot, string> = {
  close_up: "Close-up shot",
  face_focus: "Close-up shot",
  upper_body: "Medium shot",
  cowboy_shot: "Medium shot",
  full_body: "Medium-wide shot",
  wide_shot: "Wide establishing shot"
};

const VIEWPOINT_PHRASES: Partial<Record<ImageCueViewpoint, string>> = {
  side: "in profile",
  from_behind: "seen from behind",
  pov: "first-person point of view",
  over_the_shoulder: "over-the-shoulder shot"
};

const ANGLE_PHRASES: Record<NonNullable<ImageCueFrame["angle"]>, string> = {
  eye_level: "eye-level",
  from_above: "high angle",
  from_below: "low angle",
  dutch_angle: "tilted dutch angle"
};

const REGION_NAMES: Record<ImageCueBodyRegion, string> = {
  head: "face",
  torso: "upper body",
  hips: "hips",
  legs: "legs",
  feet: "feet"
};

function describeCamera(frame: CameraFrame | undefined, hasHumans: boolean): string[] {
  if (!frame) {
    return [];
  }
  const phrases: string[] = [];
  if (frame.shot && SHOT_PHRASES[frame.shot]) {
    const regions = Array.isArray(frame.visibleRegions) ? frame.visibleRegions.filter((region) => REGION_NAMES[region]) : [];
    if (hasHumans && frame.shot === "close_up" && regions.length > 0 && !regions.includes("head")) {
      // A body-region close-up (hands around a cup, a held wrist) is still a close-up — just not of the face.
      phrases.push(`Tight close-up of the ${naturalJoin(regions.map((region) => REGION_NAMES[region]))}`);
    } else {
      phrases.push(hasHumans ? SHOT_PHRASES[frame.shot] : SCENE_SHOT_PHRASES[frame.shot]);
    }
  }
  // The default look is the user's daily-look references, where the face is routinely cropped off or hidden behind a
  // phone. A story cut that shows the head must show the face, so it is said explicitly (2_4 in QA lost the face).
  const headInFrame = Array.isArray(frame.visibleRegions) && frame.visibleRegions.includes("head");
  if (hasHumans && headInFrame && frame.viewpoint !== "from_behind" && frame.shot) {
    if (frame.shot === "close_up") {
      phrases.push("the face filling most of the frame");
    } else if (frame.shot === "face_focus" || frame.shot === "upper_body" || frame.shot === "cowboy_shot") {
      phrases.push("with the face fully visible");
    }
  }
  const viewpoint = VIEWPOINT_PHRASES[frame.viewpoint];
  if (viewpoint && (hasHumans || frame.viewpoint === "pov")) {
    phrases.push(viewpoint);
  }
  if (frame.angle && ANGLE_PHRASES[frame.angle]) {
    phrases.push(ANGLE_PHRASES[frame.angle]);
  }
  return phrases;
}

function pickTrimVictim(items: PhraseItem[]): PhraseItem | undefined {
  let victim: PhraseItem | undefined;
  for (const item of items) {
    if (item.kept && (!victim || item.rank < victim.rank || (item.rank === victim.rank && item.order > victim.order))) {
      victim = item;
    }
  }
  return victim;
}

/**
 * Last resort when even the untrimmable parts (subject nouns, a long creator appearance, a long custom style)
 * exceed the budget: cut at a word boundary. The reference sentence is appended after this and never cut.
 */
function clampToLength(text: string, budget: number): string {
  if (text.length <= budget) {
    return text;
  }
  const slice = text.slice(0, Math.max(0, budget - 1));
  const lastSpace = slice.lastIndexOf(" ");
  const cut = (lastSpace > budget * 0.5 ? slice.slice(0, lastSpace) : slice).replace(/[\s,;:.-]+$/u, "");
  return cut ? `${cut}.` : "";
}

// ---------------------------------------------------------------------------------------------------------------
// Small English helpers.
// ---------------------------------------------------------------------------------------------------------------

const DETERMINER_PATTERN =
  /^(?:(?:a|an|the|her|his|their|its|my|your|our|one|two|three|four|five|some|several|many|both|each|every|another|no|all|pair of|without|with)\b|\{pron\}|\d)/u;
const MASS_NOUNS = new Set([
  "knitwear", "lingerie", "underwear", "clothing", "denim", "leather", "lace", "makeup", "jewelry", "jewellery",
  "sportswear", "swimwear", "nightwear", "outerwear", "loungewear", "activewear", "attire", "armor", "armour",
  "water", "rain", "snow", "sand", "grass", "mud", "tea", "milk", "light", "sunlight", "moonlight", "daylight",
  "lighting", "darkness", "fog", "mist", "smoke", "steam", "nature", "scenery", "foliage", "greenery", "furniture",
  "food", "hair", "indoors", "outdoors", "skin", "barefoot"
]);
const BODY_PART_WORDS =
  /^(?:back|stomach|side|mouth|lips|hip|hips|chest|breast|breasts|cheek|cheeks|face|head|shoulder|shoulders|lap|knee|knees|neck|ear|ears|arm|arms|hand|hands|leg|legs|thigh|thighs|waist|belly|forehead|chin|nose|eye|eyes|foot|feet|hair|pocket|pockets)$/u;
/** Scene fixtures that read better with "the" ("sitting on the bed", "by the window"). */
const DEFINITE_WORDS =
  /^(?:floor|ground|sky|sea|ocean|horizon|sun|moon|window|door|wall|bed|couch|sofa|table|desk|counter|stairs|street|beach|grass|water|rain|snow|mirror|camera|viewer)$/u;
const ARTICLE_TRIGGER_PATTERN =
  /\b(holding|carrying|drinking|eating|reading|writing|using|hugging|on|in|at|against|by|near|under|behind|beside|inside|onto|into|over|across|toward|towards|to)\s+([a-z][\w'-]*)/gu;
const ARTICLE_SKIP_WORDS =
  /^(?:a|an|the|her|his|their|its|my|your|our|one|two|three|some|both|each|another|other|all|own|top|front|place|line|time|hand|hands|tiptoes|fours|air|public|focus|frame|view|profile|love|shock|surprise|tears|pain|thought|bed rest)$/u;

/** NAI tags drop articles ("sitting on bed", "holding cup"); prose needs them back. Body parts become possessive. */
function insertArticles(phrase: string): string {
  const owned = phrase.replace(/\bown\s+/gu, "{pron} ");
  return owned.replace(ARTICLE_TRIGGER_PATTERN, (match, trigger: string, word: string) => {
    if (ARTICLE_SKIP_WORDS.test(word)) {
      return match;
    }
    if (BODY_PART_WORDS.test(word)) {
      return `${trigger} {pron} ${word}`;
    }
    if (DEFINITE_WORDS.test(word)) {
      return `${trigger} the ${word}`;
    }
    if (isPluralWord(word) || MASS_NOUNS.has(word)) {
      return match;
    }
    return `${trigger} ${indefiniteArticle(word)} ${word}`;
  });
}

function withArticle(phrase: string): string {
  const text = phrase.trim();
  const lower = text.toLowerCase();
  if (!text || DETERMINER_PATTERN.test(lower) || PREPOSITIONAL_PHRASE_PATTERN.test(lower)) {
    return text;
  }
  const lastWord = lower.split(/\s+/u).pop() ?? "";
  if (isPluralWord(lastWord) || MASS_NOUNS.has(lastWord)) {
    return text;
  }
  return `${indefiniteArticle(lower)} ${text}`;
}

function withBodyArticle(text: string): string {
  const lower = text.toLowerCase();
  return BODY_ARTICLE_PATTERN.test(lower) ? withArticle(lower) : lower;
}

function indefiniteArticle(word: string): string {
  return /^[aeiou]/u.test(word) && !/^(?:uni|use|usu|eu|one|once|ewe)/u.test(word) ? "an" : "a";
}

function isPluralWord(word: string): boolean {
  return /s$/u.test(word) && !/(?:ss|us|is|ous)$/u.test(word);
}

function pluralizeLastWord(phrase: string): string {
  const words = phrase.split(" ");
  const last = words.pop() ?? "";
  if (!last || isPluralWord(last) || MASS_NOUNS.has(last)) {
    return phrase;
  }
  const plural = /(?:s|sh|ch|x|z)$/u.test(last) ? `${last}es` : /[^aeiou]y$/u.test(last) ? `${last.slice(0, -1)}ies` : `${last}s`;
  return [...words, plural].join(" ");
}

function naturalJoin(parts: string[]): string {
  const items = parts.filter(Boolean);
  if (items.length <= 1) {
    return items[0] ?? "";
  }
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

function keptPhrases(items: PhraseItem[]): string[] {
  return uniqueStrings(items.filter((item) => item.kept).map((item) => item.phrase));
}

/** "smiling" next to "smiling broadly", "sitting" next to "sitting on the bed": the longer phrase says both. */
function dropPrefixDuplicates(phrases: string[]): string[] {
  return phrases.filter((phrase) => !phrases.some((other) => other !== phrase && other.startsWith(`${phrase} `)));
}

function joinPromptParts(parts: string[]): string {
  return collapseWhitespace(parts.filter(Boolean).join(" "));
}

function collapseWhitespace(text: string): string {
  return text.replace(/\s+/gu, " ").trim();
}

function capitalizeFirst(text: string): string {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

/** "An adult woman" → "an adult woman" after "On the left,"; a proper noun ("Seoyeon, ...") keeps its capital. */
function lowercaseLeadingArticle(text: string): string {
  return text.replace(/^(A|An|The)\b/u, (word) => word.toLowerCase());
}

function stripTerminalPunctuation(text: string | undefined): string {
  return (text ?? "").trim().replace(/[\s.,;:!?。]+$/u, "");
}

function splitLeadSentence(text: string): { lead: string; tail: string } {
  const value = collapseWhitespace(text);
  const match = value.match(/^(.+?[.!?])\s+(.+)$/u);
  if (!match) {
    return { lead: stripTerminalPunctuation(value), tail: "" };
  }
  return { lead: stripTerminalPunctuation(match[1]), tail: stripTerminalPunctuation(match[2]) };
}

/**
 * A creator's free text (an appearance, a style) as a photo prompt may carry it, sentence by sentence. Each comma
 * segment is kept or dropped whole, and the dropped ones are reported in `dropped`:
 * - text in any script but Latin. Korean is the common case and is dropped by contract (the settings and builder
 *   screens say so); any other script has already locked the cut upstream (selectPhotoTextForScreening) and is
 *   dropped here regardless, so it can never reach an encoder that reads it;
 * - a body-identity trait or an eating-disorder build, as for tags;
 * - a minor- or student-coded segment ("lolita, flat-chested, childish body", "a high school student"), as for tags.
 *   Prose had no scrub of its own and rendered those verbatim from an appearance.
 * A sheer qualifier loses the word and the garment stays, opaque, as for tags. Pictographs go from every segment.
 */
function sanitizePhotoSentences(text: string | undefined, dropped: string[]): string[] {
  const sentences: string[] = [];
  for (const segments of splitSceneSentences(text)) {
    const kept: string[] = [];
    for (const written of segments) {
      const segment = stripPictographs(written);
      const lower = segment.toLowerCase();
      const opaque = collapseWhitespace(stripSheerQualifiers(segment));
      if (
        hasUnverifiableScript(segment) ||
        isMinorOrStudentCodedSceneText(segment) ||
        IDENTITY_ANATOMY_TRAIT_PATTERN.test(lower) ||
        EATING_DISORDER_BODY_PATTERN.test(lower) ||
        // Nothing left once a bare "sheer" lost its word, or once the pictographs went.
        !/[\p{L}\p{N}]/u.test(opaque)
      ) {
        dropped.push(written);
        continue;
      }
      kept.push(opaque);
    }
    if (kept.length > 0) {
      sentences.push(kept.join(", "));
    }
  }
  return sentences;
}

/** sanitizePhotoSentences as one line, for the style's lead, tail and ambient defaults. */
function sanitizePhotoText(text: string | undefined, dropped: string[] = []): string {
  return sanitizePhotoSentences(text, dropped).join(". ");
}

function stripSheerQualifiers(text: string): string {
  return text.replace(SHEER_QUALIFIER_IN_TEXT_PATTERN, (_match, article: string | undefined, next: string) => {
    if (!article) {
      return "";
    }
    const fitting = indefiniteArticle(next.toLowerCase());
    return `${article.charAt(0) === article.charAt(0).toUpperCase() ? capitalizeFirst(fitting) : fitting} `;
  });
}

/**
 * A photo negative as it may reach the sampler: mergeNegativePrompts over one finished string. For callers that
 * hold a negative planned earlier (a queued job, a custom workflow's) and must not trust that it was cleaned then.
 */
export function sanitizePhotorealNegativePrompt(text: string | undefined): string {
  return mergeNegativePrompts(text);
}

// ---------------------------------------------------------------------------------------------------------------
// Photo negatives: an allow-list
// ---------------------------------------------------------------------------------------------------------------
//
// A negative prompt is the one piece of photo text the safety floor does not read as a picture, and at any CFG above
// 1 it steers the sampler away from what it names: "clothing", "denim", "turtleneck", "Kleidung" or "cl0thing" toward
// an undressed figure, "adulthood", "in her twenties", "25yo" or "breasts" toward a younger one. (Below 1 a negative
// would become the picture outright; comfyWorkflows never samples there.) A deny-list of such words leaked 70 of 75 a
// red team tried — every garment, every language, every spelling — so the photo negative is an allow-list instead:
// only image quality, artefacts, the wrong medium or look, and a few worn extras and skin marks whose absence leaves
// the same clothed adult (the settings screen offers "glasses, heavy makeup" as its example). Anything else a creator
// or a cut's negative weight puts there is dropped. Matching is on a normalised key (case, hyphens and underscores,
// spacing and plurals ignored), so "Low-Res", "lowres" and "low res" are one entry.

/** Standalone quality, artefact and medium terms. */
const NEGATIVE_QUALITY_TERMS = [
  // Resolution, focus, noise and compression.
  "blurry", "blur", "blurred", "motion blur", "out of focus", "unfocused", "soft focus", "lowres", "low res", "low resolution",
  "low quality", "worst quality", "bad quality", "poor quality", "normal quality", "average quality", "low detail",
  "lack of detail", "jpeg", "jpeg artifacts", "compression artifacts", "artifacts", "noise", "noisy", "grain", "grainy",
  "heavy grain", "film grain", "chromatic aberration", "banding", "color banding", "posterization", "moire", "halo",
  "haloing", "ghosting", "double exposure", "lens flare", "glare", "error", "glitch", "glitched", "corrupted",
  // Exposure, colour and processing.
  "overexposed", "underexposed", "overexposure", "underexposure", "oversaturated", "undersaturated", "desaturated",
  "washed out", "faded", "dull", "dull colors", "flat colors", "color cast", "harsh flash", "flash glare", "red eye",
  "hdr", "heavy hdr", "oversharpened", "over sharpened", "digital sharpening", "overprocessed", "over processed",
  "over edited", "over retouched", "retouched", "photoshopped", "beauty filter", "filter", "instagram filter",
  "airbrushed", "airbrushed skin", "plastic skin", "waxy skin", "smooth skin", "overly smooth skin", "flat lighting",
  "bad lighting", "harsh lighting", "harsh shadows", "monochrome", "black and white", "grayscale", "sepia",
  // Framing and overlays.
  "cropped", "crop", "out of frame", "cut off", "frame", "border", "letterbox", "letterboxed", "vignette", "tilted",
  "crooked horizon", "fisheye", "perspective distortion", "bad composition", "cluttered background", "busy background",
  "messy background", "distracting background", "simple background", "plain background", "watermark", "text", "logo",
  "signature", "username", "artist name", "caption", "subtitles", "words", "letters", "timestamp", "date stamp", "ui",
  // Wrong medium or look.
  "anime", "anime style", "cartoon", "illustration", "illustrated", "painting", "oil painting", "watercolor", "drawing",
  "sketch", "line art", "lineart", "comic", "manga", "cel shading", "2d", "3d", "3d render", "render", "cgi",
  "digital art", "concept art", "video game", "unreal engine", "octane render", "semi realistic", "stylized", "painterly",
  "doll", "doll like", "mannequin", "figurine", "toy", "plastic", "wax figure", "artificial", "fake", "unrealistic",
  "uncanny", "uncanny valley", "amateur", "bad art", "poorly drawn", "ugly", "gross", "morbid", "mutilated",
  "deformed", "disfigured", "distorted", "distortion", "malformed", "mutated", "mutation", "duplicate", "duplicated",
  "cloned", "clone", "warped", "stretched", "squashed", "bad anatomy", "anatomical errors", "bad proportions",
  "gross proportions", "long neck", "elongated", "too many fingers", "extra digit", "fewer digits",
  // Worn extras and skin marks: leaving them out leaves the same clothed adult.
  "glasses", "eyeglasses", "sunglasses", "spectacles", "hat", "cap", "beanie", "headwear", "tattoo", "piercing",
  "freckles", "mole", "beauty mark", "scar", "blemish", "acne", "pimple", "heavy makeup", "thick makeup",
  "excessive makeup", "too much makeup", "heavy eyeliner", "heavy contouring"
];
/** "extra fingers", "deformed hands", "bad face": a defect word with the body part it names. Never the part alone. */
const NEGATIVE_DEFECT_QUALIFIERS = [
  "bad", "poor", "poorly drawn", "ugly", "deformed", "malformed", "mutated", "mutation", "distorted", "disfigured",
  "fused", "extra", "missing", "too many", "fewer", "duplicate", "duplicated", "cloned", "broken", "twisted",
  "asymmetric", "asymmetrical", "uneven", "crossed", "misaligned", "floating", "disconnected", "wrong", "incorrect",
  "unnatural", "weird", "strange", "gross", "blurry", "melted"
];
const NEGATIVE_DEFECT_PARTS = [
  "anatomy", "proportion", "body", "body part", "hand", "finger", "thumb", "digit", "limb", "arm", "leg", "foot", "feet",
  "toe", "face", "facial feature", "feature", "eye", "pupil", "iris", "teeth", "tooth", "mouth", "lip", "ear", "nose",
  "neck", "head", "joint", "knuckle", "nail", "fingernail", "wrist", "elbow", "knee", "eyebrow", "eyelash"
];

/** A negative term in the form the vocabulary is keyed by: lowercase, no hyphens, underscores or spaces, singular. */
function negativeVocabularyKey(term: string): string {
  return term
    .toLowerCase()
    .replace(/artefact/gu, "artifact")
    .replace(/colour/gu, "color")
    .replace(/greyscale/gu, "grayscale")
    .replace(/[_\p{Pd}]/gu, " ")
    .replace(/[^\p{L}\p{N}\s]/gu, "")
    .split(/\s+/u)
    .filter(Boolean)
    .map(singularWord)
    .join("");
}

/** "fingers" → "finger", "artifacts" → "artifact", "lashes" → "lash"; short words ("ugly", "hdr", "lens") are kept. */
function singularWord(word: string): string {
  if (word.length <= 3) {
    return word;
  }
  if (/ies$/u.test(word)) {
    return `${word.slice(0, -3)}y`;
  }
  if (/(?:sh|ch|x|ss)es$/u.test(word)) {
    return word.slice(0, -2);
  }
  return /[^s]s$/u.test(word) ? word.slice(0, -1) : word;
}

const PHOTO_NEGATIVE_VOCABULARY: ReadonlySet<string> = new Set([
  ...NEGATIVE_QUALITY_TERMS.map(negativeVocabularyKey),
  ...NEGATIVE_DEFECT_QUALIFIERS.flatMap((qualifier) => NEGATIVE_DEFECT_PARTS.map((part) => negativeVocabularyKey(`${qualifier} ${part}`)))
]);

/** True for a term a photo negative may carry (see the allow-list above). */
export function isPhotoNegativeQualityTerm(term: string): boolean {
  const key = negativeVocabularyKey(term);
  return key.length > 0 && PHOTO_NEGATIVE_VOCABULARY.has(key);
}

/**
 * Comma-list merge for negative prompts: canonical text, NAI and SD weights stripped, case-insensitive dedupe, order
 * kept, and only the allow-listed vocabulary (isPhotoNegativeQualityTerm) kept, whoever wrote it — a preset, the
 * settings, a character, the profile studio's box or a cut's negative weight. A negative of "clothing", "denim",
 * "adult", "breasts" or "in her twenties" steers the sampler toward the nude or younger figure at any CFG above 1,
 * whatever the positive prompt says. "nsfw" or "child" are dropped too: they would become the picture if the guidance
 * were ever inverted, and the picture's safety rests on the positive prompt, which the floor reads.
 */
function mergeNegativePrompts(...lists: Array<string | undefined>): string {
  const seen = new Set<string>();
  const merged: string[] = [];
  for (const list of lists) {
    if (typeof list !== "string" || !list) {
      continue;
    }
    for (const raw of canonicalizeSceneText(list).split(/[,\n]/u)) {
      const part = normalizeTagText(stripPictographs(raw.replace(/^\s*-?\d+(?:\.\d+)?\s*::/u, "").replace(/^\s*::/u, "").replace(/::\s*$/u, "")));
      const key = part.toLowerCase();
      if (!part || hasUnverifiableScript(part) || isPhotoNegativeSafeguardTerm(part) || !isPhotoNegativeQualityTerm(part) || seen.has(key)) {
        continue;
      }
      seen.add(key);
      merged.push(part);
    }
  }
  return merged.join(", ");
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}
