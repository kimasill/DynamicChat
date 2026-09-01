import type { ImageCue, ImageCueBodyRegion, ImageCueFrame, ImageCueShot, ImageCueViewpoint } from "../types";

// Frame reasoning for the image pipeline.
//
// The tag contract has always TOLD the image-tag LLM to decide the crop first and emit only what that crop
// reveals, but nothing observed or enforced it, and the local injection layer (saved identity, current
// outfit, persisted posture/detail continuity) never saw the crop at all — it appended a full wardrobe,
// including footwear, into a face close-up, and refilled a whole-body pose into a crop that cannot show one.
// This module turns the crop into a first-class value: parsed from the cue when the model authored it,
// inferred from the composition tags when it did not, and then used to filter both the authored caption and
// everything DynamicChat injects.

const SHOT_VALUES: ImageCueShot[] = ["close_up", "face_focus", "upper_body", "cowboy_shot", "full_body", "wide_shot"];
const VIEWPOINT_VALUES: ImageCueViewpoint[] = ["front", "side", "from_behind", "pov", "over_the_shoulder"];
const REGION_VALUES: ImageCueBodyRegion[] = ["head", "torso", "hips", "legs", "feet"];
type ImageCueAngle = NonNullable<ImageCueFrame["angle"]>;
const ANGLE_VALUES: ImageCueAngle[] = ["eye_level", "from_above", "from_below", "dutch_angle"];

/** What each crop can physically show, used when the model did not enumerate visible_regions itself. */
const SHOT_REGIONS: Record<ImageCueShot, ImageCueBodyRegion[]> = {
  close_up: ["head"],
  face_focus: ["head"],
  upper_body: ["head", "torso"],
  cowboy_shot: ["head", "torso", "hips", "legs"],
  full_body: ["head", "torso", "hips", "legs", "feet"],
  wide_shot: ["head", "torso", "hips", "legs", "feet"]
};

/** Ordering used when a cue names two conflicting shot tags — the tighter crop wins, since it is the safer bound. */
const SHOT_TIGHTNESS: Record<ImageCueShot, number> = {
  close_up: 0,
  face_focus: 1,
  upper_body: 2,
  cowboy_shot: 3,
  full_body: 4,
  wide_shot: 5
};

const SHOT_TAG_PATTERNS: Array<{ shot: ImageCueShot; pattern: RegExp }> = [
  { shot: "face_focus", pattern: /\b(?:face focus|portrait|head shot|headshot|facial close ?-?up)\b/iu },
  { shot: "close_up", pattern: /\b(?:close ?-?up|extreme close ?-?up|macro shot)\b/iu },
  { shot: "upper_body", pattern: /\b(?:upper body|bust shot|chest ?-?up|waist ?-?up)\b/iu },
  { shot: "cowboy_shot", pattern: /\b(?:cowboy shot|knee ?-?up|thigh ?-?up|three quarters? shot)\b/iu },
  { shot: "full_body", pattern: /\b(?:full body|whole body|full ?-?length)\b/iu },
  { shot: "wide_shot", pattern: /\b(?:wide shot|long shot|establishing shot|scenery|landscape|panorama|bird'?s ?-?eye view)\b/iu }
];

const VIEWPOINT_TAG_PATTERNS: Array<{ viewpoint: ImageCueViewpoint; pattern: RegExp }> = [
  { viewpoint: "from_behind", pattern: /\b(?:from behind|back focus|rear view|view from behind|facing away|back to viewer)\b/iu },
  { viewpoint: "over_the_shoulder", pattern: /\b(?:over ?-?the ?-?shoulder|over shoulder)\b/iu },
  { viewpoint: "pov", pattern: /\b(?:pov|point of view|first ?-?person view)\b/iu },
  { viewpoint: "side", pattern: /\b(?:from side|profile view|side view|facing to the side)\b/iu },
  { viewpoint: "front", pattern: /\b(?:facing viewer|front view|looking at viewer|facing forward)\b/iu }
];

/**
 * Body region a tag belongs to. Only tags that clearly belong to ONE region are classified; anything
 * ambiguous (an action, a mood, the setting) returns undefined and is never filtered out on frame grounds.
 */
/**
 * Gaze / head-orientation phrases. Classified before the region table because the torso pattern's `\bback\b`
 * otherwise swallows "looking back" — and head orientation is exactly what a face crop DOES show.
 */
const HEAD_ORIENTATION_PATTERN = /\blooking\s+(?:back|away|at viewer|up|down|to the side|aside)\b/iu;

const REGION_TAG_PATTERNS: Array<{ region: ImageCueBodyRegion; pattern: RegExp }> = [
  {
    region: "head",
    pattern:
      /\b(?:hair|bangs|ponytail|twintails|braid|hairband|hairclip|hair ornament|hat|cap|beret|helmet|hood|headband|tiara|crown|veil|mask|eyepatch|glasses|eyewear|sunglasses|goggles|earrings?|ear ?-?piercing|eyes?|eyelashes|eyebrows?|iris|pupils?|gaze|mouth|lips?|teeth|tongue|nose|cheeks?|chin|forehead|face|smile|smiling|frown|grin|blush|blushing|crying|tears|tearful|open mouth|closed mouth|clenched teeth|ahegao|expression|makeup|lipstick|choker|collar|necklace|scarf|neck)\b/iu
  },
  {
    region: "torso",
    pattern:
      /\b(?:shirt|blouse|t-?shirt|sweater|hoodie|jacket|coat|blazer|vest|cardigan|uniform top|kimono|yukata|robe|dress|gown|apron|bra|bikini top|camisole|tank ?top|crop top|corset|breasts?|cleavage|chest|nipples?|areola|collarbone|shoulders?|back|spine|arms?|elbow|forearm|wrist|hands?|fingers?|nails|armpits?|tie|necktie|bowtie|suspenders|belt buckle|shoulder bag)\b/iu
  },
  {
    region: "hips",
    pattern:
      /\b(?:skirt|shorts|pants|trousers|jeans|leggings|panties|underwear|thong|briefs|hakama|belt|waist|hips?|navel|stomach|abs|abdomen|groin|crotch|buttocks|ass|butt|pubic|vagina|pussy|penis|anus|garter ?belt)\b/iu
  },
  {
    region: "legs",
    pattern: /\b(?:thighs?|thighhighs?|thigh-?highs|stockings|pantyhose|tights|socks|knees?|calves|shins|legs?|kneehighs?)\b/iu
  },
  {
    region: "feet",
    pattern: /\b(?:shoes?|boots?|sneakers|heels|high heels|sandals|loafers|slippers|barefoot|feet|foot|toes?|ankles?|footwear)\b/iu
  }
];

/** Tags that only make sense when a face is actually pointed at the camera. */
const FACE_DEPENDENT_PATTERN =
  /\b(?:looking at viewer|eye contact|smile|smiling|grin|frown|pout|blush|blushing|crying|tears|tearful|open mouth|closed mouth|clenched teeth|gritted teeth|ahegao|surprised|shocked|angry|sad|happy|embarrassed|seductive|expressionless|empty eyes|half-?closed eyes|wide eyes|narrowed eyes|(?:red|blue|green|brown|black|purple|golden|amber|grey|gray|violet|pink|yellow) eyes|heterochromia|eyelashes|makeup|lipstick)\b/iu;

/**
 * Garments and worn accessories, as opposed to actions, poses and body details.
 *
 * The distinction matters for how far the frame filter may go with tags the MODEL wrote. Clothing is
 * objective and positional — a thighhigh is on the legs, full stop — so a garment outside the crop is a
 * plain error and safe to drop. An action or a body detail is a judgement call the author may have made
 * deliberately (a close-up that deliberately includes a bloodied arm), so those are left alone.
 */
const GARMENT_TAG_PATTERN =
  /\b(?:hat|cap|beret|helmet|hood|headband|hairband|tiara|crown|veil|glasses|sunglasses|goggles|earrings?|necklace|choker|scarf|shirt|blouse|t-?shirt|sweater|hoodie|jacket|coat|blazer|vest|cardigan|uniform|kimono|yukata|robe|dress|gown|apron|bra|camisole|tank ?top|crop top|corset|necktie|bowtie|tie|suspenders|skirt|shorts|pants|trousers|jeans|leggings|panties|underwear|thong|briefs|hakama|belt|garter ?belt|thighhighs?|thigh-?highs|stockings|pantyhose|tights|socks|kneehighs?|shoes?|boots?|sneakers|heels|sandals|loafers|slippers|footwear)\b/iu;

export function isGarmentTag(tag: string): boolean {
  return GARMENT_TAG_PATTERN.test(toLowerBare(tag));
}

/** Whole-body posture that a head-and-shoulders crop cannot express. */
const WHOLE_BODY_POSE_PATTERN =
  /\b(?:standing|sitting|seated|lying|lying down|on back|on stomach|kneeling|crouching|squatting|leaning|bending over|walking|running|jumping|spread legs|legs apart|legs up|crossed legs|arms up|hands on hips|arms crossed|all fours|straddling|riding|carrying|full body pose|contrapposto)\b/iu;

function toLowerBare(tag: string): string {
  // Strip NovelAI weight syntax (1.5::tag::, {tag}, [tag]) so patterns match the bare term.
  return tag
    .replace(/^\s*-?\d+(?:\.\d+)?\s*::/u, "")
    .replace(/::\s*$/u, "")
    .replace(/[{}[\]]/gu, "")
    .trim()
    .toLowerCase();
}

function readEnum<T extends string>(value: unknown, allowed: T[]): T | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const normalized = value.trim().toLowerCase().replace(/[\s-]+/gu, "_");
  return allowed.find((candidate) => candidate === normalized);
}

/** Region a single tag belongs to, or undefined when it is not region-bound at all. */
export function readTagBodyRegion(tag: string): ImageCueBodyRegion | undefined {
  const bare = toLowerBare(tag);
  if (HEAD_ORIENTATION_PATTERN.test(bare)) {
    return "head";
  }
  // Head first: "neck" and "collar" sit between head and torso and read better as head-adjacent, and a
  // face-region term inside a longer tag should not be shadowed by an incidental torso word.
  for (const { region, pattern } of REGION_TAG_PATTERNS) {
    if (pattern.test(bare)) {
      return region;
    }
  }
  return undefined;
}

export function isFaceDependentTag(tag: string): boolean {
  return FACE_DEPENDENT_PATTERN.test(toLowerBare(tag));
}

export function isWholeBodyPoseTag(tag: string): boolean {
  return WHOLE_BODY_POSE_PATTERN.test(toLowerBare(tag));
}

/**
 * True when the chosen frame can actually show a face pointed at the viewer.
 *
 * Only `from_behind` hides it. A `pov` or `over_the_shoulder` cut is shot THROUGH the observer, so the
 * rendered subject is normally facing the camera — treating pov as face-hidden deleted every expression,
 * gaze and eye-colour tag from the most common composition in the shipped simulations.
 */
export function frameShowsFace(frame: ImageCueFrame): boolean {
  if (!Array.isArray(frame.visibleRegions) || !frame.visibleRegions.includes("head")) {
    return false;
  }
  return frame.viewpoint !== "from_behind";
}

/** True when the frame is tight enough that whole-body posture is out of shot. */
export function frameHidesWholeBodyPose(frame: ImageCueFrame): boolean {
  // Derived from the visible regions rather than the shot label, so a body-region close-up (which the tag
  // contract explicitly asks for) is judged by what it actually shows.
  return Array.isArray(frame.visibleRegions) && frame.visibleRegions.length <= 1;
}

/** Best-effort frame read from composition tags, used when the model did not author a frame object. */
/** Viewpoint named by the cut's own tags, or undefined when they name none. */
function readViewpointFromTags(tags: string[]): ImageCueViewpoint | undefined {
  const haystack = tags.map(toLowerBare).join(", ");
  for (const { viewpoint, pattern } of VIEWPOINT_TAG_PATTERNS) {
    if (pattern.test(haystack)) {
      return viewpoint;
    }
  }
  return undefined;
}

export function inferImageCueFrameFromTags(tags: string[]): ImageCueFrame | undefined {
  const bare = tags.map(toLowerBare);
  const haystack = bare.join(", ");

  let shot: ImageCueShot | undefined;
  for (const { shot: candidate, pattern } of SHOT_TAG_PATTERNS) {
    if (pattern.test(haystack)) {
      // Two conflicting crop tags in one cut is a model error; take the tighter one so nothing
      // out-of-frame slips through on the strength of the looser tag.
      shot = shot === undefined || SHOT_TIGHTNESS[candidate] < SHOT_TIGHTNESS[shot] ? candidate : shot;
    }
  }
  if (!shot) {
    return undefined;
  }

  return { shot, viewpoint: readViewpointFromTags(tags) ?? "front", visibleRegions: SHOT_REGIONS[shot] };
}

/**
 * Reads the model's `frame` object, falling back to tag inference. Returns undefined when neither source
 * gives a crop — callers must then leave every tag alone rather than guess a frame and filter on it.
 */
export function normalizeImageCueFrame(value: unknown, tags: string[] = []): ImageCueFrame | undefined {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    const shot = readEnum(record.shot, SHOT_VALUES);
    if (shot) {
      // Fall back to the cut's OWN tags before defaulting to "front". A model that omits `viewpoint`, or
      // writes a synonym the enum does not carry ("back view", "rear"), used to be recorded as facing the
      // camera — and stripConflictingCompositionTags then deleted the authored "from behind" tag for
      // conflicting with that invented frame, while frameShowsFace re-admitted eye and expression tags.
      // A from-behind cut therefore rendered face-on: the exact failure the frame system exists to prevent.
      const viewpoint = readEnum(record.viewpoint, VIEWPOINT_VALUES) ?? readViewpointFromTags(tags) ?? "front";
      const declaredRegions = Array.isArray(record.visible_regions ?? record.visibleRegions)
        ? ((record.visible_regions ?? record.visibleRegions) as unknown[])
            .map((region) => readEnum(region, REGION_VALUES))
            .filter((region): region is ImageCueBodyRegion => Boolean(region))
        : [];
      const shotRegions = SHOT_REGIONS[shot];
      // `close_up` is the one shot whose reach is NOT fixed — the contract asks for body-region close-ups
      // (a neck cut during a strangling, a hip cut during sex), so a declared region is authoritative there.
      const visibleRegions =
        shot === "close_up" && declaredRegions.length > 0
          ? declaredRegions
          : // For every other shot the declaration may only NARROW within the shot's reach. A declaration
            // that strays outside it is self-contradictory — an `upper_body` listing hips and legs, which a
            // real open-weight model does emit — and intersecting such a list silently deleted the head,
            // taking the whole face with it. A contradictory declaration falls back to the shot label; a
            // self-consistent one is honoured.
            declaredRegions.every((region) => shotRegions.includes(region))
            ? declaredRegions
            : [];
      return {
        shot,
        viewpoint,
        angle: readEnum(record.angle, ANGLE_VALUES),
        visibleRegions: visibleRegions.length > 0 ? visibleRegions : shotRegions
      };
    }
  }
  return inferImageCueFrameFromTags(tags);
}

/**
 * Frame for a planned cue: whatever the model authored, else inferred from its own tags.
 *
 * The authored frame is re-normalized rather than trusted verbatim — a cue can reach here carrying the raw
 * snake_case shape the model emits (`{shot, viewpoint, visible_regions}`), which has no `visibleRegions` at
 * all and would throw the moment anything read it.
 */
export function resolveImageCueFrame(cue: Pick<ImageCue, "frame" | "baseTags" | "tags">): ImageCueFrame | undefined {
  const tags = [...(cue.baseTags ?? []), ...(cue.tags ?? [])];
  return cue.frame ? normalizeImageCueFrame(cue.frame, tags) : inferImageCueFrameFromTags(tags);
}

export interface FrameFilterOptions {
  /**
   * Tags the LLM wrote for THIS cut, as opposed to ones DynamicChat injected. Authored tags keep the benefit
   * of the doubt on actions, poses and body details — those are evidence about the scene that a keyword map
   * should not overrule. They do NOT keep it on garments or on facing-camera face tags, which are objective
   * and are the two mistakes a reader actually notices (footwear on an upper-body shot, an expression on a
   * figure turned away). Open-weight models make both regularly, so the contract alone is not enough.
   */
  authored?: boolean;
}

/**
 * Drops tags the chosen frame cannot show.
 *
 * Injected tags (saved identity, current outfit, persisted posture/detail continuity) are filtered by body
 * region and by posture: that layer never saw the crop, and it is what put a full wardrobe — footwear
 * included — inside a face close-up. Authored tags are filtered ONLY for the flatly impossible case (a
 * facing-camera expression on a figure turned away), because the region map is a keyword heuristic and the
 * author has information it does not: a "close-up" that deliberately includes a bloodied arm is a real shot,
 * and silently deleting the beat the cut exists for is worse than a slightly loose crop. Keeping authored
 * tags out of frame is the tag contract's job (rule 3 / rule 10), not this filter's.
 */
export function filterTagsForFrame(
  tags: string[],
  frame: ImageCueFrame | undefined,
  options: FrameFilterOptions = {}
): string[] {
  if (!frame) {
    return tags;
  }
  const authored = options.authored === true;
  const showsFace = frameShowsFace(frame);
  const hidesPose = frameHidesWholeBodyPose(frame);
  return tags.filter((tag) => {
    if (!showsFace && isFaceDependentTag(tag)) {
      return false;
    }
    if (authored) {
      // Garments are held to the crop even when the model wrote them; everything else the author keeps.
      if (!isGarmentTag(tag)) {
        return true;
      }
      const garmentRegion = readTagBodyRegion(tag);
      return garmentRegion === undefined || frame.visibleRegions.includes(garmentRegion);
    }
    if (hidesPose && isWholeBodyPoseTag(tag)) {
      return false;
    }
    const region = readTagBodyRegion(tag);
    return region === undefined || frame.visibleRegions.includes(region);
  });
}

/** Canonical NovelAI tag for each crop. */
const SHOT_TAGS: Record<ImageCueShot, string> = {
  close_up: "close-up",
  face_focus: "face focus",
  upper_body: "upper body",
  cowboy_shot: "cowboy shot",
  full_body: "full body",
  wide_shot: "wide shot"
};

/** Canonical NovelAI tag for each viewpoint. `front` is the default and needs no tag. */
const VIEWPOINT_TAGS: Partial<Record<ImageCueViewpoint, string>> = {
  from_behind: "from behind",
  side: "from side",
  pov: "pov",
  over_the_shoulder: "over the shoulder"
};

const ANGLE_TAGS: Record<NonNullable<ImageCueFrame["angle"]>, string | undefined> = {
  eye_level: undefined,
  from_above: "from above",
  from_below: "from below",
  dutch_angle: "dutch angle"
};

/**
 * The composition tags the chosen frame implies.
 *
 * NovelAI never sees the `frame` object — only tags. Measured against a live open-weight model, the frame
 * was declared on every cue but mirrored into base_tags only about half the time, so half the crops were
 * decided and then never rendered. Deriving them here makes the crop take effect whether or not the model
 * remembered to write it, and stops the declared frame and the actual prompt from disagreeing.
 */
/** Every shot/viewpoint tag this module knows how to emit, for conflict removal. */
const ALL_COMPOSITION_TAGS = [
  ...Object.values(SHOT_TAGS),
  ...Object.values(VIEWPOINT_TAGS).filter((tag): tag is string => Boolean(tag)),
  // Common synonyms a creator's style module or a preset may carry.
  "close up",
  "closeup",
  "portrait",
  "environmental portrait",
  "bust shot",
  "knee-up",
  "waist-up",
  "medium shot",
  "long shot",
  "full-length"
];

/**
 * Removes crop tags that contradict the chosen frame.
 *
 * The creator's style module and scene presets routinely carry a baked-in shot tag (the shipped style module
 * has `environmental portrait, close-up`). Once the frame derives its own shot tag, both end up in the same
 * prompt — `close-up, wide shot` — and NovelAI is left to pick. The frame is the authority for the crop, so
 * everything else's crop tags go.
 */
export function stripConflictingCompositionTags(tags: string[], frame: ImageCueFrame | undefined): string[] {
  if (!frame) {
    return tags;
  }
  const keep = new Set(createFrameCompositionTags(frame).map((tag) => tag.toLowerCase()));
  return tags.filter((tag) => {
    const bare = toLowerBare(tag);
    if (keep.has(bare)) {
      return true;
    }
    return !ALL_COMPOSITION_TAGS.some((candidate) => bare === candidate.toLowerCase());
  });
}

export function createFrameCompositionTags(frame: ImageCueFrame | undefined): string[] {
  if (!frame) {
    return [];
  }
  return [SHOT_TAGS[frame.shot], VIEWPOINT_TAGS[frame.viewpoint], frame.angle ? ANGLE_TAGS[frame.angle] : undefined].filter(
    (tag): tag is string => Boolean(tag)
  );
}

/**
 * Negative tags that keep NovelAI from re-introducing what the crop excluded. NovelAI reliably drifts a
 * declared close-up back toward a full body unless the excluded framing is stated negatively.
 */
export function createFrameNegativeTags(frame: ImageCueFrame | undefined): string[] {
  if (!frame) {
    return [];
  }
  const negatives: string[] = [];
  const regions = Array.isArray(frame.visibleRegions) ? frame.visibleRegions : [];
  if (regions.length > 0 && !regions.includes("feet")) {
    negatives.push("feet", "shoes");
  }
  if (regions.length > 0 && regions.length <= 2) {
    negatives.push("full body");
  }
  if (regions.length <= 1) {
    negatives.push("wide shot");
  }
  if (frame.viewpoint === "from_behind") {
    negatives.push("looking at viewer", "facing viewer");
  }
  return negatives;
}

/** Human-readable summary used in traces and the image inspector. */
export function describeImageCueFrame(frame: ImageCueFrame | undefined): string | undefined {
  if (!frame) {
    return undefined;
  }
  return `${frame.shot} / ${frame.viewpoint} / [${frame.visibleRegions.join(", ")}]`;
}
