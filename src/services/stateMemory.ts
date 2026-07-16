import type { MemoryEvent } from "../types";

export const IMAGE_STATE_TYPES = [
  "Wearing",
  "OutfitTags",
  "StatusTags",
  "ExpressionTags",
  "PoseTags",
  "ActionTags",
  "InteractionTags",
  "InteractionPhaseTags",
  "HeldItemTags",
  "PhysicalStateTags",
  "BodyStateTags",
  "SceneTags",
  "ScenePhaseTags",
  "CompositionTags",
  "CameraTags",
  "LightingTags",
  "EnvironmentTags",
  "Location",
  "PhysicalCondition",
  "Emotion"
] as const;

// Canonical state types used exclusively as NovelAI image-prompt machinery. These carry no narrative
// value and must be filtered from the narrative LLM's context view — they induce annotation-mode prose.
// Aliased spellings are caught automatically because callers canonicalise via canonicalizeStateType
// before checking membership here.
// NOT included (narratively load-bearing): Wearing, Location, PhysicalCondition, Emotion, Goal,
// and creator-defined relationship-map parameter titles.
export const IMAGE_MACHINERY_STATE_TYPES = new Set<string>([
  "StatusTags",
  "ExpressionTags",
  "PoseTags",
  "ActionTags",
  "InteractionTags",
  "InteractionPhaseTags",
  "HeldItemTags",
  "PhysicalStateTags",
  "BodyStateTags",
  "SceneTags",
  "ScenePhaseTags",
  "CompositionTags",
  "CameraTags",
  "LightingTags",
  "EnvironmentTags"
]);

export type ImageStateType = (typeof IMAGE_STATE_TYPES)[number];

type ImageStateTagGroups = Partial<Record<ImageStateType, string[]>>;

const IMAGE_STATE_TYPE_SET = new Set<string>(IMAGE_STATE_TYPES);
const STATE_TYPE_ALIASES: Array<{ type: ImageStateType | "Goal" | "Thought" | "InnerThought" | "Intention" | "Relationship"; aliases: string[] }> = [
  { type: "Wearing", aliases: ["Wearing", "OutfitTags", "Outfit", "Clothing", "ClothingTags", "착용", "의상", "의상 태그", "의상태그", "복장", "옷"] },
  { type: "StatusTags", aliases: ["StatusTags", "VisualStateTags", "ConditionTags", "상태", "상태 태그", "상태태그", "컨디션 태그", "캐릭터 상태 태그"] },
  { type: "ExpressionTags", aliases: ["ExpressionTags", "Expression", "FaceTags", "표정", "표정 태그"] },
  { type: "PoseTags", aliases: ["PoseTags", "Pose", "PostureTags", "자세", "자세 태그"] },
  { type: "ActionTags", aliases: ["ActionTags", "Action", "행동", "행동 태그"] },
  { type: "InteractionTags", aliases: ["InteractionTags", "Interaction", "상호작용", "상호작용 태그"] },
  { type: "InteractionPhaseTags", aliases: ["InteractionPhaseTags", "InteractionPhase", "상호작용 단계", "상호작용 단계 태그"] },
  { type: "HeldItemTags", aliases: ["HeldItemTags", "HeldItem", "PropTags", "소지품", "소지품 태그", "손에 든 물건"] },
  { type: "PhysicalStateTags", aliases: ["PhysicalStateTags", "PhysicalConditionTags", "신체 상태 태그"] },
  { type: "BodyStateTags", aliases: ["BodyStateTags", "BodyTags", "신체 태그"] },
  { type: "SceneTags", aliases: ["SceneTags", "Scene", "장면", "장면 태그"] },
  { type: "ScenePhaseTags", aliases: ["ScenePhaseTags", "ScenePhase", "장면 단계", "장면 단계 태그"] },
  { type: "CompositionTags", aliases: ["CompositionTags", "Composition", "구도", "구도 태그"] },
  { type: "CameraTags", aliases: ["CameraTags", "Camera", "카메라", "카메라 태그"] },
  { type: "LightingTags", aliases: ["LightingTags", "Lighting", "조명", "조명 태그"] },
  { type: "EnvironmentTags", aliases: ["EnvironmentTags", "Environment", "환경", "환경 태그"] },
  { type: "Location", aliases: ["Location", "Place", "위치", "장소"] },
  { type: "PhysicalCondition", aliases: ["PhysicalCondition", "Condition", "컨디션", "신체 상태"] },
  { type: "Emotion", aliases: ["Emotion", "Mood", "감정", "기분"] },
  { type: "Goal", aliases: ["Goal", "목표"] },
  { type: "Thought", aliases: ["Thought", "생각"] },
  { type: "InnerThought", aliases: ["InnerThought", "내면", "속마음"] },
  { type: "Intention", aliases: ["Intention", "의도"] },
  { type: "Relationship", aliases: ["Relationship", "관계", "관계 상태", "관계상태"] }
];

const STATE_TYPE_BY_ALIAS = new Map<string, string>(
  STATE_TYPE_ALIASES.flatMap((entry) => entry.aliases.map((alias) => [normalizeStateTypeKey(alias), entry.type]))
);

export function canonicalizeStateType(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) {
    return undefined;
  }

  return STATE_TYPE_BY_ALIAS.get(normalizeStateTypeKey(trimmed)) ?? trimmed;
}

export function isImageStateType(value: string | undefined): value is ImageStateType {
  const canonical = canonicalizeStateType(value);
  return Boolean(canonical && IMAGE_STATE_TYPE_SET.has(canonical));
}

export function readStateMemoryKind(event: MemoryEvent): string {
  return (
    readMetadataText(event.metadata, "memory_kind") ??
    readMetadataText(event.metadata, "kind") ??
    event.tags.find((tag) => tag.startsWith("kind:"))?.slice("kind:".length) ??
    (readStateMemoryStateType(event) ? "state" : "event")
  );
}

export function readStateMemoryOwnerId(event: MemoryEvent): string | undefined {
  return (
    readMetadataText(event.metadata, "owner_id") ??
    readMetadataText(event.metadata, "local_owner_id") ??
    readMetadataText(event.metadata, "actor_id") ??
    readMetadataText(event.metadata, "character_id") ??
    event.actorId
  );
}

export function readStateMemoryTargetId(event: MemoryEvent): string | undefined {
  return readMetadataText(event.metadata, "target_id") ?? readMetadataText(event.metadata, "local_target_id");
}

export function readStateMemoryStateType(event: MemoryEvent): string | undefined {
  return canonicalizeStateType(
    readMetadataText(event.metadata, "state_type") ??
      readMetadataText(event.metadata, "stateType") ??
      readStateTypeTag(event.tags) ??
      parseStateMemoryContent(event.content)?.stateType
  );
}

export function readStateMemoryValue(event: MemoryEvent): string | undefined {
  return cleanStateValue(
    readMetadataText(event.metadata, "value") ??
      readMetadataText(event.metadata, "state_value") ??
      readMetadataText(event.metadata, "stateValue") ??
      parseStateMemoryContent(event.content)?.value
  );
}

export function parseStateMemoryContent(content: string): { stateType?: string; value?: string } | undefined {
  const stripped = stripMemoryPrefix(content);
  const assignment = /^(.*?)\s*(?:=|:|：)\s*(.+)$/u.exec(stripped);
  if (!assignment?.[1]?.trim() || !assignment[2]?.trim()) {
    return undefined;
  }

  const stateType = readStateTypeFromLeftSide(assignment[1]);
  if (!stateType) {
    return undefined;
  }

  return {
    stateType,
    value: cleanStateValue(assignment[2])
  };
}

export function classifyImageStateTags(tags: string[]): ImageStateTagGroups {
  const groups: ImageStateTagGroups = {};
  for (const rawTag of tags.flatMap(splitImageStateTagValue)) {
    const tag = normalizeImageStateTag(rawTag);
    if (!tag) {
      continue;
    }

    const stateType = classifyImageStateTag(tag);
    if (!stateType) {
      continue;
    }

    groups[stateType] = uniqueStrings([...(groups[stateType] ?? []), tag]).slice(0, 8);
  }

  return groups;
}

export function formatImageStateTypeLabel(stateType: string): string {
  const labels: Record<string, string> = {
    Wearing: "착용",
    OutfitTags: "의상 태그",
    StatusTags: "상태 태그",
    ExpressionTags: "표정 태그",
    PoseTags: "자세 태그",
    ActionTags: "행동 태그",
    InteractionTags: "상호작용 태그",
    InteractionPhaseTags: "상호작용 단계",
    HeldItemTags: "소지품 태그",
    PhysicalStateTags: "신체 상태 태그",
    BodyStateTags: "신체 태그",
    SceneTags: "장면 태그",
    ScenePhaseTags: "장면 단계",
    CompositionTags: "구도 태그",
    CameraTags: "카메라 태그",
    LightingTags: "조명 태그",
    EnvironmentTags: "환경 태그",
    Location: "위치",
    PhysicalCondition: "컨디션",
    Emotion: "감정"
  };

  return labels[stateType] ?? stateType;
}

export function splitImageStateTagValue(value: string | undefined): string[] {
  if (!value) {
    return [];
  }

  return value
    .split(/[,;|\n]+/u)
    .map((tag) => normalizeImageStateTag(tag))
    .filter((tag): tag is string => Boolean(tag));
}

function classifyImageStateTag(tag: string): ImageStateType | undefined {
  if (isCameraStateTag(tag)) {
    return "CameraTags";
  }
  if (isLightingStateTag(tag)) {
    return "LightingTags";
  }
  if (isSceneStateTag(tag)) {
    return "SceneTags";
  }
  if (isCompositionStateTag(tag)) {
    return "CompositionTags";
  }
  if (isClothingStateTag(tag)) {
    return "Wearing";
  }
  if (isHeldItemStateTag(tag)) {
    return "HeldItemTags";
  }
  if (isSexPositionStateTag(tag)) {
    return "InteractionTags";
  }
  if (isInteractionStateTag(tag)) {
    return "InteractionTags";
  }
  if (isPoseStateTag(tag)) {
    return "PoseTags";
  }
  if (isActionStateTag(tag)) {
    return "ActionTags";
  }
  if (isPhysicalStateTag(tag)) {
    return "PhysicalStateTags";
  }
  if (isExpressionOrEmotionStateTag(tag)) {
    return "StatusTags";
  }

  return undefined;
}

function readStateTypeTag(tags: string[]): string | undefined {
  const tag = tags.find((item) => /^state(?:_type|-type)?[:=]/iu.test(item));
  return tag?.replace(/^state(?:_type|-type)?[:=]/iu, "").trim();
}

function readStateTypeFromLeftSide(left: string): string | undefined {
  const normalizedLeft = normalizeStateTypeKey(left);
  const sortedAliases = [...STATE_TYPE_BY_ALIAS.entries()].sort((a, b) => b[0].length - a[0].length);
  return sortedAliases.find(([alias]) => normalizedLeft.endsWith(alias))?.[1];
}

function readMetadataText(metadata: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = metadata?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function stripMemoryPrefix(value: string): string {
  return value.replace(/^\[(?:Event|State|Observation|Belief|OpenThread|Goal)\]\s*/iu, "").trim();
}

function cleanStateValue(value: string | undefined): string | undefined {
  const normalized = value
    ?.replace(/\s+/gu, " ")
    .replace(/[。！？.!?]+$/gu, "")
    .trim();
  return normalized || undefined;
}

function normalizeStateTypeKey(value: string): string {
  return value.toLowerCase().replace(/\s+/gu, "").replace(/[^\p{L}\p{N}_:-]+/gu, "");
}

function normalizeImageStateTag(value: string | undefined): string | undefined {
  const weighted = parseWeightedTag(value);
  const tag = (weighted ?? value)
    ?.replace(/[{}[\]"'`]/gu, "")
    .replace(/[._-]+/gu, " ")
    .replace(/\s+/gu, " ")
    .trim();
  if (!tag || /[\u3131-\uD79D]/u.test(tag) || tag.length < 2 || tag.length > 64) {
    return undefined;
  }

  return tag;
}

function parseWeightedTag(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) {
    return undefined;
  }

  const leadingWeight = /^::\s*-?\d+(?:\.\d+)?\s*::\s*(.*?)\s*(?:::)?$/u.exec(trimmed);
  if (leadingWeight?.[1]?.trim()) {
    return leadingWeight[1].trim();
  }

  const wrapped = /^::\s*(.*?)\s*::$/u.exec(trimmed);
  return wrapped?.[1]?.trim();
}

function isClothingStateTag(tag: string): boolean {
  return /\b(?:school uniform|uniform|training clothes|practice clothes|casual clothes|idol stage outfit|dress|jacket|cardigan|coat|raincoat|pajamas|shirt|blouse|sweater|hoodie|pants|shorts|skirt|pencil skirt|pleated skirt|necktie|ribbon|shoes|sneakers|boots|socks|thighhighs|apron|suit|cape|cloak|hat|cap|mask|gloves|armor|changed clothes)\b/iu.test(tag);
}

function isExpressionOrEmotionStateTag(tag: string): boolean {
  return /\b(?:smile|worried|tense|nervous|surprised|frustrated|angry|sad|relieved|confident|defiant|crying|tears?|blush|sweat|open mouth|closed mouth|half closed eyes|closed eyes|wide eyes|tired|exhausted|injured|sick|fever|pain|wet hair|messy hair)\b/iu.test(tag);
}

function isPoseStateTag(tag: string): boolean {
  return /\b(?:standing|sitting|lying|kneeling|crouching|leaning|bending|reaching|arm up|arms up|hand up|hands up|legs apart|crossed arms|head tilt|turning around|looking back)\b/iu.test(tag);
}

function isActionStateTag(tag: string): boolean {
  return /\b(?:walking|running|dancing|singing|training|practicing|reading|writing|eating|drinking|cooking|fighting|searching|opening door|entering|leaving|getting up|falling|jumping|waving|struggling|crawling|collapsing|stumbling|thrusting|grinding|kneeling over|bending over)\b/iu.test(tag);
}

function isInteractionStateTag(tag: string): boolean {
  return /\b(?:hugging|holding hands|hand on|hands on|touching|grabbing|pulling|pushing|facing each other|talking|arguing|whispering|comforting|protecting|helping|strangling|choking|chokehold|hand on another's neck|hands on another's neck|pinning|pinned|pinned down|restraining|straddling|wrestling|grappling|gripping|kissing|biting|licking|carrying|lifting)\b/iu.test(tag);
}

// Sexual position / coital posture tags. These describe an ongoing position that must stay stable across cuts and
// turns until the scene changes it, so they are persisted as InteractionTags (the continuity-carried state group).
function isSexPositionStateTag(tag: string): boolean {
  return /\b(?:sex|vaginal|anal|oral|fellatio|cunnilingus|penetration|insertion|cowgirl position|reverse cowgirl|girl on top|missionary|doggystyle|doggy style|sex from behind|prone bone|mating press|spooning|spoons sex|standing sex|suspended congress|piledriver|full nelson|leg lock|leglock|face sitting|facesitting|lap sitting|lap pillow|all fours|on all fours|bent over|on back|on stomach|legs up|legs held open|spread legs|grinding|riding|cowgirl|deepthroat|paizuri|grabbing another's|hetero|doggy)\b/iu.test(tag);
}

function isHeldItemStateTag(tag: string): boolean {
  return /\b(?:holding|carrying|microphone|phone|smartphone|book|notebook|pen|bag|umbrella|key|letter|cup|bottle|weapon|sword|gun|paper|document)\b/iu.test(tag);
}

function isPhysicalStateTag(tag: string): boolean {
  return /\b(?:injured|bandage|bruise|blood|fever|sick|pain|fatigue|exhausted|sweat|wet|dirty|muddy|trembling|shivering)\b/iu.test(tag);
}

function isSceneStateTag(tag: string): boolean {
  return /\b(?:classroom|school|library|archive|bookshelf|practice room|dance studio|dormitory|apartment|kitchen|cafe|restaurant|hospital|clinic|stage|spotlight|street|alley|hallway|bedroom|rooftop|rain|snow|night|daylight|indoors|outdoors|background)\b/iu.test(tag);
}

function isCompositionStateTag(tag: string): boolean {
  return /\b(?:solo|2girls|3girls|1girl|1boy|2boys|group|upper body|full body|cowboy shot|portrait|wide shot|medium shot|close up|close-up|face focus|depth of field)\b/iu.test(tag);
}

function isCameraStateTag(tag: string): boolean {
  return /\b(?:looking at viewer|looking away|pov|from side|from behind|front view|side view|low angle|high angle|dutch angle|over the shoulder|over-the-shoulder|fisheye)\b/iu.test(tag);
}

function isLightingStateTag(tag: string): boolean {
  return /\b(?:sunlight|moonlight|warm light|cold light|rim light|backlight|spotlight|stage lights|neon lights|soft lighting|dramatic lighting|low light|window light|glow)\b/iu.test(tag);
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}
