import type { ImageCueFrame, ImageCueShot, ImageSceneCategory } from "../types";
import { inferImageCueFrameFromTags, isGarmentTag } from "./imageFrame";

// Scene classification for the image backend router.
//
// Every cut is put into exactly one ImageSceneCategory by code, never by the LLM, and always from the COMPOSED
// prompt — saved identity, authored caption, persisted continuity and outfit exactly as they will be sent. The
// authored caption alone is not enough: continuity re-injects an ongoing sex position into a cut the tag model
// wrote as a plain face close-up, and fantasy traits (animal ears, horns) usually live in the saved identity
// rather than in the cut.
//
// Part of the output is a safety floor rather than a preference: explicit content, minor-coded subjects and
// school-coded subjects in a suggestive cut never render photoreal (imageBackendRouting.ts enforces it). The
// lexicons therefore lean toward false positives. A cut wrongly kept on NovelAI costs a style mismatch; a cut
// wrongly sent to a photo model cannot be taken back.
//
// Everything is matched on CANONICAL text (canonicalizeSceneText): NFKC, invisible code points and every combining
// mark removed, accents and the Latin letters without a decomposition folded, letter-like symbols read as letters —
// and, for matching only, hyphens read as spaces. A photo encoder reads "ｎｕｄｅ", "nu​de", "🅽🅰🅺🅴🅳" and
// "school-uniform" as the plain words; lexicons that only knew the plain spelling could be walked past without
// trying. Letters still outside a–z after that (kana, Cyrillic look-alikes, Greek, CJK, Hangul, IPA small capitals)
// cannot be checked by any lexicon here, so they are reported as unverifiable, which locks the cut like the other
// floor flags.
//
// The floor lexicons also read each term a second way (readFloorVariants): leetspeak and punctuation inside a
// word folded away, a symbol read as the letter it looks like or as a masked letter, masked letters filled in, a word
// broken into letters or fragments joined back up, stretched letters collapsed. The canonical term is always read
// too, so a variant can only add a reading — "1girl" and a stated age keep working however their variant looks. A
// lexical floor cannot be complete; these are general rules for the ways a word is disguised, not lists of the
// disguises a red team happened to try. Pictographs are not letters at all: the photo builder strips them, and the
// floor counts the ones with a sexual, child or school meaning first (stripPictographs).
//
// Some codings exist only in combination, because each part is ordinary on an adult: two child-body descriptors on a
// woman in swimwear, a school uniform written as its parts, a bare number next to a person ("Yuna (14)", "14F").

export interface SceneClassificationInput {
  /** The annotation pass's beat kind (free text). Recorded as a signal, never decisive on its own. */
  cueKind?: string;
  /** COMPOSED tags per figure (identity + authored + continuity + outfit). NovelAI weight syntax may be present. */
  characterTagSets: string[][];
  /** Base caption context layer (location / props / lighting / camera) plus user-rule positives. Weights allowed. */
  contextTags: string[];
  /**
   * Free text the photo prompt carries besides the cut itself: the photoreal style's lead, tail and ambient
   * defaults, or a whole finished photo prompt. Prose or comma lists. Read by the safety floor only (explicit,
   * minor-coded, school + suggestive, unverifiable script) and never for the category: style vocabulary ("a
   * smartphone photograph", "phone-camera grain", "a minimal room") is the same in every cut, and read as tags it
   * turned an empty street into an object shot.
   */
  safetyTexts?: string[];
  /** The cut's crop, when known. */
  frame?: ImageCueFrame;
  /** cue.characters.length */
  registeredCharacterCount: number;
}

export interface SceneClassification {
  category: ImageSceneCategory;
  /** `kind:value` strings naming what decided the category, e.g. ["explicit:sex from behind", "subjects:2"]. */
  signals: string[];
  minorCoded: boolean;
  schoolCoded: boolean;
  /** Suggestive wardrobe/framing anywhere in the cut. Always true for an explicit cut. */
  suggestive: boolean;
  /**
   * Some text in the cut has letters no lexicon here can read, even after canonicalisation (kana, Cyrillic, Greek,
   * CJK, Hangul, IPA). Nothing can check what it says, so the router locks the cut like the other floor flags.
   * Reported as `unverifiable-script:<word>` signals.
   */
  unverifiable: boolean;
  /** max(figure count, subject-count tags, registered character count). */
  humanCount: number;
}

/**
 * A lexicon entry: `pattern` finds the term, `except` names the false friends that share its words.
 *
 * `except` rejects the WHOLE term, so the safety-floor lexicons (explicit, minor, school, suggestive) never use
 * it: "child in a baby blue dress" must still be minor-coded. They exclude false friends with inline lookarounds
 * scoped to the one word instead.
 */
interface TermRule {
  pattern: RegExp;
  except?: RegExp;
}

const PORTRAIT_SHOTS: ReadonlySet<ImageCueShot> = new Set<ImageCueShot>(["close_up", "face_focus", "upper_body"]);
const FULL_BODY_SHOTS: ReadonlySet<ImageCueShot> = new Set<ImageCueShot>(["cowboy_shot", "full_body", "wide_shot"]);

const MAX_SIGNALS_PER_KIND = 4;

/** `\b(?:a|b|c)\b` over lexicon fragments, which read better one concern per line than as one alternation. */
function wordAlternation(fragments: readonly string[]): RegExp {
  return new RegExp(String.raw`\b(?:${fragments.join("|")})\b`, "u");
}

// ---------------------------------------------------------------------------------------------------------------
// Explicit
// ---------------------------------------------------------------------------------------------------------------

/**
 * Body-identity traits from a character's saved NovelAI identity (futanari and its synonyms). They describe WHO the
 * character is, and every registered character's identity is injected into every cut it appears in — so treating
 * them as explicit locked every single cut of a simulation whose cast carries one, clothed cafe portraits included
 * (measured on the shipped adult seed: 16/16 cuts of a real turn pinned to NovelAI). What a cut SHOWS is decided by
 * the explicit lexicon below — nudity, genitals, acts — which still catches any cut where the trait would be
 * visible. The trait itself never reaches a photo prompt (photorealPrompt drops it), so a photo can only render
 * the clothed person the rest of the caption describes.
 */
export const IDENTITY_ANATOMY_TRAIT_PATTERN = /\b(?:futanari|futa|dickgirl|newhalf|intersex|hermaphrodite)\b/u;

// "nude" as a cosmetic, a named colour or an accessory (nude lipstick, nude-toned blazer, a nude tote bag) is
// everyday vocabulary in the daily-look style this backend exists for. Shoes and hosiery stay explicit: a photo
// model can read "nude heels" or "nude stockings" as a nude woman wearing them.
const NUDE_AS_COLOUR = String.raw`(?! (?:colou?r(?:ed)?|tone[sd]?|beige|shade|lipsticks?|lips?|lip (?:gloss|liner|tint|balm)|makeup|make up|nail polish|nails?|manicure|pedicure|polish|eye ?shadows?|palettes?|blush|foundation|concealer|bags?|handbags?|purses?|clutch(?:es)?|totes?|wallets?)\b)`;
// Words that read as a sex act or a body part only next to a certain neighbour. Each exemption names the one
// neighbour that makes it something else, so the same word anywhere else in the term is still read.
const EXPLICIT_FALSE_FRIEND_GUARDS = {
  naked: String.raw`naked(?! (?:cakes?|eye|juice|grapes?|flames?|truth)\b)`,
  porn: String.raw`(?<!food )porn\w*`,
  cum: String.raw`cum(?! laude)`,
  oral: String.raw`oral(?! (?:presentations?|hygiene|exams?|examinations?|surgery|surgeons?|health|care|history|histories|traditions?|arguments?|reports?|tests?|b|medications?|medicine|cavity|thermometers?|defen[cs]es?|interpretations?|skills?|english)\b)`,
  dick: String.raw`(?<!(?:moby|k\.?|van) )dicks?`,
  tits: String.raw`(?<!(?:blue|great|coal|marsh|willow|crested|long tailed|bearded) )tits`,
  boobies: String.raw`(?<!(?:footed|masked|brown|nazca) )boobies`,
  pussy: String.raw`puss(?:y|ies)(?! willows?)`,
  topless: String.raw`top ?less(?! (?:convertibles?|cars?|bus(?:es)?|jeeps?|roadsters?|boats?|cabriolets?)\b)`,
  bottomless: String.raw`bottomless(?! (?:brunch(?:es)?|mimosas?|drinks?|coffee|cups?|refills?|fries|pits?|wells?|glass(?:es)?|bowls?)\b)`,
  missionary: String.raw`missionar(?:y|ies)(?! (?:church(?:es)?|work|workers?|schools?|trips?|hospitals?|society|societies)\b)`,
  erection: String.raw`erections?(?! of\b)`,
  insertion: String.raw`(?<!(?:coin|card|key|needle|catheter) )insertion`,
  fingering: String.raw`fingering(?! (?:charts?|exercises?|patterns?|technique|diagrams?)\b)`,
  strip: String.raw`strip(?:ping|ped|pers?|tease|teasing)(?! (?:(?:old )?wallpaper|paint|wood|furniture|floors?|the bed|bed|sheets|linens?|wires?|varnish)\b)`,
  rape: String.raw`rap(?:e|es|ed|ing|ist|ists|ey)(?! of (?:the )?(?:sabine|proserpina|persephone|europa|lucrece|lucretia|ganymede|lock)\b)`,
  penetration: String.raw`(?<!(?:market|network|light|sun|sunlight|water|rain|armou?r) )penetration(?! test(?:s|ing)?\b)`,
  sperm: String.raw`sperm(?! whales?)`,
  creamPie: String.raw`(?<!(?:banana|coconut|boston|chocolate|lemon|lime|strawberry|vanilla|butterscotch|pumpkin|custard|apple|cherry|peanut butter|sour) )cream pies?`,
  squirting: String.raw`squirt(?:s|ed|ing)(?! (?:guns?|bottles?|water|lemon|ketchup|mustard|cream)\b)`,
  hardOn: String.raw`hard ?ons?(?! (?:her|him|them|me|you|us|herself|himself|themselves|yourself|myself|the|a|an|his|their|my|your|our|everyone|people|someone)\b)`
};

// Body words a false-friend guard lets through as birds (EXPLICIT_FALSE_FRIEND_GUARDS), and what makes the text be
// about birds after all ("blue tits, bird feeder, garden", "blue-footed boobies").
const GUARDED_BODY_WORD_PATTERN = /\b(?:great|blue|coal|marsh|willow|crested|long tailed|bearded) tits\b|\b(?:masked|brown|nazca) boobies\b/u;
const BIRD_CONTEXT_PATTERN =
  /\b(?:feeders?|birds?|birding|bird ?watch\w*|nests?|nesting|branch(?:es)?|perched|perching|flocks?|beaks?|species|wildlife|binoculars|aviary|galapagos|seabirds?|footed|chicks?|eggs?|ornitholog\w*)\b/u;

// Words that make an ambiguous Slavic "naked" (naga, naha, gola) the word it is: a woman, a body, a photo, a bed, a
// beach, "completely".
const SLAVIC_CORROBORATION = String.raw`kobieta|kobiety|kobiete|dziewczyna|dziewczyny|dziewczynka|modelka|modelki|pani|zona|zena|zeny|zenu|divka|divky|holka|holky|cura|cure|djevojka|djevojke|zenska|cialo|telo|tijelo|fotka|fotky|fotografia|fotografie|fotografija|zdjecie|zdjecia|sesja|plaza|plazi|plazy|lozku|lozko|posteli|krevetu|uplne|uplnie|zupelnie|calkiem|cala|caly|sasvim|potpuno|skroz|na plazy|na lozku|v posteli|u krevetu`;

// Terms are canonical (see normalizeSceneTag), so a compound a tag might hyphenate is written with ` ?` at the
// joint: "blow-job", "blow job" and "blowjob" all arrive as one of the two spellings the joint accepts. Inflections
// and the everyday synonyms a tag model or a creator's prose actually uses are listed: "nudes", "seminude",
// "in the buff", "coitus", "jerking off" each walked past the canonical Danbooru vocabulary before.
const EXPLICIT_RULES: TermRule[] = [
  // Nudity, and the ways of saying it without the word.
  {
    pattern: wordAlternation([
      String.raw`(?:semi|half|full|bare|stark|butt|buck) ?(?:nude[sd]?|naked)`,
      `nude[sd]?${NUDE_AS_COLOUR}`,
      EXPLICIT_FALSE_FRIEND_GUARDS.naked,
      String.raw`nudity|nakedness|nakedly|nudies?|nudists?|nudism|naturists?|naturism|nudez|noods?|noodz|nekk?[ie]d|nakk[ie]d|nakid|nakd|nakey|nakie|neked`,
      String.raw`in the buff|birthday suits?|au naturel|disrob(?:e|es|ed|ing)|unclad|full ?frontal|top ?free|free the nipple`,
      EXPLICIT_FALSE_FRIEND_GUARDS.topless,
      EXPLICIT_FALSE_FRIEND_GUARDS.bottomless,
      String.raw`(?:clothes|clothing|garments|everything) (?:removed|off|discarded|stripped)`,
      String.raw`(?:took|takes|take|taking|taken|removed?|removes|removing|stripped|stripping|shed|sheds|shedding|peeled|peels|peeling|pulled|pulls|pulling|threw|throws|tore|tears|tearing) (?:off )?(?:all )?(?:of )?(?:her |his |their |the |my |your )?(?:clothes|clothing|garments)(?: off)?`,
      String.raw`(?:fully|completely|totally|entirely|all) undressed|undressed (?:completely|fully|entirely)`,
      String.raw`(?:not|isn't|is not|wasn't|was not|aren't|weren't|never|without) wearing (?:anything|any clothes|any clothing|a thing|a stitch|clothes|clothing)`,
      String.raw`(?:without|not) a (?:stitch|thread)(?: on| of clothing)?|(?:clothing|clothes) ?free`,
      String.raw`no clothes|no clothing|without (?:any )?(?:clothes|clothing)|wearing nothing|unclothed|bare body`,
      String.raw`skinny ?dip(?:s|ped|ping|per|pers)?|streaking|streakers?|pant(?:ie|i|y)less`,
      // Closed compounds a tag model writes without the space: nudephoto, nakedwoman, sexscene.
      String.raw`(?:nude|naked)(?:photos?|pics?|pictures?|selfies?|wom[ae]n|girls?|boys?|m[ae]n|models?|scenes?|art|beach|bod(?:y|ies))|sex(?:scenes?|tapes?|videos?|toys?|dolls?|workers?|slaves?)`
    ])
  },
  // Exposed bodies and body parts.
  {
    pattern: wordAlternation([
      String.raw`nipples?|nips|nip ?slips?|areolae?|areolas|(?:breast|nipple|areola) slips?|wardrobe malfunctions?`,
      String.raw`(?:breasts?|boobs?|tits) out|bare ?breasted|bare ?bottomed|bare ?assed|crotch ?less`,
      String.raw`(?:exposed|bare|bared|naked|uncovered|visible|freed|exposing|baring|showing|flashing|revealing|displaying) (?:her |his |their |own |both |the )?(?:breasts?|boobs?|tits|nipples|crotch|groin|genitals?|privates|private parts|intimate parts|butt|buttocks|ass|bottom|bum|pussy|vagina|penis)`,
      String.raw`(?:breasts?|boobs?|tits|nipples|crotch|genitals?|privates|private parts|butt|buttocks|ass|bottom|bum) (?:exposed|bared|bare|uncovered|visible|showing|out|hanging out)`,
      EXPLICIT_FALSE_FRIEND_GUARDS.tits,
      String.raw`titties|titty|boobs?(?! tube)|oppai`,
      EXPLICIT_FALSE_FRIEND_GUARDS.boobies,
      EXPLICIT_FALSE_FRIEND_GUARDS.pussy,
      String.raw`vaginas?|vaginal|vulvas?|clit(?:oris|oral|s)?|labia|genitals?|genitalia|naughty bits|intimate parts|private parts|privates exposed|cunts?|twats?|vags?`,
      String.raw`cleft of venus|mons pubis|pubic (?:hair|stubble|tattoos?|mound|area|region)|pubes`,
      String.raw`penis(?:es)?|cocks?|schlongs?|boners?|phallus(?:es)?|phallic|testicles|scrotum|foreskin|glans|phimosis`,
      EXPLICIT_FALSE_FRIEND_GUARDS.dick,
      EXPLICIT_FALSE_FRIEND_GUARDS.erection,
      EXPLICIT_FALSE_FRIEND_GUARDS.hardOn,
      String.raw`anus|anal`,
      EXPLICIT_FALSE_FRIEND_GUARDS.oral
    ])
  },
  // Sex acts, fluids and positions.
  {
    pattern: wordAlternation([
      String.raw`fellatio|blow ?jobs?|cunnilingus|anilingus|rim ?jobs?|irrumatio|paizuri|sumata|(?:tit|hand|foot|butt|hair|thigh|boob|knee|armpit) ?jobs?|tit ?fuck\w*|armpit sex`,
      EXPLICIT_FALSE_FRIEND_GUARDS.fingering,
      String.raw`masturbat\w*|(?:jerk|jack)(?:s|ed|ing)? off|wank\w*|fap(?:s|ped|ping)?|onanii?`,
      String.raw`pleasur(?:e|es|ed|ing) (?:her|him|them|one|your|my)sel(?:f|ves)|self ?pleasur\w*|(?:touch(?:es|ed|ing)?|play(?:s|ed|ing)? with) (?:her|him|them)sel(?:f|ves)`,
      EXPLICIT_FALSE_FRIEND_GUARDS.cum,
      String.raw`cums|cumming|cum ?shots?|precum|semen|jizz|spunk|creampies?|nakadashi|bukkake|ejaculat\w*|love juices?|orgasm\w*`,
      EXPLICIT_FALSE_FRIEND_GUARDS.creamPie,
      EXPLICIT_FALSE_FRIEND_GUARDS.sperm,
      EXPLICIT_FALSE_FRIEND_GUARDS.squirting,
      String.raw`doggy ?style|cowgirl position|reverse cowgirl|mating press|prone bone|suspended congress|face ?sitting|deep ?throat\w*|sixty ?nine position`,
      EXPLICIT_FALSE_FRIEND_GUARDS.missionary,
      EXPLICIT_FALSE_FRIEND_GUARDS.penetration,
      EXPLICIT_FALSE_FRIEND_GUARDS.insertion,
      String.raw`tribadism|scissoring|pegging|humping|fuck\w*|three ?somes?|four ?somes?|orgy|orgies|gang ?bang\w*`,
      String.raw`strap ?ons?(?= (?:dildo|harness|toy)\b|$)`,
      String.raw`intercourse|coitus|coital|post ?coital|copulat\w*|fornicat\w*|shag(?:s|ged|ging)|lovemaking|love making|mak(?:e|es|ing) love|made love`,
      String.raw`sexual (?:intercourse|activity|act|acts|positions?|poses?|content|favou?rs?|encounters?|scenes?)|sexually explicit|sex(?:ual)? slaves?|seggs|secks|sexe|foreplay|climaxing|(?:having|have|had) (?:sexual )?relations`
    ])
  },
  // Coercion, fetish gear, and the genre words that mean the rest.
  {
    pattern: wordAlternation([
      EXPLICIT_FALSE_FRIEND_GUARDS.rape,
      String.raw`molest\w*|sexual(?:ly)? assault\w*|sexual(?:ly)? abus\w*|non ?consensual|non ?con|dub ?con|dubious consent|without (?:her |his |their )?consent|incest\w*|chikan|netorare|netori|ntr|cuckold\w*`,
      EXPLICIT_FALSE_FRIEND_GUARDS.strip,
      String.raw`strip (?:club|show|poker|search)|bondage|bdsm|shibari|ball ?gags?|bit gags?|ring gags?`,
      String.raw`sex toys?|vibrators?|dildos?|anal beads|butt ?plugs?|spread pussy|no panties|no underwear`,
      String.raw`(?:panty|panties|pantsu) (?:pull|aside|down)|bra (?:lift|pull)|clothes down|torn (?:panties|bra|underwear|lingerie)`,
      String.raw`grop(?:e|es|ed|ing)|fondl(?:e|es|ed|ing)|breast (?:grab|hold|press|squeeze|fondl\w*|suck\w*|grop\w*)|grabbing (?:another's |own |her |his )?breasts?`,
      String.raw`covering (?:breasts|nipples|crotch|privates)|hands? (?:over|on) (?:own |her |his |another's )?(?:crotch|breasts?|nipples|privates)|touch(?:es|ed|ing)? (?:her |his |their |own |another's )?(?:breasts?|nipples|crotch|privates)|hand bra|(?:her|his|their|own) privates`,
      String.raw`nsfw\w*|hentai|porno|xxx|x ?rated|onlyfans|r ?18 ?g?|eroge|lactat\w*|milking machine|human milking|hetero(?! couple)|ahegao|torogao|condoms?`,
      EXPLICIT_FALSE_FRIEND_GUARDS.porn,
      String.raw`exhibitionis(?:m|ts?)|public (?:nudity|indecency|sex)|voyeur\w*`,
      // Danbooru's censoring tags exist because what they censor is on the page.
      String.raw`(?:novelty|heart|hair|steam|light|convenient|mosaic|bar|pixel|blur|text|sticker|emoji|star|leaf|bubble) ?censor(?:ed|ing|s)?|(?:un)?censored`,
      String.raw`lewd tattoos?|womb tattoos?|crest tattoos?|reverse bunny ?(?:suits?|girls?)?|reverse outfits?|slingshot swimsuits?|pasties|nipple (?:tape|pasties|covers?)|maebari|body ?paint(?:ed|ing|s)?`,
      String.raw`stomach bulge|belly bulge|internal (?:cumshot|view)|life (?:drawing|modell?ing) models?|nude models?`,
      String.raw`nsfw|rating:\s*explicit|pr0n|glory ?holes?|adult (?:films?|movies?|videos?|content|entertainment|stars?|actress(?:es)?|actors?)|penthouse pets?|baise`
    ])
  },
  // Latin-script languages the photo encoders read natively (Qwen3 / Qwen3-VL). Compact on purpose: nudity, sex and
  // the vulgar body words, in the languages a creator here is likely to paste from. Accents are already folded.
  {
    pattern: wordAlternation([
      String.raw`desnud\w*|encuerad[oa]s?|en pelotas|en cueros|en bolas(?! de\b)|sin ropa|sin bragas|sin calzones|senos al aire|tetas|tetitas|chichis|pezon(?:es)?|conyo|sexo|follando|follar|follad[oa]|pene|mamada|verga`,
      String.raw`nuas?|nuazinh\w*|pelad[ao]s?|peladinh\w*|nudez|sem roupas?|sem calcinha|mamilos?|transando|transar|buceta|boceta|xoxota|piroca`,
      String.raw`nues?|toute? nues?|tout nu|tous nus|seins nus|fesses nues|sans (?:vetements|habits)|photo(?:graphie)? de nu|nu (?:integral|artistique)|nudit\w*|denudees?|a poil|deshabillees?|devetues?|fellation`,
      String.raw`\w+nackt\w*|\w+naakt\w*|semidesnud\w*|semi ?nud[aoe]s?|semi ?nu[ae]s?\b|polnag[aiy]e?|nackt\w*|nackig\w*|unbekleidet\w*|entbl(?:o|oe)s+t\w*|barbusig\w*|oben ohne|geschlechtsverkehr|bumsen|akt(?:foto|bild|fotografie|modell|aufnahme)\w*|ohne (?:kleidung|klamotten)|entkleidet|brustwarzen?|nippel|titten|muschi|fick(?:en|t)|gefickt`,
      String.raw`nud[aio]|svestit[aeio]|spogliat\w*|spogliars\w*|senza vestiti|seno nudo|seni nudi|tette|capezzoli|scopa(?:re|ndo)|sesso|pompin\w*|cazzo`,
      String.raw`naakt\w*|bloot|blote (?:borsten|billen|vrouw|lichaam|kont)|tepels?|uitgekleed\w*|uitkleden|zonder (?:kleren|kleding)|geil|kut|tieten|neuken|\w*telanjang\w*|bugil|toket|ngentot|bersetubuh|puting susu|memek|kontol|hubad|kantot|khoa than|lo nguc|nguc tran|loa the|lam tinh|\w*ciplak\w*|nahota|nahot\w*|nahy|nahat\w*|naken\w*|naket|nakna|nakne|nogen|nogne|nogenbilleder?|nogenbad|alast(?:on|i|oma\w*|onkuv\w*)|nakuna|nagos\w*`,
      String.raw`hadaka|zenra|ecchi|etchi|sekkusu|chikubi|o?manko|chinpo|chinko|ochinchin|luoti|chiluo|seqing|xingai|zuoai|nudeu|almom|seksu|nyuu?do`,
      // Polish, Czech, Slovak, Croatian, Turkish, Nordic, Finnish, Romanian, Hungarian, Catalan. A word that is also an
      // ordinary word elsewhere ("naga" the serpent people, "Naha" the city, "gola" a Portuguese collar, German "nahe")
      // counts only next to a word of its own language that names a woman, a body, a photo or "completely".
      String.raw`(?:naga|nagi|nagie|nago|nagu|gola|goly|gole|goli|naha|nahe|nahi)(?= (?:${SLAVIC_CORROBORATION})\b)|(?<=\b(?:${SLAVIC_CORROBORATION}) )(?:naga|nagi|nagie|nago|nagu|gola|goly|gole|goli|naha|nahe|nahi)`,
      String.raw`kuk|nud|ngono|naket\w*|brystvort\w*|cycki|cycuszki|sutki|sutek|rozebran\w*|rozbier\w*|bez (?:ubrania|ubran\w*|ciuchow)|rucha(?:c|nie|m|ja)|cipk\w*|kutas\w*|bradavk\w*|soulozi\w*|sukat|sukani|svlecen\w*|svlekl\w*|bez obleceni|meme ucu\w*|sikis\w*|soyunmus\w*|soyunuk|kiyafetsiz|brostvart\w*|knulla\w*|knulle\w*|kneppe\w*|fitta|fisse|pikk|avkladd|avkledd|utan klader|uten klaer|nannit|pillu`,
      String.raw`goala|goale|dezbracat\w*|sfarcuri|sfarc|pizd\w*|futut\w*|fara haine|meztelen\w*|pucer\w*|mellbimbo\w*|szex\w*|ruha nelkul|jeban\w*|sise|despullad\w*|sense roba`
    ])
  },
  // Garments cut to expose the genitals, nipples or buttocks. The opening is the garment's purpose, so whatever it is
  // worn with, the cut shows what it exposes: "open-crotch panties" and "crotchless panties" are the same picture,
  // and the shipped adult seed's own outfits carry the former. A "built-in shelf bra" is a camisole's lining.
  {
    pattern: wordAlternation([
      String.raw`open ?crotch(?:ed)?|open ?(?:cup(?:s|ped)?|bust(?:ed)?|nipples?|breasts?|butt|ass|bottom(?:ed)?) ?(?:panties|panty|knickers|briefs|thongs?|bras?|bodysuits?|leotards?|teddy|teddies|corsets?|bustiers?|lingerie|tights|pantyhose|chaps|shorts|hot pants|pants|skirts?|swimsuits?|bikinis?|catsuits?|leggings)`,
      String.raw`crotch ?(?:cutouts?|holes?|openings?|slits?|zip(?:per)?s?|less|open|free)|open (?:gusset|bum|seat|bottom|back) (?:panties|knickers|briefs|tights|pantyhose|bodystocking|thong)s?|gusset ?less|peep ?holes? bras?|quarter ?cup bras?|(?:breast|nipple|bust|crotch|butt|bum) ?baring|(?:split|slit|zip(?:per)?) crotch|ouverte?s? (?:panties|knickers|bras?|briefs|lingerie|tights|pantyhose|bod(?:y|ies)|bodysuits?|teddy|culottes?|strings?|slips?)|(?:panties|knickers|bras?|briefs|lingerie|tights|pantyhose|culottes?|slips?|strings?) ouverte?s?|cup ?less|nipple ?(?:less|cutouts?|holes?|windows?)|peek ?a ?boo bras?|(?<!built in )shelf bras?|assless|butt ?less|bum ?less|cheekless|cutout (?:crotch|nipples?)`
    ])
  },
  // Nudity said every other way, and the removal of whatever still covered it. Underwear is named on its own: "no
  // panties" was read while "without panties" was not, and taking off a bra leaves a topless woman however it is said.
  {
    pattern: wordAlternation([
      String.raw`in the altogether|in the raw(?! (?:state|form|materials?|data|footage|format|files?|images?|photos?|version|edit|edges?|bar|food|vegan))|starkers|in the nudd?y|nud(?:e|d)?y|unattired|undraped(?! (?:windows?|tables?|walls?|furniture|beds?|sofas?|couch(?:es)?))|skyclad|sky clad`,
      String.raw`(?:sans|free of|devoid of|minus|bereft of|stripped of|out of|with no|without (?:a |any |her |his |their |a single |one |even a )?|not (?:a |one )?(?:single )?(?:piece|scrap|stitch|thread|shred) of|not wearing (?:a )?(?:single )?(?:piece of )?) ?(?:her |his |their |all |any |a single |one |every )?(?:clothes|clothing|garments?|attire|apparel|outfits?|stitch(?: of clothing)?|scrap of clothing|thread of clothing|piece of clothing)`,
      String.raw`(?:wears?|wearing|wore|worn|clothed in|dressed in|clad in|has on|had on) (?:absolutely |literally |simply |completely |totally |just |utterly )?nothing(?=$| (?:at all|whatsoever|else|and|but skin|in|while|as|to|under|underneath|beneath|on(?! (?:her|his|their) (?:feet|head|face|hands?|wrists?|fingers?|neck|ears|mind)\b)|except|apart|besides|other)\b)|(?:in|wearing) (?:absolutely )?nothing at all`,
      String.raw`(?:with|has|had|having|wearing|is wearing|wears) nothing on (?:except|but|apart from|besides|other than|underneath|under|beneath)|with nothing on underneath|nothing (?:on )?(?:underneath|under|beneath) (?:her|his|their|the) (?:dress|skirt|coat|robe|bathrobe|towel|clothes|shirt|kimono|yukata|trench ?coat|gown|jacket|blanket|sheets?)|(?:wearing |with |has |had )?nothing (?:on )?underneath$`,
      String.raw`nothing (?:on )?(?:below|under|beneath|from) the (?:waist|hips|belt)(?: down)?|(?:bare|naked|nude|undressed|unclothed|stripped) (?:from the (?:waist|hips) down|below the (?:waist|hips))|nothing on (?:the |her |his |their )?(?:bottom|lower half)`,
      String.raw`(?:stripped|strips|stripping|strip) (?:bare|naked|nude|to the skin|down to nothing|to nothing|off everything)|down to (?:nothing|her skin|the skin|bare skin|her birthday suit)|(?:pants|trouser|skirt|knicker|undie|underwear|panty)less|knickerless|(?:going|goes|went|gone|go) commando|commando (?:under|beneath|style)`,
      String.raw`(?:no|without|sans|minus|forgot (?:her|his|to wear)|not wearing|isn't wearing|is not wearing|wasn't wearing|never wears|skipped|skipping|ditched|ditching|lost (?:her|his)) (?:any |her |his |their )?(?:panties|panty|knickers|undies|underwear|underpants|briefs|bottoms|g ?strings?|thongs?)|(?:no|without|sans) (?:[a-z]+ ){0,3}(?:or|nor|and) (?:any |her |his |their )?(?:panties|panty|knickers|undies|underwear|underpants|briefs)`,
      String.raw`(?:took|takes|take|taking|taken|removed?|removes|removing|stripped|stripping|shed|sheds|shedding|peeled|peels|peeling|pulled|pulls|pulling|tugged|tugs|tugging|yanked|yanks|slipped|slips|slipping|dropped|drops|dropping|lowered|lowers|lowering|unhook(?:s|ed|ing)?|unclasp(?:s|ed|ing)?|undo(?:es|ing)?|undid|unzip(?:s|ped|ping)?) (?:off |down |out of |away )?(?:her |his |their |the |my |your )?(?:bra|brassiere|bralette|panties|panty|knickers|underwear|undies|underpants|briefs|thong|g ?string|bikini (?:top|bottoms?)|swimsuit top|lingerie)(?! (?:straps?|band|hooks?|tag|size|line)\b)(?: off| down)?`,
      String.raw`(?:bra|brassiere|bralette|panties|panty|knickers|underwear|undies|underpants|briefs|thong|g ?string|bikini (?:top|bottoms?)|lingerie|skirt and panties|pants and panties) (?:off(?! white)|removed|discarded|unhooked|undone|on the floor|around (?:her|his|their) (?:ankles|knees|thighs|feet)|at (?:her|his|their) (?:ankles|knees)|(?:pulled|pushed|slid|yanked) (?:down|off|aside))`,
      String.raw`(?:bare|naked|nude) as the day (?:she|he|they|i|you) (?:was|were) born|as nature intended|^(?:completely|fully|totally|entirely|all) bare$|(?:she|he|they|i) (?:is|was|are|were|stands?|stood|sits?|sat|lies|lay|poses?|posed) (?:completely|fully|totally|entirely|all) bare|(?:completely|fully|totally|entirely) bare (?:body|bodies|skin|woman|women|girls?|model)|bare from head to (?:toe|foot)|bar(?:ing|es|ed) (?:it )?all(?=$| (?:for|to|on|in front of|before) (?:the |a )?(?:camera|lens|viewer|photographer|photo|shoot|world)\b)|bare buns|centerfolds?|centrefolds?|playboy playmates?|playmate (?:centerfold|of the (?:month|year))|skin mags?(?: models?)?|figure study models?|birth of venus|moan(?:s|ing|ed)? (?:in (?:pleasure|ecstasy|bed)|(?:his|her|their) name)|(?:on top of|straddl(?:es|ed|ing)?|mount(?:s|ed|ing)?) (?:him|her)(?= (?:in|on) (?:the |a |their |his |her )?bed\b)`,
      String.raw`(?:state|states) of undress|clothing ?optional|fkk|freikorperkultur|page (?:3|three) (?:girls?|models?|stunnas?)|lady godiva|odalisques?|nothing but (?:a smile|her skin|his skin|skin|her birthday suit)|(?:butt|ass|bum) ?cracks?|mooning(?! (?:over|about|around)\b)`,
      String.raw`(?:bare|bared|naked|nude|exposed|uncovered|free|freed|unbound) (?:her |his |their )?(?:bosoms?|bust|busts)|(?:bosoms?|bust|boobs?|breasts?|tits) (?:exposed|bared|bare|uncovered|free|freed|unbound|out|hanging out|spilling out)|(?:her|his|their|the) body (?:is )?(?:fully |completely |totally |entirely )?(?:uncovered|exposed|bared)|body (?:fully|completely|totally|entirely) (?:uncovered|exposed|bared|bare)|nothing covering (?:her|his|their) (?:body|breasts|crotch|privates)|expos(?:es|ed|ing) (?:her|him|them)sel(?:f|ves)|flash(?:es|ed|ing) (?:the camera|the viewer|the lens|everyone|passers ?by|the crowd)`,
      String.raw`(?:covering|hiding|concealing|shielding|cupping) (?:her|their) (?:bare )?(?:breasts|nipples|crotch)|(?:covering|hiding|concealing|shielding|cupping) (?:her|their) bare (?:chest|bosom)|(?:hiding|concealing|shielding|cupping) (?:her|their) (?:chest|bosom)|(?:hands?|forearms?|arms?) (?:covering|hiding|shielding|cupping) (?:her|their) (?:chest|bosom)|(?:hands?|arms?|hair|a towel|a pillow) (?:covering|over|across|hiding) (?:her|their) (?:bare )?(?:breasts|nipples|crotch)|(?:hands?|arms?|hair|a towel|a pillow) (?:covering|over|across|hiding) (?:her|their) bare (?:chest|bosom)|only (?:a |her |the |their )?(?:sheet|bedsheet|towel|blanket|pillow|hands?|arms?|hair) (?:covering|over|hiding) (?:her|him|them)`
    ])
  },
  // Acts written as a paraphrase, a participle or slang, and the rating words that announce the rest. Each verb is
  // held to the object or ending that makes it the act: "riding" a bike, "banging" her head on a shelf, "going down"
  // the stairs and a "penetrating gaze" stay ordinary.
  {
    pattern: wordAlternation([
      String.raw`(?:rides|rode|ridden|straddl(?:es|ed|ing)) (?:him|her (?:boyfriend|husband|lover|partner))(?!'s|')|\b(?:sleeping|slept|sleeps) with (?:him|her)(?= ?$)|knocking boots|doing the deed|rolling in the hay|(?:her|a) hand (?:in|inside|down) (?:his|her) (?:pants|trousers|jeans|shorts|underwear|panties|boxers)(?! ?(?:back |front )?pockets?\b)|sucking (?:his|him) (?:member|manhood|shaft|cock|dick)|riding (?:him|his (?:cock|dick|penis|shaft|lap|face|thigh)|(?:her |the |a )?(?:boyfriend|husband|lover|partner|man|guy|dude|cock|dick|stranger))(?!'s|')|grind(?:s|ing|ed)? (?:on|against|into) (?:(?:him|her)(?! (?:skateboard|board|bike|rail|teeth))|(?:his|her|a man's|her partner's) (?:lap|crotch|groin|thigh|hips|body))`,
      String.raw`legs? (?:wrapped|locked|hooked) (?:around|round) (?:his|her|their) (?:waist|hips)|(?:hands?|fingers?|head|face|mouth|tongue) (?:between|in between) (?:her|his|their) (?:legs|thighs)|(?:hands?|fingers?) (?:down|inside|into|under|up) (?:her|his|their) (?:pants|panties|underwear|knickers|skirt|shorts|trousers|jeans|dress)(?! pockets?)`,
      String.raw`(?:fingers?|tongue|cock|dick|penis|dildo|vibrator|toy) (?:deep )?(?:inside|into|up inside) (?:her|him)(?=$| (?:pussy|vagina|ass|while|and|as|hard|deep|slowly)\b)|fingered(?=$| by\b| (?:her|him)sel(?:f|ves)\b| (?:her|him)(?:$| (?:while|and|hard|deep|deeply|slowly|until)\b))|fingering (?:her|him)sel(?:f|ves)|fingering (?:her|him)(?=$| (?:while|and|hard|deep|deeply|slowly|until)\b)`,
      String.raw`(?:eat|eats|eating|eaten|ate) (?:her|him|them|me|you) out|(?:go|goes|going|gone|went) down on (?:him|her|them|me|you)(?! (?:knees?|luck|one knee|bended knee|all fours|hands)\b)|suck(?:s|ed|ing)? (?:him|her|them|me|you) off|suck(?:s|ed|ing)? (?:his|her|their) (?:cock|dick|penis|nipples?|breasts?|tits)`,
      String.raw`(?:give|gives|giving|given|gave|got|get|gets|getting) (?:him |her |them |a man |a guy )?head(?=$| (?:in|on|while|and|to|for)\b)|head ?jobs?|bjs?(?! (?:wholesale|novak|thomas|penn|'s)\b)|blowies?|doggy (?:position|pose)|(?:get|gets|getting|got|gotten) laid|carnal(?:ly)?|bonking|boinking|boinks?|nookie|nooky|booty ?calls?|quickies?|throes of passion|beast with two backs`,
      String.raw`bang(?:s|ed|ing)? (?:him|her|them|me|you|each other)(?=$| (?:hard|from behind|in bed|on the bed|while|and|all night)\b)|screwing(?=$| (?:him|her|them|each other|me|you|around)\b)|screwed (?:him|her)(?=$| (?:hard|while|and|in bed)\b)|(?:intimate|sexual|carnal) relations|(?<!(?:dog|cat|horse|cattle|animal|plant|selective|captive|cross|line|pure|in|pigeon|rabbit|bird|fish|seed) )breeding(?! (?:dogs?|cats?|horses?|cattle|programs?|grounds?|season|pairs?|stock|farms?|kennels?|rabbits?|chickens?|birds?|fish|plants?|standards?|clubs?|projects?)\b)|bred (?:her|him|by)`,
      String.raw`insemina\w*|impregnat\w*|consummat(?:e|es|ed|ing) (?:the|their|our) (?:marriage|union|relationship)|deflower\w*|(?:losing|lose|loses|lost|took|taking|take|takes) (?:her|his|their) virginity|69ing|tribb(?:ing|ed|s)|against (?:her|his|their|my|your) will|(?:taken|took|takes|taking) (?:her |him )?(?:by force|forcibly)|forc(?:ed|es|ing) (?:himself|herself|themselves) (?:on|upon)`,
      String.raw`(?:being |getting |was |is |gets |got )penetrated|penetrat(?:ed|es|e) (?:her|him|them|me|you)\b|(?<!(?:a|an|the|her|his|their|its) )penetrating (?:her|him|them|me|you)\b|penetrated$`,
      String.raw`explicit (?:content|scenes?|photos?|images?|pictures?|material|nudity|sex|version|imagery|shots?|depictions?|videos?|art)|(?:18|21) ?\+(?: (?:content|only|rated|material|version|photos?|images?))?|(?:adults?|18 ?\+|grown ?ups?) only|r ?rated|rated (?:r|x|18|r18)|x ?rated|nc ?17|smut(?:ty)?|obscen(?:e|ity)|indecent (?:exposure|photos?|acts?|images?)|pornograph\w*|bodily fluids|money ?shots?`,
      // Spellings that dodge the plain word, and closed compounds with an obscenity inside them.
      String.raw`[a-z]*f+u+c+k+[a-z]*|(?:ph|f)(?:u|v)c?k(?:s|ed|er|ers|ing|in|kin|king)?|fcuk\w*|fck(?:s|ed|er|ing|in)?|phuck\w*|fvck\w*|fuq\w*|pron|phorn\w*|seks\w*|segs|smex(?:y|ing)?|sexx+|secks|s3x|facial abuse|in flagrante(?: delicto)?|ravished|(?:cum|jizz|spunk)(?:slut|sluts|dump|dumpster|bucket|play|swap|covered|drenched|filled|guzzler|whore|rag|load|sock|stained|stains?|shot|tribute|hole)s?`,
      String.raw`titt(?:y|ys|ie|ies|iez)|titz|tiddies|tiddys|ta ?tas|boob(?:z|iez|ys)|nip(?:z|pz|pies)|pu(?:ss|zz|s)(?:i|ee|eh|ie)|puzzy|vaj(?:ayjay|ay jay|j|ay)?|cooch(?:ie|y)?|funbags|beaver shots?|(?<!wood)pecker|family jewels|(?:nut|ball) ?sacks?|butt ?holes?|member (?:erect|hard)|erect member|(?:his|throbbing|erect|hard) manhood|(?:wet|soaked|soaking|dripping|damp) (?:panties|knickers|underwear|crotch|pussy)|load (?:on|in) (?:her|his) (?:face|mouth|chest|tits|breasts)|after ?sex`
    ])
  },
  // A guard keeps "nude lipstick" and "naked eye" ordinary because of the one word next to it; it was never meant to
  // keep "a fully nude colour photograph of her" or "the naked truth of her body". An intensifier makes the word
  // nudity whatever follows, and a colour or idiom guard does not hold once the term names a body or a photograph.
  // (The bird names are read with their source: GUARDED_BODY_WORD_PATTERN.)
  {
    pattern: wordAlternation([
      String.raw`(?:fully|completely|totally|entirely|utterly|all|mostly|partly|partially|nearly|almost|half|semi) (?:nude[sd]?|naked)`,
      String.raw`nude[sd]?(?: (?:colou?r(?:ed)?|tone[sd]?|toned|shade[sd]?|beige))? (?:bod(?:y|ies)|figures?|torsos?|photo(?:graph)?s?|portraits?|selfies?|pictures?|pics?|shots?|images?|lying|in bed|on (?:the |a )?bed|wom[ae]n|girls?|lad(?:y|ies)|models?|persons?|people|couples?)`,
      String.raw`naked (?:eye|truth|flames?|juice|cakes?|grapes?)\b.*\b(?:her|his|their) (?:body|bodies|breasts|chest|torso|figure|curves)|naked eye ?level|naked eye (?:view|shot|close ?up|photo|portrait|selfie)`
    ])
  },
  // `sex` covers group sex, after sex, sex from behind, standing sex, implied sex. Same-sex and opposite-sex are
  // descriptions of a couple, not of an act.
  { pattern: /(?<!(?:same|opposite) )\bsex\b/u },
  // `undressed` alone is fully undressed; `partially undressed` is the suggestive tag of the same family.
  { pattern: /(?<!partially )\bundressed\b/u },
  // "nothing on" ends a clause or follows the verb that wears it; "nothing on her mind" does not undress anyone.
  { pattern: /\b(?:(?:with|has|had|having) nothing on$|(?:wears|wearing|wore|is wearing|there is|there's) nothing on\b)/u },
  // Whole-tag only: NovelAI reads a bare `nothing` as nudity (imageOrchestrator.assertsFullNudity does too), and
  // these Danbooru tags mean their explicit sense only standing alone ("presenting at a meeting", a medical x-ray, a
  // cake's cross-section, a tally counter, milking a cow are not).
  {
    pattern:
      /^(?:nothing|nothing on|nothing else|nothing else on|69|69 position|sixty ?nine|explicit|facial|18\+|presenting|m legs|tally|tally marks?|milking|x ?ray|cross ?section|privates|doggy|o ?face|moaning|banging|humping)$/u
  }
];

// Activities that imply a nude person even when no nudity tag is written. A photo model renders "a woman
// showering" as exactly that. Only activities, not places: an empty onsen is scenery, and "a hot spring in Hakone"
// names the place. Being IN the water is the activity, whatever the determiner or adjective in between ("soaking in
// a bathtub", "relaxing in her bath", "in an outdoor onsen") — the rule once knew only "in the bath" and missed the
// rest. "A weekend in Bath" locks too; that costs a NovelAI render.
const BATHING_PLACE = String.raw`(?:bath ?tubs?|tubs?(?! (?:of|chairs?)\b)|baths?(?! (?:towels?|mats?|robes?|bombs?|salts?|toys?|products?|time|tiles?|sheets?|caddy|caddies|pillows?|brushes|museums?|ruins|site|complex|exhibits?)\b)|bubble ?baths?|(?<!(?:rain|light|april|spring|summer|scattered|heavy|passing|meteor|baby|bridal|wedding) )showers?(?! (?:curtains?|caps?|heads?|gel|rooms?|stalls?|doors?|screens?|tiles?|caddy|caddies|rails?|trays?|of)\b)|hot ?springs?|onsens?|rotenburo|saunas?|jacuzzis?|hot ?tubs?|whirlpools?|steam ?rooms?|bath ?houses?|sentos?|thermal (?:baths?|pools?|springs?)|spa (?:pools?|baths?))`;
const IMPLIED_NUDITY_RULE: TermRule = {
  pattern: new RegExp(
    String.raw`\b(?:bathing(?! suit)|showering|(?:taking|takes|took|having|has|had|draws?|drew|enjoy(?:s|ed|ing)?) (?:a |her |his |their )?(?:long |hot |warm |relaxing |quick |bubble )?(?:bath|shower)|mixed bathing|(?:in|into|inside|out of|from) (?:a |an |the |her |his |their |my |your |its |this |that |one |some )?(?:[a-z]+ ){0,2}${BATHING_PLACE}|(?:soak|soaks|soaking|soaked|bathe|bathes|bathed) (?:in|into) (?:a |an |the |her |his |their )?(?:[a-z]+ ){0,2}(?:water|pool of water|tub)|under (?:a |an |the |her |his |their )?(?:running |hot |warm |cold )?(?:shower|shower ?head|stream of (?:hot |warm )?water)|(?:rins|wash|scrub|lather|soap)(?:es|ed|ing|s)? (?:off |down )?(?:her|his|their) (?:naked |bare )?(?:body|breasts|chest|torso|whole body))\b`,
    "u"
  )
};

// "wearing only a smile", "dressed only in jewelry", "nothing but stockings": nude, with an accessory. What the
// garments in "only X" leave bare decides the rest (readOnlyWearing): "wearing only an oversized sweater" or "only a
// bra and panties" cover both halves and are suggestive; "wearing only a bra" or "only a crop top" leave her
// bottomless, which is explicit; "wearing only jeans" or "only panties" leave the chest bare, which is a bare chest
// judged by gender like any other. A negated garment ("just a necklace and no skirt") covers nothing. Without a verb
// only the idiom "nothing but" and a whole-term "only/just X" count, and only before something worn, so "nothing but
// trouble" and "just coffee" stay ordinary.
const ONLY_WEARING_PATTERN = new RegExp(
  [
    String.raw`\b(?:(?:wearing|wears|wore|worn|dressed in|clad in|clothed in|has on|had on|having on) (?:only|just|solely|merely|simply|nothing but|nothing except|nothing other than|nothing apart from|nothing besides|nothing save)`,
    String.raw`in nothing (?:but|except|other than|apart from|besides|save)`,
    String.raw`(?:only|just|solely|merely) (?:wearing|wears|wore|dressed in|clad in|clothed in)`,
    String.raw`(?:clad|dressed|clothed) (?:only|just|solely|merely) in`,
    String.raw`all (?:she|he|they) (?:wears?|wore|is wearing|was wearing|are wearing|has on|had on|have on) (?:is|was|are|were)`,
    String.raw`the only (?:thing|things|garment|garments|item|items|piece of clothing|clothing) (?:she|he|they) (?:wears?|wore|is wearing|was wearing|are wearing|has on|had on|have on)(?: (?:is|was|are|were))?`,
    String.raw`(?:her|his|their) (?:only|sole) (?:clothing|clothes|garments?|attire|outfit|covering|cover|item of clothing|piece of clothing) (?:is|was|are|were|being)`,
    String.raw`covered (?:only|just|solely|merely) (?:by|with|in)`,
    String.raw`(?:with|has|had|having|wearing|is wearing) nothing on (?:but|except|apart from|besides|other than) (?:for )?) (.+)$`
  ].join("|"),
  "u"
);
const WITH_ONLY_ON_PATTERN = /\bwith (?:only|just) (.+?) on$/u;
const BARE_ONLY_PATTERN = /\bnothing (?:on )?(?:but|except|save|besides|apart from|other than) (?:for )?(?:an? |her |his |their |a pair of |some )?((?:[a-z']+ ){0,3}[a-z']+)$/u;
const WHOLE_ONLY_PATTERN =
  /^(?:only|just|solely) (?:an? |her |his |their |a pair of |some )?((?:[a-z']+ ){0,3}[a-z']+)$|^(?:an? |her |his |their )?((?:[a-z']+ ){0,3}[a-z']+) only$/u;
const WORN_ACCESSORY_PATTERN =
  /\b(?:heels?|high heels|shoes|boots|sneakers|socks|stockings|thigh ?highs?|tights|pantyhose|garters?|gloves|hats?|caps?|necklaces?|jewel(?:le)?ry|earrings?|bracelets?|rings?|chokers?|collars?|ribbons?|bows?|aprons?|towels?|blankets?|sheets?|lipstick|makeup|perfume|body paint|glasses|ties?|scarf|scarves|veils?|tattoos?|anklets?|pearls|chains?|harness(?:es)?|belts?|smile|grin|skin|watch|piercings?)$/u;
// What a garment covers, for readOnlyWearing. A long top (shirt, sweater, hoodie) reaches the hips and is read as
// covering both halves, as the oversized-sweater case always was; a short one (bra, crop top, tank top) does not.
const FULL_COVER_GARMENT_PATTERN =
  /\b(?:clothes|clothing|clothed|outfits?|attire|apparel|garments?|dress(?:es)?|sundress(?:es)?|gowns?|robes?|bathrobes?|kimonos?|yukatas?|hanbok|jumpsuits?|overalls|onesies?|coats?|raincoats?|cloaks?|ponchos?|bodysuits?|leotards?|catsuits?|unitards?|swim ?suits?|swimwear|bathing suits?|one pieces?|wetsuits?|uniforms?|pajamas|pyjamas|nightgowns?|nightdress(?:es)?|nighties?|slip dress(?:es)?|chemises?|babydolls?|negligees?|suits?|lingerie|underwear|undergarments?|bikinis?|bra and panties|matching set)$/u;
const LONG_TOP_GARMENT_PATTERN =
  /\b(?:shirts?|t ?shirts?|tees?|button (?:downs?|ups?)|sweaters?|jumpers?|hoodies?|sweatshirts?|cardigans?|blouses?|jerseys?|nightshirts?|tunics?|jackets?|blazers?|parkas?|fleeces?)$/u;
// A long top cut short stops at the ribs: "a cropped tee", "a baby tee", "a tiny cardigan" cover the chest only.
const SHORTENED_TOP_PATTERN = /\b(?:crop(?:ped)?|short|shortened|tiny|baby|mini|micro|cut ?off|shrunken|half|bolero)\b/u;
const SHORT_TOP_GARMENT_PATTERN =
  /\b(?:bras?|brassieres?|bralettes?|bikini tops?|crop tops?|tube tops?|tank tops?|halter tops?|camisoles?|camis?|sports bras?|corsets?|bustiers?|tops?|vests?|pasties)$/u;
const LOWER_GARMENT_PATTERN =
  /\b(?:panties|panty|knickers|briefs|boxers|boxer briefs|underpants|thongs?|g ?strings?|bikini bottoms?|jeans|pants|trousers|shorts|hot ?pants|skirts?|mini ?skirts?|leggings|sweatpants|joggers|slacks|chinos|culottes|sarongs?|bottoms)$/u;

// Posture words that are also sex-position tags in stateMemory's lexicon but have ordinary readings (lying on
// one's back asleep, riding a horse, a lap pillow, a wrestling hold). Alone they are not explicit; next to a
// suggestive tag they are. "lying on her back" is matched anywhere, so the same posture written as prose (an
// appearance, a finished photo prompt) counts like the bare tag does.
const AMBIGUOUS_POSTURE_RULE: TermRule = {
  pattern:
    /^(?:lying )?on (?:her |his |their )?(?:back|stomach)$|^(?:cowgirl|afterglow)$|\blying on (?:her|his|their) (?:back|stomach)\b|\b(?:on all fours|all fours|spread legs|legs spread|spread thighs|legs apart|bent over|bending over|legs up|legs held open|lap pillow|spooning|straddling|girl on top|boy on top|lap sitting|sitting on (?:another's |someone's )?lap|grinding|leg lock|full nelson|piledriver|thrusting|kneeling over|top down bottom up|jack o'? challenge)\b|(?<!horseback )\briding\b(?! (?:a |the |her |his )?(?:horses?|pony|ponies|bicycles?|bikes?|motorcycles?|motorbikes?|scooters?|trains?|bus|buses|cars?|vehicles?|skateboards?|surfboards?|waves?|elevators?|escalators?|brooms?|jet ?skis?|waverunners?|camels?|elephants?|roller ?coasters?|ferris wheels?|subway|trams?|ski lifts?|chairlifts?|boats?|ferr(?:y|ies)))/u
};

// Wardrobe and framing that sexualise a cut without being explicit. Adults may render photoreal with these;
// combined with a school-coded subject or an ambiguous posture they may not.
const SUGGESTIVE_RULE: TermRule = {
  pattern: wordAlternation([
    String.raw`lingerie|underwear|bras?|brassiere|bralettes?|panties|panty|pantsu|thong|g ?strings?|cleavage(?: cutout)?|swim ?suits?|swim ?wear|bathing suits?|bikinis?`,
    String.raw`see ?through|translucent|transparent (?:clothes|clothing|dress|shirt|blouse|top|bodysuit|leotard|lingerie|robe|nightgown)|sheer (?:clothes|clothing|dress|shirt|blouse|top|bodysuit|fabric|lingerie|robe|nightgown)|wet clothes|wet shirt`,
    String.raw`sideboob|underboob|garter belts?|garter straps|negligee|babydoll|towel only|wrapped in (?:a )?towel|no bra|braless`,
    String.raw`bedroom eyes|seductive|sexy|sexually suggestive|lewd|erotic|erotica|naughty face|come hither|boudoir|dominatrix|spanking|aroused|arousal|horny|camel ?toe`,
    String.raw`panty ?shot|pantyshot|upskirt|downblouse|skirt lift|lifted by self|partially undressed|undressing|(?:shirt|clothes|dress|top|sweater|pantyhose|skirt) (?:lift|pull)`,
    String.raw`open shirt|shirt open|open blouse|unbuttoned (?:shirt|blouse)|strap slip|cameltoe|covered nipples|torn (?:clothes|clothing|shirt|dress|skirt|pantyhose|stockings|leggings|jeans|top|blouse)`,
    String.raw`playboy bunny|bunny suit|bunny girls?|ass focus|butt focus|crotch focus|breast focus|bulge|crotch(?: seam)?|groin|body writing|ass visible through thighs`,
    String.raw`peeping|hidden cameras?|lap ?danc(?:e|es|ing)|pole danc(?:e|er|ers|ing)|go ?go dancers?|rating:\s*(?:questionable|sensitive)`,
    String.raw`lenceria|ropa interior|roupa intima|sous vetements|unterwasche|seins|seios|peitos|bruste|busen|payudara`,
    String.raw`borsten|piersi|prsa|gogus\w*|memeler|brost|bryster|rinnat|mellek|naizi|rufang|erotik\w*|erotyczn\w*|sin sosten|dishabille|deshabille`,
    String.raw`scantily (?:clad|dressed)|skimpily (?:clad|dressed)|barely (?:clothed|dressed|covered)|half ?dressed|tan ?lines|nothing left to the imagination|one night stands?|(?:breast|boob|cleavage|chest|bust) (?:windows?|cutouts?|keyholes?)|cam ?girls?|webcam (?:model|girl|show)s?|call girls?|prostitut\w*|hookers?|brothels?|escort (?:girl|service|agency)s?|sluts?|slutty|whores?|nymphos?|nymphomaniac\w*|exotic dancers?`
  ])
};

/**
 * Garments that cover the body, as the HEAD noun of a term ("white shirt", not "shirt collar"). A negative NovelAI
 * weight on one of them (`-2::clothing::`, `-1::apparel::`, `-1::shirt, skirt::`) is NovelAI's undressing idiom;
 * the classifier reads it as undressed, and the photo builder never turns one into a photo negative, which would
 * steer a photo model the same way. Accessories (hat, glasses, shoes, a jacket over a shirt) are not on the list:
 * "-1::hat::" only means no hat.
 */
const COVERING_GARMENT_PATTERN =
  /\b(?:clothes|clothing|clothed|outfits?|attire|apparel|wardrobe|garments?|fabric|cloth|textiles?|dressed|shirts?|t shirts?|tops?|blouses?|sweaters?|hoodies?|skirts?|pants|trousers|jeans|shorts|dress(?:es)?|sundress(?:es)?|gowns?|uniforms?|kimonos?|yukatas?|robes?|bathrobes?|pajamas|pyjamas|jumpsuits?|bras?|panties|underwear|lingerie|swim ?suits?|swimwear|bikinis?|bodysuits?|leotards?)$/u;

/**
 * Words that keep a picture covered, censored or safe. Weighted down on NovelAI (`-1::sfw::`, `-2::censored::`) they
 * ask for the opposite; as a photo negative they steer the sampler there at any CFG above 1.
 */
const SAFEGUARD_WORD_PATTERN =
  /\b(?:sfw|safe for work|family friendly|censor(?:ed|ing|ship)?|mosaic(?: censoring)?|bar censor|modest(?:y|ly)?|decent|decency|covered|covering|coverage|cover up|fully clothed|conservative (?:clothing|outfit|dress)|pg(?: 13)?|rating:?\s*(?:safe|general))\b/u;

/**
 * Words that make a figure an adult. Weighted down on NovelAI (`-1::adult::`, `-1::mature female::`) they are its
 * age-down idiom; as a photo negative they steer the sampler toward a younger subject.
 */
const ADULTHOOD_WORD_PATTERN =
  /\b(?:adults?|adult (?:woman|man|female|male|body|face|figure|proportions)|mature(?: (?:female|male|woman|man|body|face|figure|lady))?|grown(?: up| woman| man)?|grownup|old(?:er)?(?: (?:woman|man|lady|person|people|face|skin))?|aged|ageing|aging|elderly|middle aged|milf|wom[ae]n|lad(?:y|ies)|gentlem[ae]n|mothers?|wi(?:fe|ves)|wrinkles?|wrinkled(?: skin)?|crow's feet)\b/u;

/** Body words that read as an adult build; a photo negative of them steers toward a child's. Not a lock on their own. */
const ADULT_BUILD_WORD_PATTERN =
  /\b(?:tall|curvy|busty|voluptuous|thick thighs|wide hips|hourglass(?: figure)?|full figured?|(?:large|big|huge|giant|gigantic|medium|full|ample) (?:breasts|boobs|bust|chest)|broad shoulders)\b/u;

/** Anything worn on the body or wrapped around it. A photo negative of any of them undresses whoever is in frame. */
const WORN_CLOTHING_PATTERN =
  /\b(?:clothes|clothing|clothed|outfits?|attire|apparel|garments?|wardrobe|fabric|cloth|textiles?|costumes?|uniforms?|shirts?|t shirts?|tops?|blouses?|sweaters?|sweatshirts?|hoodies?|cardigans?|jackets?|coats?|blazers?|vests?|tunics?|camisoles?|corsets?|skirts?|pants|trousers|jeans|shorts|leggings|tights|stockings|pantyhose|thigh ?highs?|socks|dress(?:es)?|sundress(?:es)?|gowns?|nightdress(?:es)?|nightgowns?|nightwear|sleepwear|pajamas|pyjamas|robes?|bathrobes?|kimonos?|yukatas?|hanbok|jumpsuits?|overalls|aprons?|bras?|panties|underwear|lingerie|swim ?suits?|swimwear|bikinis?|bodysuits?|leotards?|towels?|blankets?|bed ?sheets?|sheets?|shawls?|wraps?)\b/u;

// "bare shoulders + lying on bed" is suggestive as a combination; each half alone is an ordinary portrait.
const BARE_SHOULDERS_PATTERN = /\bbare shoulders?\b/u;
const LYING_PATTERN = /\blying\b|\bon (?:her |his |their )?(?:back|stomach|side)\b|\breclining\b/u;
const BED_PATTERN = /\bbed\b|\bbedroom\b|\bfuton\b/u;

// A bare chest is topless on a woman and ordinary on a man ("a shirtless surfer waxing his board"). It is decided
// with the gender of whoever it is written about (readBareChestFlags), never as a word on its own.
const BARE_CHEST_PATTERN =
  /\b(?:shirt ?less|bare ?chest(?:ed)?|bare ?torso|torso (?:bared|exposed|bare)|chest (?:bared|exposed|bare)|exposed chest|(?:flashing|baring|bared|exposing|showing|revealing) (?:her |his |their )?(?:bare )?chest|no (?:shirt|top)(?! hat)|without (?:a |her |his |their |any )?(?:shirt|top)(?! hat)|(?:shirt|top) (?:off|removed)|(?:took|takes|take|taking|taken|pulls?|pulled|pulling) off (?:her|his|their|the) (?:shirt|top|t shirt)|nothing on (?:top|the top|her top half|his top half|(?:her|his|their) upper (?:half|body))|(?:bare|naked|nude|undressed|unclothed|stripped) (?:from|above) the (?:waist|hips|chest)(?: up)?|(?:bare|naked|nude) (?:to|up to) the waist)\b/u;
// Garments that are women's wear, read as a woman where no gender word is written ("wearing only panties").
const WOMENS_WEAR_PATTERN =
  /\b(?:panties|panty|knickers|bras?|bralettes?|brassieres?|bikinis?|lingerie|thongs?|g ?strings?|(?:mini ?)?skirts?|blouses?|leotards?|camisoles?|sundress(?:es)?|gowns?|stockings|pantyhose|garter belts?|negligees?|babydolls?)\b/u;
// Gender words that do not say who the bare chest belongs to: a comparison ("bare-chested like the guys") and a
// garment's owner or cut ("shirtless in men's jeans", "in her boyfriend's shirt"). Left in, they made a woman's bare
// chest read as a man's.
const INCIDENTAL_GENDER_PATTERN =
  /\b(?:like|as|than|unlike) (?:a |an |the |other |all the |some |most |any )?(?:guys?|m[ae]n|boys?|males?|dudes?|blokes?|lads?|gentlem[ae]n|brothers?|husbands?|boyfriends?|fathers?|dads?|kings?|princes?|heroes|actors?|bodybuilders?|surfers?|girls?|wom[ae]n|lad(?:y|ies)|females?)\b|\b(?:m[ae]n|boys?|guys?|males?|husbands?|boyfriends?|brothers?|fathers?|dads?|sons?|wom[ae]n|girls?|lad(?:y|ies)|females?|wi(?:fe|ves)|girlfriends?|sisters?|mothers?|moms?)'s?(?= (?:jeans|shirts?|t shirts?|pants|trousers|shorts|clothes|clothing|jackets?|hoodies?|sweaters?|boxers|briefs|style|cut|fit|section|department|wear|size|sizes|collection|suits?|blazers?|coats?)\b)/gu;
const FEMALE_WORD_PATTERN =
  /(?:\b|\d)(?:girls?|wom[ae]n|females?|lad(?:y|ies)|she|her|hers|herself|wi(?:fe|ves)|mothers?|moms?|daughters?|sisters?|girlfriends?|brides?|actress(?:es)?|queens?|princess(?:es)?|heroines?|waitress(?:es)?|feminine)\b/u;
const MALE_WORD_PATTERN =
  /(?:\b|\d)(?:boys?|m[ae]n|males?|guys?|gentlem[ae]n|he|him|his|himself|husbands?|fathers?|dads?|sons?|brothers?|boyfriends?|grooms?|actors?|kings?|princes?|heroes|waiters?|masculine)\b/u;

// ---------------------------------------------------------------------------------------------------------------
// Safety-floor subject coding
// ---------------------------------------------------------------------------------------------------------------

// Adults who work where children are (an elementary school teacher, a 6th grade teacher) are not minors. Scoped to
// the word right after the school, so a second, unqualified mention in the same term is still read.
const SCHOOL_STAFF_GUARD = String.raw`(?! (?:teachers?|principals?|vice principals?|staff|nurses?|counsell?ors?|librarians?|faculty|administrators?|employees?|workers?|volunteers?|aides?|assistants?|bus drivers?|janitors?|custodians?|secretar(?:y|ies)|reunion)\b)`;
// "baby" as fashion, food, a juvenile animal or an adult's milestone (baby blue sweater, baby carrots, baby octopus,
// a baby shower, a baby bump, baby boomers) is not a child. Scoped to that one word, so "child in a baby blue dress"
// and "baby oil on a baby" stay minor-coded.
const BABY_GUARD = String.raw`(?!'s breath| (?:pink|blue|yellow|green|purple|lavender|bangs|hairs?|tee|t ?shirt|powder|carrots?|spinach|corn|potato(?:es)?|back ribs|ribs|octopus|squid|grapes?|tomato(?:es)?|kale|arugula|greens|bok choy|zucchini|grand(?: piano)?|shower|showers|bump|oil|boomers?|animals?|birds?|chicks?|ducks?|elephants?|goats?|deer|seals?|turtles?|penguins?|bunn(?:y|ies)|rabbits?|cats?|dogs?|kittens?|pupp(?:y|ies)|foxes|bears?|pandas?|lions?|tigers?|monitor|wipes|food|formula|bottle|stroller|carriage|crib)\b)`;

// Any of these anywhere in the cut keeps it off the photo model, whatever the category. jk / jc / js are the
// Japanese tag shorthand for high-, middle- and elementary-school girls; with word boundaries they only ever match
// the standalone token, and never the ".js" of a code library.
const MINOR_RULE: TermRule = {
  pattern: wordAlternation([
    String.raw`lolis?|lolicon|lolli|rori|shotas?|shotacon|shouta|toddlercon|mesugaki|nymphets?|nymphettes?`,
    // "lolita fashion" (and its gothic, sweet, classic substyles) is an adult street style; the bare word is not.
    String.raw`(?<!(?:gothic|goth|sweet|classic|punk|hime|country|casual|wa|qi|sailor) )lolitas?(?! (?:fashion|dress(?:es)?|style|outfits?|coords?|clothing|clothes|skirts?|shoes|boots|accessories|aesthetic|brand|tea party)\b)`,
    String.raw`child|children|childs|childlike|child like|childish (?:body|figure|build|frame|physique|proportions|face|features|appearance|looks?)|girl ?child|boy ?child`,
    String.raw`kids?(?!'?s? (?:gloves?|leather|suede|menus?|friendly|meals?|sized?)\b)|kiddies|kiddie|kiddos?|tykes?|youngsters?|younglings?|toddlers?|(?<!tater )tots?(?! of\b)|infants?|newborns?`,
    `bab(?:y|ies)${BABY_GUARD}`,
    String.raw`baby ?faced?|babyface`,
    String.raw`(?:little|small|tiny|wee|young|youthful) (?:[a-z']+ ){0,2}(?:(?:school ?)?(?:girls?|boys?|kids?)|lass(?:es|ies)?|lads?|ones?(?! piece))`,
    `(?:daycare|day care)${SCHOOL_STAFF_GUARD}`,
    String.raw`preteens?|pre ?teens?|tweens?|tweenagers?|teen(?:s|y|ie|ies|aged?|agers?|yboppers?)?|adolescen\w*|juveniles?|(?:pre ?)?pub(?:escen(?:t|ce)|ertal)|puberty`,
    `elementary (?:school(?:ers?|girls?|boys?|kids?)?|students?|pupils?|kids?|aged?)${SCHOOL_STAFF_GUARD}`,
    `(?:grade|primary|secondary|nursery) ?school(?:ers?|girls?|boys?|kids?|students?|pupils?)?${SCHOOL_STAFF_GUARD}`,
    `middle ?school(?:ers?|girls?|boys?|kids?|students?)?${SCHOOL_STAFF_GUARD}`,
    `(?:kindergart(?:en|ner|ener)s?|pre ?k(?:indergarten)?|pre ?school(?:ers?|girls?|boys?|kids?)?)${SCHOOL_STAFF_GUARD}`,
    String.raw`junior high|j\.?r\.? high|jhs|(?:first|1st|second|2nd) years? (?:high|middle) school|high school (?:freshm[ae]n|sophomores?|first years?|1st years?)`,
    String.raw`school ?kids?|school ?child(?:ren)?|schoolchild(?:ren)?|school ?aged?|schoolage`,
    String.raw`under ?aged?|underage|aged down|age regression|jailbait|barely legal`,
    String.raw`(?:girl|boy|cub|brownie|daisy) scouts?(?! (?:cookies?|leaders?|troop leaders?|moms?|mothers?|dads?)\b)`,
    String.raw`diapers?|nappies|nappy|pacifiers?|sweet (?:sixteen|16)|quincea(?:n|ny)eras?|(?:bat|bar|bas|b'nai) mitzvahs?`,
    String.raw`p(?:a)?edophil\w*|pedos?`,
    String.raw`(?<!\.)(?:jk(?! rowling)|jc|js)`,
    // Romanised Japanese, Chinese and Korean words for a child, a little girl or a schoolchild.
    String.raw`chuu?gakusei|shou?gakusei|joshi ?(?:shou?|chuu?|kou?)gakusei|joshi ?kou?sei|joshikou?sei|youjo|yojo|shoujo|shojo|kodomo|ertong|xiaoxuesheng|chuzhongsheng|xiaon[vu] ?hai|shaon[vu]|luoli|eorini|chodeung ?haksaeng|jungdeung ?haksaeng|yeojung(?:saeng)?|yeogo(?:saeng)?|sonyeo`,
    // Other Latin-script languages. "niña" keeps its ñ as "ny" through the fold (canonicalizeSceneText); the plain
    // "nina" is a common name, so it counts only in the plural or next to a word that makes it a girl.
    String.raw`niny(?:a|o|as|os|ita|itas|ito|itos|ez)|ninas|ninos|ninit[ao]s?|(?:una|la|las|esta|esa|pequena|joven) nin[ao]s?|nin[ao]s? (?:pequen[ao]s?|de (?:la |una )?(?:escuela|colegio|primaria))|colegialas?|menor(?:es)? de edad`,
    String.raw`fillettes?|petites? filles?|enfants?|ecolier(?:e|es|s)?|collegien(?:ne|nes|s)?|mineure?s?`,
    String.raw`madchen|kleinkind(?:er)?|kindchen|minderjahrig\w*|teenagerin(?:nen)?|schulmadchen`,
    String.raw`menin(?:a|as|o|os)|criancas?|garotinhas?|bambin[aeio]|ragazzin[aeio]|minorenn[ei]|scolarett[ae]|meisjes?`,
    String.raw`anak|gadis kecil|batang (?:babae|lalaki)|tre em|be gai|be trai|hoc sinh|cocuk\w*|kiz cocugu|kizcagiz|dziewczyn(?:ka|ki|ek)|dzieck[oa]|dzieci|tytto|mtoto|watoto`,
    // The long tail of Latin-script languages. A word that is also an ordinary word elsewhere ("kind" and "barn" in
    // English, "ado" in "much ado", "Kinder" chocolate) counts only next to a word of its own language.
    String.raw`kinderen|tiener\w*|minderjarig\w*|basisschool\w*|scholiere\w*|kleuter\w*|peuter\w*|pubertje|(?:een|het|klein|kleine|jong|jonge) kind|nastolat\w*|uczennic\w*|nieletni\w*|maloletni\w*|podlotek|gimnazjalist\w*|holcick\w*|skolack\w*|nezletil\w*|dospivajic\w*|dite|deti|ergen|teini|gosses?|nimph?(?:et|ette)s?|nymfet\w*`,
    String.raw`resit olmayan|ogrenci kiz|liseli kiz|ortaokul\w*|ilkokul\w*|kucuk kiz|flickebarn|tonar(?:ing|ingar|ingen|ingarna)|minderarig\w*|mindrearig\w*|skolflick\w*|liten flicka|skolepige\w*|lille pige|(?:ett|smat?|sma|mitt|vart|lilla) barn|copil|copii|copilul|fetit\w*|minora|scolarit\w*|gyerek\w*|kislany\w*|kiskoru\w*|lapsi|lapset|alaikai\w*|djevojcic\w*|dijete|djeca|maloljetn\w*`,
    String.raw`grundschul\w*|jugendlich\w*|halbwuchsig\w*|(?:ein|das|dem|des|kleines|kleine|mein|ihr|sein|unser|deutsches) kind|(?:die|den|der|kleine|kleinen|zwei|drei|viele|unsere|meine|ihre|seine|mit|fur) kinder|preados?|(?:une|les|des|cette|petite|jeune) ados?|ados? de \d|gamines?|fillettes?|jeune fille|chiquill\w*|chamac\w*|pubers?|novinhas?|ninfet\w*|remaja|di ?bawah umur|dalagita|cheongsonyeon|migeonyeonja|yuah|youyou|xiaohai|ertong|shaonu|shaonv`,
    // English for a child the word list above did not know, and possessives that only a child takes.
    String.raw`bairns?|(?<!to )weans?(?! (?:off|from)\b)|(?<!(?:nail|cuticle|wire|hoof) )nippers?|sprogs?|rugrats?|ankle ?biters?|(?<!dunkin'? )munchkins?(?! (?:cats?|kittens?|card game|game))|whippersnappers?|moppets?|(?<!sea )urchins?(?! (?:pasta|roe|sushi|uni|spines?|shells?)\b)|wee (?:ones?|bairns?|lass(?:ie)?|lad(?:die)?)|babyish|pre ?adolescen\w*|pre ?teenagers?|infantile (?:body|figure|build|frame|face|features|proportions|physique)`,
    String.raw`(?:little|small|tiny|wee|young|youthful|baby) (?!(?:adult|grown|grown up)\b)(?:[a-z']+ )?(?:daughters?|sons?|nieces?|nephews?|sisters?|brothers?|cousins?|missy|misses|granddaughters?|grandsons?|stepdaughters?|stepsons?)|kid ?sized? (?:body|figure|frame|build|physique|torso|woman|girl|lady|person|proportions|chest|breasts|hips)|(?:girls?|boys?|kids?|child(?:ren)?|youth|toddlers?|juniors?)'?s?'? sizes?(?! (?:chart|guide)\b)`,
    String.raw`training bras?|first bras?|(?:un|under|not yet |not fully |barely )developed (?:body|chest|breasts?|figure|frame|bust|physique|hips)(?! of\b)|(?:has ?n't|has not|not yet|yet to|hasn't yet|have not|haven't) (?:fully )?developed(?=$| yet\b| (?:breasts|curves|a figure|her figure|physically)\b)|no (?:curves|breasts|bust|chest|hips) yet|(?:small|little) for (?:her|his|their) age|girlish (?:body|figure|frame|build|physique)|growing (?:girls?|boys?|kids?|child)|sippy cups?|booster seats?|middle grade (?:readers?|kids?|students?|girls?|boys?)|lower school (?:girls?|boys?|pupils?|students?|kids?)|prep school(?:ers?| (?:girls?|boys?|pupils?|kids?))|sixth form(?:ers?)?|(?:lower|upper) sixth`
  ])
};

/**
 * Being under age said without a number: "not yet an adult", "under the age of consent", "below legal age", "too
 * young to vote", "years away from adulthood". Held to the person sense: "not adult content" and "not legal advice"
 * stay ordinary.
 */
const NOT_OF_AGE_PATTERN = new RegExp(
  [
    String.raw`\b(?:not|isn't|is not|wasn't|was not|aren't|are not|never|not yet|not quite|yet to be|yet to become|far from|barely)(?: yet)? (?:an? |of )?(?:adults?|grown ?ups?|grown|legal age|age|majority|full age)(?: yet)?(?=$| (?:yet|still|and|but|or|in|on|at|with|who|wearing|girl|boy)\b)`,
    // "Legal" alone is the law's unless it is still to come ("not legal yet") or is all the term says.
    String.raw`\b(?:not|isn't|is not|wasn't|aren't) (?:yet legal|legal yet)\b|^not legal$`,
    String.raw`\b(?:under|below|beneath|short of|shy of|before|younger than) (?:the )?(?:legal age|(?:legal )?age of (?:consent|majority|adulthood)|legal (?:drinking |voting )?age|adulthood|majority|consent age|adult age)\b|\byounger than (?:the )?legal\b`,
    String.raw`\btoo young to (?:vote|drink|drive|consent|marry|date|be (?:here|out|alone))\b|\b(?:years|a year|months|a few years) (?:away )?(?:from|short of) (?:adulthood|being an adult|18|eighteen)\b|\bstill (?:a )?(?:minor|underage|child|kid)\b(?! at heart)`
  ].join("|"),
  "u"
);

// ---------------------------------------------------------------------------------------------------------------
// Pictographs and the child-body combination
// ---------------------------------------------------------------------------------------------------------------

/**
 * Emoji are neither letters nor digits, so no lexicon and no unverifiable-script check ever saw them, while the Qwen
 * encoders read "A 🧒 in a 👙 on a 🛏️" as a child in a bikini on a bed. The photo builder strips every pictograph
 * (stripPictographs); the floor counts the ones with a sexual, child or school meaning first, so a typed emoji prompt
 * locks or is refused rather than being quietly cleaned into something else. Each set is the symbol's common sexual
 * or childhood use: 🔞 🍆 🍑 💦 🍒 👅 🫦 🥵, a garment crossed out ("👙❌"), 🧒 👶 👧 👦 🚸 🍼 👼, 👙 🩲 🩱 💋 👄, 🏫 🎒.
 */
const NEGATION_EMOJI = String.raw`[\u{274C}\u{274E}\u{1F6AB}\u{26D4}\u{2716}]`;
const GARMENT_EMOJI = String.raw`[\u{1F459}\u{1F457}\u{1F455}\u{1F456}\u{1FA72}\u{1FA71}\u{1F45A}\u{1F9E5}\u{1F97B}\u{1F458}\u{1F459}]`;
const EXPLICIT_EMOJI_PATTERN = new RegExp(
  String.raw`[\u{1F51E}\u{1F346}\u{1F351}\u{1F4A6}\u{1F352}\u{1F445}\u{1FAE6}\u{1F975}]|${GARMENT_EMOJI}\s*${NEGATION_EMOJI}|${NEGATION_EMOJI}\s*${GARMENT_EMOJI}`,
  "u"
);
const MINOR_EMOJI_PATTERN = /[\u{1F9D2}\u{1F476}\u{1F467}\u{1F466}\u{1F6B8}\u{1F37C}\u{1F47C}]/u;
const SUGGESTIVE_EMOJI_PATTERN = /[\u{1F459}\u{1FA72}\u{1FA71}\u{1F48B}\u{1F444}]/u;
const SCHOOL_EMOJI_PATTERN = /[\u{1F3EB}\u{1F392}]/u;
/** Every pictograph, emoji skin tone and keycap: what the photo builder strips from photo text. */
const PICTOGRAPH_PATTERN = /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u{20E3}]/gu;

/** True when the text carries a pictograph the safety floor reads (sexual, child or school). */
export function hasFloorPictograph(text: string | undefined): boolean {
  const value = typeof text === "string" ? text : "";
  return [EXPLICIT_EMOJI_PATTERN, MINOR_EMOJI_PATTERN, SUGGESTIVE_EMOJI_PATTERN, SCHOOL_EMOJI_PATTERN].some((pattern) => pattern.test(value));
}

/** The text without pictographs, emoji skin tones or keycaps, for the photo builder. Spacing is tidied after. */
export function stripPictographs(text: string): string {
  return text.replace(PICTOGRAPH_PATTERN, " ").replace(/ {2,}/gu, " ").replace(/ +([,.;:!?])/gu, "$1").trim();
}

/**
 * Child-body descriptors, by what they describe. Each is ordinary on an adult (a petite woman, twin tails, a flat
 * chest) and no one of them codes a minor. Two different kinds on a female subject in swimwear or underwear are how a
 * child is described without the word ("a petite girl, 140 cm tall, flat-chested, with twin tails, in a bikini"), so
 * that combination is minor-coded (classifyImageScene). A height under 150 cm (4 ft 11 in) counts as a small frame.
 */
const CHILD_BODY_DESCRIPTORS: ReadonlyArray<{ kind: string; pattern: RegExp }> = [
  {
    kind: "flat chest",
    pattern: /\b(?:flat ?chest(?:ed)?|flat (?:breasts|bust)|(?:no|tiny|nonexistent|non existent|barely there) (?:breasts|bust|boobs|chest)(?! (?:hair|pain|tattoos?|pocket))|breastless|boyish chest|chest(?: is)? (?:completely )?flat)\b/u
  },
  { kind: "no hips", pattern: /\b(?:no|tiny|boyish|undeveloped|childlike|straight) hips\b|\bhipless\b|\bstraight up and down\b|\bboyish (?:figure|build|frame|body)\b/u },
  {
    kind: "small frame",
    pattern: /\bpetite\b|\bpint ?sized\b|\bdiminutive\b|\bshort stature\b|\b(?:tiny|little|short|small|slight|delicate|childlike|small boned|tiny boned|undersized|narrow)(?:,? (?:and )?(?:petite|small|slight|tiny|short|thin|skinny|slender))? (?!(?:haired|hair|sleeved|skirted|legged|door|window|picture|bed|business)\b)(?:[a-z]+ )?(?:frame|build|body|figure|stature|physique|body type|girls?|wom[ae]n|lady|thing)\b/u
  },
  { kind: "child face", pattern: /\b(?:chubby|puffy|plump|round) cheeks\b|\bbaby fat\b|\b(?:childlike|childish|cherubic|babyish) face\b|\bbig innocent eyes\b/u },
  { kind: "child hair", pattern: /\b(?:pig ?tails?|twin ?tails?|twintails|bunches|hair ?(?:ribbons?|bows?)|(?:ribbons?|bows?) in (?:her )?hair)\b/u },
  { kind: "child props", pattern: /\b(?:lollipops?|juice ?box(?:es)?|stuffed (?:animals?|toys?|bunny|bear)|teddy bears?|plush(?:ie)? toys?|crayons?|colou?ring books?)\b/u }
];
const HEIGHT_CM_PATTERN = /\b(\d{2,3}(?:[.,]\d)?) ?(?:cm|centimet(?:er|re)s?)\b/gu;
const HEIGHT_M_PATTERN = /\b1[.,]([0-4]\d?) ?(?:m|meters?|metres?)\b/u;
const HEIGHT_FT_PATTERN = /\b([3-4]) ?(?:'|ft|feet|foot)(?: ?(\d{1,2}) ?(?:"|in|inches)?)?(?: tall)?\b/gu;
const HEIGHT_CONTEXT_PATTERN = /\b(?:tall|height|stands?|standing|short|petite|stature|measures?|only|just|barely)\b/u;
// The measurement written onto the person it measures needs no height word: "a 142 cm woman", "a woman of 140cm".
const HEIGHT_OF_PERSON_PATTERN = new RegExp(
  String.raw`(?:\d ?(?:cm|centimet(?:er|re)s?|m|ft|feet|foot|in|inches|'|")(?: ?\d{1,2} ?(?:in|inches|")?)?) (?:[a-z-]+ )?(?:girls?|wom[ae]n|lad(?:y|ies)|person|figure|model|she)\b|\b(?:girls?|wom[ae]n|lad(?:y|ies)|person|model) (?:of|at|measuring|standing) \d`,
  "u"
);

/** The kinds of child-body descriptor the terms carry (see CHILD_BODY_DESCRIPTORS). */
function readChildBodyKinds(terms: string[]): Set<string> {
  const kinds = new Set<string>();
  for (const term of terms) {
    for (const { kind, pattern } of CHILD_BODY_DESCRIPTORS) {
      if (pattern.test(term)) {
        kinds.add(kind);
      }
    }
    if (isShortStature(term)) {
      kinds.add("small frame");
    }
  }
  return kinds;
}

/** A height under 150 cm, written in cm, metres or feet and inches, stated as a person's height. */
function isShortStature(term: string): boolean {
  // A bare measurement ("140 cm") is a height in a figure's tags; inside a longer term it needs a height word.
  const statedAsHeight = HEIGHT_CONTEXT_PATTERN.test(term) || HEIGHT_OF_PERSON_PATTERN.test(term) || term.split(" ").length <= 2;
  if (!statedAsHeight) {
    return false;
  }
  const centimetres = Array.from(term.matchAll(HEIGHT_CM_PATTERN), (match) => Number(match[1].replace(",", ".")));
  if (centimetres.some((value) => value >= 80 && value < 150) || HEIGHT_M_PATTERN.test(term)) {
    return true;
  }
  return Array.from(term.matchAll(HEIGHT_FT_PATTERN)).some((match) => Number(match[1]) * 12 + Number(match[2] ?? 0) < 59.1);
}

// "minor" is also an ordinary adjective (minor details, a minor injury, the minor key). It codes a person only as a
// noun: the whole term, after an article or a copula, or right before a word naming who the minor is.
const MINOR_NOUN_PATTERN =
  /(?:^|\b(?:an?|the|is|are|was|were|as|still|being) )minors?(?=$| (?:with|and|or|in|on|at|who|wearing|dressed|aged?|of|girls?|boys?|child(?:ren)?|females?|males?|kids?|students?)\b)/u;

// "sixth grader", "6th grade", "grade 6", "Year 7 pupil", "Year Nine girl", "P7 pupil", "S3 pupil", "Form 2 student",
// "Klasse 7". "sixth sense" and "color grade" never reach the ordinal + grade shape, and a school year, form or
// Scottish P/S stage needs a pupil next to it (or "in Year 7", or the bare stage as a whole term) to be one.
const SCHOOL_YEAR_NUMBER = String.raw`(?:[1-9]|1[0-3]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen)`;
const SCHOOL_PUPIL = String.raw`(?:students?|pupils?|girls?|boys?|kids?|child(?:ren)?|class(?:es)?|schoolers?|learners?)`;
const SCHOOL_GRADE_PATTERN = new RegExp(
  [
    String.raw`\b(?:(?:1st|2nd|3rd|[4-9]th|1[0-2]th|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth) ?grade(?:rs?)?|grade (?:[1-9]|1[0-2]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))\b${SCHOOL_STAFF_GUARD}`,
    String.raw`\bin year ${SCHOOL_YEAR_NUMBER}\b|\byear ${SCHOOL_YEAR_NUMBER} ${SCHOOL_PUPIL}\b|^year ${SCHOOL_YEAR_NUMBER}$`,
    String.raw`\b(?:(?:primary|p) ?[1-7]|(?:secondary|sec|s) ?[1-6]) ${SCHOOL_PUPIL}\b|\bform (?:[1-6]|one|two|three|four|five|six) ${SCHOOL_PUPIL}\b`,
    String.raw`\b(?:klasse|klasa|klass|trida) ?(?:[1-9]|1[0-2])\b|\b(?:[1-9]|1[0-2])\.? ?(?:klasse|klasa|klass)\b|^in (?:[1-9]|1[0-2])(?:st|nd|rd|th)$`
  ].join("|"),
  "u"
);

// Student identities. In a tag they are school-coded (an adult cast as a high school student is the genre's norm,
// spec §2.3), and the photo builder never renders them, so the subject stays "an adult woman". In free text — an
// appearance, a style, a studio prompt, a finished photo prompt — the words are rendered as written, and nothing
// is left to make the subject an adult, so there they are minor-coded (see classifyImageScene).
const STUDENT_COLLEGE_GUARD = String.raw`(?<!(?:college|university|uni|graduate|grad|postgraduate|undergraduate|medical|med|law|nursing|phd|doctoral|masters?|mba|adult|mature|former|ex) )`;
const STUDENT_IDENTITY_RULE: TermRule = {
  pattern: wordAlternation([
    `${STUDENT_COLLEGE_GUARD}students?`,
    `${STUDENT_COLLEGE_GUARD}(?:freshm[ae]n|sophomores?)`,
    // The randoseru is the bag only an elementary schooler carries; it names the wearer as surely as the noun.
    String.raw`school ?girls?|school ?boys?|high ?schoolers?|high school (?:girls?|boys?|pupils?|kids?|seniors?|juniors?)|(?:school|secondary) pupils?|kogals?|joshi ?kou?sei|joshikou?sei|randoseru`,
    String.raw`colegial(?:a|as|es)?|lyceen(?:ne)?s?|schulerin(?:nen)?|schuler|estudiantes?|eleves?|eleva|siswa|siswi|murid|haksaeng|gakusei|xuesheng|uczen|uczniowie|zakyne|ogrenci|oppilas`
  ])
};

// School-coded subjects. Not a lock on their own: adults in school uniform are ordinary fiction. Locked only
// together with a suggestive tag. A high schooler stays here rather than in MINOR_RULE (spec §2.3: an adult cast as
// a high school student is the genre's norm); middle and elementary schoolers are minor-coded.
const SCHOOL_RULE: TermRule = {
  pattern: wordAlternation([
    String.raw`school uniforms?|serafuku|seifuku|sailor (?:school )?uniforms?|sailor fuku|sailor suits?|sailor collars?|sailor dress(?:es)?|gakuran|academy uniforms?`,
    String.raw`(?:school|academy|prep school|student council) (?:blazers?|vests?|jackets?|cardigans?|sweaters?|skirts?|ties?|ribbons?|shoes|socks|swim ?suits?|bags?|hats?|desks?|chairs?|lockers?|hallways?|corridors?|rooftops?|gyms?|grounds|gates?|bus(?:es)?|trips?|festivals?|infirmary|library|cafeteria|courtyard|buildings?|classrooms?|pools?|clubs?|life|days?)`,
    String.raw`(?:gym|pe|p e|physical education) (?:uniforms?|clothes|class(?:es)?|shorts|kit)|bloomers|buruma|school swim ?suits?|sukumizu|randoseru|uwabaki|kogal`,
    String.raw`class ?rooms?|home ?rooms?|school ?yards?|cram school|juku|(?:at|after|before|to|in|from|skipping) school`,
    String.raw`high ?school(?:ers?)?`,
    String.raw`(?:uniforme|seragam) (?:escolar|scolaire|scolastica|sekolah)|schuluniform|escuela|colegio|escolar(?:es)?|ecole|schule|scuola|escola|sekolah|gakkou?|liseli`,
    // School life without the word: the events, rooms and things only a school has. "Detention" the facility and the
    // nurse at a hospital are other places.
    String.raw`prom(?: night| dress(?:es)?| queen| king| date| photos?)?|homecoming (?:dance|queen|king)|detention(?! (?:cent(?:er|re)s?|facilit(?:y|ies)|camps?|basins?|ponds?|hearings?|orders?)\b)|(?:school )?nurse'?s? (?:office|room)|school nurses?|yearbooks?(?: photos?)?|hall pass(?:es)?|pencil cases?|school satchels?|satchel bags?|chalk ?boards?|black ?boards?|teacher'?s? desks?|homework|field trips?|recess|gymnasium|cheerleaders?(?: uniforms?| outfits?)?|pep rally|class trips?|after school club|student id(?: cards?)?|boarding school (?:girls?|boys?)`
  ])
};
// A whole term "school" is the place; inside a longer term it needs one of the forms above.
const BARE_SCHOOL_PATTERN = /^(?:a |the )?school$/u;
// A short pleated skirt with a ribbon tie, a blazer or knee socks is how a tag model writes a school uniform without
// the word; a plaid or tartan skirt was the only one read before. A midi or maxi pleated skirt is office wear.
const SCHOOL_SKIRT_PATTERN =
  /\b(?:plaid|tartan|checked|checkered) (?:pleated |mini |micro )?skirts?\b|\bpleated (?:plaid|tartan) skirts?\b|(?<!(?:midi|maxi|long|ankle length|floor length|tea length|knee length) )\bpleated (?:(?:mini|micro|short|navy|grey|gray|blue|black|red|green|white) )?skirts?\b/u;
const SCHOOL_NECKWEAR_PATTERN = /\b(?:ribbon ties?|neck ribbons?|bow ?ties?|neckties?|neckerchiefs?|blazers?|knee ?(?:high )?socks|sailor collars?)\b/u;

// ---------------------------------------------------------------------------------------------------------------
// Stated ages
// ---------------------------------------------------------------------------------------------------------------

const SPELLED_NUMBERS: Readonly<Record<string, number>> = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12,
  thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, dozen: 12
};
const ROMAN_NUMBERS: Readonly<Record<string, number>> = {
  i: 1, ii: 2, iii: 3, iv: 4, v: 5, vi: 6, vii: 7, viii: 8, ix: 9, x: 10, xi: 11, xii: 12, xiii: 13, xiv: 14, xv: 15,
  xvi: 16, xvii: 17, xviii: 18, xix: 19
};
const SPELLED_ORDINALS: Readonly<Record<string, number>> = {
  first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7, eighth: 8, ninth: 9, tenth: 10,
  eleventh: 11, twelfth: 12, thirteenth: 13, fourteenth: 14, fifteenth: 15, sixteenth: 16, seventeenth: 17,
  eighteenth: 18
};
// A number standing alone: never the tail of "twenty-one" or "21.5", which would read "one" or "5" as the age. A
// decimal age is its whole part ("14.5 years old", "a 16.5-year-old", "17.9 yo"): refusing to read "14.5" at all
// made every fractional age read as no age. "4teen" is fourteen spelled with a digit.
const AGE_NUMBER = String.raw`((?<![\d.,])\d{1,2}(?:[.,]\d{1,2})?(?!\d)(?![.,]\d)|(?<![a-z\d])[3-9] ?teen|(?<!(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand) )(?:${Object.keys(SPELLED_NUMBERS).join("|")}))`;
const AGE_QUALIFIER = String.raw`(?:(?:just|only|about|around|roughly|approximately|nearly|almost|barely|merely|exactly|now|still|already|maybe|perhaps|probably|like|approx|circa|ca|some|~) ?)?`;
// A number followed by one of these is a quantity, not someone's age ("aged 15 days", "under 18 minutes").
const NOT_AN_AGE = String.raw`(?! ?['"] ?\d)(?! ?(?:days?|weeks?|months?|hours?|hrs?|minutes?|mins?|seconds?|secs?|%|percent|degrees?|kg|kilos?|g|grams?|lbs?|pounds?|cm|mm|m|meters?|metres?|km|miles?|feet|ft|inches|dollars?|usd|euros?|won|yen|calories|kcal|points?|pts|times|items?|pieces?|steps?|floors?|stor(?:ey|y|ies|eys)|pages?|levels?|laps?|reps?|sets?|cups?|glasses|bottles?|slices?|people|persons|guests|o'?clock|am|pm|a m|p m)\b)`;
const PERSON_NOUN = String.raw`(?:girls?|boys?|wom[ae]n|m[ae]n|lad(?:y|ies)|child(?:ren)?|kids?|persons?|daughters?|sons?|sisters?|brothers?|students?|pupils?|models?|teens?|teenagers?|lass(?:ie)?|lads?|maidens?|princess(?:es)?|princes?|1girl|1boy|she|he)`;
const PERSON_NOUN_ONLY = String.raw`(?:girls?|boys?|wom[ae]n|m[ae]n|lad(?:y|ies)|child(?:ren)?|kids?|persons?|daughters?|sons?|sisters?|brothers?|students?|pupils?|models?|teens?|teenagers?|lass(?:ie)?|lads?|maidens?|princess(?:es)?|princes?|1girl|1boy)`;
const APPROXIMATELY = String.raw`(?:around|about|roughly|approximately|approx|circa|ca|maybe|perhaps|probably|like|nearly|almost|barely|just|only|some|~)`;

interface AgeReading {
  pattern: RegExp;
  /** Ages strictly below this are minors (19 for "under 18", where the stated number is the limit). */
  limit?: number;
  /** The noun right after an "N-year-old" may own the age instead of a person ("a 12-year-old whisky"). */
  ownedByNoun?: "after-old" | "after-aged";
  /** The stated number counts this many per year instead of years ("168 months old" is 14). */
  perYear?: number;
}

/**
 * Spelled ages in other languages, each language's numbers with its own words for "years old" only: pairing every
 * number word with every unit read Danish "to" (two) + Czech "let" (years) into "rooms to let". `joined` units also
 * attach without a space ("veertienjarig", "czternastoletnia", "fjortonaring").
 */
const FOREIGN_SPELLED_AGES: ReadonlyArray<{ numbers: string; units: string; joined?: string }> = [
  { numbers: "uno|una|un|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieciseis|diecisiete", units: "anos|anitos" },
  { numbers: "un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|dix sept", units: "ans" },
  { numbers: "ein|eins|zwei|drei|vier|funf|sechs|sieben|acht|neun|zehn|elf|zwolf|dreizehn|vierzehn|funfzehn|sechzehn|siebzehn", units: "jahre|jahren", joined: String.raw`jahrig\w*` },
  { numbers: "uno|una|un|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|tredici|quattordici|quindici|sedici|diciassette", units: "anni" },
  { numbers: "um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito|nove|dez|onze|doze|treze|catorze|quatorze|quinze|dezesseis|dezasseis|dezessete|dezassete", units: "anos|aninhos" },
  { numbers: "satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|(?:dua|tiga|empat|lima|enam|tujuh) belas", units: "tahun" },
  { numbers: "een|twee|drie|vier|vijf|zes|zeven|acht|negen|tien|elf|twaalf|dertien|veertien|vijftien|zestien|zeventien", units: "jaar", joined: String.raw`jarig\w*` },
  {
    numbers:
      "jeden|jedna|dwa|dwie|trzy|cztery|piec|szesc|siedem|osiem|dziewiec|dziesiec|jedenascie|dwanascie|trzynascie|czternascie|pietnascie|szesnascie|siedemnascie|jedno|dwu|cztero|piecio|szescio|siedmio|osmio|dziewiecio|dziesiecio|jedenasto|dwunasto|trzynasto|czternasto|pietnasto|szesnasto|siedemnasto",
    units: "lat|lata",
    joined: String.raw`letni\w*`
  },
  { numbers: "jeden|jedna|jedno|dva|dve|tri|ctyri|pet|sest|sedm|osm|devet|deset|jedenact|dvanact|trinact|ctrnact|patnact|sestnact|sedmnact", units: "let|roku|roky", joined: String.raw`i?let(?:a|y|e|ou|ych)?` },
  { numbers: "bir|iki|uc|dort|bes|alti|yedi|sekiz|dokuz|on|on ?(?:bir|iki|uc|dort|bes|alti|yedi)", units: String.raw`yas\w*` },
  {
    numbers: "en|ett|tva|tre|fyra|fem|sex|sju|atta|nio|tio|elva|tolv|tretton|fjorton|femton|sexton|sjutton|to|fire|seks|syv|otte|ni|ti|elleve|tretten|fjorten|femten|seksten|sytten",
    units: String.raw`a?ar(?=$| (?:gammal|gamla|gammel|gamle|gml)\b)`,
    joined: String.raw`a?ari(?:ng|ge)\w*|a?arig\w*`
  },
  { numbers: "yksi|kaksi|kolme|nelja|viisi|kuusi|seitseman|kahdeksan|yhdeksan|kymmenen|yksitoista|kaksitoista|kolmetoista|neljatoista|viisitoista|kuusitoista|seitsemantoista", units: String.raw`vuotia\w*`, joined: String.raw`vuotia\w*` },
  { numbers: "unu|una|un|doi|doua|trei|patru|cinci|sase|sapte|opt|noua|zece|unsprezece|doisprezece|douasprezece|treisprezece|paisprezece|cincisprezece|saisprezece|saptesprezece", units: "ani" },
  { numbers: "egy|ketto|ket|harom|negy|ot|hat|het|nyolc|kilenc|tiz|tizenegy|tizenketto|tizenharom|tizennegy|tizenot|tizenhat|tizenhet", units: "eves", joined: "eves" },
  { numbers: "jedan|jedna|dva|dvije|tri|cetiri|pet|sest|sedam|osam|devet|deset|jedanaest|dvanaest|trinaest|cetrnaest|petnaest|sesnaest|sedamnaest", units: String.raw`godin[aeu]|godisnj\w*` },
  { numbers: "un|una|dos|tres|quatre|cinc|sis|set|vuit|nou|deu|onze|dotze|tretze|catorze|quinze|setze|disset", units: "anys" },
  { numbers: "isa|dalawa|tatlo|apat|lima|anim|pito|walo|siyam|sampu|labing ?isa|labindalawa|labintatlo|labing ?apat|labinlima|labing ?anim|labimpito", units: "(?:na )?taon(?:g gulang)?" },
  // Romanised Japanese, Chinese and Korean teens only: the single digits are ordinary syllables in every language.
  { numbers: "jy?uu? ?(?:ichi|ni|san|yon|shi|go|roku|nana|shichi)?", units: "sai", joined: "sai" },
  { numbers: "shi ?(?:yi|er|san|si|wu|liu|qi)?", units: "sui", joined: "sui" },
  { numbers: "yeol ?(?:han|hana|du|dul|se|set|ne|net|daseot|yeoseot|ilgop)?|yeor(?:ilgop|yeoseot)", units: "sal", joined: "sal" }
];

// Ages are written into captions and prose more often than Danbooru would, in every phrasing English has. Global,
// so a term naming two ages is read for both.
const AGE_READINGS: readonly AgeReading[] = [
  // "14 years old", "17-year-olds", "12 years young", "twelve and a half years old", "fourteen summers old", "a
  // 14-ish year old".
  {
    pattern: new RegExp(String.raw`\b${AGE_NUMBER}(?: ?ish)?(?: and a (?:half|quarter))? ?(?:years?|yrs?|summers|winters) ?(?:old|olds|of age|young)\b`, "gu"),
    ownedByNoun: "after-old"
  },
  // The unit misspelt or cut short, which an encoder still reads: "14 yeras old", "14y old", "14 yrz old", "14 year's
  // old". Any short word starting with y between the number and "old".
  { pattern: new RegExp(String.raw`\b${AGE_NUMBER}(?: ?ish)? ?y[a-z']{0,5}\.? ?olds?\b`, "gu"), ownedByNoun: "after-old" },
  // "fourteen years and two months old".
  { pattern: new RegExp(String.raw`\b${AGE_NUMBER} ?(?:years?|yrs?) and (?:a half|an? |\w+ )?(?:months?|weeks?|days?) old\b`, "gu"), ownedByNoun: "after-old" },
  // "16yo", "12 y/o", "12 y o", "twelve yo".
  { pattern: new RegExp(String.raw`\b${AGE_NUMBER} ?(?:yo|y/o|y\.o\.?|y o)(?=$|[^a-z])`, "gu") },
  // "aged 14", "age: 15", "aged just 12", "age thirteen-ish".
  { pattern: new RegExp(String.raw`\baged? ?[:=]? ?${AGE_QUALIFIER}${AGE_NUMBER}(?: ?ish)?${NOT_AN_AGE}`, "gu"), ownedByNoun: "after-aged" },
  { pattern: new RegExp(String.raw`\bages? of ${AGE_QUALIFIER}${AGE_NUMBER}${NOT_AN_AGE}`, "gu") },
  // "a girl of twelve", "a girl of about 12", "a girl around 13", "a girl, about twelve".
  { pattern: new RegExp(String.raw`\b${PERSON_NOUN} of ${AGE_QUALIFIER}${AGE_NUMBER}\b${NOT_AN_AGE}`, "gu") },
  { pattern: new RegExp(String.raw`\b${PERSON_NOUN},? (?:aged? |of |who is |who's |that is )?${APPROXIMATELY} ?${AGE_NUMBER}(?: ?ish)?\b${NOT_AN_AGE}`, "gu") },
  // "she is 12", "who is 14", "she's 15", "just turned 14", "turning thirteen", "she is 14 here". Not after a subject
  // that is plainly a quantity ("the score is 12", "it is 12").
  {
    pattern: new RegExp(
      String.raw`(?<!\b(?:it|that|this|there|which|what|score|price|cost|size|number|count|total|page|room|floor|step|day|hour|time|level|chapter|temperature|height|weight|length|width|distance|speed|rank|ratio|rating|volume|issue|episode|season) )(?:\b(?:is|was|are|were|am|be|been|being|turned|turns|turning|becomes|became|looks?|looking|appears?|appearing|seems?|seeming)(?: like| to be)?|(?<!\b(?:it|that|there|what|here|time))'s) ${AGE_QUALIFIER}${AGE_NUMBER}(?: ?ish)?(?=$| (?:years?|yrs?|now|today|this|and|but|or|yet|soon|next|last|in|on|at|with|tomorrow|already|so|too|here|there|lol|haha|though|tho|actually|ok|okay|btw|max|tops|at most|or so)\b)`,
      "gu"
    )
  },
  // "12 years", "fourteen summers" closing a term. Not a duration ("for 12 years", "in twelve years").
  {
    pattern: new RegExp(
      String.raw`(?<!\b(?:for|in|over|within|past|last|next|after|before|every|since|than|about|around|the|first|those|these|many|few|several|some) )\b${AGE_NUMBER} ?(?:years?|yrs?|summers|winters)$`,
      "gu"
    )
  },
  // "twelve and a half" as a whole term.
  { pattern: new RegExp(String.raw`^${AGE_NUMBER} and a (?:half|quarter)$`, "gu") },
  // An approximation standing as a whole term: "around 13", "roughly 14", "~13", "13-ish", "at thirteen". "at 3" is a
  // time, so "at" takes only a teen.
  {
    pattern: new RegExp(
      String.raw`^(?:${APPROXIMATELY}|age|aged|now|turning|turned) ?${AGE_NUMBER}(?: ?ish)?$|^${AGE_NUMBER} ?ish$|^at (?:age )?(1[3-7]|thirteen|fourteen|fifteen|sixteen|seventeen)$`,
      "gu"
    )
  },
  // "13ish", "fourteenish" anywhere; not a time ("at 8ish").
  { pattern: new RegExp(String.raw`(?<!\b(?:at|by|until|till|before|after|from|since|around|about) )\b${AGE_NUMBER} ?ish\b${NOT_AN_AGE}`, "gu") },
  // "a 14 year girl", "a 13 yr boy".
  { pattern: new RegExp(String.raw`\b${AGE_NUMBER} ?(?:years?|yrs?|y) (?=${PERSON_NOUN_ONLY}\b)`, "gu") },
  // "passes for 14", "could pass for fourteen", "ages 12 to 14", "aged between 12 and 14", "14 going on 15".
  { pattern: new RegExp(String.raw`\b(?:pass(?:es|ed|ing)?|could pass|can pass|would pass|might pass) (?:for|as) (?:an? )?${AGE_QUALIFIER}${AGE_NUMBER}\b${NOT_AN_AGE}`, "gu") },
  { pattern: new RegExp(String.raw`\b(?:ages?|aged) (?:between |from )?${AGE_NUMBER} (?:(?:to|and|through|or) )?${AGE_NUMBER}\b`, "gu") },
  { pattern: new RegExp(String.raw`\b${AGE_NUMBER} going on ${AGE_NUMBER}\b|\bgoing on ${AGE_NUMBER}(?=$| (?:now|soon|already|and|but|in|this|next)\b)`, "gu") },
  // "fourteen birthdays", "lived fourteen years", "fourteen years of life".
  { pattern: new RegExp(String.raw`\b${AGE_NUMBER} birthdays\b|\blived (?:for |just |only )?${AGE_NUMBER} years?(?=$| (?:so far|now|of (?:her|his|their) life)\b)|\b${AGE_NUMBER} years? of (?:life|age)\b`, "gu") },
  // "a decade old", "a decade-old girl" (a decade-old building is the building's).
  { pattern: /\b(?:a|one) (decade)(?: and (?:a half|\w+ years?))? ?old\b/gu, ownedByNoun: "after-old" },
  // "XII years old", "XIV yo". Roman numerals only next to the unit: alone, i, v and x are ordinary letters.
  { pattern: new RegExp(String.raw`\b(${Object.keys(ROMAN_NUMBERS).join("|")}) ?(?:years? (?:old|of age)|y/o|yo)\b`, "gu") },
  // "on her 13th birthday", "fourteenth birthday". An 18th birthday is an adult's.
  { pattern: new RegExp(String.raw`\b(\d{1,2}(?:st|nd|rd|th)|${Object.keys(SPELLED_ORDINALS).join("|")}) birthday\b`, "gu") },
  // "under 18", "below 18", "younger than 18", "not yet 18", "almost 18", "under the age of 18", "U18", "sub-18",
  // "<18", and the same bound in other languages ("moins de 18 ans", "unter 18", "minder dan 18", "18 yasindan kucuk").
  {
    pattern: new RegExp(
      String.raw`\b(?:under|below|younger than|less than|not yet|not quite|nearly|almost|just under|barely|shy of|sub|moins de|menos de|menor(?:es)? de|unter|minder dan|onder de|jonger dan|ponizej|sotto i|meno di|abaixo de|di ?bawah) (?:the )?(?:age (?:of )?)?${AGE_NUMBER}${NOT_AN_AGE}|\bu ?(1[0-8])\b|(?:^|[^a-z0-9])< ?(1[0-8]|[1-9])\b|\b(1[0-8]|[1-9]) (?:yasindan kucuk\w*|ev alatt\w*)`,
      "gu"
    ),
    limit: 19
  },
  // An upper bound is a minor's age when the bound is: "no older than 13", "at most 14", "13 tops", "13 max", and
  // "she'll turn 14", where the age now is below the one named.
  {
    pattern: new RegExp(
      String.raw`\b(?:(?:no|not|never) (?:older than|over|above|past)|max(?:imum)? age(?: of)?) ${AGE_NUMBER}${NOT_AN_AGE}|\b${AGE_NUMBER} (?:at the oldest|or (?:younger|under|below|less)|and (?:under|younger|below))\b|\b(?:isn't|isnt|is not|wasn't|was not|not) (?:even )?${AGE_NUMBER} yet\b|\bat most ${AGE_NUMBER}(?=$| (?:years?|yrs?|or so|tops)\b)|\b${AGE_NUMBER} (?:at most|tops|max(?:imum)?|at the most)$|\b(?:will|'ll|to|gonna|about to|shall|would|should|going to|set to|due to) turn ${AGE_NUMBER}\b${NOT_AN_AGE}(?! (?:pages?|corners?|left|right|degrees?)\b)`,
      "gu"
    ),
    limit: 19
  },
  // Numeric ages in the languages above: "de 12 años", "14 ans", "14-jähriges", "di 14 anni", "14 sai", "14 jaar oud",
  // "14-letnia", "14 yasindaki", "14 eves", "14 anys", "14 godina", "14 taon".
  {
    pattern: new RegExp(
      String.raw`\b((?<![\d.,])\d{1,2}) ?(?:anos|anitos|aninhos|ans|jahre|jahren|jahrig\w*|anni|enne|tahun|jaar|jarig\w*|lat|lata|let|leta|lety|letni\w*|letn\w*|a?ar(?=$| (?:gammal|gamla|gammel|gamle|gml)\b)|a?aring\w*|a?arig\w*|vuotia\w*|ani|eves|godin[aeu]|anys|taon|taong gulang|yas\w*|sai|sui|sal|tuoi)\b`,
      "gu"
    )
  },
  // Italian "-enne" ages: "quattordicenne".
  { pattern: /\b((?:undic|dodic|tredic|quattordic|quindic|sedic|diciassett|dec)enn[ei])\b/gu },
  ...FOREIGN_SPELLED_AGES.map(({ numbers, units, joined }) => ({
    pattern: new RegExp(String.raw`\b(${numbers}) ?(?:${units})\b${joined ? String.raw`|\b(${numbers})(?:${joined})\b` : ""}`, "gu")
  })),
  // "168 months old", "4000 days old": counted in another unit, and owned by a person (a 3-day-old loaf is not).
  ...([["months?|mos?", 12], ["weeks?|wks?", 52], ["days?", 365]] as const).map(([unit, perYear]) => ({
    pattern: new RegExp(
      String.raw`(?:^|\b${PERSON_NOUN},? (?:is |was |who is |aged )?|\b(?:she|he) (?:is|was) |\b(?:she|he)'s )(\d{1,4}) ?(?:${unit}) ?old(?=$| ${PERSON_NOUN}\b)|\b(\d{1,4}) ?(?:${unit}) ?old ${PERSON_NOUN}\b`,
      "gu"
    ),
    perYear
  }))
];

/**
 * Ages that only read as ages next to a person, because the same shape is ordinary elsewhere: "Yuna 14" and
 * "Yuna (14)" next to "iPhone 14" and "room (14)", "14F" and "f/14" next to seat 14F, an F-14 and f/8. A cut's figures
 * are people by definition; a free text needs a person word of its own (see classifyImageScene). The word before the
 * number may not be a counter, a product or a function word, and the gendered forms read only teens without a slash
 * (1F to 9F are floors, F1 is a race) and never after aperture or seat words.
 */
const CONTEXT_AGE_NUMBER = String.raw`(?:[2-9]|1[0-7]|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)`;
const NOT_A_NAME_WORDS = new Set(
  (
    "a an the and or but of in on at to by for from with as is are was were be been am it its this that no not so do did " +
    "has had have can will just too up out off over into than then also all any per via vs versus x times plus minus " +
    "under about around only nearly almost aged age ages top bottom number no. num page pages chapter chap part vol " +
    "volume issue episode ep season act scene take track disc level floor room suite unit apt apartment flat block lot " +
    "building gate seat row aisle deck cabin berth car coach platform terminal bus route line exit lane road street ave " +
    "highway interstate pier dock station stop stage step phase round game match set inning quarter period size grade " +
    "year class day week month hour minute mile km team squad jersey shirt model type version ver series gen mk mark " +
    "iphone ipad ipod galaxy pixel windows ios android macos xbox playstation ps switch apollo formula chanel boeing " +
    "airbus photo image pic picture fig figure item ref note option choice variant sample look camera lens f iso table " +
    "count total score won lost scored got made took bought sold picked ranked rated counted needs wants wins equals " +
    "ocean's catch area district zone sector ward channel radio studio hall lobby wing tower villa cottage psalm verse " +
    "article section clause rule law amendment movement symphony op opus slide panel frame shot cut tab window"
  ).split(" ")
);
// Digits only after a name: spelled, the same shape is "trapped between two" or "a table for four".
const NAME_AGE_PATTERN = /(?:^|\s)([a-z][a-z'.]+):? \(?([2-9]|1[0-7])\)?$/u;
const PARENTHESISED_AGE_PATTERN = new RegExp(String.raw`(?:^|\s)(?:([a-z][a-z'.]*) ?)?\((?:age |aged )?(${CONTEXT_AGE_NUMBER})\)`, "gu");
const GENDERED_AGE_PATTERN = new RegExp(
  String.raw`(?:^|[^a-z0-9/.@])(?:(?<before>[a-z']+) )?\(?(?:[fm] ?/ ?(?<a>[2-9]|1[0-7])|(?<b>[2-9]|1[0-7]) ?/ ?[fm]|f(?<c>1[0-7])|(?<d>1[0-7])f)\)?(?![a-z0-9/]|\.[a-z0-9])(?! (?:fighters?|jets?|tomcats?|eagles?|falcons?|raptors?|hornets?|aircraft|planes?|floor|lobby|stop|aperture|lens|race|car|tributo)\b)`,
  "gu"
);
const CAMERA_CONTEXT_PATTERN =
  /\b(?:\d+ ?mm|lens|lenses|aperture|iso ?\d+|shutter|bokeh|depth of field|exposure|f ?stops?|focal|canon|nikon|sony|fujifilm|fuji|leica|hasselblad|portra|kodak|ilford|cinestill|film stock|35mm|medium format|dslr|mirrorless)\b/u;
// A person nearby, for the contextual ages: anyone the readings of FEMALE_WORD_PATTERN / MALE_WORD_PATTERN miss.
const PERSON_CONTEXT_PATTERN = /\b(?:persons?|people|someone|somebody|they|them|portrait|selfie|child(?:ren)?|kids?|teens?|students?|pupils?|models?|1girl|1boy|1other)\b/u;
// Where a comma splits a phrase the minor reading needs whole: an adjective list before the noun, a date.
const SPLIT_PHRASE_OPENER_PATTERN = /(?:^|\s)(?:little|small|tiny|wee|young|youthful|petite)$|\b(?:born|dob|birth|birthday|class of|graduat\w*)\b/u;
// "Her seat on the flight is 14F": a seat, a row, a room or a floor owns the number however close a person stands.
const NOT_A_GENDERED_AGE_CONTEXT =
  /\b(?:seats?|rows?|flights?|gates?|cabins?|aisles?|boarding|tickets?|rooms?|floors?|apartments?|apt|units?|suites?|platforms?|terminals?|lockers?|buildings?|blocks?|lanes?|bus(?:es)?|trains?|coach(?:es)?|decks?|berths?|parking|levels?|storeys?|stories)\b/u;
const NOT_A_GENDERED_AGE_BEFORE = new Set(
  "at aperture lens stopped stop fstop shot open seat row gate room floor level on to the apt unit suite flight deck cabin berth car coach platform terminal aisle block building lot bus b bay dock".split(" ")
);

/**
 * Every minor age a contextual form states in the term (see CONTEXT_AGE_NUMBER). Next to camera words, "f/8" and
 * "f/14" are apertures.
 */
function readContextualAges(term: string, camera: boolean): number[] {
  const ages: number[] = [];
  const named = NAME_AGE_PATTERN.exec(term);
  if (named && !NOT_A_NAME_WORDS.has(named[1])) {
    ages.push(parseAgeToken(named[2]));
  }
  for (const match of term.matchAll(PARENTHESISED_AGE_PATTERN)) {
    if (!match[1] || !NOT_A_NAME_WORDS.has(match[1])) {
      ages.push(parseAgeToken(match[2]));
    }
  }
  for (const match of NOT_A_GENDERED_AGE_CONTEXT.test(term) ? [] : term.matchAll(GENDERED_AGE_PATTERN)) {
    const groups = match.groups ?? {};
    if ((groups.before && NOT_A_GENDERED_AGE_BEFORE.has(groups.before)) || (camera && /^\(?f/u.test(match[0].trim().split(" ").pop() ?? ""))) {
      continue;
    }
    ages.push(parseAgeToken(groups.a ?? groups.b ?? groups.c ?? groups.d));
  }
  return ages.filter((age) => age > 0 && age < 18);
}

// Nouns an "N-year-old" may belong to instead of a person: spirits, cheese, pets, machines, buildings. Counted only
// when the age sits right before the noun, no person noun stands right before the age, and the noun is followed by
// the end of the term or a word that cannot start a new person ("a 12-year-old cat girl" is still a girl).
const NON_PERSON_AGED_NOUN = String.raw`(?:whiske?y|scotch|bourbon|single malt|malt|wine|port|rum|brandy|cognac|sake|soju|cheese|parmesan|cheddar|gouda|vinegar|ham|beef|steak|kimchi|dog|puppy|cat|kitten|horse|pony|pet|parrot|turtle|tortoise|goldfish|golden retriever|retriever|labrador|poodle|corgi|shiba(?: inu)?|car|truck|bike|bicycle|motorcycle|scooter|van|laptop|computer|pc|phone|smartphone|iphone|tablet|camera|tv|building|house|apartment|tree|bonsai|company|business|brand|shop|store|restaurant|cafe|bakery|bar|tradition|recipe|song|album|film|movie|photo|photograph|bottle|cask|barrel|guitar|piano|violin|sofa|couch|chair|table|desk|watch|coat|jacket|bag|handbag|sweater|jeans|boots|shoes)s?`;
const NON_PERSON_FOLLOWER = String.raw`(?=$| (?:on|in|at|by|from|with|and|or|is|was|that|which|of|for|to|under|near|beside|next|naps?|napping|sleeps?|sleeping|sits?|sitting|lies|lying|runs?|running|glows?|glowing|poured|served|parked|bottle|glass|cask|barrel)\b)`;
const NON_PERSON_AFTER_OLD_PATTERN = new RegExp(String.raw`^ (?:(?!${PERSON_NOUN}\b)[a-z']+ ){0,2}${NON_PERSON_AGED_NOUN}${NON_PERSON_FOLLOWER}`, "u");
const NON_PERSON_AFTER_AGED_PATTERN = new RegExp(String.raw`^ ?(?:years?|yrs?|months?) (?:(?!${PERSON_NOUN}\b)[a-z']+ ){0,2}${NON_PERSON_AGED_NOUN}${NON_PERSON_FOLLOWER}`, "u");
const PERSON_BEFORE_PATTERN = new RegExp(String.raw`\b${PERSON_NOUN}\s*$`, "u");

// A number standing alone as a whole tag or comma segment is an appositive age ("Yuna, 14, in a bikini"); nothing
// else a caption says is a bare number under 18.
const BARE_AGE_PATTERN = new RegExp(
  String.raw`^(?:[1-9]|1[0-7]|${Object.keys(SPELLED_NUMBERS).filter((word) => SPELLED_NUMBERS[word] < 18 && word !== "dozen").join("|")})$`,
  "u"
);

// "born in 2014", "born in March 2012", "born on May 3 2013", "DOB 2013", "date of birth: 2012", "nacida en 2012",
// "geboren 2012": a birth word, then only date words (a month, a day, "on", "in", a separator) before the year. A
// year that leaves the person 18 or younger today is a minor's: someone born 18 years ago may not have had the
// birthday yet.
const MONTH_WORD = String.raw`(?:jan|feb|fev|mar|maa|apr|abr|avr|may|mai|mag|mei|maio|jun|jui|giu|jul|lug|aug|ago|aou|sep|set|oct|okt|ott|out|nov|dec|dez|dic|dek|sty|lut|kwi|maj|cze|lip|sie|wrz|paz|lis|gru|led|uno|bre|dub|kve|cer|cec|srp|zar|rij|pro|ocak|subat|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik)[a-z]*`;
const BIRTH_WORD = String.raw`(?:born|b\.|dob|d\.o\.b\.?|date of birth|birth ?(?:year|date|day)|year of birth|birthyear|nacid[oa]s?|nascid[oa]s?|nat[oa]|nee|geboren|geb\.?|urodzon[aey]|narozen[aey]?|dogum(?:lu)?|dogdu|fodd|fodt|fodda|syntynyt|nascut[a]?|szuletett|rodjen[ao]?|lahir)`;
const DATE_FILLER = String.raw`(?:[\s:,/.-]*(?:on|in|the|year|of|at|around|circa|ca|early|late|mid|spring|summer|autumn|fall|winter|en|em|el|le|il|im|am|w|v|na|di|tahun|ano|anno|annee|jahr|i|a|o|del|de|da|du|\d{1,2}(?:st|nd|rd|th|er|e)?\.?|${MONTH_WORD})(?![a-z]))*`;
const BIRTH_YEAR_PATTERN = new RegExp(String.raw`\b${BIRTH_WORD}${DATE_FILLER}[\s:,/.-]*((?:19|20)\d\d|'\d\d)'?s?\b`, "gu");
// "class of 2031", "graduating in 2032", "will graduate high school in 2031": a school class that has not graduated
// yet, so someone still in school. A college class is excepted (its members are adults).
const GRADUATION_YEAR_PATTERN =
  /\b(?:(?:class|batch|cohort|year group)(?: of)?|graduat(?:es?|ing|ion)(?: (?:high school|school|from \w+(?: \w+)?))?(?: (?:in|year|class of))?|will graduate(?: \w+){0,3}? in)[\s:]*(20\d\d|'\d\d)\b/gu;
const COLLEGE_CONTEXT_PATTERN =
  /\b(?:college|university|uni|grad school|graduate school|law school|med(?:ical)? school|mba|phd|bachelor'?s?|master'?s?|alumn(?:i|a|us|ae)|degree)\b/u;

// ---------------------------------------------------------------------------------------------------------------
// Action / fantasy / romance
// ---------------------------------------------------------------------------------------------------------------

// Dynamic motion only. Everyday actions (walking, reading, reaching for a shelf) are the static majority of
// cuts and stay portrait/full_body. Garments named after an activity are filtered where this rule is used.
const ACTION_RULE: TermRule = {
  pattern:
    /\b(?:fighting|fight|fights|battle|battling|combat|brawl|brawling|punch|punching|punches|kick|kicking|kicks|flying kick|roundhouse kick|uppercut|sword ?fight(?:ing)?|clash|clashing|duel|dueling|slashing|stabbing|attack|attacking|swinging|shooting|firing|aiming|gunfire|muzzle flash|gunshot|explosions?|exploding|motion blur|speed lines|motion lines|action lines|dodging|dodge|evading|chasing|being chased|fleeing|struggling|wrestling|grappling|tackle|tackling|strangling|chokehold|choking|slapping|running|sprinting|jumping|leaping|mid ?air|falling|tripping|stumbling|dancing|breakdancing|playing (?:soccer|football|basketball|tennis|volleyball|baseball|badminton|table tennis|ping pong|hockey|rugby|sports)|dribbling|dunking|slam dunk|swimming|surfing|skateboarding|skiing|snowboarding|parkour|martial arts|kung fu|karate|boxing|kickboxing)\b/u,
  except:
    /\bfruit punch\b|\bpunch bowl\b|\bkick scooter\b|\bheart attack\b|\bshooting stars?\b|\bphoto ?shoot(?:ing)?\b|\bswinging legs\b|\brunning (?:water|mascara|makeup|track)\b|\bfalling (?:petals?|leaves|leaf|snow|snowflakes|feathers?|stars?|cherry blossoms|sakura|rain|water|confetti|money|cards?|ash|embers|sparks|flowers?|hearts|asleep|in love)\b|\bslow danc(?:e|ing)\b|\bswimming (?:pool|trunks|costume|lessons?)\b|\bboxing (?:gloves|ring)\b/u
};

// Non-human or supernatural traits, including the ones carried by a character's saved identity.
const FANTASY_RULE: TermRule = {
  pattern:
    /\b(?:animal ears|(?:cat|fox|wolf|dog|rabbit|bunny|bear|mouse|horse|cow|sheep|goat|tiger|lion|leopard|raccoon|tanuki|deer|squirrel|bat|monkey|hamster|ferret|otter|panda|fennec|lop) ears|kemonomimi|nekomimi|horns?|demons?|demonic|devil|angels?|wings?|winged|tails?|elf|elves|pointy ears|elf ears|halo|monsters?|dragons?|slimes?|tentacles?|magic|magical|spells?|spellcasting|glowing eyes|fairy|fairies|mermaids?|merman|colored skin|(?:blue|green|purple|red|grey|gray) skin|vampires?|succubus|incubus|werewol(?:f|ves)|kitsune|oni|centaur|lamia|harpy|ghosts?|undead|zombies?|robots?|cyborgs?|androids?|mecha|unicorns?|pegasus|phoenix|levitation|levitating|summoning|witch|wizard|sorcerer|sorceress|necromancer|(?:cat|fox|wolf|dog|rabbit|dragon|demon|slime|snake|spider|bee|plant|mouse|bear|tiger|lion|sheep|goat|deer|squirrel|raccoon) ?(?:girls?|boys?))\b/u,
  // Costume, props and words that merely share a fantasy noun: a cat-ear headband, fairy lights, twin tails,
  // a wing collar, a mermaid dress, angel sleeves, a halo braid, a dragon print, chicken wings, a French horn.
  except:
    /\b(?:fake|headband|hairband|headwear|headphones|headset|hood|hat|cap|beanie|ornament|hairclip|hair clip|print|pattern|plush|plushie|stuffed|keychain|costume|cosplay|tattoo|statue|figurine|poster|sticker|doll|toy|mascot|logo)\b|\b(?:fairy|tail|string) lights?\b|\bfairy (?:tale|kei)\b|\bwings? (?:collar|sleeves?)\b|\bangel (?:sleeves?|hair)\b|\bhalo braid\b|\bmermaid (?:dress|skirt|gown|silhouette|hair)\b|\bmonster energy\b|\bdragon (?:fruit|boat)\b|\b(?:chicken|buffalo|airplane|plane|aircraft|bird) wings?\b|\bfrench horn\b|\b(?:twin|pony|side|pig|fish) ?tails?\b|\btail ?coat\b|\brobot vacuum\b/u
};

// A costume marker anywhere on a figure turns that figure's bare "rabbit ears" / "tail" into wardrobe.
const FAKE_FANTASY_TRAITS: Array<{ fake: RegExp; trait: RegExp }> = [
  {
    fake: /\bfake (?:animal )?(?:\w+ )?ears\b|\banimal ear (?:headwear|headband|hairband)\b|\bplayboy bunny\b|\bbunny (?:suit|girl|costume)\b/u,
    trait: /\bears\b/u
  },
  { fake: /\bfake (?:\w+ )?tails?\b/u, trait: /\btails?\b/u },
  { fake: /\bfake (?:\w+ )?horns?\b/u, trait: /\bhorns?\b/u },
  { fake: /\bfake (?:\w+ )?wings?\b/u, trait: /\bwings?\b/u }
];

// Romance tags that name two people by themselves (a POV kiss is still a kiss).
const ROMANCE_PAIR_RULE: TermRule = {
  pattern:
    /\b(?:(?<!(?:blowing|air) )kiss(?:es|ing)?(?! marks?)|french kiss|deep kiss|forehead kiss|cheek kiss|imminent kiss|after kiss|eskimo kiss|holding hands|hand holding|holding another's hand|couple|head on another's shoulder|lap pillow|princess carry|carrying (?:another|person|someone)|piggyback|hug from behind|arm hug|cheek-to-cheek|cheek to cheek|nuzzling)\b/u
};

// Romance tags that only mean intimacy when a second person is in the cut ("hug" a pillow, "cuddling" a cat).
const ROMANCE_NEEDS_PAIR_RULE: TermRule = {
  pattern:
    /\b(?:hug|hugging|embrace|embracing|cuddling|cuddle|snuggling|head on shoulder|leaning on (?:another|person|someone)|interlocked fingers|arms around (?:another's )?(?:neck|waist)|caressing|headpat|head pat|patting head)\b/u,
  except:
    /\b(?:own|self|object|pillow|doll|stuffed|plush|plushie|toy|book|knees|legs|tail|cushion|bag|teddy|bear|blanket|tree|cat|dog|animal|pet|instrument|guitar)\b/u
};

// ---------------------------------------------------------------------------------------------------------------
// No humans: object vs scenery
// ---------------------------------------------------------------------------------------------------------------

const NO_HUMANS_MARKER_PATTERN = /^(?:no humans|no people|no person|nobody|scenery|landscape)$/u;

// Tags that make an object the subject regardless of what else is in frame.
const OBJECT_FOCUS_PATTERN = /\b(?:still life|object focus|food focus|item focus|product focus|product shot)\b/u;

const OBJECT_RULE: TermRule = {
  pattern:
    /\b(?:food|foods|cakes?|cheesecake|cupcakes?|pastry|pastries|desserts?|sweets|bread|croissants?|sandwich(?:es)?|pizza|sushi|ramen|noodles|bento|meals?|dish(?:es)?|breakfast|lunch|dinner|snacks?|cookies?|macarons?|chocolates?|candy|fruits?|strawberr(?:y|ies)|ice cream|parfait|coffee|latte|cappuccino|espresso|tea|teacups?|teapot|cups?|mugs?|drinks?|beverages?|cocktails?|juice|letters?|envelopes?|notebooks?|diary|books?|phone screen|smartphones?|cellphones?|mobile phones?|phones?|gifts?|gift box|presents?|bouquets?|flower bouquet|rings?|jewelry|jewellery|necklaces?|bracelets?|earrings?|perfume|photographs?|polaroids?|wine glass|bottles?)\b/u,
  // Places named after an object: a coffee shop is a location, a coffee table is furniture.
  except:
    /\b(?:coffee|tea|cake) (?:shop|house|room|table|bar|stand|ceremony|machine|maker)\b|\bboxing ring\b|\bring light\b/u
};

// Locations, furniture, weather and time of day. Counted against object nouns to tell an object shot from an
// interior that happens to contain one.
const SETTING_PATTERN =
  /\b(?:indoors|outdoors|interior|exterior|architecture|room|bedroom|living room|kitchen|bathroom|classroom|library|office|caf[eé]|restaurant|bar|shop|store|street|alley|road|city|town|village|buildings?|house|apartment|park|garden|forest|woods|mountains?|beach|ocean|sea|lake|river|sky|clouds?|sunset|sunrise|rooftop|hallway|corridor|station|platform|bridge|field|meadow|snow|rain|night|evening|dusk|dawn|windows?|bookshelf|bookshelves|shelf|shelves|desk|table|bed|sofa|couch|chair|curtains|floor|wall|stairs|door|ceiling|counter)\b/u;

// Explicitly wide compositions are scenery even when they list a prop.
const WIDE_COMPOSITION_PATTERN = /^(?:scenery|landscape|cityscape|wide shot|establishing shot|panorama)$/u;

// ---------------------------------------------------------------------------------------------------------------
// Subjects
// ---------------------------------------------------------------------------------------------------------------

const SUBJECT_COUNT_PATTERN = /^(\d+)\+?\s?(girls?|boys?|others?)$/u;
const MULTIPLE_SUBJECTS_PATTERN = /^multiple (girls|boys|others|people|persons)$/u;

// ---------------------------------------------------------------------------------------------------------------
// Canonical text
// ---------------------------------------------------------------------------------------------------------------

/** Soft hyphen, zero-width space/joiners, word joiner, BOM, variation selectors, bidi controls, Hangul fillers. */
const DEFAULT_IGNORABLE_PATTERN = /\p{Default_Ignorable_Code_Point}/gu;
/**
 * Combining marks on anything but a letter of another script: accents, the strike-through and overlay marks used to
 * break up a word, and the keycap and accent marks that dress up a digit ("1⃣2⃣", "12́") while the encoder still
 * reads the number. A stray Hangul tone mark (U+302E/F) after a space goes too, and can no longer make a Latin
 * segment look Korean. On kana or Devanagari the mark is part of the letter, and that text is unverifiable anyway.
 */
const COMBINING_MARK_PATTERN = /(?<=^|[^\p{L}]|\p{Script=Latin})\p{M}+/gu;
/**
 * Letter-like symbols with no compatibility decomposition: negative circled and negative squared capitals and the
 * regional indicators. They render as letters ("🅽🅰🅺🅴🅳", a flag pair spelling "NUDE"), and NFKC leaves them alone.
 */
const LETTER_LIKE_SYMBOL_PATTERN = /[\u{1F130}-\u{1F149}\u{1F150}-\u{1F169}\u{1F170}-\u{1F189}\u{1F1E6}-\u{1F1FF}]/gu;
/**
 * Latin letters NFD does not decompose. They are Latin script (Øresund, Königstraße, Kadıköy, Łódź, Đà Nẵng), and
 * left in place they made ordinary place names unverifiable.
 */
const LATIN_LETTER_FOLDS: Readonly<Record<string, string>> = {
  ø: "o", Ø: "O", ß: "ss", ẞ: "SS", ł: "l", Ł: "L", đ: "d", Đ: "D", ð: "d", Ð: "D", ı: "i", œ: "oe", Œ: "OE",
  æ: "ae", Æ: "AE", þ: "th", Þ: "Th", ħ: "h", Ħ: "H", ŧ: "t", Ŧ: "T", ŀ: "l", Ŀ: "L", "ʻ": "'"
};
const LATIN_LETTER_FOLD_PATTERN = new RegExp(`[${Object.keys(LATIN_LETTER_FOLDS).join("")}]`, "gu");
/**
 * The few words whose ñ is all that separates them from an ordinary one: "niña"/"niño" (a girl, a boy) fold to the
 * common names Nina and Nino, "coño" to a cone. Their ñ folds to "ny" instead, in every text the floor or the builder
 * reads, so the lexicon can tell them apart wherever they turn up; every other ñ folds to n.
 */
const DISTINGUISHING_ENYE_PATTERN = /(?<!\p{L})(ni|co)[ñÑ](?=(?:a|o|as|os|ita|itas|ito|itos|ez)?(?!\p{L}))/giu;
/** A letter outside a–z, or a digit outside 0–9, once the text is canonical. */
const UNVERIFIABLE_CHARACTER_PATTERN = /[^\P{L}A-Za-z]|[^\P{N}0-9]/u;
const NEGATIVE_WEIGHT_OPENER_PATTERN = /^-\d+(?:\.\d+)?\s*::/u;
const HANGUL_PATTERN = /\p{Script=Hangul}+/gu;
/**
 * A sentence ends at . ! ? — but not after an initial or a title ("Philip K. Dick", "jr. high", "Dr. Kim"), where
 * splitting tore the phrase in two and each half was read without the other.
 */
const SENTENCE_BREAK_PATTERN = /\n+|(?<=[.!?。])(?<!\b(?:[a-z]|jr|sr|mr|mrs|ms|dr|st|vs|no|etc|approx|e\.g|i\.e)\.)\s+/iu;

/**
 * The form every lexicon here reads, and the form the photo prompt builder renders: NFKC (fullwidth letters and
 * digits, ligatures, circled and mathematical letters become plain ones), default-ignorable code points removed
 * (soft hyphen, zero-width characters, variation selectors), letter-like symbols read as letters, every combining
 * mark removed and the undecomposable Latin letters folded, typographic quotes made ASCII. Case and hyphens are
 * kept: this is also display text.
 */
export function canonicalizeSceneText(text: string | undefined): string {
  if (typeof text !== "string" || !text) {
    return "";
  }
  return foldMarks(
    text
      .normalize("NFKC")
      .replace(DEFAULT_IGNORABLE_PATTERN, "")
      .replace(LETTER_LIKE_SYMBOL_PATTERN, readLetterLikeSymbol)
      .replace(LATIN_LETTER_FOLD_PATTERN, (letter) => LATIN_LETTER_FOLDS[letter] ?? letter)
      .replace(DISTINGUISHING_ENYE_PATTERN, (_word, stem: string) => `${stem}ny`)
  )
    .replace(/[‘’‚‛ʼ′]/gu, "'")
    .replace(/[“”„‟″]/gu, '"');
}

/**
 * The text with its Hangul taken out, for the floor to read what is left. Korean is dropped from photo text by
 * contract, but a segment that mixes it with Latin words still carries those words, and skipping the whole segment
 * let "A naked woman" through behind a single Hangul mark.
 */
export function stripHangul(text: string): string {
  return text.replace(HANGUL_PATTERN, " ").replace(/\s+/gu, " ").trim();
}

/**
 * True when the text still has a letter outside a–z (or a digit outside 0–9) after canonicalisation: kana, Hangul,
 * CJK, Cyrillic or Greek look-alikes ("nаked", whose second letter is Cyrillic), IPA small capitals. No lexicon
 * here can say what such text means, while the photo encoders (Qwen3 / Qwen3-VL) read most of it, so it is never
 * trusted.
 */
export function hasUnverifiableScript(text: string | undefined): boolean {
  return UNVERIFIABLE_CHARACTER_PATTERN.test(canonicalizeSceneText(text));
}

/**
 * Free text split the way the photo prompt builder renders it: sentences, each a list of comma segments,
 * canonical and without the closing punctuation. The builder keeps or drops whole segments and the safety floor
 * reads the same segments, so whatever ends up in a photo prompt has been read segment for segment.
 */
export function splitSceneSentences(text: string | undefined): string[][] {
  return canonicalizeSceneText(text)
    .split(SENTENCE_BREAK_PATTERN)
    .map((sentence) =>
      sentence
        .split(/[,、]/u)
        .map((segment) => segment.trim().replace(/[\s.!?。]+$/u, "").trim())
        .filter(Boolean)
    )
    .filter((segments) => segments.length > 0);
}

/** True for a garment that covers the body (see COVERING_GARMENT_PATTERN), weights and case ignored. */
export function isCoveringGarmentTerm(tag: string): boolean {
  return COVERING_GARMENT_PATTERN.test(normalizeSceneTag(tag));
}

/**
 * True for a term that must never become a photo negative: anything worn on or wrapped around the body, a word
 * that keeps a picture covered, censored or safe, a word that makes the subject an adult, or an adult build. A
 * negative of any of them steers the sampler toward the undressed or younger figure the positive prompt was never
 * allowed to describe. Accessories (hat, glasses, shoes) and quality terms (blurry, lowres, watermark) are not.
 */
export function isPhotoNegativeSafeguardTerm(tag: string): boolean {
  const term = normalizeSceneTag(tag);
  return (
    COVERING_GARMENT_PATTERN.test(term) ||
    WORN_CLOTHING_PATTERN.test(term) ||
    SAFEGUARD_WORD_PATTERN.test(term) ||
    ADULTHOOD_WORD_PATTERN.test(term) ||
    ADULT_BUILD_WORD_PATTERN.test(term)
  );
}

/**
 * Strips NovelAI / Stable Diffusion weight syntax, canonicalises, reads hyphens and underscores as spaces and
 * lowercases. `1.5::sex_from_behind::` → `sex from behind`, `ｓｃｈｏｏｌ-ｕｎｉｆｏｒｍ` → `school uniform`.
 */
export function normalizeSceneTag(tag: string): string {
  let value = canonicalizeSceneText(typeof tag === "string" ? tag : "");
  // Weight syntax nests (`1.2::{nude}::`, `{(nude:1.3)}`), so peel until nothing changes. Peeled before the
  // hyphens go, or a negative weight's minus sign would be read as a word break.
  for (let pass = 0; pass < 4; pass += 1) {
    const peeled = peelWeightSyntax(value);
    if (peeled === value) {
      break;
    }
    value = peeled;
  }
  // Lowercasing can itself add a mark ("İ" → "i̇"), so the fold runs once more on the result.
  return foldMarks(
    value
      .replace(/[_\p{Pd}]/gu, " ")
      .replace(/\s+/gu, " ")
      .trim()
      .replace(/[.!?。]+$/u, "")
      .trim()
      .toLowerCase()
  );
}

function foldMarks(text: string): string {
  return text.normalize("NFD").replace(COMBINING_MARK_PATTERN, "").normalize("NFC");
}

function readLetterLikeSymbol(symbol: string): string {
  const code = symbol.codePointAt(0) ?? 0;
  const base = code >= 0x1f1e6 ? 0x1f1e6 : code >= 0x1f170 ? 0x1f170 : code >= 0x1f150 ? 0x1f150 : 0x1f130;
  return String.fromCharCode(0x61 + code - base);
}

function peelWeightSyntax(value: string): string {
  return (
    value
      .trim()
      // NovelAI numeric emphasis (`1.5::tag::`, `-1::tag::`, `::tag::`), including the halves a comma split leaves
      // when one weight wraps several tags (`1.2::nude` … `sweat::`). A negative weight is stripped like any
      // other: the photoreal prompt builder drops weights too, so "less of this" would arrive as "this".
      .replace(/^-?\d+(?:\.\d+)?\s*::/u, "")
      .replace(/^::/u, "")
      .replace(/::$/u, "")
      // Stable Diffusion emphasis `(tag:1.2)` and `((tag))`. A bracketed qualifier inside a tag
      // (`hatsune miku (cosplay)`) does not wrap the whole tag and is left alone.
      .replace(/^\(\s*([^()]*?)\s*:\s*-?\d+(?:\.\d+)?\s*\)$/u, "$1")
      .replace(/^\((.*)\)$/u, "$1")
      // NovelAI brace emphasis `{tag}` / `[tag]` at any depth.
      .replace(/[{}[\]]/gu, "")
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Disguised words
// ---------------------------------------------------------------------------------------------------------------

const LEET_LETTERS: Readonly<Record<string, string>> = {
  "0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g", "@": "a", $: "s", "€": "e", "!": "i", "+": "t"
};
const LEET_CHARACTER_PATTERN = /[0-9@$€!+]/gu;
const MASK_CHARACTER_PATTERN = /[*#]/u;
const MASK_VOWELS = ["a", "e", "i", "o", "u"];
const MASK_LETTERS = Array.from("abcdefghijklmnopqrstuvwxyz");
// Letters English almost never doubles. A doubled one ("nuude", "naaked") is a stretched word; a doubled "o" or "e"
// is just "food" or "teen", and collapsing it would rewrite the neighbour a false-friend guard looks at.
const RARE_DOUBLE_PATTERN = /([aiuyhjkqvwx])\1+/u;
// Words a split-up fragment may never be joined to: joining them reads "the pen is" or "an us" as something else.
const JOIN_STOPWORDS = new Set(
  (
    "a an the and or but if of in on at to by for from with as is are was were be am it its he she we they you i me my " +
    "his her him us our your their them this that no not so do did has had can will just too up out off over into " +
    "than then also all any one two own per via vs ok oh hi yes ex mr ms dr st jr sr ft cm mm " +
    "km kg lb tv pc id ai uk eu vr ar hd go new old hot big red sun sea sky car bus top"
  ).split(" ")
);

/**
 * Symbols that stand in for a letter they look like ("se×", "nud℮", "c∪nt", "p°rn", "t¡ts", "l○li", "ch¡ld", "|\|aked").
 * They are symbols, not letters, so they never trip the unverifiable-script lock, and the encoder still reads the
 * word. Mapped for floor matching only: the photo prompt keeps "25°C".
 */
const LOOK_ALIKE_SEQUENCES: ReadonlyArray<[RegExp, string]> = [
  [/\|\\\||\/\\\//gu, "n"],
  [/\|\\\/\||\/\\\/\\/gu, "m"],
  [/\/\\/gu, "a"],
  [/\\\//gu, "v"],
  [/\|</gu, "k"],
  [/\|\)/gu, "d"],
  [/\(\)/gu, "o"]
];
const LOOK_ALIKE_LETTERS: Readonly<Record<string, string>> = {
  "×": "x", "✕": "x", "✖": "x", "✗": "x", "✘": "x", "⨯": "x", "⤫": "x", "⤬": "x",
  "∈": "e", "∊": "e", "℮": "e", "€": "e", "∃": "e", "∋": "e", "∪": "u", "⋃": "u", "∩": "n", "⋂": "n", "∂": "d",
  "°": "o", "○": "o", "•": "o", "◦": "o", "∘": "o", "⚬": "o", "◯": "o", "⭘": "o", "●": "o", "⚪": "o", "⊙": "o",
  "⊚": "o", "∅": "o", "¡": "i", "†": "t", "∀": "a", "∆": "a", "∇": "v", "√": "v", "¢": "c", "©": "c", "®": "r",
  "¥": "y", "§": "s"
};
const LOOK_ALIKE_LETTER_PATTERN = new RegExp(`[${Object.keys(LOOK_ALIKE_LETTERS).join("")}]`, "gu");
const PIPE_LOOK_ALIKES: Readonly<Record<string, string>> = { "|\\/|": "m", "|\\|": "n", "|<": "k", "|)": "d" };
const PIPE_LOOK_ALIKE_PATTERN = /\|\\\/\||\|\\\||\|<|\|\)/gu;

/**
 * Other readings of a canonical term for the safety floor: the ways a word is disguised so that a lexicon misses it
 * while a photo encoder still reads it. Each rewrite touches only the words that carry the disguise, so a guard on a
 * neighbouring word ("food porn") still sees the neighbour as written. Never used for the category.
 */
function readFloorVariants(term: string): string[] {
  const variants = new Set<string>();
  const mapped = mapLookAlikes(term);
  for (const root of mapped === term ? [term] : [term, mapped]) {
    const words = root.split(" ");
    const folded = [foldLeetWords(words, "i"), foldLeetWords(words, "l")].filter((value): value is string[] => value !== undefined);
    // A mask this reader put in ("node*js" from "node.js") is a guess, not text: only its filled readings count, or the
    // mask itself would hide a guard's neighbour.
    const guessed = maskSymbolWords(words);
    for (const base of [words, ...folded, ...(guessed ? [guessed] : [])]) {
      const runs = joinFragmentRuns(base);
      const readings = [
        base,
        expandOrNull(base, joinSingleLetters),
        expandOrNull(base, joinFragments),
        expandOrNull(base, joinDigitRuns),
        ...runs.inPlace,
        mapWords(base, (word) => word.replace(/([a-z])\1{2,}/gu, "$1$1")),
        mapWords(base, (word) => word.replace(/([a-z])\1{2,}/gu, "$1")),
        mapWords(base, (word) => (RARE_DOUBLE_PATTERN.test(word) ? word.replace(/([aiuyhjkqvwx])\1+/gu, "$1") : word))
      ];
      for (const reading of readings) {
        if (!reading) {
          continue;
        }
        if (base !== guessed && !runs.masked.has(reading)) {
          variants.add(reading.join(" "));
        }
        for (const unmasked of fillMasks(reading)) {
          variants.add(unmasked);
        }
      }
      for (const word of runs.standalone) {
        variants.add(word);
      }
    }
  }
  variants.delete(term);
  return [...variants];
}

function mapLookAlikes(term: string): string {
  let mapped = term;
  for (const [pattern, letter] of LOOK_ALIKE_SEQUENCES) {
    mapped = mapped.replace(pattern, letter);
  }
  return mapped.replace(LOOK_ALIKE_LETTER_PATTERN, (symbol) => LOOK_ALIKE_LETTERS[symbol] ?? symbol);
}

/**
 * "n4ked", "p0rn", "$ex", "na·ked", "nu/de", "nak'd": in a word that mixes letters with digits, symbols or
 * punctuation, the look-alikes read as letters and the punctuation goes. "1" is "i" or "l", so both are tried. A
 * plain number is not a word and is left alone, so ages are untouched.
 */
function foldLeetWords(words: string[], one: "i" | "l"): string[] | undefined {
  let changed = false;
  const folded = words.map((word) => {
    if (!/[a-z]/u.test(word) || !/[^a-z]/u.test(word)) {
      return word;
    }
    const letters = word
      .replace(/1/gu, one)
      .replace(LEET_CHARACTER_PATTERN, (character) => LEET_LETTERS[character] ?? character)
      .replace(/[^a-z0-9*#]/gu, "");
    changed ||= letters !== word && letters.length > 0;
    return letters || word;
  });
  return changed ? folded : undefined;
}

/**
 * "n.ked", "n~ked", "n^ked", "n%ked", "n...ked", "nake☆": a symbol inside or at the end of a word may stand for the
 * letter it hides, not only be noise between letters (foldLeetWords drops it). Each run of symbols becomes one masked
 * letter, filled in by fillMasks. An apostrophe is a contraction, not a mask.
 */
function maskSymbolWords(words: string[]): string[] | undefined {
  const masked = words.map((word) =>
    /[a-z]/u.test(word) ? word.replace(/(?<=[a-z])[^a-z0-9\s*#']{1,3}(?=[a-z])|(?<=[a-z]{2})[^a-z0-9\s*#'.,!?]$/gu, "*") : word
  );
  return masked.some((word, index) => word !== words[index]) ? masked : undefined;
}

/** "n a k e d", "s e x": a run of three or more single letters is one word. */
function joinSingleLetters(words: string[]): string[] {
  const joined: string[] = [];
  let run: string[] = [];
  const flush = () => {
    joined.push(...(run.length >= 3 ? [run.join("")] : run));
    run = [];
  };
  for (const word of words) {
    if (/^[a-z]$/u.test(word)) {
      run.push(word);
      continue;
    }
    flush();
    joined.push(word);
  }
  flush();
  return joined;
}

/**
 * "na ked", "nu de", "se x", "nake d": two short fragments, neither of them a word that could stand there on its
 * own, are one word. At least one side has to be one or two letters long, which ordinary pairs of words rarely are.
 */
function joinFragments(words: string[]): string[] {
  const joined: string[] = [];
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index];
    const next = words[index + 1];
    if (
      next !== undefined &&
      /^[a-z]{1,4}$/u.test(word) &&
      /^[a-z]{1,4}$/u.test(next) &&
      Math.min(word.length, next.length) <= 2 &&
      !JOIN_STOPWORDS.has(word) &&
      !JOIN_STOPWORDS.has(next)
    ) {
      joined.push(word + next);
      index += 1;
      continue;
    }
    joined.push(word);
  }
  return joined;
}

/**
 * A word broken into three or more pieces, or into two that are both three or four letters ("na ke d", "n a ked",
 * "f u ck", "p o rn", "blo wjob", "sch ool girl", "s ex y"). Every window of two to five adjacent fragments is joined
 * into its own reading, so the pieces of one word come back together wherever the break fell. A fragment is a short
 * letter run that is not a common word; a single letter counts unless it stands between two whole words, as the "n"
 * of "tee n jeans" does; "a", "i", a two-letter common word and a piece of five or six letters count only next to a
 * broken-off single letter. Where a
 * separator replaced a letter ("n-ked", "ch_ld", "s x"), a pair with a vowelless side is also read with that letter
 * masked (`masked`: only its filled readings count).
 *
 * A window is joined in place only when one of its pieces is visibly broken (two letters or fewer, or no vowel). A
 * window of whole-looking words ("blo wjob", or just "cat naps", "tote bag") is read as the joined word on its own
 * (`standalone`): joined in place, "her 16-year-old catnaps" lost the cat that owns the age and "nude totebag" the bag
 * that makes "nude" a colour.
 */
function joinFragmentRuns(words: string[]): { inPlace: string[][]; masked: Set<string[]>; standalone: string[] } {
  const isLetters = (word: string | undefined) => word !== undefined && /^[a-z]+$/u.test(word);
  const isBrokenLetter = (index: number) => {
    const word = words[index];
    if (!isLetters(word) || word.length !== 1 || word === "a" || word === "i") {
      return false;
    }
    const before = words[index - 1];
    const after = words[index + 1];
    return !(isLetters(before) && before.length >= 3 && isLetters(after) && after.length >= 3);
  };
  const isFragment = (index: number) => {
    const word = words[index];
    if (!isLetters(word) || word.length > 6) {
      return false;
    }
    if (word.length > 4) {
      // A longer piece belongs to the word only where a letter was broken off right next to it ("t o pless").
      return isBrokenLetter(index - 1) || isBrokenLetter(index + 1);
    }
    if (word.length === 1) {
      return word === "a" || word === "i" ? isBrokenLetter(index - 1) || isBrokenLetter(index + 1) : isBrokenLetter(index);
    }
    if (JOIN_STOPWORDS.has(word)) {
      return word.length === 2 && (isBrokenLetter(index - 1) || isBrokenLetter(index + 1));
    }
    return true;
  };
  const isBroken = (piece: string) => piece.length <= 2 || !/[aeiouy]/u.test(piece);
  const inPlace: string[][] = [];
  const masked = new Set<string[]>();
  const standalone: string[] = [];
  let index = 0;
  while (index < words.length) {
    if (!isFragment(index)) {
      index += 1;
      continue;
    }
    let end = index;
    while (end + 1 < words.length && isFragment(end + 1)) {
      end += 1;
    }
    for (let from = index; from < end; from += 1) {
      for (let to = from + 1; to <= Math.min(end, from + 4); to += 1) {
        const window = words.slice(from, to + 1);
        if (!window.some(isBroken)) {
          standalone.push(window.join(""));
          continue;
        }
        inPlace.push([...words.slice(0, from), window.join(""), ...words.slice(to + 1)]);
        if (window.length === 2 && window.some((piece) => !/[aeiouy]/u.test(piece))) {
          const reading = [...words.slice(0, from), window.join("*"), ...words.slice(to + 1)];
          inPlace.push(reading);
          masked.add(reading);
        }
      }
    }
    index = end + 1;
  }
  return { inPlace, masked, standalone };
}

/** "1 4 years old": single digits written apart are one number. */
function joinDigitRuns(words: string[]): string[] {
  const joined: string[] = [];
  let run: string[] = [];
  const flush = () => {
    joined.push(...(run.length >= 2 && run.length <= 3 ? [run.join("")] : run));
    run = [];
  };
  for (const word of words) {
    if (/^\d$/u.test(word)) {
      run.push(word);
      continue;
    }
    flush();
    joined.push(word);
  }
  flush();
  return joined;
}

/**
 * "n*ked", "p*ssy", "nake*": a masked letter is any letter when it is the only one (26 readings), and a vowel, the
 * likeliest, when there are two (25 readings). More masks than that are not read.
 */
function fillMasks(words: string[]): string[] {
  const text = words.join(" ");
  const masks = text.match(/[*#]/gu)?.length ?? 0;
  if (masks === 0 || !/[a-z][*#]|[*#][a-z]/u.test(text)) {
    return [];
  }
  let readings = masks > 2 ? [] : [text];
  const letters = masks === 1 ? MASK_LETTERS : MASK_VOWELS;
  for (let pass = 0; pass < masks && masks <= 2; pass += 1) {
    readings = readings.flatMap((reading) => letters.map((letter) => reading.replace(MASK_CHARACTER_PATTERN, letter)));
  }
  return [...readings, ...readCensoredWords(words)];
}

// The words people censor with one "*" per letter. A masked word is read as one of these when the letters it shows
// line up ("p**n", "f**k", "c**t", "n**ed"): enumerating every filler stops being affordable past two masks, and the
// vowel guess misses the consonants these words hide.
const CENSORED_WORDS = (
  "porn porno fuck fucking fucked fucker cunt cunts tits titties pussy naked nude nudes nudity sex dick dicks cock " +
  "cocks cum slut sluts whore whores boobs anal penis vagina nipple nipples loli lolita lolis shota child teen teens " +
  "preteen nsfw hentai orgasm blowjob topless bottomless horny rape incest xxx semen"
).split(" ");

/**
 * Readings of masked words beyond fillMasks' guesses: the word with its masks dropped ("n*a*k*e*d", "s*e*x" put a
 * mask between every letter), and each censorable word the masked word could spell, letter for letter.
 */
function readCensoredWords(words: string[]): string[] {
  const readings: string[] = [];
  words.forEach((word, index) => {
    if (!/[*#]/u.test(word) || !/[a-z]/u.test(word)) {
      return;
    }
    const candidates = new Set<string>();
    const unmasked = word.replace(/[*#]/gu, "");
    if (unmasked.length >= 2) {
      candidates.add(unmasked);
    }
    const collapsed = word.replace(/[*#]+/gu, "*");
    if ((collapsed.match(/\*/gu)?.length ?? 0) === 1 && collapsed !== word) {
      for (const letter of MASK_LETTERS) {
        candidates.add(collapsed.replace("*", letter));
      }
    }
    const shape = new RegExp(`^${word.replace(/[^a-z0-9*#]/gu, "").replace(/[*#]/gu, "[a-z]")}$`, "u");
    for (const censored of CENSORED_WORDS) {
      if (shape.test(censored)) {
        candidates.add(censored);
      }
    }
    for (const candidate of candidates) {
      readings.push([...words.slice(0, index), candidate, ...words.slice(index + 1)].join(" "));
    }
  });
  return readings;
}

function expandOrNull(words: string[], rewrite: (words: string[]) => string[]): string[] | undefined {
  const rewritten = rewrite(words);
  return rewritten.join(" ") === words.join(" ") ? undefined : rewritten;
}

function mapWords(words: string[], rewrite: (word: string) => string): string[] | undefined {
  const rewritten = words.map(rewrite);
  return rewritten.some((word, index) => word !== words[index]) ? rewritten : undefined;
}

/** The terms plus their disguise readings, for the safety floor. */
function withFloorVariants(terms: string[]): string[] {
  return uniqueStrings([...terms, ...terms.flatMap(readFloorVariants)]);
}

// ---------------------------------------------------------------------------------------------------------------
// Public checks
// ---------------------------------------------------------------------------------------------------------------

/** True when the tag is explicit on its own (ambiguous postures such as `on back` are not). Weights allowed. */
export function isExplicitSceneTag(tag: string): boolean {
  const terms = toTerms([tag]);
  const reading = withFloorVariants(terms);
  return (
    reading.some((term) => isExplicitTerm(term) || matchesRule(term, IMPLIED_NUDITY_RULE) || readOnlyWearing(term) === "explicit") ||
    reading.some((term) => isBareChestTerm(term) && readSubjectGender([term]) !== "male" && readSubjectGender([term]) !== "unknown")
  );
}

/** True when the tag codes a subject as a minor (loli, child, middle schooler, "15 years old", ...). */
export function isMinorCodedSceneTag(tag: string): boolean {
  return withFloorVariants(toTerms([tag])).some((term) => isMinorCodedTermNearPerson(term, CAMERA_CONTEXT_PATTERN.test(term)));
}

/**
 * True for text the photo prompt must never carry, whether or not it would lock the cut: anything minor-coded, any
 * student identity (a tag's "schoolgirl" is school-coded for routing, but the photo subject stays an adult only if
 * the word is gone), and "lolita" in any sense. The builder drops tags and appearance segments with it.
 */
export function isMinorOrStudentCodedSceneText(text: string): boolean {
  const terms = toTerms([text]);
  const camera = terms.some((term) => CAMERA_CONTEXT_PATTERN.test(term));
  const reading = withFloorVariants(terms);
  return reading.some(
    (term) => isMinorCodedTermNearPerson(term, camera) || matchesRule(term, STUDENT_IDENTITY_RULE) || /\blolitas?\b/u.test(term)
  );
}

/**
 * Puts a cut into exactly one category. Precedence:
 * explicit > action > fantasy > romance > (no humans: object | scenery) > group > portrait | full_body.
 *
 * The cue kind is recorded but never decides anything: the annotation pass labels every beat that involves
 * movement `action`, reaching for a shelf included, so only a lexicon hit makes a cut dynamic.
 */
export function classifyImageScene(input: SceneClassificationInput): SceneClassification {
  const contextTerms = toTerms(input?.contextTags);
  const figureSets = Array.isArray(input?.characterTagSets) ? input.characterTagSets : [];
  const figureTerms = figureSets.map((set) => toTerms(set));
  const sources = [contextTerms, ...figureTerms];
  const allTerms = uniqueStrings(sources.flat());
  const safetyTexts = Array.isArray(input?.safetyTexts) ? input.safetyTexts.filter((text): text is string => typeof text === "string") : [];
  const proseTerms = toProseTerms(safetyTexts);
  // What the safety floor reads: the cut, plus the free text that is rendered with it, each term also in its
  // disguise readings. The category below is decided from the cut alone, as written.
  const floorTerms = uniqueStrings([...allTerms, ...proseTerms]);
  const floorReading = withFloorVariants(floorTerms);
  const proseReading = withFloorVariants(proseTerms);

  // Subject-count tags are counted per source and the largest source wins. Summing across sources double
  // counts: each composed figure carries its own `1girl` identity tag while the base caption says `2girls`.
  const registeredCount = Math.max(0, Math.floor(Number(input?.registeredCharacterCount) || 0));
  const humanCount = Math.max(figureSets.length, registeredCount, ...sources.map(countSubjects));
  const hasSubjectCountTag = allTerms.some(isSubjectCountTerm);
  const hasCharacterTags = figureTerms.some((terms) => terms.length > 0);
  const noHumansMarker = allTerms.find((term) => NO_HUMANS_MARKER_PATTERN.test(term));
  const noHumansByMarker = noHumansMarker !== undefined && !hasSubjectCountTag && !hasCharacterTags;
  const noHumans = humanCount === 0 || noHumansByMarker;

  // Figures are people, so a style or template laid over a cut that has them is about them (readBareChestFlags), and
  // the ages that need a person nearby are read in it (readContextualAges).
  const textsAreGlobal = figureSets.length > 0 || registeredCount > 0;
  const schoolTerms = collect(floorReading, isSchoolCodedTerm);
  if (floorReading.some((term) => SCHOOL_SKIRT_PATTERN.test(term)) && floorReading.some((term) => SCHOOL_NECKWEAR_PATTERN.test(term))) {
    schoolTerms.push("school skirt+necktie");
  }
  const bareChest = readBareChestFlags(figureTerms, contextTerms, safetyTexts, textsAreGlobal);
  const onlyWearing = floorReading.map((term) => ({ term, reading: readOnlyWearing(term) }));
  const suggestiveTerms = uniqueStrings([
    ...collect(floorReading, (term) => matchesRule(term, SUGGESTIVE_RULE) || SUGGESTIVE_EMOJI_PATTERN.test(term)),
    ...bareChest.suggestive,
    ...onlyWearing.filter(({ reading }) => reading === "suggestive").map(({ term }) => term)
  ]);
  if (
    floorTerms.some((term) => BARE_SHOULDERS_PATTERN.test(term)) &&
    floorTerms.some((term) => LYING_PATTERN.test(term)) &&
    floorTerms.some((term) => BED_PATTERN.test(term))
  ) {
    suggestiveTerms.push("bare shoulders+lying on bed");
  }

  // Each source of text with whether it is about a person. Figures are; free text always is, since every text the
  // floor is handed without figures (a typed or recorded photo prompt, an appearance, a template, a finished prompt)
  // describes the photo's subject, and "Yuna (14)" names nobody else; the base caption is only when it names one.
  const sourceReadings = [
    ...figureTerms.map((terms) => ({ terms, person: true, freeText: false })),
    { terms: contextTerms, person: textsAreGlobal || contextTerms.some(isPersonTerm), freeText: false },
    ...safetyTexts.map((text) => ({ terms: toProseTerms([text]), person: true, freeText: true }))
  ];
  // The screens hand a style over one comma segment at a time, so "f/8" in "Shot on a Sony A7, 35mm, f/8, ISO 100"
  // arrives without its camera words. Free texts are read as one style for this; a figure's tags still need their own.
  const cameraInFreeText = sourceReadings.some(({ terms, freeText }) => freeText && terms.some((term) => CAMERA_CONTEXT_PATTERN.test(term)));
  const childBodyKinds = readChildBodyKinds(floorReading);
  const femaleSubject = ["female", "mixed"].includes(readSubjectGender(floorTerms));
  const minorTerms = uniqueStrings([
    ...collect(floorReading, isMinorCodedTerm),
    // Free text renders a student identity as written; see STUDENT_IDENTITY_RULE.
    ...collect(proseReading, (term) => matchesRule(term, STUDENT_IDENTITY_RULE)),
    ...sourceReadings.flatMap(({ terms, person, freeText }) => {
      if (!person) {
        return [];
      }
      const camera = (freeText && cameraInFreeText) || terms.some((term) => CAMERA_CONTEXT_PATTERN.test(term));
      return collect(withFloorVariants(terms), (term) => readContextualAges(term, camera).length > 0);
    }),
    // Neighbouring segments read together where a comma split one phrase: "a little, shy girl", "born on May 3, 2013".
    // The screens hand each segment over as its own text, so the texts are also read as one run.
    ...collect(
      withFloorVariants(joinSplitPhrases([...figureTerms, contextTerms, ...safetyTexts.map(toSentenceSegments).flat(), safetyTexts.flatMap((text) => toSentenceSegments(text).flat())])),
      isMinorCodedTerm
    ),
    ...(childBodyKinds.size >= 2 && femaleSubject && suggestiveTerms.length > 0
      ? [`child body (${[...childBodyKinds].join(" + ")})+${suggestiveTerms[0]}`]
      : [])
  ]);
  const unverifiableWords = uniqueStrings(floorTerms.flatMap(readUnverifiableWords));

  const explicitSignals = [
    ...collect(floorReading, isExplicitTerm),
    // Text about a person is about that person's body, so a bird's name there ("great tits", "masked boobies") is not
    // a bird. Birds at a feeder stay birds in a cut without one.
    ...sourceReadings.flatMap(({ terms, person }) =>
      person && !terms.some((term) => BIRD_CONTEXT_PATTERN.test(term)) ? collect(withFloorVariants(terms), (term) => GUARDED_BODY_WORD_PATTERN.test(term)) : []
    ),
    ...bareChest.explicit,
    ...onlyWearing.filter(({ reading }) => reading === "explicit").map(({ term }) => term),
    // A person is implied by the activity itself, so only an explicit "no humans" cut is exempt.
    ...(noHumansByMarker ? [] : collect(floorReading, (term) => matchesRule(term, IMPLIED_NUDITY_RULE))),
    ...(suggestiveTerms.length > 0
      ? collect(floorReading, (term) => matchesRule(term, AMBIGUOUS_POSTURE_RULE)).map((term) => `${term}+${suggestiveTerms[0]}`)
      : [])
  ];

  const classification = (category: ImageSceneCategory, deciding: string[]): SceneClassification => ({
    category,
    signals: uniqueStrings([
      ...deciding,
      ...minorTerms.slice(0, 2).map((term) => `minor:${term}`),
      ...schoolTerms.slice(0, 2).map((term) => `school:${term}`),
      ...suggestiveTerms.slice(0, 2).map((term) => `suggestive:${term}`),
      ...unverifiableWords.slice(0, 2).map((word) => `unverifiable-script:${word}`),
      ...collect(allTerms, (term) => IDENTITY_ANATOMY_TRAIT_PATTERN.test(term)).slice(0, 1).map((term) => `identity-trait:${term}`),
      `subjects:${humanCount}`,
      ...(typeof input?.cueKind === "string" && input.cueKind.trim() ? [`kind:${input.cueKind.trim()}`] : [])
    ]),
    minorCoded: minorTerms.length > 0,
    schoolCoded: schoolTerms.length > 0,
    suggestive: suggestiveTerms.length > 0 || category === "explicit",
    unverifiable: unverifiableWords.length > 0,
    humanCount
  });

  if (explicitSignals.length > 0) {
    return classification("explicit", prefixed("explicit", uniqueStrings(explicitSignals)));
  }

  // A garment named after an activity (combat shirt, running shoes, boxing gloves) is wardrobe, not motion.
  const actionTerms = collect(allTerms, (term) => matchesRule(term, ACTION_RULE) && !isGarmentTag(term));
  if (actionTerms.length > 0) {
    return classification("action", prefixed("action", actionTerms));
  }

  const fantasyTerms = collectFantasyTerms(contextTerms, figureTerms);
  if (fantasyTerms.length > 0) {
    return classification("fantasy", prefixed("fantasy", fantasyTerms));
  }

  const pairTerms = collect(allTerms, (term) => matchesRule(term, ROMANCE_PAIR_RULE));
  const needsPairTerms = humanCount >= 2 ? collect(allTerms, (term) => matchesRule(term, ROMANCE_NEEDS_PAIR_RULE)) : [];
  if (pairTerms.length > 0 || needsPairTerms.length > 0) {
    return classification("romance", prefixed("romance", [...pairTerms, ...needsPairTerms]));
  }

  if (noHumans) {
    const { category, deciding } = classifyEmptyCut(allTerms, input?.frame, noHumansMarker);
    return classification(category, deciding);
  }

  if (humanCount >= 2) {
    return classification("group", [`group:${humanCount}`]);
  }

  const { category, deciding } = classifySingleSubjectCrop(allTerms, input?.frame);
  return classification(category, [deciding]);
}

/** object vs scenery for a cut with nobody in it. */
function classifyEmptyCut(
  terms: string[],
  frame: ImageCueFrame | undefined,
  noHumansMarker: string | undefined
): { category: "object" | "scenery"; deciding: string[] } {
  const focus = collect(terms, (term) => OBJECT_FOCUS_PATTERN.test(term));
  if (focus.length > 0) {
    return { category: "object", deciding: prefixed("object", focus) };
  }

  const objects = collect(terms, (term) => matchesRule(term, OBJECT_RULE));
  const settings = collect(terms, (term) => SETTING_PATTERN.test(term) && !matchesRule(term, OBJECT_RULE));
  const sceneryDeciding = noHumansMarker
    ? [`scenery:${noHumansMarker}`, ...prefixed("scenery", settings).slice(0, 2)]
    : settings.length > 0
      ? prefixed("scenery", settings)
      : ["scenery:empty"];
  if (objects.length === 0) {
    return { category: "scenery", deciding: sceneryDeciding };
  }

  const shot = readShot(frame);
  const wide = shot === "wide_shot" || terms.some((term) => WIDE_COMPOSITION_PATTERN.test(term));
  if (wide) {
    return { category: "scenery", deciding: sceneryDeciding };
  }
  // The object is the subject when the crop is tight on it, or when the cut names at least as many objects as
  // places: "cafe, coffee cup, cake" is a table shot, "library, bookshelf, book" is a room.
  if (shot === "close_up" || objects.length >= settings.length) {
    return { category: "object", deciding: prefixed("object", objects) };
  }
  return { category: "scenery", deciding: sceneryDeciding };
}

/** portrait vs full_body for one person: the declared crop, else the composition tags, else portrait. */
function classifySingleSubjectCrop(
  terms: string[],
  frame: ImageCueFrame | undefined
): { category: "portrait" | "full_body"; deciding: string } {
  const declared = readShot(frame);
  if (declared) {
    return { category: PORTRAIT_SHOTS.has(declared) ? "portrait" : "full_body", deciding: `frame:${declared}` };
  }
  // Same inference the rest of the image pipeline uses, so routing and the rendered crop agree.
  const inferred = readShot(inferImageCueFrameFromTags(terms));
  if (inferred) {
    return { category: PORTRAIT_SHOTS.has(inferred) ? "portrait" : "full_body", deciding: `composition:${inferred}` };
  }
  return { category: "portrait", deciding: "default:portrait" };
}

function readShot(frame: ImageCueFrame | undefined): ImageCueShot | undefined {
  const shot = frame?.shot;
  return shot && (PORTRAIT_SHOTS.has(shot) || FULL_BODY_SHOTS.has(shot)) ? shot : undefined;
}

/**
 * Fantasy terms with costumes removed. A costume marker is scoped to the figure that wears it (plus the base
 * caption, which describes everyone), so one figure's bunny-ear headband does not hide another figure's real
 * fox ears.
 */
function collectFantasyTerms(contextTerms: string[], figureTerms: string[][]): string[] {
  const contextFakes = FAKE_FANTASY_TRAITS.filter(({ fake }) => contextTerms.some((term) => fake.test(term)));
  const found: string[] = [];
  for (const terms of [contextTerms, ...figureTerms]) {
    const fakes = uniqueValues([
      ...contextFakes,
      ...FAKE_FANTASY_TRAITS.filter(({ fake }) => terms.some((term) => fake.test(term)))
    ]);
    for (const term of terms) {
      if (matchesRule(term, FANTASY_RULE) && !fakes.some(({ trait }) => trait.test(term))) {
        found.push(term);
      }
    }
  }
  return uniqueStrings(found);
}

/**
 * A bare chest per source, read with the gender of whoever it is written about: a figure's own tags, the base
 * caption, or the sentence of free text (split at ";" too, where the builder joins two people). When that source
 * names nobody, the gender of the whole text, then of the whole cut, decides. A woman, or a source naming both,
 * makes it explicit; a man makes it nothing; nobody at all makes it suggestive. "Only jeans" and the like count as a
 * bare chest here (readOnlyWearing).
 *
 * A style, a template or any other text laid over a cut that has figures of its own (`textsAreGlobal`) has no subject
 * of its own: "Bare-chested like the guys." in a style is about the woman in the cut. Its bare chest is judged with
 * the cut's gender as well as its own words, and either one naming a woman makes it explicit. Words that only compare
 * or name a garment's owner ("like the guys", "men's jeans") never decide the gender.
 */
function readBareChestFlags(
  figureTerms: string[][],
  contextTerms: string[],
  safetyTexts: string[],
  textsAreGlobal: boolean
): { explicit: string[]; suggestive: string[] } {
  const explicit: string[] = [];
  const suggestive: string[] = [];
  const textChunks = safetyTexts.map((text) =>
    canonicalizeSceneText(text)
      .toLowerCase()
      .split(/\n+|(?<=[.!?。;])\s+|;/u)
      .map((chunk) => chunk.trim())
      .filter(Boolean)
  );
  const cutGender = readSubjectGender([...figureTerms.flat(), ...contextTerms]);
  const wholeCut = readSubjectGender([...figureTerms.flat(), ...contextTerms, ...textChunks.flat()]);
  const record = (hits: string[], gender: ReturnType<typeof readGender>) => {
    if (gender === "female" || gender === "mixed") {
      explicit.push(...hits.map((hit) => `${hit}+${gender}`));
    } else if (gender === "unknown") {
      suggestive.push(...hits);
    }
  };
  const judge = (sourceTerms: string[], fallbacks: string[][]) => {
    const hits = collect(withFloorVariants(sourceTerms), isBareChestTerm);
    if (hits.length === 0) {
      return;
    }
    let gender = readSubjectGender(sourceTerms);
    for (const fallback of fallbacks) {
      if (gender !== "unknown") {
        break;
      }
      gender = readSubjectGender(fallback);
    }
    record(hits, gender === "unknown" ? wholeCut : gender);
  };
  for (const terms of figureTerms) {
    judge(terms, [contextTerms]);
  }
  judge(contextTerms, []);
  textChunks.forEach((chunks) => {
    for (const chunk of chunks) {
      const terms = toProseTerms([chunk]);
      if (!textsAreGlobal) {
        judge(terms, [chunks]);
        continue;
      }
      const hits = collect(withFloorVariants(terms), isBareChestTerm);
      if (hits.length > 0) {
        record(hits, combineGenders(readSubjectGender(terms), cutGender === "unknown" ? readSubjectGender(chunks) : cutGender));
      }
    }
  });
  return { explicit: uniqueStrings(explicit), suggestive: uniqueStrings(suggestive) };
}

function isBareChestTerm(term: string): boolean {
  return BARE_CHEST_PATTERN.test(term) || readOnlyWearing(term) === "topless";
}

/**
 * The gender of the people the texts are about: readGender without comparisons and garment owners, and with the
 * garments only women's wear counting as a woman ("wearing only panties" is a topless woman, not nobody).
 */
function readSubjectGender(texts: string[]): ReturnType<typeof readGender> {
  const subject = texts.map((text) => text.replace(INCIDENTAL_GENDER_PATTERN, " "));
  const gender = readGender(subject);
  if (subject.some((text) => WOMENS_WEAR_PATTERN.test(text))) {
    return gender === "male" || gender === "mixed" ? "mixed" : "female";
  }
  return gender;
}

function combineGenders(a: ReturnType<typeof readGender>, b: ReturnType<typeof readGender>): ReturnType<typeof readGender> {
  if (a === "unknown") {
    return b;
  }
  if (b === "unknown" || a === b) {
    return a;
  }
  return "mixed";
}

function readGender(texts: string[]): "female" | "male" | "mixed" | "unknown" {
  const female = texts.some((text) => FEMALE_WORD_PATTERN.test(text));
  const male = texts.some((text) => MALE_WORD_PATTERN.test(text));
  return female && male ? "mixed" : female ? "female" : male ? "male" : "unknown";
}

/** "wearing only a smile" is explicit; "wearing only an oversized sweater" is suggestive; see ONLY_WEARING_PATTERN. */
function readOnlyWearing(term: string): OnlyWearingReading | undefined {
  const worn = ONLY_WEARING_PATTERN.exec(term) ?? WITH_ONLY_ON_PATTERN.exec(term);
  if (worn) {
    return readWornCoverage(worn[1], true);
  }
  const bare = BARE_ONLY_PATTERN.exec(term) ?? WHOLE_ONLY_PATTERN.exec(term);
  const phrase = bare?.slice(1).find((group) => group !== undefined);
  return phrase ? readWornCoverage(phrase, false) : undefined;
}

type OnlyWearingReading = "explicit" | "topless" | "suggestive";

/**
 * What "only X" leaves bare. Each part of X covers the upper body, the lower body, both or neither; a negated part
 * ("no skirt", "without a top") covers nothing. With nothing over the hips it is explicit, with nothing over the
 * chest it is a bare chest (gendered, see readBareChestFlags), with both covered it is suggestive. Without a wearing
 * verb, X must name something worn at all, so "nothing but trouble" is not read.
 */
function readWornCoverage(phrase: string, afterWearingVerb: boolean): OnlyWearingReading | undefined {
  let upper = false;
  let lower = false;
  let worn = false;
  for (const raw of phrase.split(/\s+(?:and|with|or|plus|over|under)\s+|,|\s+&\s+/u)) {
    const part = raw.trim().replace(/\s+(?:on|only|and nothing else|nothing else)$/u, "").trim();
    if (!part) {
      continue;
    }
    if (/^(?:no|not|without|minus|sans|zero|never)\b/u.test(part)) {
      // A missing garment is still clothing vocabulary, but it covers nothing.
      worn = true;
      continue;
    }
    if (FULL_COVER_GARMENT_PATTERN.test(part) || (LONG_TOP_GARMENT_PATTERN.test(part) && !SHORTENED_TOP_PATTERN.test(part))) {
      upper = lower = worn = true;
    } else if (SHORT_TOP_GARMENT_PATTERN.test(part) || LONG_TOP_GARMENT_PATTERN.test(part)) {
      upper = worn = true;
    } else if (LOWER_GARMENT_PATTERN.test(part)) {
      lower = worn = true;
    } else if (WORN_ACCESSORY_PATTERN.test(part) || COVERING_GARMENT_PATTERN.test(part)) {
      worn = true;
    }
  }
  if (!worn && !afterWearingVerb) {
    return undefined;
  }
  return !lower ? "explicit" : !upper ? "topless" : "suggestive";
}

function isExplicitTerm(term: string): boolean {
  return EXPLICIT_RULES.some((rule) => matchesRule(term, rule)) || EXPLICIT_EMOJI_PATTERN.test(term);
}

/** A term that names a person (or an age's owner): a person noun, a pronoun, a portrait. */
function isPersonTerm(term: string): boolean {
  return FEMALE_WORD_PATTERN.test(term) || MALE_WORD_PATTERN.test(term) || PERSON_CONTEXT_PATTERN.test(term);
}

/**
 * Adjacent comma segments of one source, joined where the comma split a phrase the floor reads whole: a list of
 * adjectives before the noun ("a little, shy girl", "a young, Korean girl") and a date ("born on May 3, 2013", "class
 * of", "2031"). Only for the minor reading.
 */
function joinSplitPhrases(sources: string[][]): string[] {
  const joined: string[] = [];
  for (const terms of sources) {
    for (let index = 0; index + 1 < terms.length; index += 1) {
      if (SPLIT_PHRASE_OPENER_PATTERN.test(terms[index])) {
        joined.push(`${terms[index]} ${terms[index + 1]}`);
      }
    }
  }
  return joined;
}

/** Each sentence of a free text as its normalised comma segments, in order. */
function toSentenceSegments(text: string): string[][] {
  return splitSceneSentences(text).map((segments) => segments.map((segment) => normalizeSceneTag(segment)).filter(Boolean));
}

function isSchoolCodedTerm(term: string): boolean {
  return matchesRule(term, SCHOOL_RULE) || matchesRule(term, STUDENT_IDENTITY_RULE) || BARE_SCHOOL_PATTERN.test(term) || SCHOOL_EMOJI_PATTERN.test(term);
}

function isMinorCodedTerm(term: string): boolean {
  if (matchesRule(term, MINOR_RULE) || MINOR_NOUN_PATTERN.test(term) || SCHOOL_GRADE_PATTERN.test(term) || NOT_OF_AGE_PATTERN.test(term)) {
    return true;
  }
  return (
    readStatedAges(term).some(({ age, limit }) => age > 0 && age < limit) ||
    readBirthYears(term).some((age) => age >= 0 && age < 19) ||
    readsStillInSchool(term) ||
    BARE_AGE_PATTERN.test(term) ||
    MINOR_EMOJI_PATTERN.test(term)
  );
}

/** isMinorCodedTerm, plus the forms that are ages only next to a person (readContextualAges). */
function isMinorCodedTermNearPerson(term: string, camera: boolean): boolean {
  return isMinorCodedTerm(term) || readContextualAges(term, camera).length > 0;
}

/** Every age stated in the term, with the bound below which it is a minor ("a 25 year old and a 15 year old" is both). */
function readStatedAges(term: string): Array<{ age: number; limit: number }> {
  const ages: Array<{ age: number; limit: number }> = [];
  for (const reading of AGE_READINGS) {
    for (const match of term.matchAll(reading.pattern)) {
      const token = match.slice(1).find((group) => group !== undefined);
      const age = parseAgeToken(token) / (reading.perYear ?? 1);
      if (!Number.isFinite(age) || age <= 0) {
        continue;
      }
      if (reading.ownedByNoun && isAgeOwnedByNoun(term, match, reading.ownedByNoun)) {
        continue;
      }
      ages.push({ age, limit: reading.limit ?? 18 });
    }
  }
  return ages;
}

function parseAgeToken(token: string | undefined): number {
  if (!token) {
    return Number.NaN;
  }
  const value = token.trim();
  const digitTeen = /^([3-9]) ?teen$/u.exec(value);
  if (digitTeen) {
    return 10 + Number(digitTeen[1]);
  }
  if (/^\d/u.test(value)) {
    // The whole part of a decimal age ("14.5", "13,5").
    return Number.parseInt(value, 10);
  }
  if (value === "decade") {
    return 10;
  }
  // Every spelled form here, in any language, names a number below 18; only English and Roman ones are compared.
  return SPELLED_NUMBERS[value] ?? ROMAN_NUMBERS[value] ?? SPELLED_ORDINALS[value] ?? 1;
}

/** See NON_PERSON_AGED_NOUN. */
function isAgeOwnedByNoun(term: string, match: RegExpMatchArray, kind: "after-old" | "after-aged"): boolean {
  const start = match.index ?? 0;
  if (PERSON_BEFORE_PATTERN.test(term.slice(0, start))) {
    return false;
  }
  const after = term.slice(start + match[0].length);
  return (kind === "after-old" ? NON_PERSON_AFTER_OLD_PATTERN : NON_PERSON_AFTER_AGED_PATTERN).test(after);
}

function readBirthYears(term: string): number[] {
  const thisYear = new Date().getFullYear();
  return Array.from(term.matchAll(BIRTH_YEAR_PATTERN), (match) => {
    // "born in '12" is this century unless that would put the birth in the future.
    const written = Number(match[1].replace("'", ""));
    const year = match[1].startsWith("'") ? (2000 + written <= thisYear ? 2000 + written : 1900 + written) : written;
    return thisYear - year;
  });
}

/**
 * A graduating class still ahead: a school class graduates the summer its members turn 18, so a class year after this
 * one (or this one, before the summer) is someone under 18 now. Two-digit years ("class of '31") are this century
 * unless they would lie more than twenty years ahead.
 */
function readsStillInSchool(term: string): boolean {
  if (COLLEGE_CONTEXT_PATTERN.test(term)) {
    return false;
  }
  const now = new Date();
  const thisYear = now.getFullYear();
  return Array.from(term.matchAll(GRADUATION_YEAR_PATTERN)).some((match) => {
    const written = Number(match[1].replace("'", ""));
    const year = match[1].startsWith("'") ? (written <= (thisYear % 100) + 20 ? 2000 + written : 1900 + written) : written;
    return year > thisYear || (year === thisYear && now.getMonth() < 6);
  });
}

/** The whitespace-separated words of a canonical term that still carry an unverifiable character, for signals. */
function readUnverifiableWords(term: string): string[] {
  if (!UNVERIFIABLE_CHARACTER_PATTERN.test(term)) {
    return [];
  }
  return term
    .split(/\s+/u)
    .filter((word) => UNVERIFIABLE_CHARACTER_PATTERN.test(word))
    .map((word) => Array.from(word).slice(0, 24).join(""));
}

function isSubjectCountTerm(term: string): boolean {
  return SUBJECT_COUNT_PATTERN.test(term) || MULTIPLE_SUBJECTS_PATTERN.test(term) || term === "solo";
}

function countSubjects(terms: string[]): number {
  const byKind = { girls: 0, boys: 0, others: 0 };
  let solo = false;
  for (const term of terms) {
    const counted = SUBJECT_COUNT_PATTERN.exec(term);
    if (counted) {
      const kind = counted[2].startsWith("girl") ? "girls" : counted[2].startsWith("boy") ? "boys" : "others";
      byKind[kind] = Math.max(byKind[kind], Number(counted[1]) || 0);
      continue;
    }
    const multiple = MULTIPLE_SUBJECTS_PATTERN.exec(term);
    if (multiple) {
      const kind = multiple[1] === "girls" ? "girls" : multiple[1] === "boys" ? "boys" : "others";
      byKind[kind] = Math.max(byKind[kind], 2);
      continue;
    }
    if (term === "solo") {
      solo = true;
    }
  }
  const total = byKind.girls + byKind.boys + byKind.others;
  return total > 0 ? total : solo ? 1 : 0;
}

/**
 * Splits tag entries into bare lowercase terms. One entry can carry several tags — a weight wrapping a group
 * (`1.2::nude, sweat::`) or a caller passing a whole caption — so it is split on separators first (each half
 * then loses its own weight marker) and on any `::` still left inside.
 *
 * Weights are stripped, so `-2::clothing::` would read as plain clothing — while NovelAI, which honours the
 * negative weight, draws the figure undressed. A negative weight therefore adds a term of its own for what it asks
 * NovelAI for: a body-covering garment weighted down reads `undressed (...)`, a safeguard word (`-1::sfw::`) reads
 * `uncensored (...)`, an adulthood word (`-1::adult::`, `-1::mature female::`) reads `aged down (...)`. The group
 * form `-1::shirt, skirt::` weights down every tag inside it.
 */
function toTerms(tags: unknown): string[] {
  if (!Array.isArray(tags)) {
    return [];
  }
  const terms: string[] = [];
  for (const tag of tags) {
    if (typeof tag !== "string") {
      continue;
    }
    let insideNegativeWeight = false;
    // "|\|" and "|\/|" are letters drawn with a bar, read before the bar splits a tag list ("|\|aked").
    for (const part of canonicalizeSceneText(tag).replace(PIPE_LOOK_ALIKE_PATTERN, (sequence) => PIPE_LOOK_ALIKES[sequence] ?? sequence).split(/[,;|\n、]/u)) {
      const trimmed = part.trim();
      const opener = trimmed.match(NEGATIVE_WEIGHT_OPENER_PATTERN);
      const closes = /::\s*$/u.test(opener ? trimmed.slice(opener[0].length) : trimmed);
      const weightedDown = Boolean(opener) || insideNegativeWeight;
      insideNegativeWeight = opener ? !closes : insideNegativeWeight && !closes;
      for (const piece of normalizeSceneTag(part).split(/\s*::\s*/u)) {
        const term = normalizeSceneTag(piece);
        if (!term) {
          continue;
        }
        terms.push(term);
        if (!weightedDown) {
          continue;
        }
        if (COVERING_GARMENT_PATTERN.test(term)) {
          terms.push(`undressed (${term} weighted down)`);
        } else if (SAFEGUARD_WORD_PATTERN.test(term)) {
          terms.push(`uncensored (${term} weighted down)`);
        } else if (ADULTHOOD_WORD_PATTERN.test(term)) {
          terms.push(`aged down (${term} weighted down)`);
        }
      }
    }
  }
  return terms;
}

/**
 * Terms of free text: each comma segment of each sentence (as the photo builder renders it), plus the clauses of
 * a segment split at its conjunctions, so a clause-sized phrase can stand on its own ("she is lying on her back and
 * wearing lingerie").
 */
function toProseTerms(texts: unknown): string[] {
  if (!Array.isArray(texts)) {
    return [];
  }
  const pieces: string[] = [];
  for (const text of texts) {
    if (typeof text !== "string") {
      continue;
    }
    for (const segment of splitSceneSentences(text).flat()) {
      pieces.push(segment);
      const clauses = segment.split(/\s+(?:and|but|or|while|whereas|who|which)\s+/iu);
      if (clauses.length > 1) {
        pieces.push(...clauses);
      }
    }
  }
  return toTerms(pieces);
}

function matchesRule(term: string, rule: TermRule): boolean {
  return rule.pattern.test(term) && !(rule.except?.test(term) ?? false);
}

function collect(terms: string[], predicate: (term: string) => boolean): string[] {
  return uniqueStrings(terms.filter(predicate));
}

function prefixed(kind: string, terms: string[]): string[] {
  return terms.slice(0, MAX_SIGNALS_PER_KIND).map((term) => `${kind}:${term}`);
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

function uniqueValues<T>(values: T[]): T[] {
  return Array.from(new Set(values));
}
