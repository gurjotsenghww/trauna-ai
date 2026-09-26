/**
 * Tràuna AI — services/promptEnhancer.js
 *
 * Smart Context-Aware Prompt Enhancer & Spatial Composer (Milestone 3 / R3).
 * - Eliminates the need for users to type trigger tokens or technical tags.
 * - Automatically injects the LoRA trigger: `traunathumb, professional youtube gaming thumbnail`.
 * - ACCURACY & GLITCH ELIMINATION: Respects the user's intended subject. If the user asks
 *   for a cat, car, sword, monster, or environment, it NEVER injects unrelated streamer faces.
 * - CONTEXT-AWARE THREE-ZONE SPATIAL COMPOSITION GRAMMAR (R3):
 *     Zone 1 (Foreground Subject): Explicit positioning ('foreground solo subject on the right'
 *       or 'foreground streamer portrait on the right'), strict single-subject tags ('solo,
 *       exactly one person, single subject, sharp focus'), clean anatomy and facial features
 *       ('clean skin, realistic skin texture, coherent eyes, expressive mouth, zero monster features,
 *       zero mutant traits').
 *     Zone 2 (Optical Depth Demarcation): 'optical bokeh, shallow depth of field, blurred
 *       background, strong depth separation between foreground and background'.
 *     Zone 3 (Distant Background): Game world threats, monsters, environment placed strictly
 *       in the background / opposite third ('in the deep background on the left', 'distant
 *       Minecraft creeper behind subject, isolated from foreground character').
 * - ANTI-HYBRID & ANTI-DUPLICATION NEGATIVE MATRIX:
 *     Expanded negative prompt preventing conjoined heads, dual heads, and human-monster blending.
 */

'use strict';

// ── Anti-Hybrid & Anti-Duplication Negative Matrix ────────────────
const ANTI_HYBRID_NEGATIVE_MATRIX =
  'duplicate heads, two heads, conjoined twins, extra heads, multiple faces, mutant facial features, zombie skin on human face, zombie decay on streamer skin, green skin texture on human, green creeper skin on human face, monster teeth on streamer, hybrid monster person, blended creature features, split screen, collage, blurry, text, watermark, bad anatomy, deformed limbs';

const DEFAULT_NEGATIVE_PROMPT = ANTI_HYBRID_NEGATIVE_MATRIX;

/**
 * Merges user-supplied negative tokens with the anti-hybrid matrix without duplication.
 */
function buildNegativePrompt(customNegative = '') {
  if (!customNegative || !customNegative.trim()) {
    return ANTI_HYBRID_NEGATIVE_MATRIX;
  }
  const custom = customNegative.trim();
  const baseTokens = ANTI_HYBRID_NEGATIVE_MATRIX.split(',').map(t => t.trim().toLowerCase());
  const customTokens = custom.split(',').map(t => t.trim()).filter(Boolean);

  const additions = customTokens.filter(t => !baseTokens.includes(t.toLowerCase()));
  if (additions.length === 0) {
    return ANTI_HYBRID_NEGATIVE_MATRIX;
  }
  return `${ANTI_HYBRID_NEGATIVE_MATRIX}, ${additions.join(', ')}`;
}

// ── Creator Signatures & Knowledge Base ──────────────────────────
const CREATOR_PATTERNS = {
  beastboyshub: {
    match: /\b(beastboyshub|bbs|shub)\b/i,
    face: 'BeastBoyShub screaming excited reaction face with headphones',
    lighting: 'vibrant colorful rim lighting, high contrast',
  },
  technogamerz: {
    match: /\b(technogamerz|techno\s*gamerz|ujjw?al)\b/i,
    face: 'TechnoGamerz focused intense gaming reaction',
    lighting: 'vibrant GTA neon lighting, rim lights',
  },
  mrbeast: {
    match: /\b(mrbeast|mr\s*beast|jimmy)\b/i,
    face: 'MrBeast shocked reaction face with open mouth',
    lighting: 'bright commercial lighting, high saturation',
  },
  coryxkenshin: {
    match: /\b(coryxkenshin|cory\s*kenshin|shogun)\b/i,
    face: 'CoryxKenshin terrified jump-scare scream',
    lighting: 'horror shadows, single flashlight beam, cold blue rim',
  },
  markiplier: {
    match: /\b(markiplier|mark)\b/i,
    face: 'Markiplier horrified screaming reaction with headphones',
    lighting: 'dark atmospheric horror lighting, cold blue rim',
  },
  pewdiepie: {
    match: /\b(pewdiepie|felix)\b/i,
    face: 'PewDiePie energetic gaming reaction face with headphones',
    lighting: 'vibrant colorful studio lighting',
  },
};

// ── Game / Genre World & Threat Knowledge Base ───────────────────
const GENRE_PATTERNS = {
  minecraft: {
    match: /\b(minecraft|mc|creeper|nether|enderman|herobrine|warden|diamond|hardcore|redstone|crafting|steve|alex|piglin|wither)\b/i,
    threat: 'distant Minecraft creeper behind subject, isolated from foreground character',
    world: 'blocky 3D voxel Minecraft world, glowing torches',
    atmosphere: 'epic survival adventure',
  },
  horror: {
    match: /\b(horror|resident\s*evil|re4|re2|re8|biohazard|zombie|zombies|scary|ghost|haunted|fnaf|five\s*nights|slender|chucky|jumpscare)\b/i,
    threat: 'distant terrifying zombie silhouette behind subject, isolated from foreground character',
    world: 'terrifying dark abandoned hallway, fog, eerie shadows',
    atmosphere: 'survival horror jump-scare',
  },
  gta: {
    match: /\b(gta|gta5|gta\s*v|grand\s*theft\s*auto|heist|los\s*santos|trevor|franklin|michael)\b/i,
    threat: 'distant speeding police cruiser and helicopter behind subject, isolated from foreground character',
    world: 'Los Santos city streets, neon lights',
    atmosphere: 'high-octane action heist',
  },
  fps: {
    match: /\b(fps|shooter|valorant|cod|call\s*of\s*duty|warzone|pubg|apex|clutch|csgo|counter\s*strike)\b/i,
    threat: 'distant tactical enemy silhouette behind subject, isolated from foreground character',
    world: 'tactical arena, smoke haze, bullet tracers',
    atmosphere: 'intense clutch moment',
  },
  roblox: {
    match: /\b(roblox|bloxd|obby|brookhaven)\b/i,
    threat: 'distant obstacle course challenge behind subject, isolated from foreground character',
    world: 'colorful blocky avatar world',
    atmosphere: 'playful viral gaming',
  },
};

// ── Subject Classification Lexicons ──────────────────────────────
const ANIMAL_REGEX = /\b(cat|cats|kitten|kittens|kitty|dog|dogs|puppy|puppies|canine|feline|pet|pets|bird|birds|parrot|parrots|eagle|eagles|owl|owls|fish|shark|sharks|hamster|hamsters|lion|lions|tiger|tigers|bear|bears|wolf|wolves|fox|foxes|rabbit|rabbits|bunny|bunnies|horse|horses|monkey|monkeys|ape|apes|gorilla|gorillas|elephant|elephants|dinosaur|dinosaurs|dino|spider|spiders|snake|snakes|frog|frogs|lizard|lizards|duck|ducks|cow|pigs|pig|sheep|chicken|panda|koala|cheetah|leopard)\b/i;

const VEHICLE_REGEX = /\b(car|cars|supercar|supercars|sports\s*car|sports\s*cars|hypercar|hypercars|vehicle|vehicles|truck|trucks|tank|tanks|plane|planes|airplane|airplanes|jet|jets|fighter\s*jet|helicopter|helicopters|boat|boats|ship|ships|submarine|motorcycle|motorcycles|bike|bikes|motorbike|train|trains|rocket|rockets|spaceship|spaceships|ufo|mech|mecha|robot|robots)\b/i;

const OBJECT_REGEX = /\b(sword|swords|blade|shield|shields|gun|guns|rifle|rifles|pistol|pistols|weapon|weapons|bow|armor|helmet|diamond|diamonds|crown|gold|treasure|chest|coin|coins|potion|potions|orb|crystal|crystals|portal|card|cards|burger|pizza|computer|laptop|phone)\b/i;

const SOLO_CREATURE_REGEX = /\b(creeper|mutant\s*creeper|enderman|warden|herobrine|wither|zombie|zombies|skeleton|skeletons|dragon|dragons|demon|demons|alien|aliens|monster|monsters|ghost|ghosts|animatronic|animatronics)\b/i;

const EXPLICIT_HUMAN_REGEX = /\b(streamer|streamers|youtuber|youtubers|creator|creators|gamer|gamers|vlogger|vloggers|facecam|selfie|player|players|person|people|human|humans|man|men|woman|women|boy|boys|girl|girls|guy|guys|dude|bro)\b/i;

const THREAT_ENTITY_REGEX = /\b(mutant\s*creeper|mutant\s*zombie|mutant\s*monster|giant\s*creeper|giant\s*zombie|police\s*cruiser|speeding\s*sports\s*car|creeper|zombie|zombies|skeleton|skeletons|enderman|herobrine|warden|wither|ghost|ghosts|monster|monsters|demon|demons|alien|aliens|police|cops|swat|killer|enemy|boss|beast|dragon)\b/i;

/**
 * Analyzes raw prompt to extract subject classification, creator, genre, and secondary threats.
 */
function detectSubjectType(cleanText, options = {}) {
  const { hasPrimaryImage = false } = options;
  const lower = cleanText.toLowerCase();

  // 1. Detect Creator match
  let matchedCreator = null;
  for (const [_, data] of Object.entries(CREATOR_PATTERNS)) {
    if (data.match.test(lower)) {
      matchedCreator = data;
      break;
    }
  }

  // 2. Detect Genre match
  let matchedGenre = null;
  for (const [_, data] of Object.entries(GENRE_PATTERNS)) {
    if (data.match.test(lower)) {
      matchedGenre = data;
      break;
    }
  }

  const hasExplicitHuman = EXPLICIT_HUMAN_REGEX.test(lower);
  const isAnimal = ANIMAL_REGEX.test(lower);
  const isVehicle = VEHICLE_REGEX.test(lower);
  const isObject = OBJECT_REGEX.test(lower);
  const isSoloCreature = SOLO_CREATURE_REGEX.test(lower) && !hasExplicitHuman && !matchedCreator && !hasPrimaryImage;

  // Human Streamer classification:
  // Must have primary image, matched creator, or explicit human keyword (streamer, youtuber, etc.)
  // When an animal/vehicle/object is the sole subject (e.g. "a cat with mouth open"),
  // it is strictly non-human even if words like "reaction" or "face" or "open mouth" are present.
  let isHumanStreamer = false;
  let subjectCategory = 'streamer';

  if (hasPrimaryImage || matchedCreator) {
    isHumanStreamer = true;
    subjectCategory = matchedCreator ? 'creator' : 'streamer';
  } else if (hasExplicitHuman) {
    isHumanStreamer = true;
    subjectCategory = 'streamer';
  } else if (isAnimal) {
    isHumanStreamer = false;
    subjectCategory = 'animal';
  } else if (isVehicle) {
    isHumanStreamer = false;
    subjectCategory = 'vehicle';
  } else if (isObject) {
    isHumanStreamer = false;
    subjectCategory = 'object';
  } else if (isSoloCreature) {
    isHumanStreamer = false;
    subjectCategory = 'creature';
  } else {
    // Default fallback: if prompt mentions game world or action without human, treat as general scene
    isHumanStreamer = false;
    subjectCategory = 'general';
  }

  // Extract explicit threat entity if present
  let matchedThreat = null;
  const threatMatch = cleanText.match(THREAT_ENTITY_REGEX);
  if (threatMatch && isHumanStreamer) {
    matchedThreat = threatMatch[0].trim();
  }

  // Extract clean subject description isolated from threat and world
  let cleanSubjectDesc = cleanText;
  if (isHumanStreamer && matchedThreat) {
    // Strip interaction phrase like "running from mutant zombie" or "facing zombie"
    const interactionRegex = new RegExp(
      `\\s*(?:running\\s+from|fleeing\\s+from|chased\\s+by|facing|fighting|versus|vs\\.?|against|attacked\\s+by|escaping\\s+from|and|with)\\s+.*$`,
      'i'
    );
    const stripped = cleanText.replace(interactionRegex, '').trim();
    if (stripped) {
      cleanSubjectDesc = stripped;
    }
  }

  return {
    isHumanStreamer,
    isNonHuman: !isHumanStreamer,
    subjectCategory,
    matchedCreator,
    matchedGenre,
    matchedThreat,
    cleanSubjectDesc,
  };
}

/**
 * enhancePrompt(rawPrompt, options) -> string
 *
 * Implements Context-Aware Three-Zone Spatial Composition Grammar (R3):
 *   Zone 1 (Foreground Subject): Explicit positioning ('foreground solo subject on the right'
 *     or 'foreground streamer portrait on the right'), strict single-subject tags ('solo,
 *     exactly one person, single subject, sharp focus'), clean anatomy and facial features
 *     ('clean skin, realistic skin texture, coherent eyes, expressive mouth, zero monster features,
 *     zero mutant traits').
 *   Zone 2 (Optical Depth Demarcation): 'optical bokeh, shallow depth of field, blurred
 *     background, strong depth separation between foreground and background'.
 *   Zone 3 (Distant Background): Game world threats, monsters, environment placed strictly
 *     in the background / opposite third ('in the deep background on the left', 'distant
 *     Minecraft creeper behind subject, isolated from foreground character').
 */
function enhancePrompt(rawPrompt = '', options = {}) {
  const {
    hasPrimaryImage = false,
    thumbnailText = '',
    qualityMode = 'ultra',
  } = options;

  let text = (rawPrompt || '').trim();

  // Strip technical trigger phrases if user accidentally typed them
  let cleanUserText = text
    .replace(/traunathumb,?\s*/gi, '')
    .replace(/professional youtube gaming thumbnail,?\s*/gi, '')
    .replace(/\s+/g, ' ')
    .trim();

  // If prompt is empty, provide a clean default streamer prompt
  if (!cleanUserText) {
    cleanUserText = 'screaming gaming streamer with headphones';
  }

  const TRIGGER = 'traunathumb, professional youtube gaming thumbnail';

  // Extract subject type and semantic metadata
  const meta = detectSubjectType(cleanUserText, { hasPrimaryImage });

  // ── ZONE 1: Foreground Subject Zone ─────────────────────────────
  let zone1 = '';

  if (meta.isHumanStreamer) {
    // Explicit positioning & strict single-subject singularity tags
    const pos = 'in foreground on the right, foreground streamer portrait on the right';
    const singularity = 'solo, exactly one person, single subject, sharp focus';
    const anatomy = 'clean skin, realistic skin texture, coherent eyes, expressive mouth, zero monster features, zero mutant traits';

    let specificDesc = '';
    if (meta.matchedCreator) {
      specificDesc = meta.matchedCreator.face;
    } else if (hasPrimaryImage) {
      specificDesc = 'uploaded streamer portrait with expressive reaction face';
    } else {
      // Use clean isolated subject text, ensuring streamer identity is clear and threat is removed from Zone 1
      specificDesc = meta.cleanSubjectDesc || cleanUserText;
    }

    zone1 = `${pos}, ${singularity}, ${anatomy}, ${specificDesc}`;
  } else {
    // Non-Human Subject (animal, vehicle, object, or solitary creature)
    // NEVER inject human streamer face
    const pos = 'in foreground on the right, foreground solo subject on the right';
    const singularity = 'solo, single subject, sharp focus';

    let anatomy = '';
    if (meta.subjectCategory === 'animal') {
      anatomy = 'clean mouth anatomy, realistic fur, coherent eyes, expressive mouth, zero monster features, zero mutant traits';
    } else if (meta.subjectCategory === 'vehicle') {
      anatomy = 'flawless metallic reflections, realistic textures, coherent geometry, zero monster features, zero mutant traits';
    } else if (meta.subjectCategory === 'object') {
      anatomy = 'sharp details, realistic textures, coherent geometry, zero monster features, zero mutant traits';
    } else {
      anatomy = 'coherent anatomy, realistic textures, zero duplicate heads, zero conjoined features';
    }

    zone1 = `${pos}, ${singularity}, ${anatomy}, ${cleanUserText}`;
  }

  // ── ZONE 2: Optical Depth Demarcation ───────────────────────────
  const zone2 = 'optical bokeh, shallow depth of field, blurred background, strong depth separation between foreground and background, optical bokeh separation, depth of field separation';

  // ── ZONE 3: Distant Background Zone ─────────────────────────────
  const isolationTarget = meta.isHumanStreamer ? 'isolated from foreground character' : 'isolated from foreground subject';
  let zone3 = 'in the deep background on the left';

  if (meta.matchedThreat) {
    zone3 += `, distant ${meta.matchedThreat} behind subject, ${isolationTarget}`;
  } else if (meta.matchedGenre && meta.matchedGenre.threat && meta.isHumanStreamer) {
    zone3 += `, ${meta.matchedGenre.threat}`;
  }

  if (meta.matchedGenre) {
    zone3 += `, ${meta.matchedGenre.world}, ${isolationTarget}`;
  } else {
    if (meta.subjectCategory === 'animal') {
      zone3 += `, soft blurred living room interior, ${isolationTarget}`;
    } else if (meta.subjectCategory === 'vehicle') {
      zone3 += `, distant highway cityscape, ${isolationTarget}`;
    } else if (meta.subjectCategory === 'object') {
      zone3 += `, soft blurred ambient room, ${isolationTarget}`;
    } else {
      zone3 += `, colorful modern gaming room studio, ${isolationTarget}`;
    }
  }

  // ── ZONE 4: Lighting & Cinematography ───────────────────────────
  const lighting = meta.matchedCreator
    ? meta.matchedCreator.lighting
    : 'cinematic high contrast lighting, vibrant rim lights';

  // ── ZONE 5: Format & Model Directives ───────────────────────────
  const format = 'professional YouTube thumbnail composition, 16:9 widescreen aspect ratio, 16:9, no text, no watermark, no split screen, no collage';

  // Assemble full prompt
  const fullPrompt = [
    TRIGGER,
    zone1,
    zone2,
    zone3,
    lighting,
    format,
  ].filter(Boolean).join(', ');

  return fullPrompt;
}

// Attach constants and helpers to enhancePrompt
enhancePrompt.ANTI_HYBRID_NEGATIVE_MATRIX = ANTI_HYBRID_NEGATIVE_MATRIX;
enhancePrompt.DEFAULT_NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT;
enhancePrompt.buildNegativePrompt = buildNegativePrompt;

module.exports = {
  enhancePrompt,
  detectSubjectType,
  buildNegativePrompt,
  ANTI_HYBRID_NEGATIVE_MATRIX,
  DEFAULT_NEGATIVE_PROMPT,
  CREATOR_PATTERNS,
  GENRE_PATTERNS,
};

