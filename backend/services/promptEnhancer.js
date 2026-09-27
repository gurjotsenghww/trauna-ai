/**
 * Tràuna AI — services/promptEnhancer.js
 *
 * Compact Spatial Composer & Anti-Duplication Prompt Engine.
 *
 * CRITICAL ARCHITECTURAL CONSTRAINTS:
 * 1. CLIP 77-Token Hard Limit: SDXL/Diffusers CLIP text encoder truncates prompts past
 *    token 77. Any prompt exceeding 77 tokens loses all background/environment context,
 *    flooding the UNet with subject tokens and causing 2-headed conjoined duplicates.
 * 2. Compact Spatial Composition: Total prompt is strictly kept <= 45 words (~50-60 tokens).
 * 3. Two-Sided Latent Separation:
 *    - Foreground subject is anchored to the right ('solo ... on the right')
 *    - Background / world / threat is anchored to the left ('... on the left')
 *    - LoRA trigger: 'traunathumb, youtube thumbnail'
 *    - Format suffix: 'cinematic lighting, 16:9'
 * 4. Zero Redundant Boilerplate: Does not inject repetitive synonym phrases that waste tokens.
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
    face: 'BeastBoyShub screaming reaction face with headphones',
    defaultWorld: 'Minecraft hardcore nether fortress with glowing lava',
  },
  technogamerz: {
    match: /\b(technogamerz|techno\s*gamerz|ujjw?al)\b/i,
    face: 'TechnoGamerz focused intense gaming reaction',
    defaultWorld: 'Los Santos neon city street with supercar',
  },
  mrbeast: {
    match: /\b(mrbeast|mr\s*beast|jimmy)\b/i,
    face: 'MrBeast shocked reaction face with open mouth',
    defaultWorld: 'dangerous survival challenge arena',
  },
  coryxkenshin: {
    match: /\b(coryxkenshin|cory\s*kenshin|shogun)\b/i,
    face: 'CoryxKenshin terrified reaction face screaming',
    defaultWorld: 'dark haunted hallway with eerie shadows',
  },
  markiplier: {
    match: /\b(markiplier|mark)\b/i,
    face: 'Markiplier horrified screaming reaction with headphones',
    defaultWorld: 'dark abandoned asylum with flickering lights',
  },
  pewdiepie: {
    match: /\b(pewdiepie|felix)\b/i,
    face: 'PewDiePie energetic gaming reaction face with headphones',
    defaultWorld: 'colorful modern gaming studio',
  },
  souravjoshi: {
    match: /\b(sourav\s*joshi|sourav)\b/i,
    face: 'Sourav Joshi surprised vlogger reaction face',
    defaultWorld: 'scenic mountain vlog landscape with vivid sunlight',
  },
  carryminati: {
    match: /\b(carryminati|carry|ajey)\b/i,
    face: 'CarryMinati intense expressive reaction face with mic',
    defaultWorld: 'high contrast neon studio background',
  },
};

// ── Game / Genre World & Threat Knowledge Base ───────────────────
const GENRE_PATTERNS = {
  minecraft: {
    match: /\b(minecraft|mc|creeper|nether|enderman|herobrine|warden|diamond|hardcore|redstone|crafting|steve|alex|piglin|wither)\b/i,
    world: 'blocky 3D Minecraft world with glowing torches',
  },
  horror: {
    match: /\b(horror|resident\s*evil|re4|re2|re8|biohazard|zombie|zombies|scary|ghost|haunted|fnaf|five\s*nights|slender|chucky|jumpscare|asylum)\b/i,
    world: 'dark terrifying abandoned corridor with fog and eerie shadows',
  },
  gta: {
    match: /\b(gta|gta5|gta\s*v|grand\s*theft\s*auto|heist|los\s*santos|trevor|franklin|michael|supercar)\b/i,
    world: 'Los Santos city street with neon lights and sports car',
  },
  fps: {
    match: /\b(fps|shooter|valorant|cod|call\s*of\s*duty|warzone|pubg|apex|clutch|csgo|counter\s*strike)\b/i,
    world: 'tactical arena with smoke haze and bullet tracers',
  },
  vlogs: {
    match: /\b(vlog|vlogs|vlogger|irl|travel|daily\s*vlog|challenge)\b/i,
    world: 'bright scenic outdoor vlog location with golden hour sunlight',
  },
  roblox: {
    match: /\b(roblox|bloxd|obby|brookhaven)\b/i,
    world: 'colorful blocky obstacle course world',
  },
};

// ── Subject Classification Lexicons ──────────────────────────────
const ANIMAL_REGEX = /\b(cat|cats|kitten|kittens|kitty|dog|dogs|puppy|puppies|canine|feline|pet|pets|bird|birds|parrot|parrots|eagle|eagles|owl|owls|fish|shark|sharks|hamster|hamsters|lion|lions|tiger|tigers|bear|bears|wolf|wolves|fox|foxes|rabbit|rabbits|bunny|bunnies|horse|horses|monkey|monkeys|ape|apes|gorilla|gorillas|elephant|elephants|dinosaur|dinosaurs|dino|spider|spiders|snake|snakes|frog|frogs|lizard|lizards|duck|ducks|cow|pigs|pig|sheep|chicken|panda|koala|cheetah|leopard)\b/i;

const VEHICLE_REGEX = /\b(car|cars|supercar|supercars|sports\s*car|sports\s*cars|hypercar|hypercars|vehicle|vehicles|truck|trucks|tank|tanks|plane|planes|airplane|airplanes|jet|jets|fighter\s*jet|helicopter|helicopters|boat|boats|ship|ships|submarine|motorcycle|motorcycles|bike|bikes|motorbike|train|trains|rocket|rockets|spaceship|spaceships|ufo|mech|mecha|robot|robots)\b/i;

const OBJECT_REGEX = /\b(sword|swords|blade|shield|shields|gun|guns|rifle|rifles|pistol|pistols|weapon|weapons|bow|armor|helmet|diamond|diamonds|crown|gold|treasure|chest|coin|coins|potion|potions|orb|crystal|crystals|portal|card|cards|burger|pizza|computer|laptop|phone)\b/i;

const SOLO_CREATURE_REGEX = /\b(creeper|mutant\s*creeper|enderman|warden|herobrine|wither|zombie|zombies|skeleton|skeletons|dragon|dragons|demon|demons|alien|aliens|monster|monsters|ghost|ghosts|animatronic|animatronics)\b/i;

const EXPLICIT_HUMAN_REGEX = /\b(streamer|streamers|youtuber|youtubers|creator|creators|gamer|gamers|vlogger|vloggers|facecam|selfie|player|players|person|people|human|humans|man|men|woman|women|boy|boys|girl|girls|guy|guys|dude|bro)\b/i;

/**
 * Analyzes raw prompt to extract subject classification, creator, and genre.
 */
function detectSubjectType(cleanText, options = {}) {
  const { hasPrimaryImage = false } = options;
  const lower = cleanText.toLowerCase();

  let matchedCreator = null;
  for (const [_, data] of Object.entries(CREATOR_PATTERNS)) {
    if (data.match.test(lower)) {
      matchedCreator = data;
      break;
    }
  }

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
    isHumanStreamer = true;
    subjectCategory = 'streamer';
  }

  return {
    isHumanStreamer,
    isNonHuman: !isHumanStreamer,
    subjectCategory,
    matchedCreator,
    matchedGenre,
  };
}

/**
 * Parses user input into subject, background, and atmosphere components.
 */
function parseSubjectAndBackground(cleanUserText, meta) {
  let text = cleanUserText
    .replace(/\b(only\s+one\s+face|single\s+face|no\s+two\s+heads|exactly\s+one\s+person|16:9|widescreen)\b/gi, '')
    .replace(/\s+/g, ' ')
    .trim();

  const clauses = text.split(',').map(c => c.trim()).filter(Boolean);

  if (clauses.length >= 2) {
    const rawSubj = clauses[0];
    const rawBg = clauses[1];
    const rawAtmos = clauses.slice(2).join(', ');
    return { rawSubj, rawBg, rawAtmos };
  }

  // Single-clause prompt
  let defaultBg = 'colorful modern gaming room studio';
  if (meta.matchedCreator && meta.matchedCreator.defaultWorld) {
    defaultBg = meta.matchedCreator.defaultWorld;
  } else if (meta.matchedGenre && meta.matchedGenre.world) {
    defaultBg = meta.matchedGenre.world;
  } else if (meta.subjectCategory === 'animal') {
    defaultBg = 'living room interior';
  } else if (meta.subjectCategory === 'vehicle') {
    defaultBg = 'city highway with neon lights';
  } else if (meta.subjectCategory === 'object') {
    defaultBg = 'dungeon interior with glowing lights';
  }

  return {
    rawSubj: text,
    rawBg: defaultBg,
    rawAtmos: '',
  };
}

function truncateWords(str, maxWords) {
  const words = str.split(/\s+/).filter(Boolean);
  if (words.length <= maxWords) return str;
  let cut = words.slice(0, maxWords).join(' ');
  return cut.replace(/\s+(with|and|in|at|on|of|the|a|an|to|for|from|shooting|fiery)$/i, '').trim();
}

/**
 * enhancePrompt(rawPrompt, options) -> string
 *
 * Guaranteed <= 45 words / <= 60 CLIP tokens:
 * Format:
 *   traunathumb, youtube thumbnail, solo <subject> on the right, <background> on the left, cinematic lighting, 16:9
 */
function enhancePrompt(rawPrompt = '', options = {}) {
  const {
    hasPrimaryImage = false,
  } = options;

  let text = (rawPrompt || '').trim();

  // Strip technical trigger phrases if user accidentally typed them
  let cleanUserText = text
    .replace(/traunathumb,?\s*/gi, '')
    .replace(/professional\s+youtube\s+gaming\s+thumbnail,?\s*/gi, '')
    .replace(/youtube\s+gaming\s+thumbnail,?\s*/gi, '')
    .replace(/youtube\s+thumbnail,?\s*/gi, '')
    .replace(/\s+/g, ' ')
    .trim();

  if (!cleanUserText) {
    cleanUserText = 'shocked gaming streamer with headphones';
  }

  const meta = detectSubjectType(cleanUserText, { hasPrimaryImage });
  const { rawSubj, rawBg, rawAtmos } = parseSubjectAndBackground(cleanUserText, meta);

  // 1. Clean subject
  let subj = rawSubj.replace(/\b(on\s+the\s+(?:right|left)|in\s+foreground)\b/gi, '').trim();
  if (meta.matchedCreator && subj.length < 20) {
    subj = meta.matchedCreator.face;
  }
  if (!/\b(solo|single)\b/i.test(subj)) {
    subj = `solo ${subj.replace(/^a\s+/i, '')}`;
  }

  // ── Orientation anchor: force camera-facing portrait to prevent rear/silhouette renders ──
  if (meta.isHumanStreamer && !/\b(front.facing|looking at|facing camera|portrait)\b/i.test(subj)) {
    subj = subj + ', front-facing, looking at camera';
  }

  // ── Style override: prevent blocky avatar bleed when Minecraft/Roblox genre + human subject ──
  const BLOCKY_GENRE_REGEX = /\b(minecraft|mc|roblox|bloxd|blocky)\b/i;
  if (meta.isHumanStreamer && BLOCKY_GENRE_REGEX.test(rawBg + ' ' + rawSubj) && !/photorealistic/.test(subj)) {
    subj = subj.replace(/^solo\s+/i, 'solo photorealistic human ');
  }

  // Token budget: subject zone capped at 18 words (was 15) to accommodate orientation tokens
  const subjWords = truncateWords(subj, 18);

  // 2. Clean background
  let bg = rawBg.replace(/\b(in\s+(?:the\s+)?background|on\s+the\s+(?:right|left))\b/gi, '').trim();
  const bgWords = truncateWords(bg, 16);

  // 3. Atmosphere / Lighting suffix
  let atmosPart = '';
  if (rawAtmos) {
    const atmosClean = rawAtmos
      .replace(/\b(cinematic\s+lighting|16:9|widescreen|high\s+contrast)\b/gi, '')
      .replace(/\s+/g, ' ')
      .trim();
    if (atmosClean) {
      atmosPart = `${truncateWords(atmosClean, 4)}, `;
    }
  }

  // 4. Assemble full prompt
  const fullPrompt = `traunathumb, youtube thumbnail, ${subjWords} on the right, ${bgWords} on the left, ${atmosPart}cinematic lighting, 16:9`;

  return fullPrompt;
}

/**
 * getDualStreamPrompts(rawPrompt, options) -> { subjectPrompt, worldPrompt, isDualStreamEligible }
 *
 * Decomposes user request into two specialized prompts:
 * 1. Stream A (768x768 Portrait): Dedicated camera-facing streamer reaction portrait.
 * 2. Stream B (1024x576 Scene): Clean game environment with zero human tokens.
 */
function getDualStreamPrompts(rawPrompt = '', options = {}) {
  const { hasPrimaryImage = false } = options;
  let text = (rawPrompt || '').trim();

  let cleanUserText = text
    .replace(/traunathumb,?\s*/gi, '')
    .replace(/professional\s+youtube\s+gaming\s+thumbnail,?\s*/gi, '')
    .replace(/youtube\s+gaming\s+thumbnail,?\s*/gi, '')
    .replace(/youtube\s+thumbnail,?\s*/gi, '')
    .replace(/\s+/g, ' ')
    .trim();

  if (!cleanUserText) {
    cleanUserText = 'shocked gaming streamer with headphones';
  }

  const meta = detectSubjectType(cleanUserText, { hasPrimaryImage });
  const { rawSubj, rawBg, rawAtmos } = parseSubjectAndBackground(cleanUserText, meta);

  // 1. Dedicated Subject Prompt (768x768 Portrait)
  let subj = rawSubj.replace(/\b(on\s+the\s+(?:right|left)|in\s+foreground)\b/gi, '').trim();
  if (meta.matchedCreator && subj.length < 20) {
    subj = meta.matchedCreator.face;
  }
  if (!/\b(solo|single)\b/i.test(subj)) {
    subj = `solo ${subj.replace(/^a\s+/i, '')}`;
  }
  if (meta.isHumanStreamer && !/\b(front.facing|looking at|facing camera|portrait)\b/i.test(subj)) {
    subj = subj + ', front-facing, looking directly at camera';
  }
  const subjWords = truncateWords(subj, 20);
  const subjectPrompt = `traunathumb, ${subjWords}, expressive reaction face with open mouth, wearing LED gaming headphones, sharp focus, 1:1 portrait`;

  // 2. Dedicated World Prompt (1024x576 Scene)
  let bg = rawBg.replace(/\b(in\s+(?:the\s+)?background|on\s+the\s+(?:right|left))\b/gi, '').trim();
  const bgWords = truncateWords(bg, 18);
  let atmosClean = (rawAtmos || '').replace(/\b(cinematic\s+lighting|16:9|widescreen|high\s+contrast)\b/gi, '').trim();
  const atmosPart = atmosClean ? `${truncateWords(atmosClean, 5)}, ` : '';
  const worldPrompt = `traunathumb, ${bgWords}, ${atmosPart}cinematic lighting, vibrant saturated atmosphere, 16:9 widescreen`;

  return {
    subjectPrompt,
    worldPrompt,
    isDualStreamEligible: meta.isHumanStreamer,
  };
}

// Attach constants and helpers
enhancePrompt.ANTI_HYBRID_NEGATIVE_MATRIX = ANTI_HYBRID_NEGATIVE_MATRIX;
enhancePrompt.DEFAULT_NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT;
enhancePrompt.buildNegativePrompt = buildNegativePrompt;
enhancePrompt.getDualStreamPrompts = getDualStreamPrompts;

module.exports = {
  enhancePrompt,
  getDualStreamPrompts,
  detectSubjectType,
  buildNegativePrompt,
  ANTI_HYBRID_NEGATIVE_MATRIX,
  DEFAULT_NEGATIVE_PROMPT,
  CREATOR_PATTERNS,
  GENRE_PATTERNS,
};

