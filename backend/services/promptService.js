/**
 * Tràuna AI — services/promptService.js
 *
 * Converts simple user inputs into a structured image-generation prompt adhering to the
 * Context-Aware Three-Zone Spatial Composition Grammar (Milestone 3 / R3).
 *
 * Zone 1: Foreground Subject (Solo streamer or solo subject, clean skin/anatomy, zero monster features)
 * Zone 2: Optical Depth Demarcation (Bokeh, shallow depth of field, strong separation)
 * Zone 3: Distant Background (Game world & threats placed strictly in the background, isolated from character)
 */

'use strict';

const {
  detectSubjectType,
  ANTI_HYBRID_NEGATIVE_MATRIX,
  DEFAULT_NEGATIVE_PROMPT,
  buildNegativePrompt,
} = require('./promptEnhancer');

// ── Game-specific visual styles adhering to 3-Zone Spatial Separation ────────
const GAME_STYLES = {
  minecraft: {
    environment:  'in the deep background on the left, blocky 3D voxel Minecraft world, distant dirt and stone landscape, mountains, glowing torches, isolated from foreground character',
    backgroundThreat: 'distant Minecraft creeper behind subject, isolated from foreground character',
    lighting:     'bright daylight, Minecraft sun, vibrant saturated rim lighting on subject silhouette',
    atmosphere:   'adventurous, epic survival, dramatic sky',
    cameraAngle:  'low angle looking up at subject, heroic perspective',
  },
  minecraft_hardcore: {
    environment:  'in the deep background on the left, dark dangerous Minecraft Nether, hardcore mode, distant lava falls, nether fortress, isolated from foreground character',
    backgroundThreat: 'distant Minecraft hostile mobs and glowing fire behind subject, isolated from foreground character',
    lighting:     'dramatic low key lighting, fire glow, flickering torches, intense rim lights',
    atmosphere:   'tense, survival, life-or-death stakes, dramatic',
    cameraAngle:  'dramatic low angle, subject looming large',
  },
  fortnite: {
    environment:  'in the deep background on the left, Fortnite battle royale island, distant colourful buildings, storm closing in, isolated from foreground character',
    backgroundThreat: 'distant battle bus and purple storm wall behind subject, isolated from foreground character',
    lighting:     'golden hour storm light, electric purple storm glow rim lighting',
    atmosphere:   'intense, last-circle tension, victory royale energy',
    cameraAngle:  'dynamic action angle, subject mid-motion',
  },
  gta: {
    environment:  'in the deep background on the left, Los Santos cityscape, neon lights, distant luxury sports cars, urban street, isolated from foreground character',
    backgroundThreat: 'distant speeding sports car and police chase behind subject, isolated from foreground character',
    lighting:     'neon night lighting, city glow, cinematic vibrant rim lights',
    atmosphere:   'cool, criminal, high stakes, urban action movie aesthetic',
    cameraAngle:  'cinematic mid-shot, urban background',
  },
  valorant: {
    environment:  'in the deep background on the left, Valorant tactical arena, futuristic military base, isolated from foreground character',
    backgroundThreat: 'distant radiant energy spike and tactical smoke behind subject, isolated from foreground character',
    lighting:     'sharp tactical lighting, agent ability glow effects, crisp rim lighting',
    atmosphere:   'competitive, precise, tactical FPS energy',
    cameraAngle:  'action shot, weapon raised, targeting',
  },
  pubg: {
    environment:  'in the deep background on the left, PUBG battleground, military zone, war-torn landscape, distant buildings, isolated from foreground character',
    backgroundThreat: 'distant airdrop crate and smoke plume behind subject, isolated from foreground character',
    lighting:     'overcast military lighting, golden sunset rim lights',
    atmosphere:   'intense military survival, last-squad energy',
    cameraAngle:  'ground-level dramatic angle',
  },
  call_of_duty: {
    environment:  'in the deep background on the left, Call of Duty warzone, military combat zone, distant explosions, smoke plumes, isolated from foreground character',
    backgroundThreat: 'distant airstrike smoke and combat vehicles behind subject, isolated from foreground character',
    lighting:     'explosive fire lighting, combat zone haze, cinematic warm rim lighting',
    atmosphere:   'intense military action, war movie aesthetic',
    cameraAngle:  'action hero low angle, dynamic combat pose',
  },
  apex_legends: {
    environment:  'in the deep background on the left, Apex Legends arena, futuristic science fiction landscape, isolated from foreground character',
    backgroundThreat: 'distant dropship and shield barrier behind subject, isolated from foreground character',
    lighting:     'science fiction volumetric lighting, legend ability aura rim lights',
    atmosphere:   'fast paced battle royale, legendary status energy',
    cameraAngle:  'dynamic power pose, slight upward angle',
  },
  roblox: {
    environment:  'in the deep background on the left, Roblox game world, blocky colourful avatar obstacle world, isolated from foreground character',
    backgroundThreat: 'distant obstacle challenge platforms behind subject, isolated from foreground character',
    lighting:     'bright vibrant lighting, cartoon style highlights',
    atmosphere:   'fun, playful, energetic, youthful gaming',
    cameraAngle:  'fun character-focused angle, expressive',
  },
  among_us: {
    environment:  'in the deep background on the left, Among Us spaceship interior, dark corridors, emergency meeting room, isolated from foreground character',
    backgroundThreat: 'distant shadowed impostor silhouette behind subject, isolated from foreground character',
    lighting:     'emergency red alert lighting, spaceship panels, eerie rim glow',
    atmosphere:   'suspicious, betrayal, social deduction tension',
    cameraAngle:  'dramatic confrontation angle',
  },
  horror: {
    environment:  'in the deep background on the left, dark terrifying haunted environment, dense fog, eerie darkness, isolated from foreground character',
    backgroundThreat: 'distant terrifying zombie silhouette behind subject, isolated from foreground character',
    lighting:     'low key horror lighting, single flashlight beam, deep shadows, cold blue rim light',
    atmosphere:   'terrifying, jump scare energy, pure horror',
    cameraAngle:  'close frightened face, extreme expression',
  },
  other: {
    environment:  'in the deep background on the left, epic gaming environment, distant vista, isolated from foreground character',
    backgroundThreat: 'distant action events behind subject, isolated from foreground character',
    lighting:     'cinematic dramatic lighting, vibrant rim lights',
    atmosphere:   'exciting, high energy, YouTube thumbnail energy',
    cameraAngle:  'dynamic engaging angle',
  },
};

const DEFAULT_STYLE = GAME_STYLES.other;

/**
 * buildPrompt(options) -> string
 *
 * @param {object} options
 * @param {string} options.game              - Game ID
 * @param {string} options.videoDescription  - User description of the video
 * @param {string} options.thumbnailText     - Optional overlay text
 * @param {boolean} options.hasPrimaryImage  - Whether a primary image was uploaded
 * @param {boolean} options.hasScreenshot    - Whether a screenshot was uploaded
 * @param {boolean} options.hasAdditionalImages
 * @returns {string}
 */
function buildPrompt(options = {}) {
  const {
    game             = 'other',
    videoDescription = '',
    thumbnailText    = '',
    hasPrimaryImage  = false,
    hasScreenshot    = false,
  } = options;

  const style = GAME_STYLES[game] || DEFAULT_STYLE;
  const desc = (videoDescription || '').trim();

  // ── TRIGGER ──────────────────────────────────────────────────
  const TRIGGER = 'traunathumb, professional youtube gaming thumbnail';

  // ── ZONE 1: Foreground Subject Zone ──────────────────────────
  let zone1 = '';
  const meta = desc ? detectSubjectType(desc, { hasPrimaryImage }) : { isHumanStreamer: true, isNonHuman: false, subjectCategory: 'streamer' };

  if (hasPrimaryImage) {
    const pos = 'in foreground on the right, foreground streamer portrait on the right';
    const singularity = 'solo, exactly one person, single subject, sharp focus';
    const anatomy = 'clean skin, realistic skin texture, coherent eyes, expressive mouth, zero monster features, zero mutant traits';
    zone1 = `${pos}, ${singularity}, ${anatomy}, expressive gaming YouTuber streamer reaction face showing strong emotion`;
  } else if (desc && meta.isNonHuman) {
    // Non-human subject described in video description
    const pos = 'in foreground on the right, foreground solo subject on the right';
    const singularity = 'solo, single subject, sharp focus';
    let anatomy = 'coherent anatomy, realistic textures, zero monster features, zero mutant traits';
    if (meta.subjectCategory === 'animal') {
      anatomy = 'clean mouth anatomy, realistic fur, coherent eyes, expressive mouth, zero monster features, zero mutant traits';
    } else if (meta.subjectCategory === 'vehicle') {
      anatomy = 'flawless metallic reflections, realistic textures, coherent geometry, zero monster features, zero mutant traits';
    }
    zone1 = `${pos}, ${singularity}, ${anatomy}, ${desc}`;
  } else {
    // Default human streamer portrait
    const pos = 'in foreground on the right, foreground streamer portrait on the right';
    const singularity = 'solo, exactly one person, single subject, sharp focus';
    const anatomy = 'clean skin, realistic skin texture, coherent eyes, expressive mouth, zero monster features, zero mutant traits';
    const reaction = desc ? desc : 'screaming excited gaming streamer reaction face with headphones';
    zone1 = `${pos}, ${singularity}, ${anatomy}, ${reaction}`;
  }

  // ── ZONE 2: Optical Depth Demarcation ────────────────────────
  const zone2 = 'optical bokeh, shallow depth of field, blurred background, strong depth separation between foreground and background, optical bokeh separation, depth of field separation';

  // ── ZONE 3: Distant Background Zone ──────────────────────────
  let zone3 = style.environment;
  if (style.backgroundThreat) {
    zone3 += `, ${style.backgroundThreat}`;
  }
  if (hasScreenshot) {
    zone3 += ', distant gameplay environment reference in far background, isolated from foreground character';
  }

  // ── ZONE 4: Lighting & Cinematography ────────────────────────
  const zone4 = `Camera angle: ${style.cameraAngle}, Lighting: ${style.lighting}, Atmosphere: ${style.atmosphere}`;

  // ── ZONE 5: Format & Model Directives ────────────────────────
  const textSpaceLine = thumbnailText
    ? `Leave clear space at the top or bottom of the composition for bold overlay text reading "${thumbnailText}", `
    : 'Leave clear space at the top for potential text overlay, ';

  const zone5 = `${textSpaceLine}Strong visual hierarchy, high contrast, vibrant saturated colours, photorealistic digital art, sharp focus on subject, professional YouTube thumbnail composition, 16:9 widescreen aspect ratio, 16:9, no text rendered by the model, no watermarks, no UI elements, no split screen, no collage`;

  // ── Assemble full 3-zone prompt ──────────────────────────────
  const fullPrompt = [
    TRIGGER,
    zone1,
    zone2,
    zone3,
    zone4,
    zone5,
  ].filter(Boolean).join(', ');

  return fullPrompt;
}

// Attach negative matrix helpers
buildPrompt.ANTI_HYBRID_NEGATIVE_MATRIX = ANTI_HYBRID_NEGATIVE_MATRIX;
buildPrompt.DEFAULT_NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT;
buildPrompt.buildNegativePrompt = buildNegativePrompt;

module.exports = {
  buildPrompt,
  GAME_STYLES,
  DEFAULT_STYLE,
  ANTI_HYBRID_NEGATIVE_MATRIX,
  DEFAULT_NEGATIVE_PROMPT,
  buildNegativePrompt,
};

