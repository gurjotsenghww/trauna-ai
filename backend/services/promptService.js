/**
 * Tràuna AI — services/promptService.js
 *
 * Converts game category and video description into structured, compact two-sided prompts
 * adhering strictly to the <= 60 token ceiling to eliminate duplicate heads.
 */

'use strict';

const {
  detectSubjectType,
  ANTI_HYBRID_NEGATIVE_MATRIX,
  DEFAULT_NEGATIVE_PROMPT,
  buildNegativePrompt,
} = require('./promptEnhancer');

// ── Game-specific compact visual styles ───────────────────────────
const GAME_STYLES = {
  minecraft: {
    environment: 'blocky 3D Minecraft world with glowing torches on the left',
    lighting: 'vibrant saturated rim lighting',
  },
  minecraft_hardcore: {
    environment: 'dangerous Minecraft nether fortress with glowing lava on the left',
    lighting: 'dramatic fire glow rim lighting',
  },
  fortnite: {
    environment: 'colourful Fortnite battle royale island with storm wall on the left',
    lighting: 'electric purple storm rim lighting',
  },
  gta: {
    environment: 'Los Santos city street with neon lights and sports car on the left',
    lighting: 'neon night city rim lights',
  },
  valorant: {
    environment: 'futuristic Valorant tactical arena with energy smoke on the left',
    lighting: 'sharp tactical neon rim lighting',
  },
  pubg: {
    environment: 'PUBG military battleground with distant airdrop crate on the left',
    lighting: 'golden hour combat rim lighting',
  },
  call_of_duty: {
    environment: 'Call of Duty warzone with explosive fire smoke on the left',
    lighting: 'cinematic combat rim lighting',
  },
  apex_legends: {
    environment: 'futuristic Apex Legends arena with volumetric beams on the left',
    lighting: 'high-tech scifi rim lighting',
  },
  roblox: {
    environment: 'colourful blocky Roblox obstacle course world on the left',
    lighting: 'bright vibrant lighting',
  },
  among_us: {
    environment: 'dark spaceship corridors with glowing emergency light on the left',
    lighting: 'red alert emergency lighting',
  },
  horror: {
    environment: 'dark terrifying abandoned corridor with fog and eerie shadows on the left',
    lighting: 'horror flashlight rim lighting',
  },
  vlogs: {
    environment: 'bright scenic outdoor vlog location with golden hour sunlight on the left',
    lighting: 'golden hour sunflare lighting',
  },
  challenges: {
    environment: 'dangerous high-stakes survival challenge arena on the left',
    lighting: 'bright high saturation studio lighting',
  },
  other: {
    environment: 'colorful modern gaming room studio on the left',
    lighting: 'cinematic vibrant rim lights',
  },
};

const DEFAULT_STYLE = GAME_STYLES.other;

/**
 * buildPrompt(options) -> string
 *
 * Assembles a concise, 2-sided prompt strictly <= 45 words / <= 60 tokens:
 *   traunathumb, youtube thumbnail, solo <subject> on the right, <environment> on the left, cinematic lighting, 16:9
 */
function buildPrompt(options = {}) {
  const {
    game = 'other',
    videoDescription = '',
    hasPrimaryImage = false,
  } = options;

  const style = GAME_STYLES[game] || DEFAULT_STYLE;
  const desc = (videoDescription || '').trim();

  let subjectPart = '';
  if (hasPrimaryImage) {
    subjectPart = 'solo uploaded streamer portrait with expressive reaction face on the right';
  } else if (desc) {
    const meta = detectSubjectType(desc, { hasPrimaryImage });
    let s = desc.replace(/\b(on\s+the\s+(?:right|left)|in\s+foreground)\b/gi, '').trim();
    if (!/\b(solo|single)\b/i.test(s)) {
      s = `solo ${s}`;
    }
    const words = s.split(/\s+/).slice(0, 14).join(' ');
    subjectPart = `${words} on the right`;
  } else {
    subjectPart = 'solo screaming excited gaming streamer with headphones on the right';
  }

  const bgPart = style.environment;
  const lighting = style.lighting ? `${style.lighting}, ` : '';

  const fullPrompt = `traunathumb, youtube thumbnail, ${subjectPart}, ${bgPart}, ${lighting}cinematic lighting, 16:9`;

  return fullPrompt;
}

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
