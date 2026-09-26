/**
 * Tràuna AI — services/compositeService.js
 *
 * Post-processing pipeline:
 *   1. Crop/resize to 1280x720 (16:9)
 *   2. Overlay thumbnail text (programmatic, separate from AI)
 *   3. Export PNG and JPG
 */

'use strict';

const path  = require('path');
const fs    = require('fs');
const sharp = require('sharp');
const { v4: uuid } = require('uuid');

const TARGET_WIDTH  = 1280;
const TARGET_HEIGHT = 720;

/**
 * compose(options) -> Promise<string>
 *
 * @param {object} options
 * @param {string}  options.generatedImagePath - Path to AI-generated image
 * @param {string|null} options.primaryImagePath - Optional uploaded primary image
 * @param {string}  options.thumbnailText      - Optional text overlay
 * @param {string}  options.textColor          - Optional hex color (default #FFFFFF)
 * @param {string}  options.textPosition       - 'bottom' or 'top' (default 'bottom')
 * @param {string}  options.outputDir          - Directory to write output
 * @returns {Promise<string>} - Path to final PNG
 */
async function compose(options) {
  const {
    generatedImagePath,
    primaryImagePath = null,
    thumbnailText = '',
    textColor = '#FFFFFF',
    textPosition = 'bottom',
    outputDir,
  } = options;

  const id         = uuid();
  const pngOut     = path.join(outputDir, `thumbnail_${id}.png`);
  const jpgOut     = path.join(outputDir, `thumbnail_${id}.jpg`);

  fs.mkdirSync(outputDir, { recursive: true });

  // ── Step 1: Downsample generated master to strictly 1280x720 via Lanczos3 ──
  let pipeline = sharp(generatedImagePath).resize(TARGET_WIDTH, TARGET_HEIGHT, {
    fit:      'cover',
    position: 'centre',
    kernel:   sharp.kernel.lanczos3,
  });

  // ── Step 2: Build SVG text overlay (if text provided) ───────
  if (thumbnailText && thumbnailText.trim()) {
    const svgOverlay = buildTextSvg(thumbnailText.trim(), TARGET_WIDTH, TARGET_HEIGHT, textColor, textPosition);
    const svgBuffer  = Buffer.from(svgOverlay);

    const baseBuffer = await pipeline.png().toBuffer();

    pipeline = sharp(baseBuffer).composite([
      { input: svgBuffer, top: 0, left: 0 },
    ]);
  }

  // ── Step 3: Export PNG strictly 1280x720 ─────────────────────
  await pipeline
    .resize(TARGET_WIDTH, TARGET_HEIGHT, {
      fit:      'cover',
      position: 'centre',
      kernel:   sharp.kernel.lanczos3,
    })
    .png({ quality: 90 })
    .toFile(pngOut);

  // Guarantee the final output image is strictly 1280x720
  const metadata = await sharp(pngOut).metadata();
  if (metadata.width !== TARGET_WIDTH || metadata.height !== TARGET_HEIGHT) {
    throw new Error(`Composed thumbnail dimensions ${metadata.width}x${metadata.height} do not match strictly required ${TARGET_WIDTH}x${TARGET_HEIGHT}`);
  }

  // ── Step 4: Export JPG (web-friendly) strictly 1280x720 ──────
  await sharp(pngOut)
    .resize(TARGET_WIDTH, TARGET_HEIGHT, {
      fit:      'cover',
      position: 'centre',
      kernel:   sharp.kernel.lanczos3,
    })
    .jpeg({ quality: 88 })
    .toFile(jpgOut);

  console.log(`[Composite] PNG: ${pngOut} (${TARGET_WIDTH}x${TARGET_HEIGHT} Lanczos3)`);
  console.log(`[Composite] JPG: ${jpgOut} (${TARGET_WIDTH}x${TARGET_HEIGHT})`);

  return pngOut;
}

/**
 * buildTextSvg(text, width, height, textColor, textPosition) -> string
 *
 * Creates an SVG with bold viral gaming-style text overlay.
 */
function buildTextSvg(text, width, height, textColor = '#FFFFFF', textPosition = 'bottom') {
  const isTop      = textPosition === 'top';
  const fontSize   = Math.round(height * 0.13);   // ~94px on 720p
  const textY      = isTop ? Math.round(height * 0.18) : height - Math.round(height * 0.08);
  const shadowBlur = Math.round(fontSize * 0.15);
  const strokeW    = Math.round(fontSize * 0.06);

  // Escape XML special chars
  const safeText = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const bgRectY = isTop ? 0 : height * 0.6;
  const gradY1 = isTop ? '1' : '0';
  const gradY2 = isTop ? '0' : '1';

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
  <defs>
    <filter id="shadow">
      <feDropShadow dx="0" dy="4" stdDeviation="${shadowBlur}" flood-color="rgba(0,0,0,0.95)" />
    </filter>
    <linearGradient id="textBg" x1="0" y1="${gradY1}" x2="0" y2="${gradY2}">
      <stop offset="0%" stop-color="rgba(0,0,0,0)" />
      <stop offset="100%" stop-color="rgba(0,0,0,0.8)" />
    </linearGradient>
  </defs>

  <!-- Gradient backing for readability -->
  <rect x="0" y="${bgRectY}" width="${width}" height="${height * 0.4}"
        fill="url(#textBg)" />

  <!-- Outline / stroke pass -->
  <text
    x="${width / 2}" y="${textY}"
    text-anchor="middle"
    font-family="Impact, Arial Black, sans-serif"
    font-size="${fontSize}"
    font-weight="900"
    fill="none"
    stroke="#000000"
    stroke-width="${strokeW}"
    paint-order="stroke"
    letter-spacing="4"
  >${safeText}</text>

  <!-- Main text with custom color -->
  <text
    x="${width / 2}" y="${textY}"
    text-anchor="middle"
    font-family="Impact, Arial Black, sans-serif"
    font-size="${fontSize}"
    font-weight="900"
    fill="${textColor}"
    filter="url(#shadow)"
    letter-spacing="4"
  >${safeText}</text>
</svg>`;
}

module.exports = { compose };
