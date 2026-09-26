/**
 * Tràuna AI — routes/generate.js
 * POST /api/generate
 */

'use strict';

const express     = require('express');
const multer      = require('multer');
const path        = require('path');
const fs          = require('fs');
const { v4: uuid } = require('uuid');

const promptService    = require('../services/promptService');
const promptEnhancer   = require('../services/promptEnhancer');
const compositeService = require('../services/compositeService');
const aiService        = require('../services/aiService');

const router = express.Router();

// ── Multer storage (local disk) ─────────────────────────────────
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    const uploadDir = path.join(__dirname, '..', 'uploads');
    fs.mkdirSync(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (_req, file, cb) => {
    const ext  = path.extname(file.originalname).toLowerCase();
    const name = `${uuid()}${ext}`;
    cb(null, name);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 }, // 20 MB per file
  fileFilter: (_req, file, cb) => {
    const allowed = /\.(jpg|jpeg|png|webp|gif)$/i;
    if (!allowed.test(path.extname(file.originalname))) {
      return cb(new Error('Only image files are allowed.'));
    }
    cb(null, true);
  },
});

const uploadFields = upload.fields([
  { name: 'primaryImage',      maxCount: 1  },
  { name: 'additionalImages',  maxCount: 10 },
  { name: 'gameplayScreenshot', maxCount: 1 },
]);

// ── POST /api/generate ──────────────────────────────────────────
router.post('/', uploadFields, async (req, res) => {
  const startTime = Date.now();

  try {
    const {
      prompt: rawPrompt = '',
      game            = '',
      videoDescription = '',
      thumbnailText   = '',
      textColor       = '#FFFFFF',
      textPosition    = 'bottom',
      qualityMode     = null,
      negativePrompt: customNegative = '',
      guidanceScale: customGuidance = null,
      steps: customSteps = null,
    } = req.body;

    // ── Collect uploaded file paths ────────────────────────────
    const primaryImagePath = req.files?.primaryImage?.[0]?.path || null;
    const additionalImagePaths = (req.files?.additionalImages || []).map(f => f.path);
    const screenshotPath = req.files?.gameplayScreenshot?.[0]?.path || null;

    const requestedMode = (qualityMode || req.body.mode || req.body.bucket || 'ultra').toString().toLowerCase();
    const isFast = requestedMode === 'fast';
    const normalizedQualityMode = isFast ? 'fast' : (requestedMode === 'standard' ? 'standard' : 'ultra');

    let prompt = '';
    if (rawPrompt && rawPrompt.trim()) {
      // Smart Auto-Enhance: User never has to type trigger tokens
      prompt = promptEnhancer.enhancePrompt(rawPrompt, {
        hasPrimaryImage: !!primaryImagePath,
        thumbnailText,
        qualityMode: normalizedQualityMode,
      });
      console.log(`\n[Generate] User Input: ${rawPrompt.trim()}`);
      console.log(`[Generate] Smart Enhanced Prompt: ${prompt.substring(0, 130)}...`);
    } else {
      if (!game) {
        return res.status(400).json({ error: 'Please describe your thumbnail idea or select a game category.' });
      }

      console.log(`\n[Generate] Game: ${game}`);
      console.log(`[Generate] Description: ${videoDescription}`);
      console.log(`[Generate] Text overlay: ${thumbnailText}`);

      prompt = promptService.buildPrompt({
        game,
        videoDescription,
        thumbnailText,
        hasPrimaryImage:    !!primaryImagePath,
        hasScreenshot:      !!screenshotPath,
        hasAdditionalImages: additionalImagePaths.length > 0,
      });
    }

    if (primaryImagePath) console.log(`[Generate] Primary image: ${primaryImagePath}`);

    const rawNegative = customNegative || req.body.negative_prompt;
    const negativePrompt = promptEnhancer.buildNegativePrompt(rawNegative);

    // SDXL Turbo is an Adversarial Diffusion Distillation (ADD) model.
    // Guidance scale must default to 0.0 (or 0.0 if not specified), and steps must be 4 max.
    // Guidance > 1.0 or steps > 4 causes latent manifold tearing and duplicate/conjoined heads.
    const rawGuidance = customGuidance != null ? customGuidance : req.body.guidance_scale;
    const guidanceScale = (rawGuidance != null && rawGuidance !== '' && !isNaN(parseFloat(rawGuidance)))
      ? parseFloat(rawGuidance)
      : 0.0;

    // Cap steps at 4 for SDXL Turbo to prevent over-denoising and secondary subject nucleation
    const defaultSteps = 4;
    const numSteps = customSteps && !isNaN(parseInt(customSteps, 10)) ? Math.min(parseInt(customSteps, 10), 4) : defaultSteps;

    // Aspect-ratio bucket: 1024x576 (~0.59 MP, 16:9)
    // Fits strictly within the SDXL single-subject attention receptive field,
    // eliminating dual-head nucleation. Downscaled/upscaled cleanly via Sharp Lanczos3 to 1280x720.
    const defaultWidth  = 1024;
    const defaultHeight = 576;
    const nativeWidth   = req.body.width ? parseInt(req.body.width, 10) : defaultWidth;
    const nativeHeight  = req.body.height ? parseInt(req.body.height, 10) : defaultHeight;

    console.log(`[Generate] Quality Mode: ${normalizedQualityMode.toUpperCase()} | Native Bucket: ${nativeWidth}x${nativeHeight} | Steps: ${numSteps} | Guidance: ${guidanceScale}`);

    // ── Call AI service (generates at native 16:9 bucket) ───────
    const generatedImagePath = await aiService.generateImage(prompt, {
      width:          nativeWidth,
      height:         nativeHeight,
      steps:          numSteps,
      negativePrompt,
      guidanceScale,
      seed:           req.body.seed != null && !isNaN(parseInt(req.body.seed, 10)) ? parseInt(req.body.seed, 10) : undefined,
    });

    // ── Composite: overlay text, downsample to strictly 1280x720 ──────
    const composeResult = await compositeService.compose({
      generatedImagePath,
      primaryImagePath,
      thumbnailText,
      textColor,
      textPosition,
      outputDir: path.join(__dirname, '..', 'outputs'),
    });

    const finalPath = typeof composeResult === 'string'
      ? composeResult
      : (composeResult.path || composeResult.filePath || composeResult.pngOut);
    const filename  = (typeof composeResult === 'object' && composeResult.filename)
      ? composeResult.filename
      : path.basename(finalPath);
    const imageUrl  = (typeof composeResult === 'object' && composeResult.url)
      ? composeResult.url
      : `/outputs/${filename}`;

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[Generate] Done in ${elapsed}s -> ${filename}`);

    res.json({
      success: true,
      prompt,
      results: [
        { url: imageUrl, filename },
      ],
      elapsed: `${elapsed}s`,
    });

  } catch (err) {
    console.error('[Generate Error]', err);
    res.status(500).json({ error: err.message || 'Generation failed.' });
  }
});

module.exports = router;
