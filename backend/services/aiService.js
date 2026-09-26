/**
 * Tràuna AI — services/aiService.js
 *
 * Model-agnostic abstraction layer.
 * The rest of the application calls generateImage(prompt, options).
 * The actual model implementation lives in the Python AI process.
 */

'use strict';

const { spawn }   = require('child_process');
const path        = require('path');
const fs          = require('fs');
const { v4: uuid } = require('uuid');

const AI_DIR      = path.join(__dirname, '..', '..', 'ai');
const PYTHON_SCRIPT = path.join(AI_DIR, 'generate.py');
const OUTPUT_DIR  = path.join(__dirname, '..', 'outputs');

// Detect python executable
const PYTHON_BIN = process.platform === 'win32' ? 'python' : 'python3';
const VENV_PYTHON = path.join(AI_DIR, 'venv', 'Scripts', 'python.exe');  // Windows venv

function getPythonBin() {
  if (fs.existsSync(VENV_PYTHON)) return VENV_PYTHON;
  return PYTHON_BIN;
}

// FIFO queue ensures single GPU inference at a time on 6GB VRAM hardware
let queuePromise = Promise.resolve();

/**
 * generateImage(prompt, options) -> Promise<string>
 *
 * Calls the Python AI process and returns the path to the generated image.
 *
 * @param {string} prompt    - Full image generation prompt
 * @param {object} options   - { width, height, steps, seed, negativePrompt, guidanceScale, model, bucket }
 * @returns {Promise<string>} - Absolute path to the generated output image
 */
async function generateImage(prompt, options = {}) {
  const currentTask = queuePromise.then(() => _executeGenerate(prompt, options));
  queuePromise = currentTask.catch(() => {});
  return currentTask;
}

function _executeGenerate(prompt, options = {}) {
  return new Promise((resolve, reject) => {
    const outputId   = uuid();
    const outputPath = path.join(OUTPUT_DIR, `raw_${outputId}.png`);

    fs.mkdirSync(OUTPUT_DIR, { recursive: true });

    const width  = options.width  || 1344;
    const height = options.height || 768;
    const steps  = options.steps  || 4;

    const args = [
      PYTHON_SCRIPT,
      '--prompt',  prompt,
      '--output',  outputPath,
      '--width',   String(width),
      '--height',  String(height),
      '--steps',   String(steps),
    ];

    const negativePrompt = options.negativePrompt || options.negative_prompt;
    if (negativePrompt) {
      args.push('--negative_prompt', String(negativePrompt));
    }

    const guidanceScale = options.guidanceScale != null ? options.guidanceScale : options.guidance_scale;
    if (guidanceScale != null) {
      args.push('--guidance_scale', String(guidanceScale));
    }

    if (options.seed != null) {
      args.push('--seed', String(options.seed));
    }

    if (options.bucket) {
      args.push('--bucket', String(options.bucket));
    }

    if (options.model) {
      args.push('--model', String(options.model));
    }

    const pythonBin = getPythonBin();
    console.log(`[AI] Running: ${pythonBin} ${PYTHON_SCRIPT}`);
    console.log(`[AI] Output:  ${outputPath}`);

    const proc = spawn(pythonBin, args, {
      cwd: AI_DIR,
      env: { ...process.env },
    });

    proc.stdout.on('data', d => process.stdout.write(`[AI] ${d}`));
    proc.stderr.on('data', d => process.stderr.write(`[AI ERR] ${d}`));

    proc.on('close', code => {
      if (code !== 0) {
        return reject(new Error(`AI process exited with code ${code}`));
      }
      if (!fs.existsSync(outputPath)) {
        return reject(new Error(`AI process finished but output file not found: ${outputPath}`));
      }
      resolve(outputPath);
    });

    proc.on('error', err => {
      reject(new Error(`Failed to start AI process: ${err.message}`));
    });
  });
}

module.exports = { generateImage };
