/**
 * Tràuna AI — routes/crawl.js
 * POST /api/crawl
 * Triggers YouTube thumbnail crawler for dataset curation.
 */

'use strict';

const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const router = express.Router();
const ROOT_DIR = path.join(__dirname, '..', '..');
const CRAWLER_SCRIPT = path.join(ROOT_DIR, 'crawlers', 'youtube_crawler.py');
const PYTHON_BIN = path.join(ROOT_DIR, 'ai', 'venv', 'Scripts', 'python.exe');

router.post('/', async (req, res) => {
  const {
    channel,
    playlist,
    search,
    creator,
    game,
    limit = 30,
  } = req.body;

  const args = [CRAWLER_SCRIPT, '--limit', String(limit)];

  if (channel) args.push('--channel', channel);
  if (playlist) args.push('--playlist', playlist);
  if (search) args.push('--search', search);
  if (creator) args.push('--creator', creator);
  if (game) args.push('--game', game);

  console.log(`[Crawl API] Launching: ${args.join(' ')}`);

  const proc = spawn(PYTHON_BIN, args, {
    cwd: ROOT_DIR,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });

  let output = '';
  let errorOutput = '';

  proc.stdout.on('data', (d) => {
    output += d.toString();
    process.stdout.write(`[Crawler] ${d}`);
  });

  proc.stderr.on('data', (d) => {
    errorOutput += d.toString();
    process.stderr.write(`[Crawler ERR] ${d}`);
  });

  proc.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({
        success: false,
        error: `Crawler process exited with code ${code}`,
        details: errorOutput,
      });
    }

    res.json({
      success: true,
      message: 'Scraping completed successfully.',
      log: output,
    });
  });

  proc.on('error', (err) => {
    res.status(500).json({ success: false, error: err.message });
  });
});

module.exports = router;
