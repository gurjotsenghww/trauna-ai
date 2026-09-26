/**
 * Tràuna AI — server.js
 * Express backend for the Gaming Thumbnail Generator
 */

'use strict';

const express     = require('express');
const cors        = require('cors');
const path        = require('path');
const fs          = require('fs');
const generateRoute = require('./routes/generate');
const crawlRoute    = require('./routes/crawl');

const app  = express();
const PORT = process.env.PORT || 3001;

// ── Middleware ──────────────────────────────────────────────────
app.use(cors({ origin: '*' }));          // Allow all origins in dev
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));

// ── Ensure required directories exist ──────────────────────────
const dirs = [
  path.join(__dirname, 'uploads'),
  path.join(__dirname, 'outputs'),
];
dirs.forEach(d => { if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true }); });

// ── Static file serving ─────────────────────────────────────────
// Serve the AI outputs so the frontend can fetch generated images
app.use('/outputs', express.static(path.join(__dirname, 'outputs')));
// Serve the frontend
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// ── Routes ──────────────────────────────────────────────────────
app.use('/api/generate', generateRoute);
app.use('/api/crawl', crawlRoute);

// ── Health check ────────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ── 404 ─────────────────────────────────────────────────────────
app.use((_req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// ── Error handler ───────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  console.error('[Server Error]', err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

// ── Start ───────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n  Tràuna AI backend running at http://localhost:${PORT}`);
  console.log(`  Frontend:  http://localhost:${PORT}`);
  console.log(`  API:       http://localhost:${PORT}/api/generate`);
  console.log(`  Health:    http://localhost:${PORT}/api/health\n`);
});

module.exports = app;
