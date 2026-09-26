/**
 * Tràuna AI — app.js
 * Next-Gen Gaming & Vlog Thumbnail Studio
 */

'use strict';

// ── Configuration ──────────────────────────────────────────────
const API_BASE = 'http://localhost:3001';

const GAMES = [
  { id: 'minecraft',          label: 'Minecraft',          emoji: '⛏️' },
  { id: 'minecraft_hardcore', label: 'MC Hardcore',        emoji: '💀' },
  { id: 'gta',                label: 'GTA / Heist',        emoji: '🚗' },
  { id: 'vlogs',              label: 'Vlogs & IRL',        emoji: '📹' },
  { id: 'horror',             label: 'Horror / Jumpscare', emoji: '😱' },
  { id: 'valorant',           label: 'Valorant',           emoji: '🎯' },
  { id: 'fortnite',           label: 'Fortnite',           emoji: '🔫' },
  { id: 'pubg',               label: 'BGMI / PUBG',        emoji: '🪖' },
  { id: 'call_of_duty',       label: 'Call of Duty',       emoji: '💣' },
  { id: 'apex_legends',       label: 'Apex Legends',       emoji: '🏆' },
  { id: 'roblox',             label: 'Roblox',             emoji: '🧱' },
  { id: 'challenges',         label: 'MrBeast Challenges', emoji: '🔥' },
  { id: 'other',              label: 'Variety Gaming',     emoji: '🎮' },
];

// ── DOM References ──────────────────────────────────────────────
const form               = document.getElementById('generateForm');
const primaryInput       = document.getElementById('primaryImage');
const primaryPreview     = document.getElementById('primaryPreview');
const screenshotInput    = document.getElementById('gameplayScreenshot');
const screenshotPreview  = document.getElementById('screenshotPreview');
const gameGrid           = document.getElementById('gameGrid');
const selectedGameInput  = document.getElementById('selectedGame');
const descTextarea       = document.getElementById('videoDescription');
const descCount          = document.getElementById('descCount');
const generateBtn        = document.getElementById('generateBtn');
const btnText            = generateBtn.querySelector('.btn-text');
const btnLoader          = document.getElementById('btnLoader');
const statusBar          = document.getElementById('statusBar');
const statusText         = document.getElementById('statusText');
const errorBar           = document.getElementById('errorBar');
const errorText          = document.getElementById('errorText');
const resultsSection     = document.getElementById('resultsSection');
const resultsGrid        = document.getElementById('resultsGrid');
const customPromptInput  = document.getElementById('customPrompt');
const clearPromptBtn     = document.getElementById('clearPromptBtn');
const thumbnailTextInput = document.getElementById('thumbnailText');
const textColorInput     = document.getElementById('textColor');
const textPosInput       = document.getElementById('textPosition');

// Progress & ETA
const progressCard       = document.getElementById('progressCard');
const progressPhase      = document.getElementById('progressPhase');
const progressEtaBadge   = document.getElementById('progressEtaBadge');
const progressBarFill    = document.getElementById('progressBarFill');
const progressDetail     = document.getElementById('progressDetail');
const progressElapsed    = document.getElementById('progressElapsed');

// View Toggle & Lightbox
const btnViewClean       = document.getElementById('btnViewClean');
const btnViewFeed        = document.getElementById('btnViewFeed');
const lightboxModal      = document.getElementById('lightboxModal');
const lightboxImg        = document.getElementById('lightboxImg');
const lightboxCloseBtn   = document.getElementById('lightboxCloseBtn');
const lightboxBackdrop   = document.getElementById('lightboxBackdrop');
const lightboxDownload   = document.getElementById('lightboxDownload');

// Crawler Tab
const tabBtnGenerator    = document.getElementById('tabBtnGenerator');
const tabBtnCrawler      = document.getElementById('tabBtnCrawler');
const generatorView      = document.getElementById('generatorView');
const crawlerView        = document.getElementById('crawlerView');
const crawlerForm        = document.getElementById('crawlerForm');
const crawlerLog         = document.getElementById('crawlerLog');

// ── State ───────────────────────────────────────────────────────
let selectedGame       = '';
let activeViewMode     = 'clean'; // 'clean' | 'feed'
let lastGeneratedImage = null;
let lastPromptUsed     = '';
let progressInterval   = null;
let progressStartTime  = 0;

// ── Navigation Tabs ─────────────────────────────────────────────
function initTabs() {
  if (tabBtnGenerator && tabBtnCrawler) {
    tabBtnGenerator.addEventListener('click', () => {
      tabBtnGenerator.classList.add('active');
      tabBtnCrawler.classList.remove('active');
      generatorView.classList.remove('hidden');
      crawlerView.classList.add('hidden');
    });

    tabBtnCrawler.addEventListener('click', () => {
      tabBtnCrawler.classList.add('active');
      tabBtnGenerator.classList.remove('active');
      crawlerView.classList.remove('hidden');
      generatorView.classList.add('hidden');
    });
  }
}

// ── Prompt Box, Presets & Quality Mode ───────────────────────────
function initPromptControls() {
  // Preset chips
  document.querySelectorAll('.chip-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      if (customPromptInput) {
        customPromptInput.value = btn.dataset.idea || '';
        customPromptInput.focus();
      }
    });
  });

  // Clear button
  if (clearPromptBtn && customPromptInput) {
    clearPromptBtn.addEventListener('click', () => {
      customPromptInput.value = '';
      customPromptInput.focus();
    });
  }

  // Quality buttons
  const btnFast = document.getElementById('btnQualityFast');
  const btnStandard = document.getElementById('btnQualityStandard');
  const btnUltra = document.getElementById('btnQualityUltra');
  const qualityModeInput = document.getElementById('qualityMode');

  const setQuality = (mode, btnActive) => {
    [btnFast, btnStandard, btnUltra].forEach(b => b?.classList.remove('active'));
    btnActive?.classList.add('active');
    if (qualityModeInput) qualityModeInput.value = mode;
  };

  btnFast?.addEventListener('click', () => setQuality('fast', btnFast));
  btnStandard?.addEventListener('click', () => setQuality('standard', btnStandard));
  btnUltra?.addEventListener('click', () => setQuality('ultra', btnUltra));

  // Text color chips
  document.querySelectorAll('.color-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.color-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      if (textColorInput) textColorInput.value = chip.dataset.color || '#FFFFFF';
    });
  });

  // Text position toggles
  document.querySelectorAll('.btn-pos').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-pos').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (textPosInput) textPosInput.value = btn.dataset.pos || 'bottom';
    });
  });

  // View mode switcher (Clean vs YouTube Feed)
  btnViewClean?.addEventListener('click', () => {
    activeViewMode = 'clean';
    btnViewClean.classList.add('active');
    btnViewFeed.classList.remove('active');
    if (lastGeneratedImage) renderCurrentView(lastGeneratedImage);
  });

  btnViewFeed?.addEventListener('click', () => {
    activeViewMode = 'feed';
    btnViewFeed.classList.add('active');
    btnViewClean.classList.remove('active');
    if (lastGeneratedImage) renderCurrentView(lastGeneratedImage);
  });
}

// ── Game Chips ──────────────────────────────────────────────────
function buildGameChips() {
  gameGrid.innerHTML = '';
  GAMES.forEach(g => {
    const chip = document.createElement('div');
    chip.className = 'game-chip';
    chip.dataset.game = g.id;
    chip.innerHTML = `<span>${g.emoji}</span><span>${g.label}</span>`;
    chip.addEventListener('click', () => selectGame(g.id, chip));
    gameGrid.appendChild(chip));
  });
}

function selectGame(id, chipEl) {
  document.querySelectorAll('.game-chip').forEach(c => c.classList.remove('active'));
  chipEl.classList.add('active');
  selectedGame = id;
  selectedGameInput.value = id;
}

// ── Upload Previews ─────────────────────────────────────────────
function wireUploadZone(inputEl, zoneEl, previewEl) {
  zoneEl.addEventListener('click', () => inputEl.click());
  zoneEl.addEventListener('dragover', e => { e.preventDefault(); zoneEl.classList.add('dragover'); });
  zoneEl.addEventListener('dragleave', () => zoneEl.classList.remove('dragover'));
  zoneEl.addEventListener('drop', e => {
    e.preventDefault();
    zoneEl.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (!files.length) return;
    showSinglePreview(files[0], previewEl);
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    inputEl.files = dt.files;
  });

  inputEl.addEventListener('change', () => {
    if (!inputEl.files.length) return;
    showSinglePreview(inputEl.files[0], previewEl);
  });
}

function showSinglePreview(file, imgEl) {
  if (!imgEl) return;
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  imgEl.classList.remove('hidden');
}

// ── Character Counter ───────────────────────────────────────────
if (descTextarea && descCount) {
  descTextarea.addEventListener('input', () => {
    descCount.textContent = descTextarea.value.length;
  });
}

// ── Realistic Progress Simulator ─────────────────────────────────
function startProgressAnimation() {
  if (!progressCard) return;
  progressCard.classList.remove('hidden');
  progressBarFill.style.width = '5%';
  progressStartTime = Date.now();
  progressElapsed.textContent = 'Elapsed: 0s';
  progressPhase.textContent = 'Preparing SDXL Turbo & LoRA Weights...';
  progressEtaBadge.textContent = 'ETA: ~35s';
  progressDetail.textContent = 'Decoupling 16:9 aspect ratio buckets (1152x648)...';

  clearInterval(progressInterval);
  progressInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - progressStartTime) / 1000);
    progressElapsed.textContent = `Elapsed: ${elapsed}s`;

    if (elapsed <= 6) {
      progressBarFill.style.width = `${10 + elapsed * 3}%`;
      progressDetail.textContent = 'Applying anti-hybrid negative matrix & three-zone spatial grammar...';
    } else if (elapsed <= 18) {
      progressBarFill.style.width = `${30 + (elapsed - 6) * 3}%`;
      progressPhase.textContent = 'Synthesizing Latents (5,000-Step LoRA)...';
      progressDetail.textContent = 'Denoising ADD latents on local RTX 3050 GPU (VRAM safe: ~3.01 GB)...';
      const remaining = Math.max(1, 35 - elapsed);
      progressEtaBadge.textContent = `ETA: ~${remaining}s`;
    } else if (elapsed <= 30) {
      progressBarFill.style.width = `${66 + (elapsed - 18) * 2}%`;
      progressPhase.textContent = 'VAE Decoding & Lanczos3 Downsampling...';
      progressDetail.textContent = 'Exporting strictly 1280x720 master PNG...';
    } else {
      progressBarFill.style.width = '94%';
      progressPhase.textContent = 'Compositing SVG Typography...';
      progressDetail.textContent = 'Finalizing YouTube thumbnail export...';
    }
  }, 1000);
}

function stopProgressAnimation(success = true) {
  clearInterval(progressInterval);
  if (success && progressBarFill) {
    progressBarFill.style.width = '100%';
    progressEtaBadge.textContent = 'COMPLETE!';
    progressPhase.textContent = 'Thumbnail Generated Successfully!';
    progressDetail.textContent = 'Render complete (1280x720 Master HD).';
    setTimeout(() => {
      progressCard?.classList.add('hidden');
    }, 1200);
  } else {
    progressCard?.classList.add('hidden');
  }
}

// ── Form Submission ─────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const customPromptVal = customPromptInput ? customPromptInput.value.trim() : '';

  if (!customPromptVal && !selectedGame && !descTextarea?.value.trim()) {
    showError('Please describe your thumbnail idea or select a category.');
    return;
  }

  hideError();
  setLoading(true);
  startProgressAnimation();
  clearResults();

  const fd = new FormData(form);
  if (customPromptVal) fd.set('prompt', customPromptVal);
  if (selectedGame) fd.set('game', selectedGame);

  try {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      body: fd,
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(errBody.error || `Server error ${res.status}`);
    }

    const data = await res.json();

    if (data.results && data.results.length) {
      lastGeneratedImage = data.results[0];
      lastPromptUsed = customPromptVal || descTextarea?.value || 'Epic YouTube Gaming Thumbnail';
      stopProgressAnimation(true);
      renderCurrentView(lastGeneratedImage);
    } else {
      throw new Error('No results returned from server.');
    }

  } catch (err) {
    stopProgressAnimation(false);
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

// ── Results Rendering (Clean Master vs YouTube Feed Simulator) ───
function renderCurrentView(resultItem) {
  resultsGrid.innerHTML = '';
  resultsSection.classList.remove('hidden');

  const imgUrl = `${API_BASE}${resultItem.url}`;
  const filename = resultItem.filename || 'trauna-thumbnail.png';

  if (activeViewMode === 'feed') {
    // YouTube Feed Simulator Card
    const card = document.createElement('div');
    card.className = 'yt-feed-simulator';

    const videoTitle = (thumbnailTextInput?.value.trim())
      ? `${thumbnailTextInput.value.trim()} — ${lastPromptUsed.substring(0, 50)}`
      : (lastPromptUsed.length > 60 ? lastPromptUsed.substring(0, 60) + '...' : lastPromptUsed);

    card.innerHTML = `
      <div class="yt-card-thumb-wrap" style="cursor: pointer;">
        <img src="${imgUrl}" alt="YouTube Thumbnail Preview" />
        <span class="yt-badge-duration">14:28</span>
      </div>
      <div class="yt-card-meta">
        <div class="yt-avatar">T</div>
        <div class="yt-details">
          <div class="yt-title">${escapeHtml(videoTitle)}</div>
          <div class="yt-channel-row">
            <span class="yt-channel-name">Tràuna Gaming</span>
            <span class="yt-badge-verified" title="Verified">&#10004;</span>
          </div>
          <div class="yt-stats">1.8M views &bull; 1 day ago</div>
        </div>
      </div>
    `;

    // Click thumbnail to inspect in lightbox
    card.querySelector('.yt-card-thumb-wrap').addEventListener('click', () => {
      openLightbox(imgUrl);
    });

    resultsGrid.appendChild(card);

  } else {
    // Clean Master View Card
    const card = document.createElement('div');
    card.className = 'result-card';

    card.innerHTML = `
      <div style="position: relative; overflow: hidden; cursor: pointer;" class="thumb-click-wrap">
        <img src="${imgUrl}" alt="Generated 1280x720 Thumbnail" loading="lazy" />
        <span style="position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.75); color: #00e5ff; font-family: var(--font-head); font-size: 0.72rem; padding: 4px 8px; border-radius: 6px; border: 1px solid rgba(0,229,255,0.4);">1280 &times; 720</span>
      </div>
      <div class="result-actions">
        <button class="btn-inspect" type="button" title="Inspect full resolution">🔍 Inspect</button>
        <button class="btn-clipboard" type="button" title="Copy image to clipboard">📋 Copy</button>
        <a href="${imgUrl}" download="${filename}" class="btn-download">📥 Download PNG</a>
      </div>
    `;

    // Lightbox triggers
    card.querySelector('.thumb-click-wrap').addEventListener('click', () => openLightbox(imgUrl));
    card.querySelector('.btn-inspect').addEventListener('click', () => openLightbox(imgUrl));

    // Copy to clipboard
    const copyBtn = card.querySelector('.btn-clipboard');
    copyBtn.addEventListener('click', async () => {
      try {
        const response = await fetch(imgUrl);
        const blob = await response.blob();
        await navigator.clipboard.write([
          new ClipboardItem({ [blob.type]: blob })
        ]);
        copyBtn.textContent = '✅ Copied!';
        setTimeout(() => { copyBtn.textContent = '📋 Copy'; }, 2000);
      } catch (err) {
        showError('Could not copy image automatically. Use Download PNG.');
      }
    });

    resultsGrid.appendChild(card);
  }

  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Fullscreen Lightbox Modal ────────────────────────────────────
function openLightbox(url) {
  if (!lightboxModal || !lightboxImg) return;
  lightboxImg.src = url;
  if (lightboxDownload) lightboxDownload.href = url;
  lightboxModal.classList.remove('hidden');
}

function closeLightbox() {
  lightboxModal?.classList.add('hidden');
  if (lightboxImg) lightboxImg.src = '';
}

lightboxCloseBtn?.addEventListener('click', closeLightbox);
lightboxBackdrop?.addEventListener('click', closeLightbox);
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !lightboxModal.classList.contains('hidden')) {
    closeLightbox();
  }
});

// ── Helper Utilities ─────────────────────────────────────────────
function setLoading(on) {
  generateBtn.disabled = on;
  btnText.classList.toggle('hidden', on);
  btnLoader.classList.toggle('hidden', !on);
  statusBar.classList.toggle('hidden', !on);
}

function setStatus(msg) {
  if (statusText) statusText.textContent = msg;
}

function showError(msg) {
  if (errorText && errorBar) {
    errorText.textContent = msg;
    errorBar.classList.remove('hidden');
    setTimeout(() => errorBar.classList.add('hidden'), 8000);
  }
}

function hideError() {
  errorBar?.classList.add('hidden');
}

function clearResults() {
  resultsGrid.innerHTML = '';
  resultsSection?.classList.add('hidden');
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, m => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[m]);
}

// ── Init ─────────────────────────────────────────────────────────
function init() {
  initTabs();
  initPromptControls();
  buildGameChips();

  wireUploadZone(
    primaryInput,
    document.getElementById('primaryDropZone'),
    primaryPreview
  );

  wireUploadZone(
    screenshotInput,
    document.getElementById('screenshotDropZone'),
    screenshotPreview
  );
}

init();
