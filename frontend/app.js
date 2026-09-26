/**
 * Tràuna AI — app.js
 * Frontend logic for the Gaming Thumbnail Generator
 */

'use strict';

// ── Configuration ──────────────────────────────────────────────
const API_BASE = 'http://localhost:3001';

const GAMES = [
  { id: 'minecraft',      label: 'Minecraft',      emoji: '⛏️' },
  { id: 'fortnite',       label: 'Fortnite',        emoji: '🔫' },
  { id: 'gta',            label: 'GTA',             emoji: '🚗' },
  { id: 'valorant',       label: 'Valorant',        emoji: '🎯' },
  { id: 'pubg',           label: 'PUBG',            emoji: '🪖' },
  { id: 'call_of_duty',   label: 'Call of Duty',    emoji: '💣' },
  { id: 'apex_legends',   label: 'Apex Legends',    emoji: '🏆' },
  { id: 'roblox',         label: 'Roblox',          emoji: '🧱' },
  { id: 'among_us',       label: 'Among Us',        emoji: '👻' },
  { id: 'horror',         label: 'Horror Games',    emoji: '😱' },
  { id: 'minecraft_hardcore', label: 'MC Hardcore', emoji: '💀' },
  { id: 'other',          label: 'Other',           emoji: '🎮' },
];

// ── DOM References ──────────────────────────────────────────────
const form             = document.getElementById('generateForm');
const primaryInput     = document.getElementById('primaryImage');
const primaryPreview   = document.getElementById('primaryPreview');
const additionalInput  = document.getElementById('additionalImages');
const additionalPreviews = document.getElementById('additionalPreviews');
const screenshotInput  = document.getElementById('gameplayScreenshot');
const screenshotPreview = document.getElementById('screenshotPreview');
const gameGrid         = document.getElementById('gameGrid');
const selectedGameInput = document.getElementById('selectedGame');
const descTextarea     = document.getElementById('videoDescription');
const descCount        = document.getElementById('descCount');
const generateBtn      = document.getElementById('generateBtn');
const btnText          = generateBtn.querySelector('.btn-text');
const btnLoader        = document.getElementById('btnLoader');
const statusBar        = document.getElementById('statusBar');
const statusText       = document.getElementById('statusText');
const errorBar         = document.getElementById('errorBar');
const errorText        = document.getElementById('errorText');
const resultsSection   = document.getElementById('resultsSection');
const resultsGrid      = document.getElementById('resultsGrid');
const customPromptInput = document.getElementById('customPrompt');
const clearPromptBtn    = document.getElementById('clearPromptBtn');
const progressCard      = document.getElementById('progressCard');
const progressPhase     = document.getElementById('progressPhase');
const progressEtaBadge  = document.getElementById('progressEtaBadge');
const progressBarFill   = document.getElementById('progressBarFill');
const progressDetail    = document.getElementById('progressDetail');
const progressElapsed   = document.getElementById('progressElapsed');

// ── State ───────────────────────────────────────────────────────
let selectedGame = '';
let currentJobId = null;
let progressTimer = null;
let progressStartTime = 0;

// ── Preset Chips, Quality Mode & Clear Button ──────────────────
function initPromptBox() {
  document.querySelectorAll('.chip-preset').forEach(btn => {
    btn.addEventListener('click', () => {
      if (customPromptInput) {
        customPromptInput.value = btn.dataset.idea || btn.dataset.prompt || '';
        customPromptInput.focus();
      }
    });
  });

  const btnFast = document.getElementById('btnQualityFast');
  const btnUltra = document.getElementById('btnQualityUltra');
  const qualityModeInput = document.getElementById('qualityMode');

  if (btnFast && btnUltra && qualityModeInput) {
    btnFast.addEventListener('click', () => {
      btnFast.classList.add('active');
      btnUltra.classList.remove('active');
      qualityModeInput.value = 'fast';
    });
    btnUltra.addEventListener('click', () => {
      btnUltra.classList.add('active');
      btnFast.classList.remove('active');
      qualityModeInput.value = 'ultra';
    });
  }

  if (clearPromptBtn && customPromptInput) {
    clearPromptBtn.addEventListener('click', () => {
      customPromptInput.value = '';
      customPromptInput.focus();
    });
  }
}

// ── Game Chips ──────────────────────────────────────────────────
function buildGameChips() {
  GAMES.forEach(g => {
    const chip = document.createElement('div');
    chip.className = 'game-chip';
    chip.dataset.game = g.id;
    chip.innerHTML = `<span>${g.emoji}</span><span>${g.label}</span>`;
    chip.addEventListener('click', () => selectGame(g.id, chip));
    gameGrid.appendChild(chip);
  });
}

function selectGame(id, chipEl) {
  document.querySelectorAll('.game-chip').forEach(c => c.classList.remove('active'));
  chipEl.classList.add('active');
  selectedGame = id;
  selectedGameInput.value = id;
}

// ── Upload Previews ─────────────────────────────────────────────
function wireUploadZone(inputEl, zoneEl, previewEl, multi) {
  zoneEl.addEventListener('click', () => inputEl.click());
  zoneEl.addEventListener('dragover', e => { e.preventDefault(); zoneEl.classList.add('dragover'); });
  zoneEl.addEventListener('dragleave', () => zoneEl.classList.remove('dragover'));
  zoneEl.addEventListener('drop', e => {
    e.preventDefault();
    zoneEl.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (!files.length) return;
    if (multi) {
      showMultiPreviews(files, additionalPreviews);
    } else {
      showSinglePreview(files[0], previewEl);
      // Sync the file input
      const dt = new DataTransfer();
      dt.items.add(files[0]);
      inputEl.files = dt.files;
    }
  });

  inputEl.addEventListener('change', () => {
    if (!inputEl.files.length) return;
    if (multi) {
      showMultiPreviews(inputEl.files, previewEl);
    } else {
      showSinglePreview(inputEl.files[0], previewEl);
    }
  });
}

function showSinglePreview(file, imgEl) {
  if (!imgEl) return;
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  imgEl.classList.remove('hidden');
}

function showMultiPreviews(files, container) {
  container.innerHTML = '';
  Array.from(files).forEach(file => {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    container.appendChild(img);
  });
}

// ── Character Counter ───────────────────────────────────────────
descTextarea.addEventListener('input', () => {
  descCount.textContent = descTextarea.value.length;
});

// ── Form Submission ─────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const customPromptVal = customPromptInput ? customPromptInput.value.trim() : '';

  if (!customPromptVal && !selectedGame) {
    showError('Please enter a prompt above or select a game category below.');
    return;
  }

  hideError();
  setLoading(true);
  setStatus('Sending to AI... generating thumbnail with trained Gaming LoRA (~30-40s)');
  clearResults();

  const fd = new FormData(form);

  if (customPromptVal) {
    fd.set('prompt', customPromptVal);
  }
  if (selectedGame) {
    fd.set('game', selectedGame);
  }

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
      if (progressBarFill) progressBarFill.style.width = '100%';
      if (progressEtaBadge) progressEtaBadge.textContent = 'COMPLETE!';
      if (progressPhase) progressPhase.textContent = 'Thumbnail Generated!';
      if (progressDetail) progressDetail.textContent = 'Rendering preview in high definition...';

      setTimeout(() => {
        if (progressCard) progressCard.classList.add('hidden');
      }, 1500);

      displayResults(data.results);
    } else {
      throw new Error('No results returned from server.');
    }

  } catch (err) {
    if (progressCard) progressCard.classList.add('hidden');
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

// ── Results Display ─────────────────────────────────────────────
function displayResults(results) {
  resultsGrid.innerHTML = '';
  resultsSection.classList.remove('hidden');

  results.forEach((r, i) => {
    const card = document.createElement('div');
    card.className = 'result-card';

    const imgUrl = `${API_BASE}${r.url}`;

    card.innerHTML = `
      <img src="${imgUrl}" alt="Thumbnail ${i + 1}" loading="lazy" />
      <div class="result-actions">
        <button class="btn-secondary" data-index="${i}">&#8635; Regenerate</button>
        <a href="${imgUrl}" download="trauna-thumbnail-${i + 1}.png" class="btn-download">&#8595; Download PNG</a>
      </div>
    `;

    // Regenerate (re-submits form with same data)
    card.querySelector('.btn-secondary').addEventListener('click', () => {
      form.dispatchEvent(new Event('submit', { cancelable: true }));
    });

    resultsGrid.appendChild(card);
  });

  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── UI Helpers ──────────────────────────────────────────────────
function setLoading(on) {
  generateBtn.disabled = on;
  btnText.classList.toggle('hidden', on);
  btnLoader.classList.toggle('hidden', !on);

  if (on) {
    if (progressCard) {
      progressCard.classList.remove('hidden');
      progressCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    if (progressBarFill) progressBarFill.style.width = '2%';
    if (progressEtaBadge) progressEtaBadge.textContent = 'ETA: ~40s';
    if (progressPhase) progressPhase.textContent = 'Initializing AI Pipeline...';
    if (progressDetail) progressDetail.textContent = 'Loading SDXL Turbo & Tràuna Gaming LoRA weights...';
    if (progressElapsed) progressElapsed.textContent = 'Elapsed: 0s';

    progressStartTime = Date.now();
    const mode = document.getElementById('qualityMode')?.value || 'ultra';
    const isUltra = mode === 'ultra';
    const EXPECTED_DURATION = isUltra ? 62 : 42;
    const totalSteps = isUltra ? 6 : 4;
    const initialEta = isUltra ? 'ETA: ~60s' : 'ETA: ~40s';

    if (progressEtaBadge) progressEtaBadge.textContent = initialEta;

    if (progressTimer) clearInterval(progressTimer);
    progressTimer = setInterval(() => {
      const elapsed = Math.max(0, (Date.now() - progressStartTime) / 1000);
      const remaining = Math.max(1, Math.round(EXPECTED_DURATION - elapsed));

      if (progressElapsed) {
        progressElapsed.textContent = `Elapsed: ${Math.round(elapsed)}s`;
      }
      if (progressEtaBadge) {
        progressEtaBadge.textContent = `ETA: ~${remaining}s`;
      }

      // Smooth percentage calculation through pipeline stages
      let pct = 0;
      if (elapsed < 6) {
        pct = (elapsed / 6) * 15;
        if (progressPhase) progressPhase.textContent = 'Initializing AI Pipeline...';
        if (progressDetail) progressDetail.textContent = 'Loading SDXL Turbo & Tràuna Rank 24 LoRA weights into VRAM...';
      } else if (elapsed < (EXPECTED_DURATION - 7)) {
        const denoiseElapsed = elapsed - 6;
        const denoiseTotal = EXPECTED_DURATION - 13;
        pct = 15 + (denoiseElapsed / denoiseTotal) * 70; // 15% to 85%
        const stepNum = Math.min(totalSteps, Math.floor((denoiseElapsed / (denoiseTotal / totalSteps))) + 1);
        if (progressPhase) progressPhase.textContent = `Generating Thumbnail (Step ${stepNum}/${totalSteps})...`;
        if (progressDetail) progressDetail.textContent = isUltra 
          ? `Denoising in Ultra Detail Mode (6 steps) on RTX 3050 6GB GPU...`
          : `Denoising in Fast Mode (4 steps) on RTX 3050 6GB GPU...`;
      } else if (elapsed < (EXPECTED_DURATION - 2)) {
        const decodeElapsed = elapsed - (EXPECTED_DURATION - 7);
        pct = 85 + (decodeElapsed / 5) * 10; // 85% to 95%
        if (progressPhase) progressPhase.textContent = 'Decoding HD Image...';
        if (progressDetail) progressDetail.textContent = 'Decoding VAE latents into 1280×720 (16:9) frame...';
      } else {
        pct = Math.min(98, 95 + (elapsed - (EXPECTED_DURATION - 2)) * 0.5);
        if (progressPhase) progressPhase.textContent = 'Finalizing Composition...';
        if (progressDetail) progressDetail.textContent = 'Post-processing, contrast enhancement & text overlay...';
      }

      if (progressBarFill) {
        progressBarFill.style.width = `${Math.round(pct)}%`;
      }
    }, 400);

  } else {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
  }
}

function setStatus(msg) {
  statusText.textContent = msg;
}

function showError(msg) {
  errorText.textContent = msg;
  errorBar.classList.remove('hidden');
  setTimeout(() => errorBar.classList.add('hidden'), 8000);
}

function hideError() {
  errorBar.classList.add('hidden');
}

function clearResults() {
  resultsGrid.innerHTML = '';
  resultsSection.classList.add('hidden');
}

// ── Init ─────────────────────────────────────────────────────────
function init() {
  initPromptBox();
  buildGameChips();

  wireUploadZone(
    primaryInput,
    document.getElementById('primaryDropZone'),
    primaryPreview,
    false
  );

  wireUploadZone(
    additionalInput,
    document.getElementById('additionalDropZone'),
    additionalPreviews,
    true
  );

  wireUploadZone(
    screenshotInput,
    document.getElementById('screenshotDropZone'),
    screenshotPreview,
    false
  );
}

// ── Tabs Switching ───────────────────────────────────────────────
const tabBtnGenerator = document.getElementById('tabBtnGenerator');
const tabBtnCrawler   = document.getElementById('tabBtnCrawler');
const generatorView   = document.getElementById('generatorView');
const crawlerView     = document.getElementById('crawlerView');

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

// ── Crawler Form Handling ─────────────────────────────────────────
const crawlerForm         = document.getElementById('crawlerForm');
const crawlBtn            = document.getElementById('crawlBtn');
const crawlLoader         = document.getElementById('crawlLoader');
const crawlerProgressCard = document.getElementById('crawlerProgressCard');
const crawlerLog          = document.getElementById('crawlerLog');

crawlerForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const target   = document.getElementById('crawlTarget').value.trim();
  const creator  = document.getElementById('crawlCreator').value.trim();
  const game     = document.getElementById('crawlGame').value.trim();
  const limit    = parseInt(document.getElementById('crawlLimit').value, 10) || 30;

  crawlBtn.disabled = true;
  crawlLoader.classList.remove('hidden');
  crawlerProgressCard.classList.remove('hidden');
  crawlerLog.textContent = `🚀 Launching crawler for: ${creator || target}...\nLimit: ${limit} videos\nFetching metadata & max-res thumbnails...\n`;

  try {
    const isUrl = target.startsWith('http://') || target.startsWith('https://');
    const isPlaylist = isUrl && target.includes('playlist');
    const isChannel = isUrl && (target.includes('/videos') || target.includes('@'));

    const res = await fetch(`${API_BASE}/api/crawl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        channel: isChannel ? target : undefined,
        playlist: isPlaylist ? target : undefined,
        search: !isUrl ? target : undefined,
        creator: creator || undefined,
        game: game || undefined,
        limit,
      }),
    });

    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || 'Failed to complete scrape.');
    }

    crawlerLog.textContent = data.log || 'Scraping completed successfully!';
  } catch (err) {
    crawlerLog.textContent += `\n❌ Error: ${err.message}`;
  } finally {
    crawlBtn.disabled = false;
    crawlLoader.classList.add('hidden');
  }
});

init();
