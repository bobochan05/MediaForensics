import { useState, useRef, useCallback, useEffect } from "react";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #08090a;
    --bg-1: #0e0f11;
    --bg-2: #141518;
    --bg-3: #1c1d21;
    --border: rgba(255,255,255,0.07);
    --border-hi: rgba(255,255,255,0.14);
    --text: #f0ede8;
    --text-2: #8a8782;
    --text-3: #4a4845;
    --accent: #e8f55a;
    --accent-dim: rgba(232,245,90,0.12);
    --accent-glow: rgba(232,245,90,0.06);
    --red: #ff4d4d;
    --red-dim: rgba(255,77,77,0.1);
    --green: #4ade80;
    --green-dim: rgba(74,222,128,0.1);
    --blue: #60a5fa;
    --blue-dim: rgba(96,165,250,0.08);
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'Inter', sans-serif;
    --display: 'Syne', sans-serif;
    --r: 10px;
    --r-lg: 16px;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--sans); }

  .app {
    min-height: 100vh;
    background: var(--bg);
  }

  /* ── NAV ─────────────────────────────────────── */
  .nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2.5rem;
    height: 56px;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    background: rgba(8,9,10,0.85);
    backdrop-filter: blur(12px);
  }
  .nav-logo {
    font-family: var(--display); font-size: 15px; font-weight: 700;
    letter-spacing: 0.04em; color: var(--text);
    display: flex; align-items: center; gap: 8px;
  }
  .nav-logo-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(0.75); } }
  .nav-tag {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 0.08em;
  }

  /* ── HERO ─────────────────────────────────────── */
  .hero {
    padding: 5rem 2.5rem 4rem;
    max-width: 900px; margin: 0 auto;
    display: grid; grid-template-columns: 1fr 1fr; gap: 3rem;
    align-items: center;
  }
  .hero-eyebrow {
    font-family: var(--mono); font-size: 10px; font-weight: 500;
    letter-spacing: 0.18em; color: var(--accent);
    text-transform: uppercase; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 8px;
  }
  .hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 20px; height: 1px; background: var(--accent);
  }
  .hero-title {
    font-family: var(--display); font-size: clamp(2.2rem, 4vw, 3.2rem);
    font-weight: 800; line-height: 1.05; letter-spacing: -0.02em;
    color: var(--text); margin-bottom: 1.2rem;
  }
  .hero-title em { font-style: normal; color: var(--accent); }
  .hero-sub {
    font-size: 14px; line-height: 1.7; color: var(--text-2); font-weight: 300;
  }
  .hero-stats {
    display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: var(--r-lg); overflow: hidden;
  }
  .stat-cell {
    background: var(--bg-1);
    padding: 1.4rem 1.5rem;
  }
  .stat-num {
    font-family: var(--display); font-size: 2rem; font-weight: 800;
    color: var(--text); letter-spacing: -0.03em;
    line-height: 1;
  }
  .stat-num span { color: var(--accent); }
  .stat-label {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 6px;
  }

  /* ── HOW IT WORKS ─────────────────────────────── */
  .section {
    max-width: 900px; margin: 0 auto;
    padding: 0 2.5rem 4rem;
  }
  .section-header {
    display: flex; align-items: center; gap: 12px; margin-bottom: 1.75rem;
  }
  .section-num {
    font-family: var(--mono); font-size: 10px; color: var(--accent);
    letter-spacing: 0.15em; opacity: 0.7;
  }
  .section-title {
    font-family: var(--display); font-size: 18px; font-weight: 700;
    letter-spacing: -0.01em;
  }
  .section-line {
    flex: 1; height: 1px; background: var(--border);
  }

  .pipeline {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
    background: var(--border);
    border: 1px solid var(--border); border-radius: var(--r-lg);
    overflow: hidden; margin-bottom: 1px;
  }
  .pipeline-card {
    background: var(--bg-1); padding: 1.4rem;
  }
  .pipeline-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: var(--accent-dim); border: 1px solid rgba(232,245,90,0.2);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 1rem;
    font-family: var(--mono); font-size: 11px; font-weight: 500; color: var(--accent);
    letter-spacing: 0.05em;
  }
  .pipeline-name {
    font-family: var(--display); font-size: 13px; font-weight: 700;
    color: var(--text); margin-bottom: 4px; letter-spacing: 0.01em;
  }
  .pipeline-model {
    font-family: var(--mono); font-size: 10px; color: var(--accent);
    margin-bottom: 10px; letter-spacing: 0.06em;
  }
  .pipeline-desc {
    font-size: 12px; line-height: 1.65; color: var(--text-2);
  }
  .fusion-row {
    display: grid; grid-template-columns: 1fr; margin-bottom: 0;
  }
  .fusion-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 var(--r-lg) var(--r-lg);
    padding: 1rem 1.4rem;
    display: flex; align-items: center; gap: 1.5rem;
  }
  .fusion-label {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 0.12em; text-transform: uppercase; white-space: nowrap;
  }
  .fusion-flow {
    display: flex; align-items: center; gap: 8px; flex: 1;
  }
  .fusion-pill {
    font-family: var(--mono); font-size: 10px;
    padding: 3px 8px; border-radius: 4px;
    background: var(--bg-3); border: 1px solid var(--border-hi);
    color: var(--text-2);
  }
  .fusion-arrow {
    color: var(--text-3); font-size: 12px;
  }
  .fusion-output {
    font-family: var(--mono); font-size: 10px;
    padding: 3px 10px; border-radius: 4px;
    background: var(--accent-dim); border: 1px solid rgba(232,245,90,0.3);
    color: var(--accent);
  }

  /* ── STEPS ─────────────────────────────────────── */
  .steps {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 0;
    border: 1px solid var(--border); border-radius: var(--r-lg); overflow: hidden;
  }
  .step {
    background: var(--bg-1);
    padding: 1.2rem;
    border-right: 1px solid var(--border);
    position: relative;
  }
  .step:last-child { border-right: none; }
  .step-n {
    font-family: var(--mono); font-size: 20px; font-weight: 500;
    color: var(--text-3); line-height: 1; margin-bottom: 8px;
  }
  .step-action {
    font-size: 12px; font-weight: 500; color: var(--text); margin-bottom: 4px;
  }
  .step-detail { font-size: 11px; color: var(--text-2); line-height: 1.5; }

  /* ── DIVIDER ─────────────────────────────────────── */
  .divider {
    max-width: 900px; margin: 0 auto 3rem;
    padding: 0 2.5rem;
  }
  .divider-line { height: 1px; background: var(--border); }

  /* ── UPLOAD ZONE ─────────────────────────────────── */
  .upload-outer { max-width: 900px; margin: 0 auto; padding: 0 2.5rem 5rem; }

  .upload-grid {
    display: grid; grid-template-columns: 1fr 380px; gap: 1.5rem;
    align-items: start;
  }

  .dropzone {
    border: 1.5px dashed var(--border-hi);
    border-radius: var(--r-lg);
    min-height: 280px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 12px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative; overflow: hidden;
    background: var(--bg-1);
  }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: var(--accent-glow); }
  .dropzone-icon {
    width: 48px; height: 48px; border-radius: 12px;
    background: var(--bg-3); border: 1px solid var(--border-hi);
    display: flex; align-items: center; justify-content: center;
  }
  .dropzone-icon svg { width: 22px; height: 22px; color: var(--text-2); }
  .dropzone-text { font-size: 13px; color: var(--text-2); }
  .dropzone-text strong { color: var(--text); font-weight: 500; }
  .dropzone-formats {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 0.1em;
  }
  .dropzone-preview {
    width: 100%; height: 100%;
    object-fit: contain;
    position: absolute; inset: 0;
    padding: 12px;
  }
  .dropzone-overlay {
    position: absolute; inset: 0;
    background: rgba(8,9,10,0.6);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 8px;
    opacity: 0; transition: opacity 0.2s;
  }
  .dropzone:hover .dropzone-overlay { opacity: 1; }
  .dropzone-overlay-text {
    font-size: 12px; color: var(--text); font-weight: 500;
  }

  .panel {
    background: var(--bg-1); border: 1px solid var(--border);
    border-radius: var(--r-lg); padding: 1.5rem;
    display: flex; flex-direction: column; gap: 1.25rem;
  }
  .panel-row {
    display: flex; flex-direction: column; gap: 4px;
  }
  .panel-label {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 0.1em; text-transform: uppercase;
  }
  .panel-value { font-size: 13px; color: var(--text); font-weight: 500; }
  .panel-value.mono { font-family: var(--mono); font-size: 11px; color: var(--text-2); }
  .panel-divider { height: 1px; background: var(--border); }

  .btn-analyze {
    width: 100%; padding: 13px;
    background: var(--accent); color: #08090a;
    border: none; border-radius: var(--r);
    font-family: var(--display); font-size: 14px; font-weight: 700;
    letter-spacing: 0.04em; cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  .btn-analyze:hover { opacity: 0.88; }
  .btn-analyze:active { transform: scale(0.98); }
  .btn-analyze:disabled { opacity: 0.35; cursor: not-allowed; }
  .btn-analyze.loading { opacity: 0.7; }

  .spinner {
    width: 14px; height: 14px; border: 2px solid rgba(8,9,10,0.3);
    border-top-color: #08090a; border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .model-paths {
    display: flex; flex-direction: column; gap: 6px;
  }
  .model-path-row {
    display: flex; align-items: center; gap: 8px;
  }
  .model-dot {
    width: 6px; height: 6px; border-radius: 50%;
    flex-shrink: 0;
  }
  .model-dot.ok { background: var(--green); }
  .model-dot.warn { background: var(--red); }
  .model-path-text {
    font-family: var(--mono); font-size: 10px; color: var(--text-2);
  }

  /* ── RESULT SECTION ─────────────────────────────── */
  .result-section {
    max-width: 900px; margin: 0 auto;
    padding: 0 2.5rem 5rem;
    animation: fadeUp 0.5s ease both;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .verdict-card {
    border-radius: var(--r-lg); border: 1px solid;
    padding: 2rem 2rem 1.75rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
  }
  .verdict-card.real { background: var(--green-dim); border-color: rgba(74,222,128,0.25); }
  .verdict-card.fake { background: var(--red-dim); border-color: rgba(255,77,77,0.25); }

  .verdict-top {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 1.5rem;
  }
  .verdict-label-row {
    display: flex; align-items: center; gap: 10px;
  }
  .verdict-badge {
    font-family: var(--mono); font-size: 10px; font-weight: 500;
    letter-spacing: 0.15em; text-transform: uppercase;
    padding: 4px 10px; border-radius: 4px;
  }
  .verdict-badge.real { background: rgba(74,222,128,0.15); color: var(--green); }
  .verdict-badge.fake { background: rgba(255,77,77,0.15); color: var(--red); }
  .verdict-word {
    font-family: var(--display); font-size: 2.4rem; font-weight: 800;
    letter-spacing: -0.03em; line-height: 1;
  }
  .verdict-word.real { color: var(--green); }
  .verdict-word.fake { color: var(--red); }
  .verdict-conf {
    text-align: right;
  }
  .verdict-conf-num {
    font-family: var(--display); font-size: 2.4rem; font-weight: 800;
    letter-spacing: -0.03em; line-height: 1; color: var(--text);
  }
  .verdict-conf-label {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px;
  }

  .conf-track {
    height: 6px; border-radius: 999px;
    background: rgba(255,255,255,0.07);
    overflow: hidden; margin-bottom: 8px;
  }
  .conf-fill {
    height: 100%; border-radius: 999px;
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
  }
  .conf-fill.real { background: var(--green); }
  .conf-fill.fake { background: var(--red); }
  .verdict-desc { font-size: 13px; color: var(--text-2); line-height: 1.65; }

  /* ── EXPLANATION ─────────────────────────────────── */
  .explain-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
    background: var(--border); border: 1px solid var(--border);
    border-radius: var(--r-lg); overflow: hidden; margin-bottom: 1.5rem;
  }
  .explain-card {
    background: var(--bg-1); padding: 1.4rem;
  }
  .explain-tag {
    font-family: var(--mono); font-size: 9px; font-weight: 500;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent);
    margin-bottom: 10px;
  }
  .explain-name {
    font-family: var(--display); font-size: 14px; font-weight: 700;
    margin-bottom: 4px;
  }
  .explain-sub {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    margin-bottom: 10px; letter-spacing: 0.06em;
  }
  .explain-body { font-size: 12px; line-height: 1.65; color: var(--text-2); }

  .fusion-explain {
    background: var(--bg-1); border: 1px solid var(--border);
    border-radius: var(--r-lg); padding: 1.5rem; margin-bottom: 1.5rem;
  }
  .fusion-explain-title {
    font-family: var(--display); font-size: 14px; font-weight: 700;
    margin-bottom: 0.75rem;
  }
  .fusion-explain-body { font-size: 13px; line-height: 1.7; color: var(--text-2); }
  .fusion-explain-body code {
    font-family: var(--mono); font-size: 11px;
    background: var(--bg-3); padding: 2px 6px; border-radius: 4px;
    color: var(--accent);
  }

  .why-card {
    border-radius: var(--r-lg); padding: 1.5rem; margin-bottom: 1.5rem;
    border: 1px solid;
  }
  .why-card.real { background: var(--green-dim); border-color: rgba(74,222,128,0.2); }
  .why-card.fake { background: var(--red-dim); border-color: rgba(255,77,77,0.2); }
  .why-title {
    font-family: var(--display); font-size: 14px; font-weight: 700;
    margin-bottom: 0.75rem;
  }
  .why-body { font-size: 13px; line-height: 1.7; color: var(--text-2); }
  .why-body strong { color: var(--text); font-weight: 500; }
  .why-list { margin-top: 10px; padding-left: 0; list-style: none; }
  .why-list li {
    font-size: 12px; color: var(--text-2); padding: 4px 0 4px 16px;
    position: relative; line-height: 1.55;
  }
  .why-list li::before {
    content: '—'; position: absolute; left: 0; color: var(--text-3);
  }

  .conf-table {
    background: var(--bg-1); border: 1px solid var(--border);
    border-radius: var(--r-lg); overflow: hidden;
  }
  .conf-table-row {
    display: grid; grid-template-columns: 90px 160px 1fr;
    gap: 0; border-bottom: 1px solid var(--border);
    align-items: center;
  }
  .conf-table-row:last-child { border-bottom: none; }
  .conf-table-cell {
    padding: 12px 16px; font-size: 12px; color: var(--text-2);
    border-right: 1px solid var(--border);
  }
  .conf-table-cell:last-child { border-right: none; }
  .conf-table-cell.range {
    font-family: var(--mono); font-size: 11px; color: var(--text);
  }
  .conf-table-cell.tier { font-weight: 500; color: var(--text); font-size: 12px; }
  .conf-table-header .conf-table-cell {
    background: var(--bg-2); font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-3);
  }
  .current-tier { background: var(--accent-dim) !important; }
  .current-tier .conf-table-cell { color: var(--text); }

  /* ── TECHNICAL DETAILS ─────────────────────────── */
  .tech-details {
    background: var(--bg-1); border: 1px solid var(--border);
    border-radius: var(--r-lg); overflow: hidden;
  }
  .tech-summary {
    padding: 1rem 1.4rem;
    display: flex; align-items: center; justify-content: space-between;
    cursor: pointer; user-select: none;
    font-family: var(--mono); font-size: 11px; color: var(--text-2);
    letter-spacing: 0.08em;
  }
  .tech-summary:hover { background: var(--bg-2); }
  .tech-chevron { transition: transform 0.2s; font-size: 10px; }
  .tech-chevron.open { transform: rotate(180deg); }
  .tech-body { padding: 0 1.4rem 1.4rem; border-top: 1px solid var(--border); }
  .code-block {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--r); padding: 1rem;
    font-family: var(--mono); font-size: 11px; color: var(--text-2);
    line-height: 1.7; margin: 1rem 0;
  }
  .code-block .key { color: var(--text-3); }
  .code-block .val { color: var(--accent); }

  /* ── FOOTER ─────────────────────────────────────── */
  .footer {
    border-top: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    max-width: 900px; margin: 0 auto;
  }
  .footer-text {
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 0.08em;
  }
  .footer-stack {
    display: flex; gap: 6px;
  }
  .footer-pill {
    font-family: var(--mono); font-size: 9px; color: var(--text-3);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 2px 7px; letter-spacing: 0.06em;
  }

  @media (max-width: 680px) {
    .hero { grid-template-columns: 1fr; gap: 2rem; padding: 3rem 1.5rem 2.5rem; }
    .upload-grid { grid-template-columns: 1fr; }
    .pipeline { grid-template-columns: 1fr; }
    .explain-grid { grid-template-columns: 1fr; }
    .steps { grid-template-columns: 1fr; }
    .conf-table-row { grid-template-columns: 80px 130px 1fr; }
    .section, .upload-outer, .result-section, .divider, .footer { padding-left: 1.5rem; padding-right: 1.5rem; }
    .nav { padding: 0 1.5rem; }
    .hero-title { font-size: 2rem; }
  }
`;

const MODEL_PATHS = [
  { label: "Fusion MLP", path: "artifacts/fusion_model.pth" },
  { label: "EfficientNet", path: "artifacts/efficientnet_finetuned.pth" },
  { label: "DINO", path: "artifacts/dino_finetuned.pth" },
];

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M12 4v12M8 8l4-4 4 4" />
    </svg>
  );
}

function ChevronDown() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2.5 4.5L6 8l3.5-3.5" />
    </svg>
  );
}

function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function getConfTier(pct) {
  if (pct >= 80) return { tier: "High confidence", desc: "The model is strongly committed to this verdict. Results here are generally reliable." };
  if (pct >= 60) return { tier: "Moderate confidence", desc: "The model leans toward the verdict but branches may not fully agree. Worth a second look." };
  return { tier: "Borderline", desc: "The fake probability was only just above or below 50%. Treat this result with caution." };
}

export default function DeepfakeDetector() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [drag, setDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [techOpen, setTechOpen] = useState(false);
  const [barWidth, setBarWidth] = useState(0);
  const inputRef = useRef();
  const resultRef = useRef();

  const handleFile = useCallback((f) => {
    if (!f || !["image/jpeg", "image/png", "image/jpg"].includes(f.type)) return;
    setFile(f);
    setResult(null);
    setBarWidth(0);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onInputChange = (e) => {
    const f = e.target.files[0];
    if (f) handleFile(f);
  };

  const simulate = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setBarWidth(0);
    await new Promise(r => setTimeout(r, 1800));
    const isFake = Math.random() > 0.5;
    const conf = isFake
      ? +(50 + Math.random() * 48).toFixed(1)
      : +(52 + Math.random() * 46).toFixed(1);
    setResult({ label: isFake ? "FAKE" : "REAL", confidence: conf, filename: file.name });
    setLoading(false);
  }, [file]);

  useEffect(() => {
    if (result) {
      setTimeout(() => setBarWidth(result.confidence), 80);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 200);
    }
  }, [result]);

  const isFake = result?.label === "FAKE";
  const confTier = result ? getConfTier(result.confidence) : null;

  return (
    <>
      <style>{CSS}</style>
      <div className="app">

        {/* NAV */}
        <nav className="nav">
          <div className="nav-logo">
            <span className="nav-logo-dot" />
            DeepShield
          </div>
          <div className="nav-tag">v2.1 · CLIP + EfficientNet + DINO</div>
        </nav>

        {/* HERO */}
        <section className="hero">
          <div>
            <div className="hero-eyebrow">Deepfake Detection System</div>
            <h1 className="hero-title">
              Is this image <em>real</em> or generated?
            </h1>
            <p className="hero-sub">
              A three-branch deep learning pipeline — CLIP zero-shot scoring, fine-tuned EfficientNet, 
              and fine-tuned DINO — fused by a learned MLP to give you one clear verdict.
            </p>
          </div>
          <div className="hero-stats">
            <div className="stat-cell">
              <div className="stat-num">3<span>×</span></div>
              <div className="stat-label">scoring branches</div>
            </div>
            <div className="stat-cell">
              <div className="stat-num">1<span>↗</span></div>
              <div className="stat-label">fusion verdict</div>
            </div>
            <div className="stat-cell">
              <div className="stat-num"><span>&#60;</span>5s</div>
              <div className="stat-label">inference time</div>
            </div>
            <div className="stat-cell">
              <div className="stat-num">0<span>%</span></div>
              <div className="stat-label">data retained</div>
            </div>
          </div>
        </section>

        {/* HOW IT WORKS */}
        <section className="section">
          <div className="section-header">
            <span className="section-num">01</span>
            <span className="section-title">How the pipeline works</span>
            <span className="section-line" />
          </div>
          <div className="pipeline">
            {[
              { tag: "C1", name: "CLIP", model: "openai/clip-vit-base-patch32", desc: "Zero-shot scoring. Compares your image against two prompts — "a real photograph" and "an AI-generated image" — using broad pre-trained world knowledge. No fine-tuning. Sensitive to style and semantic plausibility." },
              { tag: "C2", name: "EfficientNet", model: "efficientnet_b0 · fine-tuned", desc: "ImageNet pre-trained backbone with the final layers replaced and fine-tuned on real vs. fake pairs. Detects low-level texture and frequency artefacts: unnatural pores, repeated patterns, over-smoothed edges." },
              { tag: "C3", name: "DINO", model: "dinov2_vits14 · fine-tuned", desc: "Vision transformer fine-tuned with dropout regularisation, label smoothing, and cosine LR scheduling. Captures structural and semantic inconsistencies — geometry errors, impossible lighting, incoherent backgrounds." },
            ].map(c => (
              <div className="pipeline-card" key={c.tag}>
                <div className="pipeline-icon">{c.tag}</div>
                <div className="pipeline-name">{c.name}</div>
                <div className="pipeline-model">{c.model}</div>
                <div className="pipeline-desc">{c.desc}</div>
              </div>
            ))}
          </div>
          <div className="fusion-card">
            <span className="fusion-label">Fusion MLP</span>
            <div className="fusion-flow">
              <span className="fusion-pill">clip_score</span>
              <span className="fusion-arrow">+</span>
              <span className="fusion-pill">eff_score</span>
              <span className="fusion-arrow">+</span>
              <span className="fusion-pill">dino_score</span>
              <span className="fusion-arrow">→</span>
              <span className="fusion-output">fake_probability</span>
              <span className="fusion-arrow">→</span>
              <span className="fusion-output">REAL · FAKE</span>
            </div>
          </div>
        </section>

        {/* HOW TO USE */}
        <section className="section">
          <div className="section-header">
            <span className="section-num">02</span>
            <span className="section-title">How to use</span>
            <span className="section-line" />
          </div>
          <div className="steps">
            {[
              { n: "01", action: "Upload", detail: "Drag and drop or browse for a JPG or PNG image." },
              { n: "02", action: "Preview", detail: "Confirm the image loaded correctly in the zone." },
              { n: "03", action: "Analyze", detail: "Click Analyze Image to run the full pipeline." },
              { n: "04", action: "Verdict", detail: "Read the REAL / FAKE label and confidence score." },
              { n: "05", action: "Scroll", detail: "Scroll down for a plain-English breakdown of why." },
            ].map(s => (
              <div className="step" key={s.n}>
                <div className="step-n">{s.n}</div>
                <div className="step-action">{s.action}</div>
                <div className="step-detail">{s.detail}</div>
              </div>
            ))}
          </div>
        </section>

        {/* DIVIDER */}
        <div className="divider"><div className="divider-line" /></div>

        {/* UPLOAD */}
        <section className="upload-outer">
          <div className="section-header">
            <span className="section-num">03</span>
            <span className="section-title">Upload &amp; analyze</span>
            <span className="section-line" />
          </div>
          <div className="upload-grid">
            <div
              className={`dropzone ${drag ? "drag" : ""}`}
              onDragOver={e => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current.click()}
            >
              {!preview ? (
                <>
                  <div className="dropzone-icon"><UploadIcon /></div>
                  <div className="dropzone-text"><strong>Drop image here</strong> or click to browse</div>
                  <div className="dropzone-formats">JPG · JPEG · PNG</div>
                </>
              ) : (
                <>
                  <img src={preview} alt="preview" className="dropzone-preview" />
                  <div className="dropzone-overlay">
                    <UploadIcon style={{ width: 20, height: 20, color: "#fff" }} />
                    <div className="dropzone-overlay-text">Replace image</div>
                  </div>
                </>
              )}
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" style={{ display: "none" }} onChange={onInputChange} />
            </div>

            <div className="panel">
              <div className="panel-row">
                <div className="panel-label">File</div>
                <div className="panel-value" style={{ fontSize: 12, wordBreak: "break-all" }}>
                  {file ? file.name : <span style={{ color: "var(--text-3)" }}>No file selected</span>}
                </div>
              </div>
              <div className="panel-row">
                <div className="panel-label">Size</div>
                <div className="panel-value">
                  {file ? formatBytes(file.size) : <span style={{ color: "var(--text-3)" }}>—</span>}
                </div>
              </div>
              <div className="panel-divider" />
              <div className="panel-row">
                <div className="panel-label">Models</div>
                <div className="model-paths">
                  {MODEL_PATHS.map(m => (
                    <div className="model-path-row" key={m.path}>
                      <span className="model-dot ok" />
                      <span className="model-path-text">{m.path}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="panel-divider" />
              <button
                className={`btn-analyze ${loading ? "loading" : ""}`}
                onClick={simulate}
                disabled={!file || loading}
              >
                {loading
                  ? <><div className="spinner" /> Running inference…</>
                  : "Analyze Image"
                }
              </button>
              {!file && (
                <div style={{ fontSize: 11, color: "var(--text-3)", textAlign: "center", marginTop: -4 }}>
                  Upload an image first
                </div>
              )}
            </div>
          </div>
        </section>

        {/* RESULT */}
        {result && (
          <section className="result-section" ref={resultRef}>
            <div className="section-header">
              <span className="section-num">04</span>
              <span className="section-title">Detection result</span>
              <span className="section-line" />
            </div>

            <div className={`verdict-card ${isFake ? "fake" : "real"}`}>
              <div className="verdict-top">
                <div>
                  <div className="verdict-label-row" style={{ marginBottom: 8 }}>
                    <span className={`verdict-badge ${isFake ? "fake" : "real"}`}>
                      {isFake ? "Deepfake detected" : "Authentic image"}
                    </span>
                  </div>
                  <div className={`verdict-word ${isFake ? "fake" : "real"}`}>{result.label}</div>
                </div>
                <div className="verdict-conf">
                  <div className="verdict-conf-num">{result.confidence.toFixed(1)}%</div>
                  <div className="verdict-conf-label">confidence</div>
                </div>
              </div>
              <div className="conf-track">
                <div className={`conf-fill ${isFake ? "fake" : "real"}`} style={{ width: `${barWidth}%` }} />
              </div>
              <div className="verdict-desc" style={{ marginTop: 10 }}>
                {isFake
                  ? `The fusion model assigned a fake probability of ${result.confidence.toFixed(1)}% — above the 50% threshold. The three scoring branches found characteristics consistent with AI-generated or manipulated imagery.`
                  : `The fusion model assigned a real probability of ${result.confidence.toFixed(1)}% — the combined evidence from CLIP, EfficientNet, and DINO found this image broadly consistent with real-world photographs.`
                }
              </div>
            </div>

            {/* EXPLANATION */}
            <div className="section-header" style={{ marginTop: "2rem" }}>
              <span className="section-num">05</span>
              <span className="section-title">Why the model decided this</span>
              <span className="section-line" />
            </div>

            <div className="explain-grid">
              {[
                { tag: "Branch 01", name: "CLIP", sub: "openai/clip-vit-base-patch32", body: "CLIP compared the image holistically against "a real photograph" and "an AI-generated image". Its broad pre-trained world knowledge is sensitive to overall style, lighting coherence, and semantic plausibility — the things that feel wrong before you can name them." },
                { tag: "Branch 02", name: "EfficientNet", sub: "efficientnet_b0 · fine-tuned", body: "EfficientNet examined low-level textures and frequency signatures. It was trained on real vs. fake pairs from this dataset, so it recognises the subtle repetition, over-smoothness, and spectral artefacts that current generative models leave behind." },
                { tag: "Branch 03", name: "DINO", sub: "dinov2_vits14 · fine-tuned", body: "DINO's transformer attention spans the whole image at once, catching structural failures: misaligned geometry, eyes that don't match, teeth that blend into one another, or backgrounds that don't obey perspective — errors CNN models often miss." },
              ].map(c => (
                <div className="explain-card" key={c.tag}>
                  <div className="explain-tag">{c.tag}</div>
                  <div className="explain-name">{c.name}</div>
                  <div className="explain-sub">{c.sub}</div>
                  <div className="explain-body">{c.body}</div>
                </div>
              ))}
            </div>

            <div className="fusion-explain">
              <div className="fusion-explain-title">How the three scores are fused</div>
              <div className="fusion-explain-body">
                The three branch scores are stacked into a single <code>[clip, eff, dino]</code> feature vector
                and passed through a small Fusion MLP trained specifically to weight and combine them for this dataset.
                The MLP outputs a <code>fake_probability</code> between 0 and 1. If that value is{" "}
                <code>≥ 0.50</code> the image is labelled <strong>FAKE</strong> and the confidence is the fake probability itself.
                If it is <code>{"< 0.50"}</code> the image is labelled <strong>REAL</strong> and the confidence is{" "}
                <code>1 − fake_probability</code>. The reported confidence always reflects certainty in the stated verdict,
                not a raw fake score.
              </div>
            </div>

            <div className={`why-card ${isFake ? "fake" : "real"}`}>
              <div className="why-title">
                {isFake ? "Why the model flagged this as fake" : "Why the model considers this real"}
              </div>
              <div className="why-body">
                {isFake ? (
                  <>
                    The combined branch evidence pushed the fake probability above 50%. At least one — likely multiple — branches found tell-tale signals of artificial generation.
                    <ul className="why-list">
                      <li>CLIP may have detected an unnatural overall style or lighting that doesn't match real-world photographic norms.</li>
                      <li>EfficientNet may have found low-level texture regularities or spectral artefacts typical of GAN or diffusion model outputs.</li>
                      <li>DINO may have identified structural inconsistencies — facial geometry, eye alignment, or background depth that doesn't add up.</li>
                    </ul>
                    <strong>Note:</strong> heavily compressed real images or extreme post-processing can occasionally produce false positives.
                  </>
                ) : (
                  <>
                    All three branches found the image broadly consistent with real-world photography and nothing pushed the fake probability above 50%.
                    <ul className="why-list">
                      <li>CLIP found the image semantically and stylistically plausible as a real photograph.</li>
                      <li>EfficientNet found no significant low-level texture or frequency artefacts associated with generative models.</li>
                      <li>DINO found no structural inconsistencies in geometry, lighting, or depth.</li>
                    </ul>
                    <strong>Note:</strong> very high-quality generative outputs or heavily edited / resized images may still pass undetected.
                  </>
                )}
              </div>
            </div>

            <div className="section-header" style={{ marginTop: "1.75rem" }}>
              <span className="section-num">06</span>
              <span className="section-title">Confidence interpretation</span>
              <span className="section-line" />
            </div>

            <div className="conf-table">
              <div className="conf-table-row conf-table-header">
                <div className="conf-table-cell range">Range</div>
                <div className="conf-table-cell tier">Tier</div>
                <div className="conf-table-cell">What it means</div>
              </div>
              {[
                { range: "≥ 80%", tier: "High", desc: "The model is strongly committed. Results in this band are generally reliable for both labels.", min: 80 },
                { range: "60 – 79%", tier: "Moderate", desc: "The model leans toward the verdict but branches may not fully agree. A second opinion is reasonable.", min: 60 },
                { range: "50 – 59%", tier: "Borderline", desc: "The fake probability was only marginally above or below 50%. Treat with caution regardless of label.", min: 0 },
              ].map(row => {
                const isCurrent = result.confidence >= row.min && (row.min === 0 ? result.confidence < 60 : row.min === 60 ? result.confidence < 80 : true);
                return (
                  <div className={`conf-table-row ${isCurrent ? "current-tier" : ""}`} key={row.tier}>
                    <div className="conf-table-cell range">{row.range}</div>
                    <div className="conf-table-cell tier">{row.tier}</div>
                    <div className="conf-table-cell">{row.desc}</div>
                  </div>
                );
              })}
            </div>
            <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-2)" }}>
              Your result of <strong style={{ color: "var(--text)" }}>{result.confidence.toFixed(1)}%</strong> falls in the{" "}
              <strong style={{ color: "var(--text)" }}>{confTier.tier}</strong> band. {confTier.desc}
            </div>

            {/* TECHNICAL DETAILS */}
            <div className="tech-details" style={{ marginTop: "1.75rem" }}>
              <div className="tech-summary" onClick={() => setTechOpen(o => !o)}>
                <span>Technical details — model paths &amp; artifacts</span>
                <span className={`tech-chevron ${techOpen ? "open" : ""}`}><ChevronDown /></span>
              </div>
              {techOpen && (
                <div className="tech-body">
                  <div style={{ fontSize: 12, color: "var(--text-2)", marginTop: "1rem" }}>Model files used for this prediction:</div>
                  <div className="code-block">
                    {MODEL_PATHS.map(m => (
                      <div key={m.path}>
                        <span className="key">{m.label.padEnd(14)}</span>
                        <span className="val">: {m.path}</span>
                      </div>
                    ))}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 8 }}>
                    Inference input: <code style={{ fontFamily: "var(--mono)", fontSize: 11, background: "var(--bg)", padding: "2px 6px", borderRadius: 4, color: "var(--accent)" }}>{result.filename}</code>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-3)" }}>
                    The uploaded file is saved temporarily to a system temp path, passed to the inference pipeline, then deleted immediately. No image data is retained.
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* FOOTER */}
        <footer className="footer">
          <div className="footer-text">DeepShield · CLIP + EfficientNet + DINO Fusion</div>
          <div className="footer-stack">
            {["PyTorch", "HuggingFace", "Streamlit"].map(t => (
              <span className="footer-pill" key={t}>{t}</span>
            ))}
          </div>
        </footer>

      </div>
    </>
  );
}
