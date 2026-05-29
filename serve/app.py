import json
import os
import csv
import base64
from datetime import datetime
import gradio as gr
import requests

API_URL = "http://localhost:8000"

MODEL_CHOICES = [
    ("Qwen2-VL QLoRA (多模态，能描述缺陷)", "qwen"),
    ("ResNet50 (分类，合格/不合格)", "resnet"),
    ("YOLOv8 (分类，合格/不合格)", "yolo"),
]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Calistoga&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;600;700;800&display=swap');

:root {
    --background: #FAFAFA;
    --foreground: #0F172A;
    --muted: #F1F5F9;
    --muted-foreground: #64748B;
    --accent: #0052FF;
    --accent-secondary: #4D7CFF;
    --accent-foreground: #FFFFFF;
    --border: #E2E8F0;
    --card: #FFFFFF;
    --ring: #0052FF;
    --shadow-sm: 0 1px 3px rgba(15,23,42,0.06);
    --shadow-md: 0 4px 12px rgba(15,23,42,0.07);
    --shadow-lg: 0 18px 40px rgba(15,23,42,0.10);
    --shadow-accent: 0 8px 24px rgba(0,82,255,0.24);
    --font-sans: "Inter", "Noto Sans SC", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
    --font-display: "Inter", "Noto Sans SC", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
    --font-brand: "Calistoga", Georgia, serif;
    --font-mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 32px 32px !important;
    color: var(--foreground) !important;
    font-family: var(--font-sans) !important;
    overflow-x: hidden !important;
    background:
        radial-gradient(circle at 80% 0%, rgba(0,82,255,0.10), transparent 28%),
        radial-gradient(circle at 10% 12%, rgba(77,124,255,0.08), transparent 24%),
        linear-gradient(180deg, #FAFAFA 0%, #FAFAFA 100%) !important;
}

#header-block {
    max-width: 1180px;
    min-height: 310px;
    margin: 28px auto 22px;
    padding: 44px 48px;
    border-radius: 24px;
    border: 1px solid rgba(226,232,240,0.14);
    background:
        radial-gradient(circle at 82% 20%, rgba(0,82,255,0.22), transparent 26%),
        radial-gradient(circle at 88% 72%, rgba(77,124,255,0.14), transparent 24%),
        linear-gradient(135deg, #0F172A 0%, #111827 58%, #020617 100%);
    box-shadow: 0 24px 80px rgba(15,23,42,0.22);
    color: var(--accent-foreground);
    position: relative;
    overflow: hidden;
}

#header-block:before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.35) 1px, transparent 1px);
    background-size: 32px 32px;
    opacity: 0.08;
    pointer-events: none;
}

#header-block:after {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    right: -70px;
    top: -70px;
    border: 1px dashed rgba(255,255,255,0.22);
    border-radius: 50%;
    animation: rotate-slow 60s linear infinite;
    pointer-events: none;
}

.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.12fr) minmax(280px, 0.88fr);
    gap: 34px;
    align-items: center;
    position: relative;
    z-index: 1;
}

.section-label {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    border: 1px solid rgba(77,124,255,0.35);
    background: rgba(0,82,255,0.10);
    color: #BFD0FF;
    border-radius: 999px;
    padding: 8px 14px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.section-label:before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
    box-shadow: 0 0 0 6px rgba(0,82,255,0.16);
    animation: pulse-dot 2s ease-in-out infinite;
}

#header-block h1 {
    color: white !important;
    margin: 18px 0 0;
    max-width: 650px;
    font-family: var(--font-brand);
    font-size: clamp(42px, 5vw, 72px);
    font-weight: 400;
    line-height: 1.04;
    letter-spacing: -0.02em;
    position: relative;
}

.gradient-text {
    color: #4D7CFF;
    position: relative;
    white-space: nowrap;
    font-family: var(--font-display);
    font-weight: 800;
    letter-spacing: -0.04em;
}

.gradient-text:after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: -7px;
    height: 12px;
    border-radius: 4px;
    background: linear-gradient(90deg, rgba(0,82,255,0.28), rgba(77,124,255,0.08));
}

#header-block p {
    color: rgba(255,255,255,0.76) !important;
    margin: 18px 0 0 0;
    max-width: 620px;
    font-size: 16px;
    line-height: 1.7;
    position: relative;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 22px;
}

.badge {
    display: inline-flex;
    align-items: center;
    min-height: 34px;
    padding: 0 12px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: rgba(255,255,255,0.86);
    border: 1px solid rgba(255,255,255,0.14);
    position: relative;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(118px, 1fr));
    gap: 12px;
    margin-top: 26px;
    max-width: 720px;
    position: relative;
}

.metric-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 16px;
    padding: 14px 15px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}

.metric-pill strong {
    display: block;
    color: white;
    font-size: 24px;
    font-weight: 800;
    line-height: 1.1;
}

.metric-pill span {
    color: rgba(255,255,255,0.62);
    font-family: var(--font-mono);
    font-size: 12px;
}

.hero-visual {
    position: relative;
    min-height: 260px;
}

.orb-card {
    position: absolute;
    border-radius: 22px;
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(226,232,240,0.70);
    box-shadow: 0 20px 50px rgba(2,6,23,0.28);
    color: var(--foreground);
    padding: 18px;
    animation: float-card 5s ease-in-out infinite;
}

.orb-card strong {
    display: block;
    font-size: 28px;
    line-height: 1;
    color: var(--accent);
}

.orb-card span {
    display: block;
    margin-top: 8px;
    color: var(--muted-foreground);
    font-size: 13px;
}

.orb-card.primary {
    width: 210px;
    top: 20px;
    right: 44px;
}

.orb-card.secondary {
    width: 180px;
    left: 22px;
    bottom: 26px;
    animation-delay: -1.4s;
}

.visual-accent {
    position: absolute;
    right: 0;
    bottom: 18px;
    width: 112px;
    height: 112px;
    border-radius: 28px 28px 68px 28px;
    background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
    box-shadow: var(--shadow-accent);
}

.metric-card {
    background: var(--card);
    border-left: 4px solid var(--accent);
    padding: 14px 18px;
    border-radius: 14px;
    margin: 10px 0;
    box-shadow: var(--shadow-sm);
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    border-radius: 18px !important;
    border-color: var(--border) !important;
    background: rgba(255,255,255,0.92) !important;
    box-shadow: var(--shadow-sm) !important;
}

.gradio-container > .contain {
    max-width: 1180px !important;
    margin: 0 auto !important;
}

button, .gr-button {
    min-height: 44px !important;
    border-radius: 14px !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, filter 180ms ease !important;
}

button:hover, .gr-button:hover {
    transform: translateY(-1px);
}

.tab-nav button,
button[role="tab"] {
    font-size: 15px !important;
    font-weight: 650 !important;
    color: var(--muted-foreground) !important;
}

button[role="tab"][aria-selected="true"],
.tab-nav button.selected {
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}

.gradio-container label,
.gradio-container .label-wrap {
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    color: var(--foreground) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    border-radius: 14px !important;
}

.gradio-container table {
    border-radius: 14px !important;
    overflow: hidden !important;
    border-color: var(--border) !important;
}

.gradio-container th {
    background: var(--muted) !important;
    color: var(--foreground) !important;
    font-weight: 700 !important;
}

.gradio-container td {
    background: white !important;
}

details,
.accordion {
    border-radius: 18px !important;
    border-color: var(--border) !important;
    background: var(--card) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow-x: hidden !important;
}

footer { display: none !important; }

.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent-secondary)) !important;
    color: var(--accent-foreground) !important;
    border: none !important;
    box-shadow: 0 10px 22px rgba(0,82,255,0.22) !important;
}

.gr-button-primary:hover {
    filter: brightness(1.08);
    box-shadow: var(--shadow-accent) !important;
}

#footer-brand {
    max-width: 1180px;
    margin: 24px auto 0;
    padding: 24px 0 8px;
    color: var(--muted-foreground);
    font-family: var(--font-mono);
    font-size: 12px;
    text-align: center;
    border-top: 1px solid var(--border);
}

#workspace-panel {
    max-width: 1180px;
    margin: 0 auto;
    padding: 30px;
    border-radius: 28px;
    border: 1px solid rgba(226,232,240,0.85);
    background:
        radial-gradient(circle at 98% 0%, rgba(0,82,255,0.08), transparent 24%),
        linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.90));
    box-shadow: 0 24px 70px rgba(15,23,42,0.08);
    overflow: hidden;
}

.console-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    margin-bottom: 22px;
}

.console-kicker {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    border: 1px solid rgba(0,82,255,0.22);
    background: rgba(0,82,255,0.06);
    color: var(--accent);
    border-radius: 999px;
    padding: 8px 13px;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.console-kicker:before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--accent);
    box-shadow: 0 0 0 5px rgba(0,82,255,0.12);
}

.console-header h2 {
    margin: 12px 0 0;
    font-family: var(--font-display);
    font-size: clamp(28px, 2.6vw, 38px);
    line-height: 1.18;
    color: var(--foreground);
    font-weight: 800;
    letter-spacing: -0.03em;
}

.console-header p {
    margin: 8px 0 0;
    color: var(--muted-foreground);
    font-size: 15px;
    line-height: 1.7;
    max-width: 650px;
}

.console-note,
.gradio-container p,
.gradio-container td,
.gradio-container th,
.gradio-container label {
    font-family: var(--font-sans) !important;
}

.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown h3,
.gradio-container .markdown strong {
    font-family: var(--font-sans) !important;
    letter-spacing: -0.02em;
}

.gradio-container table {
    font-family: var(--font-sans) !important;
}

.console-status {
    min-width: 176px;
    border-radius: 18px;
    padding: 15px 16px;
    background: var(--foreground);
    color: white;
    box-shadow: 0 16px 38px rgba(15,23,42,0.18);
}

.console-status span {
    display: block;
    color: rgba(255,255,255,0.62);
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.console-status strong {
    display: block;
    margin-top: 6px;
    color: white;
    font-size: 19px;
}

#model-compare {
    margin-bottom: 20px !important;
}

#model-compare > .label-wrap,
#model-compare summary {
    min-height: 48px !important;
    padding: 0 18px !important;
    border-radius: 16px !important;
    background: linear-gradient(90deg, rgba(0,82,255,0.08), rgba(77,124,255,0.03)) !important;
    border: 1px solid rgba(0,82,255,0.14) !important;
    font-family: var(--font-sans) !important;
}

#console-tabs {
    background: transparent !important;
}

#console-tabs > .tab-nav,
#console-tabs .tab-nav {
    display: inline-flex !important;
    gap: 8px !important;
    padding: 6px !important;
    margin-bottom: 22px !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    background: var(--muted) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
}

#console-tabs button[role="tab"],
#console-tabs .tab-nav button {
    min-height: 38px !important;
    padding: 0 18px !important;
    border-radius: 999px !important;
    color: var(--muted-foreground) !important;
    border: 1px solid transparent !important;
    background: transparent !important;
}

#console-tabs button[role="tab"][aria-selected="true"],
#console-tabs .tab-nav button.selected {
    color: var(--accent) !important;
    background: white !important;
    border-color: rgba(0,82,255,0.18) !important;
    box-shadow: 0 8px 20px rgba(0,82,255,0.10) !important;
}

.console-note {
    padding: 12px 16px;
    border-radius: 16px;
    background: linear-gradient(90deg, rgba(15,23,42,0.04), rgba(0,82,255,0.03));
    border: 1px solid rgba(226,232,240,0.85);
    color: var(--foreground);
    margin-bottom: 16px;
}

.console-grid {
    gap: 18px !important;
}

.console-card {
    border-radius: 22px !important;
    border: 1px solid rgba(226,232,240,0.96) !important;
    background: white !important;
    box-shadow: 0 14px 36px rgba(15,23,42,0.06) !important;
    padding: 14px !important;
}

.console-card:hover {
    box-shadow: 0 20px 46px rgba(15,23,42,0.09) !important;
}

.console-card label,
.console-card .label-wrap {
    display: inline-flex !important;
    width: auto !important;
    min-height: 30px !important;
    padding: 5px 10px !important;
    border-radius: 999px !important;
    background: rgba(0,82,255,0.08) !important;
    color: var(--accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

.console-card .wrap,
.console-card .container,
.console-card .image-container,
.console-card textarea,
.console-card input,
.console-card select {
    border-radius: 16px !important;
}

.primary-console-button {
    margin-top: 12px !important;
}

.primary-console-button button,
.primary-console-button .gr-button {
    width: 100% !important;
    min-height: 54px !important;
    border-radius: 18px !important;
    font-size: 16px !important;
}

.pipeline-title {
    margin-top: 22px;
    margin-bottom: 12px;
    font-family: var(--font-display);
    font-size: 24px;
    color: var(--foreground);
    font-weight: 800;
    letter-spacing: -0.02em;
}

/* designprompts.dev-inspired app shell */
.gradio-container {
    padding: 0 28px 28px 448px !important;
    background:
        radial-gradient(circle at 82% 0%, rgba(0,82,255,0.16), transparent 26%),
        linear-gradient(180deg, #06070A 0%, #08090D 100%) !important;
}

#design-sidebar {
    position: fixed;
    inset: 0 auto 0 0;
    width: 420px;
    padding: 34px 22px 24px;
    background:
        radial-gradient(circle at 0% 0%, rgba(0,82,255,0.10), transparent 30%),
        #050506;
    color: white;
    border-right: 1px solid rgba(255,255,255,0.09);
    z-index: 20;
    overflow-y: auto;
}

.sidebar-logo {
    font-family: var(--font-sans);
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #ffffff;
}

.sidebar-logo span {
    color: var(--accent-secondary);
}

.sidebar-copy {
    margin: 22px 0 32px;
    max-width: 320px;
    color: rgba(255,255,255,0.66);
    font-size: 15px;
    line-height: 1.65;
}

.sidebar-filters {
    display: grid;
    gap: 12px;
    margin-bottom: 24px;
}

.filter-row {
    display: flex;
    align-items: center;
    gap: 8px;
    color: rgba(255,255,255,0.48);
    font-family: var(--font-mono);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.filter-pill {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 0 11px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.10);
    color: rgba(255,255,255,0.72);
    background: rgba(255,255,255,0.05);
    font-family: var(--font-sans);
    font-size: 12px;
    letter-spacing: 0;
    text-transform: none;
}

.sidebar-list {
    display: grid;
    gap: 10px;
}

.sidebar-item {
    display: grid;
    grid-template-columns: 44px 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 11px;
    border-radius: 12px;
    border: 1px solid transparent;
    background: transparent;
    color: inherit;
    text-decoration: none;
    cursor: pointer;
    transition: background 180ms ease, border-color 180ms ease, transform 180ms ease;
}

.sidebar-item:hover {
    transform: translateX(2px);
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.08);
}

.sidebar-item.active {
    background: rgba(255,255,255,0.10);
    border-color: rgba(255,255,255,0.08);
}

.sidebar-swatch {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, #ffffff, #dbeafe 40%, var(--accent));
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.28), 0 8px 20px rgba(0,82,255,0.24);
}

.sidebar-item h3 {
    margin: 0;
    font-size: 15px;
    color: white;
    font-weight: 600;
}

.sidebar-item p {
    margin: 3px 0 0;
    color: rgba(255,255,255,0.48);
    font-size: 12px;
}

.sidebar-index {
    color: rgba(255,255,255,0.30);
    font-family: var(--font-mono);
    font-size: 11px;
}

#preview-shell {
    max-width: 1180px;
    margin: 18px auto 0;
    overflow: hidden;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.13);
    background: #0b0c10 !important;
    box-shadow: 0 30px 80px rgba(0,0,0,0.38);
}

.preview-topbar {
    display: grid;
    grid-template-columns: auto minmax(240px, 1fr);
    align-items: center;
    gap: 18px;
    height: 70px;
    padding: 0 18px;
    border-bottom: 1px solid rgba(255,255,255,0.10);
    background: #0b0c10 !important;
    color: white;
}

.window-controls {
    display: flex;
    gap: 8px;
}

.window-controls i {
    width: 12px;
    height: 12px;
    border-radius: 999px;
    display: block;
}

.window-controls i:nth-child(1) { background: #ff5f57; }
.window-controls i:nth-child(2) { background: #ffbd2e; }
.window-controls i:nth-child(3) { background: #28c840; }

.address-bar {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    min-width: 430px;
    justify-self: end;
    padding: 10px 18px;
    border-radius: 12px;
    background: rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.72);
    font-family: var(--font-mono);
    font-size: 12px;
}

.address-bar:before {
    content: "";
    width: 10px;
    height: 10px;
    border-radius: 999px;
    background: #22c55e;
}

#preview-page {
    padding: 28px;
    background:
        linear-gradient(180deg, #FAFAFA 0%, #F8FAFC 100%);
}

#preview-page #header-block {
    margin-top: 0;
}

#preview-page #header-block h1 {
    font-size: clamp(42px, 4.1vw, 62px);
}

#preview-page .hero-grid {
    grid-template-columns: minmax(0, 1fr) minmax(250px, 0.86fr);
}

#preview-page .orb-card.primary {
    right: 22px;
}

#preview-page .orb-card.secondary {
    left: 0;
}

#preview-page #workspace-panel {
    margin-bottom: 0;
}

#history-actions {
    display: grid;
    grid-template-columns: minmax(160px, 220px) 1fr;
    gap: 14px;
    align-items: center;
    margin: 14px 0 18px;
}

#history-actions button {
    width: 100% !important;
}

#history-actions .block {
    min-height: 54px !important;
}

@keyframes pulse-dot {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.28); opacity: 0.72; }
}

@keyframes rotate-slow {
    to { transform: rotate(360deg); }
}

@keyframes float-card {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

@media (prefers-reduced-motion: reduce) {
    #header-block:after,
    .section-label:before,
    .orb-card {
        animation: none !important;
    }
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 0 14px 24px !important;
    }
    #design-sidebar {
        position: relative;
        width: auto;
        inset: auto;
        margin: 0 -14px 16px;
        border-right: 0;
        border-bottom: 1px solid rgba(255,255,255,0.09);
        max-height: none;
    }
    #preview-shell {
        margin-top: 0;
    }
    .preview-topbar {
        grid-template-columns: 1fr;
        height: auto;
        padding: 16px;
    }
    .address-bar {
        min-width: 0;
        width: 100%;
        justify-self: stretch;
    }
    #preview-page {
        padding: 14px;
    }
    #header-block {
        padding: 32px 24px;
        border-radius: 18px;
    }
    .hero-grid {
        grid-template-columns: 1fr;
    }
    .hero-visual {
        display: none;
    }
    .metric-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .console-header,
    #history-actions {
        grid-template-columns: 1fr;
        display: grid;
    }
}

/* Final SaaS Light pass: remove the earlier dark preview shell and align the whole page. */
body {
    background: #FAFAFA !important;
}

.gradio-container,
.main,
.contain {
    max-width: 100% !important;
    padding: 0 !important;
    background:
        radial-gradient(circle at 50% 10%, rgba(0,82,255,0.08), transparent 28%),
        radial-gradient(circle, rgba(15,23,42,0.075) 1px, transparent 1px) 0 0 / 24px 24px,
        #FAFAFA !important;
    color: #0F172A !important;
    font-family: var(--font-sans) !important;
}

#design-sidebar,
.preview-topbar {
    display: none !important;
}

#preview-shell,
#preview-page {
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background:
        radial-gradient(circle at 50% 10%, rgba(0,82,255,0.08), transparent 28%),
        radial-gradient(circle, rgba(15,23,42,0.075) 1px, transparent 1px) 0 0 / 24px 24px,
        #FAFAFA !important;
    box-shadow: none !important;
    overflow: visible !important;
}

#model-compare {
    display: none !important;
}

#preview-shell > div,
#preview-page > div,
#workspace-panel > div {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.top-nav {
    max-width: 1120px;
    margin: 0 auto;
    height: 68px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}

.brand-mark {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: #0F172A;
    font-size: 14px;
    font-weight: 800;
}

.brand-dot {
    width: 22px;
    height: 22px;
    border-radius: 999px;
    background: linear-gradient(135deg, #0F172A, #0052FF);
    box-shadow: 0 8px 22px rgba(0,82,255,0.24);
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 8px;
}

.nav-links a {
    min-height: 34px;
    display: inline-flex;
    align-items: center;
    padding: 0 12px;
    border-radius: 999px;
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
    transition: all 180ms ease;
}

.nav-links a:hover {
    color: #0052FF;
    background: rgba(0,82,255,0.06);
}

.nav-action {
    min-height: 34px;
    display: inline-flex;
    align-items: center;
    padding: 0 14px;
    border-radius: 999px;
    color: white !important;
    font-size: 12px;
    font-weight: 700;
    text-decoration: none;
    background: linear-gradient(135deg, #0052FF, #4D7CFF);
    box-shadow: 0 10px 24px rgba(0,82,255,0.22);
}

#header-block {
    max-width: none !important;
    min-height: auto !important;
    margin: 0 !important;
    padding: 0 28px 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    color: #0F172A !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

#header-block:before,
#header-block:after {
    display: none !important;
}

.hero-wrap {
    max-width: 1120px;
    min-height: 520px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: minmax(0, 1.08fr) minmax(360px, 0.92fr);
    align-items: center;
    gap: 48px;
}

.hero-grid {
    display: contents !important;
}

.section-label {
    color: #0052FF !important;
    background: rgba(0,82,255,0.05) !important;
    border: 1px solid rgba(0,82,255,0.28) !important;
    padding: 7px 14px !important;
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.16em !important;
    box-shadow: 0 10px 24px rgba(0,82,255,0.08);
}

#header-block h1 {
    max-width: 600px !important;
    margin: 24px 0 0 !important;
    color: #0F172A !important;
    font-family: var(--font-sans) !important;
    font-size: clamp(46px, 5vw, 72px) !important;
    font-weight: 500 !important;
    line-height: 1.02 !important;
    letter-spacing: 0 !important;
}

.gradient-text {
    font-family: inherit !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    background: linear-gradient(90deg, #0052FF, #4D7CFF);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}

#header-block p {
    max-width: 560px !important;
    margin-top: 20px !important;
    color: #475569 !important;
    font-size: 15px !important;
    line-height: 1.85 !important;
}

.hero-actions {
    display: flex;
    gap: 12px;
    margin-top: 28px;
    flex-wrap: wrap;
}

.hero-button {
    min-height: 44px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0 18px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 700;
    text-decoration: none;
    transition: all 180ms ease;
}

.hero-button.primary {
    color: white !important;
    background: linear-gradient(135deg, #0052FF, #4D7CFF);
    box-shadow: 0 12px 26px rgba(0,82,255,0.24);
}

.hero-button.secondary {
    color: #0F172A !important;
    border: 1px solid #E2E8F0;
    background: white;
}

.hero-button:hover {
    transform: translateY(-1px);
}

.hero-visual {
    min-height: 360px !important;
    display: grid;
    place-items: center;
    position: relative;
}

.hero-visual:before {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    border: 1px dashed rgba(0,82,255,0.18);
    border-radius: 999px;
    animation: rotate-slow 70s linear infinite;
}

.product-visual {
    position: relative;
    width: min(360px, 100%);
    min-height: 300px;
    border: 1px solid rgba(226,232,240,0.92);
    border-radius: 28px;
    background:
        radial-gradient(circle at 80% 14%, rgba(0,82,255,0.12), transparent 24%),
        linear-gradient(145deg, rgba(255,255,255,0.92), rgba(241,245,249,0.88));
    box-shadow: 0 34px 80px rgba(15,23,42,0.16);
}

.visual-card {
    position: absolute;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    background: rgba(255,255,255,0.94);
    box-shadow: 0 18px 42px rgba(15,23,42,0.12);
}

.visual-card.main {
    width: 174px;
    height: 126px;
    left: 72px;
    top: 78px;
    padding: 18px;
}

.visual-card.main i {
    display: block;
    width: 18px;
    height: 18px;
    border-radius: 6px;
    background: #0052FF;
    margin-bottom: 12px;
}

.visual-line {
    height: 10px;
    border-radius: 999px;
    background: #EAF0F8;
    margin-top: 9px;
}

.visual-line.short {
    width: 68%;
}

.visual-card.metric {
    width: 120px;
    right: 38px;
    bottom: 72px;
    padding: 15px;
}

.visual-card.metric strong {
    display: block;
    color: #0F172A;
    font-size: 20px;
    line-height: 1;
}

.visual-card.metric span {
    color: #64748B;
    font-size: 10px;
}

.visual-square {
    position: absolute;
    right: -28px;
    bottom: -24px;
    width: 66px;
    height: 66px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0052FF, #4D7CFF);
    box-shadow: 0 18px 38px rgba(0,82,255,0.34);
}

.stats-band {
    margin: 10px -28px 0;
    padding: 46px 28px;
    background:
        radial-gradient(circle at 78% 16%, rgba(77,124,255,0.14), transparent 28%),
        radial-gradient(circle, rgba(255,255,255,0.14) 1px, transparent 1px) 0 0 / 32px 32px,
        #0F172A;
}

.stats-inner {
    max-width: 1120px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 44px;
}

.stat-item strong {
    display: block;
    color: white;
    font-size: clamp(28px, 3vw, 42px);
    font-weight: 600;
    line-height: 1;
}

.stat-item span {
    display: block;
    margin-top: 10px;
    color: rgba(255,255,255,0.70);
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

#workspace-panel {
    max-width: 1120px !important;
    margin: 74px auto 44px !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.console-header {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto !important;
    gap: 28px !important;
    align-items: end !important;
    margin-bottom: 24px !important;
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
}

.console-kicker {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 14px;
    padding: 7px 13px;
    border: 1px solid rgba(0,82,255,0.28);
    border-radius: 999px;
    color: #0052FF;
    background: rgba(0,82,255,0.05);
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.console-header h2 {
    margin: 0 !important;
    color: #0F172A !important;
    font-family: var(--font-sans) !important;
    font-size: clamp(34px, 4vw, 54px) !important;
    font-weight: 500 !important;
    line-height: 1.08 !important;
    letter-spacing: 0 !important;
}

.console-header p {
    max-width: 760px;
    margin: 14px 0 0 !important;
    color: #475569 !important;
    font-size: 15px !important;
    line-height: 1.8 !important;
}

.console-status {
    min-width: 190px;
    padding: 16px 18px;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    background: white;
    box-shadow: 0 18px 40px rgba(15,23,42,0.08);
}

.console-status span {
    display: block;
    color: #64748B !important;
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

.console-status strong {
    display: block;
    margin-top: 6px;
    color: #0F172A !important;
    font-size: 18px !important;
    line-height: 1.2 !important;
}

.compare-panel,
.demo-gallery,
.tabs,
.tabitem,
.console-card,
.accordion {
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,0.92) !important;
    box-shadow: 0 18px 44px rgba(15,23,42,0.07) !important;
}

.compare-panel,
.demo-gallery {
    padding: 28px !important;
    margin-bottom: 22px;
}

.section-heading {
    display: flex;
    justify-content: space-between;
    align-items: end;
    gap: 20px;
    margin-bottom: 22px;
}

.section-heading h3 {
    margin: 8px 0 0;
    color: #0F172A;
    font-size: 28px;
    font-weight: 650;
    letter-spacing: 0;
}

.section-heading p {
    margin: 8px 0 0;
    color: #64748B;
    font-size: 13px;
    line-height: 1.7;
}

.compare-table {
    display: grid;
    overflow: hidden;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    background: white;
}

.compare-row {
    display: grid;
    grid-template-columns: 1.25fr 0.72fr 0.62fr 1.45fr;
    gap: 0;
    align-items: center;
    min-height: 58px;
    border-top: 1px solid #E2E8F0;
}

.compare-row:first-child {
    border-top: 0;
}

.compare-row > div {
    padding: 14px 18px;
    color: #0F172A;
    font-size: 13px;
}

.compare-head > div {
    color: #64748B;
    background: #F8FAFC;
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.model-name {
    font-weight: 750;
}

.cap-ok,
.cap-bad {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #0F172A;
}

.cap-ok:before,
.cap-bad:before {
    content: "";
    width: 10px;
    height: 10px;
    border-radius: 999px;
    display: inline-block;
}

.cap-ok:before {
    background: #22C55E;
    box-shadow: 0 0 0 4px rgba(34,197,94,0.12);
}

.cap-bad:before {
    background: #F97316;
    box-shadow: 0 0 0 4px rgba(249,115,22,0.12);
}

.project-note {
    margin: 18px 0 0;
    padding: 15px 18px;
    border-radius: 14px;
    color: #334155;
    background: linear-gradient(135deg, rgba(0,82,255,0.06), rgba(77,124,255,0.03));
    border: 1px solid rgba(0,82,255,0.12);
    font-size: 13px;
    line-height: 1.75;
}

.gallery-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 22px;
}

.gallery-card {
    min-width: 0;
}

.gallery-image {
    aspect-ratio: 1.55 / 1;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 16px 34px rgba(15,23,42,0.08);
    background:
        linear-gradient(135deg, rgba(0,82,255,0.20), transparent 34%),
        linear-gradient(160deg, #F8FAFC 0%, #EAF0FF 100%);
    overflow: hidden;
    position: relative;
}

.gallery-image:before {
    content: "";
    position: absolute;
    inset: 18px;
    border-radius: 12px;
    border: 1px solid rgba(0,82,255,0.18);
    background: rgba(255,255,255,0.45);
}

.gallery-image.defect:after {
    content: "";
    position: absolute;
    width: 74px;
    height: 38px;
    left: 32px;
    top: 42px;
    border: 2px solid #0052FF;
    border-radius: 10px;
    box-shadow: 98px 44px 0 -16px rgba(0,82,255,0.28);
}

.gallery-image.drilling:after {
    content: "";
    position: absolute;
    left: 50%;
    top: 18%;
    width: 3px;
    height: 62%;
    background: #0052FF;
    box-shadow: -46px 36px 0 0 rgba(15,23,42,0.18), 46px 36px 0 0 rgba(15,23,42,0.18);
    transform: translateX(-50%) rotate(10deg);
}

.gallery-image.pipeline:after {
    content: "";
    position: absolute;
    left: 30px;
    right: 30px;
    top: 50%;
    height: 2px;
    background: linear-gradient(90deg, #0052FF, #4D7CFF);
    box-shadow: 0 0 0 6px rgba(0,82,255,0.08);
}

.gallery-card h4 {
    margin: 14px 0 6px;
    color: #0F172A;
    font-size: 15px;
    font-weight: 750;
}

.gallery-card p {
    margin: 0;
    color: #64748B;
    font-size: 12px;
    line-height: 1.7;
}

#console-tabs {
    margin-top: 22px !important;
}

#console-tabs .tab-nav,
.tab-nav {
    gap: 10px !important;
    padding: 8px !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important;
    background: white !important;
    box-shadow: 0 12px 32px rgba(15,23,42,0.06) !important;
}

#console-tabs button[role="tab"] {
    min-height: 40px !important;
    border-radius: 12px !important;
    color: #64748B !important;
    font-weight: 750 !important;
    font-family: var(--font-sans) !important;
}

#console-tabs button[role="tab"][aria-selected="true"] {
    color: #0052FF !important;
    background: rgba(0,82,255,0.08) !important;
}

.console-card {
    padding: 18px !important;
}

.console-note {
    margin: 0 0 16px !important;
    padding: 14px 16px !important;
    border-radius: 14px !important;
    border: 1px solid #E2E8F0 !important;
    background: white !important;
    color: #475569 !important;
    font-size: 13px !important;
    box-shadow: 0 12px 28px rgba(15,23,42,0.05) !important;
}

.primary-console-button,
button.primary,
button[variant="primary"] {
    border: 0 !important;
    color: white !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #0052FF, #4D7CFF) !important;
    box-shadow: 0 12px 26px rgba(0,82,255,0.22) !important;
}

label,
.wrap label,
.block label span,
.prose,
.markdown,
textarea,
input,
select,
button {
    font-family: var(--font-sans) !important;
    letter-spacing: 0 !important;
}

.dataframe,
table {
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    font-family: var(--font-sans) !important;
}

th {
    background: #F8FAFC !important;
    color: #64748B !important;
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

td {
    color: #0F172A !important;
    font-size: 13px !important;
}

.export-path textarea,
.export-path input {
    font-family: var(--font-mono) !important;
}

.product-visual {
    position: relative !important;
    overflow: hidden !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.hero-image,
.gallery-image img {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
}

.hero-image {
    min-height: 300px;
    border-radius: 28px;
    border: 1px solid rgba(226,232,240,0.92);
    background: white;
    box-shadow: 0 34px 80px rgba(15,23,42,0.16);
}

.product-visual .visual-card,
.product-visual .visual-square,
.gallery-image:before,
.gallery-image:after {
    display: none !important;
}

.gallery-image {
    background: white !important;
}

.hero-metric-card {
    position: absolute;
    right: 18px;
    top: 18px;
    min-width: 0;
    padding: 8px 11px;
    border-radius: 999px;
    border: 1px solid #E2E8F0;
    background: rgba(255,255,255,0.86);
    box-shadow: 0 10px 24px rgba(15,23,42,0.10);
    backdrop-filter: blur(10px);
}

.hero-metric-card strong {
    display: inline;
    color: #0052FF;
    font-size: 12px;
    font-weight: 800;
    line-height: 1;
}

.hero-metric-card span {
    display: inline;
    margin-left: 6px;
    color: #64748B;
    font-size: 11px;
    font-weight: 650;
}

.hero-blue-square {
    position: absolute;
    right: -42px;
    bottom: -28px;
    width: 78px;
    height: 78px;
    border-radius: 20px;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.28), transparent 34%),
        linear-gradient(135deg, #0052FF 0%, #4D7CFF 100%);
    box-shadow:
        0 24px 46px rgba(0,82,255,0.34),
        inset 0 1px 0 rgba(255,255,255,0.32),
        inset -10px -12px 22px rgba(15,23,42,0.14);
    transform: rotate(-1deg);
    z-index: 2;
}

.hero-blue-square:before,
.hero-blue-square:after {
    content: "";
    position: absolute;
    border-radius: inherit;
    pointer-events: none;
}

.hero-blue-square:before {
    inset: 12px -14px -12px 14px;
    background: linear-gradient(135deg, rgba(0,44,190,0.74), rgba(0,82,255,0.24));
    filter: blur(0.2px);
    transform: skewY(10deg);
    z-index: -1;
}

.hero-blue-square:after {
    inset: 9px;
    border-radius: 15px;
    background: linear-gradient(135deg, rgba(255,255,255,0.28), transparent 48%);
    opacity: 0.72;
}

#footer-brand {
    margin: 34px 0 0;
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #E2E8F0;
    background: white;
    color: #64748B;
    font-family: var(--font-mono);
    font-size: 11px;
    text-align: center;
    box-shadow: 0 16px 38px rgba(15,23,42,0.06);
}

.anchor-target {
    display: block;
    position: relative;
    top: -18px;
    visibility: hidden;
}

@media (max-width: 900px) {
    #header-block {
        padding: 0 16px !important;
    }
    .top-nav,
    .nav-links {
        flex-wrap: wrap;
        height: auto;
        padding: 14px 0;
    }
    .hero-wrap,
    .stats-inner,
    .gallery-grid,
    .console-header {
        grid-template-columns: 1fr !important;
    }
    .hero-wrap {
        min-height: auto;
        padding: 42px 0 34px;
    }
    .hero-visual {
        display: none !important;
    }
    .stats-band {
        margin-left: -16px;
        margin-right: -16px;
    }
    #workspace-panel {
        margin: 46px 16px 32px !important;
    }
    .compare-row {
        grid-template-columns: 1fr;
    }
    .compare-row > div {
        padding: 10px 14px;
    }
}
"""

SIDEBAR_HTML = """
<aside id="design-sidebar">
    <div class="sidebar-logo">design/<span>qc</span></div>
    <p class="sidebar-copy">Industrial multimodal QC demo rebuilt as a design-forward product preview: model metrics, live inspection, pipeline trace, and exportable history.</p>
    <div class="sidebar-filters">
        <div class="filter-row">Mode <span class="filter-pill">Light</span><span class="filter-pill">Demo</span></div>
        <div class="filter-row">Stack <span class="filter-pill">Qwen2-VL</span><span class="filter-pill">FastAPI</span></div>
    </div>
    <div class="sidebar-list">
        <a class="sidebar-item active" href="#overview-section">
            <div class="sidebar-swatch"></div>
            <div><h3>MultiModal-QC</h3><p>Qwen2-VL + QLoRA</p></div>
            <div class="sidebar-index">01</div>
        </a>
        <a class="sidebar-item" href="#quick-inspect-section" onclick="document.querySelectorAll('button[role=tab]')[0]?.click(); setTimeout(()=>document.getElementById('quick-inspect-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">
            <div class="sidebar-swatch"></div>
            <div><h3>Defect Detection</h3><p>MVTec image QA</p></div>
            <div class="sidebar-index">02</div>
        </a>
        <a class="sidebar-item" href="#pipeline-section" onclick="document.querySelectorAll('button[role=tab]')[1]?.click(); setTimeout(()=>document.getElementById('pipeline-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">
            <div class="sidebar-swatch"></div>
            <div><h3>Drilling Demo</h3><p>Video compliance scene</p></div>
            <div class="sidebar-index">03</div>
        </a>
        <a class="sidebar-item" href="#history-section" onclick="document.querySelectorAll('button[role=tab]')[2]?.click(); setTimeout(()=>document.getElementById('history-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">
            <div class="sidebar-swatch"></div>
            <div><h3>Agent Pipeline</h3><p>Perceive / Detect / Decide</p></div>
            <div class="sidebar-index">04</div>
        </a>
        <a class="sidebar-item" href="#showcase-section" onclick="document.querySelectorAll('button[role=tab]')[3]?.click(); setTimeout(()=>document.getElementById('showcase-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">
            <div class="sidebar-swatch"></div>
            <div><h3>Showcase</h3><p>Demo pages and visuals</p></div>
            <div class="sidebar-index">05</div>
        </a>
        <a class="sidebar-item" href="#metrics-section" onclick="document.querySelectorAll('button[role=tab]')[4]?.click(); setTimeout(()=>document.getElementById('metrics-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">
            <div class="sidebar-swatch"></div>
            <div><h3>Metrics</h3><p>Benchmark and limits</p></div>
            <div class="sidebar-index">06</div>
        </a>
        <a class="sidebar-item" href="#deploy-section" onclick="document.querySelectorAll('button[role=tab]')[5]?.click(); setTimeout(()=>document.getElementById('deploy-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">
            <div class="sidebar-swatch"></div>
            <div><h3>Deploy</h3><p>One-click package flow</p></div>
            <div class="sidebar-index">07</div>
        </a>
    </div>
</aside>
"""

PREVIEW_CHROME_HTML = """
<div class="preview-topbar">
    <div class="window-controls"><i></i><i></i><i></i></div>
    <div class="address-bar">localhost:7860 / multimodal-qc</div>
</div>
"""


def image_data_uri(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


HERO_IMAGE_URI = image_data_uri(os.path.join("docs", "assets", "demo", "hero-qc.png"))
DEFECT_IMAGE_URI = image_data_uri(os.path.join("docs", "assets", "demo", "gallery-defect.png"))
DRILLING_IMAGE_URI = image_data_uri(os.path.join("docs", "assets", "demo", "gallery-drilling.png"))
PIPELINE_IMAGE_URI = image_data_uri(os.path.join("docs", "assets", "demo", "gallery-pipeline.png"))


HEADER_HTML = """
<div id="header-block">
    <span id="overview-section"></span>
    <div class="hero-grid">
        <div>
            <div class="section-label">Industrial VLM Demo</div>
            <h1>MultiModal-QC <span class="gradient-text">质检平台</span></h1>
            <p>基于 Qwen2-VL + QLoRA 的工业多模态检测系统，将图像缺陷识别、钻井合规场景、Agent Pipeline 和中文质检报告整合为可交互 Demo。</p>
            <div class="badge-row">
                <span class="badge">Qwen2-VL-2B</span>
                <span class="badge">QLoRA 4bit</span>
                <span class="badge">FastAPI</span>
                <span class="badge">Gradio</span>
                <span class="badge">SQLite Trace</span>
                <span class="badge">RTX 3070 Ti</span>
            </div>
            <div class="metric-strip">
                <div class="metric-pill"><strong>89.02%</strong><span>Accuracy</span></div>
                <div class="metric-pill"><strong>0.893</strong><span>F1 Score</span></div>
                <div class="metric-pill"><strong>0.249</strong><span>ROUGE-L</span></div>
                <div class="metric-pill"><strong>2 Scenes</strong><span>Image + Video</span></div>
            </div>
        </div>
        <div class="hero-visual" aria-hidden="true">
            <div class="visual-accent"></div>
            <div class="orb-card primary">
                <strong>3.24%</strong>
                <span>trainable parameters with QLoRA</span>
            </div>
            <div class="orb-card secondary">
                <strong>4</strong>
                <span>Agent stages: perceive, detect, decide, report</span>
            </div>
        </div>
    </div>
</div>
"""

INTRO_MD = """
### 📊 模型对比 (MVTec-AD 测试集 173 样本)
| 模型 | Accuracy | F1 | 能力 |
|------|----------|----|----|
| **Qwen2-VL QLoRA** | 89.02% | 0.893 | ✅ 中文描述缺陷类型/位置/严重度 |
| ResNet50 | 90.12% | 0.905 | ❌ 仅分类 |
| YOLOv8n-cls | 83.24% | 0.827 | ❌ 仅分类 |

**项目亮点:** QLoRA微调2B视觉语言模型，仅3.24%可训练参数达到与ResNet相当的精度，且具备自然语言缺陷描述能力。
"""
HEADER_HTML = """
<div id="header-block">
    <span id="overview-section" class="anchor-target"></span>
    <nav class="top-nav">
        <div class="brand-mark"><span class="brand-dot"></span>MultiModal-QC</div>
        <div class="nav-links">
            <a href="#overview-section">Overview</a>
            <a href="#demo-gallery-section">Gallery</a>
            <a href="#quick-inspect-section" onclick="document.querySelectorAll('button[role=tab]')[0]?.click(); setTimeout(()=>document.getElementById('quick-inspect-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">Demo</a>
            <a href="#pipeline-section" onclick="document.querySelectorAll('button[role=tab]')[1]?.click(); setTimeout(()=>document.getElementById('pipeline-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">Pipeline</a>
            <a href="#history-section" onclick="document.querySelectorAll('button[role=tab]')[2]?.click(); setTimeout(()=>document.getElementById('history-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">History</a>
            <a href="#showcase-section" onclick="document.querySelectorAll('button[role=tab]')[3]?.click(); setTimeout(()=>document.getElementById('showcase-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">Showcase</a>
            <a href="#metrics-section" onclick="document.querySelectorAll('button[role=tab]')[4]?.click(); setTimeout(()=>document.getElementById('metrics-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">Metrics</a>
        </div>
        <a class="nav-action" href="#history-section" onclick="document.querySelectorAll('button[role=tab]')[2]?.click(); setTimeout(()=>document.getElementById('history-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">Export CSV</a>
    </nav>
    <div class="hero-wrap">
        <div>
            <div class="section-label">Now Available</div>
            <h1>工业多模态 <span class="gradient-text">质检平台</span></h1>
            <p>基于 Qwen2-VL + QLoRA 的工业质检 Demo，将 MVTec 缺陷识别、钻井合规场景、Agent Pipeline 与可导出历史记录整合为一套面向作品集展示的交互控制台。</p>
            <div class="hero-actions">
                <a class="hero-button primary" href="#quick-inspect-section" onclick="document.querySelectorAll('button[role=tab]')[0]?.click(); setTimeout(()=>document.getElementById('quick-inspect-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">开始检测</a>
                <a class="hero-button secondary" href="#pipeline-section" onclick="document.querySelectorAll('button[role=tab]')[1]?.click(); setTimeout(()=>document.getElementById('pipeline-section')?.scrollIntoView({behavior:'smooth', block:'start'}), 80); return false;">查看 Pipeline</a>
            </div>
        </div>
        <div class="hero-visual" aria-hidden="true">
            <div class="product-visual">
                <img class="hero-image" src="__HERO_IMAGE_URI__" alt="AI 工业质检 Hero 示例图">
                <div class="hero-metric-card"><strong>F1</strong><span>0.893</span></div>
                <div class="hero-blue-square"></div>
            </div>
        </div>
    </div>
    <section class="stats-band">
        <div class="stats-inner">
            <div class="stat-item"><strong>89.02%</strong><span>Accuracy</span></div>
            <div class="stat-item"><strong>0.893</strong><span>F1 Score</span></div>
            <div class="stat-item"><strong>0.249</strong><span>ROUGE-L</span></div>
            <div class="stat-item"><strong>2 Scenes</strong><span>Image + Video</span></div>
        </div>
    </section>
</div>
"""

INTRO_HTML = """
<section class="compare-panel">
    <div class="section-heading">
        <div>
            <div class="section-label">Model Benchmark</div>
            <h3>模型对比</h3>
            <p>MVTec-AD 测试集 173 样本；README 已同步真实指标，当前以 F1=0.893 作为 Qwen2-VL QLoRA 展示结果。</p>
        </div>
    </div>
    <div class="compare-table">
        <div class="compare-row compare-head">
            <div>Model</div><div>Accuracy</div><div>F1</div><div>Capability</div>
        </div>
        <div class="compare-row">
            <div class="model-name">Qwen2-VL QLoRA</div><div>89.02%</div><div>0.893</div><div><span class="cap-ok">中文描述缺陷类型 / 位置 / 严重度</span></div>
        </div>
        <div class="compare-row">
            <div>ResNet50</div><div>90.12%</div><div>0.905</div><div><span class="cap-bad">仅分类，不生成报告</span></div>
        </div>
        <div class="compare-row">
            <div>YOLOv8n-cls</div><div>83.24%</div><div>0.827</div><div><span class="cap-bad">仅分类，不支持多模态推理</span></div>
        </div>
    </div>
    <p class="project-note">项目亮点：QLoRA 微调 2B 视觉语言模型，以 3.24% 可训练参数接近传统分类模型精度，同时保留自然语言缺陷解释能力，适合作为工业多模态质检作品集 Demo。</p>
</section>

<span id="demo-gallery-section" class="anchor-target"></span>
<section class="demo-gallery">
    <div class="section-heading">
        <div>
            <div class="section-label">Demo Gallery</div>
            <h3>示例场景</h3>
            <p>三类展示卡对应当前 Demo 的核心能力：缺陷识别、钻井视频合规、Pipeline 可追踪输出。</p>
        </div>
    </div>
    <div class="gallery-grid">
        <div class="gallery-card">
            <div class="gallery-image defect"><img src="__DEFECT_IMAGE_URI__" alt="MVTec 缺陷检测示例图"></div>
            <h4>MVTec 缺陷检测</h4>
            <p>上传产品图像，输出可视化结果与中文质检描述。</p>
        </div>
        <div class="gallery-card">
            <div class="gallery-image drilling"><img src="__DRILLING_IMAGE_URI__" alt="钻井合规检测示例图"></div>
            <h4>钻井合规场景</h4>
            <p>保留钻井视频检测入口，数据映射局限性在技术报告中说明。</p>
        </div>
        <div class="gallery-card">
            <div class="gallery-image pipeline"><img src="__PIPELINE_IMAGE_URI__" alt="Agent Pipeline 追踪示例图"></div>
            <h4>Agent Pipeline</h4>
            <p>感知、检测、决策、报告四段输出可追踪，适合面试演示。</p>
        </div>
    </div>
</section>
"""

HEADER_HTML = HEADER_HTML.replace("__HERO_IMAGE_URI__", HERO_IMAGE_URI)
INTRO_HTML = (
    INTRO_HTML
    .replace("__DEFECT_IMAGE_URI__", DEFECT_IMAGE_URI)
    .replace("__DRILLING_IMAGE_URI__", DRILLING_IMAGE_URI)
    .replace("__PIPELINE_IMAGE_URI__", PIPELINE_IMAGE_URI)
)

SHOWCASE_HTML = """
<span id="showcase-section" class="anchor-target"></span>
<section class="compare-panel">
    <div class="section-heading">
        <div>
            <div class="section-label">Showcase Pages</div>
            <h3>Demo UI 展示页</h3>
            <p>用于作品集和面试演示的扩展页面：核心场景、检测链路、历史记录导出和部署路径可以分开展示。</p>
        </div>
    </div>
    <div class="gallery-grid">
        <div class="gallery-card">
            <div class="gallery-image defect"><img src="__DEFECT_IMAGE_URI__" alt="defect inspection page"></div>
            <h4>01 / Quick Inspect</h4>
            <p>上传MVTec样本，切换Qwen2-VL QLoRA、ResNet50、YOLOv8n-cls，输出检测结论和可视化。</p>
        </div>
        <div class="gallery-card">
            <div class="gallery-image pipeline"><img src="__PIPELINE_IMAGE_URI__" alt="pipeline trace page"></div>
            <h4>02 / Pipeline Trace</h4>
            <p>展示感知、检测、决策、报告四段Agent输出，适合解释系统可追踪性。</p>
        </div>
        <div class="gallery-card">
            <div class="gallery-image drilling"><img src="__DRILLING_IMAGE_URI__" alt="drilling compliance page"></div>
            <h4>03 / Drilling Scene</h4>
            <p>保留钻井视频合规检查入口；私有视频和检查表仅本地使用，不进入公开Release。</p>
        </div>
    </div>
</section>
""".replace("__DEFECT_IMAGE_URI__", DEFECT_IMAGE_URI).replace("__PIPELINE_IMAGE_URI__", PIPELINE_IMAGE_URI).replace("__DRILLING_IMAGE_URI__", DRILLING_IMAGE_URI)

METRICS_HTML = """
<span id="metrics-section" class="anchor-target"></span>
<section class="compare-panel">
    <div class="section-heading">
        <div>
            <div class="section-label">Metrics</div>
            <h3>模型指标与公开边界</h3>
            <p>公开Demo以MVTec-AD缺陷检测为可复现实验；钻井数据只保留技术入口和转换脚本。</p>
        </div>
    </div>
    <div class="compare-table">
        <div class="compare-row compare-head"><div>Track</div><div>Dataset</div><div>Result</div><div>Public Scope</div></div>
        <div class="compare-row"><div class="model-name">Qwen2-VL QLoRA</div><div>MVTec-AD test / 173</div><div>Acc 89.02% · F1 0.893</div><div><span class="cap-ok">Code + metadata + adapter release</span></div></div>
        <div class="compare-row"><div>ResNet50</div><div>MVTec-AD test / 173</div><div>Acc 90.12% · F1 0.905</div><div><span class="cap-ok">Baseline weights release</span></div></div>
        <div class="compare-row"><div>YOLOv8n-cls</div><div>MVTec-AD test / 173</div><div>Acc 83.24% · F1 0.827</div><div><span class="cap-ok">Baseline weights release</span></div></div>
        <div class="compare-row"><div>Drilling demo</div><div>Private local data</div><div>Demo entry only</div><div><span class="cap-bad">Raw video / Excel / JSON excluded</span></div></div>
    </div>
</section>
"""

DEPLOY_HTML = """
<span id="deploy-section" class="anchor-target"></span>
<section class="compare-panel">
    <div class="section-heading">
        <div>
            <div class="section-label">Deploy</div>
            <h3>一键部署包流程</h3>
            <p>Release版面向小白复刻：下载zip、运行oneclick.bat、自动安装依赖并启动FastAPI和Gradio。</p>
        </div>
    </div>
    <div class="compare-table">
        <div class="compare-row compare-head"><div>Step</div><div>Asset / Command</div><div>Output</div><div>Note</div></div>
        <div class="compare-row"><div class="model-name">1</div><div>MultiModal-QC-oneclick.zip</div><div>源码、脚本、文档、Demo素材</div><div>不含私有钻井数据</div></div>
        <div class="compare-row"><div>2</div><div>oneclick.bat</div><div>.venv + dependencies</div><div>Windows优先</div></div>
        <div class="compare-row"><div>3</div><div>Release assets</div><div>MVTec metadata + LoRA + baselines</div><div>QLoRA按outputs/lora_defect/best归类</div></div>
        <div class="compare-row"><div>4</div><div>http://127.0.0.1:7860</div><div>Demo UI</div><div>历史记录可导出CSV到指定目录</div></div>
    </div>
</section>
"""


def quick_inspect(file, scene_type, model_type):
    if file is None:
        return None, "⚠️ 请上传文件", ""
    try:
        with open(file, "rb") as f:
            resp = requests.post(
                f"{API_URL}/quick_inspect",
                files={"file": (file.split("/")[-1].split("\\")[-1], f)},
                data={"scene_type": scene_type, "model_type": model_type},
                timeout=300,
            )
        if resp.status_code != 200:
            return file, f"❌ Error: {resp.text}", ""
        data = resp.json()
        result = data.get("result", "")
        model_name = {"qwen": "Qwen2-VL QLoRA", "resnet": "ResNet50", "yolo": "YOLOv8"}.get(model_type, model_type)

        verdict = data.get("verdict", "")
        confidence = data.get("confidence", 0)
        verdict_emoji = "✅" if verdict == "合格" else ("❌" if verdict == "不合格" else "🔍")

        result_md = f"### {verdict_emoji} 检测结果\n\n"
        result_md += f"**模型**: {model_name}\n\n"
        if verdict:
            result_md += f"**判定**: {verdict}\n\n"
        if confidence:
            result_md += f"**置信度**: {confidence:.2%}\n\n"
        result_md += f"---\n\n{result}"

        vis_url = data.get("vis_url")
        vis_img = file
        if vis_url:
            try:
                vis_resp = requests.get(f"{API_URL}{vis_url}", timeout=10)
                if vis_resp.status_code == 200:
                    tmp_path = os.path.join("storage", "vis_tmp.jpg")
                    os.makedirs("storage", exist_ok=True)
                    with open(tmp_path, "wb") as vf:
                        vf.write(vis_resp.content)
                    vis_img = tmp_path
            except Exception:
                pass

        return vis_img, result_md, result
    except requests.exceptions.ConnectionError:
        return file, "🔌 连接失败：请先启动API服务\n`uvicorn serve.api:app --port 8000`", ""
    except requests.exceptions.ReadTimeout:
        return file, "⏱️ 请求超时，请稍后重试", ""
    except Exception as e:
        return file, f"❌ 错误: {e}", ""


def full_inspect(file, scene_type):
    if file is None:
        return None, "⚠️ 请上传文件", "", "", "", ""
    try:
        with open(file, "rb") as f:
            resp = requests.post(
                f"{API_URL}/inspect",
                files={"file": (file.split("/")[-1].split("\\")[-1], f)},
                data={"scene_type": scene_type},
                timeout=600,
            )
        if resp.status_code != 200:
            return file, f"❌ Error: {resp.text}", "", "", "", ""
        data = resp.json()
        if "report" in data:
            report = data["report"]
            verdict = report.get("verdict", "未知")
            confidence = report.get("confidence", 0)
            suggestion = report.get("suggestion", "")
            defects = report.get("defects", [])
            verdict_emoji = "✅" if verdict == "合格" else ("❌" if verdict == "不合格" else "🔍")

            result_md = f"### {verdict_emoji} Agent Pipeline 检测报告\n\n"
            result_md += f"**最终判定**: {verdict}　|　**置信度**: {confidence:.2%}\n\n"
            if defects:
                result_md += "**🔴 缺陷列表**:\n"
                for d in defects:
                    result_md += f"- `{d.get('type', '')}` | 位置: {d.get('location', '')} | 严重程度: {d.get('severity', '')}\n"
            if suggestion:
                result_md += f"\n**💡 建议**: {suggestion}\n"

            vis_url = report.get("vis_url")
            vis_img = file
            if vis_url:
                try:
                    vis_resp = requests.get(f"{API_URL}{vis_url}", timeout=10)
                    if vis_resp.status_code == 200:
                        tmp_path = os.path.join("storage", "vis_pipeline.jpg")
                        os.makedirs("storage", exist_ok=True)
                        with open(tmp_path, "wb") as vf:
                            vf.write(vis_resp.content)
                        vis_img = tmp_path
                except Exception:
                    pass

            return (vis_img, result_md,
                    report.get("perception_raw", ""),
                    report.get("detection_raw", ""),
                    report.get("decision_raw", ""),
                    json.dumps(defects, ensure_ascii=False, indent=2) if defects else "无缺陷")
        return file, f"ID: {data.get('inspection_id', '')}", "", "", "", ""
    except requests.exceptions.ConnectionError:
        return file, "🔌 连接失败：请先启动API服务", "", "", "", ""
    except requests.exceptions.ReadTimeout:
        return file, "⏱️ 请求超时", "", "", "", ""
    except Exception as e:
        return file, f"❌ 错误: {e}", "", "", "", ""


HISTORY_HEADERS = [
    "ID", "场景", "模型/流程", "判定", "置信度/拟合度",
    "缺陷类型", "位置", "严重度", "输入文件", "创建时间", "完成时间", "建议",
]


def _format_score(value):
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def _load_json_maybe(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _inspection_detail_row(item):
    detail = {}
    try:
        resp = requests.get(f"{API_URL}/inspections/{item.get('id')}", timeout=5)
        if resp.status_code == 200:
            detail = resp.json()
    except Exception:
        detail = {}

    results = detail.get("results", [])
    defects = detail.get("defects", [])
    result_by_step = {r.get("agent_step", ""): r for r in results}
    report = _load_json_maybe(result_by_step.get("report", {}).get("agent_output", ""))
    detect = _load_json_maybe(result_by_step.get("detect", {}).get("agent_output", ""))

    model_or_flow = "Agent Pipeline" if results else "历史记录"
    quick_steps = [s for s in result_by_step.keys() if s in {"qwen", "resnet", "yolo"}]
    if quick_steps:
        model_or_flow = {"qwen": "Qwen2-VL QLoRA", "resnet": "ResNet50", "yolo": "YOLOv8"}.get(quick_steps[0], quick_steps[0])

    verdict = report.get("verdict") or item.get("status", "")
    confidence = report.get("confidence")
    if confidence in ("", None):
        confidence = detect.get("confidence")
    if confidence in ("", None) and results:
        confidence = results[-1].get("confidence")

    first_defect = {}
    if report.get("defects"):
        first_defect = report["defects"][0]
    elif defects:
        first_defect = defects[0]

    defect_type = first_defect.get("type") or first_defect.get("defect_type", "")
    location = first_defect.get("location", "")
    severity = first_defect.get("severity", "")
    suggestion = first_defect.get("suggestion") or report.get("suggestion", "")
    input_name = os.path.basename(str(item.get("input_path", "")))

    return [
        item.get("id", ""),
        item.get("scene_type", ""),
        model_or_flow,
        verdict,
        _format_score(confidence),
        defect_type,
        location,
        severity,
        input_name,
        item.get("created_at", ""),
        item.get("completed_at", ""),
        suggestion,
    ]


def get_history_rows(limit=20):
    try:
        resp = requests.get(f"{API_URL}/inspections?limit={limit}", timeout=5)
        data = resp.json()["inspections"]
        return [_inspection_detail_row(item) for item in data]
    except Exception:
        return []


def get_history():
    return get_history_rows(limit=20)


def export_history_csv(export_dir=None):
    export_dir = (export_dir or "").strip() or os.path.join("outputs", "test_detection_data")
    export_dir = os.path.abspath(os.path.expanduser(export_dir))
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, f"test_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    rows = get_history_rows(limit=1000)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(HISTORY_HEADERS)
        writer.writerows(rows)
    return out_path


with gr.Blocks(title="MultiModal-QC 工业缺陷检测平台", css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")) as demo:
    with gr.Group(elem_id="preview-shell"):
        with gr.Group(elem_id="preview-page"):
            gr.HTML(HEADER_HTML)

            with gr.Group(elem_id="workspace-panel"):
                gr.HTML("""
                <div class="console-header">
                    <div>
                        <div class="console-kicker">Live Inspection Console</div>
                        <h2>检测工作台</h2>
                        <p>一个面向面试和作品集展示的可交互质检控制台：上传样本、切换模型、查看可视化结果，并保留完整 Pipeline 追踪。</p>
                    </div>
                    <div class="console-status">
                        <span>Runtime</span>
                        <strong>FastAPI + Gradio</strong>
                    </div>
                </div>
                """)

                with gr.Accordion("项目简介与模型对比", open=False, elem_id="model-compare"):
                    gr.HTML(INTRO_HTML)

                gr.HTML(INTRO_HTML)

                with gr.Tabs(elem_id="console-tabs"):
                    with gr.Tab("快速检测"):
                        gr.HTML('<span id="quick-inspect-section"></span>')
                        gr.Markdown("**单次推理检测** · 选择模型 → 上传图片 → 查看结果与可视化（约 3-15 秒）", elem_classes=["console-note"])
                        with gr.Row(elem_classes=["console-grid"]):
                            with gr.Column(scale=1, elem_classes=["console-card"]):
                                q_model = gr.Dropdown(
                                    choices=MODEL_CHOICES, value="qwen", label="选择模型",
                                    info="Qwen 能描述缺陷细节，ResNet/YOLO 仅分类"
                                )
                                q_scene = gr.Dropdown(
                                    choices=["defect_3d", "drilling_compliance"],
                                    value="defect_3d", label="检测场景"
                                )
                                q_file = gr.Image(label="上传产品图片", type="filepath", height=320)
                                q_btn = gr.Button("开始检测", variant="primary", size="lg", elem_classes=["primary-console-button"])
                            with gr.Column(scale=1, elem_classes=["console-card"]):
                                q_preview = gr.Image(label="检测可视化", interactive=False, height=320)
                                q_result = gr.Markdown(label="检测结果", value="*等待检测...*")
                                with gr.Accordion("模型原始输出", open=False):
                                    q_raw = gr.Textbox(label="", lines=4, show_label=False)
                        q_btn.click(quick_inspect, inputs=[q_file, q_scene, q_model],
                                    outputs=[q_preview, q_result, q_raw])

                    with gr.Tab("完整 Pipeline"):
                        gr.HTML('<span id="pipeline-section"></span>')
                        gr.Markdown("**四阶段 Agent Pipeline**: 感知 → 检测 → 决策 → 报告（约 30 秒）", elem_classes=["console-note"])
                        with gr.Row(elem_classes=["console-grid"]):
                            with gr.Column(scale=1, elem_classes=["console-card"]):
                                f_scene = gr.Dropdown(
                                    choices=["defect_3d", "drilling_compliance"],
                                    value="defect_3d", label="检测场景"
                                )
                                f_file = gr.Image(label="上传产品图片", type="filepath", height=320)
                                f_btn = gr.Button("启动 Pipeline", variant="primary", size="lg", elem_classes=["primary-console-button"])
                            with gr.Column(scale=1, elem_classes=["console-card"]):
                                f_preview = gr.Image(label="检测可视化", interactive=False, height=320)
                                f_result = gr.Markdown(label="检测报告", value="*等待检测...*")

                        gr.HTML('<div class="pipeline-title">Agent Pipeline 中间输出</div>')
                        with gr.Row(elem_classes=["console-grid"]):
                            with gr.Column(scale=1, elem_classes=["console-card"]):
                                f_perception = gr.Textbox(label="① 感知 Agent (VLM 推理)", lines=5)
                            with gr.Column(scale=1, elem_classes=["console-card"]):
                                f_detection = gr.Textbox(label="② 检测 Agent (结构化提取)", lines=5)
                        with gr.Row(elem_classes=["console-grid"]):
                            with gr.Column(scale=1, elem_classes=["console-card"]):
                                f_decision = gr.Textbox(label="③ 决策 Agent (合规判定)", lines=5)
                            with gr.Column(scale=1, elem_classes=["console-card"]):
                                f_defects = gr.Textbox(label="④ 缺陷详情 (JSON)", lines=5)
                        f_btn.click(full_inspect, inputs=[f_file, f_scene],
                                    outputs=[f_preview, f_result, f_perception, f_detection, f_decision, f_defects])

                    with gr.Tab("历史记录"):
                        gr.HTML('<span id="history-section"></span>')
                        gr.Markdown("最近 20 条检测参数记录（持久化于 SQLite）：包含模型/流程、判定、置信度/拟合度、缺陷类型、位置、严重度与建议，可导出为 CSV 表格。", elem_classes=["console-note"])
                        export_dir_box = gr.Textbox(
                            label="测试/检测数据导出目录",
                            value=os.path.abspath(os.path.join("outputs", "test_detection_data")),
                            elem_classes=["export-path"],
                        )
                        with gr.Row(elem_id="history-actions"):
                            refresh_btn = gr.Button("刷新", size="sm")
                            export_btn = gr.Button("导出 CSV", variant="primary", size="sm")
                        export_file = gr.File(label="导出的测试/检测数据表格", interactive=False)
                        history_table = gr.Dataframe(
                            headers=HISTORY_HEADERS,
                            label="", interactive=False)
                        refresh_btn.click(get_history, outputs=[history_table])
                        export_btn.click(export_history_csv, inputs=[export_dir_box], outputs=[export_file])

                    with gr.Tab("Showcase"):
                        gr.HTML(SHOWCASE_HTML)

                    with gr.Tab("Metrics"):
                        gr.HTML(METRICS_HTML)

                    with gr.Tab("Deploy"):
                        gr.HTML(DEPLOY_HTML)

                gr.HTML("""
                <div id="footer-brand">
                    MultiModal-QC · Powered by Qwen2-VL + QLoRA + FastAPI + Gradio
                </div>
                """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
