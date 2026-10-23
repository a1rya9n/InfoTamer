# 🦥 InfoTamer (The Sloth Agent)
> **Offline-First Overnight Notification & Information Overload Tamer powered by Google Gemma 2.**

InfoTamer silently intercepts, aggregates, sanitizes, and triages daily communications (Slack, Discord, WhatsApp, Telegram, RSS feeds) offline overnight on consumer hardware. Every morning, it delivers a **hyper-concise 2-line executive summary** and a structured action brief, guaranteeing **100% data privacy** with zero cloud API dependencies.

---

## 🌟 Key Features

- **100% Offline & Private:** Runs entirely on local hardware (Apple Silicon Mac, local GPU, Raspberry Pi 5) via [Ollama](https://ollama.com) using open-weight models (`gemma2:2b` or `gemma2:9b`).
- **Pre-LLM PII Sanitization:** Automatically detects and strips API keys, passwords, bearer tokens, emails, phone numbers, and IP addresses before model inference.
- **Three-Stage Processing Pipeline:**
  1. *De-noising & Categorization:* Discards memes, social greetings, and chatter; categorizes alerts, tasks, discussions, and feeds.
  2. *Action Item Extraction:* Extracts explicit deliverables (`who`, `what`, `by when`, `urgency`).
  3. *Executive Brief Synthesis:* Compiles the morning 2-line TL;DR and detailed Markdown digest.
- **Zero-Maintenance Retention Auto-Purge:** Raw notification events older than 48 hours are automatically purged to prevent disk clutter and eliminate retention liabilities.
- **Rich Terminal UI & Desktop Alerts:** Interactive ANSI dashboards and native macOS notifications.

---

## 🏗️ Architecture

```
[ Slack / Discord / WhatsApp / Telegram / RSS ]
                     │
                     ▼
       ┌───────────────────────────┐
       │   Ingestion & PII Masker  │ (Regex Redaction of Tokens/Emails/IPs)
       └─────────────┬─────────────┘
                     ▼
       ┌───────────────────────────┐
       │   Local SQLite Buffer     │ (infotamer.db, zero cloud sync)
       └─────────────┬─────────────┘
                     ▼
       ┌───────────────────────────┐
       │   Gemma 2 (2B-IT / 9B)    │ (Local Ollama REST API: localhost:11434)
       │  Stage 1: Noise Filtering │
       │  Stage 2: Task Extraction │
       │  Stage 3: Executive Brief │
       └─────────────┬─────────────┘
                     ▼
       ┌───────────────────────────┐
       │     Morning Delivery      │
       │  • 2-Line Executive TL;DR │
       │  • Markdown Digest Report │
       │  • Terminal UI & macOS OS │
       └───────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### 2. Download Gemma 2
```bash
# Pull Google's lightweight, ultra-fast Gemma 2 2B Instruct model (1.6GB)
ollama pull gemma2:2b
```

### 3. Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 5. Run Immediate Digestion
```bash
# Run with synthetic test stream
python3 main.py --mock

# Run with an RSS feed
python3 main.py --rss https://news.ycombinator.com/rss

# Test Gemma integration
python3 test_gemma.py
```

### 6. Run Overnight Scheduled Service
```bash
# Runs daily at 06:00 AM (configured in .env)
python3 main.py --daemon --mock
```

---

## 📁 Repository Structure

```
.
├── .env.example         # Environment template
├── .env                 # Local configuration
├── requirements.txt     # Python dependencies
├── ingestor.py          # PII Sanitizer, RSS collector, SQLite buffer
├── processor.py         # Ollama / Gemma 2 inference pipeline
├── reporter.py          # 2-line summary & Markdown brief formatter
├── main.py              # CLI & APScheduler background orchestrator
├── test_gemma.py        # Diagnostic script for Gemma 2
├── briefs/              # Output directory for daily morning briefs
└── data/                # Local SQLite storage directory
```

---

## ⚖️ License
MIT License. Built for Hacktoberfest 2026.