# 🤖 Agent Quick-Start Guide — Read This First

> **For any Antigravity agent picking up this project on ANY device.**

## What is this project?

A B.Tech final year project for CSE with AI specialization. We're building an **Edge-AI predictive maintenance system for server hardware** (CPU, GPU, HDD, PSU, RAM, Fans).

### 3-Device Architecture (Patient → Doctor → Hospital)

| Device | Analogy | Role |
|--------|---------|------|
| **Separate Server** (TBD) | 🏥 Patient | The machine being monitored — telemetry agent + sensors on this device |
| **Raspberry Pi 4** (4GB) | 🩺 Doctor (Edge) | Runs lightweight AI models, fast predictions, operator dashboard, rule-based chat fallback |
| **Lenovo LOQ RTX 4060** | 🏨 Hospital (Cloud) | Heavy model training, deep analysis, Grafana dashboard, **AI chatbot (Ollama + Llama 3.1 8B)** |
| **Lenovo IdeaPad S145** | 💻 Dev machine | Write code only — NOT part of the final deployed system |

> ⚠️ **The RTX 4060 laptop is NOT the server being monitored.** It is ONLY the cloud backend.

## Where to find things

| What | Where |
|------|-------|
| Project state & context | `MEMORIES.md` (root) — **READ THIS FIRST** |
| High-level overview | `PROJECT_SUMMARY.md` (root) |
| Full system design (44KB) | `docs/implementation_plan.md` |
| Design Q&A | `docs/answers_and_decisions.md` |
| Device roles detail | `docs/device_roles.md` |
| Research papers | Root folder (2 PDFs) |

## Current Status

**✅ READY FOR CODING** — Planning complete (2026-08-02). @gemini-coder should begin Phase A.

**Note:** Server hardware (patient device) is still TBD — but ~80% of code is server-hardware-independent. Start with Phase A tasks (A1-A15). See MEMORIES.md for the full task list.

## Key Rules

1. **Always read `MEMORIES.md` before doing anything**
2. **Always update `MEMORIES.md` after doing anything**
3. **The RTX 4060 laptop is NOT the monitored server — it's the CLOUD BACKEND**
4. **A SEPARATE device is the server being monitored (patient) — hardware TBD**
5. **TTF (Time-to-Failure) regression is required** — not just binary pass/fail
6. **Software-only first** — no external sensors yet, add later
7. **AI chatbot included** — Ollama + Llama 3.1 8B on LOQ, rule-based fallback on RPi4
8. **5 AI components total:** 1D-CNN, LSTM-AE, XGBoost, Isolation Forest, LLM Chatbot

## Agent Roles

- `@claude-architect` — Design & planning (**DONE ✅**)
- `@gemini-coder` — Implementation (**START Phase A now**)
- `@gemini-qa` — Testing & review (later)
- `@fix-and-summarize` — Bug fixes & docs (as needed)

## Quick Start for @gemini-coder

1. Read `MEMORIES.md` (root) — full context
2. Check `Active Sprint Tasks` section — start from first unchecked `[ ]` task
3. Phase A tasks A1-A15 can ALL be done on IdeaPad S145 (NO GPU needed)
4. Push to GitHub after A1 so other devices can sync
