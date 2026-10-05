# 📁 Antigravity Agent Handover & Prompt Repository

This permanent directory serves as the centralized hub for all Antigravity agent handoffs, specialized model prompts, implementation plans, and cross-session artifacts.

> **Current State:** Phase G1 COMPLETE. Phase G2 (TimescaleDB) is NEXT.

---

## 📂 Directory Structure

```
agent_handover/
├── README.md                                 # This file — directory overview
├── MASTER_AGENT_HANDOVER.md                  # Single-source-of-truth technical specification
├── CURRENT_SPRINT_IMPLEMENTATION_PLAN.md     # Current sprint (G2) tasks & status
├── PHASE_G_RESEARCH_PLAN.md                  # Phase G blueprint (authored by Gemini, reviewed by Claude)
├── PHASE_G2_HANDOVER.md                      # Phase G2 (TimescaleDB) — exact implementation guide
└── example_prompts/                          # Specialized copy-paste prompts for new chat sessions
    ├── PROMPT_FULL_SYSTEM_HANDOVER.md        # Complete master prompt
    ├── PROMPT_NEXT_AGENT_AUDIT.md            # QA & security audit prompt
    ├── PROMPT_NEXT_AGENT_FEATURE_DEV.md      # Feature engineering & development prompt
    ├── PROMPT_BUG_FIX_AND_HARDENING.md       # Bug fixer & hardening prompt
    └── PROMPT_VIVA_DEFENSE_PREP.md           # Examiner & viva defense prep prompt
```

---

## 🚀 How to Start a New Agent Session

### For Phase G2 (Next Task):
1. Start a new chat and give it the project folder context
2. Tell the agent: **"Read `agent_handover/MASTER_AGENT_HANDOVER.md`, then `MEMORIES.md`, then `agent_handover/PHASE_G2_HANDOVER.md`. Begin Phase G2."**

### For Complete System Handoff:
- Copy the content of [`example_prompts/PROMPT_FULL_SYSTEM_HANDOVER.md`](example_prompts/PROMPT_FULL_SYSTEM_HANDOVER.md) and paste it into the new chat session.

### For Specialized Roles (QA, Development, Bug Fixing, Viva):
- Open `agent_handover/example_prompts/` and copy the relevant prompt file.

### Persistent Project Memory:
- Always ensure the incoming agent reads `MEMORIES.md` and `PROJECT_SUMMARY.md` before generating code.
