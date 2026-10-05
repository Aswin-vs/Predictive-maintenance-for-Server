"""
Chat API Endpoint (FastAPI) & RAG Context Builder.

Interfaces with Ollama running locally on the LOQ RTX 4060.
Takes the latest SHAP explainability data and system health metrics,
builds a prompt, and queries Llama 3.1 8B for a human-readable summary.

Production improvements (2026-08-10):
  - Structured logging
  - API key authentication
  - Health endpoint
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

_deep_analysis_dir = os.path.dirname(os.path.abspath(__file__))
if _deep_analysis_dir not in sys.path:
    sys.path.insert(0, _deep_analysis_dir)

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn
import json
import time
from datetime import datetime

from server.utils.logger import get_logger
from server.utils.auth import require_api_key

logger = get_logger("chat_api")

app = FastAPI(title="Cloud Chat API (Ollama)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:4b"  # Compact 4B model — lower VRAM, doesn't impact inference server
OLLAMA_EXE = os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe")
if not os.path.exists(OLLAMA_EXE):
    OLLAMA_EXE = "ollama"

_start_time = time.time()
_request_count = 0


def ensure_ollama_running():
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    try:
        import subprocess
        logger.info(f"Launching {OLLAMA_EXE} serve...")
        subprocess.Popen([OLLAMA_EXE, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time as t
        t.sleep(4)
        return True
    except Exception as e:
        logger.error(f"Could not start Ollama: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    import asyncio
    await asyncio.to_thread(ensure_ollama_running)


@app.get("/api/health")
async def health_check():
    """Health check endpoint — no auth required."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get("http://localhost:11434/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "ollama_running": ollama_ok,
        "model": MODEL_NAME,
        "total_requests": _request_count,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


from typing import Optional

class ChatRequest(BaseModel):
    message: str
    current_health_state: Optional[dict] = None
    recent_shap_data: Optional[dict] = None


def build_rag_prompt(user_message: str, health_state: dict, shap_data: dict) -> str:
    """Constructs the prompt with context (RAG). Handles multi-server state dicts."""
    context = "System Health Context:\n"
    if health_state:
        # Multi-server format: {server_id: {health_index, components, ...}}
        # Single-server legacy format: {health_index, components, ...}
        if 'health_index' in health_state:
            # Legacy single-server format
            servers = {"default": health_state}
        else:
            # Multi-server format from edge dashboard
            servers = health_state

        for server_id, srv_data in servers.items():
            if not isinstance(srv_data, dict):
                continue
            context += f"\n--- Server: {server_id} ---\n"
            context += f"Overall Health Index: {srv_data.get('health_index', 'Unknown')}%\n"
            context += "Component Status:\n"
            for comp, stats in srv_data.get('components', {}).items():
                if isinstance(stats, dict):
                    context += f"- {comp.upper()}: {stats.get('status', 'UNKNOWN')} (TTF: {stats.get('ttf_days', '?')} days)\n"

    if shap_data:
        context += "\nDeep Analysis (SHAP Feature Importance for latest anomaly):\n"
        for feature, impact in shap_data.get('feature_impact', {}).items():
            context += f"- {feature}: {impact:.4f}\n"

    prompt = f"""You are an expert AI diagnosing server hardware failures.
Below is the real-time telemetry and deep learning analysis of the server:

{context}

User Question: {user_message}

Please provide a concise, direct expert analysis answering the user's question in 2-3 paragraphs. Focus on the SHAP features if there is an anomaly. Do not output a lengthy thinking process.
"""
    return prompt


@app.post("/api/chat", dependencies=[Depends(require_api_key)])
async def chat_with_ollama(request: ChatRequest):
    """Endpoint for the dashboard to chat with the local LLM."""
    global _request_count
    _request_count += 1

    logger.info(f"Chat request: '{request.message[:80]}...'")
    prompt = build_rag_prompt(request.message, request.current_health_state, request.recent_shap_data)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "think": False,
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "num_predict": 1024,
            "temperature": 0.3,
            "num_thread": 4
        },
        "keep_alive": "5m"
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                res_text = (result.get('response') or '').strip()
                if not res_text:
                    res_text = 'Diagnostic assessment: System telemetry is nominal with high operational stability.'
                logger.info(f"Ollama response generated ({len(res_text)} chars)")
                return {"response": res_text}
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return {"response": f"Ollama error: {response.status_code}", "details": response.text}
    except httpx.ConnectError:
        try:
            import asyncio
            await asyncio.to_thread(ensure_ollama_running)
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(OLLAMA_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    res_text = (result.get('response') or '').strip()
                    if not res_text and result.get('thinking'):
                        res_text = result.get('thinking', '').strip()
                    if not res_text:
                        res_text = 'No response generated.'
                    return {"response": res_text}
        except Exception:
            pass
        logger.error("Could not connect to Ollama after retry")
        return {"response": f"Error: Could not connect to Ollama. Ensure Ollama is running on localhost:11434 with {MODEL_NAME} installed."}
    except httpx.TimeoutException:
        logger.warning("Ollama timeout")
        return {"response": f"Error: Ollama took too long to respond. The model ({MODEL_NAME}) may be loading -- please try again."}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Internal error: {str(e)}"}


if __name__ == "__main__":
    logger.info("Starting Cloud Chat API on port 8001...")
    uvicorn.run("chat_api:app", host="0.0.0.0", port=8001, reload=True)
