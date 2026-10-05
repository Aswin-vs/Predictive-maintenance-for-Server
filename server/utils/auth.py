"""
API Key Authentication Middleware.

Validates X-API-Key header against configured keys for cloud endpoints.
Used by inference_server.py and chat_api.py.

Usage:
    from server.utils.auth import require_api_key
    
    @app.post("/api/offload", dependencies=[Depends(require_api_key)])
    async def offload_inference(request: Request):
        ...
"""

import os
import json
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))

# Load API keys from config
_API_KEYS = set()
_keys_path = os.path.join(BASE_DIR, 'configs', 'api_keys.json')
if os.path.exists(_keys_path):
    with open(_keys_path) as f:
        _API_KEYS = set(json.load(f).get('keys', []))

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)):
    """
    FastAPI dependency that validates the X-API-Key header.
    Raises 401 if the key is missing or invalid.
    Skips validation only if no API keys are configured (dev/testing mode).
    
    NOTE (2026-08-12): PM_AUTH_DISABLED backdoor removed for production safety.
    If keys are configured in configs/api_keys.json, auth is mandatory.
    """
    if not _API_KEYS:
        return api_key  # No keys configured — dev/testing pass-through
    
    if not api_key or api_key not in _API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set X-API-Key header."
        )
    return api_key
