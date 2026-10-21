"""Model loading security utilities: allowlist enforcement and helpers.

This module centralises logic to decide whether a requested model id
or local path is permitted to be loaded. It supports two mechanisms:
- `MODEL_ALLOWLIST` env var: comma-separated model IDs
- `MODEL_ALLOWLIST_FILE` env var: path to a file with one model id per line

The allowlist is cached at import time and used by the services to
refuse loading unvetted models.
"""
from __future__ import annotations

import logging
import os
from typing import Set

logger = logging.getLogger(__name__)


def _load_allowlist_from_env() -> Set[str]:
    allowlist = set()
    env = os.getenv("MODEL_ALLOWLIST")
    if env:
        for part in env.split(','):
            part = part.strip()
            if part:
                allowlist.add(part)

    file_path = os.getenv("MODEL_ALLOWLIST_FILE")
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        allowlist.add(line)
        except Exception as e:
            logger.warning(f"Failed to read MODEL_ALLOWLIST_FILE={file_path}: {e}")

    # If still empty, populate a conservative default set of audited models:
    # the OpenRouter free-tier chain (production), the Ollama offline
    # backstop, and the offline fine-tuning base model (BASE_MODEL).
    if not allowlist:
        allowlist.update({
            # Online chain: OpenRouter free tier (Railway production).
            # Primary — safety + multimodal; fallback — reasoning;
            # last resort — 1M context, high quality.
            "google/gemma-4-26b-a4b-it:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            # Offline backstop: Ollama Qwen 2.5 3B (+ 2026 education fallbacks
            # phi4-mini / qwen3:4b, served via Ollama when no online AI is
            # reachable).
            "qwen2.5:3b",
            "phi4-mini",
            "qwen3:4b",
            # Offline fine-tuning base model (BASE_MODEL).
            "Qwen/Qwen2.5-7B-Instruct",
        })

    return allowlist


_ALLOWLIST = _load_allowlist_from_env()


def _load_model_roots() -> Set[str]:
    """Directories from which local model paths may be loaded (Phase 1 hardening).

    Configured via ``MODEL_ROOTS`` (comma-separated directories, resolved to
    absolute paths). Empty set = no local filesystem paths are loadable —
    only explicit allowlisted model ids.
    """
    roots = set()
    env = os.getenv("MODEL_ROOTS", "")
    for part in env.split(','):
        part = part.strip()
        if not part:
            continue
        roots.add(os.path.realpath(part))
    return roots


_MODEL_ROOTS = _load_model_roots()


def is_local_path_allowed(model_id: str) -> bool:
    """True only for an existing path inside an operator-configured MODEL_ROOT."""
    if not _MODEL_ROOTS:
        return False
    try:
        resolved = os.path.realpath(model_id)
    except Exception:
        return False
    for root in _MODEL_ROOTS:
        try:
            if os.path.commonpath([resolved, root]) == root and os.path.exists(resolved):
                return True
        except ValueError:
            # Different drives on Windows, or incomparable paths
            continue
    return False


def is_model_allowed(model_id: str) -> bool:
    """Return True if the given model_id or path is allowed to be loaded.

    Exact match is used for model ids. Local filesystem paths are allowed
    ONLY when they resolve inside a directory configured via ``MODEL_ROOTS``
    (fail closed: with no MODEL_ROOTS configured, no filesystem path can be
    loaded — previously any existing path was trusted).
    """
    if not model_id:
        return False

    model_id = model_id.strip()
    if is_local_path_allowed(model_id):
        return True

    # Normalize common prefixes
    normalized = model_id.rstrip('/')
    allowed = normalized in _ALLOWLIST
    if not allowed:
        # also support short names like 'flan-t5-small' mapping to google/
        if '/' not in normalized:
            candidates = [f"google/{normalized}", f"microsoft/{normalized}", f"Qwen/{normalized}"]
            for c in candidates:
                if c in _ALLOWLIST:
                    return True
    return allowed


def get_allowlist() -> Set[str]:
    """Return a copy of the configured allowlist."""
    return set(_ALLOWLIST)
