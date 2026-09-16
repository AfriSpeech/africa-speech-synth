"""Minimal .env loading.

Keys must not live in a config file that gets committed, so the tool reads them
from the environment. A `.env` beside the run is the usual way to set one
without putting it in shell history, so support it rather than only promising it
in an error message. Existing environment variables always win; nothing here
overwrites what you exported.
"""
from __future__ import annotations

import os
from typing import Optional

LOADED = False


def load(path: Optional[str] = None, override: bool = False) -> dict:
    """Read KEY=value lines from `.env` (or `path`) into os.environ."""
    global LOADED
    candidates = [path] if path else [".env", os.path.expanduser("~/.env")]
    found = {}
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        with open(candidate, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                value = value.strip().strip('"').strip("'")
                found[key] = value
                if override or key not in os.environ:
                    os.environ[key] = value
        break
    LOADED = True
    return found
