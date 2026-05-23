"""Load environment variables from the repo's .env file.

Keeps secrets (notably ANTHROPIC_API_KEY) out of the shell history and out of
source control — the .env lives at the repo root and is git-ignored. Real
environment variables always win over .env values (`override=False`), so CI or
an explicit `export` still takes precedence.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


def load_env() -> bool:
    """Load `<repo>/.env` into the environment. Returns True if the file existed."""
    return load_dotenv(ENV_PATH, override=False)
