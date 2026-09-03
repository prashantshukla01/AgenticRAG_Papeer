import os
import json
from pathlib import Path
from dotenv import dotenv_values

ENV_PATH = Path(".env")
PROFILE_PATH = Path("user_profile.json")

SUPPORTED_LLM_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o3-mini",
    "o1-mini",
    "gpt-4.5-preview",
]

SUPPORTED_EMBEDDING_MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def read_env() -> dict[str, str]:
    """Read all key-value pairs from .env file directly."""
    if not ENV_PATH.exists():
        return {}
    return {k: v for k, v in dotenv_values(ENV_PATH).items() if v is not None}


def write_env(updates: dict[str, str]) -> None:
    """Update or append key-value pairs in the .env file and update os.environ."""
    current_env = read_env()
    current_env.update(updates)

    lines = []
    # Write known keys first in a clean format
    priority_keys = [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "EMBEDDING_MODEL",
        "TAVILY_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ]

    written_keys = set()
    for key in priority_keys:
        if key in current_env:
            lines.append(f"{key}={current_env[key]}")
            written_keys.add(key)

    for key, val in current_env.items():
        if key not in written_keys:
            lines.append(f"{key}={val}")

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Update current process environment
    for key, val in updates.items():
        os.environ[key] = str(val)


def get_llm_model() -> str:
    """Return the configured LLM model name from env or default."""
    return os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)


def get_embedding_model() -> str:
    """Return the configured Embedding model name from env or default."""
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_dimension(model_name: str | None = None) -> int:
    """Return the vector dimension for the given or configured embedding model."""
    name = model_name or get_embedding_model()
    return SUPPORTED_EMBEDDING_MODELS.get(name, 1536)


def load_user_profile() -> dict:
    """Load user profile details (display name, avatar, research focus)."""
    default_profile = {
        "name": "Researcher",
        "avatar": "🎓",
        "role": "Academic Researcher",
        "focus_area": "Machine Learning & AI",
        "system_notes": "Prioritize precision, citations, and mathematical clarity.",
    }
    if not PROFILE_PATH.exists():
        return default_profile
    try:
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        default_profile.update(data)
        return default_profile
    except Exception:
        return default_profile


def save_user_profile(profile_data: dict) -> None:
    """Save user profile details to disk."""
    PROFILE_PATH.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")
