import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logger = logging.getLogger(__name__)

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    _client = OpenAI(**kwargs)
    logger.info("OpenAI client initialized (base_url=%s)", base_url or "default")
    return _client


def reset_client():
    global _client
    _client = None
