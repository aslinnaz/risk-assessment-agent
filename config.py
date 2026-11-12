"""Environment settings. Keys are read from .env or Streamlit secrets; they are never printed."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:  
    pass

ROOT = Path(__file__).parent
SOURCES_DIR = Path(os.environ["TRUSTAI_SOURCES_DIR"]) if os.environ.get("TRUSTAI_SOURCES_DIR") else ROOT / "sources"  # override is used to test the no-sources case
CATALOG_DIR = ROOT / "catalogs"


def custom_dir() -> Path:
    """Approved sources added after deployment (PDF/DOCX + registry.json). Read at call time so tests and deployments can relocate SOURCES_DIR."""
    return SOURCES_DIR / "custom"

CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

MAX_QUESTIONS = 20      
TARGET_QUESTIONS = 12   
CONTEXT_MAX = 3         
MAX_RETRIES = 1         


def _secret(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    if v:
        return v
    try:  
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def api_key() -> str:
    return _secret("OPENAI_API_KEY")


def chat_model() -> str:
    return _secret("OPENAI_MODEL", "gpt-4.1-mini")


def embed_model() -> str:
    return _secret("OPENAI_EMBED_MODEL", "text-embedding-3-small")
