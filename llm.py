"""OpenAI wrapper: structured output, limited retries, error messages that do not leak the key."""
from __future__ import annotations

import json
import re

import config


class SetupError(RuntimeError):
    """API key missing / invalid: a setup message is shown to the user."""


def _clean(msg: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", str(msg))[:300]


def get_client():
    key = config.api_key()
    if not key or key.startswith("sk-your") or key == "your-key-here":
        raise SetupError("OPENAI_API_KEY is not set. Put the key in the .env file (see .env.example) and restart the application.")
    from openai import OpenAI
    return OpenAI(api_key=key, max_retries=0, timeout=60)


def structured(client, system: str, user: str, schema: dict, name: str = "out", max_tokens: int = 1100, usage: list | None = None, kind: str = "chat") -> dict:
    """A single model call; retries only on a transient error (at most config.MAX_RETRIES times)."""
    last = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            r = client.chat.completions.create(
                model=config.chat_model(), temperature=0.2, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
            )
            if usage is not None and getattr(r, "usage", None):
                usage.append({"kind": kind, "prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens})
            return json.loads(r.choices[0].message.content)
        except json.JSONDecodeError as e:
            last = e
        except Exception as e:  
            last = e
            if "401" in str(e) or "invalid_api_key" in str(e):
                raise SetupError("The API key was rejected. Check the OPENAI_API_KEY value in .env.")
    raise RuntimeError("Model call failed: " + _clean(last))


def embed_query(client, text: str, usage: list | None = None, warnings: list | None = None):
    """Embeds a query. On failure it returns None (retrieval falls back to keyword search) but the failure is NOT silent: it is added to `warnings`."""
    try:
        r = client.embeddings.create(model=config.embed_model(), input=[text[:2000]])
        if usage is not None and getattr(r, "usage", None):
            usage.append({"kind": "embedding_query", "prompt": r.usage.prompt_tokens, "completion": 0})
        return r.data[0].embedding
    except Exception as e:  # noqa: BLE001
        if warnings is not None:
            msg = "Query embedding failed; keyword search was used for this step: " + _clean(e)[:120]
            if msg not in warnings:
                warnings.append(msg)
        return None


def embed_texts(client, texts: list[str], usage: list | None = None, warnings: list | None = None):
    """Embeds several short texts in one call (used for report claims). Returns a list of vectors or None."""
    try:
        r = client.embeddings.create(model=config.embed_model(), input=[t[:2000] for t in texts])
        if usage is not None and getattr(r, "usage", None):
            usage.append({"kind": "embedding_query", "prompt": r.usage.prompt_tokens, "completion": 0})
        return [d.embedding for d in r.data]
    except Exception as e:  # noqa: BLE001
        if warnings is not None:
            warnings.append("Claim embedding failed; passage support used keyword checks only: " + _clean(e)[:120])
        return None


def usage_summary(usage: list) -> dict:
    """In-session API consumption (tokens). No price estimate is made."""
    out = {"chat_calls": 0, "chat_tokens": 0, "embed_calls": 0, "embed_tokens": 0}
    for u in usage or []:
        if u["kind"].startswith("embedding"):
            out["embed_calls"] += 1
            out["embed_tokens"] += u["prompt"]
        else:
            out["chat_calls"] += 1
            out["chat_tokens"] += u["prompt"] + u["completion"]
    return out
