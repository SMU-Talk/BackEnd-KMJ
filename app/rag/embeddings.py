"""쿼리 임베딩 클라이언트.

3가지 제공자(provider) 를 지원한다:
  - "openai":       OpenAI Embeddings API (text-embedding-3-small, 1536-dim) - 기본값.
                    학교 자체 임베딩 서버가 붙기 전까지 실제로 동작하는 RAG 경로.
  - "qwen_server":  EMBEDDING_SERVER_URL 의 Qwen3-Embedding-4B 서버 (2560-dim).
                    기존 embeddings.npy 와 dim 일치.
  - "placeholder":  결정적 placeholder. 디버그 외에는 비추천.

호출 측은 :func:`embed_query` 와 :func:`embed_texts` 만 사용하면 된다.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import List

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------- placeholder ----------

def _deterministic_placeholder(text: str, dim: int) -> List[float]:
    digest = hashlib.sha512(text.encode("utf-8")).digest()
    raw = [(b - 128) / 128.0 for b in digest]
    repeated = (raw * ((dim // len(raw)) + 1))[:dim]
    norm = sum(v * v for v in repeated) ** 0.5 or 1.0
    return [v / norm for v in repeated]


# ---------- Qwen embedding server (HTTP) ----------

def _qwen_embed(text: str) -> List[float] | None:
    settings = get_settings()
    if not settings.embedding_server_url:
        return None
    try:
        with httpx.Client(timeout=10.0) as cli:
            resp = cli.post(
                f"{settings.embedding_server_url.rstrip('/')}/embed",
                json={"text": text},
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding") or data.get("vector")
            if not emb:
                raise RuntimeError("임베딩 응답에 'embedding' 필드가 없습니다")
            return list(map(float, emb))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qwen embed server 호출 실패: %s", exc)
        return None


# ---------- OpenAI embeddings ----------

@lru_cache(maxsize=1)
def _openai_client():
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=settings.openai_api_key)


def _openai_embed_batch(texts: List[str]) -> List[List[float]]:
    settings = get_settings()
    client = _openai_client()
    resp = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [list(d.embedding) for d in resp.data]


# ---------- public API ----------

def embed_query(text: str) -> List[float]:
    settings = get_settings()
    provider = (settings.embedding_provider or "qwen_server").lower()

    if provider == "qwen_server":
        emb = _qwen_embed(text)
        if emb is not None:
            return emb
        logger.warning(
            "embed_query: Qwen 서버 응답 실패 — placeholder 폴백 (URL=%s)",
            settings.embedding_server_url or "(미설정)",
        )
    elif provider == "openai":
        try:
            return _openai_embed_batch([text])[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI 임베딩 실패 — placeholder 폴백: %s", exc)

    dim = settings.openai_embedding_dim if provider == "openai" else settings.embedding_dim
    return _deterministic_placeholder(text, dim)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """배치 임베딩. OpenAI provider 일 때만 효율적인 배치 호출."""
    settings = get_settings()
    provider = (settings.embedding_provider or "openai").lower()

    if provider == "openai":
        return _openai_embed_batch(texts)

    # 다른 provider 는 단건 호출을 반복
    return [embed_query(t) for t in texts]


def active_embedding_dim() -> int:
    settings = get_settings()
    provider = (settings.embedding_provider or "openai").lower()
    if provider == "openai":
        return settings.openai_embedding_dim
    return settings.embedding_dim


def verify_openai_connection() -> tuple[bool, str]:
    """OpenAI 연결 헬스체크. (성공 여부, 사유) 반환."""
    settings = get_settings()
    if not settings.openai_api_key:
        return False, "OPENAI_API_KEY 가 비어 있습니다."
    try:
        client = _openai_client()
        # 가벼운 호출: 1차원짜리 임베딩 1개
        client.embeddings.create(model=settings.openai_embedding_model, input=["ping"])
        return True, f"OK (model={settings.openai_embedding_model}, dim={settings.openai_embedding_dim})"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
