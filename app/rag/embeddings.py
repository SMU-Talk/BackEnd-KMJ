"""쿼리 임베딩 클라이언트.

현재는 실제 임베딩 서버(Qwen3-Embedding-4B)가 아직 붙지 않았으므로
플레이스홀더로 동작한다. 서버 연결 후 :func:`embed_query` 만 교체하면
RAG 파이프라인의 다른 부분은 그대로 사용할 수 있다.

연결 후 예상되는 호출:
    POST {EMBEDDING_SERVER_URL}/embed  {"text": "..."}
    -> {"embedding": [float, ... len=2560]}
"""

from __future__ import annotations

import hashlib
import logging
from typing import List

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _deterministic_placeholder(text: str, dim: int) -> List[float]:
    """디버그/개발용 결정적 placeholder 벡터. 동일 텍스트는 항상 같은 벡터를 반환한다.

    Chroma 가 빈 결과를 돌려주는 사태를 막기 위한 임시 값이며, 실제 검색 품질은
    임베딩 서버를 붙인 이후에야 의미가 있다.
    """
    digest = hashlib.sha512(text.encode("utf-8")).digest()
    raw = [(b - 128) / 128.0 for b in digest]
    repeated = (raw * ((dim // len(raw)) + 1))[:dim]
    # L2 정규화 (cosine 검색용)
    norm = sum(v * v for v in repeated) ** 0.5 or 1.0
    return [v / norm for v in repeated]


def embed_query(text: str) -> List[float]:
    """질의 텍스트 -> 벡터.

    TODO: 임베딩 서버 엔드포인트가 정해지면 httpx.post 로 교체.
    예시 (서버 측: Qwen3-Embedding-4B, dim=2560):

        import httpx
        with httpx.Client(timeout=10.0) as cli:
            resp = cli.post(f"{settings.embedding_server_url}/embed", json={"text": text})
            resp.raise_for_status()
            return resp.json()["embedding"]
    """
    settings = get_settings()
    logger.warning(
        "embed_query: 임베딩 서버 미연결 — 결정적 placeholder 벡터를 사용합니다. dim=%d",
        settings.embedding_dim,
    )
    return _deterministic_placeholder(text, settings.embedding_dim)
