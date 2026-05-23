"""Chroma Cloud 클라이언트와 컬렉션 초기화 로직.

CROMA_API_KEY 를 사용하여 Chroma Cloud 에 연결하고, smu_notices 컬렉션을 확보한다.
임베딩 ingest 는 :func:`ensure_collection_populated` 에서 처리한다.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Any | None = None
_collection: Any | None = None
_init_lock = threading.Lock()

# 청크 메타데이터 중 Chroma 가 받아들이는 스칼라 타입(str/int/float/bool) 만 추려서 저장한다.
_META_KEYS = (
    "source_id",
    "source_name",
    "source_scope",
    "source_url",
    "board_path",
    "article_no",
    "notice_title",
    "notice_url",
    "notice_date",
    "notice_year",
    "campus",
    "category",
    "category_label",
    "search_keywords_text",
    "title_tags_text",
    "writer",
    "views",
    "chunk_index",
    "chunk_tokens",
    "doc_type",
)


def get_chroma_client():
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    try:
        import chromadb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("chromadb 패키지가 설치되어 있지 않습니다. pip install chromadb") from exc

    if not settings.croma_api_key:
        raise RuntimeError("CROMA_API_KEY 환경변수가 비어 있습니다.")

    # chromadb >=0.5.5 CloudClient 사용 (Chroma Cloud)
    client_kwargs = {"api_key": settings.croma_api_key}
    if settings.croma_tenant_id:
        client_kwargs["tenant"] = settings.croma_tenant_id
    if settings.croma_database_name:
        client_kwargs["database"] = settings.croma_database_name

    _client = chromadb.CloudClient(**client_kwargs)
    logger.info(
        "Chroma Cloud 연결 완료: tenant=%s database=%s",
        settings.croma_tenant_id or "(auto)",
        settings.croma_database_name or "(auto)",
    )
    return _client


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    with _init_lock:
        if _collection is not None:
            return _collection
        settings = get_settings()
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=settings.croma_collection_name,
        )
        logger.info("Chroma 컬렉션 확보: %s", settings.croma_collection_name)
        return _collection


def _scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    return {key: _scalar(chunk.get(key, "")) for key in _META_KEYS}


def ensure_collection_populated(force: bool = False) -> int:
    """precomputed embeddings.npy + chunks.jsonl 을 Chroma 에 ingest.

    이미 컬렉션에 동일 건수가 들어있으면 skip. 반환값은 새로 업서트한 건수.
    """
    settings = get_settings()
    collection = get_collection()

    embeddings_path = Path(settings.embeddings_path)
    chunks_path = Path(settings.chunks_path)
    if not embeddings_path.exists() or not chunks_path.exists():
        logger.warning(
            "임베딩 데이터가 없어 ingest 를 건너뜁니다. embeddings=%s chunks=%s",
            embeddings_path,
            chunks_path,
        )
        return 0

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy 가 필요합니다. pip install numpy") from exc

    existing = collection.count()
    embeddings = np.load(str(embeddings_path), mmap_mode="r")
    total = int(embeddings.shape[0])

    if not force and existing >= total:
        logger.info("Chroma 컬렉션 이미 채워져 있음 (existing=%d, total=%d) — ingest skip", existing, total)
        return 0

    logger.info("Chroma ingest 시작: existing=%d, target=%d", existing, total)

    batch = 256
    inserted = 0
    with chunks_path.open("r", encoding="utf-8") as f:
        ids: list[str] = []
        docs: list[str] = []
        embs: list[list[float]] = []
        metas: list[dict[str, Any]] = []

        for idx, line in enumerate(f):
            if idx < existing and not force:
                continue
            if idx >= total:
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            ids.append(str(chunk.get("chunk_id", idx)))
            docs.append(chunk.get("text", ""))
            embs.append(embeddings[idx].astype("float32").tolist())
            metas.append(_build_metadata(chunk))

            if len(ids) >= batch:
                collection.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
                inserted += len(ids)
                logger.info("Chroma upsert 진행: +%d (total %d)", len(ids), inserted)
                ids, docs, embs, metas = [], [], [], []

        if ids:
            collection.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
            inserted += len(ids)

    logger.info("Chroma ingest 완료: 총 %d 건 upsert", inserted)
    return inserted
