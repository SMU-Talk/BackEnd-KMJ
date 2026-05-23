"""Chroma Cloud 클라이언트와 컬렉션 초기화 로직.

embedding_provider 값에 따라 사용하는 컬렉션이 달라진다:
  - openai     -> settings.openai_collection_name (e.g. "smu_notices_openai", 1536-dim)
  - 그 외       -> settings.croma_collection_name  (e.g. "smu_notices",        2560-dim)

ingest 함수:
  - :func:`ensure_collection_populated`   : 기존 embeddings.npy(2560-dim Qwen) 적재 (qwen_server / placeholder 모드용)
  - :func:`ingest_chunks_with_openai`     : chunks.jsonl 텍스트를 OpenAI 로 재임베딩 후 적재
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


def _provider() -> str:
    return (get_settings().embedding_provider or "openai").lower()


def active_collection_name() -> str:
    settings = get_settings()
    # OpenAI mode 에서 OPENAI_COLLECTION_NAME 을 명시하지 않으면 CROMA_COLLECTION_NAME 을 그대로 사용한다.
    if _provider() == "openai":
        override = (settings.openai_collection_name or "").strip()
        return override or settings.croma_collection_name
    return settings.croma_collection_name


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
        client = get_chroma_client()
        name = active_collection_name()
        _collection = client.get_or_create_collection(name=name)
        logger.info("Chroma 컬렉션 확보: %s (provider=%s)", name, _provider())
        return _collection


def reset_collection_cache() -> None:
    """provider 변경 후 컬렉션 캐시 무효화."""
    global _collection
    _collection = None


def verify_chroma_connection() -> tuple[bool, str]:
    try:
        col = get_collection()
        cnt = col.count()
        return True, f"OK (collection={active_collection_name()}, count={cnt})"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def active_collection_count() -> int:
    """현재 provider 가 가리키는 컬렉션의 항목 수. 실패하면 -1."""
    try:
        return int(get_collection().count())
    except Exception:  # noqa: BLE001
        return -1


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
    """precomputed embeddings.npy(2560-dim Qwen) + chunks.jsonl 적재.

    embedding_provider == "openai" 일 때는 dim 불일치라 건너뛴다.
    """
    if _provider() == "openai":
        logger.info("embedding_provider=openai 이므로 Qwen 사전 임베딩 ingest 는 건너뜁니다. "
                    "ingest_chunks_with_openai() 를 호출하세요.")
        return 0

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

    logger.info("Chroma ingest 시작 (Qwen vectors): existing=%d, target=%d", existing, total)

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


def ingest_chunks_with_openai(limit: int | None = None, batch_size: int = 100, force: bool = False) -> int:
    """chunks.jsonl 의 텍스트를 OpenAI 임베딩으로 재계산하여 OpenAI 컬렉션에 적재.

    Args:
      limit: 최대 처리 청크 수 (None = 전체). 비용/시간 통제용.
      batch_size: OpenAI embeddings 한 번 호출 당 텍스트 수. 100~256 권장.
      force: 이미 존재해도 다시 upsert.

    Returns:
      새로 upsert 한 건수.
    """
    from app.rag.embeddings import embed_texts  # 지연 임포트 (순환 방지)

    settings = get_settings()
    chunks_path = Path(settings.chunks_path)
    if not chunks_path.exists():
        logger.warning("chunks.jsonl 이 없어 OpenAI ingest 를 건너뜁니다: %s", chunks_path)
        return 0

    # OpenAI 모드에서 사용할 컬렉션을 잡는다 (active_collection_name 과 동일 규칙)
    target_name = (settings.openai_collection_name or "").strip() or settings.croma_collection_name
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=target_name)

    existing = 0 if force else collection.count()
    logger.info(
        "OpenAI ingest 시작: collection=%s existing=%d batch=%d limit=%s",
        target_name,
        existing,
        batch_size,
        limit,
    )

    inserted = 0
    buf_ids: list[str] = []
    buf_docs: list[str] = []
    buf_metas: list[dict[str, Any]] = []

    def flush():
        nonlocal inserted
        if not buf_ids:
            return
        try:
            vectors = embed_texts(buf_docs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI 배치 임베딩 실패: %s — 해당 배치 건너뜀", exc)
            buf_ids.clear()
            buf_docs.clear()
            buf_metas.clear()
            return
        collection.upsert(ids=buf_ids, documents=buf_docs, embeddings=vectors, metadatas=buf_metas)
        inserted += len(buf_ids)
        logger.info("OpenAI upsert 진행: +%d (누적 %d)", len(buf_ids), inserted)
        buf_ids.clear()
        buf_docs.clear()
        buf_metas.clear()

    with chunks_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not force and idx < existing:
                continue
            if limit is not None and inserted + len(buf_ids) >= limit:
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = (chunk.get("text") or "").strip()
            if not text:
                continue

            buf_ids.append(str(chunk.get("chunk_id", idx)))
            buf_docs.append(text)
            buf_metas.append(_build_metadata(chunk))

            if len(buf_ids) >= batch_size:
                flush()

        flush()

    logger.info("OpenAI ingest 완료: 총 %d 건 upsert", inserted)
    return inserted
