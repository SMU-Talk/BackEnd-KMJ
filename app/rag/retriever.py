"""Chroma 검색 + 프롬프트용 컨텍스트 빌더."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.rag.chroma_client import get_collection
from app.rag.embeddings import embed_query
from app.schemas import ChatFilters, Notice

logger = logging.getLogger(__name__)


def _build_where(filters: ChatFilters) -> dict[str, Any] | None:
    """ChatFilters -> Chroma where 조건. Chroma 는 단일 키 사용 시 그대로,
    다중 키는 $and 로 감싸야 한다."""
    clauses: list[dict[str, Any]] = []
    if filters.tags:
        clauses.append({"category_label": {"$in": list(filters.tags)}})
    if filters.dept:
        clauses.append({"source_name": filters.dept})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve_context(question: str, filters: ChatFilters) -> tuple[list[Notice], str]:
    """질문 -> (관련 Notice 리스트, LLM 에 줄 컨텍스트 텍스트)."""
    settings = get_settings()
    try:
        collection = get_collection()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chroma 컬렉션 연결 실패 — RAG 비활성화: %s", exc)
        return [], ""

    try:
        col_count = collection.count()
    except Exception:  # noqa: BLE001
        col_count = -1
    if col_count == 0:
        logger.warning(
            "[rag] 활성 컬렉션 '%s' 이 비어 있습니다. 답변에 사용할 공지가 없습니다. "
            "OpenAI 임베딩으로 채우려면 POST /internal/rag/ingest_openai 를 호출하세요.",
            collection.name if hasattr(collection, "name") else "?",
        )

    query_vec = embed_query(question)
    where = _build_where(filters)

    try:
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=settings.chroma_top_k,
            where=where,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chroma 질의 실패 — RAG 비활성화: %s", exc)
        return [], ""

    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    logger.info(
        "[rag] 질의 결과 hits=%d collection_count=%s filter=%s question=%r",
        len(docs),
        col_count,
        where,
        question[:80],
    )

    notices: list[Notice] = []
    context_blocks: list[str] = []
    seen_keys: set[str] = set()

    for doc, meta in zip(docs, metas, strict=False):
        meta = meta or {}
        article_no = str(meta.get("article_no", "") or "")
        key = f"{meta.get('source_id','')}::{article_no}"
        title = str(meta.get("notice_title") or "(제목 없음)")
        date = str(meta.get("notice_date") or "")
        dept = str(meta.get("source_name") or "전체")
        tag = str(meta.get("category_label") or "기타")
        snippet = (doc or "").strip().replace("\n", " ")
        snippet = snippet[:240] + ("…" if len(snippet) > 240 else "")

        if key not in seen_keys:
            seen_keys.add(key)
            notices.append(Notice(title=title, tag=tag, dept=dept, date=date, body=snippet))

        context_blocks.append(
            f"[제목] {title}\n[날짜] {date}\n[소속] {dept}\n[태그] {tag}\n[본문 일부] {snippet}\n[URL] {meta.get('notice_url','')}"
        )

    context_text = "\n---\n".join(context_blocks)
    return notices, context_text
