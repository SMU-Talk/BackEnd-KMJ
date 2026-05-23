"""챗봇 응답 생성 서비스.

- :func:`answer_notice` 는 비-스트리밍 응답 (기존 호환).
- :func:`stream_answer_notice` 는 OpenAI 스트리밍 응답 generator (SSE 용).
RAG: app.rag.retriever.retrieve_context 가 Chroma 에서 관련 청크를 가져온다.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator

from app.core.config import get_settings
from app.core.prompt import NOTICE_SYSTEM_PROMPT, build_notice_prompt
from app.rag.retriever import retrieve_context
from app.schemas import ChatFilters, ChatResponse, Notice

logger = logging.getLogger(__name__)


def _filter_label(filters: ChatFilters) -> str:
    base = filters.dept or "전체 학과"
    if filters.major:
        base = f"{base} · {filters.major}"
    if filters.tags:
        base = f"{base} / 태그: {', '.join(filters.tags)}"
    return base


def _build_messages(question: str, filters: ChatFilters, context: str) -> list[dict[str, str]]:
    user_prompt = build_notice_prompt(question, filters.model_dump(), context)
    return [
        {"role": "system", "content": NOTICE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _get_openai_client():
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=settings.openai_api_key)


def answer_notice(question: str, filters: ChatFilters) -> ChatResponse:
    """비-스트리밍 응답 (기존 /api/chat 호환). OpenAI 가 실패하면 컨텍스트 요약으로 폴백."""
    notices, context = retrieve_context(question, filters)
    label = _filter_label(filters)

    try:
        client = _get_openai_client()
        settings = get_settings()
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=_build_messages(question, filters, context),
            temperature=0.2,
        )
        message = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI 호출 실패 — RAG 결과만 반환: %s", exc)
        if notices:
            message = f"{label} 조건으로 관련 공지 {len(notices)}건을 찾았습니다. (LLM 미연결)"
        else:
            message = f"{label} 조건에 맞는 공지를 찾지 못했습니다. (LLM 미연결)"

    return ChatResponse(message=message, notices=notices)


async def stream_answer_notice(
    question: str, filters: ChatFilters
) -> AsyncGenerator[str, None]:
    """SSE 스트림. 첫 이벤트로 notices 메타데이터, 이후 token 청크, 마지막에 done."""
    notices, context = retrieve_context(question, filters)

    yield _sse_event("notices", {"notices": [n.model_dump() for n in notices]})

    try:
        client = _get_openai_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI 클라이언트 생성 실패: %s", exc)
        yield _sse_event("error", {"message": "LLM 이 연결되지 않았습니다.", "detail": str(exc)})
        yield _sse_event("done", {})
        return

    settings = get_settings()
    messages = _build_messages(question, filters, context)

    try:
        stream = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None)
            except (AttributeError, IndexError):
                token = None
            if token:
                yield _sse_event("token", {"text": token})
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI 스트리밍 실패")
        yield _sse_event("error", {"message": "LLM 응답 생성에 실패했습니다.", "detail": str(exc)})

    yield _sse_event("done", {})


def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# 하위호환: ChatRequest.filters 가 dict 로 들어올 수도 있을 때 변환 헬퍼
def ensure_filters(filters) -> ChatFilters:
    if isinstance(filters, ChatFilters):
        return filters
    if isinstance(filters, dict):
        return ChatFilters(**filters)
    return ChatFilters()
