from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.chat.service import answer_notice, stream_answer_notice
from app.login.service import require_school_session
from app.models import User
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, _: User = Depends(require_school_session)) -> ChatResponse:
    return answer_notice(payload.question.strip(), payload.filters)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    _: User = Depends(require_school_session),
) -> StreamingResponse:
    """Server-Sent Events 기반 실시간 응답.

    이벤트 종류:
      - event: notices  → {"notices": [...]}  : RAG 로 추린 공지 메타데이터
      - event: token    → {"text": "..."}     : LLM 토큰 단위 청크
      - event: error    → {"message": "...", "detail": "..."} : 오류
      - event: done     → {}                  : 스트림 종료
    """
    generator = stream_answer_notice(payload.question.strip(), payload.filters)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
