import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.chat.router import router as chat_router
from app.core.config import get_settings
from app.Crawl import run_crawl_job, start_scheduler, stop_scheduler
from app.database import create_tables
from app.login.router import router as auth_router
from app.rag import (
    active_collection_count,
    active_collection_name,
    ensure_collection_populated,
    ingest_chunks_with_openai,
    verify_chroma_connection,
    verify_openai_connection,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_tables()

    settings = get_settings()

    # === 헬스체크 (시작 직후 백그라운드) ===
    async def _healthcheck():
        try:
            ok_oai, msg_oai = await asyncio.to_thread(verify_openai_connection)
            logger.info("[health] OpenAI : %s — %s", "OK" if ok_oai else "FAIL", msg_oai)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[health] OpenAI 점검 중 오류: %s", exc)

        try:
            ok_ch, msg_ch = await asyncio.to_thread(verify_chroma_connection)
            logger.info(
                "[health] Chroma : %s — %s (provider=%s, collection=%s)",
                "OK" if ok_ch else "FAIL",
                msg_ch,
                settings.embedding_provider,
                active_collection_name(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[health] Chroma 점검 중 오류: %s", exc)

    asyncio.create_task(_healthcheck())

    # === Chroma ingest ===
    provider = (settings.embedding_provider or "").lower()
    if provider != "openai":
        async def _ingest():
            try:
                await asyncio.to_thread(ensure_collection_populated)
            except Exception:  # noqa: BLE001
                logger.exception("Chroma ingest 실패 (서비스는 계속 동작)")

        asyncio.create_task(_ingest())
    else:
        async def _check_or_autoingest():
            try:
                count = await asyncio.to_thread(active_collection_count)
            except Exception:  # noqa: BLE001
                logger.exception("Chroma 컬렉션 카운트 조회 실패")
                return

            logger.info(
                "[rag] 활성 컬렉션 '%s' 의 현재 항목 수: %d",
                active_collection_name(),
                count,
            )

            if count > 0:
                return

            if settings.auto_ingest_openai_on_startup:
                logger.warning(
                    "[rag] 컬렉션이 비어 있어 자동 OpenAI ingest 를 시작합니다 (limit=%s). "
                    "비용·시간이 소요됩니다.",
                    settings.auto_ingest_openai_limit,
                )
                try:
                    inserted = await asyncio.to_thread(
                        ingest_chunks_with_openai,
                        settings.auto_ingest_openai_limit,
                        100,
                        False,
                    )
                    logger.info("[rag] 자동 OpenAI ingest 완료: %d 건 적재", inserted)
                except Exception:  # noqa: BLE001
                    logger.exception("자동 OpenAI ingest 실패 (서비스는 계속 동작)")
            else:
                logger.warning(
                    "[rag] 활성 컬렉션 '%s' 이 비어 있습니다. RAG 답변이 동작하지 않습니다.\n"
                    "    → 다음 명령으로 적재하세요:\n"
                    "      curl -X POST 'http://localhost:8001/internal/rag/ingest_openai?limit=2000'\n"
                    "    → 또는 .env 에 AUTO_INGEST_OPENAI_ON_STARTUP=true 를 설정하세요.",
                    active_collection_name(),
                )

        asyncio.create_task(_check_or_autoingest())

    start_scheduler()

    if settings.crawl_run_on_startup:
        async def _initial_crawl():
            try:
                await run_crawl_job()
            except Exception:  # noqa: BLE001
                logger.exception("초기 크롤 실패")
        asyncio.create_task(_initial_crawl())

    try:
        yield
    finally:
        stop_scheduler()


settings = get_settings()

# Basic logging setup: stream + rotating file in ./logs/backend.log
log_level = logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
try:
    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(logs_dir / "backend.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
except Exception:
    logger.warning("로그 파일 핸들러 생성 실패; 계속 진행합니다.")

app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/full")
def health_full() -> dict[str, dict]:
    """OpenAI / Chroma 연동을 실시간 점검한다."""
    ok_oai, msg_oai = verify_openai_connection()
    ok_ch, msg_ch = verify_chroma_connection()
    return {
        "openai": {"ok": ok_oai, "detail": msg_oai},
        "chroma": {
            "ok": ok_ch,
            "detail": msg_ch,
            "provider": settings.embedding_provider,
            "collection": active_collection_name(),
            "count": active_collection_count(),
        },
    }


@app.post("/internal/crawl/run", include_in_schema=False)
async def trigger_crawl() -> dict[str, int]:
    saved = await run_crawl_job()
    return {"saved": saved}


@app.post("/internal/rag/ingest_openai", include_in_schema=False)
async def trigger_openai_ingest(
    limit: int | None = Query(default=None, description="처리할 최대 청크 수 (테스트용 제한). 미지정 시 전체."),
    batch_size: int = Query(default=100, ge=1, le=512),
    force: bool = Query(default=False, description="이미 존재해도 다시 upsert"),
) -> dict[str, int | str]:
    """chunks.jsonl 텍스트를 OpenAI 임베딩으로 재계산하여 Chroma에 적재.

    실측 비용: text-embedding-3-small 기준 1M 토큰당 약 $0.02.
    122,799 청크 전체 ingest 시 대략 $1~2 (chunks 평균 ~500토큰 가정).
    """
    inserted = await asyncio.to_thread(
        ingest_chunks_with_openai, limit, batch_size, force
    )
    return {"inserted": inserted, "collection": active_collection_name()}
