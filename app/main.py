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

    # === Chroma ingest (Qwen 사전임베딩 모드일 때만; OpenAI 모드는 별도 엔드포인트로) ===
    if (settings.embedding_provider or "").lower() != "openai":
        async def _ingest():
            try:
                await asyncio.to_thread(ensure_collection_populated)
            except Exception:  # noqa: BLE001
                logger.exception("Chroma ingest 실패 (서비스는 계속 동작)")

        asyncio.create_task(_ingest())
    else:
        logger.info(
            "embedding_provider=openai 이므로 startup 자동 ingest 는 비활성. "
            "POST /internal/rag/ingest_openai 로 트리거하세요."
        )

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
