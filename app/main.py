import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.chat.router import router as chat_router
from app.core.config import get_settings
from app.Crawl import run_crawl_job, start_scheduler, stop_scheduler
from app.database import create_tables
from app.login.router import router as auth_router
from app.rag.chroma_client import ensure_collection_populated

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_tables()

    settings = get_settings()

    # Chroma ingest 는 시작 시 백그라운드로. 실패해도 서버는 떠 있어야 함.
    async def _ingest():
        try:
            await asyncio.to_thread(ensure_collection_populated)
        except Exception:  # noqa: BLE001
            logger.exception("Chroma ingest 실패 (서비스는 계속 동작)")

    asyncio.create_task(_ingest())

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


@app.post("/internal/crawl/run", include_in_schema=False)
async def trigger_crawl() -> dict[str, int]:
    """수동 트리거 (개발/디버그용). 운영에서는 라우팅 차단 권장."""
    saved = await run_crawl_job()
    return {"saved": saved}
