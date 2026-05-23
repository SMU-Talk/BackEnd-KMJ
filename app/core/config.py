import json
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_DIR = REPO_ROOT / "smu_notice_qwen3_e1_release"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "SMU Talk API"
    api_prefix: str = "/api"
    database_url: str = "sqlite:///./smu_talk.db"

    jwt_secret_key: str = "change-this-long-random-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 8 * 60

    frontend_origins: str | list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    smu_auth_url: str = "https://smsso.smu.ac.kr/svc/tk/Auth.do?ac=Y&ifa=N&id=portal&"
    smu_login_url: str = "https://smsso.smu.ac.kr/Login.do"
    smu_portal_url: str = "https://portal.smu.ac.kr/p/S00/"
    smu_session_ttl_minutes: int = 30
    smu_session_refresh_window_minutes: int = 5
    http_timeout_seconds: float = 10.0

    # === RAG / LLM ===
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"

    # NOTE: env key is `CROMA_API_KEY` per project convention; settings field name maps via env.
    croma_api_key: str = ""
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"
    chroma_collection: str = "smu_notices"
    chroma_top_k: int = 5

    embedding_dim: int = 2560
    embeddings_path: str = str(DEFAULT_RELEASE_DIR / "embeddings.npy")
    chunks_path: str = str(DEFAULT_RELEASE_DIR / "chunks.jsonl")

    # === Crawling ===
    smu_notice_list_url: str = "https://www.smu.ac.kr/kor/life/notice.do?srCampus=smu"
    smu_notice_origin: str = "https://www.smu.ac.kr"
    crawl_run_on_startup: bool = False
    crawl_cron_utc_hour: int = 10  # KST 19:00 == UTC 10:00
    crawl_cron_utc_minute: int = 0
    crawl_max_pages: int = 10
    crawl_headless: bool = True

    @field_validator("frontend_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [origin.strip() for origin in value if isinstance(origin, str) and origin.strip()]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [origin.strip() for origin in parsed if isinstance(origin, str) and origin.strip()]
            except json.JSONDecodeError:
                pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return []


@lru_cache
def get_settings() -> Settings:
    return Settings()
