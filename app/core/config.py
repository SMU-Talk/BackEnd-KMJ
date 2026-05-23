import json
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_DIR = REPO_ROOT / "smu_notice_qwen3_e1_release"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

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

    # NOTE: env keys use `CROMA_*` per project convention. Official `CHROMA_*` names are accepted too.
    croma_api_key: str = Field("", validation_alias=AliasChoices("CROMA_API_KEY", "CHROMA_API_KEY"))
    croma_tenant_id: str = Field("", validation_alias=AliasChoices("CROMA_TENANT_ID", "CHROMA_TENANT"))
    croma_database_name: str = Field(
        "",
        validation_alias=AliasChoices("CROMA_DATABASE_NAME", "CROMA_DATABASE", "CHROMA_DATABASE"),
    )
    croma_collection_name: str = Field(
        "smu_notices",
        validation_alias=AliasChoices("CROMA_COLLECTION_NAME", "CHROMA_COLLECTION_NAME"),
    )
    chroma_top_k: int = 5

    embedding_dim: int = 2560
    embeddings_path: str = str(DEFAULT_RELEASE_DIR / "embeddings.npy")
    chunks_path: str = str(DEFAULT_RELEASE_DIR / "chunks.jsonl")
    embedding_server_url: str = ""  # validation_alias: EMBEDDING_SERVER_URL if set in env

    # === Embedding provider ===
    # "openai" | "qwen_server" | "placeholder"
    # - openai:     OpenAI Embeddings API 사용 (text-embedding-3-small). 별도 컬렉션 사용.
    # - qwen_server: embedding_server_url 의 Qwen3-Embedding-4B 서버 사용 (2560-dim). 기존 ingest 와 호환.
    # - placeholder: 결정적 placeholder (디버그용, RAG 품질 보장 안 됨).
    embedding_provider: str = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    openai_collection_name: str = Field(
        "smu_notices_openai",
        validation_alias=AliasChoices("OPENAI_COLLECTION_NAME", "CROMA_OPENAI_COLLECTION_NAME"),
    )
    # 서버 기동 시 OpenAI 컬렉션이 비어 있으면 자동 ingest. 비용/시간 이슈로 기본 False.
    auto_ingest_openai_on_startup: bool = False
    # 자동 ingest 시 최대 처리 청크 수 (None=전체). chunks.jsonl 12.3만건 전체는 비용/시간 큼.
    auto_ingest_openai_limit: int | None = 5000

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
