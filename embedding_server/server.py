"""Qwen3-Embedding-4B 쿼리 임베딩 전용 미니 서버.

`smu-talk` Chroma 컬렉션에 들어있는 사전 임베딩 (Qwen3-Embedding-4B, 2560-dim) 과
**같은 모델/같은 옵션**으로 쿼리 텍스트만 임베딩하여 반환한다.

사용법:
    cd BackEnd-KMJ/embedding_server
    pip install -r requirements.txt
    # (옵션) 모델을 미리 다운로드 받아둔 경로:
    #   export MODEL_DIR=/abs/path/to/Qwen3-Embedding-4B
    uvicorn server:app --host 0.0.0.0 --port 8100

엔드포인트:
    POST /embed   {"text": "검색 질문"} | {"texts": ["...", "..."]}
        -> {"embedding": [..len=2560..]} 또는 {"embeddings": [[...], ...]}
    GET  /health  -> {"status": "ok", "dim": 2560, "model": "...", "device": "cuda|cpu"}

백엔드 (.env):
    EMBEDDING_PROVIDER=qwen_server
    EMBEDDING_SERVER_URL=http://<host>:8100
    CROMA_COLLECTION_NAME=smu-talk
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("qwen_embed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-Embedding-4B")
MODEL_DIR = os.environ.get("MODEL_DIR", "")
DEVICE = os.environ.get("DEVICE", "auto")  # auto | cuda | cpu
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))

_model: Any | None = None
_device: str = "cpu"


def _load_model() -> tuple[Any, str]:
    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    if DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE

    model_ref = MODEL_DIR or MODEL_ID
    model_kwargs: dict[str, Any] = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    logger.info("Qwen 임베딩 모델 로드: ref=%s device=%s", model_ref, device)
    model = SentenceTransformer(
        model_ref,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model, device


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _device
    _model, _device = _load_model()
    # 워밍업 1회 호출
    try:
        _model.encode(
            ["워밍업"],
            batch_size=1,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        logger.info("Qwen 임베딩 서버 준비 완료 (dim=%d)", _model.get_sentence_embedding_dimension())
    except Exception:  # noqa: BLE001
        logger.exception("워밍업 실패 (서비스는 계속 동작)")
    try:
        yield
    finally:
        pass


app = FastAPI(title="Qwen Embedding Server", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = Field(default=None)


@app.post("/embed")
def embed(req: EmbedRequest) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(503, "모델이 아직 로드되지 않았습니다.")

    if req.text is None and not req.texts:
        raise HTTPException(400, "text 또는 texts 중 하나는 필수입니다.")

    inputs = req.texts if req.texts else [req.text or ""]
    try:
        vecs = _model.encode(
            inputs,
            batch_size=min(len(inputs), 16),
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("임베딩 실패")
        raise HTTPException(500, f"임베딩 실패: {exc}") from exc

    if req.text is not None:
        return {"embedding": vecs[0].astype("float32").tolist()}
    return {"embeddings": [v.astype("float32").tolist() for v in vecs]}


@app.get("/health")
def health() -> dict[str, Any]:
    if _model is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "model": MODEL_DIR or MODEL_ID,
        "device": _device,
        "dim": _model.get_sentence_embedding_dimension(),
        "max_seq_length": _model.max_seq_length,
    }
