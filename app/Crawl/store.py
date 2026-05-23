"""크롤링 결과를 `aiot_notices` 테이블에 저장하는 헬퍼."""

from __future__ import annotations

import logging
from typing import Any

from app.database import SessionLocal
from app.models import AiotNotice

logger = logging.getLogger(__name__)


def save_notices(notices: list[dict[str, Any]]) -> int:
    if not notices:
        return 0

    session = SessionLocal()
    try:
        rows = [AiotNotice(json_data=notice) for notice in notices]
        session.add_all(rows)
        session.commit()
        return len(rows)
    except Exception:  # noqa: BLE001
        session.rollback()
        logger.exception("aiot_notices 저장 실패")
        return 0
    finally:
        session.close()
