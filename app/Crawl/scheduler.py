"""APScheduler를 이용한 SMU 공지 크롤링 스케줄러.

UTC 10:00 == KST 19:00 에 매일 실행된다.
"""

from __future__ import annotations

import logging
from datetime import timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.Crawl.crawler import run_crawl_job
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def start_scheduler() -> AsyncIOScheduler:
    """FastAPI 라이프스팬 시작 시 호출. 이미 떠 있으면 그대로 반환."""
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        return _scheduler

    settings = get_settings()
    scheduler = AsyncIOScheduler(timezone=timezone.utc)
    scheduler.add_job(
        run_crawl_job,
        trigger=CronTrigger(
            hour=settings.crawl_cron_utc_hour,
            minute=settings.crawl_cron_utc_minute,
            timezone=timezone.utc,
        ),
        id="smu_notice_crawl",
        name="SMU notice crawl (UTC 10:00 / KST 19:00)",
        replace_existing=True,
        misfire_grace_time=60 * 30,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "크롤링 스케줄러 시작: UTC %02d:%02d (KST %02d:%02d)",
        settings.crawl_cron_utc_hour,
        settings.crawl_cron_utc_minute,
        (settings.crawl_cron_utc_hour + 9) % 24,
        settings.crawl_cron_utc_minute,
    )
    return scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("크롤링 스케줄러 종료")
    _scheduler = None
