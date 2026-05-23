"""
SMU 메인 공지(srCampus=smu) Playwright 크롤러.

한국 시간 기준 '오늘' 자정(00:00)부터 19:00 사이에 등록된 공지를 수집하여
`aiot_notices.json_data` 컬럼에 JSON 한 건씩 저장한다. (스케줄은 UTC 10:00 == KST 19:00)

공지 게시판은 날짜(yyyy-MM-dd) 단위로만 노출되므로 '오늘 날짜와 일치하는 항목'을
모두 수집하고, 시(時)/분(分) 정보가 detail 페이지에 있으면 추가 메타로 함께 저장한다.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.Crawl.store import save_notices
from app.core.config import get_settings

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


def _today_kst() -> datetime:
    return datetime.now(tz=KST)


def _today_kst_str() -> str:
    return _today_kst().strftime("%Y-%m-%d")


def _normalize_date(text: str) -> str | None:
    """공지 목록의 다양한 날짜 표기를 yyyy-MM-dd 로 정규화한다."""
    if not text:
        return None
    text = text.strip()
    m = re.search(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", text)
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _build_absolute(url: str) -> str:
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    settings = get_settings()
    if url.startswith("/"):
        return f"{settings.smu_notice_origin}{url}"
    return f"{settings.smu_notice_origin}/{url}"


async def _extract_list_rows(page) -> list[dict[str, Any]]:
    """공지 리스트의 한 페이지에서 각 행을 dict 로 추출한다.

    SMU 게시판은 `<table>` 기반이며, 보통 다음 패턴 중 하나를 따른다:
      - `table tbody tr` 의 각 컬럼이 [번호, 제목, 작성자, 작성일, 조회, 첨부]
      - 또는 `.board-list` 클래스의 li 리스트

    detail 링크는 `articleNo` 쿼리 파라미터로 식별한다.
    """
    return await page.evaluate(
        """
        () => {
          const isPinned = (row) => {
            const txt = (row.querySelector("td:first-child")?.innerText || "").trim();
            return txt === "공지" || txt === "Notice" || txt === "";
          };

          const pickDate = (row) => {
            // 셀 텍스트 후보 모두 검사하여 yyyy.mm.dd / yyyy-mm-dd / yyyy/mm/dd 매칭
            const candidates = Array.from(row.querySelectorAll("td,span,div"))
              .map((el) => (el.innerText || "").trim());
            for (const t of candidates) {
              const m = t.match(/(\\d{4})[\\.\\-\\/](\\d{1,2})[\\.\\-\\/](\\d{1,2})/);
              if (m) return `${m[1]}-${String(m[2]).padStart(2,"0")}-${String(m[3]).padStart(2,"0")}`;
            }
            return null;
          };

          const rows = Array.from(document.querySelectorAll("table tbody tr"));
          const out = [];
          for (const row of rows) {
            const anchor = row.querySelector("a[href*='articleNo']") || row.querySelector("a[href*='mode=view']") || row.querySelector("a");
            if (!anchor) continue;
            const title = (anchor.innerText || anchor.textContent || "").replace(/\\s+/g, " ").trim();
            const href = anchor.getAttribute("href") || "";
            if (!title || !href) continue;

            let articleNo = "";
            const m = href.match(/articleNo=(\\d+)/);
            if (m) articleNo = m[1];

            const date = pickDate(row);
            const writer = (row.querySelector("td:nth-of-type(3)")?.innerText || "").trim();
            const views = parseInt((row.querySelector("td:nth-last-of-type(2)")?.innerText || "0").replace(/[^0-9]/g, ""), 10) || 0;

            out.push({
              pinned: isPinned(row),
              title,
              link: href,
              article_no: articleNo,
              date,
              writer,
              views,
            });
          }
          return out;
        }
        """
    )


async def _extract_detail(page) -> dict[str, Any]:
    """공지 상세 페이지에서 본문/첨부/메타 정보를 추출한다."""
    return await page.evaluate(
        """
        () => {
          const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
          const root = document.querySelector(".board-view, .board_view, .view-content, .view_content, #content") || document.body;
          const title = norm(document.querySelector(".board-view .title, .view-title, .title")?.innerText) || norm(document.title);

          const bodyEl = document.querySelector(".board-view .view-cont, .view-cont, .view_content .cont, .board-view .cont, .view-body");
          const body = norm(bodyEl?.innerText || root.innerText);

          const attachments = Array.from(document.querySelectorAll(".board-view a[href*='download'], .view-file a, .file-list a, a[href*='/cmm/fms/']"))
            .map((a) => ({ name: norm(a.innerText), href: a.getAttribute('href') || '' }))
            .filter((x) => x.name && x.href);

          let date = null;
          const dateMatch = (root.innerText || "").match(/(\\d{4})[\\.\\-\\/](\\d{1,2})[\\.\\-\\/](\\d{1,2})(?:\\s+(\\d{1,2}):(\\d{2}))?/);
          if (dateMatch) {
            const [, y, mo, d, h, mi] = dateMatch;
            date = `${y}-${String(mo).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
            if (h && mi) date += `T${String(h).padStart(2,'0')}:${String(mi).padStart(2,'0')}:00+09:00`;
          }

          return { title, body, attachments, date_detail: date };
        }
        """
    )


async def crawl_today_notices() -> list[dict[str, Any]]:
    """오늘(KST)자 공지 메타데이터 리스트를 반환한다.

    Playwright 패키지가 설치되어 있지 않거나 브라우저 바이너리가 없으면 빈 리스트를 반환하고
    경고만 남긴다 (스케줄러는 다음 실행에서 재시도).
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError:
        logger.warning("playwright 가 설치되지 않아 크롤링을 건너뜁니다. `pip install playwright && playwright install chromium` 필요.")
        return []

    settings = get_settings()
    today = _today_kst_str()
    collected: list[dict[str, Any]] = []
    seen_article_nos: set[str] = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=settings.crawl_headless)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            locale="ko-KR",
        )
        page = await context.new_page()

        try:
            for page_no in range(1, settings.crawl_max_pages + 1):
                list_url = settings.smu_notice_list_url
                if page_no > 1:
                    sep = "&" if "?" in list_url else "?"
                    list_url = f"{list_url}{sep}article.offset={(page_no - 1) * 10}&articleLimit=10"

                logger.info("[crawl] 목록 로드 page=%s url=%s", page_no, list_url)
                try:
                    await page.goto(list_url, wait_until="domcontentloaded", timeout=20000)
                    await page.wait_for_selector("table tbody tr", timeout=10000)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[crawl] 목록 로드 실패 page=%s: %s", page_no, exc)
                    break

                rows = await _extract_list_rows(page)
                if not rows:
                    logger.info("[crawl] 목록이 비어 있어 종료 page=%s", page_no)
                    break

                page_dates = [r.get("date") for r in rows if not r.get("pinned")]
                logger.info("[crawl] page=%s rows=%d sample_dates=%s", page_no, len(rows), page_dates[:5])

                today_rows = [
                    r for r in rows
                    if not r.get("pinned")
                    and r.get("date") == today
                    and r.get("article_no")
                    and r["article_no"] not in seen_article_nos
                ]

                for row in today_rows:
                    article_no = row["article_no"]
                    seen_article_nos.add(article_no)
                    detail_url = _build_absolute(row["link"])
                    logger.info("[crawl] 상세 수집 articleNo=%s url=%s", article_no, detail_url)
                    try:
                        await page.goto(detail_url, wait_until="domcontentloaded", timeout=20000)
                        await page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[crawl] 상세 로드 실패 articleNo=%s: %s", article_no, exc)
                        continue

                    detail = await _extract_detail(page)
                    record = {
                        "source": "smu_main_notice",
                        "source_url": settings.smu_notice_list_url,
                        "article_no": article_no,
                        "title": detail.get("title") or row.get("title"),
                        "writer": row.get("writer"),
                        "views": row.get("views"),
                        "notice_date": today,
                        "notice_datetime_kst": detail.get("date_detail"),
                        "url": detail_url,
                        "body": detail.get("body", ""),
                        "attachments": detail.get("attachments", []),
                        "crawled_at_kst": _today_kst().isoformat(),
                    }
                    collected.append(record)

                # 오늘이 아닌 일자 행이 등장하면(=목록은 최신순), 더 뒤 페이지는 더 옛날이므로 종료
                non_today_dates = [r.get("date") for r in rows if not r.get("pinned") and r.get("date")]
                if non_today_dates and all(d < today for d in non_today_dates):
                    logger.info("[crawl] 오늘 자 공지 모두 수집, 페이지 순회 종료")
                    break
        finally:
            await context.close()
            await browser.close()

    logger.info("[crawl] 총 %d건의 오늘(%s) 공지를 수집했습니다.", len(collected), today)
    return collected


async def run_crawl_job() -> int:
    """스케줄러에서 호출하는 진입점. 수집 후 DB 저장까지 처리하고 저장 건수를 리턴한다."""
    try:
        notices = await crawl_today_notices()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[crawl] 크롤 작업 실패: %s", exc)
        return 0

    if not notices:
        return 0

    saved = save_notices(notices)
    logger.info("[crawl] DB 저장 완료: %d 건", saved)
    return saved


def run_crawl_job_sync() -> int:
    """CLI/테스트용 동기 실행 래퍼."""
    return asyncio.run(run_crawl_job())
