import re
from dataclasses import dataclass
from html import unescape

import httpx

from app.core.config import get_settings


class SmuLoginError(RuntimeError):
    pass


@dataclass(frozen=True)
class SmuSession:
    session_id: str


class SmuSsoClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def login(self, user_id: str, password: str) -> SmuSession:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        try:
            async with httpx.AsyncClient(
                headers=headers,
                follow_redirects=True,
                timeout=self.settings.http_timeout_seconds,
            ) as client:
                auth_response = await client.get(self.settings.smu_auth_url)
                auth_response.raise_for_status()

                login_response = await client.post(
                    self.settings.smu_login_url,
                    data=self._login_form(auth_response.text, user_id, password),
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Origin": "https://smsso.smu.ac.kr",
                        "Referer": self.settings.smu_auth_url,
                    },
                )
                login_response.raise_for_status()

                session_id = self._find_session_id(client, login_response)
                if not session_id:
                    raise SmuLoginError("학교 SSO 로그인에 실패했습니다. 아이디와 비밀번호를 확인해주세요.")

                return SmuSession(session_id=session_id)
        except httpx.HTTPError as exc:
            raise SmuLoginError("학교 SSO 서버와 통신할 수 없습니다. 잠시 후 다시 시도해주세요.") from exc

    def _login_form(self, html: str, user_id: str, password: str) -> dict[str, str]:
        return {
            "l_token": self._extract_l_token(html),
            "user_timezone_offset": "-540",
            "pwdPolicy": "N",
            "user_code": "",
            "sid": "portal",
            "user_id": user_id,
            "user_password": password,
            "saveIdIpt": "on",
            "user_id_auth": "",
            "motp_rdo": "",
        }

    @staticmethod
    def _extract_l_token(html: str) -> str:
        patterns = [
            r'name=["\']l_token["\'][^>]*value=["\']([^"\']+)["\']',
            r'value=["\']([^"\']+)["\'][^>]*name=["\']l_token["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.IGNORECASE)
            if match:
                return unescape(match.group(1))
        return ""

    @staticmethod
    def _find_session_id(client: httpx.AsyncClient, response: httpx.Response) -> str | None:
        for cookie in client.cookies.jar:
            if cookie.name == "eXSignOnSessionID":
                return cookie.value

        candidates = [response.headers.get("location", ""), response.text]
        candidates.extend(item.headers.get("location", "") for item in response.history)
        candidates.extend(item.text for item in response.history if item.text)

        for value in candidates:
            match = re.search(r"eXSignOnSessionID[=:\"']+([A-Za-z0-9._%-]+)", value)
            if match:
                return match.group(1)
        return None
