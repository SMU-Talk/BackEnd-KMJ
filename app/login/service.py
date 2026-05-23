from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import create_access_token, decode_access_token, decrypt_secret, encrypt_secret
from app.database import get_db
from app.login.smu_sso import SmuLoginError, SmuSsoClient
from app.models import User
from app.schemas import TokenResponse, UserResponse

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
_session_refreshed_at: dict[int, datetime] = {}


def forget_session_stamp(user_id: int) -> None:
    _session_refreshed_at.pop(user_id, None)


def to_user_response(user: User) -> UserResponse:
    return UserResponse(id=user.id, nickname=user.nickname)


async def login_with_smu(db: Session, user_id: str, password: str) -> TokenResponse:
    try:
        smu_session = await SmuSsoClient().login(user_id=user_id, password=password)
    except SmuLoginError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    user = db.scalar(select(User).where(User.nickname == user_id))

    encrypted_password = encrypt_secret(password)
    if user:
        user.password = encrypted_password
        user.eXSignOnSessionID = smu_session.session_id
    else:
        user = User(password=encrypted_password, nickname=user_id, eXSignOnSessionID=smu_session.session_id)
        db.add(user)

    db.commit()
    db.refresh(user)
    _session_refreshed_at[user.id] = datetime.now(timezone.utc)

    return TokenResponse(access_token=create_access_token(str(user.id)), user=to_user_response(user))


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="로그인이 필요합니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id = int(payload["sub"])
    except (ValueError, TypeError):
        raise credentials_error

    user = db.get(User, user_id)
    if not user:
        raise credentials_error
    return user


async def require_school_session(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> User:
    await refresh_school_session_if_needed(db, user)
    return user


async def refresh_school_session_if_needed(db: Session, user: User, force: bool = False) -> None:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    refreshed_at = _session_refreshed_at.get(user.id)
    refresh_after = timedelta(minutes=settings.smu_session_ttl_minutes - settings.smu_session_refresh_window_minutes)

    if not force and refreshed_at and now - refreshed_at < refresh_after:
        return

    try:
        password = decrypt_secret(user.password)
        smu_session = await SmuSsoClient().login(user_id=user.nickname, password=password)
    except (ValueError, SmuLoginError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="학교 세션이 만료되었습니다. 다시 로그인해주세요.",
        )

    user.eXSignOnSessionID = smu_session.session_id
    db.commit()
    db.refresh(user)
    _session_refreshed_at[user.id] = now
