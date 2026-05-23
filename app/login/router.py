from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import create_access_token
from app.database import get_db
from app.login.service import (
    forget_session_stamp,
    get_current_user,
    login_with_smu,
    refresh_school_session_if_needed,
    to_user_response,
)
from app.models import User
from app.schemas import LoginRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    return await login_with_smu(db, payload.user_id.strip(), payload.password)


@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return to_user_response(user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> TokenResponse:
    await refresh_school_session_if_needed(db, user, force=True)
    return TokenResponse(access_token=create_access_token(str(user.id)), user=to_user_response(user))


@router.post("/logout")
def logout(user: User = Depends(get_current_user)) -> dict[str, str]:
    forget_session_stamp(user.id)
    return {"message": "logged out"}
