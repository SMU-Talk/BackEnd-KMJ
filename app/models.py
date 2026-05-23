from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="Auto Increase")
    password: Mapped[str] = mapped_column(String(255), nullable=False, comment="암호화된 SSO 비밀번호")
    nickname: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True, comment="닉네임")
    eXSignOnSessionID: Mapped[str] = mapped_column("eXSignOnSessionID", String(255), nullable=False, comment="세션ID")


class AiotNotice(Base):
    __tablename__ = "aiot_notices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    json_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
