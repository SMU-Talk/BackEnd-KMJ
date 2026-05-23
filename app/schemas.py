from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: int
    nickname: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ChatFilters(BaseModel):
    tags: list[str] = Field(default_factory=list)
    dept: str | None = None
    major: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    filters: ChatFilters = Field(default_factory=ChatFilters)


class Notice(BaseModel):
    title: str
    tag: str
    dept: str
    date: str
    body: str


class ChatResponse(BaseModel):
    message: str
    notices: list[Notice] = Field(default_factory=list)
