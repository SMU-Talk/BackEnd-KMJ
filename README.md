# SMU Talk Backend

FastAPI 기반 백엔드입니다. 로그인 관련 코드는 `app/login`에 모아두었고, 학교 SSO 세션은 사용 중 자동으로 재발급합니다.

## 실행

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

프론트엔드는 `FrontEnd-KMJ/.env`에 아래 값을 둡니다.

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## DB

`create_tables.sql`을 MySQL에 적용하세요. 기존 설계에 사용자 학교 ID 컬럼이 없어서 현재는 `nickname`에 로그인 ID를 저장합니다. `password` 컬럼에는 평문이 아니라 `JWT_SECRET_KEY`로 암호화된 SSO 비밀번호가 저장됩니다. 학교 세션을 30분 이상 자동 갱신하려면 서버가 SSO 재로그인을 할 수 있어야 하기 때문입니다.

운영 환경에서는 반드시 `.env`의 `JWT_SECRET_KEY`를 충분히 긴 랜덤 문자열로 바꾸고, 이미 저장된 계정이 있다면 키 변경 시 재로그인이 필요합니다.

## API

- `POST /api/auth/login`: 학교 SSO 로그인 후 앱 토큰 발급
- `GET /api/auth/me`: 현재 앱 로그인 확인
- `POST /api/auth/refresh`: 학교 세션 만료 임박 시 재로그인 및 앱 토큰 재발급
- `POST /api/auth/logout`: 클라이언트 로그아웃
- `POST /api/chat`: 공지 질문 처리. 현재는 RAG 연결 전 임시 공지 데이터와 필터 로직을 사용합니다.

## RAG 프롬프트

향후 RAG 연결 시 사용할 시스템 프롬프트는 `app/core/prompt.py`에 분리되어 있습니다. 프론트에서 선택한 `dept`, `major`, `tags`를 최우선 필터로 적용하도록 작성되어 있습니다.
