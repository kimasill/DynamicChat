---
description: DynamicChat 개발 서버를 시작한다. 인자: api | web | all (기본값: all)
---

DynamicChat 개발 서버를 시작한다.

인자: $ARGUMENTS (기본값: all)

다음과 같이 처리한다:

- `api`: API 서버만 시작 (`pnpm api`, port 8788)
- `web`: Vite dev server만 시작 (`pnpm dev:web`)
- `all` 또는 인자 없음: `pnpm dev` (API + Vite 동시 실행)

시작 후 확인:
1. `http://127.0.0.1:8788/health` 응답이 200인지 체크 (api 모드)
2. 데이터 디렉토리 경로 출력 (`.dynamicchat-data/` 또는 `DYNAMICCHAT_DATA_DIR`)
3. 현재 저장된 시뮬레이션 수 (`GET /simulations`)

환경 변수가 필요하면:
- `DYNAMICCHAT_API_PORT`: 기본 8788
- `DYNAMICCHAT_DATA_DIR`: 기본 `.dynamicchat-data`
- `DYNAMICCHAT_CORS_ORIGIN`: 기본 `*`
