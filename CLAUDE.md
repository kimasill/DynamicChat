# DynamicChat — Claude Code Guide

## 프로젝트 개요

DynamicChat은 사용자가 프롬프트와 설정을 구성하면 AI가 장기 시뮬레이션을 진행하고,
필요한 설정만 RAG로 불러오며, 문맥상 필요한 순간 NovelAI 기반 이미지를 생성하는
시뮬레이션 챗 프로그램이다.

- 청사진: `docs/ai-simulation-chat-blueprint.md`
- API 경계 문서: `docs/dynamicchat-api-mvp.md`
- NeuralMap README: `S:\Project\NeuralMap\README.md`

## 서버 실행

```powershell
# API 서버만 (port 8788)
pnpm api

# API + Vite 동시 실행
pnpm dev

# Vite dev server만
pnpm dev:web
```

기본 URL: `http://127.0.0.1:8788`  
데이터 디렉토리: `.dynamicchat-data/` (또는 `DYNAMICCHAT_DATA_DIR` 환경변수)

## 핵심 아키텍처

```
DynamicChat Web (React/Vite)
  └─ DynamicChat API (server/dynamicchat-server.mjs)
       ├─ Simulation Runtime  ← src/services/simulationEngine.ts
       ├─ Prompt Module Mgr
       ├─ Image Orchestrator  ← src/services/imageOrchestrator.ts
       ├─ NeuralMap Client    ← src/services/neuralMapClient.ts
       └─ LLM Client          ← src/services/llmClient.ts
```

### 채팅 턴 흐름 (chat turn loop)

1. 사용자 메시지 수신
2. NeuralMap에 이전 이벤트 ingestion (보류 중인 것)
3. `simulation/context` 또는 `graph/query`로 Context Pack 조회
4. 프롬프트 모듈 + Context Pack 조립 (Prompt Assembler)
5. LLM 호출 → `assistant_text` + sidecar(`memory_events`, `image_cues`)
6. NeuralMap에 새 simulation event 저장
7. Image Orchestrator가 이미지 트리거 판단
8. (triger 수용 시) Image Prompt Planner → NovelAI → Asset Library 저장
9. 응답 + 이미지 업데이트 반환

## 런타임 에이전트 역할 (MVP: 단일 프로세스 순차 실행)

| 런타임 에이전트 | 파일 | 역할 |
|---|---|---|
| Narrative Agent | `simulationEngine.ts` | 채팅 응답과 시뮬레이션 진행 |
| Memory Curator | `memoryCompiler.ts` | 중요 이벤트 추출 → NeuralMap 적재 |
| Prompt Retrieval | `simulationEngine.ts` (모듈 선택 로직) | RAG 기반 프롬프트 모듈 선택 |
| Image Prompt Planner | `imageOrchestrator.ts` | 생성 여부 + NovelAI 프롬프트 작성 |
| Image Policy Validator | `imageOrchestrator.ts` (policy 검사) | 안전 등급, 사용자 규정 검증 |
| Continuity Validator | `simulationEngine.ts` (session reset) | 세션 초기화 후 기억 유지 검사 |

## Claude Code 개발 서브에이전트

개발 작업은 코드베이스 구역별 서브에이전트에 위임한다.

| 서브에이전트 | 담당 구역 | 주요 파일 |
|---|---|---|
| `server-api` | API 서버, 라우트, 퍼시스턴스 | `server/dynamicchat-server.mjs`, `server/migrations/` |
| `simulation-engine` | 턴 루프, 프롬프트 조립, 모듈 선택 | `simulationEngine.ts`, `llmClient.ts`, `memoryCompiler.ts` |
| `image-pipeline` | 이미지 생성, NovelAI 연동, 정책 검사 | `imageOrchestrator.ts`, `novelAiClient.ts`, `contentRating.ts` |
| `memory-neuralmap` | NeuralMap 연동, 세션 연속성 | `neuralMapClient.ts`, `memoryCompiler.ts` |
| `frontend` | React UI, API 클라이언트 | `App.tsx`, `dynamicChatApi.ts`, `src/styles.css` |
| `eval-quality` | 품질 평가 스크립트 | `server/*-eval.mjs` |

## 코드 규칙

- **타입**: `src/types.ts`에 집중. 새 도메인 타입은 여기에만 추가.
- **서비스**: `src/services/` — 각 파일이 단일 관심사를 담당.
- **서버 라우트**: `server/dynamicchat-server.mjs` — 라우트를 추가할 때 `matchRoute()` 배열에 등록.
- **ID 생성**: `createId()` (`src/lib/id.ts`) 사용. `crypto.randomUUID()` 직접 호출 금지.
- **환경 변수**: `DYNAMICCHAT_API_PORT`, `DYNAMICCHAT_DATA_DIR`, `DYNAMICCHAT_CORS_ORIGIN`, `DYNAMICCHAT_RATE_LIMIT_WINDOW_MS`, `DYNAMICCHAT_RATE_LIMIT_MAX`
- **LLM 백엔드**: 로컬 CLI 에이전트 브릿지 `POST /llm/cli-agent` 사용 가능 (API 키 불필요).

## 품질 평가

```powershell
pnpm eval:simulation      # 시뮬레이션 품질 평가
pnpm eval:image-prompts   # 이미지 프롬프트 품질 평가
pnpm eval:simulation-runs # 시뮬레이션 실행 품질 평가
pnpm eval:quality         # 전체 평가
```

## 이미지 생성 정책

- 이미지 프롬프트는 8개 레이어를 순서대로 합친다: 품질 → 화풍 → 작가 → 캐릭터 visual profile → 장면/배경 → 감정/의상 → 사용자 규정 → negative
- 각 캐릭터는 독립된 `char_caption`을 가진다 (NAI multi-char prompting).
- 외형/의상 정보는 LLM이 추론하지 않고 저장된 `CharacterVisualProfile` + 현재 `Wearing` 상태에서 직접 주입한다.

## 세션 초기화와 연속성

세션 초기화는 "대화 리셋"이지 "시뮬레이션 리셋"이 아니다.
- 초기화 전 Handoff Pack 생성 → 새 세션에서 Context Pack으로 복원
- `docs/continuity-reset-checklist.md` 참고
