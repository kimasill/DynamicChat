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

> 턴은 LLM을 **두 번** 호출한다. 서사 패스는 프롬프트에 태그 어휘가 전혀 없고,
> 주석 패스(`requestTurnAnnotations`)가 이미 작성된 서사를 받아 `image_cues`와
> `state_events`를 한 번에 만든다. 주석 패스에는 **이번 턴의 메시지가 포함된 스냅샷**을
> 넘겨야 한다 — 장면 캐스트 추론과 태그 프리셋 매칭이 `state.messages`에서 파생되므로,
> 턴 이전 상태를 넘기면 모든 파생 블록이 직전 턴을 설명하게 된다.

1. 사용자 메시지 수신
2. NeuralMap에 이전 이벤트 ingestion (보류 중인 것)
3. `simulation/context` 또는 `graph/query`로 Context Pack 조회
4. 프롬프트 모듈 + Context Pack 조립 (Prompt Assembler)
5. LLM 호출 → `assistant_text` (서사 전용, 태그 어휘 없음)
5b. 주석 패스(`requestTurnAnnotations`) → `image_cues` + `state_events`
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
- **환경 변수**: `DYNAMICCHAT_API_PORT`, `DYNAMICCHAT_DATA_DIR`, `DYNAMICCHAT_CORS_ORIGIN`, `DYNAMICCHAT_RATE_LIMIT_WINDOW_MS`, `DYNAMICCHAT_RATE_LIMIT_MAX`, `DYNAMICCHAT_MAX_BODY_BYTES`
  (요청 본문 상한, 기본 64MiB — 이미지 data URL 이 이 경로로 들어오므로 넉넉하게 잡혀 있다. 초과하면 413)
- **LLM 백엔드**: 프로바이더 프리셋은 `src/services/llmProviders.ts` 한 곳에 있다. UI 메타데이터
  (라벨/base URL/모델 목록)와 런타임 능력(전송 형식, JSON 모드, temperature 상한, 추가 헤더/바디,
  프록시 필요 여부)을 같이 들고 있으므로 백엔드 추가는 배열 항목 하나면 된다.
- **오픈 모델**: DeepSeek, Kimi(Moonshot), Qwen(DashScope), GLM(Z.ai), OpenRouter, Groq, Together +
  로컬(Ollama / LM Studio / OpenAI 호환). 대부분 브라우저 origin에 CORS를 열어주지 않으므로
  `requiresProxy: true` 프리셋은 `POST /llm/chat`(서버 릴레이, 호스트 allow-list)을 경유한다.
  allow-list 추가는 `DYNAMICCHAT_LLM_PROXY_HOSTS` 환경변수.
- **구독 CLI 브릿지**: `POST /llm/cli-agent` 사용 가능 (API 키 불필요).

## 품질 평가

```powershell
pnpm eval:simulation      # 시뮬레이션 품질 평가
pnpm eval:image-prompts   # 이미지 프롬프트 품질 평가 (프레임 일관성 포함)
pnpm eval:simulation-runs # 시뮬레이션 실행 품질 평가
pnpm eval:oss-models      # 오픈 모델 연동 검증 (키 불필요, 모의 업스트림)
pnpm eval:quality         # 전체 평가
```

### 오픈 모델 검증

`eval:oss-models`는 각 공급자의 실제 특성(DeepSeek thinking 토글, Qwen enable_thinking,
OpenRouter 헤더, reasoning-only 응답, HTTP-200 error 봉투, response_format 거부, 로컬 모델의
`<think>` 블록)을 흉내내는 모의 업스트림에 대고 **실제로 무엇을 전송하고 어떻게 해석하는지**를
검증한다. API 키가 없어도 어디서나 돌아간다. OpenAI 호환 서버가 떠 있으면 실제 모델로 턴 전체를
돌리는 구간도 함께 실행된다.

`eval:oss-suitability`는 합격/불합격이 아니라 **비율**을 측정한다 — N턴을 돌려 서사 품질, cue
생성률, 프레임 선언율, 태그 완결성, 상태 추출률을 뽑는다. 모델 출력은 편차가 있으므로 한 번의
샘플로는 그 백엔드로 장기 시뮬레이션을 돌릴 수 있는지 판단할 수 없다.

```powershell
$env:OSS_EVAL_BASE_URL="http://127.0.0.1:11434/v1"
$env:OSS_EVAL_MODEL="qwen3-4b-16k:latest"
$env:OSS_EVAL_TURNS="6"
pnpm eval:oss-suitability
```

### 로컬 실행 (키 없음)

권장 경로다. `ollama serve` 만 띄우면 된다 — API 키도, 서버 프록시도 필요 없다.

1. LLM 설정에서 공급자를 **Ollama (로컬)** 로 선택
2. **로컬 서버에서 불러오기** 로 설치된 모델 목록을 가져와 선택
3. 필요하면 **컨텍스트 (토큰)** 를 조정 (기본 16384)

`src/services/promptBudget.ts` 가 프롬프트를 그 컨텍스트에 맞춰 만든다. 주석(이미지 태그) 프롬프트는
턴 전체에서 가장 큰 호출이라 그냥 두면 약 10k 토큰이고, 로컬 모델은 보통 그보다 창이 작다. 그래서
**디테일 티어**(full / compact / minimal)를 만들어 넣고, 만든 뒤 실제로 재서 안 맞으면 한 단계 낮춘다
— 임계값을 추측하지 않는다. 축약 티어에서도 구조 규칙은 전부 남고 근거 설명과 참조 블록만 빠진다.

| 컨텍스트 | 티어 | 주석 프롬프트 |
|---|---|---|
| 128k+ | full | ~23.6k자 |
| 8~16k | compact | ~14.1k자 |
| 4k | minimal | ~10.1k자 |

**Ollama 주의:** OpenAI 호환 엔드포인트(`/v1/chat/completions`)는 `num_ctx` 를 **조용히 무시하고**
모든 모델을 4096으로 서빙한다(실측 확인). 그래서 Ollama 프리셋은 `transport: "ollama"` 로 네이티브
`/api/chat` 을 쓰고 `options.num_ctx` 를 직접 넘긴다. LM Studio / llama.cpp / vLLM 는 OpenAI 경로를
그대로 쓰되, 서버에서 로드한 컨텍스트를 설정 화면의 값과 맞춰야 한다.

출력 토큰 상한도 `requestProviderTextImmediate` 에서 남은 컨텍스트에 맞춰 잘린다. 이 클램프는 서사
패스에도 적용되므로 두 호출 모두 창을 넘지 않는다. 실측:

| 컨텍스트 | 서사 max_tokens | 주석 티어 |
|---|---|---|
| 8k | 2341 (축소됨) | minimal |
| 16k 이상 | 4840 (전체) | full |

**추론 모델 주의:** Qwen3 / DeepSeek-R1 distill / GLM-Z1 계열은 Ollama에서 **기본으로 사고**하고,
그 결과를 별도 `thinking` 필드에 담아 `content` 를 **빈 문자열로** 돌려준다(qwen3:14b 실측: 200토큰
전부 thinking, 본문 0자). Ollama transport 는 `think: false` 를 보내 이를 끈다. 이 앱은 호출당
완성된 JSON 하나를 원하지 사고 과정을 원하지 않는다.

### 로컬 모델 비교

```powershell
$env:OSS_COMPARE_MODELS="qwen3:4b-instruct,qwen3:14b"
pnpm eval:oss-compare
```

구조 준수율은 작은 모델도 쉽게 통과하므로, 먼저 무너지는 **산문 품질**을 함께 잰다 — 반복(4-gram),
어휘 다양성, 대사 비율, 문장 길이 분산. RTX 4080 SUPER(16GB) 실측:

| 모델 | 지연 | 분량 | 반복↓ | 어휘↑ | 대사 |
|---|---|---|---|---|---|
| qwen3:4b-instruct | 12.4s | 639자 | 3.4% | 79% | 19% |
| qwen3:14b | 22.0s | 224자 | 0.0% | 90% | 31% |

14B 는 반복이 사라지고 어휘/대사 비율이 오르지만 턴이 짧아진다. 4B 는 길게 쓰지만 같은 구절을
재활용한다. 구조 지표(캐스팅·외형주입)는 양쪽 100%.

### VRAM 이 전부다 — 모델 선택 기준

로컬에서 유일하게 중요한 기준은 **모델이 VRAM 에 통째로 들어가느냐**다. 16GB 실측:

| 모델 | 크기 | VRAM 적재 | 속도 |
|---|---|---|---|
| qwen3:14b | 11.7GB | **100%** | **55.2 tok/s** |
| qwen3:30b-a3b | 20.4GB | 14.7GB (5.7GB 유출) | **0.09 tok/s** |

**600배 차이다.** MoE(30b-a3b, 활성 3B)가 부분 유출에 강할 것이라는 추정은 **틀렸다** — 오히려
밀집 모델보다 나쁘다. 밀집 모델은 레이어를 순차적으로 읽어 접근이 예측 가능하지만, MoE 는 토큰마다
다른 전문가로 라우팅되므로 유출된 전문가를 PCIe 로 계속, 불규칙하게 가져와야 한다. 활용할 지역성이
없다.

따라서 **총 파라미터가 아니라 양자화 후 크기로 고르고, KV 캐시 몫(16k 컨텍스트 기준 1~2GB)을 남겨야
한다.** 16GB 에서는 14B 급이 상한이다.

또 `qwen3:30b-a3b` 는 `think: false` 를 보내도 사고 과정을 **본문에** 영어로 쏟아냈다. 추론 태그
모델을 쓸 때는 `-instruct` 계열을 먼저 확인할 것.

**컨텍스트도 VRAM 을 먹는다.** 같은 qwen3:14b 라도 16k 는 11.7GB / 55 tok/s, 32k 는 14.4GB 로
카드가 포화(잔여 41MiB)되어 Windows 가 조용히 페이징하면서 **0.1 tok/s** 로 무너졌다. Ollama 의
`size_vram` 은 여전히 "100% GPU" 라고 보고하므로 이 수치만 믿으면 안 된다.

### 실제 시뮬레이션(여성의 삶) 실측 — 16GB / qwen3:14b / 16k

시드 데모는 부하를 과소평가한다. 실제 시뮬레이션은 캐릭터 8명, 활성 모듈 39개(본문 17.4k자),
paragraph cadence, adult_19 이다.

| 항목 | qwen3:14b 단일 | 14b 서사 + 4b-instruct 태그 |
|---|---|---|
| 턴 성공 / fallback 없음 | 100% | 100% |
| 한국어 / 스캐폴딩 누출 없음 | 100% | 100% |
| **턴당 cue** | **7.6** | 5.4 |
| cue frame 명시 · 태그 완결 · 인물 캡션 | 100% | 100% |
| 최종 프롬프트 구도 태그 · 피사체 수 · 외형 주입 | 100% | 100% |
| 상태 이벤트 추출 | 100% | **40%** |
| 평균 서사 / 지연 | 605자 / 57.7s | 854자 / 59.5s |

**16GB 에서는 모델을 분리하지 마라.** 태그 모델을 따로 두는 기능은 있고 정상 동작하지만(설정 화면에
공급자·모델·컨텍스트·로컬 목록 불러오기가 전부 있다), 이 하드웨어에서는 손해다 — 두 모델이 VRAM 에
동시에 안 들어가 Ollama 가 매 턴 교체하고, 주석 패스는 image_cues 뿐 아니라 **state_events 도 함께
쓰기 때문에** 작은 모델로 내리면 상태 추출이 100% → 40% 로 무너진다. 분리는 태그 모델을 값싼
호스티드 모델로 돌릴 때 의미가 있다.

### 하네스 결함 두 개가 이전 측정을 무효화했다

둘 다 실제 코드가 아니라 eval 쪽 문제였고, 둘 다 "모델 성능 부족" 처럼 보였다.

1. **cadence 덮어쓰기**: eval 이 시드의 `paragraph` 를 `balanced` 로 바꿔 실행했다. balanced 에서 1컷은
   정답이므로 "턴당 1.0 cue" 는 위반이 아니었다.
2. **세계관 밖 프롬프트**: 여성의 삶 시드에 데모의 도서관 프롬프트("서가 사이를 걸으며…")를 넣었다.
   장면 캐스트 가드가 옳게 아무도 무대에 올리지 않았고, 그 결과가 인물 캡션 48% / 외형 주입 0% 로
   나와 태그 파이프라인 결함처럼 보였다. 세계관에 맞는 입력으로 바꾸자 둘 다 100% 로 돌아왔다.

시드마다 프롬프트 세트를 따로 둔다(`PROMPTS_BY_SEED`). 두 세트 모두 같은 구도 사다리
(접근 → 위로 손뻗기 → 시선 돌리기 → 뒤돌기 → 웅크리기 → 클로즈업)를 밟아 프레임 지표가 비교 가능하다.

### cue 개수는 코드가 강제한다 (모델 지시 이행에 맡기지 않는다)

작성 프롬프트는 cadence 를 **상한**("up to 8 cues")으로만 말한다. 1컷만 낸 모델은 그 규칙을 어긴 게
아니라서 아무도 눈치채지 못했다. 호스티드 모델은 "high-density cut list" 라는 맥락에서 의도를 읽지만
로컬 오픈웨이트 모델은 상한을 문자 그대로 받는다.

게다가 개수 강제 로직은 **구형 경로에만 남아 있었다.** 내러티브가 태그를 쓰던 시절의
`shouldRetryIncompleteAssistantSidecar` + 확장 재요청이 그것인데, 태그 작성을 별도 주석 패스로 옮길 때
`imageCuesOwnedElsewhere` 가 그 경로 전체를 건너뛰게 되면서 **함께 옮겨지지 않았다.**

`topUpImageCues`(`llmClient.ts`)가 새 경로에 그것을 복원한다. 첫 패스가 cadence 하한에 못 미치면 이미
쓴 anchor 목록을 넘기고 부족분만 요청하는 짧은 후속 호출을 최대 2회 돌린다. 컨텍스트 문자열을 첫
패스와 **동일하게** 재사용하므로 KV 캐시가 그대로 먹힌다.

| cadence | 하한 |
|---|---|
| image_progression | 10 (정확히) |
| paragraph | 출력 예산에 따라 6~8 |
| rich | 2 |
| balanced / sparse | 없음 — 1컷이 정답이라 강제하면 원치 않는 이미지가 생긴다 |

**판단과 실패를 구분한다.** cue 를 내고 `should_generate=false` + 억제 사유를 붙인 패스는 이 턴이
비시각적이라고 *판단*한 것이므로 존중한다. 반면 **배열이 통째로 비어 있는데** cadence 는 컷 목록을
요구하는 경우는 실패다 — 이것이 "이미지가 가끔 안 생성됨" 의 정체다. 약한 태그 모델에서 충분히 긴
산문을 두고도 5턴 중 3턴이 빈 배열을 냈고, 이 재시도로 0턴이 되었다(턴당 cue 2.2 → 5.4).

서사가 240자 미만이면 top-up 하지 않는다 — 없는 비트를 지어내게 된다. 단 `image_progression` 은
예외다: 그 모드의 assistant_text 는 의도적으로 한 줄이고 컷 목록 자체가 턴의 산출물이므로 산문 길이가
컷 수를 말해주지 않는다. 새 비트를 못 내는 라운드가 나오면 즉시 멈춘다.

### 프롬프트 예산: 어디가 큰지 재고 고쳐라

`node server/prompt-size-breakdown.mjs womanlife` 가 실제로 전송되는 프롬프트를 분해한다.
**"메인 프롬프트를 줄여라" 는 측정으로 반박된 조언이다** — 창작자의 메인 규칙은 12.5k 프롬프트 중
2.4k 토큰이라 통째로 지워도 격차가 안 메워진다. 실제로 큰 것은 DynamicChat 자신의 스캐폴딩이었다.

| 조치 | 절감 | 내용 손실 |
|---|---|---|
| 시스템 프롬프트 ↔ foundation 모듈 줄 단위 중복 제거 | 1,399 토큰 | 없음 |
| 서사 런타임 지시문 compact 티어 | 770 토큰 | 설명만, 규칙은 전부 유지 |
| 최근 전사 축소 (0.6 → 0.35 배) | 가변 | 연속성 디테일 |

설정 화면의 LLM 시스템 프롬프트와 `main_prompt` 모듈을 같은 원본으로 채우는 창작자가 많다 —
이 시뮬레이션도 둘이 2,840자 접두사를 공유하고 뒤가 갈린다. 그래서 통짜 포함관계 검사가 아니라
**줄 단위**로 지운다: 전체 비교는 중복을 통째로 남기거나 창작자의 고유 꼬리까지 지운다.

최근 전사는 "항상 들어가는" 블록 중 **유일하게 세션이 진행되며 커지는** 것이다(8개 × 최대 800자).
그래서 1턴에는 맞던 시뮬레이션이 3턴에서 안 맞기 시작했고, 증상은 서사가 중간에 무너지는 것이었다.
이제 응답 예산을 깎기 전에 전사를 먼저 줄인다. 최신 assistant 종료부는 이번 턴의 인수인계 지점이라
축소 대상에서 제외한다.

## 디자인 토큰과 테마

색은 전부 `src/styles.css` 최상단 `:root` 의 토큰에서 나오고, 야간 테마는
`src/narrative-output.css` 끝의 `:root[data-reader-theme="night"]` 가 **같은 이름을 다시 정의**해서
만든다. narrative-output.css 는 styles.css 뒤에 import 되므로 소스 순서로 이긴다.

**규칙 하나가 전부다: 채워진 표면은 쌍이다** — 바탕과 그 위의 잉크. 둘이 같이 뒤집히거나 둘 다 안
뒤집히거나다. 한쪽만 토큰이고 다른 쪽이 리터럴이면 야간에 쌍이 어긋나 밝은 바탕에 밝은 글씨, 또는
어두운 바탕에 어두운 글씨가 된다. 어두운 버튼에 흰 글씨가 리터럴로 박혀 있는 건 **정상이다** — 그
버튼은 양쪽 테마에서 어둡다.

**라이트 테마는 움직이지 않는다.** 리터럴을 토큰으로 바꿀 때 그 토큰의 라이트 값이 리터럴과
같아야 한다. 같은 값의 토큰이 없으면 근처 토큰으로 반올림하지 말고 **라이트 값을 그대로 가진 토큰을
새로 만들어라**. 단발성 한 곳이면 토큰을 만드는 대신 야간 스코프 오버라이드 한 줄이 낫다 —
`.crack-system-note`, `.crack-setup-notice button`, `.library-hero-copy p` 가 그 예다.

토큰 이름은 값이 아니라 **역할**이다. `--dc-panel` 은 카드, `--dc-paper` 는 페이지 바닥,
`--surface-2` 는 가라앉은 면, `--dc-line` 은 선. `--paper` 를 카드에 쓰면 라이트에서는 맞아 보여도
야간에 그 토큰이 가장 어두운 면이라 카드가 페이지 아래로 가라앉는다. 실제로 그렇게 틀렸었다.

야간 값이 필요 없는 토큰도 있다. 액센트 6개만 야간 값을 갖는데, 나머지(민트·골드·코럴 등)는
야간 바닥에서 이미 4.5:1 을 넘기기 때문이다. 안 움직인 토큰을 굳이 다시 적으면 하지도 않은 결정을
한 것처럼 읽힌다.

### 세피아 테마

세피아 테마는 **읽기 열(reading column)만 리페인트하고 크롬은 건드리지 않는다.** 의도적이다.
`narrative-output.css` 의 `.crack-story-stage[data-reader-theme="sepia"]` 블록이
`--rich-*` 토큰 9개만 재정의한다(잉크·규선·배경·강조·인용). 야간 테마와 달리 `:root` 블록이
없으므로 탑바·사이드바·인스펙터·대화상자는 전부 라이트 모드를 유지한다.

세피아에서 영향받는 요소: `.rich-heading`, `.rich-strong`, `.rich-quote`, `.rich-list-item::marker`,
`.rich-table`, `.rich-code`, `.rich-divider`, `.rich-link`, `.crack-setup-notice button`.
영향받지 않는 요소: 본문 산문 `.crack-markdown`, 대화 강조 `--reader-speech`, 컴포저.

### 왜 이 규칙이 생겼는가

styles.css 는 스킨을 여러 번 덧칠하면서 이전 스킨을 지우지 않았다. 같은 셀렉터에 같은 속성을
6번까지 다시 선언한 블록이 있었고, **228개 블록 / 1,276줄이 어떤 화면에서도 절대 렌더되지 않는
죽은 코드**였다. 지울 때는 눈으로 고르지 말고 증명해라: 같은 셀렉터 텍스트의 뒤쪽 블록이 그 블록의
**모든 속성을 이름 그대로** 다시 선언하면, 그 블록은 어떤 요소에서도 마지막 선언이 될 수 없으므로
삭제해도 계산값이 바뀌지 않는다. 속성 이름을 그대로 비교하는 게 핵심이다 — `background` 가
뒤에서 `background-color` 로만 덮였다면 죽은 게 아니다.

검증도 화면 하나를 보고 판단하지 마라. 클릭 세 번 들어가야 나오는 화면이 훨씬 많다. 스타일시트를
파싱해서 (셀렉터, 속성) 별 최종 값을 라이트 토큰으로 해석한 뒤 변경 전후를 비교하면 모든 화면을
한 번에 덮는다. 브라우저 대비 측정은 그 위에 얹는 확인이지 근거가 아니다 —
`getComputedStyle` 은 150ms 색 트랜지션 도중과 화면 밖 요소에서 옛 값을 돌려준다(둘 다 실제로
겪었다).

## 이미지 생성 정책

- 이미지 프롬프트는 8개 레이어를 순서대로 합친다: 품질 → 화풍 → 작가 → 캐릭터 visual profile → 장면/배경 → 감정/의상 → 사용자 규정 → negative
- 각 캐릭터는 독립된 `char_caption`을 가진다 (NAI multi-char prompting).
- 외형/의상 정보는 LLM이 추론하지 않고 저장된 `CharacterVisualProfile` + 현재 `Wearing` 상태에서 직접 주입한다.

### 구도(frame)가 모든 태그를 결정한다

모든 cue는 태그를 고르기 **전에** `frame`을 정한다 (`ImageCueFrame`, `src/services/imageFrame.ts`):

| 필드 | 값 |
|---|---|
| `shot` | close_up / face_focus / upper_body / cowboy_shot / full_body / wide_shot 중 정확히 하나 |
| `viewpoint` | front / side / from_behind / pov / over_the_shoulder |
| `visibleRegions` | head / torso / hips / legs / feet 중 이 크롭이 실제로 보여주는 것만 |

`visibleRegions`는 LLM이 쓴 태그뿐 아니라 **DynamicChat이 로컬 주입하는 태그에도 적용된다.**
얼굴 클로즈업에 신발·치마가 주입되거나, 뒤돌아선 인물에 표정·눈동자 색이 붙는 문제가
여기서 걸러진다. 프레임이 가린 태그를 뺀 것은 상태 변경이 아니다 — 저장된 `Wearing`은
그대로이고 다음 전신 컷에서 다시 나온다.

프레임 객체가 없으면 구도 태그에서 추론한다(`inferImageCueFrameFromTags`). 회귀 방지는
`server/image-prompt-quality-eval.mjs`의 `evaluateFrameConsistency`가 담당한다.

## 시뮬레이션 격리 (run isolation)

앱은 **현재 열려 있는 시뮬레이션 + 진행(progress run) 하나**에 대한 React state만 들고 있고,
모든 비동기 완료가 거기에 써 넣는다. 그래서 `await` 뒤의 모든 `setState`는 작업이 시작된 run에
아직 있는지 확인해야 한다:

```ts
const owner = readRunOwner(snapshot);           // { simulationId, progressRunId }
setState((current) => (ownsActiveRun(current, owner) ? next(current) : current));
```

`simulation.id`만 비교하면 안 된다 — 진행 전환은 `simulation.id`를 그대로 둔 채
messages/traces/assets 배열만 바꾸므로, run A의 턴이 run B에 커밋된다. `hydrateState`가
매 write마다 최상위 배열을 활성 run 스냅샷에 다시 써 넣기 때문에 이 누수는 영구적이다.

- 턴: `runTurnFromText` / `regenerateAssistantResponse`
- 이미지: `planAndQueueImageForTurn` / `runQueuedImageJob`
- 자동 진행: run이 바뀌면 effect가 루프를 정지시킨다
- 저장 디바운스: `pendingStateSavesRef`는 시뮬레이션 id별 Map

## 세션 초기화와 연속성

세션 초기화는 "대화 리셋"이지 "시뮬레이션 리셋"이 아니다.
- 초기화 전 Handoff Pack 생성 → 새 세션에서 Context Pack으로 복원
- `docs/continuity-reset-checklist.md` 참고
