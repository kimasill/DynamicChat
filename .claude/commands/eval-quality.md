---
description: DynamicChat 품질 평가를 실행한다. 인자: all | simulation | image-prompts | simulation-runs (기본값: all)
---

DynamicChat 품질 평가 스크립트를 실행한다.

인자: $ARGUMENTS (기본값: all)

다음과 같이 처리한다:

- 인자가 `simulation` 이면: `pnpm eval:simulation`
- 인자가 `image-prompts` 이면: `pnpm eval:image-prompts`
- 인자가 `simulation-runs` 이면: `pnpm eval:simulation-runs`
- 그 외(all 또는 인자 없음): `pnpm eval:quality` (전체)

실행 후 결과를 요약한다:
1. 각 평가 항목의 통과/실패 수
2. 실패한 케이스의 이름과 이유
3. 이미지 프롬프트 평가라면: 레이어별 점수 (품질, 캐릭터 일관성, 장면 묘사)
4. 개선이 필요한 상위 3개 항목 제안

평가 파일 위치:
- `server/simulation-quality-eval.mjs`
- `server/image-prompt-quality-eval.mjs`
- `server/simulation-run-quality-eval.mjs`
