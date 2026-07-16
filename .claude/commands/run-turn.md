---
description: 시뮬레이션 턴 1회를 실행하고 결과를 확인한다. 인자: <simulation_id> <"사용자 메시지">
---

DynamicChat API 서버에 채팅 턴 요청을 보내고 결과를 확인한다.

인자: $ARGUMENTS
(형식: <simulation_id> "<사용자 메시지>")

다음 단계를 수행한다:

1. 먼저 `http://127.0.0.1:8788/health`로 서버가 실행 중인지 확인한다.
   - 서버가 꺼져 있으면 `pnpm api`를 백그라운드로 실행하고 3초 대기한다.

2. 인자에서 simulation_id와 메시지를 파싱한다.
   - 인자가 없으면 `GET http://127.0.0.1:8788/simulations`로 시뮬레이션 목록을 조회하고
     가장 최근 것을 선택한다.

3. `POST http://127.0.0.1:8788/simulations/{simulation_id}/chat/turns` 요청을 보낸다:
   ```json
   {
     "userMessage": "<메시지>",
     "sessionId": "session_test"
   }
   ```
   헤더: `x-dynamicchat-owner-id: dev`, `x-dynamicchat-project-id: dev`

4. 응답에서 다음을 출력한다:
   - 어시스턴트 응답 텍스트
   - 생성된 memory_events 목록 (importance, tags, content)
   - image_cues (should_generate, reason, characters, tags)
   - 이미지 job이 생성됐으면 job ID와 status
   - TurnTrace: 선택된 프롬프트 모듈 목록

5. 응답이 501이면 "채팅 턴 실행이 아직 구현되지 않음 (AIN-19~21)" 을 안내한다.
