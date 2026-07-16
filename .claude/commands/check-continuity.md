---
description: 시뮬레이션의 세션 연속성을 검사한다. 핵심 기억이 세션 초기화 후에도 유지되는지 확인한다. 인자: <simulation_id>
---

DynamicChat 세션 연속성 검증을 수행한다.

인자: $ARGUMENTS (simulation_id)

다음 단계를 수행한다:

1. `GET http://127.0.0.1:8788/simulations/{simulation_id}`로 시뮬레이션 상태를 읽는다.

2. 현재 세션의 주요 사실을 추출한다:
   - 캐릭터 관계 (관계 상태가 있는 memory_events)
   - 약속/결정 (tags에 "promise"나 "약속"이 있는 이벤트)
   - 현재 장소/장면 상태
   - 인벤토리/보유 아이템
   - 발생한 주요 세계관 사건

3. `GET http://127.0.0.1:8788/simulations/{simulation_id}/audit`로 최근 이벤트를 확인한다.

4. `docs/continuity-reset-checklist.md`의 체크리스트 항목과 비교한다.

5. 결과를 보고한다:
   - 유지되고 있는 기억 목록 (✓)
   - 누락 위험이 있는 기억 목록 (⚠)
   - 세션 초기화를 해도 안전한지 최종 판정
   - NeuralMap Context Pack에 포함된 memory node 수

**주의**: 세션 초기화는 "대화 리셋"이지 "시뮬레이션 리셋"이 아님을 사용자에게 확인시킨다.
