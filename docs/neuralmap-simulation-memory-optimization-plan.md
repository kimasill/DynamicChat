# NeuralMap Generic Agent Knowledge Graph Extension Plan

## 목적

핵심 방향은 범용 에이전트 지식 그래프 프레임워크를 유지하면서, DynamicChat 같은 장기 시뮬레이션 에이전트도 손실 없이 얹을 수 있도록 더 유연한 graph primitive와 세분화된 API를 제공하는 것이다.

따라서 이 문서는 `simulation-only` 기능을 NeuralMap core에 박아 넣는 계획이 아니다. 대신 다음 목표를 둔다.

- NeuralMap core는 범용 `Neuron`과 `Synapse` 그래프를 다룬다.
- 도메인별 의미는 core enum이 아니라 ontology/profile/metadata/schema registry로 확장한다.
- DynamicChat은 `simulation-memory` profile을 사용하는 한 소비자다.
- 다른 앱은 같은 API로 코드 지식, 작업 이력, 고객 지식, 연구 메모리, 운영 상태, 멀티 에이전트 실행 기억을 표현할 수 있다.

## 핵심 판단

현재 DynamicChat이 원하는 최적 구조는 "대화 로그 검색"이 아니라 "시뮬레이션 상태 그래프"다. 하지만 NeuralMap을 DynamicChat에 맞춰 `Character`, `Scene`, `Event`, `State` 전용 API로 좁히면 프레임워크의 범용성이 무너진다.

더 나은 방향은 다음이다.

```txt
Domain event / agent output / tool result
→ Domain compiler
→ Generic Graph Delta
→ Neuron/Synapse upsert
→ Optional state view / perspective view / summary view
→ Context composer
```

DynamicChat의 `Event`, `State`, `Observation`, `Belief`는 NeuralMap core type이 아니라 profile label이다. NeuralMap은 이 label을 검증하고 색인하고 context composer가 활용할 수 있게만 하면 된다.

## 현재 NeuralMap에서 이미 좋은 부분

- graph node/edge, chunk, context pack, handoff pack, trace 기반이 있다.
- `POST /graph/query`가 seed retrieval과 graph expansion을 제공한다.
- `POST /context/compose`가 Context Pack 조립을 제공한다.
- `POST /ingest/*` API가 여러 소스의 기록을 graph에 넣는다.
- scope, redaction, cache, trace, artifact loop가 있다.
- content module, memory consolidation, agent registry 기반이 생기고 있다.

이 기반은 유지해야 한다. 개선은 "DynamicChat 전용화"가 아니라 "더 자유로운 knowledge graph substrate"로 가야 한다.

## 현재 제약

### 1. core node type enum이 너무 닫혀 있음

현재 schema는 `Task`, `Person`, `Document`, `Artifact`, `Summary` 같은 범용 타입 enum에 강하게 묶여 있다. 이는 초기에 안전하지만, 도메인이 늘어나면 모든 앱 요구를 core enum에 계속 추가해야 한다.

문제:

- DynamicChat은 `State`, `Observation`, `Belief`를 표현하고 싶다.
- 운영 에이전트는 `Incident`, `RunbookStep`, `MetricSnapshot`을 표현하고 싶다.
- CRM 에이전트는 `Account`, `Opportunity`, `Objection`, `Commitment`를 표현하고 싶다.
- 연구 에이전트는 `Claim`, `Evidence`, `Experiment`, `Contradiction`을 표현하고 싶다.

이걸 모두 core enum으로 만들면 NeuralMap이 도메인별 DB가 된다. 대신 core는 자유로운 label을 허용해야 한다.

### 2. edge type도 닫혀 있음

현재 edge type은 `references`, `mentions`, `related_to`, `caused_by` 등으로 시작하기 좋지만, 도메인별 관계를 담기에는 부족하다.

DynamicChat 관점에서는 `OBSERVED`, `BELIEVES`, `SUPERSEDES`, `HAS_CURRENT_STATE`가 필요하다. 하지만 다른 앱은 `BLOCKED_BY`, `APPROVED_BY`, `DEPLOYS_TO`, `CITES`, `EVALUATED_BY`, `OWNED_BY`가 필요할 수 있다.

따라서 synapse type도 registry 기반으로 확장되어야 한다.

### 3. batch graph delta API가 부족함

현재 `POST /ingest/simulation-event`는 단일 시뮬레이션 이벤트를 넣는 데는 유용하다. 하지만 범용 에이전트 메모리에서는 한 턴 또는 한 작업 결과가 여러 노드/엣지/상태 변경을 동시에 만든다.

필요한 단위는 `simulation memory delta`가 아니라 범용 `graph delta`다.

```txt
upsert neurons
upsert synapses
close/supersede temporal facts
archive stale nodes
create summary neurons
attach source trace
```

### 4. temporal state와 current view가 core primitive가 아님

DynamicChat에서 "현재 옷", 운영 에이전트에서 "현재 incident 상태", CRM에서 "현재 deal stage"는 모두 같은 패턴이다.

```txt
old state valid_to = now
new state valid_from = now
current pointer moves to new state
old fact remains queryable as history
```

이건 simulation 전용이 아니라 범용 temporal knowledge graph 기능이다.

### 5. perspective / access / belief view가 범용으로 필요함

DynamicChat에서는 캐릭터가 무엇을 알고 있는지가 중요하다. 하지만 이 문제는 시뮬레이션만의 문제가 아니다.

- 멀티 에이전트 실행에서 agent A가 본 tool result와 agent B가 본 tool result가 다를 수 있다.
- 조직 지식에서 팀/권한별로 볼 수 있는 정보가 다를 수 있다.
- 연구 그래프에서 어떤 claim을 어떤 source가 지지하거나 믿는지 분리해야 한다.
- 고객 지원 에이전트에서 고객이 말한 것, 내부자가 확인한 것, 시스템 로그가 증명한 것을 구분해야 한다.

따라서 truth/perception/belief는 DynamicChat 전용 기능이 아니라 provenance and perspective layer로 일반화해야 한다.

### 6. Context Pack이 flat evidence 중심임

범용 Context Pack은 evidence list만으로 충분하지 않다. 소비자는 목적에 따라 sectioned context를 원한다.

예:

- DynamicChat: Current Scene, Canonical State, Character Knowledge, Relevant Events, Open Threads
- Coding agent: Current Task, Relevant Files, Decisions, Failing Tests, Constraints
- Ops agent: Incident State, Recent Signals, Suspected Causes, Actions Taken, Rollback Options
- Research agent: Claims, Evidence, Contradictions, Open Questions, Citations

따라서 Context Composer는 profile-driven section renderer가 되어야 한다.

## 제안 아키텍처

### 1. Core abstraction: Neuron and Synapse

NeuralMap core의 공개 모델은 다음처럼 잡는다.

```ts
interface Neuron {
  id: string;
  labels: string[];
  title: string;
  summary?: string;
  body_ref?: string;
  source_system?: string;
  scope?: GraphScope;
  importance?: number;
  confidence?: number;
  lifecycle_status?: "active" | "archived" | "redacted" | "deprecated";
  valid_from?: string;
  valid_to?: string | null;
  ontology?: {
    profile_id: string;
    type: string;
    version?: string;
  };
  properties: Record<string, unknown>;
  provenance: Provenance[];
  created_at: string;
  updated_at: string;
}

interface Synapse {
  id: string;
  from: string;
  to: string;
  type: string;
  labels?: string[];
  weight?: number;
  confidence?: number;
  scope?: GraphScope;
  lifecycle_status?: "active" | "archived" | "redacted" | "deprecated";
  valid_from?: string;
  valid_to?: string | null;
  ontology?: {
    profile_id: string;
    type: string;
    version?: string;
  };
  properties: Record<string, unknown>;
  provenance: Provenance[];
  created_at: string;
  updated_at: string;
}
```

기존 `GraphNode`와 `GraphEdge`는 내부 호환 alias로 유지할 수 있다. 다만 public API는 닫힌 enum 대신 `labels`, `ontology.type`, `properties`를 중심으로 확장한다.

### 2. Ontology/Profile Registry

도메인 의미는 core enum이 아니라 profile로 등록한다.

```json
{
  "id": "simulation-memory",
  "version": "1",
  "neuron_types": {
    "Character": { "base_labels": ["Entity", "Actor"] },
    "Event": { "base_labels": ["Fact", "Temporal"] },
    "State": { "base_labels": ["Fact", "Temporal", "State"] },
    "Observation": { "base_labels": ["Perspective"] },
    "Belief": { "base_labels": ["Perspective", "Uncertain"] }
  },
  "synapse_types": {
    "ACTOR_OF": { "from": ["Character"], "to": ["Event"] },
    "HAS_CURRENT_STATE": { "from": ["Character"], "to": ["State"], "current_pointer": true },
    "OBSERVED": { "from": ["Character"], "to": ["Event"], "perspective_edge": true },
    "BELIEVES": { "from": ["Character"], "to": ["State"], "perspective_edge": true },
    "SUPERSEDES": { "temporal_edge": true }
  }
}
```

다른 앱은 다른 profile을 등록한다. NeuralMap core는 profile을 검증/색인/렌더링 힌트로 사용한다.

### 3. Graph Delta as generic write unit

DynamicChat의 `MemoryDelta`는 범용 `GraphDelta`의 한 사례다.

```json
{
  "idempotency_key": "local_user:sim_sunnyline:msg_128:v1",
  "profile_id": "simulation-memory",
  "source": {
    "system": "dynamicchat",
    "run_id": "run_session_001",
    "turn_id": "msg_128",
    "raw_ref": "dynamicchat://sim_sunnyline/turns/msg_128"
  },
  "scope": {
    "tenant_id": "local_user",
    "workspace_id": "local_workspace",
    "project_id": "sim_sunnyline",
    "owner_scope": "local_user"
  },
  "upsert_neurons": [],
  "upsert_synapses": [],
  "temporal_operations": [],
  "archive": [],
  "delete": []
}
```

API는 DynamicChat 이름을 쓰지 않는다.

```txt
POST /graph/deltas
```

필요하면 domain-friendly alias로 아래를 추가할 수 있지만, core 구현은 동일해야 한다.

```txt
POST /memory/deltas
POST /agents/:id/memory-deltas
```

## 제안 API

### 1. `POST /graph/deltas`

범용 batch write endpoint다.

```json
{
  "profile_id": "simulation-memory",
  "idempotency_key": "sim_sunnyline:msg_128:v1",
  "source": {
    "system": "dynamicchat",
    "run_id": "run_session_001",
    "turn_id": "msg_128",
    "raw_ref": "dynamicchat://sim_sunnyline/turns/msg_128"
  },
  "upsert_neurons": [
    {
      "id": "event:t128:mina_change_clothes",
      "labels": ["Event", "TemporalFact"],
      "ontology": { "profile_id": "simulation-memory", "type": "Event" },
      "title": "Change clothes",
      "summary": "미나가 낡은 후드티에서 남색 코트로 갈아입었다.",
      "importance": 0.58,
      "confidence": 0.96,
      "properties": {
        "event_type": "ChangeClothes",
        "sim_time": "Day 2 19:20",
        "scene_id": "scene:d2_room_evening"
      }
    },
    {
      "id": "state:mina:wearing:t128",
      "labels": ["State", "TemporalFact"],
      "ontology": { "profile_id": "simulation-memory", "type": "State" },
      "title": "Mina wearing state",
      "summary": "미나는 현재 남색 코트를 입고 있다.",
      "valid_from": "Day 2 19:20",
      "valid_to": null,
      "properties": {
        "state_type": "Wearing",
        "owner_id": "char:mina",
        "value": "남색 코트"
      }
    }
  ],
  "upsert_synapses": [
    {
      "from": "char:mina",
      "type": "ACTOR_OF",
      "to": "event:t128:mina_change_clothes",
      "ontology": { "profile_id": "simulation-memory", "type": "ACTOR_OF" }
    },
    {
      "from": "char:mina",
      "type": "HAS_CURRENT_STATE",
      "to": "state:mina:wearing:t128",
      "ontology": { "profile_id": "simulation-memory", "type": "HAS_CURRENT_STATE" },
      "properties": {
        "current_pointer_key": "char:mina:Wearing"
      }
    }
  ],
  "temporal_operations": [
    {
      "operation": "supersede_current",
      "selector": {
        "label": "State",
        "properties": {
          "owner_id": "char:mina",
          "state_type": "Wearing"
        }
      },
      "valid_to": "Day 2 19:20",
      "superseded_by": "state:mina:wearing:t128"
    }
  ],
  "scope": {}
}
```

동작 요구:

- idempotency key로 재시도 안전성을 보장한다.
- ontology profile이 있으면 타입/관계/필수 속성을 검증한다.
- `supersede_current`는 기존 active neuron의 `valid_to`를 닫고 `SUPERSEDES` synapse를 만든다.
- current pointer synapse는 같은 key의 이전 active pointer를 inactive 처리한다.
- provenance와 trace span을 자동으로 붙인다.

### 2. `POST /graph/neurons/query`

free-form semantic + structured filter를 함께 받는 generic query다.

```json
{
  "query": "미나의 현재 옷과 복통 원인",
  "profile_id": "simulation-memory",
  "labels": ["State", "Event"],
  "filters": {
    "properties.owner_id": "char:mina",
    "lifecycle_status": "active",
    "valid_to": null
  },
  "scope": {},
  "top_k": 12
}
```

### 3. `POST /graph/traverse`

seed node와 relation policy를 분리해서 더 세밀한 traversal을 제공한다.

```json
{
  "seed_node_ids": ["state:mina:pain:t132"],
  "profile_id": "simulation-memory",
  "max_hops": 2,
  "include_synapse_types": ["CAUSE_CANDIDATE_OF", "OBSERVED", "BELIEVES", "SUMMARIZES"],
  "exclude_if": {
    "lifecycle_status": ["archived", "redacted"],
    "valid_to": { "not": null }
  },
  "scope": {}
}
```

### 4. `POST /context/compose`

기존 endpoint를 유지하되, profile-driven section renderer를 확장한다.

```json
{
  "objective": "Generate next response for the active simulation scene",
  "query": "미나가 배가 아픈 이유를 묻는다",
  "profile_id": "simulation-memory",
  "template_id": "simulation-memory:turn-context:v1",
  "seed_node_ids": [],
  "context_policy": {
    "sections": [
      "current_scene",
      "canonical_state",
      "perspective_state",
      "relevant_history",
      "open_threads"
    ],
    "perspective": {
      "actor_id": "char:junho",
      "knowledge_edges": ["OBSERVED", "BELIEVES", "KNOWS", "TOLD"]
    },
    "ranking": {
      "semantic_similarity": 0.35,
      "graph_relevance": 0.25,
      "recency": 0.15,
      "importance": 0.2,
      "active_state_bonus": 0.05
    }
  },
  "token_budget": 2500,
  "scope": {}
}
```

응답은 기존 `evidence`를 유지하면서 profile-specific sections를 추가한다.

```json
{
  "id": "ctx_...",
  "objective": "Generate next response for the active simulation scene",
  "evidence": [],
  "sections": {
    "current_scene": [],
    "canonical_state": [],
    "perspective_state": [],
    "relevant_history": [],
    "open_threads": []
  },
  "metadata": {
    "profile_id": "simulation-memory",
    "template_id": "simulation-memory:turn-context:v1",
    "excluded_by_perspective": 3,
    "ranking_policy": "profile-weighted-v1"
  }
}
```

이 구조는 DynamicChat뿐 아니라 coding/ops/research profile에도 그대로 쓸 수 있다.

### 5. `POST /entities/resolve`

도메인 중립 entity resolver다.

```json
{
  "profile_id": "simulation-memory",
  "mentions": [
    { "text": "미나", "label_hint": "Character" },
    { "text": "방", "label_hint": "Location" }
  ],
  "scope": {}
}
```

DynamicChat은 character/location을 해소한다. 다른 앱은 file/service/account/claim을 해소한다.

### 6. `POST /graph/views/current`

현재 상태, 현재 owner, 현재 assignment 같은 "current view"를 범용으로 조회한다.

```json
{
  "profile_id": "simulation-memory",
  "label": "State",
  "current_key": ["properties.owner_id", "properties.state_type"],
  "filters": {
    "properties.owner_id": ["char:mina", "char:junho"]
  },
  "scope": {}
}
```

응답:

```json
{
  "items": [
    {
      "id": "state:mina:wearing:t128",
      "current_key": "char:mina:Wearing",
      "valid_from": "Day 2 19:20",
      "valid_to": null,
      "properties": {
        "owner_id": "char:mina",
        "state_type": "Wearing",
        "value": "남색 코트"
      }
    }
  ]
}
```

## DynamicChat profile 예시

DynamicChat은 NeuralMap core를 바꾸지 않고 아래 profile을 등록해서 쓴다.

```json
{
  "id": "dynamicchat-simulation-memory",
  "extends": "simulation-memory",
  "neuron_types": {
    "Character": {
      "required_properties": ["name"],
      "aliases": ["Person", "Actor"]
    },
    "Scene": {
      "required_properties": ["session_id"]
    },
    "Event": {
      "required_properties": ["source_turn_id"]
    },
    "State": {
      "required_properties": ["owner_id", "state_type", "value"]
    },
    "Observation": {
      "required_properties": ["observer_id", "target_id"]
    },
    "Belief": {
      "required_properties": ["holder_id", "content"]
    }
  },
  "context_templates": {
    "turn-context": {
      "sections": [
        "current_scene",
        "canonical_state",
        "character_knowledge",
        "relevant_events",
        "open_threads"
      ]
    }
  }
}
```

DynamicChat-specific naming은 profile에만 머문다. NeuralMap API는 `graph delta`, `neuron`, `synapse`, `context policy`만 다룬다.

## 범용 ranking policy

ranking은 domain hard-code가 아니라 policy object로 받아야 한다.

```txt
score =
  semantic_similarity * policy.semantic_similarity
+ graph_relevance * policy.graph_relevance
+ recency * policy.recency
+ importance * policy.importance
+ current_view_bonus * policy.current_view_bonus
- temporal_expired_penalty
- lifecycle_penalty
- perspective_violation_penalty
```

기본 동작:

- `valid_to != null`이면 current query에서 제외한다.
- `lifecycle_status=archived`는 summary-first query에서 낮은 점수를 준다.
- profile이 `perspective_edges`를 정의하면 actor별 knowledge view를 만들 수 있다.
- uncertain relation은 확정 relation처럼 렌더링하지 않는다.

## DynamicChat 임시 호환 전략

NeuralMap이 위 generic API를 제공하기 전까지 DynamicChat은 현재 브리지로 동작한다.

- DynamicChat 내부에서 `MemoryDelta`를 컴파일한다.
- `MemoryEvent.metadata`에 `memory_kind`, `state_type`, `valid_from`, `supersedes`, `observer_ids`를 넣는다.
- `POST /ingest/simulation-event`로 전송한다.
- Context Pack에는 DynamicChat 내부 `Structured simulation memory` evidence를 우선 삽입한다.
- NeuralMap 응답은 scope와 simulation affinity로 필터링한다.

이 방식은 과도기로만 유지한다. 최종적으로는 `POST /graph/deltas`와 profile-driven context composer로 옮긴다.

## 구현 단계

### Phase 1. Flexible schema foundation

- 기존 node/edge enum을 유지하되 public API에 `labels`, `ontology`, `properties`를 추가한다.
- enum에 없는 domain type도 profile registry를 통해 허용한다.
- 기존 `GraphNode`/`GraphEdge`는 `Neuron`/`Synapse` 호환 wrapper로 유지한다.

### Phase 2. Ontology/profile registry

- `GET /profiles`
- `POST /profiles`
- profile별 neuron/synapse validation
- profile별 render hints와 context templates

### Phase 3. Generic graph delta write path

- `POST /graph/deltas`
- batch upsert
- idempotency
- provenance/trace 자동 기록
- temporal operations
- current pointer synapse 관리

### Phase 4. Fine-grained query APIs

- `POST /graph/neurons/query`
- `POST /graph/traverse`
- `POST /graph/views/current`
- structured filters + vector retrieval + graph traversal 결합

### Phase 5. Profile-driven context composer

- 기존 `/context/compose` 확장
- section renderer
- perspective policy
- summary-first retrieval
- profile ranking weights

### Phase 6. Compaction and lifecycle policy

- generic summary neuron 생성
- `SUMMARIZES` synapse
- low-importance source archive
- active fact/current state 유지
- profile별 compaction policy

### Phase 7. Compatibility adapters

- DynamicChat adapter: simulation memory profile
- Coding adapter: repository/task/test profile
- Ops adapter: incident/service/metric profile
- Research adapter: claim/evidence/source profile

## 완료 기준

- NeuralMap core가 DynamicChat 전용 endpoint 없이도 DynamicChat memory model을 표현할 수 있다.
- core schema가 닫힌 enum에만 의존하지 않고 profile-defined labels/types를 받는다.
- `Neuron`/`Synapse` API로 node/edge를 세밀하게 batch upsert할 수 있다.
- temporal state, current view, lifecycle status가 범용 기능으로 동작한다.
- Context Pack composer가 profile-driven sectioned context를 반환한다.
- DynamicChat은 `dynamicchat-simulation-memory` profile을 사용해 Event/State/Observation/Belief를 표현한다.
- 다른 도메인도 같은 API로 자체 ontology를 등록해 사용할 수 있다.

## 결론

NeuralMap을 DynamicChat에 맞춰 변형하면 안 된다. 대신 NeuralMap을 더 범용적인 에이전트 지식 그래프 substrate로 확장해야 한다.

```txt
DynamicChat-specific API
```

가 아니라

```txt
Generic Neuron/Synapse + Profile/Ontology + Graph Delta + Context Policy
```

가 맞다.

이렇게 하면 DynamicChat의 시뮬레이션 에이전트는 자연스럽게 호환되고, 동시에 다른 앱의 에이전트도 같은 NeuralMap 위에서 각자의 지식 구조를 안전하게 표현할 수 있다.
