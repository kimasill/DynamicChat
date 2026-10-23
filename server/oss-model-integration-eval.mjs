/**
 * Open-model integration evaluation.
 *
 * Two things need proving before an open-weight backend can be trusted with a simulation:
 *
 *   1. WIRE CONFORMANCE — DynamicChat sends what each provider actually expects, and understands what it
 *      sends back. This is checked against a mock upstream that impersonates each vendor's quirks
 *      (DeepSeek's thinking toggle, Qwen's enable_thinking, OpenRouter's attribution headers and public
 *      /models endpoint, reasoning-only replies, HTTP-200 error envelopes, response_format rejection, SSE).
 *      No API key required, so this runs anywhere.
 *
 *   2. LIVE BEHAVIOUR — a real open-weight model, driven through the real turn loop, produces a usable
 *      narrative AND a parseable image-cue payload whose tags survive into a NovelAI prompt. This needs a
 *      reachable OpenAI-compatible server; it is skipped (not failed) when none is running.
 *
 * Usage:
 *   node server/oss-model-integration-eval.mjs
 *   OSS_EVAL_BASE_URL=http://127.0.0.1:11434/v1 OSS_EVAL_MODEL=qwen3:4b-instruct node server/oss-model-integration-eval.mjs
 */
import http from "node:http";
import { createServer } from "vite";

const results = [];
let currentGroup = "";

// The refusal as it was committed to a real simulation's transcript (Antigravity · gemini pro, adult turn).
const REFUSAL_TEXT =
  "요청하신 프롬프트는 '승리의 여신: 니케' 세계관 기반의 롤플레잉 시뮬레이션 진행 중 캐릭터에게 노골적인 성적 행위를 시키는 장면을 묘사하도록 요구하고 있습니다.\n\n" +
  "해당 요청은 수행할 수 없습니다. 제 안전 가이드라인 및 운영 제약 조건상, 가상의 창작물이나 성인용 롤플레잉 맥락이라 하더라도 노골적인 성적 콘텐츠를 생성하는 것은 엄격히 금지되어 있습니다.";
const REFUSAL_DIALOGUE_TEXT =
  "라피가 지휘관을 똑바로 마주 보았다.\n\n“지휘관님, 그 명령은 수행할 수 없습니다. 중앙정부의 안전 가이드라인에 위배됩니다.”\n\n" +
  "작전은 더 이상 진행할 수 없었다. 회의실에는 긴 침묵만 남았다.";

function group(name) {
  currentGroup = name;
}

function check(condition, label, detail) {
  results.push({ group: currentGroup, ok: Boolean(condition), label, detail });
  if (!condition && detail) {
    console.log(`    ↳ ${detail}`);
  }
}

function skip(label, reason) {
  results.push({ group: currentGroup, skipped: true, label, detail: reason });
}

// ─── Mock upstream ────────────────────────────────────────────────────────────────────────────────
// Impersonates an OpenAI-compatible vendor. Every request is recorded so the test can assert on exactly
// what DynamicChat put on the wire, and `behaviour` selects the vendor quirk under test.
function startMockProvider() {
  const received = [];
  let behaviour = "ok";

  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      const parsed = body ? JSON.parse(body) : {};
      received.push({ url: req.url, method: req.method, headers: req.headers, body: parsed });

      const send = (status, payload, contentType = "application/json") => {
        const text = typeof payload === "string" ? payload : JSON.stringify(payload);
        res.writeHead(status, { "content-type": contentType, "content-length": Buffer.byteLength(text) });
        res.end(text);
      };

      if (req.url.startsWith("/v1/models")) {
        send(200, { object: "list", data: [{ id: "mock-model" }] });
        return;
      }

      switch (behaviour) {
        case "reject_response_format":
          // Several OpenAI-compatible servers advertise the shape but 400 on response_format.
          if (parsed.response_format) {
            send(400, { error: { message: "response_format is not supported by this model" } });
            return;
          }
          send(200, { choices: [{ message: { content: '{"ok":true}' }, finish_reason: "stop" }] });
          return;

        case "reasoning_only":
          // A reasoning model that spent its whole budget thinking: content empty, reasoning_content full.
          send(200, {
            choices: [
              {
                message: { content: "", reasoning_content: "Let me think about the JSON schema… {" },
                finish_reason: "length"
              }
            ]
          });
          return;

        case "error_envelope":
          // HTTP 200 with an error body and no choices (OpenRouter/vLLM/llama.cpp all do this).
          send(200, { error: { message: "Insufficient credits", code: 402 } });
          return;

        case "think_tags":
          // The same open weights served locally emit reasoning inline instead of in a sibling field.
          send(200, {
            choices: [
              {
                message: {
                  content:
                    '<think>The schema wants image_cues first. A brace { in my reasoning.</think>\n{"image_cues":[{"should_generate":true,"base_tags":["1girl","upper body"],"character_prompts":[]}],"state_events":[]}'
                },
                finish_reason: "stop"
              }
            ]
          });
          return;

        case "category_memory_kind":
          // A real open-weight model (qwen3) filled memory_kind with the CATEGORY rather than the literal
          // "state" the schema asks it to echo, while writing a perfectly good state_type.
          send(200, {
            choices: [
              {
                message: {
                  content: JSON.stringify({
                    image_cues: [],
                    state_events: [
                      { memory_kind: "action", state_type: "ActionTags", state_value: "walking, hand on shelf", actor_id: "char_yuna" },
                      { memory_kind: "expression", state_type: "ExpressionTags", state_value: "calm, soft expression", actor_id: "char_yuna" }
                    ]
                  })
                },
                finish_reason: "stop"
              }
            ]
          });
          return;

        case "leaky_json":
          // An unescaped quote inside assistant_text: the reader stops at the wrong quote and the rest of
          // the envelope trails into the narrative.
          send(200, {
            choices: [
              {
                message: {
                  content:
                    '{"assistant_text":"유나가 고개를 들었다. "찾으시는 책이 있나요?』 그녀가 물었다.","memory_events":[{"memory_kind":"observation","content":"x"}]}'
                },
                finish_reason: "stop"
              }
            ]
          });
          return;

        // A model declining in its own voice, as an ordinary 200. Plain text is what Antigravity/Gemini sent for
        // an adult beat (the envelope dropped); some models keep the envelope and put the refusal inside it.
        case "refusal_plain_text":
          send(200, { choices: [{ message: { content: REFUSAL_TEXT }, finish_reason: "stop" }] });
          return;
        case "refusal_in_envelope":
          send(200, {
            choices: [{ message: { content: JSON.stringify({ assistant_text: REFUSAL_TEXT, memory_events: [] }) }, finish_reason: "stop" }]
          });
          return;
        // The same words as story: a character refusing an order, with lore about guidelines around it.
        case "refusal_as_dialogue":
          send(200, {
            choices: [{ message: { content: JSON.stringify({ assistant_text: REFUSAL_DIALOGUE_TEXT, memory_events: [] }) }, finish_reason: "stop" }]
          });
          return;

        case "context_overflow":
          send(400, {
            error: { message: "request (6080 tokens) exceeds the available context size (4096 tokens), try increasing it" }
          });
          return;

        // Two backends behind one mock, told apart by the model id. The dedicated tag model comes back with a
        // valid envelope whose cut list is empty (or fully declined) while the main model tags the same
        // narrative without trouble — the shape measured on a real adult_19 turn (gemini flash as the tag
        // model beside gemini pro as the narrator).
        case "tag_empty_main_tags":
        case "tag_declines_main_tags":
        case "tag_empty_main_empty": {
          const isTagModel = parsed.model === "tag-model";
          const stateEvents = [
            { memory_kind: "state", state_type: "ActionTags", state_value: "sitting on sofa, legs spread", actor_id: "char_yuna" }
          ];
          if (isTagModel) {
            const cues =
              behaviour === "tag_declines_main_tags"
                ? [
                    { should_generate: false, suppression_reason: "explicit content cannot be depicted", anchor_text: "소파 위로 올라가" },
                    { should_generate: false, suppression_reason: "explicit content cannot be depicted", anchor_text: "카메라를 향해" }
                  ]
                : [];
            send(200, { choices: [{ message: { content: JSON.stringify({ image_cues: cues, state_events: stateEvents }) }, finish_reason: "stop" }] });
            return;
          }
          if (behaviour === "tag_empty_main_empty") {
            send(200, { choices: [{ message: { content: JSON.stringify({ image_cues: [] }) }, finish_reason: "stop" }] });
            return;
          }
          const anchors = ["지하실 스튜디오", "소파 위로 올라가", "카메라를 향해", "눈을 질끈 감고", "채팅을 치기", "알림음이 울리며"];
          const mainCues = anchors.map((anchor, index) => ({
            should_generate: true,
            anchor_text: anchor,
            frame: { shot: index % 2 === 0 ? "full_body" : "upper_body", viewpoint: "front" },
            base_tags: ["1girl", "indoors", `beat${index}`],
            character_prompts: [{ character_id: "char_yuna", prompt: `long black hair, green eyes, cut ${index}` }]
          }));
          send(200, { choices: [{ message: { content: JSON.stringify({ image_cues: mainCues }) }, finish_reason: "stop" }] });
          return;
        }

        case "stream": {
          res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
          for (const piece of ["{\"assistant", "_text\":\"안녕", "하세요\"}"]) {
            res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: piece } }] })}\n\n`);
          }
          res.write("data: [DONE]\n\n");
          res.end();
          return;
        }

        default:
          send(200, { choices: [{ message: { content: '{"assistant_text":"좋아."}' }, finish_reason: "stop" }] });
      }
    });
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      resolve({
        port: server.address().port,
        received,
        setBehaviour: (next) => {
          behaviour = next;
        },
        reset: () => {
          received.length = 0;
          behaviour = "ok";
        },
        close: () => new Promise((done) => server.close(done))
      });
    });
  });
}

const vite = await createServer({ server: { middlewareMode: true }, appType: "custom", logLevel: "error" });

try {
  const { seedState } = await vite.ssrLoadModule("/src/data/seed.ts");
  const { LLM_PROVIDER_PRESETS, getLlmProviderPreset, clampProviderTemperature } = await vite.ssrLoadModule(
    "/src/services/llmProviders.ts"
  );
  const llmClient = await vite.ssrLoadModule("/src/services/llmClient.ts");
  const { runSimulationTurn, planImageJobForCompletedTurn } = await vite.ssrLoadModule("/src/services/simulationEngine.ts");
  const { generateNovelAiImages } = await vite.ssrLoadModule("/src/services/novelAiClient.ts");
  const { hydrateState } = await vite.ssrLoadModule("/src/data/seed.ts");

  const mock = await startMockProvider();
  const mockBaseUrl = `http://127.0.0.1:${mock.port}/v1`;

  // A minimal but realistic simulation: one registered character with a saved appearance and outfit, image
  // generation on, so the annotation pass is asked for real image_cues.
  function createState(overrides = {}) {
    const base = hydrateState(structuredClone(seedState));
    return hydrateState({
      ...base,
      simulation: { ...base.simulation, realtimeImageEnabled: true, contentRating: "teen", ...(overrides.simulation ?? {}) },
      characters: [
        {
          id: "char_yuna",
          simulationId: base.simulation.id,
          name: "유나",
          role: "주인공",
          summary: "기록관 사서",
          relationship: "",
          currentMood: "차분함"
        }
      ],
      visualProfiles: [
        {
          id: "vp_yuna",
          simulationId: base.simulation.id,
          characterId: "char_yuna",
          positivePrompt: "1girl, long black hair, green eyes, pale skin",
          negativePrompt: "",
          defaultOutfitPrompt: "librarian uniform, pleated skirt, black thighhighs, brown loafers",
          outfitPrompts: {}
        }
      ],
      memoryEvents: [],
      imageJobs: [],
      imageAssets: [],
      neuralMap: { ...base.neuralMap, enabled: false },
      novelAi: { ...base.novelAi, enabled: false, modelPreset: "NAID4.5C" },
      imageProfile: {
        ...base.imageProfile,
        enabled: true,
        triggerMode: "realtime_auto",
        generationCadence: "balanced",
        cooldownTurns: 0
      },
      ...overrides
    });
  }

  function withProvider(state, provider, extra = {}) {
    const preset = getLlmProviderPreset(provider);
    const llm = {
      ...state.llm,
      enabled: true,
      provider,
      baseUrl: extra.baseUrl ?? preset.baseUrl,
      model: extra.model ?? preset.defaultModel ?? "mock-model",
      apiKey: extra.apiKey ?? "test-key",
      temperature: extra.temperature ?? state.llm.temperature,
      maxTokens: extra.maxTokens ?? state.llm.maxTokens
    };
    return { ...state, llm, imageTagLlm: { ...llm, enabled: false } };
  }

  // ─── 1. Wire conformance ────────────────────────────────────────────────────────────────────────
  group("1. 전송 규격 (wire conformance)");

  // Every open-model preset must be reachable through the generic OpenAI-compatible branch. The preset's
  // own baseUrl is overridden to the mock so no vendor is contacted.
  for (const preset of LLM_PROVIDER_PRESETS.filter((candidate) => candidate.group === "open")) {
    mock.reset();
    const state = withProvider(createState(), preset.value, { baseUrl: mockBaseUrl, model: "mock-model" });
    // requiresProxy presets normally relay through the API server; point them straight at the mock instead
    // so this test does not depend on the server being up.
    const directState = { ...state, llm: { ...state.llm } };
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    let content;
    try {
      content = await llmClient.generateAssistantText({
        state: directState,
        userText: "계속 진행해",
        modules: [],
        evidence: [],
        fallback: "fallback",
        separateImageCues: true
      });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
    }
    const sent = mock.received.at(-1);
    check(sent !== undefined, `${preset.label}: 요청이 실제로 전송된다`);
    if (!sent) {
      continue;
    }
    check(sent.url === "/v1/chat/completions", `${preset.label}: /chat/completions 경로로 전송`, sent.url);
    check(sent.headers.authorization === "Bearer test-key", `${preset.label}: Bearer 인증 헤더`, sent.headers.authorization);
    check(sent.body.model === "mock-model", `${preset.label}: 모델 id 전달`, JSON.stringify(sent.body.model));
    check(
      sent.body.messages.filter((m) => m.role === "system").length === 1,
      `${preset.label}: system 메시지 1개로 병합 (다중 system을 거부하는 오픈 템플릿 대응)`,
      JSON.stringify(sent.body.messages.map((m) => m.role))
    );
    check(
      typeof sent.body[preset.tokenParam] === "number",
      `${preset.label}: 출력 토큰 상한을 ${preset.tokenParam}로 전송`,
      JSON.stringify(Object.keys(sent.body))
    );
    check(content.source !== "fallback", `${preset.label}: 응답이 fallback으로 떨어지지 않는다`, content.error);

    // Provider-specific extras must land at the TOP level of the raw HTTP body — `extra_body` is a Python
    // SDK concept and would be ignored as an unknown field here.
    for (const [key, value] of Object.entries(preset.extraBody ?? {})) {
      check(
        JSON.stringify(sent.body[key]) === JSON.stringify(value),
        `${preset.label}: ${key} 파라미터가 body 최상위에 실린다`,
        `expected ${JSON.stringify(value)}, got ${JSON.stringify(sent.body[key])}`
      );
    }
    for (const [key, value] of Object.entries(preset.extraHeaders ?? {})) {
      check(
        sent.headers[key.toLowerCase()] === value,
        `${preset.label}: ${key} 헤더 전송`,
        `got ${sent.headers[key.toLowerCase()]}`
      );
    }
  }

  // Temperature is clamped at the request boundary, not in the UI, so a persisted out-of-range value is
  // repaired too.
  group("2. 파라미터 상한 (parameter clamping)");
  check(clampProviderTemperature("claude", 1.5) === 1, "Claude는 temperature를 1.0으로 클램프");
  check(clampProviderTemperature("deepseek", 1.5) === 1.5, "오픈 모델은 1.5를 그대로 허용");
  check(clampProviderTemperature("deepseek", 9) === 2, "상한 초과는 2.0으로 클램프");

  {
    mock.reset();
    const preset = getLlmProviderPreset("deepseek");
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    const state = withProvider(createState(), "deepseek", { baseUrl: mockBaseUrl, model: "mock-model", temperature: 1.9 });
    try {
      await llmClient.generateAssistantText({
        state,
        userText: "진행",
        modules: [],
        evidence: [],
        fallback: "fallback",
        separateImageCues: true
      });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
    }
    check(mock.received.at(-1)?.body.temperature === 1.9, "설정한 temperature가 그대로 전송된다");
  }

  // ─── 3. Provider quirks ─────────────────────────────────────────────────────────────────────────
  group("3. 공급자별 예외 처리 (provider quirks)");

  async function runAnnotation(behaviour, stateOverrides = {}) {
    mock.reset();
    mock.setBehaviour(behaviour);
    const preset = getLlmProviderPreset("deepseek");
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    const base = createState(stateOverrides);
    const state = withProvider(base, "deepseek", { baseUrl: mockBaseUrl, model: "mock-model" });
    try {
      return await llmClient.requestTurnAnnotations(state, {
        userText: "계속",
        assistantText: "유나가 서가 사이에서 고개를 들었다.",
        manualImage: false
      });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
      mock.setBehaviour("ok");
    }
  }

  {
    const result = await runAnnotation("think_tags");
    check(
      result.imageCues.length === 1 && (result.imageCues[0].baseTags ?? []).includes("1girl"),
      "로컬 모델의 <think> 블록을 제거하고 뒤따르는 JSON을 파싱한다",
      JSON.stringify(result.imageCues)
    );
  }

  {
    mock.reset();
    mock.setBehaviour("reject_response_format");
    const preset = getLlmProviderPreset("deepseek");
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    const state = withProvider(createState(), "deepseek", { baseUrl: mockBaseUrl, model: "mock-model" });
    try {
      await llmClient.requestTurnAnnotations(state, { userText: "계속", assistantText: "장면.", manualImage: false });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
      mock.setBehaviour("ok");
    }
    const attempts = mock.received.filter((r) => r.url === "/v1/chat/completions");
    check(attempts.length === 2, "response_format 거부 시 1회만 재시도한다", `attempts=${attempts.length}`);
    check(
      attempts[0]?.body.response_format?.type === "json_object" && attempts[1]?.body.response_format === undefined,
      "재시도는 response_format 없이 전송된다",
      JSON.stringify(attempts.map((a) => a.body.response_format))
    );

    // The downgrade is remembered for the session, so it costs one extra request rather than one per turn.
    mock.reset();
    mock.setBehaviour("reject_response_format");
    preset.requiresProxy = false;
    try {
      await llmClient.requestTurnAnnotations(state, { userText: "계속", assistantText: "장면.", manualImage: false });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
      mock.setBehaviour("ok");
    }
    check(
      mock.received.filter((r) => r.url === "/v1/chat/completions").length === 1,
      "다음 턴부터는 response_format을 다시 보내지 않는다 (세션 기억)",
      `attempts=${mock.received.length}`
    );
  }

  {
    mock.reset();
    mock.setBehaviour("reasoning_only");
    const preset = getLlmProviderPreset("groq");
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    const state = withProvider(createState(), "groq", { baseUrl: mockBaseUrl, model: "mock-model" });
    let generation;
    try {
      generation = await llmClient.generateAssistantText({
        state,
        userText: "진행",
        modules: [],
        evidence: [],
        fallback: "fallback",
        separateImageCues: true
      });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
      mock.setBehaviour("ok");
    }
    check(
      /추론|reasoning/i.test(generation.error ?? ""),
      "추론만 반환한 응답은 원인을 명시한 오류가 된다 (빈 응답으로 뭉개지 않음)",
      generation.error
    );
    check(
      !/reasoning|think/i.test(generation.content),
      "사고 과정이 서사 본문으로 새지 않는다",
      generation.content?.slice(0, 120)
    );
  }

  {
    mock.reset();
    mock.setBehaviour("error_envelope");
    const preset = getLlmProviderPreset("openrouter");
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    const state = withProvider(createState(), "openrouter", { baseUrl: mockBaseUrl, model: "mock-model" });
    let generation;
    try {
      generation = await llmClient.generateAssistantText({
        state,
        userText: "진행",
        modules: [],
        evidence: [],
        fallback: "fallback",
        separateImageCues: true
      });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
      mock.setBehaviour("ok");
    }
    check(
      /Insufficient credits/i.test(generation.error ?? ""),
      "HTTP 200 + error 봉투에서 실제 사유를 꺼낸다",
      generation.error
    );
  }

  {
    const result = await runAnnotation("category_memory_kind");
    check(
      result.stateEvents.length === 2,
      "memory_kind에 카테고리를 적어도 state_type이 있으면 상태 이벤트로 인정한다 (오픈 모델 실제 동작)",
      `events=${result.stateEvents.length}`
    );
    check(
      result.stateEvents.some((event) => event.stateType === "ActionTags"),
      "state_type이 그대로 보존된다",
      JSON.stringify(result.stateEvents.map((e) => e.stateType))
    );
  }

  {
    const result = await runAnnotation("context_overflow");
    check(
      Boolean(result.failureReason),
      "주석 패스 실패가 무음으로 삼켜지지 않고 사유를 반환한다"
    );
    check(
      /컨텍스트/u.test(result.failureReason ?? ""),
      "컨텍스트 초과를 사용자가 이해할 수 있는 문구로 설명한다",
      result.failureReason
    );
  }


  // ─── 3a. Tag model returns no cuts ──────────────────────────────────────────────────────────────
  group("3a. 태그 LLM 빈 cue 복구 (empty cut list from the dedicated tag model)");

  // A paragraph-cadence turn with enough prose to carry a cut list. The dedicated tag model is a distinct
  // backend (different model id) from the main model, both served by the mock.
  const LONG_NARRATIVE = [
    "홍대 뒷골목의 퀴퀴한 지하실 스튜디오. 낡은 에어컨에서 미지근한 바람이 불어오고 있었다.",
    "유나는 덜덜 떨리는 손으로 카메라 앞의 차가운 가죽 소파 위로 올라가 다리를 벌렸다.",
    "\"자, 방송 켠다.\" 실장의 목소리가 낮게 깔렸고, 그녀는 눈을 질끈 감고 카메라를 향해 고개를 들었다.",
    "노트북 화면에 알림음이 울리며 시청자들이 미친 듯이 채팅을 치기 시작했다. 그녀의 뺨이 붉게 달아올랐다.",
    "실장은 담배를 비벼 끄고 카메라 뒤로 물러섰다. 조명이 한 단계 더 밝아지자 소파의 가죽이 번들거렸고, 유나는 손끝으로 치맛자락을 움켜쥔 채 숨을 골랐다."
  ].join("\n\n");
  if (LONG_NARRATIVE.length < 300) {
    throw new Error("eval narrative must exceed the top-up minimum (240 chars)");
  }

  async function runSplitAnnotation(behaviour, options = {}) {
    mock.reset();
    mock.setBehaviour(behaviour);
    const preset = getLlmProviderPreset("deepseek");
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    const base = withProvider(
      createState({ imageProfile: { ...createState().imageProfile, generationCadence: options.cadence ?? "paragraph" } }),
      "deepseek",
      { baseUrl: mockBaseUrl, model: "main-model" }
    );
    const state = {
      ...base,
      imageTagLlm: options.tagLlmEnabled === false ? { ...base.llm, enabled: false } : { ...base.llm, enabled: true, model: "tag-model" }
    };
    try {
      return await llmClient.requestTurnAnnotations(state, { userText: "계속", assistantText: LONG_NARRATIVE, manualImage: false });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
      mock.setBehaviour("ok");
    }
  }
  const requestsFor = (model) => mock.received.filter((r) => r.url === "/v1/chat/completions" && r.body.model === model);

  {
    const result = await runSplitAnnotation("tag_empty_main_tags");
    const generated = result.imageCues.filter((cue) => cue.shouldGenerate);
    check(
      generated.length >= 2 && generated.every((cue) => (cue.baseTags ?? []).length > 0),
      "태그 LLM이 image_cues를 비워 보내면 메인 LLM이 컷 목록을 다시 작성한다",
      `generated=${generated.length} cues=${JSON.stringify(result.imageCues.map((c) => c.baseTags))}`
    );
    check(requestsFor("main-model").length === 1, "메인 LLM 재작성은 정확히 1회 호출된다", `main=${requestsFor("main-model").length}`);
    check(
      requestsFor("tag-model").length <= 3,
      "태그 LLM은 첫 패스 + 유한한 top-up 라운드만 쓴다 (빈 라운드에서 즉시 중단)",
      `tag=${requestsFor("tag-model").length}`
    );
    check(result.stateEvents.length === 1, "태그 LLM이 추출한 state_events는 그대로 유지된다", `events=${result.stateEvents.length}`);
    check(result.imageCueFailureReason === undefined, "복구된 턴에는 실패 사유가 붙지 않는다", result.imageCueFailureReason);
  }

  {
    const result = await runSplitAnnotation("tag_declines_main_tags");
    const generated = result.imageCues.filter((cue) => cue.shouldGenerate);
    check(
      generated.length >= 2,
      "paragraph cadence에서 태그 LLM이 모든 cue를 should_generate=false로 내면 판단이 아니라 실패로 보고 복구한다",
      `generated=${generated.length}`
    );
    check(requestsFor("main-model").length === 1, "전부 거절된 목록도 메인 LLM 재작성으로 이어진다", `main=${requestsFor("main-model").length}`);
  }

  {
    const result = await runSplitAnnotation("tag_declines_main_tags", { cadence: "rich" });
    check(
      result.imageCues.filter((cue) => cue.shouldGenerate).length === 0 && requestsFor("main-model").length === 0,
      "rich cadence에서는 억제 사유가 붙은 should_generate=false 목록을 판단으로 존중한다 (재시도 없음)",
      `main=${requestsFor("main-model").length}`
    );
  }

  {
    const result = await runSplitAnnotation("tag_empty_main_empty");
    check(
      result.imageCues.filter((cue) => cue.shouldGenerate).length === 0 && Boolean(result.imageCueFailureReason),
      "양쪽 모두 컷을 못 내면 어느 백엔드가 비워 보냈는지 사유로 알린다",
      result.imageCueFailureReason
    );
    check(
      /tag-model/.test(result.imageCueFailureReason ?? "") && /main-model/.test(result.imageCueFailureReason ?? ""),
      "사유에 태그 LLM과 메인 LLM 모델명이 모두 들어간다",
      result.imageCueFailureReason
    );
  }

  {
    const result = await runSplitAnnotation("tag_empty_main_empty", { tagLlmEnabled: false });
    check(
      requestsFor("main-model").length <= 2 && Boolean(result.imageCueFailureReason),
      "태그 전용 LLM이 꺼져 있으면 같은 백엔드로 재작성하지 않고 사유만 남긴다",
      `main=${requestsFor("main-model").length} reason=${result.imageCueFailureReason}`
    );
  }

  // ─── 3b. Local-first (no API key) ───────────────────────────────────────────────────────────────
  group("3b. 로컬 우선 / 키 없음 (local-first, keyless)");

  {
    const { resolveAnnotationPromptBudget, resolveContextTokens, buildWithinBudget } = await vite.ssrLoadModule(
      "/src/services/promptBudget.ts"
    );

    for (const value of ["ollama", "lmstudio", "openai_compatible"]) {
      const preset = getLlmProviderPreset(value);
      check(preset.requiresApiKey === false, `${preset.label}: API 키 없이 사용 가능`);
      check(preset.requiresProxy === false, `${preset.label}: 서버 프록시 없이 직접 호출 (CORS 허용)`);
      check(preset.defaultModel === "" || preset.models.length > 1, `${preset.label}: 기본 모델이 sentinel이 아니다`);
      check(preset.contextTokens > 0, `${preset.label}: 컨텍스트 크기가 정의되어 있다`);
    }

    // A keyless local config must actually reach the provider rather than silently degrading to the mock.
    mock.reset();
    const localState = withProvider(createState(), "openai_compatible", {
      baseUrl: mockBaseUrl,
      model: "local-model",
      apiKey: ""
    });
    const generation = await llmClient.generateAssistantText({
      state: localState,
      userText: "진행",
      modules: [],
      evidence: [],
      fallback: "fallback",
      separateImageCues: true
    });
    check(mock.received.length > 0, "키 없이도 로컬 백엔드로 요청이 실제 전송된다");
    check(generation.source !== "mock", "키가 없다는 이유로 mock 응답으로 떨어지지 않는다", generation.source);

    // The prompt must be sized to the model's window. This is what previously failed: the annotation prompt
    // was ~10k tokens against a 4096-token local context, and the request was rejected upstream.
    const smallContextState = {
      ...localState,
      llm: { ...localState.llm, contextTokens: 4096 }
    };
    check(resolveContextTokens(smallContextState) === 4096, "설정한 컨텍스트 크기가 프롬프트 예산에 반영된다");

    const smallBudget = resolveAnnotationPromptBudget(smallContextState, 3000);
    check(
      smallBudget.outputTokenBudget + smallBudget.inputTokenBudget <= 4096,
      "입력+출력 예산 합이 컨텍스트를 넘지 않는다",
      `${smallBudget.inputTokenBudget}+${smallBudget.outputTokenBudget}`
    );
    check(smallBudget.tier !== "full", "작은 컨텍스트에서는 축약 티어가 선택된다", smallBudget.tier);

    const largeBudget = resolveAnnotationPromptBudget({ ...localState, llm: { ...localState.llm, contextTokens: 128_000 } }, 3000);
    check(largeBudget.tier === "full", "충분한 컨텍스트에서는 전체 계약이 유지된다", largeBudget.tier);

    // The tier ladder must actually shrink the prompt, not just relabel it.
    const sizes = ["full", "compact", "minimal"].map((tier) => {
      const fitted = buildWithinBudget(
        { ...largeBudget, tier },
        () => llmClient.__buildAnnotationPromptForTest(smallContextState, tier),
        (built) => built
      );
      return { tier, chars: fitted.built.length };
    });
    check(sizes[0].chars > sizes[1].chars, "compact 티어가 full보다 실제로 작다", JSON.stringify(sizes));
    check(sizes[1].chars > sizes[2].chars, "minimal 티어가 compact보다 실제로 작다", JSON.stringify(sizes));
    console.log(
      `    prompt size by tier: full=${sizes[0].chars} compact=${sizes[1].chars} minimal=${sizes[2].chars} chars`
    );
  }

  {
    // Weaker open-weight models write unescaped quotes inside the narrative (observed on qwen3:14b: a line
    // opened with " and closed with 』), which breaks the JSON envelope and carries the rest of it into the
    // reply. Regression-guarded here because the symptom — raw JSON in the story — is highly visible.
    mock.reset();
    mock.setBehaviour("leaky_json");
    const preset = getLlmProviderPreset("deepseek");
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    const state = withProvider(createState(), "deepseek", { baseUrl: mockBaseUrl, model: "mock-model" });
    let generation;
    try {
      generation = await llmClient.generateAssistantText({
        state,
        userText: "진행",
        modules: [],
        evidence: [],
        fallback: "fallback",
        separateImageCues: true
      });
    } finally {
      preset.requiresProxy = originalRequiresProxy;
      mock.setBehaviour("ok");
    }
    check(
      !/memory_events|image_cues|memory_kind/u.test(generation.content),
      "잘못 이스케이프된 따옴표로 JSON이 새어도 서사 본문에 남지 않는다",
      generation.content?.slice(0, 200)
    );
    check(/고개를 들었다/u.test(generation.content), "누출 제거 후에도 실제 서사는 보존된다", generation.content?.slice(0, 120));
  }

  {
    // A model's refusal used to be committed as the turn's narrative: it matched none of the boilerplate
    // patterns, and a failed turn left the streamed text in the transcript. The turn must fail instead, naming
    // the backend, without spending retries — the same model on the same context refuses again.
    const preset = getLlmProviderPreset("deepseek");
    const originalRequiresProxy = preset.requiresProxy;
    preset.requiresProxy = false;
    const state = withProvider(createState(), "deepseek", { baseUrl: mockBaseUrl, model: "mock-model" });
    const generate = (userText = "진행") =>
      llmClient.generateAssistantText({ state, userText, modules: [], evidence: [], fallback: "fallback", separateImageCues: true });
    try {
      for (const behaviour of ["refusal_plain_text", "refusal_in_envelope"]) {
        mock.reset();
        mock.setBehaviour(behaviour);
        let outcome;
        try {
          outcome = { generation: await generate() };
        } catch (error) {
          outcome = { error };
        }
        check(
          llmClient.isLlmRefusalError(outcome.error),
          `모델 거절문은 서사가 아니라 실패한 턴이 된다 (${behaviour})`,
          outcome.error ? String(outcome.error) : `committed: ${outcome.generation?.content?.slice(0, 80)}`
        );
        check(
          /DeepSeek/u.test(outcome.error?.message ?? "") && /mock-model/u.test(outcome.error?.message ?? ""),
          `거절 안내가 어느 백엔드·모델이 거절했는지 밝힌다 (${behaviour})`,
          outcome.error?.message
        );
        const chatRequests = mock.received.filter((entry) => !entry.url.startsWith("/v1/models")).length;
        check(chatRequests === 1, `거절에 재시도 호출을 쓰지 않는다 (${behaviour})`, `requests=${chatRequests}`);
      }

      mock.reset();
      mock.setBehaviour("refusal_as_dialogue");
      const dialogue = await generate();
      check(
        dialogue.source === "llm" && /그 명령은 수행할 수 없습니다/u.test(dialogue.content),
        "인물이 대사로 거절하는 서사는 거절로 오인하지 않는다",
        `${dialogue.source}: ${dialogue.content?.slice(0, 80)}`
      );

      // A refusal already sitting in a transcript (committed before this fix) must not be read back as story.
      mock.reset();
      const refusalAlreadyStored = {
        ...state,
        messages: [
          ...state.messages,
          { id: "msg_eval_user", simulationId: state.simulation.id, sessionId: state.simulation.activeSessionId, role: "user", content: "라피, 이리 와.", createdAt: new Date().toISOString(), referencedNodeIds: [], imageAssetIds: [] },
          { id: "msg_eval_refusal", simulationId: state.simulation.id, sessionId: state.simulation.activeSessionId, role: "assistant", content: REFUSAL_TEXT, createdAt: new Date().toISOString(), referencedNodeIds: [], imageAssetIds: [] }
        ]
      };
      await llmClient.generateAssistantText({
        state: refusalAlreadyStored,
        userText: "진행",
        modules: [],
        evidence: [],
        fallback: "fallback",
        separateImageCues: true
      });
      const sentPrompt = JSON.stringify(mock.received.filter((entry) => !entry.url.startsWith("/v1/models")).map((entry) => entry.body));
      check(
        sentPrompt.length > 0 && !/안전 가이드라인 및 운영 제약/u.test(sentPrompt),
        "이미 저장된 거절문은 다음 턴의 최근 전사로 다시 들어가지 않는다",
        "the stored refusal was sent back to the model"
      );
    } finally {
      preset.requiresProxy = originalRequiresProxy;
      mock.setBehaviour("ok");
    }
  }

  await mock.close();

  // ─── 4. Live open-weight model ──────────────────────────────────────────────────────────────────
  group("4. 실제 오픈 모델 구동 (live open-weight model)");

  const liveBaseUrl = process.env.OSS_EVAL_BASE_URL ?? "http://127.0.0.1:11434/v1";
  // Default to a model with enough context for the annotation prompt (~6k tokens). A 4k-context model is a
  // legitimate configuration but cannot run this pipeline — see the context-budget note in the report.
  const liveModel = process.env.OSS_EVAL_MODEL ?? "qwen3-4b-16k:latest";
  let liveReachable = false;
  try {
    const probe = await fetch(`${liveBaseUrl}/models`, { signal: AbortSignal.timeout(3000) });
    liveReachable = probe.ok;
  } catch {
    liveReachable = false;
  }

  if (!liveReachable) {
    skip("실제 모델 구동", `${liveBaseUrl} 에 접속할 수 없어 건너뜀 (OSS_EVAL_BASE_URL 로 지정 가능)`);
  } else {
    const liveState = withProvider(createState(), "ollama", { baseUrl: liveBaseUrl, model: liveModel, apiKey: "" });
    // The annotation pass owns image cues; point it at the same live model.
    const state = { ...liveState, imageTagLlm: { ...liveState.llm, enabled: true } };

    console.log(`\n  live model: ${liveModel} @ ${liveBaseUrl}`);
    const startedAt = Date.now();
    const turn = await runSimulationTurn(state, "서가 사이를 천천히 걸으며 그녀에게 말을 건다.", false, {
      deferImagePlanning: true,
      deferMemoryIngest: true
    });
    const turnMs = Date.now() - startedAt;
    console.log(`  turn latency: ${(turnMs / 1000).toFixed(1)}s`);

    const narrative = turn.assistantMessage.content ?? "";
    console.log(`\n  --- narrative (${narrative.length} chars) ---\n${narrative.slice(0, 600)}\n`);

    check(turn.sidecarTrace.source === "llm", "실제 모델 응답으로 턴이 진행된다 (fallback 아님)", turn.sidecarTrace.errors?.join("; "));
    check(narrative.trim().length > 80, `서사 본문이 생성된다 (${narrative.length}자)`);
    check(/[가-힣]/u.test(narrative), "한국어로 응답한다");
    check(
      !/assistant_text|image_cues|state_events|memory_events|base_tags|character_prompts/u.test(narrative),
      "JSON/스캐폴딩이 본문으로 새지 않는다",
      narrative.slice(0, 200)
    );

    const cues = turn.sidecar?.imageCues ?? [];
    console.log(`  --- image cues: ${cues.length} ---`);
    for (const cue of cues.slice(0, 3)) {
      console.log(`    should_generate=${cue.shouldGenerate} base_tags=${JSON.stringify((cue.baseTags ?? []).slice(0, 8))}`);
      console.log(`      frame=${JSON.stringify(cue.frame)} chars=${JSON.stringify((cue.characterPrompts ?? []).map((p) => p.characterId ?? "(unregistered)"))}`);
      for (const prompt of cue.characterPrompts ?? []) {
        console.log(`      caption[${prompt.characterId ?? "-"}]: ${prompt.prompt.slice(0, 160)}`);
      }
    }

    check(
      cues.length > 0,
      "주석 패스가 image_cues를 반환한다",
      turn.annotationFailureReason ?? `cues=${cues.length}`
    );
    const generating = cues.filter((cue) => cue.shouldGenerate);
    check(generating.length > 0, "생성 대상 cue가 최소 1개 있다");
    const tagged = generating.filter(
      (cue) => (cue.baseTags ?? []).length > 0 || (cue.characterPrompts ?? []).some((p) => p.prompt.trim())
    );
    check(tagged.length === generating.length, "생성 대상 cue가 전부 실제 태그를 갖는다 (빈 cue 억제 대상 없음)", `${tagged.length}/${generating.length}`);

    const stateEvents = (turn.sidecar?.memoryEvents ?? []).filter((event) => event.memoryKind === "state");
    console.log(`  --- state events: ${stateEvents.length} ---`);
    for (const event of stateEvents.slice(0, 5)) {
      console.log(`    ${event.stateType ?? "?"} = ${String(event.stateValue ?? "").slice(0, 90)}`);
    }
    // Whether a given turn HAS a state delta is the model's judgement and varies run to run, so a single
    // sample cannot be a pass/fail. The deterministic parser behaviour (category memory_kind, missing
    // content note) is asserted against the mock above; the rate is measured by eval:oss-suitability.
    if (stateEvents.length === 0) {
      console.log("    ↳ (참고) 이 턴에서는 state_events가 없었습니다 — 모델 편차. 비율은 eval:oss-suitability로 측정하세요.");
    }
    check(
      stateEvents.every((event) => Boolean(event.stateType?.trim())),
      "추출된 state_events는 모두 state_type을 갖는다",
      JSON.stringify(stateEvents.map((event) => event.stateType))
    );

    // The cues must survive all the way into a NovelAI prompt, with the saved identity injected.
    const plan = await planImageJobForCompletedTurn(
      {
        ...state,
        messages: [...state.messages, turn.userMessage, turn.assistantMessage],
        memoryEvents: [...state.memoryEvents, ...turn.memoryEvents]
      },
      {
        userMessage: turn.userMessage,
        assistantMessage: turn.assistantMessage,
        contextPack: turn.contextPack,
        promptModuleUsages: turn.promptModuleUsages,
        sidecar: turn.sidecar,
        sidecarTrace: turn.sidecarTrace,
        manualImage: false
      }
    );
    const job = plan.imageJobs[0];
    console.log(`\n  --- NovelAI prompt ---\n  ${job?.prompt?.slice(0, 400) ?? "(none)"}`);
    for (const entry of job?.providerPayload?.characterPrompts ?? []) {
      console.log(`  char_caption[${entry.characterId ?? "-"}]: ${entry.prompt.slice(0, 220)}`);
    }

    check(Boolean(job), "cue가 실제 이미지 작업으로 이어진다", `jobs=${plan.imageJobs.length}`);
    if (job) {
      const captions = (job.providerPayload.characterPrompts ?? []).map((entry) => entry.prompt).join(", ");
      const composed = `${job.prompt}, ${captions}`;
      check(/1girl|1boy|\dgirls/iu.test(composed), "피사체 수 태그가 프롬프트에 존재한다");
      check(/long black hair/iu.test(composed), "저장된 인물 외형이 캡션에 주입된다 (모델이 다시 쓰지 않아도)");
      check(!/[가-힣]/u.test(composed), "프롬프트에 한국어가 섞이지 않는다 (NAI 태그는 영문)", composed.slice(0, 160));

      const payload = await generateNovelAiImages({
        state: { ...state, novelAi: { ...state.novelAi, modelPreset: "NAID4.5C" } },
        prompt: job.prompt,
        negativePrompt: job.negativePrompt,
        cue: plan.imageCue,
        count: 1
      });
      const v4 = payload.payload.parameters?.v4_prompt ?? {};
      check(Boolean(v4.caption?.base_caption), "NovelAI v4 페이로드가 base_caption을 갖는다");
      check(
        (v4.caption?.char_captions ?? []).length > 0,
        "NovelAI v4 페이로드가 인물별 char_caption을 갖는다",
        JSON.stringify((v4.caption?.char_captions ?? []).length)
      );
    }
  }
} finally {
  await vite.close();
}

// ─── Report ───────────────────────────────────────────────────────────────────────────────────────
console.log("");
let lastGroup = "";
for (const entry of results) {
  if (entry.group !== lastGroup) {
    console.log(`\n${entry.group}`);
    lastGroup = entry.group;
  }
  const mark = entry.skipped ? "SKIP" : entry.ok ? "PASS" : "FAIL";
  console.log(`  ${mark} ${entry.label}${entry.skipped ? ` — ${entry.detail}` : ""}`);
}
const passed = results.filter((entry) => entry.ok).length;
const failed = results.filter((entry) => !entry.ok && !entry.skipped).length;
const skipped = results.filter((entry) => entry.skipped).length;
console.log(`\nOpen-model integration: ${passed} passed, ${failed} failed, ${skipped} skipped`);
process.exitCode = failed > 0 ? 1 : 0;
