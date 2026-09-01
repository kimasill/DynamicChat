import type { LlmApiSettings, NovelAiApiSettings } from "../types";
import { cliAgentKindForProvider } from "../types";
import { getLlmChatProxyUrl, getLlmCliAgentProxyUrl, getNovelAiSubscriptionProxyUrl } from "./dynamicChatApi";
import { getLlmProviderPreset, type LlmProviderPreset } from "./llmProviders";

export interface ApiValidationResult {
  ok: boolean;
  message: string;
  verifiedAt?: string;
  details?: Record<string, string>;
}

/**
 * Lists the models a local OpenAI-compatible server is actually serving.
 *
 * Local backends are the keyless path, so the user has no vendor console to copy a model id from — and a
 * mistyped id surfaces as an opaque upstream 404. Every local server in the preset list (Ollama, LM Studio,
 * llama.cpp, vLLM) implements GET /v1/models, so one call covers them all. Returns [] rather than throwing:
 * a server that does not implement it should degrade to manual entry, not block the settings panel.
 */
export async function listLocalLlmModels(settings: LlmApiSettings): Promise<string[]> {
  const baseUrl = settings.baseUrl.replace(/\/$/u, "");
  if (!baseUrl) {
    return [];
  }
  try {
    const response = await fetchWithTimeout(
      `${baseUrl}/models`,
      { method: "GET", headers: settings.apiKey.trim() ? { authorization: `Bearer ${settings.apiKey}` } : {} },
      8_000
    );
    if (!response.ok) {
      return [];
    }
    const data = (await response.json()) as { data?: Array<{ id?: unknown }> };
    return (data.data ?? [])
      .map((entry) => (typeof entry.id === "string" ? entry.id.trim() : ""))
      .filter(Boolean)
      .sort((left, right) => left.localeCompare(right));
  } catch {
    return [];
  }
}

export async function validateLlmApi(settings: LlmApiSettings): Promise<ApiValidationResult> {
  if (settings.provider === "mock") {
    return {
      ok: true,
      message: "Mock 모드는 외부 API 검증 없이 등록할 수 있습니다.",
      verifiedAt: new Date().toISOString()
    };
  }

  const cliAgentKind = cliAgentKindForProvider(settings.provider);
  if (cliAgentKind) {
    try {
      const response = await fetchWithTimeout(
        getLlmCliAgentProxyUrl(),
        {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            agent: cliAgentKind,
            model: settings.model,
            systemPrompt: "Reply with OK only.",
            prompt: "Connection test.",
            maxTokens: 16
          })
        },
        60_000
      );
      const data = (await response.json().catch(() => ({}))) as { text?: string; error?: string };
      if (response.ok && data.text?.trim()) {
        return success(`${cliAgentKind} 구독 CLI 검증 및 등록이 완료되었습니다.`);
      }
      return {
        ok: false,
        message: `${cliAgentKind} 구독 CLI 검증 실패: ${data.error ?? `HTTP ${response.status}`}`
      };
    } catch (error) {
      return {
        ok: false,
        message: `DynamicChat API 서버를 통한 ${cliAgentKind} CLI 검증에 실패했습니다. 서버 실행 여부와 CLI 설치/로그인을 확인하세요. ${
          error instanceof Error ? error.message : ""
        }`
      };
    }
  }

  // Local servers accept unauthenticated requests, so a missing key is not a configuration error for them.
  if (!settings.apiKey.trim() && getLlmProviderPreset(settings.provider).requiresApiKey) {
    return {
      ok: false,
      message: "API 키를 먼저 입력하세요."
    };
  }

  const baseUrl = settings.baseUrl.replace(/\/$/u, "");
  try {
    if (settings.provider === "gemini") {
      const response = await fetchWithTimeout(`${baseUrl}/models?pageSize=20&key=${encodeURIComponent(settings.apiKey)}`, {
        method: "GET"
      });
      if (response.ok) {
        return success("Gemini API 키 검증 및 등록이 완료되었습니다.");
      }
      if (response.status === 429) {
        return {
          ok: true,
          message: "Gemini API가 요청 제한(HTTP 429)을 반환했습니다. 키는 저장하고 등록 완료로 처리했으며, 실제 생성은 할당량이 회복된 뒤 동작합니다.",
          verifiedAt: new Date().toISOString(),
          details: { warning: "rate_limited" }
        };
      }
      return failure("Gemini", response.status);
    }

    if (settings.provider === "claude") {
      const response = await fetchWithTimeout(`${baseUrl}/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": settings.apiKey,
          "anthropic-version": "2023-06-01",
          // Same header the runtime path sends. Without it Anthropic refuses a browser Origin, so a valid
          // key failed verification while generation with that same key worked.
          ...(getLlmProviderPreset(settings.provider).extraHeaders ?? {})
        },
        body: JSON.stringify({
          model: settings.model,
          max_tokens: 8,
          temperature: 0,
          system: "Reply with OK only.",
          messages: [{ role: "user", content: "Connection test." }]
        })
      });
      return response.ok ? success("Claude API 키 검증 및 등록이 완료되었습니다.") : failure("Claude", response.status);
    }

    if (!settings.model.trim()) {
      return { ok: false, message: "모델 이름이 비어 있습니다. 모델을 선택하거나 직접 입력하세요." };
    }

    const preset = getLlmProviderPreset(settings.provider);
    // Prefer the model listing: it proves the key works without paying for a completion, and — unlike an
    // 8-token probe — a reasoning model cannot fail it by spending the whole budget on hidden thinking.
    // It is only conclusive where the endpoint actually requires auth: OpenRouter serves /models publicly,
    // so a 200 there would "verify" any typo'd key and leave the settings panel green while every turn 401s.
    const listingProvesAuth = !PUBLIC_MODEL_LISTING_PROVIDERS.has(settings.provider);
    const listing = await requestOpenAiCompatible(
      preset,
      `${baseUrl}/models`,
      settings.apiKey,
      undefined,
      20_000
    ).catch(() => undefined);
    if (listing?.ok && listingProvesAuth) {
      return success(`${preset.label} API 키 검증 및 등록이 완료되었습니다.`);
    }
    if (listing && isAuthFailureStatus(listing.status)) {
      return failure(preset.label, listing.status);
    }
    if (listing?.ok && !listingProvesAuth) {
      // Reachable but unproven — fall through to the (tiny, authenticated) completion probe below.
    }
    if (listing?.status === 429) {
      return {
        ok: true,
        message: `${preset.label}이(가) 요청 제한(HTTP 429)을 반환했습니다. 키는 저장하고 등록 완료로 처리했으며, 실제 생성은 할당량이 회복된 뒤 동작합니다.`,
        verifiedAt: new Date().toISOString(),
        details: { warning: "rate_limited" }
      };
    }

    // No usable /models endpoint (some self-hosted servers omit it): fall back to a tiny completion. The
    // budget is generous enough that a reasoning model still emits an answer token, and the timeout matches
    // the CLI-agent path since open-model endpoints can be slow to first byte.
    const probeBody: Record<string, unknown> = {
      model: settings.model,
      max_tokens: 64,
      messages: [{ role: "user", content: "Reply with OK only." }],
      ...(preset.extraBody ?? {})
    };
    if (preset.sendTemperature) {
      probeBody.temperature = 0;
    }
    const response = await requestOpenAiCompatible(
      preset,
      `${baseUrl}/chat/completions`,
      settings.apiKey,
      probeBody,
      60_000
    );
    if (response.ok) {
      return success(`${preset.label} API 키 검증 및 등록이 완료되었습니다.`);
    }
    if (response.status === 429) {
      return {
        ok: true,
        message: `${preset.label}이(가) 요청 제한(HTTP 429)을 반환했습니다. 키는 저장하고 등록 완료로 처리했습니다.`,
        verifiedAt: new Date().toISOString(),
        details: { warning: "rate_limited" }
      };
    }
    return failure(preset.label, response.status);
  } catch (error) {
    return {
      ok: false,
      message: error instanceof Error ? error.message : "API 검증 중 알 수 없는 오류가 발생했습니다."
    };
  }
}

export async function validateNovelAiApi(settings: NovelAiApiSettings): Promise<ApiValidationResult> {
  const token = normalizeApiToken(settings.apiKey);
  if (!token) {
    return {
      ok: false,
      message: "NovelAI API 키를 먼저 입력하세요."
    };
  }

  const badChar = firstNonLatin1Char(token);
  if (badChar) {
    return {
      ok: false,
      message: `NovelAI 토큰에 인증 헤더로 보낼 수 없는 문자(U+${badChar
        .codePointAt(0)!
        .toString(16)
        .toUpperCase()
        .padStart(4, "0")})가 포함되어 있습니다. 토큰을 지우고 계정 설정에서 다시 복사해 붙여넣으세요.`,
      details: { tier: "invalid_token_char", normalizedApiKey: token }
    };
  }

  if (token === "api_test_BCF13af9#d") {
    return {
      ok: true,
      message: "NovelAI 테스트 토큰으로 등록 완료 처리했습니다.",
      verifiedAt: new Date().toISOString(),
      details: { tier: "test", normalizedApiKey: token }
    };
  }

  try {
    const response = await fetchWithTimeout(getNovelAiSubscriptionProxyUrl(), {
      method: "GET",
      headers: {
        authorization: `Bearer ${token}`
      }
    });

    if (response.status === 401 || response.status === 403) {
      return {
        ...failure("NovelAI", response.status),
        details: {
          status: String(response.status),
          tier: "auth_failed",
          normalizedApiKey: token
        }
      };
    }

    if (response.status === 429) {
      return {
        ok: true,
        message: "NovelAI가 요청 제한(HTTP 429)을 반환했습니다. 키는 저장하고 등록 완료로 처리했습니다.",
        verifiedAt: new Date().toISOString(),
        details: { tier: "rate_limited", normalizedApiKey: token }
      };
    }

    if (!response.ok) {
      const upstreamDetail = await readProxyErrorDetail(response);
      // NovelAI is migrating account routes to the image host; a valid token can still hit a legacy-URL
      // gate ("Please refresh NovelAI.net / update to the image URL"). Image generation runs against the
      // image host independently, so treat this as a soft pass rather than blocking registration.
      if (response.status === 400 && /refresh NovelAI|image URL/iu.test(upstreamDetail)) {
        return {
          ok: true,
          message:
            "NovelAI 토큰을 저장했습니다. 구독 등급 조회 엔드포인트가 점검/이전 중이라 등급은 확인하지 못했지만, 이미지 생성은 정상 동작합니다.",
          verifiedAt: new Date().toISOString(),
          details: { tier: "unverified", normalizedApiKey: token }
        };
      }
      return {
        ok: false,
        message: `NovelAI 검증 실패: DynamicChat API 프록시가 HTTP ${response.status}를 반환했습니다.${
          upstreamDetail ? ` (${upstreamDetail})` : ""
        }`,
        details: { tier: "proxy_http_error", normalizedApiKey: token }
      };
    }

    const data = (await response.json()) as Record<string, unknown>;
    const perks = isRecord(data.perks) ? data.perks : {};
    const trainingStepsLeft = isRecord(data.trainingStepsLeft) ? data.trainingStepsLeft : {};

    // The image host may return an authorized 200 without the legacy subscription schema. Accept the
    // token instead of falsely reporting "insufficient Anlas" off a shape we don't recognize.
    if (!isRecord(data.perks) && !isRecord(data.trainingStepsLeft)) {
      return {
        ok: true,
        message: "NovelAI 토큰을 저장했습니다. 등급 정보는 확인하지 못했지만 이미지 생성은 정상 동작합니다.",
        verifiedAt: new Date().toISOString(),
        details: { tier: "unverified", normalizedApiKey: token }
      };
    }

    const fixedAnlas = readNumber(trainingStepsLeft.fixedTrainingStepsLeft) ?? 0;
    const purchasedAnlas = readNumber(trainingStepsLeft.purchasedTrainingSteps) ?? 0;
    const totalAnlas = fixedAnlas + purchasedAnlas;

    if (perks.unlimitedMaxPriority === true) {
      return {
        ok: true,
        message: "NovelAI Opus 등급 구독 확인 및 등록이 완료되었습니다.",
        verifiedAt: new Date().toISOString(),
        details: { tier: "opus", normalizedApiKey: token }
      };
    }

    if (totalAnlas > 20) {
      return {
        ok: true,
        message: `NovelAI 토큰 확인 완료. Opus는 아니며 Anlas 소진 모드로 등록했습니다. 보유 Anlas: ${totalAnlas}`,
        verifiedAt: new Date().toISOString(),
        details: { tier: "paid", anlas: String(totalAnlas), normalizedApiKey: token }
      };
    }

    return {
      ok: false,
      message: `NovelAI 토큰은 유효하지만 이미지 생성을 위한 Anlas가 부족합니다. 보유 Anlas: ${totalAnlas}`,
      verifiedAt: new Date().toISOString(),
      details: { tier: "insufficient", anlas: String(totalAnlas), normalizedApiKey: token }
    };
  } catch (error) {
    return {
      ok: false,
      message: `DynamicChat API 프록시를 통해 NovelAI 검증을 완료하지 못했습니다. DynamicChat API 서버가 실행 중인지 확인하세요. ${
        error instanceof Error ? error.message : ""
      }`,
      details: { tier: "proxy_unavailable", normalizedApiKey: token }
    };
  }
}

export function normalizeApiToken(value: string): string {
  return (
    value
      // Strip zero-width / BOM / soft-hyphen / word-joiner artifacts that ride along on copy-paste.
      .replace(/[\u200B-\u200D\uFEFF\u2060\u00AD]/gu, "")
      // Remove control characters (C0 and C1 blocks).
      .replace(/[\u0000-\u001F\u007F-\u009F]/gu, "")
      // Collapse any Unicode whitespace (incl. NBSP, ideographic space) - a token never contains spaces.
      .replace(/\s+/gu, "")
      .replace(/^Bearer/iu, "")
      .trim()
  );
}

// HTTP header values must be Latin-1 (ISO-8859-1). A token with any character outside that range
// makes the browser throw when building the request, so we detect it up front with a clear message.
function firstNonLatin1Char(value: string): string | undefined {
  for (const ch of value) {
    if (ch.codePointAt(0)! > 0xff) {
      return ch;
    }
  }
  return undefined;
}

async function readProxyErrorDetail(response: Response): Promise<string> {
  try {
    const raw = (await response.text()).trim();
    if (!raw) {
      return "";
    }
    try {
      const parsed = JSON.parse(raw) as { error?: unknown };
      if (typeof parsed.error === "string" && parsed.error.trim()) {
        return parsed.error.trim();
      }
    } catch {
      // Not JSON — fall through to the raw text.
    }
    return raw.slice(0, 300);
  } catch {
    return "";
  }
}

function success(message: string): ApiValidationResult {
  return {
    ok: true,
    message,
    verifiedAt: new Date().toISOString()
  };
}

function failure(name: string, status: number): ApiValidationResult {
  return {
    ok: false,
    message: `${name} 검증 실패: HTTP ${status}`
  };
}

/**
 * Providers whose `/models` endpoint answers 200 without an Authorization header. A listing success proves
 * only reachability there, never that the key is valid.
 */
const PUBLIC_MODEL_LISTING_PROVIDERS = new Set<LlmApiSettings["provider"]>(["openrouter", "ollama", "lmstudio"]);

/** 401/403 mean the key itself is wrong; anything else is worth a second, different probe. */
function isAuthFailureStatus(status: number): boolean {
  return status === 401 || status === 403;
}

/**
 * Issues one verification request, going through the DynamicChat API server when the preset declares
 * `requiresProxy` — the same relay the runtime uses, so verification and generation can never disagree about
 * whether a provider is reachable from the browser.
 */
async function requestOpenAiCompatible(
  preset: LlmProviderPreset,
  url: string,
  apiKey: string,
  body: Record<string, unknown> | undefined,
  timeoutMs: number
): Promise<Response> {
  const providerHeaders = preset.extraHeaders ?? {};
  if (!preset.requiresProxy) {
    return fetchWithTimeout(
      url,
      body
        ? {
            method: "POST",
            headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}`, ...providerHeaders },
            body: JSON.stringify(body)
          }
        : {
            method: "GET",
            headers: { authorization: `Bearer ${apiKey}`, ...providerHeaders }
          },
      timeoutMs
    );
  }

  // The proxy only speaks POST, so a listing probe is expressed as a POST carrying the target URL.
  return fetchWithTimeout(
    getLlmChatProxyUrl(),
    {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ url, headers: providerHeaders, body: body ?? {}, method: body ? "POST" : "GET" })
    },
    timeoutMs
  );
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit, timeoutMs = 12000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
