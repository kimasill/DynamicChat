import type { LlmApiSettings, NovelAiApiSettings } from "../types";
import { getNovelAiSubscriptionProxyUrl } from "./dynamicChatApi";

export interface ApiValidationResult {
  ok: boolean;
  message: string;
  verifiedAt?: string;
  details?: Record<string, string>;
}

export async function validateLlmApi(settings: LlmApiSettings): Promise<ApiValidationResult> {
  if (settings.provider === "mock") {
    return {
      ok: true,
      message: "Mock 모드는 외부 API 검증 없이 등록할 수 있습니다.",
      verifiedAt: new Date().toISOString()
    };
  }

  if (!settings.apiKey.trim()) {
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
          "anthropic-version": "2023-06-01"
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

    const response = await fetchWithTimeout(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${settings.apiKey}`
      },
      body: JSON.stringify({
        model: settings.model,
        temperature: 0,
        max_tokens: 8,
        messages: [{ role: "user", content: "Reply with OK only." }]
      })
    });
    return response.ok ? success("LLM API 키 검증 및 등록이 완료되었습니다.") : failure("LLM", response.status);
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
      return {
        ok: false,
        message: `NovelAI 검증 실패: DynamicChat API 프록시가 HTTP ${response.status}를 반환했습니다.`,
        details: { tier: "proxy_http_error", normalizedApiKey: token }
      };
    }

    const data = (await response.json()) as Record<string, unknown>;
    const perks = isRecord(data.perks) ? data.perks : {};
    const trainingStepsLeft = isRecord(data.trainingStepsLeft) ? data.trainingStepsLeft : {};
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
  return value.trim().replace(/^Bearer\s+/iu, "").trim();
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
