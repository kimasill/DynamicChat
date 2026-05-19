import type {
  AppState,
  ChatMessage,
  ImageAsset,
  ImageGenerationJob,
  PromptModule,
  RedactionRequest,
  TurnResult
} from "../types";
import {
  clearState as clearLocalState,
  loadState as loadLocalState,
  saveState as saveLocalState
} from "../lib/storage";
import { createScopeHeaders } from "./security";
import { toShareableLlmSettings, toShareableNovelAiSettings } from "./runtimeApiSettings";

const API_BASE_URL_STORAGE_KEY = "dynamicchat.apiBaseUrl";
export const DEFAULT_DYNAMICCHAT_API_BASE_URL = "http://127.0.0.1:4318";
export const DEFAULT_NOVELAI_PROXY_URL = `${DEFAULT_DYNAMICCHAT_API_BASE_URL}/novelai/generate-image`;

export const dynamicChatApiEndpoints = {
  listSimulations: "/simulations",
  createSimulation: "/simulations",
  getSimulation: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}`,
  saveSimulationState: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/state`,
  personalApiVault: "/personal-api-vault",
  createPromptModule: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/prompt-modules`,
  updatePromptModule: (moduleId: string) => `/prompt-modules/${encodeURIComponent(moduleId)}`,
  createChatTurn: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/chat/turns`,
  resetSession: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/sessions/reset`,
  createImageJob: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/image-jobs`,
  persistAssets: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/assets`,
  listAssets: (simulationId: string, assetIds: string[] = []) => {
    const uniqueAssetIds = Array.from(new Set(assetIds.filter(Boolean)));
    const query = uniqueAssetIds.length > 0 ? `?ids=${uniqueAssetIds.map(encodeURIComponent).join(",")}` : "";
    return `/simulations/${encodeURIComponent(simulationId)}/assets${query}`;
  },
  createRedaction: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/redactions`,
  createBackup: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/backup`,
  listAudit: (simulationId: string) => `/simulations/${encodeURIComponent(simulationId)}/audit`,
  cancelImageJob: (jobId: string) => `/image-jobs/${encodeURIComponent(jobId)}/cancel`,
  getImageJob: (jobId: string) => `/image-jobs/${encodeURIComponent(jobId)}`
} as const;

export interface DynamicChatApiClient {
  listSimulations(): Promise<AppState[]>;
  saveSimulationState(state: AppState): Promise<void>;
  createSimulation(state: AppState): Promise<AppState>;
  getSimulation(simulationId: string): Promise<AppState | undefined>;
  loadPersonalApiVault<T>(): Promise<T | undefined>;
  savePersonalApiVault(vault: unknown): Promise<void>;
  createPromptModule(simulationId: string, module: PromptModule): Promise<PromptModule>;
  updatePromptModule(moduleId: string, patch: Partial<PromptModule>): Promise<PromptModule>;
  createRedaction(simulationId: string, redaction: RedactionRequest): Promise<void>;
  createChatTurn(simulationId: string, userText: string, manualImage: boolean): Promise<TurnResult>;
  resetSession(simulationId: string): Promise<AppState>;
  createImageJob(simulationId: string, turnId: string): Promise<ImageGenerationJob>;
  persistImageAssets(simulationId: string, assets: ImageAsset[]): Promise<ImageAsset[]>;
  listAssets(simulationId: string, assetIds?: string[]): Promise<ImageAsset[]>;
  cancelImageJob(jobId: string): Promise<ImageGenerationJob>;
  getImageJob(jobId: string): Promise<ImageGenerationJob | undefined>;
}

export interface SaveStateOptions {
  includeImagePayloads?: boolean;
  skipServer?: boolean;
}

export function loadState(): AppState | undefined {
  return loadLocalState();
}

export function saveState(state: AppState, options: SaveStateOptions = {}): void {
  const redactedState = redactBrowserCachedSecrets(state);
  const localCacheState = createBrowserCacheState(redactedState, true);
  try {
    saveLocalState(localCacheState);
  } catch (error) {
    console.warn("DynamicChat local cache skipped large image payloads.", error);
    try {
      saveLocalState(createBrowserCacheState(localCacheState, true));
    } catch {
      console.warn("DynamicChat local cache write failed; keeping the previous saved snapshot.");
    }
  }

  if (options.skipServer) {
    return;
  }

  const serverState = options.includeImagePayloads ? redactedState : localCacheState;
  window.setTimeout(() => {
    void createDynamicChatApiClient().saveSimulationState(serverState).catch(() => undefined);
  }, 0);
}

export function clearState(): void {
  clearLocalState();
}

export function configureDynamicChatApiBaseUrl(baseUrl: string): void {
  const normalized = baseUrl.trim().replace(/\/$/u, "");
  if (normalized) {
    window.localStorage.setItem(API_BASE_URL_STORAGE_KEY, normalized);
  } else {
    window.localStorage.removeItem(API_BASE_URL_STORAGE_KEY);
  }
}

export function getConfiguredDynamicChatApiBaseUrl(): string {
  return readConfiguredApiBaseUrl() ?? DEFAULT_DYNAMICCHAT_API_BASE_URL;
}

export function getNovelAiSubscriptionProxyUrl(): string {
  return `${readConfiguredApiBaseUrl() ?? DEFAULT_DYNAMICCHAT_API_BASE_URL}/novelai/subscription`;
}

export function getNovelAiGenerateProxyUrl(): string {
  return `${readConfiguredApiBaseUrl() ?? DEFAULT_DYNAMICCHAT_API_BASE_URL}/novelai/generate-image`;
}

export function createDynamicChatApiClient(baseUrl = readConfiguredApiBaseUrl() ?? DEFAULT_DYNAMICCHAT_API_BASE_URL): DynamicChatApiClient {
  return {
    async listSimulations() {
      if (!baseUrl) {
        const local = loadState();
        return local ? [local] : [];
      }

      return request<AppState[]>(baseUrl, dynamicChatApiEndpoints.listSimulations, {
        method: "GET",
        headers: createScopeHeaders(loadState())
      });
    },

    async saveSimulationState(state) {
      if (!baseUrl) {
        return;
      }

      const redactedState = redactBrowserCachedSecrets(state);
      await request<void>(baseUrl, dynamicChatApiEndpoints.saveSimulationState(state.simulation.id), {
        method: "PUT",
        body: redactedState,
        headers: createScopeHeaders(state)
      });
    },

    async createSimulation(state) {
      if (!baseUrl) {
        saveState(state);
        return state;
      }

      return request<AppState>(baseUrl, dynamicChatApiEndpoints.createSimulation, {
        method: "POST",
        body: redactBrowserCachedSecrets(state),
        headers: createScopeHeaders(state)
      });
    },

    async getSimulation(simulationId) {
      if (!baseUrl) {
        const local = loadState();
        return local?.simulation.id === simulationId ? local : undefined;
      }

      return request<AppState>(baseUrl, dynamicChatApiEndpoints.getSimulation(simulationId), {
        method: "GET",
        headers: createSimulationScopeHeaders(simulationId)
      });
    },

    async loadPersonalApiVault<T>() {
      if (!baseUrl) {
        return undefined;
      }

      return request<T>(baseUrl, dynamicChatApiEndpoints.personalApiVault, {
        method: "GET",
        headers: createScopeHeaders(loadState())
      });
    },

    async savePersonalApiVault(vault) {
      if (!baseUrl) {
        return;
      }

      await request<void>(baseUrl, dynamicChatApiEndpoints.personalApiVault, {
        method: "PUT",
        body: vault,
        headers: createScopeHeaders(loadState())
      });
    },

    async createPromptModule(simulationId, module) {
      return requestWithLocalFallback(baseUrl, dynamicChatApiEndpoints.createPromptModule(simulationId), module, module, "POST", createSimulationScopeHeaders(simulationId));
    },

    async updatePromptModule(moduleId, patch) {
      return requestWithLocalFallback(baseUrl, dynamicChatApiEndpoints.updatePromptModule(moduleId), patch, patch as PromptModule, "PATCH", createScopeHeaders(loadState()));
    },

    async createRedaction(simulationId, redaction) {
      if (!baseUrl) {
        return;
      }

      await request<void>(baseUrl, dynamicChatApiEndpoints.createRedaction(simulationId), {
        method: "POST",
        body: redaction,
        headers: createSimulationScopeHeaders(simulationId)
      });
    },

    async createChatTurn(simulationId, userText, manualImage) {
      return request<TurnResult>(baseUrl, dynamicChatApiEndpoints.createChatTurn(simulationId), {
        method: "POST",
        body: { userText, manualImage },
        headers: createSimulationScopeHeaders(simulationId)
      });
    },

    async resetSession(simulationId) {
      return request<AppState>(baseUrl, dynamicChatApiEndpoints.resetSession(simulationId), {
        method: "POST",
        headers: createSimulationScopeHeaders(simulationId)
      });
    },

    async createImageJob(simulationId, turnId) {
      return request<ImageGenerationJob>(baseUrl, dynamicChatApiEndpoints.createImageJob(simulationId), {
        method: "POST",
        body: { turnId },
        headers: createSimulationScopeHeaders(simulationId)
      });
    },

    async persistImageAssets(simulationId, assets) {
      if (!baseUrl || assets.length === 0) {
        return assets;
      }

      return request<ImageAsset[]>(baseUrl, dynamicChatApiEndpoints.persistAssets(simulationId), {
        method: "POST",
        body: { assets },
        headers: createSimulationScopeHeaders(simulationId)
      });
    },

    async listAssets(simulationId, assetIds = []) {
      if (!baseUrl) {
        const localState = loadState();
        const localAssets = [
          ...(localState?.imageAssets ?? []),
          ...(localState?.progressRuns ?? []).flatMap((run) => run.imageAssets ?? [])
        ];
        return assetIds.length > 0 ? localAssets.filter((asset) => assetIds.includes(asset.id)) : localAssets;
      }

      return request<ImageAsset[]>(baseUrl, dynamicChatApiEndpoints.listAssets(simulationId, assetIds), {
        method: "GET",
        headers: createSimulationScopeHeaders(simulationId)
      });
    },

    async cancelImageJob(jobId) {
      return request<ImageGenerationJob>(baseUrl, dynamicChatApiEndpoints.cancelImageJob(jobId), {
        method: "POST",
        headers: createScopeHeaders(loadState())
      });
    },

    async getImageJob(jobId) {
      if (!baseUrl) {
        return loadState()?.imageJobs.find((job) => job.id === jobId);
      }

      return request<ImageGenerationJob>(baseUrl, dynamicChatApiEndpoints.getImageJob(jobId), {
        method: "GET",
        headers: createScopeHeaders(loadState())
      });
    }
  };
}

function readConfiguredApiBaseUrl(): string | undefined {
  const value = window.localStorage.getItem(API_BASE_URL_STORAGE_KEY)?.trim();
  return value ? value.replace(/\/$/u, "") : undefined;
}

function createSimulationScopeHeaders(simulationId: string): Record<string, string> {
  return {
    ...createScopeHeaders(loadState()),
    "x-dynamicchat-project-id": simulationId
  };
}

function redactBrowserCachedSecrets(state: AppState): AppState {
  return {
    ...state,
    llm: toShareableLlmSettings(state.llm),
    novelAi: toShareableNovelAiSettings(state.novelAi)
  };
}

function createBrowserCacheState(state: AppState, metadataOnly = false): AppState {
  if (!metadataOnly) {
    return state;
  }

  return {
    ...state,
    imageAssets: state.imageAssets.map(stripAssetDataUrl),
    progressRuns: state.progressRuns.map((run) => ({
      ...run,
      imageAssets: run.imageAssets.map(stripAssetDataUrl)
    })),
    imageJobs: state.imageJobs.slice(-20),
    auditLog: state.auditLog.slice(-100)
  };
}

function stripAssetDataUrl(asset: ImageAsset): ImageAsset {
  if (!asset.dataUrl) {
    return asset;
  }

  const { dataUrl: _dataUrl, ...metadataOnly } = asset;
  return metadataOnly;
}

async function requestWithLocalFallback<T>(
  baseUrl: string | undefined,
  path: string,
  body: unknown,
  fallback: T,
  method = "POST",
  headers?: Record<string, string>
): Promise<T> {
  if (!baseUrl) {
    return fallback;
  }

  return request<T>(baseUrl, path, {
    method,
    body,
    headers
  });
}

async function request<T>(
  baseUrl: string | undefined,
  path: string,
  init: {
    method: string;
    body?: unknown;
    headers?: Record<string, string>;
  }
): Promise<T> {
  if (!baseUrl) {
    throw new Error("DynamicChat API base URL is not configured.");
  }

  const response = await fetch(`${baseUrl}${path}`, {
    method: init.method,
    headers: {
      ...(init.headers ?? {}),
      ...(init.body === undefined ? {} : { "content-type": "application/json" })
    },
    body: init.body === undefined ? undefined : JSON.stringify(init.body)
  });

  if (!response.ok) {
    throw new Error(`DynamicChat API request failed: ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}
