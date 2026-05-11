import type { AppState } from "../types";

const STORAGE_KEY = "dynamicchat.appState.v1";

export function loadState(): AppState | undefined {
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return undefined;
  }

  try {
    return JSON.parse(raw) as AppState;
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return undefined;
  }
}

export function saveState(state: AppState): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function clearState(): void {
  window.localStorage.removeItem(STORAGE_KEY);
}
