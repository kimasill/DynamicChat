import type { LlmApiSettings, NovelAiApiSettings } from "../types";

export function toShareableLlmSettings(settings: LlmApiSettings): LlmApiSettings {
  return {
    ...settings,
    apiKey: "",
    registrationStatus: "idle",
    verifiedAt: undefined,
    verificationMessage: ""
  };
}

export function toShareableNovelAiSettings(settings: NovelAiApiSettings): NovelAiApiSettings {
  return {
    ...settings,
    apiKey: "",
    registrationStatus: "idle",
    verifiedAt: undefined,
    verificationMessage: "",
    subscriptionTier: undefined
  };
}
