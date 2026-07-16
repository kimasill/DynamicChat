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
    subscriptionTier: undefined,
    // Strip heavy base64 payloads: the raw image data URL and the pre-encoded vibe
    // blob can be several MB each and must not be included in server-bound state
    // bodies or simulation exports. The actual payloads live in the personal vault
    // and are re-injected at runtime by applyPersonalApiVault.
    vibeTransferReferences: settings.vibeTransferReferences.map(({ image: _image, encodedVibe: _encodedVibe, ...rest }) => ({
      ...rest,
      image: ""
    }))
  };
}
