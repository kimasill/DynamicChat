import type { NovelAiModelPreset } from "../types";

export function resolveNovelAiModelName(preset: NovelAiModelPreset, fallback = "nai-diffusion-4-5-curated"): string {
  const models: Record<NovelAiModelPreset, string> = {
    "NAID4.5F": "nai-diffusion-4-5-full",
    "NAID4.5C": "nai-diffusion-4-5-curated",
    "NAID4.0F": "nai-diffusion-4-full",
    "NAID4.0C": "nai-diffusion-4-curated-preview",
    NAID3: "nai-diffusion-3"
  };

  return models[preset] ?? fallback;
}
