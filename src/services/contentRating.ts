import type { AppState, ImageSafetyLevel } from "../types";

export function isAdultContentMode(state: AppState): boolean {
  return state.simulation.contentRating === "adult_19";
}

// Adult mode widens the image safety level to "explicit", but that widening is derived here at read
// time rather than written into imageProfile.safetyLevel. Storing it would latch: lowering the rating
// back to "general" could not restore the creator's original pick, because the original was gone.
export function resolveEffectiveImageSafetyLevel(state: AppState): ImageSafetyLevel {
  return isAdultContentMode(state) ? "explicit" : state.imageProfile.safetyLevel;
}

export function createImageUserRulesForContentRating(state: AppState): string {
  const userRules = state.imageProfile.userRules
    .split(/\n+/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .join("\n");

  if (!isAdultContentMode(state)) {
    return userRules;
  }

  return relaxImageUserRulesForAdultMode(userRules);
}

export function relaxImageUserRulesForAdultMode(userRules: string): string {
  return userRules
    .split(/\n+/u)
    .map((line) => {
      const trimmed = line.trim();
      if (trimmed === "캐릭터의 외형 일관성을 우선하고, 노골적 수위는 생성하지 않는다.") {
        return "캐릭터의 외형 일관성을 우선한다.";
      }
      return trimmed;
    })
    .filter(Boolean)
    .join("\n");
}
