import type { AppState, Id, ImageCue, PhotorealApiSettings, PhotorealWorkflowPresetId } from "../types";
import { createId } from "../lib/id";
import {
  buildComfyWorkflow,
  computeReferenceHeadCrop,
  PHOTOREAL_WORKFLOW_PRESETS,
  resolvePhotorealCfg,
  resolvePhotorealModelFiles,
  resolvePhotorealOutputSize,
  resolvePhotorealPresetForCut
} from "./comfyWorkflows";
import type { BuiltComfyWorkflow } from "./comfyWorkflows";
import { DEFAULT_DYNAMICCHAT_API_BASE_URL, getPhotorealRelayBaseUrl } from "./dynamicChatApi";
import { DEFAULT_PHOTOREAL_SETTINGS } from "./runtimeApiSettings";

export interface PhotorealReferenceInput {
  /** Placeholder name in the graph (`%ref:<name>%`). The relay accepts [A-Za-z0-9_.-]{1,64}. */
  name: string;
  /**
   * Object-store key of a picked profile image (isPickedProfileObjectKey); the relay reads the file itself. There is
   * deliberately no way to hand over bytes: a face reference is only ever an image DynamicChat generated and the user
   * picked, and an object key the store wrote for a pick is the one thing that says so.
   */
  objectKey: string;
  characterId?: Id;
  /**
   * The reference's pixel size (CharacterProfileImage.width/height). When known, the graph crops the reference to
   * its head-and-shoulders square before encoding — see computeReferenceHeadCrop for why the whole image is unsafe.
   */
  width?: number;
  height?: number;
}

export interface PhotorealImageResult {
  dataUrls: string[];
  payload: Record<string, unknown>;
}

const PROBE_MODEL_FOLDERS = ["diffusion_models", "text_encoders", "vae", "loras"] as const;
type ProbeModelFolder = (typeof PROBE_MODEL_FOLDERS)[number];

export interface ComfyUiProbeResult {
  ok: boolean;
  message: string;
  comfyuiVersion?: string;
  devices?: Array<{ name: string; vramTotalMb: number; vramFreeMb: number }>;
  models?: Partial<Record<ProbeModelFolder, string[]>>;
  nodes?: Record<string, boolean>;
  queueRemaining?: number;
}

const PHOTOREAL_ERROR_PREFIX = "실사 이미지 생성 실패: ";
const PHOTOREAL_RELAY_PATH = "/photoreal/comfyui";
/**
 * Seeds stay in 0..2^31-1. ComfyUI's INT widgets take far more, but a 31-bit seed is exact on every path it
 * travels (JSON, JS numbers, custom nodes that seed numpy, which refuses anything from 2^32 up), so the seed
 * shown in the inspector always reproduces the image.
 */
const PHOTOREAL_SEED_MODULUS = 2 ** 31;
/** Same range normalizePhotorealSettings keeps timeoutSeconds in; the relay caps a wait at 900 s. */
const DEFAULT_TIMEOUT_SECONDS = 300;
const MIN_TIMEOUT_SECONDS = 30;
const MAX_TIMEOUT_SECONDS = 900;
/**
 * The relay's wait (timeoutMs) always ends before this client gives up, so on a slow model load the user gets
 * the relay's explanation (and ComfyUI gets an interrupt) instead of a bare client-side abort.
 */
const CLIENT_TIMEOUT_MARGIN_SECONDS = 30;
/**
 * 502 means the relay could not reach ComfyUI and 503 that it could not take the job; in both cases nothing
 * was queued, so one retry cannot render a cut twice. Everything else (a rejected graph, a CUDA OOM, a
 * timeout) would fail again the same way.
 */
const RETRYABLE_RELAY_STATUS = new Set([502, 503]);
const MAX_ATTEMPTS = 2;
const RETRY_DELAY_MS = 1_500;
/** status asks ComfyUI several things (system stats, four model folders, up to 30 node definitions). */
const PROBE_TIMEOUT_MS = 20_000;
const PROBE_MAX_NODES = 30;
/** free / interrupt are fire-and-forget; they must never hold up the turn or the queue that called them. */
const BEST_EFFORT_TIMEOUT_MS = 10_000;
const BASE64_PATTERN = /^[A-Za-z0-9+/]+={0,2}$/u;
const IMAGE_MIME_PATTERN = /^image\/[a-z0-9.+-]+$/iu;
/**
 * The file name the API server gives a picked profile image: `<simulationId>/<createId("profile")>.<ext>`. The same
 * shape the server's isProfileObjectKey keeps (server/dynamicchat-server.mjs), so a key passes here exactly when the
 * server would treat it as a profile pick. Story cuts are `asset_*` and nothing else is written by a pick.
 */
const PROFILE_OBJECT_FILE_PATTERN = /^profile_[A-Za-z0-9._-]+\.(?:png|jpg|jpeg|webp)$/u;
const OBJECT_PATH_SEGMENT_PATTERN = /^[A-Za-z0-9._-]+$/u;

/**
 * Whether `objectKey` names a profile image the user picked from candidates DynamicChat generated — the only image
 * that may be a photoreal face reference. A benign recorded prompt vouches for nothing if the key could point at an
 * arbitrary file (a story cut, an upload, a hand-edited state), so the key's shape is checked as well as the prompt.
 */
export function isPickedProfileObjectKey(objectKey: unknown): boolean {
  const segments = typeof objectKey === "string" ? objectKey.split("/") : [];
  return (
    segments.length === 2 &&
    segments.every((segment) => segment !== "." && segment !== ".." && OBJECT_PATH_SEGMENT_PATTERN.test(segment)) &&
    PROFILE_OBJECT_FILE_PATTERN.test(segments[1])
  );
}

class PhotorealRelayError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryable: boolean
  ) {
    super(message);
    this.name = "PhotorealRelayError";
  }
}

interface PreparedPhotorealRequest {
  built: BuiltComfyWorkflow;
  references: PhotorealReferenceInput[];
  requestId: string;
  payload: Record<string, unknown>;
}

interface RelayGeneration {
  dataUrls: string[];
  promptId?: string;
  elapsedMs?: number;
}

/**
 * Renders one photoreal cut: builds the ComfyUI graph for the preset and runs it through the DynamicChat
 * relay (ComfyUI refuses cross-origin browser calls, and the relay is what uploads the face references).
 *
 * Disabled / mock is a DRY RUN, as with NovelAI: the built graph comes back in the payload with no images and
 * no network call, so the planner, the inspector and the evals can see exactly what would have been queued.
 *
 * `cue` is accepted for parity with generateNovelAiImages; the prompt arriving here is already the final
 * natural-language text composed from it, so nothing below reads the cue.
 *
 * `inspectWorkflow` sees the graph exactly as it will leave — this object, not a rebuild of it — before anything is
 * sent, the dry run included; throwing stops the request. The orchestrator's safety floor reads it there, which this
 * module (transport only) does not import.
 */
export async function generatePhotorealImages(input: {
  state: AppState;
  prompt: string;
  negativePrompt: string;
  cue: ImageCue;
  presetId: PhotorealWorkflowPresetId;
  references: PhotorealReferenceInput[];
  jobId?: string;
  seed?: number;
  signal?: AbortSignal;
  inspectWorkflow?: (built: BuiltComfyWorkflow) => void;
}): Promise<PhotorealImageResult> {
  const settings = readPhotorealSettings(input.state);
  const request = preparePhotorealRequest(input);
  input.inspectWorkflow?.(request.built);
  if (!settings.enabled || settings.requestMode === "mock") {
    return { dataUrls: [], payload: request.payload };
  }

  const relayBase = getPhotorealRelayBase(input.state);
  const timeoutSeconds = resolveTimeoutSeconds(settings.timeoutSeconds);
  const body = JSON.stringify({
    baseUrl: settings.baseUrl.trim() || undefined,
    workflow: request.built.workflow,
    outputNodeIds: request.built.outputNodeIds,
    references: request.references.map(toRelayReference),
    timeoutMs: timeoutSeconds * 1000,
    clientRequestId: request.requestId
  });

  for (let attempt = 1; ; attempt += 1) {
    try {
      const generation = await postRelayGeneration(relayBase, body, timeoutSeconds, input.signal);
      // The full graph is a few KB per cut and is reproducible from the prompt, preset, size and seed kept
      // beside it, so the stored copy (providerMetadata → state.json) keeps only its outline.
      const { workflow: _workflow, ...storedPayload } = request.payload;
      return {
        dataUrls: generation.dataUrls,
        payload: {
          ...storedPayload,
          promptId: generation.promptId,
          elapsedMs: generation.elapsedMs,
          mode: "provider-result"
        }
      };
    } catch (error) {
      const retryable = error instanceof PhotorealRelayError && error.retryable;
      if (!retryable || attempt >= MAX_ATTEMPTS || input.signal?.aborted) {
        throw error;
      }
      await delay(RETRY_DELAY_MS, input.signal);
    }
  }
}

/**
 * Asks the relay what the local ComfyUI has: version, GPU memory, installed model files and whether the nodes
 * the configured presets use exist. Works whether or not photoreal is enabled, because it is how the user
 * checks the setup before turning it on. Never throws; every failure is `ok: false` with a Korean message.
 */
export async function probeComfyUi(state: AppState, baseUrl?: string): Promise<ComfyUiProbeResult> {
  let relayBase = "";
  try {
    const settings = readPhotorealSettings(state);
    relayBase = getPhotorealRelayBase(state);
    const query = new URLSearchParams();
    const target = (baseUrl ?? settings.baseUrl).trim();
    if (target) {
      query.set("baseUrl", target);
    }
    const nodes = collectProbeNodes(settings);
    if (nodes.length > 0) {
      query.set("nodes", nodes.join(","));
    }
    const queryString = query.toString();
    const response = await fetch(`${relayBase}/status${queryString ? `?${queryString}` : ""}`, {
      method: "GET",
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS)
    });
    const parsed: unknown = await response.json().catch(() => undefined);
    if (!response.ok) {
      return {
        ok: false,
        message: `ComfyUI 연결 확인 실패: ${describeRelayHttpFailure(response.status, readErrorField(parsed), `${relayBase}/status`)}`
      };
    }
    if (!isRecord(parsed)) {
      return { ok: false, message: "ComfyUI 연결 확인 실패: 릴레이 응답을 해석하지 못했습니다." };
    }
    return normalizeProbeResult(parsed);
  } catch (error) {
    return { ok: false, message: describeProbeFailure(error, relayBase) };
  }
}

/**
 * Unloads ComfyUI's models and frees its VRAM, so a local LLM can have the GPU back (see vramHandoff).
 * Best effort: never throws, and gives up after a few seconds rather than hold up the turn waiting on it.
 */
export async function freeComfyUiMemory(state: AppState): Promise<void> {
  await postBestEffort(state, "free", {});
}

/** Stops a running photoreal job and drops it from ComfyUI's queue. Best effort, never throws. */
export async function interruptComfyUi(state: AppState, promptId: string): Promise<void> {
  const id = typeof promptId === "string" ? promptId.trim() : "";
  if (!id) {
    return;
  }
  await postBestEffort(state, "interrupt", { promptId: id });
}

/** `state.photoreal.proxyUrl` when set, else `${configured API base}/photoreal/comfyui`. */
export function getPhotorealRelayBase(state: AppState): string {
  const override = readPhotorealSettings(state).proxyUrl.trim();
  return normalizeRelayBase(override || readDefaultRelayBase());
}

/**
 * The graph a cut would be sent as, built with no network call: the same preparation generatePhotorealImages runs, so
 * the planner can hold the BUILT text to the floor before the cut is ever queued — and learn at plan time, not at
 * execution, that a custom template cannot build at all. Throws the builder's Korean error ("실사 이미지 생성 실패: …").
 */
export function buildPhotorealRequestWorkflow(input: {
  state: AppState;
  prompt: string;
  negativePrompt: string;
  presetId: PhotorealWorkflowPresetId;
  references: PhotorealReferenceInput[];
  jobId?: string;
  seed?: number;
}): BuiltComfyWorkflow {
  return preparePhotorealRequest(input).built;
}

/** A state that never went through hydrateState (an old export, a hand-built fixture) still works, on the defaults. */
function readPhotorealSettings(state: AppState): PhotorealApiSettings {
  return state.photoreal ?? DEFAULT_PHOTOREAL_SETTINGS;
}

function preparePhotorealRequest(input: {
  state: AppState;
  prompt: string;
  negativePrompt: string;
  presetId: PhotorealWorkflowPresetId;
  references: PhotorealReferenceInput[];
  jobId?: string;
  seed?: number;
}): PreparedPhotorealRequest {
  const settings = readPhotorealSettings(input.state);
  const candidates = selectSendableReferences(input.references);
  // A job planned before its preset existed (or edited by hand) can carry an unknown id; resolve it the way the
  // planner would instead of failing on a lookup.
  const presetId = isPhotorealPresetId(input.presetId)
    ? input.presetId
    : resolvePhotorealPresetForCut(settings, candidates.length > 0);
  const preset = PHOTOREAL_WORKFLOW_PRESETS[presetId];
  const { width, height } = resolvePhotorealOutputSize(input.state);
  const seed = resolvePhotorealSeed(settings, input.seed);
  const steps = Math.round(positiveOr(settings.steps, preset.defaults.steps));
  const cfg = resolvePhotorealCfg(settings, presetId);
  const files = resolvePhotorealModelFiles(settings, presetId);
  const requestId = toSafeRequestId(input.jobId) ?? createId("photoreal");

  // buildComfyWorkflow validates every graph it returns (a hand-written custom one included) and explains a bad
  // one in Korean, so a broken graph fails here, before the GPU loads a model, rather than as a ComfyUI 400.
  let built: BuiltComfyWorkflow;
  try {
    built = buildComfyWorkflow(
      presetId,
      {
        prompt: input.prompt,
        negativePrompt: input.negativePrompt,
        width,
        height,
        seed,
        steps,
        cfg,
        files,
        referenceNames: candidates.map((reference) => reference.name),
        referenceCrops: Object.fromEntries(
          candidates.flatMap((reference) => {
            const crop = computeReferenceHeadCrop(reference.width, reference.height);
            return crop ? [[reference.name, crop] as const] : [];
          })
        ),
        filenamePrefix: `DynamicChat/${requestId}`
      },
      presetId === "custom" ? settings.customWorkflow : undefined
    );
  } catch (error) {
    throw new Error(withPhotorealPrefix(describeError(error)));
  }

  // Only references the graph actually wires are uploaded: a preset without reference inputs (Z-Image) or one
  // at its reference cap must not ship a face it will never look at.
  const usedNames = new Set(built.usedReferences);
  const references = candidates.filter((reference) => usedNames.has(reference.name));

  return {
    built,
    references,
    requestId,
    payload: {
      provider: "photoreal",
      promptFormat: "natural-language",
      presetId: built.presetId,
      workflow: built.workflow,
      workflowSummary: summarizeWorkflow(built),
      outputNodeIds: built.outputNodeIds,
      // Names and object keys only. The bytes stay in the object store; this payload is copied into
      // providerMetadata and state.json for every cut.
      references: references.map(describeReference),
      width,
      height,
      seed,
      steps,
      cfg,
      modelFiles: files
    }
  };
}

async function postRelayGeneration(
  relayBase: string,
  body: string,
  timeoutSeconds: number,
  signal?: AbortSignal
): Promise<RelayGeneration> {
  const clientTimeoutSeconds = timeoutSeconds + CLIENT_TIMEOUT_MARGIN_SECONDS;
  const timeout = AbortSignal.timeout(clientTimeoutSeconds * 1000);
  const url = `${relayBase}/generate`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
      signal: combineSignals(signal, timeout)
    });

    if (!response.ok) {
      const detail = await readRelayErrorMessage(response);
      throw new PhotorealRelayError(
        withPhotorealPrefix(describeRelayHttpFailure(response.status, detail, url)),
        response.status,
        RETRYABLE_RELAY_STATUS.has(response.status)
      );
    }

    let parsed: unknown;
    try {
      parsed = await response.json();
    } catch (error) {
      if (signal?.aborted || timeout.aborted) {
        throw error;
      }
      throw new PhotorealRelayError(withPhotorealPrefix("릴레이 응답을 해석하지 못했습니다."), response.status, false);
    }

    const dataUrls = readRelayImages(parsed);
    if (dataUrls.length === 0) {
      throw new PhotorealRelayError(
        withPhotorealPrefix("ComfyUI가 결과 이미지를 돌려주지 않았습니다. 워크플로에 SaveImage 출력 노드가 있는지 확인하세요."),
        response.status,
        false
      );
    }
    const record = isRecord(parsed) ? parsed : {};
    return {
      dataUrls,
      promptId: typeof record.promptId === "string" ? record.promptId : undefined,
      elapsedMs: typeof record.elapsedMs === "number" && Number.isFinite(record.elapsedMs) ? record.elapsedMs : undefined
    };
  } catch (error) {
    if (error instanceof PhotorealRelayError) {
      throw error;
    }
    // A cancel from the caller stays an AbortError so the queue can tell "cancelled" from "failed".
    if (signal?.aborted) {
      throw error;
    }
    if (timeout.aborted) {
      throw new PhotorealRelayError(
        withPhotorealPrefix(`${clientTimeoutSeconds}초 안에 응답이 없어 요청을 중단했습니다. 첫 실행은 모델을 불러오느라 오래 걸릴 수 있으니 설정에서 대기 시간을 늘려 보세요.`),
        0,
        false
      );
    }
    throw new PhotorealRelayError(
      withPhotorealPrefix(`DynamicChat API 서버에 연결할 수 없습니다 (${relayBase}). API 서버를 실행했는지 확인하세요.`),
      0,
      true
    );
  }
}

async function postBestEffort(state: AppState, action: "free" | "interrupt", extra: Record<string, unknown>): Promise<void> {
  const settings = readPhotorealSettings(state);
  // In dry-run mode nothing was ever sent to ComfyUI, so there is nothing to free or stop.
  if (!settings.enabled || settings.requestMode === "mock") {
    return;
  }
  try {
    const response = await fetch(`${getPhotorealRelayBase(state)}/${action}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ baseUrl: settings.baseUrl.trim() || undefined, ...extra }),
      signal: AbortSignal.timeout(BEST_EFFORT_TIMEOUT_MS)
    });
    await response.text().catch(() => "");
  } catch {
    // Best effort by contract: a ComfyUI that is already idle, restarting or gone needs nothing from us.
  }
}

function readDefaultRelayBase(): string {
  try {
    return getPhotorealRelayBaseUrl();
  } catch {
    // The configured API base lives in localStorage. Node has none (handled there), but a sandboxed or
    // storage-blocked browser context throws a SecurityError on access; that must not fail a cut or a probe.
    return `${DEFAULT_DYNAMICCHAT_API_BASE_URL}${PHOTOREAL_RELAY_PATH}`;
  }
}

function normalizeRelayBase(value: string): string {
  // People paste the endpoint they saw in the network tab (…/photoreal/comfyui/generate); the base is the
  // part before the action.
  return value
    .trim()
    .replace(/\/+$/u, "")
    .replace(/\/(?:generate|status|free|interrupt)$/u, "");
}

function selectSendableReferences(references: PhotorealReferenceInput[]): PhotorealReferenceInput[] {
  const seen = new Set<string>();
  const selected: PhotorealReferenceInput[] = [];
  for (const reference of Array.isArray(references) ? references : []) {
    if (!isRecord(reference)) {
      continue;
    }
    const name = typeof reference.name === "string" ? reference.name.trim() : "";
    const objectKey = typeof reference.objectKey === "string" ? reference.objectKey.trim() : "";
    // Only a picked profile image, by its object key. Bytes that ride in on a reference (a data URL) are never sent,
    // whoever put them there: an image the relay uploaded from a caller's bytes and rendered back would be a "render"
    // DynamicChat never made — and as a profile candidate it could be picked as a face. A key that names anything but
    // a pick is dropped the same way, and the cut is drawn from the prompt alone (an unfilled `%ref:%` would fail it).
    if (!name || seen.has(name) || !isPickedProfileObjectKey(objectKey)) {
      continue;
    }
    seen.add(name);
    selected.push({
      name,
      objectKey,
      ...(reference.characterId ? { characterId: reference.characterId } : {}),
      // Kept so the graph can crop the reference to the head (computeReferenceHeadCrop); never sent to the relay.
      ...(Number.isFinite(reference.width) && Number.isFinite(reference.height) ? { width: reference.width, height: reference.height } : {})
    });
  }
  return selected;
}

function toRelayReference(reference: PhotorealReferenceInput): Record<string, string> {
  return { name: reference.name, objectKey: reference.objectKey };
}

function describeReference(reference: PhotorealReferenceInput): Record<string, string> {
  return {
    name: reference.name,
    objectKey: reference.objectKey,
    ...(reference.characterId ? { characterId: reference.characterId } : {})
  };
}

function summarizeWorkflow(built: BuiltComfyWorkflow): { presetId: PhotorealWorkflowPresetId; nodeCount: number; classTypes: string[] } {
  const nodes = Object.values(built.workflow);
  return {
    presetId: built.presetId,
    nodeCount: nodes.length,
    classTypes: Array.from(new Set(nodes.map((node) => node.class_type)))
  };
}

function isPhotorealPresetId(value: unknown): value is PhotorealWorkflowPresetId {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(PHOTOREAL_WORKFLOW_PRESETS, value);
}

function resolvePhotorealSeed(settings: PhotorealApiSettings, explicit?: number): number {
  const requested = normalizeSeed(explicit);
  if (requested !== undefined) {
    return requested;
  }
  if (settings.seedFixed) {
    const fixed = normalizeSeed(settings.seed);
    if (fixed !== undefined) {
      return fixed;
    }
  }
  return Math.floor(Math.random() * PHOTOREAL_SEED_MODULUS);
}

function normalizeSeed(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  // Folded rather than clamped, so distinct out-of-range seeds stay distinct (and stay reproducible).
  return Math.abs(Math.trunc(value)) % PHOTOREAL_SEED_MODULUS;
}

function positiveOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

function resolveTimeoutSeconds(value: unknown): number {
  const seconds = typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.round(value) : DEFAULT_TIMEOUT_SECONDS;
  return Math.min(MAX_TIMEOUT_SECONDS, Math.max(MIN_TIMEOUT_SECONDS, seconds));
}

/** The job id becomes ComfyUI's output file prefix, a path on the GPU host, so only filename-safe characters survive. */
function toSafeRequestId(jobId?: string): string | undefined {
  const cleaned = typeof jobId === "string" ? jobId.trim().replace(/[^A-Za-z0-9_-]+/gu, "_").slice(0, 96) : "";
  return cleaned || undefined;
}

function combineSignals(signal: AbortSignal | undefined, timeout: AbortSignal): AbortSignal {
  if (!signal) {
    return timeout;
  }
  // Without AbortSignal.any keep the caller's cancel: the relay enforces its own (shorter) timeout, so the
  // request still ends, while dropping the cancel would leave a job the user stopped running to completion.
  return typeof AbortSignal.any === "function" ? AbortSignal.any([signal, timeout]) : signal;
}

function readRelayImages(parsed: unknown): string[] {
  if (!isRecord(parsed) || !Array.isArray(parsed.images)) {
    return [];
  }
  return parsed.images.flatMap((image) => {
    if (!isRecord(image) || typeof image.base64 !== "string") {
      return [];
    }
    const base64 = image.base64.replace(/\s+/gu, "");
    if (!base64 || !BASE64_PATTERN.test(base64)) {
      return [];
    }
    const mimeType = typeof image.mimeType === "string" && IMAGE_MIME_PATTERN.test(image.mimeType)
      ? image.mimeType.toLowerCase()
      : "image/png";
    // A data URL is all the rest of the pipeline needs; no FileReader/Buffer round trip, so this is the same
    // in the browser and in Node.
    return [`data:${mimeType};base64,${base64}`];
  });
}

async function readRelayErrorMessage(response: Response): Promise<string> {
  const text = (await response.text().catch(() => "")).trim();
  // An HTML page means the URL reached some other web server (a proxyUrl pointing at the Vite dev server, say);
  // its markup is no message for a toast.
  if (!text || text.startsWith("<")) {
    return "";
  }
  try {
    const field = readErrorField(JSON.parse(text));
    if (field) {
      return field;
    }
  } catch {
    // Plain-text error from something in between; fall through to the raw text.
  }
  return text.slice(0, 500);
}

function readErrorField(parsed: unknown): string | undefined {
  if (!isRecord(parsed)) {
    return undefined;
  }
  for (const key of ["error", "message"]) {
    const value = parsed[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
}

function describeRelayHttpFailure(status: number, detail: string | undefined, url: string): string {
  // A bare 404 means the relay route itself is missing — an API server from before the photoreal relay existed,
  // or a proxyUrl that points somewhere else — and "Not found" alone would not tell the user which. A 404 that
  // carries its own message is the relay talking, so that message is kept.
  if (status === 404 && (!detail || /^not found\.?$/iu.test(detail))) {
    return `실사 릴레이 경로를 찾을 수 없습니다 (${url}). API 서버를 최신 버전으로 다시 실행하거나 프록시 주소를 확인하세요.`;
  }
  return detail || `릴레이가 HTTP ${status} 오류를 돌려주었습니다.`;
}

function describeProbeFailure(error: unknown, relayBase: string): string {
  if (error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError")) {
    return `ComfyUI 연결 확인이 ${Math.round(PROBE_TIMEOUT_MS / 1000)}초 안에 끝나지 않았습니다. ComfyUI가 시작 중이면 잠시 뒤 다시 시도하세요.`;
  }
  return `DynamicChat API 서버에 연결할 수 없습니다 (${relayBase || DEFAULT_DYNAMICCHAT_API_BASE_URL}). API 서버를 실행했는지 확인하세요.`;
}

function normalizeProbeResult(record: Record<string, unknown>): ComfyUiProbeResult {
  const result: ComfyUiProbeResult = { ok: record.ok === true, message: "" };
  if (typeof record.comfyuiVersion === "string" && record.comfyuiVersion.trim()) {
    result.comfyuiVersion = record.comfyuiVersion.trim();
  }
  const devices = readProbeDevices(record.devices);
  if (devices) {
    result.devices = devices;
  }
  const models = readProbeModels(record.models);
  if (models) {
    result.models = models;
  }
  const nodes = readProbeNodes(record.nodes);
  if (nodes) {
    result.nodes = nodes;
  }
  if (typeof record.queueRemaining === "number" && Number.isFinite(record.queueRemaining)) {
    result.queueRemaining = record.queueRemaining;
  }
  result.message = readErrorField(record) ?? composeProbeMessage(result);
  return result;
}

function readProbeDevices(value: unknown): ComfyUiProbeResult["devices"] {
  if (!Array.isArray(value)) {
    return undefined;
  }
  return value.filter(isRecord).map((device) => ({
    name: typeof device.name === "string" ? device.name : "",
    vramTotalMb: toFiniteNumber(device.vramTotalMb),
    vramFreeMb: toFiniteNumber(device.vramFreeMb)
  }));
}

function readProbeModels(value: unknown): ComfyUiProbeResult["models"] {
  if (!isRecord(value)) {
    return undefined;
  }
  const models: NonNullable<ComfyUiProbeResult["models"]> = {};
  for (const folder of PROBE_MODEL_FOLDERS) {
    const files = value[folder];
    if (Array.isArray(files)) {
      models[folder] = files.filter((file): file is string => typeof file === "string");
    }
  }
  return models;
}

function readProbeNodes(value: unknown): Record<string, boolean> | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  return Object.fromEntries(Object.entries(value).map(([name, present]) => [name, present === true]));
}

function composeProbeMessage(result: ComfyUiProbeResult): string {
  if (!result.ok) {
    return "ComfyUI에 연결하지 못했습니다.";
  }
  const version = result.comfyuiVersion ? ` ${result.comfyuiVersion}` : "";
  const missing = Object.entries(result.nodes ?? {})
    .filter(([, present]) => !present)
    .map(([name]) => name);
  const missingNote = missing.length > 0
    ? ` 설치되지 않은 노드: ${missing.join(", ")} — ComfyUI를 최신 버전으로 업데이트하거나 해당 커스텀 노드를 설치하세요.`
    : "";
  return `ComfyUI${version}에 연결되었습니다.${missingNote}`;
}

/** The node types the configured presets need, so the probe can say "update ComfyUI" before a cut fails on it. */
function collectProbeNodes(settings: PhotorealApiSettings): string[] {
  const nodes = new Set<string>();
  // The face-lock preset only runs while face lock is on (resolvePhotorealPresetForCut). Probing it regardless told a
  // Z-Image-only machine that Qwen-Image 2.1's nodes were missing, with face lock switched off.
  const presetIds = settings.faceReferenceEnabled ? [settings.workflowPreset, settings.referenceWorkflowPreset] : [settings.workflowPreset];
  for (const presetId of new Set(presetIds)) {
    if (!isPhotorealPresetId(presetId)) {
      continue;
    }
    for (const node of PHOTOREAL_WORKFLOW_PRESETS[presetId].requiredNodes) {
      nodes.add(node);
    }
    if (presetId === "custom") {
      for (const node of readCustomWorkflowClassTypes(settings.customWorkflow)) {
        nodes.add(node);
      }
    }
  }
  // The list travels comma-separated in the query string, and the relay checks at most 30.
  return Array.from(nodes)
    .filter((node) => node.length > 0 && node.length <= 128 && !node.includes(","))
    .slice(0, PROBE_MAX_NODES);
}

function readCustomWorkflowClassTypes(workflowJson: string): string[] {
  try {
    const parsed: unknown = JSON.parse(workflowJson);
    if (!isRecord(parsed)) {
      return [];
    }
    return Object.values(parsed).flatMap((node) =>
      isRecord(node) && typeof node.class_type === "string" ? [node.class_type] : []
    );
  } catch {
    // An unparseable custom graph is reported by validateComfyApiWorkflow in settings; the probe just skips it.
    return [];
  }
}

function withPhotorealPrefix(message: string): string {
  return message.startsWith(PHOTOREAL_ERROR_PREFIX) ? message : `${PHOTOREAL_ERROR_PREFIX}${message}`;
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function toFiniteNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function delay(durationMs: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      resolve();
    };
    const timer = setTimeout(finish, durationMs);
    signal?.addEventListener("abort", finish, { once: true });
  });
}
