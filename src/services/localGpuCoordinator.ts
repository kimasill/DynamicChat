import type { AppState, LlmApiSettings } from "../types";
import { getPhotorealRelayBase } from "./comfyUiClient";

// Local GPU hand-over between a local LLM (Ollama) and the local photoreal model (ComfyUI).
//
// A 16 GB card holds one of them, not both: qwen3:14b at 16k context takes about 11.7 GB and Qwen-Image 2.1 Turbo
// 13-15 GB. Neither server knows about the other, so when both are resident the second one to load spills into
// system RAM — and a spilled LLM was measured at 0.1 tok/s against 55 tok/s fully on the card, while Ollama's own
// size_vram still reported "100% GPU" (CLAUDE.md, "VRAM 이 전부다"). Nothing errors; the turn just crawls.
//
// So turns and photoreal jobs share one FIFO lock, and whenever the GPU changes hands the previous holder is
// asked to unload first: Ollama through keep_alive 0, ComfyUI through its /free endpoint (via the relay). Both are
// best effort — a server that is down or already idle needs nothing, and a failed hand-over must never stop the
// work it was clearing room for.
//
// The VRAM belongs to the MACHINE, not to the simulation on screen. Photoreal and LLM settings are per simulation, so
// the state a turn runs with can say "photoreal off" while ComfyUI still holds the model an earlier simulation's
// render loaded, and a render's state can name another Ollama model (or none) than the one an earlier simulation's
// turn left on the card. The coordinator therefore remembers what it saw load — the ComfyUI a render used, the
// Ollama models a turn used, coordinated or not — and hands the card over from THOSE, whatever the current state
// says.

export type LocalGpuWorkKind = "llm" | "image";

export interface LocalGpuHandoffEvent {
  from: LocalGpuWorkKind;
  to: LocalGpuWorkKind;
  /** Korean, for a runtime notice or a log line. */
  detail: string;
}

export interface LocalGpuRunOptions<T> {
  /**
   * Asked once this task's turn on the GPU has come, BEFORE anything is unloaded for it; false skips the hand-over
   * and the task. A render can wait a minute behind a local LLM turn, and the cut it was queued for may have been
   * cancelled, regenerated or left behind meanwhile: evicting the LLM for it would cost the next turn a model reload
   * for a picture nobody is waiting for.
   */
  shouldRun?: () => boolean;
  /** What a skipped run resolves to. Without it a skipped run rejects with an AbortError. */
  onSkip?: () => T;
}

export interface LocalGpuCoordinator {
  /**
   * Runs `task` holding the local GPU when it loads a model on this machine's card (a turn on a local Ollama, a
   * render on a local ComfyUI, unless its simulation turned the hand-over off), after handing the GPU over from the
   * previous holder; otherwise runs it straight away. Tasks run one
   * at a time in call order. Not re-entrant: a task must never await another run() on the same coordinator, or both
   * wait forever.
   */
  run<T>(kind: LocalGpuWorkKind, state: AppState, task: () => Promise<T>, options?: LocalGpuRunOptions<T>): Promise<T>;
  /**
   * The kind of work whose model was last loaded on this machine's GPU as far as this coordinator saw: a render on a
   * local ComfyUI, or a turn on a local Ollama, coordinated or not. Undefined before the first such task.
   */
  readonly holder: LocalGpuWorkKind | undefined;
}

/** A ComfyUI on this machine, as the relay reaches it. */
export interface LocalComfyUiTarget {
  /** The DynamicChat relay base (`…/photoreal/comfyui`). */
  relayBase: string;
  /** The ComfyUI address the relay forwards to; undefined means the relay's own default. */
  baseUrl?: string;
}

/** keep_alive 0 answers as soon as the model is out of memory; anything slower means Ollama is not answering. */
const OLLAMA_UNLOAD_TIMEOUT_MS = 5_000;
/** ComfyUI's /free returns once the unload is queued; the relay adds one loopback hop. */
const COMFYUI_FREE_TIMEOUT_MS = 10_000;
/** The ComfyUI relay listens on loopback by default; this is where an empty photoreal.baseUrl ends up. */
const DEFAULT_COMFYUI_BASE_URL = "http://127.0.0.1:8188";

/**
 * Whether turns and photoreal jobs must take turns on the GPU: photoreal is on and goes through the relay to a
 * ComfyUI on this machine, the hand-over is set to auto, and the narrative or tag LLM is an Ollama on this
 * machine. Anything remote does not share this card, and mock mode never touches a GPU.
 */
export function shouldCoordinateLocalGpu(state: AppState): boolean {
  return rendersOnLocalComfyUi(state) && state.photoreal?.vramHandoff === "auto" && readLocalOllamaTargets(state).length > 0;
}

/**
 * The ComfyUI on this machine that `state` renders with: photoreal on, through the relay (mock never touches a
 * GPU), at a loopback address. Undefined for anything else, and for a state whose relay address cannot be read.
 */
export function readLocalComfyUiTarget(state: AppState): LocalComfyUiTarget | undefined {
  if (!rendersOnLocalComfyUi(state)) {
    return undefined;
  }
  let relayBase: string;
  try {
    relayBase = getPhotorealRelayBase(state);
  } catch {
    // A state that never went through hydrateState can lack the relay field; such a render cannot be freed later.
    return undefined;
  }
  const baseUrl = readComfyUiBaseUrl(state);
  return { relayBase, baseUrl: baseUrl || undefined };
}

/** One model on one local Ollama server. */
export interface LocalOllamaTarget {
  /** The server root that /api/* hangs off. */
  apiRoot: string;
  model: string;
}

/**
 * The Ollama servers and models this app runs on the local GPU: the narrative LLM and the image-tag LLM, each
 * when it is an enabled Ollama provider on a loopback host with a model chosen. Deduplicated, because the two
 * usually point at the same server and often at the same model.
 */
export function readLocalOllamaTargets(state: AppState): LocalOllamaTarget[] {
  return mergeOllamaTargets([state.llm, state.imageTagLlm].map(readOllamaTarget).filter((target): target is LocalOllamaTarget => Boolean(target)));
}

/**
 * Asks each local Ollama `state` names to drop its model from VRAM. Never throws; returns the models Ollama
 * confirmed. See unloadOllamaTargets.
 */
export async function unloadOllamaModels(state: AppState, fetchImpl: typeof fetch = fetch): Promise<string[]> {
  return unloadOllamaTargets(readLocalOllamaTargets(state), fetchImpl);
}

/**
 * Asks each Ollama to drop its model from VRAM (`POST /api/generate {model, keep_alive: 0}` with no prompt unloads
 * without generating). Never throws; returns the models Ollama confirmed. One target is tried after the other, so
 * two models on one server are not unloaded in a race with each other's reload.
 */
async function unloadOllamaTargets(targets: LocalOllamaTarget[], fetchImpl: typeof fetch): Promise<string[]> {
  const unloaded: string[] = [];
  for (const target of targets) {
    try {
      const response = await fetchImpl(`${target.apiRoot}/api/generate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: target.model, keep_alive: 0 }),
        signal: AbortSignal.timeout(OLLAMA_UNLOAD_TIMEOUT_MS)
      });
      // Drained so the connection is released; the body (done_reason "unload") says nothing the status does not.
      await response.text().catch(() => "");
      if (response.ok) {
        unloaded.push(target.model);
      }
    } catch {
      // Best effort by contract: an Ollama that is stopped or restarting holds no VRAM worth freeing.
    }
  }
  return unloaded;
}

/**
 * Asks the relay to make one ComfyUI unload its models (`POST {relayBase}/free`). Called with the target the last
 * render used rather than through comfyUiClient.freeComfyUiMemory, whose photoreal-enabled guard reads the state it
 * is handed: the turn freeing the card may belong to a simulation with photoreal off, while the model sitting in
 * VRAM was loaded by another one. Never throws; true when the relay accepted the request.
 */
export async function freeLocalComfyUi(target: LocalComfyUiTarget, fetchImpl: typeof fetch = fetch): Promise<boolean> {
  try {
    const response = await fetchImpl(`${target.relayBase}/free`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ baseUrl: target.baseUrl }),
      signal: AbortSignal.timeout(COMFYUI_FREE_TIMEOUT_MS)
    });
    await response.text().catch(() => "");
    return response.ok;
  } catch {
    // Best effort by contract: a ComfyUI that is already idle, restarting or gone needs nothing from us.
    return false;
  }
}

export function createLocalGpuCoordinator(
  options: { onHandoff?: (event: LocalGpuHandoffEvent) => void; fetchImpl?: typeof fetch } = {}
): LocalGpuCoordinator {
  // The tail of the queue: every coordinated task chains onto it, so tasks start in call order and never overlap.
  // It always settles fulfilled, so one task's failure releases the lock instead of jamming every later task.
  let tail: Promise<void> = Promise.resolve();
  let holder: LocalGpuWorkKind | undefined;
  // The ComfyUI that may still hold a model: set by every render on a local ComfyUI, cleared once it was asked to
  // free. Kept apart from `holder` because both can be resident at once — a local-Ollama turn in a simulation whose
  // hand-over is off loads its model without evicting ComfyUI, and the next coordinated turn must still free it.
  let residentComfyUi: LocalComfyUiTarget | undefined;
  // The same for the other side: every local Ollama model a turn ran on since the last unload, keyed by server and
  // model. The render that takes the card next unloads these, not only the models its own state names — the turn
  // may have run in another simulation, on another model, and a render whose simulation uses a hosted LLM names
  // none at all.
  const residentOllama = new Map<string, LocalOllamaTarget>();
  const fetchImpl = options.fetchImpl ?? ((input: RequestInfo | URL, init?: RequestInit) => fetch(input, init));

  const notify = (event: LocalGpuHandoffEvent): void => {
    try {
      options.onHandoff?.(event);
    } catch {
      // A broken listener must not cost the task its turn on the GPU.
    }
  };

  const handOver = async (to: LocalGpuWorkKind, state: AppState): Promise<void> => {
    // LLM → image, including the very first image job: an unknown holder is treated as the LLM, because a model
    // loaded before this coordinator existed (a turn before a page reload, a run of an eval) still sits on the card.
    // A remembered model counts even when the last thing seen was a render: a render whose simulation turned the
    // hand-over off left it loaded.
    if (to === "image" && (holder !== "image" || residentOllama.size > 0)) {
      const targets = mergeOllamaTargets([...residentOllama.values(), ...readLocalOllamaTargets(state)]);
      // Cleared whatever the answer, like the ComfyUI side: an Ollama that cannot be reached holds nothing.
      residentOllama.clear();
      if (targets.length === 0) {
        // A render whose simulation uses a hosted LLM, with no local turn seen: nothing to take back, nothing to say.
        return;
      }
      const models = await unloadOllamaTargets(targets, fetchImpl).catch(() => [] as string[]);
      notify({
        from: "llm",
        to,
        detail:
          models.length > 0
            ? `Ollama 모델을 VRAM에서 내렸습니다 (${models.join(", ")}). 실사 이미지를 생성합니다.`
            : "Ollama 모델을 내리지 못했거나 이미 내려가 있습니다. 실사 이미지 생성을 계속합니다."
      });
      return;
    }
    // image → LLM: the ComfyUI the last render used, whichever simulation it rendered for. Before the first task of
    // any kind, the ComfyUI this state renders with is freed too, mirroring the unknown-holder rule above: a model a
    // render loaded before this coordinator existed (a page reload, an earlier session) still sits on the card, and
    // a 16k qwen3:14b beside it was measured timing out its annotation call after 157 s at 15.6 of 16.4 GB.
    const target = to === "llm" ? (residentComfyUi ?? (holder === undefined ? readLocalComfyUiTarget(state) : undefined)) : undefined;
    if (target) {
      // Cleared whatever the answer: a ComfyUI that cannot be reached holds nothing, and retrying a dead relay on
      // every later turn would only add its timeout to each of them.
      residentComfyUi = undefined;
      const freed = await freeLocalComfyUi(target, fetchImpl);
      notify({
        from: "image",
        to,
        detail: freed
          ? "ComfyUI 모델을 VRAM에서 내렸습니다. 로컬 LLM 턴을 진행합니다."
          : "ComfyUI 모델을 내리지 못했거나 이미 내려가 있습니다. 로컬 LLM 턴을 계속합니다."
      });
    }
  };

  /** Book-keeping for work that loads a model on this machine's card, coordinated or not. */
  const noteLoad = (kind: LocalGpuWorkKind, state: AppState): void => {
    if (kind === "image") {
      const target = readLocalComfyUiTarget(state);
      if (target) {
        residentComfyUi = target;
        holder = "image";
      }
      return;
    }
    const targets = readLocalOllamaTargets(state);
    if (targets.length > 0) {
      holder = "llm";
      for (const target of targets) {
        residentOllama.set(ollamaTargetKey(target), target);
      }
    }
  };

  /**
   * Whether this task must queue on the lock: every task that loads a model on this card does — a turn on a local
   * Ollama, a render on a local ComfyUI — whatever its own simulation's photoreal setting. Deciding by "is the
   * other side resident right now" was not enough: a turn that started outside the lock (photoreal off, nothing
   * resident yet) is invisible to a render that starts while it is still generating, and that render then unloaded
   * the turn's model mid-reply and rendered beside its reload (measured: 15.0–15.8 of 16.4 GB in use, and the one
   * ComfyUI crash seen in QA). Queuing costs nothing when nobody else is waiting. Not when the task's simulation
   * turned the hand-over off, which is the user saying both fit (a card bigger than 16 GB).
   */
  const coordinates = (kind: LocalGpuWorkKind, state: AppState): boolean => {
    if (shouldCoordinateLocalGpu(state)) {
      return true;
    }
    if (state.photoreal?.vramHandoff === "off") {
      return false;
    }
    return kind === "llm" ? readLocalOllamaTargets(state).length > 0 : rendersOnLocalComfyUi(state);
  };

  const skip = <T>(runOptions: LocalGpuRunOptions<T> | undefined): T | Promise<T> =>
    runOptions?.onSkip ? runOptions.onSkip() : Promise.reject(new DOMException("GPU 차례가 오기 전에 작업이 취소되었습니다.", "AbortError"));

  return {
    get holder() {
      return holder;
    },
    run<T>(kind: LocalGpuWorkKind, state: AppState, task: () => Promise<T>, runOptions?: LocalGpuRunOptions<T>): Promise<T> {
      if (!coordinates(kind, state)) {
        // Async wrapper so a task that throws synchronously still reports through the returned promise.
        return (async () => {
          if (runOptions?.shouldRun && !runOptions.shouldRun()) {
            return skip(runOptions);
          }
          noteLoad(kind, state);
          return task();
        })();
      }
      const turn = tail.then(async () => {
        // Asked only now that the GPU is free for this task, so a wait behind another task counts.
        if (runOptions?.shouldRun && !runOptions.shouldRun()) {
          return skip(runOptions);
        }
        // Hand-over errors are swallowed inside handOver; this catch is the last guarantee that they cannot
        // stop the task from running.
        await handOver(kind, state).catch(() => undefined);
        // A coordinated task always loads on this card (coordinates() checked for the local server), so this
        // records it as the holder.
        noteLoad(kind, state);
        return task();
      });
      tail = turn.then(
        () => undefined,
        () => undefined
      );
      return turn;
    }
  };
}

function rendersOnLocalComfyUi(state: AppState): boolean {
  const photoreal = state.photoreal;
  if (!photoreal?.enabled || photoreal.requestMode !== "proxy") {
    return false;
  }
  return isLoopbackUrl(readComfyUiBaseUrl(state) || DEFAULT_COMFYUI_BASE_URL);
}

function readComfyUiBaseUrl(state: AppState): string {
  const baseUrl = state.photoreal?.baseUrl;
  return typeof baseUrl === "string" ? baseUrl.trim() : "";
}

function ollamaTargetKey(target: LocalOllamaTarget): string {
  return `${target.apiRoot}\n${target.model}`;
}

/** First occurrence wins, so the order a list was given in is the order the models are unloaded in. */
function mergeOllamaTargets(targets: LocalOllamaTarget[]): LocalOllamaTarget[] {
  const merged = new Map<string, LocalOllamaTarget>();
  for (const target of targets) {
    const key = ollamaTargetKey(target);
    if (!merged.has(key)) {
      merged.set(key, target);
    }
  }
  return [...merged.values()];
}

function readOllamaTarget(settings: LlmApiSettings | undefined): LocalOllamaTarget | undefined {
  if (!settings?.enabled || settings.provider !== "ollama") {
    return undefined;
  }
  const model = typeof settings.model === "string" ? settings.model.trim() : "";
  const apiRoot = toOllamaApiRoot(settings.baseUrl);
  if (!model || !apiRoot || !isLoopbackUrl(apiRoot)) {
    return undefined;
  }
  return { apiRoot, model };
}

/**
 * The server root that /api/* hangs off. The preset stores the OpenAI-compatible base (`…:11434/v1`) and the
 * native transport strips `/v1` the same way (llmClient), but a pasted `/api` or full endpoint path must also land
 * on the root, or the unload request goes to `/v1/api/generate` and silently 404s.
 */
function toOllamaApiRoot(baseUrl: unknown): string | undefined {
  if (typeof baseUrl !== "string") {
    return undefined;
  }
  let root = baseUrl.trim().replace(/\/+$/u, "");
  for (let pass = 0; pass < 3; pass += 1) {
    const stripped = root
      .replace(/\/(?:v1\/)?chat\/completions$/u, "")
      .replace(/\/api\/(?:chat|generate|tags)$/u, "")
      .replace(/\/(?:v1|api)$/u, "")
      .replace(/\/+$/u, "");
    if (stripped === root) {
      break;
    }
    root = stripped;
  }
  return root || undefined;
}

/** localhost, 127.0.0.0/8 or ::1. Anything else is another machine with its own GPU. */
function isLoopbackUrl(value: string): boolean {
  let hostname: string;
  try {
    hostname = new URL(value).hostname.toLowerCase();
  } catch {
    return false;
  }
  const bare = hostname.replace(/^\[|\]$/gu, "");
  return bare === "localhost" || bare === "::1" || /^127(?:\.\d{1,3}){3}$/u.test(bare);
}
