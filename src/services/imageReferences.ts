import { DEFAULT_DYNAMICCHAT_API_BASE_URL, getConfiguredDynamicChatApiBaseUrl } from "./dynamicChatApi";

/**
 * The only canvases NovelAI Precise Reference accepts. Any other size is rejected, so a profile image is
 * letterboxed onto the closest of these once, when it is picked, and that copy is what every cut sends.
 * Square first so an exact tie in aspect goes to the canvas that keeps the most pixels.
 */
const NOVELAI_REFERENCE_CANVASES: ReadonlyArray<{ width: number; height: number }> = [
  { width: 1472, height: 1472 },
  { width: 1024, height: 1536 },
  { width: 1536, height: 1024 }
];

/** Same ceiling the relay applies to a reference data URL; a profile image is a few MB at most. */
const MAX_REFERENCE_BYTES = 20 * 1024 * 1024;
/** The object store is the local API server; anything slower than this is not coming back. */
const OBJECT_FETCH_TIMEOUT_MS = 15_000;
/** Keeps String.fromCharCode well below every engine's argument-count limit. */
const BASE64_CHUNK_SIZE = 0x8000;

/**
 * Reads an object-store image (a profile image or its letterboxed reference canvas) as raw base64, for the
 * providers that take reference bytes inline (NovelAI Precise Reference).
 *
 * Never throws: a missing reference must degrade to "generate without it", not fail the cut. That covers a
 * key the store no longer has, a server that is not running, and Node without any server at all.
 */
export async function fetchObjectAsBase64(objectKey: string, signal?: AbortSignal): Promise<string | undefined> {
  const objectPath = encodeObjectKeyPath(objectKey);
  if (!objectPath) {
    return undefined;
  }

  try {
    const response = await fetch(`${readApiBaseUrl()}/objects/${objectPath}`, {
      method: "GET",
      signal: withObjectFetchTimeout(signal)
    });
    if (!response.ok) {
      return undefined;
    }
    // The object route only ever serves images. Anything else means the URL resolved to some other route,
    // and those bytes must not be shipped to a provider as a "reference image".
    if (!/^image\//iu.test(response.headers.get("content-type") ?? "")) {
      return undefined;
    }
    const declaredLength = Number(response.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > MAX_REFERENCE_BYTES) {
      return undefined;
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength === 0 || bytes.byteLength > MAX_REFERENCE_BYTES) {
      return undefined;
    }
    return bytesToBase64(bytes);
  } catch {
    return undefined;
  }
}

/** Picks the Precise Reference canvas whose aspect ratio is closest to the image's (compared in log space, so 4:5 and 5:4 are equally far from square). */
export function pickNovelAiReferenceCanvas(width: number, height: number): { width: number; height: number } {
  const fallback = NOVELAI_REFERENCE_CANVASES[0];
  if (!(Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0)) {
    return { width: fallback.width, height: fallback.height };
  }

  const aspect = Math.log(width / height);
  let best = fallback;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const canvas of NOVELAI_REFERENCE_CANVASES) {
    const distance = Math.abs(aspect - Math.log(canvas.width / canvas.height));
    if (distance < bestDistance) {
      best = canvas;
      bestDistance = distance;
    }
  }
  return { width: best.width, height: best.height };
}

/**
 * Browser only. Scales the image to fit the closest Precise Reference canvas, centres it, and pads the rest
 * with opaque black. The surface is opaque, so transparent pixels flatten onto black instead of travelling
 * as alpha. `width`/`height` are the image's natural size (an HTMLImageElement reports 0 until it decodes,
 * so the caller passes what it measured).
 */
export async function letterboxForNovelAiReference(
  source: ImageBitmap | HTMLImageElement,
  width: number,
  height: number
): Promise<Blob> {
  const sourceWidth = width > 0 ? width : readNaturalWidth(source);
  const sourceHeight = height > 0 ? height : readNaturalHeight(source);
  if (!(Number.isFinite(sourceWidth) && Number.isFinite(sourceHeight) && sourceWidth > 0 && sourceHeight > 0)) {
    throw new Error("레퍼런스 이미지의 크기를 읽지 못했습니다.");
  }

  const canvasSize = pickNovelAiReferenceCanvas(sourceWidth, sourceHeight);
  const scale = Math.min(canvasSize.width / sourceWidth, canvasSize.height / sourceHeight);
  const drawWidth = Math.min(canvasSize.width, Math.max(1, Math.round(sourceWidth * scale)));
  const drawHeight = Math.min(canvasSize.height, Math.max(1, Math.round(sourceHeight * scale)));
  const offsetX = Math.floor((canvasSize.width - drawWidth) / 2);
  const offsetY = Math.floor((canvasSize.height - drawHeight) / 2);

  const surface = createOpaqueCanvas(canvasSize.width, canvasSize.height);
  const { context } = surface;
  context.fillStyle = "#000000";
  context.fillRect(0, 0, canvasSize.width, canvasSize.height);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, offsetX, offsetY, drawWidth, drawHeight);
  return surface.toPngBlob();
}

/**
 * btoa over a binary string works the same in browsers and in Node 24; FileReader does not exist in Node and
 * Buffer does not exist in browsers, and these helpers run in both (the app, and the evals through
 * vite.ssrLoadModule).
 */
export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += BASE64_CHUNK_SIZE) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + BASE64_CHUNK_SIZE));
  }
  return btoa(binary);
}

/**
 * Object keys the store writes are always `<simulationId>/<assetId>.<ext>`, and the server only accepts
 * segments of [A-Za-z0-9._-]. Anything else is refused here rather than encoded: fetch resolves the URL, so a
 * ".." segment would climb out of /objects/ into another API route (GET /simulations returns every saved
 * simulation), and those bytes would then be sent to a provider as a reference image.
 */
function encodeObjectKeyPath(objectKey: string): string | undefined {
  if (typeof objectKey !== "string") {
    return undefined;
  }
  const segments = objectKey.trim().split("/");
  if (!segments.every(isSafeObjectKeySegment)) {
    return undefined;
  }
  return segments.map(encodeURIComponent).join("/");
}

function isSafeObjectKeySegment(segment: string): boolean {
  return segment !== "." && segment !== ".." && /^[A-Za-z0-9._-]+$/u.test(segment);
}

function readApiBaseUrl(): string {
  try {
    return getConfiguredDynamicChatApiBaseUrl();
  } catch {
    // The configured base lives in localStorage, and a sandboxed or storage-blocked browser context throws a
    // SecurityError on access. "Never throws" covers that too: fall back to the default local API server.
    return DEFAULT_DYNAMICCHAT_API_BASE_URL;
  }
}

function withObjectFetchTimeout(signal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(OBJECT_FETCH_TIMEOUT_MS);
  if (!signal) {
    return timeout;
  }
  // Without AbortSignal.any keep the caller's cancel: the local server answers or refuses quickly anyway.
  return typeof AbortSignal.any === "function" ? AbortSignal.any([signal, timeout]) : signal;
}

function readNaturalWidth(source: ImageBitmap | HTMLImageElement): number {
  return "naturalWidth" in source ? source.naturalWidth : source.width;
}

function readNaturalHeight(source: ImageBitmap | HTMLImageElement): number {
  return "naturalHeight" in source ? source.naturalHeight : source.height;
}

interface OpaqueCanvas {
  context: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  toPngBlob(): Promise<Blob>;
}

function createOpaqueCanvas(width: number, height: number): OpaqueCanvas {
  // OffscreenCanvas first: it works off the main thread and in workers. The DOM canvas covers older Safari.
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { alpha: false });
    if (context) {
      return { context, toPngBlob: () => canvas.convertToBlob({ type: "image/png" }) };
    }
  }

  if (typeof document !== "undefined") {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d", { alpha: false });
    if (context) {
      return {
        context,
        toPngBlob: () =>
          new Promise<Blob>((resolve, reject) => {
            canvas.toBlob(
              (blob) => (blob ? resolve(blob) : reject(new Error("레퍼런스 이미지를 PNG로 변환하지 못했습니다."))),
              "image/png"
            );
          })
      };
    }
  }

  throw new Error("레퍼런스 이미지 변환은 브라우저에서만 할 수 있습니다.");
}
