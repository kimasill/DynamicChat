/**
 * A stand-in ComfyUI for the photoreal relay evals.
 *
 * It speaks the slice of ComfyUI's HTTP API the relay uses — /system_stats, /object_info, /models, /upload/image,
 * /prompt, /history, /view, /interrupt, /queue, /free — with the response shapes of a real server (ComfyUI 0.40 on
 * a 16 GB card), records every request so a test can assert on exactly what the relay put on the wire, and
 * switches between the situations the relay has to translate:
 *
 *   ok               a queued prompt stays out of history for `pendingPolls` polls, then succeeds with one saved
 *                    PNG per SaveImage node and one "temp" PNG per PreviewImage node (named as the real node names
 *                    them: ComfyUI_temp_<5 letters>_<counter>_.png, no subfolder, filename_prefix not taken)
 *   node_errors      POST /prompt answers 400 value_not_in_list on the UNETLoader's unet_name — a model file that
 *                    is not installed — with ComfyUI's full error body, input_config listing included
 *   execution_error  the prompt fails while running: history status_str "error" with a CUDA out-of-memory
 *   never_completes  the prompt never reaches history
 *   redirect         /system_stats and POST /prompt answer 302 to /redirect-target
 *   upload_rename    /upload/image stores the file under a different name than it was sent with
 *   upstream_down    every connection is dropped without an answer
 *
 * Like the real server it refuses a loopback request whose Origin does not match its Host (ComfyUI's
 * origin_only_middleware), so a relay that forwarded the browser's Origin would fail here as it would against
 * the real thing. A prompt keeps the behaviour that was active when it was queued, so switching behaviour
 * between tests cannot change the fate of a prompt that is still being polled.
 *
 *   const mock = await startMockComfyUi();
 *   mock.setBehaviour("node_errors");
 *   ... mock.received / mock.requests("POST", "/prompt") ...
 *   await mock.close();
 */
import http from "node:http";
import { randomUUID } from "node:crypto";
import { crc32, deflateSync } from "node:zlib";

export const MOCK_COMFYUI_BEHAVIOURS = Object.freeze([
  "ok",
  "node_errors",
  "execution_error",
  "never_completes",
  "redirect",
  "upload_rename",
  "upstream_down"
]);

export const MOCK_COMFYUI_VERSION = "0.40.0";
export const MOCK_COMFYUI_GPU = "NVIDIA GeForce RTX 4080 SUPER";

const DEFAULT_MODELS = {
  checkpoints: [],
  diffusion_models: ["qwen_image_2.1_turbo_int8_convrot.safetensors", "z_image_turbo_int8_convrot.safetensors"],
  text_encoders: ["qwen3vl_8b_int8_convrot.safetensors", "qwen_3_4b_fp8_mixed.safetensors"],
  vae: ["qwen_image_2.1_vae_bf16.safetensors", "ae.safetensors"],
  loras: []
};

// The node types /object_info reports as installed: ComfyUI core plus the Qwen-Image 2.1 nodes.
const DEFAULT_NODES = [
  "CheckpointLoaderSimple",
  "UNETLoader",
  "CLIPLoader",
  "DualCLIPLoader",
  "VAELoader",
  "LoraLoaderModelOnly",
  "CLIPTextEncode",
  "ConditioningZeroOut",
  "KSampler",
  "KSamplerAdvanced",
  "SamplerCustomAdvanced",
  "KSamplerSelect",
  "RandomNoise",
  "CFGGuider",
  "BasicScheduler",
  "ManualSigmas",
  "ModelSamplingAuraFlow",
  "CFGNorm",
  "EmptyLatentImage",
  "EmptySD3LatentImage",
  "VAEDecode",
  "VAEEncode",
  "LoadImage",
  "ImageScale",
  "FluxKontextImageScale",
  "FluxKontextMultiReferenceLatentMethod",
  "TextEncodeQwenImageEditPlus",
  "TextEncodeQwenImage21",
  "QwenImage21Cache",
  "SaveImage",
  "PreviewImage"
];

export const MOCK_COMFYUI_VRAM = Object.freeze({ totalBytes: 17_170_956_288, freeBytes: 15_838_609_408 });
const VRAM_TOTAL_BYTES = MOCK_COMFYUI_VRAM.totalBytes;
const VRAM_FREE_BYTES = MOCK_COMFYUI_VRAM.freeBytes;

/**
 * A small, genuinely valid PNG (correct signature, CRCs and zlib stream), so a test can tell "the relay passed
 * the bytes through" from "the relay produced something PNG-shaped".
 */
export function createMockPng({ width = 2, height = 2, rgb = [0xf4, 0xf0, 0xe8] } = {}) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8; // bit depth
  header[9] = 2; // colour type: truecolour RGB; compression, filter and interlace stay 0
  const rowLength = 1 + width * 3;
  const pixels = Buffer.alloc(rowLength * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      pixels.set(rgb, y * rowLength + 1 + x * 3); // byte 0 of each row is filter type 0
    }
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", header),
    pngChunk("IDAT", deflateSync(pixels)),
    pngChunk("IEND", Buffer.alloc(0))
  ]);
}

function pngChunk(type, data) {
  const typeAndData = Buffer.concat([Buffer.from(type, "latin1"), data]);
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(typeAndData) >>> 0);
  return Buffer.concat([length, typeAndData, checksum]);
}

export async function startMockComfyUi(options = {}) {
  const pendingPolls = options.pendingPolls ?? 2;
  const models = { ...DEFAULT_MODELS, ...(options.models ?? {}) };
  const knownNodes = new Set(options.nodes ?? DEFAULT_NODES);
  const outputPng = options.outputPng ?? createMockPng();
  const received = [];
  const prompts = new Map();
  const files = new Map();
  const uploads = new Map();
  let behaviour = "ok";
  let promptNumber = 0;
  let outputCounter = 0;

  const server = http.createServer((request, response) => {
    if (behaviour === "upstream_down") {
      received.push(recordRequest(request, undefined, behaviour));
      request.socket.destroy();
      return;
    }
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      const record = recordRequest(request, Buffer.concat(chunks), behaviour);
      received.push(record);
      try {
        handle(request, response, record, Buffer.concat(chunks));
      } catch (error) {
        respond(response, 500, { error: { type: "mock_failure", message: String(error?.stack ?? error) } });
      }
    });
  });

  function handle(request, response, record, raw) {
    if (rejectsMismatchedOrigin(request.headers)) {
      record.rejectedByOriginCheck = true;
      respondText(response, 403, "Forbidden");
      return;
    }
    const method = request.method ?? "GET";
    const pathname = record.path;
    const segments = pathname.split("/").filter(Boolean).map(decodeURIComponent);

    if (method === "GET" && pathname === "/system_stats") {
      if (behaviour === "redirect") {
        redirect(response);
        return;
      }
      respond(response, 200, createSystemStats());
      return;
    }
    if (method === "GET" && pathname === "/redirect-target") {
      // Answers like a healthy ComfyUI so that a relay which followed the redirect would look like it worked.
      respond(response, 200, createSystemStats());
      return;
    }
    if (method === "GET" && segments[0] === "object_info") {
      if (segments.length === 1) {
        respond(response, 200, Object.assign({}, ...[...knownNodes].map((name) => describeNode(name))));
        return;
      }
      // Unknown nodes are 200 + {} on a real server, not 404.
      respond(response, 200, knownNodes.has(segments[1]) ? describeNode(segments[1]) : {});
      return;
    }
    if (method === "GET" && segments[0] === "models") {
      if (segments.length === 1) {
        respond(response, 200, Object.keys(models).filter((folder) => Array.isArray(models[folder])));
        return;
      }
      const listing = models[segments[1]];
      if (!Array.isArray(listing)) {
        respondText(response, 404, "404: Not Found");
        return;
      }
      respond(response, 200, listing);
      return;
    }
    if (method === "POST" && pathname === "/upload/image") {
      handleUpload(response, record, raw);
      return;
    }
    if (method === "POST" && pathname === "/prompt") {
      handleQueuePrompt(response, record);
      return;
    }
    if (method === "GET" && pathname === "/prompt") {
      respond(response, 200, { exec_info: { queue_remaining: [...prompts.values()].filter(isQueued).length } });
      return;
    }
    if (method === "GET" && segments[0] === "history") {
      if (segments.length === 1) {
        respond(response, 200, Object.assign({}, ...[...prompts.values()].map((prompt) => prompt.historyEntry ?? {})));
        return;
      }
      respond(response, 200, pollHistory(segments[1]));
      return;
    }
    if (method === "GET" && pathname === "/view") {
      const query = record.query;
      const bytes = files.get(fileKey(query.type ?? "output", query.subfolder ?? "", query.filename ?? ""));
      if (!bytes) {
        respondText(response, 404, "404: Not Found");
        return;
      }
      response.writeHead(200, { "content-type": "image/png", "content-length": String(bytes.length) });
      response.end(bytes);
      return;
    }
    if (method === "POST" && pathname === "/interrupt") {
      const prompt = prompts.get(record.body?.prompt_id);
      if (prompt) {
        prompt.interrupted = true;
      }
      respondText(response, 200, "");
      return;
    }
    if (method === "GET" && pathname === "/queue") {
      const entries = [...prompts.values()].filter(isQueued).map((prompt) => [prompt.number, prompt.id, prompt.workflow, { client_id: prompt.clientId }, []]);
      respond(response, 200, { queue_running: entries.slice(0, 1), queue_pending: entries.slice(1) });
      return;
    }
    if (method === "POST" && pathname === "/queue") {
      for (const promptId of Array.isArray(record.body?.delete) ? record.body.delete : []) {
        const prompt = prompts.get(promptId);
        if (prompt) {
          prompt.deleted = true;
        }
      }
      if (record.body?.clear === true) {
        prompts.forEach((prompt) => {
          prompt.deleted = true;
        });
      }
      respondText(response, 200, "");
      return;
    }
    if (method === "POST" && pathname === "/free") {
      respondText(response, 200, "");
      return;
    }
    respondText(response, 404, "404: Not Found");
  }

  function handleUpload(response, record, raw) {
    const parts = parseMultipart(raw, record.headers["content-type"]);
    const field = (name) => parts.find((part) => part.name === name && part.filename === undefined)?.data.toString("utf8");
    const image = parts.find((part) => part.name === "image" && part.filename);
    if (!image) {
      respondText(response, 400, "No image uploaded");
      return;
    }
    const type = field("type") || "input";
    const subfolder = field("subfolder") ?? "";
    const storedName = behaviour === "upload_rename" ? `renamed_${image.filename}` : image.filename;
    files.set(fileKey(type, subfolder, storedName), image.data);
    uploads.set(subfolder ? `${subfolder}/${storedName}` : storedName, image.data);
    respond(response, 200, { name: storedName, subfolder, type });
  }

  function handleQueuePrompt(response, record) {
    if (behaviour === "redirect") {
      redirect(response);
      return;
    }
    const workflow = record.body?.prompt;
    if (!isPlainObject(workflow) || Object.keys(workflow).length === 0) {
      respond(response, 400, {
        error: { type: "invalid_prompt", message: "Cannot execute because the prompt is empty.", details: "", extra_info: {} },
        node_errors: {}
      });
      return;
    }
    if (behaviour === "node_errors") {
      respond(response, 400, createMissingModelError(workflow));
      return;
    }
    promptNumber += 1;
    const prompt = {
      id: randomUUID(),
      number: promptNumber,
      workflow,
      clientId: record.body?.client_id,
      behaviour,
      polls: 0,
      interrupted: false,
      deleted: false,
      historyEntry: undefined
    };
    prompts.set(prompt.id, prompt);
    respond(response, 200, { prompt_id: prompt.id, number: prompt.number, node_errors: {} });
  }

  function pollHistory(promptId) {
    const prompt = prompts.get(promptId);
    if (!prompt) {
      return {};
    }
    prompt.polls += 1;
    if (prompt.historyEntry) {
      return { [prompt.id]: prompt.historyEntry };
    }
    if (prompt.behaviour === "never_completes" || prompt.deleted) {
      return {};
    }
    if (prompt.behaviour === "execution_error") {
      if (prompt.polls <= 1) {
        return {};
      }
      prompt.historyEntry = createFailedHistoryEntry(prompt);
      return { [prompt.id]: prompt.historyEntry };
    }
    if (prompt.polls <= pendingPolls) {
      return {};
    }
    prompt.historyEntry = createFinishedHistoryEntry(prompt);
    return { [prompt.id]: prompt.historyEntry };
  }

  // One image per output node, named the way ComfyUI names them. SaveImage: "DynamicChat/job_1" becomes subfolder
  // "DynamicChat" and file "job_1_00001_.png" in the output folder. PreviewImage has no filename_prefix input: it
  // writes "ComfyUI_temp_<5 random letters>_00001_.png" straight into the temp folder (nodes.py, PreviewImage).
  function createFinishedHistoryEntry(prompt) {
    const outputs = {};
    for (const [nodeId, node] of Object.entries(prompt.workflow)) {
      const type = node?.class_type === "SaveImage" ? "output" : node?.class_type === "PreviewImage" ? "temp" : undefined;
      if (!type) {
        continue;
      }
      const prefix = type === "temp" ? `ComfyUI_temp_${randomLetters(5)}` : String(node.inputs?.filename_prefix ?? "ComfyUI");
      const slash = prefix.lastIndexOf("/");
      const subfolder = slash >= 0 ? prefix.slice(0, slash) : "";
      outputCounter += 1;
      const filename = `${slash >= 0 ? prefix.slice(slash + 1) : prefix}_${String(outputCounter).padStart(5, "0")}_.png`;
      files.set(fileKey(type, subfolder, filename), outputPng);
      outputs[nodeId] = { images: [{ filename, subfolder, type }] };
    }
    const timestamp = Date.now();
    return {
      prompt: [prompt.number, prompt.id, prompt.workflow, { client_id: prompt.clientId }, Object.keys(outputs)],
      outputs,
      status: {
        status_str: "success",
        completed: true,
        messages: [
          ["execution_start", { prompt_id: prompt.id, timestamp }],
          ["execution_cached", { nodes: [], prompt_id: prompt.id, timestamp }],
          ["execution_success", { prompt_id: prompt.id, timestamp }]
        ]
      },
      meta: Object.fromEntries(Object.keys(outputs).map((nodeId) => [nodeId, { node_id: nodeId, display_node: nodeId, parent_node: null, real_node_id: nodeId }]))
    };
  }

  function createFailedHistoryEntry(prompt) {
    const entries = Object.entries(prompt.workflow);
    const [nodeId, node] =
      entries.find(([, candidate]) => /^(?:KSampler|KSamplerAdvanced|SamplerCustom|SamplerCustomAdvanced)$/u.test(candidate?.class_type ?? "")) ??
      entries[0];
    const timestamp = Date.now();
    return {
      prompt: [prompt.number, prompt.id, prompt.workflow, { client_id: prompt.clientId }, []],
      outputs: {},
      status: {
        status_str: "error",
        completed: false,
        messages: [
          ["execution_start", { prompt_id: prompt.id, timestamp }],
          ["execution_cached", { nodes: [], prompt_id: prompt.id, timestamp }],
          [
            "execution_error",
            {
              prompt_id: prompt.id,
              node_id: nodeId,
              node_type: node?.class_type ?? "KSampler",
              executed: entries.map(([id]) => id).filter((id) => id !== nodeId).slice(0, 3),
              exception_message:
                "Allocation on device 0 would exceed allowed memory. (out of memory)\nCurrently allocated     : 14.92 GiB\nRequested               : 1.50 GiB\nDevice limit            : 15.99 GiB\nFree (according to CUDA): 0 bytes",
              exception_type: "torch.OutOfMemoryError",
              traceback: ['  File "C:\\ComfyUI\\execution.py", line 496, in execute\n', "torch.OutOfMemoryError: Allocation on device\n"],
              current_inputs: {},
              current_outputs: {},
              timestamp
            }
          ]
        ]
      },
      meta: {}
    };
  }

  // ComfyUI's answer to a loader naming a file that is not in its models folder, complete with the folder
  // listing in input_config that a relay should not pass through.
  function createMissingModelError(workflow) {
    const [nodeId, node] = Object.entries(workflow).find(([, candidate]) => candidate?.class_type === "UNETLoader") ?? [
      "1",
      { class_type: "UNETLoader", inputs: { unet_name: "qwen_image_2.1_turbo_int8_convrot.safetensors" } }
    ];
    const missing = String(node.inputs?.unet_name ?? "qwen_image_2.1_turbo_int8_convrot.safetensors");
    const available = models.diffusion_models.filter((file) => file !== missing);
    const outputNodeIds = Object.entries(workflow)
      .filter(([, candidate]) => candidate?.class_type === "SaveImage" || candidate?.class_type === "PreviewImage")
      .map(([id]) => id);
    return {
      error: { type: "prompt_outputs_failed_validation", message: "Prompt outputs failed validation", details: "", extra_info: {} },
      node_errors: {
        [nodeId]: {
          errors: [
            {
              type: "value_not_in_list",
              message: "Value not in list",
              details: `unet_name: '${missing}' not in [${available.map((file) => `'${file}'`).join(", ")}]`,
              extra_info: {
                input_name: "unet_name",
                input_config: [available, { tooltip: "The diffusion model file to load." }],
                received_value: missing
              }
            }
          ],
          dependent_outputs: outputNodeIds,
          class_type: "UNETLoader"
        }
      }
    };
  }

  function describeNode(name) {
    const required = {
      UNETLoader: {
        unet_name: [models.diffusion_models, { tooltip: "The diffusion model file to load." }],
        weight_dtype: [["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {}]
      },
      CLIPLoader: {
        clip_name: [models.text_encoders, {}],
        type: [["stable_diffusion", "stable_cascade", "sd3", "lumina2", "qwen_image", "hunyuan_image"], {}]
      },
      VAELoader: { vae_name: [models.vae, {}] },
      LoadImage: { image: [[...uploads.keys()].sort(), { image_upload: true }] },
      SaveImage: { images: ["IMAGE", {}], filename_prefix: ["STRING", { default: "ComfyUI" }] },
      PreviewImage: { images: ["IMAGE", {}] },
      TextEncodeQwenImage21: {
        clip: ["CLIP", {}],
        prompt: ["STRING", { multiline: true, dynamicPrompts: true }],
        negative_prompt: ["STRING", { multiline: true, dynamicPrompts: true }],
        resolution: ["INT", { default: 1024, min: 256, max: 2048, step: 16 }]
      }
    }[name] ?? {};
    const outputs = {
      UNETLoader: ["MODEL"],
      CLIPLoader: ["CLIP"],
      VAELoader: ["VAE"],
      LoadImage: ["IMAGE", "MASK"],
      TextEncodeQwenImage21: ["CONDITIONING", "CONDITIONING"],
      SaveImage: [],
      PreviewImage: []
    }[name] ?? ["*"];
    return {
      [name]: {
        input: { required, ...(name === "TextEncodeQwenImage21" ? { optional: { vae: ["VAE", {}], images: ["COMFY_AUTOGROW_V3", {}] } } : {}) },
        input_order: { required: Object.keys(required) },
        output: outputs,
        output_is_list: outputs.map(() => false),
        output_name: outputs,
        name,
        display_name: name,
        description: "",
        python_module: name.startsWith("TextEncodeQwenImage21") || name === "QwenImage21Cache" ? "comfy_extras.nodes_qwen" : "nodes",
        category: "mock",
        output_node: name === "SaveImage" || name === "PreviewImage",
        deprecated: false,
        experimental: false
      }
    };
  }

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(options.port ?? 0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const { port } = server.address();

  return {
    baseUrl: `http://127.0.0.1:${port}`,
    port,
    received,
    prompts,
    uploads,
    get behaviour() {
      return behaviour;
    },
    setBehaviour(name) {
      if (!MOCK_COMFYUI_BEHAVIOURS.includes(name)) {
        throw new Error(`Unknown mock ComfyUI behaviour: ${name}`);
      }
      behaviour = name;
    },
    // Replaces one /models/<folder> listing; null makes the folder unknown (404), as on a ComfyUI without it.
    setModels(folder, listing) {
      models[folder] = listing ?? undefined;
    },
    requests(method, pathname) {
      return received.filter((entry) => entry.method === method && entry.path === pathname);
    },
    reset() {
      received.length = 0;
      prompts.clear();
      files.clear();
      uploads.clear();
      Object.assign(models, DEFAULT_MODELS, options.models ?? {});
      behaviour = "ok";
    },
    close() {
      return new Promise((resolve) => {
        server.close(() => resolve());
        server.closeAllConnections();
      });
    }
  };
}

function recordRequest(request, raw, behaviour) {
  const url = new URL(request.url ?? "/", "http://mock-comfyui.invalid");
  return {
    method: request.method ?? "GET",
    path: url.pathname,
    query: Object.fromEntries(url.searchParams),
    headers: { ...request.headers },
    body: raw ? parseBody(request.headers["content-type"], raw) : undefined,
    behaviour,
    at: Date.now()
  };
}

function parseBody(contentType, raw) {
  if (raw.length === 0) {
    return undefined;
  }
  if (/^multipart\/form-data/iu.test(contentType ?? "")) {
    return {
      parts: parseMultipart(raw, contentType).map((part) => ({
        name: part.name,
        filename: part.filename,
        contentType: part.contentType,
        byteLength: part.data.length,
        // The first eight bytes are enough to see whether an upload is a PNG/JPEG/WebP or something else.
        magic: part.data.subarray(0, 8).toString("hex"),
        value: part.filename === undefined ? part.data.toString("utf8") : undefined
      }))
    };
  }
  const text = raw.toString("utf8");
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/**
 * Splits a multipart/form-data body into its parts by hand — the point of the mock is to see the bytes the
 * relay sent, not what a parser library would make of them.
 */
export function parseMultipart(raw, contentType) {
  const boundary = /boundary=(?:"([^"]+)"|([^\s;]+))/iu.exec(contentType ?? "");
  if (!boundary) {
    return [];
  }
  const delimiter = Buffer.from(`--${boundary[1] ?? boundary[2]}`, "latin1");
  const separator = Buffer.concat([Buffer.from("\r\n", "latin1"), delimiter]);
  const parts = [];
  let cursor = raw.indexOf(delimiter);
  while (cursor !== -1) {
    const afterDelimiter = cursor + delimiter.length;
    // "--" straight after a delimiter closes the body.
    if (raw[afterDelimiter] === 0x2d && raw[afterDelimiter + 1] === 0x2d) {
      break;
    }
    const headerStart = afterDelimiter + 2;
    const headerEnd = raw.indexOf("\r\n\r\n", headerStart, "latin1");
    if (headerEnd === -1) {
      break;
    }
    const next = raw.indexOf(separator, headerEnd + 4);
    if (next === -1) {
      break;
    }
    const headers = Object.fromEntries(
      raw
        .subarray(headerStart, headerEnd)
        .toString("utf8")
        .split("\r\n")
        .map((line) => {
          const colon = line.indexOf(":");
          return colon > 0 ? [line.slice(0, colon).trim().toLowerCase(), line.slice(colon + 1).trim()] : undefined;
        })
        .filter(Boolean)
    );
    const disposition = headers["content-disposition"] ?? "";
    parts.push({
      name: /\bname="([^"]*)"/iu.exec(disposition)?.[1] ?? "",
      filename: /\bfilename="([^"]*)"/iu.exec(disposition)?.[1],
      contentType: headers["content-type"],
      data: raw.subarray(headerEnd + 4, next)
    });
    cursor = next + 2;
  }
  return parts;
}

// ComfyUI's origin_only_middleware: on a loopback Host, an Origin naming a different host:port is refused.
function rejectsMismatchedOrigin(headers) {
  if (!headers.host || !headers.origin) {
    return false;
  }
  let host;
  let origin;
  try {
    host = new URL(`http://${headers.host}`);
    origin = new URL(headers.origin);
  } catch {
    return false;
  }
  const loopback = ["127.0.0.1", "localhost", "[::1]"].includes(host.hostname);
  return loopback && host.host !== origin.host;
}

function createSystemStats() {
  return {
    system: {
      os: "nt",
      ram_total: 68_719_476_736,
      ram_free: 41_231_686_042,
      comfyui_version: MOCK_COMFYUI_VERSION,
      required_frontend_version: "1.30.2",
      python_version: "3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]",
      pytorch_version: "2.9.0+cu130",
      embedded_python: true,
      argv: ["ComfyUI\\main.py", "--windows-standalone-build", "--listen", "127.0.0.1"]
    },
    devices: [
      {
        // The real format: device index, card name, allocator backend.
        name: `cuda:0 ${MOCK_COMFYUI_GPU} : cudaMallocAsync`,
        type: "cuda",
        index: 0,
        vram_total: VRAM_TOTAL_BYTES,
        vram_free: VRAM_FREE_BYTES,
        torch_vram_total: 0,
        torch_vram_free: 0
      }
    ]
  };
}

function isQueued(prompt) {
  return !prompt.historyEntry && !prompt.deleted && !prompt.interrupted;
}

function randomLetters(count) {
  return Array.from({ length: count }, () => String.fromCharCode(97 + Math.floor(Math.random() * 26))).join("");
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function fileKey(type, subfolder, filename) {
  return `${type}/${subfolder}/${filename}`;
}

function redirect(response) {
  response.writeHead(302, { location: "/redirect-target", "content-length": "0" });
  response.end();
}

function respond(response, status, payload) {
  const text = JSON.stringify(payload);
  response.writeHead(status, { "content-type": "application/json; charset=utf-8", "content-length": String(Buffer.byteLength(text)) });
  response.end(text);
}

function respondText(response, status, text) {
  response.writeHead(status, { "content-type": "text/plain; charset=utf-8", "content-length": String(Buffer.byteLength(text)) });
  response.end(text);
}
