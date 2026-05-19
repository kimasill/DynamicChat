import { createServer } from "node:http";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "..");
const dataDir = path.resolve(process.env.DYNAMICCHAT_DATA_DIR ?? path.join(workspaceRoot, ".dynamicchat-data"));
const defaultObjectDir = path.join(dataDir, "objects");
const backupDir = path.join(dataDir, "backups");
const statePath = path.join(dataDir, "state.json");
const secretPath = path.join(dataDir, "dev-secrets.json");
const port = Number(process.env.DYNAMICCHAT_API_PORT ?? 4318);
const novelAiSubscriptionUrl = "https://api.novelai.net/user/subscription";
const novelAiGenerateImageUrl = "https://image.novelai.net/ai/generate-image";
const rateLimitWindowMs = Number(process.env.DYNAMICCHAT_RATE_LIMIT_WINDOW_MS ?? 60_000);
const rateLimitMaxRequests = Number(process.env.DYNAMICCHAT_RATE_LIMIT_MAX ?? 180);
const rateLimitBuckets = new Map();
let stateStoreWriteQueue = Promise.resolve();

const corsHeaders = {
  "access-control-allow-origin": process.env.DYNAMICCHAT_CORS_ORIGIN ?? "*",
  "access-control-allow-methods": "GET,POST,PUT,PATCH,OPTIONS",
  "access-control-allow-headers": "content-type, authorization, x-dynamicchat-owner-id, x-dynamicchat-workspace-id, x-dynamicchat-project-id, x-dynamicchat-environment",
  "access-control-max-age": "86400"
};

await ensureStore();

const server = createServer(async (request, response) => {
  try {
    if (request.method === "OPTIONS") {
      send(response, 204);
      return;
    }

    if (!allowRequestByRateLimit(request, response)) {
      return;
    }

    const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);
    const route = matchRoute(request.method ?? "GET", url.pathname);

    if (!route) {
      sendJson(response, 404, { error: "Not found" });
      return;
    }

    await route.handler(request, response, route.params);
  } catch (error) {
    sendJson(response, 500, {
      error: error instanceof Error ? error.message : "Unknown DynamicChat API error"
    });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`DynamicChat API listening on http://127.0.0.1:${port}`);
  console.log(`Data directory: ${dataDir}`);
  console.log("Dev warning: API secrets are stored in a local development secret file, not production encryption.");
});

function matchRoute(method, pathname) {
  const routes = [
    ["GET", /^\/health$/u, health],
    ["GET", /^\/novelai\/subscription$/u, getNovelAiSubscription],
    ["POST", /^\/novelai\/generate-image$/u, proxyNovelAiGenerateImage],
    ["GET", /^\/objects\/(.+)$/u, getObjectAsset],
    ["GET", /^\/simulations$/u, listSimulations],
    ["POST", /^\/simulations$/u, createSimulation],
    ["GET", /^\/simulations\/([^/]+)$/u, getSimulation],
    ["PUT", /^\/simulations\/([^/]+)\/state$/u, saveSimulationState],
    ["GET", /^\/personal-api-vault$/u, getPersonalApiVault],
    ["PUT", /^\/personal-api-vault$/u, savePersonalApiVault],
    ["POST", /^\/simulations\/([^/]+)\/prompt-modules$/u, createPromptModule],
    ["PATCH", /^\/prompt-modules\/([^/]+)$/u, updatePromptModule],
    ["POST", /^\/simulations\/([^/]+)\/chat\/turns$/u, createChatTurn],
    ["POST", /^\/simulations\/([^/]+)\/sessions\/reset$/u, resetSession],
    ["POST", /^\/simulations\/([^/]+)\/image-jobs$/u, createImageJob],
    ["POST", /^\/simulations\/([^/]+)\/assets$/u, persistImageAssets],
    ["GET", /^\/simulations\/([^/]+)\/assets$/u, listAssets],
    ["GET", /^\/simulations\/([^/]+)\/audit$/u, listAudit],
    ["POST", /^\/simulations\/([^/]+)\/backup$/u, createBackup],
    ["POST", /^\/simulations\/([^/]+)\/redactions$/u, createRedaction],
    ["POST", /^\/image-jobs\/([^/]+)\/cancel$/u, cancelImageJob],
    ["GET", /^\/image-jobs\/([^/]+)$/u, getImageJob]
  ];

  for (const [routeMethod, pattern, handler] of routes) {
    const match = pathname.match(pattern);
    if (routeMethod === method && match) {
      return { handler, params: match.slice(1).map(decodeURIComponent) };
    }
  }

  return undefined;
}

function allowRequestByRateLimit(request, response) {
  const scope = readRequestScope(request);
  const remote = request.socket.remoteAddress ?? "local";
  const key = `${scope.ownerId}:${remote}`;
  const now = Date.now();
  const bucket = rateLimitBuckets.get(key) ?? { count: 0, resetAt: now + rateLimitWindowMs };

  if (now > bucket.resetAt) {
    bucket.count = 0;
    bucket.resetAt = now + rateLimitWindowMs;
  }

  bucket.count += 1;
  rateLimitBuckets.set(key, bucket);

  if (bucket.count > rateLimitMaxRequests) {
    response.writeHead(429, {
      ...corsHeaders,
      "content-type": "application/json; charset=utf-8",
      "retry-after": String(Math.ceil((bucket.resetAt - now) / 1000))
    });
    response.end(JSON.stringify({ error: "DynamicChat API rate limit exceeded." }));
    return false;
  }

  return true;
}

async function health(_request, response) {
  sendJson(response, 200, { ok: true, service: "dynamicchat-api", dataDir, rateLimit: { windowMs: rateLimitWindowMs, max: rateLimitMaxRequests } });
}

async function getNovelAiSubscription(request, response) {
  const token = readBearerToken(request);
  if (!token) {
    sendJson(response, 400, { error: "NovelAI API token is required." });
    return;
  }

  const upstream = await fetch(novelAiSubscriptionUrl, {
    method: "GET",
    headers: {
      authorization: `Bearer ${token}`
    }
  });
  const text = await upstream.text();

  if (!upstream.ok) {
    sendJson(response, upstream.status, {
      error: text || `NovelAI subscription request failed: ${upstream.status}`
    });
    return;
  }

  sendJson(response, 200, text ? JSON.parse(text) : {});
}

async function proxyNovelAiGenerateImage(request, response) {
  const token = readBearerToken(request);
  if (!token) {
    sendJson(response, 400, { error: "NovelAI API token is required." });
    return;
  }

  const payload = await readJsonBody(request);
  const upstream = await fetch(novelAiGenerateImageUrl, {
    method: "POST",
    headers: {
      authorization: `Bearer ${token}`,
      "content-type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  const bytes = Buffer.from(await upstream.arrayBuffer());

  response.writeHead(upstream.status, {
    ...corsHeaders,
    "content-type": upstream.headers.get("content-type") ?? "application/octet-stream",
    "content-length": String(bytes.byteLength)
  });
  response.end(bytes);
}

async function getObjectAsset(_request, response, [objectKey]) {
  let objectFile;
  try {
    objectFile = resolveExistingObjectFile(objectKey, await getRequestObjectRoots(_request));
  } catch {
    sendJson(response, 400, { error: "Invalid object key." });
    return;
  }

  if (!objectFile) {
    sendJson(response, 404, { error: "Object not found" });
    return;
  }

  const bytes = await readFile(objectFile.filePath);
  response.writeHead(200, {
    ...corsHeaders,
    "content-type": mimeTypeForObjectKey(objectKey),
    "content-length": String(bytes.byteLength),
    "cache-control": "public, max-age=31536000, immutable"
  });
  response.end(bytes);
}

async function listSimulations(request, response) {
  const store = await readStateStore();
  const scope = readRequestScope(request);
  const simulations = Object.values(store.simulations ?? {}).filter((state) => state?.simulation?.ownerId === scope.ownerId);

  sendJson(response, 200, simulations.map((state) => redactResponseSecrets(stripAssetPayloadsFromState(state))));
}

async function createSimulation(request, response) {
  const state = await readJsonBody(request);
  if (!assertRequestScope(response, request, state)) {
    return;
  }
  await upsertSimulationState(state);
  sendJson(response, 201, redactResponseSecrets(state));
}

async function getSimulation(_request, response, [simulationId]) {
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestOwnerScope(response, _request, state)) {
    return;
  }

  sendJson(response, 200, redactResponseSecrets(stripAssetPayloadsFromState(state)));
}

async function saveSimulationState(request, response, [simulationId]) {
  const state = await readJsonBody(request);
  if (state?.simulation?.id !== simulationId) {
    sendJson(response, 400, { error: "Simulation ID mismatch" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  await upsertSimulationState(state);
  send(response, 204);
}

async function getPersonalApiVault(request, response) {
  const scope = readRequestScope(request);
  const secrets = await readSecretStore();
  const ownerSecrets = secrets.owners?.[scope.ownerId] ?? {};

  sendJson(response, 200, decodePersonalApiVault(ownerSecrets));
}

async function savePersonalApiVault(request, response) {
  const scope = readRequestScope(request);
  const vault = await readJsonBody(request);
  const secrets = await readSecretStore();
  const ownerSecrets = secrets.owners?.[scope.ownerId] ?? {};

  secrets.owners = {
    ...(secrets.owners ?? {}),
    [scope.ownerId]: {
      ...ownerSecrets,
      personalApiVault: encodePersonalApiVault(vault, ownerSecrets.personalApiVault),
      updatedAt: new Date().toISOString()
    }
  };

  await writeSecretStore(secrets);
  send(response, 204);
}

async function createPromptModule(request, response, [simulationId]) {
  const module = await readJsonBody(request);
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  state.modules = [...(state.modules ?? []), module];
  state.simulation.updatedAt = new Date().toISOString();
  appendAuditEvent(state, request, "state_saved", "prompt_module", module.id, { operation: "create_prompt_module" });
  await writeStateStore(store);
  sendJson(response, 201, module);
}

async function updatePromptModule(request, response, [moduleId]) {
  const patch = await readJsonBody(request);
  const store = await readStateStore();
  const { state, module } = findModule(store, moduleId);
  if (!state || !module) {
    sendJson(response, 404, { error: "Prompt module not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  const nextModule = { ...module, ...patch, updatedAt: new Date().toISOString() };
  state.modules = state.modules.map((candidate) => (candidate.id === moduleId ? nextModule : candidate));
  state.simulation.updatedAt = new Date().toISOString();
  appendAuditEvent(state, request, "state_saved", "prompt_module", moduleId, { operation: "update_prompt_module" });
  await writeStateStore(store);
  sendJson(response, 200, nextModule);
}

async function createChatTurn(_request, response) {
  sendJson(response, 501, { error: "Chat turn execution is still handled by the React runtime in this MVP server." });
}

async function resetSession(_request, response) {
  sendJson(response, 501, { error: "Session reset execution is tracked by AIN-19." });
}

async function createImageJob(request, response, [simulationId]) {
  const body = await readJsonBody(request);
  const now = new Date().toISOString();
  const job = {
    id: `imgjob_${Date.now().toString(36)}`,
    simulationId,
    sessionId: body.sessionId ?? "server_pending",
    turnId: body.turnId,
    status: "queued",
    reason: body.reason ?? "Manual image job requested",
    prompt: body.prompt ?? "",
    negativePrompt: body.negativePrompt ?? "",
    providerPayload: body.providerPayload ?? {},
    assetIds: [],
    contextNodeIds: body.contextNodeIds ?? [],
    createdAt: now
  };
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  state.imageJobs = [...(state.imageJobs ?? []), job];
  appendAuditEvent(state, request, "generation_job_created", "image_job", job.id, { status: job.status, source: "server" });
  await writeStateStore(store);
  sendJson(response, 201, job);
}

async function listAssets(request, response, [simulationId]) {
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestOwnerScope(response, request, state)) {
    return;
  }

  const objectRoot = await getRequestObjectRoot(request);
  const requestedAssetIds = readRequestedAssetIds(request);
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  const storedAssets = collectHydratedImageAssets(state).filter((asset) => !deletedAssetIds.has(asset.id));
  const storedAssetIds = new Set(storedAssets.map((asset) => asset.id));
  const recoveredAssets = requestedAssetIds.size === 0
    ? []
    : [...requestedAssetIds]
        .filter((assetId) => !deletedAssetIds.has(assetId))
        .filter((assetId) => !storedAssetIds.has(assetId))
        .map((assetId) => createRecoveredImageAsset(state, assetId));
  const assets = await hydrateAssetCollection(
    simulationId,
    [...storedAssets, ...recoveredAssets]
      .filter((asset) => !deletedAssetIds.has(asset.id))
      .filter((asset) => requestedAssetIds.size === 0 || requestedAssetIds.has(asset.id))
      .map(stripAssetDataUrl),
    objectRoot
  );
  appendAuditEvent(state, request, "asset_accessed", "simulation", simulationId, { assetCount: assets.length });
  void writeStateStore(store).catch(() => undefined);
  sendJson(response, 200, assets);
}

async function persistImageAssets(request, response, [simulationId]) {
  const body = await readJsonBody(request);
  const assets = Array.isArray(body?.assets) ? body.assets : [];
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (state && !assertRequestOwnerScope(response, request, state)) {
    return;
  }

  const objectRoot = await getRequestObjectRoot(request);
  const persistedAssets = await persistAssetObjects(simulationId, assets, objectRoot);
  sendJson(response, 200, persistedAssets.map(stripAssetDataUrl));
}

function readRequestedAssetIds(request) {
  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "127.0.0.1"}`);
  return new Set(
    url.searchParams
      .getAll("ids")
      .flatMap((value) => value.split(","))
      .map((value) => value.trim())
      .filter(Boolean)
  );
}

async function cancelImageJob(request, response, [jobId]) {
  const store = await readStateStore();
  const { state, job } = findImageJob(store, jobId);
  if (!state || !job) {
    sendJson(response, 404, { error: "Image job not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  const nextJob = { ...job, status: "canceled", completedAt: new Date().toISOString() };
  state.imageJobs = state.imageJobs.map((candidate) => (candidate.id === jobId ? nextJob : candidate));
  appendAuditEvent(state, request, "generation_job_completed", "image_job", jobId, { status: "canceled" });
  await writeStateStore(store);
  sendJson(response, 200, nextJob);
}

async function getImageJob(request, response, [jobId]) {
  const store = await readStateStore();
  const { state, job } = findImageJob(store, jobId);
  if (!job) {
    sendJson(response, 404, { error: "Image job not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  sendJson(response, 200, job);
}

async function listAudit(request, response, [simulationId]) {
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  sendJson(response, 200, state.auditLog ?? []);
}

async function createBackup(request, response, [simulationId]) {
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  const backup = await writeBackupFile(store, `manual-${simulationId}`);
  appendAuditEvent(state, request, "backup_created", "backup", backup.id, {
    fileName: backup.fileName,
    simulationCount: Object.keys(store.simulations ?? {}).length
  });
  await writeStateStore(store);
  sendJson(response, 201, backup);
}

async function createRedaction(request, response, [simulationId]) {
  const body = await readJsonBody(request);
  const store = await readStateStore();
  const state = store.simulations[simulationId];
  if (!state) {
    sendJson(response, 404, { error: "Simulation not found" });
    return;
  }
  if (!assertRequestScope(response, request, state)) {
    return;
  }

  const result = await applyRedaction(state, request, body);
  await writeStateStore(store);
  sendJson(response, 200, result);
}

async function upsertSimulationState(inputState) {
  validateAppState(inputState);
  const store = await readStateStore();
  const secrets = await readSecretStore();
  const { state, secretPatch } = await extractPersistentState(inputState, secrets.owners[inputState.simulation.ownerId]);

  appendSecretAuditEvents(state, secretPatch);
  store.simulations[state.simulation.id] = state;
  if (secretPatch) {
    const ownerId = state.simulation.ownerId;
    secrets.owners[ownerId] = {
      ...(secrets.owners[ownerId] ?? {}),
      ...secretPatch
    };
    await writeSecretStore(secrets);
  }

  await writeStateStore(store);
}

async function extractPersistentState(inputState, existingSecrets = {}) {
  const state = structuredClone(inputState);
  const ownerId = state.simulation.ownerId;
  const updatedAt = new Date().toISOString();
  const secretPatch = {};

  if (state.llm?.apiKey) {
    const encodedSecret = Buffer.from(state.llm.apiKey, "utf8").toString("base64");
    if (existingSecrets.llm?.encodedSecret !== encodedSecret || existingSecrets.llm?.provider !== state.llm.provider || existingSecrets.llm?.model !== state.llm.model) {
      secretPatch.llm = {
      provider: state.llm.provider,
      model: state.llm.model,
      secretRef: `dev:${ownerId}:llm`,
      encodedSecret,
      updatedAt
      };
    }
    state.llm.apiKey = "";
  }

  if (state.novelAi?.apiKey) {
    const encodedSecret = Buffer.from(state.novelAi.apiKey, "utf8").toString("base64");
    if (existingSecrets.novelAi?.encodedSecret !== encodedSecret || existingSecrets.novelAi?.accountLabel !== state.novelAi.accountLabel) {
      secretPatch.novelAi = {
      provider: "novelai",
      accountLabel: state.novelAi.accountLabel,
      secretRef: `dev:${ownerId}:novelai`,
      encodedSecret,
      updatedAt
      };
    }
    state.novelAi.apiKey = "";
  }

  const objectRoot = getOwnerObjectRoot(existingSecrets);
  state.imageAssets = await persistAssetObjects(state.simulation.id, state.imageAssets ?? [], objectRoot);
  state.progressRuns = await Promise.all(
    (state.progressRuns ?? []).map(async (run) => ({
      ...run,
      imageAssets: await persistAssetObjects(run.simulationId ?? state.simulation.id, run.imageAssets ?? [], objectRoot)
    }))
  );
  return {
    state,
    secretPatch: Object.keys(secretPatch).length > 0 ? secretPatch : undefined
  };
}

async function persistAssetObjects(simulationId, assets, objectRoot = defaultObjectDir) {
  const simulationObjectDir = path.join(objectRoot, simulationId);
  await mkdir(simulationObjectDir, { recursive: true });

  return Promise.all(
    assets.map(async (asset) => {
      if (!asset.dataUrl) {
        return attachExistingAssetObjectMetadata(simulationId, asset, objectRoot);
      }

      const parsed = parseDataUrl(asset.dataUrl);
      if (!parsed) {
        return asset;
      }

      const extension = extensionForMimeType(parsed.mimeType);
      const objectKey = `${simulationId}/${asset.id}.${extension}`;
      const filePath = path.join(objectRoot, objectKey);
      await mkdir(path.dirname(filePath), { recursive: true });
      await writeFile(filePath, parsed.bytes);

      const { dataUrl: _dataUrl, ...metadataOnly } = asset;
      return {
        ...metadataOnly,
        objectKey,
        mimeType: asset.mimeType ?? parsed.mimeType
      };
    })
  );
}

function attachExistingAssetObjectMetadata(simulationId, asset, objectRoot = defaultObjectDir) {
  const objectFile = resolveAssetObjectFile(simulationId, asset, objectRoot);
  if (!objectFile) {
    return asset;
  }

  return {
    ...asset,
    objectKey: objectFile.objectKey,
    mimeType: asset.mimeType ?? mimeTypeForObjectKey(objectFile.objectKey)
  };
}

function stripAssetPayloadsFromState(state) {
  const deletedAssetIds = collectDeletedImageAssetIds(state);
  return {
    ...state,
    visualProfiles: (state.visualProfiles ?? []).map((profile) => ({
      ...profile,
      referenceImageAssetIds: (profile.referenceImageAssetIds ?? []).filter((assetId) => !deletedAssetIds.has(assetId))
    })),
    messages: (state.messages ?? []).map((message) => filterMessageImageAssets(message, deletedAssetIds)),
    turnTraces: (state.turnTraces ?? []).map((trace) => filterTraceImageAssets(trace, deletedAssetIds)),
    imageAssets: (state.imageAssets ?? []).filter((asset) => !deletedAssetIds.has(asset.id)).map(stripAssetDataUrl),
    imageJobs: (state.imageJobs ?? []).map((job) => filterJobImageAssets(job, deletedAssetIds)),
    progressRuns: (state.progressRuns ?? []).map((run) => ({
      ...run,
      messages: (run.messages ?? []).map((message) => filterMessageImageAssets(message, deletedAssetIds)),
      turnTraces: (run.turnTraces ?? []).map((trace) => filterTraceImageAssets(trace, deletedAssetIds)),
      imageAssets: (run.imageAssets ?? []).filter((asset) => !deletedAssetIds.has(asset.id)).map(stripAssetDataUrl),
      imageJobs: (run.imageJobs ?? []).map((job) => filterJobImageAssets(job, deletedAssetIds))
    }))
  };
}

function stripAssetDataUrl(asset) {
  if (!asset?.dataUrl) {
    return asset;
  }

  const { dataUrl: _dataUrl, ...metadataOnly } = asset;
  return metadataOnly;
}

async function hydrateAssetDataUrls(state, objectRoot = defaultObjectDir) {
  const simulationId = state.simulation?.id;
  return {
    ...state,
    imageAssets: await hydrateAssetCollection(simulationId, state.imageAssets ?? [], objectRoot),
    progressRuns: await Promise.all(
      (state.progressRuns ?? []).map(async (run) => ({
        ...run,
        imageAssets: await hydrateAssetCollection(run.simulationId ?? simulationId, run.imageAssets ?? [], objectRoot)
      }))
    )
  };
}

function collectHydratedImageAssets(state) {
  const assetsById = new Map();
  for (const asset of [
    ...(state.imageAssets ?? []),
    ...(state.progressRuns ?? []).flatMap((run) => run.imageAssets ?? [])
  ]) {
    if (!asset?.id) {
      continue;
    }

    assetsById.set(asset.id, mergeHydratedAsset(assetsById.get(asset.id), asset));
  }

  return Array.from(assetsById.values());
}

function mergeHydratedAsset(existing, next) {
  if (!existing) {
    return next;
  }

  return {
    ...existing,
    ...next,
    dataUrl: existing.dataUrl ?? next.dataUrl,
    objectKey: existing.objectKey ?? next.objectKey,
    mimeType: existing.mimeType ?? next.mimeType
  };
}

function collectDeletedImageAssetIds(state) {
  return new Set(
    (state.redactionQueue ?? [])
      .filter((redaction) => redaction?.targetType === "image_asset" && redaction.status !== "failed")
      .map((redaction) => redaction.targetId)
      .filter(Boolean)
  );
}

function filterMessageImageAssets(message, deletedAssetIds) {
  return {
    ...message,
    imageAssetIds: (message.imageAssetIds ?? []).filter((assetId) => !deletedAssetIds.has(assetId))
  };
}

function filterTraceImageAssets(trace, deletedAssetIds) {
  const removedCount = (trace.imageAssetIds ?? []).filter((assetId) => deletedAssetIds.has(assetId)).length;
  return {
    ...trace,
    imageAssetIds: (trace.imageAssetIds ?? []).filter((assetId) => !deletedAssetIds.has(assetId)),
    metrics: trace.metrics
      ? {
          ...trace.metrics,
          imageAssetCount: Math.max(0, (trace.metrics.imageAssetCount ?? 0) - removedCount)
        }
      : trace.metrics
  };
}

function filterJobImageAssets(job, deletedAssetIds) {
  return {
    ...job,
    assetIds: (job.assetIds ?? []).filter((assetId) => !deletedAssetIds.has(assetId)),
    representativeAssetId: deletedAssetIds.has(job.representativeAssetId) ? undefined : job.representativeAssetId
  };
}

function createRecoveredImageAsset(state, assetId) {
  const imageJob = findStateImageJobForAsset(state, assetId);
  const now = new Date().toISOString();
  return {
    id: assetId,
    simulationId: state.simulation?.id,
    title: `Recovered image ${assetId}`,
    source: "generated",
    prompt: imageJob?.prompt ?? "",
    negativePrompt: imageJob?.negativePrompt ?? "",
    safetyLevel: state.imageProfile?.safetyLevel ?? "safe",
    characterIds: [],
    tags: ["recovered"],
    createdAt: imageJob?.completedAt ?? imageJob?.updatedAt ?? imageJob?.createdAt ?? state.simulation?.updatedAt ?? now,
    jobId: imageJob?.id,
    palette: ["#f4f0e8", "#d7c7aa", "#5f5046"],
    providerMetadata: imageJob?.providerPayload
  };
}

function findStateImageJobForAsset(state, assetId) {
  return [
    ...(state.imageJobs ?? []),
    ...(state.progressRuns ?? []).flatMap((run) => run.imageJobs ?? [])
  ].find((job) => (job.assetIds ?? []).includes(assetId) || job.representativeAssetId === assetId);
}

async function hydrateAssetCollection(simulationId, assets, objectRoot = defaultObjectDir) {
  return Promise.all(assets.map((asset) => hydrateAssetDataUrl(simulationId, asset, objectRoot)));
}

async function hydrateAssetDataUrl(simulationId, asset, objectRoot = defaultObjectDir) {
  if (asset.dataUrl) {
    return asset;
  }

  const objectFile = resolveAssetObjectFile(simulationId, asset, objectRoot);
  if (!objectFile) {
    return asset;
  }

  const bytes = await readFile(objectFile.filePath);
  const mimeType = asset.mimeType ?? mimeTypeForObjectKey(objectFile.objectKey);
  return {
    ...asset,
    objectKey: objectFile.objectKey,
    mimeType,
    dataUrl: `data:${mimeType};base64,${bytes.toString("base64")}`
  };
}

function resolveAssetObjectFile(simulationId, asset, objectRoot = defaultObjectDir) {
  const objectKey = resolveAssetObjectKey(simulationId, asset, objectRoot);
  return objectKey ? resolveExistingObjectFile(objectKey, getFallbackObjectRoots(objectRoot)) : undefined;
}

function resolveAssetObjectKey(simulationId, asset, objectRoot = defaultObjectDir) {
  if (asset.objectKey && objectKeyExists(asset.objectKey, objectRoot)) {
    return asset.objectKey;
  }

  const extensions = preferredAssetExtensions(asset);
  const scopedSimulationIds = Array.from(new Set([asset.simulationId, simulationId].filter(Boolean)));
  for (const scopedSimulationId of scopedSimulationIds) {
    for (const extension of extensions) {
      const candidate = `${scopedSimulationId}/${asset.id}.${extension}`;
      if (objectKeyExists(candidate, objectRoot)) {
        return candidate;
      }
    }
  }

  return findAssetObjectKeyById(asset.id, extensions, objectRoot);
}

function findAssetObjectKeyById(assetId, extensions, objectRoot = defaultObjectDir) {
  if (!assetId) {
    return undefined;
  }

  for (const root of getFallbackObjectRoots(objectRoot)) {
    if (!existsSync(root)) {
      continue;
    }

    for (const entry of readdirSync(root, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }

      for (const extension of extensions) {
        const candidate = `${entry.name}/${assetId}.${extension}`;
        if (resolveExistingObjectFile(candidate, [root])) {
          return candidate;
        }
      }
    }
  }

  return undefined;
}

function objectKeyExists(objectKey, objectRoot = defaultObjectDir) {
  return Boolean(resolveExistingObjectFile(objectKey, getFallbackObjectRoots(objectRoot)));
}

function resolveExistingObjectFile(objectKey, objectRoots = [defaultObjectDir]) {
  for (const objectRoot of uniqueObjectRoots(...objectRoots)) {
    try {
      const filePath = resolveObjectPath(objectKey, objectRoot);
      if (existsSync(filePath)) {
        return { objectKey, objectRoot, filePath };
      }
    } catch {
      continue;
    }
  }

  return undefined;
}

function getFallbackObjectRoots(objectRoot = defaultObjectDir) {
  return uniqueObjectRoots(objectRoot, defaultObjectDir);
}

function uniqueObjectRoots(...values) {
  const roots = [];
  const seen = new Set();
  for (const value of values) {
    const normalized = normalizeImageStoragePath(value);
    if (!normalized) {
      continue;
    }

    const resolved = path.resolve(normalized);
    const key = process.platform === "win32" ? resolved.toLowerCase() : resolved;
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    roots.push(resolved);
  }

  return roots;
}

function preferredAssetExtensions(asset) {
  return Array.from(new Set([extensionForMimeType(asset.mimeType), "png", "jpg", "jpeg", "webp"].filter(Boolean)));
}

function mimeTypeForObjectKey(objectKey) {
  const extension = objectKey.split(".").at(-1)?.toLowerCase();
  if (extension === "jpg" || extension === "jpeg") {
    return "image/jpeg";
  }

  if (extension === "webp") {
    return "image/webp";
  }

  return "image/png";
}

async function writeBackupFile(store, reason) {
  const createdAt = new Date().toISOString();
  const id = `backup_${Date.now().toString(36)}`;
  const fileName = `${createdAt.replace(/[:.]/gu, "-")}_${reason}.json`;
  const filePath = path.join(backupDir, fileName);
  await mkdir(backupDir, { recursive: true });
  await writeFile(
    filePath,
    `${JSON.stringify(
      {
        id,
        reason,
        createdAt,
        schemaVersion: 1,
        store
      },
      null,
      2
    )}\n`
  );

  return {
    id,
    reason,
    createdAt,
    fileName,
    dataDir
  };
}

async function applyRedaction(state, request, body) {
  const targetType = body.targetType;
  const targetId = body.targetId;
  const reason = body.reason ?? "operator redaction";
  const now = new Date().toISOString();
  const scope = readRequestScope(request, state);
  const redaction = {
    id: `redact_${Date.now().toString(36)}`,
    simulationId: state.simulation.id,
    ownerId: state.simulation.ownerId,
    scope,
    targetType,
    targetId,
    reason,
    neuralMapNodeIds: Array.isArray(body.neuralMapNodeIds) ? body.neuralMapNodeIds : [],
    status: "applied",
    createdAt: now,
    completedAt: now
  };

  if (targetType === "memory_event") {
    const memory = state.memoryEvents?.find((event) => event.id === targetId);
    const nodeIds = [targetId, memory?.neuralMapNodeId, ...redaction.neuralMapNodeIds].filter(Boolean);
    redaction.neuralMapNodeIds = [...new Set(nodeIds)];
    state.memoryEvents = (state.memoryEvents ?? []).filter((event) => event.id !== targetId);
    state.contextPacks = (state.contextPacks ?? []).map((pack) => ({
      ...pack,
      evidence: (pack.evidence ?? []).filter((item) => !redaction.neuralMapNodeIds.includes(item.nodeId))
    }));
    state.messages = (state.messages ?? []).map((message) => ({
      ...message,
      referencedNodeIds: (message.referencedNodeIds ?? []).filter((nodeId) => !redaction.neuralMapNodeIds.includes(nodeId))
    }));
    appendAuditEvent(state, request, "memory_redacted", "memory_event", targetId, { nodeIds: redaction.neuralMapNodeIds });
  } else if (targetType === "prompt_module") {
    state.modules = (state.modules ?? []).map((module) =>
      module.id === targetId
        ? {
            ...module,
            title: `${module.title} (redacted)`,
            body: "[redacted]",
            enabled: false,
            activationTags: [],
            tokenPolicy: "disabled",
            updatedAt: now
          }
        : module
    );
    state.contextPacks = (state.contextPacks ?? []).map((pack) => ({
      ...pack,
      evidence: (pack.evidence ?? []).filter((item) => item.nodeId !== targetId)
    }));
    appendAuditEvent(state, request, "prompt_module_redacted", "prompt_module", targetId, {});
  } else if (targetType === "image_asset") {
    const assets = collectImageAssetsById(state, targetId);
    const asset = assets[0];
    state.imageAssets = (state.imageAssets ?? []).filter((candidate) => candidate.id !== targetId);
    state.messages = (state.messages ?? []).map((message) => removeImageAssetIdFromMessage(message, targetId));
    state.turnTraces = (state.turnTraces ?? []).map((trace) => removeImageAssetIdFromTrace(trace, targetId));
    state.imageJobs = (state.imageJobs ?? []).map((job) => removeImageAssetIdFromJob(job, targetId));
    state.visualProfiles = (state.visualProfiles ?? []).map((profile) => ({
      ...profile,
      referenceImageAssetIds: (profile.referenceImageAssetIds ?? []).filter((assetId) => assetId !== targetId)
    }));
    state.progressRuns = (state.progressRuns ?? []).map((run) => ({
      ...run,
      imageAssets: (run.imageAssets ?? []).filter((candidate) => candidate.id !== targetId),
      messages: (run.messages ?? []).map((message) => removeImageAssetIdFromMessage(message, targetId)),
      turnTraces: (run.turnTraces ?? []).map((trace) => removeImageAssetIdFromTrace(trace, targetId)),
      imageJobs: (run.imageJobs ?? []).map((job) => removeImageAssetIdFromJob(job, targetId))
    }));

    const deletedObjectKeys = await deleteImageAssetObjectFiles(request, state.simulation.id, targetId, assets);
    if (deletedObjectKeys.length > 0) {
      redaction.neuralMapNodeIds = deletedObjectKeys;
    }
    appendAuditEvent(state, request, "image_asset_deleted", "image_asset", targetId, { objectKey: asset?.objectKey, objectKeys: deletedObjectKeys });
  } else {
    redaction.status = "failed";
    redaction.error = `Unsupported redaction target: ${targetType}`;
  }

  state.redactionQueue = [...(state.redactionQueue ?? []), redaction];
  state.simulation.updatedAt = now;
  return {
    redaction,
    state: redactResponseSecrets(state)
  };
}

function collectImageAssetsById(state, assetId) {
  return [
    ...(state.imageAssets ?? []),
    ...(state.progressRuns ?? []).flatMap((run) => run.imageAssets ?? [])
  ].filter((asset) => asset?.id === assetId);
}

function removeImageAssetIdFromMessage(message, assetId) {
  return {
    ...message,
    imageAssetIds: (message.imageAssetIds ?? []).filter((candidateId) => candidateId !== assetId)
  };
}

function removeImageAssetIdFromTrace(trace, assetId) {
  const removed = (trace.imageAssetIds ?? []).includes(assetId);
  return {
    ...trace,
    imageAssetIds: (trace.imageAssetIds ?? []).filter((candidateId) => candidateId !== assetId),
    metrics: trace.metrics
      ? {
          ...trace.metrics,
          imageAssetCount: Math.max(0, (trace.metrics.imageAssetCount ?? 0) - (removed ? 1 : 0))
        }
      : trace.metrics
  };
}

function removeImageAssetIdFromJob(job, assetId) {
  return {
    ...job,
    assetIds: (job.assetIds ?? []).filter((candidateId) => candidateId !== assetId),
    representativeAssetId: job.representativeAssetId === assetId ? undefined : job.representativeAssetId
  };
}

async function deleteImageAssetObjectFiles(request, simulationId, assetId, assets) {
  const roots = await getRequestObjectRoots(request);
  const objectKeys = new Set(
    assets
      .flatMap((asset) => [asset.objectKey, resolveAssetObjectKey(simulationId, asset, roots[0] ?? defaultObjectDir)])
      .filter(Boolean)
  );

  for (const objectRoot of roots) {
    for (const extension of ["png", "jpg", "jpeg", "webp"]) {
      const directKey = `${simulationId}/${assetId}.${extension}`;
      if (objectKeyExists(directKey, objectRoot)) {
        objectKeys.add(directKey);
      }
    }

    if (!existsSync(objectRoot)) {
      continue;
    }

    for (const entry of readdirSync(objectRoot, { withFileTypes: true })) {
      if (!entry.isDirectory()) {
        continue;
      }

      for (const extension of ["png", "jpg", "jpeg", "webp"]) {
        const candidateKey = `${entry.name}/${assetId}.${extension}`;
        if (objectKeyExists(candidateKey, objectRoot)) {
          objectKeys.add(candidateKey);
        }
      }
    }
  }

  const deletedObjectKeys = [];
  for (const objectKey of objectKeys) {
    await Promise.all(
      roots.map(async (objectRoot) => {
        try {
          await rm(resolveObjectPath(objectKey, objectRoot), { force: true });
          deletedObjectKeys.push(objectKey);
        } catch {
          // Invalid legacy object keys are ignored during redaction cleanup.
        }
      })
    );
  }

  return [...new Set(deletedObjectKeys)];
}

function resolveObjectPath(objectKey, objectRoot = defaultObjectDir) {
  const root = path.resolve(objectRoot);
  const filePath = path.resolve(root, objectKey);
  if (filePath !== root && !filePath.startsWith(`${root}${path.sep}`)) {
    throw new Error("Invalid object key.");
  }
  return filePath;
}

function parseDataUrl(dataUrl) {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/u);
  if (!match) {
    return undefined;
  }

  return {
    mimeType: match[1],
    bytes: Buffer.from(match[2], "base64")
  };
}

function extensionForMimeType(mimeType) {
  if (mimeType === "image/jpeg") {
    return "jpg";
  }

  if (mimeType === "image/webp") {
    return "webp";
  }

  return "png";
}

function redactResponseSecrets(state) {
  return {
    ...state,
    llm: state.llm ? { ...state.llm, apiKey: "" } : state.llm,
    novelAi: state.novelAi ? { ...state.novelAi, apiKey: "" } : state.novelAi
  };
}

function validateAppState(state) {
  if (!state?.simulation?.id || !state?.simulation?.ownerId) {
    throw new Error("Invalid DynamicChat AppState payload.");
  }
}

function readRequestScope(request, state) {
  const ownerId = request.headers["x-dynamicchat-owner-id"]?.toString() || state?.simulation?.ownerId || "local_user";
  const projectId = request.headers["x-dynamicchat-project-id"]?.toString() || state?.simulation?.id || "local_project";
  return {
    ownerId,
    workspaceId: request.headers["x-dynamicchat-workspace-id"]?.toString() || state?.security?.scope?.workspaceId || "local_workspace",
    projectId,
    environment: request.headers["x-dynamicchat-environment"]?.toString() || state?.security?.scope?.environment || "local"
  };
}

function assertRequestScope(response, request, state) {
  const scope = readRequestScope(request, state);
  if (state?.simulation?.ownerId && state.simulation.ownerId !== scope.ownerId) {
    sendJson(response, 403, { error: "Owner scope mismatch." });
    return false;
  }

  if (state?.security?.scope?.projectId && state.security.scope.projectId !== scope.projectId) {
    sendJson(response, 403, { error: "Project scope mismatch." });
    return false;
  }

  return true;
}

function assertRequestOwnerScope(response, request, state) {
  const scope = readRequestScope(request, state);
  if (state?.simulation?.ownerId && state.simulation.ownerId !== scope.ownerId) {
    sendJson(response, 403, { error: "Owner scope mismatch." });
    return false;
  }

  return true;
}

function appendSecretAuditEvents(state, secretPatch) {
  if (!secretPatch) {
    return;
  }

  if (secretPatch.llm) {
    appendAuditEventFromScope(state, state.security?.scope, "api_secret_stored", "api_secret", "llm", {
      provider: secretPatch.llm.provider,
      model: secretPatch.llm.model,
      secretRef: secretPatch.llm.secretRef
    });
  }

  if (secretPatch.novelAi) {
    appendAuditEventFromScope(state, state.security?.scope, "api_secret_stored", "api_secret", "novelai", {
      provider: secretPatch.novelAi.provider,
      accountLabel: secretPatch.novelAi.accountLabel,
      secretRef: secretPatch.novelAi.secretRef
    });
  }
}

function appendAuditEvent(state, request, action, resourceType, resourceId, metadata = {}) {
  appendAuditEventFromScope(state, readRequestScope(request, state), action, resourceType, resourceId, metadata);
}

function appendAuditEventFromScope(state, scope, action, resourceType, resourceId, metadata = {}) {
  const resolvedScope = scope ?? state.security?.scope ?? {
    ownerId: state.simulation.ownerId,
    workspaceId: "local_workspace",
    projectId: state.simulation.id,
    environment: "local"
  };
  state.auditLog = [
    ...(state.auditLog ?? []),
    {
      id: `audit_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`,
      simulationId: state.simulation.id,
      ownerId: state.simulation.ownerId,
      scope: resolvedScope,
      action,
      resourceType,
      resourceId,
      metadata,
      createdAt: new Date().toISOString()
    }
  ].slice(-300);
}

function findModule(store, moduleId) {
  for (const state of Object.values(store.simulations)) {
    const module = state.modules?.find((candidate) => candidate.id === moduleId);
    if (module) {
      return { state, module };
    }
  }

  return {};
}

function findImageJob(store, jobId) {
  for (const state of Object.values(store.simulations)) {
    const job = state.imageJobs?.find((candidate) => candidate.id === jobId);
    if (job) {
      return { state, job };
    }
  }

  return {};
}

function decodePersonalApiVault(ownerSecrets) {
  const personalVault = ownerSecrets.personalApiVault ?? {};
  const llmByProvider = Object.fromEntries(
    Object.entries(personalVault.llmByProvider ?? {}).map(([provider, record]) => [
      provider,
      decodeSecretRecord(record, "개인 LLM 키가 저장되어 있습니다.")
    ])
  );

  return {
    llmByProvider,
    novelAi: decodeSecretRecord(personalVault.novelAi, "개인 NovelAI 토큰이 저장되어 있습니다."),
    imageStoragePath: normalizeImageStoragePath(personalVault.imageStoragePath),
    updatedAt: personalVault.updatedAt
  };
}

function encodePersonalApiVault(vault, existingVault = {}) {
  const now = new Date().toISOString();
  const providerNames = new Set([
    ...Object.keys(existingVault?.llmByProvider ?? {}),
    ...Object.keys(vault?.llmByProvider ?? {})
  ]);
  const llmByProvider = Object.fromEntries(
    [...providerNames].map((provider) => [
      provider,
      encodeSecretRecord(vault?.llmByProvider?.[provider], now, existingVault?.llmByProvider?.[provider])
    ])
  );

  return {
    llmByProvider,
    novelAi: encodeSecretRecord(vault?.novelAi, now, existingVault?.novelAi),
    imageStoragePath: normalizeImageStoragePath(vault?.imageStoragePath) || normalizeImageStoragePath(existingVault?.imageStoragePath),
    updatedAt: vault?.updatedAt ?? now
  };
}

async function getRequestObjectRoot(request) {
  return (await getRequestObjectRoots(request))[0] ?? defaultObjectDir;
}

async function getRequestObjectRoots(request) {
  const scope = readRequestScope(request);
  const secrets = await readSecretStore();
  return getOwnerObjectRoots(secrets.owners?.[scope.ownerId], secrets);
}

function getOwnerObjectRoot(ownerSecrets = {}) {
  return getOwnerObjectRoots(ownerSecrets)[0] ?? defaultObjectDir;
}

function getOwnerObjectRoots(ownerSecrets = {}, secrets = {}) {
  const knownOwnerRoots = Object.values(secrets.owners ?? {})
    .map((candidate) => normalizeImageStoragePath(candidate?.personalApiVault?.imageStoragePath))
    .filter(Boolean);
  return uniqueObjectRoots(normalizeImageStoragePath(ownerSecrets.personalApiVault?.imageStoragePath), defaultObjectDir, ...knownOwnerRoots);
}

function normalizeImageStoragePath(value) {
  return typeof value === "string" ? value.trim() : "";
}

function decodeSecretRecord(record, savedMessage) {
  const apiKey = record?.encodedSecret ? Buffer.from(record.encodedSecret, "base64").toString("utf8") : "";
  const registrationStatus = record?.registrationStatus ?? (apiKey ? "registered" : "idle");

  return {
    apiKey,
    registrationStatus,
    verifiedAt: record?.verifiedAt,
    verificationMessage: record?.verificationMessage ?? (apiKey ? savedMessage : ""),
    subscriptionTier: record?.subscriptionTier
  };
}

function encodeSecretRecord(record, updatedAt, existingRecord = {}) {
  const apiKey = typeof record?.apiKey === "string" ? record.apiKey.trim() : "";
  const encodedSecret = apiKey ? Buffer.from(apiKey, "utf8").toString("base64") : existingRecord?.encodedSecret ?? "";

  return {
    encodedSecret,
    registrationStatus: record?.registrationStatus ?? existingRecord?.registrationStatus ?? (encodedSecret ? "registered" : "idle"),
    verifiedAt: record?.verifiedAt ?? existingRecord?.verifiedAt,
    verificationMessage: record?.verificationMessage ?? existingRecord?.verificationMessage ?? "",
    subscriptionTier: record?.subscriptionTier ?? existingRecord?.subscriptionTier,
    updatedAt
  };
}

function readBearerToken(request) {
  return (request.headers.authorization ?? "").replace(/^Bearer\s+/iu, "").trim();
}

async function readJsonBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }

  const raw = Buffer.concat(chunks).toString("utf8").trim();
  return raw ? JSON.parse(raw) : {};
}

async function ensureStore() {
  await mkdir(dataDir, { recursive: true });
  await mkdir(defaultObjectDir, { recursive: true });
  await mkdir(backupDir, { recursive: true });
  if (!existsSync(statePath)) {
    await writeStateStore({ simulations: {} });
  }
  if (!existsSync(secretPath)) {
    await writeSecretStore({ owners: {} });
  }
}

async function readStateStore() {
  const raw = await readFile(statePath, "utf8");
  try {
    return JSON.parse(raw);
  } catch {
    const recovered = parseLeadingJsonObject(raw);
    if (!recovered) {
      throw new Error("DynamicChat state store is not valid JSON.");
    }
    await writeStateStore(recovered);
    return recovered;
  }
}

async function writeStateStore(store) {
  stateStoreWriteQueue = stateStoreWriteQueue
    .catch(() => undefined)
    .then(() => writeStateStoreNow(store));
  return stateStoreWriteQueue;
}

async function writeStateStoreNow(store) {
  await mkdir(path.dirname(statePath), { recursive: true });
  const tmpPath = `${statePath}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(tmpPath, `${JSON.stringify(store, null, 2)}\n`);
  await renameWithRetry(tmpPath, statePath);
}

async function renameWithRetry(sourcePath, targetPath) {
  const maxAttempts = 8;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    try {
      await rename(sourcePath, targetPath);
      return;
    } catch (error) {
      const code = error?.code;
      if (attempt >= maxAttempts - 1 || (code !== "EPERM" && code !== "EACCES" && code !== "EBUSY")) {
        throw error;
      }

      await delay(25 * (attempt + 1));
    }
  }
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function parseLeadingJsonObject(raw) {
  const endIndex = findLeadingJsonObjectEnd(raw);
  if (endIndex < 0) {
    return undefined;
  }

  try {
    return JSON.parse(raw.slice(0, endIndex));
  } catch {
    return undefined;
  }
}

function findLeadingJsonObjectEnd(raw) {
  const startIndex = raw.search(/\S/u);
  if (startIndex < 0 || raw[startIndex] !== "{") {
    return -1;
  }

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = startIndex; index < raw.length; index += 1) {
    const character = raw[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === "\"") {
        inString = false;
      }
      continue;
    }

    if (character === "\"") {
      inString = true;
    } else if (character === "{") {
      depth += 1;
    } else if (character === "}") {
      depth -= 1;
      if (depth === 0) {
        return index + 1;
      }
    }
  }

  return -1;
}

async function readSecretStore() {
  return JSON.parse(await readFile(secretPath, "utf8"));
}

async function writeSecretStore(store) {
  await mkdir(path.dirname(secretPath), { recursive: true });
  await writeFile(secretPath, `${JSON.stringify(store, null, 2)}\n`);
}

function send(response, statusCode) {
  response.writeHead(statusCode, corsHeaders);
  response.end();
}

function sendJson(response, statusCode, body) {
  response.writeHead(statusCode, {
    ...corsHeaders,
    "content-type": "application/json; charset=utf-8"
  });
  response.end(JSON.stringify(body));
}
