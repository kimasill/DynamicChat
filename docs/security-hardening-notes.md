# DynamicChat Security Hardening Notes

## Scope Boundary

DynamicChat now carries a local tenant scope in `AppState.security.scope`.
Every DynamicChat API request includes:

- `x-dynamicchat-owner-id`
- `x-dynamicchat-workspace-id`
- `x-dynamicchat-project-id`
- `x-dynamicchat-environment`

The development API rejects requests when the owner or project scope does not
match the stored simulation.

That scope check is **not** an authentication gate. `readRequestScope` falls
back to the stored `simulation.ownerId` when the header is absent, so a
header-less caller always satisfies it. What actually keeps other origins out is
the `Origin` guard (`rejectNonLocalSecretRequest`).

The guard covers **every handler that touches the persisted state store or the
secret store**, read or write, **plus any handler that can spend something of the
user's**: both `/personal-api-vault` routes, all `/simulations…` routes except the
two 501 stubs (`chat/turns`, `sessions/reset`), both `/image-jobs/:id` routes,
`PATCH /prompt-modules/:id`, `POST /llm/cli-agent`, and the four photoreal relay
routes (`POST /photoreal/comfyui/generate`, `GET …/status`, `POST …/free`,
`POST …/interrupt`).

An earlier pass covered only four read routes, which left `POST /simulations` —
the same `upsertSimulationState` sink as `PUT /simulations/:id/state`, with no
id-vs-URL check — and the destructive `POST /simulations/:id/redactions` open to
any origin.

`POST /llm/cli-agent` is guarded for a different reason than the rest. It is not a
relay: it spawns this machine's own CLI without `--bare`, so the request runs on
the user's *signed-in subscription*. A JSON `POST` with no custom headers is a
simple request needing no preflight, so leaving it open let any page the user
happened to be visiting spend that subscription and read the answer.

The photoreal relay routes are guarded for the same reason. A ComfyUI call carries
no caller credentials — what it spends is this machine's GPU — and `generate`
writes reference images into ComfyUI's input directory.

Deliberately **not** guarded: `GET /health`; the relays that genuinely are
stateless and genuinely do run on credentials the caller supplies in the request,
`/novelai/*` and `POST /llm/chat`; and `GET /objects/:key`, which requires an
object key that only a guarded route hands out and is loaded by the app as an
`<img>` subresource that sends no `Origin`.

Requests with no `Origin` at all (curl, the eval harness) are still allowed —
provided their `Host` header names this machine (see below).

The guard runs three checks, in this order: the `Host` header, the ComfyUI deny
list, then the `Origin` itself.

### The Host header must name this machine (DNS rebinding)

`Origin` alone cannot stop DNS rebinding. A page on
`http://attacker.example:8788` that rebinds its own name to `127.0.0.1` talks to
this server *same-origin*, and a same-origin `GET` carries no `Origin` header at
all — so it passed the guard and could read the decoded keys from
`GET /personal-api-vault`, every transcript from `GET /simulations`, and use
`GET /photoreal/comfyui/status` as an inventory of the local ComfyUI (version,
GPU, every model file) and as a loopback port scanner (`HTTP 404` for an open
port, "cannot connect" for a closed one).

What such a request cannot change is its `Host` header: the browser sends the
name the page was loaded from. The API server listens on `127.0.0.1` only, so the
app reaches it as `localhost`, a `127.x.x.x` address or `[::1]`, on any port.
`rejectNonLocalSecretRequest` now refuses every other `Host` before looking at
`Origin`:

`403 {"error":"forbidden_host","message":…}`

- **`DYNAMICCHAT_ALLOWED_HOSTS`** (comma list) names the extra hosts to accept,
  for a deliberate reverse proxy in front of the API. `name` accepts any port;
  `name:port` accepts only that port. A pasted URL is fine (`http://` and any path
  are ignored). An entry that is not a host is reported on the console at startup
  and ignored.
- Non-loopback IP literals are not exempt. They cannot be rebound, but nothing
  reaches a `127.0.0.1` socket under one except through such a proxy, which is
  what the list is for.
- A request with no `Host` at all is allowed: only a non-browser client can send
  one (Node itself answers `400` to an HTTP/1.1 request without `Host`), and such
  a client is trusted the same way a missing `Origin` is.
- The check belongs to the guard, so it applies to exactly the guarded routes.
  `GET /health`, `GET /objects/:key` and the stateless relays (`/novelai/*`,
  `POST /llm/chat`) still answer any `Host`; none of them returns stored data
  without a key that only a guarded route hands out.

### ComfyUI origins are denied

A loopback `Origin` is not proof that the request came from this app. ComfyUI
serves its web UI from loopback too (`http://127.0.0.1:8188`), and every custom
node installed into it can ship frontend JavaScript that runs on that origin.
Custom nodes install from a public registry in one click, so a malicious or
compromised one is a supply-chain event, not a hypothetical — and before this
rule its script passed the guard exactly as the app did and could read the
decoded keys from `/personal-api-vault`. The photoreal backend makes running
ComfyUI next to DynamicChat the normal setup, so `rejectNonLocalSecretRequest`
refuses these origins on **every** guarded route, not only the relay.

What is denied is a **server** — a host class and a port — not an origin string.
A browser treats every spelling of loopback as a separate origin and ComfyUI
answers on all of them; an earlier version listed exact origin strings, so
`DYNAMICCHAT_COMFYUI_URL=http://localhost:8190` left `http://127.0.0.1:8190` —
the address ComfyUI itself prints and opens — free to read the vault. Now:

- On a **ComfyUI port**, every loopback spelling is denied under `http://` and
  `https://` alike (ComfyUI keeps its port when started with TLS): `localhost`,
  any `127.x.x.x`, `[::1]`, and IPv4-mapped `[::ffff:127.x.x.x]`, in any form
  the URL parser accepts.
- The ComfyUI ports are `8188`, the port of `DYNAMICCHAT_COMFYUI_URL`, and the
  port of every `DYNAMICCHAT_DENIED_ORIGINS` entry — the server cannot guess the
  port of a ComfyUI it has not been told about.
- `DYNAMICCHAT_DENIED_ORIGINS` (comma list) takes origins
  (`http://127.0.0.1:8190`) or bare `host:port` (`127.0.0.1:8190`,
  `[::1]:8190`), the same shape `DYNAMICCHAT_COMFYUI_HOSTS` uses; a bare host
  means port 80. An entry that parses as neither is reported on the console at
  startup instead of being dropped in silence, and so is a
  `DYNAMICCHAT_COMFYUI_URL` that is not an http(s) URL.
- A non-loopback entry (or a non-loopback `DYNAMICCHAT_COMFYUI_URL`) denies that
  host and port under either scheme.

The deny list wins over `DYNAMICCHAT_CORS_ORIGIN`. The refusal has the guard's
usual shape (`403 {"error":"forbidden_origin","message":…}`). On top of the list,
the relay routes refuse a request whose `Origin` is the very ComfyUI it would
drive — same port, and either both loopback (in any spelling) or the same host —
which covers a ComfyUI on any port at the moment it is used. That check used to
compare origin strings too, so a page on `http://localhost:8190` could drive the
ComfyUI it was served from as `http://127.0.0.1:8190`.

NeuralMap calls also include the same owner/workspace/project metadata in
prompt-module sync, simulation-event ingest, context retrieval, and handoff
requests.

## Dev Web Server (Vite)

`npm run dev` serves the app through Vite on `127.0.0.1:5173`, from the project
folder — and the default `DYNAMICCHAT_DATA_DIR` (`.dynamicchat-data/`) is inside
that folder. Vite serves dotfiles, and its default CORS policy answers every
loopback origin, ComfyUI's included. So the same keys and transcripts the API
guard refuses to `http://127.0.0.1:8188` were one request away on the web port:
`/.dynamicchat-data/dev-secrets.json` (the stored provider keys, base64 only),
`state.json` (every simulation) and the files under `backups/` and `objects/`
were served like any other file, with
`Access-Control-Allow-Origin: http://127.0.0.1:8188`. `vite.config.ts` now
sets:

- **`server.cors: false`.** The app's page is served by Vite itself, so
  everything it loads from Vite is same-origin; the API server answers its own
  CORS. No response from the dev server is readable from another origin.
- **`server.fs.deny`** = Vite's defaults (`.env`, `.env.*`, `*.{crt,pem}`,
  `**/.git/**` — an explicit list replaces them, so they are restated) plus
  `**/.dynamicchat-data/**`. Vite answers `403` for the data directory by every
  route it has: the plain path, `/@fs/<absolute path>`, `?import`, `?raw`,
  `?url`, `%2E`-encoded and upper-case spellings (the glob is case-insensitive,
  which matters on Windows).

A data directory moved outside the project is already outside `server.fs.allow`.
One moved somewhere else *inside* the project needs its own `fs.deny` entry.
Vite's own `server.allowedHosts` check already refuses a foreign `Host` on the web
port, so DNS rebinding is covered there without configuration.

## Photoreal Relay (ComfyUI)

The browser cannot call ComfyUI itself — by default ComfyUI answers 403 to a
loopback request whose `Origin` does not match its `Host`, and sends no CORS
headers (both change only with `--enable-cors-header`) — so the API server
relays four fixed operations to it. The relay is an
unauthenticated local service issuing requests on a caller's behalf, so it is
deliberately narrow:

- **Base URL.** Optional; defaults to `DYNAMICCHAT_COMFYUI_URL`, then
  `http://127.0.0.1:8188`. It must be `http`/`https` with no userinfo, path,
  query or fragment, and must name a loopback host (`localhost`, any
  `127.x.x.x`, `[::1]`, `[::ffff:127.x.x.x]` — IP literals and the one name the
  OS resolves locally; not `*.localhost`, which the server would resolve through
  DNS) unless `DYNAMICCHAT_COMFYUI_HOSTS` (comma list of `host` or `host:port`)
  names it. The host of `DYNAMICCHAT_COMFYUI_URL` is trusted as
  operator configuration. A base URL that points at the API server itself is
  refused.
- **Fixed paths only:** `/system_stats`, `/models/<folder>`,
  `/object_info/<node>`, `/upload/image`, `/prompt`, `/history/<id>`, `/view`,
  `/interrupt`, `/queue`, `/free`. Path parameters are validated before they are
  encoded — prompt ids are `[A-Za-z0-9_-]`, node names must start with a letter,
  digit or underscore so `.` and `..` can never become dot segments.
- **No redirects.** Every upstream call uses `redirect: "manual"`; a 3xx is an
  error, never a hop.
- **No browser headers.** Upstream headers are built from nothing, so the
  browser's `Origin`, `Cookie`, `Referer` and `Sec-Fetch-*` never reach ComfyUI.
  (Node's fetch adds its own `sec-fetch-mode: cors`, which ComfyUI ignores.)
- **Node allow-list.** Every `class_type` in a submitted workflow must be one of
  the built-in nodes (those the bundled presets use, plus common core loaders,
  samplers and encoders) or be named in `DYNAMICCHAT_COMFYUI_EXTRA_NODES` (comma
  list; `*` turns the check off). ComfyUI custom node packs include nodes that
  execute arbitrary Python, and a workflow is only JSON naming them. A refusal is
  a 400 listing the refused class types. Workflows are also limited to 1–300
  nodes and 32 levels of nesting per node.
- **References.** At most six, each `{ name, objectKey | dataUrl }`. An
  `objectKey` must be `<segment>/<segment>.(png|jpg|jpeg|webp)` with both
  segments passing `isSafeObjectPathSegment`, and is resolved through
  `resolveExistingObjectFile` / `resolveObjectPath` — never a hand-built path. A
  `dataUrl` must be `data:image/(png|jpeg|webp);base64,`. Either way the bytes
  must decode to at most 20 MB and actually be a PNG, JPEG or WebP (checked by
  magic bytes), so nothing but images lands in ComfyUI's input directory. Every
  reference the workflow uses is checked before the first upload, and a leftover
  `%ref:` placeholder is a 400 before anything is uploaded.
- **Image loaders take only this request's references.** Every `LoadImage`-family
  node, and any node with a string `image` input, must hold exactly a `%ref:<name>%`
  placeholder the request supplied; a file name or a link is a 400 before any
  upload, even with `DYNAMICCHAT_COMFYUI_EXTRA_NODES=*`. Without it a
  `LoadImage → PreviewImage` graph returned any file in ComfyUI's input folder as a
  "render", which the builder could then pick as a character's face reference.
  The guarantee that face references are generated images lives in the app (it
  sends object keys of picked profiles only); a direct local caller's own bytes
  through `%ref:` prove nothing about where an image came from.
- **Guidance is never below 1.** Any `cfg`-like input (cfg, cfg_scale, min_cfg,
  neg_scale, nag_scale, …) below 1, as a literal, a numeric string or a link, is a
  400 listing `rejectedGuidanceInputs`, `*` included. Below 1 the sampler draws
  toward the negative conditioning, so a negative prompt becomes the picture.
- **No LoRA loaders by default.** `LoraLoader` / `LoraLoaderModelOnly` are off the
  default allow-list: no built-in preset uses one, and a LoRA's effect is
  invisible to the app's text safety floor. `DYNAMICCHAT_COMFYUI_EXTRA_NODES` can
  re-admit them.
- **Uploads are content-addressed:** `input/dynamicchat/dc_<name>_<sha256
  prefix>.<ext>`, substituted into the workflow under the name ComfyUI returns.
  LoadImage reads the file when the prompt *executes*, not when it is queued, so
  a fixed `dc_<name>` with overwrite on would let a later job's upload replace
  the face of a job still waiting in ComfyUI's queue. Identical bytes map to one
  file, so a profile image is stored there once however many cuts use it.
- **Results come back from ComfyUI's temp folder.** The built-in graphs end in
  `PreviewImage`, not `SaveImage`. `SaveImage` wrote every cut into ComfyUI's
  `output/` folder for good, beyond the reach of DynamicChat's image deletion: a
  deleted cut, a profile candidate nobody picked and the reference-check images
  all lived on there at full resolution, prompt embedded. `PreviewImage` writes
  to ComfyUI's `temp/` folder, which ComfyUI empties when it starts and when it
  shuts down (ComfyUI 0.40 also drops the temp rows from its asset database at
  the same moments), and the relay downloads it from there with
  `/view?type=temp`. The copy DynamicChat keeps is the one in its own object
  store, which redaction does delete. A custom workflow may still end in
  `SaveImage`; its output then lands in `output/` as before.
- **The declared output node is the result.** The relay returns exactly the
  images of the nodes the client names in `outputNodeIds`, whatever folder
  ComfyUI wrote them to, so a stray `SaveImage` elsewhere in a graph can never
  stand in for the cut. Each returned image carries its `nodeId`. Only when no
  node was named, or the named nodes produced nothing, does the folder decide:
  saved (`output`) images first, previews (`temp`) when nothing was saved.
- **Timeouts and cancellation.** The relay waits `timeoutMs` (default 300 s,
  max 900 s) for upload + queue + render, and at most 20 s more to download the
  images; the client waits `timeoutSeconds + 30 s`, so the relay always answers
  first. On a timeout, or when the browser disconnects mid-render, the prompt is
  deleted from ComfyUI's queue and *then* interrupted — in that order, so a
  still-pending prompt cannot start in between. `interrupt` requires a
  `promptId`: a bare ComfyUI `/interrupt` stops whatever is running, including
  work the user queued from ComfyUI's own UI.
- **Status codes.** Bad input 400; ComfyUI's own validation failure 400, with a
  missing model file named in Korean; a failure while ComfyUI runs the prompt
  500; unreachable or redirect 502; timeout 504; a body over the cap 413. 502
  only ever means nothing was rendered, because the client retries a 502 once —
  a failure after the render (no output image, `/view` refused) is a 500.
- **Status probe.** `GET /photoreal/comfyui/status` returns the ComfyUI version,
  device names and VRAM, the four model-folder listings and the requested node
  map — not `system_stats` wholesale, which carries ComfyUI's command line and
  host details. An unreachable ComfyUI is `200 {"ok":false,"message":…}`.

`node server/comfyui-relay-eval.mjs` checks all of the above against the real
server process and a mock ComfyUI (`server/mock-comfyui.mjs`); no GPU needed. It
also checks the `Host` refusal, every deny-list spelling, and the dev web server
(a real Vite with this project's config, over a throwaway root holding a canary
data directory, so a regression can never print the developer's own data).

### What DynamicChat's deletion still cannot reach in ComfyUI

These copies live in ComfyUI's folders and memory, which DynamicChat has no API to
clean. They are listed so a delete confirmation never promises more than it does:

- **Reference uploads** stay in `input/dynamicchat/` until removed by hand — one
  file per distinct profile image (content-addressed), including profiles later
  cleared or replaced. ComfyUI's API cannot remove a file from `input/` (the
  asset API that ComfyUI 0.40 adds deletes its database record and keeps the
  file), and an upload has to outlive the request that made it anyway, because a
  queued prompt reads it when it runs.
- **Rendered cuts** sit in `temp/` until ComfyUI next starts or shuts down.
- **Prompts** — the full workflow, prompt text included — stay in ComfyUI's
  in-memory `/history` until it restarts, readable by anything that can reach
  ComfyUI's API.

`PreviewImage` has one cost: it writes PNGs with the fastest compression, so a
cut is larger than the old `SaveImage` copy, and that is the size the object
store keeps. Measured on the live ComfyUI, a 768x1024 render came back at
1,930 KB — about 2.4 MB per megapixel, against about 1.45 MB for earlier
`SaveImage` cuts. Re-deflating the same pixel data at zlib level 6 gives
1,186 KB in about 140 ms, should storage ever matter more than that time.

## Secret Handling

Browser API-key caching is disabled. The browser still holds entered keys in
memory for the current session so provider verification and direct runtime calls
can work, but persisted `AppState` is redacted before localStorage writes.

The local API extracts submitted LLM and NovelAI keys into
`.dynamicchat-data/dev-secrets.json` and clears them from persisted state. This
is a development-only secret store with an explicit warning in the settings UI.
Production deployments should replace it with a vault or envelope-encrypted
secret table.

## Audit And Redaction

`AppState.auditLog` records context retrieval, image job lifecycle events, API
secret verification/storage, asset access, backup creation, and
delete/redaction actions.

`AppState.redactionQueue` records applied redactions for:

- memory events and their NeuralMap node IDs
- prompt modules and synced NeuralMap document IDs
- image assets and object keys

The local API exposes:

- `GET /simulations/:id/audit`
- `POST /simulations/:id/redactions`
- `POST /simulations/:id/backup`

The UI can redact memory events from the memory panel, redact prompt modules via
the prompt editor delete action, and delete image assets from the image feedback
rows.

### Profile images

A character's profile image (`visualProfiles[].profileImages[backend]`, plus the
NovelAI reference canvas) is not an `ImageAsset`, so `image_asset` redaction
never reaches it. A face cleared or replaced in the builder used to stay on disk,
still readable at `GET /objects/<key>`. The API server now deletes it itself:

- **On save.** When `PUT /simulations/:id/state` or `POST /simulations` replaces
  a stored simulation, each profile key the old version named and the new one
  does not is deleted from every object root, unless a stored simulation of any
  owner still names it (a copy shares its source's keys) or an image job that
  can still run (`queued`, `planning`, `generating`) names it as a face
  reference. A save older than the stored version (`simulation.updatedAt`)
  releases nothing: the app sends a save without waiting for the previous one,
  so a late, older snapshot would otherwise delete the profile just set.
- **At startup, before listening.** A builder draft folder (`objects/draft_sim_*`
  in the default root and in every owner's `imageStoragePath`) whose contents
  are all older than 7 days loses each profile image no stored simulation
  names, and goes once empty. An abandoned draft disappears; a draft that became
  a simulation keeps the picks it uses.

Only a `<folder>/<file>` key with two safe segments and a
`profile_*.(png|jpg|jpeg|webp)` file name is ever deleted, so story images
(`asset_*`) and traversal-looking keys never are. Folder and file are both
`lstat`-ed and no link is followed: a Windows junction reads as a link, but a
path *through* one reads as a plain file. A store that cannot be read deletes
nothing, and a failed delete is logged without failing the save.

Not covered: picks replaced or abandoned while editing an existing simulation
sit in that simulation's own folder, which nothing sweeps; manual backups still
name deleted keys; an older browser tab saving after a newer one can still point
the stored state at a deleted file until the newer tab saves again.
`node server/profile-objects-eval.mjs` checks the rest against the real server.

## Rate Limit And Backup

The development API has an in-memory per-IP rate limit. Defaults:

- window: `60000ms`
- max requests: `180`

Override with `DYNAMICCHAT_RATE_LIMIT_WINDOW_MS` and
`DYNAMICCHAT_RATE_LIMIT_MAX`.

The key used to include `x-dynamicchat-owner-id`, which made the limit useless:
the caller supplies that header, so rotating it bought a fresh bucket per
request and left a permanent `Map` entry behind each one. Only the remote
address is used now, and expired buckets are swept once the map passes 1000
entries.

Request bodies are capped at 64MiB (`DYNAMICCHAT_MAX_BODY_BYTES`), checked both
against the declared `content-length` and while streaming, since a chunked
request declares nothing. A value that is not a positive finite number — `64MB`,
or an exported-but-empty variable — falls back to the 64MiB default rather than
disabling the cap or refusing every body.

The two checks refuse the request differently. A body with a declared
`content-length` over the cap is refused before anything is read and the caller
gets `413 {"error":"Request body too large.","maxBytes":…}`. A chunked body that
overruns mid-stream can only be stopped by dropping the socket, which takes the
response with it: the caller sees a connection reset with **no status**. The
operator's console names the cause either way.

Manual backups are written to `.dynamicchat-data/backups`. The backup payload
contains the current store plus a schema version so future migration tooling can
read and transform it.
