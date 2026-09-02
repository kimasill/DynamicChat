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
`PATCH /prompt-modules/:id`, and `POST /llm/cli-agent`.

An earlier pass covered only four read routes, which left `POST /simulations` —
the same `upsertSimulationState` sink as `PUT /simulations/:id/state`, with no
id-vs-URL check — and the destructive `POST /simulations/:id/redactions` open to
any origin.

`POST /llm/cli-agent` is guarded for a different reason than the rest. It is not a
relay: it spawns this machine's own CLI without `--bare`, so the request runs on
the user's *signed-in subscription*. A JSON `POST` with no custom headers is a
simple request needing no preflight, so leaving it open let any page the user
happened to be visiting spend that subscription and read the answer.

Deliberately **not** guarded: `GET /health`; the relays that genuinely are
stateless and genuinely do run on credentials the caller supplies in the request,
`/novelai/*` and `POST /llm/chat`; and `GET /objects/:key`, which requires an
object key that only a guarded route hands out and is loaded by the app as an
`<img>` subresource that sends no `Origin`.

Requests with no `Origin` at all (curl, the eval harness) are still allowed.

NeuralMap calls also include the same owner/workspace/project metadata in
prompt-module sync, simulation-event ingest, context retrieval, and handoff
requests.

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
