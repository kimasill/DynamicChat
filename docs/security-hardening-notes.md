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

The development API has an in-memory owner/IP rate limit. Defaults:

- window: `60000ms`
- max requests: `180`

Override with `DYNAMICCHAT_RATE_LIMIT_WINDOW_MS` and
`DYNAMICCHAT_RATE_LIMIT_MAX`.

Manual backups are written to `.dynamicchat-data/backups`. The backup payload
contains the current store plus a schema version so future migration tooling can
read and transform it.
