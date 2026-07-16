# DynamicChat MVP API Boundary

This note documents the first persistent API boundary for AIN-18. The current
React app still has a local fallback, but runtime code now goes through
`src/services/dynamicChatApi.ts` instead of importing browser storage directly.

## Config

Start the local MVP server with:

```bash
pnpm api
```

By default it listens at:

```text
http://127.0.0.1:8788
```

For local development, the browser client reads an optional API base URL from:

```text
localStorage["dynamicchat.apiBaseUrl"]
```

You can set this in the runtime Settings panel under `DynamicChat API`. If the
value is unset, DynamicChat remains in local fallback mode. Browser persistence
redacts LLM and NovelAI API keys before writing app state to `localStorage`.
Browser secret caching is disabled; the MVP server extracts submitted secrets
into `.dynamicchat-data/dev-secrets.json` as a clearly local-only development
substitute, not production encryption.

Persistent state and object files are stored under `.dynamicchat-data/` unless
`DYNAMICCHAT_DATA_DIR` is set.

## Implemented Client Surface

The typed client exposes the MVP endpoint set from the blueprint, plus one local
development state mirror endpoint:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/simulations` | Create or import a simulation |
| `GET` | `/simulations/:id` | Read a simulation |
| `PUT` | `/simulations/:id/state` | Local-dev full-state mirror while granular APIs are filled in |
| `POST` | `/simulations/:id/prompt-modules` | Create a prompt module |
| `PATCH` | `/prompt-modules/:id` | Update a prompt module |
| `POST` | `/simulations/:id/chat/turns` | Process one chat turn |
| `POST` | `/simulations/:id/sessions/reset` | Reset LLM session and restore continuity |
| `POST` | `/simulations/:id/image-jobs` | Create a manual image job |
| `GET` | `/simulations/:id/assets` | List image assets |
| `GET` | `/simulations/:id/audit` | List audit events |
| `POST` | `/simulations/:id/redactions` | Apply a memory/module/asset redaction |
| `POST` | `/simulations/:id/backup` | Write a local store backup |
| `POST` | `/image-jobs/:id/cancel` | Cancel an image job |
| `GET` | `/image-jobs/:id` | Read image job status |

All DynamicChat API calls should include owner/workspace/project scope headers:

```text
x-dynamicchat-owner-id
x-dynamicchat-workspace-id
x-dynamicchat-project-id
x-dynamicchat-environment
```

The local server enforces owner/project scope for stored simulations and applies
an in-memory owner/IP rate limit.

The `/state` endpoint is intentionally marked as a bridge. It allows the frontend
to move away from direct browser persistence before the backend has fully
granular write paths.

The current MVP server returns `501` for chat-turn execution and NeuralMap-backed
session reset because those are tracked in AIN-19 through AIN-21. It does persist
full simulation state, prompt module edits, queued manual image jobs, image job
cancellation, image assets, and generated asset object files.

## Persistence Contract

The first schema draft lives at:

```text
server/migrations/0001_dynamicchat_core.sql
```

The schema keeps image metadata in relational tables and stores binary image
content through object storage keys. Provider payloads, context packs, and trace
metadata use JSONB so the backend can preserve exact external API details while
the app model is still stabilizing.
