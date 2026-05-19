# NovelAI Image Pipeline Notes

AIN-22 split DynamicChat image generation into a job lifecycle:

```text
queued -> planning -> generating -> completed | failed | canceled
```

The implementation keeps provider-facing payloads behind the NovelAI adapter and
stores DynamicChat-specific metadata separately in `providerPayload` and asset
metadata.

Before a job is queued, the runtime uses the main simulation LLM sidecar as the
source of `image_cues`. The main LLM must produce final usable NovelAI tags while
it writes the turn response; DynamicChat does not run a later tag planner to
repair missing tags.

## Provider Settings Checked

Official NovelAI image docs describe the core UI/runtime settings DynamicChat
tracks: resolution, number of images, steps, prompt guidance, image2image
strength/noise, undesired content, seeds, and multi-character prompting.

Sources:

- https://docs.novelai.net/en/image/
- https://docs.novelai.net/en/image/sampling/

Relevant implementation fields:

- `width`, `height`
- `n_samples`
- `steps`
- `scale`
- `sampler`
- `noise_schedule`
- `negative_prompt`
- `seed`
- V4 prompt/negative prompt wrapper fields when using NovelAI diffusion 4 models

## DynamicChat Metadata

Each image job stores:

- prompt and negative prompt
- provider payload and DynamicChat payload version
- safety level
- trigger mode and confirmation requirement
- policy warnings and blocked reason
- scene, tags, characters, and visual context from the image cue
- context node IDs
- generated asset IDs and representative asset ID

Generated assets store:

- source prompt and negative prompt
- safety level
- character IDs and tags
- provider metadata
- representative flag
- data URL in browser fallback, or object key when persisted by the local server

## Current Limits

The local MVP validates job state, confirmation, count limits, and one explicit
safety rule before provider calls. It does not yet perform a full NovelAI account
cost estimate. The adapter remains isolated so provider payload details can be
updated without changing the simulation runtime.
