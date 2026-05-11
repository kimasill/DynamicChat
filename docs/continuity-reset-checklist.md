# DynamicChat Session Reset Continuity Checklist

Use this checklist for AIN-19 manual verification until the automated eval
harness exists.

## Setup

1. Start the app.
2. Optionally start `pnpm api` and set `DynamicChat API` to
   `http://127.0.0.1:4318`.
3. Keep NeuralMap disabled for local fallback verification, then repeat with
   NeuralMap enabled when the API is available.

## 30-Turn Scenario

Run at least 30 turns that establish:

- one named character relationship
- one current mood or emotional stance
- one location or scene state
- one promise or unresolved task
- one important object or clue

At turn 30, click session reset.

## Expected Result

- A system message appears for the new session.
- The Memory panel switches to the latest Context Pack.
- A `Session handoff` block appears in the Memory panel.
- The handoff shows previous session ID to next session ID.
- Continuity facts show `OK` for relationship, mood, and recent memory when the
  evidence exists in the local/NeuralMap context.
- Any missing continuity facts appear as `WARN` with a warning in the system
  message.

## Pass Criteria

The reset passes for MVP if DynamicChat can continue without replaying the full
transcript and the inspector shows which handoff/context evidence preserved the
state.
