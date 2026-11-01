# Hermes local smoke test

## Result

**Status: incomplete.** Hermes Agent v0.22.5 was installed, Forge's
`forge-voice-story` skill was copied to `~/.hermes/skills/forge-voice-story/`, and
`hermes skills list` showed it enabled. A session loaded the skill, but the configured
local Ollama model did not return a completed response. Therefore this is evidence of
installation and invocation attempts only, not evidence that the skill's conversation or
keepsake flow works in Hermes.

## Safe test setup

- Model execution was routed to local Ollama; no Pangram or Reducto request was made.
- The prompt used only Forge's fictional blue-cup sample and explicitly excluded personal
  writing and external providers.
- No user passage, provider key, or keepsake content is included in this note.
- Hermes telemetry was declined during setup.

## Observed blockers

- The default `mistral:latest` Ollama model was rejected because Hermes Agent requires a
  context window of at least 64,000 tokens; the model exposed 32,768.
- A 64K local Llama attempt entered Hermes's request path but remained without a completed
  answer during the session.
- A Mistral 64K alias was created and another minimal-toolset attempt was started, but no
  completed answer was captured before this note was written.

## Reproduction target

Use a local model that Hermes reports as having at least 64K context, then start a fresh
session in this repository and invoke the skill with a fictional-only prompt. Example:

```bash
hermes --model forge-llama3.1-8b-64k --provider custom --reasoning none \
  --toolsets skills --skills forge-voice-story
```

Continue with fictional sample content only. Mark the gate complete only after recording
the actual assistant's consent question, one-question-at-a-time behavior, author-control
checks, and final response. A registered skill or a model request that times out is not a
successful demonstration.
