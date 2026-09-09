# Hermes model-guidance profile add-on

This pack is designed for the current schema of:
https://github.com/Noa3/hermes-guidelines-injector

Copy the provider directories from `profiles/` into the repository's `profiles/` directory.
Files such as `claude-family.yaml`, `gemini-family.yaml`, `qwen-family.yaml`, etc. intentionally
replace the current empty/null family profiles with conservative official-provider guidance.

## Design

- Prompt rules are kept short and operational.
- API/runtime recommendations are never represented as fake prompt instructions.
- Exact current models inherit from a provider/family profile where possible.
- Prefixes intentionally allow vendor snapshots and common derived names to map to a base model.
- A future derivative matcher should downgrade confidence for community fine-tunes and quantized builds.

## Important limitation

The repository's current source updater is OpenAI-specific. The source URLs and source markers in
this add-on document provenance, but `/model-guidance update` should NOT be expected to verify these
non-OpenAI sources until provider-specific update adapters and host allowlists are implemented.

## Included providers

Anthropic Claude, Google Gemini, Qwen/Alibaba, DeepSeek, Kimi/Moonshot, GLM/Z.AI,
Mistral, and xAI/Grok.

Meta/Llama and other families were not given behavioral prompt rules in this pack where a sufficiently
specific current official prompting source was not established during this review. The repository's
existing safe null profile can remain in place.
