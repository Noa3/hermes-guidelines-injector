# Hermes `model-guidance`

`model-guidance` is a native Hermes Agent plugin that compiles small, local model-compatibility hints into the existing Hermes `pre_llm_call` context channel.

It does **not** copy provider documentation into prompts, call another model, modify Hermes core, or make network requests during normal turns.

## What this plugin does

For every LLM turn, Hermes supplies the active model ID to the plugin. The plugin then performs a deterministic local pipeline:

```text
Hermes model ID
  -> normalize the identifier
  -> resolve provider and upstream model/profile
  -> apply profile inheritance and user overrides
  -> remove rules already covered by Hermes
  -> classify the current task locally
  -> compile bounded base/task guidance
  -> inject only newly activated context through pre_llm_call
```

Normal runtime has:

- zero additional LLM calls;
- zero network requests;
- zero provider discovery requests;
- zero embedding or AI-classification calls;
- fail-open behavior if a profile or runtime path is broken.

Network access is restricted to the explicit `/model-guidance update` command.

## Installation

The current Hermes loader discovers user plugins below `$HERMES_HOME/plugins/` and only loads plugins listed in `plugins.enabled`.

Copy the complete repository directory to:

```text
$HERMES_HOME/plugins/model-guidance/
```

The directory must contain at least:

```text
plugin.yaml
__init__.py
runtime.py
model_guidance_core.py
model_guidance_sources.py
profiles/
sources/
```

On Windows, `$HERMES_HOME` is profile-specific, for example:

```text
C:\Users\<user>\AppData\Local\hermes\profiles\code
```

Enable it in the profile configuration:

```yaml
plugins:
  enabled:
    - model-guidance
```

The included installer can copy the plugin into the detected Hermes homes and profiles:

```bash
python install_hermes.py
```

The installer does not overwrite user override data or secrets. It creates a backup when an older installation of this plugin already exists at the destination.

## Activation-based injection

The default mode is `injection_mode: activation`.

Hermes injects plugin context into the current user message through the official `pre_llm_call` return value. The original user text is not rewritten and no visible assistant or synthetic user message is created.

```text
User text:
Implement the feature and test it.

Internal request context:
Implement the feature and test it.

<model_guidance ...>
...
</model_guidance>
```

### Base model guidance

Base guidance contains stable model/provider behavior that is useful across tasks. It is injected once when:

- a session receives its first real user prompt;
- the active model changes;
- the session switches back to a previously used model;
- the selected profile or user override changes the compiled guidance fingerprint.

The block is explicitly marked as superseding earlier model guidance:

```xml
<model_guidance model="qwen3.8-flash" profile="qwen/qwen3.8-flash" revision="1" activation="2">
Supersedes earlier model_guidance blocks.
- ...
</model_guidance>
```

The activation number is diagnostic metadata. It is deterministic and is not used as a secret or a random identifier.

### Task guidance

Task guidance contains only rules for the current deterministic task scopes, such as:

- `coding`;
- `repository-work`;
- `research`;
- `computer-use`;
- `writing`;
- `long-running-agent`;
- `subagent`;
- `tool-heavy`;
- `kanban`.

It is rendered separately:

```xml
<model_task_guidance scope="current-task" tasks="common,coding,repository-work,tool-heavy">
Supersedes earlier model_task_guidance blocks.
- ...
</model_task_guidance>
```

Identical task scopes and task fingerprints are not injected repeatedly. When the task scope changes, only the new task block is sent. If the new scope has no special rule, the plugin can send a tiny scope-reset marker instead of repeating a large block.

### Session-safe state

Activation state is keyed by Hermes `session_id`, with `task_id` as a fallback for isolated workers. It is not keyed only by the process-global current model.

Each bounded LRU entry tracks:

- effective active model/profile identity;
- activation generation;
- last `turn_id`;
- base guidance fingerprint;
- last task scope signature;
- task guidance fingerprint;
- whether base/task guidance was injected.

The state map is bounded to 256 entries. Parent sessions and subagent sessions therefore cannot contaminate one another. If Hermes/plugin state is lost after a process restart, the next real prompt safely receives base guidance again.

Hermes currently invokes `pre_llm_call` once per user turn before the tool loop. The plugin also uses `turn_id` when supplied, so duplicate hook delivery for one turn does not duplicate context.

### Compatibility mode

For debugging or compatibility with older behavior, configure:

```yaml
plugins:
  entries:
    model-guidance:
      settings:
        injection_mode: every_turn
```

`every_turn` is deliberately not the default.

## Token budgets

The default budgets are intentionally smaller than the previous single combined prompt budget:

```yaml
plugins:
  entries:
    model-guidance:
      settings:
        max_base_guidance_chars: 1800
        max_task_guidance_chars: 700
        max_chars: 3600
        max_rules: 32
        max_family_rules: 16
        max_task_rules: 8
```

`max_chars` remains supported for compatibility with existing configurations. Base and task layers have independent budgets, while the global rule limit still applies across both layers.

The compiler selects rules deterministically by priority and stable rule ID. API/runtime recommendations are never converted into fake prompt instructions.

## Model matching

Profile loading and matching are local and deterministic. No embeddings, fuzzy semantic search, or LLM classification are used.

Matching precedence is:

1. exact official normalized ID;
2. exact explicit alias;
3. known official snapshot/version alias;
4. explicit profile prefix/pattern;
5. recognized derivative of the most specific known exact model;
6. recognized derivative of the most specific known family;
7. provider/family fallback;
8. safe unknown fallback with no invented prompt guidance.

The longest and most specific upstream match wins. For example, a derivative of `qwen3.8-flash` is resolved against that profile rather than the broader `qwen3.8` or `qwen` family profile.

Existing router/provider forms continue to work:

```text
openai/gpt-5.6
openrouter/openai/gpt-5.6
anthropic/claude-opus-5
google/gemini-3.5-flash
qwen/qwen3.8-flash
```

Known provider prefixes are stripped only where the registry says they are provider/router namespaces. Arbitrary path prefixes are not blindly trusted.

## Community and derived model identifiers

The resolver can conservatively map common Hugging Face, Ollama, quantized, fine-tuned, and locally renamed IDs to an upstream profile when the basename contains a known local profile identifier.

### Qwen examples

```text
qwen3.8-flash
  -> exact qwen/qwen3.8-flash

qwen3.8-flash-0902
  -> snapshot/pattern of qwen/qwen3.8-flash

qwen3.8-flash-obliterated
  -> derivative of qwen/qwen3.8-flash
  -> alignment derivative, medium confidence

Qwen/Qwen3.8-27B
  -> exact qwen/qwen3.8-27b

bartowski/Qwen3.8-27B-GGUF
  -> packaging derivative of qwen/qwen3.8-27b

unsloth/Qwen3.8-27B-bnb-4bit
  -> packaging derivative of qwen/qwen3.8-27b

qwen3.8:27b
  -> Ollama-style packaging derivative of qwen/qwen3.8-27b
```

The same approach handles examples such as:

```text
claude-opus-5-custom
  -> derivative of anthropic/claude-opus-5

gemini-3.5-flash-local
  -> derivative of google/gemini-3.5-flash

deepseek-v4-pro-abliterated
  -> alignment derivative of deepseek/deepseek-v4-pro

kimi-k2.7-code-custom
  -> community derivative when a matching local profile exists

glm-5.1-fp8
  -> packaging derivative of glm/glm-5.1

mistral-medium-3-5-awq
  -> packaging derivative of mistral/mistral-medium-3-5

grok-4.6-local
  -> community derivative of xai/grok-4.6
```

A basename is accepted only when it matches an exact profile, alias, or explicit profile prefix already present in the local registry. An unrelated community namespace does not cause an arbitrary model to inherit a profile.

### Derivative types and confidence

Diagnostics distinguish at least:

- `packaging`: GGUF, AWQ, GPTQ, EXL2, MLX, FP8, INT4/INT8, BNB, Q4/Q8, and similar packaging markers;
- `alignment`: abliterated, uncensored, DPO, SFT, LoRA, merged, roleplay, and similar post-training markers;
- `community`: custom, local, community, Hugging Face, or other local naming markers.

Exact and official pattern matches have high confidence. Derivative matches have medium confidence. Provider/family fallback has low confidence. The plugin does not claim that a fine-tune behaves identically to its upstream model.

A profile recommendation may set `derivative_safe: false`. Such a rule is skipped for derivative matches while family and explicitly safe rules remain eligible. Existing rules default to `derivative_safe: true` for backward compatibility.

The derivative marker registry is data-driven in:

```text
sources/derivative-modifiers.yaml
```

It can be extended without scattering new string checks through the resolver.

## Profiles and inheritance

Profiles are JSON-compatible YAML under `profiles/<provider>/`. JSON-compatible files keep normal operation independent of PyYAML.

A profile can contain:

- `provider`;
- `model_family`;
- `exact_model_id`;
- `aliases`;
- `match_prefixes`;
- `inherits`;
- `remove_rules` and `negative_overrides`;
- `prompt_recommendations`;
- `runtime_recommendations`;
- source/provenance metadata;
- `profile_revision` and `confidence`;
- optional `derivative_safe` flags on recommendations.

Inheritance is explicit. Parent profiles are compiled before the child profile, then stable rule IDs in `remove_rules` or `negative_overrides` are removed. A newer profile does not silently inherit an old workaround.

Managed updates are written separately from bundled profiles and user overrides:

```text
$HERMES_HOME/plugin-data/model-guidance/managed-profiles/<provider>/<model>.yaml
$HERMES_HOME/plugin-data/model-guidance/user-overrides/<provider>/<model>.yaml
```

User overrides remain protected and win over managed/bundled data for the same provider/model key.

## Prompt vs. runtime recommendations

Recommendations use one of these classifications:

- `PROMPT_APPLICABLE`: may be compactly injected;
- `RUNTIME_APPLICABLE`: belongs to request/API configuration and is not injected as prompt text;
- `INFORMATIONAL`: diagnostic only;
- `UNSUPPORTED_BY_CURRENT_HERMES`: useful provider information that the current plugin API cannot safely apply.

Hermes `pre_api_request` is an observer in the current API. The plugin therefore does not claim to mutate `reasoning_effort`, verbosity, compaction, tools, or other API parameters through that hook.

## Hermes overlap filtering

`sources/hermes-overlap.yaml` records reviewed guidance already supplied by Hermes itself. Overlap rules are reported diagnostically but are not injected a second time.

The overlap registry is compatibility metadata, not a second system prompt.

## Official source verification

`source_marker` belongs to the individual profile recommendation and is used only by explicit source updates.

Source markers:

- are never injected into the target model context;
- are checked against the current official source during `/model-guidance update`;
- produce verified/missing diagnostics;
- cannot create arbitrary new rule IDs or rule text from remote documentation.

Normal turns never fetch or parse provider documentation.

## Commands

```text
/model-guidance status
/model-guidance show
/model-guidance stats
/model-guidance test <model-id>
/model-guidance models
/model-guidance sources
/model-guidance reload
/model-guidance update
/model-guidance update --offline
```

`/model-guidance test <model-id>` is fully offline and reports:

- raw and normalized IDs;
- upstream model/profile;
- provider and family;
- match kind and confidence;
- derivative type and modifiers;
- inherited profiles;
- selected rules and character count.

`status` additionally reports activation generation, base/task fingerprints, injection state, current task scopes, cache state, and source-anchor state. This metadata is not sent to the target model.

## Official sources and updates

`sources/providers.yaml` is the allowlisted source registry. The current bounded source updater has an OpenAI adapter and uses official HTTPS hosts only.

```text
/model-guidance update
```

performs the explicit update workflow:

1. load the local source registry;
2. fetch only allowlisted HTTPS sources;
3. enforce timeout, redirect, and size limits;
4. calculate source hashes and review metadata;
5. verify only profile-declared `source_marker` values;
6. accept only bounded compiler-owned profile changes;
7. write managed overlays, never user overrides;
8. invalidate local caches safely.

`/model-guidance update --offline` validates the update path without network access.

Remote text is treated as untrusted data. It is never executed as code, registered as a tool, or interpreted as a free-form prompt rule generator.

## Development and tests

Install local development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the repository tests:

```bash
python -m pytest -q
python -m compileall -q .
```

The suite uses simulated IDs, local profiles, mocked hooks, temporary overlay directories, and network guards. It deliberately does not call GPT, Claude, Gemini, Qwen, DeepSeek, Kimi, GLM, Mistral, Grok, or other paid/unavailable endpoints.

Covered behavior includes:

- profile and source-marker validation;
- provider/router resolution;
- exact, alias, prefix, snapshot, and derivative matching;
- Qwen/Hugging Face/Ollama examples;
- conservative unknown fallback;
- inheritance and user overrides;
- Hermes overlap filtering;
- base/task layer compilation;
- activation-based injection and task deduplication;
- model switching and switching back;
- profile revision invalidation;
- session and subagent isolation;
- same-turn idempotency;
- bounded LRU state;
- `every_turn` compatibility mode;
- fail-open runtime hooks;
- offline update and source-anchor validation;
- zero normal-runtime network and LLM calls.

GitHub Actions runs the test and compile checks on Python 3.10, 3.11, and 3.12.

## Limitations

- Hermes `pre_api_request` is observer-only; runtime/API parameters are not mutated by this plugin.
- Activation state is process-local and intentionally bounded. A process restart may re-inject base guidance once.
- If Hermes cannot provide a stable `session_id` or `task_id`, the plugin uses safe request-local behavior and may re-inject rather than risk stale suppression.
- Fine-tuned, merged, uncensored, or abliterated derivatives can materially differ from their upstream model. Upstream guidance is best effort only.
- Quantization may change quality and tool behavior even when upstream guidance remains generally useful.
- Unknown names are not aggressively fuzzy-matched and receive no invented model-specific prompt guidance.
- Source updates are explicit and currently provider-adapter driven; documentation text is never freely converted into new prompt rules.
- The plugin must be listed in `plugins.enabled`; Hermes user plugins are opt-in.

## Adding a provider or model

1. Verify a real official source and record it in `sources/providers.yaml`.
2. Add a bounded adapter only when the source structure requires it.
3. Add an explicit profile under `profiles/<provider>/`.
4. Keep prompt and runtime/API recommendations separate.
5. Mark fragile exact-model rules with `derivative_safe: false` when appropriate.
6. Add aliases and prefixes conservatively.
7. Add simulated resolver/runtime tests.
8. Run pytest, compileall, and the offline update tests.

Do not add a fuzzy or semantic matcher to compensate for missing profile data. A safe unknown result is preferable to silently applying the wrong model's instructions.
