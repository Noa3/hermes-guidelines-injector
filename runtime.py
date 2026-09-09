"""Hermes runtime integration for model-guidance."""
from __future__ import annotations

from dataclasses import asdict
import logging
import os
from pathlib import Path
import shlex
from typing import Any, Mapping

try:  # Hermes loads this directory as a namespaced package.
    from .model_guidance_core import (
        COMPILER_VERSION,
        CompilationResult,
        HermesOverlap,
        ProfileRepository,
        compile_guidance,
        detect_tasks,
        estimate_tokens,
        load_data,
        runtime_rule_observed,
    )
    from .model_guidance_sources import SourceUpdater
except ImportError:  # local unit tests may import modules from the repository root.
    from model_guidance_core import (  # type: ignore
        COMPILER_VERSION,
        CompilationResult,
        HermesOverlap,
        ProfileRepository,
        compile_guidance,
        detect_tasks,
        estimate_tokens,
        load_data,
        runtime_rule_observed,
    )
    from model_guidance_sources import SourceUpdater  # type: ignore

LOGGER = logging.getLogger(__name__)


class ModelGuidanceRuntime:
    def __init__(self, ctx: Any, *, plugin_root: Path | None = None, data_root: Path | None = None):
        self.ctx = ctx
        self.plugin_root = Path(plugin_root or Path(__file__).resolve().parent)
        self.data_root = Path(data_root or self._default_data_root())
        self.enabled = self._as_bool(self._config("enabled", True), True)
        self.max_chars = self._bounded_int(self._config("max_chars", 3600), 200, 12000, 3600)
        self.max_rules = self._bounded_int(self._config("max_rules", 32), 0, 128, 32)
        self.max_family_rules = self._bounded_int(self._config("max_family_rules", 16), 0, 64, 16)
        self.max_task_rules = self._bounded_int(self._config("max_task_rules", 8), 0, 32, 8)
        self.timeout = self._bounded_float(self._config("source_timeout_seconds", 8.0), 1.0, 30.0, 8.0)
        self.max_bytes = self._bounded_int(self._config("source_max_bytes", 2_000_000), 10000, 2_000_000, 2_000_000)
        self.repository = ProfileRepository(
            self.plugin_root / "profiles",
            self.data_root / "managed-profiles",
            self.data_root / "user-overrides",
        )
        self.overlap = self._load_overlap()
        self.last_result: CompilationResult | None = None
        self.last_model_id = ""
        self.last_runtime_observation: dict[str, Any] = {}
        self.last_error = ""
        self._compile_cache: dict[tuple[Any, ...], CompilationResult] = {}
        self._compile_cache_limit = 128
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cache_hit = False
        self._overlap_signature = self._stat_signature(self.plugin_root / "sources" / "hermes-overlap.yaml")
        self._config_signature = self._runtime_config_signature()

    def _default_data_root(self) -> Path:
        try:
            from hermes_constants import get_hermes_home  # type: ignore
            home = get_hermes_home()
        except Exception:
            home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
        return Path(home) / "plugin-data" / "model-guidance"

    def _config(self, key: str, default: Any) -> Any:
        getter = getattr(self.ctx, "get_config", None)
        if not callable(getter):
            return default
        try:
            return getter(key, default=default)
        except TypeError:
            try:
                return getter(key)
            except Exception:
                return default
        except Exception:
            return default

    @staticmethod
    def _bounded_int(value: Any, low: int, high: int, default: int) -> int:
        try:
            return max(low, min(high, int(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bounded_float(value: Any, low: float, high: float, default: float) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return default

    def _load_overlap(self) -> HermesOverlap:
        try:
            raw = load_data(self.plugin_root / "sources" / "hermes-overlap.yaml")
            return HermesOverlap(raw.get("rules", {}))
        except Exception as exc:
            self.last_error = f"overlap registry unavailable: {exc}"
            return HermesOverlap()

    @staticmethod
    def _stat_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def _runtime_config_signature(self) -> tuple[Any, ...]:
        return (
            self._as_bool(self._config("enabled", True), True),
            self._bounded_int(self._config("max_chars", 3600), 200, 12000, 3600),
            self._bounded_int(self._config("max_rules", 32), 0, 128, 32),
            self._bounded_int(self._config("max_family_rules", 16), 0, 64, 16),
            self._bounded_int(self._config("max_task_rules", 8), 0, 32, 8),
        )

    def _clear_cache(self) -> None:
        self._compile_cache.clear()
        self.last_cache_hit = False
        self.last_result = None

    def _refresh_local_state(self) -> None:
        if self.repository.refresh_if_changed():
            self._clear_cache()
        overlap_path = self.plugin_root / "sources" / "hermes-overlap.yaml"
        overlap_signature = self._stat_signature(overlap_path)
        if overlap_signature != self._overlap_signature:
            self.overlap = self._load_overlap()
            self._overlap_signature = overlap_signature
            self._clear_cache()
        config_signature = self._runtime_config_signature()
        if config_signature != self._config_signature:
            self.enabled, self.max_chars, self.max_rules, self.max_family_rules, self.max_task_rules = config_signature
            self._config_signature = config_signature
            self._clear_cache()

    @staticmethod
    def _observation_signature(observation: Mapping[str, Any]) -> tuple[Any, ...]:
        request = observation.get("request", {})
        if not isinstance(request, Mapping):
            request = {}
        return (
            str(observation.get("api_mode", "")),
            bool(request.get("reasoning") or request.get("reasoning_effort")),
            request.get("parallel_tool_calls"),
            bool(request.get("prompt_cache_key")),
        )

    def _compile_cached(
        self,
        model: str,
        *,
        provider: str = "",
        user_message: str = "",
        conversation_history: Any = None,
        platform: str = "",
    ) -> CompilationResult:
        self._refresh_local_state()
        history = conversation_history or []
        tasks = detect_tasks(user_message, history, platform)
        key = (
            model,
            provider,
            tasks,
            self._observation_signature(self.last_runtime_observation),
            COMPILER_VERSION,
            self.repository.generation,
            self._overlap_signature,
            self.max_chars,
            self.max_rules,
            self.max_family_rules,
            self.max_task_rules,
        )
        cached = self._compile_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self.last_cache_hit = True
            return cached
        self.cache_misses += 1
        self.last_cache_hit = False
        match = self.repository.resolve(model, provider)
        result = compile_guidance(
            self.repository,
            match,
            user_message=user_message,
            conversation_history=history,
            platform=platform,
            max_chars=self.max_chars,
            max_rules=self.max_rules,
            max_family_rules=self.max_family_rules,
            max_task_rules=self.max_task_rules,
            hermes_overlap=self.overlap,
            runtime_observation=self.last_runtime_observation,
        )
        if len(self._compile_cache) >= self._compile_cache_limit:
            self._compile_cache.pop(next(iter(self._compile_cache)))
        self._compile_cache[key] = result
        return result

    def on_session_start(self, **kwargs: Any) -> None:
        # This is diagnostic-only.  pre_llm_call always uses its own model kwarg
        # as authoritative, so a model switch cannot use stale state.
        model = kwargs.get("model")
        if isinstance(model, str):
            self.last_model_id = model

    def pre_llm_call(self, **kwargs: Any) -> dict[str, str] | None:
        try:
            self._refresh_local_state()
        except Exception as exc:
            self.last_error = f"local cache refresh failed; using current state: {exc}"
            LOGGER.warning("model-guidance cache refresh failed", exc_info=True)
        if not self.enabled:
            return None
        model = kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            self.last_error = "Hermes did not supply a model ID for this turn"
            return None
        try:
            self.last_model_id = model
            result = self._compile_cached(
                model,
                provider=str(kwargs.get("provider", "")),
                user_message=str(kwargs.get("user_message", "")),
                conversation_history=kwargs.get("conversation_history") or [],
                platform=str(kwargs.get("platform", "")),
            )
            self.last_result = result
            self.last_error = "; ".join(result.errors)
            return {"context": result.injected_text} if result.injected_text else None
        except Exception as exc:  # fail open: never break Hermes' agent loop
            self.last_error = f"guidance disabled for this turn: {exc}"
            LOGGER.exception("model-guidance pre_llm_call failed")
            return None

    def pre_api_request(self, **kwargs: Any) -> None:
        """Observer only; Hermes ignores this hook's return value by contract."""
        try:
            request = kwargs.get("request")
            safe_request: dict[str, Any] = {}
            if isinstance(request, Mapping):
                for key in ("reasoning", "reasoning_effort", "parallel_tool_calls", "prompt_cache_key", "api_mode"):
                    if key in request and isinstance(request[key], (str, int, float, bool, type(None), dict)):
                        safe_request[key] = request[key]
            self.last_runtime_observation = {
                "model": str(kwargs.get("model", "")),
                "provider": str(kwargs.get("provider", "")),
                "api_mode": str(kwargs.get("api_mode", "")),
                "request": safe_request,
            }
        except Exception:
            LOGGER.debug("model-guidance could not inspect pre_api_request", exc_info=True)

    def command(self, raw_args: str) -> str:
        try:
            args = shlex.split(raw_args or "")
        except ValueError as exc:
            return f"Invalid /model-guidance arguments: {exc}"
        action = args[0].lower() if args else "status"
        try:
            if action in {"help", "?"}:
                return self._help()
            if action == "status":
                return self._status()
            if action == "show":
                return self._show()
            if action == "stats":
                return self._stats()
            if action == "models":
                return self._models()
            if action == "sources":
                return self._sources()
            if action == "reload":
                self.repository.reload()
                self.overlap = self._load_overlap()
                self._overlap_signature = self._stat_signature(self.plugin_root / "sources" / "hermes-overlap.yaml")
                self._clear_cache()
                return f"Reloaded {len(self.repository.profiles)} local model profiles."
            if action == "test":
                return self._test(args[1] if len(args) > 1 else "")
            if action == "update":
                offline = "--offline" in args[1:]
                updater = SourceUpdater(self.plugin_root, self.data_root, timeout=self.timeout, max_bytes=self.max_bytes)
                result = updater.update(self.repository, offline=offline)
                self.repository.reload()
                self._clear_cache()
                return result.render()
            return f"Unknown /model-guidance action: {action}\n\n{self._help()}"
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("model-guidance command failed")
            return f"model-guidance failed safely: {exc}"

    def _current_or_unknown(self) -> CompilationResult | None:
        self._refresh_local_state()
        if self.last_result and self.last_result.match.raw_model_id == self.last_model_id:
            return self.last_result
        if self.last_model_id:
            result = self._compile_cached(self.last_model_id)
            self.last_result = result
            return result
        return self.last_result

    def _status(self) -> str:
        result = self._current_or_unknown()
        if result is None:
            return "MODEL GUIDANCE STATUS\n\nNo model has been supplied to the plugin in this session. Use /model-guidance test <model-id> for an offline lookup."
        injected = len(result.active_prompt)
        runtime_applied = sum(1 for item in result.runtime_recommendations if self._runtime_applied(item.id))
        anchor_counts = self._source_anchor_counts(result.match.profile)
        lines = [
            "MODEL GUIDANCE STATUS",
            f"Enabled: {self.enabled}",
            f"Active model: {result.match.raw_model_id}",
            f"Normalized model: {result.match.normalized_model_id}",
            f"Provider: {result.match.provider}",
            f"Family: {result.match.model_family}",
            f"Profile: {result.match.profile.provider + '/' + result.match.profile.exact_model_id if result.match.profile else 'none'}",
            f"Match: {result.match.match_kind} ({result.match.confidence})",
            f"Sources reviewed: {', '.join(result.match.profile.source_urls) if result.match.profile else 'none'}",
            f"Source anchors: verified: {anchor_counts[0]}, missing: {anchor_counts[1]}, total: {anchor_counts[2]}, unverified/no-marker: {anchor_counts[3]}",
            f"Prompt recommendations considered: {injected + len(result.hermes_handled)}",
            f"Already handled by Hermes: {len(result.hermes_handled)}",
            f"Injected: {injected}",
            f"Runtime recommendations: {len(result.runtime_recommendations)}",
            f"Runtime recommendations observed/applied: {runtime_applied}",
            f"Informational: {len(result.informational_recommendations)}",
            f"Unsupported by current Hermes: {len(result.unsupported_recommendations)}",
            f"Injected characters: {result.characters}",
            f"Estimated injected tokens: {estimate_tokens(result.injected_text)}",
            f"Cache: {'hit' if self.last_cache_hit else 'miss'} (hits={self.cache_hits}, misses={self.cache_misses})",
            f"Task scopes: {', '.join(result.tasks)}",
        ]
        if self.last_error:
            lines.append(f"Diagnostics: {self.last_error}")
        return "\n".join(lines)

    def _stats(self) -> str:
        result = self._current_or_unknown()
        lines = [
            "MODEL GUIDANCE RUNTIME STATS",
            "Additional LLM calls: 0",
            "Normal-runtime network requests: 0",
            "Documentation lookup during normal turns: 0",
            f"Resolver cache: hits={self.repository.resolve_cache_hits}, misses={self.repository.resolve_cache_misses}",
            f"Compiler cache: hits={self.cache_hits}, misses={self.cache_misses}, entries={len(self._compile_cache)}/{self._compile_cache_limit}",
            f"Compiler version: {COMPILER_VERSION}",
        ]
        if result is None:
            lines.append("Model: none supplied")
        else:
            lines.extend([
                f"Model: {result.match.raw_model_id}",
                f"Profile: {result.match.profile.provider + '/' + result.match.profile.exact_model_id if result.match.profile else 'none'}",
                f"Task scopes: {', '.join(result.tasks)}",
                f"Injected rules: {len(result.active_prompt)}",
                f"Injected characters: {result.characters}",
                f"Estimated injected tokens: {estimate_tokens(result.injected_text)}",
                f"Truncated by limits: {result.truncated}",
            ])
        return "\n".join(lines)

    def _show(self) -> str:
        result = self._current_or_unknown()
        if result is None:
            return "No active guidance. Use /model-guidance test <model-id> for an offline compilation preview."
        def bullets(items: Any) -> list[str]:
            return [f"- {item.id}: {item.text}" for item in items]
        profile = result.match.profile
        provenance = [f"- {url}" for url in (profile.source_urls if profile else ())] or ["- (none)"]
        lines = [
            "ACTIVE PROMPT GUIDANCE",
            result.injected_text or "(none)",
            "",
            "HERMES-HANDLED GUIDANCE",
            *(bullets(result.hermes_handled) or ["(none)"]),
            "",
            "RUNTIME SETTINGS",
            *(bullets(result.runtime_recommendations) or ["(none)"]),
            "",
            "UNSUPPORTED RECOMMENDATIONS",
            *(bullets(result.unsupported_recommendations) or ["(none)"]),
            "",
            "INFORMATIONAL RECOMMENDATIONS",
            *(bullets(result.informational_recommendations) or ["(none)"]),
            "",
            "SOURCE PROVENANCE",
            *provenance,
        ]
        return "\n".join(lines)

    def _test(self, model_id: str) -> str:
        if not model_id.strip():
            return "Usage: /model-guidance test <model-id>\nThis performs only local matching and compilation; it never contacts a provider."
        result = self._compile_cached(
            model_id,
            user_message="Implement and verify a change in an existing repository.",
            platform="cli",
        )
        self.last_result = result
        self.last_model_id = model_id
        return "\n".join([
            "MODEL GUIDANCE OFFLINE TEST",
            "No provider/model request was made.",
            f"Raw model: {result.match.raw_model_id}",
            f"Normalized: {result.match.normalized_model_id}",
            f"Provider: {result.match.provider}",
            f"Family: {result.match.model_family}",
            f"Profile: {result.match.profile.provider + '/' + result.match.profile.exact_model_id if result.match.profile else 'none'}",
            f"Match kind: {result.match.match_kind}",
            f"Inherited: {', '.join(result.inherited_profiles) or '(none)'}",
            f"Injected rules: {len(result.active_prompt)}",
            f"Hermes-suppressed rules: {len(result.hermes_handled)}",
            f"Runtime rules: {len(result.runtime_recommendations)}",
            f"Unsupported rules: {len(result.unsupported_recommendations)}",
            f"Characters: {result.characters}",
            "",
            result.injected_text or "(no prompt guidance)",
        ])

    def _models(self) -> str:
        lines = ["LOCAL MODEL PROFILES", ""]
        for profile in sorted(self.repository.profiles, key=lambda item: (item.provider, item.exact_model_id)):
            aliases = ", ".join(profile.aliases) or "-"
            lines.append(f"- {profile.provider}/{profile.exact_model_id} [{profile.model_family}] aliases: {aliases}")
        if self.repository.errors:
            lines.extend(["", "Profile load warnings:", *[f"- {error}" for error in self.repository.errors]])
        return "\n".join(lines)

    def _sources(self) -> str:
        try:
            registry = load_data(self.plugin_root / "sources" / "providers.yaml")
        except Exception as exc:
            return f"Sources registry unavailable: {exc}"
        lines = ["OFFICIAL SOURCES"]
        for provider, spec in (registry.get("providers", {}) or {}).items():
            lines.append(f"{provider}: {spec.get('priority', 'unknown')}")
            for name, source in (spec.get("sources", {}) or {}).items():
                lines.append(f"- {name}: {source.get('url', '')} ({source.get('type', 'unknown')})")
        lines.extend(["", "PROFILE SOURCE ANCHORS"])
        for profile in sorted(self.repository.profiles, key=lambda item: (item.provider, item.exact_model_id)):
            verified, missing, total, unverified = self._source_anchor_counts(profile)
            lines.append(
                f"- {profile.provider}/{profile.exact_model_id}: "
                f"verified={verified}, missing={missing}, total={total}, unverified/no-marker={unverified}"
            )
        return "\n".join(lines)

    @staticmethod
    def _source_anchor_counts(profile: Any) -> tuple[int, int, int, int]:
        if profile is None:
            return 0, 0, 0, 0
        recommendations = tuple(profile.prompt_recommendations) + tuple(profile.runtime_recommendations)
        marked_ids = {item.id for item in recommendations if item.source_marker}
        metadata = profile.metadata if isinstance(profile.metadata, Mapping) else {}
        verified = {
            str(value)
            for value in metadata.get("verified_rule_anchors", [])
            if str(value) in marked_ids
        }
        missing = {
            str(value)
            for value in metadata.get("missing_rule_anchors", [])
            if str(value) in marked_ids
        }
        return len(verified), len(missing), len(marked_ids), len(recommendations) - len(marked_ids)

    def _runtime_applied(self, rule_id: str) -> bool:
        if not self.last_result:
            return False
        item = next((item for item in self.last_result.runtime_recommendations if item.id == rule_id), None)
        return bool(item and runtime_rule_observed(item, self.last_runtime_observation))

    @staticmethod
    def _help() -> str:
        return "\n".join([
            "/model-guidance status   Show active profile, filters, sizes, and provenance",
            "/model-guidance show     Show active, Hermes-handled, runtime, unsupported, and source sections",
            "/model-guidance stats     Show deterministic runtime, cache, and token statistics",
            "/model-guidance test ID  Offline matching/compilation test; never contacts the model",
            "/model-guidance models    List local profiles and aliases",
            "/model-guidance sources   List official source registry",
            "/model-guidance reload    Reload local profiles and overlap metadata",
            "/model-guidance update    Manually check official sources and stage bounded profile metadata",
            "/model-guidance update --offline  Validate the update path without network access",
        ])
