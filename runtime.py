"""Hermes runtime integration for model-guidance."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import html
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


@dataclass
class ActivationState:
    """Bounded session-local state for one model activation epoch."""

    model_key: str = ""
    activation_generation: int = 0
    last_turn_id: str = ""
    last_base_fingerprint: str = ""
    last_task_scope: tuple[str, ...] = ()
    last_task_fingerprint: str = ""
    base_injected: bool = False
    task_injected: bool = False


class ModelGuidanceRuntime:
    def __init__(self, ctx: Any, *, plugin_root: Path | None = None, data_root: Path | None = None):
        self.ctx = ctx
        self.plugin_root = Path(plugin_root or Path(__file__).resolve().parent)
        self.data_root = Path(data_root or self._default_data_root())
        self.enabled = self._as_bool(self._config("enabled", True), True)
        mode = str(self._config("injection_mode", "activation")).strip().lower()
        self.injection_mode = mode if mode in {"activation", "every_turn"} else "activation"
        self.max_chars = self._bounded_int(self._config("max_chars", 3600), 200, 12000, 3600)
        self.max_base_guidance_chars = self._bounded_int(
            self._config("max_base_guidance_chars", 1800), 200, 6000, 1800
        )
        self.max_task_guidance_chars = self._bounded_int(
            self._config("max_task_guidance_chars", 700), 100, 3000, 700
        )
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
        self.last_session_id = ""
        self.last_activation_state: ActivationState | None = None
        self._activation_states: OrderedDict[str, ActivationState] = OrderedDict()
        self._activation_state_limit = 256
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
        mode = str(self._config("injection_mode", "activation")).strip().lower()
        if mode not in {"activation", "every_turn"}:
            mode = "activation"
        return (
            self._as_bool(self._config("enabled", True), True),
            mode,
            self._bounded_int(self._config("max_chars", 3600), 200, 12000, 3600),
            self._bounded_int(self._config("max_base_guidance_chars", 1800), 200, 6000, 1800),
            self._bounded_int(self._config("max_task_guidance_chars", 700), 100, 3000, 700),
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
            (
                self.enabled,
                self.injection_mode,
                self.max_chars,
                self.max_base_guidance_chars,
                self.max_task_guidance_chars,
                self.max_rules,
                self.max_family_rules,
                self.max_task_rules,
            ) = config_signature
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
            self.max_base_guidance_chars,
            self.max_task_guidance_chars,
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
            max_base_chars=self.max_base_guidance_chars,
            max_task_chars=self.max_task_guidance_chars,
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

    def _session_key(self, kwargs: Mapping[str, Any]) -> str:
        """Prefer Hermes' conversation identity; use task identity for isolated workers."""
        session_id = str(kwargs.get("session_id") or "").strip()
        if session_id:
            return f"session:{session_id}"
        task_id = str(kwargs.get("task_id") or "").strip()
        return f"task:{task_id}" if task_id else ""

    def _activation_state_for(self, key: str) -> ActivationState:
        state = self._activation_states.pop(key, None)
        if state is None:
            state = ActivationState()
        self._activation_states[key] = state
        while len(self._activation_states) > self._activation_state_limit:
            self._activation_states.popitem(last=False)
        return state

    def _guidance_fingerprint(self, result: CompilationResult, *, base: bool) -> str:
        profile = result.match.profile
        parts = [
            COMPILER_VERSION,
            str(self.repository.generation),
            result.match.provider,
            result.match.upstream_model,
            result.match.profile.exact_model_id if profile else "",
            profile.profile_revision if profile else "",
            result.match.match_kind,
            result.match.derivative_type,
        ]
        rules = result.base_prompt if base else result.task_prompt
        parts.extend(f"{item.id}\x00{item.text}" for item in rules)
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _xml_attr(value: str) -> str:
        return html.escape(str(value), quote=True)

    def _fit_context(self, pieces: list[str]) -> str:
        """Honor the legacy total limit while preserving complete XML wrappers."""
        context = "\n\n".join(piece for piece in pieces if piece)
        if len(context) <= self.max_chars:
            return context
        base = next((piece for piece in pieces if piece.startswith("<model_guidance ")), "")
        if base and len(base) <= self.max_chars:
            return base
        candidate = base or next((piece for piece in pieces if piece), "")
        if not candidate:
            return ""
        lines = candidate.splitlines()
        if len(lines) < 2:
            return candidate[: self.max_chars]
        opening, closing = lines[0], lines[-1]
        kept = [opening]
        size = len(opening) + len(closing)
        for line in lines[1:-1]:
            if size + len(line) + 1 > self.max_chars:
                break
            kept.append(line)
            size += len(line) + 1
        kept.append(closing)
        return "\n".join(kept)

    def _render_base_guidance(self, result: CompilationResult, generation: int) -> str:
        if not result.base_text:
            return ""
        profile = result.match.profile
        profile_id = f"{profile.provider}/{profile.exact_model_id}" if profile else "unknown"
        model = result.match.normalized_model_id
        revision = profile.profile_revision if profile else "unknown"
        return (
            f'<model_guidance model="{self._xml_attr(model)}" '
            f'profile="{self._xml_attr(profile_id)}" revision="{self._xml_attr(revision)}" '
            f'activation="{generation}">\n'
            "Supersedes earlier model_guidance blocks.\n"
            f"{result.base_text}\n"
            "</model_guidance>"
        )

    def _render_task_guidance(self, result: CompilationResult, *, reset_only: bool = False) -> str:
        scopes = ",".join(result.tasks)
        body = "Supersedes earlier model_task_guidance blocks.\n"
        if reset_only:
            body += "No additional task-specific guidance applies to this scope."
        else:
            body += result.task_text
        return (
            f'<model_task_guidance scope="current-task" tasks="{self._xml_attr(scopes)}">\n'
            f"{body}\n"
            "</model_task_guidance>"
        )

    def _activation_context(
        self,
        result: CompilationResult,
        *,
        session_key: str,
        turn_id: str,
    ) -> str:
        """Return only guidance that became active for this session and turn."""
        state = self._activation_state_for(session_key) if session_key else ActivationState()
        model_key = "|".join(
            (
                result.match.normalized_model_id,
                result.match.provider,
                result.match.profile.exact_model_id if result.match.profile else "",
                result.match.match_kind,
            )
        )
        base_fingerprint = self._guidance_fingerprint(result, base=True)
        task_fingerprint = self._guidance_fingerprint(result, base=False)
        if turn_id and state.last_turn_id == turn_id:
            self.last_activation_state = state
            return ""

        model_changed = bool(state.model_key and state.model_key != model_key)
        base_changed = bool(state.last_base_fingerprint and state.last_base_fingerprint != base_fingerprint)
        first_activation = not state.model_key
        base_needed = bool(result.base_text) and (first_activation or model_changed or base_changed)
        scope_changed = bool(state.last_task_scope and state.last_task_scope != result.tasks)
        task_needed = bool(result.task_text) and (
            first_activation or model_changed or base_changed
            or state.last_task_fingerprint != task_fingerprint
            or scope_changed
        )
        reset_needed = scope_changed and not task_needed and not base_needed

        if self.injection_mode == "every_turn":
            base_needed = bool(result.base_text)
            task_needed = bool(result.task_text)
            reset_needed = False

        pieces: list[str] = []
        if base_needed:
            state.activation_generation += 1
            pieces.append(self._render_base_guidance(result, state.activation_generation))
        if task_needed or reset_needed:
            pieces.append(self._render_task_guidance(result, reset_only=reset_needed))

        state.model_key = model_key
        state.last_turn_id = turn_id
        state.last_base_fingerprint = base_fingerprint
        state.last_task_scope = result.tasks
        state.last_task_fingerprint = task_fingerprint
        state.base_injected = base_needed
        state.task_injected = task_needed or reset_needed
        self.last_activation_state = state
        return self._fit_context(pieces)

    def on_session_start(self, **kwargs: Any) -> None:
        session_id = str(kwargs.get("session_id") or "").strip()
        if session_id:
            self._activation_states.pop(f"session:{session_id}", None)
        # pre_llm_call always uses its own model kwarg as authoritative.
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
        user_message = kwargs.get("user_message", "")
        if not isinstance(user_message, str) or not user_message.strip():
            self.last_error = "No real user prompt was supplied for this turn"
            return None
        try:
            self.last_model_id = model
            result = self._compile_cached(
                model,
                provider=str(kwargs.get("provider", "")),
                user_message=user_message,
                conversation_history=kwargs.get("conversation_history") or [],
                platform=str(kwargs.get("platform", "")),
            )
            self.last_result = result
            self.last_error = "; ".join(result.errors)
            self.last_session_id = str(kwargs.get("session_id") or "")
            context = self._activation_context(
                result,
                session_key=self._session_key(kwargs),
                turn_id=str(kwargs.get("turn_id") or ""),
            )
            return {"context": context} if context else None
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
            f"Upstream: {result.match.upstream_model or 'none'}",
            f"Derivative: {result.match.derivative_type or 'none'}",
            f"Modifiers: {', '.join(result.match.modifiers) or 'none'}",
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
            f"Base layer characters: {len(result.base_text)}",
            f"Task layer characters: {len(result.task_text)}",
            f"Estimated injected tokens: {estimate_tokens(result.injected_text)}",
            f"Cache: {'hit' if self.last_cache_hit else 'miss'} (hits={self.cache_hits}, misses={self.cache_misses})",
            f"Task scopes: {', '.join(result.tasks)}",
            f"Injection mode: {self.injection_mode}",
        ]
        if self.last_activation_state:
            state = self.last_activation_state
            lines.extend([
                f"Activation model: {state.model_key or 'none'}",
                f"Activation generation: {state.activation_generation}",
                f"Base guidance fingerprint: {state.last_base_fingerprint or 'none'}",
                f"Base guidance injected: {'yes' if state.base_injected else 'no'}",
                f"Last task scope: {', '.join(state.last_task_scope) or 'none'}",
                f"Last task guidance fingerprint: {state.last_task_fingerprint or 'none'}",
                f"Task guidance injected: {'yes' if state.task_injected else 'no'}",
            ])
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
            f"Activation state entries: {len(self._activation_states)}/{self._activation_state_limit}",
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
            f"Confidence: {result.match.confidence}",
            f"Upstream: {result.match.upstream_model or 'none'}",
            f"Derivative: {result.match.derivative_type or 'none'}",
            f"Modifiers: {', '.join(result.match.modifiers) or 'none'}",
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
