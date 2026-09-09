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
        load_data,
        provider_from_model,
    )
    from .model_guidance_sources import SourceUpdater
except ImportError:  # local unit tests may import modules from the repository root.
    from model_guidance_core import (  # type: ignore
        COMPILER_VERSION,
        CompilationResult,
        HermesOverlap,
        ProfileRepository,
        compile_guidance,
        load_data,
        provider_from_model,
    )
    from model_guidance_sources import SourceUpdater  # type: ignore

LOGGER = logging.getLogger(__name__)


class ModelGuidanceRuntime:
    def __init__(self, ctx: Any, *, plugin_root: Path | None = None, data_root: Path | None = None):
        self.ctx = ctx
        self.plugin_root = Path(plugin_root or Path(__file__).resolve().parent)
        self.data_root = Path(data_root or self._default_data_root())
        self.enabled = self._as_bool(self._config("enabled", True), True)
        self.max_chars = self._bounded_int(self._config("max_chars", 3600), 400, 12000, 3600)
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

    def on_session_start(self, **kwargs: Any) -> None:
        # This is diagnostic-only.  pre_llm_call always uses its own model kwarg
        # as authoritative, so a model switch cannot use stale state.
        model = kwargs.get("model")
        if isinstance(model, str):
            self.last_model_id = model

    def pre_llm_call(self, **kwargs: Any) -> dict[str, str] | None:
        if not self.enabled:
            return None
        model = kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            self.last_error = "Hermes did not supply a model ID for this turn"
            return None
        try:
            self.last_model_id = model
            match = self.repository.resolve(model, str(kwargs.get("provider", "")))
            result = compile_guidance(
                self.repository,
                match,
                user_message=str(kwargs.get("user_message", "")),
                conversation_history=kwargs.get("conversation_history") or [],
                platform=str(kwargs.get("platform", "")),
                max_chars=self.max_chars,
                hermes_overlap=self.overlap,
                runtime_observation=self.last_runtime_observation,
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
            if action == "models":
                return self._models()
            if action == "sources":
                return self._sources()
            if action == "reload":
                self.repository.reload()
                self.overlap = self._load_overlap()
                return f"Reloaded {len(self.repository.profiles)} local model profiles."
            if action == "test":
                return self._test(args[1] if len(args) > 1 else "")
            if action == "update":
                offline = "--offline" in args[1:]
                updater = SourceUpdater(self.plugin_root, self.data_root, timeout=self.timeout, max_bytes=self.max_bytes)
                result = updater.update(self.repository, offline=offline)
                self.repository.reload()
                return result.render()
            return f"Unknown /model-guidance action: {action}\n\n{self._help()}"
        except Exception as exc:
            self.last_error = str(exc)
            LOGGER.exception("model-guidance command failed")
            return f"model-guidance failed safely: {exc}"

    def _current_or_unknown(self) -> CompilationResult | None:
        if self.last_result and self.last_result.match.raw_model_id == self.last_model_id:
            return self.last_result
        if self.last_model_id:
            result = compile_guidance(
                self.repository,
                self.repository.resolve(self.last_model_id),
                max_chars=self.max_chars,
                hermes_overlap=self.overlap,
                runtime_observation=self.last_runtime_observation,
            )
            self.last_result = result
            return result
        return self.last_result

    def _status(self) -> str:
        result = self._current_or_unknown()
        if result is None:
            return "MODEL GUIDANCE STATUS\n\nNo model has been supplied to the plugin in this session. Use /model-guidance test <model-id> for an offline lookup."
        injected = len(result.active_prompt)
        runtime_applied = sum(1 for item in result.runtime_recommendations if self._runtime_applied(item.id))
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
            f"Prompt recommendations: {injected + len(result.hermes_handled)}",
            f"Already handled by Hermes: {len(result.hermes_handled)}",
            f"Injected: {injected}",
            f"Runtime recommendations: {len(result.runtime_recommendations)}",
            f"Runtime recommendations observed/applied: {runtime_applied}",
            f"Informational: {len(result.informational_recommendations)}",
            f"Unsupported by current Hermes: {len(result.unsupported_recommendations)}",
            f"Injected characters: {result.characters}",
            f"Task scopes: {', '.join(result.tasks)}",
        ]
        if self.last_error:
            lines.append(f"Diagnostics: {self.last_error}")
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
        result = compile_guidance(
            self.repository,
            self.repository.resolve(model_id),
            user_message="Implement and verify a change in an existing repository.",
            platform="cli",
            max_chars=self.max_chars,
            hermes_overlap=self.overlap,
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
        return "\n".join(lines)

    @staticmethod
    def _runtime_applied(rule_id: str) -> bool:
        return False

    @staticmethod
    def _help() -> str:
        return "\n".join([
            "/model-guidance status   Show active profile, filters, sizes, and provenance",
            "/model-guidance show     Show active, Hermes-handled, runtime, unsupported, and source sections",
            "/model-guidance test ID  Offline matching/compilation test; never contacts the model",
            "/model-guidance models    List local profiles and aliases",
            "/model-guidance sources   List official source registry",
            "/model-guidance reload    Reload local profiles and overlap metadata",
            "/model-guidance update    Manually check official sources and stage bounded profile metadata",
            "/model-guidance update --offline  Validate the update path without network access",
        ])
