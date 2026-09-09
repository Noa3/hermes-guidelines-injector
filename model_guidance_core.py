"""Pure, offline model-guidance compilation logic.

The module intentionally has no Hermes imports and no network access.  This makes
model switching and profile selection testable with simulated model IDs and keeps
normal LLM turns independent from provider documentation availability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
COMPILER_VERSION = "1.0.0"
VALID_CLASSIFICATIONS = {
    "PROMPT_APPLICABLE",
    "RUNTIME_APPLICABLE",
    "INFORMATIONAL",
    "UNSUPPORTED_BY_CURRENT_HERMES",
}
KNOWN_PROVIDER_PREFIXES = {
    "openai", "azure", "openrouter", "vertex", "google", "google-ai-studio",
    "anthropic", "qwen", "alibaba", "deepseek", "kimi", "moonshot", "glm",
    "zhipu", "zai", "meta", "mistral", "xai", "grok", "nous", "codex",
    "together", "fireworks", "groq", "perplexity", "bedrock",
}
_PROVIDER_ALIASES = {
    "google": "google",
    "google-ai-studio": "google",
    "vertex": "google",
    "zhipu": "glm",
    "z-ai": "glm",
    "zai": "glm",
    "moonshot": "kimi",
    "moonshot-ai": "kimi",
    "x-ai": "xai",
    "grok": "xai",
    "alibaba": "qwen",
    "qwen": "qwen",
}


class ProfileError(ValueError):
    """Raised when a managed or user profile violates the bounded schema."""


@dataclass(frozen=True)
class Recommendation:
    id: str
    text: str
    category: str = "prompt"
    classification: str = "PROMPT_APPLICABLE"
    scopes: tuple[str, ...] = ("common",)
    priority: int = 50
    handled_by_hermes: bool = False
    plugin_injection: bool = True
    source_section: str = ""
    source_url: str = ""
    source_type: str = "official"
    confidence: str = "high"
    hermes_overlap_id: str = ""
    source_marker: str = ""
    applied_when: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, default_url: str = "") -> "Recommendation":
        rid = str(raw.get("id", "")).strip()
        text = str(raw.get("text", "")).strip()
        if not rid or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,127}", rid):
            raise ProfileError(f"invalid recommendation id: {rid!r}")
        if not text or len(text) > 2400:
            raise ProfileError(f"invalid recommendation text for {rid}")
        classification = str(raw.get("classification", "PROMPT_APPLICABLE")).upper()
        if classification not in VALID_CLASSIFICATIONS:
            raise ProfileError(f"invalid classification for {rid}: {classification}")
        category = str(raw.get("category", "prompt")).lower()
        if category not in {"prompt", "runtime"}:
            raise ProfileError(f"invalid category for {rid}: {category}")
        scopes = tuple(str(s).strip() for s in raw.get("scopes", ["common"]) if str(s).strip())
        if not scopes:
            scopes = ("common",)
        return cls(
            id=rid,
            text=text,
            category=category,
            classification=classification,
            scopes=scopes,
            priority=max(0, min(1000, int(raw.get("priority", 50)))),
            handled_by_hermes=bool(raw.get("handled_by_hermes", False)),
            plugin_injection=bool(raw.get("plugin_injection", True)),
            source_section=str(raw.get("source_section", "")).strip(),
            source_url=str(raw.get("source_url", default_url)).strip(),
            source_type=str(raw.get("source_type", "official")).strip(),
            confidence=str(raw.get("confidence", "high")).strip().lower(),
            hermes_overlap_id=str(raw.get("hermes_overlap_id", "")).strip(),
            source_marker=str(raw.get("source_marker", "")).strip(),
            applied_when=str(raw.get("applied_when", "")).strip(),
        )


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    model_family: str
    exact_model_id: str
    aliases: tuple[str, ...] = ()
    match_prefixes: tuple[str, ...] = ()
    inherits: tuple[str, ...] = ()
    remove_rules: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    source_review_date: str = ""
    source_hash: str = ""
    source_type: str = "official"
    profile_revision: str = "1"
    confidence: str = "high"
    prompt_recommendations: tuple[Recommendation, ...] = ()
    runtime_recommendations: tuple[Recommendation, ...] = ()
    negative_overrides: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, origin: str = "") -> "ModelProfile":
        provider = str(raw.get("provider", "")).strip().lower()
        family = str(raw.get("model_family", raw.get("family", ""))).strip().lower()
        exact = str(raw.get("exact_model_id", raw.get("model", ""))).strip().lower()
        if not provider or not family or not exact:
            raise ProfileError(f"profile {origin or '<unknown>'} lacks provider/family/exact_model_id")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._:-]{1,127}", exact):
            raise ProfileError(f"invalid model id in {origin}: {exact!r}")
        aliases = tuple(normalize_id(str(v)) for v in raw.get("aliases", []) if str(v).strip())
        prefixes = tuple(normalize_id(str(v)) for v in raw.get("match_prefixes", []) if str(v).strip())
        source_urls = tuple(str(v).strip() for v in raw.get("source_urls", []) if str(v).strip())
        for url in source_urls:
            if not (url.startswith("https://") and len(url) <= 500):
                raise ProfileError(f"unsafe source URL in {origin}: {url!r}")
        default_url = source_urls[0] if source_urls else ""
        prompt = tuple(
            Recommendation.from_mapping(item, default_url=default_url)
            for item in raw.get("prompt_recommendations", [])
            if isinstance(item, Mapping)
        )
        runtime = tuple(
            Recommendation.from_mapping({**item, "category": "runtime"}, default_url=default_url)
            for item in raw.get("runtime_recommendations", [])
            if isinstance(item, Mapping)
        )
        return cls(
            provider=provider,
            model_family=family,
            exact_model_id=exact,
            aliases=aliases,
            match_prefixes=prefixes,
            inherits=tuple(str(v).strip() for v in raw.get("inherits", []) if str(v).strip()),
            remove_rules=tuple(str(v).strip() for v in raw.get("remove_rules", []) if str(v).strip()),
            source_urls=source_urls,
            source_review_date=str(raw.get("source_review_date", "")).strip(),
            source_hash=str(raw.get("source_hash", "")).strip(),
            source_type=str(raw.get("source_type", "official")).strip(),
            profile_revision=str(raw.get("profile_revision", "1")).strip(),
            confidence=str(raw.get("confidence", "high")).strip().lower(),
            prompt_recommendations=prompt,
            runtime_recommendations=runtime,
            negative_overrides=tuple(str(v).strip() for v in raw.get("negative_overrides", []) if str(v).strip()),
            metadata=dict(raw.get("metadata", {})) if isinstance(raw.get("metadata", {}), Mapping) else {},
        )

    def to_mapping(self) -> dict[str, Any]:
        def rec(item: Recommendation) -> dict[str, Any]:
            return {
                "id": item.id,
                "text": item.text,
                "category": item.category,
                "classification": item.classification,
                "scopes": list(item.scopes),
                "priority": item.priority,
                "handled_by_hermes": item.handled_by_hermes,
                "plugin_injection": item.plugin_injection,
                "source_section": item.source_section,
                "source_url": item.source_url,
                "source_type": item.source_type,
                "confidence": item.confidence,
                "hermes_overlap_id": item.hermes_overlap_id,
                "source_marker": item.source_marker,
                "applied_when": item.applied_when,
            }
        return {
            "provider": self.provider,
            "model_family": self.model_family,
            "exact_model_id": self.exact_model_id,
            "aliases": list(self.aliases),
            "match_prefixes": list(self.match_prefixes),
            "inherits": list(self.inherits),
            "remove_rules": list(self.remove_rules),
            "source_urls": list(self.source_urls),
            "source_review_date": self.source_review_date,
            "source_hash": self.source_hash,
            "source_type": self.source_type,
            "profile_revision": self.profile_revision,
            "confidence": self.confidence,
            "prompt_recommendations": [rec(x) for x in self.prompt_recommendations],
            "runtime_recommendations": [rec(x) for x in self.runtime_recommendations],
            "negative_overrides": list(self.negative_overrides),
            "metadata": dict(self.metadata),
        }


def normalize_id(model_id: str) -> str:
    """Normalize an ID without guessing unknown provider namespaces."""
    value = re.sub(r"\s+", "", str(model_id or "").strip().lower())
    value = value.replace("\\", "/")
    if "/" not in value and ":" in value:
        prefix, remainder = value.split(":", 1)
        if prefix in KNOWN_PROVIDER_PREFIXES:
            value = f"{prefix}/{remainder}"
    parts = [part for part in value.split("/") if part]
    while len(parts) > 1 and parts[0] in KNOWN_PROVIDER_PREFIXES:
        parts.pop(0)
    return "/".join(parts)


def provider_from_model(model_id: str, profile_provider: str = "") -> str:
    raw = str(model_id or "").strip().lower().replace("\\", "/")
    if "/" not in raw and ":" in raw:
        prefix, remainder = raw.split(":", 1)
        if prefix in KNOWN_PROVIDER_PREFIXES:
            raw = f"{prefix}/{remainder}"
    parts = [part for part in raw.split("/") if part]
    first = parts[0] if parts else ""
    if first in KNOWN_PROVIDER_PREFIXES:
        # Routers commonly use openrouter/<provider>/<model>; the nested
        # provider is authoritative for profile selection.
        if first == "openrouter" and len(parts) > 1 and parts[1] in KNOWN_PROVIDER_PREFIXES:
            return _PROVIDER_ALIASES.get(parts[1], parts[1])
        return _PROVIDER_ALIASES.get(first, first)
    if profile_provider:
        return profile_provider
    model = normalize_id(raw)
    if model.startswith(("gpt-", "o1", "o3", "o4", "codex")):
        return "openai"
    if model.startswith(("claude-", "claude/")):
        return "anthropic"
    if model.startswith(("gemini-", "gemma-")):
        return "google"
    if model.startswith(("qwen", "qwq")):
        return "qwen"
    if model.startswith("deepseek"):
        return "deepseek"
    if model.startswith(("kimi", "moonshot")):
        return "kimi"
    if model.startswith(("glm", "chatglm")):
        return "glm"
    if model.startswith(("llama", "meta-llama")):
        return "meta"
    if model.startswith(("mistral", "mixtral")):
        return "mistral"
    if model.startswith("grok"):
        return "xai"
    return "unknown"


@dataclass(frozen=True)
class MatchResult:
    raw_model_id: str
    normalized_model_id: str
    provider: str
    model_family: str
    profile: ModelProfile | None
    match_kind: str
    confidence: str
    reason: str


class ProfileRepository:
    """Loads bundled profiles and safe profile overlays.

    A broken overlay never hides a valid bundled profile.  Overlay files are
    written only by the bounded update/user-override paths.
    """
    def __init__(
        self,
        root: Path,
        overlay_root: Path | None = None,
        user_override_root: Path | None = None,
    ):
        self.root = Path(root)
        self.overlay_root = Path(overlay_root) if overlay_root else None
        self.user_override_root = Path(user_override_root) if user_override_root else None
        self.errors: list[str] = []
        self.profiles: list[ModelProfile] = []
        self.reload()

    def reload(self) -> None:
        self.errors.clear()
        bundled: dict[tuple[str, str], ModelProfile] = {}
        for path in sorted(self.root.glob("*/*.yaml")):
            try:
                profile = ModelProfile.from_mapping(load_data(path), origin=str(path))
                bundled[(profile.provider, profile.exact_model_id)] = profile
            except Exception as exc:
                self.errors.append(f"ignored bundled profile {path.name}: {exc}")
                LOGGER.warning("%s", self.errors[-1])
        if self.overlay_root and self.overlay_root.exists():
            for path in sorted(self.overlay_root.glob("*/*.yaml")):
                try:
                    profile = ModelProfile.from_mapping(load_data(path), origin=str(path))
                    key = (profile.provider, profile.exact_model_id)
                    bundled[key] = profile
                except Exception as exc:
                    self.errors.append(f"ignored corrupt profile overlay {path.name}: {exc}")
                    LOGGER.warning("%s", self.errors[-1])
        if self.user_override_root and self.user_override_root.exists():
            for path in sorted(self.user_override_root.glob("*/*.yaml")):
                try:
                    profile = ModelProfile.from_mapping(load_data(path), origin=str(path))
                    key = (profile.provider, profile.exact_model_id)
                    bundled[key] = profile
                except Exception as exc:
                    self.errors.append(f"ignored corrupt user override {path.name}: {exc}")
                    LOGGER.warning("%s", self.errors[-1])
        self.profiles = list(bundled.values())

    def get(self, provider: str, exact_model_id: str) -> ModelProfile | None:
        return next((p for p in self.profiles if p.provider == provider and p.exact_model_id == exact_model_id), None)

    def resolve(self, raw_model_id: str, provider_hint: str = "") -> MatchResult:
        raw = str(raw_model_id or "")
        normalized = normalize_id(raw)
        provider_hint = _PROVIDER_ALIASES.get(str(provider_hint or "").strip().lower(), str(provider_hint or "").strip().lower())
        scored: list[tuple[int, ModelProfile, str, str]] = []
        for profile in self.profiles:
            provider = provider_hint or provider_from_model(raw, profile.provider)
            if provider not in {profile.provider, "unknown"}:
                continue
            candidates = {profile.exact_model_id, *profile.aliases}
            if normalized in candidates:
                kind = "exact" if normalized == profile.exact_model_id else "alias"
                scored.append((10000 + len(normalized), profile, kind, "explicit exact/alias match"))
                continue
            for prefix in profile.match_prefixes:
                if normalized.startswith(prefix):
                    scored.append((7000 + len(prefix), profile, "pattern", f"explicit prefix {prefix}"))
        if scored:
            _, profile, kind, reason = max(scored, key=lambda item: item[0])
            return MatchResult(raw, normalized, profile.provider, profile.model_family, profile, kind, "high", reason)
        detected = provider_hint or provider_from_model(raw)
        family = infer_family(normalized, detected)
        return MatchResult(raw, normalized, detected, family, None, "fallback", "low" if detected != "unknown" else "unknown", "no safe local profile matched")


def infer_family(normalized: str, provider: str) -> str:
    if provider == "openai":
        for prefix in ("gpt-6", "gpt-5.6", "gpt-5.5", "gpt-5.4", "gpt-5.3", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "codex"):
            if normalized == prefix or normalized.startswith(prefix + "-"):
                return prefix
    if provider == "anthropic" and normalized.startswith("claude-"):
        return normalized.split("-", 2)[0] + "-" + normalized.split("-", 2)[1]
    for family in ("gemini", "gemma", "qwen", "deepseek", "kimi", "glm", "llama", "mistral", "grok"):
        if normalized.startswith(family):
            return family
    return "unknown"


class HermesOverlap:
    """Known current Hermes prompt sections, reviewed against Hermes source."""
    def __init__(self, mapping: Mapping[str, Any] | None = None):
        self.rules = dict(mapping or {})

    def handles(self, item: Recommendation) -> bool:
        return bool(item.handled_by_hermes or (item.hermes_overlap_id and item.hermes_overlap_id in self.rules))

    def reason(self, item: Recommendation) -> str:
        return self.rules.get(item.hermes_overlap_id, "profile metadata marks this as Hermes-handled")


def detect_tasks(user_message: str = "", conversation_history: Sequence[Any] | None = None, platform: str = "") -> tuple[str, ...]:
    """Conservative deterministic task classification; never calls an LLM."""
    text = str(user_message or "").lower()
    history = conversation_history or []
    tasks: list[str] = ["common"]
    coding = re.search(r"\b(code|coding|bug|debug|fix|implement|refactor|repo|repository|git|build|compile|test|pytest|script|api|plugin|program|source code)\b|[./][\w.-]+\.(py|js|ts|cs|rs|gd|json|yaml|yml|toml)", text)
    writing = re.search(r"\b(write|writing|draft|rewrite|email|letter|article|story|copy|documentation|readme|translate|summarize)\b", text)
    research = re.search(r"\b(research|investigate|compare|sources?|citation|literature|find out|current facts?)\b", text)
    computer = re.search(r"\b(browser|website|click|screenshot|screen|gui|desktop|computer use|visual)\b", text)
    long_running = re.search(r"\b(autonomous|long[- ]running|end[- ]to[- ]end|multi[- ]step|full workflow|finish|until complete|agent)\b", text)
    subagent = re.search(r"\b(subagent|sub-agent|delegate|parallel|multi-agent|worktree)\b", text)
    tool_messages = any(isinstance(item, Mapping) and item.get("role") in {"tool", "function"} for item in history)
    if coding:
        tasks.extend(["coding", "repository-work"])
    if research:
        tasks.append("research")
    if computer:
        tasks.append("computer-use")
    if writing and not coding:
        tasks.append("writing")
    if long_running:
        tasks.append("long-running-agent")
    if subagent:
        tasks.append("subagent")
    if tool_messages or coding or computer:
        tasks.append("tool-heavy")
    if platform in {"kanban", "worker"} or "kanban" in text:
        tasks.append("kanban")
    return tuple(dict.fromkeys(tasks))


@dataclass(frozen=True)
class CompilationResult:
    match: MatchResult
    tasks: tuple[str, ...]
    injected_text: str
    active_prompt: tuple[Recommendation, ...]
    hermes_handled: tuple[Recommendation, ...]
    runtime_recommendations: tuple[Recommendation, ...]
    unsupported_recommendations: tuple[Recommendation, ...]
    informational_recommendations: tuple[Recommendation, ...]
    inherited_profiles: tuple[str, ...]
    characters: int
    truncated: bool
    errors: tuple[str, ...] = ()


def _merge_recommendations(profile: ModelProfile, repository: ProfileRepository) -> tuple[Recommendation, ...]:
    collected: list[Recommendation] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def add_profile(item: ModelProfile) -> None:
        key = f"{item.provider}/{item.exact_model_id}"
        if key in visiting:
            raise ProfileError(f"profile inheritance cycle at {key}")
        if key in seen:
            return
        visiting.add(key)
        for parent in item.inherits:
            parent_provider, _, parent_id = parent.partition("/")
            parent_profile = repository.get(parent_provider or item.provider, parent_id or parent)
            if parent_profile:
                add_profile(parent_profile)
        collected.extend(item.prompt_recommendations)
        collected.extend(item.runtime_recommendations)
        visiting.remove(key)
        seen.add(key)

    add_profile(profile)
    removed = set(profile.remove_rules) | set(profile.negative_overrides)
    return tuple(item for item in collected if item.id not in removed)


def compile_guidance(
    repository: ProfileRepository,
    match: MatchResult,
    *,
    user_message: str = "",
    conversation_history: Sequence[Any] | None = None,
    platform: str = "",
    max_chars: int = 3600,
    hermes_overlap: HermesOverlap | None = None,
    runtime_observation: Mapping[str, Any] | None = None,
) -> CompilationResult:
    overlap = hermes_overlap or HermesOverlap()
    tasks = detect_tasks(user_message, conversation_history, platform)
    errors = list(repository.errors)
    if not match.profile:
        return CompilationResult(match, tasks, "", (), (), (), (), (), (), 0, False, tuple(errors))
    try:
        all_rules = _merge_recommendations(match.profile, repository)
    except Exception as exc:
        errors.append(str(exc))
        all_rules = tuple(match.profile.prompt_recommendations + match.profile.runtime_recommendations)
    active: list[Recommendation] = []
    handled: list[Recommendation] = []
    runtime: list[Recommendation] = []
    unsupported: list[Recommendation] = []
    informational: list[Recommendation] = []
    for item in all_rules:
        if not any(scope in tasks for scope in item.scopes):
            continue
        if item.category == "runtime" or item.classification == "RUNTIME_APPLICABLE":
            if item.classification == "RUNTIME_APPLICABLE":
                runtime.append(item)
            elif item.classification == "UNSUPPORTED_BY_CURRENT_HERMES":
                unsupported.append(item)
            elif item.classification == "INFORMATIONAL":
                informational.append(item)
            continue
        if item.classification == "UNSUPPORTED_BY_CURRENT_HERMES":
            unsupported.append(item)
            continue
        if item.classification == "INFORMATIONAL":
            informational.append(item)
            continue
        if overlap.handles(item):
            handled.append(item)
            continue
        if item.plugin_injection:
            active.append(item)
    active.sort(key=lambda item: (-item.priority, item.id))
    header = (
        "<model_guidance>\n"
        f"Model-specific guidance for {match.normalized_model_id} ({match.provider}/{match.model_family}).\n"
        "Apply only the relevant recommendations below; Hermes already supplies its own execution, tool-use, and verification rules.\n"
    )
    lines = [header]
    chars = len(header) + len("</model_guidance>")
    truncated = False
    for item in active:
        line = f"- {item.text}\n"
        if chars + len(line) + len("</model_guidance>") > max(400, int(max_chars)):
            truncated = True
            continue
        lines.append(line)
        chars += len(line)
    lines.append("</model_guidance>")
    text = "".join(lines)
    observation = runtime_observation or {}
    if observation:
        runtime = tuple(item for item in runtime if _runtime_observed(item, observation)) + tuple(
            item for item in runtime if not _runtime_observed(item, observation)
        )
    inherited = tuple(match.profile.inherits)
    return CompilationResult(
        match, tasks, text if active else "", tuple(active), tuple(handled), tuple(runtime), tuple(unsupported), tuple(informational), inherited, len(text if active else ""), truncated, tuple(errors)
    )


def _runtime_observed(item: Recommendation, observation: Mapping[str, Any]) -> bool:
    key = item.id.lower()
    request = observation.get("request", {})
    if not isinstance(request, Mapping):
        request = {}
    if "reasoning" in key:
        return bool(request.get("reasoning") or request.get("reasoning_effort"))
    if "parallel" in key:
        return request.get("parallel_tool_calls") is True
    if "responses" in key or "api" in key:
        return str(observation.get("api_mode", "")).lower() in {"responses", "codex_responses", "responses_api"}
    return False


def load_data(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without executing tags or constructors."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ProfileError(f"{path} is not JSON-compatible YAML and PyYAML is unavailable") from exc
        value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ProfileError(f"expected mapping in {path}")
    return value


def write_data(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)
