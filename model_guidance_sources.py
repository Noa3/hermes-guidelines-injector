"""Bounded official-source update support.

Remote documentation is treated as data only.  The updater accepts only HTTPS
hosts declared in the local source registry, caps response size, stores content
under the plugin-owned data directory, and updates only validated profile data.
No downloaded text is imported or executed and no arbitrary path is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import logging
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from .model_guidance_core import (
        COMPILER_VERSION,
        ModelProfile,
        ProfileRepository,
        ProfileError,
        compile_guidance,
        load_data,
        write_data,
    )
except ImportError:  # local command/test imports
    from model_guidance_core import (  # type: ignore
        COMPILER_VERSION,
        ModelProfile,
        ProfileRepository,
        ProfileError,
        compile_guidance,
        load_data,
        write_data,
    )

LOGGER = logging.getLogger(__name__)
ALLOWED_HOSTS = {"developers.openai.com", "openai.com", "www.openai.com"}
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_TIMEOUT = 8.0


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "nav"}:
            self.skip += 1
        if not self.skip and tag in {"p", "li", "h1", "h2", "h3", "h4", "pre", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "nav"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def source_to_text(payload: bytes, url: str) -> str:
    text = payload.decode("utf-8", errors="replace")
    if "<html" in text.lower() or "<main" in text.lower():
        parser = _HTMLText()
        parser.feed(text)
        text = "".join(parser.parts)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_source_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"source URL is not an allowed official OpenAI HTTPS URL: {url}")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError(f"source URL contains forbidden credentials/fragment: {url}")


def fetch_official(url: str, *, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[str, bytes]:
    validate_source_url(url)
    request = Request(url, headers={"User-Agent": "hermes-model-guidance/0.1 (+official-source-update)"})
    with urlopen(request, timeout=max(1.0, min(float(timeout), 30.0))) as response:  # noqa: S310 - host validated above
        final_url = str(response.geturl())
        validate_source_url(final_url)
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(f"source exceeds configured byte limit: {final_url}")
        payload = response.read(max(1, min(int(max_bytes), DEFAULT_MAX_BYTES + 1)))
    if len(payload) > max_bytes:
        raise ValueError(f"source exceeds configured byte limit: {url}")
    return final_url, payload


@dataclass(frozen=True)
class UpdateResult:
    changed_sources: tuple[str, ...]
    updated_profiles: tuple[str, ...]
    added: tuple[str, ...]
    modified: tuple[str, ...]
    removed: tuple[str, ...]
    failures: tuple[str, ...]
    offline: bool = False

    def render(self) -> str:
        lines = ["MODEL GUIDANCE UPDATE", ""]
        if self.offline:
            lines.append("Offline mode: no network request was performed.")
        lines.append(f"Changed sources: {len(self.changed_sources)}")
        lines.append(f"Updated profiles: {len(self.updated_profiles)}")
        lines.append(f"ADDED: {len(self.added)}")
        lines.append(f"MODIFIED: {len(self.modified)}")
        lines.append(f"REMOVED: {len(self.removed)}")
        if self.changed_sources:
            lines.append("\nSources:")
            lines.extend(f"- {value}" for value in self.changed_sources)
        if self.updated_profiles:
            lines.append("\nProfiles:")
            lines.extend(f"- {value}" for value in self.updated_profiles)
        if self.failures:
            lines.append("\nFailures (existing profiles remain active):")
            lines.extend(f"- {value}" for value in self.failures)
        return "\n".join(lines)


class OpenAIAdapter:
    """Deterministic extractor for known OpenAI recommendation anchors."""
    # These are compiler-owned templates.  Remote source text can confirm an
    # anchor, but can never replace these templates or introduce instructions.
    MARKERS = {
        "openai.gpt-6-astra.autonomy": "initiative and follow-through",
        "openai.gpt-6-astra.skills": "instruction following",
        "openai.gpt-5.5.outcome-first": "outcome-first prompts",
        "openai.gpt-5.6.lean-prompts": "Favor leaner prompts",
        "openai.gpt-5.coding-validation": "check its work",
    }

    def update_profile(self, profile: ModelProfile, source_text: str, source_hash: str, reviewed_date: str) -> ModelProfile:
        lower = source_text.lower()
        verified = list(profile.metadata.get("verified_rule_anchors", [])) if isinstance(profile.metadata, Mapping) else []
        for rule_id, marker in self.MARKERS.items():
            if any(item.id == rule_id for item in profile.prompt_recommendations) and marker.lower() in lower:
                if rule_id not in verified:
                    verified.append(rule_id)
        metadata = dict(profile.metadata)
        metadata["verified_rule_anchors"] = sorted(set(verified))
        raw = profile.to_mapping()
        raw["source_hash"] = source_hash
        raw["source_review_date"] = reviewed_date
        raw["profile_revision"] = f"{profile.profile_revision}+source"
        raw["metadata"] = metadata
        return ModelProfile.from_mapping(raw, origin=f"updated:{profile.exact_model_id}")


class SourceUpdater:
    def __init__(self, plugin_root: Path, data_root: Path, *, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = DEFAULT_MAX_BYTES):
        self.plugin_root = Path(plugin_root)
        self.data_root = Path(data_root)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.registry = load_data(self.plugin_root / "sources" / "providers.yaml")

    def update(self, repository: ProfileRepository, *, offline: bool = False) -> UpdateResult:
        if offline:
            return UpdateResult((), (), (), (), (), (), offline=True)
        now = datetime.now(timezone.utc).date().isoformat()
        changed: list[str] = []
        updated: list[str] = []
        added: list[str] = []
        modified: list[str] = []
        removed: list[str] = []
        failures: list[str] = []
        metadata_path = self.data_root / "source-metadata.json"
        try:
            old_meta = load_data(metadata_path) if metadata_path.exists() else {}
        except Exception as exc:
            old_meta = {}
            failures.append(f"corrupt source metadata ignored: {exc}")
        new_meta: dict[str, Any] = {"compiler_version": COMPILER_VERSION, "retrieved_at": datetime.now(timezone.utc).isoformat(), "sources": {}}
        providers = self.registry.get("providers", {})
        if not isinstance(providers, Mapping):
            return UpdateResult((), (), (), (), (), ("providers registry is malformed",))
        adapter = OpenAIAdapter()
        for provider_name, provider in providers.items():
            if not isinstance(provider, Mapping) or provider.get("priority") != "official":
                continue
            sources = provider.get("sources", {})
            if not isinstance(sources, Mapping):
                failures.append(f"{provider_name}: sources registry malformed")
                continue
            for source_name, spec in sources.items():
                if not isinstance(spec, Mapping):
                    failures.append(f"{provider_name}/{source_name}: source entry malformed")
                    continue
                fetch_url = str(spec.get("fetch_url", spec.get("url", "")))
                display_url = str(spec.get("url", fetch_url))
                key = f"{provider_name}/{source_name}"
                try:
                    final_url, payload = fetch_official(fetch_url, timeout=self.timeout, max_bytes=self.max_bytes)
                    digest = sha256(payload).hexdigest()
                    old_digest = str((old_meta.get("sources", {}) or {}).get(key, {}).get("sha256", ""))
                    new_meta["sources"][key] = {
                        "url": display_url,
                        "fetch_url": final_url,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "sha256": digest,
                        "bytes": len(payload),
                    }
                    if digest == old_digest:
                        continue
                    changed.append(key)
                    snapshot = self.data_root / "raw" / provider_name / f"{source_name}.txt"
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.write_text(source_to_text(payload, final_url), encoding="utf-8")
                    text = source_to_text(payload, final_url)
                    for profile in list(repository.profiles):
                        if profile.provider != provider_name or display_url not in profile.source_urls:
                            continue
                        before = compile_guidance(repository, repository.resolve(profile.exact_model_id), max_chars=3600).injected_text
                        updated_profile = adapter.update_profile(profile, text, digest, now)
                        overlay = self.data_root / "managed-profiles" / provider_name / f"{profile.exact_model_id}.yaml"
                        write_data(overlay, updated_profile.to_mapping())
                        updated.append(f"{provider_name}/{profile.exact_model_id}")
                        after_repo = ProfileRepository(
                            repository.root,
                            self.data_root / "managed-profiles",
                            self.data_root / "user-overrides",
                        )
                        after = compile_guidance(after_repo, after_repo.resolve(profile.exact_model_id), max_chars=3600).injected_text
                        if before != after:
                            modified.append(f"{provider_name}/{profile.exact_model_id}")
                        repository.reload()
                except Exception as exc:
                    failures.append(f"{key}: {exc}")
                    LOGGER.warning("official source update failed for %s", key, exc_info=True)
        write_data(metadata_path, new_meta)
        return UpdateResult(tuple(changed), tuple(dict.fromkeys(updated)), tuple(added), tuple(modified), tuple(removed), tuple(failures))
