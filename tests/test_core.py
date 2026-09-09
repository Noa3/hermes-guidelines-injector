import json
from pathlib import Path

import pytest

from model_guidance_core import (
    HermesOverlap,
    ModelProfile,
    ProfileError,
    ProfileRepository,
    compile_guidance,
    detect_tasks,
    normalize_id,
    provider_from_model,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def repository():
    return ProfileRepository(ROOT / "profiles")


def test_registry_loads_without_errors(repository):
    assert len(repository.profiles) >= 10
    assert repository.errors == []


def test_normalization_and_router_provider_resolution(repository):
    assert normalize_id("OpenRouter/OpenAI/GPT-5.6") == "gpt-5.6"
    assert normalize_id("openai:gpt-5.6") == "gpt-5.6"
    assert provider_from_model("openrouter/openai/gpt-5.6") == "openai"
    assert provider_from_model("openai:gpt-5.6") == "openai"
    result = repository.resolve("openrouter/openai/gpt-5.6")
    assert result.profile is not None
    assert result.profile.exact_model_id == "gpt-5.6"
    assert result.match_kind == "exact"
    assert repository.resolve("openai:gpt-5.6").profile.exact_model_id == "gpt-5.6"


@pytest.mark.parametrize(
    ("model_id", "provider", "family"),
    [
        ("openai/gpt-5.6", "openai", "gpt-5.6"),
        ("gpt-5.1", "openai", "gpt-5.1"),
        ("openai/codex", "openai", "gpt-5.3-codex"),
        ("anthropic/claude-opus-4-1", "anthropic", "claude"),
        ("google/gemini-2.5-pro", "google", "gemini"),
        ("qwen/qwen3", "qwen", "qwen"),
        ("deepseek/deepseek-chat", "deepseek", "deepseek"),
        ("kimi/kimi-k2", "kimi", "kimi"),
        ("glm/glm-4.5", "glm", "glm"),
        ("meta/llama-4", "meta", "llama"),
        ("mistral/mistral-large", "mistral", "mistral"),
        ("xai/grok-4", "xai", "grok"),
    ],
)
def test_simulated_provider_and_family_matching(repository, model_id, provider, family):
    result = repository.resolve(model_id)
    assert result.provider == provider
    assert result.model_family == family
    assert result.profile is not None
    assert result.profile.provider == provider


def test_exact_model_wins_over_family_pattern(repository):
    exact = repository.resolve("gpt-5.6")
    pattern = repository.resolve("gpt-5.6-preview-2026")
    assert exact.profile.exact_model_id == "gpt-5.6"
    assert pattern.profile.exact_model_id == "gpt-5.6"
    assert pattern.match_kind == "pattern"


def test_model_switch_does_not_use_stale_compilation(repository):
    first = compile_guidance(repository, repository.resolve("openai/gpt-5.6"), user_message="Write a concise summary")
    second = compile_guidance(repository, repository.resolve("anthropic/claude-opus-4-1"), user_message="Write a concise summary")
    assert "gpt-5.6" in first.injected_text
    assert "gpt-5.6" not in second.injected_text
    assert second.match.profile.exact_model_id == "claude-family"
    assert second.injected_text == ""


def test_overlap_filter_suppresses_hermes_rules(repository):
    result = compile_guidance(
        repository,
        repository.resolve("openai/gpt-6-astra"),
        user_message="Finish this coding task and verify it",
        hermes_overlap=HermesOverlap({"act-dont-ask": "core execution guidance"}),
    )
    assert any(item.id == "openai.gpt-6-astra.autonomy" for item in result.hermes_handled)
    assert all(item.id != "openai.gpt-6-astra.autonomy" for item in result.active_prompt)


def test_runtime_rules_are_not_prompt_injected(repository):
    result = compile_guidance(repository, repository.resolve("openai/gpt-5.2"), user_message="Implement code")
    assert result.runtime_recommendations
    assert all(item.category == "runtime" for item in result.runtime_recommendations)
    assert all(item.text not in result.injected_text for item in result.runtime_recommendations)


def test_task_filter_is_deterministic_and_conservative():
    assert detect_tasks("Please write a short email") == ("common", "writing")
    tasks = detect_tasks("Implement the plugin in src/main.py and run pytest")
    assert "coding" in tasks and "repository-work" in tasks and "tool-heavy" in tasks


def test_size_limit_keeps_output_bounded(repository):
    result = compile_guidance(
        repository,
        repository.resolve("openai/gpt-5.4"),
        user_message="Implement, test, and verify a repository change",
        max_chars=400,
    )
    assert result.characters <= 400
    assert result.truncated is True


def test_malformed_profile_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(json.dumps({"provider": "openai", "exact_model_id": "gpt-bad"}), encoding="utf-8")
    with pytest.raises(ProfileError):
        ModelProfile.from_mapping(json.loads(path.read_text(encoding="utf-8")), origin=str(path))


def test_unknown_model_fails_safe(repository):
    result = compile_guidance(repository, repository.resolve("provider/not-in-registry"), user_message="Implement code")
    assert result.match.profile is None
    assert result.injected_text == ""
    assert result.errors == () or isinstance(result.errors, tuple)


def test_user_override_wins_and_corrupt_override_is_ignored(tmp_path):
    overrides = tmp_path / "user-overrides" / "openai"
    overrides.mkdir(parents=True)
    override = json.loads((ROOT / "profiles/openai/gpt-5.6.yaml").read_text(encoding="utf-8"))
    override["prompt_recommendations"] = [{
        "id": "user.gpt-5.6.local-rule",
        "text": "Follow the local repository's documented acceptance checks.",
        "scopes": ["common"],
        "priority": 100,
    }]
    (overrides / "gpt-5.6.yaml").write_text(json.dumps(override), encoding="utf-8")
    (overrides / "broken.yaml").write_text("{not valid", encoding="utf-8")
    repo = ProfileRepository(ROOT / "profiles", user_override_root=tmp_path / "user-overrides")
    result = compile_guidance(repo, repo.resolve("gpt-5.6"), user_message="Give a concise answer")
    assert "local repository's documented acceptance checks" in result.injected_text
    assert any("corrupt user override" in error for error in repo.errors)
