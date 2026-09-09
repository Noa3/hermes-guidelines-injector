"""Native Hermes plugin entry point for model-specific official guidance."""
from __future__ import annotations

try:
    from .runtime import ModelGuidanceRuntime
except ImportError:  # pytest imports the repository root as a package-less module
    from runtime import ModelGuidanceRuntime  # type: ignore


def register(ctx):
    runtime = ModelGuidanceRuntime(ctx)
    ctx.register_hook("pre_llm_call", runtime.pre_llm_call)
    ctx.register_hook("pre_api_request", runtime.pre_api_request)
    ctx.register_hook("on_session_start", runtime.on_session_start)
    ctx.register_command(
        "model-guidance",
        runtime.command,
        description="Inspect and update official model guidance",
        args_hint="[status|show|stats|test MODEL|models|sources|reload|update]",
    )
