"""Install model-guidance into every discovered local Hermes profile.

The script copies only this plugin's bounded files and edits only the
plugins.enabled list. It never touches secrets or user override data.
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import shutil
import sys

PLUGIN_FILES = (
    "plugin.yaml",
    "__init__.py",
    "runtime.py",
    "model_guidance_core.py",
    "model_guidance_sources.py",
    "profiles",
    "sources",
)
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".git", "tests", "install_hermes.py"}


def candidate_homes() -> list[Path]:
    values: list[Path] = []
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        values.append(Path(env_home))
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        root = Path(local_app_data) / "hermes"
        values.append(root)
        if root.is_dir():
            values.extend(child for child in root.joinpath("profiles").glob("*") if child.is_dir())
    user_home = Path.home() / ".hermes"
    if user_home.exists():
        values.append(user_home)
    deduped: list[Path] = []
    seen: set[str] = set()
    for value in values:
        resolved = str(value.expanduser().resolve())
        if resolved not in seen and (value.exists() or value == Path(env_home)):
            seen.add(resolved)
            deduped.append(Path(resolved))
    return deduped


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in PLUGIN_FILES:
        src = source / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = destination / name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*EXCLUDED_NAMES),
            )
        else:
            shutil.copy2(src, dst)


def _ensure_enabled(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    lines = config_path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    before = list(lines)
    # Remove only the plugin's own list item.  Reinsert it in the enabled list
    # below so a previous interrupted installer cannot leave malformed YAML.
    lines = [line for line in lines if line.strip() != "- model-guidance"]
    plugins_index = next((i for i, line in enumerate(lines) if line.rstrip("\r\n") == "plugins:"), None)
    if plugins_index is None:
        suffix = "" if not lines or lines[-1].endswith(("\n", "\r")) else "\n"
        lines.extend([suffix, "plugins:\n", "  enabled:\n", "    - model-guidance\n", "  disabled: []\n"])
    else:
        end = len(lines)
        for i in range(plugins_index + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped and not lines[i].startswith((" ", "\t")):
                end = i
                break
        enabled_index = next(
            (i for i in range(plugins_index + 1, end) if lines[i].strip() == "enabled:"),
            None,
        )
        if enabled_index is None:
            lines.insert(plugins_index + 1, "  enabled:\n")
            lines.insert(plugins_index + 2, "    - model-guidance\n")
        else:
            insert_at = enabled_index + 1
            enabled_indent = len(lines[enabled_index]) - len(lines[enabled_index].lstrip())
            while insert_at < end:
                candidate = lines[insert_at]
                if candidate.strip() and len(candidate) - len(candidate.lstrip()) <= enabled_indent:
                    break
                insert_at += 1
            lines.insert(insert_at, "    - model-guidance\n")
    changed = lines != before
    config_path.write_text("".join(lines), encoding="utf-8")
    return changed


def install(home: Path, source: Path) -> tuple[Path, bool]:
    destination = home / "plugins" / "model-guidance"
    if destination.exists():
        backup = home / "plugin-data" / "model-guidance" / "install-backups" / (
            "model-guidance-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(destination, backup, dirs_exist_ok=True)
    _copy_tree(source, destination)
    enabled = _ensure_enabled(home / "config.yaml")
    return destination, enabled


def main() -> int:
    source = Path(__file__).resolve().parent
    homes = candidate_homes()
    if not homes:
        print("No Hermes homes found. Set HERMES_HOME and retry.", file=sys.stderr)
        return 2
    for home in homes:
        destination, enabled = install(home, source)
        print(f"installed: {destination}")
        print(f"config_changed: {enabled} ({home / 'config.yaml'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
