"""Fork overlay layer (config.d/*.yaml) — loader contract tests.

The overlay directory predates its loader by a month: config.d files were
maintained (backups, a pre-commit guard) with no runtime ever reading them,
so two sessions' config fixes silently never took effect. These tests pin
the loader so a future refactor of ``_load_config_impl`` cannot orphan the
overlay layer again without a red test.

Subprocess-driven (not in-process monkeypatching) so HERMES_HOME resolution,
module-level caches, and the mtime-keyed config cache behave like production.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo!r})
    from hermes_cli import config as C
    cfg = C.load_config()
    result = {{
        "threshold_tokens": (cfg.get("compression") or {{}}).get("threshold_tokens"),
        "overlay_probe": cfg.get("overlay_probe"),
        "skin": (cfg.get("display") or {{}}).get("skin"),
    }}
    print("OVERLAY_RESULT=" + json.dumps(result))
    """
)


def _run_load(home: Path, env_extra: dict[str, str] | None = None) -> dict:
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER.format(repo=str(REPO_ROOT))],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(home),
        timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    for line in proc.stdout.splitlines():
        if line.startswith("OVERLAY_RESULT="):
            return json.loads(line[len("OVERLAY_RESULT="):])
    raise AssertionError(f"no result in stdout:\n{proc.stdout}\n{proc.stderr}")


def _make_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / "config.d").mkdir(parents=True)
    (home / "config.yaml").write_text(
        "compression:\n  threshold_tokens: null\noverlay_probe: from-user\n",
        encoding="utf-8",
    )
    return home


def test_overlay_overrides_user_leaf(tmp_path):
    home = _make_home(tmp_path)
    (home / "config.d" / "02-context-memory.yaml").write_text(
        "compression:\n  threshold_tokens: 48000\n", encoding="utf-8"
    )
    assert _run_load(home)["threshold_tokens"] == 48000


def test_overlay_ordering_later_file_wins(tmp_path):
    home = _make_home(tmp_path)
    (home / "config.d" / "00-a.yaml").write_text("overlay_probe: from-a\n", encoding="utf-8")
    (home / "config.d" / "01-b.yaml").write_text("overlay_probe: from-b\n", encoding="utf-8")
    assert _run_load(home)["overlay_probe"] == "from-b"


def test_bak_and_non_yaml_ignored(tmp_path):
    home = _make_home(tmp_path)
    (home / "config.d" / "02-context-memory.yaml").write_text(
        "compression:\n  threshold_tokens: 48000\n", encoding="utf-8"
    )
    (home / "config.d" / "02-context-memory.yaml.bak-20260820").write_text(
        "compression:\n  threshold_tokens: 999999\noverlay_probe: from-bak\n",
        encoding="utf-8",
    )
    (home / "config.d" / "notes.txt").write_text("overlay_probe: from-txt\n", encoding="utf-8")
    result = _run_load(home)
    assert result["threshold_tokens"] == 48000
    assert result["overlay_probe"] == "from-user"


def test_broken_overlay_skipped_user_config_survives(tmp_path):
    home = _make_home(tmp_path)
    (home / "config.d" / "02-broken.yaml").write_text(
        "compression: [unclosed\n", encoding="utf-8"
    )
    assert _run_load(home)["overlay_probe"] == "from-user"


def test_managed_scope_still_wins_over_overlay(tmp_path):
    home = _make_home(tmp_path)
    (home / "config.d" / "00-pin.yaml").write_text(
        "display:\n  skin: overlay-skin\n", encoding="utf-8"
    )
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "display:\n  skin: managed-skin\n", encoding="utf-8"
    )
    result = _run_load(home, {"HERMES_MANAGED_DIR": str(managed)})
    assert result["skin"] == "managed-skin"


def test_cache_invalidates_on_overlay_edit(tmp_path):
    home = _make_home(tmp_path)
    overlay = home / "config.d" / "02-x.yaml"
    overlay.write_text("compression:\n  threshold_tokens: 48000\n", encoding="utf-8")

    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from hermes_cli import config as C
        first = C.load_config()["compression"]["threshold_tokens"]
        from pathlib import Path
        import os
        p = Path(os.environ["HERMES_HOME"]) / "config.d" / "02-x.yaml"
        p.write_text("compression:\\n  threshold_tokens: 120000\\n", encoding="utf-8")
        second = C.load_config()["compression"]["threshold_tokens"]
        print("OVERLAY_RESULT=" + json.dumps({{"first": first, "second": second}}))
        """
    )
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=str(home), timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    line = [l for l in proc.stdout.splitlines() if l.startswith("OVERLAY_RESULT=")][0]
    result = json.loads(line[len("OVERLAY_RESULT="):])
    assert result == {"first": 48000, "second": 120000}
