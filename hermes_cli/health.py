"""
Health command for hermes CLI.

Provides a system health overview: gateway status, database sizes,
cache stats, VRAM usage, and thermal snapshot.  Outputs a letter
grade (A-F) summarizing overall health.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from hermes_cli.colors import Colors, color
from hermes_cli.config import get_hermes_home

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n  {color(title, Colors.BOLD, Colors.CYAN)}")


def _row(label: str, value: str, ok: bool = True) -> None:
    mark = color("OK", Colors.GREEN) if ok else color("!!", Colors.RED)
    print(f"    {label:<22s} {value}  {mark}")


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _read_temp(path: str) -> float | None:
    """Read a hwmon temp file (millidegrees C) and return degrees C."""
    try:
        with open(path) as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def _gateway_status() -> dict:
    """Return gateway running state, PID(s), and uptime."""
    from hermes_cli.gateway import get_gateway_runtime_snapshot
    snap = get_gateway_runtime_snapshot()
    pids = list(snap.gateway_pids)
    running = snap.running
    uptime_str = ""
    if running and pids:
        try:
            r = subprocess.run(
                ["ps", "-o", "etimes=", "-p", str(pids[0])],
                capture_output=True, text=True, timeout=3,
            )
            secs = int(r.stdout.strip())
            hrs, rem = divmod(secs, 3600)
            mins, secs = divmod(rem, 60)
            uptime_str = f"{hrs}h {mins}m {secs}s" if hrs else f"{mins}m {secs}s"
        except Exception:
            uptime_str = "(unknown)"
    return {
        "running": running,
        "pids": pids,
        "uptime": uptime_str,
        "manager": snap.manager,
    }


def _db_sizes(home: Path) -> dict:
    """Return sizes of state.db, memory_store.db, and sessions/ directory."""
    sizes = {}
    for name in ("state.db", "memory_store.db"):
        p = home / name
        sizes[name] = p.stat().st_size if p.exists() else 0
    # sessions directory
    sess_dir = home / "sessions"
    if sess_dir.exists():
        total = sum(f.stat().st_size for f in sess_dir.rglob("*") if f.is_file())
        sizes["sessions/"] = total
    else:
        sizes["sessions/"] = 0
    return sizes


def _cache_stats(home: Path) -> dict:
    """Return agent count and model cache size from state.db if present."""
    stats = {"agent_count": 0, "model_cache_bytes": 0}
    state_db = home / "state.db"
    if not state_db.exists():
        return stats
    try:
        import sqlite3
        conn = sqlite3.connect(str(state_db))
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            if "agents" in tables:
                stats["agent_count"] = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
            if "model_cache" in tables:
                stats["model_cache_bytes"] = conn.execute(
                    "SELECT SUM(LENGTH(cache_blob)) FROM model_cache"
                ).fetchone()[0] or 0
        finally:
            conn.close()
    except Exception:
        pass
    return stats


def _vram_usage() -> dict:
    """Query local model health endpoints and rocm-smi for VRAM info."""
    result = {"models": [], "vram": None}
    # Try llama-server health endpoints
    for port, label in ((8100, "35B-MoE"), (8104, "122B-MoE")):
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = json.loads(resp.read().decode())
                result["models"].append({"port": port, "name": label, "status": "ok", "detail": body})
        except Exception as exc:
            result["models"].append({"port": port, "name": label, "status": "down", "detail": str(exc)[:80]})
    # rocm-smi VRAM
    try:
        r = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        import re
        total_m = re.search(r"VRAM Total Memory \(B\): (\d+)", r.stdout)
        used_m = re.search(r"VRAM Total Used Memory \(B\): (\d+)", r.stdout)
        if total_m and used_m:
            total = int(total_m.group(1))
            used = int(used_m.group(1))
            result["vram"] = {
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": total - used,
                "percentage": (used / total * 100) if total else 0,
            }
    except Exception:
        pass
    return result


def _thermal_snapshot() -> dict:
    """Read CPU and GPU temperatures from hwmon."""
    # Probe known sensors; skip missing ones gracefully.
    sensors = {}
    hwmon = Path("/sys/class/hwmon")
    if not hwmon.exists():
        return sensors
    for entry in sorted(hwmon.iterdir()):
        name_file = entry / "name"
        try:
            sensor_name = name_file.read_text().strip()
        except Exception:
            continue
        temp_file = entry / "temp1_input"
        temp = _read_temp(str(temp_file))
        if temp is not None:
            sensors[sensor_name] = temp
    return sensors


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _compute_grade(gw: dict, vram: dict, thermal: dict) -> str:
    """Return a single-letter health grade A-F."""
    penalties = 0  # 0 = A, each +1 drops a grade tier

    # Gateway down is a big penalty
    if not gw["running"]:
        penalties += 2

    # VRAM pressure
    if vram.get("vram"):
        pct = vram["vram"]["percentage"]
        if pct > 95:
            penalties += 3
        elif pct > 85:
            penalties += 2
        elif pct > 75:
            penalties += 1

    # Model availability
    models_up = [m for m in vram.get("models", []) if m["status"] == "ok"]
    if len(models_up) == 0 and vram.get("vram"):
        penalties += 1  # VRAM present but no models

    # Thermal
    for name, temp in thermal.items():
        if "amdgpu" in name or "k10temp" in name:
            if temp > 95:
                penalties += 2
            elif temp > 85:
                penalties += 1

    grades = "ABCDF"
    idx = min(penalties, len(grades) - 1)
    return grades[idx]


def _grade_color(grade: str) -> str:
    return {
        "A": Colors.GREEN,
        "B": Colors.GREEN,
        "C": Colors.YELLOW,
        "D": Colors.YELLOW,
        "F": Colors.RED,
    }.get(grade, "")


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

def cmd_health(args) -> None:
    """Show system health overview with grade."""
    home = get_hermes_home()

    print()
    print(f"  {color('Hermes System Health', Colors.BOLD)}")
    print(f"  {'=' * 40}")

    # 1. Gateway status
    _section("Gateway")
    gw = _gateway_status()
    if gw["running"]:
        pids = ", ".join(str(p) for p in gw["pids"][:3])
        _row("Status", f"running (PID {pids})", ok=True)
        if gw["uptime"]:
            _row("Uptime", gw["uptime"], ok=True)
        _row("Manager", gw["manager"], ok=True)
    else:
        _row("Status", "stopped", ok=False)

    # 2. Database sizes
    _section("Databases")
    db = _db_sizes(home)
    for name, size in db.items():
        _row(name, _human_bytes(size), ok=size < 500 * 1024 * 1024)

    # 3. Cache stats
    _section("Cache")
    cache = _cache_stats(home)
    _row("Agents", str(cache["agent_count"]), ok=True)
    _row("Model cache", _human_bytes(cache["model_cache_bytes"]), ok=True)

    # 4. VRAM usage
    _section("VRAM / Local Models")
    vram = _vram_usage()
    for m in vram.get("models", []):
        status_ok = m["status"] == "ok"
        _row(f"{m['name']} :{m['port']}", m["status"], ok=status_ok)
    if vram.get("vram"):
        v = vram["vram"]
        _row("VRAM used",
             f"{_human_bytes(v['used_bytes'])} / {_human_bytes(v['total_bytes'])} ({v['percentage']:.1f}%)",
             ok=v["percentage"] < 85)
    else:
        _row("VRAM", "(not available)", ok=True)

    # 5. Thermal snapshot
    _section("Thermal")
    thermal = _thermal_snapshot()
    if thermal:
        for name, temp in thermal.items():
            hot = temp > 80
            _row(name, f"{temp:.1f} C", ok=not hot)
    else:
        _row("sensors", "(none found)", ok=True)

    # 6. Grade
    grade = _compute_grade(gw, vram, thermal)
    print()
    print(f"  {color('Overall Grade:', Colors.BOLD)} {color(grade, _grade_color(grade))}")
    print()
