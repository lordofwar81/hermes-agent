# GatewayRunner Refactoring Snapshot

**Date:** 2026-06-08 (Updated)
**Status:** Active decomposition from ~20,000 lines toward <500 lines target

## Current State

| Metric | Value |
|--------|-------|
| `gateway/run.py` | **7,030 lines** |
| `gateway/command_handlers.py` | **4,397 lines** |
| `gateway/agent_execution.py` | **2,662 lines** |
| `gateway/lifecycle.py` | **1,009 lines** |
| Total methods in GatewayRunner | ~200 |
| **Progress** | **31% reduction** (10,220 → 7,030) |

## Completed Extractions

### Command Handlers (in `command_handlers.py` - 43 functions)
All command handlers extracted:
- `handle_model_command`, `handle_compress_command`, `handle_reset_command`
- `handle_usage_command`, `handle_reasoning_command`, `handle_update_command`
- `handle_codex_runtime_command`, `handle_retry_command`, `handle_reload_skills_command`
- `handle_personality_command`, `handle_goal_command`, `handle_subgoal_command`
- `handle_voice_command`, `handle_footer_command`, `handle_resume_command`
- `handle_reload_mcp_command`, `execute_mcp_reload`, `handle_undo_command`
- `handle_set_home_command`, `handle_help_command`, `handle_commands_command`
- `handle_profile_command`, `handle_whoami_command`, `handle_kanban_command`
- `handle_status_command`, `handle_agents_command`, `handle_stop_command`
- `handle_platform_command`, `handle_restart_command`, `handle_rollback_command`
- `handle_fast_command`, `handle_yolo_command`, `handle_verbose_command`
- `handle_voice_channel_join`, `handle_voice_channel_leave`, `handle_title_command`
- `handle_bundles_command`, `handle_insights_command`, `handle_approve_command`
- `handle_deny_command`, `handle_debug_command`, `handle_branch_command`
- `handle_background_command` (with `_run_background_task` helper)

### Telegram Topic Handlers (in `command_handlers.py`)
- `_get_telegram_topic_capabilities`
- `_ensure_telegram_system_topic`
- `_send_telegram_topic_setup_image`
- `_should_send_telegram_capability_hint`
- `_telegram_topic_help_text`
- `_disable_telegram_topic_mode_for_chat`
- `_telegram_topic_root_status_message`
- `_restore_telegram_topic_session`
- `handle_topic_command`

### Voice Handlers (in `command_handlers.py`)
- `_handle_voice_channel_input`
- `_should_send_voice_reply`
- `_send_voice_reply`

### Error Handlers (in `command_handlers.py`)
- `_handle_adapter_fatal_error`

### Agent Execution (in `agent_execution.py`)
- `run_agent` - Main agent execution loop (extracted from `_run_agent`)
- `run_agent_via_proxy` - Proxy mode execution

### Lifecycle (in `lifecycle.py`)
- `start_gateway_runner` - Gateway startup logic (extracted from `start`)
- `stop_gateway_runner` - Gateway shutdown logic (extracted from `stop`)

### Agent Cache (in `gateway/agent_cache.py`)
- `enforce_agent_cache_cap`
- `sweep_idle_cached_agents`
- `release_evicted_agent_soft`

## Remaining Large Methods (>50 lines)

| Method | Lines | Priority | Notes |
|--------|-------|----------|-------|
| `__init__` | 230 | LOW | Constructor (keep in class) |
| `_process_handoff` | 168 | HIGH | Agent handoff |
| `_run_process_watcher` | 167 | MED | Process monitoring |
| `_launch_detached_restart_command` | 99 | LOW | Restart |
| `_send_update_notification` | 96 | LOW | Update notifications |
| `_is_user_authorized` | 87 | MED | Authorization |
| `_launch_systemd_restart_shortcut` | 77 | LOW | Restart |
| `_send_voice_reply` | 75 | DONE | Already extracted |
| `_send_restart_notification` | 72 | LOW | Restart notifications |
| `_build_process_event_source` | 71 | MED | Event source builder |
| `_schedule_resume_pending_sessions` | 68 | MED | Resume scheduling |
| `_is_stale_restart_redelivery` | 50 | LOW | Restart dedupe |

## Dead Code Cleanup (2026-06-08)

Cleaned up dead code left after extractions:
- `_run_agent`: Deleted 2,310 lines of unreachable code
- `stop`: Deleted 424 lines of unreachable code
- `start`: Deleted 48 lines of unreachable code
- `_run_agent_via_proxy`: Deleted 299 lines of unreachable code
- `_handle_bundles_command`: Deleted 20 lines
- `_handle_message_with_agent`: Deleted 18 lines
- `_prepare_inbound_message_text`: Deleted 17 lines
- `_handle_deny_command`: Deleted 13 lines
- Multiple command handlers: Deleted 4-5 lines each (docstrings)

**Total dead code removed: ~3,190 lines**

## Next Steps

1. **Extract `_process_handoff` (168 lines)** → `lifecycle.py` or `handoff.py`
2. **Extract `_run_process_watcher` (167 lines)** → `watchers.py` or `process_watcher.py`
3. **Extract notification methods** (`_send_update_notification`, `_send_restart_notification`) → `notifications.py`
4. **Extract authorization** (`_is_user_authorized`) → `authorization.py`
5. **Target: <500 lines in run.py**

## Git Strategy

```bash
# After each extraction batch:
git add gateway/run.py gateway/*.py
git commit -m "refactor: extract [module] from GatewayRunner

- Extract [methods] to gateway/[module].py
- Convert run.py methods to thin wrappers
- Remove dead code from extracted methods
- Reduce run.py to X lines"
```

## Commands to Resume

```bash
# Check current state
wc -l gateway/run.py gateway/command_handlers.py gateway/agent_execution.py gateway/lifecycle.py

# List largest methods remaining
python3 << 'PYEOF'
import re
with open('gateway/run.py') as f:
    lines = f.readlines()
methods = []
for i, line in enumerate(lines):
    m = re.match(r'(\s+)(async )?def (\w+)\(', line)
    if m:
        indent = len(m.group(1))
        for j in range(i+1, len(lines)):
            if lines[j].strip() and not lines[j].startswith('#'):
                if len(lines[j]) - len(lines[j].lstrip()) <= indent:
                    methods.append((m.group(3), j-i, i+1))
                    break
print(f"{'Method':<40} {'Lines':>8} {'Line#':>8}")
print("=" * 58)
for name, n, line in sorted(methods, key=lambda x: -x[1])[:20]:
    print(f"{name:<40} {n:>8} {line:>8}")
PYEOF
```

---

**Target:** Decompose GatewayRunner from ~20,000 → <500 lines
**Current Progress:** ~65% complete (7,030 / ~20,000 original)
**Estimated Work:** 8-12 more extraction batches
