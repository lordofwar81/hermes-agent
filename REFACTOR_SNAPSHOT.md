# GatewayRunner Refactoring Snapshot

**Date:** 2025-01-19
**Status:** Active decomposition from ~20,000 lines toward <500 lines target

## Current State

| Metric | Value |
|--------|-------|
| `gateway/run.py` | **10,220 lines** |
| `gateway/command_handlers.py` | **4,397 lines** |
| Total methods in GatewayRunner | 247 |
| Thin wrappers (≤5 lines) | 85 |
| Methods needing extraction (>50 lines) | **19** |

## Decomposition Pattern

All extracted functions follow the thin wrapper pattern:

```python
# In gateway/run.py (thin wrapper)
async def _handle_title_command(self, event: MessageEvent) -> str:
    return await command_handlers.handle_title_command(
        runner=self,
        event=event,
    )

# In gateway/command_handlers.py (implementation)
async def handle_title_command(runner: GatewayRunner, event: MessageEvent) -> str:
    # Actual implementation here
    ...
```

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

### Agent Cache (in `gateway/agent_cache.py`)
- `enforce_agent_cache_cap`
- `sweep_idle_cached_agents`
- `release_evicted_agent_soft`

## Remaining Large Methods (>50 lines)

These are the top priority for extraction:

| Method | Lines | Priority | Notes |
|--------|-------|----------|-------|
| `run_sync` (nested in _run_agent) | 859 | **HIGH** | Main agent execution loop body |
| `_stop_impl` | 412 | **HIGH** | Stop logic |
| `send_progress_messages` | 338 | MED | Progress streaming |
| `__init__` | 229 | LOW | Constructor (keep in class) |
| `_process_handoff` | 167 | MED | Agent handoff |
| `_run_process_watcher` | 166 | MED | Process monitoring |
| `progress_callback` | 119 | MED | Progress events |
| `_launch_detached_restart_command` | 98 | LOW | Restart |
| `_send_update_notification` | 95 | LOW | Update notifications |
| `_is_user_authorized` | 86 | MED | Authorization |
| `_approval_notify_sync` | 76 | MED | Approval notifications |
| `_send_voice_reply` | 74 | **DONE** | Already extracted |
| `_send_restart_notification` | 71 | LOW | Restart notifications |
| `_notify_long_running` | 71 | LOW | Long-running notify |
| `_build_process_event_source` | 70 | MED | Event source builder |
| `_schedule_resume_pending_sessions` | 67 | MED | Resume scheduling |
| `_clarify_callback_sync` | 61 | MED | Clarification callback |
| `start` | 53 | LOW | Main entry |
| `_is_stale_restart_redelivery` | 49 | LOW | Restart dedupe |

## Remaining Medium Methods (30-50 lines)

Good candidates for next extraction batch:
- `_clear_session_boundary_security_state` (46 lines)
- `_sync_voice_mode_state_to_adapter` (43 lines)
- `_resolve_turn_agent_config` (43 lines)
- `_defer_goal_status_notice_after_delivery` (42 lines)
- `_cleanup_agent_resources` (40 lines)
- `_is_duplicate_voice_transcript` (40 lines)
- `_sibling_thread_run_keys` (38 lines)
- `_inject_watch_notification` (38 lines)
- `_pause_failed_platform` (37 lines)
- `_extract_cache_busting_config` (37 lines)

## Extraction Modules (Recommended Structure)

```
gateway/
├── run.py                    # Thin wrappers + core class (<500 lines target)
├── command_handlers.py       # Command handlers (4,397 lines, done)
├── agent_cache.py            # Cache management (done)
├── agent_execution.py        # NEW: _run_agent, run_sync, progress, callbacks
├── lifecycle.py              # NEW: _stop_impl, start, _process_handoff
├── authorization.py          # NEW: _is_user_authorized, _check_slash_access
├── notifications.py          # NEW: Update/restart notifications, process watcher
├── session_state.py          # NEW: Session management, resume, env
└── callbacks.py              # NEW: Various sync callbacks
```

## Next Steps

1. **Extract `run_sync` (859 lines)** → `agent_execution.py`
   - This is the core agent loop body
   - Extract as `execute_agent_run_sync(runner, ...)`

2. **Extract `_stop_impl` (412 lines)** → `lifecycle.py`
   - Stop/shutdown logic

3. **Extract medium-sized batches** (30-50 line methods)
   - Group related methods together
   - Commit after each module

4. **Target: <500 lines in run.py**

## Git Strategy

```bash
# After each extraction batch:
git add gateway/run.py gateway/*.py
git commit -m "refactor: extract [module] from GatewayRunner

- Extract [methods] to gateway/[module].py
- Convert run.py methods to thin wrappers
- Reduce run.py to X lines"
```

## Commands to Resume

```bash
# Check current state
wc -l gateway/run.py gateway/command_handlers.py

# Find next extraction candidates
python3 -c "
import re
with open('gateway/run.py') as f:
    lines = f.readlines()
methods = []
for i, line in enumerate(lines):
    m = re.match(r'(\s+)(async )?def (\w+)\(', line)
    if m:
        indent = len(m.group(1))
        for j in range(i+1, len(lines)):
            if lines[j].strip() and not lines[j].strip().startswith('#'):
                if len(lines[j]) - len(lines[j].lstrip()) <= indent:
                    methods.append((m.group(3), j-i))
                    break
print('\n'.join(f'{n:>4} {name}' for name, n in sorted(methods, key=lambda x: -x[1])[:20]))
"
```

---

**Target:** Decompose GatewayRunner from ~20,000 → <500 lines
**Current Progress:** ~50% complete (10,220 / ~20,000 original)
**Estimated Work:** 10-15 more extraction batches

## Quick Resume Commands

```bash
# New session - read snapshot
cat REFACTOR_SNAPSHOT.md

# Check current state
wc -l gateway/run.py gateway/command_handlers.py

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
for name, n, line in sorted(methods, key=lambda x: -x[1])[:25]:
    print(f"{name:<40} {n:>8} {line:>8}")
PYEOF
```
