#!/usr/bin/env python3
"""
Tool Executor Module

Extracts tool execution logic from AIAgent to provide a clean separation of concerns.
Handles sequential and concurrent tool execution, budget management, and result
processing while maintaining all existing behaviors.

Key features:
- Sequential and concurrent tool execution paths
- Budget pressure warnings and iteration tracking
- Tool parallelization strategy determination
- Result sanitization and persistence
- Checkpoint management integration
- Comprehensive callback support
"""

import concurrent.futures
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional, Callable, List, Dict

from model_tools import handle_function_call
from tools.terminal_tool import get_active_env
from tools.tool_result_storage import maybe_persist_tool_result, enforce_turn_budget
from agent.display import (
    KawaiiSpinner,
    build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
    _detect_tool_failure,
    get_tool_emoji as _get_tool_emoji,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Module-level constants for tool parallelization
# =============================================================================

# Tools that must never run concurrently (interactive / user-facing).
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# Read-only tools with no shared mutable session state.
_PARALLEL_SAFE_TOOLS = frozenset(
    {
        "ha_get_state",
        "ha_list_entities",
        "ha_list_services",
        "read_file",
        "search_files",
        "session_search",
        "skill_view",
        "skills_list",
        "vision_analyze",
        "web_extract",
        "web_search",
    }
)

# File tools can run concurrently when they target independent paths.
_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

# Maximum number of concurrent worker threads for parallel tool execution.
_MAX_TOOL_WORKERS = 8

# =============================================================================
# Tool parallelization helper functions
# =============================================================================


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """Return the normalized file target for path-scoped tools."""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # Avoid resolve(); the file may not exist yet.
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        # Empty paths shouldn't reach here (guarded upstream), but be safe.
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]


def _is_destructive_command_patterns() -> object:
    """Return compiled regex patterns for destructive command detection."""
    import re

    destructive = re.compile(
        r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
        re.VERBOSE,
    )
    redirect_overwrite = re.compile(r"[^>]>[^>]|^>[^>]")
    return destructive, redirect_overwrite


_DESTRUCTIVE_PATTERNS, _REDIRECT_OVERWRITE = _is_destructive_command_patterns()


def _is_destructive_command(cmd: str) -> bool:
    """Heuristic: does this terminal command look like it modifies/deletes files?"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """Return True when a tool-call batch is safe to run concurrently."""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(
                _paths_overlap(scoped_path, existing) for existing in reserved_paths
            ):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True


# =============================================================================
# Tool Executor Class
# =============================================================================


class ToolExecutor:
    """Handles tool execution for AIAgent with both sequential and concurrent paths.

    This class encapsulates all tool execution logic including:
    - Sequential tool execution with display and callbacks
    - Concurrent tool execution using thread pools
    - Budget pressure warnings and iteration tracking
    - Result sanitization and persistence
    - Checkpoint management integration
    - Comprehensive callback support for progress tracking

    The executor is designed to be dependency-injected, accepting callbacks
    and configuration via its constructor rather than accessing agent state directly.
    """

    def __init__(
        self,
        tool_handler: Callable,
        print_fn: Callable,
        iteration_budget,
        max_iterations: int,
        budget_pressure_enabled: bool,
        budget_caution_threshold: float,
        budget_warning_threshold: float,
        tool_progress_callback: Optional[Callable] = None,
        tool_start_callback: Optional[Callable] = None,
        tool_complete_callback: Optional[Callable] = None,
        quiet_mode: bool = False,
        verbose_logging: bool = False,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        checkpoint_mgr=None,
        subdirectory_hints=None,
        interrupt_checker: Optional[Callable[[], bool]] = None,
        nudge_reset_callback: Optional[Callable[[str], None]] = None,
        activity_tracker: Optional[Callable[[str], None]] = None,
        valid_tool_names: Optional[set] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the ToolExecutor with dependencies and configuration.

        Args:
            tool_handler: Callable that executes a single tool call.
                Signature: (function_name: str, function_args: dict, task_id: str,
                           tool_call_id: Optional[str]) -> str
            print_fn: Callable for status output (default: builtin print)
            iteration_budget: IterationBudget instance for tracking iterations
            max_iterations: Maximum iterations allowed
            budget_pressure_enabled: Whether to enable budget pressure warnings
            budget_caution_threshold: Threshold (0-1) for caution warnings
            budget_warning_threshold: Threshold (0-1) for urgent warnings
            tool_progress_callback: Optional callback for tool progress updates
            tool_start_callback: Optional callback when tool starts
            tool_complete_callback: Optional callback when tool completes
            quiet_mode: Suppress progress output
            verbose_logging: Enable verbose logging
            log_prefix_chars: Characters to show in previews
            log_prefix: Prefix for log messages
            checkpoint_mgr: CheckpointManager instance
            subdirectory_hints: SubdirectoryHintTracker instance
            interrupt_checker: Callable that returns True if interrupt requested
            nudge_reset_callback: Callable to reset nudge counters
            activity_tracker: Callable to track activity
            valid_tool_names: Set of valid tool names
            session_id: Current session ID
        """
        self._tool_handler = tool_handler
        self._print_fn = print_fn
        self._iteration_budget = iteration_budget
        self.max_iterations = max_iterations
        self._budget_pressure_enabled = budget_pressure_enabled
        self._budget_caution_threshold = budget_caution_threshold
        self._budget_warning_threshold = budget_warning_threshold
        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.quiet_mode = quiet_mode
        self.verbose_logging = verbose_logging
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = log_prefix
        self._checkpoint_mgr = checkpoint_mgr
        self._subdirectory_hints = subdirectory_hints
        self._interrupt_checker = interrupt_checker or (lambda: False)
        self._nudge_reset_callback = nudge_reset_callback
        self._activity_tracker = activity_tracker
        self._valid_tool_names = valid_tool_names
        self._session_id = session_id

    def _safe_print(self, *args, **kwargs):
        """Print that silently handles broken pipes / closed stdout."""
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """Verbose print — can be overridden for streaming suppression."""
        # For simplicity, just route to _safe_print
        # The agent can override this method for more sophisticated behavior
        self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """Return True when quiet-mode spinner output has a safe sink."""
        if self._print_fn is not None:
            return True
        import sys

        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _should_emit_quiet_tool_messages(self) -> bool:
        """Return True when quiet-mode tool summaries should print directly."""
        return self.quiet_mode and not self.tool_progress_callback

    def _touch_activity(self, desc: str) -> None:
        """Track activity if callback provided."""
        if self._activity_tracker:
            try:
                self._activity_tracker(desc)
            except Exception:
                pass

    def _reset_nudge_counter(self, tool_name: str) -> None:
        """Reset nudge counters for specific tools."""
        if self._nudge_reset_callback:
            try:
                self._nudge_reset_callback(tool_name)
            except Exception:
                pass

    def _check_checkpoint(self, function_name: str, function_args: dict) -> None:
        """Create checkpoint before file-mutating or destructive operations."""
        if not self._checkpoint_mgr or not self._checkpoint_mgr.enabled:
            return

        try:
            if function_name in ("write_file", "patch"):
                file_path = function_args.get("path", "")
                if file_path:
                    work_dir = self._checkpoint_mgr.get_working_dir_for_path(
                        file_path
                    )
                    self._checkpoint_mgr.ensure_checkpoint(
                        work_dir, f"before {function_name}"
                    )
            elif function_name == "terminal":
                cmd = function_args.get("command", "")
                if _is_destructive_command(cmd):
                    cwd = function_args.get("workdir") or os.getenv(
                        "TERMINAL_CWD", os.getcwd()
                    )
                    self._checkpoint_mgr.ensure_checkpoint(
                        str(cwd or ""), f"before terminal: {cmd[:60]}"
                    )
        except Exception:
            pass  # never block tool execution

    def execute_tool_calls(
        self,
        assistant_message,
        messages: list,
        effective_task_id: str,
        api_call_count: int = 0,
    ) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        This is the main entry point for tool execution. It dispatches to either
        concurrent or sequential execution based on the tool batch analysis.

        Args:
            assistant_message: Message containing tool_calls
            messages: Message list to append results to
            effective_task_id: Task ID for session isolation
            api_call_count: Current API call count for budget tracking
        """
        tool_calls = assistant_message.tool_calls

        if not _should_parallelize_tool_batch(tool_calls):
            return self._execute_sequential(
                assistant_message, messages, effective_task_id, api_call_count
            )

        return self._execute_concurrent(
            assistant_message, messages, effective_task_id, api_call_count
        )

    def _execute_concurrent(
        self,
        assistant_message,
        messages: list,
        effective_task_id: str,
        api_call_count: int = 0,
    ) -> None:
        """Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages so the API sees them in the expected sequence.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── Pre-flight: interrupt check ──────────────────────────────────
        if self._interrupt_checker():
            self._vprint(
                f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)",
                force=True,
            )
            for tc in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                        "tool_call_id": tc.id,
                    }
                )
            return

        # ── Parse args + pre-execution bookkeeping ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Reset nudge counters
            self._reset_nudge_counter(function_name)

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Checkpoint for file-mutating tools
            self._check_checkpoint(function_name, function_args)

            parsed_calls.append((tool_call, function_name, function_args))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
        if not self.quiet_mode:
            self._safe_print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    self._safe_print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    self._safe_print(f"     Args: {args_str}")
                else:
                    args_preview = (
                        args_str[: self.log_prefix_chars] + "..."
                        if len(args_str) > self.log_prefix_chars
                        else args_str
                    )
                    self._safe_print(
                        f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}"
                    )

        for tc, name, args in parsed_calls:
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args in parsed_calls:
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # Each slot holds (function_name, function_args, function_result, duration, error_flag)
        results: list[Any] = [None] * num_tools

        def _run_tool(index, tool_call, function_name, function_args):
            """Worker function executed in a thread."""
            start = time.time()
            try:
                result = self._tool_handler(
                    function_name, function_args, effective_task_id, tool_call.id
                )
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error(
                    "_tool_handler raised for %s: %s",
                    function_name,
                    tool_error,
                    exc_info=True,
                )
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info(
                    "tool %s failed (%.2fs): %s", function_name, duration, result[:200]
                )
            else:
                logger.info(
                    "tool %s completed (%.2fs, %d chars)",
                    function_name,
                    duration,
                    len(result),
                )
            results[index] = (function_name, function_args, result, duration, is_error)

        # Start spinner for CLI mode
        spinner = None
        if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
            spinner = KawaiiSpinner(
                f"{face} ⚡ running {num_tools} tools concurrently",
                spinner_type="dots",
                print_fn=self._print_fn,
            )
            spinner.start()

        try:
            max_workers = min(num_tools, _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = []
                for i, (tc, name, args) in enumerate(parsed_calls):
                    f = executor.submit(_run_tool, i, tc, name, args)
                    futures.append(f)

                # Wait for all to complete (exceptions are captured inside _run_tool)
                concurrent.futures.wait(futures)
        finally:
            if spinner:
                # Build a summary message for the spinner stop
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(
                    f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total"
                )

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args) in enumerate(parsed_calls):
            r = results[i]
            if r is None:
                # Shouldn't happen, but safety fallback
                function_result = (
                    f"Error executing tool '{name}': thread did not return a result"
                )
                tool_duration = 0.0
                is_error = True
            else:
                (
                    function_name,
                    function_args,
                    function_result,
                    tool_duration,
                    is_error,
                ) = r

            if is_error:
                result_preview = (
                    function_result[:200]
                    if len(function_result) > 200
                    else function_result
                )
                logger.warning(
                    "Tool %s returned error (%.2fs): %s",
                    function_name,
                    tool_duration,
                    result_preview,
                )

            if self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed",
                        function_name,
                        None,
                        None,
                        duration=tool_duration,
                        is_error=is_error,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if self.verbose_logging:
                logging.debug(
                    f"Tool {function_name} completed in {tool_duration:.2f}s"
                )
                logging.debug(
                    f"Tool result ({len(function_result)} chars): {function_result}"
                )

            # Print cute message per tool
            if self._should_emit_quiet_tool_messages():
                cute_msg = _get_cute_tool_message_impl(
                    name, args, tool_duration, result=function_result
                )
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                if self.verbose_logging:
                    self._safe_print(
                        f"  ✅ Tool {i + 1} completed in {tool_duration:.2f}s"
                    )
                    self._safe_print(f"     Result: {function_result}")
                else:
                    response_preview = (
                        function_result[: self.log_prefix_chars] + "..."
                        if len(function_result) > self.log_prefix_chars
                        else function_result
                    )
                    self._safe_print(
                        f"  ✅ Tool {i + 1} completed in {tool_duration:.2f}s - {response_preview}"
                    )

            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=name,
                tool_use_id=tc.id,
                env=get_active_env(effective_task_id),
            )

            if self._subdirectory_hints:
                subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
                if subdir_hints:
                    function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools_executed = len(parsed_calls)
        if num_tools_executed > 0:
            turn_tool_msgs = messages[-num_tools_executed:]
            enforce_turn_budget(
                turn_tool_msgs, env=get_active_env(effective_task_id)
            )

        # ── Budget pressure injection ────────────────────────────────────
        self._inject_budget_warning(messages, api_call_count)

    def _execute_sequential(
        self,
        assistant_message,
        messages: list,
        effective_task_id: str,
        api_call_count: int = 0,
        tool_delay: float = 0.0,
    ) -> None:
        """Execute tool calls sequentially (original behavior).

        Used for single calls or interactive tools. This is the fallback path
        when concurrent execution is not safe or appropriate.
        """
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # SAFETY: check interrupt BEFORE starting each tool.
            if self._interrupt_checker():
                remaining_calls = assistant_message.tool_calls[i - 1 :]
                if remaining_calls:
                    self._vprint(
                        f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)",
                        force=True,
                    )
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            # Reset nudge counters when the relevant tool is actually used
            self._reset_nudge_counter(function_name)

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    self._safe_print(
                        f"  📞 Tool {i}: {function_name}({list(function_args.keys())})"
                    )
                    self._safe_print(f"     Args: {args_str}")
                else:
                    args_preview = (
                        args_str[: self.log_prefix_chars] + "..."
                        if len(args_str) > self.log_prefix_chars
                        else args_str
                    )
                    self._safe_print(
                        f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}"
                    )

            self._touch_activity(f"executing tool: {function_name}")

            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback(
                        "tool.started", function_name, preview, function_args
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # Checkpoint: snapshot working dir before file-mutating tools
            self._check_checkpoint(function_name, function_args)

            tool_start_time = time.time()

            # Execute the tool
            try:
                function_result = self._tool_handler(
                    function_name,
                    function_args,
                    effective_task_id,
                    tool_call_id=tool_call.id,
                )
            except Exception as tool_error:
                function_result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error(
                    "tool_handler raised for %s: %s",
                    function_name,
                    tool_error,
                    exc_info=True,
                )

            tool_duration = time.time() - tool_start_time

            if self._should_emit_quiet_tool_messages():
                self._vprint(
                    f"  {_get_cute_tool_message_impl(function_name, function_args, tool_duration, result=function_result)}"
                )

            result_preview = (
                function_result
                if self.verbose_logging
                else (
                    function_result[:200]
                    if len(function_result) > 200
                    else function_result
                )
            )

            # Log tool errors
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if _is_error_result:
                logger.warning(
                    "Tool %s returned error (%.2fs): %s",
                    function_name,
                    tool_duration,
                    result_preview,
                )
            else:
                logger.info(
                    "tool %s completed (%.2fs, %d chars)",
                    function_name,
                    tool_duration,
                    len(function_result),
                )

            if self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed",
                        function_name,
                        None,
                        None,
                        duration=tool_duration,
                        is_error=_is_error_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(
                    f"Tool result ({len(function_result)} chars): {function_result}"
                )

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(
                        tool_call.id, function_name, function_args, function_result
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=function_name,
                tool_use_id=tool_call.id,
                env=get_active_env(effective_task_id),
            )

            if self._subdirectory_hints:
                subdir_hints = self._subdirectory_hints.check_tool_call(
                    function_name, function_args
                )
                if subdir_hints:
                    function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id,
            }
            messages.append(tool_msg)

            if not self.quiet_mode:
                if self.verbose_logging:
                    self._safe_print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    self._safe_print(f"     Result: {function_result}")
                else:
                    response_preview = (
                        function_result[: self.log_prefix_chars] + "..."
                        if len(function_result) > self.log_prefix_chars
                        else function_result
                    )
                    self._safe_print(
                        f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}"
                    )

            if self._interrupt_checker() and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(
                    f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)",
                    force=True,
                )
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            if tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(tool_delay)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools_seq = len(assistant_message.tool_calls)
        if num_tools_seq > 0:
            enforce_turn_budget(
                messages[-num_tools_seq:], env=get_active_env(effective_task_id)
            )

        # ── Budget pressure injection ─────────────────────────────────
        self._inject_budget_warning(messages, api_call_count)

    def get_budget_warning(self, api_call_count: int) -> Optional[str]:
        """Return a budget pressure string, or None if not yet needed.

        Two-tier system:
          - Caution (70%): nudge to consolidate work
          - Warning (90%): urgent, must respond now
        """
        if not self._budget_pressure_enabled or self.max_iterations <= 0:
            return None
        progress = api_call_count / self.max_iterations
        remaining = self.max_iterations - api_call_count
        if progress >= self._budget_warning_threshold:
            return (
                f"[BUDGET WARNING: Iteration {api_call_count}/{self.max_iterations}. "
                f"Only {remaining} iteration(s) left. "
                "Provide your final response NOW. No more tool calls unless absolutely critical.]"
            )
        if progress >= self._budget_caution_threshold:
            return (
                f"[BUDGET: Iteration {api_call_count}/{self.max_iterations}. "
                f"{remaining} iterations left. Start consolidating your work.]"
            )
        return None

    def _inject_budget_warning(self, messages: list, api_call_count: int) -> None:
        """Inject budget warning into the last tool result if needed."""
        budget_warning = self.get_budget_warning(api_call_count)
        if budget_warning and messages and messages[-1].get("role") == "tool":
            last_content = messages[-1]["content"]
            try:
                parsed = json.loads(last_content)
                if isinstance(parsed, dict):
                    parsed["_budget_warning"] = budget_warning
                    messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            except (json.JSONDecodeError, TypeError):
                messages[-1]["content"] = last_content + f"\n\n{budget_warning}"
            if not self.quiet_mode:
                remaining = self.max_iterations - api_call_count
                tier = (
                    "⚠️  WARNING"
                    if remaining <= self.max_iterations * 0.1
                    else "💡 CAUTION"
                )
                self._safe_print(f"{self.log_prefix}{tier}: {remaining} iterations remaining")

    def should_parallelize(self, tool_calls) -> bool:
        """Determine if a batch of tool calls should be executed concurrently."""
        return _should_parallelize_tool_batch(tool_calls)


# =============================================================================
# Backward compatibility exports
# =============================================================================

__all__ = [
    "ToolExecutor",
    "_should_parallelize_tool_batch",
    "_extract_parallel_scope_path",
    "_paths_overlap",
    "_is_destructive_command",
]
