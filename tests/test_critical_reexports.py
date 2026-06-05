"""
Regression test for Iron Law III: Critical Re-Export Integrity.

Ensures that all load-bearing re-exports survive code modifications.
This test MUST pass before any gateway restart.

Root cause: The May 28 overnight optimization cron pruned imports from
run_agent.py that were re-exported for agent/tool_executor.py's _ra()
lazy import pattern. This broke ALL tool calls with:
    AttributeError: module 'run_agent' has no attribute 'handle_function_call'

This test prevents recurrence by asserting every critical re-export exists.
"""

import pytest


class TestRunAgentReExports:
    """Verify run_agent.py re-exports all symbols that downstream modules depend on."""

    @pytest.fixture(autouse=True)
    def _import_run_agent(self):
        import run_agent
        self.mod = run_agent

    # --- Tool call execution path (breaks ALL tools if missing) ---

    def test_handle_function_call_re_exported(self):
        """handle_function_call from model_tools — used by agent/tool_executor.py _ra()"""
        assert hasattr(self.mod, "handle_function_call"), (
            "run_agent.handle_function_call MISSING — ALL tool calls will fail. "
            "This re-export from model_tools is consumed by agent/tool_executor.py line 749."
        )

    def test_get_tool_definitions_re_exported(self):
        assert hasattr(self.mod, "get_tool_definitions")

    def test_get_toolset_for_tool_re_exported(self):
        assert hasattr(self.mod, "get_toolset_for_tool")

    # --- Agent identity path ---

    def test_load_soul_md_re_exported(self):
        """load_soul_md from agent/prompt_builder — used by cron jobs for identity"""
        assert hasattr(self.mod, "load_soul_md"), (
            "run_agent.load_soul_md MISSING — cron jobs building agent identity will fail. "
            "This re-export from agent/prompt_builder.py is patched in tests."
        )

    def test_build_environment_hints_re_exported(self):
        assert hasattr(self.mod, "build_environment_hints")

    def test_build_context_files_prompt_re_exported(self):
        assert hasattr(self.mod, "build_context_files_prompt")

    def test_build_skills_system_prompt_re_exported(self):
        assert hasattr(self.mod, "build_skills_system_prompt")

    # --- Core classes ---

    def test_ai_agent_class_exists(self):
        assert hasattr(self.mod, "AIAgent")

    def test_context_compressor_re_exported(self):
        """ContextCompressor from agent/context_compressor — patched in tests"""
        assert hasattr(self.mod, "ContextCompressor")

    # --- Interrupt / cleanup ---

    def test_set_interrupt_re_exported(self):
        assert hasattr(self.mod, "_set_interrupt")

    def test_sanitize_context_re_exported(self):
        assert hasattr(self.mod, "sanitize_context")

    def test_redact_sensitive_text_re_exported(self):
        assert hasattr(self.mod, "redact_sensitive_text")


class TestGatewayCustomHandlers:
    """Verify custom slash command handlers survive upstream merges."""

    def test_optimize_handler_in_running_agent_path(self):
        """The /optimize handler block after the subgoal handler must exist."""
        import inspect
        from gateway.run import GatewayRunner

        source = inspect.getsource(GatewayRunner._handle_message)
        # The optimize handler checks canonical == "optimize" in the running-agent path
        assert 'canonical == "optimize"' in source or "canonical == 'optimize'" in source, (
            "/optimize handler block MISSING from running-agent path in gateway/run.py. "
            "Likely clobbered by upstream merge. Re-apply from ~/.hermes/patches/"
        )

    def test_optimize_handler_count(self):
        """There should be at least 2 /optimize dispatch blocks in gateway/run.py."""
        with open("gateway/run.py", "r") as f:
            content = f.read()
        # Both forms: canonical == "optimize" and .name == "optimize"
        canonical_count = content.count('canonical == "optimize"')
        name_count = content.count('.name == "optimize"')
        total = canonical_count + name_count
        assert total >= 2, (
            f"Expected >= 2 /optimize dispatch blocks in gateway/run.py, found {total} "
            f"(canonical={canonical_count}, name={name_count}). "
            "Upstream merge likely clobbered them."
        )


class TestToolExecutorImportPath:
    """Verify the _ra() lazy import pattern in tool_executor works."""

    def test_ra_returns_run_agent_with_handle_function_call(self):
        """The _ra() function must return a module with handle_function_call."""
        from agent.tool_executor import _ra
        mod = _ra()
        assert hasattr(mod, "handle_function_call"), (
            "_ra().handle_function_call MISSING — the lazy import path used by "
            "execute_tool_calls_sequential is broken."
        )

    def test_ra_returns_run_agent_with_load_soul_md(self):
        from agent.tool_executor import _ra
        mod = _ra()
        assert hasattr(mod, "load_soul_md")
