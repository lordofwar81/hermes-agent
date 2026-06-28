"""Fast-path fixtures shared across tests/run_agent/.

Many tests in this directory exercise the retry/backoff paths in the
agent loop. Production code uses ``jittered_backoff(base_delay=5.0)``
with a ``while time.time() < sleep_end`` loop — a single retry test
spends 5+ seconds of real wall-clock time on backoff waits.

Mocking ``jittered_backoff`` to return 0.0 collapses the while-loop
to a no-op (``time.time() < time.time() + 0`` is false immediately),
which handles the most common case without touching ``time.sleep``.

We deliberately DO NOT mock ``time.sleep`` here — some tests
(test_interrupt_propagation, test_primary_runtime_restore, etc.) use
the real ``time.sleep`` for threading coordination or assert that it
was called with specific values. Tests that want to additionally
fast-path direct ``time.sleep(N)`` calls in production code should
monkeypatch ``run_agent.time.sleep`` locally (see
``test_anthropic_error_handling.py`` for the pattern).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fast_retry_backoff(monkeypatch):
    """Short-circuit retry backoff for all tests in this directory."""
    # Patch jittered_backoff wherever it's imported. The function lived in
    # run_agent.py initially, was extracted to agent.retry_utils, and is
    # imported by agent.conversation_loop and agent.chat_completion_helpers.
    _noop_backoff = lambda *a, **k: 0.0
    for _mod_path in ("run_agent", "agent.conversation_loop", "agent.chat_completion_helpers"):
        try:
            import importlib
            _mod = importlib.import_module(_mod_path)
            if hasattr(_mod, "jittered_backoff"):
                monkeypatch.setattr(_mod, "jittered_backoff", _noop_backoff)
        except (ImportError, ModuleNotFoundError):
            pass


@pytest.fixture(autouse=True)
def _isolate_background_review_thread(monkeypatch, request):
    """Stub the real I/O the background-review worker thread performs inline.

    ``AIAgent._spawn_background_review`` constructs a ``threading.Thread``
    whose target (``agent.background_review._run_review_in_thread``) does
    real work before building the review fork: it loads tool definitions
    (which reads live config + imports plugins) and installs a thread-local
    tool whitelist. The background_review test files patch ``run_agent.AIAgent``
    and ``threading.Thread`` so the worker runs synchronously inside the
    test, but without stubbing the tool-definition / whitelist calls that
    worker makes, the synchronous run hangs on real config/plugin I/O
    (observed: pytest-timeout fires at >10s, and under the per-file runner
    the whole file exceeds the 140s budget and gets SIGKILL'd).

    Scoped to the three background_review test files so it doesn't perturb
    unrelated tests. The stubs are inert: a fixed memory+skills tool list
    (matching what the worker requests via enabled_toolsets=["memory",
    "skills"]) so the whitelist the worker builds is realistic, and no-op
    whitelist install/clear. The action summarizer is left real (pure logic,
    no I/O). Tests that need to assert on tool definitions override locally.
    """
    _BG_REVIEW_FILES = (
        "test_background_review",
        "test_background_review_cache_parity",
        "test_background_review_toolset_restriction",
    )
    _module_name = request.module.__name__ if request.module else ""
    if not any(_module_name.endswith(_f) for _f in _BG_REVIEW_FILES):
        return

    # The worker builds its thread-local whitelist from the tool names
    # returned here. Use the real memory+skills tool names so tests that
    # assert on whitelist contents see a realistic set; keep the list
    # limited to memory/skills so "dangerous tools not in whitelist"
    # assertions still hold.
    _STUB_TOOLS = [
        {"type": "function", "function": {"name": n, "parameters": {}}}
        for n in ("memory", "skill_manage", "skill_view", "skills_list")
    ]

    # _run_review_in_thread imports these locally, so patch at the source
    # modules. Return-value shape must match what the worker iterates.
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda *a, **k: list(_STUB_TOOLS),
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.set_thread_tool_whitelist",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.clear_thread_tool_whitelist",
        lambda *a, **k: None,
    )
    # NOTE: summarize_background_review_actions is deliberately NOT stubbed
    # here. It is pure logic over the review's captured tool messages (no
    # I/O, no config reads), so it doesn't contribute to the hang, and
    # several tests assert on the actions it produces. Tests that need a
    # no-op summarizer patch it locally.
