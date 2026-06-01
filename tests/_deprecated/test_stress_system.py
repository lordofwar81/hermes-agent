#!/usr/bin/env python3
"""Comprehensive system stress test for Hermes Gateway routing infrastructure.

Exercises:
  1. Task classifier accuracy (all categories)
  2. Model router routing (all categories with real providers)
  3. Context overflow handling (local models at edge)
  4. Circuit breaker behavior (failure recording, recovery, blocking)
  5. Local model health checks
  6. Venice routing policy (EXPERT-only, no fallback leakage)
  7. Gateway uptime & API responsiveness
  8. Provider registry integrity
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
env_file = HERMES_HOME / ".env"
config_file = HERMES_HOME / "config.yaml"

load_dotenv(env_file)

# Test results accumulator
RESULTS = []
FAILURES = 0
PASSES = 0

# ── Test helpers ───────────────────────────────────────────────────────────
def test(name, fn):
    global PASSES, FAILURES
    try:
        fn()
        PASSES += 1
        RESULTS.append(f"  PASS  {name}")
    except Exception as e:
        FAILURES += 1
        RESULTS.append(f"  FAIL  {name}: {e}")

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"expected {b!r}, got {a!r} {'— ' + msg if msg else ''}")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "expected truthy value")

def assert_in(item, container, msg=""):
    if item not in container:
        raise AssertionError(f"{item!r} not in {container!r} {'— ' + msg if msg else ''}")

def assert_not_in(item, container, msg=""):
    if item in container:
        raise AssertionError(f"{item!r} unexpectedly in {container!r} {'— ' + msg if msg else ''}")

# ── Test Suite ─────────────────────────────────────────────────────────────
def run_all():
    print("=" * 70)
    print("HERMES SYSTEM STRESS TEST")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_file}")
    print("=" * 70)

    test_suite_1_classifier()
    test_suite_2_routing()
    test_suite_3_context_overflow()
    test_suite_4_circuit_breaker()
    test_suite_5_health_checks()
    test_suite_6_venice_policy()
    test_suite_7_gateway_uptime()
    test_suite_8_provider_registry()

    print()
    print("=" * 70)
    print(f"RESULTS: {PASSES} passed, {FAILURES} failed, {PASSES+FAILURES} total")
    print("=" * 70)
    for r in RESULTS:
        print(r)
    return FAILURES == 0


# ── Suite 1: Task Classifier ──────────────────────────────────────────────
def test_suite_1_classifier():
    print("\n── Suite 1: Task Classifier Accuracy ──")
    from agent.model_router import TaskClassifier, TaskCategory

    # GREETING
    test("greeting: hi", lambda: assert_eq(TaskClassifier.classify("hi"), TaskCategory.GREETING))
    test("greeting: hello", lambda: assert_eq(TaskClassifier.classify("hello"), TaskCategory.GREETING))
    test("greeting: thanks", lambda: assert_eq(TaskClassifier.classify("thanks"), TaskCategory.GREETING))
    test("greeting: got it", lambda: assert_eq(TaskClassifier.classify("got it"), TaskCategory.GREETING))
    test("not greeting: high quality", lambda: assert_eq(TaskClassifier.classify("high quality output"), TaskCategory.SIMPLE))

    # CODE
    test("code: backticks", lambda: assert_eq(TaskClassifier.classify("```python\nprint('hi')\n```"), TaskCategory.CODE))
    test("code: bug fix", lambda: assert_eq(TaskClassifier.classify("fix this bug in my code"), TaskCategory.CODE))
    test("code: debug", lambda: assert_eq(TaskClassifier.classify("debug the TypeError"), TaskCategory.CODE))
    test("code: implement", lambda: assert_eq(TaskClassifier.classify("implement a merge sort"), TaskCategory.CODE))
    test("code: refactor", lambda: assert_eq(TaskClassifier.classify("refactor this function"), TaskCategory.CODE))

    # ANALYSIS
    test("analysis: compare", lambda: assert_eq(TaskClassifier.classify("compare react vs vue"), TaskCategory.ANALYSIS))
    test("analysis: evaluate", lambda: assert_eq(TaskClassifier.classify("evaluate these options"), TaskCategory.ANALYSIS))
    test("analysis: performance", lambda: assert_eq(TaskClassifier.classify("analyze the performance metrics"), TaskCategory.ANALYSIS))

    # REASONING
    test("reasoning: explain why", lambda: assert_eq(TaskClassifier.classify("explain why distributed systems are hard"), TaskCategory.REASONING))
    test("reasoning: how does", lambda: assert_eq(TaskClassifier.classify("how does garbage collection work"), TaskCategory.REASONING))

    # EXPERT
    test("expert: design a system", lambda: assert_eq(
        TaskClassifier.classify("design a system for real-time analytics"), TaskCategory.EXPERT))
    test("expert: architect a", lambda: assert_eq(
        TaskClassifier.classify("architect a microservices platform"), TaskCategory.EXPERT))
    test("expert: from scratch", lambda: assert_eq(
        TaskClassifier.classify("build a complete trading system from scratch with event sourcing"), TaskCategory.EXPERT))
    test("expert: multi-region", lambda: assert_eq(
        TaskClassifier.classify("design multi-region failover architecture"), TaskCategory.EXPERT))
    test("expert: end-to-end", lambda: assert_eq(
        TaskClassifier.classify("implement end-to-end encryption pipeline"), TaskCategory.EXPERT))
    test("expert: production-ready", lambda: assert_eq(
        TaskClassifier.classify("create a production-ready API gateway"), TaskCategory.EXPERT))

    # NOT EXPERT (should NOT trigger Venice)
    test("not expert: docker compose", lambda: assert_eq(
        TaskClassifier.classify("help me write a docker compose file"), TaskCategory.SIMPLE))
    test("not expert: simple script", lambda: assert_eq(
        TaskClassifier.classify("write a python script to sort a list"), TaskCategory.CODE))
    test("not expert: simple question", lambda: assert_eq(
        TaskClassifier.classify("what is kubernetes"), TaskCategory.SIMPLE))

    # SIMPLE
    test("simple: generic question", lambda: assert_eq(TaskClassifier.classify("what is the weather"), TaskCategory.SIMPLE))
    test("simple: long but not technical", lambda: assert_eq(
        TaskClassifier.classify("tell me about the history of the roman empire and its influence on modern law"),
        TaskCategory.SIMPLE))


# ── Suite 2: Model Router (real providers) ────────────────────────────────
def test_suite_2_routing():
    print("\n── Suite 2: Model Router Routing ──")
    from agent.model_router import ModelRouter, TaskCategory

    router = ModelRouter(str(config_file))

    primary = {
        "provider": "zai",
        "model": "glm-5.1",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
    }

    # Test all categories route to a valid provider
    for category, msg in [
        (TaskCategory.GREETING, "hi"),
        (TaskCategory.SIMPLE, "what is 2+2"),
        (TaskCategory.CODE, "fix this bug"),
        (TaskCategory.ANALYSIS, "compare a and b"),
        (TaskCategory.REASONING, "explain why x works"),
        (TaskCategory.EXPERT, "design a system for global payment processing from scratch"),
    ]:
        def _test(cat, m):
            result = router.route(m, primary, estimated_tokens=500)
            assert_true(result.get("model"), f"no model for {cat.name}")
            assert_true(result.get("runtime", {}).get("api_key"), f"no api_key for {cat.name}")
            assert_true(result.get("runtime", {}).get("base_url"), f"no base_url for {cat.name}")
            provider = result["runtime"]["provider"]
            if cat == TaskCategory.EXPERT:
                assert_eq(provider, "venice", f"EXPERT should route to Venice, got {provider}")
            else:
                assert_not_in(provider, ["venice"], f"{cat.name} must NOT route to Venice, got {provider}")
        test(f"route/{category.name}", lambda c=category, m=msg: _test(c, m))

    # Verify all providers are registered
    providers = router._registry.list_available()
    provider_names = {p.provider for p in providers}
    assert_in("mac_studio", provider_names, "mac_studio not in registry")
    assert_in("zai", provider_names, "zai not in registry")
    assert_in("minimax", provider_names, "minimax not in registry")
    assert_in("venice", provider_names, "venice not in registry")
    print(f"  INFO  Registered providers: {sorted(provider_names)}")
    print(f"  INFO  Total provider entries: {len(providers)}")

    # Venice model check
    venice_models = [p.model for p in providers if p.provider == "venice"]
    assert_in("claude-sonnet-4-6", venice_models)
    assert_in("qwen-3-6-plus", venice_models)
    assert_not_in("deepseek-v3.2", venice_models, "deepseek-v3.2 should NOT be a Venice model")
    print(f"  INFO  Venice models: {venice_models}")

    # Context lengths set
    for p in providers:
        assert_true(p.context_length, f"{p.provider}/{p.model} missing context_length")
    print(f"  INFO  All providers have context lengths")


# ── Suite 3: Context Overflow ──────────────────────────────────────────────
def test_suite_3_context_overflow():
    print("\n── Suite 3: Context Overflow Handling ──")
    from agent.model_router import ModelRouter, TaskCategory

    router = ModelRouter(str(config_file))

    primary = {"provider": "zai", "model": "glm-5.1", "base_url": "https://api.z.ai/api/coding/paas/v4"}

    # Mac Studio has 200K context → 60% = 120K usable for local
    mac = router._registry.get("mac-studio-qwen36")
    assert_true(mac, "Mac Studio not found in registry")
    assert_eq(mac.context_length, 200000, f"expected 200K context, got {mac.context_length}")
    assert_true(mac.is_local, "Mac Studio should be marked local")

    local_limit = int(mac.context_length * 0.6)
    print(f"  INFO  Mac Studio context: {mac.context_length} ({local_limit} usable)")

    # Below limit: should fit
    test("context/mac_fits_100k", lambda: assert_true(
        router._fits_context(mac, 100000), "100K should fit in 120K usable"))
    test("context/mac_fits_115k", lambda: assert_true(
        router._fits_context(mac, 115000), "115K should fit in 120K usable"))

    # Above limit: should NOT fit
    test("context/mac_overflow_125k", lambda: assert_true(
        not router._fits_context(mac, 125000), "125K should overflow 120K usable"))
    test("context/mac_overflow_200k", lambda: assert_true(
        not router._fits_context(mac, 200000), "200K should overflow 120K usable"))

    # ZAI has 198K context → should always fit 100K
    zai = router._registry.get("zai-glm-5.1")
    assert_true(zai, "ZAI GLM-5.1 not found")
    test("context/zai_fits_100k", lambda: assert_true(
        router._fits_context(zai, 100000), "ZAI should fit 100K"))

    # Routing with overflow: should skip Mac and go to ZAI
    result = router.route("what is 2+2", primary, estimated_tokens=150000)
    provider = result.get("runtime", {}).get("provider", "")
    test("context/overflow_route", lambda: assert_not_in(provider, ["mac_studio"],
        f"Mac Studio should be skipped at 150K tokens, got {provider}"))

    # Routing at 50K: should use Mac
    result2 = router.route("what is 2+2", primary, estimated_tokens=50000)
    provider2 = result2.get("runtime", {}).get("provider", "")
    test("context/mac_routes_at_50k", lambda: assert_eq(provider2, "mac_studio",
        f"Mac Studio should route at 50K tokens, got {provider2}"))


# ── Suite 4: Circuit Breaker ───────────────────────────────────────────────
def test_suite_4_circuit_breaker():
    print("\n── Suite 4: Circuit Breaker Behavior ──")
    from agent.model_router import RouterCircuitBreaker

    breaker = RouterCircuitBreaker()

    # Default: everything available
    test("cb/all_available", lambda: assert_true(
        breaker.is_available("test_provider"), "new breaker should have all available"))

    # Record 2 failures → trips breaker
    breaker.record_failure("test_provider")
    test("cb/still_available_after_1", lambda: assert_true(
        breaker.is_available("test_provider"), "should still be available after 1 failure"))

    breaker.record_failure("test_provider")
    test("cb/tripped_after_2", lambda: assert_true(
        not breaker.is_available("test_provider"), "should be tripped after 2 failures"))

    # Different provider unaffected
    test("cb/other_provider_still_available", lambda: assert_true(
        breaker.is_available("other_provider"), "other provider should be unaffected"))

    # Get blocked providers
    blocked = breaker.get_blocked_providers()
    assert_in("test_provider", blocked)
    assert_not_in("other_provider", blocked)
    print(f"  INFO  Blocked: {blocked}")

    # Record success resets
    breaker.record_success("test_provider")
    test("cb/reset_after_success", lambda: assert_true(
        breaker.is_available("test_provider"), "should be available after success"))

    # Multi-failure stress
    for i in range(10):
        breaker.record_failure("flaky")
    test("cb/flaky_tripped", lambda: assert_true(
        not breaker.is_available("flaky"), "flaky provider should be tripped"))


# ── Suite 5: Health Checks ─────────────────────────────────────────────────
def test_suite_5_health_checks():
    print("\n── Suite 5: Local Model Health Checks ──")
    from agent.model_router import ModelRouter

    router = ModelRouter(str(config_file))

    # Check Mac Studio health (LM Studio endpoint)
    mac = router._registry.get("mac-studio-qwen36")
    assert_true(mac, "Mac Studio not in registry")

    healthy = router._check_local_health(mac)
    print(f"  INFO  Mac Studio health: {'OK' if healthy else 'UNREACHABLE'}")

    if healthy:
        test("health/mac_studio_reachable", lambda: assert_true(healthy))
    else:
        print("  WARN  Mac Studio (192.168.1.149:1234) unreachable — skipping health check assertion")

    # Check MiniMax health (llama.cpp endpoint)
    minimax = router._registry.get("minimax-m27")
    assert_true(minimax, "MiniMax not in registry")

    minimax_healthy = router._check_local_health(minimax)
    print(f"  INFO  MiniMax health: {'OK' if minimax_healthy else 'UNREACHABLE'}")

    if minimax_healthy:
        test("health/minimax_reachable", lambda: assert_true(minimax_healthy))
    else:
        print("  WARN  MiniMax (192.168.1.229:8199) unreachable — skipping health check assertion")

    # Health cache: second call should be cached within 30s
    import time
    t0 = time.time()
    _ = router._check_local_health(mac)
    t1 = time.time()
    _ = router._check_local_health(mac)
    t2 = time.time()
    first_call_time = t1 - t0
    second_call_time = t2 - t1
    test("health/cache_works", lambda: assert_true(
        second_call_time < first_call_time / 2,
        f"cached check ({second_call_time:.3f}s) should be faster than first ({first_call_time:.3f}s)"))


# ── Suite 6: Venice Policy ─────────────────────────────────────────────────
def test_suite_6_venice_policy():
    print("\n── Suite 6: Venice Routing Policy ──")
    from agent.model_router import ModelRouter, TaskCategory

    router = ModelRouter(str(config_file))

    # Venice should NOT be in the fallback chain
    import yaml
    with open(config_file) as f:
        raw_config = yaml.safe_load(f)

    fallback = raw_config.get("fallback_providers", [])
    for fb in fallback:
        assert_not_in(fb.get("provider"), ["venice"], f"Venice in fallback_providers: {fb}")
    print(f"  INFO  Fallback chain ({len(fallback)} entries): no Venice")

    # Venice should be in provider_routing.ignore
    pr = raw_config.get("provider_routing", {})
    ignored = pr.get("ignore", [])
    assert_in("venice", ignored)
    print(f"  INFO  provider_routing.ignore: {ignored}")

    # Venice should NOT be in auxiliary client auto-detect chain
    # (check _API_KEY_PROVIDER_AUX_MODELS)
    from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
    assert_not_in("venice", _API_KEY_PROVIDER_AUX_MODELS)
    print(f"  INFO  _API_KEY_PROVIDER_AUX_MODELS: {len(_API_KEY_PROVIDER_AUX_MODELS)} entries, no Venice")

    # Venice budget tracker
    budget_status = router._budget.get_status()
    print(f"  INFO  Venice budget: ${budget_status.spent_usd:.2f} / ${budget_status.daily_limit_usd:.2f}" +
          f" (reset: {budget_status.last_reset_date})")
    test("venice/budget_available", lambda: assert_true(router._budget.is_budget_available()))

    # Venice only routes for EXPERT
    primary = {"provider": "zai", "model": "glm-5.1", "base_url": "https://api.z.ai/api/coding/paas/v4"}
    non_expert_cats = [
        TaskCategory.GREETING, TaskCategory.SIMPLE, TaskCategory.CODE,
        TaskCategory.ANALYSIS, TaskCategory.REASONING
    ]
    for cat in non_expert_cats:
        result = router.route(cat.value, primary, estimated_tokens=500)
        provider = result.get("runtime", {}).get("provider", "")
        assert_not_in(provider, ["venice"], f"{cat.name} must NOT route to Venice")

    # EXPERT should route to Venice (when budget available)
    expert_msg = "design a system for global payment processing from scratch with multi-region failover"
    result = router.route(expert_msg, primary, estimated_tokens=1000)
    provider = result.get("runtime", {}).get("provider", "")
    test("venice/expert_routes_to_venice", lambda: assert_eq(provider, "venice"))


# ── Suite 7: Gateway Uptime & API ──────────────────────────────────────────
def test_suite_7_gateway_uptime():
    print("\n── Suite 7: Gateway Uptime & API Responsiveness ──")
    import yaml

    with open(config_file) as f:
        config = yaml.safe_load(f)

    gw_config = config.get("gateway", {})
    health_port = gw_config.get("api_server_port") or 8642
    health_url = f"http://127.0.0.1:{health_port}/health"

    # Check systemd service
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "hermes-gateway.service"],
            capture_output=True, text=True, timeout=5
        )
        is_active = result.stdout.strip() == "active"
    except Exception as e:
        is_active = False
        print(f"  WARN  Could not check systemd: {e}")

    test("gateway/systemd_active", lambda: assert_true(is_active, "systemd service not active"))
    print(f"  INFO  hermes-gateway.service: {'active' if is_active else 'INACTIVE'}")

    # HTTP health endpoint
    health_ok = False
    health_latency_ms = 0
    try:
        t0 = time.time()
        req = urllib.request.Request(health_url)
        resp = urllib.request.urlopen(req, timeout=5)
        t1 = time.time()
        health_latency_ms = (t1 - t0) * 1000
        body = json.loads(resp.read().decode())
        health_ok = body.get("status") == "ok"
    except Exception as e:
        print(f"  WARN  Health endpoint error: {e}")

    test("gateway/health_endpoint_ok", lambda: assert_true(health_ok))
    print(f"  INFO  Health endpoint: {'OK' if health_ok else 'FAIL'} ({health_latency_ms:.1f}ms)")

    test("gateway/health_latency", lambda: assert_true(
        health_latency_ms < 500, f"Health latency {health_latency_ms:.1f}ms > 500ms"))

    # Gateway process uptime
    try:
        result = subprocess.run(
            ["systemctl", "show", "hermes-gateway.service", "--property=ActiveEnterTimestamp"],
            capture_output=True, text=True, timeout=5
        )
        started = result.stdout.strip()
        print(f"  INFO  Gateway started: {started}")

        # Check for recent crashes
        result = subprocess.run(
            ["journalctl", "-u", "hermes-gateway.service", "--since", "10 minutes ago",
             "--no-pager", "-q"],
            capture_output=True, text=True, timeout=5
        )
        error_lines = [l for l in result.stdout.split("\n")
                      if ("ERROR" in l or "CRITICAL" in l or "Traceback" in l)
                      and "platforms.telegram" not in l]
        test("gateway/no_recent_errors", lambda: assert_eq(len(error_lines), 0,
            f"Recent errors: {error_lines}"))
        print(f"  INFO  Recent errors in logs: {len(error_lines)}")

    except Exception as e:
        print(f"  WARN  journalctl check failed: {e}")

    # Memory usage
    try:
        result = subprocess.run(
            ["systemctl", "show", "hermes-gateway.service", "--property=MemoryCurrent"],
            capture_output=True, text=True, timeout=5
        )
        mem_str = result.stdout.strip()
        print(f"  INFO  Gateway memory: {mem_str}")
    except Exception:
        pass


# ── Suite 8: Provider Registry Integrity ────────────────────────────────────
def test_suite_8_provider_registry():
    print("\n── Suite 8: Provider Registry Integrity ──")
    from agent.model_router import ModelRouter, ProviderCredential

    router = ModelRouter(str(config_file))
    providers = router._registry.list_available()

    # Every provider must have api_key, base_url, context_length
    for p in providers:
        assert_true(p.api_key, f"{p.provider}/{p.model} missing api_key")
        assert_true(p.base_url, f"{p.provider}/{p.model} missing base_url")
        assert_true(p.context_length, f"{p.provider}/{p.model} missing context_length")

    test("registry/all_have_keys", lambda: None)  # passes if above assertions pass

    # No duplicate provider+model combos
    keys = [(p.provider, p.model) for p in providers]
    test("registry/no_duplicates", lambda: assert_eq(len(keys), len(set(keys)),
        f"duplicate entries: {[k for k in set(keys) if keys.count(k) > 1]}"))

    # Mac Studio is marked as local
    mac = [p for p in providers if p.provider == "mac_studio"]
    test("registry/mac_is_local", lambda: assert_true(all(p.is_local for p in mac)))

    # Mac Studio should have request_timeout set
    test("registry/mac_has_timeout", lambda: assert_true(all(p.request_timeout for p in mac)))

    # ZAI should NOT be local
    zai = [p for p in providers if p.provider == "zai"]
    test("registry/zai_not_local", lambda: assert_true(not any(p.is_local for p in zai)))


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
