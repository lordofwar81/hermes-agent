"""Integration tests for the ``_run_agent_via_proxy`` extraction (round 53).

Exercises ``RunAgentViaProxyMixin._run_agent_via_proxy`` end-to-end with the
method moved onto ``RunAgentViaProxyMixin``. The method is the proxy-mode
agent forwarder — when ``_get_proxy_url()`` is set, it relays the message to
a remote Hermes API server via SSE-streamed ``POST /v1/chat/completions``.

Contract guarantee of the extraction:

1. **MRO resolution** —
   ``RunAgentViaProxyMixin._run_agent_via_proxy
      is GatewayRunner._run_agent_via_proxy``.
   Without this, the old inline method would still be shadowing the mixin
   (or the mixin wasn't added to the bases) and the extraction is a no-op.
   The 9 direct calls in ``test_proxy_mode.py`` resolve via MRO through this
   same identity.

Behavioral coverage (SSE parsing, stale-generation discard, stream-consumer
setup, error paths) is provided by ``test_proxy_mode.py`` — that module is
the established direct-caller harness with ``_FakeSession`` /
``_FakeSSEResponse`` doubles. This file pins only the MRO contract so the
decomposition invariant (method resolves to the mixin) is checked in
isolation, mirroring the R51/R52 extraction-test pattern.
"""

import importlib

import gateway.run as gateway_run
from gateway.run_agent_via_proxy_mixin import RunAgentViaProxyMixin


def test_method_lives_on_mixin_and_resolves_via_mro():
    """``_run_agent_via_proxy`` must resolve to the mixin through the MRO.

    The single most important assertion: proves the method actually lives on
    ``RunAgentViaProxyMixin`` and that mixin is in ``GatewayRunner``'s bases.
    If the inline method were still in run.py (shadowing the mixin) or the
    mixin wasn't appended to the bases, this identity check fails. This same
    resolution backs the 9 direct calls in ``test_proxy_mode.py``.
    """
    assert (
        RunAgentViaProxyMixin._run_agent_via_proxy
        is gateway_run.GatewayRunner._run_agent_via_proxy
    )
    assert RunAgentViaProxyMixin in gateway_run.GatewayRunner.__mro__
