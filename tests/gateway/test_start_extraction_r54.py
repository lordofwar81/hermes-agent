"""Integration tests for the ``start`` extraction (round 54).

Exercises ``GatewayStartMixin.start`` end-to-end with the method moved onto
``GatewayStartMixin``. The method is the gateway startup orchestrator —
platform connect loop, retryable/non-retryable error classification, hook
emission, background-watcher launches.

Contract guarantee of the extraction:

1. **MRO resolution** —
   ``GatewayStartMixin.start is GatewayRunner.start``.
   Without this, the old inline method would still be shadowing the mixin
   (or the mixin wasn't added to the bases) and the extraction is a no-op.
   The direct calls in ``test_runner_startup_failures``,
   ``test_platform_reconnect``, ``test_startup_preflight``, and
   ``test_start_gateway`` resolve via MRO through this same identity.

Behavioral coverage (platform connect failures, preflight gating, reconnect
watcher, degraded mode) is provided by the four existing startup test
modules — those modules are the established direct-caller harnesses and
were unchanged this round. This file pins only the MRO contract so the
decomposition invariant (method resolves to the mixin) is checked in
isolation, mirroring the R51/R52/R53 extraction-test pattern.
"""

import gateway.run as gateway_run
from gateway.start_mixin_r54 import GatewayStartMixin


def test_method_lives_on_mixin_and_resolves_via_mro():
    """``start`` must resolve to the mixin through the MRO.

    The single most important assertion: proves the method actually lives on
    ``GatewayStartMixin`` and that mixin is in ``GatewayRunner``'s bases.
    If the inline method were still in run.py (shadowing the mixin) or the
    mixin wasn't appended to the bases, this identity check fails. This same
    resolution backs the direct calls in the four startup test modules.
    """
    assert GatewayStartMixin.start is gateway_run.GatewayRunner.start
    assert GatewayStartMixin in gateway_run.GatewayRunner.__mro__
