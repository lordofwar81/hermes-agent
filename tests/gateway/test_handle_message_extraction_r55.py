"""Integration tests for the ``_handle_message`` extraction (round 55).

Exercises ``HandleMessageMixin._handle_message`` end-to-end with the method
moved onto ``HandleMessageMixin``. The method is the top-level inbound
message dispatcher — authorization, slash-command routing, busy-agent
dispatch (R50 delegate), and the inner handler call (R51 delegate).

Contract guarantee of the extraction:

1. **MRO resolution** —
   ``HandleMessageMixin._handle_message is GatewayRunner._handle_message``.
   Without this, the old inline method would still be shadowing the mixin
   (or the mixin wasn't added to the bases) and the extraction is a no-op.
   The direct calls in ``test_42039_duplicate_user_message``,
   ``test_discord_free_response``, ``test_unauthorized_dm_behavior``,
   ``test_discord_slash_commands``, ``test_discord_document_handling``,
   ``test_reload_skills_command``, ``test_signal``, ``test_session_race_guard``,
   ``test_discord_channel_prompts``, ``test_runner_harness`` resolve via MRO
   through this same identity.

Behavioral coverage (authorization denial, slash-command dispatch, busy-agent
interrupt, session-race guard, duplicate-write guard) is provided by the ten
existing direct-caller test modules — those modules are the established
harnesses and were unchanged this round. This file pins only the MRO contract
so the decomposition invariant (method resolves to the mixin) is checked in
isolation, mirroring the R51-R54 extraction-test pattern.
"""

import gateway.run as gateway_run
from gateway.handle_message_mixin import HandleMessageMixin


def test_method_lives_on_mixin_and_resolves_via_mro():
    """``_handle_message`` must resolve to the mixin through the MRO.

    The single most important assertion: proves the method actually lives on
    ``HandleMessageMixin`` and that mixin is in ``GatewayRunner``'s bases.
    If the inline method were still in run.py (shadowing the mixin) or the
    mixin wasn't appended to the bases, this identity check fails. This same
    resolution backs the direct calls in the ten _handle_message test modules.
    """
    assert (
        HandleMessageMixin._handle_message
        is gateway_run.GatewayRunner._handle_message
    )
    assert HandleMessageMixin in gateway_run.GatewayRunner.__mro__
