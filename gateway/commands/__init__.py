"""
Gateway command handler modules.

This package contains command handler mixins extracted from GatewayRunner
to improve code organization and maintainability.

Each mixin provides handlers for a specific category of slash commands:
- CoreCommandMixin: Basic gateway commands (reset, stop, help, etc.)
- ConfigCommandMixin: Configuration commands (model, reasoning, fast, etc.)
- PlatformCommandMixin: Platform-specific commands (platform, topic, voice, etc.)
- WorkflowCommandMixin: Workflow management (goal, subgoal, undo, rollback, etc.)
- InfoCommandMixin: Information display (status, agents, kanban, retry)
- AdminCommandMixin: Administrative commands (update, reload, debug, approve, etc.)
- OtherCommandMixin: Miscellaneous commands (footer, resume, branch, usage, insights, bundles)
"""

from gateway.commands.admin_commands import AdminCommandMixin
from gateway.commands.config_commands import ConfigCommandMixin
from gateway.commands.core_commands import CoreCommandMixin
from gateway.commands.info_commands import InfoCommandMixin
from gateway.commands.other_commands import OtherCommandMixin
from gateway.commands.platform_commands import PlatformCommandMixin
from gateway.commands.workflow_commands import WorkflowCommandMixin

__all__ = [
    "CoreCommandMixin",
    "ConfigCommandMixin",
    "PlatformCommandMixin",
    "WorkflowCommandMixin",
    "InfoCommandMixin",
    "AdminCommandMixin",
    "OtherCommandMixin",
]
