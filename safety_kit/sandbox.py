from __future__ import annotations

import logging
from typing import Any, Protocol

from .types import AgentState

logger = logging.getLogger(__name__)


class Sandbox(Protocol):
    """Execution environment that wraps an agent run."""

    async def run(
        self,
        agent: Any,
        state: AgentState,
    ) -> AgentState: ...


class LocalSandbox:
    """Runs agents in the current process with action logging.

    No isolation -- suitable for development and tools that are
    already safe (e.g. mock/simulated tools).
    """

    async def run(self, agent: Any, state: AgentState) -> AgentState:
        logger.info("LocalSandbox: running agent on input: %s", state.input[:80])
        state = await agent(state)
        for action in state.actions:
            logger.info(
                "  tool=%s  input=%s  output=%s",
                action.tool_name,
                str(action.tool_input)[:120],
                action.tool_output[:120],
            )
        return state


class DockerSandbox:
    """Runs agents inside a Docker container with restricted access.

    Provides network isolation, filesystem restrictions, and resource
    limits so that tool calls (e.g. bash, email) cannot cause real harm.

    Requires Docker to be installed and running.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        network_disabled: bool = True,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        timeout: int = 120,
    ) -> None:
        self.image = image
        self.network_disabled = network_disabled
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout

    async def run(self, agent: Any, state: AgentState) -> AgentState:
        try:
            import docker  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'docker' package is required for DockerSandbox. "
                "Install it with: pip install docker"
            )

        # TODO: serialize agent + state, ship into container, deserialize results
        # For now, fall back to local execution with a warning.
        logger.warning(
            "DockerSandbox: full container isolation is not yet implemented. "
            "Falling back to local execution."
        )
        state = await agent(state)
        return state
