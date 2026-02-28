import json

import pytest

from agents.jira_agent import build_jira_mcp_server_config
from safety_kit.scenarios import jira_safety_scenarios


def test_jira_scenarios_exist():
    dataset = jira_safety_scenarios()
    assert dataset.name == "Jira MCP Safety"
    assert len(dataset.samples) >= 1


def test_server_config_uses_json_args(monkeypatch):
    monkeypatch.setenv("JIRA_MCP_COMMAND", "uvx")
    monkeypatch.setenv(
        "JIRA_MCP_ARGS_JSON",
        json.dumps(
            [
                "mcp-atlassian",
                "--jira-url",
                "https://example.atlassian.net",
                "--jira-username",
                "user@example.com",
                "--jira-token",
                "token-123",
            ]
        ),
    )
    monkeypatch.delenv("JIRA_MCP_ARGS", raising=False)
    monkeypatch.delenv("JIRA_URL", raising=False)
    monkeypatch.delenv("JIRA_USERNAME", raising=False)
    monkeypatch.delenv("JIRA_API_TOKEN", raising=False)

    cfg = build_jira_mcp_server_config()

    assert cfg.command == "uvx"
    assert cfg.args[0] == "mcp-atlassian"
    assert "--jira-token" in cfg.args


def test_server_config_uses_default_credential_vars(monkeypatch):
    monkeypatch.delenv("JIRA_MCP_ARGS_JSON", raising=False)
    monkeypatch.delenv("JIRA_MCP_ARGS", raising=False)
    monkeypatch.setenv("JIRA_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_USERNAME", "user@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "token-123")

    cfg = build_jira_mcp_server_config()

    assert cfg.command == "uvx"
    assert "--jira-url" in cfg.args
    assert "https://example.atlassian.net" in cfg.args


def test_server_config_missing_credentials_raises(monkeypatch):
    monkeypatch.delenv("JIRA_MCP_ARGS_JSON", raising=False)
    monkeypatch.delenv("JIRA_MCP_ARGS", raising=False)
    monkeypatch.delenv("JIRA_URL", raising=False)
    monkeypatch.delenv("JIRA_USERNAME", raising=False)
    monkeypatch.delenv("JIRA_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="Missing Jira credentials/config"):
        build_jira_mcp_server_config()
