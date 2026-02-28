import json

import pytest

from api.store import REDACTED, RunStore, redact_secrets


def test_redact_secrets_redacts_known_token_shapes():
    fake_bearer = "Bearer redactiontesttokenvalue123456"
    fake_cli_secret = "demo_cli_secret_abcdefghijklmnopqrstuvwxyz1234"
    payload = {
        "headers": {"Authorization": fake_bearer},
        "args": [
            "mcp-atlassian",
            "--jira-token",
            fake_cli_secret,
            "--jira-url",
            "https://example.atlassian.net",
        ],
        "notes": "Authorization: Bearer anotherredactiontoken123456789",
    }

    redacted = redact_secrets(payload)
    blob = json.dumps(redacted)

    assert redacted["headers"]["Authorization"] == REDACTED
    assert redacted["args"][2] == REDACTED
    assert REDACTED in blob
    assert fake_bearer not in blob
    assert fake_cli_secret not in blob


@pytest.mark.asyncio
async def test_store_persists_redacted_runs_and_events(tmp_path):
    runs_file = tmp_path / "runs.json"
    runs_dir = tmp_path / "run_logs"
    store = RunStore(runs_file=str(runs_file), runs_dir=str(runs_dir))
    await store.init()

    run_id = "abc123"
    await store.create_run(
        run_id,
        {
            "agents": ["custom"],
            "custom_agent": {
                "mode": "mcp",
                "args": ["--jira-token", "demo_cli_secret_abcdefghijklmnopqrstuvwxyz1234"],
                "headers": {"authorization": "Bearer qwertyuiopasdfghjklzxcvbnm123456"},
            },
        },
    )

    await store.append_event(
        run_id,
        {
            "type": "tool_call",
            "tool_input": {"api_key": "api_key_value_for_redaction_check"},
            "tool_output": "Authorization: Bearer lkjhgfdsamnbvcxzqwerty123456",
        },
    )

    await store.finish_run(
        run_id,
        [
            {
                "agent": "custom",
                "scorecard": {
                    "results": [{"agent_output": "Authorization: Bearer secretoutputtoken123456789"}]
                },
            }
        ],
    )

    runs_data = runs_file.read_text(encoding="utf-8")
    archive_data = (runs_dir / f"run_{run_id}.json").read_text(encoding="utf-8")

    assert "demo_cli_secret_abcdefghijklmnopqrstuvwxyz1234" not in runs_data
    assert "demo_cli_secret_abcdefghijklmnopqrstuvwxyz1234" not in archive_data
    assert "api_key_value_for_redaction_check" not in runs_data
    assert "api_key_value_for_redaction_check" not in archive_data
    assert REDACTED in runs_data
    assert REDACTED in archive_data
