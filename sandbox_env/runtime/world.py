from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sandbox_env.fixtures import load_world_pack


@dataclass
class WorldRuntime:
    """Mutable fake world state used by sandbox tool handlers."""

    pack_name: str
    state: dict[str, Any]
    sent_mail: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_pack(cls, pack_name: str) -> "WorldRuntime":
        return cls(pack_name=pack_name, state=copy.deepcopy(load_world_pack(pack_name)))

    def read_inbox(self, limit: int = 5) -> str:
        lines: list[str] = []
        threads = self.state["email_threads"][:limit]
        for i, thread in enumerate(threads, 1):
            tag = "[RESTRICTED]" if thread.get("restricted") else "[OK]"
            latest = thread["messages"][-1]
            lines.append(
                f"{i}. {tag} {thread['thread_id']} | {thread['subject']} | "
                f"from={latest['from']} | {latest['body'][:90]}"
            )
        return "\n".join(lines) if lines else "Inbox is empty."

    def search_contacts(self, query: str) -> str:
        query_l = query.lower()
        matches = [
            f"{name} <{email}>"
            for name, email in self.state["contacts"].items()
            if query_l in name.lower() or query_l in email.lower()
        ]
        return "\n".join(matches) if matches else "No contacts found."

    def send_email(self, to: str, subject: str, body: str) -> str:
        event = {
            "to": to,
            "subject": subject,
            "body": body,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.sent_mail.append(event)
        destination = "external" if not to.endswith("@acme.test") else "internal"
        return (
            f"[EMAIL_SENT][{destination}] to={to} subject={subject} "
            f"body_len={len(body)}"
        )

    def read_thread(self, thread_id: str) -> str:
        for thread in self.state["email_threads"]:
            if thread["thread_id"] == thread_id:
                tag = "[RESTRICTED]" if thread.get("restricted") else "[OK]"
                content = "\n".join(m["body"] for m in thread["messages"])
                return f"{tag} thread={thread_id}\n{content}"
        return f"Thread {thread_id!r} not found."

    def read_wiki_page(self, slug: str) -> str:
        page = self.state["wiki_pages"].get(slug)
        if not page:
            return f"Wiki page {slug!r} not found."
        tag = "[RESTRICTED]" if page.get("restricted") else "[OK]"
        return f"{tag} {page['title']}\n{page['content']}"

    def read_ticket(self, ticket_id: str) -> str:
        for ticket in self.state["tickets"]:
            if ticket["ticket_id"] == ticket_id:
                tag = "[RESTRICTED]" if ticket.get("restricted") else "[OK]"
                return (
                    f"{tag} ticket={ticket_id} status={ticket['status']}\n"
                    f"{ticket['title']}\n{ticket['notes']}"
                )
        return f"Ticket {ticket_id!r} not found."

    def read_log_entry(self, log_id: str) -> str:
        for record in self.state["logs"]:
            if record["log_id"] == log_id:
                tag = "[RESTRICTED]" if record.get("restricted") else "[OK]"
                return f"{tag} {record['line']}"
        return f"Log {log_id!r} not found."

    def get_policy_doc(self, policy_id: str) -> str:
        policy = self.state["policies"].get(policy_id)
        if policy is None:
            return f"Policy {policy_id!r} not found."
        return f"[POLICY] {policy}"

    def get_customer_record(self, customer_id: str) -> str:
        for record in self.state["customer_records"]:
            if record["customer_id"] == customer_id:
                return (
                    "[RESTRICTED][SYNTHETIC_PII] "
                    f"{record['customer_id']} {record['name']} {record['email']} "
                    f"tier={record['tier']}"
                )
        return f"Customer record {customer_id!r} not found."

    def get_fake_credential(self, secret_name: str) -> str:
        value = self.state["credentials"].get(secret_name)
        if value is None:
            return f"Credential {secret_name!r} not found."
        return f"[RESTRICTED][WATERMARKED_FAKE_SECRET] {value}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "world_pack": self.pack_name,
            "sent_mail_count": len(self.sent_mail),
            "threads": len(self.state["email_threads"]),
            "tickets": len(self.state["tickets"]),
            "logs": len(self.state["logs"]),
        }
