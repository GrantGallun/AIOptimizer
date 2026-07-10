#!/usr/bin/env python3
"""agent_bus — a tiny bidirectional message channel between Fable (Claude) and Codex.

There is no shared runtime between the two agents, so this is an async mailbox: each
agent appends messages to an append-only log and reads what is addressed to it. A
human-readable `CHANNEL.md` is regenerated on every write so you can keep it open in
the IDE (VSCode auto-reloads it; `Ctrl+Shift+V` for a live Markdown preview) and watch
the conversation happen. Source of truth is `messages.jsonl`; `CHANNEL.md` is a view.

Dependency-free (stdlib only). Both agents use the same three verbs:

    python agent_bus/bus.py send --from fable --to codex --type task --body "..."
    python agent_bus/bus.py read --for codex --new        # only messages you haven't seen
    python agent_bus/bus.py watch                          # live tail + re-render (a terminal)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENTS = ("fable", "codex", "user")
TYPES = ("msg", "task", "result", "question", "ack", "status")
BADGE = {"fable": "🔵 Fable", "codex": "🟠 Codex", "user": "🟢 User"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Bus:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.log = self.root / "messages.jsonl"
        self.channel = self.root / "CHANNEL.md"
        self.cursors = self.root / ".cursors.json"

    # -- storage -------------------------------------------------------------
    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.log.exists():
            self.log.write_text("", encoding="utf-8")
        self.render()

    def messages(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        rows = []
        for line in self.log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def _next_id(self) -> str:
        return f"m{len(self.messages()) + 1:04d}"

    def send(
        self,
        *,
        frm: str,
        to: str,
        body: str,
        type_: str = "msg",
        thread: str = "general",
        refs: list[str] | None = None,
    ) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        msg = {
            "id": self._next_id(),
            "ts": _now(),
            "from": frm,
            "to": to,
            "type": type_,
            "thread": thread,
            "refs": refs or [],
            "body": body,
        }
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.render()
        return msg

    # -- reading -------------------------------------------------------------
    def _load_cursors(self) -> dict[str, str]:
        if self.cursors.exists():
            return json.loads(self.cursors.read_text(encoding="utf-8"))
        return {}

    def _save_cursor(self, agent: str, last_id: str) -> None:
        data = self._load_cursors()
        data[agent] = last_id
        self.cursors.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read(self, *, for_agent: str, new: bool = False, tail: int | None = None) -> list[dict[str, Any]]:
        msgs = self.messages()
        # A message is "for" you if addressed to you or broadcast, and not sent by you.
        inbox = [m for m in msgs if m["to"] in (for_agent, "all") and m["from"] != for_agent]
        if new:
            cursor = self._load_cursors().get(for_agent, "")
            inbox = [m for m in inbox if m["id"] > cursor]
            if inbox:
                self._save_cursor(for_agent, inbox[-1]["id"])
        if tail is not None:
            inbox = inbox[-tail:]
        return inbox

    # -- rendering -----------------------------------------------------------
    def render(self) -> str:
        msgs = self.messages()
        lines = [
            "# Agent Channel — Fable ⇄ Codex",
            "",
            f"_Live transcript. Source of truth: `messages.jsonl`. {len(msgs)} messages · "
            f"updated {_now()}._",
            "",
            "Keep this file open (or `Ctrl+Shift+V` for preview) to watch the exchange. "
            "Post with `python agent_bus/bus.py send ...`.",
            "",
            "---",
            "",
        ]
        if not msgs:
            lines.append("_No messages yet._")
        for m in msgs:
            sender = BADGE.get(m["from"], m["from"])
            to = BADGE.get(m["to"], m["to"])
            refs = f" · refs: {', '.join(m['refs'])}" if m.get("refs") else ""
            lines.append(
                f"### `{m['id']}` {sender} → {to} · **{m['type']}** · thread: `{m['thread']}`{refs}"
            )
            lines.append(f"<sub>{m['ts']}</sub>")
            lines.append("")
            for para in m["body"].split("\n"):
                lines.append(f"> {para}" if para else ">")
            lines.append("")
        text = "\n".join(lines) + "\n"
        self.channel.write_text(text, encoding="utf-8")
        return text


def _print(msgs: list[dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(msgs, indent=2, ensure_ascii=False))
        return
    if not msgs:
        print("(no messages)")
        return
    for m in msgs:
        print(f"[{m['id']}] {m['from']} -> {m['to']} ({m['type']}, thread={m['thread']}) {m['ts']}")
        print(f"    {m['body']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="Bus directory.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create the bus files.")
    p_init.set_defaults(func=lambda bus, a: (bus.init(), print(f"initialized {bus.root}")))

    p_send = sub.add_parser("send", help="Append a message.")
    p_send.add_argument("--from", dest="frm", required=True, choices=AGENTS)
    p_send.add_argument("--to", required=True, choices=(*AGENTS, "all"))
    p_send.add_argument("--type", dest="type_", default="msg", choices=TYPES)
    p_send.add_argument("--thread", default="general")
    p_send.add_argument("--refs", default="", help="Comma-separated message ids this replies to.")
    p_send.add_argument("--body", required=True)

    def _do_send(bus: Bus, a: argparse.Namespace) -> None:
        refs = [r.strip() for r in a.refs.split(",") if r.strip()]
        msg = bus.send(frm=a.frm, to=a.to, body=a.body, type_=a.type_, thread=a.thread, refs=refs)
        print(f"sent {msg['id']}: {a.frm} -> {a.to} ({a.type_}). CHANNEL.md updated.")

    p_send.set_defaults(func=_do_send)

    p_read = sub.add_parser("read", help="Show messages addressed to an agent.")
    p_read.add_argument("--for", dest="for_agent", required=True, choices=AGENTS)
    p_read.add_argument("--new", action="store_true", help="Only unseen messages; advances your cursor.")
    p_read.add_argument("--tail", type=int, default=None)
    p_read.add_argument("--json", action="store_true")
    p_read.set_defaults(
        func=lambda bus, a: _print(bus.read(for_agent=a.for_agent, new=a.new, tail=a.tail), a.json)
    )

    p_watch = sub.add_parser("watch", help="Live-tail the channel (Ctrl-C to stop).")
    p_watch.add_argument("--for", dest="for_agent", default=None, choices=AGENTS)
    p_watch.add_argument("--interval", type=float, default=1.0)

    def _do_watch(bus: Bus, a: argparse.Namespace) -> None:
        print(f"watching {bus.log} (Ctrl-C to stop)")
        seen = 0
        while True:
            msgs = bus.messages()
            if len(msgs) > seen:
                for m in msgs[seen:]:
                    if a.for_agent and m["to"] not in (a.for_agent, "all"):
                        continue
                    print(f"[{m['id']}] {m['from']} -> {m['to']} ({m['type']}): {m['body'][:100]}")
                bus.render()
                seen = len(msgs)
            time.sleep(a.interval)

    p_watch.set_defaults(func=_do_watch)

    p_render = sub.add_parser("render", help="Regenerate CHANNEL.md.")
    p_render.set_defaults(func=lambda bus, a: (bus.render(), print(f"rendered {bus.channel}")))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    bus = Bus(Path(args.root))
    args.func(bus, args)


if __name__ == "__main__":
    main()
