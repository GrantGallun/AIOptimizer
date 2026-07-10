import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_bus.bus import Bus


class AgentBusTests(unittest.TestCase):
    def test_send_assigns_sequential_ids_and_persists(self):
        with TemporaryDirectory() as tmp:
            bus = Bus(Path(tmp))
            a = bus.send(frm="fable", to="codex", body="hello", type_="task", thread="v2.1")
            b = bus.send(frm="codex", to="fable", body="ack", type_="ack", thread="v2.1")
            self.assertEqual((a["id"], b["id"]), ("m0001", "m0002"))
            self.assertEqual(len(bus.messages()), 2)

    def test_read_new_advances_cursor(self):
        with TemporaryDirectory() as tmp:
            bus = Bus(Path(tmp))
            bus.send(frm="fable", to="codex", body="one")
            bus.send(frm="fable", to="codex", body="two")
            first = bus.read(for_agent="codex", new=True)
            self.assertEqual([m["body"] for m in first], ["one", "two"])
            # Cursor advanced: nothing new until another message arrives.
            self.assertEqual(bus.read(for_agent="codex", new=True), [])
            bus.send(frm="fable", to="codex", body="three")
            self.assertEqual([m["body"] for m in bus.read(for_agent="codex", new=True)], ["three"])

    def test_inbox_excludes_own_messages_includes_broadcast(self):
        with TemporaryDirectory() as tmp:
            bus = Bus(Path(tmp))
            bus.send(frm="fable", to="codex", body="to-codex")
            bus.send(frm="codex", to="fable", body="from-codex")
            bus.send(frm="user", to="all", body="broadcast")
            codex_inbox = [m["body"] for m in bus.read(for_agent="codex")]
            self.assertIn("to-codex", codex_inbox)
            self.assertIn("broadcast", codex_inbox)
            self.assertNotIn("from-codex", codex_inbox)  # codex does not see its own message

    def test_render_writes_channel_with_bodies(self):
        with TemporaryDirectory() as tmp:
            bus = Bus(Path(tmp))
            bus.send(frm="fable", to="codex", body="render me", thread="v2.1")
            text = bus.channel.read_text(encoding="utf-8")
            self.assertIn("render me", text)
            self.assertIn("Fable", text)
            self.assertIn("v2.1", text)


if __name__ == "__main__":
    unittest.main()
