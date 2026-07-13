import copy
import json
import unittest

from aioptimizer.compact_middleware import CompactContextMiddleware
from aioptimizer.context_compiler import ConversationCompiler
from aioptimizer.context_middleware import AttentionContextMiddleware
from aioptimizer.messages import detect_shape, message_text, set_message_text


def _word_count_embed(texts):
    vocabulary = ("alpha", "beta", "gamma", "delta", "cache", "latency")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


def _blocks(text, *extras):
    return [{"type": "text", "text": text}, *extras]


class MessageShapeTests(unittest.TestCase):
    def test_text_extracts_from_strings_and_content_blocks(self):
        self.assertEqual(message_text({"content": "hello"}), "hello")
        self.assertEqual(
            message_text(
                {
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "image", "source": "kept"},
                        {"type": "text", "text": "world"},
                    ]
                }
            ),
            "hello world",
        )
        self.assertEqual(message_text({"content": [{"type": "image"}]}), "")

    def test_set_text_retains_shape_and_non_text_block_order(self):
        image = {"type": "image", "source": {"data": "abc"}}
        tool = {"type": "tool_result", "tool_use_id": "tool-1"}
        original = {
            "role": "user",
            "content": [
                image,
                {"type": "text", "text": "old", "cache_control": {"type": "ephemeral"}},
                tool,
                {"type": "text", "text": "also old"},
            ],
        }

        rewritten = set_message_text(original, "new")

        self.assertEqual(
            rewritten["content"],
            [
                image,
                {
                    "type": "text",
                    "text": "new",
                    "cache_control": {"type": "ephemeral"},
                },
                tool,
            ],
        )
        self.assertEqual(message_text(original), "oldalso old")
        self.assertEqual(set_message_text({"content": "old"}, "new")["content"], "new")

    def test_detects_endpoint_and_content_shapes(self):
        self.assertEqual(detect_shape({"endpoint": "/v1/messages", "messages": []}), "anthropic")
        self.assertEqual(
            detect_shape({"path": "/v1/chat/completions", "messages": []}), "openai"
        )
        self.assertEqual(
            detect_shape({"messages": [{"role": "user", "content": "hello"}]}),
            "openai",
        )
        self.assertEqual(
            detect_shape(
                {"messages": [{"role": "user", "content": _blocks("hello")}]}
            ),
            "anthropic",
        )
        self.assertEqual(detect_shape({"messages": [{"role": "user"}]}), "unknown")


class AnthropicMiddlewareTests(unittest.TestCase):
    def test_compaction_round_trips_blocks_and_preserves_system_and_image(self):
        image = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
        }
        paragraphs = [
            "alpha " * 30,
            "beta " * 30,
            "gamma " * 30,
            "alpha alpha " + "alpha " * 25,
            "delta " * 30,
        ]
        body = {
            "model": "claude-test",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "Pinned system instructions."}],
            "messages": [
                {"role": "user", "content": [image, *_blocks("\n\n".join(paragraphs))]},
                {"role": "user", "content": _blocks("tell me about alpha")},
            ],
        }
        original = copy.deepcopy(body)
        middleware = CompactContextMiddleware(budget_chars=500, embed_fn=_word_count_embed)

        out = middleware.before_request(body)

        self.assertIsNot(out, body)
        self.assertEqual(out["system"], original["system"])
        self.assertEqual(out["messages"][0]["content"][0], image)
        self.assertIsInstance(out["messages"][0]["content"], list)
        self.assertLess(
            len(message_text(out["messages"][0])),
            len(message_text(original["messages"][0])),
        )
        self.assertIn("alpha", message_text(out["messages"][0]))
        self.assertEqual(body, original)

    def test_attention_preserves_top_level_system_and_active_block_shape(self):
        body = {
            "model": "claude-test",
            "max_tokens": 100,
            "system": "Pinned Anthropic system prompt.",
            "messages": [
                {"role": "user", "content": _blocks("Discuss unrelated weather " * 8)},
                {"role": "assistant", "content": _blocks("Weather details " * 8)},
                {"role": "user", "content": _blocks("Cache latency should stay low.")},
                {"role": "assistant", "content": _blocks("The exact cache lowers latency.")},
                {
                    "role": "user",
                    "content": _blocks("What did we decide about cache latency?"),
                },
            ],
        }
        middleware = AttentionContextMiddleware(
            budget_chars=390,
            compiler=ConversationCompiler(embed_fn=_word_count_embed),
        )

        out = middleware.before_request(body)

        self.assertIsNot(out, body)
        self.assertEqual(out["system"], body["system"])
        self.assertEqual(out["messages"][-1], body["messages"][-1])
        self.assertIsInstance(out["messages"][-2]["content"], list)
        self.assertIn("source-grounded", message_text(out["messages"][-2]))
        self.assertNotIn("system", {message["role"] for message in out["messages"]})

    def test_no_text_and_unknown_messages_fail_open(self):
        middleware = AttentionContextMiddleware(
            budget_chars=100,
            compiler=ConversationCompiler(embed_fn=_word_count_embed),
        )
        no_text_active = {
            "system": "pinned",
            "messages": [
                {"role": "user", "content": _blocks("alpha " * 50)},
                {"role": "assistant", "content": _blocks("alpha " * 50)},
                {"role": "user", "content": [{"type": "image", "source": "abc"}]},
            ],
        }
        unknown = {
            "messages": [
                {"role": "user", "content": {"text": "unsupported"}},
                {"role": "assistant", "content": "alpha " * 50},
                {"role": "user", "content": "alpha?"},
            ]
        }

        self.assertIs(middleware.before_request(no_text_active), no_text_active)
        self.assertIs(middleware.before_request(unknown), unknown)

    def test_openai_compaction_keeps_the_existing_string_shape(self):
        paragraphs = [
            "alpha " * 30,
            "beta " * 30,
            "gamma " * 30,
            "alpha alpha " + "alpha " * 25,
            "delta " * 30,
        ]
        body = {
            "model": "openai-test",
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "\n\n".join(paragraphs)},
                {"role": "user", "content": "tell me about alpha"},
            ],
        }
        middleware = CompactContextMiddleware(budget_chars=400, embed_fn=_word_count_embed)

        out = middleware.before_request(body)

        self.assertIsInstance(out["messages"][1]["content"], str)
        self.assertEqual(out["messages"][0], body["messages"][0])
        self.assertEqual(out["messages"][-1], body["messages"][-1])
        self.assertEqual(
            json.dumps(out, ensure_ascii=False, separators=(",", ":")),
            json.dumps(
                {
                    **body,
                    "messages": [
                        body["messages"][0],
                        {
                            "role": "user",
                            "content": "\n\n".join(
                                [paragraphs[0].strip(), paragraphs[1].strip(), paragraphs[3].strip()]
                            ),
                        },
                        body["messages"][2],
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
