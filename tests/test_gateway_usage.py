import unittest

from aioptimizer.usage import StreamUsageAccumulator, extract_usage


class UsageExtractionTests(unittest.TestCase):
    def test_extracts_openai_anthropic_and_ollama_shapes(self):
        self.assertEqual(
            extract_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 4,
                                      "total_tokens": 14}}),
            {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )
        self.assertEqual(
            extract_usage({"usage": {"input_tokens": 8, "output_tokens": 3}}),
            {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
        )
        self.assertEqual(
            extract_usage({"prompt_eval_count": 7, "eval_count": 2}),
            {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
        )

    def test_extracts_nested_anthropic_message_start_usage(self):
        payload = {"type": "message_start", "message": {
            "usage": {"input_tokens": 12, "output_tokens": 1}}}
        self.assertEqual(
            extract_usage(payload),
            {"input_tokens": 12, "output_tokens": 1, "total_tokens": 13},
        )

    def test_stream_accumulator_handles_fragmented_sse_and_updates(self):
        accumulator = StreamUsageAccumulator()
        accumulator.feed(b'data: {"type":"message_start","message":{"usage":')
        accumulator.feed(b'{"input_tokens":12,"output_tokens":1}}}\n\n')
        accumulator.feed(b'data: {"type":"message_delta","usage":{"output_tokens":5}}\n')
        accumulator.feed(
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta",'
            b'"text":"hello"}}\n'
        )
        accumulator.feed(b'\ndata: [DONE]\n\n')
        self.assertEqual(
            accumulator.finish(),
            {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
        )
        self.assertEqual(accumulator.text(), "hello")

    def test_stream_accumulator_handles_ollama_ndjson(self):
        accumulator = StreamUsageAccumulator()
        accumulator.feed(b'{"response":"hi","done":false}\n')
        accumulator.feed(b'{"done":true,"prompt_eval_count":20,"eval_count":6}\n')
        self.assertEqual(
            accumulator.finish(),
            {"input_tokens": 20, "output_tokens": 6, "total_tokens": 26},
        )
        self.assertEqual(accumulator.text(), "hi")

    def test_missing_or_malformed_usage_is_not_invented(self):
        self.assertIsNone(extract_usage({"usage": {"prompt_tokens": "10"}}))
        accumulator = StreamUsageAccumulator()
        accumulator.feed(b'data: not-json\n\n')
        self.assertIsNone(accumulator.finish())


if __name__ == "__main__":
    unittest.main()
