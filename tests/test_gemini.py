import os
import unittest
from types import SimpleNamespace
from unittest import mock

from google.genai import errors

import shorts_maker


class ParseJsonTests(unittest.TestCase):
    def test_parses_plain_fenced_and_embedded_objects(self):
        expected = {"topic": "The Silk Road"}
        self.assertEqual(shorts_maker.parse_json('{"topic":"The Silk Road"}'), expected)
        self.assertEqual(shorts_maker.parse_json('```json\n{"topic":"The Silk Road"}\n```'), expected)
        self.assertEqual(shorts_maker.parse_json('Result: {"topic":"The Silk Road"}'), expected)

    def test_rejects_invalid_json_and_non_object_values(self):
        for response in (
            "not JSON",
            "[]",
            '[{"topic":"nested object is not a top-level response"}]',
            "",
            "```json\n{broken}\n```",
            '{"outer":{"topic":"partial"} broken}',
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                shorts_maker.parse_json(response)

    def test_validates_required_nested_visual_fields_and_count(self):
        schema = {
            "type": "object",
            "properties": {
                "visual_prompts": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "image_prompt": {"type": "string"},
                            "video_prompt": {"type": "string"},
                        },
                        "required": ["image_prompt", "video_prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["visual_prompts"],
        }
        shorts_maker.validate_json_schema(
            {"visual_prompts": [{"image_prompt": "Market", "video_prompt": "Trade"}]}, schema
        )
        with self.assertRaises(ValueError):
            shorts_maker.validate_json_schema({"visual_prompts": []}, schema)
        with self.assertRaises(ValueError):
            shorts_maker.validate_json_schema(
                {"visual_prompts": [{"image_prompt": "Market"}]}, schema
            )


class GeminiReliabilityTests(unittest.TestCase):
    def test_client_uses_bounded_exponential_retries(self):
        previous_client = shorts_maker._GEMINI_CLIENT
        shorts_maker._GEMINI_CLIENT = None
        try:
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
                 mock.patch.object(shorts_maker.genai, "Client") as client_factory:
                shorts_maker.gemini_client()
            retry_options = client_factory.call_args.kwargs["http_options"].retry_options
            self.assertEqual(retry_options.attempts, 3)
            self.assertEqual(retry_options.initial_delay, 1.0)
            self.assertEqual(retry_options.max_delay, 4.0)
            self.assertEqual(retry_options.exp_base, 2.0)
            self.assertIn(503, retry_options.http_status_codes)
        finally:
            shorts_maker._GEMINI_CLIENT = previous_client

    def test_503_uses_the_other_model_and_structured_schema(self):
        api = mock.Mock()
        api.generate_content.side_effect = [
            errors.ServerError(503, {"error": {"message": "temporary overload"}}),
            SimpleNamespace(text='{"topic":"Silk Road"}'),
        ]
        schema = {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        }
        with mock.patch.dict(os.environ, {
            "GEMINI_MODEL": "flash-primary",
            "GEMINI_LITE_MODEL": "lite-fallback",
        }), mock.patch.object(shorts_maker, "gemini_client", return_value=SimpleNamespace(models=api)), \
             self.assertLogs(shorts_maker.LOGGER, level="WARNING"):
            result = shorts_maker.gemini_generate(
                "topic prompt", model_role="flash", response_schema=schema
            )

        self.assertEqual(shorts_maker.parse_json(result), {"topic": "Silk Road"})
        self.assertEqual(api.generate_content.call_args_list[0].kwargs["model"], "flash-primary")
        self.assertEqual(api.generate_content.call_args_list[1].kwargs["model"], "lite-fallback")
        self.assertEqual(api.generate_content.call_args_list[0].kwargs["config"].response_json_schema, schema)

    def test_invalid_structured_response_uses_fallback(self):
        api = mock.Mock()
        api.generate_content.side_effect = [
            SimpleNamespace(text="model prose without JSON"),
            SimpleNamespace(text='{"topic":"Silk Road"}'),
        ]
        with mock.patch.dict(os.environ, {
            "GEMINI_MODEL": "flash-primary",
            "GEMINI_LITE_MODEL": "lite-fallback",
        }), mock.patch.object(shorts_maker, "gemini_client", return_value=SimpleNamespace(models=api)), \
             self.assertLogs(shorts_maker.LOGGER, level="WARNING"):
            result = shorts_maker.gemini_generate("topic prompt", model_role="flash")

        self.assertEqual(shorts_maker.parse_json(result), {"topic": "Silk Road"})
        self.assertEqual(api.generate_content.call_count, 2)

    def test_schema_violation_uses_fallback(self):
        api = mock.Mock()
        api.generate_content.side_effect = [
            SimpleNamespace(text='{"unexpected":"wrong shape"}'),
            SimpleNamespace(text='{"topic":"Silk Road"}'),
        ]
        schema = {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        }
        with mock.patch.dict(os.environ, {
            "GEMINI_MODEL": "flash-primary",
            "GEMINI_LITE_MODEL": "lite-fallback",
        }), mock.patch.object(shorts_maker, "gemini_client", return_value=SimpleNamespace(models=api)), \
             self.assertLogs(shorts_maker.LOGGER, level="WARNING"):
            result = shorts_maker.gemini_generate(
                "topic prompt", model_role="flash", response_schema=schema
            )

        self.assertEqual(shorts_maker.parse_json(result), {"topic": "Silk Road"})
        self.assertEqual(api.generate_content.call_count, 2)

    def test_story_calls_lite_for_topic_and_flash_for_narration(self):
        outputs = [
            '{"topic":"Silk Road","search_query":"Silk Road history"}',
            '{"topic":"Silk Road","title":"The Silk Road","script":"A short history.",'
            '"description":"A historical explainer.","tags":["history"]}',
        ]
        with mock.patch.object(shorts_maker, "gemini_generate", side_effect=outputs) as generate, \
             mock.patch.object(shorts_maker, "wikipedia_research", return_value=([{"title": "Silk Road", "excerpt": "Facts."}], ["source"])):
            story, sources = shorts_maker.make_story("historical and informative", [])

        self.assertEqual(story["topic"], "Silk Road")
        self.assertEqual(sources, ["source"])
        self.assertEqual([call.kwargs["model_role"] for call in generate.call_args_list], ["lite", "flash"])
        self.assertTrue(all(call.kwargs.get("response_schema") for call in generate.call_args_list))

    def test_visual_prompts_and_duration_rewrite_use_flash(self):
        with mock.patch.object(
            shorts_maker,
            "gemini_generate",
            side_effect=[
                '{"visual_prompts":[{"image_prompt":"A market","video_prompt":"People trade goods"}]}',
                "A concise narration.",
            ],
        ) as generate:
            prompts = shorts_maker.make_visual_prompts("Silk Road", "Narration", 1)
            rewritten = shorts_maker.rewrite_story_for_duration("Narration", "Silk Road", 120)

        self.assertEqual(len(prompts), 1)
        self.assertEqual(rewritten, "A concise narration.")
        self.assertEqual([call.kwargs["model_role"] for call in generate.call_args_list], ["flash", "flash"])
        self.assertTrue(generate.call_args_list[0].kwargs.get("response_schema"))
        self.assertFalse(generate.call_args_list[1].kwargs["json_output"])


if __name__ == "__main__":
    unittest.main()
