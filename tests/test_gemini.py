import os
import io
import json
import tempfile
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
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

    def test_fact_checker_returns_only_unsupported_claims(self):
        with mock.patch.object(
            shorts_maker,
            "gemini_generate",
            return_value='{"unsupported_claims":["The king signed the treaty in 1420."]}',
        ) as generate:
            claims = shorts_maker.unsupported_story_claims("The king signed the treaty in 1420.", [{"excerpt": "No date."}])

        self.assertEqual(claims, ["The king signed the treaty in 1420."])
        self.assertEqual(generate.call_args.kwargs["model_role"], "flash")
        self.assertIn("Reference excerpts", generate.call_args.args[0])


class ResilienceTests(unittest.TestCase):
    def test_wikimedia_fallback_downloads_image_and_license_metadata_offline(self):
        image_payload = b"\xff\xd8\xfftest-jpeg"
        api_payload = {
            "query": {
                "pages": {
                    "1": {
                        "title": "File:History.jpg",
                        "imageinfo": [{
                            "mime": "image/jpeg",
                            "thumbmime": "image/jpeg",
                            "thumburl": "https://upload.wikimedia.org/history.jpg",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:History.jpg",
                            "extmetadata": {
                                "Artist": {"value": "<a>Ada Example</a>"},
                                "LicenseShortName": {"value": "CC BY 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
                            },
                        }],
                    },
                },
            },
        }
        replies = [
            nullcontext(io.BytesIO(json.dumps(api_payload).encode("utf-8"))),
            nullcontext(io.BytesIO(image_payload)),
        ]
        with tempfile.TemporaryDirectory() as temporary_dir, mock.patch.object(
            shorts_maker.urllib.request, "urlopen", side_effect=replies
        ):
            image, license_info = shorts_maker.download_commons_image("history scene", Path(temporary_dir) / "scene-commons")

            self.assertEqual(image.suffix, ".jpg")
            self.assertEqual(image.read_bytes(), image_payload)
            self.assertEqual(license_info["author"], "Ada Example")
            self.assertEqual(license_info["license"], "CC BY 4.0")
            self.assertTrue(image.with_suffix(".license.json").is_file())

    def test_only_allows_reusable_commons_licenses(self):
        for license_name in ("CC BY-NC 4.0", "CC BY-SA 4.0", "CC BY-ND 4.0"):
            with self.subTest(license_name=license_name):
                self.assertFalse(shorts_maker.reusable_commons_license(license_name))
        self.assertTrue(shorts_maker.reusable_commons_license("CC BY 4.0"))
        self.assertTrue(shorts_maker.reusable_commons_license("Public domain"))

    def test_hugging_face_failures_use_commons_still_and_ffmpeg_zoom_pan(self):
        license_info = {
            "title": "History image",
            "author": "Ada Example",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_url": "https://commons.wikimedia.org/wiki/File:History.jpg",
        }

        def commons_fallback(_query, target_stem):
            image = target_stem.with_suffix(".jpg")
            image.write_bytes(b"\xff\xd8\xfftest-jpeg")
            image.with_suffix(".license.json").write_text(json.dumps(license_info), encoding="utf-8")
            return image, license_info

        def fake_ffmpeg(command, check):
            Path(command[-1]).write_bytes(b"fallback video")

        with tempfile.TemporaryDirectory() as temporary_dir:
            with mock.patch.object(shorts_maker, "OUT", Path(temporary_dir)), \
                 mock.patch.object(shorts_maker, "hf_generate_image", side_effect=RuntimeError("quota")), \
                 mock.patch.object(shorts_maker, "download_commons_image", side_effect=commons_fallback), \
                 mock.patch.object(shorts_maker, "hf_generate_video", side_effect=TimeoutError("queued")), \
                 mock.patch.object(shorts_maker.subprocess, "run", side_effect=fake_ffmpeg) as ffmpeg, \
                 self.assertLogs(shorts_maker.LOGGER, level="WARNING"):
                clips, licenses = shorts_maker.create_video_clips(
                    [{"image_prompt": "A historical market", "video_prompt": "People trade"}],
                    "0001",
                    "Silk Road",
                )
                self.assertTrue(clips[0].is_file())
                self.assertEqual(licenses, [license_info])
                ffmpeg_args = ffmpeg.call_args.args[0]
                self.assertIn("zoompan=", ffmpeg_args[ffmpeg_args.index("-vf") + 1])
                self.assertEqual(ffmpeg_args[ffmpeg_args.index("-t") + 1], "15")


class SchedulingAndReviewTests(unittest.TestCase):
    def test_slot_key_uses_india_time_and_schedule_slot(self):
        utc_time = datetime(2026, 11, 15, 13, 30, tzinfo=timezone.utc)
        with mock.patch.dict(os.environ, {"SCHEDULE_SLOT": "19:00"}):
            self.assertEqual(shorts_maker.upload_slot_key(utc_time), "2026-11-15T19:00")
        with mock.patch.dict(os.environ, {"SCHEDULE_SLOT": ""}):
            local_time = datetime(2026, 11, 15, 10, 0, tzinfo=shorts_maker.INDIA_TIME)
            self.assertEqual(shorts_maker.upload_slot_key(local_time), "2026-11-15T09:00")

    def test_main_skips_a_slot_already_recorded_in_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir, mock.patch.dict(os.environ, {"SCHEDULE_SLOT": "09:00"}):
            slot = shorts_maker.upload_slot_key()
            root = Path(temporary_dir)
            (root / "state.json").write_text(json.dumps({"uploaded_slots": {slot: "existing-id"}}), encoding="utf-8")
            with mock.patch.object(shorts_maker, "ROOT", root), \
                 mock.patch.object(shorts_maker, "load_env_file"), \
                 mock.patch.object(shorts_maker, "youtube_client") as youtube_client, \
                 mock.patch.object(shorts_maker.argparse.ArgumentParser, "parse_args", return_value=SimpleNamespace(auth_only=False)), \
                 mock.patch("sys.argv", ["shorts_maker.py"]), \
                 mock.patch("builtins.print") as print_output:
                shorts_maker.main()

        youtube_client.assert_not_called()
        self.assertIn("already records an upload", print_output.call_args.args[0])

    def test_main_rejects_unsupported_script_and_retries_one_new_topic(self):
        first_story = {
            "topic": "Rejected event", "title": "Rejected", "script": "Unsupported claim.",
            "description": "Test.", "tags": [], "research_excerpts": [{"excerpt": "No support."}],
        }
        second_story = {
            "topic": "Supported event", "title": "Supported", "script": "Supported claim.",
            "description": "Test.", "tags": [], "research_excerpts": [{"excerpt": "Supported claim."}],
        }

        def create_audio(_script, target):
            target.write_bytes(b"audio")

        prompts = [{"image_prompt": "Scene", "video_prompt": "Action"} for _ in range(4)]
        seen_histories = []

        def fake_make_story(_genre, history):
            seen_histories.append(list(history))
            return (first_story, []) if len(seen_histories) == 1 else (second_story, [])

        with tempfile.TemporaryDirectory() as temporary_dir, mock.patch.dict(os.environ, {
            "HF_TOKEN": "offline-test-token",
            "SCHEDULE_SLOT": "09:00",
            "MIN_VIDEO_SECONDS": "50",
            "MAX_VIDEO_SECONDS": "60",
        }):
            root = Path(temporary_dir) / "repo"
            output = root / "output"
            root.mkdir()
            (root / "state.json").write_text(json.dumps({"next_video": 0, "topics": []}), encoding="utf-8")
            with mock.patch.object(shorts_maker, "ROOT", root), \
                 mock.patch.object(shorts_maker, "OUT", output), \
                 mock.patch.object(shorts_maker, "load_env_file"), \
                 mock.patch.object(shorts_maker, "youtube_client"), \
                 mock.patch.object(shorts_maker.argparse.ArgumentParser, "parse_args", return_value=SimpleNamespace(auth_only=False)), \
                 mock.patch("sys.argv", ["shorts_maker.py"]), \
                 mock.patch.object(shorts_maker, "make_story", side_effect=fake_make_story) as make_story, \
                 mock.patch.object(shorts_maker, "create_narration", side_effect=create_audio), \
                 mock.patch.object(shorts_maker, "seconds_of_media", return_value=55.0), \
                 mock.patch.object(shorts_maker, "unsupported_story_claims", side_effect=[["unsupported"], []]), \
                 mock.patch.object(shorts_maker, "make_visual_prompts", return_value=prompts), \
                 mock.patch.object(shorts_maker, "create_video_clips", return_value=([output / f"clip{i}.mp4" for i in range(4)], [])), \
                 mock.patch.object(shorts_maker, "render"), \
                 mock.patch.object(shorts_maker, "upload", return_value="video-id"), \
                 self.assertLogs(shorts_maker.LOGGER, level="WARNING"):
                shorts_maker.main()

            self.assertEqual(make_story.call_count, 2)
            self.assertEqual(seen_histories, [[], ["Rejected event"]])
            saved_state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_state["topics"], ["Supported event"])
            self.assertTrue(next(iter(saved_state["uploaded_slots"])).endswith("T09:00"))


if __name__ == "__main__":
    unittest.main()
