import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from google.genai import errors
import shorts_maker


def story_fixture():
    idea = {
        "setting": "ऑफिस की लिफ्ट",
        "conflict": "कप कुर्सी से किराया मांगता है",
        "punchline": "फ्रिज ऑफिस निगल जाता है",
        "hook_type": "absurd flex",
        "hook_line": "मैंने पूरा ऑफिस खरीद लिया!",
        "catchphrase": "क्या सेटिंग है",
    }
    lines = [
        "मैंने पूरा ऑफिस खरीद लिया!", "अबे कौन मालिक बन गया भला?",
        "क्या सेटिंग है, मेरी कुर्सी!", "पहले किराया पूरा भर, फिर बोल!",
        "क्या सेटिंग है, ऑफिस किराए पर!", "कप ने सारी मेज अभी खरीद ली!",
        "अबे कुर्सी मेरी नौकरी भी लेगी!", "क्या सेटिंग है, HR को बुलाओ!",
        "लैपटॉप ने सबका वेतन अभी मांगा!", "ये तो पंखे का नया विभाग है!",
        "क्या सेटिंग है, अब पंखा CEO बना!", "आलू ने कुर्सी को मैनेजर अभी बनाया!",
        "फ्रिज ने पूरा ऑफिस अभी निगल लिया!", "अबे सच में?!",
    ]
    scenes = []
    for index in range(7):
        beats = []
        for offset, line in enumerate(lines[index * 2:index * 2 + 2]):
            beat_index = index * 2 + offset
            beat_type = "hook" if beat_index == 0 else "twist" if beat_index == 12 else "reaction" if beat_index == 13 else "normal"
            beats.append({
                "speaker": ("कप", "कुर्सी")[offset],
                "line": line,
                "caption_text": line,
                "emphasis": [line.split()[0].strip("!?.,")],
                "beat_type": beat_type,
            })
        scenes.append({
            "characters": ["कप", "कुर्सी"],
            "beats": beats,
            "sfx_hint": "record-scratch" if index == 1 else "cartoon bonk" if index == 3 else "bass drop" if index == 5 else "",
            "meme_text": "POV: ऑफिस सोमवार" if index == 0 else "",
            "image_prompt": "A fictional cup and chair argue in a bright office, cinematic comedy.",
            "video_prompt": "The cup and chair gesture wildly as the camera pushes in, no text.",
        })
    data = {
        "title": idea["hook_line"],
        "description": "तेज़ हिंदी ऑब्जेक्ट कॉमेडी।",
        "tags": ["hindi comedy", "brainrot"],
        "duration_seconds": 35,
        "toolkit_used": ["over-the-top reactions", "repeated catchphrase", "role reversal"],
        "scenes": scenes,
    }
    return shorts_maker.candidate_script(data, idea, idea["hook_type"])


def writer_payload(story):
    return {
        key: value for key, value in story.items()
        if key not in ("idea", "hook_type", "hook_line", "catchphrase", "script")
    }


class GeminiReliabilityTests(unittest.TestCase):
    def test_client_uses_bounded_retries(self):
        previous = shorts_maker._GEMINI_CLIENT
        shorts_maker._GEMINI_CLIENT = None
        try:
            with (
                mock.patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key"}),
                mock.patch.object(shorts_maker.genai, "Client") as factory,
            ):
                shorts_maker.gemini_client()
            options = factory.call_args.kwargs["http_options"].retry_options
            self.assertEqual((options.attempts, options.initial_delay, options.max_delay), (3, 1.0, 4.0))
            self.assertIn(503, options.http_status_codes)
        finally:
            shorts_maker._GEMINI_CLIENT = previous

    def test_503_uses_fallback_model_with_schema(self):
        api = mock.Mock()
        api.generate_content.side_effect = [
            errors.ServerError(503, {"error": {"message": "overloaded"}}),
            SimpleNamespace(text='{"topic":"Silk Road"}'),
        ]
        schema = {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}
        with (
            mock.patch.dict(os.environ, {"GEMINI_MODEL": "flash", "GEMINI_LITE_MODEL": "lite"}),
            mock.patch.object(shorts_maker, "gemini_client", return_value=SimpleNamespace(models=api)),
            self.assertLogs(shorts_maker.LOGGER, level="WARNING"),
        ):
            result = shorts_maker.gemini_generate("offline", model_role="flash", response_schema=schema)
        self.assertEqual(shorts_maker.parse_json(result), {"topic": "Silk Road"})
        self.assertEqual(api.generate_content.call_args_list[1].kwargs["model"], "lite")


class BrainrotQualityTests(unittest.TestCase):
    def setUp(self):
        self.story = story_fixture()

    def test_valid_story_passes_quality_gate(self):
        self.assertEqual(shorts_maker.validate_brainrot_story(self.story, 30, 50, []), [])

    def test_local_fallback_passes_each_hook_and_history(self):
        for hook_type in shorts_maker.HOOK_TYPES:
            story = shorts_maker.fallback_story(hook_type, [])
            self.assertEqual(shorts_maker.validate_brainrot_story(story, 30, 50, []), [])
            history = [{"idea": story["idea"]}]
            next_story = shorts_maker.fallback_story(hook_type, history)
            self.assertEqual(shorts_maker.validate_brainrot_story(next_story, 30, 50, history), [])

    def test_rejects_narrator_intro_and_invalid_hook(self):
        beat = self.story["scenes"][0]["beats"][0]
        beat["line"] = "एक बार की बात है ऑफिस खुला!"
        beat["caption_text"] = beat["line"]
        result = shorts_maker.validate_brainrot_story(self.story, 30, 50, [])
        self.assertTrue(any("narrator-style" in error for error in result))
        self.assertTrue(any("first line" in error for error in result))

    def test_first_line_must_match_its_hook_category(self):
        self.story["hook_type"] = "instant conflict"
        result = shorts_maker.validate_brainrot_story(self.story, 30, 50, [])
        self.assertTrue(any("first line is not a valid short hook" in error for error in result))

    def test_rejects_line_over_seven_words(self):
        beat = self.story["scenes"][0]["beats"][1]
        beat["line"] = "अबे ये मेरी कुर्सी ऑफिस अभी तुरंत वापस करो!"
        beat["caption_text"] = beat["line"]
        self.assertTrue(any(
            "over seven words" in error
            for error in shorts_maker.validate_brainrot_story(self.story, 30, 50, [])
        ))

    def test_checks_catchphrase_beat_density_duration_and_duplicate_ideas(self):
        for scene in self.story["scenes"]:
            for beat in scene["beats"]:
                beat["line"] = beat["line"].replace("क्या सेटिंग है", "क्या बात है")
                beat["caption_text"] = beat["line"]
        result = shorts_maker.validate_brainrot_story(self.story, 30, 50, [])
        self.assertTrue(any("catchphrase" in error for error in result))

        self.story = story_fixture()
        self.story["duration_seconds"] = 50
        result = shorts_maker.validate_brainrot_story(self.story, 30, 50, [])
        self.assertTrue(any("not enough beats" in error for error in result))
        self.assertTrue(any("scene count" in error for error in result))
        self.story["duration_seconds"] = 55
        result = shorts_maker.validate_brainrot_story(self.story, 30, 50, [])
        self.assertTrue(any("declared duration" in error for error in result))

        old_idea = {"idea": dict(self.story["idea"])}
        self.assertTrue(shorts_maker.is_duplicate_idea(self.story["idea"], [old_idea]))
        result = shorts_maker.validate_brainrot_story(self.story, 30, 50, [old_idea])
        self.assertTrue(any("similar to the saved idea history" in error for error in result))

    def test_writer_regenerates_invalid_script_once(self):
        data = writer_payload(self.story)
        bad = json.loads(json.dumps(data, ensure_ascii=False))
        bad["scenes"][0]["beats"][0]["line"] = "एक बार की बात है ऑफिस खुला!"
        bad["scenes"][0]["beats"][0]["caption_text"] = bad["scenes"][0]["beats"][0]["line"]
        second_idea = dict(self.story["idea"])
        second_idea.update({
            "setting": "बैंक की छत",
            "conflict": "पंखा कुर्सी से वेतन मांगता है",
            "punchline": "मोबाइल मैनेजर बन जाता है",
        })
        responses = [
            json.dumps(self.story["idea"], ensure_ascii=False), json.dumps(bad, ensure_ascii=False),
            json.dumps(second_idea, ensure_ascii=False), json.dumps(data, ensure_ascii=False),
        ]
        with (
            mock.patch.object(shorts_maker, "gemini_generate", side_effect=responses) as generate,
            self.assertLogs(shorts_maker.LOGGER, level="WARNING"),
        ):
            result = shorts_maker.make_story([], 30, 50, "absurd flex", attempts=2, punch_up=False)
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(result["title"], self.story["title"])
        self.assertEqual(generate.call_args_list[0].kwargs["model_role"], "lite")
        self.assertEqual(generate.call_args_list[1].kwargs["model_role"], "flash")

    def test_quality_gate_stops_after_three_writer_attempts(self):
        story = story_fixture()
        bad = writer_payload(story)
        bad["scenes"][0]["beats"][0]["line"] = "एक बार की बात है ऑफिस खुला!"
        bad["scenes"][0]["beats"][0]["caption_text"] = bad["scenes"][0]["beats"][0]["line"]
        ideas = []
        for index in range(3):
            idea = dict(story["idea"])
            idea.update({
                "setting": f"अलग जगह {index}",
                "conflict": f"नया झगड़ा {index}",
                "punchline": f"अलग बेवकूफ मोड़ {index}",
            })
            ideas.append(idea)
        responses = []
        for idea in ideas:
            responses.extend((json.dumps(idea, ensure_ascii=False), json.dumps(bad, ensure_ascii=False)))
        with (
            mock.patch.object(shorts_maker, "gemini_generate", side_effect=responses) as generate,
            self.assertLogs(shorts_maker.LOGGER, level="WARNING"),
            self.assertRaises(shorts_maker.BrainrotRejected),
        ):
            shorts_maker.make_story([], 30, 50, "absurd flex", attempts=3, punch_up=False)
        self.assertEqual(generate.call_count, 6)

    @unittest.skipUnless(shutil.which(os.getenv("FFMPEG", "ffmpeg")), "FFmpeg is required to render a caption frame")
    def test_caption_frame_render_is_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subtitle, frame = root / "captions.ass", root / "frame.png"
            shorts_maker.write_caption_file(self.story, 35.0, subtitle, 2.5)
            subtitle_arg = subtitle.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
            shorts_maker.subprocess.run(
                [
                    os.getenv("FFMPEG", "ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=540x960:d=1", "-vf", f"subtitles='{subtitle_arg}'",
                    "-frames:v", "1", str(frame),
                ],
                check=True,
            )
            self.assertTrue(frame.is_file())
            self.assertGreater(frame.stat().st_size, 1000)


class PipelineSafetyTests(unittest.TestCase):
    def test_slot_key_uses_india_time(self):
        now = datetime(2026, 11, 15, 13, 30, tzinfo=timezone.utc)
        with mock.patch.dict(os.environ, {"SCHEDULE_SLOT": "19:00"}):
            self.assertEqual(shorts_maker.upload_slot_key(now), "2026-11-15T19:00")

    def test_uploaded_slot_skips_before_youtube_auth(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"SCHEDULE_SLOT": "09:00"}):
            root = Path(directory)
            out = root / "output"
            out.mkdir()
            key = shorts_maker.upload_slot_key()
            (root / "state.json").write_text(json.dumps({"uploaded_slots": {key: "id"}}), encoding="utf-8")
            with (
                mock.patch.object(shorts_maker, "ROOT", root),
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.object(shorts_maker, "load_env_file"),
                mock.patch.object(shorts_maker, "youtube_client") as youtube,
                mock.patch.object(shorts_maker.argparse.ArgumentParser, "parse_args", return_value=SimpleNamespace(auth_only=False, script_only=False)),
                mock.patch("sys.argv", ["shorts_maker.py"]),
                mock.patch("builtins.print") as output,
            ):
                shorts_maker.main()
        youtube.assert_not_called()
        self.assertIn("already records an upload", output.call_args.args[0])

    def test_script_only_does_not_call_external_media_or_youtube(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "output"
            story = story_fixture()
            responses = [
                json.dumps(story["idea"], ensure_ascii=False),
                json.dumps(writer_payload(story), ensure_ascii=False),
            ]
            with (
                mock.patch.object(shorts_maker, "ROOT", root),
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.object(shorts_maker, "load_env_file"),
                mock.patch.object(shorts_maker, "gemini_generate", side_effect=responses) as generate,
                mock.patch.dict(os.environ, {"BRAINROT_PUNCH_UP": "false"}),
                mock.patch.object(shorts_maker, "youtube_client") as youtube,
                mock.patch.object(shorts_maker, "create_narration") as tts,
                mock.patch.object(shorts_maker, "hf_generate_image") as hf_image,
                mock.patch.object(shorts_maker, "hf_generate_video") as hf_video,
                mock.patch.object(shorts_maker, "render") as render,
                mock.patch.object(shorts_maker, "upload") as upload,
                mock.patch.object(shorts_maker.argparse.ArgumentParser, "parse_args", return_value=SimpleNamespace(auth_only=False, script_only=True)),
                mock.patch("sys.argv", ["shorts_maker.py", "--script-only"]),
                mock.patch("builtins.print"),
            ):
                shorts_maker.main()
            self.assertEqual(generate.call_count, 2)
            self.assertEqual([call.kwargs["model_role"] for call in generate.call_args_list], ["lite", "flash"])
            for external in (youtube, tts, hf_image, hf_video, render, upload):
                external.assert_not_called()
            saved_files = list(out.glob("script-only-*.json"))
            self.assertEqual(len(saved_files), 1)
            self.assertEqual(len(json.loads(saved_files[0].read_text(encoding="utf-8"))["scenes"]), 7)

    def test_video_space_failure_uses_five_second_pan_zoom_still(self):
        png = b"\x89PNG\r\n\x1a\n" + b"fake"
        def save_image(_result, target, _space, _kind):
            target.write_bytes(png)
        def pan_zoom(_image, target, duration=5.0):
            target.write_bytes(b"fallback")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "output"
            out.mkdir()
            with (
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.object(shorts_maker, "hf_generate_image", return_value=["image"]),
                mock.patch.object(shorts_maker, "save_hf_file", side_effect=save_image),
                mock.patch.object(shorts_maker, "hf_generate_video", side_effect=TimeoutError("queue")),
                mock.patch.object(shorts_maker, "create_pan_zoom_clip", side_effect=pan_zoom) as fallback,
                self.assertLogs(shorts_maker.LOGGER, level="WARNING"),
            ):
                clips = shorts_maker.create_video_clips(
                    [{"image_prompt": "A talking cup", "video_prompt": "The cup spins"}], "0001",
                )
                self.assertTrue(clips[0].is_file())
                self.assertEqual(fallback.call_args.kwargs.get("duration", 5.0), 5.0)

    def test_pan_zoom_ffmpeg_command_uses_five_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            image, target = Path(directory) / "still.png", Path(directory) / "scene.mp4"
            image.write_bytes(b"image")
            with mock.patch.object(shorts_maker.subprocess, "run") as run:
                shorts_maker.create_pan_zoom_clip(image, target)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-t") + 1], "5.0")
            self.assertIn("zoompan=", command[command.index("-vf") + 1])


if __name__ == "__main__":
    unittest.main()
