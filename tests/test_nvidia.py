import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
        "title": story["title"],
        "description": story["description"],
        "tags": story["tags"],
        "duration_seconds": story["duration_seconds"],
        "toolkit_used": story["toolkit_used"],
        "scenes": [
            {
                "characters": scene["characters"],
                "beats": [
                    {key: beat[key] for key in ("speaker", "line", "beat_type")}
                    for beat in scene["beats"]
                ],
                "sfx_hint": scene["sfx_hint"],
                "meme_text": scene["meme_text"],
            }
            for scene in story["scenes"]
        ],
    }


class NVIDIAReliabilityTests(unittest.TestCase):
    def test_pending_request_is_polled_until_nvidia_returns_completion(self):
        pending = mock.MagicMock()
        pending.status = 202
        pending.headers = {"NVCF-REQID": "request-123"}
        pending.read.return_value = b"{}"
        pending.__enter__.return_value = pending
        completed = mock.MagicMock()
        completed.status = 200
        completed.read.return_value = b'{"choices":[{"message":{"content":"done"}}]}'
        completed.__enter__.return_value = completed
        opener = mock.MagicMock()
        opener.open.side_effect = [pending, completed]

        with (
            mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "offline-test-key"}),
            mock.patch.object(shorts_maker.urllib.request, "build_opener", return_value=opener),
        ):
            result = shorts_maker.nvidia_chat_request("model", "offline", 20, False, None)

        poll_request = opener.open.call_args_list[1].args[0]
        self.assertEqual(poll_request.full_url, "https://integrate.api.nvidia.com/v1/status/request-123")
        self.assertEqual(poll_request.get_header("Authorization"), "Bearer offline-test-key")
        self.assertEqual(poll_request.get_header("Nvcf-poll-seconds"), "10")
        self.assertEqual(result["choices"][0]["message"]["content"], "done")

    def test_request_uses_nvidia_key_and_bolt_json_schema(self):
        schema = {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}
        opener = mock.MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = b'{"choices":[{"message":{"content":"{\\"topic\\":\\"Silk Road\\"}"}}]}'
        with (
            mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "offline-test-key"}),
            mock.patch.object(shorts_maker.urllib.request, "build_opener", return_value=opener),
        ):
            result = shorts_maker.nvidia_chat_request(
                "nvidia/nemotron-3.5-bolt-e4b", "offline", 50, True, schema,
            )
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://integrate.api.nvidia.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer offline-test-key")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "nvidia/nemotron-3.5-bolt-e4b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], schema)
        self.assertIn(json.dumps(schema), payload["messages"][0]["content"])
        self.assertEqual(result["choices"][0]["message"]["content"], '{"topic":"Silk Road"}')

    def test_503_gets_bounded_retries_then_falls_back_to_model(self):
        schema = {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}
        success = {"choices": [{"message": {"content": '{"topic":"Silk Road"}'}}]}
        with (
            mock.patch.dict(os.environ, {"NVIDIA_MODEL": "flash", "NVIDIA_LITE_MODEL": "lite"}),
            mock.patch.object(shorts_maker, "nvidia_chat_request", side_effect=[
                shorts_maker.NVIDIAAPIError("overloaded", 503),
                shorts_maker.NVIDIAAPIError("overloaded", 503),
                shorts_maker.NVIDIAAPIError("overloaded", 503),
                success,
            ]) as request,
            mock.patch.object(shorts_maker.time, "sleep") as sleep,
            self.assertLogs(shorts_maker.LOGGER, level="WARNING"),
        ):
            result = shorts_maker.nvidia_generate("offline", model_role="flash", response_schema=schema)
        self.assertEqual(shorts_maker.parse_json(result), {"topic": "Silk Road"})
        self.assertEqual([call.args[0] for call in request.call_args_list], ["flash", "flash", "flash", "lite"])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])


class BrainrotQualityTests(unittest.TestCase):
    def setUp(self):
        self.story = story_fixture()

    def test_valid_story_passes_quality_gate(self):
        self.assertEqual(shorts_maker.validate_brainrot_story(self.story, 30, 50, []), [])

    def test_hindi_gate_rejects_english_ideas_and_dialogue(self):
        idea = dict(self.story["idea"])
        idea["setting"] = "Office elevator"
        errors = shorts_maker.validate_brainrot_idea(idea, "absurd flex")
        self.assertTrue(any("setting" in error and "Hindi" in error for error in errors))

        beat = self.story["scenes"][0]["beats"][0]
        beat["line"] = "I bought the whole office!"
        beat["caption_text"] = beat["line"]
        errors = shorts_maker.validate_brainrot_story(self.story, 30, 50, [])
        self.assertTrue(any("not mostly Hindi" in error for error in errors))

    def test_hook_examples_match_each_requested_category_in_hindi(self):
        for hook_type, example in shorts_maker.HOOK_IDEA_EXAMPLES.items():
            with self.subTest(hook_type=hook_type):
                idea = {**example, "hook_type": hook_type}
                self.assertEqual(shorts_maker.validate_brainrot_idea(idea, hook_type), [])
        wrong_place_requirement = shorts_maker.HOOK_REQUIREMENTS["wrong-place chaos"]
        self.assertTrue(all(place in wrong_place_requirement for place in shorts_maker.HOOK_PLACES))

    def test_generated_metadata_is_aligned_and_sound_effects_are_sparse(self):
        data = writer_payload(self.story)
        data["title"] = "गलत शीर्षक"
        data["scenes"][0]["beats"][0]["line"] = "गलत शुरुआत"
        data["scenes"][0]["beats"][0]["caption_text"] = "गलत शुरुआत"
        data["scenes"][0]["characters"] = ["Cup", "Chair"]
        data["scenes"][0]["beats"][0]["speaker"] = "Cup"
        normalized = shorts_maker.normalize_object_names(data)
        aligned = shorts_maker.align_story_opening(normalized, self.story["idea"])
        aligned["scenes"][0]["characters"] = ["Manager", "Director"]
        aligned["scenes"][0]["beats"][0]["emphasis"] = ["not in caption"]
        aligned = shorts_maker.normalize_scene_details(aligned, self.story["idea"])
        sparse = shorts_maker.limit_sound_effects(aligned, 3)
        aligned = shorts_maker.candidate_script(sparse, self.story["idea"], "absurd flex")
        self.assertEqual(aligned["title"], self.story["hook_line"])
        self.assertEqual(aligned["scenes"][0]["beats"][0]["line"], self.story["hook_line"])
        self.assertEqual(aligned["scenes"][0]["characters"], ["कप", "कुर्सी"])
        self.assertEqual(aligned["scenes"][0]["beats"][0]["speaker"], "कप")
        self.assertTrue(all("animals" in s["image_prompt"] for s in aligned["scenes"]))
        self.assertTrue(all(
            word in aligned["scenes"][0]["beats"][0]["caption_text"]
            for word in aligned["scenes"][0]["beats"][0]["emphasis"]
        ))
        self.assertLessEqual(sum(bool(s["sfx_hint"]) for s in aligned["scenes"]), 3)

    def test_caption_text_is_normalized_to_spoken_line(self):
        data = writer_payload(self.story)
        data["scenes"][0]["beats"][1]["caption_text"] = "अलग कैप्शन"
        normalized = shorts_maker.normalize_scene_details(data, self.story["idea"])
        beat = normalized["scenes"][0]["beats"][1]
        self.assertEqual(beat["caption_text"], beat["line"])

    def test_normalizer_preserves_dialogue_and_keeps_a_stable_english_cast(self):
        data = writer_payload(self.story)
        original_line = "कुर्सी ने कहा कि आज पूरा ऑफिस मेरा है अभी!"
        data["scenes"][0]["beats"][1]["line"] = original_line
        data["scenes"][2]["characters"] = ["फ्रिज", "टिफिन"]
        normalized = shorts_maker.normalize_scene_details(data, self.story["idea"])
        self.assertEqual(normalized["scenes"][0]["beats"][1]["line"], original_line)
        self.assertGreater(shorts_maker.word_count(original_line), 7)
        self.assertEqual(normalized["scenes"][-1]["beats"][-1]["line"], self.story["scenes"][-1]["beats"][-1]["line"])
        self.assertTrue(all(scene["characters"] == ["कप", "कुर्सी"] for scene in normalized["scenes"]))
        self.assertTrue(all("ceramic cup" in scene["image_prompt"] and "wooden chair" in scene["image_prompt"] for scene in normalized["scenes"]))
        self.assertTrue(all("genuine continuous character animation" in scene["video_prompt"] for scene in normalized["scenes"]))

    def test_narration_speed_retries_toward_duration_window(self):
        slower = shorts_maker.adjusted_narration_speed(1.15, 27.4, 30, 50)
        faster = shorts_maker.adjusted_narration_speed(1.0, 54, 30, 50)
        self.assertLess(slower, 1.15)
        self.assertAlmostEqual(slower, 1.03, places=2)
        self.assertGreater(faster, 1.0)
        self.assertLessEqual(faster, 1.2)

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
        bad["description"] = "English description"
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
            mock.patch.object(shorts_maker, "nvidia_generate", side_effect=responses) as generate,
            self.assertLogs(shorts_maker.LOGGER, level="WARNING"),
        ):
            result = shorts_maker.make_story([], 30, 50, "absurd flex", attempts=2, punch_up=False)
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(result["title"], self.story["title"])
        self.assertEqual(generate.call_args_list[0].kwargs["model_role"], "flash")
        self.assertEqual(generate.call_args_list[1].kwargs["model_role"], "flash")
        writer_prompt = generate.call_args_list[1].args[0]
        self.assertIn("मैंने पूरा बैंक खरीद लिया!", writer_prompt)
        self.assertIn("तू मेरी कुर्सी क्यों बेच रहा?!", writer_prompt)
        self.assertNotIn("A cup claims it bought", writer_prompt)
        self.assertIn("Required duration_seconds: 40", writer_prompt)
        self.assertIn("exactly 8 scenes", writer_prompt)
        writer_schema = generate.call_args_list[1].kwargs["response_schema"]
        self.assertEqual(writer_schema["properties"]["scenes"]["minItems"], 8)
        self.assertEqual(writer_schema["properties"]["scenes"]["maxItems"], 8)
        self.assertNotIn("image_prompt", writer_schema["properties"]["scenes"]["items"]["properties"])

    def test_quality_gate_stops_after_three_writer_attempts(self):
        story = story_fixture()
        bad = writer_payload(story)
        bad["description"] = "English description"
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
            mock.patch.object(shorts_maker, "nvidia_generate", side_effect=responses) as generate,
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
    @unittest.skipUnless(shutil.which(os.getenv("FFMPEG", "ffmpeg")), "FFmpeg is required to verify animation")
    def test_video_motion_check_rejects_a_still_and_accepts_moving_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            still, moving = root / "still.mp4", root / "moving.mp4"
            for source, target in (
                ("color=c=navy:s=128x128:r=12:d=2", still),
                ("testsrc=size=128x128:rate=12:duration=2", moving),
            ):
                subprocess.run(
                    [os.getenv("FFMPEG", "ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", source,
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)],
                    check=True,
                )
            self.assertFalse(shorts_maker.video_has_motion(still))
            self.assertTrue(shorts_maker.video_has_motion(moving))

    def test_render_preserves_the_motion_clip_without_shake_or_zoom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "output"
            out.mkdir()
            clip, audio, target = root / "scene.mp4", root / "voice.mp3", root / "final.mp4"
            clip.write_bytes(b"motion")
            audio.write_bytes(b"audio")
            with (
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.object(shorts_maker, "seconds_of_media", return_value=10.0),
                mock.patch.object(shorts_maker, "write_caption_file"),
                mock.patch.object(shorts_maker.subprocess, "run") as run,
            ):
                shorts_maker.render(story_fixture(), audio, [clip], target)
            filter_complex = run.call_args.args[0][run.call_args.args[0].index("-filter_complex") + 1]
            self.assertIn("concat=n=1", filter_complex)
            self.assertNotIn("zoompan", filter_complex)
            self.assertNotIn("split=2", filter_complex)
            self.assertNotIn("sin(t*", filter_complex)

    def test_slot_key_uses_india_time(self):
        now = datetime(2026, 11, 15, 13, 30, tzinfo=timezone.utc)
        with mock.patch.dict(os.environ, {"SCHEDULE_SLOT": "19:00"}):
            self.assertEqual(shorts_maker.upload_slot_key(now), "2026-11-15T19:00")

    def test_manual_test_upload_uses_a_separate_slot_key(self):
        now = datetime(2026, 11, 15, 13, 30, 45, 123456, tzinfo=timezone.utc)
        manual_slot = shorts_maker.upload_slot_key(now, manual_test=True)
        self.assertTrue(manual_slot.startswith("manual-test-"))
        self.assertNotEqual(manual_slot, shorts_maker.upload_slot_key(now))

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
                mock.patch.object(shorts_maker.argparse.ArgumentParser, "parse_args", return_value=SimpleNamespace(auth_only=False, script_only=False, check_video_api=False, test_upload=False)),
                mock.patch("sys.argv", ["shorts_maker.py"]),
                mock.patch("builtins.print") as output,
            ):
                shorts_maker.main()
        youtube.assert_not_called()
        self.assertIn("already records an upload", output.call_args.args[0])

    def test_manual_test_upload_runs_even_if_daily_slot_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"SCHEDULE_SLOT": "19:00", "BRAINROT_PUNCH_UP": "false", "HF_TOKEN": "offline-token"}
        ):
            root = Path(directory)
            out = root / "output"
            out.mkdir()
            daily_slot = shorts_maker.upload_slot_key()
            (root / "state.json").write_text(
                json.dumps({"next_video": 3, "topics": [], "uploaded_slots": {daily_slot: "existing-video"}}),
                encoding="utf-8",
            )

            def write_audio(_script, target, speed=None):
                target.write_bytes(b"audio")

            def write_video(_data, _audio, _clips, target):
                target.write_bytes(b"video")

            with (
                mock.patch.object(shorts_maker, "ROOT", root),
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.object(shorts_maker, "load_env_file"),
                mock.patch.object(shorts_maker, "youtube_client"),
                mock.patch.object(
                    shorts_maker,
                    "make_story",
                    side_effect=[shorts_maker.BrainrotRejected("wrong hook"), story_fixture()],
                ) as make_story,
                mock.patch.object(shorts_maker, "spoken_script", return_value="offline narration"),
                mock.patch.object(shorts_maker, "create_narration", side_effect=write_audio),
                mock.patch.object(shorts_maker, "prepare_dialogue_audio"),
                mock.patch.object(shorts_maker, "seconds_of_media", return_value=35.0),
                mock.patch.object(shorts_maker, "create_video_clips", return_value=[]),
                mock.patch.object(shorts_maker, "render", side_effect=write_video),
                mock.patch.object(shorts_maker, "upload", return_value="fresh-video") as upload,
                mock.patch.object(
                    shorts_maker.argparse.ArgumentParser,
                    "parse_args",
                    return_value=SimpleNamespace(auth_only=False, script_only=False, check_video_api=False, test_upload=True),
                ),
                mock.patch("sys.argv", ["shorts_maker.py", "--test-upload"]),
                mock.patch("builtins.print"),
            ):
                shorts_maker.main()

            upload.assert_called_once()
            self.assertEqual(make_story.call_count, 2)
            self.assertEqual([call.kwargs["attempts"] for call in make_story.call_args_list], [3, 3])
            self.assertEqual(make_story.call_args_list[0].args[3], "wrong-place chaos")
            self.assertEqual(make_story.call_args_list[1].args[3], "fake-serious accusation")
            saved_state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_state["uploaded_slots"][daily_slot], "existing-video")
            test_slots = [key for key in saved_state["uploaded_slots"] if key.startswith("manual-test-")]
            self.assertEqual(len(test_slots), 1)
            self.assertEqual(saved_state["uploaded_slots"][test_slots[0]], "fresh-video")

    def test_failed_script_generation_reports_the_actual_failure(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"HF_TOKEN": "offline-token", "BRAINROT_PUNCH_UP": "false"}
        ):
            root = Path(directory)
            out = root / "output"
            out.mkdir()
            (root / "state.json").write_text(json.dumps({"next_video": 0, "topics": []}), encoding="utf-8")
            rejected = shorts_maker.BrainrotRejected("invalid script")
            with (
                mock.patch.object(shorts_maker, "ROOT", root),
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.object(shorts_maker, "load_env_file"),
                mock.patch.object(shorts_maker, "youtube_client"),
                mock.patch.object(shorts_maker, "make_story", side_effect=[rejected] * 5) as make_story,
                mock.patch.object(
                    shorts_maker.argparse.ArgumentParser,
                    "parse_args",
                    return_value=SimpleNamespace(auth_only=False, script_only=False, check_video_api=False, test_upload=False),
                ),
                mock.patch("sys.argv", ["shorts_maker.py"]),
                self.assertRaisesRegex(RuntimeError, "Could not generate a valid brainrot script"),
            ):
                shorts_maker.main()
            self.assertEqual(make_story.call_count, 5)
            self.assertTrue(all(call.kwargs["attempts"] == 3 for call in make_story.call_args_list))

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
                mock.patch.object(shorts_maker, "nvidia_generate", side_effect=responses) as generate,
                mock.patch.dict(os.environ, {"BRAINROT_PUNCH_UP": "false"}),
                mock.patch.object(shorts_maker, "youtube_client") as youtube,
                mock.patch.object(shorts_maker, "create_narration") as tts,
                mock.patch.object(shorts_maker, "hf_generate_image") as hf_image,
                mock.patch.object(shorts_maker, "hf_generate_video") as hf_video,
                mock.patch.object(shorts_maker, "render") as render,
                mock.patch.object(shorts_maker, "upload") as upload,
                mock.patch.object(shorts_maker.argparse.ArgumentParser, "parse_args", return_value=SimpleNamespace(auth_only=False, script_only=True, check_video_api=False, test_upload=False)),
                mock.patch("sys.argv", ["shorts_maker.py", "--script-only"]),
                mock.patch("builtins.print"),
            ):
                shorts_maker.main()
            self.assertEqual(generate.call_count, 2)
            self.assertEqual([call.kwargs["model_role"] for call in generate.call_args_list], ["flash", "flash"])
            for external in (youtube, tts, hf_image, hf_video, render, upload):
                external.assert_not_called()
            saved_files = list(out.glob("script-only-*.json"))
            self.assertEqual(len(saved_files), 1)
            self.assertEqual(len(json.loads(saved_files[0].read_text(encoding="utf-8"))["scenes"]), 7)

    def test_video_space_failure_stops_instead_of_publishing_a_still(self):
        png = b"\x89PNG\r\n\x1a\n" + b"fake"
        def save_image(_result, target, _space, _kind):
            target.write_bytes(png)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "output"
            out.mkdir()
            with (
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.object(shorts_maker, "hf_generate_image", return_value=["image"]),
                mock.patch.object(shorts_maker, "save_hf_file", side_effect=save_image),
                mock.patch.object(shorts_maker, "hf_generate_video", side_effect=TimeoutError("queue")),
                mock.patch.object(shorts_maker, "create_pan_zoom_clip", create=True) as fallback,
                self.assertLogs(shorts_maker.LOGGER, level="ERROR"),
                self.assertRaisesRegex(RuntimeError, "refusing to use a still image"),
            ):
                shorts_maker.create_video_clips(
                    [{"image_prompt": "A talking cup", "video_prompt": "The cup spins"}], "0001",
                )
            fallback.assert_not_called()

    def test_openrouter_check_confirms_configured_model_without_generation(self):
        model = {
            "id": "x-ai/grok-imagine-video-1.5-lite",
            "supported_durations": [5],
            "supported_resolutions": ["480p"],
            "supported_aspect_ratios": ["9:16"],
        }
        with (
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-key"}),
            mock.patch.object(shorts_maker, "openrouter_request", return_value={"data": [model]}) as request,
            mock.patch("builtins.print"),
        ):
            shorts_maker.check_openrouter_video_api()
        request.assert_called_once_with("/videos/models")

    def test_openrouter_video_job_polls_and_saves_mp4(self):
        video = b"\x00\x00\x00\x18ftyp" + b"fake-mp4"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "scene.mp4"
            with (
                mock.patch.dict(os.environ, {
                    "OPENROUTER_API_KEY": "offline-key",
                    "OPENROUTER_VIDEO_POLL_SECONDS": "1",
                }),
                mock.patch.object(shorts_maker, "openrouter_request", side_effect=[
                    {"id": "job-123", "status": "pending", "polling_url": "/api/v1/videos/job-123"},
                    {"id": "job-123", "status": "completed"},
                    video,
                ]) as request,
                mock.patch.object(shorts_maker.time, "sleep"),
            ):
                shorts_maker.openrouter_generate_video("A moving scene", target)
            self.assertEqual(target.read_bytes(), video)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(request.call_args_list[0].args[0], "/videos")
            self.assertEqual(request.call_args_list[0].kwargs["payload"]["duration"], 5)
            self.assertEqual(request.call_args_list[0].kwargs["payload"]["aspect_ratio"], "9:16")
            self.assertTrue(request.call_args_list[2].kwargs["raw"])

    def test_hugging_face_wan_is_primary_even_when_openrouter_key_is_set(self):
        png = b"\x89PNG\r\n\x1a\n" + b"fake"
        mp4 = b"\x00\x00\x00\x18ftyp" + b"fake-mp4"

        def save_file(_result, target, _space, kind):
            target.write_bytes(png if kind == "image" else mp4)

        def ffmpeg(command, **_kwargs):
            Path(command[-1]).write_bytes(mp4)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "output"
            out.mkdir()
            with (
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-key"}),
                mock.patch.object(shorts_maker, "hf_generate_image", return_value=["image"]),
                mock.patch.object(shorts_maker, "save_hf_file", side_effect=save_file),
                mock.patch.object(shorts_maker, "openrouter_generate_video", side_effect=RuntimeError("offline")) as openrouter,
                mock.patch.object(shorts_maker, "hf_generate_video", return_value=["motion"]) as hf_video,
                mock.patch.object(shorts_maker.subprocess, "run", side_effect=ffmpeg),
                mock.patch.object(shorts_maker, "video_has_motion", return_value=True),
            ):
                clips = shorts_maker.create_video_clips(
                    [{"image_prompt": "A scene", "video_prompt": "People move"}], "0001",
                )
            hf_video.assert_called_once()
            openrouter.assert_not_called()
            self.assertTrue(clips[0].is_file())

    def test_openrouter_is_fallback_when_hugging_face_wan_fails(self):
        png = b"\x89PNG\r\n\x1a\n" + b"fake"
        mp4 = b"\x00\x00\x00\x18ftyp" + b"fake-mp4"

        def save_image(_result, target, _space, _kind):
            target.write_bytes(png)

        def save_motion(_prompt, target):
            target.write_bytes(mp4)

        def ffmpeg(command, **_kwargs):
            Path(command[-1]).write_bytes(mp4)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "output"
            out.mkdir()
            with (
                mock.patch.object(shorts_maker, "OUT", out),
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-key"}),
                mock.patch.object(shorts_maker, "hf_generate_image", return_value=["image"]),
                mock.patch.object(shorts_maker, "save_hf_file", side_effect=save_image),
                mock.patch.object(shorts_maker, "hf_generate_video", side_effect=TimeoutError("Wan queue")) as wan,
                mock.patch.object(shorts_maker, "openrouter_generate_video", side_effect=save_motion) as openrouter,
                mock.patch.object(shorts_maker.subprocess, "run", side_effect=ffmpeg),
                mock.patch.object(shorts_maker, "video_has_motion", return_value=True),
                self.assertLogs(shorts_maker.LOGGER, level="WARNING"),
            ):
                clips = shorts_maker.create_video_clips(
                    [{"image_prompt": "A scene", "video_prompt": "People move"}], "0002",
                )
            wan.assert_called_once()
            openrouter.assert_called_once()
            self.assertTrue(clips[0].is_file())


if __name__ == "__main__":
    unittest.main()
