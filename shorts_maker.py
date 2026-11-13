"""Create and publish a researched historical YouTube Short each day."""

from __future__ import annotations

import json
import argparse
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def load_env_file() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def claude_message(prompt: str, max_tokens: int = 1800, *, web_search: bool = False) -> tuple[str, list[str]]:
    key = env("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env and GitHub Actions secrets.")
    payload = {
        "model": env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if web_search:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    workspace_id = env("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        headers["anthropic-workspace-id"] = workspace_id
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"), method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Claude API returned HTTP {exc.code}: {detail}") from exc
    text_blocks, sources = [], []
    for block in result.get("content", []):
        if block.get("type") != "text":
            continue
        text_blocks.append(block.get("text", ""))
        for citation in block.get("citations", []):
            url = citation.get("url")
            if url and url not in sources:
                sources.append(url)
    if not text_blocks:
        raise RuntimeError("Claude returned no text content.")
    return "\n".join(text_blocks).strip(), sources


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Claude did not return a JSON object.")


def make_story(genre: str, history: list[str]) -> tuple[dict, list[str]]:
    response, sources = claude_message(
        (
            "Choose a fascinating historical subject yourself; the user supplies only a genre. "
            "Research it on the web, then write an original documentary-style narrated story. "
            "Keep factual claims grounded in reliable sources. Do not invent quotes, events, "
            "or details; make uncertainty clear. Cite reliable sources in the description. "
            "The narration must be 250-270 words so it "
            "runs comfortably beyond 90 seconds. Open with a strong hook and end with a "
            "takeaway. Return only JSON with keys topic, title, script, description, tags. "
            "tags must be an array of short strings. Avoid repeating prior topics.\n"
            f"Genre: {genre}\nPreviously covered topics: {json.dumps(history[-30:], ensure_ascii=False)}"
        ),
        max_tokens=1800,
        web_search=True,
    )
    data = parse_json(response)
    if not all(key in data for key in ("topic", "title", "script", "description", "tags")):
        raise ValueError("Script generator returned incomplete video metadata.")
    if len(data["script"].split()) < 240:
        raise ValueError("Generated story is too short; refusing to publish it.")
    return data, sources


def make_visual_prompts(topic: str, script: str, count: int) -> list[str]:
    response, _ = claude_message(
        (
            f"Create exactly {count} sequential, coherent visual prompts for 15-second "
            "vertical cinematic b-roll clips that cover this historical documentary. "
            "Prompts must follow the narration order and depict scenes as historical "
            "illustrations/recreations, never imply generated footage is authentic archival "
            "footage. No text, subtitles, logos, or modern objects unless historically fitting. "
            "Return only JSON with a visual_prompts array of strings.\n"
            f"Topic: {topic}\nNarration: {script}"
        ),
        max_tokens=2200,
    )
    prompts = parse_json(response)["visual_prompts"]
    if len(prompts) != count:
        raise ValueError(f"Expected {count} visual prompts, received {len(prompts)}.")
    return prompts


def seconds_of_media(path: Path) -> float:
    result = subprocess.run(
        [env("FFPROBE", "ffprobe"), "-v", "error", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def ass_time(seconds: float) -> str:
    total = max(0, int(seconds * 100))
    hours, remainder = divmod(total, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def wrap_caption(text: str, width: int = 28) -> str:
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return r"\N".join(lines)


def render(data: dict, audio: Path, clips: list[Path], target: Path) -> None:
    duration = seconds_of_media(audio)
    sentences = re.split(r"(?<=[.!?])\s+", data["script"].strip())
    weights = [max(1, len(sentence.split())) for sentence in sentences]
    total = sum(weights)
    cursor = 0.0
    events = []
    for sentence, weight in zip(sentences, weights):
        end = cursor + duration * weight / total
        events.append(f"Dialogue: 0,{ass_time(cursor)},{ass_time(end)},Caption,,0,0,0,,{wrap_caption(sentence)}")
        cursor = end

    subtitle = target.with_suffix(".ass")
    safe_title = data["title"].replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    subtitle.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Title,Arial,74,&H00FFFFFF,&H000000FF,&H001B2440,&H80000000,-1,0,1,4,1,8,80,80,210,1\n"
        "Style: Caption,Arial,66,&H00FFFFFF,&H000000FF,&H001B2440,&H80000000,-1,0,1,5,1,5,100,100,390,1\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 1,0:00:00.00,0:00:03.00,Title,,0,0,0,,{safe_title}\n"
        + "\n".join(events) + "\n", encoding="utf-8")

    # FFmpeg's subtitles filter accepts a POSIX-style path; escape drive colon and apostrophe.
    subtitle_arg = subtitle.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
    inputs = [env("FFMPEG", "ffmpeg"), "-y"]
    for clip in clips:
        inputs.extend(["-i", str(clip)])
    audio_index = len(clips)
    inputs.extend(["-i", str(audio)])
    chains = []
    for index in range(len(clips)):
        chains.append(
            f"[{index}:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,fps=30,setsar=1,format=yuv420p,settb=AVTB[v{index}]"
        )
    concat_inputs = "".join(f"[v{index}]" for index in range(len(clips)))
    chains.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=0,subtitles='{subtitle_arg}'[v]")
    subprocess.run(
        inputs + ["-filter_complex", ";".join(chains), "-map", "[v]", "-map", f"{audio_index}:a:0",
         "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest",
         "-movflags", "+faststart", str(target)],
        check=True,
    )
    subtitle.unlink(missing_ok=True)


def kling_request(method: str, path: str, body: dict | None = None) -> dict:
    key = env("KLING_API_KEY")
    if not key:
        raise RuntimeError("Set KLING_API_KEY in .env to enable video generation.")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        "https://api-singapore.klingai.com" + path, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("code") != 0:
        raise RuntimeError(f"Kling API error {result.get('code')}: {result.get('message')}")
    return result


def create_video_clips(prompts: list[str], stamp: str) -> list[Path]:
    task_file = OUT / f"{stamp}.tasks.json"
    task_ids = json.loads(task_file.read_text(encoding="utf-8")) if task_file.exists() else []
    task_ids.extend([None] * (len(prompts) - len(task_ids)))
    for index, prompt in enumerate(prompts):
        clip = OUT / f"{stamp}-scene{index + 1}.mp4"
        if clip.exists() or task_ids[index]:
            continue
        result = kling_request("POST", "/text-to-video/kling-3.0", {
            "prompt": prompt,
            "settings": {
                "resolution": env("KLING_RESOLUTION", "1080p"), "aspect_ratio": "9:16",
                "duration": 15, "audio": "off", "multi_shot": False,
            },
            "options": {"watermark_info": {"enabled": False}},
        })
        task_ids[index] = result["data"]["id"]
        task_file.write_text(json.dumps(task_ids), encoding="utf-8")

    pending = [index for index in range(len(prompts)) if not (OUT / f"{stamp}-scene{index + 1}.mp4").exists()]
    deadline = time.monotonic() + int(env("KLING_TIMEOUT_SECONDS", "2400"))
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError("Kling video generation is still running; it will resume next run.")
        ids = ",".join(task_ids[index] for index in pending)
        result = kling_request("GET", "/tasks?task_ids=" + urllib.parse.quote(ids, safe=","))
        tasks = {task["id"]: task for task in result["data"]}
        for index in pending[:]:
            task = tasks.get(task_ids[index])
            if task and task["status"] == "failed":
                task_ids[index] = None
                task_file.write_text(json.dumps(task_ids), encoding="utf-8")
                raise RuntimeError(f"Kling scene {index + 1} failed: {task.get('message', 'unknown error')}")
            if task and task["status"] == "succeeded":
                url = next(output["url"] for output in task["outputs"] if output["type"] == "video")
                request = urllib.request.Request(url, headers={"User-Agent": "ShortsMaker/1.0"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    (OUT / f"{stamp}-scene{index + 1}.mp4").write_bytes(response.read())
                pending.remove(index)
        if pending:
            time.sleep(15)
    return [OUT / f"{stamp}-scene{index + 1}.mp4" for index in range(len(prompts))]


def youtube_client():
    token_file = ROOT / "youtube_token.json"
    credentials_file = ROOT / env("YOUTUBE_CLIENT_SECRETS", "client_secret.json")
    creds = Credentials.from_authorized_user_file(token_file, SCOPES) if token_file.exists() else None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not credentials_file.exists():
            raise FileNotFoundError(
                f"Add your Google OAuth Desktop client JSON at {credentials_file}"
            )
        creds = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES).run_local_server(port=0)
    token_file.write_text(creds.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=creds)


def upload(video: Path, data: dict, sources: list[str]) -> str:
    description = data.get("description", "").strip() + "\n\nNarration generated using AI."
    if sources:
        description += "\n\nSources:\n" + "\n".join(sources[:8])
    description += "\n\n#Shorts"
    body = {
        "snippet": {
            "title": data["title"][:100], "description": description[:5000],
            "tags": data.get("tags", [])[:20], "categoryId": "27",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": env("CONTAINS_SYNTHETIC_MEDIA", "true").lower() == "true",
        },
    }
    result = youtube_client().videos().insert(
        part="snippet,status", body=body,
        media_body=MediaFileUpload(str(video), mimetype="video/mp4", resumable=True),
    ).execute()
    return result["id"]


def create_narration(script: str, target: Path, speed: float | None = None) -> None:
    voice_id = env("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    payload = json.dumps({
        "text": script,
        "model_id": env("ELEVENLABS_MODEL", "eleven_v4"),
        "voice_settings": {"speed": speed if speed is not None else float(env("NARRATION_SPEED", "0.9"))},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
        data=payload, method="POST",
        headers={"xi-api-key": env("ELEVENLABS_API_KEY"), "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        target.write_bytes(response.read())


def expand_story(script: str, topic: str, sources: list[str]) -> str:
    response, _ = claude_message(
        (
            "Rewrite this historical documentary narration to 300-320 words so its spoken "
            "runtime is safely over 90 seconds. Preserve the story and its factual claims; "
            "add useful context only if supported by these sources. Return only the narration, "
            "with no title or notes.\n"
            f"Topic: {topic}\nSources: {json.dumps(sources)}\nNarration: {script}"
        ),
        max_tokens=1200,
    )
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and publish a historical YouTube Short.")
    parser.add_argument("--auth-only", action="store_true", help="Complete YouTube OAuth without generating a video.")
    args = parser.parse_args()
    load_env_file()
    youtube_client()
    if args.auth_only:
        print("YouTube OAuth is ready.")
        return
    OUT.mkdir(exist_ok=True)
    state_file = ROOT / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {"next_video": 0, "topics": []}
    index = state.get("next_video", state.get("next_topic", 0))
    history = state.get("topics", [])
    genre = env("CONTENT_GENRE", "historical and informative")
    stamp = f"{index + 1:04d}"
    audio = OUT / f"{stamp}.mp3"
    video = OUT / f"{stamp}.mp4"
    metadata = OUT / f"{stamp}.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved.get("topic"):
            data, sources = saved, saved.get("sources", [])
        else:
            data, sources = make_story(genre, history)
            metadata.write_text(json.dumps({**data, "sources": sources}, indent=2), encoding="utf-8")
    else:
        data, sources = make_story(genre, history)
        metadata.write_text(json.dumps({**data, "sources": sources}, indent=2), encoding="utf-8")

    minimum = float(env("MIN_VIDEO_SECONDS", "90"))
    if not audio.exists():
        create_narration(data["script"], audio)
    duration = seconds_of_media(audio)
    if duration < minimum:
        create_narration(data["script"], audio, speed=0.7)
        duration = seconds_of_media(audio)
    if duration < minimum:
        data["script"] = expand_story(data["script"], data["topic"], sources)
        metadata.write_text(json.dumps({**data, "sources": sources}, indent=2), encoding="utf-8")
        create_narration(data["script"], audio, speed=0.7)
        duration = seconds_of_media(audio)
    if duration < minimum:
        raise RuntimeError(f"Narration is only {duration:.1f}s; refusing to publish below {minimum:.0f}s.")

    clip_count = math.ceil(duration / 15)
    if len(data.get("visual_prompts", [])) != clip_count:
        data["visual_prompts"] = make_visual_prompts(data["topic"], data["script"], clip_count)
        metadata.write_text(json.dumps({**data, "sources": sources}, indent=2), encoding="utf-8")
    clips = create_video_clips(data["visual_prompts"], stamp)
    render(data, audio, clips, video)
    rendered_duration = seconds_of_media(video)
    if rendered_duration < minimum:
        raise RuntimeError(f"Rendered video is only {rendered_duration:.1f}s; refusing to upload below {minimum:.0f}s.")
    video_id = upload(video, data, sources)
    history.append(data["topic"])
    state_file.write_text(json.dumps({"next_video": index + 1, "topics": history, "last_video_id": video_id}, indent=2), encoding="utf-8")
    print(f"Saved {video} ({rendered_duration / 60:.1f} minutes)")
    print(f"Published https://youtube.com/shorts/{video_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ShortsMaker failed: {exc}", file=sys.stderr)
        raise
