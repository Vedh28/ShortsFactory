"""Create and publish a Hindi object-comedy YouTube Short each day."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timedelta, timezone
import logging
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
LOGGER = logging.getLogger(__name__)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_STATUS_URL = "https://integrate.api.nvidia.com/v1/status"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_DEFAULT_LITE_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "x-ai/grok-imagine-video-1.5-lite"

NVIDIA_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
NVIDIA_FALLBACK_STATUSES = {202, 400, 403, 404, 408, 429, 500, 502, 503, 504}
INDIA_TIME = timezone(timedelta(hours=5, minutes=30))
HOOK_TYPES = (
    "absurd flex",
    "instant conflict",
    "impossible claim",
    "wrong-place chaos",
    "fake-serious accusation",
)
SFX_HINTS = (
    "vine-boom hit",
    "whoosh",
    "record-scratch",
    "record-stop",
    "dramatic gasp",
    "airhorn blast",
    "cartoon bonk",
    "bass drop",
)
BRAINROT_TOOLKIT = (
    "over-the-top reactions",
    "fake-serious tone",
    "escalating bargaining",
    "repeated catchphrase",
    "fake-rich flex",
    "wrong-order mixup",
    "role reversal",
    "shouted one word",
    "dramatic gasp",
    "sudden silence then chaos",
    "impossible object",
    "non-sequitur cut",
    "mock-news or court drama",
)
IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "setting": {"type": "string"},
        "conflict": {"type": "string"},
        "punchline": {"type": "string"},
        "hook_type": {"type": "string", "enum": list(HOOK_TYPES)},
        "hook_line": {"type": "string"},
        "catchphrase": {"type": "string"},
    },
    "required": ["setting", "conflict", "punchline", "hook_type", "hook_line", "catchphrase"],
    "additionalProperties": False,
}
BRAINROT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "duration_seconds": {"type": "integer"},
        "toolkit_used": {
            "type": "array",
            "minItems": 3,
            "items": {"type": "string", "enum": list(BRAINROT_TOOLKIT)},
        },
        "scenes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "characters": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "string"}},
                    "beats": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {"type": "string"},
                                "line": {"type": "string"},
                                "caption_text": {"type": "string"},
                                "emphasis": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "items": {"type": "string"},
                                },
                                "beat_type": {
                                    "type": "string",
                                    "enum": ["hook", "normal", "punchline", "twist", "cutoff", "reaction"],
                                },
                            },
                            "required": ["speaker", "line", "caption_text", "emphasis", "beat_type"],
                            "additionalProperties": False,
                        },
                    },
                    "sfx_hint": {"type": "string", "enum": ["", *SFX_HINTS]},
                    "meme_text": {"type": "string"},
                    "image_prompt": {"type": "string"},
                    "video_prompt": {"type": "string"},
                },
                "required": ["characters", "beats", "sfx_hint", "meme_text", "image_prompt", "video_prompt"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "description", "tags", "duration_seconds", "toolkit_used", "scenes"],
    "additionalProperties": False,
}


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def upload_slot_key(now: datetime | None = None, *, manual_test: bool = False) -> str:
    local_now = datetime.now(INDIA_TIME) if now is None else (
        now.replace(tzinfo=INDIA_TIME) if now.tzinfo is None else now.astimezone(INDIA_TIME)
    )
    if manual_test:
        return f"manual-test-{local_now.strftime('%Y%m%dT%H%M%S%f')}"
    slot = env("SCHEDULE_SLOT")
    if slot not in {"09:00", "19:00"}:
        slot = "09:00" if local_now.hour < 14 else "19:00"
    return f"{local_now.date().isoformat()}T{slot}"


def slot_already_uploaded(state: dict, slot_key: str) -> bool:
    slots = state.get("uploaded_slots", {})
    return isinstance(slots, dict) and slot_key in slots


def save_state(state_file: Path, state: dict) -> None:
    temporary_file = state_file.with_suffix(state_file.suffix + ".tmp")
    temporary_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_file.replace(state_file)


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


class NVIDIAAPIError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def nvidia_chat_request(model: str, prompt: str, max_tokens: int, json_output: bool, response_schema: dict | None) -> dict:
    token = env("NVIDIA_API_KEY")
    if not token:
        raise RuntimeError("Set NVIDIA_API_KEY in .env and GitHub Actions secrets.")
    system_prompt = "You are the ShortsFactory story generator. Follow the user's prompt exactly."
    if json_output:
        system_prompt += " Return only a JSON object."
        if response_schema:
            system_prompt += " Match this JSON Schema exactly: " + json.dumps(response_schema, ensure_ascii=False)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }
    payload["stream"] = False
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    if json_output and response_schema and model.endswith("nemotron-3.5-bolt-e4b"):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "shorts_factory_response", "strict": True, "schema": response_schema},
        }
    request = urllib.request.Request(
        NVIDIA_BASE_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(_NvidiaRedirectHandler())
        with opener.open(request, timeout=180) as response:
            status = response.status
            body = response.read()
            if status == 202:
                request_id = response.headers.get("NVCF-REQID")
                if not request_id:
                    try:
                        request_id = json.loads(body.decode("utf-8")).get("requestId")
                    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                        pass
                if not isinstance(request_id, str) or not request_id:
                    raise NVIDIAAPIError("NVIDIA returned a pending response without a request ID.", 202)
                result = None
                for _ in range(18):
                    poll_request = urllib.request.Request(
                        f"{NVIDIA_STATUS_URL}/{urllib.parse.quote(request_id, safe='')}",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Accept": "application/json",
                            "NVCF-POLL-SECONDS": "10",
                        },
                        method="GET",
                    )
                    with opener.open(poll_request, timeout=15) as poll_response:
                        if poll_response.status == 202:
                            continue
                        result = json.loads(poll_response.read().decode("utf-8"))
                        break
                if result is None:
                    raise NVIDIAAPIError("NVIDIA generation remained pending after 180 seconds.", 202)
            else:
                result = json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        try:
            error_body = json.loads(detail)
            detail = error_body.get("error", {}).get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise NVIDIAAPIError(f"NVIDIA request failed ({exc.code}): {detail}", exc.code) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NVIDIAAPIError("NVIDIA returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise NVIDIAAPIError("NVIDIA returned an unexpected response format.")
    return result


class _NvidiaRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        parsed = urllib.parse.urlsplit(new_url)
        if parsed.scheme != "https" or parsed.netloc != "integrate.api.nvidia.com":
            raise RuntimeError("Refusing to forward the NVIDIA API key to a redirected host.")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def nvidia_generate(
    prompt: str,
    max_tokens: int = 1800,
    *,
    json_output: bool = True,
    model_role: str = "flash",
    response_schema: dict | None = None,
) -> str:
    flash_model = env("NVIDIA_MODEL", NVIDIA_DEFAULT_MODEL)
    lite_model = env("NVIDIA_LITE_MODEL", NVIDIA_DEFAULT_LITE_MODEL)
    if model_role == "lite":
        models = list(dict.fromkeys((lite_model, flash_model)))
    elif model_role == "flash":
        models = list(dict.fromkeys((flash_model, lite_model)))
    else:
        raise ValueError(f"Unknown NVIDIA model role: {model_role}")

    for index, model in enumerate(models):
        response = None
        for retry in range(3):
            try:
                response = nvidia_chat_request(model, prompt, max_tokens, json_output, response_schema)
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                if retry < 2:
                    delay = 2**retry
                    LOGGER.warning("NVIDIA request for %s had a network error; retrying in %ss.", model, delay)
                    time.sleep(delay)
                    continue
                if index + 1 < len(models):
                    LOGGER.warning("NVIDIA model %s is unavailable (%s); falling back to %s.", model, type(exc).__name__, models[index + 1])
                    break
                raise
            except NVIDIAAPIError as exc:
                if exc.status in NVIDIA_RETRY_STATUSES and retry < 2:
                    delay = 2**retry
                    LOGGER.warning("NVIDIA model %s returned HTTP %s; retrying in %ss.", model, exc.status, delay)
                    time.sleep(delay)
                    continue
                if index + 1 < len(models) and exc.status in NVIDIA_FALLBACK_STATUSES:
                    LOGGER.warning("NVIDIA model %s failed with HTTP %s; falling back to %s.", model, exc.status, models[index + 1])
                    break
                raise
        if response is None:
            continue
        choices = response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        output = message.get("content")
        if isinstance(output, list):
            output = "".join(part.get("text", "") for part in output if isinstance(part, dict))
        output = output.strip() if isinstance(output, str) else ""
        try:
            if not output:
                raise ValueError("NVIDIA returned no text content.")
            if json_output:
                parsed = parse_json(output)
                if response_schema is not None:
                    validate_json_schema(parsed, response_schema)
        except ValueError as exc:
            if index + 1 < len(models):
                LOGGER.warning(
                    "NVIDIA model %s returned an invalid structured response (%s); trying %s.",
                    model,
                    exc,
                    models[index + 1],
                )
                continue
            raise
        return output
    raise RuntimeError("NVIDIA generation failed without returning a response.")


def parse_json(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("NVIDIA returned an empty JSON response.")

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    if value is not None:
        raise ValueError("NVIDIA returned valid JSON, but its top-level value is not an object.")

    index = text.find("{")
    if index >= 0:
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
    raise ValueError(f"NVIDIA did not return a valid JSON object ({len(text)} characters).")


def validate_json_schema(value: object, schema: dict, path: str = "$") -> None:
    expected_type = schema.get("type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"NVIDIA JSON field {path} must be one of {schema['enum']}.")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"NVIDIA JSON field {path} must be an object.")
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"NVIDIA JSON field {path} is missing required fields: {', '.join(missing)}.")
        if schema.get("additionalProperties") is False:
            unexpected = value.keys() - properties.keys()
            if unexpected:
                raise ValueError(f"NVIDIA JSON field {path} contains unexpected fields: {', '.join(sorted(unexpected))}.")
        for key, child_schema in properties.items():
            if key in value:
                validate_json_schema(value[key], child_schema, f"{path}.{key}")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"NVIDIA JSON field {path} must be an array.")
        if len(value) < schema.get("minItems", 0) or (
            schema.get("maxItems") is not None and len(value) > schema["maxItems"]
        ):
            raise ValueError(f"NVIDIA JSON field {path} has an invalid number of items.")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_json_schema(item, item_schema, f"{path}[{index}]")
    elif expected_type == "string" and not isinstance(value, str):
        raise ValueError(f"NVIDIA JSON field {path} must be a string.")
    elif expected_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"NVIDIA JSON field {path} must be an integer.")


NARRATOR_PHRASES = (
    "एक बार की बात है", "आज हम", "दोस्तो", "दोस्तों", "इस कहानी में", "चलिए देखते हैं",
    "आज की कहानी", "नैतिक शिक्षा", "अंत में हमने सीखा",
)
UNSAFE_TERMS = (
    "राजनीति", "चुनाव", "धर्म", "जाति", "समुदाय", "मुस्लिम", "हिंदू", "ईसाई", "सिख",
    "दलित", "ब्राह्मण", "गाली", "भोसड़ी", "चूत", "लौड़ा", "हरामी", "सेक्स",
    "यौन", "खून", "हत्या", "गोली", "बम", "आत्महत्या", "नशा", "स्टंट", "जहर",
    "मोदी", "राहुल गांधी", "शाहरुख", "सलमान खान", "टेलर स्विफ्ट", "tung tung",
    "tralalero", "bombardiro", "औरतें हमेशा", "लड़कियाँ हमेशा", "मर्द हमेशा",
)
RESOLUTION_PHRASES = ("सब ठीक हो गया", "और फिर सब खुश", "कहानी खत्म", "सबने माफ किया", "समस्या सुलझ गई")
ANIMAL_TERMS = (
    "शेर", "बंदर", "लोमड़ी", "कौआ", "भालू", "हाथी", "कुत्ता", "बिल्ली", "गाय",
    "बकरी", "मगरमच्छ", "खरगोश", "साँप", "चूहा", "कबूतर", "animal", "lion", "monkey",
    "fox", "crow", "bear", "elephant", "dog", "cat",
)
HOOK_OBJECTS = (
    "फ्रिज", "कप", "कुर्सी", "चप्पल", "आलू", "पंखा", "लैपटॉप", "टिफिन", "रोबोट",
    "चाबी", "घड़ी", "टोस्टर", "मोबाइल", "जूता", "पेंसिल", "cup", "fridge", "robot",
    "चम्मच", "थाली", "गिलास", "दरवाजा", "ताला", "तिजोरी", "मेज", "प्लेट", "बोतल",
    "किताब", "पेन", "बैग", "तकिया", "कंबल", "बाल्टी", "झाड़ू", "डिब्बा", "कैमरा",
    "माइक्रोफोन", "सीढ़ी", "घंटी", "अलार्म", "कैलकुलेटर", "चार्जर", "माउस", "कीबोर्ड",
    "पर्दा", "आईना", "बिस्कुट", "समोसा", "नल", "मशीन", "कार", "बस", "ट्रेन", "साइकिल",
    "स्कूटर", "हेलमेट", "छाता", "कंघी", "साबुन", "ब्रश", "कूड़ेदान", "घड़ा", "बेंच", "रबर",
    "chair", "slipper", "potato", "fan", "laptop", "lunchbox", "key", "clock", "toaster",
    "phone", "shoe", "pencil", "spoon", "table", "door", "safe", "bottle", "book", "pen",
    "bag", "pillow", "bucket", "broom", "camera", "microphone", "bell", "alarm", "calculator",
    "charger", "mouse", "keyboard", "screen", "mirror", "car", "bus", "train", "bicycle",
    "umbrella", "comb", "soap", "brush", "bench", "eraser",
)
HOOK_PLACES = ("ऑफिस", "कोर्ट", "स्कूल", "जिम", "बैंक", "मेट्रो", "एयरपोर्ट", "अस्पताल", "पासपोर्ट", "शादी")
HOOK_IDEA_EXAMPLES = {
    "absurd flex": {
        "setting": "बैंक की लिफ्ट",
        "conflict": "कप तिजोरी से किराया मांगता है",
        "punchline": "कुर्सी बैंक की मालिक बनती है",
        "hook_line": "मैंने पूरा बैंक खरीद लिया!",
        "catchphrase": "क्या सेटिंग है",
    },
    "instant conflict": {
        "setting": "ऑफिस की कुर्सी",
        "conflict": "चप्पल सीट बेचने लगती है",
        "punchline": "कुर्सी टिकट जांचने लगती है",
        "hook_line": "तू मेरी कुर्सी क्यों बेच रहा?!",
        "catchphrase": "अबे, क्या सेटिंग है",
    },
    "impossible claim": {
        "setting": "स्कूल का फ्रिज",
        "conflict": "फ्रिज अपना पुराना पेशा बताता है",
        "punchline": "टिफिन उसे प्रधानाचार्य मानता है",
        "hook_line": "भाई, मैं कल फ्रिज था!",
        "catchphrase": "सच में, भाई",
    },
    "wrong-place chaos": {
        "setting": "ऑफिस का स्वागत कक्ष",
        "conflict": "चप्पल नौकरी का इंटरव्यू लेती है",
        "punchline": "पंखा उम्मीदवारों को घुमाता है",
        "hook_line": "ऑफिस में चप्पल मैनेजर है!",
        "catchphrase": "भाई, गजब सेटिंग",
    },
    "fake-serious accusation": {
        "setting": "कोर्ट की गवाही",
        "conflict": "चाबी पर चोरी का आरोप लगता है",
        "punchline": "कुर्सी चाबी की गवाह बनती है",
        "hook_line": "तूने मेरी चाबी चुराई, कोर्ट में मिल!",
        "catchphrase": "ऑर्डर, क्या सेटिंग",
    },
}
HOOK_REQUIREMENTS = {
    "absurd flex": "Start hook_line exactly with 'मैंने' and include 'खरीद', 'ले लिया', or 'बना दिया'.",
    "instant conflict": "Start hook_line with 'तू', 'अबे', or 'भाई'; include 'क्यों', 'किसने', 'छुआ', or 'बैठ'.",
    "impossible claim": "Start hook_line with 'भाई' or 'मैं'; include 'कल' and one of 'था', 'थी', 'थे', or 'हूँ'.",
    "wrong-place chaos": "Mention one Hindi object and one Hindi place in hook_line; put the object in the wrong place.",
    "fake-serious accusation": "Start hook_line exactly with 'तूने' and include 'चुराया', 'चुराई', 'चोरी', 'कोर्ट', or 'कचहरी'.",
}
OBJECT_NAME_TRANSLATIONS = {
    "cup": "कप", "chair": "कुर्सी", "fridge": "फ्रिज", "refrigerator": "फ्रिज",
    "slipper": "चप्पल", "fan": "पंखा", "lunchbox": "टिफिन", "tiffin": "टिफिन",
    "potato": "आलू", "laptop": "लैपटॉप", "key": "चाबी", "clock": "घड़ी",
    "toaster": "टोस्टर", "spoon": "चम्मच", "door": "दरवाजा", "safe": "तिजोरी",
    "book": "किताब", "bottle": "बोतल", "table": "मेज", "phone": "मोबाइल",
    "mobile": "मोबाइल", "shoe": "जूता", "pencil": "पेंसिल", "glass": "गिलास",
    "plate": "थाली", "jar": "डिब्बा", "bag": "बैग", "pillow": "तकिया",
    "bucket": "बाल्टी", "broom": "झाड़ू", "camera": "कैमरा", "bell": "घंटी",
    "charger": "चार्जर", "mouse": "माउस", "keyboard": "कीबोर्ड", "mirror": "आईना",
    "car": "कार", "bus": "बस", "train": "ट्रेन", "umbrella": "छाता", "comb": "कंघी",
    "soap": "साबुन", "brush": "ब्रश", "bench": "बेंच", "eraser": "रबर",
}


class BrainrotRejected(ValueError):
    pass


def dialogue_lines(data: dict) -> list[dict]:
    return [beat for scene in data.get("scenes", []) for beat in scene.get("beats", [])]


def spoken_script(data: dict) -> str:
    return "\n".join(beat.get("line", "").strip() for beat in dialogue_lines(data)).strip()


def word_count(text: str) -> int:
    return len(text.split())


def is_hindi_dominant(text: str) -> bool:
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    letters = devanagari + latin
    return devanagari >= 2 and (not letters or devanagari / letters >= 0.55)


def contains_animal(text: str) -> bool:
    punctuation = "!?.,;:।॥\"'()[]{}"
    tokens = [token.strip(punctuation) for token in text.casefold().split()]
    for term in ANIMAL_TERMS:
        if term.isascii():
            if term in tokens:
                return True
        elif any(token.startswith(term) for token in tokens):
            return True
    return False


def idea_similarity(left: str, right: str) -> float:
    punctuation = "!?.,;:।॥\"'()[]{}"
    left_tokens = {token.strip(punctuation) for token in left.casefold().split()} - {""}
    right_tokens = {token.strip(punctuation) for token in right.casefold().split()} - {""}
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens and right_tokens else 0.0


def is_duplicate_idea(idea: dict, history: list) -> bool:
    for previous in history[-50:]:
        previous_idea = previous.get("idea", previous) if isinstance(previous, dict) else {"setting": str(previous)}
        if not isinstance(previous_idea, dict):
            previous_idea = {"setting": str(previous_idea)}
        setting_match = idea_similarity(idea.get("setting", ""), previous_idea.get("setting", "")) >= 0.8
        conflict_match = idea_similarity(idea.get("conflict", ""), previous_idea.get("conflict", "")) >= 0.65
        punchline_match = idea_similarity(idea.get("punchline", ""), previous_idea.get("punchline", "")) >= 0.8
        if punchline_match or (setting_match and conflict_match):
            return True
    return False


def hook_is_valid(hook_type: str, line: str) -> bool:
    text = line.strip().casefold()
    if hook_type == "absurd flex":
        return text.startswith("मैंने") and any(word in text for word in ("खरीद", "ले लिया", "बना दिया"))
    if hook_type == "instant conflict":
        return text.startswith(("तू", "अबे", "भाई")) and any(word in text for word in ("क्यों", "किसने", "छुआ", "बैठ"))
    if hook_type == "impossible claim":
        return text.startswith(("भाई", "मैं")) and "कल" in text and any(word in text for word in ("था", "थी", "थे", "हूँ"))
    if hook_type == "wrong-place chaos":
        return any(word in text for word in HOOK_OBJECTS) and any(word in text for word in HOOK_PLACES)
    if hook_type == "fake-serious accusation":
        return text.startswith("तूने") and any(word in text for word in ("चुराया", "चुराई", "चोरी", "कोर्ट", "कचहरी"))
    return False


def validate_brainrot_idea(idea: dict, requested_hook: str) -> list[str]:
    errors_found = []
    fields = ("setting", "conflict", "punchline", "hook_line", "catchphrase")
    for field in fields:
        value = idea.get(field, "")
        if not isinstance(value, str) or not is_hindi_dominant(value):
            errors_found.append(f"idea field {field} must be colloquial Hindi in Devanagari")
    if idea.get("hook_type") != requested_hook:
        errors_found.append("idea has the wrong hook category")
    hook_line = idea.get("hook_line", "")
    if not isinstance(hook_line, str) or not 2 <= word_count(hook_line) <= 7 or not hook_is_valid(requested_hook, hook_line):
        errors_found.append(f"idea opening line does not match the {requested_hook!r} hook pattern: {hook_line!r}")
    catchphrase = idea.get("catchphrase", "")
    if (not isinstance(catchphrase, str) or not 2 <= word_count(catchphrase) <= 4
            or not is_hindi_dominant(catchphrase)):
        errors_found.append("idea catchphrase must contain two to four Hindi words")
    content = " ".join(str(idea.get(field, "")) for field in fields).casefold()
    if contains_animal(content):
        errors_found.append("idea uses an animal character")
    for term in UNSAFE_TERMS:
        if term.casefold() in content:
            errors_found.append(f"idea contains a prohibited term: {term}")
    return errors_found


def validate_brainrot_story(data: dict, minimum: float, maximum: float, history: list | None = None) -> list[str]:
    errors_found = []
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return ["missing scenes"]
    lines = dialogue_lines(data)
    if not lines:
        return ["missing dialogue"]
    for line_index, beat in enumerate(lines):
        line = beat.get("line", "").strip()
        if not line:
            errors_found.append(f"line {line_index + 1} is empty")
            continue
        if not is_hindi_dominant(line):
            errors_found.append(f"line {line_index + 1} is not mostly Hindi in Devanagari")
        if word_count(line) < 2:
            errors_found.append(f"line {line_index + 1} is under two words")
        if word_count(line) > 7:
            errors_found.append(f"line {line_index + 1} is over seven words")
        lowered = line.casefold()
        if any(phrase in lowered for phrase in NARRATOR_PHRASES):
            errors_found.append(f"line {line_index + 1} contains narrator-style setup")
        if beat.get("caption_text", "").strip() != line:
            errors_found.append(f"line {line_index + 1} caption does not match its dialogue")
        emphasis = beat.get("emphasis", [])
        if not 1 <= len(emphasis) <= 4 or any(word.casefold() not in lowered for word in emphasis):
            errors_found.append(f"line {line_index + 1} needs one to four highlighted words from its caption")

    first_line = lines[0].get("line", "").strip()
    hook_type = data.get("hook_type", "")
    if word_count(first_line) > 7 or not hook_is_valid(hook_type, first_line):
        errors_found.append("first line is not a valid short hook")
    if data.get("title", "").strip() != first_line or data.get("hook_line", "").strip() != first_line:
        errors_found.append("title idea must exactly match the opening hook")

    catchphrase = data.get("catchphrase", "").strip()
    if not catchphrase or sum(beat.get("line", "").casefold().count(catchphrase.casefold()) for beat in lines) < 3:
        errors_found.append("catchphrase is not repeated at least three times")
    if catchphrase and sum(
        any(catchphrase.casefold() in beat.get("line", "").casefold() for beat in scene.get("beats", []))
        for scene in scenes
    ) < 3:
        errors_found.append("catchphrase must recur in at least three different scenes")
    if len(lines) < math.ceil(float(data.get("duration_seconds", 0)) / 3):
        errors_found.append("there are not enough beats for a new beat every three seconds")
    if not minimum <= float(data.get("duration_seconds", 0)) <= maximum:
        errors_found.append("declared duration is outside the configured range")
    if len(scenes) != math.ceil(float(data.get("duration_seconds", 0)) / 5):
        errors_found.append("scene count does not match five-second clips")
    estimate = word_count(spoken_script(data)) / 2.5
    if not minimum <= estimate <= maximum:
        errors_found.append("dialogue word count estimates a duration outside the configured range")

    for index, scene in enumerate(scenes):
        characters = scene.get("characters", [])
        beats = scene.get("beats", [])
        if not 1 <= len(characters) <= 2:
            errors_found.append(f"scene {index + 1} must have one or two characters")
        speakers = {beat.get("speaker", "").casefold() for beat in beats}
        if any(name.casefold() not in speakers for name in characters):
            errors_found.append(f"scene {index + 1} has a character without dialogue")
        if any(beat.get("speaker") not in characters for beat in beats):
            errors_found.append(f"scene {index + 1} has a speaker outside its character list")
        if any(not any(obj in name.casefold() for obj in HOOK_OBJECTS) for name in characters):
            errors_found.append(f"scene {index + 1} must use fictional object characters")
        if any(not is_hindi_dominant(name) for name in characters):
            errors_found.append(f"scene {index + 1} character names must be in Hindi")
        if scene.get("sfx_hint", "") not in ("", *SFX_HINTS):
            errors_found.append(f"scene {index + 1} has an invalid sound hint")
        visual_text = " ".join((scene.get("image_prompt", ""), scene.get("video_prompt", ""), *characters)).casefold()
        if contains_animal(visual_text):
            errors_found.append(f"scene {index + 1} uses an animal character")
    if sum(bool(scene.get("meme_text", "").strip()) for scene in scenes) > 1:
        errors_found.append("meme text may appear at most once")
    if sum(bool(scene.get("sfx_hint")) for scene in scenes) > math.ceil(len(scenes) / 2):
        errors_found.append("sound hints are used too frequently")
    if len(set(data.get("toolkit_used", []))) < 3:
        errors_found.append("the script uses fewer than three brainrot techniques")

    if len(scenes[-1].get("beats", [])) < 2 or scenes[-1]["beats"][-2].get("beat_type") not in ("twist", "cutoff"):
        errors_found.append("the final scene has no twist or cut-off")
    elif scenes[-1]["beats"][-1].get("beat_type") != "reaction" or word_count(scenes[-1]["beats"][-1].get("line", "")) > 3:
        errors_found.append("the final twist needs a short reaction line and no resolution")

    content = " ".join(
        [data.get("title", ""), data.get("description", "")]
        + [str(data.get("idea", {}).get(field, "")) for field in ("setting", "conflict", "punchline", "hook_line", "catchphrase")]
        + [beat.get("line", "") for beat in lines]
        + [
            scene.get("image_prompt", "") + " " + scene.get("video_prompt", "") + " " + scene.get("meme_text", "")
            for scene in scenes
        ]
    ).casefold()
    if contains_animal(content):
        errors_found.append("story or visual prompt uses an animal")
    for term in UNSAFE_TERMS:
        if term.casefold() in content:
            errors_found.append(f"content contains a prohibited term: {term}")
    if not is_hindi_dominant(data.get("description", "")):
        errors_found.append("description must be written mostly in Hindi")
    toolkit_evidence = (
        any(word in content for word in ("अबे", "भाई", "सच में", "क्या बोला", "गजब", "झक्कास", "बवाल"))
        or sum(any(mark in beat.get("line", "") for mark in "!??!") for beat in lines) >= 3,
        any(word in content for word in ("किराया", "वेतन", "दाम", "मोल", "पैसे", "रुपये", "नौकरी", "तनख्वाह", "भुगतान")),
        any(word in content for word in ("मैनेजर", "ceo", "मालिक", "जज", "बॉस", "वकील", "ड्राइवर", "प्रधानाचार्य")),
        hook_type == "absurd flex" and any(word in first_line for word in ("खरीद", "ले लिया", "बना दिया")),
        sum(catchphrase.casefold() in beat.get("line", "").casefold() for beat in lines) >= 3 if catchphrase else False,
        any(word in content for word in ("ऑर्डर", "ऑब्जेक्शन", "कोर्ट", "गवाह", "आरोप", "सजा")),
        bool(scenes) and all(
            any(any(obj in character.casefold() for obj in HOOK_OBJECTS) for character in scene.get("characters", []))
            for scene in scenes
        ),
    )
    if sum(toolkit_evidence) < 3:
        errors_found.append("fewer than three brainrot techniques have dialogue evidence")
    if any(phrase in scenes[-1]["beats"][-2].get("line", "").casefold() for phrase in RESOLUTION_PHRASES):
        errors_found.append("final beat resolves the story")
    idea = data.get("idea", {})
    if history is not None and is_duplicate_idea(idea, history):
        errors_found.append("idea is too similar to the saved idea history")
    return errors_found


def candidate_script(data: dict, idea: dict, hook_type: str) -> dict:
    result = dict(data)
    result["idea"] = idea
    result["hook_type"] = hook_type
    result["catchphrase"] = idea["catchphrase"]
    result["hook_line"] = idea["hook_line"]
    result["script"] = spoken_script(result)
    return result


def align_story_opening(data: dict, idea: dict) -> dict:
    """Keep the validated idea hook identical in the title, opening dialogue, and caption."""
    result = dict(data)
    scenes = result.get("scenes", [])
    if scenes and scenes[0].get("beats"):
        beat = dict(scenes[0]["beats"][0])
        beat["line"] = idea["hook_line"]
        beat["caption_text"] = idea["hook_line"]
        beat["emphasis"] = [word for word in idea["hook_line"].split() if re.search(r"[\u0900-\u097F]", word)][:4]
        beat["beat_type"] = "hook"
        first_scene = dict(scenes[0])
        first_scene["beats"] = [beat, *first_scene["beats"][1:]]
        result["scenes"] = [first_scene, *scenes[1:]]
    result["title"] = idea["hook_line"]
    return result


def limit_sound_effects(data: dict, maximum_count: int) -> dict:
    result = dict(data)
    remaining = maximum_count
    scenes = []
    for scene in result.get("scenes", []):
        scene = dict(scene)
        if scene.get("sfx_hint"):
            if remaining:
                remaining -= 1
            else:
                scene["sfx_hint"] = ""
        scenes.append(scene)
    result["scenes"] = scenes
    return result


def normalize_scene_details(data: dict, idea: dict) -> dict:
    """Enforce a safe object cast and align visual prompts with the spoken scene."""
    result = dict(data)
    scenes = []
    setting = idea.get("setting", "एक मज़ेदार जगह")
    for index, scene in enumerate(result.get("scenes", [])):
        scene = dict(scene)
        characters = scene.get("characters", [])
        if not (1 <= len(characters) <= 2 and all(
            any(obj in str(name).casefold() for obj in HOOK_OBJECTS) for name in characters
        )):
            characters = ["कप", "कुर्सी"]
        else:
            characters = [str(name) for name in characters]
        old_speakers = list(dict.fromkeys(beat.get("speaker", "") for beat in scene.get("beats", [])))
        speaker_map = {old: characters[position % len(characters)] for position, old in enumerate(old_speakers)}
        beats = []
        for beat_index, beat in enumerate(scene.get("beats", [])):
            beat = dict(beat)
            beat["speaker"] = speaker_map.get(beat.get("speaker", ""), characters[beat_index % len(characters)])
            beat["caption_text"] = beat.get("line", "").strip()
            caption_words = beat.get("caption_text", "").split()
            emphasis = beat.get("emphasis", [])
            if not isinstance(emphasis, list) or not 1 <= len(emphasis) <= 4 or any(
                word.casefold() not in beat.get("caption_text", "").casefold() for word in emphasis
            ):
                emphasis = [word.strip("!?.,;:।॥\\\"'()[]{}") for word in caption_words[-2:]]
                beat["emphasis"] = [word for word in emphasis if word][:4] or caption_words[:1]
            beats.append(beat)
        scene["characters"] = characters
        scene["beats"] = beats
        conversation = " ".join(beat.get("line", "") for beat in beats)
        safe_cast = "cup and chair" if characters == ["कप", "कुर्सी"] else " and ".join(characters)
        scene["image_prompt"] = (
            f"Vertical cinematic 3D cartoon still of two fictional talking household objects, {safe_cast}, "
            f"in {setting}. Exaggerated comic expressions and mid-chaos pose inspired by: {conversation}. "
            "Only these objects, no animals, no humans, no extra characters, no readable text."
        )
        scene["video_prompt"] = (
            f"Animate the same fictional talking household objects, {safe_cast}, in {setting}: "
            f"{conversation}. Quick expressive gestures and a comic camera push-in. No animals, "
            "no humans, no extra characters, no readable text."
        )
        scenes.append(scene)
    result["scenes"] = scenes
    return result


def normalize_object_names(data: dict) -> dict:
    """Translate common English object labels to the Hindi character names used in dialogue."""
    result = dict(data)

    def translate(name: str) -> str:
        key = re.sub(r"[^a-z]+", " ", str(name).casefold()).strip()
        if key in OBJECT_NAME_TRANSLATIONS:
            return OBJECT_NAME_TRANSLATIONS[key]
        for word in key.split():
            if word in OBJECT_NAME_TRANSLATIONS:
                return OBJECT_NAME_TRANSLATIONS[word]
        return str(name).strip()

    scenes = []
    for scene in result.get("scenes", []):
        scene = dict(scene)
        scene["characters"] = [translate(name) for name in scene.get("characters", [])]
        scene["beats"] = [
            {**beat, "speaker": translate(beat.get("speaker", ""))}
            for beat in scene.get("beats", [])
        ]
        scenes.append(scene)
    result["scenes"] = scenes
    return result


def punch_up_story(data: dict, minimum: float, maximum: float, history: list) -> dict:
    try:
        response = nvidia_generate(
            "Punch up this Hindi brainrot skit: make it funnier, shorter, louder, and more absurd. "
            "Keep its exact opening hook and catchphrase, preserve the scene schema, and obey all "
            "dialogue, safety, pacing, caption, ending, and five-second-scene rules. Do not use "
            "animals, real people, celebrities, copyrighted characters, or copied viral memes. "
            "Return only the structured JSON skit.\n"
            f"Current skit: {json.dumps(data, ensure_ascii=False)}",
            max_tokens=4500,
            model_role="flash",
            response_schema=BRAINROT_SCHEMA,
        )
        improved = candidate_script(parse_json(response), data["idea"], data["hook_type"])
        if not validate_brainrot_story(improved, minimum, maximum, history):
            return improved
        LOGGER.warning("Optional brainrot punch-up failed validation; keeping the original script.")
    except Exception as exc:
        LOGGER.warning("Optional brainrot punch-up failed (%s); keeping the original script.", type(exc).__name__)
    return data


def make_story(
    history: list,
    minimum: float,
    maximum: float,
    hook_type: str,
    *,
    attempts: int = 3,
    rejected_ideas: list | None = None,
    feedback: str = "",
    punch_up: bool = True,
) -> dict:
    if rejected_ideas is None:
        rejected_ideas = []
    last_errors = []
    midpoint = (minimum + maximum) / 2
    candidate_durations = [seconds for seconds in range(5, math.ceil(maximum) + 1, 5) if minimum <= seconds <= maximum]
    target_duration = min(candidate_durations, key=lambda seconds: abs(seconds - midpoint)) if candidate_durations else round(midpoint)
    target_scene_count = math.ceil(target_duration / 5)
    target_spoken_words = round(target_duration * 2.5)
    minimum_spoken_words = math.ceil(minimum * 2.5)
    maximum_spoken_words = math.floor(maximum * 2.5)
    for attempt in range(attempts):
        try:
            idea_response = nvidia_generate(
                "Create a fresh Hindi brainrot skit idea using only fictional talking objects. "
                "Write setting, conflict, punchline, hook_line, and catchphrase in colloquial Hindi "
                "using Devanagari script. Do not write those fields in English or Romanized Hindi. "
                "The hook_line must be a loud, natural Hindi line of 2-7 words and match the exact "
                "requested category; use that line as the title idea. The catchphrase must be 2-4 "
                "Hindi words. No animals, humans, public figures, stereotypes, politics, religion, "
                "profanity, sexual content, gore, danger, copyrighted characters, or viral meme characters. "
                "Use this short example only as a format reference; do not reuse its plot or line: "
                f"{json.dumps({**HOOK_IDEA_EXAMPLES[hook_type], 'hook_type': hook_type}, ensure_ascii=False)}\n"
                "Return only JSON.\n"
                f"Required hook category: {hook_type}\n"
                f"Required hook line pattern: {HOOK_REQUIREMENTS[hook_type]}\n"
                f"Rejected ideas: {json.dumps(rejected_ideas[-10:], ensure_ascii=False)}\n"
                f"Previous validation feedback: {feedback}\n"
                "Before returning JSON, verify that every idea text field is Hindi in Devanagari.",
                max_tokens=450,
                model_role="flash",
                response_schema=IDEA_SCHEMA,
            )
            idea = parse_json(idea_response)
        except ValueError as exc:
            last_errors = [f"idea selection was invalid: {exc}"]
            feedback = "; ".join(last_errors)
            LOGGER.warning("Brainrot idea selection failed on attempt %d: %s", attempt + 1, exc)
            continue
        idea_errors = validate_brainrot_idea(idea, hook_type)
        if idea_errors or is_duplicate_idea(idea, history + rejected_ideas):
            last_errors = idea_errors or ["idea is too similar to saved setting/conflict/punchline history"]
            feedback = "; ".join(last_errors)
            LOGGER.warning("Brainrot idea rejected on attempt %d: %s", attempt + 1, feedback)
            rejected_ideas.append(idea)
            continue
        writer_prompt = (
            "अत्यावश्यक भाषा नियम: सभी शीर्षक, वर्णन, पात्रों के नाम और बोले गए संवाद सहज हिंदी "
            "में देवनागरी लिपि में लिखो। अंग्रेज़ी या रोमन हिंदी में संवाद बिल्कुल मत लिखो। चित्र और "
            "वीडियो प्रॉम्प्ट अंग्रेज़ी में हो सकते हैं।\n"
            "Write a Hindi brainrot comedy skit using only fictional talking objects. "
            "Never use animal characters. No narrator, voiceover, setup, intro, moral, recap, "
            "subscribe line, explanations, or realistic logic. Start frame one mid-chaos. Use "
        "only short Hindi object names as characters and speakers: कप, कुर्सी, फ्रिज, चप्पल, "
            "पंखा, टिफिन, आलू, लैपटॉप, चाबी, घड़ी, टोस्टर, चम्मच, दरवाजा, तिजोरी, किताब, बोतल. "
            "Never write a character name or speaker label in English.\n"
            "exactly two punchy dialogue beats per roughly five-second scene, and only one or "
            "two characters speaking in each scene. Each dialogue line must have 2-7 Hindi words. "
            "A new absurd escalation must happen every 2-3 seconds; every beat gets its own "
            "caption_text equal to the spoken line and 1-4 emphasized words. Repeat the supplied "
            f"catchphrase at least three times in different scenes (include scenes 1, {max(2, target_scene_count // 2)}, "
            f"and {target_scene_count - 1}), each time with a changed meaning. Use at least "
            "three brainrot techniques from the allowed list. The final scene's penultimate beat "
            "must be a dumb twist or sharp cut-off, followed by a reaction of at most three words; "
            "do not resolve the story. Use one sfx_hint from the allowed list on some punchline "
            "scenes, but leave others empty. Optional meme_text is allowed once total. Make visual "
            "prompts show the same fictional objects acting out the beats, with no readable text. "
            "No politics, religion, caste, community, stereotypes, real people, celebrities, "
            "copyrighted characters/songs/lyrics, abuse, profanity, sexual content, gore, or "
            "dangerous stunts.\n"
            "हिंदी संवाद की शैली के तीन छोटे नमूने; केवल लय सीखो, पंक्तियाँ या कथानक मत दोहराओ:\n"
            "1) कप: 'मैंने पूरा बैंक खरीद लिया!' कुर्सी: 'अब तिजोरी मेरी है!' कप: 'क्या सेटिंग है!'\n"
            "2) चप्पल: 'तू मेरी कुर्सी क्यों बेच रहा?!' कुर्सी: 'अब ये टिकट काउंटर है!' चप्पल: 'भाई, मुझे टिकट दे!'\n"
            "3) पंखा: 'मेरा वेतन कौन देगा?!' टिफिन: 'मैं बैंक का मालिक हूँ!' पंखा: 'अबे, क्या सेटिंग है!'\n"
            "Return only JSON matching the schema. Use exactly two beats per scene and exactly the "
            f"requested number of scenes. Count the spoken lines: together they must contain "
            f"{minimum_spoken_words}-{maximum_spoken_words} whitespace-separated Hindi words; aim for "
            f"about {target_spoken_words}. The duration depends on this count. Make every ordinary dialogue "
            "line six or seven whitespace-separated Hindi words, including the opening hook when possible. "
            "Only the final reaction may be shorter (two or three words). Never make ordinary lines only "
            "two to four words. Count each line and the total before returning JSON; keep the final total "
            "inside the required range so narration reaches the requested duration.\n"
            f"Selected idea: {json.dumps(idea, ensure_ascii=False)}\n"
            f"Required duration_seconds: {target_duration} (within {minimum:g}-{maximum:g}). "
            f"Use exactly {target_scene_count} scenes, each designed for a five-second clip.\n"
            f"Brainrot techniques: {json.dumps(BRAINROT_TOOLKIT, ensure_ascii=False)}\n"
            f"Sound hints: {json.dumps(SFX_HINTS, ensure_ascii=False)}\n"
            f"Previous validation feedback: {feedback}"
        )
        try:
            response = nvidia_generate(
                writer_prompt,
                max_tokens=4500,
                model_role="flash",
                response_schema=BRAINROT_SCHEMA,
            )
            parsed_story = normalize_object_names(parse_json(response))
            parsed_story = align_story_opening(parsed_story, idea)
            parsed_story = normalize_scene_details(parsed_story, idea)
            parsed_story = limit_sound_effects(parsed_story, math.ceil(target_scene_count / 2))
            data = candidate_script(parsed_story, idea, hook_type)
        except ValueError as exc:
            last_errors = [f"writer response was invalid: {exc}"]
            feedback = "; ".join(last_errors)
            LOGGER.warning("Brainrot writer output failed on attempt %d: %s", attempt + 1, exc)
            rejected_ideas.append(idea)
            continue
        errors_found = validate_brainrot_story(data, minimum, maximum, history + rejected_ideas)
        if errors_found:
            LOGGER.warning("Brainrot quality gate rejected attempt %d: %s", attempt + 1, "; ".join(errors_found))
            last_errors = errors_found
            rejected_ideas.append(idea)
            feedback = "; ".join(errors_found)
            continue
        return punch_up_story(data, minimum, maximum, history) if punch_up else data
    raise BrainrotRejected(
        f"Could not produce a valid brainrot script after {attempts} attempt(s): " + "; ".join(last_errors)
    )


def seconds_of_media(path: Path) -> float:
    result = subprocess.run(
        [env("FFPROBE", "ffprobe"), "-v", "error", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def adjusted_narration_speed(speed: float, duration: float, minimum: float, maximum: float) -> float:
    """Estimate one corrective ElevenLabs speed when a take is just outside the duration band."""
    speed = min(1.2, max(0.7, speed))
    if duration < minimum:
        return min(speed, max(0.7, speed * duration / minimum * 0.98))
    if duration > maximum:
        return max(speed, min(1.2, speed * duration / maximum * 1.02))
    return speed


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


def caption_markup(text: str, emphasis: list[str]) -> str:
    emphasized = {word.casefold() for word in emphasis}
    marked = []
    for token in text.split():
        key = token.casefold().strip("!?.,;:।॥\"'()[]{}")
        safe = token.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
        marked.append(r"{\c&H0000CCFF&\fs76\b1}" + safe + r"{\rCaption}" if key in emphasized else safe)
    return " ".join(marked)


def write_caption_file(data: dict, duration: float, subtitle: Path, beat_seconds: float) -> None:
    events = []
    beats = dialogue_lines(data)
    for index, beat in enumerate(beats):
        start = index * duration / len(beats)
        end = (index + 1) * duration / len(beats)
        events.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,"
            f"{caption_markup(beat['caption_text'], beat['emphasis'])}"
        )
    scene_seconds = duration / len(data["scenes"])
    for index, scene in enumerate(data["scenes"]):
        meme_text = scene.get("meme_text", "").strip()
        if meme_text:
            start = index * scene_seconds
            end = min(start + beat_seconds, duration)
            events.append(
                f"Dialogue: 2,{ass_time(start)},{ass_time(end)},Title,,0,0,0,,"
                f"{meme_text.replace('{', '(').replace('}', ')')}"
            )
    safe_title = data["title"].replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    subtitle.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Title,Arial,74,&H00FFFFFF,&H000000FF,&H001B2440,&H80000000,-1,0,1,4,1,8,80,80,210,1\n"
        "Style: Caption,Arial,70,&H00FFFFFF,&H000000FF,&H001B2440,&H80000000,-1,0,1,5,1,5,100,100,390,1\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f"Dialogue: 1,0:00:00.00,{ass_time(min(beat_seconds, duration))},Title,,0,0,0,,{safe_title}\n"
        + "\n".join(events) + "\n", encoding="utf-8")


def write_pcm_wave(target: Path, seconds: float, sample_at) -> None:
    sample_rate = 44100
    frames = bytearray()
    for index in range(int(sample_rate * seconds)):
        sample = max(-1.0, min(1.0, sample_at(index / sample_rate)))
        frames.extend(struct.pack("<h", int(sample * 32767)))
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)


def create_bouncy_music(target: Path) -> None:
    notes = (392, 494, 587, 494)
    def sample_at(time: float) -> float:
        beat_time = time % 0.25
        note = notes[int(time / 0.25) % len(notes)]
        envelope = min(1.0, beat_time / 0.012, max(0.0, (0.18 - beat_time) / 0.025))
        return 0.12 * envelope * (
            math.sin(2 * math.pi * note * beat_time) + 0.22 * math.sin(4 * math.pi * note * beat_time)
        )
    write_pcm_wave(target, 1.0, sample_at)


def create_sound_effect(hint: str, target: Path) -> None:
    seconds = 0.42 if hint in ("vine-boom hit", "bass drop", "whoosh") else 0.3
    def sample_at(time: float) -> float:
        progress = time / seconds
        envelope = max(0.0, math.sin(math.pi * progress))
        if hint in ("vine-boom hit", "bass drop"):
            frequency, volume = 115 - 70 * progress, 0.75
        elif hint == "whoosh":
            frequency, volume = 250 + 1200 * progress, 0.22
        elif hint in ("record-scratch", "record-stop"):
            frequency, volume = 1450 - 1000 * progress, 0.27
        elif hint == "airhorn blast":
            return 0.25 * envelope * (
                math.sin(2 * math.pi * 440 * time)
                + math.sin(2 * math.pi * 554.37 * time)
                + math.sin(2 * math.pi * 659.25 * time)
            ) / 3
        elif hint == "dramatic gasp":
            frequency, volume = 860 + 240 * math.sin(2 * math.pi * 3 * time), 0.1
        else:
            frequency, volume = 260 - 120 * progress, 0.45
        return volume * envelope * math.sin(2 * math.pi * frequency * time)
    write_pcm_wave(target, seconds, sample_at)


def prepare_dialogue_audio(audio: Path) -> None:
    temporary = audio.with_name(audio.stem + "-cleaned.mp3")
    audio_filter = (
        "silenceremove=start_periods=1:start_duration=0.04:start_threshold=-45dB:"
        "stop_periods=-1:stop_duration=0.08:stop_threshold=-45dB:stop_silence=0.04,"
        "loudnorm=I=-14:TP=-1.5:LRA=11"
    )
    subprocess.run(
        [env("FFMPEG", "ffmpeg"), "-y", "-i", str(audio), "-af", audio_filter,
         "-ar", "44100", "-c:a", "libmp3lame", "-q:a", "3", str(temporary)],
        check=True,
    )
    temporary.replace(audio)


def render(data: dict, audio: Path, clips: list[Path], target: Path) -> None:
    duration = seconds_of_media(audio)
    beat_seconds = duration / (len(clips) * 2)
    subtitle = target.with_suffix(".ass")
    write_caption_file(data, duration, subtitle, beat_seconds)

    # FFmpeg's subtitles filter accepts a POSIX-style path; escape drive colon and apostrophe.
    subtitle_arg = subtitle.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")
    inputs = [env("FFMPEG", "ffmpeg"), "-y"]
    for clip in clips:
        inputs.extend(["-i", str(clip)])
    audio_index = len(clips)
    inputs.extend(["-i", str(audio)])
    music = OUT / "brainrot-bouncy-loop.wav"
    if not music.exists():
        create_bouncy_music(music)
    music_index = audio_index + 1
    inputs.extend(["-stream_loop", "-1", "-i", str(music)])
    chains = []
    shot_duration = duration / (len(clips) * 2)
    shot_labels = []
    for index in range(len(clips)):
        shot_a, shot_b = f"shot{index}a", f"shot{index}b"
        chains.append(
            f"[{index}:v]fps=30,split=2[src{index}a][src{index}b];"
            f"[src{index}a]trim=start=0:duration={shot_duration:.3f},setpts=PTS-STARTPTS,"
            f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            f"setsar=1,format=yuv420p[{shot_a}];"
            f"[src{index}b]trim=start={shot_duration:.3f}:duration={shot_duration:.3f},"
            "setpts=PTS-STARTPTS,scale=1188:2112:force_original_aspect_ratio=increase,"
            "crop=1080:1920:x='(iw-1080)/2+3*sin(t*50)':y='(ih-1920)/2+2*sin(t*43)',"
            f"setsar=1,format=yuv420p[{shot_b}]"
        )
        shot_labels.extend((f"[{shot_a}]", f"[{shot_b}]"))
    chains.append(
        f"{''.join(shot_labels)}concat=n={len(shot_labels)}:v=1:a=0,"
        f"subtitles='{subtitle_arg}'[v]"
    )
    chains.append(f"[{music_index}:a]volume=0.07[music]")
    audio_labels = [f"[{audio_index}:a]", "[music]"]
    next_input = music_index + 1
    for index, scene in enumerate(data["scenes"]):
        hint = scene.get("sfx_hint", "")
        if not hint:
            continue
        effect = OUT / ("sfx-" + re.sub(r"[^a-z0-9]+", "-", hint.lower()).strip("-") + ".wav")
        if not effect.exists():
            create_sound_effect(hint, effect)
        inputs.extend(["-i", str(effect)])
        delay_ms = int((index * 2 + 1) * beat_seconds * 1000)
        label = f"sfx{index}"
        chains.append(f"[{next_input}:a]adelay={delay_ms}|{delay_ms},volume=0.45[{label}]")
        audio_labels.append(f"[{label}]")
        next_input += 1
    chains.append(
        f"{''.join(audio_labels)}amix=inputs={len(audio_labels)}:duration=first:dropout_transition=0,"
        "alimiter=limit=0.95[a]"
    )
    subprocess.run(
        inputs + ["-filter_complex", ";".join(chains), "-map", "[v]", "-map", "[a]",
         "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "fast", "-crf", "21",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
         "-movflags", "+faststart", str(target)],
        check=True,
    )
    subtitle.unlink(missing_ok=True)


def hf_base_url(space_id: str) -> str:
    return "https://" + space_id.replace("/", "-") + ".hf.space"


def hf_headers() -> dict:
    token = env("HF_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def hf_api_call(space_id: str, api_name: str, values: list):
    base_url = hf_base_url(space_id)
    headers = hf_headers()
    payload = json.dumps({"data": values}).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/gradio_api/call/" + urllib.parse.quote(api_name), data=payload,
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        event_id = json.loads(response.read().decode("utf-8"))["event_id"]
    result_request = urllib.request.Request(
        base_url + "/gradio_api/call/" + urllib.parse.quote(api_name) + "/" + urllib.parse.quote(event_id),
        headers={key: value for key, value in headers.items() if key != "Content-Type"},
    )
    with urllib.request.urlopen(result_request, timeout=300) as response:
        events = response.read().decode("utf-8", errors="replace")
    event_name = ""
    for line in events.splitlines():
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data = json.loads(line.partition(":")[2].strip())
            if event_name == "error":
                raise RuntimeError(f"Hugging Face {api_name} failed: {data}")
            if event_name == "complete":
                return data
    raise RuntimeError(f"Hugging Face {api_name} ended without returning a result.")


def save_hf_file(result, target: Path, space_id: str, kind: str) -> None:
    file_data = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(file_data, dict):
        path = file_data.get("path")
        url = file_data.get("url")
        if path and Path(path).is_file():
            shutil.copyfile(Path(path), target)
        else:
            if url and "/gradio_a/gradio_api/" in url:
                url = url.replace("/gradio_a/gradio_api/", "/gradio_api/", 1)
            if not url and path:
                url = hf_base_url(space_id) + "/gradio_api/file=" + urllib.parse.quote(path, safe="/")
            if not url:
                raise RuntimeError(f"Hugging Face returned no downloadable {kind} file.")
            request = urllib.request.Request(url, headers=hf_headers())
            with urllib.request.urlopen(request, timeout=180) as response:
                target.write_bytes(response.read())
    elif isinstance(file_data, str) and file_data.startswith(("https://", "http://")):
        url = file_data.replace("/gradio_a/gradio_api/", "/gradio_api/", 1)
        with urllib.request.urlopen(urllib.request.Request(url, headers=hf_headers()), timeout=180) as response:
            target.write_bytes(response.read())
    elif isinstance(file_data, (str, Path)) and Path(file_data).is_file():
        shutil.copyfile(Path(file_data), target)
    else:
        raise RuntimeError(f"Hugging Face returned no usable {kind} file.")
    content = target.read_bytes() if target.is_file() else b""
    valid = content.startswith(b"\x89PNG\r\n\x1a\n") if kind == "image" else len(content) > 8 and content[4:8] == b"ftyp"
    if not valid:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Hugging Face Space did not return a valid {kind} file.")


def hf_upload_image(space_id: str, image: Path) -> dict:
    boundary = "----ShortsMaker" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{image.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = hf_headers()
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    request = urllib.request.Request(hf_base_url(space_id) + "/gradio_api/upload", data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        uploaded = json.loads(response.read().decode("utf-8"))
    remote_path = uploaded[0] if isinstance(uploaded, list) else uploaded["path"]
    return {"path": remote_path, "orig_name": image.name, "meta": {"_type": "gradio.FileData"}}


def hf_generate_image(prompt: str, seed: int):
    return hf_api_call(env("HF_IMAGE_SPACE_ID", "saasr9/zerogpu-gen"), "generate_image", [prompt, 4, 0.0, 512, 1024, seed])


def hf_generate_video(image: Path, prompt: str, seed: int):
    space_id = env("HF_VIDEO_SPACE_ID", "zerogpu-aoti/wan2-2-fp8da-aoti-faster")
    remote_image = hf_upload_image(space_id, image)
    negative_prompt = "static frame, frozen motion, flicker, blurry, distorted faces, extra limbs, duplicate people, text, subtitles, watermark"
    return hf_api_call(space_id, "generate_video", [remote_image, prompt, 4, negative_prompt, 5.0, 1.0, 1.0, seed, False])


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class _OpenRouterRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        parsed = urllib.parse.urlsplit(new_url)
        if parsed.scheme != "https" or parsed.netloc != "openrouter.ai":
            raise RuntimeError("Refusing to forward the OpenRouter API key to a redirected host.")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def openrouter_request(path: str, method: str = "GET", payload: dict | None = None, timeout: float = 60, raw: bool = False):
    token = env("OPENROUTER_API_KEY")
    if not token:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or GitHub Actions secrets.")
    if urllib.parse.urlsplit(path).scheme:
        url = path
    elif path.lstrip("/").startswith("api/v1/"):
        url = urllib.parse.urljoin("https://openrouter.ai/", path.lstrip("/"))
    else:
        url = urllib.parse.urljoin(OPENROUTER_BASE_URL + "/", path.lstrip("/"))
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "openrouter.ai":
        raise RuntimeError("OpenRouter returned an unsafe API URL.")
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        opener = urllib.request.build_opener(_OpenRouterRedirectHandler())
        with opener.open(request, timeout=timeout) as response:
            result = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        try:
            detail = json.loads(detail).get("error", {}).get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise OpenRouterError(f"OpenRouter request failed ({exc.code}): {detail}", exc.code) from exc
    if raw:
        return result
    try:
        return json.loads(result.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenRouterError("OpenRouter returned invalid JSON.") from exc


def check_openrouter_video_api() -> None:
    model_id = env("OPENROUTER_VIDEO_MODEL", OPENROUTER_DEFAULT_MODEL)
    catalog = openrouter_request("/videos/models")
    models = catalog.get("data", [])
    model = next((item for item in models if item.get("id") == model_id), None)
    if model is None:
        raise RuntimeError(f"OpenRouter key works, but video model {model_id!r} was not found in the current catalog.")
    capabilities = (
        (5, model.get("supported_durations", []), "5-second duration"),
        ("480p", model.get("supported_resolutions", []), "480p resolution"),
        ("9:16", model.get("supported_aspect_ratios", []), "9:16 aspect ratio"),
    )
    unsupported = [label for value, supported, label in capabilities if value not in supported]
    if unsupported:
        raise RuntimeError(f"OpenRouter key works, but {model_id} does not support: {', '.join(unsupported)}.")
    print(f"OpenRouter API key works. {model_id} supports 5-second, 480p, 9:16 video. No video was generated.")


def openrouter_generate_video(prompt: str, target: Path) -> None:
    model_id = env("OPENROUTER_VIDEO_MODEL", OPENROUTER_DEFAULT_MODEL)
    job = openrouter_request("/videos", method="POST", payload={
        "model": model_id,
        "prompt": prompt,
        "duration": 5,
        "resolution": "480p",
        "aspect_ratio": "9:16",
        "generate_audio": False,
    })
    job_id = job.get("id")
    status = job.get("status")
    polling_url = job.get("polling_url") or f"/videos/{urllib.parse.quote(str(job_id), safe='')}"
    if not job_id or not status:
        raise RuntimeError("OpenRouter did not return a video job ID and status.")
    timeout = float(env("OPENROUTER_VIDEO_TIMEOUT_SECONDS", "900"))
    interval = max(1.0, float(env("OPENROUTER_VIDEO_POLL_SECONDS", "30")))
    deadline = time.monotonic() + timeout
    while status not in {"completed", "failed", "cancelled", "expired"}:
        if status not in {"pending", "in_progress"}:
            raise RuntimeError(f"OpenRouter returned unexpected video status {status!r}.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"OpenRouter video job {job_id} exceeded {timeout:g} seconds.")
        time.sleep(min(interval, remaining))
        try:
            job = openrouter_request(polling_url, timeout=60)
            status = job.get("status")
        except (urllib.error.URLError, TimeoutError) as exc:
            LOGGER.warning("OpenRouter status check failed temporarily (%s); retrying the same job.", type(exc).__name__)
            continue
        except OpenRouterError as exc:
            if exc.status not in {408, 429, 500, 502, 503, 504}:
                raise
            LOGGER.warning("OpenRouter status check returned HTTP %s; retrying the same job.", exc.status)
    if status != "completed":
        raise RuntimeError(f"OpenRouter video job {job_id} ended with {status}: {job.get('error', 'no details')}.")
    content = openrouter_request(f"/videos/{urllib.parse.quote(str(job_id), safe='')}/content?index=0", raw=True, timeout=180)
    if len(content) <= 8 or content[4:8] != b"ftyp":
        raise RuntimeError("OpenRouter did not return a valid MP4 video.")
    target.write_bytes(content)


def create_pan_zoom_clip(image: Path, target: Path, duration: float = 5.0) -> None:
    frames = int(duration * 30)
    video_filter = (
        "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
        f"zoompan=z='min(zoom+{0.12 / frames:.8f},1.12)':"
        f"x='(iw-iw/zoom)*on/{frames}':y='(ih-ih/zoom)/2':"
        f"d={frames}:s=1080x1920:fps=30,setsar=1,format=yuv420p"
    )
    subprocess.run(
        [env("FFMPEG", "ffmpeg"), "-y", "-loop", "1", "-i", str(image), "-vf", video_filter,
         "-t", str(duration), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        check=True,
    )


def create_video_clips(prompts: list[dict], stamp: str) -> list[Path]:
    video_space = env("HF_VIDEO_SPACE_ID", "zerogpu-aoti/wan2-2-fp8da-aoti-faster")
    for index, scene in enumerate(prompts):
        clip = OUT / f"{stamp}-motion-scene{index + 1}.mp4"
        image = OUT / f"{stamp}-scene{index + 1}.png"
        motion = OUT / f"{stamp}-scene{index + 1}-motion.mp4"
        if clip.exists():
            continue
        if not image.exists():
            seed = (int(stamp) * 1009 + index) % (2**31)
            result = hf_generate_image(scene["image_prompt"], seed)
            save_hf_file(result, image, env("HF_IMAGE_SPACE_ID", "saasr9/zerogpu-gen"), "image")
        seed = (int(stamp) * 1009 + index) % (2**31)
        if not motion.exists():
            try:
                try:
                    result = hf_generate_video(image, scene["video_prompt"], seed)
                    save_hf_file(result, motion, video_space, "video")
                except Exception as exc:
                    if not env("OPENROUTER_API_KEY"):
                        raise
                    LOGGER.warning(
                        "Hugging Face Wan video generation failed for scene %d (%s); trying OpenRouter.",
                        index + 1,
                        type(exc).__name__,
                    )
                    prompt = f"{scene['image_prompt']} Motion: {scene['video_prompt']}"
                    openrouter_generate_video(prompt, motion)
            except Exception as exc:
                LOGGER.warning(
                    "Video generation failed for scene %d (%s); using its still image.",
                    index + 1,
                    type(exc).__name__,
                )
                create_pan_zoom_clip(image, clip)
                continue
        duration = 5.0
        video_filter = (
            "[0:v]fps=30,trim=duration=5,setpts=PTS-STARTPTS,"
            "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            "setsar=1,format=yuv420p[v]"
        )
        subprocess.run(
            [env("FFMPEG", "ffmpeg"), "-y", "-i", str(motion), "-filter_complex", video_filter,
             "-map", "[v]", "-t", str(duration), "-an", "-c:v", "libx264", "-preset", "fast",
             "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip)],
            check=True,
        )
    return [OUT / f"{stamp}-motion-scene{index + 1}.mp4" for index in range(len(prompts))]


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


def upload(video: Path, data: dict) -> str:
    description = data.get("description", "").strip() + "\n\nAI-generated Hindi comedy dialogue."
    description += "\n\n#Shorts #HindiComedy"
    body = {
        "snippet": {
            "title": data["title"][:100], "description": description[:5000],
            "tags": data.get("tags", [])[:20], "categoryId": "24",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and publish a Hindi brainrot comedy Short.")
    parser.add_argument("--auth-only", action="store_true", help="Complete YouTube OAuth without generating a video.")
    parser.add_argument(
        "--check-video-api",
        action="store_true",
        help="Check OpenRouter key and configured video model without generating or paying for a video.",
    )
    parser.add_argument(
        "--script-only",
        action="store_true",
        help="Generate and validate a scene script without ElevenLabs, Hugging Face, or YouTube calls.",
    )
    parser.add_argument(
        "--test-upload",
        action="store_true",
        help="Generate and publish one test video using a unique slot separate from the daily schedule.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env_file()
    if sum((args.auth_only, args.script_only, args.check_video_api, args.test_upload)) > 1:
        parser.error("--auth-only, --script-only, --check-video-api, and --test-upload cannot be used together.")
    if args.check_video_api:
        check_openrouter_video_api()
        return
    if args.auth_only:
        youtube_client()
        print("YouTube OAuth is ready.")
        return

    state_file = ROOT / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {"next_video": 0, "topics": []}
    index = state.get("next_video", state.get("next_topic", 0))
    history = state.get("topics", [])
    minimum = float(env("MIN_VIDEO_SECONDS", "30"))
    maximum = float(env("MAX_VIDEO_SECONDS", "50"))
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Set positive MIN_VIDEO_SECONDS and MAX_VIDEO_SECONDS with maximum >= minimum.")
    hook_type = HOOK_TYPES[index % len(HOOK_TYPES)]
    OUT.mkdir(exist_ok=True)

    if args.script_only:
        data = make_story(
            history, minimum, maximum, hook_type,
            punch_up=env("BRAINROT_PUNCH_UP", "true").lower() == "true",
        )
        script_path = OUT / f"script-only-{datetime.now(INDIA_TIME).strftime('%Y%m%d-%H%M%S')}.json"
        script_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved validated scenes to {script_path}")
        return

    slot_key = upload_slot_key(manual_test=args.test_upload)
    if not args.test_upload and slot_already_uploaded(state, slot_key):
        print(f"Skipping: state.json already records an upload for {slot_key}.")
        return

    youtube_client()
    if not env("HF_TOKEN"):
        raise RuntimeError("Add a free Hugging Face read token as HF_TOKEN before generating scenes.")
    stamp = f"{index + 1:04d}"
    audio = OUT / f"{stamp}.mp3"
    video = OUT / f"{stamp}.mp4"
    metadata = OUT / f"{stamp}.json"
    rejected_ideas = []
    feedback = ""
    data = None
    duration = 0.0
    current_hook_type = hook_type
    max_generations = len(HOOK_TYPES)
    for attempt in range(max_generations):
        try:
            candidate = make_story(
                history, minimum, maximum, current_hook_type, attempts=1,
                rejected_ideas=rejected_ideas, feedback=feedback, punch_up=False,
            )
        except BrainrotRejected as exc:
            feedback = str(exc)
            LOGGER.warning("Brainrot story rejected on generation %d/%d: %s", attempt + 1, max_generations, exc)
            current_hook_type = HOOK_TYPES[(HOOK_TYPES.index(current_hook_type) + 1) % len(HOOK_TYPES)]
            continue
        candidate["script"] = spoken_script(candidate)
        narration_speed = float(env("NARRATION_SPEED", "1.15"))
        create_narration(candidate["script"], audio, speed=narration_speed)
        prepare_dialogue_audio(audio)
        duration = seconds_of_media(audio)
        if duration < minimum or duration > maximum:
            corrected_speed = adjusted_narration_speed(narration_speed, duration, minimum, maximum)
            if not math.isclose(corrected_speed, narration_speed, abs_tol=0.01):
                LOGGER.info(
                    "Narration was %.1fs at speed %.2f; retrying the same script at speed %.2f.",
                    duration,
                    narration_speed,
                    corrected_speed,
                )
                create_narration(candidate["script"], audio, speed=corrected_speed)
                prepare_dialogue_audio(audio)
                duration = seconds_of_media(audio)
        if minimum <= duration <= maximum:
            data = candidate
            break
        rejected_ideas.append(candidate["idea"])
        feedback = f"Measured narration was {duration:.1f}s; target {minimum:g}-{maximum:g}s."
        LOGGER.warning("Brainrot narration failed duration check on generation %d/%d: %s", attempt + 1, max_generations, feedback)
        audio.unlink(missing_ok=True)
    if data is None:
        raise RuntimeError(
            f"Could not generate narration in the {minimum:g}-{maximum:g} second range "
            f"after {max_generations} attempts."
        )

    if env("BRAINROT_PUNCH_UP", "true").lower() == "true":
        improved = punch_up_story(data, minimum, maximum, history)
        if improved is not data:
            punchup_audio = audio.with_name(audio.stem + "-punchup.mp3")
            create_narration(spoken_script(improved), punchup_audio, speed=float(env("NARRATION_SPEED", "1.15")))
            prepare_dialogue_audio(punchup_audio)
            punchup_duration = seconds_of_media(punchup_audio)
            if minimum <= punchup_duration <= maximum:
                punchup_audio.replace(audio)
                data, duration = improved, punchup_duration
            else:
                punchup_audio.unlink(missing_ok=True)
                LOGGER.warning("Punch-up narration was outside duration range; keeping the validated original.")

    metadata.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    clips = create_video_clips(data["scenes"], stamp)
    render(data, audio, clips, video)
    rendered_duration = seconds_of_media(video)
    if not minimum <= rendered_duration <= maximum:
        raise RuntimeError(f"Rendered video is {rendered_duration:.1f}s; refusing to upload outside {minimum:.0f}-{maximum:.0f}s.")
    video_id = upload(video, data)
    history.append({"idea": data["idea"]})
    state["next_video"] = index + 1
    state["topics"] = history
    state["last_video_id"] = video_id
    uploaded_slots = state.setdefault("uploaded_slots", {})
    if not isinstance(uploaded_slots, dict):
        uploaded_slots = state["uploaded_slots"] = {}
    uploaded_slots[slot_key] = video_id
    save_state(state_file, state)
    print(f"Saved {video} ({rendered_duration:.1f} seconds)")
    print(f"Published https://youtube.com/shorts/{video_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ShortsMaker failed: {exc}", file=sys.stderr)
        raise
