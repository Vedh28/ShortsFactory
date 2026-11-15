# ShortsMaker

ShortsMaker creates a researched historical explainer each day, turns it into a narrated vertical video between 50 and 60 seconds, saves the MP4 as a workflow artifact, and uploads it publicly to YouTube. Hugging Face SDXL-Turbo creates historical starting frames; Wan 2.2 image-to-video animates people and actions in four scenes. FFmpeg extends each five-second motion clip with a forward/reverse slow-motion pass and adds captions.

If Hugging Face video generation fails, FFmpeg creates a 15-second zoom/pan scene from its still image. If image generation also fails, the workflow searches Wikimedia Commons for a CC0, public-domain, or CC BY image; it saves its author/license/source metadata and adds attribution to the YouTube description. Narration claims are checked against fetched excerpts after any duration rewrite; unsupported narration triggers one new-topic retry. Runs record successful upload slots in `state.json` and skip a slot already published that day.

## Setup once

1. Install Python 3.11+ and FFmpeg (including `ffprobe`).
2. In Google Cloud, enable YouTube Data API v3, create a Desktop OAuth client, and save its JSON as `client_secret.json`.
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, and a free Hugging Face read token as `HF_TOKEN`.
4. Install packages with `python -m pip install -r requirements.txt`.
5. Run `python shorts_maker.py --auth-only` and complete Google sign-in. Then run `python shorts_maker.py` to create and publish the first Short.

Create a free Hugging Face user access token with read access at [Hugging Face token settings](https://huggingface.co/settings/tokens). It is needed to use your personal ZeroGPU quota for the moving video scenes; keep it private.

## Run daily in GitHub Actions

The cloud workflow runs at 9:00 AM and 7:00 PM India time every day and does not depend on your computer being awake. In the repository, open **Settings → Secrets and variables → Actions** and add:

- `GEMINI_API_KEY`
- `ELEVENLABS_API_KEY`
- `HF_TOKEN` — a free Hugging Face read token
- `YOUTUBE_CLIENT_SECRET_JSON` — the full contents of `client_secret.json`
- `YOUTUBE_TOKEN_JSON` — the full contents of `youtube_token.json`
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` — optional; both are needed for failure notifications

The workflow runs one job at a time, stops a job after 60 minutes, stores each finished MP4 as a downloadable artifact for three days, and saves topic history so later videos avoid repeats. If both Telegram secrets are set, a failed run sends the Actions run link; if either is missing, that notification step is skipped. Scheduled and manual runs publish publicly. Google OAuth projects in Testing can expire authorizations after seven days; refresh `YOUTUBE_TOKEN_JSON` if a scheduled upload starts failing for expired credentials.

The image and video endpoints use community Hugging Face ZeroGPU Spaces. A free account currently has five GPU minutes per 24 hours, shared across both daily runs; two videos may exceed that quota, in which case a run will not upload. Queue time or Space availability can also vary. Gemini selects topics and writes narration from Wikipedia excerpts; ElevenLabs creates the voice track. Their free-tier limits can affect runs too.

`CONTENT_GENRE` defaults to `historical and informative`; `MIN_VIDEO_SECONDS` and `MAX_VIDEO_SECONDS` default to `50` and `60`. Outputs, generated frames, moving scene clips, story metadata, and narration are stored in `output/` during local runs. Cloud runs expose the finished MP4 as an artifact.

Keep `.env`, OAuth JSON, token files, and API tokens out of Git. The video is publicly uploaded without per-video approval after the one-time Google sign-in.
