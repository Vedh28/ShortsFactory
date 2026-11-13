# ShortsMaker

ShortsMaker creates a researched historical explainer each day, turns it into a narrated vertical video of at least 90 seconds, saves the MP4 as a workflow artifact, and uploads it publicly to YouTube. Hugging Face SDXL-Turbo generates illustrated scenes; FFmpeg animates each scene with a slow pan and zoom and adds captions.

## Setup once

1. Install Python 3.11+ and FFmpeg (including `ffprobe`).
2. In Google Cloud, enable YouTube Data API v3, create a Desktop OAuth client, and save its JSON as `client_secret.json`.
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, and `HF_TOKEN`.
4. Install packages with `python -m pip install -r requirements.txt`.
5. Run `python shorts_maker.py --auth-only` and complete Google sign-in. Then run `python shorts_maker.py` to create and publish the first Short.

Create a Hugging Face user access token with read access at [Hugging Face token settings](https://huggingface.co/settings/tokens). Keep it private.

## Run daily in GitHub Actions

The cloud workflow runs at 9:17 AM India time and does not depend on your computer being awake. In the repository, open **Settings → Secrets and variables → Actions** and add:

- `GEMINI_API_KEY`
- `ELEVENLABS_API_KEY`
- `HF_TOKEN`
- `YOUTUBE_CLIENT_SECRET_JSON` — the full contents of `client_secret.json`
- `YOUTUBE_TOKEN_JSON` — the full contents of `youtube_token.json`

The workflow stores each finished MP4 as a downloadable artifact for three days and saves topic history so later videos avoid repeats. Scheduled and manual runs publish publicly. Google OAuth projects in Testing can expire authorizations after seven days; refresh `YOUTUBE_TOKEN_JSON` if a scheduled upload starts failing for expired credentials.

The image endpoint uses Hugging Face ZeroGPU. Hugging Face currently documents 5 GPU minutes per day for free accounts; generation time and queue availability can vary, and this community Space may change. If image generation fails or the quota runs out, that day's video will not upload. The video scenes are AI illustrations with camera motion, not generated continuous-action footage. Gemini selects topics and writes narration using Wikipedia article excerpts; ElevenLabs creates narration audio. Their free-tier limits can also affect runs.

`CONTENT_GENRE` defaults to `historical and informative`; `MIN_VIDEO_SECONDS` defaults to `90`. Outputs, scene images, scene MP4s, story metadata, and narration are stored in `output/` during local runs. Cloud runs expose the finished MP4 as an artifact.

Keep `.env`, OAuth JSON, token files, and API tokens out of Git. The video is publicly uploaded without per-video approval after the one-time Google sign-in.
