# ShortsMaker

Every day, the script researches a fresh historical subject, writes an original informative narration, generates cinematic Kling 3.0 scenes, adds AI voice and captions, saves the finished video in `output/`, and uploads it publicly to YouTube. No topic list or per-video approval is needed. The narration and final video are checked to be at least 90 seconds; typical output is around 1:40–2:00.

## Setup once

1. Install Python 3.11+ and FFmpeg (including `ffprobe`).
2. In Google Cloud, enable YouTube Data API v3, create a Desktop OAuth client, and save its JSON here as `client_secret.json`.
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, and `KLING_API_KEY`.
4. Install packages with `python -m pip install -r requirements.txt`.
5. Run `python shorts_maker.py --auth-only` and sign in to Google when the browser opens. This saves the OAuth token without generating a video. Then run `python shorts_maker.py` to create and publish the first Short; later scheduled runs use the saved token.
6. Register the daily task: `powershell -ExecutionPolicy Bypass -File .\setup_daily_task.ps1 -Time "09:00"`.

## Run daily in GitHub Actions

The cloud workflow runs at 9:17 AM India time every day. It does not depend on your computer being awake. The finished MP4 is saved as a downloadable workflow artifact for three days, and the workflow commits the topic history so later videos avoid repeating topics.

In the GitHub repository, open **Settings → Secrets and variables → Actions** and add these repository secrets:

- `GEMINI_API_KEY`
- `ELEVENLABS_API_KEY`
- `KLING_API_KEY`
- `YOUTUBE_CLIENT_SECRET_JSON` — the full contents of `client_secret.json`
- `YOUTUBE_TOKEN_JSON` — the full contents of `youtube_token.json`

After the workflow is pushed, open **Actions → Daily historical video → Run workflow** to start a run manually. Scheduled and manual runs both publish publicly. Keep the repository secrets private and never commit `.env`, OAuth JSON, or token files.

Google OAuth projects in Testing expire authorizations after seven days. Until the project is moved to production and meets any verification Google requires, replace `YOUTUBE_TOKEN_JSON` with a newly authorized `youtube_token.json` at least every seven days. GitHub Actions may start a scheduled run later than its scheduled time, and private-repository compute usage can exceed the plan's included amount.

`CONTENT_GENRE` defaults to `historical and informative`. Change it in `.env` to another genre if you want. Previous topics are recorded in `state.json` and passed to the model to reduce repeats. The finished MP4, voice track, story metadata, and generated scene clips remain in `output/`.

The default video generation quality is 1080p. Set `KLING_RESOLUTION=4k` for 4K source clips; the final Short is rendered at 1080x1920. Kling charges by generated duration, so a longer story and 4K setting use more credits. The task resumes submitted Kling jobs after interruption and does not advance to the next story unless upload succeeds.

The video upload is public and runs without per-video approval after the one-time Google sign-in. Synthetic-media disclosure is enabled by default for generated cinematic scenes. The cloud schedule does not depend on your computer being awake. Gemini 2.5 Flash is set as the script model; the Gemini API free tier has usage caps and can return quota errors. ElevenLabs usage consumes plan credits; Kling video generation requires available API resource credits, which are separate from free website credits.
