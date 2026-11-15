# ShortsMaker

ShortsMaker creates short, fast Hindi brainrot comedy skits with fictional talking-object characters. Groq-hosted GPT-OSS models invent ideas and write two dialogue beats per scene. ElevenLabs voices the dialogue, Hugging Face generates still images and Wan 2.2 motion clips, and FFmpeg edits, captions, and mixes the finished video before YouTube upload. OpenRouter is an optional motion-video fallback.

## Setup once

1. Install Python 3.11+ and FFmpeg (including ffprobe).
2. Create a Google Cloud Desktop OAuth client, enable YouTube Data API v3, and save its JSON as client_secret.json.
3. Add GROQ_API_KEY, ELEVENLABS_API_KEY, and a Hugging Face read token (HF_TOKEN) to .env. OPENROUTER_API_KEY is optional.
4. Install packages with python -m pip install -r requirements.txt.
5. Run python shorts_maker.py --auth-only once and complete Google sign-in.
6. Run python shorts_maker.py to make and publish a Short.

Keep .env, OAuth JSON, token files, and API tokens out of Git.

Story generation uses Groq's OpenAI-compatible Chat Completions API. The default model is `openai/gpt-oss-20b`, with `openai/gpt-oss-120b` as fallback; both currently support strict JSON Schema output. Set `GROQ_MODEL` and `GROQ_FALLBACK_MODEL` to change them. Groq account pricing, model availability, and rate limits apply.

The Hugging Face Wan 2.2 Space is tried first for five-second motion clips, with one retry for transient errors. If it fails and OPENROUTER_API_KEY is configured, OpenRouter is tried as a fallback. Every scene must pass a sustained-motion check; if both video providers fail, the run stops before upload rather than substituting a still image.

To check the OpenRouter key and model capabilities without generating video or spending video credits, run:

    python shorts_maker.py --check-video-api

This confirms the key can list models and that the selected model supports the configured five-second vertical 480p clip. Add OPENROUTER_API_KEY to GitHub Actions repository secrets only if you want it available as a fallback after Wan fails.

## Generate a script without rendering or publishing

Run:

    python shorts_maker.py --script-only

This calls Groq to create and validate an idea and its scene data, then saves a timestamped JSON file under output/. It makes no ElevenLabs, Hugging Face, FFmpeg, or YouTube calls. BRAINROT_PUNCH_UP defaults to true and uses one additional Groq call to improve a passing script; set it to false to skip that call.

## Daily publishing with GitHub Actions

The workflow runs at 9:00 AM and 7:00 PM India time without relying on your computer being awake. In repository Settings → Secrets and variables → Actions, add:

- GROQ — Groq API key (used as `GROQ_API_KEY` by GitHub Actions)
- ELEVENLABS_API_KEY
- HF_TOKEN
- OPENROUTER_API_KEY — optional; used as a fallback if the Hugging Face Wan 2.2 Space fails
- YOUTUBE_CLIENT_SECRET_JSON — contents of client_secret.json
- YOUTUBE_TOKEN_JSON — contents of youtube_token.json
- Optionally, both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for failure notices

Runs are serialized and stop after 90 minutes. The timeout allows up to ten five-second scenes at the configured 50-second maximum. A completed upload is recorded against its India-time slot in state.json; a retry for an already published slot is skipped. The workflow commits updated idea history and slot state. Finished MP4 files are available as three-day workflow artifacts. Google OAuth projects in Testing may expire authorizations after seven days.

To publish a one-off test video, choose **Run workflow** and enable `test_upload`. It generates and publicly uploads a new video using a separate manual-test state key, so an existing daily slot will not cause it to skip or be overwritten.

Each generated still is only an input frame for Wan image-to-video. Every scene must produce a moving MP4; a frame-difference check rejects static results. If Wan fails, OpenRouter is tried only when its secret is configured. Otherwise the run stops before upload. No pan/zoom still is published as animation. Hugging Face ZeroGPU has a small free daily quota, so two fully animated videos every day may exceed free usage.

## Format and quality controls

The default duration range is 30–50 seconds. Set MIN_VIDEO_SECONDS and MAX_VIDEO_SECONDS to change it. Each scene has two short dialogue beats, one consistent cast of fictional objects, captions with emphasized words, generated visual prompts, and an optional sparse sound-effect hint. The writer returns a compact, duration-specific scene schema; the app derives captions and English visual prompts from the dialogue without rewriting the generated lines. The writer must start with a rotating hook, repeat a catchphrase, escalate absurdly, and end with a twist or cut-off plus a short reaction. A local quality gate rejects narrator-style openings, invalid hooks, weak beat density, repeated ideas, unsafe terms, incorrect scene counts, and weak endings; it retries each hook category twice and can try all five before giving up.

FFmpeg preserves each animated clip as one uninterrupted scene, adds captions, trims silence between dialogue lines, applies loudness normalization, and mixes a quiet synthesized bouncy loop with sparse synthesized sound effects. Noto Sans Devanagari is installed in the workflow so Hindi captions render correctly. No new Python dependencies are required.

Run the offline test suite with python -m unittest discover -s tests -v.
