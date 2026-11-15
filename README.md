# ShortsMaker

ShortsMaker creates short, fast Hindi brainrot comedy skits with fictional talking-object characters. Gemini invents an idea, writes two dialogue beats per scene, and supplies scene visuals and caption directions. ElevenLabs voices the dialogue, Hugging Face generates still images and Wan motion clips, and FFmpeg edits, captions, and mixes the finished video before YouTube upload.

## Setup once

1. Install Python 3.11+ and FFmpeg (including ffprobe).
2. Create a Google Cloud Desktop OAuth client, enable YouTube Data API v3, and save its JSON as client_secret.json.
3. Add GEMINI_API_KEY, ELEVENLABS_API_KEY, and a Hugging Face read token (HF_TOKEN) to .env.
4. Install packages with python -m pip install -r requirements.txt.
5. Run python shorts_maker.py --auth-only once and complete Google sign-in.
6. Run python shorts_maker.py to make and publish a Short.

Keep .env, OAuth JSON, token files, and API tokens out of Git.

## Generate a script without rendering or publishing

Run:

    python shorts_maker.py --script-only

This calls Gemini to create and validate an idea and its scene data, then saves a timestamped JSON file under output/. It makes no ElevenLabs, Hugging Face, FFmpeg, or YouTube calls. BRAINROT_PUNCH_UP defaults to true and uses one additional Gemini call to improve a passing script; set it to false to skip that call.

## Daily publishing with GitHub Actions

The workflow runs at 9:00 AM and 7:00 PM India time without relying on your computer being awake. In repository Settings → Secrets and variables → Actions, add:

- GEMINI_API_KEY
- ELEVENLABS_API_KEY
- HF_TOKEN
- YOUTUBE_CLIENT_SECRET_JSON — contents of client_secret.json
- YOUTUBE_TOKEN_JSON — contents of youtube_token.json
- Optionally, both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID for failure notices

Runs are serialized and stop after 90 minutes. The timeout allows up to ten five-second scenes at the configured 50-second maximum. A completed upload is recorded against its India-time slot in state.json; a retry for an already published slot is skipped. The workflow commits updated idea history and slot state. Finished MP4 files are available as three-day workflow artifacts. Google OAuth projects in Testing may expire authorizations after seven days.

If Hugging Face video generation fails, FFmpeg makes a five-second zoom/pan clip from that scene's generated still image. Image generation still requires the Hugging Face Space. No stock image search, Wikimedia fallback, source excerpt check, or image attribution is used.

## Format and quality controls

The default duration range is 30–50 seconds. Set MIN_VIDEO_SECONDS and MAX_VIDEO_SECONDS to change it. Each scene has two short dialogue beats, at most two fictional object characters, captions with emphasized words, and an optional sparse sound-effect hint. The writer must start with a rotating hook, repeat a catchphrase, escalate absurdly, and end with a twist or cut-off plus a short reaction. A local quality gate rejects narrator-style openings, long dialogue lines, invalid hooks, weak beat density, repeated ideas, unsafe terms, missing catchphrases, incorrect scene counts, and weak endings; it regenerates up to three times.

FFmpeg splits each motion clip into two hard-cut shots with a punch-in and slight shake, trims silence between dialogue lines, applies loudness normalization, and mixes a quiet synthesized bouncy loop with sparse synthesized sound effects. No new Python dependencies are required.

Run the offline test suite with python -m unittest discover -s tests -v.
