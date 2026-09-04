# voice-dub

CLI for translating timed captions and dubbing audio or video with [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox). It clones a permitted reference voice, aligns generated speech to real waveform activity, and leaves the video stream unchanged.

## Requirements

- Python 3.10–3.13
- FFmpeg
- Apple Silicon, NVIDIA GPU, or CPU
- Internet access and several GB of disk space for the first model download

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Use

Dub a video from translated SRT or SBV captions:

```bash
voice-dub source.mp4 \
  --text-file captions.en.sbv \
  --language en \
  --output dubbed.mp4
```

Translate Russian captions to English before dubbing:

```bash
voice-dub source.mp4 \
  --text-file captions.ru.sbv \
  --source-language ru \
  --language en \
  --translation-variants 3 \
  --accent american \
  --output dubbed.mp4
```

The app automatically builds a speech-only conditioning prompt from waveform-detected speech. For the closest clone, supply your own clean 6–10 second voice-only recording and retain lossless audio:

```bash
voice-dub source.mp4 \
  --text-file captions.en.sbv \
  --language en \
  --audio-output dubbed.wav \
  --output dubbed.mp4
```

The local `ref-voice-best-window.wav` (the curated `ref-voice.wav` segment starting
at `1.86s`) is used by default. Pass `--voice-reference PATH` only when you want
to clone a different voice. Brett is retained only as a portability fallback.
The saved project defaults are `cfg-weight 0.55`, temperature `0.6`, exaggeration
`0.4`, and seed `91`; timing uses waveform analysis with source pause alignment
and natural fitting. Use `--accent american` for the American-English preset.

By default, automatic translation produces three wordings (or up to ten with `--translation-variants 10`) in one model request. They are ranked before synthesis with a source-calibrated phonetic-duration estimate that accounts for syllable density, language rate, punctuation, and the original cue window. Only the closest-length wording is sent to speech synthesis, exactly once per cue. Waveform analysis refines caption timing, matches safe intra-line pauses, and aligns generated speech onsets to the source. Placement borrows unused gaps before or after cues and shifts later cues when needed, preventing line overlap whenever the surrounding timeline has enough room. Natural fitting preserves the generated voice unchanged and adds up to 0.6 seconds of gentle silence at phrase punctuation when a take is conspicuously short. Time-stretching is disabled because it can introduce artifacts; timing is handled with wording, pauses, and placement. Pause cleanup uses a soft gate to preserve breaths and quiet consonants. Video mastering applies conservative impulse de-clicking before loudness normalization and limiting. Use `--pause-alignment off` to disable pause matching.

For editorial control, put curated alternatives in one cue separated by ` || `. The selected wording is written to the output SBV:

```text
This setup is exhausting. || Setting everything up takes a lot out of you.
```

Run `voice-dub --help` for all controls or `voice-dub --list-languages` for supported languages.

For GPT-assisted translation with explicit short, baseline, and long alternatives, use
[`TRANSLATION_PROMPT.md`](TRANSLATION_PROMPT.md). Separate alternatives with ` || ` so the
pre-synthesis duration selector can choose one wording without generating multiple takes.

Every generation appends a JSON Lines record to `voice-dub-runs.jsonl` beside the output. Records include settings, device, total and per-candidate generation time, timing error, pause mismatch, voice similarity, overlap risk, selected wording, score, output size, and failure details. Use `--run-log PATH` to choose another history file.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Only clone voices you own or have permission to use. Chatterbox adds its built-in perceptual watermark to generated audio.
