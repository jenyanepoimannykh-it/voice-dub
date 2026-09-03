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
  --voice-reference clean-voice.wav \
  --text-file captions.en.sbv \
  --language en \
  --audio-output dubbed.wav \
  --output dubbed.mp4
```

By default, automatic translation produces three wordings ranked by estimated phonetic duration against the source window. The closest-length wording is selected before synthesis, and each cue is sent to the speech model exactly once. Waveform analysis refines caption timing, matches safe intra-line pauses, and aligns generated speech onsets to the source. Placement borrows unused gaps before or after cues and shifts later cues when needed, preventing line overlap whenever the surrounding timeline has enough room. Natural speech may run slightly beyond its caption window and is preserved rather than cut; use `--fit stretch` when every take must fit its window. Pause cleanup uses a soft gate to preserve breaths and quiet consonants. Use `--pause-alignment off` to disable pause matching.

For editorial control, put curated alternatives in one cue separated by ` || `. The selected wording is written to the output SBV:

```text
This setup is exhausting. || Setting everything up takes a lot out of you.
```

Run `voice-dub --help` for all controls or `voice-dub --list-languages` for supported languages.

Every generation appends a JSON Lines record to `voice-dub-runs.jsonl` beside the output. Records include settings, device, total and per-candidate generation time, timing error, pause mismatch, voice similarity, overlap risk, selected wording, score, output size, and failure details. Use `--run-log PATH` to choose another history file.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Only clone voices you own or have permission to use. Chatterbox adds its built-in perceptual watermark to generated audio.
