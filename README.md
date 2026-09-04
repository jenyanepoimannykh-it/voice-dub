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

The local, gitignored `reference/ref-voice-best-window.wav` is used by default.
Pass `--voice-reference PATH` only when you want to clone a different voice. The
CC0 Brett sample is retained only as a portable fallback for clean checkouts.
The saved project defaults are `cfg-weight 0.55`, temperature `0.6`, exaggeration
`0.4`, and seed `91`; timing uses waveform analysis with source pause alignment
and natural fitting. Use `--accent american` for the American-English preset.

By default, automatic translation produces three wordings (or up to ten with `--translation-variants 10`) in one model request. Curated alternatives may be separated with ` || `. Synthesis always starts with the first editorial variant. If its measured duration creates more than 0.2 seconds of artificial gap or any overlap, the app generates only the next suitable longer or shorter variant and stops retrying as soon as one fits. Overlap is accepted only after the available variant pile is exhausted. Waveform analysis refines caption timing and placement can borrow unused neighboring space. Generated speech is trimmed to its voiced span before measurement, so the synthesizer's padding silence is not mistaken for speech time, and each cue's available room is measured from where the previous line actually ended rather than from its nominal caption start. Natural fitting never time-stretches. Explicit `--fit stretch` uses only the free Rubber Band R3 engine with 32-bit float audio and limits adjustment to 0.96x–1.04x; it fails clearly when Rubber Band is unavailable instead of using a lower-quality fallback. Pause cleanup uses a soft gate, and video mastering applies conservative impulse de-clicking before loudness normalization and limiting.

For editorial control, put curated alternatives in one cue separated by ` || `. The selected wording is written to the output SBV:

```text
This setup is exhausting. || Setting everything up takes a lot out of you.
```

Duration estimates are calibrated against measured Chatterbox output rather than natural speaking rates, so `estimated_spoken_duration` predicts what the synthesizer will actually produce. Check hand-written variants against it: the five alternatives should bracket the cue window, with the shortest below it and the longest above it.

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
