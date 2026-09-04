# voice-dub

Dubs a video into another language in a cloned voice, and leaves the video
stream untouched.

You supply the translated subtitles. The tool synthesizes each line, fits it to
the timing of the original speech, and muxes the result back over the original
video with `-c:v copy` — the video bitstream is byte-identical.

Built on [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox)
for zero-shot voice cloning, FFmpeg for I/O and mastering, and librosa for the
waveform analysis that drives timing.

## Requirements

- Python 3.10–3.13, FFmpeg
- Apple Silicon, an NVIDIA GPU, or CPU
- Several GB of disk for the model download on first run

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Use

```bash
voice-dub source.mp4 \
  --text-file captions.en.sbv \
  --language en \
  --output dubbed.mp4
```

Captions are SBV or SRT. Give each cue up to five wordings separated by ` || `,
ordered shortest to longest:

```text
0:00:00.000,0:00:05.080
it gets stressful, || it gets pretty stressful, || of course, it gets pretty stressful,
```

The tool synthesizes the first wording, measures it, and only tries another if
the result overlaps the next line or leaves more than 0.2s of dead air. The
wording it picks is written to an SBV beside the output.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--voice-reference PATH` | Voice to clone (default: `reference/ref-voice-best-window.wav`) |
| `--audio-output PATH` | Also save the dub as lossless WAV |
| `--source-language XX` | Machine-translate the captions first (lower quality than doing it yourself) |
| `--fit stretch` | Allow 0.96x–1.04x Rubber Band time-stretching |
| `--run-log PATH` | Where to append per-run metrics (default: beside the output) |

`voice-dub --help` for the rest, `--list-languages` for supported languages.

## Best practices

**Write variants that bracket the cue window.** This matters more than
anything else. Check them before you run:

```bash
python -c "
from voice_dub.cli import estimated_spoken_duration as e
print(e('your wording here', 'en'))"
```

The shortest variant should come in under the cue's duration and the longest
above it. If every variant is shorter than the window, the dub will leave a
hole there — no amount of tuning fixes that, only more words will.

**Use a clean, dry voice reference.** 7–10 seconds of continuous speech from a
good microphone, no music, no room echo, no clipping. Bandwidth and noise floor
matter more than length; a quiet 7-second clip beats a hissy 15-second one.

**Read the run log.** Every run appends JSON Lines with the per-cue placement,
gap, timing error, and voice similarity. `placement_shift` near zero means the
dub tracks the original; a large `gap` before a cue points at the previous cue
being too short.

**Check the video survived**, if you changed anything near muxing:

```bash
ffmpeg -v error -i in.mp4  -map 0:v:0 -c copy -f md5 -
ffmpeg -v error -i out.mp4 -map 0:v:0 -c copy -f md5 -
```

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Licensing

Only clone voices you own or have permission to use. Chatterbox adds its own
perceptual watermark to everything it generates.
