# Project memory

Durable notes for working on this repo. **This file is where project memory
belongs** — record durable context here rather than in an external per-project
memory store. Agent behaviour rules live in [`AGENTS.md`](AGENTS.md); this file
records context that is not derivable from the code or the git history.

## Owner and workflow

- Subtitles are translated **manually** by the owner, not by the built-in
  OPUS-MT path. `--source-language` exists for convenience but the real
  pipeline is: hand-written target-language SBV with ` || ` variants
  (see [`TRANSLATION_PROMPT.md`](TRANSLATION_PROMPT.md)) passed via
  `--text-file`.
- The reference voice lives in `reference/`, which is gitignored and must not
  be committed. The bundled CC0 Brett sample under `src/voice_dub/assets/` is
  only a portability fallback for clean checkouts.
- **The video stream must never be re-encoded.** `mux_video` uses `-c:v copy`;
  verify with matching `ffmpeg -map 0:v:0 -c copy -f md5 -` hashes before and
  after a change that touches muxing.
- Push over SSH with `~/.ssh/id_ed25519_jenya_nepoimannykh`
  (`GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_jenya_nepoimannykh'`).

## Duration estimation is calibrated to Chatterbox, not to human speech

`estimated_spoken_duration` rates (`en` at 7.7 units/s) look implausibly fast
for natural speech. They are correct: `phonetic_units` counts **vowel groups**,
which overcounts syllables ("because" scores 3), and the rates were fitted
against measured Chatterbox output with its padding trimmed. Nine takes gave
`duration ≈ 0.130 × units + 0.886s`; the 0.886s constant is the vocoder's
leading/trailing silence, which `trim_to_speech` now removes, leaving the
`units / 7.7` slope.

**Why it matters:** the pre-run estimate is what decides whether a hand-written
variant can fill its cue window. When these constants were set to natural
speaking rates (4.4 u/s), every hand-written variant came out ~40% too short
and left large holes in the dub.

**How to apply:** before writing target-language variants, check them with
`estimated_spoken_duration` so the five alternatives *bracket* the cue window
(shortest below it, longest above it). If Chatterbox is swapped or retuned,
re-fit the rates from a run log rather than reasoning about speaking rates.

## Timing fit depends on measuring real speech, not raw output

Chatterbox pads roughly 0.6s of silence around each phrase. Counting that as
speech time made every cue look longer than it sounded and pushed each
following cue steadily later. `trim_to_speech` runs before duration is measured
so variant selection, `available`, and the placement cursor all see real speech.

Likewise, a cue's available room is measured from
`max(cue.start, previous_audio_end)` — not from `cue.start`. Using `cue.start`
hides accumulated drift and lets the selector pick variants that cannot fit.

## The "montage" artefact is dead air, not a click

Chatterbox writes near-silence between words. Strung together across cues the
programme falls to about -70 dBFS and back many times a sentence, and the ear
reads that as a cut even though there is no discontinuity at all — measured
sample-to-sample steps at those points are below the 99th percentile of ordinary
speech slew.

Two things made it worse and one fixes it:

- `clean_pause_noise` gated *every* gap. Gaps under 180 ms were already merged
  into speech regions by `speech_intervals`, so the audible damage was on real
  pauses, which it drove from -70 to -93 dBFS. It now softens only stretches
  longer than 300 ms, and only to -15 dB rather than -24 dB.
- `room_tone` lays the source recording's own quiet passages under the whole dub
  at -35 dB relative to the speech level, the usual bed level for dialogue. This
  is the actual fix: it holds the noise floor continuous and puts the dub in the
  room the video was shot in. `--room-tone off` disables it.

**How to apply:** measure the gap floor as the 3rd percentile of frame RMS
against the 80th percentile as the speech level. Under about 40 dB of depth
reads as continuous; 45 dB and beyond reads as an edit. Take the bed level from
the *speech* level, never from the programme RMS — the silences drag that down
and the bed comes out inaudibly quiet.

## Splices must taper, and the taper must start at the right gain

Two click sources were shipped and had to be fixed:

- `clean_pause_noise` gates pauses to `floor_gain` (0.06) but ramped its region
  edges from **0.0**, so every boundary stepped down before ramping up. The step
  reached ~0.05 — as large as the 99th-percentile slew of real speech, and ~60x
  the local slew in a quiet passage.
- `align_internal_pauses` copied speech chunks into a zeroed buffer with no
  taper at all, stepping 0 -> full amplitude in one sample (~0.9, four times the
  largest legitimate transient in the programme).

**Why it matters:** both were invisible to every metric being tracked. Timing,
loudness and similarity were all fine while the audio ticked.

**How to apply:** any time audio is sliced, concatenated, gated or placed, taper
the join with `raised_cosine`, and start the ramp at the gain the neighbouring
samples already have — not at zero. `fade_edges` is the helper. Verify by
measuring the maximum sample-to-sample step in a +/-2 ms window around every
known splice against the 99th percentile of ordinary speech slew; the splices
should read 0.00000.

Two dead ends when hunting these: a naive |diff| threshold flags every plosive,
and a 5-sample median filter at 24 kHz flags every fricative. Neither finds
clicks in speech. Check the known edit points instead. `adeclick` in the master
chain does not mask them either — it measurably changes nothing here.

## Verify the mux, always

A filter FFmpeg cannot configure — `resampler=soxr` on a build without libsoxr,
for instance — fails the graph and leaves a **zero-byte** container behind while
the run otherwise looks fine. `verify_video_unchanged` now rejects an empty or
unreadable output and compares the video stream hash before and after.

**Why it matters:** this was found only because a hash check happened to run.
Worse, grepping the run's output for `Saved:` and success lines hid the FFmpeg
error entirely and the run was briefly reported as a success when it had failed.

**How to apply:** never filter a pipeline run's output down to the lines you
expect to see. Check the exit status and let errors through.

## Known dead ends

- `--accent american` only validates that the language is `en`; the
  `cfg_weight is None` branch it was meant to trigger is unreachable because
  `--cfg-weight` defaults to `0.55`. It is effectively a no-op today.
- `choose_candidate`, `choose_text_for_duration` and `choose_variant_indices`
  were dead code kept alive by their own tests; all three are gone. Selection is
  `timing_violation` plus editorial order.

## The voice is settled — and measurements did not pick it

`reference/ref-voice-best-window.wav` is the project voice. The owner compared
it by ear against the best of 153 CC0 voices and against an original Neumann U47
recording, and kept their original. It measures worst of the three by a wide
margin: SNR 21 dB against 44, and 99% of its energy below 4.5 kHz.

**Why it matters:** every objective metric available — bandwidth, noise floor,
dryness, speaker-embedding similarity — pointed at a voice the owner rejected on
hearing it. Embedding similarity in particular only measures how closely output
matches its own prompt; it cannot rank two different voices against each other.

**How to apply:** do not propose swapping the default voice, and do not treat a
metric win as a reason to. Render candidates as full dubs under distinct names
and let the owner listen. Keep every rejected candidate in `reference/` with a
note rather than deleting it.

## Choosing a voice reference

All 153 CC0 voices in [Voice-Zero](https://github.com/OwenTyme/voice-zero) were
measured (`.work/analyze.py`, `.work/rank.py`) and the shortlist was A/B'd
through Chatterbox itself. **Raw recording quality does not predict clone
quality.** `stuart_bell` had the best measured SNR, bandwidth and dryness of any
male voice and still cloned worst of the shortlist (0.919); `tamurile` had the
*lowest* source SNR and cloned best (0.937).

**How to apply:** rank candidates by measurement to build a shortlist, then pick
by synthesizing with each and comparing speaker-embedding similarity — mean and
minimum across several lines — plus the output's own noise floor. Prefer a
reference whose *minimum* similarity is high; that is what consistency across
cues depends on.

Two further findings:

- A denoised prompt yields quieter generated pauses. The Voice-Zero clips are
  noise-reduced and produced ~12 dB cleaner output than the same speaker's raw
  LibriVox source; `afftdn` recovered only ~4 dB of that.
- Longer is not better. A 10s window cut from the original scored no higher than
  the 7.3s pre-trimmed clip, which won on consistency and cleanliness.

LibriVox is the practical ceiling for CC0 speech: 128 kbps MP3 upstream, no
lossless, no documented microphones. Claims of specific studio gear cannot be
verified, so select on measured bandwidth, noise floor and dryness instead.

## Reference run

`results/shure-cap.en.sbv` dubbing `~/Desktop/shure-vid.mp4` is the worked
example: 4 cues, 9 takes, every placement within 0.05s of its source cue, voice
similarity 0.92–0.96, nothing clipped at the video boundary. Source video MD5
(video stream only) `de2c2ed61cb882d7d36f6b2b2a80ed02` — it must match in the
output.
