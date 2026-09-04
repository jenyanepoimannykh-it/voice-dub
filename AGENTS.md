# Project agent instructions

## File locations

- Keep all temporary files, intermediate audio, generated subtitles, run logs, and other task artifacts inside this project directory.
- Do not use `/private/tmp`, `/tmp`, or any other external temporary directory for this project.
- Prefer a clearly named project-local directory such as `.work/` for transient files; do not commit transient files unless explicitly requested.

## Subtitle timing and variant fallback

- Treat the original cue starts and ends as the primary timing reference.
- Prefer a translation variant whose generated speech fits the cue without creating an artificial gap.
- If a generated line overlaps the next line, go back to the previous affected cue and try a shorter
  available `||` variant; recalculate placements after the change.
- If a generated line leaves an artificial gap greater than 0.2 seconds where the original has no
  comparable gap, go back to the previous affected cue and try a longer available `||` variant;
  recalculate placements after the change.
- A line is regenerated only when the measured result violates these overlap/gap restrictions. Do not
  synthesize extra variants speculatively when the current placement satisfies them.
- Natural mode never time-stretches generated speech. Explicit `--fit stretch` may use the fine Rubber
  Band backend in the conservative 0.96x–1.04x range; outside that range, use variants/placement.
- Measure a cue's available room from the later of its caption start and the previous line's
  real end, so placement drift cannot hide overlap or gap violations.
- Trim synthesizer padding before measuring a take; padding silence is not speech time.
- Never cut off the final phrase at the video boundary. Shift it earlier when possible; use overlap only
  when no non-overlapping placement or acceptable variant can preserve the ending.
