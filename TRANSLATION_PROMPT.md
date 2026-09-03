# Subtitle translation prompt

Copy the prompt below into GPT when preparing dubbing subtitles.

```text
Translate the Russian SBV subtitles into natural spoken American English for voice dubbing.

Keep the exact timestamps and the exact number of subtitle blocks. Preserve the complete meaning,
speaker perspective, emotional tone, and conversational style. Do not merge or split blocks.

For EVERY subtitle block, produce EXACTLY FIVE English alternatives in this order:

1. SHORT: clearly shorter than the original timing target, approximately 10–15% shorter in spoken
   duration. Remove repetition or use concise natural wording, but do not remove meaning.
2. SLIGHTLY SHORT: approximately 5–8% shorter than the timing target.
3. BASELINE: the most natural and semantically accurate version, approximately the same duration.
4. SLIGHTLY LONG: approximately 5–8% longer than the timing target.
5. LONG: clearly longer than the timing target, approximately 10–15% longer, while remaining natural.

The SHORT alternatives must genuinely contain fewer spoken syllables/phonetic units than BASELINE.
The SLIGHTLY SHORT alternative must also be shorter than BASELINE. The SLIGHTLY LONG and LONG
alternatives must genuinely contain more spoken syllables/phonetic units than BASELINE. Do not make
fake length changes by adding filler words, awkward repetition, or unnatural padding.

Rank semantic quality within each length class: every alternative must still be a faithful translation.
Use natural American English and preserve sensible pauses at commas and sentence boundaries.

Output only valid SBV. Put the five alternatives on the same subtitle text line, separated exactly by
` || ` (space, two pipe characters, space). Do not use `||` anywhere else. Do not add labels, numbers,
duration estimates, explanations, or comments to the subtitle text.

Before returning the SBV, silently verify for every block:
- alternative 1 is shortest;
- alternative 2 is shorter than alternative 3;
- alternative 3 is the baseline;
- alternative 4 is longer than alternative 3;
- alternative 5 is longest;
- all five preserve the same meaning;
- no alternative is dramatically longer or shorter than the requested range.

Original subtitles:
[paste the original SBV here]
```
