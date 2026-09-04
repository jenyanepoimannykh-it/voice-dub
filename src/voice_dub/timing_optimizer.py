"""Pure subtitle-variant timing optimization; independent of audio/TTS."""


def timing_violation(
    duration: float, target: float, gap_limit: float = 0.2,
    limit: float | None = None,
) -> float:
    """Seconds outside the allowed window.

    `target` is how long the source actually speaks in this cue; `limit` is the
    hard ceiling before the next line starts. They differ whenever the source
    pauses between cues: speech should match the target, not stretch across the
    pause, but it may run into the pause rather than overlap the next line.
    """
    if limit is None:
        limit = target
    if duration > limit:
        return duration - limit
    return max(0.0, target - duration - gap_limit)


def next_variant_index(
    estimates: list[float], tried: list[int], measured_duration: float,
    target: float, gap_limit: float = 0.2, limit: float | None = None,
) -> int | None:
    """Pick an untried variant in the direction needed by measured timing."""
    if limit is None:
        limit = target
    if timing_violation(measured_duration, target, gap_limit, limit) == 0:
        return None
    want_longer = measured_duration < target - gap_limit
    current = estimates[tried[-1]]
    choices = [i for i, estimate in enumerate(estimates) if i not in tried and
               ((estimate > current) if want_longer else (estimate < current))]
    if not choices:
        return None
    return min(choices, key=lambda i: abs(estimates[i] - target))


def best_start_index(estimates: list[float], target: float,
                     gap_limit: float = 0.2, limit: float | None = None) -> int:
    """First wording predicted to fit, else the one predicted to fit best.

    Editorial order still wins ties, so among equally good predictions the
    earliest wording is used.
    """
    if limit is None:
        limit = target
    fitting = [i for i, e in enumerate(estimates)
               if timing_violation(e, target, gap_limit, limit) == 0]
    if fitting:
        return fitting[0]
    return min(range(len(estimates)),
               key=lambda i: (round(timing_violation(estimates[i], target, gap_limit, limit), 6), i))


