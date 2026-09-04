"""Pure subtitle-variant timing optimization; independent of audio/TTS."""
from itertools import product


def timing_violation(duration: float, available: float, gap_limit: float = 0.2) -> float:
    """Return seconds outside the allowed [available-gap, available] interval."""
    if duration > available:
        return duration - available
    return max(0.0, available - duration - gap_limit)


def next_variant_index(
    estimates: list[float], tried: list[int], measured_duration: float,
    available: float, gap_limit: float = 0.2,
) -> int | None:
    """Pick an untried variant in the direction needed by measured timing."""
    if timing_violation(measured_duration, available, gap_limit) == 0:
        return None
    want_longer = measured_duration < available - gap_limit
    current = estimates[tried[-1]]
    choices = [i for i, estimate in enumerate(estimates) if i not in tried and
               ((estimate > current) if want_longer else (estimate < current))]
    if not choices:
        return None
    return min(choices, key=lambda i: abs(estimates[i] - available))


def choose_variant_indices(
    starts: list[float],
    ends: list[float],
    durations: list[list[float]],
    gap_limit: float = 0.2,
) -> list[int]:
    """Choose variants with first-choice priority and no avoidable violations.

    Durations are measured/generated values. The search starts with variant 0 for
    every cue and only changes indices when needed to remove overlap or an
    artificial gap. Returned indices minimize (violations, changed cues, rank).
    """
    if not (len(starts) == len(ends) == len(durations)):
        raise ValueError("cue arrays must have equal length")
    if not starts:
        return []
    if any(not row for row in durations):
        raise ValueError("every cue must have at least one variant")
    best = None
    for indices in product(*(range(len(row)) for row in durations)):
        violations = 0
        severity = 0.0
        for i in range(len(indices) - 1):
            finish = starts[i] + durations[i][indices[i]]
            delta = starts[i + 1] - finish
            if delta < 0:
                violations += 1
                severity += -delta
            elif delta > gap_limit:
                violations += 1
                severity += delta - gap_limit
        # Prefer no violations, then the fewest changed cues, then earlier
        # variants (semantic/editorial ordering), then smaller residual error.
        key = (violations, round(severity, 6), sum(i != 0 for i in indices), indices)
        if best is None or key < best[0]:
            best = (key, list(indices))
    return best[1]
