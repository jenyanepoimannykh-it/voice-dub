"""Pure subtitle-variant timing optimization; independent of audio/TTS."""
from itertools import product


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
