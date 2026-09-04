from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import shutil
from typing import Sequence


LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

TIMING_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})(?:\s+.*)?$"
)
SBV_TIMING_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*,\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*$"
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_BUNDLED_VOICE_REFERENCE = (
    Path(__file__).resolve().parent / "assets" / "brett-condron-american-baritenor.wav"
)
# Prefer the user's curated 1.86s-offset reference in the project directory.
# Keep the bundled sample only as a portability fallback for other checkouts.
DEFAULT_VOICE_REFERENCE = Path.cwd() / "ref-voice-best-window.wav"
if not DEFAULT_VOICE_REFERENCE.is_file():
    DEFAULT_VOICE_REFERENCE = _BUNDLED_VOICE_REFERENCE
MASTER_FILTER = (
    "adeclick=w=20:o=75:a=2:t=12:b=2:m=s,"
    "highpass=f=55,loudnorm=I=-16:TP=-1.5:LRA=9,alimiter=limit=0.95,apad"
)


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Candidate:
    duration_error: float
    pause_mismatch: int
    similarity: float
    overrun: float
    waveform: object
    text: str
    cfg_weight: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-dub",
        description=(
            "Speak target-language text in the voice from a reference audio clip "
            "using Chatterbox Multilingual V3."
        ),
    )
    parser.add_argument(
        "reference", nargs="?", type=Path,
        help="reference voice audio or a video containing the reference voice",
    )
    parser.add_argument(
        "--voice-reference", type=Path,
        help=(
            "voice-only recording for cloning; defaults to ref-voice-best-window.wav "
            "(1.86s offset) while the main input still controls timing"
        ),
    )
    parser.add_argument(
        "--text-file", type=Path,
        help="UTF-8 timed text file (SBV or SRT timestamps followed by translated text)",
    )
    parser.add_argument("--language", "-l", choices=sorted(LANGUAGES), help="target language code")
    parser.add_argument(
        "--source-language", choices=sorted(LANGUAGES),
        help="caption language; when different from --language, translate captions first",
    )
    parser.add_argument(
        "--translation-variants", type=int, default=3,
        help="alternative translations evaluated per cue, from 1 to 10 (default: 3)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("results/dubbed.wav"),
        help="output WAV/video (default: results/dubbed.wav)",
    )
    parser.add_argument(
        "--audio-output", type=Path,
        help="also save the generated audio as lossless WAV",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--exaggeration", type=float, default=0.4, help="emotion strength (default: 0.4)")
    parser.add_argument("--cfg-weight", type=float, default=0.55, help="voice/pacing guidance (default: 0.55)")
    parser.add_argument("--temperature", type=float, default=0.6, help="generation randomness (default: 0.6)")
    parser.add_argument(
        "--accent", choices=("auto", "american"), default="auto",
        help="target pronunciation preset; american is available for English",
    )
    parser.add_argument("--seed", type=int, default=91, help="reproducible random seed (default: 91)")
    parser.add_argument(
        "--timing", choices=("waveform", "captions"), default="waveform",
        help="refine caption boundaries using source speech activity (default: waveform)",
    )
    parser.add_argument(
        "--timing-search", type=float, default=0.6,
        help="seconds around a caption boundary to search for speech edges (default: 0.6)",
    )
    parser.add_argument(
        "--duration-tolerance", type=float, default=0.25,
        help="maximum allowed natural duration mismatch as a fraction (default: 0.25)",
    )
    parser.add_argument(
        "--placement", choices=("center", "start"), default="center",
        help="place shorter speech within its source window (default: center)",
    )
    parser.add_argument(
        "--pause-alignment", choices=("source", "off"), default="source",
        help="match generated intra-line pauses to source speech pauses (default: source)",
    )
    parser.add_argument(
        "--fit", choices=("natural", "stretch", "trim"), default="natural",
        help="fit speech to time windows (default: natural; never slows short speech)",
    )
    parser.add_argument(
        "--translated-captions-output", type=Path,
        help="save translated captions as SBV (default: beside output video/audio)",
    )
    parser.add_argument(
        "--run-log", type=Path,
        help="append run metrics as JSON Lines (default: voice-dub-runs.jsonl beside output)",
    )
    parser.add_argument(
        "--translate-only", action="store_true",
        help="translate and save captions without generating speech",
    )
    parser.add_argument("--list-languages", action="store_true", help="print supported languages and exit")
    return parser


def parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_timed_text(value: str) -> list[Cue]:
    """Parse SBV or SRT content, with or without numeric cue identifiers."""
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[Cue] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip().lstrip("\ufeff")
        if not line or line.isdigit():
            index += 1
            continue
        match = TIMING_RE.match(line) or SBV_TIMING_RE.match(line)
        if not match:
            raise ValueError(f"expected a timestamp at line {index + 1}, got: {line!r}")
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        if end <= start:
            raise ValueError(f"cue at line {index + 1} must end after it starts")
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            if TIMING_RE.match(lines[index].strip()) or SBV_TIMING_RE.match(lines[index].strip()):
                break
            text_lines.append(lines[index].strip())
            index += 1
        text = " ".join(text_lines).strip()
        if not text:
            raise ValueError(f"cue beginning at {match.group('start')} has no text")
        if cues and start < cues[-1].start:
            raise ValueError("cues must be ordered by start time")
        cues.append(Cue(start, end, text))
    if not cues:
        raise ValueError("the timed text file contains no cues")
    return cues


def read_cues(text_file: Path | None) -> list[Cue]:
    if text_file is None:
        raise ValueError("provide translated timed text with --text-file")
    try:
        return parse_timed_text(text_file.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read text file '{text_file}': {error}") from error


def format_sbv(cues: list[Cue]) -> str:
    def timestamp(seconds: float) -> str:
        milliseconds = round(seconds * 1000)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"

    return "\n\n".join(
        f"{timestamp(cue.start)},{timestamp(cue.end)}\n{cue.text}" for cue in cues
    ) + "\n"


def choose_device(torch_module: object, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_ffmpeg(command: list[str], purpose: str) -> None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg is required but was not found on PATH") from error
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()
        message = detail[-1] if detail else "unknown FFmpeg error"
        raise RuntimeError(f"could not {purpose}: {message}")


def append_run_log(
    args: argparse.Namespace,
    metrics: dict[str, object],
    started_at: str,
    elapsed_seconds: float,
    status: str,
    error: str | None = None,
) -> Path:
    """Append a durable, machine-readable record for future comparisons."""
    output = args.output.expanduser().resolve()
    destination = (
        args.run_log.expanduser().resolve()
        if args.run_log else output.parent / "voice-dub-runs.jsonl"
    )
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    record = {
        "started_at": started_at,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "output": str(output),
        "output_bytes": output.stat().st_size if status == "success" and output.exists() else None,
        "settings": settings,
        **metrics,
    }
    if error:
        record["error"] = error
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


def extract_reference(source: Path, destination: Path) -> None:
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn",
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(destination),
        ],
        "extract reference audio",
    )


def mux_video(video: Path, audio: Path, output: Path) -> None:
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-af", MASTER_FILTER,
            "-shortest", str(output),
        ],
        "create dubbed video",
    )


def speech_intervals(waveform: object, sample_rate: int, librosa: object, np: object) -> list[tuple[float, float]]:
    """Return energy-based speech regions, joining very short internal gaps."""
    frame_length = max(32, round(sample_rate * 0.020))
    hop_length = max(16, round(sample_rate * 0.010))
    rms = librosa.feature.rms(
        y=waveform, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    if not len(rms) or float(np.max(rms)) <= 1e-8:
        return []
    db = librosa.amplitude_to_db(rms, ref=np.max)
    noise_floor = float(np.percentile(db, 20))
    threshold = min(-25.0, max(-40.0, noise_floor + 10.0))
    active = db >= threshold

    # Fill pauses shorter than 180 ms so consonants do not split a phrase.
    max_gap = max(1, round(0.18 * sample_rate / hop_length))
    index = 0
    while index < len(active):
        if active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and not active[end]:
            end += 1
        if index > 0 and end < len(active) and end - index <= max_gap:
            active[index:end] = True
        index = end

    intervals: list[tuple[float, float]] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        end = index + 1
        while end < len(active) and active[end]:
            end += 1
        start_time = max(0.0, (index * hop_length - frame_length / 2) / sample_rate)
        end_time = min(len(waveform) / sample_rate, (end * hop_length + frame_length / 2) / sample_rate)
        if end_time - start_time >= 0.12:
            intervals.append((start_time, end_time))
        index = end
    return intervals


def refine_cue_timing(
    cues: list[Cue], intervals: list[tuple[float, float]], search: float
) -> list[Cue]:
    """Snap subtitle boundaries only when a nearby waveform speech edge exists."""
    if not intervals:
        return cues
    starts = [start for start, _ in intervals]
    ends = [end for _, end in intervals]
    refined: list[Cue] = []
    for cue in cues:
        nearby_starts = [value for value in starts if abs(value - cue.start) <= search]
        nearby_ends = [value for value in ends if abs(value - cue.end) <= search]
        start = min(nearby_starts, key=lambda value: abs(value - cue.start)) if nearby_starts else cue.start
        end = min(nearby_ends, key=lambda value: abs(value - cue.end)) if nearby_ends else cue.end
        if end <= start + 0.1:
            start, end = cue.start, cue.end
        refined.append(Cue(start, end, cue.text))

    # Do not create overlaps by snapping two independent boundaries.
    result: list[Cue] = []
    for index, cue in enumerate(refined):
        start = max(cue.start, result[-1].end if result else 0.0)
        if cue.end <= start + 0.1:
            start, end = cues[index].start, cues[index].end
        else:
            end = cue.end
        result.append(Cue(start, end, cue.text))
    return result


def align_internal_pauses(
    waveform: object,
    sample_rate: int,
    cue: Cue,
    source_intervals: list[tuple[float, float]],
    generated_intervals: list[tuple[float, float]],
    np: object,
) -> object | None:
    """Rebuild a cue by preserving speech chunks and redistributing only silence."""
    source_chunks = [
        (max(cue.start, start), min(cue.end, end))
        for start, end in source_intervals
        if end > cue.start and start < cue.end
    ]
    if len(source_chunks) < 2 or len(source_chunks) != len(generated_intervals):
        return None

    generated_chunks = [
        waveform[
            max(0, round(start * sample_rate)):
            min(len(waveform), round(end * sample_rate))
        ]
        for start, end in generated_intervals
    ]
    slot_samples = max(1, round((cue.end - cue.start) * sample_rate))
    speech_samples = sum(len(chunk) for chunk in generated_chunks)
    spare_samples = slot_samples - speech_samples
    if spare_samples < 0:
        return None

    source_gaps = [max(0.0, source_chunks[0][0] - cue.start)]
    source_gaps.extend(
        max(0.0, current[0] - previous[1])
        for previous, current in zip(source_chunks, source_chunks[1:])
    )
    source_gaps.append(max(0.0, cue.end - source_chunks[-1][1]))
    internal_gaps = source_gaps[1:-1]
    # Ignore detector-scale dips; only align clearly audible pauses.
    if not any(gap >= 0.28 for gap in internal_gaps):
        return None

    internal_samples = [round(gap * sample_rate) for gap in internal_gaps]
    internal_total = sum(internal_samples)
    if internal_total > spare_samples:
        scale = spare_samples / internal_total
        internal_samples = [round(value * scale) for value in internal_samples]
    edge_samples = spare_samples - sum(internal_samples)
    gap_samples = [edge_samples // 2, *internal_samples, edge_samples - edge_samples // 2]
    aligned = np.zeros(slot_samples, dtype=np.float32)
    cursor = gap_samples[0]
    for index, chunk in enumerate(generated_chunks):
        end = min(slot_samples, cursor + len(chunk))
        aligned[cursor:end] = chunk[:end - cursor]
        cursor = end + gap_samples[index + 1]
    return aligned


def clean_pause_noise(
    waveform: object,
    intervals: list[tuple[float, float]],
    sample_rate: int,
    np: object,
    fade_ms: float = 20.0,
    margin_ms: float = 35.0,
    floor_gain: float = 0.06,
) -> object:
    """Soft-gate pauses while preserving breaths and quiet consonant edges."""
    if not intervals:
        return waveform
    cleaned = waveform * floor_gain
    fade_samples = max(1, round(sample_rate * fade_ms / 1000.0))
    margin_samples = max(0, round(sample_rate * margin_ms / 1000.0))
    for start, end in intervals:
        left = max(0, round(start * sample_rate) - margin_samples)
        right = min(len(waveform), round(end * sample_rate) + margin_samples)
        if right <= left:
            continue
        cleaned[left:right] = waveform[left:right]
        fade = min(fade_samples, (right - left) // 2)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float32)
            cleaned[left:left + fade] *= ramp
            cleaned[right - fade:right] *= ramp[::-1]
    return cleaned


def speech_only_reference(
    waveform: object,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    np: object,
    max_seconds: float = 10.0,
) -> object:
    """Concatenate voiced regions without leading, trailing, or long quiet gaps."""
    pieces: list[object] = []
    silence = np.zeros(round(0.08 * sample_rate), dtype=np.float32)
    total = 0
    limit = round(max_seconds * sample_rate)
    for start, end in intervals:
        left = max(0, round(start * sample_rate))
        right = min(len(waveform), round(end * sample_rate))
        if right <= left:
            continue
        piece = waveform[left:right]
        if total + len(piece) > limit:
            piece = piece[: max(0, limit - total)]
        if len(piece):
            if pieces and total < limit:
                separator = silence[: max(0, limit - total)]
                pieces.append(separator)
                total += len(separator)
                if total >= limit:
                    break
            pieces.append(piece)
            total += len(piece)
        if total >= limit:
            break
    if not pieces:
        return waveform
    return np.concatenate(pieces)[:limit].astype(np.float32, copy=False)


def active_duration(cue: Cue, intervals: list[tuple[float, float]]) -> float:
    """Measure voiced time inside a cue, excluding detected pauses."""
    return sum(
        max(0.0, min(cue.end, end) - max(cue.start, start))
        for start, end in intervals
    )


def phonetic_units(text: str, language: str) -> int:
    """Estimate pronounceable units for duration-aware wording selection."""
    if language in {"zh", "ja"}:
        characters = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text)
        return max(1, len(characters))
    vowel_sets = {
        "en": "aeiouy", "ru": "аеёиоуыэюя", "de": "aeiouyäöü",
        "fr": "aeiouyàâæéèêëîïôœùûü", "es": "aeiouáéíóúü",
        "it": "aeiouàèéìíîòóùú", "pt": "aeiouáâãàéêíóôõúü",
    }
    vowels = vowel_sets.get(language, "aeiouyáéíóúàèìòùäëïöüâêîôû")
    groups = re.findall(f"[{re.escape(vowels)}]+", text.lower())
    words_without_vowels = sum(
        not any(character in vowels for character in word.lower())
        for word in re.findall(r"[^\W\d_]+", text, re.UNICODE)
    )
    return max(1, len(groups) + words_without_vowels)


def estimated_spoken_duration(text: str, language: str) -> float:
    """Estimate natural speech duration, including small inter-word pauses."""
    rates = {"en": 4.4, "ru": 4.2, "de": 4.1, "fr": 4.5, "es": 4.6, "it": 4.5,
             "pt": 4.4, "zh": 4.0, "ja": 4.5}
    units = phonetic_units(text, language)
    punctuation_pause = 0.12 * len(re.findall(r"[,;:.!?]", text))
    return units / rates.get(language, 4.3) + punctuation_pause


def estimated_translation_duration(
    text: str,
    source_text: str,
    source_language: str,
    target_language: str,
    source_window: float,
) -> float:
    """Predict target duration while calibrating to the source cue's density."""
    source_estimate = estimated_spoken_duration(source_text, source_language)
    target_estimate = estimated_spoken_duration(text, target_language)
    if source_estimate <= 0 or source_window <= 0:
        return target_estimate
    # Preserve the source cue's actual pacing (including its caption pauses),
    # while accounting for different phonetic density and language rates.
    return source_window * target_estimate / source_estimate


def choose_text_for_duration(options: list[str], duration: float, language: str) -> str:
    """Choose a non-overlong wording, accepting up to one second of spare time."""
    estimates = [(text, estimated_spoken_duration(text, language)) for text in options]
    # Prefer the first semantically-ranked wording that fits or is only slightly
    # short. This preserves editorial ordering while avoiding overrun.
    acceptable = [item for item in estimates if 0.0 <= duration - item[1] <= 1.0]
    if acceptable:
        return acceptable[0][0]
    # If every wording is too long, use the shortest one as a safe fallback.
    under = [item for item in estimates if item[1] <= duration]
    if under:
        return max(under, key=lambda item: item[1])[0]
    return min(estimates, key=lambda item: item[1])[0]


def candidate_score(candidate: Candidate) -> float:
    """Balance timing, identity, pauses, and collisions on comparable scales."""
    return (
        0.45 * min(candidate.duration_error, 1.5)
        + 0.35 * min(max(1.0 - candidate.similarity, 0.0) * 4.0, 1.5)
        + 0.10 * min(candidate.pause_mismatch / 2.0, 1.0)
        + 0.10 * min(candidate.overrun, 1.0)
    )


def choose_candidate(candidates: list[Candidate], tolerance: float) -> Candidate:
    """Choose the best balanced take, falling back when none meet timing tolerance."""
    if not candidates:
        raise ValueError("cannot choose from an empty candidate list")
    eligible = [candidate for candidate in candidates if candidate.duration_error <= tolerance]
    if eligible:
        pool = eligible
    else:
        closest_error = min(candidate.duration_error for candidate in candidates)
        pool = [candidate for candidate in candidates if candidate.duration_error <= closest_error + 0.02]
    return min(pool, key=lambda candidate: (candidate_score(candidate), candidate.duration_error))


def placement_start(
    cue: Cue,
    generated_samples: int,
    sample_rate: int,
    source_regions: list[tuple[float, float]],
    generated_regions: list[tuple[float, float]],
    previous_end: float,
    next_start: float | None,
    tolerance: float,
) -> int:
    """Align voiced onsets, borrowing free gaps without overlapping prior audio."""
    base = cue.start
    source = [(max(cue.start, a), min(cue.end, b)) for a, b in source_regions if b > cue.start and a < cue.end]
    if source and generated_regions:
        base = source[0][0] - generated_regions[0][0]
    earliest = max(0.0, previous_end)
    if next_start is not None:
        latest = next_start - generated_samples / sample_rate
        if latest >= earliest:
            base = min(max(base, earliest), latest)
        else:
            # There is not enough room before the following nominal cue. Keep
            # this cue clear of prior speech; the next cue can shift later.
            base = max(base, earliest)
    else:
        # Keep the final cue inside the source duration when possible. If it is
        # longer than its slot, move it earlier rather than letting ffmpeg's
        # `-shortest` cut off the end of the phrase.
        latest_for_end = cue.end - generated_samples / sample_rate
        if latest_for_end >= 0:
            base = min(base, latest_for_end)
        base = max(base, 0.0)
        if base >= earliest:
            base = max(base, earliest)
    return round(base * sample_rate)


def add_phrase_pauses(
    waveform: object,
    text: str,
    sample_rate: int,
    missing_seconds: float,
    np: object,
    max_total: float = 0.6,
    max_each: float = 0.25,
) -> tuple[object, float]:
    """Insert restrained silence at punctuation, cutting at nearby quiet samples."""
    boundaries = [match.end() for match in re.finditer(r"[,;:—–]", text)]
    if not boundaries or missing_seconds < 0.35 or len(waveform) < sample_rate // 2:
        return waveform, 0.0
    total = min(missing_seconds, max_total, max_each * len(boundaries))
    pause_samples = round(total / len(boundaries) * sample_rate)
    search = round(0.3 * sample_rate)
    smooth = max(1, round(0.008 * sample_rate))
    energy = np.convolve(
        np.square(waveform, dtype=np.float32),
        np.ones(smooth, dtype=np.float32) / smooth,
        mode="same",
    )
    cuts: list[int] = []
    for boundary in boundaries:
        expected = round(boundary / max(len(text), 1) * len(waveform))
        left = max(smooth, expected - search)
        right = min(len(waveform) - smooth, expected + search)
        if right > left:
            cut = left + int(np.argmin(energy[left:right]))
            if not cuts or cut - cuts[-1] > smooth * 2:
                cuts.append(cut)
    if not cuts:
        return waveform, 0.0

    pieces: list[object] = []
    cursor = 0
    fade = np.linspace(1.0, 0.0, smooth, dtype=np.float32)
    silence = np.zeros(pause_samples, dtype=np.float32)
    for cut in cuts:
        piece = waveform[cursor:cut].copy()
        if len(piece) >= smooth:
            if cursor:
                piece[:smooth] *= fade[::-1]
            piece[-smooth:] *= fade
        pieces.extend((piece, silence))
        cursor = cut
    tail = waveform[cursor:].copy()
    if len(tail) >= smooth:
        tail[:smooth] *= fade[::-1]
    pieces.append(tail)
    inserted = len(cuts) * pause_samples / sample_rate
    return np.concatenate(pieces).astype(np.float32, copy=False), inserted


def professional_time_stretch(waveform: object, rate: float, sample_rate: int, np: object) -> object:
    """Use Rubber Band for clean speech stretching, with librosa fallback."""
    rubberband = shutil.which("rubberband")
    if not rubberband:
        import librosa
        return librosa.effects.time_stretch(waveform, rate=rate)
    import soundfile as sf
    work_root = Path.cwd() / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stretch-", dir=work_root) as directory:
        source = Path(directory) / "source.wav"
        target = Path(directory) / "target.wav"
        # Keep 32-bit float throughout to avoid an extra quantization stage.
        sf.write(source, np.asarray(waveform, dtype=np.float32), sample_rate, subtype="FLOAT")
        subprocess.run(
            [rubberband, "--fine", "--tempo", str(rate), str(source), str(target)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        stretched, _ = sf.read(target, dtype="float32")
    return np.asarray(stretched, dtype=np.float32)


def extract_text_options(cues: list[Cue]) -> tuple[list[Cue], list[list[str]]]:
    """Allow curated alternatives separated by ` || ` inside a timed cue."""
    primary: list[Cue] = []
    options: list[list[str]] = []
    for cue in cues:
        alternatives = list(
            dict.fromkeys(part.strip() for part in cue.text.split(" || ") if part.strip())
        )
        if not alternatives:
            alternatives = [cue.text]
        primary.append(Cue(cue.start, cue.end, alternatives[0]))
        options.append(alternatives)
    return primary, options


def translate_cues(
    cues: list[Cue], source: str, target: str, device: str, variant_count: int = 1
) -> tuple[list[Cue], list[list[str]]]:
    """Translate cues and retain alternative wordings for speech selection."""
    if source == target:
        return cues, [[cue.text] for cue in cues]
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("translation dependencies are missing; run: pip install -e .") from error

    model_id = f"Helsinki-NLP/opus-mt-{source}-{target}"
    print(f"Loading translation model {model_id}...", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        translator = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    except OSError as error:
        raise RuntimeError(
            f"no OPUS-MT model was found for {source} -> {target}; "
            "provide captions already translated into the target language"
        ) from error

    translation_device = device if device in {"cuda", "mps"} else "cpu"
    translator.to(translation_device)
    result: list[Cue] = []
    variants: list[list[str]] = []
    for number, cue in enumerate(cues, start=1):
        print(f"Translating cue {number}/{len(cues)}...", file=sys.stderr)
        tokens = tokenizer(cue.text, return_tensors="pt", truncation=True).to(translation_device)
        beam_pool = max(variant_count * 4, 4)
        with torch.inference_mode():
            translated_tokens = translator.generate(
                **tokens,
                max_new_tokens=256,
                num_beams=beam_pool,
                num_return_sequences=beam_pool,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        pool = list(dict.fromkeys(text.strip() for text in decoded if text.strip()))
        if not pool:
            raise RuntimeError(f"translation produced empty text for cue {number}")
        # Beam outputs are already ordered by translation quality. Prefer those
        # whose estimated pronunciation length fits the original speech window,
        # using beam rank to break near-ties.
        target_duration = cue.end - cue.start
        ranked = sorted(
            enumerate(pool),
            key=lambda item: (
                round(
                    abs(
                        estimated_translation_duration(
                            item[1], cue.text, source, target, target_duration
                        )
                        - target_duration
                    ),
                    3,
                ),
                item[0],
            ),
        )
        alternatives = [text for _, text in ranked[:variant_count]]
        for variant_number, text in enumerate(alternatives, start=1):
            predicted = estimated_translation_duration(
                text, cue.text, source, target, target_duration
            )
            print(
                f'  variant {variant_number}: ({predicted:.2f}s predicted) "{text}"',
                file=sys.stderr,
            )
        result.append(Cue(cue.start, cue.end, alternatives[0]))
        variants.append(alternatives)
    del translator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result, variants


def run(args: argparse.Namespace) -> Path:
    if args.reference is None:
        raise ValueError("provide a reference audio file")
    if not args.reference.is_file():
        raise ValueError(f"reference audio does not exist or is not a file: '{args.reference}'")
    if args.language is None:
        raise ValueError("provide a target language with --language")
    if not 0.0 <= args.exaggeration <= 2.0:
        raise ValueError("--exaggeration must be between 0 and 2")
    if args.cfg_weight is not None and not 0.0 <= args.cfg_weight <= 1.0:
        raise ValueError("--cfg-weight must be between 0 and 1")
    if args.accent == "american" and args.language != "en":
        raise ValueError("--accent american requires --language en")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")
    if not 1 <= args.translation_variants <= 10:
        raise ValueError("--translation-variants must be between 1 and 10")
    if args.timing_search < 0:
        raise ValueError("--timing-search cannot be negative")
    if not 0 <= args.duration_tolerance < 1:
        raise ValueError("--duration-tolerance must be at least 0 and less than 1")

    cues = read_cues(args.text_file)

    try:
        import torch
        import torchaudio
        import librosa
        import numpy as np
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as error:
        raise RuntimeError("dependencies are missing; install the project with: pip install -e .") from error

    device = choose_device(torch, args.device)
    run_metrics = getattr(args, "_run_metrics", None)
    if run_metrics is not None:
        run_metrics["device"] = device
        run_metrics["cues"] = []
    cfg_weight = args.cfg_weight
    if cfg_weight is None:
        cfg_weight = 0.0 if args.accent == "american" else 0.5
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    cues, cue_text_options = extract_text_options(cues)
    if args.source_language:
        cues, cue_text_options = translate_cues(
            cues,
            args.source_language,
            args.language,
            device,
            args.translation_variants,
        )

    captions_output = None
    has_curated_variants = any(len(options) > 1 for options in cue_text_options)
    if args.source_language or has_curated_variants:
        captions_output = args.translated_captions_output
        if captions_output is None:
            output_base = args.output.expanduser().resolve()
            captions_output = output_base.with_suffix(f".{args.language}.sbv")
        captions_output = captions_output.expanduser().resolve()
        captions_output.parent.mkdir(parents=True, exist_ok=True)
        captions_output.write_text(format_sbv(cues), encoding="utf-8")
        print(f"Saved translated captions: {captions_output}", file=sys.stderr)
    if args.translate_only:
        if not args.source_language:
            raise ValueError("--translate-only requires --source-language")
        assert captions_output is not None
        return captions_output

    print(f"Loading Chatterbox Multilingual V3 on {device}...", file=sys.stderr)
    model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
    work_root = args.output.expanduser().resolve().parent / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voice-dub-", dir=work_root) as directory:
        timing_wav = Path(directory) / "timing.wav"
        extract_reference(args.reference.resolve(), timing_wav)
        selected_voice_reference = (
            args.voice_reference.expanduser().resolve()
            if args.voice_reference else DEFAULT_VOICE_REFERENCE
        )
        if not selected_voice_reference.is_file():
            raise RuntimeError(
                f"default voice reference is missing: '{selected_voice_reference}'"
            )
        reference_wav = Path(directory) / "voice-reference.wav"
        extract_reference(selected_voice_reference, reference_wav)
        if run_metrics is not None:
            run_metrics["voice_reference_path"] = str(selected_voice_reference)
            run_metrics["voice_reference_default"] = args.voice_reference is None
        intervals: list[tuple[float, float]] = []
        if args.timing == "waveform":
            source_waveform, source_rate = librosa.load(timing_wav, sr=None, mono=True)
            intervals = speech_intervals(source_waveform, source_rate, librosa, np)
            original_cues = cues
            cues = refine_cue_timing(cues, intervals, args.timing_search)
            print(f"Detected {len(intervals)} source speech regions.", file=sys.stderr)
            for number, (before, after) in enumerate(zip(original_cues, cues), start=1):
                if before.start != after.start or before.end != after.end:
                    print(
                        f"  cue {number}: {before.start:.3f}–{before.end:.3f}s -> "
                        f"{after.start:.3f}–{after.end:.3f}s",
                        file=sys.stderr,
                    )
            if captions_output is not None:
                captions_output.write_text(format_sbv(cues), encoding="utf-8")
        # Clean every conditioning prompt, including an explicit
        # --voice-reference, before Chatterbox derives the speaker conditionals.
        reference_waveform, reference_rate = librosa.load(reference_wav, sr=None, mono=True)
        reference_intervals = speech_intervals(
            reference_waveform, reference_rate, librosa, np
        )
        if reference_intervals:
            reference_audio = speech_only_reference(
                reference_waveform, reference_rate, reference_intervals, np
            )
            reference_wav = Path(directory) / "speech-reference.wav"
            torchaudio.save(
                str(reference_wav),
                torch.from_numpy(reference_audio).unsqueeze(0),
                reference_rate,
            )
            original_seconds = len(reference_waveform) / reference_rate
            cleaned_seconds = len(reference_audio) / reference_rate
            if run_metrics is not None:
                run_metrics["voice_reference"] = {
                    "original_seconds": round(original_seconds, 3),
                    "cleaned_seconds": round(cleaned_seconds, 3),
                    "speech_regions": len(reference_intervals),
                }
            print(
                f"Using cleaned speech-only voice reference: {cleaned_seconds:.2f}s "
                f"from {len(reference_intervals)} region(s); removed leading/trailing "
                "silence and quiet gaps.",
                file=sys.stderr,
            )
        else:
            print(
                "Warning: no clear speech regions found in voice reference; using it unchanged.",
                file=sys.stderr,
            )
        model.prepare_conditionals(str(reference_wav), exaggeration=args.exaggeration)
        sample_rate = model.sr
        timeline = np.zeros(round(cues[-1].end * sample_rate), dtype=np.float32)

        reference_embedding = model.conds.t3.speaker_emb.squeeze().detach().cpu().numpy()
        reference_embedding /= np.linalg.norm(reference_embedding) + 1e-8
        chosen_cues: list[Cue] = []
        previous_audio_end = 0

        for number, cue in enumerate(cues, start=1):
            print(
                f"Generating cue {number}/{len(cues)} "
                f"({cue.start:.3f}s–{cue.end:.3f}s)...",
                file=sys.stderr,
            )
            slot_samples = max(1, round((cue.end - cue.start) * sample_rate))
            original_voice_duration = active_duration(cue, intervals) if intervals else cue.end - cue.start
            if original_voice_duration < 0.25:
                original_voice_duration = cue.end - cue.start
            candidates: list[Candidate] = []
            cue_metrics: dict[str, object] = {
                "number": number,
                "start": round(cue.start, 3),
                "end": round(cue.end, 3),
                "source_voice_duration": round(original_voice_duration, 3),
                "candidates": [],
            }
            if run_metrics is not None:
                run_metrics["cues"].append(cue_metrics)
            source_chunk_count = sum(
                end > cue.start and start < cue.end for start, end in intervals
            )
            # Always synthesize the first editorial variant initially. Timing
            # fallback decisions belong to timing_optimizer and are made only
            # after measured durations reveal a violation.
            text_options = [cue_text_options[number - 1][0]]
            if len(text_options) > 1:
                selected_text = choose_text_for_duration(
                    text_options, original_voice_duration, args.language
                )
                print(
                    f'  preselected for phonetic length: "{selected_text}"',
                    file=sys.stderr,
                )
                text_options = [selected_text]
                cue_metrics["preselected_text"] = selected_text
                cue_metrics["estimated_text_duration"] = round(
                    estimated_spoken_duration(selected_text, args.language), 3
                )
            total_candidates = 1
            candidate_number = 0
            for _ in range(1):
                indices = [0]
                for text_index in indices:
                    candidate_text = text_options[text_index]
                    if len(text_options) > 1:
                        print(
                            f'  translation {text_index + 1}/{len(text_options)}: "{candidate_text}"',
                            file=sys.stderr,
                        )
                    candidate_number += 1
                    candidate_cfg = cfg_weight
                    generation_started = time.monotonic()
                    generated = model.generate(
                        candidate_text,
                        language_id=args.language,
                        exaggeration=args.exaggeration,
                        cfg_weight=candidate_cfg,
                        temperature=args.temperature,
                    ).squeeze().cpu().numpy()
                    generation_seconds = time.monotonic() - generation_started
                    generated_duration = len(generated) / sample_rate
                    generated_regions = speech_intervals(generated, sample_rate, librosa, np)
                    pause_mismatch = (
                        abs(len(generated_regions) - source_chunk_count) if intervals else 0
                    )
                    generated_16k = librosa.resample(
                        generated, orig_sr=sample_rate, target_sr=16000
                    )
                    embedding = model.ve.embeds_from_wavs(
                        [generated_16k], sample_rate=16000
                    ).mean(axis=0)
                    embedding /= np.linalg.norm(embedding) + 1e-8
                    similarity = float(np.dot(reference_embedding, embedding))
                    duration_error = (
                        abs(generated_duration - original_voice_duration)
                        / original_voice_duration
                    )
                    next_start = cues[number].start if number < len(cues) else None
                    available = (next_start or cue.end) - cue.start
                    overrun = max(0.0, generated_duration - available) / max(available, 1e-8)
                    candidate = Candidate(
                        duration_error, pause_mismatch, similarity, overrun,
                        generated, candidate_text, candidate_cfg,
                    )
                    candidates.append(candidate)
                    cue_metrics["candidates"].append({
                        "text": candidate_text,
                        "duration": round(generated_duration, 3),
                        "duration_error": round(duration_error, 4),
                        "pause_mismatch": pause_mismatch,
                        "voice_similarity": round(similarity, 4),
                        "overrun": round(overrun, 4),
                        "cfg_weight": round(candidate_cfg, 3),
                        "generation_seconds": round(generation_seconds, 3),
                        "score": round(candidate_score(candidate), 4),
                    })
                    status = "within tolerance" if duration_error <= args.duration_tolerance else "fallback"
                    print(
                        f"  candidate {candidate_number}/{total_candidates}: "
                        f"{generated_duration:.2f}s, duration error {duration_error:.1%} "
                        f"({status}), pause mismatch {pause_mismatch}, voice score {similarity:.3f}, "
                        f"cfg {candidate_cfg:.2f}",
                        file=sys.stderr,
                    )
            # Timing leads; matching pause structure and voice break near ties. If no
            # take is within tolerance, retain the closest one instead of failing.
            chosen = choose_candidate(candidates, args.duration_tolerance)
            generated, chosen_text = chosen.waveform, chosen.text
            cue_metrics["selected"] = {
                "text": chosen.text,
                "duration_error": round(chosen.duration_error, 4),
                "pause_mismatch": chosen.pause_mismatch,
                "voice_similarity": round(chosen.similarity, 4),
                "overrun": round(chosen.overrun, 4),
                "cfg_weight": round(chosen.cfg_weight, 3),
                "score": round(candidate_score(chosen), 4),
            }
            if chosen.duration_error > args.duration_tolerance:
                print(
                    f"  no candidate within {args.duration_tolerance:.0%}; "
                    f"using best take ({chosen.duration_error:.1%} duration error)",
                    file=sys.stderr,
                )
            generated_regions = speech_intervals(generated, sample_rate, librosa, np)
            generated = clean_pause_noise(
                generated, generated_regions, sample_rate, np
            )
            chosen_cues.append(Cue(cue.start, cue.end, chosen_text))
            if args.pause_alignment == "source" and intervals and args.fit != "stretch":
                pause_aligned = align_internal_pauses(
                    generated,
                    sample_rate,
                    cue,
                    intervals,
                    generated_regions,
                    np,
                )
                if pause_aligned is not None:
                    print(
                        f"  aligned {len(generated_regions) - 1} internal pause(s) "
                        "to the source waveform",
                        file=sys.stderr,
                    )
                    generated = pause_aligned
                    generated_regions = speech_intervals(
                        generated, sample_rate, librosa, np
                    )
            if args.fit == "natural":
                generated, inserted_pause = add_phrase_pauses(
                    generated,
                    chosen_text,
                    sample_rate,
                    original_voice_duration - len(generated) / sample_rate,
                    np,
                )
                if inserted_pause:
                    generated_regions = speech_intervals(
                        generated, sample_rate, librosa, np
                    )
                    cue_metrics["inserted_phrase_pause_seconds"] = round(inserted_pause, 3)
                    print(
                        f"  inserted {inserted_pause:.2f}s of gentle phrase pauses "
                        "to reduce empty space",
                        file=sys.stderr,
                    )
            should_stretch = args.fit == "stretch"
            # Never time-stretch generated speech: synthetic voices can develop
            # clicks and distortion even with small tempo changes.
            next_start = cues[number].start if number < len(cues) else None
            if args.placement == "center":
                start_sample = placement_start(
                    cue, len(generated), sample_rate, intervals, generated_regions,
                    previous_audio_end / sample_rate, next_start, args.duration_tolerance,
                )
            else:
                start_sample = round(cue.start * sample_rate)
            copy_count = min(slot_samples, len(generated)) if args.fit == "trim" else len(generated)
            required_samples = start_sample + copy_count
            if required_samples > len(timeline):
                timeline = np.pad(timeline, (0, required_samples - len(timeline)))
            copy_end = start_sample + max(0, copy_count)
            timeline[start_sample:copy_end] += generated[:copy_count]
            previous_audio_end = copy_end
            cue_metrics["placed_start"] = round(start_sample / sample_rate, 3)
            cue_metrics["placed_end"] = round(copy_end / sample_rate, 3)
            cue_metrics["placement_shift"] = round(start_sample / sample_rate - cue.start, 3)

        peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
        if peak > 0.99:
            timeline *= 0.99 / peak
        wav = torch.from_numpy(timeline).unsqueeze(0)
        if run_metrics is not None:
            run_metrics["generated_audio_seconds"] = round(len(timeline) / sample_rate, 3)
            run_metrics["sample_rate"] = sample_rate

        if captions_output is not None:
            captions_output.write_text(format_sbv(chosen_cues), encoding="utf-8")

        if args.audio_output:
            audio_output = args.audio_output.expanduser().resolve()
            audio_output.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(audio_output), wav.cpu(), sample_rate)
            print(f"Saved lossless audio: {audio_output}", file=sys.stderr)

        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() in VIDEO_SUFFIXES:
            if args.reference.suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError("video output requires a video reference input")
            generated_wav = Path(directory) / "dubbed.wav"
            torchaudio.save(str(generated_wav), wav.cpu(), sample_rate)
            mux_video(args.reference.resolve(), generated_wav, output)
        else:
            torchaudio.save(str(output), wav.cpu(), sample_rate)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_languages:
        for code, name in LANGUAGES.items():
            print(f"{code}\t{name}")
        return 0
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    args._run_metrics = {}
    try:
        output = run(args)
    except (ValueError, RuntimeError) as error:
        try:
            log_path = append_run_log(
                args, args._run_metrics, started_at, time.monotonic() - started,
                "failed", str(error),
            )
            print(f"Run metrics: {log_path}", file=sys.stderr)
        except OSError as log_error:
            print(f"Could not save run metrics: {log_error}", file=sys.stderr)
        parser.error(str(error))
    try:
        log_path = append_run_log(
            args, args._run_metrics, started_at, time.monotonic() - started, "success"
        )
        print(f"Run metrics: {log_path}", file=sys.stderr)
    except OSError as error:
        print(f"Could not save run metrics: {error}", file=sys.stderr)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
