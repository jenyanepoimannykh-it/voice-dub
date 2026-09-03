from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import tempfile
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


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


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
        help="optional cleaner voice-only recording for cloning; the main input still controls timing",
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
        help="alternative translations evaluated per cue (default: 3)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("dubbed.wav"),
        help="output WAV, or a video to retain the input picture (default: dubbed.wav)",
    )
    parser.add_argument(
        "--audio-output", type=Path,
        help="also save the generated audio as lossless WAV",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--exaggeration", type=float, default=0.4, help="emotion strength (default: 0.4)")
    parser.add_argument("--cfg-weight", type=float, help="voice/pacing guidance (automatic by default)")
    parser.add_argument("--temperature", type=float, default=0.7, help="generation randomness (default: 0.7)")
    parser.add_argument(
        "--accent", choices=("auto", "american"), default="auto",
        help="target pronunciation preset; american is available for English",
    )
    parser.add_argument("--seed", type=int, help="optional reproducible random seed")
    parser.add_argument(
        "--candidates", type=int, default=1,
        help="generate this many versions per cue and choose the closest voice match (default: 1)",
    )
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
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,apad", "-shortest", str(output),
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
    fade_ms: float = 12.0,
) -> object:
    """Silence non-speech gaps and gently fade speech edges to prevent clicks."""
    if not intervals:
        return waveform
    cleaned = np.zeros_like(waveform)
    fade_samples = max(1, round(sample_rate * fade_ms / 1000.0))
    for start, end in intervals:
        left = max(0, round(start * sample_rate))
        right = min(len(waveform), round(end * sample_rate))
        if right <= left:
            continue
        cleaned[left:right] = waveform[left:right]
        fade = min(fade_samples, (right - left) // 2)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float32)
            cleaned[left:left + fade] *= ramp
            cleaned[right - fade:right] *= ramp[::-1]
    return cleaned


def active_duration(cue: Cue, intervals: list[tuple[float, float]]) -> float:
    """Measure voiced time inside a cue, excluding detected pauses."""
    return sum(
        max(0.0, min(cue.end, end) - max(cue.start, start))
        for start, end in intervals
    )


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
        # Keep the model's best beam, then favor concise alternatives for dubbing.
        alternatives = [pool[0]]
        for text in sorted(pool[1:], key=lambda value: (len(value.split()), len(value))):
            if text not in alternatives:
                alternatives.append(text)
            if len(alternatives) == variant_count:
                break
        for variant_number, text in enumerate(alternatives, start=1):
            print(f'  variant {variant_number}: "{text}"', file=sys.stderr)
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
    if args.candidates < 1:
        raise ValueError("--candidates must be at least 1")
    if args.translation_variants < 1:
        raise ValueError("--translation-variants must be at least 1")
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
    with tempfile.TemporaryDirectory(prefix="voice-dub-") as directory:
        timing_wav = Path(directory) / "timing.wav"
        extract_reference(args.reference.resolve(), timing_wav)
        reference_wav = timing_wav
        if args.voice_reference:
            reference_wav = Path(directory) / "voice-reference.wav"
            extract_reference(args.voice_reference.expanduser().resolve(), reference_wav)
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
        model.prepare_conditionals(str(reference_wav), exaggeration=args.exaggeration)
        sample_rate = model.sr
        timeline = np.zeros(round(cues[-1].end * sample_rate), dtype=np.float32)

        reference_embedding = model.conds.t3.speaker_emb.squeeze().detach().cpu().numpy()
        reference_embedding /= np.linalg.norm(reference_embedding) + 1e-8
        chosen_cues: list[Cue] = []

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
            minimum_duration = original_voice_duration * (1 - args.duration_tolerance)
            candidates: list[tuple[float, int, float, object, str]] = []
            shortest = None
            source_chunk_count = sum(
                end > cue.start and start < cue.end for start, end in intervals
            )
            text_options = cue_text_options[number - 1]
            total_candidates = len(text_options) * args.candidates
            candidate_number = 0
            for text_number, candidate_text in enumerate(text_options, start=1):
                if len(text_options) > 1:
                    print(
                        f'  translation {text_number}/{len(text_options)}: "{candidate_text}"',
                        file=sys.stderr,
                    )
                for _ in range(args.candidates):
                    candidate_number += 1
                    generated = model.generate(
                        candidate_text,
                        language_id=args.language,
                        exaggeration=args.exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=args.temperature,
                    ).squeeze().cpu().numpy()
                    if shortest is None or len(generated) < len(shortest[0]):
                        shortest = (generated, candidate_text)
                    if len(generated) <= slot_samples:
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
                        if generated_duration >= minimum_duration:
                            candidates.append(
                                (duration_error, pause_mismatch, -similarity, generated, candidate_text)
                            )
                            status = (
                                f"duration error {duration_error:.1%}, "
                                f"pause mismatch {pause_mismatch}, voice score {similarity:.3f}"
                            )
                        else:
                            status = (
                                f"too short; source voice is {original_voice_duration:.2f}s, "
                                f"voice score {similarity:.3f}"
                            )
                        print(
                            f"  candidate {candidate_number}/{total_candidates}: "
                            f"{generated_duration:.2f}s, {status}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  candidate {candidate_number}/{total_candidates}: "
                            f"{len(generated) / sample_rate:.2f}s (too long)",
                            file=sys.stderr,
                        )
            if candidates:
                # Timing leads; matching pause structure and voice break near ties.
                _, _, _, generated, chosen_text = min(
                    candidates,
                    key=lambda item: (round(item[0], 2), item[1], item[2]),
                )
            else:
                assert shortest is not None
                if args.fit == "natural":
                    raise RuntimeError(
                        f"cue {number} has no natural candidate within {args.duration_tolerance:.0%} "
                        f"of the original {original_voice_duration:.2f}s speech duration; "
                        "revise the translation or generate more candidates"
                    )
                generated, chosen_text = shortest
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
            should_stretch = args.fit == "stretch"
            if should_stretch and len(generated) != slot_samples:
                rate = len(generated) / slot_samples
                print(
                    f"  fitting {len(generated) / sample_rate:.2f}s of speech "
                    f"into {slot_samples / sample_rate:.2f}s (speed {rate:.2f}x)",
                    file=sys.stderr,
                )
                generated = librosa.effects.time_stretch(generated, rate=rate)
            leading_padding = 0
            if args.placement == "center" and len(generated) < slot_samples:
                leading_padding = (slot_samples - len(generated)) // 2
            start_sample = round(cue.start * sample_rate) + leading_padding
            copy_count = min(slot_samples, len(generated), len(timeline) - start_sample)
            copy_end = start_sample + max(0, copy_count)
            timeline[start_sample:copy_end] += generated[:copy_count]

        peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
        if peak > 0.99:
            timeline *= 0.99 / peak
        wav = torch.from_numpy(timeline).unsqueeze(0)

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
    try:
        output = run(args)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
