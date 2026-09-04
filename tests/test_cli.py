import argparse
import json
from pathlib import Path
import tempfile
import unittest

from voice_dub.cli import (
    DEFAULT_VOICE_REFERENCE, MASTER_FILTER, Candidate, Cue, active_duration, align_internal_pauses, append_run_log, build_parser,
    choose_candidate, choose_device, choose_text_for_duration,
    add_phrase_pauses, estimated_spoken_duration, estimated_translation_duration,
    phonetic_units, placement_start,
    clean_pause_noise, extract_text_options, format_sbv, parse_timed_text,
    speech_only_reference,
    refine_cue_timing,
)


class FakeMps:
    @staticmethod
    def is_available():
        return True


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    cuda = FakeCuda()
    backends = argparse.Namespace(mps=FakeMps())


class CliTests(unittest.TestCase):
    def test_parses_required_generation_options(self):
        args = build_parser().parse_args([
            "voice.mp3", "--text-file", "captions.txt",
            "--source-language", "ru", "-l", "fr",
        ])
        self.assertEqual(args.reference, Path("voice.mp3"))
        self.assertEqual(args.language, "fr")
        self.assertEqual(args.source_language, "ru")
        self.assertEqual(args.fit, "natural")
        self.assertEqual(args.timing, "waveform")
        self.assertEqual(args.duration_tolerance, 0.25)
        self.assertEqual(args.placement, "center")
        self.assertEqual(args.pause_alignment, "source")
        self.assertEqual(args.translation_variants, 3)
        self.assertEqual(args.accent, "auto")
        self.assertEqual(args.cfg_weight, 0.55)
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(args.seed, 91)
        self.assertEqual(DEFAULT_VOICE_REFERENCE.name, "ref-voice-best-window.wav")
        self.assertEqual(DEFAULT_VOICE_REFERENCE.parent.name, "reference")
        self.assertIsNone(args.voice_reference)
        self.assertTrue(DEFAULT_VOICE_REFERENCE.is_file())
        self.assertIn("adeclick=", MASTER_FILTER)
        self.assertIn("m=s", MASTER_FILTER)

    def test_rejects_removed_multiple_candidate_option(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["voice.wav", "--candidates", "2"])

    def test_aligns_generated_chunks_using_source_pause_proportions(self):
        import numpy as np

        waveform = np.zeros(100, dtype=np.float32)
        waveform[10:30] = 1.0
        waveform[40:60] = 2.0
        aligned = align_internal_pauses(
            waveform,
            sample_rate=100,
            cue=Cue(0.0, 1.0, "Text"),
            source_intervals=[(0.1, 0.3), (0.7, 0.9)],
            generated_intervals=[(0.1, 0.3), (0.4, 0.6)],
            np=np,
        )
        self.assertIsNotNone(aligned)
        nonzero = np.flatnonzero(aligned)
        self.assertEqual((nonzero[0], nonzero[-1]), (10, 89))
        self.assertTrue(np.all(aligned[10:30] == 1.0))
        self.assertTrue(np.all(aligned[70:90] == 2.0))

    def test_extracts_curated_text_options(self):
        cues, options = extract_text_options([
            Cue(1.0, 2.0, "Natural wording || Short wording || Natural wording")
        ])
        self.assertEqual(cues[0].text, "Natural wording")
        self.assertEqual(options[0], ["Natural wording", "Short wording"])

    def test_same_language_translation_returns_one_option_per_cue(self):
        from voice_dub.cli import translate_cues

        original = [Cue(0.0, 1.0, "Hello")]
        cues, options = translate_cues(original, "en", "en", "cpu", 3)
        self.assertEqual(cues, original)
        self.assertEqual(options, [["Hello"]])

    def test_cleans_noise_between_speech_regions(self):
        import numpy as np

        waveform = np.ones(1000, dtype=np.float32)
        cleaned = clean_pause_noise(
            waveform, [(0.1, 0.3), (0.7, 0.9)], sample_rate=1000, np=np, fade_ms=10
        )
        self.assertTrue(np.allclose(cleaned[335:665], 0.06))
        self.assertLess(cleaned[65], cleaned[70])
        self.assertLess(cleaned[330], cleaned[310])

    def test_speech_only_reference_removes_long_gaps(self):
        import numpy as np

        waveform = np.arange(1000, dtype=np.float32)
        reference = speech_only_reference(
            waveform, 1000, [(0.1, 0.2), (0.8, 0.9)], np, max_seconds=1.0
        )
        self.assertEqual(len(reference), 280)
        self.assertEqual(reference[0], 100.0)
        self.assertEqual(reference[100], 0.0)
        self.assertEqual(reference[-1], 899.0)

    def test_speech_only_reference_has_no_leading_or_trailing_separator(self):
        import numpy as np

        waveform = np.ones(1000, dtype=np.float32)
        reference = speech_only_reference(
            waveform, 1000, [(0.2, 0.3)], np, max_seconds=1.0
        )
        self.assertEqual(len(reference), 100)
        self.assertTrue(np.all(reference == 1.0))

    def test_parses_srt_style_timed_text(self):
        cues = parse_timed_text(
            "1\n00:00:01,250 --> 00:00:03,500\nПривет!\n\n"
            "2\n00:00:04.000 --> 00:00:06.000\nКак дела?\n"
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].start, 1.25)
        self.assertEqual(cues[0].text, "Привет!")
        self.assertEqual(cues[1].end, 6.0)

    def test_parses_youtube_sbv_timed_text(self):
        cues = parse_timed_text(
            "0:00:00.000,0:00:05.080\nПервая строка\nвторая строка\n\n"
            "0:00:05.080,0:00:11.960\nСледующая фраза\n"
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].end, 5.08)
        self.assertEqual(cues[0].text, "Первая строка вторая строка")

    def test_rejects_text_without_timestamps(self):
        with self.assertRaisesRegex(ValueError, "expected a timestamp"):
            parse_timed_text("Привет!")

    def test_formats_sbv(self):
        cues = parse_timed_text("0:00:01.250,0:00:03.500\nHello\n")
        self.assertEqual(format_sbv(cues), "0:00:01.250,0:00:03.500\nHello\n")

    def test_refines_boundaries_only_to_nearby_speech_edges(self):
        cues = [Cue(1.0, 3.0, "One"), Cue(4.0, 6.0, "Two")]
        refined = refine_cue_timing(cues, [(1.12, 2.82), (4.5, 6.8)], search=0.3)
        self.assertEqual(refined[0], Cue(1.12, 2.82, "One"))
        self.assertEqual(refined[1], Cue(4.0, 6.0, "Two"))

    def test_measures_only_active_speech_inside_cue(self):
        cue = Cue(1.0, 5.0, "Text")
        self.assertAlmostEqual(active_duration(cue, [(0.5, 2.0), (2.5, 4.0), (6.0, 7.0)]), 2.5)

    def test_chooses_closest_candidate_when_none_are_within_tolerance(self):
        candidates = [
            Candidate(0.40, 0, 0.9, 0.0, "long", "Long wording", 0.3),
            Candidate(0.30, 1, 0.8, 0.0, "closest", "Closest wording", 0.3),
        ]
        self.assertEqual(choose_candidate(candidates, 0.25).waveform, "closest")

    def test_prefers_candidate_within_tolerance(self):
        candidates = [
            Candidate(0.30, 0, 0.9, 0.0, "outside", "Outside wording", 0.3),
            Candidate(0.20, 1, 0.8, 0.0, "inside", "Inside wording", 0.3),
        ]
        self.assertEqual(choose_candidate(candidates, 0.25).waveform, "inside")

    def test_estimates_phonetic_length_instead_of_character_length(self):
        self.assertEqual(phonetic_units("Naturally stressful", "en"), 6)
        self.assertGreater(estimated_spoken_duration("A considerably longer sentence", "en"), 1.0)

    def test_preselects_wording_closest_to_source_duration(self):
        options = ["Very short.", "This wording should take considerably longer to say."]
        target = estimated_spoken_duration(options[1], "en")
        self.assertEqual(choose_text_for_duration(options, target, "en"), options[1])

    def test_duration_selection_never_prefers_overlong_wording(self):
        options = ["A very long wording that exceeds the available window.", "Short wording."]
        selected = choose_text_for_duration(options, 1.5, "en")
        self.assertEqual(selected, "Short wording.")

    def test_translation_duration_is_source_calibrated(self):
        source = "Это короткая фраза."
        target = "This is a short phrase."
        predicted = estimated_translation_duration(target, source, "ru", "en", 4.0)
        self.assertAlmostEqual(predicted, 4.0, delta=1.0)

    def test_placement_matches_source_voice_onset(self):
        start = placement_start(
            Cue(1.0, 4.0, "Text"), generated_samples=200, sample_rate=100,
            source_regions=[(1.4, 3.8)], generated_regions=[(0.2, 1.8)],
            previous_end=0.8, next_start=4.2, tolerance=0.25,
        )
        self.assertEqual(start, 120)

    def test_placement_uses_earlier_gap_to_avoid_next_line(self):
        start = placement_start(
            Cue(2.0, 4.0, "Text"), generated_samples=260, sample_rate=100,
            source_regions=[(2.1, 3.9)], generated_regions=[(0.1, 2.5)],
            previous_end=1.2, next_start=4.0, tolerance=0.25,
        )
        self.assertEqual(start, 140)

    def test_placement_shifts_later_to_avoid_previous_audio(self):
        start = placement_start(
            Cue(2.0, 4.0, "Text"), generated_samples=150, sample_rate=100,
            source_regions=[(2.0, 3.8)], generated_regions=[(0.0, 1.4)],
            previous_end=2.3, next_start=4.5, tolerance=0.25,
        )
        self.assertEqual(start, 230)

    def test_adds_capped_silence_at_phrase_boundaries_without_stretching_audio(self):
        import numpy as np

        waveform = np.ones(1000, dtype=np.float32)
        widened, inserted = add_phrase_pauses(
            waveform, "First phrase, second phrase, and third.", 1000, 2.0, np
        )
        self.assertEqual(inserted, 0.5)
        self.assertEqual(len(widened), 1500)

    def test_does_not_add_phrase_pause_for_small_gap(self):
        import numpy as np

        waveform = np.ones(1000, dtype=np.float32)
        widened, inserted = add_phrase_pauses(
            waveform, "First phrase, second phrase.", 1000, 0.2, np
        )
        self.assertIs(widened, waveform)
        self.assertEqual(inserted, 0.0)

    def test_appends_machine_readable_run_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runs.jsonl"
            args = build_parser().parse_args([
                "missing.wav", "--language", "en", "--output", str(Path(directory) / "out.wav"),
                "--run-log", str(log),
            ])
            append_run_log(args, {"device": "cpu"}, "2026-01-01T00:00:00+00:00", 1.25, "failed", "test")
            record = json.loads(log.read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["device"], "cpu")
            self.assertEqual(record["error"], "test")

    def test_auto_device_prefers_mps_when_cuda_is_unavailable(self):
        self.assertEqual(choose_device(FakeTorch(), "auto"), "mps")


if __name__ == "__main__":
    unittest.main()
