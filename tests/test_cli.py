import argparse
from pathlib import Path
import unittest

from voice_dub.cli import (
    Cue, active_duration, align_internal_pauses, build_parser, choose_device,
    extract_text_options, format_sbv, parse_timed_text, refine_cue_timing,
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
        self.assertEqual(args.candidates, 1)
        self.assertEqual(args.timing, "waveform")
        self.assertEqual(args.duration_tolerance, 0.25)
        self.assertEqual(args.placement, "center")
        self.assertEqual(args.pause_alignment, "source")
        self.assertEqual(args.translation_variants, 3)
        self.assertEqual(args.accent, "auto")

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

    def test_auto_device_prefers_mps_when_cuda_is_unavailable(self):
        self.assertEqual(choose_device(FakeTorch(), "auto"), "mps")


if __name__ == "__main__":
    unittest.main()
