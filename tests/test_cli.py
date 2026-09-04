import argparse
import json
from pathlib import Path
import tempfile
import unittest

from voice_dub.cli import (
    DEFAULT_VOICE_REFERENCE, GAP_LIMIT, MASTER_FILTER, Candidate, Cue, active_duration,
    align_internal_pauses, append_run_log, build_parser, default_voice_reference,
    choose_device,
    add_phrase_pauses, estimated_spoken_duration, estimated_translation_duration,
    phonetic_units, placement_start,
    clean_pause_noise, extract_text_options, format_sbv, parse_timed_text,
    speech_only_reference, fade_edges, master_filter, LOUDNESS,
    resample_filter, verify_video_unchanged, raised_cosine, room_tone,
    ROOM_TONE_HEADROOM,
    refine_cue_timing, trim_to_speech,
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

    def test_long_pauses_are_softened_but_short_gaps_are_not(self):
        import numpy as np

        waveform = np.ones(2000, dtype=np.float32)
        # 0.4 s gap between the regions: long enough to soften.
        cleaned = clean_pause_noise(
            waveform, [(0.1, 0.3), (0.7, 0.9)], sample_rate=1000, np=np,
            fade_ms=25, margin_ms=35, floor_gain=0.18, min_pause=0.30,
        )
        self.assertTrue(np.allclose(cleaned[400:600], 0.18))
        self.assertAlmostEqual(float(cleaned[200]), 1.0, places=5)

    def test_word_gaps_keep_their_ambience(self):
        import numpy as np

        waveform = np.ones(2000, dtype=np.float32)
        # 0.1 s between regions: an inter-word gap, must stay untouched.
        cleaned = clean_pause_noise(
            waveform, [(0.3, 0.6), (0.7, 1.0)], sample_rate=1000, np=np,
            fade_ms=25, margin_ms=35, floor_gain=0.18, min_pause=0.30,
        )
        self.assertTrue(np.allclose(cleaned[600:700], 1.0))


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



    def test_estimates_phonetic_length_instead_of_character_length(self):
        self.assertEqual(phonetic_units("Naturally stressful", "en"), 6)
        self.assertGreater(estimated_spoken_duration("A considerably longer sentence", "en"), 1.0)



    def test_translation_duration_is_source_calibrated(self):
        source = "Это короткая фраза."
        target = "This is a short phrase."
        predicted = estimated_translation_duration(target, source, "ru", "en", 4.0)
        self.assertAlmostEqual(predicted, 4.0, delta=1.0)

    def test_placement_matches_source_voice_onset(self):
        start = placement_start(
            Cue(1.0, 4.0, "Text"), generated_samples=200, sample_rate=100,
            source_regions=[(1.4, 3.8)], generated_regions=[(0.2, 1.8)],
            previous_end=0.8, next_start=4.2,
        )
        self.assertEqual(start, 120)

    def test_placement_uses_earlier_gap_to_avoid_next_line(self):
        start = placement_start(
            Cue(2.0, 4.0, "Text"), generated_samples=260, sample_rate=100,
            source_regions=[(2.1, 3.9)], generated_regions=[(0.1, 2.5)],
            previous_end=1.2, next_start=4.0,
        )
        self.assertEqual(start, 140)

    def test_placement_shifts_later_to_avoid_previous_audio(self):
        start = placement_start(
            Cue(2.0, 4.0, "Text"), generated_samples=150, sample_rate=100,
            source_regions=[(2.0, 3.8)], generated_regions=[(0.0, 1.4)],
            previous_end=2.3, next_start=4.5,
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

    def test_voice_reference_resolves_outside_the_project_directory(self):
        import os

        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                resolved = default_voice_reference()
            finally:
                os.chdir(original)
        self.assertEqual(resolved.name, "ref-voice-best-window.wav")
        self.assertTrue(resolved.is_file())

    def test_placement_uses_trailing_media_time_for_the_final_cue(self):
        # The generated tail is longer than its cue but fits the video, so it
        # should stay at the source onset instead of being pulled earlier.
        start = placement_start(
            Cue(2.0, 4.0, "Text"), generated_samples=230, sample_rate=100,
            source_regions=[(2.0, 4.0)], generated_regions=[(0.0, 2.3)],
            previous_end=1.0, next_start=None, hard_end=4.6,
        )
        self.assertEqual(start, 200)

    def test_placement_pulls_the_final_cue_earlier_when_media_is_too_short(self):
        start = placement_start(
            Cue(2.0, 4.0, "Text"), generated_samples=300, sample_rate=100,
            source_regions=[(2.0, 4.0)], generated_regions=[(0.0, 3.0)],
            previous_end=1.0, next_start=None, hard_end=4.5,
        )
        self.assertEqual(start, 150)

    def test_refined_cues_never_overlap_after_a_collapsed_snap(self):
        cues = [Cue(1.0, 3.0, "One"), Cue(2.9, 3.0, "Two")]
        refined = refine_cue_timing(cues, [(1.0, 3.0), (2.95, 3.0)], search=0.3)
        self.assertGreaterEqual(refined[1].start, refined[0].end)
        self.assertGreater(refined[1].end, refined[1].start)

    def test_curated_variants_split_without_surrounding_spaces(self):
        cues, options = extract_text_options([Cue(1.0, 2.0, "Long wording||Short")])
        self.assertEqual(options[0], ["Long wording", "Short"])
        self.assertEqual(cues[0].text, "Long wording")

    def test_two_pass_loudnorm_uses_the_measured_programme(self):
        measured = {"input_i": "-19.1", "input_tp": "-3.2", "input_lra": "6.4",
                    "input_thresh": "-29.3", "target_offset": "0.4"}
        chain = master_filter(measured)
        self.assertIn("measured_I=-19.1", chain)
        self.assertIn("offset=0.4", chain)
        self.assertIn("linear=true", chain)
        self.assertIn("aresample=48000", chain)
        self.assertIn(f"I={LOUDNESS['I']}", chain)

    def test_single_pass_loudnorm_when_measurement_is_unavailable(self):
        chain = master_filter(None)
        self.assertNotIn("measured_I", chain)
        self.assertIn("loudnorm=", chain)
        self.assertIn("apad", chain)

    def _max_step(self, waveform):
        """Largest sample-to-sample jump — a click is a spike in this."""
        import numpy as np

        return float(np.max(np.abs(np.diff(waveform)))) if len(waveform) > 1 else 0.0

    def test_gate_ramp_starts_at_the_floor_so_edges_do_not_step(self):
        import numpy as np

        # Continuous tone: any step in the output is the gate's own doing.
        t = np.arange(4000, dtype=np.float32) / 4000.0
        waveform = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        smooth = self._max_step(waveform)
        cleaned = clean_pause_noise(
            waveform, [(0.25, 0.75)], sample_rate=4000, np=np,
            fade_ms=20, margin_ms=0, floor_gain=0.06,
        )
        self.assertLess(self._max_step(cleaned), smooth * 3)

    def test_pause_alignment_tapers_chunks_into_silence(self):
        import numpy as np

        # Two loud blocks that start and end at full amplitude.
        waveform = np.zeros(4000, dtype=np.float32)
        waveform[400:1200] = 0.8
        waveform[1600:2400] = 0.8
        aligned = align_internal_pauses(
            waveform, sample_rate=4000, cue=Cue(0.0, 1.0, "Text"),
            source_intervals=[(0.1, 0.3), (0.7, 0.9)],
            generated_intervals=[(0.1, 0.3), (0.4, 0.6)], np=np,
        )
        self.assertIsNotNone(aligned)
        # Without a taper each chunk would step 0.0 -> 0.8 in one sample.
        self.assertLess(self._max_step(aligned), 0.4)

    def test_phrase_pauses_taper_even_short_pieces(self):
        import numpy as np

        waveform = np.full(1000, 0.5, dtype=np.float32)
        widened, inserted = add_phrase_pauses(
            waveform, "a, b, c, and d.", 1000, 2.0, np
        )
        self.assertGreater(inserted, 0.0)
        self.assertLess(self._max_step(widened), 0.25)

    def test_room_tone_loops_the_quiet_passages_to_length(self):
        import numpy as np

        rng = np.random.default_rng(0)
        waveform = (rng.standard_normal(3000) * 0.001).astype(np.float32)
        waveform[1000:2000] = 0.5  # the speech
        bed = room_tone(waveform, 1000, [(1.0, 2.0)], seconds=5.0, np=np)
        self.assertIsNotNone(bed)
        self.assertEqual(len(bed), 5000)
        self.assertLess(float(np.max(np.abs(bed))), 0.1)

    def test_room_tone_needs_a_quiet_passage_to_work_from(self):
        import numpy as np

        waveform = np.ones(1000, dtype=np.float32)
        self.assertIsNone(room_tone(waveform, 1000, [(0.0, 1.0)], 2.0, np))

    def test_room_tone_sits_far_below_the_voice(self):
        self.assertLess(ROOM_TONE_HEADROOM, 0.02)

    def test_raised_cosine_spans_the_requested_gains(self):
        import numpy as np

        ramp = raised_cosine(64, np, 0.06, 1.0)
        self.assertAlmostEqual(float(ramp[0]), 0.06, places=5)
        self.assertAlmostEqual(float(ramp[-1]), 1.0, places=5)
        self.assertTrue(np.all(np.diff(ramp) >= 0))

    def test_edge_fades_leave_the_interior_untouched(self):
        import numpy as np

        waveform = np.ones(1000, dtype=np.float32)
        faded = fade_edges(waveform, sample_rate=1000, np=np, fade_ms=10)
        self.assertEqual(faded[0], 0.0)
        self.assertEqual(faded[-1], 0.0)
        self.assertTrue(np.all(faded[20:980] == 1.0))
        self.assertTrue(np.all(waveform == 1.0))

    def test_resample_filter_never_requests_an_unavailable_engine(self):
        chain = resample_filter()
        self.assertTrue(chain.startswith("aresample=48000"))
        if "soxr" in chain:
            self.assertIn("precision=28", chain)

    def test_empty_mux_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out.mp4"
            out.touch()
            with self.assertRaisesRegex(RuntimeError, "empty file"):
                verify_video_unchanged(Path(directory) / "in.mp4", out)

    def test_missing_mux_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "empty file"):
                verify_video_unchanged(Path(directory) / "in.mp4",
                                       Path(directory) / "missing.mp4")

    def test_variant_start_defaults_to_the_predicted_wording(self):
        args = build_parser().parse_args(["v.wav", "-l", "en"])
        self.assertEqual(args.variant_start, "predicted")
        self.assertEqual(args.room_tone, "off")

    def test_gap_limit_matches_the_documented_tolerance(self):
        self.assertEqual(GAP_LIMIT, 0.2)

    def test_trims_synthesizer_padding_and_shifts_regions(self):
        import numpy as np

        waveform = np.zeros(1000, dtype=np.float32)
        waveform[300:700] = 1.0
        trimmed, regions = trim_to_speech(
            waveform, [(0.3, 0.7)], sample_rate=1000, margin_ms=50
        )
        self.assertEqual(len(trimmed), 500)
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0][0], 0.05)
        self.assertAlmostEqual(regions[0][1], 0.45)

    def test_trim_keeps_audio_without_detected_speech(self):
        import numpy as np

        waveform = np.ones(10, dtype=np.float32)
        trimmed, regions = trim_to_speech(waveform, [], sample_rate=1000)
        self.assertIs(trimmed, waveform)
        self.assertEqual(regions, [])

    def test_duration_estimate_matches_measured_synthesis_pace(self):
        # Measured from a Chatterbox take: 29 vowel groups, two commas, 4.36s
        # of speech once the vocoder padding is trimmed off.
        text = ("It's such a complex setup, it really drains all your energy, "
                "because by the time you set up one")
        self.assertEqual(phonetic_units(text, "en"), 29)
        self.assertAlmostEqual(estimated_spoken_duration(text, "en"), 4.36, delta=0.5)


if __name__ == "__main__":
    unittest.main()
