"""A/B candidate voice references through Chatterbox itself.

Metrics that matter for a conditioning prompt: how closely the model reproduces
the reference speaker, how stable the pace is, and how clean the output is.
"""
import sys, json, warnings, time
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, librosa, torch, torchaudio
from pathlib import Path
from voice_dub.cli import speech_intervals, speech_only_reference, trim_to_speech, extract_reference

CANDIDATES = ["stuart_bell", "dvoice", "tamurile", "sean_daeley",
              "larry_wilson", "padraig_o'hiceadha-lyrical", "brett_condron"]
LINES = [
    "it's a really complicated setup, and honestly it drains all your energy, "
    "because by the time you finally set up one",
    "and of course, it all gets pretty stressful,",
]
work = Path(".work/ab"); work.mkdir(parents=True, exist_ok=True)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"loading model on {device}...", file=sys.stderr)
model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
sr = model.sr

results = []
for name in CANDIDATES:
    src = Path(f".work/candidates/{name}.flac")
    if not src.is_file():
        print("missing", src, file=sys.stderr); continue
    # Prepare exactly like the pipeline does: 24 kHz mono, speech-only.
    raw = work / f"{name}.raw.wav"
    extract_reference(src.resolve(), raw)
    y, rate = librosa.load(raw, sr=None, mono=True)
    iv = speech_intervals(y, rate, librosa, np)
    cleaned = speech_only_reference(y, rate, iv, np) if iv else y
    prep = work / f"{name}.prep.wav"
    torchaudio.save(str(prep), torch.from_numpy(cleaned).unsqueeze(0), rate)

    torch.manual_seed(91)
    model.prepare_conditionals(str(prep), exaggeration=0.4)
    ref_emb = model.conds.t3.speaker_emb.squeeze().detach().cpu().numpy()
    ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)

    sims, durs, noise = [], [], []
    for i, line in enumerate(LINES):
        t0 = time.monotonic()
        wav = model.generate(line, language_id="en", exaggeration=0.4,
                             cfg_weight=0.55, temperature=0.6).squeeze().cpu().numpy()
        gen_iv = speech_intervals(wav, sr, librosa, np)
        wav_t, gen_iv = trim_to_speech(wav, gen_iv, sr)
        durs.append(len(wav_t)/sr)
        g16 = librosa.resample(wav_t, orig_sr=sr, target_sr=16000)
        e = model.ve.embeds_from_wavs([g16], sample_rate=16000).mean(axis=0)
        e = e / (np.linalg.norm(e) + 1e-8)
        sims.append(float(np.dot(ref_emb, e)))
        m = np.zeros(len(wav_t), bool)
        for a, b in gen_iv: m[int(a*sr):int(b*sr)] = True
        if (~m).sum() > sr*0.05 and m.sum() > 0:
            noise.append(20*np.log10((np.sqrt(np.mean(wav_t[m]**2))+1e-12) /
                                     (np.sqrt(np.mean(wav_t[~m]**2))+1e-12)))
        torchaudio.save(str(work / f"{name}.take{i+1}.wav"),
                        torch.from_numpy(wav_t).unsqueeze(0), sr)
        print(f"  {name} take{i+1}: {durs[-1]:.2f}s sim {sims[-1]:.4f} "
              f"({time.monotonic()-t0:.0f}s)", file=sys.stderr)
    results.append({
        "name": name,
        "ref_seconds": round(len(cleaned)/rate, 2),
        "similarity_mean": round(float(np.mean(sims)), 4),
        "similarity_min": round(float(min(sims)), 4),
        "durations": [round(d, 2) for d in durs],
        "out_snr_db": round(float(np.mean(noise)), 1) if noise else None,
    })
    json.dump(results, open(".work/ab_results.json", "w"), indent=1)

results.sort(key=lambda r: -r["similarity_mean"])
print(f"\n{'candidate':30s} {'sim(mean)':>9} {'sim(min)':>9} {'ref s':>6} {'outSNR':>7}  durations")
for r in results:
    print(f"{r['name']:30s} {r['similarity_mean']:9.4f} {r['similarity_min']:9.4f} "
          f"{r['ref_seconds']:6.2f} {str(r['out_snr_db']):>7}  {r['durations']}")
