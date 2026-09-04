"""Denoise the best-cloning window and re-test: similarity AND clean output."""
import sys, json, warnings, subprocess
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, librosa, torch, torchaudio
from pathlib import Path
from voice_dub.cli import speech_intervals, speech_only_reference, trim_to_speech

SRC = ".work/originals/dvoice.mp3"
work = Path(".work/clean"); work.mkdir(parents=True, exist_ok=True)
# (tag, start, ffmpeg filter chain)
VARIANTS = [
    ("t323_nr08", 323.0, "highpass=f=60,afftdn=nr=8:nf=-45,dynaudnorm=g=3:p=0.9"),
    ("t323_nr14", 323.0, "highpass=f=60,afftdn=nr=14:nf=-42"),
    ("t323_nr20", 323.0, "highpass=f=60,afftdn=nr=20:nf=-38"),
    ("t165_nr14", 165.5, "highpass=f=60,afftdn=nr=14:nf=-42"),
    ("t323_raw",  323.0, "highpass=f=60"),
]
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
sr = model.sr
LINES = [
    "it's a really complicated setup, and honestly it drains all your energy, "
    "because by the time you finally set up one",
    "and of course, it all gets pretty stressful,",
    "and you can't even think about making an informative podcast anymore. "
    "you just get completely exhausted.",
]
rows = []
for tag, t, chain in VARIANTS:
    cut = work / f"{tag}.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-ss",str(t),"-t","13","-i",SRC,
                    "-vn","-ac","1","-ar","24000",
                    "-af", chain + ",alimiter=limit=0.7",
                    "-c:a","pcm_s24le",str(cut)], check=True)
    y, rate = librosa.load(cut, sr=None, mono=True)
    iv = speech_intervals(y, rate, librosa, np)
    cleaned = speech_only_reference(y, rate, iv, np) if iv else y
    prep = work / f"{tag}.prep.wav"
    torchaudio.save(str(prep), torch.from_numpy(cleaned).unsqueeze(0), rate)

    torch.manual_seed(91)
    model.prepare_conditionals(str(prep), exaggeration=0.4)
    ref = model.conds.t3.speaker_emb.squeeze().detach().cpu().numpy()
    ref /= np.linalg.norm(ref) + 1e-8
    sims, snrs, durs = [], [], []
    for i, line in enumerate(LINES):
        w = model.generate(line, language_id="en", exaggeration=0.4,
                           cfg_weight=0.55, temperature=0.6).squeeze().cpu().numpy()
        iv2 = speech_intervals(w, sr, librosa, np)
        w, iv2 = trim_to_speech(w, iv2, sr)
        durs.append(round(len(w)/sr, 2))
        g = librosa.resample(w, orig_sr=sr, target_sr=16000)
        e = model.ve.embeds_from_wavs([g], sample_rate=16000).mean(axis=0)
        e /= np.linalg.norm(e) + 1e-8
        sims.append(float(np.dot(ref, e)))
        m = np.zeros(len(w), bool)
        for a, b in iv2: m[int(a*sr):int(b*sr)] = True
        if (~m).sum() > sr*0.05:
            snrs.append(20*np.log10((np.sqrt(np.mean(w[m]**2))+1e-12)/(np.sqrt(np.mean(w[~m]**2))+1e-12)))
        torchaudio.save(str(work/f"{tag}.take{i+1}.wav"), torch.from_numpy(w).unsqueeze(0), sr)
    r = dict(tag=tag, ref_seconds=round(len(cleaned)/rate,2),
             sim_mean=round(float(np.mean(sims)),4), sim_min=round(float(min(sims)),4),
             out_snr=round(float(np.mean(snrs)),1) if snrs else None, durs=durs)
    rows.append(r)
    print(f"  {tag:12s} ref {r['ref_seconds']:5.2f}s sim {r['sim_mean']:.4f} "
          f"(min {r['sim_min']:.4f}) outSNR {r['out_snr']}", file=sys.stderr)
json.dump(rows, open(".work/clean_ab.json","w"), indent=1)
rows.sort(key=lambda r: -(r["sim_mean"] + 0.003*(r["out_snr"] or 0)))
print(f"\n{'variant':12s} {'ref s':>6} {'sim mean':>9} {'sim min':>8} {'outSNR':>7}  durations")
for r in rows:
    print(f"{r['tag']:12s} {r['ref_seconds']:6.2f} {r['sim_mean']:9.4f} {r['sim_min']:8.4f} {str(r['out_snr']):>7}  {r['durs']}")
