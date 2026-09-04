"""Pick the final reference: A/B the best dvoice windows through Chatterbox."""
import sys, json, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, librosa, torch, torchaudio, subprocess
from pathlib import Path
from voice_dub.cli import speech_intervals, speech_only_reference, trim_to_speech

SRC = ".work/originals/dvoice.mp3"
# Cut 13 s so the speech-only cleanup can still fill its 10 s cap.
STARTS = [101.5, 165.5, 323.0, 345.0]
work = Path(".work/win"); work.mkdir(parents=True, exist_ok=True)

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

def prepare(tag, wav_in):
    y, rate = librosa.load(wav_in, sr=None, mono=True)
    iv = speech_intervals(y, rate, librosa, np)
    cleaned = speech_only_reference(y, rate, iv, np) if iv else y
    out = work / f"{tag}.prep.wav"
    torchaudio.save(str(out), torch.from_numpy(cleaned).unsqueeze(0), rate)
    return out, len(cleaned)/rate

cands = []
for t in STARTS:
    tag = f"dvoice_t{int(t)}"
    cut = work / f"{tag}.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-ss",str(t),"-t","13",
                    "-i",SRC,"-vn","-ac","1","-ar","24000","-c:a","pcm_s16le",str(cut)],check=True)
    cands.append((tag, *prepare(tag, cut)))
# baseline: the pre-trimmed CC0 clip that scored best earlier
subprocess.run(["ffmpeg","-y","-v","error","-i",".work/candidates/dvoice.flac",
                "-vn","-ac","1","-ar","24000","-c:a","pcm_s16le",str(work/"vz.wav")],check=True)
cands.append(("voicezero_clip", *prepare("voicezero_clip", work/"vz.wav")))

rows = []
for tag, prep, secs in cands:
    torch.manual_seed(91)
    model.prepare_conditionals(str(prep), exaggeration=0.4)
    ref = model.conds.t3.speaker_emb.squeeze().detach().cpu().numpy()
    ref /= np.linalg.norm(ref) + 1e-8
    sims, snrs, durs = [], [], []
    for i, line in enumerate(LINES):
        w = model.generate(line, language_id="en", exaggeration=0.4,
                           cfg_weight=0.55, temperature=0.6).squeeze().cpu().numpy()
        iv = speech_intervals(w, sr, librosa, np)
        w, iv = trim_to_speech(w, iv, sr)
        durs.append(round(len(w)/sr, 2))
        g = librosa.resample(w, orig_sr=sr, target_sr=16000)
        e = model.ve.embeds_from_wavs([g], sample_rate=16000).mean(axis=0)
        e /= np.linalg.norm(e) + 1e-8
        sims.append(float(np.dot(ref, e)))
        m = np.zeros(len(w), bool)
        for a, b in iv: m[int(a*sr):int(b*sr)] = True
        if (~m).sum() > sr*0.05:
            snrs.append(20*np.log10((np.sqrt(np.mean(w[m]**2))+1e-12)/(np.sqrt(np.mean(w[~m]**2))+1e-12)))
        torchaudio.save(str(work/f"{tag}.take{i+1}.wav"), torch.from_numpy(w).unsqueeze(0), sr)
    rows.append(dict(tag=tag, ref_seconds=round(secs,2),
                     sim_mean=round(float(np.mean(sims)),4), sim_min=round(float(min(sims)),4),
                     out_snr=round(float(np.mean(snrs)),1) if snrs else None, durs=durs))
    print(f"  {tag:18s} ref {secs:5.2f}s  sim {rows[-1]['sim_mean']:.4f} "
          f"(min {rows[-1]['sim_min']:.4f})  outSNR {rows[-1]['out_snr']}", file=sys.stderr)

rows.sort(key=lambda r: -(r["sim_mean"] + 0.004*(r["out_snr"] or 0)))
json.dump(rows, open(".work/window_ab.json","w"), indent=1)
print(f"\n{'window':18s} {'ref s':>6} {'sim mean':>9} {'sim min':>8} {'outSNR':>7}  durations")
for r in rows:
    print(f"{r['tag']:18s} {r['ref_seconds']:6.2f} {r['sim_mean']:9.4f} {r['sim_min']:8.4f} {str(r['out_snr']):>7}  {r['durs']}")
