"""Search a long recording for the best conditioning window.

A good Chatterbox prompt is continuous speech, quiet between words, wide-band,
dry, and expressive without being erratic. Scores every candidate window and
returns the best few.
"""
import sys, warnings, json
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import numpy as np, librosa
from pathlib import Path
from voice_dub.cli import speech_intervals

WIN, HOP = 10.0, 0.5
SKIP_HEAD, SKIP_TAIL = 45.0, 30.0     # LibriVox intro / outro boilerplate

def best_windows(path, top=6):
    y, sr = librosa.load(path, sr=None, mono=True)
    dur = len(y)/sr
    iv = speech_intervals(y, sr, librosa, np)
    voiced = np.zeros(len(y), bool)
    for a, b in iv: voiced[int(a*sr):int(b*sr)] = True
    f0, _, _ = librosa.pyin(y, sr=sr, fmin=60, fmax=400, frame_length=2048,
                            hop_length=512)
    f0_t = np.arange(len(f0)) * 512 / sr
    rows = []
    t = SKIP_HEAD
    while t + WIN < dur - SKIP_TAIL:
        s, e = int(t*sr), int((t+WIN)*sr)
        seg, vm = y[s:e], voiced[s:e]
        cover = vm.mean()
        if cover < 0.80:
            t += HOP; continue
        peak = float(np.max(np.abs(seg)))
        if peak >= 0.999:
            t += HOP; continue
        sp, ns = seg[vm], seg[~vm]
        if ns.size < sr*0.05:
            t += HOP; continue
        snr = 20*np.log10((np.sqrt(np.mean(sp**2))+1e-12)/(np.sqrt(np.mean(ns**2))+1e-12))
        S = np.abs(librosa.stft(sp, n_fft=2048))**2
        psd = S.mean(axis=1); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
        tot = psd.sum(); cum = np.cumsum(psd)/tot
        roll = fr[np.searchsorted(cum, 0.99)]
        pres = 10*np.log10(psd[(fr>=3000)&(fr<8000)].sum()/tot + 1e-12)
        m = (f0_t >= t) & (f0_t < t+WIN)
        fseg = f0[m]; fv = fseg[~np.isnan(fseg)]
        if fv.size < 20:
            t += HOP; continue
        f0med = float(np.median(fv))
        # Expressive but not erratic: semitone spread of the middle 80%.
        spread = 12*np.log2(np.percentile(fv,90)/max(np.percentile(fv,10),1e-6))
        rows.append(dict(start=round(t,2), cover=round(float(cover),3),
                         snr=round(float(snr),1), roll99=int(roll),
                         pres=round(float(pres),1), f0=round(f0med,1),
                         spread=round(float(spread),2), peak_dbfs=round(20*np.log10(peak+1e-12),1)))
        t += HOP
    def score(r):
        n = lambda v, lo, hi: max(0.0, min(1.0, (v-lo)/(hi-lo)))
        expressive = max(0.0, 1.0 - abs(r["spread"] - 7.0)/7.0)  # ~7 semitones reads natural
        return round(0.28*n(r["snr"],25,50) + 0.20*n(r["roll99"],5000,11000)
                     + 0.16*n(r["pres"],-20,-8) + 0.18*n(r["cover"],0.80,0.98)
                     + 0.18*expressive, 4)
    for r in rows: r["score"] = score(r)
    rows.sort(key=lambda r: -r["score"])
    # keep non-overlapping picks
    picks = []
    for r in rows:
        if all(abs(r["start"]-p["start"]) >= WIN for p in picks):
            picks.append(r)
        if len(picks) >= top: break
    return picks

for name in ("dvoice", "tamurile"):
    p = f".work/originals/{name}.mp3"
    print(f"=== {name} ===")
    picks = best_windows(p)
    for r in picks:
        print(f"  t={r['start']:7.2f}s score {r['score']:.3f}  SNR {r['snr']:5.1f}  "
              f"roll99 {r['roll99']:5d}  pres {r['pres']:6.1f}  cover {r['cover']:.2f}  "
              f"F0 {r['f0']:5.1f}  spread {r['spread']:4.1f}st  peak {r['peak_dbfs']:5.1f}")
    json.dump(picks, open(f".work/windows_{name}.json","w"), indent=1)
