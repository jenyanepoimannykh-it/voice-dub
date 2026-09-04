"""Rank voice references by measurable recording quality, not by claimed gear."""
import sys, json, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf
from pathlib import Path
from voice_dub.cli import speech_intervals

def analyse(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    if y.size == 0:
        return None
    out = {"file": Path(path).name, "sr": sr, "seconds": round(len(y)/sr, 2)}
    iv = speech_intervals(y, sr, librosa, np)
    if not iv:
        return None
    mask = np.zeros(len(y), bool)
    for a, b in iv:
        mask[int(a*sr):int(b*sr)] = True
    speech, noise = y[mask], y[~mask]
    out["speech_seconds"] = round(mask.sum()/sr, 2)
    out["regions"] = len(iv)

    # --- level / clipping ---
    peak = float(np.max(np.abs(y)))
    out["peak_dbfs"] = round(20*np.log10(peak + 1e-12), 2)
    out["clipped_pct"] = round(100*float(np.mean(np.abs(y) >= 0.999)), 4)

    # --- SNR: speech RMS vs noise-floor RMS ---
    srms = float(np.sqrt(np.mean(speech**2))) if speech.size else 0.0
    nrms = float(np.sqrt(np.mean(noise**2))) if noise.size > sr*0.05 else 1e-6
    out["snr_db"] = round(20*np.log10((srms + 1e-12)/(nrms + 1e-12)), 1)
    out["noise_floor_dbfs"] = round(20*np.log10(nrms + 1e-12), 1)

    # --- pitch (voice type) ---
    f0, voiced, _ = librosa.pyin(speech if speech.size > sr else y, sr=sr,
                                 fmin=60, fmax=400, frame_length=2048)
    f0v = f0[~np.isnan(f0)]
    if f0v.size < 10:
        return None
    out["f0_median"] = round(float(np.median(f0v)), 1)
    out["f0_p10"] = round(float(np.percentile(f0v, 10)), 1)
    out["f0_p90"] = round(float(np.percentile(f0v, 90)), 1)
    out["f0_iqr"] = round(float(np.percentile(f0v,75) - np.percentile(f0v,25)), 1)

    # --- bandwidth: where the spectrum actually ends (catches lossy/resampled) ---
    S = np.abs(librosa.stft(speech if speech.size > 4096 else y, n_fft=4096))
    psd = (S**2).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    total = psd.sum()
    cum = np.cumsum(psd)/total
    out["rolloff99_hz"] = int(freqs[np.searchsorted(cum, 0.99)])
    out["rolloff999_hz"] = int(freqs[np.searchsorted(cum, 0.999)])
    top = psd.max()
    alive = freqs[psd > top*1e-6]
    out["max_alive_hz"] = int(alive[-1]) if alive.size else 0

    # --- spectral balance in dB relative to total ---
    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return round(10*np.log10(psd[m].sum()/total + 1e-12), 1)
    out["b_low_80_250"]   = band(80, 250)
    out["b_lowmid_250_800"] = band(250, 800)
    out["b_mid_800_3k"]   = band(800, 3000)
    out["b_pres_3k_8k"]   = band(3000, 8000)
    out["b_air_8k_16k"]   = band(8000, 16000)

    # --- reverb proxy: energy decay in the 250 ms after each speech offset ---
    tails = []
    for a, b in iv:
        s = int(b*sr); e = min(len(y), s + int(0.25*sr))
        pre = y[max(0,int(b*sr)-int(0.15*sr)):s]
        if e - s > sr*0.05 and pre.size > 100:
            t = np.sqrt(np.mean(y[s:e]**2)); p = np.sqrt(np.mean(pre**2))
            if p > 0: tails.append(20*np.log10((t+1e-12)/(p+1e-12)))
    out["tail_drop_db"] = round(float(np.mean(tails)), 1) if tails else None

    # --- harmonics-to-noise ratio (voice clarity) ---
    h, p = librosa.effects.hpss(speech if speech.size > 4096 else y)
    out["hnr_db"] = round(20*np.log10((np.sqrt(np.mean(h**2))+1e-12)/(np.sqrt(np.mean(p**2))+1e-12)), 1)
    return out

rows = []
for f in sorted(Path(".work/candidates").glob("*.flac")):
    try:
        r = analyse(str(f))
        if r: rows.append(r)
    except Exception as e:
        print("skip", f.name, e, file=sys.stderr)
json.dump(rows, open(".work/metrics.json","w"), indent=1)
print("analysed", len(rows))
