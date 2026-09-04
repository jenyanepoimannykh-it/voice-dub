import json, sys
sys.path.insert(0,"src")
rows = json.load(open(".work/metrics.json"))

def norm(v, lo, hi):
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

def score(r):
    f0 = r["f0_median"]
    # Bari-tenor speaking fundamental: baritone ~110, tenor ~150. Peak at 132 Hz.
    fit_f0 = max(0.0, 1.0 - abs(f0 - 132.0) / 38.0)
    band   = norm(r["rolloff99_hz"], 5000, 10500)      # real HF content
    full   = norm(r["max_alive_hz"], 15000, 20500)     # not a lossy/resampled source
    snr    = norm(r["snr_db"], 32, 55)
    dry    = norm(-(r["tail_drop_db"] or -20), 20, 42) # faster decay = drier room
    hnr    = norm(r["hnr_db"], -3, 8)
    pres   = norm(r["b_pres_3k_8k"], -20, -8)          # 3-8k presence, the U87 signature
    air    = norm(r["b_air_8k_16k"], -30, -14)
    clean  = 1.0 if r["clipped_pct"] < 0.01 else 0.0
    return round(
        0.24*fit_f0 + 0.16*band + 0.08*full + 0.14*snr +
        0.12*dry + 0.10*hnr + 0.10*pres + 0.06*air, 4
    ) * clean

for r in rows:
    r["score"] = score(r)
rows.sort(key=lambda r: -r["score"])
hdr = f"{'file':34s} {'score':>6} {'F0':>6} {'roll99':>7} {'alive':>6} {'SNR':>5} {'dry':>6} {'HNR':>5} {'pres':>6} {'air':>6}"
print(hdr); print("-"*len(hdr))
for r in rows[:22]:
    print(f"{r['file'][:34]:34s} {r['score']:6.3f} {r['f0_median']:6.1f} {r['rolloff99_hz']:7d} "
          f"{r['max_alive_hz']:6d} {r['snr_db']:5.1f} {r['tail_drop_db']:6.1f} {r['hnr_db']:5.1f} "
          f"{r['b_pres_3k_8k']:6.1f} {r['b_air_8k_16k']:6.1f}")
print("\n... current bundled fallback for comparison:")
for r in rows:
    if r["file"].startswith("brett_condron"):
        print(f"{r['file'][:34]:34s} {r['score']:6.3f} {r['f0_median']:6.1f} {r['rolloff99_hz']:7d} "
              f"{r['max_alive_hz']:6d} {r['snr_db']:5.1f} {r['tail_drop_db']:6.1f} {r['hnr_db']:5.1f} "
              f"{r['b_pres_3k_8k']:6.1f} {r['b_air_8k_16k']:6.1f}  <- rank {rows.index(r)+1}/{len(rows)}")
json.dump(rows, open(".work/ranked.json","w"), indent=1)
