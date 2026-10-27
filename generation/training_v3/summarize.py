"""Scaling summary: 95th percentile of the test error (median and max over modes) per run, with v2 references."""
import glob, json, os
V3 = os.path.expanduser("~/rds/ede-v3")
rows = {}
for f in glob.glob(f"{V3}/runs/*_eval.json"):
    r = json.load(open(f))
    if "median_over_modes" not in r:
        continue
    name = os.path.basename(f).replace("_eval.json", "")
    obs = r["obs"]
    lin = obs in ("TE", "uTE", "rTE")
    conv = (lambda x: x) if lin else (lambda x: 100 * (10 ** x - 1))
    rows.setdefault(obs, []).append((name, conv(r["median_over_modes"]["95"]), conv(r["max_over_modes"]["95"]), r.get("train_time_s", 0)))
for obs in sorted(rows):
    unit = "feature units" if obs in ("TE", "uTE", "rTE") else "%"
    print(f"\n{obs}  (95th pct of |emu - CLASS|: median / max over modes, {unit})")
    for name, med, mx, t in sorted(rows[obs], key=lambda r: (not r[0].startswith("v2"), r[1])):
        print(f"  {name:62s} {med:9.4f} {mx:9.4f}  {t/60:5.1f} min")
