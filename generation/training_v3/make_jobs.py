"""Job list: data-volume scaling (4x512) and capacity tests at full N."""
import json, os
V3 = os.path.expanduser("~/rds/ede-v3"); DATA = f"{V3}/data"; CLEAN = os.path.expanduser("~/pk_ede/retrain")
specs = json.load(open(f"{V3}/v2_specs.json"))
BIG = {"TT", "EE", "BB", "PP", "TE", "uTT", "uEE", "uTE", "DAZ", "HZ", "S8Z"}

def job(obs, n, hidden="512,512,512,512", pca=None, seed=0, tag=""):
    pca = specs[obs]["pca"] if pca is None else pca
    a = ["--obs", obs, "--ntrain", str(n), "--hidden", hidden, "--pca", str(pca), "--seed", str(seed),
         "--inputs", ",".join(specs[obs]["inputs"])]
    if obs in ("PKNL", "PKNLCB"):
        files = [f"{CLEAN}/{obs}_{s}_{t}_clean.npz" for s in ("training", "testing") for t in ("params", "processed_data")]
        a += ["--params-file", files[0], "--features-file", files[1], "--test-params-file", files[2], "--test-features-file", files[3]]
        a += ["--data", CLEAN]
    else:
        files = [f"{DATA}/{obs}_{s}_{t}.npz" for s in ("training", "testing") for t in ("params", "processed_data")]
    name = f"{obs}_N{n or 'all'}_h{hidden.replace(',', 'x')}_pca{pca}_s{seed}{tag}"
    return {"name": name, "slots": 3 if obs in BIG else 1, "needs": files, "args": a}

jobs = []
for obs in ["PKL", "DER", "uBB", "PKNL", "DAZ", "HZ", "S8Z", "TT", "EE", "PP", "BB", "TE"]:
    for n in (25000, 50000, 100000, 200000, 0):
        jobs.append(job(obs, n))
for obs in ["TT", "PP", "TE", "DAZ"]:
    for h in ("1024,1024,1024,1024", "1024,1024,1024,1024,1024,1024"):
        jobs.append(job(obs, 0, hidden=h))
jobs.append(job("TE", 0, pca=128))
json.dump(jobs, open(f"{V3}/jobs_scaling.json", "w"), indent=1)
print(len(jobs), "jobs")
