"""CMB jobs on the common split (data_cs): scaling, capacity, TE strategies (PCA 64/128/256, r_l)."""
import json, os
V3 = os.path.expanduser("~/rds/ede-v3"); CS = f"{V3}/data_cs"
specs = json.load(open(f"{V3}/v2_specs.json"))
LR2 = "1e-2,1e-3,1e-4,1e-5,1e-6"
def job(obs, n, hidden="512,512,512,512", pca=0, lrs=LR2, tag=""):
    base = "TE" if obs == "rTE" else obs
    a = ["--obs", obs, "--data", CS, "--ntrain", str(n), "--hidden", hidden, "--pca", str(pca), "--seed", "0",
         "--inputs", ",".join(specs[base]["inputs"]), "--lrs", lrs]
    files = [f"{CS}/{obs}_{s}_{t}.npz" for s in ("training", "testing") for t in ("params", "processed_data")]
    name = f"cs_{obs}_N{n or 'all'}_h{hidden.replace(',', 'x')}_pca{pca}{tag}"
    return {"name": name, "slots": 3 if obs != "uBB" else 1, "needs": files, "args": a}
H4, W4 = "512,512,512,512", "1024,1024,1024,1024"
jobs = []
for obs in ("TT", "EE", "PP", "BB"):
    for n in (25000, 50000, 100000, 200000):
        jobs.append(job(obs, n))
    jobs += [job(obs, 0, H4), job(obs, 0, W4)]
jobs += [job("TE", 0, W4, pca=p) for p in (64, 128, 256)]
jobs += [job("rTE", 0, H4), job("rTE", 0, W4), job("rTE", 0, W4, pca=128)]
for obs in ("uTT", "uEE", "uBB"):
    jobs.append(job(obs, 0, W4))
jobs.append(job("uTE", 0, W4, pca=128))
old = json.load(open(f"{V3}/jobs_all.json"))
CMB = {"TT", "EE", "PP", "BB", "TE", "uTT", "uEE", "uTE", "uBB"}
keep = [j for j in old if j["args"][1] not in CMB or os.path.exists(f"{V3}/runs/{j['name']}_eval.json")]
json.dump(keep + jobs, open(f"{V3}/jobs_all.json", "w"), indent=1)
print(f"{len(keep)} non-CMB/finished jobs kept, {len(jobs)} common-split CMB jobs added")
