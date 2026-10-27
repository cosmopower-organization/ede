import json, os
exec(open("make_jobs.py").read().split("jobs = []")[0])
extra = []
for obs in ["PKL", "PKLCB", "PKNL", "PKNLCB", "DER", "DAZ", "HZ", "S8Z", "TT", "EE", "PP", "TE", "BB", "uTT", "uEE", "uTE"]:
    for h in ("512,512,512,512", "1024,1024,1024,1024"):
        j = job(obs, 0, hidden=h, tag="_lr1e-2"); j["args"] += ["--lrs", "1e-2,1e-3,1e-4,1e-5,1e-6"]; extra.append(j)
alljobs = json.load(open("jobs_all.json"))
names = {j["name"] for j in alljobs}
alljobs += [j for j in extra if j["name"] not in names]
json.dump(alljobs, open("jobs_all.json", "w"), indent=1)
print(len(alljobs), "jobs in jobs_all.json")
