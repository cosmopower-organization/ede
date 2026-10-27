"""Run train_v3.py jobs on the two GPUs: slot-based packing, waits for downloaded data, skips finished jobs."""
import json, os, subprocess, sys, time

V3 = os.path.expanduser("~/rds/ede-v3")
DATA = f"{V3}/data"
PY = os.path.expanduser("~/pk_ede/jaxenv/bin/python")
SLOTS_PER_GPU = 6
jobs = json.load(open(sys.argv[1]))
specs = json.load(open(f"{V3}/v2_specs.json"))

def downloaded(fn):
    if fn.startswith(f"{V3}/data_cs/"):
        obs = os.path.basename(fn).split("_")[0]
        return os.path.exists(f"{V3}/data_cs/built.log") and f"ok {obs}" in open(f"{V3}/data_cs/built.log").read().split("\n")
    if not fn.startswith(DATA):
        return os.path.exists(fn)
    log = open(f"{V3}/download.log").read().split("\n")
    return f"ok {os.path.basename(fn)}" in log

running = {0: [], 1: []}          # gpu -> list of (proc, slots, name)
pending = [j for j in jobs if not os.path.exists(f"{V3}/runs/{j['name']}_eval.json")]
print(f"{len(pending)} jobs to run", flush=True)
while pending or any(running.values()):
    for g in running:
        still = []
        for proc, s, name in running[g]:
            if proc.poll() is None:
                still.append((proc, s, name))
            else:
                print(f"[{time.strftime('%H:%M')}] done {name} (gpu {g}, exit {proc.returncode})", flush=True)
        running[g] = still
    for j in list(pending):
        if not all(downloaded(f) for f in j["needs"]):
            continue
        g = min(running, key=lambda g: sum(s for _, s, _ in running[g]))
        if sum(s for _, s, _ in running[g]) + j["slots"] > SLOTS_PER_GPU:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   JAX_DEFAULT_MATMUL_PRECISION="highest")
        log = open(f"{V3}/runs/{j['name']}.log", "w")
        proc = subprocess.Popen([PY, f"{V3}/train_v3.py"] + j["args"] + ["--out", f"{V3}/runs/{j['name']}"],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        running[g].append((proc, j["slots"], j["name"]))
        pending.remove(j)
        print(f"[{time.strftime('%H:%M')}] start {j['name']} on gpu {g}", flush=True)
    time.sleep(20)
print("ALL DONE", flush=True)
