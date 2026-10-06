"""BELLS vs Strata head-to-head comparison on the same Modal A10.

Each engine runs its best ~2-bit quantization of Flash-Next 177B:
- BELLS: unsloth Q2_K_XL GGUF (73.5 GiB, 3 shards)
- Strata: ISTA-DASLab GSQ-RCO Q2_0 (66.4 GB, 2 shards) packed via iq_pack.py

Strata needs its own packed model format (index.txt + dense.bin), not raw GGUF.
"""
import modal
import subprocess
import os
import signal
import time
import json
import glob as globmod
import urllib.request
import urllib.error

app = modal.App("bells-vs-strata")

vol = modal.Volume.from_name("bells-benchmark-models", create_if_missing=True)
MODEL_DIR = "/models"

comparison_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("cmake", "build-essential", "git", "wget", "curl", "ninja-build", "unzip")
    # --- Build BELLS ---
    .run_commands("git clone --branch bells-next https://github.com/DGuckert/llama.cpp-BELLS.git /opt/bells")
    .run_commands("cd /opt/bells && git pull origin bells-next && git log --oneline -3")  # bust:strata-debug-v10
    .run_commands(
        "mkdir -p /opt/bells/build && cd /opt/bells/build && "
        "cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='86'"
    )
    .run_commands("cd /opt/bells/build && cmake --build . --target ggml-cuda -j$(nproc)")
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && "
        "cd /opt/bells/build && cmake --build . --target llama-server -j$(nproc)"
    )
    # --- Build Strata ---
    # Strata needs CMake >= 3.24, Ubuntu 22.04 ships 3.22
    .run_commands("pip install cmake --upgrade && cmake --version")
    .run_commands("git clone https://github.com/Niko1221/Strata.git /opt/strata")
    .run_commands("cd /opt/strata && git log --oneline -3")
    # Strata's FetchContent grabs its own pinned llama.cpp/ggml if STRATA_GGML_DIR is unset
    .run_commands(
        "mkdir -p /opt/strata/build && cd /opt/strata/build && "
        "cmake -DCMAKE_BUILD_TYPE=Release "
        "-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF "
        "-DCMAKE_CUDA_ARCHITECTURES='86' "
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc "
        ".. || echo 'WARNING: Strata cmake configure failed'"
    )
    .run_commands(
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && "
        "cd /opt/strata/build && cmake --build . --target strata -j$(nproc) || "
        "echo 'WARNING: Strata build failed, will skip Strata benchmarks'"
    )
    .pip_install("huggingface_hub", "numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil")
)


# the cached image plus a local patch: only the touched files recompile
patched_image = (
    comparison_image
    .add_local_file("mtp_fix.patch", "/opt/mtp_fix.patch", copy=True)
    .run_commands(
        "cd /opt/bells && git apply /opt/mtp_fix.patch && git diff --stat && "
        "cd build && (cmake --build . --target llama-server -j$(nproc) > /tmp/build.log 2>&1 || "
        "(grep -nE 'error' -B3 -A8 /tmp/build.log | head -120; exit 1)) && tail -3 /tmp/build.log"
    )
)


def download_bells_model():
    """Download Flash-Next Q2_K_XL GGUF for BELLS."""
    from huggingface_hub import snapshot_download

    model_pattern = f"{MODEL_DIR}/**/*Q2_K_XL*.gguf"
    existing = sorted(globmod.glob(model_pattern, recursive=True))
    if existing:
        total_gb = sum(os.path.getsize(f) for f in existing) / (1024**3)
        print(f"BELLS model already cached: {len(existing)} shards ({total_gb:.1f} GiB)")
        return existing[0]

    print("Downloading Qwen3.8-Flash-Next UD-Q2_K_XL (~79 GB, 3 shards)...")
    snapshot_download(
        repo_id="unsloth/Qwen3.8-Flash-Next-GGUF",
        allow_patterns=["UD-Q2_K_XL/*"],
        local_dir=MODEL_DIR,
    )
    shards = sorted(globmod.glob(model_pattern, recursive=True))
    if not shards:
        raise RuntimeError("BELLS model download failed")
    return shards[0]


def download_strata_model():
    """Download Flash-Next GSQ-RCO Q2_0 GGUF for Strata.
    Returns (shard1_path, shard2_path) — shard2 is the PLE table."""
    from huggingface_hub import snapshot_download

    strata_dir = f"{MODEL_DIR}/strata-q2_0"
    pattern = f"{strata_dir}/**/*Q2_0*.gguf"
    existing = sorted(globmod.glob(pattern, recursive=True))
    if existing:
        total_gb = sum(os.path.getsize(f) for f in existing) / (1024**3)
        print(f"Strata Q2_0 model already cached: {len(existing)} shards ({total_gb:.1f} GiB)")
    else:
        print("Downloading ISTA-DASLab GSQ-RCO Q2_0 (~66 GB, 2 shards)...")
        snapshot_download(
            repo_id="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
            allow_patterns=["*Q2_0*.gguf"],
            local_dir=strata_dir,
        )
        existing = sorted(globmod.glob(pattern, recursive=True))
    if not existing:
        raise RuntimeError("Strata Q2_0 model download failed")
    shard1 = existing[0]
    shard2 = shard1.replace("00001-of-00002", "00002-of-00002") if len(existing) >= 2 else None
    if shard2 and not os.path.exists(shard2):
        shard2 = existing[1] if len(existing) >= 2 else None
    print(f"  Shard 1 (weights): {shard1}")
    print(f"  Shard 2 (PLE):     {shard2}")
    return shard1, shard2


def pack_strata_model(shard1_path):
    """Run iq_pack.py to create Strata's packed model format."""
    pack_dir = "/tmp/strata_pack"
    index_path = os.path.join(pack_dir, "index.txt")
    if os.path.exists(index_path):
        print(f"Strata pack already exists at {pack_dir}")
        return pack_dir

    print(f"Packing Strata model from {shard1_path}...")
    os.makedirs(pack_dir, exist_ok=True)
    r = subprocess.run(
        ["python3", "/opt/strata/tools/iq_pack.py",
         "--gguf", shard1_path,
         "--out", pack_dir],
        capture_output=True, text=True, cwd="/opt/strata", timeout=3600,
    )
    print(f"iq_pack.py stdout:\n{r.stdout[-2000:]}")
    if r.returncode != 0:
        print(f"iq_pack.py failed (exit {r.returncode}):\n{r.stderr[-2000:]}")
        return None

    if os.path.exists(index_path):
        print(f"Pack created at {pack_dir}")
        ls = subprocess.run(["ls", "-lh", pack_dir], capture_output=True, text=True)
        print(ls.stdout)
        return pack_dir

    print("iq_pack.py produced no index.txt")
    return None


def setup_strata_tokenizer(shard1_path):
    """Extract tokenizer from GGUF for Strata's server.py."""
    tok_dir = "/tmp/strata_tok/tokenizer"
    if os.path.exists(os.path.join(tok_dir, "vocab.json")):
        print("Tokenizer already extracted")
        return tok_dir
    print(f"Extracting tokenizer from {shard1_path}...")
    r = subprocess.run(
        ["python3", "/opt/strata/tools/strata_tokenizer.py",
         "--gguf", shard1_path, "--out", "/tmp/strata_tok"],
        capture_output=True, text=True, cwd="/opt/strata"
    )
    if r.returncode != 0:
        print(f"strata_tokenizer.py failed (exit {r.returncode}):")
        print(r.stderr[-1000:] if r.stderr else "no stderr")
        print(r.stdout[-1000:] if r.stdout else "no stdout")
        return None
    if os.path.exists(os.path.join(tok_dir, "vocab.json")):
        print(f"Tokenizer extracted to {tok_dir}")
        return tok_dir
    # Check if tokenizer ended up elsewhere
    for root, dirs, files in os.walk("/tmp/strata_tok"):
        if "vocab.json" in files:
            print(f"Tokenizer found at {root}")
            return root
    print("Tokenizer extraction produced no vocab.json")
    return None


def build_mtp_runtime(_shard1_path):
    """Build MTP runtime: fetch BF16 tensors -> pack to GGUF -> build runtime.
    Three-step pipeline from setup.py lines 4204-4207."""
    mtp_base = f"{MODEL_DIR}/strata-mtp"
    rt_dir = os.path.join(mtp_base, "rt")
    dense_txt = os.path.join(rt_dir, "dense.txt")
    if os.path.exists(dense_txt):
        print(f"MTP runtime already built at {rt_dir}")
        return rt_dir

    env = os.environ.copy()
    env["STRATA_GGUF_PY"] = "/opt/bells/gguf-py"
    os.makedirs(mtp_base, exist_ok=True)
    os.makedirs(rt_dir, exist_ok=True)

    # Step 1: Fetch raw BF16 MTP tensors from original Qwen checkpoint
    manifest = os.path.join(mtp_base, "mtp-manifest.json")
    if not os.path.exists(manifest):
        print("Step 1/3: Fetching MTP BF16 tensors (~5 GB)...")
        for cmd in ["inventory", "fetch"]:
            r = subprocess.run(
                ["python3", "/opt/strata/tools/mtp_fetch.py", cmd, "--out", mtp_base],
                capture_output=True, text=True, cwd="/opt/strata", timeout=3600, env=env,
            )
            if r.returncode != 0:
                print(f"mtp_fetch.py {cmd} failed:\n{r.stderr[-500:]}")
                return None
        print(f"  Fetched {len(globmod.glob(os.path.join(mtp_base, 'tensors', '*.bin')))} tensors")
    else:
        print("Step 1/3: MTP tensors already cached")

    # Step 2: Pack raw tensors into MTP GGUF
    mtp_gguf = os.path.join(mtp_base, "mtp-q2_0.gguf")
    if not os.path.exists(mtp_gguf):
        print("Step 2/3: Packing MTP tensors into GGUF...")
        r = subprocess.run(
            ["python3", "/opt/strata/tools/mtp_pack.py",
             "--src", mtp_base, "--experts", "q2_0",
             "--out", mtp_gguf],
            capture_output=True, text=True, cwd="/opt/strata", timeout=3600, env=env,
        )
        print(f"mtp_pack.py stdout:\n{r.stdout[-500:]}")
        if r.returncode != 0:
            print(f"mtp_pack.py failed:\n{r.stderr[-500:]}")
            return None
    else:
        print("Step 2/3: MTP GGUF already exists")

    # Step 3: Build runtime from MTP GGUF
    print("Step 3/3: Building MTP runtime...")
    r = subprocess.run(
        ["python3", "/opt/strata/tools/mtp_rt.py",
         "--gguf", mtp_gguf, "--out", rt_dir],
        capture_output=True, text=True, cwd="/opt/strata", timeout=3600, env=env,
    )
    print(f"mtp_rt.py stdout:\n{r.stdout[-500:]}")
    if r.returncode != 0:
        print(f"mtp_rt.py failed:\n{r.stderr[-500:]}")
        return None

    if os.path.exists(dense_txt):
        ls = subprocess.run(["ls", "-lh", rt_dir], capture_output=True, text=True)
        print(f"MTP runtime ready:\n{ls.stdout}")
        return rt_dir
    print("mtp_rt.py produced no dense.txt")
    return None


def find_strata_binary():
    """Locate the compiled strata binary."""
    candidates = [
        "/opt/strata/build/strata",
        "/opt/strata/build/bin/strata",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    r = subprocess.run(
        ["find", "/opt/strata/build", "-name", "strata", "-type", "f", "-executable"],
        capture_output=True, text=True
    )
    if r.stdout.strip():
        return r.stdout.strip().split('\n')[0]
    return None


def api_call(method, path, body=None, timeout=120, port=8080):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def wait_healthy(timeout=600, port=8080, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            print(f"  server process exited early (code {proc.returncode})")
            return False
        try:
            r = api_call("GET", "/health", port=port)
            if r.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def chat(prompt, max_tokens=256, port=8080):
    body = {
        "model": "qwen",
        "messages": [{"role": "user", "content": f"/no_think {prompt}"}],
        "max_tokens": max_tokens,
        "temperature": BENCH_TEMP,
        "stream": False,
    }
    if BENCH_TEMP == 0:
        # the repeat test wants the same decode every time, prompt included
        body["seed"] = 1234
        body["cache_prompt"] = False
    t0 = time.time()
    r = api_call("POST", "/v1/chat/completions", body, timeout=600, port=port)
    elapsed = time.time() - t0
    if "error" in r:
        return {"error": r["error"], "elapsed": elapsed}
    choice = r.get("choices", [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or msg.get("reasoning_content") or ""
    usage = r.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    tps = comp_tokens / elapsed if elapsed > 0 else 0
    return {
        "text": text[:400],
        "decode_tps": tps,
        "elapsed": elapsed,
        "comp_tokens": comp_tokens,
    }


BENCH_PROMPTS = [
    ("Explain TCP vs UDP in 3 sentences.", 100),
    ("Write a Python quicksort implementation.", 200),
    ("What causes OOM in Kubernetes pods?", 150),
    ("Write a Dockerfile for a Node.js app with multi-stage build.", 200),
    ("Explain database connection pooling briefly.", 100),
    ("Write a bash script to find and compress old log files.", 150),
    ("Compare Redis and Memcached for session storage.", 150),
    ("Explain OAuth 2.0 authorization code flow.", 200),
]


PROFILE_PROMPTS = [
    ("Describe how a B-tree index speeds up range queries.", 200),
    ("Write a Go function that retries an HTTP request with exponential backoff.", 200),
    ("What is the difference between a mutex and a semaphore?", 200),
    ("Write a SQL query that finds the top 5 customers by revenue per region.", 200),
    ("Explain how TLS certificate validation works.", 200),
    ("Write a Rust function that parses a CSV line with quoted fields.", 200),
    ("How does garbage collection work in the JVM?", 200),
    ("Write a GitHub Actions workflow that runs tests on every pull request.", 200),
]

ACTIVE_PROMPTS = None
BENCH_TEMP = 0.7
PP_MODE = False


def run_benchmark(engine_name, port=8080, warmup=2, bench_runs=12):
    BENCH_PROMPTS = ACTIVE_PROMPTS or globals()["BENCH_PROMPTS"]
    print(f"\n  Running warmup ({warmup} requests)...")
    for i in range(warmup):
        prompt, max_tok = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
        r = chat(prompt, max_tok, port)
        if "error" in r:
            print(f"    warmup {i+1}: ERROR {r['error']}")
        else:
            print(f"    warmup {i+1}: {r['decode_tps']:.1f} tok/s")

    print(f"  Running benchmark ({bench_runs} requests)...")
    bench_results = []
    for i in range(bench_runs):
        prompt, max_tok = BENCH_PROMPTS[(warmup + i) % len(BENCH_PROMPTS)]
        r = chat(prompt, max_tok, port)
        if "error" not in r:
            bench_results.append(r)
            print(f"    [{i+1}/{bench_runs}] {r['decode_tps']:.1f} tok/s, "
                  f"{r['comp_tokens']} tokens in {r['elapsed']:.1f}s")
            if i < 2 or BENCH_TEMP == 0:
                # the answer itself: a fast run that writes nonsense is not a result
                print(f"      TEXT[{i+1}]: " + r['text'][:300].replace(chr(10), ' / '))
        else:
            print(f"    [{i+1}/{bench_runs}] ERROR: {r['error']}")

    results = {"engine": engine_name}
    if bench_results:
        avg_tps = sum(r["decode_tps"] for r in bench_results) / len(bench_results)
        min_tps = min(r["decode_tps"] for r in bench_results)
        max_tps = max(r["decode_tps"] for r in bench_results)
        total_tokens = sum(r["comp_tokens"] for r in bench_results)
        total_time = sum(r["elapsed"] for r in bench_results)
        results.update({
            "avg_tps": round(avg_tps, 1),
            "min_tps": round(min_tps, 1),
            "max_tps": round(max_tps, 1),
            "total_tokens": total_tokens,
            "total_time": round(total_time, 1),
            "effective_tps": round(total_tokens / total_time, 1),
        })
        print(f"\n  RESULT [{engine_name}]: {avg_tps:.1f} tok/s avg "
              f"(min {min_tps:.1f}, max {max_tps:.1f})")
    else:
        results["error"] = "no successful runs"
    return results


def stop_server(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_bells_config(model_path, config_name, server_args):
    """Run a single BELLS configuration."""
    print(f"\n{'='*60}")
    print(f"ENGINE: {config_name}")
    print(f"{'='*60}")

    cmd = ["stdbuf", "-oL", "/opt/bells/build/bin/llama-server",
           "-m", model_path, "--host", "0.0.0.0", "--port", "8080"] + server_args
    print(f"CMD: {' '.join(cmd)}")

    log = open(f"/tmp/bells_{config_name}_stderr.txt", "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log)
    try:
        print("  Loading model...")
        if wait_healthy(timeout=600, proc=proc):
            result = run_benchmark(config_name, port=8080)
        else:
            log.close()
            with open(f"/tmp/bells_{config_name}_stderr.txt") as f:
                print(f"  FAILED. Last output:\n{f.read()[-2000:]}")
            result = {"engine": config_name, "error": "failed to start"}
    finally:
        stop_server(proc)
        log.close()

    trace = "-lv" in server_args
    keys = (['eval time', 'total time', 'graphs reused', 'statistics', 'assert', 'diag', 'free: hit', 'slots/layer',
             'bells_pass']
            if trace else
            ['bells', 'slots', 'auto', 'hit ', 'spec', 'mtp', 'draft', 'accept', 'assert', 'abort', 'error',
             'nextn', 'diag', 'invalid', 'cuda_profile', 'cuda_node', 'eval time', 'total time', 'graphs reused', 'statistics'])
    with open(f"/tmp/bells_{config_name}_stderr.txt", errors="replace") as f:
        for line in f:
            low = line.lower()
            if any(kw in low for kw in keys) and 'bells_timing' not in low:
                print(f"    {line.strip()[:400]}")
    if not result.get("avg_tps"):
        with open(f"/tmp/bells_{config_name}_stderr.txt") as f:
            tail = f.readlines()[-80:]
        print(f"  --- {config_name}: no result, last {len(tail)} stderr lines ---")
        for line in tail:
            print(f"    {line.rstrip()[:300]}")
    return result


def download_mtp_sidecars():
    """unsloth's MTP heads for llama.cpp's --spec-type draft-mtp. Returns {name: path}."""
    from huggingface_hub import snapshot_download, list_repo_files
    files = [f for f in list_repo_files("unsloth/Qwen3.8-Flash-Next-GGUF") if f.startswith("MTP/")]
    print("MTP files in the repo:")
    for f in files:
        print(f"  {f}")
    want = [f for f in files if f.endswith("Q4_K_M.gguf")]
    snapshot_download(repo_id="unsloth/Qwen3.8-Flash-Next-GGUF", allow_patterns=want, local_dir=MODEL_DIR)
    return {os.path.basename(f): os.path.join(MODEL_DIR, f) for f in want}


@app.function(image=patched_image, gpu="A10", memory=131072, volumes={MODEL_DIR: vol}, timeout=7200)
def repro_mtp():
    """Reproduce the BELLS + MTP failure and isolate it: stock path vs BELLS, each MTP head."""
    model = download_bells_model()
    heads = download_mtp_sidecars()
    vol.commit()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    results = []
    path = next(p for n, p in heads.items() if "shared" not in n)
    mtp = ["--spec-type", "draft-mtp", "--spec-draft-model", path, "--spec-draft-ngl", "99"]
    q4  = ["-c", "4096", "-ctk", "q4_0", "-ctv", "q4_0"]
    f16 = ["-c", "4096"]
    for cfg, args in [
        ("fix-nomtp",       ["--auto"] + q4),
        ("fix-mtp-n2-np1",  ["--auto", "--bells-reserve", "2560", "-np", "1"] + q4 + mtp + ["--spec-draft-n-max", "2"]),
        ("fix-mtp-n3-np1",  ["--auto", "--bells-reserve", "2560", "-np", "1"] + q4 + mtp + ["--spec-draft-n-max", "3"]),
        ("fix-mtp-n2-np4",  ["--auto", "--bells-reserve", "2560"] + q4 + mtp + ["--spec-draft-n-max", "2"]),
        ("fix-nomtp-np1",   ["--auto", "-np", "1"] + q4),
    ]:
        results.append(run_bells_config(model, cfg, args))
        time.sleep(5)
    print("\nSUMMARY")
    for r in results:
        print(f"  {r['engine']:<28} {r.get('avg_tps', r.get('error'))}")
    return json.dumps(results, indent=2)


@app.function(image=patched_image, gpu="A10", memory=131072, volumes={MODEL_DIR: vol}, timeout=7200)
def repro_q2():
    """BELLS on the GSQ-RCO Q2_0 GGUF Strata uses: smaller experts, so more cache slots per GiB."""
    xl = download_bells_model()
    q2, _ = download_strata_model()
    heads = download_mtp_sidecars()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    path = next(p for n, p in heads.items() if "shared" not in n)
    mtp = ["--spec-type", "draft-mtp", "--spec-draft-model", path, "--spec-draft-ngl", "99", "--spec-draft-n-max", "2"]
    q4 = ["-c", "4096", "-ctk", "q4_0", "-ctv", "q4_0", "-np", "1"]
    results = []
    shared = next((p for n, p in heads.items() if "shared" in n), None)
    mtp_sh = ["--spec-type", "draft-mtp", "--spec-draft-model", shared, "--spec-draft-ngl", "99", "--spec-draft-n-max", "2"]
    profile_path = "/tmp/bells_layer_profile.txt"
    global ACTIVE_PROMPTS

    def run(cfg, args, env=None, prompts=None):
        global ACTIVE_PROMPTS
        for k in ("BELLS_LAYER_STATS", "BELLS_LAYER_PROFILE", "GGML_CUDA_PROFILE", "GGML_CUDA_DISABLE_GRAPHS",
                  "GGML_CUDA_MMVF_MAX_COLS", "GGML_CUDA_MMVF_MAX_ELEMS", "BELLS_PASS_TIMING",
                  "BELLS_SPLIT_AT_TOPK", "GGML_CUDA_DISABLE_UNARY_CHAIN"):
            os.environ.pop(k, None)
        os.environ.update(env or {})
        ACTIVE_PROMPTS = prompts or BENCH_PROMPTS
        results.append(run_bells_config(q2, cfg, args))
        time.sleep(5)

    prof_env  = {"BELLS_LAYER_PROFILE": profile_path}
    cuda_prof = {"GGML_CUDA_PROFILE": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}

    # the routing profile comes from prompts the benchmark never uses
    def mtp_n(n):
        return ["--auto", "--bells-reserve", "2560"] + q4 + mtp[:-1] + [str(n)]

    def cols(n):
        return {"GGML_CUDA_MMVF_MAX_COLS": str(n)}

    fast = {"GGML_CUDA_MMVF_MAX_COLS": "8", "GGML_CUDA_MMVF_MAX_ELEMS": "1048576"}
    prob = ["--spec-draft-sampling", "probabilistic"]

    timing = {**fast, "BELLS_PASS_TIMING": "1"}
    best   = mtp_n(3) + prob
    sh_n3  = (["--auto", "--bells-reserve", "2048"] + q4 + mtp_sh[:-1] + ["3"] + prob)

    # pass time against cache size: what one missed expert costs
    # one identical greedy request repeated: the routing repeats, so the cache hits ~100% and the pass time left
    # is kernel time alone
    nochain = {"GGML_CUDA_DISABLE_UNARY_CHAIN": "1"}
    global BENCH_TEMP
    same = [("Write a Python quicksort implementation.", 200), ("Explain TCP vs UDP in 3 sentences.", 100)]
    # greedy, same prompts: the fused chains must write the same answers
    BENCH_TEMP = 0.0
    run("greedy-nochain", ["--auto"] + q4, nochain, same)
    run("greedy-chain",   ["--auto"] + q4, None, same)
    BENCH_TEMP = 0.7
    run("nomtp-nochain",  ["--auto"] + q4, nochain)
    run("nomtp-chain",    ["--auto"] + q4)
    run("best-nochain",   best, nochain)
    run("best-chain",     best)
    run("cudaprof-nomtp-chain", ["--auto"] + q4, cuda_prof)
    print("\nSUMMARY")
    for r in results:
        print(f"  {r['engine']:<28} {r.get('avg_tps', r.get('error'))}")
    return json.dumps(results, indent=2)


@app.function(image=patched_image, gpu="A10", memory=131072, volumes={MODEL_DIR: vol}, timeout=10800)
def final_h2h():
    """BELLS's best configuration and Strata back to back on one box, same Q2_0 weights. BELLS runs before
    and after Strata so a drift of the box during the job shows up."""
    q2, ple = download_strata_model()
    heads = download_mtp_sidecars()
    vol.commit()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    print(f"System RAM: {os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/(1024**3):.1f} GiB")
    subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True)

    pack_dir = pack_strata_model(q2)
    tok_dir  = setup_strata_tokenizer(q2)
    mtp_dir  = build_mtp_runtime(q2)

    path = next(p for n, p in heads.items() if "shared" not in n)
    q4   = ["-c", "4096", "-ctk", "q4_0", "-ctv", "q4_0", "-np", "1"]
    best = (["--auto", "--bells-reserve", "2560"] + q4 +
            ["--spec-type", "draft-mtp", "--spec-draft-model", path, "--spec-draft-ngl", "99",
             "--spec-draft-n-max", "3", "--spec-draft-sampling", "probabilistic"])
    fast = {"GGML_CUDA_MMVF_MAX_COLS": "8", "GGML_CUDA_MMVF_MAX_ELEMS": "1048576"}

    results = []

    q8 = requant_dense(q2, f"{MODEL_DIR}/q2_0-dense-q8")
    shared = next(p for n, p in heads.items() if "shared" in n)
    best_sh = (["--auto", "--bells-reserve", "1536"] + q4 +
               ["--spec-type", "draft-mtp", "--spec-draft-model", shared, "--spec-draft-ngl", "99",
                "--spec-draft-n-max", "3", "--spec-draft-sampling", "probabilistic"])
    profile_path = "/tmp/bells_layer_profile.txt"
    global ACTIVE_PROMPTS

    def bells(cfg, model, args, env=None, prompts=None):
        global ACTIVE_PROMPTS
        for k in ("BELLS_LAYER_STATS", "BELLS_LAYER_PROFILE"):
            os.environ.pop(k, None)
        os.environ.update(env or {})
        ACTIVE_PROMPTS = prompts or BENCH_PROMPTS
        results.append(run_bells_config(model, cfg, args))
        ACTIVE_PROMPTS = None
        time.sleep(5)

    def strata(tag, prompts=None):
        global ACTIVE_PROMPTS
        for k in ("BELLS_LAYER_STATS", "BELLS_LAYER_PROFILE"):
            os.environ.pop(k, None)
        ACTIVE_PROMPTS = prompts or BENCH_PROMPTS
        r = run_strata(pack_dir, tok_dir, ple_gguf=ple, native_gguf=q2, mtp_dir=mtp_dir)
        ACTIVE_PROMPTS = None
        r["engine"] = tag
        results.append(r)
        time.sleep(5)

    prof = {"BELLS_LAYER_PROFILE": profile_path}
    # the routing profile comes from prompts the benchmark never uses
    long_prompts = [(p, 600) for p in (
        "Write a detailed tutorial on implementing a thread pool in C++ with code.",
        "Explain how a relational database executes a SQL join, covering the main algorithms.",
        "Write a complete Flask REST API for a todo list with SQLite, including error handling.",
        "Describe the full lifecycle of an HTTPS request from the browser to the server and back.",
    )]
    bells("make-profile",       q8, ["--auto"] + q4, {"BELLS_LAYER_STATS": profile_path}, PROFILE_PROMPTS)
    bells("BELLS-tuned-1",      q8, best_sh, prof)
    strata("Strata-1")
    bells("BELLS-tuned-2",      q8, best_sh, prof)
    strata("Strata-2")
    bells("BELLS-origfile",     q2, best_sh, prof)
    # long answers: decode dominates, prompt and server overhead do not
    bells("BELLS-tuned-long-1", q8, best_sh, prof, long_prompts)
    strata("Strata-long-1", long_prompts)
    bells("BELLS-tuned-long-2", q8, best_sh, prof, long_prompts)
    strata("Strata-long-2", long_prompts)

    print("\nFINAL HEAD-TO-HEAD (one box, Q2_0 weights, 12 requests each)")
    for r in results:
        if r.get("avg_tps"):
            print(f"  {r['engine']:<20} {r['avg_tps']:>6} tok/s  (min {r['min_tps']}, max {r['max_tps']}, "
                  f"effective {r['effective_tps']})")
        else:
            print(f"  {r['engine']:<16} FAILED: {r.get('error')}")
    return json.dumps(results, indent=2)


def requant_dense(src_shard1, dst_dir, qtype_name="Q8_0", min_elems=1_000_000,
                  pattern=r"(hc_attn|hc_ffn)_(down|up|inject)\.weight$|ffn_gate_inp\.weight$"):
    """Copy shard 1 with its large dense BF16 matrices stored as qtype_name; every other tensor and every
    metadata key is copied unchanged. Shard 2 (the PLE table) is linked beside it. Returns the new shard 1."""
    import re
    import sys
    sys.path.insert(0, "/opt/bells/gguf-py")
    import numpy as np
    import gguf
    from gguf import quants

    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src_shard1))
    done = dst + ".done"
    shard2 = src_shard1.replace("00001-of-00002", "00002-of-00002")
    link2 = os.path.join(dst_dir, os.path.basename(shard2))
    if os.path.exists(done):
        print(f"requant: {dst} already built")
        return dst

    qtype = gguf.GGMLQuantizationType[qtype_name]
    reader = gguf.GGUFReader(src_shard1)
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    rx = re.compile(pattern)
    by_type = {}
    plan = []
    n_conv = 0
    b_before = b_after = 0
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    for t in reader.tensors:
        n = int(np.prod(t.shape))
        key = t.tensor_type.name
        by_type[key] = by_type.get(key, 0) + t.n_bytes
        cols = int(t.shape[0])
        if (t.tensor_type == gguf.GGMLQuantizationType.BF16 and len(t.shape) == 2 and n >= min_elems
                and cols % block == 0 and rx.search(t.name)):
            f = quants.dequantize(t.data, gguf.GGMLQuantizationType.BF16)
            q = quants.quantize(f, qtype)
            if n_conv < 12:
                err = float(np.abs(quants.dequantize(q, qtype) - f).max())
                print(f"requant: {t.name} {list(t.shape)} BF16 -> {qtype_name}, max abs err {err:.2e}, max abs {float(np.abs(f).max()):.2e}")
            writer.add_tensor_info(t.name, q.shape, q.dtype, q.nbytes, qtype)
            plan.append(q)
            n_conv += 1
            b_before += t.n_bytes
            b_after += q.nbytes
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            plan.append(t.data)
    print("requant: bytes by type in shard 1: " + ", ".join(f"{k} {v/2**30:.2f} GiB" for k, v in sorted(by_type.items())))
    print(f"requant: {n_conv} tensors, {b_before/2**30:.2f} GiB -> {b_after/2**30:.2f} GiB")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for d in plan:
        writer.write_tensor_data(d, tensor_endianess=reader.endianess)
    writer.close()
    if os.path.exists(shard2) and not os.path.exists(link2):
        os.symlink(shard2, link2)
    open(done, "w").close()
    print(f"requant: wrote {dst} ({os.path.getsize(dst)/2**30:.1f} GiB)")
    return dst


@app.function(image=patched_image, gpu="A10", memory=131072, volumes={MODEL_DIR: vol}, timeout=10800)
def dense_q8():
    """Does storing the dense BF16 mixer/router matrices as Q8_0 speed up a pass, and does it change answers?"""
    global ACTIVE_PROMPTS, BENCH_TEMP
    q2, _ = download_strata_model()
    heads = download_mtp_sidecars()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    t0 = time.time()
    q8 = requant_dense(q2, f"{MODEL_DIR}/q2_0-dense-q8")
    print(f"requant: {time.time()-t0:.0f} s")
    vol.commit()

    path = next(p for n, p in heads.items() if "shared" not in n)
    q4 = ["-c", "4096", "-ctk", "q4_0", "-ctv", "q4_0", "-np", "1"]
    best = (["--auto", "--bells-reserve", "2560"] + q4 +
            ["--spec-type", "draft-mtp", "--spec-draft-model", path, "--spec-draft-ngl", "99",
             "--spec-draft-n-max", "3", "--spec-draft-sampling", "probabilistic"])
    results = []

    def run(cfg, model, args, env=None, prompts=None, temp=0.7):
        global ACTIVE_PROMPTS, BENCH_TEMP
        for k in ("GGML_CUDA_PROFILE", "GGML_CUDA_DISABLE_GRAPHS", "BELLS_PASS_TIMING"):
            os.environ.pop(k, None)
        os.environ.update(env or {})
        ACTIVE_PROMPTS = prompts or BENCH_PROMPTS
        BENCH_TEMP = temp
        results.append(run_bells_config(model, cfg, args))
        time.sleep(5)

    same = [("Write a Python quicksort implementation.", 200), ("Explain TCP vs UDP in 3 sentences.", 100),
            ("What causes OOM in Kubernetes pods?", 150), ("Explain OAuth 2.0 authorization code flow.", 200)]
    timing = {"BELLS_PASS_TIMING": "1"}
    prof = {"GGML_CUDA_PROFILE": "1", "GGML_CUDA_DISABLE_GRAPHS": "1"}
    run("greedy-bf16",  q2, ["--auto"] + q4, None, same, 0.0)
    run("greedy-q8",    q8, ["--auto"] + q4, None, same, 0.0)
    run("nomtp-bf16",   q2, ["--auto"] + q4, timing)
    run("nomtp-q8",     q8, ["--auto"] + q4, timing)
    run("best-bf16",    q2, best, timing)
    run("best-q8",      q8, best, timing)
    run("cudaprof-q8",  q8, ["--auto"] + q4, prof)
    print("\nSUMMARY")
    for r in results:
        print(f"  {r['engine']:<28} {r.get('avg_tps', r.get('error'))}")
    return json.dumps(results, indent=2)


@app.function(image=patched_image, gpu="A10", memory=131072, volumes={MODEL_DIR: vol}, timeout=10800)
def tune_q8():
    """On the dense-Q8 file with the shared head: how small can the reserve go, and what is left in the pass."""
    global ACTIVE_PROMPTS
    q2, _ = download_strata_model()
    heads = download_mtp_sidecars()
    q8 = requant_dense(q2, f"{MODEL_DIR}/q2_0-dense-q8")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    shared = next(p for n, p in heads.items() if "shared" in n)
    q4 = ["-c", "4096", "-ctk", "q4_0", "-ctv", "q4_0", "-np", "1"]
    profile_path = "/tmp/bells_layer_profile.txt"
    results = []

    def cfg(reserve, n=3, extra=None):
        return (["--auto", "--bells-reserve", str(reserve)] + q4 +
                ["--spec-type", "draft-mtp", "--spec-draft-model", shared, "--spec-draft-ngl", "99",
                 "--spec-draft-n-max", str(n), "--spec-draft-sampling", "probabilistic"] + (extra or []))

    def run(name, args, env=None, prompts=None):
        global ACTIVE_PROMPTS
        for k in ("BELLS_LAYER_STATS", "BELLS_LAYER_PROFILE", "GGML_CUDA_PROFILE", "GGML_CUDA_DISABLE_GRAPHS",
                  "GGML_CUDA_PROFILE_DUMP", "BELLS_PASS_TIMING", "BELLS_SKIP_MISS_RANK"):
            os.environ.pop(k, None)
        os.environ.update(env or {})
        ACTIVE_PROMPTS = prompts or BENCH_PROMPTS
        results.append(run_bells_config(q8, name, args))
        ACTIVE_PROMPTS = None
        time.sleep(5)

    pf = {"BELLS_LAYER_PROFILE": profile_path}
    global BENCH_TEMP
    same = [("Write a Python quicksort implementation.", 200), ("Explain TCP vs UDP in 3 sentences.", 100),
            ("What causes OOM in Kubernetes pods?", 150), ("Explain OAuth 2.0 authorization code flow.", 200)]
    tm = {"BELLS_PASS_TIMING": "1"}

    def skip(k):
        return {"BELLS_SKIP_MISS_RANK": str(k)}

    run("make-profile", ["--auto"] + q4, {"BELLS_LAYER_STATS": profile_path}, PROFILE_PROMPTS)
    # greedy answers first: leaving experts out is lossy, the text shows how much
    BENCH_TEMP = 0.0
    run("greedy-full",  ["--auto"] + q4, pf, same)
    run("greedy-skip8", ["--auto"] + q4, {**pf, **skip(8)}, same)
    run("greedy-skip6", ["--auto"] + q4, {**pf, **skip(6)}, same)
    run("greedy-skip4", ["--auto"] + q4, {**pf, **skip(4)}, same)
    BENCH_TEMP = 0.7
    run("mtp-full",     cfg(1536), {**pf, **tm})
    run("mtp-skip8",    cfg(1536), {**pf, **tm, **skip(8)})
    run("mtp-skip6",    cfg(1536), {**pf, **tm, **skip(6)})
    run("mtp-skip4",    cfg(1536), {**pf, **tm, **skip(4)})
    run("mtp-full-b",   cfg(1536), {**pf, **tm})
    print("\nSUMMARY")
    for r in results:
        print(f"  {r['engine']:<28} {r.get('avg_tps', r.get('error'))}")
    return json.dumps(results, indent=2)


def pp_requests():
    """Real text of two lengths, each request starting differently so no prompt cache can reuse a prefix."""
    import glob as g
    files = sorted(g.glob("/opt/bells/docs/*.md") + g.glob("/opt/bells/docs/development/*.md") +
                   ["/opt/bells/README.md", "/opt/bells/tools/server/README.md"])
    text = ""
    for f in files:
        try:
            text += open(f, errors="replace").read() + "\n\n"
        except OSError:
            pass
    reqs = []
    off = 0
    for label, n_chars in (("1K", 4000), ("1K", 4000), ("1K", 4000), ("2.5K", 10000), ("2.5K", 10000), ("2.5K", 10000)):
        chunk = text[off:off + n_chars]
        off += n_chars + 1500
        reqs.append((label, f"Request {len(reqs) + 1}. Summarize the following documentation in one sentence.\n\n{chunk}"))
    return reqs


def pp_measure(engine_name, port=8080):
    """Time to a 16-token answer over long prompts: prompt tokens per second, prompt processing dominating."""
    out = []
    # one short request first, so model load and warmup are not billed to the first prompt
    chat("Say hello.", 16, port)
    reqs = pp_requests()
    # one extra request per size with a 40-token answer, only to read the text; the timed ones stop at 1 token
    reqs = [(l, pr, 1) for l, pr in reqs] + [("check", reqs[1][1] + " (again)", 40), ("check", reqs[4][1] + " (again)", 40)]
    for label, prompt, n_gen in reqs:
        body = {"model": "qwen", "messages": [{"role": "user", "content": f"/no_think {prompt}"}],
                "max_tokens": n_gen, "temperature": 0.0, "stream": False}
        t0 = time.time()
        r = api_call("POST", "/v1/chat/completions", body, timeout=900, port=port)
        dt = time.time() - t0
        if "error" in r:
            print(f"    pp [{engine_name}] {label}: ERROR {r['error']}")
            continue
        usage = r.get("usage", {})
        n_p = usage.get("prompt_tokens", 0)
        n_c = usage.get("completion_tokens", 0)
        print(f"    pp [{engine_name}] {label}: {n_p} prompt tokens + {n_c} generated in {dt:.2f} s "
              f"-> {n_p/dt if dt > 0 else 0:.0f} prompt tok/s")
        tm = r.get("timings") or {}
        if tm:
            print(f"      server timings: prompt {tm.get('prompt_n')} tokens {tm.get('prompt_ms', 0):.0f} ms, "
                  f"predicted {tm.get('predicted_n')} tokens {tm.get('predicted_ms', 0):.0f} ms, cached {tm.get('cache_n')}")
        msg = (r.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or msg.get("reasoning_content") or ""
        # the answer itself: a summary of the right document shows the prompt was read correctly
        if label == "check":
            print(f"      ANSWER ({n_p} prompt tokens): " + text[:170].replace(chr(10), " / "))
            continue
        out.append((label, n_p, dt))
    for label in ("1K", "2.5K"):
        rows = [(n, d) for l, n, d in out if l == label]
        if rows:
            print(f"  PP RESULT [{engine_name}] {label}: {sum(n for n, _ in rows)/sum(d for _, d in rows):.0f} prompt tok/s "
                  f"({len(rows)} requests, ~{rows[0][0]} tokens each)")
    return out


@app.function(image=patched_image, gpu="A10", memory=131072, volumes={MODEL_DIR: vol}, timeout=10800)
def pp_bench():
    """Prompt processing, BELLS and Strata on one box."""
    q2, ple = download_strata_model()
    heads = download_mtp_sidecars()
    q8 = requant_dense(q2, f"{MODEL_DIR}/q2_0-dense-q8")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    pack_dir = pack_strata_model(q2)
    tok_dir  = setup_strata_tokenizer(q2)
    mtp_dir  = build_mtp_runtime(q2)
    shared = next(p for n, p in heads.items() if "shared" in n)
    q4 = ["-c", "4096", "-ctk", "q4_0", "-ctv", "q4_0", "-np", "1"]
    mtp = ["--spec-type", "draft-mtp", "--spec-draft-model", shared, "--spec-draft-ngl", "99",
           "--spec-draft-n-max", "3", "--spec-draft-sampling", "probabilistic"]

    def bells(name, args, env=None, decode=True):
        for k in ("GGML_CUDA_PROFILE", "GGML_CUDA_DISABLE_GRAPHS", "BELLS_PREFILL_STAGE", "BELLS_PREFILL_MIN", "BELLS_PREFILL_SETS", "BELLS_PREFILL_D2D", "BELLS_PASS_TIMING", "BELLS_LAYER_PROFILE", "BELLS_LAYER_STATS"):
            os.environ.pop(k, None)
        os.environ.update(env or {})
        cmd = ["stdbuf", "-oL", "/opt/bells/build/bin/llama-server", "-m", q8,
               "--host", "0.0.0.0", "--port", "8080"] + args
        print(f"\nENGINE: {name}\nCMD: {' '.join(cmd)[-260:]}")
        log = open(f"/tmp/pp_{name}.txt", "w")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log)
        try:
            if wait_healthy(timeout=600, proc=proc):
                if decode:
                    r = run_benchmark(name, port=8080)
                    print(f"  DECODE RESULT [{name}]: {r.get('avg_tps', r.get('error'))} tok/s")
                pp_measure(name)
            else:
                print("  failed to start")
        finally:
            stop_server(proc)
            log.close()
        with open(f"/tmp/pp_{name}.txt", errors="replace") as f:
            for line in f:
                if ("prompt eval time" in line or "slots/layer" in line or "n_ubatch" in line or "free: hit" in line
                        or "prompt staging" in line or "out of memory" in line or "CUDA error" in line
                        or "staged" in line or "staging time" in line or "cuda_profile" in line
                        or "bells_pass: big" in line or "checkpoint" in line
                        or "GGML_ASSERT" in line):
                    print("    " + line.strip()[-170:])
        time.sleep(5)

    st = {"BELLS_PREFILL_STAGE": "1"}
    nomtp = ["--auto"] + q4
    tuned = ["--auto"] + q4 + mtp
    big = ["-ub", "4096", "-b", "4096"]
    pt = {"BELLS_PASS_TIMING": "1"}
    bells("warm-chunks",          nomtp, None, decode=False)
    bells("time-host-ub4096",     nomtp + big, {**st, **pt, "BELLS_PREFILL_D2D": "0"}, decode=False)
    bells("time-gather-ub4096",   nomtp + big, {**st, **pt}, decode=False)
    return "done"
    global PP_MODE
    PP_MODE = True
    r = run_strata(pack_dir, tok_dir, ple_gguf=ple, native_gguf=q2, mtp_dir=mtp_dir)
    PP_MODE = False
    return json.dumps(r)


@app.function(image=patched_image, gpu="A10", memory=131072, volumes={MODEL_DIR: vol}, timeout=7200)
def strata_inspect():
    """Strata with its own decode timing on, to read its window time, tokens per window and cache size."""
    q2, ple = download_strata_model()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=True)
    pack_dir = pack_strata_model(q2)
    tok_dir  = setup_strata_tokenizer(q2)
    mtp_dir  = build_mtp_runtime(q2)
    results = [run_strata(pack_dir, tok_dir, ple_gguf=ple, native_gguf=q2, mtp_dir=mtp_dir)]
    time.sleep(5)
    os.environ["STRATA_TIMING_RUN"] = "1"
    r = run_strata(pack_dir, tok_dir, ple_gguf=ple, native_gguf=q2, mtp_dir=mtp_dir)
    r["engine"] = "Strata-timing"
    results.append(r)
    subprocess.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv", shell=True)
    for r in results:
        print(f"  {r['engine']:<16} {r.get('avg_tps', r.get('error'))}")
    return json.dumps(results, indent=2)


def run_bells(model_path, _mtp_path=None):
    """Run BELLS benchmark. Returns list of results."""
    results = []

    result_auto = run_bells_config(model_path, "BELLS-auto", [
        "--auto", "-c", "4096", "-ctk", "q4_0", "-ctv", "q4_0",
    ])
    results.append(result_auto)

    return results


def run_strata(pack_dir, tok_dir, ple_gguf=None, native_gguf=None, mtp_dir=None):
    """Run Strata benchmark using packed Q2_0 model."""
    print("\n" + "="*60)
    print("ENGINE: Strata")
    print("="*60)

    strata_exe = find_strata_binary()
    if not strata_exe:
        print("  ERROR: Strata binary not found")
        return {"engine": "Strata", "error": "binary not found"}
    print(f"  Binary: {strata_exe}")

    if not pack_dir:
        print("  ERROR: No packed model available")
        return {"engine": "Strata", "error": "pack not available"}
    print(f"  Pack: {pack_dir}")

    strata_args = [
        "--pack", pack_dir,
        "--max-context", "4096",
        "--kv", "q4_0",
        "--spec", "4",
        "--spec-min-p", "0.5",
        "--prefill", "auto",
        "--expert-profile", "/opt/strata/data/expert-profile.bin",
        "--expert-cache", "auto",
    ]
    # setup.py's refresh_draft_vocab: the draft subset is a file in the MTP runtime, not an engine flag
    if mtp_dir and not os.path.exists(os.path.join(mtp_dir, "draft_vocab.bin")):
        import shutil
        shutil.copyfile("/opt/strata/data/draft_vocab.bin", os.path.join(mtp_dir, "draft_vocab.bin"))
        print("  Copied default draft_vocab.bin into the MTP runtime")
    if native_gguf and os.path.exists(native_gguf):
        strata_args.extend(["--native", native_gguf])
        print(f"  Native GGUF: {native_gguf}")
    if mtp_dir and os.path.exists(mtp_dir):
        strata_args.extend(["--mtp", mtp_dir])
        print(f"  MTP dir: {mtp_dir}")
    if ple_gguf and os.path.exists(ple_gguf):
        strata_args.extend(["--ple-gguf", ple_gguf])
        print(f"  PLE GGUF: {ple_gguf}")
    else:
        strata_args.append("--no-ple")
        print("  WARNING: No PLE GGUF, running with --no-ple (diagnostic ablation)")

    cfg = {
        "exe": strata_exe,
        "args": strata_args,
        "cwd": "/opt/strata",
        "log": "/tmp/strata_engine.log",
    }
    config_path = "/tmp/strata-bench.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    if not tok_dir:
        print("  ERROR: Tokenizer not available")
        return {"engine": "Strata", "error": "tokenizer not available"}

    # Debug: check Strata version and test args directly
    r_ver = subprocess.run(
        ["git", "log", "--oneline", "-5"],
        capture_output=True, text=True, cwd="/opt/strata",
    )
    print(f"  Strata git log:\n{r_ver.stdout}")
    r_test = subprocess.run(
        [strata_exe, "--serve"] + strata_args + ["--dry-run"],
        capture_output=True, text=True, cwd="/opt/strata", timeout=30,
    )
    if r_test.returncode != 0:
        # --dry-run might not exist, try just running it briefly
        r_test2 = subprocess.run(
            [strata_exe, "--serve", "--help"],
            capture_output=True, text=True, cwd="/opt/strata", timeout=30,
        )
        print(f"  strata --serve --help (first 60 lines):")
        for line in r_test2.stdout.split('\n')[:60]:
            print(f"    {line}")
        if r_test2.stderr:
            for line in r_test2.stderr.split('\n')[:20]:
                print(f"    [err] {line}")

    cmd = [
        "python3", "-m", "serve.server",
        "--engine", "strata",
        "--config", config_path,
        "--tokenizer", tok_dir,
        "--host", "0.0.0.0",
        "--port", "8080",
    ]
    print(f"CMD: {' '.join(cmd)}")

    log = open("/tmp/strata_stderr.txt", "w")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/strata/tools:/opt/strata"
    if os.environ.get("STRATA_TIMING_RUN"):
        env["STRATA_DECODE_TIMING"] = "1"
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        cwd="/opt/strata", env=env,
    )
    try:
        print("  Loading model...")
        if wait_healthy(timeout=900, proc=proc):
            if PP_MODE:
                pp_measure("Strata", port=8080)
                result = {"engine": "Strata-pp"}
            else:
                result = run_benchmark("Strata-Q2_0", port=8080)
        else:
            log.close()
            with open("/tmp/strata_stderr.txt") as f:
                tail = f.read()[-3000:]
            print(f"  FAILED. Last output:\n{tail}")
            if os.path.exists("/tmp/strata_engine.log"):
                with open("/tmp/strata_engine.log") as f:
                    elines = f.readlines()
                print(f"  Engine log ({len(elines)} lines total):")
                print("  --- FIRST 80 LINES ---")
                for line in elines[:80]:
                    print(f"    {line.rstrip()}")
                if len(elines) > 80:
                    print(f"  --- ... {len(elines)-80} more lines ... ---")
            result = {"engine": "Strata", "error": "failed to start"}
    finally:
        stop_server(proc)
        log.close()

    with open("/tmp/strata_stderr.txt") as f:
        print("  Strata log (last 30 lines):")
        lines = f.readlines()
        for line in lines[-30:]:
            print(f"    {line.rstrip()}")

    # how Strata sizes its tiers and where its decode windows go (STRATA_DECODE_TIMING)
    for path in ("/tmp/strata_engine.log", "/tmp/strata_stderr.txt"):
        if not os.path.exists(path):
            continue
        shown = 0
        with open(path, errors="replace") as f:
            for line in f:
                low = line.lower()
                if any(k in low for k in ("decode timing", "gpu stages", "expert cache:", "expert tiers", "vram",
                                          " gib", "slots", "draft head", "kv cache", "pool", "reserve")):
                    if "--" in line[:12] or "done:" in low:
                        continue
                    print(f"    [strata-log] {line.rstrip()[:420]}")
                    shown += 1
                    if shown >= 90:
                        break

    return result


@app.function(
    image=comparison_image,
    gpu="A10",
    memory=131072,
    volumes={MODEL_DIR: vol},
    timeout=10800,
)
def compare_engines():
    """Head-to-head: BELLS (Q2_K_XL) vs Strata (Q2_0) on the same A10."""
    # Download both models (each engine's best ~2-bit quant)
    bells_model = download_bells_model()
    strata_shard1, strata_ple = download_strata_model()
    vol.commit()

    print("\n" + "="*70)
    print("BELLS vs STRATA: Head-to-Head on Same A10")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                     "--format=csv"], check=True)
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    print(f"System RAM: {ram:.1f} GiB")
    print(f"BELLS model: {bells_model}")
    print(f"Strata model: {strata_shard1}")
    print(f"Strata PLE:   {strata_ple}")
    print("="*70)

    # Pack the Q2_0 model for Strata
    pack_dir = pack_strata_model(strata_shard1)
    tok_dir = setup_strata_tokenizer(strata_shard1)
    mtp_dir = build_mtp_runtime(strata_shard1)

    all_results = []

    # BELLS: just the auto baseline (MTP is broken due to KV cache bug)
    bells_results = run_bells(bells_model, None)
    all_results.extend(bells_results)
    time.sleep(5)

    strata_result = run_strata(pack_dir, tok_dir, ple_gguf=strata_ple,
                               native_gguf=strata_shard1, mtp_dir=mtp_dir)
    all_results.append(strata_result)

    print("\n" + "="*70)
    print("HEAD-TO-HEAD COMPARISON RESULTS")
    print("="*70)
    print(f"{'Engine':<25} {'Avg tok/s':>10} {'Min':>8} {'Max':>8} {'Effective':>10}")
    print("-"*61)
    for r in all_results:
        if r.get("error"):
            print(f"{r['engine']:<25} {'FAILED':>10}   {r.get('error', '')}")
        else:
            print(f"{r['engine']:<25} {r.get('avg_tps','?'):>10} "
                  f"{r.get('min_tps','?'):>8} {r.get('max_tps','?'):>8} "
                  f"{r.get('effective_tps','?'):>10}")

    best_bells = max(
        (r for r in all_results if r.get("engine", "").startswith("BELLS") and r.get("avg_tps")),
        key=lambda r: r["avg_tps"], default=None
    )
    strata = next((r for r in all_results if r.get("engine", "").startswith("Strata")), None)

    print(f"\n  Note: BELLS uses Q2_K_XL, Strata uses Q2_0 (both ~2-bit)")
    if best_bells and strata and strata.get("avg_tps"):
        ratio = best_bells["avg_tps"] / strata["avg_tps"]
        diff = best_bells["avg_tps"] - strata["avg_tps"]
        winner = "BELLS" if diff > 0 else "Strata"
        print(f"  BELLS: {best_bells['engine']} ({best_bells['avg_tps']} tok/s)")
        print(f"  Strata: {strata['engine']} ({strata['avg_tps']} tok/s)")
        print(f"  {winner} wins by {abs(diff):.1f} tok/s ({ratio:.2f}x)")
    elif best_bells and (not strata or strata.get("error")):
        print(f"  BELLS: {best_bells['engine']} ({best_bells['avg_tps']} tok/s)")
        print(f"  Strata: FAILED to produce results")

    return json.dumps(all_results, indent=2)


@app.local_entrypoint()
def main():
    print("BELLS vs Strata: Head-to-Head on Same A10")
    print("=" * 60)
    result = compare_engines.remote()
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(result)
