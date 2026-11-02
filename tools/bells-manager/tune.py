"""Time-budgeted BELLS autotuner.

Runs llama-server for a bounded number of configurations, measures decode
throughput, and prints a ready-to-run command for the fastest configuration.
The search is data-driven: slot counts come from the server's own auto-sizing,
not from per-model or per-quant constants.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import system


PROMPT = "Explain why the sky is blue in two sentences."
GEN_TOKENS = 128


@dataclass
class Trial:
    label: str
    args: list[str]
    auto: bool = False
    kv_type: str = "f16"
    pinned: bool = False
    slots: Optional[int] = None
    n_layer: Optional[int] = None
    expert_mib: Optional[float] = None
    gen_tps: float = 0.0
    prompt_tps: float = 0.0
    load_s: float = 0.0
    error: str = ""
    log_tail: str = ""

    def ok(self) -> bool:
        return self.error == "" and self.gen_tps > 0.0


@dataclass
class Tuner:
    server: str
    model: str
    context: int = 65536
    minutes: float = 10.0
    host: str = "127.0.0.1"
    allow_kv_quant: bool = True
    slots_override: Optional[int] = None
    output: Optional[str] = None
    dry_run: bool = False
    trials: list[Trial] = field(default_factory=list)
    sys_info: Optional[system.SystemInfo] = None

    def hardware(self) -> system.SystemInfo:
        info = system.detect()
        self.sys_info = info
        return info

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, 0))
            return int(s.getsockname()[1])

    def _base_args(self, pinned: bool, kv_type: str, slots: Optional[int], threads: int, auto: bool = False) -> list[str]:
        # A model that does not fit in RAM must stay mmap-backed. Use total RAM:
        # the OS can reclaim page cache while the weights are being pinned.
        model_gb = os.path.getsize(self.model) / (1024 ** 3)
        ram_gb = (self.sys_info.ram_total_mb if self.sys_info else 0) / 1024.0
        load_to_ram = pinned or model_gb < ram_gb * 0.85

        args = [
            "-m", self.model,
            "-ngl", "99",
            "--cpu-moe-pinned" if pinned else "--cpu-moe",
        ]
        if not pinned and not load_to_ram:
            args += ["-lm", "mmap"]
        else:
            args += ["-lm", "none"]

        args += [
            "-c", str(self.context),
            "-np", "1",
            "--flash-attn", "on",
            "-t", str(threads),
            "-ctk", kv_type,
            "-ctv", kv_type,
            "--host", self.host,
        ]
        if auto:
            args += ["--bells-slots", "-1", "--bells-reserve", "1024"]
        elif slots is not None:
            args += ["--bells-slots", str(slots)]
        return args

    def _run_trial(self, trial: Trial) -> Trial:
        port = self._free_port()
        args = trial.args + ["--port", str(port)]
        log_path = Path(self.output or "bench-results") / f"tune-{int(time.time())}-{len(self.trials)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        print(f"\n[{trial.label}]")
        print("  " + subprocess.list2cmdline([self.server] + args))
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(
                [self.server] + args,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            try:
                if not self._wait_health(proc, port, t0):
                    trial.error = f"server failed to start (exit {proc.poll()})"
                    return trial
                trial.load_s = time.monotonic() - t0
                trial.slots = self._parse_slots(log_path)
                trial.n_layer, trial.expert_mib = self._parse_working_set(log_path)
                trial.gen_tps, trial.prompt_tps = self._measure(port)
                if trial.gen_tps <= 0:
                    trial.error = "no timing data"
            except Exception as exc:
                trial.error = str(exc)
            finally:
                self._stop(proc)
                trial.log_tail = self._tail(log_path)
        return trial

    def _wait_health(self, proc: subprocess.Popen, port: int, t0: float) -> bool:
        deadline = time.monotonic() + max(60.0, self.minutes * 60.0)
        url = f"http://{self.host}:{port}/health"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(1.0)
        return False

    def _measure(self, port: int) -> tuple[float, float]:
        body = json.dumps({
            "prompt": PROMPT,
            "n_predict": GEN_TOKENS,
            "temperature": 0,
            "seed": 1234,
            "stream": False,
        }).encode()
        url = f"http://{self.host}:{port}/completion"
        url = f"http://{self.host}:{port}/completion"
        gens: list[float] = []
        prompts: list[float] = []
        # Three runs, median: the first warms the graph and the caches.
        for _ in range(3):
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            timings = data.get("timings") or {}
            gens.append(float(timings.get("predicted_per_second") or 0.0))
            prompts.append(float(timings.get("prompt_per_second") or 0.0))
        gens.sort()
        prompts.sort()
        return gens[len(gens)//2], prompts[len(prompts)//2]

    @staticmethod
    def _parse_slots(log_path: Path) -> Optional[int]:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        m = re.search(r"(\d+) slots/layer", text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_working_set(log_path: Path) -> tuple[Optional[int], Optional[float]]:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None, None
        m = re.search(r"working set [\d.]+ GiB \((\d+) layers x \d+ experts x ([\d.]+) MiB\)", text)
        if not m:
            return None, None
        return int(m.group(1)), float(m.group(2))

    @staticmethod
    def _tail(log_path: Path) -> str:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-8:])
        except OSError:
            return ""

    @staticmethod
    def _stop(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _estimate_slots(self) -> tuple[Optional[int], Optional[float], Optional[int], bool]:
        if self.slots_override is not None:
            return self.slots_override, None, None, True
        if not self.sys_info or not self.sys_info.gpus:
            return None, None, None, False
        try:
            root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(root / "gguf-py"))
            import gguf
            reader = gguf.GGUFReader(self.model)
            arch = reader.get_field("general.architecture").contents()
            block = reader.get_field(f"{arch}.block_count")
            if block is None:
                return None, None, None, False
            n_layer = int(block.contents())
            per_expert = 0
            for t in reader.tensors:
                if not (t.name.startswith("blk.0.ffn_") and t.name.endswith("_exps.weight")):
                    continue
                n_expert = int(t.shape[-1])
                if n_expert <= 1:
                    continue
                per_expert += t.data.nbytes / n_expert
            if per_expert <= 0:
                return None, None, n_layer, False
            free_mb = self.sys_info.gpus[0].vram_free_mb
            slots = max(4, int(free_mb * 1024 * 1024 * 0.5 / (per_expert * n_layer)))
            return slots, per_expert / (1024 * 1024), n_layer, True
        except Exception:
            return None, None, None, False

    def _candidates(self, best: Trial) -> list[Trial]:
        threads = max(1, min(self.sys_info.cpu_cores if self.sys_info else 8, 8))
        out: list[Trial] = []
        kv = "q8_0" if self.allow_kv_quant else best.kv_type

        if best.n_layer and best.expert_mib and self.sys_info and self.sys_info.gpus:
            free_mb = self.sys_info.gpus[0].vram_free_mb
            per_slot_mib = best.n_layer * best.expert_mib
            seen = {best.slots}
            for frac in (0.65, 0.55, 0.75, 0.47, 0.40):
                slots = max(4, int(free_mb * frac / per_slot_mib))
                if slots in seen:
                    continue
                seen.add(slots)
                out.append(Trial(
                    label=f"{slots} slots ({int(frac*100)}% of free VRAM) + {kv} KV",
                    args=self._base_args(best.pinned, kv, slots, threads),
                    slots=slots, kv_type=kv, pinned=best.pinned,
                    n_layer=best.n_layer, expert_mib=best.expert_mib,
                ))

        if best.slots and self.sys_info and self.sys_info.ram_total_mb / 1024 > os.path.getsize(self.model) / (1024 ** 3) * 1.15:
            out.append(Trial(
                label=f"{best.slots} slots + {kv} KV (pinned)",
                args=self._base_args(True, kv, best.slots, threads),
                slots=best.slots, kv_type=kv, pinned=True,
                n_layer=best.n_layer, expert_mib=best.expert_mib,
            ))
        return out

    def run(self) -> list[Trial]:
        info = self.hardware()
        print("Hardware")
        print(f"  CPU:  {info.cpu} ({info.cpu_cores} logical)")
        print(f"  RAM:  {info.ram_total_mb // 1024} GB total, {info.ram_free_mb // 1024} GB free")
        for g in info.gpus:
            print(f"  GPU {g.index}: {g.name}, {g.vram_mb} MB ({g.vram_free_mb} MB free), {g.vendor}")
        print(f"  Model: {self.model}")
        print(f"  Server: {self.server}")
        print(f"  Budget: {self.minutes:.1f} minutes")

        if self.dry_run:
            return []

        deadline = time.monotonic() + self.minutes * 60.0
        model_gb = os.path.getsize(self.model) / (1024 ** 3)
        ram_gb = info.ram_total_mb / 1024.0
        pinned = False  # --cpu-moe is the portable baseline; pinned is optional
        threads = max(1, min(info.cpu_cores, 8))

        base_kv = "q8_0" if self.allow_kv_quant else "f16"
        slots, expert_mib, n_layer, is_moe = self._estimate_slots()
        baseline = Trial(
            label=(f"{slots} slots (VRAM budget) + {base_kv} KV (cpu-moe)" if is_moe
                   else f"no BELLS + {base_kv} KV (cpu-moe)"),
            args=self._base_args(pinned, base_kv, slots, threads),
            auto=False, kv_type=base_kv, pinned=pinned,
            slots=slots, n_layer=n_layer, expert_mib=expert_mib,
        )
        self.trials.append(self._run_trial(baseline))
        best = max((t for t in self.trials if t.ok()), key=lambda t: t.gen_tps, default=None)
        if best is None:
            return self.trials

        load_est = max(15.0, best.load_s)
        for cand in self._candidates(best):
            remaining = deadline - time.monotonic()
            if remaining < load_est + 40.0:
                print(f"\nStopping: {remaining:.0f}s left, next load needs about {load_est:.0f}s")
                break
            self.trials.append(self._run_trial(cand))
            best = max((t for t in self.trials if t.ok()), key=lambda t: t.gen_tps, default=best)

        return self.trials

    def command_for(self, best: Trial) -> str:
        args = list(best.args)
        args += ["--port", "8099"]
        return subprocess.list2cmdline([self.server] + args)

    def report(self) -> int:
        print("\nResults")
        for t in self.trials:
            if t.ok():
                print(f"  {t.gen_tps:7.2f} t/s  load {t.load_s:5.0f}s  {t.label}")
            else:
                print(f"  {'FAIL':>7}       {t.label}: {t.error}")

        best = max((t for t in self.trials if t.ok()), key=lambda t: t.gen_tps, default=None)
        if best is None:
            print("\nNo configuration completed successfully.")
            return 1

        print(f"\nBest: {best.gen_tps:.2f} t/s ({best.prompt_tps:.2f} t/s prompt)")
        print(self.command_for(best))
        if best.kv_type == "q8_0":
            print("NOTE: q8 KV is not bit-exact. Use --no-kv-quant to benchmark f16 KV only.")

        if self.output:
            out = Path(self.output) / f"tune-{time.strftime('%Y%m%d-%H%M%S')}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "model": self.model,
                "server": self.server,
                "context": self.context,
                "best": {
                    "tps": best.gen_tps,
                    "prompt_tps": best.prompt_tps,
                    "load_s": best.load_s,
                    "slots": best.slots,
                    "kv": best.kv_type,
                    "args": best.args,
                },
                "trials": [
                    {"label": t.label, "gen_tps": t.gen_tps, "prompt_tps": t.prompt_tps,
                     "load_s": t.load_s, "slots": t.slots, "error": t.error}
                    for t in self.trials
                ],
            }, indent=2), encoding="utf-8")
            print(f"JSON: {out}")
        return 0


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", help="Path to the .gguf model")
    p.add_argument("--server", help="Path to llama-server (auto-detected by default)")
    p.add_argument("-c", "--context", type=int, default=65536, help="Context window (default: 65536)")
    p.add_argument("--minutes", type=float, default=10.0, help="Wall-clock budget (default: 10)")
    p.add_argument("--no-kv-quant", action="store_true", help="Do not test q8 KV")
    p.add_argument("--slots", type=int, help="Override the derived BELLS slots/layer")
    p.add_argument("--dry-run", action="store_true", help="Print hardware and exit")
    p.add_argument("-o", "--output", default="bench-results", help="Directory for logs and JSON")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark BELLS for a fixed time and print the best flags")
    add_arguments(p)
    args = p.parse_args(argv)

    model = args.model
    if not model:
        models = system.find_models()
        if len(models) == 1:
            model = models[0]["path"]
        else:
            print("Specify --model PATH. Local models:")
            for m in models:
                print(f"  {m['path']}  {m['quant']}  {m['size_gb']} GB")
            return 2

    server = args.server or system.find_binaries().get("llama-server")
    if not server:
        print("llama-server not found. Build BELLS first or pass --server PATH.")
        return 2

    tuner = Tuner(
        server=server,
        model=model,
        context=args.context,
        minutes=args.minutes,
        allow_kv_quant=not args.no_kv_quant,
        slots_override=args.slots,
        output=args.output,
        dry_run=args.dry_run,
    )
    tuner.run()
    if args.dry_run:
        return 0
    return tuner.report()


if __name__ == "__main__":
    raise SystemExit(main())
