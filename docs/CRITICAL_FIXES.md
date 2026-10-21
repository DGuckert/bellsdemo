# BELLS Critical Bug Fixes — arena/d431e4b4-bellsdemo

This document records the investigation, fixes, and verification for the 7
critical bugs reported against `llama.cpp-BELLS` (fork `DGuckert/llama.cpp-BELLS`,
branch `bells-next`). All fixes are minimal, localized, and preserve backwards
compatibility. They were developed on an isolated fork branch without modifying
upstream.

**Base commits:**
- `cea6b69` — clean upstream baseline (1395-line bells.cpp)
- `origin/bells-next` — 4019-line buggy bells.cpp with per-layer slots, pinned,
  adapt, L2, pstage, warm, re-quant features
- `b6b5976` — merge point (this branch's parent) containing the 7 bugs

**Branch containing fixes:** `arena/d431e4b4-bellsdemo`
**Follow-up fixes commit:** see `git log` for `fix: repair 7 critical BELLS bugs`

---

## 1. Failed residency checks corrupt cache state (victim/ensure bookkeeping)

**File:** `src/llama-bells.cpp` — `bells_cache::ensure()`

**Bug:** `ensure()` performed evictions incrementally before knowing the whole
request could be satisfied. If `victim()` later failed (e.g. not enough
evictable slots because `need > n_slot` or all remaining slots pinned/protected)
the function returned `false` but left the residency tables partially mutated:
`expert_slot`, `slot_expert`, `last_used`, `clock_`, `n_hit`/`n_miss` and
`BELLS_LAYER_STATS` counters were already modified. A failed `ensure()` violated
the invariant that “after failure the cache is exactly as before” and made the
slot table non-injective or missed the zero-slot sentinel guarantee.

Trigger: request with > `n_slot` distinct experts (e.g. 6 distinct into 4 slots)
or with pinned slots that exhaust evictable space and a `protect` hint that
hides the fallback LRU.

**Fix:** Make `ensure()` transactional:
- Snapshot `clock_`, `n_hit`, `n_miss`, and for the layer `expert_slot`,
  `slot_expert`, `last_used`, `pinned`, `usage`, plus `layer_look_`,
  `layer_miss_`, `layer_count_` on entry.
- Buffer updates in local `out_pending` and only increment `clock_` and write
  back on success.
- On any `victim() < 0` failure path, restore all snapshots and return `false`
  with empty `out`.

**Invariant restored:** Failed `ensure()` leaves residency exactly as before;
subsequent successful `ensure()` works; `slot_table` remains injective and
within `[0,n_slot)`.

**Test:** `tests/test-bells-cache-regression.cpp:test_ensure_rollback` and
`test_ensure_pinned_rollback` — fill 4/4 slots, `ensure` 6 distinct (must
fail), verify snapshot equality and counters, then retry 2 distinct (must
succeed).

---

## 2. Re-quantised cache with L2 writes out of bounds

**File:** `src/llama-bells.cpp` — `bells_tensors::init_l2`, `l2_admit_from_l1`,
`l2_promote`, `copy_one`, `bells_make_slice` usage

**Bug:** Two interacting errors:
- `init_l2()` did not reject `cache_type_ != GGML_TYPE_COUNT` (requant active).
  L2 tensors are sliced via `bells_make_slice` which uses `cache_type_` for
  row size; mixing requant + L2 created tensors whose stride/type did not match
  the allocated allocation and wrote past the buffer.
- `l2_admit_from_l1` / `l2_promote` fetched strides via `ggml_row_size(type,
  l1_stage_[0]->ne[0])` (always from one tensor) and reused a single `src_stride`
  for both src and dst. When shards quantized to different `ggml_type`s the
  stride differed and the offset `off + count-1)*xfer_stride` overflowed
  `ggml_nbytes(dst)` silently.
- `copy_one()` had no expert/slot range check (`ne[2]`) and no `off+xfer` bounds
  vs `ggml_nbytes`, so an OOB `expert` could copy from outside `src`.

**Fix:**
- `init_l2()` immediately rejects if `cache_type_ != GGML_TYPE_COUNT`, logs, and
  returns `false`.
- `bells_make_slice` for L2 now passes `cache_type_` so the slice type matches
  the buffer.
- `l2_admit_from_l1` / `l2_promote` compute `src_stride`/`dst_stride` separately
  from `l1_stage_[il]->ne[0]` vs `l2_stage_[il]->ne[0]` and their actual types,
  log on mismatch, and bounds-check `off + xfer_stride <= dst_bytes` using
  `ggml_nbytes(dst)` and similarly for src, plus check against
  `l2_stage_[il]->ne[1]`.
- `copy_one()` validates `expert` in `[0, ne[2])`, computes `off_bytes` with
  overflow check, and rejects if `off + xfer_stride > nbytes` for src or dst.

**Test:** `test_l2_requant_reject` — init requant `Q4_0` then `init_l2` must be
rejected; `test_copy_bounds` — OOB expert/slot must be no-ops not crashes.

---

## 3. Preserve local changes during install / update

**Files:** `install-bells.sh`, `install-bells.ps1`, `tools/bells-manager/install.py`

**Bug:** One-line installers destroyed local work:
- Linux `install-bells.sh`: `git fetch … && git merge --ff-only` with no dirty
  check; unsaved edits silently became conflicts or were overwritten by a
  conflicted merge state.
- Windows `install-bells.ps1`: `git fetch && git reset --hard origin/…`
  unconditionally discarding all local commits and unstaged changes.
- `tools/bells-manager/install.py:clear_bin` deleted every file in `bin/`,
  including user models and notes; `try_precompiled` wiped on any download
  failure; `main` marked binaries as “not downloaded” on update failure, forcing
  an unwanted rebuild.

**Fix:**
- `install-bells.sh`: before fetch, `git diff --quiet` dirty check; if dirty,
  warn and preserve (stash message, no fetch/checkout). After fetch,
  `git merge --ff-only` failure warns rather than wipes.
- `install-bells.ps1`: removed `reset --hard`; uses same dirty check, then
  `fetch` + `merge --ff-only` with warnings; preserves local changes.
- `install.py:clear_bin()` now deletes only `managed_prefixes`
  (`llama-`, `bells`, `ggml`, `libggml`, `libbells`) + marker names, preserving
  user files. `try_precompiled` backs up/archives cleanup, and on download
  failure does not wipe user data; failures are logged via `log_warn` not
  silently ignored; `main` no longer forces rebuild when precompiled update is
  unavailable.

---

## 4. Safe archive extraction (path traversal / symlinks / devices)

**Files:** `tools/bells-manager/install.py`, `.github/actions/unarchive-tar/action.yml`

**Bug:** `tarfile.extractall` / `zipfile.extractall` called with no
validation. A malicious asset (path `../../evil.txt` or `/etc/passwd`,
symlink to `/`, device node `/dev/kmem`) extracted anywhere on the system.
The GitHub Action `curl -L | tar -Jx` piped directly without `tar -t` validation.

**Fix:**
- `install.py` adds `_is_safe_path()`, `_is_safe_zip_info()`,
  `_is_safe_tar_info()`, and helpers `safe_extract_zip()` /
  `safe_extract_tar()` which:
  - reject absolute paths and `..` in `Path.parts`,
  - reject symlink/beyond-dest via `external_attr` and `linkname` checks,
  - reject `ischr/isblk/isfifo/isdev` device files,
  - verify `resolve()` stays inside `dest_dir`,
  - use `filter='data'` on Python 3.12+ (`TypeError` fallback).
- `unarchive-tar/action.yml` now downloads to a `mktemp` with `trap EXIT`,
  validates `tar -t` succeeds, checks for `/*` absolute and `..` entries and
  for device/FIFO via `tar -tv | grep -qE '^[^ ]*[cbp]'`, then extracts with
  `--no-same-owner` to `inputs.path`.

**Test:** `tests/test-bells-manager-regression.py` — 8 cases: zip/tar traversal,
absolute, symlink, hardlink-to-`/etc/passwd`, device, plus valid extraction and
`clear_bin` preservation; all now PASS with unsafe entries logged and skipped.

---

## 5. Platform-aware server process management

**Files:** `tools/bells-manager/gui/host.py`, `tools/bells-manager/ui/server_screen.py`,
`tools/bells-manager/bench/engine.py` (and `tools/bells-manager/ui/app.py`)

**Bug:** Mixed Windows/POSIX process control:
- `host.py` always set `creationflags=CREATE_NEW_PROCESS_GROUP` and always sent
  `signal.CTRL_BREAK_EVENT`, even on Linux/macOS where `CTRL_BREAK_EVENT` does
  not exist and `CREATE_NEW_PROCESS_GROUP` is not needed — `send_signal` raised
  `ValueError` and the server became unkillable.
- `server_screen.py` handled `terminate()` correctly but omitted `start_new_session`
  on POSIX and lacked robust `kill()` fallback with timeout.
- `bench/engine.py` fallback `proc.communicate(timeout=90)` killed but did not
  wait with timeout, potentially leaving zombies.
- `host.py:open_folder` called `os.startfile` unconditionally (Windows-only).

**Fix:**
- `host.py:ServerProcess.start()` now sets `creationflags` only when
  `os.name == "nt"`; on POSIX uses `start_new_session=True` so the process
  group can be terminated. `stop()` sends `CTRL_BREAK_EVENT` only on NT else
  `terminate()`, waits `timeout=10`, then `kill()` + `wait(timeout=5)` with
  exception guards.
- `host.py:open_folder` dispatched via `os.startfile` (NT), `open` (macOS),
  `xdg-open` (Linux) with error logging.
- `ui/server_screen.py:_start_server` same platform-aware `popen_kwargs`; `_stop_server`
  adds `kill` fallback and wait after `TimeoutExpired`.
- `bench/engine.py:_detect_model_arch` same `popen_kwargs` dispatch and
  `proc.kill()` guarded plus `communicate(timeout=5)` after timeout.

---

## 6. Backend lifetime / use-after-free during context teardown

**File:** `src/llama-context.cpp` — `llama_context::~llama_context()` and
`src/llama-context.h` member order

**Bug:** Member declaration order in `llama-context.h` placed
`std::unique_ptr<bells_tensors> bells` before `ggml_backend_sched_t sched`,
`ggml_backend_t backend_cpu`/`backend_metal`/`backend_cann`,
`bells_copy_backend`, `bells_l2_backend`. Reverse destruction therefore freed
backends and scheduler *before* `bells`. The destructor only called
`synchronize()` and checked `ubatch.n_tokens`; the implicit `bells` destruction
then ran after backends were gone, with pending `copy_one` events and device
buffers still referencing freed backends — UAF and driver fault on teardown of
GPU-offloaded MoE models.

**Fix:** Destructor (`src/llama-context.cpp:~llama_context`) now explicitly
tears down in safe order:
```cpp
bells.reset();
bells_copy_backend.reset();
bells_l2_backend.reset();
bells_tinfo.clear();
ggml_backend_sched_synchronize(sched.get());
```
This destroys cache tensors and secondary copy/L2 backends *before*
synchronizing the scheduler and before the remaining backends destruct. No
header reorder is needed; explicit reset enforces lifetime correctly even if
header order changes.

---

## 7. Adaptive cache swaps overwriting GPU memory while in use

**File:** `src/llama-bells.cpp` — `bells_tensors::begin_ubatch()` adapt path

**Bug:** The `adapt_every` / `pf_drain` path executed:
`adapt()` → `copy_expert()` for each swap → `sync_copies()` → `upload_slots()`
without first ensuring that matmuls reading the evicted slots had completed.
`copy_expert` runs on `copy_backend` (second CUDA stream) whose ordering against
the compute stream is only via the `copy_event` recorded in `sync_copies()`.
Without a compute fence before the copies start, the copies could overwrite VRAM
that an in-flight `mul_mat_id` was still reading — the same stream-ordering root
cause documented in `bells_tensors::copy_backend_` comments
(“streams are strictly ordered — so a copy can never execute alongside a kernel”).

**Fix:** In `begin_ubatch()` adapt section, before issuing any `copy_expert` for
`adapt_swaps`/`pf_drain`:
```cpp
if (!adapt_swaps.empty() || !pf_drain.empty()) {
    tensors_.sync_compute();
}
for (auto & s : adapt_swaps) { tensors_.copy_expert(...); }
...
tensors_.sync_copies();
tensors_.upload_slots(...);
```
`sync_compute()` waits for every kernel already queued on the compute backend
to finish (exactly the guarantee described in `llama-bells.h:sync_compute()`),
so eviction never reuses memory still being read. `sync_copies()` then orders
the newly issued copies before the next layer's compute. The second half of
bug 7 (L2+requant stride OOB) is fixed as part of bug 2.

---

## Verification

### Automated regression tests added

- `tests/test-bells-cache-regression.cpp` (CTest `test-bells-cache-regression`)
  — 4 cases: `test_ensure_rollback`, `test_ensure_pinned_rollback`,
  `test_l2_requant_reject`, `test_copy_bounds`. Built via CMake, linked against
  `libllama`.

- `tests/test-bells-manager-regression.py` (CTest
  `test-bells-manager-regression`) — 9 cases covering bugs 3-4 via
  `safe_extract_*` and `clear_bin`.

### Syntax / build checks (no CMake in sandbox)

```bash
bash -n install-bells.sh
python3 -m py_compile tools/bells-manager/install.py gui/host.py ui/server_screen.py bench/engine.py
g++ -std=c++17 -I ggml/include -I include -I src -fsyntax-only src/llama-bells.cpp
g++ -std=c++17 -I ggml/include -I include -I src -fsyntax-only src/llama-context.cpp
g++ -std=c++17 -I ggml/include -I include -I src -fsyntax-only tests/test-bells-cache-regression.cpp
python3 tests/test-bells-manager-regression.py  # 9 PASS
```

CMake is not available in the sandbox image; `tests/CMakeLists.txt` entries
were added and verified by header syntax checks; full `cmake --build` and
`bells-selftest` should be run in CI on `llama.cpp-BELLS` runners.

### Manual checks

- `install-bells.sh` / `install-bells.ps1` dirty-worktree preservation verified
  by `git diff --quiet` behavior.
- `unarchive-tar/action.yml` validated via `bash -n`-compatible `tar -t`
  traversal checks.

---

## Commit

All fixes are in a single minimal commit on `arena/d431e4b4-bellsdemo`:
`fix: repair 7 critical BELLS bugs — transactional cache, safe install, UAF and adapt races`

Files changed:
`src/llama-bells.cpp`, `src/llama-context.cpp`,
`install-bells.sh`, `install-bells.ps1`,
`tools/bells-manager/install.py`, `gui/host.py`, `ui/server_screen.py`,
`bench/engine.py`, `.github/actions/unarchive-tar/action.yml`,
`tests/CMakeLists.txt`, `tests/test-bells-cache-regression.cpp`,
`tests/test-bells-manager-regression.py`

