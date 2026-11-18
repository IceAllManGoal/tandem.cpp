# Windows native port: compatibility audit

Status: **P0-P2 implemented in `feature/windows-native-support`; nothing has been built, tested or benchmarked on
Windows.** Do not claim Windows support works until the steps in [Verification](#9-verification-plan-not-yet-run)
have actually run.

- Date: 2026-11-18
- Branch: `feature/windows-native-support` (from `f6dbbbf`, base `main` untouched, no PR, no merge)
- Target: native Windows x64, MSVC, NVIDIA CUDA, `llama-server.exe`
- Sources inspected: `tandem.patch` (38 files, +13088/-591, 281 hunks), `tandem-run.py`, `build.sh`,
  `BASE_COMMIT` (`f5222f25c`), `tests/`, `README.md`, `docs/WINDOWS_PORT_PLAN.md`

## 1. Method, and what was *not* done

This audit is a **static reading** of the patch and the launcher. No compiler was run, no model was
downloaded, no test or benchmark was executed, and `main` was not modified.

Upstream facts were checked against the pinned commit by fetching three files directly
(`ggml/src/ggml-cuda/vendors/cuda.h`, `ggml/src/ggml-cuda/common.cuh`,
`ggml/src/ggml-cuda/CMakeLists.txt`); a shallow clone of the whole tree was attempted and abandoned because the
fetch stalled, so the patched tree itself has **not** been materialised here.

Two host facts were verified on a Windows box (Python 3.13, `sys.platform == "win32"`):

```
os.killpg: absent   os.uname: absent   os.sched_getaffinity: absent   signal.SIGKILL: absent
```

The patch is **not** a fork of llama.cpp: it is a patch against `f5222f25c` that `build.sh` (and now
`build.ps1`) applies. None of the changes below add or remove context lines of the upstream files, so the patch
remains a plain `git apply` patch.

## 2. Compile blockers on MSVC (P0)

Three added calls were POSIX-only and unguarded. Each is a hard error for MSVC, not a warning.

| # | File / function | What it was | Why it fails | Fix (in `7572932`) |
|---|---|---|---|---|
| B1 | `src/llama-model.cpp`, `llama_model_base::load_tensors_ple_gpu()` | `setenv("GGML_CUDA_REGISTER_HOST", "1", 0);` | MSVC has no `setenv` (C2065). It sat above the `#ifdef __linux__` block of the same function, so every build compiled it. | `_putenv_s(...)` under `#if defined(_WIN32)`, `setenv` unchanged elsewhere |
| B2 | `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_host_add()` | `__atomic_thread_fence(__ATOMIC_ACQUIRE);` | GCC/Clang builtin, absent from MSVC's C front end. | `_ReadWriteBarrier()` under `#if defined(_MSC_VER)` (+ guarded `<intrin.h>`); x86-64 loads are already acquire, only the compiler barrier is needed |
| B3 | `src/llama-moe-cache.cpp`, `impl::stats_tick()` | `struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);` | `clock_gettime`/`CLOCK_REALTIME` do not exist in the UCRT. | `std::chrono::system_clock` (same wall clock, same JSON field) |

One piece of hardening went in with them:

- `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_backend_cuda_register_host_buffer()`: the retry without
  `cudaHostRegisterReadOnly` now happens on **any** failure, not only on `cudaErrorNotSupported`. Other platforms
  report that unsupported flag with other error codes; a second failing attempt is harmless.

Linux behaviour is unchanged: every new branch keys on `_WIN32`/`_MSC_VER`, and `stats_tick()` reads the same clock
through a portable API.

## 3. What was already platform-guarded (and therefore is *not* a blocker)

The patch is far more portable than the original plan assumed. Counted in `tandem.patch`: 25
`#ifdef __linux__`, 2 `#ifndef __linux__`, 3 pre-existing `_WIN32` branches (`common/arg.cpp` twice,
`common/common.cpp` once), 2 x86 guards and 5 `#if defined(GGML_USE_VMM)` blocks.

| Area | Guarded code | Behaviour off Linux |
|---|---|---|
| Address space | `mmap`/`munmap`, `madvise(MADV_*)`, `MAP_ANONYMOUS` | `load_tensors_ple_gpu()` and `load_tensors_exps_pin()` are Linux-only; on Windows they return early (feature off, no crash) |
| Page size / residency | `sysconf(_SC_PAGE_SIZE)` (7 sites), `mincore` (3), `getrusage` | guarded; `pages_resident()`/`expert_resident()` report "resident", `fault_check()` is a no-op |
| Page cache / IO | `/proc/self/maps`, `/proc/self/buddyinfo|meminfo|zoneinfo|smaps_rollup`, `/sys/dev/block/...`, `posix_fadvise`, `pread`, `open(O_DIRECT)`, `fstat` | the arena and the fault-thread pool compile to empty bodies; `arena_auto_room()` returns 0, so no arena is created |
| CPU topology | `sched_getaffinity`, `/sys/devices/system/cpu/.../topology`, `pthread_setaffinity_np` | `moe_physical_cores()` falls back to `hardware_concurrency()/2` |
| cgroups / RAM | `/proc/meminfo`, `/proc/self/cgroup`, `/sys/fs/cgroup/...` | `llama_moe_ram_limit()` returns 0 = "unknown" |
| Headers | `<sys/mman.h>`, `<unistd.h>`, `<pthread.h>`, `<sys/{resource,stat,sysmacros}.h>`, `<fcntl.h>`, `<sched.h>`, `<immintrin.h>` | only inside the guards above |
| x86 kernels | `__attribute__((target(...)))`, `_Pragma("GCC unroll 4")`, `__builtin_cpu_*`, `#pragma clang attribute`, `#pragma GCC target` | under `#if defined(__x86_64__) && (__GNUC__ || __clang__)`; MSVC defines neither, so they are **not compiled** (see §5) |
| Device code | `asm volatile("mov.u64 %0, %%globaltimer;")`, `__nanosleep`, `__shfl_*` | nvcc/device-only: identical on Windows |

## 4. CUDA on Windows

Verified in the pinned tree:

- `GGML_USE_VMM` is defined in
  [`common.cuh`](https://raw.githubusercontent.com/ggml-org/llama.cpp/f5222f25c18da062ab2b8a2dbbfbf4feae398566/ggml/src/ggml-cuda/common.cuh)
  whenever `GGML_CUDA_NO_VMM` is **not** set, for CUDA and HIP alike.
- [`ggml-cuda/CMakeLists.txt`](https://raw.githubusercontent.com/ggml-org/llama.cpp/f5222f25c18da062ab2b8a2dbbfbf4feae398566/ggml/src/ggml-cuda/CMakeLists.txt)
  makes `GGML_CUDA_NO_VMM` opt-in and links `CUDA::cuda_driver` in that case, on Windows too.
- `ggml_cuda_device_info` already carries `bool vmm; size_t vmm_granularity;`, which the patch reads.

Consequences: the MoE cache's VMM banks and the `LLAMA_MOE_FLEX` slot lending are *compiled and available* on
Windows, subject to the driver reporting `cudaDevAttrVirtualMemoryManagement` (WDDM and TCC both do with current
drivers). `cuMemCreate` on Windows needs `win32HandleMetaData` only for IPC export, which is not used.

Caveat to keep in mind for the first real run: the NVIDIA driver API documents VMM as supported on Windows, but
the granularity is larger than on Linux; `ggml_backend_cuda_vmm_chunk()` already rounds to 4 MiB and pads by
`vmm_granularity`, so this only changes the memory overhead, not correctness.

## 5. Functionality that stays Linux-only (by design, for now)

These are compiled out on Windows. They are **correct** (fallbacks exist), but they are the performance gap.

| Feature | Entry point | What Windows loses | Cost of porting |
|---|---|---|---|
| Expert arena with `O_DIRECT` reads, store mode, exclusive store mode | `LLAMA_MOE_ARENA_MIB`, `LLAMA_MOE_STORE_AUTO`, `LLAMA_MOE_ARENA_ASYNC/EXCL` | the "model larger than RAM" strategy: parallel reads from the SSD instead of one page fault at a time; store mode needs the arena | medium: `FILE_FLAG_NO_BUFFERING` + aligned buffers, and `PrefetchVirtualMemory`/`OfferVirtualMemory` only work on private, not mapped, memory |
| Fault-thread pool (parallel page touching) | `LLAMA_MOE_FAULT_THREADS` | mapped-file reads stay at queue depth ~1 | small: the body is portable except `madvise` (already guarded) |
| Copy of mapped experts into pinned RAM | `LLAMA_MOE_PIN_EXPS` (on by default on Linux) | experts are uploaded from the file mapping instead of pinned memory | small: `VirtualAlloc` instead of `mmap(MAP_ANONYMOUS)`, drop the huge-page hints |
| Anonymous copy of the per-layer embedding table | `LLAMA_PLE_GPU_COPY` (default auto) | `LLAMA_PLE_GPU=1` (a release default) usually ends on the CPU path for Qwen3.8: file-backed pages cannot be pinned, and the copy fallback is Linux-only | small: same `VirtualAlloc` change |
| Huge pages, page-cache eviction | `LLAMA_MOE_PIN_HUGE`, `MADV_PAGEOUT`, `MADV_DONTNEED` | no explicit 2 MiB THP request; file pages are managed by the Windows cache manager | n/a: Windows has no equivalent of `MADV_PAGEOUT` for file mappings |
| `mincore`-based residency, `getrusage` fault-rate switching | `LLAMA_MOE_UPLOAD_ORDER`, `LLAMA_MOE_UPLOAD_PROBE`, fault auto-on/off | decisions that need "is this expert in RAM" see "yes" unconditionally | small: `QueryWorkingSetEx` (different semantics for mapped files) |
| x86 CPU kernels: Q2_0 dot/dot4 with AVX2/AVX-VNNI, IQ2/IQ3 multi-token kernels, AVX2 lookahead router | `LLAMA_MOE_CPU_Q2`, `LLAMA_MOE_CPU_IQ*`, `LLAMA_MOE_LOOKA` | the hybrid CPU tier falls back to ggml's one-token `vec_dot`: same results, fewer tok/s once the CPU takes a share of the experts | medium: separate TU compiled with `/arch:AVX2` + CPUID dispatch, or build with clang-cl, where the existing code compiles unchanged |

`LLAMA_MOE_CPU=auto` adapts the share to whatever the CPU actually delivers, so a slower CPU tier is mostly a
throughput loss, not a correctness one.

## 6. Launcher (`tandem-run.py`) findings

All of the following are fixed in `ef0c924` unless marked otherwise.

| Problem | Where | Symptom on Windows |
|---|---|---|
| `os.uname()` | `detect()` | `AttributeError` **before the plan is printed**: the launcher did nothing at all |
| `/proc/meminfo`, `/proc/self/cgroup`, `/sys/fs/cgroup` | `detect_ram()`, `cgroup_mem_limit()` | 0 GiB RAM -> no `-lm none`, a spurious `--cache-ram 0`, a false "RAM is much smaller than the model" warning |
| `/proc/cpuinfo`, `os.sched_getaffinity` | `detect_cores()` | affinity already handled; cores fell back to `cpu_count()/2` (wrong on hybrid CPUs, wrong without SMT) |
| `os.killpg(pid, SIGTERM/SIGKILL)` | `stop_group()` and the SIGINT handler | `AttributeError` (also in calibration, which stops the server between candidates) |
| `signal.SIGKILL` | `stop_group()` | absent on Windows |
| `start_new_session=True` | server spawn | silently ignored (POSIX-only), so no group to stop |
| `libllama.so` marker scan | `build_state()` | always `"unknown"` (the marker lives in `llama.dll` / the exe) |
| `shlex.quote` in the printed command | `print_plan()`, `main()` | POSIX quoting: "copy this command" does not work in cmd/PowerShell |
| `shlex.split(args.extra)` | `choose_flags()` | backslashes in Windows paths are eaten as escapes |
| `~/.cache/tandem` | cache | works, but not where Windows apps put caches |
| `./tandem/build/bin/llama-server` | default `--server-bin` | misses `.exe` and the VS generator's `Release/` subdirectory |
| `systemd-run` / `taskset` wrappers | `build_command()` | only used with `--assume-*`; would fail with `FileNotFoundError` |
| `/tmp/tandem-test-cache` in the tests | `tests/test_plans.py` | wrote under `C:\tmp` |

## 7. Build

`build.sh` cannot be reused as-is on Windows: `cmake --build build -j --target llama-server` without
`--config Release` builds Debug with the Visual Studio generator (and then the executable is under `Release/`).

`build.ps1` (commit `9b7a221`) mirrors `build.sh` step by step: toolchain checks (`git`, `cmake`, `nvcc` from
`PATH` or `CUDA_PATH`, the VS C++ toolset via `vswhere`), clone + checkout of `BASE_COMMIT`, the same
apply / already-applied / does-not-apply logic, CMake configure with `-DGGML_CUDA=ON`, and
`cmake --build --config Release --target llama-server`. It reports the exact `.exe` path and never downloads
models.

## 8. Implemented changes (P0-P2)

| Commit | Contents |
|---|---|
| `7572932` | patch: the three MSVC blockers + the host-registration retry; hunk headers recomputed (`git apply --stat` parses: 38 files, +13088/-591) |
| `ef0c924` | launcher: Windows hardware detection, process groups and stop, quoting, cache dir, binary defaults, `--extra` splitting |
| `0e9956f` | tests: portable plan tests, new `tests/test_platform.py`, `apply_assumes()` so the fake-hardware path is testable |
| `9b7a221` | build.ps1 |

Not implemented: P3 (arena/`O_DIRECT`, `PIN_EXPS`, `PLE_GPU_COPY`, `llama_moe_ram_limit()` for Windows), x86 kernels
for MSVC, README claims, CI.

## 9. Verification plan (not yet run)

1. `git apply --check` of the patch on a fresh `f5222f25c` checkout (both platforms).
2. `build.ps1 -PatchCheckOnly`, then `build.ps1` (a full nvcc build).
3. `llama-server.exe --version`; `llama-server.exe --help | findstr moe-cache` (the new flags must be there).
4. `python tandem-run.py --model <path> --dry-run`: the hardware block must show the real GPU/VRAM/cores/RAM/disk
   and the OS line, and the plan must match the Linux plan for the same machine (`--fake-hw` compares the rules).
5. CLI smoke on a small MoE GGUF: server starts, one request answers, the log shows the expected
   `LLAMA_MOE_*` lines and no silent fallback of VMM/flex.
6. Correctness: identical tokens with `-cmoe` and without, and with `LLAMA_MOE_CPU=0` and `auto`.
7. Only then: reproducible performance numbers (same model/quant/prompt/context/hardware, warm and cold identified).

Acceptance: 1-6 pass and are reported with the exact commands and output. A CPU-only machine is not a substitute
for a CUDA runner.

## 10. Risks and open questions

- **Free-VRAM reporting under WDDM**: Windows reports device memory optimistically and the driver can spill to
  shared memory, so `--moe-cache-mib auto` may pick a size that later thrashes. If the first run shows it, raise
  `--moe-cache-margin-mib` (512 MiB is the CUDA default, tuned on Linux).
- **Commit charge**: pinned experts and CUDA host buffers count against RAM + pagefile. `detect_ram()` therefore
  uses `min(physical, commit limit)` on Windows.
- **MSVC vs clang-cl** for the x86 kernels: clang-cl compiles the existing `__attribute__((target))` code as-is
  and would restore the Q2_0/IQ fast paths. Decision deferred until the MSVC build is validated (per the
  operator's instruction).
- **Hybrid CPUs** (P/E cores): `GetLogicalProcessorInformationEx` counts physical cores correctly, but thread
  placement is still Windows' scheduler's; the CPU share may need tuning on such machines.
- **>64 logical processors**: `detect_cores_windows()` counts processor group 0 affinity and treats higher groups
  as visible; good enough for the plan, not a topology oracle.
- **Antivirus / file locks**: a 78 GiB mmap plus Windows Defender scanning the GGUF is a plausible source of
  first-run anomalies; the exclusion should be documented when measured.
- **Large `-lm none` loads**: the `CUDA_Host` pinned copy on Windows may fail earlier than on Linux
  (`cudaErrorMemoryAllocation`); the launcher's `build_state()` and the server's own RAM heuristic both assume
  `llama_moe_ram_limit()` works, which it does not yet on Windows (P3).

## 11. Limitations of this audit

- No compilation was performed, so "three blockers" means *found statically*; MSVC may still report places this
  reading did not predict (two-phase lookup in templates, `/WX` warnings, nvcc + MSVC host quirks).
- The patched tree was never materialised here; findings come from reading the diff plus three upstream files.
- Nothing here measures performance on Windows. Any number in the README remains a Linux number.
