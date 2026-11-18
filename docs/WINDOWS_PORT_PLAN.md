# Windows native port: audit and DSH handoff

Status: **P0-P2 implemented, not built and not tested on Windows.** No Windows build, runtime test or benchmark has
been performed. Do not claim Windows support works.

Last updated: 2026-11-18. The full source-level audit is in [WINDOWS_PORT_AUDIT.md](WINDOWS_PORT_AUDIT.md).

## Scope and upstream strategy

- Keep `main` as a clean upstream-tracking fork of `ewhacc/tandem.cpp`.
- Work only in `feature/windows-native-support`; submit small upstream PRs after tests.
- Preserve Linux behavior and existing MoE accuracy/performance.
- The repository is a patch set, **not** a standalone llama.cpp tree. `BASE_COMMIT` is `f5222f25c`; `build.sh`
  clones llama.cpp and applies `tandem.patch`. On Windows `build.ps1` has the same contract (pinned commit, patch
  verification, CMake/MSVC/CUDA detection, no model downloads).
- Initial target: native Windows x64, MSVC, NVIDIA CUDA, `llama-server.exe`; WSL is not the acceptance target.

## Source audit (2026-11-18, v0.4.0) - corrected

The first pass of this file listed the `mmap`/`madvise`/`/proc` regions as blockers. Reading the whole patch showed
that they are already behind `#ifdef __linux__` (25 such blocks, plus 2 `#ifndef __linux__`) with explicit
non-Linux fallbacks. The real compile blockers were three unguarded POSIX calls, now fixed:

| Area | Evidence | Status |
|---|---|---|
| MSVC has no `setenv` | `src/llama-model.cpp`, `load_tensors_ple_gpu()` | fixed: `_putenv_s` under `_WIN32` |
| MSVC has no `__atomic_thread_fence` | `ggml-cpu/ggml-cpu.c`, `ggml_compute_forward_host_add()` | fixed: `_ReadWriteBarrier` under `_MSC_VER` |
| MSVC has no `clock_gettime(CLOCK_REALTIME)` | `src/llama-moe-cache.cpp`, `impl::stats_tick()` | fixed: `std::chrono` |
| `cudaHostRegisterReadOnly` is not portable | `ggml-cuda.cu`, `ggml_backend_cuda_register_host_buffer()` | hardened: retry without it on any failure |

CUDA VMM, which the MoE cache's banks and slot lending use, is enabled by default for CUDA builds on every
platform (`GGML_USE_VMM` in `common.cuh`; `GGML_CUDA_NO_VMM` is opt-in in `ggml-cuda/CMakeLists.txt`), and
`ggml_cuda_device_info` already carries `vmm`/`vmm_granularity`.

Everything else is either already guarded or compiled out on Windows. What is compiled out is listed below: it is
the remaining performance gap, not a correctness one.

### Linux-only functionality that remains

| Feature | Flags | Consequence on Windows |
|---|---|---|
| Expert arena with `O_DIRECT`, store mode | `LLAMA_MOE_ARENA_MIB`, `LLAMA_MOE_STORE_AUTO` | no "model larger than RAM" streaming strategy |
| Fault-thread pool | `LLAMA_MOE_FAULT_THREADS` | mapped-file reads at queue depth ~1 |
| Pinned copy of mapped experts | `LLAMA_MOE_PIN_EXPS` (on by default on Linux) | experts are uploaded from the file mapping |
| Anonymous copy of the per-layer embedding table | `LLAMA_PLE_GPU_COPY` | `LLAMA_PLE_GPU=1` mostly ends on the CPU path (Qwen3.8) |
| Huge pages, page-cache eviction | `LLAMA_MOE_PIN_HUGE`, `MADV_PAGEOUT`, `MADV_DONTNEED` | no explicit huge-page request, no forced eviction of file pages |
| `mincore` residency, `getrusage` fault rate | `LLAMA_MOE_UPLOAD_ORDER`, `LLAMA_MOE_UPLOAD_PROBE` | decisions see "always resident" |
| x86 CPU kernels: Q2_0 AVX2/AVX-VNNI, IQ multi-token, AVX2 lookahead | `LLAMA_MOE_CPU_Q2`, `LLAMA_MOE_CPU_IQ*`, `LLAMA_MOE_LOOKA` | the hybrid CPU tier uses ggml's one-token `vec_dot` |
| `llama_moe_ram_limit()` | - | the server itself sees RAM as "unknown" |

## Implementation sequence

1. [x] Read README, LICENSE, BASE_COMMIT, build.sh, tandem-run.py, tandem.patch and tests; inventory the
   POSIX/Linux dependencies and classify them - see `docs/WINDOWS_PORT_AUDIT.md`.
2. [x] Smallest guarded platform abstractions; Linux path unchanged - three calls, no fork of llama.cpp.
3. [x] `build.ps1` with pinned base commit, patch verification, CMake/MSVC/CUDA detection, explicit errors, no
   model downloads.
4. [x] Port launcher hardware detection and subprocess management to Windows-safe APIs; `--dry-run` and the
   configuration surface are preserved.
5. [ ] Windows CI with compile and smoke tests where feasible. GPU inference requires a real CUDA Windows runner;
   do not present CPU-only CI as GPU validation.
6. [ ] Run on a Windows machine with an NVIDIA GPU: patch check, native compile, CLI smoke test, small GGUF
   inference, MoE cache correctness tests, then reproducible performance comparisons against unpatched pinned
   llama.cpp.
7. [ ] Document Windows prerequisites, known limitations, test commands and measured results. Only then prepare
   focused PR(s) for upstream.
8. [ ] P3, after 5-6: arena/`O_DIRECT` port, `PIN_EXPS` and `PLE_GPU_COPY` on Windows, `llama_moe_ram_limit()`
   from `GlobalMemoryStatusEx`, and the x86 kernels for MSVC (or a clang-cl build, which compiles the existing
   code unchanged).

## Validation and reporting requirements

- No builds, downloads or benchmarks until explicitly initiated in the DSH environment. The first build is a
  controlled run: `build.ps1 -PatchCheckOnly`, then `build.ps1`.
- For each change: show diff, explain rationale, run relevant test, record command/output and environment.
- Verify token/output correctness, not just tokens/s. Benchmark same model, quantization, prompt, context,
  settings and hardware, with warm/cold cases identified.
- Watch for Windows commit charge, mapped-file eviction, pinned-memory pressure, CUDA OOM, hybrid CPU scheduling,
  paths with spaces, and antivirus/file-lock behavior.
- Do not claim performance improvements without reproducible measurements; a Linux number is not a Windows number.
- Keep the upstream license/attribution intact; do not merge into `main` or open upstream PR without review.

## DSH starting prompt (next phase)

> Work in `IceAllManGoal/tandem.cpp`, branch `feature/windows-native-support`. P0-P2 are implemented and committed;
> nothing has been built yet. Start with `build.ps1 -PatchCheckOnly`, then `build.ps1`, and report the exact
> commands, toolchain versions and compiler output. Fix what the MSVC build reports without touching `main` or
> Linux behaviour. Then run the verification plan in `docs/WINDOWS_PORT_AUDIT.md` §9 on a real NVIDIA machine and
> report actual results. Never report unrun tests as passed. P3 and clang-cl are separate decisions.
