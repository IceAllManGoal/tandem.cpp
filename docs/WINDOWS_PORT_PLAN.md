# Windows native port: audit and DSH handoff

Status: **planning/audit only**. No Windows build, runtime test, or benchmark has been performed. Do not claim Windows support works.

## Scope and upstream strategy

- Keep `main` as a clean upstream-tracking fork of `ewhacc/tandem.cpp`.
- Work only in `feature/windows-native-support`; submit small upstream PRs after tests.
- Preserve Linux behavior and existing MoE accuracy/performance.
- The repository is a patch set, **not** a standalone llama.cpp tree. `BASE_COMMIT` is `f5222f25c`; `build.sh` clones llama.cpp and applies `tandem.patch`.
- Initial target: native Windows x64, MSVC, NVIDIA CUDA, `llama-server.exe`; WSL is not the acceptance target.

## Source audit (2026-11-18, v0.4.0)

Inspection of `tandem.patch` and `tandem-run.py` found these **portability blockers/candidates** (line numbers refer to the current patch/launcher, not patched C++ source):

| Area | Evidence | Required investigation |
|---|---|---|
| Linux memory mapping / reclaim | `tandem.patch` ~6038–6156: `mmap`, `munmap`, `madvise`, huge pages, `MADV_PAGEOUT`, `MADV_DONTNEED` | Abstract platform-specific memory allocation and eviction; check Windows file mapping, `VirtualAlloc`, `VirtualFree`, `DiscardVirtualMemory` / `OfferVirtualMemory` semantics rather than blindly replacing calls |
| Linux RAM / huge-page telemetry | ~5926–5959, ~6170, ~8306–8323: `/proc/buddyinfo`, `/proc/meminfo`, `/proc/self/smaps_rollup`, `/proc/zoneinfo`, `sysconf` | Windows memory APIs and graceful fallbacks; do not assume identical page-reclaim behavior |
| Linux CPU topology | ~6476–6502: `unistd.h`, `sys/resource.h`, `/sys/devices/system/cpu/...` | Windows processor topology, hybrid P/E cores, affinity and priority |
| Linux SSD / mapping diagnostics | ~7295–7305, ~7430–7453: `/proc/self/maps`, `/sys/dev/block/.../stat` | Windows mapping/IO statistics; make optional telemetry nonfatal |
| Linux page-fault/prefetch behavior | ~7710–7734, ~8002–8008: `madvise`, `sysconf(_SC_PAGE_SIZE)` | Preserve correctness first; assess `PrefetchVirtualMemory` and memory-mapped file behavior |
| Launcher hardware detection | `tandem-run.py` ~87–164: `os.sched_getaffinity`, `/proc/cpuinfo`, cgroups, `os.uname` | Platform backend for Windows CPU/RAM/OS detection |
| Launcher process lifecycle | ~618–623, ~799–801: `os.killpg`, Unix signal forwarding | Windows subprocess creation and process-tree cleanup; `.exe` path resolution |
| CUDA | Patch uses CUDA memory management and pinned RAM | Check driver/runtime/toolkit compatibility and Windows support for required CUDA VMM operations |

These are source-level findings, **not** proven compile errors. Other issues may appear when DSH runs the first Windows build.

## Implementation sequence for DSH

1. Read README, LICENSE, BASE_COMMIT, build.sh, tandem-run.py, tandem.patch and tests. Inventory all POSIX/Linux dependencies and classify compile blockers vs runtime behavior vs optional telemetry.
2. Introduce the smallest guarded platform abstractions; retain Linux path unchanged. Avoid a giant fork of llama.cpp.
3. Add a PowerShell build helper with pinned base commit, patch verification, CMake/MSVC/CUDA detection, and explicit error messages. Do not download models as part of build.
4. Port launcher hardware detection and subprocess management using Windows-safe APIs. Preserve `--dry-run` and document configuration.
5. Add Windows CI with compile and smoke tests where feasible. GPU inference requires a real CUDA Windows runner; do not present CPU-only CI as GPU validation.
6. Run on a Windows machine with NVIDIA GPU: patch check, native compile, CLI smoke test, small GGUF inference, MoE cache correctness tests, then reproducible performance comparisons against unpatched pinned llama.cpp.
7. Document Windows prerequisites, known limitations, test commands and measured results. Only then prepare focused PR(s) for upstream.

## Validation and reporting requirements

- No builds or downloads until explicitly initiated in the DSH environment.
- For each change: show diff, explain rationale, run relevant test, record command/output and environment.
- Verify token/output correctness, not just tokens/s. Benchmark same model, quantization, prompt, context, settings and hardware, with warm/cold cases identified.
- Watch for Windows commit charge, mapped-file eviction, pinned-memory pressure, CUDA OOM, hybrid CPU scheduling, paths with spaces, and antivirus/file-lock behavior.
- Do not claim performance improvements without reproducible measurements.
- Keep the upstream license/attribution intact; do not merge into `main` or open upstream PR without review.

## DSH starting prompt

> Work in `IceAllManGoal/tandem.cpp`, branch `feature/windows-native-support`. The goal is a **native Windows x64 + MSVC + NVIDIA CUDA** port of tandem.cpp v0.4.0, maintaining Linux support. Read `docs/WINDOWS_PORT_PLAN.md` and audit the entire patch/launcher first. Do not build, download models, benchmark or modify upstream/main without explicit approval. Start with a prioritized compatibility report and a proposed minimal first patch. After approval, implement incrementally, build/test on Windows, and report actual results and unresolved blockers. Never report unrun tests as passed.
