# genesis: Breeze supercell on Intel GPUs

This workspace runs [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)'s
`examples/splitting_supercell.jl` (anelastic and compressible cores, 168×168×40, 2 h) on Intel
GPUs through [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), on the KernelAbstractions 0.10
stack. It was brought up on an Intel Arc A750 and reproduced on Aurora (ALCF, one Intel Data
Center GPU Max 1550 tile) on 2026-11-11.

The three packages are git submodules pinned to branches on forks:

| Submodule | Fork branch | Adds on top of upstream |
|---|---|---|
| `Oceananigans.jl` | [`michel2323/Oceananigans.jl@ms/oneAPI-ka0.10`](https://github.com/michel2323/Oceananigans.jl/tree/ms/oneAPI-ka0.10) | KA 0.10 port (draft CliMA/Oceananigans.jl#5799) merged with `main`; the oneAPI extension from CliMA/Oceananigans.jl#4764; oneAPI.jl converts kernel arguments at launch; field data passed as pointer-only device arrays to stay under the 2048 B kernel-argument limit |
| `oneAPI.jl` | [`michel2323/oneAPI.jl@ms/subgroup-16`](https://github.com/michel2323/oneAPI.jl/tree/ms/subgroup-16) | v2.10.0 + KernelAbstractions kernels compiled at sub-group size 16 instead of 32 |
| `Breeze.jl` | [`michel2323/Breeze.jl@ms/oneapi`](https://github.com/michel2323/Breeze.jl/tree/ms/oneapi) | KA 0.10 compat, the example switched from CUDA to `GPU(oneAPIBackend())`, ports for two Oceananigans API changes |

Nothing here is merged upstream. `CLAUDE.md` has the detailed working notes.

## Layout

- `Project.toml`, `Manifest.toml`: the Julia environment; `[sources]` point at the submodules.
- `scripts/smoke_supercell.jl`: bring-up ladder on a 32×32×16 grid (see its header).
- `repro/`: standalone reproducers for the Intel compiler/driver problems below, with their own
  environment (`repro/Project.toml`, which uses the `oneAPI.jl` submodule).
- `aurora/`: PBS job scripts for Aurora (see `aurora/README.md`); logs go to `aurora/logs/`
  (ignored).
- `runs/`: run directories. Committed: the A750 and Aurora references (`runs/supercell_a750/`,
  `runs/supercell_aurora/`) and two modified copies of the example (Float32 pressure solver; CPU
  1-hour reference).

## Quick start

Needs Julia 1.12 (the commands use `juliaup`'s `julia +1.12`; drop `+1.12` if your `julia` is 1.12).

```sh
git clone --recurse-submodules https://github.com/michel2323/genesis.git
cd genesis
julia +1.12 --project=. -e 'using Pkg; Pkg.instantiate()'
```

On the A750, precompiling prints a segfault from AcceleratedKernels' oneAPI extension
(`✗ AcceleratedKernels → oneAPIExt`); it is skipped and not needed. It did not occur on Aurora.

On hardware without native Float64 (Arc A-series), first export the FP64 emulation variables;
the anelastic pressure solver runs in Float64:

```sh
export OverrideDefaultFP64Settings=1 IGC_EnableDPEmulation=1   # A750 only
```

Smoke ladder, CPU first, then oneAPI with the same `OUTDIR` (rung 6 then compares the two):

```sh
ARCH=cpu    RUNG=all OUTDIR=scripts/output julia +1.12 --project=. scripts/smoke_supercell.jl
ARCH=oneapi RUNG=all OUTDIR=scripts/output julia +1.12 --project=. scripts/smoke_supercell.jl
```

Each run ends with a `SUMMARY` table; all six rungs should say `passed`. The oneAPI run prints
`compare compressible: oneapi vs cpu` and `compare anelastic: ...` blocks. On the A750 the
relative differences `max|Δ|/max|ref|` after 10 steps were at most about 1e-3 (compressible
`ρw`) and 7e-5 (anelastic); anything near 1 or `NaN` means a broken kernel.

Full example (about 6.5 min on the A750 including compilation; writes JLD2, mp4 and PNG into
the current directory, which `.gitignore` excludes):

```sh
mkdir -p runs/mine && cd runs/mine
julia +1.12 --project=../.. ../../Breeze.jl/examples/splitting_supercell.jl 2>&1 | tee run.log
```

## Expected result (A750 reference)

Both cores reach 2 h without NaNs. Wall times from the log
(`runs/supercell_a750/wall_time_summary.txt`; the Aurora reference is in
`runs/supercell_aurora/`, see [Aurora](#aurora)):

```
case                wall (s)    wall (min)    iterations
anelastic              179.7          3.00          1404
compressible            45.5          0.76          1328
```

The anelastic time is dominated by the Float64 pressure solver (Breeze's default); with
`pressure_solver_float_type=Float32` it drops to 45.9 s, but tracks a CPU reference less closely.
Fields lose y-symmetry in the second hour (also on the CPU), so the two cores agree only
statistically after about 1 h. Compare your `supercell_max_w_comparison.png` with
[`runs/supercell_a750/supercell_max_w_comparison.png`](runs/supercell_a750/supercell_max_w_comparison.png)
or [`runs/supercell_aurora/supercell_max_w_comparison.png`](runs/supercell_aurora/supercell_max_w_comparison.png).

## Known Intel GPU constraints

- **Kernel arguments: 2048 bytes total** (A750 and PVC). Oceananigans' extension now passes field
  data as 8-byte pointers (largest Breeze kernel: 936 B). The smoke ladder's `KARGS=all` prints
  per-kernel sizes.
- **Do not set `IGC_OverrideOCLMaxParamSize`.** On the A750 and on PVC, bytes past offset 2048
  silently arrive as zero. Probe: `repro/kernel_args/` (`run_all.sh` runs `probe.jl` under
  several settings; `characterize.jl` and `split_args.jl` are single-setting variants).
- **Sub-group size 32 miscompiles emulated Float64** (libigc 2.41.5, NEO 26.35, A750):
  `repro/repro_simd32_fp64.jl` (plain oneAPI.jl, compares sub-group sizes default/8/16/32) and
  `repro/repro_ka_default.jl` (KernelAbstractions default path; prints `CORRECT`/`WRONG`).
  Hence `ms/subgroup-16`. Not seen on PVC with native Float64. Not reported to Intel or upstream
  yet.
- **Arc is Float32-only in practice**: emulated Float64 was unreliable in large kernels. The
  example sets `Oceananigans.defaults.FloatType = Float32`.

```sh
julia +1.12 --project=repro -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project=repro repro/repro_simd32_fp64.jl
julia +1.12 --project=repro repro/repro_ka_default.jl
bash repro/kernel_args/run_all.sh      # writes repro/kernel_args/log_*.txt
```

## Aurora

Tested 2026-11-11 on one Intel Data Center GPU Max 1550 tile (`ZE_AFFINITY_MASK=0`), PBS
`debug` queue, 1 node, `filesystems=flare`.

**Setup.** Only `module load julia/1.12` (Julia 1.12.7); use plain `julia` where this README
says `julia +1.12`. The module sets `ONEAPI_LTS=1`, a site preference environment
(`/soft/libraries/julia/environments/v1.12`, via `JULIA_LOAD_PATH`) that points the JLLs at the
system driver libraries, a persistent NEO kernel cache in the depot, `IGC_DisableSOAPromotion=1`,
`ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `TMPDIR` and the proxy. No `res/local.jl` and no hand-set
preferences. Libraries in use: Level Zero driver v1.3.33578 (NEO 25.18.33578 LTS), loader
1.24.0, IGC 2.34.4 from `/opt/aurora/26.181.0/intel-gpu-umd/2026.06.19`. `oneAPI.versioninfo()`
prints the JLL version numbers (NEO 26.35, libigc 2.41.5), not the libraries in use;
`aurora/probe_env.jl` lists what is actually loaded. The Manifest was reused unchanged
(KernelAbstractions 0.10.7). Do not set the FP64 emulation variables.

A user-depot `environments/v1.12/LocalPreferences.toml` takes precedence over the site one
(`@v#.#` comes first in `JULIA_LOAD_PATH`). A stale one (pointing at a removed
`libopencl-clang.so.15`) made `libigc_jll`, `NEO_jll` and `oneAPI` fail to precompile; removing
it fixed it.

**Running.** The PBS scripts are in `aurora/` (see `aurora/README.md`; the account, repo path and
depot in them are site/user specific). Submit from the repo root, one at a time (the `debug`
queue allows one queued-or-running job per user):

```sh
qsub aurora/probe.pbs       # driver probe, reproducers, kernel-argument probe
qsub aurora/smoke.pbs       # smoke ladder, CPU then oneAPI
qsub aurora/supercell.pbs   # full example in runs/supercell_aurora/
```

**Threads.** Run processes that load oneAPI with `JULIA_NUM_THREADS=1`. With 32 Julia threads
and oneAPI loaded, the smoke ladder segfaulted at the start of rung 2 (job 9077322, reproduced in
9077363), in a `set!` on a CPU-architecture field, i.e. in KernelAbstractions 0.10's CPU backend
(pocl-based, `pocl_standalone_jll`). The CPU-only pass (oneAPI not loaded) is fine with 32
threads. Root cause not established; a core dump suggested a SIGSEGV handler in
`libze_intel_gpu` being entered from Julia's GC safepoint, unconfirmed.

**Results.**

- Probe: Float32 and Float64 broadcast, kernel and `sum` match the CPU; `oneAPI.oneL0.LTS[]` is
  `true`.
- Sub-group size: PVC supports 16 and 32 only, so `repro/repro_simd32_fp64.jl` aborts at 8;
  `aurora/repro_simd32_fp64_pvc.jl` skips unsupported sizes. With native Float64 the SIMD32
  miscompile does not appear: default (32), 16 and 32 all give `max|gpu-cpu| = 0`.
  `repro/repro_ka_default.jl` prints `CORRECT`.
- Kernel arguments: same 2048 B limit (cases up to 1924 B correct, from 2016 B up fail to
  compile). `IGC_OverrideOCLMaxParamSize` is equally unsafe: at 4096 and 8192, cases from 2400 B
  up are `WRONG` with bytes past offset 2048 arriving as zero; at 1024 everything fails to
  compile.
- Smoke ladder (job 9077393; CPU 32 threads, oneAPI 1 thread): all six rungs passed on both.
  oneAPI vs CPU after 10 steps, `max|Δ|/max|ref|`: compressible at most 9.4e-4 (`ρw`) and 5.0e-4
  (`ρv`); anelastic at most 6.6e-5 (`ρw`). Largest kernel: `gpu_compute_z_momentum_tendency!`,
  936 B.
- Full example (job 9077422; unmodified `Breeze.jl/examples/splitting_supercell.jl`, 1 Julia
  thread, warm NEO kernel cache): both cores reach 2 h without NaNs; PBS walltime 6 min 19 s
  including compilation. `runs/supercell_aurora/wall_time_summary.txt`:

| case | Aurora wall (s) | Aurora iterations | A750 wall (s) | A750 iterations |
|---|---|---|---|---|
| anelastic | 44.0 | 1402 | 179.7 | 1404 |
| compressible | 31.9 | 1328 | 45.5 | 1328 |

The anelastic gain is mainly the Float64 pressure solver running natively. The max-w plot
(`runs/supercell_aurora/supercell_max_w_comparison.png`) is visually the same as the A750
reference; the anelastic curve differs slightly only near the end of the second hour,
consistent with the loss of symmetry described above.
