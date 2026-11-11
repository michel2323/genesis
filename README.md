# genesis: Breeze supercell on Intel GPUs

This workspace runs [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl)'s
`examples/splitting_supercell.jl` (anelastic and compressible cores, 168×168×40, 2 h) on Intel
GPUs through [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl), on the KernelAbstractions 0.10
stack. It was brought up on an Intel Arc A750; the target is Aurora (ALCF), where it has **not
been tested yet**.

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
- `runs/`: run directories. Committed: the A750 reference (`runs/supercell_a750/`) and two
  modified copies of the example (Float32 pressure solver; CPU 1-hour reference).

## Quick start

Needs Julia 1.12 (the commands use `juliaup`'s `julia +1.12`; drop `+1.12` if your `julia` is 1.12).

```sh
git clone --recurse-submodules https://github.com/michel2323/genesis.git
cd genesis
julia +1.12 --project=. -e 'using Pkg; Pkg.instantiate()'
```

Precompiling prints a segfault from AcceleratedKernels' oneAPI extension (`✗ AcceleratedKernels
→ oneAPIExt`); it is skipped and not needed.

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
(`runs/supercell_a750/wall_time_summary.txt`):

```
case                wall (s)    wall (min)    iterations
anelastic              179.7          3.00          1404
compressible            45.5          0.76          1328
```

The anelastic time is dominated by the Float64 pressure solver (Breeze's default); with
`pressure_solver_float_type=Float32` it drops to 45.9 s, but tracks a CPU reference less closely.
Fields lose y-symmetry in the second hour (also on the CPU), so the two cores agree only
statistically after about 1 h. Compare your `supercell_max_w_comparison.png` with
[`runs/supercell_a750/supercell_max_w_comparison.png`](runs/supercell_a750/supercell_max_w_comparison.png).

## Known Intel GPU constraints

- **Kernel arguments: 2048 bytes total.** Oceananigans' extension now passes field data as
  8-byte pointers (largest Breeze kernel: 936 B). The smoke ladder's `KARGS=all` prints
  per-kernel sizes.
- **Do not set `IGC_OverrideOCLMaxParamSize`.** On the A750, bytes past offset 2048 silently
  arrive as zero. Probe: `repro/kernel_args/` (`run_all.sh` runs `probe.jl` under several
  settings; `characterize.jl` and `split_args.jl` are single-setting variants).
- **Sub-group size 32 miscompiles emulated Float64** (libigc 2.41.5, NEO 26.35, A750):
  `repro/repro_simd32_fp64.jl` (plain oneAPI.jl, compares sub-group sizes default/8/16/32) and
  `repro/repro_ka_default.jl` (KernelAbstractions default path; prints `CORRECT`/`WRONG`).
  Hence `ms/subgroup-16`. Not reported to Intel or upstream yet.
- **Arc is Float32-only in practice**: emulated Float64 was unreliable in large kernels. The
  example sets `Oceananigans.defaults.FloatType = Float32`.

```sh
julia +1.12 --project=repro -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project=repro repro/repro_simd32_fp64.jl
julia +1.12 --project=repro repro/repro_ka_default.jl
bash repro/kernel_args/run_all.sh      # writes repro/kernel_args/log_*.txt
```

## Aurora: not yet tested

Nothing below has been checked on Aurora. Things to establish first:

1. **Driver stack.** By default oneAPI.jl uses the NEO/IGC/Level Zero loader shipped as JLL
   artifacts (`NEO_jll`, `libigc_jll`, `oneAPI_Level_Zero_Loader_jll`), a rolling release.
   To use the system stack, oneAPI.jl documents (`oneAPI.jl/docs/src/installation.md`, "Using
   System Libraries"; `oneAPI.jl/README.md`, "Using a local toolchain") running
   `julia +1.12 --project=. -e 'using oneAPI; include(joinpath(pkgdir(oneAPI), "res", "local.jl"))'`,
   which finds the system libraries and writes `*_path` preferences into the global
   `LocalPreferences.toml`, or setting those preferences by hand. Aurora runs Intel's LTS
   driver branch; for that oneAPI.jl has the opt-in `export ONEAPI_LTS=1`
   (`oneAPI.jl/docs/src/lts.md`), which switches SPIR-V generation to the Khronos translator
   and enables LTS workarounds, but does not select the driver by itself. Open questions:
   does the bundled rolling NEO work on Aurora's kernel driver at all, or must it be the
   system stack plus `ONEAPI_LTS=1`? Check with `oneAPI.versioninfo()`.
2. **Run the reproducers** (above). Does the sub-group-32 miscompile exist on PVC with native
   Float64 (try `repro_simd32_fp64.jl`)? Does `IGC_OverrideOCLMaxParamSize` behave the same
   (`repro/kernel_args/`)? Never set it for real runs either way.
3. **FP64 emulation variables** (`OverrideDefaultFP64Settings`, `IGC_EnableDPEmulation`) are
   for the A750 only; do not set them on hardware with native Float64.
4. **Manifest.** It was resolved on Linux x86-64 with Julia 1.12.7. `Pkg.instantiate()` should
   reuse it unchanged on another x86-64 Linux with Julia 1.12; if it re-resolves, check that
   KernelAbstractions is still 0.10.x and the three packages still come from the submodules
   (`julia +1.12 --project=. -e 'using Pkg; Pkg.status()'`).
5. **Smoke ladder, then the full example**, comparing rung 6 against the CPU before trusting
   anything. How to get a GPU node (modules, queues, project allocation, where to put the Julia
   depot) is not covered here.
