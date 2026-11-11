# genesis workspace

Goal: run Breeze's `examples/splitting_supercell.jl` on Intel GPUs through oneAPI.jl. Development
happened on a workstation with an Arc A750 (8 GB); the target is Aurora at ALCF. Each submodule
has its own CLAUDE.md/AGENTS.md, which applies inside it. `README.md` is the setup guide.

## Layout and branches

The root repo is `michel2323/genesis`. The three packages are git submodules; `.gitmodules`
points at the public HTTPS fork URLs and records the branch. The branches are pushed to the
forks (`michel2323/<name>`); nothing is pushed upstream and no PR was changed.

| Path | Branch | What it is |
|---|---|---|
| `Breeze.jl/` | `ms/oneapi` | `main` + KA 0.10 compat, example switched to oneAPI, ports for two Oceananigans API changes (#6440, #6443) |
| `Oceananigans.jl/` | `ms/oneAPI-ka0.10` | KA 0.10 draft (#5799) merged with `main`, plus the oneAPI extension (PR #4764) and two extension fixes |
| `oneAPI.jl/` | `ms/subgroup-16` | `main` (v2.10.0) + KA kernels compiled at sub-group size 16 |
| `Project.toml`, `Manifest.toml` | | Julia environment (resolved with Julia 1.12.7); `[sources]` point at the submodules |
| `scripts/smoke_supercell.jl` | | 32×32×16 bring-up ladder; logs go to `scripts/logs/` (ignored) |
| `repro/` | | Intel compiler/driver reproducers, own environment (`repro/Project.toml`) |
| `runs/` | | Run directories; only the modified example scripts and the A750 reference result are committed |

In the original workstation checkouts, each submodule's `origin` is the upstream repo and `fork`
is `michel2323/<name>` (SSH); push only to `fork`. In a fresh clone, `origin` is the fork.
Breeze `ms/oneapi` does not work with registry Oceananigans 0.117.2; it needs the submodule.

## Running

Use Julia 1.12 (`julia +1.12`); on the workstation the default `julia` is 1.13.

```sh
# from the repo root
julia +1.12 --project=. -e 'using Pkg; Pkg.instantiate()'
# Smoke ladder: ARCH=cpu|oneapi, RUNG=1..6|all, NSTEPS, OUTDIR, KARGS=onfail|report|all|skip
ARCH=oneapi RUNG=all OUTDIR=scripts/output_x julia +1.12 --project=. scripts/smoke_supercell.jl
# Full example (writes into the current directory)
mkdir -p runs/<dir> && cd runs/<dir> && julia +1.12 --project=../.. ../../Breeze.jl/examples/splitting_supercell.jl
# Oceananigans oneAPI tests (do not set TEST_ARCHITECTURE; it selects CUDA)
cd Oceananigans.jl && TEST_GROUP=oneapi julia +1.12 --project -e 'using Pkg; Pkg.test()'
# Reproducers
julia +1.12 --project=repro -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project=repro repro/repro_simd32_fp64.jl
```

Rung 6 of the ladder saves final fields per architecture so CPU and oneAPI can be compared (run
CPU first, then oneAPI, with the same `OUTDIR`). `KARGS=all` prints the byte size of every
kernel's arguments.

## Intel GPU constraints found so far

- **Kernel arguments are limited to 2048 bytes in total** (Arc and PVC). Breeze's tendency
  kernels needed up to 2816 B. Fixed in the Oceananigans extension by passing each field's data
  as an 8-byte pointer-only device array (`801adb90a`); the largest kernel is now 936 B.
- **Never set `IGC_OverrideOCLMaxParamSize`.** Tested here: bytes at offset ≥ 2048 inside a single
  argument silently arrive as zero (`repro/kernel_args/`). Rerun that probe on Aurora before
  trusting anything there.
- **Sub-group size 32 miscompiles emulated Float64.** Intel's compiler (libigc 2.41.5, NEO 26.35)
  gives wrong results for kernels with Float64 arithmetic at SIMD32 on the A750
  (`repro/repro_simd32_fp64.jl`, `repro/repro_ka_default.jl`). oneAPI.jl's KA backend asked
  for 32; `ms/subgroup-16` uses 16. Not reported to Intel or upstream yet.
- **The A750 is Float32-only in practice.** Float64 is emulated (`OverrideDefaultFP64Settings=1`,
  `IGC_EnableDPEmulation=1` exported in the shell) and was still unreliable in large kernels
  at sub-group sizes 16 and 8. Pass Float32 schemes explicitly in Oceananigans
  (`WENO(Float32; ...)`) or set `Oceananigans.defaults.FloatType = Float32`.
- **Breeze's anelastic pressure solver is Float64 by default** even in Float32 models
  (`AnelasticDynamics(...; pressure_solver_float_type=Float64)`). It is the only Float64 on the
  example's GPU path, and makes the full anelastic run about 4× slower here than `Float32`
  (180 s vs 46 s). It tracked a 1-hour CPU reference more closely than the Float32 solver did.
- **Full example on the A750 (2026-11-11):** both cores reach 2 h without NaNs in about 6.5 min
  total; reference in `runs/supercell_a750/` (max-w PNG and wall-time summary). Fields lose
  y-symmetry in the second hour (also on the CPU), so the two cores agree only statistically
  after about 1 h.
- **Array tracking:** the extension must not pre-convert kernel arguments
  (`convert_to_device`); oneAPI.jl converts at launch and tracks array ownership there
  (`bac7d5f6d`).
- On every `using oneAPI`, AcceleratedKernels' oneAPI extension segfaults in the driver while
  precompiling and is skipped. Runs work without it; the bug is not investigated.
- Not tested on Aurora: whether PVC (native Float64, LTS driver stack) shows the SIMD32
  miscompile or the override behaviour. See the Aurora section of `README.md`.

## Working rules

- The main session orchestrates; hands-on work goes to Opus 5.5 medium-effort subagents.
- Fix a bug in the repo that owns it, one concern per commit, on the branches above; commit in
  the submodule, push it to the fork, then commit the new submodule pointer in the root.
- Never push upstream, force-push or open/modify PRs unless asked.
- The GPU is shared with other sessions; don't trust timings taken while another job runs.
- Compare against the CPU before believing a GPU result: passing tests here mostly check that
  models step, not their values.
