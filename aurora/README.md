# Aurora job scripts

PBS scripts for one Intel Data Center GPU Max 1550 tile on Aurora (ALCF). Results and setup notes
are in the Aurora section of the top-level `README.md`.

| File | What it does |
|---|---|
| `env.sh` | Sourced by every job: `module load julia/1.12`, depot, `ZE_AFFINITY_MASK=0`, prints the environment |
| `probe.pbs` | Driver/library probe, sub-group and `repro_ka_default.jl` reproducers, kernel-argument probe (`qsub -v KARGS_PROBE=0` skips it) |
| `probe_env.jl` | `oneAPI.versioninfo()`, configured and loaded driver libraries, Float32/Float64 kernels vs CPU |
| `repro_simd32_fp64_pvc.jl` | `repro/repro_simd32_fp64.jl`, skipping sub-group sizes PVC does not support (8) |
| `smoke.pbs` | Smoke ladder, CPU (32 threads) then oneAPI (1 thread), `OUTDIR=scripts/output_aurora` |
| `supercell.pbs` | Unmodified `Breeze.jl/examples/splitting_supercell.jl` in `runs/supercell_aurora/` |
| `diag.pbs`, `diag_rung2.jl` | Step-by-step replay of smoke rung 2, used to locate the multi-thread segfault |

Job output goes to `aurora/logs/` (ignored). Submit from the repo root, in order, one at a time
(the `debug` queue allows one queued-or-running job per user):

```sh
qsub aurora/probe.pbs
qsub aurora/smoke.pbs
qsub aurora/supercell.pbs
```

For another user or project, edit `#PBS -A` and the `#PBS -o` path in the `.pbs` files, `REPO` in
the `.pbs` files, and `JULIA_DEPOT_PATH` in `env.sh`. The environment must be instantiated in that
depot first (`julia --project=. -e 'using Pkg; Pkg.instantiate()'` with `env.sh` sourced).

Run every process that loads oneAPI with `JULIA_NUM_THREADS=1`; with 32 threads the smoke ladder
segfaulted (see the top-level `README.md`). The CPU-only pass is fine with 32 threads.
`probe.pbs` and `diag.pbs` keep `env.sh`'s 32 threads (the probe passed that way; `diag.pbs`
reproduces the crash).
