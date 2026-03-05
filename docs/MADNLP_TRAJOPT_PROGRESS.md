# MadNLP TrajOpt Progress (vs IPOPT)

## Scope
This document tracks the current state of the MadNLP backend integration for trajectory optimization and compares observed behavior against the existing CasADi+IPOPT baseline.

## Current implementation status
- MadNLP runs through standalone script path in [`run_madnlp_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/run_madnlp_trajopt.py).
- Core IPOPT optimizer path in [`planning/optimizer.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/planning/optimizer.py) remains backend-independent.
- Julia bridge is implemented in [`planning/madnlp_bridge.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/planning/madnlp_bridge.py).
- Julia NLP script is implemented in [`planning/julia/madnlp_exa_solver.jl`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/planning/julia/madnlp_exa_solver.jl).
- Oval comparison smoke is in [`experiments/test_madnlp_oval.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/test_madnlp_oval.py).

Important detail:
- The current Julia model uses `JuMP + MadNLP.Optimizer` for solve stability.
- `ExaModels.MadNLPOptimizer` was attempted earlier but was timing out in `optimize!` for this setup.

## Features currently in MadNLP path
- Dynamics mode switch via `MADNLP_DYNAMICS_MODE`:
  - `simple` (most stable currently)
  - `full` (more unstable in current form)
  - `fiala` (higher fidelity attempt)
- Auto fallback via `MADNLP_MODE_FALLBACK=1`:
  - if requested mode fails, retries `simple`.
- Obstacle node constraints (node-level, no midpoint subsampling yet).
- Solver caps and tolerances:
  - `MADNLP_MAX_ITER`
  - `MADNLP_MAX_CPU_TIME`
  - `MADNLP_TOL`
  - `MADNLP_ACCEPTABLE_TOL`
- Optional bridge streaming/debug:
  - `MADNLP_EXA_STREAM=1`
  - `MADNLP_EXA_DEBUG=1`

## Environment assumptions
- Julia binary: `/home/saveas/.conda/envs/DT_trajopt/bin/julia`
- Julia project used by bridge: `/home/saveas/.conda/envs/DT_trajopt/share/julia/environments/DT_trajopt`

## Latest smoke comparisons

### 1) Oval (no explicit obstacles), `N=6`
Command pattern:
```bash
PYTHONPATH=. MADNLP_EXA_TIMEOUT_S=120 MADNLP_MAX_ITER=120 MADNLP_MAX_CPU_TIME=20 \
MADNLP_TOL=1e-4 MADNLP_ACCEPTABLE_TOL=1e-3 MADNLP_DYNAMICS_MODE=simple \
JULIA_BIN=/home/saveas/.conda/envs/DT_trajopt/bin/julia \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/test_madnlp_oval.py --N 6
```

Observed:
- IPOPT: `success=True`, `iterations=56`, `cost=14.795086`, `solve_time=0.918s`
- MadNLP: `success=True`, `iterations=28`, `cost=9.193358`, `solve_time=10.667s`
- Trajectory delta: `max|ΔX|=1.282614e+01`, `max|ΔU|=1.147396e+01`

### 2) Oval obstacle map, `N=6`, `simple`
Command pattern:
```bash
PYTHONPATH=. MADNLP_EXA_TIMEOUT_S=180 MADNLP_MAX_ITER=120 MADNLP_MAX_CPU_TIME=20 \
MADNLP_TOL=1e-4 MADNLP_ACCEPTABLE_TOL=1e-3 MADNLP_DYNAMICS_MODE=simple \
JULIA_BIN=/home/saveas/.conda/envs/DT_trajopt/bin/julia \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/test_madnlp_oval.py \
  --map-file maps/Oval_Track_260m_Obstacles.mat --N 6
```

Observed:
- IPOPT: `success=True`, `iterations=32`, `cost=18.774642`, `solve_time=0.900s`
- MadNLP: `success=True`, `iterations=12`, `cost=6.668829`, `solve_time=10.581s`
- Trajectory delta: `max|ΔX|=2.747878e+01`, `max|ΔU|=1.051010e+01`

### 3) Oval obstacle map, requested `fiala` with fallback enabled
Command pattern:
```bash
PYTHONPATH=. MADNLP_EXA_TIMEOUT_S=180 MADNLP_MAX_ITER=200 MADNLP_MAX_CPU_TIME=30 \
MADNLP_TOL=1e-5 MADNLP_ACCEPTABLE_TOL=1e-3 MADNLP_DYNAMICS_MODE=fiala \
JULIA_BIN=/home/saveas/.conda/envs/DT_trajopt/bin/julia \
/home/saveas/.conda/envs/DT_trajopt/bin/python experiments/test_madnlp_oval.py \
  --map-file maps/Oval_Track_260m_Obstacles.mat --N 6
```

Observed:
- IPOPT: `success=True`, `iterations=32`, `cost=18.774642`, `solve_time=0.828s`
- MadNLP: `success=True` (via fallback path), `iterations=13`, `cost=6.668725`, `solve_time=10.459s`
- Trajectory delta: `max|ΔX|=2.747939e+01`, `max|ΔU|=1.051042e+01`

## Interpretation
- Integration status: working (backend runs, returns trajectories, supports obstacle constraints and mode controls).
- Performance status: not parity with IPOPT yet.
  - MadNLP solve time is currently much slower in these smokes.
  - Cost/trajectory differences are large, indicating formulation/options mismatch remains.
- Stability status:
  - `simple` is currently the most reliable mode.
  - `full` / `fiala` need additional tuning and tighter formulation alignment.

## Known gaps to parity
- Midpoint/subsample obstacle constraints (currently node-only).
- Full-fidelity dynamic equivalence to Python IPOPT formulation.
- Option sweep for MadNLP linear-system/KKT settings suitable for this OCP.
- Controlled benchmark over fixed seeds/scenarios at the same `N`, same constraints, and same init.

## Next recommended steps
1. Lock one reference test (`Oval_Track_260m`, fixed `N`, no obstacles), and match IPOPT constraints term-by-term.
2. Add midpoint obstacle constraints in Julia to match Python behavior.
3. Run small grid over `MADNLP_MAX_ITER`, `MADNLP_TOL`, `MADNLP_ACCEPTABLE_TOL`, and linear-solver settings.
4. Compare objective + constraint residuals, not just solve success/time.
5. Only after parity on oval/no-obstacle, expand to obstacle maps and other tracks.
