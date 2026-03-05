# MadNLP Configuration

## Scope
This file lists MadNLP/Julia bridge runtime knobs used by:
- [`experiments/run_madnlp_trajopt.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/experiments/run_madnlp_trajopt.py)
- [`planning/madnlp_bridge.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/planning/madnlp_bridge.py)
- [`planning/julia/madnlp_exa_solver.jl`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/planning/julia/madnlp_exa_solver.jl)

## Julia bridge variables
- `JULIA_BIN`
- `MADNLP_EXA_SCRIPT`
- `MADNLP_EXA_PROJECT`
- `MADNLP_EXA_JULIA_FLAGS`
- `MADNLP_EXA_TIMEOUT_S`
- `MADNLP_EXA_STREAM`
- `MADNLP_EXA_DEBUG`

## Solver variables passed to Julia
- `MADNLP_TOL`
- `MADNLP_ACCEPTABLE_TOL`
- `MADNLP_MAX_ITER`
- `MADNLP_MAX_CPU_TIME`
- `MADNLP_LINEAR_SOLVER`
- `MADNLP_KKT_SYSTEM`
- `MADNLP_HSLIB`
- `MADNLP_PERIODIC_CONTROLS`
- `MADNLP_DYNAMICS_MODE` (`simple`, `full`, `fiala`)

## GPU behavior variables
- `MADNLP_REQUIRE_GPU`
- `MADNLP_GPU_FALLBACK`

## Current practical note
- MadNLP path is integrated and executable, but not at IPOPT parity in current measured runs.
- Use IPOPT for production trajectory generation; treat MadNLP as experimental until objective/constraint parity and solve-time behavior are validated.

## Related docs
- Progress log: [`docs/MADNLP_TRAJOPT_PROGRESS.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/MADNLP_TRAJOPT_PROGRESS.md)
- IPOPT config baseline: [`docs/OPTIMIZER_CONFIG.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/OPTIMIZER_CONFIG.md)
