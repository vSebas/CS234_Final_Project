# IPOPT TrajOpt Progress

## Scope
This file tracks the production IPOPT trajectory-optimization path status.

## Current status
- Production solver remains IPOPT via [`planning/optimizer.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/planning/optimizer.py).
- Demo entrypoint is [`run_trajopt_demo.py`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/run_trajopt_demo.py).
- Dataset generation also uses IPOPT for base laps and repair solves.

## Baseline configuration
- Reference config for current project runs:
  - `N=120`
  - `lambda_u=0.005`
  - `ux_min=3.0` (demo) / dataset-specific values in dataset scripts
  - hard obstacle constraints in production path
- Full configuration details are documented in [`docs/OPTIMIZER_CONFIG.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/OPTIMIZER_CONFIG.md).

## Role relative to FATROP and MadNLP
- IPOPT is the quality/reference baseline.
- FATROP and MadNLP are experimental solver tracks and are not the production default.
- Solver comparisons should be interpreted against IPOPT objective quality, feasibility, and solve-time behavior on matched configs.

## Related docs
- FATROP: [`docs/FATROP_TRAJOPT_PROGRESS.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/FATROP_TRAJOPT_PROGRESS.md)
- FATROP config: [`docs/FATROP_CONFIG.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/FATROP_CONFIG.md)
- MadNLP: [`docs/MADNLP_TRAJOPT_PROGRESS.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/MADNLP_TRAJOPT_PROGRESS.md)
- MadNLP config: [`docs/MADNLP_CONFIG.md`](/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/CS234_Final_Project/docs/MADNLP_CONFIG.md)
