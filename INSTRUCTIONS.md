To get context about the project, check the next files:
- PLAN.md
- README.md

To get context about dynamic model, optimizer, tracks, check:
- docs/DYNAMIC_MODEL.md
- models/
- planning/optimizer.py
- maps/
- maps/create_tracks.py

To check current trajectory optimizer results:
- results/trajectory_optimization
- experiments/ipopt_trajopt_cli.py (use `single` subcommand)

Check best cost = time_cost + reg_cost + slack_cost in optimizer
