# Decision Transformer Warm-Start for Vehicle Trajectory Optimization

## Project Overview

This project investigates whether an offline sequence model (Decision Transformer) can amortize nonlinear constrained trajectory optimization for autonomous vehicles by producing high-quality **warm-start trajectories**.

**Core Idea:** Sequential Convex Programming (SCP) is sensitive to initialization. A learned neural trajectory prior can significantly reduce SCP iterations and failure rates while preserving hard constraint satisfaction through optimization refinement.

```
┌─────────────────────────┐
│  Decision Transformer   │ ── predicts ──▶ Initial trajectory guess
│  (learned from data)    │                        │
└─────────────────────────┘                        ▼
                                    ┌──────────────────────────┐
                                    │   SCP / Trajectory Opt   │
                                    │   (refines to feasible)  │
                                    └──────────────────────────┘
                                                   │
                                                   ▼
                                         Optimal, feasible trajectory
```

---

## Vehicle Dynamics Model

### Single-Track (Bicycle) Model

Based on [Aggarwal & Gerdes, IEEE OJCS 2025](https://doi.org/10.1109/OJCSYS.2025.3635449): *"Friction-Robust Autonomous Racing Using Trajectory Optimization Over Multiple Models"*

The model captures:
- **3 DOF dynamics:** longitudinal velocity (u_x), lateral velocity (u_y), yaw rate (r)
- **Weight transfer:** longitudinal (ΔF_z,long) and lateral (ΔF_z,lat)
- **Road geometry:** grade (θ) and bank (φ) effects
- **Fiala brush tire model:** load-dependent cornering stiffness, combined slip

### State Vector (8 states)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | u_x | Longitudinal velocity | m/s |
| 1 | u_y | Lateral velocity | m/s |
| 2 | r | Yaw rate | rad/s |
| 3 | ΔF_z,long | Longitudinal weight transfer | kN |
| 4 | ΔF_z,lat | Lateral weight transfer | kN |
| 5 | t | Time | s |
| 6 | e | Lateral deviation from path | m |
| 7 | Δψ | Heading error | rad |

### Control Vector (2 inputs)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | δ | Steering angle | rad |
| 1 | F_x | Total longitudinal force | kN |

---

## Trajectory Optimization

### Problem Formulation

**Objective:** Minimize lap time
```
min  J = ∫₀^s_final (1/ṡ) ds = ∫₀^s_final dt
```

**Subject to:**
- Vehicle dynamics (collocation constraints)
- Track boundaries: `|e| ≤ track_width/2 - buffer`
- Actuator limits: `|δ| ≤ δ_max`, `F_x,min ≤ F_x ≤ F_x,max`
- Friction limits (via tire model)
- Speed bounds: `u_x ≥ u_x,min`

### Implemented Solvers

#### 1. Direct Collocation + IPOPT
```
Continuous OCP  ──transcribe──▶  NLP  ──IPOPT──▶  Solution
                (trapezoidal)
```
- **Discretization:** Trapezoidal collocation (N nodes)
- **NLP Solver:** IPOPT (interior point method)
- **Use case:** Baseline solver, generates training data

#### 2. Sequential Convex Programming (SCP)
```
Initial guess x⁰
repeat:
    1. Linearize dynamics around xᵏ
    2. Solve convex QP subproblem
    3. Update: xᵏ⁺¹ with trust region
until convergence
```
- **Linearization:** Jacobians computed via CasADi AD
- **Trust region:** Adaptive radius with expand/shrink logic
- **Iteration tracking:** Key metric for warm-start evaluation

**SCP Advantages for This Project:**
- Iteration count directly measures warm-start quality
- Convex subproblems are fast to solve
- Natural trust region framework

---

## Demo Results

Running `python run_scp_demo.py` produces:

| Method | Iterations | Time | Cost (Lap Time) | Success |
|--------|------------|------|-----------------|---------|
| Direct Collocation | 34 | 8.34s | 14.46s | Yes |
| SCP (Cold Start) | 9 | 12.73s | 26.00s | No (failed) |
| SCP (Warm Start) | 2 | 4.69s | 13.92s | Yes |

**Key Finding:** Warm-start reduces SCP iterations by **4.5x** and solves successfully where cold start fails. This validates the potential for Decision Transformer warm-starts.

---

## Code Structure

```
CS234_Final_Project/
│
├── models/
│   ├── __init__.py           # Exports: SingleTrackModel, VehicleParams, FialaBrushTire
│   ├── vehicle.py            # Unified vehicle dynamics (paper-compliant)
│   └── tire.py               # Fiala brush tire model
│
├── planning/
│   ├── __init__.py
│   ├── optimizer.py          # Direct collocation + IPOPT
│   └── scp_solver.py         # Sequential Convex Programming solver
│
├── world/
│   └── world.py              # Track geometry and boundaries
│
├── utils/
│   ├── plotting_utils.py     # Legacy plotting (with display)
│   └── visualization.py      # New visualization (saves to files)
│
├── maps/
│   └── Medium_Oval_Map_260m.mat   # Example track
│
├── results/                  # Output visualizations
│   ├── dc_trajectory.png
│   ├── scp_cold_*.png
│   ├── scp_warm_*.png
│   ├── method_comparison.png
│   ├── warm_start_analysis.png
│   └── trajectory_animation.gif
│
├── run_scp_demo.py           # Main demo script
├── test_unified_model.py     # Model verification script
├── create_oval_track.py      # Track generation script
└── README.md                 # This file
```

---

## Usage

### Setup Environment

```bash
# Create conda environment
conda create -n DT_trajopt python=3.11 -y
conda activate DT_trajopt

# Install dependencies
pip install numpy scipy matplotlib casadi PyYAML
```

### Run Demo

```bash
cd CS234_Final_Project
python run_scp_demo.py
```

This will:
1. Solve trajectory optimization with Direct Collocation
2. Solve with SCP (cold start)
3. Solve with SCP (warm start from DC result)
4. Generate comparison visualizations in `results/`

### Basic Usage

```python
from models import SingleTrackModel, VehicleParams, FialaBrushTire
from planning import TrajectoryOptimizer, SCPSolver, SCPParams
from world.world import World

# Load track
world = World("maps/Medium_Oval_Map_260m.mat", "Oval", diagnostic_plotting=False)

# Create vehicle
params = VehicleParams(...)  # See run_scp_demo.py for example
f_tire = FialaBrushTire(c0_alpha_nprad=0, c1_alpha_1prad=8.0, mu_none=0.9)
r_tire = FialaBrushTire(c0_alpha_nprad=0, c1_alpha_1prad=13.0, mu_none=0.9)
vehicle = SingleTrackModel(params, f_tire, r_tire)

# Direct Collocation
dc_opt = TrajectoryOptimizer(vehicle, world)
dc_result = dc_opt.solve(N=100, ds_m=2.6)

# SCP with warm-start
scp_opt = SCPSolver(vehicle, world, params=SCPParams(max_iterations=30))
scp_result = scp_opt.solve(N=100, ds_m=2.6, X_init=dc_result.X, U_init=dc_result.U)

print(f"SCP iterations: {scp_result.iterations}")
```

---

## References

1. **Vehicle Model:** Aggarwal & Gerdes, "Friction-Robust Autonomous Racing Using Trajectory Optimization Over Multiple Models", IEEE Open Journal of Control Systems, 2025.

2. **Decision Transformer:** Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling", NeurIPS 2021.

3. **Trajectory Transformer:** Janner et al., "Offline Reinforcement Learning as One Big Sequence Modeling Problem", NeurIPS 2021.

4. **Transformer Warm-Starts:** Guffanti et al., "Transformers for Trajectory Optimization with Application to Spacecraft Rendezvous", IEEE Aerospace 2024.

5. **SCP for Trajectory Optimization:** Mao et al., "Successive Convexification of Non-Convex Optimal Control Problems", 2016.

---

## TODO

- [x] Implement unified vehicle model
- [x] Implement direct collocation optimizer
- [x] Implement SCP solver with trust regions
- [x] Create visualization tools
- [ ] Dataset generation pipeline
- [ ] Decision Transformer training script
- [ ] Warm-start integration with SCP
- [ ] Evaluation benchmarks
- [ ] Multi-track generalization
