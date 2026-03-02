# Decision Transformer Warm-Start for Vehicle Trajectory Optimization

## Project Overview

This project investigates whether an offline sequence model (Decision Transformer) can amortize nonlinear constrained trajectory optimization for autonomous vehicles by producing high-quality **warm-start trajectories**.

**Core Idea:** Use a robust direct-transcription NLP solved by IPOPT as the production optimizer (including future obstacle-avoidance constraints), and use a learned neural prior to improve warm-start quality and reduce solve time/iterations.

```
┌─────────────────────────┐
│  Decision Transformer   │ ── predicts ──▶ Initial trajectory guess
│  (learned from data)    │                        │
└─────────────────────────┘                        ▼
                                    ┌──────────────────────────┐
                                    │ IPOPT Trajectory NLP     │
                                    │ (collocation + constraints) │
                                    └──────────────────────────┘
                                                   │
                                                   ▼
                                         Optimal, feasible trajectory
```

---

## Vehicle Dynamics Model

### Single-Track (Bicycle) Model

Based on [Aggarwal & Gerdes, IEEE OJCS 2025](https://doi.org/10.1109/OJCSYS.2025.3635449): *"Friction-Robust Autonomous Racing Using Trajectory Optimization Over Multiple Models"*
Also from:
- https://github.com/dynamicdesignlab/models
- https://github.com/dynamicdesignlab/multimodel-trajectory-optimization

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

#### 1. Production Solver: Direct Collocation + IPOPT
```
Continuous OCP  ──transcribe──▶  NLP  ──IPOPT──▶  Solution
                (trapezoidal)
```
- **Discretization:** Trapezoidal collocation (N nodes)
- **NLP Solver:** IPOPT (interior point method)
- **Use case:** Main solver for data generation and refinement
- **Obstacle handling:** Static circular obstacles with hard clearance constraints (optional slack for diagnosis)
- **Coordinate rule (critical):** Frenet-to-ENU conversion must be consistent across optimizer and plotting:
  - `E = E_cl - e*sin(psi)`
  - `N = N_cl + e*cos(psi)`

#### 2. SCP Status
- `planning/scp_solver.py` is frozen for regular development.
- SCP remains an experimental/archival branch.
- See:
  - `docs/SCP_TRAJECTORY_OPTIMIZER_STATUS.md`
  - `docs/SCP_EXPLANATION.md`
  - `docs/scp_archive/`

---

## Optimization Status

- Current robust path: IPOPT direct collocation.
- Current project direction: obstacle-aware IPOPT formulation and DT warm-starting of the IPOPT solver.
- Current demo/production configuration solves a **full lap** with periodic boundary conditions.
- Obstacle overlap visualization bug was fixed by unifying `world.map_match_vectorized` with optimizer Frenet-to-ENU convention.
- SCP is not the active production path.
- Detailed plan: `PLAN.md`.

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
│   ├── optimizer.py          # Direct collocation + IPOPT (production)
│   └── scp_solver.py         # SCP solver (frozen experimental)
│
├── world/
│   └── world.py              # Track geometry and boundaries
│
├── utils/
│   └── visualization.py      # Visualization (saves to files)
│
├── maps/
│   └── Oval_Track_260m.mat        # Example track
│
├── results/                  # Output visualizations
│   └── trajectory_optimization/
│       └── nlp/
│           ├── ipopt_single_stage_trajectory.png
│           ├── ipopt_single_stage_states.png
│           ├── ipopt_single_stage_controls.png
│           └── ipopt_single_stage_animation.gif
│
├── run_trajopt_demo.py       # Trajectory optimization demo (IPOPT production path)
├── simulate_vehicle.py       # Vehicle dynamics simulation & visualization
├── test_dynamic_model.py     # Model verification script
├── create_tracks.py          # Unified track generation script
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

# Optional: faster QP solver (may have issues on ill-conditioned problems)
# pip install osqp
```

### Run Demo

```bash
cd CS234_Final_Project
python run_trajopt_demo.py
```

This will:
1. Solve trajectory optimization with IPOPT direct collocation
2. Run current comparison pipeline and save plots/logs in `results/trajectory_optimization/nlp`

Notes:
- Production solver path is IPOPT.
- The default demo configuration is full-lap (`ds = track_length / N`, periodic closure).
- SCP outputs are kept for reference/diagnostics.

### Acceptance Policy + Batch Evaluation

The demo now uses acceptance-gated retries by default.

Default acceptance gates:
- `success == True`
- `max_obstacle_slack <= 0`
- `min_obstacle_clearance >= -0.001` m (practical epsilon for dense post-check)

Run demo with defaults:
```bash
python run_trajopt_demo.py
```

Run strict acceptance (no negative dense clearance allowed):
```bash
ACCEPT_MIN_CLEARANCE_M=0.0 python run_trajopt_demo.py
```

Batch evaluate randomized obstacle scenarios:
```bash
python run_trajopt_batch_eval.py --num-scenarios 20 --seed 42
```

Batch outputs are written to `results/trajectory_optimization/`:
- `trajopt_batch_eval_<timestamp>.json` (summary)
- `trajopt_batch_eval_<timestamp>.csv` (per-scenario details)

### Dataset Generation

Quick Stage A (no obstacles, periodic lap + circular shifts):

```bash
python data/generate_dataset.py --num-episodes 1000
```

Full Fix A + Fix B pipeline (base laps + shifts + repair segments):
```bash
python data/build_dataset.py \
  --map-files maps/Oval_Track_260m.mat,maps/TRACK1_280m.mat,maps/TRACK2.mat,maps/TRACK3_300m.mat,maps/TRACK4_315m.mat,maps/TRACK5_330m.mat \
  --base-laps 6 \
  --obstacle-laps 8 \
  --shift-episodes 1000 \
  --repair-segments 1000 \
  --N 200 \
  --H 20 \
  --seed 0
```

To generate all circular shifts per base lap, pass `--all-shifts` to `data/make_shift_episodes.py`. This produces `N+1` shifts (k0=0..N), including one duplicate due to periodic closure.

The dataset config and defaults are captured in `data/DATASET_CONFIG.md`. The full generation plan (Fix A/Fix B) is documented in `PLAN.md`.

Current saved episode schema:
- node-aligned arrays: `s_m`, `X_full`, `U`, `pos_E`, `pos_N`, `yaw_world`, `kappa`, `half_width`, `grade`, `bank`
- transition-aligned arrays: `dt`, `reward`, `rtg`
- header fields are recorded in `manifest.jsonl` alongside `map_hash`, `solver_config`, `solver_config_hash`, `base_id`, and `s_offset_m`

The current DT code in `dt/dataset.py` consumes this saved schema directly. Treat the saved field names as the source of truth for training and warm-start work.

### Decision Transformer Training

Train on a single dataset shard:

```bash
python dt/train.py \
  --data-dir data/datasets/Oval_Track_260m_shifts \
  --output-dir dt/checkpoints/oval_shifts_run1 \
  --context-length 30 \
  --batch-size 64 \
  --num-epochs 100
```

Evaluate a checkpoint on a single dataset shard:

```bash
python dt/eval.py \
  --checkpoint dt/checkpoints/oval_shifts_run1/checkpoint_best.pt \
  --data-dir data/datasets/Oval_Track_260m_shifts
```

Current training limitations:
- `dt/train.py` currently expects one `--data-dir` at a time, not the full multi-shard dataset root
- train/validation splitting is done inside the loader and still needs explicit split-by-`base_id` hygiene
- normalization statistics are currently computed before the train/validation split and should be moved to train-only stats

Recommended current usage:
- use the generated shard structure as-is
- treat `data/DATASET_CONFIG.md` as the canonical description of the on-disk dataset schema
- update the training/data-loading path before claiming final offline-RL or DT benchmark results on the full dataset

### Vehicle Simulation

Simulate the vehicle dynamics with custom throttle and steering inputs (no track required):

```bash
python simulate_vehicle.py --scenario constant_turn --duration 5
```

**Predefined Scenarios:**
| Scenario | Description |
|----------|-------------|
| `constant_turn` | Steady-state cornering (10° steering) |
| `lane_change` | Sinusoidal steering (lane change maneuver) |
| `step_steer` | Sudden steering input at t=2s |
| `acceleration` | Throttle ramp from rest |
| `braking` | Hard braking from high speed |
| `slalom` | Weaving/slalom maneuver |

**Custom Inputs:**
```bash
# Constant steering and throttle
python simulate_vehicle.py --steering 5.0 --throttle 2.0 --duration 10

# Adjust initial speed
python simulate_vehicle.py --scenario lane_change --initial-speed 20 --duration 8
```

**Options:**
- `--duration`: Simulation time in seconds (default: 10)
- `--initial-speed`: Starting velocity in m/s (default: 0)
- `--output-dir`: Output directory (default: `results/dynamic_simulations`)
- `--no-animation`: Skip GIF generation, only save static plot
- `--fps`: Animation frame rate (default: 30)

Outputs are saved to `results/dynamic_simulations/`:
- `sim_<scenario>_trajectory.png` - Static trajectory plot
- `sim_<scenario>_animation.gif` - Animated vehicle motion

### Basic Usage

```python
from models import SingleTrackModel, VehicleParams, FialaBrushTire
from planning import TrajectoryOptimizer
from world.world import World

# Load track
world = World("maps/Oval_Track_260m.mat", "Oval", diagnostic_plotting=False)

# Create vehicle
params = VehicleParams(...)  # See run_trajopt_demo.py for example
f_tire = FialaBrushTire(c0_alpha_nprad=0, c1_alpha_1prad=8.0, mu_none=0.9)
r_tire = FialaBrushTire(c0_alpha_nprad=0, c1_alpha_1prad=13.0, mu_none=0.9)
vehicle = SingleTrackModel(params, f_tire, r_tire)

# Direct Collocation
dc_opt = TrajectoryOptimizer(vehicle, world)
dc_result = dc_opt.solve(N=100, ds_m=2.6)

print(f"Collocation solve time: {dc_result.solve_time:.2f}s")
```

---

## References

1. **Vehicle Model:** Aggarwal & Gerdes, "Friction-Robust Autonomous Racing Using Trajectory Optimization Over Multiple Models", IEEE Open Journal of Control Systems, 2025.

2. **Decision Transformer:** Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling", NeurIPS 2021.

3. **Trajectory Transformer:** Janner et al., "Offline Reinforcement Learning as One Big Sequence Modeling Problem", NeurIPS 2021.

4. **Transformer Warm-Starts:** Guffanti et al., "Transformers for Trajectory Optimization with Application to Spacecraft Rendezvous", IEEE Aerospace 2024.

5. **SCP for Trajectory Optimization (background only):** Mao et al., "Successive Convexification of Non-Convex Optimal Control Problems", 2016.

---

## TODO

- [x] Implement unified vehicle model
- [x] Implement direct collocation optimizer
- [x] Create visualization tools
- [x] Archive/freeze SCP experimental branch
- [x] Implement IPOPT hard obstacle constraints in the production pipeline
- [x] Build Fix A + Fix B dataset generation pipeline
- [x] Implement Decision Transformer dataset loader, model, training, and evaluation scripts
- [x] Implement DT warm-start integration scaffold
- [ ] IPOPT obstacle-avoidance constraints (slack + staged solve)
- [ ] Evaluation benchmarks
- [ ] Train/val/test split artifacts with split-by-`base_id` hygiene
- [ ] Multi-shard DT training on the full generated dataset
- [ ] End-to-end DT-vs-baseline warm-start benchmark
- [ ] Multi-track generalization
