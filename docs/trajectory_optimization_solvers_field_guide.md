# Trajectory Optimization Solvers: A Practical Field Guide

This note is a **solver-centric** overview of common methods used for trajectory optimization (TO) in robotics/autonomy/aerospace. It focuses on *what they do, when they work, how they fail*, and **which ones are best for learned warm-starting**.  
Minimal math, but precise about the algorithmic characteristics.

---

## 0) The problem class (what “trajectory optimization” usually means)

A typical TO problem chooses a sequence of states and controls:

- States: $x_0, x_1, \dots, x_N$
- Controls: $u_0, u_1, \dots, u_{N-1}$

to minimize a cost (time, energy, tracking error, etc.) subject to:

- **Dynamics**: $x_{k+1} = f(x_k, u_k)$ (or continuous dynamics discretized)
- **Constraints**:
  - control limits ($u \in \mathcal U$),
  - state constraints (speed bounds, track boundaries),
  - obstacle avoidance,
  - terminal constraints (reach goal, match velocity, etc.).

The killer feature: the resulting optimization is almost always **nonlinear** and often **nonconvex**.

---

## 1) High-level taxonomy (the mental map)

There are a few “families” of solvers:

1. **NLP via discretization (Direct methods)**
   - Direct Collocation / Direct Transcription → solve a big **nonlinear program** (NLP).
2. **Shooting-based methods**
   - Single shooting, Multiple shooting → reduce variables by simulating dynamics.
3. **Second-order optimal control methods**
   - iLQR / DDP → dynamic programming style local optimization.
4. **Sequential approximation methods**
   - SQP (Sequential Quadratic Programming)
   - SCP / Successive Convexification
5. **Convex formulations (when possible)**
   - True convex optimal control (rare unless special structure).
6. **Discrete decision methods**
   - Mixed-Integer (MIQP/MINLP) for “left/right of obstacle” type logic.

---

## 2) Direct Collocation / Direct Transcription (NLP workhorse)

### What it is
Discretize the trajectory at $N$ nodes and enforce dynamics as constraints (e.g., trapezoidal, Hermite–Simpson). Then solve the resulting **large NLP** with a solver like IPOPT.

### Key characteristics
- **Very general**: handles nonlinear dynamics and constraints directly.
- **Good accuracy**: collocation can be high fidelity.
- **Robust baseline**: a standard “golden reference” for offline solutions.

### Pros
- Minimal “method engineering”: write constraints + objective → let solver work.
- Handles many constraints naturally.
- Often converges to high-quality solutions (local optimum).

### Cons / failure modes
- Can be **slow**, especially with obstacles or big horizons.
- Sensitive to initialization for hard nonconvexities (obstacles, tight constraints).
- Memory/time grows with horizon length (large sparse NLP).

### When it’s the best choice
- **Offline planning** where runtime isn’t tight.
- You want a trustworthy baseline to compare other methods.
- Dataset generation (“expert trajectories”).

---

## 3) Shooting methods (single & multiple shooting)

### What it is
Instead of enforcing dynamics at every node, you **simulate** dynamics forward.  
Decision variables may be only controls (single shooting) or states + controls (multiple shooting with continuity constraints).

### Single shooting
- Variables: only $u_0,\dots,u_{N-1}$
- State trajectory comes from simulation.

**Pros**
- Small number of variables.
- Easy to implement.

**Cons**
- Very sensitive to initial guess.
- Harder to handle state constraints (you don’t “own” the states).
- Can become numerically ill-conditioned for long horizons (chaotic sensitivity).

### Multiple shooting
- Variables: both states and controls in segments.
- Enforces continuity constraints between segments.

**Pros**
- More stable than single shooting.
- Better with constraints.

**Cons**
- Bigger than single shooting (closer to collocation).
- Still a general NLP.

### When shooting is attractive
- Short horizons, smooth problems, decent initial guess.
- Problems where simulation is very fast and constraints are mild.

---

## 4) iLQR / DDP (fast local optimal control)

### What it is
These methods iteratively build a **local quadratic approximation** of the cost and a **local linear approximation** of dynamics, then compute an improved policy/trajectory via backward/forward passes (like dynamic programming).

- iLQR: “iterative LQR” (common in robotics)
- DDP: “differential dynamic programming” (adds second-order dynamics terms; often approximated)

### Key characteristics
- Extremely **fast per iteration**.
- Strong when constraints are light.

### Pros
- Very fast for smooth dynamics and costs.
- Often great for MPC-like settings when constraints are simple.
- Produces stabilizing feedback structure (helpful in control).

### Cons / failure modes
- **Hard constraints** are awkward.
  - Common handling: penalties, soft constraints, projection heuristics.
- Obstacle avoidance with hard constraints tends to be messy.
- Can fail or produce nonsense when penalties dominate or constraints are tight.

### When it’s the best choice
- Fast control with mild constraints.
- Tracking problems (follow a reference) where feasibility is not tight.

---

## 5) SQP (Sequential Quadratic Programming)

### What it is
SQP solves a nonlinear constrained optimization by repeatedly solving a **QP** that approximates the NLP near the current iterate:
- Linearize constraints
- Use a quadratic model of the objective (and often the Lagrangian)

It’s the “classic” approach behind many NLP solvers.

### Key characteristics
- Strong general-purpose approach for constrained NLPs.
- Often has fast local convergence when well-behaved.

### Pros
- Powerful and widely used.
- Can handle constraints in a principled way.
- Good local convergence.

### Cons / failure modes
- QP subproblems can be **indefinite** or numerically delicate.
- Needs careful line search / trust region logic to be robust.
- Still local; obstacles/tight nonconvex constraints can cause trouble.

### When it’s the best choice
- General NLP problems with reasonable smoothness.
- When you want a principled “Newton-like” method for constrained problems.

---

## 6) SCP / Successive Convexification (convex subproblems + trust region)

### What it is
SCP (in the “successive convexification” flavor) repeatedly solves a **convex** approximation of a nonconvex TO problem:
- Linearize dynamics and nonconvex constraints around a reference trajectory.
- Use **trust regions** to keep updates local.
- Use **virtual control / slacks** to keep subproblems feasible.

### Key characteristics
- Each iteration is a **convex program** (QP/SOCP) → typically **fast and reliable**.
- Especially good when constraints are important (bounds, obstacles).

### Pros
- Convex subproblems can be solved very fast and predictably.
- Feasibility mechanisms (slacks/virtual control) make it robust to bad linearizations.
- Good for repeated solves and constraint-heavy scenarios.

### Cons / failure modes
- Needs careful scaling + trust-region tuning.
- Converges to local solutions (like others).
- If you don’t truly make subproblems convex, you lose the main advantage.

### When it’s the best choice
- **Constraint-heavy planning** (track + obstacles + bounds).
- Receding-horizon replanning where you want stable “few-iteration” refinement.
- When you want the optimization loop to be “solver-friendly” (convex).

---

## 7) True convex optimal control (rare but beautiful)

### What it is
Sometimes the problem can be written as a true convex program without approximation (e.g., certain linear dynamics + convex constraints + convex costs).

### Pros
- Global optimum.
- Very reliable.

### Cons
- Most interesting vehicle/robot problems aren’t convex without approximations.

### When it’s best
- When your system structure genuinely allows it (special cases).

---

## 8) Mixed-Integer optimization (MIQP/MINLP)

### What it is
Introduce binary variables to represent discrete choices:
- “go left vs go right”
- logical obstacle avoidance
- mode switches

### Pros
- Can handle discrete logic **correctly**.
- If solved to optimality, gives global guarantees (within the discretized model).

### Cons / failure modes
- Computationally expensive (can explode with horizon and obstacles).
- Often not real-time friendly.

### When it’s best
- When discrete decisions truly matter and local solutions are unacceptable.
- Small problems, offline planning, verification.

---

## 9) Summary comparison table (qualitative)

| Method | Handles hard constraints well? | Speed (per solve) | Robustness to bad init | Best use case |
|---|---:|---:|---:|---|
| Direct Collocation (NLP) | ✅✅ | ❌/⚠️ | ⚠️ | Offline “gold baseline”, dataset generation |
| Shooting (single) | ❌/⚠️ | ✅ | ❌ | Small problems, mild constraints |
| Shooting (multiple) | ✅/⚠️ | ⚠️ | ⚠️ | Middle ground; simulation-friendly |
| iLQR/DDP | ❌ (needs tricks) | ✅✅ | ⚠️ | Fast control, smooth objectives, mild constraints |
| SQP | ✅ | ⚠️ | ⚠️ | General constrained NLP solving |
| SCP (succ. convex.) | ✅✅ | ✅ | ✅ | Constraint-heavy planning, repeated solves/MPC |
| Mixed-Integer | ✅✅✅ | ❌❌ | ✅ | Discrete logic, offline small problems |

Legend: ✅✅ = strong, ⚠️ = depends, ❌ = weak.

---

## 10) Which solver is best for learning / warm-starting?

If the goal is **learned warm-start → refine with optimizer**, you want:
1. A solver whose runtime/iterations strongly depend on initialization
2. A solver that produces **interpretable refinement metrics**
3. A solver that remains stable under constraints and obstacles

### Best choices (typical)
#### ✅ SCP / Successive Convexification
Why it pairs well with learning:
- A good warm-start reduces the number of convex iterations dramatically.
- Iteration count and feasibility measures are clean metrics.
- Convex solves are fast → total runtime becomes “few iterations × convex solve”.

#### ✅ Direct Collocation (NLP) as baseline + dataset generator
Why it’s still valuable:
- Great for generating “expert” trajectories.
- Warm-start helps NLP too, but behavior is less clean to compare than SCP.
- Still an excellent reference solver.

### Sometimes good
#### ⚠️ iLQR/DDP (if constraints are mild or softened)
- Warm-start matters, and it’s very fast.
- But with hard constraints and obstacles, you often end up hiding issues in penalties.

### Usually not ideal (for learning-warm-start stories)
- Mixed-integer: learning won’t usually help much; the bottleneck is combinatorics.
- Single shooting: too fragile; warm-start may help but failure modes are noisy.

---

## 11) Practical recommendation for the project context you described

- Use **Direct Collocation + IPOPT** as the *high-quality baseline* and to generate trajectories.
- Use **SCP / successive convexification** as the *fast refinement method* where warm-starting shows clear gains.
- Evaluate warm-start methods by:
  - feasibility after refinement,
  - final objective (lap time or time-to-goal),
  - iterations to reach tolerance,
  - runtime.

This gives a crisp narrative: learning improves the *initializer*, SCP turns that initializer into a feasible high-quality solution quickly, and collocation provides a trusted reference.

---

## 12) Glossary (tiny)
- **Warm-start:** Initialize the solver near a good solution.
- **Local optimum:** Best solution in a neighborhood; not globally best.
- **Trust region:** Limits step size so local approximations stay valid.
- **Virtual control / slack:** Extra variables to keep subproblems feasible; penalized to vanish at convergence.
