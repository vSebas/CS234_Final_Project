# Rollout Diagnosis Results

This note records the corrected relaxed-gate rollout diagnosis for `oval_fatrop_improved_postproj_ft2_rerun + projection=soft`.

## noobs
### accepted (2)
- start_progress_m: 103.50, 4.12
- ws_fallback_count: mean 1.0000, min 1.0000, max 1.0000
- ws_projection_fraction: mean 0.6708, min 0.6333, max 0.7083
- ws_projection_total_magnitude: mean 61.4237, min 54.0671, max 68.7803
- ws_projection_max_magnitude: mean 4.0311, min 3.5690, max 4.4932
- ipopt_iterations: mean 240.0000, min 209.0000, max 271.0000
- solve_time_s: mean 55.3508, min 50.5794, max 60.1223
- rejection reasons: {"": 2}

### rejected (8)
- start_progress_m: 160.10, 149.30, 135.08, 1.66, 143.08, 251.77, 63.38, 246.08
- ws_fallback_count: mean 7.2500, min 2.0000, max 16.0000
- ws_projection_fraction: mean 0.4823, min 0.3500, max 0.7333
- ws_projection_total_magnitude: mean 102.5428, min 71.5965, max 118.7416
- ws_projection_max_magnitude: mean 4.5521, min 3.8952, max 4.9507
- ipopt_iterations: mean 198.5000, min 164.0000, max 288.0000
- solve_time_s: mean 47.5601, min 41.9850, max 66.5706
- rejection reasons: {"gate:fallback_count": 8}

## obs
### accepted (2)
- start_progress_m: 110.17, 109.90
- ws_fallback_count: mean 1.0000, min 1.0000, max 1.0000
- ws_projection_fraction: mean 0.4750, min 0.4250, max 0.5250
- ws_projection_total_magnitude: mean 67.4132, min 56.0377, max 78.7886
- ws_projection_max_magnitude: mean 3.9452, min 3.8260, max 4.0643
- ipopt_iterations: mean 304.5000, min 284.0000, max 325.0000
- solve_time_s: mean 67.5753, min 67.0061, max 68.1446
- rejection reasons: {"": 2}

### rejected (8)
- start_progress_m: 248.28, 186.05, 22.68, 254.06, 86.52, 188.32, 123.96, 13.59
- ws_fallback_count: mean 4.5000, min 0.0000, max 13.0000
- ws_projection_fraction: mean 0.5198, min 0.3417, max 0.7833
- ws_projection_total_magnitude: mean 107.2771, min 71.1740, max 144.7744
- ws_projection_max_magnitude: mean 4.7254, min 4.0986, max 6.2857
- ipopt_iterations: mean 243.8750, min 174.0000, max 395.0000
- solve_time_s: mean 55.4983, min 45.8333, max 77.3574
- rejection reasons: {"Obstacle collision at node 0: dist=1.377 < required_radius=1.931": 1, "Obstacle collision at node 18: dist=1.636 < required_radius=1.718": 1, "Obstacle collision at node 31: dist=1.244 < required_radius=2.078": 1, "Obstacle collision at node 49: dist=0.856 < required_radius=2.049": 1, "Obstacle collision at node 5: dist=1.509 < required_radius=1.942": 1, "gate:fallback_count": 2, "gate:fallback_count,projection_total": 1}

## Trace-Rich Rerun Artifacts

Corrected trace-rich reruns:
- no-obstacle:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_154442.csv`
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_154442_summary.json`
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_154442_rollout_trace.jsonl`
  - compare plots:
    - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/compare_plots/`
- obstacle:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_154442.csv`
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_154442_summary.json`
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_154442_rollout_trace.jsonl`
  - compare plots:
    - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/compare_plots/`

## Trace Findings

No-obstacle:
- acceptance remained `20%`
- accepted starts were again clustered near:
  - `~4.1 m`
  - `~103.5 m`
- rejected starts covered the other tested progress values
- first fallback often happened immediately:
  - step `0` or `1` for most rejected starts
- accepted no-obstacle starts still had only `1` fallback each, while rejected starts averaged much higher fallback counts

Obstacle:
- acceptance remained `20%`
- accepted starts clustered near:
  - `~110.2 m`
  - `~109.9 m`
- rejected obstacle cases split into two groups:
  - gate failures from high fallback / high projection
  - direct collision rejections during warm-start validation
- accepted obstacle cases were still solver-harmful:
  - iterations `284` and `325`

Projection / fallback mechanics:
- in both gates, the largest projection step is usually dominated by:
  - `e_clip`
  - sometimes `dpsi_clip`
  - occasionally `obs_push` for obstacle cases
- many peak projection events occur late in the rollout (`k` near `90-119`), but fallback often begins very early
- this suggests two separate problems:
  - an early instability / fallback trigger
  - a later persistent boundary-tracking issue that keeps clipping `e`

Early-step findings:
- many rejected scenarios hit first fallback at:
  - `k = 0` or `k = 1`
- accepted scenarios usually delay the first fallback a little:
  - no-obstacle accepted cases: `k = 1` and `k = 3`
  - obstacle accepted cases: `k = 2`
- first DT actions are often characterized by:
  - negative steering on most starts
  - either very small longitudinal force (`Fx ~ 0.07-0.42`) or strong acceleration (`Fx ~ 4.0-5.0`)
- accepted start regions tend to use less extreme first-step commands:
  - no-obstacle accepted:
    - `[-0.129, 2.348]`
    - `[-0.287, 4.002]`
  - obstacle accepted:
    - `[-0.136, 1.950]`
    - `[-0.135, 1.966]`

Important instrumentation note:
- the original rollout trace stored `x_pred_before_projection` after fallback had already replaced the failed full-model step
- so the trace could show a benign state even when fallback was triggered because the real model step was unstable
- this has now been fixed in:
  - `planning/dt_warmstart.py`
- future traces will include:
  - `x_model_raw`
  - `dt_model_s`
  - `model_step_valid`

## Raw Model-Step Diagnosis

Targeted rerun artifact:
- `docs/ROLLOUT_EARLY_STEP_DIAGNOSIS.json`

Representative rejected and accepted scenarios were rerun with the new instrumentation.

What the raw model step shows:
- when fallback triggers early, the visible state can still look benign
  - `ux`, `e`, and `dpsi` often remain moderate
- but the raw full-model integration explodes in hidden dynamic states:
  - `uy`
  - `r`

Examples:

No-obstacle rejected, scenario `0`, first fallback at `k=0`:
- first action:
  - `[-0.282, 0.088]`
- raw model step:
  - `uy = -460.75`
  - `r = 614.89`
- rejection reason from state screen:
  - `uy=-460.749; r=614.894`

No-obstacle accepted, scenario `6`, first fallback at `k=3`:
- first action at fallback step:
  - `[0.349, 4.303]`
- raw model step:
  - `uy = -10.16`
  - `r = 14.99`
- rejection reason from state screen:
  - `r=14.993`

No-obstacle accepted, scenario `7`, first fallback at `k=1`:
- first action at fallback step:
  - `[-0.242, 2.792]`
- raw model step:
  - `uy = 78.95`
  - `r = -102.77`
- rejection reason:
  - `uy=78.946; r=-102.771`

Obstacle rejected, scenario `1`, first fallback at `k=0`:
- first action:
  - `[-0.332, 0.180]`
- raw model step:
  - `uy = -605.48`
  - `r = 808.10`

Obstacle accepted, scenario `6`, first fallback at `k=2`:
- first action at fallback step:
  - `[0.113, 5.0]`
- raw model step:
  - `uy = 300.14`
  - `r = -394.92`

Interpretation:
- the early fallback is not mainly caused by track-bound or obstacle projection
- it is caused by raw dynamic propagation blowing up `uy` and `r`
- the fallback controller then replaces that unstable state with a conservative centerline-recovery state
- this explains why traces looked visually benign after fallback even though the true model step was catastrophic

Lateral-dynamics interpretation:
- the strongest current interpretation is that the DT rollout is not staying inside the dynamically valid lateral regime of the tire model
- in practical terms:
  - the policy may imitate geometry and controls reasonably in-distribution
  - but under closed-loop rollout it is not producing actions/states that keep lateral dynamics (`uy`, `r`) stable
- that makes this look less like a generic capacity problem and more like a learned lateral-dynamics / tire-model stability weakness
- the short-term engineering response can still be a wrapper fix, but the modeling diagnosis should be recorded explicitly:
  - the learned policy is weak in the part of the dynamics where lateral tire forces dominate

## Current Diagnosis

1. Start-progress matters materially.
   - Accepted no-obstacle and obstacle rollouts cluster in narrow progress regions.

2. The rollout is still unstable very early.
   - Many scenarios hit fallback at step `0` or `1`, even before obstacle pressure is relevant.
   - Early instability is triggered by the raw model step blowing up `uy` and `r`, not by projection.
   - This is consistent with weak learned lateral-dynamics stability under the tire model.

3. Accepted does not mean solver-helpful.
   - The accepted obstacle cases are still worse than baseline in iteration count.

4. The dominant projection mode is still lateral clipping.
   - `e_clip` dominates both no-obstacle and obstacle traces.
   - That supports the view that the rollout is failing mainly in lateral behavior, not longitudinal progress.

## Concrete Next Hypothesis

The next intervention should target the early rollout mechanics, not the gate:
- stabilize the first few DT rollout steps so the raw model step does not explode `uy` / `r`
- then check whether the fallback controller and/or `e` projection are still creating a solver-unfriendly path even when the gate accepts the rollout

## First Rollout-Stabilization Attempt

Implemented in:
- `planning/dt_warmstart.py`

Change:
- add a conservative action envelope for the first rollout steps before the full dynamic step:
  - tighter steering limits in the first `3-6` steps
  - tighter longitudinal-force limits in the first `3-6` steps
  - additional tightening when low-speed lateral motion (`uy`, `r`) has already started to build

Why:
- the early-step diagnosis showed raw model-step blow-ups in `uy` / `r`
- the intervention was intentionally narrow and rollout-only

Smoke-check artifacts:
- no-obstacle:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_165003_summary.json`
- obstacle:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_165003_summary.json`

Smoke-check result (`3` scenarios each):

No-obstacle:
- baseline solve mean: `47.40s`
- DT solve mean: `44.36s`
- DT total mean: `47.39s`
- acceptance: `0%`
- fallback mean: `9.33`

Obstacle:
- baseline solve mean: `62.98s`
- DT solve mean: `60.53s`
- DT total mean: `63.19s`
- acceptance: `0%`
- fallback mean: `4.00`

Read:
- the stabilizer appears to reduce fallback pressure somewhat
- solve time remains a bit better than baseline
- but acceptance is still `0%` in this smoke check
- so the first stabilization patch is promising but not decisive

Targeted before/after check with the new raw-model trace:
- artifact:
  - `docs/ROLLOUT_EARLY_STEP_DIAGNOSIS_STABILIZED.json`

Comparison against the pre-stabilizer diagnostic:

Rejected cases that improved:
- no-obstacle scenario `0`:
  - fallback count `17 -> 10`
  - first fallback `k: 0 -> 1`
- obstacle scenario `1`:
  - fallback count `13 -> 6`
  - first fallback `k: 0 -> 1`

But the raw model step still blows up badly even with the stabilizer:
- no-obstacle scenario `0`:
  - raw step still reaches `uy = 362.19`, `r = -477.72`
- obstacle scenario `1`:
  - raw step still reaches `uy = 336.03`, `r = -443.31`

Previously accepted cases did not clearly improve:
- no-obstacle scenario `6`:
  - fallback count `1 -> 2`
  - raw step now reaches `uy = -95.35`, `r = 124.76`
- obstacle scenario `6`:
  - fallback count `1 -> 2`
  - raw step now reaches `uy = -667.39`, `r = 881.83`

Read:
- the warmup action envelope helps some obviously bad starts
- but it does not cure the underlying instability
- and it is not robust enough to preserve the previously better starts
- this points away from “just clamp the DT actions a bit more”
- and toward a more structural fix for the first rollout steps

## Structural Warmup Attempt

Implemented in:
- `planning/dt_warmstart.py`

Change:
- use the stable warmup propagator (`_fallback_step`) intentionally for the first rollout steps
- hand back to the full dynamic model after warmup
- warmup steps are not counted as fallback events

Why:
- the clamp-based stabilizer was helping some bad starts
- but the raw dynamic step was still exploding `uy` / `r`
- this tests whether the first few full-model steps are the real bottleneck

Smoke-check artifacts:
- no-obstacle:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs0-0_seed52_N120/warmstart_eval_20260313_172811_summary.json`
- obstacle:
  - `dt/checkpoints/oval_fatrop_improved_postproj_ft2_rerun/warmstarts/eval/Oval_Track_260m_obs1-4_seed44_N120/warmstart_eval_20260313_172811_summary.json`

Smoke-check result (`3` scenarios each):

No-obstacle:
- baseline solve mean: `46.07s`
- DT solve mean: `43.82s`
- DT total mean: `46.19s`
- acceptance: `0%`
- fallback mean: `5.33`

Obstacle:
- baseline solve mean: `62.02s`
- DT solve mean: `59.35s`
- DT total mean: `62.04s`
- acceptance: `0%`
- fallback mean: `4.33`

Read:
- this is better than the clamp-based stabilizer on no-obstacle fallback pressure
- DT total time is now extremely close to baseline in both smoke checks
- but acceptance is still `0%`
- iteration count is still unchanged

Practical interpretation:
- structural warmup is a better wrapper intervention than first-step action clamping
- but even this does not cross the line to a clearly accepted, solver-helpful warm-start
