# Dynamic Model Documentation

This document describes the single-track vehicle dynamic model used for trajectory optimization.

**Reference:** Aggarwal & Gerdes, "Friction-Robust Autonomous Racing Using Trajectory Optimization Over Multiple Models", IEEE Open Journal of Control Systems, 2025.

---

## 1. Reference Frames

The model uses three coordinate frames: **Body**, **Global**, and **Path (Frenet)**.

### 1.1 Body Frame (Vehicle-Fixed)

The body frame is attached to the vehicle's center of gravity (CG) and moves with the vehicle.

```
            x (forward)
            ↑
            │
       y ←──●──→
      (left)│
            ↓
```

| Axis | Direction | Convention |
|------|-----------|------------|
| x | Forward (longitudinal) | Positive = forward |
| y | Left (lateral) | Positive = left (ISO/DIN) |
| z | Up (vertical) | Positive = up |

**Right-hand rule:** x × y = z → forward × left = up ✓

**Body frame states:**
- `vx` or `ux`: Longitudinal velocity [m/s]
- `vy` or `uy`: Lateral velocity [m/s]
- `r`: Yaw rate [rad/s]

> **Note:** This is the ISO/DIN convention where y points left. The SAE convention has y pointing right.

### 1.2 Global Frame (Inertial)

The global frame is fixed to the ground and used for position tracking and visualization.

```
        North (Y)
           ↑
           │
           │ψ (heading, CCW from North)
     ──────●──────→ East (X)
           │
           │
           ↓
```

| Axis | Direction |
|------|-----------|
| X | East |
| Y | North |
| Z | Up (implied) |

**Right-hand rule:** X × Y = Z → East × North = Up ✓

**Heading convention:** ψ is measured **counterclockwise from North**.

### 1.3 Path Frame (Frenet/Curvilinear)

The path frame is attached to the track centerline and used for trajectory optimization.

```
        s (arc length along centerline)
    ════════════════════════►
                │
                │ e (lateral deviation)
                ▼
             vehicle
               ↗ Δψ (heading error)
```

| Variable | Description | Units |
|----------|-------------|-------|
| s | Progress along centerline | m |
| e | Lateral offset from centerline | m |
| Δψ | Heading error (vehicle heading - path tangent) | rad |
| κ | Path curvature (1/radius) | 1/m |

---

## 2. Coordinate Transformations

### 2.1 Body to Global Transformation

The transformation from body-frame velocities to global-frame velocities is:

```
deast  = -vx·sin(ψ) - vy·cos(ψ)
dnorth =  vx·cos(ψ) - vy·sin(ψ)
```

Or in matrix form:

```
[deast ]   [-sin(ψ)  -cos(ψ)] [vx]
[dnorth] = [ cos(ψ)  -sin(ψ)] [vy]
```

### 2.2 Why This Transformation Works

This is **not** a standard rotation matrix because we combine:
1. **ENU axis ordering** (East-North-Up)
2. **ISO body frame** (y = left, not right)
3. **Heading from North** (not from East)

**Derivation:**

At heading ψ (CCW from North), the body axes point in these global directions:
- Body x (forward) → global direction: `(-sin(ψ), cos(ψ))`
- Body y (left) → global direction: `(-cos(ψ), -sin(ψ))`

Verification at key angles:

| ψ | Vehicle Points | Body x (forward) → | Body y (left) → |
|---|----------------|-------------------|-----------------|
| 0° | North | (0, 1) = North ✓ | (-1, 0) = West ✓ |
| 90° | West | (-1, 0) = West ✓ | (0, -1) = South ✓ |
| 180° | South | (0, -1) = South ✓ | (1, 0) = East ✓ |
| 270° | East | (1, 0) = East ✓ | (0, 1) = North ✓ |

The transformation respects the right-hand rule in both frames and is mathematically consistent.

**Reference implementations:** Both `models-main` and `multimodel-trajectory-optimization` use this same ENU convention with heading from North. The state variables are named `east_m` and `north_m` in both codebases.

### 2.3 Body to Path Transformation

The path kinematics relate body velocities to path coordinate rates:

```
ṡ   = (vx·cos(Δψ) - vy·sin(Δψ)) / (1 - κ·e)
ė   =  vx·sin(Δψ) + vy·cos(Δψ)
Δψ̇ =  r - κ·ṡ
```

Where:
- `ṡ` is the rate of progress along the path
- `ė` is the rate of lateral deviation
- `Δψ̇` is the rate of heading error change
- `κ` is the local path curvature

---

## 3. State and Control Vectors

### 3.1 State Vector

**Path formulation (8 states):**
```
x = [vx, vy, r, ΔFz_long, ΔFz_lat, t, e, Δψ]ᵀ
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | vx | Longitudinal velocity | m/s |
| 1 | vy | Lateral velocity | m/s |
| 2 | r | Yaw rate | rad/s |
| 3 | ΔFz_long | Longitudinal weight transfer | kN |
| 4 | ΔFz_lat | Lateral weight transfer | kN |
| 5 | t | Time | s |
| 6 | e | Lateral deviation | m |
| 7 | Δψ | Heading error | rad |

### 3.2 Control Vector

```
u = [δ, Fx]ᵀ
```

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | δ | Front steering angle | rad |
| 1 | Fx | Total longitudinal force command | kN |

---

## 4. Equations of Motion

### 4.1 Translational Dynamics

**Longitudinal:**
```
m·v̇x = -Fy,f·sin(δ) + Fx,f·cos(δ) + Fx,r - Fd + m·r·vy
```

**Lateral:**
```
m·v̇y = Fy,f·cos(δ) + Fx,f·sin(δ) + Fy,r + Fl - m·r·vx
```

Where:
- `Fx,f`, `Fx,r`: Front/rear longitudinal tire forces
- `Fy,f`, `Fy,r`: Front/rear lateral tire forces
- `Fd`: Total drag force (rolling resistance + aerodynamic)
- `Fl`: Lateral force from road bank
- `m·r·vy`, `m·r·vx`: Centripetal coupling terms

### 4.2 Rotational Dynamics

```
Iz·ṙ = a·(Fy,f·cos(δ) + Fx,f·sin(δ)) - b·Fy,r + Mz,b
```

Where:
- `Iz`: Yaw moment of inertia
- `a`, `b`: Distance from CG to front/rear axle
- `Mz,b`: Brake yaw moment (differential braking effect)

### 4.3 Weight Transfer Dynamics

**Longitudinal weight transfer (1st order):**
```
ΔḞz_long = (1/τ_long)·(m·ax·h_com/L - ΔFz_long)
```

**Lateral weight transfer (1st order):**
```
ΔḞz_lat = (1/τ_lat)·((h_com/t + g·h_l·R_φ)·Fy_total - ΔFz_lat)
```

**Axle normal loads:**
```
Fz,f = (b/L)·m·g - ΔFz_long
Fz,r = (a/L)·m·g + ΔFz_long
```

---

## 5. Force Models

### 5.1 Tire Forces (Fiala Brush Model)

The lateral tire force is computed using the Fiala brush tire model:

```
        ⎧ -Cα·tan(α) + Cα²/(3Fy_max)·|tan(α)|·tan(α) - Cα³/(27Fy_max²)·tan³(α),  |α| ≤ αsl
Fy =    ⎨
        ⎩ -Fy_max·sgn(α),  otherwise
```

Where:
- `Cα = C'α·Fz`: Load-dependent cornering stiffness
- `Fy_max = √((μ·Fz)² - Fx²)`: Maximum lateral force (friction circle)
- `αsl = tan⁻¹(3·Fy_max/Cα)`: Saturation slip angle

### 5.2 Drag Forces

**Total drag:**
```
Fd = Frr + Faero + Fgrade
```

| Term | Formula | Description |
|------|---------|-------------|
| Frr | `Cd0` | Rolling resistance (constant) |
| Faero | `Cd1·vx + Cd2·vx²` | Aerodynamic drag |
| Fgrade | `m·g·sin(θ)` | Grade resistance |

### 5.3 Road Geometry Forces

**Lateral force from road bank:**
```
Fl = -m·g·cos(θ)·sin(φ)
```

Where:
- `θ`: Road grade angle (positive = uphill)
- `φ`: Road bank angle (positive = banked right)

### 5.4 Brake Yaw Moment

During braking while cornering, differential normal loads on left/right wheels create a yaw moment:

```
Mz,b = Fx,f_brake·γ·t·ΔFz_lat/Fz,f + Fx,r_brake·(1-γ)·t·ΔFz_lat/Fz,r
```

Where:
- `γ`: Front/rear load transfer distribution parameter
- `t`: Average track width
- `Fx,f_brake`, `Fx,r_brake`: Braking-only forces (≤ 0)

---

## 6. Simplifications and Optional Terms

Several terms can be set to zero or disabled for simplified models:

### 6.1 Road Geometry (Flat Track)

Set `θ = 0` and `φ = 0` for a flat, level track:
- Eliminates `Fgrade` (grade resistance)
- Eliminates `Fl` (bank force)

```python
# In dynamics call:
theta_rad = 0.0  # No grade
phi_rad = 0.0    # No bank
```

### 6.2 Weight Transfer Dynamics

Set `enable_weight_transfer = False` to use static weight distribution:
- Eliminates `ΔFz_long` and `ΔFz_lat` as states
- Uses constant axle loads: `Fz,f = (b/L)·m·g`, `Fz,r = (a/L)·m·g`
- Reduces state dimension from 8 to 6

```python
vehicle = SingleTrackModel(params, f_tire, r_tire, enable_weight_transfer=False)
```

### 6.3 Brake Yaw Moment

The brake yaw moment `Mz,b` is automatically zero when:
- Not braking (`Fx ≥ 0`)
- Driving straight (`ay ≈ 0`, no lateral weight transfer)

To completely disable, set `γ = 0` in vehicle parameters.

### 6.4 Simplified Tire Model

For linear analysis, the tire model reduces to:
```
Fy ≈ -Cα·α
```
This applies when slip angles are small (`|α| << αsl`).

---

## 7. Summary of Model Variants

| Variant | States | Features Disabled |
|---------|--------|-------------------|
| Full model | 8 | None |
| No weight transfer | 6 | ΔFz_long, ΔFz_lat dynamics |
| Flat track | 8 | Grade, bank forces |
| Simple (linear tire) | 8 | Tire saturation |
| Minimal | 6 | Weight transfer + flat track |

---

## 8. Implementation Notes

### 8.1 Units Convention

| Quantity | Code Units | Reason |
|----------|------------|--------|
| Forces | kN | Numerical scaling |
| Velocities | m/s | SI |
| Angles | rad | Calculus compatibility |
| Mass | kg | SI |
| Inertia | kg·m² | SI |

### 8.2 Sign Conventions

| Quantity | Positive Direction |
|----------|-------------------|
| vx | Forward |
| vy | Left |
| r | Counterclockwise (turning left) |
| δ | Left turn |
| Fx | Driving (acceleration) |
| ΔFz_long | Load transfer to rear |
| e | Right of centerline (typically) |
| Δψ | Vehicle pointing right of path tangent |

### 8.3 Code Location

- Vehicle model: `models/vehicle.py`
- Tire model: `models/tire.py`
- Parameters: `models/config/vehicle_params_gti.yaml`

---

## References

1. Aggarwal & Gerdes, "Friction-Robust Autonomous Racing Using Trajectory Optimization Over Multiple Models", IEEE OJCS, 2025.
2. Pacejka, "Tire and Vehicle Dynamics", 3rd ed., 2012.
3. Subosits & Gerdes, "Impacts of Model Fidelity on Trajectory Optimization for Autonomous Vehicles in Extreme Maneuvers", IEEE T-IV, 2021.
