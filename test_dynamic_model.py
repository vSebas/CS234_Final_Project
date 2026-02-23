"""
Test script for the unified vehicle model.

Run with: conda run -n DT_trajopt python test_dynamic_model.py
"""

import numpy as np
import casadi as ca
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, '/run/media/saveas/Secondary-Storage/Masters/CS234/CS234_Reinforcement_Learning/Final Project/trajopt_unified')

from models import load_vehicle_from_yaml

print("="*70)
print("Testing Unified Single-Track Vehicle Model")
print("="*70)

# =============================================================================
# Create model
# =============================================================================
print("\n[1] Creating model...")

config_file = project_root  / "models" / "config" / "vehicle_params_gti.yaml"

if not config_file.exists():
    raise FileNotFoundError(
        f"Vehicle config not found at {config_file}. "
        "Please ensure models/config/vehicle_params_gti.yaml exists."
    )

model = load_vehicle_from_yaml(config_file, enable_weight_transfer=True)
print(f"   Vehicle: {model}")

# =============================================================================
# Test 1: Basic dynamics (flat road)
# =============================================================================
print("\n[2] Testing basic dynamics (flat road)...")

ux, uy, r = 15.0, 0.0, 0.0
dfz_long, dfz_lat = 0.0, 0.0
delta, fx = 0.0, 0.5  # Small throttle
theta, phi = 0.0, 0.0  # Flat road

derivs = model.temporal_dynamics(ux, uy, r, dfz_long, dfz_lat, delta, fx, theta, phi)
dux, duy, dr, ddfz_long, ddfz_lat = [float(d) for d in derivs]

print(f"  State: ux={ux} m/s, uy={uy} m/s, r={r} rad/s")
print(f"  Control: delta={delta} rad, fx={fx} kN")
print(f"  Road: theta={theta} rad (flat), phi={phi} rad (flat)")
print(f"  Derivatives:")
print(f"    dux/dt = {dux:.4f} m/s^2")
print(f"    duy/dt = {duy:.4f} m/s^2")
print(f"    dr/dt  = {dr:.4f} rad/s^2")
print(f"    ddfz_long/dt = {ddfz_long:.4f} kN/s")
print(f"    ddfz_lat/dt  = {ddfz_lat:.4f} kN/s")

# =============================================================================
# Test 2: Road grade effect (uphill)
# =============================================================================
print("\n[3] Testing road grade effect (5% uphill)...")

theta_uphill = np.arctan(0.05)  # 5% grade
derivs_uphill = model.temporal_dynamics(ux, uy, r, dfz_long, dfz_lat, delta, fx, theta_uphill, 0.0)
dux_uphill = float(derivs_uphill[0])

print(f"  Grade: {np.degrees(theta_uphill):.2f} deg ({np.tan(theta_uphill)*100:.1f}%)")
print(f"  dux/dt (flat):   {dux:.4f} m/s^2")
print(f"  dux/dt (uphill): {dux_uphill:.4f} m/s^2")
print(f"  Difference: {dux_uphill - dux:.4f} m/s^2 (should be negative = deceleration)")

# Expected: F_grade = -m*g*sin(theta) ≈ -1868 * 9.81 * 0.05 = -916 N
f_grade_expected = -model.params.m_kg * 9.81 * np.sin(theta_uphill)
a_grade_expected = f_grade_expected / model.params.m_kg
print(f"  Expected grade acceleration: {a_grade_expected:.4f} m/s^2")

# =============================================================================
# Test 3: Road bank effect (cornering)
# =============================================================================
print("\n[4] Testing road bank effect (5 deg bank while turning)...")

ux_turn, uy_turn, r_turn = 15.0, 0.5, 0.1
delta_turn = 0.05

phi_bank = np.radians(5.0)  # 5 degree bank
derivs_flat = model.temporal_dynamics(ux_turn, uy_turn, r_turn, 0.0, 0.0, delta_turn, 0.5, 0.0, 0.0)
derivs_banked = model.temporal_dynamics(ux_turn, uy_turn, r_turn, 0.0, 0.0, delta_turn, 0.5, 0.0, phi_bank)

duy_flat = float(derivs_flat[1])
duy_banked = float(derivs_banked[1])

print(f"  Bank angle: {np.degrees(phi_bank):.1f} deg")
print(f"  duy/dt (flat):   {duy_flat:.4f} m/s^2")
print(f"  duy/dt (banked): {duy_banked:.4f} m/s^2")
print(f"  Bank contribution: {duy_banked - duy_flat:.4f} m/s^2")

# Expected: F_bank = -m*g*sin(phi) ≈ -1868 * 9.81 * sin(5°) = -1598 N
f_bank_expected = -model.params.m_kg * 9.81 * np.sin(phi_bank)
a_bank_expected = f_bank_expected / model.params.m_kg
print(f"  Expected bank acceleration: {a_bank_expected:.4f} m/s^2")

# =============================================================================
# Test 4: Weight transfer
# =============================================================================
print("\n[5] Testing weight transfer dynamics...")

# Hard braking
fx_brake = -5.0  # 5 kN braking
derivs_brake = model.temporal_dynamics(20.0, 0.0, 0.0, 0.0, 0.0, 0.0, fx_brake, 0.0, 0.0)
ddfz_long_brake = float(derivs_brake[3])

print(f"  Braking force: {fx_brake} kN")
print(f"  ddfz_long/dt = {ddfz_long_brake:.4f} kN/s")
print(f"  (Negative = load transferring to front)")

# =============================================================================
# Test 5: Path dynamics
# =============================================================================
print("\n[6] Testing path dynamics...")

# State: [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi]
x_path = ca.DM([15.0, 0.2, 0.05, 0.0, 0.0, 0.0, 0.5, 0.02])
u_path = ca.DM([0.03, 0.5])
k_psi = 0.02  # Curvature

dx_dt, s_dot = model.dynamics_dt_path_vec(x_path, u_path, k_psi, theta=0.0, phi=0.0)

print(f"  State: ux=15, uy=0.2, r=0.05, e=0.5, dpsi=0.02")
print(f"  Curvature: k={k_psi} 1/m (R={1/k_psi:.1f} m)")
print(f"  s_dot = {float(s_dot):.4f} m/s (arc length rate)")
print(f"  de/dt = {float(dx_dt[6]):.4f} m/s (lateral deviation rate)")
print(f"  ddpsi/dt = {float(dx_dt[7]):.4f} rad/s (heading error rate)")

# =============================================================================
# Test 6: CasADi symbolic compatibility
# =============================================================================
print("\n[7] Testing CasADi symbolic compatibility...")

x_sym = ca.SX.sym('x', 8)
u_sym = ca.SX.sym('u', 2)
k_sym = ca.SX.sym('k')
theta_sym = ca.SX.sym('theta')
phi_sym = ca.SX.sym('phi')

dx_dt_sym, s_dot_sym = model.dynamics_dt_path_vec(x_sym, u_sym, k_sym, theta_sym, phi_sym)

print(f"  Created symbolic dynamics function")
print(f"  dx_dt shape: {dx_dt_sym.shape}")
print(f"  s_dot shape: {s_dot_sym.shape}")

# Create CasADi function
f_dynamics = ca.Function('f_dynamics',
                         [x_sym, u_sym, k_sym, theta_sym, phi_sym],
                         [dx_dt_sym, s_dot_sym],
                         ['x', 'u', 'k', 'theta', 'phi'],
                         ['dx_dt', 's_dot'])

# Evaluate numerically
dx_dt_num, s_dot_num = f_dynamics(x_path, u_path, k_psi, 0.0, 0.0)
print(f"  Numerical evaluation successful")
print(f"  s_dot (from function) = {float(s_dot_num):.4f} m/s")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: Unified Model Features")
print("="*70)
print("""
  State vectors:
    - Dynamics:  [ux, uy, r, dfz_long, dfz_lat] (5 states)
    - Path:      [ux, uy, r, dfz_long, dfz_lat, t, e, dpsi] (8 states)
    - Global:    [ux, uy, r, dfz_long, dfz_lat, t, east, north, psi] (9 states)

  Features from models-main:
    [x] Lateral weight transfer (dfz_lat)
    [x] Load-dependent cornering stiffness
    [x] Smooth force distribution (tanh)
    [x] Clean modular structure

  Features from multimodel-trajectory-optimization:
    [x] Road grade (theta) - affects longitudinal acceleration
    [x] Road bank (phi) - affects lateral acceleration
    [x] Time state (t) - for temporal optimization

  Ready for SCP trajectory optimization!
""")
print("="*70)
