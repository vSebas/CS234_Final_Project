from typing import Protocol

import casadi as ca
from casadi_tools.dynamics.named_arrays import NamedVector as NV
from casadi_tools.nlp_utils import casadi_builder as cb

from models import vehicle_params as vp


class TireModel(Protocol):
    """Protocol class defining what tire methods are expected."""

    def calc_fy_kn(self, alpha_rad, fz_kn, fx_kn):
        pass

    def calc_max_fx_kn(self, alpha_rad, fz_kn):
        pass

    def calc_max_fy_kn(self, fz_kn, fx_kn):
        pass


_StatesVelocity = NV.create_from_field_names(
    "StatesVelocity", ("ux_mps", "uy_mps", "r_radps")
)

_StatesWeightTransfer = NV.create_from_field_names(
    "StatesWeightTransfer", ("dfz_long_kn", "dfz_lat_kn")
)

_StatesDynamics = NV.create_from_field_names(
    "StatesDynamics", _StatesVelocity.field_names + _StatesWeightTransfer.field_names
)

_StatesGlobal = NV.create_from_field_names(
    "StatesGlobal", _StatesDynamics.field_names + ("east_m", "north_m", "psi_rad")
)

_StatesPath = NV.create_from_field_names(
    "StatesPath", _StatesDynamics.field_names + ("s_m", "e_m", "dpsi_rad")
)

_Inputs = NV.create_from_field_names(
    "Inputs", ("delta_rad", "fx_kn")
)

_InputSlews = NV.create_from_field_names(
    "InputSlews", ("delta_dot_radps", "fx_dot_knps")
)


@cb.casadi_dataclass
class Model:
    """
    Single track vehicle model with basic inputs.
    """

    params: vp.VehicleParams
    """ Vehicle parameter class """

    f_tire: TireModel
    """ Tire parameter class for front axle """

    r_tire: TireModel
    """ Tire parameter class for rear axle """

    @cb.casadi_method((1,), num_outputs=2)
    def smooth_fx_distro_kn(self, fx_kn):
        """
        Calculate smooth Fx distribution.
        """

        df = self.params.drive_f_frac
        dr = self.params.drive_r_frac
        bf = self.params.brake_f_frac
        br = self.params.brake_r_frac

        diff_front = df - bf
        diff_rear = dr - br

        sum_front = df + bf
        sum_rear = dr + br

        f_frac = -0.5 * diff_front * ca.tanh(-2.0 * (fx_kn + 0.5)) + 0.5 * sum_front
        r_frac = 0.5 * diff_rear * ca.tanh(2.0 * (fx_kn + 0.5)) + 0.5 * sum_rear

        fxf_kn = f_frac * fx_kn
        fxr_kn = r_frac * fx_kn

        return fxf_kn, fxr_kn

    @cb.casadi_method((1, 1, 1, 1), num_outputs=2)
    def slip_angles(self, ux_mps, uy_mps, r_radps, delta_rad):
        """
        Calcuate slip angles.
        """

        alpha_f_rad = ca.arctan2(uy_mps + self.params.a_m * r_radps, ux_mps) - delta_rad
        alpha_r_rad = ca.arctan2(uy_mps - self.params.b_m * r_radps, ux_mps)

        return alpha_f_rad, alpha_r_rad
   

    @cb.casadi_method((1,), num_outputs=2)
    def normal_force_weight_transfer(self, dfz_long_kn):
        """
        Calculate axle normal loads.
        """

        fzf_kn = (self.params.wf_n / 1000.0) - dfz_long_kn
        fzr_kn = (self.params.wr_n / 1000.0) + dfz_long_kn

        return fzf_kn, fzr_kn

    @cb.casadi_method((_StatesDynamics.num_fields, _Inputs.num_fields))
    def temporal_dynamics_dynamics(self, states_vec, inputs_vec):
        """
        Calculate temporal velocity state derivatives.
        """
        
        states = _StatesDynamics.from_array(states_vec)
        inputs = _Inputs.from_array(inputs_vec)

        alpha_f_rad, alpha_r_rad = self.slip_angles(
            states.ux_mps,
            states.uy_mps,
            states.r_radps,
            inputs.delta_rad,
        )

        fzf_kn, fzr_kn = self.normal_force_weight_transfer(states.dfz_long_kn)
        fxf_kn, fxr_kn = self.smooth_fx_distro_kn(inputs.fx_kn)

        # Saturating the actual Fx reacted by the tires due to the wheel-lock limit (mu*Fz*cos(alpha))
        eps_f = 0.002
        eps_r = 0.002
        mu_f = self.f_tire.mu_none
        mu_r = self.r_tire.mu_none
        fxf_kn = (
            1
            / 2
            * (
                ca.sqrt(
                    (fxf_kn + mu_f * fzf_kn * ca.cos(alpha_f_rad)) ** 2
                    + eps_f**2
                )
                - ca.sqrt(
                    (fxf_kn - mu_f * fzf_kn * ca.cos(alpha_f_rad)) ** 2
                    + eps_f**2
                )
            )
        )
        fxr_kn = (
            1
            / 2
            * (
                ca.sqrt(
                    (fxr_kn + mu_r * fzr_kn * ca.cos(alpha_r_rad)) ** 2
                    + eps_r**2
                )
                - ca.sqrt(
                    (fxr_kn - mu_r * fzr_kn * ca.cos(alpha_r_rad)) ** 2
                    + eps_r**2
                )
            )
        )

        # After the Fx commands have been saturated, calculate the lateral force Fy
        fyf_kn = self.f_tire.calc_fy_kn(alpha_f_rad, fzf_kn, fxf_kn)
        fyr_kn = self.r_tire.calc_fy_kn(alpha_r_rad, fzr_kn, fxr_kn)


        # Convert tire forces from kN -> N
        fxf_n = fxf_kn * 1000.0
        fxr_n = fxr_kn * 1000.0
        fyf_n = fyf_kn * 1000.0
        fyr_n = fyr_kn * 1000.0

        # Get drag and resistance terms
        frr_n = -self.params.cd0_n
        faero_n = -(
            self.params.cd1_nspm * states.ux_mps
            + self.params.cd2_ns2pm2 * states.ux_mps**2
        )
        fd_n = frr_n + faero_n # Convention: negative Fd is along -x.

        # Evaluate state derivatives
        dux_mps2 = (1 / self.params.m_kg) * (
            fxf_n*ca.cos(inputs.delta_rad) - fyf_n*ca.sin(inputs.delta_rad) + fxr_n + fd_n
        ) + states.r_radps*states.uy_mps

        duy_mps2 = (1 / self.params.m_kg) * (
            fyf_n*ca.cos(inputs.delta_rad) + fxf_n*ca.sin(inputs.delta_rad) + fyr_n
        ) - states.r_radps*states.ux_mps

        dr_radps2 = (1 / self.params.iz_kgm2) * (
            self.params.a_m * fyf_n * ca.cos(inputs.delta_rad)
            + self.params.a_m * fxf_n * ca.sin(inputs.delta_rad)
            - self.params.b_m * fyr_n
        )

        # Calculate the total applied forces to the vehicle
        fx_kn = fxf_n*ca.cos(inputs.delta_rad) - fyf_n*ca.sin(inputs.delta_rad) + fxr_n + fd_n
        fy_kn = fyf_n*ca.cos(inputs.delta_rad) + fxf_n*ca.sin(inputs.delta_rad) + fyr_n

        # Evaluate some helper variables for weight transfer
        l_m = self.params.a_m + self.params.b_m # Wheelbase of the vehicle
        h_rc = (self.params.h_rc_f_m*self.params.b_m + self.params.h_rc_r_m*self.params.a_m)/l_m # Height of roll axis directly below center of mass
        h_l_m = self.params.h_com_m - h_rc # Vertical distance between center of mass and roll axis height
        t_m = 0.5*(self.params.track_f_m + self.params.track_r_m) # Average track of the vehicle

        # First order weight transfer dynamics
        ddfz_long_knps = 0 #1.0/self.params.tau_long_weight_transfer_s* (self.params.h_com_m/l_m * fx_kn - states.dfz_long_kn)
        ddfz_lat_knps  = 0 #1.0/self.params.tau_lat_weight_transfer_s* ((self.params.h_com_m/t_m + 9.81*h_l_m*self.params.roll_rate_radpmps2) * fy_kn - states.dfz_lat_kn)

        dstates_out = _StatesDynamics(
            ux_mps=dux_mps2,
            uy_mps=duy_mps2,
            r_radps=dr_radps2,
            dfz_long_kn=ddfz_long_knps,
            dfz_lat_kn=ddfz_lat_knps,
        )

        return dstates_out.to_array()

    @cb.casadi_method((_StatesGlobal.num_fields, _Inputs.num_fields))
    def temporal_global_dynamics(self, states_vec, inputs_vec):
        """
        Calculate temporal global dynamic state derivatives.
        """

        states = _StatesGlobal.from_array(states_vec)

        dyn_states = _StatesDynamics(
            ux_mps=states.ux_mps,
            uy_mps=states.uy_mps,
            r_radps=states.r_radps,
            dfz_long_kn=states.dfz_long_kn,
            dfz_lat_kn=states.dfz_lat_kn,
        )

        ddyn_states_vec = self.temporal_dynamics_dynamics(
            dyn_states.to_array(), inputs_vec
        )
        ddyn_states = _StatesDynamics.from_array(ddyn_states_vec)

        deast_mps = -dyn_states.ux_mps*ca.sin(states.psi_rad) - dyn_states.uy_mps*ca.cos(states.psi_rad)
        dnorth_mps = dyn_states.ux_mps*ca.cos(states.psi_rad) - dyn_states.uy_mps*ca.sin(states.psi_rad)
        dpsi_radps = dyn_states.r_radps

        dstates_out = _StatesGlobal(
            ux_mps=ddyn_states.ux_mps,
            uy_mps=ddyn_states.uy_mps,
            r_radps=ddyn_states.r_radps,
            dfz_long_kn=ddyn_states.dfz_long_kn,
            dfz_lat_kn=ddyn_states.dfz_lat_kn,
            east_m=deast_mps,
            north_m=dnorth_mps,
            psi_rad=dpsi_radps,
        )

        return dstates_out.to_array()

    @cb.casadi_method((_StatesPath.num_fields, _Inputs.num_fields, 1))
    def temporal_path_dynamics(self, states_vec, inputs_vec, k_psi_1pm):
        """
        Calculate temporal path dynamic state derivatives.
        """

        states = _StatesPath.from_array(states_vec)

        dyn_states = _StatesDynamics(
            ux_mps=states.ux_mps,
            uy_mps=states.uy_mps,
            r_radps=states.r_radps,
            dfz_long_kn=states.dfz_long_kn,
            dfz_lat_kn=states.dfz_lat_kn,
        )

        ddyn_states_vec = self.temporal_dynamics_dynamics(
            dyn_states.to_array(), inputs_vec
        )
        ddyn_states = _StatesDynamics.from_array(ddyn_states_vec)

        ds_mps = (dyn_states.ux_mps*ca.cos(states.dpsi_rad) - dyn_states.uy_mps*ca.sin(states.dpsi_rad))/(1 - states.e_m * k_psi_1pm)
        de_mps = dyn_states.ux_mps*ca.sin(states.dpsi_rad) + dyn_states.uy_mps*ca.cos(states.dpsi_rad)
        ddpsi_radps = dyn_states.r_radps - k_psi_1pm*ds_mps

        dstates_out = _StatesPath(
            ux_mps=ddyn_states.ux_mps,
            uy_mps=ddyn_states.uy_mps,
            r_radps=ddyn_states.r_radps,
            dfz_long_kn=ddyn_states.dfz_long_kn,
            dfz_lat_kn=ddyn_states.dfz_lat_kn,
            s_m=ds_mps,
            e_m=de_mps,
            dpsi_rad=ddpsi_radps,
        )

        return dstates_out.to_array()


    @cb.casadi_method((_StatesVelocity.num_fields, _Inputs.num_fields), num_outputs=1)
    def power_limit(self, states_vec, inputs_vec):

        states = _StatesVelocity.from_array(states_vec)
        inputs = _Inputs.from_array(inputs_vec)

        power_kw = ca.dot(inputs.fx_kn, states.ux_mps)
        power_limit_kw = self.params.p_eng_max_w/1000.0

        power_ub_kw = power_kw - power_limit_kw

        return power_ub_kw