#!/usr/bin/env julia

using JSON3
using JuMP
using MadNLP
const MOI = JuMP.MOI

const G_MPS2 = 9.81

const DEBUG_LOG = get(ENV, "MADNLP_EXA_DEBUG", "0") == "1"

function dbg(msg::AbstractString)
    if DEBUG_LOG
        println(stderr, msg)
        flush(stderr)
    end
end

function smooth_abs(x; eps=1e-6)
    return sqrt(x * x + eps * eps)
end

function smooth_clamp(x, lb, ub; eps=1e-6)
    x_lb = 0.5 * (x + lb + sqrt((x - lb)^2 + eps^2))
    return 0.5 * (x_lb + ub - sqrt((x_lb - ub)^2 + eps^2))
end

function fiala_fy(alpha, fz_kn, fx_kn, tire)
    c0 = Float64(tire["c0_alpha_nprad"])
    c1 = Float64(tire["c1_alpha_1prad"])
    mu = Float64(tire["mu_none"])
    xi = Float64(get(tire, "fy_xi", 0.95))
    rho = Float64(get(tire, "max_allowed_fx_frac", 0.99))

    c_alpha = c0 / 1000.0 + c1 * fz_kn
    max_fx_abs = mu * fz_kn * cos(alpha)
    fx_c = smooth_clamp(fx_kn, -max_fx_abs, max_fx_abs)
    fy_max = sqrt((mu * fz_kn)^2 - (rho * fx_c)^2 + 1e-9)
    alpha_slide = atan((3.0 * fy_max * xi) / (c_alpha + 1e-9))
    tan_a = tan(alpha)

    fy_unsat = -(c_alpha * tan_a) +
               ((c_alpha^2) / (3.0 * fy_max) * tan_a * smooth_abs(tan_a)) -
               ((c_alpha^3) / (27.0 * fy_max^2) * tan_a^3)
    fy_sat = -c_alpha * (1.0 - 2.0 * xi + xi^2) * tan_a -
             fy_max * (3.0 * xi^2 - 2.0 * xi^3) * tanh(50.0 * alpha)
    w_unsat = 0.5 * (1.0 + tanh(20.0 * (alpha_slide - smooth_abs(alpha))))
    return w_unsat * fy_unsat + (1.0 - w_unsat) * fy_sat
end

function wrap_s_dist(sa::Float64, sb::Float64, length_m::Float64)
    d = mod(sa - sb, length_m)
    if d > 0.5 * length_m
        d -= length_m
    end
    return d
end

function parse_args()
    req = nothing
    res = nothing
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--request" && i < length(ARGS)
            req = ARGS[i + 1]
            i += 2
        elseif a == "--response" && i < length(ARGS)
            res = ARGS[i + 1]
            i += 2
        else
            i += 1
        end
    end
    if req === nothing || res === nothing
        error("Usage: madnlp_exa_solver.jl --request <request.json> --response <response.json>")
    end
    return req, res
end

function solve_problem(payload)
    dbg("[madnlp_exa] solve_problem:start")
    to_f64_vec(a) = [Float64(v) for v in a]
    function to_f64_mat(a)
        rows = length(a)
        cols = rows == 0 ? 0 : length(a[1])
        m = Array{Float64}(undef, rows, cols)
        for i in 1:rows
            for j in 1:cols
                m[i, j] = Float64(a[i][j])
            end
        end
        return m
    end

    N = Int(payload["N"])
    ds_m = Float64(payload["ds_m"])
    n_nodes = N + 1
    nx = Int(payload["vehicle"]["state_dim"])
    nu = Int(payload["vehicle"]["control_dim"])
    p = payload["vehicle"]["params"]
    tf = payload["vehicle"]["front_tire"]
    tr = payload["vehicle"]["rear_tire"]
    opts = payload["options"]

    kappa = to_f64_vec(payload["world"]["kappa_radpm"])
    s_grid = to_f64_vec(payload["world"]["s_grid_m"])
    theta = to_f64_vec(payload["world"]["grade_rad"])
    phi = to_f64_vec(payload["world"]["bank_rad"])
    track_hw = to_f64_vec(payload["world"]["track_half_width_m"])
    psi_cl = to_f64_vec(payload["world"]["psi_cl_rad"])
    posE_cl = to_f64_vec(payload["world"]["posE_cl_m"])
    posN_cl = to_f64_vec(payload["world"]["posN_cl_m"])

    X0 = payload["X_init"] === nothing ? nothing : to_f64_mat(payload["X_init"])
    U0 = payload["U_init"] === nothing ? nothing : to_f64_mat(payload["U_init"])

    obs_res = payload["obstacle_resolved"]
    obs_east = to_f64_vec(obs_res["east_m"])
    obs_north = to_f64_vec(obs_res["north_m"])
    obs_s_center = to_f64_vec(obs_res["s_center_m"])
    obs_r_tilde = to_f64_vec(obs_res["r_tilde_m"])
    n_obs = length(obs_east)

    ux_min = Float64(opts["ux_min"])
    ux_max = opts["ux_max"] === nothing ? nothing : Float64(opts["ux_max"])
    track_buffer_m = Float64(opts["track_buffer_m"])
    eps_s = Float64(opts["eps_s"])
    eps_kappa = Float64(opts["eps_kappa"])
    lambda_u = Float64(opts["lambda_u"])
    verbose = Bool(opts["verbose"])
    enforce_periodic_controls = Bool(get(opts, "enforce_periodic_controls", false))
    dynamics_mode = String(get(opts, "dynamics_mode", "simple"))
    use_full_dynamics = dynamics_mode == "full"
    use_fiala = dynamics_mode == "fiala"
    obstacle_window_m = Float64(get(opts, "obstacle_window_m", 30.0))
    obstacle_clearance_m = Float64(get(opts, "obstacle_clearance_m", 0.0))
    vehicle_radius_m = Float64(get(opts, "vehicle_radius_m", 0.0))
    length_m = Float64(payload["world"]["length_m"])

    model = Model()
    set_optimizer(model, MadNLP.Optimizer)
    set_optimizer_attribute(model, "print_level", verbose ? MadNLP.INFO : MadNLP.ERROR)
    if haskey(opts, "tol")
        set_optimizer_attribute(model, "tol", Float64(opts["tol"]))
    end
    if haskey(opts, "acceptable_tol")
        set_optimizer_attribute(model, "acceptable_tol", Float64(opts["acceptable_tol"]))
    end
    if haskey(opts, "max_iter")
        set_optimizer_attribute(model, "max_iter", Int(opts["max_iter"]))
    end
    if haskey(opts, "max_cpu_time")
        set_optimizer_attribute(model, "max_cpu_time", Float64(opts["max_cpu_time"]))
    end

    @variable(model, X[1:nx, 1:n_nodes])
    @variable(model, U[1:nu, 1:n_nodes])

    for i in 1:nx, k in 1:n_nodes
        if X0 !== nothing
            set_start_value(X[i, k], X0[i, k])
        end
    end
    for i in 1:nu, k in 1:n_nodes
        if U0 !== nothing
            set_start_value(U[i, k], U0[i, k])
        end
    end

    if X0 === nothing
        ux_init = max(5.0, ux_min + 1.0)
        for k in 1:n_nodes
            set_start_value(X[1, k], ux_init)
            set_start_value(X[2, k], 0.0)
            set_start_value(X[3, k], 0.0)
            set_start_value(X[4, k], 0.0)
            set_start_value(X[5, k], 0.0)
            set_start_value(X[6, k], (k - 1) * ds_m / ux_init)
            set_start_value(X[7, k], 0.0)
            set_start_value(X[8, k], 0.0)
        end
    end
    if U0 === nothing
        for k in 1:n_nodes
            set_start_value(U[1, k], 0.0)
            set_start_value(U[2, k], 0.5)
        end
    end

    max_delta_rad = Float64(p["max_delta_deg"]) * pi / 180.0
    min_fx_kn = Float64(p["min_fx_kn"])
    max_fx_kn = Float64(p["max_fx_kn"])

    @objective(
        model,
        Min,
        X[6, n_nodes] + lambda_u * sum(
            (U[1, k + 1] - U[1, k])^2 + (U[2, k + 1] - U[2, k])^2 for k in 1:N
        )
    )
    dbg("[madnlp_exa] model_constructed N=$(N)")

    # Dynamics and path constraints.
    function dxdt(x, u, kpsi, th, ph)
        ux = x[1]
        uy = x[2]
        r = x[3]
        dfz_long = x[4]
        e = x[7]
        dpsi = x[8]
        delta = u[1]
        fx = u[2]

        # Slip angles
        alpha_f = atan((uy + Float64(p["a_m"]) * r) / (ux + 1e-6)) - delta
        alpha_r = atan((uy - Float64(p["b_m"]) * r) / (ux + 1e-6))

        # Normal loads
        l_m = Float64(p["a_m"]) + Float64(p["b_m"])
        wf_n = Float64(p["m_kg"]) * G_MPS2 * (Float64(p["b_m"]) / l_m)
        wr_n = Float64(p["m_kg"]) * G_MPS2 * (Float64(p["a_m"]) / l_m)
        fzf_kn = wf_n / 1000.0 - dfz_long
        fzr_kn = wr_n / 1000.0 + dfz_long

        # Lightweight force model for first MadNLP backend pass.
        fxf_kn = 0.5 * fx
        fxr_kn = 0.5 * fx
        if use_full_dynamics || use_fiala
            df = Float64(p["drive_f_frac"])
            drf = Float64(p["drive_r_frac"])
            bf = Float64(p["brake_f_frac"])
            br = Float64(p["brake_r_frac"])
            diff_front = df - bf
            diff_rear = drf - br
            sum_front = df + bf
            sum_rear = drf + br
            f_frac = -0.5 * diff_front * tanh(-2.0 * (fx + 0.5)) + 0.5 * sum_front
            r_frac = 0.5 * diff_rear * tanh(2.0 * (fx + 0.5)) + 0.5 * sum_rear
            fxf_kn = f_frac * fx
            fxr_kn = r_frac * fx
            mu_f = Float64(tf["mu_none"])
            mu_r = Float64(tr["mu_none"])
            fxf_max = mu_f * fzf_kn * cos(alpha_f)
            fxr_max = mu_r * fzr_kn * cos(alpha_r)
            fxf_kn = smooth_clamp(fxf_kn, -fxf_max, fxf_max)
            fxr_kn = smooth_clamp(fxr_kn, -fxr_max, fxr_max)
        end
        if use_fiala
            fyf_kn = fiala_fy(alpha_f, fzf_kn, fxf_kn, tf)
            fyr_kn = fiala_fy(alpha_r, fzr_kn, fxr_kn, tr)
        else
            c_af = Float64(tf["c0_alpha_nprad"]) / 1000.0 + Float64(tf["c1_alpha_1prad"]) * fzf_kn
            c_ar = Float64(tr["c0_alpha_nprad"]) / 1000.0 + Float64(tr["c1_alpha_1prad"]) * fzr_kn
            fyf_kn = -c_af * alpha_f
            fyr_kn = -c_ar * alpha_r
        end

        # N units
        fxf_n = fxf_kn * 1000.0
        fxr_n = fxr_kn * 1000.0
        fyf_n = fyf_kn * 1000.0
        fyr_n = fyr_kn * 1000.0

        # Drag and road forces
        frr_n = -Float64(p["cd0_n"])
        faero_n = -(Float64(p["cd1_nspm"]) * ux + Float64(p["cd2_ns2pm2"]) * ux^2)
        f_grade_n = -Float64(p["m_kg"]) * G_MPS2 * sin(th)
        f_bank_n = -Float64(p["m_kg"]) * G_MPS2 * cos(th) * sin(ph)
        fd_n = frr_n + faero_n + f_grade_n

        ax = (1.0 / Float64(p["m_kg"])) *
             (fxf_n * cos(delta) - fyf_n * sin(delta) + fxr_n + fd_n)
        ay = (1.0 / Float64(p["m_kg"])) *
             (fyf_n * cos(delta) + fxf_n * sin(delta) + fyr_n + f_bank_n)

        dux = ax + r * uy
        duy = ay - r * ux
        dr = (1.0 / Float64(p["iz_kgm2"])) *
             (Float64(p["a_m"]) * fyf_n * cos(delta) +
              Float64(p["a_m"]) * fxf_n * sin(delta) -
              Float64(p["b_m"]) * fyr_n)

        one_minus = 1.0 - e * kpsi
        sdot = (ux * cos(dpsi) - uy * sin(dpsi)) / one_minus
        de = ux * sin(dpsi) + uy * cos(dpsi)
        ddpsi = r - kpsi * sdot

        return [dux, duy, dr, 0.0, 0.0, 1.0, de, ddpsi], sdot
    end

    for k in 1:N
        xk = X[:, k]
        xkp1 = X[:, k + 1]
        uk = U[:, k]
        ukp1 = U[:, k + 1]
        fk, sdot_k = dxdt(xk, uk, kappa[k], theta[k], phi[k])
        fkp1, sdot_kp1 = dxdt(xkp1, ukp1, kappa[k + 1], theta[k + 1], phi[k + 1])
        for i in 1:nx
            @constraint(model, X[i, k + 1] == X[i, k] + 0.5 * ds_m * (fk[i] / sdot_k + fkp1[i] / sdot_kp1))
        end
    end

    for k in 1:n_nodes
        @constraint(model, X[1, k] >= ux_min)
        if ux_max !== nothing
            @constraint(model, X[1, k] <= ux_max)
        end
        one_minus = 1.0 - kappa[k] * X[7, k]
        @constraint(model, one_minus >= eps_kappa)
        @constraint(model, (X[1, k] * cos(X[8, k]) - X[2, k] * sin(X[8, k])) / one_minus >= eps_s)
        @constraint(model, X[7, k] >= -track_hw[k] + track_buffer_m)
        @constraint(model, X[7, k] <= track_hw[k] - track_buffer_m)
        @constraint(model, U[1, k] >= -max_delta_rad)
        @constraint(model, U[1, k] <= max_delta_rad)
        @constraint(model, U[2, k] >= min_fx_kn)
        @constraint(model, U[2, k] <= max_fx_kn)

        if n_obs > 0
            posE_k = posE_cl[k] - X[7, k] * sin(psi_cl[k])
            posN_k = posN_cl[k] + X[7, k] * cos(psi_cl[k])
            for j in 1:n_obs
                if abs(wrap_s_dist(s_grid[k], obs_s_center[j], length_m)) <= obstacle_window_m
                    req_r = obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m
                    @constraint(
                        model,
                        (posE_k - obs_east[j])^2 + (posN_k - obs_north[j])^2 >= req_r^2
                    )
                end
            end
        end
    end

    @constraint(model, X[6, 1] == 0.0)
    if Bool(opts["convergent_lap"])
        for i in (1, 2, 3, 4, 5, 7, 8)
            @constraint(model, X[i, 1] == X[i, n_nodes])
        end
        if enforce_periodic_controls
            for i in 1:nu
                @constraint(model, U[i, 1] == U[i, n_nodes])
            end
        end
    end

    optimize!(model)
    dbg("[madnlp_exa] optimize_done")
    term = termination_status(model)
    success = term == MOI.LOCALLY_SOLVED || term == MOI.OPTIMAL

    X_out = Array{Float64}(undef, nx, n_nodes)
    U_out = Array{Float64}(undef, nu, n_nodes)
    if has_values(model)
        for i in 1:nx, k in 1:n_nodes
            X_out[i, k] = value(X[i, k])
        end
        for i in 1:nu, k in 1:n_nodes
            U_out[i, k] = value(U[i, k])
        end
    else
        for i in 1:nx, k in 1:n_nodes
            X_out[i, k] = start_value(X[i, k]) === nothing ? 0.0 : Float64(start_value(X[i, k]))
        end
        for i in 1:nu, k in 1:n_nodes
            U_out[i, k] = start_value(U[i, k]) === nothing ? 0.0 : Float64(start_value(U[i, k]))
        end
    end

    iter = -1
    try
        iter = Int(round(MOI.get(model, MOI.BarrierIterations())))
    catch
    end
    solve_time = 0.0
    try
        solve_time = Float64(MOI.get(model, MOI.SolveTimeSec()))
    catch
    end
    cost = has_values(model) ? objective_value(model) : Inf
    if !isfinite(cost)
        cost = 1e30
    end
    min_clearance = 1e30
    if n_obs > 0
        for k in 1:n_nodes
            posE_k = posE_cl[k] - X_out[7, k] * sin(psi_cl[k])
            posN_k = posN_cl[k] + X_out[7, k] * cos(psi_cl[k])
            for j in 1:n_obs
                if abs(wrap_s_dist(s_grid[k], obs_s_center[j], length_m)) <= obstacle_window_m
                    req_r = obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m
                    d = sqrt((posE_k - obs_east[j])^2 + (posN_k - obs_north[j])^2)
                    c = d - req_r
                    if c < min_clearance
                        min_clearance = c
                    end
                end
            end
        end
    end
    X_json = [[X_out[i, j] for j in 1:n_nodes] for i in 1:nx]
    U_json = [[U_out[i, j] for j in 1:n_nodes] for i in 1:nu]

    return Dict(
        "success" => success,
        "status" => string(term),
        "mode_used" => dynamics_mode,
        "cost" => Float64(cost),
        "iterations" => Int(iter),
        "solve_time" => Float64(solve_time),
        "max_obstacle_slack" => 0.0,
        "min_obstacle_clearance" => Float64(min_clearance),
        "X" => X_json,
        "U" => U_json,
        "backend" => "madnlp_jump",
    )
end

function main()
    req_path, res_path = parse_args()
    dbg("[madnlp_exa] reading_request")
    payload = JSON3.read(read(req_path, String))
    result = solve_problem(payload)
    dbg("[madnlp_exa] writing_response")
    open(res_path, "w") do io
        write(io, JSON3.write(result))
    end
end

main()
