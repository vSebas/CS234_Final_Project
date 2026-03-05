#!/usr/bin/env julia

using JSON3
using ExaModels
using MadNLP

const HAS_CUDA = try
    @eval using CUDA
    true
catch
    false
end

const HAS_MADNLPGPU = try
    @eval using MadNLPGPU
    true
catch
    false
end

const HAS_HYBRIDKKT = try
    @eval using HybridKKT
    true
catch
    false
end

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

function _to_backend(v, backend)
    if backend === nothing
        return v
    end
    if HAS_CUDA && backend isa CUDA.CUDABackend
        return CUDA.CuArray(v)
    end
    return v
end

function _build_and_solve(payload; force_cpu::Bool = false)
    dbg("[madnlp_exa] build:start force_cpu=$(force_cpu)")
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
    convergent_lap = Bool(opts["convergent_lap"])

    linear_solver_name = lowercase(String(get(opts, "linear_solver", "")))
    kkt_name = lowercase(String(get(opts, "kkt_system", "")))
    require_gpu = get(ENV, "MADNLP_REQUIRE_GPU", "0") == "1"
    gpu_requested = !force_cpu && (
        require_gpu ||
        linear_solver_name in ("cudss", "lapackcuda", "lapack_cudasolver") ||
        kkt_name in ("hybrid", "hybrid_condensed", "hybridcondensedkkt")
    )
    if gpu_requested && !HAS_CUDA
        error("GPU solve requested, but CUDA.jl is unavailable in this Julia project.")
    end
    if gpu_requested && !HAS_MADNLPGPU
        error("GPU solve requested, but MadNLPGPU.jl is unavailable in this Julia project.")
    end
    backend = gpu_requested ? CUDA.CUDABackend() : nothing

    ux_init = max(5.0, ux_min + 1.0)
    X_start = zeros(Float64, nx, n_nodes)
    X_start[1, :] .= ux_init
    X_start[6, :] .= cumsum(fill(ds_m / ux_init, n_nodes)) .- ds_m / ux_init
    U_start = zeros(Float64, nu, n_nodes)
    U_start[2, :] .= 0.5
    if X0 !== nothing
        X_start .= X0
    end
    if U0 !== nothing
        U_start .= U0
    end

    X_l = fill(-Inf, nx, n_nodes)
    X_u = fill(Inf, nx, n_nodes)
    U_l = fill(-Inf, nu, n_nodes)
    U_u = fill(Inf, nu, n_nodes)
    max_delta_rad = Float64(p["max_delta_deg"]) * pi / 180.0
    min_fx_kn = Float64(p["min_fx_kn"])
    max_fx_kn = Float64(p["max_fx_kn"])
    for k in 1:n_nodes
        X_l[1, k] = ux_min
        if ux_max !== nothing
            X_u[1, k] = ux_max
        end
        X_l[7, k] = -track_hw[k] + track_buffer_m
        X_u[7, k] = track_hw[k] - track_buffer_m
        U_l[1, k] = -max_delta_rad
        U_u[1, k] = max_delta_rad
        U_l[2, k] = min_fx_kn
        U_u[2, k] = max_fx_kn
    end

    c = ExaModels.ExaCore(Float64; backend=backend)
    X = ExaModels.variable(
        c,
        1:nx,
        1:n_nodes;
        start=_to_backend(X_start, backend),
        lvar=_to_backend(X_l, backend),
        uvar=_to_backend(X_u, backend),
    )
    U = ExaModels.variable(
        c,
        1:nu,
        1:n_nodes;
        start=_to_backend(U_start, backend),
        lvar=_to_backend(U_l, backend),
        uvar=_to_backend(U_u, backend),
    )

    function dxdt(x, u, kpsi, th, ph)
        ux = x[1]
        uy = x[2]
        r = x[3]
        dfz_long = x[4]
        e = x[7]
        dpsi = x[8]
        delta = u[1]
        fx = u[2]

        alpha_f = atan((uy + Float64(p["a_m"]) * r) / (ux + 1e-6)) - delta
        alpha_r = atan((uy - Float64(p["b_m"]) * r) / (ux + 1e-6))

        l_m = Float64(p["a_m"]) + Float64(p["b_m"])
        wf_n = Float64(p["m_kg"]) * G_MPS2 * (Float64(p["b_m"]) / l_m)
        wr_n = Float64(p["m_kg"]) * G_MPS2 * (Float64(p["a_m"]) / l_m)
        fzf_kn = wf_n / 1000.0 - dfz_long
        fzr_kn = wr_n / 1000.0 + dfz_long

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

        fxf_n = fxf_kn * 1000.0
        fxr_n = fxr_kn * 1000.0
        fyf_n = fyf_kn * 1000.0
        fyr_n = fyr_kn * 1000.0

        frr_n = -Float64(p["cd0_n"])
        faero_n = -(Float64(p["cd1_nspm"]) * ux + Float64(p["cd2_ns2pm2"]) * ux^2)
        f_grade_n = -Float64(p["m_kg"]) * G_MPS2 * sin(th)
        f_bank_n = -Float64(p["m_kg"]) * G_MPS2 * cos(th) * sin(ph)
        fd_n = frr_n + faero_n + f_grade_n

        ax = (1.0 / Float64(p["m_kg"])) * (fxf_n * cos(delta) - fyf_n * sin(delta) + fxr_n + fd_n)
        ay = (1.0 / Float64(p["m_kg"])) * (fyf_n * cos(delta) + fxf_n * sin(delta) + fyr_n + f_bank_n)

        dux = ax + r * uy
        duy = ay - r * ux
        dr = (1.0 / Float64(p["iz_kgm2"])) * (
            Float64(p["a_m"]) * fyf_n * cos(delta) +
            Float64(p["a_m"]) * fxf_n * sin(delta) -
            Float64(p["b_m"]) * fyr_n
        )

        one_minus = 1.0 - e * kpsi
        sdot = (ux * cos(dpsi) - uy * sin(dpsi)) / one_minus
        de = ux * sin(dpsi) + uy * cos(dpsi)
        ddpsi = r - kpsi * sdot
        return (dux, duy, dr, 0.0, 0.0, 1.0, de, ddpsi, sdot)
    end

    for k in 1:N
        xk = [X[i, k] for i in 1:nx]
        xkp1 = [X[i, k + 1] for i in 1:nx]
        uk = [U[i, k] for i in 1:nu]
        ukp1 = [U[i, k + 1] for i in 1:nu]
        fk = dxdt(xk, uk, kappa[k], theta[k], phi[k])
        fkp1 = dxdt(xkp1, ukp1, kappa[k + 1], theta[k + 1], phi[k + 1])
        sdot_k = fk[9]
        sdot_kp1 = fkp1[9]
        for i in 1:nx
            ExaModels.constraint(c, X[i, k + 1] - X[i, k] - 0.5 * ds_m * (fk[i] / sdot_k + fkp1[i] / sdot_kp1))
        end
    end

    for k in 1:n_nodes
        one_minus = 1.0 - kappa[k] * X[7, k]
        ExaModels.constraint(c, one_minus; lcon=eps_kappa)
        sdot_k = (X[1, k] * cos(X[8, k]) - X[2, k] * sin(X[8, k])) / one_minus
        ExaModels.constraint(c, sdot_k; lcon=eps_s)
    end

    active_obs = Tuple{Int,Int}[]
    for k in 1:n_nodes
        for j in 1:n_obs
            if abs(wrap_s_dist(s_grid[k], obs_s_center[j], length_m)) <= obstacle_window_m
                push!(active_obs, (k, j))
            end
        end
    end
    for (k, j) in active_obs
        req_r = obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m
        posE_k = posE_cl[k] - X[7, k] * sin(psi_cl[k])
        posN_k = posN_cl[k] + X[7, k] * cos(psi_cl[k])
        d2 = (posE_k - obs_east[j])^2 + (posN_k - obs_north[j])^2
        ExaModels.constraint(c, d2 - req_r^2; lcon=0.0)
    end

    ExaModels.constraint(c, X[6, 1]; lcon=0.0, ucon=0.0)
    if convergent_lap
        for i in (1, 2, 3, 4, 5, 7, 8)
            ExaModels.constraint(c, X[i, 1] - X[i, n_nodes])
        end
        if enforce_periodic_controls
            for i in 1:nu
                ExaModels.constraint(c, U[i, 1] - U[i, n_nodes])
            end
        end
    end

    ExaModels.objective(c, X[6, n_nodes])
    ExaModels.objective(
        c,
        lambda_u * ((U[1, k + 1] - U[1, k])^2 + (U[2, k + 1] - U[2, k])^2) for k in 1:N
    )

    nlp = ExaModels.ExaModel(c)
    kwargs = Dict{Symbol,Any}()
    kwargs[:print_level] = verbose ? MadNLP.INFO : MadNLP.ERROR
    if haskey(opts, "tol")
        kwargs[:tol] = Float64(opts["tol"])
    end
    if haskey(opts, "acceptable_tol")
        kwargs[:acceptable_tol] = Float64(opts["acceptable_tol"])
    end
    if haskey(opts, "max_iter")
        kwargs[:max_iter] = Int(opts["max_iter"])
    end
    if haskey(opts, "max_cpu_time")
        kwargs[:max_wall_time] = Float64(opts["max_cpu_time"])
    end

    gpu_active = gpu_requested
    if gpu_requested
        kwargs[:equality_treatment] = MadNLP.EnforceEquality
        kwargs[:fixed_variable_treatment] = MadNLP.MakeParameter
        if linear_solver_name == "cudss"
            kwargs[:linear_solver] = MadNLPGPU.CUDSSSolver
            kwargs[:cudss_algorithm] = MadNLP.LDL
        elseif linear_solver_name in ("lapackcuda", "lapack_cudasolver")
            kwargs[:linear_solver] = MadNLPGPU.LapackCUDASolver
        end
        if isempty(kkt_name)
            if HAS_HYBRIDKKT
                kwargs[:kkt_system] = HybridKKT.HybridCondensedKKTSystem
            else
                error("GPU solve requires HybridKKT.jl in this environment. Install HybridKKT or set MADNLP_REQUIRE_GPU=0.")
            end
        elseif kkt_name in ("hybrid", "hybrid_condensed", "hybridcondensedkkt")
            if HAS_HYBRIDKKT
                kwargs[:kkt_system] = HybridKKT.HybridCondensedKKTSystem
            else
                error("MADNLP_KKT_SYSTEM=$(kkt_name) requested, but HybridKKT.jl is unavailable.")
            end
        elseif kkt_name in ("sparse_condensed", "sparsecondensedkkt", "condensed")
            error("MADNLP_KKT_SYSTEM=$(kkt_name) is not supported with GPU equality constraints in this setup. Use 'hybrid'.")
        elseif kkt_name in ("sparse", "sparsekkt")
            error("MADNLP_KKT_SYSTEM=$(kkt_name) is not supported with current MadNLPGPU build (SparseKKT GPU incompatibility). Use 'hybrid'.")
        end
    end

    solver = MadNLP.MadNLPSolver(nlp; kwargs...)
    stats = MadNLP.solve!(solver)
    success = stats.status == MadNLP.SOLVE_SUCCEEDED || stats.status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL

    X_out = Array(ExaModels.solution(stats, X))
    U_out = Array(ExaModels.solution(stats, U))
    iter = Int(stats.iter)
    solve_time = Float64(stats.counters.total_time)
    cost = Float64(stats.objective)
    if !isfinite(cost)
        cost = 1e30
    end

    min_clearance = 1e30
    if n_obs > 0
        for (k, j) in active_obs
            req_r = obs_r_tilde[j] + obstacle_clearance_m + vehicle_radius_m
            posE_k = posE_cl[k] - X_out[7, k] * sin(psi_cl[k])
            posN_k = posN_cl[k] + X_out[7, k] * cos(psi_cl[k])
            d = sqrt((posE_k - obs_east[j])^2 + (posN_k - obs_north[j])^2)
            cmin = d - req_r
            if cmin < min_clearance
                min_clearance = cmin
            end
        end
    end

    X_json = [[X_out[i, j] for j in 1:n_nodes] for i in 1:nx]
    U_json = [[U_out[i, j] for j in 1:n_nodes] for i in 1:nu]

    return Dict(
        "success" => success,
        "status" => string(stats.status),
        "mode_used" => dynamics_mode,
        "cost" => cost,
        "iterations" => iter,
        "solve_time" => solve_time,
        "max_obstacle_slack" => 0.0,
        "min_obstacle_clearance" => Float64(min_clearance),
        "gpu_requested" => gpu_requested,
        "gpu_active" => gpu_active,
        "X" => X_json,
        "U" => U_json,
        "backend" => gpu_requested ? "madnlp_examodels_gpu" : "madnlp_examodels_cpu",
    )
end

function solve_problem(payload)
    require_gpu = get(ENV, "MADNLP_REQUIRE_GPU", "0") == "1"
    allow_gpu_fallback = get(ENV, "MADNLP_GPU_FALLBACK", "1") == "1"
    try
        return _build_and_solve(payload; force_cpu=false)
    catch err
        if !require_gpu && allow_gpu_fallback
            dbg("[madnlp_exa] GPU/primary solve failed; retrying CPU fallback")
            cpu_res = _build_and_solve(payload; force_cpu=true)
            cpu_res["gpu_active"] = false
            return cpu_res
        end
        rethrow(err)
    end
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
