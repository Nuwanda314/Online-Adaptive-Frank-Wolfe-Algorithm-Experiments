#--------------------------------------------------------------------------------#
# PACKAGES                                                                       #
#--------------------------------------------------------------------------------#

using Pkg
Pkg.add(["Distributions", "ForwardDiff", "JuMP", "LinearAlgebra", "Mosek", 
    "MosekTools", "Plots", "Printf", "Random", "Statistics"])

using Distributions, ForwardDiff, JuMP, LinearAlgebra, Mosek, MosekTools, Plots, 
    Printf, Random, Statistics

#--------------------------------------------------------------------------------#
# FUNCTIONS                                                                      #
#--------------------------------------------------------------------------------#

struct linear_quadratic_gaussian_system
    A::Vector{Matrix{Float64}}
    B::Vector{Matrix{Float64}}
    C::Vector{Matrix{Float64}}
    Q::Vector{Matrix{Float64}}
    R::Vector{Matrix{Float64}}

    X_initial::Matrix{Float64}
    W::Vector{Matrix{Float64}}
    V::Vector{Matrix{Float64}}

    T::Int64

    n::Int64
    m::Int64
    p::Int64

    X_initial_root::Matrix{Float64}
    W_root::Vector{Matrix{Float64}}
    V_root::Vector{Matrix{Float64}}

    P::Vector{Matrix{Float64}}
end

#--------------------------------------------------------------------------------#

function linear_quadratic_gaussian_system(
    A::Vector{Matrix{Float64}}, 
    B::Vector{Matrix{Float64}}, 
    C::Vector{Matrix{Float64}}, 
    Q::Vector{Matrix{Float64}}, 
    R::Vector{Matrix{Float64}}, 
    X_initial::Matrix{Float64}, 
    W::Vector{Matrix{Float64}}, 
    V::Vector{Matrix{Float64}}, 
    T::Int64
    )::linear_quadratic_gaussian_system

    n = size(A[1])[1]
    m = size(B[1])[2]
    p = size(C[1])[1]

    X_initial_root = √(X_initial)
    W_root = [√(W[t]) for t = 1:T]
    V_root = [√(V[t]) for t = 1:(T + 1)]

    P = Vector{Matrix{Float64}}(undef, T + 1)

    P[T + 1] = Q[T + 1]
    for t = T:-1:1
        c₁ = B[t]' * P[t + 1]
        c₂ = c₁ * A[t]
        c₃ = (R[t] + c₁ * B[t]) \ I
        
        P[t] = A[t]' * P[t + 1] * A[t] + Q[t] - c₂' * c₃ * c₂
    end

    return linear_quadratic_gaussian_system(A, B, C, Q, R, X_initial, W, V, T, n,
        m, p, X_initial_root, W_root, V_root, P)
end

#--------------------------------------------------------------------------------#

function generate_linear_quadratic_gaussian_system(
    n::Int64, 
    T::Int64
    )::linear_quadratic_gaussian_system

    A = diagm(ones(n))
    for i = 1:(n - 1)
        A[i, i + 1] = 1.0
    end
    A = [A .+ diagm(sin(k) * ones(n)) for k = 1:T]

    B = [1.0 0.0; 0.0 1.0]
    B = mod(n, 2) == 0 ? repeat(B, n ÷ 2, 1) : vcat(repeat(B, n ÷ 2, 1), [1.0 0.0])
    B = [B for _ = 1:T]

    d = div(n, 2)
    C = n - 2d == 0 ? [diagm(ones(d)) zeros(d, n - d)] : [diagm(ones(d + 1)) zeros(d + 1, n - d - 1)]
    C = [C for _ = 1:(T + 1)]

    Q = 0.1 * diagm(ones(n))
    Q = [[Q / k for k = 1:T]..., diagm(0.025 * ones(n))]

    R = 0.25 * diagm(ones(2))
    R = [R for _ = 1:T]

    X_initial = 10 * diagm(ones(n))

    W = [5e-2 * abs(sin(k)) * diagm(ones(n)) for k = 1:T]

    V = n - 2d == 0 ? 1e-1 * diagm(ones(d)) : 1e-2 * diagm(ones(d + 1))
    V = [abs(cos(k)) * V for k = 1:(T + 1)]

    return linear_quadratic_gaussian_system(A, B, C, Q, R, X_initial, W, V, T)
end

#--------------------------------------------------------------------------------#

function f(
    LQGS::linear_quadratic_gaussian_system, 
    Z::Vector{Matrix{Float64}}
    )::Float64

    c₁ = LQGS.C[1] * Z[1]
    c₂ = (c₁ * LQGS.C[1]' + Z[LQGS.T + 2]) \ I
    Σ = Z[1] - c₁' * c₂ * c₁

    J = tr((LQGS.Q[1] - LQGS.P[1]) * Σ) + tr(LQGS.P[1] * Z[1])

    for t = 1:LQGS.T
        Σ_predicted = LQGS.A[t] * Σ * LQGS.A[t]' + Z[t + 1]
        c₁ = LQGS.C[t + 1] * Σ_predicted
        c₂ = (c₁ * LQGS.C[t + 1]' + Z[t + LQGS.T + 2]) \ I
        Σ = Σ_predicted - c₁' * c₂ * c₁

        J += tr((LQGS.Q[t + 1] - LQGS.P[t + 1]) * Σ) + 
            tr(LQGS.P[t + 1] * Σ_predicted)
    end

    return J
end

#--------------------------------------------------------------------------------#

function ∇f(
    LQGS::linear_quadratic_gaussian_system, 
    Z::Vector{Matrix{Float64}}, 
    k::Int64
    )::Matrix{Float64}

    if k == 1
        function ∇f_X(X)
            c₁ = LQGS.C[1] * X
            c₂ = (c₁ * LQGS.C[1]' + Z[LQGS.T + 2]) \ I
            Σ = X - c₁' * c₂ * c₁

            J = tr((LQGS.Q[1] - LQGS.P[1]) * Σ) + tr(LQGS.P[1] * X)

            for t = 1:LQGS.T
                Σ_predicted = LQGS.A[t] * Σ * LQGS.A[t]' + Z[t + 1]
                c₁ = LQGS.C[t + 1] * Σ_predicted
                c₂ = (c₁ * LQGS.C[t + 1]' + Z[t + LQGS.T + 2]) \ I
                Σ = Σ_predicted - c₁' * c₂ * c₁

                J += tr((LQGS.Q[t + 1] - LQGS.P[t + 1]) * Σ) + 
                    tr(LQGS.P[t + 1] * Σ_predicted)
            end

            return J
        end

        return ForwardDiff.gradient(∇f_X, Z[1])

    elseif k ∈ 2:(LQGS.T + 1)
        function ∇f_W(W)
            c = LQGS.C[1] * Z[1]
            Σ = Z[1] - c' * inv(c * LQGS.C[1]' + Z[LQGS.T + 2]) * c

            J = tr((LQGS.Q[1] - LQGS.P[1]) * Σ) + tr(LQGS.P[1] * Z[1])

            for t = 1:LQGS.T
                Σ_predicted = LQGS.A[t] * Σ * LQGS.A[t]' + 
                    (k == t + 1 ? W : Z[t + 1])
                c = LQGS.C[t + 1] * Σ_predicted
                Σ = Σ_predicted - c' * inv(c * LQGS.C[t + 1]' + 
                    Z[t + LQGS.T + 2]) * c

                J += tr((LQGS.Q[t + 1] - LQGS.P[t + 1]) * Σ) + 
                    tr(LQGS.P[t + 1] * Σ_predicted)
            end

            return J
        end

        return ForwardDiff.gradient(∇f_W, Z[k])

    elseif k ∈ (LQGS.T + 2):(2 * LQGS.T + 2)
        function ∇f_V(V)
            c = LQGS.C[1] * Z[1]
            Σ = Z[1] - c' * inv(c * LQGS.C[1]' + 
                (k == LQGS.T + 2 ? V : Z[LQGS.T + 2])) * c

            J = tr((LQGS.Q[1] - LQGS.P[1]) * Σ) + tr(LQGS.P[1] * Z[1])

            for t = 1:LQGS.T
                Σ_predicted = LQGS.A[t] * Σ * LQGS.A[t]' + Z[t + 1]
                c = LQGS.C[t + 1] * Σ_predicted
                Σ = Σ_predicted - c' * inv(c * LQGS.C[t + 1]' + 
                    (k == t + LQGS.T + 2 ? V : Z[t + LQGS.T + 2])) * c

                J += tr((LQGS.Q[t + 1] - LQGS.P[t + 1]) * Σ) + 
                    tr(LQGS.P[t + 1] * Σ_predicted)
            end

            return J
        end

        return ForwardDiff.gradient(∇f_V, Z[k])
    end
end

#--------------------------------------------------------------------------------#

function SDP_oracle(
    G::Matrix{Float64}, 
    Z_center::Matrix{Float64}, 
    ρ::Float64
    )::Matrix{Float64}

    n = size(G)[1]

    λ = minimum(eigvals(Z_center))
    Z_center_root = sqrt(Z_center)

    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, L[1:n, 1:n], PSD)
    @variable(model, S[1:n, 1:n], PSD)

    @constraint(model, L - λ * I(n) in PSDCone())

    @variable(model, M[1:2n, 1:2n], Symmetric)
    @constraint(model, M[1:n, 1:n] .== Z_center_root * L * Z_center_root)
    @constraint(model, M[1:n, n+1:end] .== S)
    @constraint(model, M[n+1:end, 1:n] .== S)
    @constraint(model, M[n+1:end, n+1:end] .== I(n))
    @constraint(model, M in PSDCone())

    @constraint(model, tr(L) - 2tr(S) <= ρ^2 - tr(Z_center))

    @objective(model, Max, tr(G' * L))

    optimize!(model)

    return value.(L)
end

#--------------------------------------------------------------------------------#

function conditional_gradient_LQGS(
    LQGS::linear_quadratic_gaussian_system, 
    f, 
    ∇f, 
    Z_initial::Vector{Matrix{Float64}}, 
    Z_center::Vector{Matrix{Float64}}, 
    ρ::Float64; 
    τ::Float64 = 1e-6, 
    maximal_iterations::Int64 = 1000, 
    verbose::Bool = true
    )::Tuple{Vector{Matrix{Float64}}, Float64}

    Z_iteration = copy(Z_initial)
    optimal_value = f(LQGS, Z_iteration)
    optimal_value_previous = copy(optimal_value)

    if verbose
        print("+----------------------------------------------------------------")
        println("------------------+")
        print("|   FRANK-WOLFE ALGORITHM STARTED                                ")
        println("                  |")
        print("+----------------------------------------------------------------")
        println("------------------+")
        println("|   ITERATION ", lpad(0, 5, ' '), "   |   OPTIMAL VALUE = ", 
            @sprintf("%13.8f", optimal_value), "   |                        |")
    end

    time = @elapsed begin
        for k = 1:maximal_iterations
            if verbose
                print("|   ITERATION ", lpad(k, 5, ' '), "   |   ")
            end

            Z_iteration = conditional_gradient_LQGS_step(LQGS, ∇f, Z_iteration, 
                Z_center, ρ, k)

            optimal_value = f(LQGS, Z_iteration)
            Δf = abs(optimal_value - 
                optimal_value_previous) / abs(optimal_value_previous)
            optimal_value_previous = optimal_value

            if verbose
                println("OPTIMAL VALUE = ", @sprintf("%13.8f", optimal_value), 
                    "   |   Δf = ", @sprintf("%13.8f", Δf), "   |")
            end

            if Δf < τ
                break
            end
        end    
    end

    if verbose
        s = @sprintf("%.8f", time)
        n = length(s)
        print("+----------------------------------------------------------------")
        println("------------------+")
        println("|   FRANK-WOLFE ALGORITHM TERMINATED (", s, " SECONDS)", 
            lpad("", 35 - n,' '), " |")
        print("+----------------------------------------------------------------")
        println("------------------+")
    end

    return (Z_iteration, optimal_value)
end

#--------------------------------------------------------------------------------#

function conditional_gradient_LQGS_step(
    LQGS::linear_quadratic_gaussian_system, 
    ∇f, 
    Z_iteration::Vector{Matrix{Float64}}, 
    Z_center::Vector{Matrix{Float64}}, 
    ρ::Float64, 
    k::Int64
    )::Vector{Matrix{Float64}}

    L = Vector{Matrix{Float64}}(undef, 2 * LQGS.T + 2)
    for i = 1:(2 * LQGS.T + 2)
        c = ∇f(LQGS, Z_iteration, i)
        c = norm(c, 2) < 1e-8 ? zeros(size(c)) : c / norm(c, 2)
        L[i] = SDP_oracle(c, Z_center[i], ρ)
    end
    
    return Z_iteration + 2 * (L - Z_iteration) / (k + 1)
end

#--------------------------------------------------------------------------------#

function initial_estimation(
    LQGS::linear_quadratic_gaussian_system
    )::Vector{Matrix{Float64}}

    x_initial_estimated = rand(MvNormal(zeros(LQGS.n), LQGS.X_initial))
    w_estimated = [rand(MvNormal(zeros(LQGS.n), LQGS.W[t])) for t = 1:LQGS.T]
    v_estimated = [rand(MvNormal(zeros(LQGS.p), 
        LQGS.V[t])) for t = 1:(LQGS.T + 1)]

    X_initial_estimated = x_initial_estimated * x_initial_estimated'
    W_estimated = [w_estimated[t] * w_estimated[t]' for t = 1:LQGS.T]
    V_estimated = [v_estimated[t] * v_estimated[t]' for t = 1:(LQGS.T + 1)]

    return [X_initial_estimated, W_estimated..., V_estimated...]
end

#--------------------------------------------------------------------------------#

function update_estimations!(
    LQGS::linear_quadratic_gaussian_system, 
    Z_estimated::Vector{Matrix{Float64}}, 
    k::Int64
    )

    x_initial_estimated = rand(MvNormal(zeros(LQGS.n), LQGS.X_initial))
    w_estimated = [rand(MvNormal(zeros(LQGS.n), LQGS.W[t])) for t = 1:LQGS.T]
    v_estimated = [rand(MvNormal(zeros(LQGS.p), 
        LQGS.V[t])) for t = 1:(LQGS.T + 1)]

    α = 1 / k
    Z_estimated[1] = (1 - α) * Z_estimated[1] + 
        α * x_initial_estimated * x_initial_estimated'
    Z_estimated[2:(LQGS.T + 1)] = (1 - α) * Z_estimated[2:(LQGS.T + 1)] + 
        α * [w_estimated[t] * w_estimated[t]' for t = 1:LQGS.T]
    Z_estimated[(LQGS.T + 2):end] = (1 - α) * Z_estimated[(LQGS.T + 2):end] + 
        α * [v_estimated[t] * v_estimated[t]' for t = 1:(LQGS.T + 1)]

    for t = 1:(2LQGS.T + 2)
        Z_estimated[t] = (Z_estimated[t] + Z_estimated[t]') / 2
    end
end

#--------------------------------------------------------------------------------#

function analyzing_procedure(
    n::Int64, 
    T::Int64, 
    ρ::Float64; 
    N_averaging::Int64 = 50, 
    N_sampling::Int64 = 1000
    )

    LQGS = generate_linear_quadratic_gaussian_system(n, T)
    ℙ = [LQGS.X_initial, LQGS.W..., LQGS.V...]

    #----------

    f_optimal = f(LQGS, ℙ)
    _, f_optimal_robust = conditional_gradient_LQGS(LQGS, f, ∇f, ℙ, ℙ, ρ);
    println()

    #----------

    f_direct_iterations_collection = Vector{Vector{Float64}}(undef, N_averaging)
    computation_time_direct_collection = Vector{Vector{Float64}}(undef, 
        N_averaging)
    relative_error_direct_collection = Vector{Vector{Float64}}(undef, 
        N_averaging)

    f_adaptive_iterations_collection = Vector{Vector{Float64}}(undef, N_averaging)
    computation_time_adaptive_collection = Vector{Vector{Float64}}(undef, 
        N_averaging)
    relative_error_adaptive_collection = Vector{Vector{Float64}}(undef, 
        N_averaging)

    #----------

    Random.seed!(42)

    for i = 1:N_averaging
        ℙ_estimated = initial_estimation(LQGS)

        ℙ_optimal_iteration_direct = nothing
        ℙ_optimal_iteration_adaptive = nothing

        #----------

        f_direct_iterations = Vector{Float64}(undef, N_sampling)
        computation_time_direct = Vector{Float64}(undef, N_sampling)
        relative_error_direct = Vector{Float64}(undef, N_sampling)

        f_adaptive_iterations = Vector{Float64}(undef, N_sampling)
        computation_time_adaptive = Vector{Float64}(undef, N_sampling)
        relative_error_adaptive = Vector{Float64}(undef, N_sampling)

        #----------

        n = length(string(N_averaging))
        m = length(string(N_sampling))

        time = @elapsed begin 
            for k = 1:N_sampling
                print("\r")
                print(" AVERAGING RUN ", lpad(i, n, ' '), 
                    " OF $N_averaging | SAMPLE ITERATION ", lpad(k, m, ' '),
                    " OF $N_sampling |", " $(round(100 * (k + (i - 1) * 
                    N_sampling) / (N_averaging * N_sampling), digits = 2))%")

                #----------

                update_estimations!(LQGS, ℙ_estimated, k)

                if k == 1
                    ℙ_optimal_iteration_direct = ℙ_estimated
                    ℙ_optimal_iteration_adaptive = ℙ_estimated
                end

                #----------

                computation_time_direct[k] = @elapsed ℙ_optimal_iteration_direct, 
                    f_direct_iterations[k] = conditional_gradient_LQGS(LQGS, f, 
                    ∇f, ℙ_optimal_iteration_direct, ℙ_estimated, ρ; 
                    verbose = false)
                relative_error_direct[k] = norm(f_direct_iterations[k] - 
                    f_optimal_robust, 2) / abs(f_optimal_robust)
            
                computation_time_adaptive[k] = @elapsed ℙ_optimal_iteration_adaptive = 
                    conditional_gradient_LQGS_step(LQGS, ∇f, 
                    ℙ_optimal_iteration_adaptive, ℙ_estimated, ρ, k)
                f_adaptive_iterations[k] = f(LQGS, ℙ_optimal_iteration_adaptive) 
                relative_error_adaptive[k] = norm(f_adaptive_iterations[k] - 
                    f_optimal_robust, 2) / abs(f_optimal_robust)
            end
        end

        println(" (", @sprintf("%12.8f", time), " seconds)")

        f_direct_iterations_collection[i] = f_direct_iterations
        computation_time_direct_collection[i] = computation_time_direct
        relative_error_direct_collection[i] = relative_error_direct

        f_adaptive_iterations_collection[i] = f_adaptive_iterations
        computation_time_adaptive_collection[i] = computation_time_adaptive
        relative_error_adaptive_collection[i] = relative_error_adaptive
    end

    DIRECT = (f_direct_iterations_collection, computation_time_direct_collection, relative_error_direct_collection)
    ADAPTIVE = (f_adaptive_iterations_collection, computation_time_adaptive_collection, relative_error_adaptive_collection)

    return (f_optimal, f_optimal_robust, ℙ_estimation_error_collection, DIRECT, ADAPTIVE)
end

#--------------------------------------------------------------------------------#
# MAIN PROGRAM                                                                   #
#--------------------------------------------------------------------------------#

n = 4
T = 10
ρ = 0.1
N_averaging = 12
N_sampling = 100

f_optimal, f_optimal_robust, ℙ_estimation_error_collection, DIRECT, 
    ADAPTIVE = analyzing_procedure(n, T, ρ; N_averaging = N_averaging, 
    N_sampling = N_sampling);

#--------------------------------------------------------------------------------#
# PLOT GENERATION (TRAJECTORY COMPARISON)                                        #
#--------------------------------------------------------------------------------#

# compute 25, 50 and 75 percentile for the direct trajectories
direct_data = hcat(DIRECT[1]...)';

direct_q75 = map(x -> quantile(x, 0.75), eachcol(direct_data));
direct_q50 = map(x -> quantile(x, 0.50), eachcol(direct_data));
direct_q25 = map(x -> quantile(x, 0.25), eachcol(direct_data));

# compute 25, 50 and 75 percentile for the adaptive trajectories
adaptive_data = hcat(ADAPTIVE[1]...)';

adaptive_q75 = map(x -> quantile(x, 0.75), eachcol(adaptive_data));
adaptive_q50 = map(x -> quantile(x, 0.50), eachcol(adaptive_data));
adaptive_q25 = map(x -> quantile(x, 0.25), eachcol(adaptive_data));

#--------------------------------------------------------------------------------#

# set x-axis details
x_offset = 0.1 * N_sampling;
x_ticks = range(0, N_sampling, length = 5);
x_ticks_labels = [@sprintf("%d", x) for x ∈ x_ticks];

# set y-axis details 
Ix = Int64(x_offset):N_sampling;
y_minimum = 0.99 * minimum([f_optimal, f_optimal_robust, direct_q25[Ix]..., 
    adaptive_q25[Ix]...]);
y_maximum = 1.01 * maximum([f_optimal, f_optimal_robust, direct_q75[Ix]..., 
    adaptive_q75[Ix]...]);
   
#--------------------------------------------------------------------------------#

p = plot(
    title = "(a) Best-Case Frank-Wolfe Trajectory", 
    titlefontsize = 13,

    xlabel = "Number of Samples / Iterations", 
    xlims = (x_offset, N_sampling),
    xticks = x_ticks, 

    ylabel = "Cost of Control",
    ylims = (y_minimum, y_maximum),

    right_margin = 5Plots.mm,
    top_margin = 5Plots.mm,
    framestyle = :box,
    grid = true
)

plot!(
    f_optimal * ones(N_sampling), 
    lw = 2, 
    ls = :dash, 
    color = :red, 
    label = "optimal cost"
)

plot!(
    f_optimal_robust * ones(N_sampling), 
    lw = 2, 
    color = :red, 
    label = "robust optimal cost"
)

plot!(
    direct_q50, 
    ribbon = (direct_q50 .- direct_q25, direct_q75 .- direct_q50),
    lw = 1.5, 
    color = :black, 
    fillalpha = 0.2, 
    label = "meadian direct trajectory with 50% band"
)

savefig(p, "FW_direct_cost.png");

#--------------------------------------------------------------------------------#

p = plot(
    title = "(b) Online Adaptive Frank-Wolfe Trajectory", 
    titlefontsize = 13,

    xlabel = "Number of Samples / Iterations", 
    xlims = (x_offset, N_sampling),
    xticks = x_ticks, 

    ylabel = "Cost of Control",
    ylims = (y_minimum, y_maximum),

    right_margin = 5Plots.mm,
    top_margin = 5Plots.mm,
    framestyle = :box,
    grid = true,
)

plot!(
    f_optimal * ones(N_sampling), 
    lw = 2, 
    ls = :dash, 
    color = :red, 
    label = "optimal cost"
)

plot!(
    f_optimal_robust * ones(N_sampling), 
    lw = 2, 
    color = :red, 
    label = "robust optimal cost"
)

plot!(
    adaptive_q50, 
    ribbon = (adaptive_q50 .- adaptive_q25, adaptive_q75 .- adaptive_q50), 
    lw = 1.5, 
    color = :black,
    fillalpha = 0.2, 
    label = "meadian adaptive trajectory with 50% band"
)

savefig(p, "FW_adaptive_cost.png");

#--------------------------------------------------------------------------------#
# PLOT GENERATION (COMPUTATION TIMES COMPARISON)                                 #
#--------------------------------------------------------------------------------#

# compute 25, 50 and 75 percentile for the direct computation times
direct_data = hcat(DIRECT[2]...)';

direct_q75 = map(x -> quantile(x, 0.75), eachcol(direct_data));
direct_q50 = map(x -> quantile(x, 0.50), eachcol(direct_data));
direct_q25 = map(x -> quantile(x, 0.25), eachcol(direct_data));

# compute 25, 50 and 75percentile for the adaptive computation times
adaptive_data = hcat(ADAPTIVE[2]...)';

adaptive_q75 = map(x -> quantile(x, 0.75), eachcol(adaptive_data));
adaptive_q50 = map(x -> quantile(x, 0.50), eachcol(adaptive_data));
adaptive_q25 = map(x -> quantile(x, 0.25), eachcol(adaptive_data));

#--------------------------------------------------------------------------------#

# set x-axis details
x_offset = 0.1 * N_sampling;
x_ticks = range(0, N_sampling, length = 5);
x_ticks_labels = [@sprintf("%d", x) for x ∈ x_ticks];

# set y-axis details 
y_minimum = 0.95 * minimum([direct_q25[Int64(x_offset):N_sampling]..., 
    adaptive_q25[Int64(x_offset):N_sampling]...]);
y_maximum = 1.05 * maximum([direct_q75[Int64(x_offset):N_sampling]..., 
    adaptive_q75[Int64(x_offset):N_sampling]...]);

#--------------------------------------------------------------------------------#

p = plot(
    title = "Computation Times Comparison", 
    titlefontsize = 13,

    xlabel = "Number of Samples / Iterations", 
    xlims = (x_offset, N_sampling),
    xticks = x_ticks, 

    ylabel = "Computatin Times",
    ylims = (y_minimum, y_maximum),
    yscale = :log10,

    right_margin = 5Plots.mm,
    top_margin = 5Plots.mm,
    framestyle = :box,
    grid = true
)

plot!(
    direct_q50, 
    ribbon = (direct_q50 .- direct_q25, direct_q75 .- direct_q50), 
    lw = 2, 
    color = :black, 
    fillalpha = 0.2,
    label = "median direct processing time with 50% band"
)

plot!(
    adaptive_q50, 
    ribbon = (adaptive_q50 .- adaptive_q25, adaptive_q75 .- adaptive_q50), 
    lw = 2, 
    color = :red, 
    fillalpha = 0.2, 
    label = "median adaptive processing time with 50% band"
)

savefig(p, "FW_direct_cost.png")

#--------------------------------------------------------------------------------#
# PLOT GENERATION (RELATIVE ERROR COMPARISON)                                    #
#--------------------------------------------------------------------------------#

# compute 25, 50 and 75 percentile for the direct relative error
direct_data = hcat(DIRECT[3]...)';

direct_q75 = map(x -> quantile(x, 0.75), eachcol(direct_data));
direct_q50 = map(x -> quantile(x, 0.50), eachcol(direct_data));
direct_q25 = map(x -> quantile(x, 0.25), eachcol(direct_data));

# compute 25, 50 and 75percentile for the adaptive relative error
adaptive_data = hcat(ADAPTIVE[3]...)';

adaptive_q75 = map(x -> quantile(x, 0.75), eachcol(adaptive_data));
adaptive_q50 = map(x -> quantile(x, 0.50), eachcol(adaptive_data));
adaptive_q25 = map(x -> quantile(x, 0.25), eachcol(adaptive_data));

#--------------------------------------------------------------------------------#

# set x-axis details
x_offset = 0.1 * N_sampling;
x_ticks = range(0, N_sampling, length = 5);
x_ticks_labels = [@sprintf("%d", x) for x ∈ x_ticks];

# set y-axis details 
y_minimum = 0.95 * minimum([direct_q25[Int64(x_offset):N_sampling]..., 
    adaptive_q25[Int64(x_offset):N_sampling]...]);
y_maximum = 1.05 * maximum([direct_q75[Int64(x_offset):N_sampling]..., 
    adaptive_q75[Int64(x_offset):N_sampling]...]);

#--------------------------------------------------------------------------------#

p = plot(
    title = "Relative Error Comparison", 
    titlefontsize = 13,

    xlabel = "Number of Samples / Iterations", 
    xlims = (x_offset, N_sampling),
    xticks = x_ticks, 

    ylabel = "Relative Error",
    ylims = (y_minimum, y_maximum),
    yscale = :log10,

    right_margin = 5Plots.mm,
    top_margin = 5Plots.mm,
    framestyle = :box,
    grid = true
)

plot!(
    direct_q50, 
    # ribbon = (direct_q50 .- direct_q25, direct_q75 .- direct_q50), 
    lw = 2, 
    color = :blue, 
    fillalpha = 0.2,
    label = "median direct processing time with 50% band"
)

plot!(
    adaptive_q50, 
    # ribbon = (adaptive_q50 .- adaptive_q25, adaptive_q75 .- adaptive_q50), 
    lw = 2, 
    color = :red, 
    fillalpha = 0.2, 
    label = "median adaptive processing time with 50% band"
)

savefig(p, "FW_direct_cost.png")

#--------------------------------------------------------------------------------#