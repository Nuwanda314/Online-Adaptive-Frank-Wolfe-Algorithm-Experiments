#----------------------------------------------------------------------------------#
# PACKAGES                                                                         #
#----------------------------------------------------------------------------------#

using Pkg
Pkg.add(["Plots", "Printf", "ProgressMeter", "Random", "Statistics"])

using LinearAlgebra, Plots, Printf, ProgressMeter, Random, Statistics

#----------------------------------------------------------------------------------#
# INCLUDE FUNCTIONS                                                                #
#----------------------------------------------------------------------------------#

# required packages: -
include("structures.jl");

# required packages: -
include("objective_function.jl");

# required packages: -
include("oracles.jl");

# required packages: Random, Statistics 
include("domain_approximation.jl");

# required packages: Printf, ProgressMeter 
include("analyzing_procedure.jl");

# required packages: Plots, Printf, Statistics 
include("plot_generation.jl");

#----------------------------------------------------------------------------------#
# MAIN CODE                                                                        #
#----------------------------------------------------------------------------------#

a = 0;
b = 1;
x_star = 0.5;

convex_hull_representation = true;
quadratic = true;

P = extend_problem(problem(a, b, x_star));

m = 20;
τ = 2;

T = generate_theoretic_data(P, m, τ; convex_hull_representation = 
    convex_hull_representation);

N_domain = 5;
N_averaging = 25;
N_sampling = 10000;

if quadratic
    N_domain = T.N_shift

    c = (T.A_plus / T.B) ^ (1 / T.r)
    N_sampling = max(N_sampling, T.N_quadratic, 
        ceil(Int64, ((2 - 2 * c) + sqrt(4  - 4 * c)) / (2 * c)))

    if N_sampling > 1e8
        error("Number of sampling points neccessary for comparison is too 
            high: $(N_sampling)")
    end

    d = floor(Int64, log10(N_sampling))
    N_sampling = ceil(Int64, N_sampling / (10 ^ d)) * 10 ^ d
end

N = [N_domain, N_averaging, N_sampling]

(best_case_errors, adaptive_errors) = start_analysis(P, N, T; quadratic = quadratic, 
    convex_hull_representation = convex_hull_representation);

#----------------------------------------------------------------------------------#

spacing_optimal = max(round(Int64, N_sampling / 10000), 1)

p = generate_error_plot(
    N, 
    T,
    best_case_errors, 
    adaptive_errors,
    spacing = spacing_optimal,
    quadratic = quadratic,
    separate = false,
    ribbon = true, 
    plot_bound = true,
    x_offset = 0,
    y_lim_lower = -Inf,
    y_lim_upper = Inf
)

p[1]
p[2]

#----------------------------------------------------------------------------------#

savefig(p[2], "SE_BQ_adaptive.png")

#----------------------------------------------------------------------------------#