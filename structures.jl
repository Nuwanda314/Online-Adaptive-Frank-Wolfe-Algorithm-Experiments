#----------------------------------------------------------------------------------#

struct problem
    a::Real
    b::Real

    x_star::Real
end

#----------------------------------------------------------------------------------#

struct problem_extended
    a::Real
    b::Real
    
    x_star::Real

    x_optimal::Real
    P_optimal::Real
end

#----------------------------------------------------------------------------------#

function extend_problem(
    P::problem
)::problem_extended

    (x_optimal, P_optimal) = optimal_data(P)

    return problem_extended(P.a, P.b, P.x_star, x_optimal, P_optimal)
end

#----------------------------------------------------------------------------------#

struct theoretic_data
    r::Real

    c::Real
    A::Real
    A_plus::Real
    B::Real

    N_shift::Real
    N_quadratic::Real
end

#----------------------------------------------------------------------------------#

function generate_theoretic_data(
    P::Union{problem, problem_extended},
    m::Int64, 
    τ::Int64;
    convex_hull_representation::Bool = false
)::theoretic_data

    if convex_hull_representation
        r = (m - 1) / m
        c = Float64((P.b - P.a) * (10 ^ τ * factorial(big(m))) ^ (1 / (2 * m)))
    else
        r = (m - 1) / (2 * m)
        c = Float64(6 * (P.b - P.a) * (2 + (2 * 10 ^ τ * factorial(big(m))) ^ (1 / (2 * m))) ^ 2)
    end

    LL = 6 * (abs(P.b) + abs(P.a)) + 2 * abs(P.x_star) 
    CC = 12 * (abs(P.b) + abs(P.a))

    A = 2 * c * LL + CC

    C_f = 2 * (P.b - P.a)
    L_f = 2 * max(abs(P.a - P.x_star), abs(P.b - P.x_star))

    if convex_hull_representation
        A_plus = A
    else
        A_plus = 4 * c * L_f + C_f
    end

    c_extended = (9 / 2) * c * L_f + (C_f / 2)
    ρ = min(abs(P.a - P.x_star), abs(P.b - P.x_star))

    B = ((4 * (b - a) / ρ) * c_extended)^2 + c_extended

    N_shift = ceil(Int64, ((6 * c) / (P.b - P.a)) ^ (1 / r))
    N_quadratic = ceil(Int64, 2 * ((6 * c) / ρ) ^ (1 / r) - 2)

    return theoretic_data(r, c, A, A_plus, B, N_shift, N_quadratic)
end

#----------------------------------------------------------------------------------#