#----------------------------------------------------------------------------------#

function optimal_data(
    P::Union{problem, problem_extended};
    A::Real = P.a, 
    B::Real = P.b
)::Tuple{<:Real, <:Real}

    if P.x_star < A
        return (A, f(P, A))
    elseif P.x_star > B
        return (B, f(P, B))
    else
        return (P.x_star, 0)
    end
end

#----------------------------------------------------------------------------------#

function LMO(
    P::Union{problem, problem_extended}, 
    A::Real, 
    B::Real, 
    x::Real,
)::Real

    return ∇f(P, x) ≥ 0 ? A : B
end

#----------------------------------------------------------------------------------#