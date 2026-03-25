#----------------------------------------------------------------------------------#

function initial_domain_estimation(
    P::Union{problem, problem_extended}, 
    N::Int64;
    convex_hull_representation::Bool = false
)::Tuple{Float64, Float64}

    Y = (P.b - P.a) .* rand(N) .+ P.a # sample N points in D = [a, b]

    if convex_hull_representation
        return minimum(Y), maximum(Y) 
    else
        μ = mean(Y)
        σ = std(Y)

        return μ - sqrt(3) * σ, μ + sqrt(3) * σ
    end
end

#----------------------------------------------------------------------------------#

function update_domain_estimation(
    P::Union{problem, problem_extended}, 
    A::Real, 
    B::Real;
    N::Int64 = 0, 
    k::Int64 = 0, 
    convex_hull_representation::Bool = false
)::Tuple{Float64, Float64}

    y = (P.b - P.a) .* rand() .+ P.a # sample point in D = [a, b]

    if convex_hull_representation
        return min(A, y), max(B, y)
    else    
        # derive empirical estimators of mean and variance from the current 
        # approximation points A and B
        μ = (A + B) / 2
        σ_squared = (B - A) ^ 2 / 12

        μ₋ = μ

        # update the running empirical estimators of mean and standard deviation
        μ = ((N + k - 1) * μ₋ + y) / (N + k)
        σ_squared = ((N + k - 1) * σ_squared + (y - μ₋) * (y - μ)) / (N + k)
        
        return μ - sqrt(3 * σ_squared), μ + sqrt(3 * σ_squared)
    end
end

#----------------------------------------------------------------------------------#