#----------------------------------------------------------------------------------#

default(
    size = (1200, 800), 
    dpi = 150, 

    titlefontsize = 20, 
    guidefontsize = 20,
    legendfontsize = 15,
    tickfontsize = 12, 

    markersize = 3
)

#----------------------------------------------------------------------------------#

function generate_error_plot(
    N::Vector{Int64}, 
    T::theoretic_data,
    best_case_errors::Vector{<:Vector{<:Real}}, 
    adaptive_errors::Vector{<:Vector{<:Real}};
    spacing::Int64 = 1, 
    quadratic::Bool = false,
    separate::Bool = false,
    ribbon::Bool = false,
    plot_bound::Bool = false,
    x_offset::Real = 0,
    y_lim_lower::Real = -Inf, 
    y_lim_upper::Real = Inf, 
)::Union{Plots.Plot, Tuple{Plots.Plot, Plots.Plot}}

    Ix = x_offset:spacing:N[3]

    best_case_error_data = log10.(hcat(best_case_errors...)')
    best_case_error_data = best_case_error_data[:, Ix .+ 1]
    best_case_error_mean = map(x -> mean(x), eachcol(best_case_error_data))
    best_case_error_std = map(x -> std(x), eachcol(best_case_error_data))

    adaptive_error_data = log10.(hcat(adaptive_errors...)')
    adaptive_error_data = adaptive_error_data[:, Ix .+ 1]
    adaptive_error_mean = map(x -> mean(x), eachcol(adaptive_error_data))
    adaptive_error_std = map(x -> std(x), eachcol(adaptive_error_data))

    if plot_bound
        if quadratic
            if N[3] > T.N_quadratic
                initial_error_mean = 
                mean([adaptive_errors[i][T.N_quadratic] for i = 1:N[2]])
            else
                initial_error_mean = 0
            end

            Λ = log10.(T.A_plus * (2 ./ (2 .+ Ix)) .^ T.r)
            Λ_quadratic = 
                log10.((initial_error_mean + T.B) * (2 ./ (1 .+ Ix)) .^ (2 * T.r))

            N_intersect = findfirst(Λ .> Λ_quadratic)

            if N_intersect === nothing
                Λ_optimal_combined = Λ
            else
                Λ_optimal_combined = [Λ[1:(N_intersect - 1)]; 
                Λ_quadratic[N_intersect:end]]
                Λ_linear_continued = Λ[N_intersect:end]
            end

            value_collection_min = 
                [(best_case_error_mean .- best_case_error_std)..., 
                (adaptive_error_mean .- adaptive_error_std)..., 
                Λ_optimal_combined...]
            value_collection_max = 
                [(best_case_error_mean .+ best_case_error_std)..., 
                (adaptive_error_mean .+ adaptive_error_std)..., 
                Λ_optimal_combined...]
        else
            Λ = log10.(T.A * (2 ./ (2 .+ Ix)) .^ T.r)

            value_collection_min = 
                [(best_case_error_mean .- best_case_error_std)...,
                (adaptive_error_mean .- adaptive_error_std)..., Λ...]
            value_collection_max = 
                [(best_case_error_mean .+ best_case_error_std)...,
                (adaptive_error_mean .+ adaptive_error_std)..., Λ...]
        end
    else
        value_collection_min = [(best_case_error_mean .- best_case_error_std)..., 
            (adaptive_error_mean .- adaptive_error_std)...]
        value_collection_max = [(best_case_error_mean .+ best_case_error_std)..., 
            (adaptive_error_mean .+ adaptive_error_std)...]
    end

    # set x-axis details
    x_ticks = range(0, N[3], length = 5)
    x_ticks_labels = [@sprintf("%d", x) for x ∈ x_ticks]

    # set y-axis details 
    y_minimum = 0.99 * minimum(value_collection_min)
    y_maximum = 1.01 * maximum(value_collection_max)

    if y_lim_lower > -Inf
        y_minimum = y_lim_lower
    end

    if y_lim_upper < Inf
        y_maximum = y_lim_upper
    end

    if separate
        p = plot(
            title = "Best-Case Frank-Wolfe Error", 

            x_lims = (x_offset, N[3]),
            xticks = (x_ticks, x_ticks_labels), 
            xlabel = "Number of Samples / Iterations", 

            ylims = (y_minimum, y_maximum),
            ylabel = "Logarithm of Error",

            right_margin = 7.5Plots.mm,
            left_margin = 10Plots.mm,
            top_margin = 7.5Plots.mm,
            bottom_margin = 10Plots.mm,

            framestyle = :box,
            grid = true
        )

        confidence = 0
        if ribbon
            confidence = best_case_error_std
        end

        p = plot!(
            p, 
            Ix, 
            best_case_error_mean, 
            ribbon = confidence,
            fillalpha = 0.25,
            lw = 1.25, 
            color = :black, 
            label = "best-case error"
        )

        q = plot(
            title = "Adaptive Frank-Wolfe Error", 

            x_lims = (x_offset, N[3]),
            xticks = (x_ticks, x_ticks_labels), 
            xlabel = "Number of Samples / Iterations", 

            ylims = (y_minimum, y_maximum),
            ylabel = "Logarithm of Error",

            right_margin = 7.5Plots.mm,
            left_margin = 10Plots.mm,
            top_margin = 7.5Plots.mm,
            bottom_margin = 10Plots.mm,

            framestyle = :box,
            grid = true
        )

        confidence = 0
        if ribbon
            confidence = adaptive_error_std
        end

        q = plot!(
            q, 
            Ix, 
            adaptive_error_mean, 
            ribbon = confidence,
            fillalpha = 0.25,
            lw = 1.25, 
            color = :red, 
            label = "adaptive error"
        )

        if plot_bound
            for w ∈ [p, q]
                if quadratic
                    w = plot!(
                        w,
                        Ix, 
                        Λ_optimal_combined, 
                        lw = 1.75, 
                        color = :green,
                        label = "theoretic upper bound"
                    )

                    if N_intersect !== nothing
                        w = plot!(
                            w,
                            Ix[N_intersect:end], 
                            Λ_linear_continued, 
                            lw = 1.75, 
                            ls = :dot, 
                            color = :green,
                            label = "nonquadratic upper bound"
                        )
                    end
                else
                    w = plot!(
                        w,
                        Ix,
                        Λ,
                        lw = 1.75,
                        color = :green,
                        label = "theoretic upper bound"
                    )
                end
            end
        end

        return (p, q)
    else
        p = plot(
            title = "Comparison of Best-Case and Adaptive Frank-Wolfe Error", 

            x_lims = (x_offset, N[3]),
            xticks = (x_ticks, x_ticks_labels), 
            xlabel = "Number of Samples / Iterations", 

            ylims = (y_minimum, y_maximum),
            ylabel = "Logarithm of Error",

            right_margin = 7.5Plots.mm,
            left_margin = 10Plots.mm,
            top_margin = 7.5Plots.mm,
            bottom_margin = 10Plots.mm,

            framestyle = :box,
            grid = true
        )

        confidence = 0
        if ribbon
            confidence = best_case_error_std
        end

        p = plot!(
            p, 
            Ix, 
            best_case_error_mean, 
            ribbon = confidence,
            fillalpha = 0.25,
            lw = 1.25, 
            color = :black, 
            label = "best-case error"
        )

        confidence = 0
        if ribbon
            confidence = adaptive_error_std
        end

        p = plot!(
            p, 
            Ix, 
            adaptive_error_mean, 
            ribbon = confidence,
            fillalpha = 0.25,
            lw = 1.25, 
            color = :red, 
            label = "adaptive error"
        )

        if plot_bound
            if quadratic
                p = plot!(
                    p, 
                    Ix, 
                    Λ_optimal_combined, 
                    lw = 1.75, 
                    color = :green,
                    label = "theoretic upper bound"
                )

                if N_intersect !== nothing
                    p = plot!(
                        p, 
                        Ix[N_intersect:end], 
                        Λ_linear_continued, 
                        lw = 1.75, 
                        ls = :dot, 
                        color = :green,
                        label = "nonquadratic upper bound"
                    )
                end
            else
                p = plot!(
                    p, 
                    Ix,
                    Λ,
                    lw = 1.75,
                    color = :green,
                    label = "theoretic upper bound"
                )
            end
        end

        return p
    end
end

#----------------------------------------------------------------------------------#