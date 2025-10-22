using LsqFit: curve_fit

"""Helper function for characterizing the inner period."""
function estimate_period(data::Matrix{T}, index::Int, p0::Vector{T}) where T <: Real
    # Get all rows with TT of the first planet
    ftt = data[data[:,1] .== index,2:end]

    if size(ftt, 1) == 1
        error("Unable to determine the inner period. Need more than one transits")
    end
    
    # Fit the data to a linear model
    @. linear_model(x, p) = p[1]*x + p[2]
    xdata = ftt[:,1]
    ydata = ftt[:,2]

    fit_results = curve_fit(linear_model, xdata, ydata, p0)

    return fit_results.param, linear_model(xdata, fit_results.param)
end

"""Helper function to calculate kappa and period ratio deviation"""
function calculate_period_deviation(periods::Vector{T}, cfactor::Vector{T}) where T <: Real
    nplanet = length(periods)
    
    kappa = periods[2] / periods[1]

    # Period ratio deviation
    pratio_nom = Vector{T}(undef, nplanet-1)
    pratio_nom[1] = kappa

    for i = 2:nplanet-1
        pratio_nom[i] = 1/(1 + cfactor[i-1]*(1 - pratio_nom[i-1]))
    end 

    pratiodev = [(periods[i] / periods[i-1]) - pratio_nom[i-1] for i in eachindex(periods)[3:end]]

    return kappa, pratiodev
end

function estimate_initial_M(tt1::Vector{T}, periods::Vector{T}, omegas::Vector{T}, t0::T=0.0) where T <: Real
    return [rem2pi((t0-tt1[i])*2pi/periods[i] - pi/2 - omegas[i], RoundNearest) for i in 1:length(periods)]
end

# Convenient function for us to check periodicity (though only preliminarily)
function param_diff(nplanet, final_elems, init_elems)
    init_optparams = OptimParameters(nplanet, vcat(init_elems, 0.))

    final_optparams = OptimParameters(nplanet, vcat(final_elems, 0.))
    
    # Calculate the differences for each elements
    diff_e = final_optparams.e - init_optparams.e
    diff_M = rem2pi.(final_optparams.M - init_optparams.M, RoundNearest)
    diff_ωdiff = rem2pi.(final_optparams.Δω - init_optparams.Δω, RoundNearest)
    diff_pratiodev = final_optparams.Pratio - init_optparams.Pratio
    diff_inner_period = final_optparams.inner_period - init_optparams.inner_period

    diff_kappa = final_optparams.kappa - init_optparams.kappa
    diff_ω1 = rem2pi(final_optparams.ω1 - init_optparams.ω1, RoundNearest)

    # Create a vector, and appended with constant quantities
    diff = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, diff_inner_period, diff_kappa, diff_ω1)

    return diff
end