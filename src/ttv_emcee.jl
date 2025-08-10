using PyCall
using Statistics
using PyPlot

include("PeriodicOrbit.jl")
using .PeriodicOrbit
using NbodyGradient

emcee = pyimport("emcee")
np = pyimport("numpy")

struct TTVData
    planet_id::Int
    transit_numbers::Vector{Int}
    transit_times::Vector{Float64}
    linear_predictions::Vector{Float64}
    ttv_observations::Vector{Float64}
    ttv_uncertainties::Vector{Float64}
    obs_t0::Float64
    obs_period::Float64
end

function load_ttv_data(filename::String)
    data = []
    open(filename, "r") do file
        for line in eachline(file)
            if startswith(line, "#") || isempty(strip(line))
                continue
            end
            parts = split(strip(line))
            push!(data, (parse(Int, parts[1]), parse(Float64, parts[2]), 
                        parse(Float64, parts[3]), parse(Float64, parts[4])))
        end
    end
    
    if isempty(data)
        error("No data found in file: $filename")
    end
    
    planet_id = contains(filename, "planet1") ? 1 : 2
    
    transit_numbers = [d[1] for d in data]
    transit_times = [d[2] for d in data]
    linear_predictions = [d[3] for d in data]
    ttv_observations = [d[4] for d in data]
    ttv_uncertainties = fill(1.0, length(ttv_observations))
    
    A = hcat(ones(length(transit_numbers)), transit_numbers)
    ephemeris_params = A \ linear_predictions
    obs_t0, obs_period = ephemeris_params
    
    return TTVData(planet_id, transit_numbers, transit_times, linear_predictions, 
                   ttv_observations, ttv_uncertainties, obs_t0, obs_period)
end

global OBSERVED_DATA = nothing

function setup_likelihood_function(observed_data::TTVData)
    global OBSERVED_DATA = observed_data
end

function params_vector_to_mcmc(params_vector)
    log_mass1, log_mass2, period1, period_ratio, e1, e2 = params_vector[1:6]
    mean_anom1, mean_anom2, omega_diff, kappa, tsys = params_vector[7:11]
    
    return (
        log_mass1 = log_mass1,
        log_mass2 = log_mass2,
        period1 = period1,
        period_ratio = period_ratio,
        e1 = e1,
        e2 = e2,
        mean_anomaly1 = mean_anom1,
        mean_anomaly2 = mean_anom2,
        omega_diff = omega_diff,
        kappa = kappa,
        tsys = tsys
    )
end

function mcmc_to_optvec(params)
    mass1 = 10^params.log_mass1
    mass2 = 10^params.log_mass2
    
    return [params.e1, params.e2, params.mean_anomaly1, params.mean_anomaly2,
            params.omega_diff, params.period1, mass1, mass2, params.kappa, 0.0, params.tsys]
end

function compute_model_ttvs_final(params, observed_data::TTVData)
    try
        if abs(params.period1 - observed_data.obs_period) > 1.0
            return nothing
        end
        
        optvec = mcmc_to_optvec(params)
        
        if any(isnan.(optvec)) || any(isinf.(optvec))
            return nothing
        end
        
        optparams = OptimParameters(2, optvec)
        orbparams = OrbitParameters(2, [0.5])
        
        orbit = Orbit(2, optparams, orbparams)
        elems = get_orbital_elements(orbit.s, orbit.ic)
        
        derived_period = elems[2].P
        if abs(derived_period - observed_data.obs_period) > 0.5
            return nothing
        end
        
        star_edgeon = Elements(m=1.0)
        planet1_edgeon = Elements(m=elems[2].m, P=elems[2].P, e=0.005, ω=0.0, I=π/2, Ω=0.0, t0=0.0)
        planet2_edgeon = Elements(m=elems[3].m, P=elems[3].P, e=0.005, ω=π, I=π/2, Ω=0.0, t0=0.0)
        
        el_ic_edgeon = ElementsIC(0.0, 3, star_edgeon, planet1_edgeon, planet2_edgeon)
        
        P_inner = elems[2].P
        h_int = P_inner / 200.0
        
        max_obs_transit_number = maximum(observed_data.transit_numbers)
        tmax = max(P_inner * 60, (max_obs_transit_number + 10) * P_inner)
        
        state_nbody = State(el_ic_edgeon)
        tt_nbody = TransitTiming(tmax, el_ic_edgeon)
        
        Integrator(ahl21!, h_int, tmax)(state_nbody, tt_nbody)
        
        planet_idx = observed_data.planet_id
        if tt_nbody.count[planet_idx] < max_obs_transit_number + 5
            return nothing
        end
        
        model_transit_times = tt_nbody.tt[planet_idx, 1:tt_nbody.count[planet_idx]]
        
        if any(isnan.(model_transit_times)) || any(isinf.(model_transit_times))
            return nothing
        end
        
        n_model = length(model_transit_times)
        A = hcat(ones(n_model), 0:(n_model-1))
        model_ephemeris = A \ model_transit_times
        model_t0_raw, model_period_fitted = model_ephemeris
        
        if abs(model_period_fitted - observed_data.obs_period) > 0.3
            return nothing
        end
        
        target_time = observed_data.obs_t0
        time_diffs_to_obs = abs.(model_transit_times .- target_time)
        closest_idx = argmin(time_diffs_to_obs)
        
        model_t0_aligned = model_transit_times[closest_idx] - (closest_idx - 1) * model_period_fitted
        
        model_linear_pred_aligned = model_t0_aligned .+ model_period_fitted .* (0:(n_model-1))
        model_ttvs = (model_transit_times .- model_linear_pred_aligned) .* 24 * 60
        
        if any(isnan.(model_ttvs)) || any(isinf.(model_ttvs)) || maximum(abs.(model_ttvs)) > 1000
            return nothing
        end
        
        return model_transit_times, model_ttvs, (model_t0_aligned, model_period_fitted)
        
    catch e
        return nothing
    end
end

function match_transits_final(model_transit_times, model_ttvs, model_ephemeris, observed_data::TTVData)
    if model_transit_times === nothing
        return nothing
    end
    
    model_t0_aligned, model_period = model_ephemeris
    
    matched_model_ttvs = Float64[]
    matched_obs_ttvs = Float64[]
    matching_quality = Float64[]
    
    max_time_diff = 0.2
    
    for i in 1:length(observed_data.transit_numbers)
        obs_n = observed_data.transit_numbers[i]
        obs_ttv = observed_data.ttv_observations[i]
        
        predicted_model_time = model_t0_aligned + model_period * obs_n
        
        time_diffs = abs.(model_transit_times .- predicted_model_time)
        closest_idx = argmin(time_diffs)
        time_diff = time_diffs[closest_idx]
        
        if time_diff < max_time_diff
            push!(matched_model_ttvs, model_ttvs[closest_idx])
            push!(matched_obs_ttvs, obs_ttv)
            push!(matching_quality, time_diff)
        end
    end
    
    n_matched = length(matched_model_ttvs)
    n_required = max(10, div(length(observed_data.transit_numbers), 3))
    
    if n_matched < n_required
        return nothing
    end
    
    return matched_model_ttvs, matched_obs_ttvs, matching_quality
end

function julia_log_prior_gradient(params_vector)
    params = params_vector_to_mcmc(params_vector)
    
    log_prior = 0.0
    obs_period = OBSERVED_DATA.obs_period
    
    log_prior += -0.5 * ((params.log_mass1 - log10(0.001)) / 0.3)^2
    log_prior += -0.5 * ((params.log_mass2 - log10(0.0003)) / 0.3)^2
    log_prior += -0.5 * ((params.period1 - obs_period) / 2.0)^2
    log_prior += -0.5 * ((params.period_ratio - 2.5) / 0.3)^2
    log_prior += -0.5 * ((params.kappa - 2.487) / 0.05)^2
    log_prior += -0.5 * ((params.tsys - 605) / 10.0)^2
    
    if params.e1 > 0
        log_prior += -params.e1 / 0.002
    else
        log_prior += -1000
    end
    
    if params.e2 > 0
        log_prior += -params.e2 / 0.002
    else
        log_prior += -1000
    end
    
    log_prior += -0.5 * (params.mean_anomaly1 / 1.0)^2
    log_prior += -0.5 * (params.mean_anomaly2 / 1.0)^2
    log_prior += -0.5 * ((params.omega_diff - π) / 1.0)^2
    
    if params.e1 > 0.1 || params.e2 > 0.1
        log_prior += -1000
    end
    
    if abs(params.period1 - obs_period) > 20
        log_prior += -1000
    end
    
    if params.period_ratio < 1.5 || params.period_ratio > 4.0
        log_prior += -1000
    end
    
    if abs(params.mean_anomaly1) > π || abs(params.mean_anomaly2) > π
        log_prior += -1000
    end
    
    if abs(params.omega_diff) > 2π
        log_prior += -1000
    end
    
    return log_prior
end

function julia_likelihood_gradient(params_vector)
    try
        params = params_vector_to_mcmc(params_vector)
        
        period_diff = abs(params.period1 - OBSERVED_DATA.obs_period)
        if period_diff > 20
            return -1000 - period_diff^2
        elseif period_diff > 5
            return -100 - period_diff^2
        end
        
        model_result = compute_model_ttvs_final(params, OBSERVED_DATA)
        if model_result === nothing
            penalty = 0.0
            penalty += (period_diff / 2.0)^2
            penalty += ((params.period_ratio - 2.5) / 0.5)^2
            penalty += max(0, params.e1 - 0.01)^2 * 1000
            penalty += max(0, params.e2 - 0.01)^2 * 1000
            return -50 - penalty
        end
        
        model_transit_times, model_ttvs, model_ephemeris = model_result
        
        match_result = match_transits_final(model_transit_times, model_ttvs, model_ephemeris, OBSERVED_DATA)
        if match_result === nothing
            return -20 - period_diff^2
        end
        
        matched_model_ttvs, matched_obs_ttvs, matching_quality = match_result
        
        residuals = matched_obs_ttvs .- matched_model_ttvs
        obs_ttv_range = maximum(OBSERVED_DATA.ttv_observations) - minimum(OBSERVED_DATA.ttv_observations)
        base_uncertainty = max(15.0, obs_ttv_range / 10)
        uncertainties = base_uncertainty .+ matching_quality * 100
        
        chi_squared = sum((residuals ./ uncertainties).^2)
        
        model_t0_aligned, model_period = model_ephemeris
        period_penalty = ((params.period1 - OBSERVED_DATA.obs_period) / 5.0)^2
        fitted_period_penalty = ((model_period - OBSERVED_DATA.obs_period) / 3.0)^2
        
        n_data = length(matched_model_ttvs)
        normalized_chi_squared = chi_squared / n_data + period_penalty + fitted_period_penalty
        
        log_likelihood = -0.5 * normalized_chi_squared
        
        return log_likelihood
        
    catch e
        params = params_vector_to_mcmc(params_vector)
        penalty = abs(params.period1 - OBSERVED_DATA.obs_period) / 2.0
        return -100 - penalty^2
    end
end

function create_python_likelihood_gradient()
    py"""
    def log_prob_wrapper(params):
        import numpy as np
        
        try:
            params_list = params.tolist()
            
            log_prior_val = $julia_log_prior_gradient(params_list)
            if log_prior_val < -500:
                return log_prior_val
                
            log_likelihood_val = $julia_likelihood_gradient(params_list)
            return log_likelihood_val + log_prior_val
            
        except Exception as e:
            try:
                params_list = params.tolist()
                period_penalty = abs(params_list[2] - 363.2) / 2.0
                return -200 - period_penalty**2
            except:
                return -1000
    """
    
    return py"log_prob_wrapper"
end

function test_gradient_likelihood()
    ttv_filename = "bestfit_orbit_planet2_ttvs.txt"
    observed_data = load_ttv_data(ttv_filename)
    setup_likelihood_function(observed_data)
    
    test_params_good = [
        log10(0.001), log10(0.0003), 363.28, 2.48, 0.001, 0.002,
        0.0, 0.0, π, 2.487, 605.0
    ]
    
    test_params_far = [
        log10(0.002), log10(0.0005), 365.0, 2.6, 0.003, 0.004,
        0.2, -0.2, 2.5, 2.45, 600.0
    ]
    
    prior_good = julia_log_prior_gradient(test_params_good)
    like_good = julia_likelihood_gradient(test_params_good)
    
    prior_far = julia_log_prior_gradient(test_params_far)
    like_far = julia_likelihood_gradient(test_params_far)
    
    if isfinite(prior_good) && isfinite(prior_far) && isfinite(like_good) && isfinite(like_far)
        return true
    else
        return false
    end
end

function generate_conservative_walkers(observed_data::TTVData, n_walkers::Int=30)
    obs_period = observed_data.obs_period
    
    center_point = [
        log10(0.0015),
        log10(0.0004),
        obs_period + 1.0,
        2.6,
        0.002,
        0.003,
        0.2,
        -0.2,
        2.5,
        2.45,
        600.0
    ]
    
    noise_scales = [0.05, 0.05, 0.5, 0.05, 0.0005, 0.0005, 0.1, 0.1, 0.2, 0.01, 2.0]
    
    walkers = []
    
    for i in 1:n_walkers
        candidate = copy(center_point)
        
        for j in 1:length(candidate)
            candidate[j] += randn() * noise_scales[j]
        end
        
        if julia_log_prior_gradient(candidate) > -500
            push!(walkers, candidate)
        else
            candidate = copy(center_point)
            for j in 1:length(candidate)
                candidate[j] += randn() * noise_scales[j] * 0.1
            end
            push!(walkers, candidate)
        end
    end
    
    return np.array(walkers)
end

function run_emcee_sampler_gradient(observed_data::TTVData; n_walkers::Int=30, n_steps::Int=1000, n_burn::Int=300)
    setup_likelihood_function(observed_data)
    log_prob_func = create_python_likelihood_gradient()
    
    initial_walkers = generate_conservative_walkers(observed_data, n_walkers)
    n_dim = size(initial_walkers, 2)
    
    sampler = emcee.EnsembleSampler(size(initial_walkers, 1), n_dim, log_prob_func)
    
    pos, prob, state = sampler.run_mcmc(initial_walkers, n_burn, progress=true)
    sampler.reset()
    
    pos, prob, state = sampler.run_mcmc(pos, n_steps, progress=true, skip_initial_state_check=true)
    
    samples = sampler.get_chain(flat=true)
    log_probs = sampler.get_log_prob(flat=true)
    
    return samples, log_probs, sampler
end

function check_convergence(sampler)
    try
        tau = sampler.get_autocorr_time()
        return tau
    catch e
        return nothing
    end
end

function check_chain_quality(samples, param_names)
    n_samples, n_params = size(samples)
    
    for i in 1:n_params
        param_data = samples[:, i]
        n_finite = sum(isfinite.(param_data))
        n_inf = sum(isinf.(param_data))
        n_nan = sum(isnan.(param_data))
    end
    
    total_finite = sum([all(isfinite.(samples[i, :])) for i in 1:size(samples, 1)])
    
    if total_finite / n_samples < 0.5
        return false
    elseif total_finite / n_samples < 0.9
        return true
    else
        return true
    end
end

function plot_chains(samples, param_names; n_show=6)
    n_params = min(n_show, length(param_names))
    
    figure(figsize=(15, 10))
    for i in 1:n_params
        subplot(3, 2, i)
        plot(samples[:, i], alpha=0.7, linewidth=0.5)
        title(param_names[i])
        ylabel("Parameter value")
        xlabel("Sample number")
        grid(true, alpha=0.3)
    end
    tight_layout()
    savefig("chain_traces.png", dpi=150, bbox_inches="tight")
end

function create_corner_plot(samples, param_names, true_values; burn_fraction=0.5, n_show=6)
    n_samples = size(samples, 1)
    start_idx = Int(round(n_samples * burn_fraction))
    converged_samples = samples[start_idx:end, :]
    
    valid_rows = []
    for i in 1:size(converged_samples, 1)
        if all(isfinite.(converged_samples[i, :]))
            push!(valid_rows, i)
        end
    end
    
    if length(valid_rows) < 100
        return
    end
    
    clean_samples = converged_samples[valid_rows, :]
    
    key_indices = [1, 2, 3, 4, 5, 6]
    key_names = ["Mass₁ (M☉)", "Mass₂ (M☉)", "Period₁ (d)", "P₂/P₁", "e₁", "e₂"]
    key_samples = clean_samples[:, key_indices]
    key_true = true_values[key_indices]
    
    key_samples[:, 1] = 10 .^ key_samples[:, 1]
    key_samples[:, 2] = 10 .^ key_samples[:, 2]
    key_true[1] = 10^key_true[1]
    key_true[2] = 10^key_true[2]
    
    n_params = length(key_names)
    
    figure(figsize=(14, 14))
    
    for i in 1:n_params
        for j in 1:n_params
            subplot(n_params, n_params, (i-1)*n_params + j)
            
            if i == j
                param_data = key_samples[:, i]
                
                if std(param_data) < 1e-15
                    axvline(param_data[1], color="blue", linewidth=3, label="Constant")
                    axvline(key_true[i], color="red", linestyle="--", linewidth=2, label="True")
                    xlim(param_data[1] - abs(param_data[1])*0.1, param_data[1] + abs(param_data[1])*0.1)
                else
                    try
                        hist(param_data, bins=30, alpha=0.7, density=true, 
                             color="skyblue", edgecolor="navy", linewidth=0.5)
                        axvline(key_true[i], color="red", linestyle="--", linewidth=2, 
                               label="True", alpha=0.8)
                        axvline(mean(param_data), color="orange", linestyle="-", 
                               linewidth=2, label="Mean", alpha=0.8)
                        
                        if std(param_data) > 1e-10
                            p16, p84 = quantile(param_data, [0.16, 0.84])
                            axvspan(p16, p84, alpha=0.2, color="orange", label="68% CI")
                        end
                    catch e
                        axvline(mean(param_data), color="blue", linewidth=3, label="Mean")
                        axvline(key_true[i], color="red", linestyle="--", linewidth=2, label="True")
                    end
                end
                
                if i == 1
                    legend(fontsize=8, loc="upper right")
                end
                
                if i == 1 || i == 2
                    gca().ticklabel_format(style="scientific", axis="x", scilimits=(0,0))
                end
                
            elseif i > j
                x_data = key_samples[:, j]
                y_data = key_samples[:, i]
                
                if std(x_data) < 1e-15 && std(y_data) < 1e-15
                    scatter([x_data[1]], [y_data[1]], color="blue", s=100, marker="o", label="Data")
                elseif std(x_data) < 1e-15
                    y_range = [minimum(y_data), maximum(y_data)]
                    plot([x_data[1], x_data[1]], y_range, color="blue", linewidth=2, label="Data")
                elseif std(y_data) < 1e-15
                    x_range = [minimum(x_data), maximum(x_data)]
                    plot(x_range, [y_data[1], y_data[1]], color="blue", linewidth=2, label="Data")
                else
                    n_plot = min(2000, size(key_samples, 1))
                    idx = rand(1:size(key_samples, 1), n_plot)
                    scatter(x_data[idx], y_data[idx], 
                           alpha=0.1, s=1, color="blue", rasterized=true)
                end
                
                scatter([key_true[j]], [key_true[i]], color="red", s=100, 
                       marker="*", label="True", zorder=10, edgecolors="black", linewidth=0.5)
                
                scatter([mean(x_data)], [mean(y_data)], 
                       color="orange", s=60, marker="o", label="Mean", zorder=9,
                       edgecolors="black", linewidth=0.5)
                
                if j == 1 || j == 2
                    gca().ticklabel_format(style="scientific", axis="x", scilimits=(0,0))
                end
                if i == 1 || i == 2
                    gca().ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
                end
                
            else
                axis("off")
            end
            
            if i == n_params
                xlabel(key_names[j], fontsize=12)
            else
                gca().set_xticklabels([])
            end
            
            if j == 1 && i > 1
                ylabel(key_names[i], fontsize=12)
            else
                gca().set_yticklabels([])
            end
            
            gca().set_aspect("auto")
        end
    end
    
    suptitle("TTV Parameter Recovery - Posterior Distributions", fontsize=16, y=0.98)
    tight_layout(rect=[0, 0, 1, 0.96])
    
    try
        savefig("ttv_corner_plot.png", dpi=200, bbox_inches="tight")
        savefig("ttv_corner_plot.pdf", bbox_inches="tight")
    catch e
    end
end

function plot_ttv_fit(best_sample, observed_data)
    try
        params = params_vector_to_mcmc(best_sample)
        
        model_result = compute_model_ttvs_final(params, observed_data)
        if model_result !== nothing
            model_transit_times, model_ttvs, model_ephemeris = model_result
            
            match_result = match_transits_final(model_transit_times, model_ttvs, model_ephemeris, observed_data)
            if match_result !== nothing
                matched_model_ttvs, matched_obs_ttvs, matching_quality = match_result
                
                residuals = matched_obs_ttvs .- matched_model_ttvs
                rms = sqrt(mean(residuals.^2))
                
                figure(figsize=(14, 10))
                
                subplot(2, 1, 1)
                plot(1:length(matched_obs_ttvs), matched_obs_ttvs, "o", 
                     label="Observed", markersize=8, color="blue", alpha=0.7)
                plot(1:length(matched_model_ttvs), matched_model_ttvs, "s", 
                     label="Best-fit Model", markersize=6, color="red", alpha=0.7)
                xlabel("Transit Index")
                ylabel("TTV (minutes)")
                title("TTV Fit (RMS = $(round(rms, digits=2)) min)")
                legend()
                grid(true, alpha=0.3)
                
                subplot(2, 1, 2)
                plot(1:length(residuals), residuals, "o", color="green", markersize=6, alpha=0.7)
                axhline(0, color="black", linestyle="--", alpha=0.5)
                axhline(rms, color="red", linestyle=":", alpha=0.7, label="±$(round(rms, digits=1)) min")
                axhline(-rms, color="red", linestyle=":", alpha=0.7)
                xlabel("Transit Index")
                ylabel("Residuals (minutes)")
                title("TTV Residuals")
                legend()
                grid(true, alpha=0.3)
                
                tight_layout()
                savefig("ttv_best_fit.png", dpi=150, bbox_inches="tight")
            end
        end
    catch e
    end
end

function analyze_final_results(samples, log_probs, observed_data)
    best_idx = argmax(log_probs)
    best_sample = samples[best_idx, :]
    best_ll = log_probs[best_idx]
    
    param_names = ["log_mass1", "log_mass2", "period1", "period_ratio", "e1", "e2", 
                   "mean_anom1", "mean_anom2", "omega_diff", "kappa", "tsys"]
    
    true_values = [log10(0.000952), log10(0.000286), 363.2, 2.48, 0.000843, 0.001776,
                   0.0, 0.0, π, 2.487, 605.3]
    
    n_samples = size(samples, 1)
    start_idx = Int(round(n_samples * 0.5))
    converged_samples = samples[start_idx:end, :]
    
    return best_sample, converged_samples
end

function main()
    if test_gradient_likelihood()
        ttv_filename = "bestfit_orbit_planet2_ttvs.txt"
        observed_data = load_ttv_data(ttv_filename)
        
        samples, log_probs, sampler = run_emcee_sampler_gradient(observed_data, 
                                                               n_walkers=42,
                                                               n_steps=2000,
                                                               n_burn=500)
        
        best_sample, converged_samples = analyze_final_results(samples, log_probs, observed_data)
        
        tau = check_convergence(sampler)
        
        param_names = ["log_mass1", "log_mass2", "period1", "period_ratio", "e1", "e2", 
                       "mean_anom1", "mean_anom2", "omega_diff", "kappa", "tsys"]
        
        true_values = [log10(0.000952), log10(0.000286), 363.2, 2.48, 0.000843, 0.001776,
                       0.0, 0.0, π, 2.487, 605.3]
        
        chain_ok = check_chain_quality(samples, param_names)
        
        plot_chains(samples, param_names, n_show=6)
        
        if chain_ok
            create_corner_plot(samples, param_names, true_values, burn_fraction=0.5, n_show=6)
        end
        
        plot_ttv_fit(best_sample, observed_data)
        
        return samples, log_probs, best_sample, converged_samples
    else
        return nothing
    end
end

results = main()
