include("PeriodicOrbit.jl")
using .PeriodicOrbit
using NbodyGradient
using Statistics
using Random
using PyPlot

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

struct MCMCParameters
    log_mass1::Float64
    log_mass2::Float64
    period1::Float64
    period_ratio::Float64
    e1::Float64
    e2::Float64
    mean_anomaly1::Float64
    mean_anomaly2::Float64
    omega_diff::Float64
    kappa::Float64
    tsys::Float64
end

function mcmc_to_optvec(params::MCMCParameters)
    mass1 = 10^params.log_mass1
    mass2 = 10^params.log_mass2
    
    return [params.e1, params.e2, params.mean_anomaly1, params.mean_anomaly2,
            params.omega_diff, params.period1, mass1, mass2, params.kappa, 0.0, params.tsys]
end

function compute_model_ttvs_conservative(params::MCMCParameters, observed_data::TTVData)
    try
        if abs(params.period1 - observed_data.obs_period) > 3.0
            return nothing
        end
        
        optvec = mcmc_to_optvec(params)
        optparams = OptimParameters(2, optvec)
        orbparams = OrbitParameters(2, [0.5])
        
        orbit = Orbit(2, optparams, orbparams)
        elems = get_orbital_elements(orbit.s, orbit.ic)
        
        derived_period = elems[2].P
        if abs(derived_period - observed_data.obs_period) > 2.0
            return nothing
        end
        
        star_edgeon = Elements(m=1.0)
        planet1_edgeon = Elements(m=elems[2].m, P=elems[2].P, e=0.01, ω=0.0, I=π/2, Ω=0.0, t0=0.0)
        planet2_edgeon = Elements(m=elems[3].m, P=elems[3].P, e=0.01, ω=π, I=π/2, Ω=0.0, t0=0.0)
        
        el_ic_edgeon = ElementsIC(0.0, 3, star_edgeon, planet1_edgeon, planet2_edgeon)
        
        P_inner = elems[2].P
        h_int = P_inner / 100.0
        tmax = P_inner * 40
        
        state_nbody = State(el_ic_edgeon)
        tt_nbody = TransitTiming(tmax, el_ic_edgeon)
        Integrator(ahl21!, h_int, tmax)(state_nbody, tt_nbody)
        
        planet_idx = observed_data.planet_id
        if tt_nbody.count[planet_idx] < 10
            return nothing
        end
        
        model_transit_times = tt_nbody.tt[planet_idx, 1:tt_nbody.count[planet_idx]]
        
        n_model = length(model_transit_times)
        A = hcat(ones(n_model), 0:(n_model-1))
        model_ephemeris = A \ model_transit_times
        model_t0, model_period_fitted = model_ephemeris
        
        if abs(model_period_fitted - observed_data.obs_period) > 1.0
            return nothing
        end
        
        model_linear_pred = model_t0 .+ model_period_fitted .* (0:(n_model-1))
        model_ttvs = (model_transit_times .- model_linear_pred) .* 24 * 60
        
        return model_transit_times, model_ttvs, (model_t0, model_period_fitted)
        
    catch e
        return nothing
    end
end

function match_transits_conservative(model_transit_times, model_ttvs, model_ephemeris, observed_data::TTVData)
    if model_transit_times === nothing
        return nothing
    end
    
    model_t0, model_period = model_ephemeris
    
    matched_model_ttvs = Float64[]
    matched_obs_ttvs = Float64[]
    
    max_time_diff = 0.05
    
    for i in 1:length(observed_data.transit_numbers)
        obs_n = observed_data.transit_numbers[i]
        obs_ttv = observed_data.ttv_observations[i]
        
        expected_time = model_t0 + model_period * obs_n
        
        time_diffs = abs.(model_transit_times .- expected_time)
        closest_idx = argmin(time_diffs)
        
        if time_diffs[closest_idx] < max_time_diff
            push!(matched_model_ttvs, model_ttvs[closest_idx])
            push!(matched_obs_ttvs, obs_ttv)
        end
    end
    
    if length(matched_model_ttvs) < 27
        return nothing
    end
    
    return matched_model_ttvs, matched_obs_ttvs
end

function log_likelihood_conservative(params::MCMCParameters, observed_data::TTVData)
    model_result = compute_model_ttvs_conservative(params, observed_data)
    if model_result === nothing
        return -Inf
    end
    
    model_transit_times, model_ttvs, model_ephemeris = model_result
    
    match_result = match_transits_conservative(model_transit_times, model_ttvs, model_ephemeris, observed_data)
    if match_result === nothing
        return -Inf
    end
    
    matched_model_ttvs, matched_obs_ttvs = match_result
    
    residuals = matched_obs_ttvs .- matched_model_ttvs
    uncertainties = 1.0
    chi_squared = sum((residuals ./ uncertainties).^2)
    
    model_t0, model_period = model_ephemeris
    period_penalty = ((params.period1 - observed_data.obs_period) / 5.0)^2
    fitted_period_penalty = ((model_period - observed_data.obs_period) / 2.0)^2
    
    total_chi_squared = chi_squared + period_penalty + fitted_period_penalty
    
    return -0.5 * total_chi_squared
end

function log_prior_conservative(params::MCMCParameters, observed_data::TTVData)
    if params.log_mass1 < log10(0.0001) || params.log_mass1 > log10(0.005); return -Inf; end
    if params.log_mass2 < log10(0.00005) || params.log_mass2 > log10(0.002); return -Inf; end
    
    obs_period = observed_data.obs_period
    if params.period1 < (obs_period - 8.0) || params.period1 > (obs_period + 8.0); return -Inf; end
    
    if params.period_ratio < 2.0 || params.period_ratio > 3.0; return -Inf; end
    
    if params.e1 < 0 || params.e1 > 0.015; return -Inf; end
    if params.e2 < 0 || params.e2 > 0.015; return -Inf; end
    
    if abs(params.mean_anomaly1) > π; return -Inf; end
    if abs(params.mean_anomaly2) > π; return -Inf; end
    if abs(params.omega_diff) > π; return -Inf; end
    
    if params.kappa < 2.2 || params.kappa > 2.8; return -Inf; end
    if params.tsys < 580 || params.tsys > 620; return -Inf; end
    
    return 0.0
end

function propose_step_conservative(current_params::MCMCParameters, step_sizes::Dict)
    return MCMCParameters(
        current_params.log_mass1 + randn() * step_sizes[:log_mass1],
        current_params.log_mass2 + randn() * step_sizes[:log_mass2],
        current_params.period1 + randn() * step_sizes[:period1],
        current_params.period_ratio + randn() * step_sizes[:period_ratio],
        max(0.0, current_params.e1 + randn() * step_sizes[:e1]),
        max(0.0, current_params.e2 + randn() * step_sizes[:e2]),
        current_params.mean_anomaly1 + randn() * step_sizes[:mean_anomaly1],
        current_params.mean_anomaly2 + randn() * step_sizes[:mean_anomaly2],
        current_params.omega_diff + randn() * step_sizes[:omega_diff],
        current_params.kappa + randn() * step_sizes[:kappa],
        current_params.tsys + randn() * step_sizes[:tsys]
    )
end

function find_exploratory_starting_point(observed_data::TTVData, max_attempts::Int=10000)
    obs_period = observed_data.obs_period
    best_ll = -Inf
    best_params = nothing
    
    for attempt in 1:max_attempts
        candidate = MCMCParameters(
            log10(0.0001) + (log10(0.005) - log10(0.0001)) * rand(),
            log10(0.00005) + (log10(0.002) - log10(0.00005)) * rand(),
            obs_period + 10.0 * (2 * rand() - 1),
            2.0 + 1.0 * rand(),
            0.01 * rand(),
            0.01 * rand(),
            2π * rand() - π,
            2π * rand() - π,
            2π * rand() - π,
            2.3 + 0.4 * rand(),
            580.0 + 40.0 * rand()
        )
        
        if log_prior_conservative(candidate, observed_data) != -Inf
            ll = log_likelihood_conservative(candidate, observed_data)
            
            if ll > best_ll
                best_ll = ll
                best_params = candidate
                println("  Attempt $attempt: NEW BEST LL = $(round(ll, digits=1))")
            end
        end
        
        if attempt % 100 == 0
            println("  Progress: $attempt/$max_attempts attempts")
        end
    end
    
    if best_params === nothing
        error("Random search failed to find any valid starting point!")
    end
    
    return best_params, best_ll
end

function run_conservative_mcmc(observed_data::TTVData; n_iterations::Int=20000, n_burn::Int=6000, n_thin::Int=3)
    current_params, start_ll = find_exploratory_starting_point(observed_data, 500000)
    
    step_sizes = Dict(
        :log_mass1 => 0.0005,
        :log_mass2 => 0.0005,
        :period1 => 0.01,
        :period_ratio => 0.0002,
        :e1 => 0.000005,
        :e2 => 0.000005,
        :mean_anomaly1 => 0.001,
        :mean_anomaly2 => 0.001,
        :omega_diff => 0.001,
        :kappa => 0.0001,
        :tsys => 0.1
    )
    
    chain = []
    log_likelihoods = []
    
    current_log_likelihood = start_ll
    current_log_prior = log_prior_conservative(current_params, observed_data)
    current_log_posterior = current_log_likelihood + current_log_prior
    
    n_accepted = 0
    best_ll = current_log_likelihood
    n_stored = 0
    
    for i in 1:n_iterations
        proposed_params = propose_step_conservative(current_params, step_sizes)
        
        proposed_log_prior = log_prior_conservative(proposed_params, observed_data)
        if proposed_log_prior == -Inf
            if i % n_thin == 0
                push!(chain, current_params)
                push!(log_likelihoods, current_log_likelihood)
                n_stored += 1
            end
            continue
        end
        
        proposed_log_likelihood = log_likelihood_conservative(proposed_params, observed_data)
        if proposed_log_likelihood == -Inf
            if i % n_thin == 0
                push!(chain, current_params)
                push!(log_likelihoods, current_log_likelihood)
                n_stored += 1
            end
            continue
        end
        
        proposed_log_posterior = proposed_log_likelihood + proposed_log_prior
        
        if proposed_log_likelihood > best_ll
            best_ll = proposed_log_likelihood
        end
        
        log_alpha = proposed_log_posterior - current_log_posterior
        
        if log(rand()) < log_alpha
            current_params = proposed_params
            current_log_likelihood = proposed_log_likelihood
            current_log_posterior = proposed_log_posterior
            n_accepted += 1
        end
        
        if i % n_thin == 0
            push!(chain, current_params)
            push!(log_likelihoods, current_log_likelihood)
            n_stored += 1
        end
        
        if i % 1000 == 0
            acceptance_rate = n_accepted / i * 100
            println("Iteration $i: Current LL = $(round(current_log_likelihood, digits=1)), Best = $(round(best_ll, digits=1)), Accept = $(round(acceptance_rate, digits=1))%")
        end
    end
    
    final_acceptance = n_accepted / n_iterations * 100
    println("Conservative MCMC complete:")
    println("  Total iterations: $n_iterations")
    println("  Stored samples: $n_stored")
    println("  Final acceptance rate: $(round(final_acceptance, digits=1))%")
    println("  Best likelihood: $(round(best_ll, digits=1))")
    
    return chain, log_likelihoods, n_burn ÷ n_thin
end

function analyze_conservative_results(chain, log_likelihoods, n_burn, observed_data)
    post_burn_chain = chain[(n_burn+1):end]
    post_burn_ll = log_likelihoods[(n_burn+1):end]
    
    best_idx = argmax(post_burn_ll)
    best_params = post_burn_chain[best_idx]
    
    true_mass1, true_mass2 = 0.000952, 0.000286
    true_period1 = 363.2
    true_ratio = 2.48
    true_e1, true_e2 = 0.000843, 0.001776
    true_kappa = 2.487
    
    mass1_samples = [10^p.log_mass1 for p in post_burn_chain]
    mass2_samples = [10^p.log_mass2 for p in post_burn_chain]
    period1_samples = [p.period1 for p in post_burn_chain]
    ratio_samples = [p.period_ratio for p in post_burn_chain]
    e1_samples = [p.e1 for p in post_burn_chain]
    e2_samples = [p.e2 for p in post_burn_chain]
    kappa_samples = [p.kappa for p in post_burn_chain]
    
    function print_param_stats(name, samples, true_val, format_digits=6)
        mean_val = mean(samples)
        std_val = std(samples)
        bias = abs(mean_val - true_val) / true_val * 100
        
        println("$name:")
        println("  True value:    $(round(true_val, digits=format_digits))")
        println("  Recovered:     $(round(mean_val, digits=format_digits)) ± $(round(std_val, digits=format_digits))")
        println("  Bias:          $(round(bias, digits=1))%")
        println("  68% credible:  [$(round(quantile(samples, 0.16), digits=format_digits)), $(round(quantile(samples, 0.84), digits=format_digits))]")
        println()
    end
    
    print_param_stats("Mass 1 (M_sun)", mass1_samples, true_mass1)
    print_param_stats("Mass 2 (M_sun)", mass2_samples, true_mass2)
    print_param_stats("Period 1 (days)", period1_samples, true_period1, 3)
    print_param_stats("Period ratio", ratio_samples, true_ratio, 4)
    print_param_stats("Eccentricity 1", e1_samples, true_e1)
    print_param_stats("Eccentricity 2", e2_samples, true_e2)
    print_param_stats("Kappa", kappa_samples, true_kappa, 4)
    
    model_result = compute_model_ttvs_conservative(best_params, observed_data)
    if model_result !== nothing
        model_times, model_ttvs, model_eph = model_result
        match_result = match_transits_conservative(model_times, model_ttvs, model_eph, observed_data)
        
        if match_result !== nothing
            matched_model, matched_obs = match_result
            residuals = matched_obs .- matched_model
            rms = sqrt(mean(residuals.^2))
            
            figure(figsize=(12, 6))
            plot(1:length(matched_obs), matched_obs, "o", label="Observed", markersize=8)
            plot(1:length(matched_model), matched_model, "s", label="Best-fit", markersize=6)
            xlabel("Transit number")
            ylabel("TTV (minutes)")
            title("Conservative TTV Fit (RMS = $(round(rms, digits=2)) min)")
            legend()
            grid(true, alpha=0.3)
            savefig("conservative_ttv_fit.png", dpi=150, bbox_inches="tight")
        end
    end
    
    return best_params, post_burn_chain
end

function create_simple_corner_plot(samples_dict, true_values_dict)
    param_names = ["Mass 1", "Mass 2", "Period 1", "Ratio", "e₁", "e₂"]
    samples_list = [samples_dict[:mass1], samples_dict[:mass2], samples_dict[:period1], 
                   samples_dict[:ratio], samples_dict[:e1], samples_dict[:e2]]
    true_list = [true_values_dict[:mass1], true_values_dict[:mass2], true_values_dict[:period1],
                true_values_dict[:ratio], true_values_dict[:e1], true_values_dict[:e2]]
    
    n_params = length(param_names)
    
    figure(figsize=(12, 12))
    
    for i in 1:n_params
        for j in 1:n_params
            subplot(n_params, n_params, (i-1)*n_params + j)
            
            if i == j
                hist(samples_list[i], bins=30, alpha=0.7, density=true, color="skyblue")
                axvline(true_list[i], color="red", linestyle="--", linewidth=2, label="True")
                axvline(mean(samples_list[i]), color="orange", linestyle="-", linewidth=2, label="Mean")
                
                if i == 1
                    legend(fontsize=10)
                end
                
            elseif i > j
                idx = rand(1:length(samples_list[i]), min(1000, length(samples_list[i])))
                scatter(samples_list[j][idx], samples_list[i][idx], alpha=0.3, s=1, color="blue")
                scatter([true_list[j]], [true_list[i]], color="red", s=80, marker="*")
                scatter([mean(samples_list[j])], [mean(samples_list[i])], color="orange", s=40, marker="o")
            else
                axis("off")
            end
            
            if i == n_params
                xlabel(param_names[j])
            else
                gca().set_xticklabels([])
            end
            
            if j == 1 && i > 1
                ylabel(param_names[i])
            else
                gca().set_yticklabels([])
            end
        end
    end
    
    tight_layout()
    savefig("conservative_corner_plot.png", dpi=200, bbox_inches="tight")
end

function main()
    ttv_filename = "bestfit_orbit_planet2_ttvs.txt"
    if !isfile(ttv_filename)
        println("Error: TTV data file not found")
        return
    end
    
    observed_data = load_ttv_data(ttv_filename)
    
    chain, log_likelihoods, n_burn = run_conservative_mcmc(observed_data)
    
    best_params, post_burn_samples = analyze_conservative_results(chain, log_likelihoods, n_burn, observed_data)
    
    samples_dict = Dict(
        :mass1 => [10^p.log_mass1 for p in post_burn_samples],
        :mass2 => [10^p.log_mass2 for p in post_burn_samples],
        :period1 => [p.period1 for p in post_burn_samples],
        :ratio => [p.period_ratio for p in post_burn_samples],
        :e1 => [p.e1 for p in post_burn_samples],
        :e2 => [p.e2 for p in post_burn_samples]
    )
    
    true_values_dict = Dict(
        :mass1 => 0.000952, :mass2 => 0.000286, :period1 => 363.2,
        :ratio => 2.48, :e1 => 0.000843, :e2 => 0.001776
    )
    
    create_simple_corner_plot(samples_dict, true_values_dict)
end

main()
