using DelimitedFiles
using PeriodicOrbitTTV

using JLD2

using PyPlot
using LaTeXStrings
using Distributions, LinearAlgebra, StatsBase

using LsqFit
using BenchmarkTools

FILENAME = split(@__FILE__, "/")[end]
DESCRIPTION = """
Simple TTV fit for 500-day-long data. PO Weight at 1e-16

Execution time is measured with BenchmarkTools.jl
With and Without Jacobians

Experiment date 2026-08-03
"""
SOLUTION = "sample_orbit_v1.in"

function rename(str::AbstractString; prefix=nothing)
    split_str = split(str, ".")
    new_str = join([split_str[1], "jld2"], ".")

    if prefix != nothing
        new_str = join([prefix, new_str], "_")
    end

    return new_str
end

@show SAVE_FILENAME = rename(FILENAME; prefix="run")

# Load the data for all three noise levels
obj_01s = jldopen("run_freq_multi_noise_all_01_01s.jld2");
obj_03s = jldopen("run_freq_multi_noise_all_01_03s.jld2");
obj_10s = jldopen("run_freq_multi_noise_all_01_10s.jld2");

clean_params = vec(readdlm("../frequentist/sample_orbit_v1.in"))
clean_params[4:8] .= rem2pi.(clean_params[4:8], RoundNearest)

function get_valid_indices(optres_arr)
    idx_arr = []
    for i in eachindex(optres_arr)
        if isassigned(optres_arr[i], length(optres_arr[i]))
            push!(idx_arr, i)
        end
    end

    return idx_arr
end

function find_converged_pott_idx(optres_arr, po_weight; valid_idx=valid_idx)
    idx_arr = []
    for i in eachindex(optres_arr)
        if i in valid_idx
            chi2 = sum(abs2, optres_arr[i][end].resid[1:26])
            chi2_tt = sum(abs2, optres_arr[i][end].resid[27:end])
        
            if (chi2 ./ po_weight < 1e-12) & (chi2_tt <= 1e2)
                push!(idx_arr, i)
            end
        end
    end

    return idx_arr
end

function find_converged_bad_tt_idx(optres_arr, po_weight; valid_idx=valid_idx)
    idx_arr = []
    for i in eachindex(optres_arr)
        if i in valid_idx
            chi2 = sum(abs2, optres_arr[i][end].resid[1:26])
            chi2_tt = sum(abs2, optres_arr[i][end].resid[27:end])
        
            if (chi2 ./ po_weight < 1e-12) & (chi2_tt > 1e3)
                push!(idx_arr, i)
            end
        end
    end

    return idx_arr
end

valid_idx_01s = get_valid_indices(obj_01s["optres"])
converged_idx_01s = find_converged_pott_idx(obj_01s["optres"], obj_01s["weights"][end]; valid_idx=valid_idx_01s);

valid_idx_03s = get_valid_indices(obj_03s["optres"])
converged_idx_03s = find_converged_pott_idx(obj_03s["optres"], obj_03s["weights"][end]; valid_idx=valid_idx_03s);

valid_idx_10s = get_valid_indices(obj_10s["optres"])
converged_idx_10s = find_converged_pott_idx(obj_10s["optres"], obj_10s["weights"][end]; valid_idx=valid_idx_10s);

valid_all = intersect(valid_idx_01s, valid_idx_03s, valid_idx_10s)
converged_all = intersect(converged_idx_01s, converged_idx_03s, converged_idx_10s)

# Code roughly based on the parallel script

function to_matrix(tt::Matrix{T}; err=0.) where T <: Real
    tt_mat = Matrix{T}(undef, 0, 4)
    
    for i in 1:size(tt, 1), j in 1:size(tt, 2)
        if tt[i,j] != 0.0
            tt_mat = vcat(tt_mat, [i-1, j-1, tt[i,j], err]')
        end
    end

    errors = rand(Normal(0., err), size(tt_mat, 1))
    tt_mat[:,3] .= tt_mat[:,3] + errors
    
    return tt_mat
end

using PeriodicOrbitTTV: compute_diff_squared_jacobian, compute_diff_squared

function find_first_transit(nplanet::Int, data::Matrix{T}) where T <: Real
    row_index = [findfirst(isequal(i), data[:,1]) for i = 1:nplanet] 

    return data[row_index, 3]
end

function find_last_transit(nplanet::Int, data::Matrix{T}) where T <: Real
    row_index = [findlast(isequal(i), data[:,1]) for i = 1:nplanet] 

    return data[row_index, 3]
end

function residues(optvec::Vector{T}, truthvec::Vector{T}, weights_periodic::Vector{T}, nplanet::Int, orbparams::OrbitParameters{T}, tt_data::Matrix{T}, scaler::Vector{T}) where T <: Real
    residues = zeros(T, 9nplanet - 1 + size(tt_data, 1))
    optvec = optvec .* scaler
    
    # PO Contribution
    orbit = Orbit(nplanet, OptimParameters(nplanet, optvec), orbparams)
    residues[1:4nplanet-2] = (param_diff(3, orbit.final_elem, orbit.init_elem)[1:end-2]) ./ scaler[1:4nplanet-2]

    # Prior contribution
    residues[4nplanet-1:9nplanet-1] .= 0
    
    # TT Contribution
    tt = compute_tt(orbit, orbparams.obstmax)
    tmod, ip, jp = match_transits(tt_data, orbit, tt.tt, tt.count, nothing)
    # tt_mat = to_matrix(tt.tt)
    residues[9nplanet:end] = tmod
    
    return residues
end

# Read

# clean_params = vec(readdlm("sample_orbit_v1.in"))

orbparams = OrbitParameters(3, [0.5], 500.)
optparams = OptimParameters(3, clean_params)

orbit_0 = Orbit(3, optparams, orbparams)
tt = compute_tt(orbit_0, orbparams.obstmax);

function setup_initial_params(tt_data)
    first_transits = find_first_transit(3, tt_data)
    omegas = [0.0001, pi, 0.]
    init_e = [0.001, 0.001, 0.001]
    omega_diffs = [omegas[i] - omegas[i-1] for i = 2:length(omegas)]
    fitted_periods = [estimate_period(tt_data, i, [10., 0.])[1][1] for i in 1:3]
    init_Ms = estimate_initial_M(first_transits, fitted_periods, omegas)
    kappa, pdev = calculate_period_deviation(vcat(fitted_periods), [0.5])
    guess_mass = [1e-8, 1e-8, 1e-8]
    
    P1_0 = fitted_periods[1]
    
    optvec = vcat(init_e,
        vcat(init_Ms),
        omega_diffs,
        pdev,
        P1_0,
        guess_mass,
        kappa,
        omegas[1],
        3.98*P1_0
    )
    
    scaler = 10 .^ round.(log10.(abs.(optvec)))
    optvec = optvec ./ scaler
    return optvec, scaler
end

# Load the data
tt_data = jldopen("run_freq_multi_noise_01s_07_03s_data_only.jld2")["tt"][converged_all[1]]

# Prepare boundaries

current_optvec, scaler = setup_initial_params(tt_data)

lower_bounds = [
    -Inf, -Inf, -Inf,
    -Inf, -Inf, -Inf,
    -Inf, -Inf,
    -Inf,
    -Inf,
    1e-12, 1e-12, 1e-12,
    -Inf,
    -Inf,
    -Inf,
] ./ scaler

upper_bounds = [
    Inf, Inf, Inf,
    Inf, Inf, Inf,
    Inf, Inf,
    Inf,
    Inf,
    1e-2, 1e-2, 1e-2,
    Inf,
    Inf,
    Inf,
] ./ scaler

# Perform the optimization

nplanet = 3
po_weight = 1e-16
ydata = vcat(fill(0., 9nplanet-1), tt_data[:,3]);
xdata = fill(0., length(ydata));
ydata_w = vcat(fill(po_weight, 4nplanet-2), zeros(5nplanet+1), abs2.(1 ./ tt_data[:,4]));

wrapper_residues(_, θ) = residues(θ, [0.], fill(po_weight, 10), 3, orbparams, tt_data, scaler)

function llhood_jac(_, θ)
    transformed = θ .* scaler
            
    optparams = OptimParameters(nplanet, transformed)

    jac = compute_diff_squared_jacobian(optparams, orbparams, nplanet, tt_data)
    jac[1:4nplanet-2,:] .= jac[1:4nplanet-2,:] ./ scaler[1:4nplanet-2]

    return jac .* scaler'
end

fitres_with_jac = @benchmark try
    curve_fit(wrapper_residues, llhood_jac, xdata, ydata, ydata_w, current_optvec, lower=lower_bounds, upper=upper_bounds; show_trace=false, maxIter=100);
catch e
    @warn e #"An error occurred in trial $i."
end

fitres_without_jac = @benchmark try
    curve_fit(wrapper_residues, xdata, ydata, ydata_w, current_optvec, lower=lower_bounds, upper=upper_bounds; show_trace=false, maxIter=100);
catch e
    @warn e #"An error occurred in trial $i."
end