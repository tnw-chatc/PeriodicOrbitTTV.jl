using DelimitedFiles
using PeriodicOrbitTTV

using Distributions
using Random
using JLD2

using LsqFit

FILENAME = split(@__FILE__, "/")[end]
DESCRIPTION = """
Data only

Noise level = 1 second
"""
SOLUTION = "../frequentist/sample_orbit_v1.in"

function rename(str::AbstractString; prefix=nothing)
    split_str = split(str, ".")
    new_str = join([split_str[1], "jld2"], ".")

    if prefix != nothing
        new_str = join([prefix, new_str], "_")
    end

    return new_str
end

@show SAVE_FILENAME = rename(FILENAME; prefix="run")

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
    residues[1:4nplanet-2] = (param_diff(3, orbit.final_elem, orbit.init_elem)[1:end-2])

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

clean_params = vec(readdlm(SOLUTION))

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

# Initialize the experiment
nplanet = 3
n_trials = 10_000
n_continuations = 100
po_weights = vcat(10 .^ LinRange(-24.0, 23.0, n_continuations))

tt_data_list = Vector{Any}(undef, n_trials)
seeds_list = Vector{Any}(undef, n_trials)

# Maybe need to use Distributed.jl
Threads.@threads for i in 1:n_trials    
    Random.seed!(i * 4206967)
    tt_data = to_matrix(tt.tt; err=1/(24*3600))

    tt_data_list[i] = tt_data
    seeds_list[i] = i * 4206967
end

jldsave(SAVE_FILENAME;
    tt = tt_data_list,
    seeds = seeds_list,
    desc = DESCRIPTION,
    filename = FILENAME,
)

