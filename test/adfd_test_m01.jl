using PeriodicOrbitTTV
using PeriodicOrbitTTV: compute_system_init, orbital_to_cartesian, calculate_jac_time_evolution, extract_elements
using NbodyGradient

using FiniteDifferences
using Test

optvec_0 = BigFloat.([0.007634091834320847, 0.015068095357529813, 0.0008875855342026987, -1.4454314813764126, -0.728172966274584, 2.7947084462295138, 3.147750653875983, -3.162598509555049, 0.00017275886406294735, 12.142001475526312, 4.264843854263061e-5, 2.379000115670915e-5, 7.420771108125184e-5, 2.004292806015084, 0.0003710381951454858, 48.46641242308062])

orbparams = OrbitParameters(3, BigFloat.([0.5]), BigFloat(500.))

nplanet = 3
epsilon = BigFloat(eps(Float64))

# ForwardDiff
orbit_0 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

using DelimitedFiles

TT_SOURCE = "sample_orbit_v1_TT_30secs.in"
tt_data = readdlm(TT_SOURCE,'\t',comments=true,comment_char='#');
tt_data = convert(Matrix{BigFloat}, tt_data);

using PeriodicOrbitTTV: compute_diff_squared_jacobian

optparams = OptimParameters(3, optvec_0)

using PeriodicOrbitTTV: compute_diff_squared

function objective_function(vec)
    # p = scale_factor .* θ

    optparams = OptimParameters(nplanet, vec)

    orbit = Orbit(nplanet, optparams, orbparams)
    tt = compute_tt(orbit.ic, orbparams.obstmax) # TODO: Get rid of the hardcode here

    # Append the TT information
    tmod, ip, jp = match_transits(tt_data, orbit, tt.tt, tt.count, nothing)

    diff_squared = compute_diff_squared(optparams, orbparams, nplanet, tmod)

    return diff_squared
end

fd_all_jac = jacobian(central_fdm(2,1), objective_function, optvec_0)[1]
ad_all_jac = compute_diff_squared_jacobian(optparams, orbparams, 3, tt_data)