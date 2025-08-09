include("PeriodicOrbit.jl")
using .PeriodicOrbit
using NbodyGradient
using PyPlot
using Statistics

# Problem setup and optimization
m_e = 3.00274e-6
P1 = 365.242

optvec_0 = [0.00084255878, 0.001776, 0., 0., pi, P1, 
          0.0009508/0.9987640, 0.0002852/0.9987640, 1.8354065073^1.5, 0., (5/3)*P1]

lower_bounds = vcat([0., 0.], [-2pi, -2pi], [-2pi], [0.5*P1], [1e-5, 1e-5], [1.], [-2pi], [1*P1])
upper_bounds = vcat([0.01, 0.01], [2pi, 2pi], [2pi], [2.0*P1], [1e-2, 1e-2], [3.], [2pi], [5*P1])
prior_weights = vcat([0., 1e4], [0., 0.], [0.], [0.], [1e8, 1e8], [1e8], [0.], [1e8])

optparams_0 = OptimParameters(2, optvec_0)
orbparams = OrbitParameters(2, [0.5])

# Run optimization to find best-fit periodic orbit
optres = find_periodic_orbit(optparams_0, orbparams; maxit=9000, 
   lower_bounds=lower_bounds, upper_bounds=upper_bounds, prior_weights=prior_weights)

println("Optimization residual: $(sum(optres.resid))")
println("Best-fit parameters: $(optres.param)")

# Create optimized system from best-fit parameters
optparams_bestfit = OptimParameters(orbparams.nplanet, optres.param)
orbit_bestfit = Orbit(orbparams.nplanet, optparams_bestfit, orbparams)

# Extract orbital elements from best-fit solution
elems = get_orbital_elements(orbit_bestfit.s, orbit_bestfit.ic)

println("\nBest-fit orbital elements:")
for i in 2:length(elems)
   println("Planet $(i-1): P = $(round(elems[i].P, digits=2)) days, a = $(round(elems[i].a, digits=4)) AU, e = $(round(elems[i].e, digits=6))")
end

# Set up edge-on system for transit detection using best-fit parameters
star_edgeon = Elements(m=1.0)
planet1_edgeon = Elements(m=elems[2].m, P=elems[2].P, e=0.01, ω=0.0, I=π/2, Ω=0.0, t0=0.0)
planet2_edgeon = Elements(m=elems[3].m, P=elems[3].P, e=0.01, ω=π, I=π/2, Ω=0.0, t0=0.0)

el_ic_edgeon = ElementsIC(0.0, 3, star_edgeon, planet1_edgeon, planet2_edgeon)

# Integration parameters
P_inner = elems[2].P
h_int = P_inner / 100.0
tmax = P_inner * 30  # 30 inner periods for good TTV coverage

println("\nIntegration setup:")
println("Planet 1: P = $(round(elems[2].P, digits=2)) days, a = $(round(elems[2].a, digits=3)) AU")
println("Planet 2: P = $(round(elems[3].P, digits=2)) days, a = $(round(elems[3].a, digits=3)) AU")
println("Integration time: $(round(tmax, digits=1)) days ($(round(tmax/P_inner, digits=1)) inner periods)")

# Analyze detection threshold for Planet 1
function sample_z_positions(el_ic, planet_idx, P_inner)
   state_check = State(el_ic)
   z_positions = Float64[]
   times = Float64[]
   
   dt_sample = P_inner / 200.0
   max_time_sample = 2 * P_inner
   t = 0.0
   
   while t < max_time_sample
       planet_pos = state_check.x[:, planet_idx + 1]  # +1 because star is index 1
       push!(z_positions, planet_pos[3])
       push!(times, t)
       
       intr = Integrator(ahl21!, dt_sample, dt_sample)
       intr(state_check)
       t = state_check.t[1]
   end
   
   return z_positions, times
end

z_positions, times = sample_z_positions(el_ic_edgeon, 1, P_inner)

orbital_radius_p1 = elems[2].a
detection_threshold = -0.25 * orbital_radius_p1
min_z = minimum(z_positions)
max_z = maximum(z_positions)

println("\nPlanet 1 detection analysis:")
println("Orbital radius: $(round(orbital_radius_p1, digits=4)) AU")
println("Detection threshold: z < $(round(detection_threshold, digits=4)) AU")
println("Z-range: $(round(min_z, digits=4)) to $(round(max_z, digits=4)) AU")
println("Minimum z: $(round(min_z, digits=4)) AU")
println("Detection possible: $(min_z < detection_threshold ? "YES" : "NO") (needs z < $(round(detection_threshold, digits=4)))")

below_threshold = sum(z_positions .< detection_threshold)
total_samples = length(z_positions)
detection_fraction = below_threshold / total_samples * 100
println("Time below threshold: $(round(detection_fraction, digits=1))% of orbit")

# Run NbodyGradient transit detection
println("\nRunning transit detection...")
state_nbody = State(el_ic_edgeon)
tt_nbody = TransitTiming(tmax, el_ic_edgeon)
Integrator(ahl21!, h_int, tmax)(state_nbody, tt_nbody)

println("Transit detection results:")
for i in 1:2
   println("  Planet $i: $(tt_nbody.count[i]) transits")
end

# TTV analysis and data export
function compute_and_save_ttvs(transit_times, planet_num, planet_name, filename_prefix)
   if length(transit_times) < 3
       println("$planet_name: Insufficient transits for TTV analysis ($(length(transit_times)) detected)")
       return nothing
   end
   
   # Linear fit to remove constant period
   n_idx = 0:(length(transit_times)-1)
   P_fit = (transit_times[end] - transit_times[1]) / (length(transit_times) - 1)
   t0_fit = transit_times[1]
   
   # Calculate TTVs in minutes
   linear_prediction = t0_fit .+ P_fit .* n_idx
   ttv_resid = (transit_times .- linear_prediction) .* 24 * 60
   
   # Statistics
   rms_ttv = sqrt(mean(ttv_resid.^2))
   peak_to_peak = maximum(ttv_resid) - minimum(ttv_resid)
   
   println("\n$planet_name TTV results:")
   println("Number of transits: $(length(transit_times))")
   println("Fitted period: $(round(P_fit, digits=3)) days")
   println("TTV RMS: $(round(rms_ttv, digits=2)) minutes")
   println("TTV peak-to-peak: $(round(peak_to_peak, digits=2)) minutes")
   
   # Save TTV data to file
   ttv_filename = "$(filename_prefix)_planet$(planet_num)_ttvs.txt"
   open(ttv_filename, "w") do file
       println(file, "# Planet $planet_num TTV data")
       println(file, "# Generated from best-fit periodic orbit")
       println(file, "# Fitted period: $(P_fit) days")
       println(file, "# TTV RMS: $(rms_ttv) minutes")
       println(file, "# Columns: transit_number, transit_time_days, linear_prediction_days, ttv_minutes")
       for i in 1:length(transit_times)
           println(file, "$(n_idx[i]) $(transit_times[i]) $(linear_prediction[i]) $(ttv_resid[i])")
       end
   end
   println("Saved TTV data: $ttv_filename")
   
   # Save raw transit times
   times_filename = "$(filename_prefix)_planet$(planet_num)_transit_times.txt"
   open(times_filename, "w") do file
       println(file, "# Planet $planet_num raw transit times")
       println(file, "# Generated from best-fit periodic orbit")
       println(file, "# Column: transit_time_days")
       for t in transit_times
           println(file, "$t")
       end
   end
   println("Saved transit times: $times_filename")
   
   # Create TTV plot
   figure(figsize=(12, 8))
   
   subplot(2, 1, 1)
   plot(n_idx, transit_times, "o-", markersize=6, linewidth=2, label="Observed")
   plot(n_idx, linear_prediction, "--", linewidth=2, alpha=0.7, label="Linear fit")
   xlabel("Transit number")
   ylabel("Transit time (days)")
   title("$planet_name Transit Times")
   legend()
   grid(true, alpha=0.3)
   
   subplot(2, 1, 2)
   plot(n_idx, ttv_resid, "o-", color="red", markersize=6, linewidth=2)
   xlabel("Transit number")
   ylabel("TTV (minutes)")
   title("$planet_name Transit Timing Variations")
   grid(true, alpha=0.3)
   axhline(y=0, color="black", linestyle="--", alpha=0.5)
   
   # Add statistics text
   textstr = "RMS: $(round(rms_ttv, digits=1)) min\nP-to-P: $(round(peak_to_peak, digits=1)) min"
   gca().text(0.02, 0.98, textstr, transform=gca().transAxes, fontsize=10,
              verticalalignment="top", bbox=Dict("boxstyle"=>"round", "facecolor"=>"wheat", "alpha"=>0.8))
   
   tight_layout()
   plot_filename = "$(filename_prefix)_planet$(planet_num)_ttvs.png"
   savefig(plot_filename, dpi=300, bbox_inches="tight")
   println("Saved TTV plot: $plot_filename")
   
   return ttv_resid, rms_ttv
end

# Process TTVs for all detected planets
ttv_results = []
filename_prefix = "bestfit_orbit"

for i in 1:2
   if tt_nbody.count[i] >= 3
       transit_times = tt_nbody.tt[i, 1:tt_nbody.count[i]]
       planet_name = "Planet $i"
       ttv_data, rms = compute_and_save_ttvs(transit_times, i, planet_name, filename_prefix)
       push!(ttv_results, (i, rms, length(transit_times)))
   else
       println("Planet $i: Only $(tt_nbody.count[i]) transits - insufficient for TTV analysis")
   end
end

# Save optimization results
optim_filename = "$(filename_prefix)_optimization_results.txt"
open(optim_filename, "w") do file
   println(file, "# Best-fit periodic orbit optimization results")
   println(file, "# Optimization residual: $(sum(optres.resid))")
   println(file, "# Parameters: eccentricities, mean_anomalies, omega_differences, period_ratios, inner_period, masses, kappa, omega1, tsys")
   println(file, "# Best-fit parameter vector:")
   for (i, param) in enumerate(optres.param)
       println(file, "$i $param")
   end
   println(file, "")
   println(file, "# Orbital elements from best-fit:")
   for i in 2:length(elems)
       println(file, "# Planet $(i-1): P=$(elems[i].P) days, a=$(elems[i].a) AU, e=$(elems[i].e), m=$(elems[i].m)")
   end
end
println("Saved optimization results: $optim_filename")

# Summary
println("\nFinal summary:")
println("Best-fit periodic orbit optimization complete")
if length(ttv_results) > 0
   println("Successfully detected TTVs in $(length(ttv_results)) planet(s):")
   for (planet_idx, rms, n_transits) in ttv_results
       println("  Planet $planet_idx: $(round(rms, digits=1)) minutes RMS ($(n_transits) transits)")
   end
   println("TTVs indicate gravitational interactions in best-fit system")
else
   println("No planets with sufficient transits for TTV analysis")
end

if min_z >= detection_threshold
   println("Planet 1 not detected due to insufficient transit depth")
   println("  (z_min = $(round(min_z, digits=4)) AU >= threshold = $(round(detection_threshold, digits=4)) AU)")
else
   println("Planet 1 should be detectable but detection failed")
   println("  (z_min = $(round(min_z, digits=4)) AU < threshold = $(round(detection_threshold, digits=4)) AU)")
end

println("Analysis complete - TTV data saved for next project phase")