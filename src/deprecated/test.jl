using NbodyGradient
include("PeriodicOrbit.jl")
using .PeriodicOrbit

# TODO: Move tests to `test/` and integrate it with Julia unit test.
# Create a simple two-planet system
star = Elements(m=1.0)
planet1 = Elements(m=0.001, P=1.0, e=0.1, ω=0.0, I=π/2, t0=0.0)
planet2 = Elements(m=0.0005, P=2.0, e=0.05, ω=π/4, I=π/2, t0=0.0)


ic = ElementsIC(0.0, 3, star, planet1, planet2)
s = State(ic)


initial_anomalies = get_anomalies(s, ic)
println("Initial mean anomaly of planet 1: ", initial_anomalies[1][2])


target_M = 2π+.5
println("Integrating to target mean anomaly: ", target_M)


target_time, times, mean_anomalies = integrate_to_M!(s, ic, target_M, 1, 0.05)


final_anomalies = get_anomalies(s, ic)
println("Final mean anomaly of planet 1: ", final_anomalies[1][2])
println("Difference from target: ", abs(target_M - final_anomalies[1][2]))
println("Final time: ", s.t[1])
