# """Initializes a planet using mean anomaly"""
# function init_from_M!(s::State, ic::InitialConditions, M::T, index::Int) where T <: AbstractFloat
#     # TODO: Also implement a version taking an array of M instead of an individual index
#     elems = get_orbital_elements(s, ic)[index]
#     e = elems.e
#     a = elems.a
#     Ω = elems.Ω
#     I = elems.I
#     ω = elems.ω
#     n = 2π / elems.P

#     # Get true and eccentric anomalies.
#     f = kepler(M, e)
#     E = ekepler(M, e)

#     # Rotates x and v on the plane to the true frame.
#     rmag = a * (1 - e^2) / (1 + e * cos(f))
#     rplane = [rmag * cos(f), rmag * sin(f), 0]
#     vplane = [-n * a^2 * sin(E) / rmag, n * a^2 * sqrt(1-e^2) * cos(E) / rmag, 0]

#     # Rotation matrix in M&D
#     P1 = [cos(ω) -sin(ω) 0.0; sin(ω) cos(ω) 0.0; 0.0 0.0 1.0]
#     P2 = [1.0 0.0 0.0; 0.0 cos(I) -sin(I); 0.0 sin(I) cos(I)]
#     P3 = [cos(Ω) -sin(Ω) 0.0; sin(Ω) cos(Ω) 0.0; 0.0 0.0 1.0]
#     P321 = P3*P2*P1

#     r_new = P321 * rplane
#     v_new = P321 * vplane

#     s.x[:,index] .= r_new
#     s.v[:,index] .= v_new
# end