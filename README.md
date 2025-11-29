# PeriodicOrbitTTV.jl
Julia routine to fit a planetary system built on [NbodyGradient.jl](https://github.com/ericagol/NbodyGradient.jl). This routine searches for a stable periodic orbit with heliocentric orbital elements, with support for transit timing constraints. The optimization is implemented using the Levenberg-Marquardt method, and fast, accurate Jacobians are computed using Automatic Differentiation.

Under development by Tanawan Chatchadanoraset, Hurum Tohfa, and Eric Agol

`PeriodicOrbitTTV.jl` is the Julia implementation of Chatchadanoraset, Tohfa & Agol (2025)

## Installation

Clone this repository to your local machine
```
Pkg.add(url="https://github.com/tnw-chatc/PeriodicOrbitTTV.jl.git")
```

## Getting Started

Most of the time, the information of the system is encoded in an `Orbit` structure, which we will use to perform various operations.

To construct an `Orbit` structure, you will need
1. `OptimParatemers` : the set of optimization parameters.
2. `OrbitParameters` : the set of auxiliary parameters that are kept constant during the entire optimization.

### `OptimParameters`

To construct an OptimParameters: you will need a vector `vec::Vector{T}` that contains optimization parameters.

`vec::Vector{T}` has a specific order: `N` eccentricities, `N` mean anomalies, `N - 1` omega differences, and `N - 2` period ratios as defined in Gozdziewski and Migaszewski (2020), `1` innermost planet period, `N` masses, `1` Kappa, `1` innermost longitude of periastron, and `1` PO system period. `5N+1` elements in total.

One example of a four-planet system:
```julia
optvec_0 = ([0.1, 0.07, 0.05, 0.07, # Eccentricities
    0., 0., 0., 0.,                 # Mean anomalies
    0., 0., 0.,                     # Omega differences
    1e-4, 1e-4,                     # Period ratio deviations
    365.242,                        # Innermost planet period
    3e-6, 5e-6, 7e-5, 3e-5,         # Masses
    2.000,                          # Kappa
    0.00                            # Innermost longitude of periastron
    8*365.242                       # Periodic orbit system period
])
```

Note that all orbital parameters are heliocentric.

Another way to construct `OptimParameters` is to use its default constructor `OptimParameters(e, M, Δω, Pratio, inner_period, masses, kappa, ω1, tsys)`, though the first method is easier in practice as it automatically parses the argument for you.

### `OrbitParameters`

`OrbitParameters` consists of parameters that do not change during the optimization. It consists of 
1. The number of the _planets_,
2. The `C` factors defined in CTA25. Only applicable for a system with 3 or more planets.
3. The maximum observation time for transit timing.

### `Orbit`

Create an `Orbit` structure, call `Orbit(n::Int, optparams::OptimParameters{T}, orbparams::OrbitParameters{U}) where {T <: Real, U <: Real}`

## Periodic Orbit Optimization

Call
```julia
find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; 
    use_jac::Bool=true, trace::Bool=false,eccmin::T=1e-3,maxit::Int64=1000, optim_weights=nothing,
    prior_weights=nothing, lower_bounds=nothing, upper_bounds=nothing, scale_factor=nothing) where T <: Real
```
to perform the optimization. See the documentation for more information.

## Optimize with Transit Timing constraint

`PeriodicOrbit.jl` allows transit timing constraint, thanks to its compatibility with `NbodyGradient.jl`. Simply load the TT information and call

```julia
find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, tt_data::Matrix{T}; 
    use_jac::Bool=true, trace::Bool=false,eccmin::T=1e-3,maxit::Int64=1000, optim_weights=nothing,
    prior_weights=nothing, tt_weights=nothing, lower_bounds=nothing, upper_bounds=nothing, scale_factor=nothing) where T <: Real
```
to perform the optimization, taking the transit timing constraint into account. The optimization (periodic orbit) weight is defined as $w_i = 1/\sigma_i^2$, where $\sigma_i^2$ is the variance of optimization parameter $i$.

### Transit Timing Data
`tt_data` must be a matrix with `N` rows and 4 columns consisting of the transiting body, the transit number, the transit time, and the timing uncertainty, in this order. An example matrix can be found in the repository.

## Citation

...
