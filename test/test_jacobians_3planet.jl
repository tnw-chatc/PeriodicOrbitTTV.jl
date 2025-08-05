using PeriodicOrbit
using PeriodicOrbit: compute_system_init, orbital_to_cartesian, calculate_jac_time_evolution, extract_elements
using NbodyGradient

using FiniteDifferences
using Test

optvec_0 = BigFloat.([0.1, 0.07, 0.05,
0., 0., 0.,
0., 0.,
1e-4,
365.242,
3e-6, 5e-6, 7e-5,
2.000,
0.00,
4*365.242
])

orbparams = OrbitParameters(3,
                                BigFloat.([0.5, 0.5]), 
                                BigFloat(4*365.242))

nplanet = 3
epsilon = BigFloat(eps(Float64))

# ForwardDiff
orbit_0 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

@testset "Jacobian Precision" begin
    @testset "Jacobian 1" begin
        # For jac 1 testing only
        function optvec_to_mat(x, orbparams)
            optparams = OptimParameters(orbparams.nplanet, x)

            periods, mean_anoms, omegas = compute_system_init(x, orbparams)
            
            pos, vel, pos_star, vel_star = orbital_to_cartesian(optparams.masses, periods, mean_anoms, omegas, optparams.e)

            positions = hcat(pos_star, pos) 
            velocities = hcat(vel_star, vel)

            mat = vcat(vcat(positions, velocities), vcat(1., optparams.masses)')

            return mat
        end

        func_1 = x -> optvec_to_mat(x, orbparams) 

        # Step 1: Finite Difference for the entire things
        fd = FiniteDifferences.central_fdm(2, 1)
        optvec_derivatives = jacobian(fd, func_1, optvec_0)[1]

        # Append time dependence gradient to jacobian
        optvec_derivatives = vcat(optvec_derivatives, fill(0., size(optvec_derivatives, 2))')

        # Ensure that derivative w.r.t to itself is 1
        # TODO: Use dynamic type conversion here
        optvec_derivatives[end, end] = 1.

        for i in 1:1:size(optvec_derivatives, 1), j in 1:1:size(optvec_derivatives, 2)
            @test isapprox(orbit_0.jac_1[i,j], optvec_derivatives[i,j]; atol=epsilon)
        end

        global fd_jac_1 = optvec_derivatives
        global an_jac_1 = orbit_0.jac_1
    end

    @testset "Jacobian 2" begin
        function matrix_to_state(state::State, mat)
            state.x .= mat[1:3,:]
            state.v .= mat[4:6,:]
            state.m .= mat[7,:]
            return state
        end

        function state_to_matrix(state::State)
            # return vcat(reshape(vcat(state.x, state.v, state.m'), :), state.t[1])
            return vcat(state.x, state.v, state.m')
        end

        function func_2(x)
            inter_state = calculate_jac_time_evolution(matrix_to_state(deepcopy(orbit_2.s), x), orbparams.tsys, optvec_0[4*nplanet-2])[2]

            return state_to_matrix(inter_state)
        end

        orbit_2 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

        init_state_0 = orbit_2.s
        input_mat_0 = vcat(init_state_0.x, init_state_0.v, init_state_0.m')
        
        # Finite Difference
        fd = FiniteDifferences.central_fdm(2, 1)
        optvec_derivatives = jacobian(fd, func_2, input_mat_0)[1]

        final_state_0 = calculate_jac_time_evolution(matrix_to_state(deepcopy(orbit_2.s), input_mat_0), orbparams.tsys, optvec_0[4*nplanet-2])[2]

        optvec_derivatives = hcat(copy(optvec_derivatives), final_state_0.dqdt)

        for i in 1:1:size(optvec_derivatives, 1), j in 1:1:size(optvec_derivatives, 2)
            @test isapprox(orbit_0.jac_2[i,j], optvec_derivatives[i,j]; rtol=epsilon)
        end

        global fd_jac_2 = optvec_derivatives
        global an_jac_2 = orbit_0.jac_2
    end

    @testset "Jacobian 3" begin
        orbit_3 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)
        final_state_0 = orbit_3.state_final
        jac3_ic = orbit_3.ic

        input_mat_0 = vcat(final_state_0.x, final_state_0.v, final_state_0.m')

        function final_state_to_elements(input)
            xx = input[1:3,:]
            vv = input[4:6,:]
            masses = input[7,:]

            return extract_elements(xx, vv, masses, orbparams)
        end

        # Step 1: Finite Difference for the entire things
        fd = FiniteDifferences.central_fdm(2, 1)
        optvec_derivatives = jacobian(fd, final_state_to_elements, input_mat_0)[1]

        for i in 1:1:size(optvec_derivatives, 1), j in 1:1:size(optvec_derivatives, 2)
            @test isapprox(orbit_0.jac_3[i,j], optvec_derivatives[i,j]; atol=epsilon)
        end

        global fd_jac_3 = optvec_derivatives
        global an_jac_3 = orbit_0.jac_3

    end

    @testset "Combined Jacobians" begin

        func_4 = x -> Orbit(nplanet, OptimParameters(nplanet, x), orbparams).final_elem

        # Step 1: Finite Difference for the entire things
        fd = FiniteDifferences.central_fdm(2, 1)
        optvec_derivatives = jacobian(fd, func_4, optvec_0)[1]
        
        for i in 1:1:size(optvec_derivatives, 1), j in 1:1:size(optvec_derivatives, 2)
            @test isapprox(orbit_0.jac_combined[i,j], optvec_derivatives[i,j]; atol=epsilon)
        end

        global fd_jac_c = optvec_derivatives
        global an_jac_c = orbit_0.jac_combined
    end
end