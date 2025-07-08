using PeriodicOrbit
using PeriodicOrbit: compute_system_init, orbital_to_cartesian, calculate_jac_time_evolution, extract_elements
using NbodyGradient

using Test

optvec_0 = BigFloat.([0.1, 0.1, 0.1, 0.1,
1π, 1π, 0.,
0.,0., 0.,
1e-4, 1e-4,
365.242])

orbparams = OrbitParameters(BigFloat.([3e-6, 5e-6, 7e-5, 3e-5]), 
                                BigFloat.([0.5, 0.5]), 
                                BigFloat(2.000), 
                                BigFloat(8*365.242), 
                                BigFloat.([1., 1., 5., 3., 2.]))

nplanet = 4
epsilon = BigFloat(eps(Float64))

# ForwardDiff
orbit_0 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

@testset "Jacobian Precision" begin
    @testset "Jacobian 1" begin
        # For jac 1 testing only
        function optvec_to_mat(x, orbparams)
            optparams = OptimParameters(length(orbparams.mass), x)

            periods, mean_anoms, omegas = compute_system_init(x, orbparams)
            
            pos, vel, pos_star, vel_star = orbital_to_cartesian(orbparams.mass, periods, mean_anoms, omegas, optparams.e)

            positions = hcat(pos_star, pos) 
            velocities = hcat(vel_star, vel)

            mat = vcat(vcat(positions, velocities), vcat(1., orbparams.mass)')

            return mat
        end

        # Step 1: Finite Difference for the entire things
        optvec_derivatives = []
        for j in 1:length(optvec_0)

            optvec_pos = copy(optvec_0)
            optvec_neg = copy(optvec_0)

            optvec_pos[j] += epsilon
            optvec_neg[j] -= epsilon

            # Finite difference
            jac_pos = optvec_to_mat(optvec_pos, orbparams)
            jac_neg = optvec_to_mat(optvec_neg, orbparams)

            optvec_derivative = reduce(vcat, eachcol((jac_pos - jac_neg) / (2 * epsilon)))
            push!(optvec_derivatives, optvec_derivative)
        end

        optvec_derivatives = reduce(hcat, optvec_derivatives)

        for i in 1:1:size(optvec_derivatives, 1), j in 1:1:size(optvec_derivatives, 2)
            @test isapprox(orbit_0.jac_1[i,j], optvec_derivatives[i,j]; rtol=epsilon)
        end
    end

    @testset "Jacobian 2" begin
        function matrix_to_state(state::State, mat)
            state.x .= mat[1:3,:]
            state.v .= mat[4:6,:]
            state.m .= mat[7,:]
            return state
        end

        orbit_2 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

        init_state_0 = orbit_2.s
        jac2_ic = orbit_2.ic

        input_mat_0 = vcat(init_state_0.x, init_state_0.v, init_state_0.m')

        optvec_derivatives = []
        for j in 1:length(input_mat_0)

            optvec_pos = copy(input_mat_0)
            optvec_neg = copy(input_mat_0)

            optvec_pos[j] += epsilon
            optvec_neg[j] -= epsilon

            # Update state
            state_pos = matrix_to_state(deepcopy(init_state_0), optvec_pos)
            state_neg = matrix_to_state(deepcopy(init_state_0), optvec_neg)

            # Finite difference
            jac_pos = calculate_jac_time_evolution(state_pos, orbparams.tsys, optvec_0[end])[2]
            jac_neg = calculate_jac_time_evolution(state_neg, orbparams.tsys, optvec_0[end])[2]

            jac_mat_pos = vcat(jac_pos.x, jac_pos.v, jac_pos.m')
            jac_mat_neg = vcat(jac_neg.x, jac_neg.v, jac_neg.m')

            optvec_derivative = reduce(vcat, eachcol((jac_mat_pos - jac_mat_neg) / (2 * epsilon)))
            push!(optvec_derivatives, optvec_derivative)
        end

        optvec_derivatives = reduce(hcat, optvec_derivatives)

        for i in 1:1:size(optvec_derivatives, 1), j in 1:1:size(optvec_derivatives, 2)
            @test isapprox(orbit_0.jac_2[i,j], optvec_derivatives[i,j]; rtol=epsilon)
        end
    end

    @testset "Jacobian 3" begin
        orbit_3 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)
        final_state_0 = orbit_3.state_final
        jac3_ic = orbit_3.ic

        input_mat_0 = vcat(final_state_0.x, final_state_0.v, jac3_ic.m')

        # Function for Jacobian 3
        function final_state_to_elements(input)
            xx = input[1:3,:]
            vv = input[4:6,:]
            masses = input[7,:]

            return extract_elements(xx, vv, masses, orbparams)
        end

        optvec_derivatives = []
        for j in 1:length(input_mat_0)

            optvec_pos = copy(input_mat_0)
            optvec_neg = copy(input_mat_0)

            optvec_pos[j] += epsilon
            optvec_neg[j] -= epsilon

            # Finite difference
            jac_pos = final_state_to_elements(optvec_pos)
            jac_neg = final_state_to_elements(optvec_neg)

            optvec_derivative = reduce(vcat, eachcol((jac_pos - jac_neg) / (2 * epsilon)))
            push!(optvec_derivatives, optvec_derivative)
        end

        optvec_derivatives = reduce(hcat, optvec_derivatives)

        for i in 1:1:size(optvec_derivatives, 1), j in 1:1:size(optvec_derivatives, 2)
            @test isapprox(orbit_0.jac_3[i,j], optvec_derivatives[i,j]; rtol=epsilon)
        end
    end

    @testset "Combined Jacobians" begin

        # Finite Difference for the entire things
        optvec_derivatives = []
        for j in 1:length(optvec_0)

            optvec_pos = copy(optvec_0)
            optvec_neg = copy(optvec_0)

            optvec_pos[j] += epsilon
            optvec_neg[j] -= epsilon

            # Finite difference
            jac_pos = Orbit(nplanet, OptimParameters(nplanet, optvec_pos), orbparams).final_elem
            jac_neg = Orbit(nplanet, OptimParameters(nplanet, optvec_neg), orbparams).final_elem

            optvec_derivative = reduce(vcat, eachcol((jac_pos - jac_neg) / (2 * epsilon)))
            push!(optvec_derivatives, optvec_derivative)
        end

        optvec_derivatives = reduce(hcat, optvec_derivatives)

        # println(size(orbit_0.jac_combined))
        # println(orbit_0.jac_combined)

        # println()

        # println(size(optvec_derivatives))
        # println(optvec_derivatives)

        for i in 1:1:size(optvec_derivatives, 1), j in 1:1:size(optvec_derivatives, 2)
            @test isapprox(orbit_0.jac_combined[i,j], optvec_derivatives[i,j]; rtol=epsilon)
        end
    end
end