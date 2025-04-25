using NbodyGradient: State, Elements, ElementsIC, InitialConditions

function integrate_to_M!(s::State, ic::InitialConditions, target::T, index::Int64, h=10.) where T <: AbstractFloat
    times = Float64[s.t[1]]
    Ms = Float64[get_anomalies(s, ic)[index][2]]
    intr = Integrator(ahl21!, h, h)
    current_passage = 0
    while true
        intr(s)
        M = current_passage * 2π + get_anomalies(s, ic)[index][2]
        if Ms[end] > M
            current_passage += 1
            M = current_passage * 2π + get_anomalies(s, ic)[index][2]
            push!(Ms, M)
        else
            push!(Ms, M)
        end

        push!(times, s.t[1])   
        
        if M - Ms[begin] >= target
            target_time = bisection(ic, times[end-1], times[end], index, Ms[begin])
            return target_time, times, Ms
        end
    end    
end

function bisection(ic::ElementsIC, a::Float64, b::Float64, index::Int64, offset::Float64; h=1.0, tol=1e-9, doomthres=100000,)
    doom = 0
    while doom < doomthres
        half = (a + b)/2
        
        sa = State(ic)
        intr = Integrator(ahl21!, h, a)
        intr(sa)
        fa = sin(get_anomalies(sa, ic)[index][2] - offset)

        sb = State(ic)
        intr = Integrator(ahl21!, h, b)
        intr(sb)
        fb = sin(get_anomalies(sb, ic)[index][2] - offset)

        shalf = State(ic)
        intr = Integrator(ahl21!, h, half)
        intr(shalf)
        fhalf = sin(get_anomalies(shalf, ic)[index][2] - offset)

        if fhalf > 0
            a = a
            b = half
        end

        if fhalf < 0
            a = half
            b = b
        end

        # println("$a, $half, $b, $fa, $fhalf, $fb")

        if abs(fa-fb) < tol || fhalf == 0.0
            return half
        else
            doom += 1
        end     
    end
    return half
end