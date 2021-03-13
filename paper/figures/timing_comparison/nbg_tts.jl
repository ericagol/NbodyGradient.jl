using NbodyGradient, DelimitedFiles

#="""Setup NBG and output initial conditions for rebound and TTVFast"""
function setup_sim(ntransits, denom)
    elements = readdlm("elements_noprior_students.txt",',')
    elements[:,end] .= 0; # Set Ω to 0
    ic = ElementsIC(0.0, 8, elements)

    h = elements[2,2]/denom
    tmax = ntransits*elements[end,2]
    intr = Integrator(h, 0.0, tmax)

    s = State(ic)
    tt = TransitTiming(intr.tmax, ic)
    output_rebound_ICs(s)
    output_TTVFast_ICs(s)
    return s, tt, intr
end
=#
function setup_sim(tmax, denom)
    elements = readdlm("elements_noprior_students.txt",',')
    elements[:,end] .= 0; # Set Ω to 0
    ic = ElementsIC(0.0, 8, elements)

    h = elements[2,2]/denom
    intr = Integrator(h, 0.0, tmax)

    s = State(ic)
    tt = TransitTiming(tmax, ic)
    output_rebound_ICs(s)
    output_TTVFast_ICs(s)
    return s, tt, intr
end

function output_NBG_times(s,tt,intr)
    intr(s,tt,grad=false)
    times = tt.tt[2:end,1:3000]
    open("nbg_tts.txt", "w") do io
        writedlm(io, eachrow(times),',')
    end
    return times
end

function output_rebound_ICs(s::State)
    # Linear transformation to rebound coordinate system.
    A = [0 0 -1; 1 0 0; 0 -1 0]
    pos = copy(s.x)
    vel = copy(s.v)
    reb_pos = zeros(size(pos))
    reb_vel = zeros(size(vel))
    for i in 1:8
        reb_pos[:,i] .= A*pos[:,i]
        reb_vel[:,i] .= A*vel[:,i]
    end
    open("rebound_pos.txt", "w") do io
        writedlm(io, eachrow(reb_pos),',')
    end
    open("rebound_vel.txt", "w") do io
        writedlm(io, eachrow(reb_vel),',')
    end
end

function output_TTVFast_ICs(s::State)
    # Transform to ttvfast astrocentric coordinates
    xstar = copy(s.x[:,1])
    vstar = copy(s.v[:,1])
    ttvf_pos = copy(s.x[:,2:end])
    ttvf_vel = copy(s.v[:,2:end])
    for i in 1:7
        ttvf_pos[:,i] .-= xstar
        ttvf_vel[:,i] .-= vstar
    end
    A = [-1 0 0; 0 1 0; 0 0 -1]
    ttvf_pos = (A * ttvf_pos)'
    ttvf_vel = (A * ttvf_vel)'

    open("ttvfast_ics.txt", "w") do io
        writedlm(io, NbodyGradient.GNEWT) # set units
        writedlm(io, s.m[1])
        writedlm(io, eachrow([s.m[2:end] ttvf_pos ttvf_vel]), ' ')
    end
    run(`mv ttvfast_ics.txt ttvfast/`)
end

