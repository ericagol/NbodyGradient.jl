
abstract type AbstractOutput end
abstract type Output{T} <: AbstractOutput end
"""

Preallocates and holds arrays for positions, velocities, and Jacobian at every integrator step
""" 
# Should add option to choose out intervals, checkpointing, etc.
mutable struct CartesianOutput{T<:AbstractFloat,M<:Array{T,3}} <: Output{T}
    x::M
    v::M
    jac::M
    nstep::Int64
    filename::String

    function CartesianOutput(::Type{T},nbody::Int64,nstep::Int64;filename="data.jld2") where T <: AbstractFloat
        return new{T,Array{T,3}}(zeros(T,nstep,NDIM,nbody),
                                 zeros(T,nstep,NDIM,nbody),
                                 zeros(T,nstep,7*nbody,7*nbody),
                                 nstep,filename
                                )
    end
end

# Allow user to not have to specify type. Defaults to Float64
CartesianOutput(nbody::Int64,nstep::Int64;filename="data.jld2") = CartesianOutput(Float64,nbody,nstep,filename=filename)

"""

Runs integrator like (*insert doc reference here*) and output positions, velocities, and Jacobian to a JLD2 file.
"""
function (i::Integrator)(s::State{T},o::CartesianOutput{T}) where T<:AbstractFloat
    s2 = zero(T) # For compensated summation
    
    # Preallocate struct of arrays for derivatives (and pair)
    d = Derivatives(T,s.n) 
    pair = zeros(Bool,s.n,s.n)
    
    iout::Integer = 1 
    while s.t < (i.t0 + i.tmax) && iout <= o.nstep
        # Take integration step and advance time
        i.scheme(s,d,i.h,pair)
        s.t,s2 = comp_sum(s.t,s2,i.h)
        
        # Save coords to output structure
        o.x[iout,:,:] .= s.x
        o.v[iout,:,:] .= s.v
        o.jac[iout,:,:] .= s.jac_step
        iout+=1
    end
    save(o.filename,Dict("pos"=>o.x,"vel"=>o.v,"jac"=>o.jac)) 
    return
end

