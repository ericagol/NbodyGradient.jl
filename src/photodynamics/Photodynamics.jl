# Photodyanmics
struct Photodynamics{T<:AbstractFloat}
    nt::Int64
    times::Vector{T}
    bsky2::Matrix{T}
    vsky::Matrix{T}
    dbvdq0::Array{T,5}
    dbvdelements::Array{T,5}
end

function Photodynamics(times::Vector{T},ic::ElementsIC{T}) where T<:AbstractFloat
    n = ic.nbody
    nt = length(times)
    return Photodynamics(nt,times,zeros(T,n,nt),zeros(T,n,nt),zeros(T,2,n,nt,7,n),zeros(T,2,n,nt,7,n))
end

# Integrate to, and output b and v_sky for each body, for a list of times.
# Should be sorted times.
# NOTE: Need to fix so that if initial time is a 0, s.dqdt isn't 0s.
function (intr::Integrator)(s::State{T},pd::Photodynamics{T};grad::Bool=true) where T<:AbstractFloat
    if grad; dbvdq = zeros(T,2,7,s.n); end

    # Integrate to each time, using intr.h, and output b and vsky (only want primary transits for now)
    for (i,ti) in enumerate(pd.times)
        intr(s,ti;grad=grad)
        for j in 2:s.n
            if grad
                pd.vsky[j,i],pd.bsky2[j-1,i] = calc_dbvdq!(s,dbvdq,1,j)
                for ibv=1:2, k=1:7, p=1:s.n
                    pd.dbvdq0[ibv,j,i,k,p] = dbvdq[ibv,k,p]
                end
            else
                pd.vsky[j,i],pd.bsky2[j,i] = calc_bv(s,1,j)
            end
        end
    end
    calc_dbvdelements!(s,pd)
    return
end

"""

Calculate impact parameter and sky velocity
"""
function calc_bv(s::State{T},i::Int64,j::Int64) where T<:AbstractFloat
    vsky = sqrt((s.v[1,j]-s.v[1,i])^2 + (s.v[2,j]-s.v[2,i])^2)
    bsky2 = (s.x[1,j]-s.x[1,i])^2 + (s.x[2,j]-s.x[2,i])^2
    return vsky,bsky2
end

"""

Calculate derivative of impact parameter and sky velocity wrt cartesian coordinates
"""
function calc_dbvdq!(s::State{T},dbvdq::Array{T},i::Int64,j::Int64) where T<:AbstractFloat
    vsky,bsky2 = calc_bv(s,i,j)
    gsky = g!(i,j,s.x,s.v)
    gdot = gd!(i,j,s.x,s.v,s.dqdt)
    fill!(dbvdq,zero(T))

    # partial derivative b and v_{sky} with respect to time:
    dvdt = ((s.v[1,j]-s.v[1,i])*(s.dqdt[(j-1)*7+4]-s.dqdt[(i-1)*7+4])+(s.v[2,j]-s.v[2,i])*(s.dqdt[(j-1)*7+5]-s.dqdt[(i-1)*7+5]))/vsky
    dbdt = 2.0 * gsky

    # Compute derivatives:
    indj = (j-1)*7+1
    indi = (i-1)*7+1
    for p=1:s.n
        indp = (p-1)*7
        for k=1:7
            dtdq = -((s.jac_step[indj  ,indp+k]-s.jac_step[indi  ,indp+k])*(s.v[1,j]-s.v[1,i])+(s.jac_step[indj+1,indp+k]-s.jac_step[indi+1,indp+k])*(s.v[2,j]-s.v[2,i])+
                          (s.jac_step[indj+3,indp+k]-s.jac_step[indi+3,indp+k])*(s.x[1,j]-s.x[1,i])+(s.jac_step[indj+4,indp+k]-s.jac_step[indi+4,indp+k])*(s.x[2,j]-s.x[2,i]))/gdot
            #v_{sky} = sqrt((v[1,j]-v[1,i])^2+(v[2,j]-v[2,i])^2)
            dbvdq[1,k,p] = ((s.jac_step[indj+3,indp+k]-s.jac_step[indi+3,indp+k])*(s.v[1,j]-s.v[1,i])+(s.jac_step[indj+4,indp+k]-s.jac_step[indi+4,indp+k])*(s.v[2,j]-s.v[2,i]))/vsky + dvdt*dtdq
            #b_{sky}^2 = (x[1,j]-x[1,i])^2+(x[2,j]-x[2,i])^2
            dbvdq[2,k,p] = 2*((s.jac_step[indj  ,indp+k]-s.jac_step[indi  ,indp+k])*(s.x[1,j]-s.x[1,i])+(s.jac_step[indj+1,indp+k]-s.jac_step[indi+1,indp+k])*(s.x[2,j]-s.x[2,i])) + dbdt*dtdq
        end
    end
    return vsky,bsky2
end

function calc_dbvdelements!(s::State{T},pd::Photodynamics{T}) where T <: AbstractFloat
    for ibv = 1:2, i=1:s.n, j = 1:pd.nt
        if j <= pd.nt
            # Now, multiply by the initial Jacobian to convert time derivatives to orbital elements:
            for k=1:s.n, l=1:7
                pd.dbvdelements[ibv,i,j,l,k] = zero(T)
                for p=1:s.n, q=1:7
                    pd.dbvdelements[ibv,i,j,l,k] += pd.dbvdq0[ibv,i,j,q,p]*s.jac_init[(p-1)*7+q,(k-1)*7+l]
                end
            end
        end
    end
end