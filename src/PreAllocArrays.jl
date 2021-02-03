abstract type PreAllocArrays end

struct Derivatives{T<:AbstractFloat} 

    pxpr0::Vector{T}
    pxpa0::Vector{T} 
    pxpk::Vector{T}
    pxps::Vector{T} 
    pxpbeta::Vector{T}
  
    dxdr0::Vector{T}
    dxda0::Vector{T}  
    dxdk::Vector{T}
    dxdv0::Vector{T}

    prvpr0::Vector{T}
    prvpa0::Vector{T}
    prvpk::Vector{T}
    prvps::Vector{T}
    prvpbeta::Vector{T}

    drvdr0::Vector{T}
    drvda0::Vector{T}
    drvdk::Vector{T}
    drvdv0::Vector{T}

    vtmp::Vector{T}
    dvdr0::Vector{T}
    dvda0::Vector{T}
    dvdv0::Vector{T}
    dvdk::Vector{T}

    function Derivatives(::Type{T}) where T<:AbstractFloat
#        return new{T}([zeros(T,3) for _ in 1:23]...)
        return Derivatives{T}([zeros(T,3) for _ in 1:23]...)
    end
end

