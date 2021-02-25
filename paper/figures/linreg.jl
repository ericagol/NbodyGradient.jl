using LinearAlgebra,Statistics

struct Regression
    α
    β
end

function Regression(x::AbstractVector,y::AbstractVector)
    x̄ = mean(x)
    ȳ = mean(y)
    sx = x .- x̄
    sy = y .- ȳ
    
    β̂ = dot(sx,sy) / dot(sx,sx)
    α̂ = ȳ - β̂*x̄

    return Regression(α̂,β̂)
end

model(p::Regression,ϵ,x) = p.α .+ (p.β .* x) .+ ϵ 
