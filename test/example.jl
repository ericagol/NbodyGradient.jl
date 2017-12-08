function addone(x::Real)
return x+1.0
end

y = randn()
println(addone(y))
println(addone(big(y)))

function sum_array(x::Array{Real,2})
#function sum_array(x::Array{T,2}) where {T <: Real} # This seems to work.
#function sum_array(x::Array)  # <- If instead I define input as an array, then this works.
return sum(x)
end

#function sum_array(x::Array{Float64,2})  
# Also works if I define different routines using multiple dispatch, but this seems clumsy
#return sum(x)
#end

#function sum_array(x::Array{BigFloat,2})
#return sum(x)
#end

x = real(randn(5,5))
println(sum_array(x))

xbig = big.(x)
println(sum_array(xbig))
