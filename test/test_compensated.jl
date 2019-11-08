
include("../src/ttv.jl")

# Testing compensated summation:

Base.rand(::Type{BigFloat}) =  get(tryparse(BigFloat, "0." .* join(rand(['0','1'], precision(BigFloat))), 2))

# Create random numbers in BigFloat precision:
nrand = 1000000
summand_big = zeros(BigFloat,nrand)
for i=1:nrand
  summand_big[i] = rand(BigFloat)
end

# Convert summands to Float64:
summand = convert(Array{Float64,1},summand_big)

# So, now sum with compensated summation:
err = 0.0
sum_comp = 0.0
for i=1:nrand
  sum_comp,err = comp_sum(sum_comp,err,summand[i])
end

# Sum without compensated summation:

sum_flt = sum(summand)

# Sum with BigFloat precision:
sum_big = sum(summand_big)

# Now compare the difference:

println("Float - Big sum: ",sum_flt-sum_big)
println("Comp - Big sum: ",sum_comp-sum_big)

diff = sum_comp-sum_big+err
println("Error: ",err," Comp - Big +err: ",sum_comp-sum_big+err," Big - (Comp + err): ",sum_big-sum_comp-err)
println("Improvement: ",abs(diff)/abs(sum_flt-sum_big))
