
function NeumaierSum(input)
  sum = 0.0
  c = 0.0                 # A running compensation for lost low-order bits.
  for i = 1:size(input)[1]
    t = sum + input[i]
    if abs(sum) >= abs(input[i])
      c += (sum - t) + input[i] # If sum is bigger, low-order digits of input[i] are lost.
    else
      c += (input[i] - t) + sum # Else low-order digits of sum are lost.
    end
    sum = t
  end
  return sum,c              # Correction only applied once in the very end.
end


include("../src/ttv.jl")


# Testing compensated summation:

Base.rand(::Type{BigFloat}) =  get(tryparse(BigFloat, "0." .* join(rand(['0','1'], precision(BigFloat))), 2))

# Create random numbers in BigFloat precision:
nrand = 1000000
summand_big = zeros(BigFloat,nrand)
for i=1:nrand
  summand_big[i] = (rand(BigFloat)-big(0.5))*10^(rand(BigFloat)*2)
end

# Convert summands to Float64:
summand = convert(Array{Float64,1},summand_big)

# So, now sum with Kahan compensated summation:
err = 0.0
sum_comp = 0.0
for i=1:nrand
  sum_comp,err = comp_sum(sum_comp,err,summand[i])
end

# Next with Neumaier compensated summation:

sum_neum,corr_neum = NeumaierSum(summand)

# Sum without compensated summation:

sum_flt = sum(summand)

# Sum with BigFloat precision:
sum_big = sum(summand_big)

# Now compare the difference:

println("Float - Big sum: ",sum_flt-sum_big)
println("Comp - Big sum: ",sum_comp-sum_big)

diff = sum_comp-sum_big+err
println("Error: ",err," Comp - Big +err: ",sum_comp-sum_big-err," Big - (Comp + err): ",sum_big-sum_comp+err)
diff_neum = sum_neum-sum_big+corr_neum
println("Error: ",corr_neum," Neum - Big +err: ",sum_neum-sum_big+corr_neum," Big - (Comp + err): ",sum_big-sum_neum-corr_neum)
println("Improve Kahan: ",abs(diff)/abs(sum_flt-sum_big))
println("Improve Neum:  ",abs(diff_neum)/abs(sum_flt-sum_big))
