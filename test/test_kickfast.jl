# Tests the routine kickfast jacobian.  This routine
# computes the impulse gradient after Dehnen & Hernandez (2017).

import NbodyGradient: kickfast!, Derivatives

# Next, try computing three-body Keplerian Jacobian:

@testset "kickfast" begin
n = 3
H = [3,1,1]
t0 = 7257.93115525
h  = 0.05
tmax = 600.0
dlnq = big(1e-15)

elements = readdlm("elements.txt", ',')
elements[2,1] = 1.0
elements[3,1] = 1.0

m = zeros(n)
x0 = zeros(3, n)
v0 = zeros(3, n)
# Define which pairs will have impulse rather than -drift+Kepler:
pair = ones(Bool, n, n)  # This does impulses
# We want Keplerian between star & planets, and impulses between
# planets.  Impulse is indicated with 'true', -drift+Kepler with 'false':
for i = 2:n
    pair[1,i] = false  # This does a Kepler + drift
  # We don't need to define this, but let's anyways:
    pair[i,1] = false
end

# jac_step = zeros(7*n,7*n)

for k = 1:n
    m[k] = elements[k,1]
end
m0 = copy(m)

init = ElementsIC(t0, H, elements)
ic_big = ElementsIC(big(t0), H, big.(elements))
s0 = State(init)
s0big = State(ic_big)

# Tilt the orbits a bit:
s0.x[2,1] = 5e-1 * sqrt(s0.x[1,1]^2 + s0.x[3,1]^2)
s0.x[2,2] = -5e-1 * sqrt(s0.x[1,2]^2 + s0.x[3,2]^2)
s0.x[2,3] = -5e-1 * sqrt(s0.x[1,2]^2 + s0.x[3,2]^2)
s0.v[2,1] = 5e-1 * sqrt(s0.v[1,1]^2 + s0.v[3,1]^2)
s0.v[2,2] = -5e-1 * sqrt(s0.v[1,2]^2 + s0.v[3,2]^2)
s0.v[2,3] = -5e-1 * sqrt(s0.v[1,2]^2 + s0.v[3,2]^2)

# Take a step:
s0big.x .= big.(s0.x)
s0big.v .= big.(s0.v)
ahl21!(s0,h,pair)
ahl21!(s0big,big(h),pair)

# Now, copy these to compute Jacobian (so that I don't step
# x0 & v0 forward in time):
s = deepcopy(s0)
s.xerror .= 0.0; s.verror .= 0.0
# Compute jacobian exactly:
d = Derivatives(Float64, s.n);
s.jac_step .= 0.0
d.dqdt_kick .= 0.0
kickfast!(s,d,h,pair)
# Add in identity matrix:
for i = 1:7 * n
    d.jac_kick[i,i] += 1
end

# Now compute numerical derivatives, using BigFloat to avoid
# round-off errors:
jac_step_num = zeros(BigFloat, 7 * n, 7 * n)
# Save these so that I can compute derivatives numerically:
xsave = copy(s0big.x)
vsave = copy(s0big.v)
msave = copy(s0big.m)
hbig = big(h)

sbig = deepcopy(s0big)
sbig.x .= xsave
sbig.v .= vsave
sbig.m .= msave
sbig.xerror .= 0.0; sbig.verror .= 0.0
# Carry out step using BigFloat for extra precision:
kickfast!(sbig,hbig,pair)
# Save back to use in derivative calculations
xsave .= sbig.x
vsave .= sbig.v
msave .= sbig.m
sbig = deepcopy(s0big)
sbig.xerror .= 0.0; sbig.verror .= 0.0
# Compute numerical derivatives wrt time:
dqdt_num = zeros(BigFloat, 7 * n)
# Vary time:
kickfast!(sbig,hbig,pair)
# Initial positions, velocities & masses:
sbig = deepcopy(s0big)
sbig.xerror .= 0.0; sbig.verror .= 0.0
hbig = big(h)
dq = dlnq * hbig
hbig += dq
kickfast!(sbig,hbig,pair)
# Now x & v are final positions & velocities after time step
for i in 1:n, k in 1:3
    dqdt_num[(i - 1) * 7 +  k] = (sbig.x[k,i] - xsave[k,i]) / dq
    dqdt_num[(i - 1) * 7 + 3 + k] = (sbig.v[k,i] - vsave[k,i]) / dq
end
hbig = big(h)
# Vary the initial parameters of planet j:
for j = 1:n
  # Vary the initial phase-space elements:
    for jj = 1:3
  # Initial positions, velocities & masses:
        sbig = deepcopy(s0big)
        sbig.xerror .= 0.0; sbig.verror .= 0.0
        dq = dlnq * sbig.x[jj,j]
        if sbig.x[jj,j] != 0.0
            sbig.x[jj,j] +=  dq
        else
            dq = dlnq
            sbig.x[jj,j] = dq
        end
        kickfast!(sbig, hbig, pair)
        # Now x & v are final positions & velocities after time step
        for i = 1:n
            for k = 1:3
                jac_step_num[(i - 1) * 7 +  k,(j - 1) * 7 + jj] = (sbig.x[k,i] - xsave[k,i]) / dq
                jac_step_num[(i - 1) * 7 + 3 + k,(j - 1) * 7 + jj] = (sbig.v[k,i] - vsave[k,i]) / dq
            end
        end
        sbig = deepcopy(s0big)
        sbig.xerror .= 0.0; sbig.verror .= 0.0
        dq = dlnq * sbig.v[jj,j]
        if sbig.v[jj,j] != 0.0
            sbig.v[jj,j] +=  dq
        else
            dq = dlnq
            sbig.v[jj,j] = dq
        end
        kickfast!(sbig, hbig, pair)
        for i = 1:n
            for k = 1:3
                jac_step_num[(i - 1) * 7 +  k,(j - 1) * 7 + 3 + jj] = (sbig.x[k,i] - xsave[k,i]) / dq
                jac_step_num[(i - 1) * 7 + 3 + k,(j - 1) * 7 + 3 + jj] = (sbig.v[k,i] - vsave[k,i]) / dq
            end
        end
    end
    # Now vary mass of planet:
    sbig = deepcopy(s0big)
    sbig.xerror .= 0.0; sbig.verror .= 0.0
    dq = sbig.m[j] * dlnq
    sbig.m[j] += dq
    kickfast!(sbig, hbig, pair)
    for i = 1:n
        for k = 1:3
            jac_step_num[(i - 1) * 7 +  k,j * 7] = (sbig.x[k,i] - xsave[k,i]) / dq
            jac_step_num[(i - 1) * 7 + 3 + k,j * 7] = (sbig.v[k,i] - vsave[k,i]) / dq
        end
    # Mass unchanged -> identity
        jac_step_num[7 * i,7 * i] = big(1.0)
    end
end
jac_step_num = convert(Array{Float64,2}, jac_step_num)
dqdt_num = convert(Array{Float64,1}, dqdt_num)

@test isapprox(d.jac_kick, jac_step_num;norm=maxabs)
@test isapprox(d.dqdt_kick, dqdt_num;norm=maxabs)
end