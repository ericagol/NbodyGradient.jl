# Reads in the string that describes the hierarchy,
# and sets up A matrix and masses for each Keplerian.
# There are N bodies, and N-1 Keplerians.

function get_indices(string)
# Takes a Kepler hierarchy vector, and returns indices of all planets
ind1 = readdlm(IOBuffer(hierarchy[1:i]))
# Converts strings to integers
nm = size(string)
for j=1:nm
  replace(string[j],"(","")
  replace(string[j],")","")
end
indices = zeros(Int64,nm)
for j=1:nm
  indice[j]=parse(string[j])
end
return indices
end

function split_kepler(hierarchy,mass)
# Splits the hierarchy into components
ndelim = count(c -> c == ',' , hierarchy)
if ndelim > 1 
  # Find first complete Keplerian:
  i = 1
  nleft = 0
  nright = 0
  c = collect(hierarchy)
  nchar = endof(hierarchy)
  while (nleft == 0) || (nleft != nright) || i == nchar
    if c[i] == '('
      nleft += 1
    end
    if c[i] == ')'
      nright += 1
    end
    if c[i] == ','
      ndelim += 1
    end
  end
  @assert(nleft == nright)
  @assert(nleft == ndelim)
  # Compute sum of masses of both Kepler components:
  # First find the indices of planets:
  ind1 = get_indices(hierarchy[1:i])
  ind2 = get_indices(hierarchy[i+1:nchar])
  m1 = sum(mss[ind1])
  m2 = sum(mss[ind2])
  return m1,m2,hierarchy[1:i],hierarchy[i+1:nchar]
elseif ndelim == 1
  # Found a Keplerian of two bodies
  ind = get_indices(hierarchy)
  return mass[ind[1]],mass[ind[2]],split(hierarchy,",")
else
  # Down to a single body
  ind = get_indices(hierarchy)
  return mass[ind[1]],0.0,hierarchy,""
end
end

function strip_parentheses(hierarchy)
# Strips the leading and trailing parentheses (if they exist):
nchar = endof(hierarchy)
if hierarchy[1] == '(' && hierarchy[nchar] == ')'
  return hierarchy[2:nchar-1]
else
  return hierarchy
end
end

nleft  = count(c -> c == '(',hierarchy)
nright = count(c -> c == ')',hierarchy)
@assert(nleft == nright)
return
end

function setup_hierarchy(hierarchy,nbody,mass)
# Creates a routine which computes the matrix 
# and total masses in each Keplerian.
nleft  = count(c -> c == '(',hierarchy)
nright = count(c -> c == ')',hierarchy)
ndelim = count(c -> c == ',',hierarchy)
@assert(nleft == nright)
@assert(nleft == ndelim)


return A,mtot
end
