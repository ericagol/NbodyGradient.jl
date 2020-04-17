using Distributed: clear!
# Functions to generate ϵ matrix.

################################################################################
# Sets up planetary-hierarchy index matrix using an initial condition file
# of the form "test.txt" (included in "test/" directory), or in a 2d array of
# the form file = [x ; "x,y,z,..."] where x,y,z are Int64
################################################################################
function hierarchy(file::Array{Int64,1})

    # Separates the initial array into variables
    nbody = file[1]
    bins = file[2:end]
    #bins = replace(bins,"," => " ")
    #bins = readdlm(IOBuffer(bins),Int64)
    level = length(bins)

    # checks that the hierarchy can be created
    @assert bins[1] <= nbody/2 && bins[1] <= 2^level

    # Creates an empty array of the proper size
    h = zeros(Float64,nbody,nbody)
    # Starts filling the array and returns it
    bottom_level(nbody,bins,h,1)
    h[end,1:end] .= -1
    return h
end

################################################################################
# Sets up the first level of the hierchary.
################################################################################
function bottom_level(nbody::Int64,bins::Array{Int64,1},h::Array{Float64},iter::Int64)

    # Fills the very first level of the hierarchy and iterates related variables
    h[1,1] = -1.0
    h[1,2] = 1.0
    if bins[1] > 1
        for i=1:bins[1]-1
            if !(@isdefined j) || isa(j,Nothing)
                global j = i
            elseif (@isdefined j) && i == 1
                global j = 1
            end
            if (j+3) <= nbody
                h[i+1,j+2] = -1
                h[i+1,j+3] = 1
            end
            j = +(j,2)
        end
    end
    clear!(:j)
    binsp = bins[1]
    row = binsp[1] + 1
    bodies = bins[1] * 2

    # checks whether the desired hierarchy is symmetric and calls appropriate
    # filling functions
    if 2*binsp == nbody
        symmetric(nbody,bodies,bins,binsp,row,h,iter+1)
    else
        nlevel(nbody,bodies,bins,binsp,row,h,iter+1)
    end
end

################################################################################
# Fills subsequent levels of the hierarchy, recursively.
#
# *** Only works for hierarchies with a max binaries per level of 2 (unless
# symmetric). ***
################################################################################
function nlevel(nbody::Int64,bodies::Int64,bins::Array{Int64,1},binsp::Int64,row::Int64,h::Array{Float64},iter::Int64)
#print("nbody: ", nbody, "\n")
#print("bodies: ", bodies, "\n")
#print("row: ", row, "\n")
# Series of checks to know which level to fill and which bins number to use
if iter <= length(bins)
    if nbody != bodies
        if bins[iter] == binsp
            bodies = bodies + bins[iter]
        elseif bins[iter] > binsp
            bodies = bodies + 2*(bins[iter]) - 1
        end
    elseif nbody == bodies
        if row == nbody-1
            if (bodies%4) > 2
                h[row,1:row-2] .= -1
                h[row,row-1:bodies] .= 1
                return h
            elseif (bodies%4) <= 2
                h[row,1:row-1] .= -1
                h[row,row:bodies] .= 1
                return h
            end
        end
    end

    # Looks at the previous and current bins and calls nlevel again
    if nbody >= bodies
        if binsp == 1
            if bins[iter] == 1
                h[row,1:bodies-binsp] .= -1
                h[row,bodies-binsp + 1:bodies] .= 1
                row = row + binsp
                binsp = bins[iter]
                nlevel(nbody,bodies,bins,binsp,row,h,iter+1)
            elseif bins[iter] == 2
                h[row,1:bodies-(2*binsp)-1] .= -1
                h[row,bodies-(2*binsp)] = 1
                h[row+1,bodies-(2*binsp)+1] = -1
                h[row+1,bodies] = 1
                row = row + 2
                binsp = bins[iter]
                nlevel(nbody,bodies,bins,binsp,row,h,iter+1)
            end
        elseif binsp == 2
            if bins[iter] == 1
                h[row,1:row-1] .= -1
                h[row,row:bodies] .= 1
                row = row + 1
                binsp = bins[iter]
                nlevel(nbody,bodies,bins,binsp,row,h,iter+1)
            elseif bins[iter] == 2
                h[row,1:row-1] .= -1
                h[row,row:bodies-2*(binsp-1)] .= 1
                h[row+1,bodies-2*(binsp-1)+1] = -1
                h[row+1,bodies] = 1
                row = row + 2
                binsp = bins[iter]
                nlevel(nbody,bodies,bins,binsp,row,h,iter+1)
            end
        elseif binsp == 3
            if bins[iter] == 2
                h[row,1:Int64(bodies/2)-1] .= -1
                h[row,Int64(bodies/2):2*Int64(bodies/3)] .= 1
                h[row+1,2*Int64(bodies/3)+1:bodies] .= -1
                h[row+1,bodies+1] = 1
                row = row + 2
                binsp = bins[iter]
                bodies = bodies + 1
                nlevel(nbody,bodies,bins,binsp,row,h,iter+1)
            end
        end
    end
end
end

################################################################################
# Called if the hierarchy is symmetric (or if the hierarchy has all of the
# bodies on the bottom level). Fills the remaining rows.
################################################################################
function symmetric(nbody::Int64,bodies::Int64,bins::Array{Int64,1},binsp::Int64,row::Int64,h::Array{Float64},iter::Int64)
    global j = 0
    while row < nbody-1
        h[row,1+j:2+j] .= -1
        h[row,j+3:j+4] .= 1
        j = j + 4
        row = row + 1
    end
    h[row,1:binsp] .= -1
    h[row,binsp+1:nbody] .= 1
    clear!(:j)
end
