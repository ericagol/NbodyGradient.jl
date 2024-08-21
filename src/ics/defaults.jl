# Some default initial conditions for testing and quick usage.

function get_trappist1_elements(t0=0.0, n=0)
    # Trappist-1 orbital elements from Agol et al. 2021
    elements = [
        1.0 0.0 0.0 0.0 0.0 0.0 0.0
        2.5901135977661885e-5 1.510880055106516  7257.547487248826   0.02436651768325364    0.018169884000968452  1.5707963267948966 0.0
        5.7871255112412840e-5 2.4218013609356652 7258.592163817471   0.020060810686211832   0.011189705094395375  1.5707963267948966 0.0
        1.4602772830539989e-6 4.0503542353950355 7257.023855669221   0.007411490159357976  -0.02016424872931776   1.5707963267948966 0.0
        1.9235328222249013e-5 6.099281590191818  7257.816770447013   0.0011801938769616127  0.000731913417670215  1.5707963267948966 0.0
        2.7302687390082730e-5 9.20618480814173   7257.1228936246725 -699952921060827e-19    0.0002252519365921506 1.5707963267948966 0.0
        3.5331017018761430e-5 12.353988709624156 7257.667328639113  -0.0009722026578612578  0.001276000403979281  1.5707963267948966 0.0
        1.6410627049780406e-6 18.733535095576702 7250.524231929195  -0.010402303111464135  -0.014289870200773339  1.5707963267948966 0.0      
    ]
    return elements
end

function get_kepler36_elements(t0=0.0, n=0)
    # Mean Kepler-36 orbital elements from Carter et al. 2012
    # Eccentricity chosen at random from e < 0.04, inclination co-planar
    # Unit conversions using UnitfulAstro
    elements = [
        1.0 0.0 0.0 0.0 0.0 0.0 0.0
        1.2479484222582662e-5 13.83989 0.0                0.002 -0.004 1.5707963267948966 0.0
        2.2659378094037732e-5 16.23855 5.062100000213832 -0.01   0.007 1.5707963267948966 0.0
    ]
    return elements
end

# Available names for users to specify to get initial conditions
# The first element in each tuple should be the key for ELEMENTS_FUNCTIONS
# rest are "aliases" for the system, if any.
const AVAILABLE_SYSTEMS = (
    ("trappist-1", "trappist 1"),
    ("kepler-36", "kepler 36")
)

const ELEMENTS_FUNCTIONS = Dict(
    "trappist-1" => get_trappist1_elements,
    "kepler-36" => get_kepler36_elements
)

"""Return the AVAILABLE_SYSTEMS tuple. ONLY FOR TESTS."""
_available_systems() = AVAILABLE_SYSTEMS

"""Show the available default systems with implemented initial conditions."""
function available_systems()
    println(stdout, "Available Systems: ")
    for s in AVAILABLE_SYSTEMS
        key = first(s)
        rest = setdiff(s, (first(s),))
        if isempty(rest)
            println(stdout, "\"$(key)\"")
        else
            println(stdout, "\"$(key)\" ", rest)
        end
    end
    flush(stdout)
end

"""Get the default initial conditions for a particular system"""
function get_default_ICs(system_name, t0=0.0, n=0)
    # Get the key for the elements function to call
    system_key = ""
    for available_names in AVAILABLE_SYSTEMS
        if lowercase(system_name) ∈ available_names
            system_key = first(available_names)
        end
    end
    system_key == "" && throw(ArgumentError("$(system_name) not an available system name."))

    # Get the orbital elements for the selected system
    elements = ELEMENTS_FUNCTIONS[system_key]()

    # Use all the bodies if user did not specify n
    nmax = size(elements)[1]
    if n == 0
        n = nmax
    elseif n > nmax
        @warn "Maximum n is $(nmax). User asked for $(n). Setting to $(nmax)."
        n = nmax
    end
    
    return ElementsIC(t0, n, elements)
end
