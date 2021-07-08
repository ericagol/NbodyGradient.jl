using Documenter, NbodyGradient

makedocs(sitename="NbodyGradient",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/ericagol/NbodyGradient.jl.git"
)
