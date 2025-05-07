using Documenter, NbodyGradient

makedocs(sitename="NbodyGradient",
    modules = [NbodyGradient],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Index" => "index.md",
        "Tutorials" => ["basic.md", "gradients.md"],
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/ericagol/NbodyGradient.jl.git",
    devbranch="master"
)
