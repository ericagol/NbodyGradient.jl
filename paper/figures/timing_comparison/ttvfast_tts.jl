using DelimitedFiles

function output_ttvfast_times()
    data = readdlm("ttvfast/Times", ' ')
    data = sortslices(data, by=x->x[1], dims=1)
    masks = [data[:,1] .== x for x in collect(0.0:1.0:6.0)]
    times = [data[mask,3] for mask in masks]
    tts = zeros(7, length(times[1]))
    for i in 1:7
        tts[i,1:length(times[i])] = times[i][1:length(times[i])]
    end
    open("ttvfast_tts.txt", "w") do io
        writedlm(io, eachrow(tts), ',')
    end
end