# Collection of Plots recipes

"""Plot TTVs"""
@recipe function plot(tt::TransitTiming)
	N = length(tt.count)-1 # Get number of planets
	labels = Char.(98:97+N) # Get planet labels

	# Plot layout
	layout := N # N subplots
	link := :x # Share x-axis
	legend := :false

	# Compute the linear fit to times
	pavg = [mean(diff(tt[1:count])) for (tt, count) in zip(eachrow(tt.tt), tt.count)][2:end]
	lines = [p .* collect(0:count-1) .+ t0 for (p, count, t0) in zip(pavg, tt.count[2:end], tt.tt[2:end,1])]

	# Compute ttvs
	ttvs = [tt.tt[i+1,1:tt.count[i+1]] .- lines[i] for i in 1:N]

	for (i,ttv) in enumerate(ttvs)
		@series begin
			title := "Planet $(labels[i])"
			subplot := i
			tt.tt[i+1,1:tt.count[i+1]],ttv
		end
	end
end