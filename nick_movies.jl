#--- GOAL: Make movies of KdV simulations ---#

using Plots, FFTW, LinearAlgebra, Statistics, Printf
using Parameters, Distributions, Roots, Optim, Random, Test, Distributed, SharedArrays
using JLD2, DelimitedFiles

include("gibbs_sample.jl")
include("nick_KdV_solver.jl")
using .GibbsSample

# Set plot defaults.
function set_plot_defaults()
	# Set plot default.
	default(
	fontfamily = "Times New Roman",
	titlefont  = 18,  # Title size
	guidefont  = 14,  # Axis labels (xlabel and ylabel) size
	tickfont   = 11,  # Axis tick numbers size
	legendfont = 12   # Legend text size
	)
end

# Quick function just to test the sampling.
function test_sample()
	# ── Sample initial condition from main.jl ─────────────────────────────────────
	gibbs_params = GibbsParams(
		min_samps_accept = 1, seed = 1)
	xdata, accept_rate = gibbs_sample(gibbs_params)
	n_samps = size(xdata, 2)
	println("Samples accepted: ", n_samps)
	println("First sample: ", xdata[1,1])
end

# Function to make a move of the TKdV simulation.
function make_movie()
	set_plot_defaults()
	# Prevent the gksqt window from freezing the animation.
	ENV["GKSwstype"] = "100"	
	
	#--- Set Parameters. ---#
	depth1 = 12
	depth2 = 3
	bspeed = 1.0
	# 
	kdv_params1 = KdVParams(C3 = 0.2, tfin = 2.0)
	kdv_params2 = KdVParams(C3 = 4.0, C2 = 1/200, tfin = 10.0)
	# The seed for random sampling.
	seed = 0

	# Compute a few parameters.
	@unpack kmax, C2, C3, tfin, dt_save = kdv_params1	
	n_ints = 10*kmax		# The number of intervals for the physical grid.

	#--- Sample initial conditions from main.jl ---#	
	gibbs_params = GibbsParams(nmodes = kmax, cratio = C3/C2, 
					min_samps_accept=1, seed=seed)
	xdata, accept_rate = gibbs_sample(gibbs_params)
	n_samps = size(xdata, 2); println("Samples accepted: ", n_samps)

	#--- Set the initial condition u0. ---#
	u0 = zeros(ComplexF64, kmax+1)
	# A simple initial condition.
	u0[2] = 1.0
	u0 *= sqrt(gibbs_params.E0/compute_energy(u0))
	# A sampled initial condition.
	samp_idx = 1
	u0[2:kmax+1] = xdata[1:kmax,samp_idx] .- im .* xdata[kmax+1:end,samp_idx]

	# Check that the energy is good.
	@test compute_energy(u0) ≈ gibbs_params.E0
	println("The energy is correct.")

	#--- Propagate KdV. ---#
	# The first depth
	tvals1, uhats1 = Taylor_KdV(u0, kdv_params1)
	# The second depth: use the end state as the new initial condition.
	tvals2, uhats2 = Taylor_KdV(uhats1[:,end], kdv_params2)

	# Set up the grid to compute u in physical space.
	dx = 2*pi/n_ints
	xgrid = -pi .+ (0:n_ints)*dx
	# Set up the bottom topography
	bottom = 0*xgrid

	#--- Make the simulation movie. ---#
	# Initialize the canvas.
	uphys1 = uphys_many(uhats1, xgrid) .+ depth1
	uphys2 = uphys_many(uhats2, xgrid) .+ depth1
	# Combine them.
	uphys = [uphys1 uphys2]
	tvals = [tvals1; tfin .+ tvals2]
	# Construct the bottom topography.
	x_bottom = -pi .+ (0:n_ints)*(2*pi/20)
	btopo1 = 0*x_bottom
	btopo2 = 0*x_bottom .+ (depth1-depth2)
	# Function to map points to interval [-pi,pi]
	xmap(xpts) = mod.(xpts .+ pi, 2*pi) .- pi

	plt = plot([xgrid x_bottom], [uphys[:,1] btopo1], 
			seriestype = [:path :scatter],
			linewidth = [2 0],
			markersize = [0 4], markerstrokewidth = [0 0],
			size=(800, 400), xlabel="Space (x)", ylabel="u(x,t)", 
			title="TKdV", xlims=(-pi,pi), ylims=(-0.2, 14), 
			label = @sprintf("%.2f", round(tvals[1], sigdigits=3) ) )
	# Create the animation.
	anim = @animate for j in 1:length(tvals)
		# Adjust the bottom topography.
		tvals[j] < tfin ? btopo = btopo1 : btopo = btopo2
		x_bottom = xmap(x_bottom .- bspeed*dt_save)
		# Extract the plot attributes and update them.
		attrs1 = plt.series_list[1].plotattributes		
		attrs1[:y] = uphys[:,j]
		# Bottom topography.
		attrs2 = plt.series_list[2].plotattributes		
		attrs2[:y] = btopo
		attrs2[:x] = x_bottom
		# Time.		
		attrs1[:label] = @sprintf("%.2f", round(tvals[j], sigdigits=3) )

    	# Explicitly plot
    	plt
	end
	# Save the animation.
	gif(anim, "kdv_movie.gif", fps=15)
end




#test_sample()
make_movie()

