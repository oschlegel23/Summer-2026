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
	kdv_params = KdVParams(C3 = 0.2, tfin = 2.0)
	seed = 0	# Seed for the random sampling.

	# Compute a few parameters.
	@unpack kmax, C2, C3, tfin = kdv_params	
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

	#--- Propagate the initial condition forward in time via KdV. ---#
	tvals1, uhats1 = Taylor_KdV(u0, kdv_params)

	#--- Propagate the second half after a depth change. ---#
	# Use the end state as the new initial condition.
	kdv_params2 = KdVParams(C3 = 4.0, C2 = 1/200, tfin = 10.0)
	tvals2, uhats2 = Taylor_KdV(uhats1[:,end], kdv_params2)
	tvals = [tvals1; tfin .+ tvals2]
	uhats = [uhats1 uhats2]

	# Set up the grid to compute u in physical space.
	dx = 2*pi/n_ints
	xgrid = -pi .+ (0:n_ints)*dx

	#--- Make the simulation movie. ---#
	# Initialize the canvas.
	uphys = uphys_direct(uhats[:,1], xgrid)	
	plt = plot(xgrid, uphys, 
				linewidth=2, size=(800, 400), 
				xlabel="Space (x)", ylabel="u(x,t)", title="TKdV", ylims=(-1.5, 2.5), 
				label = @sprintf("%.2f", round(tvals[1], sigdigits=3) ) )
	# Create the animation.
	anim = @animate for j in 1:length(tvals)
		# Extract the attributes and update them.
		uphys = uphys_direct(uhats[:,j], xgrid)
		attrs = plt.series_list[1].plotattributes
		attrs[:y] = uphys
		attrs[:label] = @sprintf("%.2f", round(tvals[j], sigdigits=3) )
		plt
	end
	# Save the animation.
	gif(anim, "kdv_movie.gif", fps=15)
end




#test_sample()
make_movie()

