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
	K    = 16
	C2   = 1/120
	C3   = 1.0
	a    = 0.0 #?
	tfin = 10.0
	P    = 3
	dt_num  = 1e-3
	dt_save = 0.05
	save_every = round(Int, dt_save / dt_num)
	seed = 0	# Seed for the random sampling.

	#--- Sample initial conditions from main.jl ---#
	gibbs_params = GibbsParams(nmodes = K, cratio = C3/C2, 
					min_samps_accept=1, seed=seed)
	xdata, accept_rate = gibbs_sample(gibbs_params)
	n_samps = size(xdata, 2); println("Samples accepted: ", n_samps)


	#--- Set the initial condition u0. ---#
	u0 = zeros(ComplexF64, K+1)
	# A simple initial condition.
	u0[2] = -1.0
	u0 *= sqrt(gibbs_params.E0/compute_energy(u0))
	# A sampled initial condition.
	samp_idx = 1
	#u0[2:K+1] = xdata[1:K,samp_idx] .- im .* xdata[K+1:end,samp_idx]

	# Check that the energy is good.
	@test compute_energy(u0) ≈ gibbs_params.E0
	println("The energy is correct.")

	#--- Propagate the initial condition forward in time via KdV. ---#
	tvals, uhat = Taylor_KdV(C2, C3, K, a, u0, dt_num, tfin, P)
	
	# Sparsify the time to save data.
	idx = 1:save_every:size(uk_T, 2)
	t_sparse = tvals[idx]
	uh_sparse = uhat[:, idx]
	


	# TO DO
	uphys_sparse = U_phys_T[:, idx]



	N = 4*(2K+1)
	dx = 2*pi/N	
	xgrid = -pi .+ (0:N-1)*dx

	#--- Print test quauntities. ---#
	#println("Initial time: $(t_sparse[1])")

	#--- Make the simulation movie. ---#
	# Initialize the canvas.
	plt = plot(xgrid, uphys_sparse[:, 1], 
				linewidth=2, size=(800, 400), 
				xlabel="Space (x)", ylabel="u(x,t)", title="TKdV", ylims=(-1.5, 2.5), 
				label = @sprintf("%.2f", round(t_sparse[1], sigdigits=3) ) )
	# Create the animation.
	anim = @animate for j in 1:length(t_sparse)
		# Extract the attributes and update them.
		attrs = plt.series_list[1].plotattributes
		attrs[:y] = uphys_sparse[:, j]
		attrs[:label] = @sprintf("%.2f", round(t_sparse[j], sigdigits=3) )
		plt
	end
	# Save the animation.
	gif(anim, "kdv_movie.gif", fps=15)
end

#test_sample()
make_movie()

