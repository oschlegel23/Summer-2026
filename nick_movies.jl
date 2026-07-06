using Plots, FFTW, LinearAlgebra, Statistics, Printf
using Parameters, Distributions, Roots, Optim, Random, Test, Distributed, SharedArrays
using JLD2, DelimitedFiles

# Force GR to run in "headless" mode. This stops the gksqt window 
# from opening and prevents the freeze.
ENV["GKSwstype"] = "100"

include("taylor_KdV_solver.jl")
include("gibbs_sample.jl")

# Quick function just to test the sampling.
function test_sample()
	# ── Sample initial condition from main.jl ─────────────────────────────────────
	gibbs_params = GibbsParams(
		min_samps_accept = 1, max_samps_accept = 5, seed = 1)
	xdata, accept_rate = gibbs_sample(gibbs_params)
	n_samps = size(xdata, 2)
	println("Samples accepted: ", n_samps)
	println("First sample: ", xdata[1,1])
end

# Function to make a move of the TKdV simulation.
function make_movie()
	# Set plot default.
	default(
    fontfamily = "Times New Roman",
    titlefont  = 18,  # Title size
    guidefont  = 14,  # Axis labels (xlabel and ylabel) size
    tickfont   = 11,  # Axis tick numbers size
    legendfont = 12   # Legend text size
	)

	## Set Parameters
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

	# ── Sample initial condition from main.jl ─────────────────────────────────────
	gibbs_params = GibbsParams(nmodes = K, cratio = C3/C2, 
					min_samps_accept=1, max_samps_accept=5, seed=seed)
	xdata, accept_rate = gibbs_sample(gibbs_params)
	n_samps = size(xdata, 2); println("Samples accepted: ", n_samps)


	## Use the sample as the initial condition
	samp = xdata[:, 1]
	real_parts = samp[1:K]
	imag_parts = samp[K+1:end]
	u0 = zeros(ComplexF64, K+1)
	u0[2:K+1] = real_parts .+ im .* imag_parts

	# Simple IC
	#u0 = zeros(ComplexF64, K+1)
	#u0[2] = -1.0

	E0_samp = 2π * sum(abs2.(u0[2:end]))
	u0 ./= sqrt(E0_samp / gibbs_params.E0)

	###### CHECK: was u0 not natively normalized?

	## Propagate the initial condition forward in time via KdV.
	t_T, uk_T, Energy_T, M_T, H_T, H2_T, H3_T, U_phys_T = Taylor_KdV(C2, C3, K, a, u0, dt_num, tfin, P)
	
	# Sparsify the time to save data.
	idx = 1:save_every:size(uk_T, 2)
	t_sparse = t_T[idx]
	uk_sparse = uk_T[:, idx]
	uphys_sparse = U_phys_T[:, idx]

    N = 4*(2K+1)
    dx = 2*pi/N	
	xgrid = -pi .+ (0:N-1)*dx

	println("Initial time: $(t_sparse[1])")
    #println("Initial u: $(uphys_sparse[:, 1])")

	# Initialize the Canvas
	plt = plot(xgrid, uphys_sparse[:, 1], 
				linewidth=2, size=(800, 400), 
				xlabel="Space (x)", ylabel="u(x,t)", title="TKdV", ylims=(-1.5, 2.5), 
				label = @sprintf("%.2f", round(t_sparse[1], sigdigits=3) ) )

	# Define the Animation using the pre-built plot
	anim = @animate for j in 1:length(t_sparse)
		# Extract the attributes dictionary for the first series
    	attrs = plt.series_list[1].plotattributes
    
    	# Update data AND the legend label simultaneously
    	attrs[:y] = uphys_sparse[:, j]
    	attrs[:label] = @sprintf("%.2f", round(t_sparse[j], sigdigits=3) )
	    
	    # We must explicitly call plt at the end of the block so 
	    # @animate knows what to capture for the frame
	    plt
	end

	# 4. Save the Animation
	gif(anim, "kdv_movie.gif", fps=15)
end

#test_sample()

make_movie()
