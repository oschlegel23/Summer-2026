#--- GOAL: Make movies of KdV simulations ---#

using Plots, FFTW, LinearAlgebra, Statistics, Printf
using Parameters, Distributions, Roots, Optim, Random, Test, Distributed, SharedArrays
using JLD2, DelimitedFiles
using Plots.Measures

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
	# Set: depth ratio=1/4; downstream C3/C2=120; downstream C3=1
	# Use depth rescalings C3 D3^(-3/2), C2 D2^(1/2)
	# Get: upstream C3 = 1/8, C2 = 1/60 

	depth1 = 12
	depth2 = 3
	bprime = 400.
	# 
	kdv_params1 = KdVParams(C3 = 1/8, C2 = 1/60, tfin = 2.0)
	kdv_params2 = KdVParams(C3 = 1.0, C2 = 1/120, tfin = 3.0)
	# The seed for random sampling.
	seed = 0
	# The background grid step size
	c0 = 4.0 	# Linear wave speed to propagate moving frame.
	dx_bg = 2*pi/20

	# Compute a few parameters.
	@unpack kmax, C2, C3 = kdv_params1	
	tfin1 = kdv_params1.tfin
	n_ints = 10*kmax		# The number of intervals for the physical grid.
	dchange = depth1-depth2

	#--- Sample initial conditions from main.jl ---#	
	gibbs_params = GibbsParams(nmodes=kmax, cratio=C3/C2, bprime=bprime,
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

	#--- Make the simulation movie. ---#
	# Set up the grid to compute u in physical space.
	dx = 2*pi/n_ints
	xgrid = -pi .+ (0:n_ints)*dx	
	# Compute the physical displacements.
	uphys1 = uphys_many(uhats1, xgrid)
	uphys2 = uphys_many(uhats2, xgrid)
	# Combine them.
	uphys = [uphys1 uphys2] .+ depth1
	tvals = [tvals1; tfin1 .+ tvals2]


	# Heaviside function to define the depth.
	heaviside(x) = Float64(x >= 0)

	# Initialize the plot with the wave free surface.
	plt = plot(xgrid, uphys[:,1], linewidth=2, legend=:none)

	# Add the topography (just initialization)
	x_bg = -pi:dx_bg:pi
	plot!(plt, x_bg, 0*x_bg, 
		  line=:solid, marker=:circle, markersize=3, 
		  linecolor=:black, markercolor=:black, 
		  markerstrokewidth=0, legend=:none)

	# Apply global layout cosmetics to the canvas
	plot!(plt, size=(800, 400), 
		  xlabel="x", ylabel="u(x,t)", 
		  xlims=(-pi, pi), ylims=(-0.5, 1.2*(depth1+2.0)),
		  bottom_margin=10mm, left_margin=10mm )

	# Cache the attribute dictionaries ONE TIME outside the loop
	attrs1 = plt.series_list[1].plotattributes        
	attrs2 = plt.series_list[2].plotattributes       

	# Create the animation.
	anim = @animate for j in 1:length(tvals)
		tval = tvals[j]
		
		# Propagate the moving frame to adjust the topography.
		x_bg = (floor((c0*tval-pi)/dx_bg):ceil((c0*tval+pi)/dx_bg))*dx_bg
		ytopo = heaviside.(x_bg .- c0*tfin1)*dchange
 
		# Update the data arrays in memory (zero overhead lookups)
		attrs1[:y] = uphys[:,j]
		attrs2[:x] = x_bg .- c0*tval
		attrs2[:y] = ytopo
		
		# Update the global layout title as a dynamic time counter
		title!(plt, @sprintf("TKdV Simulation (t = %.2f)", round(tval, sigdigits=3)))
		
		# Explicitly return the pre-configured plot canvas
		plt
	end
	
	# Save the animation.
	gif(anim, "kdv_movie.gif", fps=15)
end




#test_sample()
make_movie()

