#---------------------------------------------------#
#= In this version, each thread draws from a normal distribution,
computes the acceptance probability, and then accepts/rejects.
=#

using Parameters, Distributions, Statistics, LinearAlgebra
using Roots, Optim, Random, FFTW, Test, Distributed, SharedArrays
using JLD2, DelimitedFiles

#= REMEMBER: In this code, uhat[k] = xsamp[k] - im*xsamp[k+K]
which is slightly different from what is in the paper, uhat = sqrt(E0/2pi)*() 
So xhat is not a unit vector; xhat = sqrt(E0/2pi) * x/norm(x). =#

# Parameters
@with_kw struct GibbsParams
	nmodes::Integer = 16
	E0::Real = 1
	bprime::Real = 40
	cratio::Real = 120
	nsamps_per_thread::Integer = 2*10^5	# Default 2e5
	min_samps_accept::Integer = 0
	max_samps_accept::Integer = 5000
	min_time::Real = 0
	parallel::Bool = true
	good_g::Bool = true
	seed::Int = 0
end

#---------------------------------------------------#
## Lowest level routines that must be fast.

# Compute H2 in the Hamiltonian.
function compute_h2(xsamp, nmodes)
	h2 = 0
	for k in 1:nmodes
		h2 += k^2 * (xsamp[k]^2 + xsamp[k+nmodes]^2)
	end
	return 2*pi*h2
end

# Compute H3 in the Hamiltonian.
function compute_h3(xsamp, nmodes)
	h3 = 0
	for n = 2:nmodes
		uhn_conj = xsamp[n] + im*xsamp[n+nmodes]
		for k = 1:n-1
			h3 += real( uhn_conj * (xsamp[k]-im*xsamp[k+nmodes]) * (xsamp[n-k]-im*xsamp[n-k+nmodes]) )
		end
	end
	return 2*pi*h3
end

# Compute the ratio f/g.
function compute_fg_ratio(xsamp, alpha, K, E0, bprime, cratio, good_g)
	h2 = compute_h2(xsamp, K)
	h3 = compute_h3(xsamp, K)
	gibbs_val = exp((bprime/(E0*K^2))*(-h2+cratio*h3))
	gval = (1 + alpha*bprime/K^3 * h2/E0)^(-K)
	if !(good_g); gval = 1.0; end
	return gibbs_val/gval
end

# Multiply a provisional normal sample by the sigmas for the good g.
function mult_sigs!(xsamps,sigmas)
	nmodes = length(sigmas)
	for n in 1:size(xsamps,2)
		for k in 1:nmodes
			xsamps[k,n] *= sigmas[k]
			xsamps[k+nmodes,n] *= sigmas[k]
		end
	end
end

# Normalize a provisional sample.
function normalize_samps!(xsamps,E0)
	cnst = sqrt(E0/(2*pi))
	for n in 1:size(xsamps,2)
		xnorm = 0.0
		for k in 1:size(xsamps,1)
			xnorm += xsamps[k,n]^2
		end
		for k in 1:size(xsamps,1)
			xsamps[k,n] *= cnst/sqrt(xnorm)
		end
	end
end

#---------------------------------------------------#
# Draw provisional samples from Gaussian and normalize.
function draw_xhats(nsamps, sigmas, nmodes, E0, good_g)
	# Draw xsamp from a standard Gaussian.
	xsamps = randn(2*nmodes, nsamps)

	# If using good_g, multiply by sigmas for anisotropic Gaussian, in parallel.
	good_g ? mult_sigs!(xsamps,sigmas) : 0

	# Normalize the samples in parallel.
	normalize_samps!(xsamps,E0)

	return xsamps
end

#---------------------------------------------------#
# Decide whether to accept or reject a sample.
accept_reject!(prob, accepted, n) = ( prob >= rand() && push!(accepted, n) )

#---------------------------------------------------#
# Generate samples from the Gibbs measure on a single thread.
function draw_gibbs_samps(alpha, rej_const, sigmas, params::GibbsParams)
	# Initialize.
	nmodes, E0, = params.nmodes, params.E0
	bprime, cratio, good_g = params.bprime, params.cratio, params.good_g

	# Draw provisional samples.
	nsamps = params.nsamps_per_thread
	xsamps = draw_xhats(nsamps, sigmas, nmodes, E0, good_g)

	# Decide whether to accept each.
	accepted = Array{Int}(undef,0)
	for n in 1:nsamps
		@inbounds prob = compute_fg_ratio(xsamps[:,n],alpha,nmodes,E0,bprime,cratio,good_g) / rej_const
		prob <= 1 ? accept_reject!(prob, accepted, n) : error("Prob > 1.")
	end
	return xsamps[:,accepted]
end

#---------------------------------------------------#
# Draw a batch of samples in parallel.
function draw_gibbs_batch(alpha, rej_const, sigmas, params::GibbsParams)
	# Initialize a vector to store data accross different threads.
	num_threads = Threads.nthreads()
	xsamp_each = Vector{Any}(undef,num_threads)

	# Have each thread generate samples.
	if params.parallel
		Threads.@threads for thread in 1:num_threads
			xsamp_each[thread] = draw_gibbs_samps(alpha, rej_const, sigmas, params)
		end
	else
		for thread in 1:num_threads
			xsamp_each[thread] = draw_gibbs_samps(alpha, rej_const, sigmas, params)
		end
	end

	# Combine all accepted samples into single array.
	xall = zeros(2*params.nmodes,0)
	for thread in 1:num_threads
		xall = [xall xsamp_each[thread]]
	end
	return xall
end

#---------------------------------------------------#
# Compute the sigma square values.
function sig_squares(alpha, nmodes, bprime)
	sigs = zeros(nmodes)
	for k in 1:nmodes
		sigs[k] = 1/(1+(alpha*bprime*(k^2)/(nmodes^3)))
	end
	return sigs
end

# Numerically maximize the ratio f/g.
function max_fg(alpha, params::GibbsParams)
	# Initialize.
	nmodes, E0, = params.nmodes, params.E0
	bprime, cratio, good_g = params.bprime, params.cratio, params.good_g
	cnst = sqrt(E0/(2*pi))

	# Short version of fg_ratio to use in optimization.
	function fg_short(xsamp::Vector)
		xsamp *= cnst / sqrt(sum(xsamp.^2))
		return compute_fg_ratio(xsamp,alpha,nmodes,E0,bprime,cratio,good_g)
	end
    res = Optim.maximize(fg_short, [cnst*ones(nmodes)/sqrt(nmodes); zeros(nmodes)], NelderMead(), Optim.Options(g_tol = 1e-2))
	biggest = Optim.maximum(res)
	
	# If cratio = 0, do not use same initial guess, use random initial guesses.
	if ( (cratio < 1e-6) || (!good_g) )
		for i in 1:100
			z = randn(2*nmodes)
			res = Optim.maximize(fg_short, z, NelderMead())
			new_max = Optim.maximum(res)
			if new_max > biggest
				biggest = new_max
			end
		end
	end
	return biggest
end

# Main program to draw several batches of samples.
function gibbs_sample(params::GibbsParams)
	# Set the overall seed for the random number generators.
	Random.seed!(params.seed)
	# Initialize nmodes and bprime.
	nmodes, bprime = params.nmodes, params.bprime

	# Precompute alpha, the rejection constant, and sigmas
	function F_alpha(alpha)
		sig_sq = sig_squares(alpha, nmodes, bprime)
		return 1 - alpha/nmodes * sum(sig_sq)
	end
	alpha = find_zero(F_alpha,1)
	rej_const = max_fg(alpha, params)
	sigmas = sqrt.(sig_squares(alpha, nmodes, bprime) )

	# Initialize computational parameters.
	min_samps_accept, min_time = params.min_samps_accept, params.min_time
	xdata = zeros(2*nmodes,0)  
	loop_count = 0; num_accepted = 0
	println("\n-----------------------------")
	println("Running main")
	println("Targets: minimum $(min_samps_accept) samples and $(min_time) minutes.")

	# Draw samples in several batches.
	t0 = time(); running_time = 0
	while (num_accepted < min_samps_accept || running_time < min_time)
		xdata = [xdata draw_gibbs_batch(alpha, rej_const, sigmas, params)]
		num_accepted = size(xdata,2)
		loop_count += 1
		running_time = (time()-t0)/60
		if loop_count % 10 == 0
			println("Iteration $(loop_count): $(num_accepted) accepted, $(round(running_time, sigdigits=2)) mins.")
		end
		if num_accepted >= params.max_samps_accept
			break
		end
	end
	# Compute the overall acceptance rate and return.
	accept_rate = num_accepted / (loop_count * params.nsamps_per_thread * Threads.nthreads())
	return xdata[:, 1:min(end, params.max_samps_accept)], accept_rate
end
