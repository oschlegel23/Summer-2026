"""
This is Nick's version of the KdV solver.
I will start to tweak the most urgent things to 
move the code towards a more polished form.
Later, this version of the code will become the new KdV solver.
I will not touch the original version, taylor_KdV_solver, any more
so that we can learn from it.
"""


# TO DO: Finish making the parameters


include("routines.jl")
using Parameters

# Parameters
@with_kw struct KdVParams
	kmax::Integer = 16
	C2::Float64 = 1/120
	C3::Float64 = 1.0
	tfin::Float64 = 10.
	order::Integer = 3
	dt_num  = 1e-3
	dt_save = 0.05	
end


"""
Compute time derivatives of û_k up to order P, and g^(p)(t0) up to order P-1.

Returns:
  dtu[p+1, k+1] = ∂ₜ^(p) û_k  (0-indexed in p and k, stored 1-indexed)
  g[p+1, k+1]   = g^(p)(t0) = Fₖ( (1/2) Σⱼ C(p,j) ∂ₜ^(j)u · ∂ₜ^(p-j)u )
"""
function compute_time_derivatives(u0_pos, alpha, C3, kpos, P, K)
	# dtu[p+1, :] = ∂ₜ^(p) û_k, shape (P+1) x (K+1)
	dtu = zeros(ComplexF64, P+1, K+1)
	g   = zeros(ComplexF64, P+1, K+1)  # g[p+1,:] = g^(p)(t0)

	# p=0: ∂ₜ^(0) û_k = û_k itself
	dtu[1, :] = u0_pos

	# g^(0)(t0) = Fₖ( (1/2) u² )
	# the Leibniz sum with p=0: (1/2) * C(0,0) * u * u = (1/2) u²
	g[1, :] = dealias_product_direct(dtu[1,:], dtu[1,:]) ./ 2

	for p in 1:P
		# ∂ₜ^(p) û_k = -αₖ ∂ₜ^(p-1) û_k - C3*ik * g^(p-1)(t0)
		dtu[p+1, :] = -alpha .* dtu[p, :] .- C3 .* im .* kpos .* g[p, :]

		# g^(p)(t0) = Fₖ( (1/2) Σⱼ₌₀ᵖ C(p,j) ∂ₜ^(j)u · ∂ₜ^(p-j)u )
		conv_sum = zeros(ComplexF64, K+1)
		for j in 0:p
			binom_coeff = binomial(p, j)
			conv_sum .+= binom_coeff .* dealias_product_direct(dtu[j+1,:], dtu[p-j+1,:])
		end
		g[p+1, :] = conv_sum ./ 2
	end

	return dtu, g
end


"""
Compute Iₚ = ∫_{t0}^{t1} e^{α t} (t-t0)^p dt recursively.

Recursion: Iₚ = (1/α)(t1-t0)^p * e^{α t1} - (p/α) * I_{p-1}
Base case: I₀ = (1/α)(e^{α t1} - e^{α t0})

Returns vector of Iₚ values for p = 0, 1, ..., P, shape (P+1) x (K+1).
Each row is the vector over all k modes.
"""
function compute_Ip(alpha, t0, t1, P, K)
	Ip = zeros(ComplexF64, P+1, K+1)
	dt = t1 - t0

	for ki in 1:K+1
		a = alpha[ki]
		if abs(a) < 1e-14
			# α=0: Iₚ = ∫(t-t0)^p dt = dt^(p+1)/(p+1)
			for p in 0:P
				Ip[p+1, ki] = dt^(p+1) / (p+1)
			end
		else
			Ip[1, ki] = (exp(a*t1) - exp(a*t0)) / a
			for p in 1:P
				Ip[p+1, ki] = (dt^p * exp(a*t1) - p * Ip[p, ki]) / a
			end
		end
	end

	return Ip
end

"""
Advance û_k by one Taylor step from t0 to t1 = t0 + h, to order P.

Formula:
  û_k(t1) = [ e^{αₖ t0} û_k(t0) - C3*ik * Σₚ (g^(p)(t0)/p!) * Iₚ ] / e^{αₖ t1}
		   = e^{-αₖ h} û_k(t0) - C3*ik * Σₚ (g^(p)(t0)/p!) * Iₚ * e^{-αₖ t1}
"""
function taylor_step(u0_pos, alpha, C3, kpos, t0, h, P, K)
	t1 = t0 + h

	dtu, g = compute_time_derivatives(u0_pos, alpha, C3, kpos, P, K)
	Ip     = compute_Ip(alpha, t0, t1, P, K)

	# Accumulate the sum Σₚ (g^(p)(t0)/p!) * Iₚ
	taylor_sum = zeros(ComplexF64, K+1)
	for p in 0:P
		taylor_sum .+= g[p+1, :] .* Ip[p+1, :] ./ factorial(p)
	end

	# Apply formula
	exp_t1 = exp.(alpha .* t1)
	exp_t0 = exp.(alpha .* t0)

	u1_pos = (exp_t0 .* u0_pos .- C3 .* im .* kpos .* taylor_sum) ./ exp_t1

	return u1_pos
end


"""
Main Taylor ODE solver for KdV in Fourier space.

Arguments:
  C2, C3  : KdV parameters
  K       : max wavenumber (uses k = 0:K)
  a       : start time
  u0      : initial condition as Fourier coefficients (length K+1, positive modes)
  h       : timestep
  tfin    : final time
  P       : Taylor order (default 4)

Returns: t, uk, Energy, M, H, H2, H3, U_phys
"""
function Taylor_KdV(u0hat, kdv_params::KdVParams)
	@unpack kmax, C2, C3, tfin, order, dt_num, dt_save = kdv_params
	save_every = round(Int, dt_save / dt_num)

	#P = P - 1  # order P+1 method uses P derivatives of g
	num_derivs = kdv_params.order - 1 # order P method uses P-1 derivatives of g
	
	kpos   = 0:kmax
	alpha = -im .* C2 .* Float64.(kpos).^3

	# Initialize and enter loop.
	uhat = copy(u0hat)
	tval = 0.
	count = 0
	uh_saved = copy(u0hat)
	tval_saved = 0.	
	while tval < tfin
		# Taylor step
		# Hack: set t0=0 for now; I think it should be the same
		# Later, alter routines to only depend on dt and not t0 and t1.
		uhat = taylor_step(uhat, alpha, C3, kpos, 0.0, dt_num, num_derivs, kmax)
		tval += dt_num
		count += 1
		if mod(count, save_every) == 0
			uh_saved = [uh_saved uhat]
			tval_saved = [tval_saved; tval]
		end
	end

	return tval_saved, uh_saved
end

