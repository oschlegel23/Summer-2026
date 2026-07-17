"""
This is Nick's version of the KdV solver.
I will start to tweak the most urgent things to 
move the code towards a more polished form.
Later, this version of the code will become the new KdV solver.
I will not touch the original version, taylor_KdV_solver, any more
so that we can learn from it.
"""



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
	dt_save = 0.025	
end


"""
Compute time derivatives of û_k and derivatives of g, both up to order num_derivs,
all evaluated at the start of the step (t = 0).

Returns:
  dtu[p+1, k+1] = ∂ₜ^(p) û_k          for p = 0, ..., num_derivs
  g[p+1, k+1]   = g^(p)(0) = Fₖ( (1/2) Σⱼ C(p,j) ∂ₜ^(j)u · ∂ₜ^(p-j)u )
                                       for p = 0, ..., num_derivs
"""
function compute_time_derivatives(u0_pos, alpha, C3, kpos, num_derivs, kmax)
	# dtu[p+1, :] = ∂ₜ^(p) û_k, shape (num_derivs+1) x (kmax+1)
	dtu = zeros(ComplexF64, num_derivs+1, kmax+1)
	g   = zeros(ComplexF64, num_derivs+1, kmax+1)  # g[p+1,:] = g^(p)(0)

	# p=0: ∂ₜ^(0) û_k = û_k itself
	dtu[1, :] = u0_pos

	# g^(0)(0) = Fₖ( (1/2) u² )
	# the Leibniz sum with p=0: (1/2) * C(0,0) * u * u = (1/2) u²
	g[1, :] = dealias_product_direct(dtu[1,:], dtu[1,:]) ./ 2

	for p in 1:num_derivs
		# ∂ₜ^(p) û_k = -αₖ ∂ₜ^(p-1) û_k - C3*ik * g^(p-1)(0)
		dtu[p+1, :] = -alpha .* dtu[p, :] .- C3 .* im .* kpos .* g[p, :]

		# g^(p)(0) = Fₖ( (1/2) Σⱼ₌₀ᵖ C(p,j) ∂ₜ^(j)u · ∂ₜ^(p-j)u )
		conv_sum = zeros(ComplexF64, kmax+1)
		for j in 0:p
			binom_coeff = binomial(p, j)
			conv_sum .+= binom_coeff .* dealias_product_direct(dtu[j+1,:], dtu[p-j+1,:])
		end
		g[p+1, :] = conv_sum ./ 2
	end

	return dtu, g
end


"""
Compute Iₚ = ∫₀ʰ e^{α(t-h)} tᵖ dt recursively.

Recursion: Iₚ = (hᵖ - p*I_{p-1}) / α
Base case: I₀ = (1 - e^{-αh}) / α
α = 0 case: Iₚ = h^(p+1)/(p+1)

Returns matrix of Iₚ values, shape (num_derivs+1) x (kmax+1).
Depends only on h and α, so for fixed dt this can be precomputed once.
"""
function compute_Ip(alpha, h, num_derivs, kmax)
	Ip = zeros(ComplexF64, num_derivs+1, kmax+1)

	for ki in 1:kmax+1
		a = alpha[ki]
		if abs(a) < 1e-14
			# α=0: Iₚ = ∫₀ʰ tᵖ dt = h^(p+1)/(p+1)
			for p in 0:num_derivs
				Ip[p+1, ki] = h^(p+1) / (p+1)
			end
		else
			Ip[1, ki] = -expm1(-a*h) / a
			for p in 1:num_derivs
				Ip[p+1, ki] = (h^p - p * Ip[p, ki]) / a
			end
		end
	end

	return Ip
end

"""
Advance û_k by one Taylor step of size h, to order.

Formula (h-only form):
  û_k(h) = e^{-αₖ h} û_k(0) - C3*ik * Σₚ (g^(p)(0)/p!) * Iₚ
"""
function taylor_step(u0_pos, alpha, C3, kpos, h, num_derivs, kmax)
	dtu, g = compute_time_derivatives(u0_pos, alpha, C3, kpos, num_derivs, kmax)
	Ip     = compute_Ip(alpha, h, num_derivs, kmax)

	# Accumulate the sum Σₚ (g^(p)(0)/p!) * Iₚ
	taylor_sum = zeros(ComplexF64, kmax+1)
	for p in 0:num_derivs
		taylor_sum .+= g[p+1, :] .* Ip[p+1, :] ./ factorial(p)
	end

	u1_pos = exp.(-alpha .* h) .* u0_pos .- C3 .* im .* kpos .* taylor_sum

	return u1_pos
end


"""
Main Taylor ODE solver for KdV in Fourier space (exponential integrator form).

Arguments:
  u0hat      : initial condition as Fourier coefficients (length kmax+1, nonnegative modes)
  kdv_params : KdVParams struct (kmax, C2, C3, tfin, order, dt_num, dt_save)

Returns:
  tval_saved : vector of save times
  uh_saved   : matrix of saved Fourier coefficients, one column per save time
"""
function Taylor_KdV(u0hat, kdv_params::KdVParams)
	@unpack kmax, C2, C3, tfin, order, dt_num, dt_save = kdv_params
	save_every = round(Int, dt_save / dt_num)

	num_derivs = order - 1 # order P method uses P-1 derivatives of g
	
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
	
		uhat = taylor_step(uhat, alpha, C3, kpos, dt_num, num_derivs, kmax)
		tval += dt_num
		count += 1
		if mod(count, save_every) == 0
			uh_saved = [uh_saved uhat]
			tval_saved = [tval_saved; tval]
		end
	end

	return tval_saved, uh_saved
end

