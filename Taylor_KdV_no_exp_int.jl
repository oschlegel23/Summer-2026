"""
KdV Taylor solver WITHOUT the exponential integrator (h-only form).
Builds on KdV_solver.jl (same derivative recursion); only the
update step changes to a plain truncated Taylor series in h, per Week 6
report, section "Taylor Method no Exponential Integrator, h dependence".
"""

include("KdV_solver.jl")
using Parameters

# Parameters
@with_kw struct KdV_noexpint_Params
	kmax::Integer = 16
	C2::Float64 = 1/120
	C3::Float64 = 1.0
	tfin::Float64 = 10.
	order::Integer = 3
	dt_num  = 1e-3
	dt_save = 0.025	
end


"""
Advance û_k by one Taylor step of size h, to order (no exponential integrator).

Formula:
  û_k(h) = û_k(0) + Σₘ₌₁^order (hᵐ/m!) ∂ₜ^(m) û_k(0)
"""
function taylor_step_noEI(u0_pos, alpha, C3, kpos, h, num_derivs, kmax)
	dtu, g = compute_time_derivatives(u0_pos, alpha, C3, kpos, num_derivs, kmax)

	taylor_sum = zeros(ComplexF64, kmax+1)
	for m in 1:num_derivs
		taylor_sum .+= dtu[m+1, :] .* h^m ./ factorial(m)
	end

	u1_pos = u0_pos .+ taylor_sum

	return u1_pos
end


"""
Main Taylor ODE solver for KdV in Fourier space (no exponential integrator).
"""
function Taylor_KdV_noEI(u0hat, kdv_params::KdV_noexpint_Params)
	@unpack kmax, C2, C3, tfin, order, dt_num, dt_save = kdv_params
	save_every = round(Int, dt_save / dt_num)

	kpos   = 0:kmax
	alpha = -im .* C2 .* Float64.(kpos).^3

	uhat = copy(u0hat)
	tval = 0.
	count = 0
	uh_saved = copy(u0hat)
	tval_saved = 0.	
	while tval < tfin
		uhat = taylor_step_noEI(uhat, alpha, C3, kpos, dt_num, order, kmax)
		tval += dt_num
		count += 1
		if mod(count, save_every) == 0
			uh_saved = [uh_saved uhat]
			tval_saved = [tval_saved; tval]
		end
	end

	return tval_saved, uh_saved
end