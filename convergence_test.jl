cd(@__DIR__)
using Plots, Random
using Plots.PlotMeasures

include("Taylor_KdV_no_exp_int.jl")   # transitively includes KdV_solver.jl
                                       # gives us Taylor_KdV (EI) and Taylor_KdV_noEI

include("gibbs_sample.jl")
using .GibbsSample

# ── Initial condition: draw one sample from the Gibbs measure ────────────────
kmax = 16
gibbs_params = GibbsParams(nmodes=kmax, E0=1, min_samps_accept=1, max_samps_accept=1)
xdata, accept_rate = gibbs_sample(gibbs_params)
xsamp = xdata[:, 1]

kpos  = 0:kmax
u0hat = zeros(ComplexF64, kmax+1)
u0hat[2:end] = xsamp[1:kmax] .- im .* xsamp[kmax+1:end]

# ── Problem / method parameters ──────────────────────────────────────
C2    = 1/120
C3    = 1.0
tfin  = 1.0
order = 3

H0 = computeH(u0hat[2:end], C2, C3)

# ── Convergence sweep over h ──────────────────────────────────────────
h_ref = 1e-2
hs = [h_ref / 2^n for n in 0:6]

errors_EI   = zeros(length(hs))
errors_noEI = zeros(length(hs))

for (i, h) in enumerate(hs)
	params_EI   = KdVParams(kmax=kmax, C2=C2, C3=C3, tfin=tfin, order=order, dt_num=h, dt_save=h)
	params_noEI = KdV_noexpint_Params(kmax=kmax, C2=C2, C3=C3, tfin=tfin, order=order, dt_num=h, dt_save=h)

	_, uh_EI   = Taylor_KdV(u0hat, params_EI)
	_, uh_noEI = Taylor_KdV_noEI(u0hat, params_noEI)

	errors_EI[i]   = abs(computeH(uh_EI[2:end, end], C2, C3)   - H0) / abs(H0)
	errors_noEI[i] = abs(computeH(uh_noEI[2:end, end], C2, C3) - H0) / abs(H0)

	# println("h = $h   EI error = $(errors_EI[i])   no-EI error = $(errors_noEI[i])")
end

# ── Plot: both methods on the same axes, with an h^3 reference line ───
i_ref = length(hs)
scale3 = 3 * errors_noEI[i_ref] / hs[i_ref]^3

convergence = plot(hs, errors_EI, xscale=:log10, yscale=:log10, marker=:circle,
	xlabel="h", ylabel="|H(t_fin) - H(0)| / |H(0)|",
	title="Hamiltonian Convergence (order $order)", label="Exponential Integrator",
	left_margin=8mm)
plot!(convergence, hs, errors_noEI, marker=:square, label="No Exponential Integrator")
plot!(convergence, hs, scale3 .* hs.^3, linestyle=:dash, color=:black, label="h^3")
display(convergence)
