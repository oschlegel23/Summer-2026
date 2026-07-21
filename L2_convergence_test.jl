cd(@__DIR__)
using Plots
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

u0hat = zeros(ComplexF64, kmax+1)
u0hat[2:end] = xsamp[1:kmax] .- im .* xsamp[kmax+1:end]

# ── Problem / method parameters ──────────────────────────────────────
C2    = 1/120
C3    = 1.0
tfin  = 1.0
order = 3

# ── Convergence sweep over h ──────────────────────────────────────────
h_ref = 1e-2
hs = [h_ref / 2^n for n in 0:6]

uh_EI_final   = Vector{Vector{ComplexF64}}(undef, length(hs))
uh_noEI_final = Vector{Vector{ComplexF64}}(undef, length(hs))

for (i, h) in enumerate(hs)
	params_EI   = KdVParams(kmax=kmax, C2=C2, C3=C3, tfin=tfin, order=order, dt_num=h, dt_save=h)
	params_noEI = KdV_noexpint_Params(kmax=kmax, C2=C2, C3=C3, tfin=tfin, order=order, dt_num=h, dt_save=h)

	_, uh_EI   = Taylor_KdV(u0hat, params_EI)
	_, uh_noEI = Taylor_KdV_noEI(u0hat, params_noEI)

	uh_EI_final[i]   = uh_EI[:, end]
	uh_noEI_final[i] = uh_noEI[:, end]
end

# ── L2 error proxy: E_2(h) = sqrt(2*compute_energy(w[2:end])), w = u(h) - u(h/2) ──
# hs[i+1] = hs[i]/2, so each h's half-step companion is already in the sweep.
hs_L2   = hs[1:end-1]
L2_EI   = zeros(length(hs_L2))
L2_noEI = zeros(length(hs_L2))

for i in 1:length(hs_L2)
	w_EI   = uh_EI_final[i]   .- uh_EI_final[i+1]
	w_noEI = uh_noEI_final[i] .- uh_noEI_final[i+1]

	L2_EI[i]   = sqrt(2 * compute_energy(w_EI[2:end]))
	L2_noEI[i] = sqrt(2 * compute_energy(w_noEI[2:end]))
end

# ── Plot: L2 error proxy, same format as the Hamiltonian convergence plot ───
i_ref = length(hs_L2)
scale3 = 3 * L2_noEI[i_ref] / hs_L2[i_ref]^3

L2_plot = plot(hs_L2, L2_EI, xscale=:log10, yscale=:log10, marker=:circle,
	xlabel="h", ylabel="E_2(h) = ||u(h) - u(h/2)||_{L^2}",
	title="Spectral L2 Error Proxy (order $order)", label="Exponential Integrator",
	left_margin=8mm)
plot!(L2_plot, hs_L2, L2_noEI, marker=:square, label="No Exponential Integrator")
plot!(L2_plot, hs_L2, scale3 .* hs_L2.^3, linestyle=:dash, color=:black, label="h^3")
display(L2_plot)
