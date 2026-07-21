cd(@__DIR__)
using Plots
using Plots.PlotMeasures
include("KdV_solver.jl")
include("Taylor_KdV_no_exp_int.jl")

include("gibbs_sample.jl")
using .GibbsSample

# ── Initial condition: draw one sample from the Gibbs measure ────────────────
kmax = 16
gibbs_params = GibbsParams(nmodes=kmax, E0=1, min_samps_accept=1, max_samps_accept=1)
xdata, accept_rate = gibbs_sample(gibbs_params)
xsamp = xdata[:, 1]

u0hat = zeros(ComplexF64, kmax+1)
u0hat[2:end] = xsamp[1:kmax] .- im .* xsamp[kmax+1:end]

# ── parameters ──────────────────────────────────────
C2    = 1/120
C3    = 1.0
tfin  = 1.0
order = 3




# ── Sweep over h, save only the final state at each resolution ────────
h_ref = 1e-2
hs = [h_ref / 2^n for n in 0:6]





# ── KdV_solver (exponential integrator) ────────────────────────────────

uh_final = Vector{Vector{ComplexF64}}(undef, length(hs))

for (i, h) in enumerate(hs)
	params = KdVParams(kmax=kmax, C2=C2, C3=C3, tfin=tfin, order=order, dt_num=h, dt_save=tfin)
	_, uh = Taylor_KdV(u0hat, params)
	uh_final[i] = uh[:, end]
end

# ── calculate the errors ────────────
hs_plot = hs[1:end-1]
E_h = zeros(length(hs_plot))

for i in 1:length(hs)-1
	w   = uh_final[i] .- uh_final[i+1]           # by construction of hs, hs[i+1] = hs[i]/2
	sum_w2 = sum(abs2.(w[2:end]))          # Σₖ₌₁^K |ŵ_k|²,  ŵ_k = û_k(t_fin; h) - û_k(t_fin; h/2)
	E_h[i] = sqrt(4*pi*sum_w2)                # derived formula:  E_h = sqrt(4π Σₖ₌₁^K |ŵ_k|²),   ŵ_k = û_k(t_fin; h) - û_k(t_fin; h/2)

end




# ── Taylor_KdV_no_exp_int (no exponential integrator) ──────────────────
uh_final_noEI = Vector{Vector{ComplexF64}}(undef, length(hs))

for (i, h) in enumerate(hs)
	params_noEI = KdV_noexpint_Params(kmax=kmax, C2=C2, C3=C3, tfin=tfin, order=order, dt_num=h, dt_save=tfin)
	_, uh = Taylor_KdV_noEI(u0hat, params_noEI)
	uh_final_noEI[i] = uh[:, end]
end

E_h_noEI = zeros(length(hs_plot))

for i in 1:length(hs_plot)
	w      = uh_final_noEI[i] .- uh_final_noEI[i+1]
	sum_w2 = sum(abs2.(w[2:end]))
	E_h_noEI[i] = sqrt(4*pi*sum_w2)
end







# ── Plot: both methods on the same axes, same format as the Hamiltonian plot ───
i_ref  = length(hs_plot)
scale3 = 3 * E_h_noEI[i_ref] / hs_plot[i_ref]^3

L2_plot = plot(hs_plot, E_h, xscale=:log10, yscale=:log10, marker=:circle,
	xlabel="h", ylabel="E_h",
	title="Spectral L2 Error Proxy (order $order)", label="Exponential Integrator",
	left_margin=8mm)
plot!(L2_plot, hs_plot, E_h_noEI, marker=:square, label="No Exponential Integrator")
plot!(L2_plot, hs_plot, scale3 .* hs_plot.^3, linestyle=:dash, color=:black, label="h^3")
display(L2_plot)





