using Plots, FFTW, LinearAlgebra, Statistics, Printf
using Parameters, Distributions, Roots, Optim, Random, Test, Distributed, SharedArrays
using JLD2, DelimitedFiles

include("trap_periodic.jl")
include("dealias_product_direct.jl")
include("taylor_KdV_solver.jl")
include("Exp_Integrator_RK2_Dealiasing_H23.jl")
include("compute_h2.jl")
include("compute_h3.jl")
include("main.jl")

# ── Sample initial condition from main.jl ─────────────────────────────────────
params = ParamSet(
    min_samps_accept = 100,
    max_samps_accept = 200,
)

xdata, accept_rate = main(params)
println("Samples accepted: ", size(xdata, 2))

if size(xdata, 2) == 0
    error("No samples accepted — check rej_const in max_fg")
end

# ── Parameters ────────────────────────────────────────────────────────────────
K    = params.nmodes
C2   = 1/120
C3   = 1
a    = 0
tfin = 5
h    = C2^-1 .* K^-3 .* 0.001
P    = 3

kpos = 0:K

# ── Reconstruct u0 from first sample ─────────────────────────────────────────
n_samps = size(xdata, 2)

# ── Run Taylor solver for all samples ────────────────────────────────────────
results = Vector{Any}(undef, n_samps)

for j in 1:n_samps
    local samp, real_parts, imag_parts, u0, E0_samp
    local t_T, uk_T, Energy_T, M_T, H_T, H2_T, H3_T, U_phys_T

    samp = xdata[:, j]
    real_parts = samp[1:K]
    imag_parts = samp[K+1:end]
    u0 = zeros(ComplexF64, K+1)
    u0[1] = 0.0
    u0[2:K+1] = real_parts .+ im .* imag_parts

    E0_samp = 2π * sum(abs2.(u0[2:end]))
    u0 ./= sqrt(E0_samp / params.E0)

    t_T, uk_T, Energy_T, M_T, H_T, H2_T, H3_T, U_phys_T = Taylor_KdV(C2, C3, K, a, u0, h, tfin, P)
    results[j] = (t_T, uk_T, Energy_T, M_T, H_T, H2_T, H3_T, U_phys_T)
    println("Sample $j max wave amplitude: ", maximum(U_phys_T))
end

max_amps_save = [maximum(results[j][8]) for j in 1:n_samps]
H3_save = [results[j][7] for j in 1:n_samps]
H2_save = [results[j][6] for j in 1:n_samps]
t_save  = [collect(results[j][1]) for j in 1:n_samps]

@save "results_small.jld2" max_amps_save H3_save H2_save t_save n_samps tfin K P




# # ── Conserved quantities plots ────────────────────────────────────────────────
# tplot_T = collect(t_T)[1:length(Energy_T)]
# # tplot_R = collect(t_R)[1:length(Energy_R)]

# pE = plot(tplot_T, Energy_T, label="Taylor P=$P", xlabel="t", ylabel="E", title="Energy")
# # plot!(pE, tplot_R, Energy_R, label="RK2", linestyle=:dash)
# plot!(pE, ylims=(0.9999, 1.0001))

# pM = plot(tplot_T, M_T, label="Taylor P=$P", xlabel="t", ylabel="M", title="Momentum")
# # plot!(pM, tplot_R, M_R, label="RK2", linestyle=:dash)

# pH = plot(tplot_T, H_T, label="Taylor P=$P", xlabel="t", ylabel="H", title="Hamiltonian")
# # plot!(pH, tplot_R, H_R, label="RK2", linestyle=:dash)
# plot!(pH, ylims=(H_T[1] - 1e-5, H_T[1] + 1e-5))

# comparison = plot(pE, pM, pH, layout=(3,1), size=(700, 700))
# display(comparison)

# # ── Hamiltonian convergence test ──────────────────────────────────────────────
# h_ref  = C2^-1 .* K^-3 .* 0.01
# hs     = [8*h_ref, 4*h_ref, 2*h_ref, sqrt(2)*h_ref, h_ref,
#           h_ref/sqrt(2), h_ref/2, h_ref/4, h_ref/8]

# errors_T = zeros(length(hs))
# # errors_R = zeros(length(hs))

# for i in 1:length(hs)
#     _, _, _, _, H_i_T, _, _, _ = Taylor_KdV(C2, C3, K, a, u0, hs[i], tfin, P)
#     # _, _, _, _, H_i_R, _, _, _ = Exp_Integrator_RK2_Dealiasing_H23(C2, C3, K, a, u0, hs[i], tfin)
#     errors_T[i] = abs(H_i_T[end] - H_i_T[1]) / abs(H_i_T[1])
#     # errors_R[i] = abs(H_i_R[end] - H_i_R[1]) / abs(H_i_R[1])
# end

# convergence_T = plot(hs, errors_T,
#                      xscale=:log10, yscale=:log10,
#                      marker=:circle, color=:steelblue,
#                      xlabel="h", ylabel="|H(t_fin) - H(0)| / |H(0)|",
#                      title="Hamiltonian Error: Taylor P=$P vs RK2",
#                      label="Taylor P=$P")

# # plot!(convergence_T, hs, errors_R,
# #       marker=:circle, color=:crimson,
# #       linestyle=:solid, label="RK2")

# ref_colors = [:gray, :orange, :green]
# for (q, col) in zip(2:P+1, ref_colors)
#     scale = errors_T[end] / hs[end]^q
#     plot!(convergence_T, hs, scale .* hs.^q,
#           linestyle=:dash, color=col, label="h^$q")
# end

# display(convergence_T)






# tplot_U = range(0, tfin, length=size(U_phys_T, 2))
# pU = plot(tplot_U, vec(maximum(U_phys_T, dims=1)), label="Taylor P=$P", xlabel="t", ylabel="max(u)", title="Max Physical Solution")
# display(pU)




# println("\n── Conserved Quantities ─────────────────────────────────────────")
# println("Sample │ Max Amplitude  │ H3(0)          │ H3(end)")
# println("───────┼────────────────┼────────────────┼────────────────")
# for j in 1:n_samps
#     local U_phys_T, H3_T
#     U_phys_T = results[j][8]
#     H3_T     = results[j][7]
#     @printf("  %3d  │  %.6f      │  %.6f      │  %.6f\n",
#             j, maximum(U_phys_T), H3_T[1], H3_T[end])
# end
# println("─────────────────────────────────────────────────────────────────")





# # ── Find top 5 by max amplitude ───────────────────────────────────────────────
# max_amps = [maximum(results[j][8]) for j in 1:n_samps]
# top5 = sortperm(max_amps, rev=true)[1:5]

# # ── Plot top 5: max(u) vs time and H3 vs time ────────────────────────────────
# top_plots = []
# for j in top5
#     local t_T, Energy_T, H3_T, U_phys_T
#     t_T, _, Energy_T, _, _, _, H3_T, U_phys_T = results[j]
#     tplot_T = collect(t_T)[1:length(Energy_T)]
#     tplot_U = range(0, tfin, length=size(U_phys_T, 2))

#     pA = plot(tplot_U, vec(maximum(U_phys_T, dims=1)),
#               xlabel="t", ylabel="max(u)",
#               title="Sample $j (amp=$(round(max_amps[j], digits=3)))",
#               label=false, color=:steelblue)

#     pH3 = plot(tplot_T, H3_T,
#                xlabel="t", ylabel="H3",
#                title="H3 - Sample $j",
#                label=false, color=:crimson)

#     push!(top_plots, pA, pH3)
# end

# top_comparison = plot(top_plots..., layout=(5, 2), size=(900, 1200))
# display(top_comparison)



# # ── Scatter plot: H3(0) vs Max Amplitude ─────────────────────────────────────
# max_amps = [maximum(results[j][8]) for j in 1:n_samps]
# H3_init  = [results[j][7][1] for j in 1:n_samps]

# x = H3_init
# y = max_amps

# x_mean = mean(x); y_mean = mean(y)
# slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
# intercept = y_mean - slope * x_mean

# y_pred = slope .* x .+ intercept
# SS_res = sum((y .- y_pred).^2)
# SS_tot = sum((y .- y_mean).^2)
# R2 = 1 - SS_res/SS_tot
# println("R² = ", round(R2, digits=4))

# x_line = range(minimum(x), maximum(x), length=100)
# y_line = slope .* x_line .+ intercept

# pS = scatter(x, y, xlabel="H3(0)", ylabel="Max Amplitude",
#              title="H3(0) vs Max Wave Amplitude", label="Samples", color=:steelblue)
# plot!(pS, x_line, y_line, label="slope=$(round(slope, digits=3)), R²=$(round(R2, digits=3))",
#       color=:crimson, linewidth=2)
# display(pS)
