cd(@__DIR__)

using Plots, Statistics, Printf, JLD2

@load "results_small.jld2" max_amps_save H3_save H2_save t_save n_samps tfin K P

# paste scatter plot code here

# ── Find top 5 by max amplitude ───────────────────────────────────────────────
max_amps = [maximum(results[j][8]) for j in 1:n_samps]
top5 = sortperm(max_amps, rev=true)[1:5]

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



# ── Scatter plot: H3(0) vs Max Amplitude ─────────────────────────────────────
x = [H3_save[j][1] for j in 1:n_samps]
y = max_amps_save

x_mean = mean(x); y_mean = mean(y)
slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
intercept = y_mean - slope * x_mean

y_pred = slope .* x .+ intercept
SS_res = sum((y .- y_pred).^2)
SS_tot = sum((y .- y_mean).^2)
R2 = 1 - SS_res/SS_tot
println("R² = ", round(R2, digits=4))

x_line = range(minimum(x), maximum(x), length=100)
y_line = slope .* x_line .+ intercept

pS = scatter(x, y, xlabel="H3(0)", ylabel="Max Amplitude",
             title="H3(0) vs Max Wave Amplitude", label="Samples", color=:steelblue)
plot!(pS, x_line, y_line, label="slope=$(round(slope, digits=3)), R²=$(round(R2, digits=3))",
      color=:crimson, linewidth=2)
display(pS)

