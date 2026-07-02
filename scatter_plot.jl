cd(@__DIR__)
using Plots, Statistics, Printf, JLD2

@load "results_small.jld2" max_amps_save H3_save H2_save U_phys_top5 t_top5 maxu_timeseries top5 n_samps tfin K P

y = max_amps_save
y_mean = mean(y)

# ── Scatter plot: H3(0) vs Max Amplitude ─────────────────────────────────────
x = [H3_save[j][1] for j in 1:n_samps]
x_mean = mean(x)
slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
intercept = y_mean - slope * x_mean
y_pred = slope .* x .+ intercept
SS_tot = sum((y .- y_mean).^2)
R2 = 1 - sum((y .- y_pred).^2) / SS_tot
println("H3 R² = ", round(R2, digits=4))

x_line = range(minimum(x), maximum(x), length=100)
pS = scatter(x, y, xlabel=L"H_3(0)", ylabel=L"Max Amplitude",
             title="H₃(0) vs Max Wave Amplitude (n=$n_samps)", label="Samples", color=:steelblue,
             left_margin=5mm)
plot!(pS, x_line, slope .* x_line .+ intercept,
      label=L"slope=%$(round(slope, digits=3)),\ R^2=%$(round(R2, digits=3))",
      color=:crimson, linewidth=2)
display(pS)

# ── Scatter plot: H2(0) vs Max Amplitude ─────────────────────────────────────
x2 = [H2_save[j][1] for j in 1:n_samps]
x2_mean = mean(x2)
slope2 = sum((x2 .- x2_mean) .* (y .- y_mean)) / sum((x2 .- x2_mean).^2)
intercept2 = y_mean - slope2 * x2_mean
y_pred2 = slope2 .* x2 .+ intercept2
R2_2 = 1 - sum((y .- y_pred2).^2) / SS_tot
println("H2 R² = ", round(R2_2, digits=4))

x2_line = range(minimum(x2), maximum(x2), length=100)
pS2 = scatter(x2, y, xlabel=L"H_2(0)", ylabel=L"Max Amplitude",
              title="H₂(0) vs Max Wave Amplitude (n=$n_samps)", label="Samples", color=:steelblue,
              left_margin=5mm)
plot!(pS2, x2_line, slope2 .* x2_line .+ intercept2,
      label=L"slope=%$(round(slope2, digits=3)),\ R^2=%$(round(R2_2, digits=3))",
      color=:crimson, linewidth=2)
display(pS2)





# ── Top 5: max(u) vs time, H3 vs time, H2 vs time ───────────────────────────
top_plots = []
for i in 1:5
    local H3_T, H2_T, tplot_T, tplot_U
    U_phys_T = U_phys_top5[i]  # already a vector of max(u) per timestep
    H3_T     = H3_save[top5[i]]
    H2_T     = H2_save[top5[i]]
    tplot_T  = range(0, tfin, length=length(H3_T))
    tplot_U  = range(0, tfin, length=length(U_phys_T))

    pA = plot(tplot_U, U_phys_T,
              xlabel="t", ylabel="max(u)",
              title="Sample $(top5[i]) (amp=$(round(max_amps_save[top5[i]], digits=3)))",
              label=false, color=:steelblue)

    pH3 = plot(tplot_T, H3_T, xlabel="t", ylabel="H3",
               title="H3 - Sample $(top5[i])", label=false, color=:crimson)

    pH2 = plot(tplot_T, H2_T, xlabel="t", ylabel="H2",
               title="H2 - Sample $(top5[i])", label=false, color=:green)

    push!(top_plots, pA, pH3, pH2)
end

top_comparison = plot(top_plots..., layout=(5, 3), size=(1200, 1200))
display(top_comparison)



# all_displacements = vcat(maxu_timeseries...)  #maxu_timeseries is max(u) at each timestep for every sample
                                                #vcat concats all 100 vectors into one long vector
# sigma = std(all_displacements)
# println("σ = ", sigma)
# println("4σ threshold = ", 4*sigma)


using Peaks  # or we can do it manually

# Manual local maxima detection
function find_peaks(v)
    peaks = Float64[]
    for i in 2:length(v)-1
        if v[i] > v[i-1] && v[i] > v[i+1]
            push!(peaks, v[i])
        end
    end
    return peaks
end

all_peaks = vcat([find_peaks(maxu_timeseries[j]) for j in 1:n_samps]...)
sigma = std(all_peaks)
mu = mean(all_peaks)
threshold = mu + 3.5*sigma
println("μ = ", round(mu, digits=4))
println("σ = ", round(sigma, digits=4))
println("3.5σ threshold = ", round(threshold, digits=4))
n_large = sum(all_peaks .>= threshold)
println("Large wave events: $n_large out of $(length(all_peaks)) peaks")
