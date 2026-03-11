using StatsPlots
using CSV
using DataFrames
using Measures
using LaTeXStrings
using ArgParse

stable = ArgParseSettings()
@add_arg_table stable begin
    "--mode", "-m"
    help = "Mode: dry, artifacts, paper"
    arg_type = String
    required = true
end
parsed_args = parse_args(stable)
mode = parsed_args["mode"]
if mode ∉ ["dry", "artifacts", "paper"]
    @error "Invalid -m value '$mode'. Must be one of: dry, artifacts, paper"
    exit(1)
end

try
    global df = CSV.read("/workspace/dp-hype/algorithms/simulation/results/$mode/sim_hgood.csv", DataFrame)
catch e
    @error "Result file not found. Please execute 'experiments/simulations.sh $mode' first." exception = e
    exit(1)
end

eps_levels = (0.25, 1.0)
ks = (1, 5, 25, 50, 100)

default(
    guidefont=font(12),
    tickfont=font(12),
    legendfont=font(12),
    titlefont=font(12),
    lw=2,
    markersize=6,
)

ks = sort(unique(df.k))
k_index = Dict(k => i for (i, k) in enumerate(ks))

base_markers = [:circle, :square, :utriangle, :diamond, :dtriangle,
    :hexagon, :star5, :cross, :xcross, :pentagon]
mk_list = [base_markers[mod1(i, length(base_markers))] for i in 1:length(ks)]
markers = Dict(k => mk_list[i] for (i, k) in enumerate(ks))

lay = @layout([a b])
p = plot(layout=lay, link=:y, size=(600, 185), left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=5mm,
    legend=false)

for (i, e) in enumerate(eps_levels)
    sdf = sort(df[df.eps.==e, :], [:k, :σ_user])

    for k in ks
        s = sdf[sdf.k.==k, :]
        @df s plot!(p, :h_abs, :top_k_acc,
            subplot=i,
            seriestype=:line,
            seriescolor=k_index[k],
            label=(i == 1 ? latexstring(k) : nothing),
            marker=markers[k])
    end

    plot!(p; subplot=i,
        xlabel="Percentage of " * L"H_{\mathrm{good}}",
        ylabel=i == 1 ? L"\mathrm{Accuracy}" : "",
        ylims=(0, 1.06),
        xscale=:log2,
        xticks=([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"]),
        yticks=([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"]),
        title="ε = $(e)")
end

savefig(p, "/workspace/dp-hype/figures/$mode/figure2b_simulation_hgood.pdf")

legend_plot = plot(legend=false, xaxis=false, yaxis=false, grid=false, framestyle=:none, size=(400, 100))

for k in ks
    plot!(legend_plot, [NaN], [NaN];
        marker=markers[k],
        seriescolor=k_index[k],
        label=latexstring(k))
end

plot!(legend_plot; legend=:bottom,
    legend_column=-1,
    legend_title=L"k")

savefig(legend_plot, "/workspace/dp-hype/figures/$mode/figure2_simulation_legend.pdf")