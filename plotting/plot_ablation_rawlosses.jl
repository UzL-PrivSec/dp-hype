using StatsPlots
using CSV
using DataFrames
using Measures
using LaTeXStrings
using Statistics
using Distributions
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--mode", "-m"
    help = "Mode: dry, artifacts, paper"
    arg_type = String
    required = true
end
parsed_args = parse_args(s)
mode = parsed_args["mode"]
if mode ∉ ["dry", "artifacts", "paper"]
    @error "Invalid -m value '$mode'. Must be one of: dry, artifacts, paper"
    exit(1)
end


try
    global df_raw_cifar10_n100_a05 = CSV.read("/workspace/dp-hype/algorithms/dphype_rawlosses/dphype/results/$mode/dp-hype_rawlosses-uoft-cs-cifar10-dirichlet-0.5-N100-epsAll.csv", DataFrame)
    global df_raw_cifar10_n100_a50 = CSV.read("/workspace/dp-hype/algorithms/dphype_rawlosses/dphype/results/$mode/dp-hype_rawlosses-uoft-cs-cifar10-dirichlet-5.0-N100-epsAll.csv", DataFrame)
catch e
    @error "Result file not found. Please execute 'experiments/ablation_rawlosses.sh $mode <device>' first." exception = e
    exit(1)
end

try
    global df_dphype_cifar10_n100_a05 = CSV.read("/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode/dp-hype-uoft-cs-cifar10-dirichlet-0.5-N100-epsAll.csv", DataFrame)
    global df_dphype_cifar10_n100_a50 = CSV.read("/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode/dp-hype-uoft-cs-cifar10-dirichlet-5.0-N100-epsAll.csv", DataFrame)
catch e
    @warn "Result file not found, plotting without dp-hype topk data. Execute 'experiments/privutility_tradeoff.sh $mode <device> 100' to include it." exception = e
    global df_dphype_cifar10_n100_a05 = DataFrame(eps=[0.1, 0.25, 0.5, 1.0, 3.0, Inf], accuracy=Float64[NaN, NaN, NaN, NaN, NaN, NaN])
    global df_dphype_cifar10_n100_a50 = DataFrame(eps=[0.1, 0.25, 0.5, 1.0, 3.0, Inf], accuracy=Float64[NaN, NaN, NaN, NaN, NaN, NaN])
end


default(
    guidefont=font(12),
    tickfont=font(12),
    legendfont=font(12),
    titlefont=font(12),
    lw=2,
    markersize=6,
    markerstrokewidth=10,
)

markers = [:circle, :square, :utriangle, :diamond, :dtriangle,
    :hexagon, :star5, :cross, :xcross, :pentagon]

eps_inf_sub = 10.0

xticks = ([0.1, 0.25, 0.5, 1.0, 3.0, eps_inf_sub], ["0.1", "0.25", "0.5", "1.0", "3.0", "ꝏ"])

p = plot(
    size=(600, 175),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=7mm,
    legend=:outerright,
)

ylims!(p, (0.1, 0.52))

data_plots = [df_raw_cifar10_n100_a50, df_raw_cifar10_n100_a05, df_dphype_cifar10_n100_a50, df_dphype_cifar10_n100_a05]

for (i, df_a) in enumerate(data_plots)
    sort!(df_a, :eps, by=x -> x == "inf" ? Inf : x)
    df_a = transform(df_a, :eps => ByRow(x -> isfinite(x) ? x : eps_inf_sub) => :eps_plot)

    stats = combine(groupby(df_a, :eps_plot)) do g
        μ = mean(g.accuracy)
        σ = std(g.accuracy; corrected=false)
        N = nrow(g)
        z = quantile(TDist(100 - 1), 0.975)   # 95% CI
        δ = z * σ / sqrt(100)
        (; eps_plot=g.eps_plot[1], mean=μ, lower=μ - δ, upper=μ + δ)
    end
    sort!(stats, :eps_plot)

    is_beta5 = (i == 1 || i == 3)
    is_binary = (i > 2)

    plot!(p, stats.eps_plot, stats.mean;
        ribbon=(stats.mean .- stats.lower, stats.upper .- stats.mean),
        label="",
        xscale=:log2,
        xticks=xticks,
        yticks=[0.1, 0.3, 0.5],
        marker=markers[(is_beta5 ? 1 : 2)],
        color=(is_binary ? 2 : 1),
        linestyle=(is_binary ? :solid : :dot),
        xlabel="Privacy Budget ε",
        ylabel="Accuracy",
    )

end


legend_specs = [
    ("Binary, β=5.0", markers[1], 2, :solid),
    ("Raw Loss, β=5.0", markers[1], 1, :dot),
    ("Binary, β=0.5", markers[2], 2, :solid),
    ("Raw Loss, β=0.5", markers[2], 1, :dot)
]

for (lbl, mkr, clr, ls) in legend_specs
    plot!(p, [NaN], [NaN]; label=lbl, marker=mkr, color=clr, linestyle=ls)
end

savefig("/workspace/dp-hype/figures/$mode/figure9_ablation_rawlosses.pdf")

